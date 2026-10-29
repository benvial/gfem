"""Designs shared by the eigenmode tests (tests/unit).

The eigenmode unit tests (tests/unit/test_eigenmode.py,
test_eigenmode_nonlinear.py) and the SLEPc seed tests that compare against
them (test_eigenmode_slepc.py) build their models here, so every eigen
solver runs on the same designs. Every builder that writes a mesh takes the
directory to write it to (a pytest ``tmp_path_factory.mktemp`` directory).

- `line_mesh` / `line`: a parallel-plate line along x, PEC plates at z = 0
  and z = h, PMC side walls and open ends; its TEM modes resonate at the
  half-wave frequencies ``m F1``.
- `terminated_line`, `line_roots`: the line with a resistor on each end, and
  the transmission-line roots of its complex frequencies.
- `conductor_line`: the line with a `gfem.Conductor` top plate (a
  nonpolynomial operator).
"""

import cmath
import math
from pathlib import Path

import numpy as np
import scipy.constants as const
import structured_meshes as sm

import gfem

L, W, H = 10.0, 2.0, 1.0  # mm
EPS = 4.0
ETA0 = const.mu_0 * const.c
Z0 = ETA0 * H / (W * math.sqrt(EPS))
F1 = const.c / (2 * L * 1e-3 * math.sqrt(EPS))  # 7.49 GHz
SIGMA = 1e5  # a poor conductor: Q of about 100


def line_mesh(directory: Path, cells: tuple[int, int, int] = (20, 1, 1)) -> gfem.Mesh:
    """The line's mesh, ``cells`` hexahedra along (x, y, z), in mm."""
    path = sm.write_msh22(Path(directory) / "line.msh", sm.strip_line(L, W, H, cells))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def line(mesh, ports=(), **material) -> gfem.Model:
    """The line, ``material`` keywords added to its dielectric."""
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS, **material)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=list(ports),
    )


def terminated_line(mesh, r) -> gfem.Model:
    """The line with a resistor ``r`` (ohm) on each end."""
    return line(
        mesh,
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=r),
            gfem.LumpedPort("p2", "port2", "+z", impedance=r),
        ],
    )


def line_roots(r, modes) -> np.ndarray:
    """Complex angular frequencies of the line with R on both ends."""
    gamma = (r - Z0) / (r + Z0)
    out = []
    for m in modes:
        # exp(-2 i beta L) = Gamma^-2  =>  beta = (2 pi m - 2 i log Gamma) / (2 L)
        beta = (2 * math.pi * m - 2j * cmath.log(gamma)) / (2 * L * 1e-3)
        out.append(beta * const.c / math.sqrt(EPS))
    return np.array(out)


def conductor_line(mesh, sigma=SIGMA) -> gfem.Model:
    """The line with a conductor of conductivity ``sigma`` as its top plate."""
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.Conductor("top", sigma)],
    )
