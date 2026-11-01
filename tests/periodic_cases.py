"""Periodic cells shared by the periodic tests, with their closed forms.

The cells C1-C4 of the periodic verification cases, built on
`structured_meshes.periodic_cell` (opposite faces named "x_min" / "x_max",
... and matching under translation). Every builder that writes a mesh takes
the directory to write it to; nothing here solves, so the tolerances can be
read without NGSolve. Meshes are in mm, wave vectors in rad/m.

- C1, `c1`: a parallel-plate TEM line periodic along x (PEC plates at z = 0
  and H, PMC y walls) with two 50 ohm lumped ports on interior sheets; its
  open-circuit impedance matrix is `bloch_line_z`.
- C2, `c2`: an empty parallel-plate cell periodic along x and y; its bands
  are the empty lattice, `empty_lattice`.
- C3, `c3`: a layered (Bragg) TEM line periodic along x; its bands are the
  transfer-matrix roots, `tmm_frequency`.
- C4, `c4`: an empty vacuum cube periodic along x, y and z (no other
  boundary); its lowest band is a doublet at `empty_lattice`.

C1 loaded by its two 50 ohm ports is also a damped Bloch eigen case: its
complex resonances are the roots of det(Z(f, k) + R) = 0,
`loaded_line_frequency`. The eigen gradient cell, `eigen_cell`, is a
damped, lossy parallel-plate cell periodic along x and y, with a resistive
sheet crossing the y seam.

The tolerances are about 10-50x the error measured on each cell at its
order and mesh.
"""

import math
from pathlib import Path

import numpy as np
import scipy.constants as const
import scipy.optimize
import structured_meshes as sm

import gfem

MM = 1e-3
C0 = const.c
ETA0 = const.mu_0 * const.c

#: |S - S_exact| on C1.
C1_S_TOL = 1e-5
#: Relative error of the 4 lowest C2 bands (real part).
C2_BAND_RTOL = 3e-4
#: |E_z(p2) / E_z(p1) - exp(-i k·Δr)| on the C2 band with G = 0.
C2_PHASE_TOL = 1e-2
#: Relative error of a complex C3 frequency.
C3_FREQ_RTOL = 2e-6
#: Relative error of the lowest C4 doublet.
C4_RTOL = 1e-4
#: Relative error of the complex resonance of C1 loaded by its ports
#: (6.6e-6 measured).
C1_EIGEN_RTOL = 1e-4


def _mesh(directory: Path, name: str, cell: tuple) -> gfem.Mesh:
    path = sm.write_msh22(Path(directory) / f"{name}.msh", cell)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _edges(size, cells):
    return tuple(np.linspace(0.0, s, n + 1) for s, n in zip(size, cells, strict=True))


# --- C1: Bloch line two-port ------------------------------------------------------

C1_A, C1_W, C1_H = 10.0, 2.0, 1.0  # mm
C1_X1, C1_X2 = 2.5, 6.25  # port sheets, mm
C1_EPS = 4.0
C1_TAN_DELTA = 0.02
C1_FREQUENCY = 5e9
C1_ORDER = 2
#: The Bloch wave vector of C1, rad/m (also solved at -k and 0).
C1_K = (0.3 * math.pi / (C1_A * MM), 0.0, 0.0)


def c1_mesh(directory: Path, cells: tuple[int, int, int] = (16, 1, 1)) -> gfem.Mesh:
    """C1's mesh; the sheets "sheet1" (x = X1) and "sheet2" (x = X2) are grid planes."""
    cell = sm.periodic_cell(
        _edges((C1_A, C1_W, C1_H), cells),
        sheets=(("sheet1", C1_X1), ("sheet2", C1_X2)),
    )
    return _mesh(directory, "c1", cell)


def c1(mesh: gfem.Mesh) -> gfem.Model:
    """C1: periodic in x, PEC at z = 0 and H, default PMC y walls, two ports."""
    return gfem.Model(
        mesh,
        materials={"cell": gfem.Material(eps_r=C1_EPS, tan_delta=C1_TAN_DELTA)},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
        ports=[
            gfem.LumpedPort("p1", "sheet1", "+z", impedance=50.0),
            gfem.LumpedPort("p2", "sheet2", "+z", impedance=50.0),
        ],
    )


def bloch_line_z(frequency: complex, k: float, eps: complex) -> np.ndarray:
    """C1's open-circuit impedance matrix (ohm) at Bloch wave number k (rad/m).

    Two shunt current sheets at X1 and X2 on a TEM line of period a, with
    E(x + a) = exp(-i k a) E(x):

        Z11 = Z22 = (Z0 / 2) [1 / (1 - q2) + q1 / (1 - q1)]
        Z21(k) = (Z0 / 2) [exp(-i b d) / (1 - q2) + exp(+i b d) q1 / (1 - q1)]
        Z12(k) = Z21(-k)

    with q1 = exp(-i (b + k) a), q2 = exp(-i (b - k) a), d = X2 - X1 and
    b = sqrt(eps) k0. ``eps`` is complex for a lossy line, ``frequency``
    for a free oscillation.
    """
    a, d = C1_A * MM, (C1_X2 - C1_X1) * MM
    k0 = 2 * np.pi * frequency / C0
    b = np.sqrt(complex(eps)) * k0
    z0 = ETA0 * C1_H / (C1_W * np.sqrt(complex(eps)))

    def q(kk):
        return np.exp(-1j * (b + kk) * a), np.exp(-1j * (b - kk) * a)

    def z21(kk):
        q1, q2 = q(kk)
        return (
            z0
            / 2
            * (np.exp(-1j * b * d) / (1 - q2) + np.exp(1j * b * d) * q1 / (1 - q1))
        )

    q1, q2 = q(k)
    z11 = z0 / 2 * (1 / (1 - q2) + q1 / (1 - q1))
    return np.array([[z11, z21(-k)], [z21(k), z11]])


#: The C1 eigen target, Hz: the loaded line's resonance near 14.18 + 2.51i GHz
#: (Q 2.8) at `C1_K`.
C1_EIGEN_TARGET = 14e9


def loaded_line_frequency(k: float, guess: complex) -> complex:
    """C1's complex resonance (Hz) nearest ``guess`` with both ports loaded.

    The root of det(Z(f, k) + R) = 0 (`bloch_line_z`, R = 50 ohm on each
    port): the port currents of a free oscillation see their loads.
    """
    eps = C1_EPS * (1 - 1j * C1_TAN_DELTA)

    def residual(fz):
        z = bloch_line_z(complex(fz[0], fz[1]), k, eps) + 50.0 * np.eye(2)
        d = np.linalg.det(z)
        return [d.real, d.imag]

    re, im = scipy.optimize.fsolve(residual, [guess.real, guess.imag], xtol=1e-14)
    return complex(re, im)


# --- C2: empty parallel-plate cell, two pairs --------------------------------------

C2_AX, C2_AY, C2_H = 10.0, 7.0, 1.0  # mm
C2_EPS = 2.0
C2_ORDER = 2
#: The Bloch wave vector of C2, rad/m.
C2_K = (0.4 * math.pi / (C2_AX * MM), -0.25 * math.pi / (C2_AY * MM), 0.0)
#: The two points (m) of C2's field sign pin.
C2_PROBES = (
    (0.31 * C2_AX * MM, 0.42 * C2_AY * MM, 0.5 * C2_H * MM),
    (0.77 * C2_AX * MM, 0.13 * C2_AY * MM, 0.5 * C2_H * MM),
)


def c2_mesh(directory: Path, cells: tuple[int, int, int] = (8, 8, 1)) -> gfem.Mesh:
    """C2's mesh."""
    return _mesh(directory, "c2", sm.periodic_cell(_edges((C2_AX, C2_AY, C2_H), cells)))


def c2(mesh: gfem.Mesh) -> gfem.Model:
    """C2: periodic in x and y (the pairs share the z edges), PEC at z = 0 and h."""
    return gfem.Model(
        mesh,
        materials={"cell": gfem.Material(eps_r=C2_EPS)},
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
    )


def empty_lattice(
    k: tuple[float, float, float],
    periods: tuple[float | None, float | None, float | None],
    eps: float = 1.0,
    n: int = 4,
) -> np.ndarray:
    """The ``n`` lowest empty-lattice frequencies c|k + G| / (2 pi sqrt(eps)), Hz.

    ``periods`` (m) per axis, ``None`` for an axis without a pair (G = 0
    there). One value per G: no polarization multiplicity (C4's lowest
    value is a doublet).
    """
    axes = [range(-3, 4) if p else range(1) for p in periods]
    kv = np.asarray(k, dtype=float)
    out = []
    for idx in np.ndindex(*(len(r) for r in axes)):
        g = [
            2 * math.pi * r[i] / p if p else 0.0
            for r, i, p in zip(axes, idx, periods, strict=True)
        ]
        out.append(C0 * np.linalg.norm(kv + g) / (2 * math.pi * math.sqrt(eps)))
    return np.sort(out)[:n]


# --- C3: Bragg line -----------------------------------------------------------------

C3_A, C3_W, C3_H = 10.0, 2.0, 1.0  # mm
C3_D1 = 4.0  # thickness of the low-index layer, mm
C3_EPS_LOW, C3_EPS_HIGH = 2.0, 9.0
C3_TAN_DELTA = 0.01  # the lossy variant
C3_ORDER = 3
#: The plain (Gamma) target, away from the zero-frequency TEM field, Hz.
C3_PLAIN_TARGET = 11e9


def c3_mesh(directory: Path, cells: tuple[int, int, int] = (10, 1, 1)) -> gfem.Mesh:
    """C3's mesh: "low" for x < D1, "high" beyond (D1 must be a grid plane)."""
    cell = sm.periodic_cell(
        _edges((C3_A, C3_W, C3_H), cells),
        lambda c: "low" if c[0] < C3_D1 else "high",
    )
    return _mesh(directory, "c3", cell)


def c3(mesh: gfem.Mesh, tan_delta: float = 0.0) -> gfem.Model:
    """C3: periodic in x, PEC at z = 0 and H, default PMC y walls."""
    return gfem.Model(
        mesh,
        materials={
            "low": gfem.Material(eps_r=C3_EPS_LOW, tan_delta=tan_delta),
            "high": gfem.Material(eps_r=C3_EPS_HIGH, tan_delta=tan_delta),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.PEC("z_min"),
            gfem.PEC("z_max"),
        ],
    )


def tmm_frequency(k: float, guess: complex, tan_delta: float = 0.0) -> complex:
    """The C3 band frequency (Hz, complex when lossy) nearest ``guess`` at k (rad/m).

    The root of the transfer-matrix (Rytov) dispersion relation of the two
    layers, cos(n1 k0 d1) cos(n2 k0 d2) - (n1/n2 + n2/n1)/2 sin(n1 k0 d1)
    sin(n2 k0 d2) = cos(k a), with n = sqrt(eps (1 - i tan_delta)).
    """
    a, d1 = C3_A * MM, C3_D1 * MM
    d2 = a - d1
    n1 = np.sqrt(C3_EPS_LOW * (1 - 1j * tan_delta))
    n2 = np.sqrt(C3_EPS_HIGH * (1 - 1j * tan_delta))

    def residual(fz):
        k0 = 2 * math.pi * (fz[0] + 1j * fz[1]) / C0
        r = (
            np.cos(n1 * k0 * d1) * np.cos(n2 * k0 * d2)
            - 0.5 * (n1 / n2 + n2 / n1) * np.sin(n1 * k0 * d1) * np.sin(n2 * k0 * d2)
            - np.cos(k * a)
        )
        return [r.real, r.imag]

    re, im = scipy.optimize.fsolve(residual, [guess.real, guess.imag], xtol=1e-14)
    return complex(re, im)


# --- C4: empty cube, three pairs ------------------------------------------------------

C4_L = 10.0  # mm
C4_ORDER = 2
#: The Bloch wave vector of C4, rad/m.
C4_K = tuple(v * math.pi / (C4_L * MM) for v in (0.3, -0.2, 0.15))


def c4_mesh(directory: Path, cells: tuple[int, int, int] = (3, 3, 3)) -> gfem.Mesh:
    """C4's mesh."""
    return _mesh(directory, "c4", sm.periodic_cell(_edges((C4_L,) * 3, cells)))


def c4(mesh: gfem.Mesh) -> gfem.Model:
    """C4: vacuum, periodic in x, y and z, no other boundary."""
    return gfem.Model(
        mesh,
        materials={"cell": gfem.Material()},
        boundaries=[gfem.Periodic((f"{a}_min", f"{a}_max")) for a in "xyz"],
    )


# --- the eigen gradient cell ---------------------------------------------------------

EG_A, EG_B, EG_H = 10.0, 3.0, 1.0  # mm
EG_SHEET = 5.0  # the resistive sheet's plane x, mm
EG_EPS = (3.5, 4.5)  # the permittivity ramp over the tetrahedra
EG_SIGMA = (0.02, 0.06)  # the conductivity ramp, S/m
EG_SHEET_R = 2000.0  # ohm per square
EG_TAN_DELTA = 0.02  # the linear lossy variant
#: The Bloch wave vector of the eigen gradient cell, rad/m.
EG_K = (0.4 * math.pi / (EG_A * MM), 0.3 * math.pi / (EG_B * MM), 0.0)
#: Targets: the Bloch cell's lowest band (about 8 GHz) and the plain
#: cell's split G = ±2π/a pair (about 15 GHz).
EG_BLOCH_TARGET = 7e9
EG_PLAIN_TARGET = 14e9


def eigen_cell_mesh(directory: Path, cells: tuple[int, int, int] = (8, 2, 2)):
    """The eigen gradient cell's mesh: the sheet "sheet" at x = EG_SHEET."""
    cell = sm.periodic_cell(
        _edges((EG_A, EG_B, EG_H), cells), sheets=(("sheet", EG_SHEET),)
    )
    return _mesh(directory, "eigen_cell", cell)


def eigen_cell(
    mesh: gfem.Mesh, eps, sigma=None, fill=None, *, tan_delta: float | None = None
) -> gfem.Model:
    """The eigen gradient cell: periodic in x and y, PEC at z = 0 and H.

    Per-element ``eps`` (and ``sigma``, S/m, a damping), and with ``fill`` a
    resistive `gfem.Impedance` sheet (`EG_SHEET_R` per square, per-face
    fill) on "sheet", which touches the y pair along its edges. Without
    ``sigma`` and ``fill``, a ``tan_delta`` makes the cell lossy but
    undamped (a linear problem).
    """
    boundaries = [
        gfem.Periodic(("x_min", "x_max")),
        gfem.Periodic(("y_min", "y_max")),
        gfem.PEC("z_min"),
        gfem.PEC("z_max"),
    ]
    if fill is not None:
        boundaries.append(
            gfem.Impedance(
                "sheet", resistance=EG_SHEET_R, fill=gfem.SheetField(fill=fill)
            )
        )
    material = gfem.MaterialField(eps_r=eps, conductivity=sigma, tan_delta=tan_delta)
    return gfem.Model(mesh, materials={"cell": material}, boundaries=boundaries)


def eigen_cell_design(mesh: gfem.Mesh, seed: int = 0) -> dict:
    """``eps``, ``sigma`` and ``fill`` of the eigen gradient cell."""
    rng = np.random.default_rng(seed)
    n = len(mesh.elements("cell"))
    faces = len(mesh.elements("sheet"))
    return {
        "eps": np.linspace(*EG_EPS, n),
        "sigma": np.linspace(*EG_SIGMA, n),
        "fill": rng.uniform(0.5, 1.0, faces),
    }


def tie_classes(mesh: gfem.Mesh, periods: tuple[float | None, ...]) -> np.ndarray:
    """A class label per node: nodes equal modulo the ``periods`` (m) share one.

    Twins of every pair (and the corners of several) move together in a
    periodic node direction.
    """
    nodes = np.asarray(mesh.imported_nodes)
    keys = nodes.copy()
    for axis, period in enumerate(periods):
        if period:
            keys[:, axis] = np.mod(keys[:, axis], period)
            keys[np.isclose(keys[:, axis], period), axis] = 0.0
    scale = 1e-6 * float(np.ptp(nodes, axis=0).max())
    _, labels = np.unique(np.round(keys / scale), axis=0, return_inverse=True)
    return labels.reshape(-1)
