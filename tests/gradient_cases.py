"""Designs shared by the gradient tests (tests/verification).

The gradient verification matrix (tests/verification/test_gradient_matrix.py)
and the per-topic gradient tests around it build their models here, so a
matrix cell and the topic tests next to it differentiate the same design.
Every builder that writes a mesh takes the directory to write it to (a
pytest ``tmp_path_factory.mktemp`` directory).

- `layered_line`: a two-layer parallel-plate line, lumped ports, second-order
  absorbing side walls (driven and adaptive material gradients).
- `stack`, `capacitor`: two-layer capacitors (electrostatic material and
  shape gradients).
- `shape_line`: the two-layer line with a conductor top plate and a lossy,
  conducting lower layer (driven shape gradients).
- `wave_mesh` / `wave_line`: a two-layer line between wave ports with a
  design "core" away from the port faces.
- `sheet_mesh` / `sheet_line`: a parallel-plate line glued at mid-height,
  the interior surface "sheet" carrying a `gfem.SheetField`.
- `sheet_cavity` / `cavity_model`: a PEC cavity with a resistive sheet where
  TE101 is largest (eigenmode fill gradients).
- `resonator_mesh` / `terminated`, `layered_resonator`: line resonators
  terminated in resistors (eigenmode material and shape gradients).
- `mirrored_cavity`: a square-based PEC box whose mesh has the square's
  symmetry (an exactly degenerate pair).
"""

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import scipy.constants as const
import structured_meshes as sm

import gfem

#: The two layers of the line designs (name, thickness in mm).
LAYERS = (("sub", 0.5), ("cover", 0.5))
#: Weights of |S|^2 in the S objectives (not symmetric on purpose).
WEIGHTS = np.array([[1.0, 0.4], [-0.6, 1.5]])
#: Weights the real and the imaginary part of omega (GHz): frequency and loss.
OMEGA_WEIGHT = (1 - 30j) / (2 * math.pi * 1e9)
EPS0 = const.epsilon_0


def tight(solver: str, **kwargs) -> gfem.SolverOptions:
    """Order 2 on ``solver``, iterative solves to 1e-12 (gradient checks)."""
    return gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12, **kwargs)


def s_functional(s):
    """``<W, |S|^2> + Re(c S21)`` summed over frequencies, S (n_f, 2, 2)."""
    return jnp.sum(WEIGHTS * jnp.abs(s) ** 2) + jnp.sum(
        jnp.real((0.3 - 0.7j) * s[:, 1, 0])
    )


def _mesh(directory: Path, name: str, mesh: tuple, unit: str = "mm") -> gfem.Mesh:
    path = sm.write_msh22(Path(directory) / name, mesh)
    return gfem.Mesh.from_gmsh(path, length_unit=unit)


# --- driven: the two-layer line ----------------------------------------------------


def layered_line(directory: Path) -> gfem.Model:
    """Two layers between PEC plates, lumped ports, absorbing (order 2) sides."""
    mesh = _mesh(
        directory, "line.msh", sm.strip_line(4.0, 2.0, 1.0, (4, 2, 1), layers=LAYERS)
    )
    return gfem.Model(
        mesh,
        materials={"sub": gfem.Material(), "cover": gfem.Material()},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=2),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def line_design(line: gfem.Model, seed: int = 0):
    """Random P0 permittivity (lossy) and conductivity over both layers."""
    rng = np.random.default_rng(seed)
    n = len(line.mesh.tetrahedra)
    eps = rng.uniform(2.0, 6.0, n) * (1 - 0.02j)
    sigma = rng.uniform(0.5, 2.0, n)
    return {"eps_r": jnp.asarray(eps), "conductivity": jnp.asarray(sigma)}, rng


def with_fields(line: gfem.Model, values: dict) -> gfem.Model:
    """``line`` with ``values`` (one per tetrahedron, mesh order) as fields."""
    n_sub = len(line.mesh.elements("sub"))
    parts = {
        "sub": {k: v[:n_sub] for k, v in values.items()},
        "cover": {k: v[n_sub:] for k, v in values.items()},
    }
    return line.replace(
        materials={r: gfem.MaterialField(**p) for r, p in parts.items()}
    )


# --- electrostatics ----------------------------------------------------------------

#: Two terminals of `stack`.
STACK_PROBLEM = gfem.Electrostatic(
    terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
)


def stack(directory: Path) -> gfem.Model:
    """A 40 x 30 um oxide / nitride stack (um mesh), materials to be set."""
    layers = (("oxide", 0.5), ("nitride", 0.25))
    mesh = _mesh(directory, "stack.msh", sm.plate(40.0, 30.0, layers, (3, 2, 2)), "um")
    return gfem.Model(
        mesh, materials={"oxide": gfem.Material(), "nitride": gfem.Material()}
    )


def stack_fields(model: gfem.Model, eps) -> gfem.Model:
    """``model`` with one P0 permittivity ``eps`` over both layers."""
    n_ox = len(model.mesh.elements("oxide"))
    return model.replace(
        materials={
            "oxide": gfem.MaterialField(eps_r=eps[:n_ox]),
            "nitride": gfem.MaterialField(eps_r=eps[n_ox:]),
        }
    )


def capacitor(directory: Path) -> gfem.Model:
    """A 2 x 2 mm capacitor of two dielectric layers, one anisotropic."""
    layers = (("oxide", 0.5), ("nitride", 0.5))
    mesh = _mesh(directory, "plate.msh", sm.plate(2.0, 2.0, layers, (3, 3, 2)))
    return gfem.Model(
        mesh,
        materials={
            "oxide": gfem.Material(eps_r=3.9),
            "nitride": gfem.Material(eps_r=(7.5, 6.0, 5.0)),
        },
    )


# --- driven: shape designs ---------------------------------------------------------


def shape_line(directory: Path) -> gfem.Model:
    """The two-layer line with every kind of shape term.

    A lossy, conducting lower layer with random per-element permittivity,
    an anisotropic magnetic cover, a conductor top plate and second-order
    absorbing side walls.
    """
    mesh = _mesh(
        directory, "line.msh", sm.strip_line(4.0, 2.0, 1.0, (4, 2, 1), layers=LAYERS)
    )
    n = len(mesh.elements("sub"))
    eps = np.random.default_rng(3).uniform(2.0, 6.0, n) * (1 - 0.02j)
    return gfem.Model(
        mesh,
        materials={
            "sub": gfem.MaterialField(eps_r=eps, conductivity=np.full(n, 0.5)),
            "cover": gfem.Material(eps_r=(2.0, 2.5, 3.0), mu_r=1.3),
        },
        boundaries=[
            gfem.PEC("bottom"),
            gfem.Conductor("top", conductivity=5e7, thickness=2e-6),
            gfem.Absorbing("side", order=2),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


# --- driven: wave ports ------------------------------------------------------------

WAVE_LAYERS = (("lo", 0.5), ("hi", 0.5))
#: The leads' materials: a lossless and a lossy lower layer.
WAVE_PORTS = {
    "lossless": {"lo": gfem.Material(eps_r=4.0), "hi": gfem.Material(eps_r=2.0)},
    "lossy": {
        "lo": gfem.Material(eps_r=4.0, tan_delta=0.05),
        "hi": gfem.Material(eps_r=2.0),
    },
}
#: Reference-plane offsets (m) of the two ports.
WAVE_OFFSETS = {"lossless": (0.0, 0.0), "lossy": (0.3e-3, 0.1e-3)}


def wave_mesh(directory: Path) -> gfem.Mesh:
    """A two-layer line with a "core" across both layers between x = 1 and 3 mm."""
    return _mesh(
        directory,
        "line.msh",
        sm.strip_line(4.0, 2.0, 1.0, (8, 2, 2), layers=WAVE_LAYERS, core=(1.0, 3.0)),
    )


def wave_line(mesh: gfem.Mesh, port: str) -> gfem.Model:
    """Wave ports at both ends; ``port="mixed"``: lossy p1, lumped p2."""
    if port == "mixed":
        return wave_line(mesh, "lossy").replace(
            ports=[
                gfem.WavePort("p1", "port1", offset=WAVE_OFFSETS["lossy"][0]),
                gfem.LumpedPort("p2", "port2", "+z", impedance=60.0),
            ]
        )
    d1, d2 = WAVE_OFFSETS[port]
    return gfem.Model(
        mesh,
        materials={**WAVE_PORTS[port], "core": gfem.Material(eps_r=3.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.WavePort("p1", "port1", offset=d1),
            gfem.WavePort("p2", "port2", offset=d2),
        ],
    )


def core_design(mesh: gfem.Mesh, seed: int = 0):
    """Random P0 permittivity (lossy) and conductivity in the core."""
    rng = np.random.default_rng(seed)
    n = len(mesh.elements("core"))
    eps = rng.uniform(2.0, 6.0, n) * (1 - 0.02j)
    sigma = rng.uniform(0.5, 2.0, n)
    return {"eps_r": jnp.asarray(eps), "conductivity": jnp.asarray(sigma)}, rng


# --- sheet fields ------------------------------------------------------------------

#: Thin film: sigma t / 2 = 5e-4 S per square, comparable with the line's
#: wave admittance over its length.
SHEET_SIGMA, SHEET_THICKNESS = 1e3, 1e-6
SHEETS = {
    "conductor": lambda fill: gfem.Conductor(
        "sheet", SHEET_SIGMA, SHEET_THICKNESS, fill=fill
    ),
    "impedance": lambda fill: gfem.Impedance(
        "sheet", resistance=2e3, inductance=2e-8, fill=fill
    ),
}


def sheet_mesh(directory: Path) -> gfem.Mesh:
    """An 8 x 2 x 1 mm line glued from two halves at the surface "sheet"."""
    return _mesh(directory, "sheet.msh", sm.sheet_line(8.0, 2.0, 1.0, (6, 2, 1)))


def sheet_line(mesh: gfem.Mesh, sheet) -> gfem.Model:
    """The glued line with boundary ``sheet`` on "sheet", 200 ohm lumped ports."""
    return gfem.Model(
        mesh,
        materials={"lower": gfem.Material(eps_r=3.0), "upper": gfem.Material()},
        boundaries=[gfem.PEC("plate1"), gfem.PEC("plate2"), sheet],
        ports=[
            gfem.LumpedPort("p1", "port1", "+y", impedance=200.0),
            gfem.LumpedPort("p2", "port2", "+y", impedance=200.0),
        ],
    )


def sheet_fill(mesh: gfem.Mesh, seed: int):
    """A random grey fill of every face of "sheet"."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.2, 0.8, len(mesh.elements("sheet")))), rng


#: The sheet cavity: A x B x D mm, glued at z = D / 2.
CAVITY = (20.0, 10.0, 20.0)
CAVITY_CELLS = (6, 3, 3)  # per half along z
#: A resistive thin film: Q of a few hundred at TE101 (10.6 GHz).
CAVITY_SIGMA, CAVITY_THICKNESS = 100.0, 1e-6
CAVITY_TARGET = 10e9


def sheet_cavity(directory: Path) -> gfem.Mesh:
    """A PEC box glued at mid-height (surface "sheet"), volumes lower / upper."""
    a, b, d = CAVITY
    edges = (
        np.linspace(0, a, CAVITY_CELLS[0] + 1),
        np.linspace(0, b, CAVITY_CELLS[1] + 1),
        np.linspace(0, d, 2 * CAVITY_CELLS[2] + 1),
    )

    def surface(c):
        if abs(c[2] - d / 2) < 1e-9:
            return "sheet"
        on = (c[0], a - c[0], c[1], b - c[1], c[2], d - c[2])
        return "pec" if min(np.abs(on)) < 1e-9 else ""

    mesh = sm.structured(edges, lambda c: "lower" if c[2] < d / 2 else "upper", surface)
    return _mesh(directory, "cavity.msh", mesh)


def cavity_fill(mesh: gfem.Mesh, seed: int = 0):
    """A partial sheet: about a third of the faces empty, the rest grey."""
    rng = np.random.default_rng(seed)
    n = len(mesh.elements("sheet"))
    fill = rng.uniform(0.2, 1.0, n) * (rng.uniform(size=n) > 0.3)
    return jnp.asarray(fill), rng


def cavity_model(mesh: gfem.Mesh, fill) -> gfem.Model:
    """The sheet cavity with a thin-film `gfem.Conductor` of ``fill``."""
    sheet = gfem.Conductor(
        "sheet", CAVITY_SIGMA, CAVITY_THICKNESS, fill=gfem.SheetField(fill=fill)
    )
    return gfem.Model(
        mesh,
        materials={"lower": gfem.Material(), "upper": gfem.Material()},
        boundaries=[gfem.PEC("pec"), sheet],
    )


def on_the_sheet(directions, fill):
    """Unit directions moving only the filled faces (a fill stays >= 0)."""
    on = np.asarray(fill) > 0
    return [d * on / np.linalg.norm(d * on) for d in directions]


# --- eigenmode: line resonators ----------------------------------------------------

#: The resonators' length, width and height (mm).
RESONATOR = (10.0, 2.0, 1.0)
RESONATOR_TARGET = 7e9


def resonator_mesh(directory: Path) -> gfem.Mesh:
    """A 10 x 2 x 1 mm parallel-plate line, one volume "dielectric"."""
    return _mesh(directory, "line.msh", sm.strip_line(*RESONATOR, (8, 1, 1)))


def ramp(mesh: gfem.Mesh, region: str, start: float, stop: float):
    """Values from ``start`` to ``stop`` over the region's tetrahedra."""
    return jnp.asarray(np.linspace(start, stop, len(mesh.elements(region))))


def terminated(mesh: gfem.Mesh, eps, sigma) -> gfem.Model:
    """The line resonator with P0 fields, ended by 500 ohm lumped ports."""
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.MaterialField(eps_r=eps, conductivity=sigma)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=500.0),
            gfem.LumpedPort("p2", "port2", "+z", impedance=500.0),
        ],
    )


def layered_resonator(directory: Path) -> gfem.Model:
    """A two-layer line resonator ended by 500 ohm lumped ports.

    A conducting lower layer (a quadratic problem); moving the interface
    "sub__cover" changes the effective permittivity, so the frequency.
    """
    mesh = _mesh(
        directory, "layered.msh", sm.strip_line(*RESONATOR, (8, 1, 1), layers=LAYERS)
    )
    return gfem.Model(
        mesh,
        materials={
            "sub": gfem.Material(eps_r=4.0, conductivity=0.3),
            "cover": gfem.Material(eps_r=2.0),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=500.0),
            gfem.LumpedPort("p2", "port2", "+z", impedance=500.0),
        ],
    )


#: The mirrored cavity: TE101 = TE011 at 18.0 GHz, TM110 at 21.2 GHz.
MIRRORED = (10.0, 15.0)  # side and depth, mm
MIRRORED_TARGET = 17e9


def mirrored_cavity(directory: Path) -> gfem.Mesh:
    """A square-based PEC box (surface "wall", volume "fill").

    Mirrored Kuhn cells: the mesh has the square's symmetry, so the
    TE101 / TE011 pair is degenerate in the discrete problem too.
    """
    side, depth = MIRRORED
    edges = (
        np.linspace(0, side, 5),
        np.linspace(0, side, 5),
        np.linspace(0, depth, 7),
    )

    def surface(c):
        on = (c[0], side - c[0], c[1], side - c[1], c[2], depth - c[2])
        return "wall" if min(np.abs(on)) < 1e-9 else ""

    mesh = sm.structured(edges, lambda _: "fill", surface, mirror=(True, True, False))
    return _mesh(directory, "cavity.msh", mesh)
