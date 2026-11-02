"""Designs shared by the gradient tests (tests/verification).

The gradient verification matrix (tests/verification/test_gradient_matrix.py)
and the per-topic gradient tests around it build their models here, so a
matrix cell and the topic tests next to it differentiate the same design.
Every builder that writes a mesh takes the directory to write it to (a
pytest ``tmp_path_factory.mktemp`` directory).

- `layered_line`: a two-layer parallel-plate line, lumped ports, second-order
  absorbing side walls (driven and adaptive material gradients); first-order
  ones for fields with axes (`with_axes_fields`, on `anisotropy.TILTED`).
- `stack`, `capacitor`: two-layer capacitors (electrostatic material and
  shape gradients).
- `shape_line`: the two-layer line with a conductor top plate and a lossy,
  conducting lower layer (driven shape gradients).
- `wave_mesh` / `wave_line`: a two-layer line between wave ports with a
  design "core" away from the port faces.
- `hybrid_mesh` / `hybrid_line`: a line whose leads carry two hybrid modes,
  both requested at each end (a shared port surface), with the same "core".
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

import anisotropy
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


def _mesh(
    directory: Path, name: str, mesh: tuple, unit: str = "mm", rotation=None
) -> gfem.Mesh:
    if rotation is not None:
        mesh = anisotropy.rotate_mesh(mesh, rotation)
    path = sm.write_msh22(Path(directory) / name, mesh)
    return gfem.Mesh.from_gmsh(path, length_unit=unit)


# --- driven: the two-layer line ----------------------------------------------------


def layered_line(
    directory: Path, absorbing_order: int = 2, rotation=None
) -> gfem.Model:
    """Two layers between PEC plates, lumped ports, absorbing sides.

    Of ``absorbing_order``: second-order sides raise next to a field with
    axes, so the principal-value designs take first-order ones. With ``rotation``,
    a 3x3 rotation matrix, the line and its port directions are turned by
    it about the origin.
    """
    mesh = _mesh(
        directory,
        "line.msh",
        sm.strip_line(4.0, 2.0, 1.0, (4, 2, 1), layers=LAYERS),
        rotation=rotation,
    )
    up = "+z" if rotation is None else tuple(float(x) for x in rotation[:, 2])
    return gfem.Model(
        mesh,
        materials={"sub": gfem.Material(), "cover": gfem.Material()},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=absorbing_order),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", up),
            gfem.LumpedPort("p2", "port2", up),
        ],
    )


def line_design(line: gfem.Model, seed: int = 0):
    """Random P0 permittivity (lossy) and conductivity over both layers."""
    rng = np.random.default_rng(seed)
    n = len(line.mesh.tetrahedra)
    eps = rng.uniform(2.0, 6.0, n) * (1 - 0.02j)
    sigma = rng.uniform(0.5, 2.0, n)
    return {"eps_r": jnp.asarray(eps), "conductivity": jnp.asarray(sigma)}, rng


def with_fields(line: gfem.Model, values: dict, axes=None) -> gfem.Model:
    """``line`` with ``values`` (one per tetrahedron, mesh order) as fields.

    With ``axes``, the values are principal values, ``(n, 3)``, on them.
    """
    n_sub = len(line.mesh.elements("sub"))
    parts = {
        "sub": {k: v[:n_sub] for k, v in values.items()},
        "cover": {k: v[n_sub:] for k, v in values.items()},
    }
    return line.replace(
        materials={r: gfem.MaterialField(**p, axes=axes) for r, p in parts.items()}
    )


# --- principal values on fixed axes --------------------------------------------


#: Factors that make one value three distinct principal values (biaxial).
BIAXIAL = np.array([1.0, 1.15, 0.85])


def line_axes_design(line: gfem.Model, seed: int = 0):
    """`line_design` per principal value: ``(n, 3)``, biaxial in every element."""
    rng = np.random.default_rng(seed)
    n = len(line.mesh.tetrahedra)
    eps = rng.uniform(2.0, 6.0, (n, 3)) * (1 - 0.02j)
    sigma = rng.uniform(0.5, 2.0, (n, 3))
    return {"eps_r": jnp.asarray(eps), "conductivity": jnp.asarray(sigma)}, rng


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


def stack_fields(model: gfem.Model, eps, axes=None) -> gfem.Model:
    """``model`` with one P0 permittivity ``eps`` over both layers.

    With ``axes``, ``eps`` holds principal values, ``(n, 3)``, on them.
    """
    n_ox = len(model.mesh.elements("oxide"))
    return model.replace(
        materials={
            "oxide": gfem.MaterialField(eps_r=eps[:n_ox], axes=axes),
            "nitride": gfem.MaterialField(eps_r=eps[n_ox:], axes=axes),
        }
    )


def capacitor(directory: Path, rotation=None) -> gfem.Model:
    """A 2 x 2 mm capacitor of two dielectric layers, one anisotropic.

    With ``rotation``, a 3x3 rotation matrix, the mesh is turned by it
    about the origin (the materials are not).
    """
    layers = (("oxide", 0.5), ("nitride", 0.5))
    mesh = _mesh(
        directory,
        "plate.msh",
        sm.plate(2.0, 2.0, layers, (3, 3, 2)),
        rotation=rotation,
    )
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
    a magnetic cover with a per-axis loss tangent (an anisotropic mass
    tensor; its real part stays isotropic, as second-order absorbing side
    walls need), and a conductor top plate.
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
            "cover": gfem.Material(eps_r=2.5, tan_delta=(0.0, 0.01, 0.02), mu_r=1.3),
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


#: The hybrid line's two requested modes on each end (a shared port surface).
HYBRID_PORTS = ("p1_m1", "p1_m2", "p2_m1", "p2_m2")
#: Its permittivities: the leads' quadrant "block" and the rest, "fill".
HYBRID_EPS = {"block": 4.0, "fill": 2.0}


def hybrid_mesh(directory: Path, cells: tuple[int, int, int] = (8, 8, 4)) -> gfem.Mesh:
    """A 4 x 2 x 1 mm line, "block" in the quadrant y < 1, z < 0.5 of its leads.

    Plates "bottom" (z = 0) and "top" (z = 1), "side" walls (y = 0, 2),
    ports "port1" (x = 0) and "port2" (x = 4); the "core" between x = 1 and
    3 mm spans the cross-section. ``cells`` along x, y and z: multiples of
    4, 2 and 2 align the grid with the core and the quadrant.
    """
    length, width, height = 4.0, 2.0, 1.0
    edges = tuple(
        np.linspace(0, size, n + 1)
        for size, n in zip((length, width, height), cells, strict=True)
    )
    eps = 1e-9

    def volume(c):
        if 1.0 < c[0] < 3.0:
            return "core"
        return "block" if c[1] < width / 2 and c[2] < height / 2 else "fill"

    def surface(c):
        for x, name in ((0.0, "port1"), (length, "port2")):
            if abs(c[0] - x) < eps:
                return name
        if abs(c[2]) < eps:
            return "bottom"
        if abs(c[2] - height) < eps:
            return "top"
        if abs(c[1]) < eps or abs(c[1] - width) < eps:
            return "side"
        return ""  # the interfaces between the volumes

    return _mesh(directory, "hybrid.msh", sm.structured(edges, volume, surface))


def hybrid_line(mesh: gfem.Mesh) -> gfem.Model:
    """Modes 1 and 2 at both ends of the hybrid line; a uniform eps 3 core.

    The quadrant block makes the leads asymmetric in y and z, so their two
    propagating modes (from about 48 GHz) are hybrid and not orthogonal in
    the port face's mass product: the shared surface's Gram matrix is full.
    """
    return gfem.Model(
        mesh,
        materials={
            **{name: gfem.Material(eps_r=eps) for name, eps in HYBRID_EPS.items()},
            "core": gfem.Material(eps_r=3.0),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.WavePort(name, f"port{name[1]}", mode=int(name[-1]))
            for name in HYBRID_PORTS
        ],
    )


def hybrid_leads_core(mesh: gfem.Mesh) -> np.ndarray:
    """Per core element, the leads' permittivity there: a uniform line."""
    nodes = np.asarray(mesh.nodes) / 1e-3  # mm
    c = nodes[mesh.tetrahedra[mesh.elements("core")]].mean(axis=1)
    block = (c[:, 1] < 1.0) & (c[:, 2] < 0.5)
    return np.where(block, HYBRID_EPS["block"], HYBRID_EPS["fill"])


def shared_surface_functional(s, rows):
    """Weighted S of excited port p1_m1 at ``rows``: co-mode and cross-mode.

    ``|S|^2`` of the transmitted co-mode entry ``S[p2_m1, p1_m1]`` and of
    the cross-mode entries ``S[p1_m2, p1_m1]`` and ``S[p2_m2, p1_m1]``, plus
    ``Re(c S[p2_m2, p1_m1])``; ``s`` a `gfem.NetworkParameters`.
    """
    co, back, cross = (s[out, "p1_m1"][rows] for out in ("p2_m1", "p1_m2", "p2_m2"))
    return jnp.sum(
        jnp.abs(co) ** 2 + 1.5 * jnp.abs(back) ** 2 - 0.8 * jnp.abs(cross) ** 2
    ) + jnp.sum(jnp.real((0.3 - 0.7j) * cross))


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


def terminated(mesh: gfem.Mesh, eps, sigma, axes=None) -> gfem.Model:
    """The line resonator with P0 fields, ended by 500 ohm lumped ports.

    With ``axes``, ``eps`` and ``sigma`` hold principal values on them.
    """
    field = gfem.MaterialField(eps_r=eps, conductivity=sigma, axes=axes)
    return gfem.Model(
        mesh,
        materials={"dielectric": field},
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
