r"""Isotropic sections and closed-form references of the mechanical eigenmode tests.

A glass-like isotropic solid (lengths in micrometres, read by
`gfem.Mesh.from_gmsh`) in rectangles whose four sides are the lines ``left``,
``right``, ``bottom`` and ``top``, one region ``film``. The references are the
closed forms of an isotropic strip, with :math:`v_T = \sqrt{\mu/\rho}` and
:math:`v_L = \sqrt{(\lambda + 2\mu)/\rho}`:

- anti-plane standing waves of a free rectangle (`sh_standing`);
- shear horizontal modes of a strip with sliding walls (`sh_strip`);
- Lamb modes of a plate and the Rayleigh wave (`lamb_roots`, `rayleigh_ratio`).

The 3D bodies are boxes on a structured Kuhn grid of tetrahedra (`box`, in
micrometres, the faces ``x0 x1 y0 y1 z0 z1``), with the closed forms of a box
with sliding walls (`box_frequencies`) and of a plate cut to a box
(`lamb_box_frequencies`).
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import piezo_cases as pc
import structured_meshes as sm
import thermal_cases as tc
from scipy.optimize import brentq

import gfem

E, NU, RHO = 70e9, 0.25, 2500.0
MU = E / (2 * (1 + NU))
LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
V_T = float(np.sqrt(MU / RHO))
V_L = float(np.sqrt((LAMBDA + 2 * MU) / RHO))
#: A micrometre in metres.
UM = 1e-6


def rectangle(path: Path, width: float, height: float, maxh: float) -> gfem.Mesh:
    """The rectangle ``[0, width] x [0, height]`` (micrometres), region ``film``."""
    edges: list[tc.Namer] = [
        (lambda p: abs(p[0]) < 1e-9, "left"),
        (lambda p: abs(p[0] - width) < 1e-9, "right"),
        (lambda p: abs(p[1]) < 1e-9, "bottom"),
        (lambda p: abs(p[1] - height) < 1e-9, "top"),
    ]
    nodes, tris, lines = tc._mesh2d([(0, 0, width, height, "film", None)], maxh, edges)
    path = Path(path)
    tc._write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def box(
    path: Path,
    size: tuple[float, float, float],
    cells: float,
    *,
    regions: Callable[[np.ndarray], str] | None = None,
) -> gfem.Mesh:
    """The box ``[0, a] x [0, b] x [0, c]`` (micrometres) on a Kuhn grid of tetrahedra.

    ``cells`` per micrometre; the faces ``x0 x1 y0 y1 z0 z1``, one region ``body``
    or the name ``regions`` gives each cell from its centroid. The grid is
    deterministic, so every platform solves the same mesh.
    """
    edges = tuple(np.linspace(0, L, round(L * cells) + 1) for L in size)

    def face(c: np.ndarray) -> str:
        for i, axis in enumerate("xyz"):
            if abs(c[i]) < 1e-9:
                return f"{axis}0"
            if abs(c[i] - size[i]) < 1e-9:
                return f"{axis}1"
        return ""

    def volume(c: np.ndarray) -> str:
        return "body" if regions is None else regions(c)

    path = Path(path)
    sm.write_msh22(path, sm.carve(edges, volume, face, None))
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def material(*, quality: float | None = None) -> gfem.Material:
    """The isotropic solid, with a uniform :math:`Q_m` if given."""
    return gfem.Material(
        stiffness=gfem.isotropic_stiffness(E, NU),
        mass_density=RHO,
        mechanical_quality_factor=quality,
    )


#: The cubic silicon of the 3D boxes (crystal axes along the box's): 165.7, 63.9 and
#: 79.6 GPa, 2329 kg/m^3.
SI_C = (165.7e9, 63.9e9, 79.6e9)
SI_RHO = 2329.0
#: The sliding walls of a box: each face keeps the displacement component along
#: its normal at zero.
ROLLERS = {"x0": "x", "x1": "x", "y0": "y", "y1": "y", "z0": "z", "z1": "z"}
#: The Lamb box: rollers on the x and y faces, the z faces free.
LAMB_ROLLERS = {k: v for k, v in ROLLERS.items() if k[0] != "z"}


def _cubic(c11: float, c12: float, c44: float) -> np.ndarray:
    """The 6x6 stiffness of a cubic crystal in its axes."""
    c = np.zeros((6, 6))
    c[:3, :3] = c12
    return c + np.diag([c11 - c12] * 3 + [c44] * 3)


SILICON = gfem.Material(stiffness=_cubic(*SI_C), mass_density=SI_RHO)


def model(mesh: gfem.Mesh, *, quality: float | None = None) -> gfem.Model:
    """The model of a `rectangle`."""
    return gfem.Model(mesh, materials={"film": material(quality=quality)})


def solve(model: gfem.Model, target: float, n_modes: int = 1, **problem):
    """`gfem.solve` of a `gfem.MechanicalEigenmode`; ``problem`` are its arguments."""
    return gfem.solve(
        model, gfem.MechanicalEigenmode(target, n_modes=n_modes, **problem)
    )


# --- closed forms --------------------------------------------------------------------


def sh_standing(n: int, m: int, length: float, height: float) -> float:
    r"""Frequency (Hz) of the anti-plane mode :math:`\cos(n\pi x/L)\cos(m\pi y/h)`.

    Lengths in micrometres; every side of the rectangle is free.
    """
    return V_T / 2 * float(np.hypot(n / (length * UM), m / (height * UM)))


def sh_strip(m: int, q: float, height: float) -> float:
    r"""Frequency (Hz) of shear horizontal mode :math:`m` of a strip at :math:`q`.

    :math:`\omega^2 = v_T^2 (q^2 + (m\pi/h)^2)`, :math:`u_x = \cos(m\pi y/h)`,
    free faces in :math:`y`; ``height`` in micrometres.
    """
    return V_T * float(np.hypot(q, m * np.pi / (height * UM))) / (2 * np.pi)


def _lamb_function(q: float, height: float, kind: str):
    r"""The Rayleigh-Lamb relation in a form real at every frequency.

    With :math:`\alpha, \beta` the longitudinal and shear wavenumbers across
    the plate (:math:`d` = half the thickness), the symmetric relation
    :math:`(q^2 - \beta^2)^2 \sin\beta d \cos\alpha d + 4\alpha\beta q^2
    \cos\beta d \sin\alpha d = 0` divided by :math:`\beta`, and the
    antisymmetric one with sine and cosine swapped divided by :math:`\alpha`,
    are even in :math:`\alpha` and :math:`\beta`: real for real and imaginary
    ones alike, without the trivial roots at :math:`\alpha, \beta = 0`.
    """
    d = height * UM / 2

    def sinc(a: complex) -> complex:
        return np.sin(a * d) / a if abs(a) > 1e-12 else d

    def function(omega: float) -> float:
        a = np.sqrt(complex((omega / V_L) ** 2 - q**2))
        b = np.sqrt(complex((omega / V_T) ** 2 - q**2))
        if kind == "S":
            value = (q**2 - b**2) ** 2 * sinc(b) * np.cos(a * d)
            value += 4 * q**2 * np.cos(b * d) * a * np.sin(a * d)
        else:
            value = (q**2 - b**2) ** 2 * np.cos(b * d) * sinc(a)
            value += 4 * q**2 * b * np.sin(b * d) * np.cos(a * d)
        return float(value.real)

    return function


def lamb_roots(q: float, height: float, kind: str, count: int = 1) -> list[float]:
    """The ``count`` lowest frequencies (Hz) of the Lamb modes of a plate at ``q``.

    ``kind`` is ``"S"`` (symmetric, S0 first) or ``"A"`` (antisymmetric, A0
    first); the plate has the thickness ``height`` (micrometres). Roots are
    located by a sign scan up to six shear phase velocities and refined with
    `scipy.optimize.brentq`.
    """
    function = _lamb_function(q, height, kind)
    omegas = np.linspace(1e-3, 6, 6000) * V_T * q
    roots = []
    previous = function(omegas[0])
    for a, b in itertools.pairwise(omegas):
        value = function(b)
        if previous * value < 0:
            roots.append(brentq(function, a, b, xtol=1e-3, rtol=1e-14) / (2 * np.pi))
            if len(roots) == count:
                break
        previous = value
    return roots


def rayleigh_ratio() -> float:
    r""":math:`v_R / v_T`, the root of the Rayleigh equation in (0, 1).

    :math:`\xi^6 - 8\xi^4 + (24 - 16\kappa^{-2})\xi^2 - 16(1 - \kappa^{-2}) = 0`
    with :math:`\kappa = v_L / v_T`.
    """
    k2 = (V_T / V_L) ** 2
    roots = np.roots([1, 0, -8, 0, 24 - 16 * k2, 0, -16 * (1 - k2)])
    real = [r.real for r in roots if abs(r.imag) < 1e-12 and 0 < r.real < 1]
    return float(max(real))


def box_frequencies(
    size: tuple[float, float, float],
    c: tuple[float, float, float],
    rho: float,
    count: int,
) -> list[float]:
    r"""The ``count`` lowest frequencies (Hz) of a cubic solid in a box with rollers.

    With the walls sliding (`ROLLERS`) the modes are the standing waves
    :math:`u_i \propto \sin(k_i x_i) \prod_{j\ne i}\cos(k_j x_j)`,
    :math:`k = \pi (l/a, m/b, n/c)`, whose amplitudes are the eigenvectors of
    the Christoffel matrix :math:`\Gamma(k)/\rho`. A component whose index is
    zero vanishes (its sine is zero), so the matrix keeps the rows and columns
    of the nonzero indices only. ``c`` is :math:`(c_{11}, c_{12}, c_{44})`,
    ``size`` in micrometres; the indices run to 3.
    """
    c11, c12, c44 = c
    out = []
    for index in itertools.product(range(4), repeat=3):
        keep = np.array(index) > 0
        if not keep.any():
            continue
        kx, ky, kz = np.pi * np.array(index) / (np.array(size) * UM)
        gamma = np.array(
            [
                [
                    c11 * kx**2 + c44 * (ky**2 + kz**2),
                    (c12 + c44) * kx * ky,
                    (c12 + c44) * kx * kz,
                ],
                [
                    (c12 + c44) * kx * ky,
                    c11 * ky**2 + c44 * (kx**2 + kz**2),
                    (c12 + c44) * ky * kz,
                ],
                [
                    (c12 + c44) * kx * kz,
                    (c12 + c44) * ky * kz,
                    c11 * kz**2 + c44 * (kx**2 + ky**2),
                ],
            ]
        )[np.ix_(keep, keep)]
        out += [
            float(np.sqrt(w)) / (2 * np.pi) for w in np.linalg.eigvalsh(gamma) / rho
        ]
    return sorted(out)[:count]


def lamb_box_frequencies(size: tuple[float, float, float], count: int) -> list[float]:
    r"""The ``count`` lowest frequencies (Hz) of the isotropic Lamb box.

    A plate of thickness ``size[2]`` (micrometres) cut to ``size[0] x size[1]``
    with rollers on the x and y faces (`LAMB_ROLLERS`): the Lamb modes
    (`lamb_roots`) at :math:`q = \pi\sqrt{(l/a)^2 + (m/b)^2}`, the shear
    horizontal modes :math:`v_T\sqrt{q^2 + (n\pi/c)^2}/2\pi` (:math:`l, m > 0`)
    and the thickness modes of :math:`q = 0` (:math:`n v_L / 2c`).
    The rigid translation along z is not a mode.
    """
    a, b, c = size
    out = [n * V_L / (2 * c * UM) for n in (1, 2)]
    for index in itertools.product(range(4), repeat=2):
        if not any(index):
            continue
        l, m = index
        q = np.pi * float(np.hypot(l / (a * UM), m / (b * UM)))
        out += lamb_roots(q, c, "S", 3) + lamb_roots(q, c, "A", 3)
        if l and m:
            out += [
                V_T * float(np.hypot(q, n * np.pi / (c * UM))) / (2 * np.pi)
                for n in range(3)
            ]
    return sorted(out)[:count]


def sandwich_model(path: Path) -> gfem.Model:
    """A 1 um soft core between two 1.5 um stiffer claddings (4x the modulus).

    Regions ``core``, ``clad_lo`` and ``clad_hi``, 0.4 um wide, maxh 0.25; the lines
    ``left``, ``right``, ``bottom`` and ``top``. Shear horizontal modes at q = 3e6
    rad/m are confined between the core's and the cladding's shear velocity
    (v_T q < omega < 2 v_T q) and leak above it.
    """
    w, core, clad = 0.4, 1.0, 1.5
    rects = [
        (0, -clad, w, 0, "clad_lo", None),
        (0, 0, w, core, "core", None),
        (0, core, w, core + clad, "clad_hi", None),
    ]
    edges = [
        (lambda p: abs(p[0]) < 1e-9, "left"),
        (lambda p: abs(p[0] - w) < 1e-9, "right"),
        (lambda p: abs(p[1] + clad) < 1e-9, "bottom"),
        (lambda p: abs(p[1] - core - clad) < 1e-9, "top"),
    ]
    nodes, tris, lines = tc._mesh2d(rects, 0.25, edges)
    path = Path(path)
    tc._write_section(path, nodes, tris, lines)
    stiff = gfem.Material(
        stiffness=gfem.isotropic_stiffness(4 * E, NU), mass_density=RHO
    )
    materials = {"core": material(), "clad_lo": stiff, "clad_hi": stiff}
    return gfem.Model(gfem.Mesh.from_gmsh(path, length_unit="um"), materials=materials)


# --- a piezoelectric plate (6mm crystal) --------------------------------------------


def piezo_plate_mesh(path: Path, width: float, height: float, maxh: float):
    """A plate's rectangle: lines ``left``, ``right``, ``bot``, ``top``."""
    edges: list[tc.Namer] = [
        (lambda p: abs(p[0]) < 1e-9, "left"),
        (lambda p: abs(p[0] - width) < 1e-9, "right"),
        (lambda p: abs(p[1]) < 1e-9, "bot"),
        (lambda p: abs(p[1] - height) < 1e-9, "top"),
    ]
    nodes, tris, lines = tc._mesh2d([(0, 0, width, height, "pz", None)], maxh, edges)
    path = Path(path)
    tc._write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def piezo_plate_model(mesh: gfem.Mesh, kind: str, *, quality=None) -> gfem.Model:
    """The plate of crystal ``kind`` (``"TE"``: c axis along y, ``"TS"``: along z)."""
    return pc.plate_model(mesh, kind, quality=quality)


def stiffened_frequencies(ky: float, q: float) -> list[float]:
    r"""Frequencies (Hz) of the plane waves :math:`k = (0, k_y, q)` of the TS crystal.

    The two in-plane branches of the piezoelectrically stiffened Christoffel
    equation :math:`\rho\omega^2 p = (B^T c B + \gamma\gamma^T/(k\cdot\epsilon k))
    p`, :math:`B` the strain of :math:`p\,e^{-ik\cdot r}` (Voigt, engineering
    shear) and :math:`\gamma = B^T e^T k`; the crystal axes are the lab's.
    """
    c, e, eps = pc.hexagonal("TS")
    k = np.array([0.0, ky, q])
    b = np.array(
        [
            [k[0], 0, 0],
            [0, k[1], 0],
            [0, 0, k[2]],
            [0, k[2], k[1]],
            [k[2], 0, k[0]],
            [k[1], k[0], 0],
        ]
    )
    d = pc.EPS0 * np.diag(eps)
    gamma = b.T @ e.T @ k
    gt = b.T @ c @ b + np.outer(gamma, gamma) / (k @ d @ k)
    values = np.linalg.eigvalsh(gt) / pc.HEX["rho"]
    # the in-plane (y, z) branches: the eigenvectors without an x component
    vectors = np.linalg.eigh(gt)[1]
    in_plane = np.abs(vectors[0]) < 1e-8
    return [
        float(np.sqrt(v)) / (2 * np.pi)
        for v, ok in zip(values, in_plane, strict=True)
        if ok
    ]


# --- editing a model's material leaves (the gradient tests) --------------------------


def material_fields(model: gfem.Model) -> dict:
    """``model``'s materials, each a `gfem.MaterialField` (a uniform one if not)."""
    return {
        region: material
        if isinstance(material, gfem.MaterialField)
        else gfem.MaterialField.uniform(region, material)
        for region, material in model.materials.items()
    }


def edited(model: gfem.Model, regions, **values) -> gfem.Model:
    """``model`` with the leaves ``values`` of the materials of ``regions`` replaced."""
    fields = material_fields(model)
    for region in regions:
        fields[region] = dataclasses.replace(fields[region], **values)
    return model.replace(materials=fields)


def symmetric_direction(seed: int) -> np.ndarray:
    """A random unit symmetric 6 x 6 direction (a stiffness's)."""
    d = np.random.default_rng(seed).standard_normal((6, 6))
    d = d + d.T
    return d / np.linalg.norm(d)


def per_element_direction(mesh: gfem.Mesh, region: str, seed: int) -> np.ndarray:
    """A random unit direction of one value per element of ``region``."""
    d = np.random.default_rng(seed).standard_normal(len(mesh.elements(region)))
    return d / np.linalg.norm(d)


def mass_energy(
    model: gfem.Model, u: gfem.FieldMap, *, rho: float = RHO, order: int = 2
):
    r""":math:`\int \rho |u|^2 \mathrm{d}V` of a displacement `gfem.FieldMap`, in SI.

    ``dV`` is the area (2D section, per metre along z) or the volume (3D body)
    of the mesh in metres: the mesh's unit to the power of its dimension.
    """
    import ngsolve as ngs

    from gfem._engine.cache import discretization
    from gfem._engine.domain import em_region, h1_space

    built = discretization(model.mesh).built
    space = h1_space(model, built, order, (), physics="mechanical", complex_valued=True)
    total = 0.0
    with ngs.TaskManager():
        for axis in "xyz":
            g = ngs.GridFunction(space)
            g.vec.FV().NumPy()[:] = np.asarray(u[axis].coefficients)
            region = em_region(model, built, "mechanical")
            total += ngs.Integrate(
                rho * (g * ngs.Conj(g)).real * built.scale**model.mesh.dim,
                built.mesh,
                definedon=region,
                order=2 * order + 2,
            )
    return total
