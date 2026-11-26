r"""Isotropic sections and closed-form references of the mechanical eigenmode tests.

A glass-like isotropic solid (lengths in micrometres, read by
`gfem.Mesh.from_gmsh`) in rectangles whose four sides are the lines ``left``,
``right``, ``bottom`` and ``top``, one region ``film``. The references are the
closed forms of an isotropic strip, with :math:`v_T = \sqrt{\mu/\rho}` and
:math:`v_L = \sqrt{(\lambda + 2\mu)/\rho}`:

- anti-plane standing waves of a free rectangle (`sh_standing`);
- shear horizontal modes of a strip with sliding walls (`sh_strip`);
- Lamb modes of a plate and the Rayleigh wave (`lamb_roots`, `rayleigh_ratio`).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import piezo_cases as pc
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


def material(*, quality: float | None = None) -> gfem.Material:
    """The isotropic solid, with a uniform :math:`Q_m` if given."""
    return gfem.Material(
        stiffness=gfem.isotropic_stiffness(E, NU),
        mass_density=RHO,
        mechanical_quality_factor=quality,
    )


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
