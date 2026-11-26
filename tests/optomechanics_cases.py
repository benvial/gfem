"""Displacements and slab closed forms shared by the optomechanics tests.

The optical section is the layered slab of the electro-optic tests (oxide, a
0.6 um film, oxide, layers along y) and the mechanical side a native 2D mesh of
the same (x, y) extent. `displacement` is the test helper that builds a
displacement `gfem.FieldMap` from an analytic field on such a mesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import acousto_optic_cases as ao
import electro_optic_cases as eo
import numpy as np
import scipy.constants
import thermal_cases as tc
from scipy.optimize import brentq

import gfem
from gfem._field import FieldMap

#: Silicon-like photoelastic constants (Voigt, cubic), the set of the published
#: suspended wire: p11, p12, p44.
P11, P12, P44 = -0.09, 0.017, -0.051
#: The film's permittivity (n = 3.48) and the oxide's.
N_FILM = 3.48
EPS_FILM = N_FILM**2


def cubic_p(p11=P11, p12=P12, p44=P44) -> np.ndarray:
    """The contracted photoelastic tensor of a cubic crystal on its axes."""
    p = np.full((3, 3), p12)
    np.fill_diagonal(p, p11)
    out = np.zeros((6, 6))
    out[:3, :3] = p
    out[3:, 3:] = np.eye(3) * p44
    return out


def displacement(
    mesh, support, gradients, offsets=(0.0, 0.0, 0.0), *, order=1, wavenumber=0.0
):
    """A displacement `gfem.FieldMap` ``u_c = offsets[c] + gradients[c] . (x, y)``.

    Built by the helper of the acousto-optic tests: exact fields on a 2D mesh at
    ``order`` 1 (an affine field is in the space), complex data allowed, in
    metres, defined on the regions ``support``, with the axial ``wavenumber``
    (rad/m). ``gradients`` is ``(3, 2)`` (per metre), ``offsets`` ``(3,)``
    (metres). This is the test helper the
    optomechanical overlap needs: a mechanical mode of a solve is a `FieldMap` of
    the same kind.
    """
    parts = {
        c: ao.complex_field(
            mesh,
            tuple(support),
            f"u{c}",
            "m",
            np.asarray(gradients[i]),
            offset=offsets[i],
            order=order,
        )
        for i, c in enumerate("xyz")
    }
    return FieldMap(parts, label="component", wavenumber=wavenumber)


#: The slab's half width and cladding thickness, um, the film's thickness: the
#: cladding is a few decay lengths (1/gamma ~ 0.09 um) so the natural walls above
#: and below do not show, the width is free (the mode is uniform in x).
HALF, CLAD, FILM = 1.0, 1.2, eo.SLAB_FILM


def _rects(margin, edge_film, edge_clad):
    h, c, t = HALF + margin, CLAD + margin, FILM
    return [
        (-h, -c, h, 0.0, "below", edge_clad),
        (-h, 0.0, h, t, "film", edge_film),
        (-h, t, h, t + c, "above", edge_clad),
    ]


def slab(path: Path, factor: float = 1.0) -> gfem.Mesh:
    """The slab (regions ``below``, ``film``, ``above``; edges ``left``, ``right``).

    The film is meshed at 0.08 ``factor`` um and the cladding at 0.12 ``factor``
    um, fine enough for the evanescent tails (decay length 0.09 um).
    """
    edges = [
        (lambda p: abs(p[0] - HALF) < 1e-9, "right"),
        (lambda p: abs(p[0] + HALF) < 1e-9, "left"),
    ]
    return tc.section(
        path, _rects(0.0, 0.08 * factor, 0.12 * factor), 0.3 * factor, edges
    )


def mechanical_mesh(path: Path, maxh: float = 0.3, margin: float = 0.5) -> gfem.Mesh:
    """A native 2D mesh over the slab's (x, y) extent, ``margin`` um larger."""
    nodes, tris, lines = tc._mesh2d(_rects(margin, None, None), maxh, [])
    tc._write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def model(mesh, film, *, pec=True):
    """The slab with ``film`` as the core between oxide layers; PEC side walls
    keep the TE mode the 1D slab's, natural (PMC) ones the TM mode."""
    oxide = gfem.Material(eps_r=eo.N_OX**2)
    return gfem.Model(
        mesh,
        materials={"below": oxide, "film": film, "above": oxide},
        boundaries=[gfem.PEC("left"), gfem.PEC("right")] if pec else [],
        ports=[gfem.WavePort("p", "front")],
    )


def slab_n_eff(eps, d, polarization, *, n_clad=eo.N_OX, wavelength=eo.WAVELENGTH):
    r"""The even mode of a film of thickness ``d`` (m) between identical claddings.

    ``eps`` is the film's diagonal permittivity ``(xx, yy, zz)``, layers along y,
    propagation along z. TE: :math:`E_x` only, :math:`\kappa^2 = k_0^2\varepsilon_{xx}
    - \beta^2`, :math:`\tan(\kappa d/2) = \gamma/\kappa`. TM: :math:`H_x` only,
    :math:`\kappa^2 = \varepsilon_{zz}(k_0^2 - \beta^2/\varepsilon_{yy})`,
    :math:`\kappa\tan(\kappa d/2)/\varepsilon_{zz} = \gamma/n_c^2`;
    :math:`\gamma^2 = \beta^2 - n_c^2 k_0^2`.
    """
    k0 = 2 * np.pi / wavelength
    ex, ey, ez = eps
    ec = n_clad**2

    def gamma(n):
        return k0 * np.sqrt(n * n - ec)

    if polarization == "TE":

        def kappa(n):
            return k0 * np.sqrt(ex - n * n)

        def residual(n):
            return np.tan(kappa(n) * d / 2) - gamma(n) / kappa(n)

        top = np.sqrt(ex)
    else:

        def kappa(n):
            return k0 * np.sqrt(ez * (1.0 - n * n / ey))

        def residual(n):
            return kappa(n) * np.tan(kappa(n) * d / 2) / ez - gamma(n) / ec

        top = np.sqrt(ey)
    lo = n_clad + 1e-12
    hi = top - 1e-12
    if polarization == "TE":
        n_pole = np.sqrt(ex - (np.pi / (k0 * d)) ** 2)
    else:
        n_pole = np.sqrt(ey * (1.0 - (np.pi / (k0 * d)) ** 2 / ez))
    # the fundamental mode is the highest root: past the tangent's last pole below
    # the film's index (residual +inf there, -inf at the film's index)
    if n_pole > n_clad:
        lo = n_pole + 1e-9
    return brentq(residual, lo, hi, xtol=1e-15, rtol=1e-15)


def central(f, x, h):
    """The fourth-order central difference of ``f`` at ``x`` with step ``h``."""
    return (8 * (f(x + h) - f(x - h)) - (f(x + 2 * h) - f(x - 2 * h))) / (12 * h)


def eps_derivative(eps, d, polarization, axis, h=0.02):
    r"""The slab's :math:`\partial n_\mathrm{eff}/\partial\varepsilon_{ii}`.

    ``axis`` is the film's diagonal component i (0, 1, 2 for x, y, z).

    Fourth-order central difference of the exact dispersion; truncation
    :math:`h^4 f^{(5)}/30 \sim 10^{-12}` for :math:`h = 0.02`, round-off
    :math:`1.5\,\delta/h \sim 10^{-13}` for the root's :math:`\delta = 10^{-15}`.
    """

    def f(delta):
        e = list(eps)
        e[axis] += delta
        return slab_n_eff(tuple(e), d, polarization)

    return central(f, 0.0, h)


def thickness_derivative(eps, d, polarization, h=1e-9):
    r"""The slab's :math:`\partial n_\mathrm{eff}/\partial d` (per metre).

    The step is 1e-3 of the film: truncation :math:`(h/d)^4 \sim 10^{-12}`.
    """
    return central(lambda delta: slab_n_eff(eps, d + delta, polarization), 0.0, h)


class SlabMoments(NamedTuple):
    """The TM slab's film moments at 1 W (see `slab_tm_moments`)."""

    n_eff: float
    beta: float
    corner: float
    gamma_y: float
    gamma_z: float
    shear: complex


def slab_tm_moments(eps=EPS_FILM, d=FILM * 1e-6) -> SlabMoments:
    r"""The closed-form moments of the TM slab mode that a backward pair reads.

    From :math:`H_x = \cos\kappa y'` in the film (:math:`y'` from its middle),
    :math:`E_y = -\beta H_x/(\omega\varepsilon_0\varepsilon)` and
    :math:`E_z = i H_x'/(\omega\varepsilon_0\varepsilon)`, at 1 W (the
    power per unit width divides every moment, so the width drops out):

    - ``corner`` :math:`C = \tfrac{c\varepsilon_0}{4}\,d\,(\varepsilon_f -
      \varepsilon_c)|E_z(d/2)|^2`, the film's two interfaces, each of
      half the dilation :math:`d/2`, in the moving-boundary correction
      :math:`-2C`;
    - ``gamma_y``, ``gamma_z`` :math:`\Gamma_{y,z} = \tfrac{c\varepsilon_0}{4}
      \int_\mathrm{film}|E_{y,z}|^2`;
    - ``shear`` :math:`M = \tfrac{c\varepsilon_0}{4}\int_\mathrm{film}
      y' E_y E_z^*`, purely imaginary.
    """
    n = slab_n_eff((eps,) * 3, d, "TM")
    k0 = 2 * np.pi / eo.WAVELENGTH
    beta, omega = n * k0, 2 * np.pi * eo.FREQUENCY
    ec = eo.N_OX**2
    kap = k0 * np.sqrt(eps - n * n)
    gam = k0 * np.sqrt(n * n - ec)
    scale = scipy.constants.c * scipy.constants.epsilon_0 / 4
    # power per unit width: (beta / (2 omega eps0)) int |H_x|^2 / eps dy
    integral = (d / 2 + np.sin(kap * d) / (2 * kap)) / eps + (
        2 * np.cos(kap * d / 2) ** 2 / (2 * gam) / ec
    )
    power = beta / (2 * omega * scipy.constants.epsilon_0) * integral
    a = omega * scipy.constants.epsilon_0 * eps
    ez_edge = kap * np.sin(kap * d / 2) / a
    moment = np.sin(kap * d) / (4 * kap**2) - d * np.cos(kap * d) / (4 * kap)
    return SlabMoments(
        n_eff=n,
        beta=beta,
        corner=scale * d * (eps - ec) * ez_edge**2 / power,
        gamma_y=scale / power * (beta / a) ** 2 * (d / 2 + np.sin(kap * d) / (2 * kap)),
        gamma_z=scale / power * (kap / a) ** 2 * (d / 2 - np.sin(kap * d) / (2 * kap)),
        shear=scale / power * (-1j * beta * kap / a**2) * moment,
    )


C = scipy.constants.c


#: The suspended film's mechanical constants (silicon-like, isotropic): Young's
#: modulus (Pa), Poisson's ratio, density (kg/m^3).
E_SI, NU_SI, RHO_SI = 165e9, 0.22, 2329.0


def film_rectangle(path: Path, maxh: float = 0.3) -> gfem.Mesh:
    """A native 2D mesh of the film alone (``x`` in +-HALF, ``y`` in 0..FILM, um).

    The optical slab's film and nothing else, the region ``film``: a rectangle
    suspended in air, every side free.
    """
    nodes, tris, lines = tc._mesh2d([(-HALF, 0.0, HALF, FILM, "film", None)], maxh, [])
    tc._write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


#: The oxide that `embedded` puts beside the film, um on each side.
SIDE = 0.5


def embedded(path: Path, factor: float = 1.0) -> gfem.Mesh:
    r"""The slab with oxide beside the film: its side walls clear the film.

    Regions ``below``, ``film``, ``above`` as in `slab` and ``side``, the oxide
    `SIDE` um wide on either side of the film; the edges ``left`` and ``right``
    at :math:`x = \pm(h + s)` for the film's half width :math:`h` (`HALF`) and
    :math:`s` = `SIDE`. A free film, whose ends move along x, leaves the walls
    in place. Edge lengths as in `slab`.
    """
    h, c, t, w = HALF + SIDE, CLAD, FILM, HALF
    rects = [
        (-h, -c, h, 0.0, "below", 0.12 * factor),
        (-h, 0.0, -w, t, "side", 0.12 * factor),
        (-w, 0.0, w, t, "film", 0.08 * factor),
        (w, 0.0, h, t, "side", 0.12 * factor),
        (-h, t, h, t + c, "above", 0.12 * factor),
    ]
    edges = [
        (lambda p: abs(p[0] - h) < 1e-9, "right"),
        (lambda p: abs(p[0] + h) < 1e-9, "left"),
    ]
    return tc.section(path, rects, 0.3 * factor, edges)


def embedded_model(mesh, film) -> gfem.Model:
    """`embedded` with ``film`` in oxide and PEC side walls."""
    oxide = gfem.Material(eps_r=eo.N_OX**2)
    return gfem.Model(
        mesh,
        materials={"below": oxide, "side": oxide, "film": film, "above": oxide},
        boundaries=[gfem.PEC("left"), gfem.PEC("right")],
        ports=[gfem.WavePort("p", "front")],
    )


def film_material(*, quality: float | None = None) -> gfem.Material:
    """The film's mechanical material, with a uniform :math:`Q_m` if given."""
    return gfem.Material(
        stiffness=gfem.isotropic_stiffness(E_SI, NU_SI),
        mass_density=RHO_SI,
        mechanical_quality_factor=quality,
    )


def film_model(mesh: gfem.Mesh, *, quality: float | None = None) -> gfem.Model:
    """The free film as a mechanical model."""
    return gfem.Model(mesh, materials={"film": film_material(quality=quality)})


# --- the suspended silicon wire (published case) ----------------------------------

#: Silicon's index and the wavelength of the published wire (Qiu et al. 2013:
#: omega = 0.203 (2 pi c / a) at a = 315 nm).
N_SI = 3.5
WIRE_WAVELENGTH = 1.552e-6


def wire_section(path: Path, width: float, height: float, h_wire: float, h_air: float):
    """The wire (region ``wire``) in a 1 um air margin, PEC outer ``wall``, as a slab.

    ``width``, ``height`` in um; edge lengths ``h_wire`` and ``h_air`` in um.
    """
    x1, y1 = width / 2, height / 2
    big_x, big_y = x1 + 1.0, y1 + 1.0
    rects = [
        (-big_x, -big_y, big_x, -y1, "air", h_air),
        (-big_x, y1, big_x, big_y, "air", h_air),
        (-big_x, -y1, -x1, y1, "air", h_air),
        (x1, -y1, big_x, y1, "air", h_air),
        (-x1, -y1, x1, y1, "wire", h_wire),
    ]
    wall = [(lambda p: max(abs(p[0]) - big_x, abs(p[1]) - big_y) > -1e-9, "wall")]
    return tc.section(path, rects, h_air, wall)


def wire_optical(mesh) -> gfem.Model:
    """The wire's optical model: silicon with the cubic photoelastic tensor."""
    return gfem.Model(
        mesh,
        materials={
            "wire": gfem.Material(eps_r=N_SI**2, photoelastic_tensor=cubic_p()),
            "air": gfem.Material(eps_r=1.0),
        },
        boundaries=[gfem.PEC("wall")],
        ports=[gfem.WavePort("p", "front")],
    )


def wire_mechanical(path: Path, width: float, height: float, h: float) -> gfem.Model:
    """The wire's mechanical model on a native 2D mesh (um in, isotropic silicon)."""
    rect = [(-width / 2, -height / 2, width / 2, height / 2, "wire", h)]
    nodes, tris, lines = tc._mesh2d(rect, h, [])
    tc._write_section(path, nodes, tris, lines)
    silicon = gfem.Material(
        stiffness=gfem.isotropic_stiffness(170e9, 0.28),
        mass_density=2329.0,
        mechanical_quality_factor=1000.0,
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"), materials={"wire": silicon}
    )


def parities(displacement, width: float, height: float):
    r"""The mode's mirror parities and :math:`u_z` share, sampled on a grid.

    Returns ``(sx, sy, uz_share)``: the eigenvalue (+1 even, -1 odd) of the
    vector field under :math:`x \to -x` and :math:`y \to -y`, as
    :math:`\sum \bar u\cdot Ru' / \sum |u|^2` over 80 interior points (m), and
    :math:`\sum|u_z|^2/\sum|u|^2`.
    """
    g = np.linspace(-0.8, 0.8, 9)
    pts = [
        (a * width / 2e6, b * height / 2e6) for a in g for b in g if (a, b) != (0, 0)
    ]

    def vec(p):
        return np.array([complex(displacement[c].at([p])[0]) for c in "xyz"])

    v = np.array([vec(p) for p in pts])
    norm = np.sum(np.abs(v) ** 2)
    out = []
    for sgn in ((-1, 1, 1), (1, -1, 1)):
        vm = np.array(
            [vec((sgn[0] * p[0], sgn[1] * p[1])) * np.array(sgn) for p in pts]
        )
        out.append(float(np.sum(np.conj(v) * vm).real / norm))
    return out[0], out[1], float(np.sum(np.abs(v[:, 2]) ** 2) / norm)


# --- a curved interface: an elliptic rod ---------------------------------------------

#: The rod's semi-axes along x and y (um) and the PEC box's half widths around it:
#: an ellipse, so the two lowest modes are split (a circular rod's are degenerate).
ROD_A, ROD_B, ROD_BOX = 0.45, 0.25, (1.2, 1.0)


def _rod_mesh2d(h_rod: float, h_air: float | None):
    """Netgen's triangles of the rod alone, or in its air box when ``h_air`` is given.

    Returns nodes (um), ``(name, vertices)`` triangles and the box's ``wall`` lines.
    """
    import ngsolve as ngs
    from netgen.occ import Glue, OCCGeometry, Rectangle, WorkPlane

    rod = WorkPlane().Ellipse(ROD_A, ROD_B).Face()
    rod.faces.name = "rod"
    rod.faces.maxh = h_rod
    if h_air is None:
        shape, maxh = rod, h_rod
    else:
        bx, by = ROD_BOX
        box = Rectangle(2 * bx, 2 * by).Face().Move((-bx, -by, 0))
        air = box - rod
        air.faces.name = "air"
        shape, maxh = Glue([air, rod]), h_air
        for edge in shape.edges:
            c = edge.center
            if max(abs(c.x) - bx, abs(c.y) - by) > -1e-9:
                edge.name = "wall"
    mesh = ngs.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
    nodes = np.array([v.point for v in mesh.vertices])
    tris = [(el.mat, [v.nr for v in el.vertices]) for el in mesh.Elements(ngs.VOL)]
    lines = [
        (el.mat, [v.nr for v in el.vertices])
        for el in mesh.Elements(ngs.BND)
        if el.mat == "wall"
    ]
    return nodes, tris, lines


def rod_meshes(path: Path, h_rod: float, h_air: float) -> tuple[gfem.Mesh, gfem.Mesh]:
    """The rod's optical slab and the conforming mechanical mesh cut from it.

    The slab holds the regions ``rod`` and ``air`` with the edge ``wall``; the
    second mesh is the slab section's own ``rod`` triangles as a native 2D mesh,
    so every interface node of the section is one of its vertices.
    """
    nodes, tris, lines = _rod_mesh2d(h_rod, h_air)
    tc._write_section(path / "rod_slab.msh", nodes, tris, lines)
    slab = gfem.Mesh.from_gmsh_cross_section(path / "rod_slab.msh")
    kept = [(name, v) for name, v in tris if name == "rod"]
    used = np.unique([i for _, v in kept for i in v])
    renumber = {int(old): new for new, old in enumerate(used)}
    rod = [(name, [renumber[i] for i in v]) for name, v in kept]
    tc._write_section(path / "rod_conforming.msh", nodes[used], rod, [])
    return slab, gfem.Mesh.from_gmsh(path / "rod_conforming.msh", length_unit="um")


def rod_mechanical(path: Path, h: float) -> gfem.Mesh:
    """The rod alone meshed on its own at edge ``h`` um: its boundary vertices are
    on the ellipse, but not the slab's, which lie off its boundary polygon."""
    nodes, tris, _ = _rod_mesh2d(h, None)
    tc._write_section(path, nodes, tris, [])
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def on_rod(xy: np.ndarray) -> np.ndarray:
    """Which points ``xy`` (``(n, 2)``, m) lie in the rod (its boundary to 1e-9)."""
    r = (xy[:, 0] / (ROD_A * 1e-6)) ** 2 + (xy[:, 1] / (ROD_B * 1e-6)) ** 2
    return r <= 1 + 1e-9


def rod_optical(mesh: gfem.Mesh) -> gfem.Model:
    """The rod (index `N_FILM`) in air in a PEC box, without a photoelastic tensor."""
    return gfem.Model(
        mesh,
        materials={
            "rod": gfem.Material(eps_r=N_FILM**2),
            "air": gfem.Material(eps_r=1.0),
        },
        boundaries=[gfem.PEC("wall")],
        ports=[gfem.WavePort("p", "front")],
    )
