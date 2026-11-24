"""The driven operator as frequency-independent pieces times scalar θ(ω).

Under e^{+iωt}, with lengths in the mesh file's unit (``scale`` metres) and
the equation multiplied by ``mu0 * scale``, the driven problem for the
electric field E (volts per file unit) is

    a(E, v; ω) = Σ_k θ_k(ω) a_k(E, v) = i k0 Σ_p b_p(v)

with k0 = ω scale / c, the wavenumber per file unit, and the pieces

- ``curl_curl``, θ = 1: ∫ mu_r^-1 curl E · curl v;
- ``mass``, θ = -k0²: ∫ eps_r (1 - i tan_delta) E · v (per principal axis
  for an anisotropic material);
- ``conductivity``, θ = i k0: ∫ eta0 sigma scale E · v (sigma a 3x3
  tensor along its principal axes for an anisotropic material); left out
  where sigma is zero everywhere (`conductivity_piece`);
- ``absorbing``, θ = i k0: ∫ sqrt(eps_r / mu_r) E_t · v_t on every
  absorbing surface (real parts of the adjacent element's values); next to
  an anisotropic material, ∫ (Y E_t) · v_t with the face's normal-incidence
  admittance tensor Y (`gfem._engine.anisotropy`);
- ``absorbing order 2``, θ = 1 / (2 i k0): ∫ mu_r^-1 (eps_r mu_r)^-1/2
  (n · curl E)(n · curl v) on second-order absorbing surfaces (the
  textbook sign, see `gfem.Absorbing`);
- a conductor, θ = i k0 eta0 / Z_s(ω): ∫ E_t · v_t on its surface, with
  Z_s the surface impedance of a metal of finite thickness (thickness
  doubled on mesh-exterior faces, see `gfem.conventions`);
- an impedance boundary or a lumped port element, θ = i k0, 1, -k0²:
  ∫ E_t · v_t times eta0 / R_s, mu0 scale / L_s, C_s / (eps0 scale), its R,
  L and C per square. A `gfem.SheetField` on a conductor or an impedance
  boundary multiplies its pieces' coefficients face by face by the fill
  (`sheet_fill`, `Piece.sheet`): the design enters the form, never θ.

- a wave port, θ = gamma_0(ω) per file unit, the complex propagation constant
  of its mode: i beta(ω) with beta real on the driven solve (tabulated on
  the output grid by the port mode sweep, see `Tabulated`), and the port
  term's complex gamma_0 at the complex frequency of an eigenmode solve
  (`gfem._engine.port_term`): ∫ (A E_t) · v_t on the port surface, with A
  the port's tangential inverse-permeability tensor (``mu_r^-1`` for an
  isotropic material, `wave_port_coefficients`).

Each θ is a closed-form scalar function of ω, or a table over the output
grid filled before the solve, and each a_k is assembled without ω, so an
adaptive sweep can project the pieces once. A term that cannot be written
this way is not a `Piece`: constructing one raises.

Each piece declares its frequency class. Most θ are polynomials of degree
at most 2 in ω (``"polynomial"``); a conductor's θ (through Z_s, which
grows as the square root of ω) and the second-order absorbing term's (a
pole at ω = 0) are not (``"nonpolynomial"``). A nonpolynomial piece also
supplies dθ/dω, and both are analytic in ω, so they hold at the complex
frequencies of an eigenmode problem (an eigenmode seed linearizes θ
around its shift with both).

Each piece also carries what the sensitivity kernel differentiates
(`gfem._engine.sensitivity`): the derivative of its coefficient by each
material property it reads, written next to the coefficient, and for a
sheet boundary's piece its coefficient per unit fill.
"""

from __future__ import annotations

import cmath
import dataclasses
import functools
import math
from collections.abc import Callable, Mapping
from typing import Any, Literal, NamedTuple

import ngsolve as ngs
import numpy as np
import scipy.constants as const

from gfem._boundaries import (
    Absorbing,
    Conductor,
    Impedance,
    LumpedPort,
    check_lumped_values,
)
from gfem._engine import anisotropy, dispersion, quadrature
from gfem._engine.coefficients import (
    anisotropic_elements,
    coefficient,
    element_axes,
    element_tensors,
    element_values,
    per_axis_elements,
    principal_values,
    surface_coefficient,
    surface_p0,
)
from gfem._engine.domain import boundary_region
from gfem._engine.mesh import EngineMesh
from gfem._engine.sensitivity import Derivative
from gfem._engine.stretch import pml_tensor
from gfem._materials import Drude, MaterialField
from gfem._morph import unit_normals
from gfem._pml import Stretch
from gfem._sheet import check_fill
from gfem.conventions import EXTERIOR_THICKNESS_FACTOR
from gfem.errors import PortGeometryError
from gfem.mesh import Mesh
from gfem.model import Model

__all__ = [
    "ETA0",
    "LumpedElement",
    "LumpedPortData",
    "LumpedValue",
    "Piece",
    "Tabulated",
    "absorbing_coefficients",
    "absorbing_derivatives",
    "absorbing_faces",
    "absorbing_shape_unsupported",
    "blend_pieces",
    "check_absorbing",
    "excitation_per_voltage",
    "face_owner",
    "law_faces",
    "lumped_ports",
    "mass_derivatives",
    "pieces",
    "polynomial_law",
    "port_vectors",
    "sheet_fill",
    "surface_impedance",
    "surface_impedance_slope",
    "susceptibility_pieces",
    "wave_port_coefficients",
    "wave_port_piece",
    "wavenumber",
]

ETA0 = const.mu_0 * const.c
#: Angular frequency at which a piece's θ is checked to be a scalar.
_PROBE = 2 * math.pi * 1e9
#: The probe of a permittivity law's pieces: in the lower half plane, where
#: every passive law is analytic under e^{+iωt}, so it meets no
#: pole, no real output frequency and no decaying eigenfrequency.
_LAW_PROBE = _PROBE * (1 - 1j)
#: |d . n| above which a port direction is not tangent to its surface.
_TANGENT = 1e-6
#: Re(h / (2 delta)) above which a metal film is thick to machine precision.
_THICK = 25.0
#: |h / (2 delta)| below which the film slope uses its series.
_SMALL = 1e-4


def wavenumber(omega: float, scale: float) -> float:
    """k0 in radians per mesh file unit."""
    return omega * scale / const.c


class Tabulated:
    """A θ(ω) known on a grid of angular frequencies only.

    A pre-stage (the port mode sweep) fills it for every output frequency
    before the solve; the sweep never asks for another frequency.
    """

    def __init__(self, omegas: np.ndarray, values: np.ndarray) -> None:
        """``values[i]`` is θ at ``omegas[i]`` (rad/s)."""
        self.omegas = np.asarray(omegas, float)
        self.values = np.asarray(values)

    def __call__(self, omega: complex) -> complex:
        """θ at a grid ω (matched to 1e-12 relative).

        Raises:
            ValueError: ``omega`` is not on the grid.
        """
        k = int(np.argmin(np.abs(self.omegas - omega)))
        if abs(self.omegas[k] - omega) > 1e-12 * abs(omega):
            raise ValueError(
                f"A tabulated driven term has no value at {omega / (2 * math.pi):g} "
                "Hz: it is known on the output grid only."
            )
        return complex(self.values[k])


@dataclasses.dataclass(frozen=True)
class LumpedValue:
    """How a piece moves with one lumped value ``p``.

    The piece's coefficient is proportional to ``p`` (a capacitance) or to
    ``1 / p`` (a resistance or an inductance), so ``dA / dp = slope A``.

    Attributes:
        kind: The design input: ``("port", index, "inductance")`` or
            ``("boundary", index, "resistance")``, as `gfem._adjoint` names
            the leaf.
        slope: ``+1 / p`` or ``-1 / p``.
        faces: The piece's form as ``(region, c)`` per surface: the piece is
            ``sum int c u_t . v_t`` over them, ``c`` a number or a face-wise
            coefficient function.
    """

    kind: tuple
    slope: float
    faces: tuple[tuple[Any, Any], ...]


@dataclasses.dataclass(frozen=True)
class Piece:
    """One frequency-independent bilinear form and its scalar factor θ(ω).

    Attributes:
        name: What the piece is (``"mass"``, ``"port p1 R"``, ...).
        theta: θ(ω), a complex scalar for an angular frequency in rad/s.
        integrand: ``(trial, test) -> integrals`` of the form, with no ω.
        kind: How the form depends on the node coordinates, for shape
            gradients: ``"curl"`` (``∫ (C curl u) · curl v``), ``"gradient"``
            (``∫ (C grad u) · grad v``, a 2D model's scalar field), ``"value"``
            (``∫ (C u) · v``), ``"trace"`` (``∫ c u_t · v_t`` on surfaces),
            ``"curl trace"`` (``∫ c (n · curl u)(n · curl v)``),
            ``"tangential gradient"`` (``∫ c ∂_t u ∂_t v`` on a 2D model's
            lines), another ``"surface"`` form, or ``"port"`` (a port's
            term, which reads only the port's own geometry: shape gradients
            hold port nodes in place).
        coefficient: ``C`` (``c``) of a volume (trace) piece, the element-
            (face-) wise coefficient its integrand reads.
        values: The values of a volume piece's ``coefficient`` per gfem
            tetrahedron, ``(n,)`` or ``(n, 3, 3)`` (what a kept fixed part
            is keyed on).
        region: The boundary region of a ``"trace"`` or ``"curl trace"``
            piece.
        frequency_class: ``"polynomial"`` (θ a polynomial of degree at most
            2 in ω) or ``"nonpolynomial"``.
        d_theta: dθ/dω of a nonpolynomial piece (``None`` for a polynomial
            one).
        derivatives: The derivative of ``coefficient`` by each material
            property it reads (`gfem._engine.sensitivity.Derivative`, per
            tetrahedron, evaluated by a backward pass only).
        owner: For a face piece whose coefficient is read from the
            tetrahedron behind each face (absorbing), that tetrahedron per
            gfem triangle, -1 off the piece's faces.
        sheet: For a piece of a sheet boundary (`Conductor` or `Impedance`
            on an interior surface), ``(index, c)``: the boundary's index in
            ``model.boundaries`` and the coefficient per unit fill (1 for a
            conductor, the R, L or C weight for an impedance).
        lumped: For the L or C piece of a lumped port, or a piece of an
            `Impedance`, how it moves with its lumped value (`LumpedValue`);
            ``None`` otherwise (a lumped port's resistance is not a design
            input).
        probe: The angular frequency at which construction checks θ:
            `_PROBE`, or `_LAW_PROBE` off the real axis for a permittivity
            law's pieces. θ itself never raises and never returns a
            placeholder; which frequencies are valid is checked before
            assembly (`gfem._engine.dispersion.check_frequencies`).
        law: For a permittivity law's piece (its susceptibility piece and its
            absorbing face pieces), ``(region, d_theta_dp)``: the law's
            region and dθ/dp(ω) by each law parameter, keyed as
            `gfem._engine.dispersion.d_eps` (the law pass);
            ``None`` otherwise.
        points: For a piece of a layer grid's region of a 2D mesh, whose
            coefficient lives on the quadrature points
            (`gfem._engine.points`), that region: its derivatives are per
            point of the region, and the point pass reads them. ``None``
            otherwise.
        position: Whether ``coefficient`` also varies in position within an
            element: a perfectly matched layer's stretch composed into a
            volume piece (shape gradients read it at the moved quadrature
            points).

    Raises:
        ValueError: ``theta`` does not give one scalar: the term is not
            affine in frequency; or a nonpolynomial piece without
            ``d_theta``.
    """

    name: str
    theta: Callable[[complex], complex]
    integrand: Callable[[Any, Any], Any]
    kind: Literal[
        "curl",
        "gradient",
        "value",
        "trace",
        "curl trace",
        "tangential gradient",
        "surface",
        "port",
    ] = "surface"
    coefficient: Any = None
    region: Any = None
    frequency_class: Literal["polynomial", "nonpolynomial"] = "polynomial"
    d_theta: Callable[[complex], complex] | None = None
    derivatives: Mapping[str, Derivative] = dataclasses.field(
        default_factory=dict, compare=False
    )
    owner: np.ndarray | None = dataclasses.field(default=None, compare=False)
    sheet: tuple[int, float] | None = None
    values: np.ndarray | None = dataclasses.field(default=None, compare=False)
    lumped: LumpedValue | None = dataclasses.field(default=None, compare=False)
    position: bool = dataclasses.field(default=False, compare=False)
    probe: complex = _PROBE
    law: tuple[str, Callable[[complex], dict[tuple, complex]]] | None = (
        dataclasses.field(default=None, compare=False)
    )
    points: str | None = None

    def __post_init__(self) -> None:
        """Check that θ is a scalar function of ω."""
        if isinstance(self.theta, Tabulated):
            if self.theta.values.ndim != 1 or not np.isfinite(self.theta.values).all():
                raise ValueError(
                    f"The driven term {self.name!r} is not tabulated as one finite "
                    "scalar per frequency."
                )
            return
        value = np.asarray(self.theta(self.probe))
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError(
                f"The driven term {self.name!r} is not affine in frequency (a "
                "frequency-independent form times a scalar function of the "
                "frequency), so gfem cannot sweep it."
            )
        if self.frequency_class not in ("polynomial", "nonpolynomial"):
            raise ValueError(f"Unknown frequency class {self.frequency_class!r}.")
        if self.frequency_class == "nonpolynomial" and self.d_theta is None:
            raise ValueError(f"The nonpolynomial term {self.name!r} needs d_theta.")


def _constant(omega: complex) -> complex:  # noqa: ARG001
    """θ = 1: a frequency-independent piece."""
    return 1.0


def surface_impedance(sigma: float, thickness: float | None, omega: complex) -> complex:
    """Surface impedance of a metal, ohm per square (the HFSS form).

    Thick metal: ``Z_s = (1 + i) / (sigma delta)`` with skin depth
    ``delta = sqrt(2 / (mu0 sigma omega))``. A film of thickness ``h``
    multiplies the real and imaginary parts by ``(sinh v ± sin v) /
    (cosh v - cos v)``, ``v = h / delta``. For a thin film it tends to
    ``2 / (sigma h)``: each face of the film carries half its current.

    Both are written as one analytic function of ω, ``Z_s = (g / sigma)
    coth(g h / 2)`` with ``g = sqrt(i omega mu0 sigma) = (1 + i) / delta``
    (principal root), so a complex ω (an eigenfrequency) continues them.

    Args:
        sigma: Conductivity, S/m.
        thickness: Thickness in m, or ``None`` for thick metal.
        omega: Angular frequency, rad/s (complex for an eigenmode).
    """
    g = cmath.sqrt(1j * omega * const.mu_0 * sigma)
    z = g / sigma
    if thickness is not None:
        u = 0.5 * g * thickness
        if u.real < _THICK:  # beyond, coth u is 1 to machine precision
            z /= cmath.tanh(u)
    return complex(z)


def surface_impedance_slope(
    sigma: float, thickness: float | None, omega: complex
) -> complex:
    """``(omega / Z_s) dZ_s/domega``: 1/2 for thick metal, 0 for a thin film.

    With ``u = g h / 2`` (see `surface_impedance`), ``(1 - 2 u / sinh 2u)
    / 2``.
    """
    if thickness is None:
        return 0.5
    g = cmath.sqrt(1j * omega * const.mu_0 * sigma)
    u = 0.5 * g * thickness
    if u.real >= _THICK:
        return 0.5
    if abs(u) < _SMALL:  # 2u / sinh 2u = 1 - (2u)^2 / 6 + ...
        return complex((2 * u) ** 2 / 12)
    return complex(0.5 * (1 - 2 * u / cmath.sinh(2 * u)))


# --- lumped ports ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LumpedElement:
    """One surface of a lumped port, in mesh file units.

    Attributes:
        surface: The surface name.
        direction: Unit excitation direction.
        length: Extent of the surface along ``direction``.
        width: Area over length.
    """

    surface: str
    direction: tuple[float, float, float]
    length: float
    width: float


@dataclasses.dataclass(frozen=True)
class LumpedPortData:
    """A lumped port's termination and elements (elements in parallel)."""

    port: LumpedPort
    elements: tuple[LumpedElement, ...]

    def per_square(self, element: LumpedElement) -> tuple[Any, Any, Any]:
        """R_s, L_s, C_s of one element (``None`` where the port has none).

        The port's elements are in parallel, each a sheet of ``width``
        across and ``length`` along the excitation.
        """
        r, ind, cap = self.port.rlc
        ratio = element.width * len(self.elements) / element.length
        return (
            None if r is None else r * ratio,
            None if ind is None else ind * ratio,
            None if cap is None else cap / ratio,
        )


def lumped_ports(model: Model, scale: float) -> tuple[LumpedPortData, ...]:
    """Element geometry of every lumped port, at the current node positions.

    Raises:
        PortGeometryError: An element's direction is not tangent to its
            surface.
    """
    mesh = model.mesh
    nodes = np.asarray(mesh.nodes, dtype=float) / scale
    out = []
    for port in model.ports:
        if not isinstance(port, LumpedPort):
            continue
        elements = []
        for surface, direction in port.elements:
            tris = mesh.triangles[mesh.triangle_tags == mesh.region(surface).tag]
            p = nodes[tris]
            cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            twice_area = np.linalg.norm(cross, axis=1)
            d = np.asarray(direction)
            off = np.abs(cross @ d) / twice_area
            if (off > _TANGENT).any():
                raise PortGeometryError(
                    f"LumpedPort {port.name!r}: direction {direction} is not "
                    f"tangent to its surface {surface!r} (|d . n| up to "
                    f"{off.max():.3g}). Give a direction along the surface, from "
                    "one conductor to the other."
                )
            along = nodes[np.unique(tris)] @ d
            length = float(along.max() - along.min())
            width = 0.5 * float(twice_area.sum()) / length
            elements.append(LumpedElement(surface, direction, length, width))
        out.append(LumpedPortData(port, tuple(elements)))
    return tuple(out)


# --- the pieces ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Tangential:
    """``∫ weight E_t · v_t`` on a boundary region, on the pinned rule.

    ``faces`` restricts it to those NGSolve surface elements (``None``:
    the whole region).
    """

    region: Any
    order: int
    faces: Any = None

    def __call__(self, u: Any, v: Any, weight: Any = 1.0) -> Any:
        ds = quadrature.ds(
            self.order, definedon=self.region, definedonelements=self.faces
        )
        return weight * u.Trace() * v.Trace() * ds


def _tangential(
    built: EngineMesh, surfaces: tuple[str, ...], order: int, faces: Any = None
) -> _Tangential:
    return _Tangential(boundary_region(built, surfaces), order, faces)


def _bits(
    mask: np.ndarray | None, order: np.ndarray, what: str, *, part: bool = False
) -> Any:
    """The NGSolve element bits of a mask over gfem ``what`` (``None``: all).

    NGSolve element ``i`` is gfem element ``order[i]``. With ``part`` the
    NGSolve mesh is a rank's part, ``order`` covers its elements only, and
    the mask is still over the whole gfem mesh.

    Raises:
        ValueError: The mask is not one boolean per gfem element.
    """
    if mask is None:
        return None
    mask = np.asarray(mask)
    if part:
        fits = mask.ndim == 1 and (len(order) == 0 or int(order.max()) < len(mask))
    else:
        fits = mask.shape == order.shape
    if mask.dtype != bool or not fits:
        count = "the whole mesh's" if part else str(len(order))
        raise ValueError(
            f"A driven piece restriction needs one boolean per {what} "
            f"({count}), got an array of {mask.dtype} of shape {mask.shape}."
        )
    return ngs.BitArray(mask[order].tolist())


def _inverse(values: np.ndarray) -> np.ndarray:
    return 1.0 / values if values.ndim == 1 else np.linalg.inv(values)


def polynomial_law(law: Any) -> bool:
    """Whether every term of ``law`` is a lossless Drude term.

    Its susceptibility piece's θ = -k0² chi = k_p² is then a constant: the
    piece is declared polynomial, so the eigenmode problem puts it in ``K``
    (and a lossless model keeps its real snap). Any damping, or any other
    term, keeps the piece nonpolynomial.
    """
    return all(
        isinstance(term, Drude) and float(term.damping) == 0 for term in law.terms
    )


def susceptibility_pieces(
    model: Model,
    built: EngineMesh,
    order: int,
    *,
    elements: np.ndarray | None = None,
) -> list[Piece]:
    """One susceptibility piece per permittivity law region.

    θ = -k0² chi(ω), chi = eps(ω) - eps_inf summed over the law's terms,
    with an analytic dθ/dω; the coefficient is the region's element
    indicator (its eps_inf is in the mass). The piece is probed at
    `_LAW_PROBE`. ``elements`` restricts the form as in `pieces`. A law of
    lossless Drude terms only gives a polynomial piece (`polynomial_law`).
    """
    scale = built.scale
    tets = _bits(elements, built.cell_order, "tetrahedron", part=built.comm is not None)
    dx = quadrature.dx(order, definedonelements=tets)

    def k2(w: complex) -> complex:
        """k0² at a real or complex ω (the construction probe is complex)."""
        return (w * scale / const.c) ** 2

    out = []
    for lr in dispersion.law_regions(model):
        values = lr.mask.astype(float)
        cf = coefficient(built, values)
        law = lr.law

        def theta(w: complex, law: Any = law) -> complex:
            return -k2(w) * dispersion.susceptibility(law, w)

        def d_theta(w: complex, law: Any = law) -> complex:
            # d/dω [-k² chi] = -(2 k² / ω) chi - k² chi', k = ω scale / c
            return -k2(w) * (
                2 / w * dispersion.susceptibility(law, w)
                + dispersion.d_susceptibility(law, w)
            )

        def by_parameter(w: complex, law: Any = law) -> dict[tuple, complex]:
            # dθ/dp = -k² d eps/dp: eps_inf's mass term (θ -k², coefficient
            # eps_inf on this region) is this piece's form too
            return {p: -k2(w) * d for p, d in dispersion.d_eps(law, w).items()}

        out.append(
            Piece(
                f"susceptibility {lr.region}",
                theta,
                lambda u, v, cf=cf: (cf * u) * v * dx,
                kind="value",
                coefficient=cf,
                values=values,
                frequency_class=(
                    "polynomial" if polynomial_law(law) else "nonpolynomial"
                ),
                d_theta=d_theta,
                probe=_LAW_PROBE,
                law=(lr.region, by_parameter),
            )
        )
    return out


def blend_pieces(
    model: Model,
    built: EngineMesh,
    order: int,
    *,
    elements: np.ndarray | None = None,
) -> list[Piece]:
    """The susceptibility pieces of each law blend.

    One piece per part of `gfem._engine.dispersion.blend_parts` (at most a
    void, a solid and a cross piece per region): θ = -k0² X(ω) with an
    analytic dθ/dω, and the coefficient c(rho) per element, whose derivative
    by the density is the material pass's ``"blend"`` property. The eps_inf
    limit is in the mass. ``elements`` restricts the form as in `pieces`.
    """
    scale = built.scale
    tets = _bits(elements, built.cell_order, "tetrahedron", part=built.comm is not None)
    dx = quadrature.dx(order, definedonelements=tets)

    def k2(w: complex) -> complex:
        return (w * scale / const.c) ** 2

    out = []
    for br in dispersion.blend_regions(model):
        for part in dispersion.blend_parts(br):
            cf = coefficient(built, part.c)

            def theta(w: complex, part: Any = part) -> complex:
                return -k2(w) * part.chi(w)

            def d_theta(w: complex, part: Any = part) -> complex:
                return -k2(w) * (2 / w * part.chi(w) + part.d_chi(w))

            out.append(
                Piece(
                    f"blend {br.region} {part.name}",
                    theta,
                    lambda u, v, cf=cf: (cf * u) * v * dx,
                    kind="value",
                    coefficient=cf,
                    values=part.c,
                    frequency_class="nonpolynomial",
                    d_theta=d_theta,
                    derivatives={
                        "blend": Derivative(lambda dc=part.dc: dc, holomorphic=True)
                    },
                    probe=_LAW_PROBE,
                )
            )
    return out


def mass_derivatives(model: Model) -> dict[str, Derivative]:
    r"""The derivatives of the mass coefficient :math:`\varepsilon_r(1 - i\tan\delta)`.

    By ``eps_r``, ``tan_delta`` and a law blend's density (``blend``, whose
    eps_inf limit is in the mass), per element, with their per-axis forms.
    """
    return {
        "eps_r": Derivative(
            lambda: _loss(model),
            holomorphic=True,
            per_axis=Derivative(lambda: _axis_loss(model), holomorphic=True),
        ),
        # a law blend's eps_inf limit moves with its density
        "blend": Derivative(lambda: _blend_mass_slope(model), holomorphic=True),
        "tan_delta": Derivative(
            # a law region has no tan_delta (nor a field): never read
            lambda: -1j * element_values(model, "eps_r", per_axis=1.0, law=1.0),
            holomorphic=False,
            per_axis=Derivative(
                lambda: -1j * principal_values(model, "eps_r"),
                holomorphic=False,
            ),
        ),
    }


def _blend_mass_slope(model: Model) -> np.ndarray:
    """D (mass value) / d rho per tetrahedron: the law blends' eps_inf limits."""
    out = np.zeros(len(model.mesh.volume_elements))
    for br in dispersion.blend_regions(model):
        out += dispersion.blend_mass(br)[1]
    return out


def conductivity_piece(
    model: Model,
    built: EngineMesh,
    order: int,
    *,
    elements: np.ndarray | None = None,
    stretch: Stretch | None = None,
) -> Piece:
    """The ``conductivity`` piece of ``model``: θ = i k0, ∫ eta0 sigma scale E · v.

    `pieces` returns it as latent when the conductivity is zero everywhere
    on the solve domain (a zero matrix, and no damping term for the eigenmode
    operator). Its derivative by the conductivity is not zero there, so a
    backward pass still reads the piece (`gfem._engine.driven.DrivenOperator.latent`).
    ``elements`` and ``stretch`` act as in `pieces`.
    """
    scale = built.scale
    k = lambda w: wavenumber(w, scale)  # noqa: E731
    tets = _bits(elements, built.cell_order, "tetrahedron", part=built.comm is not None)
    dx = quadrature.dx(order, definedonelements=tets)
    active = model._domain.active  # noqa: SLF001
    # an anisotropic material gives its 3x3 tensor
    sigma = element_tensors(model, "conductivity").real
    sigma = np.where(active.reshape((-1,) + (1,) * (sigma.ndim - 1)), sigma, 0.0)
    # eta0 sigma scale (real symmetric sigma); its derivative eta0 leaves
    # out the length scale
    values = ETA0 * scale * sigma
    cf = coefficient(built, values)
    if stretch is not None:
        cf = cf * pml_tensor(stretch, scale)[0]
    return Piece(
        "conductivity",
        lambda w: 1j * k(w),
        lambda u, v: (cf * u) * v * dx,
        kind="value",
        coefficient=cf,
        values=values,
        position=stretch is not None,
        derivatives={
            "conductivity": Derivative(
                ETA0,
                holomorphic=False,
                length_power=1,
                per_axis=Derivative(ETA0, holomorphic=False, length_power=1),
            )
        },
    )


def pieces(
    model: Model,
    built: EngineMesh,
    order: int,
    ports: tuple[LumpedPortData, ...],
    *,
    elements: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    stretch: Stretch | None = None,
) -> tuple[list[Piece], tuple[Piece, ...]]:
    r"""The driven operator's pieces on ``model`` (see the module docstring).

    Returns the pieces and the latent ones: pieces whose matrix is zero at
    this design but whose derivative is not (the conductivity piece where the
    conductivity is zero everywhere; `gfem._engine.driven.DrivenOperator.latent`).

    `PEC` is a constraint of the space and `PMC` the natural condition, so
    neither has a piece.

    ``elements`` and ``faces`` restrict the pieces to part of the mesh:
    boolean masks over the gfem tetrahedra and triangles. Every volume term
    is integrated on the selected tetrahedra only, every surface term
    (absorbing, conductor, impedance and lumped port) on the selected
    triangles only; ``None`` selects them all. The pieces of a partition of
    the mesh sum to the unrestricted pieces. θ and the coefficients do not
    change.

    A perfectly matched layer's ``stretch`` composes its PML tensor
    :math:`\Lambda`, a coefficient of position, into the volume pieces: the
    curl-curl coefficient becomes :math:`\mu_r^{-1}\Lambda^{-1}`, the mass
    :math:`\varepsilon_r\Lambda` and the conductivity :math:`\sigma\Lambda`
    (:math:`\Lambda` is the identity outside the layer). :math:`\theta`,
    ``Piece.values`` and the derivatives keep the background material.

    Raises:
        ValueError: A mask is not one boolean per tetrahedron (triangle).
    """
    scale = built.scale
    k = lambda w: wavenumber(w, scale)  # noqa: E731
    tets = _bits(elements, built.cell_order, "tetrahedron", part=built.comm is not None)
    tris = _bits(faces, built.facet_order, "triangle", part=built.comm is not None)
    dx = quadrature.dx(order, definedonelements=tets)

    # each piece carries its coefficient and, next to it, the coefficient's
    # derivatives by material property (lazy: a backward pass reads them)
    # a permittivity law keeps its eps_inf in the mass
    eps = dispersion.high_frequency_eps(model)
    nu_values = _inverse(element_tensors(model, "mu_r"))
    nu = coefficient(built, nu_values)
    eps_cf = coefficient(built, eps)
    if stretch is not None:
        lam, lam_inv = pml_tensor(stretch, scale)
        nu, eps_cf = nu * lam_inv, eps_cf * lam
    out = [
        Piece(
            "curl_curl",
            _constant,
            lambda u, v: (nu * ngs.curl(u)) * ngs.curl(v) * dx,
            kind="curl",
            coefficient=nu,
            values=nu_values,
            position=stretch is not None,
            derivatives={
                "mu_r": Derivative(
                    lambda: -(element_values(model, "mu_r", per_axis=1.0) ** -2.0),
                    holomorphic=True,
                    per_axis=Derivative(
                        lambda: -(principal_values(model, "mu_r") ** -2.0),
                        holomorphic=True,
                    ),
                )
            },
        ),
        Piece(
            "mass",
            lambda w: -(k(w) ** 2),
            lambda u, v: (eps_cf * u) * v * dx,
            kind="value",
            coefficient=eps_cf,
            values=eps,
            position=stretch is not None,
            derivatives=mass_derivatives(model),
        ),
    ]
    # the conductivity has its own i k0 piece, not folded into the mass (the
    # eigen operator splits the pieces by power of k)
    out.extend(susceptibility_pieces(model, built, order, elements=elements))
    out.extend(blend_pieces(model, built, order, elements=elements))
    conductivity = conductivity_piece(
        model, built, order, elements=elements, stretch=stretch
    )
    latent: tuple[Piece, ...] = ()
    if np.any(conductivity.values):
        out.append(conductivity)
    else:
        latent = (conductivity,)

    absorbing = [b for b in model.boundaries if isinstance(b, Absorbing)]
    if absorbing:
        check_absorbing(model, dispersive=True)
        _refuse_absorbing_blends(model)
        surfaces, owner = absorbing_faces(model, order=1)
        admittance, curl = absorbing_coefficients(model, surfaces, owner)
        # the faces a permittivity law owns leave the shared pieces
        # (coefficient and owner) for their law's face pieces
        laws = law_faces(model, surfaces, owner)
        on_law = np.zeros(len(owner), bool)
        for _, lf in laws:
            on_law |= lf
        form = _tangential(built, surfaces, order, tris)
        y_cf = surface_coefficient(built, admittance)
        out.append(
            Piece(
                "absorbing",
                lambda w: 1j * k(w),
                lambda u, v: form(u, v, y_cf),
                kind="trace",
                coefficient=y_cf,
                region=form.region,
                derivatives=absorbing_derivatives(
                    model, order=1, faces=(surfaces, owner)
                ),
                owner=np.where(on_law, -1, _owned(model, surfaces, owner, faces)),
            )
        )
        second = tuple(b.surface for b in absorbing if b.order == 2)
        if second:
            curl_cf = surface_p0(built, curl)
            region = boundary_region(built, second)
            ds = quadrature.ds(order, definedon=region, definedonelements=tris)
            out.append(
                Piece(
                    "absorbing order 2",
                    lambda w: 1 / (2j * k(w)),
                    lambda u, v: (
                        curl_cf * ngs.curl(u).Trace() * ngs.curl(v).Trace() * ds
                    ),
                    kind="curl trace",
                    coefficient=curl_cf,
                    region=region,
                    frequency_class="nonpolynomial",
                    d_theta=lambda w: -1 / (2j * k(w) * w),
                    derivatives=absorbing_derivatives(model, order=2),
                    owner=np.where(on_law, -1, _owned(model, second, owner, faces)),
                )
            )
        if laws:
            out.extend(
                _law_absorbing_pieces(
                    model,
                    built,
                    order,
                    laws=laws,
                    absorbing=(surfaces, second, owner),
                    form=form,
                    tris=tris,
                    faces=faces,
                )
            )

    for index, b in enumerate(model.boundaries):
        if isinstance(b, Conductor):
            out.extend(_conductor(model, built, order, b, tris, index=index))
        elif isinstance(b, Impedance):
            out.extend(_impedance(model, built, order, b, tris, index=index))

    for data in ports:
        index = model.port_names.index(data.port.name)
        out.extend(_port(built, order, data, tris, index=index))
    return out, latent


def absorbing_coefficients(
    model: Model, surfaces: tuple[str, ...], owner: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle coefficients of the absorbing terms.

    `_absorbing` of the real parts of the solve-domain element behind each
    triangle (``owner``). A face of ``surfaces`` with an anisotropic element
    behind it (`anisotropic_elements`) takes the admittance tensor
    `anisotropy.admittance` instead: the first coefficient is then ``(n,
    3, 3)``, holding ``sqrt(e / m)`` times the identity on the other faces.

    Raises:
        ValueError: The admittance of an anisotropic face is undefined.
    """
    # a field with axes holds no one value per element: its faces take the
    # tensor path below (`anisotropic_elements`), so the placeholder 1 there
    # reaches no absorbing face
    # a face behind a permittivity law takes its law's face pieces, so the
    # placeholder 1 there is zeroed below
    tan = np.real(element_values(model, "tan_delta", per_axis=1.0))
    e = np.real(
        element_values(model, "eps_r", per_axis=1.0, law=1.0) * (1.0 - 1j * tan)
    )
    m = np.real(element_values(model, "mu_r", per_axis=1.0))
    values = np.stack([_absorbing(e, m, 1), _absorbing(e, m, 2)], axis=1)
    out = np.zeros((len(owner), 2))
    ok = owner >= 0
    out[ok] = values[owner[ok]]
    first, curl = out[:, 0], out[:, 1]
    # a law's faces take its face pieces (`_law_absorbing_pieces`)
    for _, lf in law_faces(model, surfaces, owner):
        first[lf] = 0.0
        curl[lf] = 0.0
    mesh = model.mesh
    tensor = ok & anisotropic_elements(model)[np.maximum(owner, 0)]
    tensor &= np.isin(mesh.triangle_tags, [mesh.region(s).tag for s in surfaces])
    if not tensor.any():
        return first, curl
    eps = _tensors(np.real(dispersion.high_frequency_eps(model)))
    mu = _tensors(np.real(element_tensors(model, "mu_r")))
    normals = _face_normals(mesh)
    y = first[:, None, None] * np.eye(3)
    for surface in surfaces:
        on = tensor & (mesh.triangle_tags == mesh.region(surface).tag)
        if on.any():
            y[on] = anisotropy.admittance(
                eps[owner[on]],
                mu[owner[on]],
                normals[on],
                where=f"Absorbing {surface!r}",
            )
    return y, curl


def _face_normals(mesh: Mesh) -> np.ndarray:
    """Unit normals of every gfem triangle, ``(n, 3)``: the faces' frames."""
    return unit_normals(np.asarray(mesh.nodes, float), mesh.triangles)


def _tensors(values: np.ndarray) -> np.ndarray:
    """Per-element values as ``(n, 3, 3)`` tensors."""
    return values[:, None, None] * np.eye(3) if values.ndim == 1 else values


def _anisotropic_faces(model: Model, surface: str) -> bool:
    """Whether an anisotropic element is behind a face of ``surface``."""
    mesh = model.mesh
    owner = face_owner(model)[mesh.triangle_tags == mesh.region(surface).tag]
    return bool(anisotropic_elements(model)[owner[owner >= 0]].any())


def check_absorbing(model: Model, *, dispersive: bool = False) -> None:
    """Raise for an absorbing boundary this version cannot apply.

    Structural: it reads which materials are anisotropic or dispersive,
    never values. ``dispersive`` admits a face behind a permittivity law's
    region: the driven problem's face pieces read sqrt(Re eps(omega)), which
    is not analytic in omega, so the eigenmode problem refuses it.
    Whether Re eps > 0 there is checked per frequency
    (`gfem._engine.dispersion.check_frequencies`).

    Raises:
        NotImplementedError: A second-order `Absorbing` surface has an
            anisotropic element behind a face (`anisotropic_elements`);
            without ``dispersive``, an `Absorbing` surface has an element of
            a permittivity law's region behind a face (an absorbing boundary
            on a dispersive region).
    """
    if not dispersive:
        for region, surfaces in dispersion.absorbing_surfaces(model).items():
            raise NotImplementedError(
                f"Absorbing {surfaces[0]!r} lies on region {region!r}, whose "
                "eps_r is a permittivity law: an absorbing boundary on a "
                "dispersive region is not supported by the eigenmode problem "
                "(its admittance sqrt(Re eps(omega)) is not analytic in "
                "omega). Put a non-dispersive layer between the region and the "
                "boundary, or solve a gfem.Driven sweep."
            )
    for b in model.boundaries:
        second = isinstance(b, Absorbing) and b.order == 2
        if second and _anisotropic_faces(model, b.surface):
            raise NotImplementedError(
                f"Absorbing {b.surface!r}: order 2 needs an isotropic material "
                "behind the surface; an anisotropic eps_r / mu_r touches it. "
                "Use order=1, or put an isotropic layer between the material "
                "and the boundary."
            )


class _Masked(NamedTuple):
    """A region and its elements, as `dispersion.absorbing_surfaces` reads them."""

    region: str
    mask: np.ndarray


def _refuse_absorbing_blends(model: Model) -> None:
    """Raise for an `Absorbing` face on a law blend's region.

    Raises:
        NotImplementedError: An absorbing surface has a face on a blend.
    """
    blends = {
        r: m.blend_ends.pixels is not None
        for r, m in model.materials.items()
        if isinstance(m, MaterialField) and m.blend_ends is not None
    }
    regions = []
    for region in blends:
        mask = np.zeros(len(model.mesh.volume_elements), bool)
        mask[model.mesh.elements(region)] = True
        regions.append(_Masked(region, mask))
    for region, surfaces in dispersion.absorbing_surfaces(model, regions).items():  # ty: ignore[invalid-argument-type]
        what = (
            "a layer-grid design"
            if blends[region]
            else "a density blend with a permittivity-law end"
        )
        raise NotImplementedError(
            f"Absorbing {surfaces[0]!r} lies on region {region!r}, {what}: an "
            "absorbing boundary on a blend is not supported. Put a constant layer "
            "between the design region and the boundary."
        )


def law_faces(
    model: Model, surfaces: tuple[str, ...], owner: np.ndarray
) -> list[tuple[dispersion.LawRegion, np.ndarray]]:
    """The faces of ``surfaces`` behind each permittivity law region.

    One boolean per gfem triangle for each law region with such a face
    (``owner`` per triangle, as `absorbing_faces` gives it).
    """
    laws = dispersion.law_regions(model)
    if not laws or not surfaces:
        return []
    mesh = model.mesh
    tags = mesh._topology.surface_tags  # noqa: SLF001 - either dimension's faces
    on = np.isin(tags, [mesh.region(s).tag for s in surfaces])
    on &= owner >= 0
    out = []
    for lr in laws:
        lf = on & lr.mask[np.maximum(owner, 0)]
        if lf.any():
            out.append((lr, lf))
    return out


def _law_absorbing_pieces(
    model: Model,
    built: EngineMesh,
    order: int,
    *,
    laws: list[tuple[dispersion.LawRegion, np.ndarray]],
    absorbing: tuple[tuple[str, ...], tuple[str, ...], np.ndarray],
    form: Any,
    tris: Any,
    faces: np.ndarray | None,
) -> list[Piece]:
    """The absorbing face pieces of each permittivity law.

    On the faces a law region owns (``laws``, from `law_faces`): order 1
    θ = i k0 sqrt(Re eps(ω)) with coefficient mu_r^-1/2, and on the
    second-order surfaces θ = 1 / (2 i k0 sqrt(Re eps(ω))) with
    coefficient mu_r^-3/2: the shared pieces' sqrt(e / m) and
    1 / (m sqrt(e m)) with e read at ω. Re makes θ non-analytic, so these
    serve the driven problem only (`check_absorbing`); ``d_theta`` is the
    slope along real ω. Each is its own piece, so a law-parameter weight
    can take its key. Probed at `_LAW_PROBE`; Re eps > 0 at the output
    frequencies is `gfem._engine.dispersion.check_frequencies`' check.
    ``absorbing`` is ``(surfaces, second-order surfaces, owner)``.
    """
    surfaces, second, owner = absorbing
    scale = built.scale
    m = np.real(element_values(model, "mu_r", per_axis=1.0))[np.maximum(owner, 0)]
    first_owner = _owned(model, surfaces, owner, faces)
    mesh = model.mesh
    on_second = np.isin(mesh.triangle_tags, [mesh.region(s).tag for s in second])
    if second:
        region = boundary_region(built, second)
        ds = quadrature.ds(order, definedon=region, definedonelements=tris)
        second_owner = _owned(model, second, owner, faces)

    def k(w: complex) -> complex:
        return w * scale / const.c

    out = []
    for lr, lf in laws:
        law = lr.law

        def eps(w: complex, law: Any = law) -> float:
            """The real part of eps at ω (of its continuation off the real axis)."""
            return float(law.eps_inf) + dispersion.susceptibility(law, w).real

        def theta(w: complex, eps: Any = eps) -> complex:
            return 1j * k(w) * cmath.sqrt(eps(w))

        def d_theta(w: complex, eps: Any = eps, law: Any = law) -> complex:
            # d/dω [i k sqrt(e)] = i k (sqrt(e) / ω + e' / (2 sqrt(e)))
            e = cmath.sqrt(eps(w))
            slope = dispersion.d_susceptibility(law, w).real
            return 1j * k(w) * (e / w + slope / (2 * e))

        def by_parameter(
            w: complex, eps: Any = eps, law: Any = law
        ) -> dict[tuple, complex]:
            # d/dp [i k sqrt(Re eps)] = i k Re(d eps/dp) / (2 sqrt(Re eps))
            scale = 1j * k(w) / (2 * cmath.sqrt(eps(w)))
            return {p: scale * d.real for p, d in dispersion.d_eps(law, w).items()}

        y = np.where(lf, m**-0.5, 0.0)
        y_cf = surface_coefficient(built, y)
        out.append(
            Piece(
                f"absorbing {lr.region}",
                theta,
                lambda u, v, y_cf=y_cf: form(u, v, y_cf),
                kind="trace",
                coefficient=y_cf,
                region=form.region,
                frequency_class="nonpolynomial",
                d_theta=d_theta,
                owner=np.where(lf, first_owner, -1),
                probe=_LAW_PROBE,
                law=(lr.region, by_parameter),
            )
        )
        curl = lf & on_second
        if not curl.any():
            continue

        def theta2(w: complex, eps: Any = eps) -> complex:
            return 1 / (2j * k(w) * cmath.sqrt(eps(w)))

        def d_theta2(w: complex, eps: Any = eps, law: Any = law) -> complex:
            # d/dω [1 / (2 i k sqrt(e))] = -θ2 (1 / ω + e' / (2 e))
            slope = dispersion.d_susceptibility(law, w).real
            return -theta2(w) * (1 / w + slope / (2 * eps(w)))

        def by_parameter2(
            w: complex, eps: Any = eps, law: Any = law
        ) -> dict[tuple, complex]:
            # d/dp [1 / (2 i k sqrt(Re eps))] = -θ2 Re(d eps/dp) / (2 Re eps)
            scale = -theta2(w) / (2 * eps(w))
            return {p: scale * d.real for p, d in dispersion.d_eps(law, w).items()}

        c_cf = surface_p0(built, np.where(curl, m**-1.5, 0.0))
        out.append(
            Piece(
                f"absorbing order 2 {lr.region}",
                theta2,
                lambda u, v, c_cf=c_cf: (
                    c_cf * ngs.curl(u).Trace() * ngs.curl(v).Trace() * ds
                ),
                kind="curl trace",
                coefficient=c_cf,
                region=region,
                frequency_class="nonpolynomial",
                d_theta=d_theta2,
                owner=np.where(curl, second_owner, -1),
                probe=_LAW_PROBE,
                law=(lr.region, by_parameter2),
            )
        )
    return out


def absorbing_shape_unsupported(model: Model) -> str | None:
    """Why shape gradients of ``model`` are unavailable here, or None.

    The node pass transports a scalar absorbing coefficient only: an
    `Absorbing` surface with an anisotropic element behind a face (a tensor
    admittance) has no shape derivative in this version.
    """
    for b in model.boundaries:
        if isinstance(b, Absorbing) and _anisotropic_faces(model, b.surface):
            return (
                f"the Absorbing surface {b.surface!r} has an anisotropic material "
                "behind it, and the shape derivative of its tensor admittance is "
                "not available in this version"
            )
    return None


# The absorbing coefficients of the real e and m of the adjacent material,
# and their derivatives: first order c1 = sqrt(e / m), second order
# c2 = 1 / (m sqrt(e m)) (mu_r^-1 times 1 / sqrt(e m), the 1/k of the
# adjacent material relative to k0).
def _absorbing(e: np.ndarray, m: np.ndarray, order: int) -> np.ndarray:
    return np.sqrt(e / m) if order == 1 else 1 / (m * np.sqrt(e * m))


def _absorbing_by_e(e: np.ndarray, m: np.ndarray, order: int) -> np.ndarray:
    return 0.5 / np.sqrt(e * m) if order == 1 else -0.5 * (e * m) ** -1.5


def _absorbing_by_m(e: np.ndarray, m: np.ndarray, order: int) -> np.ndarray:
    if order == 1:
        return -0.5 * np.sqrt(e) * m**-1.5
    return -1.5 * e**-0.5 * m**-2.5


# The derivatives read the elements of fields with axes through their
# per-axis forms (`principal_values`); the isotropic forms take a placeholder
# there (`element_values(..., per_axis=1.0)`), never read for those elements.
def _loss(model: Model) -> np.ndarray:
    """``1 - i tan_delta`` per tetrahedron: the mass coefficient per unit eps_r."""
    return 1.0 - 1j * np.real(element_values(model, "tan_delta", per_axis=1.0))


def _axis_loss(model: Model) -> np.ndarray:
    """``1 - i tan_delta_k`` per tetrahedron and principal axis, ``(n, 3)``."""
    return 1.0 - 1j * np.real(principal_values(model, "tan_delta"))


def absorbing_derivatives(
    model: Model,
    order: int,
    faces: tuple[tuple[str, ...], np.ndarray] | None = None,
    *,
    swapped: bool = False,
) -> dict[str, Derivative]:
    """The derivatives of the absorbing coefficient of ``order`` per element.

    Of the real ``e = Re(eps_r (1 - i tan_delta))`` and ``m = Re(mu_r)`` of
    the isotropic element values: ``de/deps_r`` is 1 along ``Re(chain
    eps_r)`` with ``chain = 1 - i tan_delta``, and ``de/dtan_delta =
    Im(eps_r)``. With ``faces`` (the surfaces and the owner per triangle of
    the first-order term), each also has its per-axis form, per face
    (`_absorbing_axis_derivatives`). ``swapped`` exchanges the roles of
    ``e`` and ``m`` in the coefficient: a 2D model's Hz coefficients
    ``sqrt(m / e)`` and ``1 / (e sqrt(e m))``.
    """

    def em() -> tuple[np.ndarray, np.ndarray]:
        e = np.real(
            element_values(model, "eps_r", per_axis=1.0, law=1.0) * _loss(model)
        )
        return e, np.real(element_values(model, "mu_r", per_axis=1.0))

    def by_e() -> np.ndarray:
        e, m = em()
        return _absorbing_by_m(m, e, order) if swapped else _absorbing_by_e(e, m, order)

    def by_m() -> np.ndarray:
        e, m = em()
        return _absorbing_by_e(m, e, order) if swapped else _absorbing_by_m(e, m, order)

    axis = {} if faces is None else _absorbing_axis_derivatives(model, *faces)
    return {
        "eps_r": Derivative(
            by_e,
            holomorphic=False,
            chain=lambda: _loss(model),
            per_axis=axis.get("eps_r"),
        ),
        "tan_delta": Derivative(
            lambda: (
                by_e() * np.imag(element_values(model, "eps_r", per_axis=1.0, law=1.0))
            ),
            holomorphic=False,
            per_axis=axis.get("tan_delta"),
        ),
        "mu_r": Derivative(by_m, holomorphic=False, per_axis=axis.get("mu_r")),
    }


def _absorbing_axis_derivatives(
    model: Model, surfaces: tuple[str, ...], owner: np.ndarray
) -> dict[str, Derivative]:
    """The per-axis derivatives of the first-order admittance, per face.

    A face of ``surfaces`` whose owner tetrahedron (``owner``, per gfem
    triangle) lies in a field with axes takes the tensor ``Y`` of
    `anisotropy.admittance`, a function of the real principal values ``e_k
    = Re(eps_r[k] (1 - i tan_delta[k]))`` and ``m_k = Re(mu_r[k])``: its
    ``dY/de_k`` and ``dY/dm_k`` (`anisotropy.admittance_derivatives`),
    chained as the isotropic forms are. Zero on the other faces. Computed
    once, on first read (`_AxisSteps`).
    """

    @functools.cache
    def steps() -> _AxisSteps:
        mesh = model.mesh
        on = (owner >= 0) & np.isin(
            mesh.triangle_tags, [mesh.region(s).tag for s in surfaces]
        )
        on &= per_axis_elements(model)[np.maximum(owner, 0)]
        faces = np.flatnonzero(on)
        behind = owner[faces]
        eps = principal_values(model, "eps_r")
        tan = np.real(principal_values(model, "tan_delta"))
        e = np.real(eps * (1.0 - 1j * tan))[behind]
        m = np.real(principal_values(model, "mu_r"))[behind]
        by_e, by_m = anisotropy.admittance_derivatives(
            e,
            m,
            element_axes(model)[behind],
            _face_normals(mesh)[faces],
            where=f"Absorbing {', '.join(map(repr, surfaces))}",
        )
        full = []
        for part in (by_e, by_m):
            out = np.zeros((len(owner), 3, 3, 3))
            out[faces] = part
            full.append(out)
        eps_imag = np.zeros((len(owner), 3))
        eps_imag[faces] = np.imag(eps)[behind]
        return _AxisSteps(by_e=full[0], by_m=full[1], eps_imag=eps_imag)

    return {
        "eps_r": Derivative(
            lambda: steps().by_e, holomorphic=False, chain=lambda: _axis_loss(model)
        ),
        "tan_delta": Derivative(
            lambda: steps().by_e * steps().eps_imag[:, :, None, None],
            holomorphic=False,
        ),
        "mu_r": Derivative(lambda: steps().by_m, holomorphic=False),
    }


class _AxisSteps(NamedTuple):
    """`_absorbing_axis_derivatives`' arrays, per gfem triangle (zero off its faces)."""

    #: ``dY/de_k``, ``(n, 3, 3, 3)`` (k, then Y).
    by_e: np.ndarray
    #: ``dY/dm_k``, the same shape.
    by_m: np.ndarray
    #: ``Im eps_r[k]`` of the tetrahedron behind, ``(n, 3)``: ``de_k/dtan_delta_k``.
    eps_imag: np.ndarray


def _owned(
    model: Model,
    surfaces: tuple[str, ...],
    owner: np.ndarray,
    faces: np.ndarray | None,
) -> np.ndarray:
    """``owner`` on the triangles of ``surfaces`` (and of ``faces``), -1 elsewhere.

    On a 2D mesh, on its line segments.
    """
    mesh = model.mesh
    tags = mesh._topology.surface_tags  # noqa: SLF001 - either dimension's faces
    on = np.isin(tags, [mesh.region(s).tag for s in surfaces])
    if faces is not None:
        on &= np.asarray(faces)
    return np.where(on, owner, -1)


def face_owner(model: Model) -> np.ndarray:
    """The solve-domain tetrahedron whose values a face term reads, per triangle.

    The tetrahedron behind the triangle's normal when it is in the solve
    domain, else the one in front, else -1. On a 2D mesh, the area triangle
    beside each line segment.
    """
    active = model._domain.active  # noqa: SLF001
    nb = model.mesh._topology.surface_neighbours  # noqa: SLF001 - either dimension
    owner = np.full(len(nb), -1)
    for side in (1, 0):
        tet = nb[:, side]
        ok = (tet >= 0) & active[np.maximum(tet, 0)]
        owner[ok] = tet[ok]
    return owner


def absorbing_faces(model: Model, order: int = 1) -> tuple[tuple[str, ...], np.ndarray]:
    """The surfaces of the absorbing terms of ``order`` and every face's owner.

    ``order=1`` names every `Absorbing` surface (the first-order term is
    part of both orders), ``order=2`` the second-order ones; the owner per
    triangle is `face_owner`.
    """
    surfaces = tuple(
        b.surface
        for b in model.boundaries
        if isinstance(b, Absorbing) and b.order >= order
    )
    return surfaces, face_owner(model)


def _conductor(
    model: Model,
    built: EngineMesh,
    order: int,
    b: Conductor,
    faces: Any = None,
    *,
    index: int,
) -> list[Piece]:
    """One piece per thickness: a face on the mesh boundary counts it twice.

    The interior piece is ``model.boundaries[index]``'s sheet piece, with
    coefficient 1 per unit fill; a sheet is interior, so the one-sided
    piece has none.
    """
    mesh = model.mesh
    sigma, scale = b.conductivity, built.scale
    form = _tangential(built, (b.surface,), order, faces)
    on = mesh.triangle_tags == mesh.region(b.surface).tag
    one_sided = (mesh.triangle_neighbours < 0).any(axis=1)
    groups = [(1.0, on & ~one_sided), (EXTERIOR_THICKNESS_FACTOR, on & one_sided)]
    if b.thickness is None:
        groups = [(1.0, on)]
    groups = [(m, mask) for m, mask in groups if mask.any()]
    fill = sheet_fill(model, b)
    out = []
    for factor, mask in groups:
        h = None if b.thickness is None else factor * b.thickness

        def theta(w, h=h):
            return 1j * wavenumber(w, scale) * ETA0 / surface_impedance(sigma, h, w)

        coefficient: Any = 1.0

        def d_theta(w, h=h):
            # θ ∝ ω / Z_s: dθ/dω = θ (1 - (ω / Z_s) dZ_s/dω) / ω
            slope = surface_impedance_slope(sigma, h, w)
            return theta(w, h) * (1 - slope) / w

        if len(groups) == 1 and fill is None:
            integrand = form
        else:
            weight = mask.astype(float) if fill is None else mask * fill
            coefficient = surface_p0(built, weight)
            integrand = lambda u, v, c=coefficient: form(u, v, c)  # noqa: E731
        name = f"conductor {b.surface}" + ("" if factor == 1.0 else " (one-sided)")
        out.append(
            Piece(
                name,
                theta,
                integrand,
                kind="trace",
                coefficient=coefficient,
                region=form.region,
                frequency_class="nonpolynomial",
                d_theta=d_theta,
                sheet=(index, 1.0) if factor == 1.0 else None,
            )
        )
    return out


#: The lumped value of each R, L and C piece, and the power of its coefficient.
_ELEMENTS = {"R": ("resistance", -1), "L": ("inductance", -1), "C": ("capacitance", 1)}


def _rlc(
    label: str,
    scale: float,
    sheets: list,
    kind: Literal["trace", "port"],
    index: int | None = None,
    *,
    inputs: Mapping[str, tuple[tuple, Any]] | None = None,
) -> list[Piece]:
    """Pieces of per-square R, L and C in parallel on surfaces.

    ``sheets`` holds ``(form, R_s, L_s, C_s)``, ``None`` where absent, and
    optionally a face-wise fill coefficient last: the admittance i k0 eta0
    / R_s + mu0 scale / L_s - k0^2 C_s / (eps0 scale) in the equation's
    units (times the fill), one piece per kind. A ``"trace"`` piece has one
    sheet, that of ``model.boundaries[index]``: its R, L or C weight is its
    coefficient per unit fill. ``inputs`` gives, per lumped value name
    (``"inductance"``), its design input kind and value: the pieces of those
    values carry their `LumpedValue`.
    """
    weights: dict[str, list] = {"R": [], "L": [], "C": []}
    for form, rs, ls, cs, *fill in sheets:
        for element, c in _rlc_weights(rs, ls, cs, scale).items():
            weights[element].append((form, c, c * fill[0] if fill else c))
    thetas = {
        "R": lambda w: 1j * wavenumber(w, scale),
        "L": _constant,
        "C": lambda w: -(wavenumber(w, scale) ** 2),
    }
    out = []
    for element, items in weights.items():
        if items:

            def integrand(u, v, items=items):
                return sum(form(u, v, c) for form, _, c in items)

            name = f"{label} {element}"
            extra: dict[str, Any] = {}
            if kind == "trace":
                ((form, factor, c),) = items
                extra = {
                    "coefficient": c,
                    "region": form.region,
                    "sheet": None if index is None else (index, factor),
                }
            value, power = _ELEMENTS[element]
            if inputs is not None and value in inputs:
                input_kind, p = inputs[value]
                extra["lumped"] = LumpedValue(
                    input_kind,
                    power / float(p),
                    tuple((form.region, c) for form, _, c in items),
                )
            out.append(Piece(name, thetas[element], integrand, kind=kind, **extra))
    return out


def _rlc_weights(
    rs: float | None, ls: float | None, cs: float | None, scale: float
) -> dict[str, float]:
    """The R, L and C pieces' coefficients of per-square R_s, L_s and C_s."""
    out = {}
    if rs is not None:
        out["R"] = ETA0 / rs
    if ls is not None:
        out["L"] = const.mu_0 * scale / ls
    if cs is not None:
        out["C"] = cs / (const.epsilon_0 * scale)
    return out


def sheet_fill(model: Model, boundary: Conductor | Impedance) -> np.ndarray | None:
    """The fill of a sheet boundary per gfem triangle (0 off it), or ``None``.

    Raises:
        ValueError: A fill value is negative or not finite.
    """
    if boundary.fill is None:
        return None
    values = np.asarray(boundary.fill.fill, dtype=float)
    check_fill(values)
    mesh = model.mesh
    out = np.zeros(len(mesh.triangles))
    out[mesh.elements(boundary.surface)] = values
    return out


def _port(
    built: EngineMesh,
    order: int,
    data: LumpedPortData,
    faces: Any = None,
    *,
    index: int,
) -> list[Piece]:
    """The R, L and C pieces of ``model.ports[index]``; its L and C are inputs."""
    port = data.port
    check_lumped_values(port)
    sheets = [
        (_tangential(built, (e.surface,), order, faces), *data.per_square(e))
        for e in data.elements
    ]
    inputs = {
        name: (("port", index, name), getattr(port, name))
        for name in ("inductance", "capacitance")
        if getattr(port, name) is not None
    }
    return _rlc(f"port {port.name}", built.scale, sheets, "port", inputs=inputs)


def _impedance(
    model: Model,
    built: EngineMesh,
    order: int,
    b: Impedance,
    faces: Any = None,
    *,
    index: int,
) -> list[Piece]:
    check_lumped_values(b)
    form = _tangential(built, (b.surface,), order, faces)
    sheet: tuple = (form, b.resistance, b.inductance, b.capacitance)
    fill = sheet_fill(model, b)
    if fill is not None:
        sheet = (*sheet, surface_p0(built, fill))
    inputs = {
        name: (("boundary", index, name), getattr(b, name))
        for name in ("resistance", "inductance", "capacitance")
        if getattr(b, name) is not None
    }
    label = f"impedance {b.surface}"
    return _rlc(label, built.scale, [sheet], "trace", index, inputs=inputs)


def wave_port_coefficients(model: Model, surface: str) -> np.ndarray:
    """Per-triangle coefficient of a wave port's Robin term on ``surface``.

    ``1 / mu_r`` of the solve-domain element behind each face (`face_owner`),
    zero off the surface. When an anisotropic element is behind a face
    (`anisotropic_elements`: structural, never on values), the coefficient
    is ``(n, 3, 3)``: the port's tangential inverse-permeability tensor ``A
    = N mu_r^-1 N^T`` (`anisotropy.tangential_inverse_permeability`, the
    absorbing boundary's ``A``) on those faces and ``1 / mu_r`` times the
    identity on the others. On a port cross-section (transverse-anisotropic)
    ``A`` is the section's ``nu_tt`` in the global frame.
    """
    mesh = model.mesh
    owner = face_owner(model)
    on = mesh.triangle_tags == mesh.region(surface).tag
    mu = element_tensors(model, "mu_r")
    tensor = on & (owner >= 0) & anisotropic_elements(model)[np.maximum(owner, 0)]
    scalar = mu if mu.ndim == 1 else mu[:, 0, 0]
    values = np.zeros(len(owner), np.result_type(mu, float))
    values[on] = 1.0 / scalar[owner[on]]
    if not tensor.any():
        return values
    out = values[:, None, None] * np.eye(3)
    normals = _face_normals(mesh)
    out[tensor] = anisotropy.tangential_inverse_permeability(
        _tensors(mu)[owner[tensor]], normals[tensor]
    )
    return out


def wave_port_piece(
    model: Model,
    built: EngineMesh,
    order: int,
    *,
    port: str,
    surface: str,
    theta: Tabulated,
    stretch: Stretch | None = None,
) -> Piece:
    r"""The Robin term of a wave port, :math:`\theta\int (A E_t)\cdot v_t`.

    The integral runs over the port's surface.
    :math:`A` is the port's tangential inverse-permeability tensor,
    :math:`\mu_r^{-1}` for an isotropic element behind the face
    (`wave_port_coefficients`). Where the face cuts a perfectly matched
    layer (``stretch``) it is the stretched medium's,
    :math:`N\Lambda^{-1}\mu_r^{-1}N^\mathsf{T} = (N\Lambda^{-1}N^\mathsf{T})\,A`
    with :math:`N = [n]_\times` (:math:`n` a principal axis of both
    :math:`\Lambda` and :math:`\mu_r` there), as the port section's: the
    port stays exact for its own mode.
    """
    form = _tangential(built, (surface,), order)
    coef = surface_coefficient(built, wave_port_coefficients(model, surface))
    if stretch is not None:
        coef = _stretched_trace(model, built, surface, stretch) * coef
    return Piece(f"waveport {port}", theta, lambda u, v: form(u, v, coef), kind="port")


def _stretched_trace(
    model: Model, built: EngineMesh, surface: str, stretch: Stretch
) -> ngs.CoefficientFunction:
    r""":math:`N\Lambda^{-1}N^\mathsf{T}` on a planar wave port ``surface``, 3x3.

    :math:`N = [n]_\times`, the cross-product matrix of the port's normal.
    """
    mesh = model.mesh
    n = _face_normals(mesh)[mesh.elements(surface)[0]]
    matrix = ((0.0, -n[2], n[1]), (n[2], 0.0, -n[0]), (-n[1], n[0], 0.0))
    cross = ngs.CF(tuple(float(v) for row in matrix for v in row), dims=(3, 3))
    lam_inv = pml_tensor(stretch, built.scale)[1]
    return cross * lam_inv * cross.trans


def excitation_per_voltage(resistance: float) -> float:
    """``b / l`` of a lumped port of ``resistance`` ohm (see `port_vectors`).

    Each element's excitation weight is ``2 eta0 sqrt(2 R) / (R n width)``
    and its voltage weight ``1 / (n width)``: the excitation vector is the
    voltage vector times ``2 eta0 sqrt(2 / R)``.
    """
    return 2.0 * ETA0 * math.sqrt(2.0 / resistance)


def port_vectors(
    space: ngs.FESpace, built: EngineMesh, order: int, data: LumpedPortData
) -> tuple[np.ndarray, np.ndarray]:
    """A lumped port's excitation and voltage vectors over the DOFs.

    Returns ``(b, l)``: the right-hand side of 1 W incident power is
    ``i k0 b`` (incident voltage sqrt(2 R) on every element), and the port
    voltage of a field with coefficients ``x`` is ``l @ x``, the mean over
    the elements of ``(1 / width) ∫ d . E``. Without a Bloch phase the two
    are parallel, ``b = excitation_per_voltage(R) * l``, which makes S
    reciprocal and lets the port solutions serve as their own adjoints; with
    one, ``b`` is parallel to ``conj(l)`` and S(k)^T = S(-k).

    On a periodic space NGSolve assembles a linear form as Pᴴ l₀ (test
    functions take the conjugate Bloch phase), the right load; a functional
    read off x is l₀ᵀ P x, so ``l`` is the conjugate of the assembled
    vector: the integrand is real, so conj(Pᴴ l₀) = Pᵀ l₀. Without a Bloch
    phase the assembled vector is real and the conjugate changes nothing.
    """
    v = space.TestFunction()
    r = data.port.rlc[0]
    incident = math.sqrt(2.0 * r) if r is not None else 0.0
    excite = ngs.LinearForm(space)
    voltage = ngs.LinearForm(space)
    n = len(data.elements)
    for element in data.elements:
        region = boundary_region(built, (element.surface,))
        ds = quadrature.ds(order, definedon=region)
        d = ngs.CF(element.direction)
        voltage += (1.0 / (n * element.width)) * d * v.Trace() * ds
        rs = data.per_square(element)[0]
        if rs is not None:
            weight = ETA0 * 2.0 / rs * incident / element.length
            excite += weight * d * v.Trace() * ds
    excite.Assemble()
    voltage.Assemble()
    return excite.vec.FV().NumPy().copy(), np.conj(voltage.vec.FV().NumPy())
