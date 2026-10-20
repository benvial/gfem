"""The driven operator as frequency-independent pieces times scalar θ(ω).

Under e^{+iωt}, with lengths in the mesh file's unit (``scale`` metres) and
the equation multiplied by ``mu0 * scale``, the driven problem for the
electric field E (volts per file unit) is

    a(E, v; ω) = Σ_k θ_k(ω) a_k(E, v) = i k0 Σ_p b_p(v)

with k0 = ω scale / c, the wavenumber per file unit, and the pieces

- ``curl_curl``, θ = 1: ∫ mu_r^-1 curl E · curl v;
- ``mass``, θ = -k0²: ∫ eps_r (1 - i tan_delta) E · v;
- ``conductivity``, θ = i k0: ∫ eta0 sigma scale E · v;
- ``absorbing``, θ = i k0: ∫ sqrt(eps_r / mu_r) E_t · v_t on every
  absorbing surface (real parts of the adjacent element's values);
- ``absorbing order 2``, θ = 1 / (2 i k0): ∫ mu_r^-1 (eps_r mu_r)^-1/2
  (n · curl E)(n · curl v) on second-order absorbing surfaces (the
  textbook sign, see `gfem.Absorbing`);
- a conductor, θ = i k0 eta0 / Z_s(ω): ∫ E_t · v_t on its surface, with
  Z_s the surface impedance of a metal of finite thickness (thickness
  doubled on mesh-exterior faces, see `gfem.conventions`);
- an impedance boundary or a lumped port element, θ = i k0, 1, -k0²:
  ∫ E_t · v_t times eta0 / R_s, mu0 scale / L_s, C_s / (eps0 scale), its R,
  L and C per square.

- a wave port, θ = i beta(ω) per file unit (the real part of its mode's
  propagation constant, tabulated on the output grid by the port mode
  sweep, see `Tabulated`): ∫ mu_r^-1 E_t · v_t on the port surface.

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
"""

from __future__ import annotations

import cmath
import dataclasses
import math
from collections.abc import Callable
from typing import Any, Literal

import ngsolve as ngs
import numpy as np
import scipy.constants as const

from gfem._boundaries import Absorbing, Conductor, Impedance, LumpedPort
from gfem._engine import quadrature
from gfem._engine.coefficients import (
    coefficient,
    element_tensors,
    element_values,
    surface_p0,
)
from gfem._engine.domain import boundary_region
from gfem._engine.mesh import EngineMesh
from gfem.conventions import EXTERIOR_THICKNESS_FACTOR
from gfem.errors import PortGeometryError
from gfem.model import Model

__all__ = [
    "ETA0",
    "LumpedElement",
    "LumpedPortData",
    "Piece",
    "Tabulated",
    "absorbing_faces",
    "excitation_per_voltage",
    "face_owner",
    "lumped_ports",
    "pieces",
    "port_vectors",
    "surface_impedance",
    "surface_impedance_slope",
    "wave_port_piece",
    "wavenumber",
]

ETA0 = const.mu_0 * const.c
#: Angular frequency at which a piece's θ is checked to be a scalar.
_PROBE = 2 * math.pi * 1e9
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
class Piece:
    """One frequency-independent bilinear form and its scalar factor θ(ω).

    Attributes:
        name: What the piece is (``"mass"``, ``"port p1 R"``, ...).
        theta: θ(ω), a complex scalar for an angular frequency in rad/s.
        integrand: ``(trial, test) -> integrals`` of the form, with no ω.
        kind: How the form depends on the node coordinates, for shape
            gradients: ``"curl"`` (``∫ (C curl u) · curl v``), ``"value"``
            (``∫ (C u) · v``), ``"trace"`` (``∫ c u_t · v_t`` on surfaces),
            ``"curl trace"`` (``∫ c (n · curl u)(n · curl v)``), another
            ``"surface"`` form, or ``"port"`` (a port's term, which reads
            only the port's own geometry: shape gradients hold port nodes
            in place).
        coefficient: ``C`` (``c``) of a volume (trace) piece, the element-
            (face-) wise coefficient its integrand reads.
        region: The boundary region of a ``"trace"`` or ``"curl trace"``
            piece.
        frequency_class: ``"polynomial"`` (θ a polynomial of degree at most
            2 in ω) or ``"nonpolynomial"``.
        d_theta: dθ/dω of a nonpolynomial piece (``None`` for a polynomial
            one).

    Raises:
        ValueError: ``theta`` does not give one scalar: the term is not
            affine in frequency; or a nonpolynomial piece without
            ``d_theta``.
    """

    name: str
    theta: Callable[[complex], complex]
    integrand: Callable[[Any, Any], Any]
    kind: Literal["curl", "value", "trace", "curl trace", "surface", "port"] = "surface"
    coefficient: Any = None
    region: Any = None
    frequency_class: Literal["polynomial", "nonpolynomial"] = "polynomial"
    d_theta: Callable[[complex], complex] | None = None

    def __post_init__(self) -> None:
        """Check that θ is a scalar function of ω."""
        if isinstance(self.theta, Tabulated):
            if self.theta.values.ndim != 1 or not np.isfinite(self.theta.values).all():
                raise ValueError(
                    f"The driven term {self.name!r} is not tabulated as one finite "
                    "scalar per frequency."
                )
            return
        value = np.asarray(self.theta(_PROBE))
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
    """Surface impedance of a metal, ohm per square (Palace's HFSS form).

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
    """``∫ weight E_t · v_t`` on a boundary region, on the pinned rule."""

    region: Any
    order: int

    def __call__(self, u: Any, v: Any, weight: Any = 1.0) -> Any:
        ds = quadrature.ds(self.order, definedon=self.region)
        return weight * u.Trace() * v.Trace() * ds


def _tangential(
    built: EngineMesh, surfaces: tuple[str, ...], order: int
) -> _Tangential:
    return _Tangential(boundary_region(built, surfaces), order)


def _inverse(values: np.ndarray) -> np.ndarray:
    return 1.0 / values if values.ndim == 1 else np.linalg.inv(values)


def pieces(
    model: Model,
    built: EngineMesh,
    order: int,
    ports: tuple[LumpedPortData, ...],
) -> list[Piece]:
    """The driven operator's pieces on ``model`` (see the module docstring).

    `PEC` is a constraint of the space and `PMC` the natural condition, so
    neither has a piece.
    """
    scale = built.scale
    k = lambda w: wavenumber(w, scale)  # noqa: E731
    dx = quadrature.dx(order)
    active = model._domain.active  # noqa: SLF001

    eps = element_tensors(model, "eps_r") * _loss(model)
    nu = coefficient(built, _inverse(element_tensors(model, "mu_r")))
    eps_cf = coefficient(built, eps)
    out = [
        Piece(
            "curl_curl",
            _constant,
            lambda u, v: (nu * ngs.curl(u)) * ngs.curl(v) * dx,
            kind="curl",
            coefficient=nu,
        ),
        Piece(
            "mass",
            lambda w: -(k(w) ** 2),
            lambda u, v: (eps_cf * u) * v * dx,
            kind="value",
            coefficient=eps_cf,
        ),
    ]
    sigma = np.where(active, element_values(model, "conductivity").real, 0.0)
    if sigma.any():
        sigma_cf = coefficient(built, ETA0 * scale * sigma)
        out.append(
            Piece(
                "conductivity",
                lambda w: 1j * k(w),
                lambda u, v: sigma_cf * u * v * dx,
                kind="value",
                coefficient=sigma_cf,
            )
        )

    absorbing = [b for b in model.boundaries if isinstance(b, Absorbing)]
    if absorbing:
        admittance, curl = _absorbing_coefficients(model, eps)
        form = _tangential(built, tuple(b.surface for b in absorbing), order)
        y_cf = surface_p0(built, admittance)
        out.append(
            Piece(
                "absorbing",
                lambda w: 1j * k(w),
                lambda u, v: form(u, v, y_cf),
                kind="trace",
                coefficient=y_cf,
                region=form.region,
            )
        )
        second = tuple(b.surface for b in absorbing if b.order == 2)
        if second:
            curl_cf = surface_p0(built, curl)
            region = boundary_region(built, second)
            ds = quadrature.ds(order, definedon=region)
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
                )
            )

    for b in model.boundaries:
        if isinstance(b, Conductor):
            out.extend(_conductor(model, built, order, b))
        elif isinstance(b, Impedance):
            out.extend(_impedance(built, order, b))

    for data in ports:
        out.extend(_port(built, order, data))
    return out


def _loss(model: Model) -> np.ndarray:
    """The factor 1 - i tan_delta per tetrahedron, shaped like eps_r."""
    factor = 1.0 - 1j * element_values(model, "tan_delta").real
    eps = element_tensors(model, "eps_r")
    return factor if eps.ndim == 1 else factor[:, None, None]


def _absorbing_coefficients(
    model: Model, eps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle coefficients of the absorbing terms.

    From the real parts of the solve-domain element behind each triangle
    (tensors through their mean diagonal): ``sqrt(eps_r / mu_r)`` for the
    first-order term and ``1 / (mu_r sqrt(eps_r mu_r))`` for the
    second-order one (mu_r^-1 times 1 / sqrt(eps_r mu_r), the 1/k of the
    adjacent material relative to k0).
    """
    mu = element_tensors(model, "mu_r")
    if eps.ndim == 3:
        eps = np.trace(eps, axis1=1, axis2=2) / 3
    if mu.ndim == 3:
        mu = np.trace(mu, axis1=1, axis2=2) / 3
    eps, mu = np.real(eps), np.real(mu)
    values = np.stack([np.sqrt(eps / mu), 1 / (mu * np.sqrt(eps * mu))], axis=1)
    owner = face_owner(model)
    out = np.zeros((len(owner), 2))
    ok = owner >= 0
    out[ok] = values[owner[ok]]
    return out[:, 0], out[:, 1]


def face_owner(model: Model) -> np.ndarray:
    """The solve-domain tetrahedron whose values a face term reads, per triangle.

    The tetrahedron behind the triangle's normal when it is in the solve
    domain, else the one in front, else -1.
    """
    active = model._domain.active  # noqa: SLF001
    nb = model.mesh.triangle_neighbours
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
    model: Model, built: EngineMesh, order: int, b: Conductor
) -> list[Piece]:
    """One piece per thickness: a face on the mesh boundary counts it twice."""
    mesh = model.mesh
    sigma, scale = b.conductivity, built.scale
    form = _tangential(built, (b.surface,), order)
    on = mesh.triangle_tags == mesh.region(b.surface).tag
    one_sided = (mesh.triangle_neighbours < 0).any(axis=1)
    groups = [(1.0, on & ~one_sided), (EXTERIOR_THICKNESS_FACTOR, on & one_sided)]
    if b.thickness is None:
        groups = [(1.0, on)]
    groups = [(m, mask) for m, mask in groups if mask.any()]
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

        if len(groups) == 1:
            integrand = form
        else:
            coefficient = surface_p0(built, mask.astype(float))
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
            )
        )
    return out


def _rlc(
    label: str, scale: float, sheets: list, kind: Literal["trace", "port"]
) -> list[Piece]:
    """Pieces of per-square R, L and C in parallel on surfaces.

    ``sheets`` holds ``(form, R_s, L_s, C_s)``, ``None`` where absent: the
    admittance i k0 eta0 / R_s + mu0 scale / L_s - k0^2 C_s / (eps0 scale)
    in the equation's units, one piece per kind. A ``"trace"`` piece has
    one sheet.
    """
    weights: dict[str, list] = {"R": [], "L": [], "C": []}
    for form, rs, ls, cs in sheets:
        if rs is not None:
            weights["R"].append((form, ETA0 / rs))
        if ls is not None:
            weights["L"].append((form, const.mu_0 * scale / ls))
        if cs is not None:
            weights["C"].append((form, cs / (const.epsilon_0 * scale)))
    thetas = {
        "R": lambda w: 1j * wavenumber(w, scale),
        "L": _constant,
        "C": lambda w: -(wavenumber(w, scale) ** 2),
    }
    out = []
    for element, items in weights.items():
        if items:

            def integrand(u, v, items=items):
                return sum(form(u, v, c) for form, c in items)

            name = f"{label} {element}"
            extra: dict[str, Any] = {}
            if kind == "trace":
                ((form, c),) = items
                extra = {"coefficient": c, "region": form.region}
            out.append(Piece(name, thetas[element], integrand, kind=kind, **extra))
    return out


def _port(built: EngineMesh, order: int, data: LumpedPortData) -> list[Piece]:
    sheets = [
        (_tangential(built, (e.surface,), order), *data.per_square(e))
        for e in data.elements
    ]
    return _rlc(f"port {data.port.name}", built.scale, sheets, "port")


def _impedance(built: EngineMesh, order: int, b: Impedance) -> list[Piece]:
    form = _tangential(built, (b.surface,), order)
    sheet = (form, b.resistance, b.inductance, b.capacitance)
    return _rlc(f"impedance {b.surface}", built.scale, [sheet], "trace")


def wave_port_piece(
    model: Model,
    built: EngineMesh,
    order: int,
    *,
    port: str,
    surface: str,
    theta: Tabulated,
) -> Piece:
    """The Robin term of a wave port: θ ∫ mu_r^-1 E_t · v_t on its surface.

    ``mu_r`` is that of the solve-domain element behind each face.

    Raises:
        NotImplementedError: An anisotropic ``mu_r`` touches the port.
    """
    mesh = model.mesh
    owner = face_owner(model)
    on = mesh.triangle_tags == mesh.region(surface).tag
    mu = element_tensors(model, "mu_r")
    if mu.ndim == 3:
        if np.ptp(mu[owner[on]] - mu[owner[on]][:, :1, :1] * np.eye(3)) > 0:
            raise NotImplementedError(
                f"WavePort {port!r}: an anisotropic mu_r touches the port; driven "
                "wave ports support isotropic permeability only."
            )
        mu = mu[:, 0, 0]
    values = np.zeros(len(owner), np.result_type(mu, float))
    values[on] = 1.0 / mu[owner[on]]
    form = _tangential(built, (surface,), order)
    coef = surface_p0(built, values)
    return Piece(f"waveport {port}", theta, lambda u, v: form(u, v, coef), kind="port")


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
    the elements of ``(1 / width) ∫ d . E``. The two are parallel, ``b =
    excitation_per_voltage(R) * l``, which makes S reciprocal and lets the
    port solutions serve as their own adjoints.
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
    return excite.vec.FV().NumPy().copy(), voltage.vec.FV().NumPy().copy()
