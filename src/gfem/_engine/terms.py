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

Each θ is a closed-form scalar function of ω and each a_k is assembled
without ω, so an adaptive sweep can project the pieces once. A term that
cannot be written this way is not a `Piece`: constructing one raises.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

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
    "lumped_ports",
    "pieces",
    "port_vectors",
    "surface_impedance",
    "wavenumber",
]

ETA0 = const.mu_0 * const.c
#: Angular frequency at which a piece's θ is checked to be a scalar.
_PROBE = 2 * math.pi * 1e9
#: |d . n| above which a port direction is not tangent to its surface.
_TANGENT = 1e-6


def wavenumber(omega: float, scale: float) -> float:
    """k0 in radians per mesh file unit."""
    return omega * scale / const.c


@dataclasses.dataclass(frozen=True)
class Piece:
    """One frequency-independent bilinear form and its scalar factor θ(ω).

    Attributes:
        name: What the piece is (``"mass"``, ``"port p1 R"``, ...).
        theta: θ(ω), a complex scalar for an angular frequency in rad/s.
        integrand: ``(trial, test) -> integrals`` of the form, with no ω.

    Raises:
        ValueError: ``theta`` does not give one scalar: the term is not
            affine in frequency.
    """

    name: str
    theta: Callable[[float], complex]
    integrand: Callable[[Any, Any], Any]

    def __post_init__(self) -> None:
        """Check that θ is a scalar function of ω."""
        value = np.asarray(self.theta(_PROBE))
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError(
                f"The driven term {self.name!r} is not affine in frequency (a "
                "frequency-independent form times a scalar function of the "
                "frequency), so gfem cannot sweep it."
            )


def _constant(omega: float) -> complex:  # noqa: ARG001
    """θ = 1: a frequency-independent piece."""
    return 1.0


def surface_impedance(sigma: float, thickness: float | None, omega: float) -> complex:
    """Surface impedance of a metal, ohm per square (Palace's HFSS form).

    Thick metal: ``Z_s = (1 + i) / (sigma delta)`` with skin depth
    ``delta = sqrt(2 / (mu0 sigma omega))``. A film of thickness ``h``
    multiplies the real and imaginary parts by ``(sinh v ± sin v) /
    (cosh v - cos v)``, ``v = h / delta``. For a thin film it tends to
    ``2 / (sigma h)``: each face of the film carries half its current.

    Args:
        sigma: Conductivity, S/m.
        thickness: Thickness in m, or ``None`` for thick metal.
        omega: Angular frequency, rad/s.
    """
    delta = math.sqrt(2.0 / (const.mu_0 * sigma * omega))
    real = imag = 1.0 / (sigma * delta)
    if thickness is not None:
        v = thickness / delta
        if v < 50:  # beyond, the correction is 1 to machine precision
            den = math.cosh(v) - math.cos(v)
            real *= (math.sinh(v) + math.sin(v)) / den
            imag *= (math.sinh(v) - math.sin(v)) / den
    return complex(real, imag)


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


def _tangential(built: EngineMesh, surfaces: tuple[str, ...], order: int):
    region = boundary_region(built, surfaces)

    def integrand(u, v, weight=1.0):
        return weight * u.Trace() * v.Trace() * quadrature.ds(order, definedon=region)

    return integrand


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
        ),
        Piece("mass", lambda w: -(k(w) ** 2), lambda u, v: (eps_cf * u) * v * dx),
    ]
    sigma = np.where(active, element_values(model, "conductivity").real, 0.0)
    if sigma.any():
        sigma_cf = coefficient(built, ETA0 * scale * sigma)
        out.append(
            Piece(
                "conductivity", lambda w: 1j * k(w), lambda u, v: sigma_cf * u * v * dx
            )
        )

    absorbing = [b for b in model.boundaries if isinstance(b, Absorbing)]
    if absorbing:
        admittance, curl = _absorbing_coefficients(model, eps)
        form = _tangential(built, tuple(b.surface for b in absorbing), order)
        y_cf = surface_p0(built, admittance)
        out.append(
            Piece("absorbing", lambda w: 1j * k(w), lambda u, v: form(u, v, y_cf))
        )
        second = tuple(b.surface for b in absorbing if b.order == 2)
        if second:
            curl_cf = surface_p0(built, curl)
            ds = quadrature.ds(order, definedon=boundary_region(built, second))
            out.append(
                Piece(
                    "absorbing order 2",
                    lambda w: 1 / (2j * k(w)),
                    lambda u, v: (
                        curl_cf * ngs.curl(u).Trace() * ngs.curl(v).Trace() * ds
                    ),
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
    mesh = model.mesh
    active = model._domain.active  # noqa: SLF001
    mu = element_tensors(model, "mu_r")
    if eps.ndim == 3:
        eps = np.trace(eps, axis1=1, axis2=2) / 3
    if mu.ndim == 3:
        mu = np.trace(mu, axis1=1, axis2=2) / 3
    eps, mu = np.real(eps), np.real(mu)
    values = np.stack([np.sqrt(eps / mu), 1 / (mu * np.sqrt(eps * mu))], axis=1)
    nb = mesh.triangle_neighbours
    out = np.zeros((len(nb), 2))
    for side in (1, 0):
        tet = nb[:, side]
        ok = (tet >= 0) & active[np.maximum(tet, 0)]
        out[ok] = values[tet[ok]]
    return out[:, 0], out[:, 1]


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

        if len(groups) == 1:
            integrand = form
        else:
            indicator = surface_p0(built, mask.astype(float))
            integrand = lambda u, v, c=indicator: form(u, v, c)  # noqa: E731
        name = f"conductor {b.surface}" + ("" if factor == 1.0 else " (one-sided)")
        out.append(Piece(name, theta, integrand))
    return out


def _rlc(label: str, scale: float, sheets: list) -> list[Piece]:
    """Pieces of per-square R, L and C in parallel on surfaces.

    ``sheets`` holds ``(form, R_s, L_s, C_s)``, ``None`` where absent: the
    admittance i k0 eta0 / R_s + mu0 scale / L_s - k0^2 C_s / (eps0 scale)
    in the equation's units, one piece per kind.
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
    for kind, items in weights.items():
        if items:

            def integrand(u, v, items=items):
                return sum(form(u, v, c) for form, c in items)

            out.append(Piece(f"{label} {kind}", thetas[kind], integrand))
    return out


def _port(built: EngineMesh, order: int, data: LumpedPortData) -> list[Piece]:
    sheets = [
        (_tangential(built, (e.surface,), order), *data.per_square(e))
        for e in data.elements
    ]
    return _rlc(f"port {data.port.name}", built.scale, sheets)


def _impedance(built: EngineMesh, order: int, b: Impedance) -> list[Piece]:
    form = _tangential(built, (b.surface,), order)
    sheet = (form, b.resistance, b.inductance, b.capacitance)
    return _rlc(f"impedance {b.surface}", built.scale, [sheet])


def port_vectors(
    space: ngs.FESpace, built: EngineMesh, order: int, data: LumpedPortData
) -> tuple[np.ndarray, np.ndarray]:
    """A lumped port's excitation and voltage vectors over the DOFs.

    Returns ``(b, l)``: the right-hand side of 1 W incident power is
    ``i k0 b`` (incident voltage sqrt(2 R) on every element), and the port
    voltage of a field with coefficients ``x`` is ``l @ x``, the mean over
    the elements of ``(1 / width) ∫ d . E``.
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
