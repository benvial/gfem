r"""The reverse mode of one port mode over its frequency grid.

Given the cotangents of a mode's outputs (the propagation constant, the
power-voltage impedance and the field's DOF vector at each frequency), `mode_pass`
returns the cotangents of the 2D section's coefficients
(`gfem._engine.port_mode_sensitivity.SectionCotangents`), summed over the
frequencies, plus the law parameters' gradient (which needs each row's
frequency).

A cotangent on :math:`\gamma` alone is the Hellmann-Feynman term and needs no
solve. A cotangent on the field or on :math:`Z_{PV}` runs the normalization's
pullback (`gfem._engine.port_mode_normalization`) over the rows in reverse
order, then one deflated adjoint solve per row
(`gfem._engine.port_mode_adjoint`). A reduced row is first polished by one
Newton step, so that its derivative is the exact discrete mode's; the
normalization then runs on the whole free-DOF vector.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from gfem._engine.line_mode import LineModeSpace
from gfem._engine.line_mode_sensitivity import (
    LineCotangents,
    LineKernel,
    full_line_reduction,
    line_normalize_vjp,
)
from gfem._engine.port_mode import FullReduction, PortModeSpace
from gfem._engine.port_mode_adjoint import (
    POLISH_RTOL,
    pencil_cotangent,
    polish,
    shift_invert_factor,
)
from gfem._engine.port_mode_normalization import normalize_rows, normalize_vjp
from gfem._engine.port_mode_sensitivity import (
    BendDerivative,
    SectionCotangents,
    SectionKernel,
    law_cotangents,
)
from gfem._engine.port_mode_shape import ShapeForm, voltage_eulerian
from gfem.errors import NonDifferentiableError

__all__ = ["ModePass", "mode_pass"]

# A new direction joins the extended basis unless it is below this fraction of
# the vector it came from (rounding noise).
_NEW_DIRECTION = 1e-12
#: The relative imaginary part of $\bar z^\mathsf{T}z$ above which a field
#: cotangent is phase-dependent (the sum's round-off is $\sim10^{-14}$ of its
#: terms).
PHASE_RTOL = 1e-10


@dataclasses.dataclass(eq=False)
class ModePass:
    """What a mode's backward gives the section.

    Attributes:
        section: The coefficients' cotangents, summed over the rows (the
            normalization's explicit form terms included).
        laws: The gradient by each permittivity-law parameter, by input kind.
        points: The cotangent of the section's points, ``(n_points, 2)`` per
            local unit, or ``None`` when the node coordinates are not wanted.
        radius: The derivative by the bend's radius, in 1/m (zero without a
            bend, or when it is not wanted).
    """

    section: SectionCotangents | LineCotangents
    laws: dict[tuple, float]
    points: np.ndarray | None = None
    radius: float = 0.0


def mode_pass(  # noqa: PLR0912, PLR0915
    space: PortModeSpace | LineModeSpace,
    *,
    frequency: np.ndarray,
    lam: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    solved: np.ndarray,
    wavenumbers: Callable[[float], tuple[float, float, float]],
    volts: np.ndarray | None,
    polarity: np.ndarray | None = None,
    ct_gamma: np.ndarray | None,
    ct_z_pv: np.ndarray | None,
    ct_field: np.ndarray | None,
    who: str,
    nodes: bool = False,
    voltage_rule: tuple[np.ndarray, np.ndarray] | None = None,
    radius: bool = False,
) -> ModePass:
    r"""The section cotangents of a mode's outputs.

    Args:
        space: The port mode problem.
        frequency: ``(n,)`` Hz, in the order of the other arrays.
        lam: ``(n,)`` the pencil eigenvalue of each row.
        y: ``(n, m)`` the pencil vector of each row in the basis ``v``.
        v: ``(n_free, m)`` the real reduced basis.
        solved: ``(n,)`` whether a row is an exact solve.
        wavenumbers: ``frequency -> (k0, k_max, shift)`` per local unit.
        volts: The voltage functional on the free DOFs, ``None`` without a
            voltage path.
        polarity: The polarity reference's reading on the free DOFs (the
            field at its point along its arrow), ``None`` without one: it
            fixes each row's phase on its own, in the backward as in the
            sweep.
        ct_gamma: ``(n,)`` the cotangent of :math:`\gamma` in 1/m, or ``None``.
        ct_z_pv: ``(n,)`` the cotangent of :math:`Z_{PV}`, or ``None``.
        ct_field: ``(n, n_dof)`` the cotangent of the field vector, or ``None``.
        who: The port, for messages.
        nodes: Whether to also return the cotangent of the section's points
            (`ModePass.points`).
        voltage_rule: The voltage path's plane points and weights
            (`gfem._sweep.port_mode._voltage_rule`), for its Eulerian node
            derivative; ``None`` without a path.
        radius: Whether to also return the derivative by the bend's radius
            (`ModePass.radius`) of a bent section.

    Raises:
        PortModeError: An adjoint solve missed (`gfem._engine.port_mode_adjoint`).
    """
    n = len(frequency)
    scale = space.section.scale
    order = np.argsort(frequency, kind="stable")
    f = np.asarray(frequency, float)[order]
    lam_s = np.asarray(lam)[order]
    y_s = np.asarray(y)[order]
    solved_s = np.asarray(solved)[order]
    omega = 2 * math.pi * f
    cg = (
        np.zeros(n, complex)
        if ct_gamma is None
        else np.asarray(ct_gamma, complex)[order] / scale
    )
    waves = [wavenumbers(float(fi)) for fi in f]
    k0 = np.array([w[0] for w in waves])
    shift = np.array([w[2] for w in waves])
    exact = np.flatnonzero(solved_s)
    bent = getattr(space.section, "bend", None)
    factors: dict[int, Any] = {}

    def factor_of(i: int) -> Any:
        j = int(exact[np.argmin(np.abs(f[exact] - f[i]))])
        if j not in factors:
            a, b = space.pencil.at(k0[j])
            # on a bent section the layer modes crowd the sweep's shift: the
            # row's own eigenvalue is the one that separates the pair
            at = shift[j] if bent is None else float(lam_s[j].real)
            factors[j] = shift_invert_factor(a, b, at)
        return factors[j]

    x = np.array([v @ yi for yi in y_s], complex)
    lam_r = lam_s.astype(complex).copy()
    has_path = volts is not None
    needs = (ct_field is not None and np.any(ct_field)) or (
        has_path and ct_z_pv is not None and np.any(ct_z_pv)
    )
    x_bar = np.zeros_like(x)
    pairs_forms: list[Any] = []
    planar = isinstance(space, PortModeSpace)
    shape = ShapeForm(space) if nodes and planar else None
    points = len(space.section.points) if shape is not None else 0
    shape_points = np.zeros((points, 2))
    if needs or np.any(cg):
        # polish the reduced rows: the derivative is the exact mode's, also
        # for a gamma cotangent alone (the Hellmann-Feynman term reads the pair)
        for i in range(n):
            if solved_s[i]:
                continue
            a, b = space.pencil.at(k0[i])
            lam_r[i], x[i] = polish(
                a, b, lam_r[i], x[i], factor=factor_of(i), rtol=POLISH_RTOL, who=who
            )
    if needs:
        # the normalization on the whole free-DOF vector: a cotangent on the
        # field points anywhere in the space, not only along the reduced basis
        line = isinstance(space, LineModeSpace)
        fields: Any = full_line_reduction(space) if line else FullReduction(space)
        mass = space.mass
        gamma = 1j * np.sqrt(-lam_r + 0j)
        norm_rows = normalize_rows(
            fields, mass, x, gamma, omega, volts=volts, polarity=polarity
        )
        z_bar = np.zeros(norm_rows.z.shape, complex)
        if ct_field is not None:
            z_bar = np.asarray(ct_field, complex)[order][:, space.free]
            if shape is not None:
                _refuse_phase(norm_rows.z, z_bar, who)
        v_bar = np.zeros(n, complex)
        p_bar = np.zeros(n, complex)
        if has_path and ct_z_pv is not None:
            ct = np.asarray(ct_z_pv, complex)[order]
            volt = norm_rows.voltage
            phase = norm_rows.phase
            z_bar_v = ct * np.conj(volt) / (2 * np.conj(phase))
            v_bar = z_bar_v + np.conj(ct * volt / (2 * np.conj(phase)))
            p_bar = -np.conj(ct * np.abs(volt) ** 2 / (2 * np.conj(phase) ** 2))
        pulled: Any
        if line:
            pulled = line_normalize_vjp(
                fields,
                mass,
                norm_rows,
                x,
                gamma,
                omega,
                z_bar=z_bar,
                polarity=polarity,
            )
        else:
            pulled = normalize_vjp(
                fields,
                mass,
                norm_rows,
                x,
                gamma,
                omega,
                z_bar=z_bar,
                voltage_bar=v_bar,
                phase_bar=p_bar,
                volts=volts,
                polarity=polarity,
            )
        pairs_forms = list(pulled.pairs)
        x_bar = pulled.y
        cg = cg + pulled.gamma
        if (
            isinstance(space, PortModeSpace)
            and shape is not None
            and has_path
            and voltage_rule is not None
        ):
            # the path's points stay put while the mesh moves
            for i in range(n):
                if v_bar[i] != 0:
                    z_full = space.full(norm_rows.z[i])
                    shape_points += voltage_eulerian(
                        space, z_full, *voltage_rule, v_bar[i]
                    )
    gamma = 1j * np.sqrt(-lam_r + 0j)
    lam_bar = cg / (2 * gamma)

    kernel, total = _kernel_and_total(space, float(k0[0]))
    laws: dict[tuple, float] = {}
    derivative = (
        BendDerivative(space)
        if radius and bent is not None and isinstance(space, PortModeSpace)
        else None
    )
    d_u = 0.0  # d L / d u, with u = scale / R
    for i in range(n):
        if lam_bar[i] == 0 and not np.any(x_bar[i]):
            continue
        a, b = space.pencil.at(k0[i])
        q = pencil_cotangent(
            a,
            b,
            lam_r[i],
            x[i],
            lam_bar=lam_bar[i],
            x_bar=x_bar[i],
            factor=factor_of(i),
            who=who,
        )
        row = kernel.at(k0[i]).cotangents([(q, x[i], lam_r[i])])
        if derivative is not None:
            theta = space.pencil.theta(k0[i])
            d_u += derivative.pencil(theta, lam_r[i], q, x[i]).real
        if shape is not None:
            shape.add_pencil(q, x[i], lam_r[i], space.pencil.theta(k0[i]))
        if isinstance(space, PortModeSpace):
            for key, value in law_cotangents(space.section, row, omega[i]).items():
                laws[key] = laws.get(key, 0.0) + value
        total.add(row)
    if pairs_forms:
        total.add(kernel.explicit(pairs_forms))
        if derivative is not None:
            d_u += sum(derivative.pair(pair) for pair in pairs_forms).real
    # u = scale / R in metres: d u / d R = -u^2 / scale
    d_r = 0.0 if derivative is None else -d_u * scale / derivative.radius**2
    if shape is None:
        return ModePass(total, laws, radius=d_r)
    for pair in pairs_forms:
        shape.add_pair(pair)
    return ModePass(total, laws, shape.result() + shape_points, d_r)


def _refuse_phase(z: np.ndarray, z_bar: np.ndarray, who: str) -> None:
    r"""Refuse a field cotangent that depends on a mode's phase.

    The phase of a mode's field is a gauge that the node derivative does not
    carry (the quadrant, the overlap through the mass matrix and the polarity
    have no node term): a phase-dependent objective would get a silently wrong
    gradient. Under a phase $z\to e^{i\varphi}z$ the objective is stationary
    when $\operatorname{Im}\bar z^\mathsf{T}z = 0$, which holds to the
    round-off of the sum of the terms.

    Raises:
        NonDifferentiableError: The cotangent of a row has an imaginary part
            beyond $10^{-10}$ of its terms' magnitude.
    """
    drift = np.abs(np.imag(np.sum(z_bar * z, axis=-1)))
    scale = np.sum(np.abs(z_bar) * np.abs(z), axis=-1)
    if np.any(drift > PHASE_RTOL * scale):
        raise NonDifferentiableError(
            f"gfem cannot differentiate {who} through the field's phase while "
            "the node coordinates are differentiated: the objective changes "
            "under a phase factor on a mode's field, and the gauge has no node "
            "derivative in this version. Use a phase-invariant objective (an "
            "intensity, gamma_NL or |gamma_abcd|^2), or wrap the node "
            "coordinates in jax.lax.stop_gradient."
        )


def _kernel_and_total(
    space: PortModeSpace | LineModeSpace, k0: float
) -> tuple[Any, Any]:
    """The material kernel of the space, and its empty cotangents."""
    if isinstance(space, LineModeSpace):
        return LineKernel(space, k0), LineCotangents.zeros(space)
    return SectionKernel(space, k0), SectionCotangents.zeros(space.section)
