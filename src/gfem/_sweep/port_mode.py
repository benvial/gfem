"""Port modes over a frequency grid: one exact port mode solve per frequency.

Each solve factors the shifted 2D pencil with SuperLU, runs ARPACK
shift-invert at ``sigma = -1.1 (n_max k0)^2`` and keeps the requested mode in
Palace's order: converged, propagating Ritz pairs below the largest bulk
wavenumber (the formulation's spurious modes sit above it), by decreasing
``Re 1 / (lam - sigma)``. For lossless modes this is decreasing ``beta``;
strongly attenuated (leaky) modes rank lower. The
mode is normalized to 1 W through the port, its phase fixed by Palace's
quadrant rule and, with a voltage path, its sign by ``Re V > 0``.
"""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from gfem._boundaries import WavePort
from gfem._checks import is_traced
from gfem._engine.port_mode import Z0, PortModeSpace
from gfem._linalg.superlu import backward_error, shift_invert_eigs
from gfem._port_mode import C0, PortMode, PortModeField, PortModeResult
from gfem._port_section import PortSection, port_section
from gfem.errors import NonDifferentiableError, PortModeError
from gfem.model import Model
from gfem.options import SolverOptions

# The shift is this factor above the largest bulk (Re eps)(Re mu), as in Palace.
_TARGET = 1.1
# A Ritz pair propagates when Re beta exceeds this fraction of k0 n_max.
_EVANESCENT = 1e-6
# Ritz pairs with a larger backward error are unconverged (ARPACK returns some
# from the formulation's large null space below cutoff); converged ones sit
# near 1e-14, or near 1e-6 at the low-frequency accuracy floor.
_UNCONVERGED = 1e-3
# Gauss-Legendre rule for voltage line integrals: points per piece, pieces.
_GAUSS = np.polynomial.legendre.leggauss(4)
_PIECES = 32


def solve_port_mode(
    model: Model, problem: PortMode, options: SolverOptions
) -> PortModeResult:
    """Solve ``problem`` on ``model`` (the `gfem.solve` path for `PortMode`)."""
    if any(is_traced(leaf) for leaf in jax.tree_util.tree_leaves(model)):
        raise NonDifferentiableError(
            "The port mode problem is not differentiable in gfem v0.1 (port modes "
            "are constants of a solve). Solve PortMode outside the function you "
            "differentiate."
        )
    port = _wave_port(model, problem.port)
    mode = port.mode if problem.mode is None else problem.mode
    section = port_section(model, port)
    space = PortModeSpace(section, options.order)
    path = _voltage_rule(section) if port.voltage_path is not None else None

    freqs = np.asarray(problem.frequency)
    gammas, residuals, z_pv, coeffs = [], [], [], []
    for f in freqs:
        gamma, residual, c = _solve_one(space, f, mode)
        z = math.nan
        if path is not None:
            v = _voltage(space, c, path)
            if v.real < 0:
                c, v = -c, -v
            z = abs(v) ** 2 / 2.0
        gammas.append(gamma / section.scale)
        residuals.append(residual)
        z_pv.append(z)
        coeffs.append(c)
    return PortModeResult(
        port=port.name,
        mode=mode,
        frequency=jnp.asarray(freqs),
        gamma=jnp.asarray(np.array(gammas, complex)),
        Z_PV=jnp.asarray(np.array(z_pv, float)),
        residual=jnp.asarray(np.array(residuals, float)),
        e_field=PortModeField(space, np.array(coeffs)),
    )


def _wave_port(model: Model, name: str) -> WavePort:
    for p in model.ports:
        if p.name == name:
            if not isinstance(p, WavePort):
                raise ValueError(
                    f"PortMode(port={name!r}): {name!r} is a {type(p).__name__}; the "
                    "port mode problem needs a WavePort."
                )
            return p
    waves = [p.name for p in model.ports if isinstance(p, WavePort)]
    known = ", ".join(repr(w) for w in waves) if waves else "none"
    raise ValueError(
        f"PortMode(port={name!r}): the model has no port named {name!r}. Its wave "
        f"ports: {known}."
    )


def _solve_one(
    space: PortModeSpace, frequency: float, mode: int
) -> tuple[complex, float, np.ndarray]:
    """One exact port mode solve.

    Returns:
        ``gamma`` per local unit, the backward error, and the normalized full
        coefficient vector of ``(E_t, E_n)``.
    """
    section = space.section
    k0_si = 2 * math.pi * frequency / C0
    k0 = k0_si * section.scale
    a, b = space.pencil.at(k0)
    n2_shift, n2_max = section.index_squared(float(section.sigma.max()) * Z0 / k0_si)
    k_max = math.sqrt(n2_max) * k0
    shift = -_TARGET * n2_shift * k0**2

    n = a.shape[0]
    first = max(6, 2 * mode + 4)
    for want in (first, 4 * first):
        lam, vecs = shift_invert_eigs(a, b, shift, min(want, n - 2))
        beta = np.sqrt(-lam + 0j)
        res = np.array(
            [backward_error(a, b, lam[i], vecs[:, i]) for i in range(len(lam))]
        )
        ok = (
            (beta.real > _EVANESCENT * k_max)
            & (beta.real <= k_max * (1 + 1e-6))
            & (res < _UNCONVERGED)
        )
        # Palace's order: largest Re of the shift-inverted eigenvalue
        rank = (1 / (lam - shift)).real
        order = np.flatnonzero(ok)[np.argsort(-rank[ok], kind="stable")]
        if len(order) >= mode or want >= n - 2:
            break
    if len(order) < mode:
        raise PortModeError(
            f"WavePort {section.port.name!r}: mode {mode} does not propagate at "
            f"{frequency / 1e9:.6g} GHz (the cross-section carries {len(order)} "
            "propagating mode(s) there): the frequency is below the mode's cutoff. "
            "Start the band above the cutoff, pick a lower mode, or use a "
            "LumpedPort."
        )
    i = order[mode - 1]
    gamma = 1j * complex(beta[i])
    omega = 2 * math.pi * frequency
    return gamma, float(res[i]), _normalize(space, vecs[:, i], gamma, omega)


def _normalize(
    space: PortModeSpace, x: np.ndarray, gamma: complex, omega: float
) -> np.ndarray:
    """Coefficients of ``(E_t, E_n)`` at 1 W, phase fixed by Palace's rule."""
    c = space.full(x)
    c[: space.n_t] /= gamma
    c /= math.sqrt(abs(space.power(c, gamma, omega)))
    g = space.gauge(c, gamma)
    if abs(g) > 1e-12 * np.abs(c).max():
        c *= abs(g) / g
    else:
        j = int(np.argmax(np.abs(c)))
        c *= abs(c[j]) / c[j]
    return c


def _voltage_rule(section: PortSection) -> tuple[np.ndarray, np.ndarray]:
    """Plane points and weights (local direction x length, m) of the path."""
    path = np.asarray(section.port.voltage_path, float)
    _local, dist = section.to_local(path)
    if (np.abs(dist) > 1e-5 * section.diameter).any():
        raise ValueError(
            f"WavePort {section.port.name!r}: voltage_path leaves the port plane; "
            "give points (m) on the port surface."
        )
    x, w = _GAUSS
    pts, wts = [], []
    for p, q in itertools.pairwise(path):
        d_local = section.frame @ (q - p)
        for k in range(_PIECES):
            s = (k + (x + 1) / 2) / _PIECES
            pts.append(section.to_local(p + np.outer(s, q - p))[0])
            wts.append(np.outer(w / (2 * _PIECES), d_local))
    return np.concatenate(pts), np.concatenate(wts)


def _voltage(
    space: PortModeSpace, coeffs: np.ndarray, rule: tuple[np.ndarray, np.ndarray]
) -> complex:
    """``V = int E . dl`` along the voltage path, V."""
    pts, wts = rule
    e = space.evaluate(coeffs, pts)[0]
    return complex(np.sum(e * wts))
