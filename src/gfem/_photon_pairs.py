r"""Pure-JAX maths of the photon-pair figures: no finite elements, no results.

A pair source's two-photon amplitude is the product of a pump envelope, a
coupling and a phase-matching amplitude; its figures are sums over a frequency
grid. The pieces here take arrays and scalars and return arrays, so they
differentiate, jit and are checked on their own (``gfem.photon_pairs`` reads
the port modes and calls them).

Frequencies ``f`` are in Hz, angular offsets ``omega`` in rad/s.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np

#: highest degree of an arm's dispersion fit: with 6 rows or more, a quartic
#: carries :math:`\beta_2` to :math:`\beta_4` and keeps a residual degree of freedom
_MAX_DEGREE = 4
#: fewest rows of an arm: degree 2 (group-velocity dispersion) and one residual
_MIN_ROWS = 4
#: :math:`c_k = 1/(2k+1)!`, the coefficients of the series of :math:`\sinh w/w`
#: in :math:`w^2`; nine terms reach a relative 1e-17 for :math:`|w|^2 < 0.25`
_SINHC = tuple(1.0 / math.factorial(2 * k + 1) for k in range(10))
#: :math:`|w|^2` below which the series replaces the quotient, which loses all
#: its digits as :math:`w \to 0`
_SERIES_BELOW = 0.25


# --- dispersion of an arm ----------------------------------------------------------


def arm_basis(frequency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""The fixed least-squares map of one arm's rows onto its fit coefficients.

    With :math:`x = (f - f_c)/\Delta f` the frequency scaled to :math:`[-1, 1]`
    (:math:`f_c` the middle and :math:`\Delta f` the half of the rows' span) and
    degree :math:`d = \min(n - 2, 4)`, the Vandermonde matrix :math:`V` of the
    rows has pseudo-inverse :math:`V^+`; the coefficients of a quantity
    :math:`q` sampled at the rows are :math:`V^+ q`, exactly linear in
    :math:`q`.

    Returns:
        ``(pinv, vander)``, ``(d + 1, n)`` and ``(n, d + 1)``.

    Raises:
        ValueError: Fewer than 4 rows, or two rows at one frequency.
    """
    f = np.asarray(frequency, float)
    n = len(f)
    if n < _MIN_ROWS:
        raise ValueError(
            f"an arm needs at least {_MIN_ROWS} frequency rows to fit its "
            f"dispersion, got {n}."
        )
    if len(np.unique(f)) != n:
        raise ValueError("an arm's frequency rows must be distinct.")
    degree = min(n - 2, _MAX_DEGREE)
    x = scaled(f, (f.min(), f.max()))
    vander = np.vander(x, degree + 1, increasing=True)
    return np.linalg.pinv(vander), vander


def scaled(f: Any, band: tuple[float, float]) -> Any:
    """:math:`f` scaled so that the band ``(lo, hi)`` is :math:`[-1, 1]`."""
    lo, hi = band
    return (f - 0.5 * (lo + hi)) / (0.5 * (hi - lo))


def polynomial(coefficients: Any, f: Any, band: tuple[float, float]) -> Any:
    r"""The fitted quantity at ``f`` (Horner's rule in the scaled frequency).

    ``coefficients`` ``(d + 1,)`` in increasing powers of :math:`x`;
    ``f`` any shape. No bound is applied: the caller checks the span.
    """
    x = scaled(f, band)
    c = jnp.asarray(coefficients)
    value = c[-1] * jnp.ones_like(x)
    for k in range(c.shape[0] - 2, -1, -1):
        value = value * x + c[k]
    return value


# --- phase matching and pump -------------------------------------------------------


def _sinhc_series(w: Any) -> Any:
    r"""The series of :math:`\sinh w/w` in :math:`w^2`, for small :math:`|w|`."""
    w2 = w * w
    series = 0.0
    for c in reversed(_SINHC):
        series = series * w2 + c
    return series


def pair_amplitude(
    length: Any, mismatch: Any, alpha_pair: Any, alpha_pump: Any
) -> Any:
    r"""The coherent amplitude of the pairs born along the guide, in m.

    .. math::

        \int_0^L e^{-\alpha_p z - \alpha_{si}(L - z) + i\delta z}\,\mathrm{d}z
        = L\,e^{-h}\,e^{iv}\,\frac{\sinh w}{w},\quad
        w = u + iv,\ u = \frac{(\alpha_{si} - \alpha_p)L}{2},\
        v = \frac{\delta L}{2},\ h = \frac{(\alpha_{si} + \alpha_p)L}{2},

    with :math:`\alpha_{si}` the sum of the signal's and idler's field
    attenuations and :math:`\alpha_p` the pump's exponent (its attenuation in
    spontaneous down-conversion, twice that in four-wave mixing). Lossless it is
    :math:`L\,e^{iv}\,\mathrm{sinc}\,v`; its modulus squared is :math:`L^2`
    times the coincidence law of `gfem.three_wave.SecondHarmonic.efficiency`.
    Value and gradient are finite at :math:`\delta = 0` and no loss; the
    decaying exponentials are built so that none overflows for a passive guide
    (:math:`h \ge |u|`).
    """
    u = (alpha_pair - alpha_pump) * length / 2
    v = mismatch * length / 2
    half = (alpha_pair + alpha_pump) * length / 2
    w = u + 1j * v
    small = jnp.real(w * jnp.conj(w)) < _SERIES_BELOW
    series = _sinhc_series(w) * jnp.exp(-half)
    safe = jnp.where(small, 1.0, w)
    decayed = (jnp.exp(safe - half) - jnp.exp(-safe - half)) / (2 * safe)
    return length * jnp.exp(1j * v) * jnp.where(small, series, decayed)


def gaussian_width(duration: Any) -> Any:
    r""":math:`\tau` of :math:`P(t) = P_\mathrm{pk}e^{-t^2/\tau^2}` from its FWHM."""
    return duration / (2 * math.sqrt(math.log(2)))


def pump_spectrum(omega: Any, energy: Any, duration: Any) -> Any:
    r"""The pump envelope :math:`\tilde A_p` of a transform-limited Gaussian pulse.

    :math:`A(t) = \int\tilde A\,e^{-i\omega t}\mathrm{d}\omega` with
    :math:`|A(t)|^2 = P_\mathrm{pk}e^{-t^2/\tau^2}` of energy ``energy`` (J) and
    intensity FWHM ``duration`` (s):
    :math:`\tilde A = \sqrt{P_\mathrm{pk}}\,\tau/\sqrt{2\pi}\,
    e^{-\omega^2\tau^2/2}`, in :math:`\sqrt{\mathrm{W}}\,\mathrm{s}`, with
    :math:`2\pi\int|\tilde A|^2\mathrm{d}\omega` the pulse energy. ``omega`` is
    the offset from the pulse's carrier (rad/s).
    """
    tau = gaussian_width(duration)
    peak = energy / (math.sqrt(math.pi) * tau)
    return jnp.sqrt(peak) * tau / math.sqrt(2 * math.pi) * jnp.exp(-((omega * tau) ** 2) / 2)


def pump_square_spectrum(omega: Any, energy: Any, duration: Any) -> Any:
    r"""The spectrum of :math:`A_p(t)^2`, the pump's autoconvolution (W s).

    :math:`P_\mathrm{pk}\,\tau/(2\sqrt\pi)\,e^{-\omega^2\tau^2/4}` with no
    :math:`2\pi`, in the transform convention of `pump_spectrum`.
    """
    tau = gaussian_width(duration)
    peak = energy / (math.sqrt(math.pi) * tau)
    return peak * tau / (2 * math.sqrt(math.pi)) * jnp.exp(-((omega * tau) ** 2) / 4)


def poling_gain(duty_cycle: Any, order: Any) -> Any:
    r""":math:`G_m = \frac{2}{\pi m}\sin(\pi m D)` of a periodic domain reversal."""
    return 2 / (jnp.pi * order) * jnp.sin(jnp.pi * order * duty_cycle)


# --- figures -----------------------------------------------------------------------


def identical_scale(identical: bool) -> float:
    r"""The factor of a joint spectrum whose two photons are indistinguishable.

    The state of one mode family carries every pair twice in
    :math:`\iint_{\mathbb{R}^2}|F|^2` (as :math:`(\omega_s,\omega_i)` and its
    mirror), so its pairs are :math:`\tfrac12\iint|F|^2`: the amplitude carries
    :math:`1/\sqrt2`.
    """
    return 1 / math.sqrt(2) if identical else 1.0


def cw_rate(density: Any, step: float, identical: bool) -> Any:
    r"""Pairs per second :math:`\int n_s\,\mathrm{d}f_s` over the last axis.

    ``density`` is the signal's spectral density :math:`n_s` (pairs/s/Hz),
    zero outside the idler's band, on a uniform grid of spacing ``step`` (Hz);
    trapezoidal rule, halved for identical photons (each pair is two photons of
    the one density).
    """
    total = jnp.trapezoid(density, dx=step, axis=-1)
    return total / 2 if identical else total


def pair_count(jsa: Any, step_signal: float, step_idler: float) -> Any:
    r"""Pairs per pulse :math:`\sum|f|^2\,\delta f_s\,\delta f_i` of a joint spectrum."""
    return jnp.sum(jnp.abs(jsa) ** 2) * step_signal * step_idler


def trace_purity(jsa: Any) -> Any:
    r"""Heralded purity :math:`\mathrm{Tr}[(MM^\dagger)^2]/\mathrm{Tr}[MM^\dagger]^2`.

    ``jsa`` ``(Ns, Ni)`` on a uniform grid, where the quadrature weights
    cancel. It equals :math:`\sum\lambda_k^2` of the Schmidt weights without a
    singular value decomposition, whose derivative is singular at degenerate
    weights; :math:`\mathrm{Tr}[H^2] = \|H\|_F^2` for the Hermitian
    :math:`H = MM^\dagger`.
    """
    h = jsa @ jnp.conj(jsa).T
    return jnp.sum(jnp.abs(h) ** 2) / jnp.real(jnp.trace(h)) ** 2
