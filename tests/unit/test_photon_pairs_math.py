r"""The maths of photon-pair figures, with no finite elements.

Every check is a closed form, an identity of the formulas or a finite
difference of the pieces `gfem._photon_pairs` assembles into
`gfem.photon_pairs`: the dispersion fit, the lossy phase-matching amplitude,
the Gaussian pump, the continuous-wave rate and the trace purity.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.integrate
import scipy.special

from gfem import _photon_pairs as pp
from gfem.three_wave import _decayed_sinhc_sinc

EPS = np.finfo(float).eps
C = 299792458.0


# --- dispersion fit ----------------------------------------------------------------


def _rows(n, lo=190e12, hi=197e12):
    return np.linspace(lo, hi, n)


def _gamma_of(f, band, degree, seed=0):
    """A complex polynomial of the scaled frequency, with beta ~ 1e7 1/m."""
    rng = np.random.default_rng(seed)
    c = (rng.normal(size=degree + 1) + 1j * rng.normal(size=degree + 1)) * 10.0 ** (
        7 - 2 * np.arange(degree + 1)
    )
    x = pp.scaled(f, band)
    return c, sum(ck * x**k for k, ck in enumerate(c))


@pytest.mark.parametrize(("n", "degree"), [(4, 2), (5, 3), (6, 4), (9, 4), (12, 4)])
def test_the_fit_is_exact_on_a_polynomial_of_its_degree(n, degree):
    f = _rows(n)
    band = (f.min(), f.max())
    c, gamma = _gamma_of(f, band, degree)
    pinv, vander = pp.arm_basis(f)
    assert vander.shape == (n, degree + 1)
    coeffs = pinv @ gamma
    # the pseudo-inverse amplifies round-off by the condition number of the
    # Vandermonde matrix at most; 10 covers the sum over the rows
    tol = 10 * np.linalg.cond(vander) * EPS * np.abs(c).max()
    np.testing.assert_allclose(coeffs, c, rtol=0, atol=tol)
    # evaluated between and at the rows, and the residual vanishes with the fit
    off = np.linspace(f.min(), f.max(), 37)
    _, expected = _gamma_of(off, band, degree)
    got = pp.polynomial(jnp.asarray(coeffs), off, band)
    np.testing.assert_allclose(got, expected, rtol=0, atol=tol)
    assert np.abs(vander @ coeffs - gamma).max() <= tol


def test_the_fit_of_a_higher_degree_leaves_a_residual():
    f = _rows(9)
    band = (f.min(), f.max())
    _, gamma = _gamma_of(f, band, 6)
    pinv, vander = pp.arm_basis(f)
    residual = np.abs(vander @ (pinv @ gamma) - gamma).max()
    # the sixth-order part is 1e-5 times a normal deviate here; a quartic cannot
    # follow it, and the round-off of an exact fit is cond * eps * beta ~ 1e-7
    assert residual > 1e-5


def test_the_fit_is_linear_in_the_rows():
    pinv, _ = pp.arm_basis(_rows(7))
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, 7)) + 1j * rng.normal(size=(2, 7))
    np.testing.assert_allclose(
        pinv @ (2 * a - 3j * b),
        2 * (pinv @ a) - 3j * (pinv @ b),
        rtol=1e-12,
        atol=1e-13,
    )


def test_an_arm_needs_four_distinct_rows():
    with pytest.raises(ValueError, match=r"at least 4 frequency rows.*got 3"):
        pp.arm_basis(_rows(3))
    with pytest.raises(ValueError, match="distinct"):
        pp.arm_basis(np.array([1.0, 2.0, 2.0, 3.0]) * 1e14)


# --- phase-matching amplitude ------------------------------------------------------


@pytest.mark.parametrize("alphas", [(0.0, 0.0), (30.0, 12.0), (5.0, 40.0), (3.0, 3.0)])
def test_the_amplitude_is_the_z_integral_and_its_square_the_decayed_law(alphas):
    a_pair, a_pump = alphas
    length = 0.02
    for delta in (0.0, 80.0, -220.0, 1500.0):
        amp = complex(pp.pair_amplitude(length, delta, a_pair, a_pump))

        def integrand(z, delta=delta):
            return np.exp(-a_pump * z - a_pair * (length - z) + 1j * delta * z)

        re = scipy.integrate.quad(lambda z: integrand(z).real, 0, length, limit=500)
        im = scipy.integrate.quad(lambda z: integrand(z).imag, 0, length, limit=500)
        # quad's own estimate is 1e-14 relative to the integral's size
        np.testing.assert_allclose(amp, re[0] + 1j * im[0], rtol=1e-9, atol=1e-13)
        u = (a_pair - a_pump) * length / 2
        v = delta * length / 2
        h = (a_pair + a_pump) * length / 2
        law = length**2 * float(_decayed_sinhc_sinc(u, v, h))
        np.testing.assert_allclose(abs(amp) ** 2, law, rtol=1e-12)


def test_the_lossless_amplitude_is_the_sinc_with_its_phase():
    length, delta = 0.01, 313.0
    v = delta * length / 2
    got = complex(pp.pair_amplitude(length, delta, 0.0, 0.0))
    np.testing.assert_allclose(
        got, length * np.exp(1j * v) * np.sinc(v / np.pi), rtol=1e-14
    )


def test_the_amplitude_and_its_gradient_are_finite_at_zero_mismatch_and_loss():
    length = 0.01

    def parts(delta, a_pair, a_pump):
        amp = pp.pair_amplitude(length, delta, a_pair, a_pump)
        return jnp.real(amp), jnp.imag(amp)

    value = pp.pair_amplitude(length, 0.0, 0.0, 0.0)
    assert complex(value) == pytest.approx(length, rel=1e-15)
    # d/d delta = i L^2/2 and d/d alpha = -L^2/2 for the signal-idler and the
    # pump decay (the z integral of z or L - z)
    d = jax.jacfwd(parts, argnums=(0, 1, 2))(0.0, 0.0, 0.0)
    want = np.array(
        [[0.0, -(length**2) / 2, -(length**2) / 2], [length**2 / 2, 0.0, 0.0]]
    )
    got = np.array([[float(x) for x in d[0]], [float(x) for x in d[1]]])
    # the series is exact to 1e-17 relative at the origin
    np.testing.assert_allclose(got.T, want.T, rtol=1e-13, atol=1e-15)
    # no overflow of a lossy long guide (sinh alone overflows past 710)
    big = pp.pair_amplitude(1.0, 10.0, 800.0, 1.0)
    assert np.isfinite(complex(big))
    g = jax.grad(lambda a: jnp.abs(pp.pair_amplitude(1.0, 10.0, a, 1.0)))(800.0)
    assert np.isfinite(float(g))


# --- Gaussian pump -----------------------------------------------------------------


def test_the_pump_spectrum_carries_the_pulse_energy_and_the_fwhm():
    energy, duration = 3e-12, 2e-12
    tau = duration / (2 * math.sqrt(math.log(2)))
    omega = np.linspace(-12 / tau, 12 / tau, 4001)
    a = np.asarray(pp.pump_spectrum(omega, energy, duration))
    # 2 pi int |A|^2 domega = U: a Gaussian on a grid 12 widths wide is exact
    # to the trapezoid's spectral accuracy (1e-15 relative)
    np.testing.assert_allclose(
        2 * np.pi * np.trapezoid(a**2, omega), energy, rtol=1e-12
    )

    def power(t):
        field = np.trapezoid(a * np.exp(-1j * omega * t), omega)
        return abs(field) ** 2

    peak = energy / (math.sqrt(math.pi) * tau)
    np.testing.assert_allclose(power(0.0), peak, rtol=1e-12)
    np.testing.assert_allclose(power(duration / 2), peak / 2, rtol=1e-12)


def test_the_square_spectrum_is_the_autoconvolution_of_the_envelope():
    energy, duration = 1e-12, 4e-12
    tau = duration / (2 * math.sqrt(math.log(2)))
    w = np.linspace(-14 / tau, 14 / tau, 5601)
    a = np.asarray(pp.pump_spectrum(w, energy, duration))
    for shift in (0.0, 0.7 / tau, -2.3 / tau):
        conv = np.trapezoid(
            a * np.asarray(pp.pump_spectrum(shift - w, energy, duration)), w
        )
        np.testing.assert_allclose(
            conv, float(pp.pump_square_spectrum(shift, energy, duration)), rtol=1e-12
        )


# --- continuous-wave rate ----------------------------------------------------------


def test_the_cw_rate_with_a_linear_mismatch_is_the_sinc_squared_integral():
    length, slope = 5e-3, 0.3 / C  # s/m: delta = slope * 2 pi (f - f0)
    half_band = 1.5e12
    n = 4001
    f = np.linspace(-half_band, half_band, n)
    step = f[1] - f[0]
    delta = slope * 2 * np.pi * f
    amp = pp.pair_amplitude(length, jnp.asarray(delta), 0.0, 0.0)
    density = jnp.abs(amp) ** 2  # g = P = 1
    rate = float(pp.cw_rate(density, step, identical=False))
    # int sinc^2(a f) df over [-B, B] = (2/a) (Si(2X) - sin^2 X / X), X = a B,
    # a = slope * pi * L; the density is L^2 sinc^2
    a = slope * math.pi * length
    x = a * half_band
    closed = length**2 * 2 / a * (scipy.special.sici(2 * x)[0] - math.sin(x) ** 2 / x)
    # trapezoid: h^2/12 |f'(b) - f'(a)| <= h^2/6 max|f'|, |d sinc^2/dx| <= 0.45
    # (here f' = L^2 a 0.45 at most), margin 2
    bound = 2 * step**2 / 6 * length**2 * a * 0.45
    assert abs(rate - closed) <= bound
    # and the band is long enough for the tail 1/(2 a^2 B) to be what is left
    np.testing.assert_allclose(rate, length**2 * math.pi / a, rtol=2 / (math.pi * x))


def test_identical_photons_count_each_pair_once():
    density = jnp.linspace(1.0, 2.0, 50)
    both = float(pp.cw_rate(density, 1e9, identical=True))
    assert both == pytest.approx(0.5 * float(pp.cw_rate(density, 1e9, identical=False)))
    # a symmetric amplitude summed over the full square counts every off-diagonal
    # pair twice: the 1/sqrt(2) scale is the half-plane sum
    rng = np.random.default_rng(3)
    m = rng.normal(size=(20, 20)) + 1j * rng.normal(size=(20, 20))
    sym = m + m.T
    full = float(pp.pair_count(pp.identical_scale(True) * sym, 1.0, 1.0))
    half_plane = np.sum(np.abs(np.triu(sym, 1)) ** 2) + 0.5 * np.sum(
        np.abs(np.diag(sym)) ** 2
    )
    np.testing.assert_allclose(full, half_plane, rtol=1e-13)
    assert pp.identical_scale(False) == 1.0


# --- purity ------------------------------------------------------------------------


#: the Gaussian stand-in of the sinc, sinc(u) ~ exp(-gamma u^2); the literals of
#: the purity (0.8965208985 at a pump width 2, separable at sqrt(2/gamma)) are
#: those of gamma = 0.193, the usual rounding of 0.1929214
_GAMMA = 0.193


def _gaussian_jsa(sigma, gamma=_GAMMA, n=401, half=20.0):
    """Gaussian pump of amplitude width ``sigma`` times :math:`e^{-\\gamma u^2}`.

    Units of :math:`L|a| = 1`, antisymmetric group-velocity mismatch (``u`` is
    half the difference of the offsets); the pump envelope is the one of
    `pump_spectrum` (duration chosen so :math:`1/\\tau = \\sigma`).
    """
    x = np.linspace(-half, half, n)
    xs, xi = np.meshgrid(x, x, indexing="ij")
    tau = 1.0 / sigma
    duration = tau * 2 * math.sqrt(math.log(2))
    pump = np.asarray(pp.pump_spectrum(xs + xi, 1.0, duration))
    return pump * np.exp(-gamma * ((xs - xi) / 2) ** 2), gamma


def test_the_gaussian_jsa_purity_is_its_closed_form_and_the_literal():
    gamma = _GAMMA
    sigma = 2.0
    jsa, _ = _gaussian_jsa(sigma, gamma)
    a11 = 1 / sigma**2 + gamma / 2
    a12 = 1 / sigma**2 - gamma / 2
    closed = math.sqrt(1 - a12**2 / a11**2)
    got = float(pp.trace_purity(jsa))
    assert closed == pytest.approx(0.8965208985, abs=1e-10)
    # a Gaussian sampled at 10 points per pump width on a grid out to
    # exp(-gamma 20^2 / 4) = 1e-14: the discrete sums are
    # spectrally accurate, 1e-10 measured in the probe
    assert got == pytest.approx(closed, abs=1e-9)
    assert float(1 / pp.trace_purity(jsa)) == pytest.approx(1 / closed, rel=1e-9)


def test_the_gaussian_jsa_is_separable_at_the_matched_pump():
    gamma = _GAMMA
    jsa, _ = _gaussian_jsa(math.sqrt(2 / gamma), gamma, n=321, half=14.0)
    assert float(pp.trace_purity(jsa)) == pytest.approx(1.0, abs=1e-9)


def test_the_trace_purity_is_the_svd_purity_of_a_complex_spectrum():
    rng = np.random.default_rng(7)
    x = np.linspace(-3, 3, 70)
    xs, xi = np.meshgrid(x, x[:55], indexing="ij")
    jsa = np.exp(-0.5 * (1.3 * xs**2 + 1.1 * xs * xi + 0.8 * xi**2)) * np.exp(
        0.4j * xs * xi
    )
    jsa = jsa + 0.05 * (rng.normal(size=jsa.shape) + 1j * rng.normal(size=jsa.shape))
    s = np.linalg.svd(jsa, compute_uv=False)
    lam = s**2 / np.sum(s**2)
    # two matrix products and a sum of 70 x 70 squares against an SVD: a few
    # hundred eps
    np.testing.assert_allclose(float(pp.trace_purity(jsa)), np.sum(lam**2), rtol=1e-13)


def test_the_trace_purity_has_a_finite_gradient_at_a_separable_state():
    x = np.linspace(-3, 3, 31)
    base = np.exp(-(x[:, None] ** 2) - x[None, :] ** 2)

    def purity(scale):
        return pp.trace_purity(base * (1 + scale * np.outer(x, x)))

    g = float(jax.grad(purity)(0.0))
    assert np.isfinite(g)
    # purity <= 1 and the separable state is its maximum: the slope is 0
    assert abs(g) < 1e-12


# --- gradients and jit of a composite spectrum -------------------------------------

_B = 1.0e12  # half band, Hz
_N = 81
_FS = np.linspace(-_B, _B, _N)
_KS, _KI = 1.0e-9, -0.9e-9  # s/m: (beta_p' - beta_s'), (beta_p' - beta_i')
_PERIOD0 = 30e-6


def _spectrum(length, duration, period):
    xs, xi = np.meshgrid(_FS, _FS, indexing="ij")
    delta = (
        2 * np.pi * (_KS * xs + _KI * xi) + 2 * np.pi / _PERIOD0 - 2 * np.pi / period
    )
    pump = pp.pump_spectrum(2 * np.pi * (xs + xi), 1e-12, duration)
    gain = pp.poling_gain(0.5, 1)
    return 2e3 * gain * pump * pp.pair_amplitude(length, delta, 3.0, 2.0) * 2 * np.pi


def _figures(length, duration, period):
    jsa = _spectrum(length, duration, period)
    step = _FS[1] - _FS[0]
    return jnp.stack([pp.pair_count(jsa, step, step), pp.trace_purity(jsa)])


_ARGS = (5e-3, 1e-12, _PERIOD0 * 1.0003)


@pytest.mark.parametrize("which", [0, 1, 2], ids=["length", "duration", "period"])
def test_the_figures_differentiate_like_their_finite_differences(which):
    x = np.array(_ARGS)
    jac = np.asarray(jax.jacfwd(_figures, argnums=which)(*_ARGS))
    # central difference at a relative step r: truncation (h^2/6) f'''/f' with
    # h f'''/f' ~ the phase the step adds, and round-off eps / r on each value.
    # Length and duration move the phase by ~1e2 r rad: r = 1e-5 gives
    # truncation ~1e-7 and round-off 1e-11. The period moves the 5e2-rad poling
    # phase by 5e2 r: r = 1e-7 gives 1e-9 and 1e-9
    h = (1e-5, 1e-5, 1e-7)[which] * x[which]
    up, dn = np.array(_ARGS), np.array(_ARGS)
    up[which] += h
    dn[which] -= h
    fd = (np.asarray(_figures(*up)) - np.asarray(_figures(*dn))) / (2 * h)
    np.testing.assert_allclose(jac, fd, rtol=1e-6, atol=0)
    assert np.all(np.isfinite(jac))
    assert np.abs(jac).min() > 0


def test_the_jitted_figures_are_the_eager_figures():
    eager = np.asarray(_figures(*_ARGS))
    jitted = np.asarray(jax.jit(_figures)(*_ARGS))
    # XLA may fuse and reorder the sums: a few hundred eps at most
    np.testing.assert_allclose(jitted, eager, rtol=1e-12)
    grad = jax.jit(jax.jacfwd(_figures, argnums=0))(*_ARGS)
    np.testing.assert_allclose(
        np.asarray(grad),
        np.asarray(jax.jacfwd(_figures, argnums=0)(*_ARGS)),
        rtol=1e-10,
    )
