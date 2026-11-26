r"""The photon-pair figures of a port-mode guide converge to closed forms.

A homogeneously filled perfectly conducting rectangle $a \times b$ carries TE10
at every frequency with $\beta^2 = n^2\omega^2/c^2 - (\pi/a)^2$ exactly, and a
single tensor entry ($d_{yyy}$ or a Kerr index) gives every overlap in closed
form. Four tests hold the figures of `gfem.spdc` and `gfem.sfwm` to it:

- the degenerate type-0 continuous-wave rate (identical photons, first-order
  quasi-phase matching, duty cycle 1/2) is the integral of
  $n_s = \frac{4\omega_s\omega_i}{\omega_p^2}|G|^2\eta_\mathrm{norm}PL^2
  \mathrm{sinc}^2(\delta L/2)$, $|G| = 2/\pi$, over the signal band. With
  $\delta \simeq -\beta_2\Omega^2$ and the band taken to infinity it is
  $\frac{\sqrt2}{3\sqrt\pi}|G|^2\eta_\mathrm{norm}P\,L^{3/2}/\sqrt{|\beta_2|}$
  ($8.44427859\times10^{10}$ pairs/s for $a = 2$ um, $b = 1$ um, $n = 2.2$,
  $d = 20$ pm/V, 1550 nm, 1 mW, 1 cm), from $\int\mathrm{sinc}^2(x^2)\mathrm dx
  = 4\sqrt\pi/3$;
- the same figure for four-wave mixing,
  $\frac{\sqrt2}{3\sqrt\pi}|\gamma_\mathrm{NL}|^2P^2L^{3/2}/\sqrt{|\beta_2|}$
  ($3.45618916\times10^9$ pairs/s for $n = 1.5$, $n_2 = 2.6\times10^{-20}$
  m$^2$/W, 1 W, 1 m), whose continuous-wave mismatch also carries the
  nonlinear phase $-2\gamma_\mathrm{NL}P$;
- the peak density at the band centre is $|G|^2\eta_\mathrm{norm}PL^2$ with
  `gfem.second_harmonic`'s $\eta_\mathrm{norm}$ and no free factor (the
  classical-quantum identity), and a pulse with a linear mismatch carries
  $N = (R/P)\,U_p$ pairs;
- the derivative of the rate in the permittivity and of the purity in the
  pulse duration match fourth-order differences.

The figures are integrals over a band of the signal, so the closed forms of the
infinite line lose the tail the band cuts. Where that tail is not the point the
reference is the quadrature of the same integrand over the same band with the
exact dispersion, the fit of the arms (a degree-4 polynomial of the rows) and
the coupling at the band centre, the three things the source takes from the
solves; the element order then refines only the solves' error.
"""

import dataclasses
import itertools
import math
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as sc
from finite_differences import assert_gradient_matches
from scipy import integrate, special

import gfem

ORDERS = (2, 3, 4)
DATA = Path(__file__).parents[1] / "unit" / "data"

C = sc.c
LAMBDA = 1.55e-6
F0 = C / LAMBDA
W0 = 2 * math.pi * F0
#: the closed-form rectangle a x b = 2 um x 1 um; the WR-90 mesh is scaled in
#: x, y, z to this width, so its b is 8/9 um and every coupling carries 1/b
A_REF, B_REF = 2e-6, 1e-6
WR90_A, WR90_B = 22.86e-3, 10.16e-3
SCALE = A_REF / WR90_A
B_MESH = WR90_B * SCALE

# --- down-conversion --------------------------------------------------------------

N_CHI2 = 2.2
D_YYY = 20e-12
LENGTH, POWER = 1e-2, 1e-3
#: the literal (b = 1 um) rate of the beta_2 law over the whole line, pairs/s
RATE_SPDC_B1UM = 8.44427859e10
#: signal = idler band half-width and rows, Hz: +-20 THz is 10 % of the carrier
#: and 15 rad of beta_2 phase at the edge; 9 rows give a degree-4 fit
HALF_BAND = 20e12
ROWS = 9
POINTS = 801

# --- four-wave mixing -------------------------------------------------------------

N_KERR, N2_KERR = 1.5, 2.6e-20
POWER_KERR, LENGTH_KERR = 1.0, 1.0
#: gamma_NL (1/(W m)) at b = 1 um and the beta_2-law rate of the whole line
GAMMA_B1UM = 8.469900719e-2
RATE_SFWM_B1UM = 3.45618916e9
#: +-3 THz is 55 rad of beta_2 phase at the edge at 1 m; 1601 points make the
#: trapezoid step 3.75 GHz, 0.5 rad of the sinc argument per step at the edge
HALF_BAND_KERR = 3e12
ROWS_KERR = 7
POINTS_KERR = 1601

Z0 = math.sqrt(sc.mu_0 / sc.epsilon_0)


def _converges(errors, floor):
    """Each order 5x below the last, the last under ``floor``."""
    assert all(e2 < e1 / 5 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < floor, errors


def _mesh():
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    nodes = jnp.asarray(np.asarray(mesh.imported_nodes) * SCALE)
    return jax.tree.unflatten(jax.tree.structure(mesh), [nodes])


def _guide(material):
    return gfem.Model(
        _mesh(),
        materials={"air": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def _solve(model, frequency, order):
    return gfem.solve(
        model,
        gfem.PortMode("p1", frequency),
        options=gfem.SolverOptions(order=order),
    )


def _beta(w, n):
    """The exact TE10 propagation constant of the filled rectangle."""
    return np.sqrt((n * w / C) ** 2 - (math.pi / A_REF) ** 2)


def _fitted_beta(rows, n):
    """The degree-4 least-squares polynomial the source fits to ``beta`` on its rows."""
    return np.polynomial.Polynomial.fit(rows, _beta(2 * math.pi * rows, n), 4)


def _eta_closed(b):
    """eta_norm (1/(W m^2)) of the filled TE10 rectangle, d_yyy alone."""
    k0 = 2 * math.pi / LAMBDA
    n_p = math.sqrt(N_CHI2**2 - (LAMBDA / (2 * A_REF)) ** 2)
    n_h = math.sqrt(N_CHI2**2 - (LAMBDA / (4 * A_REF)) ** 2)
    return 256 * k0**2 * D_YYY**2 * Z0 / (9 * math.pi**2 * A_REF * b * n_p**2 * n_h)


def _gamma_closed(b):
    """gamma_NL (1/(W m)) of the filled TE10 rectangle at the carrier, n = 1.5."""
    n_eff2 = N_KERR**2 - (LAMBDA / (2 * A_REF)) ** 2
    return 1.5 * (W0 / C) * N2_KERR / (A_REF * b) * N_KERR**2 / n_eff2


def _beta2(n):
    """beta_2 (s^2/m) of the TE10 at the carrier."""
    beta = _beta(W0, n)
    return -(n**2 / (C**2 * beta)) * (math.pi / A_REF) ** 2 / beta**2


def _quad(density, half_band):
    """The rate (pairs/s) of a density(Omega) in pairs/s/Hz over both offset signs."""
    x_max = 2 * math.pi * half_band
    # the lobes of sinc^2 are many: split the range so quad meets each
    knots = np.linspace(0, x_max, 800)[1:-1]
    half = integrate.quad(density, 0, x_max, limit=8000, points=knots)[0]
    return 2 * half / (2 * math.pi)


# --- degenerate type-0 down-conversion --------------------------------------------


@pytest.fixture(scope="module")
def chi2_model():
    d = np.zeros((3, 6))
    d[1, 1] = D_YYY
    return _guide(gfem.Material(eps_r=N_CHI2**2, second_order_nonlinear_tensor=d))


@pytest.fixture(scope="module")
def spdc_rows():
    return np.linspace(F0 - HALF_BAND, F0 + HALF_BAND, ROWS)


@pytest.fixture(scope="module")
def spdc_sources(chi2_model, spdc_rows):
    """The degenerate source at each order: one band is signal and idler."""
    out = {}
    for p in ORDERS:
        band = _solve(chi2_model, spdc_rows, p)
        # the pump is read at the sum of the band centres only: 5 rows over 4 THz
        pump = _solve(chi2_model, np.linspace(2 * F0 - 2e12, 2 * F0 + 2e12, 5), p)
        out[p] = gfem.spdc(chi2_model, pump, band, band, points=POINTS)
    return out


def _spdc_rate_reference(rows, eta):
    r"""The rate (pairs/s) of exact formula on the source's band, ``eta`` in 1/(W m^2).

    The mismatch is the fitted polynomial's relative to the centre
    (:math:`\Delta\beta_0` is taken out by the poling), the prefactor
    :math:`4\omega_s\omega_i/\omega_p^2` is on the grid and the pump is at the sum
    of the centres, as the source evaluates them.
    """
    fit = _fitted_beta(rows, N_CHI2)

    def density(omega):
        ws, wi = W0 + omega, W0 - omega
        db = 2 * fit(F0) - fit(ws / (2 * math.pi)) - fit(wi / (2 * math.pi))
        gain = (2 / math.pi) ** 2 * 4 * ws * wi / (ws + wi) ** 2
        return gain * eta * POWER * LENGTH**2 * np.sinc(db * LENGTH / 2 / math.pi) ** 2

    # identical photons: half the integral over the line
    return 0.5 * _quad(density, HALF_BAND)


def test_the_degenerate_spdc_rate_converges_to_the_exact_quadrature(
    spdc_sources, spdc_rows
):
    eta = _eta_closed(B_MESH)
    reference = _spdc_rate_reference(spdc_rows, eta)
    errors = []
    for p in ORDERS:
        src = spdc_sources[p]
        assert src.identical
        rate = float(src.rate(LENGTH, POWER, period=src.poling_period))
        errors.append(abs(rate / reference - 1))
    # measured relative errors 5.7e-5, 2.0e-7, 9.8e-9 at orders 2-4: the
    # solves' error of beta_2 and of the overlap (a field error squared, as
    # eta_norm's 5.7e-5, 1.7e-7, 1.5e-9). The last sits on the trapezoid rule
    # of the grid, whose error scales as the step squared (SFWM below: 1.1e-7 at
    # 801 points, 2.6e-8 at 1601), so the floor is 5e-8
    _converges(errors, 5e-8)


def test_the_degenerate_spdc_rate_meets_the_literal_up_to_the_cut_tail(spdc_sources):
    eta = _eta_closed(B_MESH)
    b2 = _beta2(N_CHI2)
    x_max = 2 * math.pi * HALF_BAND
    # the literal is for b = 1 um and goes as 1/b
    literal = RATE_SPDC_B1UM * B_REF / B_MESH
    gain = 0.5 * (2 / math.pi) ** 2 * eta * POWER * LENGTH**2 / (2 * math.pi)

    def b2_law(omega):
        return np.sinc(-b2 * omega**2 * LENGTH / 2 / math.pi) ** 2

    knots = np.linspace(0, x_max, 800)[1:-1]
    inside = 2 * gain * integrate.quad(b2_law, 0, x_max, limit=8000, points=knots)[0]
    # the mean of sinc^2 beyond the band: 1/(2 x^2) with x = |beta_2| Omega^2 L/2
    tail = 2 * gain * 2 / (3 * b2**2 * LENGTH**2 * x_max**3)
    # the beta_2 law and its tail make the literal, to the oscillation of the tail
    assert (inside + tail) / literal == pytest.approx(1, abs=1e-4)
    src = spdc_sources[ORDERS[-1]]
    rate = float(src.rate(LENGTH, POWER, period=src.poling_period))
    deficit = 1 - rate / literal
    # the band cuts the tail (2.3e-3 of the literal), and the exact mismatch
    # differs from the beta_2 law by the beta_4 terms (1.3e-3, over 14 lobes in
    # the research); the sum, 3.6e-3, bounds the measured deficit from above
    assert 0 < deficit < 4e-3, deficit


def test_the_peak_density_is_the_classical_conversion_efficiency_of_the_same_modes(
    chi2_model, spdc_sources
):
    order = ORDERS[1]
    src = spdc_sources[order]
    pump, harmonic = (_solve(chi2_model, [f], order) for f in (F0, 2 * F0))
    eta = float(gfem.second_harmonic(chi2_model, pump, harmonic).eta_norm[0])
    density = src.spectral_brightness(LENGTH, POWER, period=src.poling_period)
    middle = POINTS // 2
    assert float(src.signal_frequency[middle]) == pytest.approx(F0, rel=1e-15)
    # n = |G|^2 eta_norm P L^2, |G| = 2/pi at first order and D = 1/2, with no
    # free factor. Both are the same overlap sum on two solves of the same row:
    # the eigensolve noise of the fields, 1e-13 measured; the poling period of
    # the fits puts the mismatch at round-off, sinc^2 = 1 - 1e-27. 1e-9 gates it.
    assert float(density[middle]) == pytest.approx(
        (2 / math.pi) ** 2 * eta * POWER * LENGTH**2, rel=1e-9
    )


# --- degenerate four-wave mixing --------------------------------------------------


@pytest.fixture(scope="module")
def kerr_model():
    return _guide(gfem.Material(eps_r=N_KERR**2, nonlinear_index=N2_KERR))


@pytest.fixture(scope="module")
def kerr_rows():
    return np.linspace(F0 - HALF_BAND_KERR, F0 + HALF_BAND_KERR, ROWS_KERR)


@pytest.fixture(scope="module")
def sfwm_sources(kerr_model, kerr_rows):
    """Degenerate four-wave mixing at each order: one band is pump, signal, idler."""
    out = {}
    for p in ORDERS:
        band = _solve(kerr_model, kerr_rows, p)
        out[p] = gfem.sfwm(kerr_model, band, band, band, points=POINTS_KERR)
    return out


def _sfwm_rate_reference(rows, gamma, nonlinear_phase):
    r"""The rate (pairs/s) of exact formula on the source's band, ``gamma`` in 1/(W m).

    :math:`\Gamma_{si} = \tfrac12(\sqrt{\omega_i/\omega_s} +
    \sqrt{\omega_s/\omega_i})\gamma` for one mode, the mismatch the fitted
    :math:`2\beta_p - \beta_s - \beta_i` with the pump at the carrier, and
    :math:`2(\gamma_{pppp} - \gamma_{sspp} - \gamma_{iipp})P = -2\gamma P`
    of one mode when ``nonlinear_phase``.
    """
    fit = _fitted_beta(rows, N_KERR)
    kerr = -2 * gamma * POWER_KERR if nonlinear_phase else 0.0

    def density(omega):
        ws, wi = W0 + omega, W0 - omega
        db = 2 * fit(F0) - fit(ws / (2 * math.pi)) - fit(wi / (2 * math.pi)) + kerr
        pair = 0.5 * (math.sqrt(wi / ws) + math.sqrt(ws / wi)) * gamma
        return (pair * POWER_KERR * LENGTH_KERR) ** 2 * np.sinc(
            db * LENGTH_KERR / 2 / math.pi
        ) ** 2

    return 0.5 * _quad(density, HALF_BAND_KERR)


def test_the_degenerate_sfwm_rate_converges_to_the_exact_quadrature(
    sfwm_sources, kerr_rows
):
    gamma = _gamma_closed(B_MESH)
    # the literal is for b = 1 um and goes as 1/b
    assert gamma * B_MESH / B_REF == pytest.approx(GAMMA_B1UM, rel=1e-9)
    kerr, plain = [], []
    for p in ORDERS:
        src = sfwm_sources[p]
        assert src.identical
        # the phase of the self- and cross-phase modulation of 1 W, and without it
        zero = dataclasses.replace(src, kerr_phase=jnp.zeros_like(src.kerr_phase))
        for source, errors, phase in ((src, kerr, True), (zero, plain, False)):
            rate = float(source.rate(LENGTH_KERR, POWER_KERR))
            reference = _sfwm_rate_reference(kerr_rows, gamma, phase)
            errors.append(abs(rate / reference - 1))
    # measured relative errors 1.5e-5, 2.7e-7, 2.6e-8 at orders 2-4 with the
    # phase (1.4e-5, 2.7e-7, 2.8e-8 without): the field error of gamma and of
    # beta_2, then the trapezoid rule of the grid, 2.6e-8 at 1601 points and
    # 1.1e-7 at 801 (as the step squared); the floor is 1e-7
    _converges(kerr, 1e-7)
    _converges(plain, 1e-7)


def test_the_degenerate_sfwm_rate_meets_the_literal_up_to_the_cut_tail(sfwm_sources):
    src = sfwm_sources[ORDERS[-1]]
    zero = dataclasses.replace(src, kerr_phase=jnp.zeros_like(src.kerr_phase))
    literal = RATE_SFWM_B1UM * (B_REF / B_MESH) ** 2
    rate = float(zero.rate(LENGTH_KERR, POWER_KERR))
    b2 = _beta2(N_KERR)
    x_max = 2 * math.pi * HALF_BAND_KERR
    # the mean of sinc^2 beyond the band, 1/(2 x^2) with x = |beta_2| Omega^2 L/2,
    # as a fraction of the line's rate; the literal has no nonlinear phase
    tail = 2 * 2 / (3 * b2**2 * LENGTH_KERR**2 * x_max**3)
    line = (4 * math.sqrt(math.pi) / 3) * math.sqrt(2 / (abs(b2) * LENGTH_KERR))
    fraction = tail / line
    # the sinc^2 of the tail oscillates about its mean by a relative 1.5/x_edge
    # ~ 3 % of the tail (3.5e-4), 1e-5; the trapezoid, the polynomial fit of
    # beta and the field of the solve add 1e-6 (measured 7e-6 in all); 2e-5
    assert rate / literal == pytest.approx(1 - fraction, abs=2e-5)
    # the self- and cross-phase modulation of 1 W moves the rate by 4 %: the
    # literal is the figure of a guide without it
    with_phase = float(src.rate(LENGTH_KERR, POWER_KERR))
    assert with_phase / rate > 1.03


# --- a pulse with a linear mismatch -----------------------------------------------

#: the non-degenerate bands (1.4 and 1.7 um), the half-width of each, Hz
SIGNAL, IDLER = C / 1.4e-6, C / 1.7e-6
HALF_PULSE_BAND = 1.5e12
#: group indices of the synthetic linear arms: the pump's is their mean, so the
#: mismatch depends on Omega_s - Omega_i only
N_G_SIGNAL, N_G_IDLER = 2.2, 2.6
PULSE_ENERGY, PULSE_DURATION = 3e-15, 5e-12
LENGTH_PULSE = 1e-2


def _linear_arm(beta0, group_index, band):
    """Fit coefficients of ``gamma = i (beta0 + n_g/c omega)`` over ``band`` (Hz)."""
    half = (band[1] - band[0]) / 2
    return jnp.asarray(
        [1j * beta0, 1j * 2 * math.pi * group_index / C * half, 0, 0, 0],
        dtype=complex,
    )


@pytest.fixture(scope="module")
def linear_source(chi2_model):
    """A down-conversion source of the rectangle's coupling and linear arms."""
    rows = [
        np.linspace(c - h, c + h, 4)
        for c, h in (
            (SIGNAL + IDLER, 2 * HALF_PULSE_BAND),
            (SIGNAL, HALF_PULSE_BAND),
            (IDLER, HALF_PULSE_BAND),
        )
    ]
    pump, signal, idler = (_solve(chi2_model, r, 2) for r in rows)
    src = gfem.spdc(chi2_model, pump, signal, idler, points=401)
    n_pump = 0.5 * (N_G_SIGNAL + N_G_IDLER)
    # the zero-order terms cancel in the mismatch: beta_p = beta_s + beta_i
    fit = (
        _linear_arm(2.2e7, n_pump, src.bands[0]),
        _linear_arm(1.0e7, N_G_SIGNAL, src.bands[1]),
        _linear_arm(1.2e7, N_G_IDLER, src.bands[2]),
    )
    return dataclasses.replace(src, fit=fit, fit_residual=jnp.zeros(3))


def test_a_pulse_with_a_linear_mismatch_carries_the_rate_per_watt_times_its_energy(
    linear_source,
):
    src = linear_source
    ws, wi = 2 * math.pi * SIGNAL, 2 * math.pi * IDLER
    kappa2 = abs(complex(src.coupling)) ** 2
    # R/P = g^2 L c/|n_gs - n_gi| for the mismatch (n_gs - n_gi) (Omega_s -
    # Omega_i)/(2c) (the integral of sinc^2 over a line is pi), g at the centres
    per_watt = (
        (2 * math.sqrt(ws * wi) / (ws + wi)) ** 2
        * kappa2
        * LENGTH_PULSE
        * C
        / abs(N_G_SIGNAL - N_G_IDLER)
    )
    # the band cuts the sinc^2 tails: a fraction 1/(pi X) of the pairs, X the
    # argument of the sinc at the edge of the difference coordinate
    x_edge = (
        abs(N_G_IDLER - N_G_SIGNAL)
        / (4 * C)
        * 2
        * (2 * math.pi * 2 * HALF_PULSE_BAND)
        * LENGTH_PULSE
        / 2
    )
    # exactly, 2 int_X^inf sin^2 x/x^2 dx = 2 [sin^2 X/X + pi/2 - Si(2X)], over pi
    cut = (2 / math.pi) * (
        math.sin(x_edge) ** 2 / x_edge
        + math.pi / 2
        - float(special.sici(2 * x_edge)[0])
    )
    expected = per_watt * (1 - cut)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pulse = src.joint_spectrum(
            LENGTH_PULSE, pulse_energy=PULSE_ENERGY, pulse_duration=PULSE_DURATION
        )
    n_pairs = float(pulse.pairs_per_pulse)
    # the cut is exact; what is left is the sum over the grid, which counts the
    # edge rows of the band in full where the integral would count half (the
    # sinc^2 there is 1/X^2 of its peak, a relative 1e-5 in all, 1.8e-5
    # measured), and the grid's prefactor 4 w_s w_i/w_p^2, whose variation over
    # the cut tail of 5e-3 is below 1e-6; 4e-5 gates them
    assert n_pairs / (expected * PULSE_ENERGY) == pytest.approx(1, abs=4e-5)
    # the continuous-wave rate of the same arms is the same figure per watt
    watt = 1e-3
    rate = float(src.rate(LENGTH_PULSE, watt)) / watt
    # the trapezoid of the rate is the integral up to its end terms, the sinc^2
    # of the edge, 1/X^2 of the peak per step: 2e-7 measured, 2e-6 gates it
    assert rate / expected == pytest.approx(1, abs=2e-6)


# --- gradients --------------------------------------------------------------------


def _scaled_rate(scale, reference=1.0):
    """The rate of a degenerate source of permittivity ``scale`` n^2, relative."""
    d = np.zeros((3, 6))
    d[1, 1] = D_YYY
    model = _guide(
        gfem.MaterialField(eps_r=N_CHI2**2 * scale, second_order_nonlinear_tensor=d)
    )
    band = _solve(model, np.linspace(F0 - 5e12, F0 + 5e12, 4), 2)
    pump = _solve(model, np.linspace(2 * F0 - 2e12, 2 * F0 + 2e12, 4), 2)
    src = gfem.spdc(model, pump, band, band, points=101)
    return src.rate(LENGTH, POWER, period=src.poling_period) / reference


def test_the_gradient_of_the_rate_in_the_permittivity_matches_differences():
    s0 = jnp.asarray(1.0)
    reference = float(_scaled_rate(s0))
    # the adjoint through the solves, the fits and the poling period (which
    # follows the permittivity: the mismatch stays zero at the centre) against
    # the fourth-order difference of exact solves of the same discretization.
    # Step 1e-2 as for a permittivity: the rate responds as eta_norm and
    # beta_2^(-1/2), a scale of order 1, truncation about 1e-8; round-off 1e-13
    # of the eigensolves; the gate is 1e-6.
    assert_gradient_matches(
        lambda s: _scaled_rate(s, reference), s0, [np.array(1.0)], step=1e-2, rel=1e-6
    )


def test_the_gradient_of_the_purity_in_the_pulse_duration_matches_differences(
    linear_source,
):
    def purity(scale):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", gfem.UnresolvedJointSpectrumWarning)
            pulse = linear_source.joint_spectrum(
                LENGTH_PULSE,
                pulse_energy=PULSE_ENERGY,
                pulse_duration=PULSE_DURATION * scale,
            )
        return pulse.purity

    s0 = jnp.asarray(1.0)
    assert 0 < float(purity(s0)) < 1
    # the purity moves by order 1 as the pump narrows to the phase-matching
    # width (a scale of order 1 in the duration): the fourth-order truncation
    # h^4 J^(5)/30 sits below 1e-9 at h = 1e-2, and round-off of the 401 x 401
    # matrix products below 1e-13; the trace formula is differentiated without
    # an SVD
    assert_gradient_matches(purity, s0, [np.array(1.0)], step=1e-2, rel=1e-6)
