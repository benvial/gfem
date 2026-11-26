r"""Photon-pair sources of port modes: coupling, figures, refusals and gradients.

The overlaps run on a homogeneously filled PEC rectangle (WR-90, TE10), which
carries both a second-order tensor :math:`d_{yyy}` and a Kerr index. Bands of a
fraction of a GHz keep the solves to a few rows; the figures are checked
against the formulas of `gfem.PhotonPairSource` assembled by hand from the
coupling entries, against exact identities (halving, linearity, pytrees) and
against finite differences. The closed forms of a guide's rate are the
verification tier's.
"""

import dataclasses
import math
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from slab2d import write_slab

import gfem
from gfem.errors import NonDifferentiableError

DATA = Path(__file__).parent / "data"
N, N2 = 1.5, 2.6e-20
D_YYY = 20e-12
ORDER = gfem.SolverOptions(order=2)
#: signal [8.0, 8.3] GHz and idler [8.5, 8.8] GHz: centres 8.15 and 8.65, pump 16.8
F_S = np.linspace(8.0e9, 8.3e9, 5)
F_I = np.linspace(8.5e9, 8.8e9, 5)
#: one band that carries identical photons, and the pump of both processes
F_W = np.linspace(8.0e9, 8.8e9, 5)
F_P = np.linspace(16.0e9, 17.6e9, 5)
EPS = np.finfo(float).eps

# the bands are a fraction of a GHz, a fraction of one lobe of the phase-matching
# sinc of any guide here: the continuous-wave figures are cut by them on purpose,
# these tests check their algebra, not the band's coverage (the warning has its
# own tests)
pytestmark = pytest.mark.filterwarnings("ignore::gfem.UnresolvedJointSpectrumWarning")


def d_yyy(scale=1.0):
    d = np.zeros((3, 6))
    d[1, 1] = D_YYY * scale
    return d


def guide(mesh, material):
    return gfem.Model(
        mesh,
        materials={"air": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def mode_of(model, f, **kwargs):
    return gfem.solve(model, gfem.PortMode("p1", f, **kwargs), options=ORDER)


@pytest.fixture(scope="module")
def guide_mesh():
    return gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")


@pytest.fixture(scope="module")
def crystal(guide_mesh):
    material = gfem.Material(
        eps_r=N**2, second_order_nonlinear_tensor=d_yyy(), nonlinear_index=N2
    )
    return guide(guide_mesh, material)


@pytest.fixture(scope="module")
def arms(crystal):
    return {
        "s": mode_of(crystal, F_S),
        "i": mode_of(crystal, F_I),
        "w": mode_of(crystal, F_W),
        "p": mode_of(crystal, F_P),
    }


@pytest.fixture(scope="module")
def pair(crystal, arms):
    """Non-identical down-conversion: signal and idler bands apart."""
    return gfem.spdc(crystal, arms["p"], arms["s"], arms["i"])


@pytest.fixture(scope="module")
def twin(crystal, arms):
    """Identical photons: one band is both the signal and the idler."""
    return gfem.spdc(crystal, arms["p"], arms["w"], arms["w"])


@pytest.fixture(scope="module")
def mixing(crystal, arms):
    """Four-wave mixing with the pump on the wide band."""
    return gfem.sfwm(crystal, arms["w"], arms["s"], arms["i"])


def nearest(f, target):
    return int(np.argmin(np.abs(f - target)))


# --- the coupling ------------------------------------------------------------------


def test_the_spdc_coupling_is_the_conjugate_entry_of_the_coupling_tensor(
    crystal, arms, pair
):
    waves = [arms["p"], arms["s"], arms["i"]]
    kappa = np.asarray(gfem.three_wave_coupling(crystal, waves).kappa)
    ip = nearest(F_P, 16.8e9)
    i_s = nearest(F_S, 8.15e9)
    i_i = nearest(F_I, 8.65e9)
    entry = kappa[ip, len(F_P) + i_s, len(F_P) + len(F_S) + i_i]
    # the same overlap summed over K = 18 waves against K = 3 (another order of
    # the point sums): round-off, 1e-11 relative
    assert complex(pair.coupling) == pytest.approx(complex(np.conj(entry)), rel=1e-11)
    assert pair.process == "spdc"
    assert (pair.port, pair.modes, pair.identical) == ("p1", (1, 1, 1), False)
    assert pair.kerr_phase is None


def test_the_sfwm_coupling_and_phase_entries_are_the_kerr_tensor_entries(
    crystal, arms, mixing
):
    waves = [arms["w"], arms["s"], arms["i"]]
    gamma = np.asarray(gfem.kerr_coupling(crystal, waves).gamma_nl)
    ip = nearest(F_W, 8.4e9)
    s = len(F_W) + nearest(F_S, 8.15e9)
    i = len(F_W) + len(F_S) + nearest(F_I, 8.65e9)
    want = [gamma[s, ip, ip, i], gamma[i, ip, ip, s]]
    phase = [gamma[ip, ip, ip, ip], gamma[s, s, ip, ip], gamma[i, i, ip, ip]]
    scale = np.abs(gamma).max()
    # as above: another order of the point sums, 1e-11 of the largest entry
    np.testing.assert_allclose(mixing.coupling, want, rtol=0, atol=1e-11 * scale)
    np.testing.assert_allclose(mixing.kerr_phase, phase, rtol=0, atol=1e-11 * scale)
    assert mixing.process == "sfwm"
    assert mixing.identical is False


def test_the_bands_the_grid_and_the_identical_flag_follow_the_arms(
    crystal, arms, pair, twin, mixing
):
    np.testing.assert_allclose(pair.signal_frequency, np.linspace(8.0e9, 8.3e9, 201))
    np.testing.assert_allclose(pair.idler_frequency, np.linspace(8.5e9, 8.8e9, 201))
    assert pair.bands == ((16.0e9, 17.6e9), (8.0e9, 8.3e9), (8.5e9, 8.8e9))
    # the rows the couplings were taken on: nearest the centres, exact here
    assert pair.coupling_frequency == pytest.approx((16.8e9, 8.15e9, 8.65e9), rel=1e-12)
    assert twin.identical
    assert not pair.identical
    assert not mixing.identical
    coarse = gfem.spdc(crystal, arms["p"], arms["s"], arms["i"], points=7)
    assert len(coarse.signal_frequency) == len(coarse.idler_frequency) == 7
    # a copy of the same result as signal and idler is the same mode family
    again = dataclasses.replace(arms["w"])
    assert gfem.spdc(crystal, arms["p"], arms["w"], again).identical


# --- figures against the formulas of the docstring ---------------------------------


def test_the_cw_density_at_the_centre_is_the_coupling_times_the_poling_gain(pair):
    length, power = 1e-2, 1e-3
    period = pair.poling_period
    got = pair.spectral_brightness(length, power, period=period)
    centre = got.shape[-1] // 2
    ws, wi = 2 * np.pi * 8.15e9, 2 * np.pi * 8.65e9
    # the overlap was taken on the pump row nearest 16.8 GHz, and kappa scales
    # with that row's frequency
    wp0 = 2 * np.pi * pair.coupling_frequency[0]
    g = 2 * math.sqrt(ws * wi) / wp0 * abs(complex(pair.coupling))
    # first-order poling; the mismatch at the centre is the one the period
    # cancels, so the sinc is 1 to (delta L)^2 / 24 with delta ~ 1e-13 rad/m;
    # the lossless rows leave the decay factors at 1 - 1e-8 (alpha ~ 1e-6/m)
    expected = (2 / np.pi) ** 2 * g**2 * power * length**2
    assert float(got[centre]) == pytest.approx(expected, rel=1e-5)
    assert got.shape == (201,)


def test_the_sfwm_density_at_the_centre_is_the_symmetrized_coupling(mixing):
    length, power = 1e-3, 1e-3
    got = mixing.spectral_brightness(length, power)
    c = got.shape[-1] // 2
    ws, wi = 2 * np.pi * 8.15e9, 2 * np.pi * 8.65e9
    s, i = np.asarray(mixing.coupling)
    gamma_si = 0.5 * (math.sqrt(wi / ws) * s + math.sqrt(ws / wi) * i)
    pp_, ss, ii = np.asarray(mixing.kerr_phase)
    delta = float(mixing.delta_beta[c, c]) + 2 * (pp_ - ss - ii).real * power
    expected = (
        abs(gamma_si) ** 2
        * power**2
        * length**2
        * np.sinc(delta * length / 2 / np.pi) ** 2
    )
    # decay factors 1 - 1e-9 (alpha ~ 1e-6/m over 1 mm)
    assert float(got[c]) == pytest.approx(expected, rel=1e-5)


def test_the_rate_is_the_trapezoid_of_the_brightness_and_halves_for_identical_photons(
    pair, twin
):
    args = (0.5, 2e-3)
    kwargs = {"period": pair.poling_period}
    density = np.asarray(pair.spectral_brightness(*args, **kwargs))
    step = float(pair.signal_frequency[1] - pair.signal_frequency[0])
    # the same trapezoid, two evaluation orders: 1e-12 relative
    assert float(pair.rate(*args, **kwargs)) == pytest.approx(
        np.trapezoid(density, dx=step), rel=1e-12
    )
    same = dataclasses.replace(twin, identical=False)
    kw = {"period": twin.poling_period}
    assert float(twin.rate(*args, **kw)) == pytest.approx(
        0.5 * float(same.rate(*args, **kw)), rel=1e-14
    )


def test_a_pump_power_scales_the_density_linearly_and_sfwm_quadratically(pair, mixing):
    base = np.asarray(pair.spectral_brightness(0.2, 1e-3))
    np.testing.assert_allclose(
        pair.spectral_brightness(0.2, 3e-3), 3 * base, rtol=1e-14
    )
    b = np.asarray(mixing.spectral_brightness(1e-3, 1.0))
    # the power also enters the mismatch (self- and cross-phase), which is
    # ~1e-14 rad/m here: negligible against the 1/L = 1e3 scale
    np.testing.assert_allclose(mixing.spectral_brightness(1e-3, 3.0), 9 * b, rtol=1e-9)


def test_a_batch_of_arguments_broadcasts_before_the_grid_axis(pair):
    lengths = jnp.array([0.1, 0.2, 0.4])
    period = pair.poling_period
    got = pair.spectral_brightness(lengths, 1e-3, period=period)
    assert got.shape == (3, 201)
    rate = pair.rate(lengths, 1e-3, period=period)
    assert rate.shape == (3,)
    one = pair.rate(0.2, 1e-3, period=period)
    assert float(rate[1]) == pytest.approx(float(one), rel=1e-13)


def test_the_lossy_density_is_the_decayed_law_of_the_attenuations(pair):
    length = 1e-2
    alpha = 150.0
    lossy = dataclasses.replace(
        pair, fit=(pair.fit[0], pair.fit[1], pair.fit[2].at[0].add(alpha))
    )
    period = pair.poling_period
    got = lossy.spectral_brightness(length, 1e-3, period=period)
    base = pair.spectral_brightness(length, 1e-3, period=period)
    c = got.shape[-1] // 2
    # an idler attenuation alpha on a phase-matched guide: e^{-alpha L}
    # (sinh(alpha L / 2) / (alpha L / 2))^2, and the pump and signal ~ 1e-6/m
    x = alpha * length / 2
    ratio = math.exp(-2 * x) * (math.sinh(x) / x) ** 2
    assert float(got[c] / base[c]) == pytest.approx(ratio, rel=1e-4)


def test_the_delta_beta_grid_and_the_poling_period_come_from_the_fits(pair, mixing):
    db = np.asarray(pair.delta_beta)
    assert db.shape == (201, 201)
    c = 100
    assert float(pair.poling_period) == pytest.approx(
        2 * np.pi / abs(db[c, c]), rel=1e-12
    )
    # normal dispersion of the filled guide: beta(2w) > 2 beta(w) (> 0), growing
    # with the signal frequency at a fixed idler (beta_s grows)
    assert db[c, c] > 0
    assert db[-1, c] < db[0, c]
    with pytest.raises(ValueError, match="not quasi-phase-matched"):
        _ = mixing.poling_period
    assert np.asarray(mixing.delta_beta).shape == (201, 201)


# --- the joint spectrum ------------------------------------------------------------


def test_the_joint_spectrum_carries_pairs_purity_and_schmidt_number(pair):
    with pytest.warns(gfem.UnresolvedJointSpectrumWarning, match="band edge"):
        js = pair.joint_spectrum(
            1.0, pulse_energy=1e-12, pulse_duration=5e-9, period=pair.poling_period
        )
    assert isinstance(js, gfem.JointSpectrum)
    assert js.jsa.shape == (201, 201)
    step = float(js.signal_frequency[1] - js.signal_frequency[0]) * float(
        js.idler_frequency[1] - js.idler_frequency[0]
    )
    total = float(jnp.sum(jnp.abs(js.jsa) ** 2)) * step
    assert float(js.pairs_per_pulse) == pytest.approx(total, rel=1e-13)
    s = np.linalg.svd(np.asarray(js.jsa), compute_uv=False)
    lam = s**2 / np.sum(s**2)
    # the trace formula and an SVD of a 201 x 201 matrix: a few hundred eps
    assert float(js.purity) == pytest.approx(np.sum(lam**2), rel=1e-12)
    assert float(js.schmidt_number) == pytest.approx(1 / float(js.purity), rel=1e-15)
    assert 0 < float(js.purity) <= 1
    assert "pairs/pulse" in repr(js)


def test_identical_photons_halve_the_pairs_and_leave_the_purity(twin):
    kwargs = {
        "pulse_energy": 1e-12,
        "pulse_duration": 5e-9,
        "period": twin.poling_period,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnresolvedJointSpectrumWarning)
        half = twin.joint_spectrum(1.0, **kwargs)
        full = dataclasses.replace(twin, identical=False).joint_spectrum(1.0, **kwargs)
    assert float(half.pairs_per_pulse) == pytest.approx(
        0.5 * float(full.pairs_per_pulse), rel=1e-13
    )
    # a uniform scale of the amplitude: the ratio cancels it, round-off only
    assert float(half.purity) == pytest.approx(float(full.purity), rel=1e-13)


def test_the_sfwm_joint_spectrum_and_the_pump_energy_scaling(mixing):
    kwargs = {"pulse_duration": 5e-9}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnresolvedJointSpectrumWarning)
        one = mixing.joint_spectrum(1.0, pulse_energy=1e-12, **kwargs)
        two = mixing.joint_spectrum(1.0, pulse_energy=2e-12, **kwargs)
    # the pump square spectrum is linear in the peak power, which is linear in
    # the energy: the pair number goes as the energy squared, the shape stays
    assert float(two.pairs_per_pulse) == pytest.approx(
        4 * float(one.pairs_per_pulse), rel=1e-12
    )
    assert float(two.purity) == pytest.approx(float(one.purity), rel=1e-12)


def test_the_pair_number_scales_with_the_square_of_the_coupling(pair):
    kwargs = {"pulse_energy": 1e-12, "pulse_duration": 5e-9}
    big = dataclasses.replace(pair, coupling=3 * pair.coupling)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = float(pair.joint_spectrum(1.0, **kwargs).pairs_per_pulse)
        b = float(big.joint_spectrum(1.0, **kwargs).pairs_per_pulse)
    assert b == pytest.approx(9 * a, rel=1e-12)


# --- warnings ----------------------------------------------------------------------


def _tilt(source, delta):
    """A signal arm whose beta tilts by ``delta`` rad/m over the band."""
    fit = list(source.fit)
    fit[1] = fit[1].at[1].add(1j * delta)
    return dataclasses.replace(source, fit=tuple(fit))


def test_a_resolved_joint_spectrum_inside_its_bands_does_not_warn(pair):
    # a group-velocity mismatch of 40 rad/m across the band over 1 m: the
    # amplitude sinc reaches 20 rad (0.05 of its peak) at the band edge with 5
    # points per radian... (100 points per 20 rad)
    tilted = _tilt(pair, 40.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        js = tilted.joint_spectrum(
            1.0, pulse_energy=1e-12, pulse_duration=5e-9, period=pair.poling_period
        )
    assert float(js.pairs_per_pulse) > 0


def test_an_undersampled_joint_spectrum_warns(pair):
    tilted = _tilt(pair, 400.0)
    with pytest.warns(gfem.UnresolvedJointSpectrumWarning, match="jumps by"):
        tilted.joint_spectrum(
            1.0, pulse_energy=1e-12, pulse_duration=5e-9, period=pair.poling_period
        )


def test_a_joint_spectrum_cut_by_its_bands_warns(pair):
    with pytest.warns(gfem.UnresolvedJointSpectrumWarning, match="band edge"):
        pair.joint_spectrum(
            1.0, pulse_energy=1e-12, pulse_duration=5e-9, period=pair.poling_period
        )


def test_a_high_occupation_warns_for_both_the_cw_and_the_pulsed_figures(pair):
    loud = dataclasses.replace(pair, coupling=1e9 * pair.coupling)
    with pytest.warns(gfem.HighGainWarning, match="pairs/"):
        loud.rate(1.0, 1e-3, period=pair.poling_period)
    # a tilted arm keeps the spectrum resolved: only the occupation warns
    with pytest.warns(gfem.HighGainWarning, match="per pulse"):
        _tilt(loud, 40.0).joint_spectrum(
            1.0,
            pulse_energy=1e-12,
            pulse_duration=5e-9,
            period=pair.poling_period,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.HighGainWarning)
        pair.rate(1.0, 1e-3, period=pair.poling_period)


# --- refusals ----------------------------------------------------------------------


def test_the_range_of_every_argument_is_checked(pair):
    p = pair.poling_period
    with pytest.raises(ValueError, match=r"length must be >= 0"):
        pair.rate(-1.0, 1e-3)
    with pytest.raises(ValueError, match=r"power must be >= 0"):
        pair.rate(1.0, -1e-3)
    with pytest.raises(ValueError, match=r"period must be > 0"):
        pair.rate(1.0, 1e-3, period=-p)
    with pytest.raises(ValueError, match=r"duty_cycle must be in \[0, 1\]"):
        pair.rate(1.0, 1e-3, period=p, duty_cycle=1.5)
    with pytest.raises(ValueError, match=r"order must be an integer"):
        pair.rate(1.0, 1e-3, period=p, order=1.5)
    with pytest.raises(ValueError, match=r"order must be >= 1"):
        pair.rate(1.0, 1e-3, period=p, order=0)
    with pytest.raises(ValueError, match="apply to a poled guide"):
        pair.rate(1.0, 1e-3, duty_cycle=0.4)
    with pytest.raises(TypeError, match="length must be real"):
        pair.rate(1.0j, 1e-3)
    with pytest.raises(ValueError, match=r"pulse_duration must be > 0"):
        pair.joint_spectrum(1.0, pulse_energy=1e-12, pulse_duration=0.0)
    with pytest.raises(ValueError, match="scalar"):
        pair.joint_spectrum(
            jnp.array([1.0, 2.0]), pulse_energy=1e-12, pulse_duration=1e-9
        )


def test_four_wave_mixing_takes_no_poling(mixing):
    for kwargs in ({"period": 1e-3}, {"duty_cycle": 0.5}, {"order": 1}):
        with pytest.raises(ValueError, match="not quasi-phase-matched"):
            mixing.rate(1e-3, 1.0, **kwargs)
        with pytest.raises(ValueError, match="not quasi-phase-matched"):
            mixing.spectral_brightness(1e-3, 1.0, **kwargs)
        with pytest.raises(ValueError, match="not quasi-phase-matched"):
            mixing.joint_spectrum(
                1e-3, pulse_energy=1e-12, pulse_duration=1e-9, **kwargs
            )


def test_the_arms_are_checked_for_rows_points_type_and_span(crystal, arms):
    p, s, i = arms["p"], arms["s"], arms["i"]
    few = mode_of(crystal, F_S[:3])
    with pytest.raises(ValueError, match=r"the signal has 3 rows.*at least 4"):
        gfem.spdc(crystal, p, few, i)
    with pytest.raises(ValueError, match=r"points must be an integer >= 3"):
        gfem.spdc(crystal, p, s, i, points=2)
    with pytest.raises(TypeError, match="PortModeResult as the pump"):
        gfem.spdc(crystal, "pump", s, i)  # ty: ignore[invalid-argument-type]
    bare = dataclasses.replace(s, e_field=None)
    with pytest.raises(TypeError, match=r"gfem\.PortMode"):
        gfem.spdc(crystal, p, bare, i)
    # a pump that misses the sum of the band centres (16.8 GHz)
    low = mode_of(crystal, np.linspace(16.0e9, 16.4e9, 4))
    with pytest.raises(ValueError, match=r"the pump rows span.*np\.linspace"):
        gfem.spdc(crystal, low, s, i)


def test_a_pump_that_covers_only_the_centre_serves_cw_but_not_the_pulse(crystal, arms):
    narrow = mode_of(crystal, np.linspace(16.6e9, 17.0e9, 4))
    source = gfem.spdc(crystal, narrow, arms["w"], arms["w"])
    assert float(source.rate(1e-2, 1e-3)) > 0
    with pytest.raises(
        ValueError, match=r"needs the pump from 16000000000 to 17600000000"
    ):
        source.joint_spectrum(1.0, pulse_energy=1e-12, pulse_duration=1e-9)
    with pytest.raises(ValueError, match=r"np\.linspace\("):
        _ = source.delta_beta


def test_a_fit_that_leaves_a_phase_over_the_length_is_refused(pair):
    bad = dataclasses.replace(pair, fit_residual=jnp.array([0.0, 0.0, 50.0]))
    with pytest.raises(ValueError, match=r"idler arm's polynomial fit.*0\.5 rad"):
        bad.rate(1e-2, 1e-3)
    # under a trace the check is skipped, as the residual is not concrete
    traced = jax.jit(lambda s: s.rate(1e-2, 1e-3))(bad)
    assert np.isfinite(float(traced))


def test_a_model_without_the_property_names_the_material(guide_mesh, arms):
    plain = guide(guide_mesh, gfem.Material(eps_r=N**2))
    with pytest.raises(ValueError, match="second_order_nonlinear_tensor"):
        gfem.spdc(plain, arms["p"], arms["s"], arms["i"])
    with pytest.raises(ValueError, match="no element"):
        gfem.sfwm(plain, arms["w"], arms["s"], arms["i"])


# --- line ports --------------------------------------------------------------------

SLAB_F = 193e12


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    path = write_slab(
        tmp_path_factory.mktemp("slab") / "slab.msh",
        thick=0.22,
        clear=1.0,
        h_core=0.055,
        h_clad=0.2,
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="um")
    d = np.zeros((3, 6))
    d[2, 2] = 25e-12
    model = gfem.Model(
        mesh,
        materials={
            "core": gfem.Material(
                eps_r=3.476**2, second_order_nonlinear_tensor=d, nonlinear_index=N2
            ),
            "clad": gfem.Material(eps_r=1.444**2),
        },
        ports=[gfem.WavePort("p1", "port1", pec_edges=("outer",))],
    )
    options = gfem.SolverOptions(order=3)
    ez = gfem.solve(
        model, gfem.PortMode("p1", [SLAB_F], polarization="Ez"), options=options
    )
    hz = gfem.solve(
        model, gfem.PortMode("p1", [SLAB_F], polarization="Hz"), options=options
    )
    return model, ez, hz


@pytest.mark.parametrize("process", ["spdc", "sfwm"])
def test_a_line_port_is_refused_for_both_polarizations(slab, process):
    model, ez, hz = slab
    for mode in (ez, hz):
        with pytest.raises(NotImplementedError, match="line port"):
            getattr(gfem, process)(model, mode, mode, mode)


# --- gradients, trace, pytree, repr ------------------------------------------------


def _field_model(mesh, eps):
    field = gfem.MaterialField(eps_r=eps, second_order_nonlinear_tensor=d_yyy())
    return guide(mesh, field)


FS_GRAD = np.linspace(8.0e9, 8.6e9, 4)
FP_GRAD = np.linspace(16.0e9, 17.2e9, 4)
#: a fixed poling period near the first-order match of the grid's guide (m)
PERIOD = 0.106


def _grad_source(mesh, eps):
    model = _field_model(mesh, eps)
    s = mode_of(model, FS_GRAD)
    p = mode_of(model, FP_GRAD)
    return gfem.spdc(model, p, s, s, points=41)


def test_the_rate_differentiates_in_the_permittivity_like_its_finite_differences(
    guide_mesh,
):
    seen = []

    def rate(eps):
        source = _grad_source(guide_mesh, eps)
        seen.append(repr(source))
        return source.rate(1.0, 1e-3, period=PERIOD)

    e0 = N**2
    value, grad = jax.value_and_grad(rate)(e0)
    h = 1e-5
    fd = (rate(e0 * (1 + h)) - rate(e0 * (1 - h))) / (2 * e0 * h)
    assert float(value) > 0
    # a traced source keeps a short repr: no coupling magnitude, no poling period
    assert seen[0].startswith("PhotonPairSource(spdc, port 'p1'")
    assert seen[0].endswith("THz)")
    assert "|coupling|" not in seen[0]
    # central difference: truncation ~ (h e0)^2 f'''/(6 f') measured at 5e-9
    # relative at h = 1e-5 (it scales as h^2: 5e-5 at 1e-3), and the round-off
    # of the solves 1e-14 / h ~ 1e-9; the adjoint is exact up to the solves'
    # round-off. 1e-7 leaves a factor 20.
    assert float(grad) == pytest.approx(float(fd), rel=1e-7)


def test_the_methods_jit_on_the_source_and_a_traced_spectrum_has_a_short_repr(pair):
    period = pair.poling_period
    eager = float(pair.rate(0.3, 2e-3, period=period))
    jitted = jax.jit(lambda s, length, p: s.rate(length, p, period=period))
    # XLA may fuse and reorder the sums: a few hundred eps
    assert float(jitted(pair, 0.3, 2e-3)) == pytest.approx(eager, rel=1e-12)
    seen = []

    def spectrum(source, energy):
        js = source.joint_spectrum(
            0.5, pulse_energy=energy, pulse_duration=5e-9, period=period
        )
        seen.append(repr(js))
        return js.pairs_per_pulse

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnresolvedJointSpectrumWarning)
        pairs = jax.jit(spectrum)(pair, 1e-12)
        assert float(pairs) == pytest.approx(float(spectrum(pair, 1e-12)), rel=1e-12)
    assert seen[0] == "JointSpectrum(201 x 201 grid)"
    assert seen[1].startswith("JointSpectrum(201 x 201 grid, ")


def test_the_repr_names_the_process_bands_and_poling_period(pair, twin, mixing):
    text = repr(pair)
    assert text.startswith("PhotonPairSource(spdc, port 'p1', modes 1 -> 1 + 1, pump")
    assert "signal 0.008-0.0083 THz" in text
    assert "poling period" in text
    assert "identical" in repr(twin)
    assert "poling" not in repr(mixing)
    assert "1/(W m)" in repr(mixing)


def test_the_source_is_a_pytree_of_its_coupling_and_fits(pair, mixing):
    leaves, tree = jax.tree.flatten(pair)
    # coupling, three fits, the residuals ("kerr_phase" is None: no leaf)
    assert len(leaves) == 5
    again = jax.tree.unflatten(tree, leaves)
    assert (again.process, again.modes, again.identical) == ("spdc", (1, 1, 1), False)
    np.testing.assert_array_equal(again.signal_frequency, pair.signal_frequency)
    assert again.bands == pair.bands
    assert len(jax.tree.leaves(mixing)) == 6
    doubled = jax.tree.map(lambda x: 2 * x, pair)
    assert complex(doubled.coupling) == pytest.approx(2 * complex(pair.coupling))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnresolvedJointSpectrumWarning)
        js = pair.joint_spectrum(
            0.5, pulse_energy=1e-12, pulse_duration=5e-9, period=pair.poling_period
        )
    out = jax.tree.unflatten(*reversed(jax.tree.flatten(js)))
    np.testing.assert_array_equal(out.jsa, js.jsa)
    np.testing.assert_array_equal(out.idler_frequency, js.idler_frequency)


def test_traced_node_coordinates_raise_by_name(guide_mesh, crystal):
    nodes = np.array(guide_mesh.imported_nodes)

    def objective(scale):
        mesh = jax.tree.unflatten(
            jax.tree.structure(guide_mesh), [jnp.asarray(nodes) * scale]
        )
        moved = crystal.replace(mesh=mesh)
        s = mode_of(moved, FS_GRAD)
        p = mode_of(moved, FP_GRAD)
        return jnp.real(gfem.spdc(moved, p, s, s).coupling)

    with pytest.raises(NonDifferentiableError, match=r"gfem\.spdc reads the mode"):
        jax.grad(objective)(1.0)
