r"""Photon-pair sources built by hand: warnings, refusals and prefactors.

The arms are linear dispersions (a group index each), so every figure has a
closed form and a source costs nothing to build; the finite-element side of
`gfem.photon_pairs` is in `test_photon_pairs`.
"""

import dataclasses
import re
import warnings
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem import photon_pairs as pairs

C = 299792458.0
FS0, FI0 = 190e12, 196e12
B = 0.4e12  # half band of the signal and the idler, Hz


def fit_linear(beta0, beta1, half):
    """``gamma = i beta`` with ``beta = beta0 + beta1 (f - fc)`` as a degree-4 fit."""
    return jnp.array([1j * beta0, 1j * beta1 * half, 0.0, 0.0, 0.0])


def source(process="spdc", *, points=201, identical=False, coupling=None, dn=0.2):
    """Phase-matched at the centres; signal and idler group indices differ by ``dn``."""
    k = 2 * np.pi / C
    ng_s, ng_i, ng_p = 2.2, 2.2 - dn, 2.3
    if process == "spdc":
        fp0, ph = FS0 + FI0, 2 * B
        bp = (ng_s * FS0 + ng_i * FI0) * k
        kerr = None
        coupling = jnp.asarray(1.0 + 0j) if coupling is None else coupling
    else:
        fp0, ph = 0.5 * (FS0 + FI0), B
        bp = 0.5 * (ng_s * FS0 + ng_i * FI0) * k
        kerr = jnp.zeros(3, complex)
        coupling = jnp.array([1.0 + 0j, 1.0 + 0j]) if coupling is None else coupling
    fits = (
        fit_linear(bp, ng_p * k, ph),
        fit_linear(ng_s * k * FS0, ng_s * k, B),
        fit_linear(ng_i * k * FI0, ng_i * k, B),
    )
    return pairs.PhotonPairSource(
        process=process,
        port="p",
        modes=(1, 1, 2),
        identical=identical,
        signal_frequency=np.linspace(FS0 - B, FS0 + B, points),
        idler_frequency=np.linspace(FI0 - B, FI0 + B, points),
        bands=((fp0 - ph, fp0 + ph), (FS0 - B, FS0 + B), (FI0 - B, FI0 + B)),
        coupling=coupling,
        kerr_phase=kerr,
        fit=fits,
        fit_residual=jnp.zeros(3),
    )


def caught(call):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        call()
    return w


# --- the CW figures resolve their sinc ---------------------------------------------


def test_a_cw_rate_that_the_grid_cannot_resolve_warns_and_a_fine_grid_does_not():
    # delta slope 0.2 * 2 pi / c over 0.8 THz: over 0.5 m the sinc main lobe is
    # c / (0.2 * 0.5 m) = 3 GHz wide against a 4 GHz step of 201 points
    coarse, fine = source(points=201), source(points=4001)
    with pytest.warns(gfem.UnresolvedJointSpectrumWarning, match="spectral density"):
        coarse.rate(0.5, 1e-3)
    with pytest.warns(gfem.UnresolvedJointSpectrumWarning, match="jumps by"):
        coarse.spectral_brightness(0.5, 1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fine.rate(0.5, 1e-3)
        # 5 cm: the sinc has 25 lobes in band and 8 samples per 0.4 THz step
        coarse.rate(5e-2, 1e-3)
        source("sfwm").rate(5e-2, 1.0)


def test_a_cw_spectrum_cut_by_the_signal_band_warns_on_its_edge():
    # 1.2 ps/m of mismatch at 1 mm: the sinc is 0.9 wide in x, the band a
    # fraction of one lobe, so the density is at its peak on the band edge
    with pytest.warns(gfem.UnresolvedJointSpectrumWarning, match="band edge"):
        source().rate(1e-4, 1e-3)


def test_the_cw_check_covers_each_entry_of_a_batch():
    lengths = jnp.array([5e-2, 0.5])
    with pytest.warns(gfem.UnresolvedJointSpectrumWarning, match="jumps by"):
        source().rate(lengths, 1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        source().rate(jnp.array([5e-2, 4e-2]), 1e-3)


def test_the_warnings_point_at_the_calling_line():
    s = source(points=11)
    loud = dataclasses.replace(s, coupling=jnp.asarray(1e9 + 0j))
    for call in (
        lambda: s.joint_spectrum(5e-2, pulse_energy=1e-12, pulse_duration=3e-12),
        lambda: s.rate(0.5, 1e-3),
        lambda: loud.rate(5e-2, 1e-3),
        lambda: loud.joint_spectrum(5e-2, pulse_energy=1e-12, pulse_duration=3e-12),
    ):
        w = caught(call)
        assert w
        # the lambda is this file: a frame too many lands in the module's caller
        assert {x.filename for x in w} == {__file__}


# --- the high-gain check for identical photons -------------------------------------


def test_identical_photons_warn_at_the_occupation_of_photons():
    s = source(points=101)
    kwargs = {"pulse_energy": 1e-12, "pulse_duration": 3e-12}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnresolvedJointSpectrumWarning)
        unit = s.joint_spectrum(5e-2, **kwargs)
    # pairs * sqrt(purity) of the full (non-identical) amplitude of unit
    # coupling; it scales as the coupling squared. The identical amplitude has
    # half the pairs and twice the photons per pair: the full figure
    x = float(unit.pairs_per_pulse) * float(unit.purity) ** 0.5

    def kinds(occupation):
        c = jnp.asarray((occupation / x) ** 0.5 + 0j)
        twin = dataclasses.replace(s, coupling=c, identical=True)
        return {w.category for w in caught(lambda: twin.joint_spectrum(5e-2, **kwargs))}

    assert gfem.HighGainWarning in kinds(0.15)
    assert gfem.HighGainWarning not in kinds(0.06)


# --- refusals ----------------------------------------------------------------------


def test_a_pulse_of_no_energy_is_refused():
    with pytest.raises(ValueError, match=r"pulse_energy must be > 0"):
        source().joint_spectrum(5e-2, pulse_energy=0.0, pulse_duration=3e-12)


def test_a_fit_residual_of_the_pump_counts_twice_in_four_wave_mixing():
    # a pump residual of 0.06 rad/m over 1 m: under the 0.1 rad limit for
    # down-conversion, 0.12 rad for four-wave mixing (the pump enters twice)
    residual = jnp.array([0.06, 0.0, 0.0])
    spdc = dataclasses.replace(source(), fit_residual=residual)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gfem.UnresolvedJointSpectrumWarning)
        assert float(spdc.rate(1.0, 1e-3)) >= 0
    sfwm = dataclasses.replace(source("sfwm"), fit_residual=residual)
    with pytest.raises(ValueError, match=r"pump arm's polynomial fit.*0\.12 rad"):
        sfwm.rate(1.0, 1.0)


def test_a_cw_pump_refusal_suggests_a_band_that_can_be_solved():
    s = source()
    narrow = (190e12 + 196e12 + 5e12, 190e12 + 196e12 + 6e12)
    short = dataclasses.replace(s, bands=(narrow, *s.bands[1:]))
    with pytest.raises(ValueError, match=r"np\.linspace") as err:
        short.rate(1e-2, 1e-3)
    lo, hi = (
        float(x)
        for x in re.search(
            r"np\.linspace\(([^,]+), ([^,]+), 9\)", str(err.value)
        ).groups()
    )
    # the point the amplitude needs, widened by the pulsed span
    f_pump = FS0 + FI0
    assert lo < f_pump < hi
    assert hi - lo == pytest.approx(2 * B + 2 * B, rel=1e-12)


# --- one mode over several bands ---------------------------------------------------


def wave(mode=1, port="p"):
    return SimpleNamespace(port=port, mode=mode)


@pytest.mark.parametrize(
    ("signal", "idler", "expected"),
    [
        ((8.0, 8.5), (8.0, 8.5), True),
        ((8.0, 8.5), (8.0 + 1e-12, 8.5), True),
        ((8.0, 8.5), (8.6, 9.0), False),
        ((8.0, 8.5), (8.5, 9.0), False),
        ((8.0, 8.5), (7.0, 7.9), False),
    ],
    ids=["equal", "equal to round-off", "disjoint", "touching", "below"],
)
def test_one_mode_over_equal_bands_is_identical_and_over_disjoint_bands_is_not(
    signal, idler, expected
):
    bands = ((0, 1), signal, idler)
    bands = tuple(tuple(1e9 * x for x in b) for b in bands)
    got = pairs._identical("gfem.spdc", wave(), wave(), bands)
    assert got is expected


@pytest.mark.parametrize("idler", [(8.3, 8.8), (8.1, 8.4), (7.5, 8.2), (7.9, 8.6)])
def test_one_mode_over_partly_overlapping_bands_is_refused(idler):
    bands = tuple(tuple(1e9 * x for x in b) for b in ((0, 1), (8.0, 8.5), idler))
    with pytest.raises(ValueError, match="split or merge the bands"):
        pairs._identical("gfem.spdc", wave(), wave(), bands)


def test_other_modes_and_the_same_result_are_decided_without_the_bands():
    bands = tuple(tuple(1e9 * x for x in b) for b in ((0, 1), (8.0, 8.5), (8.3, 8.8)))
    same = wave()
    assert pairs._identical("o", same, same, bands) is True
    assert pairs._identical("o", wave(1), wave(2), bands) is False
    assert pairs._identical("o", wave(port="a"), wave(port="b"), bands) is False


# --- the grid carries the frequency prefactor --------------------------------------


def test_the_sfwm_prefactor_is_the_centre_expression_and_scales_on_the_grid():
    s, i = 1.3 + 0.2j, 0.7 - 0.4j
    src = source("sfwm", coupling=jnp.array([s, i]))
    ws0, wi0 = 2 * np.pi * FS0, 2 * np.pi * FI0
    centre = complex(src._pair_coupling(ws0, wi0))
    adr = 0.5 * (np.sqrt(wi0 / ws0) * s + np.sqrt(ws0 / wi0) * i)
    # the same arithmetic in another order: a few eps
    assert centre == pytest.approx(adr, rel=1e-14)
    # away from the centre only sqrt(ws wi) varies, the entries are divided by
    # the frequencies they were taken at
    ws, wi = ws0 * 1.03, wi0 * 0.97
    grid = complex(src._pair_coupling(ws, wi))
    assert grid == pytest.approx(centre * np.sqrt(ws * wi / (ws0 * wi0)), rel=1e-14)


def test_the_spdc_prefactor_divides_by_the_pump_frequency_of_the_overlap():
    kappa = 0.8 - 0.3j
    src = dataclasses.replace(
        source(coupling=jnp.asarray(kappa)),
        coupling_frequency=(1.01 * (FS0 + FI0), FS0, FI0),
    )
    ws, wi = 2 * np.pi * 189e12, 2 * np.pi * 197e12
    want = 2 * np.sqrt(ws * wi) / (2 * np.pi * 1.01 * (FS0 + FI0)) * kappa
    assert complex(src._pair_coupling(ws, wi)) == pytest.approx(want, rel=1e-14)
