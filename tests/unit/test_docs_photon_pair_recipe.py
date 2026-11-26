"""The photon-pair recipe runs and prints what the page shows.

docs/recipes/photon-pair-sources.md runs nothing at docs build time. This
runs its blocks in order, on the ridge and strip sections built by the
repository's own docs/recipes/ridge_section.py and strip_section.py (the page
downloads them), and checks the numbers against what the physics fixes: the
poling period as the paper's, the L^(3/2) growth of a degenerate pair and the
asymmetry of a detuned period, the purity as the inverse of the Schmidt
number, the gradient against its difference, the square-law of the
four-wave mixing density in the power and its closed form, the ring's Q-cubed
law and circulating power, the grid, fit and tolerance checks, and the
equality of the pair density to the classical conversion efficiency.
"""

import importlib.util
import math

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "photon-pair-sources.md"
C = 299792458.0  # m/s

#: the plot needs matplotlib, which only the ``docs`` dependency group installs
_PLOT = ("Plot the joint spectrum",)


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("photon_pairs")
    skip = () if importlib.util.find_spec("matplotlib") else _PLOT
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir, skip=skip)


def _lines(ran, section: str, index: int = 0) -> list[str]:
    found = [example for example in ran if example.section == section]
    return found[index].printed.splitlines()


def _number(line: str, index: int = 0) -> float:
    return float(dp.numbers(line)[index])


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_only_the_fundamental_pump_mode_is_coupled_to_the_pair(ran):
    first, second = _lines(ran, "The modes of the pump and of the pair")[:2]
    assert _number(first, 1) > _number(second, 1)  # ranked by index
    assert _number(first, 2) > 1e6 * _number(second, 2)  # the other is cross-polarized


def test_the_poling_period_is_the_papers(ran):
    (line,) = _lines(ran, "The source")[1:2]
    assert _number(line) == pytest.approx(2.8, abs=0.05)  # Zhao et al.: 2.8 um


def test_the_brightness_figures_are_one_density_in_three_units(ran):
    lines = _lines(ran, "The rate and the spectral brightness of a continuous pump")
    peak = _number(lines[1])
    per_ghz, per_nm = (_number(lines[2], k) for k in (0, 1))
    assert per_ghz == pytest.approx(peak * 1e9, rel=1e-3)  # 1 mW
    assert per_nm / per_ghz == pytest.approx(C / 1.57e-6**2 * 1e-9 / 1e9, rel=1e-3)
    assert 0.001 < peak < 0.1  # low gain: below the warning, above round-off
    assert _number(lines[0]) == pytest.approx(4e10, rel=0.5)  # pairs/s at 1 mW


def test_a_degenerate_pair_grows_as_the_length_to_the_three_halves(ran):
    lines = _lines(ran, "The rate and the spectral brightness of a continuous pump", 1)
    short, mid, long_ = (_number(line, 1) for line in lines[:3])
    # a mismatch quadratic in the detuning gives L^(3/2): 2.83 per doubling,
    # while the matched band stays inside the 20 THz of the grid
    assert mid / short == pytest.approx(2**1.5, rel=0.03)
    assert long_ / mid == pytest.approx(2**1.5, rel=0.05)


def test_a_detuned_period_loses_the_rate_and_not_alike_on_both_sides(ran):
    lines = _lines(ran, "The rate and the spectral brightness of a continuous pump", 1)
    short, matched, long_ = (_number(line, 2) for line in lines[3:6])
    assert matched == pytest.approx(1.0, abs=1e-3)
    # beta_2 > 0: only a longer period keeps matched frequencies +-Omega
    assert short < 0.1
    assert 0.3 < long_ < 0.8


def test_purity_is_the_inverse_of_the_schmidt_number(ran):
    first = _lines(ran, "The joint spectrum of a pulse")
    assert "401 x 401" in first[0]
    n, purity, schmidt = (_number(first[0], k) for k in (2, 3, 4))
    assert schmidt * purity == pytest.approx(1.0, rel=2e-3)
    assert n == pytest.approx(_number(first[3]) / 1e8, rel=1e-3)  # pairs/s at 100 MHz
    width, band = _number(first[1]), _number(first[1], 1)
    assert 0.2 * band < width < band  # the photons fill a part of the band
    rows = _lines(ran, "The joint spectrum of a pulse", 1)[1:]
    table = [[_number(row, k) for k in range(4)] for row in rows]
    assert [row[0] for row in table] == sorted(row[0] for row in table)
    for _, pairs, p, k in table:
        assert 0 < p < 0.1
        assert k * p == pytest.approx(1.0, rel=2e-3)
        assert pairs < 0.1  # the low-gain regime: no HighGainWarning
    # the first row, 1 ps, is the 401-point figure of the block before
    assert table[0][1] == pytest.approx(n, rel=2e-3)
    assert table[0][2] == pytest.approx(purity, rel=0.1)
    assert table[-1][2] < table[0][2]  # a long pulse is a thinner line


def test_the_gradient_of_the_purity_matches_its_difference(ran):
    grad, difference = (
        _number(line, 0) for line in _lines(ran, "The gradient of the purity")
    )
    assert grad == pytest.approx(difference, rel=1e-3)
    rows = _lines(ran, "The joint spectrum of a pulse", 1)[1:]
    table = {_number(row): _number(row, 2) for row in rows}
    secant = (table[3.0] - table[1.5]) / 1.5  # per ps, around 2 ps
    assert grad < 0
    assert grad == pytest.approx(secant, rel=0.5)


def test_the_four_wave_mixing_density_is_quadratic_in_the_power(ran):
    section = "Four-wave mixing in a silicon wire"
    density = _lines(ran, section, 1)
    peak = _number(density[0])
    rate = _number(density[0], 1)
    assert 90 < _number(density[2]) < 110  # 10 mW against 1 mW: x100
    # |Gamma|^2 P^2 L^2 at 1 mW and 1 cm, with the ring's |Gamma|: the peak is
    # at the guard, where sinc^2 is within a percent of 1
    gamma = _number(_lines(ran, "A ring: an estimate")[0], 1)
    assert peak == pytest.approx(gamma**2 * 1e-6 * 1e-4, rel=0.03)
    assert peak * 1e12 < rate < peak * 3e13  # between 1 and 30 THz of the density
    assert 2e4 < _number(density[1]) < 2e6  # Hz/nm/mW^2: the published scatter


def test_the_filtered_four_wave_mixing_pulse_is_nearly_a_pure_state(ran):
    (line,) = _lines(ran, "Four-wave mixing in a silicon wire", 2)
    n, purity, schmidt = (_number(line, k) for k in (2, 3, 4))
    assert n * math.sqrt(purity) < 0.1  # low gain
    assert schmidt * purity == pytest.approx(1.0, rel=2e-3)
    assert 1 < schmidt < 5  # the guard band's cut, not the dispersion


def test_the_ring_follows_the_coupled_mode_formulas(ran):
    lines = _lines(ran, "A ring: an estimate")
    fsr, gamma = _number(lines[0]) * 1e9, _number(lines[0], 1)
    enhancement = _number(lines[1])
    rate, occupation = _number(lines[2]), _number(lines[2], 1)
    kappa = 2 * math.pi * C / 1.55e-6 / 1e5  # loaded, Q = 1e5
    assert enhancement == pytest.approx(2 * fsr / kappa, rel=2e-3)  # 4 k_ex FSR / k^2
    n_g = C / (fsr * 2 * math.pi * 10e-6)
    assert 4.0 < n_g < 4.4  # the silicon strip's group index
    g = gamma * (C / n_g) * enhancement * 0.1e-3  # rad/s
    assert rate == pytest.approx(g**2 / (2 * kappa), rel=0.02)
    assert occupation == pytest.approx(4 * g**2 / (2 * kappa**2), rel=0.02)
    assert occupation < 0.1  # low gain
    assert _number(lines[3]) == pytest.approx(8.0, abs=5e-3)  # Q^3


def test_the_published_comparison_is_an_observation_of_the_order_of_the_scatter(ran):
    tfln, javid, si = _lines(ran, "Compared with published sources")
    ratio = _number(tfln, 2)
    assert 4 < ratio < 40  # published on-chip numbers scatter by 4 to 20
    assert _number(tfln) == pytest.approx(
        ratio * 4.6e5, rel=1e-2
    )  # the same two numbers
    assert 1 < _number(javid) / 13 < 20
    assert 1 < _number(si, 1) / 1.9e5 < 20


def test_the_rate_converges_with_the_grid(ran):
    rates = [_number(line, 1) for line in _lines(ran, "Check your results", 0)]
    assert rates[1] == pytest.approx(rates[0], rel=1e-4)
    assert rates[2] == pytest.approx(rates[1], rel=1e-5)


def test_the_fits_leave_a_phase_far_below_the_limit(ran):
    (line,) = _lines(ran, "Check your results", 1)
    assert max(dp.numbers(line.replace("[", " ").replace("]", " "))) < 1e-3  # 0.1 rad


def test_the_coupling_is_flat_over_the_band(ran):
    (line,) = _lines(ran, "Check your results", 2)
    assert abs(_number(line)) < 1.0  # per cent


def test_the_default_port_mode_tolerance_is_far_inside_the_phase_limit(ran):
    lines = _lines(ran, "Check your results", 3)
    # even a tolerance 1000x looser keeps beta L a tenth of the 0.1 rad limit
    assert all(_number(line, 2) < 0.1 for line in lines)
    assert _number(lines[0], 2) < _number(lines[1], 2)  # the pump's band is the harder


def test_the_pair_density_is_the_classical_conversion_efficiency(ran):
    (line,) = _lines(ran, "Check your results", 4)
    centre, classical = _number(line, 0), _number(line, 1)
    assert centre == pytest.approx(classical, rel=1e-5)  # no free factor
