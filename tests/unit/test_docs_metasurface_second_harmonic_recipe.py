"""The metasurface second-harmonic recipe runs and prints what the page shows.

docs/recipes/metasurface-second-harmonic.md runs nothing at docs build time. This
runs its blocks in order on the repository's own docs/recipes/grating_cell.py (the
page downloads it) and checks the numbers against what the physics fixes: which
orders propagate at the pump's and at the harmonic's wavelength, the energy the
source emits against the power the ports carry, the symmetry of the cell under a
tensor with $d_{33}$ alone and its loss with the whole tensor, and the agreement of
the gradient through both solves with a finite difference.
"""

import math

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "metasurface-second-harmonic.md"
#: Lengths of the cell and the two wavelengths, um; fused silica's index at each
#: (Malitson), rounded to 4 digits: only the sign of the margins below uses them.
PERIOD, PUMP, HARMONIC = 0.9, 1.55, 0.775
N_SILICA = {PUMP: 1.4440, HARMONIC: 1.4533}


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("metasurface_second_harmonic")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _lines(ran, section: str, index: int = 0) -> list[str]:
    found = [example for example in ran if example.section == section]
    return found[index].printed.splitlines()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_pitch_opens_the_first_orders_at_the_harmonic_only(ran):
    # sin(theta) = lambda / (n P): a first order propagates when it is below 1
    for medium in (1.0, N_SILICA[HARMONIC]):
        assert HARMONIC / (medium * PERIOD) < 1
    for medium in (1.0, N_SILICA[PUMP]):
        assert PUMP / (medium * PERIOD) > 1
    (line,) = _lines(ran, "The grating cell")
    assert line.startswith("12 ports")  # 3 orders, 2 polarizations, 2 faces
    assert "'0, 0'" in line
    assert "'-1, 0'" in line
    assert "'1, 0'" in line


def _top_and_bottom(ran) -> tuple[list[float], list[float]]:
    """The powers (nW) of orders (-1, 0), (0, 0), (1, 0) on each face."""
    lines = _lines(ran, "The power in each order")
    return tuple([dp.numbers(line)[k] for k in (2, 5, 8)] for line in lines[:2])


def test_the_pump_is_one_gigawatt_per_square_centimetre_in_a_cell(ran):
    _, power, _, _ = dp.numbers(_lines(ran, "The power in each order")[2])
    assert power == pytest.approx(1e13 * 0.9e-6 * 0.1e-6, rel=1e-12)


def test_the_efficiency_is_the_upward_power_over_the_pump(ran):
    up, power, eta, _ = dp.numbers(_lines(ran, "The power in each order")[2])
    top, _ = _top_and_bottom(ran)
    # each power is printed to 3 digits: the sum and the ratio are good to 1e-2/3
    assert up == pytest.approx(sum(top), rel=5e-3)
    assert eta == pytest.approx(up * 1e-9 / power, rel=5e-3)


def test_the_specular_order_leads_and_the_silica_side_is_weaker(ran):
    top, bottom = _top_and_bottom(ran)
    assert top[1] > top[0]
    assert top[1] > top[2]
    assert sum(bottom) < 0.1 * sum(top)
    # the polar axis along the grating vector: the whole tensor breaks the mirror
    assert abs(top[0] / top[2] - 1) > 1e-2


def test_the_gradient_agrees_with_a_central_difference(ran):
    (line, check) = _lines(ran, "Differentiate through both solves")
    assert dp.numbers(line)[-1] > 0
    assert check.endswith("True")


def test_the_orders_carry_the_power_the_source_emits(ran):
    emitted, total = dp.numbers(_lines(ran, "Check your results")[0])
    # measured 4e-5 of the emitted power on this mesh (the evanescent orders' leak
    # at the pads), plus 3e-4 from the 4 printed digits: tolerance 1e-3
    assert total == pytest.approx(emitted, rel=1e-3)


def test_one_polar_axis_term_only_gives_equal_plus_and_minus_orders(ran):
    ratio = dp.numbers(_lines(ran, "Check your results", 1)[0])[-1]
    # the mesh and the cell are mirror-symmetric: equal to the solver's round-off
    # (measured 1e-12), well inside the 6 printed digits
    assert ratio == pytest.approx(1.0, abs=1e-6)


def test_the_first_orders_leave_at_the_angle_of_the_grating_equation():
    angle = math.degrees(math.asin(HARMONIC / PERIOD))
    assert angle == pytest.approx(59.4, abs=0.05)  # the page says 59 degrees
