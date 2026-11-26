"""The waveguide bends recipe runs and prints what the page shows.

docs/recipes/waveguide-bends.md runs nothing at docs build time. This runs its
blocks in order, on the strip section built by the repository's own
docs/recipes/strip_section.py (the page downloads it), and checks the numbers
against what the physics fixes: the loss per 90 degrees falls with the radius
and its gradient is the slope of that fall, the junction passes less than all
the power, a ring's bent group index and free spectral range sit within a
fraction of a percent of the straight guide's, and the layer moves the loss by
a few percent only.
"""

import itertools
import math
import re

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "waveguide-bends.md"

# a page of eight bent and two straight solves: minutes, not seconds
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("waveguide_bends")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _lines(ran, section: str, index: int = 0) -> list[str]:
    """The printed lines of the section's ``index``-th block that prints."""
    found = [e for e in ran if e.section == section and e.printed.strip()]
    return found[index].printed.splitlines()


def _after(line: str, key: str) -> float:
    """The number that follows ``key`` in a printed line."""
    return float(re.search(re.escape(key) + r"\s*([-+]?[\d.]+(?:e[-+]?\d+)?)", line)[1])


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_bent_strip_loses_a_fraction_of_a_db_with_its_caustic_inside(ran):
    _, first, second = _lines(ran, "A bent strip")
    assert _after(first, "n_eff") > 2.4473  # above the straight strip's
    assert 0.01 < _after(first, "loss") < 0.2  # dB per 90 degrees
    assert _after(second, "x =") < 2.0  # the layer starts past the caustic


def test_the_loss_falls_with_the_radius_at_the_rate_the_gradient_gives(ran):
    lines = _lines(ran, "Loss against radius")
    losses = [_after(line, ":") for line in lines[:3]]
    assert all(a > b for a, b in itertools.pairwise(losses))
    assert _after(lines[3], "um:") < 0  # dB per um
    # d ln L / dR at 1.5 um against the secant of ln L between 1.25 and 1.5 um:
    # the secant is the derivative at the midpoint, and ln L bends by about
    # 10 % of its slope over this step (a quadratic term), so 25 % is a margin
    secant = math.log(losses[2] / losses[1]) / 0.25
    assert _after(lines[3], ",") == pytest.approx(secant, rel=0.25)


def test_the_junction_passes_less_than_all_the_power(ran):
    first, second = _lines(ran, "Junction with a straight guide")
    tau2, junction = _after(first, "tau^2 ="), _after(first, "loss")
    assert 0.9 < tau2 < 1.0
    assert junction == pytest.approx(-10 * math.log10(tau2), abs=1e-4)
    assert _after(second, "bend:") > 2 * junction


def test_the_ring_parameters_are_those_of_a_bent_ring(ran):
    lines = _lines(ran, "A microring")
    for bent, straight in ((lines[0], lines[1]), (lines[3], lines[4])):
        # the group index of the bend is the straight guide's to 5 %, its
        # free spectral range to 1e-3, and the bend lowers |D2| of an
        # anomalous ring
        assert _after(bent, "n_g") == pytest.approx(_after(straight, "n_g"), rel=0.05)
        assert _after(bent, "D1/2pi") == pytest.approx(
            _after(straight, "D1/2pi"), rel=1e-3
        )
        assert 0 < _after(bent, "D2/2pi") < _after(straight, "D2/2pi")
    # the straight ten-micron ring is the one of the Kerr recipe
    assert _after(lines[4], "D1/2pi") == pytest.approx(1137, rel=0.01)
    assert _after(lines[4], "g0/2pi") == pytest.approx(293, rel=0.03)
    for line in (lines[2], lines[5]):
        assert "True" in line  # the radiation Q is above 1e12


def test_the_layer_moves_the_loss_by_a_few_percent(ran):
    lines = _lines(ran, "Check your results")
    assert all(abs(float(re.search(r"\(([-+][\d.]+)%", x)[1])) < 5 for x in lines[:2])
    # order 4 against order 3: the bend of 1 um is not converged to the digit
    assert _after(lines[2], "order 4:") == pytest.approx(
        _after(lines[2], "order 3:"), rel=0.05
    )
