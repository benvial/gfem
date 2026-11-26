"""The Raman gain recipe runs and prints what the page shows.

docs/recipes/raman-gain.md runs nothing at docs build time. This runs its
blocks in order, on the strip section built by the repository's own
docs/recipes/strip_section.py (the page downloads it), and checks the numbers
against what the physics fixes: the selection rule of a wire turned from
[110] to [100], the Manley-Rowe relation between the Stokes gain and the pump
loss, and the ordering of the silicon, TM and silica gains.
"""

import re

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "raman-gain.md"
SECTION = "The gain of a silicon wire"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("raman_gain")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _lines(ran, section: str, index: int = 0) -> list[str]:
    found = [example for example in ran if example.section == section]
    return found[index].printed.splitlines()


def _gains(ran):
    """TE0's and TM0's printed gains: along [110], along [100], silica cladding."""
    lines = _lines(ran, SECTION, 1)
    # the brackets of the crystal directions are numbers to the reader: drop them
    return tuple(dp.numbers(re.sub(r"\[\d+\]", "", line)) for line in lines)


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_modes_are_te0_and_tm0_with_the_pump_above_the_stokes_wave(ran):
    te, tm = (dp.numbers(part) for part in _lines(ran, SECTION)[0].split("TM0"))
    # TE0 sits above the oxide's index 1.44 and TM0 below it; both rise with
    # the frequency, and the pump is 15.6 THz above the Stokes wave
    assert te[0] == pytest.approx(2.447, abs=1e-3)
    assert tm[2] == pytest.approx(1.913, abs=1e-3)
    assert te[0] < te[1]
    assert tm[0] < tm[1]
    assert tm[0] < te[0]


def test_the_te0_gain_of_a_wire_along_110_is_the_bulk_gain_over_a_tenth_of_a_um2(ran):
    te, tm = _gains(ran)
    # 20 cm/GW over an effective area of about 0.1 um^2 is 2e3 /(W m)
    assert te[0] == pytest.approx(1749, rel=0.02)
    # the longitudinal field of TM0 gives a fifth of TE0's gain
    assert tm[0] / te[0] == pytest.approx(0.22, abs=0.03)


def test_a_wire_along_100_has_almost_no_te_gain(ran):
    te, tm = _gains(ran)
    # only the hybrid mode's longitudinal field is left: under a percent
    assert te[1] < 0.01 * te[0]
    assert tm[1] < 0.1 * tm[0]


def test_the_silica_cladding_adds_far_less_than_the_silicon_line(ran):
    te, tm = _gains(ran)
    assert 0 < te[2] < 1e-3 * te[0]
    assert 0 < tm[2] < 1e-2 * tm[0]


def test_the_pump_loses_what_the_stokes_wave_gains_in_photons(ran):
    lines = _lines(ran, "Pump loss and the Stokes column")
    stokes, loss = dp.numbers(lines[0])
    assert loss < 0 < stokes
    # the page's grid frequencies: 1.55 um and 15.6 THz above it
    ratio = 1 + 15.6e12 / (299792458 / 1.55e-6)
    # an identity of one matrix; the printed numbers round at 1e-2 of 1.9e3
    assert -loss / stokes == pytest.approx(ratio, rel=1e-5)
    assert dp.numbers(lines[1])[-1] == pytest.approx(loss, abs=1e-2)
