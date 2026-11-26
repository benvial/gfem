"""The free-carriers recipe runs and prints what the page shows.

docs/recipes/free-carriers.md runs nothing at docs build time. This runs its
blocks in order, on the strip section built by the repository's own
docs/recipes/strip_section.py (the page downloads it), and checks the numbers
against what the physics fixes: the core overlap of a strongly guided mode,
the 1/lambda^2-type trend of the generation, and the continuous-wave loss
that follows from the generation, the cross-section and the lifetime.
"""

import docs_pages as dp
import numpy as np
import pytest

PAGE = dp.DOCS / "recipes" / "free-carriers.md"
SECTION_FIGURES = "The figures of a mode"
SECTION_HANDOFF = "Handoff to a pulse equation"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("free_carriers")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _lines(ran, section: str) -> list[str]:
    return next(e for e in ran if e.section == section).printed.splitlines()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_overlap_of_the_core_exceeds_one_and_falls_with_the_wavelength(ran):
    rows = [dp.numbers(line) for line in _lines(ran, SECTION_FIGURES)[2:]]
    overlap = [row[1] for row in rows]
    # n_g > n_core in a strongly guided wire, so the core overlap is above 1;
    # the mode spreads into the oxide as the wavelength grows
    assert all(1.0 < g < 1.2 for g in overlap)
    assert overlap[0] > overlap[1] > overlap[2]


def test_the_core_area_is_the_strip_cross_section(ran):
    area = dp.numbers(_lines(ran, SECTION_FIGURES)[1])[0]
    # 500 nm x 220 nm, printed to 1e-4 um^2
    assert area == pytest.approx(0.5 * 0.22, abs=1e-4)


def test_the_generation_is_the_two_photon_loss_per_carrier_pair(ran):
    rows = [dp.numbers(line) for line in _lines(ran, SECTION_FIGURES)[2:]]
    generation = np.array([row[2] for row in rows])
    # beta_TPA / (2 hbar omega A_eff^2) with A_eff of order the core area
    # 0.11 um^2: 8e-12 / (2 * 1.28e-19 * (0.11e-12)^2) = 2.6e33 /(W^2 m^3 s);
    # the mode's own area changes it by tens of percent at most
    assert np.all((generation > 2e33) & (generation < 5e33))


def test_the_continuous_wave_loss_follows_from_the_printed_figures(ran):
    figures = [dp.numbers(line) for line in _lines(ran, SECTION_FIGURES)[2:]]
    handoff = [dp.numbers(line) for line in _lines(ran, SECTION_HANDOFF)]
    tau, power = 1e-9, 0.1
    for fig, hand in zip(figures, handoff, strict=True):
        wl_nm, overlap, generation = fig[0], fig[1], fig[2]
        sigma = 1.45e-21 * (wl_nm / 1550) ** 2
        density = generation * tau * power**2
        # the figures print four digits at worst (1.075 is 5e-4 relative, 3.10e22 is
        # 1.6e-3): a product of three is good to 4e-3, inside 5e-3
        assert hand[1] == pytest.approx(density, rel=5e-3)
        loss = 4.343e-2 * overlap * sigma * density
        assert hand[-1] == pytest.approx(loss, rel=5e-3)
    # lambda^2 scaling of sigma: the loss grows with the wavelength
    assert handoff[0][-1] < handoff[1][-1] < handoff[2][-1]
