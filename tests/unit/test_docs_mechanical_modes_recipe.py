"""The mechanical modes recipe runs and prints what the page shows.

docs/recipes/mechanical-modes.md runs nothing at docs build time. This runs its
blocks in order on a small plate built by the repository's own
docs/recipes/plate_section.py (the page downloads it), checks that the output
blocks are what the code prints, that the uniform quality factor is the
material's for every mode, that the group velocity is the band's slope, and that
a damping frame lowers every Q.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "mechanical-modes.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("mechanical_modes")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _example(ran, section: str) -> dp.Example:
    (example,) = [example for example in ran if example.section == section]
    return example


def test_every_output_block_is_what_the_code_prints(ran):
    # the page prints four significant digits: 1e-3 relative covers the rounding
    # (5e-5 of the smallest printed value) and the last digit's platform noise
    assert not dp.output_problems(ran, tolerance=1e-3), "\n".join(
        dp.output_problems(ran, tolerance=1e-3)
    )


def test_a_uniform_quality_factor_is_every_modes(ran):
    printed = _example(ran, "A solid and its modes").printed
    assert dp.numbers(printed.split("Q ")[1])[0] == pytest.approx(1000.0)
    assert printed.count("(Q 1000)") == 3


def test_the_group_velocity_is_the_slope_of_the_band(ran):
    printed = _example(ran, "Group velocity").printed
    group, difference = dp.numbers(printed)[:2]
    assert group == pytest.approx(difference, rel=1e-4)


def test_a_damping_frame_lowers_every_q(ran):
    printed = _example(ran, "A damping frame").printed
    qualities = [dp.numbers(line.split("Q =")[1])[0] for line in printed.splitlines()]
    assert len(qualities) == 3
    assert max(qualities) < 100.0
