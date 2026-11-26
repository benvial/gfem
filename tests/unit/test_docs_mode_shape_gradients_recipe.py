"""The shape-gradients recipe runs and prints what the page shows.

docs/recipes/mode-shape-gradients.md runs nothing at docs build time. This runs
its blocks in order, on the filled WR-90 guide the page downloads, and checks
the printed numbers against what the physics fixes: the stencil against the
reported group index and dispersion, the gradient against the closed form of a
scaled port face.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "mode-shape-gradients.md"
SECTION_STENCIL = "The stencil"
SECTION_GRADIENT = "The gradient with respect to the face"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("mode_shape_gradients")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _lines(ran, section: str) -> list[str]:
    return next(e for e in ran if e.section == section).printed.splitlines()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_group_index_stencil_agrees_with_the_reported_value_to_its_truncation(ran):
    n_g, reported = dp.numbers(_lines(ran, SECTION_STENCIL)[0])
    # truncation of the 3-point stencil at delta = 1e-3: 3.0e-7 relative
    # (measured); two numbers printed to 6 decimals add 2 * 5e-7 / 1.79 =
    # 5.6e-7; the sum 8.6e-7 gets a gate of 2e-6
    assert n_g == pytest.approx(reported, rel=2e-6, abs=0)


def test_the_dispersion_stencil_agrees_with_the_reported_value(ran):
    beta_2, reported = dp.numbers(_lines(ran, SECTION_STENCIL)[1])[:2]
    # the 5-point stencil at delta = 3e-3 differs by 1.8e-9 relative
    # (measured); two numbers printed to 7 digits (1e-7 relative each at
    # worst, 5.06e-20 to 1e-26) add 2e-7: gate 3e-7
    assert beta_2 == pytest.approx(reported, rel=3e-7, abs=0)


def test_the_group_index_gradient_follows_the_closed_form(ran):
    adjoint, closed = dp.numbers(_lines(ran, SECTION_GRADIENT)[0])
    # the stencil's gradient is off the true derivative by its O(delta^2)
    # truncation, measured 2.9e-6 relative at delta = 1e-3, plus two numbers
    # printed to 6 decimals, 2 * 5e-7 / 0.76 = 1.3e-6; gate 1e-5
    assert adjoint == pytest.approx(closed, rel=1e-5, abs=0)
