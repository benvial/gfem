"""The crystal Kerr recipe runs and prints what the page shows.

docs/recipes/crystal-kerr.md runs nothing at docs build time. This runs its
blocks in order, on the strip section built by the repository's own
docs/recipes/strip_section.py (the page downloads it), and checks the numbers
against what the physics fixes: a guide along [110] over one along [100] is
bounded by the uniform-field factor (1 + rho) / 2 and above 1, and the index
matrix of a birefringent core is the scalar index along the axis its mode
sees.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "crystal-kerr.md"
RHO = 1.27


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("crystal_kerr")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _lines(ran, section: str, index: int = 0) -> list[str]:
    found = [example for example in ran if example.section == section]
    return found[index].printed.splitlines()


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_a_diagonal_guide_is_between_one_and_the_uniform_field_factor(ran):
    lines = _lines(ran, "Silicon: one number for the anisotropy")
    _, g_100, _, g_110 = dp.numbers(lines[0])[:4]  # the labels are numbers too
    ratio, uniform = dp.numbers(lines[1])[-2:]
    assert ratio == pytest.approx(
        g_110 / g_100, rel=1e-3
    )  # gammas printed to 4 digits: 6e-4 relative
    assert uniform == pytest.approx((1 + RHO) / 2, abs=1e-12)
    # a hybrid mode has vertical and longitudinal parts that see other factors
    assert 1.0 < ratio < uniform


def test_the_scalar_index_of_the_axis_a_mode_sees_is_the_index_matrix(ran):
    te, tm = _lines(ran, "Lithium niobate: the index matrix")
    # TE sees the ordinary index, TM mostly the extraordinary: the scalar n_e route
    # is low by about n_e^2/n_o^2 - 1 = -6.5 % for TE and by the small ordinary
    # share of the TM field (-0.34 % measured; gate 1 %), inside the 7.0 % any
    # single index can differ by
    assert -0.01 < dp.numbers(tm)[-1] < 0.0
    assert -0.07 < dp.numbers(te)[-1] < -0.05
