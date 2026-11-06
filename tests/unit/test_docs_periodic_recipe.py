"""The periodic-cell recipe names public objects and leaves test evidence out.

docs/recipes/periodic-cell.md runs nothing at docs build time. How gfem's
periodic solves are checked is on docs/verification.md, in words; the recipe
quotes no test tolerance and no case id (ADR 0017, as amended). Its public
names and its Bloch phase are checked here.
"""

import re
from pathlib import Path

import periodic_cases as pc
import pytest

import gfem

ROOT = Path(__file__).parents[2]
PAGE = ROOT / "docs" / "recipes" / "periodic-cell.md"

#: Every tolerance the periodic verification cases assert.
TOLERANCES = (
    pc.C1_S_TOL,
    pc.C1_EIGEN_RTOL,
    pc.C2_BAND_RTOL,
    pc.C2_PHASE_TOL,
    pc.C3_FREQ_RTOL,
    pc.C4_RTOL,
)


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


def test_the_page_quotes_no_case_id(page):
    assert not re.search(r"\*\*[CR]\d\b", page)


@pytest.mark.parametrize("value", TOLERANCES)
def test_the_page_quotes_no_test_tolerance(page, value):
    mantissa, exponent = f"{value:.0e}".split("e")
    written = f"{mantissa}e{int(exponent)}"
    assert not re.search(rf"(?<![\d.e-]){re.escape(written)}(?!\d)", page)


def test_the_page_names_no_test_file(page):
    assert "tests/" not in page


def test_every_gfem_name_on_the_page_is_public(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert names
    assert names - {"conventions", "shape"} <= set(gfem.__all__)


def test_the_page_states_the_bloch_phase_convention(page):
    assert gfem.conventions.BLOCH_PHASE in page
