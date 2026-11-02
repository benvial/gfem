"""The periodic-cell recipe states only what the tests assert.

docs/recipes/periodic-cell.md runs nothing at docs build time, so its
numbers are static. Each tolerance it prints next to a case id is the one
the case's test asserts: the verification cases' constants live in
`periodic_cases`, the Palace agreements in tests/reference/test_periodic.py,
read here without importing it (ADR 0017: no Palace data in the default
tier). The page's public names and its Bloch phase are checked too.
"""

import ast
import re
from pathlib import Path

import periodic_cases as pc
import pytest

import gfem

ROOT = Path(__file__).parents[2]
PAGE = ROOT / "docs" / "recipes" / "periodic-cell.md"
REFERENCE = ROOT / "tests" / "reference" / "test_periodic.py"


def _reference_constants() -> dict:
    tree = ast.parse(REFERENCE.read_text())
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("BLOCH_")
    }


def _number(value: float) -> str:
    """``value`` as the page writes it: one digit and a short exponent, 3e-4."""
    mantissa, exponent = f"{value:.0e}".split("e")
    text = f"{mantissa}e{int(exponent)}"
    assert float(text) == value, f"{value} needs more digits than {text}"
    return text


def _claims() -> list[tuple[str, str, float]]:
    """(case id on the page, constant, value) of every number the page claims."""
    reference = _reference_constants()
    return [
        ("C1", "C1_S_TOL", pc.C1_S_TOL),
        ("C1, loaded", "C1_EIGEN_RTOL", pc.C1_EIGEN_RTOL),
        ("C2", "C2_BAND_RTOL", pc.C2_BAND_RTOL),
        ("C2", "C2_PHASE_TOL", pc.C2_PHASE_TOL),
        ("C3", "C3_FREQ_RTOL", pc.C3_FREQ_RTOL),
        ("C4", "C4_RTOL", pc.C4_RTOL),
        ("R1", "BLOCH_EIGEN_AGREEMENT", reference["BLOCH_EIGEN_AGREEMENT"]),
        ("R2", "BLOCH_S_AGREEMENT", reference["BLOCH_S_AGREEMENT"]),
    ]


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text()


def _bullet(page: str, case: str) -> str:
    """The bullet of ``case`` (``- **<case>**: ...``) up to the next item."""
    match = re.search(
        rf"^ *- \*\*{re.escape(case)}\*\*:(.*?)(?=^ *- |\n\n)",
        page,
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"no bullet for {case} on the page"
    return " ".join(match[1].split())


def test_the_number_formatting_matches_the_page():
    assert [_number(v) for v in (1e-5, 3e-4, 0.01, 2e-6)] == [
        "1e-5",
        "3e-4",
        "1e-2",
        "2e-6",
    ]


def test_the_reference_constants_are_read_without_importing():
    assert set(_reference_constants()) == {
        "BLOCH_EIGEN_AGREEMENT",
        "BLOCH_S_AGREEMENT",
    }


@pytest.mark.parametrize(
    ("case", "name", "value"), _claims(), ids=[c[1] for c in _claims()]
)
def test_each_case_states_the_tolerance_its_test_asserts(page, case, name, value):
    assert re.search(
        rf"(?<![\d.e-]){re.escape(_number(value))}(?![\d])", _bullet(page, case)
    ), f"{case}: the page does not state {name} = {_number(value)}"


def test_every_gfem_name_on_the_page_is_public(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert names
    assert names - {"conventions", "shape"} <= set(gfem.__all__)


def test_the_page_states_the_bloch_phase_convention(page):
    assert gfem.conventions.BLOCH_PHASE in page
