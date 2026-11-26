"""The effective-medium recipe names public objects and leaves test evidence out.

docs/recipes/effective-medium.md runs nothing at docs build time: its mesh is the
reader's own. This reads its code without solving: blocks parse,
`gfem.diffraction` names are public, keywords bind to the signatures.
"""

import ast
import inspect
import re

import docs_pages as dp
import pytest

import gfem
from gfem import diffraction as dif

PAGE = dp.DOCS / "recipes" / "effective-medium.md"


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


def test_the_blocks_parse_and_bind_to_the_signatures(page):
    assert dp.blocks(page, "python")
    assert dp.static_problems(page) == []


def test_every_diffraction_name_on_the_page_is_public(page):
    names = set(re.findall(r"\b(?:dif|gfem\.diffraction)\.([A-Za-z_]\w*)", page))
    assert {"floquet_ports", "effective_medium"} <= names
    assert names <= set(dif.__all__)


def test_every_keyword_of_a_diffraction_call_is_a_parameter(page):
    calls = 0
    for block in dp.blocks(page, "python"):
        for node in ast.walk(ast.parse(block)):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            value = node.func.value
            if isinstance(value, ast.Name) and value.id == "dif":
                parameters = inspect.signature(getattr(dif, node.func.attr)).parameters
                calls += 1
                assert {k.arg for k in node.keywords} <= set(parameters)
    assert calls >= 3


def test_the_fields_the_page_reads_exist(page):
    fields = set(re.findall(r"\bem\.(\w+)", page))
    assert {"permittivity", "permeability", "index", "impedance"} <= fields
    assert fields <= set(dif.EffectiveMedium.__dataclass_fields__)


def test_the_page_quotes_no_case_id_and_names_no_other_solver(page):
    assert not re.search(r"\*\*[CFR]\d\b", page)
    assert "tests/" not in page
    for word in ("palace", "flexcompute", "tidy3d"):
        assert word not in page.lower()


@pytest.mark.parametrize(
    "text",
    ["branch=1", "NonDifferentiableError", "{doc}`floquet-ports`", "jax.grad"],
)
def test_the_page_carries(page, text):
    assert text in " ".join(page.split()), text


@pytest.mark.parametrize("heading", ["Choose the branch", "Limits", "Next"])
def test_the_page_has_its_sections(page, heading):
    assert dp.section(page, heading).strip()


def test_the_names_the_page_calls_exist_in_gfem(page):
    names = set(re.findall(r"\bgfem\.([A-Za-z_]\w*)", page))
    assert names - {"conventions", "diffraction"} <= set(gfem.__all__)
