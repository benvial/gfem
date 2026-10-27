"""The PETSc and SLEPc recipe shows only options that a test runs.

docs/recipes/petsc-slepc.md runs nothing at docs build time (the docs build
has no PETSc), so its snippets are static. Each options dict it shows lives
in `petsc_recipe`, where the ``petsc`` / ``slepc`` tests solve with it; this
checks that the page shows them exactly, so the page and the tests cannot
drift apart.
"""

import json
from pathlib import Path

import petsc_recipe
import pytest

PAGE = Path(__file__).parents[2] / "docs" / "recipes" / "petsc-slepc.md"


def _python(options: dict) -> str:
    """``options`` as the page writes a Python dict: double quotes, one line."""
    items = (
        f"{json.dumps(k)}: {json.dumps(v) if isinstance(v, str) else repr(v)}"
        for k, v in options.items()
    )
    return "{" + ", ".join(items) + "}"


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text()


@pytest.mark.parametrize(
    "name", ["MUMPS_LU", "KSP_VIEW", "SLEPC_SUBSPACE", "KRYLOV_REUSE"]
)
def test_the_page_shows_each_tested_python_dict(page, name):
    assert _python(getattr(petsc_recipe, name)) in page


def test_the_page_shows_the_tested_job_options_as_json(page):
    assert f'"petsc_options": {json.dumps(petsc_recipe.JOB)}' in page


def test_the_page_selects_both_solvers_as_the_tests_do(page):
    kwargs = petsc_recipe.SOLVER_OPTIONS["canonical"]
    call = ", ".join(f"{k}={json.dumps(v)}" for k, v in kwargs.items())
    assert f"gfem.SolverOptions({call})" in page
