"""The distributed solve recipe: what it shows matches what ran and what exists.

docs/recipes/distributed-solve.md runs nothing at docs build time. The launched
`tests/mpi/test_recipe.py` runs its script; its job scripts ran once on MareNostrum 5
as written. This keeps those blocks in the shape that ran, and the page's names real.
"""

import ast
import re
import tomllib
from pathlib import Path

import pytest

import gfem

REPO = Path(__file__).parents[2]
PAGE = REPO / "docs" / "recipes" / "distributed-solve.md"
SCRIPT = REPO / "tools" / "distributed_line.py"


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


def _blocks(page: str, language: str) -> list[str]:
    return re.findall(
        rf"^```{language}\n(.*?)^```$", page, flags=re.MULTILINE | re.DOTALL
    )


def _jobs(page: str) -> list[str]:
    return [b for b in _blocks(page, "bash") if b.startswith("#!/bin/bash")]


def test_the_page_is_in_both_tables_of_contents():
    assert "recipes/distributed-solve" in (REPO / "docs" / "index.md").read_text()
    assert "{doc}`distributed-solve`" in (REPO / "docs/recipes/index.md").read_text()


@pytest.mark.parametrize("private", [".scratch", "mn5run", "ticket", "ehpc", "resl"])
def test_no_private_path_or_name_reaches_the_page(page, private):
    # MyST comments never render
    published = re.sub(r"^%.*$", "", page, flags=re.MULTILINE)
    assert private not in published


def test_the_page_takes_its_refusals_from_the_generator(page):
    assert "```{gfem-distributed-refusals}\n```" in page


def test_the_scale_section_waits_for_its_runs(page):
    assert "(distributed-scale)=\n## Scale" in page
    assert "To be filled" in page


def test_the_page_includes_the_script_it_downloads(page):
    assert "```{literalinclude} ../../tools/distributed_line.py" in page
    ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assert (REPO / "docs" / "tutorials" / "data" / "line.msh").is_file()


def test_every_python_block_parses(page):
    for block in _blocks(page, "python"):
        ast.parse(block)


def test_there_are_a_build_job_and_a_run_job(page):
    build, run = _jobs(page)
    assert "--mpi --sources sources wheelhouse-mpi" in build
    assert "srun --mpi=pmix" in run


@pytest.mark.parametrize("index", [0, 1])
def test_a_job_leaves_intel_mpi_behind_for_the_stacks_open_mpi(page, index):
    job = _jobs(page)[index]
    assert job.index("module purge") < job.index("openmpi/4.1.6-gcc")


def test_the_run_job_has_the_layout_the_page_recommends(page):
    _, run = _jobs(page)
    assert "--ntasks-per-node=28 --cpus-per-task=4" in run
    assert "--cpu-bind" not in run
    assert "OMP_NUM_THREADS" not in run


def test_the_extra_is_the_one_the_page_names(page):
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    (spec,) = project["optional-dependencies"]["mpi"]
    assert spec.startswith("mpi4py>=4.1.2,<5;")
    assert 'pip install "gfem[mpi]"' in page


def test_the_page_names_public_objects():
    for name in (
        "DistributedUnsupportedError",
        "SolverUnavailableError",
        "LinearSolverConvergenceError",
        "load_result",
        "interface_participation",
        "characteristic_impedance",
        "PortMode",
    ):
        assert hasattr(gfem, name), name
