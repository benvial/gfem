"""The distributed solve and offline cluster recipes: what they show matches what ran.

docs/recipes/distributed-solve.md and docs/recipes/offline-cluster.md run nothing at
docs build time. The launched `tests/mpi/test_recipe.py` runs the script; the cluster
page's job scripts ran once on MareNostrum 5 as written. This keeps those blocks in the
shape that ran, and the pages' names real.
"""

import ast
import re
import tomllib
from pathlib import Path

import pytest

import gfem

REPO = Path(__file__).parents[2]
PAGE = REPO / "docs" / "recipes" / "distributed-solve.md"
CLUSTER = REPO / "docs" / "recipes" / "offline-cluster.md"
SCRIPT = REPO / "tools" / "distributed_line.py"


@pytest.fixture(scope="module")
def page():
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cluster():
    return CLUSTER.read_text(encoding="utf-8")


def _blocks(page: str, language: str) -> list[str]:
    return re.findall(
        rf"^```{language}\n(.*?)^```$", page, flags=re.MULTILINE | re.DOTALL
    )


def _jobs(page: str) -> list[str]:
    return [b for b in _blocks(page, "bash") if b.startswith("#!/bin/bash")]


def test_both_pages_are_in_both_tables_of_contents_cluster_right_after():
    toctree = (REPO / "docs" / "index.md").read_text()
    assert "recipes/distributed-solve\nrecipes/offline-cluster\n" in toctree
    listing = (REPO / "docs/recipes/index.md").read_text()
    assert "{doc}`distributed-solve`\n- {doc}`offline-cluster`\n" in listing


def _published(text: str) -> str:
    """The text without MyST comments, which never render."""
    return re.sub(r"^%.*$", "", text, flags=re.MULTILINE)


@pytest.mark.parametrize("which", ["page", "cluster"])
@pytest.mark.parametrize("private", [".scratch", "mn5run", "ticket", "ehpc", "resl"])
def test_no_private_path_or_name_reaches_the_pages(request, which, private):
    assert private not in _published(request.getfixturevalue(which))


@pytest.mark.parametrize("which", ["page", "cluster"])
def test_no_commit_or_job_id_reaches_the_pages(request, which):
    text = _published(request.getfixturevalue(which))
    assert not re.findall(r"\b[0-9a-f]{7,40}\b|job [0-9]{6,}|\b4[0-9]{7}\b", text)


def test_the_page_takes_its_refusals_from_the_generator(page):
    assert "```{gfem-distributed-refusals}\n```" in page


def test_the_page_sends_measured_times_to_performance_and_installs_to_the_cluster_page(
    page,
):
    assert "{doc}`../project/scaling`" in page
    assert "{doc}`offline-cluster`" in page


def test_the_page_includes_the_script_it_downloads(page):
    assert "```{literalinclude} ../../tools/distributed_line.py" in page
    ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assert (REPO / "docs" / "tutorials" / "data" / "line.msh").is_file()


def test_every_python_block_parses(page):
    for block in _blocks(page, "python"):
        ast.parse(block)


def test_the_cluster_page_has_a_build_job_and_a_run_job(cluster):
    build, run = _jobs(cluster)
    assert "--mpi --sources sources wheelhouse-mpi" in build
    assert "srun --mpi=pmix" in run


@pytest.mark.parametrize("index", [0, 1])
def test_a_job_leaves_intel_mpi_behind_for_the_stacks_open_mpi(cluster, index):
    job = _jobs(cluster)[index]
    assert job.index("module purge") < job.index("openmpi/4.1.6-gcc")


def test_the_run_job_has_the_layout_the_distributed_page_recommends(page, cluster):
    assert "--ntasks-per-node=28 --cpus-per-task=4" in page
    _, run = _jobs(cluster)
    assert "--ntasks-per-node=28 --cpus-per-task=4" in run
    assert "--cpu-bind" not in run
    assert "OMP_NUM_THREADS" not in run


def test_the_extra_is_the_one_the_pages_name(page, cluster):
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    (spec,) = project["optional-dependencies"]["mpi"]
    assert spec.startswith("mpi4py>=4.1.2,<5;")
    assert 'pip install "gfem[mpi]"' in page
    assert "[mpi]" in cluster


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
