"""The distributed solve recipe: what it shows matches what ran.

docs/recipes/distributed-solve.md runs nothing at docs build time. The launched
`tests/mpi/test_recipe.py` runs the script. This keeps the page's blocks in the shape
that ran, and its names real.
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


def test_the_page_is_in_both_tables_of_contents():
    assert "recipes/distributed-solve\n" in (REPO / "docs" / "index.md").read_text()
    listing = (REPO / "docs/recipes/index.md").read_text()
    assert "- {doc}`distributed-solve`\n" in listing


def _published(text: str) -> str:
    """The text without MyST comments, which never render."""
    return re.sub(r"^%.*$", "", text, flags=re.MULTILINE)


@pytest.mark.parametrize("private", [".scratch", "mn5run", "ticket", "ehpc", "resl"])
def test_no_private_path_or_name_reaches_the_page(page, private):
    assert private not in _published(page)


# A commit SHA mixes hex letters and digits; a Slurm job id is a number after "job".
_SHA = r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b"
_JOB = r"(?i:\bjobs? (?:id )?[0-9]{6,})"


def _ids(text: str) -> list[str]:
    return re.findall(f"{_SHA}|{_JOB}", text)


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("gfem 009f5a6, on a 10.2 M DOF mesh", ["009f5a6"]),
        ("jobs 48441613 and 48441617 on one node", ["jobs 48441613"]),
        ("at 50 GHz (job 48473792, commit ec608c1)", ["job 48473792", "ec608c1"]),
        ("Job 48468681: 442 passed", ["Job 48468681"]),
        ("a 10200000 DOF mesh, 1234567 unknowns", []),
        ("the effaced and defaced words", []),
    ],
)
def test_the_id_check_finds_shas_and_job_ids_but_not_plain_numbers(text, found):
    assert _ids(text) == found


def test_no_commit_or_job_id_reaches_the_page(page):
    assert not _ids(_published(page))


def test_the_page_takes_its_refusals_from_the_generator(page):
    assert "```{gfem-distributed-refusals}\n```" in page


def test_the_page_sends_measured_times_to_performance(page):
    assert "{doc}`../project/scaling`" in page


def test_the_page_includes_the_script_it_downloads(page):
    assert "```{literalinclude} ../../tools/distributed_line.py" in page
    ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assert (REPO / "docs" / "tutorials" / "data" / "line.msh").is_file()


def test_every_python_block_parses(page):
    for block in _blocks(page, "python"):
        ast.parse(block)


def test_the_page_recommends_the_layout(page):
    assert "--ntasks-per-node=28 --cpus-per-task=4" in page


def test_the_extra_is_the_one_the_page_names(page):
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    (spec,) = project["optional-dependencies"]["mpi"]
    assert spec.startswith("mpi4py>=4.1.2,<5;")
    assert "{{ install_mpi }}" in page  # conf.py's install form, not PyPI's
    assert 'pip install "gfem[' not in page


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
