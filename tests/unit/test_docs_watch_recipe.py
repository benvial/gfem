"""The "Solver progress and logs" recipe shows the log lines its code writes.

docs/recipes/watch-a-solve.md runs nothing at docs build time. This runs its
Python blocks in order, as a reader does, on the repository's own
docs/tutorials/data/line.msh, and reads what each block writes to stderr,
where the log display draws. A block with an output block under it writes
those lines: the same words in the same order, with the clock and every
number masked, since timings and the machine's memory budget change from run
to run. The solve lines are compared whole but for the clock. A block without
an output block writes nothing.
"""

import contextlib
import io
import re
import shutil
import urllib.request

import docs_pages as dp
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "watch-a-solve.md"
CLOCK = re.compile(r"^\d\d:\d\d:\d\d\.\d{3} ", re.MULTILINE)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _lines(text: str) -> list[str]:
    """The log lines of ``text``, colour codes stripped and the clock masked."""
    plain = ANSI.sub("", text).replace("\r", "")
    return [CLOCK.sub("<clock> ", line) for line in plain.splitlines() if line]


def _words(line: str) -> str:
    return dp._NUMBER.sub("#", line)


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """Each Python block of the page with what it wrote to stderr."""
    workdir = tmp_path_factory.mktemp("watch_a_solve")

    def urlretrieve(url, filename):
        shutil.copy(dp.MESHES[url], workdir / filename)

    # The page's lines start from a process that holds no factorization.
    gfem.release_memory()
    namespace = {"__name__": PAGE.stem}
    written = []
    original = urllib.request.urlretrieve
    urllib.request.urlretrieve = urlretrieve
    try:
        with contextlib.chdir(workdir):
            for example in dp.examples(PAGE.read_text(encoding="utf-8")):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    exec(compile(example.code, str(PAGE), "exec"), namespace)  # noqa: S102
                written.append((example, err.getvalue()))
    finally:
        urllib.request.urlretrieve = original
        gfem.log.disable()
    return written


def test_the_page_downloads_the_tutorial_mesh():
    first = dp.examples(PAGE.read_text(encoding="utf-8"))[0]
    assert f'url = "{dp.MESH_URL}"' in first.code
    assert "gfem.log.enable()" in first.code


def test_the_first_block_writes_the_lines_the_page_shows(ran):
    example, written = ran[0]
    assert example.shown is not None
    printed, shown = _lines(written), _lines(example.shown)
    assert [_words(line) for line in printed] == [_words(line) for line in shown]
    solves = [line for line in shown if " solve at " in line]
    assert len(solves) == 3
    assert [line for line in printed if " solve at " in line] == solves


@pytest.mark.parametrize("number", range(1, 5))
def test_every_other_block_writes_what_the_page_shows(ran, number):
    example, written = ran[number]
    if example.shown is None:
        assert not written, f"{example.section}: writes\n{written}"
        return
    printed, shown = _lines(written), _lines(example.shown)
    assert [_words(line) for line in printed] == [_words(line) for line in shown]


def test_the_page_runs_one_block_per_task(ran):
    assert [example.section for example, _ in ran] == [
        "Turn it on",
        "How long each phase took",
        "Under MPI",
        "With your own handlers",
        "Turn it off",
    ]


def test_the_page_names_no_other_tool():
    text = PAGE.read_text(encoding="utf-8").lower()
    assert "palace" not in text
    assert "tqdm" not in text
