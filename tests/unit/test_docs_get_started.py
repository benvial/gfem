"""The get-started page's first solve runs and prints what the page shows.

docs/get-started.md runs nothing at docs build time, and README.md copies its
first-solve block verbatim. This runs that block as a reader would, with the
mesh download swapped for the repository's own docs/tutorials/data/line.msh,
and checks the printed |S21| against the transmission-line formula for a
20 mm, eps_r = 10 section between two matched lines, and against the output
block the page shows.
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
PAGE = ROOT / "docs" / "get-started.md"
MESH = ROOT / "docs" / "tutorials" / "data" / "line.msh"
MESH_URL = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
)


def _section(page: str, heading: str) -> str:
    """The text under ``## heading`` up to the next ``## ``."""
    pattern = rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, page, re.MULTILINE | re.DOTALL)
    assert match, f"no section {heading!r} on the page"
    return match[1]


def _blocks(section: str, lang: str) -> list[str]:
    return re.findall(rf"^```{lang}\n(.*?)^```", section, re.MULTILINE | re.DOTALL)


def _line_formula(frequencies) -> np.ndarray:
    """|S21| of a 20 mm, eps_r = 10 parallel-plate section between matched lines."""
    c, eps, length = 299792458.0, 10.0, 20e-3
    z1 = 376.73 / 6
    z2 = z1 / np.sqrt(eps)
    bl = 2 * np.pi * np.asarray(frequencies) * np.sqrt(eps) / c * length
    s21 = 2 / (2 * np.cos(bl) + 1j * np.sin(bl) * (z2 / z1 + z1 / z2))
    return np.abs(s21)


@pytest.fixture(scope="module")
def first_solve():
    return _section(PAGE.read_text(encoding="utf-8"), "A first solve")


def test_the_first_solve_downloads_the_tutorial_mesh(first_solve):
    (code,) = _blocks(first_solve, "python")
    assert f'url = "{MESH_URL}"' in code
    assert MESH.is_file()


def test_the_first_solve_prints_the_transmission_line_values(
    first_solve, tmp_path, monkeypatch, capsys
):
    (code,) = _blocks(first_solve, "python")
    (shown,) = _blocks(first_solve, "text")

    def urlretrieve(url, filename):
        assert url == MESH_URL
        shutil.copy(MESH, filename)

    monkeypatch.setattr("urllib.request.urlretrieve", urlretrieve)
    monkeypatch.chdir(tmp_path)
    # The page's own code, run as a reader runs it.
    exec(compile(code, str(PAGE), "exec"), {"__name__": "get_started"})  # noqa: S102

    printed = np.array(capsys.readouterr().out.strip().strip("[]").split(), float)
    assert printed == pytest.approx(_line_formula((2e9, 5e9, 8e9)), abs=1e-3)
    assert printed == pytest.approx([0.8306, 0.9013, 0.6052], abs=1e-4)
    page_values = np.array(shown.strip().strip("[]").split(), float)
    assert page_values == pytest.approx(printed, abs=1e-6)
