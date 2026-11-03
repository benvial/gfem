"""The distributed solve recipe's script, launched as the page launches it.

docs/recipes/distributed-solve.md includes tools/distributed_line.py as it is, so
this run is the page's. Its output agrees with one process to the solver's gate.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ranks import on_ranks_argv

REPO = Path(__file__).parents[2]
SCRIPT = REPO / "tools" / "distributed_line.py"
MESH = REPO / "docs" / "tutorials" / "data" / "line.msh"
NUMBER = r"[-+]?\d\.\d+(?:e[-+]\d+)?"


def _numbers(log: str) -> np.ndarray:
    lines = [
        re.findall(NUMBER, line)
        for line in log.splitlines()
        if line.startswith(("S21", "|S21|", "gradient"))
    ]
    return np.array([float(x) for line in lines for x in line])


def _in(directory: Path) -> Path:
    directory.mkdir()
    shutil.copy(MESH, directory)
    return directory


def _check(tmp_path: Path, solver: str, gate: float) -> None:
    one = subprocess.run(
        [sys.executable, str(SCRIPT), solver],
        cwd=_in(tmp_path / "one"),
        capture_output=True,
        text=True,
        check=True,
    )
    work = _in(tmp_path / "ranks")
    code, log = on_ranks_argv(
        [sys.executable, str(SCRIPT), solver], ranks=3, limit=300, cwd=work
    )
    assert code == 0, log
    assert log.count("ranks 3,") == 1, log  # rank 0 alone prints
    want, got = _numbers(one.stdout), _numbers(log)
    assert len(got) == len(want) > 0, log
    assert np.max(np.abs(got - want) / np.abs(want)) < gate, (got, want)
    assert (work / f"line-{solver}.s2p").is_file()
    assert (work / f"e-{solver}.vtu").is_file()


def test_the_recipes_script_on_bddc_matches_one_process(tmp_path):
    _check(tmp_path, "bddc_gmres", 1e-7)


@pytest.mark.petsc
def test_the_recipes_script_on_mumps_matches_one_process(tmp_path):
    _check(tmp_path, "direct", 1e-9)
