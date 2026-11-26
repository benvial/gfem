"""The "Result files and exports" recipe runs and prints what the page shows.

docs/recipes/save-and-export.md runs nothing at docs build time. This runs its
blocks in order, as a reader does, on the repository's own
docs/tutorials/data/line.msh; its ``python -m gfem run`` line runs as a
subprocess of this interpreter. Every file the page writes is read back here
and compared with the solve: the Touchstone files with S and its 50 ohm
renormalization, the xarray and netCDF files, the VTU file, the job file's
mesh entry with the mesh file, and the result bundle with the in-process
result. The page's sentences about where the job puts its mesh are checked
against what the job file holds. The solve cache section is checked with its
keys elided: they change with every edit of gfem's code.
"""

import hashlib
import importlib.util
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import docs_pages as dp
import meshio
import numpy as np
import pytest

import gfem

PAGE = dp.DOCS / "recipes" / "save-and-export.md"
Z_LINE = 376.73 / 6
CACHE = "Solve once, read it back"
SKRF = "S to scikit-rf"
KEY = re.compile(r"\b(?:[0-9a-f]{64}|[0-9a-f]{12})\b")


def _run(command: str) -> None:
    """Run the page's shell command with this interpreter as ``python``."""
    argv = shlex.split(command.strip())
    assert argv[:3] == ["python", "-m", "gfem"], argv
    # the gfem this test imports, whatever the working directory
    source = str(Path(gfem.__file__).parents[1])
    path = os.pathsep.join([source, os.environ.get("PYTHONPATH", "")])
    done = subprocess.run(
        [sys.executable, *argv[1:]],
        capture_output=True,
        text=True,
        check=False,
        env=os.environ | {"PYTHONPATH": path},
    )
    assert done.returncode == 0, done.stderr


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("save_and_export")
    # the scikit-rf section needs the extra; the rest of the page does not
    skip = () if importlib.util.find_spec("skrf") else (SKRF,)
    log = logging.getLogger("gfem.solve_cache")
    handlers, level = list(log.handlers), log.level
    try:
        yield workdir, dp.run_page(PAGE, workdir, shell=_run, skip=skip)
    finally:
        # the page turns the logger on for the reader's session, not the tests'
        log.handlers[:] = handlers
        log.setLevel(level)


@pytest.fixture(scope="module")
def ran(page):
    return page[1]


@pytest.fixture(scope="module")
def workdir(page):
    return page[0]


@pytest.fixture(scope="module")
def s(workdir):
    """S of the result bundle: the same job solved again by the command line."""
    return gfem.load_result(workdir / "result").s


def _printed(ran, section: str) -> list[np.ndarray]:
    found = [
        dp.numbers(example.printed) for example in ran if example.section == section
    ]
    assert found, f"no Python block under {section!r}"
    return found


def _touchstone(path) -> tuple[float, np.ndarray]:
    """The reference impedance and S of a two-port Touchstone 1.1 file."""
    lines = path.read_text().splitlines()
    (option,) = [line for line in lines if line.startswith("#")]
    rows = np.loadtxt(path, comments=("!", "#"))
    pairs = rows[:, 1::2] + 1j * rows[:, 2::2]
    # one line per frequency: N11 N21 N12 N22
    return float(option.split()[-1]), pairs.reshape(-1, 2, 2).transpose(0, 2, 1)


def test_the_page_downloads_the_tutorial_mesh():
    first = dp.examples(PAGE.read_text(encoding="utf-8"))[0]
    assert f'url = "{dp.MESH_URL}"' in first.code


def test_every_output_block_is_what_the_code_prints(ran):
    keyed = [example for example in ran if example.section != CACHE]
    assert not dp.output_problems(keyed), "\n".join(dp.output_problems(keyed))


def _cache_blocks(ran) -> list[dp.Example]:
    found = [example for example in ran if example.section == CACHE]
    assert found, f"no Python block under {CACHE!r}"
    return found


def test_every_section_with_a_number_prints_it(ran):
    printing = {example.section for example in ran if example.printed.strip()}
    assert printing - {SKRF} == {
        "S to a Touchstone file",
        "S to xarray",
        "Fields to ParaView or PyVista",
        "A run as a file",
        CACHE,
    }


def test_the_touchstone_files_hold_s_and_its_50_ohm_form(ran, workdir, s):
    z0, values = _touchstone(workdir / "line.s2p")
    assert z0 == pytest.approx(Z_LINE, rel=1e-11)
    assert values == pytest.approx(np.asarray(s.values), abs=1e-6)
    z50, values50 = _touchstone(workdir / "line50.s2p")
    assert z50 == 50.0
    renormalized = np.asarray(s.renormalize(50.0).values)
    assert values50 == pytest.approx(renormalized, abs=1e-6)
    # Renormalizing back gives the file's own S: the two files are one network.
    back = gfem.NetworkParameters(
        values50, np.full((7, 2), 50.0), s.frequency, s.ports, s.ports
    ).renormalize(Z_LINE)
    assert np.asarray(back.values) == pytest.approx(values, abs=1e-12)
    assert "True" in ran[1].printed


def test_the_scikit_rf_network_is_s_on_the_ports_own_references(ran, s):
    pytest.importorskip("skrf")
    (example,) = [e for e in ran if e.section == SKRF]
    assert example.printed.split() == ["pseudo", "62.7883", "True"]
    nw = s.to_skrf(name="line")
    assert nw.s_def == "pseudo"
    assert nw.z0[0, 0].real == pytest.approx(Z_LINE, abs=5e-5)
    assert nw.s == pytest.approx(np.asarray(s.values), abs=1e-15)


def test_the_xarray_dataset_and_its_netcdf_file(ran, workdir, s):
    (printed,) = _printed(ran, "S to xarray")
    assert printed[-1] == pytest.approx(abs(complex(s["p2", "p1"][3])), abs=1e-6)
    xarray = pytest.importorskip("xarray")
    with xarray.open_dataset(workdir / "line_s.nc") as saved:
        values = saved.s_real.values + 1j * saved.s_imag.values
        assert values == pytest.approx(np.asarray(s.values), abs=1e-6)
        assert list(saved.port.values) == ["p1", "p2"]


def test_the_vtu_file_spans_the_line_in_metres(workdir):
    grid = meshio.read(workdir / "line_e.vtu")
    assert grid.points.min(axis=0) == pytest.approx([0, 0, 0], abs=1e-12)
    assert grid.points.max(axis=0) == pytest.approx([0.04, 0.006, 0.001])
    assert set(grid.point_data) == {"E_real", "E_imag"}
    # The page: "8 times more" cells than the 2880 tetrahedra of the mesh.
    assert sum(len(block.data) for block in grid.cells) == 8 * 2880


def test_the_field_beyond_the_ceramic_is_the_transmitted_wave(ran, s):
    _, (ez, by_hand, c_b) = _printed(ran, "Fields to ParaView or PyVista")
    transmitted = abs(complex(s["p2", "p1"][3])) * np.sqrt(2 * Z_LINE) / 1e-3
    assert by_hand == pytest.approx(transmitted, rel=1e-5)
    assert ez == pytest.approx(by_hand, rel=1e-5)
    # The page: "to about 1e-4".
    assert c_b == pytest.approx(ez, rel=3e-4)


def test_to_pyvista_gives_the_same_grid(workdir):
    pytest.importorskip("pyvista")
    field = gfem.load_result(workdir / "result").e_field(5e9, "p1")
    bounds = field.to_pyvista().bounds
    assert bounds == pytest.approx([0, 0.04, 0, 0.006, 0, 0.001], abs=1e-12)


def test_the_job_refers_to_the_mesh_by_relative_path_and_sha256(ran, workdir):
    doc = json.loads((workdir / "job.json").read_text())
    mesh = doc["job"]["model"]["mesh"]
    assert mesh["path"] == "line.msh"
    digest = hashlib.sha256((workdir / "line.msh").read_bytes()).hexdigest()
    assert mesh["sha256"] == digest
    assert doc["arrays"] is None  # no arrays, so no job.npz
    assert not (workdir / "job.npz").exists()
    (job_block,) = [e for e in ran if "job.to_json" in e.code]
    assert digest in job_block.shown


def test_a_job_written_away_from_its_mesh_holds_the_absolute_path(workdir):
    job = gfem.Job.from_json(workdir / "job.json")
    elsewhere = job.to_json(workdir / "elsewhere" / "job.json")
    mesh = json.loads(elsewhere.read_text())["job"]["model"]["mesh"]
    assert mesh["path"] == (workdir / "line.msh").resolve().as_posix()


def test_the_result_bundle_reads_back_as_the_solve(ran, workdir):
    (printed,) = _printed(ran, "A run as a file")[1:]
    assert printed[0] < 1e-10  # the same S, from a second solve of the same job
    _, (ez, _, _) = _printed(ran, "Fields to ParaView or PyVista")
    assert printed[1] == pytest.approx(ez, rel=1e-9)  # the kept field came back
    assert sorted(p.name for p in (workdir / "result").iterdir()) == [
        "result.json",
        "result.npz",
    ]
    doc = json.loads((workdir / "result" / "result.json").read_text())
    (field,) = {json.dumps(f["mesh"]) for f in doc["result"]["e_fields"]}
    mesh = json.loads(field)
    # The page: "here by its absolute path, since the mesh is outside result/".
    assert mesh["path"] == (workdir / "line.msh").resolve().as_posix()


def test_the_cache_blocks_print_what_the_page_shows_but_the_keys(ran):
    for example in _cache_blocks(ran):
        assert example.shown is not None, example.code
        printed, shown = (
            KEY.sub("<key>", text) for text in (example.printed, example.shown)
        )
        # the page prints a path; Windows writes its separator as a backslash
        assert printed.replace("\\", "/").split() == shown.split()


def test_the_second_solve_is_read_back_from_the_cache(ran, workdir):
    first, *_ = _cache_blocks(ran)
    miss, hit, difference = first.printed.splitlines()
    # The page: the first call solves and stores, the second reads it back.
    key = miss.split()[1].rstrip(":")
    assert miss.startswith(f"miss {key}: solved and stored (solves")
    assert hit.startswith(f"hit {key} (solves")
    (entry,) = [p for p in (workdir / "solves").iterdir() if KEY.fullmatch(p.name)]
    assert entry.name.startswith(key)
    assert float(difference) == 0.0  # read back bit for bit


def test_the_command_line_reads_the_same_entry(ran, workdir):
    *_, last = _cache_blocks(ran)
    run = json.loads((workdir / "cached" / "result.json").read_text())["run"]
    assert run["cache"]["status"] == "hit"
    (entry,) = [p for p in (workdir / "solves").iterdir() if KEY.fullmatch(p.name)]
    assert run["cache"]["key"] == entry.name
    assert run["cache"]["solved"]["ranks"] == 1
    assert "hit" in last.printed
