"""`tools/scale_run.py`: a record for every run, however it ended, and its table."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import scale_run

TOOL = Path(scale_run.__file__)
LINE = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "line.msh"


def _case(directory: Path) -> Path:
    """A gsim-shaped case on the tutorial's line: two lumped ports, two shells."""
    directory.mkdir(parents=True)
    shutil.copy(LINE, directory / "palace.msh")
    config = {
        "Model": {"L0": 1e-6, "Mesh": "palace.msh"},
        "Domains": {"Materials": [{"Attributes": [1, 2], "Permittivity": 2.0}]},
        "Boundaries": {
            "Conductivity": [
                {"Attributes": [5, 6], "Conductivity": 3e7, "Thickness": 1.0}
            ],
            "Absorbing": {"Order": 1, "Attributes": [4]},
            "LumpedPort": [
                {
                    "Index": i + 1,
                    "R": 50.0,
                    "Elements": [{"Attributes": [tag], "Direction": "+z"}],
                }
                for i, tag in enumerate((3, 7))
            ],
        },
    }
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def _tool(*args: object, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **environment},
    )


@pytest.fixture(scope="module")
def solved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real run through the script, whose name holds a dot."""
    root = tmp_path_factory.mktemp("scale")
    done = _tool(
        "run",
        _case(root / "case"),
        root / "records" / "x_r0.5_void",
        "--solver",
        "direct",
        "--threads",
        "1",
        "--repeat",
        "2",
        SCALE_QOS="gp_test",
        SCALE_CONSTRAINT="highmem",
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return root / "records" / "x_r0.5_void"


def test_a_record_with_a_dot_in_its_name_keeps_its_full_name(solved: Path) -> None:
    assert sorted(p.name for p in solved.parent.iterdir()) == [
        "x_r0.5_void.json",
        "x_r0.5_void.npz",
    ]
    assert json.loads(Path(f"{solved}.json").read_text())["name"] == "x_r0.5_void"


def test_a_record_says_which_repeat_on_which_nodes(solved: Path) -> None:
    record = json.loads(Path(f"{solved}.json").read_text())
    assert record["repeat"] == 2
    assert record["qos"] == "gp_test"
    assert record["constraint"] == "highmem"
    if Path("/proc/meminfo").exists():  # Linux; elsewhere the field reads 0
        assert all(kb > 0 for kb in record["mem_total_kb"])


#: A killed `direct` step's log: the trace on rank 0 and MUMPS's own lines, as printed
#: by MUMPS 5.9.1 in job 48478202, which stop after the analysis.
KILLED_LOG = """\
      0.1 s  896 ranks on 32 nodes
mesh_read starts at 0.4 GiB peak
mesh_read ends at 31.0 GiB peak
solve starts at 31.0 GiB peak
forward starts at 31.0 GiB peak
mumps_setup starts at 41.5 GiB peak
Entering ZMUMPS 5.9.1 from C interface with JOB, N =   1    22356889
Leaving analysis phase with  ...
 INFOG(1)                                       =               0
 -- (20) Number of entries in factors (estim.)  =      7316285143
 --  (5) Maximum frontal size      (estimated)  =           19642
 RINFOG(1) Operations during elimination (estim)= 2.592D+13
    Maximum estim. space in Mbytes, IC facto.    (INFOG(16)):           13254
    Total space in MBytes, IC factorization      (INFOG(17)):          331670
Entering ZMUMPS 5.9.1 from C interface with JOB, N =   2    22356889
slurmstepd: error: Detected 1 oom_kill event in StepId=1.0.
"""


@pytest.fixture
def killed(tmp_path: Path) -> Path:
    """A stub, as the job script closes a step that died in MUMPS's factorization."""
    log = tmp_path / "killed.log"
    log.write_text(KILLED_LOG)
    name = tmp_path / "records" / "d_r0.5_void_32"
    name.parent.mkdir()
    closed = _tool(
        "close",
        name,
        "--log",
        log,
        "--state",
        "OUT_OF_MEMORY",
        "--rc",
        "137",
        "--case",
        "/somewhere/B2_void_split",
        "--solver",
        "direct",
        "--ghz",
        "50",
        "--ranks-per-node",
        "28",
        "--threads-per-rank",
        "4",
        "--repeat",
        "1",
        SLURM_JOB_ID="123",
        SLURM_JOB_NUM_NODES="32",
        SLURM_JOB_NODELIST="gs01r1b[01-32]",
        SCALE_COMMIT="abc1234",
        SCALE_QOS="gp_test",
        SCALE_CONSTRAINT="highmem",
    )
    assert closed.returncode == 0, closed.stdout + closed.stderr
    return name


def test_a_killed_run_leaves_a_stub_with_what_its_log_knows(killed: Path) -> None:
    stub = json.loads(Path(f"{killed}.json").read_text())
    assert stub["failed"] is True
    assert stub["state"] == "OUT_OF_MEMORY"
    assert stub["last_phase"] == "mumps_setup"
    assert stub["open_phases"] == ["solve", "forward", "mumps_setup"]
    assert stub["job"] == "123"
    assert stub["node_count"] == 32
    assert stub["ranks"] == 896
    assert stub["nodelist"] == "gs01r1b[01-32]"
    assert stub["commit"] == "abc1234"
    assert stub["mumps_printed"] == [
        {
            "n": 22356889,
            "entries_estimated": 7316285143,
            "front_estimated": 19642,
            "flops_estimated": 2.592e13,
            "max_ic_mb": 13254,
            "total_ic_mb": 331670,
        }
    ]


def test_check_reports_a_killed_run_and_does_not_raise(
    killed: Path, solved: Path
) -> None:
    done = _tool("check", killed, solved, "--gate", "direct")
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    result = json.loads(done.stdout)
    assert result["passed"] is False
    assert "OUT_OF_MEMORY" in result["reason"]
    checks = json.loads(Path(f"{killed}.json").read_text())["checks"]
    assert checks[0]["passed"] is False


def test_check_reports_a_reference_that_is_missing(
    solved: Path, tmp_path: Path
) -> None:
    copy = tmp_path / solved.name
    shutil.copy(f"{solved}.json", f"{copy}.json")
    shutil.copy(f"{solved}.npz", f"{copy}.npz")
    done = _tool("check", copy, tmp_path / "nowhere", "--observe")
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert "nowhere" in json.loads(done.stdout)["reason"]


def test_closing_a_finished_run_keeps_its_record(solved: Path, tmp_path: Path) -> None:
    copy = tmp_path / solved.name
    shutil.copy(f"{solved}.json", f"{copy}.json")
    log = tmp_path / "done.log"
    log.write_text(KILLED_LOG)
    closed = _tool("close", copy, "--log", log, "--state", "COMPLETED", "--rc", "0")
    assert closed.returncode == 0, closed.stderr
    record = json.loads(Path(f"{copy}.json").read_text())
    assert "failed" not in record
    assert record["s"]  # the solve's own record stands
    assert record["mumps_printed"][0]["total_ic_mb"] == 331670


def test_a_second_close_keeps_what_the_first_knew(killed: Path) -> None:
    closed = _tool("close", killed, "--state", "NODE_FAIL")
    assert closed.returncode == 0, closed.stderr
    stub = json.loads(Path(f"{killed}.json").read_text())
    assert stub["state"] == "NODE_FAIL"
    assert stub["last_phase"] == "mumps_setup"
    assert stub["ranks"] == 896


def _finished(directory: Path, name: str, nodes: int, seconds: float, **more) -> Path:
    """A record as `run` writes it, with only what the table reads."""
    record = {
        "name": name,
        "case": more.pop("case", "/m/B2_void_split"),
        "solver": more.pop("solver", "bddc_gmres"),
        "node_count": nodes,
        "ranks": 28 * nodes,
        "threads_read": 4,
        "frequencies_ghz": [50.0],
        "dofs": {"owned_sum": 108_570_046},
        "phases": {"solve": {"max_s": seconds}},
        "iterations": {"iterations": [[70, 71]]},
        "memory": {"peak_rss_largest_node_gib": 120.0},
        "job": str(1000 + nodes),
        **more,
    }
    path = directory / f"{name}.json"
    path.write_text(json.dumps(record))
    return path


def _rows(text: str) -> list[list[str]]:
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    return [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]


def _column(text: str, title: str) -> int:
    head = next(ln for ln in text.splitlines() if ln.startswith("|"))
    return [c.strip() for c in head.strip("|").split("|")].index(title)


def test_the_table_holds_a_rung_median_range_and_strong_efficiency(
    tmp_path: Path,
) -> None:
    paths = [
        _finished(tmp_path, "a1", 4, 100.0),
        _finished(tmp_path, "a2", 4, 120.0),
        _finished(tmp_path, "a3", 4, 110.0),
        _finished(tmp_path, "b1", 8, 80.0),
    ]
    done = _tool("table", *paths, "--base", "4")
    assert done.returncode == 0, done.stderr
    rows = _rows(done.stdout)
    assert len(rows) == 2
    four, eight = rows
    assert four[_column(done.stdout, "Repeats")] == "3"
    assert four[_column(done.stdout, "`gfem.solve` median, s")] == "110"
    assert four[_column(done.stdout, "Range, s")] == "100-120"
    assert four[_column(done.stdout, "DOF per rank")] == "969 k"
    assert four[_column(done.stdout, "DOF per node")] == "27.14 M"
    assert four[_column(done.stdout, "Efficiency")] == "1.00"
    # 110 s on 4 nodes against 80 s on 8: 110 * 4 / (80 * 8)
    assert eight[_column(done.stdout, "Efficiency")] == "0.69"
    assert "Efficiency against 4 nodes (strong)" in done.stdout


def test_weak_efficiency_reads_across_meshes(tmp_path: Path) -> None:
    paths = [
        _finished(tmp_path, "a", 1, 100.0, case="/m/r0.5_void"),
        _finished(tmp_path, "b", 2, 125.0, case="/m/B1_void"),
    ]
    done = _tool("table", *paths, "--base", "1", "--weak")
    two = _rows(done.stdout)[1]
    assert two[_column(done.stdout, "Efficiency")] == "0.80"
    assert "Efficiency against 1 nodes (weak)" in done.stdout


def test_the_table_shows_a_failed_run_as_a_row_and_names_the_solve_phase(
    killed: Path, tmp_path: Path
) -> None:
    done = _tool(
        "table", _finished(tmp_path, "a", 4, 100.0), f"{killed}.json", "--base", "4"
    )
    assert done.returncode == 0, done.stderr
    failed = _rows(done.stdout)[1]
    time_cell = failed[_column(done.stdout, "`gfem.solve` median, s")]
    assert time_cell == "OUT_OF_MEMORY in mumps_setup"
    assert failed[_column(done.stdout, "Nodes")] == "32"
    assert failed[_column(done.stdout, "Ranks x threads")] == "896 x 4"
    assert failed[_column(done.stdout, "Job")] == "123"
    assert "true residual" in done.stdout


def test_a_failed_repeat_beside_finished_ones_is_counted(
    killed: Path, tmp_path: Path
) -> None:
    done = _tool(
        "table",
        _finished(tmp_path, "a", 32, 100.0, solver="direct"),
        f"{killed}.json",
    )
    rows = _rows(done.stdout)
    assert len(rows) == 1
    assert rows[0][_column(done.stdout, "Repeats")] == "1 + 1 OUT_OF_MEMORY"
    assert rows[0][_column(done.stdout, "`gfem.solve` median, s")] == "100"


def test_a_stub_closed_by_hand_with_only_a_state_reaches_the_table(
    tmp_path: Path,
) -> None:
    name = tmp_path / "lost"
    assert _tool("close", name, "--state", "NODE_FAIL").returncode == 0
    done = _tool("table", f"{name}.json")
    assert done.returncode == 0, done.stderr
    assert "NODE_FAIL in no phase" in done.stdout


def _close(name: Path, state: str, *more: object) -> None:
    closed = _tool("close", name, "--state", state, *more)
    assert closed.returncode == 0, closed.stdout + closed.stderr


def _copy(solved: Path, directory: Path) -> Path:
    copy = directory / solved.name
    shutil.copy(f"{solved}.json", f"{copy}.json")
    shutil.copy(f"{solved}.npz", f"{copy}.npz")
    return copy


def test_a_rerun_moves_the_earlier_record_aside(solved: Path, tmp_path: Path) -> None:
    # a rerun of a finished step is killed before it writes: its row is the rerun's
    copy = _copy(solved, tmp_path)
    _close(copy, "STARTED")
    _close(copy, "CANCELLED")
    stub = json.loads(Path(f"{copy}.json").read_text())
    assert stub["failed"] is True
    assert stub["state"] == "CANCELLED"
    assert "s" not in stub
    assert not Path(f"{copy}.npz").exists()
    aside = tmp_path / f"{solved.name}.superseded-1"
    assert json.loads(Path(f"{aside}.json").read_text())["s"]
    assert Path(f"{aside}.npz").exists()
    # a second rerun keeps both earlier records
    _close(copy, "STARTED")
    assert Path(f"{tmp_path / solved.name}.superseded-2.json").exists()
    assert Path(f"{aside}.json").exists()


def test_the_table_and_check_ignore_superseded_records(
    solved: Path, tmp_path: Path
) -> None:
    copy = _copy(solved, tmp_path)
    _close(copy, "STARTED")
    aside = tmp_path / f"{solved.name}.superseded-1"
    done = _tool("table", *sorted(tmp_path.glob("*.json")))
    assert done.returncode == 0, done.stderr
    rows = _rows(done.stdout)
    assert len(rows) == 1
    median = rows[0][_column(done.stdout, "`gfem.solve` median, s")]
    assert median == "STARTED in no phase"
    checked = _tool("check", aside, solved, "--gate", "direct")
    assert checked.returncode == 1
    assert "superseded" in json.loads(checked.stdout)["reason"]
    assert "checks" not in json.loads(Path(f"{aside}.json").read_text())


def test_a_spaces_only_record_reaches_the_table(tmp_path: Path) -> None:
    path = _finished(tmp_path, "s", 1, 0.0, phases={"spaces_only": {"max_s": 5.0}})
    done = _tool("table", path)
    assert done.returncode == 0, done.stderr
    median = _rows(done.stdout)[0][_column(done.stdout, "`gfem.solve` median, s")]
    assert median == "nan"


def test_a_stub_of_a_step_with_no_launcher_records_one_rank(tmp_path: Path) -> None:
    # RPN 0: python with no launcher, whose own record says one rank
    name = tmp_path / "one"
    _close(name, "STARTED", "--ranks-per-node", "0", "--threads-per-rank", "8")
    assert json.loads(Path(f"{name}.json").read_text())["ranks"] == 1


def test_mumps_print_without_ranks_says_it_does_nothing(tmp_path: Path) -> None:
    done = _tool(
        "run",
        _case(tmp_path / "case"),
        tmp_path / "records" / "spaces",
        "--solver",
        "direct",
        "--threads",
        "1",
        "--spaces-only",
        "--mumps-print",
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.count("--mumps-print does nothing without ranks") == 1
