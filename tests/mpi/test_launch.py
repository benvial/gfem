"""Launching: refusals raise on every rank, one rank's failure ends the job.

No distributed solve exists yet, so every launched solve above one rank raises.
Each mesh is written per rank until `gfem.Mesh.from_gmsh` takes a communicator.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

import gradient_cases as cases
import mpi_cases
import numpy as np
import pytest
from ranks import (
    RanksDiffer,
    RanksFailed,
    RanksHung,
    on_ranks,
    on_ranks_argv,
    tagged,
)

import gfem
from gfem import _mpi
from gfem.__main__ import main

TESTS = Path(__file__).resolve().parents[1]
#: Importable by a launched script, as `rank_main` makes them for a function.
PYTHON_PATH = {"PYTHONPATH": f"{TESTS}{os.pathsep}{TESTS.parent / 'tools'}"}
RANK_DIR = """Path(sys.argv[1]) / f"rank-{MPI.COMM_WORLD.rank}" """


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _own_dir(directory) -> Path:
    path = Path(directory) / f"rank-{_world().rank}"
    path.mkdir(exist_ok=True)
    return path


# --- one rank's failure ends the job ---------------------------------------------


def _fail_in_a_section(failing):
    world = _world()
    with _mpi.section(world):
        if world.rank == failing:
            raise ValueError("planted local failure")
        world.Barrier()


@pytest.mark.parametrize("failing", [0, 1])
def test_a_failure_on_one_rank_aborts_the_job(failing):
    with pytest.raises(RanksFailed) as caught:
        on_ranks(_fail_in_a_section, failing, ranks=4, limit=30)
    # the section ended the job: nothing reached the worker's own handler
    assert caught.value.tracebacks == {}
    assert caught.value.returncode != 0
    assert f"gfem: rank {failing}:" in caught.value.log
    assert "planted local failure" in caught.value.log


@pytest.mark.parametrize("failing", [0, 1])
def test_a_failure_on_one_rank_aborts_under_srun(failing, monkeypatch):
    if shutil.which("srun") is None or "SLURM_JOB_ID" not in os.environ:
        pytest.skip("needs srun inside a Slurm job")
    monkeypatch.setenv(
        "GFEM_TEST_MPI_LAUNCH", "srun --mpi=pmix --cpus-per-task=1 --exact"
    )
    with pytest.raises(RanksFailed) as caught:
        on_ranks(_fail_in_a_section, failing, ranks=4, limit=30)
    assert caught.value.tracebacks == {}
    assert f"gfem: rank {failing}:" in caught.value.log


def _raise_agreed():
    raise gfem.DistributedUnsupportedError("agreed", "on every rank")


def _agreed_error_in_a_section():
    world = _world()
    try:
        with _mpi.section(world):
            _raise_agreed()
    except gfem.DistributedUnsupportedError as err:
        return err.feature, world.allreduce(1)


def test_an_agreed_error_raises_on_every_rank_and_a_collective_runs():
    assert on_ranks(_agreed_error_in_a_section, ranks=3, limit=60)[0] == ("agreed", 3)


HOOK_SCRIPT = """
import sys
from pathlib import Path

from mpi4py import MPI

import gfem
import mpi_cases

world = MPI.COMM_WORLD
directory = {rank_dir}
directory.mkdir(exist_ok=True)
model, problem, options = mpi_cases.refused("eigenmode", directory)
try:
    gfem.solve(model, problem, options=options)
except gfem.DistributedUnsupportedError:
    pass
if world.rank == int(sys.argv[2]):
    raise ValueError("planted in user code")
world.Barrier()
"""


@pytest.mark.parametrize("failing", [0, 1])
def test_a_solve_chains_a_hook_that_aborts_on_user_code(failing, tmp_path):
    script = tmp_path / "script.py"
    script.write_text(HOOK_SCRIPT.format(rank_dir=RANK_DIR))
    code, log = on_ranks_argv(
        [sys.executable, str(script), str(tmp_path), str(failing)],
        ranks=4,
        limit=30,
        env=PYTHON_PATH,
    )
    assert code != 0, log
    assert f"gfem: rank {failing}: an unhandled exception ends the job" in log
    assert "planted in user code" in log


def test_before_any_solve_python_m_mpi4py_is_the_recipe(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(
        textwrap.dedent("""
        from mpi4py import MPI

        if MPI.COMM_WORLD.rank == 1:
            raise ValueError("planted before any solve")
        MPI.COMM_WORLD.Barrier()
        """)
    )
    code, log = on_ranks_argv(
        [sys.executable, "-m", "mpi4py", str(script)], ranks=3, limit=30
    )
    assert code != 0, log
    assert "planted before any solve" in log


UNCAUGHT_SCRIPT = """
import sys
from pathlib import Path

from mpi4py import MPI

import gfem
import mpi_cases

directory = {rank_dir}
directory.mkdir(exist_ok=True)
model, problem, options = mpi_cases.refused("driven", directory)
gfem.solve(model, problem, options=options)
"""


def test_an_uncaught_refusal_ends_the_job(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(UNCAUGHT_SCRIPT.format(rank_dir=RANK_DIR))
    code, log = on_ranks_argv(
        [sys.executable, str(script), str(tmp_path)],
        ranks=3,
        limit=30,
        env=PYTHON_PATH,
    )
    assert code != 0, log
    assert "DistributedUnsupportedError" in log


# --- refusals --------------------------------------------------------------------


def _refused_then_a_collective(key, directory):
    world = _world()
    model, problem, options = mpi_cases.refused(key, _own_dir(directory))
    try:
        gfem.solve(model, problem, options=options)
    except gfem.DistributedUnsupportedError as err:
        return err.feature, world.allreduce(world.rank)
    raise AssertionError(f"{key} ran")


@pytest.mark.parametrize("key", mpi_cases.KEYS)
def test_every_rank_raises_a_refusal_and_a_later_collective_runs(key, tmp_path):
    values = on_ranks(_refused_then_a_collective, key, tmp_path, ranks=3, limit=120)
    assert values[0] == (mpi_cases.feature(key), 0 + 1 + 2)


def test_a_launched_world_of_one_raises_naming_pmix():
    code = """
    from mpi4py import MPI

    from gfem import _mpi

    try:
        _mpi.resolve_comm()
    except gfem.SolverUnavailableError as err:
        print(err)
    """
    environment = {
        k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMI"))
    }
    environment["SLURM_STEP_NUM_TASKS"] = "4"
    proc = subprocess.run(
        [sys.executable, "-c", "import gfem\n" + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--mpi=pmix" in proc.stdout
    assert "COMM_SELF" in proc.stdout


SRUN_WITHOUT_MPI = """
import gfem
from gfem import _mpi

try:
    comm = _mpi.resolve_comm()
except gfem.SolverUnavailableError as err:
    print("refused", err)
else:
    print("resolved", comm.allreduce(1))
"""


def test_srun_without_mpi_never_runs_silent_copies():
    """Open MPI 5.0.10 gives each task a world of 1, and 4.1.6 a real world of 2."""
    srun = shutil.which("srun")
    if srun is None or "SLURM_JOB_ID" not in os.environ:
        pytest.skip("needs srun inside a Slurm job")
    proc = subprocess.run(
        [
            srun,
            "--mpi=none",
            "-n",
            "2",
            "--cpus-per-task=1",
            "--exact",
            sys.executable,
            "-c",
            SRUN_WITHOUT_MPI,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    refused = proc.stdout.count("--mpi=pmix")
    resolved = proc.stdout.count("resolved 2")
    assert (refused, resolved) in {(2, 0), (0, 2)}, proc.stdout + proc.stderr


# --- rank copies -----------------------------------------------------------------


def _copy(directory, frequency, workers=1):
    from mpi4py import MPI

    line = cases.layered_line(_own_dir(directory))
    options = gfem.SolverOptions(
        linear_solver="direct", threads=1, frequency_workers=workers
    )
    result = gfem.solve(
        line, gfem.Driven(frequency), options=options, comm=MPI.COMM_SELF
    )
    return np.asarray(result.s.values)


def _one_process(directory, frequency, workers=1):
    line = cases.layered_line(directory)
    options = gfem.SolverOptions(
        linear_solver="direct", threads=1, frequency_workers=workers
    )
    return np.asarray(
        gfem.solve(line, gfem.Driven(frequency), options=options).s.values
    )


def _assert_close(got, expected, rtol=1e-11):
    assert np.abs(got - expected).max() <= rtol * np.abs(expected).max()


def test_comm_self_runs_copies(tmp_path):
    expected = _one_process(tmp_path, 5e9)
    values = on_ranks(_copy, tmp_path, 5e9, ranks=3, limit=180)
    _assert_close(values[0], expected)


def _copy_of_my_frequency(directory, frequencies):
    return _copy(directory, frequencies[_world().rank])


def test_comm_self_copies_may_differ_across_ranks(tmp_path):
    frequencies = [2e9, 5e9, 7e9]
    values = on_ranks(
        _copy_of_my_frequency,
        tmp_path,
        frequencies,
        ranks=3,
        limit=180,
        replicated=False,
    )
    for frequency, value in zip(frequencies, values, strict=True):
        _assert_close(value, _one_process(tmp_path, frequency))


def _electrostatic_copy(directory):
    from mpi4py import MPI

    result = gfem.solve(
        mpi_cases.stack(_own_dir(directory)),
        gfem.Electrostatic(terminals=mpi_cases.TERMINALS),
        options=gfem.SolverOptions(order=1, threads=1),
        comm=MPI.COMM_SELF,
    )
    return np.asarray(result.capacitance)


def test_comm_self_runs_what_ranks_refuse(tmp_path):
    expected = gfem.solve(
        mpi_cases.stack(tmp_path),
        gfem.Electrostatic(terminals=mpi_cases.TERMINALS),
        options=gfem.SolverOptions(order=1, threads=1),
    ).capacitance
    values = on_ranks(_electrostatic_copy, tmp_path, ranks=3, limit=180)
    _assert_close(values[0], np.asarray(expected))


def test_comm_self_copies_run_frequency_workers(tmp_path):
    frequencies = [2e9, 5e9, 7e9, 8e9]
    expected = _one_process(tmp_path, frequencies, workers=2)
    values = on_ranks(_copy, tmp_path, frequencies, workers=2, ranks=2, limit=240)
    _assert_close(values[0], expected)


# --- the helper ------------------------------------------------------------------


def _a_rank_differs_in_one_bit():
    bits = np.array([1.0, 2.0]) + np.array([0.0, np.spacing(2.0)]) * _world().rank
    return {"value": bits}


def test_the_bitwise_check_fails_a_planted_difference():
    with pytest.raises(RanksDiffer, match=r"\['value'\] differs"):
        on_ranks(_a_rank_differs_in_one_bit, ranks=3, limit=60)


def test_a_rank_s_own_piece_may_differ():
    values = on_ranks(_a_rank_differs_in_one_bit, ranks=3, limit=60, replicated=False)
    assert values[0]["value"][1] != values[2]["value"][1]


def _one_rank_asserts():
    assert _world().rank != 1, "planted assertion"
    _world().Barrier()


def test_an_assertion_on_one_rank_reports_its_traceback():
    with pytest.raises(RanksFailed) as caught:
        on_ranks(_one_rank_asserts, ranks=3, limit=30)
    assert "planted assertion" in caught.value.tracebacks[1]


def _one_rank_sleeps():
    if _world().rank == 0:
        time.sleep(3600)
    _world().Barrier()


def test_a_hang_passes_the_limit_and_leaves_nothing():
    with pytest.raises(RanksHung) as caught:
        on_ranks(_one_rank_sleeps, ranks=3, limit=10)
    deadline = time.monotonic() + 10
    while tagged(caught.value.tag) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert tagged(caught.value.tag) == []


# --- agreement on real ranks -----------------------------------------------------


def _agreements():
    world = _world()
    rank = world.rank
    return {
        "any_one": _mpi.agree_any(world, flag=rank == 1),
        "any_none": _mpi.agree_any(world, flag=False),
        "all_but_one": _mpi.agree_all(world, flag=rank != 1),
        "all": _mpi.agree_all(world, flag=True),
        "sum": _mpi.reduce_bcast(world, (rank + 1) * np.array([1.0 + 1.0j, 2.0])),
        "scalar": _mpi.reduce_bcast(world, float(rank + 1)),
        "empty": _mpi.reduce_bcast(world, np.zeros((0, 2))),
        "held": _mpi.node_bytes(world, rank + 1),
        "budget": _mpi.node_budget(world, 10 - rank),
    }


def test_agreement_gives_every_rank_one_answer():
    value = on_ranks(_agreements, ranks=3, limit=60)[0]
    assert value["any_one"] is True
    assert value["any_none"] is False
    assert value["all_but_one"] is False
    assert value["all"] is True
    np.testing.assert_array_equal(value["sum"], [6.0 + 6.0j, 12.0])
    assert value["scalar"] == 6.0
    assert value["empty"].shape == (0, 2)
    assert value["held"] == 6  # one node here: all ranks' bytes
    assert value["budget"] == 8


def _warned_under_a_filter():
    world = _world()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            _mpi.warn(world, "planted", UserWarning)
        except UserWarning:
            return "raised", world.allreduce(1)
    return "passed", world.allreduce(1)


def test_an_error_filter_raises_a_warning_on_every_rank():
    assert on_ranks(_warned_under_a_filter, ranks=3, limit=60)[0] == ("raised", 3)


def _warnings_seen():
    world = _world()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        _mpi.warn(world, "planted", UserWarning)
    return len(seen)


def test_only_rank_0_shows_a_warning():
    seen = on_ranks(_warnings_seen, ranks=3, limit=60, replicated=False)
    assert seen == [1, 0, 0]


# --- the command line ------------------------------------------------------------


def _job_file(tmp_path) -> Path:
    model, problem, options = mpi_cases.refused("driven", tmp_path)
    return gfem.Job(model, problem, options).to_json(tmp_path / "job.json")


def test_the_cli_exits_1_on_every_rank_with_one_report(tmp_path):
    job = _job_file(tmp_path)
    code, log = on_ranks_argv(
        [sys.executable, "-m", "gfem", "run", str(job), "--out", str(tmp_path / "out")],
        ranks=3,
        limit=120,
    )
    assert code == 1, log
    assert log.count("gfem: error: DistributedUnsupportedError") == 1, log


def _cli_with_a_job_missing_on_rank_1(job):
    path = f"{job}.missing" if _world().rank == 1 else job
    try:
        return main(["run", path, "--out", f"{job}.out"])
    except SystemExit as stop:
        return stop.code


def test_a_job_missing_on_one_rank_exits_2_everywhere(tmp_path):
    job = str(_job_file(tmp_path))
    assert on_ranks(_cli_with_a_job_missing_on_rank_1, job, ranks=3, limit=120)[0] == 2


def _cli_with_a_job_broken_on_rank_1(job, broken):
    path = broken if _world().rank == 1 else job
    return main(["run", path, "--out", f"{job}.out"])


def test_a_job_broken_on_one_rank_ends_the_job(tmp_path):
    job = _job_file(tmp_path)
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"format": "not a job"}))
    with pytest.raises(RanksFailed) as caught:
        on_ranks(
            _cli_with_a_job_broken_on_rank_1,
            str(job),
            str(broken),
            ranks=3,
            limit=120,
        )
    assert "gfem: rank 1:" in caught.value.log
