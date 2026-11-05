"""Solve threads: what a ``threads=None`` solve runs NGSolve on (ADR 0056).

``NGS_NUM_THREADS`` when it holds a positive integer, else the physical cores in the
process's affinity mask, not NGSolve's own default of every hardware thread of the
machine. Under ranks ADR 0077 decides instead (tests/unit/test_mpi.py).
"""

import os
import sys
import types
import warnings
from pathlib import Path

import ngsolve as ngs
import pytest
from isolated import isolated

from gfem import _mpi
from gfem._engine.runtime import solve_threads, task_manager


@pytest.fixture(autouse=True)
def _no_overrides(monkeypatch):
    monkeypatch.delenv("NGS_NUM_THREADS", raising=False)
    monkeypatch.delenv("PYTHON_CPU_COUNT", raising=False)


def _on_mask(monkeypatch, mask, per_core=2):
    """This process may run on ``mask``, on a machine of ``per_core`` threads a core."""
    mask = set(mask)
    read_text = Path.read_text

    def siblings(path, *args, **kwargs):
        if path.name == "thread_siblings_list":
            cpu = int(path.parent.parent.name.removeprefix("cpu"))
            return str(cpu // per_core)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", siblings)
    monkeypatch.setattr(os, "cpu_count", lambda: 224)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: mask, raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: len(mask), raising=False)


@pytest.mark.parametrize("python_has_process_cpu_count", [True, False])
def test_solve_threads_are_the_physical_cores_the_process_may_use(
    monkeypatch, python_has_process_cpu_count
):
    _on_mask(monkeypatch, range(12))  # both hardware threads of 6 cores
    if not python_has_process_cpu_count:  # Python 3.12
        monkeypatch.delattr(os, "process_cpu_count")
    assert solve_threads(None) == 6
    assert solve_threads(12) == 12  # an explicit count wins


def test_a_mask_of_one_hardware_thread_per_core_counts_each(monkeypatch):
    # srun --hint=nomultithread
    _on_mask(monkeypatch, [0, 2, 4, 6, 8])
    assert solve_threads(None) == 5


def test_a_mask_of_part_of_a_core_counts_that_core(monkeypatch):
    _on_mask(monkeypatch, [1])
    assert solve_threads(None) == 1


def test_a_kernel_that_does_not_name_siblings_counts_each_cpu(monkeypatch):
    # gVisor has no topology directory; a CPU that does not exist has none either
    mask = {123456, 123457, 123458}
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: mask, raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 3, raising=False)
    assert solve_threads(None) == 3


def test_a_system_with_no_affinity_mask_counts_what_python_counts(monkeypatch):
    # macOS, Windows: the standard library cannot tell a core from a hardware thread
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 6, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert solve_threads(None) == 6
    monkeypatch.delattr(os, "process_cpu_count")
    assert solve_threads(None) == 64


@pytest.mark.parametrize("how", ["environment", "-X"])
def test_pythons_own_cpu_count_override_counts_as_the_cpus_the_process_may_use(
    monkeypatch, how
):
    # PYTHON_CPU_COUNT, -X cpu_count (3.13): os.process_cpu_count answers with it,
    # even when it equals the mask's size
    _on_mask(monkeypatch, range(12))
    if how == "environment":
        monkeypatch.setenv("PYTHON_CPU_COUNT", "12")
    else:
        monkeypatch.setitem(sys._xoptions, "cpu_count", "12")
    assert solve_threads(None) == 12


def test_pythons_default_cpu_count_is_no_override(monkeypatch):
    _on_mask(monkeypatch, range(12))
    monkeypatch.setenv("PYTHON_CPU_COUNT", "default")
    assert solve_threads(None) == 6
    # a process_cpu_count that differs from the mask is not an override by itself
    monkeypatch.delenv("PYTHON_CPU_COUNT")
    monkeypatch.setattr(os, "process_cpu_count", lambda: 5)
    assert solve_threads(None) == 6


def test_ngs_num_threads_wins_over_the_cores_the_process_may_use(monkeypatch):
    _on_mask(monkeypatch, range(12))
    monkeypatch.setenv("NGS_NUM_THREADS", "3")
    assert solve_threads(None) == 3
    assert solve_threads(12) == 12  # an explicit count beats the variable too
    monkeypatch.setenv("NGS_NUM_THREADS", "300")
    assert solve_threads(None) == 300  # no cap


class _Node:
    """A node communicator of one rank."""

    def __init__(self, mask):
        self.mask = mask

    def allgather(self, value):
        return [value]

    def Free(self):  # noqa: N802
        pass


class _Ranks:
    """As much of a communicator as `default_threads` reads, one rank to a node."""

    rank, size = 0, 2

    def Split_type(self, _kind):  # noqa: N802
        return _Node(None)

    def allreduce(self, value, op=None):  # noqa: ARG002
        return value


def _default_threads(monkeypatch, under):
    """`threads=None` at one process, or under ranks (ADR 0077)."""
    if under == "one process":
        return solve_threads(None)
    mpi = types.SimpleNamespace(COMM_TYPE_SHARED=0, MIN="min")
    monkeypatch.setattr(_mpi, "_load_mpi", lambda: mpi)
    monkeypatch.setattr(_mpi, "comm_key", lambda _comm: 0)
    monkeypatch.setattr(_mpi, "_DEFAULT_THREADS", {})
    return _mpi.default_threads(_Ranks())


@pytest.mark.parametrize("under", ["one process", "ranks"])
@pytest.mark.parametrize("value", ["", " ", "0", "00", "-2", "abc", "2.5", "0x3"])
def test_a_malformed_ngs_num_threads_is_treated_as_unset(monkeypatch, value, under):
    _on_mask(monkeypatch, range(12))
    monkeypatch.setenv("NGS_NUM_THREADS", value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _default_threads(monkeypatch, under) == 6


@pytest.mark.parametrize("under", ["one process", "ranks"])
@pytest.mark.parametrize("value", ["3", " 3 ", "3\n", "03"])
def test_a_positive_ngs_num_threads_counts_after_whitespace(monkeypatch, value, under):
    _on_mask(monkeypatch, range(12))
    monkeypatch.setenv("NGS_NUM_THREADS", value)
    assert _default_threads(monkeypatch, under) == 3


@pytest.mark.usefixtures("ngsolve_threads")
def test_a_threads_none_call_runs_on_solve_threads_whatever_ran_before(monkeypatch):
    with task_manager(3):
        pass  # leaves the count at 3, as an earlier test does
    monkeypatch.setenv("NGS_NUM_THREADS", "2")
    with task_manager(None):
        assert ngs.GetNumThreads() == 2
    monkeypatch.delenv("NGS_NUM_THREADS")
    _on_mask(monkeypatch, range(8))
    with task_manager(None):
        assert ngs.GetNumThreads() == 4


@pytest.mark.usefixtures("ngsolve_threads")
def test_a_threads_none_call_inside_a_callers_task_manager_keeps_its_count(
    monkeypatch, capfd
):
    ngs.SetNumThreads(3)
    with ngs.TaskManager():
        monkeypatch.setenv("NGS_NUM_THREADS", "2")
        with task_manager(None):
            assert ngs.GetNumThreads() == 3
    captured = capfd.readouterr()
    assert "can't change number of threads" not in captured.out + captured.err


def _sibling_list(cpu):
    return Path(
        f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    ).read_text()


def _threads_on(cpus):
    os.sched_setaffinity(0, cpus)  # this thread starts the TaskManager's workers
    with task_manager(None):
        return ngs.GetNumThreads(), solve_threads(None)


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"), reason="sets the affinity mask (Linux)"
)
def test_a_threads_none_solve_runs_on_the_physical_cores_of_the_mask():
    # a fresh process without NGS_NUM_THREADS: NGSolve's own default would be every
    # hardware thread (224 on an MN5 node), whatever the mask
    cores: dict[str, list[int]] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        try:
            cores.setdefault(_sibling_list(cpu), []).append(cpu)
        except OSError:
            pytest.skip("the kernel does not name this CPU's siblings")
    two_cores = list(cores.values())[:2]
    cpus = [cpu for core in two_cores for cpu in core]  # every sibling in the mask
    if len(two_cores) < 2 or (os.cpu_count() or 1) <= 2:
        pytest.skip("needs a mask of 2 cores smaller than the machine")
    (threads, default), _ = isolated(_threads_on, cpus)
    assert threads == default == 2
