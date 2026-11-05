"""NGSolve threading for one engine call."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from typing import Any

import ngsolve as ngs

from gfem import _mpi
from gfem._linalg import blas

__all__ = ["available_cpus", "solve_threads", "task_manager"]


def available_cpus() -> int:
    """Physical cores in this process's affinity mask, >= 1 (ADR 0056).

    Python's own override (``PYTHON_CPU_COUNT``, ``-X cpu_count``, 3.13) counts as
    given. Without an affinity mask (macOS, Windows) the standard library cannot tell
    a core from a hardware thread, so this is the logical CPUs there.
    """
    if not hasattr(os, "sched_getaffinity"):
        if hasattr(os, "process_cpu_count"):  # Python 3.13
            return os.process_cpu_count() or 1
        return os.cpu_count() or 1
    mask = frozenset(os.sched_getaffinity(0))
    if hasattr(os, "process_cpu_count") and os.process_cpu_count() != len(mask):
        return os.process_cpu_count() or 1
    return _mpi.node_share(mask, [mask])


def solve_threads(threads: int | None) -> int:
    """One process's count: ``threads``, else ``NGS_NUM_THREADS``, else its cores.

    An ``NGS_NUM_THREADS`` that is not a positive integer is treated as unset. Under
    ranks `gfem._mpi.default_threads` decides instead (ADR 0077).
    """
    if threads is not None:
        return threads
    try:
        chosen = int(os.environ.get("NGS_NUM_THREADS", ""))
    except ValueError:
        chosen = 0
    return chosen if chosen > 0 else available_cpus()


@contextlib.contextmanager
def task_manager(threads: int | None, comm: Any = None) -> Generator[None]:
    """Run NGSolve assembly and solves threaded.

    Args:
        threads: CPU threads (``SolverOptions.threads``); ``None`` is `solve_threads`,
            or the count of a TaskManager the caller already entered, and under
            ``comm`` is `gfem._mpi.default_threads`.
        comm: The resolved communicator. The first distributed solve also sets
            each loaded BLAS to the rank's threads (ADR 0076).
    """
    if comm is not None:
        threads = _mpi.default_threads(comm) if threads is None else threads
        blas.settle(threads)
    if threads is not None:
        ngs.SetNumThreads(threads)
    elif ngs.GetNumThreads() == 1:  # 1 outside any TaskManager
        ngs.SetNumThreads(solve_threads(None))
    with ngs.TaskManager():
        yield
