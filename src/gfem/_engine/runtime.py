"""NGSolve threading for one engine call."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import Any

import ngsolve as ngs

from gfem import _mpi
from gfem._linalg import blas

__all__ = ["task_manager"]


@contextlib.contextmanager
def task_manager(threads: int | None, comm: Any = None) -> Generator[None]:
    """Run NGSolve assembly and solves threaded.

    Args:
        threads: CPU threads (``SolverOptions.threads``); ``None`` keeps
            NGSolve's current setting (by default, every core), and under ``comm``
            is `gfem._mpi.default_threads`.
        comm: The resolved communicator. The first distributed solve also sets
            each loaded BLAS to the rank's threads (ADR 0076).
    """
    if comm is not None:
        threads = _mpi.default_threads(comm) if threads is None else threads
        blas.settle(threads)
    if threads is not None:
        ngs.SetNumThreads(threads)
    with ngs.TaskManager():
        yield
