"""NGSolve threading for one engine call."""

from __future__ import annotations

import contextlib
from collections.abc import Generator

import ngsolve as ngs

__all__ = ["task_manager"]


@contextlib.contextmanager
def task_manager(threads: int | None) -> Generator[None]:
    """Run NGSolve assembly and solves threaded.

    Args:
        threads: CPU threads (``SolverOptions.threads``); ``None`` keeps
            NGSolve's current setting (by default, every core).
    """
    if threads is not None:
        ngs.SetNumThreads(threads)
    with ngs.TaskManager():
        yield
