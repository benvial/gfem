"""Frequency workers: blocks of a uniform sweep solved in spawned processes.

A frequency worker is a separate process that solves one contiguous block
of a uniform sweep's frequencies (`SolverOptions.frequency_workers`). Each
worker is a process pool of its own, started with ``spawn``, never
``fork``: JAX's and NGSolve's threads are already running in the parent,
and NGSolve's TaskManager is process-global, so threads of one process
cannot solve concurrently. A pool per worker ties a killed worker to its
own block, so the error names the frequencies it was solving. A worker
starts with the OpenMP, MKL and OpenBLAS thread variables set to its share
of the threads (its environment at spawn: the libraries read them when
imported); NGSolve's own threads follow the worker's
``SolverOptions.threads``.

`gfem.Model` does not pickle (its materials are a mapping proxy); its
pytree parts do (`model_parts`, `model_from_parts`).
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import threading
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import jax
import numpy as np

from gfem.model import Model

__all__ = ["model_from_parts", "model_parts", "run", "worker_threads"]

#: Thread-count variables of the libraries a solve may run on.
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
#: Held while this process's environment carries a worker's thread variables.
_START_LOCK = threading.Lock()


def available_cpus() -> int:
    """Logical CPUs this process may run on (its affinity mask), >= 1."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def worker_threads(threads: int | None, workers: int) -> int:
    """Each worker's share of ``threads`` (``None``: `available_cpus`), >= 1."""
    total = threads if threads is not None else available_cpus()
    return max(1, total // workers)


def model_parts(model: Model) -> tuple[Any, Any]:
    """``model`` as picklable parts, its leaves as NumPy arrays."""
    children, aux = jax.tree.map(np.asarray, model).tree_flatten()
    return aux, children


def model_from_parts(aux: Any, children: Any) -> Model:
    """The model of `model_parts`."""
    return Model.tree_unflatten(aux, children)


@contextlib.contextmanager
def _thread_variables(threads: int) -> Generator[None]:
    """This process's environment with the thread variables set to ``threads``.

    A process spawned meanwhile starts with them. The environment is
    restored on exit; `_START_LOCK` keeps two `run` calls from interleaving.
    """
    with _START_LOCK:
        saved = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
        os.environ.update(dict.fromkeys(_THREAD_VARIABLES, str(threads)))
        try:
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def run(
    function: Callable[..., Any],
    arguments: Sequence[tuple],
    labels: Sequence[str],
    *,
    threads: int,
) -> list[Any]:
    """``function(*a)`` for each ``a`` of ``arguments``, each in its own worker.

    Args:
        function: A module-level function (it is pickled by name).
        arguments: One argument tuple per worker.
        labels: What each worker solves, for errors (e.g. its frequencies).
        threads: Each worker's threads (`worker_threads`).

    Returns:
        The workers' results, in ``arguments`` order.

    Raises:
        RuntimeError: A worker stopped without a result (killed, e.g. for
            lack of memory). A worker's own exception is re-raised as is.
    """
    context = multiprocessing.get_context("spawn")
    with contextlib.ExitStack() as stack:
        futures = []
        # a pool starts its process on the first submit
        with _thread_variables(threads):
            for args in arguments:
                pool = stack.enter_context(
                    ProcessPoolExecutor(max_workers=1, mp_context=context)
                )
                futures.append(pool.submit(function, *args))
        out = []
        for future, label in zip(futures, labels, strict=True):
            try:
                out.append(future.result())
            except BrokenProcessPool as err:
                raise RuntimeError(
                    f"The frequency worker solving {label} stopped without a "
                    "result: it was killed, most likely for lack of memory (each "
                    "worker holds one solve). Use fewer "
                    "SolverOptions.frequency_workers, or frequency_workers=1."
                ) from err
    return out
