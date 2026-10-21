"""Frequency workers: blocks of a uniform sweep solved in spawned processes.

A frequency worker is a separate process that solves one contiguous block
of a uniform sweep's frequencies (`SolverOptions.frequency_workers`). Each
worker is a process pool of its own, started with ``spawn``, never
``fork``: JAX's and NGSolve's threads are already running in the parent,
and NGSolve's TaskManager is process-global, so threads of one process
cannot solve concurrently. A pool per worker ties a killed worker to its
own block, so the error names the frequencies it was solving. A worker
sets the OpenMP, MKL and OpenBLAS thread variables to its share of the
threads when it starts; NGSolve's own threads follow the worker's
``SolverOptions.threads``.

`gfem.Model` does not pickle (its materials are a mapping proxy); its
pytree parts do (`model_parts`, `model_from_parts`).
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import jax
import numpy as np

from gfem.model import Model

__all__ = ["model_from_parts", "model_parts", "run", "worker_threads"]

#: Thread-count variables of the libraries a solve may run on.
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def worker_threads(threads: int | None, workers: int) -> int:
    """Each worker's share of ``threads`` (``None``: every logical CPU), >= 1."""
    total = threads if threads is not None else os.cpu_count() or 1
    return max(1, total // workers)


def model_parts(model: Model) -> tuple[Any, Any]:
    """``model`` as picklable parts, its leaves as NumPy arrays."""
    children, aux = jax.tree.map(np.asarray, model).tree_flatten()
    return aux, children


def model_from_parts(aux: Any, children: Any) -> Model:
    """The model of `model_parts`."""
    return Model.tree_unflatten(aux, children)


def _limit_threads(threads: int) -> None:
    """Set the worker's library thread variables (its pool's initializer)."""
    for name in _THREAD_VARIABLES:
        os.environ[name] = str(threads)


def run(
    function: Callable[..., Any],
    arguments: Sequence[tuple],
    blocks: Sequence[str],
    *,
    threads: int,
) -> list[Any]:
    """``function(*a)`` for each ``a`` of ``arguments``, each in its own worker.

    Args:
        function: A module-level function (it is pickled by name).
        arguments: One argument tuple per worker.
        blocks: What each worker solves, for errors (e.g. its frequencies).
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
        for args in arguments:
            pool = stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=context,
                    initializer=_limit_threads,
                    initargs=(threads,),
                )
            )
            futures.append(pool.submit(function, *args))
        out = []
        for future, block in zip(futures, blocks, strict=True):
            try:
                out.append(future.result())
            except BrokenProcessPool as err:
                raise RuntimeError(
                    f"The frequency worker solving {block} stopped without a "
                    "result: it was killed, most likely for lack of memory (each "
                    "worker holds one solve). Use fewer "
                    "SolverOptions.frequency_workers, or frequency_workers=1."
                ) from err
    return out
