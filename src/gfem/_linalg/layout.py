"""How a space's DOFs lie across ranks: the products of load and state columns.

A column is a NumPy array over a rank's DOFs, shared ones included, and its role
names its status (ADR 0062). A **load column** (``solve``'s ``B``, ``apply``'s
result, a port's excitation or functional) is ``DISTRIBUTED``: a shared DOF's value
is the sum over the ranks that hold it. A **state column** (``solve``'s result,
``x0``, ``apply``'s ``X``) is ``CUMULATED``: each of those ranks holds the value.

A load paired with a state is a rank sum of local products. Two columns of one role
need an exchange and a sum over owned DOFs, since a plain rank sum depends on how a
load is split. NGSolve's vectors carry the status, and `mark_load` and `mark_state`
tell them what a NumPy write just put there: a state in a column vector is wrong by
far more than its size, with no error.

With one process (`ONE_PROCESS`) the roles collapse: `pair` is literally
``load.T @ state`` and nothing communicates. Each product returns an agreed value,
reduced to rank 0 and broadcast (ADR 0058).
"""

from __future__ import annotations

import contextlib
from math import sqrt
from typing import Any

import ngsolve as ngs
import numpy as np

from gfem import _mpi
from gfem._linalg import blas

__all__ = ["ONE_PROCESS", "DofLayout", "RankLayout"]


class DofLayout:
    """One process's layout, where every product is the plain one.

    Attributes:
        comm: The communicator; ``None`` on one process.
        owners: Per local DOF, whether this rank owns it; ``None`` on one process,
            where it owns them all.
    """

    comm: Any = None
    owners: np.ndarray | None = None

    def pair(self, load: np.ndarray, state: np.ndarray) -> np.ndarray:
        """``load.T @ state``: over all ranks, the same for every split of ``load``."""
        return load.T @ state

    def inner(self, a: np.ndarray, b: np.ndarray) -> complex:
        """``vdot(a, b)`` of two load columns."""
        return np.vdot(a, b)

    def norm(self, load: np.ndarray) -> float:
        """The 2-norm of a load column."""
        return float(np.linalg.norm(load))

    def exchange(self, load: np.ndarray) -> np.ndarray:
        """The state column of a load's summed values."""
        return load

    def state_norm(self, vector: ngs.BaseVector) -> float:
        """The 2-norm of an NGSolve vector holding a state."""
        return ngs.Norm(vector)

    def mark_load(self, vector: ngs.BaseVector) -> None:
        """Tell ``vector`` that a load was written into its NumPy view."""

    def mark_state(self, vector: ngs.BaseVector) -> None:
        """Tell ``vector`` that a state was written into its NumPy view."""

    def blas_threads(self, threads: int) -> contextlib.AbstractContextManager[None]:  # noqa: ARG002
        """A phase's BLAS threads; one process leaves the libraries as they are."""
        return contextlib.nullcontext()


#: The layout of every serial handle.
ONE_PROCESS = DofLayout()


class RankLayout(DofLayout):
    """A space's DOFs across the ranks of a communicator.

    A rank that holds no DOFs takes part in every product with empty columns.
    """

    def __init__(self, space: ngs.FESpace, comm: Any) -> None:
        """The layout of ``space`` on the resolved ``comm``."""
        self.comm = comm
        from gfem._linalg.base import free_mask  # base imports this module

        self.pardofs = space.ParallelDofs()
        self.owners = free_mask(self.pardofs.MasterDofs(), space.ndof)
        self._dtype = np.complex128 if space.is_complex else np.float64
        self._work = ngs.GridFunction(space).vec.CreateVector()

    def _sum(self, local: np.ndarray) -> np.ndarray:
        """The rank sum of ``local``, in this space's scalar type."""
        return _mpi.reduce_bcast(self.comm, np.asarray(local, self._dtype))

    def pair(self, load: np.ndarray, state: np.ndarray) -> np.ndarray:
        """The rank sum of the local products ``load.T @ state``."""
        return self._sum(load.T @ state)

    def inner(self, a: np.ndarray, b: np.ndarray) -> complex:
        """``vdot`` over owned DOFs of both loads, exchanged."""
        owned = self.owners
        local = np.vdot(self.exchange(a)[owned], self.exchange(b)[owned])
        return self._sum(np.array([local]))[0].item()

    def norm(self, load: np.ndarray) -> float:
        """The 2-norm over owned DOFs of the exchanged load."""
        return _owned_norm(self.comm, self.exchange(load)[self.owners])

    def exchange(self, load: np.ndarray) -> np.ndarray:
        """The sum over a shared DOF's holders, on each of them: one neighbour exchange.

        Args:
            load: ``(ndof,)`` or ``(ndof, k)``.
        """
        load = np.asarray(load)
        if load.ndim == 1:
            return self._exchange_column(load)
        out = np.zeros(load.shape, self._dtype)
        for j in range(load.shape[1]):
            out[:, j] = self._exchange_column(load[:, j])
        return out

    def _exchange_column(self, column: np.ndarray) -> np.ndarray:
        work = self._work
        work.FV().NumPy()[:] = column
        self.mark_load(work)
        work.Cumulate()
        # a copy: the next exchange writes here
        return np.array(work.FV().NumPy())

    def state_norm(self, vector: ngs.BaseVector) -> float:
        """The 2-norm over owned DOFs: agreed, unlike ``ngs.Norm``, an ``Allreduce``."""
        return _owned_norm(self.comm, vector.FV().NumPy()[self.owners])

    def mark_load(self, vector: ngs.BaseVector) -> None:
        """``vector``'s status becomes ``DISTRIBUTED``."""
        vector.SetParallelStatus(ngs.la.PARALLEL_STATUS.DISTRIBUTED)

    def mark_state(self, vector: ngs.BaseVector) -> None:
        """``vector``'s status becomes ``CUMULATED``."""
        vector.SetParallelStatus(ngs.la.PARALLEL_STATUS.CUMULATED)

    def blas_threads(self, threads: int) -> contextlib.AbstractContextManager[None]:
        """Hold every loaded BLAS at ``threads`` for a phase (ADR 0076)."""
        return blas.held(threads)


def _owned_norm(comm: Any, owned: np.ndarray) -> float:
    """The 2-norm of a column cut to each rank's owned entries."""
    squares = np.array([float(np.sum(owned.real**2 + owned.imag**2))])
    return sqrt(_mpi.reduce_bcast(comm, squares)[0])
