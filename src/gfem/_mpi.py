"""The one module that touches mpi4py: launch resolution, agreement, failures.

A distributed solve is SPMD (ADR 0057). A resolved communicator of ``None`` is
the one-process path: every function here is then trivial, mpi4py is never
imported, and a call site reads the same on one rank or many. A value that
decides control flow must be held bitwise by all ranks, so it goes through
`agree_any`, `agree_all` or `reduce_bcast` (ADR 0058). A failure one rank alone
sees cannot be raised, because its peers may sit in a C++ collective, so it
ends the job through `section` or the abort hook.
"""

from __future__ import annotations

import collections.abc
import contextlib
import os
import sys
import traceback
import warnings
from typing import Any

import numpy as np

from gfem.errors import (
    DistributedUnsupportedError,
    LinearSolverConvergenceError,
    SolverUnavailableError,
)

__all__ = [
    "AGREED_ERRORS",
    "agree_all",
    "agree_any",
    "install_abort_hook",
    "launched_tasks",
    "node_budget",
    "node_bytes",
    "rank",
    "reduce_bcast",
    "refuse",
    "resolve_comm",
    "section",
    "warn",
]

#: In launcher order. Never ``SLURM_NTASKS``, which a batch script with no launcher
#: also has, and never a task id.
_TASK_VARIABLES = ("OMPI_COMM_WORLD_SIZE", "SLURM_STEP_NUM_TASKS", "PMI_SIZE")

#: Raised by every rank together: what `section` lets through.
AGREED_ERRORS: tuple[type[BaseException], ...] = (
    DistributedUnsupportedError,
    LinearSolverConvergenceError,
    SolverUnavailableError,
)

_COPIES = "or run independent one-process copies with comm=MPI.COMM_SELF"

#: Each refusal's text. The keys are `gfem.DistributedUnsupportedError.feature`.
_REFUSALS = {
    "adaptive_sweep": (
        f"An adaptive sweep has no distributed path: use a uniform sweep, {_COPIES}."
    ),
    "eigenmode": f"An eigenmode solve has no distributed path: {_COPIES}.",
    "electrostatic": f"An electrostatic solve has no distributed path: {_COPIES}.",
    "periodic": f"A periodic model has no distributed path: {_COPIES}.",
    "bloch": f"A Bloch wave vector has no distributed path: {_COPIES}.",
    "frequency_workers": (
        "Frequency workers have no distributed path: use frequency_workers=1, "
        f"one solve per communicator from comm.Split, {_COPIES}."
    ),
    "device": f"A GPU has no distributed path: use device='cpu', {_COPIES}.",
    "bddc_order_1": (
        "BDDC at order 1 has a coarse problem as large as the system, so it is "
        f"refused above one rank: use linear_solver='direct' or order=2, {_COPIES}."
    ),
    "driven": f"A distributed driven solve is not built yet: {_COPIES}.",
    "port_mode": f"A distributed port mode solve is not built yet: {_COPIES}.",
}


def launched_tasks(environ: collections.abc.Mapping[str, str] | None = None) -> int:
    """The number of launched tasks, without importing mpi4py.

    The first launcher variable set decides, since an ``mpirun`` inside a Slurm
    step has both. mpi4py's world size counts as well if it is already loaded.

    Args:
        environ: The environment; ``None`` reads ``os.environ``.
    """
    environ = os.environ if environ is None else environ
    tasks = 1
    for name in _TASK_VARIABLES:
        value = environ.get(name, "")
        if value.isdecimal():
            tasks = int(value)
            break
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is not None and mpi.Is_initialized() and not mpi.Is_finalized():
        tasks = max(tasks, mpi.COMM_WORLD.size)
    return tasks


def _load_mpi() -> Any:
    """``mpi4py.MPI``, or `SolverUnavailableError` naming the remedy."""
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError) as err:
        # mpi4py with no library anywhere raises RuntimeError, not ImportError
        raise SolverUnavailableError(
            f"gfem was launched on several ranks but cannot load MPI ({err}). A "
            'distributed solve needs Open MPI and mpi4py: pip install "gfem[mpi]", '
            "with Open MPI's module or package loaded. Without a launcher gfem runs "
            "on one process and needs neither."
        ) from err
    return MPI


def resolve_comm(comm: Any = None) -> Any:
    """The communicator a solve runs on; ``None`` is the one-process path.

    A given communicator is distributed if and only if its size is above 1. With
    ``None`` the launcher decides: at one task mpi4py is not imported, above one
    the solve is on the world communicator or raises. Above size 1 only Open MPI
    passes, because the NGSolve wheel gives silently wrong numbers on the MPICH
    ABI, and the abort hook is chained.

    Args:
        comm: An mpi4py intracommunicator, or ``None``.

    Raises:
        TypeError: ``comm`` is not a communicator.
        SolverUnavailableError: mpi4py or an MPI library cannot be loaded, the
            launcher did not start MPI, or the MPI is not Open MPI.
    """
    if comm is None:
        tasks = launched_tasks()
        if tasks <= 1:
            return None
        comm = _load_mpi().COMM_WORLD
        if comm.size == 1:
            raise SolverUnavailableError(
                f"The launcher started {tasks} tasks, but each one sees an MPI world "
                "of size 1, so they cannot solve together. Under Slurm launch with "
                "srun --mpi=pmix, or pass comm=MPI.COMM_SELF for independent "
                "one-process copies."
            )
    else:
        if not isinstance(getattr(comm, "size", None), int):
            raise TypeError(
                f"comm must be an mpi4py communicator, got {type(comm).__name__}."
            )
        if comm.size == 1:
            return None
    mpi = _load_mpi()
    vendor = mpi.get_vendor()[0]
    if vendor != "Open MPI":
        raise SolverUnavailableError(
            f"A solve above one rank needs Open MPI, but mpi4py runs on {vendor}: "
            "the NGSolve wheel gives silently wrong numbers on other MPIs. Load "
            "Open MPI and reinstall mpi4py against it, or pass comm=MPI.COMM_SELF "
            "for independent one-process copies."
        )
    install_abort_hook()
    return comm


def refuse(model: Any, problem: Any, options: Any) -> None:
    """Raise if this solve has no distributed path, before any mesh or rule (ADR 0072).

    Reads static replicated values only, so it needs no communication and raises
    on every rank, also while JAX traces. A problem it does not know falls through,
    so `gfem.solve`'s own type error stands.

    Raises:
        DistributedUnsupportedError: Its ``feature`` names what has no path.
    """
    from gfem.problems import Driven, Eigenmode, Electrostatic, PortMode

    if isinstance(problem, Eigenmode):
        _refuse("eigenmode")
    if isinstance(problem, Electrostatic):
        _refuse("electrostatic")
    # No distributed solve exists yet. Runs on rank 0 and broadcasts by ADR 0059.
    if isinstance(problem, PortMode):
        _refuse("port_mode")
    if isinstance(problem, Driven):
        if problem.sweep == "adaptive":
            _refuse("adaptive_sweep")
        if problem.wave_vector is not None:
            _refuse("bloch")
        if model._periodic:  # noqa: SLF001
            _refuse("periodic")
        if min(options.frequency_workers, len(problem.frequency)) > 1:
            _refuse("frequency_workers")
        if options.device.startswith("cuda"):
            _refuse("device")
        if options.order == 1 and options.linear_solver in ("auto", "bddc_gmres"):
            _refuse("bddc_order_1")
        # No distributed solve exists yet: no silent copies either.
        _refuse("driven")


def _refuse(feature: str) -> None:
    raise DistributedUnsupportedError(feature, _REFUSALS[feature])


def rank(comm: Any) -> int:
    """This process's rank; 0 on the one-process path."""
    return 0 if comm is None else comm.rank


def agree_any(comm: Any, *, flag: bool) -> bool:
    """Whether ``flag`` holds on some rank, the same answer on every rank."""
    if comm is None:
        return bool(flag)
    return comm.allreduce(bool(flag), op=_load_mpi().LOR)


def agree_all(comm: Any, *, flag: bool) -> bool:
    """Whether ``flag`` holds on all ranks, the same answer on every rank."""
    if comm is None:
        return bool(flag)
    return comm.allreduce(bool(flag), op=_load_mpi().LAND)


def reduce_bcast(comm: Any, values: Any) -> Any:
    """The rank sum of ``values``, with the same bits on every rank.

    It is reduced to rank 0 and then broadcast: an ``Allreduce`` was bitwise equal
    in a probe, but MPI does not promise that, and a one-bit drift would stop a
    design loop at its next entry check. On the one-process path ``values`` itself
    comes back.
    """
    if comm is None:
        return values
    local = np.ascontiguousarray(values)
    total = np.empty_like(local)
    comm.Reduce(local, total, op=_load_mpi().SUM, root=0)
    comm.Bcast(total, root=0)
    return total


def node_bytes(comm: Any, local: int) -> int:
    """Bytes held per node: summed over each node's ranks, the largest node's sum.

    The one number eviction reads against `node_budget`: rank 0 holds far more than
    the others, so an equal share per rank would evict every entry.
    """
    if comm is None:
        return int(local)
    mpi = _load_mpi()
    node = comm.Split_type(mpi.COMM_TYPE_SHARED)
    try:
        held = node.allreduce(int(local), op=mpi.SUM)
    finally:
        node.Free()
    return comm.allreduce(held, op=mpi.MAX)


def node_budget(comm: Any, local: int) -> int:
    """The per-node budget: the smallest of what each rank reads for its node.

    A highmem node reads four times a standard one in the same job.
    """
    if comm is None:
        return int(local)
    return comm.allreduce(int(local), op=_load_mpi().MIN)


def warn(
    comm: Any, message: str, category: type[Warning] = UserWarning, stacklevel: int = 1
) -> None:
    """Warn once per job on a value all ranks hold.

    Rank 0 warns normally. The others warn inside ``catch_warnings(record=True)``,
    which prints nothing and still raises under an ``error`` filter, so a filter
    acts on every rank.
    """
    if rank(comm) == 0:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
        return
    with warnings.catch_warnings(record=True):
        warnings.warn(message, category, stacklevel=stacklevel + 1)


def _abort(comm: Any, error: BaseException) -> None:
    """Print the traceback with the rank and end the whole job."""
    text = "".join(traceback.format_exception(error))
    print(  # noqa: T201
        f"gfem: rank {rank(comm)}: a failure on one rank ends the job:\n{text}",
        file=sys.stderr,
        flush=True,
    )
    _load_mpi().COMM_WORLD.Abort(1)


@contextlib.contextmanager
def section(
    comm: Any, agreed: tuple[type[BaseException], ...] = AGREED_ERRORS
) -> collections.abc.Generator[None]:
    """Host work under ranks: agreed errors raise, anything else ends the job.

    Peers of a rank-local failure may be inside a C++ collective and never reach an
    agreement point. On the one-process path every error passes through.

    Args:
        comm: The resolved communicator.
        agreed: The classes all ranks raise together.
    """
    try:
        yield
    except agreed:
        raise
    except BaseException as error:
        if comm is None:
            raise
        _abort(comm, error)


def install_abort_hook() -> None:
    """Chain a ``sys.excepthook`` that ends the job, once.

    An unhandled exception on one rank otherwise hangs its peers, under ``srun`` and
    ``mpirun`` alike. The hook sits on ``sys.excepthook`` itself, so whoever
    replaces that hook later also takes this state with it.
    """
    previous = sys.excepthook
    if getattr(previous, "_gfem_abort", False):
        return
    mpi = _load_mpi()

    def hook(kind, value, trace) -> None:
        try:
            print(  # noqa: T201
                f"gfem: rank {mpi.COMM_WORLD.rank}: "
                "an unhandled exception ends the job:",
                file=sys.stderr,
                flush=True,
            )
            previous(kind, value, trace)
            sys.stderr.flush()
        finally:
            mpi.COMM_WORLD.Abort(1)

    hook._gfem_abort = True  # ty: ignore[unresolved-attribute]  # noqa: SLF001
    sys.excepthook = hook
