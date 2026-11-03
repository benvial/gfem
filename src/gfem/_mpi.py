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

import collections
import collections.abc
import contextlib
import dataclasses
import hashlib
import itertools
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from gfem.errors import (
    DistributedUnsupportedError,
    LinearSolverConvergenceError,
    MeshQualityError,
    PortGeometryError,
    PortModeError,
    RegionNotFoundError,
    SolverUnavailableError,
)

__all__ = [
    "AGREED_ERRORS",
    "FORWARD_ERRORS",
    "AgreedError",
    "StaticComm",
    "agree_all",
    "agree_any",
    "agree_max",
    "agree_min",
    "agreed",
    "bcast_arrays",
    "check_call",
    "comm_key",
    "default_threads",
    "entry_check",
    "gather_rows",
    "install_abort_hook",
    "launched_tasks",
    "max_bcast",
    "node_budget",
    "node_bytes",
    "node_share",
    "on_root",
    "rank",
    "reduce_bcast",
    "refuse",
    "resolve_comm",
    "section",
    "static_comm",
    "warn",
]

#: In launcher order. Never ``SLURM_NTASKS``, which a batch script with no launcher
#: also has, and never a task id.
_TASK_VARIABLES = ("OMPI_COMM_WORLD_SIZE", "SLURM_STEP_NUM_TASKS", "PMI_SIZE")


class AgreedError(RuntimeError):
    """Ranks found together, on an agreed value, that they cannot go on.

    Each rank raises it after the same collective, so the job can catch it and go
    on, unlike a failure one rank alone sees.
    """


#: Raised by every rank together: what `section` lets through.
AGREED_ERRORS: tuple[type[BaseException], ...] = (
    AgreedError,
    DistributedUnsupportedError,
    LinearSolverConvergenceError,
    PortModeError,
    SolverUnavailableError,
)

#: What `section` lets through around a forward or a backward: the agreed errors, and
#: those every rank raises by itself, from replicated inputs alone.
FORWARD_ERRORS: tuple[type[BaseException], ...] = (
    *AGREED_ERRORS,
    MeshQualityError,
    PortGeometryError,
    RegionNotFoundError,
)

_comm_keyval: Any = None
_comm_keys = itertools.count()

#: The page that tells how to install and launch a distributed solve.
RECIPE = "the 'Run a distributed solve' recipe in the gfem docs"
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
    "general_operator": (
        "Only a system equal to its transpose has a distributed solver, and this "
        f"one has none: {_COPIES}."
    ),
    "field_export": (
        "A field across ranks is not sampled into one object, which would hold the "
        "whole sampled mesh in one process: write one file with Field.to_vtu, read "
        "values with Field.at, or solve with comm=MPI.COMM_SELF."
    ),
    "load_result_fields": (
        "load_result of a bundle with fields reads the mesh and holds whole columns "
        "on each rank, so it is refused above one rank: read it on one process, or "
        "pass comm=MPI.COMM_SELF for one-process fields on each rank."
    ),
    "power_flow": (
        f"gfem.power_flow has no distributed path: read Field.at, {_COPIES}."
    ),
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
            f"with Open MPI's module or package loaded, as {RECIPE} describes. "
            "Without a launcher gfem runs on one process and needs neither."
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
            f"Open MPI as {RECIPE} describes, or pass comm=MPI.COMM_SELF for "
            "independent one-process copies."
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
    from gfem.problems import Driven, Eigenmode, Electrostatic

    if isinstance(problem, Eigenmode):
        refuse_feature("eigenmode")
    if isinstance(problem, Electrostatic):
        refuse_feature("electrostatic")
    if isinstance(problem, Driven):
        if problem.sweep == "adaptive":
            refuse_feature("adaptive_sweep")
        if problem.wave_vector is not None:
            refuse_feature("bloch")
        if model._periodic:  # noqa: SLF001
            refuse_feature("periodic")
        if min(options.frequency_workers, len(problem.frequency)) > 1:
            refuse_feature("frequency_workers")
        if options.device.startswith("cuda"):
            refuse_feature("device")
        if options.order == 1 and options.linear_solver in ("auto", "bddc_gmres"):
            refuse_feature("bddc_order_1")


def refuse_feature(feature: str) -> None:
    """Raise the refusal of ``feature``, also from a path below `refuse`."""
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


def agree_min(comm: Any, value: int) -> int:
    """The smallest ``value`` over ranks, the same answer on every rank."""
    if comm is None:
        return value
    return comm.allreduce(value, op=_load_mpi().MIN)


def agree_max(comm: Any, value: int) -> int:
    """The largest ``value`` over ranks, the same answer on every rank."""
    if comm is None:
        return value
    return comm.allreduce(value, op=_load_mpi().MAX)


def agreed[E: BaseException](error: E) -> E:
    """``error``, marked as raised by every rank together, so `section` lets it pass."""
    error._gfem_agreed = True  # ty: ignore[unresolved-attribute]  # noqa: SLF001
    return error


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


def max_bcast(comm: Any, values: np.ndarray) -> np.ndarray:
    """The entrywise maximum of ``values`` over ranks, the same on every rank."""
    if comm is None:
        return values
    local = np.ascontiguousarray(values)
    largest = np.empty_like(local)
    comm.Reduce(local, largest, op=_load_mpi().MAX, root=0)
    comm.Bcast(largest, root=0)
    return largest


def bcast_arrays(comm: Any, arrays: dict[str, np.ndarray] | None) -> dict:
    """Rank 0's named arrays on every rank, without pickling them.

    Pickling would copy each array and stop at 2 GiB in one message.

    Args:
        comm: The resolved communicator, above one rank.
        arrays: The arrays, C-contiguous, on rank 0; ``None`` elsewhere.
    """
    if arrays is None:
        meta = comm.bcast(None, root=0)
        arrays = {name: np.empty(*m) for name, m in meta.items()}
    else:
        comm.bcast({n: (a.shape, a.dtype.str) for n, a in arrays.items()}, root=0)
    for array in arrays.values():
        comm.Bcast(array, root=0)
    return arrays


#: Elements of one message of `gather_rows`: past 2^31 a count overflows, and a
#: ``Gatherv`` of 3.2e9 doubles raised ``MPI_ERR_ARG`` (ADR 0064).
GATHER_MESSAGE = 2**30


def gather_rows(comm: Any, rows: np.ndarray) -> np.ndarray | None:
    """Every rank's ``rows`` on rank 0, in rank order; ``None`` elsewhere.

    Point to point, so the total has no limit. Every rank's rows have one shape past
    the first axis and one type.

    Args:
        comm: The resolved communicator, above one rank.
        rows: This rank's rows, perhaps none.
    """
    rows = np.ascontiguousarray(rows)
    flat = rows.reshape(-1)
    sizes = comm.gather(flat.size, root=0)
    if comm.rank != 0:
        for start in range(0, flat.size, GATHER_MESSAGE):
            comm.Send(flat[start : start + GATHER_MESSAGE], dest=0)
        return None
    gathered = np.empty(sum(sizes), rows.dtype)
    gathered[: flat.size] = flat
    position = flat.size
    for source, size in enumerate(sizes[1:], start=1):
        for start in range(0, size, GATHER_MESSAGE):
            part = gathered[position : position + min(GATHER_MESSAGE, size - start)]
            comm.Recv(part, source=source)
            position += part.size
    return gathered.reshape(-1, *rows.shape[1:])


def comm_key(comm: Any) -> int | None:
    """A hashable name of this communicator, for a cache in this process.

    The handle is no such name, since MPI reuses it after ``Free``: the key is kept
    on the communicator as an attribute, which dies with it. A duplicate gets its
    own. ``None`` is the one-process path.
    """
    global _comm_keyval  # noqa: PLW0603
    if comm is None:
        return None
    if _comm_keyval is None:
        _comm_keyval = _load_mpi().Comm.Create_keyval()
    key = comm.Get_attr(_comm_keyval)
    if key is None:
        key = next(_comm_keys)
        comm.Set_attr(_comm_keyval, key)
    return key


@dataclasses.dataclass(frozen=True)
class StaticComm:
    """A communicator as a static pytree value: hashable, equal by `comm_key`.

    Attributes:
        key: The communicator's `comm_key`.
        comm: The communicator.
    """

    key: int | None
    comm: Any = dataclasses.field(compare=False)


def static_comm(comm: Any) -> StaticComm | None:
    """``comm`` as a static pytree value; ``None`` is the one-process path."""
    return None if comm is None else StaticComm(comm_key(comm), comm)


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


def _sibling_core(cpu: int) -> str:
    """A name of ``cpu``'s physical core; itself when the kernel does not say."""
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        return Path(path).read_text().strip()
    except OSError:
        return str(cpu)


def node_share(
    mask: frozenset[int],
    node_masks: collections.abc.Sequence[frozenset[int]],
    core: collections.abc.Callable[[int], collections.abc.Hashable] = _sibling_core,
) -> int:
    """A rank's threads: its mask's physical cores over the ranks holding that mask.

    ADR 0056's count alone reads 14 on 7 cores under ``--cpu-bind=cores`` and 224 per
    rank under ``--cpu-bind=none``.

    Args:
        mask: The CPUs this rank may run on.
        node_masks: Each rank's mask on this node, this rank's included.
        core: The physical core of a CPU.

    Returns:
        At least 1.
    """
    cores = len({core(cpu) for cpu in mask})
    return max(1, cores // node_masks.count(mask))


_DEFAULT_THREADS: dict[int | None, int] = {}


def default_threads(comm: Any) -> int:
    """``threads=None`` across ranks: one count for all, read once (ADR 0077).

    ``NGS_NUM_THREADS`` if set, as ADR 0056 says, else `node_share`; then the minimum
    over ranks. Collective on the first call on a communicator.
    """
    key = comm_key(comm)
    if key in _DEFAULT_THREADS:
        return _DEFAULT_THREADS[key]
    mask = frozenset(os.sched_getaffinity(0))
    node = comm.Split_type(_load_mpi().COMM_TYPE_SHARED)
    try:
        node_masks = node.allgather(mask)
    finally:
        node.Free()
    chosen = os.environ.get("NGS_NUM_THREADS", "")
    own = int(chosen) if chosen.isdecimal() else node_share(mask, node_masks)
    _DEFAULT_THREADS[key] = agree_min(comm, max(1, own))
    return _DEFAULT_THREADS[key]


#: Seconds a rank waits at an entry check before it says so, once.
ENTRY_PATIENCE = 60.0
_entry_calls: collections.Counter[int | None] = collections.Counter()


def _meet(comm: Any, call: str) -> None:
    """Wait for every rank in Python, saying so on stderr after `ENTRY_PATIENCE`.

    `if rank == 0: gfem.solve(...)` hangs in the first collective with nothing to
    say. A slow rank is legitimate, so this only reports: it never aborts.
    """
    request = comm.Ibarrier()
    start = time.monotonic()
    pause, told = 1e-4, False
    while not request.Test():
        if not told and time.monotonic() - start > ENTRY_PATIENCE:
            told = True
            print(  # noqa: T201
                f"gfem: rank {comm.rank}: waiting over {ENTRY_PATIENCE:g} s for the "
                f"other ranks at {call}: every rank must make the same gfem calls.",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(pause)
        pause = min(2 * pause, 0.05)


def _leaf_digest(leaf: Any) -> str:
    """A digest of a pytree leaf; a tracer's bytes are not known, its shape is."""
    from gfem._checks import is_traced

    if is_traced(leaf):
        return f"traced {leaf.shape} {leaf.dtype}"
    array = np.asarray(leaf)
    if array.dtype == object:
        return _text_digest(repr(leaf))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"{array.shape} {array.dtype.str}".encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _text_digest(text: str) -> str:
    """A digest of a repr, which may print an object's address."""
    return hashlib.blake2b(
        re.sub(r"0x[0-9a-f]+", "", text).encode(), digest_size=16
    ).hexdigest()


def entry_check(
    comm: Any,
    call: str,
    model: Any,
    problem: Any,
    options: Any,
    *,
    arrays: collections.abc.Mapping[str, Any] | None = None,
) -> None:
    """Every rank at this collective call, with the same inputs (ADR 0058).

    Compares a digest of the call's number and name and of the replicated inputs, so
    a rank that drifted is told at the door and not by a hang or a silent wrong answer.

    Args:
        comm: The resolved communicator.
        call: The call's name.
        model: The model, whose leaves are compared one by one.
        problem: The problem, by its printed values.
        options: The solver options, by their printed values.
        arrays: Further replicated values of the call, by name.

    Raises:
        AgreedError: On every rank, naming the first input that differs.
    """
    import jax

    _meet(comm, call)
    leaves, _ = jax.tree_util.tree_flatten_with_path(model)
    items = {
        "the problem": _text_digest(repr(problem)),
        "the options": _text_digest(repr(options)),
        "the ports": _text_digest(repr(model.ports)),
        "the boundaries": _text_digest(repr(model.boundaries)),
        "the mesh's size": f"{model.mesh.tetrahedra.shape} {model.mesh.nodes.shape}",
    }
    items |= {
        f"the material of {region!r}": _text_digest(repr(material))
        for region, material in model.materials.items()
    }
    items |= {
        f"the model's {jax.tree_util.keystr(path)}": _leaf_digest(leaf)
        for path, leaf in leaves
    }
    items |= {name: _leaf_digest(value) for name, value in (arrays or {}).items()}
    _compare(comm, call, items)


def check_call(comm: Any, call: str, arrays: collections.abc.Mapping[str, Any]) -> None:
    """`entry_check` of a collective call with no model: a method of a kept field.

    Args:
        comm: The resolved communicator.
        call: The call's name.
        arrays: The call's replicated values, by name. A rank's piece of a field is
            no such value: its shape stands for it (ADR 0063).

    Raises:
        AgreedError: On every rank, naming the first input that differs.
    """
    _meet(comm, call)
    _compare(comm, call, {n: _leaf_digest(v) for n, v in arrays.items()})


def _compare(comm: Any, call: str, items: dict[str, str]) -> None:
    """Raise on every rank if this call's number, name or ``items`` differ."""
    key = comm_key(comm)
    _entry_calls[key] += 1
    everyone = comm.allgather({"the call": f"{_entry_calls[key]} {call}", **items})
    for other, theirs in enumerate(everyone):
        for name in sorted(theirs.keys() | everyone[0].keys()):
            if theirs.get(name) != everyone[0].get(name):
                raise AgreedError(
                    f"At {call}, rank {other} and rank 0 hold different values of "
                    f"{name}. Every rank must make the same gfem calls with the same "
                    "inputs: a design update computed per rank drifts on unequal "
                    "masks or BLAS thread counts."
                )


def warn(
    comm: Any, message: str, category: type[Warning] = UserWarning, stacklevel: int = 1
) -> None:
    """Warn once per job on a value all ranks hold.

    Rank 0 warns normally. The others warn inside ``catch_warnings(record=True)``,
    which prints nothing and still raises under an ``error`` filter, so a filter
    acts on every rank. What such a filter raises is an agreed error, so `section`
    lets it through.
    """
    try:
        if rank(comm) == 0:
            warnings.warn(message, category, stacklevel=stacklevel + 1)
            return
        with warnings.catch_warnings(record=True):
            warnings.warn(message, category, stacklevel=stacklevel + 1)
    except Warning as raised:
        raised._gfem_agreed = True  # ty: ignore[unresolved-attribute]  # noqa: SLF001
        raise


def on_root(
    comm: Any,
    compute: collections.abc.Callable[[], dict[str, np.ndarray]],
    *,
    agreed: tuple[type[Exception], ...] = (),
) -> dict[str, np.ndarray]:
    """``compute()``'s named arrays on every rank, computed once on rank 0.

    Placement of a whole computation (ADR 0059): every rank must call this, a rank
    holding no DOFs too, so no local test may stand in front of it. Warnings are
    recorded on rank 0 whatever its filters say and replayed by `warn`, and one of
    the ``agreed`` errors is raised on every rank after them, and `section` lets it
    through. Anything else ends the job. On the one-process path this is ``compute()``.

    Args:
        comm: The resolved communicator.
        compute: The computation; run on rank 0 alone.
        agreed: The errors that are a result of the computation, for all ranks.
    """
    if comm is None:
        return compute()
    arrays = None
    outcome = None
    if comm.rank == 0:
        failure = None
        with section(comm, agreed=()), warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            try:
                arrays = {k: np.asarray(a, order="C") for k, a in compute().items()}
            except agreed as error:
                failure = error
        outcome = (failure, [(str(w.message), w.category) for w in seen])
    failure, recorded = comm.bcast(outcome, root=0)
    shared = {} if failure is not None else bcast_arrays(comm, arrays)
    for message, category in recorded:
        warn(comm, message, category, stacklevel=2)
    if failure is not None:
        failure._gfem_agreed = True  # noqa: SLF001
        raise failure
    return shared


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
        if comm is None or getattr(error, "_gfem_agreed", False):
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
