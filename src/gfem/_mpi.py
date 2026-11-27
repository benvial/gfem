"""The one module that touches mpi4py: launch resolution, agreement, failures.

A distributed solve is SPMD. A resolved communicator of ``None`` is
the one-process path: every function here that takes one is then trivial, except
`default_threads`, which has no count to read there, mpi4py is never imported, and
a call site reads the same on one rank or many. A value that decides control flow
must be held bitwise by all ranks, so it goes through `agree_any`, `agree_all` or
`reduce_bcast`. A failure one rank alone
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
import logging
import os
import re
import sys
import time
import traceback
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

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

# an explicit name: a private module path would leak into users' filters
_log = logging.getLogger("gfem.mpi")

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
    "available",
    "bcast_arrays",
    "check_call",
    "comm_key",
    "default_threads",
    "entry_check",
    "gather_rows",
    "install_abort_hook",
    "is_agreed",
    "launched_tasks",
    "max_bcast",
    "meet",
    "node_budget",
    "node_bytes",
    "node_share",
    "on_root",
    "raised_together",
    "rank",
    "reduce_bcast",
    "refuse",
    "refuse_feature",
    "resolve_comm",
    "section",
    "static_comm",
    "warn",
    "world_rank",
]

#: In launcher order. Never ``SLURM_NTASKS``, which a batch script with no launcher
#: also has, and never a task id.
_TASK_VARIABLES = ("OMPI_COMM_WORLD_SIZE", "SLURM_STEP_NUM_TASKS", "PMI_SIZE")
#: A task's rank. ``PMI_RANK`` before ``SLURM_PROCID``: an ``mpiexec`` inside a
#: one-task Slurm step gives every rank ``SLURM_PROCID=0``.
_RANK_VARIABLES = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID")


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
_RUN_COPIES = "run independent one-process copies with comm=MPI.COMM_SELF"
_COPIES = f"or {_RUN_COPIES}"

#: Each refusal's text. The keys are `gfem.DistributedUnsupportedError.feature`.
_REFUSALS = {
    "curved_mesh": (
        "A curved (second-order) mesh has no distributed path: regenerate the "
        f"mesh with gmsh `Mesh.ElementOrder 1`, {_COPIES}."
    ),
    "adaptive_sweep": (
        f"An adaptive sweep has no distributed path: use a uniform sweep, {_COPIES}."
    ),
    "arpack": (
        "ARPACK runs on one process only: leave eigen_solver='auto', which is "
        f"SLEPc above one rank, or set eigen_solver='slepc', {_COPIES}."
    ),
    "mode_field_gradient": (
        "A gradient through an eigenmode's mode field (q_port, the participations, "
        "e_fields) has no distributed path: differentiate omega, frequency or Q, "
        "wrap the value in jax.lax.stop_gradient to use it without its "
        f"derivative, {_COPIES}."
    ),
    "thermal": f"A thermal solve has no distributed path: {_RUN_COPIES}.",
    "dc_conduction": f"A DC conduction solve has no distributed path: {_RUN_COPIES}.",
    "periodic": (
        "Periodic pairs on more than one axis, and an electrostatic solve on a "
        f"periodic model, have no distributed path: {_RUN_COPIES}."
    ),
    "floquet_port": f"A Floquet port has no distributed path: {_RUN_COPIES}.",
    "bloch": f"A Bloch eigenmode solve has no distributed path: {_RUN_COPIES}.",
    "eigenmode_wave_port": (
        "An eigenmode solve with a wave port has no distributed path: each step "
        "of its nonlinear iteration evaluates the port at a complex frequency. "
        "Find the resonances from a gfem.Driven sweep, which runs wave ports "
        f"across ranks, {_COPIES}."
    ),
    "frequency_workers": (
        "Frequency workers have no distributed path: use frequency_workers=1, "
        f"one solve per communicator from comm.Split, {_COPIES}."
    ),
    "device": f"A GPU has no distributed path: use device='cpu', {_COPIES}.",
    "bddc_order_1": (
        "BDDC at order 1 has a coarse problem as large as the system, so it is "
        f"refused above one rank: use linear_solver='direct' or order=2, {_COPIES}."
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
    "optical_force": (
        "gfem.optical_force has no distributed path: write the result with "
        "Job.run(out=) and call it on one process on the field load_result "
        f"gives, {_COPIES}."
    ),
    "region_mean": (
        "gfem.region_mean has no distributed path: write the result with "
        "Job.run(out=) and call it on one process on the field load_result "
        f"gives, {_COPIES}."
    ),
    "element_mean": (
        "gfem.element_mean has no distributed path: write the result with "
        "Job.run(out=) and call it on one process on the field load_result "
        f"gives, {_COPIES}."
    ),
    "dissipated_power": (
        "gfem.dissipated_power has no distributed path: write the result with "
        "Job.run(out=) and call it on one process on the field load_result "
        f"gives, {_COPIES}."
    ),
    "electromagnetic_heating": (
        "gfem.electromagnetic_heating has no distributed path: write the result "
        "with Job.run(out=) and call it on one process on the result load_result "
        f"gives, {_COPIES}."
    ),
    "electric_energy": (
        "gfem.electric_energy has no distributed path: write the result with "
        "Job.run(out=) and call it on one process on the field load_result "
        f"gives, {_COPIES}."
    ),
    "pmg_gmres": (
        "linear_solver='pmg_gmres' runs on one process in this version of gfem. "
        "Across MPI ranks use linear_solver='bddc_gmres' (or 'auto', which picks "
        f"it) or 'direct' with the MPI PETSc stack, {_COPIES}."
    ),
    "petsc_preconditioner": (
        "This PETSc preconditioner has no distributed form (pc_type ilu or icc, "
        "or lu or cholesky with a pc_factor_mat_solver_type other than mumps): "
        "use pc_type bjacobi or asm with sub_pc_type, or the default MUMPS "
        f"factorization, {_COPIES}."
    ),
    "error_estimate": (
        "gfem.error_estimate has no distributed path: estimate a result solved "
        "on one process, or run independent one-process copies with "
        "comm=MPI.COMM_SELF."
    ),
    "mesh_refinement": (
        f"Mesh refinement has no distributed path: refine on one process, {_COPIES}, "
        "then solve the refined model on ranks."
    ),
    "cad_morph": (
        "gfem.shape.cad_morph builds the CAD model with gmsh, which is not "
        "distributed; build and optimize on one process"
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


def world_rank(environ: collections.abc.Mapping[str, str] | None = None) -> int:
    """This process's rank among the launched tasks, without importing mpi4py.

    The first launcher variable set decides; else mpi4py's world rank if it is
    already loaded; else 0.

    Args:
        environ: The environment; ``None`` reads ``os.environ``.
    """
    environ = os.environ if environ is None else environ
    for name in _RANK_VARIABLES:
        value = environ.get(name, "")
        if value.isdecimal():
            return int(value)
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is not None and mpi.Is_initialized() and not mpi.Is_finalized():
        return mpi.COMM_WORLD.rank
    return 0


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


def _world_of_one(tasks: int) -> SolverUnavailableError:
    """The error of a launcher's tasks that each see a world of size 1."""
    return SolverUnavailableError(
        f"The launcher started {tasks} tasks, but each one sees an MPI world "
        "of size 1, so they cannot solve together. Under Slurm launch with "
        "srun --mpi=pmix, or pass comm=MPI.COMM_SELF for independent "
        "one-process copies."
    )


def resolve_comm(comm: Any = None) -> Any:
    """The communicator a solve runs on; ``None`` is the one-process path.

    A given communicator is distributed if and only if its size is above 1, but a
    world of size 1 while the launcher reports more tasks raises, as with ``None``:
    those are silent copies nobody asked for, and ``COMM_SELF`` asks for them by
    name. With ``None`` the launcher decides: at one task mpi4py is not imported,
    above one the solve is on the world communicator or raises. Above size 1 only
    Open MPI passes, because the NGSolve wheel gives silently wrong numbers on the
    MPICH ABI, and the abort hook is chained.

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
            raise _world_of_one(tasks)
    else:
        if not isinstance(getattr(comm, "size", None), int):
            raise TypeError(
                f"comm must be an mpi4py communicator, got {type(comm).__name__}."
            )
        if comm.size == 1:
            mpi = sys.modules.get("mpi4py.MPI")
            tasks = launched_tasks()
            if mpi is not None and tasks > 1 and comm == mpi.COMM_WORLD:
                raise _world_of_one(tasks)
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
    """Raise if this solve has no distributed path, before any mesh or rule.

    Reads static replicated values only, so it needs no communication and raises
    on every rank, also while JAX traces. A problem it does not know falls through,
    so `gfem.solve`'s own type error stands.

    Raises:
        DistributedUnsupportedError: Its ``feature`` names what has no path.
    """
    from gfem._domain import physics_of
    from gfem._floquet import floquet_ports_in
    from gfem.problems import Driven, Eigenmode, Electrostatic

    if model.mesh.curved:
        refuse_feature("curved_mesh")
    physics = physics_of(problem)
    if physics != "em":
        # thermal (steady and transient) or DC conduction: no distributed solve
        refuse_feature(physics)
    if isinstance(problem, Driven):
        if problem.sweep == "adaptive":
            refuse_feature("adaptive_sweep")
        if floquet_ports_in(model.ports):
            refuse_feature("floquet_port")
        if len(model._periodic) > 1:  # noqa: SLF001
            refuse_feature("periodic")
        if min(options.frequency_workers, len(problem.frequency)) > 1:
            refuse_feature("frequency_workers")
    if isinstance(problem, Eigenmode):
        _refuse_eigenmode(model, problem, options)
    if isinstance(problem, Electrostatic) and model._periodic:  # noqa: SLF001
        refuse_feature("periodic")
    if isinstance(problem, (Driven, Eigenmode, Electrostatic)):
        _refuse_options(options)


def _refuse_eigenmode(model: Any, problem: Any, options: Any) -> None:
    """Refuse what an `gfem.Eigenmode` cannot run above one rank.

    Bloch before periodic, since a wave vector needs a pair.
    """
    from gfem._boundaries import first_wave_port

    if problem.wave_vector is not None:
        refuse_feature("bloch")
    if len(model._periodic) > 1:  # noqa: SLF001
        refuse_feature("periodic")
    wave = first_wave_port(model.ports)
    if wave is not None:
        message = _REFUSALS["eigenmode_wave_port"]
        raise DistributedUnsupportedError(
            "eigenmode_wave_port", f"WavePort {wave.name!r}: {message}"
        )
    if options.eigen_solver == "arpack":
        refuse_feature("arpack")


def _refuse_options(options: Any) -> None:
    """Refuse the options no 3D EM problem runs above one rank, whatever its type.

    A `gfem.PortMode` is not checked: it ignores ``device`` and ``linear_solver``
    and runs its 2D solve on rank 0 alone.
    """
    if options.device.startswith("cuda"):
        refuse_feature("device")
    if options.order == 1 and options.linear_solver in ("auto", "bddc_gmres"):
        refuse_feature("bddc_order_1")
    if options.linear_solver == "pmg_gmres":
        refuse_feature("pmg_gmres")
    if options.linear_solver == "petsc" and _serial_only_preconditioner(
        dict(options.petsc_options)
    ):
        refuse_feature("petsc_preconditioner")


def _serial_only_preconditioner(petsc_options: dict[str, Any]) -> bool:
    """Whether ``pc_type`` cannot set up on a distributed matrix on this stack.

    MUMPS is the stack's only distributed factorization, and ``lu`` and
    ``cholesky`` reach it unless another package is named.
    """
    kind = str(petsc_options.get("pc_type", "")).lower()
    if kind in ("ilu", "icc"):
        return True
    package = str(petsc_options.get("pc_factor_mat_solver_type", "mumps")).lower()
    return kind in ("lu", "cholesky") and package != "mumps"


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


def available[T](
    comm: Any, check: collections.abc.Callable[[], T], elsewhere: str
) -> T:
    """``check()``'s value where it passes on every rank; else raise on every rank.

    A rank whose ``check`` raised `gfem.SolverUnavailableError` raises its own error,
    the others one saying ``elsewhere``; both `agreed`.
    """
    error = None
    value = None
    try:
        value = check()
    except SolverUnavailableError as err:
        error = err
    if not agree_all(comm, flag=error is None):
        raise agreed(error or SolverUnavailableError(elsewhere))
    return cast("T", value)


def is_agreed(
    error: BaseException, classes: tuple[type[BaseException], ...] = AGREED_ERRORS
) -> bool:
    """Whether every rank raises ``error`` together: one of ``classes``, or `agreed`."""
    return isinstance(error, classes) or getattr(error, "_gfem_agreed", False)


@contextlib.contextmanager
def raised_together(
    *errors: type[BaseException],
) -> collections.abc.Generator[None]:
    """``errors`` raised inside are raised by every rank together.

    Around code whose failures read agreed or replicated values only: each such
    error is marked `agreed` and raised on.
    """
    try:
        yield
    except errors as error:
        agreed(error)
        raise


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
        comm: The resolved communicator.
        arrays: The arrays, C-contiguous, on rank 0; ``None`` elsewhere.
    """
    if comm is None:
        return {} if arrays is None else arrays
    if arrays is None:
        meta = comm.bcast(None, root=0)
        arrays = {name: np.empty(*m) for name, m in meta.items()}
    else:
        comm.bcast({n: (a.shape, a.dtype.str) for n, a in arrays.items()}, root=0)
    for array in arrays.values():
        comm.Bcast(array, root=0)
    return arrays


#: Elements of one message of `gather_rows`: past 2^31 a count overflows, and a
#: ``Gatherv`` of 3.2e9 doubles raised ``MPI_ERR_ARG``.
GATHER_MESSAGE = 2**30


def gather_rows(comm: Any, rows: np.ndarray) -> np.ndarray | None:
    """Every rank's ``rows`` on rank 0, in rank order; ``None`` elsewhere.

    Point to point, so the total has no limit. Every rank's rows have one shape past
    the first axis and one type.

    Args:
        comm: The resolved communicator.
        rows: This rank's rows, perhaps none.
    """
    if comm is None:
        return rows
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


#: Per communicator key, `node_ids`.
_NODE_IDS: dict[int | None, tuple[int, ...]] = {}


def node_ids(comm: Any) -> tuple[int, ...]:
    """The node of each rank of ``comm``, numbered from 0, the same on every rank.

    Each node's ranks share the rank of its first one in ``comm``, gathered from
    every rank. Collective on the first call on a communicator; ``(0,)`` for one
    process.
    """
    if comm is None:
        return (0,)
    key = comm_key(comm)
    if key in _NODE_IDS:
        return _NODE_IDS[key]
    node = comm.Split_type(_load_mpi().COMM_TYPE_SHARED)
    try:
        leader = node.bcast(comm.rank, root=0)
    finally:
        node.Free()
    leaders = comm.allgather(leader)
    number = {first: i for i, first in enumerate(sorted(set(leaders)))}
    _NODE_IDS[key] = tuple(number[first] for first in leaders)
    return _NODE_IDS[key]


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
    """A rank's threads: each physical core of its mask, over the ranks that hold it.

    A core held by ``n`` ranks of the node counts ``1 / n`` for each; the sum is
    rounded down. Identical masks of ``c`` cores on ``n`` ranks give ``c // n``,
    disjoint masks their own cores, and partly overlapping masks split only the
    cores both hold. The physical-core count alone reads 14 on 7 cores under
    ``--cpu-bind=cores`` and 224 per rank under ``--cpu-bind=none``.

    Args:
        mask: The CPUs this rank may run on.
        node_masks: Each rank's mask on this node, this rank's included.
        core: The physical core of a CPU.

    Returns:
        At least 1.
    """
    holders = collections.Counter(
        held for other in node_masks for held in {core(cpu) for cpu in other}
    )
    share = sum(Fraction(1, holders[held]) for held in {core(cpu) for cpu in mask})
    return max(1, int(share))


_DEFAULT_THREADS: dict[int | None, int] = {}


def ngs_num_threads() -> int | None:
    """``NGS_NUM_THREADS`` as a thread count, or ``None`` when unset.

    A positive integer, after stripping whitespace, counts; anything else (``"0"``,
    empty, junk) is unset. One rule at one process and under ranks.
    """
    value = os.environ.get("NGS_NUM_THREADS", "").strip()
    if value.isascii() and value.isdecimal() and int(value) > 0:
        return int(value)
    return None


def default_threads(comm: Any) -> int:
    """``threads=None`` across ranks: one count for all, read once.

    ``NGS_NUM_THREADS`` if set (`ngs_num_threads`), else
    `node_share`; then the minimum over ranks. Collective on the first call on a
    communicator.

    Args:
        comm: The resolved communicator, never ``None``: one process's count is
            `gfem._engine.runtime.solve_threads`.
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
    own = ngs_num_threads() or node_share(mask, node_masks)
    _DEFAULT_THREADS[key] = agree_min(comm, max(1, own))
    return _DEFAULT_THREADS[key]


#: Seconds a rank waits at an entry check before it says so, once.
ENTRY_PATIENCE = 60.0
_entry_calls: collections.Counter[int | None] = collections.Counter()


def meet(comm: Any, call: str) -> None:
    """Wait for every rank in Python, saying so after `ENTRY_PATIENCE`.

    `if rank == 0: gfem.solve(...)` hangs in the first collective with nothing to
    say. A slow rank is legitimate, so this only reports: it never aborts. The report
    is one WARNING record on the ``gfem.mpi`` logger, with ``extra`` ``rank``,
    ``call`` and ``patience``, not a warning: the wait is rank-local and a warning
    may only be decided on an agreed value. With no logging configured, Python's
    last-resort handler prints its bare message on stderr.
    """
    if comm is None:
        return
    request = comm.Ibarrier()
    # perf_counter, not monotonic: before Python 3.13, Windows' monotonic clock
    # ticks every 15.6 ms, so a wait shorter than a tick reads as no wait at all
    start = time.perf_counter()
    pause, told = 1e-4, False
    while not request.Test():
        if not told and time.perf_counter() - start > ENTRY_PATIENCE:
            told = True
            _log.warning(
                "gfem: rank %d: waiting over %g s for the other ranks at %s: every "
                "rank must make the same gfem calls.",
                comm.rank,
                ENTRY_PATIENCE,
                call,
                extra={"rank": comm.rank, "call": call, "patience": ENTRY_PATIENCE},
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
    """Every rank at this collective call, with the same inputs.

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
    if comm is None:
        return
    import jax

    meet(comm, call)
    leaves, _ = jax.tree_util.tree_flatten_with_path(model)
    items = {
        "the problem": _text_digest(repr(problem)),
        "the options": _text_digest(repr(options)),
        "the ports": _text_digest(repr(model.ports)),
        "the boundaries": _text_digest(repr(model.boundaries)),
        # the incident columns' loads read it on every rank
        "the background": _text_digest(repr(getattr(model, "background", None))),
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
            no such value: its shape stands for it.

    Raises:
        AgreedError: On every rank, naming the first input that differs.
    """
    if comm is None:
        return
    meet(comm, call)
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


_GFEM_DIR = str(Path(__file__).parent) + os.sep


def outside_gfem() -> int:
    """The ``stacklevel`` of the first frame outside gfem, for the caller's caller.

    Counted as `warnings.warn` counts it when called by the function that
    calls this one: a warning there points at the user's line that called
    gfem, however deep inside gfem it is raised.
    """
    frame = sys._getframe(1)  # noqa: SLF001
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_GFEM_DIR):
        frame = frame.f_back
        level += 1
    return level


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
        agreed(raised)
        raise


def on_root(
    comm: Any,
    compute: collections.abc.Callable[[], dict[str, np.ndarray]],
    *,
    results: tuple[type[Exception], ...] = (),
    broadcast: bool = True,
) -> dict[str, np.ndarray]:
    """``compute()``'s named arrays on every rank, computed once on rank 0.

    Placement of a whole computation: every rank must call this, a rank
    holding no DOFs too, so no local test may stand in front of it. Warnings are
    recorded on rank 0 whatever its filters say and replayed by `warn`, and one of
    the ``results`` errors is raised on every rank after them, and `section` lets it
    through. Anything else ends the job. On the one-process path this is ``compute()``.

    Args:
        comm: The resolved communicator.
        compute: The computation; run on rank 0 alone.
        results: The errors that are a result of the computation, for all ranks.
        broadcast: ``False`` when only rank 0 reads the arrays: the other ranks
            then get an empty dict and the arrays are not sent (the failures
            and warnings still are).
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
            except results as error:
                failure = error
        outcome = (failure, [(str(w.message), w.category) for w in seen])
    failure, recorded = comm.bcast(outcome, root=0)
    if failure is not None:
        shared = {}
    elif broadcast:
        shared = bcast_arrays(comm, arrays)
    else:
        shared = arrays or {}
    for message, category in recorded:
        warn(comm, message, category, stacklevel=2)
    if failure is not None:
        raise agreed(failure)
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
    except BaseException as error:
        if comm is None or is_agreed(error, agreed):
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
