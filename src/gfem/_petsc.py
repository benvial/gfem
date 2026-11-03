"""gfem's one way into PETSc and SLEPc, and the errors when they cannot be used.

petsc4py and slepc4py are imported lazily, and only here: `require_petsc`
and `require_slepc` import them on the first PETSc or SLEPc solve, once per
process, and raise `gfem.SolverUnavailableError` when they are missing or
unusable. They are initialised without command-line arguments (never the
script's ``sys.argv``), and a PETSc another library initialised first is
accepted as it is.

gfem documents two builds (ADR 0067). The no-MPI stack serves one process and rank
copies, which pass PETSc its launch option; only the MPI stack serves a distributed
solve (`require_mpi_petsc`). `has_mpi` tells them apart without initialising PETSc.

Every PETSc or SLEPc object gfem creates gets its own options prefix
(`next_prefix`). gfem's options sit in PETSc's global options database only
while that object reads them (`scoped_options`), so they never reach a
user's own PETSc objects, and a user's unprefixed options never reach
gfem's. One process-wide reentrant lock, `LOCK`, guards every PETSc call.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import re
import threading
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem import _mpi
from gfem.errors import SolverUnavailableError

if TYPE_CHECKING:
    from gfem.options import PetscOptions

__all__ = [
    "LOCK",
    "MIN_VERSION",
    "error_text",
    "has_mpi",
    "mpi_stack_usable",
    "next_prefix",
    "require_mpi_petsc",
    "require_petsc",
    "require_slepc",
    "scoped_options",
    "unread",
    "unwrap",
]

#: The oldest PETSc / SLEPc (major, minor) gfem supports: the oldest its
#: PETSc tests run on.
MIN_VERSION = (3, 26)
#: Held around every PETSc call gfem makes (reentrant: SLEPc's shell
#: operator calls gfem's PETSc linear solver from inside a SLEPc solve).
LOCK = threading.RLock()
_PREFIXES = itertools.count()

_RECIPE = "the 'Solve with PETSc and SLEPc' recipe in the gfem docs"
#: Lets a PETSc without MPI start under a launcher: each rank is a copy of its own.
_ALLOW_LAUNCH = "-mpiuni-allow-multiprocess-launch"
#: What a solver field asks for, and the solvers it could use instead.
_SOLVERS = {
    "linear_solver": ("petsc", "use linear_solver='direct' or 'bddc_gmres'"),
    "eigen_solver": ("slepc", "use eigen_solver='arpack'"),
}
#: A PETSc traceback line: ``[0] KSPSetUp() at /path/itfunc.c:123``.
_TRACEBACK = re.compile(r"\(\) at \S+:\d+$")
_RANK = re.compile(r"^\[\d+\]\s*")


@functools.cache
def has_mpi() -> bool:
    """Whether the installed PETSc stack is the MPI one; False without a stack.

    Reads ``PETSC_HAVE_MPIUNI`` in ``petscconf.h``: ``hasExternalPackage("mpi")`` is
    True on both stacks. It initialises nothing, so a distributed solve can refuse
    a stack whose ``init`` would stop it.
    """
    try:
        petsc4py = __import__("petsc4py")
        directory = Path(petsc4py.get_config()["PETSC_DIR"])
        header = (directory / "include" / "petscconf.h").read_text()
    except Exception:  # noqa: BLE001 - also a broken install: `_import` tells why
        return False
    return "PETSC_HAVE_MPIUNI" not in header


def _load(package: str, library: str) -> tuple[Any, Any]:
    """``package`` initialised without the script's arguments, and its ``library``.

    A launched PETSc without MPI stops at ``init`` unless told that each rank runs
    alone, which is what a rank copy does; a distributed solve never gets this far
    on that stack (`require_mpi_petsc`).
    """
    module = __import__(package)
    copies = package == "petsc4py" and _mpi.launched_tasks() > 1 and not has_mpi()
    # before importing the library, which would read sys.argv
    module.init([_ALLOW_LAUNCH] if copies else [])
    return module, getattr(__import__(package, fromlist=[library]), library)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", text)[:2])


def _import(subject: str, package: str, library: str, build: str) -> tuple[Any, Any]:
    """`_load`, raising `SolverUnavailableError` for ``subject`` on failure."""
    value, other = _SOLVERS[subject]
    head = f"{subject}={value!r} cannot run"
    fix = f"{build} as {_RECIPE} describes, or {other}."
    try:
        module, lib = _load(package, library)
    except Exception as err:  # also a broken shared library, a failed init
        if isinstance(err, ModuleNotFoundError) and err.name == package:
            reason = f"{package} is not installed ({err})"
        else:
            reason = f"{package} cannot be imported: {type(err).__name__}: {err}"
        raise SolverUnavailableError(f"{head}: {reason}. {fix}") from err
    version = str(getattr(module, "__version__", None) or "")
    floor = ".".join(map(str, MIN_VERSION))
    if not _version(version):
        raise SolverUnavailableError(
            f"{head}: {package} of unknown version (gfem supports {floor} or "
            f"newer). {fix}"
        )
    if _version(version) < MIN_VERSION:
        raise SolverUnavailableError(
            f"{head}: {package} {version} is older than gfem supports ({floor}). {fix}"
        )
    return module, lib


@functools.cache
def require_petsc(subject: str = "linear_solver") -> Any:
    """The ``petsc4py.PETSc`` module, after checking that gfem can use it.

    Imported and initialised once per process, on first use.

    Args:
        subject: The `gfem.SolverOptions` field that asks for PETSc
            (``"linear_solver"``, or ``"eigen_solver"`` for SLEPc), which
            the error message names.

    Raises:
        gfem.SolverUnavailableError: petsc4py is missing or cannot be
            imported, its PETSc has real scalars, or it is older than
            `MIN_VERSION` or of unknown version.
    """
    build = "Build PETSc and petsc4py with complex scalars"
    _, PETSc = _import(subject, "petsc4py", "PETSc", build)
    scalar = np.dtype(PETSc.ScalarType)
    if scalar.kind != "c":
        value, other = _SOLVERS[subject]
        raise SolverUnavailableError(
            f"{subject}={value!r} cannot run: the installed PETSc uses real "
            f"scalars (PETSc.ScalarType is {scalar.name}), and gfem's operators "
            "are complex. Rebuild PETSc with --with-scalar-type=complex as "
            f"{_RECIPE} describes, or {other}."
        )
    return PETSc


def mpi_stack_usable() -> bool:
    """Whether this rank can use the MPI stack; initialises PETSc if it can."""
    if not has_mpi():
        return False
    try:
        require_petsc()
    except SolverUnavailableError:
        return False
    return True


def require_mpi_petsc(kind: str, comm: Any) -> Any:
    """`require_petsc` for a distributed solve, on the MPI stack only.

    Args:
        kind: The linear solver that asks, for the error message.
        comm: The resolved communicator. All ranks refuse together.

    Raises:
        gfem.SolverUnavailableError: The MPI stack is not installed (checked before
            ``init``, which a no-MPI stack fails under a launcher), or as
            `require_petsc`.
    """
    if not _mpi.agree_all(comm, flag=has_mpi()):
        raise SolverUnavailableError(
            f"linear_solver={kind!r} above one rank needs the MPI PETSc stack, and "
            "this environment does not have it. Build it with "
            f"tools/petsc_wheelhouse.sh --mpi as {_mpi.RECIPE} describes, use "
            "linear_solver='bddc_gmres', or pass comm=MPI.COMM_SELF for independent "
            "one-process copies."
        )
    return require_petsc("linear_solver")


@functools.cache
def require_slepc(subject: str = "eigen_solver") -> Any:
    """The ``slepc4py.SLEPc`` module, after checking that gfem can use it.

    Checks petsc4py first (`require_petsc`), which slepc4py needs.

    Args:
        subject: The `gfem.SolverOptions` field that asks for SLEPc, which
            the error message names.

    Raises:
        gfem.SolverUnavailableError: As `require_petsc`, or slepc4py is
            missing, cannot be imported, or is older than `MIN_VERSION` or
            of unknown version.
    """
    require_petsc(subject)
    build = "Build SLEPc and slepc4py on a complex PETSc"
    _, SLEPc = _import(subject, "slepc4py", "SLEPc", build)
    return SLEPc


def next_prefix() -> str:
    """A new options prefix, ``gfem<n>_``, for one PETSc or SLEPc object."""
    return f"gfem{next(_PREFIXES)}_"


def _value(value: object) -> str | None:
    """A `gfem.options.PetscValue` as PETSc's options database reads it."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)


@contextlib.contextmanager
def scoped_options(prefix: str, pairs: PetscOptions) -> Generator[None]:
    """``pairs`` in PETSc's options database under ``prefix``, for one scope.

    Each ``(name, value)`` is set as ``-{prefix}{name}`` (``None``: a flag,
    a ``bool``: ``true`` / ``false``) on entry, and every one is deleted on
    exit, also when the scope raises. Run the object's ``setFromOptions`` /
    ``setUp`` inside (and a SLEPc ``solve``, which reads its viewer options
    at its end); `LOCK` is held throughout.
    """
    PETSc = require_petsc()
    with LOCK:
        options = PETSc.Options()
        names = [f"-{prefix}{name}" for name, _ in pairs]
        try:
            for name, (_, value) in zip(names, pairs, strict=True):
                options.setValue(name, _value(value))
            yield
        finally:
            for name in names:
                options.delValue(name)


def unread(prefix: str, names: list[str]) -> list[str]:
    """The ``names`` PETSc's options database holds unread under ``prefix``.

    Call inside `scoped_options`, after the object has read its options.
    """
    database = require_petsc().Options()
    return [name for name in names if not database.used(f"{prefix}{name}")]


def unwrap(err: BaseException) -> BaseException:
    """The exception a Python callback raised inside a PETSc call.

    slepc4py's wrapper reports it as a `SystemError` ("returned a result with
    an exception set") whose cause is the original exception; any other
    exception is returned as it is.
    """
    while isinstance(err, SystemError) and err.__cause__ is not None:
        err = err.__cause__
    return err


def error_text(err: BaseException) -> str:
    """PETSc's reason for ``err``, without its traceback lines.

    A ``PETSc.Error`` reads "error code N" and then lines like ``[0]
    KSPSetUp() at file.c:123`` and ``[0] Unable to find requested KSP type
    gmress``; this keeps the latter, joined, and appends ``(PETSc error N)``.
    Other exceptions (after `unwrap`) give their own text.
    """
    err = unwrap(err)
    code = getattr(err, "ierr", None)
    if code is None:
        return str(err)
    lines = (_RANK.sub("", line).strip() for line in str(err).splitlines()[1:])
    reason = " ".join(line for line in lines if line and not _TRACEBACK.search(line))
    return f"{reason} (PETSc error {code})"
