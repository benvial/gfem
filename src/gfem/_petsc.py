"""gfem's one way into PETSc and SLEPc, and the errors when they cannot be used.

petsc4py and slepc4py are imported lazily, and only here: `require_petsc`
and `require_slepc` import them on the first PETSc or SLEPc solve, once per
process, and raise `gfem.SolverUnavailableError` when they are missing or
unusable. They are initialised without command-line arguments (never the
script's ``sys.argv``), and a PETSc another library initialised first is
accepted as it is.

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
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem.errors import SolverUnavailableError

if TYPE_CHECKING:
    from gfem.options import PetscOptions

__all__ = [
    "LOCK",
    "MIN_VERSION",
    "error_text",
    "next_prefix",
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

_RECIPE = "the 'PETSc and SLEPc' recipe in the gfem docs"
#: What a solver field asks for, and the solvers it could use instead.
_SOLVERS = {
    "linear_solver": ("petsc", "use linear_solver='direct' or 'bddc_gmres'"),
    "eigen_solver": ("slepc", "use eigen_solver='arpack'"),
}
#: A PETSc traceback line: ``[0] KSPSetUp() at /path/itfunc.c:123``.
_TRACEBACK = re.compile(r"\(\) at \S+:\d+$")
_RANK = re.compile(r"^\[\d+\]\s*")


def _load(package: str, library: str) -> tuple[Any, Any]:
    """``package`` initialised without arguments, and its ``library`` module."""
    module = __import__(package)
    module.init([])  # before importing the library, which would read sys.argv
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
    version = str(getattr(module, "__version__", ""))
    if _version(version) < MIN_VERSION:
        floor = ".".join(map(str, MIN_VERSION))
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
            `MIN_VERSION`.
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


@functools.cache
def require_slepc(subject: str = "eigen_solver") -> Any:
    """The ``slepc4py.SLEPc`` module, after checking that gfem can use it.

    Checks petsc4py first (`require_petsc`), which slepc4py needs.

    Args:
        subject: The `gfem.SolverOptions` field that asks for SLEPc, which
            the error message names.

    Raises:
        gfem.SolverUnavailableError: As `require_petsc`, or slepc4py is
            missing, cannot be imported, or is older than `MIN_VERSION`.
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

    petsc4py reports it as a `SystemError` ("returned a result with an
    exception set") whose cause is the original exception; any other
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
