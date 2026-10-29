"""gfem's MKL PARDISO inverse type, with gfem's flags (the ``mkl`` extra).

NGSolve's MKL PARDISO wrapper reads its flags from the matrix it factors
(``inverse_flags``). A matrix NGSolve builds itself, such as BDDC's coarse
(wirebasket) matrix, carries none, so the wrapper's defaults would factor it
as unsymmetric LU with minimum degree and up to 16 refinement steps. gfem
registers a subclass that applies `PARDISO_FLAGS` itself, under two inverse
names: one symmetric, one for general operators (without ``symmetric``).
`pardiso_inverse` gives the name to pass as ``inverse=``. NGSolve's own
``"pardiso"`` name is never used for them.

Every call of the subclass runs with the calling thread's Intel OpenMP block
time at 0 and sets the previous value back after it. NGSolve's wrapper
suspends the TaskManager only during the call; at the default block time
(200 ms) MKL's OpenMP threads then busy-wait on the cores the TaskManager
resumes on. libiomp5 is only ever the copy MKL already loaded: without it
(another MKL threading layer, or no MKL), the call is left as it is.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import sys
from typing import Any, ClassVar

import ngsolve as ngs

from gfem._linalg import Symmetry
from gfem._linalg.base import equals_its_transpose

#: gfem's flags for NGSolve's MKL PARDISO wrapper: MKL's OpenMP parallel METIS
#: ordering (the same nested dissection and fill as serial METIS, with a
#: faster analysis, phase 11, at 2-4 threads), the symmetric matrix type
#: (gfem stores the full matrix, so NGSolve would otherwise factor it as
#: unsymmetric LU, with twice the fill), and refinement 0: none unless a
#: pivot was perturbed (``iparm[13] > 0``), then two steps per solve.
#: NGSolve's wrapper asks for up to 16 refinement steps with no
#: backward-error stop: they stagnate without lowering the residual and make
#: a solve cost about 4.5 substitutions instead of one. 0 is MKL's default.
PARDISO_FLAGS = {"ordering": "parallel_metis", "symmetric": True, "refinement": 0}
#: The inverse name of gfem's PARDISO with `PARDISO_FLAGS` (``A^T = A``).
SYMMETRIC = "gfem_pardiso"
#: The inverse name of gfem's PARDISO for a general operator: `PARDISO_FLAGS`
#: without ``symmetric`` (LU).
GENERAL = "gfem_pardiso_general"


@functools.cache
def pardiso_available() -> bool:
    """Whether NGSolve found MKL PARDISO in a pip ``mkl`` (the ``mkl`` extra).

    Reads private attributes of NGSolve's wrapper: if they are missing (another
    NGSolve version), gfem's solvers stay sparse Cholesky.
    """
    try:
        from ngsolve.solvers import mkl_pardiso
    except ImportError:
        return False
    return (
        getattr(mkl_pardiso, "_pardiso", None) is not None
        and getattr(mkl_pardiso, "_mkl_free_buffers", None) is not None
    )


def pardiso_inverse(symmetry: Symmetry, *, is_complex: bool) -> str | None:
    """The PARDISO inverse name for an operator of ``symmetry``, or ``None``.

    `SYMMETRIC` when the operator has ``A^T = A`` (`equals_its_transpose`),
    `GENERAL` otherwise (a complex Hermitian or a general operator); ``None``
    without MKL PARDISO (`pardiso_available`). The type is registered on
    first use.
    """
    if not pardiso_available():
        return None
    _register()
    return (
        SYMMETRIC if equals_its_transpose(symmetry, is_complex=is_complex) else GENERAL
    )


@functools.cache
def _register() -> None:
    """Register gfem's PARDISO inverse types with NGSolve, once."""
    from ngsolve.solvers.mkl_pardiso import MKLPardiso

    class GfemPardiso(MKLPardiso):
        """NGSolve's MKL PARDISO wrapper with `PARDISO_FLAGS`."""

        flags: ClassVar[dict[str, Any]] = PARDISO_FLAGS

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._apply_flags(self.flags)

        def _call_pardiso(self, phase: int) -> None:
            # PARDISO refuses an empty matrix: a coarse problem whose DOFs
            # are all fixed has nothing to factor or solve
            if self._n[0] > 0:  # ty: ignore[unresolved-attribute]
                with _block_time_zero():
                    super()._call_pardiso(phase)

    class GfemPardisoGeneral(GfemPardiso):
        """gfem's PARDISO for a general operator: unsymmetric LU."""

        flags: ClassVar[dict[str, Any]] = {
            k: v for k, v in PARDISO_FLAGS.items() if k != "symmetric"
        }

    ngs.la.RegisterInverseType(SYMMETRIC, GfemPardiso)
    ngs.la.RegisterInverseType(GENERAL, GfemPardisoGeneral)


# --- MKL's OpenMP threads ------------------------------------------------------------


@contextlib.contextmanager
def _block_time_zero():
    """Run the block with the calling thread's OpenMP block time at 0.

    The block time in force when a PARDISO call starts its parallel regions
    is how long MKL's threads busy-wait after it; it is a per-thread setting.
    The previous value is set back after the block, also when it raises.
    Without libiomp5, a block that completes settles the lookup (`_OpenMP`);
    one that raises does not.
    """
    openmp = _openmp()
    if openmp is None:
        yield
        _OPENMP.record_completed_call()
        return
    previous = openmp.kmp_get_blocktime()
    openmp.kmp_set_blocktime(0)
    try:
        yield
    finally:
        openmp.kmp_set_blocktime(previous)


class _OpenMP:
    """Intel OpenMP (libiomp5) as MKL loaded it, looked up until settled.

    MKL loads libiomp5 during its first call, so a miss before any PARDISO
    call has completed is looked up again (a call that raised may have
    stopped before MKL loaded it); a hit, or a miss once a call has
    completed (MKL uses another threading layer), is final.
    """

    def __init__(self) -> None:
        self._library: Any = None
        self._settled = False
        self._completed = False

    def find(self) -> Any:
        """The bound library, or ``None``."""
        if not self._settled:
            self._library = _load_libiomp5()
            self._settled = self._library is not None or self._completed
        return self._library

    def record_completed_call(self) -> None:
        """Record that a PARDISO call completed without libiomp5 found."""
        self._completed = True


_OPENMP = _OpenMP()


def _openmp() -> Any:
    """libiomp5's ``kmp_get_blocktime``/``kmp_set_blocktime``, or ``None``."""
    return _OPENMP.find()


def _load_libiomp5() -> Any:
    """libiomp5 if it is already loaded, bound; never loads it, never raises.

    A copy loaded by path could be one MKL is not using. ``None`` when it is
    not loaded or lacks the block-time functions.
    """
    try:
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            kernel32.GetModuleHandleW.restype = ctypes.c_void_p
            if not kernel32.GetModuleHandleW("libiomp5md.dll"):
                return None
            library = ctypes.CDLL("libiomp5md.dll")
        else:
            name = "libiomp5.dylib" if sys.platform == "darwin" else "libiomp5.so"
            library = ctypes.CDLL(name, mode=os.RTLD_NOLOAD)
        library.kmp_get_blocktime.argtypes = []
        library.kmp_get_blocktime.restype = ctypes.c_int
        library.kmp_set_blocktime.argtypes = [ctypes.c_int]
        library.kmp_set_blocktime.restype = None
    except (OSError, AttributeError):
        return None
    return library
