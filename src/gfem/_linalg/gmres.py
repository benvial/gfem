"""gfem's GMRES: NGSolve's GMRES iteration, its vector work on gfem's threads.

`Gmres` is NGSolve 6.2.2608's left-preconditioned GMRES (Givens rotations,
the ``tol``/``atol`` test of ``CheckResidual``) with its Arnoldi step
orthogonalized by CGS2 against a Krylov basis held in NumPy blocks of
`BLOCK` vectors. NGSolve runs that vector work (its complex inner product
and complex-scalar update have no threaded kernel) on one thread; here the
coefficients, updates, norms, scaling, basis copy and ``x += Q y`` run on a
thread pool of ``ngs.GetNumThreads()`` threads, one column range per thread,
each calling single-threaded BLAS through ctypes (the calls release the GIL):
SciPy's bundled OpenBLAS, MKL from the ``mkl`` extra past OpenBLAS's thread
limit (below), or NumPy ``einsum`` without either. Vectors shorter than twice
`COLUMNS` skip the pool.

MKL's thread count is per calling thread: each pool thread, and the solving
thread for the solve, set it to one (`_mkl_one_thread`).

OpenBLAS's thread count is process-wide, and its own threads busy-wait after
each call while NGSolve's TaskManager runs the preconditioner on the same
cores. So while any gfem GMRES solve runs, SciPy's OpenBLAS is held at one
thread (`_OneBlasThread`): the first solve to enter saves its count and sets
1, the last to leave restores it. No BLAS thread pool is woken. OpenBLAS also
has a fixed number of call buffers, two per thread of its build's
``MAX_THREADS`` (64 in the PyPI wheels); more concurrent callers overflow
into an auxiliary table that corrupts the heap. So past ``MAX_THREADS``
(`_ScipyOpenBlas.max_threads`) the pool runs MKL's gemv, which has no such
limit, and without MKL it has ``MAX_THREADS`` threads. Below the limit
OpenBLAS stays: its gemv streams about 30 % faster than MKL's on these
shapes (8 threads, i5-1145G7).

A lucky breakdown (the new direction in the Krylov space) follows NGSolve
upstream 3c33638dff: a relative test, the rotation applied and the stopping
test called before the loop leaves, so the iterate is right.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import importlib.metadata
import itertools
import os
import re
import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from typing import Any

import ngsolve as ngs
import numpy as np
from ngsolve import Matrix, Vector
from ngsolve.krylovspace import GMRESSolver

#: Krylov basis vectors per NumPy block: at most BLOCK - 1 spare vectors
#: (1.1 GB at 10 M complex DOFs).
BLOCK = 8
#: Lucky breakdown: the new direction's norm after orthogonalization is at
#: most this fraction of its norm before.
BREAKDOWN = 1e-12
#: Fewest vector entries per pool thread: shorter vectors (under twice this)
#: run inline, without the pool's hand-offs.
COLUMNS = 1 << 16
#: OpenBLAS's ``MAX_THREADS`` when its build string does not give it: the
#: PyPI wheels' (``NUM_THREADS=64`` in MacPython/openblas-libs).
OPENBLAS_MAX_THREADS = 64


@dataclasses.dataclass(frozen=True)
class _ScipyOpenBlas:
    """SciPy's bundled OpenBLAS: its thread count and the gemv kernels."""

    get_threads: Callable[[], int]
    set_threads: Callable[[int], None]
    zgemv: Any
    dgemv: Any
    #: Concurrent callers its call buffers serve: the build's ``MAX_THREADS``.
    max_threads: int = OPENBLAS_MAX_THREADS


@dataclasses.dataclass(frozen=True)
class _MklBlas:
    """MKL's runtime library: the calling thread's thread count and the gemv kernels."""

    set_threads_local: Callable[[int], int]
    zgemv: Any
    dgemv: Any


def _bind_gemv(zgemv: Any, dgemv: Any) -> None:
    """CBLAS ``zgemv``/``dgemv`` signatures with 32-bit integers (LP64)."""
    P, I, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    zgemv.argtypes, zgemv.restype = [I, I, I, I, P, P, I, P, I, P, P, I], None
    dgemv.argtypes, dgemv.restype = [I, I, I, I, D, P, I, P, I, D, P, I], None


def _max_threads(config: str) -> int:
    """``MAX_THREADS`` from an OpenBLAS build string, else `OPENBLAS_MAX_THREADS`."""
    found = re.search(r"MAX_THREADS=(\d+)", config)
    return int(found.group(1)) if found else OPENBLAS_MAX_THREADS


def _symbol(lib: ctypes.CDLL, name: str) -> Any:
    """``scipy_``-prefixed ``name``, else ``name``; AttributeError if neither."""
    try:
        return getattr(lib, f"scipy_{name}")
    except AttributeError:
        return getattr(lib, name)


@functools.cache
def scipy_openblas() -> _ScipyOpenBlas | None:
    """SciPy's bundled OpenBLAS through ctypes, or None.

    Found among the ``scipy`` distribution's files (``scipy.libs``,
    ``scipy/.dylibs``) after ``scipy.linalg.blas`` is imported, so the handle
    is the library SciPy already mapped. Needs the thread-count getter and
    setter and the ``scipy_``-prefixed CBLAS ``zgemv``/``dgemv`` of
    scipy-openblas32; anything else (SciPy on Accelerate) is None.
    ``max_threads`` is read from ``openblas_get_config``.
    """
    from scipy.linalg import blas  # noqa: F401  (maps the library)

    dist = importlib.metadata.distribution("scipy")
    for file in dist.files or ():
        name = file.name.lower()
        if "openblas" not in name or "64_" in name:
            continue
        if not name.endswith((".so", ".dylib", ".dll")):
            continue
        try:
            lib = ctypes.CDLL(str(dist.locate_file(file)))
            get_threads = _symbol(lib, "openblas_get_num_threads")
            set_threads = _symbol(lib, "openblas_set_num_threads")
            zgemv, dgemv = lib.scipy_cblas_zgemv, lib.scipy_cblas_dgemv
        except (OSError, AttributeError):
            return None
        get_threads.argtypes, get_threads.restype = [], ctypes.c_int
        set_threads.argtypes, set_threads.restype = [ctypes.c_int], None
        _bind_gemv(zgemv, dgemv)
        config = ""
        with contextlib.suppress(AttributeError):
            get_config = _symbol(lib, "openblas_get_config")
            get_config.argtypes, get_config.restype = [], ctypes.c_char_p
            config = (get_config() or b"").decode(errors="replace")
        return _ScipyOpenBlas(
            get_threads, set_threads, zgemv, dgemv, _max_threads(config)
        )
    return None


@functools.cache
def mkl_blas() -> _MklBlas | None:
    """MKL's runtime library from the ``mkl`` distribution through ctypes, or None.

    The ``mkl_rt`` file NGSolve's PARDISO wrapper loads, so the handle is the
    library already mapped once PARDISO has run. None without the ``mkl``
    distribution, without the per-thread thread-count setter or the CBLAS
    gemv kernels, or with ``MKL_INTERFACE_LAYER`` asking for ILP64 (the
    kernels pass 32-bit integers).
    """
    if "ILP64" in os.environ.get("MKL_INTERFACE_LAYER", "").upper():
        return None
    try:
        dist = importlib.metadata.distribution("mkl")
    except importlib.metadata.PackageNotFoundError:
        return None
    for file in dist.files or ():
        if "mkl_rt" not in file.name:
            continue
        try:
            lib = ctypes.CDLL(str(dist.locate_file(file)))
            set_local = lib.MKL_Set_Num_Threads_Local
            zgemv, dgemv = lib.cblas_zgemv, lib.cblas_dgemv
        except (OSError, AttributeError):
            return None
        set_local.argtypes, set_local.restype = [ctypes.c_int], ctypes.c_int
        _bind_gemv(zgemv, dgemv)
        return _MklBlas(set_local, zgemv, dgemv)
    return None


class _OneBlasThread:
    """Holds SciPy's OpenBLAS at one thread while any GMRES solve runs.

    OpenBLAS's count is process-wide (``openblas_set_num_threads_local``
    sets it too, in the pthreads build SciPy ships), so concurrent solves
    share one hold: the first in saves the count and sets 1, the last out
    restores it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entered = 0
        self._saved = 1

    @contextlib.contextmanager
    def held(self, lib: _ScipyOpenBlas | None) -> Generator[None]:
        """One OpenBLAS thread inside the block; nothing without ``lib``."""
        if lib is None:
            yield
            return
        with self._lock:
            if self._entered == 0:
                self._saved = lib.get_threads()
                lib.set_threads(1)
            self._entered += 1
        try:
            yield
        finally:
            with self._lock:
                self._entered -= 1
                if self._entered == 0:
                    lib.set_threads(self._saved)


_ONE_BLAS_THREAD = _OneBlasThread()


@contextlib.contextmanager
def _mkl_one_thread(mkl: _MklBlas | None) -> Generator[None]:
    """The calling thread's MKL at one thread inside the block; nothing without ``mkl``.

    The previous per-thread count (0: MKL's global one) is set back after it.
    """
    if mkl is None:
        yield
        return
    previous = mkl.set_threads_local(1)
    try:
        yield
    finally:
        mkl.set_threads_local(previous)


def _serial_blas(mkl: _MklBlas | None) -> None:
    """The pool's thread initializer, and a profiling seam.

    Sets the pool thread's MKL to one thread (the thread ends with the pool,
    so nothing is set back); on OpenBLAS nothing: `_OneBlasThread` holds it
    at one thread for the solve. It runs once on each pool thread as it
    starts (looked up when the pool is made), so a profiler can wrap it to
    record the pool's thread ids.
    """
    if mkl is not None:
        mkl.set_threads_local(1)


_COL, _NOTRANS, _TRANS, _CONJTRANS = 102, 111, 112, 113
_ONE, _ZERO = np.array([1.0 + 0j]), np.array([0j])


class _Gemv:
    """``Q_s^H q_s`` and ``q_s += Q_s y`` on one column range, by one BLAS gemv.

    A ``(BLOCK, n)`` row-major block is an ``n x BLOCK`` column-major matrix
    with leading dimension n; a column range is its rows ``s``.
    """

    def __init__(self, lib: _ScipyOpenBlas | _MklBlas) -> None:
        self.lib = lib

    def coefficients(
        self, b: np.ndarray, r: int, s: slice, q: np.ndarray
    ) -> np.ndarray:
        """``h_i = <b_i[s], q[s]>`` for the first ``r`` rows of ``b``."""
        n, m, size = b.shape[1], s.stop - s.start, b.itemsize
        h = np.zeros(r, b.dtype)
        P = ctypes.c_void_p
        B, x = P(b.ctypes.data + size * s.start), P(q.ctypes.data + size * s.start)
        if b.dtype.kind == "c":
            one, zero = P(_ONE.ctypes.data), P(_ZERO.ctypes.data)
            self.lib.zgemv(
                _COL, _CONJTRANS, m, r, one, B, n, x, 1, zero, P(h.ctypes.data), 1
            )
        else:
            self.lib.dgemv(
                _COL, _TRANS, m, r, 1.0, B, n, x, 1, 0.0, P(h.ctypes.data), 1
            )
        return h

    def add(
        self, b: np.ndarray, r: int, s: slice, y: np.ndarray, q: np.ndarray
    ) -> None:
        """``q[s] += sum_i y_i b_i[s]`` over the first ``r`` rows of ``b``."""
        n, m, size = b.shape[1], s.stop - s.start, b.itemsize
        y = np.ascontiguousarray(y, b.dtype)
        P = ctypes.c_void_p
        B, x = P(b.ctypes.data + size * s.start), P(q.ctypes.data + size * s.start)
        if b.dtype.kind == "c":
            one = P(_ONE.ctypes.data)
            self.lib.zgemv(
                _COL, _NOTRANS, m, r, one, B, n, P(y.ctypes.data), 1, one, x, 1
            )
        else:
            self.lib.dgemv(
                _COL, _NOTRANS, m, r, 1.0, B, n, P(y.ctypes.data), 1, 1.0, x, 1
            )


class _Einsum:
    """The same kernels without BLAS: portable, about 3x slower per thread."""

    def coefficients(
        self, b: np.ndarray, r: int, s: slice, q: np.ndarray
    ) -> np.ndarray:
        """``h_i = <b_i[s], q[s]>`` for the first ``r`` rows of ``b``."""
        if b.dtype.kind == "c":
            return np.conj(np.einsum("ij,j->i", b[:r, s], np.conj(q[s])))
        return np.einsum("ij,j->i", b[:r, s], q[s])

    def add(
        self, b: np.ndarray, r: int, s: slice, y: np.ndarray, q: np.ndarray
    ) -> None:
        """``q[s] += sum_i y_i b_i[s]`` over the first ``r`` rows of ``b``."""
        q[s] += np.einsum("i,ij->j", np.asarray(y, b.dtype), b[:r, s])


class _Columns:
    """An n-vector's column ranges, one per thread, and the pool running them."""

    def __init__(self, n: int, threads: int, pool: ThreadPoolExecutor | None) -> None:
        edges = np.linspace(0, n, threads + 1).astype(int)
        self.ranges = [
            slice(int(a), int(b)) for a, b in itertools.pairwise(edges) if b > a
        ]
        self.pool = pool

    def map(self, fn: Callable[[slice], Any]) -> list:
        """``fn`` on every range, on the pool when there is one."""
        if self.pool is None:
            return [fn(s) for s in self.ranges]
        return list(self.pool.map(fn, self.ranges))


class _Basis:
    """The Krylov basis as rows of ``(BLOCK, n)`` blocks."""

    def __init__(self, n: int, dtype: np.dtype, cols: _Columns, kernel) -> None:
        self.n, self.dtype, self.cols, self.kernel = n, dtype, cols, kernel
        self.blocks: list[np.ndarray] = []
        self.k = 0

    def _views(self, k: int) -> list[tuple[np.ndarray, int]]:
        """Each block holding one of the first ``k`` vectors, and its count."""
        return [
            (b, min(BLOCK, k - i * BLOCK))
            for i, b in enumerate(self.blocks)
            if k > i * BLOCK
        ]

    def append(self, q: np.ndarray) -> None:
        """Copy ``q`` in as the next basis vector."""
        if self.k % BLOCK == 0:
            self.blocks.append(np.empty((BLOCK, self.n), self.dtype))
        row = self.blocks[-1][self.k % BLOCK]

        def copy(s: slice) -> None:
            row[s] = q[s]

        self.cols.map(copy)
        self.k += 1

    def norm(self, q: np.ndarray) -> float:
        """The 2-norm of ``q``."""

        def squares(s: slice) -> float:
            v = q[s].view(np.float64)
            return float(np.einsum("i,i->", v, v))

        return sqrt(sum(self.cols.map(squares)))

    def scale(self, q: np.ndarray, alpha: float) -> None:
        """``q *= alpha``."""

        def multiply(s: slice) -> None:
            q[s] *= alpha

        self.cols.map(multiply)

    def _coefficients(self, q: np.ndarray) -> np.ndarray:
        views = self._views(self.k)

        def part(s: slice) -> np.ndarray:
            return np.concatenate(
                [self.kernel.coefficients(b, r, s, q) for b, r in views]
            )

        return sum(self.cols.map(part))

    def add(self, q: np.ndarray, y: np.ndarray, k: int) -> None:
        """``q += Q[:k] y``."""
        views = self._views(k)
        y = np.asarray(y, self.dtype)

        def update(s: slice) -> None:
            for i, (b, r) in enumerate(views):
                self.kernel.add(b, r, s, y[i * BLOCK : i * BLOCK + r], q)

        self.cols.map(update)

    def orthogonalize(self, q: np.ndarray) -> np.ndarray:
        """CGS2 of ``q`` against the basis, in place; the summed coefficients."""
        h = self._coefficients(q)
        self.add(q, -h, self.k)
        h2 = self._coefficients(q)
        self.add(q, -h2, self.k)
        return h + h2


def _zeros(n: int, is_complex: bool) -> Any:
    """An NGSolve vector of ``n`` zeros."""
    out = Vector(n, is_complex)
    out[:] = 0
    return out


def _givens(v1, v2):
    """The Givens rotation ``(c, s)`` zeroing ``v2`` against ``v1``."""
    if v2 == 0:
        return 1, 0
    if v1 == 0:
        return 0, v2 / abs(v2)
    t = sqrt((v1.conjugate() * v1 + v2.conjugate() * v2).real)
    return abs(v1) / t, v1 / abs(v1) * v2.conjugate() / t


def _rotate(h, cs, sn, k: int) -> None:
    """Apply the earlier rotations to the column ``h``, then zero ``h[k+1]``."""
    for i in range(k):
        temp = cs[i] * h[i] + sn[i] * h[i + 1]
        h[i + 1] = -sn[i].conjugate() * h[i] + cs[i].conjugate() * h[i + 1]
        h[i] = temp
    cs[k], sn[k] = _givens(h[k], h[k + 1])
    h[k] = cs[k] * h[k] + sn[k] * h[k + 1]
    h[k + 1] = 0


class Gmres(GMRESSolver):
    """NGSolve's GMRES with its Arnoldi vector work on gfem's thread pool.

    The pool is a ``ThreadPoolExecutor`` of as many threads as NGSolve's
    TaskManager (``ngs.GetNumThreads()``), made for each solve: at most
    OpenBLAS's ``MAX_THREADS`` when MKL is not there to take over past it.
    It is not the TaskManager's own threads, which run the operator and the
    preconditioner.

    ``Solve``, ``residuals``, ``iterations``, ``maxiter`` and ``tol``/``atol``
    (read on every solve) are NGSolve's ``GMRESSolver``'s.
    """

    def __init__(
        self,
        mat: ngs.BaseMatrix,
        pre: ngs.BaseMatrix,
        *,
        tol: float | None = None,
        atol: float | None = None,
        maxiter: int = 100,
    ) -> None:
        super().__init__(mat, pre, tol=tol, atol=atol, maxiter=maxiter)

    def _SolveImpl(  # noqa: N802 (NGSolve's name)
        self, rhs: ngs.BaseVector, sol: ngs.BaseVector
    ) -> ngs.BaseVector:
        threads = ngs.GetNumThreads()
        lib = scipy_openblas()
        mkl = None
        if lib is None or threads > lib.max_threads:
            mkl = mkl_blas()
        if mkl is not None:
            lib = None
        elif lib is not None:
            threads = min(threads, lib.max_threads)
        blas = mkl or lib
        kernel = _Einsum() if blas is None else _Gemv(blas)
        n = len(rhs)
        with _ONE_BLAS_THREAD.held(lib), _mkl_one_thread(mkl):
            if threads < 2 or n < 2 * COLUMNS:
                return self._gmres(rhs, sol, _Columns(n, 1, None), kernel)
            with ThreadPoolExecutor(
                threads,
                thread_name_prefix="gfem-gmres",
                initializer=_serial_blas,
                initargs=(mkl,),
            ) as pool:
                return self._gmres(rhs, sol, _Columns(n, threads, pool), kernel)

    def _gmres(self, rhs, sol, cols: _Columns, kernel) -> ngs.BaseVector:
        """NGSolve 6.2.2608's iteration, on the blocked basis."""
        is_complex = rhs.is_complex
        # GMRESSolver sets these in __init__
        A, pre, m = self.mat, self.pre, self.maxiter  # ty: ignore[unresolved-attribute]
        sn, cs = _zeros(m, is_complex), _zeros(m, is_complex)
        r = rhs.CreateVector()
        tmp = rhs.CreateVector()
        tmp.data = rhs - A * sol
        r.data = pre * tmp
        ra = r.FV().NumPy()
        Q = _Basis(len(rhs), ra.dtype, cols, kernel)
        r_norm = Q.norm(ra)
        if self.CheckResidual(abs(r_norm)):
            return sol
        Q.scale(ra, 1.0 / r_norm)
        Q.append(ra)
        # the newest basis vector as an NGSolve vector, for A * v
        last, spare = r, rhs.CreateVector()
        H = []
        beta = _zeros(m + 1, is_complex)
        beta[0] = r_norm

        k = 0
        for k in range(m):
            q = spare
            tmp.data = A * last
            q.data = pre * tmp
            qa = q.FV().NumPy()
            before = Q.norm(qa)
            coefficients = Q.orthogonalize(qa)
            h = _zeros(m + 1, is_complex)
            for i in range(k + 1):
                c = coefficients[i]
                h[i] = complex(c) if is_complex else float(c.real)
            h[k + 1] = Q.norm(qa)
            # lucky breakdown: q was in the Krylov space
            broke = abs(h[k + 1]) <= BREAKDOWN * before
            if not broke:
                Q.scale(qa, 1.0 / h[k + 1].real)
                Q.append(qa)
            H.append(h)
            _rotate(h, cs, sn, k)
            beta[k + 1] = -sn[k].conjugate() * beta[k]
            beta[k] = cs[k] * beta[k]
            if self.CheckResidual(abs(beta[k + 1])) or broke:
                break
            last, spare = q, last
        # x += Q y, y from the rotated triangle
        triangle = Matrix(k + 1, k + 1, is_complex)
        for i in range(k + 1):
            triangle[:, i] = H[i][: k + 1]
        rs = Vector(k + 1, is_complex)
        rs[:] = beta[: k + 1]
        y = triangle.I * rs
        Q.add(sol.FV().NumPy(), np.array([y[i] for i in range(k + 1)]), k + 1)
        return sol
