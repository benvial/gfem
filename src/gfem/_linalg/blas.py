"""The thread count of every loaded BLAS, set from Python under ranks (ADR 0076).

No launcher sets it safely: under ``mpirun`` each rank's OpenBLAS takes every CPU
of its mask, a slowdown of 10x or more with no error. A factorization needs the
rank's threads and BDDC's coarse apply runs best at one, so gfem sets the count
itself, per phase (`held`), on each OpenBLAS copy and MKL that is loaded: setting
one copy alone would leave NumPy's at a count that can oversubscribe and break
bitwise design loops. Nothing here runs on one process.
"""

from __future__ import annotations

import collections.abc
import contextlib
import ctypes
import dataclasses
import re
from collections.abc import Callable
from pathlib import Path

__all__ = ["held", "libraries", "set_threads", "settle"]

#: Names of OpenBLAS's getter and setter across its builds: plain, SciPy's and
#: NumPy's prefixed ones, and the 64-bit index suffix.
_OPENBLAS = (
    ("openblas_get_num_threads", "openblas_set_num_threads"),
    ("openblas_get_num_threads64_", "openblas_set_num_threads64_"),
    ("scipy_openblas_get_num_threads", "scipy_openblas_set_num_threads"),
    ("scipy_openblas_get_num_threads64_", "scipy_openblas_set_num_threads64_"),
)
_MAPPED = re.compile(r"(/\S*(?:openblas|mkl_rt)\S*\.so\S*)")

_settled = [False]


@dataclasses.dataclass(frozen=True)
class Library:
    """A loaded BLAS's thread count."""

    path: str
    get: Callable[[], int]
    set: Callable[[int], object]


def _open(path: str) -> Library | None:
    """``path``'s thread count, or ``None`` when it has no known symbol."""
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    if "mkl_rt" in Path(path).name:
        pairs = (("MKL_Get_Max_Threads", "MKL_Set_Num_Threads"),)
    else:
        pairs = _OPENBLAS
    for getter, setter in pairs:
        if hasattr(lib, getter) and hasattr(lib, setter):
            get, put = getattr(lib, getter), getattr(lib, setter)
            get.argtypes, get.restype = [], ctypes.c_int
            put.argtypes, put.restype = [ctypes.c_int], None
            return Library(path, get, put)
    return None


def libraries(maps: collections.abc.Iterable[str] | None = None) -> list[Library]:
    """Each OpenBLAS copy and MKL mapped into this process now.

    Args:
        maps: The lines of ``/proc/self/maps``; ``None`` reads them. Libraries load
            at any time, so nothing here is kept between calls.
    """
    if maps is None:
        try:
            maps = Path("/proc/self/maps").read_text().splitlines()
        except OSError:
            return []
    paths = dict.fromkeys(m[1] for m in map(_MAPPED.search, maps) if m)
    return [lib for lib in map(_open, paths) if lib is not None]


def set_threads(threads: int) -> None:
    """Set every loaded BLAS to ``threads``."""
    for lib in libraries():
        lib.set(threads)


def settle(threads: int) -> None:
    """Set every loaded BLAS to the rank's ``threads``, at the first distributed solve.

    Once per process: a count the user sets later stands, except inside a phase.
    """
    if not _settled[0]:
        _settled[0] = True
        set_threads(threads)


@contextlib.contextmanager
def held(threads: int) -> collections.abc.Generator[None]:
    """Every loaded BLAS at ``threads`` inside the block, as it was after."""
    saved = [(lib, lib.get()) for lib in libraries()]
    for lib, _ in saved:
        lib.set(threads)
    try:
        yield
    finally:
        for lib, count in saved:
            lib.set(count)
