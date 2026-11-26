"""The solve cache: a finished result stored under its solve key, read back.

The **solve key** of a solve is the sha256 (64 hex characters) of everything
the result depends on: the model, the problem, the solver options that change
the numerical method or its stopping rule, and the environment that computes
it (gfem's version and code, NGSolve's version, and PETSc's or SLEPc's when
they solve). Two calls with the same key give the same result to gfem's
documented precision.

The key hashes, in this order:

1. A compact JSON document ``{"key_version", "schema_version", "env",
   "job"}`` (``separators=(",", ":")``, ASCII, no NaN, mappings in insertion
   order). ``job`` is the `gfem._codec` encoding of the job under
   ``Context(key=True)``, which writes a mesh by its in-memory topology
   (`gfem.mesh._mesh_sha256`) instead of its file; the encoder's
   ``numpy`` flags are dropped and a scalar :math:`-0` is written as
   :math:`0`.
2. Every array of the encoder's table, sorted by leaf path, as
   `gfem.mesh._hash_array` feeds it.

The options that say only where and how fast a solve runs (``threads``,
``frequency_workers``, ``retain_memory``, ``device``) are reset to their
defaults first, so a result computed on one machine serves another.

A `Store` is the directory the user names. It holds a marker
(``solve-cache.json``: format, layout, ``max_bytes``), ``CACHEDIR.TAG``, a
``.gitignore``, a ``.tmp/`` area, and one **cache entry** per key: a
directory named by the key holding ``entry.json`` and, when the result has
arrays, ``arrays.npz``. An entry's fields refer to the model's mesh by its
topology digest and are read back onto the caller's ``model.mesh``. Entries
are built in ``.tmp/`` and published by one directory rename, so a reader
sees a whole entry or none; the least recently used leave once the entries
pass ``max_bytes``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import errno
import functools
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import time
import warnings
import zipfile
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Self

import numpy as np

import gfem
from gfem import _codec, _mpi, errors
from gfem._checks import is_traced
from gfem.errors import SolveCacheWarning
from gfem.job import Job
from gfem.mesh import _hash_array
from gfem.options import SolverOptions

#: The version of the key's rules. Bump it when anything the key hashes, or
#: how, changes: entries of an older version then miss.
KEY_VERSION = 1

#: The options reset before keying: they change where and how fast a solve
#: runs, not its result beyond documented precision.
_PLACEMENT = {
    "threads": None,
    "frequency_workers": 1,
    "retain_memory": None,
    "device": "cpu",
}


def solve_key(
    model: Any, problem: Any, options: SolverOptions
) -> tuple[str, dict[str, str | None]]:
    """The solve key of a solve, and the environment it includes.

    Args:
        model: The model.
        problem: The problem.
        options: The solver options.

    Returns:
        ``(key, env)``: the sha256 as 64 hex characters, and the environment
        record (package versions and gfem's code digest) that entered it.

    Raises:
        gfem._codec.UnencodableError: An input of a type gfem cannot write.
    """
    env = _env(options)
    job = Job(model, problem, dataclasses.replace(options, **_PLACEMENT))
    encoder = _codec.Encoder(_codec.Context(Path("/"), key=True))
    doc = {
        "key_version": KEY_VERSION,
        "schema_version": _codec.SCHEMA_VERSION,
        "env": env,
        "job": _canonical(encoder.encode(job)),
    }
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    h = hashlib.sha256(text.encode("ascii"))
    for path in sorted(encoder.arrays):
        _hash_array(h, path, encoder.arrays[path])
    return h.hexdigest(), env


def _canonical(node: Any) -> Any:
    """The encoded JSON without ``numpy`` flags, :math:`-0` written as :math:`0`."""
    if isinstance(node, list):
        return [_canonical(v) for v in node]
    if isinstance(node, float):
        return node + 0.0
    if isinstance(node, dict):
        array = "$array" in node
        return {
            k: _canonical(v) for k, v in node.items() if not (array and k == "numpy")
        }
    return node


def _env(options: SolverOptions) -> dict[str, str | None]:
    env: dict[str, str | None] = {
        "gfem": gfem.__version__,
        "gfem_code": _gfem_code(),
        "ngsolve": _version("ngsolve"),
    }
    if options.linear_solver == "petsc":
        env["petsc4py"] = _version("petsc4py")
    if options.eigen_solver == "slepc":
        env["slepc4py"] = _version("slepc4py")
    return env


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


@functools.cache
def _gfem_code() -> str:
    """The sha256 of the installed gfem's ``.py`` files: paths, then bytes."""
    root = Path(gfem.__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        name = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        h.update(len(name).to_bytes(8, "little"))
        h.update(name)
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


# --- the store ------------------------------------------------------------------

#: The marker's format name, and the layout this gfem reads and writes.
CACHE_FORMAT = "gfem solve cache"
LAYOUT = 1
#: An entry's format name: not a result bundle's, so `gfem.load_result`
#: refuses it.
ENTRY_FORMAT = "gfem solve cache entry"
#: The size cap a new cache gets: 10 GiB.
DEFAULT_MAX_BYTES = 10 * 1024**3

MARKER = "solve-cache.json"
ENTRY_JSON = "entry.json"
ENTRY_NPZ = "arrays.npz"
TMP = ".tmp"
_CACHEDIR_TAG = (
    "Signature: 8a477f597d28d172789f06886806bc55\n"
    "# This file is a cache directory tag created by gfem's solve cache.\n"
    "# For information about cache directory tags, see:\n"
    "#\thttps://bford.info/cachedir/\n"
)
#: An entry's name: a solve key.
_KEY = re.compile(r"[0-9a-f]{64}\Z")
#: ``.tmp/`` children older than this (crashed writers) are removed.
_TMP_AGE_S = 24 * 3600
#: What reading a damaged entry raises; ``JSONDecodeError`` is a ``ValueError``,
#: ``TypeError`` a hand edit that breaks a constructor.
_READ_ERRORS = (OSError, ValueError, EOFError, zipfile.BadZipFile, KeyError, TypeError)

_log = logging.getLogger("gfem.solve_cache")

#: A decoded entry: the result, the solving run's record, the recorded warnings.
type Entry = tuple[Any, dict[str, Any], tuple[tuple[type[Warning], str], ...]]


class Store:
    """A solve cache directory: entries read, written and evicted.

    A path that does not exist, or an empty directory, becomes a cache. A
    cache of another layout warns once and is left alone: `read` misses and
    `write` stores nothing (`usable` is False).

    Every method that warns takes ``stacklevel``, as `warnings.warn` does
    for a warning raised in the method's caller (1: the calling line).

    Attributes:
        root: The directory.
        usable: False for a cache of another layout.
        max_bytes: The cap on the entries' total size, from the marker.

    Raises:
        ValueError: A non-empty directory without a readable marker, or a
            marker whose ``max_bytes`` is not a whole number of bytes.
        OSError: The directory cannot be created or initialized.
    """

    def __init__(self, directory: str | os.PathLike, *, stacklevel: int = 1) -> None:
        """Open, or create, the cache at ``directory``."""
        self.root = Path(directory)
        self.usable = True
        self.max_bytes = DEFAULT_MAX_BYTES
        if not self.root.is_dir() or not _occupied(self.root):
            self.root.mkdir(parents=True, exist_ok=True)
            self._initialize()
        marker = self._marker()
        if marker["layout"] != LAYOUT:
            self.usable = False
            warnings.warn(
                f"{self.root} is a solve cache of another layout "
                f"({marker['layout']}, this gfem writes {LAYOUT}), so this call "
                "neither reads nor stores; the result is correct. Name another "
                "directory, or delete this one.",
                SolveCacheWarning,
                stacklevel=stacklevel + 1,
            )
            return
        self.max_bytes = marker["max_bytes"]

    def _initialize(self) -> None:
        """Write the marker (atomically), then the tag, ``.gitignore`` and ``.tmp/``."""
        text = json.dumps(
            {"format": CACHE_FORMAT, "layout": LAYOUT, "max_bytes": DEFAULT_MAX_BYTES}
        )
        fd, tmp = tempfile.mkstemp(prefix=MARKER + ".", dir=self.root)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text + "\n")
            Path(tmp).replace(self.root / MARKER)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        (self.root / "CACHEDIR.TAG").write_text(_CACHEDIR_TAG)
        (self.root / ".gitignore").write_text("*\n")
        (self.root / TMP).mkdir(exist_ok=True)

    def _marker(self) -> dict[str, Any]:
        path = self.root / MARKER
        try:
            marker = json.loads(path.read_text())
        except (OSError, ValueError):
            marker = None
        if (
            not isinstance(marker, dict)
            or marker.get("format") != CACHE_FORMAT
            or not _whole(marker.get("layout"))
        ):
            raise ValueError(
                f"{self.root} is not empty and is not a gfem solve cache (no "
                f"readable {MARKER}); gfem writes a cache only into a new or "
                "empty directory. Name another directory for cache=."
            )
        if marker["layout"] == LAYOUT and not (
            _whole(marker.get("max_bytes")) and marker["max_bytes"] >= 0
        ):
            raise ValueError(
                f"{path} has max_bytes {marker.get('max_bytes')!r}: give the "
                "solve cache's size cap as a whole number of bytes."
            )
        return marker

    def read(self, key: str, model: Any, *, stacklevel: int = 1) -> Entry | None:
        """The entry of ``key``, its fields on ``model.mesh``; None on a miss.

        A hit marks the entry as used now. An entry that cannot be read (a
        damaged file, a key or mesh that does not match) is a miss with a
        `gfem.SolveCacheWarning`, and is removed if it can be.

        Args:
            key: The solve key.
            model: The model of the call: the entry's fields come back on
                its mesh object.
            stacklevel: Where the warning points, as for `warnings.warn`.

        Returns:
            ``(result, run, recorded)``: the result, the solving run's record
            (``device``, ``options``, ``ranks``, ``wall_time_s``) and the
            warnings recorded while solving, as ``(category, message)``.
        """
        found = self.read_document(key, model, stacklevel=stacklevel + 1)
        return None if found is None else found[0]

    def read_document(
        self, key: str, model: Any, *, stacklevel: int = 1
    ) -> tuple[Entry, dict[str, Any], dict[str, np.ndarray]] | None:
        """`read`, with the entry's document and arrays as stored.

        Other ranks decode those with `decode_entry`.

        Returns:
            ``((result, run, recorded), document, arrays)``, or None on a miss.
        """
        if not self.usable:
            return None
        entry = self.root / key
        if not entry.is_dir():
            return None
        try:
            found = _load(entry, key, model)
        except _READ_ERRORS as err:
            if not entry.exists():
                return None  # evicted while being read
            if _discard(self.root, entry):
                fate = (
                    ", so it was removed and this call solved again; the new "
                    "result is stored"
                )
            else:
                fate = (
                    " and could not be removed, so this call solved again; "
                    "delete that directory"
                )
            warnings.warn(
                f"The solve cache entry {entry} could not be read "
                f"({type(err).__name__}: {err}){fate}. Entries are written "
                "whole, so something outside gfem changed it (a full disk, a "
                "sync tool, a hand edit).",
                SolveCacheWarning,
                stacklevel=stacklevel + 1,
            )
            return None
        with contextlib.suppress(OSError):  # a read-only cache still serves hits
            os.utime(entry / ENTRY_JSON)
        return found

    def write(  # noqa: PLR0917
        self,
        key: str,
        model: Any,
        env: dict[str, str | None],
        result: Any,
        run: dict[str, Any],
        recorded: Sequence[tuple[type[Warning], str]],
        *,
        stacklevel: int = 1,
    ) -> Path | str:
        """Store ``result`` under ``key``, then evict down to ``max_bytes``.

        A result that cannot be stored (a failed write, or larger than
        ``max_bytes``) warns with `gfem.SolveCacheWarning` and is not stored;
        nothing is raised. When another process publishes the same key first,
        its entry stays.

        Args:
            key: The solve key.
            model: The model solved: fields on its mesh refer to it.
            env: The environment that entered the key.
            result: The result.
            run: The solving run's record: ``device``, ``options``,
                ``ranks``, ``wall_time_s``.
            recorded: The warnings to replay on a hit, as
                ``(category, message)``; categories are `gfem.errors` classes.
            stacklevel: Where the warning points, as for `warnings.warn`.

        Returns:
            The entry's directory, or why nothing was stored.
        """
        if not self.usable:
            return "a solve cache of another layout"
        encoder = _codec.Encoder(_codec.Context(self.root / key, model_mesh=model.mesh))
        try:
            doc: dict[str, Any] = {
                "format": ENTRY_FORMAT,
                "layout": LAYOUT,
                "key": key,
                "env": env,
                "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "bytes": 0,
                "arrays": None,
                "run": encoder.encode(run, ".run"),
                "warnings": [
                    {"category": c.__name__, "message": m} for c, m in recorded
                ],
                "result": encoder.encode(result),
            }
        except _codec.UnencodableError as err:
            warnings.warn(
                f"This solve's result could not be stored in the solve cache "
                f"{self.root} ({err}); the result is correct.",
                SolveCacheWarning,
                stacklevel=stacklevel + 1,
            )
            return f"the result cannot be written: {err}"
        size = sum(a.nbytes for a in encoder.arrays.values())
        target = self.root / key
        if size <= self.max_bytes:
            try:
                size = _publish(self.root, target, doc, encoder.arrays, self.max_bytes)
            except OSError as err:
                warnings.warn(
                    f"This solve's result could not be stored in the solve cache "
                    f"{self.root} ({type(err).__name__}: {err}); the result is "
                    "correct. Check that the directory is writable and has space.",
                    SolveCacheWarning,
                    stacklevel=stacklevel + 1,
                )
                return f"the write failed: {err}"
        if size > self.max_bytes:
            warnings.warn(
                f"This solve's result ({_size(size)}) is larger than the solve "
                f"cache's limit ({_size(self.max_bytes)}) and was not stored; the "
                f"result is correct. Raise max_bytes in {self.root / MARKER} to "
                "keep results this large.",
                SolveCacheWarning,
                stacklevel=stacklevel + 1,
            )
            return "larger than the cache's max_bytes"
        self._evict()
        _sweep(self.root / TMP)
        return target

    def _evict(self) -> None:
        """Remove the least recently used entries until within ``max_bytes``."""
        entries = _entries(self.root)
        total = sum(size for _, size, _ in entries)
        evicted = freed = 0
        for _, size, path in sorted(entries, key=lambda e: e[0]):
            if total <= self.max_bytes:
                break
            if _discard(self.root, path):
                total -= size
                evicted += 1
                freed += size
        if evicted:
            _log.info("evicted %d entries, %s", evicted, _size(freed))


def _publish(
    root: Path,
    target: Path,
    doc: dict[str, Any],
    arrays: dict[str, np.ndarray],
    max_bytes: int,
) -> int:
    """Build the entry in ``.tmp/`` and rename it to ``target``; its size.

    An entry larger than ``max_bytes`` is built, measured and dropped. A
    rename that finds ``target`` taken leaves the other writer's entry.
    """
    (root / TMP).mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{target.name}.", dir=root / TMP))
    try:
        size = _write_entry(tmp, doc, arrays)
        if size <= max_bytes:
            try:
                tmp.rename(target)
            except OSError as err:
                if err.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return size


def _entries(root: Path) -> list[tuple[float, int, Path]]:
    """Every entry as ``(last used, bytes, directory)``.

    A damaged entry (no readable ``entry.json``, or no whole ``bytes`` in it)
    counts by its files' size on disk, last used when its directory last
    changed, and is evicted like the others.
    """
    entries = []
    try:
        scan = list(os.scandir(root))
    except OSError:
        return []
    for item in scan:
        if not _KEY.match(item.name) or not item.is_dir(follow_symlinks=False):
            continue
        path = Path(item.path)
        try:
            used = (path / ENTRY_JSON).stat().st_mtime
            size = json.loads((path / ENTRY_JSON).read_text())["bytes"]
            if not _whole(size):
                raise TypeError(size)  # noqa: TRY301
        except (OSError, ValueError, KeyError, TypeError):
            try:  # damaged
                used, size = path.stat().st_mtime, _disk_size(path)
            except OSError:
                continue  # removed meanwhile
        entries.append((used, size, path))
    return entries


def _disk_size(directory: Path) -> int:
    """The total size of the files under ``directory``."""
    return sum(p.lstat().st_size for p in directory.rglob("*") if not p.is_dir())


def _sweep(tmp: Path) -> None:
    """Remove what crashed writers left in ``.tmp/`` more than a day ago."""
    try:
        children = list(tmp.iterdir())
    except OSError:
        return
    now = time.time()
    for child in children:
        try:
            if now - child.lstat().st_mtime < _TMP_AGE_S:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink()
        except OSError:
            pass


def _occupied(root: Path) -> bool:
    """Whether ``root`` holds anything besides a marker being written."""
    return any(not p.name.startswith(MARKER + ".") for p in root.iterdir())


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _load(
    entry: Path, key: str, model: Any
) -> tuple[Entry, dict[str, Any], dict[str, np.ndarray]]:
    """Read and decode one entry, raising one of `_READ_ERRORS` if it is damaged.

    Returns:
        The decoded entry, and the document and arrays it was decoded from.
    """
    doc = json.loads((entry / ENTRY_JSON).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("format") != ENTRY_FORMAT:
        raise ValueError(f"{entry / ENTRY_JSON} is not a solve cache entry.")
    if doc.get("layout") != LAYOUT:
        raise ValueError(f"the entry has layout {doc.get('layout')!r}, not {LAYOUT}")
    if doc.get("key") != key:
        raise ValueError(f"the entry holds key {doc.get('key')!r}, not its name")
    arrays: dict[str, np.ndarray] = {}
    if doc["arrays"]:
        # np.load leaves its file open when a damaged zip raises, and Windows
        # cannot then move the entry away: open and close the file here
        with (
            (entry / ENTRY_NPZ).open("rb") as f,
            np.load(f, allow_pickle=False) as npz,
        ):
            arrays = {k: npz[k] for k in npz.files}
    return decode_entry(entry, doc, arrays, model), doc, arrays


def decode_entry(
    entry: Path, doc: dict[str, Any], arrays: dict[str, np.ndarray], model: Any
) -> Entry:
    """An entry's ``(result, run, recorded)``, its fields on ``model.mesh``.

    Args:
        entry: The entry's directory.
        doc: Its ``entry.json`` document.
        arrays: Its arrays.
        model: The model of the call.
    """
    context = _codec.Context(entry, model_mesh=model.mesh)
    decoder = _codec.Decoder(context, arrays, path=entry / ENTRY_JSON)
    run = decoder.decode(doc["run"])
    if not isinstance(run, dict):
        raise ValueError("the entry's run is not a record")  # noqa: TRY004
    recorded = tuple(
        (_category(w["category"]), str(w["message"])) for w in doc["warnings"]
    )
    return decoder.decode(doc["result"]), run, recorded


def _category(name: Any) -> type[Warning]:
    """A gfem warning class by its name."""
    cls = getattr(errors, str(name), None)
    if not (isinstance(cls, type) and issubclass(cls, Warning)):
        raise ValueError(f"the entry records an unknown warning class {name!r}")  # noqa: TRY004
    return cls


def _write_entry(tmp: Path, doc: dict[str, Any], arrays: dict[str, np.ndarray]) -> int:
    """Write ``entry.json`` (and ``arrays.npz``) into ``tmp``; their total size.

    ``bytes`` in ``entry.json`` is that total, the document's own size included.
    """
    size = 0
    if arrays:
        with (tmp / ENTRY_NPZ).open("wb") as f:
            np.savez(f, allow_pickle=False, **arrays)
        doc["arrays"] = ENTRY_NPZ
        size = (tmp / ENTRY_NPZ).stat().st_size
    total = size
    while True:  # the size's digits change the size: at most a few rounds
        doc["bytes"] = total
        data = (json.dumps(doc, indent=1, allow_nan=False) + "\n").encode()
        if size + len(data) == total:
            break
        total = size + len(data)
    (tmp / ENTRY_JSON).write_bytes(data)
    return total


def _discard(root: Path, entry: Path) -> bool:
    """Remove an entry: renamed into ``.tmp/`` first, so no reader sees half of it."""
    dead = root / TMP / f"{entry.name}.{secrets.token_hex(8)}.old"
    try:
        (root / TMP).mkdir(exist_ok=True)
        entry.rename(dead)
    except OSError:
        return False
    shutil.rmtree(dead, ignore_errors=True)
    return True


def _size(n: int) -> str:
    """A byte count for people, in binary units: ``"4.5 MiB"``."""
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


# --- one call -------------------------------------------------------------------

#: The gfem warnings a solve raises about its inputs or its result: recorded
#: on a miss, replayed on a hit. Others (a device's memory, a
#: third party's deprecation) are facts of one run and are not stored.
REPLAYED = (
    errors.MeshQualityWarning,
    errors.LowFrequencyWarning,
    errors.SweepNotConvergedWarning,
    errors.SweepRankDeficientWarning,
    errors.UnrequestedOrderWarning,
    errors.FloquetOrderCapWarning,
    errors.ObliqueAbsorberWarning,
    errors.SkippedPortModeWarning,
    errors.UnrequestedPortModeWarning,
    errors.PortModeTrackingWarning,
    errors.ThinLayerWarning,
    errors.LayerClearanceWarning,
    errors.FittedRangeWarning,
)


def _incident_function(problem: Any) -> str | None:
    """The first user incident function among the problem's sources, named; or None."""
    from gfem._sources import IncidentField, qualified

    for source in getattr(problem, "sources", ()):
        if isinstance(source, IncidentField):
            return f"{source.name!r} (function {qualified(source.function)})"
    return None


class SolveCache:
    """One call's use of the solve cache named by ``cache=``.

    ``gfem.solve`` (through ``_solve``) drives it in three steps: `session`,
    a context around the whole call that decides a bypass and catches the
    call's warnings; `lookup` after every check; `store` after the forward.
    Leaving the session shows the warnings at the caller of gfem: the
    replayed classes (`REPLAYED`) and `gfem.SolveCacheWarning` at its line,
    as a hit shows its stored ones, the others where they were raised, so a
    miss and a hit print alike. gfem runs one solve per process at a time,
    so catching warnings process-wide is safe.

    What happened is left in `status`, `key` and `solved`; `report` writes
    them as a run record's ``cache`` item.

    Attributes:
        directory: The directory, as given.
        status: ``"hit"``, ``"miss"`` or ``"bypass"``; None before the lookup.
        key: The solve key; None on a bypass.
        solved: On a hit, the solving run's ``device``, ``ranks`` and
            ``wall_time_s``; otherwise None.
    """

    def __init__(self, directory: str | os.PathLike) -> None:
        """Name the directory; nothing is read or created yet.

        Raises:
            TypeError: ``directory`` is not a path.
        """
        if not isinstance(directory, (str, os.PathLike)):
            raise TypeError(
                "cache must be a directory path (str or os.PathLike), got "
                f"{type(directory).__name__}."
            )
        self.directory = directory
        self.status: str | None = None
        self.key: str | None = None
        self.solved: dict[str, Any] | None = None
        self._store: Store | None = None
        self._env: dict[str, str | None] = {}
        self._replay: tuple[tuple[type[Warning], str], ...] = ()
        self._catch: warnings.catch_warnings | None = None
        self._caught: list[warnings.WarningMessage] = []
        self._start = 0.0
        self._comm: Any = None
        self._kept_on = 0
        self._doc: dict[str, Any] | None = None
        self._arrays: dict[str, np.ndarray] = {}

    def report(self) -> dict[str, Any]:
        """The ``cache`` item of a run record: status, key, solving run on a hit."""
        item: dict[str, Any] = {"status": self.status, "key": self.key}
        if self.solved is not None:
            item["solved"] = self.solved
        return item

    def session(
        self, model: Any, problem: Any, *, comm: Any, adjoint: bool
    ) -> SolveCache:
        """The context of one call, with its bypass decided.

        Args:
            model: The model.
            problem: The problem.
            comm: The resolved communicator; None on one process. Above one
                rank, rank 0 alone keys, reads and writes, and a solve that
                keeps fields bypasses.
            adjoint: Whether the call solves adjoints with the forward
                (``refine_mesh``).
        """
        self._start = time.perf_counter()
        self._comm = comm
        if is_traced((model, problem)):
            self._bypass("traced inputs")
        elif adjoint:
            self._bypass("adjoint loads (refine_mesh)")
        elif comm is not None and getattr(problem, "keep_fields", False):
            self._kept_on = comm.size
            self._bypass(f"fields kept on {comm.size} ranks")
        return self

    def _bypass(self, reason: str) -> None:
        self.status = "bypass"
        _log.debug("bypass: %s", reason)

    def lookup(self, model: Any, problem: Any, options: SolverOptions) -> Any:
        """The stored result of this call, or None to solve.

        Keys the call, opens the store and reads its entry. An input the key
        cannot encode (decided before the directory is touched), or a cache
        of another layout, makes the call a bypass with a
        `gfem.SolveCacheWarning`. Above one rank, every rank calls
        this: rank 0 does the work and every rank decodes what it found.

        Raises:
            ValueError: A non-empty directory that is not a solve cache.
            OSError: The directory cannot be created.
        """
        if self._kept_on:
            _mpi.warn(
                self._comm,
                f"cache={os.fspath(self.directory)!r} was given, but this solve "
                f"keeps fields on {self._kept_on} ranks, which the solve cache "
                "does not store, so it was neither read from nor stored in it. "
                "The result is correct. For a result readable later, fields "
                "included, use Job.run(out=); to cache a solve with fields, run "
                "it on one process (comm=MPI.COMM_SELF).",
                SolveCacheWarning,
                stacklevel=_mpi.outside_gfem(),
            )
        if self.status == "bypass":
            return None
        if self._comm is None:
            return self._lookup(model, problem, options)
        return self._lookup_on_root(model, problem, options)

    def _lookup_on_root(self, model: Any, problem: Any, options: SolverOptions) -> Any:
        """`_lookup` on rank 0; its outcome, document and arrays sent to all."""
        comm = self._comm
        found = None

        def compute() -> dict[str, np.ndarray]:
            nonlocal found
            found = self._lookup(model, problem, options)
            head = {"status": self.status, "key": self.key, "document": self._doc}
            text = np.frombuffer(json.dumps(head).encode(), dtype=np.uint8)
            return {_HEAD: text, **{_ARRAY + k: a for k, a in self._arrays.items()}}

        shared = _mpi.on_root(comm, compute, results=(ValueError, OSError))
        self._doc, self._arrays = None, {}
        if comm.rank == 0:
            return found
        head = json.loads(shared.pop(_HEAD).tobytes())
        self.status, self.key = head["status"], head["key"]
        if self.status != "hit":
            return None
        arrays = {k.removeprefix(_ARRAY): a for k, a in shared.items()}
        with _mpi.section(comm, agreed=()):
            entry = Path(self.directory) / str(self.key)
            return self._hit(decode_entry(entry, head["document"], arrays, model))

    def _lookup(self, model: Any, problem: Any, options: SolverOptions) -> Any:
        stacklevel = _mpi.outside_gfem()
        function = _incident_function(problem)
        if function is not None:
            # a function's code is not in the key, and its name says nothing of it
            self._bypass(f"the IncidentField {function}")
            warnings.warn(
                f"cache={os.fspath(self.directory)!r} was given, but this solve "
                f"was neither read from nor stored in it: the IncidentField "
                f"{function} cannot be keyed, since the key holds data and not "
                "code. The result is correct. Use gfem.PlaneWave or "
                "gfem.GaussianBeam, or drop cache= for this call.",
                SolveCacheWarning,
                stacklevel=stacklevel,
            )
            return None
        try:  # first: an input the key cannot encode leaves the directory alone
            key, self._env = solve_key(model, problem, options)
        except _codec.UnencodableError as err:
            self._bypass(f"cannot key on {err.path}")
            warnings.warn(
                f"cache={os.fspath(self.directory)!r} was given, but this solve "
                "was neither read from nor stored in it: the solve cache cannot "
                f"key on {err.path.lstrip('.') or 'the input'} ({err}). The "
                "result is correct. Use the gfem class instead of a subclass, "
                "or drop cache= for this call.",
                SolveCacheWarning,
                stacklevel=stacklevel,
            )
            return None
        store = Store(self.directory, stacklevel=stacklevel)
        if not store.usable:
            self._bypass("a solve cache of another layout")
            return None
        self.key, self._store = key, store
        found = store.read_document(key, model, stacklevel=stacklevel)
        if found is None:
            self.status = "miss"
            return None
        entry, doc, arrays = found
        if self._comm is not None:
            self._doc, self._arrays = doc, arrays
        _log.info("hit %s (%s)", self.key[:12], store.root / self.key)
        return self._hit(entry)

    def _hit(self, entry: Entry) -> Any:
        result, run, self._replay = entry
        self.status = "hit"
        self.solved = {k: run.get(k) for k in ("device", "ranks", "wall_time_s")}
        return result

    def store(self, model: Any, result: Any, options: SolverOptions) -> None:
        """Store a missed call's result, with the replayed warnings caught.

        Above one rank, every rank calls this and rank 0 writes.
        """
        if self.status != "miss" or self.key is None:
            return
        if self._comm is None:
            self._write(model, result, options)
            return

        def compute() -> dict[str, np.ndarray]:
            self._write(model, result, options)
            return {}

        _mpi.on_root(self._comm, compute)

    def _write(self, model: Any, result: Any, options: SolverOptions) -> None:
        if self._store is None or self.key is None:
            return
        run = {
            "device": options.device,
            "options": options,
            "ranks": 1 if self._comm is None else self._comm.size,
            "wall_time_s": time.perf_counter() - self._start,
        }
        recorded = tuple(
            (w.category, str(w.message))
            for w in self._caught
            if issubclass(w.category, REPLAYED)
        )
        where = self._store.write(
            self.key,
            model,
            self._env,
            result,
            run,
            recorded,
            stacklevel=_mpi.outside_gfem(),
        )
        if isinstance(where, Path):
            _log.info("miss %s: solved and stored (%s)", self.key[:12], where)
        else:
            _log.info("miss %s: nothing stored (%s)", self.key[:12], where)

    def __enter__(self) -> Self:
        if self.status != "bypass":
            self._catch = warnings.catch_warnings(record=True)
            self._caught = self._catch.__enter__()
            warnings.simplefilter("always")
        return self

    def __exit__(self, kind: object, error: object, tb: object) -> None:
        """Show the call's warnings at the caller of gfem.

        Above one rank, a call that returns shows rank 0's warnings once, as
        `gfem._mpi.warn` does: rank 0 shows them, the other ranks record them,
        so an ``error`` filter raises on every rank.
        """
        if self._catch is None:
            return
        self._catch.__exit__(None, None, None)
        self._catch = None
        if error is not None and self.status == "miss" and self.key is not None:
            _log.info(
                "miss %s: nothing stored (the solve raised %s)",
                self.key[:12],
                type(error).__name__,
            )
        hit = self.status == "hit"
        stored: list[_Shown] = [(m, c, None, 0, None) for c, m in self._replay]
        mine: list[_Shown] = [
            (str(w.message), w.category, w.filename, w.lineno, w.source)
            for w in self._caught
            # on a hit, raised by the checks before the lookup: stored too
            if not (hit and issubclass(w.category, REPLAYED))
        ]
        comm = self._comm
        if comm is None or error is not None:
            _show(stored + mine)
            return
        sent = [(m, c, f, n, None) for m, c, f, n, _ in mine]
        root = comm.bcast(sent if comm.rank == 0 else None, root=0)
        try:
            if comm.rank == 0:
                _show(stored + mine)
            else:
                _show(mine)  # this rank's own, shown as without a cache
                _show(stored + root, quiet=True)
        except Warning as raised:
            raise _mpi.agreed(raised) from None


#: A caught warning: message, category, file, line (None: the caller's), source.
type _Shown = tuple[str, type[Warning], str | None, int, Any]
#: The names of a hit's broadcast: the outcome, and the entry's arrays.
_HEAD = "solve cache outcome"
_ARRAY = "array "


def _show(items: Sequence[_Shown], *, quiet: bool = False) -> None:
    """Warn ``items``: the replayed classes at the caller of gfem, others in place.

    ``quiet`` records instead of showing; an ``error`` filter still raises.
    """
    stacklevel = _mpi.outside_gfem()
    with warnings.catch_warnings(record=True) if quiet else contextlib.nullcontext():
        for message, category, filename, lineno, source in items:
            if filename is None or issubclass(category, (*REPLAYED, SolveCacheWarning)):
                warnings.warn(message, category, stacklevel=stacklevel)
            else:
                warnings.warn_explicit(
                    message, category, filename, lineno, source=source
                )
