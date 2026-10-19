"""The engine's two caches.

**Discretizations** (`discretization`): the NGSolve mesh of a `gfem.Mesh` at
given node coordinates, and its finite-element spaces keyed by
``(physics domain, space, order)``. Optimization iterations and batch
members on one mesh share them. Bounded by count.

**Solve entries** (`ENTRIES`): what a differentiated forward solve leaves for
its backward pass, a solver `Handle` (factors or preconditioner) and the
forward fields, under an integer key that travels with the JAX residuals.
Bounded in bytes by ``SolverOptions.retain_memory``, least recently used out
first; an entry is never freed by being read, so repeated pullbacks work.
A missing entry is not an error: the backward pass recomputes it from the
design state in its residuals (time, never correctness).
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import itertools
import os
from collections.abc import Callable, Hashable
from typing import Any

import numpy as np

from gfem._engine.mesh import EngineMesh, to_ngsolve
from gfem.mesh import Mesh

__all__ = [
    "ENTRIES",
    "Discretization",
    "Entry",
    "EntryCache",
    "default_retain_memory",
    "discretization",
]

_MAX_DISCRETIZATIONS = 4


@dataclasses.dataclass(eq=False)
class Discretization:
    """The NGSolve mesh of one mesh topology at one set of node coordinates.

    Attributes:
        built: The NGSolve mesh and its element maps.
        spaces: Finite-element spaces by ``(physics domain, space, order)``.
    """

    built: EngineMesh
    topology: Any
    spaces: dict[Hashable, Any] = dataclasses.field(default_factory=dict)

    def space(self, key: Hashable, build: Callable[[], Any]) -> Any:
        """The cached space ``key``, built by ``build()`` on first use."""
        if key not in self.spaces:
            self.spaces[key] = build()
        return self.spaces[key]


_DISCRETIZATIONS: collections.OrderedDict[tuple, Discretization] = (
    collections.OrderedDict()
)


def _digest(nodes: np.ndarray) -> str:
    return hashlib.blake2b(
        np.ascontiguousarray(nodes).tobytes(), digest_size=16
    ).hexdigest()


def discretization(mesh: Mesh) -> Discretization:
    """The cached discretization of ``mesh`` at its current node coordinates."""
    topology = mesh._topology  # noqa: SLF001
    nodes = np.asarray(mesh.nodes, dtype=float)
    key = (id(topology), _digest(nodes))
    found = _DISCRETIZATIONS.get(key)
    if found is not None and found.topology is topology:
        _DISCRETIZATIONS.move_to_end(key)
        return found
    entry = Discretization(built=to_ngsolve(mesh), topology=topology)
    _DISCRETIZATIONS[key] = entry
    while len(_DISCRETIZATIONS) > _MAX_DISCRETIZATIONS:
        _DISCRETIZATIONS.popitem(last=False)
    return entry


# --- solve entries ---------------------------------------------------------------


@dataclasses.dataclass(eq=False)
class Entry:
    """What a forward solve keeps for its backward pass.

    Attributes:
        physics: The physics domain the solve ran on (``"em"``, ...).
        handle: The prepared solver, or ``None`` if the backward needs none.
        fields: Forward fields as NumPy arrays (DOF vectors), by name.
    """

    physics: str
    handle: Any = None
    fields: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        """Host memory held, in bytes."""
        held = sum(int(f.nbytes) for f in self.fields.values())
        return held + (int(self.handle.nbytes) if self.handle is not None else 0)

    def free(self) -> None:
        """Release the solver handle and the fields."""
        if self.handle is not None:
            self.handle.free()
            self.handle = None
        self.fields = {}


def default_retain_memory() -> int:
    """Half of the physical memory, in bytes (8 GiB if it cannot be read)."""
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        total = 16 * 2**30
    return int(total) // 2


class EntryCache:
    """Solve entries by integer key, least recently used evicted over budget."""

    def __init__(self) -> None:
        """An empty cache."""
        self._entries: collections.OrderedDict[int, Entry] = collections.OrderedDict()
        self._keys = itertools.count()

    def put(self, entry: Entry, budget: int) -> int:
        """Store ``entry`` and return its key; evict down to ``budget`` bytes.

        An entry larger than the budget alone is freed at once and evicts
        nothing (with ``budget=0`` nothing is kept); its key reads as
        evicted, so the backward pass recomputes it.
        """
        key = next(self._keys)
        if entry.nbytes > budget:
            entry.free()
            return key
        self._entries[key] = entry
        while self._entries and self.nbytes > budget:
            _, old = self._entries.popitem(last=False)
            old.free()
        return key

    def get(self, key: int) -> Entry | None:
        """The entry under ``key``, or ``None`` if it was evicted."""
        entry = self._entries.get(int(key))
        if entry is not None:
            self._entries.move_to_end(int(key))
        return entry

    @property
    def nbytes(self) -> int:
        """Bytes held by every entry."""
        return sum(e.nbytes for e in self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Free every entry."""
        while self._entries:
            _, old = self._entries.popitem(last=False)
            old.free()


ENTRIES = EntryCache()
