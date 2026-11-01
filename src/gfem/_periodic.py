"""Node pairs of the periodic pairs of a model, matched by translation.

Pure NumPy and SciPy on the `gfem.Mesh` arrays. A `Pairing` is computed once
per mesh topology and pair of surface names, from the imported nodes; the
engine replays it as netgen point identifications. NGSolve checks no
coordinates, so every check that the two surfaces really match is here.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.spatial import KDTree

from gfem._domain import EMDomain
from gfem._pytree import readonly
from gfem.mesh import Mesh, _shortest_edges, _Topology

if TYPE_CHECKING:
    from gfem.model import Model

#: Node match tolerance, relative to the shortest edge near the pair.
REL_TOL = 1e-6
#: Largest angle (rad) between the planes of two paired planar surfaces.
ANGLE_TOL = 1e-6
#: Largest component of a wave vector off the span of the translations,
#: relative to its length.
SPAN_RTOL = 1e-9

_SETPERIODIC = (
    "Mesh the pair as periodic (gmsh: `gmsh.model.mesh.setPeriodic(2, "
    "second_tags, first_tags, translation)`) and import it again."
)


@dataclasses.dataclass(frozen=True, eq=False)
class Pairing:
    """The node pairs of one `gfem.Periodic`, by mesh topology.

    Compares by identity: it is static pytree data of a `Model`, cached per
    topology, so every model on one imported mesh shares it.

    Attributes:
        first: The first surface's name.
        second: The second surface's name.
        master: Per node pair, the node on ``first``.
        minion: Per node pair, the node on ``second``; x[minion] =
            x[master] + translation.
        translation: a, from ``first`` to ``second``, m.
        tol: The node match tolerance, m.
    """

    first: str
    second: str
    master: np.ndarray
    minion: np.ndarray
    translation: tuple[float, float, float]
    tol: float

    @property
    def label(self) -> str:
        """``Periodic(('first', 'second'))``, opening error messages."""
        return f"Periodic({(self.first, self.second)!r})"


# Per mesh topology, the pairings by (first, second).
_PAIRINGS: weakref.WeakKeyDictionary[_Topology, dict[tuple[str, str], Pairing]] = (
    weakref.WeakKeyDictionary()
)


def _vec(x: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(v):.6g}" for v in x) + ")"


def pairing(mesh: Mesh, first: str, second: str) -> Pairing:
    """The node pairs of ``first`` and ``second``, matched by translation.

    Checks, in order: equal node counts, parallel planes (planar surfaces),
    every node matched within the tolerance, identical triangulations, and
    no element touching both surfaces. Cached per mesh topology.

    Raises:
        ValueError: The surfaces do not form a periodic pair; the message
            gives the remedy.
    """
    topology = mesh._topology  # noqa: SLF001
    cache = _PAIRINGS.setdefault(topology, {})
    key = (first, second)
    if key not in cache:
        cache[key] = _pair(mesh, first, second)
    return cache[key]


def pair_of_surface(mesh: Mesh, name: str) -> Pairing | None:
    """The periodic pair a model on ``mesh``'s topology made of surface ``name``.

    Read from the per-topology cache of `pairing`, so it knows the pairs of
    every `gfem.Periodic` model built on this topology; ``None`` when no
    pair has ``name`` as a surface.
    """
    for p in _PAIRINGS.get(mesh._topology, {}).values():  # noqa: SLF001
        if name in (p.first, p.second):
            return p
    return None


def tie_roots(n_nodes: int, pairings: Iterable[Pairing]) -> np.ndarray:
    """Per node, the smallest node of its tie class.

    A tie class is a set of nodes the periodic pairs identify (a corner of a
    3-pair cell is a class of 8 nodes, a node of one surface a class of 2,
    an unpaired node a class of its own). Keys built from the roots are the
    same for an edge or face and its images across the pairs.
    """
    pairs = list(pairings)
    root = np.arange(n_nodes)
    if not pairs:
        return root
    rows = np.concatenate([p.master for p in pairs])
    cols = np.concatenate([p.minion for p in pairs])
    graph = scipy.sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes)
    )
    _, label = scipy.sparse.csgraph.connected_components(graph, directed=False)
    smallest = np.full(label.max() + 1, n_nodes)
    np.minimum.at(smallest, label, root)
    return smallest[label]


def _surface_triangles(mesh: Mesh, name: str) -> np.ndarray:
    return mesh.triangles[mesh.triangle_tags == mesh.region(name).tag]


def _pair(mesh: Mesh, first: str, second: str) -> Pairing:
    label = f"Periodic({(first, second)!r})"
    x = mesh.imported_nodes
    tf, ts = _surface_triangles(mesh, first), _surface_triangles(mesh, second)
    f, s = np.unique(tf), np.unique(ts)
    if len(f) != len(s):
        raise ValueError(
            f"{label}: {first!r} has {len(f)} nodes and {second!r} has {len(s)}, "
            "so their meshes cannot match under a translation. " + _SETPERIODIC
        )
    angle = _plane_angle(x, tf, ts)
    if angle > ANGLE_TOL:
        raise ValueError(
            f"{label}: the surfaces are not parallel (angle {np.degrees(angle):.1f}°); "
            "gfem pairs surfaces by a translation only, and rotational or "
            "mirror periodicity is not supported."
        )
    near = np.isin(mesh.tetrahedra, np.concatenate([f, s])).any(axis=1)
    tol = REL_TOL * float(_shortest_edges(mesh)[near].min())
    a0 = x[s].mean(axis=0) - x[f].mean(axis=0)
    dist, match = KDTree(x[f]).query(x[s] - a0)
    far = dist > tol
    if far.any():
        worst = int(np.argmax(dist))
        raise ValueError(
            f"{label}: no translation maps {first!r} onto {second!r}: with a = "
            f"{_vec(a0)} m, {int(far.sum())} nodes of {second!r} are farther than "
            f"{tol:.3g} m from any node of {first!r} (worst {dist[worst]:.3g} m at "
            f"{_vec(x[s[worst]])} m). " + _SETPERIODIC
        )
    master = f[match]
    if len(np.unique(master)) != len(master):
        raise ValueError(
            f"{label}: two nodes of {second!r} match the same node of {first!r} "
            f"within {tol:.3g} m, so the match is not one to one. " + _SETPERIODIC
        )
    a = (x[s] - x[master]).mean(axis=0)
    to_first = np.full(len(x), -1, dtype=np.int64)
    to_first[s] = master
    twins = {tuple(t) for t in np.sort(to_first[ts], axis=1).tolist()}
    missing = len(twins - {tuple(t) for t in np.sort(tf, axis=1).tolist()})
    if missing or len(tf) != len(ts):
        raise ValueError(
            f"{label}: the nodes match under a = {_vec(a)} m but "
            f"{max(missing, abs(len(ts) - len(tf)))} triangles of {second!r} have no "
            f"translated twin on {first!r}; the two surface meshes must be "
            "identical up to the translation (gmsh setPeriodic)."
        )
    on_f = np.isin(mesh.tetrahedra, f).any(axis=1)
    on_s = np.isin(mesh.tetrahedra, s).any(axis=1)
    both = int((on_f & on_s).sum())
    if both:
        raise ValueError(
            f"{label}: {both} elements touch both {first!r} and {second!r}; a "
            "periodic cell needs at least two elements across each period. "
            "Refine the mesh along a."
        )
    return Pairing(
        first=first,
        second=second,
        master=readonly(master.astype(np.int64)),
        minion=readonly(s.astype(np.int64)),
        translation=(float(a[0]), float(a[1]), float(a[2])),
        tol=tol,
    )


def _plane_angle(x: np.ndarray, tf: np.ndarray, ts: np.ndarray) -> float:
    """The angle between the planes of two planar surfaces; 0 if one is curved."""
    normals = []
    for tris in (tf, ts):
        p = x[tris]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        ref = n[0]
        if np.linalg.norm(np.cross(n, ref), axis=1).max() > ANGLE_TOL:
            return 0.0  # not planar: the node match decides
        normals.append(ref)
    cos = min(1.0, abs(float(normals[0] @ normals[1])))
    return float(np.arccos(cos))


def twin_faces(mesh: Mesh, p: Pairing) -> tuple[np.ndarray, np.ndarray]:
    """Triangle indices of ``first`` and of their twins on ``second``, aligned."""
    tri_f = np.flatnonzero(mesh.triangle_tags == mesh.region(p.first).tag)
    tri_s = np.flatnonzero(mesh.triangle_tags == mesh.region(p.second).tag)
    to_first = np.arange(len(mesh.imported_nodes))
    to_first[p.minion] = p.master
    index = {
        tuple(t): i
        for i, t in zip(
            tri_f, np.sort(mesh.triangles[tri_f], axis=1).tolist(), strict=True
        )
    }
    mapped = np.sort(to_first[mesh.triangles[tri_s]], axis=1).tolist()
    return np.array([index[tuple(t)] for t in mapped], dtype=np.int64), tri_s


def check_domain(mesh: Mesh, p: Pairing, domain: EMDomain) -> None:
    """Twin faces border the solve domain alike (excluded volumes on both sides).

    Raises:
        ValueError: A face borders the solve domain where its twin borders
            only excluded volume.
    """
    tri_f, tri_s = twin_faces(mesh, p)
    nb = mesh.triangle_neighbours
    active = (nb >= 0) & domain.active[np.maximum(nb, 0)]
    on_f, on_s = active[tri_f].any(axis=1), active[tri_s].any(axis=1)
    bad = np.flatnonzero(on_f != on_s)
    if not bad.size:
        return
    i = int(bad[0])
    inside, outside, tri = (
        (p.second, p.first, tri_f[i]) if on_s[i] else (p.first, p.second, tri_s[i])
    )
    tets = nb[tri][nb[tri] >= 0]
    names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    region = names[int(mesh.tetrahedron_tags[tets[0]])]
    raise ValueError(
        f"{p.label}: {bad.size} faces of {inside!r} border the solve domain where "
        f"their twins on {outside!r} border excluded volume {region!r} (or the "
        "reverse): exclude matching volumes on both sides."
    )


def check_pec(mesh: Mesh, p: Pairing, pec: Iterable[str]) -> None:
    """PEC surfaces meet the two surfaces of the pair along twin edges.

    NGSolve spreads a Dirichlet condition across a pair, so a PEC edge on
    one side without a twin would silently become PEC on the other.

    Raises:
        ValueError: A PEC surface meets one side along an edge whose twin is
            not PEC.
    """
    n = len(mesh.imported_nodes)
    side = np.zeros(n, dtype=np.int8)
    side[p.master], side[p.minion] = 1, 2
    to_first = np.arange(n)
    to_first[p.minion] = p.master
    edges: dict[str, dict[int, set[tuple[int, int]]]] = {}
    for name in pec:
        tris = _surface_triangles(mesh, name)
        e = np.sort(tris[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2), axis=1)
        out: dict[int, set[tuple[int, int]]] = {1: set(), 2: set()}
        for k in (1, 2):
            on = e[(side[e[:, 0]] == k) & (side[e[:, 1]] == k)]
            out[k] = {tuple(v) for v in np.sort(to_first[on], axis=1).tolist()}
        edges[name] = out
    every = {k: set().union(*(o[k] for o in edges.values())) for k in (1, 2)}
    for name, out in edges.items():
        for k, other in ((2, 1), (1, 2)):
            lone = out[k] - every[other]
            if lone:
                here, there = (p.second, p.first) if k == 2 else (p.first, p.second)
                raise ValueError(
                    f"{p.label}: PEC {name!r} meets {here!r} along {len(lone)} "
                    f"edges that have no PEC twin on {there!r}; NGSolve would "
                    "impose PEC on both sides. Make the PEC surfaces periodic, "
                    "or move them off the pair."
                )


def check_nodes(nodes: np.ndarray, pairings: Iterable[Pairing]) -> None:
    """Every node pair is still related by its pair's translation.

    Raises:
        ValueError: Moved nodes break a pair; the message names the remedy.
    """
    x = np.asarray(nodes, dtype=float)
    for p in pairings:
        d = np.linalg.norm(x[p.minion] - x[p.master] - p.translation, axis=1)
        far = d > p.tol
        if far.any():
            worst = int(np.argmax(d))
            raise ValueError(
                f"The moved mesh breaks {p.label}: {int(far.sum())} node pairs "
                f"are no longer related by the translation a = "
                f"{_vec(p.translation)} m (worst {d[worst]:.3g} m at "
                f"{_vec(x[p.minion[worst]])} m). Move paired nodes by the same "
                "displacement: build the shape with gfem.boundary_morph on this "
                "periodic model, which ties them."
            )


def tie_classes(n_nodes: int, pairings: Iterable[Pairing]) -> np.ndarray:
    """The tie class of every node, by its representative: the node it moves with.

    The classes of `tie_roots`; the representative is the class's lowest
    node that is no pair's minion, which lies on the first surface of every
    pair it is in (for `gfem.shape.boundary_morph`'s tied motion).

    Returns:
        ``(n_nodes,)`` int: the representative of each node's class (the
        node itself for an unpaired node).
    """
    pairs = list(pairings)
    roots = tie_roots(n_nodes, pairs)
    if not pairs:
        return roots
    # prefer the nodes that are no minion, then the lowest index
    key = np.arange(n_nodes)
    key[np.concatenate([p.minion for p in pairs])] += n_nodes
    lowest = np.full(n_nodes, 2 * n_nodes)
    np.minimum.at(lowest, roots, key)
    return lowest[roots] % n_nodes


def bloch_phases(
    model: Model, wave_vector: Iterable[float] | None
) -> tuple[complex, ...] | None:
    """NGSolve's phase of each periodic pair, in ``boundaries`` order.

    ``exp(-i k·a_j)`` for the wave vector k (rad/m) and pair j's translation
    a_j (m), the factor from a pair's first surface to its second
    (`gfem.conventions.BLOCH_PHASE`); ``None`` without a wave vector (plain
    periodicity).
    """
    if wave_vector is None:
        return None
    k = np.asarray(tuple(wave_vector), dtype=float)
    return tuple(
        complex(np.exp(-1j * float(k @ np.asarray(a))))
        for a in model.periodic_translations
    )


def check_wave_vector(
    model: Model, wave_vector: tuple[float, float, float] | None, owner: str
) -> None:
    """A wave vector needs a periodic pair, and must lie in their translations' span.

    Raises:
        ValueError: ``wave_vector`` is set on a model without a
            `gfem.Periodic` pair, or has a component that no pair's phase
            reads (it would be ignored).
    """
    if wave_vector is None:
        return
    translations = np.asarray(model.periodic_translations, dtype=float)
    if not len(translations):
        raise ValueError(
            f"{owner}.wave_vector sets a Bloch phase, but the model has no "
            "gfem.Periodic pair. Pair the cell's opposite surfaces with "
            "gfem.Periodic, or leave wave_vector=None."
        )
    k = np.asarray(wave_vector, dtype=float)
    basis, _ = np.linalg.qr(translations.T)
    projected = basis @ (basis.T @ k)
    off = float(np.linalg.norm(k - projected))
    if off > SPAN_RTOL * float(np.linalg.norm(k)):
        raise ValueError(
            f"{owner}.wave_vector {_vec(k)} rad/m has a component of "
            f"{off:.6g} rad/m outside the span of the periodic translations "
            f"{', '.join(_vec(a) for a in translations)} m, which no pair's phase "
            "reads, so it would be ignored. Give the wave vector in that span, "
            f"e.g. {_vec(projected)} rad/m."
        )
