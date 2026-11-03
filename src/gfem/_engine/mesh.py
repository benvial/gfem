"""Build the NGSolve mesh of a `gfem.Mesh` from its arrays.

``netgen.read_gmsh`` is never used: it drops gmsh physical numbers and sets
every face's ``domin`` / ``domout`` to 1 / 0.
"""

from __future__ import annotations

import dataclasses
import weakref
from typing import Any

import netgen.meshing as ngm
import ngsolve as ngs
import numpy as np

from gfem import _mpi
from gfem._periodic import Pairing, check_nodes
from gfem.mesh import LENGTH_UNITS, Mesh


@dataclasses.dataclass(frozen=True, eq=False)
class EngineMesh:
    """An NGSolve mesh and how its elements map back to the `gfem.Mesh`.

    Under ranks ``mesh`` is this rank's part and each order has a local length: the
    same sentence holds for a local element or vertex (ADR 0061). The first rank
    holds nothing.

    Attributes:
        mesh: Coordinates in the gmsh file's length unit (``scale`` metres).
            Domain ``k`` (1-based) is one gmsh volume entity, named after its
            region; boundary ``j`` is one (surface region, domin, domout)
            triple, named after its region, with ``bc`` = the gmsh number.
        scale: Metres per mesh coordinate unit.
        domains: ``(physical tag, entity tag)`` of domain ``k`` at ``k - 1``.
        tetrahedron_order: NGSolve volume element ``i`` is gfem tetrahedron
            ``tetrahedron_order[i]``.
        triangle_order: NGSolve surface element ``i`` is gfem triangle
            ``triangle_order[i]``.
        node_order: NGSolve vertex ``i`` is gfem node ``node_order[i]``.
        comm: The communicator of a distributed mesh; ``None`` on one process.
        surface_spaces: `gfem._engine.coefficients`' surface P0 spaces and maps of
            a distributed mesh, by whether the space is complex.
    """

    mesh: ngs.Mesh
    scale: float
    domains: tuple[tuple[int, int], ...]
    tetrahedron_order: np.ndarray
    triangle_order: np.ndarray
    node_order: np.ndarray
    comm: Any = None
    surface_spaces: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(eq=False)
class _Partition:
    """A rank's first maps of one topology on one communicator."""

    node_order: np.ndarray
    tetrahedron_order: np.ndarray
    triangle_order: np.ndarray


_PARTITIONS: weakref.WeakKeyDictionary[Any, dict[int | None, _Partition]] = (
    weakref.WeakKeyDictionary()
)


def partition_memo(topology: Any) -> dict[int | None, _Partition]:
    """The maps a topology has had, by `gfem._mpi.comm_key`."""
    return _PARTITIONS.setdefault(topology, {})


def inverse_order(order: np.ndarray, n: int) -> np.ndarray:
    """The inverse of an element order at full length, ``-1`` where a rank holds none.

    Args:
        order: `EngineMesh.tetrahedron_order` or `EngineMesh.triangle_order`.
        n: The number of such elements in the whole `gfem.Mesh`.
    """
    inverse = np.full(n, -1, dtype=np.int64)
    inverse[order] = np.arange(len(order))
    return inverse


def to_ngsolve(
    mesh: Mesh, periodic: tuple[Pairing, ...] = (), comm: Any = None
) -> EngineMesh:
    """Build the NGSolve mesh at the current node coordinates of ``mesh``.

    Each pairing of ``periodic`` (a model's periodic pairs, in order) is
    replayed as the netgen point identifications number j + 1, first
    surface's node as netgen's master: NGSolve derives the periodic edges and
    faces from them, and ``Periodic`` spaces read them. Plain spaces are the
    same on the identified mesh.

    Args:
        mesh: The mesh, the same on every rank.
        periodic: A model's periodic pairs.
        comm: A resolved communicator: collective, and each rank gets its part
            (ADR 0061). ``None`` is one process.

    Raises:
        ValueError: The nodes have moved so that a pair no longer matches
            under its translation (NGSolve checks no coordinates).
        gfem.DistributedUnsupportedError: ``periodic`` above one rank.
        gfem._mpi.AgreedError: This partition differs from the first one of this
            topology and communicator.
    """
    if comm is not None:
        if periodic:
            _mpi.refuse_feature("periodic")
        with _mpi.section(comm):
            return _distributed(mesh, comm)
    check_nodes(np.asarray(mesh.nodes, float), periodic)
    scale = LENGTH_UNITS[mesh.length_unit]
    points = np.asarray(mesh.nodes, float) / scale
    ngmesh, domains, tet_order, tri_order = _netgen_mesh(mesh, points, periodic)
    return EngineMesh(
        mesh=ngs.Mesh(ngmesh),
        scale=scale,
        domains=domains,
        tetrahedron_order=tet_order,
        triangle_order=tri_order,
        node_order=np.arange(len(points)),
    )


def _netgen_mesh(
    mesh: Mesh, points: np.ndarray, periodic: tuple[Pairing, ...]
) -> tuple[Any, tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    """The whole netgen mesh on ``points``, its domains and its element orders."""
    names = {(r.dim, r.tag): r.name for r in mesh.regions}
    ngmesh = ngm.Mesh(dim=3)  # ty: ignore[unresolved-attribute]
    ngmesh.AddPoints(np.ascontiguousarray(points))

    keys = np.stack([mesh.tetrahedron_tags, mesh.tetrahedron_entities], axis=1)
    domains, tet_domain = np.unique(keys, axis=0, return_inverse=True)
    tet_domain = tet_domain.ravel() + 1
    tet_order = np.argsort(tet_domain, kind="stable")
    for k, (tag, _) in enumerate(domains, start=1):
        ngmesh.SetMaterial(k, names[3, int(tag)])
        ngmesh.AddElements(
            dim=3, index=k, data=mesh.tetrahedra[tet_domain == k], base=0
        )

    nb = mesh.triangle_neighbours
    din = np.where(nb[:, 0] >= 0, tet_domain[np.maximum(nb[:, 0], 0)], 0)
    dout = np.where(nb[:, 1] >= 0, tet_domain[np.maximum(nb[:, 1], 0)], 0)
    tris = mesh.triangles.copy()
    # netgen faces point out of domin; a face with only a front side is flipped.
    flip = din == 0
    tris[flip] = tris[flip][:, ::-1]
    din[flip], dout[flip] = dout[flip], 0
    keys = np.stack([mesh.triangle_tags, din, dout], axis=1)
    faces, tri_face = np.unique(keys, axis=0, return_inverse=True)
    tri_face = tri_face.ravel() + 1
    tri_order = np.argsort(tri_face, kind="stable")
    for j, (tag, domin, domout) in enumerate(faces, start=1):
        fd = ngm.FaceDescriptor(  # ty: ignore[unresolved-attribute]
            surfnr=j, domin=int(domin), domout=int(domout), bc=int(tag)
        )
        ngmesh.Add(fd)
        ngmesh.SetBCName(j - 1, names[2, int(tag)])
        ngmesh.AddElements(dim=2, index=j, data=tris[tri_face == j], base=0)

    # before the wrap: identifications added after it are not seen
    point = ngm.PointId  # ty: ignore[unresolved-attribute]
    for j, p in enumerate(periodic, start=1):
        for first, second in zip(
            p.first_nodes.tolist(), p.second_nodes.tolist(), strict=True
        ):
            ngmesh.AddPointIdentification(
                point(nr0=first), point(nr0=second), identnr=j, type=2
            )
    return ngmesh, tuple((int(t), int(e)) for t, e in domains), tet_order, tri_order


def _distributed(mesh: Mesh, comm: Any) -> EngineMesh:
    """Rank 0 partitions, every rank reads its gfem numbers (ADR 0061).

    The wheel's Python layer binds no global number, so each point crosses
    ``Distribute`` with its gfem node number in ``x``: partitioning and sending read
    connectivity only. Rank 0 does not wrap the whole mesh, which would cost time and
    memory for nothing.
    """
    nodes = np.asarray(mesh.nodes, float)
    scale = LENGTH_UNITS[mesh.length_unit]
    channel = ngm.MPI_Comm(comm)
    domains = None
    if comm.rank == 0:
        ids = np.zeros_like(nodes)
        ids[:, 0] = np.arange(len(nodes))
        ngmesh, domains, _, _ = _netgen_mesh(mesh, ids, ())
        ngmesh.Partition(comm.size)
        ngmesh.Distribute(channel)
    else:
        ngmesh = ngm.Mesh.Receive(channel)  # ty: ignore[unresolved-attribute]
    domains = comm.bcast(domains, root=0)

    node_order = np.zeros(0, dtype=np.int64)
    # a rank METIS gave nothing has no point array: `Coordinates()` raises
    if ngmesh.ne > 0:
        coordinates = ngmesh.Coordinates()
        node_order = np.rint(coordinates[:, 0]).astype(np.int64)
        coordinates[:] = nodes[node_order] / scale
    tets = node_order[np.asarray(ngmesh._get3dElementsAsTets()).reshape(-1, 4)]  # noqa: SLF001
    tris = node_order[np.asarray(ngmesh._get2dElementsAsTriangles()).reshape(-1, 3)]  # noqa: SLF001

    memo = partition_memo(mesh._topology)  # noqa: SLF001
    key = _mpi.comm_key(comm)
    if key in memo:
        first = memo[key]
        same = np.array_equal(node_order, first.node_order) and (
            _same_rows(tets, mesh.tetrahedra[first.tetrahedron_order])
            and _same_rows(tris, mesh.triangles[first.triangle_order])
        )
    else:
        first = _Partition(
            node_order,
            _match(tets, mesh.tetrahedra, node_order, len(nodes)),
            _match(tris, mesh.triangles, node_order, len(nodes)),
        )
        memo[key] = first
        same = True
    if not _mpi.agree_all(comm, flag=same):
        raise _mpi.AgreedError(
            "This mesh's partition differs from the first one of its topology on "
            "this communicator, though partitioning reads connectivity only. Report "
            "it; a wheel upgrade may have changed netgen (ADR 0061)."
        )
    return EngineMesh(
        mesh=ngs.Mesh(ngmesh),
        scale=scale,
        domains=domains,
        tetrahedron_order=first.tetrahedron_order,
        triangle_order=first.triangle_order,
        node_order=first.node_order,
        comm=comm,
    )


def _same_rows(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether each row of ``a`` holds the nodes of the same row of ``b``."""
    return a.shape == b.shape and np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1))


def _match(
    local: np.ndarray, rows: np.ndarray, node_order: np.ndarray, n_nodes: int
) -> np.ndarray:
    """For each local element, given in gfem nodes, its row number in ``rows``.

    A `gfem.Mesh` repeats no element, so rows of sorted nodes are distinct.
    """
    if len(local) == 0:
        return np.zeros(0, dtype=np.int64)
    held = np.zeros(n_nodes, dtype=bool)
    held[node_order] = True
    candidates = np.flatnonzero(held[rows].all(axis=1))
    # gfem's rows first, and lexsort is stable: each local row follows its gfem row
    stacked = np.vstack([np.sort(rows[candidates], axis=1), np.sort(local, axis=1)])
    order = np.lexsort(stacked.T[::-1])
    in_order = stacked[order]
    start = np.r_[True, (in_order[1:] != in_order[:-1]).any(axis=1)]
    gfem_row = order[start][np.cumsum(start) - 1]
    found = order >= len(candidates)
    if (gfem_row[found] >= len(candidates)).any():
        raise ValueError("A local element has no row in the gfem.Mesh.")
    out = np.empty(len(local), dtype=np.int64)
    out[order[found] - len(candidates)] = candidates[gfem_row[found]]
    return out
