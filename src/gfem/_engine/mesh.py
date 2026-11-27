"""Build the NGSolve mesh of a `gfem.Mesh` from its arrays.

``netgen.read_gmsh`` is never used: it drops gmsh physical numbers and sets
every face's ``domin`` / ``domout`` to 1 / 0.
"""

from __future__ import annotations

import collections.abc
import contextlib
import dataclasses
import logging
import weakref
from typing import Any

import netgen.meshing as ngm
import ngsolve as ngs
import numpy as np

from gfem import _mpi
from gfem._engine.seam_check import (
    SeamCounts,
    expected_surface_counts,
    seam_counts,
    surface_counts,
)
from gfem._periodic import Pairing, check_nodes, pair_names
from gfem.mesh import _TET_EDGES, LENGTH_UNITS, Mesh

# an explicit name: a private module path would leak into users' filters
_log = logging.getLogger("gfem.mesh")


@dataclasses.dataclass(frozen=True, eq=False)
class EngineMesh:
    """An NGSolve mesh and how its elements map back to the `gfem.Mesh`.

    Under ranks ``mesh`` is this rank's part and each order has a local length: the
    same sentence holds for a local element or vertex. The first rank
    holds nothing.

    Attributes:
        mesh: Coordinates in the gmsh file's length unit (``scale`` metres).
            Domain ``k`` (1-based) is one gmsh volume entity (an area of a 2D
            mesh), named after its region; boundary ``j`` is one (surface
            region, domin, domout) triple, named after its region, with
            ``bc`` = the gmsh number on a tetrahedral mesh.
        scale: Metres per mesh coordinate unit.
        domains: ``(physical tag, entity tag)`` of domain ``k`` at ``k - 1``.
        cell_order: NGSolve volume element ``i`` is gfem volume element
            ``cell_order[i]`` (a row of `gfem.Mesh.volume_elements`).
        facet_order: NGSolve surface element ``i`` is gfem surface element
            ``facet_order[i]`` (a row of `gfem.Mesh.surface_elements`).
        node_order: NGSolve vertex ``i`` is gfem node ``node_order[i]``.
        edge_nodes: On a curved mesh, NGSolve edge ``i`` carries the mid-edge
            gfem node ``edge_nodes[i]``; ``None`` on a straight mesh.
        edge_vertices: On a curved mesh, NGSolve edge ``i`` joins the NGSolve
            vertices ``edge_vertices[i]`` (shape ``(n_edges, 2)``), read once at
            the build; ``None`` on a straight mesh.
        comm: The communicator of a distributed mesh; ``None`` on one process.
        pairs: The periodic pairs as the mesh carries them: a model's pairs, or,
            under ranks, the pair swapped when the search kept the swapped
            orientation (its surfaces and nodes exchanged, its translation negated).
        surface_spaces: `gfem._engine.coefficients`' surface P0 spaces and maps of
            a distributed mesh, by whether the space is complex.
    """

    mesh: ngs.Mesh
    scale: float
    domains: tuple[tuple[int, int], ...]
    cell_order: np.ndarray
    facet_order: np.ndarray
    node_order: np.ndarray
    comm: Any = None
    pairs: tuple[Pairing, ...] = ()
    surface_spaces: dict = dataclasses.field(default_factory=dict)
    edge_nodes: np.ndarray | None = None
    edge_vertices: np.ndarray | None = None


#: Partitions a periodic search may take, swaps included.
TRIES = 8
#: A retry shuffles the tetrahedra within blocks of this many consecutive ones.
BLOCK = 4096


@dataclasses.dataclass(eq=False)
class _Partition:
    """A rank's first maps of one topology on one communicator with one set of pairs.

    Attributes:
        seed: The element order the partition was taken from: 0 is the file order,
            and ``s`` shuffles it within blocks with ``default_rng(s)``
            (`block_order`). Every set of pairs of a topology on a communicator
            shares it, so their meshes have the same local tetrahedra.
        swapped: Per pair, whether the mesh carries it swapped.
    """

    node_order: np.ndarray
    cell_order: np.ndarray
    facet_order: np.ndarray
    seed: int = 0
    swapped: tuple[bool, ...] = ()


_PARTITIONS: weakref.WeakKeyDictionary[Any, dict[tuple, _Partition]] = (
    weakref.WeakKeyDictionary()
)


def partition_memo(topology: Any) -> dict[tuple, _Partition]:
    """The maps a topology has had, by `gfem._mpi.comm_key` and pair names."""
    return _PARTITIONS.setdefault(topology, {})


@dataclasses.dataclass
class _Forced:
    """A test's start seed, forced swap and budget for the periodic search."""

    start: int | None = None
    swap: bool = False
    tries: int = TRIES


_FORCED = _Forced()


@contextlib.contextmanager
def forced_search(
    *, start: int | None = None, swap: bool = False, tries: int = TRIES
) -> collections.abc.Generator[None]:
    """Run periodic searches from seed ``start``, swapping a clean partition.

    A private test hook: ``swap`` distributes a clean partition again with the
    pair swapped, so the swapped path runs on a mesh that needs none.
    """
    global _FORCED  # noqa: PLW0603
    previous, _FORCED = _FORCED, _Forced(start, swap, tries)
    try:
        yield
    finally:
        _FORCED = previous


def block_order(n: int, seed: int) -> np.ndarray | None:
    """The tetrahedron order of try ``seed``: ``None`` (file order) for 0.

    A shuffle within blocks of `BLOCK` consecutive tetrahedra keeps METIS near its
    cost on the file order, and was accepted as often as a full shuffle.
    """
    if seed == 0:
        return None
    rng = np.random.default_rng(seed)
    return np.argsort(np.arange(n) // BLOCK + rng.random(n), kind="stable")


def inverse_order(order: np.ndarray, n: int) -> np.ndarray:
    """The inverse of an element order at full length, ``-1`` where a rank holds none.

    Args:
        order: `EngineMesh.cell_order` or `EngineMesh.facet_order`.
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
        comm: A resolved communicator: collective, and each rank gets its part.
            ``None`` is one process.

    Raises:
        ValueError: The nodes have moved so that a pair no longer matches
            under its translation (NGSolve checks no coordinates).
        gfem.DistributedUnsupportedError: More than one pair above one rank.
        gfem._mpi.AgreedError: This partition differs from the first one of this
            topology and communicator; no partition passed the periodic check
            within the budget; or the search left the seed a partition of this
            topology and communicator already took.
        NotImplementedError: A 2D mesh with periodic pairs or a communicator.
    """
    if mesh.dim != 3 and (periodic or comm is not None):
        what = "periodic pairs" if periodic else "a communicator (ranks)"
        raise NotImplementedError(
            f"A 2D mesh is built on one process without periodic pairs, but this "
            f"build asked for {what}. Solve it on one process, or read the file "
            "as a slab with gfem.Mesh.from_gmsh_cross_section."
        )
    if comm is not None and mesh.curved:
        _mpi.refuse_feature("curved_mesh")
    if comm is not None and len(periodic) > 1:
        _mpi.refuse_feature("periodic")
    check_nodes(np.asarray(mesh.nodes, float), periodic)
    if comm is not None:
        with _mpi.section(comm):
            return _distributed(mesh, comm, periodic)
    scale = LENGTH_UNITS[mesh.length_unit]
    points = np.asarray(mesh.nodes, float) / scale
    ngmesh, domains, tet_order, tri_order = _netgen_mesh(mesh, points, periodic)
    ngs_mesh = ngs.Mesh(ngmesh)
    node_order = _node_order(mesh)
    vertices = _edge_vertices(ngs_mesh) if mesh.curved else None
    return EngineMesh(
        mesh=ngs_mesh,
        scale=scale,
        domains=domains,
        cell_order=tet_order,
        facet_order=tri_order,
        node_order=node_order,
        pairs=periodic,
        edge_nodes=(
            _edge_nodes(mesh, vertices, node_order) if vertices is not None else None
        ),
        edge_vertices=vertices,
    )


def _edge_vertices(built: ngs.Mesh) -> np.ndarray:
    """The two NGSolve vertex numbers of each NGSolve edge, ``(n_edges, 2)``."""
    return np.array([[v.nr for v in e.vertices] for e in built.edges], dtype=np.int64)


def _edge_nodes(mesh: Mesh, vertices: np.ndarray, node_order: np.ndarray) -> np.ndarray:
    """The mid-edge gfem node on each NGSolve edge of a curved mesh.

    An edge is its two corner nodes; a tetrahedron's column ``k`` of
    ``volume_midnodes`` is the mid node of its edge ``_TET_EDGES[k]``.
    """
    topology = mesh._topology  # noqa: SLF001
    n = len(mesh.nodes)
    corners = topology.volume_elements
    ends = np.array(_TET_EDGES)
    low = np.minimum(corners[:, ends[:, 0]], corners[:, ends[:, 1]])
    high = np.maximum(corners[:, ends[:, 0]], corners[:, ends[:, 1]])
    keys = (low * n + high).ravel()
    order = np.argsort(keys, kind="stable")
    keys, mids = keys[order], topology.volume_midnodes.ravel()[order]
    nodes = node_order[vertices]
    wanted = np.minimum(nodes[:, 0], nodes[:, 1]) * n + np.maximum(
        nodes[:, 0], nodes[:, 1]
    )
    index = np.minimum(np.searchsorted(keys, wanted), len(keys) - 1)
    if not np.array_equal(keys[index], wanted):
        raise AssertionError("an NGSolve edge is in no tetrahedron of the mesh")
    return mids[index]


def _node_order(mesh: Mesh) -> np.ndarray:
    """NGSolve vertex ``i`` is gfem node ``order[i]``.

    The nodes in file order on a straight mesh. A curved mesh's NGSolve vertices
    are its corner nodes in increasing order, then its mid-edge nodes are added
    as points beyond them (netgen counts the vertices as one more than the largest
    corner index), so the order lists the corners only.
    """
    topology = mesh._topology  # noqa: SLF001
    if topology.volume_midnodes is None:
        return np.arange(len(mesh.nodes))
    return np.unique(topology.volume_elements)


def cell_domains(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """The ``(physical tag, entity tag)`` of each domain, and each cell's 1-based one.

    The cells are the volume elements (`gfem.Mesh.volume_elements`). One process's
    NGSolve volume elements are the cells in order of their domain, each domain in
    file order.
    """
    topology = mesh._topology  # noqa: SLF001
    keys = np.stack([topology.volume_tags, topology.volume_entities], axis=1)
    domains, tet_domain = np.unique(keys, axis=0, return_inverse=True)
    return domains, tet_domain.ravel() + 1


def _netgen_mesh(
    mesh: Mesh,
    points: np.ndarray,
    periodic: tuple[Pairing, ...],
    order: np.ndarray | None = None,
) -> tuple[Any, tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    """The whole netgen mesh on ``points``, its domains and its element orders.

    ``order`` (`block_order`) feeds the tetrahedra of each domain to netgen in
    that order instead of the file's.
    """
    if mesh.dim == 2:
        return _netgen_mesh_2d(mesh, points)
    names = {(r.dim, r.tag): r.name for r in mesh.regions}
    ngmesh = ngm.Mesh(dim=3)  # ty: ignore[unresolved-attribute]
    number = None
    if mesh.curved:
        points, number = _corners_first(mesh, points)
    ngmesh.AddPoints(np.ascontiguousarray(points))

    domains, tet_domain = cell_domains(mesh)
    if order is None:
        order = np.arange(len(tet_domain))
    ordered = tet_domain[order]
    tet_order = order[np.argsort(ordered, kind="stable")]
    tets = mesh.tetrahedra[order]
    volume_mid = mesh._topology.volume_midnodes  # noqa: SLF001
    surface_mid = mesh._topology.surface_midnodes  # noqa: SLF001
    if number is not None and volume_mid is not None:
        # netgen's TET10: the corners, then the mid nodes on `_TET_EDGES`
        tets = number[np.concatenate([tets, volume_mid[order]], axis=1)]
    for k, (tag, _) in enumerate(domains, start=1):
        ngmesh.SetMaterial(k, names[3, int(tag)])
        ngmesh.AddElements(dim=3, index=k, data=tets[ordered == k], base=0)

    nb = mesh.triangle_neighbours
    din = np.where(nb[:, 0] >= 0, tet_domain[np.maximum(nb[:, 0], 0)], 0)
    dout = np.where(nb[:, 1] >= 0, tet_domain[np.maximum(nb[:, 1], 0)], 0)
    tris = mesh.triangles.copy()
    if number is not None and surface_mid is not None:
        # netgen's TRIG6: the corners, then the mid nodes opposite vertices 0, 1, 2
        tris = np.concatenate([tris, surface_mid], axis=1)
    # netgen faces point out of domin; a face with only a front side is flipped.
    flip = din == 0
    tris[flip] = _reversed_triangles(tris[flip])
    if number is not None:
        tris = number[tris]
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
    _identify(ngmesh, mesh, periodic, number)
    return ngmesh, tuple((int(t), int(e)) for t, e in domains), tet_order, tri_order


def _identify(
    ngmesh: Any, mesh: Mesh, periodic: tuple[Pairing, ...], number: np.ndarray | None
) -> None:
    """The periodic point identifications of a netgen mesh.

    netgen's points are the corners-first numbering ``number`` of a curved
    mesh, and a mid-edge node is not one of them, so its pairs are dropped.
    """
    point = ngm.PointId  # ty: ignore[unresolved-attribute]
    n_points = len(mesh.nodes) if number is None else len(_node_order(mesh))
    for j, p in enumerate(periodic, start=1):
        first, second = p.first_nodes, p.second_nodes
        if number is not None:
            first, second = number[first], number[second]
        for a, b in zip(first.tolist(), second.tolist(), strict=True):
            if a < n_points and b < n_points:
                ngmesh.AddPointIdentification(
                    point(nr0=a), point(nr0=b), identnr=j, type=2
                )


def _corners_first(mesh: Mesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A curved mesh's points with its corners first, and each node's new number.

    The corners keep their file order (`_node_order`), then every other node
    (mid-edge nodes, and any node in no element) follows in file order.
    """
    corners = _node_order(mesh)
    rest = np.setdiff1d(np.arange(len(points)), corners)
    order = np.concatenate([corners, rest])
    number = np.empty(len(points), dtype=np.int64)
    number[order] = np.arange(len(points))
    return points[order], number


def _reversed_triangles(tris: np.ndarray) -> np.ndarray:
    """Triangles of 3 or 6 nodes with the opposite orientation.

    The corners reverse, and each mid node stays opposite the corner it was
    opposite, so the mid nodes reverse with them.
    """
    if tris.shape[1] == 3:
        return tris[:, ::-1]
    return tris[:, [2, 1, 0, 5, 4, 3]]


def _netgen_mesh_2d(
    mesh: Mesh, points: np.ndarray
) -> tuple[Any, tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    """The netgen mesh of a 2D mesh: areas, then the lines' segments.

    A segment's normal is its direction crossed with +z: NGSolve's, and the one
    `gfem.Mesh.adjacency` reads, so it points out of ``domin``. A segment with
    only a front side is reversed so that its normal leaves the mesh, as a 3D
    face with only a front side is flipped.
    """
    topology = mesh._topology  # noqa: SLF001
    names = {(r.dim, r.tag): r.name for r in mesh.regions}
    ngmesh = ngm.Mesh(dim=2)  # ty: ignore[unresolved-attribute]
    ngmesh.AddPoints(np.ascontiguousarray(points))

    domains, cell_domain = cell_domains(mesh)
    cell_order = np.argsort(cell_domain, kind="stable")
    cells = topology.volume_elements
    for k, (tag, _) in enumerate(domains, start=1):
        ngmesh.SetMaterial(k, names[2, int(tag)])
        ngmesh.AddElements(
            dim=2, index=k, data=np.ascontiguousarray(cells[cell_domain == k]), base=0
        )

    nb = topology.surface_neighbours
    din = np.where(nb[:, 0] >= 0, cell_domain[np.maximum(nb[:, 0], 0)], 0)
    dout = np.where(nb[:, 1] >= 0, cell_domain[np.maximum(nb[:, 1], 0)], 0)
    segments = topology.surface_elements.copy()
    flip = din == 0
    segments[flip] = segments[flip][:, ::-1]
    din[flip], dout[flip] = dout[flip], 0
    keys = np.stack([topology.surface_tags, din, dout], axis=1)
    groups, segment_group = np.unique(keys, axis=0, return_inverse=True)
    segment_group = segment_group.ravel() + 1
    facet_order = np.argsort(segment_group, kind="stable")
    for j, (tag, _, _) in enumerate(groups, start=1):
        ngmesh.SetBCName(j - 1, names[1, int(tag)])
        ngmesh.AddElements(
            dim=1,
            index=j,
            data=np.ascontiguousarray(segments[segment_group == j]),
            base=0,
        )
    return ngmesh, tuple((int(t), int(e)) for t, e in domains), cell_order, facet_order


@dataclasses.dataclass(eq=False)
class _Received:
    """One partition as a rank received it, before the wrap."""

    ngmesh: Any
    domains: tuple[tuple[int, int], ...]
    node_order: np.ndarray
    tets: np.ndarray
    tris: np.ndarray


@dataclasses.dataclass(eq=False)
class _Try:
    """A checked periodic partition and its agreed counts."""

    received: _Received
    cell_order: np.ndarray
    facet_order: np.ndarray
    missing: int
    reverse: int
    surfaces: bool


def _receive(mesh: Mesh, comm: Any, pairs: tuple[Pairing, ...], seed: int) -> _Received:
    """Rank 0 partitions and sends, every rank reads its gfem numbers.

    The wheel's Python layer binds no global number, so each point crosses
    ``Distribute`` with its gfem node number in ``x``: partitioning and sending read
    connectivity only. Rank 0 does not wrap the whole mesh, which would cost time and
    memory for nothing.
    """
    nodes = np.asarray(mesh.nodes, float)
    channel = ngm.MPI_Comm(comm)
    domains = None
    if comm.rank == 0:
        ids = np.zeros_like(nodes)
        ids[:, 0] = np.arange(len(nodes))
        order = block_order(len(mesh.tetrahedra), seed)
        ngmesh, domains, _, _ = _netgen_mesh(mesh, ids, pairs, order)
        ngmesh.Partition(comm.size)
        ngmesh.Distribute(channel)
    else:
        ngmesh = ngm.Mesh.Receive(channel)  # ty: ignore[unresolved-attribute]
    domains = comm.bcast(domains, root=0)
    node_order = np.zeros(0, dtype=np.int64)
    # a rank METIS gave nothing has no point array: `Coordinates()` raises
    if ngmesh.ne > 0:
        node_order = np.rint(ngmesh.Coordinates()[:, 0]).astype(np.int64)
    tets = node_order[np.asarray(ngmesh._get3dElementsAsTets()).reshape(-1, 4)]  # noqa: SLF001
    tris = node_order[np.asarray(ngmesh._get2dElementsAsTriangles()).reshape(-1, 3)]  # noqa: SLF001
    return _Received(ngmesh, domains, node_order, tets, tris)


def _checked(mesh: Mesh, comm: Any, pairs: tuple[Pairing, ...], seed: int) -> _Try:
    """One periodic partition, judged on agreed counts.

    Every rank counts its missing and reverse twins (`seam_counts`) and its
    triangles per surface tag; one reduction sums them with a flag per rank whose
    check raised.

    Raises:
        gfem._mpi.AgreedError: The check raised on some rank, which logged it.
    """
    received = _receive(mesh, comm, pairs, seed)
    n_nodes = len(mesh.nodes)
    tags = np.unique(mesh.triangle_tags)
    paired = {mesh.region(name).tag for p in pairs for name in (p.first, p.second)}
    local = np.zeros(4 + len(tags) + comm.size, dtype=np.int64)
    tet_order = tri_order = np.zeros(0, dtype=np.int64)
    try:
        tet_order = _match(received.tets, mesh.tetrahedra, received.node_order, n_nodes)
        tri_order = _match(received.tris, mesh.triangles, received.node_order, n_nodes)
        if received.ngmesh.ne > 0:
            local[:4] = seam_counts(received.ngmesh)  # `SeamCounts`'s order
        local[4 : 4 + len(tags)] = surface_counts(mesh.triangle_tags, tri_order, tags)
    except Exception:  # noqa: BLE001
        _log.warning(
            "gfem: rank %d: the periodic mesh check raised:",
            comm.rank,
            extra={"rank": comm.rank},
            exc_info=True,
        )
        local[4 + len(tags) + comm.rank] = 1
    total = _mpi.reduce_bcast(comm, local)
    failed = np.flatnonzero(total[4 + len(tags) :])
    if len(failed):
        raise _mpi.AgreedError(
            "The periodic mesh check raised on ranks "
            f"{', '.join(map(str, failed))}; each logged its traceback."
        )
    want = expected_surface_counts(mesh.triangle_tags, tags, paired)
    seam = SeamCounts(*total[:4].tolist())
    return _Try(
        received,
        tet_order,
        tri_order,
        missing=seam.missing_edges + seam.missing_faces,
        reverse=seam.reverse_edges + seam.reverse_faces,
        surfaces=bool(np.array_equal(total[4 : 4 + len(tags)], want)),
    )


def _swap(p: Pairing) -> Pairing:
    """``p`` with its surfaces exchanged and its translation negated."""
    return dataclasses.replace(
        p,
        first=p.second,
        second=p.first,
        first_nodes=p.second_nodes,
        second_nodes=p.first_nodes,
        translation=(-p.translation[0], -p.translation[1], -p.translation[2]),
    )


def _search(
    mesh: Mesh,
    comm: Any,
    periodic: tuple[Pairing, ...],
    memo: dict[tuple, _Partition],
    key: int | None,
) -> tuple[_Try, int, bool]:
    """The strict search for a partition of one pair: try, seed, swapped.

    A try is kept unswapped with no missing and no reverse twins. With missing twins
    and no reverse ones, the same order is distributed again with the pair swapped,
    and kept if it misses nothing. Anything else moves to the next seed. Surface
    counts off their expected values fail a try too. A topology already distributed
    on this communicator holds its seed; a search for other pairs, or the same
    pairs again, starts there in the orientation it kept.

    Raises:
        gfem._mpi.AgreedError: No partition passed in the budget, or the search
            left the seed already held.
    """
    forced = _FORCED
    names = pair_names(periodic)
    held = {k[1]: p for k, p in memo.items() if k[0] == key}
    fixed = next((p.seed for p in held.values()), None)
    again = held.get(names)
    seed = forced.start if forced.start is not None else (fixed or 0)
    replay = again is not None and again.swapped == (True,) and forced.start is None
    rows: list[str] = []
    used = 0
    swap = False
    while used < forced.tries:
        if fixed is not None and seed != fixed:
            raise _mpi.AgreedError(_conflict(() in held))
        if not replay:
            found = _checked(mesh, comm, periodic, seed)
            used += 1
            rows.append(f"seed {seed}: {found.missing}, {found.reverse}")
            if (
                found.surfaces
                and found.missing == 0
                and found.reverse == 0
                and not forced.swap
            ):
                return found, seed, False
            swap = found.surfaces and found.reverse == 0
            swap = swap and (found.missing > 0 or forced.swap)
            del found
        if replay or (swap and used < forced.tries):
            replay = False
            found = _checked(mesh, comm, tuple(_swap(p) for p in periodic), seed)
            used += 1
            rows.append(f"seed {seed} swapped: {found.missing}, {found.reverse}")
            if found.surfaces and found.missing == 0:
                return found, seed, True
            del found
        seed += 1
    raise _mpi.AgreedError(
        f"No partition of this periodic mesh on {comm.size} ranks passed gfem's "
        f"check in {forced.tries} tries (missing, reverse per try: "
        f"{'; '.join(rows)}). Try another number of ranks, or run one-process "
        "copies with comm=MPI.COMM_SELF."
    )


def _conflict(plain: bool) -> str:
    first = "without its pairs" if plain else "with other pairs"
    return (
        f"This mesh was distributed {first} first on this communicator, with "
        "another partition than its periodic model needs; solve the periodic "
        "model first, or use comm.Dup()."
    )


def _distributed(
    mesh: Mesh, comm: Any, periodic: tuple[Pairing, ...] = ()
) -> EngineMesh:
    """This rank's part of ``mesh``, the same partition each time.

    A mesh with a pair takes the partition `_search` keeps. A plain mesh follows
    the seed a periodic search of its topology and communicator took, so both have
    the same local tetrahedra in the same order.
    """
    nodes = np.asarray(mesh.nodes, float)
    scale = LENGTH_UNITS[mesh.length_unit]
    memo = partition_memo(mesh._topology)  # noqa: SLF001
    key = _mpi.comm_key(comm)
    names = pair_names(periodic)
    pairs = periodic
    tet_order = tri_order = None
    if periodic:
        found, seed, swapped = _search(mesh, comm, periodic, memo, key)
        received = found.received
        tet_order, tri_order = found.cell_order, found.facet_order
        if swapped:
            pairs = tuple(_swap(p) for p in periodic)
    else:
        seed = next((p.seed for k, p in memo.items() if k[0] == key), 0)
        swapped = False
        received = _receive(mesh, comm, (), seed)
    node_order = received.node_order
    tets, tris = received.tets, received.tris
    first = memo.get((key, names))
    if first is not None:
        same = np.array_equal(node_order, first.node_order) and (
            _same_rows(tets, mesh.tetrahedra[first.cell_order])
            and _same_rows(tris, mesh.triangles[first.facet_order])
        )
    else:
        if tet_order is None or tri_order is None:
            tet_order = _match(tets, mesh.tetrahedra, node_order, len(nodes))
            tri_order = _match(tris, mesh.triangles, node_order, len(nodes))
        first = _Partition(
            node_order,
            tet_order,
            tri_order,
            seed=seed,
            swapped=(swapped,) if periodic else (),
        )
        memo[key, names] = first
        same = True
    if not _mpi.agree_all(comm, flag=same):
        raise _mpi.AgreedError(
            "This mesh's partition differs from the first one of its topology on "
            "this communicator, though partitioning reads connectivity only. Report "
            "it; a wheel upgrade may have changed netgen."
        )
    ngmesh = received.ngmesh
    if ngmesh.ne > 0:
        ngmesh.Coordinates()[:] = nodes[node_order] / scale
    return EngineMesh(
        mesh=ngs.Mesh(ngmesh),
        scale=scale,
        domains=received.domains,
        cell_order=first.cell_order,
        facet_order=first.facet_order,
        node_order=first.node_order,
        comm=comm,
        pairs=pairs,
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
