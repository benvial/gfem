"""Engine kernels behind `gfem.Field`: its NGSolve space, sampling, curl.

A field's coefficients are DOF values of a compressed NGSolve space on the
mesh in its file length unit, ``definedon`` the support minus the excluded
volume entities. That numbering depends only on the mesh topology, the
support, the space and the order, so the space can be rebuilt from the
field alone. The engine value of a field times ``scale ** LENGTH_POWER``
(``scale`` in metres per file unit) is the SI value: H(curl) values are
covariant (1/length), H(div) values contravariant (1/area).
"""

from __future__ import annotations

import collections
import weakref
from typing import TYPE_CHECKING, Any

import ngsolve as ngs
import numpy as np

from gfem import _mpi
from gfem._engine import quadrature
from gfem._engine.cache import discretization
from gfem._engine.coefficients import p0
from gfem._engine.mesh import EngineMesh, cell_domains, inverse_order
from gfem._engine.runtime import serial_spaces
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from collections.abc import Callable

    from gfem._field import Field

__all__ = [
    "LENGTH_POWER",
    "curl_matrix",
    "dirichlet_h1",
    "dof_shape",
    "element_rows",
    "face_keys",
    "fe_space",
    "h1_key",
    "mean_rows",
    "one_process_column",
    "one_process_dofs",
    "rank_piece",
    "sample",
    "support_region",
    "surface_triangles",
    "tetrahedron_face_keys",
    "vertex_dofs",
]

LENGTH_POWER = {"h1": 0, "l2": 0, "hcurl": -1, "hdiv": -2}


def support_region(
    built: EngineMesh,
    mesh: Mesh,
    support: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
) -> ngs.Region:
    """The NGSolve volume region of ``support`` minus ``excluded`` entities."""
    names = mesh._region_names()  # noqa: SLF001
    wanted, out = set(support), set(excluded)
    bits = ngs.BitArray(len(built.domains))
    bits.Clear()
    for k, (tag, entity) in enumerate(built.domains):
        if names[tag] in wanted and (tag, entity) not in out:
            bits.Set(k)
    return ngs.Region(built.mesh, ngs.VOL, bits)


def _build(region: ngs.Region, space: str, order: int) -> ngs.FESpace:
    with serial_spaces():
        return _built(region, space, order)


def _built(region: ngs.Region, space: str, order: int) -> ngs.FESpace:
    mesh = region.mesh
    if space == "h1":
        fes = ngs.H1(mesh, order=order, definedon=region)
    elif space == "hcurl":
        fes = ngs.HCurl(mesh, order=order, type1=True, definedon=region)
    elif space == "hdiv":
        fes = ngs.HDiv(mesh, order=order, definedon=region)
    elif space == "l2":
        fes = ngs.L2(mesh, order=order, definedon=region)
    else:  # pragma: no cover - Field validates the space
        raise ValueError(f"Unknown space {space!r}.")
    return ngs.Compress(fes)


def fe_space(
    mesh: Mesh,
    support: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
    space: str,
    order: int,
    *,
    comm: Any = None,
) -> tuple[EngineMesh, ngs.FESpace, ngs.Region]:
    """The real NGSolve space of this layout on ``mesh`` at its node positions.

    Its DOF numbering is that of every compressed space of the same kind
    and order on the same region, real or complex, so solves and fields
    agree. Under ``comm`` it is this rank's part, and a miss is collective.
    """
    disc = discretization(mesh, (), comm)
    built = disc.built
    key = (("field", tuple(support), tuple(excluded)), space, order)

    def build():
        region = support_region(built, mesh, support, excluded)
        return _build(region, space, order), region

    fes, region = disc.space(key, build)
    return built, fes, region


def dirichlet_h1(
    mesh: Mesh,
    physics: str,
    regions: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
    *,
    order: int,
    fixed: tuple[str, ...],
) -> tuple[ngs.FESpace, EngineMesh]:
    """The compressed H1 space of a scalar physics, fixed on ``fixed`` surfaces.

    It is defined on ``regions`` minus the ``excluded`` volume entities and
    cached on the mesh's discretization under ``physics``.
    """
    disc = discretization(mesh)
    built = disc.built

    def build() -> ngs.FESpace:
        names = built.mesh.GetBoundaries()
        with serial_spaces():
            space = ngs.H1(
                built.mesh,
                order=order,
                definedon=support_region(built, mesh, regions, excluded),
                dirichlet=[j + 1 for j, n in enumerate(names) if n in fixed],
            )
            return ngs.Compress(space)

    return disc.space(h1_key(physics, regions, excluded, order, fixed), build), built


def h1_key(
    physics: str,
    regions: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
    order: int,
    fixed: tuple[str, ...],
) -> tuple:
    """The key of `dirichlet_h1`'s space in its discretization: no node positions."""
    return ((physics, regions, excluded), "h1", order, fixed)


def vertex_dofs(space: ngs.FESpace, built: EngineMesh | None = None) -> np.ndarray:
    """The vertex DOFs of the H1 ``space``; with ``built``, per gfem node (-1 off it).

    NGSolve's H1 basis is hierarchical: the vertex hats are its partition of
    unity, and a constant is its value on the vertex DOFs and zero on the
    edge, face and cell DOFs.
    """
    n = space.mesh.nv
    dofs = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        numbers = space.GetDofNrs(ngs.NodeId(ngs.VERTEX, i))
        if numbers and numbers[0] >= 0:
            dofs[i] = numbers[0]
    if built is None:
        return dofs[dofs >= 0]
    out = np.full(int(built.node_order.max()) + 1, -1, dtype=np.int64)
    out[built.node_order] = dofs
    return out


def face_keys(faces: np.ndarray, n: int) -> np.ndarray:
    """One integer per triangle ``(k, 3)`` of ``n`` nodes, whatever its node order."""
    f = np.sort(faces, axis=1).astype(np.int64)
    return (f[:, 0] * n + f[:, 1]) * n + f[:, 2]


def tetrahedron_face_keys(mesh: Mesh, tets: np.ndarray) -> np.ndarray:
    """The `face_keys` of the four faces of each of ``tets``, ``(4 k,)``.

    ``tets`` indexes ``mesh.tetrahedra`` (indices or a boolean mask).
    """
    faces = mesh.tetrahedra[tets][:, _TET_FACES].reshape(-1, 3)
    return face_keys(faces, len(mesh.imported_nodes))


def surface_triangles(mesh: Mesh, surfaces: tuple[str, ...]) -> np.ndarray:
    """The triangles (node triples) of the named surfaces."""
    tags = [mesh.region(s).tag for s in surfaces]
    return mesh.triangles[np.isin(mesh.triangle_tags, tags)]


_SHAPES: weakref.WeakKeyDictionary[Any, dict[tuple, tuple[int, int]]] = (
    weakref.WeakKeyDictionary()
)


def dof_shape(
    mesh: Mesh,
    support: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
    space: str,
    order: int,
    *,
    comm: Any = None,
) -> tuple[int, int]:
    """``(width, ndof)`` of a layout: a piece's rows at the JAX seam, and its own.

    ``width`` is the same on every rank, the largest ``ndof``, since a callback whose
    read outputs are all empty does not run on that rank; on one process
    it is ``ndof``. A shape step's nodes are tracers when this is needed, so it reads
    the imported nodes, and is memoized: a topology and a rank count have one
    partition. A miss is collective.
    """
    topology = mesh._topology  # noqa: SLF001
    memo = _SHAPES.setdefault(topology, {})
    key = (_mpi.comm_key(comm), tuple(support), tuple(excluded), space, order)
    if key not in memo:
        still = Mesh(mesh.imported_nodes, topology)
        _, fes, _ = fe_space(still, support, excluded, space, order, comm=comm)
        memo[key] = (_mpi.agree_max(comm, int(fes.ndof)), int(fes.ndof))
    return memo[key]


def rank_piece(
    columns: np.ndarray, width: Callable[[], int], comm: Any, dtype: type = complex
) -> np.ndarray:
    """``columns`` padded with zero rows to ``width()``: one shape on all ranks.

    A rank's piece of a field. On one process (``comm`` ``None``)
    ``columns`` itself comes back and ``width`` is not called, since under ranks
    it may be a collective.
    """
    if comm is None:
        return columns
    piece = np.zeros((width(), columns.shape[1]), dtype)
    piece[: columns.shape[0]] = columns
    return piece


def _check_size(field: Field, fes: ngs.FESpace, width: int | None = None) -> None:
    """Raise if the field's size is not its space's: ``width`` under ranks."""
    expected = fes.ndof if width is None else width
    if expected != field.coefficients.shape[0]:
        raise _mpi.agreed(
            ValueError(
                f"The field has {field.coefficients.shape[0]} coefficients but its "
                f"{field.space} order-{field.order} space on {list(field.support)} "
                f"has {expected} DOFs: the field does not belong to this mesh."
            )
        )


def _holders(space: ngs.FESpace) -> np.ndarray:
    """Per local DOF of ``space``, the number of ranks holding it: one exchange."""
    ones = ngs.GridFunction(space).vec.CreateVector()
    ones.SetParallelStatus(ngs.la.PARALLEL_STATUS.DISTRIBUTED)
    ones.FV().NumPy()[:] = 1.0
    ones.Cumulate()
    return np.array(ones.FV().NumPy())


# --- one-process numbering --------------------------------------------------------

#: Vertex pairs of a tet's edges and vertex triples of its faces, with sorted rows.
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_TET_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

_MAPS: weakref.WeakKeyDictionary[Any, dict[tuple, np.ndarray]] = (
    weakref.WeakKeyDictionary()
)


def _entity_keys(tets: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """The edges and faces of ``tets``, each sorted and lexicographic, as integers.

    An edge is ``low * n_nodes + high``, a face its low edge's rank in the first
    array, then ``* n_nodes + highest``.
    """
    rows = np.sort(tets, axis=1)
    edges = rows[:, _TET_EDGES].reshape(-1, 2)
    edge_keys = np.unique(edges[:, 0] * n_nodes + edges[:, 1])
    faces = rows[:, _TET_FACES].reshape(-1, 3)
    low = np.searchsorted(edge_keys, faces[:, 0] * n_nodes + faces[:, 1])
    return edge_keys, np.unique(low * n_nodes + faces[:, 2])


def _one_process_tables(
    mesh: Mesh, support: tuple[str, ...], excluded: tuple[tuple[int, int], ...]
) -> dict[str, np.ndarray]:
    """One process's vertices, edges and faces of the support, and each tet's cell.

    ``vertices`` are the used gfem nodes, ascending; ``cells`` numbers each support
    tet, ``-1`` off the support.
    """
    names = mesh._region_names()  # noqa: SLF001
    domains, tet_domain = cell_domains(mesh)
    in_support = np.array(
        [
            False,
            *(
                names[int(tag)] in support and (int(tag), int(entity)) not in excluded
                for tag, entity in domains
            ),
        ]
    )
    used = in_support[tet_domain]
    edges, faces = _entity_keys(mesh.tetrahedra[used], len(mesh.imported_nodes))
    in_order = np.argsort(tet_domain, kind="stable")
    in_order = in_order[used[in_order]]
    cells = np.full(len(used), -1, dtype=np.int64)
    cells[in_order] = np.arange(len(in_order))
    vertices = np.unique(mesh.tetrahedra[used])
    return {"edges": edges, "faces": faces, "cells": cells, "vertices": vertices}


def one_process_dofs(
    mesh: Mesh,
    support: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
    order: int,
    comm: Any = None,
    *,
    space: str = "hcurl",
) -> np.ndarray:
    """Per local DOF of an H(curl) or H1 layout, its number on one process.

    NumPy on the replicated arrays, with no walk and no serial mesh. In a local mesh
    and in one process's alike, edges and faces are lexicographic in their gfem nodes.
    H(curl) DOFs are edges' lowest order, edges' higher orders, faces, then cells,
    each in its entities' order. H1 DOFs are the used vertices first, in local vertex
    order locally and by gfem node on one process, then ``p-1`` per edge,
    ``(p-1)(p-2)/2`` per face and ``(p-1)(p-2)(p-3)/6`` per cell. This rests on
    netgen's behaviour, so the counts are checked. Rank 0 builds the one-process
    tables, the expensive part. Memoized per topology, since a moved mesh has the
    same numbers. A miss is collective.

    Args:
        mesh: The field's mesh.
        support: The field's support regions.
        excluded: Its excluded volume entities.
        order: Its order.
        comm: The resolved communicator; ``None`` on one process.
        space: ``"hcurl"`` or ``"h1"``.

    Raises:
        gfem._mpi.AgreedError: The map's counts are not the space's.
    """
    topology = mesh._topology  # noqa: SLF001
    memo = _MAPS.setdefault(topology, {})
    key = (_mpi.comm_key(comm), tuple(support), tuple(excluded), order, space)
    if key in memo:
        return memo[key]
    still = Mesh(mesh.imported_nodes, topology)
    built, fes, _ = fe_space(still, support, excluded, space, order, comm=comm)
    tables = _mpi.on_root(comm, lambda: _one_process_tables(still, support, excluded))
    cells = tables["cells"]
    mine = built.cell_order
    mine = mine[cells[mine] >= 0]
    nodes, edges, faces = _local_entities(built, mesh, tables)
    if space == "h1":
        numbers, total = _h1_numbers(
            built,
            mesh,
            tables,
            nodes=nodes,
            mine=mine,
            edges=edges,
            faces=faces,
            order=order,
        )
    else:
        numbers, total = _hcurl_numbers(tables, cells[mine], edges, faces, order)
    if not _mpi.agree_all(
        comm, flag=len(numbers) == fes.ndof and total == fes.ndofglobal
    ):
        raise _mpi.AgreedError(
            f"This {_SPACE_NAMES[space]} space's DOF counts are not those of the map "
            "to one-process numbering, which rests on netgen's numbering. Report it; "
            "a wheel upgrade may have changed netgen."
        )
    memo[key] = numbers
    return numbers


_SPACE_NAMES = {"hcurl": "H(curl)", "h1": "H1"}

#: Vertex pairs of a triangle's edges, with sorted rows.
_TRIANGLE_EDGES = np.array([[0, 1], [0, 2], [1, 2]])


def _find(table: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each key's position in the sorted ``table``, and whether it is there."""
    at = np.searchsorted(table, keys)
    found = at < len(table)
    found[found] = table[at[found]] == keys[found]
    return at, found


def _local_entities(
    built: EngineMesh, mesh: Mesh, tables: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The gfem nodes, one-process edges and one-process faces holding local DOFs.

    Every support entity a rank holds carries its DOFs there, whichever local
    element brings it: a support tet, a support face netgen sent with the tet on
    its other side, or an edge or vertex of a tet off the support (DOFs are
    agreed across a shared entity's holders). Edges and faces come ascending,
    which is lexicographic in their gfem nodes.
    """
    n_nodes = len(mesh.imported_nodes)
    edge_table, face_table = tables["edges"], tables["faces"]
    tets = np.sort(mesh.tetrahedra[built.cell_order], axis=1)
    triangles = np.sort(np.asarray(mesh.triangles)[built.facet_order], axis=1)
    triples = np.concatenate([tets[:, _TET_FACES].reshape(-1, 3), triangles])
    pairs = triples[:, _TRIANGLE_EDGES].reshape(-1, 2)
    edges, on_edge = _find(edge_table, pairs[:, 0] * n_nodes + pairs[:, 1])
    low, on_low = _find(edge_table, triples[:, 0] * n_nodes + triples[:, 1])
    faces, on_face = _find(face_table, low * n_nodes + triples[:, 2])
    on_face &= on_low
    nodes = np.unique(triples)
    nodes = nodes[_find(tables["vertices"], nodes)[1]]
    return nodes, np.unique(edges[on_edge]), np.unique(faces[on_face])


def _blocks(first: int, entities: np.ndarray, per_entity: int) -> np.ndarray:
    """``per_entity`` consecutive numbers per entity, from ``first``."""
    return (first + entities[:, None] * per_entity + np.arange(per_entity)).ravel()


def _hcurl_numbers(
    tables: dict[str, np.ndarray],
    cells: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    order: int,
) -> tuple[np.ndarray, int]:
    """An H(curl) layout's one-process numbers, and one process's DOF count."""
    on_edge, on_face, in_cell = (
        order,
        order * (order - 1),
        order * (order - 1) * (order - 2) // 2,
    )
    n_edges, n_faces = len(tables["edges"]), len(tables["faces"])
    first_face = n_edges * on_edge
    first_cell = first_face + n_faces * on_face
    numbers = np.concatenate(
        [
            edges,
            _blocks(n_edges, edges, on_edge - 1),
            _blocks(first_face, faces, on_face),
            _blocks(first_cell, cells, in_cell),
        ]
    )
    total = int(first_cell + np.count_nonzero(tables["cells"] >= 0) * in_cell)
    return numbers, total


def _h1_numbers(
    built: EngineMesh,
    mesh: Mesh,
    tables: dict[str, np.ndarray],
    *,
    nodes: np.ndarray,
    mine: np.ndarray,
    edges: np.ndarray,
    faces: np.ndarray,
    order: int,
) -> tuple[np.ndarray, int]:
    """An H1 layout's one-process numbers, and one process's DOF count.

    The local vertex order need not follow gfem nodes, so each used local vertex
    (of ``nodes``, gfem nodes) finds its gfem node among one process's used
    vertices.
    """
    vertex = np.full(len(mesh.imported_nodes), -1, dtype=np.int64)
    vertex[built.node_order] = np.arange(len(built.node_order))
    used = np.unique(vertex[nodes])
    on_edge = order - 1
    on_face = (order - 1) * (order - 2) // 2
    in_cell = (order - 1) * (order - 2) * (order - 3) // 6
    n_vertices = len(tables["vertices"])
    first_face = n_vertices + len(tables["edges"]) * on_edge
    first_cell = first_face + len(tables["faces"]) * on_face
    numbers = np.concatenate(
        [
            np.searchsorted(tables["vertices"], built.node_order[used]),
            _blocks(n_vertices, edges, on_edge),
            _blocks(first_face, faces, on_face),
            _blocks(first_cell, tables["cells"][mine], in_cell),
        ]
    )
    total = int(first_cell + np.count_nonzero(tables["cells"] >= 0) * in_cell)
    return numbers, total


def one_process_column(field: Field) -> np.ndarray | None:
    """A piece's whole column in one-process numbering on rank 0, ``None`` elsewhere.

    Collective. A DOF several ranks hold takes the last of their values.
    """
    comm = field._communicator  # noqa: SLF001
    numbers = one_process_dofs(
        field.mesh,
        field.support,
        field._excluded,  # noqa: SLF001
        field.order,
        comm,
        space=field.space,
    )
    positions = _mpi.gather_rows(comm, numbers)
    values = _mpi.gather_rows(comm, np.asarray(field.coefficients)[: len(numbers)])
    if positions is None or values is None:
        return None
    column = np.empty(positions.max(initial=-1) + 1, values.dtype)
    column[positions] = values
    return column


# --- sampling ------------------------------------------------------------------


def _lattice(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Points and sub-tetrahedra of the reference tetrahedron cut ``n`` times.

    Returns the barycentric lattice points ``(i, j, k) / n`` (reference
    coordinates, ``i + j + k <= n``) and ``n**3`` positively oriented
    sub-tetrahedra as point indices.
    """
    ijk = [
        (i, j, k)
        for k in range(n + 1)
        for j in range(n + 1 - k)
        for i in range(n + 1 - k - j)
    ]
    index = {p: m for m, p in enumerate(ijk)}

    def at(i, j, k):
        return index[i, j, k]

    tets = []
    for i, j, k in ijk:
        s = i + j + k
        if s <= n - 1:
            corner = [at(i, j, k), at(i + 1, j, k), at(i, j + 1, k), at(i, j, k + 1)]
            tets.append(corner)
        if s <= n - 2:
            # the octahedron between the corner tetrahedra, cut along one diagonal
            a, b = at(i + 1, j, k), at(i, j + 1, k + 1)
            ring = [
                at(i, j + 1, k),
                at(i + 1, j + 1, k),
                at(i + 1, j, k + 1),
                at(i, j, k + 1),
            ]
            tets += [[a, b, ring[m], ring[(m + 1) % 4]] for m in range(4)]
        if s <= n - 3:
            tets.append(
                [
                    at(i + 1, j + 1, k),
                    at(i + 1, j, k + 1),
                    at(i, j + 1, k + 1),
                    at(i + 1, j + 1, k + 1),
                ]
            )
    points = np.array(ijk, dtype=float) / n
    cells = _orient(points, np.array(tets, dtype=np.int64))
    return points, cells


def _triangle_lattice(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Points and sub-triangles of the reference triangle cut ``n`` times.

    Returns the lattice points ``(i, j) / n`` (reference coordinates,
    ``i + j <= n``) and ``n**2`` counterclockwise sub-triangles as point indices.
    """
    ij = [(i, j) for j in range(n + 1) for i in range(n + 1 - j)]
    index = {p: m for m, p in enumerate(ij)}
    tris = []
    for i, j in ij:
        if i + j <= n - 1:
            tris.append([index[i, j], index[i + 1, j], index[i, j + 1]])
        if i + j <= n - 2:
            tris.append([index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]])
    return np.array(ij, dtype=float) / n, np.array(tris, dtype=np.int64)


def _orient_triangles(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """``cells`` with the last two vertices swapped where the area is < 0."""
    p = points[cells]
    d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    out = cells.copy()
    flip = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0] < 0
    out[flip, 1], out[flip, 2] = cells[flip, 2], cells[flip, 1]
    return out


def _orient(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """``cells`` with the last two vertices swapped where the volume is < 0."""
    p = points[cells]
    vol = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    out = cells.copy()
    flip = vol < 0
    out[flip, 2], out[flip, 3] = cells[flip, 3], cells[flip, 2]
    return out


def sample(
    field: Field,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """The field at the vertices of its support's tetrahedra cut ``order`` times.

    On a 2D mesh, of its support's triangles, the points at z = 0.

    Across ranks this is collective: each rank samples its own tetrahedra, and rank 0
    gets them in one process's order, the others ``None``.

    Returns:
        ``(points (N, 3) in m, cells (M, dim + 1), point_data, cell_data)``;
        the points of each element are its own (no sharing).
    """
    mesh = field.mesh
    excluded = field._excluded  # noqa: SLF001
    comm = field._communicator  # noqa: SLF001
    built, fes, region = fe_space(
        mesh, field.support, excluded, field.space, field.order, comm=comm
    )
    width = None
    if comm is not None:
        width = dof_shape(
            mesh, field.support, excluded, field.space, field.order, comm=comm
        )[0]
    _check_size(field, fes, width)
    n = max(field.order, 2 if mesh.curved else 1)
    flat = mesh.dim == 2
    ref, sub = _triangle_lattice(n) if flat else _lattice(n)
    rule = ngs.IntegrationRule([tuple(p) for p in ref], [0.0] * len(ref))
    mips = built.mesh.MapToAllElements(rule, region)
    # a rank with no element reads empty arrays with no second axis
    position = (ngs.x, ngs.y, 0) if flat else (ngs.x, ngs.y, ngs.z)
    xyz = np.asarray(ngs.CF(position)(mips)).reshape(-1, 3) * built.scale

    centre = ngs.IntegrationRule(
        [(1 / 3, 1 / 3)] if flat else [(0.25, 0.25, 0.25)], [0.0]
    )
    topology = mesh._topology  # noqa: SLF001
    owner = p0(built, np.arange(len(topology.volume_elements), dtype=float))
    tet = np.rint(
        np.asarray(owner(built.mesh.MapToAllElements(centre, region))).reshape(-1)
    ).astype(np.int64)

    coefficients = np.asarray(field.coefficients)[: fes.ndof]
    factor = built.scale ** LENGTH_POWER[field.space]
    components = 3 if field.space in ("hcurl", "hdiv") else 1
    parts = (
        {f"{field.quantity}_real": coefficients.real}
        | {f"{field.quantity}_imag": coefficients.imag}
        if np.iscomplexobj(coefficients)
        else {field.quantity: coefficients}
    )
    point_data = {}
    gf = ngs.GridFunction(fes)
    for name, values in parts.items():
        gf.vec.FV().NumPy()[:] = values
        sampled = np.asarray(gf(mips)).reshape(-1, components) * factor
        point_data[name] = sampled[:, 0] if components == 1 else sampled
    if comm is not None:
        gathered = _gather_sample(comm, mesh, tet, {"points": xyz, **point_data})
        if gathered is None:
            return None
        tet, point_data = gathered
        xyz = point_data.pop("points")
    n_el = len(tet)
    cells = (sub[None] + len(ref) * np.arange(n_el)[:, None, None]).reshape(
        -1, sub.shape[1]
    )
    cells = _orient_triangles(xyz, cells) if flat else _orient(xyz, cells)
    cell_data = {
        "region": np.repeat(topology.volume_tags[tet], len(sub)),
        "entity": np.repeat(topology.volume_entities[tet], len(sub)),
    }
    return xyz, cells, point_data, cell_data


def _gather_sample(
    comm: Any, mesh: Mesh, tet: np.ndarray, rows: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Each rank's samples on rank 0, ordered as one process orders its elements.

    Args:
        comm: The resolved communicator.
        mesh: The field's mesh.
        tet: This rank's tetrahedra.
        rows: Arrays of an equal number of rows per tetrahedron, by name.

    Returns:
        The tetrahedra and ``rows`` of all ranks on rank 0; ``None`` elsewhere.
    """
    tets = _mpi.gather_rows(comm, tet)
    gathered = {name: _mpi.gather_rows(comm, values) for name, values in rows.items()}
    if tets is None:
        return None
    _, tet_domain = cell_domains(mesh)
    position = inverse_order(np.argsort(tet_domain, kind="stable"), len(tet_domain))
    in_order = np.argsort(position[tets])

    def ordered(values: Any) -> np.ndarray:
        by_tet = values.reshape(len(tets), -1, *values.shape[1:])
        return by_tet[in_order].reshape(values.shape)

    return tets[in_order], {k: ordered(v) for k, v in gathered.items()}


# --- curl ------------------------------------------------------------------------

_CURLS: collections.OrderedDict[tuple, tuple] = collections.OrderedDict()
_MAX_CURLS = 4


def curl_matrix(field: Field) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """The map from H(curl) coefficients to the H(div) coefficients of the curl.

    Exact: H(div) of order ``max(p - 1, 0)`` contains the curl of first-kind
    Nédélec of order ``p``. For affine elements the map depends only on the
    mesh topology (the Piola maps cancel), so it is built at the imported
    node positions and serves moved and traced meshes alike.

    Across ranks it is this rank's part, and a miss is collective. Each row of
    ``local_mat`` is multiplied by the number of ranks holding its H(div) DOF: only
    that matched one process, ``Cumulate`` did not. ``n_rows`` is then the
    seam's width, the same on every rank.

    Returns:
        ``(rows, cols, values, n_rows)`` of the sparse matrix.
    """
    mesh = field.mesh
    topology = mesh._topology  # noqa: SLF001
    comm = field._communicator  # noqa: SLF001
    key = (
        id(topology),
        field.support,
        field._excluded,  # noqa: SLF001
        field.order,
        _mpi.comm_key(comm),
    )
    found = _CURLS.get(key)
    if found is not None and found[0] is topology:
        _CURLS.move_to_end(key)
        return found[1]
    still = Mesh(mesh.imported_nodes, topology)
    support, excluded = field.support, field._excluded  # noqa: SLF001
    order = max(field.order - 1, 0)
    _, hcurl, _ = fe_space(still, support, excluded, "hcurl", field.order, comm=comm)
    width = dof_shape(still, support, excluded, "hcurl", field.order, comm=comm)[0]
    _check_size(field, hcurl, width)
    _, hdiv, _ = fe_space(still, support, excluded, "hdiv", order, comm=comm)
    op = ngs.ConvertOperator(
        spacea=hcurl, spaceb=hdiv, trial_proxy=ngs.curl(hcurl.TrialFunction())
    )
    rows, cols, values = (op if comm is None else op.local_mat).COO()
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if comm is not None:
        values = values * _holders(hdiv)[rows]
        if rows.size == 0:
            # A rank with no entry would read no coefficient and no cotangent, and
            # under jit JAX drops a callback whose outputs go unread on that rank
            # alone: its peers then wait for it. A zero entry reads them.
            rows, cols = np.zeros(1, np.int64), np.zeros(1, np.int64)
            values = np.zeros(1)
    matrix = (
        rows,
        cols,
        values,
        dof_shape(still, support, excluded, "hdiv", order, comm=comm)[0],
    )
    _CURLS[key] = (topology, matrix)
    while len(_CURLS) > _MAX_CURLS:
        _CURLS.popitem(last=False)
    return matrix


# --- region means -----------------------------------------------------------------

_MEANS: collections.OrderedDict[tuple, tuple] = collections.OrderedDict()
_MAX_MEANS = 8


def mean_rows(
    mesh: Mesh,
    layout: tuple[tuple[str, ...], tuple[tuple[int, int], ...], str, int],
    regions: tuple[str, ...],
) -> np.ndarray:
    """``(components, ndof)``: a field's mean over ``regions`` is ``rows @ c``.

    Row ``k`` is |R|^-1 int_R phi_j . e_k over the regions' elements in the
    support (``R``), in SI units (``scale ** LENGTH_POWER``): exact for
    the discrete field, since the integrand is a polynomial of the space's
    order on each straight-sided element. It depends on the node positions,
    which ``mesh`` holds as numbers. ``layout`` is the field's (support,
    excluded entities, space, order). One process only.
    """
    support, excluded, space, order = layout
    topology = mesh._topology  # noqa: SLF001
    nodes = np.ascontiguousarray(mesh.nodes)
    key = (
        id(topology),
        support,
        excluded,
        space,
        order,
        regions,
        hash(nodes.tobytes()),
    )
    found = _MEANS.get(key)
    if found is not None and found[0] is topology:
        _MEANS.move_to_end(key)
        return found[1]
    built, fes, _ = fe_space(mesh, support, excluded, space, order)
    inside = tuple(r for r in regions if r in support)
    region = support_region(built, mesh, inside, excluded)
    dx = quadrature.dx(order, definedon=region)
    volume = ngs.Integrate(ngs.CF(1.0) * dx, built.mesh)
    v = fes.TestFunction()
    components = [v] if v.dim == 1 else [v[k] for k in range(v.dim)]
    rows = np.zeros((len(components), fes.ndof))
    for k, part in enumerate(components):
        form = ngs.LinearForm(fes)
        form += part * dx
        form.Assemble()
        rows[k] = form.vec.FV().NumPy()
    rows *= built.scale ** LENGTH_POWER[space] / volume
    _MEANS[key] = (topology, rows)
    while len(_MEANS) > _MAX_MEANS:
        _MEANS.popitem(last=False)
    return rows


_ELEMENT_ROWS: collections.OrderedDict[tuple, tuple] = collections.OrderedDict()


def element_rows(
    mesh: Mesh,
    layout: tuple[tuple[str, ...], tuple[tuple[int, int], ...], str, int],
    tetrahedra: np.ndarray,
) -> Any:
    """``(len(tetrahedra), ndof)`` sparse rows: the element means are ``rows @ c``.

    Row ``i`` is |e|^-1 int_e phi_j over tetrahedron ``tetrahedra[i]`` (gfem
    order): exact for the discrete field, on the pinned rule of its order.
    Every listed tetrahedron must lie in the field's support. ``layout`` is
    the field's (support, excluded entities, space, order); a scalar space
    (h1, l2). One process only.
    """
    import scipy.sparse

    support, excluded, space, order = layout
    topology = mesh._topology  # noqa: SLF001
    nodes = np.ascontiguousarray(mesh.nodes)
    tetrahedra = np.asarray(tetrahedra, np.int64)
    key = (
        id(topology),
        support,
        excluded,
        space,
        order,
        hash(tetrahedra.tobytes()),
        hash(nodes.tobytes()),
    )
    found = _ELEMENT_ROWS.get(key)
    if found is not None and found[0] is topology:
        _ELEMENT_ROWS.move_to_end(key)
        return found[1]
    built, fes, _ = fe_space(mesh, support, excluded, space, order)
    l2 = ngs.L2(built.mesh, order=0)
    dx = quadrature.dx(order)
    q = l2.TestFunction()
    form = ngs.BilinearForm(trialspace=fes, testspace=l2)
    form += fes.TrialFunction() * q * dx
    form.Assemble()
    volume = ngs.LinearForm(l2)
    volume += q * dx
    volume.Assemble()
    rows, cols, values = form.mat.COO()
    element = np.asarray(rows, np.int64)
    matrix = scipy.sparse.csr_matrix(
        (np.asarray(values) / volume.vec.FV().NumPy()[element], (element, cols)),
        shape=(l2.ndof, fes.ndof),
    )
    # NGSolve element i is gfem tetrahedron cell_order[i]
    of_tet = np.empty(len(built.cell_order), np.int64)
    of_tet[built.cell_order] = np.arange(len(built.cell_order))
    out = matrix[of_tet[tetrahedra]]
    _ELEMENT_ROWS[key] = (topology, out)
    while len(_ELEMENT_ROWS) > _MAX_MEANS:
        _ELEMENT_ROWS.popitem(last=False)
    return out
