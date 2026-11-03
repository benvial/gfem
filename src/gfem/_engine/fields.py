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
from gfem._engine.cache import discretization
from gfem._engine.coefficients import p0
from gfem._engine.mesh import EngineMesh, inverse_order, tetrahedron_domains
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._field import Field

__all__ = [
    "LENGTH_POWER",
    "curl_matrix",
    "dof_shape",
    "fe_space",
    "one_process_column",
    "one_process_dofs",
    "sample",
    "support_region",
]

LENGTH_POWER = {"h1": 0, "l2": 0, "hcurl": -1, "hdiv": -2}


def support_region(
    built: EngineMesh,
    mesh: Mesh,
    support: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
) -> ngs.Region:
    """The NGSolve volume region of ``support`` minus ``excluded`` entities."""
    names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    wanted, out = set(support), set(excluded)
    bits = ngs.BitArray(len(built.domains))
    bits.Clear()
    for k, (tag, entity) in enumerate(built.domains):
        if names[tag] in wanted and (tag, entity) not in out:
            bits.Set(k)
    return ngs.Region(built.mesh, ngs.VOL, bits)


def _build(region: ngs.Region, space: str, order: int) -> ngs.FESpace:
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
    read outputs are all empty does not run on that rank (ADR 0063); on one process
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
    """One process's edges and faces of the support, and each tet's cell number."""
    names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    domains, tet_domain = tetrahedron_domains(mesh)
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
    return {"edges": edges, "faces": faces, "cells": cells}


def one_process_dofs(
    mesh: Mesh,
    support: tuple[str, ...],
    excluded: tuple[tuple[int, int], ...],
    order: int,
    comm: Any = None,
) -> np.ndarray:
    """Per local DOF of an H(curl) layout, its number on one process (ADR 0064).

    NumPy on the replicated arrays, with no walk and no serial mesh. In a local mesh
    and in one process's alike, edges and faces are lexicographic in their gfem nodes,
    and DOFs are edges' lowest order, edges' higher orders, faces, then cells, each in
    its entities' order. This rests on netgen's behaviour, so the counts are checked.
    Rank 0 builds the one-process tables, the expensive part. Memoized per topology,
    since a moved mesh has the same numbers. A miss is collective.

    Raises:
        gfem._mpi.AgreedError: The map's counts are not the space's.
    """
    topology = mesh._topology  # noqa: SLF001
    memo = _MAPS.setdefault(topology, {})
    key = (_mpi.comm_key(comm), tuple(support), tuple(excluded), order)
    if key in memo:
        return memo[key]
    still = Mesh(mesh.imported_nodes, topology)
    built, fes, _ = fe_space(still, support, excluded, "hcurl", order, comm=comm)
    tables = _mpi.on_root(comm, lambda: _one_process_tables(still, support, excluded))
    cells = tables["cells"]
    mine = built.tetrahedron_order
    mine = mine[cells[mine] >= 0]
    n_nodes = len(mesh.imported_nodes)
    edge_keys, face_keys = _entity_keys(mesh.tetrahedra[mine], n_nodes)
    # a local face's key holds a local edge's rank: it moves to one process's rank
    edges = np.searchsorted(tables["edges"], edge_keys)
    faces = np.searchsorted(
        tables["faces"], edges[face_keys // n_nodes] * n_nodes + face_keys % n_nodes
    )
    on_edge, on_face, in_cell = (
        order,
        order * (order - 1),
        order * (order - 1) * (order - 2) // 2,
    )
    n_edges, n_faces = len(tables["edges"]), len(tables["faces"])
    high = n_edges + edges[:, None] * (on_edge - 1) + np.arange(on_edge - 1)
    first_face = n_edges * on_edge
    first_cell = first_face + n_faces * on_face
    numbers = np.concatenate(
        [
            edges,
            high.ravel(),
            (first_face + faces[:, None] * on_face + np.arange(on_face)).ravel(),
            (first_cell + cells[mine][:, None] * in_cell + np.arange(in_cell)).ravel(),
        ]
    )
    total = first_cell + np.count_nonzero(cells >= 0) * in_cell
    if not _mpi.agree_all(
        comm, flag=len(numbers) == fes.ndof and total == fes.ndofglobal
    ):
        raise _mpi.AgreedError(
            "This H(curl) space's DOF counts are not those of the map to one-process "
            "numbering, which rests on netgen's numbering. Report it; a wheel upgrade "
            "may have changed netgen (ADR 0064)."
        )
    memo[key] = numbers
    return numbers


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

    Across ranks this is collective: each rank samples its own tetrahedra, and rank 0
    gets them in one process's order, the others ``None``.

    Returns:
        ``(points (N, 3) in m, cells (M, 4), point_data, cell_data)``; the
        points of each tetrahedron are its own (no sharing).
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
    n = max(field.order, 1)
    ref, sub = _lattice(n)
    rule = ngs.IntegrationRule([tuple(p) for p in ref], [0.0] * len(ref))
    mips = built.mesh.MapToAllElements(rule, region)
    # a rank with no element reads empty arrays with no second axis
    xyz = np.asarray(ngs.CF((ngs.x, ngs.y, ngs.z))(mips)).reshape(-1, 3) * built.scale

    centre = ngs.IntegrationRule([(0.25, 0.25, 0.25)], [0.0])
    owner = p0(built, np.arange(len(mesh.tetrahedra), dtype=float))
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
    cells = (sub[None] + len(ref) * np.arange(n_el)[:, None, None]).reshape(-1, 4)
    cells = _orient(xyz, cells)
    cell_data = {
        "region": np.repeat(mesh.tetrahedron_tags[tet], len(sub)),
        "entity": np.repeat(mesh.tetrahedron_entities[tet], len(sub)),
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
    _, tet_domain = tetrahedron_domains(mesh)
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
    that matched one process, ``Cumulate`` did not (ADR 0063). ``n_rows`` is then the
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
            # alone: its peers then wait for it (ADR 0063). A zero entry reads them.
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
