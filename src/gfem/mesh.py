"""Mesh import (gmsh ``.msh`` 2.2 ASCII and 4.1) and `Mesh`."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import weakref
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np

from gfem import _mpi
from gfem._checks import is_traced
from gfem._pytree import readonly
from gfem.errors import CurvedMeshUnsupportedError, RegionNotFoundError

#: Supported ``length_unit`` values and their size in metres.
LENGTH_UNITS = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}

#: What a failed read raises on one process, so on every rank above one.
_READ_ERRORS = (ValueError, OSError, meshio.ReadError)

_TOPOLOGY_STATICS = ("length_unit", "regions", "source", "section_thickness", "dim")
#: Every array field of a `_Topology`, by role: the volume elements (the
#: highest dimension) and the surface elements (one below) with their gmsh
#: numbers and the volume elements on each side of each surface element.
_ROLE_ARRAYS = (
    "imported_nodes",
    "volume_elements",
    "volume_tags",
    "volume_entities",
    "surface_elements",
    "surface_tags",
    "surface_entities",
    "surface_neighbours",
)
#: The names of a tetrahedral mesh's arrays, `_ROLE_ARRAYS` in order: what a
#: dimension-3 mesh with no file writes and what its digest hashes. They are
#: the names these arrays had before meshes had a dimension, so job files and
#: digests of tetrahedral meshes stay as they were.
_TOPOLOGY_ARRAYS = (
    "imported_nodes",
    "tetrahedra",
    "tet_tags",
    "tet_entities",
    "triangles",
    "tri_tags",
    "tri_entities",
    "tri_neighbours",
)

#: The mid-edge node arrays of a curved mesh, as (stored name, `_Topology` field):
#: a straight mesh has none, so its arrays, digest and job files are the same
#: as before curved meshes existed.
_MIDNODE_ARRAYS = (
    ("tet_midnodes", "volume_midnodes"),
    ("tri_midnodes", "surface_midnodes"),
)

# Local face k of a tetrahedron is the one opposite its vertex k.
_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
# Local vertex pairs of a tetrahedron's six edges.
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
# Local edge k of a triangle is the one opposite its vertex k.
_TRI_EDGES = np.array([[1, 2], [0, 2], [0, 1]])

#: The end faces a cross-section's slab gains, at the section's plane and one
#: thickness along +z.
END_FACES = ("front", "back")


@dataclasses.dataclass(frozen=True)
class Region:
    """A named volume or surface of the mesh: one gmsh physical group.

    A region whose dimension is the mesh's is a volume, one a dimension lower
    a surface: on a 2D mesh the areas are its volumes and the lines its
    surfaces.

    Attributes:
        name: The gmsh physical-group name, an opaque string.
        tag: The gmsh physical-group number.
        dim: The region's dimension: 3 for a volume and 2 for a surface of a
            tetrahedral mesh, 2 for an area and 1 for a line of a 2D mesh.
    """

    name: str
    tag: int
    dim: int


@dataclasses.dataclass(frozen=True, eq=False)
class _Topology:
    """Everything about a mesh except its node coordinates.

    This is the static part of the `Mesh` pytree. It compares by identity, so
    a jitted function retraces only when it sees a different imported mesh.
    Its arrays are named by role (`_ROLE_ARRAYS`): tetrahedra and triangles
    on a mesh of dimension 3, triangles and line segments on one of dimension 2.
    """

    length_unit: str
    imported_nodes: np.ndarray
    regions: tuple[Region, ...]
    volume_elements: np.ndarray
    volume_tags: np.ndarray
    volume_entities: np.ndarray
    surface_elements: np.ndarray
    surface_tags: np.ndarray
    surface_entities: np.ndarray
    surface_neighbours: np.ndarray
    source: Path | None = None
    section_thickness: float | None = None
    dim: int = 3
    # Mid-edge nodes of a curved mesh: column k of ``volume_midnodes`` (n, 6) is
    # the node on edge `_TET_EDGES`[k] of the tetrahedron, column k of
    # ``surface_midnodes`` (n, 3) the one on the edge opposite vertex k of the
    # triangle. Both are None on a straight mesh.
    volume_midnodes: np.ndarray | None = None
    surface_midnodes: np.ndarray | None = None


def _array_names(dim: int) -> tuple[str, ...]:
    """The names a topology of dimension ``dim`` stores its arrays under."""
    return _TOPOLOGY_ARRAYS if dim == 3 else _ROLE_ARRAYS


def _topology_arrays(topology: _Topology) -> dict[str, np.ndarray]:
    """A topology's arrays by stored name (`_array_names`), in that order.

    A curved mesh's mid-edge node arrays (`_MIDNODE_ARRAYS`) come last.
    """
    arrays = {
        name: getattr(topology, field)
        for name, field in zip(_array_names(topology.dim), _ROLE_ARRAYS, strict=True)
    }
    for name, field in _MIDNODE_ARRAYS:
        if getattr(topology, field) is not None:
            arrays[name] = getattr(topology, field)
    return arrays


def _topology_from_arrays(arrays, **statics) -> _Topology:
    """The topology of `_topology_arrays` and the statics (`_TOPOLOGY_STATICS`)."""
    names = _array_names(statics.get("dim", 3))
    fields = {
        field: arrays[name] for name, field in zip(names, _ROLE_ARRAYS, strict=True)
    }
    fields |= {field: arrays[name] for name, field in _MIDNODE_ARRAYS if name in arrays}
    return _Topology(**statics, **fields)


_TOPOLOGY_DIGESTS: weakref.WeakKeyDictionary[_Topology, str] = (
    weakref.WeakKeyDictionary()
)


def _topology_sha256(topology: _Topology) -> str:
    """The sha256 (64 hex) of a topology's arrays, computed once per topology.

    Each array enters as `_hash_array` feeds it, under its stored name, in the
    order of `_topology_arrays`; a mesh of another dimension than 3 hashes its
    dimension first. The topology is immutable, so the digest is memoized on it.
    """
    digest = _TOPOLOGY_DIGESTS.get(topology)
    if digest is None:
        h = hashlib.sha256()
        if topology.dim != 3:
            _hash_array(h, "dim", np.int64(topology.dim))
        for name, array in _topology_arrays(topology).items():
            _hash_array(h, name, array)
        digest = h.hexdigest()
        _TOPOLOGY_DIGESTS[topology] = digest
    return digest


def _mesh_sha256(mesh: Mesh) -> str:
    """The topology digest of a mesh (`_topology_sha256`); its nodes do not enter."""
    return _topology_sha256(mesh._topology)  # noqa: SLF001


def _at_imported_nodes(mesh: Mesh) -> Mesh:
    """``mesh`` at its imported nodes: the same topology, no traced values."""
    return Mesh(mesh.imported_nodes, mesh._topology)  # noqa: SLF001


def _hash_array(h, path: str, array) -> None:
    """Feed one array to the hash ``h``, independent of its memory layout.

    A length-prefixed JSON header ``[path, dtype.str, shape]`` comes first,
    then the little-endian C-order bytes. Float and complex values are
    normalized first: :math:`-0` becomes :math:`+0` (``a + 0``) and every NaN
    one quiet NaN.
    """
    a = np.asarray(array)
    if a.dtype.kind in "fc":
        a = a + a.dtype.type(0)
        if a.dtype.kind == "f":
            a = np.where(np.isnan(a), a.dtype.type(np.nan), a)
        else:
            nan = a.real.dtype.type(np.nan)
            re = np.where(np.isnan(a.real), nan, a.real)
            im = np.where(np.isnan(a.imag), nan, a.imag)
            a = np.empty(a.shape, a.dtype)
            a.real, a.imag = re, im
    a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"), copy=False))
    head = json.dumps([path, a.dtype.str, list(a.shape)]).encode()
    h.update(len(head).to_bytes(8, "little"))
    h.update(head)
    h.update(memoryview(a.reshape(-1)).cast("B"))


def _read_gmsh(path: Path, length_unit: str) -> _Topology:
    """Parse the file: what `Mesh.from_gmsh` does on one process."""
    _check_unit(length_unit)
    raw = meshio.read(path, file_format="gmsh")
    if _is_plane(raw):
        return _read_plane(raw, path, length_unit)
    regions = _regions(raw)
    cells = _cells(raw, path)
    tets, tet_tags, tet_entities = cells["tetra"]
    tris, tri_tags, tri_entities = cells["triangle"]
    tet_mid, tri_mid = cells.get("midnodes", (None, None))
    points = np.asarray(raw.points, dtype=float)
    neighbours = _triangle_neighbours(points, tets, tris, path)
    topology = _Topology(
        length_unit=length_unit,
        imported_nodes=readonly(points * LENGTH_UNITS[length_unit]),
        regions=regions,
        volume_elements=readonly(tets),
        volume_tags=readonly(tet_tags),
        volume_entities=readonly(tet_entities),
        surface_elements=readonly(tris),
        surface_tags=readonly(tri_tags),
        surface_entities=readonly(tri_entities),
        surface_neighbours=readonly(neighbours),
        source=path.resolve(),
        volume_midnodes=None if tet_mid is None else readonly(tet_mid),
        surface_midnodes=None if tri_mid is None else readonly(tri_mid),
    )
    _check_tags(topology, path)
    if tet_mid is not None:
        _check_curved_quality(topology, path)
    return topology


def _check_curved_quality(topology: _Topology, path: Path) -> None:
    """Refuse a file whose curved tetrahedra are folded (`gfem.shape` measure)."""
    from gfem.shape import _check_curved_read  # gfem.shape imports this module

    _check_curved_read(Mesh(jnp.asarray(topology.imported_nodes), topology), path)


def _is_plane(raw: meshio.Mesh) -> bool:
    """Whether a file holds a 2D mesh: triangles and no 3D element."""
    types = {block.type for block in raw.cells}
    return bool(types & {"triangle", "triangle6"}) and not types & set(_VOLUME_TYPES)


def _read_plane(raw: meshio.Mesh, path: Path, length_unit: str) -> _Topology:
    """A triangle-only file as a mesh of dimension 2 (`Mesh.from_gmsh`).

    The same checks as a cross-section's, then z is dropped: the nodes are
    (n, 2), the areas its volumes and the lines its surfaces.
    """
    regions = _regions(raw, top=2)
    section = _read_section(
        raw,
        path,
        length_unit,
        {r.tag: r.name for r in regions if r.dim == 1},
        kind="2D mesh",
    )
    nodes = np.ascontiguousarray(section.points[:, :2])
    neighbours = _segment_neighbours(nodes, section.tris, section.lines, path)
    topology = _Topology(
        length_unit=length_unit,
        imported_nodes=readonly(nodes),
        regions=regions,
        volume_elements=readonly(section.tris),
        volume_tags=readonly(section.area_tags),
        volume_entities=readonly(section.area_entities),
        surface_elements=readonly(section.lines),
        surface_tags=readonly(section.line_tags),
        surface_entities=readonly(section.line_entities),
        surface_neighbours=readonly(neighbours),
        source=path.resolve(),
        dim=2,
    )
    _check_tags(topology, path)
    return topology


def _check_unit(length_unit: str) -> None:
    if length_unit not in LENGTH_UNITS:
        raise ValueError(
            f"Unknown length_unit {length_unit!r}; use one of "
            f"{', '.join(repr(u) for u in LENGTH_UNITS)}."
        )


def _check_thickness(thickness) -> float:
    try:
        value = float(thickness)
    except (TypeError, ValueError):
        value = float("nan")
    if not (np.isfinite(value) and value > 0):
        raise ValueError(
            f"thickness must be a positive length in metres, got {thickness!r}; "
            "leave it None to let gfem pick one."
        )
    return value


def _check_section_plane(points: np.ndarray, path: Path, kind: str) -> None:
    """The section (a ``kind``) must lie in a plane z = constant."""
    extent = np.ptp(points[:, :2], axis=0).max()
    if np.ptp(points[:, 2]) > 1e-9 * extent:
        it = "the section" if kind == "cross-section" else "it"
        raise ValueError(
            f"{path.name}: the {kind}'s nodes do not lie in one plane "
            f"z = constant. Mesh {it} in the xy plane in gmsh."
        )


def _check_lines_on_edges(
    lines: np.ndarray,
    tags: np.ndarray,
    edges: np.ndarray,
    names: dict[int, str],
    path: Path,
) -> None:
    """Every grouped line must be an edge of the section's triangles.

    ``names`` names the line groups by gmsh number.
    """
    off = (lines < 0).any(axis=1)
    known = {tuple(e) for e in edges.tolist()}
    off |= np.array([tuple(sorted(e)) not in known for e in lines.tolist()], bool)
    if off.any():
        group = names.get(int(tags[np.flatnonzero(off)[0]]))
        raise ValueError(
            f"{path.name}: a line of the 1D physical group {group!r} is not an edge "
            "of the section's triangles, so the lines and the triangles do not "
            "conform. Mesh the lines as part of the 2D mesh in gmsh."
        )


def _default_thickness(points: np.ndarray, edges: np.ndarray) -> float:
    """The mean in-plane edge length: well-shaped tetrahedra."""
    return float(
        np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1).mean()
    )


def _cut_rank(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Each section node's place in (x, y) order: by x, then y.

    Coordinates are compared on a grid of a millionth of the shortest edge, so
    round-off cannot reorder nodes that share an x. The order depends on the
    coordinates only, and a translation keeps it, so a section meshed
    periodically cuts its paired edges alike whatever its node numbering.
    """
    h = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1).min()
    key = np.round(points[:, :2] / (1e-6 * h)).astype(np.int64)
    rank = np.empty(len(points), dtype=np.int64)
    rank[np.lexsort((key[:, 1], key[:, 0]))] = np.arange(len(points))
    return rank


def _split_prisms(
    tris: np.ndarray, n: int, nodes: np.ndarray, rank: np.ndarray
) -> np.ndarray:
    """Three positive tetrahedra per prism over each triangle, rows per triangle.

    Each quadrilateral side is cut along the diagonal from its node of lowest
    ``rank`` (`_cut_rank`), so two prisms that share a side cut it alike and the
    tetrahedra conform, and two sides that are translates cut alike too.
    """
    a, b, c = np.take_along_axis(tris, np.argsort(rank[tris], axis=1), axis=1).T
    tets = np.stack(
        [
            np.stack([a, b, c, c + n], axis=1),
            np.stack([a, b, b + n, c + n], axis=1),
            np.stack([a, a + n, b + n, c + n], axis=1),
        ],
        axis=1,
    ).reshape(-1, 4)
    x = nodes[tets]
    e = x[:, 1:] - x[:, :1]
    negative = np.linalg.det(e) < 0
    tets[negative] = tets[negative][:, [0, 2, 1, 3]]
    return tets


def _split_quads(lines: np.ndarray, n: int, rank: np.ndarray) -> np.ndarray:
    """Two triangles per extruded line, cut as `_split_prisms` cuts the sides.

    Each triangle's normal is the line's direction crossed with +z, so a
    consistently oriented 1D group gives a consistently oriented surface.
    """
    p, q = lines.T
    low = rank[p] < rank[q]
    first = np.where(
        low[:, None], np.stack([p, q, q + n], 1), np.stack([p, q, p + n], 1)
    )
    second = np.where(
        low[:, None], np.stack([p, q + n, p + n], 1), np.stack([q, q + n, p + n], 1)
    )
    return np.stack([first, second], axis=1).reshape(-1, 3)


@dataclasses.dataclass(frozen=True)
class _Section:
    """A 2D file's triangles and grouped lines on the triangles' nodes, checked.

    ``points`` are (n, 3) in metres, in file order; ``edges`` are the
    triangles' edges, each sorted and once.
    """

    points: np.ndarray
    tris: np.ndarray
    area_tags: np.ndarray
    area_entities: np.ndarray
    lines: np.ndarray
    line_tags: np.ndarray
    line_entities: np.ndarray
    edges: np.ndarray


def _read_section(
    raw: meshio.Mesh,
    path: Path,
    length_unit: str,
    line_names: dict[int, str],
    *,
    kind: str = "cross-section",
) -> _Section:
    """The checked triangles and lines of a 2D file.

    The checks a cross-section and a 2D mesh share: the cell types, the
    overlaps and duplicates, one plane z = constant, and every grouped line
    on an edge of the triangles (``line_names`` names the line groups).
    Messages name the file a ``kind``: ``"cross-section"`` or ``"2D mesh"``.
    """
    cells = _cells(raw, path, section=True, file=kind)
    tris, area_tags, area_entities = cells["triangle"]
    lines, line_tags, line_entities = cells["line"]
    if len(tris) == 0:
        raise ValueError(
            f"{path.name} has no triangle in a 2D physical group: a {kind} "
            "needs its areas meshed and named as gmsh physical groups."
        )
    # keep the section's nodes, in file order
    used = np.unique(tris)
    index = np.full(len(raw.points), -1, dtype=np.int64)
    index[used] = np.arange(len(used))
    points = np.asarray(raw.points, dtype=float)[used] * LENGTH_UNITS[length_unit]
    tris, lines = index[tris], index[lines]
    edges = np.unique(np.sort(tris[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2)), axis=0)
    _check_section_plane(points, path, kind)
    _check_lines_on_edges(lines, line_tags, edges, line_names, path)
    return _Section(
        points, tris, area_tags, area_entities, lines, line_tags, line_entities, edges
    )


def _read_cross_section(
    path: Path, length_unit: str, thickness: float | None
) -> _Topology:
    """Parse a triangle section and extrude it (`Mesh.from_gmsh_cross_section`)."""
    _check_unit(length_unit)
    if thickness is not None:
        thickness = _check_thickness(thickness)
    raw = meshio.read(path, file_format="gmsh")
    regions = _regions(raw, section=True)
    taken = [r.name for r in regions if r.name in END_FACES]
    if taken:
        raise ValueError(
            f"{path.name} names a physical group {taken[0]!r}, but a cross-section's "
            f"end faces are named {END_FACES[0]!r} and {END_FACES[1]!r}. Rename the "
            "group in gmsh and export the mesh again."
        )
    section = _read_section(
        raw, path, length_unit, {r.tag: r.name for r in regions if r.dim == 2}
    )
    points, tris, lines = section.points, section.tris, section.lines
    line_tags, line_entities = section.line_tags, section.line_entities
    if thickness is None:
        thickness = _default_thickness(points, section.edges)
    n = len(points)
    nodes = np.concatenate([points, points + np.array([0.0, 0.0, thickness])])
    rank = _cut_rank(points, section.edges)
    tets = _split_prisms(tris, n, nodes, rank)
    sides = _split_quads(lines, n, rank)
    tag = max((r.tag for r in regions if r.dim == 2), default=0)
    front, back = Region(END_FACES[0], tag + 1, 2), Region(END_FACES[1], tag + 2, 2)
    ends = np.full(len(tris), 1, dtype=np.int64)
    triangles = np.concatenate([sides, tris, tris + n])
    tri_tags = np.concatenate(
        [np.repeat(line_tags, 2), ends * front.tag, ends * back.tag]
    )
    tri_entities = np.concatenate(
        [np.repeat(line_entities, 2), ends * front.tag, ends * back.tag]
    )
    neighbours = _triangle_neighbours(nodes, tets, triangles, path)
    topology = _Topology(
        length_unit=length_unit,
        imported_nodes=readonly(nodes),
        regions=(*regions, front, back),
        volume_elements=readonly(tets),
        volume_tags=readonly(np.repeat(section.area_tags, 3)),
        volume_entities=readonly(np.repeat(section.area_entities, 3)),
        surface_elements=readonly(triangles),
        surface_tags=readonly(tri_tags),
        surface_entities=readonly(tri_entities),
        surface_neighbours=readonly(neighbours),
        source=path.resolve(),
        section_thickness=float(thickness),
    )
    _check_tags(topology, path, section=True)
    return topology


def _read_on_rank_0(
    path: Path, length_unit: str, comm, *, section: bool = False, thickness=None
) -> _Topology:
    """Parse on rank 0 and broadcast; a failed read raises on every rank.

    With ``section``, read a cross-section (`Mesh.from_gmsh_cross_section`,
    whose ``thickness`` argument this is).

    Raises:
        _mpi.AgreedError: Ranks asked for different paths or units.
    """
    name = "Mesh.from_gmsh_cross_section" if section else "Mesh.from_gmsh"
    _mpi.meet(comm, name)
    asked = comm.allgather((str(path.resolve()), length_unit, section, thickness))
    if len(set(asked)) > 1:
        raise _mpi.AgreedError(
            f"{name} is collective, but ranks asked for {sorted(set(asked), key=str)}. "
            "Ranks reading files of their own pass comm=MPI.COMM_SELF."
        )
    topology, error = None, None
    # a read error is sent to every rank; anything else on rank 0 ends the job
    with _mpi.section(comm, agreed=()):
        if comm.rank == 0:
            try:
                topology = (
                    _read_cross_section(path, length_unit, thickness)
                    if section
                    else _read_gmsh(path, length_unit)
                )
            except _READ_ERRORS as err:
                error = err
        error = comm.bcast(error, root=0)
        if error is None:
            statics, arrays = None, None
            if topology is not None:
                statics = {f: getattr(topology, f) for f in _TOPOLOGY_STATICS}
                arrays = _topology_arrays(topology)
            statics = comm.bcast(statics, root=0)
            arrays = _mpi.bcast_arrays(comm, arrays)
    if error is not None:
        raise error
    if topology is not None:
        return topology
    for array in arrays.values():
        array.flags.writeable = False
    return _topology_from_arrays(arrays, **statics)


@jax.tree_util.register_pytree_with_keys_class
class Mesh:
    """A tetrahedral or triangle mesh with named regions, straight or curved.

    Build one with `Mesh.from_gmsh` from a gmsh ``.msh`` file (format 2.2
    ASCII or 4.1), or with `Mesh.from_gmsh_cross_section` from a 2D triangle
    section, read as a one-element-thick slab; ``length_unit`` names the unit
    of the file's coordinates, and `nodes` are in metres. Regions are the
    gmsh physical groups, by name and number.

    A tetrahedral mesh has `dim` 3. A triangle-only file read with
    `Mesh.from_gmsh` gives a 2D mesh (`dim` 2) in the (x, y) plane: its
    `nodes` are (n, 2), its areas are its volumes and its lines its surfaces.
    `volume_elements` and `surface_elements` hold the elements of both kinds
    of mesh.

    A second-order gmsh file (``tetra10`` and ``triangle6``; gmsh
    ``Mesh.ElementOrder 2``) is read as a curved mesh (`curved`): `nodes` hold
    every file node, corners and mid-edge, and a solve integrates on the
    quadratic element map. Orders above 2, a mix of orders and a curved 2D
    file are refused.

    A `Mesh` is a JAX
    `pytree <https://docs.jax.dev/en/latest/pytrees.html>`__:
    `jax.grad` reaches `nodes`; the elements, regions and imported
    coordinates are fixed. Element arrays are 0-based NumPy integer arrays,
    read-only, in file order.

    Example:
        A mesh written in micrometres::

            mesh = gfem.Mesh.from_gmsh("filter.msh", length_unit="um")
    """

    def __init__(self, nodes, topology: _Topology) -> None:
        """Internal: use `Mesh.from_gmsh` or `Mesh.from_gmsh_cross_section`."""
        self.nodes = nodes
        self._topology = topology

    def tree_flatten_with_keys(self):
        """JAX pytree protocol: the one leaf is ``nodes``."""
        return ((jax.tree_util.GetAttrKey("nodes"), self.nodes),), self._topology

    def tree_flatten(self):
        """JAX pytree protocol."""
        return (self.nodes,), self._topology

    @classmethod
    def tree_unflatten(cls, topology: _Topology, children) -> Mesh:
        """JAX pytree protocol."""
        return cls(children[0], topology)

    @classmethod
    def from_gmsh(cls, path: str | Path, length_unit: str = "um", *, comm=None) -> Mesh:
        """Read a gmsh ``.msh`` file (format 2.2 ASCII or 4.1).

        Every physical group becomes a region with its gmsh name and number:
        3D groups are volumes, 2D groups are surfaces. Elements outside any
        physical group, points and lines are ignored.

        A file of triangles only, in one plane z = constant, is a 2D mesh
        (`dim` 2): z is dropped, so the nodes are (n, 2); 2D groups are its
        volumes (areas) and 1D groups its surfaces (lines), and every grouped
        line must be an edge of the triangles. To read such a file as a
        one-element-thick slab instead, use `from_gmsh_cross_section`.

        Above one rank this is collective: rank 0 reads the file once and every
        rank ends up with the same whole mesh, so a call on one rank alone hangs.

        Args:
            path: The ``.msh`` file.
            length_unit: The length unit of the file's coordinates, one of
                `LENGTH_UNITS`. gsim writes micrometres.
            comm: The mpi4py communicator, resolved as `gfem.solve` resolves its
                own. Ranks reading files of their own pass ``MPI.COMM_SELF``.

        Returns:
            The mesh, with node coordinates in metres.

        Raises:
            ValueError: The unit is unknown, the mesh has elements of order 3
                or more or mixes first- and second-order ones, has elements
                other than tetrahedra and triangles (or, in 2D, triangles and
                lines), a 2D mesh is not in one plane z = constant or has a
                line off its triangles' edges, or its physical groups cannot be
                addressed by name.
            CurvedMeshUnsupportedError: The file is a second-order 2D mesh
                (``feature="2d_mesh"``).
            MeshQualityError: A curved tetrahedron is folded; regenerate the
                mesh with gmsh ``Mesh.HighOrderOptimize 1``.
            RuntimeError: Ranks asked for different paths or units.
        """
        comm = _mpi.resolve_comm(comm)
        path = Path(path)
        if comm is None:
            topology = _read_gmsh(path, length_unit)
        else:
            topology = _read_on_rank_0(path, length_unit, comm)
        return cls(jnp.asarray(topology.imported_nodes), topology)

    @classmethod
    def from_gmsh_cross_section(
        cls,
        path: str | Path,
        length_unit: str = "um",
        *,
        thickness: float | None = None,
        comm=None,
    ) -> Mesh:
        """Read a 2D gmsh cross-section as a one-element-thick slab.

        The file holds triangles only (format 2.2 ASCII or 4.1), in the xy
        plane. gfem extrudes the section one element thick along +z and
        splits each prism into three tetrahedra, so neighbouring prisms share
        faces. The split depends on the node coordinates, not their
        numbering, so a section meshed periodically (gmsh ``setPeriodic``)
        pairs under `gfem.Periodic` for any lattice. Every 2D physical group
        becomes a volume and every 1D physical group a surface, with its gmsh
        name and number; the end faces are the surfaces ``front`` (the
        section's plane) and ``back`` (one thickness along +z). Elements
        outside any physical group, points and other lines are ignored.

        The thickness is a numerical choice: a z-invariant problem's solution
        does not depend on it, and quantities per unit length divide the slab's
        totals by `section_thickness`.

        Above one rank this is collective, as `from_gmsh` is.

        Args:
            path: The ``.msh`` file of the section.
            length_unit: The length unit of the file's coordinates, one of
                `LENGTH_UNITS`.
            thickness: The slab thickness in metres. ``None`` picks the mean
                in-plane edge length, which keeps the tetrahedra well shaped.
            comm: The mpi4py communicator, as for `from_gmsh`.

        Returns:
            The slab mesh, with node coordinates in metres.

        Raises:
            ValueError: The unit is unknown, the thickness is not a positive
                length, the file holds 3D elements (read it with
                `from_gmsh`), the section is not in one plane z =
                constant, a 1D group is not on the triangles' edges, a group is
                named ``front`` or ``back``, or the groups cannot be addressed
                by name.
            CurvedMeshUnsupportedError: The section is second order
                (``feature="cross_section"``).
            RuntimeError: Ranks asked for different paths, units or thicknesses.

        Example:
            A heater cross-section written in micrometres::

                mesh = gfem.Mesh.from_gmsh_cross_section("heater.msh")
                mesh.section_thickness  # m
        """
        comm = _mpi.resolve_comm(comm)
        path = Path(path)
        if comm is None:
            topology = _read_cross_section(path, length_unit, thickness)
        else:
            topology = _read_on_rank_0(
                path, length_unit, comm, section=True, thickness=thickness
            )
        return cls(jnp.asarray(topology.imported_nodes), topology)

    def with_nodes(self, nodes) -> Mesh:
        """This mesh with its nodes at new coordinates.

        The moved mesh has the same elements, regions, length unit and
        `imported_nodes`; only `nodes` change. It applies a node map, a
        JAX function of your design parameters that returns node
        coordinates: `jax.grad` of an objective of a solve on the moved mesh
        reaches ``nodes``, and from there the map's parameters, as for a
        `gfem.shape.BoundaryMorph`.

        Concrete coordinates are checked over the whole mesh: an inverted
        element raises, and one below a quarter of its imported quality
        warns (`gfem.shape.quality`). Traced coordinates (under `jax.jit` or
        `jax.grad`) stay traced, and the solve checks the elements it solves
        on.

        Args:
            nodes: The node coordinates in metres, shape ``(n_nodes, dim)``
                as `nodes`, NumPy or JAX.

        Returns:
            The moved mesh.

        Raises:
            ValueError: ``nodes`` has another shape than `nodes` (also under
                tracing), or a coordinate is not finite.
            TypeError: ``nodes`` is complex.
            MeshQualityError: An element is inverted or flat.

        Example:
            A mesh stretched by 1 % along x::

                moved = mesh.with_nodes(mesh.nodes * jnp.array([1.01, 1.0, 1.0]))
                model = model.replace(mesh=moved)
        """
        from gfem.shape import _check_quality  # gfem.shape imports this module

        if jnp.iscomplexobj(nodes):
            raise TypeError(
                "Mesh.with_nodes takes real node coordinates, and these are "
                f"complex ({jnp.result_type(nodes)})."
            )
        nodes = jnp.asarray(nodes, dtype=jnp.float64)
        if nodes.shape != self.imported_nodes.shape:
            raise ValueError(
                f"Mesh.with_nodes got nodes of shape {nodes.shape}, and this "
                f"mesh has {self.imported_nodes.shape}: one row of {self.dim} "
                "coordinates per node."
            )
        moved = Mesh(nodes, self._topology)
        if not is_traced(nodes):
            if not np.isfinite(np.asarray(nodes)).all():
                raise ValueError(
                    "Mesh.with_nodes got node coordinates that are not finite "
                    "(NaN or infinite)."
                )
            _check_quality(moved, stacklevel=2)
        return moved

    def __repr__(self) -> str:
        section = self.section_thickness
        kind = "tetrahedra" if self.dim == 3 else "triangles"
        return (
            f"Mesh({len(self.imported_nodes)} nodes, {len(self.volume_elements)} "
            f"{kind}, volumes={list(self.volumes)}, "
            f"surfaces={list(self.surfaces)}, length_unit={self.length_unit!r}"
            + ("" if section is None else f", section_thickness={section:.4g} m")
            + ")"
        )

    # --- regions -------------------------------------------------------------

    @property
    def curved(self) -> bool:
        """Whether the mesh is curved: read from a second-order gmsh file.

        A curved mesh keeps the mid-edge node of every tetrahedron edge and
        triangle edge, and solves on the quadratic element map.
        """
        return self._topology.volume_midnodes is not None

    @property
    def dim(self) -> int:
        """The dimension of the mesh, 3 for a tetrahedral mesh and 2 for a 2D mesh."""
        return self._topology.dim

    @property
    def regions(self) -> tuple[Region, ...]:
        """All regions, volumes then surfaces, each by tag."""
        return self._topology.regions

    @property
    def volumes(self) -> tuple[str, ...]:
        """Names of the volume regions (the areas of a 2D mesh), by tag."""
        return tuple(r.name for r in self.regions if r.dim == self.dim)

    @property
    def surfaces(self) -> tuple[str, ...]:
        """Names of the surface regions (the lines of a 2D mesh), by tag."""
        return tuple(r.name for r in self.regions if r.dim == self.dim - 1)

    def _region_names(self, *, surfaces: bool = False) -> dict[int, str]:
        """Tag to name of the volume regions, or of the surface regions."""
        dim = self.dim - 1 if surfaces else self.dim
        return {r.tag: r.name for r in self.regions if r.dim == dim}

    def region(self, name: str) -> Region:
        """Look up a region by its gmsh physical-group name.

        Raises:
            RegionNotFoundError: No region has this name.
        """
        for r in self.regions:
            if r.name == name:
                return r
        raise RegionNotFoundError(name, (r.name for r in self.regions))

    def elements(self, region: str) -> np.ndarray:
        """Indices of a region's elements, in mesh order.

        For a volume, rows of `volume_elements` (the order of a
        `MaterialField`'s per-element values); for a surface, rows of
        `surface_elements`.

        Raises:
            RegionNotFoundError: No region has this name.
        """
        r = self.region(region)
        t = self._topology
        tags = t.volume_tags if r.dim == t.dim else t.surface_tags
        return readonly(np.flatnonzero(tags == r.tag))

    def adjacency(self, surface: str) -> tuple[tuple[str | None, str | None], ...]:
        """The ``(domin, domout)`` volume pairs on the two sides of a surface.

        For each triangle of the surface, ``domin`` is the volume behind it
        and ``domout`` the volume in front of it, along the triangle's normal
        (right-hand rule on its node order). On a 2D mesh a line segment's
        normal is its direction crossed with +z, so ``domin`` is the area on
        its left. ``None`` stands for outside the mesh. Each distinct pair
        appears once, sorted by tag.

        Raises:
            RegionNotFoundError: The mesh has no surface of this name.
        """
        tag = self._surface(surface).tag
        t = self._topology
        nb = t.surface_neighbours[t.surface_tags == tag]
        sides = np.where(nb >= 0, t.volume_tags[np.maximum(nb, 0)], -1)
        name = self._region_names()
        pairs = sorted({(int(a), int(b)) for a, b in sides})
        return tuple((name.get(a), name.get(b)) for a, b in pairs)

    def adjacent_volumes(self, surface: str) -> tuple[str, ...]:
        """The volume regions that border a surface, by tag.

        Raises:
            RegionNotFoundError: The mesh has no surface of this name.
        """
        touching = {v for pair in self.adjacency(surface) for v in pair}
        return tuple(v for v in self.volumes if v in touching)

    def _surface(self, name: str) -> Region:
        for r in self.regions:
            if r.name == name and r.dim == self.dim - 1:
                return r
        raise RegionNotFoundError(name, self.surfaces, kind="surface")

    # --- export --------------------------------------------------------------

    def to_meshio(self) -> meshio.Mesh:
        """The mesh as a `meshio.Mesh`, coordinates in metres.

        Tetrahedra and triangles (on a 2D mesh, triangles and lines) carry two
        cell-data arrays: ``region`` (the gmsh physical-group number) and
        ``entity`` (the gmsh entity number). A curved mesh writes second-order
        cells, ``tetra10`` and ``triangle6`` in meshio's (VTK) node order.
        """
        t = self._topology
        kinds = ("tetra", "triangle") if t.dim == 3 else ("triangle", "line")
        volumes, surfaces = t.volume_elements, t.surface_elements
        volume_mid, surface_mid = t.volume_midnodes, t.surface_midnodes
        if volume_mid is not None and surface_mid is not None:
            kinds = ("tetra10", "triangle6")
            volumes = np.hstack([volumes, volume_mid[:, _TET_MID_TO_MESHIO]])
            surfaces = np.hstack([surfaces, surface_mid[:, _TRI_MID_TO_MESHIO]])
        return meshio.Mesh(
            np.asarray(self.nodes, dtype=float),
            [(kinds[0], volumes), (kinds[1], surfaces)],
            cell_data={
                "region": [t.volume_tags, t.surface_tags],
                "entity": [t.volume_entities, t.surface_entities],
            },
        )

    def to_vtu(self, path: str | Path) -> None:
        """Write the mesh and its region ids to a ``.vtu`` file (see `to_meshio`).

        Args:
            path: The output file.
        """
        meshio.write(path, self.to_meshio(), file_format="vtu")

    # --- elements ------------------------------------------------------------

    @property
    def length_unit(self) -> str:
        """The length unit the mesh file was written in."""
        return self._topology.length_unit

    @property
    def section_thickness(self) -> float | None:
        """The slab thickness (m) of a cross-section, ``None`` for any other mesh.

        Set by `from_gmsh_cross_section`; a quantity per unit length is the
        slab's total divided by it.
        """
        return self._topology.section_thickness

    @property
    def source(self) -> Path | None:
        """The absolute path of the imported file."""
        return self._topology.source

    @property
    def imported_nodes(self) -> np.ndarray:
        """Node coordinates (m) as imported or refined, kept when the nodes move."""
        return self._topology.imported_nodes

    @property
    def volume_elements(self) -> np.ndarray:
        """Node indices of each volume element, shape ``(n, dim + 1)``.

        The tetrahedra of a tetrahedral mesh (`tetrahedra`), the triangles
        of a 2D mesh. `elements` gives a volume's rows.
        """
        return self._topology.volume_elements

    @property
    def surface_elements(self) -> np.ndarray:
        """Node indices of each surface-region element, shape ``(n, dim)``.

        The triangles of a tetrahedral mesh (`triangles`), the line segments
        of a 2D mesh. `elements` gives a surface's rows.
        """
        return self._topology.surface_elements

    def _tetrahedral(self, name: str, role: str) -> _Topology:
        """The topology, after checking that ``name`` exists on this mesh."""
        if self._topology.dim != 3:
            raise ValueError(
                f"Mesh.{name} is an array of a tetrahedral mesh, and this mesh "
                f"has dimension {self._topology.dim}: its triangles are its "
                f"volume elements. Read Mesh.{role} (and Mesh.elements(region) "
                "for a region's rows) instead."
            )
        return self._topology

    @property
    def tetrahedra(self) -> np.ndarray:
        """Node indices of each tetrahedron, shape ``(n_tet, 4)``.

        Raises:
            ValueError: The mesh is a 2D mesh; read `volume_elements`.
        """
        return self._tetrahedral("tetrahedra", "volume_elements").volume_elements

    @property
    def tetrahedron_tags(self) -> np.ndarray:
        """Physical-group number of each tetrahedron's volume region.

        Raises:
            ValueError: The mesh is a 2D mesh; read `volume_elements`.
        """
        return self._tetrahedral("tetrahedron_tags", "volume_elements").volume_tags

    @property
    def tetrahedron_entities(self) -> np.ndarray:
        """Gmsh entity (elementary volume) number of each tetrahedron.

        One region may hold several entities: gsim, for example, tags meshed
        conductor volumes with the name of the surrounding dielectric.

        Raises:
            ValueError: The mesh is a 2D mesh; read `volume_elements`.
        """
        topology = self._tetrahedral("tetrahedron_entities", "volume_elements")
        return topology.volume_entities

    @property
    def triangles(self) -> np.ndarray:
        """Node indices of each surface-region triangle, shape ``(n_tri, 3)``.

        Raises:
            ValueError: The mesh is a 2D mesh, whose triangles are its volume
                elements; read `surface_elements` for its lines.
        """
        return self._tetrahedral("triangles", "surface_elements").surface_elements

    @property
    def triangle_tags(self) -> np.ndarray:
        """Physical-group number of each triangle's surface region.

        Raises:
            ValueError: The mesh is a 2D mesh; read `surface_elements`.
        """
        return self._tetrahedral("triangle_tags", "surface_elements").surface_tags

    @property
    def triangle_entities(self) -> np.ndarray:
        """Gmsh entity (elementary surface) number of each triangle.

        Raises:
            ValueError: The mesh is a 2D mesh; read `surface_elements`.
        """
        topology = self._tetrahedral("triangle_entities", "surface_elements")
        return topology.surface_entities

    @property
    def triangle_neighbours(self) -> np.ndarray:
        """The tetrahedra on each side of each triangle, shape ``(n_tri, 2)``.

        Column 0 is the tetrahedron behind the triangle (``domin``), column 1
        the one in front of it (``domout``), along the triangle's normal;
        -1 means outside the mesh.

        Raises:
            ValueError: The mesh is a 2D mesh; read `surface_elements` and
                `adjacency`.
        """
        topology = self._tetrahedral("triangle_neighbours", "surface_elements")
        return topology.surface_neighbours


#: The `_Topology` arrays that, with its regions and dimension, define its
#: elements: the element part of `_ROLE_ARRAYS`, which `_same_topology`
#: compares.
_ELEMENT_ARRAYS = (
    "volume_elements",
    "volume_tags",
    "volume_entities",
    "surface_elements",
    "surface_tags",
    "surface_entities",
    "volume_midnodes",
    "surface_midnodes",
)


def _same_array(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    """Equal arrays, or both absent (the mid-edge nodes of a straight mesh)."""
    if a is None or b is None:
        return a is b
    return np.array_equal(a, b)


def _same_topology(a: Mesh, b: Mesh) -> bool:
    """Whether two meshes have the same elements, tags and regions.

    The same object, or equal arrays: the same file read twice (a loaded
    result's mesh and the model's) gives two objects.
    """
    ta, tb = a._topology, b._topology  # noqa: SLF001
    if ta is tb:
        return True
    return (
        ta.dim == tb.dim
        and a.regions == b.regions
        and all(
            _same_array(getattr(ta, name), getattr(tb, name))
            for name in _ELEMENT_ARRAYS
        )
    )


def _is_model_mesh(mesh: Mesh, model_mesh: Mesh) -> bool:
    """Whether ``mesh`` is the model's mesh: one a result can refer to by digest.

    The same object, or the same elements and regions (`_same_topology`),
    length unit, cross-section thickness, topology digest
    (`_mesh_sha256`) and nodes. A solve cache entry stores such a mesh
    as its digest, and `gfem.error_estimate` needs it.
    """
    if mesh is model_mesh:
        return True
    return (
        _same_topology(mesh, model_mesh)
        and mesh.length_unit == model_mesh.length_unit
        and mesh.section_thickness == model_mesh.section_thickness
        and _mesh_sha256(mesh) == _mesh_sha256(model_mesh)
        and np.array_equal(np.asarray(mesh.nodes), np.asarray(model_mesh.nodes))
    )


# Per volume element, its shortest edge, by mesh topology (`_shortest_edges`).
_SHORTEST_EDGES: weakref.WeakKeyDictionary[_Topology, np.ndarray] = (
    weakref.WeakKeyDictionary()
)


def _shortest_edges(mesh: Mesh) -> np.ndarray:
    """The shortest edge (m) of each volume element on the imported nodes.

    Cached per mesh topology: moved nodes do not change it.
    """
    topology = mesh._topology  # noqa: SLF001
    h = _SHORTEST_EDGES.get(topology)
    if h is None:
        x = mesh.imported_nodes[topology.volume_elements]
        sq = np.full(len(x), np.inf)
        pairs = _TET_EDGES if topology.dim == 3 else ((0, 1), (0, 2), (1, 2))
        for i, j in pairs:
            sq = np.minimum(sq, np.sum((x[:, i] - x[:, j]) ** 2, axis=1))
        h = _SHORTEST_EDGES[topology] = readonly(np.sqrt(sq))
    return h


def _regions(
    raw: meshio.Mesh, *, section: bool = False, top: int = 3
) -> tuple[Region, ...]:
    """The physical groups of dimension ``top`` and one below.

    On a cross-section (``section``) 2D groups are lifted to volumes and 1D
    groups to surfaces.
    """
    lift = 1 if section else 0
    regions = (
        Region(name=name, tag=int(tag), dim=int(dim) + lift)
        for name, (tag, dim) in raw.field_data.items()
        if int(dim) + lift in (top - 1, top)
    )
    return tuple(sorted(regions, key=lambda r: (-r.dim, r.tag)))


# meshio cell types with a 3D (volume) element
_VOLUME_TYPES = (
    "tetra",
    "tetra10",
    "tetra20",
    "tetra35",
    "tetra56",
    "hexahedron",
    "wedge",
    "pyramid",
)
# meshio's second-order simplices, by the first-order kind they curve
_SECOND_ORDER = {"tetra10": "tetra", "triangle6": "triangle", "line3": "line"}
# meshio's `tetra10` is VTK's order: after the corners its mid nodes are on edges
# (0,1) (1,2) (0,2) (0,3) (1,3) (2,3), where gmsh puts (2,3) before (1,3);
# `_TET_EDGES` lists (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
_TET_MID_FROM_MESHIO = [0, 2, 3, 1, 4, 5]
# meshio's `triangle6` mid nodes are on edges (0,1) (1,2) (2,0); `_TRI_EDGES` lists
# the edges opposite vertices 0, 1, 2: (1,2) (0,2) (0,1).
_TRI_MID_FROM_MESHIO = [1, 2, 0]
# the converse, from storage order back to meshio's
_TET_MID_TO_MESHIO = [0, 3, 1, 2, 4, 5]
_TRI_MID_TO_MESHIO = [2, 0, 1]
# the plural of each kind of triangle file, in messages
_KINDS = {"cross-section": "cross-sections", "2D mesh": "2D meshes"}
# the dimension of each cell type a reader keeps
_CELL_DIM = {"tetra": 3, "triangle": 2, "line": 1}


def _refuse_orders(types: list[str], path: Path, *, section: bool, file: str) -> None:
    """Refuse the files gfem cannot read as a curved or a straight mesh.

    Raises:
        ValueError: A cell of order 3 or more, or first- and second-order
            cells together (the message names the gmsh option).
        gfem.CurvedMeshUnsupportedError: A second-order 2D file or
            cross-section (``section``).
    """
    # meshio names higher-order cells by their node count (tetra20, line4, ...)
    above = [
        t
        for t in types
        if t.startswith(("tetra", "triangle", "line"))
        and t not in ("tetra", "triangle", "line", *_SECOND_ORDER)
    ]
    if above:
        raise ValueError(
            f"{path.name} has {' / '.join(above)} elements: order 3 or more. gfem "
            "reads second-order (curved) meshes at most. Regenerate it with gmsh "
            '`Mesh.ElementOrder 2` (Python: gmsh.option.setNumber("Mesh.ElementOrder", '
            "2); command line: gmsh -order 2), or `ElementOrder 1` for a "
            "straight-sided mesh."
        )
    second = [t for t in types if t in _SECOND_ORDER]
    if section and second:
        flat = "2d_mesh" if file == "2D mesh" else "cross_section"
        raise CurvedMeshUnsupportedError(
            flat,
            f"{path.name} is a second-order (curved) {file}: it has "
            f"{' / '.join(second)} elements. gfem solves a curved mesh in 3D only: "
            "NGSolve curves the areas but keeps the boundary segments straight, "
            "so boundary terms would see chords. Regenerate it with gmsh "
            '`Mesh.ElementOrder 1` (Python: gmsh.option.setNumber("Mesh.ElementOrder", '
            "1); command line: gmsh -order 1).",
        )
    first = [t for t in types if t in ("tetra", "triangle")]
    curved = [t for t in second if t != "line3"]
    if curved and first and not section:
        raise ValueError(
            f"{path.name} mixes first- and second-order elements "
            f"({' / '.join([*first, *curved])}). Regenerate it with gmsh "
            "`Mesh.ElementOrder 2` for a curved mesh, or `ElementOrder 1` for a "
            "straight-sided one, on the whole mesh."
        )


def _cells(
    raw: meshio.Mesh, path: Path, *, section: bool = False, file: str = "cross-section"
) -> dict[str, tuple[np.ndarray, ...]]:
    """Volume and surface cells with their physical and entity tags.

    For `Mesh.from_gmsh` tetrahedra and triangles; for a cross-section or a
    2D mesh (``section``, the ``file`` kind named in messages) triangles and
    lines. A curved file (`tetra10`, `triangle6`) gives the corner nodes of its
    cells and, under ``"midnodes"``, the tetrahedra's and triangles' mid-edge
    nodes in the storage order of `_Topology`.
    """
    if "gmsh:physical" not in raw.cell_data:
        dims = "areas and lines" if section else "volumes and surfaces"
        raise ValueError(
            f"{path.name} defines no physical groups. Name the {dims} as gmsh "
            "physical groups and export the mesh again."
        )
    width = {"triangle": 3, "line": 2} if section else {"tetra": 4, "triangle": 3}
    volume = "triangle" if section else "tetra"
    types = sorted({block.type for block in raw.cells})
    _refuse_orders(types, path, section=section, file=file)
    # a 3D file's second-order cells; its line3 are ignored like its lines
    curved = any(t in ("tetra10", "triangle6") for t in types)
    full = {"tetra": 10, "triangle": 6} if curved else width
    solid = [t for t in types if t in _VOLUME_TYPES]
    if section and solid:
        raise ValueError(
            f"{path.name} has {' / '.join(solid)} elements: it is a 3D mesh. Read "
            "it with gfem.Mesh.from_gmsh; Mesh.from_gmsh_cross_section reads a 2D "
            "section of triangles only."
        )
    known = (*width, *(t for t, k in _SECOND_ORDER.items() if k in width))
    other = [t for t in types if t not in (*known, "line", "line3", "vertex")]
    if other:
        supported = (
            f"{_KINDS[file]} of triangles only (lines on boundaries)"
            if section
            else "tetrahedral meshes only (triangles on surfaces)"
        )
        raise ValueError(
            f"{path.name} has {' / '.join(other)} elements. gfem supports {supported}."
        )
    parts: dict[str, list[tuple]] = {k: [] for k in width}
    for block, phys, geom in zip(
        raw.cells,
        raw.cell_data["gmsh:physical"],
        raw.cell_data["gmsh:geometrical"],
        strict=True,
    ):
        kind = _SECOND_ORDER.get(block.type, block.type)
        if kind in width:
            parts[kind].append((block.data, phys, geom))
    _check_overlaps_in_cell_sets(raw, path, section=section)
    out: dict[str, tuple[np.ndarray, ...]] = {}
    mids: dict[str, np.ndarray] = {}
    for kind, items in parts.items():
        if items:
            data, phys, geom = (np.concatenate(c) for c in zip(*items, strict=True))
        else:
            data, phys, geom = np.empty((0, full[kind])), np.empty(0), np.empty(0)
        grouped = phys != 0  # gmsh writes group 0 for elements outside any group
        if kind == volume and not grouped.all():
            what = (
                ("triangles", "2D", "area", "surfaces")
                if section
                else ("tetrahedra", "volume", "volume", "volumes")
            )
            raise ValueError(
                f"{path.name}: {int((~grouped).sum())} {what[0]} lie outside any "
                f"{what[1]} physical group. Every {what[2]} needs a named region: "
                f"add the missing {what[3]} to a physical group in gmsh."
            )
        data = data[grouped].astype(np.int64)
        out[kind] = (
            data[:, : width[kind]],
            phys[grouped].astype(np.int64),
            geom[grouped].astype(np.int64),
        )
        _check_duplicates(out[kind][0], out[kind][1], raw, kind, path)
        if curved:
            mids[kind] = data[:, width[kind] :]
    if curved:
        out["midnodes"] = (
            mids["tetra"][:, _TET_MID_FROM_MESHIO],
            mids["triangle"][:, _TRI_MID_FROM_MESHIO],
        )
    return out


def _overlap_error(path: Path, names: set[str]) -> ValueError:
    return ValueError(
        f"{path.name}: elements belong to more than one physical group "
        f"({', '.join(repr(n) for n in sorted(names))}). gfem needs each "
        "tetrahedron in exactly one volume region and each triangle in at most "
        "one surface region: split the overlapping groups in gmsh so that no "
        "entity is in two of them, and export the mesh again."
    )


def _check_overlaps_in_cell_sets(
    raw: meshio.Mesh, path: Path, *, section: bool = False
) -> None:
    """Msh 4.1: meshio keeps the first physical tag; its cell sets keep all."""
    dims = (1, 2) if section else (2, 3)
    names = [n for n, (_, dim) in raw.field_data.items() if int(dim) in dims]
    sets = {n: raw.cell_sets[n] for n in names if n in (raw.cell_sets or {})}
    for i, block in enumerate(raw.cells):
        members = np.zeros(len(block.data), dtype=np.int64)
        for cells in sets.values():
            members[np.asarray(cells[i], dtype=np.int64)] += 1
        if (members > 1).any():
            shared = np.flatnonzero(members > 1)
            raise _overlap_error(
                path, {n for n, c in sets.items() if np.isin(c[i], shared).any()}
            )


def _check_duplicates(
    data: np.ndarray, tags: np.ndarray, raw: meshio.Mesh, kind: str, path: Path
) -> None:
    """Msh 2.2: an element in two physical groups is written twice."""
    _, inverse, counts = np.unique(
        np.sort(data, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    hit = counts[inverse.ravel()] > 1
    if not hit.any():
        return
    dim = _CELL_DIM[kind]
    name = {int(t): n for n, (t, d) in raw.field_data.items() if int(d) == dim}
    raise _overlap_error(path, {name.get(int(t), str(t)) for t in tags[hit]})


def _triangle_neighbours(
    points: np.ndarray, tets: np.ndarray, tris: np.ndarray, path: Path
) -> np.ndarray:
    """For each triangle, the tetrahedron behind it and the one in front."""
    neighbours = np.full((len(tris), 2), -1, dtype=np.int64)
    if len(tris) == 0:
        return neighbours
    faces = np.sort(tets[:, _TET_FACES].reshape(-1, 3), axis=1)
    _, ids = np.unique(
        np.concatenate([faces, np.sort(tris, axis=1)]), axis=0, return_inverse=True
    )
    ids = ids.ravel()
    face_ids, tri_ids = ids[: len(faces)], ids[len(faces) :]
    counts = np.bincount(face_ids, minlength=ids.max() + 1)
    if (counts > 2).any():
        raise ValueError(f"{path.name}: a face is shared by more than two tetrahedra.")
    n_tets = counts[tri_ids]
    if (n_tets == 0).any():
        bad = int(np.flatnonzero(n_tets == 0)[0])
        raise ValueError(
            f"{path.name}: surface triangle {bad} is not a face of any "
            "tetrahedron, so the surface and volume meshes do not conform."
        )
    order = np.argsort(face_ids, kind="stable")
    first = np.searchsorted(face_ids[order], tri_ids)
    a, b, c = (points[tris[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    for slot in (0, 1):
        has = n_tets > slot
        face = order[first[has] + slot]
        tet, opposite = face // 4, tets.reshape(-1)[face]
        side = np.einsum("ij,ij->i", normal[has], points[opposite] - a[has])
        rows = np.flatnonzero(has)
        neighbours[rows[side < 0], 0] = tet[side < 0]
        neighbours[rows[side > 0], 1] = tet[side > 0]
    two = n_tets == 2
    if (neighbours[two] < 0).any():
        raise ValueError(
            f"{path.name}: two tetrahedra overlap on the same side of a face."
        )
    return neighbours


def _segment_neighbours(
    points: np.ndarray, tris: np.ndarray, lines: np.ndarray, path: Path
) -> np.ndarray:
    """For each line segment, the triangle behind it and the one in front.

    A segment's normal is its direction crossed with +z, as for the faces a
    cross-section's slab extrudes from its lines: column 0 (``domin``) is the
    triangle on the segment's left, column 1 (``domout``) the one on its
    right, -1 outside the mesh. Every segment is an edge of the triangles
    (`_check_lines_on_edges`).
    """
    neighbours = np.full((len(lines), 2), -1, dtype=np.int64)
    if len(lines) == 0:
        return neighbours
    edges = np.sort(tris[:, _TRI_EDGES].reshape(-1, 2), axis=1)
    _, ids = np.unique(
        np.concatenate([edges, np.sort(lines, axis=1)]), axis=0, return_inverse=True
    )
    ids = ids.ravel()
    edge_ids, line_ids = ids[: len(edges)], ids[len(edges) :]
    counts = np.bincount(edge_ids, minlength=ids.max() + 1)
    if (counts > 2).any():
        raise ValueError(f"{path.name}: an edge is shared by more than two triangles.")
    n_tris = counts[line_ids]
    order = np.argsort(edge_ids, kind="stable")
    first = np.searchsorted(edge_ids[order], line_ids)
    a, b = points[lines[:, 0]], points[lines[:, 1]]
    t = b - a
    normal = np.stack([t[:, 1], -t[:, 0]], axis=1)  # t x z
    for slot in (0, 1):
        has = n_tris > slot
        edge = order[first[has] + slot]
        tri, opposite = edge // 3, tris.reshape(-1)[edge]
        side = np.einsum("ij,ij->i", normal[has], points[opposite] - a[has])
        rows = np.flatnonzero(has)
        neighbours[rows[side < 0], 0] = tri[side < 0]
        neighbours[rows[side > 0], 1] = tri[side > 0]
    two = n_tris == 2
    if (neighbours[two] < 0).any():
        raise ValueError(
            f"{path.name}: two triangles overlap on the same side of an edge."
        )
    return neighbours


def _check_tags(topology: _Topology, path: Path, *, section: bool = False) -> None:
    """Every element's physical group must exist and be named."""
    top = topology.dim
    kinds = ("tetrahedra", "triangles") if top == 3 else ("triangles", "lines")
    for dim, tags, kind in (
        (top, topology.volume_tags, kinds[0]),
        (top - 1, topology.surface_tags, kinds[1]),
    ):
        named = {r.tag for r in topology.regions if r.dim == dim}
        missing = sorted(set(np.unique(tags).tolist()) - named)
        if missing:
            # a cross-section's message names the file's elements and groups
            in_file = (
                (dim - 1, "triangles" if dim == 3 else "lines")
                if section
                else (dim, kind)
            )
            raise ValueError(
                f"{path.name}: {in_file[1]} belong to {in_file[0]}D physical group(s) "
                f"{missing} that have no name of their own in $PhysicalNames. "
                "gfem addresses regions by name: give every physical group a "
                "unique name in gmsh and export the mesh again."
            )
