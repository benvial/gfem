"""Mesh import (gmsh ``.msh`` 2.2 ASCII and 4.1) and the `Mesh` pytree."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np

from gfem.errors import RegionNotFoundError

#: Supported ``length_unit`` values and their size in metres.
LENGTH_UNITS = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}

# Local face k of a tetrahedron is the one opposite its vertex k.
_TET_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


@dataclasses.dataclass(frozen=True)
class Region:
    """A named volume or surface of the mesh: one gmsh physical group.

    Attributes:
        name: The gmsh physical-group name, an opaque string.
        tag: The gmsh physical-group number.
        dim: 3 for a volume, 2 for a surface.
    """

    name: str
    tag: int
    dim: int


@dataclasses.dataclass(frozen=True, eq=False)
class _Topology:
    """Everything about a mesh except its node coordinates.

    This is the static part of the `Mesh` pytree. It compares by identity, so
    a jitted function retraces only when it sees a different imported mesh.
    """

    length_unit: str
    imported_nodes: np.ndarray
    regions: tuple[Region, ...]
    tetrahedra: np.ndarray
    tet_tags: np.ndarray
    tet_entities: np.ndarray
    triangles: np.ndarray
    tri_tags: np.ndarray
    tri_entities: np.ndarray
    tri_neighbours: np.ndarray
    source: Path | None = None


@jax.tree_util.register_pytree_with_keys_class
class Mesh:
    """A straight-sided tetrahedral mesh with named regions.

    Build one with `Mesh.from_gmsh`. A `Mesh` is a JAX pytree whose only leaf
    is `nodes`, the node coordinates in metres. Everything else is static:
    elements, regions (gmsh physical groups, by name and number), the gmsh
    entity of each element, which volumes border each surface, and the
    imported node coordinates.

    Element arrays are 0-based NumPy integer arrays, read-only, in file order.
    """

    def __init__(self, nodes, topology: _Topology) -> None:
        """Internal: use `Mesh.from_gmsh`."""
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
    def from_gmsh(cls, path: str | Path, length_unit: str = "um") -> Mesh:
        """Read a gmsh ``.msh`` file (format 2.2 ASCII or 4.1).

        Every physical group becomes a region with its gmsh name and number:
        3D groups are volumes, 2D groups are surfaces. Elements outside any
        physical group, points and lines are ignored.

        Args:
            path: The ``.msh`` file.
            length_unit: The length unit of the file's coordinates, one of
                `LENGTH_UNITS`. gsim writes micrometres.

        Returns:
            The mesh, with node coordinates in metres.

        Raises:
            ValueError: The unit is unknown, the mesh is second order (curved),
                has elements other than tetrahedra and triangles, or its
                physical groups cannot be addressed by name.
        """
        if length_unit not in LENGTH_UNITS:
            raise ValueError(
                f"Unknown length_unit {length_unit!r}; use one of "
                f"{', '.join(repr(u) for u in LENGTH_UNITS)}."
            )
        path = Path(path)
        raw = meshio.read(path, file_format="gmsh")
        regions = _regions(raw)
        cells = _cells(raw, path)
        tets, tet_tags, tet_entities = cells["tetra"]
        tris, tri_tags, tri_entities = cells["triangle"]
        points = np.asarray(raw.points, dtype=float)
        neighbours = _triangle_neighbours(points, tets, tris, path)
        nodes = points * LENGTH_UNITS[length_unit]
        topology = _Topology(
            length_unit=length_unit,
            imported_nodes=_readonly(nodes),
            regions=regions,
            tetrahedra=_readonly(tets),
            tet_tags=_readonly(tet_tags),
            tet_entities=_readonly(tet_entities),
            triangles=_readonly(tris),
            tri_tags=_readonly(tri_tags),
            tri_entities=_readonly(tri_entities),
            tri_neighbours=_readonly(neighbours),
            source=path.resolve(),
        )
        _check_tags(topology, path)
        return cls(jnp.asarray(nodes), topology)

    def __repr__(self) -> str:
        return (
            f"Mesh({len(self.imported_nodes)} nodes, {len(self.tetrahedra)} "
            f"tetrahedra, volumes={list(self.volumes)}, "
            f"surfaces={list(self.surfaces)}, length_unit={self.length_unit!r})"
        )

    # --- regions -------------------------------------------------------------

    @property
    def regions(self) -> tuple[Region, ...]:
        """Every region: volumes, then surfaces, each by tag."""
        return self._topology.regions

    @property
    def volumes(self) -> tuple[str, ...]:
        """Names of the volume regions, by tag."""
        return tuple(r.name for r in self.regions if r.dim == 3)

    @property
    def surfaces(self) -> tuple[str, ...]:
        """Names of the surface regions, by tag."""
        return tuple(r.name for r in self.regions if r.dim == 2)

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

        For a volume, rows of `tetrahedra` (the order of a `MaterialField`'s
        per-element values); for a surface, rows of `triangles`.

        Raises:
            RegionNotFoundError: No region has this name.
        """
        r = self.region(region)
        tags = self.tetrahedron_tags if r.dim == 3 else self.triangle_tags
        return _readonly(np.flatnonzero(tags == r.tag))

    def adjacency(self, surface: str) -> tuple[tuple[str | None, str | None], ...]:
        """The ``(domin, domout)`` volume pairs on the two sides of a surface.

        For each triangle of the surface, ``domin`` is the volume behind it
        and ``domout`` the volume in front of it, along the triangle's normal
        (right-hand rule on its node order). ``None`` stands for outside the
        mesh. Each distinct pair appears once, sorted by tag.

        Raises:
            RegionNotFoundError: The mesh has no surface of this name.
        """
        tag = self._surface(surface).tag
        tet_tags = self.tetrahedron_tags
        nb = self.triangle_neighbours[self.triangle_tags == tag]
        sides = np.where(nb >= 0, tet_tags[np.maximum(nb, 0)], -1)
        name = {r.tag: r.name for r in self.regions if r.dim == 3}
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
            if r.name == name and r.dim == 2:
                return r
        raise RegionNotFoundError(name, self.surfaces, kind="surface")

    # --- export --------------------------------------------------------------

    def to_meshio(self) -> meshio.Mesh:
        """The mesh as a `meshio.Mesh`, coordinates in metres.

        Tetrahedra and triangles carry two cell-data arrays: ``region`` (the
        gmsh physical-group number) and ``entity`` (the gmsh entity number).
        """
        return meshio.Mesh(
            np.asarray(self.nodes, dtype=float),
            [("tetra", self.tetrahedra), ("triangle", self.triangles)],
            cell_data={
                "region": [self.tetrahedron_tags, self.triangle_tags],
                "entity": [self.tetrahedron_entities, self.triangle_entities],
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
    def source(self) -> Path | None:
        """The absolute path of the imported file."""
        return self._topology.source

    @property
    def imported_nodes(self) -> np.ndarray:
        """Node coordinates (m) as imported, kept when the nodes move."""
        return self._topology.imported_nodes

    @property
    def tetrahedra(self) -> np.ndarray:
        """Node indices of each tetrahedron, shape ``(n_tet, 4)``."""
        return self._topology.tetrahedra

    @property
    def tetrahedron_tags(self) -> np.ndarray:
        """Physical-group number of each tetrahedron's volume region."""
        return self._topology.tet_tags

    @property
    def tetrahedron_entities(self) -> np.ndarray:
        """Gmsh entity (elementary volume) number of each tetrahedron.

        One region may hold several entities: gsim, for example, tags meshed
        conductor volumes with the name of the surrounding dielectric.
        """
        return self._topology.tet_entities

    @property
    def triangles(self) -> np.ndarray:
        """Node indices of each surface-region triangle, shape ``(n_tri, 3)``."""
        return self._topology.triangles

    @property
    def triangle_tags(self) -> np.ndarray:
        """Physical-group number of each triangle's surface region."""
        return self._topology.tri_tags

    @property
    def triangle_entities(self) -> np.ndarray:
        """Gmsh entity (elementary surface) number of each triangle."""
        return self._topology.tri_entities

    @property
    def triangle_neighbours(self) -> np.ndarray:
        """The tetrahedra on each side of each triangle, shape ``(n_tri, 2)``.

        Column 0 is the tetrahedron behind the triangle (``domin``), column 1
        the one in front of it (``domout``), along the triangle's normal;
        -1 means outside the mesh.
        """
        return self._topology.tri_neighbours


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def _regions(raw: meshio.Mesh) -> tuple[Region, ...]:
    regions = (
        Region(name=name, tag=int(tag), dim=int(dim))
        for name, (tag, dim) in raw.field_data.items()
        if int(dim) in (2, 3)
    )
    return tuple(sorted(regions, key=lambda r: (-r.dim, r.tag)))


def _cells(raw: meshio.Mesh, path: Path) -> dict[str, tuple[np.ndarray, ...]]:
    """Tetrahedra and triangles with their physical and entity tags."""
    if "gmsh:physical" not in raw.cell_data:
        raise ValueError(
            f"{path.name} defines no physical groups. Name the volumes and "
            "surfaces as gmsh physical groups and export the mesh again."
        )
    width = {"tetra": 4, "triangle": 3}
    types = sorted({block.type for block in raw.cells})
    # meshio names higher-order cells by their node count (tetra10, line3, ...).
    curved = [t for t in types if t[-1].isdigit()]
    if curved:
        raise ValueError(
            f"{path.name} is a second-order (curved) mesh: it has "
            f"{' / '.join(curved)} elements. gfem supports straight-sided meshes "
            "only. Regenerate it with gmsh `ElementOrder 1` (Python: "
            'gmsh.option.setNumber("Mesh.ElementOrder", 1); command line: '
            "gmsh -order 1)."
        )
    other = [t for t in types if t not in (*width, "line", "vertex")]
    if other:
        raise ValueError(
            f"{path.name} has {' / '.join(other)} elements. gfem supports "
            "tetrahedral meshes only (triangles on surfaces)."
        )
    parts: dict[str, list[tuple]] = {k: [] for k in width}
    for block, phys, geom in zip(
        raw.cells,
        raw.cell_data["gmsh:physical"],
        raw.cell_data["gmsh:geometrical"],
        strict=True,
    ):
        if block.type in width:
            parts[block.type].append((block.data, phys, geom))
    _check_overlaps_in_cell_sets(raw, path)
    out = {}
    for kind, items in parts.items():
        if items:
            data, phys, geom = (np.concatenate(c) for c in zip(*items, strict=True))
        else:
            data, phys, geom = np.empty((0, width[kind])), np.empty(0), np.empty(0)
        grouped = phys != 0  # gmsh writes group 0 for elements outside any group
        if kind == "tetra" and not grouped.all():
            raise ValueError(
                f"{path.name}: {int((~grouped).sum())} tetrahedra lie outside any "
                "volume physical group. Every volume needs a named region: add "
                "the missing volumes to a physical group in gmsh."
            )
        out[kind] = (
            data[grouped].astype(np.int64),
            phys[grouped].astype(np.int64),
            geom[grouped].astype(np.int64),
        )
        _check_duplicates(out[kind][0], out[kind][1], raw, kind, path)
    return out


def _overlap_error(path: Path, names: set[str]) -> ValueError:
    return ValueError(
        f"{path.name}: elements belong to more than one physical group "
        f"({', '.join(repr(n) for n in sorted(names))}). gfem needs each "
        "tetrahedron in exactly one volume region and each triangle in at most "
        "one surface region: split the overlapping groups in gmsh so that no "
        "entity is in two of them, and export the mesh again."
    )


def _check_overlaps_in_cell_sets(raw: meshio.Mesh, path: Path) -> None:
    """Msh 4.1: meshio keeps the first physical tag; its cell sets keep all."""
    names = [n for n, (_, dim) in raw.field_data.items() if int(dim) in (2, 3)]
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
    dim = 3 if kind == "tetra" else 2
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


def _check_tags(topology: _Topology, path: Path) -> None:
    """Every element's physical group must exist and be named."""
    for dim, tags, kind in (
        (3, topology.tet_tags, "tetrahedra"),
        (2, topology.tri_tags, "triangles"),
    ):
        named = {r.tag for r in topology.regions if r.dim == dim}
        missing = sorted(set(np.unique(tags).tolist()) - named)
        if missing:
            raise ValueError(
                f"{path.name}: {kind} belong to {dim}D physical group(s) "
                f"{missing} that have no name of their own in $PhysicalNames. "
                "gfem addresses regions by name: give every physical group a "
                "unique name in gmsh and export the mesh again."
            )
