"""The electromagnetic solve domain of a mesh: which elements are excluded.

Pure NumPy on the `gfem.Mesh` arrays; the engine maps the result to NGSolve.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from gfem.mesh import _TET_FACES, Mesh


@dataclasses.dataclass(frozen=True, eq=False)
class EMDomain:
    """Excluded gmsh volume entities and how each surface meets the domain.

    Compares by identity: it is static pytree data of a `Model`, derived once
    from the mesh topology and the ``excluded=`` entries.

    Attributes:
        excluded: ``(physical tag, entity)`` pairs removed from the domain.
        active: Per tetrahedron, whether it is in the domain.
        sides: Per surface name, how many of its triangles have the domain
            on 0, 1 and 2 sides (``(n0, n1, n2)``).
    """

    excluded: tuple[tuple[int, int], ...]
    active: np.ndarray
    sides: dict[str, tuple[int, int, int]]

    def exterior(self, surface: str) -> bool:
        """Some triangle of ``surface`` has the domain on one side only."""
        return self.sides[surface][1] > 0

    def touches(self, surface: str) -> bool:
        """Some triangle of ``surface`` borders the domain."""
        return self.sides[surface][1] + self.sides[surface][2] > 0

    def interior(self, surface: str) -> bool:
        """Some triangle of ``surface`` has the domain on both sides."""
        return self.sides[surface][2] > 0


def entity_keys(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Distinct ``(tag, entity)`` rows and the row of each tetrahedron."""
    keys = np.stack([mesh.tetrahedron_tags, mesh.tetrahedron_entities], axis=1)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    return uniq, inverse.ravel()


def enclosed_entities(mesh: Mesh, surfaces: tuple[str, ...]) -> set[tuple[int, int]]:
    """Volume entities whose whole boundary lies on the named surfaces.

    An entity's boundary is every tetrahedron face whose other side is
    another entity or outside the mesh.
    """
    keys, dom = entity_keys(mesh)
    tags = [mesh.region(s).tag for s in surfaces]
    shell = np.sort(mesh.triangles[np.isin(mesh.triangle_tags, tags)], axis=1)
    faces = np.sort(mesh.tetrahedra[:, _TET_FACES].reshape(-1, 3), axis=1)
    face_dom = np.repeat(dom, 4)
    _, ids = np.unique(np.concatenate([faces, shell]), axis=0, return_inverse=True)
    ids = ids.ravel()
    face_ids, shell_ids = ids[: len(faces)], ids[len(faces) :]
    # A face seen twice from the same entity is inside it; once, on its boundary.
    pair = face_ids * len(keys) + face_dom
    _, pair_inv, pair_count = np.unique(pair, return_inverse=True, return_counts=True)
    on_boundary = pair_count[pair_inv.ravel()] == 1
    on_shell = np.isin(face_ids, shell_ids)
    bounded = set(np.unique(face_dom[on_boundary]).tolist())
    leaky = set(np.unique(face_dom[on_boundary & ~on_shell]).tolist())
    return {(int(keys[d][0]), int(keys[d][1])) for d in bounded - leaky}


def em_domain(mesh: Mesh, excluded: set[tuple[int, int]]) -> EMDomain:
    """The domain left once the ``excluded`` entities are removed."""
    keys = np.stack([mesh.tetrahedron_tags, mesh.tetrahedron_entities], axis=1)
    out = np.zeros(len(keys), dtype=bool)
    for tag, ent in excluded:
        out |= (keys[:, 0] == tag) & (keys[:, 1] == ent)
    active = ~out
    active.flags.writeable = False
    nb = mesh.triangle_neighbours
    n_active = ((nb >= 0) & active[np.maximum(nb, 0)]).sum(axis=1)
    sides = {}
    for r in mesh.regions:
        if r.dim == 2:
            counts = np.bincount(n_active[mesh.triangle_tags == r.tag], minlength=3)
            sides[r.name] = (int(counts[0]), int(counts[1]), int(counts[2]))
    return EMDomain(excluded=tuple(sorted(excluded)), active=active, sides=sides)
