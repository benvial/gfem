"""Build the NGSolve mesh of a `gfem.Mesh` from its arrays.

``netgen.read_gmsh`` is never used: it drops gmsh physical numbers and sets
every face's ``domin`` / ``domout`` to 1 / 0.
"""

from __future__ import annotations

import dataclasses

import netgen.meshing as ngm
import ngsolve as ngs
import numpy as np

from gfem._periodic import Pairing, check_nodes
from gfem.mesh import LENGTH_UNITS, Mesh


@dataclasses.dataclass(frozen=True, eq=False)
class EngineMesh:
    """An NGSolve mesh and how its elements map back to the `gfem.Mesh`.

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
    """

    mesh: ngs.Mesh
    scale: float
    domains: tuple[tuple[int, int], ...]
    tetrahedron_order: np.ndarray
    triangle_order: np.ndarray


def to_ngsolve(mesh: Mesh, periodic: tuple[Pairing, ...] = ()) -> EngineMesh:
    """Build the NGSolve mesh at the current node coordinates of ``mesh``.

    Each pairing of ``periodic`` (a model's periodic pairs, in order) is
    replayed as the netgen point identifications number j + 1, first
    surface's node as the master: NGSolve derives the periodic edges and
    faces from them, and ``Periodic`` spaces read them. Plain spaces are the
    same on the identified mesh.

    Raises:
        ValueError: The nodes have moved so that a pair no longer matches
            under its translation (NGSolve checks no coordinates).
    """
    check_nodes(np.asarray(mesh.nodes, float), periodic)
    scale = LENGTH_UNITS[mesh.length_unit]
    names = {(r.dim, r.tag): r.name for r in mesh.regions}
    ngmesh = ngm.Mesh(dim=3)  # ty: ignore[unresolved-attribute]
    ngmesh.AddPoints(np.ascontiguousarray(np.asarray(mesh.nodes, float) / scale))

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
        for master, minion in zip(p.master.tolist(), p.minion.tolist(), strict=True):
            ngmesh.AddPointIdentification(
                point(nr0=master), point(nr0=minion), identnr=j, type=2
            )

    return EngineMesh(
        mesh=ngs.Mesh(ngmesh),
        scale=scale,
        domains=tuple((int(t), int(e)) for t, e in domains),
        tetrahedron_order=tet_order,
        triangle_order=tri_order,
    )
