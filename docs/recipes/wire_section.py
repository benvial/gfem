"""The suspended silicon wire of the optomechanical coupling recipe, meshed with netgen.

The section in the (x, y) plane, centred on the origin: a rectangular
``"wire"`` of the given ``width`` and ``height``, surrounded by ``"air"`` that
reaches ``margin`` beyond it on every side. Two meshes of the same wire come
back, with the same global (x, y):

- `optical_section`, for the light: the wire and its air margin as a slab
  section read by `gfem.Mesh.from_gmsh_cross_section`, whose end face
  ``"front"`` carries a wave port and whose outer edge is the line ``"wall"``;
- `mechanical_section`, for the sound: the wire alone as a native 2D mesh, every
  edge free (the wire is suspended).

Netgen meshes on one thread here, so a section meshes the same way on every run
and the page's numbers reproduce.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen.occ import Glue, OCCGeometry, Rectangle

import gfem

UM = 1e-6


def _read(rectangles, maxh, wall_at=None, *, cross_section):
    """Mesh glued ``(x0, y0, x1, y1, name, h)`` rectangles (um) and read them back.

    ``wall_at``: the half sizes (x, y) of the outer edge to name ``"wall"``.
    """
    faces = []
    for x0, y0, x1, y1, name, h in rectangles:
        face = Rectangle(x1 - x0, y1 - y0).Face().Move((x0, y0, 0))
        face.faces.name = name
        face.faces.maxh = h
        faces.append(face)
    shape = Glue(faces)
    if wall_at is not None:
        for edge in shape.edges:
            c = edge.center
            if max(abs(c.x) - wall_at[0], abs(c.y) - wall_at[1]) > -1e-9:
                edge.name = "wall"
    mesh = ngs.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
    nodes = np.array([v.point for v in mesh.vertices])
    triangles = [(el.mat, [v.nr for v in el.vertices]) for el in mesh.Elements(ngs.VOL)]
    lines = [
        (el.mat, [v.nr for v in el.vertices])
        for el in mesh.Elements(ngs.BND)
        if el.mat == "wall"
    ]
    tags: dict[tuple[int, str], int] = {}
    for dim, items in ((1, lines), (2, triangles)):
        for name, _ in items:
            tags.setdefault((dim, name), len(tags) + 1)
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(tags))]
    out += [f'{dim} {k} "{name}"' for (dim, name), k in tags.items()]
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(nodes[:, :2])]
    out += ["$EndNodes", "$Elements", str(len(lines) + len(triangles))]
    k = 0
    for dim, kind, items in ((1, 1, lines), (2, 2, triangles)):
        for name, vertices in items:
            k += 1
            tag = tags[dim, name]
            out.append(
                f"{k} {kind} 2 {tag} {tag} " + " ".join(str(v + 1) for v in vertices)
            )
    out.append("$EndElements")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "wire.msh"
        path.write_text("\n".join(out) + "\n")
        if cross_section:
            return gfem.Mesh.from_gmsh_cross_section(path, length_unit="um")
        return gfem.Mesh.from_gmsh(path, length_unit="um")


def optical_section(
    width: float,
    height: float,
    margin: float = 1.0 * UM,
    *,
    h_wire: float = 0.04 * UM,
    h_air: float = 0.25 * UM,
) -> gfem.Mesh:
    """The wire in its air margin for the light: every length in m."""
    w, t, m = width / UM / 2, height / UM / 2, margin / UM
    big = (w + m, t + m)
    a, b = h_air / UM, h_wire / UM
    rectangles = [
        (-big[0], -big[1], big[0], -t, "air", a),
        (-big[0], t, big[0], big[1], "air", a),
        (-big[0], -t, -w, t, "air", a),
        (w, -t, big[0], t, "air", a),
        (-w, -t, w, t, "wire", b),
    ]
    return _read(rectangles, a, big, cross_section=True)


def mechanical_section(
    width: float, height: float, *, h_wire: float = 0.02 * UM
) -> gfem.Mesh:
    """The wire alone, free on every edge, as a native 2D mesh: every length in m."""
    w, t, h = width / UM / 2, height / UM / 2, h_wire / UM
    return _read([(-w, -t, w, t, "wire", h)], h, cross_section=False)
