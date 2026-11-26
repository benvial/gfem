"""The suspended plate of the mechanical modes recipe, meshed with netgen.

The section in the (x, y) plane: a ``"film"`` of the given ``thickness`` and
``length``, with a ``"frame"`` region ``frame`` long at each end (``"frame_left"``
and ``"frame_right"``), the whole plate ``thickness`` tall. The two outer end
faces are the line ``"end"`` (the plate is anchored there) and the faces above
and below are free. Netgen meshes on one thread here, so a section meshes the
same way on every run and the page's numbers reproduce.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen.occ import Glue, OCCGeometry, Rectangle

import gfem

UM = 1e-6


def plate_section(
    thickness: float = 0.5 * UM,
    length: float = 3 * UM,
    frame: float = 3 * UM,
    *,
    h: float = 0.25 * UM,
) -> gfem.Mesh:
    """Mesh the plate: every length in m, ``h`` the edge length."""
    t, length, frame, h = (v / UM for v in (thickness, length, frame, h))
    pieces = [
        ("frame_left", -frame - length / 2, frame),
        ("film", -length / 2, length),
        ("frame_right", length / 2, frame),
    ]
    faces = []
    for name, x0, width in pieces:
        face = Rectangle(width, t).Face().Move((x0, 0, 0))
        face.faces.name = name
        faces.append(face)
    shape = Glue(faces)
    half = length / 2 + frame
    for edge in shape.edges:
        if abs(abs(edge.center.x) - half) < 1e-9:
            edge.name = "end"
    mesh = ngs.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=h))
    nodes = np.array([v.point for v in mesh.vertices])
    triangles = [(el.mat, [v.nr for v in el.vertices]) for el in mesh.Elements(ngs.VOL)]
    lines = [
        (el.mat, [v.nr for v in el.vertices])
        for el in mesh.Elements(ngs.BND)
        if el.mat == "end"
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
        path = Path(directory) / "plate.msh"
        path.write_text("\n".join(out) + "\n")
        return gfem.Mesh.from_gmsh(path, length_unit="um")
