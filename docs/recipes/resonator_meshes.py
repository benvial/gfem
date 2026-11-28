"""The cantilever of the resonators recipe, meshed with netgen.

``cantilever`` is a straight beam along x, ``length`` long, ``width`` wide (y) and
``thickness`` thick (z), one ``"beam"`` region. Its root face, x = 0, is the
surface ``"anchor"``; the other faces are free. Netgen meshes on one thread
here, so the beam meshes the same way on every run and the page's numbers
reproduce.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen.occ import Box, OCCGeometry, X

import gfem

UM = 1e-6


def cantilever(
    length: float = 100 * UM,
    width: float = 10 * UM,
    thickness: float = 2 * UM,
    *,
    h: float = 1 * UM,
) -> gfem.Mesh:
    """Mesh the beam: every length in m, ``h`` the edge length."""
    size = tuple(v / UM for v in (length, width, thickness))
    shape = Box((0, 0, 0), size)
    shape.mat("beam")
    shape.faces.Min(X).name = "anchor"
    mesh = ngs.Mesh(OCCGeometry(shape).GenerateMesh(maxh=h / UM))
    nodes = np.array([v.point for v in mesh.vertices])
    tets = [[v.nr for v in el.vertices] for el in mesh.Elements(ngs.VOL)]
    tris = [
        [v.nr for v in el.vertices]
        for el in mesh.Elements(ngs.BND)
        if el.mat == "anchor"
    ]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "2"]
    out += [
        '2 1 "anchor"',
        '3 2 "beam"',
        "$EndPhysicalNames",
        "$Nodes",
        str(len(nodes)),
    ]
    out += [f"{i + 1} {x:.17g} {y:.17g} {z:.17g}" for i, (x, y, z) in enumerate(nodes)]
    out += ["$EndNodes", "$Elements", str(len(tris) + len(tets))]
    k = 0
    for kind, tag, items in ((2, 1, tris), (4, 2, tets)):
        for vertices in items:
            k += 1
            out.append(
                f"{k} {kind} 2 {tag} {tag} " + " ".join(str(v + 1) for v in vertices)
            )
    out.append("$EndElements")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cantilever.msh"
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return gfem.Mesh.from_gmsh(path, length_unit="um")
