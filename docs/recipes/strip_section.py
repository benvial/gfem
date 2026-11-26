"""The silicon strip of the waveguide dispersion recipe, meshed with netgen.

The section in the (x, y) plane, y up: a rectangular core ``"core"``,
``width`` wide and ``height`` tall, its bottom at y = 0 and centred on x = 0,
in a rectangular cladding ``"cladding"`` that reaches ``clearance`` beyond the
core on every side, or ``outer`` and ``inner`` beyond it on its +x and -x
sides. With ``layer`` a region ``"pml"``, that thick, follows the cladding on
the +x side: the outer layer of a bent guide. `strip_section` meshes it, writes the mesh as a gmsh 2.2
file and reads it back through `gfem.Mesh.from_gmsh_cross_section`, as a mesh
of your own would be: its end face ``"front"`` carries a wave port. Netgen
meshes on one thread here, so a section meshes the same way on every run and
the page's numbers reproduce.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen.occ import Glue, OCCGeometry, Rectangle

import gfem

UM = 1e-6


def strip_section(
    width: float,
    height: float = 0.22 * UM,
    clearance: float = 1.0 * UM,
    *,
    h_core: float = 0.05 * UM,
    h_cladding: float = 0.4 * UM,
    outer: float | None = None,
    inner: float | None = None,
    layer: float | None = None,
) -> gfem.Mesh:
    """Mesh the strip: every length in m, edge lengths ``h_core`` and ``h_cladding``.

    ``outer`` and ``inner`` replace ``clearance`` on the +x and -x sides;
    ``layer`` adds the ``"pml"`` region beyond the outer cladding.
    """
    w, t, g = width / UM, height / UM, clearance / UM
    g_out = g if outer is None else outer / UM
    g_in = g if inner is None else inner / UM
    cladding = Rectangle(w + g_in + g_out, t + 2 * g).Face().Move((-w / 2 - g_in, -g, 0))
    core = Rectangle(w, t).Face().Move((-w / 2, 0, 0))
    cladding.faces.name, core.faces.name = "cladding", "core"
    core.faces.maxh = h_core / UM
    parts = [cladding, core]
    if layer is not None:
        outer_layer = Rectangle(layer / UM, t + 2 * g).Face().Move((w / 2 + g_out, -g, 0))
        outer_layer.faces.name = "pml"
        parts.append(outer_layer)
    geometry = OCCGeometry(Glue(parts), dim=2)
    mesh = ngs.Mesh(geometry.GenerateMesh(maxh=h_cladding / UM))
    nodes = np.array([v.point for v in mesh.vertices])
    triangles = [(el.mat, [v.nr for v in el.vertices]) for el in mesh.Elements(ngs.VOL)]
    tags = {name: k for k, name in enumerate(dict.fromkeys(n for n, _ in triangles), 1)}
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames"]
    lines += [str(len(tags)), *(f'2 {k} "{name}"' for name, k in tags.items())]
    lines += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(nodes[:, :2])]
    lines += ["$EndNodes", "$Elements", str(len(triangles))]
    lines += [
        f"{k} 2 2 {tags[name]} {tags[name]} " + " ".join(str(v + 1) for v in vertices)
        for k, (name, vertices) in enumerate(triangles, 1)
    ]
    lines.append("$EndElements")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "strip.msh"
        path.write_text("\n".join(lines) + "\n")
        return gfem.Mesh.from_gmsh_cross_section(path, length_unit="um")
