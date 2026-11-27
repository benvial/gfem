"""The 1x2 multimode interference coupler of the mode cascade recipe, in netgen.

A 2D device in the (x, y) plane, lengths in um, in a box ``WIDTH`` tall whose
top and bottom walls (``"side"``) are perfect conductors. The input guide
(``"core"``, ``GUIDE`` wide, centred at ``y = WIDTH / 2``) meets the
multimode section at ``x = 0``; the section is ``CORE`` wide and ``length``
long; two output guides, centred ``CORE / 4`` either side of the axis, leave
it. The rest is ``"clad"``. A mesh spans ``[x0, x1]`` of that device; its end
lines are ``"left"`` and ``"right"``:

- `input_block` ends ``depth`` inside the multimode section;
- `output_block` starts ``depth`` inside it;
- `full_device` is the whole coupler at a given ``length``.

Netgen meshes on one thread here, so a mesh is the same on every run and the
page's numbers reproduce.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ngsolve as ngs
from netgen.occ import Glue, OCCGeometry, Rectangle

import gfem

WIDTH = 3.0  # box height, um
CORE = 2.0  # multimode section width, um
GUIDE = 0.5  # access guide width, um
LEAD = 1.0  # length of each access guide, um
H_CORE, H_CLAD = 0.08, 0.2  # edge lengths in the core and in the cladding, um


def _rectangle(xs, ys, name, *, h=None):
    face = Rectangle(xs[1] - xs[0], ys[1] - ys[0]).Face().Move((xs[0], ys[0], 0))
    face.faces.name = name
    if h is not None:
        face.faces.maxh = h
    return face


def device_mesh(x0: float, x1: float, mmi: tuple[float, float]) -> gfem.Mesh:
    """Mesh the coupler between ``x0`` and ``x1``; the multimode section is ``mmi``.

    The input guide fills ``x < mmi[0]``, the output guides ``x > mmi[1]``;
    the section may reach beyond the mesh.
    """
    yc = WIDTH / 2
    a, b = mmi
    cores = [
        (x0, min(a, x1), yc - GUIDE / 2, yc + GUIDE / 2),
        (max(a, x0), min(b, x1), yc - CORE / 2, yc + CORE / 2),
    ]
    for sign in (-1, 1):
        y = yc + sign * CORE / 4
        cores.append((max(b, x0), x1, y - GUIDE / 2, y + GUIDE / 2))
    # the cladding is the box less the cores, so no two faces overlap
    cladding = _rectangle((x0, x1), (0.0, WIDTH), "clad", h=H_CLAD)
    parts = [_rectangle(c[:2], c[2:], "core", h=H_CORE) for c in cores if c[1] > c[0]]
    for part in parts:
        cladding = cladding - part
    cladding.faces.name = "clad"
    shape = Glue([cladding, *parts])
    for edge in shape.edges:
        c = edge.center
        if abs(c.x - x0) < 1e-9:
            edge.name = "left"
        elif abs(c.x - x1) < 1e-9:
            edge.name = "right"
        elif min(abs(c.y), abs(c.y - WIDTH)) < 1e-9:
            edge.name = "side"
    mesh = ngs.Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=H_CLAD))
    nodes = [v.point for v in mesh.vertices]
    triangles = [(el.mat, [v.nr for v in el.vertices]) for el in mesh.Elements(ngs.VOL)]
    segments = [
        (el.mat, [v.nr for v in el.vertices])
        for el in mesh.Elements(ngs.BND)
        if el.mat in ("left", "right", "side")
    ]
    tags: dict[tuple[int, str], int] = {}
    for dim, items in ((1, segments), (2, triangles)):
        for name, _ in items:
            tags.setdefault((dim, name), len(tags) + 1)
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames"]
    lines += [
        str(len(tags)),
        *(f'{dim} {k} "{name}"' for (dim, name), k in tags.items()),
    ]
    lines += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [f"{i + 1} {p[0]:.17g} {p[1]:.17g} 0" for i, p in enumerate(nodes)]
    lines += ["$EndNodes", "$Elements", str(len(segments) + len(triangles))]
    k = 0
    for dim, kind, items in ((1, 1, segments), (2, 2, triangles)):
        for name, vertices in items:
            k += 1
            tag = tags[dim, name]
            lines.append(
                f"{k} {kind} 2 {tag} {tag} " + " ".join(str(v + 1) for v in vertices)
            )
    lines.append("$EndElements")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mmi.msh"
        path.write_text("\n".join(lines) + "\n")
        return gfem.Mesh.from_gmsh(path, length_unit="um")


def input_block(depth: float) -> gfem.Mesh:
    """The input end: the access guide and ``depth`` um of the multimode section."""
    return device_mesh(-LEAD, depth, (0.0, depth + 1.0))


def output_block(depth: float) -> gfem.Mesh:
    """The output end: ``depth`` um of the multimode section and the two guides."""
    return device_mesh(-depth, LEAD, (-depth - 1.0, 0.0))


def full_device(length: float) -> gfem.Mesh:
    """The whole coupler with a multimode section ``length`` um long."""
    return device_mesh(-LEAD, length + LEAD, (0.0, length))
