"""The meshes of the resonators recipe, built with netgen.

``cantilever`` is a straight beam along x, ``length`` long, ``width`` wide (y) and
``thickness`` thick (z), one ``"beam"`` region. Its root face, x = 0, is the
surface ``"anchor"``; the other faces are free.

``anchored_beam`` is a 2D section: an in-plane cantilever on a support, with a
layer of the ``"layer"`` region around the support. Netgen meshes on one thread
here, so each mesh is the same on every run and the page's numbers reproduce.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import ngsolve as ngs
import numpy as np
from netgen.occ import Box, Glue, OCCGeometry, Rectangle, X

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


#: The support's material, silicon-like: Young's modulus (Pa), Poisson's ratio and
#: density (kg/m^3), as the recipe's `gfem.Material`.
YOUNG, POISSON, DENSITY = 199e9, 0.25, 2329.0


def anchored_beam(ratio: float = 10.0, *, maxh: float = 0.5) -> gfem.Mesh:
    """Mesh an in-plane cantilever on a support, ended by a layer.

    The beam is 1 um thick and ``ratio`` um long, standing on the top edge of the
    support (regions ``"beam"`` and ``"support"``, all lengths in um). The support
    is half a shear wavelength deep and wide at the beam's bending frequency and
    the region ``"layer"``, one pressure wavelength thick, wraps its sides and
    bottom. ``maxh`` is the element size in the beam and under its root; the far
    field takes the shear wavelength times ``maxh / 2``.
    """
    mu = YOUNG / (2 * (1 + POISSON))
    lam = YOUNG * POISSON / ((1 + POISSON) * (1 - 2 * POISSON))
    frequency = (  # plane-strain Euler-Bernoulli, first mode of a clamped-free beam
        1.875104068711961**2
        / (2 * np.pi)
        * (1 * UM)
        / (ratio * UM) ** 2
        * np.sqrt(YOUNG / (1 - POISSON**2) / (12 * DENSITY))
    )
    shear = np.sqrt(mu / DENSITY) / frequency / UM
    pressure = np.sqrt((lam + 2 * mu) / DENSITY) / frequency / UM
    a, d, r = shear / 2, pressure, 2.0
    boxes = [  # x0, y0, x1, y1, region, element size (None: the far field's)
        (-0.5, 0.0, 0.5, ratio, "beam", maxh),
        (-r, -r, r, 0.0, "support", maxh),
        (-a, -a, -r, 0.0, "support", None),
        (r, -a, a, 0.0, "support", None),
        (-r, -a, r, -r, "support", None),
        (-a - d, -a, -a, 0.0, "layer", None),
        (a, -a, a + d, 0.0, "layer", None),
        (-a, -a - d, a, -a, "layer", None),
        (-a - d, -a - d, -a, -a, "layer", None),
        (a, -a - d, a + d, -a, "layer", None),
    ]
    faces = []
    for x0, y0, x1, y1, name, size in boxes:
        face = Rectangle(x1 - x0, y1 - y0).Face().Move((x0, y0, 0))
        face.faces.name = name
        if size is not None:
            face.faces.maxh = size
        faces.append(face)
    geometry = OCCGeometry(Glue(faces), dim=2)
    mesh = ngs.Mesh(geometry.GenerateMesh(maxh=shear * maxh / 2))
    nodes = np.array([v.point for v in mesh.vertices])
    triangles = [
        (el.mat, [v.nr + 1 for v in el.vertices]) for el in mesh.Elements(ngs.VOL)
    ]
    names = list(dict.fromkeys(name for name, _ in triangles))
    out = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        str(len(names)),
    ]
    out += [f'2 {i + 1} "{name}"' for i, name in enumerate(names)]
    out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    out += [f"{i + 1} {x:.17g} {y:.17g} 0" for i, (x, y) in enumerate(nodes[:, :2])]
    out += ["$EndNodes", "$Elements", str(len(triangles))]
    for k, (name, vertices) in enumerate(triangles):
        tag = names.index(name) + 1
        out.append(f"{k + 1} 2 2 {tag} {tag} " + " ".join(map(str, vertices)))
    out.append("$EndElements")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "anchored_beam.msh"
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return gfem.Mesh.from_gmsh(path, length_unit="um")
