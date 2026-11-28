"""The lithium niobate grating cell of the metasurface second-harmonic recipe.

A periodic cell in um, ``period`` along x and ``depth`` along y, whose bottom
face (``"bottom"``) is inside the silica ``"substrate"`` and whose top face
(``"top"``) is in the air ``"cover"``. The ridges ``"ridge"`` are ``width``
wide, ``height`` tall and centred in the cell; the air between two of them is
``"gap"``. The mesh is a box of hexahedra each cut into six tetrahedra (the
same cut everywhere, so opposite faces match under a translation, as
`gfem.Periodic` needs), written as a gmsh 2.2 file and read back through
`gfem.Mesh.from_gmsh`, as a mesh of your own would be.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import gfem

BOX = np.array(list(itertools.product((0, 1), repeat=3)))[:, ::-1]  # x fastest


def grating_cell(
    path: str | Path,
    period: float = 0.9,
    width: float = 0.45,
    height: float = 0.35,
    *,
    depth: float = 0.1,
    substrate: float = 0.4,
    cover: float = 0.5,
    h: float = 0.05,
) -> gfem.Mesh:
    """Mesh the cell with edge lengths near ``h`` (lengths in um)."""
    axes = [
        np.linspace(0, length, max(2, round(length / h)) + 1)
        for length in (period, depth, substrate + height + cover)
    ]
    # the ridge edges must be grid planes: an even number of cells across the ridge
    # puts them there when the ridge is centred in a period that is a multiple of
    # its width
    across = 2 * max(1, round(width / (2 * h)))
    axes[0] = np.linspace(0, period, round(across * period / width) + 1)
    shape = [len(a) for a in axes]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    ids = np.arange(len(nodes)).reshape(shape)
    # the cells of the upper half in x are mirror images of the lower half's, so the
    # mesh is symmetric under x -> period - x (the faces along x keep their cut)
    half = (shape[0] - 1) // 2
    cells = [
        [ids[i + (1 - a if i >= half else a), j + b, k + c] for a, b, c in BOX]
        for i, j, k in itertools.product(*(range(n - 1) for n in shape))
    ]
    kuhn = []  # one tetrahedron per order of walking x, y, z from corner 0 to 7
    for order in itertools.permutations(range(3)):
        step, walk = np.zeros(3, int), [0]
        for axis in order:
            step[axis] = 1
            walk.append(int(np.flatnonzero((step == BOX).all(1))[0]))
        kuhn.append(walk)
    tets = np.array(cells)[:, kuhn].reshape(-1, 4)
    p = nodes[tets]
    volume = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    tets[volume < 0] = tets[volume < 0][:, [0, 2, 1, 3]]

    def region(c):
        if c[2] < substrate:
            return "substrate"
        if c[2] > substrate + height:
            return "cover"
        return "ridge" if abs(c[0] - period / 2) < width / 2 else "gap"

    names = [region(c) for c in nodes[tets].mean(axis=1)]
    faces = tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3)
    _, first, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    sides = {  # a boundary triangle lies in a plane of the box
        (0, 0): "x_min", (0, 1): "x_max", (1, 0): "y_min", (1, 1): "y_max",
        (2, 0): "bottom", (2, 1): "top",
    }  # fmt: skip
    top = nodes.max(axis=0)
    triangles = []
    for f in first[counts == 1]:
        x = nodes[faces[f]]
        for axis, end in itertools.product(range(3), (0, 1)):
            if np.allclose(x[:, axis], end * top[axis]):
                triangles.append((sides[axis, end], faces[f]))
    tags = {
        n: k for k, n in enumerate(dict.fromkeys(names + [t for t, _ in triangles]), 1)
    }
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames"]
    lines += [str(len(tags))]
    lines += [f'{3 if n in names else 2} {k} "{n}"' for n, k in tags.items()]
    lines += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [
        f"{i + 1} {x:.17g} {y:.17g} {z:.17g}" for i, (x, y, z) in enumerate(nodes)
    ]
    lines += ["$EndNodes", "$Elements", str(len(triangles) + len(tets))]
    elements = [(2, t, f) for t, f in triangles] + list(
        zip((4,) * len(tets), names, tets, strict=True)
    )
    lines += [
        f"{k} {kind} 2 {tags[n]} {tags[n]} " + " ".join(str(v + 1) for v in vertices)
        for k, (kind, n, vertices) in enumerate(elements, 1)
    ]
    lines += ["$EndElements"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gfem.Mesh.from_gmsh(str(path), length_unit="um")
