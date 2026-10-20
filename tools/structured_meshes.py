"""Structured tetrahedral meshes written as gmsh 2.2 files, for tests.

No gmsh needed: a box of hexahedra is cut into six tetrahedra each (the Kuhn
split, conforming across hexahedra), optionally mapped to another shape, and
every boundary face is named by the caller from its centroid in the box's
parameter space.

    path = write_msh22(tmp / "plate.msh", plate(1.0, 1.0, (("oxide", 0.1),), (2, 2, 3)))
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path

import numpy as np

# Kuhn split of the unit cube: one tetrahedron per permutation of the axes.
_CORNERS = np.array(list(itertools.product((0, 1), repeat=3)))[:, ::-1]


def _kuhn() -> np.ndarray:
    tets = []
    for perm in itertools.permutations(range(3)):
        path = [np.zeros(3, int)]
        for axis in perm:
            step = path[-1].copy()
            step[axis] = 1
            path.append(step)
        tets.append([int(np.flatnonzero((p == _CORNERS).all(1))[0]) for p in path])
    return np.array(tets)


Classify = Callable[[np.ndarray], str]


def structured(
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    volume: Classify,
    surface: Classify,
    mapping: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    mirror: tuple[bool, bool, bool] = (False, False, False),
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    """Tetrahedra of the box grid with axis ``edges``, named by centroid.

    Args:
        edges: Grid coordinates along each parameter axis.
        volume: Region name of a tetrahedron from its parameter centroid.
        surface: Region name of a boundary or interface triangle from its
            parameter centroid; ``""`` leaves the triangle out.
        mapping: Maps parameter points ``(n, 3)`` to coordinates.
        mirror: Per axis, whether the cells of the upper half are the
            mirror images of the lower half's (an even number of cells and
            edges symmetric about their midpoint give a mesh symmetric
            under that reflection; the Kuhn split alone is not).

    Returns:
        nodes, tetrahedra, tetrahedron region names, triangles, triangle
        region names. Interface triangles (between two volume regions) are
        offered to ``surface`` too.
    """
    nx, ny, nz = (len(e) for e in edges)
    grid = np.stack(np.meshgrid(*edges, indexing="ij"), axis=-1).reshape(-1, 3)
    ids = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    cubes = []
    halves = [
        (n - 1) // 2 if m else n for n, m in zip((nx, ny, nz), mirror, strict=True)
    ]
    for i, j, k in itertools.product(range(nx - 1), range(ny - 1), range(nz - 1)):
        flip = [c >= h for c, h in zip((i, j, k), halves, strict=True)]
        cubes.append(
            [
                ids[
                    i + (1 - a if flip[0] else a),
                    j + (1 - b if flip[1] else b),
                    k + (1 - c if flip[2] else c),
                ]
                for a, b, c in _CORNERS
            ]
        )
    tets = np.array(cubes)[:, _kuhn()].reshape(-1, 4)
    nodes = grid if mapping is None else mapping(grid)
    # positive orientation in mapped coordinates
    p = nodes[tets]
    det = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    tet_names = [volume(c) for c in grid[tets].mean(axis=1)]

    faces = tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3)
    key = np.sort(faces, axis=1)
    uniq, first, inverse, counts = np.unique(
        key, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    owner = np.repeat(np.arange(len(tets)), 4)
    region_of = {}
    for f, t in zip(inverse.ravel(), owner, strict=True):
        region_of.setdefault(int(f), set()).add(tet_names[t])
    tris, tri_names = [], []
    for f in range(len(uniq)):
        if counts[f] == 2 and len(region_of[f]) == 1:
            continue
        name = surface(grid[uniq[f]].mean(axis=0))
        if name:
            tris.append(faces[first[f]])
            tri_names.append(name)
    return nodes, tets, tet_names, np.array(tris).reshape(-1, 3), tri_names


def write_msh22(path: str | Path, mesh: tuple) -> Path:
    """Write a gmsh 2.2 ASCII file; physical numbers follow first appearance.

    Args:
        path: The file to write.
        mesh: ``(nodes, tets, tet_names, tris, tri_names)`` from `structured`.
    """
    path = Path(path)
    nodes, tets, tet_names, tris, tri_names = mesh
    vols = list(dict.fromkeys(tet_names))
    surfs = list(dict.fromkeys(tri_names))
    tag = {(3, n): i + 1 for i, n in enumerate(vols)}
    tag |= {(2, n): len(vols) + i + 1 for i, n in enumerate(surfs)}
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames"]
    lines.append(str(len(tag)))
    lines += [f'{dim} {t} "{name}"' for (dim, name), t in tag.items()]
    lines += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [
        f"{i + 1} {x:.17g} {y:.17g} {z:.17g}" for i, (x, y, z) in enumerate(nodes)
    ]
    lines += ["$EndNodes", "$Elements", str(len(tris) + len(tets))]
    k = 0
    for tri, name in zip(tris, tri_names, strict=True):
        k += 1
        t = tag[2, name]
        lines.append(f"{k} 2 2 {t} {t} " + " ".join(str(v + 1) for v in tri))
    for tet, name in zip(tets, tet_names, strict=True):
        k += 1
        t = tag[3, name]
        lines.append(f"{k} 4 2 {t} {t} " + " ".join(str(v + 1) for v in tet))
    lines.append("$EndElements")
    path.write_text("\n".join(lines) + "\n")
    return path


def plate(
    width: float,
    depth: float,
    layers: tuple[tuple[str, float], ...],
    n: tuple[int, int, int],
):
    """A parallel-plate stack: ``layers`` of (name, thickness) from z = 0.

    Surfaces: "bottom" (z = 0), "top" (z = total), "side" (x, y walls),
    and "<lower>__<upper>" between layers. ``n`` cells along x, y and each
    layer's z.
    """
    tops = np.cumsum([t for _, t in layers])
    zs = [np.zeros(1)]
    for (_, t), z1 in zip(layers, tops, strict=True):
        zs.append(z1 - t + np.linspace(0, t, n[2] + 1)[1:])
    edges = (
        np.linspace(0, width, n[0] + 1),
        np.linspace(0, depth, n[1] + 1),
        np.concatenate(zs),
    )
    eps = 1e-9 * max(width, depth, tops[-1])

    def volume(c):
        return layers[int(np.searchsorted(tops, c[2]))][0]

    def surface(c):
        if abs(c[2]) < eps:
            return "bottom"
        if abs(c[2] - tops[-1]) < eps:
            return "top"
        for (lo, _), (hi, _), z in zip(layers, layers[1:], tops, strict=False):
            if abs(c[2] - z) < eps:
                return f"{lo}__{hi}"
        return "side"

    return structured(edges, volume, surface)


def coax_quarter(
    inner: float, outer: float, length: float, *, n_r: int, n_theta: int, n_z: int = 1
):
    """A quarter of a coaxial line, a < r < b, 0 < theta < pi/2, 0 < z < length.

    Volume "dielectric". Surfaces "inner" (r = a), "outer" (r = b),
    "symmetry" (theta = 0 and pi/2) and "ends" (z = 0 and length). The
    polygonal conductors have their nodes on the circles.
    """
    edges = (
        np.linspace(inner, outer, n_r + 1),
        np.linspace(0, np.pi / 2, n_theta + 1),
        np.linspace(0, length, n_z + 1),
    )
    tol = 1e-9

    def surface(c):
        if abs(c[0] - inner) < tol * outer:
            return "inner"
        if abs(c[0] - outer) < tol * outer:
            return "outer"
        if abs(c[2]) < tol * length or abs(c[2] - length) < tol * length:
            return "ends"
        return "symmetry"

    def mapping(p):
        r, t, z = p.T
        return np.stack([r * np.cos(t), r * np.sin(t), z], axis=1)

    return structured(edges, lambda _: "dielectric", surface, mapping)


def strip_line(
    length: float,
    width: float,
    height: float,
    n: tuple[int, int, int],
    *,
    layers: tuple[tuple[str, float], ...] | None = None,
    split_ports: bool = False,
    core: tuple[float, float] | None = None,
):
    """A parallel-plate line along x: plates at z = 0 and z = height.

    Volume "dielectric" (or ``layers`` of (name, thickness) from z = 0).
    Surfaces "bottom" (z = 0), "top" (z = height), "port1" (x = 0),
    "port2" (x = length) and "side" (y walls); with ``split_ports`` each
    port face is cut at y = width / 2 into "<port>a" (y below) and
    "<port>b". Layer interfaces are "<lower>__<upper>". ``n`` cells along
    x, y and z (per layer). With ``core = (x0, x1)`` the cells between x0
    and x1 (grid planes) form one volume "core" across every layer; its
    faces towards the layers are left out.
    """
    layers = (("dielectric", height),) if layers is None else layers
    tops = np.cumsum([t for _, t in layers])
    zs = [np.zeros(1)]
    for (_, t), z1 in zip(layers, tops, strict=True):
        zs.append(z1 - t + np.linspace(0, t, n[2] + 1)[1:])
    edges = (
        np.linspace(0, length, n[0] + 1),
        np.linspace(0, width, n[1] + 1),
        np.concatenate(zs),
    )
    eps = 1e-9 * max(length, width, tops[-1])

    def volume(c):
        if core is not None and core[0] < c[0] < core[1]:
            return "core"
        return layers[int(np.searchsorted(tops, c[2]))][0]

    def surface(c):
        for x, name in ((0.0, "port1"), (length, "port2")):
            if abs(c[0] - x) < eps:
                if split_ports:
                    return name + ("a" if c[1] < width / 2 else "b")
                return name
        if abs(c[2]) < eps:
            return "bottom"
        if abs(c[2] - tops[-1]) < eps:
            return "top"
        for (lo, _), (hi, _), z in zip(layers, layers[1:], tops, strict=False):
            if abs(c[2] - z) < eps:
                return f"{lo}__{hi}"
        return "side"

    def named(c):
        # the core's faces towards the layers are interior: left out
        if core is not None and min(abs(c[0] - x) for x in core) < eps:
            return ""
        return surface(c)

    return structured(edges, volume, named)
