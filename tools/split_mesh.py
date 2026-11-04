"""One uniform split of a gmsh mesh: ``python tools/split_mesh.py IN.msh OUT.msh``.

Each tetrahedron becomes 8 and each triangle 4, on midpoints that neighbours share.
Inside a tetrahedron the four children around the octahedron take its shortest
diagonal. Children keep the parent's physical and entity tags and orientation; the
physical names are copied. Line and point elements are dropped, since gfem does not
read them. The geometry is the mesh's own: a split does not follow a curved surface.

NumPy on meshio's arrays, so it needs no gmsh, and writes binary MSH 2.2. On
a mesh with E edges, F faces and T tetrahedra, order 2 has 2E + 2F DOFs before and
4E + 14F + 18T after.
"""

import argparse
from pathlib import Path

import meshio
import numpy as np

#: A tetrahedron's six edges, in the order of the midpoint columns below.
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
#: The four corner children, by column of ``[corners, midpoints]``.
CORNERS = ((0, 4, 5, 6), (4, 1, 7, 8), (5, 7, 2, 9), (6, 8, 9, 3))
#: Each octahedron diagonal, and the cycle of the four other midpoints around it.
DIAGONALS = (
    ((4, 9), (5, 6, 8, 7)),
    ((5, 8), (4, 6, 9, 7)),
    ((6, 7), (4, 5, 9, 8)),
)
#: A triangle's three edges, and its four children by column of ``[corners, midpoints]``.
TRIANGLE_EDGES = ((0, 1), (0, 2), (1, 2))
TRIANGLE_CHILDREN = ((0, 3, 4), (3, 1, 5), (4, 5, 2), (3, 5, 4))


def _signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    a = points[tets[:, 0]]
    edges = [points[tets[:, k]] - a for k in (1, 2, 3)]
    return np.einsum("ij,ij->i", np.cross(edges[0], edges[1]), edges[2])


def _inner_children(points: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """``(n, 4, 4)``: the octahedron's four tetrahedra on its shortest diagonal."""
    lengths = np.stack(
        [
            np.linalg.norm(points[nodes[:, a]] - points[nodes[:, b]], axis=1)
            for (a, b), _ in DIAGONALS
        ],
        axis=1,
    )
    choice = np.argmin(lengths, axis=1)
    children = np.empty((len(nodes), 4, 4), dtype=np.int64)
    for k, ((a, b), cycle) in enumerate(DIAGONALS):
        rows = choice == k
        for c in range(4):
            local = (a, b, cycle[c], cycle[(c + 1) % 4])
            children[rows, c] = nodes[rows][:, local]
    return children


def _same_way_as(points: np.ndarray, parents: np.ndarray, children: np.ndarray):
    """``children`` ``(n, 8, 4)`` with each one's sign of volume its parent's."""
    sign = np.sign(_signed_volumes(points, parents))
    shape = children.shape
    flat = children.reshape(-1, 4)
    flip = np.sign(_signed_volumes(points, flat)) != np.repeat(sign, shape[1])
    flat[flip] = flat[flip][:, (0, 1, 3, 2)]
    return flat


def split(mesh: meshio.Mesh) -> meshio.Mesh:
    """The mesh split once; see the module's docstring.

    Raises:
        ValueError: The mesh holds a cell type other than tetrahedra, triangles,
            lines and points, or has no gmsh tags on its cells.
    """
    if "gmsh:physical" not in mesh.cell_data:
        raise ValueError("The mesh has no gmsh:physical tags to keep.")
    width = {"tetra": 4, "triangle": 3}
    blocks: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        kind: [] for kind in width
    }
    for block, physical, geometrical in zip(
        mesh.cells,
        mesh.cell_data["gmsh:physical"],
        mesh.cell_data["gmsh:geometrical"],
        strict=True,
    ):
        if block.type in width:
            blocks[block.type].append((block.data, physical, geometrical))
        elif block.type not in ("line", "vertex"):
            raise ValueError(f"Cannot split {block.type} cells.")
    arrays = {}
    for kind, parts in blocks.items():
        if parts:
            arrays[kind] = tuple(np.concatenate(p) for p in zip(*parts, strict=True))
        else:
            arrays[kind] = (
                np.empty((0, width[kind]), dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
    tets = arrays["tetra"][0].astype(np.int64)
    tris = arrays["triangle"][0].astype(np.int64)
    points = np.asarray(mesh.points, float)
    n = len(points)

    # edges as one integer each: a * n + b, a < b
    ends = np.concatenate(
        [tets[:, list(e)] for e in EDGES] + [tris[:, list(e)] for e in TRIANGLE_EDGES]
    )
    keys = ends.min(axis=1) * n + ends.max(axis=1)
    del ends
    unique, inverse = np.unique(keys, return_inverse=True)
    del keys
    a, b = np.divmod(unique, n)
    new_points = np.concatenate([points, (points[a] + points[b]) / 2])
    midpoint = inverse.reshape(-1) + n
    del inverse, a, b, unique
    tet_mid = midpoint[: 6 * len(tets)].reshape(6, -1).T
    tri_mid = midpoint[6 * len(tets) :].reshape(3, -1).T

    nodes = np.concatenate([tets, tet_mid], axis=1)
    corners = np.stack([nodes[:, list(c)] for c in CORNERS], axis=1)
    inner = _inner_children(new_points, nodes)
    children = _same_way_as(
        new_points, tets, np.concatenate([corners, inner], axis=1)
    )
    corner_nodes = np.concatenate([tris, tri_mid], axis=1)
    faces = np.stack([corner_nodes[:, list(c)] for c in TRIANGLE_CHILDREN], axis=1)

    return meshio.Mesh(
        new_points,
        [("triangle", faces.reshape(-1, 3)), ("tetra", children)],
        cell_data={
            name: [
                np.repeat(arrays["triangle"][k], 4).astype(np.int32),
                np.repeat(arrays["tetra"][k], 8).astype(np.int32),
            ]
            for k, name in ((1, "gmsh:physical"), (2, "gmsh:geometrical"))
        },
        field_data=dict(mesh.field_data),
    )


def main() -> None:
    """Read, split, write and print the counts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    out = split(meshio.read(args.source, file_format="gmsh"))
    meshio.gmsh.write(args.target, out, fmt_version="2.2", binary=True)
    tets, tris = out.cells_dict["tetra"], out.cells_dict["triangle"]
    print(  # noqa: T201
        f"{args.target}: {len(tets)} tetrahedra, {len(tris)} triangles, "
        f"{len(out.points)} nodes"
    )


if __name__ == "__main__":
    main()
