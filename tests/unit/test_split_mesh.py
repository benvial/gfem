"""`tools/split_mesh.py`: one uniform split keeps the geometry and every name on it."""

import subprocess
import sys
from pathlib import Path

import meshio
import numpy as np
import split_mesh

import gfem

TOOL = Path(split_mesh.__file__)
LINE = Path(__file__).parents[2] / "docs" / "tutorials" / "data" / "line.msh"


def _unit_tet() -> meshio.Mesh:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    return meshio.Mesh(
        points,
        [("triangle", np.array([[0, 2, 1]])), ("tetra", np.array([[0, 1, 2, 3]]))],
        cell_data={
            "gmsh:physical": [np.array([7]), np.array([9])],
            "gmsh:geometrical": [np.array([70]), np.array([90])],
        },
        field_data={"bottom": np.array([7, 2]), "body": np.array([9, 3])},
    )


def _volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]]) / 6


def _normals(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    corner = points[tris[:, 0]]
    return np.cross(points[tris[:, 1]] - corner, points[tris[:, 2]] - corner) / 2


def _measures(mesh: gfem.Mesh) -> dict[str, float]:
    """Each region's volume or area, from gfem's own reading of the file."""
    nodes = np.asarray(mesh.nodes)
    out = {}
    for region in mesh.regions:
        rows = mesh.elements(region.name)
        if region.dim == 3:
            size = np.abs(_volumes(nodes, mesh.tetrahedra[rows]))
        else:
            size = np.linalg.norm(_normals(nodes, mesh.triangles[rows]), axis=1)
        out[region.name] = float(size.sum())
    return out


def test_one_tetrahedron_splits_into_eight_of_an_eighth_of_its_volume():
    split = split_mesh.split(_unit_tet())
    tets = split.cells_dict["tetra"]
    assert len(split.points) == 10
    assert len(tets) == 8
    assert np.allclose(_volumes(split.points, tets), 1 / 48)


def test_one_triangle_splits_into_four_facing_the_same_way():
    split = split_mesh.split(_unit_tet())
    tris = split.cells_dict["triangle"]
    assert len(tris) == 4
    assert np.allclose(_normals(split.points, tris), [0, 0, -1 / 8])


def test_tags_and_groups_follow_each_child():
    split = split_mesh.split(_unit_tet())
    tags = {
        block.type: (physical.tolist(), geometrical.tolist())
        for block, physical, geometrical in zip(
            split.cells,
            split.cell_data["gmsh:physical"],
            split.cell_data["gmsh:geometrical"],
            strict=True,
        )
    }
    assert tags == {"triangle": ([7] * 4, [70] * 4), "tetra": ([9] * 8, [90] * 8)}
    assert {k: v.tolist() for k, v in split.field_data.items()} == {
        "bottom": [7, 2],
        "body": [9, 3],
    }


def test_the_inner_tetrahedra_share_the_shortest_diagonal():
    # stretched along x, so the midpoints of 01 and 23 are 4 apart and those of
    # 02 and 13, or 03 and 12, about 4.03: nothing shorter than the first pair
    points = np.array([[0, 0, 0], [8, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    mesh = meshio.Mesh(
        points,
        [("tetra", np.array([[0, 1, 2, 3]]))],
        cell_data={
            "gmsh:physical": [np.array([1])],
            "gmsh:geometrical": [np.array([1])],
        },
    )
    split = split_mesh.split(mesh)
    inner = split.cells_dict["tetra"][4:]
    shared = set(inner[0]) & set(inner[1]) & set(inner[2]) & set(inner[3])
    a, b = split.points[sorted(shared)]
    assert np.isclose(np.linalg.norm(a - b), np.sqrt(16 + 0.25 + 0.25))
    assert np.all(_volumes(split.points, split.cells_dict["tetra"]) > 0)


def test_the_command_writes_a_mesh_gfem_reads_with_its_regions_kept(tmp_path):
    out = tmp_path / "split.msh"
    done = subprocess.run(
        [sys.executable, str(TOOL), str(LINE), str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "tetrahedra" in done.stdout

    source = gfem.Mesh.from_gmsh(LINE, length_unit="mm")
    mesh = gfem.Mesh.from_gmsh(out, length_unit="mm")
    assert mesh.regions == source.regions
    want, got = _measures(source), _measures(mesh)
    assert want.keys() == got.keys()
    for name, size in want.items():
        assert np.isclose(got[name], size, rtol=1e-12), name
    for region in source.regions:
        children = 8 if region.dim == 3 else 4
        assert len(mesh.elements(region.name)) == children * len(
            source.elements(region.name)
        )
