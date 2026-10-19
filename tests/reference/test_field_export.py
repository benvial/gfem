"""Field export on the reference plate stack (electrostatic/r1).

The TOP terminal's potential is written to VTU and opened in pyvista: every
tetrahedron is cut into order**3 = 8 sub-tetrahedra that tile the mesh, and
the sampled potential is 1 V on the TOP plate, 0 V on the grounded and
other-terminal conductors, and stays within [0, 1] V up to the order-2
discretization's overshoot.
"""

import os
from pathlib import Path

import meshio
import numpy as np
import pytest

import gfem


@pytest.fixture(scope="module")
def cap():
    ref = os.environ.get("GFEM_REFERENCE_DIR")
    if not ref:
        pytest.skip("GFEM_REFERENCE_DIR not set (needs the electrostatic/r1 mesh)")
    mesh = gfem.Mesh.from_gmsh(Path(ref) / "electrostatic" / "r1" / "palace.msh")
    model = gfem.Model(
        mesh,
        materials={"sio2": gfem.Material(eps_r=3.9), "air": gfem.Material()},
    )

    def net(name):
        return [s for s in mesh.surfaces if s.endswith(f"@{name}")]

    problem = gfem.Electrostatic(
        terminals={"TOP": net("TOP"), "LEFT": net("net1"), "RIGHT": net("net2")},
        ground=net("net3"),
    )
    return gfem.solve(model, problem, options=gfem.SolverOptions(order=2)), model


def _volumes(points, cells):
    p = points[cells]
    a, b, c = (p[:, k] - p[:, 0] for k in (1, 2, 3))
    return np.einsum("ij,ij->i", np.cross(a, b), c) / 6


def test_plate_stack_potential_exports_to_vtu_and_pyvista(cap, tmp_path):
    result, model = cap
    v = result.potential["TOP"]
    path = tmp_path / "top.vtu"
    v.to_vtu(path)
    out = meshio.read(path)
    cells = out.cells_dict["tetra"]
    mesh = model.mesh
    assert len(cells) == 8 * len(mesh.tetrahedra)
    vol = _volumes(out.points, cells)
    assert (vol > 0).all()
    np.testing.assert_allclose(
        vol.sum(), np.abs(_volumes(np.asarray(mesh.nodes), mesh.tetrahedra)).sum()
    )
    values = out.point_data["V"]
    assert values.min() > -0.02
    assert values.max() < 1.02

    # 1 V on the TOP plate, 0 V on the other conductors
    def on(surfaces):
        tags = [mesh.region(s).tag for s in surfaces]
        nodes = np.asarray(mesh.nodes)[
            np.unique(mesh.triangles[np.isin(mesh.triangle_tags, tags)])
        ]
        hit = np.isin(
            np.round(out.points / 1e-9).view([("", out.points.dtype)] * 3),
            np.round(nodes / 1e-9).view([("", nodes.dtype)] * 3),
        ).ravel()
        assert hit.sum() >= len(nodes)
        return values[hit]

    top = [s for s in mesh.surfaces if s.endswith("@TOP")]
    others = [s for s in mesh.surfaces if "@net" in s]
    np.testing.assert_allclose(on(top), 1.0, atol=1e-12)
    np.testing.assert_allclose(on(others), 0.0, atol=1e-12)

    pv = pytest.importorskip("pyvista")
    grid = v.to_pyvista()
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells == len(cells)
    np.testing.assert_allclose(grid.point_data["V"], values)
