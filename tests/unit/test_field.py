"""Fields carried by results: the descriptor, quantity rules and export."""

import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import structured_meshes as sm

import gfem

DATA = Path(__file__).parent / "data"
LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, LAYERS, (2, 2, 1)))
    return gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)


@pytest.fixture(scope="module")
def cap(stack):
    problem = gfem.Electrostatic(
        terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
    )
    return gfem.solve(stack, problem, options=gfem.SolverOptions(order=2))


def test_potential_is_one_field_per_terminal_in_terminal_order(cap):
    assert list(cap.potential) == ["top", "mid"]
    v = cap.potential["top"]
    assert isinstance(v, gfem.Field)
    assert (v.quantity, v.unit, v.space, v.order) == ("V", "V", "h1", 2)
    assert v.support == ("oxide", "nitride")
    assert v.omega is None
    assert v.coefficients.ndim == 1
    assert v.mesh is cap.potential["mid"].mesh


def test_unknown_terminal_names_the_known_ones(cap):
    with pytest.raises(KeyError, match="'top', 'mid'"):
        cap.potential["bottom"]


def _analytic(name, z_um):
    """Terminal potentials of the stack: 1 V on the terminal, 0 V elsewhere."""
    if name == "top":
        return np.where(z_um <= 0.5, 0.0, (z_um - 0.5) / 0.25)
    return np.where(z_um <= 0.5, z_um / 0.5, (0.75 - z_um) / 0.25)


def _volumes(points, cells):
    p = points[cells]
    return (
        np.einsum(
            "ij,ij->i",
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
            p[:, 3] - p[:, 0],
        )
        / 6
    )


def _like(v, **change):
    kwargs = {
        "coefficients": v.coefficients,
        "mesh": v.mesh,
        "quantity": "V",
        "unit": "V",
        "space": "h1",
        "order": v.order,
        "support": v.support,
    }
    return gfem.Field(**(kwargs | change))


@pytest.mark.parametrize("name", ["top", "mid"])
def test_meshio_export_samples_the_potential_in_si(cap, name):
    out = cap.potential[name].to_meshio()
    z_um = out.points[:, 2] * 1e6
    np.testing.assert_allclose(out.point_data["V"], _analytic(name, z_um), atol=1e-12)


def test_export_cuts_each_tetrahedron_order_cubed_times_without_gaps(cap, stack):
    out = cap.potential["top"].to_meshio()
    (block,) = out.cells
    mesh = stack.mesh
    assert block.type == "tetra"
    assert len(block.data) == 8 * len(mesh.tetrahedra)
    vol = _volumes(out.points, block.data)
    assert (vol > 0).all()
    np.testing.assert_allclose(vol.sum(), 4e-6 * 3e-6 * 0.75e-6, rtol=1e-12)
    regions = out.cell_data["region"][0]
    assert sorted(np.unique(regions)) == sorted(np.unique(mesh.tetrahedron_tags))
    oxide = mesh.region("oxide").tag
    np.testing.assert_allclose(
        vol[regions == oxide].sum(), 4e-6 * 3e-6 * 0.5e-6, rtol=1e-12
    )


def test_vtu_file_round_trips_through_meshio(cap, tmp_path):
    import meshio

    path = tmp_path / "top.vtu"
    cap.potential["top"].to_vtu(path)
    back = meshio.read(path)
    direct = cap.potential["top"].to_meshio()
    np.testing.assert_allclose(back.points, direct.points)
    np.testing.assert_allclose(back.point_data["V"], direct.point_data["V"])
    assert "region" in back.cell_data
    assert "entity" in back.cell_data


def test_complex_fields_export_real_and_imaginary_parts(cap):
    v = cap.potential["mid"]
    out = _like(v, coefficients=v.coefficients * (1 - 2j)).to_meshio()
    assert set(out.point_data) == {"V_real", "V_imag"}
    exact = _analytic("mid", out.points[:, 2] * 1e6)
    np.testing.assert_allclose(out.point_data["V_real"], exact, atol=1e-12)
    np.testing.assert_allclose(out.point_data["V_imag"], -2 * exact, atol=1e-12)


def test_export_leaves_out_excluded_volumes():
    mesh = gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")
    model = gfem.Model(
        mesh,
        materials={"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()},
        excluded=(gfem.enclosed_by("metal_xy", "metal_z"),),
    )
    problem = gfem.Electrostatic(
        terminals={"metal": ["metal_xy", "metal_z"]}, ground="sub__None"
    )
    v = gfem.solve(model, problem, options=gfem.SolverOptions(order=1)).potential[
        "metal"
    ]
    assert set(v.support) == {"sub", "air"}
    out = v.to_meshio()
    ((_, entity),) = model.excluded_entities
    assert entity not in out.cell_data["entity"][0]
    assert len(out.cells[0].data) == int(model._domain.active.sum())


def test_potential_has_no_magnetic_field(cap):
    with pytest.raises(ValueError, match="quantity 'E'"):
        cap.potential["top"].b()


def test_export_under_jit_raises_with_the_remedy(cap):
    v = cap.potential["top"]
    with pytest.raises(TypeError, match="outside the transformed"):
        jax.jit(lambda c: _like(v, coefficients=c).to_meshio())(v.coefficients)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"space": "nedelec"}, "Unknown field space"),
        ({"order": -1}, "order"),
        ({"quantity": ""}, "quantity"),
        ({"support": ("nowhere",)}, "'nowhere'"),
        ({"coefficients": np.zeros((3, 2))}, "one-dimensional"),
    ],
)
def test_inconsistent_descriptors_raise(cap, change, message):
    with pytest.raises(ValueError, match=message):
        _like(cap.potential["top"], **change)


def test_a_field_from_another_mesh_is_refused_on_export(cap):
    v = cap.potential["top"]
    with pytest.raises(ValueError, match="does not belong"):
        _like(v, coefficients=v.coefficients[:-1]).to_meshio()


def test_result_fields_are_pytree_leaves_with_readable_paths(cap):
    leaves = jax.tree_util.tree_leaves_with_path(cap)
    paths = [jax.tree_util.keystr(p) for p, _ in leaves]
    assert ".potential['top'].coefficients" in paths
    assert ".potential['mid'].mesh.nodes" in paths
    doubled = jax.tree.map(lambda x: 2 * x, cap)
    assert list(doubled.potential) == ["top", "mid"]
    np.testing.assert_allclose(
        doubled.potential["mid"].coefficients, 2 * cap.potential["mid"].coefficients
    )
    assert doubled.potential["mid"].support == cap.potential["mid"].support


OMEGA = 2 * np.pi * 1e9


@pytest.fixture(scope="module")
def e_field(stack):
    """E = (-y, x, 0) V/um**2 in mesh units: curl E = 2 z / um**2."""
    import ngsolve as ngs

    from gfem._engine.fields import fe_space

    mesh = stack.mesh
    support = ("oxide", "nitride")
    _, space, _ = fe_space(mesh, support, (), "hcurl", 2)
    gf = ngs.GridFunction(space)
    gf.Set(ngs.CF((-ngs.y, ngs.x, 0)))
    return gfem.Field(
        coefficients=gf.vec.FV().NumPy().copy(),
        mesh=mesh,
        quantity="E",
        unit="V/m",
        space="hcurl",
        order=2,
        support=support,
        omega=OMEGA,
    )


def test_e_field_exports_in_volts_per_metre(e_field):
    out = e_field.to_meshio()
    x, y = out.points[:, 0], out.points[:, 1]
    # 1 V/um**2 * (x / 1 um) at x metres -> x / 1e-12 V/m
    np.testing.assert_allclose(out.point_data["E"][:, 0], -y / 1e-12, atol=1e-6)
    np.testing.assert_allclose(out.point_data["E"][:, 1], x / 1e-12, atol=1e-6)


def test_b_is_curl_e_over_minus_i_omega_in_tesla(e_field):
    b = e_field.b()
    assert (b.quantity, b.unit, b.space, b.order) == ("B", "T", "hdiv", 1)
    assert b.omega == e_field.omega
    out = b.to_meshio()
    # curl E = 2 V/um**2 = 2e12 V/m**2 along z; B = curl E / (-i w) = 2e12 i / w
    np.testing.assert_allclose(out.point_data["B_real"], 0, atol=1e-6)
    np.testing.assert_allclose(out.point_data["B_imag"][:, :2], 0, atol=1e-6)
    np.testing.assert_allclose(out.point_data["B_imag"][:, 2], 2e12 / OMEGA)


def test_b_is_differentiable_and_jittable(e_field):
    eager = e_field.b().coefficients
    jitted = jax.jit(lambda f: f.b().coefficients)(e_field)
    np.testing.assert_allclose(jitted, eager, rtol=1e-14)

    def energy(omega):
        f = gfem.Field(
            coefficients=e_field.coefficients,
            mesh=e_field.mesh,
            quantity="E",
            unit="V/m",
            space="hcurl",
            order=2,
            support=e_field.support,
            omega=omega,
        )
        return (abs(f.b().coefficients) ** 2).sum()

    # |B|^2 scales as 1 / w**2
    np.testing.assert_allclose(
        jax.grad(energy)(OMEGA), -2 * energy(OMEGA) / OMEGA, rtol=1e-12
    )


def test_b_needs_a_time_harmonic_field(e_field):
    static = _like(e_field, quantity="E", unit="V/m", space="hcurl", omega=None)
    with pytest.raises(ValueError, match="omega is None"):
        static.b()


def test_to_pyvista_carries_points_cells_and_values(cap):
    pv = pytest.importorskip("pyvista")
    grid = cap.potential["top"].to_pyvista()
    out = cap.potential["top"].to_meshio()
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells == len(out.cells[0].data)
    np.testing.assert_allclose(grid.points, out.points)
    np.testing.assert_allclose(grid.point_data["V"], out.point_data["V"])
    np.testing.assert_array_equal(grid.cell_data["region"], out.cell_data["region"][0])


def test_to_pyvista_without_pyvista_names_the_extra(cap, monkeypatch):
    # a None entry in sys.modules makes `import pyvista` raise ImportError
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(ImportError, match=r"gfem\[viz\]"):
        cap.potential["top"].to_pyvista()
