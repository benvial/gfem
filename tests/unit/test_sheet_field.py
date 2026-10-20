"""Sheet fields: a per-face fill scaling a Conductor or Impedance sheet.

The line of these tests is a parallel-plate line along x whose plates are
its y walls, glued from two halves at mid-height: the interior surface
"sheet" there is tangential to the TEM field, so a sheet loads the line.
A fill scales the solid sheet's admittance face by face, so the forward
checks are exact equivalences: a uniform fill w on an impedance of R per
square is the impedance R / w, a fill of 1 on part of the sheet is the
solid sheet on that part alone (the same mesh with fewer faces named
"sheet"), and a fill of 0 is no sheet at all.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from jax.test_util import check_grads

import gfem

L, W, H = 8.0, 2.0, 1.0  # mm
CELLS = (8, 2, 1)
F = 6e9
SPAN = (2.0, 5.0)  # mm


def _mesh(tmp_path_factory, name, **kwargs):
    path = tmp_path_factory.mktemp(name) / "sheet.msh"
    sm.write_msh22(path, sm.sheet_line(L, W, H, CELLS, **kwargs))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return _mesh(tmp_path_factory, "full")


@pytest.fixture(scope="module")
def partial_mesh(tmp_path_factory):
    return _mesh(tmp_path_factory, "partial", span=SPAN)


def _model(mesh, *sheet):
    return gfem.Model(
        mesh,
        materials={"lower": gfem.Material(eps_r=2.0), "upper": gfem.Material()},
        boundaries=[gfem.PEC("plate1"), gfem.PEC("plate2"), *sheet],
        ports=[
            gfem.LumpedPort("p1", "port1", "+y", impedance=50.0),
            gfem.LumpedPort("p2", "port2", "+y", impedance=50.0),
        ],
    )


def _s(model):
    options = gfem.SolverOptions(linear_solver="direct")
    return np.asarray(
        gfem.solve(model, gfem.Driven(frequency=F), options=options).s.values
    )


def _faces(mesh):
    return len(mesh.elements("sheet"))


def test_uniform_fill_scales_the_sheet_admittance(mesh):
    fill = gfem.SheetField(fill=np.full(_faces(mesh), 0.25))
    filled = _s(_model(mesh, gfem.Impedance("sheet", resistance=40.0, fill=fill)))
    solid = _s(_model(mesh, gfem.Impedance("sheet", resistance=160.0)))
    np.testing.assert_allclose(filled, solid, rtol=0, atol=1e-12)
    bare = _s(_model(mesh))
    assert np.abs(filled - bare).max() > 1e-2  # the sheet does load the line


def _centroids(mesh, surface="sheet"):
    nodes = np.asarray(mesh.nodes)
    return nodes[mesh.triangles[mesh.elements(surface)]].mean(axis=1)


def test_fill_of_one_on_part_of_a_sheet_is_that_part_alone(mesh, partial_mesh):
    x = _centroids(mesh)[:, 0] * 1e3
    on = (SPAN[0] < x) & (x < SPAN[1])
    fill = gfem.SheetField(fill=on.astype(float))
    sigma, t = 1e5, 1e-6
    filled = _s(_model(mesh, gfem.Conductor("sheet", sigma, t, fill=fill)))
    part = _s(_model(partial_mesh, gfem.Conductor("sheet", sigma, t)))
    np.testing.assert_allclose(filled, part, rtol=0, atol=1e-12)


def test_fill_of_zero_is_no_sheet(mesh):
    fill = gfem.SheetField(fill=np.zeros(_faces(mesh)))
    empty = _s(_model(mesh, gfem.Impedance("sheet", inductance=1e-12, fill=fill)))
    np.testing.assert_allclose(empty, _s(_model(mesh)), rtol=0, atol=1e-12)


# --- the value class ---------------------------------------------------------------


def test_sheet_field_holds_one_real_nonnegative_fill_per_face():
    field = gfem.SheetField(fill=[0.0, 0.5, 2.0], surface="sheet")
    assert field.n_faces == 3
    assert field.fill.dtype == np.float64
    assert "3 faces" in repr(field)
    with pytest.raises(ValueError, match="real"):
        gfem.SheetField(fill=np.ones(3) * (1 + 1j))
    with pytest.raises(ValueError, match="one value per face"):
        gfem.SheetField(fill=np.ones((2, 2)))
    with pytest.raises(ValueError, match=">= 0"):
        gfem.SheetField(fill=[0.5, -0.1])


def test_exponential_fill_is_log_scale_with_exact_ends():
    rho = np.array([0.0, 0.5, 1.0])
    w = np.asarray(gfem.SheetField.interpolate(rho).fill)
    np.testing.assert_allclose(w, [0.0, (1e5 - 1) / (1e10 - 1), 1.0], rtol=1e-12)
    w = np.asarray(gfem.SheetField.interpolate(rho, contrast=100.0).fill)
    np.testing.assert_allclose(w, [0.0, 9 / 99, 1.0], rtol=1e-12)
    linear = gfem.SheetField.interpolate(rho, scheme="linear", surface="s")
    np.testing.assert_array_equal(linear.fill, rho)
    assert linear.surface == "s"
    with pytest.raises(ValueError, match="scheme"):
        gfem.SheetField.interpolate(rho, scheme="simp")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="contrast"):
        gfem.SheetField.interpolate(rho, contrast=1.0)


def test_interpolate_gradients_pass_check_grads():
    rho = np.linspace(0.1, 0.9, 5)
    for scheme in ("exponential", "linear"):
        check_grads(
            lambda r, s=scheme: gfem.SheetField.interpolate(r, scheme=s).fill,
            (rho,),
            order=1,
        )


def test_sheet_field_is_a_pytree_leaf_of_the_model(mesh):
    fill = gfem.SheetField(fill=np.full(_faces(mesh), 0.5))
    model = _model(mesh, gfem.Conductor("sheet", 1e5, 1e-6, fill=fill))
    leaves = jax.tree_util.tree_flatten_with_path(model)[0]
    assert ".boundaries[2].fill.fill" in [jax.tree_util.keystr(p) for p, _ in leaves]


# --- model and problem checks ------------------------------------------------------


def test_model_checks_the_fill_length_and_surface(mesh):
    n = _faces(mesh)
    short = gfem.SheetField(fill=np.ones(3))
    with pytest.raises(ValueError, match=f"{n} faces"):
        _model(mesh, gfem.Conductor("sheet", 1e5, fill=short))
    other = gfem.SheetField(fill=np.ones(n), surface="elsewhere")
    with pytest.raises(ValueError, match="built for surface 'elsewhere'"):
        _model(mesh, gfem.Conductor("sheet", 1e5, fill=other))


def test_sheet_field_needs_an_interior_surface(mesh):
    side = gfem.SheetField(fill=np.ones(len(mesh.elements("side"))))
    with pytest.raises(ValueError, match="interior"):
        _model(mesh, gfem.Impedance("side", resistance=10.0, fill=side))
    # with the lower half excluded, the sheet is the shell of an excluded volume
    fill = gfem.SheetField(fill=np.ones(_faces(mesh)))
    with pytest.raises(ValueError, match="interior"):
        gfem.Model(
            mesh,
            materials={"upper": gfem.Material()},
            boundaries=[gfem.Conductor("sheet", 1e5, fill=fill)],
            excluded=("lower",),
        )


def test_negative_fill_raises_in_the_forward(mesh):
    def s(values):
        fill = gfem.SheetField(fill=values)
        model = _model(mesh, gfem.Impedance("sheet", resistance=40.0, fill=fill))
        return gfem.solve(model, gfem.Driven(frequency=F)).s.values

    bad = -np.ones(_faces(mesh))
    with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match=">= 0"):
        jax.jit(s)(bad)


def test_electrostatic_raises_on_a_sheet_field(mesh):
    fill = gfem.SheetField(fill=np.ones(_faces(mesh)))
    model = _model(mesh, gfem.Conductor("sheet", 1e5, fill=fill))
    problem = gfem.Electrostatic(terminals={"a": "plate1"}, ground="plate2")
    with pytest.raises(NotImplementedError, match="sheet field"):
        gfem.solve(model, problem)


# --- wave ports: the port mode reads the sheet on the port's edges --------------------


def _wave_model(mesh, sheet):
    return gfem.Model(
        mesh,
        materials={"lower": gfem.Material(eps_r=2.0), "upper": gfem.Material()},
        boundaries=[gfem.PEC("plate1"), gfem.PEC("plate2"), sheet],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def test_port_mode_reads_the_fill_on_the_port_edges(mesh):
    fill = gfem.SheetField(fill=np.full(_faces(mesh), 0.25))
    problem = gfem.PortMode("p1", F)
    filled = gfem.solve(
        _wave_model(mesh, gfem.Impedance("sheet", resistance=400.0, fill=fill)), problem
    )
    solid = gfem.solve(
        _wave_model(mesh, gfem.Impedance("sheet", resistance=1600.0)), problem
    )
    bare = gfem.solve(
        _wave_model(mesh, gfem.Impedance("sheet", resistance=400.0)), problem
    )
    np.testing.assert_allclose(filled.gamma, solid.gamma, rtol=1e-10)
    assert abs(complex(filled.gamma[0]) - complex(bare.gamma[0])) > 1e-3 * abs(
        complex(bare.gamma[0])
    )


def test_fill_beside_a_wave_port_is_not_differentiable(mesh):
    def objective(values):
        sheet = gfem.Impedance(
            "sheet", resistance=400.0, fill=gfem.SheetField(fill=values)
        )
        s = gfem.solve(_wave_model(mesh, sheet), gfem.Driven(frequency=F)).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    fill = jnp.full(_faces(mesh), 0.5)
    assert np.isfinite(float(objective(fill)))  # the forward runs
    with pytest.raises(gfem.PortGeometryError, match=r"boundaries\[2\].*stop_gradient"):
        jax.make_jaxpr(jax.grad(objective))(fill)
