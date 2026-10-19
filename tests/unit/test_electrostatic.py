"""gfem.solve on an Electrostatic problem: inputs, ground rules, result."""

from pathlib import Path

import jax
import numpy as np
import pytest
import structured_meshes as sm

import gfem

DATA = Path(__file__).parent / "data"
LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}
STACK = {"top": "top", "mid": "oxide__nitride"}


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))
    return gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)


def _c(model, problem, **options):
    return gfem.solve(model, problem, options=gfem.SolverOptions(**options))


# --- the problem value -----------------------------------------------------------


def test_terminals_keep_their_order_and_surfaces():
    p = gfem.Electrostatic(terminals={"b": ["s1", "s2"], "a": "s3"}, ground="g")
    assert p.terminal_names == ("b", "a")
    assert p.terminals == (("b", ("s1", "s2")), ("a", ("s3",)))
    assert p.ground == ("g",)


def test_a_list_of_surfaces_makes_one_terminal_per_surface():
    p = gfem.Electrostatic(terminals=["trace", "jj"])
    assert p.terminals == (("trace", ("trace",)), ("jj", ("jj",)))
    assert p.ground == ()


def test_problem_is_frozen_hashable_and_static_under_jax():
    p = gfem.Electrostatic(terminals={"a": "s"}, ground=["g"])
    assert p == gfem.Electrostatic(terminals={"a": ("s",)}, ground="g")
    assert hash(p) == hash(gfem.Electrostatic(terminals={"a": "s"}, ground="g"))
    assert jax.tree.leaves(p) == []
    with pytest.raises(AttributeError):
        p.ground = ()  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"terminals": {}}, "at least one terminal"),
        ({"terminals": {"a": "s", "b": ["s"]}}, "terminals 'a' and 'b'"),
        ({"terminals": {"a": "s"}, "ground": "s"}, "both in terminal 'a'"),
        ({"terminals": {"a": ["s", "s"]}}, "twice"),
    ],
)
def test_inconsistent_terminals_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        gfem.Electrostatic(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"terminals": {"a": 3}}, {"terminals": {1: "s"}}, {"terminals": "s", "ground": 2}],
)
def test_non_names_raise_type_error(kwargs):
    with pytest.raises(TypeError):
        gfem.Electrostatic(**kwargs)


# --- solving -----------------------------------------------------------------------


def test_result_is_labelled_symmetric_with_palace_signs(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    c = np.asarray(cap.capacitance)
    assert cap.terminals == ("top", "mid")
    assert cap.index("mid") == 1
    np.testing.assert_array_equal(c, c.T)
    assert (np.diag(c) > 0).all()
    assert c[0, 1] < 0
    assert "fF" in repr(cap)
    assert "mid" in repr(cap)


def test_result_is_a_pytree_with_the_matrix_as_its_leaf(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    leaves, tree = jax.tree.flatten(cap)
    assert len(leaves) == 1
    assert leaves[0] is cap.capacitance
    assert isinstance(cap.capacitance, jax.Array)
    assert cap.capacitance.dtype == np.float64
    back = jax.tree.unflatten(tree, [leaves[0] * 2])
    assert back.terminals == cap.terminals


def test_unknown_terminal_index_names_the_terminals(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    with pytest.raises(KeyError, match="'top', 'mid'"):
        cap.index("bottom")


def test_without_ground_each_row_sums_to_zero(stack):
    cap = _c(stack, gfem.Electrostatic(terminals={**STACK, "low": "bottom"}))
    c = np.asarray(cap.capacitance)
    np.testing.assert_allclose(c.sum(axis=1), 0, atol=1e-12 * np.abs(c).max())


@pytest.mark.parametrize("condition", [gfem.PEC, gfem.Conductor])
def test_model_conductors_that_are_not_terminals_are_grounded(stack, condition):
    boundary = (
        gfem.PEC("bottom")
        if condition is gfem.PEC
        else gfem.Conductor("bottom", conductivity=5.8e7, thickness=1e-6)
    )
    grounded = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    with_pec = _c(
        stack.replace(boundaries=[boundary]), gfem.Electrostatic(terminals=STACK)
    )
    np.testing.assert_allclose(with_pec.capacitance, grounded.capacitance, rtol=1e-12)


def test_a_terminal_overrides_the_model_pec_on_its_surface(stack):
    model = stack.replace(boundaries=[gfem.PEC("top"), gfem.PEC("bottom")])
    plain = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    cap = _c(model, gfem.Electrostatic(terminals=STACK))
    np.testing.assert_allclose(cap.capacitance, plain.capacitance, rtol=1e-12)


def test_other_boundaries_carry_no_charge(stack):
    model = stack.replace(boundaries=[gfem.Absorbing("side"), gfem.PMC("top")])
    plain = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    cap = _c(model, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    np.testing.assert_allclose(cap.capacitance, plain.capacitance, rtol=1e-12)


def test_conductivity_and_loss_do_not_change_the_capacitance(stack):
    lossy = {
        "oxide": gfem.Material(eps_r=3.9, tan_delta=0.1, conductivity=1.0, mu_r=4.0),
        "nitride": gfem.Material(eps_r=7.5, conductivity=1e3),
    }
    problem = gfem.Electrostatic(terminals=STACK, ground="bottom")
    plain = _c(stack, problem)
    cap = _c(stack.replace(materials=lossy), problem)
    np.testing.assert_allclose(cap.capacitance, plain.capacitance, rtol=1e-13)


def test_solving_twice_gives_the_same_matrix(stack):
    problem = gfem.Electrostatic(terminals=STACK, ground="bottom")
    first = _c(stack, problem, order=2)
    _c(stack, problem, order=1)
    second = _c(stack, problem, order=2)
    np.testing.assert_array_equal(first.capacitance, second.capacitance)


def test_moved_nodes_change_the_capacitance(stack):
    problem = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
    nodes = stack.mesh.nodes
    z = nodes[:, 2]
    stretched = stack.replace(
        mesh=jax.tree.unflatten(
            jax.tree.structure(stack.mesh), [nodes.at[:, 2].set(2 * z)]
        )
    )
    c = float(_c(stack, problem).capacitance[0, 0])
    c2 = float(_c(stretched, problem).capacitance[0, 0])
    assert c2 == pytest.approx(c / 2, rel=1e-11)


# --- errors ------------------------------------------------------------------------


def test_an_inverted_moved_mesh_raises_before_solving(stack):
    mirrored = jax.tree.unflatten(
        jax.tree.structure(stack.mesh), [stack.mesh.nodes.at[:, 2].multiply(-1.0)]
    )
    with pytest.raises(gfem.MeshQualityError):
        _c(stack.replace(mesh=mirrored), gfem.Electrostatic(terminals=STACK))


def test_unknown_surface_raises_with_a_hint(stack):
    with pytest.raises(gfem.RegionNotFoundError, match="'top'"):
        _c(stack, gfem.Electrostatic(terminals={"t": "tpo"}, ground="bottom"))
    with pytest.raises(gfem.RegionNotFoundError):
        _c(stack, gfem.Electrostatic(terminals={"t": "top"}, ground="bottm"))


def test_a_volume_terminal_asks_for_its_surfaces(stack):
    with pytest.raises(ValueError, match="volume"):
        _c(stack, gfem.Electrostatic(terminals={"t": "oxide"}))


def test_touching_terminals_raise(stack):
    # "side" shares the plate edges with "top" and "bottom"
    with pytest.raises(ValueError, match="touch"):
        _c(stack, gfem.Electrostatic(terminals={"t": "top", "s": "side"}))
    with pytest.raises(ValueError, match="touch"):
        _c(stack, gfem.Electrostatic(terminals={"t": "top"}, ground="side"))


def test_a_terminal_inside_an_excluded_volume_raises():
    mesh = gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        excluded=("sub",),
    )
    with pytest.raises(ValueError, match="does not border"):
        _c(model, gfem.Electrostatic(terminals={"m": "metal_xy"}))


def test_terminal_on_an_excluded_conductor_shell():
    mesh = gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")
    materials = {"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()}
    problem = gfem.Electrostatic(
        terminals={"metal": ["metal_xy", "metal_z"]}, ground="sub__None"
    )
    meshed = _c(gfem.Model(mesh, materials=materials), problem)
    excluded = gfem.Model(
        mesh,
        materials=materials,
        excluded=(gfem.enclosed_by("metal_xy", "metal_z"),),
    )
    # the conductor's inside is at one potential: excluding it changes nothing
    cap = _c(excluded, problem)
    np.testing.assert_allclose(cap.capacitance, meshed.capacitance, rtol=1e-10)
    assert float(cap.capacitance[0, 0]) > 0


@pytest.mark.parametrize("solver", ["bddc_gmres"])
def test_unavailable_linear_solver_raises(stack, solver):
    with pytest.raises(NotImplementedError, match="direct"):
        _c(stack, gfem.Electrostatic(terminals=STACK), linear_solver=solver)


def test_gpu_device_raises_not_available(stack):
    with pytest.raises(NotImplementedError, match="device"):
        _c(
            stack,
            gfem.Electrostatic(terminals=STACK),
            linear_solver="direct",
            device="cuda",
        )


def test_solve_rejects_wrong_argument_types(stack):
    with pytest.raises(TypeError, match="problem type"):
        gfem.solve(stack, "electrostatic")
    with pytest.raises(TypeError, match=r"gfem\.Model"):
        gfem.solve(stack.mesh, gfem.Electrostatic(terminals=STACK))
    with pytest.raises(TypeError, match="SolverOptions"):
        gfem.solve(stack, gfem.Electrostatic(terminals=STACK), options={"order": 2})
