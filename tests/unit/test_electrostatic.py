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


def test_result_is_a_pytree_of_the_matrix_and_the_region_energies(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    leaves, tree = jax.tree.flatten(cap)
    assert len(leaves) == 3
    assert leaves[0] is cap.capacitance
    assert isinstance(cap.capacitance, jax.Array)
    assert cap.capacitance.dtype == np.float64
    assert all(isinstance(e, jax.Array) for e in cap.energy.values())
    back = jax.tree.unflatten(tree, [leaf * 2 for leaf in leaves])
    assert back.terminals == cap.terminals
    np.testing.assert_allclose(back.mutual, 2 * cap.mutual)


def test_mutual_and_inverse_follow_the_maxwell_matrix(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    c = np.asarray(cap.capacitance)
    m = np.asarray(cap.mutual)
    # Palace terminal-Cm.csv: -C off the diagonal, the row sums on it
    np.testing.assert_allclose(m - np.diag(np.diag(m)), -(c - np.diag(np.diag(c))))
    np.testing.assert_allclose(np.diag(m), c.sum(axis=1))
    np.testing.assert_allclose(np.asarray(cap.inverse) @ c, np.eye(2), atol=1e-12)


def test_mutual_and_inverse_are_differentiable_functions_of_the_matrix(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))

    def to_ground(c):
        leaves = jax.tree.leaves(cap)
        res = jax.tree.unflatten(jax.tree.structure(cap), [c, *leaves[1:]])
        return res.mutual[1, 1] + res.inverse[0, 0]

    g = jax.grad(to_ground)(cap.capacitance)
    assert g.shape == (2, 2)
    assert np.isfinite(np.asarray(g)).all()


def test_region_energies_add_up_to_half_the_self_capacitance(stack):
    cap = _c(stack, gfem.Electrostatic(terminals=STACK, ground="bottom"))
    assert list(cap.energy) == ["oxide", "nitride"]
    total = sum(np.asarray(e) for e in cap.energy.values())
    np.testing.assert_allclose(total, np.diag(cap.capacitance) / 2, rtol=1e-12)
    assert all(np.asarray(e).shape == (2,) for e in cap.energy.values())


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
    # the field-free inside of the conductor stores no energy
    assert set(cap.energy) == set(meshed.energy)
    np.testing.assert_allclose(
        sum(cap.energy.values()), sum(meshed.energy.values()), rtol=1e-10
    )


@pytest.mark.parametrize("solver", ["auto", "bddc_gmres"])
def test_bddc_default_agrees_with_the_direct_solver(tmp_path, solver):
    # order 3: face DOFs are off the wirebasket, so BDDC + CG really iterates
    path = tmp_path / "fine.msh"
    sm.write_msh22(path, sm.plate(4.0, 3.0, LAYERS, (3, 3, 2)))
    model = gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)
    problem = gfem.Electrostatic(terminals=STACK, ground="bottom")
    direct = _c(model, problem, order=3, linear_solver="direct")
    bddc = _c(model, problem, order=3, linear_solver=solver)
    np.testing.assert_allclose(bddc.capacitance, direct.capacitance, rtol=1e-9)


def test_solve_rejects_wrong_argument_types(stack):
    with pytest.raises(TypeError, match="problem type"):
        gfem.solve(stack, "electrostatic")
    with pytest.raises(TypeError, match=r"gfem\.Model"):
        gfem.solve(stack.mesh, gfem.Electrostatic(terminals=STACK))
    with pytest.raises(TypeError, match="SolverOptions"):
        gfem.solve(stack, gfem.Electrostatic(terminals=STACK), options={"order": 2})
