"""The raise table: every result leaf x every model / problem leaf.

Each (problem, result leaf, input leaf) cell is one of three outcomes:
a finite, nonzero gradient; an exact zero, because the problem's physics
never reads that input; or `gfem.NonDifferentiableError` at trace time.
Never a silent zero for an input the physics reads. The cells are
generated from the pytrees, so a new leaf without an expected outcome
fails here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._adjoint import solve_vjp
from gfem._adjoint.solve_vjp import input_kind
from gfem._engine.electrostatic import ElectrostaticRule

GRADIENT, ZERO, RAISES = "gradient", "zero", "raises"

ALL_PROPERTIES = {
    "eps_r": 3.9 * (1 - 0.01j),
    "mu_r": 1.0,
    "tan_delta": 0.01,
    "conductivity": 2.0,
    "thermal_conductivity": 1.4,
    "density": 2200.0,
    "specific_heat": 700.0,
}

# (problem name, result leaf, input kind) -> outcome. Input kinds name the
# model leaf: ("nodes",) or ("material", <region>, <property>).
EXPECTED = {
    ("electrostatic", "capacitance", ("nodes",)): RAISES,
    ("electrostatic", "capacitance", ("material", "oxide", "eps_r")): GRADIENT,
    ("electrostatic", "capacitance", ("material", "oxide", "mu_r")): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "tan_delta")): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "conductivity")): ZERO,
    (
        "electrostatic",
        "capacitance",
        ("material", "oxide", "thermal_conductivity"),
    ): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "density")): ZERO,
    ("electrostatic", "capacitance", ("material", "oxide", "specific_heat")): ZERO,
}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    layers = (("oxide", 0.5), ("nitride", 0.25))
    sm.write_msh22(path, sm.plate(4.0, 3.0, layers, (1, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path)
    n = len(mesh.elements("oxide"))
    field = gfem.MaterialField(**{k: np.full(n, v) for k, v in ALL_PROPERTIES.items()})
    return gfem.Model(
        mesh,
        materials={"oxide": field, "nitride": gfem.Material(eps_r=7.5)},
        boundaries=[gfem.PEC("bottom")],
    )


PROBLEMS = {
    "electrostatic": gfem.Electrostatic(
        terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
    ),
}


def _cells(model):
    for problem_name, problem in PROBLEMS.items():
        result = gfem.solve(model, problem)
        outputs = jax.tree_util.tree_flatten_with_path(result)[0]
        inputs = [
            (i, input_kind(path, "model"))
            for i, (path, _) in enumerate(
                jax.tree_util.tree_flatten_with_path(model)[0]
            )
        ] + [
            (None, input_kind(path, "problem"))
            for path, _ in jax.tree_util.tree_flatten_with_path(problem)[0]
        ]
        for out_path, _ in outputs:
            out_name = jax.tree_util.keystr(out_path, simple=True, separator=".")
            for index, kind in inputs:
                yield problem_name, problem, out_name, out_path, index, kind


def _objective(model, problem, out_path, index):
    leaves, treedef = jax.tree.flatten(model)

    def objective(x):
        moved = jax.tree.unflatten(treedef, [*leaves[:index], x, *leaves[index + 1 :]])
        result = gfem.solve(moved, problem)
        (value,) = [
            v
            for p, v in jax.tree_util.tree_flatten_with_path(result)[0]
            if p == out_path
        ]
        weights = np.random.default_rng(1).uniform(0.5, 1.5, value.shape)
        return jnp.sum(jnp.real(value) * weights)

    return objective, leaves[index]


def test_every_cell_has_an_expected_outcome(model):
    cells = {(p, o, k) for p, _, o, _, _, k in _cells(model)}
    assert cells == set(EXPECTED)


def test_raise_table(model):
    for problem_name, problem, out_name, out_path, index, kind in _cells(model):
        expected = EXPECTED[(problem_name, out_name, kind)]
        objective, x = _objective(model, problem, out_path, index)
        cell = f"{problem_name} {out_name} x {kind}"
        if expected == RAISES:
            # raised while tracing, before any host work
            with pytest.raises(gfem.NonDifferentiableError, match="stop_gradient"):
                jax.make_jaxpr(jax.grad(objective))(x)
            continue
        grad = np.asarray(jax.grad(objective)(x))
        assert np.isfinite(grad).all(), cell
        if expected == ZERO:
            assert not grad.any(), cell
        else:
            assert np.abs(grad).max() > 0, cell


def test_stop_gradient_opts_a_read_input_out(model):
    problem = PROBLEMS["electrostatic"]

    def objective(m):
        frozen = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return gfem.solve(frozen, problem).capacitance[1, 1]  # mid sees the oxide

    grads = jax.grad(objective)(model)
    assert not np.asarray(grads.mesh.nodes).any()
    assert np.abs(np.asarray(grads.materials["oxide"].eps_r)).max() > 0


class _RuleWithExtraOutput(ElectrostaticRule):
    """The electrostatic rule plus an output its backward cannot handle."""

    def output_specs(self, model):
        return {**super().output_specs(model), "extra": jax.ShapeDtypeStruct((), float)}

    def forward(self, model, *, keep):
        outputs, entry = super().forward(model, keep=keep)
        return {**outputs, "extra": np.trace(outputs["capacitance"])}, entry


def test_nondifferentiable_output_raises_unless_stop_gradient(model):
    problem = PROBLEMS["electrostatic"]
    rule = _RuleWithExtraOutput(model, problem, gfem.SolverOptions())
    n = len(model.mesh.elements("oxide"))

    def objective(eps, *, stop):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=eps))
        out = solve_vjp(rule, m, problem)
        extra = jax.lax.stop_gradient(out["extra"]) if stop else out["extra"]
        return out["capacitance"][1, 1] + extra

    eps = jnp.full(n, 3.9)
    with pytest.raises(gfem.NonDifferentiableError, match="'extra'"):
        jax.make_jaxpr(jax.grad(lambda e: objective(e, stop=False)))(eps)
    grad = jax.grad(lambda e: objective(e, stop=True))(eps)

    def c_mid(e):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=e))
        return gfem.solve(m, problem).capacitance[1, 1]

    assert np.abs(np.asarray(grad)).max() > 0
    np.testing.assert_allclose(grad, jax.grad(c_mid)(eps), rtol=1e-14)
