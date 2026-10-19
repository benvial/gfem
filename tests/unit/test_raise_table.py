"""The raise table: every result leaf x every model / problem leaf.

Each (problem, result leaf, input leaf) cell is one of three outcomes:
a finite, nonzero gradient; an exact zero, because the problem's physics
never reads that input; or `gfem.NonDifferentiableError` at trace time.
Never a silent zero for an input the physics reads. The cells are
generated from the pytrees, so a new leaf without an expected outcome
fails here.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._adjoint import solve_vjp
from gfem._adjoint.solve_vjp import input_kind
from gfem._engine.electrostatic import ElectrostaticRule

DATA = Path(__file__).parent / "data"
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
# Region energies are not differentiable in this version: a read input
# raises; an input the electrostatic physics never reads is still a zero.
EXPECTED |= {
    ("electrostatic", f"energy.{region}", kind): (
        ZERO if outcome == ZERO else RAISES
    )
    for region in ("oxide", "nitride")
    for (_, out, kind), outcome in list(EXPECTED.items())
    if out == "capacitance"
}
# The port mode problem is not differentiable: every result leaf raises.
PORT_MODE_OUTPUTS = ("frequency", "gamma", "Z_PV", "residual")
EXPECTED |= {
    ("port_mode", out, kind): RAISES
    for out in PORT_MODE_OUTPUTS
    for kind in [("nodes",)] + [("material", "fill", p) for p in ALL_PROPERTIES]
}


def _field(mesh, region):
    n = len(mesh.elements(region))
    return gfem.MaterialField(**{k: np.full(n, v) for k, v in ALL_PROPERTIES.items()})


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    layers = (("oxide", 0.5), ("nitride", 0.25))
    sm.write_msh22(path, sm.plate(4.0, 3.0, layers, (1, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path)
    return gfem.Model(
        mesh,
        materials={"oxide": _field(mesh, "oxide"), "nitride": gfem.Material(eps_r=7.5)},
        boundaries=[gfem.PEC("bottom")],
    )


@pytest.fixture(scope="module")
def cases(model):
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    port_model = gfem.Model(
        mesh,
        materials={"fill": _field(mesh, "fill")},
        boundaries=[gfem.PEC("plates")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    return {
        "electrostatic": (model, PROBLEMS["electrostatic"]),
        "port_mode": (port_model, gfem.PortMode("p1", 10e9)),
    }


PROBLEMS = {
    "electrostatic": gfem.Electrostatic(
        terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
    ),
}


def _cells(cases):
    for problem_name, (model, problem) in cases.items():
        result = gfem.solve(model, problem)
        model_leaves = jax.tree_util.tree_flatten_with_path(model)[0]
        kinds = [input_kind(path, "model") for path, _ in model_leaves] + [
            input_kind(path, "problem")
            for path, _ in jax.tree_util.tree_flatten_with_path(problem)[0]
        ]
        for out_path, _ in jax.tree_util.tree_flatten_with_path(result)[0]:
            out_name = jax.tree_util.keystr(out_path, simple=True, separator=".")
            for index, kind in enumerate(kinds):
                yield problem_name, model, problem, out_name, out_path, index, kind


def _objective(model, problem, out_path, index):
    """The weighted sum of one result leaf as a function of one input leaf."""
    leaves, treedef = jax.tree.flatten((model, problem))

    def objective(x):
        moved, moved_problem = jax.tree.unflatten(
            treedef, [*leaves[:index], x, *leaves[index + 1 :]]
        )
        result = gfem.solve(moved, moved_problem)
        (value,) = [
            v
            for p, v in jax.tree_util.tree_flatten_with_path(result)[0]
            if p == out_path
        ]
        weights = np.random.default_rng(1).uniform(0.5, 1.5, value.shape)
        return jnp.sum(jnp.real(value) * weights)

    return objective, leaves[index]


def test_every_cell_has_an_expected_outcome(cases):
    cells = {(p, o, k) for p, _, _, o, _, _, k in _cells(cases)}
    assert cells == set(EXPECTED)


def test_raise_table(cases):
    for name, model, problem, out_name, out_path, index, kind in _cells(cases):
        expected = EXPECTED[(name, out_name, kind)]
        objective, x = _objective(model, problem, out_path, index)
        cell = f"{name} {out_name} x {kind}"
        if expected == RAISES:
            # raised while tracing, before any host work
            with pytest.raises(gfem.NonDifferentiableError):
                jax.make_jaxpr(jax.grad(objective))(x)
            continue
        grad = np.asarray(jax.grad(objective)(x))
        assert np.isfinite(grad).all(), cell
        if expected == ZERO:
            assert not grad.any(), cell
        else:
            assert np.abs(grad).max() > 0, cell


def test_unsupported_input_error_names_the_input_and_the_opt_out(model):
    objective, x = _objective(
        model, PROBLEMS["electrostatic"], (jax.tree_util.GetAttrKey("capacitance"),), 0
    )
    with pytest.raises(
        gfem.NonDifferentiableError, match=r"mesh\.nodes.*stop_gradient"
    ):
        jax.make_jaxpr(jax.grad(objective))(x)


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
