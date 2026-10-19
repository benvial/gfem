"""The raise table: every result leaf x every model / problem leaf.

Each (problem, result leaf, input leaf) cell is one of three outcomes:
a finite, nonzero gradient; an exact zero, because the problem's physics
never reads that input (or the leaf is a constant of it, like a port's
reference impedance); or `gfem.NonDifferentiableError` at trace time.
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
# Terminal potentials: their coefficients differentiate like the capacitance;
# their mesh is the input mesh passed through (node gradients raise in solve).
EXPECTED |= {
    ("electrostatic", f"potential.{t}.{leaf}", kind): (
        EXPECTED[("electrostatic", "capacitance", kind)]
        if leaf == "coefficients" or kind == ("nodes",)
        else ZERO
    )
    for t in ("top",)
    for leaf in ("coefficients", "mesh.nodes")
    for kind in [("nodes",)] + [("material", "oxide", p) for p in ALL_PROPERTIES]
}
# Region energies are not functions of the capacitance (they would need one
# adjoint solve per terminal): they raise for every input the solve reads,
# and are exact zeros for the inputs it never reads. (`mutual` and `inverse`
# are jnp functions of `capacitance`, not leaves: they differentiate.)
EXPECTED |= {
    ("electrostatic", f"energy.{region}", kind): (
        ZERO if kind[0] == "material" and kind[2] != "eps_r" else RAISES
    )
    for region in ("oxide", "nitride")
    for kind in [("nodes",)] + [("material", "oxide", p) for p in ALL_PROPERTIES]
}
# The port mode problem is not differentiable: every result leaf raises.
PORT_MODE_OUTPUTS = ("frequency", "gamma", "Z_PV", "residual", "solved")
EXPECTED |= {
    ("port_mode", out, kind): RAISES
    for out in PORT_MODE_OUTPUTS
    for kind in [("nodes",)] + [("material", "fill", p) for p in ALL_PROPERTIES]
}

# Driven: S and the kept field coefficients differentiate with respect to
# every electromagnetic property and are exact zeros for the thermal ones.
# z_ref (the port resistances), the kept fields' mesh (the input mesh passed
# through) and their angular frequency are constants of the materials. Node
# coordinates raise for every leaf (no shape gradients yet).
EM_PROPERTIES = ("eps_r", "mu_r", "tan_delta", "conductivity")
_DRIVEN_DEPENDENT = (
    "s.values",
    *(f"e_fields.{k}.coefficients" for k in (0, 1)),
)
EXPECTED |= {
    ("driven", out, kind): (
        RAISES
        if kind == ("nodes",)
        else GRADIENT
        if out in _DRIVEN_DEPENDENT and kind[2] in EM_PROPERTIES
        else ZERO
    )
    for out in (
        "s.values",
        "s.z_ref",
        # kept fields (one per excited port)
        *(
            f"e_fields.{k}.{leaf}"
            for k in (0, 1)
            for leaf in ("coefficients", "mesh.nodes", "omega")
        ),
    )
    for kind in [("nodes",)] + [("material", "dielectric", p) for p in ALL_PROPERTIES]
}

# Eigenmode results are not differentiable yet (eigenvalue gradients come
# later): like driven, every leaf raises for the inputs the forms read.
EXPECTED |= {
    ("eigenmode", out, kind): (
        ZERO if kind[0] == "material" and kind[2] not in EM_PROPERTIES else RAISES
    )
    for out in (
        "omega",
        "error",
        "q_port.p1",
        "q_port.p2",
        "bulk_participation.dielectric",
        *(f"e_fields.0.{leaf}" for leaf in ("coefficients", "mesh.nodes", "omega")),
    )
    for kind in [("nodes",)] + [("material", "dielectric", p) for p in ALL_PROPERTIES]
}


def _field(mesh, region, **overrides):
    # values vary per element, so the terminal potentials depend on them
    n = len(mesh.elements(region))
    ramp = np.linspace(1.0, 1.5, n)
    values = ALL_PROPERTIES | overrides
    return gfem.MaterialField(**{k: v * ramp for k, v in values.items()})


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
def driven_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (2, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": _field(mesh, "dielectric")},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.fixture(scope="module")
def eigen_model(tmp_path_factory):
    # weakly damped (resistive ports, little conduction): the modes are sharp
    path = tmp_path_factory.mktemp("resonator") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": _field(mesh, "dielectric", conductivity=1e-3)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=1000.0),
            gfem.LumpedPort("p2", "port2", "+z", impedance=1000.0),
        ],
    )


@pytest.fixture(scope="module")
def cases(model, driven_model, eigen_model):
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
        "driven": (driven_model, gfem.Driven(frequency=5e9, keep_fields=True)),
        "eigenmode": (eigen_model, gfem.Eigenmode(target=17e9)),
    }


PROBLEMS = {
    "electrostatic": gfem.Electrostatic(terminals={"top": "top"}, ground="bottom"),
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


def test_energy_error_names_the_output_and_the_opt_out(model):
    problem = PROBLEMS["electrostatic"]
    n = len(model.mesh.elements("oxide"))

    def objective(eps, *, stop):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=eps))
        result = gfem.solve(m, problem)
        energy = result.energy["oxide"]
        energy = jax.lax.stop_gradient(energy) if stop else energy
        return result.capacitance[0, 0] + energy.sum()

    eps = jnp.full(n, 3.9)
    with pytest.raises(
        gfem.NonDifferentiableError, match=r"result\.energy.*stop_gradient"
    ):
        jax.make_jaxpr(jax.grad(lambda e: objective(e, stop=False)))(eps)
    assert np.abs(np.asarray(jax.grad(lambda e: objective(e, stop=True))(eps))).max()


def test_stop_gradient_opts_a_read_input_out(model):
    problem = PROBLEMS["electrostatic"]

    def objective(m):
        frozen = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return gfem.solve(frozen, problem).capacitance[0, 0]  # in series with oxide

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
        return out["capacitance"][0, 0] + extra

    eps = jnp.full(n, 3.9)
    with pytest.raises(gfem.NonDifferentiableError, match="'extra'"):
        jax.make_jaxpr(jax.grad(lambda e: objective(e, stop=False)))(eps)
    grad = jax.grad(lambda e: objective(e, stop=True))(eps)

    def c_top(e):
        m = model.with_material("oxide", gfem.MaterialField(eps_r=e))
        return gfem.solve(m, problem).capacitance[0, 0]

    assert np.abs(np.asarray(grad)).max() > 0
    np.testing.assert_allclose(grad, jax.grad(c_top)(eps), rtol=1e-14)
