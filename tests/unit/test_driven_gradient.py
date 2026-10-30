"""Driven gradients of every electromagnetic material property.

A small line with second-order absorbing side walls and lossy, magnetic,
conducting P0 fields: each property's gradient (complex ``eps_r`` along
both axes, with a loss tangent that mixes them in the absorbing terms)
against central differences on the direct solver.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches, unit_directions
from solver_misses import adjoint_cannot_converge_at, cannot_converge_at

import gfem
from gfem._engine import driven, sensitivity, terms

F = (21e9,)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 2, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material()},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=2),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _values(line):
    rng = np.random.default_rng(7)
    n = len(line.mesh.tetrahedra)
    return {
        "eps_r": jnp.asarray(rng.uniform(2.0, 5.0, n) * (1 - 0.1j)),
        "mu_r": jnp.asarray(rng.uniform(1.0, 2.0, n)),
        "tan_delta": jnp.asarray(rng.uniform(0.01, 0.2, n)),
        "conductivity": jnp.asarray(rng.uniform(0.1, 1.0, n)),
    }, rng


def _objective(line, values, name, problem=None):
    problem = problem or gfem.Driven(frequency=F)
    options = gfem.SolverOptions(linear_solver="direct")

    def objective(p):
        model = line.with_material(
            "dielectric", gfem.MaterialField(**{**values, name: p})
        )
        s = gfem.solve(model, problem, options=options).s.values
        return jnp.sum(jnp.abs(s) ** 2 * jnp.array([[1.0, 0.5], [2.0, -1.0]]))

    return objective


# fourth-order differences, see tests/finite_differences.py
STEPS = {"eps_r": 1e-2, "mu_r": 1e-2, "tan_delta": 1e-3, "conductivity": 5e-2}


@pytest.mark.parametrize("name", ["eps_r", "mu_r", "tan_delta", "conductivity"])
def test_gradient_of_each_property_matches_central_differences(line, name):
    values, rng = _values(line)
    objective = _objective(line, values, name)
    p = values[name]
    (d,) = unit_directions(rng, p.shape, 1)
    # a complex eps_r is checked along both axes (cotangent dJ/dx - i dJ/dy)
    directions = [d, 1j * d] if jnp.iscomplexobj(p) else [d]
    assert_gradient_matches(objective, p, directions, step=STEPS[name])


def test_gradient_beside_a_per_axis_lossy_crystal(tmp_path):
    # a static anisotropic region (per-axis eps_r and tan_delta) next to the
    # design field: its tensor enters the forward, never the derivative
    path = tmp_path / "layered.msh"
    layers = (("crystal", 0.4), ("field", 0.6))
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 2, 2), layers=layers))
    crystal = gfem.Material(
        eps_r=(9.3, 9.3, 11.5),
        tan_delta=(0.03, 0.03, 0.086),
        axes=((0.8, 0.6, 0), (-0.6, 0.8, 0), (0, 0, 1)),
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"crystal": crystal, "field": gfem.Material()},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.Absorbing("side")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    n = len(model.mesh.elements("field"))
    rng = np.random.default_rng(3)
    options = gfem.SolverOptions(linear_solver="direct")

    def objective(eps):
        field = gfem.MaterialField(eps_r=eps, tan_delta=jnp.full(n, 0.02))
        moved = model.with_material("field", field)
        s = gfem.solve(moved, gfem.Driven(F), options=options).s.values
        return jnp.sum(jnp.abs(s) ** 2 * jnp.array([[1.0, 0.5], [2.0, -1.0]]))

    p = jnp.asarray(rng.uniform(2.0, 5.0, n))
    directions = unit_directions(rng, p.shape, 2)
    assert_gradient_matches(objective, p, directions, step=STEPS["eps_r"])


def test_unexcited_port_rows_use_an_adjoint_solve(line):
    values, _ = _values(line)
    full = jax.grad(_objective(line, values, "eps_r"))(values["eps_r"])

    def s21(problem):
        def objective(p):
            model = line.with_material(
                "dielectric", gfem.MaterialField(**{**values, "eps_r": p})
            )
            s = gfem.solve(
                model, problem, options=gfem.SolverOptions(linear_solver="direct")
            ).s
            return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

        return jax.grad(objective)(values["eps_r"])

    np.testing.assert_allclose(
        s21(gfem.Driven(frequency=F, excite="p1")),
        s21(gfem.Driven(frequency=F)),
        rtol=1e-9,
    )
    assert np.abs(np.asarray(full)).max() > 0


def test_a_gradient_through_a_solve_that_does_not_converge_raises(line, monkeypatch):
    # the adjoint needs every frequency: JAX keeps the message, not the class
    values, _ = _values(line)
    problem = gfem.Driven(frequency=(20e9, 21e9))
    options = gfem.SolverOptions(linear_solver="bddc_gmres")

    def objective(p):
        model = line.with_material(
            "dielectric", gfem.MaterialField(**{**values, "eps_r": p})
        )
        return jnp.sum(jnp.abs(gfem.solve(model, problem, options=options).s.values))

    cannot_converge_at(monkeypatch, (21e9,))
    with pytest.raises(
        jax.errors.JaxRuntimeError, match=r"At 2\.1e\+10 Hz.*did not converge"
    ):
        jax.grad(objective)(values["eps_r"])


#: Excites p1 only: the S21 cotangent takes an adjoint solve (no reciprocity)
ADJOINT = gfem.Driven(frequency=(20e9, 21e9), excite="p1")


def test_an_adjoint_solve_that_does_not_converge_names_its_frequency(line, monkeypatch):
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    rule = driven.DrivenRule(line, ADJOINT, options)
    _, entry = rule.forward(line, keep=True)
    ct = np.zeros((2, 2, 1), complex)
    ct[:, 1, 0] = 1.0  # S21 at both frequencies
    adjoint_cannot_converge_at(monkeypatch, (21e9,))
    with pytest.raises(
        gfem.LinearSolverConvergenceError, match=r"At 2\.1e\+10 Hz, the adjoint"
    ) as err:
        rule.backward(line, entry, {"s": ct})
    assert err.value.frequency == 21e9
    assert err.value.unsolved is None
    assert err.value.result is None
    assert "did not converge" in str(err.value)


def test_a_gradient_whose_adjoint_solve_does_not_converge_raises(line, monkeypatch):
    values, _ = _values(line)
    options = gfem.SolverOptions(linear_solver="bddc_gmres")

    def objective(p):
        model = line.with_material(
            "dielectric", gfem.MaterialField(**{**values, "eps_r": p})
        )
        s = gfem.solve(model, ADJOINT, options=options).s
        return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

    adjoint_cannot_converge_at(monkeypatch, (21e9,))
    with pytest.raises(
        jax.errors.JaxRuntimeError, match=r"At 2\.1e\+10 Hz, the adjoint"
    ):
        jax.grad(objective)(values["eps_r"])


def test_pair_weights_are_the_thetas_of_the_operator_pieces(line):
    op = driven.driven_operator(
        line.with_material("dielectric", gfem.Material(eps_r=2.0, conductivity=1.0)),
        order=1,
    )
    omega = 2 * math.pi * 13e9
    weights = op.weights(omega)
    thetas = {p.name: complex(p.theta(omega)) for p in op.pieces}
    # every material term the kernel reads, with its theta; no port piece
    k0 = op.wavenumber(omega)
    expected = {
        "curl_curl": 1.0,
        "mass": -(k0**2),
        "conductivity": 1j * k0,
        "absorbing": 1j * k0,
        "absorbing order 2": 1 / (2j * k0),
    }
    assert set(expected) <= set(sensitivity.TERMS)
    for name, theta in expected.items():
        assert weights[name] == pytest.approx(theta, rel=1e-15)
    assert set(weights) == {n for n in thetas if not n.startswith("port ")}


def test_excitation_is_the_voltage_vector_times_the_reciprocity_factor(line):
    op = driven.driven_operator(line, order=2)
    for data in op.ports:
        b, voltage = terms.port_vectors(op.space, op.built, op.order, data)
        factor = terms.excitation_per_voltage(data.port.rlc[0])
        scale = np.abs(b).max()
        np.testing.assert_allclose(b, factor * voltage, rtol=1e-13, atol=1e-13 * scale)
