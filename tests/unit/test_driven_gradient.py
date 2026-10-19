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


@pytest.mark.parametrize("name", ["eps_r", "mu_r", "tan_delta", "conductivity"])
def test_gradient_of_each_property_matches_central_differences(line, name):
    values, rng = _values(line)
    objective = _objective(line, values, name)
    p = values[name]
    grad = jax.grad(objective)(p)
    h = 1e-5
    d = rng.standard_normal(p.shape)
    fd = (objective(p + h * d) - objective(p - h * d)) / (2 * h)
    assert float(jnp.real(grad) @ d) == pytest.approx(float(fd), rel=1e-6)
    if jnp.iscomplexobj(p):  # JAX's cotangent: dJ/dx - i dJ/dy
        fd = (objective(p + 1j * h * d) - objective(p - 1j * h * d)) / (2 * h)
        assert float(-grad.imag @ d) == pytest.approx(float(fd), rel=1e-6)


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


def test_pair_weights_are_the_thetas_of_the_operator_pieces(line):
    op = driven.driven_operator(
        line.with_material("dielectric", gfem.Material(eps_r=2.0, conductivity=1.0)),
        order=1,
    )
    omega = 2 * math.pi * 13e9
    weights = driven._weights(op.wavenumber(omega))
    thetas = {p.name: complex(p.theta(omega)) for p in op.pieces}
    for name, theta in thetas.items():
        if name in sensitivity.TERMS:
            assert weights[name] == pytest.approx(theta, rel=1e-15)
    assert set(weights) <= set(sensitivity.TERMS)
    assert {"curl_curl", "mass", "conductivity", "absorbing"} <= set(thetas)
    assert "absorbing order 2" in thetas


def test_excitation_is_the_voltage_vector_times_the_reciprocity_factor(line):
    op = driven.driven_operator(line, order=2)
    for data in op.ports:
        b, voltage = terms.port_vectors(op.space, op.built, op.order, data)
        factor = terms.excitation_per_voltage(data.port.rlc[0])
        scale = np.abs(b).max()
        np.testing.assert_allclose(b, factor * voltage, rtol=1e-13, atol=1e-13 * scale)
