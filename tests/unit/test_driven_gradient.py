"""Driven gradients of every electromagnetic material property.

A small line with second-order absorbing side walls and lossy, magnetic,
conducting P0 fields: each property's gradient (complex ``eps_r`` along
both axes, with a loss tangent that mixes them in the absorbing terms)
against central differences on the direct solver.

The same per principal value of a field with axes (first-order absorbing
side walls, whose admittance tensors move with the values behind them),
per element and for a uniform field, and the identities that tie the
per-axis cotangents to the isotropic ones.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from anisotropy import TILTED
from finite_differences import assert_gradient_matches, unit_directions
from solver_misses import adjoint_cannot_converge_at, cannot_converge_at

import gfem
from gfem._engine import driven, terms

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


# --- principal values on fixed axes ---------------------------------------------


@pytest.fixture(scope="module")
def axes_line(line):
    # second-order absorbing walls raise next to a field with axes
    return line.replace(
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.Absorbing("side")]
    )


def _axis_values(line):
    """Biaxial per-element values, ``(n, 3)``, of every property."""
    rng = np.random.default_rng(11)
    n = (len(line.mesh.tetrahedra), 3)
    return {
        "eps_r": jnp.asarray(rng.uniform(2.0, 5.0, n) * (1 - 0.1j)),
        "mu_r": jnp.asarray(rng.uniform(1.0, 2.0, n)),
        "tan_delta": jnp.asarray(rng.uniform(0.01, 0.2, n)),
        "conductivity": jnp.asarray(rng.uniform(0.1, 1.0, n)),
    }, rng


def _field_objective(line):
    """The S functional of the line with a field of ``values`` (a dict)."""
    options = gfem.SolverOptions(linear_solver="direct")

    def objective(values, axes=TILTED):
        field = gfem.MaterialField(**values, axes=axes)
        model = line.with_material("dielectric", field)
        s = gfem.solve(model, gfem.Driven(frequency=F), options=options).s.values
        return jnp.sum(jnp.abs(s) ** 2 * jnp.array([[1.0, 0.5], [2.0, -1.0]]))

    return objective


@pytest.mark.parametrize("name", ["eps_r", "mu_r", "tan_delta", "conductivity"])
def test_gradient_of_each_principal_value_matches_central_differences(axes_line, name):
    # the curl (mu_r) and loss (tan_delta) paths beside the matrix's eps_r
    # and conductivity cells, through the walls' admittance tensors
    values, rng = _axis_values(axes_line)
    objective = _field_objective(axes_line)
    p = values[name]
    (d,) = unit_directions(rng, p.shape, 1)
    directions = [d, 1j * d] if jnp.iscomplexobj(p) else [d]
    assert_gradient_matches(
        lambda q: objective({**values, name: q}), p, directions, step=STEPS[name]
    )


CRYSTAL = gfem.Material(
    eps_r=(3.0, 4.5, 6.0),
    mu_r=(1.2, 1.0, 1.5),
    tan_delta=(0.02, 0.05, 0.01),
    conductivity=(0.3, 0.1, 0.6),
    axes=TILTED,
)


@pytest.mark.parametrize("name", ["eps_r", "mu_r", "tan_delta", "conductivity"])
def test_gradient_of_a_uniform_field_of_principal_values(axes_line, name):
    # MaterialField.uniform of a crystal: (3,) leaves
    field = gfem.MaterialField.uniform("dielectric", CRYSTAL)
    values = {n: getattr(field, n) for n in field.properties}
    objective = _field_objective(axes_line)
    rng = np.random.default_rng(5)
    directions = unit_directions(rng, (3,), 2)
    assert_gradient_matches(
        lambda q: objective({**values, name: q}, axes=field.axes),
        values[name],
        directions,
        step=STEPS[name],
    )


def _relative(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def test_equal_principal_values_give_the_isotropic_s_and_cotangents(axes_line):
    # sum_k a_k a_k^T = I: the field with axes and equal principal values is
    # the isotropic field (its wall faces take the tensor path), and its
    # per-axis cotangents sum to the isotropic one
    values, _ = _values(axes_line)
    objective = _field_objective(axes_line)
    equal = {k: jnp.repeat(v[:, None], 3, axis=1) for k, v in values.items()}
    iso, iso_grad = jax.value_and_grad(lambda v: objective(v, axes=None))(values)
    per_axis, axes_grad = jax.value_and_grad(objective)(equal)
    # measured on MN5: S 7.3e-15 relative, cotangents 1.1e-15 .. 1.5e-15
    assert abs(float(per_axis) - float(iso)) <= 3e-14 * abs(float(iso))
    for name, g in iso_grad.items():
        assert _relative(axes_grad[name].sum(axis=1), g) <= 5e-15, name


def test_a_uniform_cotangent_is_the_sum_of_the_per_element_ones(axes_line):
    field = gfem.MaterialField.uniform("dielectric", CRYSTAL)
    uniform = {n: getattr(field, n) for n in field.properties}
    n = len(axes_line.mesh.elements("dielectric"))
    per_element = {k: jnp.tile(v, (n, 1)) for k, v in uniform.items()}
    objective = _field_objective(axes_line)
    uniform_grad = jax.grad(lambda v: objective(v, axes=field.axes))(uniform)
    element_grad = jax.grad(lambda v: objective(v, axes=field.axes))(per_element)
    # round-off of a sum over the elements: 3.3e-16 .. 5.6e-16 relative on MN5,
    # up to 1.1e-14 on the CI runners (Windows, py3.14)
    for name, g in uniform_grad.items():
        assert _relative(element_grad[name].sum(axis=0), g) <= 1e-13, name


def test_principal_values_beside_a_second_order_wall_of_isotropic_material(
    tmp_path,
):
    # the second-order absorbing term has no per-axis derivative: it reads
    # only the isotropic air behind it, so it adds nothing to the crystal's
    side, length = 2.0, 6.0
    tol = 1e-9 * length

    def surface(c):
        if abs(c[2]) < tol:
            return "port"
        if abs(c[2] - length) < tol:
            return "end"
        return "ywalls" if min(abs(c[1]), abs(c[1] - side)) < tol else ""

    def volume(c):
        return "crystal" if c[2] < length / 2 else "air"

    edges = (
        np.linspace(0, side, 3),
        np.linspace(0, side, 3),
        np.linspace(0, length, 7),
    )
    path = sm.write_msh22(tmp_path / "slab.msh", sm.structured(edges, volume, surface))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    options = gfem.SolverOptions(linear_solver="direct")

    def objective(eps):
        model = gfem.Model(
            mesh,
            materials={
                "crystal": gfem.MaterialField(eps_r=eps, axes=TILTED),
                "air": gfem.Material(),
            },
            boundaries=[gfem.PEC("ywalls"), gfem.Absorbing("end", order=2)],
            ports=[gfem.LumpedPort("p1", "port", "+y")],
        )
        s = gfem.solve(model, gfem.Driven(frequency=10e9), options=options).s
        return jnp.sum(jnp.abs(s.values) ** 2)

    rng = np.random.default_rng(13)
    n = len(mesh.elements("crystal"))
    eps = jnp.asarray(rng.uniform(2.0, 5.0, (n, 3)))
    directions = unit_directions(rng, eps.shape, 1)
    assert_gradient_matches(objective, eps, directions, step=STEPS["eps_r"])


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
    model = line.with_material("dielectric", gfem.Material(eps_r=2.0, conductivity=1.0))
    op = driven.driven_operator(model, order=1)
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
    differentiated = driven.operator_terms(model, op, nodes=False)
    assert set(expected) <= {t.name for t in differentiated if t.derivatives}
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
