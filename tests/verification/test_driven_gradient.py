"""S-parameter and field gradients of a driven solve against finite differences.

A two-layer line between PEC plates, lumped ports at both ends and
second-order absorbing side walls, so every material-dependent term of
the driven operator (curl-curl, lossy mass, conductivity, both absorbing
face terms) carries a random per-element permittivity and conductivity.
Each cell of the verification matrix (eps or sigma x solver) checks the
gradient along random directions against central differences and checks
that the Taylor remainder falls at second order.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem

SOLVERS = ["direct", "bddc_gmres"]
FREQUENCIES = (18e9, 27e9)
LAYERS = (("sub", 0.5), ("cover", 0.5))


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(4.0, 2.0, 1.0, (4, 2, 1), layers=LAYERS))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"sub": gfem.Material(), "cover": gfem.Material()},
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


def _design(line, seed=0):
    """Random P0 permittivity (lossy) and conductivity over both layers."""
    rng = np.random.default_rng(seed)
    n = len(line.mesh.tetrahedra)
    eps = rng.uniform(2.0, 6.0, n) * (1 - 0.02j)
    sigma = rng.uniform(0.5, 2.0, n)
    return {"eps_r": jnp.asarray(eps), "conductivity": jnp.asarray(sigma)}, rng


def _model(line, values):
    n_sub = len(line.mesh.elements("sub"))
    parts = {
        "sub": {k: v[:n_sub] for k, v in values.items()},
        "cover": {k: v[n_sub:] for k, v in values.items()},
    }
    return line.replace(
        materials={r: gfem.MaterialField(**p) for r, p in parts.items()}
    )


def _options(solver):
    return gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12)


WEIGHTS = np.array([[1.0, 0.4], [-0.6, 1.5]])


def _s_objective(line, values, name, options, problem=None):
    """J(p) = <W, |S|^2> + Re(c S21) summed over frequencies, p = values[name]."""
    problem = problem or gfem.Driven(frequency=FREQUENCIES)

    def objective(p):
        s = gfem.solve(
            _model(line, {**values, name: p}), problem, options=options
        ).s.values
        return jnp.sum(WEIGHTS * jnp.abs(s) ** 2) + jnp.sum(
            jnp.real((0.3 - 0.7j) * s[:, 1, 0])
        )

    return objective


def _check_fd(objective, p, rng, h, *, rel=1e-6, reference=None):
    """grad(objective) . d against central differences of ``reference``."""
    reference = reference or objective
    grad = jax.grad(objective)(p)
    for _ in range(3):
        d = rng.standard_normal(p.shape)
        d /= np.linalg.norm(d)
        fd = (reference(p + h * d) - reference(p - h * d)) / (2 * h)
        assert float(jnp.real(grad) @ d) == pytest.approx(float(fd), rel=rel)


def _check_taylor(objective, p, rng, hs):
    j0, grad = jax.value_and_grad(objective)(p)
    d = rng.standard_normal(p.shape)
    d /= np.linalg.norm(d)
    remainders = [
        abs(float(objective(p + h * d) - j0 - h * (jnp.real(grad) @ d))) for h in hs
    ]
    slopes = np.diff(np.log(remainders)) / np.diff(np.log(hs))
    assert slopes == pytest.approx([2.0] * (len(hs) - 1), abs=0.1)


# --- the verification matrix: driven x {eps, sigma} x solver x uniform sweep -----

STEPS = {"eps_r": 1e-4, "conductivity": 1e-3}
TAYLOR = {"eps_r": [1e-1, 1e-2, 1e-3], "conductivity": [1.0, 1e-1, 1e-2]}


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["eps_r", "conductivity"])
def test_s_gradient_matches_central_differences(line, solver, name):
    values, rng = _design(line)
    objective = _s_objective(line, values, name, _options(solver))
    _check_fd(objective, values[name], rng, STEPS[name])


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["eps_r", "conductivity"])
def test_s_taylor_remainder_is_second_order(line, solver, name):
    values, rng = _design(line, seed=1)
    objective = _s_objective(line, values, name, _options(solver))
    _check_taylor(objective, values[name], rng, TAYLOR[name])


def test_complex_permittivity_gradient_has_both_parts(line):
    # S is holomorphic in eps_r: JAX's cotangent dJ/dx - i dJ/dy carries
    # the derivatives along the real and the imaginary axis.
    values, rng = _design(line, seed=2)
    objective = _s_objective(line, values, "eps_r", _options("direct"))
    eps = values["eps_r"]
    grad = jax.grad(objective)(eps)
    d = rng.standard_normal(eps.shape)
    d /= np.linalg.norm(d)
    h = 1e-4
    along_real = (objective(eps + h * d) - objective(eps - h * d)) / (2 * h)
    along_imag = (objective(eps + 1j * h * d) - objective(eps - 1j * h * d)) / (2 * h)
    assert float(grad.real @ d) == pytest.approx(float(along_real), rel=1e-6)
    assert float(-grad.imag @ d) == pytest.approx(float(along_imag), rel=1e-6)


# --- consistency: reciprocity vs an explicit adjoint solve -----------------------


@pytest.mark.parametrize("solver", SOLVERS)
def test_reciprocity_gradient_equals_the_explicit_adjoint(line, solver):
    # Exciting p1 only leaves S21 without p2's solution: its gradient takes
    # an adjoint solve. With both ports excited it comes from reciprocity.
    values, _ = _design(line, seed=3)

    def s21(problem):
        def objective(eps):
            s = gfem.solve(
                _model(line, {**values, "eps_r": eps}),
                problem,
                options=_options(solver),
            ).s
            return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

        return jax.grad(objective)(values["eps_r"])

    full = s21(gfem.Driven(frequency=FREQUENCIES))
    adjoint = s21(gfem.Driven(frequency=FREQUENCIES, excite="p1"))
    np.testing.assert_allclose(adjoint, full, rtol=1e-8, atol=1e-10 * abs(full).max())


# --- field functionals: one adjoint solve per kept frequency ---------------------


def _field_objective(line, values, name, options):
    problem = gfem.Driven(frequency=FREQUENCIES, keep_fields=FREQUENCIES[1])

    def objective(p):
        result = gfem.solve(_model(line, {**values, name: p}), problem, options=options)
        e = result.e_field(FREQUENCIES[1], "p1").coefficients
        g = jnp.cos(jnp.arange(e.size))
        return jnp.abs(jnp.sum(g * e)) ** 2 + jnp.sum(jnp.abs(result.s.values) ** 2)

    return objective


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["eps_r", "conductivity"])
def test_field_functional_gradient_matches_central_differences(line, solver, name):
    # The field functional is large next to its directional derivatives, so
    # the forward's 1e-12 GMRES residual would show in the differences: they
    # are taken on the direct solve, the gradient on the solver under test.
    values, rng = _design(line, seed=4)
    objective = _field_objective(line, values, name, _options(solver))
    reference = _field_objective(line, values, name, _options("direct"))
    _check_fd(objective, values[name], rng, STEPS[name], reference=reference)
