"""S-parameter and field gradients of a driven solve against finite differences.

A two-layer line between PEC plates, lumped ports at both ends and
second-order absorbing side walls, so every material-dependent term of
the driven operator (curl-curl, lossy mass, conductivity, both absorbing
face terms) carries a random per-element permittivity and conductivity
(`gradient_cases.layered_line`). The S cells of the verification matrix
(eps or sigma x solver x sweep) are in test_gradient_matrix.py; here the
complex permittivity and the kept-field functionals.
"""

import jax.numpy as jnp
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import layered_line, line_design, s_functional, tight, with_fields

import gfem

SOLVERS = ["direct", "bddc_gmres"]
FREQUENCIES = (18e9, 27e9)
# Fourth-order differences (see tests/finite_differences.py).
STEPS = {"eps_r": 1e-2, "conductivity": 5e-2}


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return layered_line(tmp_path_factory.mktemp("line"))


def test_complex_permittivity_gradient_has_both_parts(line):
    # S is holomorphic in eps_r: JAX's cotangent dJ/dx - i dJ/dy carries
    # the derivatives along the real and the imaginary axis.
    values, rng = line_design(line, seed=2)
    problem = gfem.Driven(frequency=FREQUENCIES)

    def objective(eps):
        model = with_fields(line, {**values, "eps_r": eps})
        return s_functional(
            gfem.solve(model, problem, options=tight("direct")).s.values
        )

    eps = values["eps_r"]
    (d,) = unit_directions(rng, eps.shape, 1)
    assert_gradient_matches(objective, eps, [d, 1j * d], step=STEPS["eps_r"])


# --- field functionals: one adjoint solve per kept frequency ---------------------


def _field_objective(line, values, name, options):
    problem = gfem.Driven(frequency=FREQUENCIES, keep_fields=FREQUENCIES[1])

    def objective(p):
        model = with_fields(line, {**values, name: p})
        result = gfem.solve(model, problem, options=options)
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
    values, rng = line_design(line, seed=4)
    objective = _field_objective(line, values, name, tight(solver))
    reference = _field_objective(line, values, name, tight("direct"))
    p = values[name]
    directions = unit_directions(rng, p.shape, 3)
    assert_gradient_matches(
        objective, p, directions, step=STEPS[name], reference=reference
    )
