"""The shared finite-difference gradient check (tests/finite_differences.py).

Importing gfem switches JAX to float64, as in every solve test.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import (
    assert_gradient_matches,
    assert_taylor_second_order,
    derivative,
    unit_directions,
)

import gfem  # noqa: F401


def _exp_sum(p):
    return jnp.sum(jnp.exp(p))


def test_the_difference_is_fourth_order_in_the_step():
    p = jnp.linspace(-0.5, 0.5, 7)
    d = unit_directions(np.random.default_rng(0), p.shape, 1)[0]
    exact = float(jnp.exp(p) @ d)
    errors = [abs(derivative(_exp_sum, p, d, h) - exact) for h in (0.2, 0.1)]
    assert errors[0] < 1e-4 * abs(exact)
    assert errors[0] / errors[1] == pytest.approx(16.0, rel=0.05)


def test_unit_directions_are_reproducible_unit_vectors():
    a = unit_directions(np.random.default_rng(3), (4, 5), 3)
    b = unit_directions(np.random.default_rng(3), (4, 5), 3)
    assert len(a) == 3
    for x, y in zip(a, b, strict=True):
        assert x.shape == (4, 5)
        assert np.linalg.norm(x) == pytest.approx(1.0)
        np.testing.assert_array_equal(x, y)


def test_a_derivative_far_below_the_gradient_norm_is_resolved_under_noise():
    # A direction almost orthogonal to the gradient: the derivative is 1e-3
    # of |grad|, the objective carries 1e-15 relative noise (a threaded
    # assembly's summation order), and the check still holds at 1e-6.
    rng = np.random.default_rng(1)
    c = rng.uniform(-1.0, 1.0, 50)
    noise = np.random.default_rng(2)

    def objective(p):
        value = jnp.sum(c * jnp.sin(p))
        return value * (1 + 1e-15 * noise.standard_normal())

    p = jnp.asarray(rng.uniform(0.0, 1.0, 50))
    grad = c * jnp.cos(p)
    d = rng.standard_normal(50)
    d -= (d @ grad) / (grad @ grad) * grad  # orthogonal ...
    d += 1e-3 * np.linalg.norm(d) * grad / np.linalg.norm(grad)  # ... almost
    d /= np.linalg.norm(d)
    assert abs(float(grad @ d)) < 2e-3 * float(jnp.linalg.norm(grad))
    for _ in range(20):
        assert_gradient_matches(objective, p, [d], step=3e-2, grad=grad)


def test_a_wrong_gradient_fails_with_both_numbers_in_the_message():
    p = jnp.linspace(0.0, 1.0, 5)
    grad = jnp.exp(p) * (1 + 1e-5)
    directions = unit_directions(np.random.default_rng(0), p.shape, 2)
    with pytest.raises(AssertionError, match=r"gradient .* finite difference"):
        assert_gradient_matches(_exp_sum, p, directions, step=1e-2, grad=grad)


def test_the_gradient_is_taken_from_the_objective_by_default():
    p = jnp.linspace(0.0, 1.0, 5)
    directions = unit_directions(np.random.default_rng(0), p.shape, 2)
    assert_gradient_matches(_exp_sum, p, directions, step=1e-2)


def test_differences_of_a_reference_objective_check_the_gradient():
    p = jnp.linspace(0.0, 1.0, 5)
    directions = unit_directions(np.random.default_rng(0), p.shape, 2)
    wrong = lambda p: 2 * _exp_sum(p)  # noqa: E731
    with pytest.raises(AssertionError):
        assert_gradient_matches(_exp_sum, p, directions, step=1e-2, reference=wrong)
    assert_gradient_matches(wrong, p, directions, step=1e-2, reference=wrong)


def test_imaginary_directions_check_the_cotangent_of_a_complex_input():
    # JAX's gradient over a complex leaf is dJ/dx - i dJ/dy; the derivative
    # along i d is then Re(grad . i d) = -Im(grad) . d.
    rng = np.random.default_rng(4)
    c = jnp.asarray(rng.standard_normal(6) + 1j * rng.standard_normal(6))
    p = jnp.asarray(rng.standard_normal(6) + 1j * rng.standard_normal(6))

    def objective(p):
        return jnp.sum(jnp.abs(c * p**2) ** 2)

    d = unit_directions(rng, p.shape, 1)[0]
    assert_gradient_matches(objective, p, [d, 1j * d], step=1e-3)


def test_the_taylor_remainder_check_finds_second_order():
    p = jnp.linspace(0.0, 1.0, 5)
    d = unit_directions(np.random.default_rng(0), p.shape, 1)[0]
    assert_taylor_second_order(_exp_sum, p, d, [1e-1, 1e-2, 1e-3])
    grad = jnp.exp(p) * 1.01
    with pytest.raises(AssertionError):
        assert_taylor_second_order(_exp_sum, p, d, [1e-1, 1e-2, 1e-3], grad=grad)
