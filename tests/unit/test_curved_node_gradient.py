"""Node gradients on curved elements: ``jax.grad`` against central differences.

On a curved mesh every node moves the element map, corners and mid-edge nodes
alike, so the gradient with respect to ``mesh.nodes`` has a cotangent in all
of them. The check is a capacitance (``Electrostatic``) and an eigenfrequency
sum (``Eigenmode``, p = 2) on the order-2 fixtures, at 20 random entries (10
corner and 10 mid-edge coordinates) and along two directions (one on every
corner, one on every mid-edge node).

Step and tolerance, per entry. A central difference with step ``h`` of a
function with errors ``eps`` in its evaluation has round-off ``eps / h`` and
truncation ``M3 h^2 / 6``, so ``h ~ (3 eps / M3)^(1/3)``. Truncation is
estimated entry by entry from the step doubled (Richardson):
``|D_h - D_2h| / 3``. The tolerance is three times the sum of both,

    tau = 3 (|D_h - D_2h| / 3 + eps J / h),

with ``eps`` the relative evaluation noise of the quantity (`NOISE`). The
metres of the fixtures are of order 1, so ``h = 1e-5`` sits at the optimum
for ``eps ~ 1e-14`` and ``M3 ~ 1``. A wrong rule (the cheaper quadrature of
the derivative terms, or the first-order velocity that drags mid nodes at
half speed) is off by 1e-3 to 1e-1 of the gradient, orders of magnitude over
``tau``, so the check discriminates.
"""

from pathlib import Path

import jax
import numpy as np
import pytest

import gfem

DATA = Path(__file__).parent / "data"
STEP = 1e-5
#: Relative noise of one evaluation of each quantity. Measured as
#: |J(X + d e) - J(X) - g d| / |J| at d = 1e-9 (truncation d^2 ~ 1e-18 is
#: below it), 12 entries: 4e-16 (capacitance), 6e-16 (frequency sum), on ostau.
#: The constants are 25x and 160x those, for other BLAS/eigensolver builds
#: (the eigensolver's convergence is looser than a direct solve's round-off).
#: The worst entry then uses 5e-2 and 4e-4 of its tolerance.
NOISE = {"capacitance": 1e-14, "omega": 1e-13}
N_ENTRIES = 10


def _mesh(name):
    return gfem.Mesh.from_gmsh(DATA / f"{name}.msh", length_unit="m")


def _capacitance(mesh):
    model = gfem.Model(mesh, materials={"fill": gfem.Material(eps_r=1.0)})
    problem = gfem.Electrostatic(terminals={"in": "inner"}, ground="outer")
    return gfem.solve(model, problem).capacitance[0, 0]


def _omegas(mesh):
    model = gfem.Model(
        mesh,
        materials={"ball": gfem.Material(eps_r=1.0)},
        boundaries=[gfem.PEC("wall")],
    )
    # a nearly degenerate pair: its sum is smooth where each one is not
    problem = gfem.Eigenmode(target=1.4e8, n_modes=2)
    result = gfem.solve(model, problem, options=gfem.SolverOptions(order=2))
    return result.omega.real.sum()


CASES = {
    "capacitance": ("coax_order2", _capacitance),
    "omega": ("octant_order2", _omegas),
}


def _corners(mesh):
    return np.unique(mesh._topology.volume_elements)


def _central(function, nodes, direction, h):
    plus = function(nodes + h * direction)
    minus = function(nodes - h * direction)
    return (float(plus) - float(minus)) / (2 * h)


def _tolerance(function, nodes, direction, value, noise):
    fine = _central(function, nodes, direction, STEP)
    coarse = _central(function, nodes, direction, 2 * STEP)
    truncation = abs(fine - coarse) / 3
    return fine, 3 * (truncation + noise * abs(value) / STEP)


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    name, quantity = CASES[request.param]
    mesh = _mesh(name)
    nodes = np.asarray(mesh.nodes, float)

    def function(x):
        return quantity(mesh.with_nodes(x))

    value, gradient = jax.value_and_grad(function)(nodes)
    return request.param, mesh, nodes, function, float(value), np.asarray(gradient)


def _check(case, direction, derivative):
    kind, _, nodes, function, value, _ = case
    fine, tau = _tolerance(function, nodes, direction, value, NOISE[kind])
    assert abs(derivative - fine) < tau, (derivative, fine, tau)


@pytest.mark.parametrize("kind", ["corner", "mid"])
def test_gradient_matches_central_differences_at_random_nodes(case, kind):
    _, mesh, nodes, _, _, gradient = case
    corners = _corners(mesh)
    pool = corners if kind == "corner" else np.setdiff1d(np.arange(len(nodes)), corners)
    rng = np.random.default_rng(7)
    picks = rng.choice(pool, N_ENTRIES, replace=False)
    axes = rng.integers(0, 3, N_ENTRIES)
    for node, axis in zip(picks, axes, strict=True):
        direction = np.zeros_like(nodes)
        direction[node, axis] = 1.0
        _check(case, direction, gradient[node, axis])


@pytest.mark.parametrize("kind", ["corner", "mid"])
def test_gradient_matches_a_directional_difference_on_every_node_of_a_kind(case, kind):
    _, mesh, nodes, _, _, gradient = case
    corners = _corners(mesh)
    on = np.zeros(len(nodes), bool)
    on[corners] = kind == "corner"
    on[np.setdiff1d(np.arange(len(nodes)), corners)] = kind == "mid"
    rng = np.random.default_rng(11)
    direction = rng.uniform(-1, 1, nodes.shape) * on[:, None]
    _check(case, direction, float(np.sum(gradient * direction)))
