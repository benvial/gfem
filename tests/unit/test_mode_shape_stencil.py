r"""The frequency stencil of exact gamma rows gives n_g and beta_2 shape gradients.

``n_g`` and ``beta_2`` of a port mode have no backward pass; a central
difference of the rows of ``gamma`` of one ``PortMode`` does. Filled WR-90
(:math:`N = 1.5`, PEC walls, 8 GHz, order 2); the port face scaled,
:math:`X \to (1 + s) X` about its centre, which has closed forms in the
discrete problem: :math:`\mathrm{d}n_g/\mathrm{d}s = -N^2(N^2 -
n_\mathrm{eff}^2)/n_\mathrm{eff}^3`.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as sc

import gfem

DATA = Path(__file__).parent / "data"
ORDER = gfem.SolverOptions(order=2)
N, F = 1.5, 8e9


@pytest.fixture(scope="module")
def guide():
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    nodes = np.asarray(mesh.nodes, float)
    face = np.isclose(nodes[:, 2], 0.0)  # port1 lies in z = 0
    move = np.zeros_like(nodes)
    move[face, :2] = nodes[face, :2] - nodes[face, :2].mean(axis=0)

    def solve(nodes, frequency):
        model = gfem.Model(
            mesh.with_nodes(nodes),
            materials={"air": gfem.Material(eps_r=N**2)},
            boundaries=[gfem.PEC("walls")],
            ports=[gfem.WavePort("p1", "port1")],
        )
        return gfem.solve(model, gfem.PortMode("p1", frequency), options=ORDER)

    return solve, jnp.asarray(nodes), jnp.asarray(move)


def _n_g(solve, nodes, delta):
    beta = jnp.imag(solve(nodes, F * (1 + delta * np.arange(-1, 2))).gamma)
    return sc.c * (beta[2] - beta[0]) / (2 * 2 * np.pi * F * delta)


def _beta_2(solve, nodes, delta):
    beta = jnp.imag(solve(nodes, F * (1 + delta * np.arange(-2, 3))).gamma)
    w = 2 * np.pi * F * delta
    return (-beta[4] + 16 * beta[3] - 30 * beta[2] + 16 * beta[1] - beta[0]) / (
        12 * w**2
    )


def test_the_gradient_of_the_stencil_matches_a_difference_of_the_stencil(guide):
    solve, nodes, move = guide
    delta = 1e-2

    def stencil(s):
        return _n_g(solve, nodes + s * move, delta)

    got = float(jnp.sum(jax.grad(_n_g, argnums=1)(solve, nodes, delta) * move))
    # fourth-order central difference in the scale s, h = 1e-3: truncation
    # about h^4 = 1e-12 of the figure's scale; solve noise 1e-14 of beta over
    # the stencil's 2 delta = 2e-2 is 1e-12, over h is 1e-9. Measured 3.9e-9;
    # gate 1e-6
    h = 1e-3
    f = [float(stencil(t * h)) for t in (2, 1, -1, -2)]
    fd = (-f[0] + 8 * f[1] - 8 * f[2] + f[3]) / (12 * h)
    assert got == pytest.approx(fd, rel=1e-6, abs=0)


def test_the_group_index_gradient_follows_the_closed_form(guide):
    solve, nodes, move = guide
    mode = solve(nodes, [F])
    n_eff = float(mode.n_eff[0].real)
    closed = -(N**2) * (N**2 - n_eff**2) / n_eff**3
    got = float(jnp.sum(jax.grad(_n_g, argnums=1)(solve, nodes, 1e-3) * move))
    # the stencil's gradient is the exact derivative of the stencil, which
    # differs from the true derivative by O(delta^2) = 1e-6 times a
    # coefficient near 3 (near cutoff); measured 2.9e-6, gate 1e-5
    assert got == pytest.approx(closed, rel=1e-5, abs=0)


def test_the_dispersion_stencil_matches_the_reported_value(guide):
    solve, nodes, _ = guide
    reported = float(solve(nodes, [F]).beta_2[0].real)
    # 5-point truncation O(delta^4) = 1e-10 times a coefficient of a few;
    # measured 1.8e-9 at delta = 3e-3 (noise 1e-14 / delta^2 = 1e-9 added);
    # gate 1e-7
    assert float(_beta_2(solve, nodes, 3e-3)) == pytest.approx(
        reported, rel=1e-7, abs=0
    )
