"""The node derivative of a port mode's field.

The field's DOF vector differentiates with respect to the face's node
coordinates at fixed DOFs (the Lagrangian derivative). The check is a
phase-invariant quadratic form of the vector on the slab guide, against a
fourth-order central difference over moved nodes and a fresh solve; an
objective that depends on the phase of the field, and a point read (`at`),
raise.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem._mode_overlap import _vector

DATA = Path(__file__).parent / "data"
F = 16e9
OPTIONS = gfem.SolverOptions(order=2)
MESH = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
MATERIALS = {"slab": gfem.Material(eps_r=4.0), "air": gfem.Material()}


def solve_at(nodes):
    model = gfem.Model(
        MESH,
        materials=MATERIALS,
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    model = model.replace(mesh=model.mesh.with_nodes(nodes))
    return gfem.solve(model, gfem.PortMode("p1", [F]), options=OPTIONS)


NODES = np.asarray(MESH.nodes, float)
SECTION = solve_at(jnp.asarray(NODES)).e_field._space.section


def displacement():
    """A random in-plane displacement (m) of the port face's nodes."""
    used = np.unique(np.asarray(MESH.triangles)[SECTION.faces])
    local = np.random.default_rng(3).standard_normal((len(used), 2))
    v = np.zeros_like(NODES)
    v[used] = local @ SECTION.frame[:2] * SECTION.scale
    p = SECTION.points[np.asarray(SECTION.triangles)]
    edge = np.min(np.linalg.norm(p[:, 1] - p[:, 0], axis=1)) * SECTION.scale
    return v, edge


def test_the_field_node_derivative_matches_a_central_difference():
    v, edge = displacement()
    n_dof = _vector(solve_at(jnp.asarray(NODES)).e_field).shape[-1]
    w = jnp.asarray(np.random.default_rng(5).uniform(0.5, 1.5, n_dof))

    def objective(nodes):
        vec = _vector(solve_at(nodes).e_field)[0]
        return jnp.sum(w * jnp.abs(vec) ** 2)  # invariant under the phase of vec

    got = float(jnp.sum(jax.grad(objective)(jnp.asarray(NODES)) * v))
    # fourth-order central difference: the truncation is $(h/\ell)^4 \sim
    # 10^{-12}$ at $h = 10^{-3}\ell_\mathrm{min}$ and the noise of the solve
    # over $h/\ell$ is $10^{-10}$ to $10^{-8}$; the gate leaves two decades
    h = 1e-3 * edge
    f = [float(objective(jnp.asarray(NODES + t * h * v))) for t in (2, 1, -1, -2)]
    fd = (-f[0] + 8 * f[1] - 8 * f[2] + f[3]) / (12 * h)
    assert got == pytest.approx(fd, rel=1e-6)


def test_a_phase_dependent_objective_under_node_gradients_raises():
    c = jnp.asarray(np.random.default_rng(7).standard_normal(1) + 0.3j)

    def objective(nodes):
        vec = _vector(solve_at(nodes).e_field)[0]
        return jnp.real(jnp.sum(c * vec))

    # raised on the host in the backward pass: jax.grad hands it on
    with pytest.raises(jax.errors.JaxRuntimeError, match="through the field's phase"):
        jax.grad(objective)(jnp.asarray(NODES))


def test_a_point_read_under_node_gradients_still_raises():
    def objective(nodes):
        field = solve_at(nodes).e_field
        return jnp.abs(field.at([0.0, 0.0, 0.0])[0, 0])

    with pytest.raises(gfem.NonDifferentiableError, match=r"PortModeField\.at"):
        jax.grad(objective)(jnp.asarray(NODES))
