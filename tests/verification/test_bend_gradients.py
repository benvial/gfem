r"""Gradients of a bent port mode: in the radius, in the material and in the section.

The radius gradient meets the exact $\mathrm{d}\gamma/\mathrm{d}R$ of the slab
(implicit-function derivative of the Bessel/Hankel root) and a fourth-order
central difference of the discrete solves for $\gamma$, `bend_loss` and
$|E|^2$; the material and node-coordinate gradients meet differences too.
"""

from __future__ import annotations

import bend_cases as bc
import finite_differences as fd
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

# The difference's step is $\delta = 10^{-3}\rho$ with $\rho$ the distance from
# the axis of the bend to the nearest wall ($R - a/2$ for the guide and the
# line, $R - 1.5$ um for the slab's wall): the truncation of the fourth-order
# difference is $(\delta/\rho)^4 \approx 10^{-12}$ and its round-off
# $1.5\,\epsilon\,|\gamma|/(\delta|f'|)$ for an objective $f$ of $\gamma$ solved
# to the relative error $\epsilon$. $|\gamma|/(\delta|f'|)$ is $1.3\times10^5$
# for the guide's $\beta$, $10^5$ for the slab's $\mathrm{Im}\,\gamma$,
# $3.6\times10^5$ (Hy) and $1.4\times10^6$ (Ey) for its $\mathrm{Re}\,\gamma$
# and loss. ARPACK's tolerance $10^{-10}$ is on $\theta = 1/(\lambda-\sigma)$,
# so it bounds $\lambda = \gamma^2$ to $10^{-10}|\lambda-\sigma|/|\lambda|$, with
# the shift $\sigma$ at the closed straight mode: 0.013 (Ey) and 0.017 (Hy) at
# $R = 5$ um, 0.013 at 30 mm. The worst-case round-off is then $1.2\times10^{-7}$
# for $\beta$, $4.5\times10^{-7}$ for the Hy slab's $\mathrm{Re}\,\gamma$ and
# loss and $1.3\times10^{-6}$ for the Ey slab's, at the repository's gate 1e-6;
# and the field and impedance objectives read the vector, which the tolerance
# bounds to about $10^{-10}$ only. The gate holds because the pairs come out
# converged to round-off, not to the tolerance (backward error
# $3\times10^{-15}$, and $\gamma$ moves by less than $4\times10^{-16}$ when the
# tolerance falls to $10^{-15}$, p = 2 to 5): the measured agreement is 1e-10
# and below.
STEP = 1e-3
RTOL = 1e-6
OPTIONS = gfem.SolverOptions(order=3, threads=1)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("bend_gradient_verification")


def _wr90_model(workdir, *, voltage: bool = False, material=None):
    mesh = gfem.Mesh.from_gmsh(bc.wr90_mesh(workdir / "wr90.msh"), length_unit="mm")
    a, b = bc.WR90_A * 1e-3, bc.WR90_B * 1e-3
    path = ((a / 2, 0.0, 0.0), (a / 2, b, 0.0)) if voltage else None
    return gfem.Model(
        mesh,
        materials={"air": material or gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "p1", voltage_path=path)],
    )


def _mode(model, frequency, bend, options=OPTIONS):
    problem = gfem.PortMode("p1", (frequency,), bend=bend)
    return gfem.solve(model, problem, options=options)


def _check_radius(objective, radius, rho):
    fd.assert_gradient_matches(
        objective, jnp.asarray(radius), [1.0], step=STEP * rho, rel=RTOL
    )


# --- the radius ---------------------------------------------------------------


@pytest.mark.parametrize(("pol", "radius"), [("Hy", 5), ("Ey", 3)])
def test_the_exact_derivative_of_the_slab_gamma_by_the_radius(workdir, pol, radius):
    model = bc.slab_model(bc.slab_mesh(workdir / f"slab_{pol}.msh", pol))
    options = gfem.SolverOptions(order=4, threads=1)

    def part(p):
        return lambda r: p(_mode(model, bc.SLAB_F, bc.slab_bend_m(r), options).gamma[0])

    r = radius * 1e-6
    got = complex(jax.grad(part(jnp.real))(r), jax.grad(part(jnp.imag))(r))
    got *= 1e-12  # 1/m^2 to 1/um^2
    ref = bc.SLAB_DGAMMA_DR[(radius, pol)]
    # measured 5.3e-10 (Hy) and 8.2e-10 (Ey) at p = 4 on the structured mesh (the
    # discretization's error of the root's derivative), gate 5e-9
    assert abs(got - ref) <= 5e-9 * abs(ref)


def test_the_radius_gradients_of_the_wr90_match_differences(workdir):
    model = _wr90_model(workdir, voltage=True)
    radius = 30e-3
    rho = radius - bc.WR90_A / 2 * 1e-3
    point = np.array([[0.3 * bc.WR90_A * 1e-3, 0.4 * bc.WR90_B * 1e-3, 0.0]])

    def gamma(r):
        return jnp.imag(_mode(model, bc.WR90_F, bc.wr90_bend_m(r)).gamma[0])

    def intensity(r):
        field = _mode(model, bc.WR90_F, bc.wr90_bend_m(r)).e_field.at(point)
        return jnp.sum(jnp.abs(field) ** 2)

    def impedance(r):
        return jnp.real(_mode(model, bc.WR90_F, bc.wr90_bend_m(r)).Z_PV[0])

    for objective in (gamma, intensity, impedance):
        _check_radius(objective, radius, rho)


def test_the_radius_gradient_of_the_line_gamma_matches_differences(workdir):
    model = bc.line_model(bc.line_mesh(workdir / "line.msh"))
    radius = 20e-3
    rho = radius - bc.LINE_A / 2 * 1e-3

    def gamma(r):
        bend = gfem.Bend(r, outward="+x", at=(bc.LINE_A / 2 * 1e-3, 0.0, 0.0))
        return jnp.imag(_mode(model, bc.LINE_F, bend).gamma[0])

    _check_radius(gamma, radius, rho)


@pytest.mark.parametrize("pol", ["Hy", "Ey"])
def test_the_radius_gradients_of_the_slab_match_differences(workdir, pol):
    model = bc.slab_model(bc.slab_mesh(workdir / f"slab_{pol}.msh", pol))
    radius = 5e-6
    rho = radius - 1.5e-6
    point = np.array([[0.2e-6, 0.0, 0.0]])

    def objective(part):
        return lambda r: part(_mode(model, bc.SLAB_F, bc.slab_bend_m(r)))

    _check_radius(objective(lambda m: jnp.real(m.gamma[0])), radius, rho)
    _check_radius(objective(lambda m: jnp.imag(m.gamma[0])), radius, rho)
    _check_radius(objective(lambda m: m.bend_loss[0]), radius, rho)
    _check_radius(
        objective(lambda m: jnp.sum(jnp.abs(m.e_field.at(point)) ** 2)), radius, rho
    )


# --- the material --------------------------------------------------------------


def test_the_filling_permittivity_gradient_of_the_bent_wr90_matches_differences(
    workdir,
):
    # no layer: the metric of the bend alone is on the material kernel's path
    mesh = gfem.Mesh.from_gmsh(bc.wr90_mesh(workdir / "wr90.msh"), length_unit="mm")
    n = len(mesh.elements("air"))
    bend = bc.wr90_bend(30.0)

    def model(eps):
        return gfem.Model(
            mesh,
            materials={"air": gfem.MaterialField(eps_r=eps)},
            boundaries=[gfem.PEC("walls")],
            ports=[gfem.WavePort("p1", "p1")],
        )

    def beta(eps):
        return jnp.imag(_mode(model(eps), bc.WR90_F, bend).gamma[0])

    eps = jnp.full(n, 1.5)
    directions = fd.unit_directions(np.random.default_rng(0), eps.shape, 2)
    # the step is 1e-2 of the filling's value, as the straight port mode's tests
    fd.assert_gradient_matches(beta, eps, directions, step=1.5e-2, rel=RTOL)


# --- the section's nodes -------------------------------------------------------


@pytest.mark.parametrize(
    ("where", "axes"),
    [((0.3, 0.0), (0, 1)), ((0.1, 0.0), (1,)), ((3.0, 0.0), (1,))],
    ids=["core edge node", "core node", "layer node"],
)
def test_the_node_gradient_of_the_slab_gamma_matches_differences(workdir, where, axes):
    model = bc.slab_model(bc.slab_mesh(workdir / "slab_Hy.msh", "Hy"))
    mesh = model.mesh
    x = np.asarray(mesh.imported_nodes)
    node = int(np.argmin(np.linalg.norm(x - [where[0] * 1e-6, where[1], 0.0], axis=1)))
    structure = jax.tree_util.tree_structure(mesh)
    x0 = jnp.asarray(mesh.nodes)

    def beta(nodes):
        moved = jax.tree_util.tree_unflatten(structure, [nodes])
        return jnp.imag(
            _mode(model.replace(mesh=moved), bc.SLAB_F, bc.slab_bend(5.0)).gamma[0]
        )

    directions = []
    for axis in axes:
        d = np.zeros(x0.shape)
        d[node, axis] = 1.0
        directions.append(d)
    # NODE_STEP: 1e-3 of the local edge length (0.1 um cells)
    fd.assert_gradient_matches(beta, x0, directions, step=1e-3 * 0.1e-6, rel=RTOL)


@pytest.mark.parametrize(
    ("where", "axis"),
    [((0.0, 0.5), 0), ((0.5, 0.0), 1)],
    ids=["side wall node, along x", "bottom wall node, along y"],
)
def test_the_node_gradient_of_the_bent_wr90_gamma_matches_differences(
    workdir, where, axis
):
    # no layer: the node kernel takes the bend's metric on its position path.
    # A wall node moved out of the wall changes the section's width or height,
    # so the gradient is large (-1.4e3, 1.8e3 1/m per m): an interior node's
    # is a discretization error of 1e-4, at the round-off of the differences
    model = _wr90_model(workdir)
    mesh = model.mesh
    x = np.asarray(mesh.imported_nodes) * 1e3
    target = [where[0] * bc.WR90_A, where[1] * bc.WR90_B, 0.0]
    node = int(np.argmin(np.linalg.norm(x - target, axis=1)))
    structure = jax.tree_util.tree_structure(mesh)
    x0 = jnp.asarray(mesh.nodes)

    def beta(nodes):
        moved = jax.tree_util.tree_unflatten(structure, [nodes])
        return jnp.imag(
            _mode(model.replace(mesh=moved), bc.WR90_F, bc.wr90_bend(30.0)).gamma[0]
        )

    d = np.zeros(x0.shape)
    d[node, axis] = 1.0
    # 1e-3 of the local edge length (a/6 = 3.8 mm cells, in metres); measured
    # 9e-12 and 2e-11
    fd.assert_gradient_matches(beta, x0, [d], step=1e-3 * 3.8e-3, rel=RTOL)
