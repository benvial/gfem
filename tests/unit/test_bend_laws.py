r"""A permittivity law on a bent section: its metric, radius, node and law gradients.

The leaky Ey slab of `bend_cases` at $R = 5$ um, its core a lossless Lorentz
law whose $\varepsilon(f)$ at the solve's frequency equals the constant core's
$n_\mathrm{co}^2$: the bent mode, its radius derivative and its law gradient.
"""

from __future__ import annotations

import bend_cases as bc
import finite_differences as fd
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

OPTIONS = gfem.SolverOptions(order=2, threads=1)
RADIUS = 5e-6
#: The Lorentz term: far from resonance, so $\chi$ is real and smooth in its
#: parameters.
DELTA_EPS = 1.0
RESONANCE = 3 * bc.SLAB_F


def _term(delta_eps):
    return gfem.Lorentz(delta_eps=delta_eps, resonance_frequency=RESONANCE, damping=0.0)


def _law(delta_eps=DELTA_EPS):
    r"""The core's law: $\varepsilon_\infty$ sets $\varepsilon(f) = n_\mathrm{co}^2$."""
    chi = DELTA_EPS * RESONANCE**2 / (RESONANCE**2 - bc.SLAB_F**2)
    return gfem.PermittivityLaw(
        eps_inf=bc.SLAB["nco"] ** 2 - chi, terms=[_term(delta_eps)]
    )


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return bc.slab_mesh(tmp_path_factory.mktemp("bend_laws") / "slab_Ey.msh", "Ey")


@pytest.fixture(scope="module")
def plain(mesh):
    return bc.slab_model(mesh)


def _with_law(model, law):
    return model.with_material("core", gfem.Material(eps_r=law))


@pytest.fixture(scope="module")
def lawful(plain):
    return _with_law(plain, _law())


def _gamma(model, radius):
    problem = gfem.PortMode("p1", (bc.SLAB_F,), bend=bc.slab_bend_m(radius))
    return gfem.solve(model, problem, options=OPTIONS).gamma[0]


# The two cores assemble different pencils (one eps piece, or eps_inf plus the
# susceptibility's own piece) that agree to round-off. Each solve's $\lambda$ is
# within ARPACK's $10^{-10}|\lambda-\sigma|$ of its pencil's ($10^{-10}$ is on
# $1/(\lambda-\sigma)$), the shift $\sigma$ at the closed straight mode with
# $|\lambda-\sigma|/|\lambda| = 0.013$: the two $\nu$ agree to
# $2 \times 0.013\times10^{-10}/2 = 1.3\times10^{-12}$ at worst. Measured
# $4\times10^{-16}$ (the same Krylov path on both); a core without the bend's
# metric is $10^{-3}$ off.
NU_RTOL = 2e-12


def test_a_law_core_bends_as_the_constant_core_it_equals(plain, lawful):
    nu_plain = -1j * complex(_gamma(plain, RADIUS)) * RADIUS
    nu_law = -1j * complex(_gamma(lawful, RADIUS)) * RADIUS
    np.testing.assert_allclose(nu_law, nu_plain, rtol=NU_RTOL)


def _dgamma_dr(model):
    r"""$\mathrm{d}\gamma/\mathrm{d}R$ from one forward solve.

    The cotangent $c$ pulls back $\mathrm{Re}(c\gamma)$: $c = 1$ gives
    $\mathrm{Re}\,\gamma'$ and $c = i$ gives $-\mathrm{Im}\,\gamma'$.
    """
    _, pull = jax.vjp(lambda r: _gamma(model, r), jnp.asarray(RADIUS))
    (re,), (im,) = pull(jnp.asarray(1.0 + 0j)), pull(jnp.asarray(1j))
    return complex(re) - 1j * complex(im)


# The radius derivative reads the mode's vector, which the eigen-solve's
# tolerance bounds to about $10^{-10}$ (the residual of $1/(\lambda-\sigma)$ over
# its gap to the next one): the gate is that worst case on
# $|\mathrm{d}\gamma/\mathrm{d}R|$, and on the node derivative below. Measured
# $5\times10^{-15}$; a law core without the bend's metric is 14 % off.
GRAD_RTOL = 1e-10


def test_a_law_core_has_the_constant_cores_radius_derivative(plain, lawful):
    want = _dgamma_dr(plain)
    assert abs(_dgamma_dr(lawful) - want) <= GRAD_RTOL * abs(want)


def test_a_law_core_has_the_constant_cores_node_derivative(plain, lawful):
    # a core node off the axis: the law's metric moves with its position
    x = np.asarray(plain.mesh.imported_nodes)
    node = int(np.argmin(np.linalg.norm(x - [0.1e-6, 0.0, 0.0], axis=1)))
    structure = jax.tree_util.tree_structure(plain.mesh)
    x0 = jnp.asarray(plain.mesh.nodes)

    def derivative(model):
        def beta(nodes):
            moved = jax.tree_util.tree_unflatten(structure, [nodes])
            return jnp.imag(_gamma(model.replace(mesh=moved), RADIUS))

        return float(jax.grad(beta)(x0)[node, 0])

    want = derivative(plain)
    assert abs(derivative(lawful) - want) <= GRAD_RTOL * abs(want)


# Fourth-order central difference in the term's $\Delta\varepsilon$: truncation
# $(\delta/\Delta\varepsilon)^4 \approx 10^{-12}$ at $\delta = 10^{-3}$ (measured
# $5\times10^{-12}$ at $3\times10^{-3}$, falling as $\delta^4$), round-off
# $1.5\,\epsilon\,|\gamma|/(\delta|\beta'|) = 1.1\times10^4\epsilon$ with
# $|\gamma|/|\beta'| = 7.2$ and $\epsilon \le 6.5\times10^{-13}$ the solve's worst
# case (as for `NU_RTOL`), so a round-off of $7\times10^{-9}$ at most; the
# measured agreement is $2\times10^{-13}$.
LAW_STEP = 1e-3
LAW_RTOL = 1e-8


def test_the_law_gradient_of_a_bent_mode_matches_differences(plain):
    def objective(delta_eps):
        return jnp.imag(_gamma(_with_law(plain, _law(delta_eps)), RADIUS))

    fd.assert_gradient_matches(
        objective, jnp.asarray(DELTA_EPS), [1.0], step=LAW_STEP, rel=LAW_RTOL
    )
