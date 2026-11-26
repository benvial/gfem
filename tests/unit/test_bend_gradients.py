r"""Gradients of a bent port mode in the radius: against differences, refusals, jit."""

from __future__ import annotations

import bend_cases as bc
import finite_differences as fd
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

# Every gradient is the exact derivative of the discrete mode, so it meets a
# fourth-order central difference of the same discrete solves. With the step
# $\delta = 10^{-3}\rho$ ($\rho$ the distance from the axis of the bend to the
# nearest wall) the truncation is $(\delta/\rho)^4 \approx 10^{-12}$ and the
# round-off $1.5\,\epsilon\,|\gamma|/(\delta|f'|)$ for an objective $f$ of
# $\gamma$ solved to the relative error $\epsilon$; $|\gamma|/(\delta|f'|)$ is
# $1.3\times10^5$ for the guide's $\beta$ and $n_\mathrm{eff}$, $4\times10^5$
# for the slab's $\mathrm{Re}\,\gamma$ and loss, $8\times10^4$ for its
# $\mathrm{Im}\,\gamma$. ARPACK's tolerance $10^{-10}$ is on
# $\theta = 1/(\lambda-\sigma)$, so it bounds $\lambda = \gamma^2$ only to
# $10^{-10}|\lambda-\sigma|/|\lambda|$, with the shift $\sigma$ the closed
# straight mode's eigenvalue: $|\lambda-\sigma|/|\lambda| = 0.013$ (guide) and
# 0.017 (slab), $\epsilon \le 8.3\times10^{-13}$, a round-off of up to
# $5.5\times10^{-7}$.
# The pairs come out converged to round-off, not to the tolerance (measured
# at p = 2 and 3: backward error $3\times10^{-15}$, and $\gamma$ moves by less
# than $4\times10^{-16}$ when the tolerance falls to $10^{-15}$), so the gates
# below are the measured agreement times 10-20: under the tolerance's worst
# case, as the straight port mode's verification gates are (its exact
# $n_\mathrm{eff}$ to $10^{-12}$, from a shift beyond every mode).
STEP = 1e-3
OPTIONS = gfem.SolverOptions(order=2, threads=1)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("bend_gradients")


@pytest.fixture(scope="module")
def wr90(workdir):
    return bc.wr90_model(bc.wr90_mesh(workdir / "wr90.msh"))


@pytest.fixture(scope="module")
def slab(workdir):
    return bc.slab_model(bc.slab_mesh(workdir / "slab_Hy.msh", "Hy"))


def _gamma(model, frequency, bend_of, radius):
    problem = gfem.PortMode("p1", (frequency,), bend=bend_of(radius))
    return gfem.solve(model, problem, options=OPTIONS).gamma[0]


def _check(objective, radius, rho, rel):
    """``jax.grad`` of a real ``objective(radius)`` against the difference."""
    fd.assert_gradient_matches(
        objective, jnp.asarray(radius), [1.0], step=STEP * rho, rel=rel
    )


# measured 2e-10 (Im gamma) and 2e-10 (n_eff) at p = 2, R = 30 mm: gate 10x
WR90_RTOL = 2e-9


def test_the_wr90_radius_gradient_of_gamma_matches_differences(wr90):
    radius = 30e-3
    rho = radius - bc.WR90_A / 2 * 1e-3

    def beta(r):
        return jnp.imag(_gamma(wr90, bc.WR90_F, bc.wr90_bend_m, r))

    _check(beta, radius, rho, WR90_RTOL)


def test_the_wr90_n_eff_gradient_matches_differences(wr90):
    radius = 30e-3
    rho = radius - bc.WR90_A / 2 * 1e-3

    def n_eff(r):
        problem = gfem.PortMode("p1", (bc.WR90_F,), bend=bc.wr90_bend_m(r))
        return jnp.real(gfem.solve(wr90, problem, options=OPTIONS).n_eff[0])

    _check(n_eff, radius, rho, WR90_RTOL)


# the slab's complex metric (the layer's coordinate inside $h$): measured 3e-11
# (Re gamma), 5e-11 (Im gamma) and 3e-11 (bend_loss) at p = 2, gate 20x
SLAB_RTOL = 1e-9


@pytest.mark.parametrize("part", [jnp.real, jnp.imag])
def test_the_slab_radius_gradient_of_gamma_matches_differences(slab, part):
    radius = 5e-6
    rho = radius - 1.5e-6

    def objective(r):
        return part(_gamma(slab, bc.SLAB_F, bc.slab_bend_m, r))

    _check(objective, radius, rho, SLAB_RTOL)


def test_the_slab_bend_loss_gradient_matches_differences(slab):
    radius = 5e-6

    def loss(r):
        problem = gfem.PortMode("p1", (bc.SLAB_F,), bend=bc.slab_bend_m(r))
        return gfem.solve(slab, problem, options=OPTIONS).bend_loss[0]

    _check(loss, radius, radius - 1.5e-6, SLAB_RTOL)


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["n_g", "beta_2"])
def test_the_dispersion_outputs_do_not_differentiate(wr90, name):
    def objective(r):
        problem = gfem.PortMode("p1", (bc.WR90_F,), bend=bc.wr90_bend_m(r))
        out = getattr(gfem.solve(wr90, problem, options=OPTIONS), name)
        return jnp.real(out[0])

    with pytest.raises(gfem.NonDifferentiableError, match=name):
        jax.grad(objective)(jnp.asarray(30e-3))


def test_a_traced_outward_or_point_or_frequency_raises(wr90):
    def by_outward(x):
        bend = gfem.Bend(30e-3, outward=(1.0, x, 0.0), at=(0.0, 0.0, 0.0))
        return (
            gfem.solve(
                wr90, gfem.PortMode("p1", (bc.WR90_F,), bend=bend), options=OPTIONS
            )
            .gamma[0]
            .imag
        )

    def by_point(x):
        bend = gfem.Bend(30e-3, outward="+x", at=(x, 0.0, 0.0))
        return (
            gfem.solve(
                wr90, gfem.PortMode("p1", (bc.WR90_F,), bend=bend), options=OPTIONS
            )
            .gamma[0]
            .imag
        )

    def by_frequency(f):
        problem = gfem.PortMode("p1", (f,), bend=bc.wr90_bend_m(30e-3))
        return gfem.solve(wr90, problem, options=OPTIONS).gamma[0].imag

    for objective in (by_outward, by_point, by_frequency):
        with pytest.raises(gfem.NonDifferentiableError):
            jax.grad(objective)(0.1 if objective is not by_frequency else bc.WR90_F)


# --- jit -----------------------------------------------------------------------


def test_the_radius_gradient_under_jit_equals_the_eager_one(wr90):
    def beta(r):
        return jnp.imag(_gamma(wr90, bc.WR90_F, bc.wr90_bend_m, r))

    radius = jnp.asarray(30e-3)
    eager = float(jax.grad(beta)(radius))
    jitted = float(jax.jit(jax.grad(beta))(radius))
    # the same solves and adjoint run on the host either way (measured: equal
    # to the last bit here); the gate allows a different reduction order on
    # another platform, $\sim 10^2\epsilon_\mathrm{mach}$ through the solve
    np.testing.assert_allclose(jitted, eager, rtol=1e-10)
