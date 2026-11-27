r"""``jax.grad`` through a strip's CAD morph against differences.

The tight check differences the same mesh moved by the morph; the remesh check
differences two full re-meshes of the CAD model.
"""

import cad_cases as cc
import finite_differences as fd
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

P0 = {"width": cc.W0, "height": cc.T0}

#: The strip's mesh size, um: a few elements across the 0.5 um core.
LC = 0.15


@pytest.fixture(scope="module")
def morph():
    # Inside a fixture, not at module level: collecting this file must not
    # import gmsh in jobs that have no system libraries for it.
    pytest.importorskip("gmsh")
    return gfem.shape.cad_morph(cc.strip_build(LC), P0, cc.make_model)


def test_the_gradient_over_width_and_height_matches_differences_on_the_mesh(morph):
    # Step $h = 2^{-9}$ um along unit directions of (width, height). The
    # fourth-order difference has truncation $C_4 h^4$ with the measured
    # $C_4 \approx 1.2\ \mu\mathrm{m}^{-4}$ (strip and disk, both extensions):
    # $1.2 \cdot (2^{-9})^4 = 1.7\times10^{-11}$ (measured 2.1e-11 at
    # $h = 2\times10^{-3}$). Its round-off is $1.5\,\delta\,\kappa/h$ with the
    # measured single-solve round-off $\delta = 10^{-15}$ and
    # $\kappa = |J/J'| = 7.5$ um: $5.8\times10^{-12}$; a hundred times that
    # round-off, from another platform's summation order, is $5.8\times10^{-10}$.
    # Both stay more than $10^3$ below the gate $10^{-6}$. A second-order
    # difference would truncate at $4\times10^{-7}$, too close to the gate.
    names = list(morph.names)

    def objective(p):
        moved = morph({n: p[i] for i, n in enumerate(names)})
        return cc.capacitance(morph.model.replace(mesh=moved))

    p0 = jnp.asarray([P0[n] for n in names])
    rng = np.random.default_rng(3)
    fd.assert_gradient_matches(
        objective,
        p0,
        fd.unit_directions(rng, p0.shape, 3),
        step=2.0**-9,
        rel=fd.RTOL,
    )


def test_the_width_gradient_matches_a_difference_of_two_remeshes(morph):
    # Central difference of J over two re-meshes at $w_0 \pm h$,
    # $h = 2^{-3}$ um (width 0.375 to 0.625, no topology change), against
    # `jax.grad` at $w_0$. Each re-mesh draws a new mesh, so J carries white
    # noise of relative standard deviation $\sigma$, and the difference
    # of two has $\sigma\kappa/(\sqrt2 h)$ relative to $J'$, $\kappa = |J/J'|$.
    # The budget on the relative gap is
    # $$ 2\left[\,5\sigma\kappa/(\sqrt2 h) + c_2 h^2 + \beta\,\right] $$
    # with, measured on this strip at lc 0.15 (7.1k tetrahedra),
    # $\sigma = 1.3\times10^{-5}$ (quartic-fit residual over 42 re-meshes),
    # $\kappa = 7.5$ um, $c_2 = J'''/(6J') = 0.10\ \mu\mathrm{m}^{-2}$ and a
    # bias $\beta \le 3.5\times10^{-3}$ between the morph gradient and the
    # smooth re-meshed trend (measured +0.5e-3 to +3.3e-3 over six draws).
    # Noise 2.8e-3, truncation 1.6e-3 and bias 3.5e-3 sum to $7.9\times10^{-3}$,
    # so rtol = 2e-2 leaves a margin of 2.5. A structural error in the
    # velocities moves the gradient by 40 % or more. A new geometry needs its
    # $\sigma$ measured first, with a sweep of re-meshes.
    h = 2.0**-3
    grad = jax.grad(
        lambda w: cc.capacitance(
            morph.model.replace(mesh=morph({"width": w, "height": cc.T0}))
        )
    )(cc.W0)
    plus = cc.capacitance(morph.remesh({"width": cc.W0 + h}).model)
    minus = cc.capacitance(morph.remesh({"width": cc.W0 - h}).model)
    difference = (plus - minus) / (2 * h)
    gap = abs(float(grad) - float(difference)) / abs(float(difference))
    assert gap < 2e-2, f"gap {gap:.2e}"
