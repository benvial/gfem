r"""Junction coupling of the straight and the bent leaky slab: exact $\tau^2$.

The six literals of `bend_cases.SLAB_TAU2` (Ey and Hy, $R$ = 3, 5, 8 um) are the
exact overlap of the straight slab mode with the bent one (the research
reference). The pairing is the engine's on the whole section, layer included.
The error falls from $p = 2$ to $p = 5$ on the coarse structured mesh (a
$p$-refinement); the gates are twice the error measured at $p = 5$, under the
ceiling $10^{-6}$.
"""

from __future__ import annotations

import bend_cases as bc
import finite_differences as fd
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

ORDERS = (2, 3, 4, 5)
#: The required fall of the error from $p = 2$ to $p = 5$.
DROP = 1e2
CEILING = 1e-6
#: Twice the relative error of $\tau^2$ measured at $p = 5$ (ladder $p = 2..5$ of
#: each case in the comment of the test), except that the Ey gates are not below
#: $2\times10^{-12}$: the literals hold 12 significant digits, so a comparison
#: cannot resolve less than $\sim 5\times10^{-13}$ whatever the discretization.
GATE = {
    ("Ey", 3): 2e-12,
    ("Ey", 5): 2e-12,
    ("Ey", 8): 2e-12,
    ("Hy", 3): 3.2e-11,
    ("Hy", 5): 2.7e-11,
    ("Hy", 8): 4.8e-11,
}


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("bend_junction_verification")


def _slab(workdir, pol, radius):
    # a radiating Ey bend at R = 8 um loses 1e-4 per radian: a layer
    # reflection of R0 = 1e-10 would show in the coupling, so that case
    # runs at R0 = 1e-12 (as the bent-mode ladder does)
    r0 = 1e-12 if (pol, radius) == ("Ey", 8) else 1e-10
    return bc.slab_model(bc.slab_mesh(workdir / f"slab_{pol}.msh", pol), r0)


@pytest.mark.parametrize("radius", [3, 5, 8])
@pytest.mark.parametrize("pol", ["Ey", "Hy"])
def test_the_slab_junction_matches_the_exact_coupling(workdir, pol, radius):
    model = _slab(workdir, pol, radius)
    ref = bc.SLAB_TAU2[(radius, pol)]
    # measured relative error of tau^2 at p = 2, 3, 4, 5 (MN5):
    #   Ey R=3: 1.7e-6, 4.6e-9, 1.1e-10, 6.7e-13
    #   Ey R=5: 4.0e-7, 2.8e-9, 1.6e-12, 1.9e-13
    #   Ey R=8: 2.0e-7, 1.7e-9, 2.0e-12, 9.8e-13
    #   Hy R=3: 1.6e-5, 1.0e-7, 4.7e-10, 1.6e-11
    #   Hy R=5: 2.2e-6, 1.2e-8, 3.4e-11, 1.3e-11
    #   Hy R=8: 4.5e-7, 2.7e-9, 2.8e-11, 2.4e-11
    errors = []
    for order in ORDERS:
        straight = bc.solve_mode(model, bc.SLAB_F, order=order)
        bent = bc.solve_mode(model, bc.SLAB_F, bc.slab_bend(radius), order=order)
        tau2 = complex(np.asarray(gfem.junction_coupling(model, straight, bent))[0])
        errors.append(abs(tau2 / ref - 1))
    assert errors[0] / errors[-1] >= DROP
    assert errors[-1] <= min(GATE[(pol, radius)], CEILING)


def test_the_radius_gradient_of_the_slab_junction_matches_differences(workdir):
    model = _slab(workdir, "Hy", 5)
    options = gfem.SolverOptions(order=3, threads=1)
    straight = gfem.solve(model, gfem.PortMode("p1", (bc.SLAB_F,)), options=options)
    radius = 5e-6
    rho = radius - 1.5e-6

    def objective(r):
        problem = gfem.PortMode("p1", (bc.SLAB_F,), bend=bc.slab_bend_m(r))
        bent = gfem.solve(model, problem, options=options)
        return jnp.abs(gfem.junction_coupling(model, straight, bent)[0])

    # fourth-order central difference, step $10^{-3}\rho$: truncation
    # $(\delta/\rho)^4 \approx 10^{-12}$, round-off
    # $1.5\,\epsilon_\mathrm{solve}\,\rho/\delta \approx 1.5\times10^{-10}$ for a
    # solve accurate to $10^{-13}$, so the repository's gate $10^{-6}$ keeps
    # four orders of room
    fd.assert_gradient_matches(
        objective, jnp.asarray(radius), [1.0], step=1e-3 * rho, rel=fd.RTOL
    )
