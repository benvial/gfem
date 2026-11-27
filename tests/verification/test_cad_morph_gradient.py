r"""``jax.grad`` through a cylinder's CAD morph against differences of re-meshes."""

import cad_cases as cc
import jax
import pytest

import gfem

#: The disk's mesh size, um: 20k tetrahedra.
LC = 0.1


@pytest.fixture(scope="module")
def morph():
    # Inside a fixture, not at module level: collecting this file must not
    # import gmsh in jobs that have no system libraries for it.
    pytest.importorskip("gmsh")
    return gfem.shape.cad_morph(cc.disk_build(LC), {"radius": cc.R0}, cc.make_model)


def test_the_radius_gradient_matches_a_difference_of_two_remeshes(morph):
    # Central difference of J over two re-meshes at $R_0 \pm h$,
    # $h = 2^{-3}$ um (radius 0.275 to 0.525, no topology change), against
    # `jax.grad` at $R_0$. The relative gap budget is
    # $$ 2\left[\,5\sigma\kappa/(\sqrt2 h) + c_2 h^2 + \beta\,\right] $$
    # with, measured on this disk at lc 0.1 (20k tetrahedra),
    # $\sigma = 10^{-5}$ (quadratic-fit residual over re-meshes),
    # $\kappa = |J/J'| = 5.1$ um and $c_2 = 0.07\ \mu\mathrm{m}^{-2}$, so noise
    # is $1.4\times10^{-3}$ and truncation $1.1\times10^{-3}$ before the
    # bias. The bias is the faceting of the curved wall: the morph keeps the
    # number of facets fixed, so its gradient is the derivative of a polygon
    # and falls short of the smooth trend by $\beta = 0.18(\mathrm{lc}/R)^2
    # = 1.2\times10^{-2}$ (measured +1.0e-2 to +1.2e-2 over three draws).
    # The sum is $1.45\times10^{-2}$, so rtol = 3e-2 leaves a margin of 2.1.
    h = 2.0**-3
    grad = jax.grad(
        lambda r: cc.capacitance(morph.model.replace(mesh=morph({"radius": r})))
    )(cc.R0)
    plus = cc.capacitance(morph.remesh({"radius": cc.R0 + h}).model)
    minus = cc.capacitance(morph.remesh({"radius": cc.R0 - h}).model)
    difference = (plus - minus) / (2 * h)
    gap = abs(float(grad) - float(difference)) / abs(float(difference))
    assert gap < 3e-2, f"gap {gap:.2e}"
