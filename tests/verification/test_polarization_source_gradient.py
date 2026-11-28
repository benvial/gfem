r"""Finite differences through the pump solve and the generated solve.

The $\chi^{(2)}$ slab of `second_order_cases` (10 cells/um along z, 336
tetrahedra) with a per-element relative permittivity $p$ in both solves: the pump
at $\omega$ feels $p$, the polarization $P \propto E_\omega E_\omega$ is formed
from its kept field, and the generated solve at $2\omega$ feels $p$ again, so the
gradient reaches $p$ through the generated operator, the pump's operator and
the polarization. $J = (|b_\mathrm{top}|^2 + 2|b_\mathrm{bottom}|^2)/J_0$ of the
TM Floquet waves, with $J_0$ its value at the base design (so $J = O(1)$). The
pump and generated solves take orders 2/2, 2/1 and 1/2, which tests the rule
points shared by two Nédélec orders.

Step. The fourth-order central difference (`finite_differences`) has truncation
$h^4 J^{(5)}/30$ and round-off $1.5\delta/h$ relative to the derivative; the
direct solves' $\delta$ is below $10^{-14}$ (condition number below $10^{6}$ times
$10^{-16}$, with a margin of 100), and a direction can sit well below the gradient's
norm. Measured relative errors of the gradient against the difference, over the
three directions of each order: at $h = 10^{-2}$ up to $3\times10^{-8}$ (order 1/2,
a direction whose derivative is small: round-off, since the error falls to
$4.5\times10^{-9}$ at $2\times10^{-2}$) and $\le 5\times10^{-10}$ elsewhere; at
$h = 2\times10^{-2}$ up to $4.5\times10^{-9}$ (the truncation $h^4/30$ times the
fifth derivative, about $5\times10^{-9}\,J^{(5)}/J'$ at most, is within the same
measure). So $h = 2\times10^{-2}$, 200 times below the gate $10^{-6}$.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from second_order_cases import FREQUENCY, D, slab_mesh, slab_model

import gfem

pytestmark = [
    pytest.mark.verification,
    # the order-1 faces of the coarse cell resolve fewer Floquet orders than the
    # cap asks for; the warning is about the face mesh, not about the gradient
    pytest.mark.filterwarnings("ignore::gfem.errors.FloquetOrderCapWarning"),
]

#: Difference step along a unit direction (see the module docstring).
STEP = 2e-2
#: The pump and generated solve orders.
ORDERS = [(2, 2), (2, 1), (1, 2)]


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return slab_mesh(tmp_path_factory.mktemp("cell") / "cell.msh", 10)


def _objective(mesh, orders):
    pump_order, generated_order = orders
    tensor = np.zeros((3, 6))
    tensor[0, 0] = D

    def waves(p):
        pump_model = slab_model(mesh, 2.2, chi2=False).with_material(
            "slab", gfem.MaterialField(eps_r=p, second_order_nonlinear_tensor=tensor)
        )
        generated_model = slab_model(mesh, 2.2, chi2=False).with_material(
            "slab", gfem.MaterialField(eps_r=p)
        )
        pump = gfem.solve(
            pump_model,
            gfem.Driven(FREQUENCY, excite="z_min TM", keep_fields=True),
            options=gfem.SolverOptions(order=pump_order, linear_solver="direct"),
        )
        source = gfem.second_order_polarization(
            pump_model,
            gfem.Pump(pump, FREQUENCY, incident_power=1.0),
            generated_order=generated_order,
        )
        generated = gfem.solve(
            generated_model,
            gfem.Driven(source.frequency, excite=(), sources=source),
            options=gfem.SolverOptions(order=generated_order, linear_solver="direct"),
        )
        top = generated.port_wave("z_max TM", source.name)[0]
        bottom = generated.port_wave("z_min TM", source.name)[0]
        return jnp.abs(top) ** 2 + 2 * jnp.abs(bottom) ** 2

    n = len(mesh.elements("slab"))
    rng = np.random.default_rng(7)
    p0 = jnp.asarray(2.2**2 + 0.3 * rng.uniform(-1, 1, n))
    base = waves(p0)
    return (lambda p: waves(p) / base), p0, rng


@pytest.mark.parametrize("orders", ORDERS, ids=lambda o: f"{o[0]}-{o[1]}")
def test_the_gradient_through_both_solves_matches_finite_differences(mesh, orders):
    objective, p0, rng = _objective(mesh, orders)
    assert_gradient_matches(objective, p0, unit_directions(rng, p0.shape, 3), step=STEP)
