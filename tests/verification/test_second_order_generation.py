r"""Second-harmonic generation in a thin slab against the undepleted closed form.

A $\chi^{(2)}$ slab ($d_{xxx} = 20$ pm/V, 1 um, 1550 nm) in a homogeneous medium on a
periodic cell is pumped by 1 W per cell of the TM (0,0) Floquet wave, and its
second-harmonic polarization (`gfem.second_order_polarization`) is solved in a second
model at twice the frequency (no reflection at the slab in either solve). The
generated TM waves at the two Floquet ports are

.. math::

    b \propto \mathrm{sinc}(\Delta k L/2),\qquad \Delta k = k_{2\omega} - 2k_\omega

forward (top) and with $\Sigma = k_{2\omega} + 2k_\omega$ backward (bottom), the
formulas of `second_order_cases`. The frozen powers
$|b_\mathrm{top}|^2 = 7.441643048\times10^{-6}$ W (matched,
$n_{2\omega} = n_\omega = 2.2$) and $2.564315177\times10^{-6}$ W
($n_{2\omega} = 2.5875$, $\Delta kL = \pi$) are the closed form at the exact 1 W
per cell pump amplitude.

Measured on the order-3 cell of 20 cells/um along $z$ (720 tetrahedra): $|b|^2$ to
$1.2\times10^{-5}$ (matched) and $7.2\times10^{-5}$ (mismatched); the complex waves,
against the closed form with the pump amplitude fitted to the solved field, to
$1.4\times10^{-4}$ (top) and $2.1\times10^{-4}$ (bottom); the cross-polarized wave
$\le 2\times10^{-7}\sqrt{\mathrm W}$. The tolerances are at least 4 times those
(the closed form is exact for the undepleted slab; the rest is the discretization of
the pump's phase and of the load, which the next test shows falls as the mesh
refines).
"""

import pytest
from second_order_cases import run_slab, slab_mesh

pytestmark = pytest.mark.verification

#: The frozen closed-form powers (W) at the 1 W per cell pump, exact amplitude.
FROZEN = {False: 7.441643048e-6, True: 2.564315177e-6}
#: $|b|^2$ against it: measured 1.2e-5 and 7.2e-5, a margin of 7 for the coarse seeds.
REL_POWER = 5e-4
#: The complex waves against the fitted closed form: measured 2.2e-4, margin 4.5 and 9.
REL_TOP, REL_BOTTOM = 1e-3, 2e-3
#: The cross-polarized wave against the copolarized one.
REL_CROSS = 1e-3


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return slab_mesh(tmp_path_factory.mktemp("slab") / "slab.msh", 20)


@pytest.fixture(scope="module", params=[False, True], ids=["matched", "mismatched"])
def case(request, mesh):
    return request.param, run_slab(mesh, order=3, mismatched=request.param)


def test_generated_power_against_the_frozen_closed_form(case):
    mismatched, run = case
    top, _, _ = run.waves()
    assert abs(top) ** 2 == pytest.approx(FROZEN[mismatched], rel=REL_POWER)


def test_complex_waves_against_the_closed_form(case):
    _, run = case
    top, bottom, cross = run.waves()
    ref_top, ref_bottom = run.closed_form()
    assert abs(top - ref_top) <= REL_TOP * abs(ref_top)
    assert abs(bottom - ref_bottom) <= REL_BOTTOM * abs(ref_bottom)
    assert cross <= REL_CROSS * abs(top)


def test_emitted_power_is_the_sum_of_the_port_powers(case):
    # a lossless cell: the power the source emits leaves through the four ports
    _, run = case
    name = run.source.name
    ports = ("z_min TE", "z_min TM", "z_max TE", "z_max TM")
    leaving = sum(abs(complex(run.generated.port_wave(p, name)[0])) ** 2 for p in ports)
    # measured 1e-7 and 7e-7 (the balance misses the evanescent orders' leak at
    # the ports, which the pads of 0.25 um do not fully remove), tolerance 14 times
    assert float(run.generated.emitted_power[name][0]) == pytest.approx(
        leaving, rel=1e-5
    )


def test_error_falls_with_the_mesh(tmp_path):
    # order 3 at 10 cells/um has a larger error than at 20 cells/um (measured 1.9e-3
    # against 1.2e-5 on the matched slab's power: a factor 150, the test asks 5)
    errors = []
    for cells in (10, 20):
        run = run_slab(slab_mesh(tmp_path / f"slab{cells}.msh", cells), order=3)
        errors.append(abs(abs(run.waves()[0]) ** 2 / FROZEN[False] - 1))
    assert errors[0] > 5 * errors[1]
