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

The reciprocity section compares the generated wave of a uniform guide with the
overlap model of its port modes.

A parallel-plate line (PEC plates a height $H$ apart, the sides left as the default
PMC, field along the plates' normal) filled with a $\chi^{(2)}$ crystal of
$d_{yyy} = 20$ pm/V over the whole section $0 \le s \le L$, $L = 1$ um, with a wave
port at each end. The pump is the port-1 mode at 1 W; its second-order polarization
(`gfem.second_order_polarization`) is solved at the generated frequency $\omega_3$ in
a second model of the same mesh. With the modes of port 1 (the pump's, the generated
one's) and $\kappa$ = `gfem.three_wave_coupling` of them, the generated wave at
port 2 is

.. math::

    b_3 = -i\,g\,\overline{\kappa_{3ab}}\,a_a a_b\,L\,\mathrm{sinc}(\Delta\beta L/2)\,
    e^{-i(\beta_3 + \beta_a + \sigma\beta_b)L/2},\qquad
    \Delta\beta = \beta_3 - \beta_a - \sigma\beta_b,

with $g = 1$ for second-harmonic generation ($a = b$) and $g = 2$ for sum
($\sigma = +1$) and difference ($\sigma = -1$, pump $b$ conjugated) frequency
generation, $a$ the pump's amplitude and the $\beta$ the modes' propagation constants.
Both ports carry a polarity arrow along $+y$, so the modes' signs agree at the two
ends and the phase of $b_3$ is compared too.
"""

import dataclasses
import math

import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm
from second_order_cases import run_slab, slab_mesh

import gfem

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


# The guide section below.
LAMBDA_UM, D, LENGTH_UM, WIDTH_UM, HEIGHT_UM = 1.55, 20e-12, 1.0, 0.25, 0.25
F0 = const.c / (LAMBDA_UM * 1e-6)
N0 = 2.2
K0 = 2 * math.pi / (LAMBDA_UM * 1e-6)
LENGTH = LENGTH_UM * 1e-6
#: Cells per um along the guide, at order 3: about 20 cells per pump wavelength.
CELLS_PER_UM, ORDER = 30, 3
#: $|b|$ against the overlap model: measured 8e-6 to 1.8e-4 over the cases at 30 cells
#: per um, 1.1e-4 to 2.7e-3 at 15 and 3.9e-5 to 8.8e-4 at 20: the error falls as
#: the 4th power of the cell size (the guide's numerical dispersion, on the phase of
#: the pump and of the generated wave). The gate is 5.6 times the 30-cell maximum.
REL_MODULUS = 1e-3
#: The phase of $b$ in rad, beyond the line's own discretization error $\delta\varphi$
#: (below): measured at most 0.5 $\delta\varphi$ (3.9e-4 against 7.8e-4 at worst).
PHASE_FLOOR = 2e-3


def guide_mesh(path, cells_per_um):
    """The crystal guide along $z$, a single cell across; ports ``p1``, ``p2``."""
    n = max(1, round(cells_per_um * LENGTH_UM))
    edges = (
        np.linspace(0, WIDTH_UM, 2),
        np.linspace(0, HEIGHT_UM, 2),
        np.linspace(0, LENGTH_UM, n + 1),
    )
    tol = 1e-9

    def surface(c):
        if abs(c[2]) < tol:
            return "p1"
        if abs(c[2] - LENGTH_UM) < tol:
            return "p2"
        if abs(c[1]) < tol or abs(c[1] - HEIGHT_UM) < tol:
            return "plates"
        return ""

    mesh = sm.structured(edges, lambda _: "crystal", surface)
    return gfem.Mesh.from_gmsh(sm.write_msh22(path, mesh), length_unit="um")


def guide_model(mesh, n, chi2):
    d = np.zeros((3, 6))
    d[1, 1] = D
    x, y = 0.5 * WIDTH_UM * 1e-6, 0.5 * HEIGHT_UM * 1e-6
    return gfem.Model(
        mesh,
        materials={
            "crystal": gfem.Material(
                eps_r=n**2, second_order_nonlinear_tensor=d if chi2 else None
            )
        },
        boundaries=[gfem.PEC("plates")],
        ports=[
            gfem.WavePort("p1", "p1", polarity=((x, y, 0.0), "+y")),
            gfem.WavePort("p2", "p2", polarity=((x, y, LENGTH), "+y")),
        ],
    )


@dataclasses.dataclass(frozen=True)
class Case:
    """Pump frequencies (units of ``F0``), the process and the generated index."""

    pumps: tuple[float, ...]
    process: str | None
    n_generated: float

    @property
    def sigma(self):
        return -1 if self.process == "difference-frequency" else 1

    @property
    def generated(self):
        a, b = self.pumps[0], self.pumps[-1]
        return a + b if self.process != "difference-frequency" else a - b


def _index_for_pi(pumps, process, generated):
    r"""The generated index that gives $\Delta\beta L = \pi$ for pump index ``N0``."""
    sigma = -1 if process == "difference-frequency" else 1
    a, b = pumps[0], pumps[-1]
    beta_pumps = N0 * K0 * (a + sigma * b)
    return (beta_pumps + math.pi / LENGTH) / (generated * K0)


SHG, SFG, DFG = (1.0,), (1.0, 1.5), (2.5, 1.5)
CASES = {
    "shg-matched": Case(SHG, None, N0),
    "shg-pi": Case(SHG, None, _index_for_pi(SHG, None, 2.0)),
    "sfg-matched": Case(SFG, "sum-frequency", N0),
    "sfg-pi": Case(SFG, "sum-frequency", _index_for_pi(SFG, "sum-frequency", 2.5)),
    "dfg-matched": Case(DFG, "difference-frequency", N0),
}


@dataclasses.dataclass
class Run:
    case: Case
    b3: complex
    reference: complex
    #: The phase the solved line's own dispersion adds to the three waves, rad.
    line_phase: float
    delta_beta_length: float


def _line_phase(result, beta, index=0):
    r"""The phase the solved line adds beyond $e^{-i\beta L}$ between the ports."""
    s21 = np.asarray(result.s.values)[index, 1, 0]
    return abs(float(np.angle(s21 * np.exp(1j * beta * LENGTH))))


def run_case(mesh, case, order=ORDER):
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    pump_f = [f * F0 for f in case.pumps]
    pump_model = guide_model(mesh, N0, chi2=True)
    pump = gfem.solve(
        pump_model,
        gfem.Driven(pump_f, excite="p1", keep_fields=True),
        options=options,
    )
    pumps = [gfem.Pump(pump, f, incident_power=1.0) for f in pump_f]
    f3 = case.generated * F0
    source = gfem.second_order_polarization(
        pump_model,
        pumps[0] if case.process is None else tuple(pumps),
        process=case.process,
    )
    gen_model = guide_model(mesh, case.n_generated, chi2=False)
    generated = gfem.solve(
        gen_model,
        gfem.Driven(source.frequency, excite=(), sources=source),
        options=options,
    )
    b3 = complex(generated.port_wave("p2", source.name)[0])

    # the modes of port 1, the generated one's first, then the pump's
    harmonic = gfem.solve(gen_model, gfem.PortMode("p1", [f3]), options=options)
    modes = pump.port_modes["p1"]
    kappa = np.asarray(gfem.three_wave_coupling(pump_model, [harmonic, modes]).kappa)
    beta3 = float(harmonic.gamma[0].imag)
    betas = [float(g.imag) for g in np.asarray(modes.gamma)]
    pump_phase = sum(_line_phase(pump, b, i) for i, b in enumerate(betas))
    if case.process is None:
        kap, g, beta_a, beta_b = kappa[0, 1, 1], 1.0, betas[0], betas[0]
        pump_phase *= 2
    else:
        kap, g = kappa[0, 1, 2], 2.0
        beta_a, beta_b = betas
    delta = beta3 - beta_a - case.sigma * beta_b
    mean = (beta3 + beta_a + case.sigma * beta_b) / 2
    ref = (
        -1j
        * g
        * np.conj(kap)
        * LENGTH
        * np.sinc(delta * LENGTH / (2 * math.pi))
        * np.exp(-1j * mean * LENGTH)
    )
    # the solved lines' phase over L against the port modes' beta: the two pump
    # lines (the same line twice for second harmonic) and the generated one
    line = gfem.solve(gen_model, gfem.Driven(f3, excite="p1"), options=options)
    phase = _line_phase(line, beta3) + pump_phase
    return Run(case, b3, complex(ref), phase, float(delta * LENGTH))


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    return guide_mesh(tmp_path_factory.mktemp("guide") / "guide.msh", CELLS_PER_UM)


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def guide_run(request, guide):
    return run_case(guide, CASES[request.param])


def test_modulus_against_the_overlap_model(guide_run):
    assert abs(guide_run.b3) == pytest.approx(abs(guide_run.reference), rel=REL_MODULUS)


def test_phase_against_the_overlap_model(guide_run):
    error = abs(float(np.angle(guide_run.b3 / guide_run.reference)))
    assert error <= PHASE_FLOOR + guide_run.line_phase


def test_the_mismatched_cases_are_at_pi(guide_run):
    expected = math.pi if guide_run.case.n_generated != N0 else 0.0
    assert abs(guide_run.delta_beta_length) == pytest.approx(expected, abs=1e-6)
