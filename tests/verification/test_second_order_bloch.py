r"""Second-harmonic generation on a Bloch cell: an oblique TE pump on a uniform slab.

The $\chi^{(2)}$ slab of `second_order_cases` ($d_{yyy} = 20$ pm/V, 1 um) is pumped
by a TE plane wave at $30^\circ$ in the medium of index $n_\omega$, 1 W per cell
(`gfem.Driven(wave_vector=k)` with $k = n_\omega k_0 \sin 30^\circ$ along $x$). The
generated solve is `gfem.Driven(wave_vector=2k)` on a model of index $n_{2\omega}$,
which leaves the slab at $\sin\theta_2 = n_\omega\sin\theta_1 / n_{2\omega}$. The
undepleted closed form is the normal-incidence one with
$k_{1z} = n_\omega k_0\cos\theta_1$, $k_{2z} = 2 n_{2\omega} k_0\cos\theta_2$ for
the propagation constants, and
$k_0/n_2 \to k_0/(n_2\cos\theta_2)$:

.. math::

    E_\mathrm{top} = -i\frac{k_0}{n_2\cos\theta_2}\,d\,E_0^2 L\,
    \mathrm{sinc}(\Delta k L/2)\,e^{i\Delta k L/2}\,e^{-ik_{2z}(L + z_0)},\qquad
    E_\mathrm{bot} = -i\frac{k_0}{n_2\cos\theta_2}\,d\,E_0^2 L\,
    \mathrm{sinc}(\Sigma L/2)\,e^{-i\Sigma L/2}\,e^{-ik_{2z}z_0},

$\Delta k = k_{2z} - 2k_{1z}$, $\Sigma = k_{2z} + 2k_{1z}$, and a port wave is
$b = E\sqrt{A\, n_2\cos\theta_2/(2\eta_0)}$ ($A$ the area of the cell). The pump's
$E_0$ is fitted to the solved field.

Cell-width invariance: a cell of twice the width along $x$ carries another Bloch phase
across its pair ($e^{-2ik_xa}$ against $e^{-ik_xa}$, and twice the unpaired phase of the
generated field). Fed the pump power of equal intensity (2 W per cell), it must give the
same generated wave per $\sqrt A$. A conjugation slip of the load's restriction, or a
generated vector other than $2k$, changes the answer by order one.

Measured at order 3, 20 cells/um along $z$ and 3 elements per side (1620
tetrahedra): the closed form to $7\times10^{-5}$ (top) and $1.6\times10^{-4}$ (bottom)
for $n_{2\omega} = 2.5875$ (with 2 elements per side: $1.0\times10^{-3}$ and
$1.1\times10^{-3}$, the lateral resolution of the oblique phase), the tolerance
$10^{-3}$ at least 6 times that. A cell of twice the width repeats the same mesh, so its
folded problem is the same discrete one: the generated waves per $\sqrt A$ agree to
$2\times10^{-14}$ (round-off of two solves).
"""

import math

import numpy as np
import pytest
from second_order_cases import (
    ETA0,
    FREQUENCY,
    LAMBDA,
    N1,
    N2_MATCHED,
    N2_MISMATCHED,
    PAD,
    PERIODIC,
    SIDE,
    SLAB,
    D,
    sinc,
    slab_mesh,
)

import gfem
from gfem import diffraction as dif

pytestmark = pytest.mark.verification

THETA = math.radians(30.0)
TOP, BOTTOM = "z_max", "z_min"
#: The closed form against the solve: see the module docstring for the measured values.
REL_TOP, REL_BOTTOM = 1e-3, 1e-3
#: Cell-width invariance of the generated waves per $\sqrt A$: the two solves are one
#: discrete problem, so they differ by round-off, measured 2e-14 (a direct solve of
#: condition number below 1e6 is good to 1e-10 on any platform); the gate is far
#: below what a wrong phase gives (order one).
REL_WIDTH = 1e-6
K0 = 2 * math.pi / (LAMBDA * 1e-6)
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")


def _model(mesh, n, *, chi2, ports_at=None):
    """The cell at index ``n`` with the $d_{yyy}$ tensor when ``chi2``."""
    d = np.zeros((3, 6))
    d[1, 1] = D
    model = gfem.Model(
        mesh,
        materials={
            "slab": gfem.Material(
                eps_r=n**2, second_order_nonlinear_tensor=d if chi2 else None
            ),
            "pad": gfem.Material(eps_r=n**2),
        },
        boundaries=PERIODIC,
    )
    k, f = ports_at
    ports = [
        p for s in (BOTTOM, TOP) for p in dif.floquet_ports(model, s, f, wave_vector=k)
    ]
    return model.replace(ports=ports)


class Run:
    """One oblique pump and generated solve on a mesh of width ``width``."""

    def __init__(self, mesh, width, n2):
        self.area = (width * SIDE * 1e-6) * (SIDE * 1e-6)
        self.n2 = n2
        kx = N1 * K0 * math.sin(THETA)
        self.k = (kx, 0.0, 0.0)
        pump_model = _model(mesh, N1, chi2=True, ports_at=(self.k, FREQUENCY))
        # equal intensity at every width: 1 W per unit cell
        self.power = power = float(width)
        self.pump = gfem.solve(
            pump_model,
            gfem.Driven(
                FREQUENCY,
                excite=f"{BOTTOM} (0, 0) TE",
                wave_vector=self.k,
                keep_fields=True,
            ),
            options=OPTIONS,
        )
        self.source = gfem.second_order_polarization(
            pump_model, gfem.Pump(self.pump, FREQUENCY, incident_power=power)
        )
        k_gen = tuple(2 * c for c in self.k)
        self.k_gen = k_gen
        gen_model = _model(mesh, n2, chi2=False, ports_at=(k_gen, 2 * FREQUENCY))
        self.generated = gfem.solve(
            gen_model,
            gfem.Driven(
                self.source.frequency,
                excite=(),
                sources=self.source,
                wave_vector=k_gen,
            ),
            options=OPTIONS,
        )

    def wave(self, surface):
        name = f"{surface} (0, 0) TE"
        return complex(self.generated.port_wave(name, self.source.name)[0])

    def closed_form(self):
        """``(b_top, b_bottom)`` of the closed form, from the fitted pump."""
        n1, n2 = N1, self.n2
        sin2 = n1 * math.sin(THETA) / n2
        cos1, cos2 = math.cos(THETA), math.sqrt(1 - sin2**2)
        k1z, k2z = n1 * K0 * cos1, 2 * n2 * K0 * cos2
        length, z0 = SLAB * 1e-6, PAD * 1e-6
        dk, sigma = k2z - 2 * k1z, k2z + 2 * k1z
        e0 = self._pump_amplitude(k1z)
        scale = -1j * (K0 / (n2 * cos2)) * D * e0**2 * length
        top = scale * sinc(dk * length / 2) * np.exp(1j * dk * length / 2)
        top *= np.exp(-1j * k2z * (length + z0))
        bottom = scale * sinc(sigma * length / 2) * np.exp(-1j * sigma * length / 2)
        bottom *= np.exp(-1j * k2z * z0)
        factor = math.sqrt(self.area * n2 * cos2 / (2 * ETA0))
        return complex(top * factor), complex(bottom * factor)

    def _pump_amplitude(self, k1z):
        r"""$E_0$ of $E_y = E_0 e^{-ik_xx - ik_{1z}(z - z_0)}$, fitted mid-slab."""
        field = self.pump.e_field(FREQUENCY, f"{BOTTOM} (0, 0) TE")
        z = np.linspace(PAD + 0.4 * SLAB, PAD + 0.6 * SLAB, 33) * 1e-6
        x = np.full_like(z, 0.3 * SIDE * 1e-6)
        at = np.stack([x, np.full_like(z, 0.5 * SIDE * 1e-6), z], axis=1)
        ey = np.asarray(field.at(at))[:, 1]
        phase = np.exp(1j * (self.k[0] * x + k1z * (z - PAD * 1e-6)))
        # the kept field is the 1 W column; the pump feeds `power` W
        return math.sqrt(self.power) * complex(np.mean(ey * phase))


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    """The cell of 3 elements per side, and of 2 per side at widths 1 and 2."""
    path = tmp_path_factory.mktemp("bloch")
    return {
        "fine": slab_mesh(path / "fine.msh", 20, lateral=3),
        1: slab_mesh(path / "w1.msh", 20),
        2: slab_mesh(path / "w2.msh", 20, width=2),
    }


@pytest.fixture(scope="module", params=[False, True], ids=["matched", "mismatched"])
def narrow(request, meshes):
    return Run(meshes["fine"], 1, N2_MISMATCHED if request.param else N2_MATCHED)


def test_the_generated_wave_follows_the_pump_vector_doubled(narrow):
    source = narrow.source
    assert source.wave_vector == pytest.approx(narrow.k_gen, rel=1e-12)
    assert len(source.lattice) == 2


def test_oblique_waves_against_the_closed_form(narrow):
    top, bottom = narrow.closed_form()
    assert abs(narrow.wave(TOP) - top) <= REL_TOP * abs(top)
    assert abs(narrow.wave(BOTTOM) - bottom) <= REL_BOTTOM * abs(bottom)


def test_the_generated_power_is_the_sum_of_the_ports(narrow):
    # the cell is lossless and the only propagating order is (0, 0)
    ports = abs(narrow.wave(TOP)) ** 2 + abs(narrow.wave(BOTTOM)) ** 2
    emitted = float(narrow.generated.emitted_power[narrow.source.name][0])
    assert emitted == pytest.approx(ports, rel=3e-3)


def test_a_cell_of_twice_the_width_gives_the_same_wave_per_root_area(meshes):
    mismatched = N2_MISMATCHED
    one = Run(meshes[1], 1, mismatched)
    two = Run(meshes[2], 2, mismatched)
    for surface in (TOP, BOTTOM):
        a = one.wave(surface) / math.sqrt(one.area)
        b = two.wave(surface) / math.sqrt(two.area)
        assert abs(a - b) <= REL_WIDTH * abs(a)


def test_a_wrong_wave_vector_is_refused(meshes, narrow):
    # no wave_vector on a source of the vector 2k: refused, not solved as another array
    gen_model = _model(
        meshes["fine"], narrow.n2, chi2=False, ports_at=(narrow.k_gen, 2 * FREQUENCY)
    )
    with pytest.raises(ValueError, match="Bloch wave vector"):
        gfem.solve(
            gen_model,
            gfem.Driven(narrow.source.frequency, excite=(), sources=narrow.source),
            options=OPTIONS,
        )
