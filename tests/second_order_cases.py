r"""A thin $\chi^{(2)}$ slab on a periodic cell, the second-order source's reference.

A cell of side $A$ in a homogeneous medium, periodic along $x$ and $y$, with Floquet
ports on $z_\mathrm{min}$ and $z_\mathrm{max}$ and a slab of length $L$ between pads.
The pump model is $n_1^2$ everywhere and the generated model $n_2^2$ everywhere (no
reflection at the slab), and a tensor $d_{xxx} = d$ alone acts on the TM pump at normal
incidence. Under $e^{+i\omega t}$ the pump is $E_x = E_0 e^{-ik_1(z - z_0)}$ and the
undepleted generated waves (forward at the top port, backward at the bottom one) are

.. math::

    E_\mathrm{top} = -i\frac{k_0}{n_2}\,d\,E_0^2 L\,\mathrm{sinc}(\Delta k L/2)\,
    e^{i\Delta k L/2}\,e^{-ik_2(L + z_0)},\qquad
    E_\mathrm{bot} = -i\frac{k_0}{n_2}\,d\,E_0^2 L\,\mathrm{sinc}(\Sigma L/2)\,
    e^{-i\Sigma L/2}\,e^{-ik_2 z_0},

with $\Delta k = k_2 - 2k_1$, $\Sigma = k_2 + 2k_1$, $k_2 = 2 n_2 k_0$ and $z_0$ the
slab's lower face; a port wave is $b = E\sqrt{A^2 n_2 / (2\eta_0)}$.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.constants as const
import structured_meshes as sm

import gfem

#: Vacuum wavelength (um), tensor $d_{xxx}$ (m/V), slab length, cell side, pad (um).
LAMBDA, D, SLAB, SIDE, PAD = 1.55, 20e-12, 1.0, 0.25, 0.25
#: Indices: the pump's, and the generated wave's when matched or at $\Delta k L = \pi$.
N1, N2_MATCHED, N2_MISMATCHED = 2.2, 2.2, 2.5875
FREQUENCY = const.c / (LAMBDA * 1e-6)
ETA0 = const.physical_constants["characteristic impedance of vacuum"][0]
#: Periodic boundaries in the order the cell is paired.
PERIODIC = [gfem.Periodic(("x_min", "x_max")), gfem.Periodic(("y_min", "y_max"))]


def sinc(u: float) -> float:
    return 1.0 if u == 0 else math.sin(u) / u


def slab_mesh(
    path: Path, cells_per_um: float, *, lateral: int = 2, unit_um: float = 1.0
) -> gfem.Mesh:
    """The cell, ``lateral`` cells per side, with the slab as region ``slab``.

    The file's length unit is ``unit_um`` micrometres (1 for um, 1e-6 for m).
    """

    def n(length: float) -> int:
        return max(1, round(cells_per_um * length))

    z = unit_um * np.concatenate(
        [
            np.linspace(0, PAD, n(PAD) + 1),
            np.linspace(PAD, PAD + SLAB, n(SLAB) + 1)[1:],
            np.linspace(PAD + SLAB, 2 * PAD + SLAB, n(PAD) + 1)[1:],
        ]
    )
    lat = unit_um * np.linspace(0, SIDE, lateral + 1)
    cell = sm.periodic_cell(
        (lat, lat, z),
        lambda c: "slab" if PAD < c[2] / unit_um < PAD + SLAB else "pad",
        paired="xy",
    )
    return gfem.Mesh.from_gmsh(
        sm.write_msh22(path, cell), length_unit="um" if unit_um == 1.0 else "m"
    )


def slab_model(mesh: gfem.Mesh, n: float, *, chi2: bool) -> gfem.Model:
    """The cell at index ``n``, with the tensor in the slab when ``chi2``."""
    d = np.zeros((3, 6))
    d[0, 0] = D
    ports = [
        gfem.FloquetPort(f"{s} {p}", s, polarization=p)
        for s in ("z_min", "z_max")
        for p in ("TE", "TM")
    ]
    return gfem.Model(
        mesh,
        materials={
            "slab": gfem.Material(
                eps_r=n**2, second_order_nonlinear_tensor=d if chi2 else None
            ),
            "pad": gfem.Material(eps_r=n**2),
        },
        boundaries=PERIODIC,
        ports=ports,
    )


@dataclasses.dataclass
class SlabCase:
    """One slab solve pair: the pump, the source and the generated result."""

    pump: Any
    source: Any
    generated: Any
    n2: float

    def waves(self) -> tuple[complex, complex, float]:
        """``(b_top, b_bottom, TE cross-polarized |b|)`` of the TM generated waves."""
        name = self.source.name
        top = complex(self.generated.port_wave("z_max TM", name)[0])
        bottom = complex(self.generated.port_wave("z_min TM", name)[0])
        cross = max(
            abs(complex(self.generated.port_wave(f"{s} TE", name)[0]))
            for s in ("z_min", "z_max")
        )
        return top, bottom, cross

    def pump_amplitude(self) -> complex:
        r"""$E_0$ at the slab's lower face, fitted to the solved pump field mid-slab."""
        field = self.pump.e_field(FREQUENCY, "z_min TM")
        k1 = N1 * 2 * math.pi / (LAMBDA * 1e-6)
        z = np.linspace(PAD + 0.4 * SLAB, PAD + 0.6 * SLAB, 33) * 1e-6
        at = np.stack([np.full_like(z, SIDE * 0.5e-6)] * 2 + [z], axis=1)
        ex = np.asarray(field.at(at))[:, 0]
        return complex(np.mean(ex * np.exp(1j * k1 * (z - PAD * 1e-6))))

    def closed_form(self) -> tuple[complex, complex]:
        """The closed form's ``(b_top, b_bottom)`` from the fitted pump amplitude."""
        k0 = 2 * math.pi / (LAMBDA * 1e-6)
        k1, k2 = N1 * k0, 2 * self.n2 * k0
        length, z0 = SLAB * 1e-6, PAD * 1e-6
        dk, sigma = k2 - 2 * k1, k2 + 2 * k1
        e0 = self.pump_amplitude()
        scale = -1j * (k0 / self.n2) * D * e0**2 * length
        top = scale * sinc(dk * length / 2) * np.exp(1j * dk * length / 2)
        top *= np.exp(-1j * k2 * (length + z0))
        bottom = scale * sinc(sigma * length / 2) * np.exp(-1j * sigma * length / 2)
        bottom *= np.exp(-1j * k2 * z0)
        factor = math.sqrt((SIDE * 1e-6) ** 2 * self.n2 / (2 * ETA0))
        return complex(top * factor), complex(bottom * factor)


def run_slab(
    mesh: gfem.Mesh,
    *,
    order: int,
    mismatched: bool = False,
    generated_order: int | None = None,
) -> SlabCase:
    """Pump at 1 W per cell, then the generated solve at twice the frequency."""
    n2 = N2_MISMATCHED if mismatched else N2_MATCHED
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    pump_model = slab_model(mesh, N1, chi2=True)
    pump = gfem.solve(
        pump_model,
        gfem.Driven(FREQUENCY, excite="z_min TM", keep_fields=True),
        options=options,
    )
    source = gfem.second_order_polarization(
        pump_model,
        gfem.Pump(pump, FREQUENCY, incident_power=1.0),
        generated_order=generated_order,
    )
    generated = gfem.solve(
        slab_model(mesh, n2, chi2=False),
        gfem.Driven(source.frequency, excite=(), sources=source),
        options=options,
    )
    return SlabCase(pump, source, generated, n2)
