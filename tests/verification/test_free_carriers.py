r"""The free-carrier figures of a port mode converge to closed forms.

The 220 nm silicon slab of the Kerr verification (Ez line port, perfectly
conducting ends 3 um from the core) has closed forms from the dispersion
relation at 30 digits (``mpmath``): the overlaps
:math:`\Gamma_R = \frac{c\varepsilon_0}{2} n_R \int_R |E|^2` of core and
cladding, and from the core's fourth-power integral the generation
:math:`\theta`. The filled WR-90 gives :math:`\Gamma = n/n_\mathrm{eff}` of the
computed index, and the identities tie the figures to the solve's group index
and to `gfem.kerr_nonlinearity`.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest
import scipy.constants as sc
from slab2d import write_slab

import gfem

DATA = Path(__file__).parents[1] / "unit" / "data"
N_SI, N_OX = 3.476, 1.444
LAMBDA = 1.55e-6
TPA = 8e-12
HBAR = sc.hbar
#: closed forms of the slab (mpmath, 30 digits)
GAMMA_CORE, GAMMA_CLAD = 0.989022611351953, 0.0962014455707568
IM_GAMMA_CORE, THETA_CORE = 1.88228260955881e-5, 6.67601209856558e20
CORE_THICKNESS = 0.22e-6
#: filled WR-90 at 8 GHz, TE10
N_FILL, F_GUIDE = 1.5, 8e9
GUIDE_GAMMA, GUIDE_THETA = 1.19402151596215, 2.99150457728006e19


@pytest.fixture(scope="module")
def slab_model(tmp_path_factory):
    path = write_slab(
        tmp_path_factory.mktemp("slab") / "slab.msh",
        thick=0.22,
        clear=3.0,
        h_core=0.0275,
        h_clad=0.05,
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={
            "core": gfem.Material(eps_r=N_SI**2, two_photon_absorption=TPA),
            # the cladding carries a (tiny) absorption so that it can be a
            # carrier region on its own: the overlaps do not depend on it
            "clad": gfem.Material(eps_r=N_OX**2, two_photon_absorption=1e-14),
        },
        ports=[gfem.WavePort("p1", "port1", pec_edges=("outer",))],
    )


@pytest.fixture(scope="module")
def slab_modes(slab_model):
    f = sc.c / LAMBDA
    return {
        p: gfem.solve(
            slab_model,
            gfem.PortMode("p1", f, polarization="Ez"),
            options=gfem.SolverOptions(order=p),
        )
        for p in (2, 3, 4)
    }


@pytest.fixture(scope="module")
def guide_model():
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    material = gfem.Material(eps_r=N_FILL**2, two_photon_absorption=TPA)
    return gfem.Model(
        mesh,
        materials={"air": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def _overlap(model, mode, regions):
    return float(gfem.free_carriers(model, mode, regions=regions).overlap[0])


def test_the_slab_overlaps_converge_to_their_closed_forms(slab_model, slab_modes):
    # measured relative errors 3.1e-5, 8.2e-8, 1.1e-10 at orders 2-4 (the
    # field's mesh error): each order 5x below the last, the last under 1e-9
    for region, exact in (("core", GAMMA_CORE), ("clad", GAMMA_CLAD)):
        errors = [
            abs(_overlap(slab_model, m, region) - exact) / exact
            for m in slab_modes.values()
        ]
        assert all(b < a / 5 for a, b in itertools.pairwise(errors)), errors
        assert errors[-1] < 1e-9, errors


def test_the_slab_generation_matches_its_closed_form(slab_model, slab_modes):
    # theta is the Kerr integral's imaginary part over hbar omega A on the
    # same quadrature; its field error is the fourth-power one (quadratic
    # overlap error doubled), well under the 1e-6 gate at order 4
    mode = slab_modes[4]
    fc = gfem.free_carriers(slab_model, mode, regions="core")
    assert fc.regions == ("core",)
    assert fc.area == pytest.approx(CORE_THICKNESS, rel=1e-12, abs=0)
    assert float(fc.generation[0]) == pytest.approx(THETA_CORE, rel=1e-6)


def test_the_overlaps_weighted_by_index_sum_to_the_group_index(slab_model, slab_modes):
    # sum_R n_R Gamma_R = n_g: the same discrete integrals up to round-off
    # (measured 1.2e-12 relative); gate 1e-10
    for mode in slab_modes.values():
        total = N_SI * _overlap(slab_model, mode, "core") + N_OX * _overlap(
            slab_model, mode, "clad"
        )
        assert total == pytest.approx(float(np.real(mode.n_g[0])), rel=1e-10)


def test_the_filled_wr90_overlap_and_generation(guide_model):
    # Gamma = n / n_eff of the computed index at any order (measured 1.2e-12);
    # theta is the closed form of the Kerr integral at the converged order
    for p in (2, 3):
        mode = gfem.solve(
            guide_model,
            gfem.PortMode("p1", [F_GUIDE]),
            options=gfem.SolverOptions(order=p),
        )
        fc = gfem.free_carriers(guide_model, mode)
        n_eff = float(np.real(mode.n_eff[0]))
        assert float(fc.overlap[0]) == pytest.approx(N_FILL / n_eff, rel=1e-10)
        if p == 3:
            assert float(fc.overlap[0]) == pytest.approx(GUIDE_GAMMA, rel=1e-6)
            assert float(fc.generation[0]) == pytest.approx(GUIDE_THETA, rel=1e-6)


def test_generation_times_photon_energy_and_area_is_the_kerr_loss(guide_model):
    # same quadrature and operations up to the order of the sums: round-off
    # of a sum of ~1e4 terms, 1e-12 relative. The pinned value is the closed
    # form theta hbar omega A: the field's mesh error against it is 6.7e-6 at
    # order 2 and 1.2e-7 measured at order 3, under the 1e-6 gate
    mode = gfem.solve(
        guide_model, gfem.PortMode("p1", [F_GUIDE]), options=gfem.SolverOptions(order=3)
    )
    fc = gfem.free_carriers(guide_model, mode)
    im = float(np.imag(gfem.kerr_nonlinearity(guide_model, mode).gamma_nl[0]))
    lhs = float(fc.generation[0]) * HBAR * 2 * np.pi * F_GUIDE * fc.area
    assert lhs == pytest.approx(im, rel=1e-12, abs=0)
    assert im == pytest.approx(3.68303309923265e-8, rel=1e-6, abs=0)
