"""The free-carrier figures of a port mode: regions, refusals and consistency."""

import dataclasses

import jax
import numpy as np
import pml_cases
import pytest
import scipy.constants as sc

import gfem

ORDER = gfem.SolverOptions(order=2)
TPA = 8e-12
RHO = 1.27


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    # a coarse strip with a wide cladding and a layer (3 regions: core, clad, pml)
    path = pml_cases.strip_guide(
        tmp_path_factory.mktemp("strip") / "strip.msh",
        0.8,
        quarter=False,
        length=0.15,
        core_h=0.2,
        clad_h=0.3,
        thickness=0.5,
    )
    model = pml_cases.strip_model(path)
    mode = gfem.solve(model, gfem.PortMode("p1", pml_cases.STRIP_F), options=ORDER)
    return model, mode


def _with(model, **changes):
    materials = dict(model.materials)
    for region, update in changes.items():
        materials[region] = dataclasses.replace(materials[region], **update)
    return model.replace(materials=materials)


def _kerr_loss(model, mode, fc):
    hw = sc.hbar * 2 * np.pi * fc.frequency
    return float(fc.generation[0] * hw[0] * fc.area), float(
        np.imag(gfem.kerr_nonlinearity(model, mode).gamma_nl[0])
    )


def test_the_default_regions_are_those_with_an_absorption(strip):
    model, mode = strip
    model = _with(model, core={"two_photon_absorption": TPA})
    fc = gfem.free_carriers(model, mode)
    assert fc.regions == ("core",)
    assert fc.overlap.shape == fc.generation.shape == (1,)
    assert float(fc.overlap[0]) > 0
    assert fc.area > 0
    assert "FreeCarriers" in repr(fc)
    # the layer's triangles are skipped: a property on the layer changes nothing
    on_layer = _with(model, pml={"two_photon_absorption": 1e-11})
    assert gfem.free_carriers(on_layer, mode).regions == ("core",)


def test_the_generation_is_linear_in_the_absorption(strip):
    model, mode = strip
    one = gfem.free_carriers(_with(model, core={"two_photon_absorption": TPA}), mode)
    two = gfem.free_carriers(
        _with(model, core={"two_photon_absorption": 2 * TPA}), mode
    )
    # the same quadrature, a factor 2 on beta: exact up to one rounding
    assert float(two.generation[0]) == pytest.approx(
        2 * float(one.generation[0]), rel=1e-14
    )
    assert float(two.overlap[0]) == float(one.overlap[0])


def test_the_generation_is_the_kerr_loss_of_the_region(strip):
    model, mode = strip
    # R holds every absorbing element: theta hbar omega A_R = Im gamma_NL, the same
    # integrand summed in another order (relative round-off of ~1e3 terms, 1e-12)
    model = _with(model, core={"two_photon_absorption": TPA})
    lhs, rhs = _kerr_loss(model, mode, gfem.free_carriers(model, mode))
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_the_consistency_holds_with_a_cubic_anisotropy(strip):
    model, mode = strip
    model = _with(
        model,
        core={"two_photon_absorption": TPA, "nonlinear_anisotropy": RHO},
    )
    model = _with(model, core={"nonlinear_index": 6e-18})
    fc = gfem.free_carriers(model, mode)
    lhs, rhs = _kerr_loss(model, mode, fc)
    assert lhs == pytest.approx(rhs, rel=1e-12)
    # rho changes the figure: the isotropic one differs
    iso = _with(model, core={"nonlinear_anisotropy": None})
    assert float(gfem.free_carriers(iso, mode).generation[0]) != pytest.approx(
        float(fc.generation[0]), rel=1e-3
    )


def test_the_consistency_holds_with_a_birefringent_cladding(strip):
    model, mode = strip
    model = _with(
        model,
        core={"two_photon_absorption": TPA, "nonlinear_index": 6e-18},
        clad={"eps_r": (2.1, 2.1, 2.3), "nonlinear_index": 2.6e-20},
    )
    fc = gfem.free_carriers(model, mode)
    lhs, rhs = _kerr_loss(model, mode, fc)
    assert lhs == pytest.approx(rhs, rel=1e-12)
    both = gfem.free_carriers(model, mode, regions=["core", "clad"])
    assert float(both.overlap[0]) > float(fc.overlap[0])


def test_the_overlap_of_a_region_without_absorption_needs_one_beside_it(strip):
    model, mode = strip
    model = _with(model, core={"two_photon_absorption": TPA})
    both = gfem.free_carriers(model, mode, regions=("core", "clad"))
    core = gfem.free_carriers(model, mode, regions="core")
    assert both.regions == ("core", "clad")
    assert both.area > core.area
    assert float(both.overlap[0]) > float(core.overlap[0])


def test_the_refusals(strip):
    model, mode = strip
    absorbing = _with(model, core={"two_photon_absorption": TPA})
    with pytest.raises(ValueError, match=r"no region.*two_photon_absorption"):
        gfem.free_carriers(model, mode)
    with pytest.raises(ValueError, match=r"'nope' is not in wave port 'p1'"):
        gfem.free_carriers(absorbing, mode, regions=["core", "nope"])
    with pytest.raises(ValueError, match=r"'pml'.*wholly in a perfectly matched"):
        gfem.free_carriers(
            _with(absorbing, pml={"two_photon_absorption": 1e-11}), mode, regions="pml"
        )
    with pytest.raises(ValueError, match=r"\['clad'\].*no two_photon_absorption"):
        gfem.free_carriers(absorbing, mode, regions="clad")
    with pytest.raises(ValueError, match=r"repeat"):
        gfem.free_carriers(absorbing, mode, regions=["core", "core"])
    with pytest.raises(TypeError, match=r"regions"):
        gfem.free_carriers(absorbing, mode, regions=[1])
    with pytest.raises(TypeError, match=r"PortModeResult"):
        gfem.free_carriers(absorbing, "mode")
    # an absorption of zero is no source either
    zero = _with(model, core={"two_photon_absorption": 0.0})
    with pytest.raises(ValueError, match=r"zero"):
        gfem.free_carriers(zero, mode)


def test_the_figures_differentiate_through_the_absorption(strip):
    model, mode = strip
    core = _with(model, core={"two_photon_absorption": TPA}).materials["core"]
    field = gfem.MaterialField.uniform("core", core)

    def theta(beta):
        traced = dataclasses.replace(field, two_photon_absorption=beta)
        m = model.replace(materials={**model.materials, "core": traced})
        return gfem.free_carriers(m, mode).generation[0]

    # linear in beta: the derivative is the value over beta, to round-off
    g = float(jax.grad(theta)(jax.numpy.asarray(TPA)))
    assert g == pytest.approx(float(theta(jax.numpy.asarray(TPA))) / TPA, rel=1e-12)
