r"""The mechanical eigenmodes of a 3D body: closed-form boxes, identities, refusals.

The bodies are boxes of 1 x 0.8 x 0.6 um on a structured Kuhn grid
(`mechanical_cases.box`), P2 elements, 5 cells per um (1.6 k DOFs) and 10 cells
per um (12 k DOFs). The references are the closed forms of a box with sliding
walls (`mechanical_cases.box_frequencies`) and of a plate cut to a box whose
z faces are free (`mechanical_cases.lamb_box_frequencies`). Each gate is at
least twice the error it bounds, the number in its comment, measured through
`gfem.solve` on these meshes.
"""

import mechanical_cases as mc
import numpy as np
import pml_cases
import pytest

import gfem
from gfem._engine.mechanical_eigen import MechanicalEigenmodeRule

SIZE = (1.0, 0.8, 0.6)
ISOTROPIC = (mc.LAMBDA + 2 * mc.MU, mc.LAMBDA, mc.MU)


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    """The box at 5 and 10 cells per um."""
    path = tmp_path_factory.mktemp("box")
    return {c: mc.box(path / f"box{c}.msh", SIZE, c) for c in (5, 10)}


def _model(mesh, material=None):
    return gfem.Model(mesh, materials={"body": material or mc.material()})


def _errors(result, exact):
    """Per-mode relative errors against the closed form nearest each mode."""
    exact = np.asarray(exact)
    found = np.sort(np.asarray(result.frequency))
    return np.abs(found - exact) / exact


def _solve(mesh, target, n_modes, clamped, material=None, **problem):
    model = _model(mesh, material)
    return gfem.solve(
        model,
        gfem.MechanicalEigenmode(target, n_modes=n_modes, clamped=clamped, **problem),
    )


@pytest.fixture(scope="module")
def isotropic(meshes):
    """The five lowest modes of the roller box, on both meshes."""
    return {c: _solve(meshes[c], 3.2e9, 5, mc.ROLLERS) for c in meshes}


# --- closed forms --------------------------------------------------------------------


def test_the_isotropic_roller_box_converges_to_its_standing_waves(isotropic):
    # the five lowest modes are simple (the (1,1,1) pair, the sixth and seventh,
    # is not asked for); measured max error 4.9e-3 (5 cells/um) and 3.8e-4
    # (10 cells/um), gates 2x each; the mesh halved gains at least 10x per mode
    # (measured 13)
    exact = mc.box_frequencies(SIZE, ISOTROPIC, mc.RHO, 5)
    coarse, fine = (_errors(isotropic[c], exact) for c in (5, 10))
    assert coarse.max() < 1e-2
    assert fine.max() < 8e-4
    assert (coarse / fine >= 10).all()


def test_the_cubic_silicon_roller_box_converges_to_its_standing_waves(meshes):
    # the same box in silicon (crystal axes along the box's), five modes near
    # 4.5 GHz; measured max error 5.7e-3 (5 cells/um) and 4.5e-4 (10), gates
    # 2x each; the ratio per mode is at least 12.8 (gate 10)
    exact = mc.box_frequencies(SIZE, mc.SI_C, mc.SI_RHO, 5)
    errors = [
        _errors(_solve(meshes[c], 4.5e9, 5, mc.ROLLERS, mc.SILICON), exact)
        for c in (5, 10)
    ]
    assert errors[0].max() < 1.2e-2
    assert errors[1].max() < 9e-4
    assert (errors[0] / errors[1] >= 10).all()


def test_the_lamb_box_converges_to_its_plate_modes(meshes):
    # a plate of 0.6 um cut to 1 x 0.8 um, rollers on x and y, z free: the
    # flexural modes A(1,0), A(0,1) and A(1,1) (the next mode is 1.07 GHz away
    # from 1.6 GHz); measured max error 5.7e-3 (5 cells/um; 2.0e-3, 3.2e-3,
    # 5.7e-3) and 4.5e-4 (10 cells/um; 1.5e-4, 2.4e-4, 4.5e-4), gates 2x each;
    # the ratio per mode is at least 12.7 (gate 10)
    exact = mc.lamb_box_frequencies(SIZE, 3)
    errors = [
        _errors(_solve(meshes[c], 1.6e9, 3, mc.LAMB_ROLLERS), exact) for c in (5, 10)
    ]
    assert errors[0].max() < 1.2e-2
    assert errors[1].max() < 1e-3
    assert (errors[0] / errors[1] >= 10).all()


# --- identities ----------------------------------------------------------------------


def test_a_uniform_quality_factor_scales_the_frequency_exactly(meshes, isotropic):
    # the stiffness is (1 + i/Q) K on the same mesh, so omega scales by
    # sqrt(1 + i/Q); both solves are held to the eigen tolerance (1e-10
    # backward error on a normal pencil: the eigenvalue is as accurate)
    quality = 40.0
    lossy = _solve(meshes[5], 3.2e9, 5, mc.ROLLERS, mc.material(quality=quality))
    lossless = isotropic[5]
    ratio = np.asarray(lossy.omega) / np.asarray(lossless.omega)
    assert np.abs(ratio - np.sqrt(1 + 1j / quality)).max() < 1e-9
    # a lossless mode is real, so its Q is infinite
    assert np.abs(np.imag(lossless.omega)).max() == 0.0
    assert np.isinf(np.asarray(lossless.q)).all()


def test_a_mode_carries_one_joule_with_a_real_largest_coefficient(meshes, isotropic):
    result = isotropic[5]
    for mode in range(2):
        u = result.displacement(mode)
        coefficients = np.concatenate([np.asarray(u[a].coefficients) for a in "xyz"])
        largest = coefficients[np.argmax(np.abs(coefficients))]
        # real to round-off (a symmetric mode has several equal peaks)
        assert abs(largest.imag) <= 1e-14 * abs(largest)
        assert largest.real > 0
        omega = float(np.real(result.omega[mode]))
        energy = 0.5 * omega**2 * mc.mass_energy(_model(meshes[5]), u)
        # the eigen tolerance 1e-10 on a vector scaled in floating point
        assert energy == pytest.approx(1.0, rel=1e-10)


def test_a_body_has_no_axial_wavenumber_or_group_velocity(isotropic):
    result = isotropic[5]
    assert result.wavenumber == 0.0
    assert result.displacement(0).wavenumber == 0.0
    velocity = np.asarray(result.group_velocity)
    assert velocity.shape == (5,)
    assert np.isnan(velocity).all()


def test_the_error_is_the_backward_error_of_the_returned_modes(isotropic):
    tolerance = gfem.SolverOptions().eigen_tolerance
    assert 0 < np.asarray(isotropic[5].error).max() <= tolerance


# --- refusals ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def body(meshes):
    return _model(meshes[5])


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (gfem.SolverOptions(linear_solver="bddc_gmres"), "bddc_gmres"),
        (gfem.SolverOptions(linear_solver="pmg_gmres"), "pmg_gmres"),
        (gfem.SolverOptions(device="cuda"), "device='cuda'"),
        (gfem.SolverOptions(linear_solver="petsc"), "no PETSc path"),
    ],
)
def test_the_numerics_a_body_does_not_have_refuse_by_name(body, options, match):
    problem = gfem.MechanicalEigenmode(3.2e9, clamped=mc.ROLLERS)
    with pytest.raises(NotImplementedError, match=match):
        gfem.solve(body, problem, options=options)


def test_a_body_refuses_a_communicator_in_its_rule(body):
    # gfem.solve resolves a communicator first (and refuses a stand-in as a
    # type error), so the rule is built directly with what resolve returns
    problem = gfem.MechanicalEigenmode(3.2e9, clamped=mc.ROLLERS)
    with pytest.raises(NotImplementedError, match="several processes"):
        MechanicalEigenmodeRule(body, problem, gfem.SolverOptions(), comm=object())


def test_a_wavenumber_on_a_body_refuses(body):
    problem = gfem.MechanicalEigenmode(3.2e9, wavenumber=1e6)
    with pytest.raises(ValueError, match="a 3D body has no axial wavenumber"):
        gfem.solve(body, problem)


def test_a_clamped_name_that_is_not_a_surface_says_surface(body):
    with pytest.raises(ValueError, match="clamped 'body' must be a surface bordering"):
        gfem.solve(body, gfem.MechanicalEigenmode(3.2e9, clamped=["body"]))


def test_a_model_layer_over_the_solid_refuses(tmp_path):
    mesh = gfem.Mesh.from_gmsh(pml_cases.box(tmp_path / "layer.msh", ("+x",)))
    model = gfem.Model(
        mesh,
        materials={"air": mc.material(), "pml": mc.material()},
        pml=gfem.PML("pml"),
    )
    with pytest.raises(ValueError, match="does not read the model's electromagnetic"):
        gfem.solve(model, gfem.MechanicalEigenmode(3.2e9))
