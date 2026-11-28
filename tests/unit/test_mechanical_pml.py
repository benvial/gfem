r"""The elastic perfectly matched layer of a mechanical eigenmode: forward solves.

Leaky columns with a closed form (`elastic_pml_cases`): a layer on a matched
substrate ended by a layer region, whose first root is
:math:`\omega_1 = (c_1/h)[\pi + i\,\mathrm{artanh}\,\zeta]` (Q = 157.07).
"""

import json

import elastic_pml_cases as ec
import numpy as np
import pytest
import thermal_cases as tc

import gfem
from gfem._engine.mechanical_eigen import MechanicalEigenmodeRule, MechanicalSystem

#: The quality factor the complex shift expects: a rounded estimate (an
#: estimate from a damping frame is as rough).
Q_SHIFT = 150.0


def _solve(column, order, *, n_modes=1, **kwargs):
    problem = gfem.MechanicalEigenmode(
        column.target,
        n_modes=n_modes,
        pml=column.pml,
        clamped=column.clamped,
        **kwargs,
    )
    return gfem.solve(column.model, problem, options=gfem.SolverOptions(order=order))


def _error(result, column):
    return abs(complex(result.omega[0]) / column.omega - 1)


@pytest.fixture(scope="module")
def antiplane(tmp_path_factory):
    path = tmp_path_factory.mktemp("antiplane") / "column.msh"
    column = ec.leaky_column(path, 0.1)
    return column, _solve(column, 3)


# --- the closed form ------------------------------------------------------------------


def test_the_antiplane_mode_is_found_with_a_real_shift(antiplane):
    column, result = antiplane
    # measured 2.4e-9 (order 3, h = 0.1 um: the h^6 discretization error of a
    # frequency); the gate is four times that
    assert _error(result, column) < 1e-8
    assert abs(column.quality / 157.0743965521 - 1) < 1e-9


def test_the_radiating_mode_decays_without_a_loss(antiplane):
    column, result = antiplane
    omega = complex(result.omega[0])
    # the leak alone damps it: Im omega > 0, with no mechanical quality factor
    # anywhere; the quality factor is within the discretization error
    assert omega.imag > 0
    assert abs(omega.real / (2 * omega.imag) / column.quality - 1) < 1e-6


def test_a_uniform_quality_factor_scales_the_eigenvalue(tmp_path):
    # with the same Q_m in every region, layer included, K -> (1 + i/Q_m) K and
    # so omega^2 -> (1 + i/Q_m) omega^2 of the lossless-solid problem, in the
    # discrete problem too. Both solves meet eigen_tolerance = 1e-10 on the
    # backward error, which bounds the relative eigenvalue error by about that
    # times the condition number (a few); the gate is 1e-9
    quality = 100.0
    bare = ec.leaky_column(tmp_path / "bare.msh", 0.2)
    lossy = ec.leaky_column(tmp_path / "lossy.msh", 0.2, quality=quality)
    w0 = complex(_solve(bare, 3, target_quality_factor=Q_SHIFT).omega[0])
    w1 = complex(_solve(lossy, 3, target_quality_factor=Q_SHIFT).omega[0])
    assert abs(w1**2 / (w0**2 * (1 + 1j / quality)) - 1) < 1e-9


# --- the stretch-sensitivity filter ---------------------------------------------------


@pytest.mark.parametrize("quality", [None, 100.0])
def test_the_filter_separates_the_modes_of_the_device_from_the_layer(tmp_path, quality):
    column = ec.leaky_column(tmp_path / "column.msh", 0.2, quality=quality)
    problem = gfem.MechanicalEigenmode(
        column.target, pml=column.pml, clamped=column.clamped
    )
    options = gfem.SolverOptions(order=3)
    system = MechanicalSystem(
        column.model,
        problem,
        options,
        MechanicalEigenmodeRule(column.model, problem, options),
    )
    sigma = (2 * np.pi * column.target) ** 2
    system._handle = system._factor(sigma)
    lam, vectors = system._search(system._handle, sigma, 48)
    found = [(k, x, 0.0) for k, x in zip(lam, vectors.T, strict=True)]
    eta = np.array(system.sensitivities(found, sigma, 48))
    system.release()
    # rigid motions are not modes of either
    moving = np.abs(lam) > 1e-8 * abs(sigma)
    eta, lam = eta[moving], lam[moving]
    # the same Q_m everywhere multiplies every eigenvalue by 1 + i / Q_m
    omega = np.sqrt(lam / (1 + 1j / (quality or np.inf)))
    f, q = omega.real / (2 * np.pi), omega.real / (2 * omega.imag)
    # a mode of the device: the discretization error of its field in the layer
    # (measured 4.6e-6 at most for the leaky modes; 8.1e-4 for the trapped mode
    # below); a mode of the layer: about |l| / Re l >= 1 or more (measured 0.97
    # at least). The gates are a decade on each side of the threshold, 0.1
    assert (eta <= 1e-2).sum() >= 4
    assert (eta >= 0.5).sum() >= 10
    assert not ((eta > 1e-2) & (eta < 0.5)).any()
    assert eta[np.argmin(np.abs(lam - sigma))] <= 1e-6
    # the substrate between the rollers carries a shear branch u_z cos(pi x/w)
    # cut off at c_T / 2w = 4.87 GHz. Uniform in y, that mode is exact, real and
    # does not see the stretch (both forms take the same factor s_y): it is
    # kept. Just above it the layer's modes of the branch, lambda - lambda_c
    # proportional to 1/l^2 with the complex depth l = a + d (1 - i sigma_max/3),
    # move with the stretch by |l| / Re l of their decay rate (the first, Q
    # about 490, by 3e-3 of |lambda|): dropped. Measured 1.48 against 1.51
    # (the film is not a rigid end of the substrate); the gate is 5 %
    cutoff = ec.C_T / (2 * 0.6 * ec.UM)
    trapped = np.argmin(np.abs(f - cutoff))
    assert abs(f[trapped] / cutoff - 1) < 1e-4
    assert eta[trapped] <= 1e-2
    branch = np.flatnonzero((np.abs(f / cutoff - 1) < 1e-3) & (q > 100) & (q < 1e4))
    assert len(branch) == 1
    a, d = ec._depths()
    k_ref = 2 * np.pi * column.target / ec.C_L
    smax = 3 * np.log(1 / ec.R0) / (2 * k_ref * d * ec.UM)
    depth = a + d * (1 - 1j * smax / 3)
    assert abs(eta[branch[0]] / (abs(depth) / depth.real) - 1) < 0.05


def test_the_convergence_message_counts_the_dropped_layer_modes(tmp_path):
    column = ec.leaky_column(tmp_path / "column.msh", 0.2)
    with pytest.raises(
        gfem.EigenmodeConvergenceError,
        match=r"\d+ of 60 modes near .*\((\d+) converged modes were modes of the "
        r"layer and were dropped\)",
    ) as caught:
        _solve(column, 2, n_modes=60)
    dropped = int(caught.value.args[0].split("(")[1].split(" converged")[0])
    assert dropped > 0


# --- side and corner layers -----------------------------------------------------------


def test_side_and_corner_layers_leave_a_uniform_mode_exact(tmp_path):
    # the mode does not depend on x, so the layers on the sides (and in the
    # corners, stretched along x and y) multiply both forms by the same
    # factor and the discrete problem keeps its closed form: the error is the
    # one of the column (order 4, h = 0.2 um: measured 1.7e-10, the gate is
    # the 1e-8 of the column's h = 0.1 um order 3 result)
    column = ec.leaky_column(tmp_path / "sides.msh", 0.2, side_layers=True)
    assert len(column.pml.regions) == 7
    result = _solve(column, 4, target_quality_factor=Q_SHIFT)
    assert _error(result, column) < 1e-8


# --- a body ---------------------------------------------------------------------------


def test_a_column_stretched_along_z(tmp_path):
    column = ec.leaky_column(tmp_path / "body.msh", 0.2, dim=3)
    result = _solve(column, 2)
    # a thickness-shear mode u_x(z): the Kuhn grid at order 2, h = 0.2 um, has
    # the 2D mode's discretization error with a fixed stencil, measured 9.2e-5
    # (a netgen mesh of the same size: 2.0e-4); the gate is about twice
    assert _error(result, column) < 2e-4
    assert complex(result.omega[0]).imag > 0


# --- the job -------------------------------------------------------------------------


def test_a_job_round_trips_the_layer_and_the_shift(tmp_path):
    column = ec.leaky_column(tmp_path / "column.msh", 0.2)
    problem = gfem.MechanicalEigenmode(
        column.target,
        pml=column.pml,
        clamped=column.clamped,
        target_quality_factor=Q_SHIFT,
    )
    options = gfem.SolverOptions(order=2)
    job = gfem.Job(column.model, problem, options)
    path = job.to_json(tmp_path / "job.json")
    back = gfem.Job.from_json(path)
    assert back.problem == problem
    assert back.problem.pml == column.pml
    first = job.run(tmp_path / "first")
    second = back.run(tmp_path / "second")
    # the same assembly and the same eigensolver start: the eigenvalues agree
    # to the eigensolver's tolerance (1e-10 on the backward error), 1e-9
    np.testing.assert_allclose(
        np.asarray(second.omega), np.asarray(first.omega), rtol=1e-9
    )
    loaded = gfem.load_result(tmp_path / "second")
    np.testing.assert_allclose(
        np.asarray(loaded.omega), np.asarray(first.omega), rtol=1e-9
    )


def _problem_entry(path):
    """The ``MechanicalEigenmode`` object of a job file."""

    def find(node):
        if isinstance(node, dict):
            if node.get("$type") == "MechanicalEigenmode":
                return node
            node = list(node.values())
        for child in node if isinstance(node, list) else ():
            if (found := find(child)) is not None:
                return found
        return None

    return find(json.loads(path.read_text(encoding="utf-8")))


def test_a_problem_without_a_layer_writes_neither_key(tmp_path):
    column = ec.leaky_column(tmp_path / "column.msh", 0.5)
    options = gfem.SolverOptions(order=1)
    plain = gfem.MechanicalEigenmode(1e9, clamped=column.clamped)
    path = gfem.Job(column.model, plain, options).to_json(tmp_path / "plain.json")
    entry = _problem_entry(path)
    assert "pml" not in entry
    assert "target_quality_factor" not in entry
    layered = gfem.MechanicalEigenmode(1e9, clamped=column.clamped, pml=column.pml)
    path = gfem.Job(column.model, layered, options).to_json(tmp_path / "layered.json")
    entry = _problem_entry(path)
    assert entry["pml"]["regions"] == ["pml"]
    assert "target_quality_factor" not in entry


# --- refusals ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coarse(tmp_path_factory):
    return ec.leaky_column(tmp_path_factory.mktemp("coarse") / "c.msh", 0.5)


def _refuses(column, error, match, **kwargs):
    problem = gfem.MechanicalEigenmode(column.target, clamped=column.clamped, **kwargs)
    with pytest.raises(error, match=match):
        gfem.solve(column.model, problem, options=gfem.SolverOptions(order=1))


def test_one_absorber_per_problem(coarse):
    with pytest.raises(ValueError, match="one absorber per problem"):
        gfem.MechanicalEigenmode(
            1e9, pml=coarse.pml, damping_frame=gfem.DampingFrame(["pml"])
        )


def test_a_target_quality_factor_needs_a_layer(coarse):
    with pytest.raises(ValueError, match=r"target_quality_factor.*pml=gfem\.PML"):
        gfem.MechanicalEigenmode(1e9, target_quality_factor=100.0)
    with pytest.raises(ValueError, match="must be > 0"):
        gfem.MechanicalEigenmode(1e9, pml=coarse.pml, target_quality_factor=-1.0)


def test_a_layer_region_outside_the_solve_domain_refuses(coarse):
    model = coarse.model.replace(excluded={"mechanical": ["pml"]})
    column = ec.Column(model, coarse.pml, coarse.omega, coarse.clamped, 0.0)
    _refuses(column, ValueError, "outside the mechanical domain", pml=coarse.pml)


def test_an_inside_layer_and_an_uneven_side_refuse(coarse, tmp_path):
    # the substrate lies between the film and the layer: it is inside the box
    # of the regions that are not layers
    _refuses(
        coarse,
        ValueError,
        "lies inside the interior box",
        pml=gfem.PML(["sub"]),
    )
    # two layer regions on one side that end on different planes
    rects = [
        (0.0, 0.0, 1.0, 1.0, "film", None),
        (0.0, -1.0, 0.5, 0.0, "shallow", None),
        (0.5, -2.0, 1.0, 0.0, "deep", None),
    ]
    edges = [
        (lambda p: abs(p[0]) < 1e-9, "left"),
        (lambda p: abs(p[0] - 1.0) < 1e-9, "right"),
    ]
    nodes, tris, lines = tc._mesh2d(rects, 0.5, edges)
    tc._write_section(tmp_path / "uneven.msh", nodes, tris, lines)
    mesh = gfem.Mesh.from_gmsh(tmp_path / "uneven.msh", length_unit="um")
    material = ec._material(ec.MU_LAYER, ec.RHO_LAYER, None)
    model = gfem.Model(
        mesh, materials={"film": material, "shallow": material, "deep": material}
    )
    problem = gfem.MechanicalEigenmode(
        3e9, pml=gfem.PML(["shallow", "deep"]), clamped={"left": "x", "right": "x"}
    )
    with pytest.raises(ValueError, match="does not end on one plane"):
        gfem.solve(model, problem, options=gfem.SolverOptions(order=1))


def test_an_absorbing_line_on_a_layer_face_refuses(coarse):
    model = coarse.model.replace(boundaries=(gfem.Absorbing("out_bottom"),))
    column = ec.Column(model, coarse.pml, coarse.omega, coarse.clamped, 0.0)
    _refuses(column, ValueError, "Absorbing on 'out_bottom'", pml=coarse.pml)


def test_a_material_field_in_a_layer_refuses(coarse):
    materials = dict(coarse.model.materials)
    materials["pml"] = gfem.MaterialField.uniform("pml", materials["pml"])
    model = coarse.model.replace(materials=materials)
    column = ec.Column(model, coarse.pml, coarse.omega, coarse.clamped, 0.0)
    _refuses(column, NotImplementedError, "constant Material", pml=coarse.pml)


def test_an_electromagnetic_layer_over_the_solid_names_the_remedy(tmp_path):
    body = ec.leaky_column(tmp_path / "body.msh", 0.6, dim=3)
    model = body.model.replace(pml=body.pml)
    _refuses(
        ec.Column(model, body.pml, body.omega, body.clamped, 0.0),
        ValueError,
        r"does not read the model's electromagnetic layer: pass it as "
        r"MechanicalEigenmode\(pml=",
    )
