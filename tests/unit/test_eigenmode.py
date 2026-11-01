"""gfem.solve on an Eigenmode problem: resonances of a parallel-plate line.

The line runs along x between PEC plates at z = 0 and z = h, with PMC side
walls (the default), so its TEM field is exact in the discrete space: a
uniform E_z along y and z. With PMC ends (open) it resonates at the
half-wave frequencies f_m = m c / (2 L sqrt(eps_r)); with a resistor R on
each end its modes are the roots of the transmission-line round trip
Gamma^2 exp(-2 i beta L) = 1, Gamma = (R - Z0) / (R + Z0), complex
frequencies with Q = |omega| / (2 Im omega). Both hold up to the
discretization's dispersion error.
"""

import cmath
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm
from anisotropy import turned
from eigen_cases import (
    EPS,
    ETA0,
    F1,
    Z0,
    H,
    L,
    W,
    line,
    line_mesh,
    line_roots,
    terminated_line,
)
from preconditioners import coarse_columns, count_preconditioners
from scipy.optimize import brentq

import gfem
from gfem._eigen import rule as eigen_rule
from gfem._eigen.operator import RESHIFT_SOLVES, EigenOperator
from gfem._eigen.seed import MIN_SUBSPACE, seed_operator
from gfem._engine import terms
from gfem._engine.driven import driven_operator
from gfem._engine.eigen import projector
from gfem._engine.runtime import task_manager
from gfem._linalg import prepare
from gfem._linalg.base import PlannedHandle
from gfem._linalg.bddc import BddcHandle


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return line_mesh(tmp_path_factory.mktemp("line"))


# --- the problem value -----------------------------------------------------------


def test_eigenmode_problem_is_a_static_value():
    p = gfem.Eigenmode(target=5e9, n_modes=2)
    assert (p.target, p.n_modes) == (5e9, 2)
    assert gfem.Eigenmode(5e9).n_modes == 1
    assert p == gfem.Eigenmode(target=5e9, n_modes=2)
    assert hash(p) == hash(gfem.Eigenmode(target=5e9, n_modes=2))
    assert jax.tree.leaves(p) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": -1.0}, "target"),
        ({"target": float("nan")}, "target"),
        ({"target": 1e9, "n_modes": 0}, "n_modes"),
        ({"target": 1e9, "n_modes": 1.5}, "n_modes"),
    ],
)
def test_eigenmode_rejects_bad_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        gfem.Eigenmode(**kwargs)


# --- linear problem: no damping ----------------------------------------------------


def test_open_line_resonates_at_the_half_wave_frequencies(mesh):
    result = gfem.solve(line(mesh), gfem.Eigenmode(target=8e9, n_modes=2))
    assert isinstance(result, gfem.EigenmodeResult)
    # nearest the target first
    np.testing.assert_allclose(result.frequency, [F1, 2 * F1], rtol=1e-5)
    omega = np.asarray(result.omega)
    assert omega.shape == (2,)
    assert not omega.imag.any()  # a lossless, undamped problem: real frequencies
    assert np.isinf(np.asarray(result.q)).all()
    assert (np.asarray(result.error) < 1e-6).all()


def test_mode_field_stores_one_joule_of_electric_energy(mesh):
    result = gfem.solve(line(mesh), gfem.Eigenmode(target=8e9))
    field = result.e_field(0)
    assert (field.quantity, field.unit, field.space) == ("E", "V/m", "hcurl")
    assert complex(field.omega) == complex(result.omega[0])
    assert gfem.conventions.MODE_ENERGY == 1.0
    # E_z = A cos(pi x / L): 1/2 eps0 eps_r A^2 W H L / 2 = 1 J (the peak of
    # the discrete cosine, 20 cells along x, is within 0.3 %)
    amplitude = math.sqrt(4 / (const.epsilon_0 * EPS * W * H * L * 1e-9))
    sampled = field.to_meshio().point_data
    np.testing.assert_allclose(np.abs(sampled["E_real"]).max(), amplitude, rtol=5e-3)
    assert np.abs(sampled["E_imag"]).max() < 1e-9 * amplitude
    np.testing.assert_allclose(result.bulk_participation["dielectric"], [1.0])
    with pytest.raises(IndexError, match="1 mode"):
        result.e_field(1)


def _jittered(mesh, seed, rel=1e-13):
    """``mesh`` with every node coordinate moved by ``rel`` (relative)."""
    nodes = np.array(mesh.imported_nodes)
    nodes *= 1 + rel * np.random.default_rng(seed).uniform(-1, 1, nodes.shape)
    return jax.tree.unflatten(jax.tree.structure(mesh), [jnp.asarray(nodes)])


def test_mode_sign_ties_break_on_the_lowest_dof_index(mesh):
    # E_z = A cos(pi x / L) on the symmetric line: the largest coefficients
    # sit at both ends with opposite signs and equal magnitudes up to
    # rounding, so a rounding-level node move must not flip the mode's sign.
    def mode(m):
        return gfem.solve(line(m), gfem.Eigenmode(target=8e9)).e_field(0)

    near_port1 = np.array([[L / 10 * 1e-3, W / 2 * 1e-3, H / 2 * 1e-3]])
    for m in (mesh, *(_jittered(mesh, seed) for seed in range(6))):
        field = mode(m)
        x = np.asarray(field.coefficients)
        magnitude = np.abs(x)
        first = np.flatnonzero(magnitude >= (1 - 1e-12) * magnitude.max())[0]
        assert x[first].real > 0
        assert np.asarray(field.at(near_port1))[0, 2].real > 0


def test_loss_tangent_divides_the_eigenvalue_exactly(mesh):
    # K x = k^2 eps_r (1 - i tan_delta) M x: the lossy k^2 is the lossless one
    # over (1 - i tan_delta), in the discrete problem too
    problem = gfem.Eigenmode(target=8e9)
    lossless = complex(gfem.solve(line(mesh), problem).omega[0])
    lossy = gfem.solve(line(mesh, tan_delta=0.01), problem)
    expected = lossless / cmath.sqrt(1 - 0.01j)
    np.testing.assert_allclose(complex(lossy.omega[0]), expected, rtol=1e-10)
    assert lossy.omega[0].imag > 0  # decays under exp(+i omega t)
    np.testing.assert_allclose(lossy.q, abs(expected) / (2 * expected.imag))


def test_per_axis_loss_tangent_acts_along_its_axis(mesh):
    # the mode is E_z: only the principal values along z (the first axis
    # here) matter, as an isotropic eps_r, tan_delta with those values (up
    # to the discrete field's small E_x / E_y part; the mean tan_delta would
    # give a 17x larger Im omega)
    problem = gfem.Eigenmode(target=8e9)
    isotropic = complex(gfem.solve(line(mesh, tan_delta=0.01), problem).omega[0])
    per_axis = gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.Material(
                eps_r=(EPS, 7.0, 2.0),
                tan_delta=(0.01, 0.3, 0.2),
                axes=((0, 0, 1), (1, 0, 0), (0, 1, 0)),
            )
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    )
    omega = complex(gfem.solve(per_axis, problem).omega[0])
    np.testing.assert_allclose(omega.real, isotropic.real, rtol=1e-5)
    np.testing.assert_allclose(omega.imag, isotropic.imag, rtol=1e-3)


def test_bulk_participation_shares_the_energy_between_regions(tmp_path):
    # one dielectric in two layers: the TEM field is uniform in z, so each
    # layer holds its thickness's share of the energy (to the cosine's
    # discretization error along x)
    path = tmp_path / "layered.msh"
    layers = (("low", 0.3), ("high", 0.7))
    sm.write_msh22(path, sm.strip_line(L, W, H, (20, 1, 1), layers=layers))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    material = gfem.Material(eps_r=EPS)
    model = gfem.Model(
        mesh,
        materials={"low": material, "high": material},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    )
    result = gfem.solve(model, gfem.Eigenmode(target=8e9))
    p = result.bulk_participation
    np.testing.assert_allclose(p["low"], [0.3], rtol=1e-5)
    np.testing.assert_allclose(p["high"], [0.7], rtol=1e-5)
    np.testing.assert_allclose(p["low"] + p["high"], [1.0], rtol=1e-14)


def _crystal_line(mesh, **losses):
    """The line in a biaxial crystal on axes turned 30 degrees about x."""
    crystal = gfem.Material(eps_r=(3.0, 4.0, 6.0), axes=turned(0, 30), **losses)
    return gfem.Model(
        mesh,
        materials={"dielectric": crystal},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    )


def test_principal_participation_gives_a_per_axis_loss_exactly(mesh):
    # E lies along z, off every principal axis: (a_k . z)^2 = 0, 1/4, 3/4.
    # Per axis, 1/Q = sum_k p_k (tan_delta_k + sigma_k / (eps0 eps_k Re omega))
    # (design 1/Q = 3.18e-3), to O(1/Q) against the complex eigenvalue, with
    # p from the lossy solve or from a lossless one (assumed losses)
    problem = gfem.Eigenmode(target=6.4e9)
    lossless = gfem.solve(_crystal_line(mesh), problem)
    eps = np.array([3.0, 4.0, 6.0])
    tan_delta = np.array([1e-2, 0.0, 3e-3])
    w0 = float(lossless.omega[0].real)
    sigma = np.array([0.0, 4e-3 * const.epsilon_0 * 4.0 * w0, 0.0])
    lossy = gfem.solve(
        _crystal_line(mesh, tan_delta=tuple(tan_delta), conductivity=tuple(sigma)),
        problem,
    )
    omega = complex(lossy.omega[0])
    inverse_q = 2 * abs(omega.imag) / abs(omega)
    per_axis = tan_delta + sigma / (const.epsilon_0 * eps * omega.real)
    for result in (lossy, lossless):
        assert set(result.principal_participation) == {"dielectric"}
        p = np.asarray(result.principal_participation["dielectric"])
        assert p.shape == (1, 3)
        np.testing.assert_allclose(
            p.sum(axis=1), result.bulk_participation["dielectric"], rtol=1e-12
        )
        # measured on MN5: 3.2e-6 (well inside O(1/Q) = 3e-3)
        np.testing.assert_allclose(p[0] @ per_axis, inverse_q, rtol=1e-5)
    # the rule of one loss tangent, with their mean, is far off (36 % on MN5)
    mean = float(lossy.bulk_participation["dielectric"][0]) * tan_delta.mean()
    assert abs(mean / inverse_q - 1) > 0.2


def test_principal_participation_covers_the_anisotropic_regions_only(tmp_path):
    # the layered line: "high" a field on turned axes with equal principal
    # values, "low" isotropic. The TEM field is E_z, so high's share 0.7
    # splits as (a_k . z)^2 = 0, 1/4, 3/4
    path = tmp_path / "layered.msh"
    layers = (("low", 0.3), ("high", 0.7))
    sm.write_msh22(path, sm.strip_line(L, W, H, (20, 1, 1), layers=layers))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    n = len(mesh.elements("high"))
    high = gfem.MaterialField(eps_r=jnp.full((n, 3), EPS), axes=turned(0, 30))
    model = gfem.Model(
        mesh,
        materials={"low": gfem.Material(eps_r=EPS), "high": high},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    )
    result = gfem.solve(model, gfem.Eigenmode(target=8e9, n_modes=2))
    assert list(result.principal_participation) == ["high"]
    p = np.asarray(result.principal_participation["high"])
    assert p.shape == (2, 3)
    np.testing.assert_allclose(
        p.sum(axis=1), result.bulk_participation["high"], rtol=1e-12
    )
    # measured on MN5: 3.6e-4 relative, 5.2e-6 along x (the discrete field's
    # small E_x / E_y part)
    np.testing.assert_allclose(p, [[0.0, 0.175, 0.525]] * 2, rtol=1e-3, atol=2e-5)
    pytest.importorskip("xarray")
    ds = result.to_xarray()
    da = ds["principal_participation"]
    assert da.dims == ("mode", "anisotropic_region", "axis")
    assert list(ds["anisotropic_region"].values) == ["high"]
    np.testing.assert_array_equal(ds["axis"], [0, 1, 2])
    np.testing.assert_array_equal(da.sel(anisotropic_region="high"), p)


# --- quadratic problem: resistive ports --------------------------------------------


def test_terminated_line_modes_are_the_transmission_line_roots(mesh):
    model = terminated_line(mesh, 1000.0)
    result = gfem.solve(model, gfem.Eigenmode(target=8e9, n_modes=2))
    expected = line_roots(1000.0, [1, 2])
    np.testing.assert_allclose(result.omega, expected, rtol=1e-5)
    q = np.abs(expected) / (2 * expected.imag)
    np.testing.assert_allclose(result.q, q, rtol=1e-4)
    assert (np.asarray(result.error) < 1e-6).all()
    assert set(result.q_port) == {"p1", "p2"}
    assert result.participation == {}


def test_coupling_q_of_weakly_coupled_ports_adds_up_to_the_mode_q(mesh):
    # 1/Q = sum 1/Q_port to first order in the (small) port loss
    result = gfem.solve(terminated_line(mesh, 1e5), gfem.Eigenmode(target=8e9))
    q = float(result.q[0])
    q1, q2 = float(result.q_port["p1"][0]), float(result.q_port["p2"][0])
    np.testing.assert_allclose(q1, q2, rtol=1e-6)  # symmetric line
    np.testing.assert_allclose(1 / q1 + 1 / q2, 1 / q, rtol=5 / q)


def test_target_far_below_the_mode_still_finds_the_resonance(mesh):
    # the curl-curl null space (frequency 0) is nearer the shift than the mode:
    # the divergence-free projection keeps it out
    result = gfem.solve(terminated_line(mesh, 1000.0), gfem.Eigenmode(target=1e9))
    np.testing.assert_allclose(result.omega, line_roots(1000.0, [1]), rtol=1e-5)


@pytest.mark.parametrize(
    "linear_solver", ["bddc_gmres", pytest.param("petsc", marks=pytest.mark.petsc)]
)
def test_shift_solves_on_each_linear_solver_give_the_direct_modes(mesh, linear_solver):
    model = terminated_line(mesh, 1000.0)
    problem = gfem.Eigenmode(target=8e9)
    direct = gfem.solve(model, problem)
    other = gfem.solve(
        model, problem, options=gfem.SolverOptions(linear_solver=linear_solver)
    )
    np.testing.assert_allclose(other.omega, direct.omega, rtol=1e-9)


@pytest.mark.parametrize("n_modes", [3, 4])
@pytest.mark.parametrize("linear_solver", ["direct", "bddc_gmres"])
def test_seeds_far_from_the_last_reshift_still_give_their_own_modes(
    mesh, linear_solver, n_modes
):
    # after mode 2's reshift, the seeds of roots 3 and 4 are rougher than
    # TRUSTED: refined on mode 2's shift, they used to drift onto mode 2 and
    # be dropped as duplicates (or onto a mirror root), so a farther root
    # took their place
    model = terminated_line(mesh, 1000.0)
    result = gfem.solve(
        model,
        gfem.Eigenmode(target=8e9, n_modes=n_modes),
        options=gfem.SolverOptions(linear_solver=linear_solver),
    )
    # the dispersion error grows with frequency: 1.1e-4 at root 4
    expected = line_roots(1000.0, range(1, n_modes + 1))
    np.testing.assert_allclose(result.omega, expected, rtol=3e-4)


def test_a_far_seed_refined_onto_a_farther_root_still_gives_its_own_mode(
    mesh, monkeypatch
):
    # on Windows (py3.14), root 3's seed refined on mode 2's shift converged
    # to root 4 or 5, not yet accepted, which then took root 3's place; here
    # that refinement is made to land on root 4 on every platform
    model = terminated_line(mesh, 1000.0)
    roots = line_roots(1000.0, range(1, 5))
    rule = eigen_rule.EigenmodeRule
    seeds, shift_invert, refine = {}, rule._shift_invert, eigen_rule.refine

    def capture(self, op, *args, **kwargs):
        k, vectors = shift_invert(self, op, *args, **kwargs)
        seeds.update(k=k, vectors=vectors)
        return k, vectors

    def onto_root_4(op, k, x, **kwargs):
        omega = np.asarray(seeds["k"]) * const.c / op.scale
        root_3 = abs(k * const.c / op.scale - roots[2]) < 1e-3 * abs(roots[2])
        if root_3 and abs(op.shift - k) > 1e-3 * abs(k):  # a shift nearer another
            j = int(np.argmin(np.abs(omega - roots[3])))
            op.reshift(seeds["k"][j])
            return refine(op, seeds["k"][j], seeds["vectors"][:, j], **kwargs)
        return refine(op, k, x, **kwargs)

    monkeypatch.setattr(rule, "_shift_invert", capture)
    monkeypatch.setattr(eigen_rule, "refine", onto_root_4)
    result = gfem.solve(model, gfem.Eigenmode(target=8e9, n_modes=3))
    np.testing.assert_allclose(result.omega, roots[:3], rtol=3e-4)


def test_a_pair_with_its_weight_on_the_fixed_dofs_is_not_converged(mesh, monkeypatch):
    # on CI, root 3's seed refined on mode 2's shift ended with 1e-8 of its
    # norm on the free DOFs: measured on the whole vector, its backward
    # error passed the tolerance at 22.62-0.13j GHz, off any root
    rule = eigen_rule.EigenmodeRule
    select, refined = rule._select, []

    def refine_a_stray_pair(self, op, *args, **kwargs):
        rng = np.random.default_rng(0)
        free = rng.standard_normal(len(op.free)) + 1j * rng.standard_normal(
            len(op.free)
        )
        free = np.where(op.free, free, 0)
        stray = np.where(op.free, 1e-12 * free / np.linalg.norm(free), 1.0)
        k = op.rayleigh(stray, op.shift)
        refined.append(
            eigen_rule.refine(op, k, stray, tolerance=1e-10, max_iterations=0)
        )
        return select(self, op, *args, **kwargs)

    monkeypatch.setattr(rule, "_select", refine_a_stray_pair)
    gfem.solve(terminated_line(mesh, 1000.0), gfem.Eigenmode(target=8e9))
    assert not refined[0].converged


def test_far_seeds_of_a_polynomial_operator_are_refined_at_their_own_ritz_value(
    mesh, monkeypatch
):
    # root 3's seed (backward error 2.6e-6) used to be refined on mode 2's
    # shift, where it wandered through stall reshifts off any root before
    # landing where round-off took it; a polynomial operator's Ritz values
    # are its own, so each far seed is refactorized at its own one
    model = terminated_line(mesh, 1000.0)
    roots = line_roots(1000.0, range(1, 4))
    reshift, shifts = EigenOperator.reshift, []

    def spy(self, k, **kwargs):
        shifts.append(k * const.c / self.scale)
        return reshift(self, k, **kwargs)

    monkeypatch.setattr(EigenOperator, "reshift", spy)
    result = gfem.solve(model, gfem.Eigenmode(target=8e9, n_modes=3))
    np.testing.assert_allclose(result.omega, roots, rtol=3e-4)
    np.testing.assert_allclose(shifts, roots[1:], rtol=3e-4)


def _count_shift_prepares(monkeypatch) -> list[int]:
    """Spy on the eigen operator's `prepare`: one entry per ``T(k)`` handle."""
    calls = []

    def spy(*args, **kwargs):
        if kwargs["symmetry"] == "complex_symmetric":  # not the projector's H1
            calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr("gfem._linalg.prepare", spy)
    return calls


@pytest.mark.parametrize(
    "linear_solver", ["direct", pytest.param("petsc", marks=pytest.mark.petsc)]
)
def test_a_reshift_refactorizes_the_prepared_operator(mesh, monkeypatch, linear_solver):
    # the second mode, far from the target, is refined at its own shift
    model = terminated_line(mesh, 1000.0)
    problem = gfem.Eigenmode(target=8e9, n_modes=2)
    options = gfem.SolverOptions(linear_solver=linear_solver)
    calls = _count_shift_prepares(monkeypatch)
    result = gfem.solve(model, problem, options=options)
    assert len(calls) == 1
    # a handle that cannot refactorize is prepared afresh at the new shift
    monkeypatch.delattr(PlannedHandle, "refactorize")
    calls.clear()
    fresh = gfem.solve(model, problem, options=options)
    assert len(calls) == 2
    np.testing.assert_allclose(result.omega, fresh.omega, rtol=1e-12)
    np.testing.assert_allclose(result.q, fresh.q, rtol=1e-10)


def test_a_bddc_reshift_sets_the_preconditioner_up_at_the_new_shift(mesh, monkeypatch):
    # shift-invert solves start cold, so a reshift never solves on the
    # preconditioner of the old shift: the modes are those of fresh handles
    model = terminated_line(mesh, 1000.0)
    problem = gfem.Eigenmode(target=8e9, n_modes=2)
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    builds = count_preconditioners(monkeypatch)
    result = gfem.solve(model, problem, options=options)
    assert len(builds) == 2
    monkeypatch.delattr(BddcHandle, "refactorize")
    builds.clear()
    fresh = gfem.solve(model, problem, options=options)
    assert len(builds) == 2
    np.testing.assert_allclose(result.omega, fresh.omega, rtol=1e-12)
    np.testing.assert_allclose(result.q, fresh.q, rtol=1e-10)


def test_a_bddc_shift_sets_its_coarse_problem_up_for_the_seed_solve(mesh, monkeypatch):
    # ARPACK's seed solve applies T(sigma)^-1 at least MIN_SUBSPACE times
    model = terminated_line(mesh, 1000.0)
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    columns = coarse_columns(monkeypatch)
    gfem.solve(model, gfem.Eigenmode(target=8e9), options=options)
    assert columns == [MIN_SUBSPACE]


def test_a_bddc_reshift_sets_its_coarse_problem_up_for_the_refinement(
    mesh, monkeypatch
):
    # the second mode is refined at its own shift, which serves a few solves
    model = terminated_line(mesh, 1000.0)
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    columns = coarse_columns(monkeypatch)
    gfem.solve(model, gfem.Eigenmode(target=8e9, n_modes=2), options=options)
    assert columns == [MIN_SUBSPACE, RESHIFT_SOLVES]


def test_a_retry_after_a_reshift_seeds_at_the_target(mesh, monkeypatch):
    # too few modes after refinement reshifted: the wider seed solve runs
    # at the target again, on a coarse problem counted for the seed
    model = terminated_line(mesh, 1000.0)
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    rule = eigen_rule.EigenmodeRule
    shifts, select, shift_invert = [], rule._select, rule._shift_invert

    def seed(self, op, *args, **kwargs):
        shifts.append(op.shift)
        return shift_invert(self, op, *args, **kwargs)

    def too_few(self, *args, **kwargs):
        accepted, failed, guard = select(self, *args, **kwargs)
        if len(shifts) == 1:  # the first seed solve's modes: one short
            return accepted[:-1], [], None
        return accepted, failed, guard

    monkeypatch.setattr(rule, "_shift_invert", seed)
    monkeypatch.setattr(rule, "_select", too_few)
    columns = coarse_columns(monkeypatch)
    gfem.solve(model, gfem.Eigenmode(target=8e9, n_modes=2), options=options)
    target = 2 * math.pi * 8e9 * 1e-3 / const.c  # k0 per mm
    assert shifts[0] == pytest.approx(target, rel=1e-12)
    assert shifts == [shifts[0], shifts[0]]
    assert columns == [MIN_SUBSPACE, RESHIFT_SOLVES, MIN_SUBSPACE, RESHIFT_SOLVES]


@pytest.mark.parametrize(
    ("tolerances", "inner"),
    [({"eigen_tolerance": 1e-12, "tolerance": 1e-12}, 1e-13), ({}, 1e-12)],
)
def test_inner_solves_are_not_asked_below_the_round_off_floor(
    mesh, monkeypatch, tolerances, inner
):
    # a tight eigen tolerance would ask BDDC + GMRES for 1e-14, below the
    # ~1e-13 true preconditioned residual the shift-invert operator allows
    # in double precision; the defaults ask 1e-2 * 1e-10
    options = gfem.SolverOptions(linear_solver="bddc_gmres", **tolerances)
    rule = eigen_rule.EigenmodeRule(line(mesh), gfem.Eigenmode(target=F1), options)
    asked = []

    def capture(*_, tolerance, **__):
        asked.append(tolerance)

    monkeypatch.setattr(eigen_rule, "eigen_operator", capture)
    rule.operator(None, 1.0)
    np.testing.assert_allclose(asked, [inner], rtol=1e-12)


# --- inductive ports -----------------------------------------------------------------


def test_inductor_participation_matches_the_loaded_line(mesh):
    # an inductor L_p across the x = 0 end, the other end open: resonance at
    # tan(beta L) = Z0 / (omega L_p); peak energies 1/2 L_p |V(0) / omega L_p|^2
    # in the inductor and 1/2 C' integral of |V|^2 on the line
    inductance = 1e-9
    model = line(mesh, ports=[gfem.LumpedPort("jj", "port1", "+z", inductance=1e-9)])
    length = L * 1e-3
    speed = const.c / math.sqrt(EPS)

    def resonance(w):
        return math.tan(w / speed * length) - Z0 / (w * inductance)

    w = brentq(resonance, 1e6, 0.4999 * math.pi * speed / length)
    bl = w / speed * length
    c_line = const.epsilon_0 * EPS * W / H
    electric = 0.5 * c_line * (length / 2 + math.sin(2 * bl) * length / (4 * bl))
    stored = 0.5 * inductance * (math.cos(bl) / (w * inductance)) ** 2
    result = gfem.solve(model, gfem.Eigenmode(target=w / (2 * math.pi)))
    np.testing.assert_allclose(result.omega, [w], rtol=1e-5)
    assert set(result.participation) == {"jj"}
    assert result.q_port == {}  # no resistance
    np.testing.assert_allclose(
        np.abs(result.participation["jj"]), [stored / electric], rtol=1e-4
    )


# --- unsupported terms and failures ----------------------------------------------


def test_wave_port_raises_from_solve_with_a_remedy(mesh):
    # the model itself stays valid (a driven problem can use the wave port);
    # only the eigenmode solve refuses it
    model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[gfem.WavePort("o1", "port1")],
    )
    with pytest.raises(
        NotImplementedError, match=r"(?s)wave port 'o1'.*LumpedPort.*gfem\.Driven"
    ):
        gfem.solve(model, gfem.Eigenmode(target=8e9))


def test_conductivity_damps_the_line_exactly(mesh):
    # K x = (k^2 eps_r - i k eta0 sigma) M x for the uniform TEM field: with the
    # lossless k0, k^2 - i a k - k0^2 = 0, a = eta0 sigma / eps_r, in the
    # discrete problem too
    sigma = 0.05
    problem = gfem.Eigenmode(target=8e9)
    k0 = complex(gfem.solve(line(mesh), problem).omega[0]) / const.c
    lossy = gfem.solve(line(mesh, conductivity=sigma), problem)
    a = ETA0 * sigma / EPS
    k = 0.5j * a + cmath.sqrt(k0**2 - a**2 / 4)
    np.testing.assert_allclose(complex(lossy.omega[0]), k * const.c, rtol=1e-10)


def test_per_axis_conductivity_damps_along_its_axis(mesh):
    # the mode is E_z: only the conductivity along z (the first axis here)
    # damps it, as an isotropic conductivity of that value (up to the
    # discrete field's small E_x / E_y part; the mean would damp 34x more)
    sigma = 0.05
    problem = gfem.Eigenmode(target=8e9)
    isotropic = complex(gfem.solve(line(mesh, conductivity=sigma), problem).omega[0])
    per_axis = line(
        mesh,
        conductivity=(sigma, 50 * sigma, 50 * sigma),
        axes=((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    )
    omega = complex(gfem.solve(per_axis, problem).omega[0])
    # measured on MN5: Re omega 1.6e-8, Im omega 5.4e-5 relative
    np.testing.assert_allclose(omega.real, isotropic.real, rtol=5e-8)
    np.testing.assert_allclose(omega.imag, isotropic.imag, rtol=2e-4)


def test_a_uniform_field_of_a_crystal_has_the_crystal_s_modes(mesh):
    # a tilted biaxial lossy, conducting, magnetic crystal: its uniform field
    # holds the same principal values on the same axes
    crystal = gfem.Material(
        eps_r=(3.0, 5.0, 6.0),
        mu_r=(1.5, 1.0, 1.2),
        tan_delta=(0.01, 0.02, 0.03),
        conductivity=(0.02, 0.05, 0.03),
        axes=((1 / 3, 2 / 3, 2 / 3), (2 / 3, 1 / 3, -2 / 3), (2 / 3, -2 / 3, 1 / 3)),
    )

    def omega(material):
        model = gfem.Model(
            mesh,
            materials={"dielectric": material},
            boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        )
        problem = gfem.Eigenmode(target=5e9, n_modes=2)
        return np.asarray(gfem.solve(model, problem).omega)

    expected = omega(crystal)
    field = omega(gfem.MaterialField.uniform("dielectric", crystal))
    np.testing.assert_array_equal(field, expected)  # measured on MN5: bit for bit


def test_first_order_absorbing_walls_damp_the_modes(tmp_path):
    # a wide line with absorbing side walls: a quadratic problem whose modes
    # radiate through the walls
    path = tmp_path / "wide.msh"
    sm.write_msh22(path, sm.strip_line(L, 4 * L, H, (10, 10, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.Absorbing("side")],
    )
    result = gfem.solve(model, gfem.Eigenmode(target=F1))
    assert 1 < float(result.q[0]) < 100
    np.testing.assert_allclose(result.frequency, [F1], rtol=0.1)
    assert (np.asarray(result.error) < 1e-6).all()


def test_unconverged_modes_raise_without_a_partial_result(mesh):
    options = gfem.SolverOptions(eigen_tolerance=1e-17, max_refine_iterations=1)
    with pytest.raises(
        gfem.EigenmodeConvergenceError, match=r"(?s)0 of the 1.*Remedies"
    ):
        gfem.solve(terminated_line(mesh, 1000.0), gfem.Eigenmode(8e9), options=options)


# --- the divergence-free projection ----------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_divfree_projection_matches_the_complex_product(tmp_path, order):
    # the real G applied to the real and imaginary parts gives what SciPy's
    # complex product (G upcast to complex) gives: bit for bit on x86_64, to
    # rounding on arm64, where the complex product contracts into FMAs; two
    # layers in z leave free H1 nodes between the plates at order 1 too
    path = tmp_path / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (6, 1, 2)))
    model = line(gfem.Mesh.from_gmsh(path, length_unit="mm"))
    with task_manager(1):
        op = driven_operator(model, order)
        project = projector(model, op)
        g = project._g.astype(complex)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(g.shape[0]) + 1j * rng.standard_normal(g.shape[0])
        expected = x - g @ project._laplace.solve(g.T @ project.energy(x))
        projected = project(x)
        project.free()
    assert np.linalg.norm(projected - x) > 0.1 * np.linalg.norm(x)
    np.testing.assert_allclose(projected, expected, rtol=1e-13, atol=1e-13)


def test_the_companion_apply_projects_once(mesh, monkeypatch):
    # on vectors whose first half is projected, [P a; y1 + sigma P a] is the
    # companion step [P a; P(y1 + sigma a)], a = -T(sigma)^-1 (M (y2 + sigma
    # y1) + D' y1), for one projection and one seed-mass product (ADR 0035)
    model = terminated_line(mesh, 1000.0)
    problem = gfem.Eigenmode(target=8e9)
    rule = eigen_rule.EigenmodeRule(model, problem, gfem.SolverOptions())
    with task_manager(1):
        op = driven_operator(model, rule.options.order)
        sigma = terms.wavenumber(2 * math.pi * problem.target, op.built.scale)
        eop = rule.operator(op, sigma)
        eop.factorize()
        divfree = projector(model, op)
        try:
            assert eop.seed_damped
            n = len(eop.free)
            rng = np.random.default_rng(0)
            r = rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))
            y1, y2 = divfree(r[0]), r[1]
            a = -eop.solve(
                eop.seed_mass(y2) + eop.seed_damping(y1) + sigma * eop.seed_mass(y1)
            )
            expected = np.concatenate([divfree(a), divfree(y1 + sigma * a)])

            calls = {"project": 0, "mass": 0}
            mass = eop.seed_mass

            def project(x):
                calls["project"] += 1
                return divfree(x)

            def counted_mass(x):
                calls["mass"] += 1
                return mass(x)

            monkeypatch.setattr(eop, "seed_mass", counted_mass)
            seed = seed_operator(eop, project)
            calls.update(project=0, mass=0)
            applied = seed.apply(np.concatenate([y1, y2]))
        finally:
            eop.free_handle()
            divfree.free()
    assert calls == {"project": 1, "mass": 1}
    np.testing.assert_allclose(
        applied, expected, rtol=0, atol=1e-12 * np.linalg.norm(expected)
    )


# --- serialization -----------------------------------------------------------------


def test_eigenmode_job_round_trips_its_result(mesh, tmp_path):
    model = terminated_line(mesh, 1000.0)
    problem = gfem.Eigenmode(target=8e9, n_modes=2)
    job = gfem.Job.from_json(gfem.Job(model, problem).to_json(tmp_path / "job.json"))
    assert job.problem == problem
    result = job.run(out=tmp_path / "results")
    loaded = gfem.load_result(tmp_path / "results")
    assert type(loaded) is gfem.EigenmodeResult
    np.testing.assert_array_equal(loaded.omega, result.omega)
    np.testing.assert_array_equal(loaded.error, result.error)
    np.testing.assert_array_equal(loaded.q_port["p2"], result.q_port["p2"])
    for a, b in zip(loaded.e_fields, result.e_fields, strict=True):
        np.testing.assert_array_equal(a.coefficients, b.coefficients)
        assert complex(a.omega) == complex(b.omega)
    assert "GHz" in repr(loaded)
