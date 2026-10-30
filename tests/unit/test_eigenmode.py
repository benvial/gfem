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
from gfem._eigen.operator import RESHIFT_SOLVES
from gfem._eigen.seed import MIN_SUBSPACE
from gfem._engine import eigen as engine_eigen
from gfem._engine.driven import driven_operator
from gfem._engine.eigen import projector
from gfem._engine.runtime import task_manager
from gfem._linalg import prepare
from gfem._linalg.bddc import BddcHandle
from gfem._linalg.direct import DirectHandle
from gfem._linalg.petsc import PetscHandle


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


def _count_shift_prepares(monkeypatch) -> list[int]:
    """Spy on the eigen operator's `prepare`: one entry per ``T(k)`` handle."""
    calls = []

    def spy(*args, **kwargs):
        if kwargs["symmetry"] == "complex_symmetric":  # not the projector's H1
            calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(engine_eigen, "prepare", spy)
    return calls


@pytest.mark.parametrize(
    ("linear_solver", "handle"),
    [
        ("direct", DirectHandle),
        pytest.param("petsc", PetscHandle, marks=pytest.mark.petsc),
    ],
)
def test_a_reshift_refactorizes_the_prepared_operator(
    mesh, monkeypatch, linear_solver, handle
):
    # the second mode, far from the target, is refined at its own shift
    model = terminated_line(mesh, 1000.0)
    problem = gfem.Eigenmode(target=8e9, n_modes=2)
    options = gfem.SolverOptions(linear_solver=linear_solver)
    calls = _count_shift_prepares(monkeypatch)
    result = gfem.solve(model, problem, options=options)
    assert len(calls) == 1
    # a handle that cannot refactorize is prepared afresh at the new shift
    monkeypatch.delattr(handle, "refactorize")
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
