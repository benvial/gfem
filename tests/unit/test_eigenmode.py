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
import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm
from scipy.optimize import brentq

import gfem

L, W, H = 10.0, 2.0, 1.0  # mm
EPS = 4.0
ETA0 = const.mu_0 * const.c
Z0 = ETA0 * H / (W * math.sqrt(EPS))
F1 = const.c / (2 * L * 1e-3 * math.sqrt(EPS))  # 7.49 GHz


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (20, 1, 1)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line(mesh, ports=(), **material):
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS, **material)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=list(ports),
    )


def _terminated(mesh, r):
    return _line(
        mesh,
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=r),
            gfem.LumpedPort("p2", "port2", "+z", impedance=r),
        ],
    )


def _line_roots(r, modes):
    """Complex angular frequencies of the line with R on both ends."""
    gamma = (r - Z0) / (r + Z0)
    out = []
    for m in modes:
        # exp(-2 i beta L) = Gamma^-2  =>  beta = (2 pi m - 2 i log Gamma) / (2 L)
        beta = (2 * math.pi * m - 2j * cmath.log(gamma)) / (2 * L * 1e-3)
        out.append(beta * const.c / math.sqrt(EPS))
    return np.array(out)


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
    result = gfem.solve(_line(mesh), gfem.Eigenmode(target=8e9, n_modes=2))
    assert isinstance(result, gfem.EigenmodeResult)
    # nearest the target first
    np.testing.assert_allclose(result.frequency, [F1, 2 * F1], rtol=1e-5)
    omega = np.asarray(result.omega)
    assert omega.shape == (2,)
    assert not omega.imag.any()  # a lossless, undamped problem: real frequencies
    assert np.isinf(np.asarray(result.q)).all()
    assert (np.asarray(result.error) < 1e-6).all()


def test_mode_field_stores_one_joule_of_electric_energy(mesh):
    result = gfem.solve(_line(mesh), gfem.Eigenmode(target=8e9))
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


def test_loss_tangent_divides_the_eigenvalue_exactly(mesh):
    # K x = k^2 eps_r (1 - i tan_delta) M x: the lossy k^2 is the lossless one
    # over (1 - i tan_delta), in the discrete problem too
    problem = gfem.Eigenmode(target=8e9)
    lossless = complex(gfem.solve(_line(mesh), problem).omega[0])
    lossy = gfem.solve(_line(mesh, tan_delta=0.01), problem)
    expected = lossless / cmath.sqrt(1 - 0.01j)
    np.testing.assert_allclose(complex(lossy.omega[0]), expected, rtol=1e-10)
    assert lossy.omega[0].imag > 0  # decays under exp(+i omega t)
    np.testing.assert_allclose(lossy.q, abs(expected) / (2 * expected.imag))


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
    model = _terminated(mesh, 1000.0)
    result = gfem.solve(model, gfem.Eigenmode(target=8e9, n_modes=2))
    expected = _line_roots(1000.0, [1, 2])
    np.testing.assert_allclose(result.omega, expected, rtol=1e-5)
    q = np.abs(expected) / (2 * expected.imag)
    np.testing.assert_allclose(result.q, q, rtol=1e-4)
    assert (np.asarray(result.error) < 1e-6).all()
    assert set(result.q_port) == {"p1", "p2"}
    assert result.participation == {}


def test_coupling_q_of_weakly_coupled_ports_adds_up_to_the_mode_q(mesh):
    # 1/Q = sum 1/Q_port to first order in the (small) port loss
    result = gfem.solve(_terminated(mesh, 1e5), gfem.Eigenmode(target=8e9))
    q = float(result.q[0])
    q1, q2 = float(result.q_port["p1"][0]), float(result.q_port["p2"][0])
    np.testing.assert_allclose(q1, q2, rtol=1e-6)  # symmetric line
    np.testing.assert_allclose(1 / q1 + 1 / q2, 1 / q, rtol=5 / q)


def test_target_far_below_the_mode_still_finds_the_resonance(mesh):
    # the curl-curl null space (frequency 0) is nearer the shift than the mode:
    # the divergence-free projection keeps it out
    result = gfem.solve(_terminated(mesh, 1000.0), gfem.Eigenmode(target=1e9))
    np.testing.assert_allclose(result.omega, _line_roots(1000.0, [1]), rtol=1e-5)


def test_bddc_gmres_shift_solves_give_the_direct_modes(mesh):
    model = _terminated(mesh, 1000.0)
    problem = gfem.Eigenmode(target=8e9)
    direct = gfem.solve(model, problem)
    iterative = gfem.solve(
        model, problem, options=gfem.SolverOptions(linear_solver="bddc_gmres")
    )
    np.testing.assert_allclose(iterative.omega, direct.omega, rtol=1e-9)


# --- inductive ports -----------------------------------------------------------------


def test_inductor_participation_matches_the_loaded_line(mesh):
    # an inductor L_p across the x = 0 end, the other end open: resonance at
    # tan(beta L) = Z0 / (omega L_p); peak energies 1/2 L_p |V(0) / omega L_p|^2
    # in the inductor and 1/2 C' integral of |V|^2 on the line
    inductance = 1e-9
    model = _line(mesh, ports=[gfem.LumpedPort("jj", "port1", "+z", inductance=1e-9)])
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


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"boundaries": [gfem.Conductor("side", 5.8e7)]}, "Conductor.*'side'"),
        ({"boundaries": [gfem.Absorbing("side", order=2)]}, "second-order"),
        ({"ports": [gfem.WavePort("o1", "port1")]}, "wave port 'o1'.*LumpedPort"),
    ],
)
def test_nonlinear_terms_raise_with_a_remedy(mesh, extra, match):
    model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), *extra.get("boundaries", [])],
        ports=extra.get("ports", []),
    )
    with pytest.raises(NotImplementedError, match=match):
        gfem.solve(model, gfem.Eigenmode(target=8e9))


def test_conductivity_damps_the_line_exactly(mesh):
    # K x = (k^2 eps_r - i k eta0 sigma) M x for the uniform TEM field: with the
    # lossless k0, k^2 - i a k - k0^2 = 0, a = eta0 sigma / eps_r, in the
    # discrete problem too
    sigma = 0.05
    problem = gfem.Eigenmode(target=8e9)
    k0 = complex(gfem.solve(_line(mesh), problem).omega[0]) / const.c
    lossy = gfem.solve(_line(mesh, conductivity=sigma), problem)
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
        gfem.solve(_terminated(mesh, 1000.0), gfem.Eigenmode(8e9), options=options)


# --- serialization -----------------------------------------------------------------


def test_eigenmode_job_round_trips_its_result(mesh, tmp_path):
    model = _terminated(mesh, 1000.0)
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
