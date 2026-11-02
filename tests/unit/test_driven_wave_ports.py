"""Driven problems with wave ports: `gfem.solve(model, gfem.Driven(...))`.

The parallel-plate fixture carries an exact TEM port mode, and its 3D field
is a plane wave, so a two-port line has S21 = exp(-gamma L) and S11 = 0 up
to the 3D discretization error.
"""

import dataclasses
import itertools
import math
import re
import warnings
from pathlib import Path

import anisotropy as anisotropy_reference
import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import pytest
import scipy.constants
import scipy.sparse as sp
import structured_meshes as sm
from gradient_cases import HYBRID_PORTS, hybrid_leads_core, hybrid_line, hybrid_mesh
from solver_misses import cannot_converge_at

import gfem
from gfem._engine import anisotropy, terms
from gfem._engine.driven import driven_operator
from gfem._engine.wave_ports import trace_dofs, wave_port_couplings
from gfem._linalg import prepare
from gfem._linalg.base import PlannedHandle, free_mask

DATA = Path(__file__).parent / "data"
C0 = 299792458.0
ETA0 = 376.730313412
W, D, L = 40e-6, 10e-6, 20e-6  # plate width, gap and length, m
EPS = 4.0
F = (1.5e12, 2.0e12, 2.5e12)


@pytest.fixture(scope="module")
def plate_mesh():
    return gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")


def line(mesh, *, offsets=(0.0, 0.0), second="wave", path=True):
    vp = ((W / 2, 0.0, 0.0), (W / 2, D, 0.0)) if path else None
    ports = [gfem.WavePort("p1", "port1", offset=offsets[0], voltage_path=vp)]
    if second == "wave":
        ports.append(gfem.WavePort("p2", "port2", offset=offsets[1]))
    else:
        ports.append(
            gfem.LumpedPort("p2", "port2", direction=(0, 1, 0), impedance=z_tem())
        )
    return gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("plates")],
        ports=ports,
    )


def beta(f):
    return 2 * math.pi * f / C0 * math.sqrt(EPS)


def z_tem():
    return ETA0 / math.sqrt(EPS) * D / W


OPTIONS = gfem.SolverOptions(order=3)


@pytest.fixture(scope="module")
def plate_result(plate_mesh):
    return gfem.solve(line(plate_mesh), gfem.Driven(F), options=OPTIONS)


def test_a_matched_line_transmits_its_tem_mode(plate_result):
    s = np.asarray(plate_result.s.values)
    expected = np.exp(-1j * np.array([beta(f) for f in F]) * L)
    np.testing.assert_allclose(s[:, 1, 0], expected, atol=1e-5)
    np.testing.assert_allclose(s[:, 0, 1], expected, atol=1e-5)
    assert np.abs(s[:, 0, 0]).max() < 1e-5
    assert np.abs(s[:, 1, 1]).max() < 1e-5


def test_the_result_holds_each_wave_ports_mode(plate_result):
    modes = plate_result.port_modes
    assert set(modes) == {"p1", "p2"}
    gamma = np.asarray(modes["p1"].gamma)
    np.testing.assert_allclose(gamma, 1j * np.array([beta(f) for f in F]), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(modes["p1"].Z_PV), z_tem(), rtol=1e-8)
    assert np.isnan(np.asarray(modes["p2"].Z_PV)).all()
    assert np.asarray(modes["p1"].solved).all()  # three frequencies: exact solves
    assert modes["p1"].e_field is None  # no fields kept


def test_reference_impedance_is_z_pv_or_nan(plate_result):
    z = np.asarray(plate_result.s.z_ref)
    np.testing.assert_allclose(z[:, 0], z_tem(), rtol=1e-8)
    assert np.isnan(z[:, 1]).all()


def test_offsets_move_the_reference_planes(plate_mesh, plate_result):
    d1, d2 = 3e-6, 5e-6
    shifted = gfem.solve(
        line(plate_mesh, offsets=(d1, d2)), gfem.Driven(F), options=OPTIONS
    )
    assert dict(shifted.offsets) == {"p1": d1, "p2": d2}
    assert dict(plate_result.offsets) == {"p1": 0.0, "p2": 0.0}
    g = 1j * np.array([beta(f) for f in F])
    s0, s = np.asarray(plate_result.s.values), np.asarray(shifted.s.values)
    np.testing.assert_allclose(
        s[:, 1, 0], s0[:, 1, 0] * np.exp(g * (d1 + d2)), atol=1e-9
    )
    np.testing.assert_allclose(s[:, 0, 0], s0[:, 0, 0] * np.exp(2 * g * d1), atol=1e-9)


def test_a_wave_port_and_a_matched_lumped_port_mix(plate_mesh):
    r = gfem.solve(line(plate_mesh, second="lumped"), gfem.Driven(F), options=OPTIONS)
    s = np.asarray(r.s.values)
    expected = np.exp(-1j * np.array([beta(f) for f in F]) * L)
    np.testing.assert_allclose(s[:, 1, 0], expected, atol=1e-3)
    np.testing.assert_allclose(s[:, 0, 1], s[:, 1, 0], atol=1e-6)
    assert np.abs(s[:, 0, 0]).max() < 1e-3
    assert set(r.port_modes) == {"p1"}


def _jittered(mesh, rel=1e-13, seed=0):
    """``mesh`` with in-plane node coordinates moved by ``rel`` (relative)."""
    nodes = np.array(mesh.imported_nodes)
    nodes[:, :2] *= 1 + rel * np.random.default_rng(seed).uniform(
        -1, 1, (len(nodes), 2)
    )
    return jax.tree.unflatten(jax.tree.structure(mesh), [jnp.asarray(nodes)])


# jittered: a rounding-level node move must not change the ports' phase
# references (it once put S01 - S10 at 2.3e-6 on some CPUs).
@pytest.mark.parametrize("nodes", ["imported", "jittered"])
def test_conductor_plates_make_a_lossy_passive_reciprocal_line(plate_mesh, nodes):
    mesh = plate_mesh if nodes == "imported" else _jittered(plate_mesh)
    model = line(mesh).replace(
        boundaries=[gfem.Conductor("plates", conductivity=5.8e7)]
    )
    # direct: GMRES at tolerance 1e-6 leaves S off by up to ~1e-6, so S01 - S10
    # (2.4e-7 here, 2.6e-6 on another CPU) would gate the solver, not the
    # physics. The direct solve's asymmetry is 1.6e-10 at any thread count.
    options = dataclasses.replace(OPTIONS, linear_solver="direct")
    r = gfem.solve(model, gfem.Driven(F), options=options)
    s = np.asarray(r.s.values)
    gamma = np.asarray(r.port_modes["p1"].gamma)
    assert (gamma.real > 0).all()
    np.testing.assert_allclose(s[:, 0, 1], s[:, 1, 0], rtol=0, atol=1e-8)
    power = np.abs(s[:, 0, 0]) ** 2 + np.abs(s[:, 1, 0]) ** 2
    assert (power < 1).all()
    # the Robin term keeps Re k_n only: the lossy line is matched to O(alpha/beta)
    beta = gamma.imag
    np.testing.assert_allclose(
        np.abs(s[:, 1, 0]), np.exp(-gamma.real * L), atol=4 * (gamma.real / beta).max()
    )


GRID = tuple(np.linspace(1.0e12, 3.0e12, 9))


def test_adaptive_s_matches_the_uniform_sweep(plate_mesh):
    model = line(plate_mesh)
    uniform = gfem.solve(model, gfem.Driven(GRID), options=OPTIONS)
    adaptive = gfem.solve(model, gfem.Driven(GRID, sweep="adaptive"), options=OPTIONS)
    assert len(adaptive.sweep.sampled_frequencies) < len(GRID)
    np.testing.assert_allclose(
        np.asarray(adaptive.s.values), np.asarray(uniform.s.values), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(adaptive.port_modes["p1"].gamma),
        np.asarray(uniform.port_modes["p1"].gamma),
        rtol=1e-12,
    )


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
def test_a_direct_sweep_plans_once_and_matches_fresh_handles(
    plate_mesh, monkeypatch, sweep
):
    # one base handle, wrapped per frequency in a borrowing Woodbury update
    model = line(plate_mesh, second="lumped")
    problem = gfem.Driven(F, sweep=sweep)
    options = dataclasses.replace(OPTIONS, linear_solver="direct")
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr("gfem._linalg.prepare", spy)
    s = np.asarray(gfem.solve(model, problem, options=options).s.values)
    assert len(calls) == 1
    # a handle that cannot refactorize is prepared afresh at each frequency
    monkeypatch.delattr(PlannedHandle, "refactorize")
    calls.clear()
    result = gfem.solve(model, problem, options=options)
    solved = len(F) if sweep == "uniform" else len(result.sweep.sampled_frequencies)
    assert len(calls) == solved > 1
    np.testing.assert_allclose(result.s.values, s, rtol=0, atol=1e-12)


def test_wave_port_s_comes_out_of_jit_unchanged(plate_mesh, plate_result):
    @jax.jit
    def s(model):
        return gfem.solve(model, gfem.Driven(F), options=OPTIONS).s.values

    np.testing.assert_allclose(
        s(line(plate_mesh)), plate_result.s.values, rtol=1e-10, atol=1e-12
    )


def test_a_design_touching_a_wave_port_face_raises_under_grad(plate_mesh):
    # "fill" fills the whole line, so its elements sit on both port faces:
    # the port mode would depend on the design.
    model = line(plate_mesh).with_material(
        "fill", gfem.MaterialField.uniform("fill", gfem.Material(eps_r=EPS))
    )

    def s21(m):
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        result = gfem.solve(m, gfem.Driven(F[:1]), options=OPTIONS)
        return jnp.abs(result.s.values[0, 1, 0])

    with pytest.raises(gfem.PortGeometryError, match=r"WavePort 'p1'.*'fill'"):
        jax.grad(s21)(model)
    # a forward solve of the same model is fine, and so is stop_gradient
    assert np.isfinite(float(s21(model)))
    frozen = jax.grad(lambda m: s21(jax.lax.stop_gradient(m)))(model)
    assert not np.any(np.asarray(frozen.materials["fill"].eps_r))


def test_thermal_properties_beside_a_wave_port_have_zero_gradient(plate_mesh):
    # the driven solve (port mode included) never reads them: exact zeros
    field = gfem.MaterialField.uniform(
        "fill", gfem.Material(eps_r=EPS, thermal_conductivity=1.0)
    )
    model = line(plate_mesh).with_material("fill", field)

    def s21(k):
        m = model.with_material(
            "fill", dataclasses.replace(field, thermal_conductivity=k)
        )
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        result = gfem.solve(m, gfem.Driven(F[:1]), options=OPTIONS)
        return jnp.abs(result.s.values[0, 1, 0])

    assert float(jax.grad(s21)(field.thermal_conductivity).sum()) == 0.0


def test_kept_fields_keep_the_port_mode_fields(plate_mesh):
    r = gfem.solve(
        line(plate_mesh), gfem.Driven(F, excite="p1", keep_fields=F[1]), options=OPTIONS
    )
    field = r.port_modes["p1"].e_field
    e = np.asarray(field.at([W / 4, D / 2, 0.0]))  # (n_freq, 3)
    np.testing.assert_allclose(np.abs(e[:, 1]), math.sqrt(2 * z_tem()) / D, rtol=1e-8)


def test_a_sweep_that_does_not_converge_keeps_its_solved_port_modes(
    plate_mesh, monkeypatch
):
    problem = gfem.Driven(F, excite="p1", keep_fields=F[0])
    options = gfem.SolverOptions(order=3, linear_solver="bddc_gmres")
    clean = gfem.solve(line(plate_mesh), problem, options=options).port_modes["p1"]
    cannot_converge_at(monkeypatch, (F[1],))
    with pytest.raises(gfem.LinearSolverConvergenceError) as caught:
        gfem.solve(line(plate_mesh), problem, options=options)
    mode = caught.value.result.port_modes["p1"]
    np.testing.assert_array_equal(mode.frequency, F[:1])
    # the port mode sweeps of two solves agree to rounding, not bit for bit
    np.testing.assert_allclose(mode.gamma, np.asarray(clean.gamma)[:1], rtol=1e-12)
    point = [W / 4, D / 2, 0.0]
    expected = np.asarray(clean.e_field.at(point))[:1]
    np.testing.assert_allclose(
        mode.e_field.at(point), expected, rtol=0, atol=1e-10 * np.abs(expected).max()
    )


def test_wave_port_results_round_trip_through_a_bundle(plate_mesh, tmp_path):
    result = gfem.Job(
        line(plate_mesh, offsets=(1e-6, 0.0)), gfem.Driven(F), OPTIONS
    ).run(out=tmp_path / "out")
    loaded = gfem.load_result(tmp_path / "out")
    np.testing.assert_array_equal(loaded.s.values, result.s.values)
    assert loaded.offsets == result.offsets
    np.testing.assert_array_equal(
        loaded.port_modes["p1"].gamma, result.port_modes["p1"].gamma
    )


def test_characteristic_impedance_needs_kept_port_mode_fields(plate_mesh, plate_result):
    path = ((W / 2, 0.0, 0.0), (W / 2, D, 0.0))
    with pytest.raises(ValueError, match="keep_fields"):
        gfem.characteristic_impedance(plate_result.port_modes["p1"], path)
    kept = gfem.solve(
        line(plate_mesh), gfem.Driven(F, excite="p1", keep_fields=True), options=OPTIONS
    )
    z = np.asarray(gfem.characteristic_impedance(kept.port_modes["p1"], path))
    np.testing.assert_allclose(z, z_tem(), rtol=1e-8)


def test_a_singular_reaction_raises_instead_of_dropping_the_correction(
    plate_mesh, monkeypatch
):
    from gfem._engine import wave_ports

    built = wave_ports.PortSurfaceCoupling

    def vanishing(piece, vectors, reaction, ports):
        return built(piece, vectors, np.zeros_like(reaction), ports)

    monkeypatch.setattr(wave_ports, "PortSurfaceCoupling", vanishing)
    with pytest.raises(gfem.PortModeError, match="singular") as caught:
        gfem.solve(line(plate_mesh), gfem.Driven(F), options=OPTIONS)
    message = str(caught.value)
    assert "'p1'" in message
    assert "'port1'" in message
    assert f"{F[0] / 1e9:.6g} GHz" in message


# --- hybrid modes: the modal correction -------------------------------------------

HYBRID_F = (9e9, 16e9)
HYBRID_LENGTH = 2e-3  # m, the fixture's length


@pytest.fixture(scope="module")
def hybrid_guide():
    # WR-90 with an eps_r = 6 block along its whole length: the port mode is
    # hybrid (E_n != 0), so the guide is matched only by a port that imposes
    # the mode's full n x H, grad E_n included.
    mesh = gfem.Mesh.from_gmsh(DATA / "hybrid_waveguide.msh", length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"block": gfem.Material(eps_r=6.0), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def test_a_uniform_guide_matches_its_hybrid_mode(hybrid_guide):
    # without the modal correction S11 is 0.03-0.4 here (Robin term only)
    options = gfem.SolverOptions(order=3, linear_solver="direct")
    problem = gfem.Driven(HYBRID_F)
    result = gfem.solve(hybrid_guide, problem, options=options)
    s = np.asarray(result.s.values)
    gamma = np.asarray(result.port_modes["p1"].gamma)
    assert np.abs(gamma.real).max() < 1e-9 * np.abs(gamma).max()  # lossless
    assert np.abs(s[:, 0, 0]).max() < 1e-4
    assert np.abs(s[:, 1, 1]).max() < 1e-4
    np.testing.assert_allclose(s[:, 1, 0], np.exp(-gamma * HYBRID_LENGTH), atol=1e-4)
    np.testing.assert_allclose(s[:, 0, 1], s[:, 1, 0], atol=1e-8)


# --- an open dielectric guide: an absorbing boundary --------------------------

N_CLAD, N_CORE = 1.444, 2.25
CORE_W, GUIDE_L, HALF_BOX, HALF_PORT = 0.55, 2.0, 1.5, 1.1  # um
SLAB_H = 0.15  # um: the slab thickness and the mesh size
F0 = C0 / 1.55e-6
N_EFF_SLAB = 2.057557  # fundamental mode of the 0.55 um slab, analytic


def _grid(lines, h):
    """Points on ``lines`` with spacing at most ``h`` in between."""
    pts = sorted(lines)
    steps = [np.array(pts[:1])]
    for a, b in itertools.pairwise(pts):
        steps.append(np.linspace(a, b, math.ceil((b - a) / h - 1e-9) + 1)[1:])
    return np.concatenate(steps)


@pytest.fixture(scope="module")
def dielectric_guide(tmp_path_factory):
    # a straight n 2.25 core in n 1.444 cladding between PEC plates, open to
    # the sides through a second-order absorbing boundary; the port windows
    # stop short of the box and close their edges with PEC
    edges = (
        _grid((-GUIDE_L / 2, GUIDE_L / 2), SLAB_H),
        _grid(
            (-HALF_BOX, -HALF_PORT, -CORE_W / 2, CORE_W / 2, HALF_PORT, HALF_BOX),
            SLAB_H,
        ),
        np.array([0.0, SLAB_H]),
    )
    eps = 1e-9

    def volume(c):
        return "core" if abs(c[1]) <= CORE_W / 2 else "cladding"

    def surface(c):
        x, y, z = c
        if abs(z) < eps or abs(z - SLAB_H) < eps:
            return "plates"
        for x0, name in ((-GUIDE_L / 2, "port1"), (GUIDE_L / 2, "port2")):
            if abs(x - x0) < eps:
                return name if abs(y) < HALF_PORT else "outer"
        return "outer" if abs(abs(y) - HALF_BOX) < eps else ""

    path = tmp_path_factory.mktemp("guide") / "guide.msh"
    sm.write_msh22(path, sm.structured(edges, volume, surface))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={
            "cladding": gfem.Material(eps_r=N_CLAD**2),
            "core": gfem.Material(eps_r=N_CORE**2),
        },
        boundaries=[gfem.PEC("plates"), gfem.Absorbing("outer", order=2)],
        ports=[
            gfem.WavePort("p1", "port1", pec_edges=("outer",)),
            gfem.WavePort("p2", "port2", pec_edges=("outer",)),
        ],
    )
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    return gfem.solve(model, gfem.Driven(F0, excite=None), options=options)


def test_an_open_dielectric_guide_transmits_its_fundamental_mode(dielectric_guide):
    s = np.asarray(dielectric_guide.s.values)[0]
    s11, s21, s12 = s[0, 0], s[1, 0], s[0, 1]
    assert abs(s21) ** 2 > 0.999
    assert abs(s11) ** 2 < 1e-4
    assert abs(s11) ** 2 + abs(s21) ** 2 <= 1 + 1e-6  # passive
    assert abs(s12 - s21) < 1e-8  # reciprocal
    n_eff = float(np.real(np.asarray(dielectric_guide.port_modes["p1"].n_eff)[0]))
    assert abs(n_eff - N_EFF_SLAB) < 1e-4
    beta = 2 * math.pi * F0 / C0 * n_eff
    assert abs(np.angle(s21 * np.exp(1j * beta * GUIDE_L * 1e-6))) < 0.02


# --- anisotropic permeability: the Robin tensor and the modal coupling --------

#: WR-28 filled with mu_r = (1, 2, 1.5), eps_r = (2, 4, 3): TE10 alone
#: propagates at 16 GHz (research note of ADR 0054).
F_WR28 = 16e9


def test_the_robin_tensor_is_the_plane_wave_n_cross_h():
    # A E_t = -n x (eta0 H) for a plane wave along n, with A = N mu^-1 N^T
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        mu = anisotropy_reference.transverse(rng, n, 0.5, 4.0)
        a = anisotropy.tangential_inverse_permeability(mu[None], n[None])[0]
        e = np.cross(n, rng.normal(size=3))
        h = anisotropy_reference.plane_wave_n_cross_h(mu, n, e)
        worst = max(worst, np.abs(a @ e + h).max() / np.abs(h).max())
        flipped = anisotropy.tangential_inverse_permeability(mu[None], -n[None])[0]
        np.testing.assert_allclose(flipped, a, rtol=0, atol=1e-14 * np.abs(a).max())
    assert worst < 1e-12
    n = np.array([0.6, 0.0, 0.8])
    a = anisotropy.tangential_inverse_permeability(2.5 * np.eye(3)[None], n[None])[0]
    np.testing.assert_allclose(a, (np.eye(3) - np.outer(n, n)) / 2.5, atol=1e-15)


@pytest.fixture(scope="module")
def wr28_mesh(tmp_path_factory):
    return anisotropy_reference.wr28(tmp_path_factory.mktemp("wr28"))


def _wr28(mesh, material):
    return gfem.Model(
        mesh,
        materials={"fill": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


@pytest.fixture(scope="module")
def turned_wr28(wr28_mesh):
    # mu_tt and eps_tt turned 30 degrees about the guide axis: the port
    # frame's mu_tt is a full 2x2 matrix
    material = gfem.Material(
        eps_r=(2.0, 4.0, 3.0),
        mu_r=(1.0, 2.0, 1.5),
        axes=anisotropy_reference.turned(2, 30.0),
    )
    model = _wr28(wr28_mesh, material)
    op = driven_operator(model, 2)
    (surface, _) = wave_port_couplings(
        model,
        op.built,
        op.space,
        order=2,
        frequency=(F_WR28,),
        tolerance=gfem.SolverOptions().port_mode_tolerance,
    )
    return model, op, surface


def test_the_modal_coupling_carries_the_modes_power(turned_wr28):
    # |(i / (omega mu0)) int E_t . conj(n x H)| = 2 |P|: the coupling forms
    # give the mode's n x H with the same tensor as its power
    _, _, surface = turned_wr28
    (coupling,) = surface.ports
    sweep = coupling.sweep
    modes = sweep.space
    m = sweep.basis.shape[1] // 2
    gamma = sweep.gamma[0] * coupling.scale  # per file unit
    c = sweep.coords[0]
    g_form, t_form = modes.coupling_forms
    w = sweep.basis[modes.free]
    h = gamma * (t_form @ (w[:, :m] @ c[:m])) + g_form @ (w[:, m:] @ c[m:])
    e_t = sweep.basis[: modes.n_t, :m] @ c[:m]
    omega = 2 * math.pi * F_WR28
    projection = coupling.scale * abs(e_t @ np.conj(h)) / (omega * scipy.constants.mu_0)
    power = modes.power(sweep.basis @ c, gamma, omega)
    # measured 0 (round-off); the rotated mu_tt^-1 gave 1.386 for 2
    np.testing.assert_allclose(projection, 2 * abs(power), rtol=1e-14)


def test_the_port_reaction_is_symmetric(turned_wr28):
    _, _, surface = turned_wr28
    m = surface.reaction.shape[1]
    r = surface.reaction[:m]
    assert np.abs(r - r.T).max() <= 1e-14 * np.abs(r).max()  # measured 0


def test_the_robin_coefficient_is_the_port_sections_inverse_permeability(
    turned_wr28,
):
    model, _, surface = turned_wr28
    section = surface.ports[0].sweep.space.section
    coef = terms.wave_port_coefficients(model, "port1")
    p = section.frame[:2]
    lifted = p.T @ section.nu_tt @ p
    assert coef.shape == (len(model.mesh.triangles), 3, 3)
    # measured 0
    np.testing.assert_allclose(coef[section.faces], lifted, rtol=0, atol=1e-14)
    off = np.ones(len(coef), bool)
    off[section.faces] = False
    assert (coef[off] == 0).all()


def test_the_port_imposes_its_modes_n_cross_h(turned_wr28):
    # (i beta M + W) e = h on the port's own mode e: Robin term and modal
    # correction read the same tensor, so the port does not reflect it
    model, op, surface = turned_wr28
    (port,) = surface.ports
    sweep, space = port.sweep, op.space
    idx3, idx2 = trace_dofs(model, sweep.space.section, sweep.space, op.built, space)
    e = np.zeros(space.ndof, complex)
    e[idx3] = (sweep.basis @ sweep.coords[0])[idx2]
    u, v = space.TnT()
    form = ngs.BilinearForm(space)
    form += surface.piece.integrand(u, v)
    form.Assemble()
    rows, cols, vals = form.mat.COO()
    robin = sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(space.ndof, space.ndof),
    )
    omega = 2 * math.pi * F_WR28
    vectors = surface.vectors
    w = vectors @ (surface.correction_coefficients(omega) @ (vectors.T @ e))
    h = vectors @ port.phi[0]
    residual = complex(surface.piece.theta(omega)) * (robin @ e) + w - h
    free = free_mask(space.FreeDofs(), space.ndof)
    assert np.abs(residual[free]).max() <= 4e-15 * np.abs(h).max()  # measured 1.3e-15


def test_the_turned_ports_correction_is_the_lone_closed_form(turned_wr28):
    # the per-surface builder on an anisotropic port (ADR 0054 with ADR
    # 0046): W = h hᵀ / (hᵀ e) - h_s h_sᵀ / (h_sᵀ e), complex symmetric
    model, op, surface = turned_wr28
    (port,) = surface.ports
    sweep, space = port.sweep, op.space
    idx3, idx2 = trace_dofs(model, sweep.space.section, sweep.space, op.built, space)
    e = np.zeros(space.ndof, complex)
    e[idx3] = (sweep.basis @ sweep.coords[0])[idx2]
    u = surface.vectors
    m = u.shape[1] // 2
    coords = sweep.coords[0]
    h = u @ port.phi[0]
    gamma = sweep.gamma[0] * port.scale
    h_s = u @ np.concatenate([gamma * coords[:m], np.zeros(m)])
    c = surface.correction_coefficients(2 * math.pi * F_WR28)
    assert np.linalg.norm(c - c.T) <= 1e-14 * np.linalg.norm(c)
    # on random vectors: W itself is a dense ndof x ndof
    x = np.random.default_rng(0).standard_normal((u.shape[0], 6))
    closed = np.outer(h, h @ x) / (h @ e) - np.outer(h_s, h_s @ x) / (h_s @ e)
    w = u @ (c @ (u.T @ x))
    assert np.linalg.norm(w - closed) <= 1e-14 * np.linalg.norm(closed)


def test_an_isotropic_port_keeps_a_scalar_robin_coefficient(wr28_mesh):
    model = _wr28(wr28_mesh, gfem.Material(eps_r=4.0, mu_r=2.0))
    coef = terms.wave_port_coefficients(model, "port1")
    on = wr28_mesh.triangle_tags == wr28_mesh.region("port1").tag
    assert coef.ndim == 1
    assert (coef[on] == 0.5).all()
    assert (coef[~on] == 0).all()


def test_an_anisotropic_permittivity_at_a_port_takes_the_tensor_path(wr28_mesh):
    # structural: an anisotropic Material behind the face, whatever its mu_r
    model = _wr28(wr28_mesh, gfem.Material(eps_r=(2.0, 4.0, 3.0), mu_r=2.0))
    coef = terms.wave_port_coefficients(model, "port1")
    on = wr28_mesh.triangle_tags == wr28_mesh.region("port1").tag
    assert coef.shape == (len(on), 3, 3)
    expected = np.broadcast_to(np.diag([0.5, 0.5, 0.0]), coef[on].shape)
    np.testing.assert_allclose(coef[on], expected, rtol=0, atol=1e-15)


# --- a shared port surface: several modes of one port --------------------------

# Modes 1 (TEM) and 2 (cos(pi x / W)) propagate; mode 2 is cut off at 1.874
# THz and mode 3 at 3.75 THz.
F2 = (2.0e12, 2.5e12)
SHARED = ("p1_TEM", "p1_m2", "p2_TEM", "p2_m2")
# direct: GMRES leaves S - Sᵀ at its tolerance (3.6e-7 here)
SHARED_OPTIONS = dataclasses.replace(OPTIONS, linear_solver="direct")


def shared_line(mesh, order=SHARED):
    # paths along y where each mode's E_y is positive: every polarity is fixed
    def path(x, z):
        return ((x, 0.0, z), (x, D, z))

    ports = {
        "p1_TEM": gfem.WavePort("p1_TEM", "port1", 1, voltage_path=path(W / 2, 0.0)),
        "p1_m2": gfem.WavePort("p1_m2", "port1", 2, voltage_path=path(W / 4, 0.0)),
        "p2_TEM": gfem.WavePort("p2_TEM", "port2", 1, voltage_path=path(W / 2, L)),
        "p2_m2": gfem.WavePort("p2_m2", "port2", 2, voltage_path=path(W / 4, L)),
    }
    return gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("plates")],
        ports=[ports[name] for name in order],
    )


def beta2(f):
    return math.sqrt(beta(f) ** 2 - (math.pi / W) ** 2)


@pytest.fixture(scope="module")
def shared_result(plate_mesh):
    return gfem.solve(shared_line(plate_mesh), gfem.Driven(F2), options=SHARED_OPTIONS)


def _entries(s, pairs):
    """``s[:, i, j]`` for each ``(i, j)``, and the largest other ``|s|``."""
    n = s.shape[1]
    rest = np.ones((n, n), bool)
    for i, j in pairs:
        rest[i, j] = False
    return [s[:, i, j] for i, j in pairs], np.abs(s[:, rest]).max()


def test_a_shared_port_surface_transmits_each_mode_of_a_parallel_plate(shared_result):
    s = np.asarray(shared_result.s.values)
    (tem, tem_back, m2, m2_back), rest = _entries(s, [(2, 0), (0, 2), (3, 1), (1, 3)])
    expected_tem = np.exp(-1j * np.array([beta(f) for f in F2]) * L)
    expected_m2 = np.exp(-1j * np.array([beta2(f) for f in F2]) * L)
    np.testing.assert_allclose(tem, expected_tem, atol=6e-6)  # measured 1.8e-6
    np.testing.assert_allclose(tem_back, expected_tem, atol=6e-6)
    np.testing.assert_allclose(m2, expected_m2, atol=1e-5)  # measured 2.8e-6
    np.testing.assert_allclose(m2_back, expected_m2, atol=1e-5)
    # reflections and cross-mode entries, measured 4.7e-6
    assert rest < 1.5e-5
    modes = shared_result.port_modes
    # measured: TEM 5e-16, mode 2 8.2e-7
    for name, b, rtol in (("p1_TEM", beta, 1e-8), ("p2_m2", beta2, 3e-6)):
        gamma = np.asarray(modes[name].gamma)
        np.testing.assert_allclose(gamma, 1j * np.array([b(f) for f in F2]), rtol=rtol)


def test_a_shared_port_surface_is_lossless_and_reciprocal(shared_result):
    s = np.asarray(shared_result.s.values)
    # every propagating mode is requested: S is unitary
    unitarity = np.einsum("fki,fkj->fij", s.conj(), s) - np.eye(4)
    assert np.abs(unitarity).max() < 3e-6  # measured 7.1e-7
    assert np.abs(s - np.swapaxes(s, 1, 2)).max() < 1e-8


def test_a_shared_port_surface_s_follows_the_port_order(plate_mesh, shared_result):
    # one surface's wave ports need not be adjacent
    order = ("p2_m2", "p1_TEM", "p2_TEM", "p1_m2")
    result = gfem.solve(
        shared_line(plate_mesh, order), gfem.Driven(F2), options=SHARED_OPTIONS
    )
    assert result.s.ports == order
    k = [SHARED.index(name) for name in order]
    s = np.asarray(shared_result.s.values)[:, k][:, :, k]
    np.testing.assert_allclose(result.s.values, s, rtol=0, atol=1e-10)


def test_a_shared_port_surface_excites_one_mode_alone(plate_mesh, shared_result):
    problem = gfem.Driven(F2, excite="p1_m2")
    result = gfem.solve(shared_line(plate_mesh), problem, options=SHARED_OPTIONS)
    column = np.asarray(shared_result.s.values)[:, :, SHARED.index("p1_m2")]
    np.testing.assert_allclose(
        np.asarray(result.s.values)[:, :, 0], column, rtol=0, atol=1e-10
    )


# Modes 1 and 2 of the hybrid guide both propagate from 16 GHz.
HYBRID_SHARED_F = (16e9, 20e9)
HYBRID_SHARED = ("p1_m1", "p1_m2", "p2_m1", "p2_m2")


def shared_hybrid(guide):
    return guide.replace(
        ports=[
            gfem.WavePort(name, f"port{name[1]}", mode=int(name[-1]))
            for name in HYBRID_SHARED
        ]
    )


@pytest.fixture(scope="module")
def shared_hybrid_result(hybrid_guide):
    options = gfem.SolverOptions(order=3, linear_solver="direct")
    problem = gfem.Driven(HYBRID_SHARED_F)
    return gfem.solve(shared_hybrid(hybrid_guide), problem, options=options)


def test_a_shared_hybrid_guide_transmits_both_modes(shared_hybrid_result):
    s = np.asarray(shared_hybrid_result.s.values)
    pairs = [(2, 0), (0, 2), (3, 1), (1, 3)]
    transmitted, rest = _entries(s, pairs)
    modes = shared_hybrid_result.port_modes
    for (i, j), t in zip(pairs, transmitted, strict=True):
        gamma = np.asarray(modes[HYBRID_SHARED[j]].gamma)
        expected = np.exp(-gamma * HYBRID_LENGTH)
        # no voltage path: the sign of each mode is a mesh-dependent rule
        error = np.minimum(np.abs(t - expected), np.abs(t + expected))
        # measured: mode 1 1.9e-4 (at 20 GHz; a lone port gives the same S21
        # to 4 digits, so it is the order-3 mesh's error), mode 2 1.1e-4
        assert error.max() < (6e-4 if i % 2 == 0 else 4e-4)
    assert rest < 3e-4  # reflections and cross-mode entries, measured 9.8e-5
    assert np.abs(s - np.swapaxes(s, 1, 2)).max() < 1e-8  # measured 1.2e-15
    unitarity = np.einsum("fki,fkj->fij", s.conj(), s) - np.eye(4)
    assert np.abs(unitarity).max() < 3e-5  # measured 7.8e-6


def _port1_surface(model, frequencies, order):
    """The engine's operator and port 1 surface coupling of a driven solve."""
    from gfem._engine.driven import DrivenRule

    problem = gfem.Driven(frequencies)
    rule = DrivenRule(model, problem, gfem.SolverOptions(order=order))
    system, _ = rule._ports(model)
    return system.operator, system.surfaces[0]


def _gram_coupling(surface, k):
    """``|G12| / sqrt(G11 G22)`` of a two-mode surface at output frequency k."""
    _, _, g = surface._blocks(k)  # gamma_0 G: the ratio is the same
    return abs(g[0, 1]) / math.sqrt(abs(g[0, 0] * g[1, 1]))


@pytest.fixture(scope="module")
def hybrid_couplings(hybrid_guide):
    """The engine's port 1 surface of the shared and of the lone hybrid guide."""
    out = {}
    for name, model in (
        ("shared", shared_hybrid(hybrid_guide)),
        ("lone", hybrid_guide),
    ):
        out[name] = (model, *_port1_surface(model, HYBRID_SHARED_F, order=3))
    return out


def _robin_and_traces(model, operator, surface):
    """The surface's Robin mass (θ = 1), the free DOFs and each mode's 3D trace."""
    import ngsolve as ngs
    import scipy.sparse

    from gfem._engine.wave_ports import trace_dofs
    from gfem._linalg.base import free_mask

    space = operator.space
    u, v = space.TnT()
    form = ngs.BilinearForm(space)
    form += surface.piece.integrand(u, v)
    form.Assemble()
    data, cols, rowptr = form.mat.CSR()
    mass = scipy.sparse.csr_matrix(
        (np.array(data), np.array(cols), np.array(rowptr)), shape=(space.ndof,) * 2
    )
    free = free_mask(space.FreeDofs(), space.ndof)
    sweep = surface.ports[0].sweep
    modes = sweep.space
    idx3, idx2 = trace_dofs(model, modes.section, modes, operator.built, space)
    idx3, idx2 = idx3[free[idx3]], idx2[free[idx3]]
    m = surface.vectors.shape[1] // 2
    traces = []
    for c in surface.ports:
        e_t = sweep.basis[: modes.n_t, :m] @ c.sweep.coords[0, :m]
        e = np.zeros(space.ndof, complex)
        e[idx3] = e_t[idx2]
        traces.append(e)
    return mass, free, traces


def test_a_shared_hybrid_surface_imposes_each_requested_mode(hybrid_couplings):
    # (i beta_0 M + W) E_k = H_k: the port is reflectionless on every mode
    model, operator, surface = hybrid_couplings["shared"]
    omega = 2 * math.pi * HYBRID_SHARED_F[0]
    mass, free, traces = _robin_and_traces(model, operator, surface)
    u, c = surface.vectors, surface.correction_coefficients(omega)
    theta = complex(surface.piece.theta(omega))
    assert len(traces) == 2
    for port, e in zip(surface.ports, traces, strict=True):
        h = u @ port.phi[0]
        imposed = theta * (mass @ e) + u @ (c @ (u.T @ e))
        # measured 1.2e-15
        assert np.linalg.norm((imposed - h)[free]) <= 1e-14 * np.linalg.norm(h[free])


def test_the_shared_hybrid_correction_is_symmetric_of_rank_2n(hybrid_couplings):
    _, _, surface = hybrid_couplings["shared"]
    c = surface.correction_coefficients(2 * math.pi * HYBRID_SHARED_F[0])
    assert np.linalg.norm(c - c.T) <= 1e-14 * np.linalg.norm(c)  # measured 2.4e-17
    sigma = np.linalg.svd(c, compute_uv=False)
    assert sigma[3] > 1e-8 * sigma[0]
    assert sigma[4] <= 1e-12 * sigma[0]  # rank 2 n = 4
    # W = U C Uᵀ on random vectors (W itself is a dense ndof x ndof)
    u = surface.vectors
    x = np.random.default_rng(0).standard_normal((u.shape[0], 6))
    w = x.T @ (u @ (c @ (u.T @ x)))
    assert np.linalg.norm(w - w.T) <= 1e-14 * np.linalg.norm(w)  # measured 7.9e-16


def test_a_lone_hybrid_port_correction_is_the_closed_form(hybrid_couplings):
    # W = h hᵀ / (hᵀ e) - h_s h_sᵀ / (h_sᵀ e) (Palace's W_full - W_scalar)
    model, operator, surface = hybrid_couplings["lone"]
    _, _, (e,) = _robin_and_traces(model, operator, surface)
    (port,) = surface.ports
    u = surface.vectors
    m = u.shape[1] // 2
    coords = port.sweep.coords[0]
    h = u @ port.phi[0]
    gamma = port.sweep.gamma[0] * port.scale
    h_s = u @ np.concatenate([gamma * coords[:m], np.zeros(m)])
    # on random vectors: W itself is a dense ndof x ndof
    x = np.random.default_rng(0).standard_normal((u.shape[0], 6))
    closed = np.outer(h, h @ x) / (h @ e) - np.outer(h_s, h_s @ x) / (h_s @ e)
    c = surface.correction_coefficients(2 * math.pi * HYBRID_SHARED_F[0])
    w = u @ (c @ (u.T @ x))
    # measured 1.7e-15
    assert np.linalg.norm(w - closed) <= 1e-14 * np.linalg.norm(closed)


# --- warnings: skipped port modes and cross-projections -----------------------

PORT_MODE_WARNINGS = (gfem.SkippedPortModeWarning, gfem.PortModeProjectionWarning)


def _silent(model, problem, options):
    """Solve with both port mode warnings turned into errors."""
    with warnings.catch_warnings():
        for category in PORT_MODE_WARNINGS:
            warnings.simplefilter("error", category)
        return gfem.solve(model, problem, options=options)


def plate_modes(mesh, modes):
    """The parallel plate with a wave port per mode on each end, no paths."""
    return gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("plates")],
        ports=[
            gfem.WavePort(f"p{end}_m{mode}", f"port{end}", mode)
            for end in (1, 2)
            for mode in modes
        ],
    )


def test_a_lone_higher_mode_warns_that_the_modes_below_it_are_skipped(plate_mesh):
    with pytest.warns(gfem.SkippedPortModeWarning) as record:
        result = gfem.solve(
            plate_modes(plate_mesh, (2,)), gfem.Driven(F2), options=OPTIONS
        )
    messages = [str(w.message) for w in record]
    assert len(messages) == 2  # once per surface
    for end, message in zip((1, 2), messages, strict=True):
        assert f"'p{end}_m2'" in message
        assert f"'port{end}'" in message
        assert "skips mode 1," in message
        assert "partly reflected" in message
    # a warning, not an error: the lone mode 2 is solved
    gamma = np.asarray(result.port_modes["p1_m2"].gamma)
    np.testing.assert_allclose(gamma, 1j * np.array([beta2(f) for f in F2]), rtol=3e-6)


def test_a_gap_between_requested_modes_warns_skipped_port_mode(plate_mesh):
    # modes 1, 2 and 3 propagate at 4 THz; 1 and 3 are requested
    with pytest.warns(gfem.SkippedPortModeWarning) as record:
        gfem.solve(
            plate_modes(plate_mesh, (1, 3)), gfem.Driven(4.0e12), options=OPTIONS
        )
    message = str(record[0].message)
    assert len(record) == 2  # once per surface
    for name in ("'port1'", "'p1_m1'", "'p1_m3'", "skips mode 2,"):
        assert name in message


def test_a_lone_first_mode_never_warns_though_mode_2_propagates(plate_mesh):
    # mode 2 propagates at 2.0 and 2.5 THz
    _silent(line(plate_mesh), gfem.Driven(F), OPTIONS)


# A rectangular guide (a = 4 mm, b = 1.5 mm, 2 mm long) with an eps_r = 4
# block in one quadrant of its cross-section, eps_r = 2 elsewhere: no
# symmetry, so one mode's conjugated S functional reads the other at the
# order of the block's loss. (The hybrid guide's modes 1 and 2 stay
# orthogonal: measured cross-projection <= 5e-5 up to tan_delta = 0.45,
# where its modes swap ranks.) Modes 1 and 2 propagate at 50 and 55 GHz,
# mode 3 is cut off (PortMode solves: beta_2 = 363 and 832 rad/m).
QUADRANT_F = (50e9, 55e9)
QUADRANT_A, QUADRANT_B, QUADRANT_L = 4.0, 1.5, 2.0  # mm
QUADRANT_OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")


def _quadrant_mesh(tmp_path_factory, a, b):
    """An ``a`` x ``b`` mm guide, QUADRANT_L long, its block in one quadrant."""
    eps = 1e-9

    def volume(c):
        return "block" if c[1] < a / 2 and c[2] < b / 2 else "fill"

    def surface(c):
        x, y, z = c
        if abs(x) < eps:
            return "port1"
        if abs(x - QUADRANT_L) < eps:
            return "port2"
        if min(y, z, a - y, b - z) < eps:
            return "walls"
        return ""

    edges = (
        np.linspace(0, QUADRANT_L, 5),
        np.linspace(0, a, 9),
        np.linspace(0, b, 5),
    )
    path = tmp_path_factory.mktemp("quadrant") / "quadrant.msh"
    sm.write_msh22(path, sm.structured(edges, volume, surface))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def quadrant_mesh(tmp_path_factory):
    return _quadrant_mesh(tmp_path_factory, QUADRANT_A, QUADRANT_B)


def quadrant_guide(mesh, tan_delta, names=HYBRID_SHARED):
    return gfem.Model(
        mesh,
        materials={
            "block": gfem.Material(eps_r=4.0, tan_delta=tan_delta),
            "fill": gfem.Material(eps_r=2.0),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[
            gfem.WavePort(name, f"port{name[1]}", mode=int(name[-1])) for name in names
        ],
    )


def test_a_lossless_shared_port_surface_warns_nothing(
    plate_mesh, hybrid_guide, quadrant_mesh
):
    _silent(shared_line(plate_mesh), gfem.Driven(F2), SHARED_OPTIONS)
    options = gfem.SolverOptions(order=3, linear_solver="direct")
    _silent(shared_hybrid(hybrid_guide), gfem.Driven(HYBRID_SHARED_F), options)
    # cross-projection measured 6.8e-15
    lossless = quadrant_guide(quadrant_mesh, 0.0)
    _silent(lossless, gfem.Driven(QUADRANT_F), QUADRANT_OPTIONS)


def test_a_lossy_shared_port_surface_warns_its_cross_projection(quadrant_mesh):
    model = quadrant_guide(quadrant_mesh, 0.02)
    with pytest.warns(gfem.PortModeProjectionWarning) as record:
        gfem.solve(model, gfem.Driven(QUADRANT_F), options=QUADRANT_OPTIONS)
    messages = [str(w.message) for w in record]
    assert len(messages) == 2  # once per surface
    for end, message in zip((1, 2), messages, strict=True):
        for name in (f"'p{end}_m1'", f"'p{end}_m2'", f"'port{end}'", "biorthogonal"):
            assert name in message
        assert "at 50 GHz" in message  # where it is largest
        value = float(re.search(r"cross-projection of ([0-9.e+-]+)", message)[1])
        # measured 2.67e-2 (1.34 tan_delta): the order of the block's loss
        assert 2e-2 < value < 3.5e-2


# A narrower quadrant guide (2 x 1 mm): mode 1 propagates at 55 and 65 GHz,
# mode 2 is evanescent at both. Its eigenpairs there keep a small Re beta:
# beta = 6.8 - 2541j 1/m at 55 GHz with loss, and a complex pair
# beta = 4.0 +- 4059j 1/m without.
NARROW_F = (55e9, 65e9)


@pytest.fixture(scope="module")
def narrow_quadrant_mesh(tmp_path_factory):
    return _quadrant_mesh(tmp_path_factory, 2.0, 1.0)


@pytest.mark.parametrize("tan_delta", [0.0, 0.02])
def test_an_evanescent_mode_of_a_shared_port_surface_raises_its_cutoff(
    narrow_quadrant_mesh, tan_delta
):
    model = quadrant_guide(narrow_quadrant_mesh, tan_delta)
    with pytest.raises(
        gfem.PortModeError,
        match=r"WavePort 'p1_m2' on surface 'port1' \(shared with 'p1_m1'\): mode 2 "
        r"does not propagate in the band 55-65 GHz .* the band is below the mode's "
        r"cutoff\. .*or drop that wave port\.",
    ):
        gfem.solve(model, gfem.Driven(NARROW_F), options=QUADRANT_OPTIONS)


@pytest.mark.parametrize("tan_delta", [0.0, 0.02])
def test_an_evanescent_lone_mode_raises_its_cutoff(narrow_quadrant_mesh, tan_delta):
    model = quadrant_guide(narrow_quadrant_mesh, tan_delta, names=("p1_m2", "p2_m2"))
    with pytest.raises(
        gfem.PortModeError,
        match=r"^WavePort 'p1_m2': mode 2 does not propagate in the band 55-65 GHz "
        r".* the band is below the mode's cutoff\. Start the band above the cutoff, "
        r"pick a lower mode, or use a LumpedPort\.$",
    ):
        gfem.solve(model, gfem.Driven(NARROW_F), options=QUADRANT_OPTIONS)


# --- a mirror half model: ranks count the modes the symmetry plane admits -------


@pytest.fixture(scope="module")
def half_plate(tmp_path_factory):
    # x in [0, W/2] with a PEC mirror at x = W/2: the plate's TEM mode is
    # gone, so mode 1 of the half model is cos(pi x / W), the full plate's
    # mode 2
    w, d, length = W / 2 * 1e6, D * 1e6, L * 1e6  # um
    eps = 1e-9

    def surface(c):
        x, y, z = c
        if abs(y) < eps or abs(y - d) < eps:
            return "plates"
        if abs(x - w) < eps:
            return "mirror"
        if abs(z) < eps:
            return "port1"
        if abs(z - length) < eps:
            return "port2"
        return ""

    cells = (4, 2, 4)
    edges = tuple(
        np.linspace(0, n, k + 1) for n, k in zip((w, d, length), cells, strict=True)
    )
    path = tmp_path_factory.mktemp("half") / "half_plate.msh"
    sm.write_msh22(path, sm.structured(edges, lambda _: "fill", surface))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={"fill": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("plates"), gfem.PEC("mirror")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def test_a_half_model_s_mode_1_is_the_first_mode_its_plane_admits(half_plate):
    # a lone mode=1: the full plate's mode 2 is the half model's first
    result = _silent(half_plate, gfem.Driven(F2), SHARED_OPTIONS)
    expected = np.array([beta2(f) for f in F2])
    gamma = np.asarray(result.port_modes["p1"].gamma)
    np.testing.assert_allclose(gamma, 1j * expected, rtol=6e-8)  # measured 2.1e-8
    s21 = np.asarray(result.s.values)[:, 1, 0]
    # no voltage path: the mode's sign is a mesh-dependent rule
    transmitted = np.exp(-1j * expected * L)
    error = np.minimum(np.abs(s21 - transmitted), np.abs(s21 + transmitted))
    assert error.max() < 6e-7  # measured 1.8e-7


# --- a shared port surface whose modes overlap on the port face -----------------
# The gradient matrix's hybrid line (tests/gradient_cases.py): a quadrant
# block in the leads makes modes 1 and 2 hybrid in both y and z, so they are
# not orthogonal in the port face's mass product and the Gram matrix G of the
# modal correction is used in full. (On the hybrid guide above G is diagonal
# to 2.6e-5.) Its core carries the leads' quadrant: a uniform line.

HYBRID_LINE_F = (55e9, 65e9)
HYBRID_LINE_LENGTH = 4e-3


@pytest.fixture(scope="module")
def hybrid_line_model(tmp_path_factory):
    mesh = hybrid_mesh(tmp_path_factory.mktemp("hybrid_line"))
    core = gfem.MaterialField(eps_r=hybrid_leads_core(mesh))
    return hybrid_line(mesh).with_material("core", core)


def test_a_shared_surface_of_overlapping_modes_has_a_full_gram_matrix(
    hybrid_line_model,
):
    _, surface = _port1_surface(hybrid_line_model, HYBRID_LINE_F, order=2)
    assert [c.port.name for c in surface.ports] == ["p1_m1", "p1_m2"]
    coupling = [_gram_coupling(surface, k) for k in range(len(HYBRID_LINE_F))]
    # measured 0.043 (55 GHz), 0.035 (65 GHz), the same at order 3: three
    # orders above the hybrid guide's, far above the reflection floor (6e-4)
    assert min(coupling) > 0.02


def test_a_shared_surface_of_overlapping_modes_is_reflectionless(hybrid_line_model):
    options = gfem.SolverOptions(linear_solver="direct")
    result = gfem.solve(hybrid_line_model, gfem.Driven(HYBRID_LINE_F), options=options)
    s = np.asarray(result.s.values)
    pairs = [(2, 0), (0, 2), (3, 1), (1, 3)]
    transmitted, rest = _entries(s, pairs)
    errors = []
    for (_, j), t in zip(pairs, transmitted, strict=True):
        gamma = np.asarray(result.port_modes[HYBRID_PORTS[j]].gamma)
        expected = np.exp(-gamma * HYBRID_LINE_LENGTH)
        # no voltage path: the sign of each mode is a mesh-dependent rule
        errors.append(np.minimum(np.abs(t - expected), np.abs(t + expected)).max())
    reciprocity = np.abs(s - np.swapaxes(s, 1, 2)).max()
    # measured (order 2, 8 x 8 x 4 cells): mode 1 2.2e-3, mode 2 1.4e-3 (at
    # 65 GHz; 2.2e-5 and 9.1e-6 at order 3: the mesh's error)
    assert max(errors) < 6e-3
    assert rest < 2e-3  # reflections and cross-mode entries, measured 6.2e-4
    assert reciprocity < 1e-8  # measured 2.3e-15
