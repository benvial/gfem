"""Driven problems with wave ports: `gfem.solve(model, gfem.Driven(...))`.

The parallel-plate fixture carries an exact TEM port mode, and its 3D field
is a plane wave, so a two-port line has S21 = exp(-gamma L) and S11 = 0 up
to the 3D discretization error.
"""

import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

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


def test_conductor_plates_make_a_lossy_passive_reciprocal_line(plate_mesh):
    model = line(plate_mesh).replace(
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
