"""The port mode problem: `gfem.solve(model, gfem.PortMode(...))`.

The parallel-plate fixture carries an exact TEM mode (a uniform field is in
the discrete space), so its propagation constant and impedance are checked
to solver precision. Convergence to analytic waveguide modes is in the
verification tier.
"""

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
EPS0 = 1 / (ETA0 * C0)
W, D = 40e-6, 10e-6  # parallel plate width and gap, m
A, B = 22.86e-3, 10.16e-3  # WR-90, m


@pytest.fixture(scope="module")
def plate_mesh():
    return gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")


@pytest.fixture(scope="module")
def guide_mesh():
    return gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")


def plate(mesh, material=None, *, path=True, **port):
    vp = ((W / 2, 0.0, 0.0), (W / 2, D, 0.0)) if path else None
    return gfem.Model(
        mesh,
        materials={"fill": material or gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("plates")],
        ports=[
            gfem.WavePort("p1", "port1", voltage_path=vp, **port),
            gfem.WavePort("p2", "port2"),
        ],
    )


def guide(mesh, walls=None, **port):
    return gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[walls or gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1", **port)],
    )


# --- the problem type ---------------------------------------------------------


def test_port_mode_takes_one_frequency_or_a_grid():
    assert gfem.PortMode("p1", 5e9).frequency == (5e9,)
    assert gfem.PortMode("p1", np.array([1e9, 2e9])).frequency == (1e9, 2e9)
    assert gfem.PortMode("p1", [1e9], mode=2).mode == 2


@pytest.mark.parametrize("bad", [0.0, -1e9, math.nan, [], [[1e9]], "x"])
def test_port_mode_rejects_bad_frequencies(bad):
    with pytest.raises(ValueError, match="frequency"):
        gfem.PortMode("p1", bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_port_mode_rejects_bad_mode_numbers(bad):
    with pytest.raises(ValueError, match="mode"):
        gfem.PortMode("p1", 1e9, mode=bad)


def test_port_mode_is_frozen():
    problem = gfem.PortMode("p1", 1e9)
    with pytest.raises(AttributeError):
        problem.port = "p2"  # ty: ignore[invalid-assignment]


# --- the parallel-plate TEM mode -------------------------------------------------


def test_tem_mode_of_a_parallel_plate_is_exact(plate_mesh):
    f = np.array([1e9, 20e9, 50e9])
    result = gfem.solve(plate(plate_mesh), gfem.PortMode("p1", f))
    k0 = 2 * np.pi * f / C0
    np.testing.assert_allclose(np.asarray(result.gamma), 1j * k0 * 2.0, rtol=1e-8)
    np.testing.assert_allclose(np.asarray(result.n_eff), 2.0, rtol=1e-8)
    np.testing.assert_allclose(np.asarray(result.frequency), f)
    assert result.port == "p1"
    assert result.mode == 1
    assert np.asarray(result.residual).max() < 1e-7


def test_z_pv_of_a_parallel_plate_is_eta_d_over_w(plate_mesh):
    result = gfem.solve(plate(plate_mesh), gfem.PortMode("p1", [10e9, 40e9]))
    np.testing.assert_allclose(np.asarray(result.Z_PV), ETA0 / 2.0 * D / W, rtol=1e-8)


def test_z_pv_is_nan_without_a_voltage_path(plate_mesh):
    result = gfem.solve(plate(plate_mesh, path=False), gfem.PortMode("p1", 10e9))
    assert np.isnan(np.asarray(result.Z_PV)).all()


def test_mode_field_carries_one_watt_and_points_along_the_voltage_path(plate_mesh):
    result = gfem.solve(plate(plate_mesh), gfem.PortMode("p1", [10e9, 30e9]))
    e = np.asarray(result.e_field.at([[W / 2, D / 2, 0.0], [3e-6, 8e-6, 0.0]]))
    assert e.shape == (2, 2, 3)
    # 1 W = |E|^2 W D / (2 Z_w), with Z_w = eta0 / sqrt(eps_r)
    expected = math.sqrt(2 * ETA0 / 2.0 / (W * D))
    np.testing.assert_allclose(e[..., 1], expected, rtol=1e-8)
    np.testing.assert_allclose(e[..., [0, 2]], 0.0, atol=1e-8 * expected)
    assert result.e_field.unit == "V/m"
    assert result.e_field.normal == (0.0, 0.0, 1.0)


def test_mode_field_is_zero_off_the_cross_section(plate_mesh):
    result = gfem.solve(plate(plate_mesh), gfem.PortMode("p1", 10e9))
    e = np.asarray(result.e_field.at([1e-3, 5e-6, 0.0]))
    assert e.shape == (1, 3)
    np.testing.assert_array_equal(e, 0.0)


def test_mode_field_rejects_points_off_the_port_plane(plate_mesh):
    result = gfem.solve(plate(plate_mesh), gfem.PortMode("p1", 10e9))
    with pytest.raises(ValueError, match="not on the plane"):
        result.e_field.at([W / 2, D / 2, 1e-6])


def test_a_port_facing_minus_z_has_the_same_mode(plate_mesh):
    model = plate(plate_mesh)
    result = gfem.solve(model, gfem.PortMode("p2", 10e9))
    np.testing.assert_allclose(
        np.asarray(result.gamma), 2j * 2 * np.pi * 10e9 / C0, rtol=1e-8
    )
    assert result.e_field.normal == (0.0, 0.0, -1.0)
    e = np.asarray(result.e_field.at([W / 2, D / 2, 20e-6]))
    assert abs(e[0, 1]) > 0


def test_lossy_dielectric_gives_the_complex_propagation_constant(plate_mesh):
    tan_delta = 0.02
    model = plate(plate_mesh, gfem.Material(eps_r=4.0, tan_delta=tan_delta))
    f = 10e9
    result = gfem.solve(model, gfem.PortMode("p1", f))
    k0 = 2 * np.pi * f / C0
    expected = 1j * k0 * np.sqrt(4.0 * (1 - 1j * tan_delta))
    np.testing.assert_allclose(np.asarray(result.gamma)[0], expected, rtol=1e-8)
    assert np.asarray(result.gamma)[0].real > 0  # attenuates into the domain
    assert np.asarray(result.n_eff)[0].imag < 0  # e^{+iwt}: n' - i n''


def test_conductivity_enters_the_permittivity(plate_mesh):
    sigma, f = 0.5, 5e9
    model = plate(plate_mesh, gfem.Material(eps_r=4.0, conductivity=sigma))
    result = gfem.solve(model, gfem.PortMode("p1", f))
    omega = 2 * np.pi * f
    eps = 4.0 - 1j * sigma / (omega * EPS0)
    expected = 1j * omega / C0 * np.sqrt(eps)
    np.testing.assert_allclose(np.asarray(result.gamma)[0], expected, rtol=1e-8)


def test_a_material_field_sets_the_cross_section_permittivity(plate_mesh):
    n = len(plate_mesh.elements("fill"))
    field = gfem.MaterialField(eps_r=jnp.full(n, 9.0))
    result = gfem.solve(plate(plate_mesh, field), gfem.PortMode("p1", 10e9))
    np.testing.assert_allclose(np.asarray(result.n_eff), 3.0, rtol=1e-8)


def test_magnetic_material_scales_the_index_and_impedance(plate_mesh):
    model = plate(plate_mesh, gfem.Material(eps_r=4.0, mu_r=2.0))
    result = gfem.solve(model, gfem.PortMode("p1", 10e9))
    np.testing.assert_allclose(np.asarray(result.n_eff), math.sqrt(8.0), rtol=1e-8)
    z_w = ETA0 * math.sqrt(2.0 / 4.0)
    np.testing.assert_allclose(np.asarray(result.Z_PV), z_w * D / W, rtol=1e-8)


def test_anisotropic_material_uses_the_normal_and_in_plane_components(plate_mesh):
    # TEM along z with E along y sees eps_yy; mu_xx (H along x)
    material = gfem.Material(eps_r=(2.0, 4.0, 9.0), mu_r=(1.5, 1.0, 1.0))
    result = gfem.solve(plate(plate_mesh, material), gfem.PortMode("p1", 10e9))
    np.testing.assert_allclose(
        np.asarray(result.n_eff), math.sqrt(4.0 * 1.5), rtol=1e-8
    )


def test_anisotropy_without_an_axis_along_the_normal_raises(plate_mesh):
    s = math.sqrt(0.5)
    material = gfem.Material(
        eps_r=(2.0, 4.0, 9.0), axes=((1, 0, 0), (0, s, s), (0, -s, s))
    )
    with pytest.raises(ValueError, match="principal axis along the port normal"):
        gfem.solve(plate(plate_mesh, material), gfem.PortMode("p1", 10e9))


def test_the_tem_mode_has_a_voltage_of_positive_real_part(plate_mesh):
    flipped = ((W / 2, D, 0.0), (W / 2, 0.0, 0.0))
    up = gfem.solve(plate(plate_mesh), gfem.PortMode("p1", 10e9))
    model = plate(plate_mesh, path=False)
    model = model.replace(
        ports=[gfem.WavePort("p1", "port1", voltage_path=flipped), model.ports[1]]
    )
    down = gfem.solve(model, gfem.PortMode("p1", 10e9))
    point = [W / 2, D / 2, 0.0]
    e_up = np.asarray(up.e_field.at(point))[0, 1]
    e_down = np.asarray(down.e_field.at(point))[0, 1]
    np.testing.assert_allclose(e_down, -e_up, rtol=1e-10)


def test_result_is_a_pytree_of_its_arrays(plate_mesh):
    result = gfem.solve(plate(plate_mesh), gfem.PortMode("p1", [1e9, 2e9]))
    leaves = jax.tree_util.tree_leaves(result)
    assert len(leaves) == 5  # frequency, gamma, Z_PV, residual, solved
    assert "PortModeResult" in repr(result)
    assert "Z_PV" in repr(result)


# --- waveguide modes -----------------------------------------------------------


def test_the_second_mode_is_te20(guide_mesh):
    f = 16e9
    result = gfem.solve(
        guide(guide_mesh),
        gfem.PortMode("p1", f, mode=2),
        options=gfem.SolverOptions(order=3),
    )
    k0 = 2 * np.pi * f / C0
    beta = np.sqrt(k0**2 - (2 * np.pi / A) ** 2)
    np.testing.assert_allclose(np.asarray(result.gamma).imag, beta, rtol=1e-4)
    assert result.mode == 2


def test_wave_port_mode_is_the_default(guide_mesh):
    model = guide(guide_mesh, mode=2)
    result = gfem.solve(model, gfem.PortMode("p1", 16e9))
    assert result.mode == 2


def test_a_band_below_cutoff_raises_port_mode_error(guide_mesh):
    with pytest.raises(gfem.PortModeError, match=r"does not propagate in the band"):
        gfem.solve(guide(guide_mesh), gfem.PortMode("p1", [5e9, 4e9]))


def test_a_cutoff_inside_the_band_raises_with_its_estimate(guide_mesh):
    # TE10 of WR-90 is cut off at c / (2 a) = 6.557 GHz
    with pytest.raises(
        gfem.PortModeError, match=r"cutoff inside the band 5-10 GHz, at about 6.5[56]"
    ):
        gfem.solve(guide(guide_mesh), gfem.PortMode("p1", np.linspace(5e9, 10e9, 41)))


def test_conductor_walls_give_te10_its_conductor_loss(guide_mesh):
    # Pozar (3.96): alpha_c = R_s (2 b pi^2 + a^3 k^2) / (a^3 b beta k eta0)
    sigma, f = 5.8e7, 10e9
    walls = gfem.Conductor("walls", conductivity=sigma)
    result = gfem.solve(guide(guide_mesh, walls), gfem.PortMode("p1", f))
    k = 2 * np.pi * f / C0
    beta = np.sqrt(k**2 - (np.pi / A) ** 2)
    rs = np.sqrt(2 * np.pi * f * 4e-7 * np.pi / (2 * sigma))
    alpha = rs * (2 * B * np.pi**2 + A**3 * k**2) / (A**3 * B * beta * k * ETA0)
    gamma = complex(np.asarray(result.gamma)[0])
    assert gamma.real == pytest.approx(alpha, rel=2e-3)
    assert gamma.imag == pytest.approx(beta, rel=1e-4)


def test_conductor_plates_follow_the_lossy_line_model(plate_mesh):
    # TEM on a lossy line: gamma^2 = (i w L' + Z') i w C' with Z' = 2 Z_s / W
    sigma, eps, f = 1e6, 4.0, (50e9, 100e9)
    model = gfem.Model(
        plate_mesh,
        materials={"fill": gfem.Material(eps_r=eps)},
        boundaries=[gfem.Conductor("plates", conductivity=sigma)],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    result = gfem.solve(model, gfem.PortMode("p1", f))
    w = 2 * np.pi * np.array(f)
    k = w * np.sqrt(eps) / C0
    zs = (1 + 1j) * np.sqrt(w * 4e-7 * np.pi / (2 * sigma))
    gamma = np.sqrt(-(k**2) + 2j * w * eps * EPS0 * zs / D)
    np.testing.assert_allclose(np.asarray(result.gamma), gamma, rtol=1e-4)


def _jittered(mesh, seed, rel):
    """``mesh`` with in-plane node coordinates moved by ``rel`` (relative)."""
    nodes = np.array(mesh.imported_nodes)
    rng = np.random.default_rng(seed)
    nodes[:, :2] *= 1 + rel * rng.uniform(-1, 1, (len(nodes), 2))
    return jax.tree.unflatten(jax.tree.structure(mesh), [jnp.asarray(nodes)])


def test_the_mode_phase_does_not_jump_under_rounding_level_node_moves(plate_mesh):
    # A lossy mode's phase is fixed by its tangential field on the upper-right
    # quadrant of the port. The structured fixture puts quadrature points on
    # the quadrant's midlines, so a rounding-level move of the nodes must not
    # change which side they fall on (it changed the phase by 6e-6 rad, which
    # broke S12 = S21 on some CPUs).
    def phase(mesh):
        model = plate(mesh, path=False).replace(
            boundaries=[gfem.Conductor("plates", conductivity=5.8e7)]
        )
        result = gfem.solve(model, gfem.PortMode("p1", 1.5e12))
        e = np.asarray(result.e_field.at(np.array([[W / 2, D / 2, 0.0]])))
        return np.angle(e[0, 0, 1])

    reference = phase(plate_mesh)
    for seed in range(3):
        moved = phase(_jittered(plate_mesh, seed, 1e-13))
        assert abs(moved - reference) < 1e-10


def test_impedance_plates_follow_the_lossy_line_model(plate_mesh):
    rs, eps, f = 2.0, 4.0, (50e9, 100e9)
    model = gfem.Model(
        plate_mesh,
        materials={"fill": gfem.Material(eps_r=eps)},
        boundaries=[gfem.Impedance("plates", resistance=rs)],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    result = gfem.solve(model, gfem.PortMode("p1", f))
    w = 2 * np.pi * np.array(f)
    k = w * np.sqrt(eps) / C0
    gamma = np.sqrt(-(k**2) + 2j * w * eps * EPS0 * rs / D)
    np.testing.assert_allclose(np.asarray(result.gamma), gamma, rtol=1e-4)


def test_pec_edges_make_conductor_walls_perfect_in_the_port_problem(guide_mesh):
    walls = gfem.Conductor("walls", conductivity=5.8e7)
    model = guide(guide_mesh, walls, pec_edges=("walls",))
    ref = gfem.solve(guide(guide_mesh), gfem.PortMode("p1", 10e9))
    result = gfem.solve(model, gfem.PortMode("p1", 10e9))
    np.testing.assert_allclose(np.asarray(result.gamma), np.asarray(ref.gamma))


def test_pmc_walls_give_the_dual_of_te10(guide_mesh):
    # PMC walls: TM10 of the dual guide, with the cutoff of the PEC guide's TE10
    f = 10e9
    result = gfem.solve(guide(guide_mesh, gfem.PMC("walls")), gfem.PortMode("p1", f))
    k0 = 2 * np.pi * f / C0
    beta = np.sqrt(k0**2 - (np.pi / A) ** 2)
    np.testing.assert_allclose(np.asarray(result.gamma).imag, beta, rtol=1e-4)


def test_absorbing_edges_are_accepted(guide_mesh):
    model = guide(guide_mesh, gfem.Absorbing("walls"))
    result = gfem.solve(model, gfem.PortMode("p1", 10e9))
    assert np.isfinite(np.asarray(result.gamma)).all()


# --- errors ---------------------------------------------------------------------


def test_unknown_port_lists_the_wave_ports(plate_mesh):
    with pytest.raises(ValueError, match=r"no port named 'p9'.*'p1', 'p2'"):
        gfem.solve(plate(plate_mesh), gfem.PortMode("p9", 1e9))


def test_lumped_port_is_not_a_port_mode_problem(plate_mesh):
    model = gfem.Model(
        plate_mesh,
        materials={"fill": gfem.Material()},
        boundaries=[gfem.PEC("plates")],
        ports=[gfem.LumpedPort("p1", "port1", "+y")],
    )
    with pytest.raises(ValueError, match="needs a WavePort"):
        gfem.solve(model, gfem.PortMode("p1", 1e9))


def test_a_non_planar_port_raises_port_geometry_error(guide_mesh):
    model = gfem.Model(
        guide_mesh,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("port1"), gfem.PEC("port2")],
        ports=[gfem.WavePort("w", "walls")],
    )
    with pytest.raises(gfem.PortGeometryError, match="not planar"):
        gfem.solve(model, gfem.PortMode("w", 10e9))


def test_a_voltage_path_off_the_port_plane_raises(plate_mesh):
    model = plate(plate_mesh, path=False)
    off = ((W / 2, 0.0, 1e-6), (W / 2, D, 1e-6))
    model = model.replace(
        ports=[gfem.WavePort("p1", "port1", voltage_path=off), model.ports[1]]
    )
    with pytest.raises(ValueError, match="voltage_path leaves the port plane"):
        gfem.solve(model, gfem.PortMode("p1", 1e9))


def _fill(mesh, **values):
    n = len(mesh.elements("fill"))
    return gfem.MaterialField(**{k: jnp.full(n, v) for k, v in values.items()})


def test_port_mode_is_not_differentiable(plate_mesh):
    def objective(eps):
        model = plate(plate_mesh, _fill(plate_mesh, eps_r=eps))
        return jnp.real(gfem.solve(model, gfem.PortMode("p1", 1e9)).gamma[0])

    with pytest.raises(gfem.NonDifferentiableError, match="eps_r"):
        jax.grad(objective)(4.0)


def test_port_mode_runs_under_jit_without_its_field(plate_mesh):
    problem = gfem.PortMode("p1", [10e9, 40e9])
    exact = gfem.solve(plate(plate_mesh), problem)

    @jax.jit
    def gamma(eps):
        return gfem.solve(plate(plate_mesh, _fill(plate_mesh, eps_r=eps)), problem)

    traced = gamma(4.0)
    np.testing.assert_allclose(traced.gamma, exact.gamma, rtol=1e-10)
    np.testing.assert_allclose(traced.Z_PV, exact.Z_PV, rtol=1e-10)
    with pytest.raises(TypeError, match="e_field"):
        traced.e_field.at([W / 2, D / 2, 0.0])


def test_port_mode_gradient_of_an_unread_property_is_an_exact_zero(plate_mesh):
    def objective(k):
        model = plate(plate_mesh, _fill(plate_mesh, eps_r=4.0, thermal_conductivity=k))
        return jnp.real(gfem.solve(model, gfem.PortMode("p1", 1e9)).gamma[0])

    assert jax.grad(objective)(1.4) == 0.0


def test_port_mode_stop_gradient_opts_a_read_input_out(plate_mesh):
    def objective(eps):
        fill = _fill(plate_mesh, eps_r=jax.lax.stop_gradient(eps))
        mode = gfem.solve(plate(plate_mesh, fill), gfem.PortMode("p1", 1e9))
        return jnp.real(mode.gamma[0]) / 1e3 + eps

    assert jax.grad(objective)(4.0) == 1.0


@pytest.mark.parametrize(
    "options",
    [
        {"linear_solver": "direct"},
        {"linear_solver": "bddc_gmres"},
        {"linear_solver": "direct", "device": "cuda"},
        {"linear_solver": "petsc"},
        {"eigen_solver": "slepc"},
        {"eigen_solver": "arpack"},
    ],
)
def test_port_mode_rejects_solver_choices_it_does_not_use(plate_mesh, options):
    with pytest.raises(ValueError, match="SuperLU and SciPy on the CPU"):
        gfem.solve(
            plate(plate_mesh),
            gfem.PortMode("p1", 1e9),
            options=gfem.SolverOptions(**options),
        )


def test_solve_rejects_unknown_problems_and_options(plate_mesh):
    model = plate(plate_mesh)
    with pytest.raises(TypeError, match="problem type"):
        gfem.solve(model, "PortMode")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="SolverOptions"):
        gfem.solve(model, gfem.PortMode("p1", 1e9), options={"order": 2})  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"gfem\.Model"):
        gfem.solve(None, gfem.PortMode("p1", 1e9))  # ty: ignore[invalid-argument-type]
