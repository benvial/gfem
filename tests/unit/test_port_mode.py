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
    assert len(leaves) == 4
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


def test_below_cutoff_raises_port_mode_error(guide_mesh):
    with pytest.raises(gfem.PortModeError, match=r"does not propagate at 5 GHz"):
        gfem.solve(guide(guide_mesh), gfem.PortMode("p1", [10e9, 5e9]))


def test_conductor_on_a_port_edge_asks_for_pec_edges(guide_mesh):
    walls = gfem.Conductor("walls", conductivity=5.8e7)
    with pytest.raises(NotImplementedError, match=r"pec_edges=\('walls',\)"):
        gfem.solve(guide(guide_mesh, walls), gfem.PortMode("p1", 10e9))


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


def test_port_mode_is_not_differentiable(plate_mesh):
    n = len(plate_mesh.elements("fill"))

    def objective(eps):
        model = plate(plate_mesh, gfem.MaterialField(eps_r=jnp.full(n, eps)))
        return jnp.real(gfem.solve(model, gfem.PortMode("p1", 1e9)).gamma[0])

    with pytest.raises(gfem.NonDifferentiableError, match="not differentiable"):
        jax.grad(objective)(4.0)


def test_solve_rejects_unknown_problems_and_options(plate_mesh):
    model = plate(plate_mesh)
    with pytest.raises(TypeError, match="problem type"):
        gfem.solve(model, "PortMode")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="SolverOptions"):
        gfem.solve(model, gfem.PortMode("p1", 1e9), options={"order": 2})  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"gfem\.Model"):
        gfem.solve(None, gfem.PortMode("p1", 1e9))  # ty: ignore[invalid-argument-type]
