"""`gfem.characteristic_impedance`: Z_PV of a port mode along any path.

The parallel-plate fixture's TEM mode is exact in the discrete space, so
its impedance eta / sqrt(eps_r) * d / w holds to solver precision along any
path from one plate to the other.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

DATA = Path(__file__).parent / "data"
ETA0 = 376.730313412
W, D = 40e-6, 10e-6  # plate width and gap, m


@pytest.fixture(scope="module")
def mode():
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("plates")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    return gfem.solve(model, gfem.PortMode("p1", [10e9, 40e9]))


def test_impedance_along_a_plate_to_plate_path_is_eta_d_over_w(mode):
    assert np.isnan(np.asarray(mode.Z_PV)).all()  # the port declares no path
    z = gfem.characteristic_impedance(mode, [(W / 2, 0.0, 0.0), (W / 2, D, 0.0)])
    assert z.shape == (2,)
    np.testing.assert_allclose(z, ETA0 / 2.0 * D / W, rtol=1e-8)


def test_any_path_between_the_plates_gives_the_same_impedance(mode):
    # a slanted, two-segment path, run from the top plate down
    path = [(0.9 * W, D, 0.0), (0.5 * W, 0.4 * D, 0.0), (0.1 * W, 0.0, 0.0)]
    z = gfem.characteristic_impedance(mode, path)
    np.testing.assert_allclose(z, ETA0 / 2.0 * D / W, rtol=1e-8)


def test_matches_the_port_voltage_path_impedance():
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    path = ((W / 3, 0.0, 0.0), (W / 3, D, 0.0))
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("plates")],
        ports=[gfem.WavePort("p1", "port1", voltage_path=path)],
    )
    result = gfem.solve(model, gfem.PortMode("p1", 20e9))
    np.testing.assert_allclose(
        gfem.characteristic_impedance(result, path), result.Z_PV, rtol=1e-13
    )


@pytest.mark.parametrize(
    ("path", "match"),
    [
        ([(0.0, 0.0, 0.0)], "two or more points"),
        ([(0.0, 0.0), (1.0, 1.0)], "two or more points"),
        ([(W / 2, 0.0, 0.0), (W / 2, D, 1e-3)], "leaves the port plane"),
        ([(W / 2, 0.0, 0.0), (np.nan, D, 0.0)], "finite"),
    ],
)
def test_bad_paths_raise(mode, path, match):
    with pytest.raises(ValueError, match=match):
        gfem.characteristic_impedance(mode, path)


def test_needs_a_port_mode_result(mode):
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.characteristic_impedance(mode.e_field, [(0, 0, 0), (0, D, 0)])


def test_traced_paths_raise(mode):
    with pytest.raises(ValueError, match="fixed numbers"):
        jax.jit(lambda p: gfem.characteristic_impedance(mode, p))(
            jnp.array([(W / 2, 0.0, 0.0), (W / 2, D, 0.0)])
        )
