"""Boundary conditions and ports: value classes that validate their physics."""

import jax
import numpy as np
import pytest

import gfem


def test_simple_boundaries_name_one_surface():
    assert gfem.PEC("ground").surface == "ground"
    assert gfem.PMC("sym").surface == "sym"
    with pytest.raises(TypeError, match="surface"):
        gfem.PEC(("a", "b"))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("order", [1, 2])
def test_absorbing_order_is_one_or_two(order):
    assert gfem.Absorbing("outer", order=order).order == order


def test_absorbing_rejects_other_orders():
    with pytest.raises(ValueError, match="order"):
        gfem.Absorbing("outer", order=3)  # ty: ignore[invalid-argument-type]


def test_conductor_takes_electrical_conductivity_and_thickness():
    c = gfem.Conductor("metal", conductivity=5.8e7, thickness=2e-6)
    assert (c.conductivity, c.thickness, c.fill) == (5.8e7, 2e-6, None)
    assert gfem.Conductor("metal", conductivity=5.8e7).thickness is None
    with pytest.raises(ValueError, match="conductivity"):
        gfem.Conductor("metal", conductivity=0.0)
    with pytest.raises(ValueError, match="thickness"):
        gfem.Conductor("metal", conductivity=1e7, thickness=-1e-6)


def test_impedance_needs_at_least_one_positive_element():
    z = gfem.Impedance("sheet", inductance=1e-12)
    assert (z.resistance, z.inductance, z.capacitance, z.fill) == (
        None,
        1e-12,
        None,
        None,
    )
    with pytest.raises(ValueError, match="resistance, inductance or capacitance"):
        gfem.Impedance("sheet")
    with pytest.raises(ValueError, match="resistance"):
        gfem.Impedance("sheet", resistance=-50.0)


def test_fill_takes_a_sheet_field_not_an_array():
    with pytest.raises(TypeError, match="SheetField"):
        gfem.Conductor("metal", conductivity=1e7, fill=np.ones(3))
    with pytest.raises(TypeError, match="SheetField"):
        gfem.Impedance("sheet", resistance=1.0, fill=np.ones(3))


def test_boundary_parameters_are_not_differentiable():
    def objective(sigma):
        return gfem.Conductor("metal", conductivity=sigma).conductivity

    with pytest.raises(gfem.NonDifferentiableError, match=r"Conductor\.conductivity"):
        jax.grad(objective)(1e7)


def test_boundaries_are_hashable_values_with_no_leaves():
    for b in (
        gfem.PEC("a"),
        gfem.PMC("a"),
        gfem.Absorbing("a"),
        gfem.Conductor("a", conductivity=1.0),
        gfem.Impedance("a", resistance=1.0),
    ):
        hash(b)
        assert jax.tree.leaves(b) == []
        assert b == jax.tree.map(lambda x: x, b)


# --- ports ----------------------------------------------------------------------


def test_lumped_port_defaults_to_a_50_ohm_resistor():
    p = gfem.LumpedPort("p1", "port1", direction="+x")
    assert p.name == "p1"
    assert p.surface == "port1"
    assert p.rlc == (50.0, None, None)
    assert p.elements == (("port1", (1.0, 0.0, 0.0)),)


def test_lumped_port_impedance_is_a_resistive_shortcut():
    assert gfem.LumpedPort("p", "s", "-Y", impedance=25.0).rlc == (25.0, None, None)
    p = gfem.LumpedPort("p", "s", "+z", resistance=50.0, inductance=1e-9)
    assert p.rlc == (50.0, 1e-9, None)
    assert p.impedance is None
    with pytest.raises(ValueError, match="impedance"):
        gfem.LumpedPort("p", "s", "+z", impedance=50.0, resistance=50.0)
    with pytest.raises(ValueError, match="capacitance"):
        gfem.LumpedPort("p", "s", "+z", capacitance=0.0)


def test_multi_element_lumped_port_has_one_direction_per_element():
    p = gfem.LumpedPort("p1", ["P1_E0", "P1_E1"], direction=["+Y", "-Y"])
    assert p.surface == ("P1_E0", "P1_E1")
    assert p.elements == (
        ("P1_E0", (0.0, 1.0, 0.0)),
        ("P1_E1", (0.0, -1.0, 0.0)),
    )
    with pytest.raises(ValueError, match="one direction per element"):
        gfem.LumpedPort("p1", ["P1_E0", "P1_E1"], direction="+Y")
    with pytest.raises(ValueError, match="one direction per element"):
        gfem.LumpedPort("p1", ["P1_E0", "P1_E1"], direction=["+Y"])
    with pytest.raises(ValueError, match="twice"):
        gfem.LumpedPort("p1", ["P1_E0", "P1_E0"], direction=["+Y", "-Y"])


def test_lumped_port_direction_may_be_a_vector():
    p = gfem.LumpedPort("p", "s", direction=(3.0, 4.0, 0.0))
    np.testing.assert_allclose(p.elements[0][1], (0.6, 0.8, 0.0))
    with pytest.raises(ValueError, match="direction"):
        gfem.LumpedPort("p", "s", direction="up")
    with pytest.raises(ValueError, match="direction"):
        gfem.LumpedPort("p", "s", direction=(0.0, 0.0, 0.0))


def test_wave_port_holds_mode_offset_voltage_path_and_port_edge_pec():
    w = gfem.WavePort(
        "w1",
        "port_a",
        mode=2,
        offset=10e-6,
        voltage_path=[(0, 0, 0), (0, 1e-5, 0)],
        pec_edges=["side_metal"],
    )
    assert (w.mode, w.offset) == (2, 10e-6)
    assert w.voltage_path == ((0.0, 0.0, 0.0), (0.0, 1e-5, 0.0))
    assert w.pec_edges == ("side_metal",)
    plain = gfem.WavePort("w1", "port_a")
    assert (plain.mode, plain.offset, plain.voltage_path, plain.pec_edges) == (
        1,
        0.0,
        None,
        (),
    )
    with pytest.raises(ValueError, match="mode"):
        gfem.WavePort("w", "s", mode=0)
    with pytest.raises(ValueError, match="offset"):
        gfem.WavePort("w", "s", offset=-1e-6)
    with pytest.raises(ValueError, match="voltage_path"):
        gfem.WavePort("w", "s", voltage_path=[(0, 0, 0)])


def test_port_names_must_be_non_empty_strings():
    with pytest.raises(ValueError, match="name"):
        gfem.LumpedPort("", "s", "+x")
    with pytest.raises(TypeError, match="name"):
        gfem.WavePort(1, "s")  # ty: ignore[invalid-argument-type]
