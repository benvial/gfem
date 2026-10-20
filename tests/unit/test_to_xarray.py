"""`.to_xarray()`: results as labelled `xarray.Dataset` objects (``xarray`` extra).

Each result's arrays keep their values and get named dimensions and
coordinates (frequencies in Hz, port, terminal, region and mode labels) and
a ``units`` attribute.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

xr = pytest.importorskip("xarray")

F = (1e9, 2e9, 3e9)


@pytest.fixture
def network():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, 2, 1)) + 1j * rng.normal(size=(3, 2, 1))
    return gfem.NetworkParameters(
        values=jnp.asarray(values),
        z_ref=jnp.array([[50.0, 25.0]] * 3),
        frequency=F,
        ports=("p1", "p2"),
        excited=("p1",),
    )


@pytest.fixture
def capacitance():
    c = jnp.array([[2e-15, -1e-15], [-1e-15, 3e-15]])
    return gfem.ElectrostaticResult(
        capacitance=c,
        terminals=("top", "mid"),
        energy={"oxide": jnp.array([1e-15, 2e-15]), "air": jnp.array([0.5e-15, 1e-15])},
    )


@pytest.fixture
def modes():
    return gfem.EigenmodeResult(
        omega=jnp.array([2 * np.pi * 5e9 * (1 + 1e-4j), 2 * np.pi * 7e9 * (1 + 2e-4j)]),
        error=jnp.array([1e-12, 2e-12]),
        q_port={"p1": jnp.array([1e4, 2e4])},
        participation={"j1": jnp.array([0.9, 0.1])},
        bulk_participation={"si": jnp.array([0.8, 0.7]), "air": jnp.array([0.2, 0.3])},
    )


@pytest.fixture
def port_mode():
    return gfem.PortModeResult(
        port="p1",
        mode=1,
        frequency=jnp.asarray(F),
        gamma=jnp.array([1j * 40.0, 1j * 80.0, 1j * 120.0]) + 0.1,
        Z_PV=jnp.array([50.0, 50.1, 50.2]),
        residual=jnp.array([1e-14, 1e-9, 1e-14]),
        solved=jnp.array([True, False, True]),
    )


def test_network_parameters_label_s_by_frequency_and_ports(network):
    ds = network.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert ds["s"].dims == ("frequency", "port", "excited")
    np.testing.assert_array_equal(ds["frequency"], F)
    assert list(ds["port"].values) == ["p1", "p2"]
    assert list(ds["excited"].values) == ["p1"]
    np.testing.assert_array_equal(
        ds["s"].sel(port="p2", excited="p1"), network["p2", "p1"]
    )
    assert ds["z_ref"].dims == ("frequency", "port")
    assert float(ds["z_ref"].sel(port="p2", frequency=2e9)) == 25.0
    assert ds["frequency"].attrs["units"] == "Hz"
    assert ds["z_ref"].attrs["units"] == "ohm"


def test_capacitance_is_labelled_by_terminal_and_excitation(capacitance):
    ds = capacitance.to_xarray()
    assert ds["capacitance"].dims == ("terminal", "excitation")
    assert float(ds["capacitance"].sel(terminal="mid", excitation="top")) == -1e-15
    assert ds["capacitance"].attrs["units"] == "F"
    assert ds["energy"].dims == ("region", "excitation")
    assert float(ds["energy"].sel(region="air", excitation="mid")) == 1e-15
    assert ds["energy"].attrs["units"] == "J"


def test_eigenmode_scalars_are_labelled_by_mode(modes):
    ds = modes.to_xarray()
    np.testing.assert_array_equal(ds["mode"], [0, 1])
    for name in ("omega", "frequency", "q", "error", "cluster"):
        assert ds[name].dims == ("mode",), name
    np.testing.assert_allclose(ds["frequency"], modes.frequency)
    np.testing.assert_allclose(ds["q"], modes.q)
    assert ds["frequency"].attrs["units"] == "Hz"
    assert ds["omega"].attrs["units"] == "rad/s"
    # port-keyed outputs share one port axis; a port missing one is NaN
    assert list(ds["port"].values) == ["p1", "j1"]
    assert float(ds["q_port"].sel(port="p1", mode=1)) == 2e4
    assert np.isnan(ds["q_port"].sel(port="j1")).all()
    assert float(ds["participation"].sel(port="j1", mode=0)) == 0.9
    assert ds["bulk_participation"].dims == ("mode", "region")
    assert float(ds["bulk_participation"].sel(region="air", mode=1)) == 0.3


def test_port_mode_is_labelled_by_frequency(port_mode):
    ds = port_mode.to_xarray()
    for name in ("gamma", "n_eff", "Z_PV", "residual", "solved"):
        assert ds[name].dims == ("frequency",), name
    np.testing.assert_array_equal(ds["frequency"], F)
    np.testing.assert_allclose(ds["n_eff"], port_mode.n_eff)
    assert ds.attrs["port"] == "p1"
    assert ds.attrs["mode"] == 1
    assert ds["gamma"].attrs["units"] == "1/m"


def test_to_xarray_without_xarray_names_the_extra(network, monkeypatch):
    # a None entry in sys.modules makes `import xarray` raise ImportError
    monkeypatch.setitem(sys.modules, "xarray", None)
    with pytest.raises(ImportError, match=r"gfem\[xarray\]"):
        network.to_xarray()


def test_to_xarray_needs_concrete_values(capacitance):
    def export(c):
        result = gfem.ElectrostaticResult(capacitance=c, terminals=("top", "mid"))
        result.to_xarray()
        return c.sum()

    with pytest.raises(TypeError, match="concrete"):
        jax.jit(export)(capacitance.capacitance)
