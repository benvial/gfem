"""`gfem.NetworkParameters`: Y, Z, renormalization and Touchstone export.

Expected values come from closed forms: a load's power-wave reflection
(Kurokawa) and a transmission line's ABCD matrix, converted to Z by
Z11 = A / C, Z12 = det / C, Z21 = 1 / C, Z22 = D / C.
"""

import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

F = (1e9, 2e9, 3e9)


def _net(values, z_ref, ports=("p1", "p2"), excited=None, frequency=F):
    values = jnp.asarray(values, complex)
    n_f = values.shape[0]
    return gfem.NetworkParameters(
        values=values,
        z_ref=jnp.asarray(np.broadcast_to(z_ref, (n_f, len(ports)))),
        frequency=tuple(frequency[:n_f]),
        ports=ports,
        excited=ports if excited is None else excited,
    )


def _one_port(load, z_ref):
    """S of a load under power waves with reference impedance ``z_ref``."""
    s = (load - np.conj(z_ref)) / (load + z_ref)
    return _net(np.reshape(s, (-1, 1, 1)), np.reshape(z_ref, (-1, 1)), ports=("p",))


def _line(f, z0=80.0, eps=4.0, length=0.01):
    beta = 2 * math.pi * f / 299_792_458.0 * math.sqrt(eps)
    c, s = math.cos(beta * length), math.sin(beta * length)
    return np.array([[c, 1j * z0 * s], [1j * s / z0, c]])


def _s_from_abcd(m, r):
    (a, b), (c, d) = m
    den = a + b / r + c * r + d
    return np.array(
        [
            [(a + b / r - c * r - d) / den, 2 * (a * d - b * c) / den],
            [2 / den, (-a + b / r - c * r + d) / den],
        ]
    )


def _z_from_abcd(m):
    (a, b), (c, d) = m
    return np.array([[a / c, (a * d - b * c) / c], [1 / c, d / c]])


@pytest.fixture
def line():
    return _net([_s_from_abcd(_line(f), 50.0) for f in F], 50.0)


# --- Y and Z ---------------------------------------------------------------------


def test_z_of_a_loaded_port_is_the_load_impedance_for_real_and_complex_z_ref():
    loads = np.array([20.0 + 5j, 75.0 - 30j, 1e3 + 0j])
    for z_ref in (50.0, 30.0 + 10j):
        z = np.full(3, z_ref)
        net = _one_port(loads, z)
        np.testing.assert_allclose(net.z()[:, 0, 0], loads, rtol=1e-12)
        np.testing.assert_allclose(net.y()[:, 0, 0], 1 / loads, rtol=1e-12)


def test_z_and_y_of_a_line_match_its_abcd_matrix(line):
    expected = np.array([_z_from_abcd(_line(f)) for f in F])
    np.testing.assert_allclose(line.z(), expected, rtol=1e-10)
    np.testing.assert_allclose(line.y(), np.linalg.inv(expected), rtol=1e-10)
    assert line.z().shape == (3, 2, 2)


def test_y_and_z_put_columns_in_port_order_when_excitation_order_differs(line):
    swapped = _net(np.asarray(line.values)[:, :, ::-1], 50.0, excited=("p2", "p1"))
    np.testing.assert_allclose(swapped.z(), line.z(), rtol=1e-12)


def test_y_z_and_renormalize_need_every_port_excited(line):
    partial = _net(np.asarray(line.values)[:, :, :1], 50.0, excited=("p1",))
    for method in (partial.y, partial.z, lambda: partial.renormalize(75.0)):
        with pytest.raises(ValueError, match=r"excite.*'p2'"):
            method()


# --- renormalization --------------------------------------------------------------


def test_renormalize_to_a_new_real_reference_matches_the_line(line):
    new = line.renormalize(75.0)
    expected = np.array([_s_from_abcd(_line(f), 75.0) for f in F])
    np.testing.assert_allclose(new.values, expected, atol=1e-12)
    np.testing.assert_array_equal(new.z_ref, np.full((3, 2), 75.0))
    assert (new.ports, new.excited, new.frequency) == (line.ports, line.ports, F)


def test_renormalize_to_a_complex_reference_gives_the_power_wave_reflection():
    loads = np.array([20.0 + 5j, 75.0 - 30j, 10.0 + 0j])
    new = _one_port(loads, np.full(3, 50.0)).renormalize(40.0 - 25j)
    np.testing.assert_allclose(
        new.values[:, 0, 0], (loads - (40.0 + 25j)) / (loads + 40.0 - 25j), atol=1e-12
    )


def test_renormalize_takes_a_reference_per_port_and_round_trips(line):
    there = line.renormalize([25.0, 100.0 + 20j])
    np.testing.assert_allclose(there.z_ref, np.tile([25.0, 100.0 + 20j], (3, 1)))
    back = there.renormalize(50.0)
    np.testing.assert_allclose(back.values, line.values, atol=1e-12)
    # Z does not depend on the reference
    np.testing.assert_allclose(there.z(), line.z(), rtol=1e-10)


def test_renormalize_refuses_a_passive_reference_without_resistance(line):
    with pytest.raises(ValueError, match="positive real part"):
        line.renormalize(0.0)
    with pytest.raises(ValueError, match="2 ports"):
        line.renormalize([50.0, 50.0, 50.0])


def test_network_functions_differentiate_and_jit(line):
    def objective(values):
        net = _net(values, 50.0)
        return jnp.abs(net.renormalize(75.0).values[0, 1, 0]) + jnp.abs(net.z()).sum()

    grad = jax.grad(objective)(line.values)
    step = 1e-7
    bump = np.zeros((3, 2, 2), complex)
    bump[0, 1, 0] = step
    fd = (objective(line.values + bump) - objective(line.values - bump)) / (2 * step)
    # the real part of JAX's gradient for a complex input is d/d(Re)
    assert float(jnp.real(grad[0, 1, 0])) == pytest.approx(float(fd), rel=1e-6)
    jitted = jax.jit(lambda v, z: _net(v, 50.0).renormalize(z).values)
    np.testing.assert_allclose(
        jitted(line.values, 75.0), line.renormalize(75.0).values, atol=1e-14
    )


# --- Touchstone -------------------------------------------------------------------


def _read_touchstone(path):
    """Minimal Touchstone 1.1 reader: option line, comments, data."""
    comments, option, numbers = [], None, []
    for raw in path.read_text().splitlines():
        line, _, comment = raw.partition("!")
        if comment and not line.strip():
            comments.append(comment.strip())
        if line.strip().startswith("#"):
            option = line.split()
        elif line.strip():
            numbers += [float(x) for x in line.split()]
    return comments, option, np.array(numbers)


def test_touchstone_two_port_uses_column_major_order_and_its_z_ref(line, tmp_path):
    path = line.to_touchstone(tmp_path / "line.s2p")
    assert path == tmp_path / "line.s2p"
    comments, option, data = _read_touchstone(path)
    assert [o.upper() for o in option] == ["#", "HZ", "S", "RI", "R", "50"]
    rows = data.reshape(3, 9)
    np.testing.assert_allclose(rows[:, 0], F, rtol=0)
    pairs = rows[:, 1::2] + 1j * rows[:, 2::2]
    s = np.asarray(line.values)
    # 2-port files list S11 S21 S12 S22
    expected = np.stack([s[:, 0, 0], s[:, 1, 0], s[:, 0, 1], s[:, 1, 1]], axis=1)
    np.testing.assert_allclose(pairs, expected, rtol=1e-14, atol=1e-16)
    text = " ".join(comments)
    assert "p1" in text
    assert "p2" in text
    assert re.search(r"power waves", text, re.IGNORECASE)


def test_touchstone_with_more_ports_uses_rows(tmp_path):
    s = np.arange(27).reshape(3, 3, 3) * (0.01 + 0.02j)
    net = _net(s, 50.0, ports=("a", "b", "c"))
    _, _, data = _read_touchstone(net.to_touchstone(tmp_path / "net.s3p"))
    rows = data.reshape(3, 19)
    pairs = rows[:, 1::2] + 1j * rows[:, 2::2]
    np.testing.assert_allclose(pairs, s.reshape(3, 9), rtol=1e-14, atol=1e-16)


def test_touchstone_renormalizes_to_z0(line, tmp_path):
    _, option, data = _read_touchstone(line.to_touchstone(tmp_path / "a.s2p", z0=75))
    assert option[-1] == "75"
    pairs = data.reshape(3, 9)[:, 1::2] + 1j * data.reshape(3, 9)[:, 2::2]
    s = np.asarray(line.renormalize(75.0).values)
    np.testing.assert_allclose(pairs[:, 1], s[:, 1, 0], rtol=1e-13)


@pytest.mark.parametrize(
    ("z_ref", "message"),
    [
        ([50.0, 75.0], "z0="),
        ([50.0 + 1j, 50.0 + 1j], "z0="),
    ],
)
def test_touchstone_never_labels_a_reference_it_does_not_have(
    line, tmp_path, z_ref, message
):
    net = _net(line.values, z_ref)
    with pytest.raises(ValueError, match=message):
        net.to_touchstone(tmp_path / "x.s2p")
    assert not (tmp_path / "x.s2p").exists()
    # the remedy works
    net.to_touchstone(tmp_path / "x.s2p", z0=50.0)


def test_touchstone_refuses_ports_without_a_reference_impedance(line, tmp_path):
    values = np.asarray(line.values).copy()
    values[:, 1, :] = np.nan
    net = _net(values, [50.0, np.nan])
    for z0 in (None, 50.0):
        with pytest.raises(ValueError, match=r"'p2'.*no reference impedance"):
            net.to_touchstone(tmp_path / "x.s2p", z0=z0)


def test_touchstone_checks_the_file_name_and_z0(line, tmp_path):
    with pytest.raises(ValueError, match=r"\.s2p"):
        line.to_touchstone(tmp_path / "line.s3p")
    with pytest.raises(ValueError, match="z0"):
        line.to_touchstone(tmp_path / "line.s2p", z0=50.0 + 5j)
    partial = _net(np.asarray(line.values)[:, :, :1], 50.0, excited=("p1",))
    with pytest.raises(ValueError, match="excite"):
        partial.to_touchstone(tmp_path / "line.s2p")


# --- conventions ------------------------------------------------------------------


def test_conventions_module_states_the_conventions():
    doc = gfem.conventions.__doc__
    for phrase in ("e^{+iωt}", "power wave", "peak", "1 W", "z_ref", "1/2"):
        assert phrase in doc, phrase
    assert gfem.conventions.INCIDENT_POWER == 1.0
    assert gfem.conventions.TIME_DEPENDENCE == "exp(+iωt)"
