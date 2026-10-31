"""Sweep outputs at their interface: rows, join, the seam's names and specs.

Small arrays and duck-typed wave port couplings; no mesh, no solve.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gfem._engine.sweep_outputs import DIFFERENTIABLE, SweepOutputs

N_FREQ, N_PORTS, N_EXC, NDOF, M = 4, 3, 2, 7, 5
HOST_ONLY = {"iterations", "port_sweeps", "sweep"}


def _complex(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _wave(rng, name):
    """A wave port coupling with only its port mode sweep over the grid."""
    space = SimpleNamespace(section=SimpleNamespace(port=SimpleNamespace(name=name)))
    sweep = SimpleNamespace(
        space=space,
        basis=rng.standard_normal((NDOF, M)),
        coords=_complex(rng, N_FREQ, M),
        gamma=_complex(rng, N_FREQ),
        z_pv=rng.random(N_FREQ),
        residual=rng.random(N_FREQ),
        solved=rng.random(N_FREQ) > 0.5,
    )
    return SimpleNamespace(sweep=sweep)


def _uniform(rng, rows, *, waves=(), fields=True):
    return SweepOutputs.uniform(
        rows,
        s=_complex(rng, N_FREQ, N_PORTS, N_EXC),
        z_ref=rng.random((N_FREQ, N_PORTS)),
        iterations=[(3, 4)] * len(rows),
        e_field=_complex(rng, NDOF, N_EXC) if fields else None,
        waves=list(waves),
    )


def _adaptive(rng, *, waves=(), fields=True):
    n = 6
    reduced = (rng.standard_normal((NDOF, n)), _complex(rng, N_EXC, n))
    return SweepOutputs.adaptive(
        s=_complex(rng, N_FREQ, N_PORTS, N_EXC),
        z_ref=rng.random((N_FREQ, N_PORTS)),
        iterations=[(5, 5), (6, 7)],
        sweep={"samples": [0, 3], "errors": [1.0], "converged": True, "basis_size": 4},
        reduced=reduced if fields else None,
        waves=list(waves),
    )


def test_uniform_trims_every_array_and_the_port_mode_coordinates_together():
    rng = np.random.default_rng(0)
    waves = [_wave(rng, "w1"), _wave(rng, "w2")]
    s = _complex(rng, N_FREQ, N_PORTS, N_EXC)
    z_ref = rng.random((N_FREQ, N_PORTS))
    out = SweepOutputs.uniform(
        (0, 1), s=s, z_ref=z_ref, iterations=[(1,), (2,)], e_field=None, waves=waves
    )
    assert out.rows == (0, 1)
    np.testing.assert_array_equal(out.s, s[:2])
    np.testing.assert_array_equal(out.z_ref, z_ref[:2])
    ports = out.ports
    assert ports is not None
    for name in ("gamma", "z_pv", "residual", "solved"):
        expected = np.stack([getattr(w.sweep, name) for w in waves], axis=1)[:2]
        np.testing.assert_array_equal(getattr(ports, name), expected)
    for k, wave in enumerate(waves):
        field = ports.field(k)
        assert field is not None
        space, basis, coords = field._parts
        assert space is wave.sweep.space
        assert basis is wave.sweep.basis
        np.testing.assert_array_equal(coords, wave.sweep.coords[[0, 1]])


def test_every_row_leaves_the_port_mode_coordinates_unsliced():
    rng = np.random.default_rng(1)
    waves = [_wave(rng, "w1")]
    out = _uniform(rng, tuple(range(N_FREQ)), waves=waves)
    field = out.ports.field(0)
    assert field._parts[2] is waves[0].sweep.coords


def test_no_wave_port_has_no_port_modes():
    out = _uniform(np.random.default_rng(2), (0, 1, 2))
    assert out.ports is None
    assert not any(name.startswith("port_") for name in out.outputs())


def test_join_maps_each_blocks_rows_to_grid_indices():
    rng = np.random.default_rng(3)
    first = SweepOutputs.uniform(
        (0, 1),
        s=_complex(rng, 3, N_PORTS, N_EXC),
        z_ref=rng.random((3, N_PORTS)),
        iterations=[(1,), (2,)],
        e_field=_complex(rng, NDOF, N_EXC),
        waves=[],
    )
    second = SweepOutputs.uniform(
        (0, 1, 2),
        s=_complex(rng, 3, N_PORTS, N_EXC),
        z_ref=rng.random((3, N_PORTS)),
        iterations=[(3,), (4,), (5,)],
        e_field=_complex(rng, NDOF, 2 * N_EXC),
        waves=[],
    )
    out = SweepOutputs.join([((0, 1, 2), first), ((3, 4, 5), second)])
    assert out.rows == (0, 1, 3, 4, 5)
    np.testing.assert_array_equal(out.s, np.concatenate([first.s, second.s]))
    np.testing.assert_array_equal(
        out.z_ref, np.concatenate([first.z_ref, second.z_ref])
    )
    assert out.iterations == [(1,), (2,), (3,), (4,), (5,)]
    np.testing.assert_array_equal(
        out.e_field, np.concatenate([first.e_field, second.e_field], axis=1)
    )
    assert out.ports is None
    assert out.sweep is None


def test_join_keeps_no_field_when_no_block_kept_any():
    rng = np.random.default_rng(4)
    blocks = [
        ((0, 1), _uniform(rng, (0,), fields=False)),
        ((2, 3), _uniform(rng, (0, 1), fields=False)),
    ]
    out = SweepOutputs.join(blocks)
    assert out.rows == (0, 2, 3)
    assert out.e_field is None
    assert "e_field" not in out.outputs()


def test_join_refuses_blocks_with_wave_ports():
    rng = np.random.default_rng(5)
    block = _uniform(rng, (0, 1), waves=[_wave(rng, "w1")])
    with pytest.raises(ValueError, match="wave ports"):
        SweepOutputs.join([((0, 1, 2, 3), block)])


def _check_specs(out, specs):
    outputs = out.outputs()
    assert set(specs) <= set(outputs)
    for name, spec in specs.items():
        np.asarray(outputs[name], spec.dtype).reshape(spec.shape)
    host = {"iterations", "sweep"} & set(outputs)
    if out.ports is not None:
        host.add("port_sweeps")
    assert set(outputs) - set(specs) == host
    assert host <= HOST_ONLY


def test_a_uniform_encoding_matches_its_specs():
    rng = np.random.default_rng(6)
    out = _uniform(rng, tuple(range(N_FREQ)), waves=[_wave(rng, "w1")])
    specs = SweepOutputs.specs(
        n_freq=N_FREQ,
        n_ports=N_PORTS,
        n_excited=N_EXC,
        n_wave=1,
        ndof=NDOF,
        n_fields=N_EXC,
    )
    assert list(specs) == [
        "s",
        "z_ref",
        "port_gamma",
        "port_z_pv",
        "port_residual",
        "port_solved",
        "e_field",
    ]
    _check_specs(out, specs)


def test_an_adaptive_encoding_matches_its_specs():
    rng = np.random.default_rng(7)
    out = _adaptive(rng, waves=[_wave(rng, "w1")])
    specs = SweepOutputs.specs(
        n_freq=N_FREQ,
        n_ports=N_PORTS,
        n_excited=N_EXC,
        n_wave=1,
        ndof=NDOF,
        n_fields=N_EXC,
        reduced=6,
    )
    assert list(specs)[-2:] == ["e_basis", "e_coordinates"]
    assert "sweep" in out.outputs()
    _check_specs(out, specs)


def test_a_fieldless_encoding_matches_its_specs():
    rng = np.random.default_rng(8)
    out = _uniform(rng, tuple(range(N_FREQ)), fields=False)
    specs = SweepOutputs.specs(
        n_freq=N_FREQ, n_ports=N_PORTS, n_excited=N_EXC, n_wave=0
    )
    assert list(specs) == ["s", "z_ref"]
    _check_specs(out, specs)


def test_the_differentiable_outputs_are_spec_names():
    specs = SweepOutputs.specs(
        n_freq=1, n_ports=1, n_excited=1, n_wave=1, ndof=1, n_fields=1
    )
    reduced = SweepOutputs.specs(
        n_freq=1, n_ports=1, n_excited=1, n_wave=0, ndof=1, n_fields=1, reduced=1
    )
    assert set(specs) | set(reduced) >= DIFFERENTIABLE
    assert "e_basis" not in DIFFERENTIABLE


@pytest.mark.parametrize("kind", ["uniform", "adaptive"])
def test_reading_the_encoding_gives_back_every_field(kind):
    rng = np.random.default_rng(9)
    waves = [_wave(rng, "w1"), _wave(rng, "w2")]
    if kind == "uniform":
        out = _uniform(rng, tuple(range(N_FREQ)), waves=waves)
    else:
        out = _adaptive(rng, waves=waves)
    back = SweepOutputs.read(out.outputs())
    assert back.rows == out.rows
    for name in ("s", "z_ref", "iterations", "e_field", "sweep"):
        assert getattr(back, name) is getattr(out, name)
    if out.reduced is None:
        assert back.reduced is None
    else:
        assert all(a is b for a, b in zip(back.reduced, out.reduced, strict=True))
    for name in ("gamma", "z_pv", "residual", "solved"):
        assert getattr(back.ports, name) is getattr(out.ports, name)
    assert back.ports.rows == out.ports.rows
    assert all(a is b for a, b in zip(back.ports.sweeps, out.ports.sweeps, strict=True))
    assert back.ports.field(1)._parts[2] is waves[1].sweep.coords


def test_reading_the_spec_names_only_gives_no_host_only_field():
    rng = np.random.default_rng(10)
    out = _adaptive(rng, waves=[_wave(rng, "w1")])
    specs = SweepOutputs.specs(
        n_freq=N_FREQ,
        n_ports=N_PORTS,
        n_excited=N_EXC,
        n_wave=1,
        ndof=NDOF,
        n_fields=N_EXC,
        reduced=6,
    )
    outputs = out.outputs()
    back = SweepOutputs.read({name: outputs[name] for name in specs})
    assert back.rows == tuple(range(N_FREQ))
    assert back.iterations is None
    assert back.sweep is None
    assert back.ports.sweeps is None
    assert back.ports.field(0) is None
    np.testing.assert_array_equal(back.s, out.s)
    assert back.reduced[0] is out.reduced[0]
