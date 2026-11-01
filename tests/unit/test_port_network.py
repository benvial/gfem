"""The port network of a driven solve, at its interface.

`split_cotangent` on small arrays (no mesh), a `PortNetwork` from small
arrays and duck-typed wave port couplings, and one `PortNetwork.build` on a
parallel-plate line's driven operator (assembled, no linear solve).
"""

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import driven
from gfem._engine.port_network import (
    PARALLEL_RTOL,
    PortNetwork,
    resistance,
    split_cotangent,
)
from gfem._sweep.adaptive import AffineTerm

N = 6
OMEGAS = (2 * math.pi * 4e9, 2 * math.pi * 5e9)


def _complex(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _adaptive_fold(functional, ct):
    """The adaptive backward's fold before the port network (the oracle)."""
    out = np.zeros((functional.shape[0], ct.shape[1]), complex)
    for i in np.flatnonzero((ct != 0).any(axis=1)):
        if np.isfinite(functional[:, i]).all():
            out += np.outer(functional[:, i], ct[i])
        else:
            out[:, ct[i] != 0] = np.nan
    return out


def test_an_excited_parallel_row_is_its_own_solution_scaled():
    rng = np.random.default_rng(0)
    rhs, x = _complex(rng, N, 2), _complex(rng, N, 2)
    scale = 0.3 - 2.1j
    functional = np.zeros((N, 2), complex)
    functional[:, 1] = scale * rhs[:, 0]  # port 1 is column 0
    ct = np.zeros((2, 2), complex)
    ct[1] = [1.5 + 0.5j, -0.25j]
    split = split_cotangent(ct, np.ones(2), [1, 0], functional, own=(rhs, x))
    np.testing.assert_allclose(
        split.free, np.outer(scale * x[:, 0], ct[1]), rtol=1e-12, atol=0
    )
    assert not split.loads.any()
    assert not split.undefined.any()


def test_a_lossy_wave_port_row_is_a_load():
    rng = np.random.default_rng(1)
    rhs, x = _complex(rng, N, 1), _complex(rng, N, 1)
    functional = (2 - 1j) * rhs + 1e-6 * _complex(rng, N, 1)  # not parallel
    residual = np.linalg.norm(functional - (2 - 1j) * rhs)
    assert residual > PARALLEL_RTOL * np.linalg.norm(functional)
    ct = np.array([[1.0 + 1.0j]])
    split = split_cotangent(ct, np.ones(1), [0], functional, own=(rhs, x))
    np.testing.assert_array_equal(split.loads, np.outer(functional[:, 0], ct[0]))
    assert not split.free.any()


def test_an_unexcited_row_is_a_load_even_when_parallel():
    rng = np.random.default_rng(2)
    rhs, x = _complex(rng, N, 1), _complex(rng, N, 1)
    functional = np.stack([rhs[:, 0], 3 * rhs[:, 0]], axis=1)
    ct = np.array([[0.0], [2.0 - 1.0j]])
    split = split_cotangent(ct, np.ones(2), [0], functional, own=(rhs, x))
    np.testing.assert_array_equal(split.loads, np.outer(functional[:, 1], ct[1]))
    assert not split.free.any()


def test_a_cotangent_on_an_undefined_row_marks_its_columns():
    rng = np.random.default_rng(3)
    functional = _complex(rng, N, 3)
    functional[:, 2] = np.nan  # a port without a resistance
    ct = np.zeros((3, 3), complex)
    ct[0] = [1.0, 0.0, 2.0j]
    ct[2] = [0.0, 0.0, 1.0]
    split = split_cotangent(ct, np.ones(3), [0, 1, 2], functional)
    np.testing.assert_array_equal(split.undefined, [False, False, True])
    assert np.isfinite(split.loads).all()
    np.testing.assert_array_equal(split.loads, np.outer(functional[:, 0], ct[0]))


def test_an_undefined_row_without_a_cotangent_is_skipped():
    rng = np.random.default_rng(4)
    functional = _complex(rng, N, 2)
    functional[:, 1] = np.nan
    ct = np.array([[1.0 - 1.0j, 0.5], [0.0, 0.0]])
    split = split_cotangent(ct, np.ones(2), [0, 1], functional)
    assert not split.undefined.any()
    np.testing.assert_array_equal(split.loads, np.outer(functional[:, 0], ct[0]))


def test_the_cotangent_is_de_embedded_on_its_row_and_column():
    rng = np.random.default_rng(5)
    functional = _complex(rng, N, 3)
    ct = _complex(rng, 3, 2)
    d = np.exp(1j * np.array([0.1, -0.7, 1.3]))
    columns = [2, 0]
    split = split_cotangent(ct, d, columns, functional)
    expected = sum(
        np.outer(functional[:, i], ct[i] * d[i] * d[columns]) for i in range(3)
    )
    np.testing.assert_allclose(split.loads, expected, rtol=1e-14, atol=1e-14)


def test_without_reciprocity_the_split_is_the_adaptive_fold():
    rng = np.random.default_rng(6)
    functional = _complex(rng, N, 4)
    functional[:, 1] = np.nan
    ct = _complex(rng, 4, 3)
    ct[3] = 0.0
    ct[1, 2] = 0.0
    d = np.exp(1j * np.array([0.2, 0.0, -0.4, 0.9]))
    columns = [0, 2, 3]
    oracle = _adaptive_fold(functional, ct * d[:, None] * d[columns][None, :])
    split = split_cotangent(ct, d, columns, functional)
    np.testing.assert_array_equal(split.undefined, np.isnan(oracle).any(axis=0))
    defined = ~split.undefined
    np.testing.assert_array_equal(split.loads[:, defined], oracle[:, defined])
    assert not split.free.any()


def _wave(index, offset=0.0):
    return SimpleNamespace(
        index=index,
        omegas=np.array(OMEGAS),
        deembedding=lambda w, offset=offset: complex(np.exp(1j * offset * w / 1e10)),
    )


def test_scattering_subtracts_the_incident_wave_and_de_embeds():
    waves = [_wave(1, offset=0.5)]
    network = PortNetwork([], [], waves, [1, 0], np.full((2, 2), 50.0))
    outgoing = _complex(np.random.default_rng(7), 2, 2, 2)
    s = network.scattering(outgoing)
    d = np.ones((2, 2), complex)
    d[:, 1] = [np.exp(1j * 0.5 * w / 1e10) for w in OMEGAS]
    expected = outgoing.copy()
    expected[:, 1, 0] -= 1.0
    expected[:, 0, 1] -= 1.0
    expected *= d[:, :, None] * d[:, None, [1, 0]]
    np.testing.assert_allclose(s, expected, rtol=1e-14, atol=0)


def test_adjoint_loads_skip_an_excited_lossless_wave_port():
    rng = np.random.default_rng(8)
    e = _complex(rng, N, 2)
    f = np.stack([2j * e[:, 0], e[:, 1] + 0.5 * _complex(rng, N), _complex(rng, N)], 1)
    network = PortNetwork(
        [AffineTerm(e, lambda _: np.eye(2))],
        [AffineTerm(f, lambda _: np.eye(3))],
        [_wave(0), _wave(1), _wave(2)],
        [0, 1],
        np.full((2, 3), 50.0),
    )
    # port 0 lossless and excited, port 1 lossy, port 2 not excited
    np.testing.assert_array_equal(network.adjoint_loads(OMEGAS[0]), f[:, [1, 2]])
    assert network.nbytes == e.nbytes + f.nbytes


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", resistance=25.0),
            gfem.LumpedPort("p2", "port2", "+z", capacitance=1e-12),
        ],
    )


def test_build_reads_a_lumped_port_as_its_voltage_over_sqrt_2r(line):
    op = driven.driven_operator(line, 1)
    network = PortNetwork.build(line, op, (), columns=[0], n_freq=len(OMEGAS))
    assert resistance(line.ports[0]) == 25.0
    np.testing.assert_array_equal(network.z_ref[:, 0], 25.0)
    assert np.isnan(network.z_ref[:, 1]).all()  # no resistance: no z_ref
    (term,) = network.outputs
    rows = term.coefficients(OMEGAS[0])
    assert rows[0, 0] == 1.0 / math.sqrt(50.0)
    assert np.isnan(rows[1, 1])
    functional = network.functional(OMEGAS[0])
    assert np.isfinite(functional[:, 0]).all()
    assert np.isnan(functional[:, 1]).all()
    assert network.rhs(OMEGAS[0]).shape == (op.space.ndof, 1)


def _mode(index, column):
    return SimpleNamespace(
        index=index,
        omegas=np.array(OMEGAS),
        excitation=lambda w, column=column: column * w / OMEGAS[0],
        functional=lambda _, column=column: 2j * column,
        z_ref=np.full(len(OMEGAS), 10.0 * (index + 1)),
        deembedding=lambda _: 1.0,
    )


def test_build_gives_one_term_per_wave_port_surface(line):
    # two lumped ports dropped from the operator: indices 0 and 1 are the
    # modes of one duck-typed surface
    op = dataclasses.replace(driven.driven_operator(line, 1), ports=())
    rng = np.random.default_rng(9)
    vectors = _complex(rng, op.space.ndof, 4)
    first, second = _complex(rng, 4), _complex(rng, 4)
    surface = SimpleNamespace(
        vectors=vectors, ports=(_mode(1, second), _mode(0, first))
    )
    network = PortNetwork.build(line, op, [surface], columns=[1, 0], n_freq=2)
    assert len(network.excitation) == 1
    assert len(network.outputs) == 1
    assert [c.index for c in network.waves] == [0, 1]
    np.testing.assert_array_equal(network.z_ref, [[10.0, 20.0], [10.0, 20.0]])
    w = OMEGAS[1]
    rhs = vectors @ np.stack([second, first], axis=1) * (w / OMEGAS[0])
    np.testing.assert_allclose(network.rhs(w), rhs, rtol=1e-14, atol=0)
    x = _complex(rng, op.space.ndof, 2)
    out = np.stack([first, second]) @ (np.conj(vectors).T @ x) * 2j
    np.testing.assert_allclose(network.outgoing(w, x), out, rtol=1e-12, atol=0)
