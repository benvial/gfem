"""Port modes converge to analytic waveguide modes as the element order rises.

One coarse fixed mesh per guide; p-refinement only. Cases:
- empty WR-90: TE10 propagation constant, power-voltage impedance
  ``2 b Z_TE / a`` along the centre line, and the ``sin(pi x / a)`` profile;
- WR-90 with a full-height dielectric slab against one wall: the TE10-like
  mode of the transcendental dispersion relation (a hybrid material case).
"""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

import gfem

DATA = Path(__file__).parents[1] / "unit" / "data"
C0 = 299792458.0
ETA0 = 376.730313412
A, B = 22.86e-3, 10.16e-3
SLAB, EPS_SLAB = 8e-3, 4.0
FREQS = np.array([10e9, 16e9])
ORDERS = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def te10():
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    path = ((A / 2, 0.0, 0.0), (A / 2, B, 0.0))
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1", voltage_path=path)],
    )
    return {
        p: gfem.solve(
            model, gfem.PortMode("p1", FREQS), options=gfem.SolverOptions(order=p)
        )
        for p in ORDERS
    }


def _te10_beta(f):
    k0 = 2 * np.pi * f / C0
    return np.sqrt(k0**2 - (np.pi / A) ** 2)


def test_te10_propagation_constant_converges_with_order(te10):
    beta = _te10_beta(FREQS)
    errors = [np.abs(np.asarray(te10[p].gamma) / (1j * beta) - 1).max() for p in ORDERS]
    assert all(e2 < e1 / 10 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < 1e-10, errors
    for p in ORDERS:
        assert np.abs(np.asarray(te10[p].gamma).real).max() < 1e-12 * beta.min()


def test_te10_power_voltage_impedance_converges_with_order(te10):
    k0 = 2 * np.pi * FREQS / C0
    z_te = ETA0 * k0 / _te10_beta(FREQS)
    exact = 2 * B * z_te / A
    errors = [np.abs(np.asarray(te10[p].Z_PV) / exact - 1).max() for p in ORDERS]
    assert all(e2 < e1 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < 1e-6, errors


def test_te10_profile_is_a_half_sine_at_one_watt(te10):
    x = np.linspace(0.05, 0.95, 7) * A
    points = np.stack([x, np.full_like(x, B / 3), np.zeros_like(x)], axis=1)
    e = np.asarray(te10[4].e_field.at(points))
    k0 = 2 * np.pi * FREQS / C0
    z_te = ETA0 * k0 / _te10_beta(FREQS)
    e0 = np.sqrt(4 * z_te / (A * B))  # P = a b E0^2 / (4 Z_TE) = 1 W
    expected = e0[:, None] * np.sin(np.pi * x / A)[None, :]
    np.testing.assert_allclose(e[..., 1], expected, rtol=2e-5)
    np.testing.assert_allclose(e[..., [0, 2]], 0.0, atol=2e-5 * e0.max())


def _slab_beta(f):
    k0 = 2 * math.pi * f / C0

    def mismatch(beta):
        k1 = np.sqrt(complex(EPS_SLAB * k0**2 - beta**2))
        k2 = np.sqrt(complex(k0**2 - beta**2))
        # k1 cot(k1 s) + k2 cot(k2 (a - s)) = 0, multiplied through to be pole-free
        # and real for real or imaginary k2.
        s2 = np.sin(k2 * (A - SLAB)) / k2
        return (
            k1 * np.cos(k1 * SLAB) * s2 + np.cos(k2 * (A - SLAB)) * np.sin(k1 * SLAB)
        ).real

    grid = np.linspace(1e-6, math.sqrt(EPS_SLAB) * k0 * (1 - 1e-9), 4000)
    values = [mismatch(b) for b in grid]
    roots = [
        brentq(mismatch, grid[i], grid[i + 1])
        for i in range(len(grid) - 1)
        if values[i] * values[i + 1] < 0
    ]
    return max(roots)


def test_slab_loaded_guide_mode_converges_with_order():
    mesh = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"slab": gfem.Material(eps_r=EPS_SLAB), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    freqs = np.array([8e9, 12e9])
    beta = np.array([_slab_beta(f) for f in freqs])
    errors = []
    for p in ORDERS:
        result = gfem.solve(
            model, gfem.PortMode("p1", freqs), options=gfem.SolverOptions(order=p)
        )
        errors.append(np.abs(np.asarray(result.gamma) / (1j * beta) - 1).max())
    assert all(e2 < e1 / 10 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < 1e-9, errors
