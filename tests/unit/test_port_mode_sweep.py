"""The guarded port mode sweep behind `gfem.PortMode` over a frequency grid.

A grid's port modes come from a reduced port mode model checked at every
frequency, with exact solves only where the check fails. These tests run on
small waveguide fixtures; the CPW and hybrid-waveguide accuracy targets are in
the verification tier.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import gfem
from gfem._engine.port_mode import PortModeSpace
from gfem._port_section import port_section
from gfem._sweep.port_mode import _Candidates, _Greedy, port_mode_sweep

DATA = Path(__file__).parent / "data"
C0 = 299792458.0
A, B = 22.86e-3, 10.16e-3  # WR-90, m


@pytest.fixture(scope="module")
def slab_mesh():
    return gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")


def slab(mesh, *, eps_r=4.0, tan_delta=0.0, **port):
    return gfem.Model(
        mesh,
        materials={
            "slab": gfem.Material(eps_r=eps_r, tan_delta=tan_delta),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1", **port)],
    )


def exact(model, frequency, mode=None):
    """One frequency is always an exact port mode solve."""
    result = gfem.solve(model, gfem.PortMode("p1", frequency, mode=mode))
    assert bool(np.asarray(result.solved)[0])
    return result


def plane_points(n=12):
    """Points spread over the WR-90 port at z = 0, m."""
    x = (np.arange(n) + 0.5) / n * A
    y = (np.arange(n // 2) + 0.5) / (n // 2) * B
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)


def relative(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


GRID = np.linspace(10e9, 20e9, 50)


@pytest.fixture(scope="module")
def swept(slab_mesh):
    return gfem.solve(slab(slab_mesh), gfem.PortMode("p1", GRID))


# --- few exact solves, exact accuracy --------------------------------------------


def test_a_sweep_solves_only_a_few_frequencies_exactly(swept):
    solved = np.asarray(swept.solved)
    assert solved.shape == GRID.shape
    assert solved[0]
    assert solved[-1]
    assert 2 < solved.sum() <= 12
    assert np.asarray(swept.residual).max() <= 1e-6


def test_the_swept_mode_matches_exact_solves(slab_mesh, swept):
    model = slab(slab_mesh)
    points = plane_points()
    for i in (3, 17, 30, 44):
        ref = exact(model, GRID[i])
        np.testing.assert_allclose(
            np.asarray(swept.gamma)[i], np.asarray(ref.gamma)[0], rtol=1e-8
        )
        e = np.asarray(swept.e_field.at(points))[i]
        e_ref = np.asarray(ref.e_field.at(points))[0]
        assert relative(e, e_ref) < 1e-5


def test_up_to_three_frequencies_are_all_solved_exactly(slab_mesh):
    result = gfem.solve(slab(slab_mesh), gfem.PortMode("p1", [10e9, 15e9, 20e9]))
    assert np.asarray(result.solved).all()


def test_frequencies_keep_their_given_order(slab_mesh, swept):
    shuffled = np.random.default_rng(0).permutation(len(GRID))
    result = gfem.solve(slab(slab_mesh), gfem.PortMode("p1", GRID[shuffled]))
    np.testing.assert_array_equal(np.asarray(result.frequency), GRID[shuffled])
    np.testing.assert_allclose(
        np.asarray(result.gamma), np.asarray(swept.gamma)[shuffled], rtol=1e-8
    )


def test_a_looser_port_mode_tolerance_needs_fewer_exact_solves(slab_mesh):
    model = slab(slab_mesh)
    counts = {}
    for tol in (1e-3, 1e-9):
        result = gfem.solve(
            model,
            gfem.PortMode("p1", GRID),
            options=gfem.SolverOptions(port_mode_tolerance=tol),
        )
        assert np.asarray(result.residual).max() <= tol
        counts[tol] = int(np.asarray(result.solved).sum())
    assert counts[1e-3] < counts[1e-9]


# --- phase and tracking ------------------------------------------------------------


def test_the_mode_phase_is_continuous_over_the_band(slab_mesh):
    model = slab(slab_mesh, tan_delta=0.05)
    result = gfem.solve(model, gfem.PortMode("p1", GRID))
    e = np.asarray(result.e_field.at(plane_points()))
    flat = e.reshape(len(GRID), -1)
    overlap = np.sum(flat[:-1].conj() * flat[1:], axis=1)
    norms = np.linalg.norm(flat[:-1], axis=1) * np.linalg.norm(flat[1:], axis=1)
    assert (overlap.real / norms > 0.99).all()
    # the lowest frequency follows Palace's phase rule, like an exact solve
    first = np.asarray(exact(model, GRID[0]).e_field.at(plane_points()))[0]
    assert relative(e[0], first) < 1e-8


def test_a_mode_crossing_warns_and_follows_the_physical_mode(slab_mesh):
    model = slab(slab_mesh)
    grid = np.linspace(16e9, 40e9, 60)
    with pytest.warns(gfem.PortModeTrackingWarning, match="mode crossing"):
        result = gfem.solve(model, gfem.PortMode("p1", grid, mode=4))
    n_eff = np.asarray(result.n_eff).real
    # continuous along the physical branch ...
    assert np.abs(np.diff(n_eff)).max() < 0.05
    # ... which is no longer the 4th in Palace's order at the top of the band
    top = exact(model, grid[-1], mode=4)
    assert abs(n_eff[-1] - np.asarray(top.n_eff).real[0]) > 1e-3


def test_two_requested_modes_of_a_shared_surface_cross_cleanly(slab_mesh):
    # mode 4 crosses mode 5 near 18 GHz (rank order at 16 GHz)
    model = slab(slab_mesh).replace(
        ports=[
            gfem.WavePort("p1_m4", "port1", mode=4),
            gfem.WavePort("p1_m5", "port1", mode=5),
        ]
    )
    grid = np.linspace(16e9, 40e9, 60)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m4 = gfem.solve(model, gfem.PortMode("p1_m4", grid))
    texts = [str(w.message) for w in caught]
    assert all(w.category is gfem.PortModeTrackingWarning for w in caught)
    # each port whose rank changes warns
    assert any("WavePort 'p1_m4': the tracked mode 4 is mode 5" in t for t in texts)
    assert any("WavePort 'p1_m5': the tracked mode 5 is mode 4" in t for t in texts)
    with pytest.warns(gfem.PortModeTrackingWarning):
        m5 = gfem.solve(model, gfem.PortMode("p1_m5", grid))
    n4, n5 = np.asarray(m4.n_eff).real, np.asarray(m5.n_eff).real
    # each continuous along its physical branch: measured 0.035 and 0.071
    # (mode 5 rises steeply just above its cutoff)
    assert np.abs(np.diff(n4)).max() < 0.05
    assert np.abs(np.diff(n5)).max() < 0.1
    # mode 4 is the branch a lone sweep follows (measured 1.2e-13)
    lone = gfem.solve(slab(slab_mesh), gfem.PortMode("p1", grid, mode=4))
    np.testing.assert_allclose(n4, np.asarray(lone.n_eff).real, rtol=0, atol=1e-11)
    # distinct at every frequency, closest at the crossing: measured 3.9e-4
    assert np.abs(n4 - n5).min() > 1e-4


def test_no_tracking_warning_without_a_crossing(slab_mesh):
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.PortModeTrackingWarning)
        gfem.solve(slab(slab_mesh), gfem.PortMode("p1", GRID))


def test_a_degenerate_mode_raises_port_mode_error():
    # a square guide whose port mesh keeps the 90 degree rotation symmetry:
    # TE10 and TE01 are one degenerate pair, so the port field is not defined
    model = gfem.Model(
        gfem.Mesh.from_gmsh(DATA / "square_waveguide.msh", length_unit="mm"),
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    with pytest.raises(gfem.PortModeError, match="degenerate"):
        gfem.solve(model, gfem.PortMode("p1", np.linspace(20e9, 30e9, 10)))


# --- the sweep's reduced representation (consumed by driven wave ports) ---------


def test_the_sweep_field_is_affine_in_its_coordinates_at_one_watt(slab_mesh):
    model = slab(slab_mesh, tan_delta=0.02, voltage_path=((4e-3, 0, 0), (4e-3, B, 0)))
    space = PortModeSpace(port_section(model, model.ports[0]), 2)
    [sweep] = port_mode_sweep(space, GRID, [1], tolerance=1e-6, ports=[model.ports[0]])
    m = sweep.basis.shape[1]
    assert sweep.basis.shape == (space.space.ndof, m)
    assert sweep.coords.shape == (len(GRID), m)
    assert m <= 2 * 2 * int(sweep.solved.sum())  # real and imaginary parts, 2 blocks
    for i in (0, 21, 49):
        c = sweep.basis @ sweep.coords[i]
        gamma = sweep.gamma[i] * space.section.scale
        power = space.power(c, gamma, 2 * math.pi * GRID[i])
        assert abs(abs(power) - 1.0) < 1e-10
    assert (sweep.voltage.real > 0).all()
    np.testing.assert_allclose(sweep.z_pv, np.abs(sweep.voltage) ** 2 / 2)


# --- several modes of one port surface -------------------------------------------

# mode 4 of the slab guide propagates from 16 GHz (the crossing test above)
JOINT = np.linspace(16e9, 20e9, 8)


def shared_ports():
    return [
        gfem.WavePort("in_TE0", "port1", mode=1),
        gfem.WavePort("in_TM0", "port1", mode=2),
    ]


@pytest.fixture(scope="module")
def joint(slab_mesh):
    """Two requested modes, kept at the lowest frequency's exact solve."""
    model = slab(slab_mesh)
    space = PortModeSpace(port_section(model, model.ports[0]), 2)
    greedy = _Greedy(space, JOINT, [1, 2], 1e-6, shared_ports())
    first = greedy._solve(0)
    greedy._keep(0, first, [0, 1])
    return greedy, first


def test_a_reduced_tie_of_two_modes_gives_both_an_infinite_backward_error(joint):
    greedy, _ = joint
    k0, k_max = greedy.k0[1], greedy.k_max[1]
    apart = [greedy.coords_at(0, 0), greedy.coords_at(0, 1)]
    assert all(
        math.isfinite(eta) for _, _, eta in greedy.reduced.solve(k0, k_max, apart)
    )
    tied = [greedy.coords_at(0, 0)] * 2
    assert all(math.isinf(eta) for _, _, eta in greedy.reduced.solve(k0, k_max, tied))


def test_an_exact_tie_of_two_modes_raises_naming_both_ports_and_the_surface(joint):
    greedy, first = joint
    # one candidate left: both requested modes can only follow it
    one = _Candidates(
        lam=first.lam[:1], vecs=first.vecs[:, :1], residual=first.residual[:1]
    )
    with pytest.raises(
        gfem.PortModeError,
        match=r"WavePorts 'in_TE0' and 'in_TM0' on surface 'port1' follow the same "
        r"mode at 16\.5714 GHz \(an unresolved crossing between two requested modes\)",
    ):
        greedy._closest(1, one, 0, certain=True)
    # against an uncertain reference the tie is accepted provisionally
    assert greedy._closest(1, one, 0, certain=False) == [0, 0]


def test_a_joint_sweep_of_one_surface_shares_its_solves_and_basis(slab_mesh):
    model = slab(slab_mesh)
    space = PortModeSpace(port_section(model, model.ports[0]), 2)
    te, tm = port_mode_sweep(space, JOINT, [1, 2], tolerance=1e-6, ports=shared_ports())
    assert te.basis is tm.basis
    np.testing.assert_array_equal(te.solved, tm.solved)
    assert max(te.residual.max(), tm.residual.max()) <= 1e-6
    # each mode is the one a lone exact solve gives at every exact sample
    for mode, sweep in ((1, te), (2, tm)):
        for i in np.flatnonzero(sweep.solved):
            lone = exact(model, JOINT[i], mode=mode)
            np.testing.assert_allclose(
                sweep.gamma[i], np.asarray(lone.gamma)[0], rtol=1e-9
            )
