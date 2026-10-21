"""Frequency workers: a uniform driven sweep spread over spawned processes.

`SolverOptions.frequency_workers=k` splits the frequencies into k
contiguous blocks, each solved by a frequency worker; the parent joins the
blocks back in frequency order. Every test compares with the sequential
sweep (``frequency_workers=1``) on a small parallel-plate line. Each
worker sweep pays the workers' start-up (seconds), so the tests are few.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import driven, workers
from gfem._engine.driven import DrivenRule
from gfem._linalg import prepare

DATA = Path(__file__).parent / "data"
FREQUENCIES = (3e9, 3.5e9, 4e9, 4.5e9)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def test_wave_ports_need_one_worker():
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("plates")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    options = gfem.SolverOptions(order=1, frequency_workers=2)
    with pytest.raises(NotImplementedError, match="frequency_workers=1"):
        gfem.solve(model, gfem.Driven(FREQUENCIES), options=options)


def _forward(model, problem, *, keep=False, **options):
    rule = DrivenRule(model, problem, gfem.SolverOptions(**options))
    return rule.forward(model, keep=keep)


def _count_prepares(monkeypatch) -> list[int]:
    """Spy on this process's driven `prepare`: workers solve elsewhere."""
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(driven, "prepare", spy)
    return calls


def test_two_direct_workers_give_the_sequential_sweep(line, monkeypatch):
    # a kept frequency in each block: the fields join in frequency order
    problem = gfem.Driven(FREQUENCIES, keep_fields=(3.5e9, 4.5e9))
    calls = _count_prepares(monkeypatch)
    one, one_entry = _forward(line, problem, keep=True, linear_solver="direct")
    assert len(calls) == 1
    two, two_entry = _forward(
        line, problem, keep=True, linear_solver="direct", frequency_workers=2
    )
    assert len(calls) == 1  # the parent solved nothing
    np.testing.assert_allclose(two["s"], one["s"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(two["z_ref"], one["z_ref"])
    assert two["iterations"] == one["iterations"] == [()] * 4
    assert two["e_field"].shape == one["e_field"].shape == (one["e_field"].shape[0], 4)
    np.testing.assert_allclose(two["e_field"], one["e_field"], rtol=0, atol=1e-12)
    solution, sequential = two_entry.fields["solution"], one_entry.fields["solution"]
    assert solution.shape == sequential.shape
    np.testing.assert_allclose(solution, sequential, rtol=0, atol=1e-12)
    two_entry.free()
    one_entry.free()


def test_two_bddc_workers_agree_to_the_solver_tolerance(line):
    problem = gfem.Driven(FREQUENCIES)
    one, _ = _forward(line, problem, linear_solver="bddc_gmres")
    two, _ = _forward(line, problem, linear_solver="bddc_gmres", frequency_workers=2)
    # each block's GMRES starts cold, so S moves within the tolerance
    np.testing.assert_allclose(two["s"], one["s"], rtol=0, atol=1e-6)
    assert len(two["iterations"]) == 4
    assert all(len(its) == 2 for its in two["iterations"])  # per excited port


def test_a_killed_worker_names_its_frequencies():
    with pytest.raises(RuntimeError, match=r"4e\+09 to 4.5e\+09 Hz.*fewer"):
        workers.run(os._exit, [(1,)], ["4e+09 to 4.5e+09 Hz"], threads=1)


def test_a_gradient_through_two_workers_matches_one(line):
    # p2 is not excited, so its S row takes an adjoint solve in the parent;
    # weights that differ per frequency catch blocks joined out of order
    n = len(line.mesh.tetrahedra)
    eps = jnp.asarray(np.random.default_rng(5).uniform(2.0, 5.0, n))
    weights = jnp.array([1.0, -2.0, 0.5, 3.0])[:, None, None]

    def objective(eps, workers):
        model = line.with_material("dielectric", gfem.MaterialField(eps_r=eps))
        options = gfem.SolverOptions(linear_solver="direct", frequency_workers=workers)
        problem = gfem.Driven(FREQUENCIES, excite="p1")
        s = gfem.solve(model, problem, options=options).s.values
        return jnp.sum(weights * jnp.abs(s) ** 2)

    one = jax.grad(objective)(eps, 1)
    two = jax.grad(objective)(eps, 2)
    assert np.abs(np.asarray(one)).max() > 0
    np.testing.assert_allclose(two, one, rtol=1e-10, atol=1e-14)
