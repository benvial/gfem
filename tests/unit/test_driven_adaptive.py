"""Driven(sweep="adaptive") on a parallel-plate line with two lumped ports.

Between 1 and 20 GHz the 10 mm line (eps_r 4) passes through several
half-wave resonances, so S is a rational function of frequency with poles
near the band: the greedy needs several samples, far fewer than the grid.
The uniform sweep of the same discrete problem is the reference.
"""

import math
import warnings

import jax
import numpy as np
import pytest
import structured_meshes as sm
from solver_misses import cannot_converge_at

import gfem

L, W, H = 10.0, 2.0, 1.0  # mm
EPS = 4.0
GRID = np.linspace(1e9, 20e9, 58)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (10, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS, tan_delta=1e-3)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


OPTIONS = gfem.SolverOptions(linear_solver="direct")


@pytest.fixture(scope="module")
def uniform(line):
    return gfem.solve(
        line, gfem.Driven(frequency=GRID, keep_fields=True), options=OPTIONS
    )


@pytest.fixture(scope="module")
def adaptive(line):
    problem = gfem.Driven(frequency=GRID, sweep="adaptive", keep_fields=True)
    return gfem.solve(line, problem, options=OPTIONS)


# --- the problem value -----------------------------------------------------------


def test_samples_freeze_grid_frequencies_in_their_order():
    p = gfem.Driven(frequency=GRID, sweep="adaptive", samples=[GRID[7], GRID[0]])
    assert p.samples == (GRID[7], GRID[0])
    assert gfem.Driven(frequency=GRID, sweep="adaptive").samples is None
    assert hash(p) == hash(
        gfem.Driven(frequency=GRID, sweep="adaptive", samples=(GRID[7], GRID[0]))
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": [GRID[1]]}, "sweep='adaptive'"),
        ({"sweep": "adaptive", "samples": [1.5e9]}, "not one of"),
        ({"sweep": "adaptive", "samples": []}, "at least one"),
        ({"sweep": "adaptive", "samples": [GRID[1], GRID[1]]}, "twice"),
    ],
)
def test_bad_samples_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        gfem.Driven(frequency=GRID, **kwargs)


# --- the sweep -------------------------------------------------------------------


def test_adaptive_s_matches_the_uniform_sweep(uniform, adaptive):
    diff = np.abs(np.asarray(adaptive.s.values) - np.asarray(uniform.s.values)).max()
    assert diff <= 1e-6
    assert adaptive.s.frequency == uniform.s.frequency


def test_sweep_info_reports_what_was_solved(adaptive):
    info = adaptive.sweep
    assert info.kind == "adaptive"
    assert info.converged
    assert info.tolerance == OPTIONS.adaptive_tolerance
    sampled = info.sampled_frequencies
    assert sampled[:2] == (GRID[0], GRID[-1])
    assert 4 <= len(sampled) < len(GRID) // 2
    assert set(sampled) <= set(GRID.tolist())
    assert len(info.solved) == len(GRID)
    assert sum(info.solved) == len(sampled)
    assert [f for f, s in zip(GRID, info.solved, strict=True) if s] == sorted(sampled)
    errors = info.sample_errors
    assert len(errors) == len(sampled)
    assert math.isinf(errors[0])
    assert math.isinf(errors[1])
    assert all(e < info.tolerance for e in errors[-2:])
    # two excitations, real and imaginary parts
    assert 0 < info.basis_size <= 4 * len(sampled)
    assert "adaptive" in repr(adaptive)


def test_uniform_results_report_every_frequency_solved(uniform):
    info = uniform.sweep
    assert info.kind == "uniform"
    assert all(info.solved)
    assert info.sampled_frequencies == tuple(GRID)
    assert info.converged
    assert info.basis_size == 0
    assert all(math.isinf(e) for e in info.sample_errors)
    assert info == gfem.results.SweepInfo.uniform(tuple(GRID))


def test_kept_fields_are_basis_backed_and_match_the_full_solve(uniform, adaptive):
    fields = adaptive.e_fields
    assert len(fields) == len(uniform.e_fields) == 2 * len(GRID)
    assert fields.basis.shape[1] == adaptive.sweep.basis_size
    assert adaptive.kept == uniform.kept
    for f in (GRID[0], GRID[13], GRID[40]):
        for port in ("p1", "p2"):
            e, ref = adaptive.e_field(f, port), uniform.e_field(f, port)
            assert (e.space, e.order, e.support) == (ref.space, ref.order, ref.support)
            assert float(e.omega) == pytest.approx(2 * math.pi * f)
            x, x_ref = np.asarray(e.coefficients), np.asarray(ref.coefficients)
            assert np.linalg.norm(x - x_ref) <= 1e-3 * np.linalg.norm(x_ref)


def test_frozen_samples_reproduce_the_greedy_without_searching(line, adaptive):
    sampled = adaptive.sweep.sampled_frequencies
    problem = gfem.Driven(frequency=GRID, sweep="adaptive", samples=sampled)
    frozen = gfem.solve(line, problem, options=OPTIONS)
    assert frozen.sweep.sampled_frequencies == sampled
    assert frozen.sweep.converged
    np.testing.assert_allclose(frozen.s.values, adaptive.s.values, atol=1e-12)


def test_few_frozen_samples_still_give_a_result(line, uniform):
    problem = gfem.Driven(frequency=GRID, sweep="adaptive", samples=[GRID[29]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a frozen set never warns
        result = gfem.solve(line, problem, options=OPTIONS)
    assert result.sweep.sampled_frequencies == (GRID[29],)
    assert not result.sweep.converged
    np.testing.assert_allclose(result.s.values[29], uniform.s.values[29], atol=1e-10)


def test_the_sample_cap_warns_and_returns_the_reduced_result(line):
    options = gfem.SolverOptions(linear_solver="direct", adaptive_max_samples=3)
    problem = gfem.Driven(frequency=GRID, sweep="adaptive")
    with pytest.warns(gfem.SweepNotConvergedWarning, match="adaptive_max_samples"):
        result = gfem.solve(line, problem, options=options)
    assert not result.sweep.converged
    assert len(result.sweep.sampled_frequencies) == 3
    assert np.isfinite(np.asarray(result.s.values)).all()


def test_bddc_gmres_samples_match_direct(line, adaptive):
    problem = gfem.Driven(frequency=GRID, sweep="adaptive")
    result = gfem.solve(line, problem)
    np.testing.assert_allclose(result.s.values, adaptive.s.values, atol=1e-5)


def test_an_adaptive_sweep_that_does_not_converge_has_no_partial_result(
    line, monkeypatch
):
    # the first sample misses; the reduced model needs every sample
    problem = gfem.Driven(frequency=GRID, sweep="adaptive", samples=[GRID[7], GRID[30]])
    cannot_converge_at(monkeypatch, (GRID[7],))
    options = gfem.SolverOptions(linear_solver="bddc_gmres")
    with pytest.raises(
        gfem.LinearSolverConvergenceError, match="did not converge"
    ) as caught:
        gfem.solve(line, problem, options=options)
    assert caught.value.frequency == GRID[7]
    assert f"{GRID[7]:g} Hz" in str(caught.value)
    assert caught.value.result is None
    assert caught.value.unsolved is None


def test_a_greedy_sweep_cannot_be_traced(line):
    problem = gfem.Driven(frequency=GRID, sweep="adaptive")

    @jax.jit
    def s(model):
        return gfem.solve(model, problem, options=OPTIONS).s.values

    with pytest.raises(NotImplementedError, match="samples="):
        s(line)

    field = gfem.MaterialField.uniform("dielectric", gfem.Material(eps_r=EPS))
    model = line.replace(materials={"dielectric": field})

    def s21(m):
        result = gfem.solve(m, problem, options=OPTIONS)
        return jax.numpy.abs(result.s["p2", "p1"][0])

    with pytest.raises(NotImplementedError, match="frozen sample set"):
        jax.grad(s21)(model)


def _eps_model(line, eps):
    return line.with_material("dielectric", gfem.MaterialField(eps_r=eps))


FROZEN = gfem.Driven(
    frequency=GRID,
    sweep="adaptive",
    samples=[GRID[0], GRID[-1], GRID[30]],
    keep_fields=[GRID[3], GRID[30]],
)


def test_frozen_samples_trace_and_match_the_eager_solve(line):
    eager = gfem.solve(line, FROZEN, options=OPTIONS)
    n = len(line.mesh.tetrahedra)

    @jax.jit
    def solve(eps):
        result = gfem.solve(_eps_model(line, eps), FROZEN, options=OPTIONS)
        return result.s.values, result.e_fields.basis, result.e_fields.coordinates

    s, basis, coordinates = solve(jax.numpy.full(n, EPS * (1 - 1e-3j)))
    np.testing.assert_allclose(s, eager.s.values, rtol=1e-10, atol=1e-12)
    # late basis vectors are ill-conditioned in the design: compare fields
    assert basis.shape == eager.e_fields.basis.shape
    fields = np.asarray(basis @ coordinates.T)
    expected = np.asarray(eager.e_fields.basis @ eager.e_fields.coordinates.T)
    np.testing.assert_allclose(fields, expected, atol=1e-9 * abs(expected).max())


def test_frozen_samples_pad_the_basis_to_a_size_known_before_the_solve(line):
    # two excited ports, real and imaginary parts, three samples
    result = gfem.solve(line, FROZEN, options=OPTIONS)
    assert result.e_fields.basis.shape[1] == 2 * 2 * 3
    size = result.sweep.basis_size
    assert size <= 12
    assert not np.asarray(result.e_fields.basis)[:, size:].any()


def test_a_traced_sweep_reports_its_frozen_samples_but_no_diagnostics(line):
    seen = {}

    def objective(eps):
        result = gfem.solve(_eps_model(line, eps), FROZEN, options=OPTIONS)
        seen["sweep"] = result.sweep
        return jax.numpy.sum(jax.numpy.abs(result.s.values) ** 2)

    n = len(line.mesh.tetrahedra)
    jax.grad(objective)(jax.numpy.full(n, EPS))
    info = seen["sweep"]
    assert info.kind == "adaptive"
    assert info.sampled_frequencies == FROZEN.samples
    assert sum(info.solved) == 3
    assert info.tolerance == OPTIONS.adaptive_tolerance
    assert info.sample_errors is None
    assert info.converged is None
    assert info.basis_size is None
    assert "no diagnostics" in repr(info)


def test_the_frozen_basis_itself_is_not_differentiable(line):
    n = len(line.mesh.tetrahedra)

    def basis_sum(eps):
        result = gfem.solve(_eps_model(line, eps), FROZEN, options=OPTIONS)
        return jax.numpy.sum(result.e_fields.basis)

    with pytest.raises(gfem.NonDifferentiableError, match="e_basis"):
        jax.grad(basis_sum)(jax.numpy.full(n, EPS))


def test_adaptive_job_and_result_round_trip(line, tmp_path):
    grid = GRID[::4]
    problem = gfem.Driven(
        frequency=grid,
        sweep="adaptive",
        samples=[grid[0], grid[-1], grid[6]],
        keep_fields=GRID[8],
    )
    job = gfem.Job.from_json(
        gfem.Job(line, problem, OPTIONS).to_json(tmp_path / "j.json")
    )
    assert job.problem == problem
    result = job.run(out=tmp_path / "out")
    loaded = gfem.load_result(tmp_path / "out")
    assert loaded.sweep == result.sweep
    np.testing.assert_array_equal(loaded.s.values, result.s.values)
    e, back = result.e_field(GRID[8], "p2"), loaded.e_field(GRID[8], "p2")
    np.testing.assert_array_equal(back.coefficients, e.coefficients)
    np.testing.assert_array_equal(loaded.e_fields.basis, result.e_fields.basis)
