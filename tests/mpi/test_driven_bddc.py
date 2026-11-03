"""A lumped-port driven solve through the real `gfem.solve` on the NGSolve wheel.

Every test compares ranks with the same function in pytest's one process: S bit
for bit across ranks (`on_ranks`), to `S_GATE` against one process, and the Krylov
iterations exactly. Only the count shows a Krylov method that lost its global
products, since such a method still converges to the right S (ADR 0069).
"""

import os
import sys
from pathlib import Path

import gradient_cases as cases
import numpy as np
import pytest
from ranks import RanksFailed, on_ranks, on_ranks_argv

import gfem
from gfem import _mpi, _petsc
from gfem._engine.cache import ENTRIES, Entry
from gfem._engine.driven import DrivenRule, driven_operator
from gfem._engine.port_network import parallel_scale
from gfem._engine.runtime import task_manager
from gfem._engine.system_operator import SystemOperator
from gfem._linalg import ResolvedLinearSolver, bddc, blas, coarse, prepare
from gfem._linalg.base import local_matrix
from gfem._linalg.direct import INVERSE
from gfem._linalg.gmres import Gmres
from gfem._linalg.pardiso import pardiso_inverse
from gfem._linalg.pieces import AssembledPieces

#: Relative to the largest entry of the one-process S: ADR 0070's BDDC gate.
S_GATE = 1e-9
FREQUENCY = 5e9
#: Close enough for one BDDC setup to serve all three (ADR 0036).
SWEEP = (4.9e9, 5.0e9, 5.1e9)


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _largest(comm, values: np.ndarray) -> float:
    from mpi4py import MPI

    return comm.allreduce(float(np.abs(values).max(initial=0.0)), op=MPI.MAX)


def _s_and_iterations(directory, frequencies):
    """S, the Krylov iterations and BDDC's setups of `layered_line`, one thread each."""
    seen = {"setups": 0}
    forward = DrivenRule.forward
    set_up = bddc._set_up

    def count_setups(*args, **kwargs):
        seen["setups"] += 1
        return set_up(*args, **kwargs)

    def spy(self, model, *, keep):
        outputs, entry = forward(self, model, keep=keep)
        seen["iterations"] = outputs["iterations"]
        return outputs, entry

    DrivenRule.forward = spy
    bddc._set_up = count_setups
    try:
        result = gfem.solve(
            cases.layered_line(directory),
            gfem.Driven(list(frequencies)),
            options=cases.tight("bddc_gmres", threads=1),
        )
    finally:
        DrivenRule.forward = forward
        bddc._set_up = set_up
    return np.asarray(result.s.values), np.asarray(seen["iterations"]), seen["setups"]


def _assert_agrees(directory, frequencies, ranks):
    s_one, iterations_one, setups_one = _s_and_iterations(directory, frequencies)
    (s, iterations, setups), *_ = on_ranks(
        _s_and_iterations, directory, frequencies, ranks=ranks, limit=300
    )
    error = np.abs(s - s_one).max() / np.abs(s_one).max()
    print(  # noqa: T201
        f"ranks={ranks} S error={error:.3g} iterations={iterations.tolist()} "
        f"one process={iterations_one.tolist()} setups={setups}"
    )
    assert error < S_GATE, f"S off by {error:.3g} relative"
    np.testing.assert_array_equal(iterations, iterations_one)
    assert setups == setups_one
    return setups


@pytest.mark.parametrize("ranks", [3, 4, 8])
def test_lumped_s_and_iterations_match_one_process(tmp_path, ranks):
    _assert_agrees(tmp_path, (FREQUENCY,), ranks)


@pytest.mark.parametrize("ranks", [3, 4])
def test_a_sweep_reusing_bddc_matches_one_process(tmp_path, ranks):
    assert _assert_agrees(tmp_path, SWEEP, ranks) < len(SWEEP)


def test_a_capped_warm_solve_sets_bddc_up_on_every_rank(tmp_path):
    # BDDC set up at 1 GHz and reused at 60 GHz stops at its cap, set up again, and
    # goes on from the partial iterate, by iteration counts all ranks hold
    assert _assert_agrees(tmp_path, (1e9, 60e9), 4) == 2


def _a_short_stop(directory):
    """S with one GMRES stopped early, above the tolerance: it restarts from there."""
    original = Gmres._SolveImpl
    iterations = []

    def stops_short(self, rhs, sol):
        maxiter = self.maxiter
        if not iterations:
            self.maxiter = min(2, maxiter)
        try:
            original(self, rhs=rhs, sol=sol)
        finally:
            self.maxiter = maxiter
        iterations.append(self.iterations)

    Gmres._SolveImpl = stops_short
    try:
        s, *_ = _s_and_iterations(directory, (FREQUENCY,))
    finally:
        Gmres._SolveImpl = original
    return s, iterations[:2]


def test_a_short_stop_restarts_on_every_rank(tmp_path):
    s_one, *_ = _s_and_iterations(tmp_path, (FREQUENCY,))
    (s, iterations), *_ = on_ranks(_a_short_stop, tmp_path, ranks=4, limit=300)
    assert iterations[0] == 2
    assert iterations[1] > 0
    assert np.abs(s - s_one).max() / np.abs(s_one).max() < S_GATE


# --- the coarse problem's two routes ------------------------------------------------

needs_stack = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")


#: `coarse.reduction` per route: a tiny mesh's own is to one rank.
EVERY_RANK, HALF, ONE_RANK = "every rank", "half", "one rank"
REDUCTIONS = {
    EVERY_RANK: lambda _, __: 1,
    HALF: lambda _, __: 2,
    ONE_RANK: coarse.reduction,
}


def _on_a_route(directory, frequencies, route):
    """`_s_and_iterations` with the coarse problem on ``route``, and what factored it.

    ``"rank 0"`` is what a wheel-only install runs.
    """
    factored = []
    if route == "rank 0":
        bddc._coarse_on_mumps = lambda *_, **__: False
    else:
        coarse.reduction = REDUCTIONS[route]
        set_up = coarse.CoarseMumps.set_up

        def spy(self):
            set_up(self)
            pc = self._ksp.getPC()
            factored.append((pc.getType(), self._factor))

        coarse.CoarseMumps.set_up = spy
    return (*_s_and_iterations(directory, frequencies), factored)


def _assert_routes_agree(directory, frequencies, ranks, *, routes):
    reference, *_ = on_ranks(
        _on_a_route, directory, frequencies, "rank 0", ranks=ranks, limit=300
    )
    s_zero, iterations_zero, setups_zero, _ = reference
    seen = {}
    for route in routes:
        (s, iterations, setups, factored), *_ = on_ranks(
            _on_a_route, directory, frequencies, route, ranks=ranks, limit=300
        )
        error = np.abs(s - s_zero).max() / np.abs(s_zero).max()
        print(  # noqa: T201
            f"ranks={ranks} {route}: S error={error:.3g} "
            f"iterations={iterations.tolist()} rank 0={iterations_zero.tolist()} "
            f"factored={factored}"
        )
        assert error < S_GATE, f"S off by {error:.3g} relative"
        np.testing.assert_array_equal(iterations, iterations_zero)
        assert setups == setups_zero
        seen[route] = factored
    return seen


@needs_stack
@pytest.mark.parametrize("ranks", [3, 4, 8])
def test_coarse_on_mumps_matches_the_rank_0_route(tmp_path, ranks):
    seen = _assert_routes_agree(tmp_path, (FREQUENCY,), ranks, routes=[EVERY_RANK])
    assert seen[EVERY_RANK] == [("cholesky", 1)]


@needs_stack
@pytest.mark.parametrize("ranks", [3, 4, 8])
def test_telescoping_changes_neither_iterations_nor_s(tmp_path, ranks):
    seen = _assert_routes_agree(tmp_path, (FREQUENCY,), ranks, routes=[HALF, ONE_RANK])
    assert seen[HALF] == [("telescope", 2)]
    assert seen[ONE_RANK] == [("telescope", ranks)]


@needs_stack
def test_a_sweep_reusing_bddc_on_either_coarse_route(tmp_path):
    _assert_routes_agree(tmp_path, SWEEP, 4, routes=[EVERY_RANK, ONE_RANK])


def _with_the_default_inverse_forbidden(directory):
    """NGSolve's default inverse raises: the coarse route must not fall back on it."""
    import ngsolve as ngs

    class Forbidden(ngs.la.SparseFactorizationInterface):
        def __init__(self, *_, **__):
            raise AssertionError("gfem used NGSolve's default inverse type")

    ngs.la.RegisterInverseType("gfem_forbidden", Forbidden)
    ngs.BaseMatrix.SetDefaultInverseType("gfem_forbidden")
    return _s_and_iterations(directory, (FREQUENCY,))[1]


@needs_stack
def test_the_coarse_route_never_reads_ngsolves_default_inverse(tmp_path):
    iterations, *_ = on_ranks(_with_the_default_inverse_forbidden, tmp_path, ranks=3)
    assert iterations.size


def _rank_0_flags(directory):
    """The flags BDDC is given when the coarse problem stays on rank 0."""
    import ngsolve as ngs

    seen = []
    made = ngs.Preconditioner

    def spy(form, kind, **flags):
        seen.append(flags)
        return made(form, kind, **flags)

    bddc._coarse_on_mumps = lambda *_, **__: False
    bddc.ngs.Preconditioner = spy
    _s_and_iterations(directory, (FREQUENCY,))
    return seen


def test_the_rank_0_route_names_its_inverse(tmp_path):
    # the coarse rule of one process: PARDISO with the mkl extra, else sparse Cholesky
    (flags,), *_ = on_ranks(_rank_0_flags, tmp_path, ranks=3)
    print(f"rank 0 inverse: {flags}")  # noqa: T201
    assert flags.keys() == {"inverse"}
    assert flags["inverse"] in (
        pardiso_inverse("complex_symmetric", is_complex=True),
        INVERSE,
    )


def _forced_coarse_failures(directory, route, failures):
    """INFOG(1)=-9 ``failures`` times on the real route: a retry's count."""
    coarse.reduction = REDUCTIONS[route]
    calls = []
    real = coarse.CoarseMumps._agreed_failure

    def forced(self):
        calls.append(1)
        return coarse.MUMPS_WORKSPACE_ERROR if len(calls) <= failures else real(self)

    coarse.CoarseMumps._agreed_failure = forced
    options = cases.tight("bddc_gmres", threads=1)
    problem = gfem.Driven(FREQUENCY)
    try:
        s = gfem.solve(cases.layered_line(directory), problem, options=options).s.values
    except _mpi.AgreedError as err:
        return "coarse problem failed" in str(err), len(calls), _world().allreduce(1)
    return np.asarray(s), len(calls)


@needs_stack
@pytest.mark.parametrize("route", [EVERY_RANK, ONE_RANK])
def test_a_coarse_workspace_too_small_retries_on_a_new_ksp(tmp_path, route):
    s_one, *_ = _s_and_iterations(tmp_path, (FREQUENCY,))
    (s, calls), *_ = on_ranks(
        _forced_coarse_failures, tmp_path, route, 1, ranks=4, limit=300
    )
    assert calls == 2
    assert np.abs(s - s_one).max() < S_GATE * np.abs(s_one).max()


def _a_singular_coarse_problem(directory, route):
    """A real failure in MUMPS, not a forced value: the real paths must see it."""
    coarse.reduction = REDUCTIONS[route]
    built = coarse.distributed_aij

    def zero(*args):
        aij, index = built(*args)
        aij.zeroEntries()
        return aij, index

    coarse.distributed_aij = zero
    options = cases.tight("bddc_gmres", threads=1)
    problem = gfem.Driven(FREQUENCY)
    try:
        gfem.solve(cases.layered_line(directory), problem, options=options)
    except _mpi.AgreedError as err:
        return str(err), _world().allreduce(1)
    raise AssertionError("a zero matrix factored")


@needs_stack
@pytest.mark.parametrize("route", [EVERY_RANK, ONE_RANK])
def test_a_factorization_that_fails_is_an_agreed_error_with_no_nan(tmp_path, route):
    (text, ranks), *_ = on_ranks(
        _a_singular_coarse_problem, tmp_path, route, ranks=4, limit=300
    )
    assert "coarse problem failed" in text
    assert ranks == 4
    print(text)  # noqa: T201


# --- what lies under the solve ------------------------------------------------------


def _system(directory):
    model = cases.layered_line(directory)
    comm = _mpi.resolve_comm()
    operator = driven_operator(model, 2, comm=comm)
    linear = ResolvedLinearSolver.resolve(
        gfem.SolverOptions(linear_solver="bddc_gmres"),
        default="bddc_gmres",
        problem="driven",
    )
    return comm, SystemOperator(operator, [], linear, width=1, adjoint_tolerance=1e-8)


def _piece_sum_against_a_fresh_assembly(directory):
    comm, system = _system(directory)
    worst = 0.0
    for frequency in SWEEP:
        operator = system.operator_at(2 * np.pi * frequency)
        operator.form.Assemble()
        fresh = local_matrix(operator.form.mat, comm).AsVector().FV().NumPy()
        summed = operator.piece_sum.matrix()
        values = local_matrix(summed, comm).AsVector().FV().NumPy()
        worst = max(worst, _largest(comm, values - fresh) / _largest(comm, fresh))
    return worst


def test_the_piece_sum_equals_a_fresh_assembly(tmp_path):
    worst, *_ = on_ranks(_piece_sum_against_a_fresh_assembly, tmp_path, ranks=4)
    assert worst < 1e-14


def _disjoint_pieces(directory):
    """Two pieces on two surfaces: no rank's pattern holds the other's entries."""
    import ngsolve as ngs

    comm, system = _system(directory)

    def on(surface):
        def integrand(u, v):
            return u.Trace() * v.Trace() * ngs.ds(surface)

        return integrand

    pieces = AssembledPieces(
        system.operator.space, [on("port1"), on("port2")], comm=comm
    )
    return pieces.pattern is None


def test_no_common_pattern_is_the_same_answer_on_every_rank(tmp_path):
    outside, *_ = on_ranks(_disjoint_pieces, tmp_path, ranks=4)
    assert outside


def _layout_counts(directory):
    """Each product of the layout on one load: 1 at each DOF, split to its owner."""
    comm, system = _system(directory)
    space = system.operator.space
    layout = system.operator.layout
    load = layout.owners.astype(complex)
    state = layout.exchange(load)
    return {
        "pair": layout.pair(load, state),
        "pair of columns": layout.pair(np.stack([load, load], axis=1), state),
        "inner": layout.inner(load, load),
        "norm": layout.norm(load),
        "parallel": parallel_scale(2j * load, load, layout),
        # one entry off, on the last rank alone: each rank must see it
        "not parallel": parallel_scale(_nudged(comm, 2j * load), load, layout),
        "global": space.ndofglobal,
        "local": comm.allreduce(space.ndof),
    }


def _nudged(comm, load):
    if comm.rank == comm.size - 1:
        load[0] += 1.0
    return load


def test_a_layout_counts_each_dof_once(tmp_path):
    rows, *_ = on_ranks(_layout_counts, tmp_path, ranks=4)
    n = rows["global"]
    assert rows["local"] > n  # some DOF is shared, or this proves nothing
    assert rows["pair"] == n
    np.testing.assert_array_equal(rows["pair of columns"], [n, n])
    assert rows["inner"] == n
    assert rows["norm"] == pytest.approx(np.sqrt(n), rel=1e-14)
    assert rows["parallel"] == 2j
    assert rows["not parallel"] is None


def _apply_then_solve(directory):
    """``apply`` takes a state and gives a load; ``solve`` brings the state back."""
    _, system = _system(directory)
    operator = system.operator_at(2 * np.pi * FREQUENCY)
    handle = prepare(
        operator, symmetry="complex_symmetric", kind="bddc_gmres", tolerance=1e-12
    )
    layout = handle.layout
    free = np.array(operator.freedofs, dtype=bool)
    state = layout.exchange(np.where(layout.owners & free, 1.0 + 0.5j, 0.0))
    load = handle.apply(state)
    back, counts = handle.solve_counted(load)
    return (
        layout.pair(load, state),
        layout.pair(handle.apply(back), state),
        counts,
    )


def test_apply_takes_a_state_and_gives_a_load(tmp_path):
    (energy, again, counts), *_ = on_ranks(_apply_then_solve, tmp_path, ranks=4)
    assert abs(again - energy) < 1e-8 * abs(energy)
    assert counts[0] > 0


def _prepare_a_general_operator(directory):
    _, system = _system(directory)
    operator = system.operator_at(2 * np.pi * FREQUENCY)
    try:
        prepare(operator, symmetry="general", kind="bddc_gmres")
    except gfem.DistributedUnsupportedError as err:
        return err.feature
    raise AssertionError("a general operator was prepared")


def test_a_general_operator_is_refused_on_every_rank(tmp_path):
    feature, *_ = on_ranks(_prepare_a_general_operator, tmp_path, ranks=3)
    assert feature == "general_operator"


# --- what a job sees ----------------------------------------------------------------


def _entries_kept(sizes_by_rank, budget):
    """Rank-dependent sizes: every rank must evict the same entries."""
    world = _world()

    class Sized:
        def __init__(self, nbytes):
            self.nbytes = nbytes

        def free(self):
            pass

    keys = [
        ENTRIES.put(Entry("em", Sized(size)), budget, world)
        for size in sizes_by_rank[world.rank]
    ]
    kept = [ENTRIES.get(k) is not None for k in keys]
    ENTRIES.clear()
    return kept


def test_every_rank_evicts_the_same_entries():
    # one node here: 102 bytes per put, so a budget of 150 keeps the newest alone,
    # and a rank's own bytes would keep all three on two ranks of three
    sizes = [[100, 100, 100], [1, 1, 1], [1, 1, 1]]
    kept, *_ = on_ranks(_entries_kept, sizes, 150, ranks=3)
    assert kept == [False, False, True]


def _mismatched_input(directory):
    """One rank changes one material: every rank hears which leaf, then goes on."""
    world = _world()
    model = cases.layered_line(directory)
    eps = 2.0 if world.rank == world.size - 1 else 1.0
    material = gfem.Material(eps_r=eps)
    model = model.replace(materials={**model.materials, "sub": material})
    try:
        gfem.solve(model, gfem.Driven(FREQUENCY))
    except _mpi.AgreedError as err:
        return "sub" in str(err), world.allreduce(1)
    raise AssertionError("different inputs ran")


def test_a_mismatched_input_raises_on_every_rank_naming_the_leaf(tmp_path):
    named, *_ = on_ranks(_mismatched_input, tmp_path, ranks=3, limit=60)
    assert named == (True, 3)


def _one_rank_fails_in_the_solve(directory):
    world = _world()
    if world.rank == world.size - 1:
        import gfem._linalg.gmres as module

        def broken(*_args, **_kwargs):
            raise OSError("this rank's own failure")

        module.Gmres._gmres = broken
    gfem.solve(cases.layered_line(directory), gfem.Driven(FREQUENCY))


def test_a_failure_on_one_rank_in_the_solve_ends_the_job(tmp_path):
    with pytest.raises(RanksFailed, match="this rank's own failure"):
        on_ranks(_one_rank_fails_in_the_solve, tmp_path, ranks=3, limit=60)


def _a_miss_on_every_rank(directory):
    """Two iterations cannot converge: the error is agreed, and a job goes on."""
    world = _world()
    prepare.__kwdefaults__["maxiter"] = 2
    try:
        gfem.solve(cases.layered_line(directory), gfem.Driven(list(SWEEP)))
    except gfem.LinearSolverConvergenceError as err:
        return err.frequency, list(err.unsolved), world.allreduce(1)
    raise AssertionError("two iterations converged")


def test_a_convergence_error_reaches_every_rank(tmp_path):
    (frequency, unsolved, ranks), *_ = on_ranks(
        _a_miss_on_every_rank, tmp_path, ranks=3, limit=120
    )
    assert (frequency, unsolved, ranks) == (SWEEP[0], list(SWEEP), 3)


def _threads_inside_a_task_manager(*, read_the_mask):
    """What ``threads=None`` gives NGSolve and each BLAS, variable or no variable."""
    import ngsolve as ngs

    if read_the_mask:
        del os.environ["NGS_NUM_THREADS"]
    comm = _mpi.resolve_comm()
    with task_manager(None, comm):
        return ngs.GetNumThreads(), sorted({lib.get() for lib in blas.libraries()})


def test_threads_none_reads_the_variable_first():
    (count, counts), *_ = on_ranks(
        _threads_inside_a_task_manager, read_the_mask=False, ranks=3, threads=2
    )
    assert count == 2
    assert counts == [2]


def test_threads_none_without_the_variable_is_one_count_on_every_rank():
    # bitwise across ranks by `on_ranks`; ranks share one mask here, unbound
    (count, counts), *_ = on_ranks(
        _threads_inside_a_task_manager, read_the_mask=True, ranks=3
    )
    assert count == max(1, _physical_cores() // 3)
    assert counts == [count]


def _physical_cores() -> int:
    mask = frozenset(os.sched_getaffinity(0))
    return _mpi.node_share(mask, [mask])


def _blas_in_each_phase(directory):
    """Each loaded BLAS's count in a setup, in a solve, and after: ADR 0076's phases."""
    import gfem._linalg.gmres as krylov

    seen = {"setup": set(), "solve": set()}
    set_up, solve = bddc._set_up, krylov.Gmres._gmres

    def spy_set_up(*args, **kwargs):
        seen["setup"] |= {lib.get() for lib in blas.libraries()}
        return set_up(*args, **kwargs)

    def spy_solve(*args, **kwargs):
        seen["solve"] |= {lib.get() for lib in blas.libraries()}
        return solve(*args, **kwargs)

    bddc._set_up, krylov.Gmres._gmres = spy_set_up, spy_solve
    gfem.solve(
        cases.layered_line(directory),
        gfem.Driven(list(SWEEP)),
        options=gfem.SolverOptions(linear_solver="bddc_gmres", threads=2),
    )
    return seen, sorted({lib.get() for lib in blas.libraries()})


def test_blas_runs_at_the_ranks_threads_in_a_setup_and_at_one_in_a_solve(tmp_path):
    (seen, after), *_ = on_ranks(_blas_in_each_phase, tmp_path, ranks=3, threads=1)
    assert seen == {"setup": {2}, "solve": {1}}
    assert after == [2]


#: What a launched script imports the test helpers from.
SCRIPT_PATH = (Path(__file__).parents[1], Path(__file__).parents[2] / "tools")
WAITING = """
import sys
import time

from mpi4py import MPI

import gfem
from gfem import _mpi
import gradient_cases as cases

_mpi.ENTRY_PATIENCE = 0.2
model = cases.layered_line(sys.argv[1])
if MPI.COMM_WORLD.rank == 1:
    time.sleep(2)
_mpi.entry_check(
    MPI.COMM_WORLD, "solve", model, gfem.Driven(5e9), gfem.SolverOptions()
)
"""


def test_a_long_wait_at_the_door_says_so_once_per_rank_and_goes_on(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(WAITING)
    code, log = on_ranks_argv(
        [sys.executable, str(script), str(tmp_path)],
        ranks=3,
        limit=60,
        env={"PYTHONPATH": os.pathsep.join(map(str, SCRIPT_PATH))},
    )
    assert code == 0, log
    assert log.count("waiting over 0.2 s") == 2, log
