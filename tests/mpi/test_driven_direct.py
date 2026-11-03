"""`direct` and `petsc` across ranks: PETSc MUMPS on the MPI PETSc stack (ADR 0068).

The tests of the solve need that stack and are skipped without it. S is compared bit
for bit across ranks (`on_ranks`) and to `S_GATE` against one process's `direct`.
What a stack without MPI does sits at the end, and is skipped on the MPI stack.
"""

import contextlib
import os
import sys
from pathlib import Path

import gradient_cases as cases
import numpy as np
import pytest
import structured_meshes as sm
from ranks import on_ranks, on_ranks_argv

import gfem
from gfem import _mpi, _petsc
from gfem._engine.driven import driven_operator
from gfem._engine.system_operator import SystemOperator
from gfem._linalg import ResolvedLinearSolver, blas, prepare
from gfem._linalg import petsc as petsc_handle

#: Relative to the largest entry of the one-process S: ADR 0070's direct gate.
S_GATE = 1e-11
FREQUENCY = 5e9
#: Far enough apart that a value added to the previous frequency's shows in S.
TWO = (4e9, 6e9)

needs_stack = pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack")


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _s(directory, frequencies, options):
    model = cases.layered_line(directory)
    problem = gfem.Driven(list(frequencies))
    return np.asarray(gfem.solve(model, problem, options=options).s.values)


def _assert_agrees(directory, frequencies, solver, ranks, **kwargs):
    one = cases.tight("direct", threads=1)
    s_one = _s(directory, frequencies, one)
    options = cases.tight(solver, threads=1, **kwargs)
    s, *_ = on_ranks(_s, directory, frequencies, options, ranks=ranks, limit=300)
    error = np.abs(s - s_one).max() / np.abs(s_one).max()
    print(f"{solver} ranks={ranks} S error={error:.3g}")  # noqa: T201
    assert error < S_GATE, f"S off by {error:.3g} relative"


@needs_stack
@pytest.mark.parametrize("ranks", [3, 4, 8, 16])
def test_direct_s_matches_one_process(tmp_path, ranks):
    _assert_agrees(tmp_path, (FREQUENCY,), "direct", ranks)


@needs_stack
@pytest.mark.parametrize("ranks", [3, 4])
def test_petsc_by_default_is_the_same_exact_solve(tmp_path, ranks):
    _assert_agrees(tmp_path, (FREQUENCY,), "petsc", ranks)


@needs_stack
def test_a_second_frequency_replaces_the_first_ones_values(tmp_path):
    # with ADD_VALUES the second frequency factors the sum of both, with no error
    _assert_agrees(tmp_path, TWO, "direct", 4)


def _on_half_of_the_ranks(directory, solver):
    """Two solves side by side, on two communicators of two ranks."""
    world = _world()
    half = world.Split(world.rank % 2)
    model = cases.layered_line(directory)
    options = cases.tight(solver, threads=1)
    result = gfem.solve(model, gfem.Driven(FREQUENCY), options=options, comm=half)
    half.Free()
    return np.asarray(result.s.values)


@needs_stack
@pytest.mark.parametrize(("solver", "gate"), [("direct", S_GATE), ("bddc_gmres", 1e-9)])
def test_a_solve_runs_on_a_split_communicator(tmp_path, solver, gate):
    s_one = _s(tmp_path, (FREQUENCY,), cases.tight("direct", threads=1))
    pieces = on_ranks(
        _on_half_of_the_ranks, tmp_path, solver, ranks=4, replicated=False, limit=300
    )
    for s in pieces:
        assert np.abs(s - s_one).max() < gate * np.abs(s_one).max()


# --- under the solve ----------------------------------------------------------------


def _system(directory, solver="direct"):
    model = cases.layered_line(directory)
    comm = _mpi.resolve_comm()
    operator = driven_operator(model, 2, comm=comm)
    linear = ResolvedLinearSolver.resolve(
        gfem.SolverOptions(linear_solver=solver),
        default="direct",
        problem="driven",
        comm=comm,
    )
    return comm, SystemOperator(operator, [], linear, width=1, adjoint_tolerance=1e-8)


def _handle(directory, frequency=FREQUENCY, **kwargs):
    comm, system = _system(directory)
    operator = system.operator_at(2 * np.pi * frequency)
    handle = prepare(operator, symmetry="complex_symmetric", kind="direct", **kwargs)
    return comm, system, operator, handle


def _columns_of_a_block(directory):
    """A block solve against its columns, on rows no rank agrees on by itself."""
    comm, _, operator, handle = _handle(directory)
    layout = handle.layout
    free = np.array(operator.freedofs, dtype=bool)
    state = layout.exchange(np.where(layout.owners & free, 1.0 + 0.5j, 0.0))
    load = handle.apply(state)
    block = np.stack([load, 2j * load, (1 - 3j) * load], axis=1)
    together = handle.solve(block)
    apart = np.stack([handle.solve(block[:, j]) for j in range(3)], axis=1)
    transposed = handle.solve(block, transpose=True)
    assert handle.kind == "direct"
    assert handle.ndof == len(free)
    handle.free()
    largest = comm.allreduce(np.abs(together).max(initial=0.0), op=_max())
    return {
        "block to its columns": comm.allreduce(
            np.abs(together - apart).max(initial=0.0), op=_max()
        ),
        "transposed": comm.allreduce(
            np.abs(together - transposed).max(initial=0.0), op=_max()
        ),
        "to the state": comm.allreduce(
            np.abs(together[:, 0] - state).max(initial=0.0), op=_max()
        ),
        "on a fixed row": comm.allreduce(
            np.abs(together[~free]).max(initial=0.0), op=_max()
        ),
        "scale": largest,
    }


def _max():
    from mpi4py import MPI

    return MPI.MAX


@needs_stack
def test_a_block_a_column_and_a_transposed_solve_agree(tmp_path):
    rows, *_ = on_ranks(_columns_of_a_block, tmp_path, ranks=4)
    assert rows["block to its columns"] < 1e-12 * rows["scale"]
    assert rows["transposed"] == 0
    assert rows["to the state"] < 1e-11 * rows["scale"]
    assert rows["on a fixed row"] == 0


def _memory_of_a_handle(directory):
    """This rank's bytes: the sum over ranks must count MUMPS's memory once."""
    comm, _, _, handle = _handle(directory)
    factor = handle._ksp.getPC().getFactorMatrix()
    info = int(factor.getMumpsInfo(22))
    total = int(factor.getMumpsInfog(22))
    nbytes = handle.nbytes
    handle.free()
    return nbytes >= 10**6 * info, comm.allreduce(info), total, handle.nbytes


@needs_stack
def test_a_handle_reports_its_rank_s_memory(tmp_path):
    (counted, summed, total, freed), *_ = on_ranks(
        _memory_of_a_handle, tmp_path, ranks=3
    )
    assert counted
    assert summed == total
    assert freed == 0


def _forced_workspace_errors(directory, failures):
    """INFOG(1)=-9 ``failures`` times, on the real path: each new KSP's ICNTL(14)."""
    workspace = []
    real = petsc_handle.RankPetscHandle._agreed_failure

    def forced(self):
        factor = self._ksp.getPC().getFactorMatrix()
        workspace.append(int(factor.getMumpsIcntl(14)))
        if len(workspace) <= failures:
            return petsc_handle.MUMPS_WORKSPACE_ERROR
        return real(self)

    petsc_handle.RankPetscHandle._agreed_failure = forced
    options = cases.tight("direct", threads=1)
    model = cases.layered_line(directory)
    problem = gfem.Driven(FREQUENCY)
    try:
        s = np.asarray(gfem.solve(model, problem, options=options).s.values)
    except _mpi.AgreedError as err:
        return "factorization failed" in str(err), workspace, _world().allreduce(1)
    return s, workspace


@needs_stack
def test_a_too_small_workspace_retries_on_a_new_ksp(tmp_path):
    s_one = _s(tmp_path, (FREQUENCY,), cases.tight("direct", threads=1))
    (s, workspace), *_ = on_ranks(_forced_workspace_errors, tmp_path, 1, ranks=4)
    assert workspace == [60, 120]
    assert np.abs(s - s_one).max() < S_GATE * np.abs(s_one).max()


@needs_stack
def test_a_workspace_too_small_for_every_retry_is_an_agreed_error(tmp_path):
    # no NaN comes back, and every rank goes on to a later collective
    limit = petsc_handle.MUMPS_RETRIES + 1
    (named, workspace, ranks), *_ = on_ranks(
        _forced_workspace_errors, tmp_path, limit, ranks=4
    )
    assert named
    assert workspace == [60, 120, 240]
    assert ranks == 4


def _working_memory_of_one_megabyte(directory):
    """MUMPS itself out of memory, not a forced value."""
    options = gfem.SolverOptions(
        linear_solver="petsc", petsc_options={"mat_mumps_icntl_23": 1}
    )
    model = cases.layered_line(directory)
    try:
        gfem.solve(model, gfem.Driven(FREQUENCY), options=options)
    except _mpi.AgreedError as err:
        return str(err), _world().allreduce(1)
    raise AssertionError("one megabyte factored the system")


@needs_stack
def test_a_real_factorization_failure_is_an_agreed_error_with_no_nan(tmp_path):
    (text, ranks), *_ = on_ranks(_working_memory_of_one_megabyte, tmp_path, ranks=4)
    assert "INFOG(1)=-" in text
    assert ranks == 4
    print(text)  # noqa: T201


def _a_misnamed_option(directory):
    options = gfem.SolverOptions(
        linear_solver="petsc", petsc_options={"mat_mumps_icntl_not_one": 1}
    )
    model = cases.layered_line(directory)
    try:
        gfem.solve(model, gfem.Driven(FREQUENCY), options=options)
    except ValueError as err:
        return "mat_mumps_icntl_not_one" in str(err), _world().allreduce(1)
    raise AssertionError("an unread option passed")


@needs_stack
def test_an_unread_option_raises_on_every_rank_and_the_job_goes_on(tmp_path):
    (named, ranks), *_ = on_ranks(_a_misnamed_option, tmp_path, ranks=3)
    assert named
    assert ranks == 3


@needs_stack
def test_the_users_options_sit_over_gfems(tmp_path):
    # METIS's sequential analysis on one rank, gfem's PT-Scotch otherwise
    _assert_agrees(
        tmp_path,
        (FREQUENCY,),
        "petsc",
        3,
        petsc_options={"mat_mumps_icntl_28": 1, "mat_mumps_icntl_7": 5},
    )


@needs_stack
def test_a_krylov_method_runs_across_ranks(tmp_path):
    _assert_agrees(tmp_path, TWO, "petsc", 3, petsc_options={"ksp_type": "gmres"})


def _blas_in_a_factorization(directory):
    """Each BLAS's count in every factorization of a sweep, and after (ADR 0076)."""
    seen = []
    held = blas.held

    @contextlib.contextmanager
    def spy(threads):
        with held(threads):
            seen.append(sorted({lib.get() for lib in blas.libraries()}))
            yield

    blas.held = spy
    options = gfem.SolverOptions(order=2, linear_solver="direct", threads=2)
    model = cases.layered_line(directory)
    gfem.solve(model, gfem.Driven(list(TWO)), options=options)
    return seen, sorted({lib.get() for lib in blas.libraries()})


@needs_stack
def test_every_factorization_of_a_sweep_runs_at_the_ranks_threads(tmp_path):
    (seen, after), *_ = on_ranks(_blas_in_a_factorization, tmp_path, ranks=3, threads=1)
    assert seen == [[2], [2]]
    assert after == [2]


# --- what else passes the refusal --------------------------------------------------


def _pmc_line(directory):
    model = cases.layered_line(directory)
    return model.replace(
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.PMC("side")]
    )


def _sheet_line(directory, name):
    mesh = cases.sheet_mesh(directory)
    fill = gfem.SheetField(fill=np.full(len(mesh.elements("sheet")), 0.5))
    return cases.sheet_line(mesh, cases.SHEETS[name](fill))


def _anisotropic_line(directory):
    model = cases.layered_line(directory, absorbing_order=1)
    return model.replace(
        materials={
            "sub": gfem.Material(eps_r=(2.0, 3.0, 4.0)),
            "cover": gfem.Material(),
        }
    )


def _line_around_a_hole(directory):
    """A line with a PEC block left out of the domain, between its ports."""

    def inside(c):
        return 1.5 < c[0] < 2.5 and 0.5 < c[1] < 1.5 and c[2] < 0.5

    def volume(c):
        return "hole" if inside(c) else "dielectric"

    def surface(c):
        for x, name in ((0.0, "port1"), (4.0, "port2")):
            if abs(c[0] - x) < 1e-9:
                return name
        if abs(c[2]) < 1e-9:
            return "bottom"
        if abs(c[2] - 1) < 1e-9:
            return "top"
        if abs(c[1]) < 1e-9 or abs(c[1] - 2) < 1e-9:
            return "side"
        return "wall"

    edges = (np.linspace(0, 4, 9), np.linspace(0, 2, 5), np.linspace(0, 1, 3))
    raw = sm.structured(edges, volume, surface)
    path = sm.write_msh22(Path(directory) / "hole.msh", raw)
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material()},
        excluded=("hole",),
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.PEC("wall"),
            gfem.Absorbing("side", order=1),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


VARIANTS = {
    "pmc": _pmc_line,
    "conductor sheet": lambda directory: _sheet_line(directory, "conductor"),
    "impedance sheet": lambda directory: _sheet_line(directory, "impedance"),
    "anisotropic": _anisotropic_line,
    "excluded": _line_around_a_hole,
}


def _variant_s(directory, name, solver):
    model = VARIANTS[name](directory)
    options = cases.tight(solver, threads=1)
    problem = gfem.Driven(FREQUENCY)
    return np.asarray(gfem.solve(model, problem, options=options).s.values)


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize(("solver", "gate"), [("direct", S_GATE), ("bddc_gmres", 1e-9)])
def test_each_model_feature_matches_one_process(tmp_path, name, solver, gate):
    if solver == "direct" and not _petsc.has_mpi():
        pytest.skip("needs the MPI stack")
    s_one = _variant_s(tmp_path, name, "direct")
    s, *_ = on_ranks(_variant_s, tmp_path, name, solver, ranks=4, limit=300)
    error = np.abs(s - s_one).max() / np.abs(s_one).max()
    print(f"{name} on {solver}: S error={error:.3g}")  # noqa: T201
    assert error < gate


# --- without the MPI stack ----------------------------------------------------------


def _refusal(directory, solver):
    model = cases.layered_line(directory)
    options = gfem.SolverOptions(linear_solver=solver)
    try:
        gfem.solve(model, gfem.Driven(FREQUENCY), options=options)
    except gfem.SolverUnavailableError as err:
        return str(err), _world().allreduce(1), "petsc4py.PETSc" in sys.modules
    raise AssertionError("a solve without the MPI stack ran")


@pytest.mark.parametrize("solver", ["direct", "petsc"])
def test_without_the_stack_a_distributed_solver_refuses_before_init(tmp_path, solver):
    if _petsc.has_mpi():
        pytest.skip("the MPI stack is installed")
    (text, ranks, initialised), *_ = on_ranks(_refusal, tmp_path, solver, ranks=3)
    assert "petsc_wheelhouse.sh --mpi" in text
    assert ranks == 3
    assert not initialised


def _copy_on_this_stack(directory, solver):
    model = cases.layered_line(directory)
    options = gfem.SolverOptions(linear_solver=solver)
    problem = gfem.Driven(FREQUENCY)
    from mpi4py import MPI

    result = gfem.solve(model, problem, options=options, comm=MPI.COMM_SELF)
    return np.asarray(result.s.values)


def test_rank_copies_run_on_either_stack(tmp_path):
    pytest.importorskip("petsc4py")
    copies = on_ranks(
        _copy_on_this_stack, tmp_path, "petsc", ranks=2, replicated=False, limit=300
    )
    alone = _copy_on_this_stack_one_process(tmp_path)
    for s in copies:
        assert np.abs(s - alone).max() < S_GATE * np.abs(alone).max()


def _copy_on_this_stack_one_process(directory):
    model = cases.layered_line(directory)
    options = gfem.SolverOptions(linear_solver="direct")
    return np.asarray(
        gfem.solve(model, gfem.Driven(FREQUENCY), options=options).s.values
    )


#: mpi4py at the top: a spawned worker imports the script again.
WORKERS = """
import sys

from mpi4py import MPI

import gfem
import gradient_cases as cases


def main():
    model = cases.layered_line(sys.argv[1])
    options = gfem.SolverOptions(linear_solver=sys.argv[2], frequency_workers=2)
    problem = gfem.Driven([4e9, 6e9])
    result = gfem.solve(model, problem, options=options, comm=MPI.COMM_SELF)
    print("workers solved", result.s.values.shape, flush=True)


if __name__ == "__main__":
    main()
"""

#: Where a launched script finds the test helpers.
SCRIPT_PATH = (Path(__file__).parents[1], Path(__file__).parents[2] / "tools")


@pytest.mark.parametrize("solver", ["direct", "petsc"])
def test_workers_under_a_script_importing_mpi4py_at_its_top(tmp_path, solver):
    if solver == "petsc":
        pytest.importorskip("petsc4py")
    script = tmp_path / "script.py"
    script.write_text(WORKERS)
    code, log = on_ranks_argv(
        [sys.executable, str(script), str(tmp_path), solver],
        ranks=2,
        limit=300,
        env={"PYTHONPATH": os.pathsep.join(map(str, SCRIPT_PATH))},
    )
    assert code == 0, log
    assert log.count("workers solved (2, 2, 2)") == 2, log
