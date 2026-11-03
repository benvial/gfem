"""Reverse-mode gradients through the real `gfem.solve` across ranks.

Every row compares ranks with the same function in pytest's one process: gradient,
value and each finite-difference sample bit for bit across ranks (`on_ranks`, since a
missing rank sum raises nothing and exits 0), the gradient entry by entry to
its solver's gate in `GATES` against one process on that solver, and
`tests/finite_differences.py` unchanged. BDDC runs on either stack, its coarse problem
on MUMPS where the MPI stack is; `direct` is PETSc MUMPS, on that stack only.
"""

import importlib
import threading
from pathlib import Path

import gradient_cases as cases
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from anisotropy import TILTED
from finite_differences import assert_gradient_matches, unit_directions
from ranks import RanksDiffer, on_ranks
from solver_misses import adjoint_cannot_converge_at, cannot_converge_at

import gfem
from gfem import _linalg, _mpi, _petsc
from gfem._engine import system_operator
from gfem._engine.cache import ENTRIES
from gfem._engine.driven import DrivenRule
from gfem._mpi import AgreedError
from gfem.shape import boundary_morph

#: Relative to the largest entry of the one-process gradient: ADR 0070's gates.
GATES = {"bddc_gmres": 1e-9, "direct": 1e-11}
SOLVERS = (
    "bddc_gmres",
    pytest.param(
        "direct",
        marks=pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack"),
    ),
)
RANKS = (3, 4, 8)
FREQUENCIES = (18e9, 27e9)
WAVE_FREQUENCIES = (12e9, 24e9)
SHEET_FREQUENCIES = (6e9, 9e9)
HYBRID_FREQUENCIES = (55e9, 65e9)
SHAPE_FREQUENCY = 20e9
STEPS = {
    "eps_r": 1e-2,
    "conductivity": 5e-2,
    "mu_r": 1e-2,
    "tan_delta": 2e-3,
    "nodes": 2e-6,
    "fill": 5e-3,
}
#: What a backward's error is, seen from a script (ADR 0065).
BACKWARD_ERRORS = (jax.errors.JaxRuntimeError, ValueError)


#: The solver of the design being built and solved, in this process.
_SOLVER = {"name": "bddc_gmres"}


def _options(**kwargs):
    return cases.tight(_SOLVER["name"], threads=1, **kwargs)


def _on(name, solver, *args, **kwargs):
    """The module's function ``name`` on ``solver``: arguments travel by pickle."""
    _SOLVER["name"] = solver
    try:
        return globals()[name](*args, **kwargs)
    finally:
        _SOLVER["name"] = "bddc_gmres"


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _s21(result):
    return jnp.sum(jnp.abs(result.s["p2", "p1"]) ** 2)


# --- the rows: objective, design, directions, step -----------------------------------


def _line_row(name, seed, directory, directions=2, values=None, **problem):
    line = cases.layered_line(Path(directory))
    design, rng = cases.line_design(line, seed=seed)
    design = design if values is None else values(rng, len(line.mesh.tetrahedra))
    driven = gfem.Driven(frequency=FREQUENCIES, **problem)

    def objective(p):
        model = cases.with_fields(line, {**design, name: p})
        result = gfem.solve(model, driven, options=_options())
        return _s21(result) if problem else cases.s_functional(result.s.values)

    p = design[name]
    return objective, p, unit_directions(rng, p.shape, directions), STEPS[name]


def _eps(directory):
    objective, p, directions, step = _line_row("eps_r", 2, directory, directions=1)
    return objective, p, [directions[0], 1j * directions[0]], step


def _mu_r(rng, n):
    return {
        "eps_r": jnp.asarray(rng.uniform(2.0, 6.0, n)),
        "mu_r": jnp.asarray(rng.uniform(1.0, 2.0, n)),
    }


def _tan_delta(rng, n):
    return {
        "eps_r": jnp.asarray(rng.uniform(2.0, 6.0, n)),
        "tan_delta": jnp.asarray(rng.uniform(0.01, 0.05, n)),
    }


def _big(directory):
    directory = Path(directory)
    mesh = cases._mesh(
        directory,
        "big.msh",
        sm.strip_line(4.0, 2.0, 1.0, (16, 8, 4), layers=cases.LAYERS),
    )
    line = cases.layered_line(directory).replace(mesh=mesh)
    values, _ = cases.line_design(line, seed=2)
    driven = gfem.Driven(frequency=FREQUENCIES)

    def objective(p):
        model = cases.with_fields(line, {**values, "eps_r": p})
        return cases.s_functional(
            gfem.solve(model, driven, options=_options()).s.values
        )

    return objective, values["eps_r"], [], STEPS["eps_r"]


def _axes_row(name, seed, directory):
    line = cases.layered_line(Path(directory), absorbing_order=1)
    values, rng = cases.line_axes_design(line, seed=seed)
    driven = gfem.Driven(frequency=FREQUENCIES)

    def objective(p):
        model = cases.with_fields(line, {**values, name: p}, TILTED)
        return cases.s_functional(
            gfem.solve(model, driven, options=_options()).s.values
        )

    p = values[name]
    directions = unit_directions(rng, p.shape, 2)
    if jnp.iscomplexobj(p):
        directions = [directions[0], 1j * directions[0]]
    return objective, p, directions, STEPS[name]


def _fill_row(kind, seed, directory):
    mesh = cases.sheet_mesh(Path(directory))
    fill, rng = cases.sheet_fill(mesh, seed)
    driven = gfem.Driven(frequency=SHEET_FREQUENCIES)

    def objective(f):
        model = cases.sheet_line(mesh, cases.SHEETS[kind](gfem.SheetField(fill=f)))
        return cases.s_functional(
            gfem.solve(model, driven, options=_options()).s.values
        )

    return objective, fill, unit_directions(rng, fill.shape, 2), STEPS["fill"]


def _shape_objective(line, mesh_of):
    driven = gfem.Driven(frequency=SHAPE_FREQUENCY)

    def objective(x):
        model = line.replace(mesh=mesh_of(x))
        s = gfem.solve(model, driven, options=_options()).s.values
        return jnp.sum(cases.WEIGHTS * jnp.abs(s[0]) ** 2)

    return objective


def _nodes_raw(directory):
    line = cases.shape_line(Path(directory))
    mesh = line.mesh
    x0 = jnp.asarray(mesh.nodes)
    structure = jax.tree_util.tree_structure(mesh)
    objective = _shape_objective(
        line, lambda x: jax.tree_util.tree_unflatten(structure, [x])
    )
    tags = [mesh.region(s).tag for s in ("port1", "port2")]
    held = np.unique(mesh.triangles[np.isin(mesh.triangle_tags, tags)])
    directions = []
    for d in unit_directions(np.random.default_rng(5), x0.shape, 2):
        d[held] = 0.0
        directions.append(d / np.linalg.norm(d))
    return objective, x0, directions, STEPS["nodes"]


def _nodes_morph(directory):
    line = cases.shape_line(Path(directory))
    morph = boundary_morph(line, "sub__cover", grows="sub")
    t = jnp.zeros(morph.n_params)
    directions = unit_directions(np.random.default_rng(3), t.shape, 2)
    return _shape_objective(line, morph), t, directions, STEPS["nodes"]


def _wave_row(port, directory, name="eps_r", seed=4, **problem):
    mesh = cases.wave_mesh(Path(directory))
    model = cases.wave_line(mesh, port)
    values, rng = cases.core_design(mesh, seed=seed)
    driven = gfem.Driven(frequency=WAVE_FREQUENCIES, **problem)

    def objective(p):
        design = model.with_material("core", gfem.MaterialField(**{**values, name: p}))
        design = design.replace(mesh=jax.lax.stop_gradient(design.mesh))
        result = gfem.solve(design, driven, options=_options())
        return _s21(result) if problem else cases.s_functional(result.s.values)

    p = values[name]
    return objective, p, unit_directions(rng, p.shape, 2), STEPS[name]


def _wave_nodes(directory):
    model = cases.wave_line(cases.wave_mesh(Path(directory)), "lossy")
    morph = boundary_morph(model, "top")
    driven = gfem.Driven(frequency=WAVE_FREQUENCIES)

    def objective(t):
        design = model.replace(mesh=morph(t))
        return cases.s_functional(
            gfem.solve(design, driven, options=_options()).s.values
        )

    t = jnp.zeros(morph.n_params)
    directions = unit_directions(np.random.default_rng(6), t.shape, 2)
    return objective, t, directions, STEPS["nodes"]


def _shared(directory, cells=(4, 2, 2)):
    mesh = cases.hybrid_mesh(Path(directory), cells=cells)
    model = cases.hybrid_line(mesh)
    values, rng = cases.core_design(mesh, seed=4)
    driven = gfem.Driven(frequency=HYBRID_FREQUENCIES, excite="p1_m1")
    rows = jnp.asarray([0, 1])

    def objective(p):
        design = model.with_material(
            "core", gfem.MaterialField(**{**values, "eps_r": p})
        )
        design = design.replace(mesh=jax.lax.stop_gradient(design.mesh))
        result = gfem.solve(design, driven, options=_options())
        return cases.shared_surface_functional(result.s, rows)

    p = values["eps_r"]
    return objective, p, unit_directions(rng, p.shape, 2), STEPS["eps_r"]


#: One row per line of ticket 09's table, then `mu_r` and `tan_delta`.
ROWS = {
    "eps": _eps,
    "sigma": lambda d: _line_row("conductivity", 4, d),
    "s21_adjoint": lambda d: _line_row("eps_r", 3, d, directions=1, excite="p1"),
    "big": _big,
    "axes_eps": lambda d: _axes_row("eps_r", 5, d),
    "axes_sigma": lambda d: _axes_row("conductivity", 6, d),
    "fill_conductor": lambda d: _fill_row("conductor", 7, d),
    "fill_impedance": lambda d: _fill_row("impedance", 8, d),
    "nodes_raw": _nodes_raw,
    "nodes_morph": _nodes_morph,
    "wave_lossless": lambda d: _wave_row("lossless", d),
    "wave_lossy": lambda d: _wave_row("lossy", d),
    "wave_mixed": lambda d: _wave_row("mixed", d),
    "wave_sigma": lambda d: _wave_row("lossy", d, "conductivity", seed=5),
    "wave_s21": lambda d: _wave_row("lossless", d, seed=3, excite="p1"),
    "wave_nodes": _wave_nodes,
    "shared": _shared,
    # fine enough that port faces lie across ranks: 3 ranks share no port DOF above
    "shared_fine": lambda d: _shared(d, cells=(8, 8, 4)),
    "mu_r": lambda d: _line_row("mu_r", 9, d, values=_mu_r),
    "tan_delta": lambda d: _line_row("tan_delta", 10, d, values=_tan_delta),
}


def _gradient(row, directory, *, differences=True):
    """The value, the gradient and every finite-difference sample of a row."""
    objective, p, directions, step = ROWS[row](directory)
    samples = []

    def sampled(x):
        value = objective(x)
        samples.append(float(jnp.real(value)))
        return value

    value, grad = jax.value_and_grad(objective)(p)
    grad = np.asarray(grad)
    if differences and directions:
        assert_gradient_matches(sampled, p, directions, step=step, grad=grad)
    return np.asarray(value), grad, np.asarray(samples)


def _relative(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(b).max())


def _assert_below_the_gate(a, b, what, solver="bddc_gmres"):
    error = _relative(a, b)
    print(f"{what} {solver} error={error:.3g}")  # noqa: T201
    assert error < GATES[solver], f"{what} off by {error:.3g} relative"


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("ranks", RANKS)
@pytest.mark.parametrize("row", sorted(ROWS))
def test_a_gradient_matches_one_process_and_differences(tmp_path, row, ranks, solver):
    _, grad_one, _ = _on("_gradient", solver, row, tmp_path, differences=False)
    (_, grad, samples), *_ = on_ranks(
        _on, "_gradient", solver, row, tmp_path, ranks=ranks, limit=900
    )
    assert np.isfinite(grad).all()
    assert len(samples) or row == "big"
    _assert_below_the_gate(grad, grad_one, f"{row} ranks={ranks}", solver)


def _a_rank_without_the_sum(directory):
    """The row `eps` with the sum gone: what a lost rank sum looks like."""
    module = importlib.import_module("gfem._adjoint.solve_vjp")

    class NoSum:
        reduce_bcast = staticmethod(lambda _comm, values: values)

        def __getattr__(self, name):
            return getattr(_mpi, name)

    module._mpi = NoSum()
    return _gradient("eps", directory, differences=False)


def test_a_lost_rank_sum_is_seen_across_ranks(tmp_path):
    with pytest.raises(RanksDiffer):
        on_ranks(_a_rank_without_the_sum, tmp_path, ranks=3, limit=300)


# --- JAX's transformations ------------------------------------------------------------


def _two_outputs(directory):
    line = cases.layered_line(Path(directory))
    values, _ = cases.line_design(line, seed=2)
    driven = gfem.Driven(frequency=FREQUENCIES)

    def outputs(p):
        model = cases.with_fields(line, {**values, "eps_r": p})
        s = gfem.solve(model, driven, options=_options()).s.values
        return jnp.stack([jnp.abs(s[0, 1, 0]) ** 2, jnp.abs(s[1, 0, 0]) ** 2])

    return outputs, values["eps_r"]


def _transforms(directory):
    objective, p, _, _ = ROWS["eps"](directory)
    grad = jax.grad(objective)
    batch = jnp.stack([p, p * 1.01])
    outputs, q = _two_outputs(directory)
    _, pull = jax.vjp(outputs, q)
    return {
        "grad": np.asarray(grad(p)),
        "second": np.asarray(grad(batch[1])),
        "jit": np.asarray(jax.block_until_ready(jax.jit(grad)(p))),
        "vmap": np.asarray(jax.block_until_ready(jax.vmap(grad)(batch))),
        "jacrev": np.asarray(jax.block_until_ready(jax.jacrev(outputs)(q))),
        "pull": [np.asarray(pull(jnp.eye(2)[i])[0]) for i in range(2)],
    }


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("ranks", RANKS)
def test_jit_vmap_jacrev_and_a_pullback_twice_match_grad(tmp_path, ranks, solver):
    got, *_ = on_ranks(_on, "_transforms", solver, tmp_path, ranks=ranks, limit=900)
    one = _on("_transforms", solver, tmp_path)
    for name in ("grad", "second", "jit", "vmap", "jacrev"):
        _assert_below_the_gate(got[name], one[name], f"{name} ranks={ranks}", solver)
    _assert_below_the_gate(got["jit"], got["grad"], "jit against grad", solver)
    _assert_below_the_gate(got["vmap"][0], got["grad"], "vmap's first", solver)
    _assert_below_the_gate(got["vmap"][1], got["second"], "vmap's second", solver)
    for i in range(2):
        _assert_below_the_gate(
            got["pull"][i], got["jacrev"][i], f"pullback {i}", solver
        )


def _counted_forwards(function, *args):
    """``function(*args)`` and how many forwards ran under it."""
    seen = []
    forward = DrivenRule.forward

    def spy(self, model, *, keep):
        seen.append(keep)
        return forward(self, model, keep=keep)

    DrivenRule.forward = spy
    try:
        return function(*args), len(seen)
    finally:
        DrivenRule.forward = forward


def _retained_nothing(directory):
    line = cases.layered_line(Path(directory))
    values, _ = cases.line_design(line, seed=2)
    driven = gfem.Driven(frequency=FREQUENCIES)

    def objective(p):
        model = cases.with_fields(line, {**values, "eps_r": p})
        options = _options(retain_memory=0)
        return cases.s_functional(gfem.solve(model, driven, options=options).s.values)

    grad, forwards = _counted_forwards(jax.grad(objective), values["eps_r"])
    return np.asarray(grad), forwards, len(ENTRIES)


@pytest.mark.parametrize("solver", SOLVERS)
def test_retain_memory_zero_runs_the_forward_again_on_every_rank(tmp_path, solver):
    (_, one, _), *_ = on_ranks(
        _on,
        "_gradient",
        solver,
        "eps",
        tmp_path,
        ranks=4,
        limit=300,
        differences=False,
    )
    (grad, forwards, kept), *_ = on_ranks(
        _on, "_retained_nothing", solver, tmp_path, ranks=4, limit=300
    )
    assert forwards == 2
    assert kept == 0
    _assert_below_the_gate(grad, one, "retain_memory=0", solver)


def _an_entry_lost_on_one_rank(directory, lost):
    objective, p, _, _ = ROWS["eps"](directory)
    reference = np.asarray(jax.grad(objective)(p))
    _, pull = jax.vjp(objective, p)
    # the entry reads as gone on one rank alone; freeing is every rank's, together
    get = ENTRIES.get
    if _world().rank == lost:
        ENTRIES.get = lambda _key: None
    try:
        grad, forwards = _counted_forwards(pull, jnp.asarray(1.0))
    finally:
        ENTRIES.get = get
    return reference, np.asarray(grad[0]), forwards, len(ENTRIES)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("lost", [0, 1])
def test_an_entry_lost_on_one_rank_runs_every_forward_again(tmp_path, lost, solver):
    (reference, grad, forwards, entries), *_ = on_ranks(
        _on, "_an_entry_lost_on_one_rank", solver, tmp_path, lost, ranks=4, limit=300
    )
    assert forwards == 1
    assert entries == 1  # the first grad's: the pullback's was freed by all
    _assert_below_the_gate(grad, reference, "after the loss", solver)


# --- one base handle per solve --------------------------------------------------------


def _counted_prepares(function, *args):
    """``function(*args)``, the handles prepared and the refactorizations under it."""
    prepared = []
    refactorized = []
    prepare = _linalg.prepare
    refactorize_or_prepare = system_operator.refactorize_or_prepare

    def spy(*pargs, **kwargs):
        prepared.append(1)
        return prepare(*pargs, **kwargs)

    def spy_refactorize(handle, *pargs, **kwargs):
        if handle is not None:
            refactorized.append(1)
        return refactorize_or_prepare(handle, *pargs, **kwargs)

    _linalg.prepare = spy
    system_operator.refactorize_or_prepare = spy_refactorize
    try:
        return function(*args), len(prepared), len(refactorized)
    finally:
        _linalg.prepare = prepare
        system_operator.refactorize_or_prepare = refactorize_or_prepare


def _prepares_of(row, directory):
    objective, p, _, _ = ROWS[row](directory)
    grad, prepares, refactorized = _counted_prepares(jax.grad(objective), p)
    return np.asarray(grad), prepares, refactorized


def _prepares_of_one_frequency(directory):
    """One frequency, lossy wave ports, an adjoint through Woodbury."""
    mesh = cases.wave_mesh(Path(directory))
    model = cases.wave_line(mesh, "lossy")
    values, _ = cases.core_design(mesh, seed=4)
    driven = gfem.Driven(frequency=WAVE_FREQUENCIES[0], excite="p1")

    def objective(p):
        design = model.with_material(
            "core", gfem.MaterialField(**{**values, "eps_r": p})
        )
        design = design.replace(mesh=jax.lax.stop_gradient(design.mesh))
        return _s21(gfem.solve(design, driven, options=_options()))

    grad, prepares, refactorized = _counted_prepares(
        jax.grad(objective), values["eps_r"]
    )
    return np.asarray(grad), prepares, refactorized


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_two_frequency_gradient_prepares_one_base(tmp_path, solver):
    grad_one, serial, _ = _on("_prepares_of", solver, "s21_adjoint", tmp_path)
    (grad, prepares, refactorized), *_ = on_ranks(
        _on, "_prepares_of", solver, "s21_adjoint", tmp_path, ranks=4, limit=300
    )
    print(  # noqa: T201
        f"prepares: {prepares} under ranks, {serial} on one process; "
        f"refactorizations {refactorized}"
    )
    assert serial == 3  # the sweep's, and one per frequency for the adjoint
    assert prepares == 1
    # the sweep's second frequency, then the adjoint at each of the two
    assert refactorized == 3
    _assert_below_the_gate(grad, grad_one, "one base", solver)


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_one_frequency_woodbury_gradient_factorizes_once(tmp_path, solver):
    grad_one, serial, _ = _on("_prepares_of_one_frequency", solver, tmp_path)
    (grad, prepares, refactorized), *_ = on_ranks(
        _on, "_prepares_of_one_frequency", solver, tmp_path, ranks=4, limit=300
    )
    print(f"prepares: {prepares} under ranks, {serial} on one process")  # noqa: T201
    assert serial == 2
    assert prepares == 1
    assert refactorized == 0
    _assert_below_the_gate(grad, grad_one, "one base, Woodbury", solver)


def _bases_held() -> int:
    return sum(
        entry.handle.system._adjoint_sweep._base is not None
        for entry in ENTRIES._entries.values()
    )


def _a_design_loop(directory):
    """Two designs alive: at most one factorization, and either pullback still runs."""
    objective, p, _, _ = ROWS["s21_adjoint"](directory)
    held = []
    _, first = jax.vjp(objective, p)
    held.append(_bases_held())
    _, second = jax.vjp(objective, p * 1.01)
    held.append(_bases_held())
    entries = len(ENTRIES)
    (grad_first,), prepares, _ = _counted_prepares(first, jnp.asarray(1.0))
    held.append(_bases_held())
    (grad_second,), more, _ = _counted_prepares(second, jnp.asarray(1.0))
    held.append(_bases_held())
    return {
        "held": held,
        "entries": entries,
        "prepares": (prepares, more),
        "first": np.asarray(grad_first),
        "second": np.asarray(grad_second),
        "fresh": np.asarray(jax.grad(objective)(p)),
    }


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_design_loop_holds_one_factorization(tmp_path, solver):
    got, *_ = on_ranks(_on, "_a_design_loop", solver, tmp_path, ranks=4, limit=300)
    assert got["held"] == [1, 1, 1, 1]
    assert got["entries"] == 2
    # each pullback finds its base gone, and prepares it again
    assert got["prepares"] == (1, 1)
    _assert_below_the_gate(
        got["first"], got["fresh"], "after the base was freed", solver
    )
    assert np.abs(got["second"] - got["first"]).max() > 0


# --- failures -----------------------------------------------------------------------


def _forward_miss(directory):
    """A sweep that misses at its second frequency, eagerly and under JAX."""
    line = cases.layered_line(Path(directory))
    driven = gfem.Driven(frequency=FREQUENCIES)
    whole = gfem.solve(line, driven, options=_options()).s.values
    with pytest.MonkeyPatch.context() as patch:
        cannot_converge_at(patch, (FREQUENCIES[1],))
        try:
            gfem.solve(line, driven, options=_options())
        except gfem.LinearSolverConvergenceError as error:
            partial = error.result.s.values
            miss = (error.frequency, list(error.unsolved))
        values, _ = cases.line_design(line, seed=2)

        def objective(p):
            model = cases.with_fields(line, {**values, "eps_r": p})
            return cases.s_functional(
                gfem.solve(model, driven, options=_options()).s.values
            )

        try:
            jax.grad(objective)(values["eps_r"])
        except BACKWARD_ERRORS as error:
            jaxed = "did not converge" in str(error)
    return {
        "whole": np.asarray(whole),
        "partial": np.asarray(partial),
        "miss": miss,
        "jaxed": jaxed,
        "after": np.asarray(jax.grad(objective)(values["eps_r"])),
    }


def test_a_forward_miss_gives_a_partial_result_on_every_rank(tmp_path):
    got, *_ = on_ranks(_forward_miss, tmp_path, ranks=4, limit=300)
    assert got["miss"] == (FREQUENCIES[1], [FREQUENCIES[1]])
    assert got["jaxed"]
    assert got["partial"].shape[0] == 1
    assert _relative(got["partial"][0], got["whole"][0]) < GATES["bddc_gmres"]
    assert np.isfinite(got["after"]).all()


def _adjoint_miss(directory):
    """An adjoint that misses, first and after the same jitted call has succeeded."""
    objective, p, _, _ = ROWS["s21_adjoint"](directory)
    reference = np.asarray(jax.grad(objective)(p))
    compiled = jax.jit(jax.grad(objective))
    compiled(p).block_until_ready()
    seen = {}
    with pytest.MonkeyPatch.context() as patch:
        adjoint_cannot_converge_at(patch, (FREQUENCIES[1],))
        for name, call in (("grad", jax.grad(objective)), ("jit", compiled)):
            try:
                call(p)
                seen[name] = None
            except BACKWARD_ERRORS as error:
                named = "the adjoint" in str(error) and "did not converge" in str(error)
                seen[name] = (type(error).__name__, named)
    _world().Barrier()
    return seen, reference, np.asarray(jax.grad(objective)(p))


def test_an_adjoint_miss_is_caught_on_every_rank_and_the_job_goes_on(tmp_path):
    (seen, reference, after), *_ = on_ranks(_adjoint_miss, tmp_path, ranks=4, limit=300)
    assert seen["grad"] == ("JaxRuntimeError", True)
    assert seen["jit"] == ("ValueError", True)
    _assert_below_the_gate(after, reference, "after the miss")


def _a_design_differing_on_one_rank(directory):
    objective, p, _, _ = ROWS["eps"](directory)
    other = np.array(p)
    if _world().rank == 1:
        other[0] += 1e-3
    compiled = jax.jit(jax.grad(objective))
    try:
        compiled(jnp.asarray(other))
        message = ""
    except (*BACKWARD_ERRORS, AgreedError) as error:
        message = str(error)
    _world().Barrier()
    return "hold different values" in message, np.isfinite(compiled(p)).all()


def test_a_host_call_on_different_designs_raises_on_every_rank(tmp_path):
    (named, finite), *_ = on_ranks(
        _a_design_differing_on_one_rank, tmp_path, ranks=3, limit=300
    )
    assert named
    assert finite


# --- threads ----------------------------------------------------------------------


def _on_the_default_threads(directory):
    line = cases.layered_line(Path(directory))
    values, _ = cases.line_design(line, seed=2)
    driven = gfem.Driven(frequency=FREQUENCIES)
    options = gfem.SolverOptions(order=2, linear_solver="bddc_gmres", tolerance=1e-12)

    def objective(p):
        model = cases.with_fields(line, {**values, "eps_r": p})
        return cases.s_functional(gfem.solve(model, driven, options=options).s.values)

    return np.asarray(jax.grad(objective)(values["eps_r"]))


def test_threads_none_follows_the_rank_rule_in_the_backward(tmp_path):
    one = _on_the_default_threads(tmp_path)
    grad, *_ = on_ranks(
        _on_the_default_threads, tmp_path, ranks=3, threads=2, limit=300
    )
    _assert_below_the_gate(grad, one, "threads=None")


# --- off the main thread --------------------------------------------------------------


def _off_the_main_thread(directory):
    objective, p, _, _ = ROWS["eps"](directory)
    out = {}

    def run():
        out["value"], out["grad"] = jax.value_and_grad(objective)(p)
        out["name"] = threading.current_thread().name

    thread = threading.Thread(target=run, name="solver")
    thread.start()
    thread.join()
    return out["name"], np.asarray(out["grad"]), np.asarray(jax.grad(objective)(p))


def test_a_whole_solve_runs_off_the_main_thread(tmp_path):
    (name, grad, main), *_ = on_ranks(
        _off_the_main_thread, tmp_path, ranks=4, limit=300
    )
    assert name == "solver"
    _assert_below_the_gate(grad, main, "off the main thread")
