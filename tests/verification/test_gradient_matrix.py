"""The gradient verification matrix: every v0.1 gradient against differences.

One cell per problem type x input kind x linear solver x frequency sweep:

- problems: electrostatic capacitance (and a potential functional), driven
  S on lumped ports and on wave ports (lossless and lossy), eigenfrequency
  (frequency and loss of a simple mode);
- inputs: per-element permittivity ``eps_r`` and conductivity, the fill of a
  `gfem.SheetField`, and the mesh node coordinates (through
  `gfem.shape.boundary_morph`);
- solvers: ``direct`` and ``bddc_gmres``; sweeps: uniform and adaptive (with
  frozen samples; the objective reads S at the samples, where the reduced
  model reproduces the full solve of the perturbed design too, so the
  differences of the adaptive pipeline check the frozen-basis gradient).

The PETSc cells (`PETSC_CELLS`: ``linear_solver="petsc"``, and SLEPc seeds
for the eigenmode cell) run the same checks where the PETSc stack is
installed, and are skipped elsewhere (the ``petsc`` / ``slepc`` markers).
They stay out of `CELLS`, from which the docs build derives the Capabilities
page's claims: the verification job that runs on every PR has no PETSc.

Each cell checks the gradient along 3 random unit directions against
fourth-order central differences (tests/finite_differences.py, gate 1e-6)
at tight tolerance (iterative solves to 1e-12, eigenpairs to 1e-12), and
that the Taylor remainder falls at second order at production tolerance
(`gfem.SolverOptions` defaults: iterative solves to 1e-6, eigenpairs to
1e-10), with gradient and differences both carrying that solver error.

Then the consistency checks (JAX transformations and evicted cache entries
per input kind, reciprocity against an explicit adjoint, adaptive against
uniform at the samples) and the degenerate eigenmode clusters. Pairs this
version does not differentiate are rows of the raise table
(tests/verification/test_gradient_matrix_raise_table.py), not cells: the
electrostatic conductivity (an exact zero) and sheet fill (not an
electrostatic object), eigenmode field quantities, the port mode problem.

Run the gate with ``pytest -m verification -k gradient_matrix``.
"""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import (
    assert_gradient_matches,
    assert_taylor_second_order,
    unit_directions,
)
from gradient_cases import (
    CAVITY_TARGET,
    EPS0,
    MIRRORED_TARGET,
    OMEGA_WEIGHT,
    RESONATOR_TARGET,
    SHEETS,
    STACK_PROBLEM,
    capacitor,
    cavity_fill,
    cavity_model,
    core_design,
    layered_line,
    layered_resonator,
    line_design,
    mirrored_cavity,
    on_the_sheet,
    ramp,
    resonator_mesh,
    s_functional,
    shape_line,
    sheet_cavity,
    sheet_fill,
    sheet_line,
    sheet_mesh,
    stack,
    stack_fields,
    terminated,
    tight,
    wave_line,
    wave_mesh,
    with_fields,
)

import gfem
from gfem.shape import boundary_morph

SOLVERS = ("direct", "bddc_gmres")


@dataclasses.dataclass(frozen=True)
class Cell:
    """One cell: problem x input kind x solver x sweep (x variant)."""

    problem: str
    input: str
    solver: str
    sweep: str = ""
    variant: str = ""

    def __str__(self) -> str:
        parts = (self.problem, self.input, self.variant, self.solver, self.sweep)
        return "-".join(p for p in parts if p)


def _cells() -> list[Cell]:
    cells = []
    for solver in SOLVERS:
        cells += [
            Cell("electrostatic", "eps_r", solver),
            Cell("electrostatic", "nodes", solver, variant="capacitance"),
            Cell("electrostatic", "nodes", solver, variant="potential"),
        ]
        for sweep in ("uniform", "adaptive"):
            cells += [
                Cell("driven", "eps_r", solver, sweep),
                Cell("driven", "conductivity", solver, sweep),
                Cell("driven", "nodes", solver, sweep),
            ]
            sheets = (
                ("conductor", "impedance") if sweep == "uniform" else ("impedance",)
            )
            cells += [Cell("driven", "fill", solver, sweep, s) for s in sheets]
            # Wave ports: every port x input on the direct solver; on BDDC +
            # GMRES the lossy ports (the adjoint solve through the port
            # operator's Woodbury update, the adjoint loads' extra columns at
            # the samples) in the permittivity, whose conductivity cells take
            # the same code path, and in the nodes on the uniform sweep (the
            # adaptive backward is the solver's only through those solves).
            bddc = {"eps_r"} | ({"nodes"} if sweep == "uniform" else set())
            cells += [
                Cell("driven_wave", name, solver, sweep, port)
                for port in ("lossless", "lossy")
                for name in ("eps_r", "conductivity", "nodes")
                if solver == "direct" or (port == "lossy" and name in bddc)
            ]
        cells += [
            Cell("eigenmode", name, solver)
            for name in ("eps_r", "conductivity", "fill", "nodes")
        ]
    return cells


CELLS = _cells()
#: The PETSc stack's cells, one per path its solves take: electrostatic,
#: driven on both sweeps, the adjoint through a wave port operator's
#: Woodbury update, and the eigenmode seed on SLEPc.
PETSC_CELLS = [
    Cell("electrostatic", "eps_r", "petsc"),
    Cell("driven", "eps_r", "petsc", "uniform"),
    Cell("driven", "eps_r", "petsc", "adaptive"),
    Cell("driven_wave", "eps_r", "petsc", "uniform", "lossy"),
    Cell("eigenmode", "eps_r", "petsc"),
]
PETSC_PARAMS = [
    pytest.param(
        cell,
        marks=pytest.mark.slepc if cell.problem == "eigenmode" else pytest.mark.petsc,
    )
    for cell in PETSC_CELLS
]


@dataclasses.dataclass
class Case:
    """A cell's objective, its input and how to perturb it.

    ``step``: fourth-order difference step along a unit direction (see
    tests/finite_differences.py); ``taylor``: the largest of the Taylor
    test's three steps along one unit direction, each ``taylor_ratio``
    times the next, from ``taylor_at`` (``p`` when None). At production
    tolerance the objective carries the solver's error (about 1e-7 relative
    for GMRES at 1e-6): the remainders must stay far above it.
    """

    objective: Callable
    p: jax.Array
    rng: np.random.Generator
    step: float
    taylor: float
    reference: Callable | None = None
    taylor_at: jax.Array | None = None
    taylor_ratio: float = 10**0.5
    mask: Callable = lambda directions: directions

    def directions(self, n: int) -> list[np.ndarray]:
        return self.mask(unit_directions(self.rng, self.p.shape, n))


class Designs:
    """The cells' designs, each built on first use (module-scoped)."""

    def __init__(self, factory) -> None:
        self._factory = factory

    def _dir(self, name):
        return self._factory.mktemp(name)

    @functools.cached_property
    def stack(self):
        return stack(self._dir("stack"))

    @functools.cached_property
    def capacitor(self):
        return capacitor(self._dir("capacitor"))

    @functools.cached_property
    def line(self):
        return layered_line(self._dir("line"))

    @functools.cached_property
    def shape_line(self):
        return shape_line(self._dir("shape_line"))

    @functools.cached_property
    def wave_mesh(self):
        return wave_mesh(self._dir("wave"))

    @functools.cached_property
    def sheet_mesh(self):
        return sheet_mesh(self._dir("sheet"))

    @functools.cached_property
    def sheet_cavity(self):
        return sheet_cavity(self._dir("sheet_cavity"))

    @functools.cached_property
    def resonator(self):
        return resonator_mesh(self._dir("resonator"))

    @functools.cached_property
    def layered_resonator(self):
        return layered_resonator(self._dir("layered_resonator"))

    @functools.cached_property
    def mirrored_cavity(self):
        return mirrored_cavity(self._dir("mirrored"))


@pytest.fixture(scope="module")
def designs(tmp_path_factory):
    return Designs(tmp_path_factory)


def _options(cell: Cell, *, production: bool) -> gfem.SolverOptions:
    extra = {}
    if cell.problem == "eigenmode" and cell.solver == "petsc":
        extra["eigen_solver"] = "slepc"  # a PETSc cell runs the whole PETSc stack
    if production:
        return gfem.SolverOptions(order=2, linear_solver=cell.solver, **extra)
    if cell.problem == "eigenmode":
        extra["eigen_tolerance"] = 1e-12
    return tight(cell.solver, **extra)


# --- the cells' objectives ---------------------------------------------------------

#: Steps on unit directions: fourth-order truncation and round-off both well
#: below the 1e-6 gate (measured worst errors 1e-8 .. 1e-10).
STEPS = {"eps_r": 1e-2, "conductivity": 5e-2, "nodes": 2e-6}
#: Grids of the driven cells: uniform frequencies, adaptive grid and samples.
LINE_FREQUENCIES = (18e9, 27e9)
LINE_GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 21))
LINE_SAMPLES = (LINE_GRID[0], LINE_GRID[20], LINE_GRID[10], LINE_GRID[4], LINE_GRID[16])
GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 11))
WAVE_FREQUENCIES = (12e9, 24e9)
WAVE_SAMPLES = (GRID[0], GRID[10], GRID[5], GRID[2])
SHAPE_FREQUENCIES = (GRID[2], GRID[8])
SHAPE_SAMPLES = (GRID[0], GRID[10], GRID[5])
SHEET_FREQUENCIES = (6e9, 9e9)
SHEET_GRID = tuple(float(f) for f in np.linspace(5e9, 10e9, 6))
SHEET_SAMPLES = (SHEET_GRID[0], SHEET_GRID[5], SHEET_GRID[2])


def _sweep(cell: Cell, frequencies, grid, samples):
    """The cell's problem and the grid rows its objective reads."""
    if cell.sweep == "uniform":
        return gfem.Driven(frequency=frequencies), list(range(len(frequencies)))
    problem = gfem.Driven(frequency=grid, sweep="adaptive", samples=samples)
    return problem, [grid.index(f) for f in samples]


def _electrostatic_eps(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """J(eps) = <W, C(eps)> on a two-terminal stack with a random P0 field."""
    model = d.stack
    weights = np.array([[1.0, 0.3], [-0.7, 2.0]])  # not symmetric on purpose

    def objective(eps):
        c = gfem.solve(stack_fields(model, eps), STACK_PROBLEM, options=options)
        return jnp.sum(weights * c.capacitance) / (EPS0 * 1e-6)  # O(1) values

    rng = np.random.default_rng(42 + seed)
    eps = jnp.asarray(rng.uniform(2.0, 9.0, len(model.mesh.tetrahedra)))
    return Case(objective, eps, rng, step=STEPS["eps_r"], taylor=1e-1)


ELECTROSTATIC = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")


def _electrostatic_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """C (pF) of a capacitor whose dielectric interface moves (+ a potential)."""
    model = d.capacitor
    morph = boundary_morph(model, "oxide__nitride", grows="oxide")
    weights = np.random.default_rng(4).uniform(0.5, 1.5, 200)

    def objective(t):
        result = gfem.solve(
            model.replace(mesh=morph(t)), ELECTROSTATIC, options=options
        )
        value = result.capacitance[0, 0] * 1e12
        if cell.variant == "potential":
            phi = result.potential["top"].coefficients
            value = value + jnp.sum(weights[: phi.shape[0]] * phi[:200])
        return value

    t = jnp.zeros(morph.n_params)
    return Case(
        objective,
        t,
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _driven_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the two-layer line in its per-element eps_r or conductivity."""
    line = d.line
    values, rng = line_design(line, seed)
    problem, rows = _sweep(cell, LINE_FREQUENCIES, LINE_GRID, LINE_SAMPLES)
    name = cell.input

    def objective(p):
        model = with_fields(line, {**values, name: p})
        s = gfem.solve(model, problem, options=options).s.values
        return s_functional(s[jnp.asarray(rows)])

    taylor = {"eps_r": 1e-1, "conductivity": 1.0}[name]
    return Case(objective, values[name], rng, step=STEPS[name], taylor=taylor)


def _driven_fill(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the glued line in the per-face fill of its sheet."""
    mesh = d.sheet_mesh
    problem, rows = _sweep(cell, SHEET_FREQUENCIES, SHEET_GRID, SHEET_SAMPLES)

    def objective_on(problem, rows):
        def objective(fill):
            model = sheet_line(mesh, SHEETS[cell.variant](gfem.SheetField(fill=fill)))
            s = gfem.solve(model, problem, options=options).s.values
            return s_functional(s[jnp.asarray(rows)])

        return objective

    reference = None
    if cell.sweep == "adaptive":
        # the differences are taken on a uniform sweep of the samples
        reference = objective_on(gfem.Driven(frequency=SHEET_SAMPLES), [0, 1, 2])
    fill, rng = sheet_fill(mesh, seed)
    return Case(
        objective_on(problem, rows),
        fill,
        rng,
        step=5e-3,
        taylor=5e-2,
        reference=reference,
    )


def _driven_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the line with every kind of shape term; its interface moves."""
    line = d.shape_line
    morph = boundary_morph(line, "sub__cover", grows="sub")
    problem, rows = _sweep(cell, SHAPE_FREQUENCIES, GRID, SHAPE_SAMPLES)

    def objective(t):
        s = gfem.solve(line.replace(mesh=morph(t)), problem, options=options).s.values
        return s_functional(s[jnp.asarray(rows)])

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        # the line's curvature in t is small: its remainder reaches the
        # BDDC + GMRES error (1e-6) below 5e-5, a third-order bend above 4e-4
        taylor=2e-4,
        taylor_ratio=2.0,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _wave_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S through wave ports in the core's per-element eps_r or conductivity."""
    mesh = d.wave_mesh
    model = wave_line(mesh, cell.variant)
    values, rng = core_design(mesh, seed)
    problem, rows = _sweep(cell, WAVE_FREQUENCIES, GRID, WAVE_SAMPLES)
    name = cell.input

    def objective(p):
        m = model.with_material("core", gfem.MaterialField(**{**values, name: p}))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        s = gfem.solve(m, problem, options=options).s.values
        return s_functional(s[jnp.asarray(rows)])

    # (the lossy uniform BDDC cell's remainder bends above 3e-2)
    taylor = {"eps_r": 3e-2, "conductivity": 1.0}[name]
    return Case(objective, values[name], rng, step=STEPS[name], taylor=taylor)


def _wave_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S through wave ports; the top plate moves, held on the port faces."""
    model = wave_line(d.wave_mesh, cell.variant)
    morph = boundary_morph(model, "top")
    problem, rows = _sweep(cell, WAVE_FREQUENCIES, GRID, WAVE_SAMPLES)

    def objective(t):
        s = gfem.solve(model.replace(mesh=morph(t)), problem, options=options).s.values
        return s_functional(s[jnp.asarray(rows)])

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _omega(model, options, target):
    result = gfem.solve(model, gfem.Eigenmode(target=target), options=options)
    return jnp.real(result.omega[0] * OMEGA_WEIGHT)


def _eigen_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """omega of the terminated line resonator in its eps_r or conductivity."""
    mesh = d.resonator
    values = {
        "eps_r": ramp(mesh, "dielectric", 3.5, 4.5),
        "conductivity": ramp(mesh, "dielectric", 0.2, 0.6),
    }
    name = cell.input

    def objective(p):
        fields = values | {name: p}
        model = terminated(mesh, fields["eps_r"], fields["conductivity"])
        return _omega(model, options, RESONATOR_TARGET)

    # a unit step of 1 S/m would make the 0.2 S/m end of the ramp negative
    taylor = {"eps_r": 1e-1, "conductivity": 3e-1}[name]
    return Case(
        objective,
        values[name],
        np.random.default_rng(seed),
        step=STEPS[name],
        taylor=taylor,
    )


def _eigen_fill(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """omega of the sheet cavity in the per-face fill of its resistive film."""
    mesh = d.sheet_cavity
    fill, rng = cavity_fill(mesh, seed=1 + seed)

    def objective(p):
        return _omega(cavity_model(mesh, p), options, CAVITY_TARGET)

    return Case(
        objective,
        fill,
        rng,
        step=5e-2,
        taylor=1e-1,
        mask=lambda directions: on_the_sheet(directions, fill),
    )


def _eigen_nodes(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """omega of the two-layer resonator; its interface moves."""
    model = d.layered_resonator
    morph = boundary_morph(model, "sub__cover", grows="sub")

    def objective(t):
        return _omega(model.replace(mesh=morph(t)), options, RESONATOR_TARGET)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


BUILDERS = {
    ("electrostatic", "eps_r"): _electrostatic_eps,
    ("electrostatic", "nodes"): _electrostatic_nodes,
    ("driven", "eps_r"): _driven_material,
    ("driven", "conductivity"): _driven_material,
    ("driven", "fill"): _driven_fill,
    ("driven", "nodes"): _driven_nodes,
    ("driven_wave", "eps_r"): _wave_material,
    ("driven_wave", "conductivity"): _wave_material,
    ("driven_wave", "nodes"): _wave_nodes,
    ("eigenmode", "eps_r"): _eigen_material,
    ("eigenmode", "conductivity"): _eigen_material,
    ("eigenmode", "fill"): _eigen_fill,
    ("eigenmode", "nodes"): _eigen_nodes,
}


def _case(designs: Designs, cell: Cell, *, production: bool, seed: int = 0) -> Case:
    build = BUILDERS[cell.problem, cell.input]
    return build(designs, cell, _options(cell, production=production), seed)


# --- the cells -----------------------------------------------------------------------


@pytest.mark.parametrize("cell", [*CELLS, *PETSC_PARAMS], ids=str)
def test_gradient_matrix_matches_central_differences(designs, cell):
    case = _case(designs, cell, production=False)
    assert_gradient_matches(
        case.objective,
        case.p,
        case.directions(3),
        step=case.step,
        reference=case.reference,
    )


@pytest.mark.parametrize("cell", [*CELLS, *PETSC_PARAMS], ids=str)
def test_gradient_matrix_taylor_remainder_is_second_order(designs, cell):
    case = _case(designs, cell, production=True, seed=1)
    (d,) = case.directions(1)
    at = case.p if case.taylor_at is None else case.taylor_at
    steps = [case.taylor / case.taylor_ratio**k for k in range(3)]
    assert_taylor_second_order(case.objective, at, d, steps)


# --- consistency: the same gradient however it is asked for --------------------------

#: One cell per input kind and problem (direct solver), for the transformations.
FAMILIES = [
    Cell("electrostatic", "eps_r", "direct"),
    Cell("electrostatic", "nodes", "direct", variant="potential"),
    Cell("driven", "eps_r", "direct", "uniform"),
    Cell("driven", "eps_r", "direct", "adaptive"),
    Cell("driven", "fill", "direct", "uniform", "conductor"),
    Cell("driven", "nodes", "direct", "uniform"),
    Cell("driven_wave", "eps_r", "direct", "uniform", "lossy"),
    Cell("eigenmode", "conductivity", "direct"),
    Cell("eigenmode", "fill", "direct"),
    Cell("eigenmode", "nodes", "direct"),
]
#: Two solves of one design agree to about 1 ulp (threaded assembly), so do
#: their gradients. An eigenmode gradient is only as reproducible as the
#: refined eigenvector, to the eigen tolerance (1e-12) times the operator's
#: condition: each solve's Arnoldi run starts from a random vector, so two
#: solves' gradients differ by up to ~1e-7 (the value by 1e-15).
RTOL = {"eigenmode": 1e-6}


def _assert_same(got, want, cell):
    got, want = np.asarray(got), np.asarray(want)
    error = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert error <= RTOL.get(cell.problem, 1e-9), f"{cell}: relative error {error:.2e}"


@pytest.mark.parametrize("cell", FAMILIES, ids=str)
def test_gradient_matrix_transformations_and_eviction_agree(designs, cell):
    # vmap(grad) is a loop of solves, a pullback can be called twice, and a
    # backward without its cache entry recomputes it from the residual design
    case = _case(designs, cell, production=False)
    (d,) = case.directions(1)
    batch = jnp.stack([case.p, case.p + 10 * case.step * d])
    grad = jax.grad(case.objective)
    looped = jnp.stack([grad(p) for p in batch])
    assert np.abs(np.asarray(looped)).max() > 0
    vmapped = jax.vmap(grad)(batch)
    for k in range(2):
        _assert_same(vmapped[k], looped[k], cell)

    _, pullback = jax.vjp(case.objective, case.p)
    first = pullback(1.0)[0]
    _assert_same(first, looped[0], cell)
    # the same kept entry: the second pullback repeats the first exactly
    np.testing.assert_allclose(pullback(2.0)[0], 2 * first, rtol=1e-12)

    options = dataclasses.replace(_options(cell, production=False), retain_memory=0)
    evicted = BUILDERS[cell.problem, cell.input](designs, cell, options, 0)
    _assert_same(jax.grad(evicted.objective)(case.p), looped[0], cell)


@pytest.mark.parametrize(
    "solver", [*SOLVERS, pytest.param("petsc", marks=pytest.mark.petsc)]
)
def test_gradient_matrix_reciprocity_equals_the_explicit_adjoint(designs, solver):
    # Exciting p1 only leaves S21 without p2's solution: its gradient takes
    # an adjoint solve. With both ports excited it comes from reciprocity.
    line = designs.line
    values, _ = line_design(line, seed=3)
    options = tight(solver)

    def s21_gradient(problem):
        def objective(eps):
            model = with_fields(line, {**values, "eps_r": eps})
            s = gfem.solve(model, problem, options=options).s
            return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

        return jax.grad(objective)(values["eps_r"])

    full = s21_gradient(gfem.Driven(frequency=LINE_FREQUENCIES))
    adjoint = s21_gradient(gfem.Driven(frequency=LINE_FREQUENCIES, excite="p1"))
    np.testing.assert_allclose(adjoint, full, rtol=1e-8, atol=1e-10 * abs(full).max())


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["eps_r", "conductivity"])
def test_gradient_matrix_adaptive_equals_the_full_solve_at_the_samples(
    designs, solver, name
):
    # at a sample the full solution lies in the frozen basis
    line = designs.line
    values, _ = line_design(line)
    options = tight(solver)
    adaptive_cell = Cell("driven", name, solver, "adaptive")
    adaptive = _driven_material(designs, adaptive_cell, options, 0).objective

    def full(p):
        model = with_fields(line, {**values, name: p})
        problem = gfem.Driven(frequency=LINE_SAMPLES)
        return s_functional(gfem.solve(model, problem, options=options).s.values)

    p = values[name]
    value, grad = jax.value_and_grad(adaptive)(p)
    full_value, full_grad = jax.value_and_grad(full)(p)
    assert float(value) == pytest.approx(float(full_value), rel=1e-9)
    assert float(jnp.linalg.norm(grad - full_grad) / jnp.linalg.norm(full_grad)) <= 1e-7


# --- degenerate eigenmode clusters -------------------------------------------------
# The TE101 / TE011 pair of a square-based PEC box on a mesh with the
# square's symmetry: exactly degenerate. Only functions symmetric in the pair
# differentiate; one member alone raises gfem.DegenerateModeError (on the
# host, in the backward pass: JAX re-raises it as a JaxRuntimeError that
# keeps the message).

DEGENERATE = jax.errors.JaxRuntimeError
CLUSTER_OPTIONS = gfem.SolverOptions(eigen_tolerance=1e-12)


def _cavity_modes(mesh, eps, n_modes):
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.MaterialField(eps_r=eps, tan_delta=1e-3)},
        boundaries=[gfem.PEC("wall")],
    )
    problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=n_modes)
    return gfem.solve(model, problem, options=CLUSTER_OPTIONS)


def _uniform(mesh):
    return jnp.ones(len(mesh.elements("fill")))


def test_gradient_matrix_degenerate_pair_is_labelled_as_one_cluster(designs):
    cavity = designs.mirrored_cavity
    result = _cavity_modes(cavity, _uniform(cavity), 3)
    omega = np.asarray(result.omega)
    assert abs(omega[0] - omega[1]) <= 1e-12 * abs(omega[0])
    np.testing.assert_array_equal(np.asarray(result.cluster), [0, 0, 1])


@pytest.mark.parametrize("overlap", [0.95, 0.999])
def test_gradient_matrix_nearly_parallel_ritz_vectors_keep_the_pair(
    designs, monkeypatch, overlap
):
    # ARPACK returns an arbitrary basis of a multiple eigenvalue's eigenspace:
    # how parallel its two vectors are depends on round-off (process state)
    import gfem._eigen.rule as eigen_rule

    shift_invert = eigen_rule.shift_invert

    def nearly_parallel(*args, **kwargs):
        k, vectors = shift_invert(*args, **kwargs)
        i, j = np.argsort(np.where(k.real > 0, k.real, np.inf))[:2]
        assert abs(k[i] - k[j]) <= 1e-12 * abs(k[i])  # the pair
        q = np.linalg.qr(vectors[:, [i, j]])[0]
        vectors = vectors.copy()
        vectors[:, i] = q[:, 0]
        vectors[:, j] = overlap * q[:, 0] + np.sqrt(1 - overlap**2) * q[:, 1]
        return k, vectors

    monkeypatch.setattr(eigen_rule, "shift_invert", nearly_parallel)
    cavity = designs.mirrored_cavity
    result = _cavity_modes(cavity, _uniform(cavity), 3)
    np.testing.assert_array_equal(np.asarray(result.cluster), [0, 0, 1])


def test_gradient_matrix_cluster_sum_matches_central_differences(designs):
    cavity = designs.mirrored_cavity

    def objective(eps):
        omega = _cavity_modes(cavity, eps, 2).omega
        return jnp.real(jnp.sum(omega) * OMEGA_WEIGHT)

    eps = _uniform(cavity)
    rng = np.random.default_rng(3)
    directions = unit_directions(rng, eps.shape, 3)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_gradient_matrix_cluster_smooth_max_matches_central_differences(designs):
    # any function symmetric in the pair: here a smooth max of the frequencies
    cavity = designs.mirrored_cavity

    def objective(eps):
        f = _cavity_modes(cavity, eps, 3).frequency[:2] / 1e9
        return jax.nn.logsumexp(10 * f) / 10

    eps = _uniform(cavity)
    rng = np.random.default_rng(4)
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_gradient_matrix_single_mode_of_a_cluster_raises(designs):
    cavity = designs.mirrored_cavity

    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 2).omega[0])

    with pytest.raises(DEGENERATE, match="DegenerateModeError") as info:
        jax.grad(objective)(_uniform(cavity))
    message = str(info.value)
    assert "18.0" in message  # the cluster's frequencies
    assert "sum or mean" in message  # the remedy


def test_gradient_matrix_cluster_cut_by_n_modes_raises(designs):
    # n_modes=1 keeps one member of the pair: the guard mode shows the rest
    cavity = designs.mirrored_cavity

    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 1).omega[0])

    with pytest.raises(DEGENERATE, match=r"DegenerateModeError.*n_modes"):
        jax.grad(objective)(_uniform(cavity))


def test_gradient_matrix_simple_mode_beside_a_cluster_differentiates(designs):
    # TM110 alone: no cotangent reaches the pair
    cavity = designs.mirrored_cavity

    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 3).omega[2] * OMEGA_WEIGHT)

    eps = _uniform(cavity)
    rng = np.random.default_rng(5)
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_gradient_matrix_stop_gradient_on_the_other_member_still_raises(designs):
    # the cotangents differ (one is zero): not symmetric, whatever the opt-out
    cavity = designs.mirrored_cavity

    def objective(eps):
        omega = _cavity_modes(cavity, eps, 2).omega
        return jnp.real(omega[0] + jax.lax.stop_gradient(omega[1]))

    with pytest.raises(DEGENERATE, match="DegenerateModeError"):
        jax.grad(objective)(_uniform(cavity))


def test_gradient_matrix_cluster_sum_in_the_node_coordinates(designs):
    # a shape design that keeps the square's symmetry: one side wall moves
    # (so the pair splits along the design direction); the cluster sum is
    # symmetric whatever the split
    cavity = designs.mirrored_cavity
    model = gfem.Model(
        cavity,
        materials={"fill": gfem.Material(eps_r=1.0, tan_delta=1e-3)},
        boundaries=[gfem.PEC("wall")],
    )
    morph = boundary_morph(model, "wall", where=lambda x: np.abs(x[:, 2]) < 1e-9)
    problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=2)

    def objective(t):
        moved = model.replace(mesh=morph(t))
        omega = gfem.solve(moved, problem, options=CLUSTER_OPTIONS).omega
        return jnp.real(jnp.sum(omega) * OMEGA_WEIGHT)

    t = jnp.zeros(morph.n_params)
    rng = np.random.default_rng(6)
    directions = unit_directions(rng, t.shape, 2)
    assert_gradient_matches(objective, t, directions, step=STEPS["nodes"])
