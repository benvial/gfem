"""Sheet-field gradients (driven and eigenmode) against finite differences.

**Driven**: a parallel-plate line glued from two halves at mid-height, its
TEM field tangential to the interior surface "sheet" between them, lumped
ports at both ends. A thin-film `Conductor` or an R + L `Impedance` on the
sheet carries a per-face fill; S differentiates with respect to the fill
through the face pass of the sensitivity kernel. The whole design chain
(pixels on a `LayerGrid` of the sheet, Helmholtz filter, projection,
exponential fill) differentiates end to end.

**Eigenmode**: a 20 x 10 x 20 mm PEC cavity glued at z = 10 mm, where the
TE101 field is largest, with a resistive thin-film `Conductor` sheet of
random per-face fill there. The film's ``Z_s(omega)`` makes the operator
nonpolynomial and the fill makes that term design-dependent. The refined
root equals the root of the same discrete operator found by a Rayleigh
iteration that refactorizes ``T(omega)`` at every step (assembled here
independently, the fill as a face-wise NGSolve coefficient), and
``d omega / d fill`` passes finite differences and the Taylor test.

Each cell runs on the direct solver and on BDDC + GMRES.
"""

import cmath
import math

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import pytest
import scipy.constants as const
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import structured_meshes as sm
from finite_differences import (
    assert_gradient_matches,
    assert_taylor_second_order,
    unit_directions,
)

import gfem
from gfem import design
from gfem._engine.mesh import to_ngsolve

SOLVERS = ["direct", "bddc_gmres"]
ETA0 = const.mu_0 * const.c

# --- driven: the glued line --------------------------------------------------------

L, W, H = 8.0, 2.0, 1.0  # mm
FREQUENCIES = (6e9, 9e9)
#: Thin film: sigma t / 2 = 5e-4 S per square, comparable with the line's
#: wave admittance over its length.
SIGMA, THICKNESS = 1e3, 1e-6
SHEETS = {
    "conductor": lambda fill: gfem.Conductor("sheet", SIGMA, THICKNESS, fill=fill),
    "impedance": lambda fill: gfem.Impedance(
        "sheet", resistance=2e3, inductance=2e-8, fill=fill
    ),
}


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "sheet.msh"
    sm.write_msh22(path, sm.sheet_line(L, W, H, (6, 2, 1)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line_model(mesh, sheet):
    return gfem.Model(
        mesh,
        materials={"lower": gfem.Material(eps_r=3.0), "upper": gfem.Material()},
        boundaries=[gfem.PEC("plate1"), gfem.PEC("plate2"), sheet],
        ports=[
            gfem.LumpedPort("p1", "port1", "+y", impedance=200.0),
            gfem.LumpedPort("p2", "port2", "+y", impedance=200.0),
        ],
    )


def _options(solver, **kw):
    return gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12, **kw)


WEIGHTS = np.array([[1.0, 0.4], [-0.6, 1.5]])


def _s_objective(mesh, sheet, options, problem=None, rows=None):
    problem = problem or gfem.Driven(frequency=FREQUENCIES)

    def objective(fill):
        model = _line_model(mesh, SHEETS[sheet](gfem.SheetField(fill=fill)))
        s = gfem.solve(model, problem, options=options).s.values
        if rows is not None:
            s = s[jnp.asarray(rows)]
        return jnp.sum(WEIGHTS * jnp.abs(s) ** 2) + jnp.sum(
            jnp.real((0.3 - 0.7j) * s[:, 1, 0])
        )

    return objective


def _fill(mesh, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.2, 0.8, len(mesh.elements("sheet")))), rng


#: Fourth-order differences on unit directions (tests/finite_differences.py).
FILL_STEP = 5e-3
FILL_TAYLOR = [5e-2, 5e-3, 5e-4]


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("sheet", sorted(SHEETS))
def test_s_gradient_in_the_fill_matches_central_differences(line, solver, sheet):
    fill, rng = _fill(line, 0)
    objective = _s_objective(line, sheet, _options(solver))
    directions = unit_directions(rng, fill.shape, 3)
    assert_gradient_matches(objective, fill, directions, step=FILL_STEP)


@pytest.mark.parametrize("solver", SOLVERS)
def test_s_taylor_remainder_in_the_fill_is_second_order(line, solver):
    fill, rng = _fill(line, 1)
    objective = _s_objective(line, "conductor", _options(solver))
    (d,) = unit_directions(rng, fill.shape, 1)
    assert_taylor_second_order(objective, fill, d, FILL_TAYLOR)


def test_adaptive_sweep_gradient_in_the_fill_matches_central_differences(line):
    # frozen samples: the sheet is one more design-dependent affine piece. At
    # the samples the frozen-basis gradient is the full solve's, so the
    # differences are taken on a uniform sweep of the samples.
    grid = tuple(float(f) for f in np.linspace(5e9, 10e9, 6))
    samples = (grid[0], grid[5], grid[2])
    adaptive = gfem.Driven(frequency=grid, sweep="adaptive", samples=samples)
    rows = [grid.index(f) for f in sorted(samples)]
    options = _options("direct")
    objective = _s_objective(line, "impedance", options, adaptive, rows)
    full = _s_objective(
        line, "impedance", options, gfem.Driven(frequency=sorted(samples))
    )
    fill, rng = _fill(line, 2)
    directions = unit_directions(rng, fill.shape, 2)
    assert_gradient_matches(objective, fill, directions, step=FILL_STEP, reference=full)


def test_pixel_design_chain_differentiates_end_to_end(line):
    # pixels -> filter -> projection -> faces -> exponential fill -> S
    grid = design.LayerGrid(line, "sheet", pitch=1e-3)
    blur = design.helmholtz_filter(grid, radius=2e-3)
    problem = gfem.Driven(frequency=FREQUENCIES)
    options = _options("direct")

    def objective(rho):
        density = grid.to_faces(design.project(blur(rho), beta=2.0))
        fill = gfem.SheetField.interpolate(density, contrast=1e4)
        model = _line_model(line, SHEETS["conductor"](fill))
        s = gfem.solve(model, problem, options=options).s.values
        return jnp.sum(WEIGHTS * jnp.abs(s) ** 2)

    rng = np.random.default_rng(3)
    rho = jnp.asarray(rng.uniform(0.3, 0.7, grid.shape))
    directions = unit_directions(rng, rho.shape, 2)
    assert_gradient_matches(objective, rho, directions, step=FILL_STEP)


# --- eigenmode: the glued cavity -----------------------------------------------------

A, B, D = 20.0, 10.0, 20.0  # mm
SCALE = 1e-3
CELLS = (6, 3, 3)  # per half along z
#: A resistive thin film: Q of a few hundred at TE101 (10.6 GHz).
CAVITY_SIGMA, CAVITY_THICKNESS = 100.0, 1e-6
OMEGA_101 = const.c * math.pi * math.hypot(1 / (A * SCALE), 1 / (D * SCALE))
CAVITY_TARGET = 10e9
TIGHT = {"eigen_tolerance": 1e-12}
#: Weights the real and the imaginary part of omega (GHz): frequency and loss.
OMEGA_WEIGHT = (1 - 30j) / (2 * math.pi * 1e9)


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    edges = (
        np.linspace(0, A, CELLS[0] + 1),
        np.linspace(0, B, CELLS[1] + 1),
        np.linspace(0, D, 2 * CELLS[2] + 1),
    )

    def surface(c):
        if abs(c[2] - D / 2) < 1e-9:
            return "sheet"
        on = (c[0], A - c[0], c[1], B - c[1], c[2], D - c[2])
        return "pec" if min(np.abs(on)) < 1e-9 else ""

    path = tmp_path_factory.mktemp("cavity") / "cavity.msh"
    mesh = sm.structured(edges, lambda c: "lower" if c[2] < D / 2 else "upper", surface)
    sm.write_msh22(path, mesh)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _cavity_fill(mesh, seed=0):
    # a partial sheet: about a third of the faces empty, the rest grey
    rng = np.random.default_rng(seed)
    n = len(mesh.elements("sheet"))
    fill = rng.uniform(0.2, 1.0, n) * (rng.uniform(size=n) > 0.3)
    return jnp.asarray(fill), rng


def _cavity_model(mesh, fill):
    sheet = gfem.Conductor(
        "sheet", CAVITY_SIGMA, CAVITY_THICKNESS, fill=gfem.SheetField(fill=fill)
    )
    return gfem.Model(
        mesh,
        materials={"lower": gfem.Material(), "upper": gfem.Material()},
        boundaries=[gfem.PEC("pec"), sheet],
    )


def _cavity_omega(mesh, fill, solver="direct", target=CAVITY_TARGET):
    options = gfem.SolverOptions(linear_solver=solver, **TIGHT)
    problem = gfem.Eigenmode(target=target)
    return gfem.solve(_cavity_model(mesh, fill), problem, options=options).omega[0]


def _film_impedance(omega):
    """Thin-film Z_s, analytic in omega: (g / sigma) coth(g t / 2)."""
    g = cmath.sqrt(1j * omega * const.mu_0 * CAVITY_SIGMA)
    return g / CAVITY_SIGMA / cmath.tanh(0.5 * g * CAVITY_THICKNESS)


def _hfss_film_impedance(omega):
    """The finite-thickness surface impedance of HFSS / Palace, real omega."""
    delta = math.sqrt(2 / (const.mu_0 * CAVITY_SIGMA * omega))
    v = CAVITY_THICKNESS / delta
    num = (math.sinh(v) + math.sin(v)) + 1j * (math.sinh(v) - math.sin(v))
    return num / (CAVITY_SIGMA * delta * (math.cosh(v) - math.cos(v)))


def _reference_root(mesh, fill) -> complex:
    """TE101 of the discrete operator, refactorizing T(omega) at every step."""
    built = to_ngsolve(mesh)
    space = ngs.HCurl(built.mesh, order=2, type1=True, dirichlet="pec", complex=True)
    u, v = space.TnT()
    per_face = np.zeros(len(mesh.triangles))
    per_face[mesh.elements("sheet")] = np.asarray(fill)
    weight = ngs.GridFunction(ngs.SurfaceL2(built.mesh, order=0))
    weight.vec.FV().NumPy()[:] = per_face[built.triangle_order]

    def assemble(integrand):
        form = ngs.BilinearForm(space)
        form += integrand
        form.Assemble()
        rows, cols, values = form.mat.COO()
        shape = (space.ndof, space.ndof)
        return sp.csr_matrix(
            (np.asarray(values), (np.asarray(rows), np.asarray(cols))), shape
        )

    free = np.flatnonzero(np.array(list(space.FreeDofs()), dtype=bool))
    stiffness, mass, sheet = (
        assemble(f)[free][:, free].tocsc()
        for f in (
            ngs.curl(u) * ngs.curl(v) * ngs.dx,
            u * v * ngs.dx,
            weight * u.Trace() * v.Trace() * ngs.ds("sheet"),
        )
    )

    def operator(omega):
        k = omega * SCALE / const.c
        return (
            stiffness - k * k * mass + (1j * k * ETA0 / _film_impedance(omega)) * sheet
        )

    def derivative(omega):  # dT/domega, central difference (T is analytic)
        h = 1e-6 * abs(omega)
        return (operator(omega + h) - operator(omega - h)) / (2 * h)

    def rayleigh(omega, x):
        for _ in range(30):
            step = (x @ (operator(omega) @ x)) / (x @ (derivative(omega) @ x))
            omega -= step
            if abs(step) < 1e-16 * abs(omega):
                break
        return omega

    k2, vectors = spla.eigsh(
        stiffness.real, k=1, M=mass.real, sigma=(OMEGA_101 * SCALE / const.c) ** 2
    )
    omega = complex(math.sqrt(k2[0]) * const.c / SCALE)
    x = vectors[:, 0].astype(complex)
    for _ in range(10):
        omega = rayleigh(omega, x)
        x = spla.splu(operator(omega)).solve(derivative(omega) @ x)
        x /= np.linalg.norm(x)
    return rayleigh(omega, x)


def test_film_impedance_is_the_hfss_form_continued():
    for f in (1e9, 10.6e9, 30e9):
        w = 2 * math.pi * f
        assert abs(_film_impedance(w) - _hfss_film_impedance(w)) <= 1e-9 * abs(
            _hfss_film_impedance(w)
        )


@pytest.fixture(scope="module")
def reference(cavity):
    fill, _ = _cavity_fill(cavity)
    return fill, _reference_root(cavity, fill)


@pytest.mark.parametrize("target", [8.5e9, 10e9, 13e9])
def test_sheet_mode_matches_a_refactorizing_reference(cavity, reference, target):
    fill, exact = reference
    assert abs(exact.real / (2 * math.pi) - 10.6e9) < 0.1e9
    q = abs(exact) / (2 * exact.imag)
    assert 50 < q < 2000  # the sheet loads the mode
    omega = complex(_cavity_omega(cavity, fill, target=target))
    assert abs(omega - exact) <= 1e-12 * abs(exact)


#: Fourth-order differences of omega in the fill (unit directions).
EIGEN_STEP = 5e-2


def _on_the_sheet(directions, fill):
    """Unit directions moving only the filled faces (a fill stays >= 0)."""
    on = np.asarray(fill) > 0
    return [d * on / np.linalg.norm(d * on) for d in directions]


@pytest.mark.parametrize("solver", SOLVERS)
def test_mode_gradient_in_the_fill_matches_central_differences(cavity, solver):
    fill, rng = _cavity_fill(cavity, seed=1)

    def objective(p):
        return jnp.real(_cavity_omega(cavity, p, solver) * OMEGA_WEIGHT)

    directions = _on_the_sheet(unit_directions(rng, fill.shape, 3), fill)
    assert_gradient_matches(objective, fill, directions, step=EIGEN_STEP)


def test_mode_taylor_remainder_in_the_fill_is_second_order(cavity):
    fill, rng = _cavity_fill(cavity, seed=2)

    def objective(p):
        return jnp.real(_cavity_omega(cavity, p) * OMEGA_WEIGHT)

    (d,) = _on_the_sheet(unit_directions(rng, fill.shape, 1), fill)
    assert_taylor_second_order(objective, fill, d, [1e-1, 1e-2, 1e-3])


def test_q_gradient_in_the_fill_has_the_sign_of_the_loss(cavity):
    # more sheet, more loss: every face's fill lowers Q where E_y is tangential
    fill, _ = _cavity_fill(cavity, seed=3)
    options = gfem.SolverOptions(**TIGHT)

    def q(p):
        result = gfem.solve(
            _cavity_model(cavity, p),
            gfem.Eigenmode(target=CAVITY_TARGET),
            options=options,
        )
        return result.q[0]

    grad = np.asarray(jax.grad(q)(fill))
    assert (grad < 0).mean() > 0.9
