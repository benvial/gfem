"""Sheet-field gradients (driven and eigenmode) against finite differences.

**Driven**: a parallel-plate line glued from two halves at mid-height, its
TEM field tangential to the interior surface "sheet" between them, lumped
ports at both ends. A thin-film `Conductor` or an R + L `Impedance` on the
sheet carries a per-face fill; S differentiates with respect to the fill
through the face pass of the sensitivity kernel. The whole design chain
(pixels on a `LayerGrid` of the sheet, Helmholtz filter, projection,
exponential fill) differentiates end to end, and so do node coordinates
moving the filled sheet itself.

**Eigenmode**: a 20 x 10 x 20 mm PEC cavity glued at z = 10 mm, where the
TE101 field is largest, with a resistive thin-film `Conductor` sheet of
random per-face fill there. The film's ``Z_s(omega)`` makes the operator
nonpolynomial and the fill makes that term design-dependent. The refined
root equals the root of the same discrete operator found by a Rayleigh
iteration that refactorizes ``T(omega)`` at every step (assembled here
independently, the fill as a face-wise NGSolve coefficient), and
``d omega / d fill`` passes finite differences and the Taylor test (the fill
cells of the verification matrix, driven and eigenmode, are in
test_gradient_matrix.py; the designs in `gradient_cases`).
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
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import (
    CAVITY,
    CAVITY_SIGMA,
    CAVITY_TARGET,
    CAVITY_THICKNESS,
    SHEETS,
    WEIGHTS,
    cavity_fill,
    cavity_model,
    sheet_cavity,
    sheet_fill,
    sheet_line,
    sheet_mesh,
    tight,
)

import gfem
from gfem import design
from gfem._engine.mesh import to_ngsolve

ETA0 = const.mu_0 * const.c

# --- driven: the glued line --------------------------------------------------------

FREQUENCIES = (6e9, 9e9)


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    return sheet_mesh(tmp_path_factory.mktemp("line"))


#: Fourth-order differences on unit directions (tests/finite_differences.py).
FILL_STEP = 5e-3


def test_pixel_design_chain_differentiates_end_to_end(line):
    # pixels -> filter -> projection -> faces -> exponential fill -> S
    grid = design.LayerGrid(line, "sheet", pitch=1e-3)
    blur = design.helmholtz_filter(grid, radius=2e-3)
    problem = gfem.Driven(frequency=FREQUENCIES)
    options = tight("direct")

    def objective(rho):
        density = grid.to_faces(design.project(blur(rho), beta=2.0))
        fill = gfem.SheetField.interpolate(density, contrast=1e4)
        model = sheet_line(line, SHEETS["conductor"](fill))
        s = gfem.solve(model, problem, options=options).s.values
        return jnp.sum(WEIGHTS * jnp.abs(s) ** 2)

    rng = np.random.default_rng(3)
    rho = jnp.asarray(rng.uniform(0.3, 0.7, grid.shape))
    directions = unit_directions(rng, rho.shape, 2)
    assert_gradient_matches(objective, rho, directions, step=FILL_STEP)


def test_shape_gradient_through_a_filled_sheet(line):
    # the sheet itself moves vertically: its term is a surface trace term
    # whose per-face fill travels with the faces
    from gfem.shape import boundary_morph

    fill, rng = sheet_fill(line, 4)
    sheet = SHEETS["impedance"](gfem.SheetField(fill=fill))
    base = sheet_line(line, sheet)
    morph = boundary_morph(base, "sheet", grows="lower")
    problem = gfem.Driven(frequency=FREQUENCIES)
    options = tight("direct")

    def objective(t):
        s = gfem.solve(base.replace(mesh=morph(t)), problem, options=options).s
        return jnp.sum(WEIGHTS * jnp.abs(s.values) ** 2)

    t = jnp.zeros(morph.n_params)
    directions = unit_directions(rng, t.shape, 2)
    assert_gradient_matches(objective, t, directions, step=2e-6)


# --- eigenmode: the glued cavity -----------------------------------------------------

A, B, D = CAVITY  # mm
SCALE = 1e-3
OMEGA_101 = const.c * math.pi * math.hypot(1 / (A * SCALE), 1 / (D * SCALE))
TIGHT = {"eigen_tolerance": 1e-12}


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    return sheet_cavity(tmp_path_factory.mktemp("cavity"))


def _cavity_omega(mesh, fill, solver="direct", target=CAVITY_TARGET):
    options = gfem.SolverOptions(linear_solver=solver, **TIGHT)
    problem = gfem.Eigenmode(target=target)
    return gfem.solve(cavity_model(mesh, fill), problem, options=options).omega[0]


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
    fill, _ = cavity_fill(cavity)
    return fill, _reference_root(cavity, fill)


@pytest.mark.parametrize("target", [8.5e9, 10e9, 13e9])
def test_sheet_mode_matches_a_refactorizing_reference(cavity, reference, target):
    fill, exact = reference
    assert abs(exact.real / (2 * math.pi) - 10.6e9) < 0.1e9
    q = abs(exact) / (2 * exact.imag)
    assert 50 < q < 2000  # the sheet loads the mode
    omega = complex(_cavity_omega(cavity, fill, target=target))
    assert abs(omega - exact) <= 1e-12 * abs(exact)


def test_q_gradient_in_the_fill_has_the_sign_of_the_loss(cavity):
    # more sheet, more loss: every face's fill lowers Q where E_y is tangential
    fill, _ = cavity_fill(cavity, seed=3)
    options = gfem.SolverOptions(**TIGHT)

    def q(p):
        result = gfem.solve(
            cavity_model(cavity, p),
            gfem.Eigenmode(target=CAVITY_TARGET),
            options=options,
        )
        return result.q[0]

    grad = np.asarray(jax.grad(q)(fill))
    assert (grad < 0).mean() > 0.9
