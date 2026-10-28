"""gfem's GMRES: NGSolve's GMRES iteration, its vector work on gfem's threads."""

import ngsolve as ngs
import numpy as np
import pytest
from netgen.geom2d import unit_square

from gfem._engine.runtime import task_manager
from gfem._linalg import gmres
from gfem._linalg.gmres import Gmres


@pytest.fixture(autouse=True)
def _restore_threads():
    """``SetNumThreads`` is process-wide: put the count back after each test."""
    with ngs.TaskManager():
        threads = ngs.GetNumThreads()
    yield
    ngs.SetNumThreads(threads)


@pytest.fixture(scope="module")
def mesh():
    return ngs.Mesh(unit_square.GenerateMesh(maxh=0.05))


def _helmholtz(mesh, complex_valued):
    """Helmholtz on H1 order 2 with a Jacobi preconditioner: hard for GMRES."""
    space = ngs.H1(mesh, order=2, complex=complex_valued, dirichlet="left")
    u, v = space.TnT()
    k2 = 30 - 2j if complex_valued else 30
    form = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx - k2 * u * v * ngs.dx)
    pre = ngs.Preconditioner(form, "local")
    form.Assemble()
    rhs = ngs.LinearForm(ngs.x * v * ngs.dx).Assemble()
    return form.mat, pre.mat, rhs.vec


def _diagonal(values):
    """A sparse diagonal matrix and the identity of its size."""
    n = len(values)
    index = list(range(n))
    A = ngs.la.SparseMatrixd.CreateFromCOO(index, index, list(values), n, n)
    eye = ngs.la.SparseMatrixd.CreateFromCOO(index, index, [1.0] * n, n, n)
    return A, eye


def _relative(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(y)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("complex_valued", [True, False])
@pytest.mark.parametrize("kernel", ["auto", "einsum"])
def test_gmres_matches_ngsolve_gmres(
    mesh, monkeypatch, threads, complex_valued, kernel
):
    if kernel == "einsum":
        monkeypatch.setattr(gmres, "scipy_openblas", lambda: None)
    # a small problem: lower the minimum so that 4 threads run the pool
    monkeypatch.setattr(gmres, "COLUMNS", 64)
    A, pre, b = _helmholtz(mesh, complex_valued)
    with task_manager(threads):
        assert ngs.GetNumThreads() == threads
        out = {}
        for cls in (ngs.solvers.GMResSolver, Gmres):
            cold = cls(A, pre, tol=1e-10, maxiter=400)
            x = cold.Solve(b)
            # warm: an absolute tolerance, from a perturbed iterate
            warm = cls(A, pre, atol=1e-10 * cold.residuals[0], maxiter=400)
            x0 = x.CreateVector()
            x0.FV().NumPy()[:] = x.FV().NumPy() * (1 + 1e-3)
            warm.Solve(b, sol=x0, initialize=False)
            out[cls] = (cold.iterations, x.FV().NumPy().copy())
            out[cls] += (warm.iterations, x0.FV().NumPy().copy())
    ours, theirs = out[Gmres], out[ngs.solvers.GMResSolver]
    assert theirs[0] > 100  # Jacobi on Helmholtz: CGS2 needs to be as good as MGS
    assert ours[0] == theirs[0]
    assert _relative(ours[1], theirs[1]) < 1e-12
    assert ours[2] == theirs[2]
    assert _relative(ours[3], theirs[3]) < 1e-12


def test_gmres_returns_the_solution_on_a_lucky_breakdown():
    # three distinct eigenvalues: the Krylov space is exhausted at the third step
    diagonal = np.repeat([1.0, 2.0, 5.0], 10)
    A, eye = _diagonal(diagonal)
    b = A.CreateColVector()
    b.FV().NumPy()[:] = np.arange(1, len(diagonal) + 1)
    solver = Gmres(A, eye, tol=1e-10, maxiter=50)
    x = solver.Solve(b)
    exact = b.FV().NumPy() / diagonal
    assert _relative(x.FV().NumPy(), exact) < 1e-12
    assert solver.residuals[-1] <= 1e-10 * solver.residuals[0]
    assert solver.iterations < 50


class _Recording(gmres.ThreadPoolExecutor):
    """A thread pool that records how it was made."""

    made: list[dict] = []

    def __init__(self, max_workers=None, *args, **kwargs):
        type(self).made.append({"max_workers": max_workers, **kwargs})
        super().__init__(max_workers, *args, **kwargs)


@pytest.mark.parametrize(
    ("threads", "n", "pools"),
    [
        (3, 3 * gmres.COLUMNS, 1),
        (1, 3 * gmres.COLUMNS, 0),
        (3, 2 * gmres.COLUMNS - 1, 0),
    ],
)
def test_gmres_runs_on_the_task_manager_threads(monkeypatch, threads, n, pools):
    monkeypatch.setattr(gmres, "ThreadPoolExecutor", _Recording)
    monkeypatch.setattr(_Recording, "made", [])
    diagonal = np.resize([1.0, 2.0, 5.0], n)
    A, eye = _diagonal(diagonal)
    b = A.CreateColVector()
    b.FV().NumPy()[:] = 1.0
    with task_manager(threads):
        x = Gmres(A, eye, tol=1e-10, maxiter=10).Solve(b)
    assert _relative(x.FV().NumPy(), 1.0 / diagonal) < 1e-12
    assert len(_Recording.made) == pools
    for made in _Recording.made:
        assert made["max_workers"] == threads
        assert made["initializer"] is gmres._serial_blas


def _kernels():
    lib = gmres.scipy_openblas()
    openblas = pytest.param(
        "openblas",
        marks=pytest.mark.skipif(lib is None, reason="SciPy's OpenBLAS not found"),
    )
    return [openblas, "einsum"]


@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
@pytest.mark.parametrize("name", _kernels())
def test_gmres_kernels_agree(name, dtype):
    kernel = (
        gmres._OpenBlas(gmres.scipy_openblas())
        if name == "openblas"
        else gmres._Einsum()
    )
    rng = np.random.default_rng(3)

    def random(*shape):
        out = rng.standard_normal(shape)
        if dtype is np.complex128:
            out = out + 1j * rng.standard_normal(shape)
        return out.astype(dtype)

    n, rows = 1000, 5
    block, q, y = random(gmres.BLOCK, n), random(n), random(rows)
    columns = slice(130, 870)
    h = kernel.coefficients(block, rows, columns, q)
    expected = [np.vdot(block[i, columns], q[columns]) for i in range(rows)]
    assert _relative(h, np.array(expected)) < 1e-13
    out = q.copy()
    kernel.add(block, rows, columns, y, out)
    expected = q.copy()
    expected[columns] += y @ block[:rows, columns]
    assert _relative(out, expected) < 1e-13
