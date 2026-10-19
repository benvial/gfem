"""The solver-handle seam: prepare once, solve many right-hand sides, free."""

import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import quadrature
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import Operator, factorize, prepare


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 2)))
    return to_ngsolve(gfem.Mesh.from_gmsh(path))


def _laplace(built, complex_valued=False, shift=0.0, order=2):
    space = ngs.H1(
        built.mesh, order=order, dirichlet="bottom|top", complex=complex_valued
    )
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += (ngs.grad(u) * ngs.grad(v) + shift * u * v) * quadrature.dx(order)
    return Operator(form, space.FreeDofs()), space


def _dense(handle, n):
    return handle.apply(np.eye(n))


def test_direct_solves_many_right_hand_sides_on_the_free_dofs(built):
    op, space = _laplace(built)
    handle = prepare(op, symmetry="hermitian", kind="direct")
    free = np.array(list(space.FreeDofs()))
    rng = np.random.default_rng(0)
    B = rng.standard_normal((space.ndof, 3))
    X = handle.solve(B)
    assert X.shape == (space.ndof, 3)
    np.testing.assert_array_equal(X[~free], 0)
    A = _dense(handle, space.ndof)
    np.testing.assert_allclose((A @ X)[free], B[free], atol=1e-10 * np.abs(B).max())
    # one column at a time, and again after many solves: the same factors
    np.testing.assert_allclose(handle.solve(B[:, 1]), X[:, 1], rtol=1e-14)
    np.testing.assert_allclose(handle.solve(B, transpose=True), X, rtol=1e-14)
    assert handle.nbytes > 0


def test_direct_always_uses_sparse_cholesky_never_the_default(built):
    # NGSolve's default inverse is UMFPACK, which exhausts memory at scale.
    assert ngs.BaseMatrix.GetDefaultInverseType() != "sparsecholesky"
    handle = prepare(_laplace(built)[0], symmetry="hermitian", kind="direct")
    assert type(handle._inverse).__name__.startswith("SparseCholesky")


def test_complex_symmetric_operator_and_complex_rhs(built):
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct")
    free = np.array(list(space.FreeDofs()))
    rng = np.random.default_rng(1)
    B = rng.standard_normal(space.ndof) + 1j * rng.standard_normal(space.ndof)
    X = handle.solve(B)
    A = _dense(handle, space.ndof)
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    np.testing.assert_allclose((A @ X)[free], B[free], atol=1e-10)


def test_real_operator_solves_complex_rhs_by_parts(built):
    op, space = _laplace(built)
    handle = prepare(op, symmetry="hermitian", kind="direct")
    rng = np.random.default_rng(2)
    re, im = rng.standard_normal((2, space.ndof))
    X = handle.solve(re + 1j * im)
    np.testing.assert_allclose(X, handle.solve(re) + 1j * handle.solve(im))


def test_freed_handle_refuses_to_solve(built):
    handle = prepare(_laplace(built)[0], symmetry="hermitian", kind="direct")
    n = handle.ndof
    handle.free()
    assert handle.nbytes == 0
    with pytest.raises(RuntimeError, match="freed"):
        handle.solve(np.ones(n))
    with pytest.raises(RuntimeError, match="freed"):
        handle.apply(np.ones(n))


def test_wrong_shape_raises(built):
    handle = prepare(_laplace(built)[0], symmetry="hermitian", kind="direct")
    with pytest.raises(ValueError, match="shape"):
        handle.solve(np.ones(handle.ndof + 1))


@pytest.mark.parametrize(
    ("complex_valued", "symmetry"), [(False, "general"), (True, "hermitian")]
)
def test_direct_refuses_operators_cholesky_cannot_factor(
    built, complex_valued, symmetry
):
    op, _ = _laplace(built, complex_valued=complex_valued)
    with pytest.raises(ValueError, match="A\\^T = A"):
        prepare(op, symmetry=symmetry, kind="direct")


def test_unknown_backends_raise(built):
    # the GPU solver is kind="direct" with device="cuda" (tests/unit/test_cuda.py)
    op, _ = _laplace(built)
    with pytest.raises(NotImplementedError, match="cudss"):
        prepare(op, symmetry="hermitian", kind="cudss")


# --- BDDC -------------------------------------------------------------------------


def _residual(handle, B, X, free):
    return np.abs((handle.apply(X) - B)[free]).max() / np.abs(B[free]).max()


def test_bddc_solves_many_right_hand_sides_to_the_tolerance(built):
    op, space = _laplace(built)
    handle = prepare(op, symmetry="hermitian", kind="bddc_gmres", tolerance=1e-10)
    free = np.array(list(space.FreeDofs()))
    rng = np.random.default_rng(3)
    B = rng.standard_normal((space.ndof, 3))
    X = handle.solve(B)
    assert X.shape == (space.ndof, 3)
    np.testing.assert_array_equal(X[~free], 0)
    exact = prepare(_laplace(built)[0], symmetry="hermitian", kind="direct").solve(B)
    np.testing.assert_allclose(X, exact, atol=1e-8 * np.abs(exact).max())
    assert _residual(handle, B, X, free) < 1e-8
    np.testing.assert_allclose(handle.solve(B, transpose=True), X, rtol=1e-12)
    assert handle.nbytes > 0
    assert handle.iterations[-1] < 30


def test_bddc_tolerance_sets_the_residual(built):
    # at order 2 every H1 DOF is on the wirebasket and BDDC is exact
    rng = np.random.default_rng(4)
    loose, tight = (
        prepare(
            _laplace(built, order=4)[0],
            symmetry="hermitian",
            kind="bddc_gmres",
            tolerance=tol,
        )
        for tol in (1e-2, 1e-12)
    )
    B = rng.standard_normal(loose.ndof)
    assert sum(loose.iterations) == 0
    loose.solve(B)
    tight.solve(B)
    assert loose.iterations[-1] < tight.iterations[-1]


def test_one_bddc_handle_solves_an_adjoint_at_its_own_tolerance(built):
    # the forward's preconditioner serves an adjoint solve to another tolerance
    rng = np.random.default_rng(5)
    op, space = _laplace(built, order=4)
    handle = prepare(op, symmetry="hermitian", kind="bddc_gmres", tolerance=1e-2)
    free = np.array(list(space.FreeDofs()))
    B = rng.standard_normal(handle.ndof)
    handle.solve(B)
    forward = handle.iterations[-1]
    X = handle.solve(B, tolerance=1e-12)
    assert handle.iterations[-1] > forward
    assert _residual(handle, B, X, free) < 1e-8
    handle.solve(B)  # back to the prepared tolerance
    assert handle.iterations[-1] == forward
    direct = prepare(_laplace(built, order=4)[0], symmetry="hermitian", kind="direct")
    np.testing.assert_array_equal(direct.solve(B, tolerance=1e-3), direct.solve(B))


def test_bddc_coarse_inverse_is_sparse_cholesky_never_the_default(built):
    handle = prepare(_laplace(built)[0], symmetry="hermitian", kind="bddc_gmres")
    flags = handle._preconditioner.flags.ToDict()
    assert flags["coarsetype"] == "direct"
    assert flags["inverse"] == "sparsecholesky"


def test_bddc_complex_symmetric_operator_uses_gmres(built):
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry="complex_symmetric", kind="bddc_gmres")
    free = np.array(list(space.FreeDofs()))
    rng = np.random.default_rng(5)
    B = rng.standard_normal(space.ndof) + 1j * rng.standard_normal(space.ndof)
    X = handle.solve(B)
    assert _residual(handle, B, X, free) < 1e-5
    np.testing.assert_allclose(handle.solve(B, transpose=True), X, rtol=1e-12)


def test_bddc_needs_the_fixed_dofs_marked_in_the_space(built):
    space = ngs.H1(built.mesh, order=2)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += ngs.grad(u) * ngs.grad(v) * quadrature.dx(2)
    free = space.FreeDofs() & ~space.GetDofs(built.mesh.Boundaries("top"))
    with pytest.raises(ValueError, match="Dirichlet"):
        prepare(Operator(form, free), symmetry="hermitian", kind="bddc_gmres")


def test_bddc_raises_when_it_does_not_converge(built):
    handle = prepare(
        _laplace(built, order=4)[0],
        symmetry="hermitian",
        kind="bddc_gmres",
        tolerance=1e-14,
        maxiter=2,
    )
    with pytest.raises(RuntimeError, match="converge"):
        handle.solve(np.random.default_rng(6).standard_normal(handle.ndof))


def test_freed_bddc_handle_refuses_to_solve(built):
    handle = prepare(_laplace(built)[0], symmetry="hermitian", kind="bddc_gmres")
    handle.free()
    assert handle.nbytes == 0
    with pytest.raises(RuntimeError, match="freed"):
        handle.solve(np.ones(handle.ndof))


# --- SciPy matrices ---------------------------------------------------------------


def test_direct_factorizes_a_scipy_spd_matrix():
    import scipy.sparse as sp

    n = 50
    A = sp.diags([-np.ones(n - 1), 2.5 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    handle = factorize(A)
    rng = np.random.default_rng(7)
    b = rng.standard_normal(n)
    np.testing.assert_allclose(A @ handle.solve(b), b, atol=1e-12)
    B = rng.standard_normal((n, 2))
    np.testing.assert_allclose(A @ handle.solve(B), B, atol=1e-12)
    assert type(handle._inverse).__name__.startswith("SparseCholesky")
    with pytest.raises(ValueError, match="square"):
        factorize(sp.eye(3, 4))
