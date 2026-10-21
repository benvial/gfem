"""The solver-handle seam: prepare once, solve many right-hand sides, free."""

import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import quadrature
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import Operator, factorize, low_rank_update, prepare


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


def test_direct_refactorization_solves_like_a_fresh_handle(built):
    # one sweep, two frequencies: the second refactorizes the first's plan
    handle = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j)[0],
        symmetry="complex_symmetric",
        kind="direct",
    )
    op, space = _laplace(built, complex_valued=True, shift=3 + 1j)
    handle.refactor(op)
    fresh = prepare(
        _laplace(built, complex_valued=True, shift=3 + 1j)[0],
        symmetry="complex_symmetric",
        kind="direct",
    )
    rng = np.random.default_rng(12)
    B = rng.standard_normal((space.ndof, 2)) + 1j * rng.standard_normal((space.ndof, 2))
    X, ref = handle.solve(B), fresh.solve(B)
    assert np.linalg.norm(X - ref) / np.linalg.norm(ref) < 1e-12
    np.testing.assert_allclose(handle.apply(X), fresh.apply(X), rtol=1e-14)


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


def test_gmres_initial_guess_saves_iterations_at_the_same_tolerance(built):
    # GMRES stops at tolerance * |P b| whatever the start, so a good initial
    # guess (the previous frequency's solution) saves iterations.
    def handle_at(shift):
        op, _ = _laplace(built, complex_valued=True, shift=shift, order=4)
        return prepare(
            op, symmetry="complex_symmetric", kind="bddc_gmres", tolerance=1e-8
        )

    first, second = handle_at(1 - 2j), handle_at(1.1 - 2j)
    free = second._free
    rng = np.random.default_rng(8)
    B = rng.standard_normal((first.ndof, 2)) + 0j
    previous = first.solve(B)
    cold = second.solve(B)
    warm = second.solve(B, x0=previous)
    assert all(
        w < c for w, c in zip(second.iterations[2:], second.iterations[:2], strict=True)
    )
    np.testing.assert_allclose(warm, cold, atol=1e-6 * np.abs(cold).max())
    cold_residual = _residual(second, B[:, 0], cold[:, 0], free)
    assert _residual(second, B[:, 0], warm[:, 0], free) < 10 * cold_residual
    np.testing.assert_array_equal(warm[~free], 0)
    # a guess worse than zero is dropped; the exact solution needs no iteration
    second.solve(B[:, 0], x0=1e6 * np.ones(second.ndof))
    assert second.iterations[-1] <= second.iterations[0]
    second.solve(B[:, 0], x0=warm[:, 0])
    assert second.iterations[-1] <= 1


def test_bddc_refactorization_without_a_guess_solves_like_a_fresh_handle(built):
    # the preconditioner is kept for warm starts only: a cold solve after a
    # refactorization sets BDDC up again at the new operator
    def operator(shift):
        return _laplace(built, complex_valued=True, shift=shift, order=4)[0]

    handle, fresh = (
        prepare(operator(s), symmetry="complex_symmetric", kind="bddc_gmres")
        for s in (1 - 2j, -40 + 1j)
    )
    handle.refactor(operator(-40 + 1j))
    rng = np.random.default_rng(14)
    B = rng.standard_normal((handle.ndof, 2)) + 0j
    X, ref = handle.solve(B), fresh.solve(B)
    assert handle.iterations == fresh.iterations
    np.testing.assert_allclose(X, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


def test_direct_ignores_the_initial_guess(built):
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct")
    b = np.ones(space.ndof, complex)
    np.testing.assert_array_equal(
        handle.solve(b, x0=np.ones(space.ndof)), handle.solve(b)
    )


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
    # the direct backend's factorization (tests/unit/test_linalg_mkl.py)
    assert type(handle._inverse).__name__.startswith(("SparseCholesky", "MKLPardiso"))
    with pytest.raises(ValueError, match="square"):
        factorize(sp.eye(3, 4))


# --- low-rank updates -------------------------------------------------------------


def _update(space, rng, p=2):
    free = np.array(list(space.FreeDofs()))
    U = rng.standard_normal((space.ndof, p)) + 1j * rng.standard_normal((space.ndof, p))
    U[~free] = 0  # an update lives on the free DOFs
    C = rng.standard_normal((p, p)) + 1j * rng.standard_normal((p, p))
    return U, C, free


@pytest.mark.parametrize("kind", ["direct", "bddc_gmres"])
def test_a_low_rank_update_solves_the_updated_operator(built, kind):
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    rng = np.random.default_rng(9)
    U, C, free = _update(space, rng)
    C = C + C.T  # complex symmetric, as a wave port's correction
    base = prepare(op, symmetry="complex_symmetric", kind=kind, tolerance=1e-10)
    handle = low_rank_update(base, U, C)
    B = rng.standard_normal((space.ndof, 3)) + 0j
    X = handle.solve(B)
    # Krylov iterations count the right-hand sides, not the update's columns
    assert len(handle.iterations) == (3 if kind == "bddc_gmres" else 0)
    A = _dense(base, space.ndof) + U @ C @ U.T
    np.testing.assert_allclose(handle.apply(X), A @ X, atol=1e-10 * np.abs(A @ X).max())
    np.testing.assert_allclose((A @ X)[free], B[free], atol=1e-8 * np.abs(B).max())
    np.testing.assert_array_equal(X[~free], 0)
    np.testing.assert_allclose(
        handle.solve(B, transpose=True), X, atol=1e-8 * np.abs(X).max()
    )
    np.testing.assert_allclose(handle.solve(B[:, 0]), X[:, 0], rtol=1e-8)
    assert handle.kind == kind
    assert handle.ndof == base.ndof
    assert handle.nbytes >= base.nbytes
    if kind == "bddc_gmres":
        x0 = handle.solve(B[:, :2], x0=X[:, :2])
        np.testing.assert_allclose(x0, X[:, :2], atol=1e-8 * np.abs(X).max())
    handle.free()
    with pytest.raises(RuntimeError, match="freed"):
        handle.solve(B)


def test_a_low_rank_update_transposes_its_coefficients(built):
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    rng = np.random.default_rng(10)
    U, C, free = _update(space, rng, p=3)  # C not symmetric
    handle = low_rank_update(
        prepare(op, symmetry="complex_symmetric", kind="direct"), U, C
    )
    A = _dense(handle, space.ndof)  # apply includes the update
    b = rng.standard_normal(space.ndof) + 1j * rng.standard_normal(space.ndof)
    x = handle.solve(b, transpose=True)
    np.testing.assert_allclose((A.T @ x)[free], b[free], atol=1e-9 * np.abs(b).max())
    x = handle.solve(b)
    np.testing.assert_allclose((A @ x)[free], b[free], atol=1e-9 * np.abs(b).max())


def test_a_zero_update_is_the_operator_itself(built):
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    base = prepare(op, symmetry="complex_symmetric", kind="direct")
    rng = np.random.default_rng(11)
    U, _, _ = _update(space, rng)
    b = rng.standard_normal(space.ndof) + 0j
    x = low_rank_update(base, U, np.zeros((2, 2))).solve(b)
    np.testing.assert_allclose(x, base.solve(b), atol=1e-12 * np.abs(x).max())


def test_a_borrowing_low_rank_update_frees_only_its_own_columns(built):
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    rng = np.random.default_rng(13)
    U, C, _ = _update(space, rng)
    base = prepare(op, symmetry="complex_symmetric", kind="direct")
    b = rng.standard_normal(space.ndof) + 0j
    x = base.solve(b)
    handle = low_rank_update(base, U, C, borrow=True)
    handle.solve(b)
    handle.free()
    assert handle.nbytes == 0
    with pytest.raises(RuntimeError, match="freed"):
        handle.solve(b)
    np.testing.assert_array_equal(base.solve(b), x)  # the base is still usable
