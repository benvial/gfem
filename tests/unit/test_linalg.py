"""The solver-handle seam: prepare once, solve many right-hand sides, free."""

import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import quadrature
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import Operator, prepare


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 2)))
    return to_ngsolve(gfem.Mesh.from_gmsh(path))


def _laplace(built, complex_valued=False, shift=0.0):
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=complex_valued)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += (ngs.grad(u) * ngs.grad(v) + shift * u * v) * quadrature.dx(2)
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


def test_unavailable_backends_and_devices_raise(built):
    op, _ = _laplace(built)
    with pytest.raises(NotImplementedError, match="bddc_gmres"):
        prepare(op, symmetry="hermitian", kind="bddc_gmres")
    with pytest.raises(NotImplementedError, match="device"):
        prepare(op, symmetry="hermitian", kind="direct", device="cuda:0")
