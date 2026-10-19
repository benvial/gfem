import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from gfem._linalg.superlu import backward_error, shift_invert_eigs


def _pencil(n, dtype, seed=0):
    rng = np.random.default_rng(seed)
    a = sp.random(n, n, density=0.2, random_state=rng) + sp.eye(n) * 3
    b = sp.random(n, n, density=0.2, random_state=rng) + sp.eye(n) * 2
    if dtype is complex:
        a = a + 1j * sp.random(n, n, density=0.1, random_state=rng)
    return sp.csr_matrix(a), sp.csr_matrix(b)


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("sigma", [0.3, -1.0 + 0.5j])
def test_finds_the_eigenvalues_nearest_the_shift(dtype, sigma):
    a, b = _pencil(40, dtype)
    lam, vecs = shift_invert_eigs(a, b, sigma, 4)
    ref = sla.eigvals(a.toarray(), b.toarray())
    nearest = ref[np.argsort(np.abs(ref - sigma))[:4]]
    assert len(lam) == 4
    for value in nearest:
        assert np.abs(lam - value).min() < 1e-9 * abs(value)
    for i in range(len(lam)):
        assert backward_error(a, b, lam[i], vecs[:, i]) < 1e-10


def test_an_indefinite_shifted_operator_is_factored():
    # A - sigma B indefinite (eigenvalues on both sides of sigma)
    a = sp.diags(np.arange(1.0, 21.0)).tocsr()
    b = sp.eye(20, format="csr")
    lam, _ = shift_invert_eigs(a, b, 10.4, 2)
    np.testing.assert_allclose(np.sort(lam.real), [10.0, 11.0])


def test_k_is_capped_by_the_problem_size():
    a = sp.diags(np.arange(1.0, 6.0)).tocsr()
    lam, vecs = shift_invert_eigs(a, sp.eye(5, format="csr"), 0.0, 50)
    assert len(lam) == 3
    assert vecs.shape == (5, 3)


def test_backward_error_of_an_exact_pair_is_zero():
    a = sp.diags([1.0, 2.0]).tocsr()
    b = sp.eye(2, format="csr")
    assert backward_error(a, b, 2.0, np.array([0.0, 1.0])) == 0.0
    assert backward_error(a, b, 1.0, np.array([0.0, 1.0])) > 0.1
