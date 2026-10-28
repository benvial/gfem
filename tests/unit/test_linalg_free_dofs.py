"""Free-DOF sets as NumPy masks and their comparison, without a per-DOF loop."""

import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine.mesh import to_ngsolve
from gfem._linalg.base import free_mask, same_free_dofs

SIZES = (0, 1, 7, 8, 9, 63, 64, 65, 1001)


def _bits(mask) -> ngs.BitArray:
    bits = ngs.BitArray(len(mask))
    bits.Clear()
    for i in np.flatnonzero(mask):
        bits.Set(int(i))
    return bits


def _reference(bits) -> np.ndarray:
    """The per-DOF conversion the mask replaces, as the source of truth."""
    return np.array(list(bits), dtype=bool)  # ty: ignore[invalid-argument-type]


def _masks(n):
    rng = np.random.default_rng(n)
    yield "none", np.zeros(n, dtype=bool)
    yield "random", rng.random(n) < 0.5
    yield "all", np.ones(n, dtype=bool)


@pytest.fixture(scope="module")
def constrained(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 2)))
    mesh = to_ngsolve(gfem.Mesh.from_gmsh(path)).mesh
    return ngs.HCurl(mesh, order=2, complex=True, dirichlet="bottom|top").FreeDofs()


@pytest.mark.parametrize("n", SIZES)
def test_the_free_mask_is_the_bitarray_bit_for_bit(n):
    for name, mask in _masks(n):
        bits = _bits(mask)
        got = free_mask(bits, n)
        assert got.dtype == bool, name
        np.testing.assert_array_equal(got, _reference(bits), err_msg=name)


def test_the_free_mask_of_a_constrained_space_is_its_bitarray(constrained):
    expected = _reference(constrained)
    assert 0 < expected.sum() < len(expected)
    np.testing.assert_array_equal(free_mask(constrained, len(constrained)), expected)


def test_no_free_dof_set_means_every_dof_is_free():
    np.testing.assert_array_equal(free_mask(None, 5), np.ones(5, dtype=bool))


def test_a_free_dof_set_is_the_same_as_itself_and_its_copy(constrained):
    assert same_free_dofs(constrained, constrained)
    assert same_free_dofs(constrained, ngs.BitArray(constrained))


def test_one_flipped_dof_makes_the_free_dof_sets_differ(constrained):
    for i in (0, len(constrained) // 2, len(constrained) - 1):
        flipped = ngs.BitArray(constrained)
        if flipped[i]:
            flipped.Clear(i)
        else:
            flipped.Set(i)
        assert not same_free_dofs(constrained, flipped), i
        assert not same_free_dofs(flipped, constrained), i


def test_free_dof_sets_of_different_lengths_differ():
    assert not same_free_dofs(_bits(np.ones(8, bool)), _bits(np.ones(9, bool)))
    assert not same_free_dofs(_bits(np.zeros(9, bool)), _bits(np.zeros(8, bool)))
