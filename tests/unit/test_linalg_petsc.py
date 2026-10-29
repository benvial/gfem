"""The PETSc linear solver: PETSc KSP / PC behind the solver-handle seam.

By default an exact MUMPS LDL^T factorization (``ksp_type preonly``,
``pc_type cholesky``); ``petsc_options`` configures any other KSP / PC.
Every test needs complex petsc4py: skipped without it (the ``petsc`` marker).
"""

import dataclasses
import re

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm
from petsc_recipe import KSP_VIEW, MUMPS_LU

import gfem
from gfem import _petsc
from gfem._engine import quadrature
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import Operator, prepare
from gfem._linalg.petsc import PetscHandle
from gfem._linalg.pieces import AssembledPieces, PieceSum

pytestmark = pytest.mark.petsc


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


def _free(space):
    return np.array(list(space.FreeDofs()))


def test_the_petsc_stack_is_complex_with_mumps_and_no_non_free_package():
    # The licence control (no non-free or non-commercial dependency) for a
    # stack the licence gate cannot see (MUMPS and METIS are compiled into
    # the petsc wheel): what tools/petsc_wheelhouse.sh builds, and none of
    # the packages an MPI build pulls in (ParMETIS is not free for
    # commercial use).
    PETSc = _petsc.require_petsc()
    assert np.dtype(PETSc.ScalarType) == np.complex128
    assert PETSc.Sys.hasExternalPackage("mumps")
    assert PETSc.Sys.hasExternalPackage("metis")
    for package in ("parmetis", "ptscotch", "scalapack"):
        assert not PETSc.Sys.hasExternalPackage(package), package


def test_petsc_solves_many_right_hand_sides_on_the_free_dofs(built):
    op, space = _laplace(built)
    handle = prepare(op, symmetry="hermitian", kind="petsc")
    assert handle.kind == "petsc"
    free = _free(space)
    rng = np.random.default_rng(0)
    B = rng.standard_normal((space.ndof, 3))
    X = handle.solve(B)
    assert X.shape == (space.ndof, 3)
    assert X.dtype == np.float64  # a real operator gives real solutions
    np.testing.assert_array_equal(X[~free], 0)
    A = handle.apply(np.eye(space.ndof))
    np.testing.assert_allclose((A @ X)[free], B[free], atol=1e-10 * np.abs(B).max())
    # one column at a time (a single solve, not a block): the same factors
    np.testing.assert_allclose(handle.solve(B[:, 1]), X[:, 1], rtol=1e-12)
    np.testing.assert_allclose(handle.solve(B, transpose=True), X, rtol=1e-12)


def test_petsc_solves_a_complex_symmetric_operator(built):
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry="complex_symmetric", kind="petsc")
    free = _free(space)
    rng = np.random.default_rng(1)
    B = rng.standard_normal((space.ndof, 2)) + 1j * rng.standard_normal((space.ndof, 2))
    X = handle.solve(B)
    A = handle.apply(np.eye(space.ndof))
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    np.testing.assert_allclose((A @ X)[free], B[free], atol=1e-10 * np.abs(B).max())
    np.testing.assert_allclose(handle.solve(B, transpose=True), X, rtol=1e-12)
    direct = prepare(
        _laplace(built, True, 1 - 2j)[0], symmetry="complex_symmetric", kind="direct"
    )
    np.testing.assert_allclose(X, direct.solve(B), rtol=1e-10)


def test_a_real_operator_solves_a_complex_rhs_by_parts(built):
    op, space = _laplace(built)
    handle = prepare(op, symmetry="hermitian", kind="petsc")
    rng = np.random.default_rng(2)
    real, imag = rng.standard_normal((2, space.ndof))
    X = handle.solve(real + 1j * imag)
    np.testing.assert_allclose(X, handle.solve(real) + 1j * handle.solve(imag))


def test_without_options_petsc_is_an_exact_mumps_cholesky(built):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry="complex_symmetric", kind="petsc")
    ksp = handle._ksp
    pc = ksp.getPC()
    assert (ksp.getType(), pc.getType()) == ("preonly", "cholesky")
    assert pc.getFactorSolverType() == "mumps"
    A, _ = ksp.getOperators()
    assert A.isSymmetricKnown() == (True, True)
    assert not A.isHermitianKnown()[1]
    assert handle.iterations == []


def _gfem_keys():
    PETSc = _petsc.require_petsc()
    return [k for k in PETSc.Options().getAll() if k.startswith("gfem")]


def test_options_live_under_the_handles_prefix_only_while_it_reads_them(built):
    PETSc = _petsc.require_petsc()
    options = PETSc.Options()
    # a user's own, unprefixed options never reach gfem's KSP
    options.setValue("ksp_type", "gmres")
    options.setValue("pc_type", "jacobi")
    try:
        op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
        handle = prepare(op, symmetry="complex_symmetric", kind="petsc")
        assert re.fullmatch(r"gfem\d+_", handle._ksp.getOptionsPrefix())
        assert handle._ksp.getType() == "preonly"
        assert _gfem_keys() == []
        handle.solve(np.ones(space.ndof, complex))
        handle.refactorize(_laplace(built, complex_valued=True, shift=2 + 1j)[0])
        assert _gfem_keys() == []
        handle.free()
        assert _gfem_keys() == []
    finally:
        options.delValue("ksp_type")
        options.delValue("pc_type")


def _pieces(built, integrands, weights):
    """Complex H1 pieces as a form and as a piece sum (`Operator.piece_sum`)."""
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    for w, integrand in zip(weights, integrands, strict=True):
        form += w * integrand(u, v)
    pieces = AssembledPieces(space, integrands)
    return Operator(form, space.FreeDofs(), PieceSum(pieces, weights)), pieces


PIECES = (
    lambda u, v: ngs.grad(u) * ngs.grad(v) * quadrature.dx(2),
    lambda u, v: u * v * quadrature.dx(2),
    lambda u, v: u * v * ngs.ds("side"),
)


def _agree(handle, fresh, seed):
    n = handle.ndof
    A = fresh.apply(np.eye(n))
    assert np.abs(handle.apply(np.eye(n)) - A).max() <= 1e-14 * np.abs(A).max()
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))
    X, ref = handle.solve(B), fresh.solve(B)
    assert np.linalg.norm(X - ref) / np.linalg.norm(ref) < 1e-12


def test_a_refactorization_solves_like_a_fresh_handle(built):
    # a sweep's second frequency: new values on the first one's solver plan
    handle = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j)[0],
        symmetry="complex_symmetric",
        kind="petsc",
    )
    ksp = handle._ksp
    handle.refactorize(_laplace(built, complex_valued=True, shift=3 + 1j)[0])
    fresh = prepare(
        _laplace(built, complex_valued=True, shift=3 + 1j)[0],
        symmetry="complex_symmetric",
        kind="petsc",
    )
    assert handle._ksp is ksp
    _agree(handle, fresh, 12)


def test_a_refactorization_sums_the_handles_own_pieces_in_place(built):
    op, pieces = _pieces(built, PIECES, (1.0, 2 - 1j, 3j))
    handle = prepare(op, symmetry="complex_symmetric", kind="petsc")
    with pytest.raises(Exception, match="assemble"):  # the form is not assembled
        _ = op.form.mat
    matrix = handle._matrix
    weights = (1.0, -1 + 1j, 0.5)
    op, _ = _pieces(built, PIECES, weights)
    handle.refactorize(dataclasses.replace(op, piece_sum=PieceSum(pieces, weights)))
    assert handle._matrix is matrix  # summed into the held matrix
    fresh = prepare(
        dataclasses.replace(op, piece_sum=None),
        symmetry="complex_symmetric",
        kind="petsc",
    )
    _agree(handle, fresh, 3)


def _on_the_side(built, volume):
    """An operator on the side's DOFs, with or without a volume term.

    On those DOFs a side mass term alone is definite; without the volume
    term its matrix couples fewer DOFs: another sparsity pattern.
    """
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()
    free = space.FreeDofs() & space.GetDofs(built.mesh.Boundaries("side"))
    form = ngs.BilinearForm(space, symmetric=True)
    form += (2 - 1j) * u * v * ngs.ds("side")
    if volume:
        form += ngs.grad(u) * ngs.grad(v) * quadrature.dx(2)
    return Operator(form, free)


def test_a_refactorization_on_another_pattern_plans_again(built):
    handle = prepare(
        _on_the_side(built, True), symmetry="complex_symmetric", kind="petsc"
    )
    nonzeros = handle._values.size
    handle.refactorize(_on_the_side(built, False))
    assert handle._values.size < nonzeros
    fresh = prepare(
        _on_the_side(built, False), symmetry="complex_symmetric", kind="petsc"
    )
    _agree(handle, fresh, 4)


def _factorizations():
    PETSc = _petsc.require_petsc()
    PETSc.Log.begin()
    return lambda: {
        name: PETSc.Log.Event(name).getPerfInfo()["count"]
        for name in ("MatCholFctrSym", "MatCholFctrNum")
    }


def test_a_same_pattern_refactorization_is_numeric_only(built):
    counts = _factorizations()
    handle = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j)[0],
        symmetry="complex_symmetric",
        kind="petsc",
    )
    before = counts()
    handle.refactorize(_laplace(built, complex_valued=True, shift=3 + 1j)[0])
    after = counts()
    assert after["MatCholFctrSym"] == before["MatCholFctrSym"]
    assert after["MatCholFctrNum"] == before["MatCholFctrNum"] + 1


def test_a_freed_handle_refuses_to_solve_and_holds_nothing(built):
    handle = prepare(_laplace(built)[0], symmetry="hermitian", kind="petsc")
    n = handle.ndof
    assert handle.nbytes > 0
    handle.free()
    handle.free()  # idempotent
    assert handle.nbytes == 0
    with pytest.raises(RuntimeError, match="freed"):
        handle.solve(np.ones(n))
    with pytest.raises(RuntimeError, match="freed"):
        handle.apply(np.ones(n))
    with pytest.raises(RuntimeError, match="This solver handle was freed"):
        handle.refactorize(_laplace(built)[0])


@pytest.mark.parametrize("summed", [True, False])
def test_a_handle_freed_during_refactorization_raises_freed(built, summed):
    # another thread frees the handle while refactorize reads the piece sum
    op, _ = _pieces(built, PIECES, (1.0, 2 - 1j, 3j))
    handle = prepare(op, symmetry="complex_symmetric", kind="petsc")

    class FreedOnRead:
        def __init__(self, piece_sum):
            self._piece_sum = piece_sum

        def __getattr__(self, name):
            handle.free()
            return getattr(self._piece_sum, name)

    # the handle's pieces are summed into its matrix; others are summed anew
    other = op if summed else _pieces(built, PIECES, (1.0, 2 - 1j, 3j))[0]
    piece_sum = FreedOnRead(other.piece_sum)
    with pytest.raises(RuntimeError, match="This solver handle was freed"):
        handle.refactorize(dataclasses.replace(op, piece_sum=piece_sum))


def test_nbytes_counts_the_mumps_factorization(built):
    # MUMPS reports its memory in millions of bytes: a larger operator
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j, order=5)
    exact = prepare(op, symmetry="complex_symmetric", kind="petsc")
    factor = exact._ksp.getPC().getFactorMatrix()
    assert factor.getMumpsInfog(22) > 0
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j, order=5)
    options = (("ksp_type", "gmres"), ("pc_type", "jacobi"))
    krylov = prepare(
        op, symmetry="complex_symmetric", kind="petsc", petsc_options=options
    )
    assert exact.nbytes - krylov.nbytes == 10**6 * factor.getMumpsInfog(22)


# --- Krylov configurations --------------------------------------------------------

GMRES = (("ksp_type", "gmres"), ("pc_type", "jacobi"))


def _krylov(built, shift=1 - 2j, tolerance=1e-8, options=GMRES, **kwargs):
    op, _ = _laplace(built, complex_valued=True, shift=shift, order=4)
    return prepare(
        op,
        symmetry="complex_symmetric",
        kind="petsc",
        tolerance=tolerance,
        petsc_options=options,
        **kwargs,
    )


def _relative_residual(handle, B, X):
    free = handle._free
    return np.linalg.norm((handle.apply(X) - B)[free]) / np.linalg.norm(B[free])


def test_a_krylov_solve_stops_at_the_true_relative_residual(built):
    handle = _krylov(built)
    ksp = handle._ksp
    PETSc = _petsc.require_petsc()
    # the unpreconditioned norm, so the preconditioner goes on the right
    assert ksp.getNormType() == PETSc.KSP.NormType.UNPRECONDITIONED
    assert ksp.getPCSide() == PETSc.PC.Side.RIGHT
    rng = np.random.default_rng(13)
    B = rng.standard_normal((handle.ndof, 2)) + 0j
    X = handle.solve(B)
    assert len(handle.iterations) == 2
    assert all(n > 1 for n in handle.iterations)
    for j in range(2):
        assert _relative_residual(handle, B[:, j], X[:, j]) <= 1e-8
    exact = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j, order=4)[0],
        symmetry="complex_symmetric",
        kind="direct",
    ).solve(B)
    np.testing.assert_allclose(X, exact, atol=1e-6 * np.abs(exact).max())
    np.testing.assert_array_equal(X[~handle._free], 0)


def test_a_user_norm_type_is_kept(built):
    PETSc = _petsc.require_petsc()
    handle = _krylov(built, options=(*GMRES, ("ksp_norm_type", "preconditioned")))
    assert handle._ksp.getNormType() == PETSc.KSP.NormType.PRECONDITIONED


def test_a_krylov_solve_at_another_tolerance_restores_the_prepared_one(built):
    handle = _krylov(built, tolerance=1e-3)
    b = np.random.default_rng(14).standard_normal(handle.ndof) + 0j
    handle.solve(b)
    loose = handle.iterations[-1]
    x = handle.solve(b, tolerance=1e-10)
    assert handle.iterations[-1] > loose
    assert _relative_residual(handle, b, x) <= 1e-10
    handle.solve(b)
    assert handle.iterations[-1] == loose


def test_a_krylov_initial_guess_saves_iterations(built):
    handle = _krylov(built)
    b = np.random.default_rng(15).standard_normal(handle.ndof) + 0j
    cold = handle.solve(b)
    warm = handle.solve(b, x0=cold * (1 + 1e-6))
    assert handle.iterations[1] < handle.iterations[0]
    np.testing.assert_allclose(warm, cold, atol=1e-6 * np.abs(cold).max())
    handle.solve(b)  # no guess: from zero again
    assert handle.iterations[2] == handle.iterations[0]


def test_a_krylov_solve_that_does_not_converge_raises(built):
    handle = _krylov(built, tolerance=1e-14, maxiter=3)
    with pytest.raises(
        RuntimeError, match=r"PETSc gmres \+ jacobi did not converge"
    ) as err:
        handle.solve(np.ones(handle.ndof, complex))
    assert "DIVERGED_MAX_IT" in str(err.value)
    assert "1e-14 in 3 iterations" in str(err.value)


def test_a_krylov_solve_that_does_not_converge_reports_four_digits(built):
    # at two digits a stall just above the tolerance can print as the
    # tolerance itself, hiding how far off it is
    handle = _krylov(built, tolerance=1e-14, maxiter=10)
    b = np.ones(handle.ndof, complex)
    with pytest.raises(RuntimeError, match="DIVERGED_MAX_IT") as err:
        handle.solve(b)
    stalled = handle._ksp.getResidualNorm() / np.linalg.norm(b[handle._free])
    reached = re.search(r"reached (\S+)\)", str(err.value))
    assert reached is not None
    assert float(reached.group(1)) == pytest.approx(stalled, rel=1e-3)


def test_cg_on_a_complex_symmetric_operator_points_to_gmres(built):
    handle = _krylov(
        built, shift=-40 + 1j, options=(("ksp_type", "cg"), ("pc_type", "none"))
    )
    b = np.random.default_rng(16).standard_normal(handle.ndof) + 0j
    with pytest.raises(RuntimeError, match="DIVERGED_INDEFINITE_MAT") as err:
        handle.solve(b)
    assert "use ksp_type gmres" in str(err.value)


def _pc_setups():
    PETSc = _petsc.require_petsc()
    PETSc.Log.begin()
    return lambda: PETSc.Log.Event("PCSetUp").getPerfInfo()["count"]


@pytest.mark.parametrize("reuse", [True, False])
def test_reuse_preconditioner_false_overrides_petscs_reuse(built, reuse):
    # the adaptive sweep prepares with reuse_preconditioner=False
    options = (*GMRES, ("ksp_reuse_preconditioner", True))
    handle = _krylov(built, options=options, reuse_preconditioner=reuse)
    b = np.ones(handle.ndof, complex)
    handle.solve(b)
    setups = _pc_setups()
    before = setups()
    op, _ = _laplace(built, complex_valued=True, shift=1.1 - 2j, order=4)
    handle.refactorize(op)
    handle.solve(b)
    assert setups() - before == (0 if reuse else 1)


def test_preonly_refuses_petscs_preconditioner_reuse(built):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    with pytest.raises(
        ValueError, match=r"ksp_reuse_preconditioner.*previous operator"
    ):
        prepare(
            op,
            symmetry="complex_symmetric",
            kind="petsc",
            petsc_options=(("ksp_reuse_preconditioner", None),),
        )
    assert _gfem_keys() == []


# --- what PETSc and gfem refuse ---------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        (("ksp_tpye", "gmres"),),  # a typo
        (("mat_mumps_icntl_14", 50), ("pc_type", "jacobi")),  # not MUMPS's PC
    ],
)
def test_an_option_petsc_did_not_read_raises(built, options):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    unread = options[0][0]
    with pytest.raises(ValueError, match=rf"\['{unread}'\] was not read") as err:
        prepare(op, symmetry="complex_symmetric", kind="petsc", petsc_options=options)
    assert "check the name" in str(err.value)
    assert _gfem_keys() == []


def test_an_option_petsc_reads_is_honoured(built):
    handle = prepare(
        _on_the_side(built, True),
        symmetry="complex_symmetric",
        kind="petsc",
        petsc_options=(("mat_mumps_icntl_14", 55), ("pc_type", "lu")),
    )
    pc = handle._ksp.getPC()
    assert (pc.getType(), pc.getFactorSolverType()) == ("lu", "mumps")
    assert pc.getFactorMatrix().getMumpsIcntl(14) == 55
    # read again by a new pattern's MUMPS instance
    handle.refactorize(_on_the_side(built, False))
    assert handle._ksp.getPC().getFactorMatrix().getMumpsIcntl(14) == 55


def _recipe(built, options):
    """A complex symmetric operator, solved with one of the recipe's dicts."""
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    pairs = tuple(options.items())
    handle = prepare(
        op, symmetry="complex_symmetric", kind="petsc", petsc_options=pairs
    )
    direct = prepare(
        _laplace(built, True, 1 - 2j)[0], symmetry="complex_symmetric", kind="direct"
    )
    rng = np.random.default_rng(3)
    b = rng.standard_normal(space.ndof) + 1j * rng.standard_normal(space.ndof)
    return handle, handle.solve(b), direct.solve(b)


def test_the_recipes_mumps_lu_options_give_mumps_more_workspace(built):
    handle, x, direct = _recipe(built, MUMPS_LU)
    pc = handle._ksp.getPC()
    assert (pc.getType(), pc.getFactorSolverType()) == ("lu", "mumps")
    assert pc.getFactorMatrix().getMumpsIcntl(14) == 50
    np.testing.assert_allclose(x, direct, rtol=1e-10)


def test_the_recipes_ksp_view_prints_the_solver(built, capfd):
    capfd.readouterr()
    _, x, direct = _recipe(built, KSP_VIEW)
    printed = capfd.readouterr().out
    assert "type: preonly" in printed
    assert "MUMPS" in printed
    np.testing.assert_allclose(x, direct, rtol=1e-10)


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ((("ksp_type", "gmress"),), "Unable to find requested KSP type gmress"),
        ((*GMRES, ("ksp_max_it", "many")), "PETSc error 63"),
    ],
)
def test_options_petsc_refuses_raise_a_value_error(built, options, reason):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    with pytest.raises(ValueError, match="refused by PETSc") as err:
        prepare(op, symmetry="complex_symmetric", kind="petsc", petsc_options=options)
    assert reason in str(err.value)
    assert _gfem_keys() == []


def test_an_error_in_petscs_set_up_raises_a_runtime_error(built):
    # the recipe's MUMPS has no OpenMP
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    with pytest.raises(RuntimeError, match=r"PETSc failed in KSPSetUp \(KSP preonly"):
        prepare(
            op,
            symmetry="complex_symmetric",
            kind="petsc",
            petsc_options=(("mat_mumps_use_omp_threads", 2),),
        )
    assert _gfem_keys() == []


def test_a_singular_operator_raises_a_factorization_error(built):
    # a side mass term alone: the rows of the inner DOFs are zero
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += u * v * ngs.ds("side")
    with pytest.raises(RuntimeError, match="factorization failed") as err:
        prepare(
            Operator(form, space.FreeDofs()), symmetry="complex_symmetric", kind="petsc"
        )
    assert "MUMPS INFOG(1)=" in str(err.value)
    assert "mat_mumps_icntl_14" in str(err.value)


def _aijs(monkeypatch, error=None) -> list:
    """The AIJs PETSc creates from now on; or every creation raises ``error``."""
    PETSc = _petsc.require_petsc()
    made = []

    class Mat(PETSc.Mat):
        def createAIJWithArrays(self, *args, **kwargs):  # noqa: N802 - PETSc name
            if error is not None:
                raise error
            made.append(self)
            return super().createAIJWithArrays(*args, **kwargs)

    monkeypatch.setattr(PETSc, "Mat", Mat)
    return made


class CreationError(Exception):
    """Raised by a PETSc object creation."""


def test_a_failed_ksp_creation_destroys_the_aij(built, monkeypatch):
    PETSc = _petsc.require_petsc()
    made = _aijs(monkeypatch)

    class KSP(PETSc.KSP):
        def create(self, comm=None):  # noqa: ARG002 - PETSc signature
            raise CreationError("no KSP")

    monkeypatch.setattr(PETSc, "KSP", KSP)
    with pytest.raises(CreationError, match="no KSP"):
        prepare(_laplace(built)[0], symmetry="hermitian", kind="petsc")
    assert len(made) == 1
    assert not made[0]  # destroyed: its handle is null


def test_a_failed_replan_keeps_the_handle_on_its_operator(built, monkeypatch):
    handle = prepare(
        _on_the_side(built, True), symmetry="complex_symmetric", kind="petsc"
    )
    with monkeypatch.context() as patch:
        _aijs(patch, CreationError("no AIJ"))
        with pytest.raises(CreationError, match="no AIJ"):
            handle.refactorize(_on_the_side(built, False))  # another pattern
    fresh = prepare(
        _on_the_side(built, True), symmetry="complex_symmetric", kind="petsc"
    )
    _agree(handle, fresh, 5)
    # and it refactorizes on the pattern it holds
    handle.refactorize(_on_the_side(built, True))
    _agree(handle, fresh, 6)


def test_a_replan_sets_the_new_aij_up_before_destroying_the_old(built, monkeypatch):
    PETSc = _petsc.require_petsc()
    events = []

    class Mat(PETSc.Mat):
        def destroy(self):
            events.append(("destroy", id(self)))
            return super().destroy()

    class KSP(PETSc.KSP):
        def setOperators(self, A, P=None):  # noqa: N802 - PETSc name
            events.append(("setOperators", id(A)))
            return super().setOperators(A, P)

        def setUp(self):
            events.append(("setUp", None))
            return super().setUp()

    monkeypatch.setattr(PETSc, "Mat", Mat)
    monkeypatch.setattr(PETSc, "KSP", KSP)
    handle = prepare(
        _on_the_side(built, True), symmetry="complex_symmetric", kind="petsc"
    )
    old = id(handle._mat)
    events.clear()
    handle.refactorize(_on_the_side(built, False))
    new = id(handle._mat)
    assert events == [("setOperators", new), ("setUp", None), ("destroy", old)]
    handle.free()


@pytest.mark.parametrize(
    ("complex_valued", "symmetry"), [(False, "general"), (True, "hermitian")]
)
def test_petsc_refuses_operators_ldlt_cannot_factor(built, complex_valued, symmetry):
    op, _ = _laplace(built, complex_valued=complex_valued)
    with pytest.raises(ValueError, match=r"needs A\^T = A"):
        prepare(op, symmetry=symmetry, kind="petsc")
    with pytest.raises(TypeError, match="assemble"):
        _ = op.form.mat  # refused before the assembly


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"symmetric_storage": True}, "symmetric \\(triangular\\) storage"),
        ({"nonassemble": True}, "sparse matrix"),
    ],
)
def test_petsc_needs_the_full_sparse_matrix(built, flags, message):
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=True)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True, **flags)
    form += (ngs.grad(u) * ngs.grad(v) + u * v) * quadrature.dx(2)
    with pytest.raises(ValueError, match=message):
        prepare(
            Operator(form, space.FreeDofs()), symmetry="complex_symmetric", kind="petsc"
        )


# --- petsc_options reach every solve ----------------------------------------------


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"box": gfem.Material(eps_r=1.0)}
    )


def _record(monkeypatch) -> list:
    """Spy on the PETSc handles' creation: the options each one reads."""
    calls = []
    from_operator = PetscHandle.from_operator.__func__

    def spy(cls, *args, **kwargs):
        calls.append(kwargs["options"])
        return from_operator(cls, *args, **kwargs)

    monkeypatch.setattr(PetscHandle, "from_operator", classmethod(spy))
    return calls


LU_GMRES = {"ksp_type": "gmres", "pc_type": "lu"}


def test_petsc_options_reach_the_forward_and_adjoint_driven_solves(line, monkeypatch):
    calls = _record(monkeypatch)
    # p2 is not excited: its S row needs an adjoint solve
    problem = gfem.Driven(frequency=(3e9, 4e9), excite="p1")
    options = gfem.SolverOptions(linear_solver="petsc", petsc_options=LU_GMRES)
    n = len(line.mesh.tetrahedra)

    def objective(eps, options):
        model = line.with_material("dielectric", gfem.MaterialField(eps_r=eps))
        s = gfem.solve(model, problem, options=options).s.values
        return jnp.sum(jnp.abs(s) ** 2)

    eps = jnp.full(n, 4.0)
    grad = jax.grad(objective)(eps, options)
    assert len(calls) >= 2  # the sweep's handle, and the adjoints'
    assert all(dict(c) == LU_GMRES for c in calls)
    direct = jax.grad(objective)(eps, gfem.SolverOptions(linear_solver="direct"))
    np.testing.assert_allclose(grad, direct, rtol=0, atol=1e-8 * np.abs(direct).max())


@pytest.mark.parametrize("problem", ["electrostatic", "driven", "eigenmode"])
def test_an_unread_petsc_option_raises_from_every_problem(line, plate, problem):
    model, p = {
        "electrostatic": (
            plate,
            gfem.Electrostatic(terminals={"t": "top"}, ground="bottom"),
        ),
        "driven": (line, gfem.Driven((3e9, 4e9))),
        "eigenmode": (line, gfem.Eigenmode(target=12e9, n_modes=1)),
    }[problem]
    options = gfem.SolverOptions(
        order=1, linear_solver="petsc", petsc_options={"ksp_tpye": "gmres"}
    )
    with pytest.raises(ValueError, match=r"\['ksp_tpye'\] was not read"):
        gfem.solve(model, p, options=options)
