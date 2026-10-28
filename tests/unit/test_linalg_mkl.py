"""The ``direct`` backend's inverse: MKL PARDISO with METIS when ``gfem[mkl]`` is in.

These tests run with and without the ``mkl`` extra (``uv run --extra mkl
pytest``): without it the handle is NGSolve's sparse Cholesky, with it
MKL PARDISO, and the answers agree to the direct solve's accuracy.
"""

import ast
import importlib.metadata
import platform
import sys
import types
from pathlib import Path

import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import quadrature
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import Operator, direct, pardiso, prepare
from gfem._linalg.base import host_bytes


def _mkl_installed() -> bool:
    # the oracle is the installed distribution, not gfem's own detection
    try:
        importlib.metadata.distribution("mkl")
    except importlib.metadata.PackageNotFoundError:
        return False
    return sys.platform in {"linux", "win32"} and platform.machine() in {
        "x86_64",
        "AMD64",
    }


MKL = _mkl_installed()
needs_mkl = pytest.mark.skipif(not MKL, reason="needs the mkl extra")

SRC = Path(gfem.__file__).parent


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


@pytest.mark.parametrize(
    ("complex_valued", "symmetry", "matrix_type"),
    [(False, "hermitian", -2), (True, "complex_symmetric", 6)],
)
def test_direct_is_pardiso_with_parallel_metis_exactly_when_mkl_is_installed(
    built, complex_valued, symmetry, matrix_type
):
    op, space = _laplace(built, complex_valued, shift=1 - 2j if complex_valued else 0)
    handle = prepare(op, symmetry=symmetry, kind="direct")
    inverse = handle._inverse
    if not MKL:
        assert type(inverse).__name__.startswith("SparseCholesky")
        return
    assert type(inverse).__name__ == "MKLPardiso"
    assert inverse._params[1] == 3  # iparm[1] = 3: OpenMP parallel METIS
    # iparm[7] = 0: no iterative refinement (MKL's default), not the up to 16
    # steps of NGSolve's defaults
    assert inverse._params[7] == 0
    # the symmetric (LDL^T) matrix type, not the unsymmetric LU of NGSolve's
    # defaults for a fully stored matrix
    assert inverse._matrixtype[0] == matrix_type
    free = np.array(list(space.FreeDofs()))
    rng = np.random.default_rng(3)
    B = rng.standard_normal(space.ndof)
    if complex_valued:
        B = B + 1j * rng.standard_normal(space.ndof)
    X = handle.solve(B)
    residual = (handle.apply(X) - B)[free]
    assert np.abs(residual).max() < 1e-10 * np.abs(B).max()
    np.testing.assert_allclose(handle.solve(B, transpose=True), X, rtol=1e-14)


def test_direct_handles_an_operator_without_free_dofs(built):
    # every DOF fixed (an electrostatic terminal and ground on a one-cell
    # plate): PARDISO refuses an empty matrix, so this cannot reach it
    op, space = _laplace(built)
    none_free = ngs.BitArray(space.ndof)
    none_free.Clear()
    handle = prepare(Operator(op.form, none_free), symmetry="hermitian", kind="direct")
    np.testing.assert_array_equal(handle.solve(np.ones(space.ndof)), 0)


@pytest.fixture
def no_detection_cache():
    """Detect PARDISO afresh in the test, and again after it."""
    direct.pardiso_available.cache_clear()
    yield
    direct.pardiso_available.cache_clear()


@pytest.mark.usefixtures("no_detection_cache")
def test_an_unknown_ngsolve_pardiso_wrapper_falls_back_to_sparse_cholesky(
    built, monkeypatch
):
    # the wrapper's private attributes may move in another NGSolve version:
    # direct solves keep working on sparse Cholesky, never raise
    wrapper = types.ModuleType("ngsolve.solvers.mkl_pardiso")
    monkeypatch.setitem(sys.modules, "ngsolve.solvers.mkl_pardiso", wrapper)
    monkeypatch.setattr(ngs.solvers, "mkl_pardiso", wrapper, raising=False)
    assert not direct.pardiso_available()
    op, _ = _laplace(built)
    handle = prepare(op, symmetry="hermitian", kind="direct")
    assert type(handle._inverse).__name__.startswith("SparseCholesky")


def test_direct_memory_counts_the_factors(built):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct")
    assert handle.nbytes > host_bytes(handle._matrix)


@needs_mkl
def test_pardiso_refactorization_keeps_its_plan_and_solves_like_a_fresh_handle(
    built,
):
    handle = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j)[0],
        symmetry="complex_symmetric",
        kind="direct",
    )
    inverse = handle._inverse
    op, space = _laplace(built, complex_valued=True, shift=3 + 1j)
    handle.refactorize(op)
    assert handle._inverse is inverse  # Update(): numeric factorization only
    fresh = prepare(
        _laplace(built, complex_valued=True, shift=3 + 1j)[0],
        symmetry="complex_symmetric",
        kind="direct",
    )
    rng = np.random.default_rng(4)
    B = rng.standard_normal(space.ndof) + 1j * rng.standard_normal(space.ndof)
    X, ref = handle.solve(B), fresh.solve(B)
    assert np.linalg.norm(X - ref) / np.linalg.norm(ref) < 1e-12


@needs_mkl
def test_pardiso_refactorization_gathers_the_new_values_into_the_wrapper_buffer(
    built, monkeypatch
):
    handle = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j)[0],
        symmetry="complex_symmetric",
        kind="direct",
    )
    inverse = handle._inverse

    def no_extraction(self):  # noqa: ARG001
        raise AssertionError("the wrapper extracted the upper triangle again")

    monkeypatch.setattr(type(inverse), "get_csr", no_extraction)
    op, _ = _laplace(built, complex_valued=True, shift=3 + 1j)
    handle.refactorize(op)
    monkeypatch.undo()
    fresh = prepare(
        _laplace(built, complex_valued=True, shift=3 + 1j)[0],
        symmetry="complex_symmetric",
        kind="direct",
    )
    # the fresh wrapper's buffer is its own extraction of the new values
    np.testing.assert_array_equal(inverse._data_np, fresh._inverse._data_np)
    # the full matrix stays current: `apply` reads it
    X = np.ones(handle.ndof, complex)
    np.testing.assert_array_equal(handle.apply(X), fresh.apply(X))


@needs_mkl
@pytest.mark.parametrize("change", ["no phase call", "another buffer order"])
def test_a_pardiso_wrapper_gfem_cannot_gather_into_refactorizes_by_update(
    built, monkeypatch, change
):
    # another NGSolve version may name its PARDISO call otherwise, or order
    # its buffer otherwise: PARDISO stays, and refactorizes through Update()
    from ngsolve.solvers import mkl_pardiso

    factors = []

    class Older(mkl_pardiso.MKLPardiso):
        def __getattribute__(self, name):
            caller = sys._getframe(1).f_globals.get("__name__")
            value = super().__getattribute__(name)
            if caller == mkl_pardiso.__name__:
                return value
            if change == "no phase call" and name == "_call_pardiso":
                raise AttributeError(name)
            if change == "another buffer order" and name == "_data_np":
                return value[::-1]
            return value

        def Factor(self):  # noqa: N802 (NGSolve's name)
            factors.append(self)
            super().Factor()

    ngs.la.RegisterInverseType("gfem_older_pardiso", Older)
    monkeypatch.setattr(direct, "PARDISO", "gfem_older_pardiso")
    handle = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j)[0],
        symmetry="complex_symmetric",
        kind="direct",
    )
    inverse = handle._inverse
    assert type(inverse) is Older
    factors.clear()
    op, space = _laplace(built, complex_valued=True, shift=3 + 1j)
    handle.refactorize(op)
    assert handle._inverse is inverse
    assert factors == [inverse]  # Update(): the wrapper's own Factor
    monkeypatch.undo()
    fresh = prepare(
        _laplace(built, complex_valued=True, shift=3 + 1j)[0],
        symmetry="complex_symmetric",
        kind="direct",
    )
    rng = np.random.default_rng(5)
    B = rng.standard_normal(space.ndof) + 1j * rng.standard_normal(space.ndof)
    X, ref = handle.solve(B), fresh.solve(B)
    assert np.linalg.norm(X - ref) / np.linalg.norm(ref) < 1e-12


@needs_mkl
@pytest.mark.parametrize(
    "failure", ["a matrix without CSR", "a failing CSR", "a failing wrapper buffer"]
)
def test_a_gather_index_gfem_cannot_build_leaves_refactorization_to_update(
    built, monkeypatch, failure
):
    # building the gather index never raises: without one, PARDISO stays and
    # refactorizes through Update()
    from ngsolve.solvers import mkl_pardiso

    def from_gfem():
        return sys._getframe(2).f_globals.get("__name__", "").startswith("gfem.")

    form = _laplace(built, complex_valued=True)[0].form
    form.Assemble()
    matrix_type = type(form.mat)
    csr = matrix_type.CSR

    def failing_csr(self):
        if not from_gfem():
            return csr(self)
        if failure == "a matrix without CSR":
            raise AttributeError("CSR")
        raise RuntimeError("no CSR of this matrix")

    class FailingBuffer(mkl_pardiso.MKLPardiso):
        def __getattribute__(self, name):
            if name == "_data_np" and from_gfem():
                raise RuntimeError("no buffer")
            return super().__getattribute__(name)

    ngs.la.RegisterInverseType("gfem_failing_buffer_pardiso", FailingBuffer)
    if failure == "a failing wrapper buffer":
        monkeypatch.setattr(direct, "PARDISO", "gfem_failing_buffer_pardiso")
    else:
        monkeypatch.setattr(matrix_type, "CSR", failing_csr)
    handle = prepare(
        _laplace(built, complex_valued=True, shift=1 - 2j)[0],
        symmetry="complex_symmetric",
        kind="direct",
    )
    assert handle._gather is None
    monkeypatch.undo()
    assert isinstance(handle._inverse, mkl_pardiso.MKLPardiso)
    op, space = _laplace(built, complex_valued=True, shift=3 + 1j)
    handle.refactorize(op)
    fresh = prepare(
        _laplace(built, complex_valued=True, shift=3 + 1j)[0],
        symmetry="complex_symmetric",
        kind="direct",
    )
    rng = np.random.default_rng(6)
    B = rng.standard_normal(space.ndof) + 1j * rng.standard_normal(space.ndof)
    X, ref = handle.solve(B), fresh.solve(B)
    assert np.linalg.norm(X - ref) / np.linalg.norm(ref) < 1e-12


@needs_mkl
def test_pardiso_memory_counts_the_gather_index(built):
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct")
    index = handle._gather
    held = handle.nbytes
    handle._gather = None
    assert held - handle.nbytes == index.nbytes > 0


# --- BDDC's coarse factorization ---------------------------------------------------


@pytest.fixture
def coarse_wrappers(monkeypatch):
    """Record each PARDISO wrapper NGSolve plans: its type, flags and matrix type."""
    from ngsolve.solvers import mkl_pardiso

    seen = []
    analyze = mkl_pardiso.MKLPardiso.Analyze

    def recording(self):
        analyze(self)
        seen.append(
            {
                "type": type(self),
                "symmetric": self._force_symmetric,
                "ordering": self._params[1],
                "refinement": self._params[7],
                "matrix_type": self._matrixtype[0],
            }
        )

    monkeypatch.setattr(mkl_pardiso.MKLPardiso, "Analyze", recording)
    return seen


def _bddc_solve(built, complex_valued, symmetry):
    shift = 1 - 2j if symmetry == "complex_symmetric" else 1.0
    op, space = _laplace(built, complex_valued, shift=shift, order=4)
    handle = prepare(op, symmetry=symmetry, kind="bddc_gmres", tolerance=1e-10)
    rng = np.random.default_rng(7)
    B = rng.standard_normal(space.ndof)
    if complex_valued:
        B = B + 1j * rng.standard_normal(space.ndof)
    return handle, handle.solve(B)


@needs_mkl
@pytest.mark.parametrize(
    ("complex_valued", "symmetry", "symmetric", "matrix_type"),
    [
        (True, "complex_symmetric", True, 6),
        (True, "general", False, 13),
        (False, "hermitian", True, -2),  # electrostatics: real SPD, CG
        # a complex Hermitian coarse matrix is not A^T = A: unsymmetric LU
        (True, "hermitian", False, 13),
    ],
)
def test_bddc_coarse_problem_is_factored_by_gfem_flagged_pardiso(  # noqa: PLR0917
    built,
    monkeypatch,
    coarse_wrappers,
    complex_valued,
    symmetry,
    symmetric,
    matrix_type,
):
    handle, X = _bddc_solve(built, complex_valued, symmetry)
    inverse = handle._preconditioner.flags.ToDict()["inverse"]
    assert inverse.startswith("gfem_")  # never NGSolve's own "pardiso"
    assert len(coarse_wrappers) == 1
    (wrapper,) = coarse_wrappers
    assert wrapper["type"].__module__ == "gfem._linalg.pardiso"
    assert wrapper["symmetric"] is symmetric
    assert wrapper["ordering"] == 3  # OpenMP parallel METIS
    assert wrapper["refinement"] == 0
    assert wrapper["matrix_type"] == matrix_type
    # the same iterations and solution as a sparse Cholesky coarse problem
    monkeypatch.setattr(pardiso, "pardiso_available", lambda: False)
    ref_handle, ref = _bddc_solve(built, complex_valued, symmetry)
    assert ref_handle._preconditioner.flags.ToDict()["inverse"] == "sparsecholesky"
    assert len(coarse_wrappers) == 1
    assert handle.iterations == ref_handle.iterations
    assert handle.iterations[-1] > 1  # order 4: the coarse problem is not all of A
    assert np.linalg.norm(X - ref) <= 1e-10 * np.linalg.norm(ref)


@pytest.mark.parametrize("symmetry", ["complex_symmetric", "general"])
def test_bddc_coarse_problem_without_mkl_is_sparse_cholesky(
    built, monkeypatch, symmetry
):
    monkeypatch.setattr(pardiso, "pardiso_available", lambda: False)
    assert pardiso.pardiso_inverse(symmetry) is None
    op, _ = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry=symmetry, kind="bddc_gmres")
    assert handle._preconditioner.flags.ToDict()["inverse"] == "sparsecholesky"


@pytest.mark.parametrize("symmetry", ["hermitian", "general"])
def test_bddc_handles_a_coarse_problem_without_free_dofs(tmp_path, symmetry):
    # every wirebasket DOF fixed (an electrostatic terminal and ground on a
    # one-cell plate): PARDISO refuses an empty matrix, so it is never called
    path = tmp_path / "cell.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (1, 1, 1)))
    op, space = _laplace(to_ngsolve(gfem.Mesh.from_gmsh(path)), order=1)
    assert space.FreeDofs().NumSet() == 0
    handle = prepare(op, symmetry=symmetry, kind="bddc_gmres")
    np.testing.assert_array_equal(handle.solve(np.ones(space.ndof)), 0)


# --- no reliance on NGSolve's default inverse ---------------------------------------


def test_every_gfem_inverse_call_names_its_type():
    # installing mkl switches NGSolve's process-wide default inverse to
    # PARDISO LU with minimum degree, so no call may leave it implicit
    unnamed = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            keywords = {k.arg for k in node.keywords}
            if name == "Inverse" and "inverse" not in keywords:
                unnamed.append(f"{path.relative_to(SRC)}:{node.lineno}")
            # an explicit keyword: ``**flags`` may not hold it
            if name == "Preconditioner" and "inverse" not in keywords:
                unnamed.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert unnamed == []


class _Forbidden(ngs.la.SparseFactorizationInterface):
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        raise AssertionError("gfem used NGSolve's default inverse type")


@pytest.fixture
def forbidden_default():
    """Make NGSolve's default inverse type raise for the test's duration."""
    previous = ngs.BaseMatrix.GetDefaultInverseType()
    ngs.la.RegisterInverseType("gfem_forbidden", _Forbidden)
    ngs.BaseMatrix.SetDefaultInverseType("gfem_forbidden")
    yield
    ngs.BaseMatrix.SetDefaultInverseType(previous)


@pytest.mark.usefixtures("forbidden_default")
@pytest.mark.parametrize("kind", ["direct", "bddc_gmres"])
def test_no_backend_falls_back_to_the_default_inverse(built, kind):
    op, space = _laplace(built, complex_valued=True, shift=1 - 2j)
    handle = prepare(op, symmetry="complex_symmetric", kind=kind)
    assert np.isfinite(handle.solve(np.ones(space.ndof, complex))).all()


# --- the same answers through either factorization ----------------------------------

L, W, H = 10.0, 2.0, 1.0  # mm


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (10, 1, 1)))
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0, tan_delta=1e-3)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


DIRECT = gfem.SolverOptions(linear_solver="direct")


@needs_mkl
def test_pardiso_s_matches_sparse_cholesky(line, monkeypatch):
    problem = gfem.Driven(frequency=(3e9, 5e9, 7e9))
    s = np.asarray(gfem.solve(line, problem, options=DIRECT).s.values)
    monkeypatch.setattr(direct, "pardiso_available", lambda: False)
    ref = np.asarray(gfem.solve(line, problem, options=DIRECT).s.values)
    np.testing.assert_allclose(s, ref, rtol=0, atol=1e-10)


@needs_mkl
def test_pardiso_eigenmodes_match_sparse_cholesky(line, monkeypatch):
    problem = gfem.Eigenmode(target=8e9, n_modes=2)
    result = gfem.solve(line, problem, options=DIRECT)
    monkeypatch.setattr(direct, "pardiso_available", lambda: False)
    ref = gfem.solve(line, problem, options=DIRECT)
    np.testing.assert_allclose(result.omega, ref.omega, rtol=1e-10)
    np.testing.assert_allclose(result.q, ref.q, rtol=1e-8)
