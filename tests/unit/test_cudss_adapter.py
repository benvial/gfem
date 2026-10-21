"""The cuDSS adapter's plumbing, against a stand-in for nvmath's DirectSolver.

The stand-in keeps nvmath's contract (a lower-triangle CSR view with int32
indices, a column-major right-hand side of fixed shape, NumPy in and out) and
solves with SciPy, so these tests run without a GPU. tests/cuda runs the
same handle on a real device.
"""

import types

import ngsolve as ngs
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import structured_meshes as sm

import gfem
from gfem import _cuda
from gfem._engine import quadrature
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import Operator, cudss, prepare

GIB = 2**30


class FakeDirectSolver:
    """nvmath.sparse.advanced.DirectSolver on the CPU."""

    instances: list = []  # noqa: RUF012

    def __init__(self, a, b, *, options, execution):
        assert isinstance(a, sp.csr_matrix)
        assert a.indices.dtype == np.int32
        assert a.indptr.dtype == np.int32
        assert options.sparse_system_view == "LOWER"
        assert sp.triu(a, 1).nnz == 0, "only the lower triangle is passed"
        assert b.flags.f_contiguous
        self.a, self.b, self.options, self.execution = a, b, options, execution
        self.freed = False
        self.solves = 0
        FakeDirectSolver.instances.append(self)

    def plan(self):
        estimates = types.SimpleNamespace(
            peak_device_memory=GIB,
            permanent_device_memory=1000,
            permanent_host_memory=10,
        )
        return types.SimpleNamespace(memory_estimates=estimates)

    def factorize(self):
        strict = sp.tril(self.a, -1)
        mirror = strict.T
        if self.options.sparse_system_type == "HERMITIAN":
            mirror = mirror.conj()
        self._lu = spla.splu((self.a + mirror).tocsc(), permc_spec="MMD_AT_PLUS_A")

    def reset_operands(self, *, b):
        if b.shape != self.b.shape or b.dtype != self.b.dtype:
            raise TypeError("The shape of 'b' doesn't match the original one.")
        assert b.flags.f_contiguous
        self.b = b

    def solve(self):
        self.solves += 1
        return self._lu.solve(self.b)

    def free(self):
        self.freed = True


FAKE_NVMATH = types.SimpleNamespace(
    DirectSolver=FakeDirectSolver,
    DirectSolverOptions=types.SimpleNamespace,
    DirectSolverMatrixType=types.SimpleNamespace(
        SYMMETRIC="SYMMETRIC", HERMITIAN="HERMITIAN"
    ),
    DirectSolverMatrixViewType=types.SimpleNamespace(LOWER="LOWER"),
    ExecutionCUDA=types.SimpleNamespace,
    HybridMemoryModeOptions=types.SimpleNamespace,
)


@pytest.fixture
def gpu(monkeypatch):
    """A fake GPU with ``gpu.free`` bytes of free memory."""
    monkeypatch.setattr(_cuda, "_query", lambda: _cuda.Probe(2))
    monkeypatch.setattr(cudss, "_nvmath", lambda: FAKE_NVMATH)
    state = types.SimpleNamespace(free=2 * GIB)
    monkeypatch.setattr(cudss, "_free_device_memory", lambda _: state.free)
    _cuda.probe.cache_clear()
    FakeDirectSolver.instances.clear()
    yield state
    _cuda.probe.cache_clear()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 2)))
    return to_ngsolve(gfem.Mesh.from_gmsh(path))


def _space(built, complex_valued):
    return ngs.H1(built.mesh, order=2, dirichlet="bottom|top", complex=complex_valued)


def _symmetric(built, shift=1 - 2j):
    space = _space(built, complex_valued=True)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += (ngs.grad(u) * ngs.grad(v) + shift * u * v) * quadrature.dx(2)
    return Operator(form, space.FreeDofs()), space


def _hermitian(built):
    # grad u . grad v + i (b . grad u) v - i u (b . grad v): A^H = A, A^T != A
    space = _space(built, complex_valued=True)
    u, v = space.TnT()
    b = ngs.CF((0.3, -0.2, 0.5))
    form = ngs.BilinearForm(space)
    form += (
        ngs.grad(u) * ngs.grad(v)
        + 1j * (b * ngs.grad(u)) * v
        - 1j * u * (b * ngs.grad(v))
        + u * v
    ) * quadrature.dx(2)
    return Operator(form, space.FreeDofs()), space


def _real(built):
    space = _space(built, complex_valued=False)
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += ngs.grad(u) * ngs.grad(v) * quadrature.dx(2)
    return Operator(form, space.FreeDofs()), space


def _rhs(n, k, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))


def _reference(handle, B, free, transpose=False):
    """``A_ff^-1 B_f`` (or ``A_ff^-T``) by dense algebra; zero on fixed rows."""
    A = handle.apply(np.eye(handle.ndof))[np.ix_(free, free)]
    X = np.zeros(B.shape, dtype=complex)
    X[free] = np.linalg.solve(A.T if transpose else A, B[free])
    return X


def test_lower_pattern_gathers_the_free_lower_triangle():
    # NGSolve's CSR: symmetric pattern, sorted unique columns per row
    rng = np.random.default_rng(3)
    n = 40
    upper = sp.random(n, n, density=0.15, random_state=rng, dtype=complex)
    upper.data += 1j * rng.standard_normal(upper.nnz)
    A = (upper + upper.T + sp.eye(n)).tocsr()
    A.sort_indices()
    free = rng.random(n) > 0.3
    idx = np.flatnonzero(free)
    reference = sp.tril(A[idx][:, idx], format="csr")
    reference.sort_indices()

    indptr, indices, gather = cudss.lower_pattern(A.indptr, A.indices, free)

    np.testing.assert_array_equal(indptr, reference.indptr)
    np.testing.assert_array_equal(indices, reference.indices)
    np.testing.assert_array_equal(np.take(A.data, gather), reference.data)
    assert indptr.dtype == indices.dtype == np.int32


@pytest.mark.usefixtures("gpu")
def test_complex_symmetric_solves_match_the_cpu_solver(built):
    op, space = _symmetric(built)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda")
    cpu = prepare(_symmetric(built)[0], symmetry="complex_symmetric", kind="direct")
    assert handle.kind == "cudss"
    B = _rhs(space.ndof, 3)
    free = np.array(list(space.FreeDofs()))
    X = handle.solve(B)
    np.testing.assert_allclose(X, cpu.solve(B), rtol=1e-10, atol=1e-14)
    np.testing.assert_array_equal(X[~free], 0)
    np.testing.assert_allclose(handle.solve(B, transpose=True), X, rtol=1e-14)
    (solver,) = FakeDirectSolver.instances
    assert solver.options.sparse_system_type == "SYMMETRIC"  # L D L^T, no conjugate


@pytest.mark.usefixtures("gpu")
def test_complex_hermitian_operator_is_factored_with_the_conjugate(built):
    op, space = _hermitian(built)
    handle = prepare(op, symmetry="hermitian", kind="direct", device="cuda")
    free = np.array(list(space.FreeDofs()))
    A = handle.apply(np.eye(space.ndof))[np.ix_(free, free)]
    assert np.abs(A - A.T).max() > 1e-3  # not complex symmetric
    np.testing.assert_allclose(A, A.conj().T, atol=1e-12)
    B = _rhs(space.ndof, 2)
    np.testing.assert_allclose(handle.solve(B), _reference(handle, B, free), rtol=1e-10)
    np.testing.assert_allclose(
        handle.solve(B, transpose=True),
        _reference(handle, B, free, transpose=True),
        rtol=1e-10,
    )
    (solver,) = FakeDirectSolver.instances
    assert solver.options.sparse_system_type == "HERMITIAN"


@pytest.mark.usefixtures("gpu")
def test_real_operator_solves_real_and_complex_rhs(built):
    op, space = _real(built)
    handle = prepare(op, symmetry="hermitian", kind="direct", device="cuda")
    free = np.array(list(space.FreeDofs()))
    B = _rhs(space.ndof, 1)[:, 0]
    np.testing.assert_allclose(handle.solve(B), _reference(handle, B, free), rtol=1e-10)
    np.testing.assert_allclose(
        handle.solve(B.real), _reference(handle, B.real, free).real, rtol=1e-10
    )
    (solver,) = FakeDirectSolver.instances
    assert solver.a.dtype == np.float64
    assert solver.options.sparse_system_type == "SYMMETRIC"


@pytest.mark.usefixtures("gpu")
@pytest.mark.parametrize("k", [1, 2, 5])
def test_blocks_of_any_width_reuse_one_factorization(built, k):
    # width 2: one column is zero-padded, five go in three chunks
    op, space = _symmetric(built)
    handle = prepare(
        op, symmetry="complex_symmetric", kind="direct", device="cuda", width=2
    )
    B = _rhs(space.ndof, k, seed=k)
    free = np.array(list(space.FreeDofs()))
    np.testing.assert_allclose(handle.solve(B), _reference(handle, B, free), rtol=1e-10)
    (solver,) = FakeDirectSolver.instances
    assert solver.b.shape == (free.sum(), 2)
    assert solver.solves == -(-k // 2)


@pytest.mark.usefixtures("gpu")
def test_the_ordinal_selects_the_gpu(built):
    op, _ = _symmetric(built)
    prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda:1")
    (solver,) = FakeDirectSolver.instances
    assert solver.execution.device_id == 1


def test_factors_go_to_host_memory_when_the_gpu_is_too_small(gpu, built):
    gpu.free = GIB // 2  # the plan's peak estimate is 1 GiB
    op, _ = _symmetric(built)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda")
    first, second = FakeDirectSolver.instances
    assert first.freed
    assert not first.execution.hybrid_memory_mode_options.hybrid_memory_mode
    assert second.execution.hybrid_memory_mode_options.hybrid_memory_mode
    assert handle.hybrid_memory


@pytest.mark.usefixtures("gpu")
def test_factors_stay_on_the_gpu_when_they_fit(built):
    op, _ = _symmetric(built)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda")
    (solver,) = FakeDirectSolver.instances
    assert not solver.execution.hybrid_memory_mode_options.hybrid_memory_mode
    assert not handle.hybrid_memory


@pytest.mark.usefixtures("gpu")
def test_nbytes_counts_device_memory_and_free_releases_it(built):
    op, _ = _symmetric(built)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda")
    assert handle.nbytes > 1000  # the plan's permanent device bytes
    handle.free()
    (solver,) = FakeDirectSolver.instances
    assert solver.freed
    assert handle.nbytes == 0
    with pytest.raises(RuntimeError, match="freed"):
        handle.solve(np.ones(handle.ndof))


@pytest.mark.usefixtures("gpu")
def test_general_operators_are_refused(built):
    op, _ = _symmetric(built)
    with pytest.raises(NotImplementedError, match="general"):
        prepare(op, symmetry="general", kind="direct", device="cuda")


@pytest.mark.usefixtures("gpu")
def test_electrostatic_solve_on_the_gpu_matches_the_cpu(tmp_path):
    path = tmp_path / "plate.msh"
    layers = (("lo", 0.5), ("hi", 0.5))
    sm.write_msh22(path, sm.plate(1.0, 1.0, layers, (2, 2, 2)))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"lo": gfem.Material(eps_r=3.9), "hi": gfem.Material(eps_r=1.0)},
    )
    problem = gfem.Electrostatic(terminals={"a": "top", "b": "lo__hi"}, ground="bottom")

    def run(device):
        options = gfem.SolverOptions(linear_solver="direct", device=device)
        return gfem.solve(model, problem, options=options).capacitance

    np.testing.assert_allclose(run("cuda"), run("cpu"), rtol=1e-10)
    (solver,) = FakeDirectSolver.instances
    assert solver.b.shape[1] == 2  # one column per terminal


def _wheel(tmp_path, layout):
    lib = tmp_path / "nvidia" / layout / "lib"
    lib.mkdir(parents=True)
    path = lib / "libcudss_mtlayer_gomp.so.0"
    path.write_bytes(b"")
    return str(path)


@pytest.mark.usefixtures("gpu")
@pytest.mark.parametrize("layout", ["cu12", "cu13"])
def test_threading_layer_is_found_in_either_wheel_layout(
    built, monkeypatch, tmp_path, layout
):
    expected = _wheel(tmp_path, layout)
    monkeypatch.setattr(cudss, "_nvidia_roots", lambda: [tmp_path / "nvidia"])
    op, _ = _symmetric(built)
    prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda")
    (solver,) = FakeDirectSolver.instances
    assert solver.options.multithreading_lib == expected


@pytest.mark.usefixtures("gpu")
def test_missing_threading_layer_falls_back_to_single_thread(
    built, monkeypatch, tmp_path, recwarn
):
    (tmp_path / "nvidia" / "cu13" / "lib").mkdir(parents=True)
    monkeypatch.setattr(cudss, "_nvidia_roots", lambda: [tmp_path / "nvidia"])
    op, space = _symmetric(built)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda")
    (solver,) = FakeDirectSolver.instances
    assert solver.options.multithreading_lib is None
    assert handle.solve(_rhs(space.ndof, 1)).shape == (space.ndof, 1)
    assert not recwarn.list


@pytest.mark.usefixtures("gpu")
def test_one_thread_asks_for_no_threading_layer(built, monkeypatch, tmp_path):
    _wheel(tmp_path, "cu13")
    monkeypatch.setattr(cudss, "_nvidia_roots", lambda: [tmp_path / "nvidia"])
    op, _ = _symmetric(built)
    prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda", threads=1)
    (solver,) = FakeDirectSolver.instances
    assert solver.options.multithreading_lib is None
