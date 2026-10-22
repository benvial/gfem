"""The cuDSS handle on a real GPU: run with ``pytest -m cuda``.

Small problems (a few thousand DOFs, well under 100 MB of device memory), so a
2 GB laptop GPU runs them. Each solve is checked against NGSolve's sparse
Cholesky on the CPU.
"""

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import quadrature
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import Operator, cudss, prepare
from gfem._linalg.cudss import CudssHandle

DATA = Path(__file__).parents[1] / "unit" / "data"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (4, 4, 4)))
    return to_ngsolve(gfem.Mesh.from_gmsh(path))


def _maxwell(built, k=4.0):
    """Complex-symmetric curl-curl with a first-order absorbing boundary."""
    space = ngs.HCurl(built.mesh, order=2, complex=True, dirichlet="bottom")
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += (ngs.curl(u) * ngs.curl(v) - k**2 * u * v) * quadrature.dx(2)
    form += -1j * k * u.Trace() * v.Trace() * quadrature.ds(2)
    return Operator(form, space.FreeDofs()), space


def _hermitian(built):
    space = ngs.H1(built.mesh, order=2, complex=True, dirichlet="bottom|top")
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


def _rhs(n, k, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, k)) + 1j * rng.standard_normal((n, k))


def _residual(handle, B, X, free):
    R = (handle.apply(X) - B)[free]
    return np.linalg.norm(R) / np.linalg.norm(B[free])


def test_complex_symmetric_maxwell_matches_sparse_cholesky(built):
    op, space = _maxwell(built)
    gpu = prepare(
        op, symmetry="complex_symmetric", kind="direct", device="cuda", width=4
    )
    cpu = prepare(_maxwell(built)[0], symmetry="complex_symmetric", kind="direct")
    free = np.array(list(space.FreeDofs()))
    B = _rhs(space.ndof, 4)
    X = gpu.solve(B)
    assert isinstance(X, np.ndarray)
    assert _residual(gpu, B, X, free) < 1e-10
    ref = cpu.solve(B)
    assert np.linalg.norm(X - ref) / np.linalg.norm(ref) < 1e-9
    np.testing.assert_array_equal(X[~free], 0)
    gpu.free()


def test_adjoint_and_narrow_blocks_reuse_the_factors(built):
    op, space = _maxwell(built)
    handle = prepare(
        op, symmetry="complex_symmetric", kind="direct", device="cuda", width=3
    )
    free = np.array(list(space.FreeDofs()))
    B = _rhs(space.ndof, 7, seed=1)  # three chunks, the last zero-padded
    X = handle.solve(B)
    assert _residual(handle, B, X, free) < 1e-10
    g = _rhs(space.ndof, 1, seed=2)[:, 0]  # one adjoint column, padded to 3
    lam = handle.solve(g, transpose=True)
    assert _residual(handle, g, lam, free) < 1e-10
    # s = g^T x is the same through the forward and the adjoint solve
    f = B[:, 0]
    np.testing.assert_allclose(g @ X[:, 0], lam @ f, rtol=1e-9)
    handle.free()


def test_complex_hermitian_operator(built):
    op, space = _hermitian(built)
    handle = prepare(op, symmetry="hermitian", kind="direct", device="cuda", width=2)
    free = np.array(list(space.FreeDofs()))
    B = _rhs(space.ndof, 2, seed=3)
    assert _residual(handle, B, handle.solve(B), free) < 1e-10
    XT = handle.solve(B, transpose=True)
    A = handle.apply(np.eye(space.ndof))[np.ix_(free, free)]
    np.testing.assert_allclose(A.T @ XT[free], B[free], atol=1e-9)
    handle.free()


def test_hybrid_memory_keeps_the_factors_on_the_host(built, monkeypatch):
    monkeypatch.setattr(cudss, "_free_device_memory", lambda _: 0)
    op, space = _maxwell(built)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda")
    assert handle.hybrid_memory
    free = np.array(list(space.FreeDofs()))
    B = _rhs(space.ndof, 1, seed=4)
    assert _residual(handle, B, handle.solve(B), free) < 1e-10
    handle.free()


def test_device_memory_is_released(built):
    free_before = cudss._free_device_memory(0)
    op, _ = _maxwell(built)
    handle = prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda")
    assert handle.nbytes > 0
    handle.free()
    assert handle.nbytes == 0
    # cuDSS releases the factors; allow for allocator caching (64 MiB)
    assert cudss._free_device_memory(0) > free_before - 64 * 2**20


def test_out_of_range_ordinal_names_the_device_count(built):
    op, _ = _maxwell(built)
    with pytest.raises(gfem.DeviceUnavailableError, match="CUDA device"):
        prepare(op, symmetry="complex_symmetric", kind="direct", device="cuda:64")


def test_electrostatic_capacitance_on_the_gpu(tmp_path):
    path = tmp_path / "plate.msh"
    layers = (("lo", 0.5), ("hi", 0.5))
    sm.write_msh22(path, sm.plate(1.0, 1.0, layers, (3, 3, 2)))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"lo": gfem.Material(eps_r=3.9), "hi": gfem.Material(eps_r=1.0)},
    )
    problem = gfem.Electrostatic(terminals={"a": "top", "b": "lo__hi"}, ground="bottom")

    def capacitance(device):
        options = gfem.SolverOptions(linear_solver="direct", device=device)
        return gfem.solve(model, problem, options=options).capacitance

    np.testing.assert_allclose(capacitance("cuda"), capacitance("cpu"), rtol=1e-9)


# --- one solver plan per sweep ----------------------------------------------------


@pytest.fixture
def plans(monkeypatch):
    """Count cuDSS plans (nvmath ``DirectSolver.plan`` calls)."""
    solver = cudss._nvmath().DirectSolver
    plan = solver.plan
    calls = []

    def spy(self, *args, **kwargs):
        calls.append(1)
        return plan(self, *args, **kwargs)

    monkeypatch.setattr(solver, "plan", spy)
    return calls


def test_refactorization_solves_like_a_fresh_handle_on_one_plan(built, plans, caplog):
    handle = prepare(
        _maxwell(built, k=4.0)[0],
        symmetry="complex_symmetric",
        kind="direct",
        device="cuda",
        width=2,
    )
    op, space = _maxwell(built, k=5.0)
    free = np.array(list(space.FreeDofs()))
    with caplog.at_level(logging.WARNING):
        handle.refactorize(op)
    assert len(plans) == 1
    # nvmath warns at every reset of the matrix; a sweep resets it on purpose
    assert "Resetting the LHS" not in caplog.text
    fresh = prepare(
        _maxwell(built, k=5.0)[0],
        symmetry="complex_symmetric",
        kind="direct",
        device="cuda",
        width=2,
    )
    B = _rhs(space.ndof, 3, seed=5)
    X, ref = handle.solve(B), fresh.solve(B)
    assert np.linalg.norm(X - ref) / np.linalg.norm(ref) < 1e-12
    assert _residual(handle, B, X, free) < 1e-10
    handle.free()
    fresh.free()


def test_a_changed_pattern_plans_again(built, plans, monkeypatch):
    # one space keeps its pattern, so a changed one is simulated
    handle = prepare(
        _maxwell(built, k=4.0)[0],
        symmetry="complex_symmetric",
        kind="direct",
        device="cuda",
    )
    monkeypatch.setattr(cudss, "same_pattern", lambda *_: False)
    op, space = _maxwell(built, k=5.0)
    handle.refactorize(op)
    assert len(plans) == 2
    free = np.array(list(space.FreeDofs()))
    B = _rhs(space.ndof, 1, seed=6)
    assert _residual(handle, B, handle.solve(B), free) < 1e-10
    handle.free()


def _plate_line(second):
    """The unit tests' parallel-plate line: a wave port and a ``second`` port."""
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    vp = ((20e-6, 0.0, 0.0), (20e-6, 10e-6, 0.0))
    ports = [gfem.WavePort("p1", "port1", voltage_path=vp)]
    if second == "wave":
        ports.append(gfem.WavePort("p2", "port2"))
    else:
        ports.append(
            gfem.LumpedPort("p2", "port2", direction=(0, 1, 0), impedance=47.0)
        )
    return gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("plates")],
        ports=ports,
    )


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
@pytest.mark.parametrize("second", ["wave", "lumped"])
def test_a_gpu_sweep_plans_once_and_matches_per_frequency_handles(
    plans, monkeypatch, sweep, second
):
    model = _plate_line(second)
    problem = gfem.Driven((1.5e12, 2.0e12, 2.5e12, 3.0e12), sweep=sweep)
    options = gfem.SolverOptions(order=3, linear_solver="direct", device="cuda")
    s = np.asarray(gfem.solve(model, problem, options=options).s.values)
    assert len(plans) == 1
    # a handle that cannot refactorize is planned afresh at each frequency
    monkeypatch.delattr(CudssHandle, "refactorize")
    plans.clear()
    result = gfem.solve(model, problem, options=options)
    assert len(plans) > 1
    np.testing.assert_allclose(result.s.values, s, rtol=0, atol=1e-12)


@pytest.fixture
def alive(monkeypatch):
    """The most cuDSS factorizations alive at once (handles prepared, not freed)."""
    held: set[int] = set()
    most = [0]
    init, free = CudssHandle.__init__, CudssHandle.free

    def spy_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        held.add(id(self))
        most[0] = max(most[0], len(held))

    def spy_free(self):
        held.discard(id(self))
        free(self)

    monkeypatch.setattr(CudssHandle, "__init__", spy_init)
    monkeypatch.setattr(CudssHandle, "free", spy_free)
    return most


def test_gpu_sweep_gradient_matches_the_cpu_on_one_factorization(
    tmp_path, plans, alive
):
    # S21 with only p1 excited: every frequency takes an adjoint solve
    path = tmp_path / "line.msh"
    sm.write_msh22(path, sm.strip_line(3.0, 2.0, 1.0, (3, 2, 1)))
    line = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material()},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=1),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    rng = np.random.default_rng(3)
    eps_r = jnp.asarray(rng.uniform(2.0, 5.0, len(line.mesh.tetrahedra)))
    problem = gfem.Driven(frequency=(15e9, 18e9, 21e9), excite="p1")

    def gradient(device):
        options = gfem.SolverOptions(linear_solver="direct", device=device)

        def objective(p):
            model = line.with_material("dielectric", gfem.MaterialField(eps_r=p))
            s = gfem.solve(model, problem, options=options).s
            return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

        return np.asarray(jax.grad(objective)(eps_r))

    gpu = gradient("cuda")
    # one plan for the forward sweep, one for the backward's adjoint solves
    assert len(plans) == 2
    assert alive[0] == 1
    cpu = gradient("cpu")
    assert np.abs(cpu).max() > 0
    np.testing.assert_allclose(gpu, cpu, rtol=1e-9, atol=1e-12 * np.abs(cpu).max())
