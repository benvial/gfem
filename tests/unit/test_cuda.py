"""The no-GPU contract: `device="cuda"` raises, never falls back to the CPU.

`gfem.cuda_available()` starts a CUDA context; these tests replace that probe,
so they hold on any machine, with or without a GPU.
"""

import contextlib
import warnings

import ngsolve as ngs
import pytest
import structured_meshes as sm

import gfem
from gfem import _cuda
from gfem._engine import quadrature
from gfem._engine.mesh import to_ngsolve
from gfem._linalg import Operator, prepare


@pytest.fixture
def probe(monkeypatch):
    """Set what the CUDA probe finds: ``probe(count, reason=None)``."""

    def set_(count, reason=None, *, platform="linux"):
        monkeypatch.setattr(_cuda, "_query", lambda: _cuda.Probe(count, reason))
        monkeypatch.setattr(_cuda.sys, "platform", platform)
        _cuda.probe.cache_clear()

    yield set_
    _cuda.probe.cache_clear()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 2)))
    return to_ngsolve(gfem.Mesh.from_gmsh(path))


def _laplace(built):
    space = ngs.H1(built.mesh, order=2, dirichlet="bottom|top")
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True)
    form += ngs.grad(u) * ngs.grad(v) * quadrature.dx(2)
    return Operator(form, space.FreeDofs())


def test_cuda_available_is_false_without_a_device(probe):
    probe(0, "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected")
    assert gfem.cuda_available() is False
    probe(1)
    assert gfem.cuda_available() is True


def test_device_unavailable_error_is_a_runtime_error():
    assert issubclass(gfem.DeviceUnavailableError, RuntimeError)


def test_no_device_raises_with_the_cuda_error_text(probe, built):
    probe(0, "CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected")
    with pytest.raises(gfem.DeviceUnavailableError, match="CUDA_ERROR_NO_DEVICE"):
        prepare(_laplace(built), symmetry="hermitian", kind="direct", device="cuda")


def test_missing_extra_names_the_install_command(probe, built):
    probe(0, None)  # the cuda extra is not installed
    with pytest.raises(gfem.DeviceUnavailableError, match=r"gfem\[cuda12\]"):
        prepare(_laplace(built), symmetry="hermitian", kind="direct", device="cuda")


def test_macos_says_cuda_is_unsupported(probe, built):
    probe(0, None, platform="darwin")
    with pytest.raises(gfem.DeviceUnavailableError, match="not supported on macOS"):
        prepare(_laplace(built), symmetry="hermitian", kind="direct", device="cuda")


def test_ordinal_out_of_range_names_the_device_count(probe, built):
    probe(2)
    with pytest.raises(gfem.DeviceUnavailableError, match="2 CUDA devices"):
        prepare(_laplace(built), symmetry="hermitian", kind="direct", device="cuda:2")


def test_the_device_is_checked_before_the_assembly(probe, built):
    probe(0, "CUDA_ERROR_NO_DEVICE")
    op = _laplace(built)
    with pytest.raises(gfem.DeviceUnavailableError):
        prepare(op, symmetry="hermitian", kind="direct", device="cuda")
    with pytest.raises(TypeError, match="assemble"):
        _ = op.form.mat  # never assembled


def test_only_the_direct_solver_runs_on_a_gpu(probe, built):
    probe(1)
    with pytest.raises(ValueError, match="direct"):
        prepare(_laplace(built), symmetry="hermitian", kind="bddc_gmres", device="cuda")


def test_solve_raises_instead_of_falling_back(probe, tmp_path):
    probe(0, "CUDA_ERROR_NO_DEVICE")
    path = tmp_path / "plate.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (2, 2, 1)))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"box": gfem.Material(eps_r=1.0)}
    )
    options = gfem.SolverOptions(linear_solver="direct", device="cuda")
    with pytest.raises(gfem.DeviceUnavailableError, match="CUDA_ERROR_NO_DEVICE"):
        gfem.solve(
            model,
            gfem.Electrostatic(terminals={"t": "top"}, ground="bottom"),
            options=options,
        )


@pytest.mark.skipif(
    gfem.cuda_available(), reason="this machine has a usable CUDA device"
)
def test_without_a_gpu_device_cuda_raises_for_real(built):
    with pytest.raises(gfem.DeviceUnavailableError):
        prepare(_laplace(built), symmetry="hermitian", kind="direct", device="cuda")


# --- the JAX preallocation warning ---------------------------------------------


@pytest.fixture
def jax_on_gpu(monkeypatch):
    monkeypatch.setattr(_cuda, "_jax_backend", lambda: "gpu")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    monkeypatch.setattr(_cuda, "_warned", False)


@pytest.mark.usefixtures("jax_on_gpu")
def test_jax_preallocation_warns_once():
    with pytest.warns(gfem.DeviceMemoryWarning, match="XLA_PYTHON_CLIENT_PREALLOCATE"):
        _cuda.warn_if_jax_preallocates()
    with _no_warning():
        _cuda.warn_if_jax_preallocates()


@pytest.mark.parametrize(
    "env",
    [
        {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        {"XLA_PYTHON_CLIENT_MEM_FRACTION": ".3"},
    ],
)
@pytest.mark.usefixtures("jax_on_gpu")
def test_no_warning_when_jax_does_not_preallocate(monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with _no_warning():
        _cuda.warn_if_jax_preallocates()


@pytest.mark.usefixtures("jax_on_gpu")
def test_no_warning_when_jax_runs_on_the_cpu(monkeypatch):
    monkeypatch.setattr(_cuda, "_jax_backend", lambda: "cpu")
    with _no_warning():
        _cuda.warn_if_jax_preallocates()


@contextlib.contextmanager
def _no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.DeviceMemoryWarning)
        yield
