"""Whether a CUDA GPU can be used, and the errors and warnings when it cannot.

Importing the CUDA libraries is not enough: on a machine whose GPU is absent
or in an error state they import fine and fail later. The probe initializes
the driver and starts a context on every device, once per process.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import sys
import warnings

from gfem.errors import DeviceMemoryWarning, DeviceUnavailableError

__all__ = ["Probe", "cuda_available", "ordinal", "probe", "require"]

_INSTALL = (
    "The GPU solver needs the cuda extra: pip install 'gfem[cuda12]' (CUDA 12 "
    "drivers) or 'gfem[cuda13]' (CUDA 13 drivers)."
)
_RUN_ON_CPU = "Use SolverOptions(device='cpu') or --device cpu to run on the CPU."


@dataclasses.dataclass(frozen=True)
class Probe:
    """What the CUDA probe found.

    Attributes:
        count: Usable CUDA devices.
        reason: Why none is usable: the CUDA driver's error, or ``None``
            when the cuda extra is not installed.
    """

    count: int
    reason: str | None = None


def _error_text(driver, err) -> str:
    _, name = driver.cuGetErrorName(err)
    _, text = driver.cuGetErrorString(err)
    name = name.decode() if name else str(err)
    return f"{name}: {text.decode()}" if text else name


def _query() -> Probe:
    """Initialize the CUDA driver and start a context on every device."""
    try:
        import nvmath  # noqa: F401  (the cuda extra)
    except ImportError:
        return Probe(0)
    try:
        from cuda.bindings import driver

        return _start(driver)
    except Exception as err:  # noqa: BLE001  (no driver library, a broken install)
        return Probe(0, f"{type(err).__name__}: {err}")


def _start(driver) -> Probe:
    ok = driver.CUresult.CUDA_SUCCESS
    (err,) = driver.cuInit(0)
    if err != ok:
        return Probe(0, _error_text(driver, err))
    err, count = driver.cuDeviceGetCount()
    if err != ok:
        return Probe(0, _error_text(driver, err))
    if count == 0:
        return Probe(0, "the CUDA driver reports no device")
    for i in range(count):
        err, device = driver.cuDeviceGet(i)
        if err == ok:
            err, _ = driver.cuDevicePrimaryCtxRetain(device)
        if err != ok:
            return Probe(0, f"cuda:{i}: {_error_text(driver, err)}")
        driver.cuDevicePrimaryCtxRelease(device)
    return Probe(count)


@functools.cache
def probe() -> Probe:
    """The probe's result, computed once per process."""
    return _query()


def cuda_available() -> bool:
    """Whether ``SolverOptions(device="cuda")`` can run on this machine.

    True when the cuda extra is installed and a CUDA context starts on every
    GPU. The check runs once per process. Use it to choose a device::

        device = "cuda" if gfem.cuda_available() else "cpu"
        options = gfem.SolverOptions(linear_solver="direct", device=device)

    Returns:
        Whether a CUDA GPU is usable.
    """
    found = probe()
    return found.count > 0 and found.reason is None


def ordinal(device: str) -> int:
    """The GPU number in ``"cuda"`` (0) or ``"cuda:N"`` (N)."""
    _, _, n = device.partition(":")
    return int(n) if n else 0


def require(device: str) -> int:
    """The ordinal of ``device``, after checking that it can be used.

    Raises:
        DeviceUnavailableError: The cuda extra is missing, the driver or the
            device fails, or ``"cuda:N"`` is out of range.
    """
    n = ordinal(device)
    found = probe()
    if found.count > 0 and found.reason is None:
        if n < found.count:
            return n
        raise DeviceUnavailableError(
            f"device={device!r} does not exist: this machine has {found.count} "
            f"CUDA device{'s' if found.count != 1 else ''} (cuda:0 to "
            f"cuda:{found.count - 1}). {_RUN_ON_CPU}"
        )
    if sys.platform == "darwin":
        reason = "CUDA is not supported on macOS."
    elif found.reason is None:
        reason = _INSTALL
    else:
        reason = f"CUDA cannot start: {found.reason}."
    raise DeviceUnavailableError(
        f"device={device!r} cannot run: {reason} {_RUN_ON_CPU}"
    )


def _jax_backend() -> str:
    import jax

    return jax.default_backend()


_warned = False


def warn_if_jax_preallocates() -> None:
    """Warn once per process when JAX on the GPU may starve the GPU solver."""
    global _warned  # noqa: PLW0603
    if _warned or _jax_backend() != "gpu":
        return
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() == "false":
        return
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
        return
    _warned = True
    warnings.warn(
        "JAX runs on the GPU and preallocates 75% of its memory by default, "
        "which can leave too little for the GPU solver's factorization. Set "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false (or XLA_PYTHON_CLIENT_MEM_FRACTION) "
        "in the environment before importing JAX.",
        DeviceMemoryWarning,
        stacklevel=3,
    )
