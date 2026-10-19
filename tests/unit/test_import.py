import os
import subprocess
import sys
from importlib.metadata import version

import pytest


def _run_fresh(code: str, jax_enable_x64: str | None = None) -> str:
    """Run `code` in a fresh interpreter, so JAX config set by other tests is absent."""
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    if jax_enable_x64 is not None:
        env["JAX_ENABLE_X64"] = jax_enable_x64
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("jax_enable_x64", [None, "0", "false"])
def test_importing_gfem_enables_jax_double_precision(jax_enable_x64):
    out = _run_fresh(
        "import gfem, jax.numpy as jnp;"
        "print(jnp.asarray(1.0).dtype, jnp.asarray(1j).dtype)",
        jax_enable_x64,
    )
    assert out == "float64 complex128"


def test_double_precision_holds_when_jax_was_imported_first():
    out = _run_fresh(
        "import jax, jax.numpy as jnp; jnp.zeros(1);"
        "import gfem; print(jnp.asarray(1.0).dtype)"
    )
    assert out == "float64"


def test_version_comes_from_installed_metadata():
    import gfem

    assert gfem.__version__ == version("gfem")
