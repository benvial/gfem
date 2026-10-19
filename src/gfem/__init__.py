"""gfem: differentiable finite-element electromagnetics on NGSolve and JAX.

Importing gfem turns on JAX double precision (``jax_enable_x64``). Every gfem
solve and gradient is computed in float64 / complex128, and single precision
silently ruins adjoint gradients, so this is not optional.
"""

from importlib.metadata import version as _version

import jax as _jax

_jax.config.update("jax_enable_x64", True)

# Submodules are imported after the switch, so no gfem array is ever float32.
from gfem.errors import RegionNotFoundError  # noqa: E402
from gfem.mesh import Mesh, Region  # noqa: E402

__version__ = _version("gfem")

__all__ = ["Mesh", "Region", "RegionNotFoundError", "__version__"]
