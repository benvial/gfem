"""gfem: differentiable finite-element electromagnetics on NGSolve and JAX.

Importing gfem turns on JAX double precision (``jax_enable_x64``). Every gfem
solve and gradient is computed in float64 / complex128, and single precision
silently ruins adjoint gradients, so this is not optional.
"""

from importlib.metadata import version as _version

import jax as _jax

_jax.config.update("jax_enable_x64", True)

__version__ = _version("gfem")

__all__ = ["__version__"]
