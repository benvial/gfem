"""gfem: differentiable finite-element electromagnetics on NGSolve and JAX.

Importing gfem turns on JAX double precision (``jax_enable_x64``). Every gfem
solve and gradient is computed in float64 / complex128, and single precision
silently ruins adjoint gradients, so this is not optional.
"""

from importlib.metadata import version as _version

import jax as _jax

_jax.config.update("jax_enable_x64", True)

# Submodules are imported after the switch, so no gfem array is ever float32.
from gfem.errors import NonDifferentiableError, RegionNotFoundError  # noqa: E402
from gfem.mesh import Mesh, Region  # noqa: E402
from gfem.model import (  # noqa: E402
    PEC,
    PMC,
    Absorbing,
    Conductor,
    Impedance,
    LumpedPort,
    Material,
    MaterialField,
    Model,
    WavePort,
    enclosed_by,
)
from gfem.options import SolverOptions  # noqa: E402
from gfem.problems import Electrostatic  # noqa: E402
from gfem.results import ElectrostaticResult  # noqa: E402
from gfem.solve import solve  # noqa: E402

__version__ = _version("gfem")

__all__ = [
    "PEC",
    "PMC",
    "Absorbing",
    "Conductor",
    "Electrostatic",
    "ElectrostaticResult",
    "Impedance",
    "LumpedPort",
    "Material",
    "MaterialField",
    "Mesh",
    "Model",
    "NonDifferentiableError",
    "Region",
    "RegionNotFoundError",
    "SolverOptions",
    "WavePort",
    "__version__",
    "enclosed_by",
    "solve",
]
