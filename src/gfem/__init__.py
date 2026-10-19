"""gfem: differentiable finite-element electromagnetics on NGSolve and JAX.

Importing gfem turns on JAX double precision (``jax_enable_x64``). Every gfem
solve and gradient is computed in float64 / complex128, and single precision
silently ruins adjoint gradients, so this is not optional.
"""

from importlib.metadata import version as _version

import jax as _jax

_jax.config.update("jax_enable_x64", True)

# Submodules are imported after the switch, so no gfem array is ever float32.
from gfem import shape  # noqa: E402
from gfem.errors import (  # noqa: E402
    MeshQualityError,
    MeshQualityWarning,
    NonDifferentiableError,
    PortGeometryError,
    PortModeError,
    RegionNotFoundError,
)
from gfem.job import Job, load_result  # noqa: E402
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
from gfem.problems import Electrostatic, PortMode  # noqa: E402
from gfem.results import (  # noqa: E402
    ElectrostaticResult,
    Field,
    PortModeField,
    PortModeResult,
)
from gfem.solve import solve  # noqa: E402

__version__ = _version("gfem")

__all__ = [
    "PEC",
    "PMC",
    "Absorbing",
    "Conductor",
    "Electrostatic",
    "ElectrostaticResult",
    "Field",
    "Impedance",
    "Job",
    "LumpedPort",
    "Material",
    "MaterialField",
    "Mesh",
    "MeshQualityError",
    "MeshQualityWarning",
    "Model",
    "NonDifferentiableError",
    "PortGeometryError",
    "PortMode",
    "PortModeError",
    "PortModeField",
    "PortModeResult",
    "Region",
    "RegionNotFoundError",
    "SolverOptions",
    "WavePort",
    "__version__",
    "enclosed_by",
    "load_result",
    "shape",
    "solve",
]
