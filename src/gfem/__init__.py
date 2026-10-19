"""gfem: differentiable finite-element electromagnetics on NGSolve and JAX.

Importing gfem turns on JAX double precision (``jax_enable_x64``). Every gfem
solve and gradient is computed in float64 / complex128, and single precision
silently ruins adjoint gradients, so this is not optional.
"""

from importlib.metadata import version as _version

import jax as _jax

_jax.config.update("jax_enable_x64", True)

# Submodules are imported after the switch, so no gfem array is ever float32.
from gfem import conventions, shape  # noqa: E402
from gfem._cuda import cuda_available  # noqa: E402
from gfem.errors import (  # noqa: E402
    DeviceMemoryWarning,
    DeviceUnavailableError,
    MeshQualityError,
    MeshQualityWarning,
    NonDifferentiableError,
    PortGeometryError,
    PortModeError,
    PortModeTrackingWarning,
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
from gfem.problems import Driven, Electrostatic, PortMode  # noqa: E402
from gfem.results import (  # noqa: E402
    DrivenResult,
    ElectrostaticResult,
    Field,
    NetworkParameters,
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
    "DeviceMemoryWarning",
    "DeviceUnavailableError",
    "Driven",
    "DrivenResult",
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
    "NetworkParameters",
    "NonDifferentiableError",
    "PortGeometryError",
    "PortMode",
    "PortModeError",
    "PortModeField",
    "PortModeResult",
    "PortModeTrackingWarning",
    "Region",
    "RegionNotFoundError",
    "SolverOptions",
    "WavePort",
    "__version__",
    "conventions",
    "cuda_available",
    "enclosed_by",
    "load_result",
    "shape",
    "solve",
]
