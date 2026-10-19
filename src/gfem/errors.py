"""gfem's exception and warning classes."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


class RegionNotFoundError(LookupError):
    """A region name that the mesh does not define.

    Regions are gmsh physical groups, addressed by their name. The message
    suggests the closest names the mesh does define.
    """

    def __init__(self, name: str, known: Iterable[str], kind: str = "region") -> None:
        """Build the message from the missing name and the known ones.

        Args:
            name: The name that was asked for.
            known: The names the mesh defines.
            kind: What was looked up ("region", "volume" or "surface").
        """
        known = sorted(known)
        close = difflib.get_close_matches(name, known, n=3)
        hint = (
            f" Did you mean {', '.join(repr(c) for c in close)}?"
            if close
            else f" Known {kind}s: {', '.join(repr(k) for k in known)}."
        )
        super().__init__(
            f"No {kind} named {name!r} in the mesh (gmsh physical-group names).{hint}"
        )
        self.name = name


class NonDifferentiableError(TypeError):
    """A gradient was asked of something gfem does not differentiate.

    The message names the quantity, the reason and the remedy. To use such a
    value in an objective without its derivative, wrap it in
    `jax.lax.stop_gradient`.
    """


class MeshQualityError(ValueError):
    """A moved mesh has an inverted element, so a solve on it is meaningless.

    The message names the element's region and position, and the remedies:
    take a smaller step, accept steps on `gfem.shape.quality`, parametrize
    the shape more smoothly, or remesh at the current geometry and restart.
    """


class MeshQualityWarning(UserWarning):
    """A moved mesh has elements that lost most of their imported quality.

    It fires when an element's scaled Jacobian falls below a quarter of its
    value on the imported mesh. The solve still runs, but its accuracy
    degrades and the next steps may invert elements. Filter it with Python's
    `warnings` module.
    """


class PortGeometryError(ValueError):
    """A port whose geometry the solver cannot use.

    For example a wave port surface that is not planar. The message names the
    port and the remedy.
    """


class PortModeError(RuntimeError):
    """The port mode problem has no usable mode.

    For example the requested mode is below cutoff (evanescent) at some
    frequency of the band. The message names the port, the frequency and the
    remedy.
    """


class PortModeTrackingWarning(UserWarning):
    """A wave port's mode crosses another mode inside the frequency band.

    gfem follows the physical mode (by its overlap with the neighbouring
    frequency's mode) through the crossing, so beyond it the mode is no
    longer the n-th in Palace's per-frequency order and S-parameters differ
    from Palace's there. Filter it with Python's `warnings` module.
    """


class DeviceUnavailableError(RuntimeError):
    """``SolverOptions.device`` names a GPU that this machine cannot use.

    Raised when the solve starts, never when the options are built, so a job
    written on a laptop can run on a GPU machine. gfem never falls back to
    the CPU silently. The message gives the reason and the fix: install
    ``gfem[cuda12]`` (or ``gfem[cuda13]``), the CUDA driver's error, or the
    number of devices for an out-of-range ``"cuda:N"``. Pass
    ``device="cpu"`` (``--device cpu`` on the command line) to run on the
    CPU instead.
    """


class DeviceMemoryWarning(UserWarning):
    """JAX may hold most of the GPU memory that the GPU solver needs.

    JAX on a GPU preallocates 75% of the device memory by default. Set the
    environment variable ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (or
    ``XLA_PYTHON_CLIENT_MEM_FRACTION``) before importing JAX to leave room
    for the solver's factorization.
    """


class EigenmodeConvergenceError(RuntimeError):
    """Fewer modes than ``Eigenmode.n_modes`` converged.

    gfem returns no partial result. The message names the modes that did
    not converge, their best backward error, and the remedies: move
    ``target`` closer to the wanted modes, loosen
    ``SolverOptions.eigen_tolerance`` or raise
    ``SolverOptions.max_refine_iterations``, or, for a mode with a Q below
    about 2, find the resonance from a `Driven` sweep instead.
    """


class SweepNotConvergedWarning(UserWarning):
    """An adaptive frequency sweep stopped at its sample cap before converging.

    The result is still returned: its S-parameters and fields come from the
    reduced model built on the samples solved so far, and
    ``result.sweep.converged`` is False. Raise
    ``SolverOptions.adaptive_max_samples``, loosen
    ``SolverOptions.adaptive_tolerance``, or solve a uniform sweep. Filter it
    with Python's `warnings` module.
    """
