"""gfem's exception and warning classes."""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gfem.results import DrivenResult


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


class NonDifferentiableError(TypeError, ValueError):
    """A gradient was asked of something gfem does not differentiate.

    The message names the quantity, the reason and the remedy. To use such a
    value in an objective without its derivative, wrap it in
    `jax.lax.stop_gradient`. Fixed physics values (a problem's frequencies,
    a port's direction) raise it when JAX traces them; it is also a
    `ValueError`, so a handler for invalid values still catches those.
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

    For example a wave port surface that is not planar, or, under
    differentiation, a design that touches a wave port face (the port mode
    is a constant of the solve, so the materials beside the face must be
    too). The message names the port and the remedy.
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
    longer the n-th in the per-frequency rank order and S-parameters differ
    from those of a solver that ranks modes at each frequency. Filter it
    with Python's `warnings` module.
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


class SolverUnavailableError(RuntimeError):
    """``SolverOptions`` names PETSc or SLEPc, which this Python cannot use.

    ``linear_solver="petsc"`` needs complex petsc4py, ``eigen_solver="slepc"``
    needs slepc4py built on that complex petsc4py. Raised when the solve
    starts, never when the options are built, so a job written without PETSc
    can run where it is installed. gfem never falls back to another solver
    silently. The message gives the reason (petsc4py or slepc4py not
    importable, with the import error; a PETSc built with real scalars; a
    version older than gfem supports) and the fix: build them as the "Solve
    with PETSc and SLEPc" recipe in the gfem docs describes, or choose
    another solver.
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


class DegenerateModeError(ValueError):
    """An eigenmode gradient through one member of a degenerate cluster.

    Modes whose frequencies agree to ``SolverOptions.cluster_rtol`` form a
    cluster (``EigenmodeResult.cluster`` labels them). Inside a cluster
    each mode, and so its derivative, depends on an arbitrary choice of
    basis: only objectives symmetric in the cluster's modes (their sum or
    mean, a smooth aggregate such as a log-sum-exp in place of ``max``) are
    differentiable. Raised during the backward pass (so under `jax.grad` it
    reaches the user as ``jax.errors.JaxRuntimeError`` with this message),
    also when the cluster extends beyond the ``n_modes`` modes asked for.
    The message names the modes, their frequencies and their gap, and the
    remedies: a symmetric objective, ``n_modes`` covering the whole
    cluster, a design that breaks the symmetry, or a `Driven` surrogate.
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


class LowFrequencyWarning(UserWarning):
    """A driven frequency is below the mesh's low-frequency floor.

    The floor is k0 * h_min = 1e-6 (f * h_min = 48 Hz*m), with k0 the
    vacuum wavenumber and h_min the shortest edge of the solve domain.
    Below it the E-field formulation's rounding error grows like
    1e-16 / (k0 * h_min)^2: the fields and their gradients, and with an
    exact factorization (``linear_solver="direct"``, or ``"petsc"`` with
    its default MUMPS factorization) the capacitive part of S, lose
    digits. MKL PARDISO (``"direct"`` with the ``mkl`` extra) becomes
    unreliable below k0 * h_min = 3e-8. Above that, and down to at least
    k0 * h_min = 1e-8 with the other exact factorizations, inductance read
    from S and reciprocity are unaffected. For capacitance at low frequency solve
    `gfem.Electrostatic`; otherwise raise the lowest frequency or coarsen
    the smallest elements. The solve still runs. Filter it with Python's
    `warnings` module.
    """


class LinearSolverConvergenceError(RuntimeError):
    """A Krylov linear solve stopped above its tolerance.

    Raised by every iterative linear solve that does not converge
    (``linear_solver="bddc_gmres"``, or ``"petsc"`` with a Krylov
    ``ksp_type``) in every problem type. It is a `RuntimeError`, so
    ``except RuntimeError`` still catches it. A factorization that fails (a
    singular operator, a MUMPS or PARDISO error) raises a plain
    `RuntimeError` instead. The message names the solver, the tolerance,
    the residual reached and the remedies, such as
    ``linear_solver="direct"`` or a looser tolerance.

    A uniform `gfem.Driven` sweep stops at the first frequency that does
    not converge and keeps what it solved before it: ``result`` is the
    partial result, a `gfem.DrivenResult` over exactly the solved
    frequencies, in order, with no NaN padding, and ``unsolved`` lists the
    rest, ready to solve again with another linear solver, e.g.
    ``gfem.solve(model, gfem.Driven(err.unsolved, excite=...),
    options=gfem.SolverOptions(linear_solver="direct"))``. With frequency
    workers each block stops at its own first such frequency and the error
    holds the union of the blocks. An adaptive sweep, an eigenmode or
    electrostatic solve, and any solve under `jax.grad`, `jax.jit` or
    `jax.vmap` (forward or adjoint) return no partial result; under JAX the
    error reaches the user as ``jax.errors.JaxRuntimeError`` with this
    message.

    Attributes:
        frequency: The frequency in Hz where the solve did not converge
            (the lowest one when several frequency workers failed, the
            sample frequency in an adaptive sweep, the adjoint solve's
            frequency in a driven gradient's backward pass), or ``None``
            outside a driven solve.
        unsolved: Every frequency of a uniform sweep without a result, in
            Hz and in ``Driven.frequency`` order: the one that failed and
            those after it or, with frequency workers, each block's failing
            frequency and those after it in that block. ``None`` outside a
            uniform sweep.
        result: The partial result of a uniform sweep: S, the kept fields
            and the wave-port modes at the solved frequencies only.
            ``None`` when no frequency was solved, and outside a uniform
            sweep.
    """

    def __init__(
        self,
        message: str,
        *,
        frequency: float | None = None,
        unsolved: tuple[float, ...] | None = None,
        result: DrivenResult | None = None,
    ) -> None:
        """Hold the message and what the sweep knows about the failure.

        Args:
            message: What did not converge, and the remedies.
            frequency: The frequency in Hz where the solve did not converge.
            unsolved: The sweep's frequencies without a result, in Hz.
            result: The partial result over the solved frequencies.
        """
        super().__init__(message)
        self.frequency = frequency
        self.unsolved = unsolved
        self.result = result
