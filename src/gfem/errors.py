"""gfem's exception and warning classes."""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

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
    a port's direction) raise it when they are computed inside `jax.grad`,
    `jax.jit` or `jax.vmap`; it is also a
    `ValueError`, so a handler for invalid values still catches those.
    """


class MeshQualityError(ValueError):
    """A moved mesh has an inverted element, so a solve on it is meaningless.

    The message names the element's region and position, and the remedies:
    take a smaller step, accept steps on `gfem.shape.quality`, parametrize
    the shape more smoothly, or remesh at the current geometry and restart.
    """


#: An example value per property, for `MissingMaterialPropertyError`.
_EXAMPLES = {
    "thermal_conductivity": "1.4",  # silica
    "mass_density": "2200",
    "specific_heat": "700",
    "stiffness": "gfem.isotropic_stiffness(72e9, 0.17)",
}
#: Units of the properties outside the property table.
_UNITS = {"stiffness": "Pa, 6x6", "piezoelectric_tensor": "C/m^2, 3x6"}
#: Per physics: the problem that reads the properties, a material that gives
#: them, and where the region is left out.
_READERS = {
    "thermal": (
        "A thermal problem",
        (
            "gfem.Material(thermal_conductivity=1.4, mass_density=2200, "
            "specific_heat=700) for silica"
        ),
        "thermal problems",
    ),
    "mechanical": (
        "A piezoelectric problem",
        (
            "gfem.Material(stiffness=gfem.isotropic_stiffness(72e9, 0.17), "
            "mass_density=2200) for silica"
        ),
        "the mechanical domain (the air, for example)",
    ),
}


class MissingMaterialPropertyError(ValueError):
    """A problem reads a material property that a region does not give.

    Steady thermal problems read ``thermal_conductivity``; transient ones also
    read ``mass_density`` and ``specific_heat``; piezoelectric problems read
    ``stiffness`` and ``mass_density`` on the mechanical domain. Every volume
    of the problem's domain needs them; a region with no material lacks them
    all. The message lists every missing (region, property) pair at once,
    with each property's unit and an example value where known, and both
    remedies: give the property on the region's material, or exclude the
    region from the physics with ``Model(excluded={physics: (...)})``.

    Attributes:
        missing: The (region, property) pairs, in the order given.
        physics: The physics whose domain the regions are in, a key of
            ``excluded=``.
    """

    def __init__(
        self, missing: Iterable[tuple[str, str]], physics: str = "thermal"
    ) -> None:
        """Build the message from the missing (region, property) pairs.

        Args:
            missing: Each region and the property it does not give; repeats
                are dropped. Any property name.
            physics: ``"thermal"`` or ``"mechanical"``: the physics whose
                problem reads the properties.
        """
        from gfem._materials import PROPERTIES

        units = {p.name: p.unit for p in PROPERTIES} | _UNITS
        self.physics = physics
        self.missing: tuple[tuple[str, str], ...] = tuple(
            dict.fromkeys((str(r), str(p)) for r, p in missing)
        )
        by_region: dict[str, list[str]] = {}
        for region, prop in self.missing:
            unit = units.get(prop)
            example = _EXAMPLES.get(prop)
            detail = ", ".join(
                x for x in (unit, None if example is None else f"e.g. {example}") if x
            )
            by_region.setdefault(region, []).append(
                f"{prop} ({detail})" if detail else prop
            )
        listed = "; ".join(
            f"region {region!r}: {', '.join(props)}"
            for region, props in by_region.items()
        )
        regions = tuple(by_region)
        reader, example, where = _READERS[physics]
        super().__init__(
            f"{reader} needs material properties that are not given: "
            f"{listed}. Give each on the region's material (for example "
            f"{example}), or leave the region out of {where} with "
            f"Model(excluded={{{physics!r}: {regions!r}}})."
        )


class MeshQualityWarning(UserWarning):
    """A moved mesh has elements that lost most of their imported quality.

    It fires when an element's scaled Jacobian falls below a quarter of its
    value on the imported mesh. The solve still runs, but its accuracy
    degrades and the next steps may invert elements. Filter it with Python's
    `warnings` module.
    """


class UnsettledTraceWarning(UserWarning):
    """A trace had not settled when its rise time or time constant was measured.

    `gfem.rise_time` and `gfem.time_constant` normalise a trace to its last
    sample, so a run that ends before the trace settles biases both times
    (short). The warning fires when the change still to come after the last
    sample, estimated from the geometric tail of the last increments after
    the last waveform switch, exceeds 1 % of the measured swing, or when
    fewer than three samples follow that switch. Run longer (a larger
    ``duration`` with the same time step), or filter it with Python's
    `warnings` module.
    """


class PortGeometryError(ValueError):
    """A port whose geometry the solver cannot use.

    For example a wave port surface that is not planar, or, under
    differentiation, a design at a wave port face of an eigenmode problem
    (a `gfem.Driven` solve differentiates it), or a moved node of a lumped or
    Floquet port surface. The message names the port and the remedy.
    """


class PortModeError(RuntimeError):
    """The port mode problem has no usable mode.

    For example the requested mode is below cutoff (evanescent) at some
    frequency of the band, or an eigenmode target lies below a requested
    port mode's cutoff. The message names the port, the frequency and the
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


class LostBandWarning(UserWarning):
    """A band found inside the window has no partner at a neighbouring wave vector.

    `gfem.band_diagram` connects each wave vector's bands to the previous
    one's by field overlap. A band inside the window that continues into
    none at a neighbouring wave vector means the path is too coarse there, or
    that a solve missed a mode; the bands keep frequency order, but beyond
    that wave vector it may put one band's values in another's place.
    `gfem.BandDiagram.lost` marks them. Use more ``n_points``, a target
    nearer the bands, or more ``n_bands``. Filter it with Python's
    `warnings` module.
    """


class UnanchoredBandWindowWarning(UserWarning):
    """A band diagram's target is not below half its highest band at a wave vector.

    The bands found there may then not be the lowest, so their order is a
    rank within the window, not the band number, and a gap read there can
    be wrong. A window around a high target is a legitimate look at higher
    bands: filter the warning with Python's `warnings` module. With
    ``n_bands=1`` on a slab (at most 2 in 3D) it can fire falsely at Γ,
    where every requested band is static. `gfem.BandDiagram.anchored`
    marks the wave vectors.
    """


class SkippedPortModeWarning(UserWarning):
    """A port surface skips a port mode below its highest requested mode.

    It fires when no wave port on a surface requests a mode ranked below the
    highest ``mode`` requested there, for example a lone ``mode=2`` (mode 1
    is skipped) or modes 1 and 3 (mode 2 is skipped). That mode propagates
    wherever the higher one does, and the port partly reflects it instead of
    absorbing it, so S misses the power the device sends into it, and an
    eigenmode's Q and coupling Q miss the power it carries. Add a
    `gfem.WavePort` on the surface for each skipped mode. A lone ``mode=1``
    never warns. Filter it with Python's `warnings` module.
    """


class UnrequestedPortModeWarning(UserWarning):
    r"""A mode of an eigenmode solve that its wave ports' requested modes do not carry.

    A wave port absorbs the modes it requests exactly and any other field on
    its surface only partly (the rest reflects), so a mode carried by an
    unrequested port mode family, or trapped against the port, has a Q and a
    coupling Q that are the port's, not the device's. The solve warns once per
    wave port surface and message form, listing the modes, when
    :math:`\eta_s < 1/2` and the surface takes at least 1 % of the mode's loss
    (:math:`Q / q_s \ge 0.01`), with :math:`\eta_s` the
    `gfem.EigenmodeResult.requested_power_share`.

    Two readings, told by whether an unrequested mode of the surface
    propagates at the mode's frequency. If one does, the mode may be carried
    by it: add a `gfem.WavePort` on the surface for that mode and solve
    again; if the warning persists, the mode is trapped. If none does, the
    mode is trapped inside the model, where its port mode family propagates:
    its Q is the port's absorption of the evanescent tail, a lower bound on
    the device's, which grows as the port moves away; move the port. A share
    from 1/2 to 1 is reported on the result, never warned. Eigenmode solves
    only; a driven solve's missing modes are `SkippedPortModeWarning`'s.
    Filter it with Python's `warnings` module.
    """


class UnrequestedOrderWarning(UserWarning):
    r"""A propagating diffraction order that no Floquet port requests.

    It fires at solve time, once per Floquet surface, when an order
    propagates at some frequency of the sweep and no `gfem.FloquetPort` on
    that surface requests it in either polarization. The surface still
    absorbs the order, so nothing reflects, but the power it carries is
    missing from S (the power defect :math:`1 - \sum |S|^2` shows it). Add a
    Floquet port for the order to read it, or filter the warning with
    Python's `warnings` module.
    """


class ObliqueAbsorberWarning(UserWarning):
    r"""A first-order absorbing face of a periodic cell meets a wave obliquely.

    A `gfem.Absorbing` face reflects about
    :math:`(1 - \cos\theta) / (1 + \cos\theta)` of the amplitude of a plane
    wave meeting it at angle :math:`\theta`. In a periodic cell the
    Bloch wave vector and every propagating diffraction order meet the face
    at an angle, so the driven solve warns once per face when that
    reflection reaches 1%. A `gfem.FloquetPort` on the face absorbs every
    order at any angle, and so does a perfectly matched layer (`gfem.PML`)
    meshed beyond it, which also takes a lossy or layered substrate. Filter
    it with Python's `warnings` module.
    """


class ThinLayerWarning(UserWarning):
    r"""A perfectly matched layer is thin against the reference wavelength.

    It fires when a side of the layer has :math:`k_\text{ref} d < 2\pi/100`,
    with :math:`d` its thickness and :math:`k_\text{ref}` the vacuum
    wavenumber at its reference frequency. The stretch maximum
    :math:`\sigma_\max = -3\ln R_0/(2 k_\text{ref} d)` is then large, so the
    operator is badly conditioned and the mesh cannot follow the field's
    decay across the layer. Make the layer about half a wavelength thick, or
    filter the warning with Python's `warnings` module.
    """


class AntiBoundRootWarning(UserWarning):
    """An eigenmode root below a wave port mode's cutoff was dropped.

    An eigenmode solve with wave ports continues each port's outgoing wave
    to complex frequencies. Below the cutoff of a requested port mode that
    continuation has roots that are not resonances: real ones (anti-bound
    roots, whose field grows into the guide) and their mirrors that grow in
    time. The search drops such a root, names it in this warning, and
    returns the next mode instead. A complex root below cutoff that decays
    in time is a resonance and is kept. Filter the warning with Python's
    `warnings` module.
    """


class LayerClearanceWarning(UserWarning):
    """A mode keeps a noticeable share of its energy in a perfectly matched layer.

    It fires when a wave port's mode, or an eigenmode, holds more than about
    1% of its electric energy in the layer, or when an eigenmode grows in
    time beyond round-off: the layer sits too close to the guided field
    (too little clearance), so it absorbs or distorts the mode itself. Move
    the layer away from the structure, or filter the warning with Python's
    `warnings` module.
    """


class FloquetOrderCapWarning(UserWarning):
    """A Floquet surface's mesh cannot resolve every order it should model.

    A Floquet surface models every propagating order and the evanescent
    orders whose round trip to the nearest structure decays by less than
    ``SolverOptions.floquet_order_tolerance``, but never an order beyond what
    its face mesh resolves (the order cap). The warning names the surface and
    the distance from the structure at which the cap stops binding: move the
    face out, or refine the face mesh. Filter it with Python's `warnings`
    module.
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


class DistributedUnsupportedError(NotImplementedError):
    """This solve has no path above one MPI rank.

    Raised on every rank, before any mesh is built, so a script can catch it and
    the job goes on. A method of a kept field raises it the same way, and a
    gradient through a mode's field raises it when `jax.grad` traces the backward
    pass. Rank copies (``comm=MPI.COMM_SELF``) run every feature as one process.
    Existing one-process refusals keep their classes.

    Attributes:
        feature: What has no path, e.g. ``"adaptive_sweep"`` or ``"periodic"``.
    """

    def __init__(self, feature: str, message: str) -> None:
        """Store the key and the message, which gives the remedy."""
        super().__init__(feature, message)
        self.feature = feature
        self.message = message

    def __str__(self) -> str:
        """The message, without the key."""
        return self.message


class MeshRefinementUnsupportedError(NotImplementedError):
    """`gfem.refine_mesh` or `gfem.error_estimate` has no path for this problem.

    Raised before any solve, so a script can catch it and go on.

    Attributes:
        feature: What has no path: ``"adaptive sweep"``, ``"port mode"``,
            ``"periodic"``, ``"bloch"``, ``"floquet port"``,
            ``"perfectly matched layer"``, ``"incident field"``,
            ``"point dipole"``, or an estimator
            or problem type that this version of gfem cannot estimate yet.
    """

    def __init__(self, feature: str, message: str) -> None:
        """Store the key and the message, which gives the remedy."""
        super().__init__(feature, message)
        self.feature = feature
        self.message = message

    def __str__(self) -> str:
        """The message, without the key."""
        return self.message


class MeshRefinementNotConvergedWarning(UserWarning):
    """Mesh refinement stopped before its error estimate met the tolerance.

    It fires when `gfem.refine_mesh` stops on ``max_cycles``, on
    ``max_tetrahedra`` or on a solve that failed, with the last error
    estimate above ``tolerance``. The record is still returned: it holds the
    last solved model, its result and its estimate, and ``converged`` is
    False. Raise the cap, loosen ``tolerance``, or start from a finer mesh.
    Filter it with Python's `warnings` module.
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

    Also raised when a solve is launched on several ranks and cannot use them:
    mpi4py or an MPI library cannot be loaded, the launcher did not start MPI
    (``srun --mpi=none``), or the MPI is not Open MPI.
    """


class DeviceMemoryWarning(UserWarning):
    """JAX may hold most of the GPU memory that the GPU solver needs.

    JAX on a GPU preallocates 75% of the device memory by default. Set the
    environment variable ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (or
    ``XLA_PYTHON_CLIENT_MEM_FRACTION``) before importing JAX to leave room
    for the solver's factorization.
    """


class EigenmodeConvergenceError(RuntimeError):
    r"""Fewer modes than ``Eigenmode.n_modes`` converged.

    gfem returns no partial result. The message names the modes that did
    not converge, their best backward error, and the remedies: move
    ``target`` closer to the wanted modes, loosen
    ``SolverOptions.eigen_tolerance`` or raise
    ``SolverOptions.max_refine_iterations``, or, for a mode with a Q below
    about 2, find the resonance from a `Driven` sweep instead.

    It is also raised when the eigen solver's first pass stops after 100
    restarts (``eps_max_it`` in ``SolverOptions.petsc_options`` overrides
    the bound on SLEPc). The modes near the target are then closer together
    than their linewidth, so the nearest cannot be told apart. The message
    names the lossy region with the most modes within one linewidth of the
    target when it holds about 10 or more, and for a permittivity law its
    term nearest the target. Move ``target`` where the modes are separated,
    or find the resonances from a `Driven` sweep.

    With wave ports it is also raised when the search cannot certify that it
    listed every resonance of the strip it searches (the modes with
    :math:`\mathrm{Im}\,\omega \le \omega_t / 4`, nearest ``target``) after
    the seed solves: the strip holds more modes of other families than the
    solves can list. The message says so; lower ``n_modes`` or move
    ``target``. It is also raised when a seed's port mode cannot be followed
    to the seed's complex frequency (a port term refusal): the message lists
    each such seed on a ``lost:`` line with its frequency and the port, and,
    when it is the seed shift itself, asks to move ``target`` away from the
    port mode's cutoff.
    """


class DegenerateModeError(ValueError):
    """An eigenmode gradient through one member of a degenerate cluster.

    Modes whose frequencies agree to ``SolverOptions.cluster_rtol`` form a
    cluster (``EigenmodeResult.cluster`` labels them). Inside a cluster
    each mode, and so its derivative, depends on an arbitrary choice of
    basis: only objectives symmetric in the cluster's modes (their sum or
    mean, a smooth aggregate such as a log-sum-exp in place of ``max``) are
    differentiable. It is raised while `jax.grad` computes the gradient, also
    when the cluster extends beyond the ``n_modes`` modes asked for. Under
    `jax.grad` it arrives as ``jax.errors.JaxRuntimeError``, or as
    ``ValueError`` from a jitted function that has already run
    successfully, with this message; see the gradient recipes ("When a
    gradient raises"). The message names the
    modes, their frequencies and their gap, and the remedies: a symmetric
    objective, ``n_modes`` covering the whole cluster, a design that breaks
    the symmetry, or a `Driven` surrogate.
    """


class ModeLabelError(RuntimeError):
    """A dressed level no longer resembles one bare Fock state.

    `gfem.quantum.dressed_modes` labels each level it uses (the ground
    state, one and two photons in a mode, one photon in each of two modes)
    by its largest overlap with a bare Fock state; above 1/2 that label is
    unique. At 1/2 or less two modes have hybridized (near a resonance
    between them, or in the straddling regime), and "the qubit" and "the
    resonator" are no longer separate objects. The message names the level,
    its overlap and the two bare frequencies involved, and the remedies:
    move the modes apart, or leave the hybridized mode out of the call.
    ``method="first_order"`` is not a remedy: it gives a number for a
    picture that no longer holds. With concrete inputs it is raised as is;
    under a JAX transformation (`jax.jit`, `jax.grad`, `jax.vmap`) it
    arrives as ``jax.errors.JaxRuntimeError`` carrying this class's name
    and the message.
    """


class SolveCacheWarning(UserWarning):
    """The solve cache could not read or store this call's result.

    The returned result is correct either way; only what is on disk differs
    from what ``cache=`` asks for. The message says what happened and the
    remedy: an entry that could not be read (removed, solved again and
    stored), a result that could not be stored (a read-only or full
    directory), a result larger than the cache's ``max_bytes``, a directory
    of another cache layout, an input the cache cannot key on, or fields
    kept on more than one rank, which are not cached. Silence it
    as a whole with
    ``warnings.filterwarnings("ignore", category=gfem.SolveCacheWarning)``.
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


class SweepRankDeficientWarning(UserWarning):
    """An adaptive sweep's rational interpolation lost rank.

    The sweep is near the accuracy of its full solves. The sweep goes on.
    Tighten ``SolverOptions.tolerance`` or loosen
    ``SolverOptions.adaptive_tolerance`` if it then fails to converge. Filter
    it with Python's `warnings` module.
    """


class RationalFitNotConvergedWarning(UserWarning):
    """A rational fit reached its pole cap above its tolerance.

    `gfem.NetworkParameters.fit_rational` raised the order up to
    ``max_poles`` without its largest per-element RMS error reaching
    ``tolerance``. The most accurate model tried is still returned; its
    ``fit_error`` says how far it is. Raise ``max_poles``, fit a narrower
    band, or remove a known delay first: a long line needs about one pole
    pair per phase wrap in the band, and a wave port's reference plane can
    be moved with ``DrivenResult.with_offsets``. Filter it with Python's
    `warnings` module.
    """


class NonPassiveDataWarning(UserWarning):
    """Making a rational fit passive moved S by more than the fit tolerance.

    `gfem.NetworkParameters.fit_rational` fitted S within ``tolerance``, but
    the passivity enforcement then changed it by more: the data themselves
    are not passive on this reference (a singular value of S above 1). The
    passive model is still returned; its ``error`` is above its
    ``fit_error``. The message names the largest singular value of the data
    and where. Usual causes: a reference plane moved by
    ``DrivenResult.with_offsets`` past a lossy line, a nominal modal
    reference on a lossy mode, or a mesh too coarse for S. Filter it with
    Python's `warnings` module.
    """


class PassivityEnforcementError(RuntimeError):
    """A rational fit could not be made passive.

    The residue perturbation of `gfem.NetworkParameters.fit_rational`
    ended with S still above 1 somewhere. The message names the remaining
    intervals in Hz. Fit with ``enforce_passivity=False`` to inspect the
    model's ``passivity_violations``.
    """


class UnsampledPixelsWarning(UserWarning):
    """Some pixels of a `gfem.design.LayerGrid` are sampled by no centroid.

    No element (or face) centroid falls in them, so they get no gradient and
    do not reach the solve. Refine the mesh in the region or coarsen the
    pitch; pixels outside a non-rectangular footprint are always unsampled,
    mask them with ``grid.sampled``. Filter it with Python's `warnings`
    module.
    """


class FittedRangeWarning(UserWarning):
    """A solve reads a permittivity law outside its fitted range.

    A `gfem.PermittivityLaw` with a ``frequency_range`` (a fitted law, or a
    library material's) was matched to data or to a source's formula over
    that range only. A driven sweep, an eigenmode target or mode, a port
    mode solve, or an electrostatic solve (which reads the law at 0 Hz) that
    reads it outside the range still runs: the law stays passive everywhere,
    so the problem is well posed, but its accuracy there is unknown. One
    warning per solve names each region read outside its range, and how
    far. Narrow the frequencies, fit data that cover them, give an
    electrostatic model a constant permittivity, or clear the range with
    ``dataclasses.replace(law, frequency_range=None)`` once the law has been
    checked there; or filter the warning with Python's `warnings` module.
    """


class PermittivityFitWarning(UserWarning):
    r"""A permittivity fit stopped above its tolerance.

    Its error in :math:`\varepsilon` or in :math:`k` is above the
    tolerance, and `gfem.fit_permittivity` still returns its best law: the
    one with the smallest relative error in :math:`\varepsilon` up to its
    term cap. The message gives the error and the remedies: more Lorentz
    terms, ``drude=True`` for a metal, a narrower range, or a looser
    tolerance when the table's own scatter is that large. A :math:`k` error
    above the tolerance means the law has loss where the data have little or
    none, as below an absorption edge: fit only where the material absorbs,
    or fit the transparent part without ``k``. Filter it with Python's
    `warnings` module.
    """


class LowFrequencyWarning(UserWarning):
    r"""A driven frequency or an eigenmode is below the mesh's low-frequency floor.

    The floor is :math:`k_0 h_\mathrm{min} = 10^{-6}`
    (:math:`f h_\mathrm{min} \approx` 48 Hz·m), with :math:`k_0` the vacuum
    wavenumber and :math:`h_\mathrm{min}` the shortest edge of the solve
    domain. The solve still runs; see `gfem.Driven`, Notes, for how far each
    result holds. Filter the warning with Python's `warnings` module.

    A driven solve through the tree-cotree gauge (``SolverOptions.gauge``,
    on by default for a direct solve on the CPU in one process) keeps its
    digits below the floor and does not warn, unless the model has a
    `gfem.WavePort`. The wave ports' mode problem is not gauged: below the
    floor their modes, and S through them, carry an error that grows like
    :math:`1/f^2`, and a mode solve may report a TEM mode as evanescent
    (`gfem.PortModeError`). Use lumped ports at these frequencies, or raise
    the lowest frequency.

    An ungauged driven solve (``gauge="none"``, ``"bddc_gmres"``,
    ``"pmg_gmres"``, ``"petsc"``, ``device="cuda"``, several MPI ranks, a
    periodic or Floquet model) loses digits in its fields and gradients below the
    floor, and with an exact factorization in the capacitive part of S. The
    message ends with why the solve is ungauged and, where one exists, the
    setting that gauges it. For capacitance at low frequency solve
    `gfem.Electrostatic`; otherwise raise the lowest frequency or coarsen the
    smallest elements.

    A `gfem.Eigenmode` solve warns when its target or its lowest mode is
    below the floor: the eigenfrequencies, fields and participations lose
    the same digits, and `gfem.EigenmodeResult.error` does not show it.
    """


class LinearSolverConvergenceError(RuntimeError):
    """A Krylov linear solve stopped above its tolerance.

    Raised by every iterative linear solve that does not converge
    (``linear_solver="bddc_gmres"`` or ``"pmg_gmres"``, or ``"petsc"`` with
    a Krylov ``ksp_type``) in every problem type. It is a `RuntimeError`, so
    ``except RuntimeError`` still catches it. A factorization that fails (a
    singular operator, a MUMPS or PARDISO error) raises a plain
    `RuntimeError` instead. The message names the solver, the tolerance,
    the residual reached and the remedies, such as
    ``linear_solver="direct"`` or a looser tolerance, or, with
    ``"pmg_gmres"``, a check of the mesh for layers of thin cells.

    A uniform `gfem.Driven` sweep stops at the first frequency that does
    not converge and keeps what it solved before it: ``result`` is the
    partial result, a `gfem.DrivenResult` over exactly the solved
    frequencies, in order, with no NaN padding, and ``unsolved`` lists the
    rest, ready to solve again with another linear solver, e.g.
    ``gfem.solve(model, gfem.Driven(err.unsolved, excite=...),
    options=gfem.SolverOptions(linear_solver="direct"))``. With frequency
    workers each block stops at its own first such frequency and the error
    holds the union of the blocks. `gfem.Job.run` with ``out``, and so
    ``python -m gfem run``, write the partial result as a partial bundle
    (``partial.json`` and ``partial.npz``) and add a note saying where. An
    adaptive sweep, an eigenmode or electrostatic solve, and any solve under
    `jax.grad`, `jax.jit` or `jax.vmap` (forward or adjoint) return no
    partial result. Under
    `jax.grad` it arrives as ``jax.errors.JaxRuntimeError``, or as
    ``ValueError`` from a jitted function that has already run
    successfully, with this message; see the gradient recipes ("When a
    gradient raises").

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
        #: A missed uniform sweep's outputs, until `gfem.solve` builds
        #: ``result`` from them.
        self._partial: Any = None
