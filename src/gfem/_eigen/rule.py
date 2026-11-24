"""Eigenmode rule: resonant modes of a model in complex Nédélec elements.

The operator is the driven operator's pieces (`gfem._engine.driven`) sorted
by power of the frequency (`gfem._eigen.operator`): ``T(k) = K + k D +
k^2 M + A(k)``, ``k = ω scale / c``. Without damping the problem is linear
in ``k^2``; with lumped or impedance resistance, conductivity or a
first-order absorbing boundary it is quadratic; a `Conductor` or a
second-order absorbing boundary adds the nonpolynomial ``A(k)``, which the
seed freezes at the target. One factorization of ``T(sigma)`` at the
target serves the shift-invert Arnoldi of the seed (a retry with a wider
subspace, after refinement reshifted, first refactorizes at the target)
(`gfem._eigen.arpack` or `gfem._eigen.slepc`, chosen by
`SolverOptions.eigen_solver`, on the shared `gfem._eigen.seed` operator,
with the divergence-free projection), the Rayleigh functional and the
residual inverse iteration that refines every seed on the true ``T(k)``
(`gfem._eigen.refine`). A `WavePort` adds its surface's Robin term and
modal correction, evaluated at the complex wavenumber from its port mode
sweep around the target (`gfem._engine.eigen.eigen_ports`); a root below a
requested port mode's cutoff that does not decay is dropped with
`AntiBoundRootWarning`. Everything that touches
NGSolve (the assembly of ``T(k)``, the divergence-free projection, the
energy per element) is in `gfem._engine.eigen`.

Gradients (``omega``, so frequency and Q) need no solve: a simple mode's
``dk/dp = -x^T (dT/dp) x / x^T T'(k) x``, ``T'(k) = D + 2 k M + Σ θ_p'(k)
A_p``, at the refined pair; ``dT/dp`` is a material, sheet-fill or
node-coordinate derivative of the driven operator's pieces (the shape
terms are the driven rule's, port surfaces held). The modes of a
degenerate cluster (frequencies within ``SolverOptions.cluster_rtol``) are
an arbitrary basis of their eigenspace: only an objective symmetric in them
differentiates, through the basis-invariant
``d Σ k = -tr((X^T T' X)^-1 X^T (dT/dp) X)``. A differentiated solve
refines one guard mode beyond ``n_modes`` from the seed solve's remaining
pairs, so a cluster cut by the request is detected when its next member is
among them (best-effort: no Arnoldi attempt is added for it).

The ratios of the mode fields (``q_port``, ``participation``, the region,
principal and sheet shares) differentiate for isolated modes with
respect to materials, sheet fills and node coordinates (port surfaces
held), and the normalized mode fields themselves (through the
normalization multipliers of `gfem._eigen.ratios`) with respect to
materials and sheet fills, never the node coordinates; on one CPU process
with the ``direct`` linear solver: per mode with a cotangent, one deflated
GMRES solve of the singular system of the mode-field adjoint, ``T(k) lam0
= -w``, ``t^T lam0 = 0``, on the live factors the forward kept
(`gfem._eigen.adjoint`), the load ``(w, kappa, explicit terms)``
of `gfem._eigen.ratios`, and the pairs
``(2 lam, x)`` and ``(conj(x), x)`` on the mode energy's ``Re(eps_r)``
terms and the sheets' L and C pieces. What does not
differentiate is refused by name: at trace time from the model, the
options and the wanted inputs (`EigenmodeRule.nondifferentiable_output`),
in the backward from the modes (a cluster, a near pair, a defective mode,
an infinite ``q_port``).

Post-processing: electric energy ``E = 1/2 Re ∫ D* . E`` per
volume region plus ``C |V|^2 / 2`` per capacitive lumped port and ``1/2 ∫
w C_s |E_t|^2`` per capacitive `Impedance` sheet (the quadratic form of
its C piece); a port's voltage is its driven voltage functional.
Each mode is scaled to ``E = 1 J`` with ``∫ D . E`` real and positive. An
anisotropic region's energy is also split along its principal axes (``1/2
ε0 eps_k ∫ |a_k . E|^2``).

On a model with `gfem.Periodic` pairs the modes are solved on the periodic
space and their fields expanded to the unpaired space (``e = P x``,
`DrivenOperator.expand`); the backward pass's pairs are expanded too
(``P X``, and ``conj(P) Y`` on the left). With ``Eigenmode.wave_vector``
(Bloch) ``T(k)`` is general: the refinement is two-sided, its
left eigenvectors ``Y`` replace ``X`` on the left of every gradient (``G =
Y^T T'(k_c) X``), only the ``direct`` linear solver runs it (transposed
solves on its factors), the real snap of a lossless cell tests that ``K``
and ``M`` are Hermitian, and the phase of a mode makes its largest
solve-space coefficient real and positive (``∫ E . E`` over a Bloch cell can
vanish).
"""

from __future__ import annotations

import cmath
import dataclasses
import functools
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem import _mpi, _petsc, _timing
from gfem._boundaries import (
    FloquetPort,
    Impedance,
    LumpedPort,
    WavePort,
    wave_port_surfaces,
)
from gfem._eigen import arpack, slepc, velocity
from gfem._eigen.adjoint import deflated_solve, reshift_offset
from gfem._eigen.kept import KeptOperator
from gfem._eigen.operator import EigenOperator
from gfem._eigen.ratios import (
    Ratios,
    energy_name,
    sheet_capacitors,
    sheet_inductors,
)
from gfem._eigen.reach import FIELD_TOLERANCE, ModeReach, blocker
from gfem._eigen.refine import Refined
from gfem._eigen.search import FoundModes, ModeSearch
from gfem._eigen.share import Unrequested, requested_power_shares
from gfem._engine import dispersion, terms
from gfem._engine.cache import Entry, discretization
from gfem._engine.coefficients import anisotropic_regions
from gfem._engine.dispersion import energy_eps
from gfem._engine.domain import em_region
from gfem._engine.driven import (
    LOW_FREQUENCY_FLOOR,
    DrivenOperator,
    LowFrequencyFloor,
    PortDesign,
    driven_operator,
    lumped_terms,
    operator_terms,
    owned_dofs,
)
from gfem._engine.eigen import (
    DivFreeProjector,
    EigenPort,
    eigen_operator,
    eigen_ports,
    element_energy,
    law_scale,
    layer_energy,
    principal_energy,
    projector,
    scale_per_tetrahedron,
)
from gfem._engine.fields import dof_shape, fe_space, one_process_dofs, rank_piece
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import FieldPair, Term, is_lumped, is_thermal_value
from gfem._engine.solve_request import SolveRequest
from gfem._linalg import Kind, ResolvedLinearSolver
from gfem._linalg.layout import ONE_PROCESS, DofLayout, RankLayout
from gfem._materials import Material, PermittivityLaw
from gfem._periodic import check_bloch, check_wave_vector, pair_names
from gfem._pml import Stretch, physical_regions, wave_ports_in_layer
from gfem._pml import reference_frequency as layer_reference_frequency
from gfem._pml import stretch as layer_stretch
from gfem._sign import largest_entry, largest_value
from gfem._text import hertz
from gfem.conventions import MODE_ENERGY
from gfem.errors import (
    AntiBoundRootWarning,
    DegenerateModeError,
    EigenmodeConvergenceError,
    LayerClearanceWarning,
    LinearSolverConvergenceError,
    LowFrequencyWarning,
    SolverUnavailableError,
    UnrequestedPortModeWarning,
)
from gfem.mesh import Mesh, _at_imported_nodes
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Eigenmode
from gfem.results import EigenmodeResult, Field

__all__ = ["DEFAULT_SOLVER", "EigenmodeRule"]

#: The linear solver for ``linear_solver="auto"``: shift-invert wants exact
#: inner solves.
DEFAULT_SOLVER: Kind = "direct"
#: Inner solves of an iterative backend run this much tighter than the
#: eigen tolerance.
_INNER = 1e-2
#: ... but never below this: the true preconditioned residual of the
#: shift-invert operators stays near 1e-13, a double-precision limit of the
#: operator, and BDDC + GMRES ends within 0.1 % of 1e-14, too thin a margin
#: to ask for. It also bounds PETSc Krylov inner solves, whose ``ksp_rtol``
#: floor is no lower.
_INNER_FLOOR = 1e-13
#: A returned mode of a model with a perfectly matched layer whose
#: ``-Im k`` exceeds this fraction of ``|k|`` grows in time (Q below 5e5):
#: the stretch acting on an evanescent tail that reaches the layer
#: (measured 3e-3 for a mode trapped 3 mm in front of a layer). Below it
#: the gain cannot be told from the layer's discretization error.
_GAIN = 1e-6
#: The modal overlap from which a seed solve's error names the region: every
#: case below it converged in 47 shift solves or fewer, every hang had 70 or
#: more.
_CROWDED = 10.0
#: Cotangents on the members of a cluster that agree to this (relative to
#: the largest) make an objective symmetric in the cluster to first order
#: (a smooth symmetric function's cotangents differ by O(gap)).
_SYMMETRIC_RTOL = 1e-3
#: A Bloch cluster's ``G = Y^T T' X`` with a condition number above this is
#: singular: a defective eigenvalue, whose frequency has no derivative.
_DEFECTIVE = velocity.DEFECTIVE
#: A mode with ``|y^T T'(k) x|`` below this times ``|y| |T'(k) x|`` is
#: defective: the singular system of its mode-field adjoint has no solution.
_DEFECTIVE_FIELD = 1e-8
#: The outputs that are quantities of the mode fields (mode-field outputs),
#: with their names on `EigenmodeResult` for messages.
_MODE_FIELD = {
    "q_port": "q_port",
    "participation": "participation",
    "bulk": "bulk_participation",
    "principal": "principal_participation",
    "sheet": "sheet_participation",
    "e_field": "e_fields (the normalized mode fields)",
}


@dataclasses.dataclass(frozen=True)
class _InputRefusal:
    """Why a mode-field output does not differentiate by a wanted input.

    Attributes:
        kind: ``"nodes"`` (no node gradient of the normalized field),
            ``"lumped"`` (a lumped value the output has no gradient by yet)
            or ``"law"`` (a permittivity law's parameter: the mode-field
            outputs read the mode energy).
        message: The reason, starting "with respect to".
    """

    kind: Literal["nodes", "lumped", "law"]
    message: str


@dataclasses.dataclass(frozen=True)
class DualRequest:
    """A participation dual `gfem.refine_mesh` asks the forward to solve.

    Attributes:
        start: The mode's index on the first mesh (the key it is kept by).
        omega: Its omega on the previous mesh, rad/s: the forward takes the
            mode nearest it.
        regions: The regions whose bulk participation the objective reads.
    """

    start: int
    omega: complex
    regions: tuple[str, ...]


@dataclasses.dataclass(frozen=True, eq=False)
class ParticipationDual:
    """The mode-field dual of one region's bulk participation.

    Attributes:
        mode: The mode's index on the result it was solved for.
        omega: The mode's omega, rad/s.
        lam: ``lam = lam0 - kappa u / (u^T t)`` on the solve space, for the
            result's normalized mode ``u``.
        weights: The load ``w = sum_r weights[r] conj(A_r u)``, ``A_r`` the
            region's ``Re eps_r`` mass form (`energy_parts`), by region.
        applies: The deflated solve's operator applies.
    """

    mode: int
    omega: complex
    lam: np.ndarray
    weights: dict[str, float]
    applies: int


class EigenmodeRule:
    """Host-side forward of an `Eigenmode` problem on a `Model`."""

    physics = "em"
    #: The operator family (a key of `sensitivity.PROPERTIES`).
    form = "eigenmode"
    #: Material properties the eigenmode forms read.
    reads = frozenset({"eps_r", "mu_r", "tan_delta", "conductivity"})
    #: Outputs whose cotangents the backward pass can handle: the eigenvalues
    #: (``cluster`` holds integer labels: its cotangent is always zero) and
    #: the mode-field outputs, within the reach `nondifferentiable_output`
    #: checks for the wanted inputs.
    differentiable_outputs = frozenset({"omega", "cluster", *_MODE_FIELD})
    #: Outputs that are a rank's own piece under ranks: the ranks
    #: compare their cotangents' shapes, not their values. The mode fields'
    #: ratios are values.
    pieces = frozenset({"e_field"})
    #: `nondifferentiable_output` takes the wanted input kinds and may
    #: return ``None`` (`gfem._adjoint.solve_vjp`).
    input_aware = True

    def __init__(
        self,
        model: Model,
        problem: Eigenmode,
        options: SolverOptions,
        *,
        comm: Any = None,
    ) -> None:
        """Check that the problem fits the model (see `gfem.Eigenmode`).

        ``comm`` is the resolved communicator of a distributed solve: its seed
        solver is SLEPc, also for ``eigen_solver="auto"``.

        Raises:
            NotImplementedError: A feature this version cannot solve.
            ValueError: ``wave_vector`` is set without a periodic pair, or
                outside the span of the pairs' translations, or with a
                linear solver other than ``"direct"``.
            gfem.SolverUnavailableError: ``linear_solver="petsc"`` or
                ``eigen_solver="slepc"`` names a library that cannot be used,
                or under ``comm`` the MPI PETSc stack with SLEPc is missing.
        """
        if not isinstance(problem, Eigenmode):
            raise TypeError(f"Expected gfem.Eigenmode, got {problem!r}.")
        self.problem = problem
        self.options = options
        self.comm = comm
        # before the Bloch refusals: a wave vector without a pair is the
        # first thing to fix
        check_wave_vector(model, problem.wave_vector, "Eigenmode")
        if problem.wave_vector is not None:
            check_bloch(
                "Eigenmode",
                device=options.device,
                linear_solver=options.linear_solver,
                solvers=("direct",),
                reason=(
                    "cannot solve the transposed system that two-sided "
                    "refinement solves on the shift's factors"
                ),
            )
        # before any assembly
        self.linear = ResolvedLinearSolver.resolve(
            options, default=DEFAULT_SOLVER, problem="eigenmode", comm=comm
        )
        if comm is not None:
            _require_rank_slepc(comm)
        elif options.eigen_solver == "slepc":
            _petsc.require_slepc("eigen_solver")
        _check_supported(model)
        _check_wave_ports(model, problem)
        lumped = [p for p in model.ports if isinstance(p, LumpedPort)]
        #: Ports with a resistance (coupling Q) and with an inductance
        #: (participation), in model order.
        self.resistive = tuple(p.name for p in lumped if p.rlc[0] is not None)
        #: The ports with a coupling Q (``q_port``), in model order: the
        #: resistive lumped ports and the wave ports.
        self.coupled = tuple(
            p.name
            for p in model.ports
            if isinstance(p, WavePort) or p.name in self.resistive
        )
        self.inductive = tuple(p.name for p in lumped if p.rlc[1] is not None)
        #: The wave port surfaces (``requested_power_share``), in the order of
        #: each surface's first wave port.
        self.port_surfaces = tuple(wave_port_surfaces(model.ports))
        #: The surfaces of the `Impedance` sheets with an inductance (sheet
        #: participation), in model order.
        self.kinetic = tuple(
            b.surface
            for b in model.boundaries
            if isinstance(b, Impedance) and b.inductance is not None
        )
        mesh, active = model.mesh, model._domain.active  # noqa: SLF001
        #: Volume regions with tetrahedra in the solve domain: the field support.
        self.regions = tuple(r for r in mesh.volumes if active[mesh.elements(r)].any())
        #: The regions of the field support outside a perfectly matched layer:
        #: the energies and participations, whose element arrays hold the
        #: layer's background, not its stretch.
        self.physical = physical_regions(self.regions, model.pml)
        #: The regions with an anisotropic material: the principal split.
        self.anisotropic = anisotropic_regions(model, self.physical)
        self._pml = model.pml
        #: The layer's reference frequency: pinned, else the target.
        self._reference_frequency = (
            None
            if model.pml is None
            else layer_reference_frequency(model.pml, [problem.target])
        )
        self._mesh = model.mesh
        self._materials = dict(model.materials)
        self._periodic_surfaces = pair_names(model._periodic)  # noqa: SLF001
        self._excluded = model._domain.excluded  # noqa: SLF001
        #: Why the node coordinates cannot be differentiated, or None.
        self.nodes_unsupported = terms.absorbing_shape_unsupported(model)
        #: What the wave ports' modes read, which the design must stay off.
        self.port_design = PortDesign.of(model)
        #: The last backward's explicit terms in the lumped ports' values.
        self._explicit: dict[tuple, float] = {}
        #: Whether a region holds a permittivity law: only then can a
        #: converged pair be a longitudinal mode (`ModeSearch._longitudinal`).
        self._laws = bool(dispersion.law_regions(model))

    # --- forward -----------------------------------------------------------------

    def forward(  # noqa: PLR0915 - one search between its target and mode checks
        self, model: Model, *, keep: bool, request: SolveRequest | None = None
    ) -> tuple[dict[str, object], Entry | None]:
        """Find the modes.

        Args:
            model: The model, with concrete leaves.
            keep: Keep what the backward pass needs (the modes, their
                ``x^T T' x`` per cluster, whether a cluster reaches the
                guard mode, refined beyond ``n_modes`` for this, each mode's
                gap, and the eigen operator as the search leaves it, in a
                `KeptOperator`, instead of freeing it).
            request: Participation duals (`DualRequest`) for
                `gfem.refine_mesh` to solve on the forward's live factors;
                the duals solved land in its ``adjoints`` by ``(start,
                region)``.

        Returns:
            ``{"omega": (n,) complex rad/s, "error": (n,), "share":
            (n_surfaces, n), "q_port":
            (n_coupled, n), "participation": (n_inductive, n), "bulk":
            (n_regions, n), "principal": (n_anisotropic, n, 3), "sheet":
            (n_kinetic, n), "e_field": (ndof, n) complex, "cluster": (n,)
            int}``, modes nearest the
            target first, and the kept `Entry` (``None`` unless ``keep``).

        Raises:
            EigenmodeConvergenceError: Fewer than ``n_modes`` modes
                converged, the modes found would skip a pair that nearly
                converged, or an isolated mode's polish did not converge.
            LinearSolverConvergenceError: A shift solve of an iterative
                linear solver did not converge (the message names
                ``SolverOptions.eigen_tolerance``), or a polish step's
                deflated solve was not accepted.
        """
        request = SolveRequest() if request is None else request
        opts = self.options
        order = opts.order
        comm = self.comm
        # before any assembly: a target in a law's unresolved band
        with _mpi.raised_together(ValueError):
            dispersion.check_band(model, self.problem.target, "target")
        # the modes below repeat no region warned for here
        warned = dispersion.warn_fitted_range(
            model, [self.problem.target], "eigenmode target", comm=comm
        )
        with task_manager(opts.threads, comm):
            stretch = self._stretch(model, warn=True)
            with _timing.phase("discretize") as span:
                op = driven_operator(
                    model,
                    order,
                    wave_vector=self.problem.wave_vector,
                    comm=comm,
                    stretch=stretch,
                )
                built = op.built
                sigma = terms.wavenumber(2 * math.pi * self.problem.target, built.scale)
                eop, ports = self._shifted(model, op, sigma, stretch)
                span.set(ndof=owned_dofs(op))
            divfree = None
            entry = None
            try:
                eop.factorize()  # before the projector's setup
                divfree = projector(model, op, self.problem.wave_vector, stretch)
                # the whole search is one solve: its shift solves are internal
                with (
                    _timing.phase("solve", iterations=()),
                    _mpi.raised_together(EigenmodeConvergenceError),
                ):
                    found = self._search_on(model, op, ports).find(
                        eop,
                        divfree,
                        sigma,
                        guard=keep or self.polishable,
                        polish_isolated=self.polishable,
                    )
                modes = found.modes
                per_hz = const.c / built.scale / (2 * math.pi)
                for j, mode in enumerate(modes):
                    f = mode.k.real * per_hz
                    with _mpi.raised_together(ValueError):
                        dispersion.check_band(model, f, f"mode {j} at")
                        # the mode energy's weight
                        dispersion.check_energy(model, f, f"mode {j} at")
                dispersion.warn_fitted_range(
                    model,
                    [m.k.real * per_hz for m in modes],
                    "eigenmode",
                    comm=comm,
                    noun="modes",
                    skip=warned,
                )
                outputs, x = self._outputs(model, op, divfree, modes, ports)
                outputs["v_g"] = self._band_velocity(
                    model, op, eop, found=found, x=x, stretch=stretch
                )
                ks = [m.k for m in modes]
                share, loss = requested_power_shares(
                    ports, op.space, x, ks, scale=built.scale, pair=_layout(op).pair
                )
                outputs["share"] = share
                self._warn_modes(found, built.scale)
                self._warn_unrequested(ports, share, loss, ks, built.scale)
                lowest = min(k.real for k in ks) * const.c / built.scale
                _check_low_frequency(
                    model,
                    self.problem.target,
                    lowest / (2 * math.pi),
                    stacklevel=2,
                    comm=comm,
                )
                outputs["cluster"] = found.reach.labels
                if request.adjoint_loads:
                    self._ride_duals(
                        model,
                        op,
                        eop,
                        x=x,
                        ks=ks,
                        found=found,
                        loads=cast("Sequence[DualRequest]", request.adjoint_loads),
                        into=request.adjoints,
                    )
                if keep:
                    y = None
                    if eop.general:
                        # every pair of a general operator carries its left one
                        y = np.stack(
                            [m.x if m.y is None else m.y for m in modes], axis=1
                        )
                    # the backward maps vectors with the operator as it is
                    # left: the entry's holder owns it from here
                    store = None if op.fixed is None else op.fixed.store
                    entry = self._entry(
                        KeptOperator(eop, store, ports),
                        ks,
                        x,
                        y,
                        reach=found.reach,
                    )
            except LinearSolverConvergenceError as err:
                raise self._inner_failure(err) from err
            finally:
                if entry is None:
                    eop.free_handle()
                if divfree is not None:
                    divfree.free()
        return outputs, entry

    def _band_velocity(
        self,
        model: Model,
        op: DrivenOperator,
        eop: EigenOperator,
        *,
        found: FoundModes,
        x: np.ndarray,
        stretch: Stretch | None,
    ) -> np.ndarray:
        r"""``(n, 3)`` complex :math:`\partial\omega/\partial\mathbf{k}`, m/s.

        NaN without a wave vector (the result then holds ``None``) and where the
        phase template does not stand for ``P``: under ranks, or when its
        build check failed. Snapped real for a mode whose ``omega`` is.
        """
        n = len(found.modes)
        nan = np.full((n, 3), np.nan + 0j)
        if self.problem.wave_vector is None:
            return nan
        if op.expansion is None or op.expansion.lattice_vectors is None:
            return nan
        plain = driven_operator(
            model, self.options.order, paired=False, stretch=stretch
        )
        y = np.stack([m.x if m.y is None else m.y for m in found.modes], axis=1)
        v = velocity.group_velocity(
            op, plain, eop, ks=np.array(found.ks), x=x, y=y, gaps=found.gaps
        )
        for j, k in enumerate(found.ks):
            if k.imag == 0:
                v[j] = v[j].real
        return v

    def _ride_duals(
        self,
        model: Model,
        op: DrivenOperator,
        eop: EigenOperator,
        *,
        x: np.ndarray,
        ks: list[complex],
        found: FoundModes,
        loads: Sequence[DualRequest],
        into: dict[tuple[int, str], ParticipationDual],
    ) -> None:
        """Solve the participation duals of ``loads`` on the live factors.

        Each request takes the mode nearest its previous omega, as
        `gfem.refine_mesh` follows it; a mode out of mode-field reach
        (``found.reach.isolated``) gets none, and the estimate raises for it.
        ``found`` holds the gap set and the modes' reach
        (`ModeSearch.find`). The duals land in ``into`` by
        ``(start, region)``.
        """
        gaps = found.gaps
        reach = found.reach.isolated
        per_k = const.c / op.built.scale
        omegas = np.array(ks, dtype=complex) * per_k
        for request in loads:
            j = int(np.argmin(np.abs(omegas - request.omega)))
            if not reach[j]:
                continue
            duals = self.participation_duals(
                model,
                op,
                eop,
                u=x[:, j],
                k=ks[j],
                gap=gaps.gap(ks[j]),
                mode=j,
                regions=request.regions,
            )
            for region, dual in duals.items():
                into[request.start, region] = dual

    def participation_duals(
        self,
        model: Model,
        op: DrivenOperator,
        eop: EigenOperator,
        *,
        u: np.ndarray,
        k: complex,
        gap: float,
        mode: int,
        regions: Sequence[str],
    ) -> dict[str, ParticipationDual]:
        """The dual of each region's bulk participation of mode ``(k, u)``.

        Per region, the load of a unit cotangent on its participation
        (`Ratios.load`), the deflated solve ``T(k) lam0 = -w``, ``t^T lam0 =
        0`` on ``eop``'s factors (`deflated_solve`), and ``lam = lam0 - kappa
        u / (u^T t)``: the backward's own adjoint. ``u`` is the mode on the
        solve space at any scale (the ratios are invariant). The load's
        volume weights go with it, for the dual flux's equilibration.
        """
        ratios = Ratios(
            model,
            op,
            regions=self.physical,
            anisotropic=self.anisotropic,
            resistive=self.resistive,
            inductive=self.inductive,
            kinetic=self.kinetic,
        )
        ut = u @ eop.derivative(k, u)
        omega = complex(k * const.c / op.built.scale)
        out = {}
        for region in regions:
            unit = np.zeros(len(self.physical))
            unit[self.physical.index(region)] = 1.0
            load = ratios.load(u, k, {"bulk": unit}, mode=mode)
            solved = deflated_solve(eop, u, None, -load.w, k=k, gap=gap, mode=mode)
            weights = {
                r: load.energy[energy_name(r)]
                for r in self.physical
                if energy_name(r) in load.energy
            }
            out[region] = ParticipationDual(
                mode=mode,
                omega=omega,
                lam=solved.z - load.kappa * u / ut,
                weights=weights,
                applies=solved.applies,
            )
        return out

    def _inner_failure(
        self, err: LinearSolverConvergenceError
    ) -> LinearSolverConvergenceError:
        """``err`` of a shift solve, with the remedy that acts on this path.

        The linear solver's message offers "a looser tolerance", but the
        inner solves run at `_inner_tolerance`, set by
        ``SolverOptions.eigen_tolerance`` unless ``tolerance`` is tighter.
        """
        remedy = (
            "a looser SolverOptions.eigen_tolerance (the inner solves run at "
            "min(tolerance, 0.01 * eigen_tolerance), here "
            f"{self._inner_tolerance():g})"
        )
        message = str(err)
        if "a looser tolerance" in message:
            message = message.replace("a looser tolerance", remedy)
        else:
            message = f"{message} For an eigenmode solve, try {remedy}."
        return LinearSolverConvergenceError(message)

    def _stretch(self, model: Model, *, warn: bool = False) -> Stretch | None:
        """The layer's stretch for this solve: at the target unless pinned.

        With ``warn``, a thin layer warns (`gfem._pml.Stretch.warn_thin`).
        """
        stretch = layer_stretch(model, [self.problem.target])
        if warn and stretch is not None:
            stretch.warn_thin()
        return stretch

    def _search_on(
        self, model: Model, op: DrivenOperator, ports: Sequence[EigenPort] = ()
    ) -> ModeSearch:
        """`search` on ``op``: it drops the modes of a perfectly matched layer.

        By their layer and total energies (`layer_energy` of ``op``, summed
        across the ranks: `ModeSearch.layer_energy`). With wave port
        surfaces ``ports`` it drops the roots below a requested port mode's
        cutoff that do not decay (`ModeSearch.below_cutoff`). Without a
        layer or ports, `search` as it is.
        """
        search = self.search()
        if ports:
            search = dataclasses.replace(search, below_cutoff=_below_cutoff(ports))
        energy = layer_energy(model, op)
        if energy is None:
            return search
        comm = self.comm
        if comm is None:
            return dataclasses.replace(search, layer_energy=energy)
        return dataclasses.replace(
            search, layer_energy=lambda x: _mpi.reduce_bcast(comm, energy(x))
        )

    def _warn_gain(self, ks: list[complex], scale: float) -> None:
        """Warn `LayerClearanceWarning` for each mode that grows in time.

        On a model with a perfectly matched layer, ``Im k < 0`` beyond
        `_GAIN` comes from the stretch acting on an evanescent tail that
        reaches the layer, not from the device. Never without a layer.
        """
        if self._pml is None:
            return
        regions = ", ".join(map(repr, self._pml.regions))
        per_hz = const.c / scale / (2 * math.pi)
        for j, k in enumerate(ks):
            if k.imag < -_GAIN * abs(k):
                q = abs(k) / (2 * abs(k.imag))
                _mpi.warn(
                    self.comm,
                    f"Eigenmode: mode {j} at {hertz(k.real * per_hz)} grows in "
                    f"time (Im omega < 0, Q = {q:.4g} with the gain sign) on a "
                    "model with a perfectly matched layer "
                    f"({regions}): its evanescent "
                    "tail most likely reaches the layer, which then pumps it "
                    "instead of absorbing it. Its Q is not physical. Move the "
                    "layer farther from the structure (more clearance).",
                    LayerClearanceWarning,
                    stacklevel=4,
                )

    def _shifted(
        self, model: Model, op: DrivenOperator, sigma: float, stretch: Stretch | None
    ) -> tuple[EigenOperator, tuple[EigenPort, ...]]:
        """`operator` at ``sigma`` with the model's wave port surfaces, and those.

        Each surface's port mode sweep runs here, over its band around the
        target (`gfem._engine.eigen.eigen_ports`).
        """
        ports = eigen_ports(
            model,
            op,
            target=self.problem.target,
            tolerance=self.options.port_mode_tolerance,
            stretch=stretch,
        )
        return self.operator(op, sigma, ports=ports), ports

    def _warn_modes(self, found: FoundModes, scale: float) -> None:
        """`_warn_gain` for the modes found, and `_warn_anti_bound`."""
        self._warn_gain(found.ks, scale)
        self._warn_anti_bound(found.anti_bound, scale)

    def _warn_unrequested(
        self,
        ports: Sequence[EigenPort],
        share: np.ndarray,
        loss: np.ndarray,
        ks: Sequence[complex],
        scale: float,
    ) -> None:
        """Warn `UnrequestedPortModeWarning` once per surface and form."""
        q = np.array([abs(k) / (2 * abs(k.imag)) if k.imag else math.inf for k in ks])
        for i, port in enumerate(ports):
            flagged = Unrequested(port, share[i], loss[i])
            if not len(flagged.flagged):
                continue
            name = port.term.ports[0].surface
            for message in flagged.messages(name, ks, scale, q):
                _mpi.warn(self.comm, message, UnrequestedPortModeWarning, stacklevel=4)

    def _warn_anti_bound(self, roots: Sequence[complex], scale: float) -> None:
        """Warn `AntiBoundRootWarning` once for the roots dropped below cutoff."""
        if not roots:
            return
        per_hz = const.c / scale / (2 * math.pi)
        listed = ", ".join(hertz(k.real * per_hz) for k in roots)
        _mpi.warn(
            self.comm,
            f"Eigenmode: dropped {len(roots)} root(s) at {listed}, below the "
            "cutoff of a wave port's mode: there the port's outgoing wave grows "
            "into its guide (a real, anti-bound root) or the root "
            "grows in time, so it is not a resonance. The modes returned are "
            "the next nearest the target.",
            AntiBoundRootWarning,
            stacklevel=4,
        )

    @property
    def polishable(self) -> bool:
        """Whether the forward polishes its isolated modes.

        On the CPU with ``linear_solver="direct"``, without a periodic pair
        or a Bloch wave vector: where mode-field gradients reach.
        """
        return (
            blocker(
                self.options,
                self.linear.kind,
                wave_vector=self.problem.wave_vector,
                periodic=bool(self._periodic_surfaces),
                tolerance=False,
            )
            is None
        )

    def search(self) -> ModeSearch:
        """The mode search of this problem and options.

        Its eigen solver is ``SolverOptions.eigen_solver``'s
        ``shift_invert``; above one rank ``"auto"`` is SLEPc.
        """
        opts = self.options
        if opts.eigen_solver == "slepc" or self.comm is not None:
            solver = functools.partial(slepc.shift_invert, options=opts.petsc_options)
        else:
            solver = arpack.shift_invert
        return ModeSearch(
            n_modes=self.problem.n_modes,
            target=self.problem.target,
            eigen_tolerance=opts.eigen_tolerance,
            max_refine_iterations=opts.max_refine_iterations,
            cluster_rtol=opts.cluster_rtol,
            laws=self._laws,
            eigen_solver=solver,
            crowding=self._crowding,
        )

    def _crowding(self) -> str | None:
        """The seed-bound error's line on a region crowded with modes, or ``None``.

        Names the lossy region with the largest modal overlap at the target
        (`_modal_overlap`) when it is `_CROWDED` or more, and for a law
        region its term nearest the target: a diagnosis only.
        """
        target = self.problem.target
        overlaps = {
            region: _modal_overlap(self._mesh, region, material, target)
            for region, material in self._materials.items()
            if region in self.regions and isinstance(material, Material)
        }
        if overlaps:
            region = max(overlaps, key=lambda r: overlaps[r])
            if overlaps[region] >= _CROWDED:
                eps_r = self._materials[region].eps_r
                term = (
                    f" ({dispersion.nearest_term(eps_r, target)})"
                    if isinstance(eps_r, PermittivityLaw)
                    else ""
                )
                return (
                    f"Region {region!r} holds about {overlaps[region]:.0f} modes "
                    f"within one linewidth of the target{term}."
                )
        return None

    def operator(
        self, op: DrivenOperator, sigma: float, *, ports: Sequence[EigenPort] = ()
    ) -> EigenOperator:
        """The eigen operator ``T(k)`` of the driven operator, shifted at ``sigma``.

        With the terms of the wave port surfaces ``ports``
        (`gfem._engine.eigen.eigen_ports`). Nothing is factorized until the
        handle is first used (`gfem._engine.eigen.eigen_operator`).
        """
        return eigen_operator(
            op,
            sigma,
            linear=self.linear,
            tolerance=self._inner_tolerance(),
            numbers=self._start_numbers(op),
            ports=ports,
        )

    def _numbers(self) -> np.ndarray | None:
        """Each field row's one-process number under ranks; ``None`` on one process.

        `gfem._engine.fields.one_process_dofs`, memoized per topology: the rows
        of a result field, which are the solve space's without periodic pairs.
        """
        if self.comm is None:
            return None
        return one_process_dofs(
            self._mesh, self.regions, self._excluded, self.options.order, self.comm
        )

    def _start_numbers(self, op: DrivenOperator) -> np.ndarray | None:
        """A number per solve-space DOF for the norm start; ``None`` on one process.

        Without periodic pairs each DOF's one-process number (`_numbers`). On a
        periodic space, whose DOFs no one-process number names (a swapped
        partition changes the seam's basis), the layout's global numbers
        (``ParallelDofs.EnumerateGlobally``): they depend on the partition, and
        only the norm estimates read them.
        """
        if self.comm is None or op.expansion is None:
            return self._numbers()
        numbers, _ = _layout(op).pardofs.EnumerateGlobally()  # ty: ignore[unresolved-attribute]
        return np.asarray(numbers, dtype=np.int64)

    def _inner_tolerance(self) -> float:
        """The tolerance of the shift solves of an iterative linear solver."""
        opts = self.options
        return min(opts.tolerance, max(_INNER * opts.eigen_tolerance, _INNER_FLOOR))

    def _entry(
        self,
        holder: KeptOperator,
        ks: list[complex],
        x: np.ndarray,
        y: np.ndarray | None,
        *,
        reach: ModeReach,
    ) -> Entry:
        """What the backward pass needs: the kept operator, the modes and their T'.

        ``x`` holds the modes on the solve space, ``y`` their left
        eigenvectors (``None``: ``x``, complex symmetric). ``gram`` is block
        diagonal by cluster: ``y_i^T T'(k_c) x_j`` for modes i, j of cluster
        c, with ``k_c`` the cluster's mean; under ranks a pairing of loads
        and states (`DofLayout.pair`), the same on every rank. ``reach``
        gives ``cluster``, ``incomplete`` (the modes of a cluster the guard
        mode belongs to), ``gap`` (each mode's relative gap to its gap set,
        `GapSet.gap`, ``inf`` without a neighbour, 0 for a mode whose polish
        failed) and ``polish_failed`` (those modes, never isolated: the
        backward names the failure) (`ModeReach.fields`).
        """
        left = x if y is None else y
        labels = reach.labels
        k = np.array(ks, dtype=complex)
        gram = np.zeros((len(k), len(k)), dtype=complex)
        op = cast("EigenOperator", holder.operator)
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            mean = k[members].mean()
            tx = np.stack([op.derivative(mean, x[:, j]) for j in members], axis=1)
            if self.comm is None:
                gram[np.ix_(members, members)] = left[:, members].T @ tx
            else:
                # the loads T'(k_c) x paired with the states y: agreed values
                gram[np.ix_(members, members)] = op.layout.pair(tx, left[:, members]).T
        fields = {"x": x, "k": k, "cluster": labels, "gram": gram}
        fields.update(reach.fields())
        if y is not None:
            fields["y"] = y
        return Entry(self.physics, handle=holder, fields=fields)

    # --- post-processing ---------------------------------------------------------

    def _postprocess(
        self,
        model: Model,
        op,
        divfree: DivFreeProjector,
        modes: list[Refined],
    ) -> tuple[dict[str, object], np.ndarray]:
        """The outputs, and the modes scaled to them on the solve space.

        ``e_field`` holds the modes expanded to the unpaired space on a
        periodic model (``P x``). A mode is a state, an energy form's or a
        port's functional a load: their products are the layout's pairings.
        Under ranks a rank integrates its own tetrahedra's energies, summed over
        the ranks per region (`_volume_energies`), the sign follows one process's
        largest entry (`gfem._sign.largest_value`), and ``e_field`` is the rank's
        piece, padded (`gfem._engine.fields.rank_piece`).
        """
        built, space, order = op.built, op.space, op.order
        layout, numbers = _layout(op), self._numbers()
        scale = built.scale
        ports = {data.port.name: data for data in op.ports}
        voltage = {
            name: terms.port_vectors(space, built, order, data)[1]
            for name, data in ports.items()
        }
        mesh = model.mesh
        density = element_energy(model, op)
        sheets = _sheet_energy(op)
        split = (
            principal_energy(model, op, self.anisotropic) if self.anisotropic else None
        )
        n = len(modes)
        omega = np.array([m.k * const.c / scale for m in modes], dtype=complex)
        solve = np.zeros((space.ndof, n), dtype=complex)
        bulk = np.zeros((len(self.physical), n))
        principal = np.zeros((len(self.anisotropic), n, 3))
        q_port = np.zeros((len(self.resistive), n))
        participation = np.zeros((len(self.inductive), n))
        for j, mode in enumerate(modes):
            x = self._phase(mode.x, op, divfree, numbers)
            # electric energy per element (gfem order) at this amplitude
            per_tet = np.zeros(len(mesh.tetrahedra))
            per_tet[built.cell_order] = (
                0.5 * const.epsilon_0 * scale * np.real(density(x))
            )
            per_tet *= self._law_factor(model, mode.k, scale)
            volts = {name: complex(layout.pair(v, x)) for name, v in voltage.items()}
            capacitors = sum(
                0.5 * data.port.rlc[2] * abs(volts[name]) ** 2
                for name, data in ports.items()
                if data.port.rlc[2] is not None
            )
            per_axis = None
            if split is not None:
                # the same energy along each principal axis
                per_axis = np.zeros((len(mesh.tetrahedra), 3))
                per_axis[built.cell_order] = (
                    0.5 * const.epsilon_0 * scale * np.real(split(x))
                )
            volume, regions, axes = self._volume_energies(mesh, per_tet, per_axis)
            # + 0.0 without a capacitive sheet leaves the sum bit-identical
            energy = volume + capacitors + (0.0 if sheets is None else sheets(x))
            amplitude = math.sqrt(MODE_ENERGY / energy)
            solve[:, j] = x * amplitude
            for r in range(len(self.physical)):
                bulk[r, j] = regions[r] / energy
            for r in range(len(self.anisotropic)):
                principal[r, j] = axes[r] / energy
            w = omega[j].real
            for p, name in enumerate(self.resistive):
                resistance = ports[name].port.rlc[0]
                kappa = abs(volts[name]) ** 2 / (2 * resistance * energy)
                q_port[p, j] = w / kappa if kappa > 0 else math.inf
            for p, name in enumerate(self.inductive):
                inductance = ports[name].port.rlc[1]
                current = volts[name] / (1j * w * inductance)
                ratio = inductance * abs(current) ** 2 / (2 * energy)
                participation[p, j] = math.copysign(ratio, volts[name].real)
        outputs = {
            "omega": omega,
            "error": np.array([m.error for m in modes]),
            "q_port": q_port,
            "participation": participation,
            "bulk": bulk,
            "principal": principal,
            "sheet": self._sheet_shares(op, solve, omega),
            "e_field": rank_piece(
                op.expand(solve), lambda: self._field_width(model), self.comm
            ),
        }
        return outputs, solve

    def _outputs(
        self,
        model: Model,
        op: DrivenOperator,
        divfree: DivFreeProjector,
        modes: list[Refined],
        ports: Sequence[EigenPort],
    ) -> tuple[dict[str, object], np.ndarray]:
        """`_postprocess`, with every coupled port's ``q_port`` (`_coupling_q`)."""
        outputs, x = self._postprocess(model, op, divfree, modes)
        outputs["q_port"] = self._coupling_q(
            outputs, x, [m.k for m in modes], ports, op
        )
        return outputs, x

    def _coupling_q(
        self,
        outputs: dict[str, object],
        x: np.ndarray,
        ks: list[complex],
        ports: Sequence[EigenPort],
        op: DrivenOperator,
    ) -> np.ndarray:
        r"""``(n_coupled, n)`` the coupling Q of `coupled`, in model order.

        A resistive lumped port's row is ``outputs["q_port"]``'s. A wave
        port's is :math:`\mathrm{Re}\,\omega\,E / P_p` with :math:`P_p`
        the power its mode carries out (`PortTerm.outgoing_power`) for the
        modes ``x`` normalized to :math:`E` = `MODE_ENERGY`, ``ks`` their
        wavenumbers.
        """
        lumped = cast("np.ndarray", outputs["q_port"])
        out = np.zeros((len(self.coupled), len(ks)))
        rows = {name: lumped[p] for p, name in enumerate(self.resistive)}
        per_k = const.c / op.built.scale
        pair = _layout(op).pair
        for port in ports:
            term = port.term
            for j, k in enumerate(ks):
                power = term.outgoing_power(k, x[:, j], pair)
                w = (k * per_k).real
                for p, wave in enumerate(term.ports):
                    row = rows.setdefault(wave.name, np.zeros(len(ks)))
                    row[j] = w * MODE_ENERGY / power[p] if power[p] > 0 else math.inf
        for i, name in enumerate(self.coupled):
            out[i] = rows[name]
        return out

    def _phase(
        self,
        x: np.ndarray,
        op: DrivenOperator,
        divfree: DivFreeProjector,
        numbers: np.ndarray | None,
    ) -> np.ndarray:
        r"""``x`` with the phase and sign of the mode convention (`gfem.conventions`).

        :math:`\int E \cdot E` real and positive, then the largest entry's real part
        positive (one process's entry under ranks, `largest_value`). On a
        periodic model that entry is the expanded field's (``P x`` on the field
        rows, ``numbers`` theirs), on one process and under ranks alike: a
        partition can change the periodic basis on the seam, never ``P x``.
        """
        if self.problem.wave_vector is not None:
            # a Bloch mode's ∫ E . E depends on the cell and can vanish:
            # its largest coefficient is made real and positive instead
            # (set exactly: the product's imaginary part is round-off)
            i = largest_entry(x)
            largest = x[i]
            x = x * (np.conj(largest) / abs(largest))
            x[i] = abs(largest)
            return x
        layout = _layout(op)
        s = layout.pair(divfree.energy(x), x)
        x = x * np.exp(-0.5j * np.angle(s))
        if op.expansion is None:
            reference = largest_value(x, layout, numbers)
        else:
            reference = largest_value(op.expand(x), self._field_layout(), numbers)
        if reference.real < 0:
            x = -x
        return x

    def _field_layout(self) -> DofLayout:
        """The field rows' layout across ranks; `ONE_PROCESS` on one process.

        The plain mesh's H(curl) space of `fe_space`, whose rows are a result
        field's (`gfem._engine.driven.Expansion.field_rows`), memoized with it.
        """
        if self.comm is None:
            return ONE_PROCESS
        still = _at_imported_nodes(self._mesh)
        regions, excluded, order = self.regions, self._excluded, self.options.order
        _, space, _ = fe_space(still, regions, excluded, "hcurl", order, comm=self.comm)
        disc = discretization(still, (), self.comm)
        key = (("field", tuple(regions), tuple(excluded)), "hcurl", order, "layout")
        return disc.space(key, lambda: RankLayout(space, self.comm))

    def _volume_energies(
        self, mesh: Mesh, per_tet: np.ndarray, per_axis: np.ndarray | None
    ) -> tuple[float, list, list]:
        """The volume energy, each region's, and each anisotropic region's per axis.

        ``per_tet`` and ``per_axis`` hold this rank's tetrahedra (gfem order;
        zero elsewhere). Under ranks the sums are rank sums of the local ones, one
        reduction of ``1 + n_regions + 3 n_anisotropic`` values, never of a whole
        array.
        """
        volume = per_tet.sum()
        regions = [per_tet[mesh.elements(region)].sum() for region in self.physical]
        axes = []
        if per_axis is not None:
            axes = [
                per_axis[mesh.elements(region)].sum(axis=0)
                for region in self.anisotropic
            ]
        if self.comm is None:
            return volume, regions, axes
        local = np.concatenate([[volume], regions, np.ravel(axes)])
        total = _mpi.reduce_bcast(self.comm, local)
        n = len(self.physical)
        return (
            float(total[0]),
            list(total[1 : 1 + n]),
            list(total[1 + n :].reshape(-1, 3)),
        )

    def _field_width(self, model: Model) -> int:
        """A mode field's rows on every rank: `dof_shape`'s width."""
        return dof_shape(
            model.mesh,
            self.regions,
            self._excluded,
            "hcurl",
            self.options.order,
            comm=self.comm,
        )[0]

    def _law_factor(self, model: Model, k: complex, scale: float) -> np.ndarray | float:
        """The mode energy's factor per tetrahedron at the mode ``k``.

        eps_W / eps_inf on a law region (its energy permittivity at ``Re
        omega`` over the eps_inf that `element_energy` integrates), 1
        elsewhere; the number 1 without a law (bit-identical).
        """
        if not self._laws:
            return 1.0
        omega = k.real * const.c / scale
        return scale_per_tetrahedron(model, law_scale(model, omega, energy_eps))

    def _sheet_shares(
        self, op: DrivenOperator, solve: np.ndarray, omega: np.ndarray
    ) -> np.ndarray:
        """``(n_kinetic, n)`` sheet participations of the modes scaled to 1 J.

        A sheet's L piece is ``mu0 scale / L_s ∫ w u_t . v_t`` (the fill
        ``w`` as in the operator): ``∫ w |E_t|^2 / (2 omega^2
        L_s)`` is its form over ``2 mu0 scale omega^2``.
        """
        out = np.zeros((len(self.kinetic), len(omega)))
        layout = _layout(op)
        for s, piece in enumerate(sheet_inductors(op, self.kinetic)):
            applied = op.apply(piece, solve)
            for j in range(len(omega)):
                form = float(np.real(layout.pair(applied[:, j], np.conj(solve[:, j]))))
                out[s, j] = form / (
                    2 * const.mu_0 * op.built.scale * omega[j].real ** 2 * MODE_ENERGY
                )
        return out

    # --- the solve_vjp interface ---------------------------------------------------

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output."""
        n = self.problem.n_modes
        if self.comm is None:
            mesh = model.mesh
            still = _at_imported_nodes(mesh)
            with task_manager(self.options.threads):
                _, space, _ = fe_space(
                    still, self.regions, self._excluded, "hcurl", self.options.order
                )
            rows = space.ndof
        else:
            with task_manager(self.options.threads, self.comm):
                rows = self._field_width(model)
        return {
            "omega": jax.ShapeDtypeStruct((n,), jnp.complex128),
            "error": jax.ShapeDtypeStruct((n,), jnp.float64),
            "share": jax.ShapeDtypeStruct((len(self.port_surfaces), n), jnp.float64),
            "q_port": jax.ShapeDtypeStruct((len(self.coupled), n), jnp.float64),
            "participation": jax.ShapeDtypeStruct(
                (len(self.inductive), n), jnp.float64
            ),
            "bulk": jax.ShapeDtypeStruct((len(self.physical), n), jnp.float64),
            "principal": jax.ShapeDtypeStruct(
                (len(self.anisotropic), n, 3), jnp.float64
            ),
            "sheet": jax.ShapeDtypeStruct((len(self.kinetic), n), jnp.float64),
            "e_field": jax.ShapeDtypeStruct((rows, n), jnp.complex128),
            "cluster": jax.ShapeDtypeStruct((n,), jnp.int64),
            "v_g": jax.ShapeDtypeStruct((n, 3), jnp.complex128),
        }

    def role(self, kind: tuple) -> str:
        """``"read"`` if the eigenmode solve depends on input ``kind``.

        A thermal boundary value is not read: its gradient is an exact zero.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        return "unread" if is_thermal_value(kind) else "read"

    def unsupported_input(self, kind: tuple) -> str | None:
        """Why a read input cannot be differentiated here, or ``None``.

        The node coordinates, next to a tensor absorbing admittance
        (`terms.absorbing_shape_unsupported`); a permittivity law's
        parameter on a periodic model (no verification cell).

        Raises:
            PortGeometryError: ``kind`` is an input a wave port's mode reads,
                (`PortDesign.check`): the wave port's term at a complex
                eigenfrequency holds the mode.
        """
        self.port_design.check(kind)
        if kind == ("nodes",):
            return self.nodes_unsupported
        if kind[0] == "law" and (
            self._periodic_surfaces or self.problem.wave_vector is not None
        ):
            return (
                "gradients with respect to a permittivity law's parameters are not "
                "verified on a periodic eigenmode model in this version"
            )
        return None

    def nondifferentiable_output(
        self, name: str, kinds: frozenset[tuple] | set[tuple] = frozenset()
    ) -> str | None:
        """Why a cotangent on output ``name`` cannot be pulled back, or ``None``.

        Input-aware: ``kinds`` are the wanted input kinds. The
        eigenvalues differentiate with respect to every read input. The
        mode-field outputs (``q_port``, ``participation``,
        ``bulk_participation``, ``principal_participation``,
        ``sheet_participation`` and the normalized fields ``e_fields``)
        differentiate on one CPU process with ``linear_solver="direct"``,
        ``eigen_tolerance`` at most 1e-10, without periodic pairs or a Bloch
        wave vector, with respect to materials, sheet fills, node
        coordinates (port surfaces held) and lumped values; the normalized
        fields never with respect to the node coordinates, the principal
        shares not with respect to a lumped value, and none of them with
        respect to a permittivity law's parameter. What depends on the modes
        (a cluster, a near pair, a port without voltage) raises in the
        backward.

        Raises:
            DistributedUnsupportedError: Above one rank, any mode-field output
                (key ``mode_field_gradient``). It reads replicated values
                only, so every rank raises it at the backward's trace.
        """
        if name in ("omega", "cluster"):
            return None
        if name in _MODE_FIELD and self.comm is not None:
            # replicated values only, so every rank raises it at the backward's
            # trace
            _mpi.refuse_feature("mode_field_gradient")
        public = _MODE_FIELD.get(name, name)
        head = f"gfem cannot differentiate the eigenmode output {public!r}"
        opt_out = (
            "Wrap it in jax.lax.stop_gradient to use its value without its derivative."
        )
        if name not in _MODE_FIELD:
            what = (
                "a diagnostic of the found mode, no derivative"
                if name == "share"
                else "the band group velocity, read from the modes, which has no "
                "derivative with respect to anything"
                if name == "v_g"
                else "the backward error of the solve, which has no derivative "
                "with respect to anything"
            )
            if name == "share":
                head = (
                    "gfem cannot differentiate the eigenmode output "
                    "'requested_power_share'"
                )
            return f"{head}: it is {what}. " + opt_out
        why = self._solve_refusal()
        refusal = None if why is not None else self._input_refusal(name, kinds)
        if refusal is not None and refusal.kind == "law":
            return (
                f"{head} {refusal.message}: the mode fields and the ratios of a "
                "dispersive model have no verified gradient by its permittivity "
                "law in this version. Differentiate the eigenfrequency or Q "
                "(omega) instead, or wrap the value in jax.lax.stop_gradient to "
                "use it without its derivative."
            )
        if refusal is not None and refusal.kind == "lumped":
            return (
                f"{head} {refusal.message} in this version; the eigenfrequencies "
                "differentiate with respect to it. Wrap the value in "
                "jax.lax.stop_gradient to hold it."
            )
        if refusal is not None:
            why = refusal.message
        if why is None:
            return None
        return (
            f"{head} {why}. The eigenfrequencies (omega, so frequency and q) "
            "keep their gradients. " + opt_out
        )

    def _solve_refusal(self) -> str | None:
        """Why no mode-field output of this solve differentiates, or ``None``."""
        options = self.options
        why = blocker(
            options,
            self.linear.kind,
            wave_vector=self.problem.wave_vector,
            periodic=bool(self._periodic_surfaces),
        )
        if why is None:
            return None
        texts = {
            "wave_vector": (
                "at a Bloch wave vector: mode-field gradients are for "
                "non-periodic models only"
            ),
            "periodic": (
                "of a periodic model: mode-field gradients are for "
                "non-periodic models only"
            ),
            "device": (
                f'on device="{options.device}": its mode-field adjoint runs on '
                'the CPU with linear_solver="direct" only'
            ),
            "linear": (
                f'on linear_solver="{self.linear.kind}": its mode-field adjoint '
                'runs on linear_solver="direct" only'
            ),
            "tolerance": (
                f"solved with SolverOptions.eigen_tolerance = "
                f"{options.eigen_tolerance:g}: mode-field gradients need "
                f"eigen_tolerance <= {FIELD_TOLERANCE:g} (the default)"
            ),
        }
        return texts[why]

    @staticmethod
    def _input_refusal(
        name: str, kinds: frozenset[tuple] | set[tuple]
    ) -> _InputRefusal | None:
        """Why output ``name`` does not differentiate by ``kinds``, or ``None``.

        Every mode-field output by a permittivity law's parameter,
        the normalized field by the node coordinates, and the principal
        shares by a lumped value.
        """
        laws = [k for k in kinds if k[0] == "law"]
        if laws:
            region = min(laws, key=str)[1]
            return _InputRefusal(
                "law",
                f"with respect to the permittivity law of model.materials[{region!r}]",
            )
        if name == "e_field" and ("nodes",) in kinds:
            return _InputRefusal(
                "nodes",
                "with respect to the mesh node coordinates: the normalized mode "
                "field (and the measurements of it: Field.at, "
                "interface_participation) has no node gradient; it "
                "differentiates with respect to materials and sheet fills. For "
                "a shape derivative take finite differences of the measurement "
                "over meshes moved by gfem.shape.boundary_morph",
            )
        lumped = sorted(k for k in kinds if is_lumped(k))
        # the principal shares have no finite-difference cell by a lumped
        # value yet; every other output does
        if lumped and name == "principal":
            kind = lumped[0]
            where = "ports" if kind[0] == "port" else "boundaries"
            return _InputRefusal(
                "lumped", f"with respect to model.{where}[{kind[1]}].{kind[2]}"
            )
        return None

    def backward(
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        wanted: frozenset | None = None,
    ) -> list[FieldPair]:
        """Fold the cotangents into field pairs: per cluster, and per mode.

        **Eigenvalues.** With ``ω = k c / scale``, cluster c's modes ``X``,
        their left eigenvectors ``Y`` (``X`` unless Bloch) and ``G = Y^T
        T'(k_c) X``: ``ct_c d(Σ ω)/dp = -(c / scale) ct_c tr(G^-1 Y^T
        (dT/dp) X)``, the pair ``(-(c / scale) ct_c Y, X G^-1)`` weighted with
        the pieces' θ at ``k_c``. A simple mode is a cluster of one (``G =
        y^T T' x``). No solve. On a periodic model the pair is expanded to
        the unpaired space, ``(conj(P) Y, P X G^-1)``: ``y^T P^H a P x =
        (conj(P) y)^T a (P x)`` for the unpaired forms ``a``
        (`DrivenOperator.pair`). Under ranks ``gram`` is agreed and the
        pairs hold each rank's state columns, whose element passes the
        caller sums over ranks.

        **Mode-field outputs** (`_field_pairs`): per mode with a cotangent
        on its ratios, one adjoint solve on the kept eigen operator and two
        pairs, ``(2 lam, x)`` on the pieces and the explicit energy pair;
        the explicit terms of the lumped ports' values are kept for
        `explicit_inputs`.

        Raises:
            DegenerateModeError: The ``omega`` cotangents on a cluster's
                modes differ (an objective not symmetric in them), a cluster
                with a nonzero cotangent extends beyond the ``n_modes``
                modes, a Bloch cluster's ``G`` is singular (a defective
                eigenvalue), or a mode with a mode-field cotangent is not
                isolated or is defective.
            NonDifferentiableError: A nonzero cotangent on an infinite
                ``q_port``.
            LinearSolverConvergenceError: An adjoint solve was not accepted.
        """
        del wanted  # every wanted input reads the same pairs
        fields = entry.fields
        x, k, labels = fields["x"], fields["k"], fields["cluster"]
        y = fields.get("y", x)
        gram, incomplete = fields["gram"], fields["incomplete"]
        ct = cotangents.get("omega")
        pairs = []
        self._explicit = {}
        with task_manager(self.options.threads, self.comm):
            op = self.driven(model)
            domain = em_region(model, op.built)
            per_k = const.c / op.built.scale
            if ct is not None and np.any(ct):
                ct = np.asarray(ct, dtype=complex)
                for label in np.unique(labels):
                    members = np.flatnonzero(labels == label)
                    c = ct[members]
                    if not c.any():
                        continue
                    spread = np.abs(c - c.mean()).max()
                    symmetric = spread <= _SYMMETRIC_RTOL * np.abs(c).max()
                    # from agreed values (the cotangents, the clusters, gram):
                    # every rank raises it together
                    if incomplete[members].any() or not symmetric:
                        raise _mpi.agreed(
                            DegenerateModeError(
                                self._degenerate_message(
                                    k, members, c, incomplete, per_k
                                )
                            )
                        )
                    g = gram[np.ix_(members, members)]
                    if "y" in fields and np.linalg.cond(g) > _DEFECTIVE:
                        raise _mpi.agreed(
                            DegenerateModeError(
                                self._defective_message(
                                    k, members, np.linalg.cond(g), per_k
                                )
                            )
                        )
                    pairs.append(
                        op.pair(
                            -per_k * c.mean() * y[:, members],
                            x[:, members] @ np.linalg.inv(g),
                            weights=op.weights(k[members].mean() * per_k),
                            domain=domain,
                        )
                    )
            pairs += self._field_pairs(model, entry, cotangents, op, domain)
        return pairs

    def _field_pairs(
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        op: DrivenOperator,
        domain,
    ) -> list[FieldPair]:
        """The mode-field outputs' pairs.

        Per mode ``(k, x)`` with a nonzero cotangent on its ratios or on its
        normalized field: the load ``(w, kappa, explicit terms)``
        (`Ratios.load`, with the field's normalization multipliers ``mu1``,
        ``mu2``), the deflated solve ``T(k) lam0 = -w``, ``t^T lam0 = 0``
        (``t = T'(k) x``) on the kept operator's live factors
        (`deflated_solve`), ``lam = lam0 - kappa x / (x^T t)``, and the pairs
        ``(2 lam, x)`` with the pieces' θ at the mode (``dG/dp = 2 Re(lam^T
        dT/dp x)``), ``(conj(x), x)`` with the explicit energy terms' weights
        and ``(x, x)`` with the phase rule's. The load's explicit terms in
        the lumped ports' values (`Load.lumped`) add up in ``_explicit``. An
        entry without its operator (over ``retain_memory``) gets one from
        ``op``, prepared at ``k (1 + d)`` of its first such mode, and freed
        at the end.
        """
        names = ("q_port", "participation", "bulk", "principal", "sheet")
        cts = {
            n: np.real(np.asarray(cotangents[n]))
            for n in names
            if n in cotangents and np.any(cotangents[n])
        }
        # the normalized field's cotangent, complex (non-periodic: the solve
        # space is the unpaired one)
        field = cotangents.get("e_field")
        if field is not None and not np.any(field):
            field = None
        if not cts and field is None:
            return []
        fields = entry.fields
        x, k = fields["x"], fields["k"]
        n = len(k)

        def column(c: np.ndarray, j: int) -> np.ndarray:
            return c[:, j] if c.ndim == 2 else c[:, j, :]

        wanted = [
            j
            for j in range(n)
            if any(column(c, j).any() for c in cts.values())
            or (field is not None and field[:, j].any())
        ]
        if not wanted:
            return []
        per_k = const.c / op.built.scale
        self._check_reach(entry, wanted, per_k)
        holder = entry.handle
        eop = getattr(holder, "operator", None)
        # the forward's port terms: the backward never sweeps a port again
        terms_kept = [p.term for p in getattr(holder, "ports", ())]
        own = eop is None
        gap = fields["gap"]
        if own:
            first = wanted[0]
            ports = tuple(
                EigenPort(term, term.robin_piece(model, op.built, op.order))
                for term in terms_kept
            )
            eop = self.operator(
                op, k[first] * (1 + reshift_offset(float(gap[first]))), ports=ports
            )
        pairs = []
        try:
            # every mode's guard before any form is built
            gram = {}
            for j in wanted:
                u, kj = x[:, j], complex(k[j])
                t = eop.derivative(kj, u)
                gram[j] = u @ t
                size = np.linalg.norm(u) * np.linalg.norm(t)
                if abs(gram[j]) < _DEFECTIVE_FIELD * size:
                    raise DegenerateModeError(
                        self._defective_message(
                            k, np.array([j]), size / max(abs(gram[j]), 1e-300), per_k
                        )
                    )
            ratios = Ratios(
                model,
                op,
                regions=self.physical,
                anisotropic=self.anisotropic,
                resistive=self.resistive,
                inductive=self.inductive,
                kinetic=self.kinetic,
                coupled=self.coupled,
                waves={
                    wave.name: (term, p)
                    for term in terms_kept
                    for p, wave in enumerate(term.ports)
                },
                pair=_layout(op).pair,
                cluster_rtol=self.options.cluster_rtol,
            )
            index = {port.name: i for i, port in enumerate(model.ports)}
            for j in wanted:
                u, kj, ut = x[:, j], complex(k[j]), gram[j]
                load = ratios.load(
                    u,
                    kj,
                    {name: column(c, j) for name, c in cts.items()},
                    mode=j,
                    field=None
                    if field is None
                    else np.where(eop.free, np.asarray(field[:, j], complex), 0),
                )
                lam = (
                    deflated_solve(
                        eop, u, None, -load.w, k=kj, gap=float(gap[j]), mode=j
                    ).z
                    - load.kappa * u / ut
                )
                pairs.append(
                    op.pair(
                        2 * lam[:, None],
                        u[:, None],
                        weights=op.weights(kj * per_k),
                        domain=domain,
                    )
                )
                if load.energy:
                    pairs.append(
                        op.pair(
                            np.conj(u)[:, None],
                            u[:, None],
                            weights=load.energy,
                            domain=domain,
                        )
                    )
                if load.phase:
                    pairs.append(
                        op.pair(
                            u[:, None], u[:, None], weights=load.phase, domain=domain
                        )
                    )
                for (port, value), c in load.lumped.items():
                    kind = ("port", index[port], value)
                    self._explicit[kind] = self._explicit.get(kind, 0.0) + c
        finally:
            if own:
                eop.free_handle()
        return pairs

    def _check_reach(self, entry: Entry, modes: list[int], per_k: float) -> None:
        """Raise unless every mode of ``modes`` is isolated (`ModeReach.isolated`).

        Raises:
            DegenerateModeError: A mode's polish failed in the forward, it
                belongs to a cluster of several modes or one cut by
                ``n_modes``, or its relative gap is below
                `gfem._eigen.reach.ISOLATED`
                (``cluster_rtol`` if larger): nearly degenerate.
        """
        k = entry.fields["k"]
        rtol = self.options.cluster_rtol
        stored = ModeReach.from_fields(entry.fields, rtol)
        labels, incomplete, gap = stored.labels, stored.incomplete, stored.gap
        failed, threshold = stored.polish_failed, stored.threshold
        reach = stored.isolated
        for j in modes:
            if reach[j]:
                continue
            members = np.flatnonzero(labels == labels[j])
            hz = k * per_k / (2 * math.pi)
            named = ", ".join(
                f"{int(i)} ({hertz(hz[i].real, 9)})" for i in members if i != j
            )
            mode = f"mode {j} ({hertz(hz[j].real, 9)})"
            if failed[j]:
                reason = (
                    "its polish in the forward did not converge (a sign of a "
                    "neighbour the solve did not return)"
                )
            elif len(members) > 1:
                reason = (
                    f"it belongs to a degenerate cluster with mode(s) {named} "
                    f"(relative gap at most SolverOptions.cluster_rtol = {rtol:g})"
                )
            elif incomplete[j]:
                reason = (
                    "its cluster extends beyond the "
                    f"{self.problem.n_modes} mode(s) asked for (Eigenmode.n_modes)"
                )
            else:
                reason = (
                    f"it is nearly degenerate: its relative gap to the nearest "
                    f"mode found is {gap[j]:.3g}, below {threshold:g} (the "
                    f"isolation threshold 1e-4, or SolverOptions.cluster_rtol = "
                    f"{rtol:g} if larger)"
                )
            raise DegenerateModeError(
                f"DegenerateModeError: a quantity of the field of {mode} "
                f"(participation, port Q, a field) is not differentiable: "
                f"{reason}. Mode-field gradients are for isolated modes only: "
                "near another mode the field is a mix of the two "
                "that any perturbation rotates. Remedies: separate the modes "
                "(break the design's symmetry, move them apart); differentiate "
                "the frequencies only; or wrap the field quantity in "
                "jax.lax.stop_gradient."
            )

    def driven(self, model: Model) -> DrivenOperator:
        """The model's driven operator (the pieces of ``T(k)``) at its nodes.

        The last model's operator is kept, so that a backward pass and its
        terms build it once, while its discretization is the model's cached
        one: an operator on an evicted discretization would pair with
        kernel objects of the new one.
        """
        last = getattr(self, "_last_driven", None)
        if (
            last is not None
            and last[0] is model
            and last[1].built
            is discretization(model.mesh, model._periodic, self.comm).built  # noqa: SLF001
        ):
            return last[1]
        op = driven_operator(
            model,
            self.options.order,
            wave_vector=self.problem.wave_vector,
            comm=self.comm,
            stretch=self._stretch(model),
        )
        self._last_driven = (model, op)
        return op

    def terms(self, model: Model, *, nodes: bool) -> list[Term]:
        """The pieces of ``T(k)`` as the kernel sees them: one `Term` per piece.

        The driven operator's (`gfem._engine.driven.operator_terms`): a
        mode's ``dk/dp = -x^T (dT/dp) x / x^T T'(k) x`` contracts the same
        pieces, weighted with their θ at the mode, for materials, fills and
        node coordinates. Port pieces are left out: shape gradients hold
        every port surface.

        The explicit energy terms of the mode-field outputs follow
        (`Ratios.terms`): ``Re(eps_r)`` per region and per
        principal axis, which only their energy pairs weight. They are
        ``"value"`` terms on the element-wise ``Re(eps_r)`` masked to their
        part, so the node pass transports them as it does the mass piece; the
        port voltages read only the held port surfaces, so a node gradient
        of the ratios needs no other term.

        Raises:
            PortGeometryError: ``nodes`` and a node of a port surface has
                moved (with an ``omega`` or a mode-field cotangent alike).
        """
        op = self.driven(model)
        ratios = Ratios(
            model,
            op,
            regions=self.physical,
            anisotropic=self.anisotropic,
            resistive=(),
            inductive=(),
        )
        # a density blend never reaches an eigenmode solve: its
        # mass derivative is the driven family's only
        pieces = [
            dataclasses.replace(
                t, derivatives={k: d for k, d in t.derivatives.items() if k != "blend"}
            )
            if "blend" in t.derivatives
            else t
            for t in operator_terms(model, op, nodes=nodes)
        ]
        return pieces + ratios.terms()

    def lumped_terms(self, model: Model) -> list[Term]:
        """The pieces of ``T(k)`` that move with a lumped value.

        The driven operator's (`gfem._engine.driven.lumped_terms`), weighted
        with their θ at the mode like the other pieces.
        """
        return lumped_terms(self.driven(model))

    def explicit_inputs(self) -> dict[tuple, Any]:
        """The last backward's explicit terms by input kind (lumped values only).

        The lumped pass contracts the pairs with the pieces that move with
        a lumped value: the implicit part ``2 Re(lam^T dT/dp x)``, and a
        sheet's L and C through the explicit energy pair. A lumped port's L
        and C enter the mode-field outputs outside any piece, at fixed mode:
        ``-p / L`` in its participation, ``|V|^2 / 2`` in the mode energy
        (and so in every ratio and, through ``mu1``, the normalized field).
        These are scalars, by ``("port", index, "inductance" |
        "capacitance")``, which `solve_vjp` adds to the lumped pass's values.
        The channel takes the cotangent of any input kind the same way (a
        law, a sheet fill, an impedance value).
        """
        return dict(self._explicit)

    def _degenerate_message(
        self,
        k: np.ndarray,
        members: np.ndarray,
        ct: np.ndarray,
        incomplete: np.ndarray,
        per_k: float,
    ) -> str:
        hz = k[members] * per_k / (2 * math.pi)
        gap = np.ptp(np.abs(k[members])) / np.abs(k[members]).max()
        modes = ", ".join(
            f"{int(j)} ({hertz(f.real, 9)})" for j, f in zip(members, hz, strict=True)
        )
        if incomplete[members].any():
            reason = (
                "the cluster extends beyond the "
                f"{self.problem.n_modes} mode(s) asked for (Eigenmode.n_modes), so "
                "no objective of these modes is symmetric in the whole cluster"
            )
        else:
            reason = (
                "the objective is not symmetric in them (cotangents "
                + ", ".join(f"{complex(v):.4g}" for v in ct)
                + ")"
            )
        return (
            f"DegenerateModeError: mode(s) {modes} belong to a degenerate cluster "
            f"(relative gap {gap:.2g}, SolverOptions.cluster_rtol = "
            f"{self.options.cluster_rtol:g}): the modes of a cluster are an "
            "arbitrary basis of their eigenspace, so one mode's frequency is not "
            f"differentiable, and {reason}. Remedies: differentiate a function "
            "symmetric in the cluster's modes (their sum or mean, a smooth "
            "aggregate such as log-sum-exp instead of max), with n_modes "
            "covering the whole cluster; break the symmetry of the design; or "
            "use a gfem.Driven sweep as a surrogate."
        )

    def _defective_message(
        self, k: np.ndarray, members: np.ndarray, condition: float, per_k: float
    ) -> str:
        hz = k[members] * per_k / (2 * math.pi)
        modes = ", ".join(
            f"{int(j)} ({hertz(f.real, 9)})" for j, f in zip(members, hz, strict=True)
        )
        return (
            f"DegenerateModeError: mode(s) {modes} have a singular Y^T T'(k) X "
            f"(condition number {condition:.3g}): the eigenvalue is defective "
            "(an exceptional point of a lossy problem), where its left and "
            "right eigenvectors are orthogonal, and neither its frequency nor "
            "a quantity of its field is differentiable. Remedies: move the "
            "design (or the wave vector) off the exceptional point, or use a "
            "gfem.Driven sweep as a surrogate."
        )

    def result(self, outputs: dict) -> EigenmodeResult:
        """The public result from the forward outputs."""
        omega = outputs["omega"]
        fields = [
            Field(
                coefficients=outputs["e_field"][:, j],
                mesh=self._mesh,
                quantity="E",
                unit="V/m",
                space="hcurl",
                order=self.options.order,
                support=self.regions,
                omega=omega[j],
                _excluded=self._excluded,
                _comm=_mpi.static_comm(self.comm),
                _reference_frequency=self._reference_frequency,
            )
            for j in range(self.problem.n_modes)
        ]
        return EigenmodeResult(
            omega=jnp.asarray(omega),
            error=jnp.asarray(outputs["error"]),
            requested_power_share={
                surface: jnp.asarray(outputs["share"][i])
                for i, surface in enumerate(self.port_surfaces)
            },
            q_port={
                name: jnp.asarray(outputs["q_port"][p])
                for p, name in enumerate(self.coupled)
            },
            participation={
                name: jnp.asarray(outputs["participation"][p])
                for p, name in enumerate(self.inductive)
            },
            bulk_participation={
                region: jnp.asarray(outputs["bulk"][r])
                for r, region in enumerate(self.physical)
            },
            principal_participation={
                region: jnp.asarray(outputs["principal"][r])
                for r, region in enumerate(self.anisotropic)
            },
            e_fields=tuple(fields),
            sheet_participation={
                surface: jnp.asarray(outputs["sheet"][s])
                for s, surface in enumerate(self.kinetic)
            },
            materials=self._materials,
            periodic_surfaces=self._periodic_surfaces,
            cluster=jnp.asarray(outputs["cluster"]),
            pml=self._pml,
            v_g=(
                None
                if self.problem.wave_vector is None
                else jnp.asarray(outputs["v_g"])
            ),
        )


def _modal_overlap(
    mesh: Mesh, region: str, material: Material, frequency: float
) -> float:
    """How many of ``region``'s modes lie within one linewidth of ``frequency``.

    Weyl's density of cavity modes times the linewidth, ``M = V k0^3 |Im
    eps| sqrt(Re eps) / pi^2``, with ``V`` the region's volume and ``eps``
    its complex permittivity at ``frequency`` (the mean of its principal
    values, or its permittivity law's value; ``tan_delta`` and
    ``conductivity`` give the loss). 0 where ``Re eps <= 0``. It matched the
    counted modes to 10-30 %.
    """
    omega = 2 * math.pi * frequency
    sigma = np.mean(np.asarray(material.conductivity, dtype=float))
    if isinstance(material.eps_r, PermittivityLaw):
        law = material.eps_r
        eps = complex(law.eps_inf) + dispersion.susceptibility(law, omega)
    else:
        # a scalar eps_r may sit beside three principal loss tangents
        eps_r, tan = np.broadcast_arrays(
            np.asarray(material.eps_r, dtype=float),
            np.asarray(material.tan_delta, dtype=float),
        )
        eps = complex(np.mean(eps_r * (1 - 1j * tan)))
    eps -= 1j * sigma / (omega * const.epsilon_0)
    if not cmath.isfinite(eps):
        return 0.0
    if eps.real <= 0:
        return 0.0
    nodes = np.asarray(mesh.nodes, dtype=float)[mesh.tetrahedra[mesh.elements(region)]]
    edges = nodes[:, 1:] - nodes[:, :1]
    volume = float(np.abs(np.linalg.det(edges)).sum()) / 6
    k0 = omega / const.c
    return volume * k0**3 * abs(eps.imag) * math.sqrt(eps.real) / math.pi**2


def _check_low_frequency(
    model: Model, target: float, lowest: float, *, stacklevel: int, comm: Any = None
) -> None:
    """Warn if the target or the lowest mode is below the low-frequency floor.

    The floor is the driven one, ``k0 h_min = LOW_FREQUENCY_FLOOR`` with
    h_min the shortest edge of the solve domain: below it the E-field
    formulation loses about ``1e-16 / (k0 h_min)^2`` of the modes, which
    their backward error does not show (measured on a line loaded by an
    inductor: frequency off by 2e-5 at ``k0 h_min = 2e-6`` and by 0.2 at
    2e-8, with a reported error of 1e-16).

    Args:
        model: The model, its solve domain and its mesh.
        target: ``Eigenmode.target`` (Hz).
        lowest: The lowest ``Re f`` of the modes found (Hz).
        stacklevel: Passed on to `warnings.warn`, counted from the caller.
        comm: The resolved communicator: the warning is the job's, once.
    """
    floor = LowFrequencyFloor.of(model)
    f = min(target, lowest)
    k0h = floor.k0h(f)
    if k0h >= LOW_FREQUENCY_FLOOR:
        return
    what = "target" if target <= lowest else "lowest mode found"
    _mpi.warn(
        comm,
        f"Eigenmode: the {what}, {hertz(f, 4)}, is below {floor.describe(k0h)}. "
        "Below it the eigenfrequencies, the mode fields "
        "and the participations carry rounding errors up to about "
        "1e-16/(k0*h_min)^2, which EigenmodeResult.error does not show. Raise "
        "the resonance or coarsen the smallest elements.",
        LowFrequencyWarning,
        stacklevel=stacklevel + 1,
    )


def _require_rank_slepc(comm: Any) -> None:
    """SLEPc on the MPI PETSc stack, on every rank, or raise on every rank.

    Raises:
        gfem.SolverUnavailableError: The MPI stack is missing, or a rank cannot
            use slepc4py.
    """
    if not _mpi.agree_all(comm, flag=_petsc.has_mpi()):
        raise SolverUnavailableError(
            "An eigenmode solve above one rank runs SLEPc on the MPI PETSc stack, "
            "and this environment does not have it. Build it with "
            f"tools/petsc_wheelhouse.sh --mpi as {_mpi.RECIPE} describes, or pass "
            "comm=MPI.COMM_SELF for independent one-process copies."
        )
    _mpi.available(
        comm,
        lambda: _petsc.require_slepc("eigen_solver"),
        "eigen_solver='slepc' cannot run: another rank cannot use slepc4py "
        "(that rank's error names why).",
    )


def _layout(op: DrivenOperator) -> DofLayout:
    """The driven operator's layout: `ONE_PROCESS` on one process."""
    return op.layout or ONE_PROCESS


def _sheet_energy(op: DrivenOperator) -> Callable[[np.ndarray], float] | None:
    """``x -> Σ 1/2 ∫ w C_s |E_t|^2 dS`` of the capacitive sheets (J), or ``None``.

    A capacitive `Impedance`'s C piece is ``∫ w C_s / (eps0 scale) u_t .
    v_t``, as the mass piece is ``∫ eps_r u . v``: its quadratic form with
    ``(conj(x), x)`` times ``eps0 scale / 2`` is the sheet's electric
    energy, as `element_energy` gives the volume's. ``None``
    without a capacitive sheet, so such models add nothing.
    """
    sheets = sheet_capacitors(op)
    if not sheets:
        return None
    factor = 0.5 * const.epsilon_0 * op.built.scale
    pair = _layout(op).pair

    def energy(x: np.ndarray) -> float:
        # the load A x paired with the state conj(x)
        return sum(
            factor * float(np.real(pair(op.apply(k, x), np.conj(x)))) for k in sheets
        )

    return energy


def _below_cutoff(ports: Sequence[EigenPort]) -> Callable[[float], bool]:
    """Whether a requested port mode of ``ports`` is below cutoff at a real k.

    From the modes' cutoffs (`gfem._engine.port_term.PortTerm.below_cutoff`):
    the port term's basis stays the forward's.
    """

    def below(k: float) -> bool:
        return any(p.term.below_cutoff(k) for p in ports)

    return below


def _check_wave_ports(model: Model, problem: Eigenmode) -> None:
    """Raise for a wave port this version cannot solve as an eigenmode.

    The first wave port in model order is named: under a Bloch wave vector
    (two-sided refinement with the port's term is unverified), or with a face
    in a perfectly matched layer (its section's layer modes are not
    filtered at a complex port wavenumber).
    """
    ports = [p for p in model.ports if isinstance(p, WavePort)]
    if not ports:
        return
    if problem.wave_vector is not None:
        raise NotImplementedError(
            f"Eigenmode does not support WavePort {ports[0].name!r} with a "
            "wave_vector in this version of gfem: the Bloch modes refine on the "
            "transposed system too, which is not verified with the port's term. "
            "Find the resonances and Q from a gfem.Driven sweep at the same "
            "wave_vector."
        )
    if cut := wave_ports_in_layer(model):
        raise NotImplementedError(
            f"Eigenmode does not support WavePort {cut[0].name!r} on "
            f"{cut[0].surface!r}, whose face crosses the perfectly matched layer, "
            "in this version of gfem: the layer's own modes of the port section "
            "are not told apart at a complex frequency. Keep the port face in "
            "the air box, or find the resonances and Q from a gfem.Driven sweep."
        )


def _check_supported(model: Model) -> None:
    """Raise for model terms this version cannot solve as eigenmodes.

    The first Floquet port in model order is reported.
    """
    for port in model.ports:
        if isinstance(port, FloquetPort):
            raise NotImplementedError(
                f"Eigenmode does not support Floquet port {port.name!r} in this "
                "version of gfem: its orders' propagation constants must be "
                "evaluated at the complex eigenfrequency. Find the resonances and "
                "Q from a gfem.Driven sweep, or end the cell with gfem.Absorbing "
                f"on {port.surface!r} (first order: it reflects oblique waves)."
            )
    terms.check_absorbing(model)
