"""Driven rule: time-harmonic E field in complex Nédélec elements, lumped ports.

At each frequency the operator ``A(ω) = Σ θ_k(ω) A_k`` (see
`gfem._engine.terms`) is assembled and prepared once, and every excited
port's right-hand side is solved in one handle call: GMRES + BDDC by
default, each port starting from its solution at the previous frequency,
or one LDL^T factorization (``direct``: MKL PARDISO with the ``mkl`` extra,
NGSolve sparse Cholesky otherwise). The system operator
(`gfem._engine.system_operator.SystemOperator`) answers the forward and
adjoint solves. A sweep keeps one base handle: a direct solver plans at the
first frequency and refactorizes at the others, and sums each frequency's
matrix from the pieces ``A_k`` assembled once per forward (the piece sum,
kept with the system operator for the backward pass) instead of assembling
it.
With ``SolverOptions.frequency_workers`` above 1, a uniform sweep splits
into contiguous blocks of frequencies, each swept the same way in a
frequency worker (`gfem._engine.workers`), joined in frequency order; the
backward pass stays in this process. ``A`` is complex
symmetric (A^T = A), and a port's excitation vector is proportional to its
voltage functional, so S is symmetric to solver precision.

S is read from the port voltages: with the excited port fed 1 W incident
power (incident voltage sqrt(2 R)), ``S_ij = V_i / sqrt(2 R_i) - δ_ij``,
the power wave of port i with reference impedance R_i. The solution
itself (volts per mesh file unit) is that 1 W field's peak phasor in the
`gfem.Field` coefficient convention, so kept fields are the solution
columns as they are.

Gradients (`DrivenRule.backward`): with ``A x_j = b_j`` and port i's S
functional ``f_i`` parallel to its excitation ``b_i`` (every lumped port; a
wave port whose mode is real up to one phase), port i's solution is (a
multiple of) the adjoint of its own S row, ``A^-1 f_i = c x_i`` (A^T = A,
no transpose, no conjugate). An S cotangent therefore needs no solve when
every port it reads is excited and parallel: per frequency it folds into
one pair ``(-sum_i W_ij A^-1 f_i, x_j)``. A kept-field cotangent, an S row
of a port that was not excited, or of a lossy wave port (its S functional
projects on the conjugated mode), takes one multi-column adjoint solve per
frequency with the system operator (wave-port modal corrections
included); that frequency's operator is prepared on first use at the
adjoint tolerance and kept in the cache entry for later pullbacks. Wave
ports' modes are constants of the solve: a design may not touch a wave
port face (`gfem.PortGeometryError`), so the excitations, the S
functionals and the modal corrections have no design derivative.

An adaptive sweep differentiates with its reduced basis V and sample set
held fixed (`DrivenRule._backward_adaptive`): the cache entry keeps only V
and the reduced model, the backward solves n x n reduced adjoints and folds
them into at most n field pairs ``(V a, V b)`` per material term. Each
sample also solves the S functionals of the wave ports whose adjoint the
solutions miss (`PortNetwork.adjoint_loads`) into V, so the frozen-basis
gradient equals the full solve's at the samples. Under a JAX
transformation the sample set must be frozen (``Driven.samples``), so
every output shape is known before the solve.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from typing import Any, NoReturn

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem._adjoint import adjoint_tolerance
from gfem._boundaries import (
    LumpedPort,
    WavePort,
    sheet_of,
    surfaces_of,
)
from gfem._engine import terms, workers
from gfem._engine.cache import Entry, discretization
from gfem._engine.domain import domain_key, em_region, hcurl_space, pec_surfaces
from gfem._engine.fields import fe_space
from gfem._engine.mesh import EngineMesh
from gfem._engine.port_network import PortNetwork, resistance, split_cotangent
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import FieldPair, ShapeTerm, is_sheet
from gfem._engine.system_operator import SystemOperator
from gfem._engine.wave_ports import WavePortCoupling, wave_port_couplings
from gfem._linalg import Kind, ResolvedLinearSolver
from gfem._sweep.adaptive import AffineSystem, AffineTerm, adaptive_sweep
from gfem.errors import (
    LinearSolverConvergenceError,
    LowFrequencyWarning,
    PortGeometryError,
)
from gfem.mesh import Mesh, _shortest_edges
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Driven
from gfem.results import (
    DrivenResult,
    Field,
    NetworkParameters,
    PortModeField,
    PortModeResult,
    ReducedFields,
    SweepInfo,
)

__all__ = [
    "DEFAULT_SOLVER",
    "LOW_FREQUENCY_FLOOR",
    "DrivenOperator",
    "DrivenRule",
    "check_ports_held",
    "driven_operator",
    "operator_shape_terms",
    "real_space",
]

#: The linear solver for ``linear_solver="auto"``: GMRES + BDDC.
DEFAULT_SOLVER: Kind = "bddc_gmres"

#: k0 * h_min below which a driven solve warns (gfem.LowFrequencyWarning).
LOW_FREQUENCY_FLOOR = 1e-6
#: k0 * h_min below which MKL PARDISO becomes unreliable (with iparm(10) =
#: 16; MUMPS and NGSolve's sparse Cholesky hold to 1e-8, the lowest point
#: measured), named in the low-frequency warning.
_PARDISO_FLOOR = 3e-8


@dataclasses.dataclass(eq=False)
class DrivenOperator:
    """The driven operator of a model, as pieces over one H(curl) space.

    Attributes:
        built: The engine mesh.
        space: Complex first-kind Nédélec space, `PEC` DOFs fixed.
        pieces: The affine pieces (`gfem._engine.terms.Piece`).
        ports: Lumped port geometry, in model port order.
        order: Polynomial order.
    """

    built: EngineMesh
    space: ngs.FESpace
    pieces: list[terms.Piece]
    ports: tuple[terms.LumpedPortData, ...]
    order: int

    def form(self, omega: float) -> ngs.BilinearForm:
        """``Σ θ_k(ω) a_k``, not assembled (a backend assembles it)."""
        u, v = self.space.TnT()
        form = ngs.BilinearForm(self.space, symmetric=True)
        for piece in self.pieces:
            form += complex(piece.theta(omega)) * piece.integrand(u, v)
        return form

    def weights(self, omega: complex) -> dict[str, complex]:
        """θ(ω) of each piece but the ports', by piece name (`gfem._engine.terms`).

        The `FieldPair.weights` of a backward pass, driven (real ω) or
        eigenmode (the complex eigenvalue's ω). The sensitivity kernel looks
        them up by name: the material terms and sheet factors are named
        after the pieces, and so are the shape terms
        (`operator_shape_terms`, which leaves the ports' pieces out too).
        The ports' pieces are design-independent (a port's R, L, C, its
        surface and a wave port's mode are fixed), so no pair carries them.
        """
        return {
            p.name: complex(p.theta(omega)) for p in self.pieces if p.kind != "port"
        }

    def wavenumber(self, omega: float) -> float:
        """k0 per mesh file unit."""
        return terms.wavenumber(omega, self.built.scale)

    @functools.cached_property
    def _piece_forms(self) -> list[ngs.BilinearForm]:
        u, v = self.space.TnT()
        forms = []
        for piece in self.pieces:
            form = ngs.BilinearForm(self.space, symmetric=True, nonassemble=True)
            form += piece.integrand(u, v)
            forms.append(form)
        return forms

    def apply(self, k: int, x: np.ndarray) -> np.ndarray:
        """``A_k X`` of piece ``k`` (its form without θ), matrix-free.

        ``X`` is ``(ndof,)`` or ``(ndof, m)``; fixed DOFs are not special.
        """
        form = self._piece_forms[k]
        columns = x.reshape(x.shape[0], -1)
        source = ngs.GridFunction(self.space).vec
        target = source.CreateVector()
        out = np.empty(columns.shape, complex)
        for j, column in enumerate(columns.T):
            source.FV().NumPy()[:] = column
            form.Apply(source, target)
            out[:, j] = target.FV().NumPy()
        return out.reshape(x.shape)


def driven_operator(
    model: Model,
    order: int,
    *,
    elements: np.ndarray | None = None,
    faces: np.ndarray | None = None,
) -> DrivenOperator:
    """The driven operator of ``model`` at its current node positions.

    Every `PEC` surface is a Dirichlet constraint of the space; every
    other boundary and lumped port contributes pieces. ``elements`` and
    ``faces`` restrict the pieces to those gfem tetrahedra and triangles
    (boolean masks, ``None`` for all; see `terms.pieces`); the space is the
    unrestricted one.
    """
    disc = discretization(model.mesh)
    built = disc.built
    pec = pec_surfaces(model)
    space = disc.space(
        (domain_key(model), "hcurl", order, pec),
        lambda: hcurl_space(model, built, order, dirichlet=pec),
    )
    ports = terms.lumped_ports(model, built.scale)
    pieces = terms.pieces(model, built, order, ports, elements=elements, faces=faces)
    return DrivenOperator(built, space, pieces, ports, order)


class DrivenRule:
    """Host-side forward of a `Driven` problem on a `Model`.

    Construction runs every static check; `forward` reads the model's
    leaves (node coordinates, material values) and solves.
    """

    physics = "em"
    #: The operator family in the property-to-form-term table.
    form = "driven"
    #: Material properties the driven forms read.
    reads = frozenset({"eps_r", "mu_r", "tan_delta", "conductivity"})
    #: Outputs whose cotangents the backward pass handles. ``z_ref`` and the
    #: wave ports' modes (``port_*``) are constant in every differentiable
    #: input: a design may not touch a wave port face (`unsupported_input`).
    #: An adaptive sweep's kept fields differentiate through their reduced
    #: coordinates; its basis ``e_basis`` is frozen, so not differentiable.
    differentiable_outputs = frozenset(
        {
            "s",
            "z_ref",
            "e_field",
            "e_coordinates",
            "port_gamma",
            "port_z_pv",
            "port_residual",
            "port_solved",
        }
    )

    def __init__(self, model: Model, problem: Driven, options: SolverOptions) -> None:
        """Check that the problem fits the model (see `gfem.Driven`).

        Raises:
            NotImplementedError: A feature this version cannot solve.
            ValueError: The problem does not fit the model.
            gfem.SolverUnavailableError: ``linear_solver="petsc"`` and PETSc
                cannot be used.
        """
        if not isinstance(problem, Driven):
            raise TypeError(f"Expected gfem.Driven, got {problem!r}.")
        self.problem = problem
        self.options = options
        # before any assembly, and before any worker
        self.linear = ResolvedLinearSolver.resolve(
            options, default=DEFAULT_SOLVER, problem="driven"
        )
        _check_supported(model)
        if not model.ports:
            raise ValueError(
                "The model has no ports, so a driven problem has nothing to "
                "excite. Add a gfem.LumpedPort to the model's ports."
            )
        names = model.port_names
        #: The model's port names: the rows of S.
        self.port_names = names
        #: The excited port names: the columns of S.
        self.excite = names if problem.excite is None else problem.excite
        for name in self.excite:
            if name not in names:
                raise ValueError(
                    f"Driven.excite names port {name!r}, but the model's ports are "
                    f"{', '.join(map(repr, names))}."
                )
            port = model.ports[names.index(name)]
            if isinstance(port, LumpedPort) and math.isnan(resistance(port)):
                raise ValueError(
                    f"LumpedPort {name!r} has no resistance, so it has no reference "
                    "impedance to feed 1 W through. Give it a resistance, or leave "
                    "it out of Driven.excite."
                )
        #: Indices of the frequencies whose fields are kept.
        self.kept = tuple(problem.frequency.index(f) for f in problem.kept_frequencies)
        #: Whether the sweep is adaptive (a reduced model).
        self.adaptive = problem.sweep == "adaptive"
        mesh, active = model.mesh, model._domain.active  # noqa: SLF001
        #: Volume regions with tetrahedra in the solve domain: the field support.
        self.regions = tuple(r for r in mesh.volumes if active[mesh.elements(r)].any())
        #: The wave ports' names, in model port order.
        self.wave_ports = tuple(p.name for p in model.ports if isinstance(p, WavePort))
        workers_run = min(options.frequency_workers, len(problem.frequency)) > 1
        if self.wave_ports and workers_run and not self.adaptive:
            # each worker would track the port modes over its own block only
            raise NotImplementedError(
                f"WavePort {self.wave_ports[0]!r}: frequency workers cannot solve "
                "wave ports in this version of gfem, so a uniform sweep with "
                f"wave ports needs frequency_workers=1, got "
                f"{options.frequency_workers}."
            )
        #: The wave ports' modes.
        self.modes = {p.name: p.mode for p in model.ports if isinstance(p, WavePort)}
        #: The wave ports' de-embedding offsets, m.
        self.offsets = tuple(
            (p.name, p.offset) for p in model.ports if isinstance(p, WavePort)
        )
        #: Per wave port, the volume regions with an element on its face:
        #: the port mode reads their materials.
        self.port_regions = _port_regions(model)
        #: Per sheet field (index into ``model.boundaries``), the wave ports
        #: whose face shares an edge with the sheet: the port mode reads it.
        self.sheet_ports = _sheet_ports(model)
        # The fields live on the mesh as given (traced or not).
        self._mesh = model.mesh
        self._excluded = model._domain.excluded  # noqa: SLF001
        #: Whether a uniform sweep whose linear solve does not converge keeps
        #: what it solved for `partial_result`: set by `gfem.solve`'s eager
        #: path and by a frequency worker, never under JAX, where no caller
        #: reads it.
        self.keeps_partial = False
        #: What the last forward's uniform sweep solved before a linear solve
        #: did not converge (see `partial_result`); only with `keeps_partial`.
        self._partial: _Partial | None = None
        _check_low_frequency(model, problem.frequency, stacklevel=3)

    def operator(self, model: Model) -> DrivenOperator:
        """The model's driven operator at its current node positions.

        The last model's operator is kept, so that a backward pass and its
        shape terms build it once.
        """
        last = getattr(self, "_last_operator", None)
        if last is not None and last[0] is model:
            return last[1]
        op = driven_operator(model, self.options.order)
        self._last_operator = (model, op)
        return op

    def forward(
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, object], Entry | None]:
        """Solve every excited port at every frequency.

        With ``frequency_workers`` above 1 the frequencies are split over
        workers (`_forward_workers`); the outputs are the same.

        Args:
            model: The model, with concrete leaves.
            keep: Keep every port solution for a backward pass (fields
                only: the backward needs no solver for S).

        Returns:
            ``{"s": (n_freq, n_ports, n_excited) complex, "z_ref":
            (n_freq, n_ports) ohm, "iterations": Krylov iterations per
            frequency and excited port (empty for direct solves, including
            linear_solver='petsc' with ksp_type preonly)}``, plus
            ``"e_field"``, ``(ndof, n_kept * n_excited)`` complex, when
            fields are kept (frequency-major), the wave ports' modes (see
            `_port_outputs`) when there are wave ports, and the cache entry
            (``fields["solution"]``, ``(ndof, n_freq * n_excited)``,
            frequency-major) or ``None``.

        Raises:
            LinearSolverConvergenceError: A linear solve did not converge.
                A uniform sweep stops at that frequency (each frequency
                worker's block at its own first one), sets the error's
                ``frequency`` and ``unsolved``, and keeps the outputs at
                the solved frequencies for `partial_result` (with
                `keeps_partial`). An adaptive sweep sets ``frequency`` to the
                sample's.
        """
        self._partial = None
        if self.adaptive:
            return self._forward_adaptive(model, keep=keep)
        frequencies = self.problem.frequency
        if min(self.options.frequency_workers, len(frequencies)) > 1:
            return self._forward_workers(model, keep=keep)
        width = len(self.excite)
        iterations: list[tuple[int, ...]] = []
        fields: list[np.ndarray] = []
        solutions: list[np.ndarray] = []
        miss: LinearSolverConvergenceError | None = None
        n_solved = len(frequencies)
        with task_manager(self.options.threads):
            system, ports = self._ports(model)
            s = np.zeros((len(frequencies), len(self.port_names), width), complex)
            previous = None
            with system.sweep(solves=len(frequencies)) as sweep:
                for n, f in enumerate(frequencies):
                    omega = 2 * math.pi * f
                    try:
                        x, its = sweep.solve(omega, ports.rhs(omega), x0=previous)
                    except LinearSolverConvergenceError as err:
                        miss, n_solved = err, n
                        break
                    iterations.append(its)
                    previous = x
                    if keep:
                        solutions.append(np.array(x, dtype=complex))
                    if n in self.kept:
                        fields.append(np.array(x, dtype=complex))
                    s[n] = ports.outgoing(omega, x)
        # no backward pass will prepare from the pieces
        if not keep or miss is not None:
            system.free()
        outputs = {
            "s": ports.scattering(s)[:n_solved],
            "z_ref": ports.z_ref[:n_solved],
            "iterations": iterations,
            # the port mode fields' coordinates are trimmed in `_port_modes`
            **{
                name: value[:n_solved] if isinstance(value, np.ndarray) else value
                for name, value in self._port_outputs(ports.waves).items()
            },
        }
        if fields:
            outputs["e_field"] = np.concatenate(fields, axis=1)
        if miss is not None:
            self._raise_unsolved(
                outputs, tuple(range(n_solved)), [(n_solved, str(miss))], miss
            )
        entry = None
        if keep:
            entry = Entry(
                physics=self.physics,
                handle=_Kept(system, ports),
                fields={"solution": np.concatenate(solutions, axis=1)},
            )
        return outputs, entry

    def _forward_workers(
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, object], Entry | None]:
        """`forward` over frequency workers (`gfem._engine.workers`).

        The frequencies split into ``frequency_workers`` contiguous blocks
        (`numpy.array_split`), each swept by `forward` in a worker with its
        share of ``threads``; the blocks join in frequency order, so every
        output stays frequency-major. The entry holds the solutions only:
        a backward pass builds its port network and system operator here. A
        block whose linear solve does not converge returns the rows it
        solved before (`_solve_block`); once every worker is done, one error
        reports every block's miss.
        """
        frequencies = self.problem.frequency
        indices = np.array_split(
            np.arange(len(frequencies)), self.options.frequency_workers
        )
        indices = [block for block in indices if len(block)]
        threads = workers.worker_threads(self.options.threads, len(indices))
        options = dataclasses.replace(
            self.options, threads=threads, frequency_workers=1
        )
        problems = [
            Driven(
                [frequencies[i] for i in block],
                excite=self.excite,
                keep_fields=[frequencies[i] for i in block if i in self.kept],
            )
            for block in indices
        ]
        parts = workers.model_parts(model)
        results = workers.run(
            _solve_block,
            [(*parts, problem, options, keep) for problem in problems],
            [_describe(problem.frequency) for problem in problems],
            threads=threads,
        )
        outputs: dict[str, object] = {
            "s": np.concatenate([r["s"] for r in results]),
            "z_ref": np.concatenate([r["z_ref"] for r in results]),
            "iterations": [its for r in results for its in r["iterations"]],
        }
        kept = [r["e_field"] for r in results if "e_field" in r]
        if kept:
            outputs["e_field"] = np.concatenate(kept, axis=1)
        # each block stopped at its first miss: its solved rows lead it
        n_solved = [len(r["s"]) for r in results]
        misses = [
            (int(block[n]), r["miss"]["message"])
            for block, n, r in zip(indices, n_solved, results, strict=True)
            if "miss" in r
        ]
        if misses:
            solved = tuple(
                int(i)
                for block, n in zip(indices, n_solved, strict=True)
                for i in block[:n]
            )
            self._raise_unsolved(outputs, solved, misses)
        entry = None
        if keep:
            solution = np.concatenate([r["solution"] for r in results], axis=1)
            entry = Entry(
                physics=self.physics,
                handle=None,
                fields={"solution": solution},
            )
        return outputs, entry

    def _raise_unsolved(
        self,
        outputs: dict[str, object],
        solved: tuple[int, ...],
        misses: list[tuple[int, str]],
        cause: BaseException | None = None,
    ) -> NoReturn:
        """Raise a uniform sweep's `LinearSolverConvergenceError`.

        With `keeps_partial`, keeps ``outputs``, the forward outputs at the
        grid indices ``solved`` only, for `partial_result`.

        Args:
            outputs: The forward outputs over the solved frequencies.
            solved: Grid indices of the solved frequencies, in order.
            misses: ``(grid index, the linear solver's message)`` of each
                frequency where a block's linear solve did not converge
                (one per frequency worker that stopped).
            cause: The linear solver's error, when there is one.
        """
        frequencies = self.problem.frequency
        first, reason = min(misses)
        if self.keeps_partial:
            self._partial = _Partial(solved, outputs, reason)
        where = "there"
        if len(misses) > 1:
            stops = " and ".join(f"{frequencies[i]:g} Hz" for i, _ in sorted(misses))
            where = f"at {stops} (one frequency worker each)"
        message = (
            f"At {frequencies[first]:g} Hz, {reason} The uniform sweep stopped "
            f"{where}, with {len(solved)} of {len(frequencies)} frequencies solved."
        )
        raise LinearSolverConvergenceError(
            message,
            frequency=frequencies[first],
            unsolved=tuple(f for i, f in enumerate(frequencies) if i not in solved),
        ) from cause

    def partial_result(self) -> DrivenResult | None:
        """The partial result of the last forward, if its uniform sweep missed.

        `gfem.solve`'s eager path sets it as the error's ``result``, outside
        any JAX host callback. ``None`` when the last forward solved every
        frequency, was adaptive, or solved none. It is read once: the
        outputs it was built from are dropped.
        """
        partial, self._partial = self._partial, None
        if partial is None or not partial.solved:
            return None
        return self._uniform_result(partial.outputs, partial.solved)

    def _ports(self, model: Model) -> tuple[SystemOperator, PortNetwork]:
        """The system operator and the port network of one driven solve.

        Both from one set of wave port couplings: the system operator adds
        their Robin pieces and modal corrections, and the port network holds
        the excitations and S functionals. Runs each wave port's port mode
        sweep over the output grid.
        """
        op = self.operator(model)
        names = self.port_names
        columns = [names.index(n) for n in self.excite]
        couplings = wave_port_couplings(
            model,
            op.built,
            op.space,
            order=op.order,
            frequency=self.problem.frequency,
            tolerance=self.options.port_mode_tolerance,
        )
        system = SystemOperator(
            op,
            couplings,
            self.linear,
            width=len(columns),
            adjoint_tolerance=adjoint_tolerance(self.options),
        )
        ports = PortNetwork.build(
            model,
            system.operator,
            couplings,
            columns=columns,
            n_freq=len(self.problem.frequency),
        )
        return system, ports

    def _port_outputs(self, waves: list[WavePortCoupling]) -> dict[str, object]:
        """The wave ports' modes over the grid, ``(n_freq, n_wave_ports)`` each.

        ``"port_gamma"`` (1/m), ``"port_z_pv"`` (ohm), ``"port_residual"``,
        ``"port_solved"``, and the host-only ``"port_sweeps"`` (the
        `gfem._sweep.port_mode.PortModeSweep` of each, for kept mode fields).
        """
        if not waves:
            return {}
        return {
            "port_gamma": np.stack([c.sweep.gamma for c in waves], axis=1),
            "port_z_pv": np.stack([c.sweep.z_pv for c in waves], axis=1),
            "port_residual": np.stack([c.sweep.residual for c in waves], axis=1),
            "port_solved": np.stack([c.sweep.solved for c in waves], axis=1),
            "port_sweeps": [c.sweep for c in waves],
        }

    def _forward_adaptive(
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, object], Entry | None]:
        """The adaptive sweep of `forward` (see `gfem._sweep.adaptive`).

        Outputs as `forward`'s, except that ``"iterations"`` is per sample,
        kept fields come as ``"e_basis"`` ``(ndof, n)`` real and
        ``"e_coordinates"`` ``(n_kept * n_excited, n)`` (frequency-major),
        and ``"sweep"`` holds ``{"samples": grid indices in solve order,
        "errors", "converged", "basis_size"}``. With frozen samples, ``n``
        is `_padded_size` (zero columns after the basis), so the kept
        fields' shapes are known before the solve.

        The entry (``keep``) holds the reduced model only: the basis
        ``"basis"`` ``(ndof, n)``, the reduced pieces ``"operators"``
        ``(n_pieces, n, n)``, the reduced port voltages ``"voltage"``
        ``(n_ports, n)`` and the reduced solutions ``"coordinates"``
        ``(n_freq, n, n_excited)``; no solver.
        """
        frequencies = self.problem.frequency
        options = self.options
        iterations: list[tuple[int, ...]] = []
        # Palace's dimensionless frequency: omega times the largest extent / c
        extent = np.ptp(np.asarray(model.mesh.nodes), axis=0).max()
        z_scale = float(extent) / scipy.constants.c
        with task_manager(options.threads):
            system, ports = self._ports(model)
            samples = self.problem.samples
            # samples lie far apart: a reused BDDC would mostly hit its cap
            with system.sweep(reuse_preconditioner=False) as system_sweep:

                def solve(
                    omega: float, rhs: np.ndarray, x0: np.ndarray | None
                ) -> np.ndarray:
                    try:
                        x, its = system_sweep.solve(omega, rhs, x0=x0)
                    except LinearSolverConvergenceError as err:
                        raise _sample_error(frequencies, omega, err) from err
                    iterations.append(its)
                    return x

                affine = AffineSystem(
                    thetas=[piece.theta for piece in system.operator.pieces],
                    apply=system.operator.apply,
                    excitation=ports.excitation,
                    outputs=ports.outputs,
                    solve=solve,
                    corrections=[
                        AffineTerm(c.vectors, c.correction_coefficients)
                        for c in ports.waves
                    ],
                    adjoint_loads=ports.adjoint_loads,
                )
                sweep = adaptive_sweep(
                    affine,
                    2 * np.pi * np.asarray(frequencies),
                    tolerance=options.adaptive_tolerance,
                    max_samples=options.adaptive_max_samples,
                    memory=options.adaptive_memory,
                    samples=None
                    if samples is None
                    else [frequencies.index(f) for f in samples],
                    z_scale=z_scale,
                )
        system.free()  # the backward pass solves the reduced model only
        outputs: dict[str, object] = {
            "s": ports.scattering(sweep.outputs),
            "z_ref": ports.z_ref,
            "iterations": iterations,
            "sweep": {
                "samples": sweep.samples,
                "errors": sweep.errors,
                "converged": sweep.converged,
                "basis_size": sweep.model.size,
            },
            **self._port_outputs(ports.waves),
        }
        size = sweep.model.size
        if self.kept:
            coordinates = sweep.coordinates[list(self.kept)]  # (n_kept, n, n_exc)
            coordinates = np.swapaxes(coordinates, 1, 2).reshape(-1, size)
            width = size if samples is None else self._padded_size()
            basis = np.zeros((sweep.basis.shape[0], width))
            basis[:, :size] = sweep.basis
            padded = np.zeros((len(coordinates), width), complex)
            padded[:, :size] = coordinates
            outputs["e_basis"] = basis
            outputs["e_coordinates"] = padded
        entry = None
        if keep:
            omegas = 2 * np.pi * np.asarray(frequencies)
            reduced = sweep.model
            entry = Entry(
                physics=self.physics,
                fields={
                    "basis": sweep.basis,
                    "matrices": np.stack([reduced.matrix(w) for w in omegas]),
                    "functionals": np.stack(
                        [ports.reduced_functional(w, reduced.outputs) for w in omegas]
                    ),
                    "deembedding": ports.deembedding,
                    "coordinates": sweep.coordinates,
                },
            )
        return outputs, entry

    def _padded_size(self) -> int:
        """Most basis vectors frozen samples give.

        Two per sample and excited port, and two per sample and wave port
        for its S functional's adjoint (`PortNetwork.adjoint_loads`).
        """
        samples = self.problem.samples or ()
        return 2 * len(samples) * (len(self.excite) + len(self.wave_ports))

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output."""
        if self.adaptive and self.problem.samples is None:
            raise NotImplementedError(
                "Driven(sweep='adaptive') runs under jax.grad, jax.jit or jax.vmap "
                "only with a frozen sample set: the greedy's samples and basis size "
                "are only known after the solve. Solve once outside the "
                "transformation and pass Driven(..., samples="
                "result.sweep.sampled_frequencies), or use sweep='uniform'."
            )
        n_f, n_p = len(self.problem.frequency), len(model.ports)
        n_exc = len(self.excite)
        specs = {
            "s": jax.ShapeDtypeStruct((n_f, n_p, n_exc), jnp.complex128),
            "z_ref": jax.ShapeDtypeStruct((n_f, n_p), jnp.float64),
        }
        n_w = len(self.wave_ports)
        if n_w:
            if self.kept:
                raise NotImplementedError(
                    "Driven(keep_fields=...) with wave ports cannot run under "
                    "jax.jit, jax.grad or jax.vmap in this version of gfem (the "
                    "port mode fields' size is known only after the solve). Keep "
                    "no fields, or call gfem.solve outside the transformation."
                )
            specs["port_gamma"] = jax.ShapeDtypeStruct((n_f, n_w), jnp.complex128)
            specs["port_z_pv"] = jax.ShapeDtypeStruct((n_f, n_w), jnp.float64)
            specs["port_residual"] = jax.ShapeDtypeStruct((n_f, n_w), jnp.float64)
            specs["port_solved"] = jax.ShapeDtypeStruct((n_f, n_w), jnp.bool_)
        if self.kept:
            mesh = model.mesh
            still = Mesh(mesh.imported_nodes, mesh._topology)  # noqa: SLF001
            with task_manager(self.options.threads):
                _, space, _ = fe_space(
                    still, self.regions, self._excluded, "hcurl", self.options.order
                )
            n_fields = len(self.kept) * n_exc
            if self.adaptive:
                n = self._padded_size()
                specs["e_basis"] = jax.ShapeDtypeStruct((space.ndof, n), jnp.float64)
                specs["e_coordinates"] = jax.ShapeDtypeStruct(
                    (n_fields, n), jnp.complex128
                )
            else:
                specs["e_field"] = jax.ShapeDtypeStruct(
                    (space.ndof, n_fields), jnp.complex128
                )
        return specs

    def role(self, kind: tuple) -> str:
        """``"read"`` if the driven solve depends on input ``kind``.

        The electromagnetic material properties, the node coordinates and
        any boundary or port value are read; thermal properties are not, so
        their gradient is an exact zero.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        return "read"

    def shape_terms(self, model: Model) -> list[ShapeTerm]:
        """The operator's pieces as node-coordinate (shape) terms.

        See `operator_shape_terms`.

        Raises:
            PortGeometryError: A node of a port surface has moved.
        """
        return operator_shape_terms(model, self.operator(model))

    def real_space(self, model: Model) -> ngs.FESpace:
        """A real H(curl) space with the solve's DOF numbering (for ``DiffShape``)."""
        return real_space(model, self.operator(model))

    def unsupported_input(self, kind: tuple) -> str | None:
        """Why a read input cannot be differentiated here, or ``None``.

        A wave port's mode is a constant of the solve, so the design must
        stay off its face: a perturbed material of a region with an element
        on a wave port face raises.

        A sheet field whose surface shares an edge with a wave port face
        raises likewise: the port mode reads the sheet on that edge.

        Raises:
            PortGeometryError: ``kind`` is the material of a region that
                touches a wave port face, or the fill of a sheet beside one.
        """
        if is_sheet(kind):
            ports = self.sheet_ports.get(kind[1], ())
            if ports:
                raise PortGeometryError(
                    f"WavePort {ports[0]!r}: the sheet field of model.boundaries"
                    f"[{kind[1]}] shares an edge with the port face, so gfem cannot "
                    "differentiate with respect to its fill: the port mode reads "
                    "the sheet there and is a constant of the solve. Keep the "
                    "design away from wave port faces (split the sheet so the part "
                    "touching the port is a boundary of its own without fill), or "
                    "wrap the fill in jax.lax.stop_gradient."
                )
            return None
        if kind[0] != "material":
            return None
        region, prop = kind[1], kind[2]
        touched = [p for p, regions in self.port_regions.items() if region in regions]
        if touched:
            raise PortGeometryError(
                f"WavePort {touched[0]!r}: region {region!r} has elements on the "
                f"port face, so gfem cannot differentiate with respect to "
                f"model.materials[{region!r}].{prop}: the port mode reads that "
                "material and is a constant of the solve. Keep the design away "
                "from wave port faces (give the elements beside the port a "
                "region of their own with a constant gfem.Material), or wrap "
                "the value in jax.lax.stop_gradient."
            )
        return None

    def nondifferentiable_output(self, name: str) -> str:
        """The error message for a cotangent on output ``name``."""
        return (
            f"gfem cannot differentiate the driven output {name!r} in this "
            "version. Wrap it in jax.lax.stop_gradient to use its value without "
            "its derivative."
        )

    def backward(
        self, model: Model, entry: Entry, cotangents: dict[str, np.ndarray]
    ) -> list[FieldPair]:
        """Fold the S and kept-field cotangents into one field pair per frequency.

        At frequency n, with ``X`` the excited ports' solutions, ``f_i`` port
        i's S functional (``S_ij = f_iᵀ x_j``, de-embedding aside) and ``W``
        the S cotangent, the pair is ``(-Lambda, X)`` with ``Lambda_j =
        sum_i W_ij A^-1 f_i + A^-1 g_j``, ``g_j`` the cotangent of a kept
        field of port j and ``A`` the system operator (wave-port Robin terms
        and modal corrections included, complex symmetric): ``sum ct dy/dp
        = -sum_j Lambda_j^T (dA/dp) x_j`` (the excitations and the modal
        corrections depend on the port modes only). When ``f_i`` is parallel
        to port i's excitation (every lumped port, a wave port whose mode is
        real up to a phase), ``A^-1 f_i`` is port i's own solution scaled
        (no solve). The remaining loads (kept-field cotangents, rows of
        ports that were not excited, lossy wave ports, whose S functional
        projects on the conjugated mode) take one multi-column adjoint
        solve. ``z_ref`` contributes nothing.
        """
        if self.adaptive:
            return self._backward_adaptive(model, entry, cotangents)
        solution = entry.fields["solution"]
        width = len(self.excite)
        ct_s = cotangents.get("s")
        ct_e = cotangents.get("e_field")
        pairs = []
        with task_manager(self.options.threads):
            if not isinstance(entry.handle, _Kept):  # frequency workers keep none
                entry.handle = _Kept(*self._ports(model))
            system, ports = entry.handle.system, entry.handle.ports
            op = system.operator
            domain = em_region(model, op.built)
            for n, f in enumerate(self.problem.frequency):
                omega = 2 * math.pi * f
                x = solution[:, n * width : (n + 1) * width]
                if ct_s is not None:
                    split = ports.pullback(n, omega, ct_s[n], x)
                    adjoint, load = split.free, split.loads
                    undefined = split.undefined
                else:
                    adjoint, load = np.zeros_like(x), np.zeros_like(x)
                    undefined = np.zeros(width, bool)
                if ct_e is not None and n in self.kept:
                    k = self.kept.index(n)
                    load += np.asarray(ct_e)[:, k * width : (k + 1) * width]
                if load.any():
                    try:
                        adjoint += system.solve_adjoint(n, omega, load)
                    except LinearSolverConvergenceError as err:
                        raise _adjoint_error(f, err) from err
                adjoint[:, undefined] = np.nan
                pairs.append(
                    FieldPair(
                        space=op.space,
                        left=-adjoint,
                        right=x,
                        weights=op.weights(omega),
                        order=op.order,
                        domain=domain,
                    )
                )
        return pairs

    def _backward_adaptive(
        self, model: Model, entry: Entry, cotangents: dict[str, np.ndarray]
    ) -> list[FieldPair]:
        """The frozen-basis backward of an adaptive sweep: no full-size solve.

        With the basis V and the sample set held fixed, the reduced solution
        at ω solves ``A_r(ω) y = b_r(ω)``, ``A_r = Vᵀ A(ω) V`` complex
        symmetric (wave-port Robin terms and modal corrections included).
        A cotangent ``g`` on ``y`` (from S through the reduced S
        functionals, the conjugated-mode projection for wave ports, and from
        kept fields' coordinates) gives the reduced adjoint ``λ = A_r^-T g``
        (n x n) and ``sum ct dy/dp = -Σ_t Σ_ab G_t,ab a_t'(V_a, V_b)`` with
        ``G_t = Σ_ω θ_t(ω) Λ(ω) Y(ω)ᵀ``. Each ``G_t`` (n x n) factors by
        SVD into at most n pairs ``(V a, V b)``, so the element passes do
        not grow with the number of frequencies. A cotangent on the basis
        itself is refused by the wrapper.
        """
        fields = entry.fields
        basis, matrices = fields["basis"], fields["matrices"]
        functionals, coordinates = fields["functionals"], fields["coordinates"]
        deembedding = fields["deembedding"]
        columns = [self.port_names.index(p) for p in self.excite]
        n, n_exc = basis.shape[1], len(self.excite)
        ct_s = cotangents.get("s")
        ct_c = cotangents.get("e_coordinates")
        accumulated: dict[str, np.ndarray] = {}
        with task_manager(self.options.threads):
            op = self.operator(model)
            domain = em_region(model, op.built)
            for i, f in enumerate(self.problem.frequency):
                omega = 2 * math.pi * f
                g = np.zeros((n, n_exc), complex)
                if ct_s is not None:
                    split = split_cotangent(
                        ct_s[i], deembedding[i], columns, functionals[i].T
                    )
                    g += split.loads
                    g[:, split.undefined] = np.nan
                if ct_c is not None and i in self.kept:
                    k = self.kept.index(i)
                    g += np.asarray(ct_c)[k * n_exc : (k + 1) * n_exc, :n].T
                if not g.any():
                    continue
                outer = np.linalg.solve(matrices[i].T, g) @ coordinates[i].T
                for term, theta in op.weights(omega).items():
                    accumulated[term] = accumulated.get(term, 0) + theta * outer
            pairs = []
            for term, g in accumulated.items():
                left, right = _factor(g)
                pairs.append(
                    FieldPair(
                        space=op.space,
                        left=-(basis @ left),
                        right=basis @ right,
                        weights={term: 1.0},
                        order=op.order,
                        domain=domain,
                    )
                )
        return pairs

    def result(self, outputs: dict) -> DrivenResult:
        """The public result from the forward outputs."""
        frequencies = self.problem.frequency
        kept = [(frequencies[n], port) for n in self.kept for port in self.excite]
        if self.adaptive:
            return self._adaptive_result(outputs, kept)
        return self._uniform_result(outputs)

    def _uniform_result(
        self, outputs: dict, solved: tuple[int, ...] | None = None
    ) -> DrivenResult:
        """`result` of a uniform sweep over the grid indices ``solved``.

        ``outputs`` holds the solved frequencies' rows only; ``solved=None``
        is every frequency.
        """
        grid = self.problem.frequency
        frequencies = grid if solved is None else tuple(grid[i] for i in solved)
        kept = [
            (grid[n], port)
            for n in self.kept
            if solved is None or n in solved
            for port in self.excite
        ]
        e_fields = [
            Field(
                coefficients=outputs["e_field"][:, k],
                mesh=self._mesh,
                quantity="E",
                unit="V/m",
                space="hcurl",
                order=self.options.order,
                support=self.regions,
                omega=jnp.asarray(2 * math.pi * f),
                _excluded=self._excluded,
            )
            for k, (f, _) in enumerate(kept)
        ]
        return DrivenResult(
            s=NetworkParameters(
                values=jnp.asarray(outputs["s"]),
                z_ref=jnp.asarray(outputs["z_ref"]),
                frequency=frequencies,
                ports=self.port_names,
                excited=tuple(self.excite),
            ),
            e_fields=tuple(e_fields),
            kept=tuple(kept),
            port_modes=self._port_modes(outputs, solved),
            offsets=self.offsets,
        )

    def _port_modes(
        self, outputs: dict, solved: tuple[int, ...] | None = None
    ) -> dict[str, PortModeResult]:
        """The wave ports' `PortModeResult` from the forward outputs.

        ``solved``: the grid indices the outputs' rows are at (``None``:
        every frequency).
        """
        if not self.wave_ports:
            return {}
        sweeps = outputs.get("port_sweeps")
        grid = self.problem.frequency
        rows = list(range(len(grid)) if solved is None else solved)
        frequency = jnp.asarray([grid[i] for i in rows])
        out = {}
        for k, name in enumerate(self.wave_ports):
            field = None
            if self.kept and sweeps is not None:
                sweep = sweeps[k]
                # `forward` trims the other port outputs to the solved rows
                coords = sweep.coords if solved is None else sweep.coords[rows]
                field = PortModeField(sweep.space, sweep.basis, coords)
            out[name] = PortModeResult(
                port=name,
                mode=self.modes[name],
                frequency=frequency,
                gamma=jnp.asarray(outputs["port_gamma"][:, k]),
                Z_PV=jnp.asarray(outputs["port_z_pv"][:, k]),
                residual=jnp.asarray(outputs["port_residual"][:, k]),
                solved=jnp.asarray(outputs["port_solved"][:, k]),
                e_field=field,
            )
        return out

    def _adaptive_result(
        self, outputs: dict, kept: list[tuple[float, str]]
    ) -> DrivenResult:
        """`result` of an adaptive sweep: basis-backed fields, `SweepInfo`."""
        frequencies = self.problem.frequency
        # Under a JAX transformation only the frozen sample set is known.
        info: dict[str, Any] = outputs.get("sweep") or {
            "samples": [frequencies.index(f) for f in self.problem.samples or ()],
            "errors": None,
            "converged": None,
            "basis_size": None,
        }
        sampled = set(info["samples"])
        sweep = SweepInfo(
            kind="adaptive",
            solved=tuple(i in sampled for i in range(len(frequencies))),
            sampled_frequencies=tuple(frequencies[i] for i in info["samples"]),
            sample_errors=info["errors"],
            converged=info["converged"],
            tolerance=self.options.adaptive_tolerance,
            basis_size=info["basis_size"],
        )
        e_fields: ReducedFields | tuple = ()
        if kept:
            e_fields = ReducedFields(
                basis=outputs["e_basis"],
                coordinates=outputs["e_coordinates"],
                omega=jnp.asarray([2 * math.pi * f for f, _ in kept]),
                mesh=self._mesh,
                quantity="E",
                unit="V/m",
                space="hcurl",
                order=self.options.order,
                support=self.regions,
                _excluded=self._excluded,
            )
        return DrivenResult(
            s=NetworkParameters(
                values=jnp.asarray(outputs["s"]),
                z_ref=jnp.asarray(outputs["z_ref"]),
                frequency=frequencies,
                ports=self.port_names,
                excited=tuple(self.excite),
            ),
            e_fields=e_fields,
            kept=tuple(kept),
            sweep=sweep,
            port_modes=self._port_modes(outputs),
            offsets=self.offsets,
        )


def operator_shape_terms(model: Model, op: DrivenOperator) -> list[ShapeTerm]:
    """The pieces of ``model``'s driven operator as node-coordinate (shape) terms.

    Every piece but the ports': a port's terms, excitation and voltage
    read only its own surface, which shape gradients hold in place. The
    eigenmode rule's operator is the same pieces, so it shares these.

    Raises:
        PortGeometryError: A node of a port surface has moved.
    """
    check_ports_held(model)
    out = []
    for piece in op.pieces:
        if piece.kind == "port":
            continue
        if piece.kind == "surface":
            term = ShapeTerm(
                piece.name,
                "surface",
                integrand=piece.integrand,
                space=real_space(model, op),
            )
        else:
            term = ShapeTerm(
                piece.name,
                piece.kind,
                coefficient=piece.coefficient,
                region=piece.region,
            )
        out.append(term)
    return out


def real_space(model: Model, op: DrivenOperator) -> ngs.FESpace:
    """A real H(curl) space with ``op``'s DOF numbering (for ``DiffShape``)."""
    pec = pec_surfaces(model)
    return discretization(model.mesh).space(
        (domain_key(model), "hcurl", op.order, pec, "real"),
        lambda: hcurl_space(
            model, op.built, op.order, complex_valued=False, dirichlet=pec
        ),
    )


def check_ports_held(model: Model) -> None:
    """Raise unless every port surface's nodes sit at their imported positions.

    Shape gradients treat each port's geometry (its terms, excitation and
    voltage) as frozen: exact for every displacement that holds the ports.

    Raises:
        PortGeometryError: A node of a port surface has moved.
    """
    mesh = model.mesh
    nodes = np.asarray(mesh.nodes)
    moved = np.any(nodes != mesh.imported_nodes, axis=1)
    if not moved.any():
        return
    for port in model.ports:
        for surface in surfaces_of(port):
            on = mesh.triangles[mesh.triangle_tags == mesh.region(surface).tag]
            off = np.unique(on)[moved[np.unique(on)]]
            if len(off):
                distance = np.linalg.norm(
                    nodes[off] - mesh.imported_nodes[off], axis=1
                ).max()
                raise PortGeometryError(
                    f"{type(port).__name__} {port.name!r}: {len(off)} node(s) of its "
                    f"surface {surface!r} moved (up to {distance:.3g} m), so gfem "
                    "cannot differentiate the solve with respect to the node "
                    "coordinates: shape gradients hold every port's geometry fixed. "
                    "Keep the displacement zero on port surfaces: "
                    "gfem.shape.boundary_morph holds the model's ports (and "
                    "fixed= surfaces) for you."
                )


def _check_low_frequency(
    model: Model, frequencies: tuple[float, ...], *, stacklevel: int
) -> None:
    """Warn once if a frequency is below the mesh's low-frequency floor.

    h_min is the shortest edge of the solve domain on the imported nodes,
    so the message is the same at every step of a shape optimization.

    Args:
        model: The model, its solve domain and its mesh.
        frequencies: The output frequencies (Hz).
        stacklevel: Passed on to `warnings.warn`, counted from the caller.
    """
    active = model._domain.active  # noqa: SLF001
    h_min = float(_shortest_edges(model.mesh)[active].min())
    f = np.asarray(frequencies)
    k0h = 2 * math.pi * f / scipy.constants.c * h_min
    low = k0h < LOW_FREQUENCY_FLOOR
    if not low.any():
        return
    f_floor = LOW_FREQUENCY_FLOOR * scipy.constants.c / (2 * math.pi * h_min)
    lowest = int(np.argmin(f))
    msg = (
        f"Driven: {int(low.sum())} of {len(f)} frequencies, the lowest "
        f"{f[lowest]:.4g} Hz, are below this mesh's low-frequency floor of "
        f"{f_floor:.4g} Hz (k0*h_min = {k0h[lowest]:.2g} < "
        f"{LOW_FREQUENCY_FLOOR:g}, with h_min = {h_min:.3g} m the shortest edge "
        "in the solve domain). Below it the fields and their gradients, and "
        "with linear_solver='direct', or 'petsc' with its default "
        "factorization, the capacitive part of S, "
        "carry rounding errors up to about 1e-16/(k0*h_min)^2, and MKL "
        "PARDISO ('direct' with the mkl extra) becomes unreliable below "
        f"k0*h_min = {_PARDISO_FLOOR:g}. For capacitance, solve "
        "gfem.Electrostatic instead of reading it from S; otherwise raise the "
        "lowest frequency or coarsen the smallest elements."
    )
    warnings.warn(msg, LowFrequencyWarning, stacklevel=stacklevel + 1)


def _solve_block(
    aux: Any, children: Any, problem: Driven, options: SolverOptions, keep: bool
) -> dict[str, Any]:
    """A frequency worker's `DrivenRule.forward` of its block, as arrays.

    ``aux`` and ``children`` are the model's parts (`gfem._engine.workers`).
    A block whose linear solve does not converge at some frequency stops
    there and returns the rows it solved, plus ``"miss"``: ``{"message":
    the linear solver's message}``, plain data that pickles (the parent
    raises once for every block).
    """
    model = workers.model_from_parts(aux, children)
    with warnings.catch_warnings():
        # the parent's rule already warned for the whole sweep
        warnings.simplefilter("ignore", LowFrequencyWarning)
        rule = DrivenRule(model, problem, options)
    rule.keeps_partial = True
    names = ("s", "z_ref", "iterations", "e_field")
    try:
        outputs, entry = rule.forward(model, keep=keep)
    except LinearSolverConvergenceError:
        partial = rule._partial  # noqa: SLF001
        if partial is None:  # pragma: no cover - a uniform sweep always sets it
            raise
        block = {
            name: partial.outputs[name] for name in names if name in partial.outputs
        }
        block["miss"] = {"message": partial.reason}
        return block
    block = {name: outputs[name] for name in names if name in outputs}
    if entry is not None:
        block["solution"] = entry.fields["solution"]
        entry.free()
    return block


def _sample_error(
    frequencies: tuple[float, ...], omega: float, err: LinearSolverConvergenceError
) -> LinearSolverConvergenceError:
    """An adaptive sweep's error where a sample's linear solve did not converge.

    No partial result: every output of the reduced model depends on every
    sample. ``frequency`` is the sample's grid frequency.
    """
    f = min(frequencies, key=lambda g: abs(2 * math.pi * g - omega))
    return LinearSolverConvergenceError(
        f"At the adaptive sweep's sample {f:g} Hz, {err} The reduced model needs "
        "every sample, so no frequency has a result: use linear_solver='direct', "
        "or a uniform sweep to find the frequencies that converge.",
        frequency=f,
    )


def _adjoint_error(
    f: float, err: LinearSolverConvergenceError
) -> LinearSolverConvergenceError:
    """A backward pass's error where the adjoint solve at ``f`` Hz did not converge.

    No partial result: a gradient needs every frequency's adjoint.
    """
    return LinearSolverConvergenceError(
        f"At {f:g} Hz, the adjoint solve of the gradient's backward pass: {err}",
        frequency=f,
    )


def _describe(frequencies: tuple[float, ...]) -> str:
    """A block of frequencies, for errors."""
    first, last = frequencies[0], frequencies[-1]
    if first == last:
        return f"{first:g} Hz"
    return f"{first:g} to {last:g} Hz"


#: Singular values of a reduced cotangent matrix below this fraction of the
#: largest one are dropped (their pairs add nothing).
FACTOR_RANK = 1e-14


def _factor(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(a, b)`` with ``g = a @ b.T`` and few columns (truncated SVD).

    A matrix with a non-finite entry (an S row of a port without a
    resistance) gives a NaN column, so the gradient is NaN like the value.
    """
    if not np.isfinite(g).all():
        column = np.full((g.shape[0], 1), np.nan)
        return column, column
    u, sigma, vh = np.linalg.svd(g)
    rank = max(1, int(np.count_nonzero(sigma > FACTOR_RANK * sigma[0])))
    return u[:, :rank] * sigma[:rank], vh[:rank].T


@dataclasses.dataclass(eq=False)
class _Kept:
    """A forward's system operator and port network, kept for its backward pass.

    The cache entry's ``handle`` (`gfem._engine.cache.Entry`): a backward
    pass reuses the adjoint handles and the pieces, and runs no port mode
    sweep again.
    """

    system: SystemOperator
    ports: PortNetwork

    @property
    def nbytes(self) -> int:
        """Host memory held, in bytes."""
        return self.system.nbytes + self.ports.nbytes

    def free(self) -> None:
        """Release the system operator's handles and pieces, and drop both."""
        self.system.free()
        del self.system, self.ports


@dataclasses.dataclass(frozen=True)
class _Partial:
    """What a uniform sweep solved before a linear solve did not converge."""

    #: Grid indices of the solved frequencies, in order.
    solved: tuple[int, ...]
    #: The forward outputs at the solved frequencies only.
    outputs: dict[str, object]
    #: The linear solver's message where the (first) linear solve did not
    #: converge.
    reason: str


def _sheet_ports(model: Model) -> dict[int, tuple[str, ...]]:
    """Per sheet-field boundary index, the wave ports sharing an edge with it."""
    mesh = model.mesh
    out = {}
    for i, b in enumerate(model.boundaries):
        if sheet_of(b) is None:
            continue
        sheet = _edges(mesh.triangles[mesh.elements(b.surface)])
        ports = tuple(
            p.name
            for p in model.ports
            if isinstance(p, WavePort)
            and sheet & _edges(mesh.triangles[mesh.elements(p.surface)])
        )
        if ports:
            out[i] = ports
    return out


def _edges(triangles: np.ndarray) -> set[tuple[int, int]]:
    pairs = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    return {(int(a), int(b)) for a, b in pairs}


def _port_regions(model: Model) -> dict[str, tuple[str, ...]]:
    """Per wave port, the solve-domain volume regions beside its face."""
    mesh = model.mesh
    active = model._domain.active  # noqa: SLF001
    names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    out = {}
    for port in model.ports:
        if not isinstance(port, WavePort):
            continue
        on = mesh.triangle_tags == mesh.region(port.surface).tag
        nb = mesh.triangle_neighbours[on]
        tets = nb[nb >= 0]
        tags = np.unique(mesh.tetrahedron_tags[tets[active[tets]]])
        out[port.name] = tuple(names[int(t)] for t in tags)
    return out


def _check_supported(model: Model) -> None:
    """Raise for ports this version cannot solve."""
    for port in model.ports:
        if not isinstance(port, (LumpedPort, WavePort)):  # pragma: no cover
            raise TypeError(f"Unknown port {port!r}.")
