"""Driven rule: time-harmonic E field in complex Nédélec elements, lumped ports.

At each frequency the operator ``A(ω) = Σ θ_k(ω) A_k`` (see
`gfem._engine.terms`) is assembled and prepared once, and every excited
port's right-hand side is solved in one handle call: GMRES + BDDC by
default, each port starting from its solution at the previous frequency,
or one sparse Cholesky factorization (``direct``). ``A`` is complex
symmetric (A^T = A), and a port's excitation vector is proportional to its
voltage functional, so S is symmetric to solver precision.

S is read from the port voltages: with the excited port fed 1 W incident
power (incident voltage sqrt(2 R)), ``S_ij = V_i / sqrt(2 R_i) - δ_ij``,
the power wave of port i with reference impedance R_i. The solution
itself (volts per mesh file unit) is that 1 W field's peak phasor in the
`gfem.Field` coefficient convention, so kept fields are the solution
columns as they are.

Gradients (`DrivenRule.backward`): with ``A x_j = i k0 b_j`` and ``b_i``
parallel to the voltage vector ``l_i``, port i's solution is (a multiple
of) the adjoint of its own voltage, ``A^-1 l_i = x_i / (i k0 b_i / l_i)``
(A^T = A, no transpose, no conjugate). An S cotangent therefore needs no
solve when every port it reads is excited: per frequency it folds into
one pair ``(-sum_i W_ij A^-1 l_i, x_j)``. A kept-field cotangent, or an S
row of a port that was not excited, takes one multi-column adjoint solve
per frequency; that frequency's operator is prepared on first use at the
adjoint tolerance and kept in the cache entry for later pullbacks.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem._adjoint import adjoint_tolerance
from gfem._boundaries import PEC, LumpedPort, WavePort
from gfem._engine import terms
from gfem._engine.cache import Entry, discretization
from gfem._engine.domain import domain_key, em_region, hcurl_space
from gfem._engine.fields import fe_space
from gfem._engine.mesh import EngineMesh
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import FieldPair
from gfem._engine.wave_ports import WavePortCoupling, wave_port_couplings
from gfem._linalg import KINDS, Handle, Operator, prepare
from gfem._sweep.adaptive import AffineSystem, AffineTerm, adaptive_sweep
from gfem.mesh import Mesh
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

__all__ = ["DEFAULT_SOLVER", "DrivenOperator", "DrivenRule", "driven_operator"]

#: The linear solver for ``linear_solver="auto"``: GMRES + BDDC.
DEFAULT_SOLVER = "bddc_gmres"


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


def driven_operator(model: Model, order: int) -> DrivenOperator:
    """The driven operator of ``model`` at its current node positions.

    Every `PEC` surface is a Dirichlet constraint of the space; every
    other boundary and lumped port contributes pieces.
    """
    disc = discretization(model.mesh)
    built = disc.built
    pec = tuple(sorted({b.surface for b in model.boundaries if isinstance(b, PEC)}))
    space = disc.space(
        (domain_key(model), "hcurl", order, pec),
        lambda: hcurl_space(model, built, order, dirichlet=pec),
    )
    ports = terms.lumped_ports(model, built.scale)
    pieces = terms.pieces(model, built, order, ports)
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
    #: Outputs whose cotangents the backward pass handles (``z_ref``, the
    #: port resistances, is constant in every differentiable input).
    differentiable_outputs = frozenset({"s", "z_ref", "e_field"})

    def __init__(self, model: Model, problem: Driven, options: SolverOptions) -> None:
        """Check that the problem fits the model (see `gfem.Driven`).

        Raises:
            NotImplementedError: A feature this version cannot solve.
            ValueError: The problem does not fit the model.
        """
        if not isinstance(problem, Driven):
            raise TypeError(f"Expected gfem.Driven, got {problem!r}.")
        self.problem = problem
        self.options = options
        kind = options.linear_solver
        self.kind = DEFAULT_SOLVER if kind == "auto" else kind
        if self.kind not in KINDS:
            raise NotImplementedError(
                f"linear_solver={kind!r} is not available for driven problems in "
                "this version of gfem; use one of "
                f"{', '.join(repr(k) for k in KINDS)}."
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
            if isinstance(port, LumpedPort) and math.isnan(_resistance(port)):
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
        #: The wave ports' modes.
        self.modes = {p.name: p.mode for p in model.ports if isinstance(p, WavePort)}
        #: The wave ports' de-embedding offsets, m.
        self.offsets = tuple(
            (p.name, p.offset) for p in model.ports if isinstance(p, WavePort)
        )
        # The fields live on the mesh as given (traced or not).
        self._mesh = model.mesh
        self._excluded = model._domain.excluded  # noqa: SLF001

    def operator(self, model: Model) -> DrivenOperator:
        """The model's driven operator at its current node positions."""
        return driven_operator(model, self.options.order)

    def forward(
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, object], Entry | None]:
        """Solve every excited port at every frequency.

        Args:
            model: The model, with concrete leaves.
            keep: Keep every port solution for a backward pass (fields
                only: the backward needs no solver for S).

        Returns:
            ``{"s": (n_freq, n_ports, n_excited) complex, "z_ref":
            (n_freq, n_ports) ohm, "iterations": Krylov iterations per
            frequency and excited port (empty for direct)}``, plus
            ``"e_field"``, ``(ndof, n_kept * n_excited)`` complex, when
            fields are kept (frequency-major), the wave ports' modes (see
            `_port_outputs`) when there are wave ports, and the cache entry
            (``fields["solution"]``, ``(ndof, n_freq * n_excited)``,
            frequency-major) or ``None``.
        """
        if self.adaptive:
            # never under differentiation: output_specs refuses adaptive sweeps
            return self._forward_adaptive(model), None
        frequencies = self.problem.frequency
        width = len(self.excite)
        iterations: list[tuple[int, ...]] = []
        fields: list[np.ndarray] = []
        solutions: list[np.ndarray] = []
        with task_manager(self.options.threads):
            ports = self._ports(model)
            op = ports.operator
            free = op.space.FreeDofs()
            s = np.zeros((len(frequencies), len(self.port_names), width), complex)
            previous = None
            for n, f in enumerate(frequencies):
                omega = 2 * math.pi * f
                handle = prepare(
                    Operator(op.form(omega), free),
                    symmetry="complex_symmetric",
                    kind=self.kind,
                    device=self.options.device,
                    tolerance=self.options.tolerance,
                    width=width,
                )
                try:
                    x = handle.solve(ports.system.rhs(omega), x0=previous)
                    its = getattr(handle, "iterations", [])
                    iterations.append(tuple(int(i) for i in its))
                finally:
                    handle.free()
                previous = x
                if keep:
                    solutions.append(np.array(x, dtype=complex))
                if n in self.kept:
                    fields.append(np.array(x, dtype=complex))
                s[n] = sum(
                    np.asarray(t.coefficients(omega)) @ (t.vectors.T @ x)
                    for t in ports.system.outputs
                )
        outputs = {
            "s": ports.scattering(s),
            "z_ref": ports.z_ref,
            "iterations": iterations,
            **self._port_outputs(ports.waves),
        }
        if self.kept:
            outputs["e_field"] = np.concatenate(fields, axis=1)
        entry = None
        if keep:
            entry = Entry(
                physics=self.physics,
                handle=_Handles(),
                fields={"solution": np.concatenate(solutions, axis=1)},
            )
        return outputs, entry

    def _ports(self, model: Model) -> _Ports:
        """Every port's terms (see `_Ports`).

        The operator with its wave-port pieces, and the excitations and S
        functionals as affine terms. Runs each wave port's port mode sweep
        over the output grid.
        """
        op = self.operator(model)
        names = self.port_names
        columns = [names.index(n) for n in self.excite]
        n_ports, width = len(names), len(columns)
        couplings = wave_port_couplings(
            model,
            op.built,
            op.space,
            order=op.order,
            frequency=self.problem.frequency,
            tolerance=self.options.port_mode_tolerance,
        )
        if couplings:
            op = dataclasses.replace(
                op, pieces=[*op.pieces, *(c.piece for c in couplings)]
            )
        excitation: list[AffineTerm] = []
        outputs: list[AffineTerm] = []
        lumped = [names.index(data.port.name) for data in op.ports]
        if lumped:
            vectors = [
                terms.port_vectors(op.space, op.built, op.order, data)
                for data in op.ports
            ]
            resistance = np.array([_resistance(model.ports[i]) for i in lumped])
            fed = [(k, columns.index(i)) for k, i in enumerate(lumped) if i in columns]
            if fed:
                b = np.stack([vectors[k][0] for k, _ in fed], axis=1)
                select = np.zeros((len(fed), width))
                for a, (_, pos) in enumerate(fed):
                    select[a, pos] = 1.0

                def lumped_rhs(w: float, select=select) -> np.ndarray:
                    return 1j * op.wavenumber(w) * select

                excitation.append(AffineTerm(b, lumped_rhs))
            rows = np.zeros((n_ports, len(lumped)))
            rows[lumped, range(len(lumped))] = 1.0 / np.sqrt(2 * resistance)
            voltage = np.stack([v[1] for v in vectors], axis=1)
            outputs.append(AffineTerm(voltage, lambda _, rows=rows: rows))
        for c in couplings:
            m2 = c.vectors.shape[1]
            if c.index in columns:
                pos = columns.index(c.index)

                def wave_rhs(w: float, c=c, pos=pos, m2=m2) -> np.ndarray:
                    out = np.zeros((m2, width), complex)
                    out[:, pos] = c.excitation(w)
                    return out

                excitation.append(AffineTerm(c.vectors, wave_rhs))

            def wave_row(w: float, c=c, m2=m2) -> np.ndarray:
                out = np.zeros((n_ports, m2), complex)
                out[c.index] = c.functional(w)
                return out

            outputs.append(AffineTerm(np.conj(c.vectors), wave_row))
        z_ref = np.zeros((len(self.problem.frequency), n_ports))
        for i in lumped:
            z_ref[:, i] = _resistance(model.ports[i])
        for c in couplings:
            z_ref[:, c.index] = c.z_ref
        system = AffineSystem(
            thetas=[piece.theta for piece in op.pieces],
            apply=op.apply,
            excitation=excitation,
            outputs=outputs,
            solve=_no_solve,  # the caller solves
        )
        return _Ports(op, system, couplings, columns, z_ref)

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

    def _forward_adaptive(self, model: Model) -> dict[str, object]:
        """The adaptive sweep of `forward` (see `gfem._sweep.adaptive`).

        Outputs as `forward`'s, except that ``"iterations"`` is per sample,
        kept fields come as ``"e_basis"`` ``(ndof, n)`` real and
        ``"e_coordinates"`` ``(n_kept * n_excited, n)`` (frequency-major),
        and ``"sweep"`` holds ``{"samples": grid indices in solve order,
        "errors", "converged", "basis_size"}``.
        """
        frequencies = self.problem.frequency
        options = self.options
        iterations: list[tuple[int, ...]] = []
        # Palace's dimensionless frequency: omega times the largest extent / c
        extent = np.ptp(np.asarray(model.mesh.nodes), axis=0).max()
        z_scale = float(extent) / scipy.constants.c
        with task_manager(options.threads):
            ports = self._ports(model)
            op = ports.operator
            free = op.space.FreeDofs()

            def solve(
                omega: float, rhs: np.ndarray, x0: np.ndarray | None
            ) -> np.ndarray:
                handle = prepare(
                    Operator(op.form(omega), free),
                    symmetry="complex_symmetric",
                    kind=self.kind,
                    device=options.device,
                    tolerance=options.tolerance,
                    width=len(self.excite),
                )
                try:
                    x = handle.solve(rhs, x0=x0)
                    its = getattr(handle, "iterations", [])
                    iterations.append(tuple(int(i) for i in its))
                finally:
                    handle.free()
                return x

            system = dataclasses.replace(ports.system, solve=solve)
            samples = self.problem.samples
            sweep = adaptive_sweep(
                system,
                2 * np.pi * np.asarray(frequencies),
                tolerance=options.adaptive_tolerance,
                max_samples=options.adaptive_max_samples,
                memory=options.adaptive_memory,
                samples=None
                if samples is None
                else [frequencies.index(f) for f in samples],
                z_scale=z_scale,
            )
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
        if self.kept:
            coordinates = sweep.coordinates[list(self.kept)]  # (n_kept, n, n_exc)
            outputs["e_basis"] = sweep.basis
            outputs["e_coordinates"] = np.swapaxes(coordinates, 1, 2).reshape(
                -1, sweep.model.size
            )
        return outputs

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output."""
        if self.adaptive:
            raise NotImplementedError(
                "Driven(sweep='adaptive') cannot run under jax.jit, jax.grad or "
                "jax.vmap in this version of gfem (no gradients through adaptive "
                "sweeps yet): its sample set and basis size "
                "are only known after the solve. Use sweep='uniform', or call "
                "gfem.solve outside the transformation."
            )
        n_f, n_p = len(self.problem.frequency), len(model.ports)
        specs = {
            "s": jax.ShapeDtypeStruct((n_f, n_p, len(self.excite)), jnp.complex128),
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
            width = len(self.kept) * len(self.excite)
            specs["e_field"] = jax.ShapeDtypeStruct((space.ndof, width), jnp.complex128)
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

    def unsupported_input(self, kind: tuple) -> str | None:
        """Why a read input cannot be differentiated here, or ``None``."""
        del kind
        if self.wave_ports:
            return (
                "gradients through wave ports are not available in this version "
                "of gfem (wave port "
                + ", ".join(repr(n) for n in self.wave_ports)
                + "); use lumped ports to differentiate"
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

        At frequency n, with ``X`` the excited ports' solutions and ``W =
        ct_s[n] / sqrt(2 R)`` (rows: ports), the pair is ``(-Lambda, X)``
        with ``Lambda_j = sum_i W_ij A^-1 l_i + A^-1 g_j``, ``g_j`` the
        cotangent of a kept field of port j: ``sum ct dy/dp = -sum_j
        Lambda_j^T (dA/dp) x_j``. ``A^-1 l_i`` of an excited port is its
        own solution scaled (no solve); the remaining loads (kept-field
        cotangents, rows of ports that were not excited) take one
        multi-column adjoint solve. ``z_ref`` contributes nothing.
        """
        solution = entry.fields["solution"]
        resistance = np.array([_resistance(p) for p in model.ports])
        names = model.port_names
        columns = [names.index(n) for n in self.excite]
        width = len(columns)
        ct_s = cotangents.get("s")
        ct_e = cotangents.get("e_field")
        if entry.handle is None:
            entry.handle = _Handles()
        pairs = []
        with task_manager(self.options.threads):
            op = self.operator(model)
            domain = em_region(model, op.built)
            voltage = None
            for n, f in enumerate(self.problem.frequency):
                omega = 2 * math.pi * f
                k0 = op.wavenumber(omega)
                x = solution[:, n * width : (n + 1) * width]
                adjoint = np.zeros_like(x)
                load = np.zeros_like(x)
                if ct_s is not None:
                    w = _row_weights(np.asarray(ct_s[n]), resistance)
                    for m, i in enumerate(columns):  # reciprocity: no solve
                        per_voltage = (
                            1j * k0 * terms.excitation_per_voltage(resistance[i])
                        )
                        adjoint += np.outer(x[:, m] / per_voltage, w[i])
                    undefined = ~np.isfinite(w)  # S rows of ports without R
                    adjoint[:, undefined.any(axis=0)] = np.nan
                    w[undefined] = 0
                    rows = [
                        i for i in range(len(names)) if i not in columns and w[i].any()
                    ]
                    if rows:
                        if voltage is None:
                            voltage = self._voltages(op)
                        load += voltage[:, rows] @ w[rows]
                if ct_e is not None and n in self.kept:
                    k = self.kept.index(n)
                    load += np.asarray(ct_e)[:, k * width : (k + 1) * width]
                if load.any():
                    handle = entry.handle.get(
                        n, lambda omega=omega: self._prepare(op, omega)
                    )
                    adjoint += handle.solve(
                        load, tolerance=adjoint_tolerance(self.options)
                    )
                pairs.append(
                    FieldPair(
                        space=op.space,
                        left=-adjoint,
                        right=x,
                        weights=_weights(k0),
                        order=op.order,
                        domain=domain,
                    )
                )
        return pairs

    def _prepare(self, op: DrivenOperator, omega: float) -> Handle:
        """The operator at ``omega``, prepared at the adjoint tolerance."""
        return prepare(
            Operator(op.form(omega), op.space.FreeDofs()),
            symmetry="complex_symmetric",
            kind=self.kind,
            device=self.options.device,
            tolerance=adjoint_tolerance(self.options),
            width=len(self.excite),
        )

    @staticmethod
    def _voltages(op: DrivenOperator) -> np.ndarray:
        """Every port's voltage vector, ``(ndof, n_ports)``."""
        return np.stack(
            [
                terms.port_vectors(op.space, op.built, op.order, data)[1]
                for data in op.ports
            ],
            axis=1,
        )

    def result(self, outputs: dict) -> DrivenResult:
        """The public result from the forward outputs."""
        frequencies = self.problem.frequency
        kept = [(frequencies[n], port) for n in self.kept for port in self.excite]
        if self.adaptive:
            return self._adaptive_result(outputs, kept)
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
            port_modes=self._port_modes(outputs),
            offsets=self.offsets,
        )

    def _port_modes(self, outputs: dict) -> dict[str, PortModeResult]:
        """The wave ports' `PortModeResult` from the forward outputs."""
        if not self.wave_ports:
            return {}
        sweeps = outputs.get("port_sweeps")
        frequency = jnp.asarray(self.problem.frequency)
        out = {}
        for k, name in enumerate(self.wave_ports):
            field = None
            if self.kept and sweeps is not None:
                sweep = sweeps[k]
                field = PortModeField(sweep.space, sweep.basis, sweep.coords)
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
        info = outputs["sweep"]
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


def _no_solve(*_: object) -> np.ndarray:
    """The placeholder full solve of `_Ports.system`."""
    raise NotImplementedError  # pragma: no cover - replaced before use


def _weights(k0: float) -> dict[str, complex]:
    """θ(ω) of each material-dependent piece (see `gfem._engine.terms`)."""
    return {
        "curl_curl": 1.0,
        "mass": -(k0**2),
        "conductivity": 1j * k0,
        "absorbing": 1j * k0,
        "absorbing order 2": 1 / (2j * k0),
    }


def _row_weights(ct: np.ndarray, resistance: np.ndarray) -> np.ndarray:
    """``ct / sqrt(2 R)`` per S row; rows without a cotangent are exact zeros.

    A row of a port without a resistance is NaN (so is its S row); its
    weights stay NaN only where its cotangent is nonzero.
    """
    with np.errstate(invalid="ignore"):
        w = ct / np.sqrt(2 * resistance)[:, None]
    return np.where(ct == 0, 0, w)


@dataclasses.dataclass(eq=False)
class _Ports:
    """Every port's terms in one driven solve.

    Attributes:
        operator: The driven operator, wave-port Robin pieces included.
        system: Excitations (one column per excited port) and S functionals
            (one row per port) as affine terms; ``solve`` is a placeholder.
        waves: The wave ports' couplings.
        columns: The excited ports' indices.
        z_ref: ``(n_freq, n_ports)`` reference impedances, ohm.
    """

    operator: DrivenOperator
    system: AffineSystem
    waves: list[WavePortCoupling]
    columns: list[int]
    z_ref: np.ndarray

    def scattering(self, waves: np.ndarray) -> np.ndarray:
        """S from the outgoing waves ``(n_freq, n_ports, n_excited)``.

        Subtracts the incident wave at each excited port, then de-embeds
        each wave port by ``exp(gamma offset)`` on its row and column.
        """
        s = np.array(waves, dtype=complex)
        s[:, self.columns, range(len(self.columns))] -= 1.0
        if any(c.port.offset for c in self.waves):
            for n, omega in enumerate(self.waves[0].omegas):
                d = np.ones(s.shape[1], complex)
                for c in self.waves:
                    d[c.index] = c.deembedding(omega)
                s[n] *= d[:, None] * d[self.columns][None, :]
        return s


class _Handles:
    """Prepared driven operators by frequency index, for adjoint solves."""

    def __init__(self) -> None:
        self._handles: dict[int, Handle] = {}

    def get(self, n: int, build: Callable[[], Handle]) -> Handle:
        """The handle of frequency ``n``, prepared by ``build()`` on first use."""
        if n not in self._handles:
            self._handles[n] = build()
        return self._handles[n]

    @property
    def nbytes(self) -> int:
        """Host memory held, in bytes."""
        return sum(int(h.nbytes) for h in self._handles.values())

    def free(self) -> None:
        """Release every handle."""
        for handle in self._handles.values():
            handle.free()
        self._handles = {}


def _resistance(port: LumpedPort | WavePort) -> float:
    """A lumped port's resistance in ohm, NaN without one."""
    if not isinstance(port, LumpedPort):  # pragma: no cover - refused earlier
        return math.nan
    r = port.rlc[0]
    return math.nan if r is None else r


def _check_supported(model: Model) -> None:
    """Raise for ports this version cannot solve."""
    for port in model.ports:
        if not isinstance(port, (LumpedPort, WavePort)):  # pragma: no cover
            raise TypeError(f"Unknown port {port!r}.")
