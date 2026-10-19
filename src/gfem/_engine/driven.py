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
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np

from gfem._boundaries import PEC, Absorbing, Impedance, LumpedPort, WavePort
from gfem._engine import terms
from gfem._engine.cache import Entry, discretization
from gfem._engine.domain import domain_key, hcurl_space
from gfem._engine.fields import fe_space
from gfem._engine.mesh import EngineMesh
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import FieldPair
from gfem._linalg import KINDS, Operator, prepare
from gfem.mesh import Mesh
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Driven
from gfem.results import DrivenResult, Field, NetworkParameters

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
    #: Outputs whose cotangents the backward pass handles: none yet.
    differentiable_outputs = frozenset()

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
        if problem.sweep != "uniform":
            raise NotImplementedError(
                "Driven(sweep='adaptive') is not available in this version of "
                "gfem; use sweep='uniform'."
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
            if math.isnan(_resistance(port)):
                raise ValueError(
                    f"LumpedPort {name!r} has no resistance, so it has no reference "
                    "impedance to feed 1 W through. Give it a resistance, or leave "
                    "it out of Driven.excite."
                )
        #: Indices of the frequencies whose fields are kept.
        self.kept = tuple(problem.frequency.index(f) for f in problem.kept_frequencies)
        mesh, active = model.mesh, model._domain.active  # noqa: SLF001
        #: Volume regions with tetrahedra in the solve domain: the field support.
        self.regions = tuple(r for r in mesh.volumes if active[mesh.elements(r)].any())
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
            keep: Ignored: driven results are not differentiable yet, so
                nothing is kept for a backward pass.

        Returns:
            ``{"s": (n_freq, n_ports, n_excited) complex, "z_ref":
            (n_freq, n_ports) ohm, "iterations": Krylov iterations per
            frequency and excited port (empty for direct)}``, plus
            ``"e_field"``, ``(ndof, n_kept * n_excited)`` complex, when
            fields are kept (frequency-major), and ``None``.
        """
        del keep
        names = model.port_names
        resistance = np.array([_resistance(p) for p in model.ports])
        columns = [names.index(n) for n in self.excite]
        frequencies = self.problem.frequency
        s = np.zeros((len(frequencies), len(names), len(columns)), complex)
        iterations: list[tuple[int, ...]] = []
        fields: list[np.ndarray] = []
        with task_manager(self.options.threads):
            op = self.operator(model)
            vectors = [
                terms.port_vectors(op.space, op.built, op.order, data)
                for data in op.ports
            ]
            excite = np.stack([vectors[j][0] for j in columns], axis=1)
            voltage = np.stack([b_l[1] for b_l in vectors], axis=1)
            free = op.space.FreeDofs()
            previous = None
            for n, f in enumerate(frequencies):
                omega = 2 * math.pi * f
                handle = prepare(
                    Operator(op.form(omega), free),
                    symmetry="complex_symmetric",
                    kind=self.kind,
                    device=self.options.device,
                    tolerance=self.options.tolerance,
                    width=len(columns),
                )
                try:
                    rhs = 1j * op.wavenumber(omega) * excite
                    x = handle.solve(rhs, x0=previous)
                    its = getattr(handle, "iterations", [])
                    iterations.append(tuple(int(i) for i in its))
                finally:
                    handle.free()
                previous = x
                if n in self.kept:
                    fields.append(np.array(x, dtype=complex))
                waves = (voltage.T @ x) / np.sqrt(2 * resistance)[:, None]
                waves[columns, range(len(columns))] -= 1.0
                s[n] = waves
        z_ref = np.broadcast_to(resistance, (len(frequencies), len(names))).copy()
        outputs = {"s": s, "z_ref": z_ref, "iterations": iterations}
        if self.kept:
            outputs["e_field"] = np.concatenate(fields, axis=1)
        return outputs, None

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output."""
        n_f, n_p = len(self.problem.frequency), len(model.ports)
        specs = {
            "s": jax.ShapeDtypeStruct((n_f, n_p, len(self.excite)), jnp.complex128),
            "z_ref": jax.ShapeDtypeStruct((n_f, n_p), jnp.float64),
        }
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
        """Not available: driven gradients raise at trace time."""
        del model, entry, cotangents
        raise NotImplementedError("Driven gradients are not available yet.")

    def result(self, outputs: dict) -> DrivenResult:
        """The public result from the forward outputs."""
        frequencies = self.problem.frequency
        kept = [(frequencies[n], port) for n in self.kept for port in self.excite]
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
        )


def _resistance(port: LumpedPort | WavePort) -> float:
    """A lumped port's resistance in ohm, NaN without one."""
    if not isinstance(port, LumpedPort):  # pragma: no cover - refused earlier
        return math.nan
    r = port.rlc[0]
    return math.nan if r is None else r


def _check_supported(model: Model) -> None:
    """Raise for boundaries and ports this version cannot solve."""
    for port in model.ports:
        if isinstance(port, WavePort):
            raise NotImplementedError(
                f"WavePort {port.name!r}: wave ports in driven problems are not "
                "available in this version of gfem; use a gfem.LumpedPort."
            )
        if not isinstance(port, LumpedPort):  # pragma: no cover - Model checks
            raise TypeError(f"Unknown port {port!r}.")
    for b in model.boundaries:
        if isinstance(b, Impedance):
            raise NotImplementedError(
                f"Impedance on {b.surface!r}: impedance boundaries in driven "
                "problems are not available in this version of gfem."
            )
        if isinstance(b, Absorbing) and b.order != 1:
            raise NotImplementedError(
                f"Absorbing(order=2) on {b.surface!r} is not available in driven "
                "problems in this version of gfem; use Absorbing(order=1)."
            )
