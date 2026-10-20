"""Eigenmode rule: resonant modes of a model in complex Nédélec elements.

The operator is the driven operator's pieces (`gfem._engine.driven`) sorted
by power of the frequency (`gfem._eigen.operator`): ``T(k) = K + k D +
k^2 M``, ``k = ω scale / c``. Without damping the problem is linear in
``k^2``; with lumped or impedance resistance, conductivity or a first-order
absorbing boundary it is quadratic. One factorization of ``T(sigma)`` at
the target serves the shift-invert Arnoldi (`gfem._eigen.arpack`, with the
divergence-free projection), the Rayleigh functional and the refinement of
any pair short of the tolerance (`gfem._eigen.refine`).

Post-processing follows Palace: electric energy ``E = 1/2 Re ∫ D* . E`` per
volume region plus ``C |V|^2 / 2`` per capacitive lumped port; a port's
voltage is its driven voltage functional. Each mode is scaled to ``E =
1 J`` with ``∫ D . E`` real and positive.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants as const

from gfem._boundaries import PEC, Absorbing, Conductor, LumpedPort, WavePort
from gfem._eigen.arpack import shift_invert
from gfem._eigen.divfree import DivFreeProjector
from gfem._eigen.operator import EigenOperator, assemble, split
from gfem._eigen.refine import Refined, refine
from gfem._engine import terms
from gfem._engine.cache import Entry, discretization
from gfem._engine.coefficients import coefficient, element_tensors
from gfem._engine.domain import domain_key, h1_space, hcurl_space
from gfem._engine.driven import driven_operator
from gfem._engine.fields import fe_space
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import FieldPair
from gfem._linalg import KINDS, Operator, prepare
from gfem.conventions import MODE_ENERGY
from gfem.errors import EigenmodeConvergenceError
from gfem.mesh import Mesh
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Eigenmode
from gfem.results import EigenmodeResult, Field

__all__ = ["DEFAULT_SOLVER", "EigenmodeRule"]

#: The linear solver for ``linear_solver="auto"``: shift-invert wants exact
#: inner solves.
DEFAULT_SOLVER = "direct"
#: Inner solves of an iterative backend run this much tighter than the
#: eigen tolerance.
_INNER = 1e-2
#: Seeds with Re k at or below this fraction of the shift are the curl-curl
#: null space or relaxation modes, not resonances.
_STATIC = 1e-6
#: Two refined modes closer than this many eigen tolerances (relative), with
#: overlapping fields, are the same mode.
_DUPLICATE = 10.0
_OVERLAP = 0.9
#: Arnoldi runs, each asking for twice as many pairs as the last, before
#: too few modes near the target is an error.
_ATTEMPTS = 3


class EigenmodeRule:
    """Host-side forward of an `Eigenmode` problem on a `Model`."""

    physics = "em"
    #: The operator family in the property-to-form-term table.
    form = "eigenmode"
    #: Material properties the eigenmode forms read.
    reads = frozenset({"eps_r", "mu_r", "tan_delta", "conductivity"})
    #: Outputs whose cotangents the backward pass handles: none yet.
    differentiable_outputs = frozenset()

    def __init__(
        self, model: Model, problem: Eigenmode, options: SolverOptions
    ) -> None:
        """Check that the problem fits the model (see `gfem.Eigenmode`).

        Raises:
            NotImplementedError: A feature this version cannot solve.
        """
        if not isinstance(problem, Eigenmode):
            raise TypeError(f"Expected gfem.Eigenmode, got {problem!r}.")
        self.problem = problem
        self.options = options
        kind = options.linear_solver
        self.kind = DEFAULT_SOLVER if kind == "auto" else kind
        if self.kind not in KINDS:  # pragma: no cover - SolverOptions validates
            raise NotImplementedError(
                f"linear_solver={kind!r} is not available for eigenmode problems."
            )
        _check_supported(model)
        lumped = [p for p in model.ports if isinstance(p, LumpedPort)]
        #: Ports with a resistance (coupling Q) and with an inductance
        #: (participation), in model order.
        self.resistive = tuple(p.name for p in lumped if p.rlc[0] is not None)
        self.inductive = tuple(p.name for p in lumped if p.rlc[1] is not None)
        mesh, active = model.mesh, model._domain.active  # noqa: SLF001
        #: Volume regions with tetrahedra in the solve domain: the field support.
        self.regions = tuple(r for r in mesh.volumes if active[mesh.elements(r)].any())
        self._mesh = model.mesh
        self._materials = dict(model.materials)
        self._excluded = model._domain.excluded  # noqa: SLF001

    # --- forward -----------------------------------------------------------------

    def forward(
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, object], Entry | None]:
        """Find the modes.

        Args:
            model: The model, with concrete leaves.
            keep: Ignored: eigenmode results are not differentiable yet.

        Returns:
            ``{"omega": (n,) complex rad/s, "error": (n,), "q_port":
            (n_resistive, n), "participation": (n_inductive, n), "bulk":
            (n_regions, n), "e_field": (ndof, n) complex}``, modes nearest
            the target first, and ``None``.

        Raises:
            EigenmodeConvergenceError: Fewer than ``n_modes`` modes
                converged.
        """
        del keep
        opts = self.options
        order = opts.order
        n_modes = self.problem.n_modes
        with task_manager(opts.threads):
            op = driven_operator(model, order)
            built, space = op.built, op.space
            scale = built.scale
            omega_t = 2 * math.pi * self.problem.target
            sigma = terms.wavenumber(omega_t, scale)
            classes = split(op.pieces, scale, sigma)
            free = np.array(list(space.FreeDofs()), dtype=bool)  # ty: ignore[invalid-argument-type]

            def shifted(k: complex):
                return prepare(
                    Operator(op.form(k * const.c / scale), space.FreeDofs()),
                    symmetry="complex_symmetric",
                    kind=self.kind,
                    device=opts.device,
                    tolerance=min(opts.tolerance, _INNER * opts.eigen_tolerance),
                    width=1,
                )

            eop = EigenOperator(
                shift=sigma,
                prepare=shifted,
                damping=(
                    assemble(space, classes[1], matrix_free=True)
                    if classes[1]
                    else None
                ),
                mass=assemble(space, classes[2]),
                free=free,
                scale=scale,
            )
            projector = None
            try:
                eop.handle  # noqa: B018 - factorize before the projector's setup
                projector = self._projector(model, built, order)
                # the companion problem also holds each mode's mirror image
                # -conj(k), as near the shift as the mode's Re k is
                nev = max(2 * n_modes, n_modes + 2)
                nev *= 1 if eop.damping is None else 2
                for _ in range(_ATTEMPTS):
                    k, vectors = shift_invert(
                        eop, projector, nev, tol=opts.eigen_tolerance
                    )
                    modes, failed = self._select(eop, k, vectors, sigma)
                    if len(modes) == n_modes or failed:
                        break
                    nev *= 2  # too few candidates near the target: look wider
                if len(modes) < n_modes:
                    raise EigenmodeConvergenceError(
                        self._convergence_message(eop, modes, failed)
                    )
                if eop.damping is None and _is_real(eop):
                    modes = [
                        Refined(m.k.real, m.x, m.error, m.iterations, m.converged)
                        for m in modes
                    ]
                outputs = self._postprocess(model, op, projector, modes)
            finally:
                eop.free_handle()
                if projector is not None:
                    projector.free()
        return outputs, None

    def _projector(self, model: Model, built, order: int) -> DivFreeProjector:
        disc = discretization(model.mesh)
        pec = tuple(sorted({b.surface for b in model.boundaries if isinstance(b, PEC)}))
        key = domain_key(model)
        real = disc.space(
            (key, "hcurl", order, pec, "real"),
            lambda: hcurl_space(
                model, built, order, complex_valued=False, dirichlet=pec
            ),
        )
        h1 = disc.space(
            (key, "h1", order, pec), lambda: h1_space(model, built, order, pec)
        )
        eps = coefficient(built, np.real(element_tensors(model, "eps_r")))
        nodes = np.asarray(model.mesh.nodes) / built.scale
        diameter = float(np.linalg.norm(np.ptp(nodes, axis=0)))
        return DivFreeProjector(real, h1, eps, order, diameter)

    def _select(
        self, op: EigenOperator, k: np.ndarray, vectors: np.ndarray, sigma: float
    ) -> tuple[list[Refined], list[Refined]]:
        """Refine the seeds nearest ``sigma`` until ``n_modes`` distinct ones.

        Seeds (and refined pairs) with ``Re k`` near zero or negative are the
        curl-curl null space, relaxation modes or mirror images, not
        resonances: they are dropped.

        Returns:
            The accepted modes, nearest ``sigma`` first (at most
            ``n_modes``), and the seeds that did not converge.
        """
        n_modes = self.problem.n_modes
        tolerance = self.options.eigen_tolerance
        seeds = [j for j in range(len(k)) if k[j].real > _STATIC * sigma]
        seeds.sort(key=lambda j: abs(k[j].real - sigma))
        accepted: list[Refined] = []
        failed: list[Refined] = []
        for j in seeds:
            mode = refine(
                op,
                complex(k[j]),
                vectors[:, j],
                tolerance=tolerance,
                max_iterations=self.options.max_refine_iterations,
            )
            if mode.k.real <= _STATIC * sigma:
                continue
            if not mode.converged:
                failed.append(mode)
                continue
            if any(_same(mode, other, tolerance) for other in accepted):
                continue
            accepted.append(mode)
            if len(accepted) == n_modes:
                break
        accepted.sort(key=lambda m: abs(m.k.real - sigma))
        return accepted, failed

    def _convergence_message(
        self, op: EigenOperator, accepted: list[Refined], failed: list[Refined]
    ) -> str:
        c = const.c / op.scale / (2 * math.pi)
        found = len(accepted)
        lines = [
            (
                f"Eigenmode found {found} of the {self.problem.n_modes} requested "
                f"mode(s) near {self.problem.target / 1e9:.6g} GHz to backward "
                f"error {self.options.eigen_tolerance:g} "
                "(SolverOptions.eigen_tolerance)."
            ),
            *(
                f"  not converged: {m.k.real * c / 1e9:.6g} GHz, best backward "
                f"error {m.error:.3g} after {m.iterations} refinement steps"
                for m in failed
            ),
        ]
        lines.append(
            "Remedies: move Eigenmode.target closer to the wanted modes; loosen "
            "SolverOptions.eigen_tolerance or raise max_refine_iterations; for a "
            "mode with Q below about 2, find the resonance from a gfem.Driven "
            "sweep instead."
        )
        return "\n".join(lines)

    # --- post-processing ---------------------------------------------------------

    def _postprocess(
        self, model: Model, op, projector: DivFreeProjector, modes: list[Refined]
    ) -> dict[str, object]:
        built, space, order = op.built, op.space, op.order
        scale = built.scale
        ports = {data.port.name: data for data in op.ports}
        voltage = {
            name: terms.port_vectors(space, built, order, data)[1]
            for name, data in ports.items()
        }
        mesh = model.mesh
        eps = element_tensors(model, "eps_r")
        eps_cf = coefficient(built, np.real(eps))
        n = len(modes)
        omega = np.array([m.k * const.c / scale for m in modes], dtype=complex)
        fields = np.zeros((space.ndof, n), dtype=complex)
        bulk = np.zeros((len(self.regions), n))
        q_port = np.zeros((len(self.resistive), n))
        participation = np.zeros((len(self.inductive), n))
        gf = ngs.GridFunction(space)
        for j, mode in enumerate(modes):
            x = mode.x
            s = x @ projector.energy(x)
            x = x * np.exp(-0.5j * np.angle(s))
            if x[np.argmax(np.abs(x))].real < 0:
                x = -x
            # electric energy per element (gfem order) at this amplitude
            gf.vec.FV().NumPy()[:] = x
            density = ngs.Integrate(
                (eps_cf * gf) * ngs.Conj(gf),
                built.mesh,
                order=2 * order,
                element_wise=True,
            )
            per_tet = np.zeros(len(mesh.tetrahedra))
            per_tet[built.tetrahedron_order] = (
                0.5 * const.epsilon_0 * scale * np.real(density.NumPy())
            )
            volts = {name: complex(v @ x) for name, v in voltage.items()}
            capacitors = sum(
                0.5 * data.port.rlc[2] * abs(volts[name]) ** 2
                for name, data in ports.items()
                if data.port.rlc[2] is not None
            )
            energy = per_tet.sum() + capacitors
            amplitude = math.sqrt(MODE_ENERGY / energy)
            fields[:, j] = x * amplitude
            for r, region in enumerate(self.regions):
                bulk[r, j] = per_tet[mesh.elements(region)].sum() / energy
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
        return {
            "omega": omega,
            "error": np.array([m.error for m in modes]),
            "q_port": q_port,
            "participation": participation,
            "bulk": bulk,
            "e_field": fields,
        }

    # --- the solve_vjp interface ---------------------------------------------------

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output."""
        n = self.problem.n_modes
        mesh = model.mesh
        still = Mesh(mesh.imported_nodes, mesh._topology)  # noqa: SLF001
        with task_manager(self.options.threads):
            _, space, _ = fe_space(
                still, self.regions, self._excluded, "hcurl", self.options.order
            )
        return {
            "omega": jax.ShapeDtypeStruct((n,), jnp.complex128),
            "error": jax.ShapeDtypeStruct((n,), jnp.float64),
            "q_port": jax.ShapeDtypeStruct((len(self.resistive), n), jnp.float64),
            "participation": jax.ShapeDtypeStruct(
                (len(self.inductive), n), jnp.float64
            ),
            "bulk": jax.ShapeDtypeStruct((len(self.regions), n), jnp.float64),
            "e_field": jax.ShapeDtypeStruct((space.ndof, n), jnp.complex128),
        }

    def role(self, kind: tuple) -> str:
        """``"read"`` if the eigenmode solve depends on input ``kind``."""
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        return "read"

    def nondifferentiable_output(self, name: str) -> str:
        """The error message for a cotangent on output ``name``."""
        return (
            f"gfem cannot differentiate the eigenmode output {name!r} in this "
            "version. Wrap it in jax.lax.stop_gradient to use its value without "
            "its derivative."
        )

    def backward(
        self, model: Model, entry: Entry, cotangents: dict[str, np.ndarray]
    ) -> list[FieldPair]:
        """Not available: eigenmode gradients raise at trace time."""
        del model, entry, cotangents
        raise NotImplementedError("Eigenmode gradients are not available yet.")

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
            )
            for j in range(self.problem.n_modes)
        ]
        return EigenmodeResult(
            omega=jnp.asarray(omega),
            error=jnp.asarray(outputs["error"]),
            q_port={
                name: jnp.asarray(outputs["q_port"][p])
                for p, name in enumerate(self.resistive)
            },
            participation={
                name: jnp.asarray(outputs["participation"][p])
                for p, name in enumerate(self.inductive)
            },
            bulk_participation={
                region: jnp.asarray(outputs["bulk"][r])
                for r, region in enumerate(self.regions)
            },
            e_fields=tuple(fields),
            materials=self._materials,
        )


def _same(a: Refined, b: Refined, tolerance: float) -> bool:
    if abs(a.k - b.k) > _DUPLICATE * tolerance * abs(b.k):
        return False
    overlap = abs(np.vdot(a.x, b.x)) / (np.linalg.norm(a.x) * np.linalg.norm(b.x))
    return overlap > _OVERLAP


def _is_real(op: EigenOperator) -> bool:
    """Whether ``K`` and ``M`` are real: a lossless, undamped problem."""
    probe = np.where(op.free, np.random.default_rng(1).standard_normal(len(op.free)), 0)
    kx, _, mx = op.parts(probe.astype(complex))
    return not kx.imag.any() and not mx.imag.any()


def _check_supported(model: Model) -> None:
    """Raise for model terms this version cannot solve as eigenmodes."""
    for port in model.ports:
        if isinstance(port, WavePort):
            raise NotImplementedError(
                f"Eigenmode does not support wave port {port.name!r} in this "
                "version of gfem: its propagation constant must be solved at the "
                "complex eigenfrequency. Replace it with gfem.LumpedPort("
                f"{port.name!r}, ..., impedance=<line impedance>), or find the "
                "resonance and Q from a gfem.Driven sweep."
            )
    for b in model.boundaries:
        if isinstance(b, Conductor):
            raise NotImplementedError(
                f"Eigenmode with the gfem.Conductor on {b.surface!r} is not "
                "available in this version of gfem: its surface impedance grows "
                "as the square root of the frequency, which makes the eigenproblem "
                "nonlinear. Use gfem.PEC for the metal, or find the resonance and Q "
                "from a gfem.Driven sweep."
            )
        if isinstance(b, Absorbing) and b.order == 2:
            raise NotImplementedError(
                f"Eigenmode with the second-order gfem.Absorbing on {b.surface!r} "
                "is not available in this version of gfem: its 1/frequency term "
                "makes the eigenproblem nonlinear. Use Absorbing(order=1), or find "
                "the resonance and Q from a gfem.Driven sweep."
            )
