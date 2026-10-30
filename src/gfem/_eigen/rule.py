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
(`gfem._eigen.refine`). A `WavePort` raises: its propagation constant would
have to be solved at the complex eigenfrequency. Everything that touches
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

Post-processing follows Palace: electric energy ``E = 1/2 Re ∫ D* . E`` per
volume region plus ``C |V|^2 / 2`` per capacitive lumped port; a port's
voltage is its driven voltage functional. Each mode is scaled to ``E =
1 J`` with ``∫ D . E`` real and positive.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem import _petsc
from gfem._boundaries import LumpedPort, WavePort
from gfem._eigen import slepc
from gfem._eigen.arpack import shift_invert
from gfem._eigen.operator import EigenOperator
from gfem._eigen.refine import Refined, refine
from gfem._eigen.seed import MIN_SUBSPACE
from gfem._engine import terms
from gfem._engine.cache import Entry
from gfem._engine.domain import em_region
from gfem._engine.driven import (
    DrivenOperator,
    driven_operator,
    operator_shape_terms,
)
from gfem._engine.eigen import (
    DivFreeProjector,
    eigen_operator,
    element_energy,
    projector,
)
from gfem._engine.fields import fe_space
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import FieldPair, ShapeTerm
from gfem._linalg import KINDS
from gfem._sign import largest_entry
from gfem.conventions import MODE_ENERGY
from gfem.errors import DegenerateModeError, EigenmodeConvergenceError
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
#: ... but never below this: the true preconditioned residual of the
#: shift-invert operators stays near 1e-13, a double-precision limit of the
#: operator, and BDDC + GMRES ends within 0.1 % of 1e-14, too thin a margin
#: to ask for. It also bounds PETSc Krylov inner solves, whose ``ksp_rtol``
#: floor is no lower.
_INNER_FLOOR = 1e-13
#: Seeds with Re k at or below this fraction of the shift are the curl-curl
#: null space, a lossy medium's relaxation modes (Re k = 0), mirror roots or
#: a conductor's linearization cluster (Re k near -sigma), not resonances. A
#: second-order absorbing wall's cluster lies near Re k = 2 sigma: this test
#: keeps it, and only the distance to the target ranks the modes ahead of it.
_STATIC = 1e-6
#: Two refined modes closer than this many eigen tolerances (relative, and
#: at least ``SolverOptions.cluster_rtol``), with overlapping fields, are the
#: same mode.
_DUPLICATE = 10.0
_OVERLAP = 0.9
#: Ritz values this close (relative) are one multiple eigenvalue. The seed
#: solver returns an arbitrary basis of its eigenspace, as nearly parallel as
#: round-off makes it: the seeds are orthonormalized, or two members would
#: overlap like one mode refined twice.
_MULTIPLE = 1e-8
#: A seed that needs correction steps is refined on a new factorization at
#: its own Ritz value when the current shift would contract its error by
#: less than this per step (the shift is nearer another Ritz value: residual
#: inverse iteration would drift to that mode).
_CONTRACTION = 0.5
#: Arnoldi runs, each asking for twice as many pairs as the last, before
#: too few modes near the target is an error.
_ATTEMPTS = 3
#: Arnoldi tolerance of a nonlinear problem's frozen seeds. Their roots are
#: off the true ones by the freeze error anyway and every seed is refined,
#: so converging them further only spends shift solves. The linearization
#: cluster's solves are avoided by asking for few pairs (see the nev comment
#: in the forward pass, ADR 0035), not by this tolerance.
_SEED_TOLERANCE = 1e-3
#: Cotangents on the members of a cluster that agree to this (relative to
#: the largest) make an objective symmetric in the cluster to first order
#: (a smooth symmetric function's cotangents differ by O(gap)).
_SYMMETRIC_RTOL = 1e-3


class EigenmodeRule:
    """Host-side forward of an `Eigenmode` problem on a `Model`."""

    physics = "em"
    #: The operator family in the property-to-form-term table.
    form = "eigenmode"
    #: Material properties the eigenmode forms read.
    reads = frozenset({"eps_r", "mu_r", "tan_delta", "conductivity"})
    #: Outputs whose cotangents the backward pass handles: the eigenvalues
    #: (``cluster`` holds integer labels: its cotangent is always zero).
    differentiable_outputs = frozenset({"omega", "cluster"})

    def __init__(
        self, model: Model, problem: Eigenmode, options: SolverOptions
    ) -> None:
        """Check that the problem fits the model (see `gfem.Eigenmode`).

        Raises:
            NotImplementedError: A feature this version cannot solve.
            gfem.SolverUnavailableError: ``linear_solver="petsc"`` or
                ``eigen_solver="slepc"`` names a library that cannot be used.
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
        # before any assembly
        if self.kind == "petsc":
            _petsc.require_petsc("linear_solver")
        if options.eigen_solver == "slepc":
            _petsc.require_slepc("eigen_solver")
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
            keep: Keep what the backward pass needs (the modes, their
                ``x^T T' x`` per cluster and whether a cluster reaches the
                guard mode, refined beyond ``n_modes`` for this).

        Returns:
            ``{"omega": (n,) complex rad/s, "error": (n,), "q_port":
            (n_resistive, n), "participation": (n_inductive, n), "bulk":
            (n_regions, n), "e_field": (ndof, n) complex, "cluster": (n,)
            int}``, modes nearest the target first, and the kept `Entry`
            (``None`` unless ``keep``).

        Raises:
            EigenmodeConvergenceError: Fewer than ``n_modes`` modes
                converged.
        """
        opts = self.options
        order = opts.order
        n_modes = self.problem.n_modes
        with task_manager(opts.threads):
            op = driven_operator(model, order)
            built = op.built
            sigma = terms.wavenumber(2 * math.pi * self.problem.target, built.scale)
            eop = self.operator(op, sigma)
            divfree = None
            try:
                eop.factorize()  # before the projector's setup
                divfree = projector(model, op)
                seed_tolerance = opts.eigen_tolerance
                # the companion problem also holds each mode's mirror root
                # -conj(k), |k + sigma| from the shift (behind its mode, about
                # 2 sigma away for a mode near the target), and pairs near
                # k = 0 about sigma away; they take slots in front of wanted
                # modes farther than that, so a damped linear seed asks for
                # twice as many pairs. A nonpolynomial seed asks for one pair
                # beyond n_modes, the guard's slot: behind its modes lies the
                # linearization cluster, which costs shift solves to converge
                # and holds no mode (ADR 0035).
                if eop.nonpolynomial:
                    nev = n_modes + 1
                    seed_tolerance = max(seed_tolerance, _SEED_TOLERANCE)
                else:
                    nev = max(2 * n_modes, n_modes + 2)
                    nev *= 2 if eop.seed_damped else 1
                guard = None
                for _ in range(_ATTEMPTS):
                    if eop.shift != sigma:
                        # a retry is a seed solve: at the target, on a
                        # factorization counted for the seed (refinement
                        # reshifted it)
                        eop.reshift(sigma, solves=MIN_SUBSPACE)
                    k, vectors = self._shift_invert(
                        eop, divfree, nev, tol=seed_tolerance
                    )
                    modes, failed, guard = self._select(
                        eop, k, vectors, sigma, guard=keep
                    )
                    if len(modes) == n_modes or failed:
                        break
                    nev *= 2  # too few candidates near the target: look wider
                if len(modes) < n_modes:
                    raise EigenmodeConvergenceError(
                        self._convergence_message(eop, modes, failed)
                    )
                if not eop.seed_damped and not eop.nonpolynomial and _is_real(eop):
                    modes = [
                        Refined(m.k.real, m.x, m.error, m.iterations, m.converged)
                        for m in modes
                    ]
                    if guard is not None:
                        guard = dataclasses.replace(guard, k=guard.k.real)
                outputs = self._postprocess(model, op, divfree, modes)
                ks = [m.k for m in modes]
                outputs["cluster"] = clusters(ks, opts.cluster_rtol)
                entry = None
                if keep:
                    entry = self._entry(eop, ks, outputs, guard)
            finally:
                eop.free_handle()
                if divfree is not None:
                    divfree.free()
        return outputs, entry

    def _shift_invert(
        self, op: EigenOperator, project, nev: int, *, tol: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """The seed solve on ``SolverOptions.eigen_solver`` (see `shift_invert`)."""
        if self.options.eigen_solver == "slepc":
            options = self.options.petsc_options
            return slepc.shift_invert(op, project, nev, tol=tol, options=options)
        # the module's name, looked up now: tests replace it
        return shift_invert(op, project, nev, tol=tol)

    def operator(self, op: DrivenOperator, sigma: float) -> EigenOperator:
        """The eigen operator ``T(k)`` of the driven operator, shifted at ``sigma``.

        Nothing is factorized until the handle is first used
        (`gfem._engine.eigen.eigen_operator`).
        """
        opts = self.options
        return eigen_operator(
            op,
            sigma,
            kind=self.kind,
            device=opts.device,
            threads=opts.threads,
            tolerance=min(
                opts.tolerance, max(_INNER * opts.eigen_tolerance, _INNER_FLOOR)
            ),
            petsc_options=opts.petsc_options,
        )

    def _entry(
        self,
        op: EigenOperator,
        ks: list[complex],
        outputs: dict[str, object],
        guard: Refined | None,
    ) -> Entry:
        """What the backward pass needs: no handle, the modes and their T'.

        ``gram`` is block diagonal by cluster: ``x_i^T T'(k_c) x_j`` for
        modes i, j of cluster c, with ``k_c`` the cluster's mean.
        ``incomplete`` flags the modes of a cluster the guard mode belongs
        to.
        """
        x = np.asarray(outputs["e_field"])
        labels = np.asarray(outputs["cluster"])
        k = np.array(ks, dtype=complex)
        gram = np.zeros((len(k), len(k)), dtype=complex)
        incomplete = np.zeros(len(k), dtype=bool)
        rtol = self.options.cluster_rtol
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            mean = k[members].mean()
            tx = np.stack([op.derivative(mean, x[:, j]) for j in members], axis=1)
            gram[np.ix_(members, members)] = x[:, members].T @ tx
            if guard is not None and any(_close(guard.k, k[j], rtol) for j in members):
                incomplete[members] = True
        return Entry(
            self.physics,
            fields={
                "x": x,
                "k": k,
                "cluster": labels,
                "gram": gram,
                "incomplete": incomplete,
            },
        )

    def _select(
        self,
        op: EigenOperator,
        k: np.ndarray,
        vectors: np.ndarray,
        sigma: float,
        *,
        guard: bool = False,
    ) -> tuple[list[Refined], list[Refined], Refined | None]:
        """Refine the seeds nearest ``sigma`` until ``n_modes`` distinct ones.

        Seeds (and refined pairs) with ``Re k`` near zero or negative are the
        curl-curl null space, a lossy medium's relaxation modes, mirror
        roots or a conductor's linearization cluster, not resonances: they
        are dropped.
        A second-order absorbing wall's cluster has ``Re k`` near ``2
        sigma``, so it is not dropped; the modes nearer the target come
        first. A seed far from the current shift whose refinement gives
        nothing to accept (see `_lost`) is refined again on a factorization
        at its own Ritz value. With ``guard``, the remaining
        seeds are refined (without
        refactorizing) until one more distinct mode converges: the guard
        mode. It is best-effort: none is found when no remaining seed
        converges to a mode.

        Returns:
            The accepted modes, nearest ``sigma`` first (at most
            ``n_modes``), the seeds that did not converge, and the guard
            mode (``None`` without ``guard`` or when no seed gave one).
        """
        n_modes = self.problem.n_modes
        tolerance = self.options.eigen_tolerance
        seeds = [j for j in range(len(k)) if k[j].real > _STATIC * sigma]
        seeds.sort(key=lambda j: abs(k[j].real - sigma))
        vectors = _orthonormal_multiples(k, vectors)
        accepted: list[Refined] = []
        failed: list[Refined] = []
        extra = None
        window = max(_DUPLICATE * tolerance, self.options.cluster_rtol)
        for j in seeds:
            full = len(accepted) == n_modes
            if full and not guard:
                break
            seed = complex(k[j])
            far = _contraction(seed, op.shift, k) > _CONTRACTION
            mode = refine(
                op,
                seed,
                vectors[:, j],
                tolerance=tolerance,
                max_iterations=self.options.max_refine_iterations,
                reshift=not full,
                reshift_first=far,
            )
            if far and not full and _lost(mode, seed, sigma, accepted, window):
                # a seed rougher than TRUSTED, refined on a shift nearer
                # another mode, wanders where round-off takes it: refine it
                # again on a factorization at its own Ritz value
                op.reshift(seed)
                mode = refine(
                    op,
                    seed,
                    vectors[:, j],
                    tolerance=tolerance,
                    max_iterations=self.options.max_refine_iterations,
                )
            if mode.k.real <= _STATIC * sigma:
                continue
            if not mode.converged:
                if not full:
                    failed.append(mode)
                continue
            if any(_same(mode, other, window) for other in accepted):
                continue
            if full:
                extra = mode
                break
            accepted.append(mode)
        accepted.sort(key=lambda m: abs(m.k.real - sigma))
        return accepted, failed, extra

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
        self,
        model: Model,
        op,
        divfree: DivFreeProjector,
        modes: list[Refined],
    ) -> dict[str, object]:
        built, space, order = op.built, op.space, op.order
        scale = built.scale
        ports = {data.port.name: data for data in op.ports}
        voltage = {
            name: terms.port_vectors(space, built, order, data)[1]
            for name, data in ports.items()
        }
        mesh = model.mesh
        density = element_energy(model, op)
        n = len(modes)
        omega = np.array([m.k * const.c / scale for m in modes], dtype=complex)
        fields = np.zeros((space.ndof, n), dtype=complex)
        bulk = np.zeros((len(self.regions), n))
        q_port = np.zeros((len(self.resistive), n))
        participation = np.zeros((len(self.inductive), n))
        for j, mode in enumerate(modes):
            x = mode.x
            s = x @ divfree.energy(x)
            x = x * np.exp(-0.5j * np.angle(s))
            if x[largest_entry(x)].real < 0:
                x = -x
            # electric energy per element (gfem order) at this amplitude
            per_tet = np.zeros(len(mesh.tetrahedra))
            per_tet[built.tetrahedron_order] = (
                0.5 * const.epsilon_0 * scale * np.real(density(x))
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
            "cluster": jax.ShapeDtypeStruct((n,), jnp.int64),
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
            "version: only the eigenfrequencies (omega, so frequency and q) "
            "are differentiable, not quantities of the mode fields (the "
            "fields themselves, participations, port Q, interface "
            "participation, probes). Wrap it in jax.lax.stop_gradient to use "
            "its value without its derivative."
        )

    def backward(
        self, model: Model, entry: Entry, cotangents: dict[str, np.ndarray]
    ) -> list[FieldPair]:
        """Fold the ``omega`` cotangents into one field pair per cluster.

        With ``ω = k c / scale``, cluster c's modes ``X`` and ``G = X^T
        T'(k_c) X``: ``ct_c d(Σ ω)/dp = -(c / scale) ct_c tr(G^-1 X^T (dT/dp)
        X)``, the pair ``(-(c / scale) ct_c X, X G^-1)`` weighted with the
        pieces' θ at ``k_c``. A simple mode is a cluster of one (``G =
        x^T T' x``). No solve.

        Raises:
            DegenerateModeError: The cotangents on a cluster's modes differ
                (an objective not symmetric in them), or a cluster with a
                nonzero cotangent extends beyond the ``n_modes`` modes.
        """
        ct = cotangents.get("omega")
        if ct is None or not np.any(ct):
            return []
        fields = entry.fields
        x, k, labels = fields["x"], fields["k"], fields["cluster"]
        gram, incomplete = fields["gram"], fields["incomplete"]
        ct = np.asarray(ct, dtype=complex)
        pairs = []
        with task_manager(self.options.threads):
            op = self.driven(model)
            domain = em_region(model, op.built)
            per_k = const.c / op.built.scale
            for label in np.unique(labels):
                members = np.flatnonzero(labels == label)
                c = ct[members]
                if not c.any():
                    continue
                spread = np.abs(c - c.mean()).max()
                symmetric = spread <= _SYMMETRIC_RTOL * np.abs(c).max()
                if incomplete[members].any() or not symmetric:
                    raise DegenerateModeError(
                        self._degenerate_message(k, members, c, incomplete, per_k)
                    )
                g = gram[np.ix_(members, members)]
                pairs.append(
                    FieldPair(
                        space=op.space,
                        left=-per_k * c.mean() * x[:, members],
                        right=x[:, members] @ np.linalg.inv(g),
                        weights=op.weights(k[members].mean() * per_k),
                        order=op.order,
                        domain=domain,
                    )
                )
        return pairs

    def driven(self, model: Model) -> DrivenOperator:
        """The model's driven operator (the pieces of ``T(k)``) at its nodes.

        The last model's operator is kept, so that a backward pass and its
        shape terms build it once.
        """
        last = getattr(self, "_last_driven", None)
        if last is not None and last[0] is model:
            return last[1]
        op = driven_operator(model, self.options.order)
        self._last_driven = (model, op)
        return op

    def shape_terms(self, model: Model) -> list[ShapeTerm]:
        """The pieces of ``T(k)`` as node-coordinate (shape) terms.

        The driven operator's (`gfem._engine.driven.operator_shape_terms`):
        a mode's ``dk/dX = -x^T (dT/dX) x / x^T T'(k) x`` contracts the same
        pieces, weighted with their θ at the mode, as the material terms.
        Port pieces are left out: shape gradients hold every port surface.

        Raises:
            PortGeometryError: A node of a port surface has moved.
        """
        return operator_shape_terms(model, self.driven(model))

    def _degenerate_message(
        self,
        k: np.ndarray,
        members: np.ndarray,
        ct: np.ndarray,
        incomplete: np.ndarray,
        per_k: float,
    ) -> str:
        ghz = k[members] * per_k / (2 * math.pi * 1e9)
        gap = np.ptp(np.abs(k[members])) / np.abs(k[members]).max()
        modes = ", ".join(
            f"{int(j)} ({f.real:.9g} GHz)" for j, f in zip(members, ghz, strict=True)
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
            cluster=jnp.asarray(outputs["cluster"]),
        )


def _close(a: complex, b: complex, rtol: float) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def clusters(ks: list[complex], rtol: float) -> np.ndarray:
    """A label per eigenvalue: chains of eigenvalues within ``rtol`` share one.

    Labels count from 0 in order of first appearance.
    """
    labels = list(range(len(ks)))
    for i in range(len(ks)):
        for j in range(i):
            if _close(ks[i], ks[j], rtol) and labels[i] != labels[j]:
                old, new = labels[i], labels[j]
                labels = [new if v == old else v for v in labels]
    renumber: dict[int, int] = {}
    for v in labels:
        renumber.setdefault(v, len(renumber))
    return np.array([renumber[v] for v in labels], dtype=np.int64)


def _contraction(seed: complex, shift: complex, ritz: np.ndarray) -> float:
    """Residual inverse iteration's error ratio per step for ``seed``.

    ``|seed - shift|`` over the distance from ``shift`` to the nearest other
    Ritz value: the component of that mode grows unless the ratio is below 1.
    """
    others = ritz[np.abs(ritz - seed) > _MULTIPLE * abs(seed)]
    if not len(others):
        return 0.0
    nearest = float(np.min(np.abs(others - shift)))
    return abs(seed - shift) / nearest if nearest > 0 else math.inf


def _orthonormal_multiples(k: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """``vectors`` with each multiple Ritz value's vectors orthonormalized.

    Any basis of a multiple eigenvalue's eigenspace is as good a set of
    seeds; an orthonormal one keeps its members apart under `_same`.
    """
    out = vectors
    done = np.zeros(len(k), dtype=bool)
    for j in range(len(k)):
        if done[j]:
            continue
        members = np.flatnonzero(np.abs(k - k[j]) <= _MULTIPLE * abs(k[j]))
        done[members] = True
        if len(members) > 1:
            if out is vectors:
                out = vectors.copy()
            out[:, members] = np.linalg.qr(vectors[:, members])[0]
    return out


def _same(a: Refined, b: Refined, window: float) -> bool:
    if abs(a.k - b.k) > window * abs(b.k):
        return False
    overlap = abs(np.vdot(a.x, b.x)) / (np.linalg.norm(a.x) * np.linalg.norm(b.x))
    return overlap > _OVERLAP


def _lost(
    mode: Refined,
    seed: complex,
    sigma: float,
    accepted: list[Refined],
    window: float,
) -> bool:
    """Whether ``seed``'s refinement gave nothing to accept.

    It did not converge, converged to a static or mirror root, or converged
    to an accepted mode the seed is not near.
    """
    if not mode.converged or mode.k.real <= _STATIC * sigma:
        return True
    return any(
        _same(mode, other, window) and not _close(seed, other.k, window)
        for other in accepted
    )


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
