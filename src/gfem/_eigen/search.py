"""The eigen mode search: seeds near the target, refined into found modes.

`ModeSearch.find` runs the seed solves at the target on an injected eigen
solver (`EigenSolver`: ARPACK or SLEPc, chosen by the eigenmode rule), each
attempt asking for twice as many pairs as the last; refines the seeds
nearest the target (`gfem._eigen.refine`), drops static, mirror,
longitudinal and below-cutoff port pairs (`ModeSearch.below_cutoff`),
refines far seeds again at their own Ritz value and twins of a degenerate
pair the seed solve missed; checks that no nearly converged pair is
skipped; polishes the isolated modes (`gfem._eigen.adjoint.polish`); and
snaps a lossless problem's ``k`` real.
It raises `EigenmodeConvergenceError` with the reasons, and returns the
modes, the guard mode and the gap set as `FoundModes`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Protocol

import numpy as np
import scipy.constants as const

from gfem import _mpi, _progress
from gfem._eigen.adjoint import polish
from gfem._eigen.operator import EigenOperator
from gfem._eigen.reach import GapSet, ModeReach, _close
from gfem._eigen.refine import TRUSTED, Refined, refine
from gfem._eigen.seed import MIN_SUBSPACE, SeedBoundReached
from gfem._engine.port_term import PortTermRefusalError
from gfem._linalg import report
from gfem._linalg.layout import ONE_PROCESS, DofLayout
from gfem._text import hertz
from gfem.errors import EigenmodeConvergenceError, LinearSolverConvergenceError

__all__ = ["EigenSolver", "FoundModes", "ModeSearch"]

#: Seeds with Re k at or below this fraction of the shift are the curl-curl
#: null space, a lossy medium's relaxation modes (Re k = 0), mirror roots or
#: a conductor's linearization cluster (Re k near -sigma), not resonances. A
#: second-order absorbing wall's cluster lies near Re k = 2 sigma: this test
#: keeps it, and only the distance to the target ranks the modes ahead of it.
_STATIC = 1e-6
#: A pair with more than this share of its background-weighted electric
#: energy in a perfectly matched layer is a mode of the layer, not of the
#: device: dropped like a static pair (on a leaky guide the physical modes
#: measured shares up to 0.24, the layer's modes 0.70 and up).
_PML_SHARE = 0.5
#: A converged pair of a law model whose curl energy is at most this fraction
#: of its electric energy (`EigenOperator.curl_ratio`) is a gradient field at
#: a zero of eps(omega), a longitudinal mode: dropped like a static pair
#: (round-off reaches it only below k h_max ~ 3e-6,
#: transverse modes measured 0.55 and up).
_LONGITUDINAL = 1e-4
#: A refined mode closer than this many eigen tolerances (relative, and at
#: least ``SolverOptions.cluster_rtol``) to accepted modes, with this much of
#: its norm in their span, is one of them again.
_DUPLICATE = 10.0
_OVERLAP = 0.9
#: A duplicate with more than this fraction of its norm outside the span
#: landed on a degenerate eigenspace: that part is refined as the twin the
#: seed solve may not have returned (a frozen seed solve gives one Ritz
#: value for a pair).
_TWIN = 1e-2
#: Ritz values this close (relative) are one multiple eigenvalue. The seed
#: solver returns an arbitrary basis of its eigenspace, as nearly parallel as
#: round-off makes it: the seeds are orthonormalized, or two members would
#: overlap like one mode refined twice. Round-off also splits the multiple
#: Ritz value: by up to 3e-8 for a frozen nonlinear seed at its 1e-3 seed
#: tolerance.
_MULTIPLE = 1e-6
#: A seed that needs correction steps is refined on a new factorization at
#: its own Ritz value when the current shift would contract its error by
#: less than this per step (the shift is nearer another Ritz value: residual
#: inverse iteration would drift to that mode).
_CONTRACTION = 0.5
#: Arnoldi runs, each asking for twice as many pairs as the last, before
#: too few modes near the target is an error.
_ATTEMPTS = 3
#: Arnoldi tolerance of a nonlinear problem's frozen seeds. Their roots are
#: off the true ones by the freeze error anyway (small for a conductor,
#: whose seeds reach backward error 1e-8 to 1e-4 now that the projection
#: fixes its potential on the wall; up to 18 % for a second-order
#: absorbing wall) and every seed is refined, so converging them further
#: only spends shift solves. The linearization
#: cluster's solves are avoided by asking for few pairs (see the nev comment
#: in the forward pass), not by this tolerance.
_SEED_TOLERANCE = 1e-3
#: ``|v^H A u - conj(u^H A v)|`` below this times ``|A| |u| |v|`` for ``K``
#: and ``M``: a Hermitian (lossless) Bloch problem, whose eigenvalues are real.
_HERMITIAN = 1e-12


class EigenSolver(Protocol):
    """A seed solve: up to ``nev`` eigenpairs ``(k, x)`` nearest ``op.shift``.

    `gfem._eigen.arpack.shift_invert` or `gfem._eigen.slepc.shift_invert`
    with its options bound. ``project`` is the divergence-free projection
    and ``tol`` the Arnoldi tolerance; it may raise `SeedBoundReached`
    instead of returning.
    """

    def __call__(
        self, op: EigenOperator, project, nev: int, *, tol: float
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclasses.dataclass(frozen=True)
class FoundModes:
    """What a mode search found.

    Attributes:
        modes: The ``n_modes`` modes, nearest the target first, snapped real
            on a lossless problem.
        guard: The guard mode, or ``None`` when none was asked for or no
            seed gave one.
        gaps: The gap set of the modes: the pool, the guard mode's ``k`` and
            the Ritz values.
        reach: The modes' mode-field reach (`ModeReach`): their cluster
            labels at ``SolverOptions.cluster_rtol``, the failed polishes
            and which modes are isolated, on the snapped ``k``.
        anti_bound: The ``k`` of the converged roots dropped below a wave
            port mode's cutoff (`ModeSearch.below_cutoff`), each once.
    """

    modes: list[Refined]
    guard: Refined | None
    gaps: GapSet
    reach: ModeReach
    anti_bound: tuple[complex, ...] = ()

    @property
    def ks(self) -> list[complex]:
        """The modes' ``k``."""
        return [m.k for m in self.modes]

    @property
    def labels(self) -> np.ndarray:
        """The modes' cluster labels (`ModeReach.labels`)."""
        return self.reach.labels

    @property
    def polish_failed(self) -> np.ndarray:
        """Whether each mode's polish failed (`ModeReach.polish_failed`)."""
        return self.reach.polish_failed


@dataclasses.dataclass(frozen=True)
class ModeSearch:
    """The search for the ``n_modes`` modes nearest a target.

    Attributes:
        n_modes: ``Eigenmode.n_modes``.
        target: ``Eigenmode.target``, Hz (for messages).
        eigen_tolerance: ``SolverOptions.eigen_tolerance``.
        max_refine_iterations: ``SolverOptions.max_refine_iterations``.
        cluster_rtol: ``SolverOptions.cluster_rtol``.
        laws: Whether a region holds a permittivity law: only then can a
            converged pair be a longitudinal mode (`_longitudinal`).
        eigen_solver: The seed solve.
        crowding: The seed-bound error's diagnosis line, or ``None``; called
            only when that error is raised.
        refine: Mode refinement (`gfem._eigen.refine.refine`).
        polish: Mode polishing (`gfem._eigen.adjoint.polish`).
        layer_energy: The layer and total energies of a solve-space vector,
            over every rank (`gfem._engine.eigen.layer_energy`): the
            selection drops a mode of a perfectly matched layer
            (`_in_layer`). ``None`` without a layer.
        below_cutoff: Whether a requested wave port mode is below its
            cutoff at a real wavenumber per file unit: the selection drops
            a converged root there that is real (an anti-bound root, its
            field growing into the port's guide) or grows in time (its
            mirror), `_anti_bound`. ``None`` without wave ports.
    """

    n_modes: int
    target: float
    eigen_tolerance: float
    max_refine_iterations: int
    cluster_rtol: float
    laws: bool
    eigen_solver: EigenSolver
    crowding: Callable[[], str | None]
    refine: Callable[..., Refined] = refine
    polish: Callable[..., Refined] = polish
    layer_energy: Callable[[np.ndarray], np.ndarray] | None = None
    below_cutoff: Callable[[float], bool] | None = None

    def find(
        self,
        eop: EigenOperator,
        project,
        sigma: float,
        *,
        guard: bool,
        polish_isolated: bool,
    ) -> FoundModes:
        """The ``n_modes`` modes, the guard mode, the gap set, and failed polishes.

        Seed solves at the target (each attempt asking for twice as many
        pairs as the last) and the selection of their refined pairs
        (`_select`). A seed solve that stops at its restart bound ends the
        attempts: its converged pairs are selected, and a wider retry would
        converge none. ``guard``: refine one mode beyond
        ``n_modes`` (``None`` when no seed gave one). With
        ``polish_isolated``, each isolated mode is polished (`_polish`)
        on the factors the search leaves, before a lossless problem's ``k``
        are snapped real; ``polish_failed`` flags the modes whose polish
        failed (kept unpolished). ``project`` is the divergence-free
        projection the eigen solver applies.

        Raises:
            EigenmodeConvergenceError: Fewer than ``n_modes`` modes
                converged, or the modes found would skip a pair that nearly
                converged (`_skipped`).
            LinearSolverConvergenceError: A shift solve of an iterative
                linear solver did not converge.
        """
        # stage notes only: how many seed solves and refinements a search
        # takes is not known in advance
        with _progress.task(_progress.MODE_SEARCH) as row:
            return self._find(
                eop,
                project,
                sigma,
                guard=guard,
                polish_isolated=polish_isolated,
                row=row,
            )

    def _find(
        self,
        eop: EigenOperator,
        project,
        sigma: float,
        *,
        guard: bool,
        polish_isolated: bool,
        row: _progress.ProgressRow,
    ) -> FoundModes:
        """`find`, its stages noted on the mode search's ``row``."""
        n_modes = self.n_modes
        seed_tolerance = self.eigen_tolerance
        # the companion problem also holds each mode's mirror root
        # -conj(k), |k + sigma| from the shift (behind its mode, about
        # 2 sigma away for a mode near the target), and pairs near
        # k = 0 about sigma away; they take slots in front of wanted
        # modes farther than that, so a damped linear seed asks for
        # twice as many pairs. A nonpolynomial seed asks for one pair
        # beyond n_modes, the guard's slot: behind its modes lies the
        # linearization cluster, which costs shift solves to converge
        # and holds no mode.
        if eop.nonpolynomial:
            nev = n_modes + 1
            seed_tolerance = max(seed_tolerance, _SEED_TOLERANCE)
        else:
            nev = max(2 * n_modes, n_modes + 2)
            nev *= 2 if eop.seed_damped else 1
        found = None
        bound = None
        # the k of the pairs the attempt's selection dropped as modes of the
        # layer
        dropped: list[complex] = []
        # the roots dropped below a port mode's cutoff, over every attempt
        anti_bound: list[complex] = []
        for attempt in range(1, _ATTEMPTS + 1):
            if eop.shift != sigma:
                # a retry is a seed solve: at the target, on a
                # factorization counted for the seed (refinement
                # reshifted it)
                eop.reshift(sigma, solves=MIN_SUBSPACE)
            dropped = []
            row.note(seed_solve=f"{attempt}/{_ATTEMPTS}")
            try:
                k, vectors = self.eigen_solver(eop, project, nev, tol=seed_tolerance)
            except SeedBoundReached as err:
                bound = err
                k, vectors = err.k, err.vectors
            modes, failed, found, pool, ritz = self._select(
                eop,
                k,
                vectors,
                sigma,
                guard=guard,
                dropped=dropped,
                anti_bound=anti_bound,
                row=row,
            )
            report.seed_solve(
                attempt=attempt,
                pairs=nev,
                returned=len(k),
                modes=len(modes),
                wanted=n_modes,
                bound=bound is not None,
            )
            if len(modes) == n_modes or failed or bound is not None:
                break
            nev *= 2  # too few candidates near the target: look wider
        if bound is not None and len(modes) < n_modes:
            raise EigenmodeConvergenceError(self._seed_message(bound))
        skipped = self._skipped(modes, failed, sigma, layout=eop.layout)
        if len(modes) < n_modes or skipped:
            raise EigenmodeConvergenceError(
                self._convergence_message(eop, modes, failed, skipped, dropped)
            )
        guard_k = None if found is None else found.k
        unpolished = np.zeros(len(modes), dtype=bool)
        if polish_isolated:
            # isolation before the snap, on the unsnapped k: the stored reach
            # below is taken again after it
            gap_set = GapSet(pool, guard_k, ritz)
            pre = ModeReach.of(
                [m.k for m in modes], guard_k, gap_set, unpolished, self.cluster_rtol
            )
            modes, unpolished = self._polish(eop, modes, pre)
        lossless = _is_hermitian if eop.general else _is_real
        if not eop.seed_damped and not eop.nonpolynomial and lossless(eop):
            modes = [dataclasses.replace(m, k=m.k.real) for m in modes]
            if found is not None:
                found = dataclasses.replace(found, k=found.k.real)
            pool = pool.real.astype(complex)
        guard_k = None if found is None else found.k
        gaps = GapSet(pool, guard_k, ritz)
        reach = ModeReach.of(
            [m.k for m in modes], guard_k, gaps, unpolished, self.cluster_rtol
        )
        return FoundModes(modes, found, gaps, reach, tuple(anti_bound))

    def _polish(
        self,
        eop: EigenOperator,
        modes: list[Refined],
        reach: ModeReach,
    ) -> tuple[list[Refined], np.ndarray]:
        """``modes`` with each isolated one polished (`polish`), and which failed.

        A mode is isolated (``reach.isolated``) when it is alone in its
        cluster, the guard mode is not in its cluster, and its relative gap
        is at least `gfem._eigen.reach.ISOLATED` (``cluster_rtol`` if
        larger). The guard mode is never
        polished. A polish that does not converge (or whose solve is not
        accepted) is evidence of a neighbour the gap set missed, such as the
        twin of a degenerate pair the seed solve never returned: the mode
        is kept as mode refinement left it and flagged, and its entry
        records it as not isolated. A
        solve never raises because of polishing.
        """
        isolated = reach.isolated
        failed = np.zeros(len(modes), dtype=bool)
        out = []
        for i, m in enumerate(modes):
            if isolated[i]:
                try:
                    m = self.polish(eop, m, float(reach.gap[i]), index=i)  # noqa: PLW2901
                except (EigenmodeConvergenceError, LinearSolverConvergenceError):
                    failed[i] = True
            out.append(m)
        return out, failed

    def _select(  # noqa: PLR0912, PLR0915 - one selection loop over the seeds
        self,
        op: EigenOperator,
        k: np.ndarray,
        vectors: np.ndarray,
        sigma: float,
        *,
        guard: bool = False,
        dropped: list[complex],
        anti_bound: list[complex],
        row: _progress.ProgressRow,
    ) -> tuple[list[Refined], list[Refined], Refined | None, np.ndarray, np.ndarray]:
        """Refine the seeds nearest ``sigma`` until ``n_modes`` distinct ones.

        Seeds (and refined pairs) with ``Re k`` near zero or negative are the
        curl-curl null space, a lossy medium's relaxation modes, mirror
        roots or a conductor's linearization cluster, not resonances: they
        are dropped, and so is a converged longitudinal mode of a law model
        (`_longitudinal`).
        A second-order absorbing wall's cluster has ``Re k`` near ``2
        sigma``, so it is not dropped; the modes nearer the target come
        first. A seed far from the current shift whose refinement gives
        nothing to accept as its mode (see `_lost`) is refined again on a
        factorization at its own Ritz value. A converged pair that is not
        static and not in the span of the accepted modes (`_in_span`) is
        never discarded: the first refinement of such a seed is accepted
        too when it is one, so the accepted modes can outnumber
        ``n_modes``; they are sorted by distance to the target and cut. A
        pair in that span with more than `_TWIN` of its norm outside it
        landed on a degenerate eigenspace: its outside part is refined and,
        converged, is the twin the seed solve did not give.
        With ``guard``, the remaining seeds are refined (without
        refactorizing) until one more distinct mode converges: the guard
        mode, or the nearest mode cut off when there is one. It is
        best-effort: none is found when no remaining seed converges to a
        mode. A seed or converged pair of a perfectly matched layer
        (`_in_layer`) is dropped, its ``k`` recorded in ``dropped``; a
        converged root below a wave port mode's cutoff (`_anti_bound`) is
        dropped, its ``k`` recorded in ``anti_bound``. Each seed's
        refinement is noted on the mode search's progress ``row``.

        Returns:
            The accepted modes, nearest ``sigma`` first (at most
            ``n_modes``), the seeds that did not converge, the guard
            mode (``None`` without ``guard`` or when no seed gave one), the
            pool of the gap set (`GapSet`): the ``k`` of every converged,
            distinct pair accepted, those cut beyond ``n_modes`` included,
            and of a mode found after the count was met, nearest ``sigma``
            first; and the seeds' unrefined Ritz values (none for a
            nonpolynomial operator, whose frozen seeds are too rough).
        """
        n_modes = self.n_modes
        tolerance = self.eigen_tolerance
        seeds = [
            j
            for j in range(len(k))
            if k[j].real > _STATIC * sigma
            and not self._in_layer(k[j], vectors[:, j], dropped)
        ]
        seeds.sort(key=lambda j: abs(k[j].real - sigma))
        layout = op.layout
        vectors = _orthonormal_multiples(k, vectors, layout=layout)
        accepted: list[Refined] = []
        failed: list[Refined] = []
        extra = None
        window = self._window()
        steps = self.max_refine_iterations

        def distinct(mode: Refined) -> Refined | None:
            # a converged pair in the span of the accepted modes is a duplicate
            # unless its part outside the span is more than round-off: then it
            # landed on a degenerate eigenspace whose twin the seed solve
            # missed, and that part, refined, is the twin
            if not _in_span(mode, accepted, window, layout=layout):
                return mode
            rest = _outside(mode, accepted, window, layout=layout)
            if rest is None:
                return None
            twin = self.refine(
                op,
                mode.k,
                rest,
                tolerance=tolerance,
                max_iterations=steps,
                reshift=False,
            )
            if (
                twin.converged
                and twin.k.real > _STATIC * sigma
                and not self._longitudinal(op, twin)
                and not _in_span(twin, accepted, window, layout=layout)
                and not self._in_layer(twin.k, twin.x, dropped)
                and not self._anti_bound(twin, anti_bound)
            ):
                return twin
            return None

        for n, j in enumerate(seeds, 1):
            full = len(accepted) >= n_modes
            if full and not guard:
                break
            row.note(mode_refinement=f"{n}/{len(seeds)}")
            seed = complex(k[j])
            far = _contraction(seed, op.shift, k) > _CONTRACTION
            mode = self.refine(
                op,
                seed,
                vectors[:, j],
                tolerance=tolerance,
                max_iterations=steps,
                reshift=not full,
                reshift_first=far,
            )
            if (
                far
                and not full
                and (
                    _lost(
                        mode,
                        seed,
                        k,
                        sigma,
                        accepted=accepted,
                        window=window,
                        layout=layout,
                    )
                    or self._longitudinal(op, mode)
                    or (mode.converged and self._in_layer(mode.k, mode.x, dropped))
                    or self._anti_bound(mode, anti_bound)
                )
            ):
                # a frozen seed rougher than TRUSTED, refined on a shift nearer
                # another mode, wanders where round-off takes it (onto an
                # accepted mode, a mirror root, or another seed's mode): refine
                # it again on a factorization at its own Ritz value. Where it
                # wandered may be a mode no other seed gives: it is kept
                if (
                    mode.converged
                    and mode.k.real > _STATIC * sigma
                    and not self._longitudinal(op, mode)
                    and not self._in_layer(mode.k, mode.x, dropped)
                    and not self._anti_bound(mode, anti_bound)
                ):
                    kept = distinct(mode)
                    if kept is not None:
                        accepted.append(kept)
                try:
                    op.reshift(seed)
                except PortTermRefusalError as refusal:
                    mode = Refined(
                        seed, vectors[:, j], math.inf, 0, False, None, refusal
                    )
                else:
                    mode = self.refine(
                        op,
                        seed,
                        vectors[:, j],
                        tolerance=tolerance,
                        max_iterations=steps,
                    )
            if (
                mode.k.real <= _STATIC * sigma
                or self._longitudinal(op, mode)
                or (mode.converged and self._in_layer(mode.k, mode.x, dropped))
                or self._anti_bound(mode, anti_bound)
            ):
                continue
            if not mode.converged:
                if not full:
                    failed.append(mode)
                continue
            mode = distinct(mode)
            if mode is None:
                continue
            if full:
                extra = mode
                break
            accepted.append(mode)
        accepted.sort(key=lambda m: abs(m.k.real - sigma))
        pool = [m.k for m in accepted] + ([] if extra is None else [extra.k])
        pool.sort(key=lambda z: abs(z.real - sigma))
        ritz = [complex(k[j]) for j in seeds] if not op.nonpolynomial else []
        if guard and len(accepted) > n_modes:
            # nearer the target than a guard found after the count was met
            extra = accepted[n_modes]
        return (
            accepted[:n_modes],
            failed,
            extra,
            np.array(pool, dtype=complex),
            np.array(ritz, dtype=complex),
        )

    def _longitudinal(self, op: EigenOperator, mode: Refined) -> bool:
        """Whether ``mode`` is a converged longitudinal mode of a law model.

        A gradient field at a zero of a law's eps(omega): its curl energy
        is at most `_LONGITUDINAL` of its electric energy
        (`EigenOperator.curl_ratio`). One ``K`` apply; never on a
        model without a law, where eps has no zero.
        """
        if not self.laws or not mode.converged:
            return False
        return op.curl_ratio(mode.k, mode.x) <= _LONGITUDINAL

    def _anti_bound(self, mode: Refined, anti_bound: list[complex]) -> bool:
        """Whether ``mode`` is a converged root below a wave port mode's cutoff.

        On the outgoing branch, below a requested port mode's cutoff at ``Re
        k`` (`below_cutoff`), a real root (to the duplicate window) is an
        anti-bound root and one with ``Im k < 0`` its mirror, growing in
        time; neither is a resonance. A root there that decays in time is
        one, and is kept. Its ``k`` joins ``anti_bound`` unless one there is
        within 1e-3. Never without wave ports.
        """
        if self.below_cutoff is None or not mode.converged:
            return False
        k = mode.k
        if abs(k.imag) > self._window() * abs(k) and k.imag > 0:
            return False
        if not self.below_cutoff(k.real):
            return False
        if not any(_close(k, other, 1e-3) for other in anti_bound):
            anti_bound.append(complex(k))
        return True

    def _in_layer(self, k: complex, x: np.ndarray, dropped: list[complex]) -> bool:
        """Whether the pair ``(k, x)`` is a mode of the layer.

        It holds more than `_PML_SHARE` of its energy in the layer. Its ``k``
        joins ``dropped`` unless one there is within 1e-3 (one mode met
        as a seed and again refined). Never without a layer.
        """
        if self.layer_energy is None:
            return False
        parts = self.layer_energy(x)
        if not parts[0] > _PML_SHARE * parts[1]:
            return False
        if not any(_close(k, other, 1e-3) for other in dropped):
            dropped.append(complex(k))
        return True

    def _window(self) -> float:
        """Relative distance in ``k`` within which two refined pairs can be one mode."""
        return max(_DUPLICATE * self.eigen_tolerance, self.cluster_rtol)

    def _skipped(
        self,
        accepted: list[Refined],
        failed: list[Refined],
        sigma: float,
        *,
        layout: DofLayout = ONE_PROCESS,
    ) -> list[Refined]:
        """The unconverged pairs that the accepted modes may have skipped.

        A pair that stopped within `TRUSTED` of an eigenpair, nearer the
        target than the farthest accepted mode and outside the span of the
        accepted modes, is a mode the set would silently leave out. Rougher
        pairs (members of a linearization cluster) are not modes.
        """
        if len(accepted) < self.n_modes:
            return []  # the count already raises
        reach = max(abs(m.k.real - sigma) for m in accepted)
        window = self._window()
        return [
            m
            for m in failed
            if m.error <= TRUSTED
            and abs(m.k.real - sigma) < reach
            and not _in_span(m, accepted, window, layout=layout)
        ]

    def _convergence_message(
        self,
        op: EigenOperator,
        accepted: list[Refined],
        failed: list[Refined],
        skipped: list[Refined] | None = None,
        dropped: list[complex] | None = None,
    ) -> str:
        """Why the solve raises: too few modes, or ``skipped`` (see `_skipped`).

        ``dropped`` holds the ``k`` of the pairs the selection dropped as
        modes of a perfectly matched layer.
        """
        c = const.c / op.scale / (2 * math.pi)
        target = hertz(self.target)
        tolerance = (
            f"backward error {self.eigen_tolerance:g} (SolverOptions.eigen_tolerance)"
        )
        if skipped:
            farthest = hertz(accepted[-1].k.real * c)  # sorted by distance
            head = (
                f"Eigenmode found {len(accepted)} mode(s) near {target} to "
                f"{tolerance}, the farthest at {farthest}, but a pair "
                "nearer the target did not converge: the modes found would "
                "skip it."
            )
        else:
            head = (
                f"Eigenmode found {len(accepted)} of the {self.n_modes} "
                f"requested mode(s) near {target} to {tolerance}."
            )
        lines = [head, *(_unconverged_line(m, c) for m in (skipped or failed))]
        if dropped:
            lines.append(
                f"  dropped: {len(dropped)} pair(s) with more than half "
                "their electric energy in the perfectly matched layer (modes of "
                "the layer, not of the device)."
            )
        lines.append(
            "Remedies: move Eigenmode.target closer to the wanted modes; loosen "
            "SolverOptions.eigen_tolerance or raise max_refine_iterations; for a "
            "mode with Q below about 2, find the resonance from a gfem.Driven "
            "sweep instead."
        )
        return "\n".join(lines)

    def _seed_message(self, bound: SeedBoundReached) -> str:
        """Why the solve raises after a seed solve stopped at its restart bound.

        Between the head and the remedies, the line of ``crowding`` (a
        region crowded with modes at the target), when it gives one: a
        diagnosis only.
        """
        target = self.target
        head = (
            f"Eigenmode's seed solve near {hertz(target)} did not converge in "
            f"{bound.restarts} restarts ({bound.applies} shift solves): the modes "
            "there are closer together than their linewidth, so the nearest ones "
            "cannot be told apart."
        )
        lines = [head]
        crowding = self.crowding()
        if crowding is not None:
            lines.append(crowding)
        lines.append(
            "Remedies: move Eigenmode.target where the modes are separated (away "
            "from a lossy resonance); find overlapping resonances from a "
            "gfem.Driven sweep; raise eps_max_it in SolverOptions.petsc_options "
            "(SLEPc) only if the modes are known to be separated."
        )
        return "\n".join(lines)


def _contraction(seed: complex, shift: complex, ritz: np.ndarray) -> float:
    """Residual inverse iteration's error ratio per step for ``seed``.

    ``|seed - shift|`` over the distance from ``shift`` to the nearest other
    Ritz value: the component of that mode grows unless the ratio is below 1.
    The static fields at :math:`k = 0` count as a Ritz value.
    """
    others = ritz[np.abs(ritz - seed) > _MULTIPLE * abs(seed)]
    # the projection keeps the static fields out of the seed solve, not out
    # of the seed: one that is not exactly divergence-free (a lossy cell
    # whose loss varies in space) carries a part of them, which a shift
    # nearer 0 than the seed amplifies at each refinement step
    others = np.append(others, 0.0)
    nearest = float(np.min(np.abs(others - shift)))
    return abs(seed - shift) / nearest if nearest > 0 else math.inf


def _orthonormal_multiples(
    k: np.ndarray, vectors: np.ndarray, *, layout: DofLayout = ONE_PROCESS
) -> np.ndarray:
    """``vectors`` with each multiple Ritz value's vectors orthonormalized.

    Any basis of a multiple eigenvalue's eigenspace is as good a set of
    seeds; an orthonormal one keeps its members apart under `_in_span`.
    The vectors are states, orthonormal over ``layout``'s owned DOFs.
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
            out[:, members] = layout.state_qr(vectors[:, members])
    return out


def _same(
    a: Refined, b: Refined, window: float, *, layout: DofLayout = ONE_PROCESS
) -> bool:
    """Whether ``a`` and ``b`` are one mode: near in ``k``, parallel in ``x``."""
    if abs(a.k - b.k) > window * abs(b.k):
        return False
    norm = layout.state_norm
    overlap = abs(layout.state_inner(a.x, b.x)) / (norm(a.x) * norm(b.x))
    return overlap > _OVERLAP


def _in_span(
    mode: Refined,
    accepted: list[Refined],
    window: float,
    *,
    layout: DofLayout = ONE_PROCESS,
) -> bool:
    """Whether ``mode`` is a combination of accepted modes: a duplicate.

    The norm of its projection on the span of the accepted modes within
    ``window`` (relative) of its ``k`` exceeds `_OVERLAP` of its own. Against
    the span, not each mode: any vector of a degenerate cluster's eigenspace
    is a duplicate, also one at 45 degrees from two accepted members.
    The modes are states, their products ``layout``'s.
    """
    near = [a.x for a in accepted if abs(mode.k - a.k) <= window * abs(a.k)]
    if not near:
        return False
    basis = layout.state_qr(np.stack(near, axis=1))
    inside = np.linalg.norm(layout.state_gram(basis, mode.x))
    return bool(inside > _OVERLAP * layout.state_norm(mode.x))


def _outside(
    mode: Refined,
    accepted: list[Refined],
    window: float,
    *,
    layout: DofLayout = ONE_PROCESS,
) -> np.ndarray | None:
    """``mode``'s part outside the span of the accepted modes near its ``k``.

    ``None`` when it is at most `_TWIN` of its norm: round-off of a duplicate
    of a simple mode (1e-11 to 2e-9 measured), not a twin (0.36 to 0.94 on the
    lossy-wall mirrored cavity). The modes are
    states, their products ``layout``'s.
    """
    near = [a.x for a in accepted if abs(mode.k - a.k) <= window * abs(a.k)]
    if not near:
        return None
    basis = layout.state_qr(np.stack(near, axis=1))
    rest = mode.x - basis @ layout.state_gram(basis, mode.x)
    norm = layout.state_norm(rest)
    if norm <= _TWIN * layout.state_norm(mode.x):
        return None
    return rest / norm


def _unconverged_line(mode: Refined, c: float) -> str:
    """The line of a seed that did not converge, or was lost to a port refusal.

    ``c``: file-unit wavenumber to hertz.
    """
    refusal = mode.refusal
    if refusal is None:
        return (
            f"  not converged: {hertz(mode.k.real * c)}, best backward "
            f"error {mode.error:.3g} after {mode.iterations} refinement steps"
        )
    f = refusal.k * c
    sign = "-" if f.imag < 0 else "+"
    return (
        f"  lost: {hertz(mode.k.real * c)}, an iterate at complex frequency "
        f"{hertz(f.real)} {sign} i {hertz(abs(f.imag))} left the range where "
        f"{refusal.label} "
        "can follow its port mode"
    )


def _lost(
    mode: Refined,
    seed: complex,
    ritz: np.ndarray,
    sigma: float,
    *,
    accepted: list[Refined],
    window: float,
    layout: DofLayout = ONE_PROCESS,
) -> bool:
    """Whether ``seed``'s refinement gave nothing to accept as its mode.

    It did not converge, converged to a static or mirror root, converged
    nearer another of the ``ritz`` values than ``seed`` (that seed's mode,
    which would take this one's place), or converged to an accepted mode
    the seed is not near.
    """
    if not mode.converged or mode.k.real <= _STATIC * sigma:
        return True
    others = ritz[np.abs(ritz - seed) > _MULTIPLE * abs(seed)]
    if len(others) and np.min(np.abs(others - mode.k)) < abs(mode.k - seed):
        return True
    return any(
        _same(mode, other, window, layout=layout) and not _close(seed, other.k, window)
        for other in accepted
    )


def _is_real(op: EigenOperator) -> bool:
    """Whether ``K`` and ``M`` are real: a lossless, undamped problem."""
    probe = np.where(op.free, np.random.default_rng(1).standard_normal(len(op.free)), 0)
    kx, _, mx = op.parts(probe.astype(complex))
    # a real matrix gives a real load on any rank's DOFs: every rank must agree
    real = not kx.imag.any() and not mx.imag.any()
    return _mpi.agree_all(op.layout.comm, flag=real)


def _is_hermitian(op: EigenOperator) -> bool:
    """Whether ``K`` and ``M`` are Hermitian: a lossless Bloch problem.

    A Bloch-phased ``K`` is complex even when lossless (`_is_real` never
    holds): two random complex probes ``u``, ``v`` test ``|v^H A u -
    conj(u^H A v)| <= _HERMITIAN |A| |u| |v|`` for ``K`` and ``M``.
    """
    rng = np.random.default_rng(1)
    n = len(op.free)
    u, v = (
        np.where(op.free, rng.standard_normal(n) + 1j * rng.standard_normal(n), 0)
        for _ in range(2)
    )
    ku, _, mu = op.parts(u)
    kv, _, mv = op.parts(v)
    nk, _, nm, *_ = op.norms
    size = np.linalg.norm(u) * np.linalg.norm(v)
    return all(
        abs(np.vdot(v, au) - np.conj(np.vdot(u, av))) <= _HERMITIAN * norm * size
        for au, av, norm in ((ku, kv, nk), (mu, mv, nm))
    )
