"""Adaptive frequency sweep: a greedy reduced model sampled on the output grid.

The system is affine in frequency,

    A(ω) = Σ_k θ_k(ω) A_k,    B(ω) = Σ_j E_j φ_j(ω),    y(ω) = Σ_j ψ_j(ω) L_jᵀ x,

with every A_k frequency-independent and complex symmetric (A_kᵀ = A_k),
E_j and L_j blocks of full-length vectors and θ_k, φ_j, ψ_j scalars or
small matrices of ω. One column of B per excitation.

The sweep solves the full system at a few grid frequencies (all excitations
in one call) and projects it on a real orthonormal basis V of the real and
imaginary parts of those solutions (Galerkin: Vᵀ A(ω) V is complex
symmetric again). The two grid endpoints come first; each next sample is
the unsampled grid frequency where the minimal rational interpolation
(MRI) of the solutions has the smallest denominator, i.e. its largest
error estimate. The error of a sample, measured before it joins the basis,
is the largest relative l2 difference between the full and reduced
solutions over the excitations. The sweep has converged when ``memory``
consecutive samples fall below ``tolerance`` or the grid is used up; it
stops with `gfem.SweepNotConvergedWarning` at ``max_samples``. Outputs at
every grid frequency come from the reduced model.

Pure NumPy: the engine supplies the full solve and the piece products.
Excitations and outputs are sums of blocks with frequency-dependent
coefficient matrices, so a term that is affine in other coordinates (a
wave port's reduced port-mode coordinates, tabulated on the grid) is one
more block.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Callable, Sequence

import numpy as np

from gfem.errors import SweepNotConvergedWarning

__all__ = ["AffineSystem", "AffineTerm", "ReducedModel", "Sweep", "adaptive_sweep"]

#: A basis candidate whose orthogonal part is below this fraction of its
#: norm is dropped (already in the span).
DROP = 1e-10
#: Singular values of the MRI snapshot matrix below this fraction of the
#: largest one are treated as zero.
MRI_RANK = 1e-12


@dataclasses.dataclass(frozen=True, eq=False)
class AffineTerm:
    """A block of full-length vectors and its coefficient matrix over ω.

    Attributes:
        vectors: ``(N, m)``.
        coefficients: ``ω -> (m, n_excitations)`` for an excitation term,
            ``ω -> (n_outputs, m)`` for an output term.
    """

    vectors: np.ndarray
    coefficients: Callable[[float], np.ndarray]


@dataclasses.dataclass(frozen=True, eq=False)
class AffineSystem:
    """The full system, as the sweep sees it.

    Attributes:
        thetas: θ_k(ω) of each operator piece.
        apply: ``(k, X) -> A_k X`` for a real ``(N, m)`` block ``X``.
        excitation: The right-hand side terms (one column per excitation).
        outputs: The output terms.
        solve: ``(ω, B, x0) -> X``: the full solve at ω for the right-hand
            sides ``B`` ``(N, n_excitations)``, from the initial guess
            ``x0`` (``None`` for none).
    """

    thetas: Sequence[Callable[[float], complex]]
    apply: Callable[[int, np.ndarray], np.ndarray]
    excitation: Sequence[AffineTerm]
    outputs: Sequence[AffineTerm]
    solve: Callable[[float, np.ndarray, np.ndarray | None], np.ndarray]

    def rhs(self, omega: float) -> np.ndarray:
        """The right-hand sides at ω, ``(N, n_excitations)``."""
        return sum(
            t.vectors @ np.asarray(t.coefficients(omega)) for t in self.excitation
        )


class ReducedModel:
    """The Galerkin projection of an `AffineSystem` on a growing real basis."""

    def __init__(self, system: AffineSystem) -> None:
        """An empty model of ``system``."""
        self.system = system
        n_rows = system.excitation[0].vectors.shape[0]
        #: The orthonormal basis V, ``(N, n)`` real.
        self.basis = np.zeros((n_rows, 0))
        #: Vᵀ A_k V per piece.
        self.operators = [np.zeros((0, 0), complex) for _ in system.thetas]
        #: Vᵀ E_j per excitation term.
        self.excitation = [np.zeros((0, t.vectors.shape[1])) for t in system.excitation]
        #: L_jᵀ V per output term.
        self.outputs = [np.zeros((t.vectors.shape[1], 0)) for t in system.outputs]

    @property
    def size(self) -> int:
        """The basis size n."""
        return self.basis.shape[1]

    def extend(self, snapshot: np.ndarray) -> int:
        """Add the real and imaginary parts of ``snapshot`` ``(N, p)``.

        Two-pass classical Gram-Schmidt; a column already in the span (to
        `DROP`) is dropped. Projects the pieces on the new columns only
        (each piece is symmetric).

        Returns:
            The number of columns added.
        """
        columns = np.stack([snapshot.real, snapshot.imag], axis=2).reshape(
            snapshot.shape[0], -1
        )
        old = self.basis
        basis = old
        for column in columns.T:
            norm = np.linalg.norm(column)
            if norm == 0:
                continue
            w = column.copy()
            for _ in range(2):
                w -= basis @ (basis.T @ w)
            remaining = np.linalg.norm(w)
            if remaining <= DROP * norm:
                continue
            basis = np.column_stack([basis, w / remaining])
        new = basis[:, old.shape[1] :]
        if not new.shape[1]:
            return 0
        for k, reduced in enumerate(self.operators):
            product = np.asarray(self.system.apply(k, new))
            top = old.T @ product
            corner = new.T @ product
            corner = 0.5 * (corner + corner.T)
            self.operators[k] = np.block([[reduced, top], [top.T, corner]])
        self.excitation = [
            np.vstack([r, new.T @ t.vectors])
            for r, t in zip(self.excitation, self.system.excitation, strict=True)
        ]
        self.outputs = [
            np.hstack([r, t.vectors.T @ new])
            for r, t in zip(self.outputs, self.system.outputs, strict=True)
        ]
        self.basis = basis
        return new.shape[1]

    def coordinates(self, omega: float) -> np.ndarray:
        """Reduced solution at ω, ``(n, n_excitations)``: V y is the field."""
        system = self.system
        matrix = sum(
            complex(theta(omega)) * a
            for theta, a in zip(system.thetas, self.operators, strict=True)
        )
        rhs = sum(
            r @ np.asarray(t.coefficients(omega))
            for r, t in zip(self.excitation, system.excitation, strict=True)
        )
        return np.linalg.solve(matrix, rhs)

    def evaluate(self, omega: float, coordinates: np.ndarray) -> np.ndarray:
        """The outputs of the reduced solution, ``(n_outputs, n_excitations)``."""
        return sum(
            np.asarray(t.coefficients(omega)) @ (r @ coordinates)
            for r, t in zip(self.outputs, self.system.outputs, strict=True)
        )


@dataclasses.dataclass(frozen=True, eq=False)
class Sweep:
    """What an adaptive sweep solved and what it gives on the grid.

    Attributes:
        samples: Grid indices solved in full, in solve order.
        errors: The error of each sample when it was added; ``inf`` where
            there was nothing to compare with yet (the greedy's endpoints,
            the first frozen sample).
        converged: Whether the sweep met its stopping rule.
        model: The reduced model.
        coordinates: Reduced solution per grid frequency, ``(n_freq, n,
            n_excitations)``.
        outputs: Outputs per grid frequency, ``(n_freq, n_outputs,
            n_excitations)``.
    """

    samples: tuple[int, ...]
    errors: tuple[float, ...]
    converged: bool
    model: ReducedModel
    coordinates: np.ndarray
    outputs: np.ndarray

    @property
    def basis(self) -> np.ndarray:
        """The reduced basis V, ``(N, n)`` real."""
        return self.model.basis


def adaptive_sweep(
    system: AffineSystem,
    omegas: Sequence[float] | np.ndarray,
    *,
    tolerance: float,
    max_samples: int,
    memory: int,
    samples: Sequence[int] | None = None,
    z_scale: float = 1.0,
) -> Sweep:
    """Sweep ``system`` over the grid ``omegas`` (see the module docstring).

    Args:
        system: The full system.
        omegas: The output grid, rad/s (any order, distinct).
        tolerance: Sample error under which a sample counts towards
            convergence.
        max_samples: Most frequencies the greedy solves.
        memory: Consecutive samples under ``tolerance`` that converge.
        samples: Grid indices to solve, in order, instead of the greedy.
        z_scale: ω times ``z_scale`` is the MRI's (dimensionless)
            interpolation variable.

    Returns:
        The `Sweep`.
    """
    omegas = np.asarray(omegas, dtype=float)
    z = omegas * z_scale
    model = ReducedModel(system)
    chosen: list[int] = []
    errors: list[float] = []
    snapshots: list[np.ndarray] = []

    def add(i: int, *, measure: bool) -> None:
        omega = float(omegas[i])
        guess = None
        if model.size:
            guess = model.basis @ model.coordinates(omega)
        x = np.asarray(system.solve(omega, system.rhs(omega), guess))
        error = math.inf
        if measure and guess is not None:
            error = _error(x, guess)
        model.extend(x)
        snapshots.append(model.basis.T @ x)
        chosen.append(i)
        errors.append(error)

    n_freq = len(omegas)
    if samples is None:
        for i in dict.fromkeys((0, n_freq - 1)):
            add(i, measure=False)
        streak = 0
        while streak < memory and len(chosen) < min(n_freq, max_samples):
            add(_next_sample(z, chosen, snapshots, model.size), measure=True)
            streak = streak + 1 if errors[-1] < tolerance else 0
        converged = streak >= memory or len(chosen) == n_freq
        if not converged:
            warnings.warn(
                f"The adaptive sweep stopped at adaptive_max_samples={max_samples} "
                f"frequencies before converging: its last sample error is "
                f"{errors[-1]:.3g} against adaptive_tolerance={tolerance:g}. The "
                "S-parameters come from the reduced model as it stands. Raise "
                "SolverOptions.adaptive_max_samples, loosen adaptive_tolerance, or "
                "use sweep='uniform'.",
                SweepNotConvergedWarning,
                stacklevel=2,
            )
    else:
        for k, i in enumerate(samples):
            add(int(i), measure=k > 0)
        streak = next(
            (k for k, e in enumerate(reversed(errors)) if not e < tolerance),
            len(errors),
        )
        converged = streak >= memory or len(set(chosen)) == n_freq

    coordinates = np.stack([model.coordinates(float(w)) for w in omegas])
    outputs = np.stack(
        [model.evaluate(float(w), y) for w, y in zip(omegas, coordinates, strict=True)]
    )
    return Sweep(
        samples=tuple(chosen),
        errors=tuple(errors),
        converged=bool(converged),
        model=model,
        coordinates=coordinates,
        outputs=outputs,
    )


def _error(x: np.ndarray, guess: np.ndarray) -> float:
    """Largest relative l2 error of ``guess`` over the columns of ``x``."""
    worst = 0.0
    for full, reduced in zip(x.T, guess.T, strict=True):
        norm = np.linalg.norm(full)
        diff = np.linalg.norm(full - reduced)
        worst = max(worst, diff / norm if norm else diff)
    return worst


def _next_sample(
    z: np.ndarray, chosen: list[int], snapshots: list[np.ndarray], size: int
) -> int:
    """The unsampled grid index with the smallest MRI denominator."""
    columns = []
    for i, c in zip(chosen, snapshots, strict=True):
        padded = np.zeros((size, c.shape[1]), complex)
        padded[: c.shape[0]] = c
        flat = padded.reshape(-1)
        columns.append(np.concatenate([flat, 1j * z[i] * flat]))
    u = np.column_stack(columns)
    _, sigma, vh = np.linalg.svd(u)
    sigma = np.concatenate([sigma, np.zeros(len(chosen) - len(sigma))])
    m = len(chosen) - 1
    while m > 0 and sigma[m] < MRI_RANK * sigma[0]:
        m -= 1
    if m < len(chosen) - 1:
        warnings.warn(
            "The adaptive sweep's rational interpolation is rank-deficient "
            f"(singular value {sigma[-1]:.3g} against {sigma[0]:.3g}): the sweep is "
            "near the accuracy of its full solves. Tighten SolverOptions.tolerance "
            "or loosen adaptive_tolerance if it fails to converge.",
            UserWarning,
            stacklevel=3,
        )
    q = vh[m].conj()
    zs = z[chosen]
    candidates = np.setdiff1d(np.arange(len(z)), chosen)
    denominator = np.abs((q[None, :] / (z[candidates, None] - zs[None, :])).sum(axis=1))
    return int(candidates[np.argmin(denominator)])
