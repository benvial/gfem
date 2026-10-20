"""Port modes over a frequency grid: the guarded port mode sweep.

A wave port's mode over the whole output grid comes from a small reduced
model instead of one exact solve per frequency:

1. **Exact solves** factor the shifted 2D pencil with SuperLU and run ARPACK
   shift-invert at ``sigma = -1.1 (n_max k0)^2``. At the lowest frequency
   the requested mode is picked in Palace's order: converged, propagating
   Ritz pairs below the largest bulk wavenumber (the formulation's spurious
   modes sit above it), by decreasing ``Re 1 / (lam - sigma)``.
2. **Reduced solves** are Rayleigh-Ritz on the affine pencil projected onto
   a real orthonormal basis of the real and imaginary parts of every exact
   mode.
3. **The guard** is the backward error ``|A x - lam B x| / (|A x| +
   |lam| |B x|)`` of every lifted reduced pair, computed exactly from a QR
   factor of the projected pencil pieces (``O(n^2)`` per frequency). A
   frequency passes at ``max(tolerance, 10 x the backward error of the
   exact solve at the nearest sample)``: the second term is the
   formulation's low-frequency accuracy floor.
4. **Enrichment**: an exact solve at the worst frequency, until every
   frequency passes. With at most three frequencies, each is solved exactly.

Across frequencies the mode is **tracked** by its overlap (L2 inner product)
with the neighbouring tabulated mode, never by ranking, which follows the
physical mode through crossings and ignores spurious Ritz values. Each mode
is normalized to 1 W. Its phase follows Palace's rule at the lowest
frequency and is continuous after it (the overlap with the previous
frequency's mode is real and positive); with a voltage path, ``Re V > 0`` at
every frequency.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants
import scipy.linalg

from gfem._boundaries import WavePort
from gfem._checks import is_traced
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.runtime import task_manager
from gfem._linalg.superlu import backward_error, shift_invert_eigs
from gfem._port_section import PortSection, port_section
from gfem.errors import PortModeError, PortModeTrackingWarning
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import PortMode
from gfem.results import PortModeField, PortModeResult

__all__ = ["PortModeRule", "PortModeSweep", "path_voltage", "port_mode_sweep"]

# The shift is this factor above the largest bulk (Re eps)(Re mu), as in Palace.
_TARGET = 1.1
# A Ritz pair propagates when Re beta exceeds this fraction of k0 n_max.
_EVANESCENT = 1e-6
# Ritz pairs with a larger backward error are unconverged (ARPACK returns some
# from the formulation's large null space below cutoff); converged ones sit
# near 1e-14, or near 1e-6 at the low-frequency accuracy floor.
_UNCONVERGED = 1e-3
# A reduced pair passes at this multiple of the nearest exact backward error.
_FLOOR = 10.0
# Two candidates whose overlaps with the tracked mode differ by less than
# this cannot be told apart.
_AMBIGUOUS = 0.05
# Relative distance of two propagation constants that are one degenerate mode.
_DEGENERATE = 1e-8
# A surface impedance on the port's edges (the internal inductance of a
# conductor, an inductive sheet) slows the mode beyond the bulk wavenumber;
# propagating candidates may then reach this multiple of it.
_SLOW_WAVE = 2.0
# With at most this many frequencies every one is solved exactly.
_EXACT_ONLY = 3
# A new mode adds its part orthogonal to the basis unless that part is below
# this fraction of the mode (rounding noise). It must be tiny: a smooth mode
# has |A x| << |A| |x|, so a part of 1e-8 still carries a backward error ~1e-6.
_NEW_DIRECTION = 1e-12
# Gauss-Legendre rule for voltage line integrals: points per piece, pieces.
_GAUSS = np.polynomial.legendre.leggauss(4)
_PIECES = 32
_Z0 = scipy.constants.mu_0 * scipy.constants.c


@dataclasses.dataclass(frozen=True, eq=False)
class PortModeSweep:
    """One wave port's mode over a frequency grid.

    The field at ``frequency[i]`` is ``basis @ coords[i]`` (full coefficient
    vector of ``(E_t, E_n)`` on ``space``, V/m at 1 W). The basis is real and
    fixed; it splits into a tangential and a normal block, so the field is
    affine in ``coords`` even though the pencil vector is ``(gamma E_t,
    E_n)``.

    Attributes:
        space: The discrete port mode problem.
        frequency: ``(n_f,)`` Hz, in the order given.
        gamma: ``(n_f,)`` propagation constants ``alpha + i beta``, 1/m.
        residual: ``(n_f,)`` backward error of each frequency's mode.
        solved: ``(n_f,)`` True where the mode is an exact solve.
        floor: ``(n_f,)`` the residual each frequency had to reach.
        basis: ``(ndof, m)`` real.
        coords: ``(n_f, m)`` complex.
        voltage: ``(n_f,)`` complex voltage along the port's voltage path,
            V; NaN without one.
    """

    space: PortModeSpace
    frequency: np.ndarray
    gamma: np.ndarray
    residual: np.ndarray
    solved: np.ndarray
    floor: np.ndarray
    basis: np.ndarray
    coords: np.ndarray
    voltage: np.ndarray

    @property
    def z_pv(self) -> np.ndarray:
        """Power-voltage impedance ``|V|^2 / (2 P)`` at ``P = 1 W``, ohm."""
        return np.abs(self.voltage) ** 2 / 2.0


class PortModeRule:
    """Host-side solve of a `PortMode` problem on a `Model`.

    Construction runs the static checks; `forward` extracts the port's
    cross-section and runs the port mode sweep over its frequencies. Port
    modes are constants of a solve: under `jax.grad` a perturbed input the
    port mode problem reads raises `gfem.NonDifferentiableError`, one it
    never reads (a thermal property) gets an exact zero, and `jax.jit` /
    `jax.vmap` run the solve (the result's field is then not available).
    """

    physics = "em"
    form = "port mode"
    #: Material properties the port mode problem reads.
    reads = frozenset({"eps_r", "mu_r", "tan_delta", "conductivity"})
    #: No output has a gradient (`unsupported_input` raises for read inputs).
    differentiable_outputs = frozenset()

    def __init__(self, model: Model, problem: PortMode, options: SolverOptions) -> None:
        """Check that the problem fits the model (see `gfem.PortMode`).

        Raises:
            ValueError: ``problem.port`` is not a wave port of the model, or
                ``options`` asks for a linear solver or a device: the 2D
                port mode problem always runs SuperLU on the CPU.
        """
        if not isinstance(problem, PortMode):
            raise TypeError(f"Expected gfem.PortMode, got {problem!r}.")
        if options.device != "cpu" or options.linear_solver != "auto":
            raise ValueError(
                "The port mode problem always uses SuperLU on the CPU (a 2D "
                f"problem), but the options ask for device={options.device!r}, "
                f"linear_solver={options.linear_solver!r}. Solve PortMode with "
                "the default device and linear_solver."
            )
        self.problem = problem
        self.options = options
        self.port = _wave_port(model, problem.port)
        self.mode = self.port.mode if problem.mode is None else problem.mode
        self._model = model

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output (one value per frequency)."""
        del model
        n = (len(self.problem.frequency),)
        return {
            "gamma": jax.ShapeDtypeStruct(n, jnp.complex128),
            "Z_PV": jax.ShapeDtypeStruct(n, jnp.float64),
            "residual": jax.ShapeDtypeStruct(n, jnp.float64),
            "solved": jax.ShapeDtypeStruct(n, jnp.bool_),
        }

    def role(self, kind: tuple) -> str:
        """``"read"`` if the port mode problem depends on input ``kind``.

        The electromagnetic material properties, the node coordinates and
        any boundary value are read; thermal properties are not.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        return "read"

    def unsupported_input(self, kind: tuple) -> str | None:
        """Every read input: port modes are constants of a solve."""
        del kind
        return (
            "the port mode problem is not differentiable in this version (port "
            "modes are constants of a solve)"
        )

    def nondifferentiable_output(self, name: str) -> str:
        """The error message for a cotangent on output ``name``."""
        return (
            f"gfem cannot differentiate the port mode output {name!r}: the port "
            "mode problem is not differentiable in this version. Wrap it in "
            "jax.lax.stop_gradient to use its value without its derivative."
        )

    def forward(self, model: Model, *, keep: bool) -> tuple[dict, None]:
        """Run the port mode sweep.

        Args:
            model: The model, with concrete leaves.
            keep: Ignored: nothing is kept for a backward pass.

        Returns:
            The per-frequency arrays of `output_specs` plus ``"sweep"`` (the
            `PortModeSweep`), and ``None``.
        """
        del keep
        section = port_section(model, self.port)
        with task_manager(self.options.threads):
            space = PortModeSpace(section, self.options.order)
            sweep = port_mode_sweep(
                space,
                self.problem.frequency,
                self.mode,
                tolerance=self.options.port_mode_tolerance,
            )
        outputs = {
            "gamma": sweep.gamma,
            "Z_PV": sweep.z_pv,
            "residual": sweep.residual,
            "solved": sweep.solved,
            "sweep": sweep,
        }
        return outputs, None

    def result(self, outputs: dict) -> PortModeResult:
        """The public result from the forward outputs.

        Without ``"sweep"`` (a solve traced by JAX) the mode field is built
        on first use, from the model's values when they are concrete.
        """
        sweep: PortModeSweep | None = outputs.get("sweep")
        if sweep is None:
            field = PortModeField._deferred(self.port.name, self._field)  # noqa: SLF001
        else:
            field = PortModeField(sweep.space, sweep.basis, sweep.coords)
        return PortModeResult(
            port=self.port.name,
            mode=self.mode,
            frequency=jnp.asarray(self.problem.frequency),
            gamma=jnp.asarray(outputs["gamma"]),
            Z_PV=jnp.asarray(outputs["Z_PV"]),
            residual=jnp.asarray(outputs["residual"]),
            solved=jnp.asarray(outputs["solved"]),
            e_field=field,
        )

    def _field(self) -> tuple[PortModeSpace, np.ndarray, np.ndarray]:
        if is_traced(self._model):
            raise TypeError(
                "PortModeResult.e_field is not available from a port mode solve "
                "traced by JAX (jax.jit, jax.grad, jax.vmap): the mode field needs "
                "concrete model values. Solve PortMode outside the transformed "
                "function to read e_field."
            )
        sweep = self.forward(self._model, keep=False)[0]["sweep"]
        return sweep.space, sweep.basis, sweep.coords


def _wave_port(model: Model, name: str) -> WavePort:
    for p in model.ports:
        if p.name == name:
            if not isinstance(p, WavePort):
                raise ValueError(
                    f"PortMode(port={name!r}): {name!r} is a {type(p).__name__}; the "
                    "port mode problem needs a WavePort."
                )
            return p
    waves = [p.name for p in model.ports if isinstance(p, WavePort)]
    known = ", ".join(repr(w) for w in waves) if waves else "none"
    raise ValueError(
        f"PortMode(port={name!r}): the model has no port named {name!r}. Its wave "
        f"ports: {known}."
    )


# --- the sweep ------------------------------------------------------------------


def port_mode_sweep(
    space: PortModeSpace,
    frequency: tuple[float, ...] | np.ndarray,
    mode: int,
    *,
    tolerance: float,
) -> PortModeSweep:
    """The guarded port mode sweep of one wave port (see the module docstring).

    Args:
        space: The port's discrete port mode problem.
        frequency: The output grid, Hz (any order).
        mode: Which mode at the lowest frequency, in Palace's order.
        tolerance: Backward error a reduced mode must reach.

    Raises:
        PortModeError: The mode is below cutoff somewhere in the band, or is
            degenerate, or two candidates cannot be told apart.

    Warns:
        PortModeTrackingWarning: The tracked mode stops being the ``mode``-th
            in Palace's order inside the band (a crossing).
    """
    f_given = np.asarray(frequency, float)
    section = space.section
    rule = _voltage_rule(section) if section.port.voltage_path is not None else None
    order = np.argsort(f_given, kind="stable")
    sweep = _Greedy(space, f_given[order], mode, tolerance)
    sweep.run()
    basis, coords, gamma, voltage = sweep.normalized(rule)
    out = np.empty_like(order)
    out[order] = np.arange(len(order))
    return PortModeSweep(
        space=space,
        frequency=f_given,
        gamma=gamma[out],
        residual=sweep.residual[out],
        solved=sweep.solved[out],
        floor=sweep.floor[out],
        basis=basis,
        coords=coords[out],
        voltage=voltage[out],
    )


@dataclasses.dataclass(frozen=True)
class _Candidates:
    """The converged propagating eigenpairs of one exact solve, in Palace's order."""

    lam: np.ndarray
    vecs: np.ndarray
    residual: np.ndarray


class _Greedy:
    """State of one port mode sweep over a sorted grid.

    Tracking is certified link by link: a frequency's mode is trusted only
    when chosen by overlap with an accurate mode at the previous grid
    frequency. An exact solve whose reference was further away (the top of
    the band at the start, or a frequency after a failing stretch) is
    re-checked once its neighbour below passes, and re-selected if another
    of its candidates continues the mode better.
    """

    def __init__(
        self, space: PortModeSpace, f: np.ndarray, mode: int, tolerance: float
    ) -> None:
        self.space = space
        self.f = f
        self.mode = mode
        self.tolerance = tolerance
        n = len(f)
        self.k0 = np.empty(n)
        self.k_max = np.empty(n)
        self.shift = np.empty(n)
        for i, fi in enumerate(f):
            self.k0[i], self.k_max[i], self.shift[i] = _wavenumbers(space.section, fi)
        self.reduced = _Reduced(space)
        self.lam = np.zeros(n, complex)
        self.coords: list[np.ndarray] = [np.zeros(0)] * n
        self.residual = np.full(n, np.inf)
        self.solved = np.zeros(n, bool)
        self.floor = np.full(n, tolerance)
        # exact solves: candidates and the position (in Palace's order) kept
        self.candidates: dict[int, _Candidates] = {}
        self.pick: dict[int, int] = {}

    @property
    def name(self) -> str:
        return self.space.section.port.name

    # --- driver ---------------------------------------------------------------

    def run(self) -> None:
        n = len(self.f)
        first = self._solve(0)
        if len(first.lam) < self.mode:
            raise self._cutoff_error(first)
        self._check_degenerate(first)
        self._keep(0, first, self.mode - 1)
        if n <= _EXACT_ONLY:
            for j in range(1, n):
                self._exact(j, j - 1, certain=True)
        else:
            self._exact(n - 1, 0, certain=False)
        for _ in range(4 * n + 8):
            passed = self._tabulate()
            if passed is None:  # a sample was re-selected
                self._rebuild()
                continue
            if passed.all():
                self._warn_crossing()
                return
            excess = self.residual / self.floor
            j = int(np.argmax(np.where(passed, -np.inf, excess)))
            if passed[j - 1]:
                self._exact(j, j - 1, certain=True)
            else:
                good = np.flatnonzero(passed)
                self._exact(j, int(good[np.argmin(np.abs(good - j))]), certain=False)
        raise PortModeError(  # pragma: no cover - a safeguard
            f"WavePort {self.name!r}: the port mode sweep could not settle which "
            f"mode continues mode {self.mode} over {self._band()}. Narrow the band "
            "or use a LumpedPort."
        )

    # --- exact solves -----------------------------------------------------------

    def _solve(self, j: int) -> _Candidates:
        space, mode = self.space, self.mode
        a, b = space.pencil.at(self.k0[j])
        n = a.shape[0]
        first = max(6, 2 * mode + 4)
        for want in (first, 4 * first):
            lam, vecs = shift_invert_eigs(a, b, self.shift[j], min(want, n - 2))
            beta = np.sqrt(-lam + 0j)
            res = np.array(
                [backward_error(a, b, lam[i], vecs[:, i]) for i in range(len(lam))]
            )
            k_max = self.k_max[j]
            ok = (
                (beta.real > _EVANESCENT * k_max)
                & (beta.real <= k_max * (1 + 1e-6))
                & (res < _UNCONVERGED)
            )
            # Palace's order: largest Re of the shift-inverted eigenvalue
            rank = (1 / (lam - self.shift[j])).real
            ranked = np.flatnonzero(ok)[np.argsort(-rank[ok], kind="stable")]
            if len(ranked) >= mode or want >= n - 2:
                break
        return _Candidates(lam=lam[ranked], vecs=vecs[:, ranked], residual=res[ranked])

    def _keep(self, j: int, cand: _Candidates, pos: int) -> None:
        """Make candidate ``pos`` of the exact solve at ``j`` the mode there."""
        new = j not in self.candidates
        self.candidates[j] = cand
        self.pick[j] = pos
        self.solved[j] = True
        self.lam[j] = cand.lam[pos]
        self.residual[j] = cand.residual[pos]
        if new:
            self.reduced.add(cand.vecs[:, pos])

    def _exact(self, j: int, ref: int, *, certain: bool) -> None:
        """Exact solve at ``j``, keeping the candidate closest to mode ``ref``.

        Args:
            j: Grid index.
            ref: Grid index of the reference mode.
            certain: The reference is an accurate mode at the grid neighbour,
                so an ambiguous choice is an error.
        """
        cand = self._solve(j)
        if len(cand.lam) == 0:
            raise self._evanescent_error(j)
        self._keep(j, cand, self._closest(j, cand, ref, certain=certain))

    def _closest(self, j: int, cand: _Candidates, ref: int, *, certain: bool) -> int:
        x_ref = self.reduced.lift(self.coords_at(ref))
        overlap = _overlaps(self.space.mass, x_ref, cand.vecs)
        best = np.argsort(-overlap, kind="stable")
        if (
            certain
            and len(best) > 1
            and overlap[best[1]] > overlap[best[0]] - _AMBIGUOUS
        ):
            raise PortModeError(
                f"WavePort {self.name!r}: two modes overlap the tracked mode "
                f"{self.mode} almost equally at {self.f[j] / 1e9:.6g} GHz (overlaps "
                f"{overlap[best[0]]:.3f} and {overlap[best[1]]:.3f}): a degenerate "
                "pair or an unresolved crossing. Break the cross-section's symmetry, "
                "or make the unwanted mode non-propagating (for example with "
                "WavePort(pec_edges=...))."
            )
        return int(best[0])

    def _rebuild(self) -> None:
        """A fresh basis from the exact modes kept now."""
        self.reduced = _Reduced(self.space)
        for j in sorted(self.candidates):
            self.reduced.add(self.candidates[j].vecs[:, self.pick[j]])

    def _warn_crossing(self) -> None:
        for j in sorted(self.pick):
            if self.pick[j] != self.mode - 1:
                warnings.warn(
                    f"WavePort {self.name!r}: the tracked mode {self.mode} is mode "
                    f"{self.pick[j] + 1} in Palace's order at "
                    f"{self.f[j] / 1e9:.6g} GHz (a mode crossing inside the band). "
                    "gfem keeps following the physical mode; Palace would switch "
                    "modes there, so S-parameters differ from Palace's beyond the "
                    "crossing.",
                    PortModeTrackingWarning,
                    stacklevel=6,
                )
                return

    def coords_at(self, j: int) -> np.ndarray:
        """Reduced coordinates of the tabulated mode at ``j`` (current basis)."""
        if self.solved[j]:
            return self.reduced.project(self.candidates[j].vecs[:, self.pick[j]])
        y = self.coords[j]
        return np.concatenate([y, np.zeros(self.reduced.size - len(y))])

    # --- reduced solves -----------------------------------------------------------

    def _tabulate(self) -> np.ndarray | None:
        """Walk up the grid, filling unsolved frequencies from the reduced model.

        Each frequency tracks the mode at the previous one.

        Returns:
            Which frequencies pass the guard, or ``None`` when an exact sample
            was re-selected (the basis must be rebuilt).
        """
        n = len(self.f)
        solved = np.flatnonzero(self.solved)
        passed = np.zeros(n, bool)
        for i in range(n):
            nearest = solved[np.argmin(np.abs(self.f[solved] - self.f[i]))]
            self.floor[i] = max(self.tolerance, _FLOOR * self.residual[nearest])
            if self.solved[i]:
                if i > 0 and passed[i - 1]:
                    cand = self.candidates[i]
                    pos = self._closest(i, cand, i - 1, certain=True)
                    if pos != self.pick[i]:
                        self._keep(i, cand, pos)
                        return None
                self.coords[i] = self.coords_at(i)
            else:
                lam, y, eta = self.reduced.solve(
                    self.k0[i], self.k_max[i], self.coords_at(i - 1)
                )
                self.lam[i], self.coords[i], self.residual[i] = lam, y, eta
            passed[i] = self.residual[i] <= self.floor[i]
        beta = np.sqrt(-self.lam + 0j).real
        gone = passed & ~self.solved & (beta <= _EVANESCENT * self.k_max)
        if gone.any():
            raise self._evanescent_error(int(np.flatnonzero(gone)[0]))
        return passed

    # --- normalization ----------------------------------------------------------

    def normalized(
        self, rule: tuple[np.ndarray, np.ndarray] | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The fixed basis, 1 W coordinates, gamma (1/m) and voltages."""
        space, red = self.space, self.reduced
        m = red.size
        tangential = space.free < space.n_t
        w = np.zeros((len(space.free), 2 * m))
        w[tangential, :m] = red.v[tangential]
        w[~tangential, m:] = red.v[~tangential]
        g_form, t_form = space.power_forms
        g_red, t_red = w.T @ (g_form @ w), w.T @ (t_form @ w)
        mass = w.T @ (space.mass @ w)
        ln, lt = space.gauge_forms
        ln, lt = ln @ w, lt @ w
        basis = np.zeros((space.space.ndof, 2 * m))
        basis[space.free] = w
        volts = None if rule is None else _voltage_functional(space, basis, rule)

        n = len(self.f)
        coords = np.zeros((n, 2 * m), complex)
        voltage = np.full(n, complex(math.nan, math.nan))
        gamma = 1j * np.sqrt(-self.lam + 0j)
        for i in range(n):
            y = self.coords_at(i)
            z = np.concatenate([y / gamma[i], y])
            omega = 2 * math.pi * self.f[i]
            power = (
                0.5
                * space.section.scale
                * 1j
                / (omega * scipy.constants.mu_0)
                * (z @ g_red @ z.conj() + gamma[i].conjugate() * (z @ t_red @ z.conj()))
            )
            z /= math.sqrt(abs(power))
            if i == 0:
                g = -1j * (ln @ z + gamma[i] * (lt @ z))
                c = w @ z
                if abs(g) > 1e-12 * np.abs(c).max():
                    z *= abs(g) / g
                else:
                    k = int(np.argmax(np.abs(c)))
                    z *= abs(c[k]) / c[k]
            else:
                o = coords[i - 1].conj() @ mass @ z
                z *= abs(o) / o
            if volts is not None:
                voltage[i] = volts @ z
                if voltage[i].real < 0:
                    z, voltage[i] = -z, -voltage[i]
            coords[i] = z
        return basis, coords, gamma / space.section.scale, voltage

    # --- errors -------------------------------------------------------------------

    def _band(self) -> str:
        return f"{self.f[0] / 1e9:.6g}-{self.f[-1] / 1e9:.6g} GHz"

    def _cutoff_error(self, first: _Candidates) -> PortModeError:
        """The mode is evanescent at the lowest frequency: locate its cutoff."""
        n, mode = len(self.f), self.mode
        remedy = (
            "Start the band above the cutoff, pick a lower mode, or use a LumpedPort."
        )
        top = self._solve(n - 1) if n > 1 else first
        if n == 1 or len(top.lam) < mode:
            return PortModeError(
                f"WavePort {self.name!r}: mode {mode} does not propagate in the band "
                f"{self._band()} (the cross-section carries {len(first.lam)} "
                f"propagating mode(s) at {self.f[0] / 1e9:.6g} GHz and "
                f"{len(top.lam)} at {self.f[-1] / 1e9:.6g} GHz): the band is below "
                f"the mode's cutoff. {remedy}"
            )
        lo, hi, cand = 0, n - 1, top
        while hi - lo > 1:
            mid = (lo + hi) // 2
            c = self._solve(mid)
            if len(c.lam) >= mode:
                hi, cand = mid, c
            else:
                lo = mid
        cutoff = self._cutoff_estimate(hi, cand, top)
        return PortModeError(
            f"WavePort {self.name!r}: mode {mode} has its cutoff inside the band "
            f"{self._band()}, at about {cutoff / 1e9:.4g} GHz (it is evanescent at "
            f"{self.f[lo] / 1e9:.6g} GHz and propagates at {self.f[hi] / 1e9:.6g} "
            f"GHz). {remedy}"
        )

    def _cutoff_estimate(self, hi: int, cand: _Candidates, top: _Candidates) -> float:
        """Cutoff frequency from ``beta^2`` linear in ``k0^2`` (bracketed)."""
        f_lo, f_hi = self.f[hi - 1], self.f[hi]
        n = len(self.f)
        if hi == n - 1:
            return 0.5 * (f_lo + f_hi)
        b1 = -cand.lam[self.mode - 1].real
        b2 = -top.lam[self.mode - 1].real
        k1, k2 = self.k0[hi] ** 2, self.k0[n - 1] ** 2
        slope = (b2 - b1) / (k2 - k1)
        if slope <= 0:
            return 0.5 * (f_lo + f_hi)
        kc2 = max(k1 - b1 / slope, 0.0)
        fc = math.sqrt(kc2) / self.k0[hi] * self.f[hi]
        return min(max(fc, f_lo), f_hi)

    def _evanescent_error(self, j: int) -> PortModeError:
        return PortModeError(
            f"WavePort {self.name!r}: the tracked mode {self.mode} stops propagating "
            f"at {self.f[j] / 1e9:.6g} GHz inside the band {self._band()} (a cutoff "
            "in the band). Restrict the band to where the mode propagates, or use a "
            "LumpedPort."
        )

    def _check_degenerate(self, first: _Candidates) -> None:
        beta = np.sqrt(-first.lam + 0j)
        k = self.mode - 1
        close = np.abs(beta - beta[k]) <= _DEGENERATE * abs(beta[k])
        close[k] = False
        if close.any():
            raise PortModeError(
                f"WavePort {self.name!r}: mode {self.mode} is degenerate at "
                f"{self.f[0] / 1e9:.6g} GHz (another mode has the same propagation "
                "constant), so the port's field is not defined. Break the "
                "cross-section's symmetry (for example with the mesh or a "
                "WavePort(pec_edges=...) wall)."
            )


class _Reduced:
    """Rayleigh-Ritz model of the affine pencil on a growing real basis."""

    def __init__(self, space: PortModeSpace) -> None:
        self.space = space
        self.v = np.zeros((len(space.free), 0))
        self._pieces = [
            (is_b, k, m)
            for is_b, mats in ((False, space.pencil.a), (True, space.pencil.b))
            for k, m in enumerate(mats)
            if m.count_nonzero()
        ]

    @property
    def size(self) -> int:
        return self.v.shape[1]

    def add(self, x: np.ndarray) -> None:
        """Add the real and imaginary parts of ``x`` (Gram-Schmidt twice)."""
        scale = np.linalg.norm(x)
        for part in (x.real, x.imag):
            w = part.copy()
            for _ in range(2):
                w -= self.v @ (self.v.T @ w)
            norm = np.linalg.norm(w)
            if norm > _NEW_DIRECTION * scale:
                self.v = np.column_stack([self.v, w / norm])
        products = [m @ self.v for _, _, m in self._pieces]
        self._r = np.linalg.qr(np.concatenate(products, axis=1), mode="r")
        self._red = [self.v.T @ p for p in products]
        self._mass = self.v.T @ (self.space.mass @ self.v)

    def project(self, x: np.ndarray) -> np.ndarray:
        return self.v.T @ x

    def lift(self, y: np.ndarray) -> np.ndarray:
        return self.v @ y

    def solve(
        self, k0: float, k_max: float, ref: np.ndarray
    ) -> tuple[complex, np.ndarray, float]:
        """The reduced pair that tracks ``ref`` and its backward error.

        The backward error is infinite when no pair qualifies or two overlap
        ``ref`` almost equally.
        """
        theta = self.space.pencil.theta(k0)
        n = self.size
        ar = np.zeros((n, n), complex)
        br = np.zeros((n, n), complex)
        for (is_b, k, _), red in zip(self._pieces, self._red, strict=True):
            if is_b:
                br += theta[k] * red
            else:
                ar += theta[k] * red
        lam, vecs = scipy.linalg.eig(ar, br)
        beta = np.sqrt(-lam + 0j)
        ok = np.isfinite(lam) & (beta.real <= k_max * (1 + 1e-6))
        if not ok.any():
            return complex(math.nan), np.zeros(n), math.inf
        idx = np.flatnonzero(ok)
        y = vecs[:, idx]
        num = np.abs(ref.conj() @ self._mass @ y)
        den = np.sqrt(
            (ref.conj() @ self._mass @ ref).real
            * np.einsum("ik,ij,jk->k", y.conj(), self._mass, y).real
        )
        overlap = num / den
        best = np.argsort(-overlap, kind="stable")
        i = int(best[0])
        lam_i, y_i = complex(lam[idx[i]]), y[:, i]
        if len(best) > 1 and overlap[best[1]] > overlap[i] - _AMBIGUOUS:
            return lam_i, y_i, math.inf
        return lam_i, y_i, self._backward_error(theta, lam_i, y_i)

    def _backward_error(
        self, theta: tuple[complex, ...], lam: complex, y: np.ndarray
    ) -> float:
        """``|A x - lam B x| / (|A x| + |lam| |B x|)`` of ``x = V y``, exactly."""
        n = self.size
        ax = np.zeros(self._r.shape[0], complex)
        bx = np.zeros(self._r.shape[0], complex)
        for p, (is_b, k, _) in enumerate(self._pieces):
            block = self._r[:, p * n : (p + 1) * n] @ (theta[k] * y)
            if is_b:
                bx += block
            else:
                ax += block
        den = np.linalg.norm(ax) + abs(lam) * np.linalg.norm(bx)
        return float(np.linalg.norm(ax - lam * bx) / den) if den > 0 else math.inf


def _overlaps(mass, x_ref: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """``|<x_ref, v>| / (|x_ref| |v|)`` in the mass inner product, per column."""
    mv = mass @ vecs
    num = np.abs(x_ref.conj() @ mv)
    den = np.sqrt(
        (x_ref.conj() @ (mass @ x_ref)).real * np.sum(vecs.conj() * mv, 0).real
    )
    return num / den


def _wavenumbers(section: PortSection, frequency: float) -> tuple[float, float, float]:
    """``k0``, the largest propagating wavenumber and the shift, per local unit.

    The largest wavenumber is the largest bulk one, times `_SLOW_WAVE` when
    `Conductor` or `Impedance` edges can slow the mode down.
    """
    k0_si = 2 * math.pi * frequency / scipy.constants.c
    k0 = k0_si * section.scale
    n2_shift, n2_max = section.index_squared(float(section.sigma.max()) * _Z0 / k0_si)
    k_max = math.sqrt(n2_max) * k0
    if len(section.impedance_terms):
        k_max *= _SLOW_WAVE
    return k0, k_max, -_TARGET * n2_shift * k0**2


def _voltage_functional(
    space: PortModeSpace, basis: np.ndarray, rule: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """``V = int E . dl`` along the voltage path for each basis column."""
    pts, wts = rule
    e = space.evaluate(basis.T, pts)
    return np.einsum("kpc,pc->k", e, wts)


def _voltage_rule(
    section: PortSection, path: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Plane points and weights (local direction x length, m) of a path.

    ``path`` defaults to the wave port's ``voltage_path``.
    """
    path = np.asarray(section.port.voltage_path if path is None else path, float)
    _local, dist = section.to_local(path)
    if (np.abs(dist) > 1e-5 * section.diameter).any():
        raise ValueError(
            f"WavePort {section.port.name!r}: voltage_path leaves the port plane; "
            "give points (m) on the port surface."
        )
    x, w = _GAUSS
    pts, wts = [], []
    for p, q in itertools.pairwise(path):
        d_local = section.frame @ (q - p)
        for k in range(_PIECES):
            s = (k + (x + 1) / 2) / _PIECES
            pts.append(section.to_local(p + np.outer(s, q - p))[0])
            wts.append(np.outer(w / (2 * _PIECES), d_local))
    return np.concatenate(pts), np.concatenate(wts)


def path_voltage(
    space: PortModeSpace, basis: np.ndarray, coords: np.ndarray, path: np.ndarray
) -> np.ndarray:
    """``V = int E . dl`` along ``path`` (m) of each field ``basis @ coords[i]``."""
    rule = _voltage_rule(space.section, path)
    return coords @ _voltage_functional(space, basis, rule)
