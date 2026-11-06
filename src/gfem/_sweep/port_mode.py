"""Port modes over a frequency grid: the guarded port mode sweep.

A wave port's mode over the whole output grid comes from a small reduced
model instead of one exact solve per frequency:

1. **Exact solves** factor the shifted 2D pencil with SuperLU and run ARPACK
   shift-invert at ``sigma = -1.1 (n_max k0)^2``. At the lowest frequency
   the requested mode is picked in rank order: converged, propagating
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
physical mode through crossings and ignores spurious Ritz values. Several
modes of one port surface share the sweep: every exact solve and the reduced
basis serve all of them, the guard passes a frequency when every mode passes,
and two modes never follow the same candidate. Each mode is normalized to
1 W. Its phase is fixed at the lowest frequency by a quadrant integral of its
tangential magnetic field and is continuous after it (the overlap with the
previous frequency's mode is real and positive); with a voltage path,
``Re V > 0`` at every frequency.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants
import scipy.linalg

from gfem import _mpi
from gfem._boundaries import (
    WavePort,
    lead_port,
    wave_port_label,
    wave_port_surfaces,
)
from gfem._checks import is_traced
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.runtime import task_manager
from gfem._linalg.superlu import backward_error, shift_invert_eigs
from gfem._port_section import PortSection, port_section
from gfem._sign import largest_entry
from gfem.errors import PortModeError, PortModeTrackingWarning
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import PortMode
from gfem.results import PortModeField, PortModeResult

__all__ = [
    "PortModeRule",
    "PortModeSweep",
    "path_voltage",
    "port_mode_sweep",
    "swept_once",
]

# The shift is this factor above the largest bulk (Re eps)(Re mu), as in Palace.
_TARGET = 1.1
# A Ritz pair propagates when Re beta exceeds this fraction of k0 n_max, and
# exceeds |Im beta| (Re beta^2 > 0: above cutoff; below it a lossy mode, or a
# complex pair of a lossless one, keeps a small Re beta).
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
# The fields of a `PortModeSweep` that differ between the modes of one surface.
_PER_MODE = ("gamma", "residual", "floor", "coords", "voltage")
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

    def __init__(
        self,
        model: Model,
        problem: PortMode,
        options: SolverOptions,
        *,
        comm: Any = None,
    ) -> None:
        """Check that the problem fits the model (see `gfem.PortMode`).

        ``comm`` is the resolved communicator of a distributed solve: every result
        is the one sweep rank 0 ran (ADR 0059).

        Raises:
            ValueError: ``problem.port`` is not a wave port of the model, or
                ``options`` asks for a linear solver, an eigen solver or a
                device: the 2D port mode problem always runs SuperLU and
                SciPy on the CPU.
        """
        if not isinstance(problem, PortMode):
            raise TypeError(f"Expected gfem.PortMode, got {problem!r}.")
        chosen = (options.device, options.linear_solver, options.eigen_solver)
        if chosen != ("cpu", "auto", "auto"):
            raise ValueError(
                "The port mode problem always uses SuperLU and SciPy on the CPU "
                f"(a 2D problem), but the options ask for device={options.device!r}, "
                f"linear_solver={options.linear_solver!r}, "
                f"eigen_solver={options.eigen_solver!r}. Solve PortMode with the "
                "default device, linear_solver and eigen_solver."
            )
        self.problem = problem
        self.options = options
        self.comm = comm
        self.port = _wave_port(model, problem.port)
        self.mode = self.port.mode if problem.mode is None else problem.mode
        # its own mode: the surface's joint sweep, with the surface's wave
        # ports in model order, as a driven problem runs it
        self._ports = (
            list(wave_port_surfaces(model.ports)[self.port.surface])
            if problem.mode is None
            else [self.port]
        )
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
        ports = self._ports
        modes = [self.mode] if len(ports) == 1 else [p.mode for p in ports]
        lead = lead_port(ports)
        section = port_section(model, lead)
        with task_manager(self.options.threads, self.comm):
            space = PortModeSpace(section, self.options.order)
            sweeps, _ = swept_once(
                self.comm,
                space,
                self.problem.frequency,
                modes,
                tolerance=self.options.port_mode_tolerance,
                ports=ports,
            )
        sweep = sweeps[ports.index(self.port)]
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
                "computed inside jax.grad, jax.jit or jax.vmap: the mode field "
                "needs concrete model values. Solve PortMode outside the transformed "
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
    modes: list[int],
    *,
    tolerance: float,
    ports: list[WavePort],
) -> list[PortModeSweep]:
    """The guarded port mode sweep of one port surface (see the module docstring).

    Every exact solve serves all requested modes, and the reduced model
    shares one basis; each mode is tracked by its own overlap, and two modes
    never follow the same candidate. One mode is today's single-port sweep.

    Args:
        space: The surface's discrete port mode problem.
        frequency: The output grid, Hz (any order).
        modes: Which modes at the lowest frequency, in rank order (distinct).
        tolerance: Backward error a reduced mode must reach.
        ports: The wave port of each mode (its name for messages, its
            voltage path).

    Returns:
        One `PortModeSweep` per mode, sharing ``space``, ``basis`` and
        ``solved``.

    Raises:
        PortModeError: A mode is below cutoff somewhere in the band, or is
            degenerate, or two candidates cannot be told apart, or two modes
            follow the same candidate.

    Warns:
        PortModeTrackingWarning: A tracked mode stops being its ``mode``-th
            in rank order inside the band (a crossing).
    """
    f_given = np.asarray(frequency, float)
    section = space.section
    rules = [
        None
        if p.voltage_path is None
        else _voltage_rule(section, np.asarray(p.voltage_path, float), p.name)
        for p in ports
    ]
    order = np.argsort(f_given, kind="stable")
    sweep = _Greedy(space, f_given[order], modes, tolerance, ports)
    sweep.run()
    basis, coords, gamma, voltage = sweep.normalized(rules)
    out = np.empty_like(order)
    out[order] = np.arange(len(order))
    return [
        PortModeSweep(
            space=space,
            frequency=f_given,
            gamma=gamma[k, out],
            residual=sweep.residual[k, out],
            solved=sweep.solved[out],
            floor=sweep.floor[k, out],
            basis=basis,
            coords=coords[k, out],
            voltage=voltage[k, out],
        )
        for k in range(len(modes))
    ]


def swept_once(
    comm: Any,
    space: PortModeSpace,
    frequency: tuple[float, ...] | np.ndarray,
    modes: list[int],
    *,
    tolerance: float,
    ports: list[WavePort],
    derived: Callable[[np.ndarray], dict[str, np.ndarray]] | None = None,
) -> tuple[list[PortModeSweep], dict[str, np.ndarray]]:
    """`port_mode_sweep` once, on rank 0 of ``comm``, with its arrays on every rank.

    A `PortModeError` and the sweep's warnings reach every rank (ADR 0059), and each
    rank's sweeps stand on its own ``space``. Collective, on every rank of ``comm``.

    Args:
        comm: The resolved communicator.
        space: The surface's discrete port mode problem, built on every rank.
        frequency: The output grid, Hz.
        modes: As `port_mode_sweep`.
        tolerance: As `port_mode_sweep`.
        ports: As `port_mode_sweep`.
        derived: More arrays read off the sweep's basis, computed beside it and
            sent with it, since an assembled form may round differently per rank.

    Returns:
        The sweeps and ``derived``'s arrays.
    """

    def compute() -> dict[str, np.ndarray]:
        sweeps = port_mode_sweep(
            space, frequency, modes, tolerance=tolerance, ports=ports
        )
        first = sweeps[0]
        arrays = {
            name: np.stack([getattr(sweep, name) for sweep in sweeps])
            for name in _PER_MODE
        }
        arrays |= {"basis": first.basis, "solved": first.solved}
        return arrays | (derived(first.basis) if derived is not None else {})

    # ValueError: a voltage path off the port, on replicated inputs
    arrays = _mpi.on_root(comm, compute, results=(PortModeError, ValueError))
    sweeps = [
        PortModeSweep(
            space=space,
            frequency=np.asarray(frequency, float),
            basis=arrays["basis"],
            solved=arrays["solved"],
            **{name: arrays[name][k] for name in _PER_MODE},
        )
        for k in range(len(modes))
    ]
    return sweeps, arrays


@dataclasses.dataclass(frozen=True)
class _Candidates:
    """The converged propagating eigenpairs of one exact solve, in rank order."""

    lam: np.ndarray
    vecs: np.ndarray
    residual: np.ndarray


class _Greedy:
    """State of one port mode sweep over a sorted grid.

    Tracking is certified link by link: a frequency's modes are trusted only
    when chosen by overlap with accurate modes at the previous grid
    frequency. An exact solve whose reference was further away (the top of
    the band at the start, or a frequency after a failing stretch) is
    re-checked once its neighbour below passes, and re-selected if another
    of its candidates continues a mode better.

    Per mode ``k`` (row of ``lam``, ``residual``, ``floor``; entry of
    ``coords`` and of each ``pick``); ``solved`` is shared by the modes.
    """

    def __init__(
        self,
        space: PortModeSpace,
        f: np.ndarray,
        modes: list[int],
        tolerance: float,
        ports: list[WavePort],
    ) -> None:
        self.space = space
        self.f = f
        self.modes = list(modes)
        self.ports = list(ports)
        self.tolerance = tolerance
        n, n_modes = len(f), len(self.modes)
        self.k0 = np.empty(n)
        self.k_max = np.empty(n)
        self.shift = np.empty(n)
        for i, fi in enumerate(f):
            self.k0[i], self.k_max[i], self.shift[i] = _wavenumbers(space.section, fi)
        self.reduced = _Reduced(space)
        self.lam = np.zeros((n_modes, n), complex)
        self.coords: list[list[np.ndarray]] = [[np.zeros(0)] * n for _ in self.modes]
        self.residual = np.full((n_modes, n), np.inf)
        self.solved = np.zeros(n, bool)
        self.floor = np.full((n_modes, n), tolerance)
        # exact solves: candidates and the positions (in Palace's order) kept
        self.candidates: dict[int, _Candidates] = {}
        self.pick: dict[int, list[int]] = {}

    # --- driver ---------------------------------------------------------------

    def run(self) -> None:
        n = len(self.f)
        first = self._solve(0)
        if len(first.lam) < max(self.modes):
            raise self._cutoff_error(first)
        for k in range(len(self.modes)):
            self._check_degenerate(first, k)
        self._keep(0, first, [mode - 1 for mode in self.modes])
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
                for k in range(len(self.modes)):
                    self._warn_crossing(k)
                return
            # the worst mode at each frequency decides where to enrich
            excess = (self.residual / self.floor).max(axis=0)
            j = int(np.argmax(np.where(passed, -np.inf, excess)))
            if passed[j - 1]:
                self._exact(j, j - 1, certain=True)
            else:
                good = np.flatnonzero(passed)
                self._exact(j, int(good[np.argmin(np.abs(good - j))]), certain=False)
        raise self._unsettled_error()  # pragma: no cover - a safeguard

    # --- exact solves -----------------------------------------------------------

    def _solve(self, j: int) -> _Candidates:
        space, mode = self.space, max(self.modes)
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
                _propagates(beta, k_max)
                & (beta.real <= k_max * (1 + 1e-6))
                & (res < _UNCONVERGED)
            )
            # Palace's order: largest Re of the shift-inverted eigenvalue
            rank = (1 / (lam - self.shift[j])).real
            ranked = np.flatnonzero(ok)[np.argsort(-rank[ok], kind="stable")]
            if len(ranked) >= mode or want >= n - 2:
                break
        return _Candidates(lam=lam[ranked], vecs=vecs[:, ranked], residual=res[ranked])

    def _keep(self, j: int, cand: _Candidates, picks: list[int]) -> None:
        """Make candidate ``picks[k]`` of the exact solve at ``j`` mode ``k`` there."""
        new = j not in self.candidates
        self.candidates[j] = cand
        self.pick[j] = list(picks)
        self.solved[j] = True
        self.lam[:, j] = cand.lam[picks]
        self.residual[:, j] = cand.residual[picks]
        if new:
            self.reduced.add(cand.vecs[:, picks])

    def _exact(self, j: int, ref: int, *, certain: bool) -> None:
        """Exact solve at ``j``, keeping the candidates closest to the modes at ``ref``.

        Args:
            j: Grid index.
            ref: Grid index of the reference modes.
            certain: The reference is accurate modes at the grid neighbour,
                so an ambiguous choice is an error.
        """
        cand = self._solve(j)
        if len(cand.lam) < len(self.modes):
            # too few candidates for distinct modes
            raise self._evanescent_error(j, self._lost(cand, ref, certain=certain))
        self._keep(j, cand, self._closest(j, cand, ref, certain=certain))

    def _lost(self, cand: _Candidates, ref: int, *, certain: bool) -> list[int]:
        """The modes left without a candidate when ``cand`` has too few.

        Each candidate goes to the mode at ``ref`` it overlaps most, the
        largest overlaps first, as the tracker matches them. Against an
        uncertain reference with every mode still at its rank, the
        highest-ranked modes are the ones lost.
        """
        n = len(self.modes)
        ranks = [mode - 1 for mode in self.modes]
        if not certain and all(pick == ranks for pick in self.pick.values()):
            by_rank = np.argsort(self.modes, kind="stable")
            return sorted(int(k) for k in by_rank[len(cand.lam) :])
        overlap = np.array(
            [
                _overlaps(
                    self.space.mass,
                    self.reduced.lift(self.coords_at(ref, k)),
                    cand.vecs,
                )
                for k in range(n)
            ]
        ).reshape(n, len(cand.lam))
        lost = set(range(n))
        for _ in range(len(cand.lam)):
            k, c = np.unravel_index(int(np.argmax(overlap)), overlap.shape)
            lost.discard(int(k))
            overlap[k, :] = -np.inf
            overlap[:, c] = -np.inf
        return sorted(lost)

    def _closest(
        self, j: int, cand: _Candidates, ref: int, *, certain: bool
    ) -> list[int]:
        """Each mode's candidate at ``j``: the closest to that mode at ``ref``.

        With an uncertain reference two modes may provisionally share a
        candidate; the certain re-check in `_tabulate` raises if they still do.
        """
        picks = []
        for k, mode in enumerate(self.modes):
            x_ref = self.reduced.lift(self.coords_at(ref, k))
            overlap = _overlaps(self.space.mass, x_ref, cand.vecs)
            best = np.argsort(-overlap, kind="stable")
            if (
                certain
                and len(best) > 1
                and overlap[best[1]] > overlap[best[0]] - _AMBIGUOUS
            ):
                raise PortModeError(
                    f"{self._who(k)}: two modes overlap the tracked mode "
                    f"{mode} almost equally at {self.f[j] / 1e9:.6g} GHz (overlaps "
                    f"{overlap[best[0]]:.3f} and {overlap[best[1]]:.3f}): a degenerate "
                    "pair or an unresolved crossing. Break the cross-section's "
                    "symmetry, or make the unwanted mode non-propagating (for "
                    "example with WavePort(pec_edges=...))."
                )
            picks.append(int(best[0]))
        if certain:
            for k, other in itertools.combinations(range(len(picks)), 2):
                if picks[k] == picks[other]:
                    raise PortModeError(
                        f"WavePorts {self.ports[k].name!r} and "
                        f"{self.ports[other].name!r} on surface "
                        f"{self.ports[k].surface!r} follow the same mode at "
                        f"{self.f[j] / 1e9:.6g} GHz (an unresolved crossing between "
                        "two requested modes). Narrow the band or break the "
                        "cross-section's symmetry."
                    )
        return picks

    def _rebuild(self) -> None:
        """A fresh basis from the exact modes kept now."""
        self.reduced = _Reduced(self.space)
        for j in sorted(self.candidates):
            self.reduced.add(self.candidates[j].vecs[:, self.pick[j]])

    def _warn_crossing(self, k: int) -> None:
        mode = self.modes[k]
        for j in sorted(self.pick):
            if self.pick[j][k] != mode - 1:
                warnings.warn(
                    f"WavePort {self.ports[k].name!r}: the tracked mode {mode} is "
                    f"mode {self.pick[j][k] + 1} in rank order at "
                    f"{self.f[j] / 1e9:.6g} GHz (a mode crossing inside the band). "
                    "gfem keeps following the physical mode; a solver that ranks "
                    "modes at each frequency would switch modes there, so its "
                    "S-parameters differ beyond the crossing.",
                    PortModeTrackingWarning,
                    stacklevel=9,
                )
                return

    def coords_at(self, j: int, k: int) -> np.ndarray:
        """Reduced coordinates of mode ``k`` tabulated at ``j`` (current basis)."""
        if self.solved[j]:
            return self.reduced.project(self.candidates[j].vecs[:, self.pick[j][k]])
        y = self.coords[k][j]
        return np.concatenate([y, np.zeros(self.reduced.size - len(y))])

    # --- reduced solves -----------------------------------------------------------

    def _tabulate(self) -> np.ndarray | None:
        """Walk up the grid, filling unsolved frequencies from the reduced model.

        Each frequency tracks the modes at the previous one; it passes when
        every mode passes.

        Returns:
            Which frequencies pass the guard, or ``None`` when an exact sample
            was re-selected (the basis must be rebuilt).
        """
        n, n_modes = len(self.f), len(self.modes)
        solved = np.flatnonzero(self.solved)
        passed = np.zeros(n, bool)
        for i in range(n):
            nearest = solved[np.argmin(np.abs(self.f[solved] - self.f[i]))]
            self.floor[:, i] = np.maximum(
                self.tolerance, _FLOOR * self.residual[:, nearest]
            )
            if self.solved[i]:
                if i > 0 and passed[i - 1]:
                    cand = self.candidates[i]
                    picks = self._closest(i, cand, i - 1, certain=True)
                    if picks != self.pick[i]:
                        self._keep(i, cand, picks)
                        return None
                for k in range(n_modes):
                    self.coords[k][i] = self.coords_at(i, k)
            else:
                refs = [self.coords_at(i - 1, k) for k in range(n_modes)]
                pairs = self.reduced.solve(self.k0[i], self.k_max[i], refs)
                for k, (lam, y, eta) in enumerate(pairs):
                    self.lam[k, i], self.coords[k][i], self.residual[k, i] = lam, y, eta
            passed[i] = (self.residual[:, i] <= self.floor[:, i]).all()
        beta = np.sqrt(-self.lam + 0j)
        gone = passed & ~self.solved & ~_propagates(beta, self.k_max)
        if gone.any():
            i = int(np.flatnonzero(gone.any(axis=0))[0])
            raise self._evanescent_error(i, [int(np.flatnonzero(gone[:, i])[0])])
        return passed

    # --- normalization ----------------------------------------------------------

    def normalized(
        self, rules: list[tuple[np.ndarray, np.ndarray] | None]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The fixed basis, then per mode 1 W coordinates, gamma (1/m) and voltages.

        Args:
            rules: Each mode's voltage path rule, or ``None`` without a path.

        Returns:
            ``basis (ndof, 2m)``, ``coords (n_modes, n_f, 2m)``, ``gamma
            (n_modes, n_f)`` and ``voltage (n_modes, n_f)``.
        """
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

        n, n_modes = len(self.f), len(self.modes)
        coords = np.zeros((n_modes, n, 2 * m), complex)
        voltage = np.full((n_modes, n), complex(math.nan, math.nan))
        gamma = 1j * np.sqrt(-self.lam + 0j)
        for k, rule in enumerate(rules):
            volts = None if rule is None else _voltage_functional(space, basis, rule)
            for i in range(n):
                y = self.coords_at(i, k)
                z = np.concatenate([y / gamma[k, i], y])
                omega = 2 * math.pi * self.f[i]
                power = (
                    0.5
                    * space.section.scale
                    * 1j
                    / (omega * scipy.constants.mu_0)
                    * (
                        z @ g_red @ z.conj()
                        + gamma[k, i].conjugate() * (z @ t_red @ z.conj())
                    )
                )
                z /= math.sqrt(abs(power))
                if i == 0:
                    g = -1j * (ln @ z + gamma[k, i] * (lt @ z))
                    c = w @ z
                    if abs(g) > 1e-12 * np.abs(c).max():
                        z *= abs(g) / g
                    else:
                        e = largest_entry(c)
                        z *= abs(c[e]) / c[e]
                else:
                    o = coords[k, i - 1].conj() @ mass @ z
                    z *= abs(o) / o
                if volts is not None:
                    voltage[k, i] = volts @ z
                    if voltage[k, i].real < 0:
                        z, voltage[k, i] = -z, -voltage[k, i]
                coords[k, i] = z
        return basis, coords, gamma / space.section.scale, voltage

    # --- errors -------------------------------------------------------------------

    def _band(self) -> str:
        return f"{self.f[0] / 1e9:.6g}-{self.f[-1] / 1e9:.6g} GHz"

    def _who(self, k: int) -> str:
        """Mode ``k``'s wave port, with its surface when that is shared."""
        port = self.ports[k]
        others = [p.name for p in self.ports if p is not port]
        if not others:
            return f"WavePort {port.name!r}"
        shared = ", ".join(repr(name) for name in others)
        return (
            f"WavePort {port.name!r} on surface {port.surface!r} (shared with {shared})"
        )

    def _remedy(self, first: str) -> str:
        """``first``, then the LumpedPort remedy (and dropping a shared port)."""
        if len(self.ports) == 1:
            return f"{first}, or use a LumpedPort."
        return f"{first}, use a LumpedPort, or drop that wave port."

    def _unsettled_error(self) -> PortModeError:  # pragma: no cover - a safeguard
        if len(self.modes) == 1:
            which = f"mode continues mode {self.modes[0]}"
        else:
            *rest, last = (str(mode) for mode in self.modes)
            which = f"modes continue modes {', '.join(rest)} and {last}"
        return PortModeError(
            f"{wave_port_label(self.ports)}: the port mode sweep could not settle "
            f"which {which} over {self._band()}. Narrow the band or use a "
            "LumpedPort."
        )

    def _cutoff_error(self, first: _Candidates) -> PortModeError:
        """A mode is evanescent at the lowest frequency: locate its cutoff.

        The lowest-ranked failing mode is the one reported.
        """
        k = min(
            (k for k, mode in enumerate(self.modes) if mode > len(first.lam)),
            key=lambda k: self.modes[k],
        )
        n, mode, who = len(self.f), self.modes[k], self._who(k)
        remedy = self._remedy("Start the band above the cutoff, pick a lower mode")
        top = self._solve(n - 1) if n > 1 else first
        if n == 1 or len(top.lam) < mode:
            return PortModeError(
                f"{who}: mode {mode} does not propagate in the band "
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
        cutoff = self._cutoff_estimate(hi, cand, top, mode)
        return PortModeError(
            f"{who}: mode {mode} has its cutoff inside the band "
            f"{self._band()}, at about {cutoff / 1e9:.4g} GHz (it is evanescent at "
            f"{self.f[lo] / 1e9:.6g} GHz and propagates at {self.f[hi] / 1e9:.6g} "
            f"GHz). {remedy}"
        )

    def _cutoff_estimate(
        self, hi: int, cand: _Candidates, top: _Candidates, mode: int
    ) -> float:
        """Cutoff frequency from ``beta^2`` linear in ``k0^2`` (bracketed)."""
        f_lo, f_hi = self.f[hi - 1], self.f[hi]
        n = len(self.f)
        if hi == n - 1:
            return 0.5 * (f_lo + f_hi)
        b1 = -cand.lam[mode - 1].real
        b2 = -top.lam[mode - 1].real
        k1, k2 = self.k0[hi] ** 2, self.k0[n - 1] ** 2
        slope = (b2 - b1) / (k2 - k1)
        if slope <= 0:
            return 0.5 * (f_lo + f_hi)
        kc2 = max(k1 - b1 / slope, 0.0)
        fc = math.sqrt(kc2) / self.k0[hi] * self.f[hi]
        return min(max(fc, f_lo), f_hi)

    def _evanescent_error(self, j: int, lost: list[int]) -> PortModeError:
        """Modes ``lost`` (indices into ``modes``) stop propagating at ``j``."""
        if len(lost) == 1:
            (k,) = lost
            remedy = self._remedy("Restrict the band to where the mode propagates")
            return PortModeError(
                f"{self._who(k)}: the tracked mode {self.modes[k]} stops "
                f"propagating at {self.f[j] / 1e9:.6g} GHz inside the band "
                f"{self._band()} (a cutoff in the band). {remedy}"
            )
        *rest, last = (str(self.modes[k]) for k in lost)
        label = wave_port_label([self.ports[k] for k in lost], lone_surface=True)
        remedy = self._remedy("Restrict the band to where the modes propagate")
        return PortModeError(
            f"{label}: the tracked modes {', '.join(rest)} and {last} stop "
            f"propagating at {self.f[j] / 1e9:.6g} GHz inside the band "
            f"{self._band()} (a cutoff in the band). {remedy}"
        )

    def _check_degenerate(self, first: _Candidates, k: int) -> None:
        """Mode ``k`` against every candidate at the lowest frequency."""
        beta = np.sqrt(-first.lam + 0j)
        pos = self.modes[k] - 1
        close = np.abs(beta - beta[pos]) <= _DEGENERATE * abs(beta[pos])
        close[pos] = False
        if not close.any():
            return
        partners = [i for i, mode in enumerate(self.modes) if close[mode - 1]]
        if partners:
            port, other = self.ports[k], self.ports[partners[0]]
            raise PortModeError(
                f"WavePorts {port.name!r} and {other.name!r} on surface "
                f"{port.surface!r} request modes {self.modes[k]} and "
                f"{self.modes[partners[0]]}, which are degenerate at "
                f"{self.f[0] / 1e9:.6g} GHz (the same propagation constant), so the "
                "ports' fields are not defined. Break the cross-section's symmetry "
                "(for example with the mesh or a WavePort(pec_edges=...) wall)."
            )
        raise PortModeError(
            f"{self._who(k)}: mode {self.modes[k]} is degenerate at "
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

    def add(self, xs: np.ndarray) -> None:
        """Add the real and imaginary parts of each column of ``xs``, in order.

        Gram-Schmidt twice per part.
        """
        for x in xs.T:
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
        self, k0: float, k_max: float, refs: list[np.ndarray]
    ) -> list[tuple[complex, np.ndarray, float]]:
        """The reduced pair that tracks each of ``refs``, and its backward error.

        A backward error is infinite when no pair qualifies, two overlap its
        reference almost equally, or two references pick the same pair.
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
            return [(complex(math.nan), np.zeros(n), math.inf) for _ in refs]
        idx = np.flatnonzero(ok)
        y = vecs[:, idx]
        norms = np.einsum("ik,ij,jk->k", y.conj(), self._mass, y).real
        picks, clear = [], []
        for ref in refs:
            num = np.abs(ref.conj() @ self._mass @ y)
            overlap = num / np.sqrt((ref.conj() @ self._mass @ ref).real * norms)
            best = np.argsort(-overlap, kind="stable")
            i = int(best[0])
            picks.append(i)
            clear.append(len(best) == 1 or overlap[best[1]] <= overlap[i] - _AMBIGUOUS)
        out = []
        for i, ok_i in zip(picks, clear, strict=True):
            lam_i, y_i = complex(lam[idx[i]]), y[:, i]
            if ok_i and picks.count(i) == 1:
                out.append((lam_i, y_i, self._backward_error(theta, lam_i, y_i)))
            else:  # ambiguous, or two modes on one pair
                out.append((lam_i, y_i, math.inf))
        return out

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


def _propagates(beta: np.ndarray, k_max: float | np.ndarray) -> np.ndarray:
    """Which propagation constants are of propagating modes (see `_EVANESCENT`)."""
    return (beta.real > _EVANESCENT * k_max) & (beta.real > np.abs(beta.imag))


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
    n2_shift, n2_max = section.index_squared(section.max_conductivity * _Z0 / k0_si)
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
    section: PortSection, path: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Plane points and weights (local direction x length, m) of a path.

    ``name`` is the wave port the path belongs to, for messages.
    """
    path = np.asarray(path, float)
    _local, dist = section.to_local(path)
    if (np.abs(dist) > 1e-5 * section.diameter).any():
        raise ValueError(
            f"WavePort {name!r}: voltage_path leaves the port plane; "
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
    space: PortModeSpace,
    basis: np.ndarray,
    coords: np.ndarray,
    path: np.ndarray,
    name: str,
) -> np.ndarray:
    """``V = int E . dl`` along ``path`` (m) of each field ``basis @ coords[i]``.

    ``name`` is the wave port the field belongs to, for messages.
    """
    rule = _voltage_rule(space.section, path, name)
    return coords @ _voltage_functional(space, basis, rule)
