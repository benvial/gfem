"""Problem types: what to solve on a model, as frozen values.

A problem names the physics analysis and its physical inputs (terminals,
frequencies, ...). The model holds everything about the device; the solver
options hold the numerics. ``gfem.solve(model, problem)`` dispatches on the
problem's type.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import jax
import numpy as np

from gfem._checks import is_traced

__all__ = ["Driven", "Eigenmode", "Electrostatic", "PortMode"]


def _surfaces(value: Any, what: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)) and all(isinstance(s, str) for s in value):
        return tuple(value)
    raise TypeError(
        f"{what} must be a surface name or a list of surface names, got {value!r}."
    )


@dataclasses.dataclass(frozen=True, init=False)
class Electrostatic:
    """The Maxwell capacitance matrix between named terminals.

    For each terminal in turn, the potential is 1 V on that terminal and
    0 V on every other terminal and on ground, and the charge-free Laplace
    equation div(eps grad V) = 0 is solved in the electromagnetic solve
    domain (the model's excluded volumes are left out). Surfaces that are
    neither terminal nor ground carry no surface charge (zero normal
    displacement), except that every surface of a `PEC` or `Conductor`
    boundary of the model that is not a terminal is grounded, as conductors
    are. Ports and other boundaries are charge-free.

    Only the real part of the permittivity is read: losses
    (``tan_delta``, ``conductivity``, a negative imaginary ``eps_r``) and
    ``mu_r`` do not enter the electrostatic problem.

    Args:
        terminals: Terminal name -> its surface name or names (a conductor's
            shell), in capacitance-matrix order. A list of surface names
            makes one terminal per surface, named after it.
        ground: The surface name or names held at 0 V besides the model's
            conductors. Empty: only the model's `PEC` / `Conductor`
            surfaces, or no ground at all (then each row of the matrix sums
            to zero).

    Raises:
        TypeError: A name is not a string.
        ValueError: No terminal, or a surface in two terminals or in a
            terminal and the ground.

    Example:
        ``gfem.Electrostatic(terminals={"trace": ["m1_xy", "m1_z"]},
        ground="ground")``
    """

    terminals: tuple[tuple[str, tuple[str, ...]], ...]
    ground: tuple[str, ...]
    _rule: ClassVar[str] = "gfem._engine.electrostatic:ElectrostaticRule"

    def __init__(
        self,
        terminals: Mapping[str, str | list[str] | tuple[str, ...]] | list[str],
        ground: str | list[str] | tuple[str, ...] = (),
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        if isinstance(terminals, Mapping):
            items = []
            for name, surfaces in terminals.items():
                if not isinstance(name, str):
                    raise TypeError(f"Terminal names must be strings, got {name!r}.")
                items.append((name, _surfaces(surfaces, f"terminals[{name!r}]")))
        else:
            items = [(s, (s,)) for s in _surfaces(terminals, "terminals")]
        if not items:
            raise ValueError(
                "Electrostatic needs at least one terminal, e.g. "
                "terminals={'trace': 'trace_surface'}."
            )
        ground = _surfaces(ground, "ground") if ground else ()
        owner: dict[str, str] = {}
        for name, surfaces in items:
            if len(set(surfaces)) != len(surfaces):
                raise ValueError(f"Terminal {name!r} lists a surface twice.")
            for s in surfaces:
                if s in owner:
                    raise ValueError(
                        f"Surface {s!r} is in terminals {owner[s]!r} and {name!r}; "
                        "a surface belongs to one terminal."
                    )
                owner[s] = name
        for s in ground:
            if s in owner:
                raise ValueError(
                    f"Surface {s!r} is both in terminal {owner[s]!r} and in the "
                    "ground; remove it from one of them."
                )
        object.__setattr__(self, "terminals", tuple(items))
        object.__setattr__(self, "ground", tuple(dict.fromkeys(ground)))

    @property
    def terminal_names(self) -> tuple[str, ...]:
        """Terminal names, in capacitance-matrix order."""
        return tuple(name for name, _ in self.terminals)

    def __repr__(self) -> str:
        terms = ", ".join(f"{n!r}: {list(s)}" for n, s in self.terminals)
        ground = f", ground={list(self.ground)}" if self.ground else ""
        return f"Electrostatic(terminals={{{terms}}}{ground})"


jax.tree_util.register_static(Electrostatic)


def _frequencies(value: Any, owner: str = "PortMode") -> tuple[float, ...]:
    if is_traced(value):
        raise ValueError(f"{owner}.frequency must be fixed numbers, not traced.")
    try:
        f = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        f = np.empty((0, 0))
    if f.ndim > 1 or f.size == 0:
        raise ValueError(
            f"{owner}.frequency must be one frequency or a 1D sequence of them, in "
            f"Hz; got {value!r}."
        )
    f = f.reshape(-1)
    if not np.isfinite(f).all() or (f <= 0).any():
        raise ValueError(f"{owner}.frequency must be finite and > 0 Hz, got {value!r}.")
    return tuple(float(x) for x in f)


@dataclasses.dataclass(frozen=True, init=False)
class PortMode:
    """The 2D port mode problem of a wave port over a frequency grid.

    Solves the port's cross-section for its propagating mode at each
    frequency: propagation constant, effective index, mode field and the
    power-voltage impedance. The cross-section's materials are those of the
    volume elements behind the port; its edges take the boundary conditions
    of the surfaces that meet the port (`PEC` edges are perfect conductors,
    `Absorbing` edges first-order absorbing, all others natural, i.e. PMC).

    A grid runs the port mode sweep that driven problems use for their wave
    ports: a few exact port mode solves (typically 5-12 per band) and a
    reduced port mode model checked at every frequency against
    ``SolverOptions.port_mode_tolerance``. With up to three frequencies each
    is solved exactly. The mode is picked at the lowest frequency and
    followed through the band by its overlap with the neighbouring
    frequency's mode, so it stays the same physical mode through crossings.

    Args:
        port: The name of a `WavePort` of the model.
        frequency: One frequency or a 1D sequence of them, Hz.
        mode: Which propagating mode at the lowest frequency, in Palace's
            order: by decreasing effective index for lossless modes (1 = the
            most confined), with strongly attenuated modes ranked lower.
            ``None``: the wave port's own ``mode``.

    Raises:
        TypeError: ``port`` is not a name.
        ValueError: A frequency is not finite and positive, or ``mode`` is
            not an integer >= 1.

    `gfem.solve` raises `PortModeError` when the mode is below cutoff
    somewhere in the band (naming the estimated cutoff) or degenerate, and
    warns with `PortModeTrackingWarning` when it crosses another mode.

    Example:
        ``gfem.PortMode("p1", np.linspace(1e9, 100e9, 100))``
    """

    port: str
    frequency: tuple[float, ...]
    mode: int | None
    _rule: ClassVar[str] = "gfem._sweep.port_mode:PortModeRule"

    def __init__(self, port: str, frequency: Any, mode: int | None = None) -> None:
        """Validate and normalize (see the class docstring)."""
        if not isinstance(port, str):
            raise TypeError(
                f"PortMode.port must be the name of a WavePort, got {port!r}."
            )
        if mode is not None and (
            isinstance(mode, bool)
            or not isinstance(mode, (int, np.integer))
            or mode < 1
        ):
            raise ValueError(
                f"PortMode.mode must be an integer >= 1 or None, got {mode!r}."
            )
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "frequency", _frequencies(frequency))
        object.__setattr__(self, "mode", None if mode is None else int(mode))


jax.tree_util.register_static(PortMode)


_SWEEPS = ("uniform", "adaptive")
#: Frequency grids up to this size are listed in full in error messages.
_LISTED = 8


def _port_names(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = (value,) if isinstance(value, str) else tuple(value)
    if not all(isinstance(n, str) for n in names):
        raise TypeError(f"Driven.excite takes port names (strings), got {value!r}.")
    if not names:
        raise ValueError(
            "Driven.excite needs at least one port name; None excites every port."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Driven.excite lists a port twice: {value!r}.")
    return names


@dataclasses.dataclass(frozen=True, init=False)
class Driven:
    """The frequency-domain response of a model to its ports.

    At each frequency, each excited port in turn is fed 1 W time-averaged
    incident power while every other port is terminated in its own
    impedance, and the electric field solves the time-harmonic Maxwell
    equations (e^{+iωt}) in the electromagnetic solve domain. The result is
    the scattering matrix (power waves, each port's reference impedance its
    resistance; see `gfem.NetworkParameters`) and, on request, the electric
    fields (see `gfem.conventions` for their amplitude).

    Args:
        frequency: One frequency or a 1D sequence of them, Hz.
        excite: The port name or names fed in turn, one column of S each, in
            this order. ``None``: every port of the model, in model order.
        sweep: ``"uniform"`` solves every frequency in full.
            ``"adaptive"`` (a reduced model sampled at a few frequencies) is
            not available in this version.
        keep_fields: Which electric fields the result keeps, one per kept
            frequency and excited port: ``False`` (the default) none,
            ``True`` all, or a frequency or list of frequencies of
            ``frequency``. A field is a complex vector per DOF (16 bytes per
            DOF; 160 MB at 10M DOFs), so keep only what you use.

    Raises:
        TypeError: ``excite`` is not port names, or ``keep_fields`` is not a
            bool or frequencies.
        ValueError: A frequency is not finite and positive, ``excite`` is
            empty or repeats a port, ``sweep`` is unknown, or a
            ``keep_fields`` frequency is not one of ``frequency``.

    Example:
        ``gfem.Driven(frequency=np.linspace(1e9, 100e9, 100))``
    """

    frequency: tuple[float, ...]
    excite: tuple[str, ...] | None
    sweep: str
    keep_fields: bool | tuple[float, ...]
    _rule: ClassVar[str] = "gfem._engine.driven:DrivenRule"

    def __init__(
        self,
        frequency: Any,
        *,
        excite: str | Sequence[str] | None = None,
        sweep: str = "uniform",
        keep_fields: bool | float | Sequence[float] = False,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        if sweep not in _SWEEPS:
            raise ValueError(
                f"Driven.sweep must be 'uniform' or 'adaptive', got {sweep!r}."
            )
        frequencies = _frequencies(frequency, "Driven")
        object.__setattr__(self, "frequency", frequencies)
        object.__setattr__(self, "excite", _port_names(excite))
        object.__setattr__(self, "sweep", sweep)
        object.__setattr__(self, "keep_fields", _kept(keep_fields, frequencies))

    @property
    def kept_frequencies(self) -> tuple[float, ...]:
        """The frequencies whose fields are kept, in ``frequency`` order."""
        if self.keep_fields is True:
            return self.frequency
        if self.keep_fields is False:
            return ()
        return self.keep_fields


def _kept(value: Any, frequencies: tuple[float, ...]) -> bool | tuple[float, ...]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if is_traced(value):
        raise ValueError("Driven.keep_fields must be fixed frequencies, not traced.")
    try:
        wanted = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise TypeError(
            f"Driven.keep_fields takes True, False or frequencies in Hz, got {value!r}."
        ) from None
    grid = np.asarray(frequencies)
    kept = set()
    for f in wanted:
        match = np.flatnonzero(np.isclose(grid, f, rtol=1e-9, atol=0))
        if not match.size:
            listed = (
                ", ".join(f"{g:g}" for g in grid)
                if len(grid) <= _LISTED
                else f"{len(grid)} frequencies, {grid.min():g} to {grid.max():g}"
            )
            raise ValueError(
                f"Driven.keep_fields frequency {f:g} Hz is not one of the "
                f"problem's frequencies ({listed} Hz)."
            )
        kept.add(int(match[0]))
    return tuple(frequencies[i] for i in sorted(kept))


jax.tree_util.register_static(Driven)


@dataclasses.dataclass(frozen=True, init=False)
class Eigenmode:
    """The resonant modes of a model nearest a target frequency.

    Finds the source-free solutions of the time-harmonic Maxwell equations
    (e^{+iωt}) in the electromagnetic solve domain: complex angular
    frequencies ``omega`` (``Im omega > 0`` for a decaying mode) and their
    electric fields. Losses (``tan_delta``, ``conductivity``), resistive
    lumped ports and `Impedance` boundaries, and first-order `Absorbing`
    boundaries make the modes decay; each mode's Q is
    ``|omega| / (2 Im omega)``. The `n_modes` modes whose frequency
    ``Re omega / 2 pi`` is nearest ``target`` are returned, nearest first.

    Lumped ports are terminations here: nothing is excited. A port with a
    resistance gets a coupling Q, a port with an inductance a participation
    ratio (see `gfem.EigenmodeResult`).

    Args:
        target: The frequency to search around, Hz.
        n_modes: How many modes.

    Raises:
        ValueError: ``target`` is not a finite positive number, or
            ``n_modes`` is not an integer >= 1.

    `gfem.solve` raises `gfem.EigenmodeConvergenceError` when fewer than
    ``n_modes`` modes converge (never a partial result), and
    NotImplementedError for a `WavePort`, a `Conductor` or a second-order
    `Absorbing` boundary, whose frequency dependence this version cannot
    solve for.

    Example:
        ``gfem.Eigenmode(target=5e9, n_modes=2)``
    """

    target: float
    n_modes: int
    _rule: ClassVar[str] = "gfem._eigen.rule:EigenmodeRule"

    def __init__(self, target: float, n_modes: int = 1) -> None:
        """Validate and normalize (see the class docstring)."""
        if is_traced(target):
            raise ValueError("Eigenmode.target must be a fixed number, not traced.")
        try:
            value = float(target)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"Eigenmode.target must be a finite frequency > 0 Hz, got {target!r}."
            )
        if (
            isinstance(n_modes, bool)
            or not isinstance(n_modes, (int, np.integer))
            or n_modes < 1
        ):
            raise ValueError(
                f"Eigenmode.n_modes must be an integer >= 1, got {n_modes!r}."
            )
        object.__setattr__(self, "target", value)
        object.__setattr__(self, "n_modes", int(n_modes))


jax.tree_util.register_static(Eigenmode)
