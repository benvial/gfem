"""Problem types: what to solve on a model, as frozen values.

A problem names the physics analysis and its physical inputs (terminals,
frequencies, ...). The model holds everything about the device; the solver
options hold the numerics. ``gfem.solve(model, problem)`` dispatches on the
problem's type.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, get_args

import jax
import numpy as np

from gfem._boundaries import LumpedPort, _temperature
from gfem._checks import check_fixed, static_float
from gfem._pytree import register_fields
from gfem._sources import (
    DIPOLES,
    INCIDENT,
    ElectricDipole,
    GaussianBeam,
    IncidentField,
    MagneticDipole,
    PlaneWave,
    Source,
)
from gfem._text import hertz_band
from gfem.thermal import HeatSource, Trace, _check_finite

__all__ = [
    "DCConduction",
    "DampingFrame",
    "Driven",
    "Eigenmode",
    "ElectricDipole",
    "Electrostatic",
    "GaussianBeam",
    "IncidentField",
    "MagneticDipole",
    "MechanicalEigenmode",
    "Piezoelectric",
    "PlaneWave",
    "PortMode",
    "Source",
    "SteadyThermal",
    "TransientThermal",
]


def _surfaces(value: Any, what: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)) and all(isinstance(s, str) for s in value):
        return tuple(value)
    raise TypeError(
        f"{what} must be a surface name or a list of surface names, got {value!r}."
    )


def _terminals(
    terminals: Any, ground: Any, owner: str
) -> tuple[list[tuple[str, tuple[str, ...]]], tuple[str, ...]]:
    """``terminals`` and ``ground`` normalized: ``(name, surfaces)`` pairs, surfaces.

    Raises:
        TypeError: A name is not a string.
        ValueError: No terminal, or a surface in two terminals or in a
            terminal and the ground.
    """
    items: list[tuple[str, tuple[str, ...]]]
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
            f"{owner} needs at least one terminal, e.g. "
            "terminals={'trace': 'trace_surface'}."
        )
    ground = _surfaces(ground, "ground") if ground else ()
    owners: dict[str, str] = {}
    for name, surfaces in items:
        if len(set(surfaces)) != len(surfaces):
            raise ValueError(f"Terminal {name!r} lists a surface twice.")
        for s in surfaces:
            if s in owners:
                raise ValueError(
                    f"Surface {s!r} is in terminals {owners[s]!r} and {name!r}; "
                    "a surface belongs to one terminal."
                )
            owners[s] = name
    for s in ground:
        if s in owners:
            raise ValueError(
                f"Surface {s!r} is both in terminal {owners[s]!r} and in the "
                "ground; remove it from one of them."
            )
    return items, tuple(dict.fromkeys(ground))


def _wave_vector(value: Any, owner: str) -> tuple[float, float, float] | None:
    """``None``, or a Bloch wave vector as three finite floats (rad/m)."""
    if value is None:
        return None
    check_fixed(value, f"{owner}.wave_vector must be fixed numbers")
    try:
        k = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        k = np.empty(0)
    if k.shape != (3,) or not np.isfinite(k).all():
        raise ValueError(
            f"{owner}.wave_vector must be a 3-vector in rad/m (kx, ky, kz), or "
            f"None for plain periodicity; got {value!r}."
        )
    return (float(k[0]), float(k[1]), float(k[2]))


def _wave_vector_frequency(value: Any, wave_vector: Any) -> float | None:
    """``None``, or the frequency (Hz) at which ``wave_vector`` holds."""
    if value is None:
        return None
    check_fixed(value, "Driven.wave_vector_frequency must be a fixed number")
    if wave_vector is None:
        raise ValueError(
            "Driven.wave_vector_frequency scales Driven.wave_vector with "
            "frequency, but wave_vector is None. Give the Bloch wave vector at "
            "that frequency, e.g. Model.bloch_wave_vector(surface, f, theta=...)."
        )
    ok = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )
    if not ok or not math.isfinite(value) or value <= 0:
        raise ValueError(
            "Driven.wave_vector_frequency must be a positive finite frequency in "
            f"Hz, or None for the same wave vector at every frequency; got {value!r}."
        )
    return float(value)


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

    Example:
        The capacitance of a trace made of two surfaces, against ground::

            problem = gfem.Electrostatic(
                terminals={"trace": ["m1_xy", "m1_z"]}, ground="ground"
            )

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

    `gfem.solve` raises NotImplementedError on a model with a
    `gfem.Periodic` pair.
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
        items, ground = _terminals(terminals, ground, "Electrostatic")
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


@dataclasses.dataclass(frozen=True, init=False)
class DCConduction:
    """The DC conductance matrix between named terminals of conductors.

    For each terminal in turn, the potential is 1 V on that terminal and
    0 V on every other terminal and on the ground, and the steady current
    equation div(sigma grad V) = 0 is solved in the conductors. Every
    other surface of a conductor is insulated (no current crosses it).

    The conductors are not named: they are every volume whose material
    has ``conductivity`` > 0, minus the volumes excluded from DC conduction
    (``Model(excluded={"dc_conduction": (...)})``). A conducting piece
    that touches no terminal and no ground carries no current: it is left
    out, and its heat is exactly zero. A tensor ``conductivity`` is read as
    a tensor.

    The result (`gfem.DCConductionResult`) holds the conductance and the
    dissipation per element at unit voltages; `gfem.joule_heating` applies
    a voltage, current or power drive to it and gives the heat source of a
    thermal problem, with no new solve.

    On a cross-section mesh (`gfem.Mesh.from_gmsh_cross_section`), put the
    terminals on its end faces ``"front"`` and ``"back"``: the current runs
    along the waveguide. A conducting volume that touches both end faces
    (a doped substrate, say) then joins the circuit; exclude it from DC
    conduction if no current flows in it.

    Example:
        A heater between two pads::

            dc = gfem.solve(
                model,
                gfem.DCConduction(terminals={"pad_a": "pad_a_top"}, ground="pad_b_top"),
            )
            dc.resistance  # ohm

    Args:
        terminals: Terminal name -> its surface name or names, in
            conductance-matrix order. A list of surface names makes one
            terminal per surface, named after it.
        ground: The surface name or names held at 0 V. Empty: no ground
            (then each row of the conductance matrix sums to zero).

    Raises:
        TypeError: A name is not a string.
        ValueError: No terminal, or a surface in two terminals or in a
            terminal and the ground.

    `gfem.solve` raises ValueError when no volume conducts, when a
    terminal or the ground touches no conducting volume, or when an
    element of a conducting volume has zero conductivity (a
    `gfem.MaterialField` with exact zeros: give its void a small positive
    value).
    """

    terminals: tuple[tuple[str, tuple[str, ...]], ...]
    ground: tuple[str, ...]
    _rule: ClassVar[str] = "gfem._engine.dc_conduction:DCConductionRule"
    _physics: ClassVar[str] = "dc_conduction"

    def __init__(
        self,
        terminals: Mapping[str, str | list[str] | tuple[str, ...]] | list[str],
        ground: str | list[str] | tuple[str, ...] = (),
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        items, ground = _terminals(terminals, ground, "DCConduction")
        object.__setattr__(self, "terminals", tuple(items))
        object.__setattr__(self, "ground", ground)

    @property
    def terminal_names(self) -> tuple[str, ...]:
        """Terminal names, in conductance-matrix order."""
        return tuple(name for name, _ in self.terminals)

    def __repr__(self) -> str:
        terms = ", ".join(f"{n!r}: {list(s)}" for n, s in self.terminals)
        ground = f", ground={list(self.ground)}" if self.ground else ""
        return f"DCConduction(terminals={{{terms}}}{ground})"


jax.tree_util.register_static(DCConduction)


def _heat_sources(heat: Any, owner: str) -> tuple[HeatSource, ...]:
    """``heat`` as a tuple of `gfem.HeatSource` (one, or a sequence of them)."""
    if isinstance(heat, HeatSource):
        return (heat,)
    if isinstance(heat, Sequence) and not isinstance(heat, str):
        sources = tuple(heat)
        if all(isinstance(s, HeatSource) for s in sources):
            return sources
    raise TypeError(
        f"{owner}.heat must be a gfem.HeatSource or a sequence of them (from "
        "gfem.HeatSource.uniform or gfem.joule_heating), got "
        f"{type(heat).__name__}."
    )


@dataclasses.dataclass(frozen=True, init=False)
class SteadyThermal:
    """The steady temperature of a model under heat sources.

    Solves -div(k grad T) = q on the thermal domain: every volume of the
    mesh minus those excluded from thermal problems
    (``Model(excluded={"thermal": (...)})``), with k the materials'
    ``thermal_conductivity``. The boundary conditions are the model's
    `gfem.FixedTemperature`, `gfem.Convection` and `gfem.HeatFlux`; every
    other exterior surface of the thermal domain is insulated
    (`gfem.Model.default_insulated`), and unnamed interior interfaces are in
    perfect thermal contact. Temperatures are absolute, in kelvin.

    The result (`gfem.SteadyThermalResult`) holds the temperature and its
    rise: the part linear in the heat sources given here, which scales
    with their power. The boundary data (fixed temperatures, ambients and
    heat fluxes) set the source-free temperature the rise is added to.

    Example:
        A heater dissipating 10 mW over a substrate held at 293.15 K::

            model = gfem.Model(
                mesh,
                materials=materials,
                boundaries=[gfem.FixedTemperature("bottom", temperature=293.15)],
            )
            heat = gfem.HeatSource.uniform("heater", power=10e-3)
            result = gfem.solve(model, gfem.SteadyThermal(heat))
            result.rise.at(point)  # K

    Args:
        heat: A `gfem.HeatSource` or a sequence of them; they superpose.
            Empty: the temperature set by the boundaries alone.

    Raises:
        TypeError: ``heat`` is not a heat source or a sequence of them.

    `gfem.solve` raises ValueError when a connected piece of the thermal
    domain has no heat sink (no `gfem.FixedTemperature` or
    `gfem.Convection` on it: its steady temperature is undetermined), and
    `gfem.errors.MissingMaterialPropertyError` when a region of the
    thermal domain has no ``thermal_conductivity``.
    """

    heat: tuple[HeatSource, ...]
    _rule: ClassVar[str] = "gfem._engine.thermal:SteadyThermalRule"
    _physics: ClassVar[str] = "thermal"

    def __init__(self, heat: HeatSource | Sequence[HeatSource] = ()) -> None:
        """Validate and normalize (see the class docstring)."""
        object.__setattr__(self, "heat", _heat_sources(heat, "SteadyThermal"))

    def __repr__(self) -> str:
        return f"SteadyThermal(heat={list(self.heat)!r})"


register_fields(SteadyThermal, ("heat",))


#: Two grid times closer than this fraction of the step are the same time.
_GRID_RTOL = 1e-9
_FIXED_TIME = (
    "Build the problem from concrete values, outside the transformed function; "
    "the waveform scales are the differentiable part of a waveform."
)


def _grid_index(time: float, dt: float, steps: int, what: str) -> int:
    """The grid index ``n`` of ``time = n dt``, or a ValueError naming the nearest.

    Raises:
        ValueError: ``time`` is outside [0, duration] or between grid times.
    """
    k = time / dt
    n = round(k)
    if not -_GRID_RTOL <= k <= steps * (1 + _GRID_RTOL):
        raise ValueError(
            f"TransientThermal: {what} {time:g} s is outside the run, from 0 to "
            f"duration={steps * dt:g} s."
        )
    if abs(k - n) > _GRID_RTOL * max(1.0, abs(k)):
        low = math.floor(k)
        raise ValueError(
            f"TransientThermal: {what} {time:g} s is not on the time grid (dt = "
            f"duration / steps = {dt:g} s); the nearest grid times are "
            f"{low * dt:g} s and {(low + 1) * dt:g} s. Move it to a grid time, or "
            "choose steps so that it is one."
        )
    return n


def _waveform(value: Any, dt: float, steps: int) -> tuple[tuple[float, ...], Any]:
    """The waveform's switch times (static) and scales (a JAX array)."""
    try:
        pairs = [tuple(p) for p in value]
    except TypeError:
        pairs = []
    if not pairs or any(len(p) != 2 for p in pairs):
        raise TypeError(
            "TransientThermal.waveform must be a non-empty sequence of (start time "
            f"in s, scale) pairs, e.g. [(0.0, 1.0), (30e-6, 0.0)]; got {value!r}."
        )
    times = tuple(
        static_float(t, "A TransientThermal.waveform start time", _FIXED_TIME)
        for t, _ in pairs
    )
    if any(b <= a for a, b in itertools.pairwise(times)):
        raise ValueError(
            "TransientThermal.waveform start times must increase strictly, got "
            f"{list(times)}."
        )
    for t in times:
        _grid_index(t, dt, steps, "the waveform switch time")
    scales = jax.numpy.stack(
        [_check_finite(v, "A TransientThermal.waveform scale") for _, v in pairs]
    )
    return times, scales


def _kept_times(value: Any, dt: float, steps: int) -> bool | tuple[float, ...]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    check_fixed(value, "TransientThermal.keep_fields must be fixed times")
    try:
        wanted = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise TypeError(
            "TransientThermal.keep_fields takes True, False or grid times in s, got "
            f"{value!r}."
        ) from None
    kept = {_grid_index(float(t), dt, steps, "the kept time") for t in wanted}
    return tuple(n * dt for n in sorted(kept))


def _traces(value: Any) -> dict[str, Trace]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(
        isinstance(k, str) and isinstance(v, Trace) for k, v in value.items()
    ):
        raise TypeError(
            "TransientThermal.traces must be a mapping from name to gfem.Trace, "
            f"e.g. {{'core': gfem.Trace.mean('core')}}; got {value!r}."
        )
    return dict(value)


@dataclasses.dataclass(frozen=True, init=False)
class TransientThermal:
    """The temperature of a model over time, under switched heat sources.

    Solves rho c_p dT/dt - div(k grad T) = s(t) q on the thermal domain
    (as `gfem.SteadyThermal`, with the materials' ``mass_density`` and
    ``specific_heat`` too) on the uniform time grid t_n = n dt, dt =
    ``duration / steps``. The heat sources q follow the waveform s(t):
    piecewise constant, off before its first start time, the value on
    [t_n, t_n+1) applied over that step. Every switch time must be a grid
    time. `gfem.SolverOptions.time_scheme` picks the scheme (TR-BDF2 by
    default; one factorization serves the whole run).

    The result (`gfem.TransientThermalResult`) records each trace at all
    ``steps + 1`` grid times and keeps the temperature field at the final
    time and at the ``keep_fields`` times. Temperatures are absolute, in
    kelvin; the rise is the temperature minus the source-free temperature
    (the steady solve with the heat sources off).

    Example:
        A heater switched on for 30 us, then off for 30 us::

            problem = gfem.TransientThermal(
                gfem.HeatSource.uniform("heater", power=10e-3),
                duration=60e-6,
                steps=240,
                waveform=[(0.0, 1.0), (30e-6, 0.0)],
                traces={"core": gfem.Trace.mean("core")},
            )
            result = gfem.solve(model, problem)
            result.traces["core"]  # K, at result.time

    Args:
        heat: A `gfem.HeatSource` or a sequence of them; they superpose and
            share the waveform.
        duration: The run's length in s (> 0).
        steps: The number of time steps (>= 1).
        waveform: ``(start time in s, scale)`` pairs with increasing start
            times on the grid; the scale multiplies every source from its
            start time to the next one. The default is a step on at t = 0.
        initial_temperature: ``None`` starts from the source-free
            temperature (the steady temperature with the sources off), so
            the rise starts at zero; a number starts from that uniform
            temperature in K (> 0). A model with a part of the thermal
            domain that touches no heat sink needs a number.
        traces: A mapping from name to `gfem.Trace`, recorded every step.
        keep_fields: Which temperature fields the result keeps besides the
            final one: ``False`` none, ``True`` every grid time, or a
            sequence of grid times in s.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: ``duration`` or ``steps`` out of range; a switch time
            or a kept time off the grid (the message names the nearest
            grid times) or outside the run; start times not increasing; a
            temperature <= 0 K.
        NonDifferentiableError: ``duration``, ``steps``, a waveform start
            time, ``initial_temperature`` or a kept time is traced by JAX
            (the waveform scales and the heat sources are differentiable).
    """

    heat: tuple[HeatSource, ...]
    duration: float
    steps: int
    initial_temperature: float | None
    traces: dict[str, Trace]
    keep_fields: bool | tuple[float, ...]
    #: The waveform's start times in s.
    _switches: tuple[float, ...]
    #: The waveform's scales, a JAX array (one per start time).
    _scales: Any
    _rule: ClassVar[str] = "gfem._engine.thermal:TransientThermalRule"
    _physics: ClassVar[str] = "thermal"

    def __init__(
        self,
        heat: HeatSource | Sequence[HeatSource] = (),
        *,
        duration: float,
        steps: int,
        waveform: Sequence[tuple[float, Any]] = ((0.0, 1.0),),
        initial_temperature: float | None = None,
        traces: Mapping[str, Trace] | None = None,
        keep_fields: bool | Sequence[float] = False,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        set_ = object.__setattr__
        set_(self, "heat", _heat_sources(heat, "TransientThermal"))
        duration = static_float(duration, "TransientThermal.duration", _FIXED_TIME)
        if duration <= 0:
            raise ValueError(
                f"TransientThermal.duration must be > 0 s, got {duration}."
            )
        check_fixed(steps, "TransientThermal.steps must be a fixed integer")
        if (
            not isinstance(steps, (int, np.integer))
            or isinstance(steps, bool)
            or steps < 1
        ):
            raise ValueError(
                f"TransientThermal.steps must be an integer >= 1, got {steps!r}."
            )
        steps = int(steps)
        dt = duration / steps
        set_(self, "duration", duration)
        set_(self, "steps", steps)
        times, scales = _waveform(waveform, dt, steps)
        set_(self, "_switches", times)
        set_(self, "_scales", scales)
        if initial_temperature is not None:
            initial_temperature = _temperature(
                initial_temperature, "TransientThermal.initial_temperature"
            )
        set_(self, "initial_temperature", initial_temperature)
        set_(self, "traces", _traces(traces))
        set_(self, "keep_fields", _kept_times(keep_fields, dt, steps))

    @property
    def dt(self) -> float:
        """The time step ``duration / steps``, in s."""
        return self.duration / self.steps

    @property
    def waveform(self) -> tuple[tuple[float, Any], ...]:
        """The ``(start time in s, scale)`` pairs."""
        return tuple(zip(self._switches, self._scales, strict=True))

    def __repr__(self) -> str:
        try:
            scales = [float(v) for v in np.asarray(self._scales)]
            wave = [(t, v) for t, v in zip(self._switches, scales, strict=True)]
        except (TypeError, jax.errors.TracerArrayConversionError):
            wave = list(self._switches)
        return (
            f"TransientThermal(heat={list(self.heat)!r}, duration={self.duration:g}, "
            f"steps={self.steps}, waveform={wave}, traces={list(self.traces)})"
        )


register_fields(TransientThermal, ("heat", "traces", "_scales"))


def _frequencies(value: Any, owner: str = "PortMode") -> tuple[float, ...]:
    check_fixed(value, f"{owner}.frequency must be fixed numbers")
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
    out = tuple(float(x) for x in f)
    seen: set[float] = set()
    for x in out:
        if x in seen:
            raise ValueError(
                f"{owner}.frequency names {x:g} Hz twice: each frequency is solved "
                "once. Drop the repeats, for example with numpy.unique(frequency)."
            )
        seen.add(x)
    return out


Polarization = Literal["Ez", "Hz"]
_POLARIZATIONS = get_args(Polarization)


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
    On a shared port surface it runs the surface's sweep, with the other
    wave ports of its surface, so it gives the same mode a driven problem
    uses; an explicit ``mode`` sweeps that rank alone. The port's
    ``polarity`` fixes the phase of its own mode as in a driven problem; an
    explicit other ``mode`` takes the default phase.

    Each row also carries the mode's group index and dispersion (``n_g``,
    ``beta_2``, ``beta_3``, ``D``), exact frequency derivatives of
    the propagation constant, for about a fifth more time per exact solve.

    Example:
        The mode of wave port ``"p1"`` from 1 to 100 GHz::

            problem = gfem.PortMode("p1", np.linspace(1e9, 100e9, 100))
            modes = gfem.solve(model, problem)

    Args:
        port: The name of a `WavePort` of the model.
        frequency: One frequency or a 1D sequence of distinct frequencies, Hz,
            in any order; results keep the order given.
        mode: Which propagating mode at the lowest frequency, in rank
            order: by decreasing effective index for lossless modes (1 = the
            most confined), with strongly attenuated modes ranked lower.
            ``None``: the wave port's own ``mode``.
        polarization: On a 2D mesh, the field normal to the plane that the
            line port's mode carries: ``"Ez"`` (its field is
            `PortModeResult.e_field`) or ``"Hz"`` (`PortModeResult.h_field`).
            Required there, and ``None`` on a 3D mesh, as for `Driven`.

    Raises:
        TypeError: ``port`` is not a name.
        ValueError: A frequency is not finite and positive, ``mode`` is
            not an integer >= 1, or ``polarization`` is not ``"Ez"``, ``"Hz"``
            or ``None``. `gfem.solve` raises it for no ``polarization`` on a
            2D mesh and for one on a 3D mesh.
        NonDifferentiableError: ``frequency`` is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap` from their arguments; pass a fixed value.

    `gfem.solve` raises `PortModeError` when the mode is below cutoff
    somewhere in the band (naming the estimated cutoff) or degenerate, or
    follows the same mode as another wave port of its surface, and
    warns with `PortModeTrackingWarning` when it crosses another mode. The
    2D problem always runs SuperLU and SciPy on the CPU, so it ignores
    `SolverOptions.device`, ``linear_solver`` and ``eigen_solver``, as a
    driven solve's port mode sweep does: one `SolverOptions` serves a
    `PortMode` check and the driven solve after it.

    Port modes are differentiable: under `jax.grad`, ``gamma`` (so ``n_eff``),
    ``Z_PV`` and the field's values (`PortModeField.at`) differentiate with
    respect to every input the problem reads that the same model
    differentiates elsewhere: the material values and `gfem.MaterialField`
    designs of the regions on the port face, permittivity-law parameters, the
    fill of a sheet field and the R, L, C of an `Impedance` on an edge of the
    section, and the node coordinates of the face (not through ``at``).
    ``residual`` and the group index and dispersion (``n_g``, ``beta_2``,
    ``beta_3``, ``D``) raise `gfem.NonDifferentiableError`, and a thermal
    property gets an exact zero. ``gamma`` alone adds no solve to the backward
    pass; a field or ``Z_PV`` objective adds one 2D solve per mode and
    frequency. A line port's mode (a 2D mesh) differentiates with respect to
    the material field values of the regions on its line, in Ez and Hz
    (``Z_PV`` is NaN there: a line port has no voltage path); its node
    coordinates and law parameters raise `gfem.NonDifferentiableError`.
    """

    port: str
    frequency: tuple[float, ...]
    mode: int | None
    polarization: Polarization | None = None
    _rule: ClassVar[str] = "gfem._sweep.port_mode:PortModeRule"

    def __init__(
        self,
        port: str,
        frequency: Any,
        *,
        mode: int | None = None,
        polarization: Polarization | None = None,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        _polarization(polarization, "PortMode")
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
        object.__setattr__(self, "polarization", polarization)


jax.tree_util.register_static(PortMode)


Sweep = Literal["uniform", "adaptive"]
_SWEEPS = get_args(Sweep)
#: Frequency grids up to this size are listed in full in error messages.
_LISTED = 8


def _port_names(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = (value,) if isinstance(value, str) else tuple(value)
    if not all(isinstance(n, str) for n in names):
        raise TypeError(f"Driven.excite takes port names (strings), got {value!r}.")
    if len(set(names)) != len(names):
        raise ValueError(f"Driven.excite lists a port twice: {value!r}.")
    return names


_SOURCE_TYPES = (*INCIDENT, *DIPOLES)


def _sources(value: Any) -> tuple[Source, ...]:
    """``sources`` as a tuple of sources with distinct names."""
    items = (
        tuple(value)
        if isinstance(value, Sequence) and not isinstance(value, str)
        else (value,)
    )
    for item in items:
        if not isinstance(item, _SOURCE_TYPES):
            raise TypeError(
                "Driven.sources takes a source or a sequence of them (gfem."
                "PlaneWave, GaussianBeam, IncidentField, ElectricDipole, "
                f"MagneticDipole), got {type(item).__name__}."
            )
    seen: set[str] = set()
    for item in items:
        if item.name in seen:
            raise ValueError(
                f"Driven.sources names {item.name!r} twice: each source labels "
                "its own column and kept fields. Give each a name of its own."
            )
        seen.add(item.name)
    return items


def _check_sources(problem: Driven) -> None:
    """The refusals of ``sources`` that the problem alone shows."""
    sources = problem.sources
    if not isinstance(problem.source_fields, bool):
        raise TypeError(
            f"Driven.source_fields takes True or False, got {problem.source_fields!r}."
        )
    if problem.excite == () and not sources:
        raise ValueError(
            "Driven.excite=() with no sources leaves nothing to excite: name the "
            "ports to feed (None feeds every port), or give sources=."
        )
    if not sources:
        if not problem.source_fields:
            raise ValueError(
                "Driven(source_fields=False) gives up the sources' own fields, but "
                "the problem has no source: give sources=, or leave source_fields "
                "at True."
            )
        return
    for name in problem.excite or ():
        if name in {s.name for s in sources}:
            raise ValueError(
                f"{name!r} names both an excited port and a source: a name labels "
                "one column. Rename the source."
            )
    if problem.wave_vector is not None:
        raise NotImplementedError(
            "Driven.sources with a Bloch wave_vector: incident fields and dipoles "
            "on a periodic cell are not supported. For a plane wave on a "
            "periodic cell, use a gfem.FloquetPort."
        )
    if problem.polarization is not None:
        raise NotImplementedError(
            "Driven.sources solve on a 3D mesh only; a 2D model (polarization="
            f"{problem.polarization!r}) takes no sources. Solve a slab model, "
            "one element thick, for a 2D-like geometry."
        )
    incident = [s for s in sources if isinstance(s, INCIDENT)]
    if problem.sweep == "adaptive" and incident:
        raise NotImplementedError(
            f"The adaptive sweep cannot take the incident field {incident[0].name!r} "
            f"({type(incident[0]).__name__}): its load is not a fixed vector times "
            "a function of frequency. Use sweep='uniform'; dipoles are accepted "
            "by the adaptive sweep."
        )
    if not problem.source_fields:
        _check_reciprocal_sources(problem, incident)


def _check_reciprocal_sources(problem: Driven, incident: Sequence[Source]) -> None:
    """The refusals of ``source_fields=False`` that the problem alone shows."""
    if incident:
        raise NotImplementedError(
            "Driven(source_fields=False) reads port waves by reciprocity for point "
            f"dipoles only; the incident field {incident[0].name!r} "
            f"({type(incident[0]).__name__}) is not supported. Solve it with "
            "source_fields=True (the default)."
        )
    if problem.excite == () and problem.kept_frequencies:
        raise ValueError(
            "Driven(source_fields=False, excite=()) solves no field that "
            "keep_fields could keep: the sources' fields are not solved and no "
            "port is excited. Drop keep_fields, or excite a port."
        )


@dataclasses.dataclass(frozen=True, init=False)
class Driven:
    r"""The frequency-domain response of a model to its ports and sources.

    At each frequency, each excited port in turn is fed 1 W time-averaged
    incident power while every other port is terminated in its own impedance,
    and the electric field solves the time-harmonic Maxwell equations
    (:math:`e^{+i\omega t}`) in the electromagnetic solve domain. The result is
    the scattering matrix (pseudo-waves, each lumped port's reference
    impedance its resistance; see `gfem.NetworkParameters`) and, on request,
    the electric fields (see `gfem.conventions` for their amplitude).

    Each source (an incident field, `gfem.PlaneWave`, `gfem.GaussianBeam` or
    `gfem.IncidentField`, or a point dipole, `gfem.ElectricDipole` or
    `gfem.MagneticDipole`) is one more column, solved after the excited
    ports with every port terminated: its result is the outgoing wave at
    each port (``DrivenResult.port_waves``), a dipole's emitted power
    (``DrivenResult.emitted_power``) and, on request, its total field. With
    ``source_fields=False`` a point dipole gets no column: its port waves
    are read from the ports' own solutions by reciprocity. Port
    names and source names label the same columns, so they are distinct.

    A `WavePort` is fed by its own port mode, solved before the 3D solve to
    ``SolverOptions.port_mode_tolerance``; S is the field's projection on
    that mode, and ``WavePort(offset=...)`` de-embeds it. The port absorbs
    its own mode without reflection, hybrid modes included, and the modes
    are on ``DrivenResult.port_modes``.

    On a 2D model (a mesh of dimension 2) the solve computes one
    polarization, ``polarization="Ez"`` or ``"Hz"``, and each excited port is
    fed 1 W/m per metre along :math:`z` (`gfem.conventions`, 2D models).

    Example:
        Solve 100 frequencies in full, or read a 300-point sweep from a few
        full solves::

            uniform = gfem.Driven(frequency=np.linspace(1e9, 100e9, 100))
            adaptive = gfem.Driven(
                frequency=np.linspace(1e9, 100e9, 300), sweep="adaptive"
            )
            result = gfem.solve(model, adaptive)

    Args:
        frequency: One frequency or a 1D sequence of distinct frequencies, Hz,
            in any order; results keep the order given.
        excite: The port name or names fed in turn, one column of S each, in
            this order. ``None``: every port of the model, in model order.
            ``()``: no port, with ``sources`` only (S then has no column).
        sweep: ``"uniform"`` solves every frequency in full.
            ``"adaptive"`` solves a few of them in full (every excited port
            at once, and one more right-hand side per port left out of
            ``excite``, which keeps gradients exact there) and reads S and
            the kept fields at every frequency from a reduced model built
            on those solutions: the greedy picks each next frequency from
            ``frequency`` where the reduced model is least accurate, until
            ``SolverOptions.adaptive_memory`` samples in a row agree with it to
            ``SolverOptions.adaptive_tolerance`` (relative error of the
            field). S is then accurate to roughly the square of that error.
            Worth it above about 20 frequencies; the result's ``sweep``
            tells which frequencies were solved.
        samples: With ``sweep="adaptive"``, the frequencies to solve in full,
            in order, instead of the greedy's choice: each must be one of
            ``frequency``. Freezing the samples of a first sweep
            (``samples=result.sweep.sampled_frequencies``) keeps the reduced
            model's sample set fixed while the model changes, e.g. across
            optimizer iterations, so the objective is smooth in the design.
            Under `jax.grad`, `jax.jit` or `jax.vmap` an adaptive sweep
            needs frozen samples. Its gradients hold the reduced basis
            fixed: equal to the full solve's at the sample frequencies,
            accurate to the sweep's field error between them.
        keep_fields: Which electric fields the result keeps, one per kept
            frequency and column (each excited port and, unless
            ``source_fields=False``, each source): ``False`` (the default) none,
            ``True`` all, or a frequency or list of frequencies of
            ``frequency``. A field is a complex vector per DOF (16 bytes per
            DOF; 160 MB at 10M DOFs), so keep only what you use.
        wave_vector: The Bloch wave vector k = (kx, ky, kz), rad/m
            (Cartesian), of a model with `gfem.Periodic` pairs: across each
            pair, E(r + a) = exp(-i k·a) E(r), with a the pair's translation
            (`gfem.conventions`, Bloch periodicity). The same k at every
            frequency. ``None``: plain periodicity, the fields equal on both
            surfaces of every pair. Any vector, (0, 0, 0) included, declares
            a Bloch solve: a uniform sweep on the CPU, with
            ``linear_solver`` ``"direct"`` or ``"bddc_gmres"``, on a model
            with one pair across MPI ranks, whose S is
            not symmetric (S(k)ᵀ = S(-k)) and is referenced to the field E,
            not to its periodic envelope (`gfem.conventions`).
        wave_vector_frequency: The frequency, Hz, at which ``wave_vector``
            holds, for a fixed-angle sweep: the solve at frequency f uses the
            wave vector ``wave_vector * f / wave_vector_frequency``. A Floquet
            port's half-space has a real constant index, so this keeps the
            incidence angle fixed in every port medium at once. Build
            ``wave_vector`` with `gfem.Model.bloch_wave_vector` at this
            frequency. ``None``: the same wave vector at every frequency.
        polarization: On a 2D mesh, the field normal to the plane that the
            solve computes: ``"Ez"`` (E normal to the plane, H in it; kept
            fields read with ``DrivenResult.e_field``) or ``"Hz"`` (H normal
            to the plane; read with ``DrivenResult.h_field``). Required
            there, since a perfect conductor is a different boundary
            condition for each. ``None`` (the default) on a 3D mesh, whose
            solve holds both.
        sources: A source or a sequence of them, one column each after the
            excited ports (none for a point dipole with
            ``source_fields=False``), in this order, labelled by their
            names. An
            incident field needs ``Model(background=...)``, the medium it is
            a solution of.
        source_fields: Whether each source's own field is solved. ``True``
            (the default): each source is a column (see ``sources``).
            ``False``: no source is solved, and nothing that needs a source's
            field exists (its emitted power, Purcell factor, kept field, far
            field). gfem solves each port's own field once, the excited
            ports' columns and one load per other port with a defined wave,
            and reads each dipole's port waves from them by reciprocity,
            :math:`b_i = -\tfrac{i\omega}{4}\,\mathbf{p}\cdot\mathbf{E}_i(\mathbf{r}_0)`
            with :math:`\mathbf{E}_i` the field of port :math:`i` fed 1 W, at
            the cost of one point overlap per dipole: a coupling map over any
            number of positions. Unlike ``keep_fields``, which chooses which
            solved fields are stored, this chooses whether the sources' fields
            are solved at all. Point dipoles only. The gradients cost one
            adjoint solve per port at each frequency whatever the number of
            dipoles. On a uniform sweep they are exact. On an adaptive sweep
            they are exact at the samples and accurate to the sweep's field
            error between them, at one full-size factorization per frequency
            whose port waves carry a cotangent: cheap for an objective at a
            few frequencies, about a uniform backward over the whole grid
            when every frequency counts.

    Raises:
        TypeError: ``excite`` is not port names, ``keep_fields`` is not a
            bool or frequencies, ``sources`` holds something else than a
            source, or ``source_fields`` is not a bool.
        ValueError: A frequency is not finite and positive, ``excite`` is
            empty with no sources or repeats a port, two sources share a
            name, a source is named like an excited port, ``sweep`` is
            unknown, ``samples`` is
            given without ``sweep="adaptive"``, is empty or repeats a
            frequency, a ``samples`` or ``keep_fields`` frequency is not one
            of ``frequency``, ``source_fields`` is ``False`` with no
            sources, or with ``excite=()`` and ``keep_fields``,
            ``wave_vector`` is not a finite 3-vector, or
            ``wave_vector_frequency`` is not a positive finite frequency or
            is given without ``wave_vector``, or ``polarization`` is not
            ``"Ez"``, ``"Hz"`` or ``None``. `gfem.solve` also raises it
            for no ``polarization`` on a 2D mesh and for one on a 3D mesh.
            With ``wave_vector``,
            `gfem.solve` also raises it on a model without a `gfem.Periodic`
            pair, for a k with a component outside the span of the pairs'
            translations, and with
            ``linear_solver="petsc"``; use ``"direct"`` or ``"bddc_gmres"``.
            With ``source_fields=False``, `gfem.solve` also raises it for a
            model with no port with a defined wave (a wave port, or a lumped
            port with a resistance).
            With ``sources``, `gfem.solve` also raises it for a source named
            like a port of the model, and for an incident field on a model
            without ``background``, with no PML layer and no
            `gfem.Absorbing` face, with an exterior `gfem.Impedance`,
            `gfem.Conductor` or `gfem.LumpedPort` face, or above
            :math:`10^{-3}` of its peak where a scatterer meets the layer or
            an absorbing face, on a wave port face, or across an exterior
            PEC or PMC wall (`gfem.conventions`, Incident fields). With
            ``frequency_workers`` above 1, it raises it before spawning any
            worker for a `gfem.IncidentField` whose function a spawned
            worker cannot load (a lambda, a closure, or a function of a
            notebook or REPL).
        NotImplementedError: ``sources`` with a ``wave_vector`` or a 2D
            ``polarization``, an incident field with
            ``sweep="adaptive"`` (dipoles are accepted) or with
            ``source_fields=False``. `gfem.solve` raises
            it for ``wave_vector`` with ``sweep="adaptive"`` or a GPU
            ``device``; solve a uniform sweep on the CPU; for
            ``sources`` on a model with a `gfem.Periodic` pair; and for a
            `gfem.GaussianBeam` in a layered or a lossy background or a
            `gfem.PlaneWave` from a lossy layer.
        NonDifferentiableError: ``frequency``, ``samples``, ``keep_fields``,
            ``wave_vector`` or ``wave_vector_frequency`` is computed inside
            `jax.grad`, `jax.jit` or `jax.vmap` from their arguments; pass a
            fixed value.
        LinearSolverConvergenceError: `gfem.solve` raises it when an
            iterative linear solve (``linear_solver="bddc_gmres"`` or
            ``"pmg_gmres"``, or ``"petsc"`` with a Krylov ``ksp_type``) does
            not converge at some frequency. A uniform sweep stops there and
            keeps what it solved: the error's ``.result`` holds the
            frequencies solved before it, its ``.unsolved`` the rest; solve
            ``err.unsolved`` again with ``linear_solver="direct"``. An
            adaptive sweep keeps neither, since its reduced model needs every
            sample; rerun it with
            ``linear_solver="direct"``.

    Notes:
        The mesh's low-frequency floor is
        :math:`k_0 h_\mathrm{min} = 10^{-6}` (:math:`f h_\mathrm{min} \approx`
        48 Hz·m, with :math:`h_\mathrm{min}` the shortest edge, at order 2):
        48 MHz with 1 µm elements and 5 GHz with 10 nm elements. Below it, the
        gradients :math:`\nabla \phi`, on which the curl-curl term vanishes,
        dominate the E-field system. With the default
        ``SolverOptions.gauge="auto"``, a direct solve on the CPU in one
        process goes through the tree-cotree gauge, which carries those
        gradients as node potentials and one floating potential per
        conductor: S, the fields and their gradients keep their digits below
        the floor, and the solve does not warn.

        The other driven solves run ungauged: ``gauge="none"``,
        ``linear_solver="bddc_gmres"``, ``"pmg_gmres"`` or ``"petsc"``,
        ``device="cuda"``,
        several MPI ranks, and models with a `gfem.Periodic` pair, a Bloch
        ``wave_vector`` or a `gfem.FloquetPort`. Below the floor they warn
        with `gfem.LowFrequencyWarning`, and their rounding error grows like
        :math:`10^{-16} / (k_0 h_\mathrm{min})^2`, so the fields and
        gradients lose digits. With an exact factorization (``"direct"``
        with ``gauge="none"`` or on the GPU, or ``"petsc"`` with its default
        MUMPS factorization) the capacitive part of S loses digits too. The
        inductance read from S and reciprocity are unaffected down to at
        least :math:`k_0 h_\mathrm{min} = 10^{-8}`. Ungauged MKL PARDISO
        (``"direct"`` with the ``mkl`` extra) becomes unreliable below
        :math:`k_0 h_\mathrm{min} \approx 3 \times 10^{-8}`. With
        ``bddc_gmres`` or ``pmg_gmres``, S stays at
        ``SolverOptions.tolerance``, but a capacitance read from S is good only to about
        ``tolerance`` / :math:`(\omega C Z_0)`, while B and `gfem.power_flow`
        lose more (see ``SolverOptions.tolerance``).

        The wave ports' mode problem is not gauged and keeps its own floor:
        below the floor, the modes of a `gfem.WavePort`, and S through them, carry
        an error that grows like :math:`1/f^2`, and a gauged solve with wave
        ports still warns. Use lumped ports at these frequencies. For
        capacitance at low frequency, solve `gfem.Electrostatic`.

        A driven solve ignores ``SolverOptions.eigen_solver``.
    """

    frequency: tuple[float, ...]
    excite: tuple[str, ...] | None
    sweep: Sweep
    keep_fields: bool | tuple[float, ...]
    samples: tuple[float, ...] | None = None
    wave_vector: tuple[float, float, float] | None = None
    wave_vector_frequency: float | None = None
    polarization: Polarization | None = None
    sources: tuple[Source, ...] = ()
    source_fields: bool = True
    _rule: ClassVar[str] = "gfem._engine.driven:DrivenRule"
    #: The rule on a 2D mesh: the scalar field normal to the plane.
    _rule_2d: ClassVar[str] = "gfem._engine.driven2d:Driven2DRule"

    def __init__(
        self,
        frequency: Any,
        *,
        excite: str | Sequence[str] | None = None,
        sweep: Sweep = "uniform",
        samples: Sequence[float] | None = None,
        keep_fields: bool | float | Sequence[float] = False,
        wave_vector: Sequence[float] | None = None,
        wave_vector_frequency: float | None = None,
        polarization: Polarization | None = None,
        sources: Source | Sequence[Source] = (),
        source_fields: bool = True,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        if sweep not in _SWEEPS:
            raise ValueError(
                f"Driven.sweep must be 'uniform' or 'adaptive', got {sweep!r}."
            )
        _polarization(polarization, "Driven")
        frequencies = _frequencies(frequency, "Driven")
        object.__setattr__(self, "frequency", frequencies)
        object.__setattr__(self, "excite", _port_names(excite))
        object.__setattr__(self, "sweep", sweep)
        object.__setattr__(self, "keep_fields", _kept(keep_fields, frequencies))
        object.__setattr__(self, "samples", _samples(samples, sweep, frequencies))
        object.__setattr__(self, "wave_vector", _wave_vector(wave_vector, "Driven"))
        object.__setattr__(
            self,
            "wave_vector_frequency",
            _wave_vector_frequency(wave_vector_frequency, self.wave_vector),
        )
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "sources", _sources(sources))
        object.__setattr__(self, "source_fields", source_fields)
        _check_sources(self)

    @property
    def kept_frequencies(self) -> tuple[float, ...]:
        """The frequencies whose fields are kept, in ``frequency`` order."""
        if self.keep_fields is True:
            return self.frequency
        if self.keep_fields is False:
            return ()
        return self.keep_fields


def check_problem(problem: Any, model: Any) -> None:
    """The refusals of a problem that need the model, before any assembly.

    Raises:
        ValueError: See `_check_polarization` and `_check_model_sources`.
        NotImplementedError: See `_check_model_sources`.
    """
    _check_polarization(problem, model.mesh.dim)
    _check_model_sources(problem, model)


def _check_model_sources(problem: Any, model: Any) -> None:
    """The refusals of a driven problem's sources that need the model.

    Raises:
        ValueError: A source is named like a port of the model, or an
            incident field has no background stack.
        NotImplementedError: The model has a periodic pair.
    """
    if not isinstance(problem, Driven) or not problem.sources:
        return
    ports = set(model.port_names)
    for source in problem.sources:
        kind = f"{type(source).__name__} {source.name!r}"
        if source.name in ports:
            raise ValueError(
                f"{source.name!r} names both a port of the model and a source "
                f"({kind}): a name labels one column. Rename the source."
            )
        if model.periodic_translations:
            raise NotImplementedError(
                f"The {kind} on a model with periodic pairs: incident fields and "
                "dipoles on a periodic cell are not supported. For a plane wave "
                "on a periodic cell, use a gfem.FloquetPort."
            )
        if isinstance(source, INCIDENT) and model.background is None:
            raise ValueError(
                f"The {kind} is an incident field, the solution of a background "
                "stack, but the model has none: give Model(background=...), the "
                "medium the field solves (a gfem.Material when homogeneous)."
            )
    if not problem.source_fields:
        _check_reciprocal_ports(model)


def _check_reciprocal_ports(model: Any) -> None:
    """Raise unless some port of the model has a defined wave."""
    start = (
        "Driven(source_fields=False) reads each source's port waves from the "
        "ports' own solutions, but "
    )
    if not model.ports:
        raise ValueError(
            start + "the model has no port: solve with source_fields=True for the "
            "emitted power."
        )
    defined = [
        not (isinstance(p, LumpedPort) and p.rlc[0] is None) for p in model.ports
    ]
    if not any(defined):
        raise ValueError(
            start + "no port of the model has a defined wave (a wave port, or a "
            "lumped port with a resistance): give a port a resistance, or solve "
            "with source_fields=True for the emitted power."
        )


def _polarization(value: Any, owner: str) -> None:
    """Raise for a ``polarization`` that is not ``"Ez"``, ``"Hz"`` or ``None``."""
    if value is not None and value not in _POLARIZATIONS:
        raise ValueError(
            f"{owner}.polarization must be 'Ez' (E normal to the plane of a 2D "
            "mesh) or 'Hz' (H normal to it), or None on a 3D mesh, got "
            f"{value!r}."
        )


def _check_polarization(problem: Any, dim: int) -> None:
    """Raise when a problem's polarization does not fit a mesh of ``dim``.

    Raises:
        ValueError: No polarization on a 2D mesh, or one on a 3D mesh, for a
            driven or port mode problem.
    """
    if not isinstance(problem, (Driven, PortMode)):
        return
    owner = type(problem).__name__
    if dim == 2 and problem.polarization is None:
        raise ValueError(
            f'A 2D mesh solves one polarization: pass {owner}(polarization="Ez") '
            '(E normal to the plane) or polarization="Hz" (H normal to the '
            "plane)."
        )
    if dim != 2 and problem.polarization is not None:
        raise ValueError(
            f"{owner}.polarization applies to a 2D mesh; a 3D solve has both "
            f"polarizations. Leave it out (got {problem.polarization!r})."
        )


def _kept(
    value: Any, frequencies: tuple[float, ...], owner: str = "Driven"
) -> bool | tuple[float, ...]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    check_fixed(value, f"{owner}.keep_fields must be fixed frequencies")
    try:
        wanted = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise TypeError(
            f"{owner}.keep_fields takes True, False or frequencies in Hz, got "
            f"{value!r}."
        ) from None
    kept = set(_on_grid(wanted, frequencies, "keep_fields", owner))
    return tuple(frequencies[i] for i in sorted(kept))


def _samples(
    value: Any, sweep: str, frequencies: tuple[float, ...]
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if sweep != "adaptive":
        raise ValueError(
            "Driven.samples freezes the samples of an adaptive sweep; pass "
            "sweep='adaptive' with it."
        )
    check_fixed(value, "Driven.samples must be fixed frequencies")
    try:
        wanted = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise TypeError(
            f"Driven.samples takes frequencies in Hz, got {value!r}."
        ) from None
    if not wanted.size:
        raise ValueError("Driven.samples needs at least one frequency.")
    indices = _on_grid(wanted, frequencies, "samples")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Driven.samples names a frequency twice: {list(wanted)} Hz.")
    return tuple(frequencies[i] for i in indices)


def _on_grid(
    wanted: np.ndarray,
    frequencies: tuple[float, ...],
    what: str,
    owner: str = "Driven",
) -> list[int]:
    """Index in ``frequencies`` of each of ``wanted`` (1e-9 relative)."""
    grid = np.asarray(frequencies)
    out = []
    for f in wanted:
        match = np.flatnonzero(np.isclose(grid, f, rtol=1e-9, atol=0))
        if not match.size:
            listed = (
                ", ".join(f"{g:g}" for g in grid)
                if len(grid) <= _LISTED
                else f"{len(grid)} frequencies, {grid.min():g} to {grid.max():g}"
            )
            raise ValueError(
                f"{owner}.{what} frequency {f:g} Hz is not one of the "
                f"problem's frequencies ({listed} Hz)."
            )
        out.append(int(match[0]))
    return out


jax.tree_util.register_static(Driven)


@dataclasses.dataclass(frozen=True, init=False)
class Eigenmode:
    r"""The resonant modes of a model nearest a target frequency.

    Finds the source-free solutions of the time-harmonic Maxwell equations
    (:math:`e^{+i\omega t}`) in the electromagnetic solve domain: complex
    angular frequencies ``omega`` (:math:`\mathrm{Im}\,\omega > 0` for a
    decaying mode) and their
    electric fields. Losses (``tan_delta``, ``conductivity``), resistive
    lumped ports, wave ports and `Impedance` boundaries, and first-order
    `Absorbing` boundaries make the modes decay; each mode's Q is
    :math:`|\omega| / (2\, \mathrm{Im}\,\omega)`. The `n_modes` modes whose
    frequency :math:`\mathrm{Re}\,\omega / 2\pi` is nearest ``target`` are
    returned, nearest first.

    A `Conductor` (its surface impedance grows as the square root of the
    frequency), a second-order `Absorbing` boundary (a term in 1 /
    frequency) or a `WavePort` (its propagation constant is a square root in
    frequency) makes the eigenproblem nonlinear. The eigen solver's first pass
    then freezes those factors at ``target``. Each mode is refined on the
    exact frequency dependence, reusing that pass's factorization, so
    frequency and Q do not depend on ``target``. Conductor and radiation
    losses must be perturbations (Q above about 5), and so must a wave port's
    outside the strip below; for a strongly radiating structure, find the
    resonance from a `gfem.Driven` sweep.

    With a perfectly matched layer (`gfem.PML`) the radiation it absorbs
    need not be a perturbation, unlike an `Absorbing` boundary's: Q is the
    total Q, radiation included. The layer's reference frequency is
    ``target`` unless pinned, the modes that live in the layer are dropped,
    and the 1 J normalization and the participations count the regions
    outside the layer.

    A `WavePort`, on its own surface or one shared with other wave ports,
    lets its requested mode leave without reflection, so a mode's Q includes
    its radiation through the port. The port's mode is evaluated at the
    complex eigenfrequency on the outgoing branch,
    :math:`\gamma = i\sqrt{\varepsilon_r\mu_r k_0^2 - k_c^2}` continued from
    real frequencies above the cutoff :math:`k_c`, so at a decaying mode
    :math:`\mathrm{Re}\,\gamma < 0`: the field grows along the guide away
    from the model, as a quasi-normal mode's does. ``target`` must lie above
    the cutoff of every requested port mode. Below a cutoff the outgoing
    branch also has roots that are not resonances: a real one (an anti-bound
    root) or one that grows in time is dropped with
    `gfem.AntiBoundRootWarning`, and the next mode is returned; a root that
    decays in time is a resonance and is kept. Each port's mode sweep covers
    a band around ``target``, solved to ``SolverOptions.port_mode_tolerance``.
    With wave ports the modes returned are the ``n_modes`` with
    :math:`\mathrm{Im}\,\omega \le \omega_t / 4` (:math:`\omega_t = 2\pi`
    ``target``; Q above about 2 near the target) whose
    :math:`\mathrm{Re}\,\omega / 2\pi` is nearest ``target``, nearest first:
    the search certifies that it listed every resonance of that strip within
    the farthest mode's distance, so the set does not depend on ``n_modes``
    beyond its length. A resonance beyond the strip (Q below about 2) is out of
    reach, also when it lies nearest ``target``; find it from a `gfem.Driven`
    sweep. When the certificate cannot be met (the strip holds more modes than
    the search's seed solves can list) `gfem.solve` raises
    `gfem.EigenmodeConvergenceError`: lower ``n_modes`` or move ``target``.
    A port mode family no wave port requests is partly reflected, as in a
    driven problem, so a mode that family carries, or a mode trapped inside
    the model against the port, has a Q that is the port's, not the
    device's. `gfem.EigenmodeResult.requested_power_share` reads it, and a
    share below one half warns `gfem.UnrequestedPortModeWarning`.

    Ports are terminations here: nothing is excited. A lumped port with a
    resistance and every wave port get a coupling Q, a lumped port with an
    inductance a participation ratio (see `gfem.EigenmodeResult`).

    Example:
        The two modes nearest 5 GHz::

            result = gfem.solve(model, gfem.Eigenmode(target=5e9, n_modes=2))

    Args:
        target: The frequency to search around, Hz.
        n_modes: How many modes.
        wave_vector: The Bloch wave vector k = (kx, ky, kz), rad/m
            (Cartesian), of a model with `gfem.Periodic` pairs: across each
            pair, E(r + a) = exp(-i k·a) E(r), with a the pair's translation
            (`gfem.conventions`, Bloch periodicity). A band diagram is one
            problem per k on the same model (one k per problem). ``None``:
            plain periodicity, the fields equal on both surfaces of every
            pair. Any vector, (0, 0, 0) included, declares a Bloch solve:
            a lossless, undamped cell gives real ``omega``, and each mode's
            phase makes its largest coefficient real and positive.

    Raises:
        ValueError: ``target`` is not a finite positive number,
            ``n_modes`` is not an integer >= 1, or ``wave_vector`` is not a
            finite 3-vector.
        NonDifferentiableError: ``target`` or ``wave_vector`` is computed
            inside `jax.grad`, `jax.jit` or `jax.vmap` from their arguments;
            pass a fixed value.

    `gfem.solve` raises `gfem.EigenmodeConvergenceError` when fewer than
    ``n_modes`` modes converge (never a partial result), including when the
    eigen solver's first pass stops at its bound of 100 restarts. That happens
    where the modes near ``target`` are closer together than their linewidth:
    above a modal overlap of about 10 in a lossy region, "nearest" is
    ill-defined, and the message names the region. Close to a lossless resonance
    of a `gfem.PermittivityLaw` the modes crowd too, and a mode one step farther
    can come back in place of a nearer one: ask for more modes and compare two
    targets. A target or a mode in a law region's unresolved band, where its
    wavelength is shorter than two of its longest edges, raises ValueError. A
    ``target`` below the cutoff of a requested port mode raises
    `gfem.PortModeError` naming the port, the mode and the cutoff. A wave
    port with ``wave_vector``, a wave port face crossing a `gfem.PML`, an
    absorbing edge beside a `gfem.PermittivityLaw` in a port's
    cross-section, and a `gfem.FloquetPort` raise NotImplementedError: use a
    `gfem.Driven` sweep. With ``wave_vector``, it raises
    ValueError on a model without a `gfem.Periodic` pair, for a k with a
    component outside the span of the pairs' translations, and with any
    ``linear_solver`` but ``"direct"``; and NotImplementedError on a GPU
    ``device``. Across MPI ranks (the "Run a distributed solve" recipe) the
    eigen solver is SLEPc, and a ``wave_vector``, ``eigen_solver="arpack"``,
    two or three `gfem.Periodic` pairs, a `gfem.WavePort` and a gradient
    through a mode's field raise `gfem.DistributedUnsupportedError`.

    Notes:
        When ``target`` or the lowest mode found is below the mesh's
        low-frequency floor, :math:`k_0 h_\mathrm{min} = 10^{-6}`
        (:math:`f h_\mathrm{min} \approx` 48 Hz·m, with :math:`h_\mathrm{min}`
        the shortest edge; see `gfem.Driven`, Notes), the solve warns with
        `gfem.LowFrequencyWarning`. Below the floor the eigenfrequencies,
        the fields and the participations lose digits like
        :math:`10^{-16} / (k_0 h_\mathrm{min})^2`, and
        `gfem.EigenmodeResult.error` does not show it: a mode at
        :math:`k_0 h_\mathrm{min} = 2 \times 10^{-7}` had a frequency off by 4e-4 with a
        reported error of 1e-16.
    """

    target: float
    n_modes: int
    wave_vector: tuple[float, float, float] | None = None
    _rule: ClassVar[str] = "gfem._eigen.rule:EigenmodeRule"

    def __init__(
        self,
        target: float,
        *,
        n_modes: int = 1,
        wave_vector: Sequence[float] | None = None,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        check_fixed(target, "Eigenmode.target must be a fixed number")
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
        object.__setattr__(self, "wave_vector", _wave_vector(wave_vector, "Eigenmode"))


jax.tree_util.register_static(Eigenmode)


_FRAME_FIXED = (
    "The loss factor of a damping frame is fixed: give a plain number, outside "
    "the transformed function."
)


@dataclasses.dataclass(frozen=True, init=False)
class DampingFrame:
    r"""The regions that absorb the acoustic waves leaving a piezoelectric section.

    Pass one as ``gfem.Piezoelectric(..., damping_frame=...)``, naming every
    frame region. Each region keeps its layer's material in ``materials``
    (the lithium niobate frame region holds the lithium niobate): the frame
    multiplies its stiffness by :math:`1 + i\eta r^2`, added to the
    material's own :math:`1/Q_m`, so that the stiffness is

    .. math::

        c\left(1 + i\left(\frac{1}{Q_m} + \eta\, r^2\right)\right),

    with :math:`r` the normalised depth into the frame and :math:`\eta` the
    ``loss_factor``. Only the stiffness is damped, not the piezoelectric
    coupling or the permittivity.

    The geometry is read from the imported mesh, as `gfem.PML` reads its
    own: the interior box is the bounding box of the mechanical-domain
    triangles outside the frame regions. A frame triangle beyond one of the
    box's edges has :math:`r` = the distance past the edge over the frame's
    extent on that side; in a corner the depths add as :math:`r_x^2 + r_y^2`.
    Mesh the frame as rectangles around the interior box, every region on one
    side ending on one straight line. The depth follows the material
    points when the nodes move.

    For a plane wave at normal incidence, a frame of thickness :math:`L` in
    a medium with wavelength :math:`\lambda` reflects
    :math:`|R| \approx e^{-2\pi\eta (L/\lambda)/3}` down to a floor of about
    :math:`3.4\times10^{-3}\,\eta\,(\lambda/L)^2`: a larger :math:`\eta`
    lowers the decay term but raises the floor. The default is the best
    choice from four wavelengths on (about :math:`4\lambda` for
    :math:`|R| < 10^{-3}`); :math:`\eta = 2` is best for a frame of three
    wavelengths.

    Attributes:
        regions: The frame's area region names, a string or a sequence of
            strings.
        loss_factor: :math:`\eta`, the loss factor added at the frame's outer
            edge, finite and > 0.

    Raises:
        TypeError: ``regions`` is not a name or a sequence of names.
        ValueError: No region, a region listed twice, or a loss factor that
            is not finite and > 0.
        NonDifferentiableError: The loss factor is traced by JAX.
    """

    regions: tuple[str, ...]
    loss_factor: float

    def __init__(self, regions: Any, *, loss_factor: float = 1.2) -> None:
        """Validate and normalize (see the class docstring)."""
        names = (regions,) if isinstance(regions, str) else regions
        try:
            names = tuple(names)
        except TypeError:
            names = ()
        if not names or not all(isinstance(n, str) for n in names):
            raise TypeError(
                "DampingFrame: regions must be area region names (a string or a "
                f"non-empty sequence of strings), got {regions!r}."
            )
        for name in names:
            if names.count(name) > 1:
                raise ValueError(f"DampingFrame: region {name!r} is listed twice.")
        eta = static_float(loss_factor, "DampingFrame.loss_factor", _FRAME_FIXED)
        if eta <= 0:
            raise ValueError(
                f"DampingFrame.loss_factor must be > 0, got {eta!r}; the default "
                "1.2 absorbs well from four wavelengths of frame."
            )
        object.__setattr__(self, "regions", names)
        object.__setattr__(self, "loss_factor", eta)

    def __repr__(self) -> str:
        return f"DampingFrame({list(self.regions)}, loss_factor={self.loss_factor:g})"


jax.tree_util.register_static(DampingFrame)


@dataclasses.dataclass(frozen=True, init=False)
class Piezoelectric:
    r"""The admittance of piezoelectric electrodes on a 2D section, over frequency.

    Solves linear piezoelectricity in the stress-charge form on a native 2D
    mesh (`gfem.Mesh.from_gmsh` of a triangle-only file), for fields that do
    not vary along :math:`z`. With :math:`e^{+i\omega t}`, the displacement
    :math:`u` (three components) on the mechanical domain and the potential
    :math:`\varphi` on the electromagnetic one solve

    .. math::

        \nabla\cdot T = -\omega^2 \rho\, u, \quad
        \nabla\cdot D = 0, \quad
        T = \hat c\, S - e^T E, \quad D = e\, S + \varepsilon^S E,

    with :math:`S` the strain, :math:`E = -\nabla\varphi` and the loss
    :math:`\hat c = c\,(1 + i/Q_m)` of each material's
    ``mechanical_quality_factor`` (plus a `DampingFrame`'s). The materials are
    the model's: ``stiffness`` :math:`c^E`, ``piezoelectric_tensor`` :math:`e`,
    ``eps_r`` :math:`\varepsilon^S/\varepsilon_0`, ``mass_density``
    :math:`\rho`. Every region of the mechanical domain (every volume not in
    ``excluded["mechanical"]``) needs a stiffness and a density.

    For each terminal in turn, the potential is 1 V on that terminal and 0 V
    on every other terminal and on the ground; every other line is
    traction-free and charge-free, except the ``clamped`` lines
    (:math:`u = 0`). The result holds the admittance matrix per metre along
    :math:`z`, :math:`Y'_{ij}(f)`, the current into terminal :math:`i` per
    volt on terminal :math:`j`, from the reaction of the system weighted by
    the terminal's lifting, and the clamped capacitance :math:`C'_0` of the
    potential alone with :math:`\varepsilon^S` and :math:`u = 0`. The
    device's admittance is :math:`Y'` times the aperture.

    An electrode is a terminal naming a line (a zero-thickness electrode, or
    the shell of a metal excluded from the electromagnetic domain) or an area
    (a meshed electrode that carries displacement and mass, with the
    potential fixed on the whole metal). An area must stay in the
    electromagnetic domain. The handle's electrical model is the model's:
    exclude it from ``"em"`` (no potential in it), keep it as a dielectric, or
    name it in ``ground``. A line that is not named is a free surface; a
    `gfem.PEC` line that is not a terminal is grounded. A `gfem.Absorbing`
    line bordering the mechanical domain raises: absorb acoustic waves with a
    `DampingFrame`.

    Each frequency is one sparse Cholesky factorization shared by every
    terminal, whatever ``SolverOptions.linear_solver`` says of the direct
    solver: on a long strip section it uses half the memory of the
    alternatives. ``SolverOptions.order`` is the order of all four unknowns
    (default 2).

    Example:
        A transducer with a signal electrode against grounded ones, over a
        band::

            problem = gfem.Piezoelectric(
                np.linspace(0.85e9, 0.87e9, 41),
                terminals={"signal": "Al_signal"},
                ground=["Al_ground"],
                clamped="bottom",
                damping_frame=gfem.DampingFrame(["LN_frame", "Si_frame"]),
            )

    Args:
        frequency: One frequency or a sequence of distinct frequencies, Hz.
        terminals: Terminal name -> its line or area name or names, in
            admittance-matrix order. A list of names makes one terminal per
            name.
        ground: The line or area names held at 0 V.
        clamped: The line names where the displacement is zero; each must
            border the mechanical domain.
        damping_frame: A `DampingFrame` naming the absorbing edge regions, or
            ``None``.
        keep_fields: ``True`` keeps the displacement and potential of every
            terminal at every frequency, a list of frequencies those of the
            listed ones, ``False`` (default) none. Each kept (frequency,
            terminal) holds four complex fields: mind the memory.

    Raises:
        TypeError: A name is not a string, or ``damping_frame`` is not a
            `DampingFrame`.
        ValueError: No terminal, a surface in two terminals or in a terminal
            and the ground, a bad frequency, or a ``keep_fields`` frequency
            that is not in ``frequency``.
        NonDifferentiableError: A frequency is traced by JAX.

    `gfem.solve` raises `NotImplementedError` on a 3D or slab mesh, with
    ``linear_solver="petsc"``, with ``frequency_workers`` above 1 and under
    ``comm``; `ValueError` for an area terminal outside the electromagnetic
    domain; `gfem.MissingMaterialPropertyError` for a region of the
    mechanical domain without a stiffness or density.
    """

    frequency: tuple[float, ...]
    terminals: tuple[tuple[str, tuple[str, ...]], ...]
    ground: tuple[str, ...]
    clamped: tuple[str, ...]
    damping_frame: DampingFrame | None
    keep_fields: bool | tuple[float, ...]
    _rule_2d: ClassVar[str] = "gfem._engine.piezoelectric:PiezoelectricRule"
    #: Moved nodes solve (a node map is this problem's design input).
    _moves_nodes_2d: ClassVar[bool] = True

    def __init__(
        self,
        frequency: Any,
        *,
        terminals: Mapping[str, str | list[str] | tuple[str, ...]] | list[str],
        ground: str | list[str] | tuple[str, ...] = (),
        clamped: str | list[str] | tuple[str, ...] = (),
        damping_frame: DampingFrame | None = None,
        keep_fields: bool | float | Sequence[float] = False,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        frequencies = _frequencies(frequency, "Piezoelectric")
        items, grounded = _terminals(terminals, ground, "Piezoelectric")
        fixed = _surfaces(clamped, "clamped") if clamped else ()
        if damping_frame is not None and not isinstance(damping_frame, DampingFrame):
            raise TypeError(
                "Piezoelectric.damping_frame must be a gfem.DampingFrame or None, "
                f"got {type(damping_frame).__name__}; write "
                'damping_frame=gfem.DampingFrame(["LN_frame"]).'
            )
        object.__setattr__(self, "frequency", frequencies)
        object.__setattr__(self, "terminals", tuple(items))
        object.__setattr__(self, "ground", tuple(dict.fromkeys(grounded)))
        object.__setattr__(self, "clamped", tuple(dict.fromkeys(fixed)))
        object.__setattr__(self, "damping_frame", damping_frame)
        object.__setattr__(
            self, "keep_fields", _kept(keep_fields, frequencies, "Piezoelectric")
        )

    @property
    def terminal_names(self) -> tuple[str, ...]:
        """Terminal names, in admittance-matrix order."""
        return tuple(name for name, _ in self.terminals)

    @property
    def kept_frequencies(self) -> tuple[float, ...]:
        """The frequencies whose fields are kept, in ``frequency`` order."""
        if self.keep_fields is True:
            return self.frequency
        if self.keep_fields is False:
            return ()
        return self.keep_fields

    def __repr__(self) -> str:
        n = len(self.frequency)
        band = hertz_band(min(self.frequency), max(self.frequency), 4)
        what = f"{n} frequencies {band}" if n > 1 else band
        terms = ", ".join(f"{k!r}: {list(v)}" for k, v in self.terminals)
        parts = [what, f"terminals={{{terms}}}"]
        if self.ground:
            parts.append(f"ground={list(self.ground)}")
        if self.clamped:
            parts.append(f"clamped={list(self.clamped)}")
        if self.damping_frame is not None:
            parts.append(f"damping_frame={self.damping_frame!r}")
        return f"Piezoelectric({', '.join(parts)})"


jax.tree_util.register_static(Piezoelectric)


def _clamped_components(value: Any) -> tuple[tuple[str, str], ...]:
    """``clamped`` as ``(name, sorted component letters)`` pairs.

    Raises:
        TypeError: Not a name, a sequence of names or a mapping of names to
            component strings.
        ValueError: A component string that is empty or not a subset of
            ``"xyz"``.
    """
    if isinstance(value, Mapping):
        out: dict[str, str] = {}
        for name, letters in value.items():
            if not isinstance(name, str) or not isinstance(letters, str):
                raise TypeError(
                    "MechanicalEigenmode.clamped as a mapping takes surface "
                    "names to component strings, e.g. {'left': 'x', "
                    f"'bottom': 'xyz'}}, got {value!r}."
                )
            if not letters or not set(letters) <= set("xyz"):
                raise ValueError(
                    f"MechanicalEigenmode.clamped[{name!r}] must be a non-empty "
                    f"string of the components 'x', 'y' and 'z', got {letters!r}."
                )
            out[name] = "".join(sorted(set(letters)))
        return tuple(out.items())
    names = _surfaces(value, "MechanicalEigenmode.clamped") if value else ()
    return tuple((name, "xyz") for name in dict.fromkeys(names))


@dataclasses.dataclass(frozen=True, init=False)
class MechanicalEigenmode:
    r"""The mechanical modes of a 2D section nearest a target frequency.

    Finds the source-free solutions of linear elasticity on a native 2D mesh
    (`gfem.Mesh.from_gmsh` of a triangle-only file) for fields
    :math:`u(x, y)\,e^{i\omega t - iqz}` that propagate along :math:`z` with the
    axial wavenumber ``wavenumber`` :math:`q`: complex angular frequencies
    :math:`\omega` and displacements :math:`u` (three components) with

    .. math::

        \nabla\cdot T = -\omega^2 \rho\, u, \quad T = \hat c\, S, \quad
        \hat c = c\,(1 + i/Q_m),

    under :math:`e^{+i\omega t}` and a damping frame's loss (`DampingFrame`),
    exactly as `gfem.Piezoelectric`. The strain carries
    :math:`\partial_z \to -iq`, so that the discrete problem is the linear
    pencil :math:`(\hat K(q) - \omega^2 M)\,x = 0`, with :math:`\hat K(q)`
    Hermitian for a lossless section and :math:`M` real. Each
    mode's quality factor is :math:`Q = \mathrm{Re}\,\omega /
    (2\,\mathrm{Im}\,\omega)`. The materials are the model's: ``stiffness``
    :math:`c`, ``mass_density`` :math:`\rho` and ``mechanical_quality_factor``
    :math:`Q_m`; every region of the mechanical domain (every volume not in
    ``excluded["mechanical"]``) needs a stiffness and a density. Every line
    is traction-free except the ``clamped`` ones.

    The ``n_modes`` modes whose frequency :math:`\mathrm{Re}\,\omega / 2\pi`
    is nearest ``target`` are returned, nearest first. Each is normalized to
    :math:`\tfrac12 (\mathrm{Re}\,\omega)^2 \int \rho\,|u|^2\,dA = 1` J/m,
    with its largest displacement coefficient real and positive. A section
    that is free at :math:`q = 0` has rigid-body motions at :math:`\omega = 0`:
    they are not returned.

    A model whose mechanical-domain materials carry a nonzero piezoelectric
    tensor couples the potential to the displacement: the potential joins the
    unknowns on the electromagnetic domain, as in `gfem.Piezoelectric`, the
    lines and areas of ``ground`` (and any `gfem.PEC` line) hold it at zero,
    and ``result.potential(mode)`` returns it, in V, scaled with the
    displacement. At :math:`q = 0` an ungrounded section's constant potential
    is fixed by pinning one DOF per connected component (the potential is
    defined up to that constant).

    Example:
        Two modes of a strip near 2 GHz at :math:`q = 10^5` rad/m, the strip's
        side faces clamped along :math:`x`::

            problem = gfem.MechanicalEigenmode(
                2e9, n_modes=2, wavenumber=1e5, clamped={"left": "x", "right": "x"}
            )
            result = gfem.solve(model, problem)

    Args:
        target: The frequency to search around, Hz.
        n_modes: How many modes.
        wavenumber: The axial wavenumber :math:`q`, rad/m, finite and real;
            the modes' phase is :math:`e^{-iqz}`, so a mode with
            :math:`q > 0` travels along :math:`+z` where its group velocity
            is positive.
        clamped: Where the displacement is fixed. A line name or a sequence
            of names fixes all three components there; a mapping from a name
            to a string of components of ``"xyz"`` (the lab axes) fixes
            those, for a symmetry plane or a guide with sliding walls. Each
            line must border the mechanical domain.
        ground: Line or area names held at 0 V; only for a model with a
            piezoelectric region.
        damping_frame: A `DampingFrame` naming the absorbing edge regions, or
            ``None``.

    Raises:
        ValueError: ``target`` is not a finite positive number, ``n_modes`` is
            not an integer >= 1, ``wavenumber`` is not a finite real number,
            or a ``clamped`` component string is empty or not made of ``x``,
            ``y`` and ``z``.
        TypeError: A name is not a string, ``clamped`` is not a name, a
            sequence of names or a mapping, or ``damping_frame`` is not a
            `DampingFrame`.
        NonDifferentiableError: ``target`` or ``wavenumber`` is computed
            inside `jax.grad`, `jax.jit` or `jax.vmap` from their arguments;
            pass a fixed value.

    `gfem.solve` raises `NotImplementedError` on a 3D or slab mesh, with
    ``linear_solver="petsc"``, under ``comm`` and on a GPU ``device``;
    `ValueError` for a ``clamped`` or
    ``ground`` name that does not fit the model (``ground`` on a model
    without a piezoelectric region included) and for a `gfem.Absorbing` line
    bordering the mechanical domain (absorb with ``damping_frame``);
    `gfem.MissingMaterialPropertyError` for a region of the mechanical
    domain without a stiffness or density; `gfem.EigenmodeConvergenceError`
    when fewer than ``n_modes`` modes meet ``SolverOptions.eigen_tolerance``.

    ``omega`` (so frequency and Q) differentiates by the stiffness, piezoelectric
    tensor, relative permittivity, density and quality factor of the materials
    (as `gfem.MaterialField` leaves) and by the node coordinates of the mesh
    (`gfem.Mesh.with_nodes`), at every wavenumber, with no further solve.
    `gfem.NonDifferentiableError` names the rest: a cotangent on the
    ``displacement`` or ``potential`` fields, on ``group_velocity`` or on
    ``error``, and a mode that shares its ``cluster`` with another (break the
    symmetry of the section, or hold the value with `jax.lax.stop_gradient`).
    The cluster refusal is raised in the backward pass, inside a host callback,
    so a ``jax.grad`` caller sees it as a ``jax.errors.JaxRuntimeError``
    carrying its message. The cluster is found among the converged modes returned by
    the eigensolver (``n_modes`` plus a guard of two); a degenerate partner
    that it does not return is not detected.
    The result's ``displacement`` maps carry the ``wavenumber`` that
    `gfem.strain` needs.
    """

    target: float
    n_modes: int
    wavenumber: float
    clamped: tuple[tuple[str, str], ...]
    ground: tuple[str, ...]
    damping_frame: DampingFrame | None
    _rule_2d: ClassVar[str] = "gfem._engine.mechanical_eigen:MechanicalEigenmodeRule"
    #: Moved nodes solve (a node map is this problem's design input).
    _moves_nodes_2d: ClassVar[bool] = True

    def __init__(
        self,
        target: float,
        *,
        n_modes: int = 1,
        wavenumber: float = 0.0,
        clamped: str | Sequence[str] | Mapping[str, str] = (),
        ground: str | list[str] | tuple[str, ...] = (),
        damping_frame: DampingFrame | None = None,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        check_fixed(target, "MechanicalEigenmode.target must be a fixed number")
        try:
            value = float(target)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "MechanicalEigenmode.target must be a finite frequency > 0 Hz, "
                f"got {target!r}."
            )
        if (
            isinstance(n_modes, bool)
            or not isinstance(n_modes, (int, np.integer))
            or n_modes < 1
        ):
            raise ValueError(
                f"MechanicalEigenmode.n_modes must be an integer >= 1, got {n_modes!r}."
            )
        q = static_float(
            wavenumber,
            "MechanicalEigenmode.wavenumber",
            "Fix it outside the transformed function: a solve at each wavenumber "
            "is one problem.",
        )
        grounded = _surfaces(ground, "MechanicalEigenmode.ground") if ground else ()
        if damping_frame is not None and not isinstance(damping_frame, DampingFrame):
            raise TypeError(
                "MechanicalEigenmode.damping_frame must be a gfem.DampingFrame or "
                f"None, got {type(damping_frame).__name__}; write "
                'damping_frame=gfem.DampingFrame(["frame"]).'
            )
        object.__setattr__(self, "target", value)
        object.__setattr__(self, "n_modes", int(n_modes))
        object.__setattr__(self, "wavenumber", q)
        object.__setattr__(self, "clamped", _clamped_components(clamped))
        object.__setattr__(self, "ground", tuple(dict.fromkeys(grounded)))
        object.__setattr__(self, "damping_frame", damping_frame)

    def __repr__(self) -> str:
        parts = [hertz_band(self.target, self.target, 4)]
        if self.n_modes != 1:
            parts.append(f"n_modes={self.n_modes}")
        if self.wavenumber:
            parts.append(f"wavenumber={self.wavenumber:g}")
        if self.clamped:
            parts.append(f"clamped={dict(self.clamped)}")
        if self.ground:
            parts.append(f"ground={list(self.ground)}")
        if self.damping_frame is not None:
            parts.append(f"damping_frame={self.damping_frame!r}")
        return f"MechanicalEigenmode({', '.join(parts)})"


jax.tree_util.register_static(MechanicalEigenmode)
