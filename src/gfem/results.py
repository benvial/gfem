"""Results of `gfem.solve`: frozen results with labelled arrays."""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants
from numpy.typing import ArrayLike

from gfem import _objective
from gfem._checks import is_traced, static_float
from gfem._circuit_ports import Circuit, check_circuit, circuit_ports, circuit_z0
from gfem._field import Field, FieldMap, ReducedFields
from gfem._pytree import register_fields
from gfem._sources import DIPOLES, INCIDENT, Source
from gfem._text import hertz, hertz_band
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    import skrf.network
    import xarray

    from gfem._engine.line_mode import ModeSpace
    from gfem._line_section import LineSection
    from gfem._pml import PML
    from gfem._port_section import PortSection
    from gfem.broadband import RationalModel, TimeResponse
    from gfem.problems import Polarization

__all__ = [
    "DCConductionResult",
    "DiffractionOrderResult",
    "DrivenResult",
    "EigenmodeResult",
    "ElectrostaticResult",
    "Field",
    "MechanicalEigenmodeResult",
    "NetworkParameters",
    "PiezoelectricResult",
    "PortModeField",
    "PortModeResult",
    "ReducedFields",
    "SteadyThermalResult",
    "SweepInfo",
    "TransientThermalResult",
]


def _xarray(result: Any) -> Any:
    """The ``xarray`` module for ``result.to_xarray()``, with the install remedy.

    Raises:
        TypeError: ``result`` holds JAX tracers.
        ImportError: xarray is not installed.
    """
    owner = type(result).__name__
    if is_traced(result):
        raise TypeError(
            f"{owner}.to_xarray needs concrete values and cannot run under "
            "jax.jit, jax.grad or jax.vmap; export the result outside the "
            "transformed function."
        )
    try:
        return importlib.import_module("xarray")
    except ImportError:
        raise ImportError(
            f'{owner}.to_xarray needs xarray: pip install "gfem[xarray]".'
        ) from None


def _variable(dims: tuple[str, ...], values: Any, units: str | None = None) -> tuple:
    """An xarray variable ``(dims, values, attrs)`` of host values."""
    attrs = {} if units is None else {"units": units}
    return dims, np.asarray(values), attrs


@dataclasses.dataclass(frozen=True, eq=False)
class ElectrostaticResult:
    """The result of an `Electrostatic` problem.

    Excitation ``i`` puts terminal ``i`` at 1 V and every other terminal
    and the ground at 0 V.

    On a cross-section mesh the values are those of the slab the mesh
    stands for (`gfem.Mesh.section_thickness`): read
    `capacitance_per_length` rather than `capacitance`.

    Attributes:
        capacitance: The Maxwell capacitance matrix in farads, shape
            ``(n_terminals, n_terminals)``, rows and columns in `terminals`
            order: ``capacitance[i, j]`` is the charge on terminal ``i`` in
            excitation ``j``. The diagonal is positive and the off-diagonal
            entries are negative or zero.
        terminals: The terminal names (static labels).
        energy: Electric energy in joules stored in each volume region of
            the solve domain, by region name: an ``(n_terminals,)`` array,
            entry ``i`` for excitation ``i`` at 1 V. Excluded volumes are
            left out. Summed over the regions, excitation ``i`` stores
            ``capacitance[i, i] / 2``. Differentiable with respect to
            ``eps_r`` and the mesh nodes, at the cost of one adjoint solve
            for all terminals together.
        potential: The potential of each terminal's solve, a `Field` by
            terminal name (in `terminals` order): 1 V on that terminal, 0 V on
            every other terminal and the ground. Quantity ``"V"``, in volts,
            on H1 of the solve order. Functionals of the coefficients are
            differentiable like the capacitance (one adjoint solve, shared
            by the terminals).
    """

    capacitance: jax.Array
    terminals: tuple[str, ...]
    energy: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    potential: Mapping[str, Field] = dataclasses.field(default_factory=dict)
    #: The surfaces held at a fixed potential (terminals and ground), which
    #: `gfem.error_estimate` cuts its flux recovery along.
    _fixed: tuple[str, ...] = ()
    #: The slab thickness in m of a cross-section mesh, else ``None``.
    _thickness: float | None = None

    def __post_init__(self) -> None:
        """Hold the potentials as a read-only, ordered mapping."""
        if not isinstance(self.potential, FieldMap):
            fields = FieldMap(self.potential, label="terminal")
            object.__setattr__(self, "potential", fields)

    @property
    def mutual(self) -> jax.Array:
        """The mutual capacitance matrix in farads.

        Off the diagonal, ``mutual[i, j] = -capacitance[i, j]``, the
        capacitance between terminals ``i`` and ``j``; on it, the row sum
        of `capacitance`, the capacitance of terminal ``i`` to ground.
        """
        c = self.capacitance
        return jnp.diag(c.sum(axis=1)) - (c - jnp.diag(jnp.diag(c)))

    @property
    def inverse(self) -> jax.Array:
        """The inverse of `capacitance` in 1/F.

        Entry ``[i, j]`` is the voltage of terminal ``i`` per coulomb on
        terminal ``j``, the others carrying no charge. Without a ground the
        rows of `capacitance` sum to zero and it has no inverse.
        """
        return jnp.linalg.inv(self.capacitance)

    @property
    def capacitance_per_length(self) -> jax.Array:
        """`capacitance` per unit length along a cross-section, in F/m.

        Electrodes of length L have the Maxwell matrix
        ``capacitance_per_length * L``.

        Raises:
            ValueError: The result was not solved on a cross-section mesh.
        """
        if self._thickness is None:
            raise _not_a_section(
                "ElectrostaticResult", "capacitance_per_length", "capacitance"
            )
        return self.capacitance / self._thickness

    def index(self, terminal: str) -> int:
        """Row / column of ``terminal`` in `capacitance`.

        Raises:
            KeyError: ``terminal`` is not a terminal of this result.
        """
        return _terminal_index(self.terminals, terminal)

    def to_xarray(self) -> xarray.Dataset:
        """The capacitance matrix and region energies as an `xarray.Dataset`.

        ``capacitance`` (F, the slab's on a cross-section) has dimensions
        ``("terminal", "excitation")``, both labelled by `terminals`, and so
        has ``capacitance_per_length`` (F/m) on a cross-section; ``energy``
        (J), when present, has ``("region", "excitation")``. Needs the
        ``xarray`` extra: ``pip install "gfem[xarray]"``.

        Raises:
            ImportError: xarray is not installed.
            TypeError: The result holds JAX tracers.
        """
        xr = _xarray(self)
        data = {
            "capacitance": _variable(("terminal", "excitation"), self.capacitance, "F")
        }
        if self._thickness is not None:
            data["capacitance_per_length"] = _variable(
                ("terminal", "excitation"), self.capacitance_per_length, "F/m"
            )
        coords: dict[str, Any] = {
            "terminal": list(self.terminals),
            "excitation": list(self.terminals),
        }
        if self.energy:
            data["energy"] = _variable(
                ("region", "excitation"), list(self.energy.values()), "J"
            )
            coords["region"] = list(self.energy)
        return xr.Dataset(data, coords=coords)

    def __repr__(self) -> str:
        try:
            c = np.asarray(self.capacitance) * 1e15
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"ElectrostaticResult(terminals={list(self.terminals)})"
        width = max(len(t) for t in self.terminals)

        def rows(matrix: np.ndarray) -> str:
            return "\n".join(
                f"  {t:>{width}}: " + "  ".join(f"{v:10.4g}" for v in row)
                for t, row in zip(self.terminals, matrix, strict=True)
            )

        head = f"ElectrostaticResult(terminals={list(self.terminals)}, "
        if self._thickness is None:
            return f"{head}capacitance in fF:\n{rows(c)})"
        # pF/m from fF: 1e-3 for the unit, over the thickness in m
        per_length = c * 1e-3 / self._thickness
        return (
            f"{head}slab capacitance in fF:\n{rows(c)}\n"
            f"capacitance per length in pF/m:\n{rows(per_length)})"
        )


register_fields(ElectrostaticResult, ("capacitance", "energy", "potential"))


def _terminal_index(terminals: Sequence[str], terminal: str) -> int:
    """The position of ``terminal`` in ``terminals``.

    Raises:
        KeyError: ``terminal`` is not one of them.
    """
    try:
        return list(terminals).index(terminal)
    except ValueError:
        raise KeyError(
            f"No terminal {terminal!r}; the terminals are "
            f"{', '.join(repr(t) for t in terminals)}."
        ) from None


def _not_a_section(owner: str, name: str, total: str) -> ValueError:
    """The error of a per-length property read on a mesh that is no cross-section."""
    return ValueError(
        f"{owner}.{name} is defined on a cross-section mesh only "
        "(gfem.Mesh.from_gmsh_cross_section), and this result was solved on a "
        f"3D mesh: read {owner}.{total} instead."
    )


@dataclasses.dataclass(frozen=True, eq=False)
class DCConductionResult:
    """The result of a `gfem.DCConduction` problem.

    Excitation ``i`` puts terminal ``i`` at 1 V and every other terminal
    and the ground at 0 V. `gfem.joule_heating` turns it into a heat
    source for any voltage, current or power drive, with no new solve.

    On a cross-section mesh the values are those of the slab the mesh
    stands for (`gfem.Mesh.section_thickness`): read `resistance_per_length`
    rather than `resistance`.

    Attributes:
        conductance: The conductance matrix in siemens, shape
            ``(n_terminals, n_terminals)``, rows and columns in `terminals`
            order: ``conductance[i, j]`` is the current into terminal ``i``
            in excitation ``j``. Symmetric; the diagonal is positive.
            Differentiable with respect to ``conductivity`` (no adjoint
            solve).
        terminals: The terminal names (static labels).
        potential: The potential of each terminal's solve, a `Field` by
            terminal name (in `terminals` order), in volts, on H1 of the
            solve order over the conductors that carry current.
        dissipation: The power density per tetrahedron at unit voltages,
            shape ``(n_tetrahedra, n_terminals, n_terminals)`` in W/m^3 per
            V^2, in `gfem.Mesh.tetrahedra` order and zero off the
            conductors: with terminal voltages ``V``, element ``e``
            dissipates ``V @ dissipation[e] @ V`` (the exact element mean).
            An internal layout, read through `gfem.joule_heating`.
    """

    conductance: jax.Array
    terminals: tuple[str, ...]
    potential: Mapping[str, Field] = dataclasses.field(default_factory=dict)
    dissipation: jax.Array | None = None
    #: The surfaces held at a fixed potential (terminals and ground).
    _fixed: tuple[str, ...] = ()
    #: The conducting volume regions that carry current, mesh order.
    _regions: tuple[str, ...] = ()
    #: The slab thickness in m of a cross-section mesh, else ``None``.
    _thickness: float | None = None
    #: Whether some terminal or the ground holds 0 V (a power drive needs it).
    _grounded: bool = False

    def __post_init__(self) -> None:
        """Hold the potentials as a read-only, ordered mapping."""
        if not isinstance(self.potential, FieldMap):
            fields = FieldMap(self.potential, label="terminal")
            object.__setattr__(self, "potential", fields)

    @property
    def resistance(self) -> jax.Array:
        """The inverse of `conductance` in ohms; ``1 / G`` for one terminal.

        Entry ``[i, j]`` is the voltage of terminal ``i`` per ampere into
        terminal ``j``, no current into the others. Without a ground the
        rows of `conductance` sum to zero and it has no inverse.
        """
        return jnp.linalg.inv(self.conductance)

    @property
    def resistance_per_length(self) -> jax.Array:
        """`resistance` per unit length along a cross-section, in ohm/m.

        A heater of length L between the two end faces has resistance
        ``resistance_per_length * L``.

        Raises:
            ValueError: The result was not solved on a cross-section mesh.
        """
        if self._thickness is None:
            raise _not_a_section(
                "DCConductionResult", "resistance_per_length", "resistance"
            )
        return self.resistance / self._thickness

    def index(self, terminal: str) -> int:
        """Row / column of ``terminal`` in `conductance`.

        Raises:
            KeyError: ``terminal`` is not a terminal of this result.
        """
        return _terminal_index(self.terminals, terminal)

    def to_xarray(self) -> xarray.Dataset:
        """The conductance matrix as an `xarray.Dataset`.

        ``conductance`` (S) has dimensions ``("terminal", "excitation")``,
        both labelled by `terminals`. Needs the ``xarray`` extra: ``pip
        install "gfem[xarray]"``.

        Raises:
            ImportError: xarray is not installed.
            TypeError: The result holds JAX tracers.
        """
        xr = _xarray(self)
        data = {
            "conductance": _variable(("terminal", "excitation"), self.conductance, "S")
        }
        coords = {"terminal": list(self.terminals), "excitation": list(self.terminals)}
        return xr.Dataset(data, coords=coords)

    def __repr__(self) -> str:
        head = f"DCConductionResult(terminals={list(self.terminals)}"
        try:
            g = np.asarray(self.conductance)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        if g.shape == (1, 1) and self._grounded:
            if self._thickness is None:
                return f"{head}, resistance {1 / g[0, 0]:.6g} ohm)"
            per_length = 1 / (g[0, 0] * self._thickness)
            return f"{head}, resistance per length {per_length:.6g} ohm/m)"
        width = max(len(t) for t in self.terminals)
        rows = [
            f"  {t:>{width}}: " + "  ".join(f"{v:10.4g}" for v in row)
            for t, row in zip(self.terminals, g, strict=True)
        ]
        what = (
            "conductance in S" if self._thickness is None else "slab conductance in S"
        )
        return f"{head}, {what}:\n" + "\n".join(rows) + ")"


register_fields(DCConductionResult, ("conductance", "potential", "dissipation"))


@dataclasses.dataclass(frozen=True, eq=False)
class SteadyThermalResult:
    """The result of a `gfem.SteadyThermal` problem.

    On a cross-section mesh the totals are those of the slab the mesh
    stands for (`gfem.Mesh.section_thickness`): read `power_per_length`
    rather than `power`.

    Attributes:
        temperature: The absolute temperature, a `Field` (quantity ``"T"``,
            in K) on H1 of the solve order over the thermal domain.
        rise: The temperature rise, a `Field` (quantity ``"dT"``, in K):
            `temperature` minus the source-free temperature (the solve with
            the problem's heat sources off and the model's boundary data
            kept). It is linear in the heat sources and scales with their
            power; when every fixed temperature and ambient share one value
            and no `gfem.HeatFlux` is given, it is the temperature minus
            that value.
        heat: The summed power density of the problem's heat sources as
            solved, a `Field` (quantity ``"q"``, in W/m^3) constant on each
            tetrahedron (L2 of order 0) over the thermal domain.
        power: The total power of the heat sources in W (on a cross-section,
            the slab's). A `gfem.HeatFlux` is boundary data, not counted.
    """

    temperature: Field
    rise: Field
    heat: Field
    power: jax.Array
    #: The slab thickness in m of a cross-section mesh, else ``None``.
    _thickness: float | None = None
    #: The sources' face heat per gfem triangle (W/m^2), when a source has
    #: any (the residual of `gfem.heat_flow` reads it), else ``None``.
    _face_heat: jax.Array | None = None

    @property
    def power_per_length(self) -> jax.Array:
        """`power` per unit length along a cross-section, in W/m.

        Raises:
            ValueError: The result was not solved on a cross-section mesh.
        """
        if self._thickness is None:
            raise _not_a_section("SteadyThermalResult", "power_per_length", "power")
        return self.power / self._thickness

    def to_xarray(self) -> xarray.Dataset:
        """The heat sources' total power as an `xarray.Dataset`.

        ``power`` (W, the slab's on a cross-section) is a scalar; on a
        cross-section ``power_per_length`` (W/m) too. The fields are
        `Field` objects: export them with `gfem.Field.to_vtu`. Needs the
        ``xarray`` extra: ``pip install "gfem[xarray]"``.

        Raises:
            ImportError: xarray is not installed.
            TypeError: The result holds JAX tracers.
        """
        xr = _xarray(self)
        data = {"power": _variable((), self.power, "W")}
        if self._thickness is not None:
            data["power_per_length"] = _variable((), self.power_per_length, "W/m")
        return xr.Dataset(data)

    def __repr__(self) -> str:
        regions = list(self.temperature.support)
        try:
            power = float(np.asarray(self.power))
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"SteadyThermalResult(regions={regions})"
        if self._thickness is None:
            total = f"power {power:.6g} W"
        else:
            total = f"power per length {power / self._thickness:.6g} W/m"
        return f"SteadyThermalResult(regions={regions}, {total})"


register_fields(
    SteadyThermalResult, ("temperature", "rise", "heat", "power", "_face_heat")
)


@dataclasses.dataclass(frozen=True, eq=False)
class TransientThermalResult:
    """The result of a `gfem.TransientThermal` problem.

    On a cross-section mesh the totals are those of the slab the mesh
    stands for (`gfem.Mesh.section_thickness`): read `power_per_length`
    rather than `power`.

    Attributes:
        time: The grid times in s, ``(steps + 1,)`` from 0 to the duration.
        traces: Each trace of the problem by name, ``(steps + 1,)`` at
            `time` (a temperature trace in K).
        kept_times: The grid times in s whose fields `temperature` and
            `rise` give, increasing; the final time is always one.
        switch_times: The waveform's start times in s, for the settling
            check of `gfem.rise_time` and `gfem.time_constant`.
        power: The heat sources' total power in W at waveform scale 1 (on
            a cross-section, the slab's).
    """

    time: jax.Array
    traces: dict[str, jax.Array]
    #: The kept rises, ``(len(kept_times), ndof)`` coefficients in K.
    _rises: jax.Array
    #: The source-free temperature, a `Field` (quantity "T", in K).
    _source_free: Field
    power: jax.Array
    kept_times: tuple[float, ...] = ()
    switch_times: tuple[float, ...] = ()
    #: The slab thickness in m of a cross-section mesh, else ``None``.
    _thickness: float | None = None

    def _index(self, time: Any, name: str) -> int:
        what = f"TransientThermalResult.{name}"
        t = static_float(
            time, f"{what} time", "Read a kept time given as a plain number."
        )
        kept = np.asarray(self.kept_times)
        step = float(self.time[1] - self.time[0]) if len(self.time) > 1 else 1.0
        close = np.flatnonzero(np.abs(kept - t) <= 1e-9 * step)
        if not close.size:
            listed = ", ".join(f"{k:g}" for k in self.kept_times)
            raise ValueError(
                f"{what}({t:g}): the result keeps the temperature at {listed} s "
                "only. Read one of those, or solve again with that grid time in "
                "keep_fields (traces record every step)."
            )
        return int(close[0])

    def rise(self, time: float) -> Field:
        """The temperature rise at a kept time, a `Field` (quantity "dT", in K).

        The temperature minus the source-free temperature.

        Raises:
            ValueError: ``time`` is not one of `kept_times`; the message
                lists them.
        """
        k = self._index(time, "rise")
        return dataclasses.replace(
            self._source_free, coefficients=self._rises[k], quantity="dT"
        )

    def temperature(self, time: float) -> Field:
        """The absolute temperature at a kept time, a `Field` (quantity "T", in K).

        Raises:
            ValueError: ``time`` is not one of `kept_times`; the message
                lists them.
        """
        k = self._index(time, "temperature")
        free = self._source_free
        return dataclasses.replace(
            free, coefficients=free.coefficients + self._rises[k]
        )

    @property
    def power_per_length(self) -> jax.Array:
        """`power` per unit length along a cross-section, in W/m.

        Raises:
            ValueError: The result was not solved on a cross-section mesh.
        """
        if self._thickness is None:
            raise _not_a_section("TransientThermalResult", "power_per_length", "power")
        return self.power / self._thickness

    def to_xarray(self) -> xarray.Dataset:
        """The traces over ``time`` (s) as an `xarray.Dataset`.

        One variable per trace, and the scalar ``power`` (W; on a
        cross-section ``power_per_length`` in W/m too). Needs the
        ``xarray`` extra: ``pip install "gfem[xarray]"``.

        Raises:
            ImportError: xarray is not installed.
            TypeError: The result holds JAX tracers.
        """
        xr = _xarray(self)
        data = {name: _variable(("time",), y) for name, y in self.traces.items()}
        data["power"] = _variable((), self.power, "W")
        if self._thickness is not None:
            data["power_per_length"] = _variable((), self.power_per_length, "W/m")
        return xr.Dataset(data, coords={"time": _variable(("time",), self.time, "s")})

    def __repr__(self) -> str:
        steps = len(self.time) - 1
        head = (
            f"TransientThermalResult({steps} steps to {float(self.time[-1]):.6g} s, "
            f"traces={list(self.traces)}"
        )
        try:
            power = float(np.asarray(self.power))
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        if self._thickness is None:
            return f"{head}, power {power:.6g} W)"
        return f"{head}, power per length {power / self._thickness:.6g} W/m)"


register_fields(
    TransientThermalResult, ("time", "traces", "_rises", "_source_free", "power")
)


class PortModeField:
    r"""The field of a port mode on its cross-section, over frequency.

    Peak phasor of the mode carrying 1 W time-averaged power through the
    port (1 W/m through a line port of a 2D mesh), propagating into the solve
    domain as :math:`e^{-\gamma s}`, :math:`s` the distance along the inward
    port normal. Fields follow :math:`e^{+i\omega t}`. Under the port's
    ``polarity`` the field at its point along its direction is real and
    positive at every frequency.

    Under `jax.grad` the values `at` returns differentiate with respect to
    the section's material, law, sheet and `Impedance` inputs (see
    `gfem.PortMode`), not to the node coordinates: `at` raises
    `gfem.NonDifferentiableError` there. The phase follows the gauge of
    `gfem.conventions`, so prefer phase-invariant objectives such as
    :math:`|E|^2`. A field objective costs one 2D solve per mode and
    frequency.

    Attributes:
        quantity: ``"E"``; on a line port, ``"Ez"`` or ``"Hz"``.
        unit: ``"V/m"``; ``"A/m"`` for ``"Hz"``.
        normal: Unit inward port normal, global frame (2 components on a
            line port).
    """

    quantity = "E"
    unit = "V/m"

    def __init__(self, space: ModeSpace, basis: np.ndarray, coords: np.ndarray) -> None:
        """Internal: built by `gfem.solve` (field ``i`` is ``basis @ coords[i]``)."""
        self._parts: tuple[ModeSpace, np.ndarray, np.ndarray] | None = (
            space,
            basis,
            coords,
        )
        self._build: Callable[[], tuple] | None = None
        self._vector: jax.Array | None = None
        self._locate: Callable[[], ModeSpace] | None = None
        self._port = space.section.port.name
        # a space that does not name its field is a planar port's: E in V/m
        self.quantity = getattr(space, "quantity", "E")
        self.unit = getattr(space, "unit", "V/m")

    @classmethod
    def _on_vector(
        cls,
        port: str,
        locate: Callable[[], ModeSpace],
        vector: jax.Array,
        *,
        quantity: str = "E",
        unit: str = "V/m",
    ) -> PortModeField:
        """Internal: the field as a JAX array of its full DOF vector, per frequency.

        A port mode solve traced by JAX returns this vector (``(n_freq,
        n_dof)``, linear in the mode's normalized coordinates); ``locate()``
        gives the port's discrete space, built from the geometry alone,
        when a point is read, and raises when the node coordinates are traced.
        """
        field = object.__new__(cls)
        field._parts = None
        field._build = None
        field._vector = vector
        field._locate = locate
        field._port = port
        field.quantity, field.unit = quantity, unit
        return field

    @classmethod
    def _deferred(
        cls,
        port: str,
        build: Callable[[], tuple[ModeSpace, np.ndarray, np.ndarray]],
        *,
        quantity: str = "E",
        unit: str = "V/m",
    ) -> PortModeField:
        """Internal: a field built by ``build()`` on first use.

        A port mode solve traced by JAX returns its arrays only; the field
        is then solved again from concrete model values when it is read,
        and ``build`` raises when the values are tracers.
        """
        field = object.__new__(cls)
        field._parts = None
        field._build = build
        field._vector = None
        field._locate = None
        field._port = port
        field.quantity, field.unit = quantity, unit
        return field

    def _data(self) -> tuple[ModeSpace, np.ndarray, np.ndarray]:
        if self._parts is None and self._locate is not None:
            self._parts = (self._locate(), np.zeros((0, 0)), np.zeros((0, 0)))
        if self._parts is None:
            if self._build is None:  # pragma: no cover - set by the constructors
                raise RuntimeError("PortModeField has no data.")
            self._parts = self._build()
            self._build = None
        return self._parts

    @property
    def _space(self) -> ModeSpace:
        return self._data()[0]

    def _concrete(self) -> tuple[ModeSpace, np.ndarray, np.ndarray]:
        """The data of a field that carries its basis and coordinates.

        Raises:
            NonDifferentiableError: The field is a traced vector, which holds
                no basis or coordinates.
        """
        if self._vector is not None:
            raise NonDifferentiableError(
                "this port mode field is a traced vector inside jax.grad, jax.jit "
                "or jax.vmap, with no basis or coordinates of its own: read it "
                "with PortModeField.at, or solve outside the transformation."
            )
        return self._data()

    @property
    def _basis(self) -> np.ndarray:
        return self._concrete()[1]

    @property
    def _coords(self) -> np.ndarray:
        return self._concrete()[2]

    @property
    def _section(self) -> PortSection | LineSection:
        return self._space.section

    @property
    def normal(self) -> tuple[float, ...]:
        """Unit inward port normal, global frame."""
        return tuple(float(x) for x in self._section.normal)

    def at(self, points: Any) -> jax.Array:
        """The field at points on the port, ``(n_freq, n_points, 3)`` complex.

        Points off the cross-section's elements (inside metal holes or outside
        the port) give zero. On a line port of a 2D mesh, the scalar field at
        ``(n_points, 2)`` points: ``(n_freq, n_points)``. Points inside a
        perfectly matched layer give the stretched field, not a physical one.

        Args:
            points: ``(n_points, 3)`` (or one ``(3,)``) global coordinates, m,
                on the port plane; ``(n_points, 2)`` on a line port.

        Raises:
            ValueError: A point is not on the port plane (line), or has the
                wrong number of coordinates.
        """
        if self._vector is not None:
            return self._at_vector(points)
        p = np.asarray(points, float)
        single = p.ndim == 1
        p = np.atleast_2d(p)
        dim = len(self.normal)
        if dim == 2:
            if p.shape[-1] != 2:
                raise ValueError(
                    f"points on line port {self._port!r} must be (x, y) pairs in m, "
                    f"got shape {p.shape}."
                )
            columns = self._space.evaluate(self._basis.T, self._local(p))
            u = self._coords @ columns
            return jnp.asarray(u[:, 0] if single else u)
        if p.shape[-1] != 3:
            raise ValueError(f"points must be 3-vectors in m, got shape {p.shape}.")
        local = self._local(p)
        columns = self._space.evaluate(self._basis.T, local)
        e = np.einsum("fk,kpc->fpc", self._coords, columns) @ self._section.frame
        return jnp.asarray(e[:, 0] if single else e)

    def _at_vector(self, points: Any) -> jax.Array:
        """`at` on the traced DOF vector: host-built columns times the vector."""
        assert self._locate is not None  # noqa: S101 - set with the vector
        space: Any = self._locate()
        section = space.section
        p = np.asarray(points, float)
        single = p.ndim == 1
        p = np.atleast_2d(p)
        if len(section.normal) == 2:
            if p.shape[-1] != 2:
                raise ValueError(
                    f"points on line port {self._port!r} must be (x, y) pairs in m, "
                    f"got shape {p.shape}."
                )
            dofs, weights = space.point_columns(_local_points(section, p))
            picked = jnp.asarray(self._vector)[:, dofs]
            u = jnp.einsum("pl,fpl->fp", jnp.asarray(weights), picked)
            return u[:, 0] if single else u
        if p.shape[-1] != 3:
            raise ValueError(f"points must be 3-vectors in m, got shape {p.shape}.")
        local = _local_points(section, p)
        dofs, weights = space.point_columns(local)
        picked = jnp.asarray(self._vector)[:, dofs]
        e = jnp.einsum("pcl,fpl->fpc", jnp.asarray(weights), picked)
        e = e @ jnp.asarray(section.frame)
        return e[:, 0] if single else e

    def _local(self, p: np.ndarray) -> np.ndarray:
        return _local_points(self._section, p)

    def __repr__(self) -> str:
        if self._vector is not None:
            n = f"{len(self._vector)} frequencies, traced"
        else:
            n = (
                "not built"
                if self._parts is None
                else f"{len(self._coords)} frequencies"
            )
        return f"PortModeField({self.quantity}, {self.unit}, {n}, port {self._port!r})"


def _local_points(section: PortSection | LineSection, p: np.ndarray) -> np.ndarray:
    """Plane coordinates of global points (m) on a port's plane (line).

    Raises:
        ValueError: A point is not on the plane.
    """
    local, dist = section.to_local(p)
    tol = 1e-5 * section.diameter
    if (np.abs(dist) > tol).any():
        worst = p[int(np.argmax(np.abs(dist)))]
        where = "line" if len(section.normal) == 2 else "plane"
        raise ValueError(
            f"Point {tuple(float(x) for x in worst)} (m) is not on the {where} "
            f"of wave port {section.port.name!r}."
        )
    return local


@dataclasses.dataclass(frozen=True, eq=False)
class PortModeResult:
    r"""The mode of a wave port over frequency.

    Differentiable on its own (`gfem.PortMode`); on a `DrivenResult`
    the modes follow a design at the wave port's face, as S does: ``gamma``
    and ``Z_PV`` differentiate, a design the port's section does not read gives
    them an exact zero, and ``residual`` is a diagnostic that raises
    `gfem.NonDifferentiableError` beside such a design. The group index and
    dispersion (``n_g``, ``beta_2``, ``beta_3``, ``D``) are not differentiable: a
    gradient through them raises `gfem.NonDifferentiableError` wherever
    ``gamma`` differentiates.

    Fields follow :math:`e^{+i\omega t}`; a mode propagates into the solve
    domain as :math:`e^{-\gamma s}`.

    Attributes:
        port: The wave port's name.
        mode: Which mode (1 = the most propagating).
        frequency: ``(n_freq,)`` Hz.
        gamma: Propagation constant :math:`\gamma = \alpha + i\beta`,
            ``(n_freq,)``, 1/m (:math:`\alpha \ge 0` is the attenuation).
        Z_PV: Complex power-voltage characteristic impedance
            :math:`|V|^2 / (2 P^*)` along the port's ``voltage_path``, with
            :math:`P` the mode's complex power, ``(n_freq,)``, ohm; NaN
            without a voltage path. Real for a lossless line; a wave port's
            reference impedance is its modulus (`gfem.conventions`). For
            the power-current and voltage-current definitions, see
            `gfem.characteristic_impedance`. On a section cut by a perfectly
            matched layer, :math:`P` counts the section outside the layer, so
            :math:`Z_{PV}` is low by the mode's share inside it.
        residual: Not differentiable. Backward error of the mode at each frequency,
            :math:`|A x - \lambda B x| / (|A x| + |\lambda| |B x|)` on the 2D port mode
            problem. It exceeds ``SolverOptions.port_mode_tolerance`` only at
            the formulation's low-frequency accuracy floor, where it is at
            most 10x that of an exact solve.
        solved: ``(n_freq,)`` bool, True where the port's cross-section was
            solved exactly (shared by the wave ports of a shared port
            surface); elsewhere it comes from the reduced port mode model.
        e_field: The mode's electric field, normalized to 1 W (1 W/m for a
            line port of a 2D mesh) through the section outside any
            perfectly matched layer (S does not depend on it); a traced field
            inside `jax.grad`, `jax.jit` or `jax.vmap` (read it with
            `PortModeField.at`); ``None`` on a driven result computed there,
            and for an Hz line port, whose field is ``h_field``.
        h_field: The magnetic field ``Hz`` of an Hz line port's mode,
            normalized to 1 W/m; ``None`` otherwise.
        pml: The model's perfectly matched layer (`gfem.PML`) the mode was
            solved with, or ``None``. In its regions the field is the
            stretched one, not a physical field.
        n_g: Complex group index :math:`(c/i)\,\mathrm{d}\gamma/\mathrm{d}\omega`,
            written :math:`n_g' - in_g''` as `n_eff`:
            :math:`n_g' = c\,\mathrm{d}\beta/\mathrm{d}\omega`,
            :math:`n_g'' = c\,\mathrm{d}\alpha/\mathrm{d}\omega`;
            ``(n_freq,)``. On a lossless section, :math:`c` times stored
            energy over power, with the energy permittivity. Exact frequency
            derivative of `gamma` (no step); its error is the mesh's and the
            truncation's. ``None`` on a driven result's port modes: solve
            `gfem.PortMode` on the same grid for them.
        beta_2: Group-velocity dispersion
            :math:`(1/i)\,\mathrm{d}^2\gamma/\mathrm{d}\omega^2`,
            ``(n_freq,)``, s²/m; real part
            :math:`\mathrm{d}^2\beta/\mathrm{d}\omega^2`, positive for normal
            dispersion; needs more clearance from a truncating layer or wall
            than `n_eff`. See `gfem.conventions` (Group index and
            dispersion). ``None`` on a driven result's port modes.
        beta_3: Third-order dispersion
            :math:`(1/i)\,\mathrm{d}^3\gamma/\mathrm{d}\omega^3`,
            ``(n_freq,)``, s³/m. ``None`` on a driven result's port modes.
    """

    port: str
    mode: int
    frequency: jax.Array
    gamma: jax.Array
    Z_PV: jax.Array
    residual: jax.Array
    solved: jax.Array
    e_field: PortModeField | None = None
    h_field: PortModeField | None = None
    pml: PML | None = None
    n_g: jax.Array | None = None
    beta_2: jax.Array | None = None
    beta_3: jax.Array | None = None

    @property
    def n_eff(self) -> jax.Array:
        r"""Complex effective index :math:`\gamma/(ik_0)`, written :math:`n'-in''`."""
        k0 = 2 * math.pi * self.frequency / scipy.constants.c
        return self.gamma / (1j * k0)

    @property
    def D(self) -> jax.Array | None:  # noqa: N802 - the dispersion parameter's symbol
        r"""Dispersion parameter :math:`-(2\pi c/\lambda^2)\,\beta_2`, s/m².

        :math:`\lambda = c/f`; positive is anomalous; ``D * 1e6`` is in
        ps/(nm km). See `gfem.conventions` (Group index and dispersion).
        ``None`` when `beta_2` is.
        """
        if self.beta_2 is None:
            return None
        c = scipy.constants.c
        return -(2 * math.pi * self.frequency**2 / c) * self.beta_2

    def to_xarray(self) -> xarray.Dataset:
        """The mode's per-frequency values as an `xarray.Dataset`.

        ``gamma`` (1/m), ``n_eff``, ``Z_PV`` (ohm), ``residual`` and
        ``solved`` over the dimension ``frequency`` (Hz), then, where the
        result carries them, ``n_g``, ``beta_2`` (s^2/m), ``beta_3``
        (s^3/m) and ``D`` (s/m^2); the attributes ``port`` and ``mode`` name
        the mode. The field is not exported.
        Needs the ``xarray`` extra: ``pip install "gfem[xarray]"``.

        Raises:
            ImportError: xarray is not installed.
            TypeError: The result holds JAX tracers.
        """
        xr = _xarray(self)
        f = ("frequency",)
        data = {
            "gamma": _variable(f, self.gamma, "1/m"),
            "n_eff": _variable(f, self.n_eff),
            "Z_PV": _variable(f, self.Z_PV, "ohm"),
            "residual": _variable(f, self.residual),
            "solved": _variable(f, self.solved),
        }
        if self.n_g is not None:
            data["n_g"] = _variable(f, self.n_g)
        if self.beta_2 is not None:
            data["beta_2"] = _variable(f, self.beta_2, "s^2/m")
            data["D"] = _variable(f, self.D, "s/m^2")
        if self.beta_3 is not None:
            data["beta_3"] = _variable(f, self.beta_3, "s^3/m")
        return xr.Dataset(
            data,
            coords={"frequency": _variable(f, self.frequency, "Hz")},
            attrs={"port": self.port, "mode": self.mode},
        )

    def __repr__(self) -> str:
        n_f = np.shape(self.frequency)[-1]
        head = f"PortModeResult(port {self.port!r}, mode {self.mode}, {n_f} frequencies"
        try:
            f = np.asarray(self.frequency)
            n = np.asarray(self.n_eff)
            z = np.asarray(self.Z_PV)
            # per frequency: a vmapped result stacks one sweep per batch entry
            solved = np.asarray(self.solved).reshape(-1, n_f).any(axis=0)
            n_g = None if self.n_g is None else np.asarray(self.n_g)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        span = hertz_band(f.min(), f.max(), 4)
        zs = (
            "no voltage_path"
            if np.isnan(z).all()
            else (f"|Z_PV| {np.abs(z).min():.4g}-{np.abs(z).max():.4g} ohm")
        )
        gs = "" if n_g is None else f", n_g {n_g.real.min():.5g}-{n_g.real.max():.5g}"
        return (
            f"{head} {span} ({int(solved.sum())} exact solves), n_eff "
            f"{n.real.min():.5g}-{n.real.max():.5g}{gs}, {zs})"
        )


register_fields(
    PortModeResult,
    (
        "frequency",
        "gamma",
        "Z_PV",
        "residual",
        "solved",
        "n_g",
        "beta_2",
        "beta_3",
    ),
)


@dataclasses.dataclass(frozen=True, eq=False)
class DiffractionOrderResult:
    r"""The diffraction order of a Floquet port over frequency.

    Constants of the solve, read off the lattice and the medium behind the
    Floquet surface: not differentiable. The arrays are NumPy constants when the
    solve runs inside a JAX transformation (concrete inside ``jit``, ``grad``
    and ``vmap``); they are traced if the result itself enters ``jit`` or
    ``vmap`` as an argument.
    Plane waves follow :math:`e^{+i\omega t}`; the order leaves the cell into
    the half-space as :math:`e^{-\gamma s}`, :math:`s` the distance from the
    face (see `gfem.conventions`).

    Attributes:
        port: The Floquet port's name.
        frequency: ``(n_freq,)`` Hz.
        order: The diffraction order (m, n).
        polarization: ``"TE"`` or ``"TM"``.
        surface: The Floquet surface.
        index: The refractive index :math:`\sqrt{\varepsilon_r \mu_r}` of the
            medium behind it.
        wave_vector: ``(n_freq, 3)`` the order's in-plane wave vector
            :math:`\kappa`, rad/m.
        gamma: ``(n_freq,)`` complex, 1/m: :math:`i\beta` where the order
            propagates, :math:`\alpha > 0` where it is evanescent, 0 at
            grazing.
        propagating: ``(n_freq,)`` bool, :math:`\gamma^2 < 0` strictly (False
            at grazing).
        theta: ``(n_freq,)`` rad, the angle from the face's inward normal,
            :math:`\arcsin(|\kappa| / k)`; :math:`\pi/2` at grazing, NaN where
            evanescent.
        phi: ``(n_freq,)`` rad, the azimuth of :math:`\kappa` from
            :math:`\hat e_1` toward :math:`\hat e_2` in the
            lattice frame; NaN where evanescent.
    """

    port: str
    frequency: np.ndarray
    order: tuple[int, int]
    polarization: str
    surface: str
    index: float
    wave_vector: np.ndarray
    gamma: np.ndarray
    propagating: np.ndarray
    theta: np.ndarray
    phi: np.ndarray

    def __post_init__(self) -> None:
        """Hold the order as two ints."""
        object.__setattr__(self, "order", tuple(int(i) for i in self.order))
        object.__setattr__(self, "index", float(self.index))

    def __repr__(self) -> str:
        f = np.asarray(self.frequency)
        span = hertz_band(f.min(), f.max(), 4)
        n = int(np.asarray(self.propagating).sum())
        return (
            f"DiffractionOrderResult(port {self.port!r}, order {self.order}, "
            f"{self.polarization}, surface {self.surface!r}, {len(f)} frequencies "
            f"{span}, propagating at {n})"
        )


register_fields(
    DiffractionOrderResult,
    ("frequency", "wave_vector", "gamma", "propagating", "theta", "phi"),
)


@dataclasses.dataclass(frozen=True, eq=False)
class NetworkParameters:
    r"""Scattering parameters over frequency, labelled by port name.

    Pseudo-waves (Marks and Williams; power waves on a real reference) under
    :math:`e^{+i\omega t}`, each port with its own reference impedance
    ``z_ref``: ``values[f, i, j]`` is the outgoing wave at port ``ports[i]``
    when port ``excited[j]`` is fed a unit wave (1 W incident power on a real
    reference) and every other port is terminated in its reference impedance.
    For a `LumpedPort` the reference impedance is its resistance; for a
    `WavePort` with a ``voltage_path`` its mode's complex ``Z_PV``; an inductance or
    capacitance of the port loads the network in parallel. A port without
    a resistance has no reference impedance: its ``z_ref`` and its row of S
    are NaN. `y`, `z` and `renormalize` are `jax.numpy` functions of S and
    ``z_ref``, so they differentiate like S; `to_touchstone` writes a file.
    The conventions are spelled out in `gfem.conventions`.

    Attributes:
        values: S, complex, shape ``(n_freq, n_ports, n_excited)``.
        z_ref: Reference impedance of each port in ohm, shape ``(n_freq,
            n_ports)``; complex when a `WavePort` has a ``voltage_path``,
            real otherwise.
        frequency: The frequencies in Hz (static labels).
        ports: Port names, the row order of S (the model's port order).
        excited: Names of the excited ports, the column order of S.
    """

    values: jax.Array
    z_ref: jax.Array
    frequency: tuple[float, ...]
    ports: tuple[str, ...]
    excited: tuple[str, ...]

    def __getitem__(self, key: tuple[str, str]) -> jax.Array:
        """``S[to, from]`` over frequency, ``(n_freq,)``: e.g. ``s["p2", "p1"]``.

        Raises:
            KeyError: ``to`` is not a port, or ``from`` is not an excited port.
        """
        to, source = key
        if to not in self.ports:
            raise KeyError(
                f"No port {to!r}; the ports are {', '.join(map(repr, self.ports))}."
            )
        if source not in self.excited:
            raise KeyError(
                f"Port {source!r} was not excited; the excited ports are "
                f"{', '.join(map(repr, self.excited))} (see Driven.excite)."
            )
        return self.values[:, self.ports.index(to), self.excited.index(source)]

    def db(self, to: str, source: str) -> jax.Array:
        r""":math:`20 \log_{10} |S_\mathrm{to,from}|` over frequency, dB."""
        return 20 * jnp.log10(jnp.abs(self[to, source]))

    def y(self) -> jax.Array:
        r"""The admittance matrix in siemens, ``(n_freq, n_ports, n_ports)``.

        Rows and columns in `ports` order. With
        :math:`K = \mathrm{diag}(\sqrt{\mathrm{Re}\, z_\mathrm{ref}} /
        |z_\mathrm{ref}|)` and :math:`G = \mathrm{diag}(z_\mathrm{ref})`,
        :math:`Y = G^{-1} K^{-1} (I + S)^{-1} (I - S) K` (pseudo-waves). Written
        in `jax.numpy`, so it differentiates like S. Entries are NaN where a
        port has no reference impedance.

        Raises:
            ValueError: Not every port was excited, so S is not square.
        """
        s, g, ratio = self._s_g_ratio("y()")
        eye = jnp.broadcast_to(jnp.eye(s.shape[-1]), s.shape)
        return jnp.linalg.solve(eye + s, eye - s) * ratio / g[:, :, None]

    def z(self) -> jax.Array:
        r"""The impedance matrix in ohm, ``(n_freq, n_ports, n_ports)``.

        Rows and columns in `ports` order. With
        :math:`K = \mathrm{diag}(\sqrt{\mathrm{Re}\, z_\mathrm{ref}} /
        |z_\mathrm{ref}|)` and :math:`G = \mathrm{diag}(z_\mathrm{ref})`,
        :math:`Z = K^{-1} (I - S)^{-1} (I + S) K G` (pseudo-waves). Written in
        `jax.numpy`, so it differentiates like S. Entries are NaN where a port
        has no reference impedance.

        Raises:
            ValueError: Not every port was excited, so S is not square.
        """
        s, g, ratio = self._s_g_ratio("z()")
        eye = jnp.broadcast_to(jnp.eye(s.shape[-1]), s.shape)
        return jnp.linalg.solve(eye - s, eye + s) * ratio * g[:, None, :]

    def renormalize(self, z_new: Any) -> NetworkParameters:
        """The same network under new reference impedances.

        Pseudo-wave renormalization (`gfem.conventions`): the network (its Z
        matrix) is unchanged, only the waves that describe it change.
        Written in `jax.numpy`, so it differentiates with respect to S and ``z_new``.

        Args:
            z_new: The new reference impedance in ohm: one value for every
                port, one per port in `ports` order, or an array
                ``(n_freq, n_ports)``. May be complex, with a positive real
                part.

        Returns:
            `NetworkParameters` with ``z_ref`` set to ``z_new``, every port
            excited, columns in `ports` order.

        Raises:
            ValueError: Not every port was excited, ``z_new`` has the wrong
                shape, or a real part is not positive.
        """
        s = self._square("renormalize()")
        n_f, n = s.shape[0], s.shape[1]
        if not is_traced(z_new):
            arr = np.asarray(z_new)
            if (arr.ndim == 1 and arr.shape != (n,)) or arr.ndim > 2:
                raise ValueError(
                    "renormalize takes one reference impedance, one per port "
                    f"({n} ports: {', '.join(map(repr, self.ports))}) or an array "
                    f"({n_f}, {n}); got shape {arr.shape}."
                )
            if not (np.real(arr) > 0).all():
                raise ValueError(
                    "Reference impedances need a positive real part "
                    f"(a resistance), got {z_new!r}."
                )
        new = jnp.broadcast_to(jnp.asarray(z_new), (n_f, n))
        z = jnp.asarray(self.z_ref, dtype=complex)
        zn = new.astype(complex)
        # pseudo-waves a = k (V + z I), b = k (V - z I), k = sqrt(Re z) / |z|:
        # a' = c ((z + z') a + (z - z') b), b' = c ((z - z') a + (z + z') b) per
        # port with c = k' / (2 k z), so S' = C (U + P S)(P + U S)^-1 C^-1.
        p, u = z + zn, z - zn
        num = _diag(u) + p[:, :, None] * s
        den = _diag(p) + u[:, :, None] * s
        s0 = jnp.swapaxes(
            jnp.linalg.solve(jnp.swapaxes(den, 1, 2), jnp.swapaxes(num, 1, 2)), 1, 2
        )
        c = _pseudo(zn) / (_pseudo(z) * z)
        return NetworkParameters(
            values=s0 * c[:, :, None] / c[:, None, :],
            z_ref=new,
            frequency=self.frequency,
            ports=self.ports,
            excited=self.ports,
        )

    def to_sdict(
        self,
        *,
        circuit: Circuit,
        ports: Mapping[str, str] | None = None,
        z0: Any = None,
    ) -> dict[tuple[str, str], jax.Array]:
        r"""S between circuit ports as a SAX dictionary (an ``SDict``).

        A plain ``dict`` keyed ``(input, output)`` by circuit port name:
        ``sd["o1", "o2"]`` is :math:`S_{21}`, the wave out of the port named
        ``"o2"`` when the port named ``"o1"`` is fed, over `frequency` in its
        order, shape ``(n_freq,)``. It holds every pair of exposed ports,
        zeros included, because SAX reads a missing key as zero; gfem does not
        import SAX. Written in `jax.numpy`, so it differentiates like S.

        The exposed ports' sub-network comes first: their rows and columns
        of S, with their ``z_ref``. The other ports terminate the device as
        in the solve (a wave port absorbs its mode, a lumped port is its own
        R, L and C). Then ``circuit`` sets the conventions:

        - ``"optical"``: SAX's photonic models are :math:`e^{-i\omega t}`,
          so the dictionary holds :math:`\overline{S}`, the same fields in
          that convention. S keeps its normalization (unit modal power on a
          wave port).
        - ``"rf"``: SAX's RF models are :math:`e^{+i\omega t}`, as gfem, and
          share one real reference impedance, which an ``SDict`` does not
          carry. S is renormalized (`renormalize`) from the ports' own
          ``z_ref`` to ``z0``, 50 ohm unless given, SAX's RF default.

        Only names reach this method, not the ports: the sign of a wave
        port's row and column of S is physical only when the port has a
        polarity reference (``WavePort(polarity=)``) or a ``voltage_path``.
        Otherwise the mode's default sign depends on the mesh, and so does
        the interference a circuit computes from it.

        Args:
            circuit: ``"optical"`` or ``"rf"``, the kind of SAX circuit
                that uses the dictionary; never inferred, because the wrong
                one still composes, with wrong interference.
            ports: gfem port name -> circuit port name, e.g.
                ``{"p1": "o1", "p2": "o2"}``, in the order the keys are
                listed. A circuit port name is a Python identifier, or
                ``name@mode`` (SAX multimode) for every port or none. Ports
                left out are terminated, not exposed. ``None`` exposes every
                port under its own name.
            z0: For ``"rf"``, the reference impedance S is renormalized
                to, ohm, one positive real scalar; ``None`` means 50 ohm. It
                may be traced. Refused for ``"optical"``.

        Returns:
            ``{(input, output): S over frequency}`` for every pair of
            exposed ports.

        Raises:
            ValueError: ``circuit`` is neither kind; a name in ``ports`` is
                not a port, a circuit port name is invalid or repeated;
                an exposed port was not excited (its column of S is
                unknown); ``z0`` is given for ``"optical"`` or is not one
                positive real value; or, for ``"rf"``, an exposed port has no
                reference impedance (checked on concrete values).
        """
        check_circuit(circuit, "the time convention and the normalization of S")
        exposed = circuit_ports(ports, self.ports)
        sub = self._subnetwork(tuple(exposed), "to_sdict()")
        if circuit == "rf":
            z0 = circuit_z0(z0)
            if not is_traced(sub.z_ref):
                _require_reference(sub, "An RF circuit renormalizes S to z0")
            sub = sub.renormalize(z0)
        elif z0 is not None:
            raise ValueError(
                "z0 is the reference impedance of an RF circuit; an optical "
                "circuit keeps S as solved (unit modal power), so drop z0= or "
                "use circuit='rf'."
            )
        values = jnp.asarray(sub.values)
        if circuit == "optical":
            # gfem's e^{+i w t} phasors -> SAX's photonic e^{-i w t}
            values = jnp.conj(values)
        names = list(exposed.values())
        return {
            (c_in, c_out): values[:, i_out, i_in]
            for i_in, c_in in enumerate(names)
            for i_out, c_out in enumerate(names)
        }

    def to_skrf(
        self, *, ports: Sequence[str] | None = None, name: str | None = None
    ) -> skrf.network.Network:
        r"""S as a scikit-rf `~skrf.network.Network`, on the ports' own references.

        scikit-rf uses gfem's :math:`e^{+i\omega t}` and pseudo-waves on a
        complex reference per port and frequency (``s_def="pseudo"``), so S
        and `z_ref` pass unchanged: the network's ``z``, ``y`` and
        ``renormalize`` are exact. Rows are put in increasing frequency, as
        scikit-rf expects, whatever the order of `frequency`, and the port
        names are the network's ``port_names``. Not differentiable: S is
        copied to NumPy. Needs the ``skrf`` extra:
        ``pip install "gfem[skrf]"``.

        For one real reference, renormalize the network itself, e.g.
        ``nw.renormalize(50.0)``. scikit-rf's ``Network.a`` (ABCD) assumes
        power waves: it is wrong on a complex pseudo-wave reference and
        right after renormalizing to a real one.

        Args:
            ports: The ports to export, in order; the other ports terminate
                the device as in the solve, so a model with an internal
                lumped load (a port with only L or C) exports the rest.
                ``None`` exports every port.
            name: The network's name.

        Returns:
            The network of the chosen ports.

        Raises:
            ImportError: scikit-rf is not installed.
            TypeError: The values are computed inside `jax.grad`, `jax.jit`
                or `jax.vmap`; export the result outside them.
            ValueError: ``ports`` names a port that does not exist or repeats
                one, a chosen port was not excited, or a chosen port has no
                reference impedance (scikit-rf could not label S with it).
        """
        if is_traced(self.values) or is_traced(self.z_ref):
            raise TypeError(
                "NetworkParameters.to_skrf needs concrete values and cannot run "
                "under jax.jit, jax.grad or jax.vmap; export the result outside "
                "the transformed function."
            )
        try:
            rf = importlib.import_module("skrf")
        except ImportError:
            raise ImportError(
                'NetworkParameters.to_skrf needs scikit-rf: pip install "gfem[skrf]".'
            ) from None
        sub = self._subnetwork(_port_selection(ports, self.ports), "to_skrf()")
        _require_reference(
            sub,
            "scikit-rf labels S with it",
            " A network of wave ports without voltage_path can be written under "
            "a nominal label with to_touchstone(z0=).",
        )
        order = np.argsort(self.frequency, kind="stable")
        network = rf.Network(
            f=np.asarray(self.frequency)[order],
            s=np.asarray(sub.values)[order],
            z0=np.asarray(sub.z_ref)[order],
            s_def="pseudo",
        )
        if name is not None:
            network.name = name
        # an attribute, because scikit-rf 1.x ignores the port_names keyword
        network.port_names = list(sub.ports)
        return network

    def _subnetwork(self, ports: tuple[str, ...], what: str) -> NetworkParameters:
        """The network of ``ports`` with the other ports terminated as solved.

        Rows and columns of S in ``ports`` order, with their ``z_ref``.

        Raises:
            ValueError: A port in ``ports`` was not excited.
        """
        missing = [p for p in ports if p not in self.excited]
        if missing:
            raise ValueError(
                f"{what} needs the column of S of every exposed port, but "
                f"{', '.join(map(repr, missing))} was not excited (the excited "
                f"ports are {', '.join(map(repr, self.excited))}). Solve with "
                "Driven(excite=None), or leave the port out of ports=."
            )
        rows = np.array([self.ports.index(p) for p in ports])
        cols = np.array([self.excited.index(p) for p in ports])
        return NetworkParameters(
            values=jnp.asarray(self.values)[:, rows][:, :, cols],
            z_ref=jnp.asarray(self.z_ref)[:, rows],
            frequency=self.frequency,
            ports=ports,
            excited=ports,
        )

    def to_xarray(self) -> xarray.Dataset:
        """S and the reference impedances as an `xarray.Dataset`.

        ``s`` has dimensions ``("frequency", "port", "excited")`` (as
        `values`) and ``z_ref`` (ohm) ``("frequency", "port")``; the
        coordinates are the frequencies (Hz) and the port names. Needs the
        ``xarray`` extra: ``pip install "gfem[xarray]"``.

        Raises:
            ImportError: xarray is not installed.
            TypeError: The parameters hold JAX tracers.
        """
        xr = _xarray(self)
        return xr.Dataset(
            {
                "s": _variable(("frequency", "port", "excited"), self.values),
                "z_ref": _variable(("frequency", "port"), self.z_ref, "ohm"),
            },
            coords={
                "frequency": _variable(("frequency",), self.frequency, "Hz"),
                "port": list(self.ports),
                "excited": list(self.excited),
            },
        )

    def to_touchstone(self, path: str | Path, z0: float | None = None) -> Path:
        """Write S to a Touchstone 1.1 file (``.sNp``, real/imaginary, Hz).

        Not collective: across ranks a script calls it on rank 0.

        A Touchstone 1.1 file has one real reference impedance for every
        port. Without ``z0`` the file is labelled with `z_ref`, which must
        then be one real value for every port and frequency: gfem never
        labels a port with an impedance S was not computed for. With
        ``z0``, S is first renormalized to it (`renormalize`). Port names
        and the conventions are written as comments. Rows are written in
        increasing frequency, as the format requires, whatever the order of
        `frequency`.

        A network in which no port has a reference impedance (every port a
        `WavePort` without ``voltage_path``, S finite) is written with an
        explicit ``z0`` only: S, normalized to unit modal power, is written
        unchanged, and a comment line says that ``z0`` is a nominal label.

        Args:
            path: The file, named ``*.s<n_ports>p``.
            z0: The reference impedance to renormalize to, ohm (real, > 0);
                the nominal label of a network without reference impedances.

        Returns:
            The path written.

        Raises:
            ValueError: The file name does not match the port count, not
                every port was excited, some but not all ports have no
                reference impedance (or a row of S is NaN; a `FloquetPort`
                has none where its order does not propagate), or ``z0`` is not
                given and `z_ref` is not one real value or no port has one.
            TypeError: The values are computed inside `jax.grad`, `jax.jit`
                or `jax.vmap`; write the file from a result outside them.
        """
        path = Path(path)
        n = len(self.ports)
        suffix = f".s{n}p"
        if path.suffix.lower() != suffix:
            raise ValueError(
                f"A {n}-port Touchstone file is named *{suffix}; got {path.name!r}."
            )
        if is_traced(self.values) or is_traced(self.z_ref):
            raise TypeError(
                "to_touchstone needs concrete values and cannot run under jax.jit, "
                "jax.grad or jax.vmap; write the file outside the transformed "
                "function."
            )
        s = np.asarray(self._square("to_touchstone()"))
        z_ref = np.asarray(self.z_ref)
        nan = np.isnan(z_ref).any(axis=0)
        missing = [p for p, bad in zip(self.ports, nan, strict=True) if bad]
        # a modal network: every port without a reference at every frequency
        # (a Floquet port's is NaN only where its order does not propagate)
        if np.isnan(z_ref).all() and np.isfinite(s).all():
            if z0 is None:
                raise ValueError(
                    "No port has a reference impedance (z_ref is NaN: wave ports "
                    "without voltage_path), so S is normalized to unit modal power "
                    "and has no impedance to label the file with. Pass z0= (e.g. "
                    "z0=50.0) to write S unchanged under that nominal label, or "
                    "give each WavePort a voltage_path."
                )
            path.write_text(_touchstone(self, s, _touchstone_z0(z0), modal=True))
            return path
        if missing:
            where = np.isnan(z_ref[:, nan]).any(axis=1)
            freqs = ", ".join(
                hertz(float(f))
                for f, bad in zip(np.asarray(self.frequency), where, strict=True)
                if bad
            )
            raise ValueError(
                f"Port {', '.join(map(repr, missing))} has no reference impedance "
                "(z_ref is NaN): a LumpedPort without a resistance, a WavePort "
                "without a voltage_path, or a FloquetPort whose order is "
                f"evanescent at {freqs}. S cannot be labelled or renormalized "
                "without one; give the port a resistance or a voltage_path, or "
                "for a FloquetPort drop the port or narrow the grid."
            )
        if z0 is None:
            first = z_ref.flat[0]
            if not (np.all(z_ref == first) and np.imag(first) == 0):
                found = np.unique(np.round(z_ref, 12))
                raise ValueError(
                    "A Touchstone 1.1 file has one real reference impedance for "
                    f"every port, but z_ref takes the values {found.tolist()} ohm. "
                    "Pass z0= (e.g. z0=50.0) to renormalize S to one reference "
                    "first."
                )
            z0 = float(np.real(first))
        else:
            z0 = _touchstone_z0(z0)
            s = np.asarray(self.renormalize(z0).values)
        path.write_text(_touchstone(self, s, z0))
        return path

    def fit_rational(
        self,
        *,
        tolerance: float = 1e-3,
        max_poles: int | None = None,
        weights: ArrayLike | None = None,
        enforce_passivity: bool = True,
        z0: float | Sequence[float] | None = None,
    ) -> RationalModel:
        r"""Fit a rational model to S over the network's frequencies.

        :math:`S(f) = D + \sum_n R_n / (2\pi i f - p_n)` with poles shared by
        every element (`gfem.RationalModel`), by fast relaxed vector fitting:
        real poles and conjugate pairs, a fitted constant :math:`D` and no
        term proportional to frequency. The order rises in steps of 4 poles
        until the largest per-element RMS of :math:`|S_\mathrm{model} -
        S_\mathrm{data}|` over the data is at most ``tolerance``; then each
        pole (or pair) whose removal, with the residues refitted, raises that
        error by less than ``tolerance``/10 is removed. A zero-frequency sample in the
        network is used like any other; none is added. A long line needs
        about one pole pair per phase wrap in the band (two when reflections
        bounce on it).

        By default the model is then made passive (no singular value of S
        above 1 at any frequency, the bounded-real condition on a real
        reference): the poles are kept and the residues changed by the least
        weighted change of S on the data, until the Hamiltonian test finds no
        passivity violation; the singular values of :math:`D` are clipped to
        :math:`1 - 10^{-4}`. The model's ``fit_error`` is the error before
        this, ``error`` after. The test's cost grows as :math:`(2NP)^3` for
        :math:`N` poles and :math:`P` ports.

        The model has one constant real reference impedance per port, fixed
        here: a `z_ref` real and constant over frequency is kept, each port
        at its mean real value (real and constant to a relative
        :math:`10^{-9}`, as a TEM wave port's :math:`Z_{PV}` is up to the
        mode solve's round-off); otherwise ``z0`` is needed and S is
        renormalized to it first; a network in which no port has a reference
        impedance (wave ports without ``voltage_path``) is fitted unchanged
        under ``z0`` as a nominal label. Not collective: deterministic host
        work, the same on every rank. Not differentiable.

        Args:
            tolerance: Target of the largest per-element RMS error, > 0.
            max_poles: The most poles tried, from 4 to the number of
                frequencies minus one; default ``min(100, n_freq - 1)``.
            weights: Positive per-frequency weights ``(n_freq,)``, in the
                network's frequency order; None for uniform. The reported
                errors are unweighted.
            enforce_passivity: Make the model passive. False fits only: the
                model still reports its ``passivity_violations`` and
                `gfem.RationalModel.passive`.
            z0: Reference impedance to fit on, ohm: one real value > 0 or one
                per port in `ports` order; the nominal label of a network
                without reference impedances.

        Returns:
            The model, with its errors on the data.

        Raises:
            ValueError: An argument is out of range, fewer than 6 distinct
                frequencies, a negative frequency, S not finite or not every
                port excited, or the reference impedance cannot be resolved
                (complex or frequency-dependent without ``z0``, none without
                ``z0``, or some ports without one).
            NonDifferentiableError: S, `z_ref`, ``z0``, ``weights`` or an
                option is traced by JAX, or the fit is staged by
                ``jax.jit``.
            PassivityEnforcementError: Passivity violations remain after
                the enforcement's 30 iterations; the message names them.

        Warns:
            RationalFitNotConvergedWarning: No order up to ``max_poles``
                reached ``tolerance``; the most accurate model is returned.
            NonPassiveDataWarning: Making the model passive changed S by
                more than ``tolerance``: the data are not passive on this
                reference.
        """
        from gfem._broadband._vector_fit import fit_rational

        return fit_rational(
            self,
            tolerance=tolerance,
            max_poles=max_poles,
            weights=weights,
            enforce_passivity=enforce_passivity,
            z0=z0,
        )

    def impulse_response(
        self,
        *,
        mode: Literal["low_pass", "band_pass"] = "low_pass",
        kaiser_beta: float = 6.0,
        n_time: int | None = None,
        z0: float | Sequence[float] | None = None,
    ) -> TimeResponse:
        r"""The impulse response of S, a `gfem.TimeResponse` (dimensionless).

        Low pass (default), as a network analyzer's low-pass mode: S must be
        on a harmonic grid, :math:`f_k = k\,\Delta f` for :math:`k = 1..N`
        (:math:`N \ge 3`, checked to :math:`10^{-6}`; a 0 Hz point, if
        present, is the DC value). Without one, DC is
        :math:`\mathrm{Re}(15 S_1 - 6 S_2 + S_3)/10`, the even polynomial
        through the first three points. S times the upper half of a Kaiser
        window of parameter :math:`\beta` goes through a real inverse FFT,
        with no conjugation under :math:`e^{+i\omega t}`, so a delay is
        positive time. The result is divided by the window kernel's peak:
        a frequency-independent reflection :math:`\Gamma` peaks at
        :math:`\Gamma`. The record is two-sided and periodic,
        :math:`t_m = (m - n_t/2)/(n_t\,\Delta f)`, of length
        :math:`T = 1/\Delta f`: later responses alias, so the record should
        be at least 5 times the settling time. Resolution is set by the
        span and :math:`\beta`; ``n_time`` only interpolates.

        Band pass: any evenly spaced grid (no DC needed, e.g. a waveguide
        above cutoff), a full Kaiser window over the band, a complex
        inverse FFT, and the magnitude :math:`|h(t)|`, kernel-peak
        normalized, on the same two-sided axis; the sign of a reflection is
        lost.

        Written in `jax.numpy`: differentiates with respect to S and ``z0``.

        Args:
            mode: ``"low_pass"`` or ``"band_pass"``.
            kaiser_beta: The Kaiser window's :math:`\beta`, finite,
                :math:`\ge 0`: 0 (no window, the sharpest edges, 13 dB
                sidelobes), 6 (the usual default), 13 (the smoothest).
            n_time: Samples in the record, even and :math:`\ge 2N`; default
                the smallest power of two :math:`\ge 20N`.
            z0: Reference impedance to renormalize S to first, ohm (real,
                :math:`> 0`; one value or one per port in `ports` order;
                needs every port excited). Without it, `z_ref` must be one
                real value per port over frequency, to a relative
                :math:`10^{-9}` (each port then on its mean real value), or
                NaN for every port (the response of S on unit modal power);
                when `z_ref` is traced by JAX this cannot be checked, and
                ``z0`` is needed.

        Raises:
            ValueError: The grid is not harmonic (low pass) or not evenly
                spaced (band pass), and the message gives the grid to solve
                on and the explicit resampling through a rational model; an
                argument is out of range; or `z_ref` is complex, frequency
                dependent, NaN for some ports only or traced by JAX (pass
                ``z0=``).
            NonDifferentiableError: ``mode``, ``kaiser_beta`` or ``n_time``
                is traced by JAX.
        """
        from gfem._broadband import _time

        return _time.impulse_response(
            self, mode=mode, kaiser_beta=kaiser_beta, n_time=n_time, z0=z0
        )

    def step_response(
        self,
        *,
        kaiser_beta: float = 6.0,
        n_time: int | None = None,
        z0: float | Sequence[float] | None = None,
    ) -> TimeResponse:
        r"""The step response of S, a `gfem.TimeResponse` (dimensionless).

        The low-pass impulse response (`impulse_response`, before its
        normalization) summed over the whole two-sided record, so it starts
        near 0 and settles at :math:`S(0)`. The reflected step at a port is
        what a TDR shows, and `gfem.TimeResponse.impedance` its impedance
        profile; ``step["p2", "p1"]`` is the transmitted step (TDT), a wave
        ratio (the voltage ratio is
        :math:`S_{21} \sqrt{Z_{0,2}/Z_{0,1}}`). The 10-90 % rise of an edge
        is about :math:`0.99/f_\mathrm{max}` at :math:`\beta = 6`. Needs a
        harmonic grid (see `impulse_response`): a step needs DC, so there is
        no band-pass step.

        Written in `jax.numpy`: differentiates with respect to S and ``z0``.

        Args:
            kaiser_beta: The Kaiser window's :math:`\beta` (see
                `impulse_response`).
            n_time: Samples in the record, even and :math:`\ge 2N`; default
                the smallest power of two :math:`\ge 20N`.
            z0: Reference impedance to renormalize S to first (see
                `impulse_response`).

        Raises:
            ValueError: As `impulse_response` in low pass.
            NonDifferentiableError: ``kaiser_beta`` or ``n_time`` is traced
                by JAX.
        """
        from gfem._broadband import _time

        return _time.step_response(self, kaiser_beta=kaiser_beta, n_time=n_time, z0=z0)

    def _square(self, what: str) -> jax.Array:
        """S with every port as a column, in `ports` order."""
        missing = [p for p in self.ports if p not in self.excited]
        if missing:
            raise ValueError(
                f"{what} needs the full scattering matrix, but only "
                f"{', '.join(map(repr, self.excited))} were excited, not "
                f"{', '.join(map(repr, missing))}. Solve with Driven(excite=None) "
                "to excite every port."
            )
        order = [self.excited.index(p) for p in self.ports]
        return jnp.asarray(self.values)[:, :, order]

    def _s_g_ratio(self, what: str) -> tuple[jax.Array, jax.Array, jax.Array]:
        s = self._square(what)
        g = jnp.asarray(self.z_ref, dtype=complex)
        k = _pseudo(g)
        # K^-1 M K: entry (i, j) times k_j / k_i
        return s, g, k[:, None, :] / k[:, :, None]

    def __repr__(self) -> str:
        return (
            f"NetworkParameters(ports={list(self.ports)}, "
            f"excited={list(self.excited)}, {len(self.frequency)} frequencies)"
        )


register_fields(NetworkParameters, ("values", "z_ref"))


def _pseudo(z: jax.Array) -> jax.Array:
    """The pseudo-wave factor ``sqrt(Re z) / |z|`` of reference impedances ``z``."""
    return jnp.sqrt(jnp.real(z)) / jnp.abs(z)


def _diag(v: jax.Array) -> jax.Array:
    """``(n_freq, n)`` -> stacked diagonal matrices ``(n_freq, n, n)``."""
    return v[:, :, None] * jnp.eye(v.shape[-1])


def _require_reference(net: NetworkParameters, why: str, also: str = "") -> None:
    """Raise if a port of ``net`` has no reference impedance (a NaN ``z_ref``).

    ``why`` says what needs one; ``also`` is one more remedy sentence.
    """
    nan = np.isnan(np.asarray(net.z_ref)).any(axis=0)
    missing = [p for p, bad in zip(net.ports, nan, strict=True) if bad]
    if missing:
        raise ValueError(
            f"Port {', '.join(map(repr, missing))} has no reference impedance "
            "(z_ref is NaN): a LumpedPort without a resistance, a WavePort "
            "without a voltage_path, or a FloquetPort whose order is evanescent. "
            f"{why} and needs one: give the port a resistance or a voltage_path, "
            f"or leave it out of ports= to keep it inside the device.{also}"
        )


def _port_selection(ports: Sequence[str] | None, names: tuple[str, ...]) -> tuple:
    """``ports`` checked: port names in the given order, every port if ``None``.

    Raises:
        TypeError: ``ports`` is one string, not a sequence of names.
        ValueError: ``ports`` is empty, repeats a port or names one that
            does not exist.
    """
    if ports is None:
        return names
    if isinstance(ports, str):
        raise TypeError(
            f"ports= is a sequence of port names, e.g. [{ports!r}]; got the "
            f"string {ports!r}."
        )
    chosen = tuple(ports)
    listed = ", ".join(map(repr, names))
    if not chosen:
        raise ValueError(f"ports= selects no port; the ports are {listed}.")
    unknown = [p for p in chosen if p not in names]
    if unknown:
        raise ValueError(
            f"No port {', '.join(map(repr, unknown))}; the ports are {listed}."
        )
    repeated = sorted({p for p in chosen if chosen.count(p) > 1})
    if repeated:
        raise ValueError(f"Port {', '.join(map(repr, repeated))} is listed twice.")
    return chosen


def _per_terminal(
    value: Any, n: int, name: str, terminals: tuple[str, ...]
) -> jax.Array:
    """``value`` as one entry per terminal: a scalar is repeated."""
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (n,))
    if arr.shape != (n,):
        raise ValueError(
            f"{name} takes one value or one per terminal ({n} terminals: "
            f"{', '.join(map(repr, terminals))}); got shape {arr.shape}."
        )
    return arr


def _touchstone_z0(z0: object) -> float:
    """``to_touchstone``'s ``z0``, checked: one real value > 0."""
    if (
        isinstance(z0, (bool, complex, np.complexfloating))
        or not isinstance(z0, (int, float, np.integer, np.floating))
        or not math.isfinite(z0)
        or z0 <= 0
    ):
        raise ValueError(
            "to_touchstone z0 must be one real reference impedance > 0 "
            f"ohm, got {z0!r}; use renormalize() for complex references."
        )
    return float(z0)


def _touchstone(
    net: NetworkParameters, s: np.ndarray, z0: float, *, modal: bool = False
) -> str:
    """Touchstone 1.1 text of ``s`` (port order) with reference ``z0``.

    ``modal``: no port has a reference impedance, and ``z0`` is a label.
    """
    n = len(net.ports)
    lines = [
        f"! gfem {version('gfem')}: S-parameters, {n} port(s), {len(s)} frequencies",
        "! Power waves (Kurokawa), time convention exp(+j omega t); |a|^2 is the",
        "! time-averaged incident power in W",
        *(f"! Port[{i}] = {name}" for i, name in enumerate(net.ports, start=1)),
    ]
    if modal:
        lines.append(
            "! No port has a reference impedance (wave ports without "
            "voltage_path): S is normalized to unit modal power and was not "
            "renormalized; R is a nominal label"
        )
    lines.append(f"# Hz S RI R {z0:.12g}")

    def pairs(values: np.ndarray) -> list[str]:
        return [f"{v.real: .16e} {v.imag: .16e}" for v in values]

    # the format wants increasing frequency; the result keeps the order given
    for at in np.argsort(net.frequency, kind="stable"):
        f, m = net.frequency[at], s[at]
        head = f"{f:.16e}"
        if n <= 2:
            # 1- and 2-port files list N11 N21 N12 N22 on one line
            lines.append(" ".join([head, *pairs(m.T.reshape(-1))]))
            continue
        for i, row in enumerate(m):
            items = pairs(row)
            for k in range(0, n, 4):
                lead = head if i == 0 and k == 0 else " " * len(head)
                lines.append(" ".join([lead, *items[k : k + 4]]))
    return "\n".join(lines) + "\n"


@dataclasses.dataclass(frozen=True)
class SweepInfo:
    """How a `Driven` sweep was solved: which frequencies, and how well.

    A uniform sweep solves every frequency in full. An adaptive sweep
    (``Driven(sweep="adaptive")``) solves a few and reads every frequency
    from a reduced model built on them; at a solved frequency the reduced
    model reproduces the full solve to the solver tolerance.

    Attributes:
        kind: ``"uniform"`` or ``"adaptive"``.
        solved: For each frequency of the result (in order), whether it was
            solved in full.
        sampled_frequencies: The frequencies solved in full, Hz, in solve
            order (for an adaptive sweep: the greedy's selection order, or
            ``Driven.samples``). Pass them to ``Driven(samples=...)`` to
            freeze the sample set.
        sample_errors: For each sampled frequency, the relative l2 error of
            the reduced model's field against the full solve there, measured
            before the sample joined the model (the largest over the excited
            ports and, with ``Driven(source_fields=False)``, over the port
            solutions the port waves are read from); ``inf`` where there was
            no model to compare with yet (the two band edges the greedy
            starts from, the first frozen sample, and every frequency of a
            uniform sweep).
        converged: Whether the last ``SolverOptions.adaptive_memory`` sample
            errors are below the tolerance (or every frequency was solved).
            Always True for a uniform sweep. An adaptive greedy that stops at
            ``adaptive_max_samples`` instead warns with
            `gfem.SweepNotConvergedWarning`.
        tolerance: ``SolverOptions.adaptive_tolerance`` of an adaptive
            sweep, ``None`` for a uniform one.
        basis_size: The number of reduced basis vectors (0 for uniform).
        iterations: For each sampled frequency (in solve order), the Krylov
            iterations of each right-hand side of the sample's solve (the
            excited ports' columns, then the sources' columns and the loads
            the sweep spans, every port's own load with
            ``Driven(source_fields=False)``); ``()`` for a direct solve on
            fresh factors.
        preconditioner_reused: For each sampled frequency, whether its
            solve ran on a preconditioner built for another matrix (a BDDC
            or p-multigrid preconditioner built at an earlier frequency, or
            direct factors from an earlier design: preconditioner reuse).

    Under `jax.grad`, `jax.jit` or `jax.vmap` an adaptive sweep needs
    frozen samples (``Driven(samples=...)``), and ``sample_errors``,
    ``converged`` and ``basis_size`` are ``None``: they are only known
    after the solve, and this record is static. Solve outside the
    transformation to read them. ``iterations`` and
    ``preconditioner_reused`` are ``None`` there for any sweep, and in a
    result saved by a gfem that did not record them.
    """

    kind: str
    solved: tuple[bool, ...]
    sampled_frequencies: tuple[float, ...]
    sample_errors: tuple[float, ...] | None
    converged: bool | None
    tolerance: float | None = None
    basis_size: int | None = 0
    iterations: tuple[tuple[int, ...], ...] | None = None
    preconditioner_reused: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        """Normalize to plain Python values (hashable, JSON-friendly)."""
        set_ = object.__setattr__
        set_(self, "solved", tuple(bool(v) for v in self.solved))
        set_(
            self,
            "sampled_frequencies",
            tuple(float(f) for f in self.sampled_frequencies),
        )
        if self.sample_errors is not None:
            set_(self, "sample_errors", tuple(float(e) for e in self.sample_errors))
        if self.converged is not None:
            set_(self, "converged", bool(self.converged))
        if self.basis_size is not None:
            set_(self, "basis_size", int(self.basis_size))
        if self.tolerance is not None:
            set_(self, "tolerance", float(self.tolerance))
        if self.iterations is not None:
            set_(
                self,
                "iterations",
                tuple(tuple(int(i) for i in its) for its in self.iterations),
            )
        if self.preconditioner_reused is not None:
            set_(
                self,
                "preconditioner_reused",
                tuple(bool(r) for r in self.preconditioner_reused),
            )

    @classmethod
    def uniform(
        cls,
        frequency: tuple[float, ...],
        *,
        iterations: Sequence[Sequence[int]] | None = None,
        preconditioner_reused: Sequence[bool] | None = None,
    ) -> SweepInfo:
        """Every frequency of ``frequency`` solved in full."""
        n = len(frequency)
        return cls(
            kind="uniform",
            solved=(True,) * n,
            sampled_frequencies=frequency,
            sample_errors=(math.inf,) * n,
            converged=True,
            iterations=None if iterations is None else tuple(map(tuple, iterations)),
            preconditioner_reused=None
            if preconditioner_reused is None
            else tuple(preconditioner_reused),
        )

    def __repr__(self) -> str:
        if self.kind == "uniform":
            return f"SweepInfo(uniform, {len(self.solved)} frequencies solved)"
        head = (
            f"SweepInfo(adaptive, {len(self.sampled_frequencies)} of "
            f"{len(self.solved)} frequencies solved"
        )
        if self.converged is None:
            return f"{head}, traced: no diagnostics)"
        state = "converged" if self.converged else "NOT converged"
        return (
            f"{head}, basis {self.basis_size}, {state} at tolerance {self.tolerance:g})"
        )


@dataclasses.dataclass(frozen=True, eq=False)
class DrivenResult:
    r"""The result of a `Driven` problem.

    ``s.values`` and the kept fields' coefficients are differentiable with
    respect to every electromagnetic `MaterialField` value (``eps_r``,
    ``mu_r``, ``tan_delta``, ``conductivity``); ``s.z_ref`` is constant.

    Attributes:
        s: The scattering parameters (`NetworkParameters`).
        e_fields: The kept electric fields (``Driven(keep_fields=...)``), one
            `Field` per kept frequency and excited port, labelled by `kept`;
            empty by default. Read them with `e_field`. Each is the peak phasor
            (:math:`e^{+i\omega t}`) in V/m for 1 W time-averaged incident
            power at the excited port, every other port terminated in its
            reference impedance ``z_ref``, so time-averaged densities carry a
            factor 1/2 (see `gfem.conventions`). A wave port is
            terminated by absorbing its mode (matched), including the other
            modes of the excited port's own surface. Quantity ``"E"``, on
            H(curl) of the solve order, over the solve domain.
            An adaptive sweep's fields are basis-backed (`ReducedFields`):
            the result holds the reduced basis and one short coordinate
            vector per field, and a field's coefficients are formed when it
            is read. Their accuracy is the sweep's field error, about
            ``SolverOptions.adaptive_tolerance`` or better. Gradients of an
            adaptive sweep hold its basis fixed (see ``Driven.samples``).
            On a 2D model, the kept Ez fields of an Ez solve (below).
        kept: ``(frequency in Hz, excited port)`` of each of `e_fields`, or of
            `h_fields` (static labels), frequency-major.
        sweep: How the frequencies were solved (`SweepInfo`).
        port_modes: The mode of each wave port over the frequencies
            (`PortModeResult`, by port name), as the port mode sweep found
            it before the 3D solve; empty without wave ports. Each carries
            its mode field, with or without ``Driven(keep_fields=...)``,
            except inside `jax.jit`, `jax.grad` or `jax.vmap`, where it is
            ``None``. ``gamma`` and ``Z_PV`` follow a design at the port's
            face; ``residual`` is not differentiable.
        offsets: ``(port name, distance in m)`` of each wave and Floquet
            port (static labels; ``dict(result.offsets)``), from
            ``WavePort(offset=...)`` or ``FloquetPort(offset=...)``: S of
            such a port is referred to a plane that far inside the domain,
            ``S_ij`` multiplied by ``exp(gamma_i d_i) exp(gamma_j d_j)`` (a
            Floquet port's factor is 1 where its order does not propagate).
            `with_offsets` moves the planes after the solve.
        diffraction_orders: The diffraction order of each Floquet port over
            the frequencies (`DiffractionOrderResult`, by port name); empty
            without Floquet ports. Constants of the solve: not
            differentiable.
        h_fields: The kept magnetic fields of a 2D model's Hz solve
            (``Driven(polarization="Hz")``), labelled by `kept` as
            `e_fields` are, which are then empty; empty otherwise. Read them
            with `h_field`.
        polarization: The field normal to the plane that a 2D model's solve
            computed, ``"Ez"`` or ``"Hz"``; ``None`` for a 3D model. A
            static label.
        sources: The problem's sources (``Driven(sources=...)``) by name, in
            column order: what a measurement reads of them (a direction, an
            amplitude, a position). Static; empty without sources.
        port_waves: The outgoing wave at each port for each source (its column, or
            by reciprocity under ``Driven(source_fields=False)``),
            ``(n_freq, n_ports, n_sources)``, √W, with every port terminated
            in its reference impedance (`gfem.conventions`, Incident fields):
            on a real ``z_ref``, :math:`|b|^2` is the power the source sends
            into the port, in W. Read one with `port_wave`. Differentiable
            like S. ``None`` without sources or without ports.
        emitted_power: The time-averaged power each point dipole gives the
            field, W, ``(n_freq,)``, by source name (`gfem.conventions`,
            Point dipoles). Empty without dipoles, or with
            ``Driven(source_fields=False)``.

    On a 2D model (a mesh of dimension 2, `gfem.conventions`) every port is
    fed 1 W/m time-averaged incident power per metre along :math:`z`, and S is
    dimensionless. A kept field is the scalar normal component, peak phasor
    for that 1 W/m: quantity ``"Ez"`` in V/m in `e_fields`, or ``"Hz"`` in
    A/m in `h_fields`, in H1 of the solve order over the solve domain's
    areas. Its `Field.at` takes ``(m, 2)`` points in the plane.
    """

    s: NetworkParameters
    e_fields: Sequence[Field] = ()
    kept: tuple[tuple[float, str], ...] = ()
    sweep: SweepInfo | None = None
    port_modes: Mapping[str, PortModeResult] = dataclasses.field(default_factory=dict)
    offsets: tuple[tuple[str, float], ...] = ()
    diffraction_orders: Mapping[str, DiffractionOrderResult] = dataclasses.field(
        default_factory=dict
    )
    h_fields: Sequence[Field] = ()
    polarization: Polarization | None = None
    sources: Mapping[str, Source] = dataclasses.field(default_factory=dict)
    port_waves: jax.Array | None = None
    emitted_power: Mapping[str, jax.Array] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Hold the fields as a sequence matching their labels."""
        for name in ("e_fields", "h_fields"):
            if not isinstance(getattr(self, name), ReducedFields):
                object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.e_fields) and len(self.h_fields):
            raise ValueError(
                "A driven result keeps e_fields or h_fields (an Hz solve of a 2D "
                "model), not both."
            )
        if self.sweep is None:
            object.__setattr__(self, "sweep", SweepInfo.uniform(self.s.frequency))
        object.__setattr__(
            self, "kept", tuple((float(f), str(p)) for f, p in self.kept)
        )
        object.__setattr__(self, "port_modes", dict(self.port_modes))
        object.__setattr__(self, "diffraction_orders", dict(self.diffraction_orders))
        object.__setattr__(
            self,
            "offsets",
            tuple((str(k), float(v)) for k, v in dict(self.offsets).items()),
        )
        fields = self.h_fields if len(self.h_fields) else self.e_fields
        if len(fields) != len(self.kept):
            raise ValueError(
                f"{len(fields)} fields but {len(self.kept)} labels in kept."
            )
        object.__setattr__(self, "sources", dict(self.sources))
        object.__setattr__(self, "emitted_power", dict(self.emitted_power))
        self._check_port_waves()

    def _check_port_waves(self) -> None:
        """Drop empty port waves; check the shape of the others."""
        waves = self.port_waves
        if waves is None:
            return
        if not self.sources or not self.s.ports:
            # no source column, or no port row: nothing to read
            object.__setattr__(self, "port_waves", None)
            return
        if is_traced(waves):
            return
        shape = (len(self.s.frequency), len(self.s.ports), len(self.sources))
        if tuple(np.shape(waves)) != shape:
            raise ValueError(
                f"port_waves must be (n_freq, n_ports, n_sources) = {shape}, got "
                f"{tuple(np.shape(waves))}."
            )

    @property
    def _excitations(self) -> tuple[str, ...]:
        """The column labels: the excited ports, then the sources."""
        return (*self.s.excited, *self.sources)

    def port_wave(self, port: str, source: str) -> jax.Array:
        """The outgoing wave at ``port`` for the column of ``source``, ``(n_freq,)``.

        √W (`port_waves`); :math:`|b|^2` is the power in W the source sends
        into the port on a real reference impedance.

        Raises:
            KeyError: ``port`` is not a port of the result, or ``source`` is
                not one of its sources.
        """
        ports = self.s.ports
        if port not in ports:
            raise KeyError(
                f"No port {port!r}; the ports are {', '.join(map(repr, ports))}."
            )
        if not self.sources:
            raise KeyError(
                "This result solved no source (Driven(sources=...)), so it has no "
                "port waves."
            )
        if source not in self.sources:
            raise KeyError(
                f"No source {source!r}; the sources are "
                f"{', '.join(map(repr, self.sources))}."
            )
        waves = self.port_waves
        if waves is None:
            raise KeyError("This result holds no port waves.")
        return waves[:, ports.index(port), list(self.sources).index(source)]

    @property
    def frequency(self) -> np.ndarray:
        """The frequencies in Hz, ``(n_freq,)``."""
        return np.array(self.s.frequency)

    def e_field(
        self,
        frequency: float,
        excite: str | None = None,
        *,
        part: Literal["total", "scattered", "incident"] = "total",
    ) -> Field:
        """The kept electric field at ``frequency`` of one excitation.

        On a 2D model, the kept Ez field of an Ez solve.

        Args:
            frequency: A kept frequency, Hz (matched to 1e-9 relative).
            excite: The excited port or the source; may be left out when the
                problem had one of them.
            part: ``"total"``, the field (the default); for an incident-field
                column also ``"scattered"``, the total minus the incident
                field, or ``"incident"``, the incident field on the same
                space, so that the three parts add exactly.

        Raises:
            ValueError: No field was kept at ``frequency``, ``excite`` is
                needed, ``part`` is unknown or is not ``"total"`` for a port
                or a dipole, the result is a 2D model's Hz solve (read
                `h_field`), or ``excite`` is a dipole of a
                ``Driven(source_fields=False)`` solve, whose field was not
                solved.
            KeyError: ``excite`` is neither an excited port nor a source.
        """
        if part not in _PARTS:
            raise ValueError(
                "DrivenResult.e_field: part must be 'total', 'scattered' or "
                f"'incident', got {part!r}."
            )
        if self.polarization == "Hz":
            raise ValueError(
                "This result solved the Hz polarization of a 2D model: its kept "
                "field is Hz, read result.h_field(frequency, excite)."
            )
        if excite is not None:
            self._refuse_unsolved_source(
                excite, "so no field of it is kept", why="its field was not solved"
            )
        field = self._kept_field(self.e_fields, frequency, excite)
        if part == "total":
            return field
        name = excite if excite is not None else self._excitations[0]
        source = self.sources.get(name)
        # a value error: the part asked for does not exist for this column
        if not isinstance(source, INCIDENT):
            kind = "port" if source is None else type(source).__name__
            raise ValueError(  # noqa: TRY004
                f"{name!r} is a {kind} column: it has no incident field, so its "
                f"only part is the total field; got part={part!r}."
            )
        return _incident_part(field, name, part)

    def _refuse_unsolved_source(
        self,
        name: str,
        consequence: str,
        *,
        why: str = (
            "its port waves were read by reciprocity and its own field was not solved"
        ),
        remedy: str = "it",
    ) -> None:
        """Raise for a dipole of a ``source_fields=False`` solve.

        Such a dipole is among the sources but has no emitted power: gfem
        reads its port waves by reciprocity and never solves its field.

        Raises:
            ValueError: ``name`` is a point dipole without an emitted power.
        """
        source = self.sources.get(name)
        if isinstance(source, DIPOLES) and name not in self.emitted_power:
            raise ValueError(
                f"{name!r} was solved with Driven(source_fields=False): {why}, "
                f"{consequence}. Solve with source_fields=True (the default) "
                f"for {remedy}."
            )

    def h_field(self, frequency: float, excite: str | None = None) -> Field:
        """The kept Hz field of a 2D model's Hz solve at ``frequency``.

        Args:
            frequency: A kept frequency, Hz (matched to 1e-9 relative).
            excite: The excited port; may be left out when one port was
                excited.

        Raises:
            ValueError: The result is not an Hz solve (read `e_field`), no
                field was kept at ``frequency``, or ``excite`` is needed.
            KeyError: ``excite`` is not an excited port.
        """
        if self.polarization != "Hz":
            solved = (
                "the Ez polarization of a 2D model: its kept field is Ez"
                if self.polarization == "Ez"
                else "a 3D model: h_field reads the Hz polarization of a 2D model"
            )
            raise ValueError(
                f"This result solved {solved}; read result.e_field(frequency, excite)."
            )
        return self._kept_field(self.h_fields, frequency, excite)

    def _kept_field(
        self, fields: Sequence[Field], frequency: float, excite: str | None
    ) -> Field:
        if not self.kept:
            raise ValueError(
                "This result keeps no fields: solve with Driven(..., "
                "keep_fields=True) or keep_fields=[frequencies] to keep them."
            )
        # a dipole read by reciprocity was never solved: it is no column
        kept = {port for _, port in self.kept}
        columns = tuple(c for c in self._excitations if c in kept) or tuple(
            self._excitations
        )
        if excite is None:
            if len(columns) > 1:
                what = "Ports" if not self.sources else "Excitations"
                verb = "excited" if not self.sources else "solved"
                raise ValueError(
                    f"{what} {', '.join(map(repr, columns))} were {verb}; "
                    "pass excite= to choose one."
                )
            excite = columns[0]
        if excite not in columns:
            if not self.sources:
                raise KeyError(
                    f"Port {excite!r} was not excited; the excited ports are "
                    f"{', '.join(map(repr, columns))}."
                )
            raise KeyError(
                f"{excite!r} was not excited; the excited ports and the sources "
                f"are {', '.join(map(repr, columns))}."
            )
        frequencies = sorted({f for f, _ in self.kept})
        for k, (f, port) in enumerate(self.kept):
            if port == excite and math.isclose(f, frequency, rel_tol=1e-9):
                return fields[k]
        raise ValueError(
            f"No field kept at {frequency:g} Hz; the kept frequencies are "
            f"{', '.join(f'{f:g}' for f in frequencies)} Hz (Driven.keep_fields)."
        )

    def with_offsets(self, offsets: Mapping[str, float]) -> DrivenResult:
        r"""This result with S referred to other wave or Floquet port planes.

        ``offsets`` gives, by port name, the distance in m of the new plane
        inside the domain, as ``WavePort(offset=...)`` and
        ``FloquetPort(offset=...)`` do: absolute, not
        an increment, so a result of a solve without offsets shifted to
        ``d`` equals the result of a solve with ``WavePort(offset=d)``, and
        shifting twice to the same planes changes nothing. :math:`S_{ij}` is
        multiplied by :math:`e^{\gamma_i (d_i - d_i^\mathrm{old})}
        e^{\gamma_j (d_j - d_j^\mathrm{old})}` at each frequency,
        :math:`\gamma` the ports' propagation
        constants in `port_modes` (a Floquet port's order's in
        `diffraction_orders`, the factor 1 where the order does not
        propagate), and `offsets` records the new distances.
        A port not named keeps its plane. ``z_ref``, the kept fields,
        `port_modes` and `sweep` are unchanged.

        Written in `jax.numpy`, so the new S differentiates like S, and
        through ``gamma``, which follows a design at the port's face.

        Args:
            offsets: The new distance of each named wave or Floquet port, m,
                >= 0.

        Returns:
            A new result; this one is not modified.

        Raises:
            KeyError: A name is not a port of this result.
            ValueError: A named port is not a wave or Floquet port (it has no
                propagation constant), or a distance is negative or not
                finite.
            NonDifferentiableError: A distance is computed inside
                `jax.grad`, `jax.jit` or `jax.vmap`: offsets are fixed
                values, like ``WavePort.offset``.
        """
        ports = self.s.ports
        old = dict(self.offsets)
        new = dict(old)
        for name, value in dict(offsets).items():
            if name not in ports:
                raise KeyError(
                    f"No port {name!r}; the ports are {', '.join(map(repr, ports))}."
                )
            if name not in old or (
                name not in self.port_modes and name not in self.diffraction_orders
            ):
                raise ValueError(
                    f"Port {name!r} is not a wave or Floquet port: it has no "
                    "propagation constant to move a reference plane with. The wave "
                    f"and Floquet ports are {', '.join(map(repr, old)) or 'none'}."
                )
            distance = static_float(
                value,
                f"with_offsets: the offset of port {name!r}",
                "Pass a concrete distance; the shifted S still differentiates "
                "with respect to the design.",
            )
            if distance < 0:
                raise ValueError(
                    f"with_offsets: the offset of port {name!r} must be >= 0 m, "
                    f"got {distance}."
                )
            new[name] = distance
        factor = jnp.stack(
            [
                self._plane_shift(p, new[p] - old[p])
                if p in new and new[p] != old[p]
                else jnp.ones(len(self.s.frequency), dtype=complex)
                for p in ports
            ],
            axis=1,
        )
        columns = [ports.index(p) for p in self.s.excited]
        values = self.s.values * factor[:, :, None] * factor[:, None, columns]
        return dataclasses.replace(
            self,
            s=dataclasses.replace(self.s, values=values),
            offsets=tuple(new.items()),
        )

    def _plane_shift(self, port: str, distance: float) -> jax.Array:
        """``(n_freq,)``: ``exp(gamma distance)`` of a wave or Floquet port.

        A Floquet port's factor is 1 where its order does not propagate (its
        row reads 0 there, and ``exp(|gamma| d)`` would overflow).
        """
        if port in self.port_modes:
            gamma = jnp.asarray(_objective.constant(self.port_modes[port].gamma))
            return jnp.exp(gamma * distance)
        order = self.diffraction_orders[port]
        propagating = jnp.asarray(_objective.constant(order.propagating))
        gamma = jnp.asarray(_objective.constant(order.gamma))
        return jnp.exp(jnp.where(propagating, gamma, 0.0) * distance)

    def __repr__(self) -> str:
        f = self.frequency
        span = hertz_band(f.min(), f.max(), 4)
        kept = len(self.h_fields) or len(self.e_fields)
        fields = f", {kept} kept fields" if kept else ""
        sweep = "" if self.sweep is None else f", {self.sweep!r}"
        sources = f", sources={list(self.sources)}" if self.sources else ""
        return (
            f"DrivenResult({len(f)} frequencies {span}, ports={list(self.s.ports)}, "
            f"excited={list(self.s.excited)}{sources}{fields}{sweep})"
        )


#: The parts of a kept field (``DrivenResult.e_field(part=...)``).
_PARTS = ("total", "scattered", "incident")


def _incident_part(field: Field, name: str, part: str) -> Field:
    r"""The ``"scattered"`` or ``"incident"`` part of an incident column's field.

    The incident part is the source's moment interpolant on the field's
    space, zero in the perfectly matched layer (where the kept field is the
    scattered one), evaluated again from the source; the scattered part is
    the kept field minus it. Both are fields of the same space as the kept
    field, so the three add exactly; the incident part is a constant of any
    design.

    Raises:
        ValueError: The field carries no incident part (read from a file
            written before gfem kept it), or its node coordinates are traced.
    """
    from gfem._engine.incident import incident_coefficients

    reference = field._incident  # noqa: SLF001
    if reference is None:
        raise ValueError(
            f"The kept field of {name!r} carries no incident part: solve again "
            "to read its scattered and incident parts."
        )
    if is_traced(field.mesh.nodes) or is_traced(field.omega):
        raise ValueError(
            f"The {part} part of {name!r} needs concrete node coordinates: the "
            "incident part is interpolated on the host. Read it outside jax.jit, "
            "or from a field whose mesh is not traced."
        )
    incident = jnp.asarray(incident_coefficients(field, reference))
    if part == "scattered":
        return dataclasses.replace(
            field,
            coefficients=field.coefficients - incident,
            _incident=None,
            _scattered=True,
        )
    # marked, so that power_flow refuses it (its flux balances no load)
    marked = dataclasses.replace(reference, is_incident=True)
    return dataclasses.replace(field, coefficients=incident, _incident=marked)


register_fields(
    DrivenResult,
    (
        "s",
        "e_fields",
        "port_modes",
        "diffraction_orders",
        "h_fields",
        # static source values: nodes without leaves
        "sources",
        "port_waves",
        "emitted_power",
    ),
)


@dataclasses.dataclass(frozen=True, eq=False)
class EigenmodeResult:
    r"""The result of an `Eigenmode` problem: ``n_modes`` modes, nearest first.

    Every array has one entry per mode, in order of increasing distance of
    the mode's frequency from ``Eigenmode.target``.

    Differentiation: ``omega`` differentiates with respect to every model
    input the capabilities matrix lists. The ratios (`q_port`,
    `participation`, `bulk_participation`, `principal_participation`,
    `sheet_participation`) and the mode fields are read from the modes'
    fields; they differentiate with respect to materials, sheet fill, mesh
    node coordinates and lumped values for an isolated mode: a relative
    frequency gap of at least ``max(1e-4, SolverOptions.cluster_rtol)`` to
    every other converged mode the selection accepted (those cut beyond
    ``n_modes`` included) and to the guard mode (the nearest converged mode
    beyond ``n_modes``), on
    ``linear_solver="direct"`` on the CPU, in one process, with
    ``eigen_tolerance`` at most 1e-10, on a non-periodic model. The fields
    raise with respect to node coordinates, and `principal_participation`
    with respect to a lumped value. Outside that reach a gradient through
    them raises `gfem.NonDifferentiableError` (or `gfem.DegenerateModeError`
    for a mode that is not isolated); ``error`` is not differentiable. With
    wave ports, a design region touching a wave port face raises
    `gfem.PortGeometryError`, as in a driven problem.

    Attributes:
        omega: Complex angular frequencies, rad/s, ``(n_modes,)``. Under
            :math:`e^{+i\omega t}` a decaying mode has
            :math:`\mathrm{Im}\,\omega > 0`.
        error: Backward error of each mode,
            :math:`|T(\omega) x| / ((|K| + |\omega| |C| + |\omega|^2 |M|) |x|)`
            for the discrete operator
            :math:`T(\omega) = K + i \omega C - \omega^2 M`, plus the
            frequency-dependent terms of conductors, second-order absorbing
            boundaries and wave ports; at most
            ``SolverOptions.eigen_tolerance``.
        requested_power_share: The share of the power each wave port surface
            absorbs from the mode that the surface's requested modes carry,
            by surface name in the order of each surface's first wave port,
            ``(n_modes,)``:
            :math:`\eta_s = \sum_{p \in s} P_p / P_s`, with :math:`P_p`
            the power wave port :math:`p`'s requested mode carries out (the
            one `q_port` divides) and :math:`P_s = \mathrm{Im}(\bar x^\mathsf{T}
            B_s x) / (2\, \mathrm{Re}\,\omega\, \mu_0 s_\ell)` the power
            the surface's port term :math:`B_s` absorbs. The port absorbs its
            requested modes exactly and any other field on its surface only
            through its Robin scalar, so :math:`\eta_s` is near 1 (within
            :math:`O(1/Q^2)`, up to 1.004 measured) for a mode of the
            requested families, and 1e-3 or less for a mode carried by an
            unrequested family or trapped against the port. Between one half
            and 1 a discontinuity close to the port puts an evanescent tail
            on it: :math:`1 - \eta_s` is the part of the port loss no
            requested mode carries, and the Q over :math:`\eta_s` is close to
            the Q of a port moved away. Empty without wave ports; NaN where
            the surface absorbs no power from the mode. Below one half a
            solve warns `gfem.UnrequestedPortModeWarning`. A diagnostic of the
            found mode, like ``error``: not differentiable.
        q_port: Coupling Q of each port that takes power out of the mode, by
            port name, in model order: each lumped port with a resistance R
            and each wave port. A lumped port's is
            :math:`\mathrm{Re}\,\omega / \kappa` with the port's coupling
            rate :math:`\kappa = R |I|^2 / (2 E)`, :math:`I = V / R` the
            current through R for the port voltage V of the mode and
            :math:`E` the mode's electric
            energy. V is the port's mean voltage
            over its elements, as in a driven problem; when the elements of
            a multi-element port see different voltages, R dissipates more
            than :math:`|V|^2 / (2 R)`, and :math:`1 / Q` exceeds the sum of the
            ``1 / q_port`` and the dielectric losses. A wave port's is
            :math:`q_p = \mathrm{Re}\,\omega\, E / P_p`, with
            :math:`P_p = \tfrac12 \mathrm{Re}\int_{S_p} (\mathbf{E} \times
            \mathbf{H}^*) \cdot \mathbf{n}\, dS` the power the port's requested
            mode carries out of the model at the complex frequency, from that
            mode's own amplitude only: on a shared port surface the cross
            terms between modes, zero on a lossless section, are dropped. With
            no other loss, :math:`\sum_p 1 / q_p` matches :math:`1 / Q` to
            :math:`O(1/Q^2)`, since `q` divides :math:`|\omega|` and the
            coupling Q is a power ratio. Differentiable as the ratios above,
            but an infinite value (a mode with no amplitude on the port)
            takes no gradient: differentiate ``1 / q_port``. A wave port's
            value also raises `gfem.NonDifferentiableError` when the port
            modes of its surface are degenerate.
        participation: Inductive energy participation ratio of each lumped
            port with an inductance L, by port name: :math:`L |I|^2 / (2 E)`
            with :math:`I = V / (i\, \mathrm{Re}(\omega) L)`, the junction
            participation of a transmon. Its sign is that of
            :math:`\mathrm{Re}\, V`: the field's phase makes a lossless mode
            real, so the sign is the junction's polarity along its port
            direction (not :math:`\mathrm{Re}\, I`, which is zero for a real
            field). Under a Bloch wave vector the sign is relative to the
            mode's phase convention (its largest coefficient real). For a
            lossless mode with a field uniform across the port it is
            :math:`-2\, d\ln\omega / d\ln L`. Differentiable as the ratios
            above.
        bulk_participation: Electric energy participation of each volume
            region of the solve domain outside a perfectly matched layer, by
            region name: the fraction of the
            mode's electric energy ``E`` stored in it (in a region with a
            permittivity law, its share of the mode's total energy, weighed
            by the energy permittivity: see `gfem.conventions`). For an isotropic
            loss, the region's contribution to :math:`1 / Q` is
            :math:`p (\tan\delta + \sigma / (\varepsilon_0 \varepsilon_r
            \mathrm{Re}\,\omega))`; for a loss given as
            principal values use `principal_participation`.
            Differentiable as the ratios above.
        principal_participation: The electric energy participation of each
            anisotropic volume region (an anisotropic `Material`, or a
            `MaterialField` with ``axes``) split along its principal axes, by
            region name: ``(n_modes, 3)``, in the order of the material's
            ``axes`` (the coordinate axes when ``None``),
            :math:`p_k = \frac{1}{2} \varepsilon_0 \int \varepsilon_{r,k}
            |a_k \cdot E|^2 \, dV / E`; the three add up to the region's
            `bulk_participation`. The region's contribution to :math:`1 / Q`
            is :math:`\sum_k p_k (\tan\delta_k + \sigma_k / (\varepsilon_0
            \varepsilon_{r,k} \mathrm{Re}\,\omega))` for any field direction
            (with losses uniform over the region). Isotropic regions have no
            entry. Differentiable as the ratios above, except with respect to a
            lumped value, which raises: use `bulk_participation` there.
        sheet_participation: Inductive energy participation of each
            `Impedance` sheet with an inductance ``L_s``, by surface name,
            ``(n_modes,)``:
            :math:`\int w |E_t|^2 \, dS / (2\, \mathrm{Re}(\omega)^2 L_s E)`,
            the energy of the current through ``L_s`` over the mode's
            electric energy ``E``, with ``w`` the sheet's fill (1 without a
            `SheetField`), one number for the whole patterned sheet. It is
            non-negative and, for a lossless mode, the sheet's kinetic
            inductance fraction :math:`-2\, d\ln\omega / d\ln L_s`, in [0, 1]; only
            the current through L counts when the sheet also has R or C. A
            `Conductor` and a sheet without an inductance have no entry.
            Differentiable as the ratios above, ``L_s`` included.
        e_fields: The mode fields, one `Field` per mode (read them with
            `e_field`): peak phasors in V/m, quantity ``"E"``, on H(curl)
            of the solve order over the solve domain, each with its mode's
            complex ``omega``. Each mode stores 1 J of electric energy,
            :math:`E = \frac{1}{2} \mathrm{Re} \int D^* \cdot E \, dV` plus the
            lumped capacitors' :math:`C |V|^2 / 2` and the capacitive sheets'
            :math:`\frac{1}{2} \int w C_s |E_t|^2 \, dS` (the total
            time-averaged energy of a lossless mode), and its phase makes
            :math:`\int D \cdot E \, dV` real and positive.
        materials: The model's materials the modes were solved with, by
            volume region (`Material` or `MaterialField`). Measurements
            that need the media around a surface read them
            (`gfem.interface_participation`).
        periodic_surfaces: The ``(first, second)`` surface names of each
            `gfem.Periodic` pair of the model the modes were solved on;
            empty for a model with no pair, and for a result saved before
            gfem recorded them. `gfem.interface_participation` refuses
            these surfaces: they are interior to the periodic cell.
        pml: The model's perfectly matched layer (`gfem.PML`) the modes were
            solved with, or ``None``. Its regions hold no energy of the
            mode: the 1 J, the participations and
            `gfem.interface_participation` count the physical regions only,
            and in the layer the field is the stretched one, not a physical
            field.
        cluster: Integer label per mode, ``(n_modes,)``: modes with the same
            label form a degenerate cluster, frequencies within
            ``SolverOptions.cluster_rtol`` of each other. A cluster's modes
            are an arbitrary basis of its eigenspace, so only objectives
            symmetric in them (sum, mean) are differentiable; a gradient
            through one member raises `gfem.DegenerateModeError`. Labels
            count from 0 in mode order; a simple mode has a label of its
            own.
        v_g: The band group velocity :math:`\partial\omega/\partial\mathbf{k}`
            of each mode of a Bloch solve (``Eigenmode.wave_vector``), m/s,
            ``(n_modes, 3)`` complex: the Cartesian vector, in the span of the
            cell's translations (its other components are 0); ``None``
            without a wave vector. It is the exact derivative of the
            discrete band, from the mode and its left eigenvector. On a
            lossy cell it is the derivative of the complex band: the real
            part is the group velocity of a weakly lossy band, the
            imaginary part the derivative of its decay rate; on a lossless
            cell it is real. Modes whose frequencies lie within
            :math:`10^{-3}` relative of each other form a group: they
            share one value when their slopes agree (a symmetry doublet at
            a high-symmetry point), and it is NaN when the bands touch with
            different slopes (the frequency is not differentiable there),
            for a group that ``n_modes`` cuts, for a defective group, and
            (then for every mode) where the expansion's phase template does
            not stand for the periodic pairs.
            Not differentiable: a gradient through it raises
            `gfem.NonDifferentiableError`. The group index is
            :math:`c / |v_g|`.
    """

    omega: jax.Array
    error: jax.Array
    q_port: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    participation: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    bulk_participation: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    e_fields: tuple[Field, ...] = ()
    materials: dict[str, Any] = dataclasses.field(default_factory=dict)
    cluster: jax.Array | None = None
    principal_participation: dict[str, jax.Array] = dataclasses.field(
        default_factory=dict
    )
    periodic_surfaces: tuple[tuple[str, str], ...] = ()
    sheet_participation: dict[str, jax.Array] = dataclasses.field(default_factory=dict)
    pml: PML | None = None
    v_g: jax.Array | None = None
    requested_power_share: dict[str, jax.Array] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Hold fields, materials and pairs as tuples and dict; label simple modes."""
        object.__setattr__(self, "e_fields", tuple(self.e_fields))
        object.__setattr__(self, "materials", dict(self.materials))
        pairs = tuple((str(a), str(b)) for a, b in self.periodic_surfaces)
        object.__setattr__(self, "periodic_surfaces", pairs)
        if self.cluster is None:  # every mode simple (results of older files)
            n = int(np.shape(self.omega)[0])
            object.__setattr__(self, "cluster", jnp.arange(n))

    @property
    def frequency(self) -> jax.Array:
        r"""Mode frequencies :math:`\mathrm{Re}\,\omega / (2\pi)`, Hz."""
        return jnp.real(self.omega) / (2 * math.pi)

    @property
    def q(self) -> jax.Array:
        r"""Quality factors :math:`|\omega| / (2 |\mathrm{Im}\,\omega|)`.

        ``inf`` for a lossless mode. The total Q: material and port losses,
        and with a perfectly matched layer (`gfem.PML`) the radiation it
        absorbs. A mode whose :math:`\mathrm{Im}\,\omega` has the gain sign
        on such a model warns `gfem.LayerClearanceWarning`: its Q is the
        layer's, not the device's.
        """
        return jnp.abs(self.omega) / (2 * jnp.abs(jnp.imag(self.omega)))

    @property
    def n_modes(self) -> int:
        """The number of modes."""
        return int(np.shape(self.omega)[0])

    def e_field(self, mode: int) -> Field:
        """The electric field of mode ``mode`` (0 = nearest the target).

        Raises:
            IndexError: No such mode.
        """
        if not -self.n_modes <= mode < self.n_modes:
            raise IndexError(
                f"Mode {mode} does not exist: this result has {self.n_modes} "
                "mode(s), numbered from 0."
            )
        return self.e_fields[mode]

    def to_xarray(self) -> xarray.Dataset:
        """The modes' scalar outputs as an `xarray.Dataset`.

        ``omega`` (rad/s), ``frequency`` (Hz), ``q``, ``error`` and
        ``cluster`` over the dimension ``mode`` (0 = nearest the target);
        `q_port` and `participation` over ``("mode", "port")``, one port
        axis for both (NaN where a port has no value), and
        `bulk_participation` over ``("mode", "region")``, and
        `principal_participation` over ``("mode", "anisotropic_region",
        "axis")`` (axis 0, 1, 2 in each material's ``axes`` order),
        `sheet_participation` over ``("mode", "sheet")``,
        `requested_power_share` over ``("mode", "port_surface")``, and `v_g`
        (m/s) over ``("mode", "direction")`` (x, y, z) of a Bloch solve. The fields
        and materials are not exported. Needs the ``xarray`` extra: ``pip
        install "gfem[xarray]"``.

        Raises:
            ImportError: xarray is not installed.
            TypeError: The result holds JAX tracers.
        """
        xr = _xarray(self)
        m = ("mode",)
        data = {
            "omega": _variable(m, self.omega, "rad/s"),
            "frequency": _variable(m, self.frequency, "Hz"),
            "q": _variable(m, self.q),
            "error": _variable(m, self.error),
            "cluster": _variable(m, self.cluster),
        }
        coords: dict[str, Any] = {"mode": np.arange(self.n_modes)}
        ports = list(dict.fromkeys([*self.q_port, *self.participation]))
        if ports:
            coords["port"] = ports
            for name, values in (
                ("q_port", self.q_port),
                ("participation", self.participation),
            ):
                table = np.full((self.n_modes, len(ports)), np.nan)
                for k, port in enumerate(ports):
                    if port in values:
                        table[:, k] = np.asarray(values[port])
                data[name] = _variable(("mode", "port"), table)
        if self.bulk_participation:
            coords["region"] = list(self.bulk_participation)
            data["bulk_participation"] = _variable(
                ("mode", "region"),
                np.stack([np.asarray(v) for v in self.bulk_participation.values()], 1),
            )
        if self.principal_participation:
            coords["anisotropic_region"] = list(self.principal_participation)
            coords["axis"] = np.arange(3)
            data["principal_participation"] = _variable(
                ("mode", "anisotropic_region", "axis"),
                np.stack(
                    [np.asarray(v) for v in self.principal_participation.values()], 1
                ),
            )
        if self.sheet_participation:
            coords["sheet"] = list(self.sheet_participation)
            data["sheet_participation"] = _variable(
                ("mode", "sheet"),
                np.stack([np.asarray(v) for v in self.sheet_participation.values()], 1),
            )
        if self.v_g is not None:
            coords["direction"] = ["x", "y", "z"]
            data["v_g"] = _variable(("mode", "direction"), self.v_g, "m/s")
        if self.requested_power_share:
            coords["port_surface"] = list(self.requested_power_share)
            data["requested_power_share"] = _variable(
                ("mode", "port_surface"),
                np.stack(
                    [np.asarray(v) for v in self.requested_power_share.values()], 1
                ),
            )
        return xr.Dataset(data, coords=coords)

    def __repr__(self) -> str:
        try:
            f = np.asarray(self.frequency)
            q = np.asarray(self.q)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"EigenmodeResult({self.n_modes} modes)"
        rows = ", ".join(f"{hertz(a)} (Q {b:.4g})" for a, b in zip(f, q, strict=True))
        return f"EigenmodeResult([{rows}])"


register_fields(
    EigenmodeResult,
    (
        "omega",
        "error",
        "q_port",
        "participation",
        "bulk_participation",
        "e_fields",
        "materials",
        "cluster",
        "principal_participation",
        "sheet_participation",
        "v_g",
        "requested_power_share",
    ),
)


@dataclasses.dataclass(frozen=True, eq=False)
class PiezoelectricResult:
    r"""The result of a `Piezoelectric` problem.

    Excitation ``j`` puts terminal ``j`` at 1 V and every other terminal and
    the ground at 0 V. A 2D section states extensive quantities per metre
    along :math:`z`; the device's admittance is
    ``admittance_per_length`` times its aperture.

    Attributes:
        admittance_per_length: :math:`Y'_{ij}(f)` in S/m, complex, shape
            ``(n_frequencies, n_terminals, n_terminals)``, in `terminals`
            order: ``[k, i, j]`` is the current into terminal ``i`` per volt
            on terminal ``j`` at frequency ``k``, per metre along
            :math:`z`, :math:`Y' = i\omega q'` with :math:`q'` the charge per
            length. Symmetric, with a positive semidefinite real part.
        clamped_capacitance_per_length: :math:`C'_0` in F/m, shape
            ``(n_terminals, n_terminals)``: the Maxwell matrix of the
            potential alone, with the clamped permittivity
            :math:`\varepsilon^S` and no displacement. It is not the
            low-frequency limit of :math:`Y'/i\omega`, the free capacitance
            (larger by :math:`1 + K^2`).
        terminals: The terminal names (static labels), the matrix order.
        frequencies: The frequencies in Hz, in the problem's order (static;
            `frequency` reads them as an array).
        displacement_fields: The kept displacements
            (``Piezoelectric(keep_fields=...)``), one ``FieldMap`` per kept
            (frequency, terminal) labelled by `kept`, with the keys ``"x"``,
            ``"y"`` and ``"z"``: scalar `Field` s of quantity ``"ux"``,
            ``"uy"`` or ``"uz"`` in m, per volt on the terminal, on H1 of the
            solve order over the mechanical domain. Empty by default.
        potential_fields: The kept potentials, one `Field` per kept
            (frequency, terminal), quantity ``"V"``, per volt on the
            terminal, on H1 over the electromagnetic domain.
        kept: ``(frequency in Hz, terminal)`` of each kept field, frequency
            major (static labels).
    """

    admittance_per_length: jax.Array
    clamped_capacitance_per_length: jax.Array
    terminals: tuple[str, ...]
    frequencies: tuple[float, ...]
    displacement_fields: Sequence[FieldMap] = ()
    potential_fields: Sequence[Field] = ()
    kept: tuple[tuple[float, str], ...] = ()

    def __post_init__(self) -> None:
        """Hold the kept fields as sequences matching their labels."""
        object.__setattr__(self, "terminals", tuple(self.terminals))
        object.__setattr__(
            self, "frequencies", tuple(float(f) for f in self.frequencies)
        )
        # a result read back from a file holds plain mappings
        object.__setattr__(
            self,
            "displacement_fields",
            tuple(
                m if isinstance(m, FieldMap) else FieldMap(m, label="component")
                for m in self.displacement_fields
            ),
        )
        object.__setattr__(self, "potential_fields", tuple(self.potential_fields))
        object.__setattr__(
            self, "kept", tuple((float(f), str(t)) for f, t in self.kept)
        )
        if (
            not len(self.displacement_fields)
            == len(self.potential_fields)
            == len(self.kept)
        ):
            raise ValueError(
                f"{len(self.displacement_fields)} displacements and "
                f"{len(self.potential_fields)} potentials but {len(self.kept)} "
                "labels in kept."
            )

    @property
    def frequency(self) -> np.ndarray:
        """The frequencies in Hz, ``(n_frequencies,)``."""
        return np.array(self.frequencies)

    def index(self, terminal: str) -> int:
        """Row / column of ``terminal`` in `admittance_per_length`.

        Raises:
            KeyError: ``terminal`` is not a terminal of this result.
        """
        return _terminal_index(self.terminals, terminal)

    def displacement(self, frequency: float, terminal: str | None = None) -> FieldMap:
        """The kept displacement at ``frequency`` for 1 V on ``terminal``.

        Args:
            frequency: A kept frequency, Hz (matched to 1e-9 relative).
            terminal: The excited terminal; may be left out when there is one.

        Returns:
            The scalar `Field` s ``"x"``, ``"y"`` and ``"z"`` of the
            displacement, in m per volt.

        Raises:
            ValueError: No field was kept at ``frequency``, or ``terminal``
                is needed.
            KeyError: ``terminal`` is not a terminal of this result.
        """
        return self._kept_field(self.displacement_fields, frequency, terminal)

    def potential(self, frequency: float, terminal: str | None = None) -> Field:
        """The kept potential at ``frequency`` for 1 V on ``terminal``.

        Raises:
            ValueError: No field was kept at ``frequency``, or ``terminal``
                is needed.
            KeyError: ``terminal`` is not a terminal of this result.
        """
        return self._kept_field(self.potential_fields, frequency, terminal)

    def _kept_field(
        self, fields: Sequence[Any], frequency: float, terminal: str | None
    ) -> Any:
        if not self.kept:
            raise ValueError(
                "This result keeps no fields: solve with Piezoelectric(..., "
                "keep_fields=True) or keep_fields=[frequencies] to keep them."
            )
        if terminal is None:
            if len(self.terminals) > 1:
                raise ValueError(
                    f"Terminals {', '.join(map(repr, self.terminals))} were "
                    "solved; pass terminal= to choose one."
                )
            terminal = self.terminals[0]
        self.index(terminal)
        for k, (f, name) in enumerate(self.kept):
            if name == terminal and math.isclose(f, frequency, rel_tol=1e-9):
                return fields[k]
        listed = ", ".join(f"{f:g}" for f in sorted({f for f, _ in self.kept}))
        raise ValueError(
            f"No field kept at {frequency:g} Hz; the kept frequencies are {listed} "
            "Hz (Piezoelectric.keep_fields)."
        )

    def network_parameters(
        self,
        aperture: Any,
        *,
        series_resistance: Any = 0.0,
        reference_impedance: Any = 50.0,
    ) -> NetworkParameters:
        r"""The device's scattering parameters on a reference impedance.

        The admittance of an aperture :math:`W` is :math:`Y = W Y'`. A series
        resistance :math:`R = \mathrm{diag}(R_s)` in each terminal's lead
        gives :math:`Y_t = Y (I + R Y)^{-1}`, which needs no inverse of
        :math:`Y`. The scattering matrix is that of :math:`Y_t` on
        ``reference_impedance`` (pseudo-waves, `gfem.conventions`); for one
        terminal on a real :math:`Z_0` it is :math:`S_{11} = (Z_L - Z_0) /
        (Z_L + Z_0)` with :math:`Z_L = 1/Y + R_s`. Written in `jax.numpy`, so
        it differentiates in all three arguments and in the result, and the
        returned `NetworkParameters` writes a Touchstone file, renormalizes
        and converts to Y and Z. The series resistance does not change the
        field solve, so it is an argument here and not of the problem; its
        dependence on the aperture is the caller's.

        Args:
            aperture: The device's length along :math:`z`, in m (> 0).
            series_resistance: The lead resistance in ohm (>= 0): one value or
                one per terminal in `terminals` order.
            reference_impedance: The reference impedance in ohm, with a positive
                real part: one value or one per terminal.

        Returns:
            `NetworkParameters` with ``ports`` and ``excited`` the `terminals`
            and ``z_ref`` of shape ``(n_frequencies, n_terminals)``.

        Raises:
            ValueError: ``aperture`` is not positive, a resistance is negative,
                or a value has the wrong shape or a non-positive real part.
        """
        n = len(self.terminals)
        n_f = len(self.frequencies)
        if not is_traced(aperture) and not (np.isfinite(aperture) and aperture > 0):
            raise ValueError(
                f"aperture must be a positive length in m, got {aperture!r}."
            )
        r = _per_terminal(series_resistance, n, "series_resistance", self.terminals)
        if not is_traced(series_resistance) and not (np.real(np.asarray(r)) >= 0).all():
            raise ValueError(
                f"series_resistance must be >= 0 ohm, got {series_resistance!r}."
            )
        z0 = _per_terminal(
            reference_impedance, n, "reference_impedance", self.terminals
        )
        if (
            not is_traced(reference_impedance)
            and not (np.real(np.asarray(z0)) > 0).all()
        ):
            raise ValueError(
                "reference_impedance needs a positive real part (a resistance), "
                f"got {reference_impedance!r}."
            )
        y = aperture * jnp.asarray(self.admittance_per_length, dtype=complex)
        eye = jnp.eye(n)
        # Y (I + R Y)^-1 = (I + Y R)^-1 Y
        yt = jnp.linalg.solve(eye + y * r[None, None, :], y)
        z_ref = jnp.broadcast_to(z0, (n_f, n))
        g = jnp.asarray(z_ref, dtype=complex)
        k = _pseudo(g)
        # K Y K^-1 has entries Y_ij k_i / k_j; S = (I - G N)(I + G N)^-1
        gn = g[:, :, None] * yt * k[:, :, None] / k[:, None, :]
        s = jnp.swapaxes(
            jnp.linalg.solve(
                jnp.swapaxes(eye + gn, 1, 2), jnp.swapaxes(eye - gn, 1, 2)
            ),
            1,
            2,
        )
        return NetworkParameters(
            values=s,
            z_ref=z_ref,
            frequency=self.frequencies,
            ports=self.terminals,
            excited=self.terminals,
        )

    def to_xarray(self) -> xarray.Dataset:
        """The admittance and clamped capacitance as an `xarray.Dataset`.

        ``admittance_per_length`` (S/m) has dimensions ``("frequency",
        "terminal", "excitation")`` and ``clamped_capacitance_per_length``
        (F/m) ``("terminal", "excitation")``, labelled by `terminals` and the
        frequencies. Needs the ``xarray`` extra: ``pip install
        "gfem[xarray]"``.

        Raises:
            ImportError: xarray is not installed.
            TypeError: The result holds JAX tracers.
        """
        xr = _xarray(self)
        data = {
            "admittance_per_length": _variable(
                ("frequency", "terminal", "excitation"),
                self.admittance_per_length,
                "S/m",
            ),
            "clamped_capacitance_per_length": _variable(
                ("terminal", "excitation"), self.clamped_capacitance_per_length, "F/m"
            ),
        }
        coords = {
            "frequency": self.frequency,
            "terminal": list(self.terminals),
            "excitation": list(self.terminals),
        }
        return xr.Dataset(data, coords=coords)

    def __repr__(self) -> str:
        f = self.frequency
        band = hertz_band(f.min(), f.max(), 4)
        what = f"{len(f)} frequencies {band}" if len(f) > 1 else band
        head = f"PiezoelectricResult(terminals={list(self.terminals)}, {what}"
        try:
            c = np.asarray(self.clamped_capacitance_per_length) * 1e12
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"{head})"
        width = max(len(t) for t in self.terminals)
        rows = "\n".join(
            f"  {t:>{width}}: " + "  ".join(f"{v:10.4g}" for v in row)
            for t, row in zip(self.terminals, c, strict=True)
        )
        return f"{head}, clamped capacitance per length in pF/m:\n{rows})"


register_fields(
    PiezoelectricResult,
    (
        "admittance_per_length",
        "clamped_capacitance_per_length",
        "displacement_fields",
        "potential_fields",
    ),
)


@dataclasses.dataclass(frozen=True, eq=False)
class MechanicalEigenmodeResult:
    r"""The result of a `MechanicalEigenmode` problem: modes, nearest first.

    Every array has one entry per mode, in order of increasing distance of
    the mode's frequency from ``MechanicalEigenmode.target``. The modes are
    :math:`u(x, y)\,e^{i\omega t - iqz}` of one axial wavenumber :math:`q`.

    Attributes:
        omega: Complex angular frequencies, rad/s, ``(n_modes,)``. Under
            :math:`e^{+i\omega t}` a decaying mode has
            :math:`\mathrm{Im}\,\omega > 0`.
        error: Backward error of each mode,
            :math:`|(\hat K - \omega^2 M) x| / ((|\hat K| + |\omega|^2 |M|)
            |x|)` for the discrete pencil; at most
            ``SolverOptions.eigen_tolerance``.
        group_velocity: :math:`\mathrm{Re}\,d\omega/dq`, m/s, ``(n_modes,)``,
            the exact derivative of the discrete band: the real part of
            :math:`y^T (\partial_q \hat K)\, x / (2\omega\, y^T M x)` with
            :math:`y` the left eigenvector. The derivative of the decay rate
            is not returned. Not differentiable.
        wavenumber: The axial wavenumber :math:`q` of the solve, rad/m.
        cluster: Integer label per mode, ``(n_modes,)``: modes with the same
            label form a degenerate cluster, frequencies within
            ``SolverOptions.cluster_rtol`` of each other (a simple mode has
            a label of its own, counting from 0 in mode order).
        displacement_fields: One ``FieldMap`` per mode (read them with
            `displacement`): the components ``x``, ``y`` and ``z`` of the
            displacement, in m, on H1 of the solve order over the mechanical
            domain, each with its mode's complex ``omega``. Each mode stores
            1 J/m of energy: :math:`\tfrac12 (\mathrm{Re}\,\omega)^2 \int
            \rho\,|u|^2\,dA`, the time-averaged energy per metre along
            :math:`z` of a lossless mode, and its largest coefficient is real
            and positive.
        potential_fields: One `Field` per mode of the potential, in V, for a
            model with piezoelectric coupling; empty otherwise.
    """

    omega: jax.Array
    error: jax.Array
    group_velocity: jax.Array
    wavenumber: float
    cluster: jax.Array | None = None
    displacement_fields: Sequence[FieldMap] = ()
    potential_fields: Sequence[Field] = ()

    def __post_init__(self) -> None:
        """Hold the fields as tuples; label simple modes."""
        object.__setattr__(self, "wavenumber", float(self.wavenumber))
        # a result read back from a file holds plain mappings
        object.__setattr__(
            self,
            "displacement_fields",
            tuple(
                m
                if isinstance(m, FieldMap)
                else FieldMap(m, label="component", wavenumber=self.wavenumber)
                for m in self.displacement_fields
            ),
        )
        object.__setattr__(self, "potential_fields", tuple(self.potential_fields))
        if self.cluster is None:
            n = int(np.shape(self.omega)[0])
            object.__setattr__(self, "cluster", jnp.arange(n))

    @property
    def frequency(self) -> jax.Array:
        r"""Mode frequencies :math:`\mathrm{Re}\,\omega / (2\pi)`, Hz."""
        return jnp.real(self.omega) / (2 * math.pi)

    @property
    def q(self) -> jax.Array:
        r"""Quality factors :math:`\mathrm{Re}\,\omega / (2\,\mathrm{Im}\,\omega)`.

        ``inf`` for a lossless mode, whose :math:`\omega` is real. The real part,
        not :math:`|\omega|`: the two differ at :math:`O(1/Q^2)`. The axial
        wavenumber is `wavenumber`.
        """
        return jnp.real(self.omega) / (2 * jnp.imag(self.omega))

    @property
    def n_modes(self) -> int:
        """The number of modes."""
        return int(np.shape(self.omega)[0])

    def _check_mode(self, mode: int) -> None:
        if not -self.n_modes <= mode < self.n_modes:
            raise IndexError(
                f"Mode {mode} does not exist: this result has {self.n_modes} "
                "mode(s), numbered from 0."
            )

    def displacement(self, mode: int) -> FieldMap:
        """The displacement of mode ``mode`` (0 = nearest the target), in m.

        Raises:
            IndexError: No such mode.
        """
        self._check_mode(mode)
        return self.displacement_fields[mode]

    def potential(self, mode: int) -> Field:
        """The potential of mode ``mode`` (0 = nearest the target), in V.

        Raises:
            IndexError: No such mode.
            ValueError: The model has no piezoelectric region, so no
                potential was solved.
        """
        self._check_mode(mode)
        if not self.potential_fields:
            raise ValueError(
                "This result has no potential: the potential is solved only "
                "for a model with a piezoelectric tensor in its mechanical "
                "domain, and this model has none."
            )
        return self.potential_fields[mode]

    def __repr__(self) -> str:
        try:
            f = np.asarray(self.frequency)
            q = np.asarray(self.q)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"MechanicalEigenmodeResult({self.n_modes} modes)"
        rows = ", ".join(f"{hertz(a)} (Q {b:.4g})" for a, b in zip(f, q, strict=True))
        where = f", wavenumber={self.wavenumber:g} rad/m" if self.wavenumber else ""
        return f"MechanicalEigenmodeResult([{rows}]{where})"


register_fields(
    MechanicalEigenmodeResult,
    (
        "omega",
        "error",
        "group_velocity",
        "cluster",
        "displacement_fields",
        "potential_fields",
    ),
)
