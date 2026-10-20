"""Boundary conditions and ports: frozen values naming mesh surfaces."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Literal

import jax
import numpy as np

from gfem._checks import check_fixed, is_traced, optional_positive, static_float
from gfem._pytree import register_fields
from gfem._sheet import SheetField

_FIXED = (
    "Boundary and port parameters are not differentiable in gfem v0.1; wrap the "
    "value in jax.lax.stop_gradient or keep it a plain number."
)


def _surface_name(value: Any, owner: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{owner}: surface must be one surface name (a string), got {value!r}. "
            "Give one boundary condition per surface."
        )
    return value


def _check_fill(fill: Any, owner: str) -> None:
    if fill is not None and not isinstance(fill, SheetField):
        raise TypeError(
            f"{owner}: fill takes a gfem.SheetField (one fill value per face of "
            f"the surface), got {type(fill).__name__}; build one with "
            "gfem.SheetField(fill=values) or gfem.SheetField.interpolate(density), "
            "or leave fill=None for the solid sheet."
        )


@dataclasses.dataclass(frozen=True)
class PEC:
    """Perfect electric conductor on a surface: n x E = 0.

    Attributes:
        surface: The surface region's name.
    """

    surface: str
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "PEC")


@dataclasses.dataclass(frozen=True)
class PMC:
    """Perfect magnetic conductor (symmetry plane) on a surface: n x H = 0.

    Exterior surfaces that no boundary condition or port names get this
    condition by default in electromagnetic problems.

    Attributes:
        surface: The surface region's name.
    """

    surface: str
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "PMC")


@dataclasses.dataclass(frozen=True)
class Absorbing:
    """First- or second-order absorbing boundary on an exterior surface.

    Under e^{+iωt}, with k = k0 sqrt(eps_r mu_r) in the material behind the
    surface and n its outward normal, the condition is

        n x (mu_r^-1 curl E) = i (k / mu_r) E_t + β curl[n (n · mu_r^-1 curl E)]

    with β = 1 / (2ik) at order 2 and β = 0 at order 1: the plane-wave
    expansion k_n ≈ k - k_t² / (2k) of an outgoing wave (Jin, *The Finite
    Element Method in Electromagnetics*, §9.3, with the radius of curvature
    taken infinite). Its weak form adds i (k / mu_r) ∮ E_t · v_t and, at
    order 2, (1 / (2ik)) ∮ mu_r^-1 (n · curl E)(n · curl v). Only the curl part of the
    second-order term is kept; the β ∇_t(∇ · E_t) part is dropped, as in
    Palace and COMSOL, because the tangential divergence of a Nédélec trace
    is not square-integrable. Palace ≤ 0.18 uses the opposite sign of the
    curl term, which reflects more than order 1; gfem uses the textbook
    sign only, so on a tight air box the two codes differ at order 2 (they
    agree at order 1 and on adequately sized boxes).

    Attributes:
        surface: The surface region's name.
        order: 1 or 2.
    """

    surface: str
    order: Literal[1, 2] = 1
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "Absorbing")
        if (
            is_traced(self.order)
            or self.order not in (1, 2)
            or isinstance(self.order, bool)
        ):
            raise ValueError(f"Absorbing: order must be 1 or 2, got {self.order!r}.")
        object.__setattr__(self, "order", int(self.order))


@dataclasses.dataclass(frozen=True)
class Conductor:
    """A finite-conductivity metal surface, modelled by its surface impedance.

    Each face adds iω / Z_s(ω) ∫ E_t · v_t, with Z_s the surface impedance of
    one face of a metal film (HFSS / Palace finite-thickness form). On an
    interior surface it is one term for the whole sheet, so a film much
    thinner than its skin depth is a sheet admittance of about sigma t / 2; on
    an exterior surface the metal is one-sided and its thickness counts
    twice (sigma t). This is Palace's convention; see `gfem.conventions`.

    Attributes:
        surface: The surface region's name.
        conductivity: Electrical conductivity of the metal, S/m, > 0.
        thickness: Metal thickness in m, or ``None`` for a thick conductor
            (skin effect only).
        fill: A `SheetField` scaling the sheet's admittance face by face
            (a sheet design, interior surfaces only), or ``None`` for the
            solid sheet.
    """

    surface: str
    conductivity: float
    thickness: float | None = None
    fill: SheetField | None = None
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "Conductor")
        sigma = static_float(self.conductivity, "Conductor.conductivity", _FIXED)
        if sigma <= 0:
            raise ValueError(
                f"Conductor.conductivity must be > 0 S/m, got {sigma}. For a "
                "perfect conductor use gfem.PEC."
            )
        object.__setattr__(self, "conductivity", sigma)
        object.__setattr__(
            self,
            "thickness",
            optional_positive(self.thickness, "Conductor.thickness", "m", _FIXED),
        )
        _check_fill(self.fill, "Conductor")


@dataclasses.dataclass(frozen=True)
class Impedance:
    """A surface impedance: per-square R (ohm), L (H) and C (F) in parallel.

    Attributes:
        surface: The surface region's name.
        resistance: Sheet resistance, ohm per square, or ``None``.
        inductance: Sheet inductance, H per square, or ``None``.
        capacitance: Sheet capacitance, F per square, or ``None``.
        fill: A `SheetField` scaling the sheet's admittance face by face
            (a sheet design, interior surfaces only), or ``None`` for the
            solid sheet.
    """

    surface: str
    resistance: float | None = None
    inductance: float | None = None
    capacitance: float | None = None
    fill: SheetField | None = None
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "Impedance")
        _set_rlc(self, "Impedance")
        if (self.resistance, self.inductance, self.capacitance) == (None, None, None):
            raise ValueError(
                "Impedance needs at least one of resistance, inductance or "
                "capacitance (per square, in parallel)."
            )
        _check_fill(self.fill, "Impedance")


def _set_rlc(obj: Any, owner: str) -> None:
    for name, unit in (
        ("resistance", "ohm"),
        ("inductance", "H"),
        ("capacitance", "F"),
    ):
        value = optional_positive(getattr(obj, name), f"{owner}.{name}", unit, _FIXED)
        object.__setattr__(obj, name, value)


for _cls in (PEC, PMC, Absorbing):
    jax.tree_util.register_static(_cls)
register_fields(Conductor, ("fill",))
register_fields(Impedance, ("fill",))

Boundary = PEC | PMC | Absorbing | Conductor | Impedance


# --- ports ----------------------------------------------------------------------

_AXES = {"x": 0, "y": 1, "z": 2}

Direction = (
    Literal["+x", "-x", "+y", "-y", "+z", "-z", "+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    | tuple[float, float, float]
)


def _port_name(name: Any, owner: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{owner}: name must be a string, got {name!r}.")
    if not name:
        raise ValueError(f"{owner}: name must not be empty.")
    return name


def _direction(value: Any, owner: str) -> tuple[float, float, float]:
    if isinstance(value, str):
        v = value.strip().lower()
        if len(v) == 2 and v[0] in "+-" and v[1] in _AXES:
            out = [0.0, 0.0, 0.0]
            out[_AXES[v[1]]] = 1.0 if v[0] == "+" else -1.0
            return (out[0], out[1], out[2])
        raise ValueError(
            f"{owner}: direction {value!r} is not an axis; use '+x', '-x', '+y', "
            "'-y', '+z', '-z' or a 3-vector."
        )
    check_fixed(value, f"{owner}: direction must be fixed numbers")
    try:
        d = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{owner}: direction {value!r} is not a 3-vector.") from None
    norm = float(np.linalg.norm(d)) if d.shape == (3,) else 0.0
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{owner}: direction {value!r} is not a non-zero 3-vector.")
    d = d / norm
    return (float(d[0]), float(d[1]), float(d[2]))


def _is_one_direction(value: Any) -> bool:
    """A single direction (axis name or 3 numbers), not a sequence of them."""
    if isinstance(value, str):
        return True
    return not any(isinstance(v, str) for v in value) and np.ndim(value) == 1


@dataclasses.dataclass(frozen=True)
class LumpedPort:
    """A lumped port on one surface, or several surfaces driven together.

    Each element (surface) has its own excitation direction; a coplanar
    waveguide port is two elements with opposite directions. The port is
    terminated by R, L and C in parallel, or by a resistive ``impedance``.

    Attributes:
        name: The port's name; ports index S-parameters by name, in the
            order of the model's ``ports``.
        surface: One surface name, or a sequence of them (multi-element).
        direction: The excitation direction of each element: an axis
            ('+x', '-y', ...) or a 3-vector. With several surfaces, a
            sequence of the same length.
        resistance: Termination resistance, ohm.
        inductance: Termination inductance, H.
        capacitance: Termination capacitance, F.
        impedance: A resistive termination in ohm, the same as
            ``resistance``. If no termination is given, 50 ohm.
    """

    name: str
    surface: str | tuple[str, ...]
    direction: Direction | tuple[Direction, ...]
    resistance: float | None = None
    inductance: float | None = None
    capacitance: float | None = None
    impedance: float | None = None
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        owner = f"LumpedPort {self.name!r}"
        _port_name(self.name, "LumpedPort")
        single = isinstance(self.surface, str)
        if single:
            surfaces: tuple[str, ...] = (self.surface,)
            directions = [self.direction]
        else:
            surfaces = tuple(_surface_name(s, owner) for s in self.surface)
            if not surfaces:
                raise ValueError(f"{owner}: needs at least one surface.")
            if len(set(surfaces)) != len(surfaces):
                raise ValueError(f"{owner}: a surface is listed twice.")
            if _is_one_direction(self.direction):
                raise ValueError(
                    f"{owner}: a multi-element port needs one direction per element "
                    f"(surface); got one direction for {len(surfaces)} surfaces."
                )
            directions = list(self.direction)
            if len(directions) != len(surfaces):
                raise ValueError(
                    f"{owner}: needs one direction per element (surface); got "
                    f"{len(directions)} directions for {len(surfaces)} surfaces."
                )
        dirs = tuple(_direction(d, owner) for d in directions)
        object.__setattr__(self, "surface", surfaces[0] if single else surfaces)
        object.__setattr__(self, "direction", dirs[0] if single else dirs)
        _set_rlc(self, "LumpedPort")
        object.__setattr__(
            self,
            "impedance",
            optional_positive(self.impedance, "LumpedPort.impedance", "ohm", _FIXED),
        )
        rlc = (self.resistance, self.inductance, self.capacitance)
        if self.impedance is not None and rlc != (None, None, None):
            raise ValueError(
                f"{owner}: give either impedance (a resistive termination) or "
                "resistance / inductance / capacitance, not both."
            )
        if self.impedance is None and rlc == (None, None, None):
            object.__setattr__(self, "impedance", 50.0)

    @property
    def surfaces(self) -> tuple[str, ...]:
        """The port's surfaces, one per element."""
        return (self.surface,) if isinstance(self.surface, str) else self.surface

    @property
    def elements(self) -> tuple[tuple[str, tuple[float, float, float]], ...]:
        """``(surface, unit direction)`` per element."""
        dirs = (self.direction,) if isinstance(self.surface, str) else self.direction
        return tuple(zip(self.surfaces, dirs, strict=True))  # ty: ignore[invalid-return-type]

    @property
    def rlc(self) -> tuple[float | None, float | None, float | None]:
        """The termination as ``(R, L, C)`` in parallel, ``impedance`` read as R."""
        if self.impedance is not None:
            return (self.impedance, None, None)
        return (self.resistance, self.inductance, self.capacitance)


@dataclasses.dataclass(frozen=True)
class WavePort:
    """A wave port on an exterior surface, fed by its port mode.

    Attributes:
        name: The port's name; ports index S-parameters by name, in the
            order of the model's ``ports``.
        surface: The port's surface region.
        mode: Which port mode excites the port, 1 = the first.
        offset: De-embedding distance in m, >= 0.
        voltage_path: Points (m) of a path across the port; it fixes the
            mode's polarity and defines a voltage, hence a reference
            impedance. ``None``: no voltage.
        pec_edges: Surfaces whose edges on the port are perfect conductors
            in the port mode problem only (where the 3D problem models them
            otherwise, e.g. as a `Conductor`).
    """

    name: str
    surface: str
    mode: int = 1
    offset: float = 0.0
    voltage_path: tuple[tuple[float, float, float], ...] | None = None
    pec_edges: tuple[str, ...] = ()
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        owner = f"WavePort {self.name!r}"
        _port_name(self.name, "WavePort")
        _surface_name(self.surface, owner)
        if (
            isinstance(self.mode, bool)
            or not isinstance(self.mode, (int, np.integer))
            or self.mode < 1
        ):
            raise ValueError(
                f"{owner}: mode must be an integer >= 1, got {self.mode!r}."
            )
        object.__setattr__(self, "mode", int(self.mode))
        offset = static_float(self.offset, "WavePort.offset", _FIXED)
        if offset < 0:
            raise ValueError(f"{owner}: offset must be >= 0 m, got {offset}.")
        object.__setattr__(self, "offset", offset)
        if self.voltage_path is not None:
            object.__setattr__(self, "voltage_path", _path(self.voltage_path, owner))
        edges = (self.pec_edges,) if isinstance(self.pec_edges, str) else self.pec_edges
        object.__setattr__(
            self, "pec_edges", tuple(_surface_name(s, owner) for s in edges)
        )


def _path(value: Any, owner: str) -> tuple[tuple[float, float, float], ...]:
    check_fixed(value, f"{owner}: voltage_path must be fixed numbers")
    try:
        p = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        p = np.empty(0)
    if p.ndim != 2 or p.shape[-1] != 3 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError(
            f"{owner}: voltage_path must be two or more points (m), each a "
            f"3-vector, got {value!r}."
        )
    return tuple((float(r[0]), float(r[1]), float(r[2])) for r in p.reshape(-1, 3))


for _cls in (LumpedPort, WavePort):
    jax.tree_util.register_static(_cls)

Port = LumpedPort | WavePort


def surfaces_of(item: Boundary | Port) -> tuple[str, ...]:
    """The surfaces a boundary condition or port occupies."""
    if isinstance(item, LumpedPort):
        return item.surfaces
    return (item.surface,)
