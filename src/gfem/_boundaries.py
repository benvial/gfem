"""Boundary conditions and ports: frozen values naming mesh surfaces."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import jax
import numpy as np

from gfem._checks import check_fixed, is_traced, optional_positive, static_float
from gfem._pytree import register_fields
from gfem._sheet import SheetField

_FIXED = (
    "Of the boundary and port values, gfem differentiates a lumped port's "
    "inductance and capacitance and an Impedance's resistance, inductance and "
    "capacitance only; wrap this value in jax.lax.stop_gradient or keep it a "
    "plain number."
)
#: The lumped values of a lumped port or an `Impedance`, with their units.
_RLC = (("resistance", "ohm"), ("inductance", "H"), ("capacitance", "F"))


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
    r"""First- or second-order absorbing boundary on an exterior surface.

    Under :math:`e^{+i\omega t}`, with
    :math:`k = k_0 \sqrt{\varepsilon_r \mu_r}` in the material behind the
    surface and :math:`n` its outward normal, the condition is

    .. math::

        n \times (\mu_r^{-1} \nabla \times E) = i \frac{k}{\mu_r} E_t
            + \beta \nabla \times [n (n \cdot \mu_r^{-1} \nabla \times E)]

    with :math:`\beta = 1 / (2ik)` at order 2 and :math:`\beta = 0` at order
    1: the plane-wave expansion :math:`k_n \approx k - k_t^2 / (2k)` of an
    outgoing wave (Jin, *The Finite Element Method in Electromagnetics*,
    §9.3, with the radius of curvature taken infinite). Its weak form adds
    :math:`i (k / \mu_r) \oint E_t \cdot v_t` and, at order 2,
    :math:`\frac{1}{2ik} \oint \mu_r^{-1} (n \cdot \nabla \times E)
    (n \cdot \nabla \times v)`. Only the curl part of the second-order term
    is kept; the :math:`\beta \nabla_t (\nabla \cdot E_t)` part is dropped, as in
    COMSOL, because the tangential divergence of a Nédélec trace is not
    square-integrable. gfem uses the textbook sign of the curl term only;
    the opposite sign reflects more than order 1, so on a tight air box a
    solver using it differs from gfem at order 2 (they agree at order 1 and
    on adequately sized boxes).

    Both orders are exact at normal incidence only. A plane wave arriving at
    the angle :math:`\theta` from the normal is reflected with amplitude
    :math:`(1 - \cos\theta) / (1 + \cos\theta)` at order 1 (0.017 at 15°,
    0.072 at 30°, 0.17 at 45°), for both polarizations. Order 2 lowers this
    about tenfold up to 30° for the polarization with :math:`E` parallel to
    the surface and normal to :math:`k_t`; for the
    other one (E in the plane of incidence, no normal magnetic field) the
    curl term vanishes and the reflection is that of order 1. In a periodic
    cell under a Bloch wave vector with a component :math:`k_t` along the
    surface, :math:`\sin\theta = |k_t| / k`.

    Next to an anisotropic material, order 1 uses the normal-incidence
    admittance tensor of that material for the face normal
    (:math:`i k_0 \oint (Y E_t) \cdot v_t`, exact for a plane wave at normal
    incidence; real parts of :math:`\varepsilon` and :math:`\mu`, as for
    isotropic faces). Order 2 raises `NotImplementedError` there.

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
    r"""A finite-conductivity metal surface, modelled by its surface impedance.

    Each face adds :math:`\frac{i\omega}{Z_s(\omega)} \int E_t \cdot v_t`,
    with :math:`Z_s` the surface impedance of one face of a metal film (HFSS
    finite-thickness form). On an interior surface it is one term for the
    whole sheet, so a film much thinner than its skin depth is a sheet
    admittance of about :math:`\sigma t / 2`; on an exterior surface the
    metal is one-sided and its thickness counts twice (:math:`\sigma t`);
    see `gfem.conventions`.

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

    The three values are design inputs (pytree leaves): driven
    S-parameters, kept fields and eigenfrequencies differentiate with respect
    to them. A value built under `jax.grad`, `jax.jit` or `jax.vmap` is
    checked when the solve runs, where a non-finite or non-positive one
    raises `ValueError`. On a surface that shares an edge with a wave port's
    face the port mode reads them, so differentiating them there raises
    `gfem.PortGeometryError`.

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
        _set_rlc(self, "Impedance", traced=("resistance", "inductance", "capacitance"))
        if all(v is None for v in (self.resistance, self.inductance, self.capacitance)):
            raise ValueError(
                "Impedance needs at least one of resistance, inductance or "
                "capacitance (per square, in parallel)."
            )
        _check_fill(self.fill, "Impedance")


@dataclasses.dataclass(frozen=True)
class Periodic:
    """A periodic pair: two exterior surfaces related by a translation.

    The fields on the two surfaces are identified node for node. The
    translation a runs from ``surfaces[0]`` to ``surfaces[1]`` and is
    measured from the mesh (`Model.periodic_translations`); the two surface
    meshes must be identical up to it (gmsh: ``setPeriodic``). A model takes
    one to three pairs, along independent directions. With no Bloch wave
    vector on the problem the fields are equal on both surfaces (plain
    periodicity); with ``wave_vector=k`` (rad/m) on `gfem.Driven` or
    `gfem.Eigenmode`, E(r + a) = exp(-i k·a) E(r) (`gfem.conventions`,
    Bloch periodicity). The order of the two surfaces is not physical:
    swapping them flips a and gives the same field.

    `gfem.Eigenmode` and `gfem.Driven` solve periodic models: a driven
    problem with a wave vector on the uniform sweep only, without one on
    both sweeps. `gfem.Electrostatic` raises on a periodic model. Across MPI
    ranks a model with one pair solves both problems, a Bloch eigenmode solve
    excepted; two or three pairs raise `gfem.DistributedUnsupportedError`. A paired
    surface takes no other boundary or port, and the default PMC skips it.
    The two meshes are checked when the `gfem.Model` is built. See the
    "Solve a periodic cell" recipe.

    Attributes:
        surfaces: The two surface region names (first, second).

    Raises:
        TypeError: ``surfaces`` is not two surface names.
        ValueError: The same name is given twice.

    Example:
        ``gfem.Periodic(("x_min", "x_max"))``
    """

    surfaces: tuple[str, str]
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        value = self.surfaces
        if (
            isinstance(value, str)
            or not isinstance(value, (list, tuple))
            or len(value) != 2
            or not all(isinstance(s, str) for s in value)
        ):
            raise TypeError(
                "Periodic: surfaces must be two surface names (first, second), "
                f"got {value!r}."
            )
        if value[0] == value[1]:
            raise ValueError(
                f"Periodic: the two surfaces must differ, got {value[0]!r} twice; "
                "a periodic pair joins two opposite surfaces of the cell."
            )
        object.__setattr__(self, "surfaces", (value[0], value[1]))


#: The attribute names of the thermal boundary values (the pytree leaves).
THERMAL_VALUES = ("temperature", "coefficient", "ambient", "flux")


def _temperature(value: Any, what: str) -> float:
    """An absolute temperature: a finite host float > 0 K."""
    out = static_float(value, what, "")
    if out <= 0:
        raise ValueError(
            f"{what} must be > 0 K, got {out}: temperatures are absolute, in "
            "kelvin (for example 293.15 for 20 degrees Celsius), not a rise."
        )
    return out


def _coefficient(value: Any, what: str) -> float:
    """A heat transfer coefficient: a finite host float > 0."""
    out = static_float(value, what, "")
    if out <= 0:
        raise ValueError(
            f"{what} must be > 0 W/(m^2 K), got {out}; leave the surface "
            "without a thermal condition for an insulated one."
        )
    return out


def _flux(value: Any, what: str) -> float:
    """A heat flux: a finite host float of any sign."""
    return static_float(value, what, "")


#: The check of each thermal value, by attribute name.
_THERMAL_CHECKS = {
    "temperature": _temperature,
    "ambient": _temperature,
    "coefficient": _coefficient,
    "flux": _flux,
}


def _set_thermal(obj: Any, owner: str) -> None:
    """Check the thermal values of ``obj``; a traced one is checked per forward.

    A concrete value is checked and stored as a host float. A traced one (under
    `jax.grad`, `jax.jit` or `jax.vmap`) is kept as it is: its value is checked on
    the host, on every forward (`check_thermal_values`).
    """
    for name in THERMAL_VALUES:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if is_traced(value):
            continue
        object.__setattr__(obj, name, _THERMAL_CHECKS[name](value, f"{owner}.{name}"))


def check_thermal_values(item: Any) -> None:
    """Raise unless every thermal value of ``item`` is a valid host number.

    The constructor checks concrete values; a value built under a JAX
    transformation reaches the host unchecked, so every thermal forward checks
    the values it reads here.

    Raises:
        ValueError: A value is not finite, a temperature is not > 0 K, or a
            heat transfer coefficient is not > 0.
    """
    owner = type(item).__name__
    for name in THERMAL_VALUES:
        if hasattr(item, name):
            value = getattr(item, name)
            out = np.asarray(value)
            if out.shape or np.iscomplexobj(out):
                raise ValueError(
                    f"{owner}.{name} must be a real number, got {value!r}."
                )
            _THERMAL_CHECKS[name](out.item(), f"{owner}.{name}")


@dataclasses.dataclass(frozen=True)
class FixedTemperature:
    """A fixed absolute temperature on a surface (a heat sink), for thermal problems.

    It may sit on any surface that touches the thermal domain, exterior or
    interior; on an interior surface both sides see it. Two fixed
    temperatures with different values must not share a mesh node. Where it
    meets a `Convection` or `HeatFlux` surface, the fixed value wins on the
    shared nodes.

    Attributes:
        surface: The surface region's name.
        temperature: The temperature, absolute, in K (> 0). May be traced by
            JAX: the gradient of a thermal result with respect to it is the
            adjoint's reaction on the surface. Two touching surfaces with one
            traced value give the gradient of that common value.
    """

    surface: str
    _: dataclasses.KW_ONLY
    temperature: float
    _physics: ClassVar[str] = "thermal"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "FixedTemperature")
        _set_thermal(self, "FixedTemperature")


@dataclasses.dataclass(frozen=True)
class Convection:
    """Convective cooling of an exterior surface, for thermal problems.

    The heat flux out of the domain is h (T - T_ambient), with h the heat
    transfer coefficient. The surface must be exterior to the thermal domain:
    an interior face has no ambient.

    Attributes:
        surface: The surface region's name.
        coefficient: The heat transfer coefficient h, W/(m^2 K), > 0.
        ambient: The ambient temperature, absolute, in K (> 0). Both
            ``coefficient`` and ``ambient`` may be traced by JAX; the rise
            reads only ``coefficient``.
    """

    surface: str
    _: dataclasses.KW_ONLY
    coefficient: float
    ambient: float
    _physics: ClassVar[str] = "thermal"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "Convection")
        _set_thermal(self, "Convection")


@dataclasses.dataclass(frozen=True)
class HeatFlux:
    """A heat flux through a surface, for thermal problems.

    On an exterior surface it is the flux into the thermal domain; on an
    interior surface it is a surface heat source of ``flux`` W per m^2 of
    surface, split between the two sides by the solve.

    Attributes:
        surface: The surface region's name.
        flux: W/m^2, any sign, positive into the thermal domain. May be
            traced by JAX, also at 0; the rise does not read it.
    """

    surface: str
    _: dataclasses.KW_ONLY
    flux: float
    _physics: ClassVar[str] = "thermal"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "HeatFlux")
        _set_thermal(self, "HeatFlux")


def _set_rlc(obj: Any, owner: str, traced: tuple[str, ...]) -> None:
    """Check the lumped values of ``obj``; those named in ``traced`` may be traced.

    A traced value is kept as it is: its value is checked on the host, on every
    forward (`check_lumped_values`). A concrete one is checked here.
    """
    for name, unit in _RLC:
        value = getattr(obj, name)
        if name in traced and is_traced(value):
            continue
        value = optional_positive(value, f"{owner}.{name}", unit, _FIXED)
        object.__setattr__(obj, name, value)


def lumped_label(item: LumpedPort | Impedance) -> str:
    """``LumpedPort 'p1'`` or ``Impedance on surface 's'``, for messages."""
    if isinstance(item, LumpedPort):
        return f"LumpedPort {item.name!r}"
    return f"Impedance on surface {item.surface!r}"


def check_lumped_values(item: LumpedPort | Impedance) -> None:
    """Raise unless every lumped value of ``item`` is a finite number > 0.

    The constructor checks concrete values; a value built under `jax.grad`,
    `jax.jit` or `jax.vmap` reaches the host unchecked, so every forward checks
    the values it reads here.

    Raises:
        ValueError: A value is not finite, or not > 0.
    """
    for name, unit in _RLC:
        value = getattr(item, name)
        if value is None:
            continue
        out = np.asarray(value)
        if out.shape or np.iscomplexobj(out) or not np.isfinite(out) or out <= 0:
            raise ValueError(
                f"{lumped_label(item)}: {name} must be a finite number > 0 {unit}, "
                f"got {out.item() if out.ndim == 0 else value!r}."
            )


for _cls in (PEC, PMC, Absorbing, Periodic):
    jax.tree_util.register_static(_cls)
register_fields(FixedTemperature, ("temperature",))
register_fields(Convection, ("coefficient", "ambient"))
register_fields(HeatFlux, ("flux",))
register_fields(Conductor, ("fill",))
register_fields(Impedance, ("resistance", "inductance", "capacitance", "fill"))

#: A boundary condition of a model: any one of the classes this union names
#: (electromagnetic conditions and the thermal `FixedTemperature`,
#: `Convection` and `HeatFlux`).
Boundary = (
    PEC
    | PMC
    | Absorbing
    | Conductor
    | Impedance
    | Periodic
    | FixedTemperature
    | Convection
    | HeatFlux
)


# --- ports ----------------------------------------------------------------------

_AXES = {"x": 0, "y": 1, "z": 2}

Direction = (
    Literal[
        "x",
        "y",
        "z",
        "X",
        "Y",
        "Z",
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
        "+X",
        "-X",
        "+Y",
        "-Y",
        "+Z",
        "-Z",
    ]
    | tuple[float, float, float]
)

#: A wave port's polarity reference as stored: the point (m; 3 coordinates,
#: or 2 on a line port) and the unit direction.
Polarity = tuple[tuple[float, ...], tuple[float, float, float]]


def _port_name(name: Any, owner: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{owner}: name must be a string, got {name!r}.")
    if not name:
        raise ValueError(f"{owner}: name must not be empty.")
    return name


def _direction(value: Any, owner: str) -> tuple[float, float, float]:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _AXES:  # no sign means the positive axis
            v = "+" + v
        if len(v) == 2 and v[0] in "+-" and v[1] in _AXES:
            out = [0.0, 0.0, 0.0]
            out[_AXES[v[1]]] = 1.0 if v[0] == "+" else -1.0
            return (out[0], out[1], out[2])
        raise ValueError(
            f"{owner}: direction {value!r} is not an axis; use 'x', 'y', 'z' with "
            "an optional sign ('+x', '-z'; no sign means '+') or a 3-vector."
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

    Example:
        A coplanar waveguide port: one gap on each side of the signal line,
        each excited from the signal line towards ground::

            port = gfem.LumpedPort(
                "p1", surface=("gap_left", "gap_right"), direction=("-x", "+x")
            )

    Attributes:
        name: The port's name; ports index S-parameters by name, in the
            order of the model's ``ports``.
        surface: One surface name, or a sequence of them (multi-element).
        direction: The excitation direction of each element, along its
            surface from one conductor to the other: the reference direction
            of E across the gap, so it sets the polarity of the port voltage.
            An axis ('+x', '-y', ...; no sign means +) or a 3-vector. With
            several surfaces, a sequence of the same length.
        resistance: Termination resistance, ohm. It is the port's reference
            impedance, a fixed value: a traced one raises
            `gfem.NonDifferentiableError`.
        inductance: Termination inductance, H. A design input (a pytree
            leaf): driven S-parameters, kept fields and eigenfrequencies
            differentiate with respect to it.
        capacitance: Termination capacitance, F. A design input, like
            ``inductance``.
        impedance: A resistive termination in ohm, the same as
            ``resistance``. If no termination is given, 50 ohm.

    A traced inductance or capacitance is checked when the solve runs, where a
    non-finite or non-positive one raises `ValueError`.
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
        _set_rlc(self, "LumpedPort", traced=("inductance", "capacitance"))
        object.__setattr__(
            self,
            "impedance",
            optional_positive(self.impedance, "LumpedPort.impedance", "ohm", _FIXED),
        )
        none = all(
            v is None for v in (self.resistance, self.inductance, self.capacitance)
        )
        if self.impedance is not None and not none:
            raise ValueError(
                f"{owner}: give either impedance (a resistive termination) or "
                "resistance / inductance / capacitance, not both."
            )
        if self.impedance is None and none:
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
    r"""A wave port on an exterior surface, fed by its port mode.

    Several wave ports on one surface each request a different mode (a
    shared port surface). A propagating mode no port on the surface requests
    is partly reflected, not converted.

    In a `gfem.Eigenmode` problem the port absorbs its requested mode at the
    complex eigenfrequency and reports a coupling Q in
    ``EigenmodeResult.q_port``.

    The cross-section may contain anisotropic materials when every tensor
    has a principal axis along the port normal (any orientation in the port
    plane); off-axis tensors would make forward and backward port modes
    differ. Elsewhere in the model the principal axes may point in any
    direction.

    Example:
        Arrows that fix the sign of each port's mode, so S keeps its
        relative port signs when the device is rotated, mirrored or
        remeshed. On a straight guide the same arrow at both ends; round a
        bend the arrow follows the field::

            gfem.WavePort("p1", "in", polarity=((0.0, 0.2e-6, 0.0), "+y"))
            gfem.WavePort("p2", "out", polarity=((5e-6, 40e-6, 0.0), "-x"))

    Attributes:
        name: The port's name; ports index S-parameters by name, in the
            order of the model's ``ports``.
        surface: The port's surface region.
        mode: Which port mode excites the port: its rank at the lowest
            frequency among the modes the model admits (a symmetry plane
            halves them), 1 = the most confined.
        offset: De-embedding distance in m, >= 0.
        voltage_path: Points (m) of a path across the port; it fixes the
            mode's polarity and defines a voltage, hence a reference
            impedance. ``None``: no voltage.
        pec_edges: Surfaces whose edges on the port are perfect conductors
            in the port mode problem only (where the 3D problem models them
            otherwise, e.g. as a `Conductor`).
        polarity: ``(point, direction)``: the mode's field at the point
            along the direction is real and positive at every frequency,
            which fixes the sign and phase of the port's row and column of
            S without defining a voltage. The point is in m on the port
            ((x, y, z), or (x, y) on a line port of a 2D mesh), the
            direction an axis ('+y'; no sign means +) or a 3-vector, stored
            as a unit 3-vector. In 3D the arrow reads the in-plane E field;
            on a line port it is '+z' or '-z' and reads the solved field
            normal to the plane (:math:`E_z` or :math:`H_z`). Put the point
            where the mode is strong and the direction along its field: the
            solve raises `gfem.PortModeError` where the field read there is
            below :math:`10^{-2}` of the port's peak, at any frequency. ``None``: the
            default sign (see `gfem.conventions`).
        allow_evanescent: The mode may be below cutoff (evanescent) at
            frequencies of the band, on a line port of a 2D mesh. Its row and
            column of S are then pseudo-waves that carry no power, so
            :math:`|S|` may exceed 1, and :math:`n_\mathrm{eff} = -i\alpha/k_0`
            with :math:`\gamma = \alpha > 0`. Set by `WavePort.kept_modes`.
            Without it a mode below cutoff raises `gfem.PortModeError`.

    Raises:
        ValueError: ``mode`` is not a positive integer, ``offset`` is negative,
            ``voltage_path`` is not two or more 3-vectors, ``polarity`` is
            not a point of 2 or 3 finite coordinates and a direction, both
            ``polarity`` and ``voltage_path`` are given, or
            ``allow_evanescent`` comes with a ``voltage_path`` or a nonzero
            ``offset``.
        NonDifferentiableError: ``offset``, ``voltage_path`` or
            ``polarity`` is computed inside `jax.grad`, `jax.jit` or
            `jax.vmap`.
    """

    name: str
    surface: str
    mode: int = 1
    offset: float = 0.0
    voltage_path: tuple[tuple[float, float, float], ...] | None = None
    pec_edges: tuple[str, ...] = ()
    polarity: Polarity | None = None
    allow_evanescent: bool = False
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
        if self.polarity is not None:
            if self.voltage_path is not None:
                raise ValueError(
                    f"{owner}: give a polarity or a voltage_path, not both: each "
                    "fixes the sign of the port's mode (a voltage_path also "
                    "defines a voltage)."
                )
            object.__setattr__(self, "polarity", _polarity(self.polarity, owner))
        if self.allow_evanescent and self.voltage_path is not None:
            raise ValueError(
                f"{owner}: allow_evanescent does not combine with a voltage_path: "
                "a mode below cutoff carries no power, so it defines no voltage."
            )
        if self.allow_evanescent and offset != 0:
            raise ValueError(
                f"{owner}: allow_evanescent does not combine with an offset: "
                "moving the plane of a mode below cutoff would scale its "
                "pseudo-wave by a growing exponential."
            )

    @classmethod
    def kept_modes(
        cls,
        section: str,
        surface: str,
        modes: int,
        *,
        pec_edges: Iterable[str] | str = (),
        polarity: Mapping[int, Polarity] | None = None,
    ) -> tuple[WavePort, ...]:
        r"""The internal plane ports of a uniform section: modes :math:`1..N`.

        Returns one wave port per kept mode on ``surface``, named
        ``f"{section}@{m}"``, each with ``mode=m`` and
        ``allow_evanescent=True``. A mode cascade joins the ports of two
        blocks that carry the same section name.

        Args:
            section: The section's name, without ``"@"``.
            surface: The port surface of the plane.
            modes: The number :math:`N` of kept modes, >= 1.
            pec_edges: As ``WavePort.pec_edges``, for every port.
            polarity: ``{rank: (point, direction)}``, as ``WavePort.polarity``
                of the port of that rank; ranks without an entry keep the
                default sign.

        Raises:
            ValueError: ``modes`` is below 1, ``section`` contains ``"@"``, or
                a ``polarity`` key falls outside :math:`1..N`.
        """
        if isinstance(modes, bool) or not isinstance(modes, (int, np.integer)):
            raise TypeError(f"kept_modes: modes must be an integer, got {modes!r}.")
        if modes < 1:
            raise ValueError(f"kept_modes: modes must be >= 1, got {modes}.")
        if not isinstance(section, str) or "@" in section:
            raise ValueError(
                f"kept_modes: the section name must be a string without '@' "
                f"(ports are named 'section@rank'), got {section!r}."
            )
        pol = dict(polarity or {})
        bad = sorted(k for k in pol if k not in range(1, int(modes) + 1))
        if bad:
            raise ValueError(
                f"kept_modes: polarity keys must be mode ranks 1..{int(modes)}, "
                f"got {bad}."
            )
        return tuple(
            cls(
                f"{section}@{m}",
                surface,
                mode=m,
                pec_edges=pec_edges,  # ty: ignore[invalid-argument-type]
                polarity=pol.get(m),
                allow_evanescent=True,
            )
            for m in range(1, int(modes) + 1)
        )


def _polarity(value: Any, owner: str) -> Polarity:
    owner = f"{owner} polarity"
    check_fixed(value, f"{owner}: the point and direction must be fixed numbers")
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            f"{owner}: give (point, direction), e.g. ((0.0, 1e-6, 0.0), '+y'), "
            f"got {value!r}."
        )
    point, direction = value
    check_fixed(point, f"{owner}: the point must be fixed numbers")
    try:
        p = np.asarray(point, dtype=float)
    except (TypeError, ValueError):
        p = np.empty((0, 0))
    if p.ndim != 1 or len(p) not in (2, 3):
        raise ValueError(
            f"{owner}: the point must be 2 or 3 coordinates (m): (x, y, z) on a "
            f"3D mesh, (x, y) on a 2D one, got {point!r}."
        )
    if not np.isfinite(p).all():
        raise ValueError(f"{owner}: the point must be finite, got {point!r}.")
    d = _direction(direction, owner)
    if not isinstance(direction, str):
        given = np.asarray(direction, dtype=float)
        if abs(float(np.linalg.norm(given)) - 1.0) <= 4 * np.finfo(float).eps:
            # already unit: kept as given, so a decoded job equals the model
            d = (float(given[0]), float(given[1]), float(given[2]))
    return tuple(float(x) for x in p), d


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


@dataclasses.dataclass(frozen=True)
class FloquetPort:
    r"""A Floquet port: one diffraction order in one polarization.

    It sits on an open face of a periodic cell (a Floquet surface): an
    exterior planar face whose rim lies on the seams of two `Periodic` pairs
    whose translations :math:`a_1`, :math:`a_2` span the face, with one
    homogeneous lossless isotropic medium behind it. The order :math:`(m, n)`
    is the plane wave with in-plane wave vector
    :math:`\kappa = k_t + m b_1 + n b_2`
    (:math:`b_i \cdot a_j = 2\pi \delta_{ij}`, :math:`k_t` the problem's
    ``wave_vector``) in that medium. Several Floquet ports share one surface,
    one (order, polarization) each; the surface takes no other port or
    boundary. `Model.propagating_orders` lists the orders that
    propagate and `Model.bloch_wave_vector` turns incidence angles into a
    wave vector. See `gfem.conventions` for the lattice frame and the TE/TM
    references.

    Attributes:
        name: The port's name; ports index S-parameters by name, in the
            order of the model's ``ports``.
        surface: The Floquet surface.
        order: The diffraction order (m, n), two integers; (0, 0) is the
            specular order.
        polarization: ``"TE"`` (s: :math:`E` along
            :math:`\hat e_3 \times \hat\kappa`) or ``"TM"`` (p: :math:`E`
            tangential along :math:`\hat\kappa`), relative to the order's
            plane of incidence.
        offset: De-embedding distance in m, >= 0, along the face normal.

    Raises:
        TypeError: ``name`` or ``surface`` is not a string, or ``order`` is
            not two integers.
        ValueError: An empty name, an unknown polarization or a negative or
            non-finite offset.
        NonDifferentiableError: ``offset`` is traced by JAX.

    Example:
        ``gfem.FloquetPort("T-1_TE", "bottom", order=(-1, 0), polarization="TE")``
    """

    name: str
    surface: str
    _: dataclasses.KW_ONLY
    order: tuple[int, int] = (0, 0)
    polarization: Literal["TE", "TM"]
    offset: float = 0.0
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        owner = f"FloquetPort {self.name!r}"
        _port_name(self.name, "FloquetPort")
        _surface_name(self.surface, owner)
        order = self.order
        if (
            isinstance(order, str)
            or not isinstance(order, (list, tuple))
            or len(order) != 2
            or not all(
                isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_))
                for i in order
            )
        ):
            raise TypeError(
                f"{owner}: order must be two integers (m, n), got {order!r}."
            )
        object.__setattr__(self, "order", (int(order[0]), int(order[1])))
        if self.polarization not in ("TE", "TM"):
            raise ValueError(
                f"{owner}: polarization must be 'TE' or 'TM', got "
                f"{self.polarization!r}."
            )
        offset = static_float(self.offset, "FloquetPort.offset", _FIXED)
        if offset < 0:
            raise ValueError(f"{owner}: offset must be >= 0 m, got {offset}.")
        object.__setattr__(self, "offset", offset)


jax.tree_util.register_static(WavePort)
jax.tree_util.register_static(FloquetPort)
register_fields(LumpedPort, ("inductance", "capacitance"))

#: A port of a model: a lumped port, a wave port or a Floquet port.
Port = LumpedPort | WavePort | FloquetPort


def sheet_of(item: Boundary | Port) -> SheetField | None:
    """The sheet field (``fill=``) a boundary carries, or ``None``."""
    return item.fill if isinstance(item, (Conductor, Impedance)) else None


def surfaces_of(item: Boundary | Port) -> tuple[str, ...]:
    """The surfaces a boundary condition or port occupies."""
    if isinstance(item, (LumpedPort, Periodic)):
        return item.surfaces
    return (item.surface,)


def wave_port_surfaces(ports: Iterable[Port]) -> dict[str, tuple[WavePort, ...]]:
    """The wave ports among ``ports`` by surface, in model order.

    Surfaces come in the order of their first wave port; a surface with
    several wave ports is a shared port surface.
    """
    surfaces: dict[str, list[WavePort]] = {}
    for port in ports:
        if isinstance(port, WavePort):
            surfaces.setdefault(port.surface, []).append(port)
    return {surface: tuple(group) for surface, group in surfaces.items()}


def first_wave_port(ports: Iterable[Port]) -> WavePort | None:
    """The first wave port among ``ports`` in model order, ``None`` without one."""
    return next((p for p in ports if isinstance(p, WavePort)), None)


def lead_port(group: Sequence[WavePort]) -> WavePort:
    """The wave port with the smallest ``mode`` on a surface.

    Its propagation constant is the surface's Robin ``gamma_0``.
    """
    return min(group, key=lambda p: p.mode)


def wave_port_label(
    group: Sequence[WavePort], *, lone_surface: bool = False, modes: bool = False
) -> str:
    """``WavePorts 'a', 'b' on surface 's'`` for a surface's wave ports.

    Args:
        group: The wave ports of one surface.
        lone_surface: Name the surface of a lone wave port too
            (``WavePort 'a' on surface 's'``); else ``WavePort 'a'``.
        modes: Follow each name with its mode (``'a' (mode 1)``).
    """
    names = ", ".join(
        f"{p.name!r} (mode {p.mode})" if modes else repr(p.name) for p in group
    )
    surface = f" on surface {group[0].surface!r}"
    if len(group) == 1:
        return f"WavePort {names}{surface if lone_surface else ''}"
    return f"WavePorts {names}{surface}"
