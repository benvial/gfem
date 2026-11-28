r"""The sources of a driven problem: incident fields, point dipoles and polarizations.

A source is one right-hand-side column of a `gfem.Driven` problem beside its
excited ports, named like a port. Its parameters are fixed values: gfem does
not differentiate with respect to an angle, a waist, a position or a moment.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, get_args

import jax
import numpy as np

from gfem._checks import is_traced, static_float
from gfem._pytree import register_fields
from gfem.errors import NonDifferentiableError

__all__ = [
    "ElectricDipole",
    "GaussianBeam",
    "IncidentField",
    "MagneticDipole",
    "PlaneWave",
    "PolarizationSource",
    "Source",
]

#: A named polarization: the Jones pair :math:`(a_\mathrm{TM}, a_\mathrm{TE})`
#: it stands for.
Polarization = Literal["TE", "TM", "RCP", "LCP"]
_NAMED = {
    "TE": (0.0, 1.0),
    "TM": (1.0, 0.0),
    "RCP": (math.sqrt(0.5), -1j * math.sqrt(0.5)),
    "LCP": (math.sqrt(0.5), 1j * math.sqrt(0.5)),
}
_POLARIZATIONS = get_args(Polarization)
_FIXED = (
    "Build the source from concrete values, outside the transformed function: "
    "source parameters are constants of the solve."
)


def propagation_basis(theta: float, phi: float) -> tuple[np.ndarray, ...]:
    r"""The unit vectors :math:`(\hat k, \hat\theta, \hat\phi)` of a direction.

    :math:`\hat k = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)`,
    :math:`\hat\theta = \partial_\theta \hat k`, :math:`\hat\phi = \hat z
    \times \hat\kappa` with :math:`\hat\kappa = (\cos\phi, \sin\phi, 0)`, so
    :math:`(\hat\theta, \hat\phi, \hat k)` is right-handed and :math:`\phi`
    fixes the plane of incidence along :math:`\pm\hat z`.
    """
    st, ct, sp, cp = math.sin(theta), math.cos(theta), math.sin(phi), math.cos(phi)
    k_hat = np.array([st * cp, st * sp, ct])
    theta_hat = np.array([ct * cp, ct * sp, -st])
    phi_hat = np.array([-sp, cp, 0.0])
    return k_hat, theta_hat, phi_hat


def jones(polarization: Polarization | tuple[complex, complex]) -> np.ndarray:
    r"""The normalized Jones pair :math:`(a_\mathrm{TM}, a_\mathrm{TE})`."""
    pair = np.asarray(_NAMED.get(polarization, polarization), dtype=complex)
    return pair / np.linalg.norm(pair)


def polarization_vector(
    theta: float, phi: float, polarization: Polarization | tuple[complex, complex]
) -> np.ndarray:
    r"""The unit complex field direction of a polarization.

    :math:`a_\mathrm{TM}\hat\theta + a_\mathrm{TE}\hat\phi`, with the pair
    :math:`a` normalized and the basis of `propagation_basis`.
    """
    _, theta_hat, phi_hat = propagation_basis(theta, phi)
    a_tm, a_te = jones(polarization)
    return a_tm * theta_hat + a_te * phi_hat


# --- validation ------------------------------------------------------------------


def _name(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(
            f"{owner}.name must be a non-empty string, the label of its column "
            f"and of its kept fields; got {value!r}."
        )
    return value


def _angle(value: Any, what: str, *, polar: bool) -> float:
    out = static_float(value, what, _FIXED)
    if polar and not 0.0 <= out <= math.pi:
        raise ValueError(
            f"{what} must be in [0, pi] rad (the propagation direction's angle "
            f"from +z), got {out}."
        )
    return out


def _check_traced(value: Any, what: str) -> None:
    if is_traced(value):
        raise NonDifferentiableError(
            f"{what} is a fixed value and gfem does not differentiate with "
            f"respect to it. {_FIXED}"
        )


def _complex(value: Any, what: str) -> complex:
    _check_traced(value, what)
    try:
        out = complex(np.asarray(value).item())
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a complex number, got {value!r}.") from None
    if not (math.isfinite(out.real) and math.isfinite(out.imag)):
        raise ValueError(f"{what} must be finite, got {out}.")
    return out


def _polarization(value: Any, owner: str) -> Polarization | tuple[complex, complex]:
    what = f"{owner}.polarization"
    if isinstance(value, str):
        if value not in _POLARIZATIONS:
            raise ValueError(
                f"{what} must be 'TE', 'TM', 'RCP', 'LCP' or a Jones pair "
                f"(a_TM, a_TE), got {value!r}."
            )
        return value
    _check_traced(value, what)
    try:
        pair = np.asarray(value, dtype=complex)
    except (TypeError, ValueError):
        pair = np.empty(0)
    if pair.shape != (2,):
        raise ValueError(
            f"{what} must be 'TE', 'TM', 'RCP', 'LCP' or a Jones pair "
            f"(a_TM, a_TE) of two complex numbers, got {value!r}."
        )
    if not np.isfinite(pair).all():
        raise ValueError(f"{what}: the Jones pair {value!r} must be finite.")
    if not np.abs(pair).any():
        raise ValueError(
            f"{what}: the Jones pair {value!r} is zero, so it has no direction."
        )
    return (complex(pair[0]), complex(pair[1]))


def _vector(value: Any, what: str, *, dtype: type, unit: str) -> tuple:
    _check_traced(value, what)
    try:
        arr = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError):
        arr = np.empty(0)
    if arr.shape != (3,):
        raise ValueError(f"{what} must be 3 {unit}, got {value!r}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{what} must be 3 finite {unit}, got {value!r}.")
    return tuple(dtype(x) for x in arr)


def _point(value: Any, what: str) -> tuple[float, float, float]:
    return _vector(value, what, dtype=float, unit="coordinates")


def _moment(value: Any, what: str) -> tuple[complex, complex, complex]:
    return _vector(value, what, dtype=complex, unit="components")


def _set(obj: Any, **values: Any) -> None:
    for name, value in values.items():
        object.__setattr__(obj, name, value)


# --- incident fields ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, init=False)
class PlaneWave:
    r"""A plane wave of given direction, polarization and peak amplitude.

    The incident field in the background stack (``Model(background=...)``)
    is, in a homogeneous background of wavenumber :math:`k`,

    .. math::

        E_\mathrm{inc}(r) = E_0\,
            \frac{a_\mathrm{TM}\hat\theta + a_\mathrm{TE}\hat\phi}{\lVert a\rVert}
            \, e^{-i k \hat k\cdot r},
        \qquad \hat k = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta),

    with :math:`r` in model coordinates, so its phase is referred to the
    model origin (`gfem.conventions`, Incident fields). The direction is that
    of propagation: :math:`\theta < \pi/2` travels up (from the bottom of the
    stack), :math:`\theta > \pi/2` down; a wave arriving 30° off the normal
    from above is ``theta=np.pi - np.radians(30)``.

    Example:
        A wave travelling along +z with E along +x::

            wave = gfem.PlaneWave("x", theta=0.0, polarization="TM")
            problem = gfem.Driven(f, sources=wave, keep_fields=True)

    Args:
        name: The label of its column and of its kept fields; unique among
            the problem's sources and the model's ports.
        theta: The angle of the propagation direction from +z, rad, in
            :math:`[0, \pi]`.
        phi: Its azimuth from +x, rad. Along :math:`\pm z` it fixes the
            plane of incidence: TM is then :math:`\pm(\cos\phi, \sin\phi, 0)`.
        polarization: ``"TM"`` (:math:`\hat\theta`, p), ``"TE"``
            (:math:`\hat\phi`, s), ``"RCP"``
            (:math:`(\hat\theta - i\hat\phi)/\sqrt2`, right-handed in the
            IEEE sense under :math:`e^{+i\omega t}`), ``"LCP"``
            (:math:`(\hat\theta + i\hat\phi)/\sqrt2`), or a Jones pair
            :math:`(a_\mathrm{TM}, a_\mathrm{TE})` of complex numbers, which
            is normalized.
        amplitude: :math:`E_0`, the peak field in V/m in the layer the wave
            comes from; complex for a phase.

    Raises:
        TypeError: ``name`` is not a non-empty string.
        ValueError: ``theta`` is outside :math:`[0, \pi]`, an angle or the
            amplitude is not finite, or ``polarization`` is not one of the
            names or a finite nonzero pair.
        NonDifferentiableError: A parameter is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`; source parameters are fixed values.
    """

    name: str
    theta: float
    phi: float
    polarization: Polarization | tuple[complex, complex]
    amplitude: complex

    def __init__(
        self,
        name: str,
        *,
        theta: float,
        phi: float = 0.0,
        polarization: Polarization | tuple[complex, complex],
        amplitude: complex = 1.0,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        _set(
            self,
            name=_name(name, "PlaneWave"),
            theta=_angle(theta, "PlaneWave.theta", polar=True),
            phi=_angle(phi, "PlaneWave.phi", polar=False),
            polarization=_polarization(polarization, "PlaneWave"),
            amplitude=_complex(amplitude, "PlaneWave.amplitude"),
        )

    def __repr__(self) -> str:
        amplitude = "" if self.amplitude == 1 else f", amplitude={self.amplitude!r}"
        return (
            f"PlaneWave({self.name!r}, theta={self.theta!r}, phi={self.phi!r}, "
            f"polarization={self.polarization!r}{amplitude})"
        )


@dataclasses.dataclass(frozen=True, init=False)
class GaussianBeam:
    r"""A focused Gaussian beam carrying 1 W, of given waist, focus and direction.

    The beam is an exact solution of Maxwell's equations in a homogeneous
    lossless background, whose paraxial limit is the fundamental Gaussian
    beam of waist :math:`w_0`. Like an excited port it is fed 1 W: 1 W
    time-averaged power through any plane normal to its axis, with the
    focal field real and positive along the polarization (zero phase at the
    focus; `gfem.conventions`, Incident fields). Every output is linear in
    that power, so there is no amplitude; `focal_field` and
    `focal_intensity` give the field and intensity at the focus, which
    `gfem.cross_sections` divides by.

    In the beam's frame (focus at the origin, axis :math:`\hat z`, TM along
    :math:`\hat x`) it is the field of an electric :math:`\hat x` and a
    magnetic :math:`\hat y` dipole at the complex point :math:`-ib\hat z`,
    :math:`b = k w_0^2/2`: a superposition of propagating plane waves of
    weight :math:`(1 + \cos\theta)\,e^{a(\cos\theta - 1)}`,
    :math:`a = kb`, regular everywhere, so the focus may lie anywhere,
    inside a scatterer too. TE is the TM beam turned by 90° about the axis.

    The background must be one lossless material, and :math:`k w_0` at
    least 0.2 at every solved frequency, where the 1 W normalization is
    exact to round-off; `gfem.solve` refuses anything else.

    Example:
        A beam of waist 3 µm coming down 10° off the normal, focused on
        the surface::

            beam = gfem.GaussianBeam(
                "fiber",
                waist=3e-6,
                theta=np.pi - np.radians(10),
                polarization="TE",
            )

    Args:
        name: The label of its column and of its kept fields; unique among
            the problem's sources and the model's ports.
        waist: :math:`w_0`, m, the same at every frequency.
        focus: The focal point, m, model coordinates.
        theta: The angle of the beam axis (the propagation direction) from
            +z, rad, in :math:`[0, \pi]`, as for `PlaneWave`.
        phi: Its azimuth from +x, rad.
        polarization: As for `PlaneWave`: ``"TE"``, ``"TM"``, ``"RCP"``,
            ``"LCP"`` or a Jones pair :math:`(a_\mathrm{TM}, a_\mathrm{TE})`.

    Raises:
        TypeError: ``name`` is not a non-empty string.
        ValueError: ``waist`` is not finite and positive, ``focus`` is not 3
            finite coordinates, ``theta`` is outside :math:`[0, \pi]`, or
            ``polarization`` is not one of the names or a finite nonzero
            pair.
        NonDifferentiableError: A parameter is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`; source parameters are fixed values.
    """

    name: str
    waist: float
    focus: tuple[float, float, float]
    theta: float
    phi: float
    polarization: Polarization | tuple[complex, complex]

    def __init__(
        self,
        name: str,
        *,
        waist: float,
        focus: Sequence[float] = (0.0, 0.0, 0.0),
        theta: float,
        phi: float = 0.0,
        polarization: Polarization | tuple[complex, complex],
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        w0 = static_float(waist, "GaussianBeam.waist", _FIXED)
        if w0 <= 0:
            raise ValueError(f"GaussianBeam.waist must be > 0 m, got {w0}.")
        _set(
            self,
            name=_name(name, "GaussianBeam"),
            waist=w0,
            focus=_point(focus, "GaussianBeam.focus"),
            theta=_angle(theta, "GaussianBeam.theta", polar=True),
            phi=_angle(phi, "GaussianBeam.phi", polar=False),
            polarization=_polarization(polarization, "GaussianBeam"),
        )

    def focal_field(self, frequency: float, *, model: Any) -> float:
        r"""The peak field at the focus, V/m, real and positive.

        It depends on the background's wavenumber :math:`k`, so on the
        frequency and the model's background: about
        :math:`\sqrt{4\eta/(\pi w_0^2)}` for 1 W in the paraxial limit,
        less for a tight waist.

        Args:
            frequency: Hz.
            model: The model the beam lights, whose background
                (``Model(background=...)``) it travels in.

        Raises:
            TypeError: ``model`` is not a `gfem.Model`.
            ValueError: The model has no background, or :math:`k w_0 < 0.2`,
                too tight to normalize.
            NotImplementedError: A layered or lossy background.
        """
        from gfem import _beam

        return _beam.focal_field(self, _background(model, self.name), frequency)

    def focal_intensity(self, frequency: float, *, model: Any) -> float:
        r"""The focal intensity :math:`I_0 = |E(r_f)|^2/(2\eta)`, W/m².

        The intensity `gfem.cross_sections` divides a beam's powers by;
        :math:`\eta` is the background's wave impedance. Arguments and
        errors as for `focal_field`.
        """
        from gfem import _beam

        background = _background(model, self.name)
        peak = _beam.focal_field(self, background, frequency)
        return peak**2 / (2 * _beam.impedance(self, background))

    def __repr__(self) -> str:
        return (
            f"GaussianBeam({self.name!r}, waist={self.waist!r}, "
            f"focus={self.focus!r}, theta={self.theta!r}, phi={self.phi!r}, "
            f"polarization={self.polarization!r})"
        )


def _background(model: Any, name: str) -> Any:
    """The background of ``model``, which a beam's focal values read."""
    from gfem.model import Model

    if not isinstance(model, Model):
        raise TypeError(
            f"The GaussianBeam {name!r} reads its medium from model= a "
            f"gfem.Model, got {type(model).__name__}."
        )
    if model.background is None:
        raise ValueError(
            f"The GaussianBeam {name!r} travels in the model's background: "
            "give it one, Model(background=...)."
        )
    return model.background


def qualified(function: Callable) -> str:
    module = getattr(function, "__module__", None)
    name = getattr(function, "__qualname__", None) or type(function).__qualname__
    return f"{module}.{name}" if module else name


@dataclasses.dataclass(frozen=True, init=False)
class IncidentField:
    r"""An incident field given by a function of position and frequency.

    ``function(points, frequency)`` takes ``points``, an ``(n, 3)`` real
    array of positions in m (model coordinates), and ``frequency`` in Hz,
    and returns the ``(n, 3)`` complex incident electric field there, peak
    V/m under :math:`e^{+i\omega t}`. It must be a solution of Maxwell's
    equations in the background stack (``Model(background=...)``) at that
    frequency: gfem does not check it. It must be a pure function of its
    arguments, NumPy in and out; gfem does not differentiate through it.

    A result holding one cannot be written to a file (`gfem.Job.to_json`,
    result bundles), and a solve cache passes it by with a
    `gfem.SolveCacheWarning`: the function is code, not data. Across MPI
    ranks each rank calls it on its own points, so it must be the same
    function on every rank. Over frequency workers it must load in a
    spawned process: a function at module level of a script or an
    importable module, not a lambda or one defined in a notebook
    (`ValueError` before any worker starts).

    Args:
        name: The label of its column and of its kept fields; unique among
            the problem's sources and the model's ports.
        function: ``(points, frequency) -> E``.

    Raises:
        TypeError: ``name`` is not a non-empty string, or ``function`` is not
            callable.
    """

    name: str
    function: Callable[[np.ndarray, float], np.ndarray]

    def __init__(
        self, name: str, function: Callable[[np.ndarray, float], np.ndarray]
    ) -> None:
        """Validate (see the class docstring)."""
        if not callable(function):
            raise TypeError(
                "IncidentField.function must be callable as function(points, "
                f"frequency) -> E, got {type(function).__name__}."
            )
        _set(self, name=_name(name, "IncidentField"), function=function)

    def __repr__(self) -> str:
        return f"IncidentField({self.name!r}, function={qualified(self.function)})"


# --- point dipoles -----------------------------------------------------------------


@dataclasses.dataclass(frozen=True, init=False, repr=False)
class _PointDipole:
    """A point source's name, position and moment (see the two dipoles)."""

    name: str
    position: tuple[float, float, float]
    moment: tuple[complex, complex, complex]

    def __init__(
        self, name: str, *, position: Sequence[float], moment: Sequence[complex]
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        owner = type(self).__name__
        _set(
            self,
            name=_name(name, owner),
            position=_point(position, f"{owner}.position"),
            moment=_moment(moment, f"{owner}.moment"),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.name!r}, position={self.position!r}, "
            f"moment={self.moment!r})"
        )


@dataclasses.dataclass(frozen=True, init=False, repr=False)
class ElectricDipole(_PointDipole):
    r"""A point electric dipole of moment :math:`p` (C·m) at a position (m).

    Its current is :math:`J = i\omega p\,\delta(r - r_0)` under
    :math:`e^{+i\omega t}`, with :math:`p` the same at every frequency; an
    antenna current moment :math:`Il` (A·m) is :math:`p = Il/(i\omega)`
    (`gfem.conventions`, Point dipoles).

    Args:
        name: The label of its column and of its kept fields; unique among
            the problem's sources and the model's ports.
        position: :math:`r_0`, m, model coordinates.
        moment: :math:`p`, 3 complex components, C·m: orientation and
            magnitude.

    Raises:
        TypeError: ``name`` is not a non-empty string.
        ValueError: ``position`` or ``moment`` is not 3 finite values.
        NonDifferentiableError: A parameter is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`; source parameters are fixed values.
    """


@dataclasses.dataclass(frozen=True, init=False, repr=False)
class MagneticDipole(_PointDipole):
    r"""A point magnetic dipole of moment :math:`m` (A·m²) at a position (m).

    Its current is :math:`J = \nabla\times(m\,\delta(r - r_0))`, the
    magnetic current :math:`M = i\omega\mu_0 m\,\delta(r - r_0)` under
    :math:`e^{+i\omega t}`, with :math:`m` the same at every frequency
    (`gfem.conventions`, Point dipoles).

    Args:
        name: The label of its column and of its kept fields; unique among
            the problem's sources and the model's ports.
        position: :math:`r_0`, m, model coordinates.
        moment: :math:`m`, 3 complex components, A·m²: orientation and
            magnitude.

    Raises:
        TypeError: ``name`` is not a non-empty string.
        ValueError: ``position`` or ``moment`` is not 3 finite values.
        NonDifferentiableError: A parameter is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap`; source parameters are fixed values.
    """


# --- polarization sources ----------------------------------------------------------

_PROCESSES = ("second-harmonic", "sum-frequency", "difference-frequency")


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class PolarizationSource:
    r"""A polarization :math:`P` at the points of a quadrature rule, as a load column.

    The current is :math:`J = +i\omega P` under :math:`e^{+i\omega t}` and the
    column is its load, :math:`b_j = \mu_0 L\omega^2 \int P\cdot\varphi_j\,
    \mathrm{d}V` (the dipole's, integrated). It is built by
    `gfem.second_order_polarization` from the fields of a pump, not by hand: the
    values live at the points of the rule of the mesh it was built on, in
    the order of the elements ``elements`` (volume elements of the mesh, in
    increasing order) and, within an element, of the rule. Solve it with
    ``gfem.Driven(source.frequency, excite=(), sources=source)`` on a model
    of the same mesh.

    Attributes:
        name: The label of its column and of its kept fields.
        frequency: The generated frequency, Hz.
        process: ``"second-harmonic"``, ``"sum-frequency"`` or
            ``"difference-frequency"``.
        pump_frequencies: The pumps' frequencies, Hz.
        regions: The regions carrying a second-order tensor (a zero tensor
            counts: the gradient of P reaches it).
        order: The order :math:`Q` of the rule space of its points (rule
            degree :math:`2Q`).
        generated_order: The highest Nédélec order of a generated solve that
            integrates the load exactly.
        elements: ``(n_elements,)`` the mesh's volume elements holding points.
        mesh_elements: The number of volume elements of the mesh.
        points: ``(n_points, 3)`` the points' coordinates in the mesh's file
            unit, which a solve compares with its own rule's.
        re: ``(n_points, 3)`` the real part of :math:`P` (C/m²).
        im: ``(n_points, 3)`` the imaginary part of :math:`P` (C/m²).
        wave_vector: The Bloch wave vector (rad/m) its pumps imply,
            :math:`2\mathbf k` for the second harmonic, :math:`\mathbf k_1 +
            \mathbf k_2` for the sum and :math:`\mathbf k_1 - \mathbf k_2`
            for the difference: the vector to give
            ``gfem.Driven(wave_vector=...)`` on a periodic cell. ``None``
            for pumps of plain periodicity or without a periodic pair.
        lattice: The translations (m) of the periodic pairs of the pumps'
            model, which the generated model repeats; ``()`` without pairs.
    """

    name: str
    frequency: float
    process: str
    pump_frequencies: tuple[float, ...]
    regions: tuple[str, ...]
    order: int
    generated_order: int
    elements: np.ndarray
    mesh_elements: int
    points: np.ndarray
    re: Any
    im: Any
    wave_vector: tuple[float, float, float] | None = None
    lattice: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize (see the class docstring)."""
        _name(self.name, "PolarizationSource")
        if self.process not in _PROCESSES:
            raise ValueError(
                f"PolarizationSource.process must be one of {_PROCESSES}, got "
                f"{self.process!r}."
            )
        _set(
            self,
            frequency=float(self.frequency),
            pump_frequencies=tuple(float(f) for f in self.pump_frequencies),
            regions=tuple(self.regions),
            order=int(self.order),
            generated_order=int(self.generated_order),
            elements=np.asarray(self.elements, np.int64),
            mesh_elements=int(self.mesh_elements),
            points=np.asarray(self.points, float),
            wave_vector=(
                None
                if self.wave_vector is None
                else tuple(float(c) for c in self.wave_vector)
            ),
            lattice=tuple(tuple(float(c) for c in a) for a in self.lattice),
        )
        for field in ("re", "im"):
            value = getattr(self, field)
            shape = np.shape(value)
            if len(shape) != 2 or shape[1] != 3 or shape[0] != len(self.points):
                raise ValueError(
                    f"PolarizationSource.{field} must be (n_points, 3) = "
                    f"({len(self.points)}, 3), got {shape}."
                )
            if not is_traced(value):
                _set(self, **{field: np.asarray(value, float)})

    def __repr__(self) -> str:
        return (
            f"PolarizationSource({self.name!r}, frequency={self.frequency:.4g}, "
            f"process={self.process!r}, regions={self.regions!r}, "
            f"points={len(self.points)})"
        )


register_fields(PolarizationSource, ("re", "im"))

#: A source of a driven problem: an incident field, a point dipole or a polarization.
Source = (
    PlaneWave
    | GaussianBeam
    | IncidentField
    | ElectricDipole
    | MagneticDipole
    | PolarizationSource
)
#: The sources whose column is an incident field in the background stack.
INCIDENT = (PlaneWave, GaussianBeam, IncidentField)
#: The point sources.
DIPOLES = (ElectricDipole, MagneticDipole)
#: The sources whose column is a load spread over elements.
POLARIZATION = (PolarizationSource,)

for _cls in (PlaneWave, GaussianBeam, IncidentField, ElectricDipole, MagneticDipole):
    jax.tree_util.register_static(_cls)
