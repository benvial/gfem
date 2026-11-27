r"""Multipole decomposition of a scattered field: result types and the projection.

The scattered field of an incident-field column outside a sphere that holds
every source is expanded in outgoing vector spherical waves,

.. math::

    E_s = E_\mathrm{inc} \sum_{l\ge1}\sum_{m=-l}^{l}
    \big[\alpha_{lm} N^{(2)}_{lm} + \beta_{lm} M^{(2)}_{lm}\big],

with the conventions of `gfem.conventions` (Multipoles). This module holds
what needs no solved field:

- `Multipoles` and `CurrentMoments`, the frozen results and their
  properties (the partial cross sections, the exact moments as fixed linear
  maps of the :math:`l\le2` coefficients);
- `_gauss_grid` and `_project`, the closed-form projection of far-field
  amplitudes sampled on a Gauss-Legendre by trapezoid grid onto the
  coefficients, a constant linear map written in `jax.numpy`;
- `_wiscombe`, the default degree.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import math
import numbers
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import sph_harm_y, spherical_jn

from gfem import _curved as curved
from gfem import _incident
from gfem._background import Background, lossless
from gfem._boundaries import PMC, WavePort, surfaces_of
from gfem._checks import check_fixed, is_traced
from gfem._dim2 import refuse_entry
from gfem._far_field import _measure
from gfem._field_kernels import engine
from gfem._power_flow import concrete_field
from gfem._pytree import readonly, register_fields
from gfem._sources import INCIDENT, PlaneWave
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from gfem._engine.far_field import Layer
    from gfem._engine.power_flow import Surface
    from gfem._field import Field
    from gfem.model import Model
    from gfem.results import DrivenResult

__all__ = ["CurrentMoments", "Multipoles", "current_moments", "multipoles"]


class _Grid(NamedTuple):
    """Directions and quadrature weights, each ``(n_theta, 2 * n_theta)``."""

    theta: np.ndarray
    phi: np.ndarray
    weight: np.ndarray


def _gauss_grid(n_theta: int) -> _Grid:
    r"""Gauss-Legendre in :math:`\cos\theta` by trapezoid in :math:`\phi`.

    ``n_theta`` polar nodes (never on a pole) and ``2 * n_theta`` azimuths
    from 0; the weights sum to :math:`4\pi`. The rule is exact for every
    spherical harmonic product of total degree up to ``2 * n_theta - 1``.
    """
    if n_theta < 1:
        raise ValueError(f"n_theta must be at least 1, got {n_theta}.")
    x, w = np.polynomial.legendre.leggauss(n_theta)
    n_phi = 2 * n_theta
    phi = np.arange(n_phi) * (2 * math.pi / n_phi)
    theta_g, phi_g = np.meshgrid(np.arccos(x), phi, indexing="ij")
    weight = np.outer(w, np.full(n_phi, 2 * math.pi / n_phi))
    return _Grid(theta_g, phi_g, weight)


def _wiscombe(x: float) -> int:
    r"""Wiscombe's degree count :math:`\max(2, \lceil x + 4x^{1/3} + 2\rceil)`."""
    return max(2, math.ceil(x + 4 * x ** (1 / 3) + 2))


def _flat_lm(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    ls = [l for l in range(1, lmax + 1) for _ in range(-l, l + 1)]
    ms = [m for l in range(1, lmax + 1) for m in range(-l, l + 1)]
    return np.array(ls, dtype=int), np.array(ms, dtype=int)


def _y(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Condon-Shortley :math:`Y_{lm}`, zero for :math:`|m| > l`."""
    if abs(m) > l:
        return np.zeros(np.broadcast(theta, phi).shape, complex)
    return sph_harm_y(l, m, theta, phi)


def _x_lm(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Jackson's :math:`X_{lm}` as its :math:`(\hat\theta, \hat\phi)` components."""
    s = math.sqrt(l * (l + 1))
    y = _y(l, m, theta, phi)
    dy = m / np.tan(theta) * y + math.sqrt((l - m) * (l + m + 1)) * np.exp(
        -1j * phi
    ) * _y(l, m + 1, theta, phi)
    return -m * y / (np.sin(theta) * s), -1j * dy / s


@functools.lru_cache(maxsize=4)
def _basis(n_theta: int, lmax: int) -> tuple[np.ndarray, ...]:
    r"""The projection matrices for ``n_theta`` and ``lmax``.

    With :math:`F` the far-field amplitude, the absolute coefficients are
    :math:`c^E_{lm} = k\,i^{-l}\oint\overline{\hat r\times X_{lm}}\cdot F\,d\Omega`
    and :math:`c^M_{lm} = k\,i^{-(l+1)}\oint\overline{X_{lm}}\cdot F\,d\Omega`;
    the four matrices hold everything but :math:`k`, ``(n_lm, n_theta * n_phi)``
    each: electric on :math:`F_\theta`, electric on :math:`F_\phi`, magnetic
    on :math:`F_\theta`, magnetic on :math:`F_\phi`.
    """
    grid = _gauss_grid(n_theta)
    th, ph, w = (a.ravel() for a in grid)
    ls, ms = _flat_lm(lmax)
    out = [np.empty((len(ls), th.size), complex) for _ in range(4)]
    for j, (l, m) in enumerate(zip(ls, ms, strict=True)):
        xt, xp = _x_lm(int(l), int(m), th, ph)
        # rhat x X has components (-X_phi, X_theta).
        ce, cm = (1j) ** (-int(l)), (1j) ** (-int(l) - 1)
        out[0][j] = w * ce * np.conj(-xp)
        out[1][j] = w * ce * np.conj(xt)
        out[2][j] = w * cm * np.conj(xt)
        out[3][j] = w * cm * np.conj(xp)
    for a in out:
        a.flags.writeable = False
    return tuple(out)


def _project(
    e_theta: Any,
    e_phi: Any,
    grid: _Grid,
    *,
    k: Any,
    lmax: int,
) -> tuple[jax.Array, jax.Array]:
    r"""Absolute spherical coefficients of far-field amplitudes on a Gauss grid.

    Args:
        e_theta: :math:`F_\theta` (V) on ``grid``, ``(n_freq, *grid.theta.shape)``.
        e_phi: :math:`F_\phi` (V), same shape.
        grid: The directions and weights of `_gauss_grid`.
        k: The background wavenumber per frequency, rad/m, ``(n_freq,)``.
        lmax: The highest degree.

    Returns:
        ``(c_alpha, c_beta)``, :math:`E_\mathrm{inc}\alpha_{lm}` and
        :math:`E_\mathrm{inc}\beta_{lm}` in V/m, each ``(n_freq, lmax * (lmax + 2))``
        in the flat order :math:`j = l^2 + l + m - 1`. Linear in the
        amplitudes, so a gradient flows through it.
    """
    if grid.theta.ndim != 2 or grid.theta.shape[1] != 2 * grid.theta.shape[0]:
        raise ValueError("grid must come from _gauss_grid.")
    ft = jnp.reshape(jnp.asarray(e_theta, complex), (jnp.shape(e_theta)[0], -1))
    fp = jnp.reshape(jnp.asarray(e_phi, complex), (jnp.shape(e_phi)[0], -1))
    if ft.shape[1] != grid.theta.size or fp.shape != ft.shape:
        raise ValueError(
            f"The amplitudes must have {grid.theta.size} directions per "
            f"frequency (the grid's), got {ft.shape[1]} and {fp.shape[1]}."
        )
    ket, kep, kmt, kmp = (jnp.asarray(a) for a in _basis(grid.theta.shape[0], lmax))
    kk = jnp.reshape(jnp.asarray(k, float), (-1, 1))
    c_alpha = kk * (ft @ ket.T + fp @ kep.T)
    c_beta = kk * (ft @ kmt.T + fp @ kmp.T)
    return c_alpha, c_beta


def _spherical_unit_vectors() -> np.ndarray:
    r""":math:`\hat e_{-1}, \hat e_0, \hat e_{+1}` as rows."""
    return np.array(
        [
            [1, -1j, 0],
            [0, 0, 1.0],
            [-1, -1j, 0],
        ]
    ) / np.array([[math.sqrt(2)], [1.0], [math.sqrt(2)]])


def _clebsch_gordan_11(m1: int, m2: int) -> float:
    r""":math:`\langle 1\,m_1\,1\,m_2\,|\,2\,(m_1+m_2)\rangle`."""
    if m1 == m2 != 0:
        return 1.0
    if {m1, m2} in ({1, 0}, {-1, 0}):
        return 1 / math.sqrt(2)
    if {m1, m2} == {1, -1}:
        return 1 / math.sqrt(6)
    return math.sqrt(2 / 3)  # m1 == m2 == 0


def _rank2_tensors() -> np.ndarray:
    r""":math:`T_{2M}`, :math:`M = -2..2`, ``(5, 3, 3)``."""
    e = _spherical_unit_vectors()  # index m + 1
    out = np.zeros((5, 3, 3), complex)
    for big_m in range(-2, 3):
        for m1 in (-1, 0, 1):
            m2 = big_m - m1
            if abs(m2) <= 1:
                out[big_m + 2] += _clebsch_gordan_11(m1, m2) * np.outer(
                    e[m1 + 1], e[m2 + 1]
                )
    return out


_E_HAT = _spherical_unit_vectors()
_T2 = _rank2_tensors()
# Flat index j = l^2 + l + m - 1: the l = 1 block is 0..2, the l = 2 block 3..7.
_DEGREE_ONE = slice(0, 3)
_DEGREE_TWO = slice(3, 8)


def _format_origin(origin: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{x:.4g}" for x in origin) + ") m"


@dataclasses.dataclass(frozen=True, eq=False)
class Multipoles:
    r"""The spherical multipoles of a scattered field, per kept frequency.

    Under :math:`e^{+i\omega t}` the scattered field outside a sphere that
    holds every source is

    .. math::

        E_s = E_\mathrm{inc} \sum_{l\ge1}\sum_{m=-l}^{l}
        \big[\alpha_{lm} N^{(2)}_{lm} + \beta_{lm} M^{(2)}_{lm}\big],

    with :math:`M^{(2)}_{lm} = h^{(2)}_l(kr) X_{lm}`, :math:`N^{(2)}_{lm} =
    k^{-1}\nabla\times M^{(2)}_{lm}`, Jackson's :math:`X_{lm}` and
    Condon-Shortley harmonics (`gfem.conventions`, Multipoles), about
    ``origin``. The coefficients are dimensionless and power-normalized: the
    partial scattering cross section of an electric (magnetic) multipole of
    degree :math:`l` is :math:`k^{-2}\sum_m|\alpha_{lm}|^2`
    (:math:`k^{-2}\sum_m|\beta_{lm}|^2`). Every array has the kept
    frequencies first, and the coefficients are stored :math:`l`-major,
    :math:`m` ascending: flat index :math:`j = l^2 + l + m - 1`.

    Attributes:
        frequency: The kept frequencies of the column, Hz.
        origin: The multipole origin, m, in model coordinates.
        lmax: The highest degree, the same at every frequency.
        wavenumber: The background :math:`k`, rad/m, one per frequency.
        impedance: The background :math:`\eta`, ohm.
        incident_amplitude: :math:`E_\mathrm{inc}`, V/m, ``(n_freq,)``,
            complex: a plane wave's amplitude, a beam's focal field
            :math:`\sqrt{2\eta I_0}`, :math:`\sqrt{2\eta I}` of an
            `gfem.IncidentField`. Not `gfem.FarField.incident_amplitude`,
            which is the real :math:`|E_0|` of a plane wave and ``None`` of a
            beam.
        alpha: :math:`\alpha_{lm}` on :math:`N^{(2)}_{lm}`, the electric
            (transverse magnetic) coefficients, ``(n_freq, lmax * (lmax + 2))``.
        beta: :math:`\beta_{lm}` on :math:`M^{(2)}_{lm}`, the magnetic
            (transverse electric) coefficients, same shape.
    """

    frequency: tuple[float, ...]
    origin: tuple[float, float, float]
    lmax: int
    wavenumber: tuple[float, ...]
    impedance: float
    incident_amplitude: jax.Array
    alpha: jax.Array
    beta: jax.Array

    def __post_init__(self) -> None:
        if not isinstance(self.lmax, int) or self.lmax < 2:
            raise ValueError(
                f"lmax must be an integer of at least 2, got {self.lmax!r}."
            )
        n_freq, n_lm = len(self.frequency), self.lmax * (self.lmax + 2)
        if len(self.wavenumber) != n_freq or len(self.origin) != 3:
            raise ValueError(
                "frequency and wavenumber must have one entry per kept "
                "frequency and origin three."
            )
        for name in ("incident_amplitude", "alpha", "beta"):
            want = (n_freq,) if name == "incident_amplitude" else (n_freq, n_lm)
            if tuple(jnp.shape(getattr(self, name))) != want:
                raise ValueError(
                    f"{name} must have shape {want}, got "
                    f"{tuple(jnp.shape(getattr(self, name)))}."
                )

    # -- indices -----------------------------------------------------------
    @property
    def l(self) -> np.ndarray:  # noqa: E743
        """The degree of each flat index, a read-only integer array."""
        return _flat_index_arrays(self.lmax)[0]

    @property
    def m(self) -> np.ndarray:
        """The azimuthal index of each flat index, a read-only integer array."""
        return _flat_index_arrays(self.lmax)[1]

    def index(self, l: int, m: int) -> int:
        r"""The flat index :math:`j = l^2 + l + m - 1` of :math:`(l, m)`.

        ``mp.alpha[:, mp.index(2, -1)]`` is :math:`\alpha_{2,-1}`;
        ``mp.alpha[:, mp.l == 2]`` is the whole degree.

        Raises:
            TypeError: ``l`` or ``m`` is not an integer (``bool`` included).
            ValueError: :math:`(l, m)` is not in the expansion.
        """
        for name, v in (("l", l), ("m", m)):
            if isinstance(v, bool) or not isinstance(v, numbers.Integral):
                raise TypeError(
                    f"Multipoles.index: {name} must be an int, got {type(v).__name__}."
                )
        if not (1 <= l <= self.lmax and -l <= m <= l):
            raise ValueError(
                f"Multipoles.index: (l, m) = ({l}, {m}) is not a multipole of "
                f"this result: 1 <= l <= lmax = {self.lmax} and -l <= m <= l."
            )
        return int(l * l + l + m - 1)

    # -- derived outputs ---------------------------------------------------
    def _k(self) -> np.ndarray:
        return np.asarray(self.wavenumber, float)

    def _per_frequency(self, x: Any, ndim: int) -> Any:
        return jnp.reshape(x, (-1,) + (1,) * (ndim - 1))

    @property
    def intensity(self) -> jax.Array:
        r"""The incident intensity :math:`|E_\mathrm{inc}|^2/(2\eta)`, W/m²."""
        e = self.incident_amplitude
        return (e.real**2 + e.imag**2) / (2 * self.impedance)

    def _partial(self, c: jax.Array) -> jax.Array:
        ls = _flat_index_arrays(self.lmax)[0]
        onehot = jnp.asarray(ls[:, None] == np.arange(1, self.lmax + 1)[None])
        k2 = jnp.asarray(self._k() ** 2)[:, None]
        return ((c.real**2 + c.imag**2) @ onehot.astype(float)) / k2

    @property
    def electric_scattering(self) -> jax.Array:
        r""":math:`\sigma^E_l = k^{-2}\sum_m|\alpha_{lm}|^2`, m², ``(n_freq, lmax)``.

        Column :math:`l - 1`: dipole, quadrupole, octupole for :math:`l = 1,
        2, 3`. Equal to the cross sections of the exact :math:`p` and
        :math:`Q^e` for :math:`l = 1, 2`.
        """
        return self._partial(self.alpha)

    @property
    def magnetic_scattering(self) -> jax.Array:
        r""":math:`\sigma^M_l = k^{-2}\sum_m|\beta_{lm}|^2`, m², ``(n_freq, lmax)``."""
        return self._partial(self.beta)

    @property
    def scattering(self) -> jax.Array:
        r"""The sum of the partial scattering cross sections, m², ``(n_freq,)``.

        The far-field power over :math:`I` up to ``lmax``. It is a different
        discretization from `gfem.cross_sections`' scattering cross
        section and differs from it by a fraction of a percent at order 3.
        """
        return jnp.sum(self.electric_scattering, axis=1) + jnp.sum(
            self.magnetic_scattering, axis=1
        )

    def _epsilon(self) -> np.ndarray:
        omega = 2 * math.pi * np.asarray(self.frequency, float)
        return self._k() / (omega * self.impedance)

    @property
    def electric_dipole(self) -> jax.Array:
        r"""The exact electric dipole :math:`p`, C·m, ``(n_freq, 3)``.

        :math:`p = -\sqrt{6\pi}\,\varepsilon E_\mathrm{inc} k^{-3}
        \sum_m\alpha_{1m}\hat e_m`, in the model's :math:`x, y, z`.
        """
        s = jnp.asarray(_E_HAT)
        scale = -math.sqrt(6 * math.pi) * jnp.asarray(self._epsilon() / self._k() ** 3)
        return (
            scale[:, None]
            * self.incident_amplitude[:, None]
            * (self.alpha[:, _DEGREE_ONE] @ s)
        )

    @property
    def magnetic_dipole(self) -> jax.Array:
        r"""The exact magnetic dipole :math:`m`, A·m², ``(n_freq, 3)``.

        :math:`m = -i\sqrt{6\pi}\,E_\mathrm{inc}(k^{3}\eta)^{-1}
        \sum_m\beta_{1m}\hat e_m`.
        """
        s = jnp.asarray(_E_HAT)
        scale = (
            -1j
            * math.sqrt(6 * math.pi)
            / (jnp.asarray(self._k() ** 3) * self.impedance)
        )
        return (
            scale[:, None]
            * self.incident_amplitude[:, None]
            * (self.beta[:, _DEGREE_ONE] @ s)
        )

    @property
    def electric_quadrupole(self) -> jax.Array:
        r"""The exact electric quadrupole :math:`Q^e`, C·m², ``(n_freq, 3, 3)``.

        Symmetric and traceless, with Jackson's normalization
        :math:`\int\rho(3r_\alpha r_\beta - r^2\delta_{\alpha\beta})\,dV` in
        the long-wavelength limit.
        """
        t = jnp.asarray(_T2)
        scale = (
            -12 * math.sqrt(5 * math.pi) * jnp.asarray(self._epsilon() / self._k() ** 4)
        )
        return (
            scale[:, None, None]
            * self.incident_amplitude[:, None, None]
            * (jnp.einsum("fM,Mab->fab", self.alpha[:, _DEGREE_TWO], t))
        )

    @property
    def magnetic_quadrupole(self) -> jax.Array:
        r"""The exact magnetic quadrupole :math:`Q^m`, A·m³, ``(n_freq, 3, 3)``.

        Symmetric and traceless; long-wavelength limit
        :math:`\int[r_\alpha(r\times J)_\beta + r_\beta(r\times J)_\alpha]\,dV`,
        three times the magnetic quadrupole of Evlyukhin and the anapole
        review.
        """
        t = jnp.asarray(_T2)
        scale = (
            -12j
            * math.sqrt(5 * math.pi)
            / (jnp.asarray(self._k() ** 4) * self.impedance)
        )
        return (
            scale[:, None, None]
            * self.incident_amplitude[:, None, None]
            * (jnp.einsum("fM,Mab->fab", self.beta[:, _DEGREE_TWO], t))
        )

    def __repr__(self) -> str:
        try:
            parts = [
                np.asarray(x)
                for x in (
                    self.electric_scattering,
                    self.magnetic_scattering,
                )
            ]
        except (TypeError, jax.errors.TracerArrayConversionError):
            return "Multipoles(traced)"
        se, sm = parts
        rows = "; ".join(
            f"{f:.6g} Hz: ED {e[0]:.4g}, MD {m[0]:.4g}, EQ {e[1]:.4g}, MQ {m[1]:.4g} m²"
            for f, e, m in zip(self.frequency, se, sm, strict=True)
        )
        return (
            f"Multipoles(lmax={self.lmax}, origin={_format_origin(self.origin)}; "
            f"{rows})"
        )


@functools.lru_cache(maxsize=16)
def _flat_index_arrays(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    ls, ms = _flat_lm(lmax)
    return readonly(ls), readonly(ms)


register_fields(Multipoles, ("incident_amplitude", "alpha", "beta"))


@dataclasses.dataclass(frozen=True, eq=False)
class CurrentMoments:
    r"""The long-wavelength electric dipole and the toroidal dipole of a scatterer.

    Integrals of the induced current :math:`J = +i\omega\varepsilon_0
    (\varepsilon_r - \varepsilon_b)E` over the scatterer, with :math:`r`
    measured from ``origin``. They approximate the exact dipole of
    `gfem.Multipoles` as :math:`p \approx p_0 - ikT`; they are not
    orthogonal multipoles, so no partial cross section is defined for them.

    Attributes:
        frequency: The kept frequencies of the column, Hz.
        origin: The origin of :math:`r`, m, in model coordinates.
        wavenumber: The background :math:`k`, rad/m, one per frequency.
        long_wavelength_dipole: :math:`p_0 = (i\omega)^{-1}\int J\,dV`, C·m,
            ``(n_freq, 3)``.
        toroidal_dipole: :math:`T = (10v)^{-1}\int[(r\cdot J)r - 2r^2J]\,dV`,
            C·m², ``(n_freq, 3)``, with :math:`v = \omega/k`.
    """

    frequency: tuple[float, ...]
    origin: tuple[float, float, float]
    wavenumber: tuple[float, ...]
    long_wavelength_dipole: jax.Array
    toroidal_dipole: jax.Array

    def __repr__(self) -> str:
        try:
            p0 = np.linalg.norm(np.asarray(self.long_wavelength_dipole), axis=1)
            kt = np.linalg.norm(np.asarray(self.toroidal_dipole), axis=1) * np.asarray(
                self.wavenumber
            )
        except (TypeError, jax.errors.TracerArrayConversionError):
            return "CurrentMoments(traced)"
        rows = "; ".join(
            f"{f:.6g} Hz: |p0| {a:.4g}, |kT| {b:.4g} C·m"
            for f, a, b in zip(self.frequency, p0, kt, strict=True)
        )
        return f"CurrentMoments(origin={_format_origin(self.origin)}; {rows})"


register_fields(CurrentMoments, ("long_wavelength_dipole", "toroidal_dipole"))


# --- the entry ----------------------------------------------------------------------


@functools.cache
def _scatterer_surfaces() -> tuple[type, ...]:
    """The surfaces that belong to the scatterer when they lie inside the mesh.

    The ones a cross-section measurement counts, and a PMC face too, which a
    multipole expansion cannot take as a wall but reads as a magnetic sheet.
    Read on first use: the cross-section engine imports NGSolve, which
    importing gfem must not.
    """
    xs = importlib.import_module("gfem._engine.cross_sections")
    return (*xs._SCATTERER_SURFACES, PMC)  # noqa: SLF001


def _background(model: Model, entry: str = "multipoles") -> Background:
    """The model's background.

    Raises:
        ValueError: The model has none.
    """
    if model.background is None:
        raise ValueError(
            f"{entry} needs the model's background (Model(background=...)): "
            "pass the model the result was solved with."
        )
    return model.background


def _scatterer_radius(model: Model, origin: np.ndarray) -> float:
    """The scatterer's circumradius about ``origin``, m.

    The largest distance to a node of the scatterer's elements (the regions
    whose material is not the background's constant one, chosen per region:
    never by comparing the values of a design) and of the sheets, conductors,
    lumped ports and perfect-conductor faces inside the mesh.
    """
    from gfem._engine.incident import scatterer_elements

    mesh = model.mesh
    nodes = np.asarray(mesh.nodes, float)
    used = np.zeros(len(nodes), bool)
    inside = scatterer_elements(model, _background(model))
    used[np.unique(np.asarray(mesh.tetrahedra)[inside])] = True
    neighbours = np.asarray(mesh.triangle_neighbours)
    triangles = np.asarray(mesh.triangles)
    for item in (*model.boundaries, *model.ports):
        if not isinstance(item, _scatterer_surfaces()):
            continue
        for surface in surfaces_of(item):
            rows = mesh.elements(surface)
            # a face with one element beside it is a wall, not a sheet
            rows = rows[(neighbours[rows] >= 0).all(axis=1)]
            used[np.unique(triangles[rows])] = True
    if not used.any():
        return 0.0
    return float(np.linalg.norm(nodes[used] - origin, axis=1).max())


def _default_lmax(model: Model, origin: Any, frequency: float) -> int:
    r"""The default degree: Wiscombe's count of the scatterer about ``origin``.

    :math:`x = kR_s` with :math:`R_s` the scatterer's circumradius
    (`_scatterer_radius`) and :math:`k` the background's wavenumber at
    ``frequency`` (the highest kept one); at least 2.
    """
    k = _incident.wavenumber(_background(model), frequency)
    return _wiscombe(k * _scatterer_radius(model, np.asarray(origin, float)))


def _layer_radius(model: Model, layer: Layer, origin: np.ndarray) -> float:
    """The far-field layer's circumradius about ``origin``, m."""
    nodes = np.asarray(model.mesh.nodes, float)
    used = np.unique(np.asarray(model.mesh.tetrahedra)[layer.tetrahedra])
    return float(np.linalg.norm(nodes[used] - origin, axis=1).max())


def _check_walls(model: Model, entry: str = "multipoles") -> None:
    """Raise if the physical region ends on a PEC or PMC wall of the mesh.

    A ground plane, a mirror half model or a slab model: the multipoles of
    the meshed part are not those of the scatterer.

    Raises:
        NotImplementedError: A face on the mesh's outer boundary, owned by an
            element outside the perfectly matched layer, that is a PEC or a
            PMC wall (a face no boundary names is the natural PMC wall).
    """
    from gfem._engine.incident import Faces, layer_regions

    mesh = model.mesh
    faces = Faces.of(model)
    layer = np.zeros(len(mesh.tetrahedra), bool)
    for region in layer_regions(model):
        layer[mesh.elements(region)] = True
    owner = faces.tets[:, 0]
    rows = faces.hull & (owner >= 0)
    rows[rows] = ~layer[owner[rows]]
    named: dict[str, str] = {}
    for item in (*model.boundaries, *model.ports):
        for name in surfaces_of(item):
            named.setdefault(name, type(item).__name__)
    for row in np.flatnonzero(rows):
        on = [named[s] for s in faces.surfaces[row] if s in named]
        kind = on[0] if on else "PMC"
        if kind in ("PEC", "PMC"):
            face = faces.surfaces[row][0] if faces.surfaces[row] else "(untagged)"
            raise NotImplementedError(
                f"{entry}: the model ends on the {kind} wall {face!r} beside "
                "its physical region (a ground plane, mirror half model or slab "
                "model), which is not computed in this gfem version: the "
                "multipoles of the meshed part are not those of the scatterer. "
                "Mesh the whole scatterer inside a PML or Absorbing box."
            )


def _incident_amplitude(source: Any, i0: np.ndarray, eta: float) -> np.ndarray:
    r"""The incident amplitude :math:`E_\mathrm{inc}` per frequency, V/m."""
    if isinstance(source, PlaneWave):
        return np.full(len(i0), complex(source.amplitude))
    return np.sqrt(2 * eta * i0).astype(complex)


def _expand(
    fields: list[Field],
    model: Model,
    layer: Layer,
    *,
    origin: np.ndarray,
    lmax: int,
    wavenumbers: np.ndarray,
) -> tuple[jax.Array, jax.Array]:
    """Absolute coefficients (V/m) of ``fields``' radiation about ``origin``."""
    pf = importlib.import_module("gfem._engine.power_flow")
    ff = engine("far_field")
    n_theta = math.ceil(wavenumbers.max() * _layer_radius(model, layer, origin))
    grid = _gauss_grid(n_theta + lmax + 4)
    theta, phi = grid.theta.reshape(-1), grid.phi.reshape(-1)
    rhat, that, phat = ff.directions(theta, phi)
    where: Surface = pf.surface_of(concrete_field(fields[0]), layer.inside, model)
    amplitude = jnp.stack(
        [_measure(f, model, layer, where, (rhat, that, phat))[0] for f in fields]
    )
    # the far field is referred to the model origin: refer it to ``origin``
    phase = jnp.exp(-1j * jnp.asarray(wavenumbers)[:, None] * (rhat @ origin)[None])
    amplitude = amplitude * phase[..., None]
    return _project(
        amplitude[..., 0], amplitude[..., 1], grid, k=wavenumbers, lmax=lmax
    )


def _build_layer(fields: list[Field], model: Model, incident: bool) -> Layer:
    """The far-field layer on its default side (the layer touching the truncation)."""
    concrete = concrete_field(fields[0])
    layer = engine("far_field").exterior_layer(
        concrete,
        model,
        None,
        model.background,
        nodes=np.asarray(concrete.mesh.nodes),
        incident=incident,
    )
    curved.refuse_tets(
        model.mesh,
        "multipoles",
        layer.tetrahedra,
        "The far-field layer of gfem.multipoles",
    )
    return layer


def _column_coefficients(
    result: DrivenResult,
    name: str,
    *,
    model: Model,
    origin: Any = (0.0, 0.0, 0.0),
    lmax: int | None = None,
    layer: Layer | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""The absolute spherical coefficients of any kept column, V/m.

    :math:`(E_\mathrm{inc}\alpha, E_\mathrm{inc}\beta)` of the scattered field
    of an incident-field column, or of the total field of a dipole or port
    column, at every kept frequency, ``(n_freq, lmax * (lmax + 2))`` each. No
    column check and no refusal beyond the far-field layer's own.
    """
    origin_a = np.asarray(origin, float)
    frequencies = [f for f, c in result.kept if c == name]
    incident = name in result.sources and isinstance(result.sources[name], INCIDENT)
    part = "scattered" if incident else "total"
    fields = [result.e_field(f, name, part=part) for f in frequencies]
    if layer is None:
        layer = _build_layer(fields, model, incident)
    if lmax is None:
        lmax = _default_lmax(model, origin_a, max(frequencies))
    k = np.array([_incident.wavenumber(_background(model), f) for f in frequencies])
    return _expand(fields, model, layer, origin=origin_a, lmax=lmax, wavenumbers=k)


def _check_column(
    result: DrivenResult, source: str, model: Model, entry: str = "multipoles"
) -> tuple[Any, list[float], Any]:
    """Validate the column: ``(source, kept frequencies, background)``.

    Refuses a non-incident column, unkept fields, a missing, lossy or layered
    background, and a PEC or PMC wall beside the physical region.
    """
    from gfem.measurements import _column_kind

    kind = _column_kind(result, source)
    if kind is not None:
        what = (
            "multipoles are decomposed"
            if entry == "multipoles"
            else "the moments of the induced current are taken"
        )
        raise ValueError(
            f"{entry}: {source!r} is a {kind} column; {what} for an incident "
            "field (gfem.PlaneWave, gfem.GaussianBeam, gfem.IncidentField). "
            "For a dipole's or port's radiation, read gfem.far_field."
        )
    wave = result.sources[source]
    frequencies = [f for f, label in result.kept if label == source]
    if not frequencies:
        raise ValueError(
            f"{entry} needs the kept fields of {source!r}: solve with "
            "Driven(..., keep_fields=True) or keep_fields=[frequencies]."
        )
    background = _background(model, entry)
    for material in background.layers:
        if not lossless(material):
            raise NotImplementedError(
                f"{entry} in a lossy background ({material!r}) is not "
                "computed in this gfem version: a lossy medium carries no "
                "scattered power to infinity. Give a lossless background."
            )
    if _incident.layered(background):
        raise NotImplementedError(
            f"{entry} over a layered background (a substrate or a ground "
            "plane) is not computed in this gfem version: the field radiated "
            "into a stack is not a sum of free-space multipoles. Give a "
            "homogeneous background."
        )
    _check_walls(model, entry)
    return wave, frequencies, background


def _check_types(result: Any, model: Any, entry: str) -> None:
    """Refuse a 2D mesh and a result or model of the wrong type.

    Raises:
        TypeError: ``result`` is not a `gfem.DrivenResult` or ``model`` not a
            `gfem.Model`.
    """
    from gfem.model import Model
    from gfem.results import DrivenResult

    refuse_entry(getattr(model, "mesh", None), entry)
    if not isinstance(result, DrivenResult):
        raise TypeError(
            f"{entry} needs a gfem.DrivenResult, got {type(result).__name__}."
        )
    if not isinstance(model, Model):
        raise TypeError(
            f"{entry} needs model= a gfem.Model, got {type(model).__name__}."
        )


def _truncation_side(
    model: Model, background: Background, field: Field, entry: str
) -> Surface:
    """The truncation side of ``field``, once no scatterer reaches its layer.

    Raises:
        ValueError: A scatterer region or surface reaches the truncation layer.
    """
    from gfem._truncation import truncation

    xs = importlib.import_module("gfem._engine.cross_sections")
    pf = importlib.import_module("gfem._engine.power_flow")
    where = pf.surface_of(concrete_field(field), truncation(model).side, model)
    xs.check_truncation_layer(
        model,
        background,
        where,
        entry=entry,
        outcome="finite multipole expansion",
        remedy="Leave at least one background element between every "
        "scatterer and the PML layer or Absorbing face.",
    )
    return where


def multipoles(
    result: DrivenResult,
    source: str,
    *,
    model: Model,
    origin: Any = (0.0, 0.0, 0.0),
    lmax: int | None = None,
    intensity: Any = None,
) -> Multipoles:
    r"""The spherical multipoles of the field an incident wave scatters.

    The scattered field of an incident-field column, expanded outside the
    sphere holding the scatterer in outgoing vector spherical waves about
    ``origin`` (`Multipoles`; `gfem.conventions`, Multipoles). The
    coefficients are read from the far-field amplitude of the scattered field
    (`gfem.far_field`'s reciprocity integral over the element layer just
    inside the truncation) sampled on a Gauss-Legendre by trapezoid grid of
    directions, then projected in closed form. Any scatterer the solve
    accepts is decomposed: dielectric, magnetic, lossy, perfectly
    conducting, sheets, lumped ports.

    The background must be one lossless medium, the whole scatterer meshed
    inside a perfectly matched layer or `gfem.Absorbing` faces, with at
    least one background element between it and the truncation. A PEC or
    PMC wall beside the physical region (a ground plane, a mirror half model,
    a slab model) is refused: the multipoles of the meshed part are not
    those of the scatterer.

    On a result of a distributed solve it is a collective call: every rank
    calls it with the same arguments and gets the same multipoles.

    Differentiable through the field: inside `jax.grad` the multipoles of a
    material design in the scatterer (a `gfem.MaterialField`) carry its
    gradient; not by the node coordinates or ``origin``. ``intensity``,
    ``lmax``, the source and the frequency are fixed.
    Call it outside `jax.jit`: the far-field layer is built on the host from
    the mesh, so the call cannot be traced as a whole (`jax.grad` is fine).

    Args:
        result: A driven result with kept fields, solved with ``source``.
        source: The name of an incident-field source of the result.
        model: The model the result was solved with.
        origin: The multipole origin, m, in model coordinates: three finite
            numbers.
        lmax: The highest degree, at least 2; ``None`` for Wiscombe's count
            of the scatterer about ``origin`` at the highest kept frequency.
        intensity: W/m², a positive number or one per kept frequency: the
            intensity of a `gfem.IncidentField`; ``None`` for any other
            source.

    Returns:
        A `Multipoles`.

    Raises:
        TypeError: ``result`` or ``model`` is of the wrong type; ``lmax`` is
            not an integer.
        KeyError: ``source`` is not a column of the result.
        ValueError: ``origin`` is not three finite numbers; ``lmax`` is
            below 2 or above what a source inside the far-field layer
            radiates; ``source`` is a port or a dipole (read
            `gfem.far_field`); the result keeps no field of it, or no
            incident part; the model has no background; ``intensity`` is
            missing for a `gfem.IncidentField`, given for another source,
            or not positive and finite; a scatterer reaches the truncation
            layer or a wave port.
        NotImplementedError: A 2D mesh; a lossy or layered background; a PEC
            or PMC wall beside the physical region.
        NonDifferentiableError: ``origin`` or ``intensity`` is traced; the
            node coordinates are traced; a design reaches the truncation
            layer.
    """
    from gfem.measurements import _intensity

    _check_types(result, model, "multipoles")
    check_fixed(origin, "multipoles' origin must be fixed numbers")
    _check_nodes(model.mesh.nodes)
    origin_a = _origin(origin)
    _check_lmax_form(lmax)
    wave, frequencies, background = _check_column(result, source, model)
    i0 = _intensity(wave, intensity, frequencies, background, "multipoles")
    eta = _incident.impedance(background)
    totals = [result.e_field(f, source) for f in frequencies]
    _check_nodes(totals[0].mesh.nodes)
    if any(field._incident is None for field in totals):  # noqa: SLF001
        raise ValueError(
            f"The kept field of {source!r} carries no incident part: solve "
            "again to decompose its scattered field."
        )
    xs = importlib.import_module("gfem._engine.cross_sections")
    for field in totals:
        where = _truncation_side(model, background, field, "multipoles")
        xs.check_side(
            model,
            where,
            "multipoles",
            remedy="Keep the port outside the side, at least one element away.",
        )
    fields = [result.e_field(f, source, part="scattered") for f in frequencies]
    layer = _build_layer(fields, model, True)
    k = np.array([_incident.wavenumber(background, f) for f in frequencies])
    top = float(max(frequencies))
    cap = _wiscombe(
        _incident.wavenumber(background, top) * _layer_radius(model, layer, origin_a)
    )
    if lmax is None:
        lmax = _default_lmax(model, origin_a, top)
    if lmax > cap:
        radius = _layer_radius(model, layer, origin_a)
        raise ValueError(
            f"multipoles: lmax={lmax} is above {cap}, the highest degree a "
            "source inside the far-field layer radiates (Wiscombe's count for "
            f"its circumradius {radius:.3g} m about the origin, kR = "
            f"{_incident.wavenumber(background, top) * radius:.3g}, at "
            f"{top:.6g} Hz): give lmax <= {cap}, or None."
        )
    c_alpha, c_beta = _column_coefficients(
        result, source, model=model, origin=origin_a, lmax=lmax, layer=layer
    )
    e_inc = jnp.asarray(_incident_amplitude(wave, i0, eta))[:, None]
    return Multipoles(
        frequency=tuple(float(f) for f in frequencies),
        origin=(float(origin_a[0]), float(origin_a[1]), float(origin_a[2])),
        lmax=int(lmax),
        wavenumber=tuple(float(x) for x in k),
        impedance=float(eta),
        incident_amplitude=e_inc[:, 0],
        alpha=c_alpha / e_inc,
        beta=c_beta / e_inc,
    )


def _check_induced_current(model: Model, background: Background) -> None:
    """Refuse what the induced current of a volume does not hold.

    Raises:
        NotImplementedError: A perfect conductor or magnetic wall, a
            `gfem.Conductor`, a `gfem.Impedance` or a `gfem.LumpedPort` inside
            the mesh (a surface current), or a scatterer region whose
            permeability differs from the background's (a magnetization
            current).
    """
    from gfem._engine.incident import scatterer_regions

    mesh = model.mesh
    neighbours = np.asarray(mesh.triangle_neighbours)
    for item in (*model.boundaries, *model.ports):
        if not isinstance(item, _scatterer_surfaces()):
            continue
        for surface in surfaces_of(item):
            # a face with one element beside it is a wall, not a sheet
            if (neighbours[mesh.elements(surface)] >= 0).all(axis=1).any():
                raise NotImplementedError(
                    f"current_moments: the {type(item).__name__} surface "
                    f"{surface!r} carries a surface current that is not "
                    "decomposed in this gfem version. Give the metal as a "
                    "volume material (conductivity= or a permittivity law), or "
                    "read its exact moments with gfem.multipoles."
                )
    mu_b = float(_incident.background_material(background).mu_r)
    for region in scatterer_regions(model, background):
        mu = getattr(model.materials[region], "mu_r", None)
        try:
            differs = mu is not None and bool(np.any(np.asarray(mu) != mu_b))
        except (TypeError, ValueError):  # not a number: a law
            differs = True
        if differs:
            raise NotImplementedError(
                f"current_moments: region {region!r} differs from the "
                "background in mu_r: its magnetization current is not "
                "decomposed in this gfem version. Read its exact moments with "
                "gfem.multipoles."
            )


def _check_wave_ports(model: Model, background: Background) -> None:
    """Refuse a scatterer element sharing a node with a wave port's surface.

    Raises:
        ValueError: Naming the region and the port.
    """
    from gfem._engine.incident import region_of, scatterer_elements

    mesh = model.mesh
    ports = [p for p in model.ports if isinstance(p, WavePort)]
    if not ports:
        return
    tets = np.asarray(mesh.tetrahedra)
    inside = np.flatnonzero(scatterer_elements(model, background))
    names = region_of(mesh)
    for port in ports:
        for surface in surfaces_of(port):
            on_port = np.unique(np.asarray(mesh.triangles)[mesh.elements(surface)])
            touching = np.isin(tets[inside], on_port).any(axis=1)
            if touching.any():
                region = names[inside[touching][0]]
                raise ValueError(
                    f"current_moments: the scatterer region {region!r} meets "
                    f"wave port {port.name!r}: it continues beyond the model "
                    "and has no finite multipole expansion."
                )


def current_moments(
    result: DrivenResult,
    source: str,
    *,
    model: Model,
    origin: Any = (0.0, 0.0, 0.0),
) -> CurrentMoments:
    r"""The long-wavelength electric dipole and the toroidal dipole of a scatterer.

    Integrals of the induced current :math:`J = i\omega\varepsilon_0
    (\varepsilon_r - \varepsilon_b)E`, with :math:`E` the total field of an
    incident-field column, over the scatterer's elements outside the
    perfectly matched layer, with :math:`r` measured from ``origin``
    (`CurrentMoments`): :math:`p_0 = (i\omega)^{-1}\int J\,dV` and
    :math:`T = (10v)^{-1}\int[(r\cdot J)r - 2r^2J]\,dV`. They approximate
    the exact dipole as :math:`p \approx p_0 - ikT`, which is how an anapole
    reads (:math:`p \approx 0`); read the exact moments, which carry the
    whole expansion, with `gfem.multipoles`. A scatterer with no region
    differing from the background gives exact zeros.

    The permittivity is read per element at the frequency of the field, with
    its losses and conductivity (the conduction current included), and a
    tensor permittivity is a tensor. The current of a perfect conductor, a
    `gfem.Conductor`, a `gfem.Impedance`, a `gfem.LumpedPort` or a perfect
    magnetic wall is a surface current, and a permeability contrast is a
    magnetization current: neither is decomposed here; `gfem.multipoles`
    reads their exact moments.

    The background must be one lossless medium, the whole scatterer meshed
    inside a perfectly matched layer or `gfem.Absorbing` faces, with at least
    one background element between it and the truncation, and not touching a
    wave port. A PEC or PMC wall beside the physical region is refused.

    On a result of a distributed solve it is a collective call: every rank
    calls it with the same arguments and gets the same moments.

    Not differentiable: the result and the model must be fixed values, not
    computed inside `jax.grad`, `jax.jit` or `jax.vmap`
    (`jax.lax.stop_gradient` inside `jax.grad` is the opt-out). The exact
    moments of `gfem.multipoles` are.

    Args:
        result: A driven result with kept fields, solved with ``source``.
        source: The name of an incident-field source of the result.
        model: The model the result was solved with.
        origin: The origin of :math:`r`, m, in model coordinates: three
            finite numbers.

    Returns:
        A `CurrentMoments`.

    Raises:
        TypeError: ``result`` or ``model`` is of the wrong type.
        KeyError: ``source`` is not a column of the result.
        ValueError: ``origin`` is not three finite numbers; ``source`` is a
            port or a dipole (read `gfem.far_field`); the result keeps no
            field of it; the model has no background; a scatterer reaches the
            truncation layer or a wave port.
        NotImplementedError: A 2D mesh; a lossy or layered background; a PEC
            or PMC wall beside the physical region; a surface current or a
            permeability contrast in the scatterer.
        NonDifferentiableError: ``result``, ``model`` or ``origin`` is traced.
    """
    _check_types(result, model, "current_moments")
    if is_traced(result) or is_traced(model):
        raise NonDifferentiableError(
            "gfem.current_moments is not differentiable in this gfem version: "
            "the moments of the induced current read the scatterer's "
            "permittivity directly, and their result and model must be fixed "
            "values, not computed inside jax.grad, jax.jit or jax.vmap. Read "
            "the exact moments, differentiable, with gfem.multipoles, or call "
            "current_moments outside the function JAX transforms "
            "(jax.lax.stop_gradient on the result and model inside jax.grad)."
        )
    check_fixed(origin, "current_moments' origin must be fixed numbers")
    origin_a = _origin(origin, "current_moments")
    _, frequencies, background = _check_column(result, source, model, "current_moments")
    fields = [result.e_field(f, source) for f in frequencies]
    # before the side is built: a port beside it is refused there with no region
    _check_wave_ports(model, background)
    _truncation_side(model, background, fields[0], "current_moments")
    _check_induced_current(model, background)
    k = np.array([_incident.wavenumber(background, f) for f in frequencies])
    p0, t = _induced_moments(
        fields, model, background, frequencies=frequencies, origin=origin_a, k=k
    )
    return CurrentMoments(
        frequency=tuple(float(f) for f in frequencies),
        origin=(float(origin_a[0]), float(origin_a[1]), float(origin_a[2])),
        wavenumber=tuple(float(x) for x in k),
        long_wavelength_dipole=jnp.asarray(p0),
        toroidal_dipole=jnp.asarray(t),
    )


def _induced_moments(
    fields: list[Field],
    model: Model,
    background: Background,
    *,
    frequencies: list[float],
    origin: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r""":math:`p_0` and :math:`T` per frequency, each ``(n_freq, 3)``."""
    engine_ = importlib.import_module("gfem._engine.multipoles")
    rows = engine_.elements(model, background)
    curved.refuse_tets(
        model.mesh, "current_moments", rows, "The scatterer of gfem.current_moments"
    )
    p0 = np.zeros((len(fields), 3), complex)
    t = np.zeros((len(fields), 3), complex)
    if len(rows) == 0:
        return p0, t
    for i, (field, f) in enumerate(zip(fields, frequencies, strict=True)):
        omega = 2 * math.pi * f
        cur = engine_.current(field, model, background, rows, omega)
        r = cur.points - origin
        j = cur.density
        w = cur.weights[:, None]
        p0[i] = np.sum(w * j, axis=0) / (1j * omega)
        rj = np.sum(r * j, axis=1)[:, None]
        r2 = np.sum(r * r, axis=1)[:, None]
        v = omega / k[i]
        t[i] = np.sum(w * (rj * r - 2 * r2 * j), axis=0) / (10 * v)
    return p0, t


def _riccati_derivative(l: int, x: np.ndarray) -> np.ndarray:
    r""":math:`(x\,j_l(x))'`."""
    return spherical_jn(l, x) + x * spherical_jn(l, x, derivative=True)


def _regular_waves(
    l: int, m: int, k: float, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r""":math:`\mathrm{Rg}M_{lm}` and :math:`\mathrm{Rg}N_{lm}` at ``points``.

    Cartesian components, ``(n, 3)`` each.

    The regular vector spherical waves,
    :math:`\mathrm{Rg}M = j_l(kr)X_{lm}` and
    :math:`\mathrm{Rg}N = k^{-1}\nabla\times\mathrm{Rg}M`, in Cartesian
    components.
    """
    r = np.linalg.norm(points, axis=1)
    theta = np.arccos(np.clip(points[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    kr = k * r
    xt, xp = _x_lm(l, m, theta, phi)
    rxt, rxp = -xp, xt  # rhat x X
    s = math.sqrt(l * (l + 1))
    jl = spherical_jn(l, kr)
    d = _riccati_derivative(l, kr) / kr
    radial = 1j * s * jl * _y(l, m, theta, phi) / kr
    return (
        _spherical_to_cartesian(0 * xt, jl * xt, jl * xp, theta, phi),
        _spherical_to_cartesian(radial, d * rxt, d * rxp, theta, phi),
    )


def _spherical_to_cartesian(
    v_r: np.ndarray,
    v_t: np.ndarray,
    v_p: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    return np.stack(
        [
            v_r * st * cp + v_t * ct * cp - v_p * sp,
            v_r * st * sp + v_t * ct * sp + v_p * cp,
            v_r * ct - v_t * st,
        ],
        axis=-1,
    )


def _volume_coefficients(
    result: DrivenResult,
    name: str,
    *,
    model: Model,
    origin: Any = (0.0, 0.0, 0.0),
    lmax: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""The absolute spherical coefficients from the induced current, V/m.

    The route that needs no far-field layer: with the induced current
    :math:`J` over the scatterer (`current_moments`' integrand),

    .. math::

        c^E_{lm} = -k^2\eta\int \overline{\mathrm{Rg}N_{lm}}\cdot J\,dV,
        \qquad
        c^M_{lm} = -k^2\eta\int \overline{\mathrm{Rg}M_{lm}}\cdot J\,dV,

    the same :math:`E_\mathrm{inc}\alpha_{lm}` and :math:`E_\mathrm{inc}\beta_{lm}`
    as `_column_coefficients`, for a scatterer made of volume materials only
    (no surface currents, no permeability contrast). Test-only: it holds the
    two routes against each other.
    """
    origin_a = np.asarray(origin, float)
    frequencies = [f for f, c in result.kept if c == name]
    background = _background(model)
    fields = [result.e_field(f, name) for f in frequencies]
    if lmax is None:
        lmax = _default_lmax(model, origin_a, max(frequencies))
    engine_ = importlib.import_module("gfem._engine.multipoles")
    rows = engine_.elements(model, background)
    eta = _incident.impedance(background)
    n = lmax * (lmax + 2)
    c_alpha = np.zeros((len(fields), n), complex)
    c_beta = np.zeros((len(fields), n), complex)
    ls, ms = _flat_lm(lmax)
    for i, (field, f) in enumerate(zip(fields, frequencies, strict=True)):
        k = _incident.wavenumber(background, f)
        cur = engine_.current(field, model, background, rows, 2 * math.pi * f)
        r = cur.points - origin_a
        for j, (l, m) in enumerate(zip(ls, ms, strict=True)):
            rg_m, rg_n = _regular_waves(int(l), int(m), k, r)
            for out, rg in ((c_alpha, rg_n), (c_beta, rg_m)):
                out[i, j] = (
                    -(k**2)
                    * eta
                    * np.sum(cur.weights * np.sum(np.conj(rg) * cur.density, axis=1))
                )
    return c_alpha, c_beta


def _check_nodes(nodes: Any) -> None:
    if is_traced(nodes):
        raise NonDifferentiableError(
            "multipoles needs fixed node coordinates, but they are computed "
            "inside jax.grad, jax.jit or jax.vmap here: the scattered field "
            "subtracts the incident field's interpolant on the mesh, which has "
            "no derivative by the node coordinates. Use a material design (a "
            "gfem.MaterialField), or wrap the nodes in jax.lax.stop_gradient."
        )


def _origin(origin: Any, entry: str = "multipoles") -> np.ndarray:
    try:
        value = np.asarray(origin, float)
        ok = value.shape == (3,) and bool(np.isfinite(value).all())
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(
            f"{entry}: origin must be three finite numbers, the multipole "
            f"origin in m (model coordinates), got {origin!r}."
        )
    return value


def _check_lmax_form(lmax: Any) -> None:
    if lmax is None:
        return
    if isinstance(lmax, bool) or not isinstance(lmax, numbers.Integral):
        raise TypeError(
            "multipoles: lmax must be an int, the highest multipole degree, "
            f"or None for the default, got {lmax!r}."
        )
    if lmax < 2:
        raise ValueError(
            f"multipoles: lmax={lmax} is below 2: the quadrupoles need degree "
            "2, at no extra cost. Give lmax >= 2, or None."
        )
