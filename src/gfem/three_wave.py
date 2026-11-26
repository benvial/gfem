r"""Second-order (three-wave) coupling of port modes and second-harmonic generation.

A material's ``second_order_nonlinear_tensor`` (:math:`d = \chi^{(2)}/2`, m/V)
couples the modes of a wave port that `gfem.solve` found with
`gfem.PortMode`. Two functions read those results and the model's materials:

- `three_wave_coupling`: the overlap tensor :math:`\kappa_{abc}` between the
  waves (the rows) of one or several results, for sum-frequency, difference-
  frequency and parametric processes;
- `second_harmonic`: the second-harmonic figures of a pump and its harmonic row
  by row (normalized efficiency, phase mismatch, poling period) and the
  undepleted-pump efficiency of a guide of a given length, poled or not.

Both integrate the modes at 1 W (1 W/m on a line port) on the port's
cross-section, outside any perfectly matched layer, and read no permittivity.
They hold on lossy sections while :math:`\alpha \ll \beta`. Under `jax.grad`
they read the traced mode fields, so they differentiate with respect to the
section's materials (not the node coordinates).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem import _chi2
from gfem._checks import is_traced
from gfem._materials import Material
from gfem._mode_overlap import _elements, _waves, refuse_bend
from gfem._pml import layer_tetrahedra
from gfem._pytree import register_fields
from gfem.results import PortModeResult

if TYPE_CHECKING:
    from gfem.model import Model

__all__ = [
    "SecondHarmonic",
    "ThreeWaveCoupling",
    "second_harmonic",
    "three_wave_coupling",
]

_EPS0 = scipy.constants.epsilon_0
_PROPERTY = "second_order_nonlinear_tensor"
#: relative mismatch of a harmonic's frequency from twice the pump's: doubling is
#: exact in floating point, and 1e-9 absorbs a wavelength-to-frequency round trip
_PAIRING_TOL = 1e-9


@dataclasses.dataclass(frozen=True, eq=False)
class ThreeWaveCoupling:
    r"""The second-order coupling tensor between the waves of one port.

    A wave is one row of a `gfem.PortModeResult`: a mode at a frequency. In
    the equation of wave :math:`a`'s amplitude (:math:`|A|^2` a power,
    :math:`e^{-i\omega t}`, the convention of coupled-mode tools),

    .. math::

        \frac{\mathrm{d}A_a}{\mathrm{d}z} \ni i \sum_{bc}
        \kappa_{abc}\, A_b A_c\, e^{-i\Delta\beta_{abc} z},
        \qquad \Delta\beta_{abc} = \beta_a - \beta_b - \beta_c,

    summed over every ordered pair with no degeneracy factor, so the factor 2
    of sum-frequency generation against second-harmonic generation comes out
    of the sum. :math:`\kappa_{abc} = \overline{T_{abc}}`,

    .. math::

        T_{abc} = \frac{\omega_a \varepsilon_0}{4} \int_S \sum_{ijk}
        d_{ijk}\, \overline{E_{a,i}}\, E_{b,j}\, E_{c,k}\,\mathrm{d}A,

    with :math:`E` the modes at 1 W, :math:`\omega_a = 2\pi f_a` the angular
    frequency of wave :math:`a` and :math:`d` the material's
    ``second_order_nonlinear_tensor`` in the global frame. It is symmetric in
    :math:`b, c`. Every tuple is computed; an equation uses only those with
    :math:`f_a = f_b + f_c`.

    The difference-frequency and parametric equations follow from the same
    entries: for a pump :math:`a` and two waves :math:`b, c`,
    :math:`\mathrm{d}A_b/\mathrm{d}z \ni i\,(2\omega_b/\omega_a)\,
    \overline{\kappa_{abc}}\,A_a \overline{A_c}\,e^{+i\Delta\beta_{abc} z}`.
    This relation requires a :math:`d` with full permutation symmetry
    (Kleinman symmetry), which one frequency-independent tensor must have for
    energy to be conserved.

    Attributes:
        port: The wave port's name.
        waves: ``(mode, frequency)`` of each wave, in order (Hz).
        kappa: :math:`\kappa_{abc}`, ``(K, K, K)`` complex,
            1/(:math:`\sqrt{\mathrm{W}}` m); W\ :sup:`-1/2` m\ :sup:`-1/2` on
            an Ez line port of a 2D mesh (fields at 1 W/m).
    """

    port: str
    waves: tuple[tuple[int, float], ...]
    kappa: jax.Array

    def __repr__(self) -> str:
        head = f"ThreeWaveCoupling(port {self.port!r}, {len(self.waves)} waves"
        try:
            k = np.asarray(self.kappa)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        return f"{head}, |kappa| up to {np.abs(k).max():.4g} 1/(sqrt(W) m))"


register_fields(ThreeWaveCoupling, ("kappa",))


@dataclasses.dataclass(frozen=True, eq=False)
class SecondHarmonic:
    r"""Second-harmonic generation of a pump mode into a harmonic mode, per row.

    Row :math:`i` pairs the pump at :math:`f_i` with the harmonic at
    :math:`2f_i`. :math:`\kappa = \kappa_{hpp}` is the entry of
    `ThreeWaveCoupling` with the harmonic as wave :math:`a` and the pump as
    :math:`b = c`,

    .. math::

        \kappa = \overline{\frac{\omega\varepsilon_0}{2} \int_S \sum_{ijk}
        d_{ijk}\,\overline{E_{h,i}}\,E_{p,j}\,E_{p,k}\,\mathrm{d}A},

    with :math:`\omega` the pump's angular frequency. For an unpoled guide the
    undepleted harmonic power is :math:`P_{2\omega} = \eta_\mathrm{norm}
    P_\omega^2 L^2\,\mathrm{sinc}^2(\Delta\beta L/2)`; `efficiency` adds
    poling and loss.

    Attributes:
        port: The wave port's name.
        modes: The pump's and the harmonic's mode numbers.
        frequency: ``(n,)`` the pump's frequencies, Hz.
        kappa: :math:`\kappa`, ``(n,)`` complex, 1/(:math:`\sqrt{\mathrm{W}}`
            m); W\ :sup:`-1/2` m\ :sup:`-1/2` on a line port.
        pump_gamma: ``(n,)`` complex :math:`\gamma = \alpha + i\beta` of the
            pump rows, 1/m.
        harmonic_gamma: ``(n,)`` complex :math:`\gamma` of the harmonic rows,
            1/m.
    """

    port: str
    modes: tuple[int, int]
    frequency: np.ndarray
    kappa: jax.Array
    pump_gamma: jax.Array
    harmonic_gamma: jax.Array

    @property
    def delta_beta(self) -> jax.Array:
        r""":math:`\Delta\beta = \beta_{2\omega} - 2\beta_\omega` (1/m), ``(n,)``.

        :math:`\beta = \mathrm{Im}\,\gamma`; it is the exact mismatch of the
        solved modes. Positive under normal dispersion.
        """
        return jnp.imag(self.harmonic_gamma) - 2 * jnp.imag(self.pump_gamma)

    @property
    def poling_period(self) -> jax.Array:
        r"""First-order quasi-phase-matching period :math:`2\pi/|\Delta\beta|` (m).

        Order :math:`m` is :math:`m` times this; ``inf`` where :math:`\Delta\beta
        = 0`.
        """
        return 2 * jnp.pi / jnp.abs(self.delta_beta)

    @property
    def eta_norm(self) -> jax.Array:
        r"""Normalized efficiency :math:`|\kappa|^2`, 1/(W m²).

        1/(W m) on a line port (powers in W/m). Multiply by :math:`10^{-2}`
        for %/(W cm²).
        """
        return jnp.abs(self.kappa) ** 2

    def efficiency(
        self,
        length: Any,
        power: Any,
        *,
        period: Any = None,
        duty_cycle: Any = None,
        order: Any = None,
    ) -> jax.Array:
        r"""The efficiency :math:`P_{2\omega}(L)/P_\omega(0)` in the undepleted pump.

        .. math::

            \frac{P_{2\omega}}{P_\omega} = |G|^2 |\kappa|^2 P_\omega\,
            e^{-(\alpha_{2\omega} + 2\alpha_\omega) L}\,
            \frac{\sinh^2(\Delta\alpha L/2) + \sin^2(\delta L/2)}
            {(\Delta\alpha/2)^2 + (\delta/2)^2},

        with :math:`\alpha = \mathrm{Re}\,\gamma` of each row and
        :math:`\Delta\alpha = \alpha_{2\omega} - 2\alpha_\omega`. Unpoled
        (``period=None``): :math:`G = 1`, :math:`\delta = \Delta\beta`. Poled
        with period :math:`\Lambda`, duty cycle :math:`D` and order :math:`m`:
        :math:`G = \frac{2}{\pi m}\sin(\pi m D)` and :math:`\delta =
        |\Delta\beta| - 2\pi m/\Lambda`. Written so that it, and its
        gradient, are finite at :math:`\Delta\alpha = \delta = 0`, where it is
        :math:`L^2`-quadratic. Lossless it is :math:`|G|^2\eta_\mathrm{norm}
        P_\omega L^2\,\mathrm{sinc}^2(\delta L/2)`.

        A reversed domain carries :math:`-d` (domain inversion of a 3m crystal
        flips every nonzero entry); the first domain's phase matters only for
        parametric phases and is not a parameter. Poling is analytic: the
        overlap is that of the unreversed domain. The formula holds for the
        undepleted pump and :math:`\alpha \ll \beta`.

        Args:
            length: The guide's length :math:`L`, m, :math:`\ge 0`.
            power: The pump's power :math:`P_\omega(0)`, W (W/m on a line
                port), :math:`\ge 0`.
            period: The poling period :math:`\Lambda`, m, > 0; ``None``
                for an unpoled guide.
            duty_cycle: The fraction :math:`D` of a period carrying
                :math:`+d`, in [0, 1]; 0.5 when ``period`` is given and this
                is not. Giving it without ``period`` raises.
            order: The quasi-phase-matching order :math:`m`, an integer
                :math:`\ge 1`; 1 when ``period`` is given and this is not.
                Giving it without ``period`` raises.

        Returns:
            The ratio, broadcast of ``length``, ``power``, ``period``,
            ``duty_cycle`` and ``order`` against the ``(n,)`` rows.
            Differentiable in every argument.

        Raises:
            ValueError: A concrete argument is out of its range, or
                ``duty_cycle`` or ``order`` is given without ``period``.
            TypeError: A concrete argument is complex.
        """
        if period is None and (duty_cycle is not None or order is not None):
            raise ValueError(
                "efficiency: duty_cycle and order apply to a poled guide: give "
                "the period, or leave both out for an unpoled one."
            )
        duty_cycle = 0.5 if duty_cycle is None else duty_cycle
        order = 1 if order is None else order
        _check_range("length", length, lo=0.0)
        _check_range("power", power, lo=0.0)
        if period is not None:
            _check_range("period", period, lo=0.0, strict=True)
        _check_range("duty_cycle", duty_cycle, lo=0.0, hi=1.0)
        if not is_traced(order):
            _check_range("order", order, lo=1.0)
            m = np.asarray(order)
            if m.dtype.kind not in "iu" and not (
                np.issubdtype(m.dtype, np.floating) and (m == np.round(m)).all()
            ):
                raise ValueError(
                    f"efficiency: order must be an integer >= 1, got {order!r}."
                )
        length, power, duty, order = (
            jnp.asarray(x, dtype=jnp.float64)
            for x in (length, power, duty_cycle, order)
        )
        db = self.delta_beta
        if period is None:
            gain, mismatch = 1.0, db
        else:
            gain = (2 / (jnp.pi * order) * jnp.sin(jnp.pi * order * duty)) ** 2
            mismatch = jnp.abs(db) - 2 * jnp.pi * order / jnp.asarray(
                period, dtype=jnp.float64
            )
        alpha_p = jnp.real(self.pump_gamma)
        alpha_h = jnp.real(self.harmonic_gamma)
        u = (alpha_h - 2 * alpha_p) * length / 2
        v = mismatch * length / 2
        half = (alpha_h + 2 * alpha_p) * length / 2
        return (
            gain * self.eta_norm * power * length**2 * _decayed_sinhc_sinc(u, v, half)
        )

    def __repr__(self) -> str:
        n = int(np.shape(self.frequency)[-1])
        head = (
            f"SecondHarmonic(port {self.port!r}, modes {self.modes[0]} -> "
            f"{self.modes[1]}, {n} frequencies"
        )
        try:
            eta = np.asarray(self.eta_norm) * 1e-2
            period = np.asarray(self.poling_period) * 1e6
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        return (
            f"{head}, eta_norm {_range(eta)} %/(W cm^2), "
            f"poling period {_range(period)} um)"
        )


register_fields(SecondHarmonic, ("kappa", "pump_gamma", "harmonic_gamma"))


def _range(a: np.ndarray) -> str:
    lo, hi = float(np.min(a)), float(np.max(a))
    return f"{lo:.4g}" if lo == hi else f"{lo:.4g}-{hi:.4g}"


def _check_range(
    name: str,
    value: Any,
    *,
    lo: float,
    hi: float | None = None,
    strict: bool = False,
) -> None:
    """Raise unless a concrete ``value`` lies in the range (a tracer is skipped)."""
    if is_traced(value):
        return
    if np.iscomplexobj(value):
        raise TypeError(
            f"efficiency: {name} must be real, got a complex value {value!r}."
        )
    a = np.asarray(value, dtype=float)
    bad = (a <= lo) if strict else (a < lo)
    if hi is not None:
        bad = bad | (a > hi)
    if not np.isfinite(a).all() or bad.any():
        bound = f"> {lo:g}" if strict else f">= {lo:g}"
        if hi is not None:
            bound = f"in [{lo:g}, {hi:g}]"
        raise ValueError(f"efficiency: {name} must be {bound}, got {value!r}.")


# :math:`c_k = 2^{2k-1} / (2k)!`, the coefficients of `_decayed_sinhc_sinc`'s series
_SERIES = tuple(2.0 ** (2 * k - 1) / math.factorial(2 * k) for k in range(1, 10))
#: u^2 + v^2 below which the series (truncated after k = 9, a relative 1e-17
#: here) replaces the quotient, which loses all digits as both terms vanish
_SERIES_BELOW = 0.25


def _decayed_sinhc_sinc(u: Any, v: Any, half: Any) -> Any:
    r""":math:`e^{-2h}(\sinh^2 u + \sin^2 v)/(u^2 + v^2)`, smooth at the origin.

    :math:`h` is ``half``, :math:`\ge |u|` for a passive row. The quotient is
    built as :math:`((e^{|u|-h} - e^{-|u|-h})/2)^2 + e^{-2h}\sin^2 v`, whose
    exponents are all :math:`\le 0`: :math:`\sinh^2 u` alone overflows at
    :math:`|u| \gtrsim 355` while the decay underflows, a NaN.

    With :math:`x = u^2`, :math:`y = v^2`, :math:`\sinh^2 u + \sin^2 v =
    \sum_k c_k (x^k - (-y)^k)` and each term divides by :math:`x + y`
    exactly: :math:`\sum_{j<k} x^{k-1-j}(-y)^j`.
    """
    x, y = u * u, v * v
    r2 = x + y
    small = r2 < _SERIES_BELOW
    series = 0.0
    for k, c in enumerate(_SERIES, start=1):
        series = series + c * sum(x ** (k - 1 - j) * (-y) ** j for j in range(k))
    safe = jnp.where(small, 1.0, r2)
    # large arguments overflow in sinh: only the branch that is used is built
    su = jnp.where(small, 0.0, u)
    sv = jnp.where(small, 0.0, v)
    decay = jnp.exp(-2 * half)
    au = jnp.abs(su)
    sinh_decayed = ((jnp.exp(au - half) - jnp.exp(-au - half)) / 2) ** 2
    quotient = (sinh_decayed + decay * jnp.sin(sv) ** 2) / safe
    return jnp.where(small, series * decay, quotient)


# --- materials -------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Chi2:
    """The tensor :math:`d` of the section's elements, in the columns' order.

    ``d`` ``(n_el, 3, 6)`` in the global frame (0 where an element has none),
    ``keep`` ``(n_el,)`` False on a perfectly matched layer's triangles.
    """

    d: jax.Array
    keep: np.ndarray


def _chi2_materials(
    model: Model, section: Any, frequency: np.ndarray, port: str, owner: str
) -> _Chi2:
    """Read the tensor of the elements of ``section`` by owner element.

    Raises:
        ValueError: No element outside a perfectly matched layer carries it.
    """
    del frequency  # the tensor is one constant per material
    mesh = model.mesh
    cells, tags = _elements(model, section)
    names = mesh._region_names()  # noqa: SLF001
    n_el = len(cells)
    d = jnp.zeros((n_el, 3, 6))
    carries = np.zeros(n_el, bool)
    for tag in np.unique(tags):
        region = names[int(tag)]
        material = model.materials[region]
        value = getattr(material, _PROPERTY)
        if value is None:
            continue
        rows = np.flatnonzero(tags == tag)
        pos = np.searchsorted(mesh.elements(region), cells[rows])
        if isinstance(material, Material):
            rotated = jnp.asarray(
                _chi2.in_global_frame(np.asarray(value), material.axes)
            )
            per = jnp.broadcast_to(rotated, (len(rows), 3, 6))
        else:
            rotated = _chi2.in_global_frame(jnp.asarray(value), material.axes)
            per = (
                jnp.broadcast_to(rotated, (len(rows), 3, 6))
                if rotated.ndim == 2
                else rotated[pos]
            )
        carries[rows] = True
        d = d.at[rows].set(per)
    # from the mesh, as the section of a traced field has no layer to read
    layer = layer_tetrahedra(mesh, model.pml) if len(section.normal) == 3 else None
    keep = np.ones(n_el, bool) if layer is None else ~layer[cells]
    if not (carries & keep).any():
        regions = ", ".join(repr(names[int(t)]) for t in np.unique(tags))
        raise ValueError(
            f"{owner}: no element of wave port {port!r}'s section outside a "
            f"perfectly matched layer (whose triangles are skipped) has a "
            f"{_PROPERTY} (its regions are {regions}), so the coupling is "
            "zero. Set one on the core's material."
        )
    return _Chi2(d, keep)


def _three_components(e: jax.Array) -> jax.Array:
    """A mode's point values as three components: an Ez line port is ``(0, 0, E_z)``."""
    if e.shape[-1] == 3:
        return e
    zeros = jnp.zeros_like(e)
    return jnp.concatenate([zeros, zeros, e], axis=-1)


# --- public functions ------------------------------------------------------------


def three_wave_coupling(
    model: Model, modes: PortModeResult | Sequence[PortModeResult]
) -> ThreeWaveCoupling:
    r"""The second-order coupling :math:`\kappa_{abc}` between the waves of a port.

    Every row of every result in ``modes`` is one wave, in order: another mode
    number, another frequency, or both. Entry :math:`abc` is the coefficient
    of :math:`A_b A_c` in wave :math:`a`'s amplitude equation, summed over all
    ordered pairs with no degeneracy factor; see `ThreeWaveCoupling` for the
    formula and the sign convention. Its special cases are second-harmonic
    generation (:math:`\kappa_{hpp}`, `second_harmonic`), sum-frequency
    generation (:math:`\mathrm{d}A_3/\mathrm{d}z = 2i\kappa_{312}A_1A_2
    e^{-i\Delta\beta z}`, with :math:`\Delta\beta = \beta_3 - \beta_1 -
    \beta_2` and, undepleted, :math:`P_3 = 4|\kappa_{312}|^2P_1P_2L^2\,
    \mathrm{sinc}^2(\Delta\beta L/2)`) and difference-frequency or parametric
    processes (the :math:`2\omega_b/\omega_a` relation of `ThreeWaveCoupling`).

    The tensor is read from ``model``'s materials by owner element, rotated to
    the global frame; the waves may come from other models of the same mesh
    that differ in permittivity only (a crystal's permittivity is constant at
    one frequency, so the modes at :math:`\omega` and :math:`2\omega` are
    solved on two models). No permittivity is read: anisotropic and
    dispersive crystals are accepted. The integral runs over the
    cross-section outside any perfectly matched layer, with a quadrature rule
    exact for the modes' polynomial degree. Lossy and leaky sections are
    computed with the modes normalized to 1 W; the formulas hold while
    :math:`\alpha \ll \beta`. An Ez line port of a 2D mesh couples through
    :math:`d_{zzz}` of the global frame only; an Hz line port is not
    supported.

    Exactly linear in the tensor; under `jax.grad` the traced mode fields make
    it differentiable with respect to the section's ``eps_r``, laws, losses
    and density blends (one 2D adjoint per row), not to the node coordinates
    (read :math:`\Delta\beta` from the results' ``gamma`` for shape
    gradients). All waves must share one port, element order and
    cross-section. The cost grows as :math:`K^3`: pass the few waves of a
    process.

    Args:
        model: A model of the mesh the modes were solved on, whose crystal
            regions carry ``second_order_nonlinear_tensor``.
        modes: One `gfem.PortModeResult` or a sequence of them.

    Returns:
        The ``(K, K, K)`` tensor for the :math:`K` waves.

    Raises:
        ValueError: The results are of different ports, element orders or
            section geometries; the model is of another mesh or section; no
            element outside a perfectly matched layer carries the tensor.
        NotImplementedError: An Hz line port; a bent mode (`gfem.Bend`).
        TypeError: An entry is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates of ``model``, or of a
            mode's solve, are traced by `jax.grad`.
    """
    refuse_bend("gfem.three_wave_coupling", modes)
    waves = _waves(model, modes, "gfem.three_wave_coupling", _chi2_materials)
    e = _three_components(waves.field())
    d = waves.materials.d[:, None]
    omega = jnp.asarray(2 * np.pi * waves.frequency)

    @jax.checkpoint
    def overlap(eb: jax.Array) -> jax.Array:
        # P[c] = d : E_b E_c at every point, then the overlap with conj(E_a)
        p = _chi2.contract(d, eb[None], e)
        return jnp.einsum("aeqi,ceqi,eq->ac", jnp.conj(e), p, waves.weights)

    # one wave b at a time: the point values of all (b, c) pairs at once would
    # be (K, K, n_el, n_q, 3) complex
    t = jnp.moveaxis(jax.lax.map(overlap, e), 0, 1)
    kappa = jnp.conj(omega[:, None, None] * _EPS0 / 4 * t)
    return ThreeWaveCoupling(
        port=waves.port,
        waves=tuple(
            (m, float(f)) for m, f in zip(waves.modes, waves.frequency, strict=True)
        ),
        kappa=kappa,
    )


def second_harmonic(
    model: Model, pump: PortModeResult, harmonic: PortModeResult
) -> SecondHarmonic:
    r"""The second-harmonic figures of a pump mode and its harmonic, row by row.

    ``pump`` and ``harmonic`` are one `gfem.PortModeResult` each with the same
    number of rows; row :math:`i` pairs the pump at :math:`f_i` with the
    harmonic at :math:`2f_i`. Solve them on two models of one mesh when the
    crystal's permittivity differs at the two frequencies (the tensor is read
    from ``model``). Cost: one overlap per row, so a curve over frequency
    costs no extra solve. See `SecondHarmonic` for :math:`\kappa`,
    `SecondHarmonic.efficiency` for the guide's efficiency and
    `three_wave_coupling` for the material, the integral and the gradient
    (the same rules: exactly linear in the tensor, differentiable in the
    section's permittivities through the traced modes, not in the node
    coordinates, where ``delta_beta`` and ``poling_period`` taken from the
    results' ``gamma`` still differentiate).

    Args:
        model: A model of the mesh the modes were solved on, with the tensor.
        pump: The result of ``gfem.solve(model, gfem.PortMode(...))`` at
            :math:`f`.
        harmonic: The result at :math:`2f`, on the same port.

    Returns:
        The coupling, mismatch and propagation constants of each row.

    Raises:
        ValueError: The row counts differ or a harmonic's frequency is not
            twice the pump's (relative :math:`10^{-9}`); otherwise as
            `three_wave_coupling`.
        NotImplementedError: An Hz line port; a bent mode (`gfem.Bend`).
        TypeError: An entry is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates of ``model``, or of a
            mode's solve, are traced by `jax.grad`.
    """
    owner = "gfem.second_harmonic"
    refuse_bend(owner, pump, harmonic)
    for name, r in (("pump", pump), ("harmonic", harmonic)):
        if not isinstance(r, PortModeResult):
            raise TypeError(
                f"{owner} takes one gfem.PortModeResult as the {name}, got "
                f"{type(r).__name__}: solve gfem.solve(model, gfem.PortMode(...)) "
                "and pass that result."
            )
    f_p = np.atleast_1d(np.asarray(pump.frequency, float))
    f_h = np.atleast_1d(np.asarray(harmonic.frequency, float))
    if len(f_p) != len(f_h):
        raise ValueError(
            f"{owner}: the pump has {len(f_p)} rows but the harmonic {len(f_h)}; "
            "row i pairs the pump at f with the harmonic at 2 f."
        )
    bad = np.abs(f_h - 2 * f_p) > _PAIRING_TOL * 2 * f_p
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{owner}: row {i} of the harmonic is at {f_h[i]:.12g} Hz, not twice "
            f"the pump's {f_p[i]:.12g} Hz (relative tolerance {_PAIRING_TOL:g})."
        )
    n = len(f_p)
    waves = _waves(model, [pump, harmonic], owner, _chi2_materials)
    d = waves.materials.d[:, None]
    weights = waves.weights

    @jax.checkpoint
    def overlap(pair: tuple[jax.Array, jax.Array]) -> jax.Array:
        ep = _three_components(waves.row(pair[0]))
        eh = _three_components(waves.row(pair[1]))
        p = _chi2.contract(d, ep, ep)
        return jnp.sum(weights * jnp.sum(jnp.conj(eh) * p, axis=-1))

    # one row at a time: the point values of a row are (n_el, n_q, 3) complex
    t = jax.lax.map(overlap, (waves.vectors[:n], waves.vectors[n:]))
    omega = jnp.asarray(2 * np.pi * f_p)
    return SecondHarmonic(
        port=waves.port,
        modes=(int(pump.mode), int(harmonic.mode)),
        frequency=f_p,
        kappa=jnp.conj(omega * _EPS0 / 2 * t),
        pump_gamma=jnp.atleast_1d(jnp.asarray(pump.gamma)),
        harmonic_gamma=jnp.atleast_1d(jnp.asarray(harmonic.gamma)),
    )
