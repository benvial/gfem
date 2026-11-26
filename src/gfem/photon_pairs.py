r"""Photon-pair sources: spontaneous down-conversion and four-wave mixing.

A waveguide whose pump spontaneously emits signal-idler photon pairs is a
photon-pair source, described in the low-gain regime (at most a pair per mode
and pulse). Two functions read the port modes `gfem.solve` found with
`gfem.PortMode` and the model's materials once and return a `PhotonPairSource`:

- `spdc`: spontaneous parametric down-conversion in a :math:`\chi^{(2)}` guide
  (pump :math:`\to` signal + idler), poled or not;
- `sfwm`: degenerate-pump spontaneous four-wave mixing in a :math:`\chi^{(3)}`
  guide (two pump photons :math:`\to` signal + idler).

The source's methods give the figures of a guide of a given length: the pair
rate and spectral brightness of a continuous pump, and the joint spectral
amplitude (`JointSpectrum`) of a Gaussian pulse, with its pairs per pulse,
heralded purity and Schmidt number. Length, pump and poling are method
arguments, so a design sweep costs no overlap.

The coupling is one overlap on the rows nearest the bands' centres; the
dispersion of each band is a polynomial fit of the rows' ``gamma``, so
group-velocity mismatch, which sets the purity, differentiates. The bands act
as ideal rectangular filters: every figure counts the pairs inside them. See
`gfem.conventions` (Photon pairs) for the normalization and units.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem import _photon_pairs as pp
from gfem._checks import is_traced
from gfem._mode_overlap import _NODES, _vector
from gfem._pytree import readonly, register_fields
from gfem.errors import (
    HighGainWarning,
    NonDifferentiableError,
    UnresolvedJointSpectrumWarning,
)
from gfem.kerr import kerr_coupling
from gfem.results import PortModeField, PortModeResult
from gfem.three_wave import _decayed_sinhc_sinc, three_wave_coupling

__all__ = ["JointSpectrum", "PhotonPairSource", "sfwm", "spdc"]

#: pairs per mode (CW: pairs/(s Hz); pulsed: ``pairs_per_pulse * sqrt(purity)``,
#: an upper bound on the mean number of pairs in the strongest Schmidt mode)
#: above which the low-gain result is suspect. The exact phase-matched density
#: is :math:`\sinh^2(gL)` against the first order :math:`(gL)^2`, whose relative
#: error is :math:`\approx (gL)^2/3`; a mean number :math:`\bar n` also means a
#: multi-pair probability :math:`\sim\bar n`. 0.1 keeps both near 3 %.
_OCCUPATION_LIMIT = 0.1
#: largest jump of :math:`|f|` between grid neighbours, as a fraction of its
#: peak, before a joint spectrum is called unresolved. The slope of
#: :math:`\mathrm{sinc}\,x` peaks at 0.44 per unit :math:`x` and a lobe is
#: :math:`2\pi` wide, so a jump of 0.25 allows 0.57 rad per step: at least 11
#: points per lobe.
_JUMP_LIMIT = 0.25
#: :math:`|f|` on the band edge, as a fraction of its peak, above which the
#: bands cut the pairs. An amplitude :math:`\mathrm{sinc}` of 0.1 sits near
#: :math:`3\pi`, beyond which about 3 % of the pairs lie (the tail of
#: :math:`\mathrm{sinc}^2` past :math:`X` holds :math:`1/(\pi X)`).
_EDGE_LIMIT = 0.1
#: a fit's largest row residual times the guide length, in rad, above which the
#: dispersion model no longer follows the solved rows
_FIT_PHASE_LIMIT = 0.1
#: relative slack on a frequency range check, absorbing a round trip through
#: wavelength
_RANGE_TOL = 1e-9

_ARMS = ("pump", "signal", "idler")


# --- results -----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class JointSpectrum:
    r"""The joint spectral amplitude of the pairs of one pump pulse.

    :math:`f(f_s, f_i) = 2\pi F` with :math:`F` the two-photon amplitude of
    `gfem.spdc` or `gfem.sfwm` (the pump envelope times the phase-matching
    amplitude, in the published :math:`e^{-i\omega t}` convention; purity and
    Schmidt number do not depend on that choice), scaled so that

    .. math::

        \sum_{jk} |f_{jk}|^2\,\delta f_s\,\delta f_i

    is the mean number of pairs per pulse inside the bands, with
    :math:`\delta f` the grid spacings. For identical photons (one mode family
    carries both) :math:`f` carries a further factor :math:`1/\sqrt2`: the
    mirror points :math:`(f_s, f_i)` and :math:`(f_i, f_s)` are one pair.

    Attributes:
        signal_frequency: ``(Ns,)`` Hz, uniform.
        idler_frequency: ``(Ni,)`` Hz, uniform.
        jsa: ``(Ns, Ni)`` complex, 1/Hz.
    """

    signal_frequency: np.ndarray
    idler_frequency: np.ndarray
    jsa: jax.Array

    @property
    def pairs_per_pulse(self) -> jax.Array:
        r"""Mean pairs per pulse in the bands, :math:`\sum|f|^2\delta f_s\delta f_i`."""
        return pp.pair_count(
            self.jsa, _step(self.signal_frequency), _step(self.idler_frequency)
        )

    @property
    def purity(self) -> jax.Array:
        r"""Heralded purity :math:`\sum_k\lambda_k^2` of the Schmidt weights.

        Computed as :math:`\mathrm{Tr}[(MM^\dagger)^2]/\mathrm{Tr}[MM^\dagger]^2`
        with :math:`M` the amplitude on the uniform grid: no singular value
        decomposition, whose derivative is singular at degenerate weights, so
        it differentiates everywhere. It is 1 for a separable amplitude. For
        identical photons whose bands overlap it is the purity of the
        symmetric amplitude as sampled.
        """
        return pp.trace_purity(self.jsa)

    @property
    def schmidt_number(self) -> jax.Array:
        r"""The Schmidt number :math:`K = 1/\sum_k\lambda_k^2` (1 if separable)."""
        return 1 / self.purity

    def __repr__(self) -> str:
        shape = f"{len(self.signal_frequency)} x {len(self.idler_frequency)}"
        try:
            n, p = float(self.pairs_per_pulse), float(self.purity)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"JointSpectrum({shape} grid)"
        return (
            f"JointSpectrum({shape} grid, {n:.4g} pairs/pulse, purity {p:.4g}, "
            f"Schmidt number {1 / p:.4g})"
        )


register_fields(JointSpectrum, ("jsa",))


@dataclasses.dataclass(frozen=True, eq=False)
class PhotonPairSource:
    r"""A photon-pair source: the pump, signal and idler modes of one waveguide.

    Built by `gfem.spdc` or `gfem.sfwm`. The two-photon amplitude is, with
    :math:`\omega = 2\pi f` and :math:`\tilde A_p` the pump envelope,

    .. math::

        F(\omega_s,\omega_i) = g_{si}\,G_m\,\tilde A_p(\omega_s+\omega_i)\,
        L\,e^{i\delta L/2}\,\mathrm{sinc}\frac{\delta L}{2},
        \qquad
        g_{si} = \frac{2\sqrt{\omega_s\omega_i}}{\omega_p}\,\overline{\kappa_{psi}}

    for spontaneous down-conversion (:math:`\delta = \Delta\beta -
    \mathrm{sign}(\Delta\beta_0)\,2\pi m/\Lambda`, :math:`\Delta\beta =
    \beta_p - \beta_s - \beta_i`, :math:`|G_m| = \frac{2}{\pi m}|\sin\pi mD|`,
    both 1 and 0 unpoled; :math:`\kappa_{psi}` the entry of
    `gfem.three_wave_coupling` with the pump as wave :math:`a`), and

    .. math::

        F(\omega_s,\omega_i) = \Gamma_{si}\,[\tilde A_p\ast\tilde A_p]
        (\omega_s+\omega_i)\,L\,e^{i\delta L/2}\,\mathrm{sinc}\frac{\delta L}{2},
        \quad
        \Gamma_{si} = \tfrac12\Big(\sqrt{\tfrac{\omega_i}{\omega_s}}\,
        \gamma_{sppi} + \sqrt{\tfrac{\omega_s}{\omega_i}}\,\gamma_{ipps}\Big)

    for four-wave mixing (:math:`\Delta\beta = 2\beta_p - \beta_s - \beta_i`
    with :math:`\beta_p` at :math:`(\omega_s+\omega_i)/2`; :math:`\gamma_{abcd}`
    the entries of `gfem.kerr_coupling`). Under loss the factor
    :math:`L\,e^{i\delta L/2}\mathrm{sinc}(\delta L/2)` becomes the coherent
    :math:`\int_0^L e^{-\alpha_p z - (\alpha_s + \alpha_i)(L-z) + i\delta z}
    \mathrm{d}z` (:math:`2\alpha_p` for four-wave mixing), with
    :math:`\alpha = \mathrm{Re}\,\gamma` of each fitted arm: coincidences only,
    no singles or heralding efficiency.

    The coupling is taken on the rows nearest the bands' centres. The solved
    bands are the grid and act as ideal filters. The spontaneous-down-conversion
    coupling rests on the difference-frequency rule of
    `gfem.ThreeWaveCoupling`, which needs a :math:`d` with Kleinman symmetry; it
    is documented, not checked. A pulsed four-wave mixing amplitude omits the
    self- and cross-phase modulation (valid while :math:`\mathrm{Re}\,
    \gamma_\mathrm{NL}P_\mathrm{peak}L \ll 1`); the continuous-wave figures
    include it.

    Attributes:
        process: ``"spdc"`` or ``"sfwm"``.
        port: The wave port's name.
        modes: The pump's, signal's and idler's mode numbers.
        identical: Whether signal and idler are one mode family over
            overlapping bands: the pairs are indistinguishable and counted
            once.
        signal_frequency: ``(Ns,)`` Hz, uniform over the signal rows' span.
        idler_frequency: ``(Ni,)`` Hz, uniform over the idler rows' span.
        bands: The ``(lowest, highest)`` row frequency of the pump, signal and
            idler, Hz.
        coupling: :math:`\overline{\kappa_{psi}}`, 1/(:math:`\sqrt{\mathrm{W}}`
            m), for down-conversion; :math:`(\gamma_{sppi}, \gamma_{ipps})`,
            1/(W m), for four-wave mixing.
        kerr_phase: Four-wave mixing only, :math:`(\gamma_{pppp}, \gamma_{sspp},
            \gamma_{iipp})`, 1/(W m); ``None`` otherwise.
        fit: Each arm's polynomial coefficients of ``gamma`` in the scaled
            frequency, 1/m.
        fit_residual: ``(3,)`` the largest row residual of each arm's fit, 1/m.
    """

    process: str
    port: str
    modes: tuple[int, int, int]
    identical: bool
    signal_frequency: np.ndarray
    idler_frequency: np.ndarray
    bands: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    coupling: jax.Array
    kerr_phase: jax.Array | None
    fit: tuple[jax.Array, jax.Array, jax.Array]
    fit_residual: jax.Array

    # --- geometry of the bands ---

    @property
    def _centres(self) -> tuple[float, float, float]:
        """Signal and idler band centres and the pump's: Hz."""
        fs0 = 0.5 * sum(self.bands[1])
        fi0 = 0.5 * sum(self.bands[2])
        pump = fs0 + fi0 if self.process == "spdc" else 0.5 * (fs0 + fi0)
        return fs0, fi0, pump

    def _pump_frequency(self, fs: Any, fi: Any) -> Any:
        """The pump frequency that energy conservation pairs with ``fs`` and ``fi``."""
        total = fs + fi
        return total if self.process == "spdc" else total / 2

    def _gamma(self, arm: int, f: Any) -> jax.Array:
        return pp.polynomial(self.fit[arm], f, self.bands[arm])

    def _mismatch(self, fs: Any, fi: Any) -> tuple[jax.Array, ...]:
        """``(delta_beta, alpha_s, alpha_i, alpha_p)`` at ``fs``, ``fi`` (broadcast)."""
        gs, gi = self._gamma(1, fs), self._gamma(2, fi)
        gp = self._gamma(0, self._pump_frequency(fs, fi))
        pump = 1 if self.process == "spdc" else 2
        db = pump * jnp.imag(gp) - jnp.imag(gs) - jnp.imag(gi)
        return db, jnp.real(gs), jnp.real(gi), jnp.real(gp)

    def _centre_mismatch(self) -> jax.Array:
        fs0, fi0, _ = self._centres
        return self._mismatch(fs0, fi0)[0]

    def _require_pump(self, owner: str, lo: float, hi: float) -> None:
        """Raise unless the pump rows cover ``[lo, hi]`` Hz (no extrapolation)."""
        have = self.bands[0]
        slack = _RANGE_TOL * (have[1] - have[0])
        if lo < have[0] - slack or hi > have[1] + slack:
            raise ValueError(_missing(owner, "pump", have, (lo, hi), self.process))

    @property
    def delta_beta(self) -> jax.Array:
        r"""The linear mismatch :math:`\Delta\beta` on the grid, 1/m, ``(Ns, Ni)``.

        :math:`\beta_p - \beta_s - \beta_i` for down-conversion,
        :math:`2\beta_p - \beta_s - \beta_i` for four-wave mixing (without the
        power term), from the fitted arms; differentiable in the rows'
        ``gamma``, so also in the shape of the guide.

        Raises:
            ValueError: The pump rows do not cover the sums of the grid.
        """
        fs, fi = self.signal_frequency, self.idler_frequency
        self._require_pump("gfem.photon_pairs", *self._pump_range())
        return self._mismatch(fs[:, None], fi[None, :])[0]

    def _pump_range(self) -> tuple[float, float]:
        fs, fi = self.signal_frequency, self.idler_frequency
        lo, hi = fs[0] + fi[0], fs[-1] + fi[-1]
        return (lo, hi) if self.process == "spdc" else (lo / 2, hi / 2)

    @property
    def poling_period(self) -> jax.Array:
        r"""First-order quasi-phase-matching period :math:`2\pi/|\Delta\beta_0|` (m).

        :math:`\Delta\beta_0` is the mismatch at the bands' centres of the
        fitted arms; ``inf`` where it vanishes. Order :math:`m` is :math:`m`
        times this. Differentiable in the rows' ``gamma``.

        Raises:
            ValueError: Four-wave mixing is not quasi-phase-matched by poling.
        """
        if self.process != "spdc":
            raise ValueError(_NO_POLING)
        return 2 * jnp.pi / jnp.abs(self._centre_mismatch())

    # --- poling and arguments ---

    def _poling(
        self, owner: str, period: Any, duty_cycle: Any, order: Any
    ) -> tuple[Any, Any, Any]:
        """The checked ``(period, duty_cycle, order)`` as float arrays, or ``None``s."""
        if self.process != "spdc":
            if period is not None or duty_cycle is not None or order is not None:
                raise ValueError(f"{owner}: {_NO_POLING}.")
            return None, None, None
        if period is None and (duty_cycle is not None or order is not None):
            raise ValueError(
                f"{owner}: duty_cycle and order apply to a poled guide: give "
                "the period, or leave both out for an unpoled one."
            )
        if period is None:
            return None, None, None
        duty_cycle = 0.5 if duty_cycle is None else duty_cycle
        order = 1 if order is None else order
        _check_range(owner, "period", period, lo=0.0, strict=True)
        _check_range(owner, "duty_cycle", duty_cycle, lo=0.0, hi=1.0)
        if not is_traced(order):
            _check_range(owner, "order", order, lo=1.0)
            m = np.asarray(order)
            if m.dtype.kind not in "iu" and not (
                np.issubdtype(m.dtype, np.floating) and (m == np.round(m)).all()
            ):
                raise ValueError(
                    f"{owner}: order must be an integer >= 1, got {order!r}."
                )
        a, b, c = (
            jnp.asarray(x, dtype=jnp.float64) for x in (period, duty_cycle, order)
        )
        return a, b, c

    def _poling_terms(
        self, period: Any, duty_cycle: Any, order: Any, centre: jax.Array
    ) -> tuple[Any, Any]:
        """``(gain |G|^2-signed amplitude, mismatch shift K)`` of a poled guide."""
        if period is None:
            return 1.0, 0.0
        gain = pp.poling_gain(duty_cycle, order)
        sign = jnp.where(centre < 0, -1.0, 1.0)
        return gain, sign * 2 * jnp.pi * order / period

    def _check_fit(self, owner: str, length: Any) -> None:
        """Raise when an arm's fit leaves a phase above the limit over ``length``."""
        if is_traced(length) or is_traced(self.fit_residual):
            return
        phase = np.asarray(self.fit_residual) * float(np.max(np.asarray(length)))
        worst = int(np.argmax(phase))
        if phase[worst] > _FIT_PHASE_LIMIT:
            raise ValueError(
                f"{owner}: the {_ARMS[worst]} arm's polynomial fit leaves a "
                f"residual of {np.asarray(self.fit_residual)[worst]:.3g} 1/m, "
                f"{phase[worst]:.3g} rad over this length (limit "
                f"{_FIT_PHASE_LIMIT} rad): solve more rows or a narrower band."
            )

    # --- continuous wave ---

    def _density(
        self,
        owner: str,
        length: Any,
        power: Any,
        *,
        period: Any,
        duty_cycle: Any,
        order: Any,
    ) -> jax.Array:
        r"""The signal's spectral density :math:`n_s`, ``(..., Ns)``, pairs/(s Hz)."""
        period, duty_cycle, order = self._poling(owner, period, duty_cycle, order)
        _check_range(owner, "length", length, lo=0.0)
        _check_range(owner, "power", power, lo=0.0)
        self._check_fit(owner, length)
        fs0, fi0, f_pump = self._centres
        self._require_pump(owner, f_pump, f_pump)
        length, power = (jnp.asarray(x, dtype=jnp.float64) for x in (length, power))
        shape = jnp.broadcast_shapes(
            *(
                jnp.shape(x)
                for x in (length, power, period, duty_cycle, order)
                if x is not None
            )
        )
        length, power = (jnp.broadcast_to(x, shape)[..., None] for x in (length, power))
        if period is not None:
            period, duty_cycle, order = (
                jnp.broadcast_to(x, shape)[..., None]
                for x in (period, duty_cycle, order)
            )
        fs = np.asarray(self.signal_frequency)
        fi = fs0 + fi0 - fs
        lo, hi = self.bands[2]
        slack = _RANGE_TOL * (hi - lo)
        inside = (fi >= lo - slack) & (fi <= hi + slack)
        fi = np.clip(fi, lo, hi)
        ws, wi = 2 * np.pi * fs, 2 * np.pi * fi
        gs, gi = self._gamma(1, fs), self._gamma(2, fi)
        gp = self._gamma(0, f_pump)
        if self.process == "spdc":
            db = jnp.imag(gp) - jnp.imag(gs) - jnp.imag(gi)
            gain, shift = self._poling_terms(
                period, duty_cycle, order, self._centre_mismatch()
            )
            coupling = 2 * np.sqrt(ws * wi) / (ws + wi) * jnp.abs(self.coupling)
            strength = (gain * coupling) ** 2 * power
            a_pump = jnp.real(gp)
            mismatch = db - shift
        else:
            if self.kerr_phase is None:
                raise TypeError("a four-wave mixing source carries its Kerr phase.")
            db = 2 * jnp.imag(gp) - jnp.imag(gs) - jnp.imag(gi)
            s, i = self.coupling[0], self.coupling[1]
            gamma_si = 0.5 * (np.sqrt(wi / ws) * s + np.sqrt(ws / wi) * i)
            strength = jnp.abs(gamma_si) ** 2 * power**2
            pp_, ss, ii = self.kerr_phase
            mismatch = db + 2 * jnp.real(pp_ - ss - ii) * power
            a_pump = 2 * jnp.real(gp)
        a_pair = jnp.real(gs) + jnp.real(gi)
        u = (a_pair - a_pump) * length / 2
        v = mismatch * length / 2
        h = (a_pair + a_pump) * length / 2
        density = strength * length**2 * _decayed_sinhc_sinc(u, v, h) * inside
        _warn_gain(owner, density)
        return density

    def spectral_brightness(
        self,
        length: Any,
        power: Any,
        *,
        period: Any = None,
        duty_cycle: Any = None,
        order: Any = None,
    ) -> jax.Array:
        r"""The signal spectral density of pairs, pairs/(s Hz), on the signal grid.

        For a continuous pump of power :math:`P` the idler sits at
        :math:`f_{s0} + f_{i0} - f_s` (energy conservation with the pump at the
        sum of the band centres; four-wave mixing: the pump at half). The
        density is zero where the idler leaves its band, and

        .. math::

            n_s = g_{si}^2 |G_m|^2 P L^2\,\mathrm{sinc}^2\frac{\delta L}{2}
            \quad\Big(|\Gamma_{si}|^2 P^2 L^2\,\mathrm{sinc}^2\frac{\delta L}{2}\Big)

        (lossless; under loss the coincidence law of
        `gfem.SecondHarmonic.efficiency` with the sums of attenuations of
        `PhotonPairSource`), in the pump's low-gain regime, where it is also
        the occupation of each spectral-temporal mode. Four-wave mixing's
        :math:`\delta` includes :math:`2\,\mathrm{Re}(\gamma_{pppp} -
        \gamma_{sspp} - \gamma_{iipp})P`. For identical photons it is the
        density of photons, each pair being two of them. Divide by ``power``
        (squared for four-wave mixing) for a brightness per watt; multiply by
        :math:`c/\lambda^2` for per nm.

        Args:
            length: The guide's length, m, :math:`\ge 0`.
            power: The pump's power, W, :math:`\ge 0`.
            period: Poling period :math:`\Lambda`, m, > 0; ``None`` unpoled.
                Down-conversion only.
            duty_cycle: Fraction :math:`D` of a period carrying :math:`+d`, in
                [0, 1]; 0.5 when ``period`` is given and this is not.
            order: Quasi-phase-matching order :math:`m`, an integer
                :math:`\ge 1`; 1 when ``period`` is given and this is not.

        Returns:
            The density, the broadcast shape of the arguments followed by
            ``(Ns,)``. Differentiable in every argument and, through the
            overlaps and fits, in the section's materials.

        Raises:
            ValueError: A concrete argument is out of range; a poling argument
                is given to four-wave mixing, or ``duty_cycle`` or ``order``
                without ``period``; the pump rows do not cover the pump
                frequency; a fit leaves more than 0.1 rad over the length.
            TypeError: A concrete argument is complex.

        Warns:
            HighGainWarning: The density exceeds 0.1 pairs/(s Hz).
        """
        return self._density(
            "spectral_brightness",
            length,
            power,
            period=period,
            duty_cycle=duty_cycle,
            order=order,
        )

    def rate(
        self,
        length: Any,
        power: Any,
        *,
        period: Any = None,
        duty_cycle: Any = None,
        order: Any = None,
    ) -> jax.Array:
        r"""The pair rate inside the bands, pairs/s, of a continuous pump.

        :math:`R = \int n_s\,\mathrm{d}f_s` over the signal band (the idler is
        fixed by energy conservation), halved for identical photons. Arguments
        and refusals as `spectral_brightness`; the integral is a trapezoidal
        rule on the grid, so widen the band or add ``points`` until it
        converges.

        Returns:
            The rate, the broadcast shape of the arguments. Differentiable in
            every argument and in the section's materials.

        Warns:
            HighGainWarning: The peak density exceeds 0.1 pairs/(s Hz).
        """
        density = self._density(
            "rate", length, power, period=period, duty_cycle=duty_cycle, order=order
        )
        return pp.cw_rate(density, _step(self.signal_frequency), self.identical)

    # --- pulsed ---

    def joint_spectrum(
        self,
        length: Any,
        *,
        pulse_energy: Any,
        pulse_duration: Any,
        period: Any = None,
        duty_cycle: Any = None,
        order: Any = None,
    ) -> JointSpectrum:
        r"""The joint spectral amplitude of a transform-limited Gaussian pump pulse.

        The pulse is centred at the sum of the band centres (four-wave mixing:
        their half), has energy :math:`U` and intensity FWHM :math:`T`, so
        :math:`|A(t)|^2 = P_\mathrm{pk}e^{-t^2/\tau^2}` with
        :math:`\tau = T/(2\sqrt{\ln2})` and :math:`P_\mathrm{pk} =
        U/(\sqrt\pi\tau)`. Its envelope (for four-wave mixing the spectrum of
        :math:`A^2`, the autoconvolution) multiplies the amplitude of
        `PhotonPairSource`. The pump rows must cover the sums of the grid
        (four-wave mixing: their half). Self- and cross-phase modulation are
        omitted.

        Args:
            length: The guide's length, m, :math:`\ge 0`.
            pulse_energy: The pulse's energy, J, :math:`\ge 0`.
            pulse_duration: The pulse's intensity FWHM, s, > 0.
            period: Poling period, m, > 0; ``None`` unpoled. Down-conversion
                only.
            duty_cycle: Fraction of a period carrying :math:`+d`, in [0, 1];
                0.5 with ``period``.
            order: Quasi-phase-matching order, integer :math:`\ge 1`; 1 with
                ``period``.

        Returns:
            The `JointSpectrum`. Every argument is a scalar (``jax.vmap`` a
            sweep); differentiable in each and in the section's materials.

        Raises:
            ValueError: As `spectral_brightness`; an argument is not a scalar;
                the pump rows do not cover the sums of the grid.

        Warns:
            HighGainWarning: ``pairs_per_pulse * sqrt(purity)`` exceeds 0.1.
            UnresolvedJointSpectrumWarning: The grid is too coarse for the
                lobes of the amplitude, or the bands cut it.
        """
        owner = "joint_spectrum"
        period, duty_cycle, order = self._poling(owner, period, duty_cycle, order)
        _check_range(owner, "length", length, lo=0.0)
        _check_range(owner, "pulse_energy", pulse_energy, lo=0.0)
        _check_range(owner, "pulse_duration", pulse_duration, lo=0.0, strict=True)
        for name, x in (
            ("length", length),
            ("pulse_energy", pulse_energy),
            ("pulse_duration", pulse_duration),
            ("period", period),
            ("duty_cycle", duty_cycle),
            ("order", order),
        ):
            if x is not None and jnp.ndim(x) != 0:
                raise ValueError(
                    f"{owner}: {name} must be a scalar (use jax.vmap for a sweep), "
                    f"got shape {jnp.shape(x)}."
                )
        self._check_fit(owner, length)
        self._require_pump(owner, *self._pump_range())
        length, energy, duration = (
            jnp.asarray(x, dtype=jnp.float64)
            for x in (length, pulse_energy, pulse_duration)
        )
        fs = np.asarray(self.signal_frequency)[:, None]
        fi = np.asarray(self.idler_frequency)[None, :]
        ws, wi = 2 * np.pi * fs, 2 * np.pi * fi
        fs0, fi0, _ = self._centres
        offset = 2 * np.pi * (fs + fi - fs0 - fi0)
        db, a_s, a_i, a_p = self._mismatch(fs, fi)
        if self.process == "spdc":
            gain, shift = self._poling_terms(
                period, duty_cycle, order, self._centre_mismatch()
            )
            g = 2 * np.sqrt(ws * wi) / (ws + wi) * self.coupling
            envelope = g * gain * pp.pump_spectrum(offset, energy, duration)
            mismatch, a_pump = db - shift, a_p
        else:
            s, i = self.coupling[0], self.coupling[1]
            gamma_si = 0.5 * (np.sqrt(wi / ws) * s + np.sqrt(ws / wi) * i)
            envelope = gamma_si * pp.pump_square_spectrum(offset, energy, duration)
            mismatch, a_pump = db, 2 * a_p
        amplitude = pp.pair_amplitude(length, mismatch, a_s + a_i, a_pump)
        jsa = 2 * np.pi * pp.identical_scale(self.identical) * envelope * amplitude
        out = JointSpectrum(self.signal_frequency, self.idler_frequency, jsa)
        _warn_resolution(jsa)
        if not is_traced(jsa):
            n, p = float(out.pairs_per_pulse), float(out.purity)
            if n * math.sqrt(p) > _OCCUPATION_LIMIT:
                warnings.warn(
                    f"gfem.{self.process}: {n:.3g} pairs per pulse at purity "
                    f"{p:.3g} put up to {n * math.sqrt(p):.3g} pairs in the "
                    f"strongest mode, above {_OCCUPATION_LIMIT}: the low-gain "
                    "result under-counts and multi-pair emission matters. Lower "
                    "the pump.",
                    HighGainWarning,
                    stacklevel=2,
                )
        return out

    def __repr__(self) -> str:
        bands = ", ".join(
            f"{name} {lo * 1e-12:.5g}-{hi * 1e-12:.5g} THz"
            for name, (lo, hi) in zip(_ARMS, self.bands, strict=True)
        )
        head = (
            f"PhotonPairSource({self.process}, port {self.port!r}, modes "
            f"{self.modes[0]} -> {self.modes[1]} + {self.modes[2]}"
            f"{', identical' if self.identical else ''}, {bands}"
        )
        try:
            c = np.abs(np.asarray(self.coupling)).max()
            period = (
                f", poling period {float(self.poling_period) * 1e6:.5g} um"
                if self.process == "spdc"
                else ""
            )
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        unit = "1/(sqrt(W) m)" if self.process == "spdc" else "1/(W m)"
        return f"{head}, |coupling| {c:.4g} {unit}{period})"


register_fields(PhotonPairSource, ("coupling", "kerr_phase", "fit", "fit_residual"))

_NO_POLING = "four-wave mixing is not quasi-phase-matched by poling"


# --- checks and warnings -----------------------------------------------------------


def _step(f: np.ndarray) -> float:
    return float(f[1] - f[0])


def _check_range(
    owner: str,
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
        raise TypeError(f"{owner}: {name} must be real, got a complex value {value!r}.")
    a = np.asarray(value, dtype=float)
    bad = (a <= lo) if strict else (a < lo)
    if hi is not None:
        bad = bad | (a > hi)
    if not np.isfinite(a).all() or bad.any():
        bound = f"> {lo:g}" if strict else f">= {lo:g}"
        if hi is not None:
            bound = f"in [{lo:g}, {hi:g}]"
        raise ValueError(f"{owner}: {name} must be {bound}, got {value!r}.")


def _missing(
    owner: str,
    arm: str,
    have: tuple[float, float],
    need: tuple[float, float],
    process: str,
) -> str:
    lo, hi = need
    n = 9
    return (
        f"{owner}: the {arm} rows span {have[0]:.12g} to {have[1]:.12g} Hz but "
        f"the {process} amplitude needs the {arm} from {lo:.12g} to {hi:.12g} Hz "
        "(an arm is never extrapolated): solve it with "
        f"np.linspace({lo:.12g}, {hi:.12g}, {n})."
    )


def _warn_gain(owner: str, density: jax.Array) -> None:
    if is_traced(density):
        return
    peak = float(np.max(np.asarray(density)))
    if peak > _OCCUPATION_LIMIT:
        warnings.warn(
            f"gfem.photon_pairs.{owner}: the peak spectral density is {peak:.3g} "
            f"pairs/(s Hz), above {_OCCUPATION_LIMIT}: the low-gain result "
            "under-counts and multi-pair emission matters. Lower the pump power "
            "or the length.",
            HighGainWarning,
            stacklevel=4,
        )


def _warn_resolution(jsa: jax.Array) -> None:
    if is_traced(jsa):
        return
    a = np.abs(np.asarray(jsa))
    peak = float(a.max())
    if peak == 0.0:
        return
    jump = max(np.abs(np.diff(a, axis=0)).max(), np.abs(np.diff(a, axis=1)).max())
    edge = max(a[0].max(), a[-1].max(), a[:, 0].max(), a[:, -1].max())
    if jump > _JUMP_LIMIT * peak:
        warnings.warn(
            f"gfem.photon_pairs: the joint spectrum jumps by {jump / peak:.2f} of "
            f"its peak between grid neighbours (limit {_JUMP_LIMIT}): fewer than "
            "about 11 points per lobe. Use more points, or a shorter guide.",
            UnresolvedJointSpectrumWarning,
            stacklevel=4,
        )
    if edge > _EDGE_LIMIT * peak:
        warnings.warn(
            f"gfem.photon_pairs: the joint spectrum is {edge / peak:.2f} of its "
            f"peak on a band edge (limit {_EDGE_LIMIT}): the bands filter the "
            "pairs; widen them to count all.",
            UnresolvedJointSpectrumWarning,
            stacklevel=4,
        )


# --- construction ------------------------------------------------------------------


def _row(result: PortModeResult, i: int) -> PortModeResult:
    """Row ``i`` of ``result`` as a one-frequency result (the field's vector sliced)."""
    field = result.e_field
    if field is None:
        raise TypeError(
            f"gfem.photon_pairs: the result of wave port {result.port!r} carries no "
            "electric field (a driven result's port modes do not): solve "
            "gfem.solve(model, gfem.PortMode(...)) and pass that result."
        )
    sliced = PortModeField._on_vector(  # noqa: SLF001
        result.port,
        lambda: field._space,  # noqa: SLF001
        _vector(field)[i : i + 1],
        quantity=field.quantity,
        unit=field.unit,
    )
    parts = {
        name: getattr(result, name)[i : i + 1]
        for name in (
            "frequency",
            "gamma",
            "Z_PV",
            "residual",
            "solved",
            "n_g",
            "beta_2",
            "beta_3",
        )
        if getattr(result, name) is not None
    }
    return dataclasses.replace(result, e_field=sliced, **parts)


def _build(
    process: str,
    model: Any,
    pump: PortModeResult,
    signal: PortModeResult,
    idler: PortModeResult,
    *,
    points: int,
) -> PhotonPairSource:
    owner = f"gfem.{process}"
    arms = (pump, signal, idler)
    for name, r in zip(_ARMS, arms, strict=True):
        if not isinstance(r, PortModeResult):
            raise TypeError(
                f"{owner} takes one gfem.PortModeResult as the {name}, got "
                f"{type(r).__name__}: solve gfem.solve(model, gfem.PortMode(...)) "
                "and pass that result."
            )
    if (
        isinstance(points, bool)
        or not isinstance(points, int | np.integer)
        or points < 3
    ):
        raise ValueError(f"{owner}: points must be an integer >= 3, got {points!r}.")
    for r in arms:
        try:
            line = r.h_field is not None or (
                r.e_field is not None and len(r.e_field._space.section.normal) != 3  # noqa: SLF001
            )
        except NonDifferentiableError as err:
            raise NonDifferentiableError(_NODES.format(owner=owner)) from err
        if line:
            raise NotImplementedError(
                f"{owner}: wave port {r.port!r} is a line port (a 2D mesh): a pair "
                "rate of a slab is a density over transverse wavenumbers, not a "
                "figure of a guide. Use a 3D wave port section."
            )
    freqs, fits, residuals, bands = [], [], [], []
    for name, r in zip(_ARMS, arms, strict=True):
        f = np.atleast_1d(np.asarray(r.frequency, float))
        if len(f) < 4:
            raise ValueError(
                f"{owner}: the {name} has {len(f)} rows; each arm needs at least 4 "
                "to fit its dispersion (solve a band, e.g. np.linspace(lo, hi, 9))."
            )
        pinv, vander = pp.arm_basis(f)
        gamma = jnp.atleast_1d(jnp.asarray(r.gamma))
        coeffs = jnp.asarray(pinv) @ gamma
        residuals.append(jnp.max(jnp.abs(jnp.asarray(vander) @ coeffs - gamma)))
        fits.append(coeffs)
        freqs.append(f)
        bands.append((float(f.min()), float(f.max())))
    fs0 = 0.5 * sum(bands[1])
    fi0 = 0.5 * sum(bands[2])
    f_pump = fs0 + fi0 if process == "spdc" else 0.5 * (fs0 + fi0)
    slack = _RANGE_TOL * (bands[0][1] - bands[0][0])
    if not bands[0][0] - slack <= f_pump <= bands[0][1] + slack:
        raise ValueError(_missing(owner, "pump", bands[0], (f_pump, f_pump), process))
    same_mode = signal.port == idler.port and int(signal.mode) == int(idler.mode)
    overlap = max(bands[1][0], bands[2][0]) <= min(bands[1][1], bands[2][1])
    identical = signal is idler or (same_mode and overlap)
    near = [
        int(np.argmin(np.abs(f - c)))
        for f, c in zip(freqs, (f_pump, fs0, fi0), strict=True)
    ]
    rows = [_row(r, i) for r, i in zip(arms, near, strict=True)]
    kerr_phase = None
    if process == "spdc":
        kappa = three_wave_coupling(model, rows).kappa
        coupling = jnp.conj(kappa[0, 1, 2])
    else:
        gamma_nl = kerr_coupling(model, rows).gamma_nl
        coupling = jnp.stack([gamma_nl[1, 0, 0, 2], gamma_nl[2, 0, 0, 1]])
        kerr_phase = jnp.stack(
            [gamma_nl[0, 0, 0, 0], gamma_nl[1, 1, 0, 0], gamma_nl[2, 2, 0, 0]]
        )
    return PhotonPairSource(
        process=process,
        port=pump.port,
        modes=(int(pump.mode), int(signal.mode), int(idler.mode)),
        identical=bool(identical),
        signal_frequency=readonly(np.linspace(bands[1][0], bands[1][1], points)),
        idler_frequency=readonly(np.linspace(bands[2][0], bands[2][1], points)),
        bands=tuple(bands),
        coupling=coupling,
        kerr_phase=kerr_phase,
        fit=tuple(fits),
        fit_residual=jnp.stack(residuals),
    )


def spdc(
    model: Any,
    pump: PortModeResult,
    signal: PortModeResult,
    idler: PortModeResult,
    *,
    points: int = 201,
) -> PhotonPairSource:
    r"""Spontaneous parametric down-conversion of a pump into signal-idler pairs.

    A :math:`\chi^{(2)}` guide (poled or not) turns pump photons into pairs
    with :math:`\omega_p = \omega_s + \omega_i`. ``pump``, ``signal`` and
    ``idler`` are `gfem.PortModeResult`\ s of one mode each on the same port,
    solved on a band of at least 4 frequencies (``np.linspace`` over the band
    of interest); the rows of each arm are fitted by a polynomial, never
    extrapolated. The pump must cover the sums of the signal and idler bands
    for a pulsed figure and the sum of their centres for a continuous one.
    Pass the same result as ``signal`` and ``idler`` (or one mode family over
    overlapping bands) for identical photons: type-0 or type-I degenerate
    emission, counted once.

    The coupling is one `gfem.three_wave_coupling` overlap on the rows nearest
    the centres of the bands, as :math:`\overline{\kappa_{psi}}`; the
    prefactor :math:`2\sqrt{\omega_s\omega_i}/\omega_p` is evaluated on the
    grid. It needs the model's crystal to carry a
    ``second_order_nonlinear_tensor`` with Kleinman symmetry (not checked).
    See `PhotonPairSource` for the amplitude and `gfem.three_wave_coupling` for
    the material, the integral and the gradients (through ``gamma``, the
    traced fields and the materials; not the node coordinates).

    Args:
        model: A model of the mesh the modes were solved on, with the tensor.
        pump: The pump's mode over its band.
        signal: The signal's mode over its band.
        idler: The idler's mode over its band.
        points: Uniform samples over each of the signal and idler bands, the
            grid of every figure; at least 3. The bands are ideal filters.

    Returns:
        The `PhotonPairSource`.

    Raises:
        NotImplementedError: A line port (a 2D mesh).
        ValueError: ``points < 3``; an arm has fewer than 4 rows; the pump
            rows miss the sum of the band centres; otherwise as
            `gfem.three_wave_coupling`.
        TypeError: An arm is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates are traced by `jax.grad`.
    """
    return _build("spdc", model, pump, signal, idler, points=points)


def sfwm(
    model: Any,
    pump: PortModeResult,
    signal: PortModeResult,
    idler: PortModeResult,
    *,
    points: int = 201,
) -> PhotonPairSource:
    r"""Spontaneous four-wave mixing of a degenerate pump into signal-idler pairs.

    A :math:`\chi^{(3)}` guide turns two pump photons into a pair with
    :math:`2\omega_p = \omega_s + \omega_i`. Arguments, grid and refusals as
    `spdc`; the pump must cover the half-sums of the bands for a pulsed figure
    and the half-sum of their centres for a continuous one. The coupling is one
    `gfem.kerr_coupling` overlap on the rows nearest the centres: the pair
    coupling :math:`\Gamma_{si}` from :math:`\gamma_{sppi}` and
    :math:`\gamma_{ipps}` (symmetrized, since the stored tensor evaluates
    :math:`n^2` at the output wave) and the self- and cross-phase entries of
    the continuous-wave mismatch. The methods take no poling arguments.
    Raman scattering, two-photon absorption of the pump beyond the complex
    :math:`\gamma` and free carriers are not modelled.

    Returns:
        The `PhotonPairSource`.

    Raises:
        NotImplementedError: A line port (a 2D mesh).
        ValueError: ``points < 3``; an arm has fewer than 4 rows; the pump
            rows miss the half-sum of the band centres; otherwise as
            `gfem.kerr_coupling`.
        TypeError: An arm is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates are traced by `jax.grad`.
    """
    return _build("sfwm", model, pump, signal, idler, points=points)
