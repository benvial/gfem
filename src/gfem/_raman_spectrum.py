"""`RamanSpectrum`: a bulk Raman gain spectrum, the input of a Raman gain overlap."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced
from gfem._pytree import register_fields

_SYMMETRIES = ("cubic", "isotropic")

# Lin-Agrawal silica response, in seconds: the Blow-Wood oscillator
# (tau_1, tau_2) of the two Raman terms, the boson peak tau_b, and the
# fractions of the three contributions (a, b, c).
_TAU_1, _TAU_2, _TAU_B = 12.2e-15, 32e-15, 96e-15
_FRACTIONS = (0.75, 0.21, 0.04)
# the tabulated range (Hz) and step of the silica spectrum; the response is
# negligible beyond the first
_LA_RANGE, _LA_STEP = 40e12, 10e9
# the Lorentzian grid: nodes at multiples of an eighth of the half width up to
# four half widths from the line, then a geometric tail
_LORENTZ_FINE = 4.0
_LORENTZ_RATIO = 1.25


def _oscillator(omega: np.ndarray) -> np.ndarray:
    r"""The Blow-Wood response :math:`\tilde h_a(\Omega)` (unit area)."""
    a, b = 1 / _TAU_2, 1 / _TAU_1
    return (a * a + b * b) / ((a - 1j * omega) ** 2 + b * b)


def _boson(omega: np.ndarray) -> np.ndarray:
    r"""The boson-peak response :math:`\tilde h_b(\Omega)` (unit area)."""
    a = 1 / _TAU_B
    return (2 / _TAU_B) / (a - 1j * omega) - (1 / _TAU_B**2) / (a - 1j * omega) ** 2


@dataclasses.dataclass(frozen=True, eq=False)
class RamanSpectrum:
    r"""A bulk Raman gain spectrum: the material input of Raman gain.

    The gain :math:`g(\Omega)` is the measured, sign-free power gain of a
    Stokes wave shifted by :math:`\Omega` below a pump, per unit pump
    intensity, in m/W (:math:`\mathrm{d}I_S/\mathrm{d}z = g I_P I_S`), for
    light of the reference ``frequency``. It is linearly interpolated, zero
    beyond the grid, and scales with the Stokes frequency as
    :math:`g(\Omega)\,\omega_S/\omega_\mathrm{ref}`.

    The ``symmetry`` class says how the gain depends on the two polarizations
    and is required: a wrong default changes the gain of a mode with a
    longitudinal field several-fold.

    - ``"cubic"``: crystalline silicon. ``gain`` is the bulk gain of co-polarized
      light along :math:`\langle 110\rangle`, and the Raman tensors are the
      three :math:`R^{x,y,z}` of the :math:`m\bar 3m` class on the
      `Material.axes` of the region.
    - ``"isotropic"``: an amorphous material. ``gain`` is the co-polarized gain
      :math:`g_\parallel` and ``perpendicular_gain`` the cross-polarized gain
      :math:`g_\perp` (zero if not given), both independent of the direction.

    Args:
        shift: The shifts :math:`\Omega/2\pi` in Hz, strictly increasing,
            starting at 0. A fixed array.
        gain: The gain :math:`g_\parallel` at each shift in m/W, :math:`\ge 0`,
            0 at the first shift. It may be traced.
        frequency: The reference frequency in Hz of the Stokes wave, > 0.
        symmetry: ``"cubic"`` or ``"isotropic"``.
        perpendicular_gain: :math:`g_\perp` on the same grid in m/W
            (``"isotropic"`` only), :math:`\ge 0`, 0 at the first shift.

    Raises:
        ValueError: A shift grid that is not 1D, finite, strictly increasing
            and starting at 0; a gain of another length, negative, not finite
            or non-zero at shift 0; a ``frequency`` not > 0; an unknown
            ``symmetry``; a ``perpendicular_gain`` beside ``"cubic"``.
    """

    shift: Any
    gain: Any
    frequency: float
    symmetry: str
    perpendicular_gain: Any = None

    def __post_init__(self) -> None:
        name = "RamanSpectrum"
        shift = np.array(self.shift, dtype=float)
        if shift.ndim != 1 or shift.size < 2 or not np.isfinite(shift).all():
            raise ValueError(f"{name}: shift must be a finite 1D grid of >= 2 values.")
        if shift[0] != 0 or (np.diff(shift) <= 0).any():
            raise ValueError(f"{name}: shift must start at 0 and increase strictly.")
        shift.flags.writeable = False
        frequency = float(self.frequency)
        if not (math.isfinite(frequency) and frequency > 0):
            raise ValueError(f"{name}: frequency must be a finite number > 0 (Hz).")
        if self.symmetry not in _SYMMETRIES:
            raise ValueError(
                f"{name}: symmetry must be one of {_SYMMETRIES}, got "
                f"{self.symmetry!r}; it is required because a default would "
                "change the gain of a mode with a longitudinal field."
            )
        if self.perpendicular_gain is not None and self.symmetry == "cubic":
            raise ValueError(
                f"{name}: perpendicular_gain is for symmetry 'isotropic'; the "
                "cubic class has one gain and fixed selection rules."
            )
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "gain", self._grid(self.gain, "gain", shift))
        if self.perpendicular_gain is not None:
            grid = self._grid(self.perpendicular_gain, "perpendicular_gain", shift)
            object.__setattr__(self, "perpendicular_gain", grid)

    @staticmethod
    def _grid(value: Any, label: str, shift: np.ndarray) -> Any:
        """A gain on the grid: checked when concrete, a traced one passes."""
        name = f"RamanSpectrum: {label}"
        if is_traced(value):
            out = jnp.asarray(value)
            if out.shape != shift.shape:
                raise ValueError(f"{name} must have one value per shift.")
            return out
        arr = np.array(value, dtype=float)
        if arr.shape != shift.shape:
            raise ValueError(f"{name} must have one value per shift.")
        if not np.isfinite(arr).all() or (arr < 0).any():
            raise ValueError(f"{name} must be finite and >= 0 (a gain, in m/W).")
        if arr[0] != 0:
            raise ValueError(f"{name} must be 0 at shift 0.")
        arr.flags.writeable = False
        return arr

    def _key(self) -> tuple:
        def part(a: Any) -> Any:
            if a is None or is_traced(a):
                return id(a)
            arr = np.asarray(a)
            return (arr.shape, arr.tobytes())

        return (
            part(self.shift),
            part(self.gain),
            self.frequency,
            self.symmetry,
            part(self.perpendicular_gain),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RamanSpectrum):
            return NotImplemented
        return self is other or self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        grid = f"{self.shift.size} shifts to {self.shift[-1]:.4g} Hz"
        perp = "" if self.perpendicular_gain is None else ", perpendicular_gain"
        return (
            f"RamanSpectrum({grid}, frequency={self.frequency:.6g} Hz, "
            f"symmetry={self.symmetry!r}{perp})"
        )

    def gain_at(self, shift: Any, frequency: Any) -> jax.Array:
        r"""The co-polarized gain at a shift, scaled to a Stokes frequency.

        Returns :math:`\mathrm{sign}(\Omega)\,g(|\Omega|)\,
        \omega_S/\omega_\mathrm{ref}`: linear interpolation, zero beyond the
        grid, odd in the shift (a wave above the pump loses what a wave
        below gains).

        Args:
            shift: :math:`\Omega/2\pi` in Hz, any shape, either sign.
            frequency: The Stokes frequency in Hz, broadcast with ``shift``.
        """
        return self._at(self.gain, shift, frequency)

    def perpendicular_gain_at(self, shift: Any, frequency: Any) -> jax.Array:
        r"""The cross-polarized gain :math:`g_\perp`, as `gain_at`; 0 if absent."""
        if self.perpendicular_gain is None:
            return jnp.zeros(
                jnp.broadcast_shapes(jnp.shape(shift), jnp.shape(frequency))
            )
        return self._at(self.perpendicular_gain, shift, frequency)

    def _at(self, gain: Any, shift: Any, frequency: Any) -> jax.Array:
        omega = jnp.asarray(shift, dtype=jnp.float64)
        g = jnp.interp(jnp.abs(omega), self.shift, gain, left=0.0, right=0.0)
        scale = jnp.asarray(frequency, dtype=jnp.float64) / self.frequency
        return jnp.sign(omega) * g * scale

    @classmethod
    def lorentzian(
        cls,
        peak_gain: float,
        shift: float,
        width: float,
        frequency: float,
        *,
        symmetry: str,
    ) -> RamanSpectrum:
        r"""A Lorentzian line, the Raman gain of crystalline silicon.

        .. math::

            g(\Omega) = g_\mathrm{peak}
            \frac{(\Gamma/2)^2}{(\Omega/2\pi - f_0)^2 + (\Gamma/2)^2},

        with the line at :math:`f_0` and the full width at half maximum
        :math:`\Gamma`, tabulated on a grid with nodes at the peak and at the
        half-maximum points (so both are exact) and a geometric tail up to
        twice :math:`f_0`. The gain is 0 at shift 0, as every spectrum's is,
        which leaves a small ramp below the line.

        Args:
            peak_gain: The gain at the line centre in m/W, the user's value
                (for silicon the literature scatters from 4 to 76 cm/GW).
            shift: The line centre :math:`f_0` in Hz (15.6 THz for silicon).
            width: The full width at half maximum :math:`\Gamma` in Hz
                (105 GHz for silicon).
            frequency: The reference Stokes frequency of ``peak_gain`` in Hz.
            symmetry: ``"cubic"`` or ``"isotropic"``.

        Raises:
            ValueError: ``peak_gain`` < 0 or not finite, or ``shift`` or
                ``width`` not > 0 with ``width`` < 2 ``shift``.
        """
        name = "RamanSpectrum.lorentzian"
        peak = float(peak_gain)
        f0, w = float(shift), float(width)
        if not (math.isfinite(peak) and peak >= 0):
            raise ValueError(f"{name}: peak_gain must be finite and >= 0 (m/W).")
        if not (math.isfinite(f0) and math.isfinite(w) and 0 < w < 2 * f0):
            raise ValueError(f"{name}: need 0 < width < 2 * shift (Hz).")
        half = w / 2
        fine = np.arange(1, int(8 * _LORENTZ_FINE) + 1) / 8 * half
        tail = fine[-1] * _LORENTZ_RATIO ** np.arange(1, 400)
        offsets = np.concatenate([fine, tail])
        # nodes below the line stay above 0 (the grid's first node)
        below = f0 - offsets[offsets < f0 * 0.999]
        above = f0 + offsets[offsets < f0]
        nodes = np.unique(np.concatenate([[0.0, f0], below, above]))
        gain = peak * half**2 / ((nodes - f0) ** 2 + half**2)
        gain[0] = 0.0
        return cls(nodes, gain, frequency, symmetry)

    @classmethod
    def lin_agrawal(cls, peak_gain: float, frequency: float) -> RamanSpectrum:
        r"""The Raman gain of fused silica, in the Lin-Agrawal shape.

        The isotropic spectrum of their response function (an oscillator
        and a boson peak, fractions 0.75, 0.21 and 0.04): the co-polarized
        gain :math:`g_\parallel \propto \mathrm{Im}[(f_a + f_c)\tilde h_a +
        f_b\tilde h_b]` and the cross-polarized gain
        :math:`g_\perp \propto \mathrm{Im}[(f_b\tilde h_b + f_c\tilde h_a)/2]`,
        on one common scale set by ``peak_gain`` (the maximum of
        :math:`g_\parallel`, at 13.04 THz). Only the shape is the model's; the
        size is the user's, for example :math:`1.2\times10^{-13}` m/W at
        795.5 nm for a model :math:`n_2 = 2.6\times10^{-20}` m²/W and
        :math:`f_R = 0.245`, which scales to :math:`6.18\times10^{-14}` m/W at
        1550 nm. The table ends at 40 THz.

        Args:
            peak_gain: The maximum of :math:`g_\parallel` in m/W at the
                reference ``frequency``.
            frequency: The reference Stokes frequency in Hz.

        Raises:
            ValueError: ``peak_gain`` not finite or <= 0.
        """
        peak = float(peak_gain)
        if not (math.isfinite(peak) and peak > 0):
            raise ValueError("RamanSpectrum.lin_agrawal: peak_gain must be > 0 (m/W).")
        fa, fb, fc = _FRACTIONS
        nodes = np.arange(0.0, _LA_RANGE + _LA_STEP / 2, _LA_STEP)
        omega = 2 * np.pi * nodes
        parallel = ((fa + fc) * _oscillator(omega) + fb * _boson(omega)).imag
        perpendicular = (0.5 * (fb * _boson(omega) + fc * _oscillator(omega))).imag
        parallel[0] = perpendicular[0] = 0.0
        scale = peak / parallel.max()
        return cls(
            nodes, scale * parallel, frequency, "isotropic", scale * perpendicular
        )


register_fields(RamanSpectrum, ("gain", "perpendicular_gain"))
