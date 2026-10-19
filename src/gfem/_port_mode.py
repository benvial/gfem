"""The port mode problem type and its result.

Re-exported by `gfem.problems` (`PortMode`) and `gfem.results`
(`PortModeResult`, `PortModeField`).
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced
from gfem._pytree import register_fields

if TYPE_CHECKING:
    from gfem._engine.port_mode import PortModeSpace
    from gfem._port_section import PortSection

C0 = 299792458.0  # speed of light in vacuum, m/s


def _frequencies(value: Any) -> tuple[float, ...]:
    if is_traced(value):
        raise ValueError("PortMode.frequency must be fixed numbers, not traced.")
    try:
        f = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        f = np.empty((0, 0))
    if f.ndim > 1 or f.size == 0:
        raise ValueError(
            "PortMode.frequency must be one frequency or a 1D sequence of them, in "
            f"Hz; got {value!r}."
        )
    f = f.reshape(-1)
    if not np.isfinite(f).all() or (f <= 0).any():
        raise ValueError(
            f"PortMode.frequency must be finite and > 0 Hz, got {value!r}."
        )
    return tuple(float(x) for x in f)


@dataclasses.dataclass(frozen=True)
class PortMode:
    """The 2D port mode problem of a wave port over a frequency grid.

    Solves the port's cross-section for its propagating mode at each
    frequency: propagation constant, effective index, mode field and the
    power-voltage impedance. The cross-section's materials are those of the
    volume elements behind the port; its edges take the boundary conditions
    of the surfaces that meet the port (`PEC` edges are perfect conductors,
    `Absorbing` edges first-order absorbing, all others natural, i.e. PMC).

    Attributes:
        port: The name of a `WavePort` of the model.
        frequency: One frequency or a 1D sequence of them, Hz.
        mode: Which propagating mode, in Palace's order: by decreasing
            effective index for lossless modes (1 = the most confined), with
            strongly attenuated modes ranked lower. ``None``: the wave port's
            own ``mode``.
    """

    port: str
    frequency: float | tuple[float, ...]
    mode: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.port, str):
            raise TypeError(
                f"PortMode.port must be the name of a WavePort, got {self.port!r}."
            )
        object.__setattr__(self, "frequency", _frequencies(self.frequency))
        if self.mode is not None and (
            isinstance(self.mode, bool)
            or not isinstance(self.mode, (int, np.integer))
            or self.mode < 1
        ):
            raise ValueError(
                f"PortMode.mode must be an integer >= 1 or None, got {self.mode!r}."
            )
        if self.mode is not None:
            object.__setattr__(self, "mode", int(self.mode))


jax.tree_util.register_static(PortMode)


class PortModeField:
    """The electric field of a port mode on its cross-section, over frequency.

    Peak phasor of the mode carrying 1 W time-averaged power through the
    port, propagating into the solve domain as ``exp(-gamma s)``, ``s`` the
    distance along the inward port normal. Fields follow e^{+iωt}.

    Attributes:
        quantity: ``"E"``.
        unit: ``"V/m"``.
        normal: Unit inward port normal, global frame.
    """

    quantity = "E"
    unit = "V/m"

    def __init__(self, space: PortModeSpace, coeffs: np.ndarray) -> None:
        """Internal: built by `gfem.solve`."""
        self._space = space
        self._coeffs = coeffs
        self.normal = tuple(float(x) for x in space.section.frame[2])

    @property
    def _section(self) -> PortSection:
        return self._space.section

    def at(self, points: Any) -> jax.Array:
        """The field at points on the port, ``(n_freq, n_points, 3)`` complex.

        Points off the cross-section's elements (inside metal holes or outside
        the port) give zero.

        Args:
            points: ``(n_points, 3)`` (or one ``(3,)``) global coordinates, m,
                on the port plane.

        Raises:
            ValueError: A point is not on the port plane.
        """
        p = np.asarray(points, float)
        single = p.ndim == 1
        p = np.atleast_2d(p)
        if p.shape[-1] != 3:
            raise ValueError(f"points must be 3-vectors in m, got shape {p.shape}.")
        local = self._local(p)
        e = self._space.evaluate(self._coeffs, local) @ self._section.frame
        return jnp.asarray(e[:, 0] if single else e)

    def _local(self, p: np.ndarray) -> np.ndarray:
        s = self._section
        local, dist = s.to_local(p)
        tol = 1e-5 * s.diameter
        if (np.abs(dist) > tol).any():
            worst = p[int(np.argmax(np.abs(dist)))]
            raise ValueError(
                f"Point {tuple(float(x) for x in worst)} (m) is not on the plane of "
                f"wave port {s.port.name!r}."
            )
        return local

    def __repr__(self) -> str:
        n = len(self._coeffs)
        return (
            f"PortModeField(E, V/m, {n} frequencies, port {self._section.port.name!r})"
        )


@dataclasses.dataclass(frozen=True, eq=False)
class PortModeResult:
    """The mode of a wave port over frequency. Not differentiable.

    Fields follow e^{+iωt}; a mode propagates into the solve domain as
    ``exp(-gamma s)``.

    Attributes:
        port: The wave port's name.
        mode: Which mode (1 = the most propagating).
        frequency: ``(n_freq,)`` Hz.
        gamma: Propagation constant ``alpha + i beta``, ``(n_freq,)``, 1/m
            (``alpha >= 0`` is the attenuation).
        Z_PV: Power-voltage characteristic impedance ``|V|^2 / (2 P)`` along
            the port's ``voltage_path``, ohm; NaN without a voltage path.
        residual: Backward error of each port mode solve,
            ``|A x - lam B x| / (|A x| + |lam| |B x|)``.
        e_field: The mode's electric field, normalized to 1 W.
    """

    port: str
    mode: int
    frequency: jax.Array
    gamma: jax.Array
    Z_PV: jax.Array
    residual: jax.Array
    e_field: PortModeField

    @property
    def n_eff(self) -> jax.Array:
        """Complex effective index ``gamma / (i k0)``: ``n' - i n''``."""
        k0 = 2 * math.pi * self.frequency / C0
        return self.gamma / (1j * k0)

    def __repr__(self) -> str:
        f = np.asarray(self.frequency)
        n = np.asarray(self.n_eff)
        z = np.asarray(self.Z_PV)
        span = (
            f"{f.min() / 1e9:.4g}-{f.max() / 1e9:.4g} GHz"
            if len(f) > 1
            else (f"{f[0] / 1e9:.4g} GHz")
        )
        zs = (
            "no voltage_path"
            if np.isnan(z).all()
            else (f"Z_PV {z.min():.4g}-{z.max():.4g} ohm")
        )
        return (
            f"PortModeResult(port {self.port!r}, mode {self.mode}, {len(f)} "
            f"frequencies {span}, n_eff {n.real.min():.5g}-{n.real.max():.5g}, {zs})"
        )


register_fields(PortModeResult, ("frequency", "gamma", "Z_PV", "residual"))
