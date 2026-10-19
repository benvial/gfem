"""Results of `gfem.solve`: frozen pytrees of arrays with static labels."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._field import Field, FieldMap
from gfem._pytree import register_fields

if TYPE_CHECKING:
    from gfem._engine.port_mode import PortModeSpace
    from gfem._port_section import PortSection

__all__ = ["ElectrostaticResult", "Field", "PortModeField", "PortModeResult"]


@dataclasses.dataclass(frozen=True, eq=False)
class ElectrostaticResult:
    """The result of an `Electrostatic` problem.

    Excitation ``i`` puts terminal ``i`` at 1 V and every other terminal
    and the ground at 0 V.

    Attributes:
        capacitance: The Maxwell capacitance matrix in farads, shape
            ``(n_terminals, n_terminals)``, rows and columns in `terminals`
            order: ``capacitance[i, j]`` is the charge on terminal ``i`` in
            excitation ``j``. The diagonal is positive and the off-diagonal
            entries are negative or zero (Palace's ``terminal-C.csv``
            convention).
        terminals: The terminal names (static labels).
        energy: Electric energy in joules stored in each volume region of
            the solve domain, by region name: an ``(n_terminals,)`` array,
            entry ``i`` for excitation ``i`` (Palace's ``domain-E.csv`` at
            1 V). Excluded volumes are left out. Summed over the regions,
            excitation ``i`` stores ``capacitance[i, i] / 2``. Not
            differentiable: a gradient through it raises
            `gfem.NonDifferentiableError` (`mutual` and `inverse` are).
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

    def __post_init__(self) -> None:
        """Hold the potentials as a read-only, ordered mapping."""
        if not isinstance(self.potential, FieldMap):
            fields = FieldMap(self.potential, label="terminal")
            object.__setattr__(self, "potential", fields)

    @property
    def mutual(self) -> jax.Array:
        """The mutual capacitance matrix in farads (Palace ``terminal-Cm.csv``).

        Off the diagonal, ``mutual[i, j] = -capacitance[i, j]``, the
        capacitance between terminals ``i`` and ``j``; on it, the row sum
        of `capacitance`, the capacitance of terminal ``i`` to ground.
        """
        c = self.capacitance
        return jnp.diag(c.sum(axis=1)) - (c - jnp.diag(jnp.diag(c)))

    @property
    def inverse(self) -> jax.Array:
        """The inverse of `capacitance` in 1/F (Palace ``terminal-Cinv.csv``).

        Entry ``[i, j]`` is the voltage of terminal ``i`` per coulomb on
        terminal ``j``, the others carrying no charge. Without a ground the
        rows of `capacitance` sum to zero and it has no inverse.
        """
        return jnp.linalg.inv(self.capacitance)

    def index(self, terminal: str) -> int:
        """Row / column of ``terminal`` in `capacitance`.

        Raises:
            KeyError: ``terminal`` is not a terminal of this result.
        """
        try:
            return self.terminals.index(terminal)
        except ValueError:
            raise KeyError(
                f"No terminal {terminal!r}; the terminals are "
                f"{', '.join(repr(t) for t in self.terminals)}."
            ) from None

    def __repr__(self) -> str:
        try:
            c = np.asarray(self.capacitance) * 1e15
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"ElectrostaticResult(terminals={list(self.terminals)})"
        width = max(len(t) for t in self.terminals)
        rows = [
            f"  {t:>{width}}: " + "  ".join(f"{v:10.4g}" for v in row)
            for t, row in zip(self.terminals, c, strict=True)
        ]
        return (
            f"ElectrostaticResult(terminals={list(self.terminals)}, "
            "capacitance in fF:\n" + "\n".join(rows) + ")"
        )


register_fields(ElectrostaticResult, ("capacitance", "energy", "potential"))


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
        k0 = 2 * math.pi * self.frequency / scipy.constants.c
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
