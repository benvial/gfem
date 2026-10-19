"""Results of `gfem.solve`: frozen pytrees of arrays with static labels."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._checks import is_traced
from gfem._field import Field, FieldMap, ReducedFields
from gfem._pytree import register_fields

if TYPE_CHECKING:
    from gfem._engine.port_mode import PortModeSpace
    from gfem._port_section import PortSection

__all__ = [
    "DrivenResult",
    "ElectrostaticResult",
    "Field",
    "NetworkParameters",
    "PortModeField",
    "PortModeResult",
    "ReducedFields",
    "SweepInfo",
]


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
            differentiable with respect to what the solve reads: a gradient
            through it that reaches ``eps_r`` or the mesh nodes raises
            `gfem.NonDifferentiableError` (`mutual` and `inverse` do
            differentiate).
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

    def __init__(
        self, space: PortModeSpace, basis: np.ndarray, coords: np.ndarray
    ) -> None:
        """Internal: built by `gfem.solve` (field ``i`` is ``basis @ coords[i]``)."""
        self._space = space
        self._basis = basis
        self._coords = coords
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
        columns = self._space.evaluate(self._basis.T, local)
        e = np.einsum("fk,kpc->fpc", self._coords, columns) @ self._section.frame
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
        n = len(self._coords)
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
        residual: Backward error of the mode at each frequency,
            ``|A x - lam B x| / (|A x| + |lam| |B x|)`` on the 2D port mode
            problem. It exceeds ``SolverOptions.port_mode_tolerance`` only at
            the formulation's low-frequency accuracy floor, where it is at
            most 10x that of an exact solve.
        solved: ``(n_freq,)`` bool, True where the mode is an exact port mode
            solve; elsewhere it comes from the reduced port mode model.
        e_field: The mode's electric field, normalized to 1 W.
    """

    port: str
    mode: int
    frequency: jax.Array
    gamma: jax.Array
    Z_PV: jax.Array
    residual: jax.Array
    solved: jax.Array
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
        exact = int(np.asarray(self.solved).sum())
        return (
            f"PortModeResult(port {self.port!r}, mode {self.mode}, {len(f)} "
            f"frequencies {span} ({exact} exact solves), n_eff "
            f"{n.real.min():.5g}-{n.real.max():.5g}, {zs})"
        )


register_fields(PortModeResult, ("frequency", "gamma", "Z_PV", "residual", "solved"))


@dataclasses.dataclass(frozen=True, eq=False)
class NetworkParameters:
    """Scattering parameters over frequency, labelled by port name.

    Power waves (Kurokawa) under e^{+iωt}, each port with its own reference
    impedance ``z_ref``: ``values[f, i, j]`` is the outgoing wave at port
    ``ports[i]`` when port ``excited[j]`` is fed 1 W incident power and
    every other port is terminated in its reference impedance. For a
    `LumpedPort` the reference impedance is its resistance; an inductance or
    capacitance of the port loads the network in parallel. A port without
    a resistance has no reference impedance: its ``z_ref`` and its row of S
    are NaN. `y`, `z` and `renormalize` are `jax.numpy` functions of S and
    ``z_ref``, so they differentiate like S; `to_touchstone` writes a file.
    The conventions are spelled out in `gfem.conventions`.

    Attributes:
        values: S, complex, shape ``(n_freq, n_ports, n_excited)``.
        z_ref: Reference impedance of each port in ohm, shape
            ``(n_freq, n_ports)``.
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
        """``20 log10 |S[to, from]|`` over frequency, dB."""
        return 20 * jnp.log10(jnp.abs(self[to, source]))

    def y(self) -> jax.Array:
        """The admittance matrix in siemens, ``(n_freq, n_ports, n_ports)``.

        Rows and columns in `ports` order. With ``F = diag(1 / (2 sqrt(Re
        z_ref)))`` and ``G = diag(z_ref)``, ``Y = F^-1 (S G + G*)^-1 (I - S) F``
        (power waves). Written in `jax.numpy`, so it differentiates like S.
        Entries are NaN where a port has no reference impedance.

        Raises:
            ValueError: Not every port was excited, so S is not square.
        """
        s, g, ratio = self._s_g_ratio("y()")
        eye = jnp.broadcast_to(jnp.eye(s.shape[-1]), s.shape)
        sg = s * g[:, None, :] + _diag(jnp.conj(g))
        return jnp.linalg.solve(sg, eye - s) * ratio

    def z(self) -> jax.Array:
        """The impedance matrix in ohm, ``(n_freq, n_ports, n_ports)``.

        Rows and columns in `ports` order. With ``F = diag(1 / (2 sqrt(Re
        z_ref)))`` and ``G = diag(z_ref)``, ``Z = F^-1 (I - S)^-1 (S G + G*) F``
        (power waves). Written in `jax.numpy`, so it differentiates like S.
        Entries are NaN where a port has no reference impedance.

        Raises:
            ValueError: Not every port was excited, so S is not square.
        """
        s, g, ratio = self._s_g_ratio("z()")
        eye = jnp.broadcast_to(jnp.eye(s.shape[-1]), s.shape)
        sg = s * g[:, None, :] + _diag(jnp.conj(g))
        return jnp.linalg.solve(eye - s, sg) * ratio

    def renormalize(self, z_new: Any) -> NetworkParameters:
        """The same network under new reference impedances.

        Power-wave renormalization: the network (its Z matrix) is unchanged,
        only the waves that describe it change. Written in `jax.numpy`, so
        it differentiates with respect to S and ``z_new``.

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
                    "Power-wave reference impedances need a positive real part "
                    f"(a resistance), got {z_new!r}."
                )
        new = jnp.broadcast_to(jnp.asarray(z_new), (n_f, n))
        z = jnp.asarray(self.z_ref, dtype=complex)
        zn = new.astype(complex)
        # a' = P a + Q b and b' = U a + T b per port, so S' = (U + T S)(P + Q S)^-1,
        # each side scaled by 1 / (2 sqrt(Re Z Re Z')).
        p = jnp.conj(z) + zn
        q = z - zn
        u = jnp.conj(z) - jnp.conj(zn)
        t = z + jnp.conj(zn)
        num = _diag(u) + t[:, :, None] * s
        den = _diag(p) + q[:, :, None] * s
        s0 = jnp.swapaxes(
            jnp.linalg.solve(jnp.swapaxes(den, 1, 2), jnp.swapaxes(num, 1, 2)), 1, 2
        )
        d = jnp.sqrt(jnp.real(z) * jnp.real(zn))
        return NetworkParameters(
            values=s0 * d[:, None, :] / d[:, :, None],
            z_ref=new,
            frequency=self.frequency,
            ports=self.ports,
            excited=self.ports,
        )

    def to_touchstone(self, path: str | Path, z0: float | None = None) -> Path:
        """Write S to a Touchstone 1.1 file (``.sNp``, real/imaginary, Hz).

        A Touchstone 1.1 file has one real reference impedance for every
        port. Without ``z0`` the file is labelled with `z_ref`, which must
        then be one real value for every port and frequency: gfem never
        labels a port with an impedance S was not computed for. With
        ``z0``, S is first renormalized to it (`renormalize`). Port names
        and the conventions are written as comments.

        Args:
            path: The file, named ``*.s<n_ports>p``.
            z0: The reference impedance to renormalize to, ohm (real, > 0).

        Returns:
            The path written.

        Raises:
            ValueError: The file name does not match the port count, not
                every port was excited, a port has no reference impedance,
                or `z_ref` is not one real value and ``z0`` is not given.
            TypeError: The values are traced by JAX.
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
        if missing:
            raise ValueError(
                f"Port {', '.join(map(repr, missing))} has no reference impedance "
                "(z_ref is NaN): a LumpedPort without a resistance, or a WavePort "
                "without a voltage_path. S cannot be labelled or renormalized "
                "without one; give the port a resistance or a voltage_path."
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
            z0 = float(z0)
            s = np.asarray(self.renormalize(z0).values)
        path.write_text(_touchstone(self, s, z0))
        return path

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
        r = jnp.sqrt(jnp.real(g))
        # F^-1 M F: entry (i, j) times F_j / F_i = sqrt(Re z_i / Re z_j)
        return s, g, r[:, :, None] / r[:, None, :]

    def __repr__(self) -> str:
        return (
            f"NetworkParameters(ports={list(self.ports)}, "
            f"excited={list(self.excited)}, {len(self.frequency)} frequencies)"
        )


register_fields(NetworkParameters, ("values", "z_ref"))


def _diag(v: jax.Array) -> jax.Array:
    """``(n_freq, n)`` -> stacked diagonal matrices ``(n_freq, n, n)``."""
    return v[:, :, None] * jnp.eye(v.shape[-1])


def _touchstone(net: NetworkParameters, s: np.ndarray, z0: float) -> str:
    """Touchstone 1.1 text of ``s`` (port order) with reference ``z0``."""
    n = len(net.ports)
    lines = [
        f"! gfem {version('gfem')}: S-parameters, {n} port(s), {len(s)} frequencies",
        "! Power waves (Kurokawa), time convention exp(+j omega t); |a|^2 is the",
        "! time-averaged incident power in W",
        *(f"! Port[{i}] = {name}" for i, name in enumerate(net.ports, start=1)),
        f"# Hz S RI R {z0:.12g}",
    ]

    def pairs(values: np.ndarray) -> list[str]:
        return [f"{v.real: .16e} {v.imag: .16e}" for v in values]

    for f, m in zip(net.frequency, s, strict=True):
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
            ports); ``inf`` where there was no model to compare with yet
            (the two band edges the greedy starts from, the first frozen
            sample, and every frequency of a uniform sweep).
        converged: Whether the last ``SolverOptions.adaptive_memory`` sample
            errors are below the tolerance (or every frequency was solved).
            Always True for a uniform sweep. An adaptive greedy that stops at
            ``adaptive_max_samples`` instead warns with
            `gfem.SweepNotConvergedWarning`.
        tolerance: ``SolverOptions.adaptive_tolerance`` of an adaptive
            sweep, ``None`` for a uniform one.
        basis_size: The number of reduced basis vectors (0 for uniform).
    """

    kind: str
    solved: tuple[bool, ...]
    sampled_frequencies: tuple[float, ...]
    sample_errors: tuple[float, ...]
    converged: bool
    tolerance: float | None = None
    basis_size: int = 0

    def __post_init__(self) -> None:
        """Normalize to plain Python values (hashable, JSON-friendly)."""
        set_ = object.__setattr__
        set_(self, "solved", tuple(bool(v) for v in self.solved))
        set_(
            self,
            "sampled_frequencies",
            tuple(float(f) for f in self.sampled_frequencies),
        )
        set_(self, "sample_errors", tuple(float(e) for e in self.sample_errors))
        set_(self, "converged", bool(self.converged))
        set_(self, "basis_size", int(self.basis_size))
        if self.tolerance is not None:
            set_(self, "tolerance", float(self.tolerance))

    @classmethod
    def uniform(cls, frequency: tuple[float, ...]) -> SweepInfo:
        """Every frequency of ``frequency`` solved in full."""
        n = len(frequency)
        return cls(
            kind="uniform",
            solved=(True,) * n,
            sampled_frequencies=frequency,
            sample_errors=(math.inf,) * n,
            converged=True,
        )

    def __repr__(self) -> str:
        if self.kind == "uniform":
            return f"SweepInfo(uniform, {len(self.solved)} frequencies solved)"
        state = "converged" if self.converged else "NOT converged"
        return (
            f"SweepInfo(adaptive, {len(self.sampled_frequencies)} of "
            f"{len(self.solved)} frequencies solved, basis {self.basis_size}, "
            f"{state} at tolerance {self.tolerance:g})"
        )


@dataclasses.dataclass(frozen=True, eq=False)
class DrivenResult:
    """The result of a `Driven` problem.

    Attributes:
        s: The scattering parameters (`NetworkParameters`).
        e_fields: The kept electric fields (``Driven(keep_fields=...)``), one
            `Field` per kept frequency and excited port, labelled by
            `kept`; empty by default. Read them with `e_field`. Each is the
            peak phasor (e^{+iωt}) in V/m for 1 W time-averaged incident
            power at the excited port, every other port terminated in its
            reference impedance ``z_ref``, so time-averaged densities carry
            a factor 1/2 (see `gfem.conventions`). Quantity ``"E"``, on
            H(curl) of the solve order, over the solve domain.
            An adaptive sweep's fields are basis-backed (`ReducedFields`):
            the result holds the reduced basis and one short coordinate
            vector per field, and a field's coefficients are formed when it
            is read. Their accuracy is the sweep's field error, about
            ``SolverOptions.adaptive_tolerance`` or better.
        kept: ``(frequency in Hz, excited port)`` of each of `e_fields`
            (static labels), frequency-major.
        sweep: How the frequencies were solved (`SweepInfo`).
    """

    s: NetworkParameters
    e_fields: Sequence[Field] = ()
    kept: tuple[tuple[float, str], ...] = ()
    sweep: SweepInfo | None = None

    def __post_init__(self) -> None:
        """Hold the fields as a sequence matching their labels."""
        if not isinstance(self.e_fields, ReducedFields):
            object.__setattr__(self, "e_fields", tuple(self.e_fields))
        if self.sweep is None:
            object.__setattr__(self, "sweep", SweepInfo.uniform(self.s.frequency))
        object.__setattr__(
            self, "kept", tuple((float(f), str(p)) for f, p in self.kept)
        )
        if len(self.e_fields) != len(self.kept):
            raise ValueError(
                f"{len(self.e_fields)} fields but {len(self.kept)} labels in kept."
            )

    @property
    def frequency(self) -> np.ndarray:
        """The frequencies in Hz, ``(n_freq,)``."""
        return np.array(self.s.frequency)

    def e_field(self, frequency: float, excite: str | None = None) -> Field:
        """The kept electric field at ``frequency`` with port ``excite`` fed.

        Args:
            frequency: A kept frequency, Hz (matched to 1e-9 relative).
            excite: The excited port; may be left out when one port was
                excited.

        Raises:
            ValueError: No field was kept at ``frequency``, or ``excite`` is
                needed.
            KeyError: ``excite`` is not an excited port.
        """
        if not self.kept:
            raise ValueError(
                "This result keeps no fields: solve with Driven(..., "
                "keep_fields=True) or keep_fields=[frequencies] to keep them."
            )
        if excite is None:
            if len(self.s.excited) > 1:
                raise ValueError(
                    f"Ports {', '.join(map(repr, self.s.excited))} were excited; "
                    "pass excite= to choose one."
                )
            excite = self.s.excited[0]
        if excite not in self.s.excited:
            raise KeyError(
                f"Port {excite!r} was not excited; the excited ports are "
                f"{', '.join(map(repr, self.s.excited))}."
            )
        frequencies = sorted({f for f, _ in self.kept})
        for k, (f, port) in enumerate(self.kept):
            if port == excite and math.isclose(f, frequency, rel_tol=1e-9):
                return self.e_fields[k]
        raise ValueError(
            f"No field kept at {frequency:g} Hz; the kept frequencies are "
            f"{', '.join(f'{f:g}' for f in frequencies)} Hz (Driven.keep_fields)."
        )

    def __repr__(self) -> str:
        f = self.frequency
        span = (
            f"{f.min() / 1e9:.4g}-{f.max() / 1e9:.4g} GHz"
            if len(f) > 1
            else f"{f[0] / 1e9:.4g} GHz"
        )
        fields = f", {len(self.e_fields)} kept fields" if self.e_fields else ""
        sweep = "" if self.sweep is None else f", {self.sweep!r}"
        return (
            f"DrivenResult({len(f)} frequencies {span}, ports={list(self.s.ports)}, "
            f"excited={list(self.s.excited)}{fields}{sweep})"
        )


register_fields(DrivenResult, ("s", "e_fields"))
