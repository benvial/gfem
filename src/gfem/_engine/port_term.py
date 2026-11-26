r"""A wave port surface's term at a complex frequency, for an eigenmode problem.

An eigenmode problem with wave ports evaluates each port surface's two terms
(`gfem._engine.wave_ports`) at complex wavenumbers :math:`k`: the Robin term
:math:`\gamma_0(k)\int (A E_t)\cdot v_t` and the modal correction
:math:`W(k) = U C(k) U^\mathsf{T}`. Both come from the Rayleigh-Ritz model of
a real port mode sweep over a band around the target, whose basis is fixed,
so :math:`U` is fixed and only the small :math:`C(k)` moves:

- the band is :math:`[\max(1.05 f_c, 0.7 f_t),\ 1.4 f_t]`, 9 points, with
  :math:`f_t` the target and :math:`f_c` the highest cutoff of the surface's
  requested modes (a narrower band misses roots off the target, and the
  lower bound keeps every requested mode above its cutoff);
- at a complex :math:`k` the reduced pencil gives each requested mode's
  :math:`\lambda = \gamma^2` and coordinates :math:`y`, tracked by overlap
  with the mode at the band frequency nearest :math:`\mathrm{Re}\,k` (on a
  band frequency itself, the sweep's tabulated mode, so the term is the
  driven coupling's there); their
  :math:`k`-derivatives come from the bordered dense solve
  :math:`\begin{bmatrix} A - \lambda B & -By \\ y^\mathsf{T} & 0
  \end{bmatrix} \begin{bmatrix} y' \\ \lambda' \end{bmatrix} =
  \begin{bmatrix} -(A' - \lambda B') y \\ 0 \end{bmatrix}`, with the
  pencil's slopes in :math:`k_0` (laws and conductors included);
- :math:`C = P D P^\mathsf{T}` as on the real grid
  (`gfem._engine.wave_ports.PortSurfaceCoupling.correction`), and
  :math:`C'` by the product rule;
- the propagation constant is the outgoing branch with its cut rotated onto
  the negative imaginary axis, :math:`\gamma = i\,e^{i\pi/4}\sqrt{i\lambda}`:
  above cutoff at a decaying :math:`k` it is the principal outgoing root
  :math:`i\sqrt{-\lambda}`, and below cutoff it stays continuous across the
  real axis, where the principal root's cut lies;
- the guard is each reduced pair's exact backward error, which passes at the
  sweep's own floor at the nearest band frequency (``port_mode_tolerance``,
  or 10 times the exact solve's error there). When it fails, an exact 2D
  solve at the complex wavenumber enriches the basis: :math:`U` and the
  shape of :math:`C` change, and `PortTerm.version` counts the change.

Wavenumbers are per mesh file unit, which the port section shares.
"""

from __future__ import annotations

import cmath
import dataclasses
import math
from collections.abc import Callable

import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.linalg

from gfem._boundaries import WavePort, lead_port, wave_port_label, wave_port_surfaces
from gfem._engine import terms
from gfem._engine.line_mode import mode_space
from gfem._engine.mesh import EngineMesh
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.wave_ports import face_vectors
from gfem._linalg.superlu import shift_invert_eigs
from gfem._pml import Stretch
from gfem._port_section import port_section
from gfem._sweep.port_mode import (
    _AMBIGUOUS,
    PortModeSweep,
    _overlaps,
    _Reduced,
    _wavenumbers,
    planar,
    port_mode_sweep,
    propagating_modes,
)
from gfem._text import complex_hertz, hertz
from gfem.errors import PortModeError
from gfem.model import Model

__all__ = [
    "PortTerm",
    "PortTermRefusalError",
    "PortTermValue",
    "port_terms",
    "sweep_band",
]

# The band [max(_CUTOFF f_c, _LOW f_t), _HIGH f_t] with _POINTS frequencies.
_CUTOFF, _LOW, _HIGH, _POINTS = 1.05, 0.7, 1.4, 9
# The cutoff is bracketed to this relative width.
_CUTOFF_RTOL = 1e-3
# Doublings of the target that bracket a cutoff above it, for its error.
_DOUBLINGS = 6
# The width of that cutoff's estimate.
_ESTIMATE_RTOL = 1e-2
# The width of a cutoff that classifies a root below it (`PortTerm.cutoff`).
_BELOW_RTOL = 1e-6
# Exact solves that may enrich the basis at one wavenumber.
_ENRICHMENTS = 3
# Pairs asked of an exact solve at a complex wavenumber.
_NEV = 6
# Values kept per surface (a refinement revisits a few iterates).
_CACHE = 64
# exp(i pi / 4) of the outgoing branch with its cut rotated.
_ROTATION = cmath.exp(0.25j * math.pi)
# The step of the power's k-difference, relative to |k|.
_POWER_STEP = 1e-3


def outgoing(lam: complex) -> complex:
    r"""The outgoing branch :math:`\gamma = i\,e^{i\pi/4}\sqrt{i\lambda}`.

    The square root's cut lies on :math:`\lambda \in i\,\mathbb{R}^+`,
    :math:`\beta^2 = -\lambda = \varepsilon\mu k_0^2 - k_c^2` on the negative
    imaginary axis.
    """
    return 1j * _ROTATION * cmath.sqrt(1j * lam)


def sweep_band(
    modes: PortModeSpace, target: float, mode: int, label: str
) -> np.ndarray:
    r"""The port mode sweep's band of an eigenmode solve, Hz.

    :math:`[\max(1.05 f_c, 0.7 f_t),\ 1.4 f_t]` with 9 points, :math:`f_c`
    the cutoff of mode ``mode`` in rank order, bracketed by exact solves only
    when it can reach the band.

    Args:
        modes: The surface's port mode problem.
        target: The eigenmode target :math:`f_t`, Hz.
        mode: The surface's highest requested mode.
        label: The surface's wave ports, for messages.

    Raises:
        PortModeError: Mode ``mode`` does not propagate at the target.
    """
    if propagating_modes(modes, target, mode) < mode:
        raise _below_cutoff(modes, target, mode, label)
    lo, hi = _LOW / _CUTOFF * target, target
    if propagating_modes(modes, lo, mode) >= mode:
        start = _LOW * target
    else:
        start = _CUTOFF * _cutoff(modes, lo, hi, mode, _CUTOFF_RTOL)
    return np.linspace(start, _HIGH * target, _POINTS)


def _cutoff(
    modes: PortModeSpace, lo: float, hi: float, mode: int, rtol: float
) -> float:
    """The cutoff of mode ``mode`` bisected in ``[lo, hi]`` to ``rtol``, Hz.

    Mode ``mode`` does not propagate at ``lo`` and does at ``hi``.
    """
    while hi - lo > rtol * hi:
        mid = 0.5 * (lo + hi)
        if propagating_modes(modes, mid, mode) >= mode:
            hi = mid
        else:
            lo = mid
    return hi


def _cutoff_below(modes: PortModeSpace, hi: float, mode: int) -> float:
    """The cutoff of mode ``mode``, which propagates at ``hi``, Hz.

    Bracketed below ``hi`` by halving, then bisected to `_BELOW_RTOL`; 0
    when the mode still propagates after `_DOUBLINGS` halvings (a TEM mode).
    """
    for _ in range(_DOUBLINGS):
        lo = hi / 2
        if propagating_modes(modes, lo, mode) < mode:
            return _cutoff(modes, lo, hi, mode, _BELOW_RTOL)
        hi = lo
    return 0.0


def _below_cutoff(
    modes: PortModeSpace, target: float, mode: int, label: str
) -> PortModeError:
    """The error of a target below mode ``mode``'s cutoff, naming that cutoff.

    The cutoff is bracketed above the target by doubling, then bisected to 1 %.
    """
    lo = hi = target
    for _ in range(_DOUBLINGS):
        lo, hi = hi, 2 * hi
        if propagating_modes(modes, hi, mode) >= mode:
            cutoff = _cutoff(modes, lo, hi, mode, _ESTIMATE_RTOL)
            where = f"its cutoff is near {hertz(cutoff, 3)}"
            break
    else:
        where = f"it does not propagate up to {hertz(hi)}"
    return PortModeError(
        f"{label}: mode {mode} does not propagate at the eigenmode target "
        f"{hertz(target)} ({where}), so the port has no outgoing wave there. "
        "Raise the target above the cutoff, pick a lower mode, or find the "
        "response from a gfem.Driven sweep."
    )


class PortTermRefusalError(PortModeError):
    """The port term cannot evaluate at an iterate of a mode refinement.

    Internal: the eigen search catches it and loses the seed (a mode
    refinement may wander to a complex frequency where the port's mode cannot
    be followed), so it never reaches a caller. The public `PortModeError`
    stays for a target below a requested mode's cutoff.

    Attributes:
        label: The surface's wave ports, for messages.
        k: The complex wavenumber (per file unit) of the iterate.
    """

    def __init__(self, label: str, k: complex, why: str) -> None:
        super().__init__(why)
        self.label = label
        self.k = k


@dataclasses.dataclass(frozen=True)
class PortTermValue:
    r"""A surface's port term at one complex wavenumber :math:`k`.

    Attributes:
        k: The wavenumber, per file unit.
        gamma: ``(n,)`` each wave port's propagation constant on the outgoing
            branch, per file unit, in the surface's port order.
        d_gamma: ``(n,)`` :math:`d\gamma/dk`.
        lead: The position of the lead port (its ``gamma`` is
            :math:`\gamma_0`).
        c: ``(2 m, 2 m)`` :math:`C(k)`, complex symmetric, with :math:`W =
            U C U^\mathsf{T}` on the `PortTerm.vectors` of ``version``.
        d_c: :math:`dC/dk`, likewise.
        error: The largest backward error of the reduced pairs.
        version: The basis this value stands on (`PortTerm.version`).
        y: ``(m, n)`` each port's mode as reduced pencil coordinates
            :math:`y`, whose field coordinates are :math:`(y/\gamma, y)`.
        d_y: :math:`dy/dk`, with :math:`y^\mathsf{T} y' = 0`.
    """

    k: complex
    gamma: np.ndarray
    d_gamma: np.ndarray
    lead: int
    c: np.ndarray
    d_c: np.ndarray
    error: float
    version: int
    y: np.ndarray
    d_y: np.ndarray

    @property
    def gamma0(self) -> complex:
        r""":math:`\gamma_0(k)`, the Robin term's scalar."""
        return complex(self.gamma[self.lead])

    @property
    def d_gamma0(self) -> complex:
        r""":math:`\gamma_0'(k)`."""
        return complex(self.d_gamma[self.lead])


class PortTerm:
    """One wave port surface's Robin term and modal correction at complex k.

    Built by `port_terms`. `at` evaluates; `vectors` is :math:`U`, fixed
    until an enrichment (`version` counts them); `robin_piece` is the Robin
    term as a nonpolynomial driven piece.

    Attributes:
        ports: The surface's wave ports, in model port order.
        modes: The surface's port mode problem.
        sweeps: Each port's real sweep over ``band``.
        band: The sweep's frequencies, Hz.
        tolerance: ``port_mode_tolerance``.
        version: How many times an exact solve has enriched the basis.
        enrichments: The wavenumbers at which it did, in order.
    """

    def __init__(
        self,
        ports: tuple[WavePort, ...],
        modes: PortModeSpace,
        sweeps: list[PortModeSweep],
        *,
        tolerance: float,
        lift: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Hold the sweep's reduced model; ``lift`` maps 2D columns to 3D."""
        self.ports = ports
        self.modes = modes
        self.sweeps = sweeps
        self.band = sweeps[0].frequency
        self.tolerance = tolerance
        self.version = 0
        self.enrichments: list[complex] = []
        self._lift = lift
        self._lead = list(ports).index(lead_port(list(ports)))
        basis = sweeps[0].basis
        m = basis.shape[1] // 2
        w = basis[modes.free]
        # the field basis splits the pencil basis by rows (`SectionReduction`)
        self._reduced = _Reduced.on(modes, w[:, :m] + w[:, m:])
        grid = [_wavenumbers(modes.section, f) for f in self.band]
        self._k = np.array([g[0] for g in grid])
        self._k_max = np.array([g[1] for g in grid])
        self._cache: dict[complex, PortTermValue] = {}
        self._cutoffs: dict[int, float] = {}
        self._columns()

    @property
    def vectors(self) -> np.ndarray:
        """:math:`U`, ``(ndof, 2 m)``: the 3D columns of the current basis."""
        return self._vectors

    @property
    def label(self) -> str:
        """The surface's wave ports, for messages."""
        return wave_port_label(list(self.ports), lone_surface=True)

    def _columns(self) -> None:
        """``vectors`` and the face reactions of the current basis."""
        modes = self.modes
        # the tracking mass splits by field part: $e_t = \gamma E_t$ scales with the
        # file unit, $e_n$ does not (`_pairs` weighs the parts alike)
        v = self._reduced.v
        n_t = getattr(modes, "n_t", None)
        if n_t is None:  # a line port's mass has no normal part
            self._tangential = None
            self._mass_t, self._mass_n = self._reduced.mass, None
        else:
            self._tangential = np.asarray(modes.free) < n_t
            t = self._tangential[:, None]
            self._mass_t = v.T @ (modes.mass @ (v * t))
            self._mass_n = v.T @ (modes.mass @ (v * ~t))
        self._reduction = modes.reduce(self._reduced.v)
        w = self._reduction.w
        basis = np.zeros((modes.space.ndof, w.shape[1]))
        basis[modes.free] = w
        columns = modes.port_columns(basis)
        self._reaction = columns["reaction"]
        self._vectors = self._lift(columns["r"])

    # --- evaluation -------------------------------------------------------------

    def at(self, k: complex) -> PortTermValue:
        """The term at the complex wavenumber ``k`` (per file unit).

        Raises:
            NotImplementedError: The cross-section has an absorbing edge
                beside a permittivity law (not analytic in the frequency).
            PortTermRefusalError: Enrichment does not bring a mode's reduced pair
                under the guard (a `PortModeError`).
        """
        k = complex(k)
        hit = self._cache.get(k)
        if hit is not None:
            return hit
        try:
            d_theta = self.modes.pencil.d_theta(k)
        except NotImplementedError as error:
            raise NotImplementedError(f"{self.label}: {error}") from None
        for attempt in range(_ENRICHMENTS + 1):
            pairs, floor = self._pairs(k)
            bad = [i for i, (_, _, eta) in enumerate(pairs) if not eta <= floor[i]]
            if not bad:
                break
            self._refuse_ambiguous(k, pairs, bad, floor)
            if attempt == _ENRICHMENTS:
                raise PortTermRefusalError(
                    self.label,
                    k,
                    f"{self.label}: the port mode at the complex wavenumber "
                    f"{k:.6g} per file unit has a backward error "
                    f"{max(pairs[i][2] for i in bad):.1e} above the port mode "
                    f"tolerance {self.tolerance:.1e} after {_ENRICHMENTS} exact "
                    "solves.",
                )
            self._enrich(k, pairs, bad)
        value = self._value(k, pairs, d_theta)
        if len(self._cache) >= _CACHE:
            self._cache.clear()
        self._cache[k] = value
        return value

    def _refuse_ambiguous(
        self, k: complex, pairs: list, bad: list[int], floor: list[float]
    ) -> None:
        """Refuse at once a pair that passes the guard but is not told apart.

        An exact solve cannot separate two accurate modes (it moves the basis
        and the pair stays ambiguous), so enriching would only spend solves.

        Raises:
            PortTermRefusalError: A bad pair is finite, passes the guard on its
                own backward error and overlaps two reduced modes almost equally.
        """
        theta = self.modes.pencil.theta(k)  # ty: ignore[invalid-argument-type]
        for i in bad:
            lam, y, eta = pairs[i]
            if eta != math.inf or not np.isfinite(lam):
                continue
            if self._reduced.backward_error(theta, lam, y) > floor[i]:
                continue
            first, second = self._reduced.margins[i]
            if second <= first - _AMBIGUOUS:
                continue  # two references on one pair: enrichment may separate them
            f = k * scipy.constants.c / (2 * math.pi * self.modes.section.scale)
            raise PortTermRefusalError(
                self.label,
                k,
                f"{self.label}: two reduced modes overlap the port mode at "
                f"the complex frequency {complex_hertz(f)} almost equally "
                f"(overlaps {first:.3f} and {second:.3f}), so it cannot be "
                "followed there.",
            )

    def below_cutoff(self, k: float) -> bool:
        """Whether a port mode of the surface is below its cutoff at the real ``k``.

        ``k`` per file unit. From each mode's `cutoff`, never from `at`: far
        below the band the reduced model needs exact solves there, which
        would change the basis the forward stands on, and may not converge.
        """
        f = k * scipy.constants.c / (2 * math.pi * self.modes.section.scale)
        return any(f < self.cutoff(p.mode) for p in self.ports)

    def cutoff(self, mode: int) -> float:
        """The cutoff of mode ``mode`` in rank order, Hz; 0 without one in reach.

        Bisected to `_BELOW_RTOL` on the port mode problem below the band
        (where the mode propagates), on first use, then kept.
        """
        hit = self._cutoffs.get(mode)
        if hit is None:
            hit = self._cutoffs[mode] = _cutoff_below(
                self.modes, float(self.band[0]), mode
            )
        return hit

    def _nearest(self, k: complex) -> int:
        return int(np.argmin(np.abs(self._k - k.real)))

    def _ref(self, i: int, j: int) -> np.ndarray:
        """Mode ``i``'s pencil coordinates at band frequency ``j`` (current basis)."""
        z = self.sweeps[i].coords[j]
        y = z[len(z) // 2 :]  # the field coordinates are (y / gamma, y)
        return np.concatenate([y, np.zeros(self._reduced.size - len(y))])

    def _pairs(self, k: complex) -> tuple[list, list[float]]:
        """Each mode's ``(lam, y, error)`` at ``k``, and the guard's floors.

        On a band frequency it is the sweep's tabulated mode, so the term is
        the driven coupling's there.
        """
        j = self._nearest(k)
        floors = [float(s.floor[j]) for s in self.sweeps]
        refs = [self._ref(i, j) for i in range(len(self.ports))]
        if k == self._k[j]:
            scale = self.modes.section.scale
            return [
                (complex((s.gamma[j] * scale) ** 2), y, float(s.residual[j]))
                for s, y in zip(self.sweeps, refs, strict=True)
            ], floors
        k_max = 2 * self._k_max[j] * abs(k) / self._k[j]
        pairs = self._reduced.solve(k, k_max, refs, mass=self._tracking_mass(k))  # ty: ignore[invalid-argument-type]
        return pairs, floors

    def _tracking_mass(self, k: complex) -> np.ndarray:
        r"""The reduced matrix of :math:`M_t + |k|^2 M_n`, the pairs' inner product.

        The tangential part of a port mode's field carries :math:`\gamma`
        (:math:`e_t = \gamma E_t`) and the normal part does not, so the plain
        mass weighs them by the file unit; :math:`|k|` per file unit balances
        them, and the margin between two pairs no longer depends on the unit.
        """
        if self._mass_n is None:
            return self._mass_t
        return self._mass_t + abs(k) ** 2 * self._mass_n

    def _weights(self, k: complex) -> np.ndarray:
        """The diagonal ``D`` of the tracking inner product ``D M D`` on free DOFs."""
        if self._tangential is None:
            return np.ones(len(self.modes.free))
        return np.where(self._tangential, 1.0, abs(k))

    def _enrich(self, k: complex, pairs: list, bad: list[int]) -> None:
        """Exact solves at ``k`` for modes ``bad``, added to the basis."""
        a, b = self.modes.pencil.at(k)  # ty: ignore[invalid-argument-type]
        j = self._nearest(k)
        n = a.shape[0]
        found = []
        for i in bad:
            lam, y, _ = pairs[i]
            if not np.isfinite(lam):  # no reduced pair: start from the band
                g = self.sweeps[i].gamma[j] * self.modes.section.scale
                lam, y = g**2 * (k / self._k[j]) ** 2, self._ref(i, j)
            _, vecs = shift_invert_eigs(a, b, lam, min(_NEV, n - 2))
            ref = self._reduced.lift(y)
            weight = self._weights(k)
            found.append(
                vecs[
                    :,
                    int(
                        np.argmax(
                            _overlaps(
                                self.modes.mass, ref * weight, vecs * weight[:, None]
                            )
                        )
                    ),
                ]
            )
        self._reduced.add(np.stack(found, axis=1))
        self._columns()
        self._cache.clear()
        self.version += 1
        self.enrichments.append(k)

    def _value(
        self, k: complex, pairs: list, d_theta: tuple[complex, ...]
    ) -> PortTermValue:
        theta = self.modes.pencil.theta(k)  # ty: ignore[invalid-argument-type]
        a, b = self._reduced.matrices(theta)
        da, db = self._reduced.matrices(d_theta)
        ys, dys, gammas, d_gammas = [], [], [], []
        for lam, y, _ in pairs:
            dy, d_lam = _bordered((a, b), (da, db), lam, y)
            gamma = outgoing(lam)
            ys.append(y)
            dys.append(dy)
            gammas.append(gamma)
            d_gammas.append(d_lam / (2 * gamma))
        c, d_c = self._correction(ys, dys, gammas, d_gammas)
        return PortTermValue(
            k=k,
            gamma=np.array(gammas),
            d_gamma=np.array(d_gammas),
            lead=self._lead,
            c=c,
            d_c=d_c,
            error=max(eta for _, _, eta in pairs),
            version=self.version,
            y=np.stack(ys, axis=1),
            d_y=np.stack(dys, axis=1),
        )

    def _correction(self, ys, dys, gammas, d_gammas) -> tuple[np.ndarray, np.ndarray]:
        r""":math:`C = P D P^\mathsf{T}` and :math:`C'`, as on the real grid.

        The field coordinates are :math:`(y/\gamma, y)`, so a mode's n x H
        on the columns is :math:`(y, y)` and its tangential trace
        :math:`y/\gamma`.
        """
        n, lead = len(ys), self._lead
        g0, dg0 = gammas[lead], d_gammas[lead]
        y, dy = np.stack(ys, axis=1), np.stack(dys, axis=1)
        g, dg = np.array(gammas), np.array(d_gammas)
        t = y / g
        dt = (dy * g - y * dg) / g**2
        zero = np.zeros_like(t)
        p = np.concatenate([np.concatenate([y, y]), np.concatenate([g0 * t, zero])], 1)
        dp = np.concatenate(
            [np.concatenate([dy, dy]), np.concatenate([dg0 * t + g0 * dt, zero])], 1
        )
        rt, rdt = self._reaction @ t, self._reaction @ dt
        reactions = p.T @ rt
        d_reactions = dp.T @ rt + p.T @ rdt
        r, dr = reactions[:n], d_reactions[:n]
        r_inv = np.linalg.inv((r + r.T) / 2)
        g_inv = np.linalg.inv(reactions[n:])
        d = scipy.linalg.block_diag(r_inv, -g_inv)
        dd = scipy.linalg.block_diag(
            -r_inv @ ((dr + dr.T) / 2) @ r_inv, g_inv @ d_reactions[n:] @ g_inv
        )
        c = p @ d @ p.T
        d_c = dp @ d @ p.T + p @ dd @ p.T + p @ d @ dp.T
        return c, d_c

    # --- the outgoing power ---------------------------------------------------

    def outgoing_power(
        self,
        k: complex,
        x: np.ndarray,
        pair: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda a, b: a.T @ b,
    ) -> np.ndarray:
        r"""``(n,)`` the time-averaged power each port's mode carries out, W.

        For the 3D field ``x`` (volts per file unit, on the eigen operator's
        space) at the complex wavenumber ``k``: :math:`P_p = |a_p|^2\,
        \mathrm{Re}\,P_{\text{mode},p}`, with :math:`a_p` the field's
        biorthogonal modal amplitude on port :math:`p`'s mode (the driven S
        functional's, :math:`a = R^{-1}\Phi^\mathsf{T} U^\mathsf{T} x`, in
        SI field units) and :math:`P_{\text{mode},p} = \tfrac12\int (e_p
        \times h_p^*)\cdot n\,dA` the mode's own complex power at
        :math:`\omega = k c / s`. The cross terms between the modes of a
        shared surface are dropped: they vanish on a lossless section.
        Positive for a wave leaving the domain.

        Args:
            k: The wavenumber, per file unit.
            x: The field, a state column.
            pair: The layout's pairing of load columns and a state
                (``load.T @ state`` on one process).
        """
        value = self.at(k)
        return self._power(k, value.y, value.gamma, pair(self.vectors, x))

    def _power(
        self, k: complex, y: np.ndarray, gamma: np.ndarray, ux: np.ndarray
    ) -> np.ndarray:
        """`outgoing_power` from the modes' ``y`` and ``gamma``, ``ux = U^T x``."""
        amplitude, power = self._amplitude(k, y, gamma, ux)
        return np.abs(amplitude) ** 2 * power

    def _amplitude(
        self, k: complex, y: np.ndarray, gamma: np.ndarray, ux: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Each mode's amplitude (SI) and :math:`\mathrm{Re}\,P_\text{mode}`."""
        # the field coordinates (y / gamma, y): n x H is (y, y) on the columns
        t = y / gamma
        phi = np.concatenate([y, y])
        amplitude = np.linalg.solve(phi.T @ (self._reaction @ t), phi.T @ ux)
        scale = self.modes.section.scale
        omega = k * scipy.constants.c / scale
        power = np.array(
            [
                self._reduction.power(self._reduction.coords(y[:, p], g), g, omega)
                for p, g in enumerate(gamma)
            ]
        )
        # the field is scale times the SI one
        return amplitude / scale, power.real

    def power_slopes(
        self,
        k: complex,
        x: np.ndarray,
        pair: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda a, b: a.T @ b,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""`outgoing_power` and its Wirtinger partials in ``x`` and ``k``.

        :math:`P_p = |a_p|^2 S_p` with :math:`a = M^{-1}\Phi^\mathsf{T}
        U^\mathsf{T} x` linear in :math:`x`, so :math:`\partial P_p/\partial x
        = S_p\,\overline{a_p}\,U\Phi M^{-\mathsf{T}} e_p`. In :math:`k` at
        fixed :math:`x`, :math:`P_p` moves with :math:`y(k)`,
        :math:`\gamma(k)` and :math:`\omega`: a fourth-order difference
        along the first-order path :math:`y + y'\delta`, :math:`\gamma +
        \gamma'\delta` (exact to first order, so the difference has only its
        truncation, :math:`10^{-12}` relative at the step
        :math:`10^{-3}|k|`; :math:`P_p` does not depend on the scale of
        :math:`y`, so the bordered solve's normalization does not matter),
        and :math:`\partial P/\partial k = (\partial_{\mathrm{Re}\,k} -
        i\,\partial_{\mathrm{Im}\,k}) P / 2`.

        Returns:
            ``(power (n,), d_x (ndof, n), d_k (n,))``: ``d_x`` load columns.
        """
        value = self.at(k)
        y, gamma, dy, dg = value.y, value.gamma, value.d_y, value.d_gamma
        ux = pair(self.vectors, x)
        amplitude, power = self._amplitude(k, y, gamma, ux)
        phi = np.concatenate([y, y])
        m = phi.T @ (self._reaction @ (y / gamma))
        scale = self.modes.section.scale
        # d a_p / dx: U Phi M^-T e_p, the amplitude in SI
        rows = self.vectors @ (phi @ np.linalg.inv(m).T) / scale
        d_x = rows * (power * np.conj(amplitude))
        h = _POWER_STEP * abs(k)

        def along(delta: complex) -> np.ndarray:
            return self._power(k + delta, y + dy * delta, gamma + dg * delta, ux)

        parts = []
        for unit in (1.0, 1j):
            f = [along(s * h * unit) for s in (-2, -1, 1, 2)]
            parts.append((f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h))
        d_k = 0.5 * (parts[0] - 1j * parts[1])
        return np.abs(amplitude) ** 2 * power, d_x, d_k

    def degenerate(self, k: complex, rtol: float) -> list[int]:
        r"""The ports whose mode has another within ``rtol`` at ``k``.

        Among the reduced pencil's eigenvalues :math:`\lambda = \gamma^2`
        (relative distance): there :math:`y'(k)` is not defined.
        """
        value = self.at(k)
        a, b = self._reduced.matrices(self.modes.pencil.theta(k))  # ty: ignore[invalid-argument-type]
        lam = scipy.linalg.eigvals(a, b)
        lam = lam[np.isfinite(lam)]
        out = []
        for p, g in enumerate(value.gamma):
            own = g * g
            if np.count_nonzero(np.abs(lam - own) <= rtol * abs(own)) > 1:
                out.append(p)
        return out

    @property
    def nbytes(self) -> int:
        """The dense state a backward reads: the basis, its 3D columns, the sweeps."""
        held = self._vectors.nbytes + self._reaction.nbytes + self._reduced.v.nbytes
        held += sum(int(r.nbytes) for r in self._reduced._red)  # noqa: SLF001
        for sweep in self.sweeps:
            held += int(sweep.basis.nbytes)
        return held

    # --- the driven piece -------------------------------------------------------

    def robin_piece(self, model: Model, built: EngineMesh, order: int) -> terms.Piece:
        r"""The Robin term with :math:`\theta(\omega) = \gamma_0(k)`, nonpolynomial.

        :math:`k = \omega s / c` with :math:`s` the scale; ``d_theta`` is
        :math:`\gamma_0'(k)\,s/c`. Its ``kind`` stays ``"port"``.
        """
        lead = self.ports[self._lead]
        scale = self.modes.section.scale
        per_omega = scale / scipy.constants.c
        omegas = 2 * np.pi * self.band
        piece = terms.wave_port_piece(
            model,
            built,
            order,
            port=lead.name,
            surface=lead.surface,
            theta=terms.Tabulated(omegas, self.sweeps[self._lead].gamma * scale),
            stretch=self.modes.section.stretch,
        )

        def theta(omega: complex) -> complex:
            return self.at(omega * per_omega).gamma0

        def d_theta(omega: complex) -> complex:
            return self.at(omega * per_omega).d_gamma0 * per_omega

        return dataclasses.replace(
            piece,
            theta=theta,
            frequency_class="nonpolynomial",
            d_theta=d_theta,
            probe=complex(omegas[len(omegas) // 2]),
        )


def _bordered(
    pencil: tuple[np.ndarray, np.ndarray],
    slope: tuple[np.ndarray, np.ndarray],
    lam: complex,
    y: np.ndarray,
) -> tuple[np.ndarray, complex]:
    r""":math:`y'` and :math:`\lambda'` of a pair, with :math:`y^\mathsf{T} y' = 0`.

    ``pencil`` is :math:`(A, B)` and ``slope`` :math:`(A', B')`.
    """
    (a, b), (da, db) = pencil, slope
    n = len(y)
    mat = np.zeros((n + 1, n + 1), complex)
    mat[:n, :n] = a - lam * b
    mat[:n, n] = -(b @ y)
    mat[n, :n] = y
    rhs = np.concatenate([-(da - lam * db) @ y, [0.0]])
    sol = np.linalg.solve(mat, rhs)
    return sol[:n], complex(sol[n])


def port_terms(
    model: Model,
    built: EngineMesh,
    space: ngs.FESpace,
    *,
    order: int,
    target: float,
    tolerance: float,
    stretch: Stretch | None = None,
) -> list[PortTerm]:
    """Each wave port surface's term at complex wavenumbers, one process.

    Runs each surface's port mode sweep over its band (`sweep_band`).

    Args:
        model: The model.
        built: Its engine mesh.
        space: The 3D space of the eigenmode operator.
        order: The element order (the port mode problem uses the same).
        target: The eigenmode target, Hz.
        tolerance: ``SolverOptions.port_mode_tolerance``.
        stretch: The perfectly matched layer's stretch of the solve.

    Returns:
        One per surface with wave ports, in the order of each surface's first
        wave port.

    Raises:
        PortModeError: A requested mode does not propagate at the target, or
            as `gfem._sweep.port_mode.port_mode_sweep`.
    """
    out = []
    for group in wave_port_surfaces(model.ports).values():
        ports = tuple(group)
        section = port_section(model, lead_port(list(ports)), stretch)
        modes = planar(mode_space(section, order, None))
        label = wave_port_label(list(ports), lone_surface=True)
        band = sweep_band(modes, target, max(p.mode for p in ports), label)
        # the absorbing term does not depend on a mode's sign or phase, and a
        # band near cutoff may read a polarity point at a node of the field
        unsigned = [dataclasses.replace(p, polarity=None) for p in ports]
        sweeps = port_mode_sweep(
            modes, band, [p.mode for p in ports], tolerance=tolerance, ports=unsigned
        )
        lift = face_vectors(model, modes, built, space)
        out.append(PortTerm(ports, modes, sweeps, tolerance=tolerance, lift=lift))
    return out
