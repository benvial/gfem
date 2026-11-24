r"""The share of a mode's port loss that its wave ports' requested modes carry.

A wave port surface absorbs the modes it requests exactly and any other
tangential field through its Robin scalar only, so a mode carried by an
unrequested port mode family, or trapped against the port, loses power there
that no requested mode accounts for. For a mode :math:`x` at the complex
wavenumber :math:`k` and a surface :math:`s` the requested power share is

.. math:: \eta_s = \frac{\sum_{p \in s} P_p}{P_s}, \qquad
   P_s = \frac{\mathrm{Im}\,(\bar x^\mathsf{T} B_s(k)\, x)}
   {2\,\mathrm{Re}\,\omega\,\mu_0\, s_\ell},

with :math:`P_p` the power the requested mode :math:`p` carries out
(`PortTerm.outgoing_power`, the one the coupling Q uses), :math:`B_s =
\gamma_0 M_s + U C U^\mathsf{T}` the surface's port term and :math:`s_\ell`
the mesh length scale. The constant in :math:`P_s` is calibrated: on a lossless
line :math:`\sum_p P_p / P_s = 1 + 10^{-7}`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import scipy.constants as const

from gfem._engine.eigen import EigenPort, assemble, matvec
from gfem._sweep.port_mode import propagating_modes
from gfem._text import hertz
from gfem.conventions import MODE_ENERGY

#: A mode warns when its share on a surface is below this: the gap between a
#: mode of the requested families (0.87 and up) and any other (4e-4 and
#: below) is structural, so the threshold is not an option.
THRESHOLD = 0.5
#: ... and when the surface takes at least this part of the mode's loss
#: (:math:`Q / q_s`), so a mode whose Q is the material's is not called the
#: port's.
LOSS_FLOOR = 0.01


def requested_power_shares(
    ports: Sequence[EigenPort],
    space,
    x: np.ndarray,
    ks: Sequence[complex],
    *,
    scale: float,
    pair: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    r"""``(n_surfaces, n)`` the shares :math:`\eta_s` and :math:`Q / q_s`.

    Args:
        ports: The eigen operator's wave port surfaces, in model order.
        space: The eigen operator's finite element space.
        x: The modes, columns normalized to `MODE_ENERGY`.
        ks: Their wavenumbers, per file unit.
        scale: The mesh length scale, m per file unit.
        pair: The layout's pairing of load columns and a state.

    Returns:
        :math:`\eta_s` (NaN where the surface absorbs no power from the mode)
        and :math:`Q / q_s = Q P_s / (\mathrm{Re}\,\omega\,E)` the surface's
        share of the mode's loss.
    """
    share = np.full((len(ports), len(ks)), np.nan)
    loss = np.full((len(ports), len(ks)), np.nan)
    mu0 = 4e-7 * math.pi
    per_k = const.c / scale
    for i, port in enumerate(ports):
        face = assemble(space, [(1.0, port.piece)])
        term = port.term
        for j, k in enumerate(ks):
            xj = x[:, j]
            value = term.at(k)
            bx = value.gamma0 * matvec(face, xj) + term.vectors @ (
                value.c @ pair(term.vectors, xj)
            )
            omega = k * per_k
            absorbed = float(np.vdot(xj, bx).imag) / (2 * omega.real * mu0 * scale)
            if not absorbed > 0:
                continue
            requested = float(np.sum(term.outgoing_power(k, xj, pair)))
            share[i, j] = requested / absorbed
            q = abs(omega) / (2 * abs(omega.imag)) if omega.imag else math.inf
            loss[i, j] = q * absorbed / (omega.real * MODE_ENERGY)
    return share, loss


class Unrequested:
    """The modes a surface's requested modes do not carry, and how to say so.

    Attributes:
        flagged: The modes with a share below `THRESHOLD` and a loss share of
            at least `LOSS_FLOOR`.
    """

    def __init__(self, port: EigenPort, share: np.ndarray, loss: np.ndarray) -> None:
        """Flag the modes of one surface from its two rows."""
        self.term = port.term
        with np.errstate(invalid="ignore"):
            self.flagged = np.flatnonzero((share < THRESHOLD) & (loss >= LOSS_FLOOR))
        self._share = share

    def rank(self, f: float) -> int | None:
        """The lowest unrequested rank propagating at ``f`` (Hz), or ``None``.

        A rank below the highest requested one propagates wherever that one
        does; the one above it is asked of the port mode problem.
        """
        requested = sorted(p.mode for p in self.term.ports)
        for r in range(1, requested[-1]):
            if r not in requested:
                return r
        top = requested[-1] + 1
        if propagating_modes(self.term.modes, f, top) >= top:
            return top
        return None

    def messages(
        self, name: str, ks: Sequence[complex], scale: float, q: np.ndarray
    ) -> list[str]:
        """The warning texts of surface ``name``: form (i), then form (ii).

        Form (i) lists the flagged modes at a frequency where an unrequested
        rank propagates; form (ii) the others. A form without a mode has no
        text.
        """
        per_hz = const.c / scale / (2 * math.pi)
        ranks: dict[int, int | None] = {
            int(j): self.rank(ks[j].real * per_hz) for j in self.flagged
        }
        out = []
        for propagating in (True, False):
            modes = [j for j, r in ranks.items() if (r is not None) is propagating]
            if not modes:
                continue
            listed = ", ".join(
                f"{j} at {hertz(ks[j].real * per_hz, 9)} (Q {q[j]:.3g}, share "
                f"{self._share[j]:.2g})"
                for j in modes
            )
            head = (
                f"Eigenmode: mode(s) {listed} on wave port surface {name!r}: the "
                "requested mode(s) carry less than half of the power the port "
                "absorbs. "
            )
            trapped = (
                "the mode is trapped inside the model: its port mode family is "
                "below cutoff at the port, so its Q is the port's absorption of "
                "its evanescent tail, which grows as the port moves away. Read "
                "its Q as a lower bound, or move the port farther from the "
                "structure."
            )
            if not propagating:
                out.append(head + trapped[0].upper() + trapped[1:])
                continue
            first = min(r for j in modes if (r := ranks[j]) is not None)
            out.append(
                head + f"Mode {first} of the surface propagates at that frequency and "
                "no wave port requests it, so the port partly reflects it; if the "
                "mode is carried by it, its Q and coupling Q are not the device's "
                "and the mode may not exist: add a WavePort on "
                f"{name!r} with mode={first} and solve again. If the warning "
                f"persists, {trapped}"
            )
        return out
