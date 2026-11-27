r"""The port pass of a driven 2D backward: the gradient through a line port's mode.

The line counterpart of `gfem._engine.port_pass.pullback_row` and of its
face loop. A line port enters the 2D driven system through its scalar mode
:math:`u` (:math:`E_z` or :math:`H_z`) on a 1D mesh: the Robin term
:math:`\gamma_0\int w\,u\,v`, the incident load, the S functional, the
de-embedding and the outputs :math:`\gamma`. With :math:`M = \sum_j
\theta_j(\omega)\int w_j\,\phi\,\phi'` the :math:`w`-weighted line mass, the
mode's :math:`n\times H` on the line is :math:`h = \gamma M u`, its
tangential field :math:`e = u` and the Robin vector :math:`s = M u`, so the
pullback is that of the planar port with :math:`x = u`, no tangential split
and no voltage path.

The rows are independent, as in the planar pass: a driven row reads its mode
at its own frequency, so each is normalized against its own stored
coordinates.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import scipy.sparse as sp

from gfem._engine.line_mode import LineModeSpace
from gfem._engine.line_mode_sensitivity import (
    LineCotangents,
    LineKernel,
    LinePair,
    full_line_reduction,
    line_normalize_vjp,
)
from gfem._engine.port_mode_adjoint import (
    pencil_cotangent,
    polish,
)
from gfem._engine.port_mode_normalization import normalize_rows
from gfem._engine.port_rows import Mode, ModeValues, Row, RowInputs, sample_factor
from gfem._sweep.port_mode import _branch_gamma, _wavenumbers

__all__ = ["LineFaceForms", "line_face", "line_pullback_row"]


@dataclasses.dataclass(eq=False)
class LineFaceForms:
    r"""The constants of one line port's pullback.

    Attributes:
        space: The line's port mode problem.
        lead: The position of the lead port (its :math:`\gamma_0` is the Robin
            term's) among the surface's modes.
        excited: Per mode, its column among the solve's, or ``-1``.
        embed: ``(ndof, n_free)`` the 0-1 embedding of the free DOFs.
    """

    space: LineModeSpace
    lead: int
    excited: list[int]
    embed: sp.csr_matrix

    @classmethod
    def of(
        cls, space: LineModeSpace, *, lead: int, excited: list[int]
    ) -> LineFaceForms:
        """The forms of ``space`` whose lead port is ``lead``-th of its modes."""
        n = len(space.free)
        embed = sp.csr_matrix(
            (np.ones(n), (space.free, np.arange(n))), shape=(space.space.ndof, n)
        )
        return cls(space, lead, list(excited), embed)

    def mass(self, omega: float) -> sp.csr_matrix:
        r""":math:`M(\omega)`: rows every DOF, columns the free ones."""
        space = self.space
        k0 = float(space._k0(omega))  # noqa: SLF001
        weights = space._weights(k0)  # noqa: SLF001
        return sum(
            (w * m for w, m in zip(weights, space._coupling_forms, strict=True)),  # noqa: SLF001
            sp.csr_matrix((space.space.ndof, len(space.free)), dtype=complex),
        ).tocsr()


@dataclasses.dataclass(eq=False)
class LineRowBars:
    r"""The cotangents `line_pullback_row` returns.

    Attributes:
        field: ``(n_free, nk)`` of the normalized fields.
        gamma: ``(nk,)`` of the propagation constants.
        pairs: The weight's explicit cotangents (the coupling and Robin masses).
    """

    field: np.ndarray
    gamma: np.ndarray
    pairs: list[LinePair]


def line_pullback_row(  # noqa: PLR0915
    forms: LineFaceForms, modes: ModeValues, row: RowInputs
) -> LineRowBars:
    r"""The derivative of :math:`\Psi` by one line port's modes at one frequency.

    The algebra of `gfem._engine.port_pass.pullback_row` with :math:`h_k =
    \gamma_k M u_k`, :math:`e_k = u_k`, :math:`s_k = M u_k`, no voltage path and
    the pairing :math:`-i/(2\omega v)` per metre (``v`` the vacuum constant of
    the polarization).

    Args:
        forms: The line's constants.
        modes: Its modes at the frequency (`gfem._engine.port_pass.ModeValues`).
        row: The frequency's traces and cotangents
            (`gfem._engine.port_pass.RowInputs`).
    """
    space = forms.space
    field, gam = modes.field, modes.gamma
    n_modes = field.shape[1]
    a, u, weights = row.adjoint, row.solution, row.weights
    lead = forms.lead
    gam0 = gam[lead]
    pairing = -0.5j / (row.omega * space._vacuum)  # noqa: SLF001
    mass = forms.mass(row.omega)
    theta0 = space._weights(float(space._k0(row.omega)))[0]  # noqa: SLF001

    s = mass @ field
    h = gam[None, :] * s
    e = forms.embed @ field
    r = h.T @ e
    r_inv = np.linalg.inv(r)
    r_sym = 0.5 * (r + r.T)
    r_sym_inv = np.linalg.inv(r_sym)
    g = s.T @ e
    g_s = gam0 * g
    g_s_inv = np.linalg.inv(g_s)
    dual = r_inv @ h.T
    kappa = np.sqrt(pairing * np.diag(r) + 0j)

    # the S functional: f_k = kappa_k D_k, so that S reads f . u
    f_bar = weights @ u.T
    kappa_bar = np.einsum("kt,kt->k", dual, f_bar)
    dual_bar = kappa[:, None] * f_bar

    # the incident load b_k = 2 h_k / kappa_k, dotted with the adjoint column
    h_bar = np.zeros_like(h)
    for k, col in enumerate(forms.excited):
        if col < 0:
            continue
        b_bar = a[:, col]
        h_bar[:, k] += 2 * b_bar / kappa[k]
        kappa_bar[k] += -2 / kappa[k] ** 2 * (b_bar @ h[:, k])

    # D = R^-1 H^T
    r_bar = -r_inv.T @ dual_bar @ dual.T
    h_bar += dual_bar.T @ r_inv

    # the Robin scalar gamma_0 sum_j a_j^T M u_j, and the modal correction
    u_free = u[space.free]
    gam0_bar = -np.sum(a * (mass @ u_free))
    p_mat = np.concatenate([h, gam0 * s], axis=1)
    pa, pu = p_mat.T @ a, p_mat.T @ u
    n1 = pa[:n_modes] @ pu[:n_modes].T
    n2 = pa[n_modes:] @ pu[n_modes:].T
    m1, m2 = -r_sym_inv, g_s_inv
    r_sym_bar = r_sym_inv.T @ n1 @ r_sym_inv.T
    g_s_bar = -g_s_inv.T @ n2 @ g_s_inv.T
    r_bar += 0.5 * (r_sym_bar + r_sym_bar.T)
    pa_bar = np.concatenate([m1 @ pu[:n_modes], m2 @ pu[n_modes:]])
    pu_bar = np.concatenate([m1.T @ pa[:n_modes], m2.T @ pa[n_modes:]])
    p_bar = a @ pa_bar.T + u @ pu_bar.T
    h_bar += p_bar[:, :n_modes]
    s_pre_bar = p_bar[:, n_modes:]  # of gamma_0 S
    s_bar = gam0 * s_pre_bar
    gam0_bar += np.sum(s_pre_bar * s)
    g_bar = gam0 * g_s_bar
    gam0_bar += np.sum(g_s_bar * g)

    # kappa reads the pairing: kappa^2 = pairing R_kk
    r_bar[np.diag_indices(n_modes)] += kappa_bar * pairing / (2 * kappa)

    # R = H^T E and G = S^T E
    h_bar += e @ r_bar.T
    e_bar = h @ r_bar
    s_bar += e @ g_bar.T
    e_bar += s @ g_bar

    # back to the fields and the propagation constants: h = gamma s, s = M u
    s_total = s_bar + gam[None, :] * h_bar
    gamma_bar = np.sum(h_bar * s, axis=0) + row.gamma
    gamma_bar[lead] += gam0_bar
    field_bar = mass.T @ s_total + forms.embed.T @ e_bar
    pairs = [LinePair(s_total[:, k], field[:, k], theta0) for k in range(n_modes)] + [
        LinePair(a[:, j], u_free[:, j], -gam0 * theta0) for j in range(a.shape[1])
    ]
    return LineRowBars(field_bar, gamma_bar, pairs)


def line_face(
    coupling: Any,
    space: LineModeSpace,
    forms: LineFaceForms,
    *,
    index: list[int],
    rows: dict[int, Row],
    position: int,
    who: Any,
) -> LineCotangents:
    """The line's cotangents, summed over the stored rows and the modes.

    Args:
        coupling: The line's `gfem._engine.wave_ports.PortSurfaceCoupling`.
        space: Its port mode problem.
        forms: Its pullback constants.
        index: Per mode, the model's port index.
        rows: The stored rows by frequency index
            (`gfem._engine.port_pass.PortPass`).
        position: The line's position among the stored line traces.
        who: ``who(k)`` labels the ``k``-th mode for messages.
    """
    section = space.section
    fields = full_line_reduction(space)
    mass = space.mass
    sweeps = [c.sweep for c in coupling.ports]
    solved = np.flatnonzero(sweeps[0].solved)
    freq = sweeps[0].frequency
    kernel = LineKernel(space, 1.0)
    total = LineCotangents.zeros(space)
    factors: dict[int, Any] = {}
    for n in sorted(rows):
        row = rows[n]
        sample = int(solved[np.argmin(np.abs(freq[solved] - freq[n]))])
        if sample not in factors:
            factors.clear()
            factors[sample] = sample_factor(space, freq[sample])
        k0 = _wavenumbers(section, freq[n])[0]
        pencil = space.pencil.at(k0)
        n_modes = len(sweeps)
        field = np.zeros((len(space.free), n_modes), complex)
        gamma = np.zeros(n_modes, complex)
        state = []
        for k, sw in enumerate(sweeps):
            stored = sw.basis[space.free] @ sw.coords[n]
            gam = sw.gamma[n] * section.scale
            x, lam = stored, gam**2
            if not sw.solved[n]:
                lam, x = polish(
                    pencil[0], pencil[1], lam, x, factor=factors[sample], who=who(k)
                )
                gam = _branch_gamma(lam)
            norm = normalize_rows(
                fields,
                mass,
                x[None, :],
                np.array([gam]),
                np.array([row.omega]),
                reference=stored,
            )
            field[:, k] = norm.z[0]
            gamma[k] = gam
            state.append(Mode(x, lam, gam, norm, stored))
        modes = ModeValues(
            field,
            gamma,
            np.full(n_modes, complex(math.nan, math.nan)),
            np.ones(n_modes, complex),
        )
        inputs = RowInputs(
            omega=row.omega,
            adjoint=row.line_traces[position][0],
            solution=row.line_traces[position][1],
            weights=row.weights[index],
            gamma=row.gamma[index],
            impedance=row.impedance[index],
        )
        bars = line_pullback_row(forms, modes, inputs)
        pairs: list[Any] = list(bars.pairs)
        triples = []
        for k, st in enumerate(state):
            nv = line_normalize_vjp(
                fields,
                mass,
                st.norm,
                st.x[None, :],
                np.array([st.gamma]),
                np.array([row.omega]),
                z_bar=bars.field[:, k][None, :],
                reference=st.reference,
            )
            pairs += nv.pairs
            lam_bar = (bars.gamma[k] + nv.gamma[0]) / (2 * st.gamma)
            q = pencil_cotangent(
                *pencil,
                st.lam,
                st.x,
                lam_bar=lam_bar,
                x_bar=nv.y[0],
                factor=factors[sample],
                who=who(k),
            )
            triples.append((q, st.x, st.lam))
        at = kernel.at(k0)
        cot = at.cotangents(triples)
        cot.add(at.explicit(pairs))
        total.add(cot)
    return total
