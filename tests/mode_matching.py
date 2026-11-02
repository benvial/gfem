"""Mode matching for a dielectric-slab insert in a rectangular waveguide.

An H-plane problem: a guide of width ``a`` low enough that only TE_n0 modes
(``E = E_y(x)``, no y dependence) take part, empty leads, and an insert of
length ``length`` holding a slab of ``eps`` against the wall ``x in [0,
slab]``. Every cross-section field is expanded in the same sine basis
``s_m = sqrt(2/a) sin(m pi x / a)``: the leads' modes are its members, the
insert's are the eigenvectors of ``k0^2 E - diag(kc^2)`` with ``E`` the
slab's permittivity Gram matrix in that basis (a 1D Sturm-Liouville
problem). Continuity of ``E_y`` and ``H_x`` at both junctions is then exact
in that basis, so the only error is the truncation to ``n`` modes.

Time goes as ``exp(+i omega t)``, as in gfem. S is power-normalized (a TE
mode carries power proportional to ``beta |E|^2``) with the reference planes
at the junctions, and each mode's ``E_y`` is positive near ``x = 0``. Rows
and columns are the requested modes at the left junction, then the same
modes at the right one. A full-width slab gives the single-mode
Fabry-Perot reflection and transmission.
"""

from __future__ import annotations

import numpy as np

C0 = 299792458.0


def _eps_gram(n, a, slab, eps):
    # int_0^a eps(x) s_i s_j dx with eps = ``eps`` on [0, slab], 1 elsewhere
    m = np.arange(1, n + 1)
    p = (m[:, None] - m[None, :]) * np.pi / a
    q = (m[:, None] + m[None, :]) * np.pi / a
    diff = np.where(p == 0, slab, np.sin(p * slab) / np.where(p == 0, 1, p))
    inner = (diff - np.sin(q * slab) / q) / a
    return np.eye(n) + (eps - 1) * inner


def _gamma(beta2):
    # +i beta for a propagating mode (exp(+i omega t), `gfem.conventions`),
    # alpha > 0 for an evanescent one; negate before going complex, since
    # -(x + 0j) is x - 0j, on the other side of sqrt's branch cut
    return np.sqrt((-beta2).astype(complex))


def s_matrix(f, *, a, slab, eps, length, modes=(1, 2), n=320):
    """Power-normalized S of the insert at frequency ``f`` (Hz), SI lengths.

    Args:
        f: Frequency, Hz.
        a: Guide width, m.
        slab: Slab width against the wall ``x = 0``, m.
        eps: Slab relative permittivity.
        length: Insert length, m.
        modes: TE_m0 indices, at each junction, of S's rows and columns.
        n: Sine modes kept.

    Returns:
        ``(2 len(modes), 2 len(modes))`` complex S.
    """
    k0 = 2 * np.pi * f / C0
    kc2 = (np.arange(1, n + 1) * np.pi / a) ** 2
    g1 = _gamma(k0**2 - kc2)  # leads
    b2, v = np.linalg.eigh(k0**2 * _eps_gram(n, a, slab, eps) - np.diag(kc2))
    g2 = _gamma(b2)
    p = np.diag(np.exp(-g2 * length))
    eye, zero = np.eye(n), np.zeros((n, n))
    big1, vg2 = np.diag(g1), v @ np.diag(g2)
    # unknowns: reflected b1 (left), insert c (+z) and d (-z), transmitted a3
    # (right); E_y and H_x continuous at z = 0 and z = length
    system = np.block(
        [
            [eye, -v, -v @ p, zero],
            [big1, vg2, -vg2 @ p, zero],
            [zero, v @ p, v, -eye],
            [zero, -vg2 @ p, vg2, big1],
        ]
    )
    idx = [m - 1 for m in modes]
    rhs = np.zeros((4 * n, 2 * len(idx)), complex)
    for j, q in enumerate(idx):
        rhs[q, j], rhs[n + q, j] = -1, g1[q]  # incident from the left
        rhs[2 * n + q, len(idx) + j] = 1  # incident from the right
        rhs[3 * n + q, len(idx) + j] = g1[q]
    x = np.linalg.solve(system, rhs)
    out = np.concatenate([x[:n][idx], x[3 * n :][idx]])
    beta = np.imag(np.concatenate([g1[idx], g1[idx]]))
    return out * np.sqrt(beta[:, None] / beta[None, :])
