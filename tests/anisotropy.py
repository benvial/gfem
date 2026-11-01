"""Plane-wave references for the per-face tensors of anisotropic materials.

Independent of `gfem._engine.anisotropy`: no geometric mean, no Schur
complement. Under e^{+iωt}, in normalized units (eta0 = 1), a plane wave
``exp(-i k0 n_eff n . r)`` along the unit normal n satisfies Faraday and
Ampère as the 6x6 pencil ``[[0, mu], [-eps, 0]] x = n_eff [[N, 0], [0, N]]
x`` on ``x = (E, H)``, with ``N = [n]x``. Its finite eigenvalues are ``±n_1``
and ``±n_2``. For each of the two forward modes (``Re n_eff > 0``),
``n x mu^-1 curl E / (i k0) = -n_eff n x mu^-1 (n x E)``; the admittance is
the unique tangential map that holds for both (ADR 0051).
"""

import numpy as np
import scipy.linalg


def cross(n):
    """``[n]x``, 3x3."""
    return np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])


def frame(n):
    """A 2x3 orthonormal tangent frame (rows t1, t2 with t1 x t2 = n)."""
    t = scipy.linalg.null_space(n[None, :]).T
    return t if np.linalg.det(np.vstack([t, n])) > 0 else t[::-1]


def rotation(rng):
    """A random rotation matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def symmetric(rng, lo, hi):
    """A random real symmetric tensor with eigenvalues in ``[lo, hi]``."""
    q = rotation(rng)
    return q @ np.diag(rng.uniform(lo, hi, 3)) @ q.T


def _forward_modes(eps, mu, n):
    zero = np.zeros((3, 3))
    left = np.block([[zero, mu], [-eps, zero]])
    right = np.block([[cross(n), zero], [zero, cross(n)]])
    values, vectors = scipy.linalg.eig(left, right)
    ok = np.isfinite(values) & (np.abs(values) < 1e6) & (values.real > 0)
    assert ok.sum() == 2, values
    return values[ok], vectors[:3, ok]


def admittance(eps, mu, n):
    """``Y``, 3x3 in the global frame: ``n x mu^-1 curl E = i k0 Y E_t``.

    For each forward mode at normal incidence, from the 6x6 pencil.
    """
    t = frame(n)
    n_eff, e = _forward_modes(eps, mu, n)
    nn = cross(n)
    inv_mu = np.linalg.inv(mu)
    g = np.column_stack(
        [t @ (-n_eff[k] * (nn @ inv_mu @ (nn @ e[:, k]))) for k in range(2)]
    )
    return t.T @ (g @ np.linalg.inv(t @ e)) @ t
