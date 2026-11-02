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


def plane_wave_n_cross_h(mu, n, e):
    """``n x H`` (times eta0) of a plane wave ``e exp(-i k0 n . r)`` along n.

    Faraday's law alone, by cross products: ``curl E = -i k0 n x e`` and
    ``curl E = -i k0 mu (eta0 H)``. Any field ``e`` and any invertible
    ``mu``.
    """
    return np.cross(n, np.linalg.solve(mu, np.cross(n, e)))


def transverse(rng, n, lo, hi):
    """A random real tensor with a principal axis along ``n``.

    Principal values in ``[lo, hi]``; the two others along random
    directions of the plane normal to ``n``.
    """
    t = frame(n)
    angle = rng.uniform(0, np.pi)
    c, s = np.cos(angle), np.sin(angle)
    axes = np.vstack([np.array([[c, s], [-s, c]]) @ t, n])
    return axes.T @ np.diag(rng.uniform(lo, hi, 3)) @ axes


def rodrigues(axis, angle):
    """The rotation by ``angle`` (rad) about the vector ``axis``."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    k_cross = cross(k)
    return np.eye(3) + np.sin(angle) * k_cross + (1 - np.cos(angle)) * k_cross @ k_cross


#: Principal axes tilted off every coordinate axis and plane (rows a_k: the
#: coordinate axes turned by 0.7 rad about (1, 2, 3)), so every off-diagonal
#: entry of each a_k a_k^T is nonzero.
TILTED = tuple(tuple(float(x) for x in row) for row in rodrigues((1, 2, 3), 0.7).T)


def turned(axis, degrees):
    """`gfem.Material` axes: the coordinate axes turned about coordinate ``axis``."""
    t = np.radians(degrees)
    i, j = (k for k in range(3) if k != axis)
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = np.cos(t), -np.sin(t), np.sin(t), np.cos(t)
    return tuple(tuple(float(x) for x in row) for row in r.T)


# --- the WR-28 guide ------------------------------------------------------------

#: WR-28 inner width and height, mm (Palace issue #835's guide).
WR28 = (7.112, 3.556)


def wr28(directory, length=1.0, cells=(12, 6, 1)):
    """A WR-28 guide along z, 0 < z < ``length`` mm, as a `gfem.Mesh`.

    Volume "fill"; surfaces "port1" (z = 0), "port2" (z = length) and
    "walls". ``cells`` along x, y and z.
    """
    from pathlib import Path

    import structured_meshes as sm

    import gfem

    a, b = WR28
    edges = tuple(
        np.linspace(0, s, n + 1) for s, n in zip((a, b, length), cells, strict=True)
    )
    tol = 1e-9 * a

    def surface(c):
        if abs(c[2]) < tol:
            return "port1"
        if abs(c[2] - length) < tol:
            return "port2"
        return "walls"

    path = sm.write_msh22(
        Path(directory) / "wr28.msh", sm.structured(edges, lambda _: "fill", surface)
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")
