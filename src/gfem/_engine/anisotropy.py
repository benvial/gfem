"""Per-face wave admittances of anisotropic materials (ADR 0051).

Under e^{+iωt}, in normalized units (eta0 = 1), a plane wave leaving a
material through a face of unit normal n satisfies at normal incidence
``n x (mu^-1 curl E) = i k0 Y E_t`` with the 2x2 **admittance** ``Y`` of
the tangential field. In an orthonormal tangent frame ``T`` (rows t1, t2):

- ``A = T N mu^-1 N^T T^T`` (`tangential_inverse_permeability`, the
  tangential block of ``mu^-1`` turned by ``N = [n]x``);
- ``eps_eff = (T eps^-1 T^T)^-1``, the Schur complement that eliminates the
  normal field;
- ``Y = A # eps_eff``, their matrix geometric mean, from the 2x2 closed form
  ``A # B = sqrt(ab) C / sqrt(det C)`` with ``a = sqrt(det A)``, ``b =
  sqrt(det B)`` and ``C = A / a + B / b`` (no matrix square root).

An isotropic material gives ``Y = sqrt(eps / mu)`` on the tangent plane.
``Y`` and ``A`` do not depend on the frame or on the sign of n; both are
returned as 3x3 tensors in the global frame, zero along n. The formulas
hold for complex symmetric tensors (lossy media) with the principal square
root; gfem's absorbing boundary passes real parts.
"""

from __future__ import annotations

import numpy as np

__all__ = ["admittance", "tangent_frames", "tangential_inverse_permeability"]


def _cross(n: np.ndarray) -> np.ndarray:
    """``[n]x`` per row of ``n``, ``(m, 3, 3)``."""
    out = np.zeros((*n.shape[:-1], 3, 3), n.dtype)
    out[..., 0, 1], out[..., 0, 2] = -n[..., 2], n[..., 1]
    out[..., 1, 0], out[..., 1, 2] = n[..., 2], -n[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -n[..., 1], n[..., 0]
    return out


def tangent_frames(normals: np.ndarray) -> np.ndarray:
    """Orthonormal tangent frames ``(m, 2, 3)``: rows t1, t2 with t1 x t2 = n.

    ``normals`` is ``(m, 3)``, of unit length.
    """
    n = np.asarray(normals, float)
    # the coordinate axis least aligned with n, projected on the plane
    helper = np.eye(3)[np.argmin(np.abs(n), axis=-1)]
    t1 = helper - np.sum(helper * n, axis=-1, keepdims=True) * n
    t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    return np.stack([t1, np.cross(n, t1)], axis=-2)


def _in_frame(frames: np.ndarray, tensors: np.ndarray) -> np.ndarray:
    """``T M T^T`` per face: a 3x3 tensor's 2x2 tangential block."""
    return frames @ tensors @ np.swapaxes(frames, -1, -2)


def _to_global(frames: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """``T^T M T`` per face: a 2x2 tangential block as a 3x3 tensor, zero along n."""
    return np.swapaxes(frames, -1, -2) @ blocks @ frames


def tangential_inverse_permeability(mu: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """``A = N mu^-1 N^T`` per face, ``(m, 3, 3)``, with ``N = [n]x``.

    The operator that ``n x (mu^-1 curl E)`` applies to a field varying
    along n only (``curl E = n x dE/dn``). It is zero along n and even in
    n. ``mu`` is ``(m, 3, 3)`` and ``normals`` ``(m, 3)``, unit.
    """
    cross = _cross(np.asarray(normals, float))
    return cross @ np.linalg.inv(mu) @ np.swapaxes(cross, -1, -2)


def _geometric_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``A # B`` of 2x2 symmetric positive-definite stacks (closed form)."""
    da = np.sqrt(np.linalg.det(a))[..., None, None]
    db = np.sqrt(np.linalg.det(b))[..., None, None]
    c = a / da + b / db
    return np.sqrt(da * db) * c / np.sqrt(np.linalg.det(c))[..., None, None]


def _positive_definite(blocks: np.ndarray) -> np.ndarray:
    """Whether each real part of a 2x2 symmetric stack is positive definite."""
    m = blocks.real
    return (m[..., 0, 0] > 0) & (np.linalg.det(m) > 0)


def admittance(
    eps: np.ndarray, mu: np.ndarray, normals: np.ndarray, *, where: str
) -> np.ndarray:
    """The normal-incidence admittance ``Y = A # eps_eff`` per face, ``(m, 3, 3)``.

    Args:
        eps: Relative permittivity of the material behind each face, ``(m,
            3, 3)``, symmetric (complex symmetric for a lossy medium).
        mu: Relative permeability, the same shape.
        normals: Unit face normals, ``(m, 3)`` (either sign).
        where: What the faces are, for the error message (``"Absorbing
            'box'"``).

    Returns:
        ``Y`` in the global frame: symmetric, zero along the normal.

    Raises:
        ValueError: The real part of ``eps_eff`` or of ``A`` is not positive
            definite on some face, so the admittance has no real geometric
            mean.
    """
    eps, mu = np.asarray(eps), np.asarray(mu)
    frames = tangent_frames(normals)
    a = _in_frame(frames, tangential_inverse_permeability(mu, normals))
    b = np.linalg.inv(_in_frame(frames, np.linalg.inv(eps)))
    for name, blocks, what in (
        ("eps_eff", b, "permittivity"),
        ("A", a, "inverse permeability"),
    ):
        if not _positive_definite(blocks).all():
            raise ValueError(
                f"{where}: the material behind the surface has no wave admittance "
                f"there: the real part of its {name} (the tangential {what} "
                "block) is not positive definite. Check that eps_r and mu_r of "
                "that material are positive definite."
            )
    y = _geometric_mean(a, b)
    return _to_global(frames, 0.5 * (y + np.swapaxes(y, -1, -2)))
