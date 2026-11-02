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

``Y`` is real-analytic in the principal values ``e_k`` and ``m_k`` of
coaxial real tensors while ``A`` and ``eps_eff`` stay positive definite,
so `admittance_derivatives` takes ``dY/de_k`` and ``dY/dm_k`` by complex
step on the closed form: exact to rounding, with no derivative of a matrix
function written by hand.

A wave port's cross-section holds transverse-anisotropic materials only
(ADR 0052): a principal axis along the port normal ``n``. There ``Y`` of an
absorbing face that meets the port along an edge is diagonal in the edge
tangent and ``n``, and `edge_admittance` gives those two values from the
section's 2x2 blocks in closed form.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CROSS_N",
    "admittance",
    "admittance_derivatives",
    "edge_admittance",
    "tangent_frames",
    "tangential_inverse_permeability",
]


#: ``a x n = CROSS_N a`` for a vector ``a = (a_u, a_v)`` in the plane of a
#: right-handed frame ``(u, v, n)``.
CROSS_N = np.array([[0.0, 1.0], [-1.0, 0.0]])


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


#: The complex step of `admittance_derivatives`: far below rounding of
#: principal values of order 1, far above the smallest normal number.
_STEP = 1e-30


def admittance_derivatives(
    e: np.ndarray,
    m: np.ndarray,
    axes: np.ndarray,
    normals: np.ndarray,
    *,
    where: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``dY/de_k`` and ``dY/dm_k`` per face, each ``(f, 3, 3, 3)`` (k, then Y).

    ``Y`` is `admittance` of the real tensors ``eps = sum_k e_k a_k a_k^T``
    and ``mu = sum_k m_k a_k a_k^T`` (coaxial: one set of axes per face).
    Each derivative is the imaginary part of ``Y`` at the principal value
    stepped by ``i h``, over ``h`` (complex step: no difference, so no
    cancellation).

    Args:
        e: Real principal permittivities behind each face, ``(f, 3)``.
        m: Real principal permeabilities, ``(f, 3)``.
        axes: The principal axes behind each face, ``(f, 3, 3)``, rows a_k.
        normals: Unit face normals, ``(f, 3)``.
        where: What the faces are, for the error message.

    Raises:
        ValueError: As `admittance`.
    """
    e, m = np.asarray(e, float), np.asarray(m, float)
    axes = np.asarray(axes, float)

    def tensors(values: np.ndarray) -> np.ndarray:
        return np.einsum("fk,fki,fkj->fij", values, axes, axes)

    out = []
    for values, other, first in ((e, m, True), (m, e, False)):
        steps = []
        for k in range(3):
            stepped = values + 1j * _STEP * np.eye(3)[k]
            pair = (stepped, other) if first else (other, stepped)
            y = admittance(tensors(pair[0]), tensors(pair[1]), normals, where=where)
            steps.append(y.imag / _STEP)
        out.append(np.stack(steps, axis=1))
    return out[0], out[1]


def edge_admittance(
    eps_tt: np.ndarray,
    eps_nn: np.ndarray,
    nu_tt: np.ndarray,
    nu_nn: np.ndarray,
    normals: np.ndarray,
    *,
    where: str,
) -> np.ndarray:
    """The admittances ``(Y_tau, Y_n)`` of absorbing edges, ``(m, 2)``.

    An absorbing edge of the cross-section is the trace of a 3D absorbing
    face, whose admittance is ``Y = A # eps_eff`` (ADR 0051) from the real
    parts of the material's tensors. With ``n`` principal, as the port mode
    problem requires, ``Y`` is diagonal in the face's tangents, the edge
    tangent ``tau`` and ``n``. In the frame ``(m, tau, n)``, ``m`` the
    in-plane edge normal and ``p = (m, tau)`` the in-plane block:

    - ``Y_tau = sqrt(det eps_p / (eps_mm mu_nn))`` for the field along ``tau``;
    - ``Y_n = sqrt(eps_nn mu_mm / det mu_p)`` for the field along ``n``.

    Both are ``sqrt(eps / mu)`` in an isotropic medium.

    Args:
        eps_tt, eps_nn, nu_tt, nu_nn: The medium beside each edge, as in
            `gfem._port_section.PortSection`: ``(m, 2, 2)`` and ``(m,)``
            blocks in a frame ``(u, v, n)``, ``nu_tt = R^T mu_tt^-1 R``
            with ``R = CROSS_N``.
        normals: In-plane unit edge normals ``(m, 2)`` (either sign).
        where: The surface, for the error message.

    Raises:
        ValueError: A factor is not positive, so the medium has no real
            admittance there.
    """
    m = np.asarray(normals, float)
    eps = np.asarray(eps_tt).real
    mu = (CROSS_N @ np.linalg.inv(nu_tt) @ CROSS_N.T).real  # mu_tt = R nu_tt^-1 R^T
    eps_n = np.asarray(eps_nn).real
    mu_n = (1 / np.asarray(nu_nn)).real
    eps_mm = np.einsum("ki,kij,kj->k", m, eps, m)
    mu_mm = np.einsum("ki,kij,kj->k", m, mu, m)
    det_eps, det_mu = np.linalg.det(eps), np.linalg.det(mu)
    if not all((f > 0).all() for f in (det_eps, eps_mm, mu_n, eps_n, mu_mm, det_mu)):
        raise ValueError(
            f"{where}: the material beside the port edge has no wave admittance "
            "there: the real parts of its eps_r and mu_r are not positive definite."
        )
    return np.sqrt(
        np.stack([det_eps / (eps_mm * mu_n), eps_n * mu_mm / det_mu], axis=1)
    )
