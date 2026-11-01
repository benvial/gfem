"""Wave ports in the driven problem: the port mode sweep coupled to the 3D field.

A wave port enters the driven problem through four terms, all in Palace's
form (the 3D field ``E`` and test function ``v`` in the units of
`gfem._engine.terms`, the mode ``(E_t, E_n)`` propagating into the domain as
``exp(-gamma s)``, ``gamma = alpha + i beta``):

- a Robin term ``i beta ∫ mu_r^-1 E_t · v_t`` (`gfem._engine.terms`), with
  ``beta`` the real part of the mode's wavenumber (its attenuation dropped);
- a rank-two modal correction ``W = h hᵀ / (hᵀ e) - h_s h_sᵀ / (h_sᵀ e)``
  per port, never assembled, with ``h = ∫ mu_r^-1 (grad E_n + gamma E_t) ·
  v_t`` the mode's n x H as a 3D vector, ``h_s`` the same without ``grad
  E_n`` and ``e`` the mode's tangential trace (Palace's ``W_full -
  W_scalar``, ``(-i omega / R) s sᵀ`` with ``R = sᵀ e``, in gfem's units).
  On the port's own mode the Robin term and ``W`` together impose the
  mode's full n x H, so a hybrid mode (``E_n != 0``) leaves the port
  without reflection; for a TEM mode ``h = h_s`` and ``W = 0``. ``W`` is
  complex symmetric and does not depend on the mode's scale;
- the incident wave's excitation ``2 h``;
- the S functional, the projection of the field on the conjugated mode,
  ``S = (1/2) (i / (omega mu0)) ∫ E_t · conj(mu_r^-1 (grad E_n + gamma
  E_t))`` minus 1 at the excited port, with the mode scaled so that this
  projection of the incident mode itself has modulus 2 (1 W, in gfem's
  convention of time-averaged power; Palace's modes carry 1 W peak).

The 3D trace on the port face and the 2D Nédélec space of the port mode
problem share the face's triangles and order, and NGSolve orders both
bases by global vertex numbers, so the trace of a 3D field is the 2D field
with the same coefficients on matching edges and faces (`trace_dofs`). With
the mode written in the port mode sweep's fixed basis, ``mode(omega) = W
c(omega)``, the excitation and the functional are fixed vectors (one per
basis column, assembled on the 2D port once) times tabulated scalars, and
``W(omega) = vectors C(omega) vectorsᵀ`` with a tabulated ``(2 m, 2 m)``
coefficient of rank two, which is what the adaptive sweep needs.

The wave ports of one surface share its Robin term, its vectors and its
modal correction (`PortSurfaceCoupling`); the excitation and the S
functional are each port's own (`WavePortCoupling`).
"""

from __future__ import annotations

import dataclasses
import math

import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem._boundaries import WavePort
from gfem._engine import terms
from gfem._engine.mesh import EngineMesh
from gfem._engine.port_mode import PortModeSpace
from gfem._linalg.base import free_mask
from gfem._port_section import PortSection, port_section
from gfem._sweep.port_mode import PortModeSweep, port_mode_sweep
from gfem.errors import PortModeError
from gfem.model import Model

__all__ = [
    "PortSurfaceCoupling",
    "WavePortCoupling",
    "trace_dofs",
    "wave_port_couplings",
]


def _at(omegas: np.ndarray, omega: float) -> int:
    """The index of ``omega`` on the output grid ``omegas``."""
    k = int(np.argmin(np.abs(omegas - omega)))
    if abs(omegas[k] - omega) > 1e-12 * abs(omega):  # pragma: no cover
        raise ValueError(f"No port mode at {omega / (2 * math.pi):g} Hz.")
    return k


@dataclasses.dataclass(eq=False)
class WavePortCoupling:
    """One wave port's terms in the driven problem, over the output grid.

    Its surface's `PortSurfaceCoupling` holds the terms the wave ports of
    one surface share.

    Attributes:
        index: The port's position in the model's ports.
        port: The wave port.
        sweep: Its port mode sweep over the output frequencies.
        phi: ``(n_freq, 2 m)``: the mode's n x H on the columns of its
            surface's ``vectors``, ``gamma c_t`` and ``c_n`` (gamma per file
            unit): ``h = vectors @ phi``.
        norm: ``(n_freq,)``: the real factor that rescales the sweep's 1 W
            mode to Palace's power normalization (1 for a lossless mode).
        omegas: ``(n_freq,)`` the output grid, rad/s.
        scale: Metres per mesh file unit.
    """

    index: int
    port: WavePort
    sweep: PortModeSweep
    phi: np.ndarray
    norm: np.ndarray
    omegas: np.ndarray
    scale: float

    def excitation(self, omega: float) -> np.ndarray:
        """``(2 m,)``: the incident wave's right-hand side is ``vectors @ this``."""
        k = _at(self.omegas, omega)
        return 2.0 * self.scale * self.norm[k] * self.phi[k]

    def functional(self, omega: float) -> np.ndarray:
        """``(2 m,)``: the port's S row is ``this @ (conj(vectors).T @ x)``."""
        k = _at(self.omegas, omega)
        mu0 = scipy.constants.mu_0
        return 0.5j / (omega * mu0) * self.norm[k] * np.conj(self.phi[k])

    @property
    def z_ref(self) -> np.ndarray:
        """``(n_freq,)`` reference impedance ``|V|^2 / 2`` of the scaled mode."""
        return self.norm**2 * self.sweep.z_pv

    def deembedding(self, omega: float) -> complex:
        """``exp(gamma offset)``: S moves its reference plane into the domain."""
        k = _at(self.omegas, omega)
        return complex(np.exp(self.sweep.gamma[k] * self.port.offset))


@dataclasses.dataclass(eq=False)
class PortSurfaceCoupling:
    """The terms the wave ports of one surface share, over the output grid.

    Attributes:
        piece: The surface's Robin term (θ tabulated on the grid), with
            ``beta`` of the wave port with the smallest ``mode``.
        vectors: ``(ndof, 2 m)``: per mode basis column, the 3D vector of
            ``∫ mu_r^-1 (column) · v_t`` (tangential columns) or ``∫
            mu_r^-1 grad(column) · v_t`` (normal columns).
            Zero on the 3D space's fixed DOFs.
        reaction: ``(2 m, m)``: the port-face products of the columns of
            ``vectors`` with the mode basis's tangential columns, so ``hᵀ e
            = phi @ reaction @ c_t``.
        ports: The surface's wave ports, in model port order; the first
            with the smallest ``mode`` is the lead.
    """

    piece: terms.Piece
    vectors: np.ndarray
    reaction: np.ndarray
    ports: tuple[WavePortCoupling, ...]

    def __post_init__(self) -> None:
        for k in range(len(self.lead.omegas)):
            _, r_sym, g = self._blocks(k)
            for name, block in (("reaction", r_sym), ("Gram", g)):
                if _singular(block):
                    raise self._singular_error(name, k)

    @property
    def lead(self) -> WavePortCoupling:
        """The wave port with the smallest ``mode``: its gamma is ``gamma_0``."""
        return min(self.ports, key=lambda c: c.port.mode)

    def _blocks(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``P``, ``R_sym`` and ``gamma_0 G`` at output frequency ``k``."""
        lead = self.lead
        m, n = self.reaction.shape[1], len(self.ports)
        gamma0 = lead.sweep.gamma[k] * lead.scale
        full = np.stack([c.phi[k] for c in self.ports], axis=1)
        tangential = np.stack([c.sweep.coords[k, :m] for c in self.ports], axis=1)
        scalar = np.concatenate([gamma0 * tangential, np.zeros_like(tangential)])
        p = np.concatenate([full, scalar], axis=1)
        reactions = p.T @ (self.reaction @ tangential)
        r = reactions[:n]
        # R_sym, not R: it keeps D, hence C, complex symmetric
        return p, (r + r.T) / 2, reactions[n:]

    def _singular_error(self, name: str, k: int) -> PortModeError:
        lead = self.lead
        names = ", ".join(repr(c.port.name) for c in self.ports)
        kind = "WavePort" if len(self.ports) == 1 else "WavePorts"
        f = lead.omegas[k] / (2 * math.pi)
        return PortModeError(
            f"{kind} {names} on surface {lead.port.surface!r}: the modal "
            f"correction's {name} matrix is singular at {f / 1e9:.6g} GHz (the "
            "port mode's products with its own trace on the port face vanish), so "
            "the port cannot impose its mode. Check the port's cross-section and "
            "mesh, or use a LumpedPort."
        )

    def correction(self, omega: float) -> tuple[np.ndarray, np.ndarray]:
        """The modal correction at ω as ``W = vectors P D Pᵀ vectorsᵀ``.

        With ``n`` wave ports, the modes' n x H ``Φ`` and tangential
        coordinates ``C_t``, ``R = Φᵀ reaction C_t`` and ``G = C_tᵀ
        reaction[:m] C_t``.

        Returns:
            ``P`` ``(2 m, 2 n)``, ``[Φ, gamma_0 [C_t; 0]]`` (for one port
            the n x H of its mode with and without ``grad E_n``), and ``D``
            ``(2 n, 2 n)``, ``blockdiag(R_sym⁻¹, -(gamma_0 G)⁻¹)`` with
            ``R_sym = (R + Rᵀ) / 2`` (for one port ``diag(1 / R, -1 /
            R_s)``, the unconjugated reactions). Both blocks are regular:
            the coupling raises `PortModeError` when built otherwise.
        """
        p, r_sym, g = self._blocks(_at(self.lead.omegas, omega))
        n = len(self.ports)
        d = np.zeros((2 * n, 2 * n), complex)
        d[:n, :n] = np.linalg.inv(r_sym)
        d[n:, n:] = -np.linalg.inv(g)
        return p, d

    def correction_coefficients(self, omega: float) -> np.ndarray:
        """``C(ω)`` ``(2 m, 2 m)`` with ``W = vectors C vectorsᵀ`` (rank ``2 n``).

        ``C`` is complex symmetric: the adjoint reuses the forward operator
        untransposed.
        """
        p, d = self.correction(omega)
        return (p @ d) @ p.T


def _singular(a: np.ndarray) -> bool:
    """Non-finite, ``sigma_max = 0`` or ``sigma_min <= 1e-12 sigma_max``."""
    if not np.all(np.isfinite(a)):
        return True
    sigma = np.linalg.svd(a, compute_uv=False)
    return bool(sigma[0] == 0 or sigma[-1] <= 1e-12 * sigma[0])


def wave_port_couplings(
    model: Model,
    built: EngineMesh,
    space: ngs.FESpace,
    *,
    order: int,
    frequency: tuple[float, ...],
    tolerance: float,
) -> list[PortSurfaceCoupling]:
    """Run each wave port surface's port mode sweep and build its driven terms.

    Args:
        model: The model.
        built: Its engine mesh.
        space: The 3D driven space.
        order: The element order (the port mode problem uses the same).
        frequency: The output grid, Hz.
        tolerance: ``SolverOptions.port_mode_tolerance``.

    Returns:
        One coupling per surface with wave ports, in the order of each
        surface's first wave port.
    """
    omegas = 2 * np.pi * np.asarray(frequency, float)
    surfaces: dict[str, list[tuple[int, WavePort]]] = {}
    for index, port in enumerate(model.ports):
        if isinstance(port, WavePort):
            surfaces.setdefault(port.surface, []).append((index, port))
    out = []
    for group in surfaces.values():
        if len(group) > 1:  # pragma: no cover - the model rejects it
            names = ", ".join(repr(p.name) for _, p in group)
            raise NotImplementedError(f"WavePorts {names} share one surface.")
        lead = min((p for _, p in group), key=lambda p: p.mode)
        section = port_section(model, lead)
        modes = PortModeSpace(section, order)
        sweeps = [
            port_mode_sweep(modes, frequency, p.mode, tolerance=tolerance)
            for _, p in group
        ]
        idx3, idx2 = trace_dofs(model, section, modes, built, space)
        free = free_mask(space.FreeDofs(), space.ndof)
        idx3, idx2 = idx3[free[idx3]], idx2[free[idx3]]
        g_form, t_form = modes.coupling_forms
        basis = sweeps[0].basis
        w = basis[modes.free]
        m = w.shape[1] // 2
        r = np.concatenate([t_form @ w[:, :m], g_form @ w[:, m:]], axis=1)
        vectors = np.zeros((space.ndof, 2 * m), r.dtype)
        vectors[idx3] = r[idx2]
        # hᵀ e on the port face, as Palace (the mode vanishes on fixed DOFs)
        reaction = r.T @ basis[: modes.n_t, :m]
        scale = section.scale
        q = basis[: modes.n_t, :m].T @ np.conj(r)
        ports = []
        for (index, port), sweep in zip(group, sweeps, strict=True):
            coords = sweep.coords
            gamma = sweep.gamma * scale
            phi = np.concatenate([gamma[:, None] * coords[:, :m], coords[:, m:]], 1)
            # Palace's normalization: |(i / (omega mu0)) int E_t . conj(h)| = 2
            proj = np.einsum("fi,ij,fj->f", coords[:, :m], q, np.conj(phi))
            power = np.abs(scale * proj / (omegas * scipy.constants.mu_0))
            ports.append(
                WavePortCoupling(
                    index=index,
                    port=port,
                    sweep=sweep,
                    phi=phi,
                    norm=np.sqrt(2.0 / power),
                    omegas=omegas,
                    scale=scale,
                )
            )
        beta = next(c for c in ports if c.port is lead).sweep.gamma.imag * scale
        piece = terms.wave_port_piece(
            model,
            built,
            order,
            port=lead.name,
            surface=lead.surface,
            theta=terms.Tabulated(omegas, 1j * beta),
        )
        out.append(PortSurfaceCoupling(piece, vectors, reaction, tuple(ports)))
    return out


def trace_dofs(
    model: Model,
    section: PortSection,
    modes: PortModeSpace,
    built: EngineMesh,
    space: ngs.FESpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Matching DOFs of the 3D space and the port's 2D Nédélec space.

    The 3D field's tangential trace on the port has, on 2D DOF ``idx2[k]``,
    the coefficient of 3D DOF ``idx3[k]``: edges are matched through their
    end points, faces through their triangle.

    Returns:
        ``(idx3, idx2)``.
    """
    if section.faces is None:  # pragma: no cover - sections built by gfem have them
        raise ValueError("The port section does not record its mesh triangles.")
    used = np.unique(model.mesh.triangles[section.faces])
    bnd_of = np.empty(len(built.triangle_order), int)
    bnd_of[built.triangle_order] = np.arange(len(built.triangle_order))
    nd = modes.space.components[0]
    mesh3, mesh2 = built.mesh, modes.mesh
    pairs: dict[int, int] = {}

    def add(d3, d2) -> None:
        if len(d3) != len(d2):  # pragma: no cover - same order and element type
            raise RuntimeError("The 3D trace and the port space do not match.")
        for a, b in zip(d3, d2, strict=True):
            if a >= 0:
                pairs[int(b)] = int(a)

    for el in mesh2.Elements(ngs.VOL):
        bnd = mesh3[ngs.ElementId(ngs.BND, int(bnd_of[section.faces[el.nr]]))]
        edges3 = {tuple(sorted(v.nr for v in mesh3[e].vertices)): e for e in bnd.edges}
        for e in el.edges:
            key = tuple(sorted(int(used[v.nr]) for v in mesh2[e].vertices))
            add(
                space.GetDofNrs(edges3[key]),
                nd.GetDofNrs(ngs.NodeId(ngs.EDGE, e.nr)),
            )
        add(space.GetDofNrs(bnd.faces[0]), nd.GetDofNrs(el.faces[0]))
    idx2 = np.fromiter(pairs, int, len(pairs))
    idx3 = np.fromiter(pairs.values(), int, len(pairs))
    return idx3, idx2
