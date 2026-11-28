r"""The 1D port mode problem of a line port on NGSolve: mesh, space and pencil.

A line port of a 2D mesh carries a mode :math:`u(t)\,e^{-\gamma s}`, ``s``
the distance along the inward normal and ``t`` the coordinate along the line,
with :math:`u = E_z` or :math:`u = H_z` by polarization. In H1 elements on
the line's own 1D mesh it solves

.. math::

    (w u', v') - k_0^2 (m u, v) + \sum_\text{ends} i k_0 Y u v
    = \gamma^2 (w u, v),

with :math:`w = \mu_r^{-1}`, :math:`m = \varepsilon_r` for Ez and
:math:`w = \varepsilon_r^{-1}`, :math:`m = \mu_r` for Hz. A PEC end is
:math:`u = 0` in Ez and natural in Hz; a PMC end the reverse; an absorbing
end adds the point term with :math:`Y = \sqrt{\varepsilon_r/\mu_r}` (Ez) or
:math:`\sqrt{\mu_r/\varepsilon_r}` (Hz), real parts of the segment at the end.

The pencil is affine as `gfem._engine.port_mode.Pencil`: matrices times 1,
:math:`-k_0^2`, :math:`ik_0`, then further scalars. In Ez the conductivity
:math:`-i\sigma/(\omega\varepsilon_0)` enters the :math:`ik_0` piece and a
law's susceptibility its own :math:`-k_0^2\chi(\omega)` piece. In Hz the
weight :math:`w` itself depends on the frequency where the permittivity does
(conductivity or a law): each such group of segments gets the pieces
:math:`\int u'v'` (in ``A``) and :math:`\int uv` (in ``B``) with the scalar
:math:`1/\varepsilon_r(\omega)`. An absorbing end beside a law gets its own
point piece with the scalar :math:`ik_0 Y(\omega)`. Coordinates are in the
mesh file's unit, so :math:`k_0` is per local unit.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import netgen.meshing as ngm
import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.sparse as sp

from gfem._engine import quadrature
from gfem._engine.dispersion import susceptibility
from gfem._engine.port_mode import (
    LawEdgeScalar,
    LawScalar,
    Pencil,
    PencilScalar,
    PortModeSpace,
    admittance,
    omega_of,
    susceptibility_series,
)
from gfem._engine.runtime import serial_spaces
from gfem._engine.taylor import Taylor
from gfem._linalg.base import free_mask
from gfem._line_section import LineSection
from gfem._port_section import PortSection
from gfem.problems import Polarization

Z0 = scipy.constants.mu_0 * scipy.constants.c  # vacuum wave impedance, ohm

#: Per polarization: which end condition is essential, the field's name and
#: unit, and the vacuum constant of its power (``mu0`` for Ez, ``eps0`` for Hz).
_POLARIZATION = {
    "Ez": ("pec", "Ez", "V/m", scipy.constants.mu_0),
    "Hz": ("pmc", "Hz", "A/m", scipy.constants.epsilon_0),
}


def mode_space(
    section: PortSection | LineSection,
    order: int,
    polarization: Polarization | None = None,
) -> ModeSpace:
    """The discrete port mode problem of a section: planar, or a line's.

    Raises:
        ValueError: A line section without a polarization.
    """
    if isinstance(section, LineSection):
        if polarization is None:
            raise ValueError(
                f"WavePort {section.port.name!r} is a line port: its mode needs "
                'a polarization, "Ez" or "Hz".'
            )
        return LineModeSpace(section, order, polarization)
    return PortModeSpace(section, order)


@dataclasses.dataclass(frozen=True)
class InverseEpsScalar:
    r""":math:`1/\varepsilon_r(\omega)` of a group of Hz segments.

    Attributes:
        eps: The relative permittivity (eps_inf beside a law).
        sigma: The conductivity, S/m.
        law: The permittivity law, or ``None``.
        scale: The local unit in metres.
    """

    eps: complex
    sigma: float
    law: Any
    scale: float

    def __call__(self, k0: float) -> complex:
        """The value at wavenumber ``k0`` (per local unit)."""
        omega = omega_of(k0, self.scale)
        value = self.eps - 1j * self.sigma / (omega * scipy.constants.epsilon_0)
        if self.law is not None:
            value += susceptibility(self.law, omega)
        return 1 / value

    def taylor(self, k0: float) -> np.ndarray:
        r""":math:`\theta^{(n)}(k_0)/n!` for :math:`n \le 3` along real ``k0``."""
        omega = Taylor.variable(k0) * (scipy.constants.c / self.scale)
        value = self.eps - omega.reciprocal() * (
            1j * self.sigma / scipy.constants.epsilon_0
        )
        if self.law is not None:
            value = value + susceptibility_series(self.law, self.scale, k0)
        return value.reciprocal().c


class LineModeSpace:
    """The discrete port mode problem of one `LineSection`.

    The space protocol of `gfem._engine.port_mode.PortModeSpace` that the port
    mode sweep and the driven couplings read, for a scalar mode.

    Attributes:
        section: The line's cross-section.
        order: H1 element order.
        polarization: ``"Ez"`` or ``"Hz"``.
        quantity: The mode field's name, ``"Ez"`` or ``"Hz"``.
        unit: Its unit, ``"V/m"`` or ``"A/m"``.
        mesh: The 1D NGSolve mesh in the line's coordinate ``t``; vertex ``k``
            is node ``k`` of the section.
        element_segment: The section segment of each NGSolve element.
        space: The H1 space.
        free: Indices of the free (non-Dirichlet) DOFs.
        pencil: The affine pieces of ``A`` and ``B``.
    """

    def __init__(
        self, section: LineSection, order: int, polarization: Polarization
    ) -> None:
        """Build the mesh and space and assemble the pencil."""
        if polarization not in _POLARIZATION:
            raise ValueError(f"Unknown polarization {polarization!r}.")
        self.section = section
        self.order = order
        self.polarization = polarization
        essential, self.quantity, self.unit, self._vacuum = _POLARIZATION[polarization]
        self.mesh, self.element_segment = _build_mesh(section, essential)
        with serial_spaces():
            self.space = ngs.H1(
                self.mesh, order=order, complex=True, dirichlet="essential"
            )
        self.free = np.flatnonzero(free_mask(self.space.FreeDofs(), self.space.ndof))
        self.pencil, self._b_weights, self._b_values = self._assemble()

    def __reduce__(
        self,
    ) -> tuple[type[LineModeSpace], tuple[LineSection, int, Polarization]]:
        """Pickle as the section, order and polarization, rebuilt on load."""
        return (LineModeSpace, (self.section, self.order, self.polarization))

    # --- assembly -------------------------------------------------------------

    def _p0(self, values: np.ndarray) -> ngs.CoefficientFunction:
        """A per-segment coefficient (``values`` in section segment order)."""
        values = np.asarray(values, complex)
        gf = ngs.GridFunction(ngs.L2(self.mesh, order=0, complex=True))
        gf.vec.FV().NumPy()[:] = values[self.element_segment]
        return gf

    def _dx(self):
        q = quadrature.degree(self.order)
        return ngs.dx(intrules={ngs.SEGM: ngs.IntegrationRule(ngs.SEGM, q)})

    def _assemble(self) -> tuple[Pencil, list, list]:
        """The pencil, and each slot's ``B`` weight: a coefficient and its values.

        The values are per section segment (``None`` where the slot has no
        ``B`` piece).
        """
        s = self.section
        u, v = self.space.TnT()
        dx = self._dx()
        du, dv = ngs.grad(u), ngs.grad(v)
        index = np.full(len(s.eps), -1) if s.law_index is None else s.law_index
        zero = []
        ez = self.polarization == "Ez"
        # Hz: segments whose permittivity depends on the frequency
        groups: dict[tuple, list[int]] = {}
        if not ez:
            for k in np.flatnonzero((s.sigma > 0) | (index >= 0)):
                key = (complex(s.eps[k]), float(s.sigma[k]), int(index[k]))
                groups.setdefault(key, []).append(int(k))
        steady = np.ones(len(s.eps))
        for rows in groups.values():
            steady[rows] = 0.0
        if ez:
            w, m = 1 / s.mu, s.eps
        else:
            w, m = steady / s.eps, s.mu
        a = [
            [self._p0(w) * du * dv * dx],
            [self._p0(m) * u * v * dx],
            [self._p0(Z0 * s.scale * s.sigma) * u * v * dx] if ez else zero,
        ]
        b = [self._p0(w), None, None]
        values: list[np.ndarray | None] = [np.asarray(w, complex), None, None]
        points = [None, None, self._end_points(law=False)]
        extra: list[PencilScalar] = []
        if ez:
            for g, (_, law) in enumerate(s.laws):
                a.append([self._p0(index == g) * u * v * dx])
                b.append(None)
                values.append(None)
                points.append(None)
                extra.append(LawScalar(law, s.scale))
        else:
            for (eps, sigma, g), rows in groups.items():
                inside = np.zeros(len(s.eps))
                inside[rows] = 1.0
                a.append([self._p0(inside) * du * dv * dx])
                b.append(self._p0(inside))
                values.append(inside.astype(complex))
                points.append(None)
                law = None if g < 0 else s.laws[g][1]
                extra.append(InverseEpsScalar(eps, sigma, law, s.scale))
        for g in range(len(s.laws)):
            beside = self._end_points(law=g)
            if beside is not None:
                a.append(zero)
                b.append(None)
                values.append(None)
                points.append(beside)
                extra.append(self._law_end_theta(g, ez=ez))
        mats_a = tuple(self._matrix(f, p) for f, p in zip(a, points, strict=True))
        mats_b = tuple(self._matrix([] if c is None else [c * u * v * dx]) for c in b)
        return Pencil(a=mats_a, b=mats_b, extra=tuple(extra)), b, values

    def _end_points(self, law: int | bool) -> sp.csr_matrix | None:
        """Absorbing ends' point terms: off the laws (``False``) or beside law ``g``.

        The weight is the end segment's admittance (real parts), 1 beside a law
        (its scalar holds the admittance).
        """
        s = self.section
        n = len(self.free)
        out = sp.lil_matrix((n, n), dtype=complex)
        found = False
        for node, kind, seg in zip(s.ends, s.end_kinds, s.end_segments, strict=True):
            if kind != "absorbing":
                continue
            g = -1 if s.law_index is None else int(s.law_index[seg])
            if (law is False and g >= 0) or (law is not False and g != law):
                continue
            dof = self.space.GetDofNrs(ngs.NodeId(ngs.VERTEX, int(node)))[0]
            pos = int(np.searchsorted(self.free, dof))
            eps, mu = float(s.eps[seg].real), float(s.mu[seg].real)
            ez = self.polarization == "Ez"
            y = admittance(eps, mu) if ez else admittance(mu, eps)
            out[pos, pos] += 1.0 if g >= 0 else y
            found = True
        return out.tocsr() if found else None

    def _law_end_theta(self, g: int, *, ez: bool) -> LawEdgeScalar:
        r""":math:`ik_0 Y(\omega)` of an absorbing end beside law ``g``."""
        s = self.section
        rows = np.asarray(s.law_index) == g
        mu = float(s.mu[rows][0].real)
        return LawEdgeScalar(s.laws[g][1], mu, s.scale, inverse=not ez)

    def _full_matrix(self, integrands) -> sp.csr_matrix:
        n = self.space.ndof
        form = ngs.BilinearForm(self.space)
        for term in integrands:
            form += term
        form.Assemble()
        rows, cols, vals = form.mat.COO()
        return sp.csr_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n)
        )

    def _matrix(self, integrands, points: sp.csr_matrix | None = None) -> sp.csr_matrix:
        n = len(self.free)
        m = (
            self._full_matrix(integrands)[self.free][:, self.free].tocsr()
            if integrands
            else sp.csr_matrix((n, n), dtype=complex)
        )
        return m if points is None else (m + points).tocsr()

    @functools.cached_property
    def _b_slots(self) -> list[int]:
        """The pencil slots with a ``B`` piece: the weight ``w`` is their sum."""
        return [j for j, c in enumerate(self._b_weights) if c is not None]

    @functools.cached_property
    def mass(self) -> sp.csr_matrix:
        """Unweighted L2 mass of ``u`` on free DOFs (real SPD): tracks the mode."""
        u, v = self.space.TnT()
        return self._matrix([u * v * self._dx()]).real.tocsr()

    @functools.cached_property
    def _gauge_forms(self) -> list[np.ndarray]:
        r"""Per ``B`` slot, the free-DOF vector of :math:`\int w_j v`."""
        v = self.space.TestFunction()
        out = []
        for j in self._b_slots:
            form = ngs.LinearForm(self.space)
            form += self._b_weights[j] * v * self._dx()
            form.Assemble()
            out.append(np.array(form.vec.FV().NumPy())[self.free])
        return out

    @functools.cached_property
    def _coupling_forms(self) -> list[sp.csr_matrix]:
        r"""Per ``B`` slot, :math:`\int w_j \phi_j \phi_i`.

        Rows every DOF, columns the free ones.
        """
        u, v = self.space.TnT()
        return [
            self._full_matrix([self._b_weights[j] * u * v * self._dx()])[
                :, self.free
            ].tocsr()
            for j in self._b_slots
        ]

    def _weights(self, k0: float) -> np.ndarray:
        """The scalar of each ``B`` slot at ``k0`` (per local unit)."""
        theta = self.pencil.theta(k0)
        return np.array([theta[j] for j in self._b_slots])

    @property
    def weight_values(self) -> list[np.ndarray]:
        r"""Per ``B`` slot, its weight :math:`w_j` on each section segment.

        The line's :math:`w` is :math:`\sum_j \theta_j(\omega) w_j`, the
        scalars `weights` gives.
        """
        return [self._b_values[j] for j in self._b_slots]

    def weights(self, omega: np.ndarray) -> np.ndarray:
        """``(n_freq, n_slots)``: each ``B`` slot's scalar at ``omega`` (rad/s)."""
        return np.array([self._weights(k) for k in self._k0(omega)])

    def _k0(self, omega) -> np.ndarray:
        return np.asarray(omega, float) * self.section.scale / scipy.constants.c

    # --- the sweep's protocol -------------------------------------------------

    def reduce(self, v: np.ndarray) -> LineReduction:
        """The sweep's fixed field basis on the reduced basis ``v``: ``v`` itself."""
        return LineReduction(self, v)

    def port_columns(self, basis: np.ndarray) -> dict[str, np.ndarray]:
        r"""What the line's forms give per column of the sweep's ``basis``.

        ``r``: per ``B`` slot ``j`` and column :math:`c`, the vector of
        :math:`\int w_j c\,v` over every DOF (the trace of the 2D field,
        Dirichlet DOFs included); ``reaction``: :math:`r^T e` with :math:`e`
        the basis (the mode vanishes on fixed DOFs).
        """
        w = basis[self.free]
        r = np.concatenate([m @ w for m in self._coupling_forms], axis=1)
        return {"r": r, "reaction": r.T @ basis}

    def phi(self, coords: np.ndarray, gamma: np.ndarray, omega: np.ndarray):
        r"""``(n_freq, n_b m)``: :math:`h = \gamma M e` on the columns of ``r``.

        ``gamma`` per file unit; ``M`` the ``w``-weighted line mass, a sum of
        the ``B`` slots times their scalars at ``omega`` (rad/s).
        """
        weights = np.array([self._weights(k) for k in self._k0(omega)])
        return np.concatenate(
            [
                (gamma * weights[:, j])[:, None] * coords
                for j in range(weights.shape[1])
            ],
            axis=1,
        )

    def scalar(self, trace: np.ndarray, gamma0: complex, omega: float) -> np.ndarray:
        r""":math:`\gamma_0 M e` on the columns of ``r``, for traces ``(m, n)``."""
        weights = self._weights(float(self._k0(omega)))
        return np.concatenate([gamma0 * wj * trace for wj in weights])

    def pairing_factor(self, omegas: np.ndarray) -> np.ndarray:
        r"""The factor of the reciprocity pairing per metre, W/m.

        :math:`Q' = \text{factor}\,h^T e`.
        """
        return -0.5j / (omegas[:, None] * self._vacuum)

    def pairing(self, coeffs: np.ndarray, gamma: complex, omega: float) -> complex:
        r"""Reciprocity pairing :math:`p(\omega)\,\gamma\,u^\mathsf{T} M(\omega) u`, W.

        :math:`p = -i/(2\omega v)` per metre (``v`` the vacuum constant of the
        polarization) and :math:`M` the :math:`w`-weighted line mass, so that
        :math:`\gamma\,u^\mathsf{T} M u` is the :math:`h^\mathsf{T} e` of the
        port mode solve, unconjugated: the square of the pseudo-wave scale of a
        port without a voltage path. The mirror of
        `gfem._engine.port_mode.PortModeSpace.pairing`.

        Args:
            coeffs: Full coefficient vector of the field.
            gamma: Propagation constant per local unit.
            omega: Angular frequency, rad/s.
        """
        weights = self._weights(float(self._k0(omega)))
        coeffs = np.asarray(coeffs, complex)
        free = coeffs[self.free]
        mass = sum(
            (
                w * (m @ free)
                for w, m in zip(weights, self._coupling_forms, strict=True)
            ),
            np.zeros(self.space.ndof, complex),
        )
        return -0.5j / (omega * self._vacuum) * gamma * complex(coeffs @ mass)

    def layer_share(self, vecs: np.ndarray, lam: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Each eigenvector's layer energy share: zero, a line meets no layer."""
        return np.zeros(np.asarray(vecs).reshape(len(self.free), -1).shape[1])

    # --- fields ---------------------------------------------------------------

    def full(self, x: np.ndarray) -> np.ndarray:
        """A full coefficient vector from free-DOF values (zeros elsewhere)."""
        out = np.zeros(self.space.ndof, complex)
        out[self.free] = x
        return out

    def point_columns(self, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The field at line points as a linear map of the full DOF vector.

        Args:
            local: Coordinates along ``t`` ``(n,)``, local units.

        Returns:
            ``dofs (n, k)`` the DOF numbers of the element holding each point
            (padded with 0) and ``weights (n, k)`` such that ``u`` at point
            ``p`` is ``weights[p] @ x[dofs[p]]``; zero off the line.
        """
        points = np.asarray(local, float)
        mips = [self.mesh(float(t)) for t in points]
        owners = [
            list(self.space.GetDofNrs(ngs.ElementId(ngs.VOL, mip.nr)))
            if mip.nr >= 0
            else []
            for mip in mips
        ]
        width = max((len(d) for d in owners), default=0)
        dofs = np.zeros((len(mips), width), np.int64)
        weights = np.zeros((len(mips), width), complex)
        gf = ngs.GridFunction(self.space)
        values = gf.vec.FV().NumPy()
        for i, (mip, element) in enumerate(zip(mips, owners, strict=True)):
            for j, dof in enumerate(element):
                if dof < 0:  # an unnumbered DOF of the element
                    continue
                dofs[i, j] = dof
                values[:] = 0.0
                values[dof] = 1.0
                weights[i, j] = gf(mip)
        return dofs, weights

    def evaluate(self, coeffs: np.ndarray, local: np.ndarray) -> np.ndarray:
        """``u`` at coordinates ``t`` along the line; zero off it.

        Args:
            coeffs: Full coefficient vectors, ``(m, ndof)``.
            local: Coordinates along ``t`` ``(n,)``, local units.

        Returns:
            ``(m, n)`` complex.
        """
        coeffs = np.atleast_2d(coeffs)
        mips = [self.mesh(float(t)) for t in np.asarray(local, float)]
        found = [i for i, mip in enumerate(mips) if mip.nr >= 0]
        out = np.zeros((len(coeffs), len(mips)), complex)
        gf = ngs.GridFunction(self.space)
        for j, c in enumerate(coeffs):
            gf.vec.FV().NumPy()[:] = c
            for i in found:
                out[j, i] = gf(mips[i])
        return out


#: The discrete port mode problem of a planar or a line section.
ModeSpace = PortModeSpace | LineModeSpace


@dataclasses.dataclass(eq=False)
class LineReduction:
    """A line mode's power and gauge on a fixed reduced basis.

    The field basis is the reduced basis itself (a scalar mode has one
    block), so a pencil vector's coordinates are the field's.

    Attributes:
        space: The line's port mode problem.
        w: ``(n_free, m)`` the field basis on the free DOFs.
    """

    space: LineModeSpace
    w: np.ndarray

    def __post_init__(self) -> None:
        space = self.space
        self._b = [
            self.w.T @ (space.pencil.b[j] @ self.w)
            for j in space._b_slots  # noqa: SLF001
        ]
        self._gauge = [g @ self.w for g in space._gauge_forms]  # noqa: SLF001

    def coords(self, y: np.ndarray, gamma: complex) -> np.ndarray:
        """The field's coordinates of a pencil vector's (the same)."""
        del gamma
        return np.array(y, complex)

    def power(self, z: np.ndarray, gamma: complex, omega: float) -> complex:
        r"""Complex power per metre :math:`\frac12\int (E\times H^*)\cdot n\,dt`, W/m.

        :math:`\frac{i\gamma^*}{2\omega\mu_0}\int|u|^2/\mu_r^*` for Ez and
        :math:`\frac{-i\gamma}{2\omega\varepsilon_0}\int|u|^2/\varepsilon_r`
        for Hz (``gamma`` per local unit: the scale cancels against ``dt``).
        """
        space = self.space
        k0 = float(space._k0(omega))  # noqa: SLF001
        b = sum(
            wj * m
            for wj, m in zip(space._weights(k0), self._b, strict=True)  # noqa: SLF001
        )
        if space.polarization == "Ez":
            quad = z @ np.conj(b) @ z.conj()
            return 1j * np.conj(gamma) / (2 * omega * space._vacuum) * quad  # noqa: SLF001
        quad = z @ b @ z.conj()
        return -1j * gamma / (2 * omega * space._vacuum) * quad  # noqa: SLF001

    def gauge(self, z: np.ndarray, gamma: complex, omega: float) -> complex:
        r"""Phase reference: :math:`\int w u\,dt`."""
        del gamma
        space = self.space
        k0 = float(space._k0(omega))  # noqa: SLF001
        weights = space._weights(k0)  # noqa: SLF001
        return complex(
            sum(wj * (g @ z) for wj, g in zip(weights, self._gauge, strict=True))
        )


def _build_mesh(section: LineSection, essential: str) -> tuple[ngs.Mesh, np.ndarray]:
    """The 1D mesh, and the section segment of each NGSolve element.

    The ends' points carry ``"essential"`` where their condition is the
    polarization's Dirichlet one, ``"natural"`` otherwise.
    """
    m = ngm.Mesh(dim=1)  # ty: ignore[unresolved-attribute]
    ids = [
        m.Add(ngm.MeshPoint(ngm.Pnt(float(t), 0.0, 0.0)))  # ty: ignore[unresolved-attribute]
        for t in section.points
    ]
    for a, b in section.segments:
        m.Add(ngm.Element1D([ids[int(a)], ids[int(b)]], index=1))  # ty: ignore[unresolved-attribute]
    m.SetMaterial(1, "line")
    m.SetBCName(0, "essential")
    m.SetBCName(1, "natural")
    for node, kind in zip(section.ends, section.end_kinds, strict=True):
        m.Add(ngm.Element0D(ids[int(node)], index=1 if kind == essential else 2))  # ty: ignore[unresolved-attribute]
    mesh = ngs.Mesh(m)
    by_nodes = {
        (int(min(a, b)), int(max(a, b))): k for k, (a, b) in enumerate(section.segments)
    }
    order = np.array(
        [
            by_nodes[tuple(sorted(v.nr for v in el.vertices))]
            for el in mesh.Elements(ngs.VOL)
        ]
    )
    return mesh, order
