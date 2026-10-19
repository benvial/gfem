"""The 2D port mode problem on NGSolve: mesh, spaces and the affine pencil.

Formulation (Lee, Sun and Cendes): a port mode is
``E = (E_t + n E_n) exp(-gamma s)``, ``s`` the distance along the inward
port normal ``n``. With ``e_t = gamma E_t`` and ``e_n = E_n`` in
first-kind Nédélec x H1 elements of the same order, it solves
``A x = lam B x`` with ``lam = gamma^2`` and

    A = [ (nu_nn curl e_t, curl v_t) - k0^2 (eps_tt e_t, v_t), 0 ; 0, 0 ]
    B = [ (nu_tt (e_t + grad e_n), v_t + grad v_n) - k0^2 (eps_nn e_n, v_n) ]

(``B`` holds the whole second row and column), where ``eps`` includes
``-i conductivity / (omega eps0)`` and an absorbing edge adds
``i k0 Y (E_T, V_T)`` to both diagonal blocks. Both matrices are affine in
k0: each is a sum of fixed matrices times 1, -k0^2 or i k0, assembled once.
Coordinates are in the mesh file's unit, so ``k0`` is per local unit.
"""

from __future__ import annotations

import dataclasses

import netgen.meshing as ngm
import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from gfem._port_section import PortSection

Z0 = 376.730313412  # vacuum wave impedance, ohm (CODATA 2018)


@dataclasses.dataclass(frozen=True)
class Pencil:
    """``A(k0) = sum theta_j(k0) A_j`` and ``B(k0)`` likewise, on free DOFs.

    Attributes:
        a: Matrices multiplying 1, -k0^2 and i k0 (in that order).
        b: Likewise for ``B``.
    """

    a: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]
    b: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]

    def at(self, k0: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """``A`` and ``B`` at wavenumber ``k0`` (per local length unit)."""
        theta = (1.0, -(k0**2), 1j * k0)
        a = sum(t * m for t, m in zip(theta, self.a, strict=True))
        b = sum(t * m for t, m in zip(theta, self.b, strict=True))
        return sp.csr_matrix(a), sp.csr_matrix(b)


class PortModeSpace:
    """The discrete port mode problem of one `PortSection`.

    Attributes:
        section: The cross-section.
        order: Element order (Nédélec first kind and H1).
        mesh: The 2D NGSolve mesh in local plane coordinates.
        space: The compound space ``HCurl x H1``.
        free: Indices of the free (non-Dirichlet) DOFs.
        n_t: Number of Nédélec DOFs (the first block of ``space``).
        pencil: The affine pieces of ``A`` and ``B``.
    """

    def __init__(self, section: PortSection, order: int) -> None:
        """Build the mesh and spaces and assemble the pencil."""
        self.section = section
        self.order = order
        self.mesh = _build_mesh(section)
        dirichlet = "pec" if len(section.pec_edges) else ""
        nd = ngs.HCurl(
            self.mesh, order=order, type1=True, complex=True, dirichlet=dirichlet
        )
        h1 = ngs.H1(self.mesh, order=order, complex=True, dirichlet=dirichlet)
        self.space = ngs.FESpace([nd, h1])
        self.n_t = nd.ndof
        free = self.space.FreeDofs()
        self.free = np.flatnonzero(np.array(list(free), dtype=bool))  # ty: ignore[invalid-argument-type]
        self._coef = _coefficients(self.mesh, section)
        self.pencil = self._assemble()

    # --- assembly -------------------------------------------------------------

    def _assemble(self) -> Pencil:
        c = self._coef
        (et, en), (vt, vn) = self.space.TnT()
        q = 2 * self.order
        dx = ngs.dx(intrules={ngs.TRIG: ngs.IntegrationRule(ngs.TRIG, q)})
        ds_abc = ngs.ds(
            "absorbing.*", intrules={ngs.SEGM: ngs.IntegrationRule(ngs.SEGM, q)}
        )
        sigma_scale = Z0 * self.section.scale
        wt = et + ngs.grad(en)
        wv = vt + ngs.grad(vn)
        has_abc = len(self.section.absorbing_edges) > 0
        abc_t = c["Y"] * et.Trace() * vt.Trace() * ds_abc if has_abc else None
        abc_n = c["Y"] * en.Trace() * vn.Trace() * ds_abc if has_abc else None
        forms_a = (
            [c["nu_nn"] * ngs.curl(et) * ngs.curl(vt) * dx],
            [(c["eps_tt"] * et) * vt * dx],
            [sigma_scale * c["sigma"] * et * vt * dx, abc_t],
        )
        forms_b = (
            [(c["nu_tt"] * wt) * wv * dx],
            [c["eps_nn"] * en * vn * dx],
            [sigma_scale * c["sigma"] * en * vn * dx, abc_n],
        )
        a = tuple(self._matrix(f) for f in forms_a)
        b = tuple(self._matrix(f) for f in forms_b)
        return Pencil(a=a, b=b)  # ty: ignore[invalid-argument-type]

    def _matrix(self, integrands) -> sp.csr_matrix:
        n = self.space.ndof
        form = ngs.BilinearForm(self.space)
        for term in integrands:
            if term is not None:
                form += term
        form.Assemble()
        rows, cols, vals = form.mat.COO()
        m = sp.csr_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n)
        )
        return m[self.free][:, self.free].tocsr()

    # --- fields ---------------------------------------------------------------

    def full(self, x: np.ndarray) -> np.ndarray:
        """A full coefficient vector from free-DOF values (zeros elsewhere)."""
        out = np.zeros(self.space.ndof, complex)
        out[self.free] = x
        return out

    def _gridfunction(self, coeffs: np.ndarray) -> ngs.GridFunction:
        gf = ngs.GridFunction(self.space)
        gf.vec.FV().NumPy()[:] = coeffs
        return gf

    def evaluate(self, coeffs: np.ndarray, local: np.ndarray) -> np.ndarray:
        """``(E_u, E_v, E_n)`` at plane points; zero off the cross-section.

        Args:
            coeffs: Full coefficient vectors of ``(E_t, E_n)``, ``(m, ndof)``.
            local: Plane coordinates ``(n, 2)`` in local units.

        Returns:
            ``(m, n, 3)`` complex.
        """
        coeffs = np.atleast_2d(coeffs)
        mips = [self.mesh(float(x), float(y)) for x, y in np.asarray(local, float)]
        found = [i for i, mip in enumerate(mips) if mip.nr >= 0]
        out = np.zeros((len(coeffs), len(mips), 3), complex)
        gf = ngs.GridFunction(self.space)
        et, en = gf.components
        for j, c in enumerate(coeffs):
            gf.vec.FV().NumPy()[:] = c
            for i in found:
                out[j, i, :2] = et(mips[i])
                out[j, i, 2] = en(mips[i])
        return out

    def power(self, coeffs: np.ndarray, gamma: complex, omega: float) -> complex:
        """Complex power ``1/2 int (E x H*) . n dA`` through the port, W.

        Args:
            coeffs: Full coefficient vector of ``(E_t, E_n)`` (V/m).
            gamma: Propagation constant per local unit.
            omega: Angular frequency, rad/s.
        """
        et, en = self._gridfunction(coeffs).components
        a = ngs.grad(en) + gamma * et
        mu0 = Z0 / 299792458.0
        cf = et * ngs.Conj(self._coef["nu_tt"] * a)
        value = ngs.Integrate(cf, self.mesh, order=2 * self.order)
        return 0.5 * self.section.scale * 1j / (omega * mu0) * complex(value)

    def gauge(self, coeffs: np.ndarray, gamma: complex) -> complex:
        """Palace's phase reference: ``int_Q t . (n_out x H) dA`` up to a factor.

        ``Q`` is the upper-right quadrant of the port's bounding box along its
        two largest global extents and ``t`` the in-plane part of (1, 1, 1);
        the factor is real and positive.
        """
        et, en = self._gridfunction(coeffs).components
        a = ngs.grad(en) + gamma * et
        s = self.section
        glob = [
            s.origin[k] + ngs.x * s.frame[0, k] + ngs.y * s.frame[1, k]
            for k in (0, 1, 2)
        ]
        corners = s.to_global(s.points) / s.scale
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        da, db = np.argsort(hi - lo)[::-1][:2]
        ca, cb = 0.5 * (lo + hi)[da], 0.5 * (lo + hi)[db]
        inside = ngs.IfPos(glob[da] - ca, 1.0, 0.0) * ngs.IfPos(glob[db] - cb, 1.0, 0.0)
        t = ngs.CF((float(s.frame[0].sum()), float(s.frame[1].sum())))
        cf = inside * (t * (-1j * (self._coef["nu_tt"] * a)))
        return complex(ngs.Integrate(cf, self.mesh, order=2 * self.order))


def _build_mesh(section: PortSection) -> ngs.Mesh:
    m = ngm.Mesh(dim=2)  # ty: ignore[unresolved-attribute]
    pts = np.zeros((len(section.points), 3))
    pts[:, :2] = section.points
    m.AddPoints(pts)
    m.Add(ngm.FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))  # ty: ignore[unresolved-attribute]
    m.SetMaterial(1, "port")
    m.AddElements(dim=2, index=1, data=section.triangles, base=0)
    m.SetBCName(0, "pec")
    for edge in section.pec_edges:
        m.Add(ngm.Element1D([int(edge[0]) + 1, int(edge[1]) + 1], index=1))  # ty: ignore[unresolved-attribute]
    groups = _admittance_groups(section)
    for g, (_, edges) in enumerate(groups):
        m.SetBCName(g + 1, f"absorbing{g}")
        for edge in edges:
            m.Add(ngm.Element1D([int(edge[0]) + 1, int(edge[1]) + 1], index=g + 2))  # ty: ignore[unresolved-attribute]
    return ngs.Mesh(m)


def _admittance_groups(section: PortSection) -> list[tuple[float, np.ndarray]]:
    values, inverse = np.unique(section.absorbing_admittance, return_inverse=True)
    return [
        (float(v), section.absorbing_edges[inverse.ravel() == g])
        for g, v in enumerate(values)
    ]


def _p0(mesh: ngs.Mesh, values: np.ndarray) -> ngs.CoefficientFunction:
    """A piecewise-constant coefficient, one value per element (mesh order)."""
    values = np.asarray(values)
    if np.abs(values - values[0]).max() <= 1e-14 * max(np.abs(values).max(), 1.0):
        return ngs.CF(complex(values[0]))
    gf = ngs.GridFunction(ngs.L2(mesh, order=0, complex=True))
    gf.vec.FV().NumPy()[:] = values
    return gf


def _coefficients(mesh: ngs.Mesh, s: PortSection) -> dict[str, ngs.CoefficientFunction]:
    def matrix(t: np.ndarray) -> ngs.CoefficientFunction:
        parts = [_p0(mesh, t[:, i, j]) for i in (0, 1) for j in (0, 1)]
        return ngs.CF(tuple(parts), dims=(2, 2))

    groups = _admittance_groups(s)
    y = mesh.BoundaryCF(
        {f"absorbing{g}": v for g, (v, _) in enumerate(groups)},
        default=ngs.CF(0.0),
    )
    return {
        "eps_tt": matrix(s.eps_tt),
        "eps_nn": _p0(mesh, s.eps_nn),
        "nu_tt": matrix(s.nu_tt),
        "nu_nn": _p0(mesh, s.nu_nn),
        "sigma": _p0(mesh, s.sigma.astype(complex)),
        "Y": y,
    }
