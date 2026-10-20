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
``i k0 Y (E_T, V_T)`` to both diagonal blocks. A `Conductor` or `Impedance`
edge adds its admittance the same way, as the driven problem's face term
(`gfem._engine.terms`): ``i k0 eta0 / Z_s(omega)`` for a conductor, ``i k0
eta0 / R``, ``mu0 scale / L`` and ``-k0^2 C / (eps0 scale)`` for an
impedance. Both matrices are affine: each is a sum of fixed matrices times
1, -k0^2, i k0 or a conductor's own scalar, assembled once. Coordinates are
in the mesh file's unit, so ``k0`` is per local unit.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import netgen.meshing as ngm
import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.sparse as sp

from gfem._boundaries import Conductor
from gfem._engine import quadrature
from gfem._engine.terms import surface_impedance
from gfem._port_section import PortSection

Z0 = scipy.constants.mu_0 * scipy.constants.c  # vacuum wave impedance, ohm


@dataclasses.dataclass(frozen=True)
class Pencil:
    """``A(k0) = sum theta_j(k0) A_j`` and ``B(k0)`` likewise, on free DOFs.

    Attributes:
        a: Matrices multiplying 1, -k0^2, i k0, then each of ``extra``.
        b: Likewise for ``B``.
        extra: The further scalars of k0 (one per conductor edge group).
    """

    a: tuple[sp.csr_matrix, ...]
    b: tuple[sp.csr_matrix, ...]
    extra: tuple[Callable[[float], complex], ...] = ()

    def theta(self, k0: float) -> tuple[complex, ...]:
        """The scalar of each matrix at wavenumber ``k0`` (per local unit)."""
        return (1.0, -(k0**2), 1j * k0, *(f(k0) for f in self.extra))

    def at(self, k0: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """``A`` and ``B`` at wavenumber ``k0`` (per local length unit)."""
        theta = self.theta(k0)
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
        q = quadrature.degree(self.order)
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
        forms_a = [
            [c["nu_nn"] * ngs.curl(et) * ngs.curl(vt) * dx],
            [(c["eps_tt"] * et) * vt * dx],
            [sigma_scale * c["sigma"] * et * vt * dx, abc_t],
        ]
        forms_b = [
            [(c["nu_tt"] * wt) * wv * dx],
            [c["eps_nn"] * en * vn * dx],
            [sigma_scale * c["sigma"] * en * vn * dx, abc_n],
        ]
        extra = []
        fill = _edge_fill(self.mesh, self.section)
        for g, (item, factor) in enumerate(self.section.impedance_terms):
            ds = ngs.ds(
                f"impedance{g}", intrules={ngs.SEGM: ngs.IntegrationRule(ngs.SEGM, q)}
            )
            edge_t = fill * et.Trace() * vt.Trace() * ds
            edge_n = fill * en.Trace() * vn.Trace() * ds
            if isinstance(item, Conductor):
                forms_a.append([edge_t])
                forms_b.append([edge_n])
                extra.append(self._conductor_theta(item, factor))
                continue
            for slot, weight in self._impedance_weights(item):
                forms_a[slot].append(weight * edge_t)
                forms_b[slot].append(weight * edge_n)
        a = tuple(self._matrix(f) for f in forms_a)
        b = tuple(self._matrix(f) for f in forms_b)
        return Pencil(a=a, b=b, extra=tuple(extra))

    def _conductor_theta(
        self, item: Conductor, factor: float
    ) -> Callable[[float], complex]:
        """``i k0 eta0 / Z_s(omega)`` of a conductor edge group."""
        scale = self.section.scale
        h = None if item.thickness is None else factor * item.thickness
        sigma = item.conductivity

        def theta(k0: float) -> complex:
            omega = k0 * scipy.constants.c / scale
            return 1j * k0 * Z0 / surface_impedance(sigma, h, omega)

        return theta

    def _impedance_weights(self, item) -> list[tuple[int, float]]:
        """(pencil slot, weight) of an `Impedance` edge's R, L and C."""
        scale = self.section.scale
        out = []
        if item.resistance is not None:
            out.append((2, Z0 / item.resistance))
        if item.inductance is not None:
            out.append((0, scipy.constants.mu_0 * scale / item.inductance))
        if item.capacitance is not None:
            out.append((1, item.capacitance / (scipy.constants.epsilon_0 * scale)))
        return out

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

    def _vector(self, integrand) -> np.ndarray:
        form = ngs.LinearForm(self.space)
        form += integrand
        form.Assemble()
        return np.array(form.vec.FV().NumPy())[self.free]

    def _dx(self):
        q = quadrature.degree(self.order)
        return ngs.dx(intrules={ngs.TRIG: ngs.IntegrationRule(ngs.TRIG, q)})

    @functools.cached_property
    def mass(self) -> sp.csr_matrix:
        """Unweighted L2 mass of ``(e_t, e_n)`` on free DOFs (real SPD).

        The inner product that tracks a mode across frequencies.
        """
        (et, en), (vt, vn) = self.space.TnT()
        return self._matrix([(et * vt + en * vn) * self._dx()]).real.tocsr()

    @functools.cached_property
    def power_forms(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """The matrices of the power integral, on free DOFs.

        ``int E_t . conj(nu_tt (grad E_n + gamma E_t)) = c^T G conj(c) +
        conj(gamma) c^T T conj(c)``; rows belong to the unconjugated factor
        (see `power`).
        """
        (et, en), (vt, _vn) = self.space.TnT()
        nu = ngs.Conj(self._coef["nu_tt"])
        g = self._matrix([(nu * ngs.grad(en)) * vt * self._dx()])
        t = self._matrix([(nu * et) * vt * self._dx()])
        return g, t

    @functools.cached_property
    def coupling_forms(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """The matrices that couple a mode to the 3D field on the port face.

        ``M_G[i, j] = int (mu_tt^-1 grad psi_j) . phi_i`` and ``M_T[i, j] =
        int (mu_tt^-1 phi_j) . phi_i``: rows over every Nédélec DOF ``phi_i``
        (the trace of the 3D field, Dirichlet DOFs included), columns over the
        free DOFs (the mode). ``mu_tt`` is the in-plane relative permeability.
        """
        (et, en), (vt, _vn) = self.space.TnT()
        muinv = _rotate(self._coef["nu_tt"])
        dx = self._dx()
        out = []
        for integrand in ((muinv * ngs.grad(en)) * vt * dx, (muinv * et) * vt * dx):
            form = ngs.BilinearForm(self.space)
            form += integrand
            form.Assemble()
            rows, cols, vals = form.mat.COO()
            n = self.space.ndof
            m = sp.csr_matrix(
                (np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n)
            )
            out.append(m[: self.n_t][:, self.free].tocsr())
        return out[0], out[1]

    @functools.cached_property
    def gauge_forms(self) -> tuple[np.ndarray, np.ndarray]:
        """``l_n`` and ``l_t`` with ``gauge(c, gamma) = -i (l_n + gamma l_t) . c``.

        Free-DOF vectors; see `gauge`.
        """
        _, (vt, vn) = self.space.TnT()
        weight = self._quadrant() * self._coef["nu_tt"]
        t = self._gauge_direction()
        dx = self._dx()
        ln = self._vector((t * (weight * ngs.grad(vn))) * dx)
        lt = self._vector((t * (weight * vt)) * dx)
        return ln, lt

    def _quadrant(self) -> ngs.CoefficientFunction:
        s = self.section
        glob = [
            s.origin[k] + ngs.x * s.frame[0, k] + ngs.y * s.frame[1, k]
            for k in (0, 1, 2)
        ]
        corners = s.to_global(s.points) / s.scale
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        da, db = np.argsort(hi - lo)[::-1][:2]
        ca, cb = 0.5 * (lo + hi)[da], 0.5 * (lo + hi)[db]
        return ngs.IfPos(glob[da] - ca, 1.0, 0.0) * ngs.IfPos(glob[db] - cb, 1.0, 0.0)

    def _gauge_direction(self) -> ngs.CoefficientFunction:
        s = self.section
        return ngs.CF((float(s.frame[0].sum()), float(s.frame[1].sum())))

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
        mu0 = scipy.constants.mu_0
        cf = et * ngs.Conj(self._coef["nu_tt"] * a)
        value = ngs.Integrate(cf, self.mesh, order=quadrature.degree(self.order))
        return 0.5 * self.section.scale * 1j / (omega * mu0) * complex(value)

    def gauge(self, coeffs: np.ndarray, gamma: complex) -> complex:
        """Palace's phase reference: ``int_Q t . (n_out x H) dA`` up to a factor.

        ``Q`` is the upper-right quadrant of the port's bounding box along its
        two largest global extents and ``t`` the in-plane part of (1, 1, 1);
        the factor is real and positive.
        """
        et, en = self._gridfunction(coeffs).components
        a = ngs.grad(en) + gamma * et
        inside = self._quadrant()
        t = self._gauge_direction()
        cf = inside * (t * (-1j * (self._coef["nu_tt"] * a)))
        return complex(
            ngs.Integrate(cf, self.mesh, order=quadrature.degree(self.order))
        )


def _rotate(nu_tt: ngs.CoefficientFunction) -> ngs.CoefficientFunction:
    """``mu_tt^-1 = R nu_tt R^T`` from ``nu_tt = R^T mu_tt^-1 R`` (``R a = a x n``)."""
    r = ngs.CF((0.0, 1.0, -1.0, 0.0), dims=(2, 2))
    return r * nu_tt * r.trans


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
    first = len(groups) + 1
    for g in range(len(section.impedance_terms)):
        m.SetBCName(first + g, f"impedance{g}")
        mask = section.impedance_group == g
        for edge in section.impedance_edges[mask]:
            m.Add(
                ngm.Element1D(  # ty: ignore[unresolved-attribute]
                    [int(edge[0]) + 1, int(edge[1]) + 1], index=first + g + 1
                )
            )
    return ngs.Mesh(m)


def _edge_fill(mesh: ngs.Mesh, section: PortSection) -> ngs.CoefficientFunction:
    """The sheet fields' fill per impedance edge (1 without a sheet field)."""
    if section.impedance_fill is None or (section.impedance_fill == 1.0).all():
        return ngs.CF(1.0)
    fill = {
        (int(min(a, b)), int(max(a, b))): float(w)
        for (a, b), w in zip(
            section.impedance_edges, section.impedance_fill, strict=True
        )
    }
    space = ngs.SurfaceL2(mesh, order=0)
    gf = ngs.GridFunction(space)
    values = gf.vec.FV().NumPy()
    values[:] = 1.0
    for el in mesh.Elements(ngs.BND):
        a, b = sorted(v.nr for v in el.vertices)
        if (a, b) in fill:
            values[space.GetDofNrs(el)[0]] = fill[a, b]
    return gf


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
