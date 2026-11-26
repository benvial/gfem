r"""The section kernel: material derivatives of a port mode's 2D forms.

Given pairs :math:`(q, x)` of 2D vectors on a port section's free DOFs (and
the eigenvalue :math:`\lambda`), `SectionKernel.cotangents` returns, for
every non-shape input the pencil reads, the derivative of

.. math::

    \operatorname{Re}\,q^T\,(\mathrm{d}A - \lambda\,\mathrm{d}B)\,x

by that input, as the cotangent of the coefficient the form list
(`gfem._engine.port_mode.FormItem`) names: with :math:`\mathrm{d}L =
\operatorname{Re}\,\bar c\,\mathrm{d}c`, :math:`\bar c` is the integral over
the triangle (edge) of the form's bare integrand, :math:`q` on the test side
and :math:`x` on the trial side, times the slot's :math:`\theta_j` (and
:math:`-\lambda` on the ``B`` side). Entries of the matrix coefficients
(:math:`\varepsilon_{tt}`, :math:`\sigma_{tt}`, :math:`\nu_{tt}`) are
cotangents of each entry. A perfectly matched layer's stretch multiplies
the coefficients as the forward assembles them, so it is inside the
integrals.

The forms the normalization and the 3D coupling read are not on the list;
`SectionKernel.explicit` gives their :math:`\nu` cotangents (`FormPair`).

`integrated` maps the material cotangents to the 3D owner tetrahedra as an
`gfem._engine.sensitivity.Integrated`, so the existing material chain
(loss tangent, conductivity, `MaterialField`, per axis) applies: the values
and :math:`3\times3` frame moments :math:`F^Tm_{tt}F + nn^Tm_{nn}` of the
driven operator's ``mass``, ``conductivity``, ``curl_curl`` and
``absorbing`` terms. Laws, sheet fills and `Impedance` values return as
direct cotangents by input kind.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Sequence
from typing import Any

import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem._boundaries import sheet_of, surfaces_of
from gfem._engine import quadrature
from gfem._engine.anisotropy import CROSS_N
from gfem._engine.dispersion import d_eps
from gfem._engine.port_mode import (
    PortModeSpace,
    _section_stretch,
)
from gfem._engine.port_mode_adjoint import scalar_cotangents
from gfem._engine.sensitivity import Integrated
from gfem._port_section import PortSection, _edge_keys

__all__ = [
    "BendDerivative",
    "FormPair",
    "SectionCotangents",
    "SectionKernel",
    "boundary_index",
    "edge_triangles",
    "impedance_cotangents",
    "integrated",
    "law_cotangents",
    "owned",
    "owner_tetrahedra",
    "section_inputs",
]

#: The forms `SectionKernel.explicit` differentiates.
EXPLICIT_FORMS = (
    "power_g",
    "power_t",
    "gauge_n",
    "gauge_t",
    "coupling_g",
    "coupling_t",
)


@dataclasses.dataclass(eq=False)
class FormPair:
    r"""A form contracted with a pair of vectors, with a weight.

    :math:`\mathrm{d}L = \operatorname{Re}\,w\,\ell^T\,\mathrm{d}F\,r` for a
    bilinear form of the port section (rows the test side, ``left``) and
    :math:`\operatorname{Re}\,w\,\mathrm{d}F\cdot r` for a linear functional.

    Attributes:
        form: One of `EXPLICIT_FORMS`: ``"power_g"``, ``"power_t"`` (the
            power integral's matrices, over the physical triangles,
            :math:`\bar\nu_{tt}`), ``"gauge_n"``, ``"gauge_t"`` (the phase
            gauge's functionals), ``"coupling_g"``, ``"coupling_t"`` (the
            coupling matrices, :math:`\nu_{tt}`).
        left: The test-side vector: on the free DOFs, or on the Nédélec DOFs
            for a coupling form (its rows); ``None`` for a functional.
        right: The trial-side vector on the free DOFs.
        weight: The complex weight.
        tag: A label of where the pair came from (``"robin"`` for the Robin
            term), which a report of the node cotangent keeps apart.
    """

    form: str
    left: np.ndarray | None
    right: np.ndarray
    weight: complex
    tag: str = ""


@dataclasses.dataclass(eq=False)
class SectionCotangents:
    r"""The cotangents of the coefficients a port section's forms read.

    :math:`\mathrm{d}L = \operatorname{Re}\sum \bar c\,\mathrm{d}c`. The
    ``eps``, ``sigma`` and ``nu`` blocks are those of
    `gfem._port_section.PortSection` (per triangle, ``sigma`` without its
    :math:`\eta_0 s` factor), ``y_tau`` and ``y_n`` the absorbing
    admittances per ``absorbing_edges``, ``fill`` the edge fills per
    ``impedance_edges``.

    Attributes:
        law: ``(n_laws,)`` the cotangent of each law's susceptibility
            scalar :math:`\theta_j`.
        absorbing_law: Likewise of the absorbing edges beside each law that
            has some, by law index.
        impedance: By ``(group, name)``: the cotangent of the weight of an
            `Impedance` group's resistance, inductance or capacitance.
    """

    eps_tt: np.ndarray
    eps_nn: np.ndarray
    sigma_tt: np.ndarray
    sigma_nn: np.ndarray
    nu_tt: np.ndarray
    nu_nn: np.ndarray
    y_tau: np.ndarray
    y_n: np.ndarray
    fill: np.ndarray
    law: np.ndarray
    absorbing_law: dict[int, complex]
    impedance: dict[tuple[int, str], complex]

    @classmethod
    def zeros(cls, section: PortSection) -> SectionCotangents:
        """All cotangents zero, in the shapes of ``section``."""
        n = len(section.triangles)
        edges = len(section.absorbing_edges)
        return cls(
            eps_tt=np.zeros((n, 2, 2), complex),
            eps_nn=np.zeros(n, complex),
            sigma_tt=np.zeros((n, 2, 2), complex),
            sigma_nn=np.zeros(n, complex),
            nu_tt=np.zeros((n, 2, 2), complex),
            nu_nn=np.zeros(n, complex),
            y_tau=np.zeros(edges, complex),
            y_n=np.zeros(edges, complex),
            fill=np.zeros(len(section.impedance_edges), complex),
            law=np.zeros(len(section.laws), complex),
            absorbing_law={},
            impedance={},
        )

    def add(self, other: SectionCotangents) -> None:
        """Add ``other`` in place."""
        for f in dataclasses.fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, dict):
                for k, v in b.items():
                    a[k] = a.get(k, 0) + v
            else:
                a += b


class SectionKernel:
    """The material derivatives of one `PortModeSpace` at a wavenumber.

    Args:
        space: The port mode problem.
        k0: The wavenumber per local unit of the pairs' frequency.
    """

    def __init__(self, space: PortModeSpace, k0: float) -> None:
        self.space = space
        self.k0 = float(k0)
        section = space.section
        self.theta = space.pencil.theta(self.k0)
        self._dx = space._dx()  # noqa: SLF001
        # the metric of a layer or a bend (the masks stay layer-only)
        stretched = (
            section.stretch is not None and section.layer is not None
        ) or section.bend is not None
        if stretched:
            lam_tt, lam_nn = _section_stretch(section)
            self._lam_tt = lam_tt
            self._lam_nn = lam_nn
            self._det = lam_tt[0, 0] * lam_tt[1, 1] - lam_tt[0, 1] * lam_tt[1, 0]
        else:
            self._lam_tt = ngs.CF((1, 0, 0, 1), dims=(2, 2))
            self._lam_nn = ngs.CF(1.0)
            self._det = ngs.CF(1.0)
        self._physical = space._coef["physical"]  # noqa: SLF001
        self._p0 = ngs.L2(space.mesh, order=0, complex=True)
        self._surface = ngs.SurfaceL2(space.mesh, order=0, complex=True)
        self._edge_dof = {}
        for el in space.mesh.Elements(ngs.BND):
            a, b = sorted(v.nr for v in el.vertices)
            self._edge_dof[a, b] = self._surface.GetDofNrs(el)[0]

    def at(self, k0: float) -> SectionKernel:
        """A copy of this kernel at another wavenumber (the forms are shared)."""
        out = copy.copy(self)
        out.k0 = float(k0)
        out.theta = self.space.pencil.theta(out.k0)
        return out

    # --- integrals -------------------------------------------------------------

    def _fields(self, v: np.ndarray, *, nedelec: bool = False) -> tuple[Any, Any]:
        """``(E_t, E_n)`` of a vector on the free DOFs (or the Nédélec ones)."""
        space = self.space
        full = np.zeros(space.space.ndof, complex)
        if nedelec:
            full[: space.n_t] = v
        else:
            full[space.free] = v
        return space._gridfunction(full).components  # noqa: SLF001

    def _per_triangle(self, cfs: Sequence[Any]) -> np.ndarray:
        """The integral of each coefficient function over each triangle."""
        out = np.zeros((len(cfs), self.space.mesh.ne), complex)
        test = self._p0.TestFunction()
        for k, cf in enumerate(cfs):
            form = ngs.LinearForm(self._p0)
            form += cf * test * self._dx
            form.Assemble()
            out[k] = form.vec.FV().NumPy()
        return out

    def _per_edge(self, region: str, cfs: Sequence[Any]) -> dict[tuple, np.ndarray]:
        """The integral of each function over each boundary element of ``region``."""
        q = quadrature.degree(self.space.order)
        ds = ngs.ds(region, intrules={ngs.SEGM: ngs.IntegrationRule(ngs.SEGM, q)})
        test = self._surface.TestFunction()
        forms = []
        for cf in cfs:
            form = ngs.LinearForm(self._surface)
            form += cf * test * ds
            form.Assemble()
            forms.append(form.vec.FV().NumPy().copy())
        return {
            key: np.array([f[dof] for f in forms])
            for key, dof in self._edge_dof.items()
        }

    def _material_integrals(self, left: Any, right: Any) -> dict[str, np.ndarray]:
        """The bare integrals of the volume forms, per triangle.

        ``left`` and ``right`` are ``(E_t, E_n)`` of the test and the trial
        side.
        """
        (lt, ln), (rt, rn) = left, right
        al, ar = lt + ngs.grad(ln), rt + ngs.grad(rn)
        lam = self._lam_tt
        scaled_x = lam * rt
        scaled_q = (1.0 / self._det) * (lam * al)
        cfs = [ngs.curl(lt) * ngs.curl(rt) / self._lam_nn]
        cfs += [lt[i] * scaled_x[j] for i in (0, 1) for j in (0, 1)]
        cfs += [self._lam_nn * ln * rn]
        cfs += [scaled_q[i] * ar[j] for i in (0, 1) for j in (0, 1)]
        v = self._per_triangle(cfs)
        return {
            "nu_nn": v[0],
            "value_tt": v[1:5].T.reshape(-1, 2, 2),
            "value_nn": v[5],
            "nu_tt": v[6:10].T.reshape(-1, 2, 2),
        }

    # --- the pencil's items ------------------------------------------------------

    def cotangents(
        self, pairs: Sequence[tuple[np.ndarray, np.ndarray, complex]]
    ) -> SectionCotangents:
        r"""The cotangents of :math:`\mathrm{Re}\,q^T(\mathrm{d}A-\lambda\mathrm{d}B)x`.

        Args:
            pairs: ``(q, x, lam)`` per pair: vectors on the free DOFs and
                the eigenvalue :math:`\lambda = \gamma^2` the pair belongs
                to (summed over the pairs).
        """
        space, section = self.space, self.space.section
        out = SectionCotangents.zeros(section)
        theta = self.theta
        inside, absorbing = _law_slots(space)
        for q, x, lam in pairs:
            left, right = self._fields(q), self._fields(x)
            m = self._material_integrals(left, right)
            out.nu_nn += theta[0] * m["nu_nn"]
            out.nu_tt += -lam * theta[0] * m["nu_tt"]
            out.eps_tt += theta[1] * m["value_tt"]
            out.eps_nn += -lam * theta[1] * m["value_nn"]
            out.sigma_tt += theta[2] * m["value_tt"]
            out.sigma_nn += -lam * theta[2] * m["value_nn"]
            slots = scalar_cotangents(space.pencil, lam, q, x)
            for g, slot in inside.items():
                out.law[g] += slots[slot]
            for g, slot in absorbing.items():
                out.absorbing_law[g] = out.absorbing_law.get(g, 0) + slots[slot]
            self._edges(out, left, right, lam)
        return out

    def _edges(
        self, out: SectionCotangents, left: Any, right: Any, lam: complex
    ) -> None:
        space, section = self.space, self.space.section
        (lt, ln), (rt, rn) = left, right
        t_cf, n_cf = lt.Trace() * rt.Trace(), ln.Trace() * rn.Trace()
        theta = self.theta
        if len(section.absorbing_edges):
            found = self._per_edge("absorbing.*", [t_cf, n_cf])
            for e, edge in enumerate(section.absorbing_edges):
                key = tuple(sorted(map(int, edge)))
                if key in found and _on_group(section, e):
                    out.y_tau[e] += theta[2] * found[key][0]
                    out.y_n[e] += -lam * theta[2] * found[key][1]
        if not len(section.impedance_edges):
            return
        fill = (
            np.ones(len(section.impedance_edges))
            if section.impedance_fill is None
            else np.asarray(section.impedance_fill, float)
        )
        for g in range(len(section.impedance_terms)):
            found = self._per_edge(f"impedance{g}", [t_cf, n_cf])
            rows = np.flatnonzero(section.impedance_group == g)
            for e in rows:
                key = tuple(sorted(map(int, section.impedance_edges[e])))
                t_int, n_int = found[key]
                edge = t_int - lam * n_int
                slots = _group_slots(space, g)
                for slot, weight, name in slots:
                    if name is None:  # a conductor: the scalar's own slot
                        out.fill[e] += theta[slot] * edge
                        continue
                    out.fill[e] += theta[slot] * weight * edge
                    key2 = (g, name)
                    out.impedance[key2] = out.impedance.get(key2, 0) + (
                        fill[e] * theta[slot] * edge
                    )

    # --- the forms beside the list -------------------------------------------------

    def explicit(self, pairs: Sequence[FormPair]) -> SectionCotangents:
        r"""The :math:`\nu` cotangents of the power, gauge and coupling forms.

        Each `FormPair` contributes :math:`\operatorname{Re}\,w\,\ell^T\,
        \mathrm{d}F\,r`. The power forms carry the conjugate of the
        inverse permeability, so their cotangent is conjugated.
        """
        out = SectionCotangents.zeros(self.space.section)
        for pair in pairs:
            if pair.form not in EXPLICIT_FORMS:
                msg = f"Unknown form {pair.form!r}; the forms are {EXPLICIT_FORMS}."
                raise ValueError(msg)
            nedelec = pair.form.startswith("coupling")
            left = (
                None if pair.left is None else self._fields(pair.left, nedelec=nedelec)
            )
            right = self._fields(pair.right)
            cfs, conj, factor = self._explicit_integrands(pair.form, left, right)
            k = self._per_triangle(cfs)[:4].T.reshape(-1, 2, 2)
            k = np.conj(pair.weight * k) if conj else pair.weight * k
            out.nu_tt += k * factor
        return out

    def _explicit_integrands(
        self, form: str, left: Any, right: Any
    ) -> tuple[list[Any], bool, float]:
        lam, det = self._lam_tt, self._det
        if form in ("gauge_n", "gauge_t"):
            direction = self.space._gauge_direction()  # noqa: SLF001
            quadrant = self.space._quadrant()  # noqa: SLF001
            pt = (1.0 / det) * (lam * direction)
            rt, rn = right
            field = ngs.grad(rn) if form == "gauge_n" else rt
            return (
                [quadrant * pt[i] * field[j] for i in (0, 1) for j in (0, 1)],
                False,
                1.0,
            )
        (lt, _), (rt, rn) = left, right
        field = ngs.grad(rn) if form.endswith("_g") else rt
        pt = (1.0 / det) * (_conj(lam) * lt if form.startswith("power") else lam * lt)
        phys = self._physical if form.startswith("power") else 1.0
        cfs = [phys * pt[i] * field[j] for i in (0, 1) for j in (0, 1)]
        return cfs, form.startswith("power"), 1.0


class BendDerivative:
    r"""The derivative of a bent section's forms by the inverse radius :math:`u`.

    The pencil pieces and the power and gauge forms are differentiated as
    forms, with :math:`u` the section's ``ngs.Parameter`` (`PortModeSpace.bend_u`):
    the coefficient's derivative enters the same integrand the forward
    assembles, so the metric's dependence on :math:`u` through the layer's
    complex coordinate is inside it. A per-triangle chain through the
    coefficient values would miss the part of :math:`h` that varies inside a
    triangle.

    Args:
        space: The port mode problem of a bent section.
    """

    def __init__(self, space: PortModeSpace) -> None:
        u, bend = space.bend_u, space.section.bend
        if u is None or bend is None:
            msg = "BendDerivative needs the space of a bent section."
            raise ValueError(msg)
        self.space = space
        self._u = u
        #: The radius of the bend, m.
        self.radius = float(bend.radius)
        n = len(space.pencil.a)
        forms: dict[str, list[list[Any]]] = {
            side: [[] for _ in range(n)] for side in ("A", "B")
        }
        for item in space.form_items:
            # a law's susceptibility carries the bend's metric as eps does
            if item.input[0] not in ("eps", "sigma", "nu", "law"):
                continue
            moved = dataclasses.replace(item, coefficient=item.coefficient.Diff(u))
            forms[item.side][item.slot].append(space._integrand(moved))  # noqa: SLF001
        self.a = tuple(space._matrix(f) for f in forms["A"])  # noqa: SLF001
        self.b = tuple(space._matrix(f) for f in forms["B"])  # noqa: SLF001
        self._pairs: dict[str, Any] = {}

    def pencil(
        self, theta: Sequence[complex], lam: complex, q: np.ndarray, x: np.ndarray
    ) -> complex:
        r"""The complex :math:`q^T(\partial_u A - \lambda\,\partial_u B)\,x`.

        ``theta`` is the pencil's scalar of each slot at the pair's wavenumber.
        """
        return sum(
            th * (q @ (da @ x) - lam * (q @ (db @ x)))
            for th, da, db in zip(theta, self.a, self.b, strict=True)
        )

    def pair(self, pair: FormPair) -> complex:
        r"""The complex :math:`w\,\ell^T\,\partial_u F\,r` of a power or gauge form."""
        if pair.form not in ("power_g", "power_t", "gauge_n", "gauge_t"):
            msg = f"The form {pair.form!r} has no derivative by the bend radius."
            raise ValueError(msg)
        if pair.form not in self._pairs:
            self._pairs[pair.form] = self._form(pair.form)
        form = self._pairs[pair.form]
        if pair.left is None:
            return complex(pair.weight * (form @ pair.right))
        return complex(pair.weight * (pair.left @ (form @ pair.right)))

    def _form(self, name: str) -> Any:
        r"""The matrix (power) or vector (gauge) of a form, :math:`\partial_u`."""
        space = self.space
        (et, en), (vt, vn) = space.space.TnT()
        nu = space._coef["nu_tt"]  # noqa: SLF001
        dnu = nu.Diff(self._u)
        dx = space._dx()  # noqa: SLF001
        if name.startswith("power"):
            # u is real, so the conjugate of the derivative is the derivative
            # of the conjugate (`Conj` has no `Diff`)
            c = space._coef["physical"] * ngs.Conj(dnu)  # noqa: SLF001
            field = ngs.grad(en) if name.endswith("_g") else et
            return space._matrix([(c * field) * vt * dx])  # noqa: SLF001
        weight = space._quadrant() * dnu  # noqa: SLF001
        t = space._gauge_direction()  # noqa: SLF001
        field = ngs.grad(vn) if name.endswith("_n") else vt
        return space._vector((t * (weight * field)) * dx)  # noqa: SLF001


def _conj(m: Any) -> Any:
    return ngs.Conj(m)


def _law_slots(space: PortModeSpace) -> tuple[dict[int, int], dict[int, int]]:
    """The pencil slot of each law, and of the absorbing edges beside it."""
    inside, absorbing = {}, {}
    for item in space.form_items:
        if item.input[0] == "law" and item.side == "A":
            inside[item.input[1]] = item.slot
        elif item.input[0] == "absorbing_law" and item.side == "A":
            absorbing[item.input[1]] = item.slot
    return inside, absorbing


def _group_slots(space: PortModeSpace, g: int) -> list[tuple[int, float, str | None]]:
    """``(slot, weight, name)`` of the side-A items of an impedance group's edges."""
    out = []
    for item in space.form_items:
        if item.side != "A" or item.kind != "edge_t":
            continue
        if item.input == ("fill", g):
            out.append((item.slot, 1.0, None))
        elif item.input[0] == "impedance" and item.input[1] == g:
            out.append((item.slot, float(item.weight or 1.0), item.input[2]))
    return out


def _on_group(section: PortSection, e: int) -> bool:
    """Whether absorbing edge ``e`` is in a shared group (not beside a law)."""
    law = section.absorbing_law
    return law is None or law[e] < 0


# --- the 3D owner tetrahedra -----------------------------------------------------


def edge_triangles(section: PortSection, edges: np.ndarray) -> np.ndarray:
    """The section triangle each boundary edge (node pair) belongs to."""
    keys = {}
    for t, tri in enumerate(section.triangles):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            keys[tuple(sorted((int(tri[a]), int(tri[b]))))] = t
    return np.array([keys[tuple(sorted(map(int, e)))] for e in edges], dtype=int)


def owner_tetrahedra(model: Any, section: PortSection) -> np.ndarray:
    """The solve-domain tetrahedron behind each triangle of ``section``."""
    from gfem._engine.terms import face_owner

    if section.faces is None:
        msg = "The section does not record its mesh triangles."
        raise ValueError(msg)
    return face_owner(model)[section.faces]


def integrated(
    section: PortSection, cot: SectionCotangents, owners: np.ndarray, n_tets: int
) -> Integrated:
    r"""The material cotangents as per-tetrahedron integrals of the 3D terms.

    ``mass`` (:math:`\varepsilon`), ``conductivity`` (without
    :math:`\eta_0 s`), ``curl_curl`` (:math:`\nu = \mu^{-1}`, the
    inverse-permeability block :math:`R\nu_{tt}R^T`) and ``absorbing`` (the
    admittance of the triangle beside each absorbing edge, isotropic
    material only): each triangle's value adds to its owner. The moments
    are the :math:`3\times3` :math:`F^T\operatorname{sym}(m_{tt})F + nn^Tm_{nn}`
    of the frame :math:`F` (``section.frame[:2]``) and normal :math:`n`;
    their trace is the value. A cotangent here is the complex number whose
    real part, times the coefficient's derivative, is the gradient.

    Raises:
        NotImplementedError: An absorbing edge with a cotangent has an
            anisotropic material beside it.
    """
    f = section.frame[:2]
    n = section.frame[2]
    values: dict[str, np.ndarray] = {}
    moments: dict[str, np.ndarray] = {}
    blocks = {
        "mass": (cot.eps_tt, cot.eps_nn),
        "conductivity": (cot.sigma_tt, cot.sigma_nn),
        "curl_curl": (
            np.einsum("ab,tbc,dc->tad", CROSS_N, cot.nu_tt, CROSS_N),
            cot.nu_nn,
        ),
    }
    on = owners >= 0
    for name, (tt, nn) in blocks.items():
        sym = 0.5 * (tt + np.swapaxes(tt, 1, 2))
        m3 = np.einsum("ia,tab,bj->tij", f.T, sym, f) + nn[:, None, None] * np.outer(
            n, n
        )
        mom = np.zeros((n_tets, 3, 3), complex)
        np.add.at(mom, owners[on], m3[on])
        moments[name] = mom
        values[name] = np.trace(mom, axis1=1, axis2=2)
    absorbing = np.zeros(n_tets, complex)
    if len(cot.y_tau):
        tri = edge_triangles(section, section.absorbing_edges)
        used = (np.abs(cot.y_tau) + np.abs(cot.y_n)) > 0
        iso = _isotropic(section)
        if (used & ~iso[tri]).any():
            msg = (
                "The absorbing-edge admittance of an anisotropic material beside a "
                "wave port's cross-section has no per-tetrahedron derivative here."
            )
            raise NotImplementedError(msg)
        keep = used & (owners[tri] >= 0)
        np.add.at(absorbing, owners[tri][keep], (cot.y_tau + cot.y_n)[keep])
    values["absorbing"] = absorbing
    return Integrated(values=values, moments=moments)


def _isotropic(section: PortSection) -> np.ndarray:
    """Whether each triangle's eps and mu are the same scalar in every direction."""
    eye = np.eye(2)
    eps = np.einsum("tij,ij->t", section.eps_tt, eye) / 2
    nu = np.einsum("tij,ij->t", section.nu_tt, eye) / 2
    return _iso(section, eps, nu, eye)


def _iso(
    section: PortSection, eps: np.ndarray, nu: np.ndarray, eye: np.ndarray
) -> np.ndarray:
    def same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        scale = np.abs(b).reshape(len(b), -1).max(axis=1)
        return np.abs(a - b).reshape(len(a), -1).max(axis=1) <= 1e-12 * scale

    return (
        same(section.eps_tt, eps[:, None, None] * eye)
        & same(section.eps_nn, eps)
        & same(section.nu_tt, nu[:, None, None] * eye)
        & same(section.nu_nn, nu)
    )


# --- laws, fills and impedances -------------------------------------------------------


def law_cotangents(
    section: PortSection, cot: SectionCotangents, omega: float
) -> dict[tuple, float]:
    r"""The gradient by each permittivity-law parameter of the section.

    By input kind ``("law", region, "eps_inf")`` and ``("law", region,
    term_index, name)``: the susceptibility piece's
    :math:`\theta = -k_0^2\chi(\omega)` moves with the term parameters, the
    permittivity of the law's triangles and of the admittance
    :math:`ik_0\sqrt{\operatorname{Re}\varepsilon/\mu}` of its absorbing
    edges with :math:`\varepsilon_\infty` and the real part of :math:`\chi`.
    ``omega`` is the angular frequency (rad/s) of the pairs.
    """
    out: dict[tuple, float] = {}
    k0 = omega * section.scale / scipy.constants.c
    index = None if section.law_index is None else np.asarray(section.law_index)
    for g, (region, law) in enumerate(section.laws):
        d = d_eps(law, omega)
        rows = index == g
        chi_bar = cot.law[g]
        absorbing = cot.absorbing_law.get(g)
        eps_re = float(law.eps_inf) + _chi_real(law, omega)
        mu = 1 / float(section.nu_nn[rows][0].real)
        slope_abs = 0.0 if absorbing is None else 1j * k0 / (2 * np.sqrt(eps_re * mu))
        for key, value in d.items():
            if key == ("eps_inf",):
                direct = (
                    cot.eps_tt[rows][:, 0, 0].sum()
                    + cot.eps_tt[rows][:, 1, 1].sum()
                    + cot.eps_nn[rows].sum()
                )
                total = direct + (0 if absorbing is None else absorbing * slope_abs)
            else:
                total = chi_bar * (-(k0**2)) * value
                if absorbing is not None:
                    total += absorbing * slope_abs * value.real
            out[("law", region, *key)] = float(np.real(total))
    return out


def _chi_real(law: Any, omega: float) -> float:
    from gfem._engine.dispersion import susceptibility

    return float(susceptibility(law, omega).real)


def impedance_cotangents(
    section: PortSection, cot: SectionCotangents
) -> dict[tuple[int, str], float]:
    r"""The gradient by each `Impedance` group's resistance, inductance, capacitance.

    By ``(group, name)``, the derivative by the value in the boundary's own
    units: the weight is :math:`\eta_0/R`, :math:`\mu_0 s/L` and
    :math:`C/(\varepsilon_0 s)`.
    """
    out: dict[tuple[int, str], float] = {}
    s = section.scale
    for (g, name), bar in cot.impedance.items():
        item = section.impedance_terms[g][0]
        z0 = scipy.constants.mu_0 * scipy.constants.c
        if name == "resistance":
            slope = -z0 / item.resistance**2
        elif name == "inductance":
            slope = -scipy.constants.mu_0 * s / item.inductance**2
        else:
            slope = 1 / (scipy.constants.epsilon_0 * s)
        out[g, name] = float(np.real(bar * slope))
    return out


def boundary_index(model: Any, item: Any) -> int:
    """The index in ``model.boundaries`` of the boundary ``item`` stands for.

    The section keeps the boundary objects of the model it was built from, which
    a backward's rebuilt model does not hold: the match is the kind and surface.
    """
    return next(
        i
        for i, b in enumerate(model.boundaries)
        if type(b) is type(item) and surfaces_of(b) == surfaces_of(item)
    )


def owned(cot: SectionCotangents, section: PortSection) -> SectionCotangents:
    r"""``cot`` as the 3D owners' materials read it.

    Without the permittivity of the law regions' triangles: a law's
    :math:`\varepsilon_\infty` goes through the law's own cotangents
    (`law_cotangents`), the region has no ``eps_r`` leaf. Without the
    admittance of an absorbing edge beside an anisotropic material, which has
    no per-tetrahedron derivative: its owner's design is refused before the
    solve, and a constant owner takes none.
    """
    out = dataclasses.replace(cot)
    index = section.law_index
    if index is not None and len(section.laws):
        rows = np.asarray(index) >= 0
        out.eps_tt = np.where(rows[:, None, None], 0.0, cot.eps_tt)
        out.eps_nn = np.where(rows, 0.0, cot.eps_nn)
    if len(cot.y_tau):
        iso = _isotropic(section)[edge_triangles(section, section.absorbing_edges)]
        out.y_tau = np.where(iso, cot.y_tau, 0.0)
        out.y_n = np.where(iso, cot.y_n, 0.0)
    return out


def fill_cotangents(
    model: Any, section: PortSection, cot: SectionCotangents
) -> dict[tuple, Any]:
    """The gradient by each sheet field's fill, per face of its surface.

    Each section impedance edge reads the fill of one face of the sheet's
    surface along it (the last face listing the edge, as
    `gfem._port_section._sheet_fill_by_edge`).
    """
    if section.impedance_fill is None or not len(section.impedance_edges):
        return {}
    mesh = model.mesh
    used = np.unique(mesh.triangles[section.faces])
    out: dict[tuple, Any] = {}
    for g, (item, _) in enumerate(section.impedance_terms):
        if sheet_of(item) is None:
            continue
        (surface,) = surfaces_of(item)
        faces = mesh.elements(surface)
        keys = _edge_keys(mesh.triangles[faces])
        owner = {(int(p), int(q)): k // 3 for k, (p, q) in enumerate(keys)}
        grad = out.setdefault(
            ("boundary", boundary_index(model, item), "fill", "fill"),
            np.zeros(len(faces), complex),
        )
        for e in np.flatnonzero(section.impedance_group == g):
            p, q = sorted(int(v) for v in used[section.impedance_edges[e]])
            grad[owner[p, q]] += cot.fill[e]
    return out


def section_inputs(
    model: Any, section: PortSection, cot: SectionCotangents, n_tets: int
) -> tuple[Integrated, dict[tuple, Any]]:
    """A section's cotangents as owner integrals and cotangents by input kind.

    The integrals are the 3D owners' (`integrated` of `owned`); the kinds are
    the `Impedance` values and the sheet fills. A law's parameters are the
    caller's (`law_cotangents`).
    """
    owners = owner_tetrahedra(model, section)
    explicit: dict[tuple, Any] = fill_cotangents(model, section, cot)
    for (g, name), value in impedance_cotangents(section, cot).items():
        kind = ("boundary", boundary_index(model, section.impedance_terms[g][0]), name)
        explicit[kind] = value
    return integrated(section, owned(cot, section), owners, n_tets), explicit
