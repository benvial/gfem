r"""The node derivative of a port mode's 2D forms: the section's shape kernel.

Moving the nodes of a wave port's face moves the points of its 2D section
mesh, which changes every 2D form the port reads: the pencil pieces, the
power forms of the normalization, the coupling forms the 3D operator pairs
with (the Robin term, the S functional, the excitation) and the voltage
functional. `ShapeForm` is the transport of those forms by a displacement
:math:`W` of the section's points, written by hand on the complex fields as
they are. With :math:`G = \nabla W` (:math:`G_{ij} = \partial_j W_i`, :math:`W`
piecewise linear), covariant values :math:`a, b` (the Nédélec trace and
:math:`\nabla e_n`) and a coefficient :math:`C`:

.. math::

    \delta\!\int (Ca)\cdot b = \int \big((Ca\cdot b)\,I - a\otimes C^\mathsf{T}b
    - b\otimes Ca\big) : G,

and for the other kinds, with :math:`\tau` the tangent of an edge,

.. math::

    \delta\!\int c\,\operatorname{curl}a\operatorname{curl}b
    = -\int c\,\operatorname{curl}a\operatorname{curl}b\,\operatorname{tr}G,
    \quad
    \delta\!\int c\,ab = \int c\,ab\,\operatorname{tr}G,

.. math::

    \delta\!\int c\,a_\tau b_\tau\,\mathrm{d}s
    = -\int c\,a_\tau b_\tau\,(\tau\cdot G\tau)\,\mathrm{d}s,
    \quad
    \delta\!\int c\,ab\,\mathrm{d}s = \int c\,ab\,(\tau\cdot G\tau)\,\mathrm{d}s.

A coefficient that is a fixed function of space (the stretch of a perfectly
matched layer) adds :math:`\int ((W\cdot\nabla)C\,a)\cdot b`: the points of
the layer move through it. Every integral runs on the forward's pinned
rule, so the result is the exact derivative of the discrete form.

The voltage functional reads the field at points that stay put while the mesh
moves, so its derivative is Eulerian (`voltage_eulerian`):
:math:`\delta E = -G^\mathsf{T}E - (\nabla E)\,W(p)`. The phase gauge, the
overlap and the polarity read no node derivative: every output that
differentiates under moving nodes is independent of the mode's phase, so the
cotangent on the normalized coordinates is orthogonal to the phase direction.

`chain_to_nodes` carries the section points' cotangent back to the 3D nodes
of the face with the section frame and origin held: the section is invariant
under rigid in-plane motion, so the derivative is exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import ngsolve as ngs
import numpy as np

from gfem._engine import quadrature
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.port_mode_sensitivity import FormPair
from gfem._engine.runtime import serial_spaces
from gfem._port_section import PortSection

__all__ = [
    "SHAPE_FORMS",
    "ShapeForm",
    "chain_to_nodes",
    "voltage_eulerian",
]

#: The `FormPair` forms that carry a node derivative; the gauge forms do not.
SHAPE_FORMS = ("power_g", "power_t", "coupling_g", "coupling_t")

_COORDS = (ngs.x, ngs.y)


class ShapeForm:
    r"""The node cotangent of weighted 2D forms of one port section, summed.

    Each ``add_*`` call adds :math:`\operatorname{Re}(w\,\mathrm{d}F(\ell, r))`
    to one linear form on the section's piecewise linear displacements;
    `result` assembles it into the cotangent of the section's points.

    Args:
        space: The port mode problem.
    """

    def __init__(self, space: PortModeSpace) -> None:
        self.space = space
        self.mesh = space.mesh
        self._q = quadrature.degree(space.order)
        with serial_spaces():
            self._w_space = ngs.VectorH1(self.mesh, order=1)
        self._w = self._w_space.TestFunction()
        self._form = ngs.LinearForm(self._w_space)
        self._empty = True
        section = space.section
        # a layer's or a bend's coefficients move with the position
        self._position = (
            section.stretch is not None and section.layer is not None
        ) or section.bend is not None

    # --- fields ------------------------------------------------------------------

    def fields(self, v: np.ndarray, *, nedelec: bool = False) -> tuple[Any, Any]:
        """``(E_t, E_n)`` of a vector on the free DOFs (or the Nédélec ones)."""
        space = self.space
        full = np.zeros(space.space.ndof, complex)
        if nedelec:
            full[: space.n_t] = v
        else:
            full[space.free] = v
        return space._gridfunction(full).components  # noqa: SLF001

    # --- integrals ----------------------------------------------------------------

    def _dx(self) -> Any:
        return ngs.dx(intrules={ngs.TRIG: ngs.IntegrationRule(ngs.TRIG, self._q)})

    def _ds(self, region: str) -> Any:
        return ngs.ds(
            region, intrules={ngs.SEGM: ngs.IntegrationRule(ngs.SEGM, self._q)}
        )

    @staticmethod
    def _transport(coef: Any, a: Any, b: Any) -> Any:
        r"""The tensor that contracts with G.

        :math:`(C a \cdot b) I - a \otimes C^T b - b \otimes C a`.
        """
        ident = ngs.Id(2)
        if len(getattr(coef, "dims", (1,))) == 2:
            ca, ctb = coef * a, coef.trans * b
            return (ca * b) * ident - ngs.OuterProduct(a, ctb) - ngs.OuterProduct(b, ca)
        return coef * (
            (a * b) * ident - ngs.OuterProduct(a, b) - ngs.OuterProduct(b, a)
        )

    def add_value(
        self,
        coef: Any,
        weight: complex,
        left: tuple[Any, Any],
        right: tuple[Any, Any],
        *,
        kind: str,
        diff: Any = None,
        position: bool = False,
    ) -> None:
        r"""Add :math:`\operatorname{Re}\,w\,\mathrm{d}\!\int (C a)\cdot b`.

        ``kind`` selects the covariant values: ``"value_t"`` has
        :math:`a = r_t`, :math:`b = l_t`; ``"grad_t"`` the full
        :math:`a = r_t + \nabla r_n`, :math:`b = l_t + \nabla l_n`;
        ``"value_n"`` the products :math:`c\,r_n l_n`; ``"curl"`` the
        scalar curls. ``left`` and ``right`` are ``(E_t, E_n)`` of the test
        and the trial side. ``diff(x)`` is the coefficient's derivative by a
        coordinate, read where ``position`` (a stretched coefficient).
        """
        (lt, ln), (rt, rn) = left, right
        g = ngs.grad(self._w)
        tr_g = g[0, 0] + g[1, 1]
        if kind in ("value_t", "grad_t"):
            if kind == "value_t":
                a, b = rt, lt
            else:
                a, b = rt + ngs.grad(rn), lt + ngs.grad(ln)
            tensor = self._transport(coef, a, b)
            weighted = weight * tensor
            # InnerProduct conjugates a complex argument: contract the real part
            total = ngs.InnerProduct(
                g, weighted.real if weighted.is_complex else weighted
            )
            if position:
                total = total + weight * sum(
                    (((diff(x) * a) * b) * self._w[j] for j, x in enumerate(_COORDS)),
                    start=ngs.CF(0.0),
                )
            self._form += total.real * self._dx()
        elif kind in ("value_n", "curl"):
            if kind == "value_n":
                inner = rn * ln
                scalar = coef * inner
                sign = 1.0
            else:
                inner = ngs.curl(rt) * ngs.curl(lt)
                scalar = coef * inner
                sign = -1.0
            total = sign * weight * scalar * tr_g
            if position:
                total = total + weight * sum(
                    ((diff(x) * inner) * self._w[j] for j, x in enumerate(_COORDS)),
                    start=ngs.CF(0.0),
                )
            self._form += total.real * self._dx()
        else:
            msg = f"Unknown form kind {kind!r}."
            raise ValueError(msg)
        self._empty = False

    def add_edge(
        self,
        coef: Any,
        weight: complex,
        left: tuple[Any, Any],
        right: tuple[Any, Any],
        *,
        kind: str,
        region: str,
    ) -> None:
        r"""Add :math:`\operatorname{Re}\,w\,\mathrm{d}\!\int c\,r\,l\,\mathrm{d}s`.

        ``kind`` is ``"edge_t"`` (the traces of the Nédélec parts, a
        product of tangential components) or ``"edge_n"`` (the normal
        components' traces) on the boundary group ``region``.
        """
        (lt, ln), (rt, rn) = left, right
        gs = ngs.grad(self._w).Trace()
        tgt = gs[0, 0] + gs[1, 1]  # the trace of the surface gradient: tau . G tau
        c = 1.0 if coef is None else coef
        if kind == "edge_t":
            integrand = -c * (rt.Trace() * lt.Trace()) * tgt
        elif kind == "edge_n":
            integrand = c * rn.Trace() * ln.Trace() * tgt
        else:
            msg = f"Unknown form kind {kind!r}."
            raise ValueError(msg)
        self._form += (weight * integrand).real * self._ds(region)
        self._empty = False

    # --- the pencil -----------------------------------------------------------------

    def add_pencil(
        self,
        q: np.ndarray,
        x: np.ndarray,
        lam: complex,
        theta: Sequence[complex],
    ) -> None:
        r"""Add :math:`\operatorname{Re}\,q^\mathsf{T}(\delta A-\lambda\,\delta B)x`.

        Walks the form list the pencil is assembled from
        (`PortModeSpace.form_items`), with :math:`A = \sum\theta_j A_j`.
        """
        left, right = self.fields(q), self.fields(x)
        for item in self.space.form_items:
            weight = theta[item.slot] * (1.0 if item.side == "A" else -lam)
            if item.weight is not None:
                weight = weight * item.weight
            if weight == 0:
                continue
            if item.region == "dx":
                stretched = item.input[0] in ("eps", "sigma", "nu")
                self.add_value(
                    item.coefficient,
                    complex(weight),
                    left,
                    right,
                    kind=item.kind,
                    diff=item.coefficient.Diff
                    if self._position and stretched
                    else None,
                    position=self._position and stretched,
                )
            else:
                self.add_edge(
                    item.coefficient,
                    complex(weight),
                    left,
                    right,
                    kind=item.kind,
                    region=item.region,
                )

    # --- the forms beside the pencil -----------------------------------------------

    def add_pair(self, pair: FormPair) -> None:
        r"""Add :math:`\operatorname{Re}\,w\,\ell^\mathsf{T}\,\mathrm{d}F\,r` of a form.

        The power forms carry the conjugate of the inverse permeability
        (a coefficient, so its conjugate is differentiated); the coupling
        forms have the Nédélec DOFs as rows. The gauge forms have no node
        derivative and are skipped.
        """
        if pair.form not in SHAPE_FORMS or pair.left is None:
            return
        space = self.space
        nedelec = pair.form.startswith("coupling")
        left = self.fields(pair.left, nedelec=nedelec)
        rt, rn = self.fields(pair.right)
        nu = space._coef["nu_tt"]  # noqa: SLF001
        if pair.form.startswith("power"):
            phys = space._coef["physical"]  # noqa: SLF001
            coef = phys * ngs.Conj(nu)

            def diff(x: Any) -> Any:  # Conj has no Diff: the coordinate is real
                return phys * ngs.Conj(nu.Diff(x))

        else:
            coef, diff = nu, nu.Diff
        trial = ngs.grad(rn) if pair.form.endswith("_g") else rt
        self.add_value(
            coef,
            complex(pair.weight),
            left,
            (trial, rn),
            kind="value_t",
            diff=diff,
            position=self._position,
        )

    # --- result -----------------------------------------------------------------------

    def result(self) -> np.ndarray:
        """The cotangent of the section's points, ``(n_points, 2)`` per local unit."""
        n = self.mesh.nv
        if self._empty:
            return np.zeros((n, 2))
        self._form.Assemble()
        values = self._form.vec.FV().NumPy().copy()
        return values.reshape(2, -1).T


def voltage_eulerian(
    space: PortModeSpace,
    coefficients: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    voltage_bar: complex,
) -> np.ndarray:
    r"""The node derivative of a voltage functional at fixed path points.

    :math:`V = \sum_q w_q\cdot E(p_q)` with the coefficients of
    :math:`E` held: the points stay put while the mesh moves, so
    :math:`\delta E = -G^\mathsf{T}E - (\nabla E)\,W(p)` at each point. The
    return value is :math:`\operatorname{Re}\,\bar V\,\partial V/\partial P`
    per local unit, over the section's points, ``(n_points, 2)``.

    Args:
        space: The port mode problem.
        coefficients: The field's coefficients on the whole space (the
            free DOFs lifted, `PortModeSpace.full`).
        points: ``(n, 2)`` the path's points in the plane.
        weights: ``(n, 3)`` the path's direction times length at each point.
        voltage_bar: The cotangent of the voltage.
    """
    gf = space._gridfunction(np.asarray(coefficients, complex))  # noqa: SLF001
    et = gf.components[0]
    grad_e = ngs.Grad(et)
    section = space.section
    out = np.zeros((len(section.points), 2))
    for (x, y), w in zip(points, weights, strict=True):
        mip = space.mesh(float(x), float(y))
        if mip.nr < 0:
            continue
        element = space.mesh[ngs.ElementId(ngs.VOL, mip.nr)]
        corners = np.array([v.nr for v in element.vertices])
        p = section.points[corners]
        t = np.array([p[1] - p[0], p[2] - p[0]]).T
        l12 = np.linalg.solve(t, np.array([x, y]) - p[0])
        lam = np.array([1 - l12.sum(), *l12])
        head = np.linalg.inv(t)  # rows: the gradients of the barycentric 2 and 3
        grad_lam = np.vstack([-head.sum(axis=0), head])  # (3, 2)
        e = np.array(et(mip))
        ge = np.array(grad_e(mip)).reshape(2, 2)  # [i, j] = d_j E_i
        wt = w[:2]
        for a in range(3):
            bar = -(wt @ grad_lam[a]) * e - lam[a] * (ge.T @ wt)
            out[corners[a]] += np.real(voltage_bar * bar)
    return out


def chain_to_nodes(
    section: PortSection, triangles: np.ndarray, cot: np.ndarray, n_nodes: int
) -> np.ndarray:
    r"""The 3D nodes' cotangent of a cotangent on the section's points.

    The section's points are :math:`(X/s - o)F_2^\mathsf{T}` in terms of the
    face nodes :math:`X` (metres), with the frame rows :math:`F_2`, the unit
    length :math:`s` and the origin :math:`o` held, so the cotangent of
    :math:`X` is :math:`\bar P F_2 / s`. Rigid in-plane motion leaves the
    section unchanged, so holding the frame is exact.

    Args:
        section: The port section.
        triangles: The mesh's triangles (node indices), whose rows
            ``section.faces`` are the face's.
        cot: ``(n_points, 2)`` the points' cotangent, per local unit.
        n_nodes: The mesh's node count.

    Returns:
        ``(n_nodes, 3)`` per metre.
    """
    if section.faces is None:
        msg = "The section does not record its mesh triangles."
        raise ValueError(msg)
    used = np.unique(np.asarray(triangles)[section.faces])
    out = np.zeros((n_nodes, 3))
    out[used] = (cot @ section.frame[:2]) / section.scale
    return out
