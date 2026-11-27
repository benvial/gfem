"""The pinned quadrature rule every form uses.

NGSolve picks a form's quadrature order from its test and trial functions,
so a forward form and the sensitivity form of the same term (a P0 test
function against two fields) would be integrated by different rules, and
the discrete adjoint would no longer match the discrete forward (30-50 %
gradient errors). gfem therefore integrates every volume and surface form of
a discretization of polynomial order p with one fixed rule of degree 2p:
exact for every mass and stiffness term of order-p fields with element-wise
constant coefficients on straight-sided elements. On a curved element the
integrands are rational in the reference coordinates and no rule is exact:
the rule is then a definition shared by the forward form, the adjoint and the
shape forms, and the shape gradient is the exact derivative of the
discretized forward form only because all three use it.

NGSolve applies a rule only to the element type it is keyed by, and silently
integrates other elements at its own order. Each rule is therefore keyed by
the volume and surface element types of both mesh dimensions: tetrahedra and
triangles on a tetrahedral mesh, triangles and segments on a 2D mesh.
"""

from __future__ import annotations

import ngsolve as ngs

__all__ = ["degree", "ds", "dx", "dx_derivatives"]


def degree(order: int) -> int:
    """The quadrature degree of a discretization of polynomial ``order``."""
    return 2 * order


#: The volume element types (a tetrahedral mesh's, a 2D mesh's).
_VOLUME = (ngs.ET.TET, ngs.ET.TRIG)
#: The surface element types, in the same order.
_SURFACE = (ngs.ET.TRIG, ngs.ET.SEGM)


def _rules(types: tuple, q: int) -> dict:
    """The rule of degree ``q`` keyed by each element type of ``types``."""
    return {t: ngs.IntegrationRule(t, q) for t in types}


def dx(order: int, **kwargs) -> ngs.comp.DifferentialSymbol:
    """Volume integration on the pinned tetrahedron or triangle rule of ``order``."""
    return ngs.dx(intrules=_rules(_VOLUME, degree(order)), **kwargs)


def ds(order: int, **kwargs) -> ngs.comp.DifferentialSymbol:
    """Surface integration on the pinned triangle or segment rule of ``order``."""
    return ngs.ds(intrules=_rules(_SURFACE, degree(order)), **kwargs)


def dx_derivatives(order: int, **kwargs) -> ngs.comp.DifferentialSymbol:
    """Volume integration of products of two first derivatives, exactly.

    A product of two curls (or gradients) of order-p fields, times
    element-wise constants, is a polynomial of degree 2(p - 1) on a
    straight-sided element: the rule of that degree integrates it exactly,
    so it gives the pinned rule's values to round-off, at a fraction of its
    points. The shape-gradient kernel uses it for its curl and gradient
    terms (the transport tensor of ``curl u . curl v`` is such a product).
    It is exact only on straight elements: on a curved mesh the shape forms
    use `dx`, the forward's own rule.
    """
    return ngs.dx(intrules=_rules(_VOLUME, max(0, degree(order) - 2)), **kwargs)
