"""The pinned quadrature rule every form uses.

NGSolve picks a form's quadrature order from its test and trial functions,
so a forward form and the sensitivity form of the same term (a P0 test
function against two fields) would be integrated by different rules, and
the discrete adjoint would no longer match the discrete forward (30-50 %
gradient errors). gfem therefore integrates every volume and surface form of
a discretization of polynomial order p with one fixed rule of degree 2p:
exact for every mass and stiffness term of order-p fields with element-wise
constant coefficients on straight-sided elements.
"""

from __future__ import annotations

import ngsolve as ngs

__all__ = ["degree", "ds", "dx", "dx_derivatives"]


def degree(order: int) -> int:
    """The quadrature degree of a discretization of polynomial ``order``."""
    return 2 * order


def dx(order: int, **kwargs) -> ngs.comp.DifferentialSymbol:
    """Volume integration on the pinned tetrahedron rule of ``order``."""
    rule = ngs.IntegrationRule(ngs.ET.TET, degree(order))
    return ngs.dx(intrules={ngs.ET.TET: rule}, **kwargs)


def ds(order: int, **kwargs) -> ngs.comp.DifferentialSymbol:
    """Surface integration on the pinned triangle rule of ``order``."""
    rule = ngs.IntegrationRule(ngs.ET.TRIG, degree(order))
    return ngs.ds(intrules={ngs.ET.TRIG: rule}, **kwargs)


def dx_derivatives(order: int, **kwargs) -> ngs.comp.DifferentialSymbol:
    """Volume integration of products of two first derivatives, exactly.

    A product of two curls (or gradients) of order-p fields, times
    element-wise constants, is a polynomial of degree 2(p - 1) on a
    straight-sided element: the rule of that degree integrates it exactly,
    so it gives the pinned rule's values to round-off, at a fraction of its
    points. The shape-gradient kernel uses it for its curl and gradient
    terms (the transport tensor of ``curl u . curl v`` is such a product).
    """
    rule = ngs.IntegrationRule(ngs.ET.TET, max(0, degree(order) - 2))
    return ngs.dx(intrules={ngs.ET.TET: rule}, **kwargs)
