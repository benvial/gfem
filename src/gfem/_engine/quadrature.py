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

__all__ = ["degree", "ds", "dx"]


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
