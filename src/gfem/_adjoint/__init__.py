"""Solve-level discrete adjoints: the generic JAX ``custom_vjp`` wrapper."""

from gfem._adjoint.solve_vjp import adjoint_tolerance, solve_vjp

__all__ = ["adjoint_tolerance", "solve_vjp"]
