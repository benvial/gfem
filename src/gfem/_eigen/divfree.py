"""Divergence-free projection: remove the gradients from an H(curl) vector.

The curl-curl operator vanishes on gradients, so every gradient of an H1
function (zero on the `PEC` surfaces) is an eigenvector of the eigenmode
problem at frequency 0, a huge null space next to the physical modes.
When the target lies below about half a mode's frequency, shift-invert
returns these first. Projecting every Arnoldi vector onto the complement
of the gradients, orthogonal in the electric-energy inner product ``M0``
(``∫ Re(eps_r) u . v``), keeps them out:

    x -> x - G (G^T M0 G)^-1 G^T M0 x

with ``G`` the exact embedding of ``grad H1(p)`` into first-kind Nédélec
of order p and ``G^T M0 G`` the H1 form ``∫ Re(eps_r) grad φ . grad ψ``,
factorized once. A volume of the solve domain without `PEC` has constant
potentials (zero gradient) in the kernel of that form; a mass term about
1e-10 of the stiffness removes them without affecting the projection.
"""

from __future__ import annotations

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from gfem._engine import quadrature
from gfem._linalg import Operator, prepare

__all__ = ["DivFreeProjector"]

#: Relative weight of the mass term that makes the H1 form definite.
_REGULARIZATION = 1e-10


class DivFreeProjector:
    """``x -> x - G (G^T M0 G)^-1 G^T M0 x`` on an H(curl) space.

    Args:
        hcurl: The real H(curl) space (the solve's numbering and `PEC`
            DOFs).
        h1: The real H1 space of the same order and domain, `PEC` DOFs
            fixed.
        eps: ``Re(eps_r)`` per element, an NGSolve coefficient (scalar or
            3x3).
        order: The polynomial order (quadrature).
        diameter: Size of the domain in mesh file units.
    """

    def __init__(
        self,
        hcurl: ngs.FESpace,
        h1: ngs.FESpace,
        eps: ngs.CoefficientFunction,
        order: int,
        diameter: float,
    ) -> None:
        """Build ``G``, ``M0`` and factorize the H1 form."""
        dx = quadrature.dx(order)
        u, v = hcurl.TnT()
        mass = ngs.BilinearForm(hcurl, symmetric=True)
        mass += (eps * u) * v * dx
        mass.Assemble()
        #: ``M0``, the electric-energy form on the real H(curl) space.
        self.mass = mass.mat
        phi = h1.TrialFunction()
        g = ngs.ConvertOperator(h1, hcurl, trial_proxy=ngs.grad(phi))
        rows, cols, values = g.COO()
        self._g = sp.csr_matrix(
            (np.asarray(values), (np.asarray(rows), np.asarray(cols))),
            shape=(g.height, g.width),
        )
        self._g.eliminate_zeros()
        phi, psi = h1.TnT()
        laplace = ngs.BilinearForm(h1, symmetric=True)
        laplace += (eps * ngs.grad(phi)) * ngs.grad(psi) * dx
        laplace += (_REGULARIZATION / diameter**2) * phi * psi * dx
        self._laplace = prepare(
            Operator(laplace, h1.FreeDofs()), symmetry="hermitian", kind="direct"
        )

    def energy(self, x: np.ndarray) -> np.ndarray:
        """``M0 x`` (complex ``x`` part by part)."""
        return _apply(self.mass, x.real) + 1j * _apply(self.mass, x.imag)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """``x`` without its gradient part (``(n,)`` complex)."""
        rhs = self._g.T @ self.energy(x)
        phi = self._laplace.solve(rhs)
        return x - self._g @ phi

    def free(self) -> None:
        """Release the H1 factorization."""
        self._laplace.free()


def _apply(matrix, x: np.ndarray) -> np.ndarray:
    a = matrix.CreateColVector()
    a.FV().NumPy()[:] = x
    b = matrix.CreateColVector()
    b.data = matrix * a
    return b.FV().NumPy().copy()
