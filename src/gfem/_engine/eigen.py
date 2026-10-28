"""The NGSolve side of the eigenmode solve (`gfem._eigen` stays NGSolve-free).

`eigen_operator` assembles the eigenmode operator ``T(k) = K + k D + k^2 M
+ A(k)`` from the driven operator's pieces, sorted by `gfem._eigen.operator.
split`: ``D``, ``M`` and each nonpolynomial ``A_p`` become NumPy linear maps
of the H(curl) space, and ``T(k)`` is a `gfem._linalg` handle of the driven
form at ``ω = k c / scale``. `projector` builds the divergence-free
projection (`DivFreeProjector`) and `element_energy` the electric energy of
a mode per element.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import ngsolve as ngs
import numpy as np
import scipy.constants as const
import scipy.sparse as sp

from gfem._eigen.operator import EigenOperator, NonPolynomial, split
from gfem._engine import quadrature, terms
from gfem._engine.cache import discretization
from gfem._engine.coefficients import coefficient, element_tensors
from gfem._engine.domain import domain_key, h1_space, pec_surfaces
from gfem._engine.driven import DrivenOperator, real_space
from gfem._linalg import Handle, Operator, prepare
from gfem._linalg.base import free_mask
from gfem.model import Model

__all__ = [
    "DivFreeProjector",
    "assemble",
    "eigen_operator",
    "element_energy",
    "matvec",
    "projector",
]

#: Relative weight of the mass term that makes the H1 form definite.
_REGULARIZATION = 1e-10


def eigen_operator(
    op: DrivenOperator,
    sigma: float,
    *,
    kind: str,
    device: str,
    threads: int | None,
    tolerance: float,
) -> EigenOperator:
    """The eigen operator ``T(k)`` of the driven operator, shifted at ``sigma``.

    Nothing is factorized until the handle is first used. ``kind``,
    ``device``, ``threads`` and ``tolerance`` configure the `gfem._linalg`
    backend of every shift.
    """
    space, scale = op.space, op.built.scale
    classes = split(op.pieces, scale, sigma)
    powers = classes.powers
    free = free_mask(space.FreeDofs(), space.ndof)

    def operator(k: complex) -> Operator:
        return Operator(op.form(k * const.c / scale), space.FreeDofs())

    def prepare_shift(t: Operator) -> Handle:
        """Prepare ``t``: ``T(k)`` at one shift (`operator`)."""
        return prepare(
            t,
            symmetry="complex_symmetric",
            kind=kind,
            device=device,
            threads=threads,
            tolerance=tolerance,
            width=1,
        )

    return EigenOperator(
        shift=sigma,
        operator=operator,
        prepare=prepare_shift,
        damping=(
            _linear_map(assemble(space, powers[1], matrix_free=True))
            if powers[1]
            else None
        ),
        mass=_linear_map(assemble(space, powers[2])),
        free=free,
        scale=scale,
        nonpolynomial=tuple(
            NonPolynomial.from_piece(
                piece,
                _linear_map(assemble(space, [(1.0, piece)], matrix_free=True)),
                scale,
            )
            for piece in classes.nonpolynomial
        ),
    )


def assemble(
    space: ngs.FESpace,
    items: list[tuple[complex, terms.Piece]],
    *,
    matrix_free: bool = False,
):
    """``Σ c_p a_p`` as an NGSolve operator.

    Assembled, or applied element by element when ``matrix_free`` (no
    matrix stored: the damping is mostly boundary terms, which an assembled
    matrix would hold in a graph of the whole space).
    """
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True, nonassemble=matrix_free)
    for c, piece in items:
        form += c * piece.integrand(u, v)
    if not matrix_free:
        form.Assemble()
    return form.mat


def matvec(matrix, x: np.ndarray) -> np.ndarray:
    """An NGSolve matrix applied to a NumPy vector."""
    a = matrix.CreateColVector()
    a.FV().NumPy()[:] = x
    b = matrix.CreateColVector()
    b.data = matrix * a
    return b.FV().NumPy().copy()


def _linear_map(matrix) -> Callable[[np.ndarray], np.ndarray]:
    """``x -> matrix x`` on NumPy vectors."""
    return functools.partial(matvec, matrix)


def projector(model: Model, op: DrivenOperator) -> DivFreeProjector:
    """The divergence-free projection on ``op``'s space (`DivFreeProjector`).

    The real H(curl) space has the solve's DOF numbering
    (`gfem._engine.driven.real_space`); the H1 space of the same order fixes
    the same `PEC` surfaces.
    """
    built, order = op.built, op.order
    pec = pec_surfaces(model)
    h1 = discretization(model.mesh).space(
        (domain_key(model), "h1", order, pec),
        lambda: h1_space(model, built, order, pec),
    )
    eps = coefficient(built, np.real(element_tensors(model, "eps_r")))
    nodes = np.asarray(model.mesh.nodes) / built.scale
    diameter = float(np.linalg.norm(np.ptp(nodes, axis=0)))
    return DivFreeProjector(real_space(model, op), h1, eps, order, diameter)


def element_energy(
    model: Model, op: DrivenOperator
) -> Callable[[np.ndarray], np.ndarray]:
    """``x -> ∫ Re(eps_r) x . conj(x)`` per element of ``op``'s space.

    The returned function takes a coefficient vector of ``op.space`` and
    gives the integral over each tetrahedron in the engine mesh's element
    order (`gfem._engine.mesh.EngineMesh.tetrahedron_order` maps it to the
    mesh's), in the mesh's file units, at quadrature order ``2 order``.
    """
    built = op.built
    eps = coefficient(built, np.real(element_tensors(model, "eps_r")))
    gf = ngs.GridFunction(op.space)

    def density(x: np.ndarray) -> np.ndarray:
        gf.vec.FV().NumPy()[:] = x
        values = ngs.Integrate(
            (eps * gf) * ngs.Conj(gf),
            built.mesh,
            order=2 * op.order,
            element_wise=True,
        )
        return values.NumPy()

    return density


class DivFreeProjector:
    """``x -> x - G (G^T M0 G)^-1 G^T M0 x`` on an H(curl) space.

    The curl-curl operator vanishes on gradients, so every gradient of an
    H1 function (zero on the `PEC` surfaces) is an eigenvector of the
    eigenmode problem at frequency 0, a huge null space next to the
    physical modes. When the target lies below about half a mode's
    frequency, shift-invert returns these first. Projecting every Arnoldi
    vector onto the complement of the gradients, orthogonal in the
    electric-energy inner product ``M0`` (``∫ Re(eps_r) u . v``), keeps
    them out, with ``G`` the exact embedding of ``grad H1(p)`` into
    first-kind Nédélec of order p and ``G^T M0 G`` the H1 form ``∫ Re(eps_r)
    grad φ . grad ψ``, factorized once. A volume of the solve domain without
    `PEC` has constant potentials (zero gradient) in the kernel of that
    form; a mass term about 1e-10 of the stiffness removes them without
    affecting the projection.

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
        # G^T as CSR, so a product with it runs row by row like G's
        self._gt = self._g.T.tocsr()
        phi, psi = h1.TnT()
        laplace = ngs.BilinearForm(h1, symmetric=True)
        laplace += (eps * ngs.grad(phi)) * ngs.grad(psi) * dx
        laplace += (_REGULARIZATION / diameter**2) * phi * psi * dx
        self._laplace = prepare(
            Operator(laplace, h1.FreeDofs()), symmetry="hermitian", kind="direct"
        )

    def energy(self, x: np.ndarray) -> np.ndarray:
        """``M0 x`` (complex ``x`` part by part)."""
        return matvec(self.mass, x.real) + 1j * matvec(self.mass, x.imag)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """``x`` without its gradient part (``(n,)`` complex).

        The real ``G`` and ``G^T`` act on the real and imaginary parts
        separately: a product with a complex vector would copy ``G`` to
        complex on every call. The result is bit-identical.
        """
        energy = self.energy(x)
        rhs = self._gt @ energy.real + 1j * (self._gt @ energy.imag)
        phi = self._laplace.solve(rhs)
        return x - (self._g @ phi.real + 1j * (self._g @ phi.imag))

    def free(self) -> None:
        """Release the H1 factorization."""
        self._laplace.free()
