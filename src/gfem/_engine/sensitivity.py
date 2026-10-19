"""The sensitivity kernel: field pairs contracted against dA/dp.

Every differentiated solve ends here. A rule's backward pass turns the
output cotangents into **field pairs** ``(left, right)``: the gradient of
``sum_k ct_k y_k`` with respect to a design value ``p`` is
``sum_pairs sum_t theta_t * sum_e (dc_t / dp)_e * a_t,e(left, right)``,
where ``a_t`` is one affine form term of the operator (``grad u . grad v``,
later the curl-curl and mass terms), ``theta_t`` its scalar weight in this
problem (a physical constant, a frequency factor) and ``c_t`` its
coefficient per element (a function of the material values). The kernel
never sees the problem type: it reads which coefficient each property feeds
from `FORM_TERMS`, the property-to-form-term table, and integrates every
term it needs once over all pairs, on the pinned quadrature rule, in one
element-wise (P0) pass.

Adding a problem type means adding table rows (and, for a new operator
term, an entry of `TERMS`), not a code path. Later passes plug in beside
the P0 element pass: a face pass for sheet fields, transport tensors for
node coordinates, and load pairs ``int lambda psi`` for problem-leaf
sources.

Design inputs are named by kind tuples, the same ones `gfem._adjoint` uses:
``("material", region, property)`` is a `MaterialField` leaf and
``("nodes",)`` the mesh node coordinates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Mapping

import ngsolve as ngs
import numpy as np

from gfem._engine import quadrature
from gfem._engine.cache import discretization
from gfem._engine.coefficients import element_values
from gfem.model import Model

__all__ = [
    "FORM_TERMS",
    "TERMS",
    "Coefficient",
    "FieldPair",
    "Term",
    "sensitivity",
    "unsupported",
]

Values = Callable[[str], np.ndarray]


@dataclasses.dataclass(frozen=True)
class Term:
    """One affine form term ``a_t(u, v)`` of an operator.

    Attributes:
        integrand: The integrand, bilinear in two NGSolve fields, without the
            coefficient (``lambda u, v: grad(u) * grad(v)``).
        length_power: The power of the mesh length unit in SI: the integral
            on a mesh in file units times ``scale ** length_power`` is the
            SI value (1 for ``grad . grad`` in 3D, 3 for a mass term).
    """

    integrand: Callable[[ngs.CoefficientFunction, ngs.CoefficientFunction], ngs.CF]
    length_power: int


TERMS: dict[str, Term] = {
    "grad_grad": Term(lambda u, v: ngs.grad(u) * ngs.grad(v), length_power=1),
}


@dataclasses.dataclass(frozen=True)
class Coefficient:
    """How one material property enters one form term's coefficient.

    Attributes:
        term: The form term (a key of `TERMS`).
        derivative: ``dc_t / dp`` per element (or a constant), given the
            element values of every property (``values("eps_r")``).
        holomorphic: Whether ``c_t`` is holomorphic in a complex property
            value. If not, it depends on the real part only (the
            electrostatic permittivity), and the cotangent is real.
    """

    term: str
    derivative: Callable[[Values], np.ndarray | float | complex]
    holomorphic: bool


#: The property-to-form-term table: ``(form, property) -> coefficients``.
#: A ``(form, property)`` pair absent here is read by no term the kernel
#: can differentiate.
FORM_TERMS: dict[tuple[str, str], tuple[Coefficient, ...]] = {
    # a(u, v) = eps0 int Re(eps_r) grad u . grad v
    ("electrostatic", "eps_r"): (
        Coefficient("grad_grad", lambda _: 1.0, holomorphic=False),
    ),
}


@dataclasses.dataclass(frozen=True, eq=False)
class FieldPair:
    """Columns of two fields whose form terms are contracted per element.

    The pair contributes ``sum_k theta_t a_t(left[:, k], right[:, k])`` to
    each term ``t`` in `weights`. Rules fold their output cotangents into
    ``right`` (or ``left``) so a backward pass needs few pairs.

    Attributes:
        space: The NGSolve space of both fields.
        left: DOF vectors, shape ``(ndof,)`` or ``(ndof, k)``.
        right: DOF vectors of the same shape.
        weights: ``theta_t`` per form term in this problem.
        order: The discretization order (fixes the quadrature rule).
        domain: The NGSolve volume region the fields live on.
    """

    space: ngs.FESpace
    left: np.ndarray
    right: np.ndarray
    weights: Mapping[str, complex]
    order: int
    domain: ngs.Region


def unsupported(form: str, kind: tuple) -> str | None:
    """Why the kernel cannot differentiate ``form`` by input ``kind``, or None."""
    if kind[0] == "material":
        if (form, kind[2]) in FORM_TERMS:
            return None
        return f"the {form} forms of this version do not differentiate {kind[2]}"
    if kind[0] == "nodes":
        return (
            "gradients with respect to mesh node coordinates (shape "
            "gradients) are not available in this version of gfem"
        )
    return "this version of gfem has no gradient for this input"


def _columns(a: np.ndarray) -> np.ndarray:
    return a[:, None] if a.ndim == 1 else a


def _gridfunctions(space: ngs.FESpace, vectors: np.ndarray) -> list:
    out = []
    for column in _columns(vectors).T:
        gf = ngs.GridFunction(space)
        gf.vec.FV().NumPy()[:] = column
        out.append(gf)
    return out


def _parts(space: ngs.FESpace, vectors: np.ndarray) -> list[tuple[complex, list]]:
    """``vectors`` as ``[(factor, fields)]``: real fields on a real space."""
    if space.is_complex or not np.iscomplexobj(vectors):
        return [(1.0, _gridfunctions(space, vectors))]
    return [
        (1.0, _gridfunctions(space, vectors.real)),
        (1j, _gridfunctions(space, vectors.imag)),
    ]


def _pair_integrand(term: Term, pair: FieldPair) -> ngs.CoefficientFunction:
    """``sum_k a_t(left_k, right_k)`` as one coefficient function (bilinear)."""
    parts = [
        (a * b) * term.integrand(u, v)
        for a, lefts in _parts(pair.space, pair.left)
        for b, rights in _parts(pair.space, pair.right)
        for u, v in zip(lefts, rights, strict=True)
    ]
    return sum(parts[1:], parts[0])


def _element_pass(model: Model, pairs: list[FieldPair], term: str) -> np.ndarray:
    """``sum_pairs theta_t sum_k a_t,e(left_k, right_k)`` per gfem tetrahedron (SI)."""
    built = discretization(model.mesh).built
    definition = TERMS[term]
    pairs = [p for p in pairs if term in p.weights]
    is_complex = any(
        np.iscomplexobj(p.left)
        or np.iscomplexobj(p.right)
        or np.iscomplexobj(p.weights[term])
        for p in pairs
    )
    p0 = ngs.L2(built.mesh, order=0, complex=is_complex)
    q = p0.TestFunction()
    form = ngs.LinearForm(p0)
    for pair in pairs:
        dx = quadrature.dx(pair.order, definedon=pair.domain)
        form += pair.weights[term] * _pair_integrand(definition, pair) * q * dx
    form.Assemble()
    out = np.zeros(len(model.mesh.tetrahedra), dtype=complex if is_complex else float)
    out[built.tetrahedron_order] = form.vec.FV().NumPy()
    return out * built.scale**definition.length_power


def sensitivity(
    model: Model, pairs: list[FieldPair], wants: set[Hashable], *, form: str
) -> dict[Hashable, np.ndarray]:
    """The cotangent of every wanted design input, from the rule's pairs.

    Args:
        model: The design state (concrete leaves) the pairs were solved on.
        pairs: The rule's field pairs, cotangents folded in.
        wants: Input kinds, ``("material", region, property)``.
        form: The operator family, the first key of `FORM_TERMS`.

    Returns:
        Per wanted kind, one value per tetrahedron of the region (mesh
        order): the real cotangent, or for a holomorphic dependence the
        complex one (``sum ct * dy/dp``).

    Raises:
        NotImplementedError: A wanted kind is not supported (see
            `unsupported`); `gfem._adjoint` checks this before tracing.
    """
    for kind in wants:
        reason = unsupported(form, kind)  # ty: ignore[invalid-argument-type]
        if reason is not None:
            raise NotImplementedError(reason)
    integrals: dict[str, np.ndarray] = {}
    cache: dict[str, np.ndarray] = {}

    def values(name: str) -> np.ndarray:
        if name not in cache:
            cache[name] = element_values(model, name)
        return cache[name]

    per_property: dict[str, np.ndarray] = {}
    for prop in sorted({k[2] for k in wants}):  # ty: ignore[not-subscriptable]
        total: np.ndarray | float = 0.0
        for coefficient in FORM_TERMS[(form, prop)]:
            if coefficient.term not in integrals:
                integrals[coefficient.term] = _element_pass(
                    model, pairs, coefficient.term
                )
            part = coefficient.derivative(values) * integrals[coefficient.term]
            total = total + (part if coefficient.holomorphic else np.real(part))
        per_property[prop] = np.asarray(total)
    return {
        kind: per_property[kind[2]][model.mesh.elements(kind[1])]  # ty: ignore[not-subscriptable]
        for kind in wants
    }
