"""The sensitivity kernel: field pairs contracted against dA/dp.

Every differentiated solve ends here. A rule's backward pass turns the
output cotangents into **field pairs** ``(left, right)``: the gradient of
``sum_k ct_k y_k`` with respect to a design value ``p`` is
``sum_pairs sum_t theta_t * sum_e (dc_t / dp)_e * a_t,e(left, right)``,
where ``a_t`` is one affine form term of the operator (``grad u . grad v``,
the driven curl-curl, mass and absorbing terms), ``theta_t`` its scalar
weight in this problem (a physical constant, a frequency factor) and
``c_t`` its coefficient per element (a function of the material values).
The kernel never sees the problem type: it reads which coefficient each
property feeds from `FORM_TERMS`, the property-to-form-term table, and
integrates every term it needs once over all pairs, on the pinned
quadrature rule, in one element-wise (P0) pass.

A **face term** (an absorbing boundary, whose coefficient is read from the
tetrahedron behind each face) is integrated per triangle in a face-wise
(surface P0) pass, and each face's integral is added to its owning
tetrahedron, so its coefficient rows read like a volume term's.

Adding a problem type means adding table rows (and, for a new operator
term, an entry of `TERMS`), not a code path. Later passes plug in beside
these: a face pass for sheet fields, transport tensors for node
coordinates, and load pairs ``int lambda psi`` for problem-leaf sources.

Design inputs are named by kind tuples, the same ones `gfem._adjoint` uses:
``("material", region, property)`` is a `MaterialField` leaf and
``("nodes",)`` the mesh node coordinates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Mapping
from typing import Any

import ngsolve as ngs
import numpy as np

from gfem._engine import quadrature, terms
from gfem._engine.cache import discretization
from gfem._engine.coefficients import element_values
from gfem._engine.domain import boundary_region
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
Faces = Callable[[Model], tuple[tuple[str, ...], np.ndarray]]


@dataclasses.dataclass(frozen=True)
class Term:
    """One affine form term ``a_t(u, v)`` of an operator.

    Attributes:
        integrand: The integrand, bilinear in two NGSolve fields, without the
            coefficient (``lambda u, v: grad(u) * grad(v)``).
        length_power: The power of the mesh length unit in SI: the integral
            on a mesh in file units times ``scale ** length_power`` is the
            SI value (1 for ``grad . grad`` in 3D, 3 for a mass term). The
            driven terms are in the driven equation's own units (file
            units, see `gfem._engine.terms`): 0, or 1 where the coefficient
            carries the length scale.
        faces: For a face term, ``faces(model) -> (surfaces, owner)``: the
            surfaces it is integrated on and, per gfem triangle, the
            tetrahedron whose values its coefficient reads (-1 for none).
            ``None`` for a volume term.
    """

    integrand: Callable[[ngs.CoefficientFunction, ngs.CoefficientFunction], ngs.CF]
    length_power: int
    faces: Faces | None = None


def _dot(u: ngs.CoefficientFunction, v: ngs.CoefficientFunction) -> ngs.CF:
    return u * v


def _tangential(u: Any, v: Any) -> Any:
    """``u_t . v_t`` on a surface (an H(curl) field's trace)."""
    return u.Trace() * v.Trace()


def _second_order_faces(model: Model) -> tuple[tuple[str, ...], np.ndarray]:
    return terms.absorbing_faces(model, order=2)


TERMS: dict[str, Term] = {
    "grad_grad": Term(lambda u, v: ngs.grad(u) * ngs.grad(v), length_power=1),
    # the driven pieces of `gfem._engine.terms`
    "curl_curl": Term(lambda u, v: ngs.curl(u) * ngs.curl(v), length_power=0),
    "mass": Term(_dot, length_power=0),
    "conductivity": Term(_dot, length_power=1),  # coefficient eta0 sigma scale
    "absorbing": Term(_tangential, length_power=0, faces=terms.absorbing_faces),
    "absorbing order 2": Term(
        lambda u, v: ngs.curl(u).Trace() * ngs.curl(v).Trace(),
        length_power=0,
        faces=_second_order_faces,
    ),
}


@dataclasses.dataclass(frozen=True)
class Coefficient:
    """How one material property enters one form term's coefficient.

    Attributes:
        term: The form term (a key of `TERMS`).
        derivative: ``dc_t / dp`` per element (or a constant), given the
            element values of every property (``values("eps_r")``).
        holomorphic: Whether ``c_t`` is holomorphic in a complex property
            value. If not, it is a function of one real quantity ``r`` (the
            electrostatic permittivity reads ``r = Re(eps_r)``), and
            ``derivative`` is ``dc_t / dr``.
        chain: For a non-holomorphic ``c_t``, ``r = Re(chain * p)`` for a
            complex value ``p`` (``None``: ``chain = 1``). The cotangent is
            then ``chain * Re(sum ct * dy/dr)``, which is JAX's convention
            (``dJ/dRe(p) - i dJ/dIm(p)``); real when ``chain`` is 1.
    """

    term: str
    derivative: Callable[[Values], np.ndarray | float | complex]
    holomorphic: bool
    chain: Callable[[Values], np.ndarray] | None = None


def _loss(values: Values) -> np.ndarray:
    """``1 - i tan_delta``: the driven mass coefficient per unit ``eps_r``."""
    return 1.0 - 1j * np.real(values("tan_delta"))


def _boundary_eps(values: Values) -> np.ndarray:
    """What the absorbing terms read: ``Re(eps_r (1 - i tan_delta))``."""
    return np.real(values("eps_r") * _loss(values))


def _boundary_mu(values: Values) -> np.ndarray:
    return np.real(values("mu_r"))


# The absorbing coefficients of the real e and m above: first order
# c1 = sqrt(e / m), second order c2 = 1 / (m sqrt(e m)).
def _d_c1_d_eps(v: Values) -> np.ndarray:
    return 0.5 / np.sqrt(_boundary_eps(v) * _boundary_mu(v))


def _d_c2_d_eps(v: Values) -> np.ndarray:
    return -0.5 * (_boundary_eps(v) * _boundary_mu(v)) ** -1.5


def _d_c1_d_mu(v: Values) -> np.ndarray:
    return -0.5 * np.sqrt(_boundary_eps(v)) * _boundary_mu(v) ** -1.5


def _d_c2_d_mu(v: Values) -> np.ndarray:
    return -1.5 * _boundary_eps(v) ** -0.5 * _boundary_mu(v) ** -2.5


#: The property-to-form-term table: ``(form, property) -> coefficients``.
#: A ``(form, property)`` pair absent here is read by no term the kernel
#: can differentiate.
#: Rows read isotropic element values: design inputs are `MaterialField`
#: leaves, which are isotropic. A `Material` with per-axis eps_r / tan_delta
#: is a constant (its tensor enters the forward pieces only), and the values
#: its elements show here (means) are never differentiated.
FORM_TERMS: dict[tuple[str, str], tuple[Coefficient, ...]] = {
    # a(u, v) = eps0 int Re(eps_r) grad u . grad v
    ("electrostatic", "eps_r"): (
        Coefficient("grad_grad", lambda _: 1.0, holomorphic=False),
    ),
    # driven: mass eps_r (1 - i tan_delta), curl-curl 1 / mu_r, conductivity
    # eta0 sigma scale (real sigma), absorbing c1 and c2 of the real e and m
    ("driven", "eps_r"): (
        Coefficient("mass", _loss, holomorphic=True),
        Coefficient("absorbing", _d_c1_d_eps, holomorphic=False, chain=_loss),
        Coefficient("absorbing order 2", _d_c2_d_eps, holomorphic=False, chain=_loss),
    ),
    ("driven", "tan_delta"): (
        Coefficient("mass", lambda v: -1j * v("eps_r"), holomorphic=False),
        Coefficient(
            "absorbing",
            lambda v: _d_c1_d_eps(v) * np.imag(v("eps_r")),
            holomorphic=False,
        ),
        Coefficient(
            "absorbing order 2",
            lambda v: _d_c2_d_eps(v) * np.imag(v("eps_r")),
            holomorphic=False,
        ),
    ),
    ("driven", "mu_r"): (
        Coefficient("curl_curl", lambda v: -(v("mu_r") ** -2.0), holomorphic=True),
        Coefficient("absorbing", _d_c1_d_mu, holomorphic=False),
        Coefficient("absorbing order 2", _d_c2_d_mu, holomorphic=False),
    ),
    ("driven", "conductivity"): (
        Coefficient("conductivity", lambda _: terms.ETA0, holomorphic=False),
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
    """``sum_pairs theta_t sum_k a_t,e(left_k, right_k)`` per gfem tetrahedron (SI).

    A face term's integrals per triangle are summed into each triangle's
    owning tetrahedron.
    """
    built = discretization(model.mesh).built
    definition = TERMS[term]
    pairs = [p for p in pairs if term in p.weights]
    is_complex = any(
        np.iscomplexobj(p.left)
        or np.iscomplexobj(p.right)
        or np.iscomplexobj(p.weights[term])
        for p in pairs
    )
    out = np.zeros(len(model.mesh.tetrahedra), dtype=complex if is_complex else float)
    if not pairs:
        return out
    if definition.faces is None:
        p0 = ngs.L2(built.mesh, order=0, complex=is_complex)
        q = p0.TestFunction()
        form = ngs.LinearForm(p0)
        for pair in pairs:
            dx = quadrature.dx(pair.order, definedon=pair.domain)
            form += pair.weights[term] * _pair_integrand(definition, pair) * q * dx
        form.Assemble()
        out[built.tetrahedron_order] = form.vec.FV().NumPy()
        return out * built.scale**definition.length_power
    surfaces, owner = definition.faces(model)
    if not surfaces:
        return out
    region = boundary_region(built, surfaces)
    p0 = ngs.SurfaceL2(built.mesh, order=0, complex=is_complex)
    q = p0.TestFunction()
    form = ngs.LinearForm(p0)
    for pair in pairs:
        ds = quadrature.ds(pair.order, definedon=region)
        form += pair.weights[term] * _pair_integrand(definition, pair) * q * ds
    form.Assemble()
    per_face = np.zeros(len(owner), dtype=out.dtype)
    per_face[built.triangle_order] = form.vec.FV().NumPy()
    mesh = model.mesh
    tags = [mesh.region(s).tag for s in surfaces]
    on = np.isin(mesh.triangle_tags, tags) & (owner >= 0)
    np.add.at(out, owner[on], per_face[on])
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
        order): the real cotangent, or for a complex dependence the complex
        one (``sum ct * dy/dp`` when holomorphic, see `Coefficient`).

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
            if not coefficient.holomorphic:
                part = np.real(part)
                if coefficient.chain is not None:
                    part = coefficient.chain(values) * part
            total = total + part
        per_property[prop] = np.asarray(total)
    return {
        kind: per_property[kind[2]][model.mesh.elements(kind[1])]  # ty: ignore[not-subscriptable]
        for kind in wants
    }
