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

**Node coordinates** (shape gradients) are a third pass over the same
pairs, against the rule's `ShapeTerm` list, the operator's pieces with
their coefficients. Under a node displacement ``W`` (a real P1 vector
field, one test function per node and axis) a volume term changes by
``int T(u, v) : grad W``, with hand-written **transport tensors** ``T`` on
the complex fields themselves (gradients and H(curl) values transform
covariantly, curls contravariantly; element-wise constant coefficients
travel with their elements):

- value, ``int (C u) . v``: ``T = (C u . v) I - u (x) C^T v - v (x) C u``;
- curl, ``int (C curl u) . curl v``: ``T = C^T cv (x) cu + C cu (x) cv -
  (C cu . cv) I``;
- gradient (H1), ``int (C grad u) . grad v``: the value tensor of the
  gradients.

Surface terms of the two shapes the operators have, tangential traces
``int c u_t . v_t`` and normal curls ``int c (n . curl u)(n . curl v)``,
use the same kind of hand-written transport on their faces (see
`_trace_shape`): on gsim meshes the metal shells carry as many faces as the
volume has elements, so they are not cheap. Any other surface form is
differentiated by NGSolve's ``DiffShape``, on the real Lagrangian
``Re(theta a(l, r))`` written with real fields only (``DiffShape``
mistreats an H(curl) field multiplied by a complex number before its dot
product); the trace terms are checked against it. Every form is integrated exactly
on straight-sided elements: on the pinned rule of degree 2p, or, for the
curl and gradient terms (products of first derivatives, degree 2p - 2), on
the exact rule of that degree, which gives the same values with fewer
points.

A **sheet field** (`gfem.SheetField`, a fill per face of a `Conductor` or
`Impedance` surface) multiplies the coefficient of each of its boundary's
pieces face by face, ``c_p,f = fill_f c_p``, so ``dA / dfill_f = sum_p
theta_p c_p B_f`` with ``B_f`` the face's tangential trace form. Its
**face pass** integrates ``sum_p theta_p c_p left_t . right_t`` per
triangle of the surface (surface P0, the pinned rule) and returns one value
per face, with no owner tetrahedron: the fill is a property of the face.

Adding a problem type means adding table rows (and, for a new operator
term, an entry of `TERMS`), not a code path. A later pass plugs in beside
these: load pairs ``int lambda psi`` for problem-leaf sources.

Design inputs are named by kind tuples, the same ones `gfem._adjoint` uses:
``("material", region, property)`` is a `MaterialField` leaf,
``("boundary", index, "fill", "fill")`` the fill of the sheet field of
``model.boundaries[index]`` and ``("nodes",)`` the mesh node coordinates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Literal

import ngsolve as ngs
import numpy as np

from gfem._boundaries import Conductor, Impedance
from gfem._engine import quadrature, terms
from gfem._engine.cache import discretization
from gfem._engine.coefficients import element_values
from gfem._engine.domain import boundary_region
from gfem.model import Model

__all__ = [
    "FORM_TERMS",
    "SHAPE_FORMS",
    "SHEET_FORMS",
    "TERMS",
    "Coefficient",
    "FieldPair",
    "ShapeTerm",
    "Term",
    "is_sheet",
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
# eigenmode: the driven pieces (`gfem._eigen.operator`), weighted at the
# complex eigenvalue
FORM_TERMS |= {
    ("eigenmode", prop): FORM_TERMS[("driven", prop)]
    for prop in ("eps_r", "tan_delta", "mu_r", "conductivity")
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


@dataclasses.dataclass(frozen=True, eq=False)
class ShapeTerm:
    """One operator term ``a_t(u, v)`` as the node-coordinate pass reads it.

    Attributes:
        name: The term's key in the pairs' `FieldPair.weights`.
        kind: ``"gradient"`` (``int (C grad u) . grad v``, H1), ``"value"``
            (``int (C u) . v``, H(curl)), ``"curl"`` (``int (C curl u) .
            curl v``), ``"trace"`` (``int c u_t . v_t`` on a surface),
            ``"curl trace"`` (``int c (n . curl u)(n . curl v)``), or
            ``"surface"`` (any other surface form, differentiated by
            ``DiffShape``).
        coefficient: For a volume kind, ``C``: an element-wise constant
            NGSolve coefficient, scalar or 3 x 3, real or complex, the same
            the forward form reads; for a trace kind, the face-wise
            constant scalar ``c``.
        region: For a trace kind, the boundary region it is integrated on.
        integrand: For ``"surface"``: ``(u, v) -> integrals`` of the form
            with its coefficient and pinned quadrature (real coefficients),
            for real fields ``u`` and ``v``.
        space: For ``"surface"``: a real space with the pairs' DOF numbering.
        length_power: The integral on the mesh in file units times
            ``scale ** length_power`` is the term in the units the pairs'
            weights expect (as `Term.length_power`).
    """

    name: str
    kind: Literal["gradient", "value", "curl", "trace", "curl trace", "surface"]
    coefficient: Any = None
    region: Any = None
    integrand: Callable[[Any, Any], Any] | None = None
    space: ngs.FESpace | None = None
    length_power: int = 0


#: Operator families whose rules provide `ShapeTerm` lists (``shape_terms``).
SHAPE_FORMS = frozenset({"electrostatic", "driven", "eigenmode"})
#: Operator families whose pairs carry the sheet pieces' weights (by name).
SHEET_FORMS = frozenset({"driven", "eigenmode"})


def is_sheet(kind: tuple) -> bool:
    """Whether input ``kind`` is the fill of a boundary's sheet field."""
    return kind[0] == "boundary" and tuple(kind[2:]) == ("fill", "fill")


def unsupported(form: str, kind: tuple) -> str | None:
    """Why the kernel cannot differentiate ``form`` by input ``kind``, or None."""
    if is_sheet(kind):
        return (
            None
            if form in SHEET_FORMS
            else f"the {form} forms of this version do not read sheet fields"
        )
    if kind[0] == "material":
        if (form, kind[2]) in FORM_TERMS:
            return None
        return f"the {form} forms of this version do not differentiate {kind[2]}"
    if kind[0] == "nodes":
        if form in SHAPE_FORMS:
            return None
        return (
            f"gradients of {form} results with respect to mesh node coordinates "
            "(shape gradients) are not available in this version of gfem"
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


def _sheet_pass(model: Model, pairs: list[FieldPair], index: int) -> np.ndarray:
    """``sum_pairs sum_p theta_p c_p int_f left_t . right_t`` per face of a sheet.

    ``p`` runs over the pieces of ``model.boundaries[index]``, ``c_p`` their
    coefficient per unit fill (`gfem._engine.terms.sheet_factors`). The
    terms are in the driven equation's units (length power 0).
    """
    boundary = model.boundaries[index]
    if not isinstance(boundary, (Conductor, Impedance)):  # pragma: no cover
        msg = f"model.boundaries[{index}] carries no sheet field."
        raise TypeError(msg)
    built = discretization(model.mesh).built
    factors = terms.sheet_factors(boundary, built.scale)
    faces = model.mesh.elements(boundary.surface)
    weighted = []
    for pair in pairs:
        weight = sum(pair.weights.get(name, 0.0) * c for name, c in factors.items())
        if weight != 0:
            weighted.append((complex(weight), pair))
    if not weighted:
        return np.zeros(len(faces), dtype=complex)
    region = boundary_region(built, (boundary.surface,))
    p0 = ngs.SurfaceL2(built.mesh, order=0, complex=True)
    q = p0.TestFunction()
    form = ngs.LinearForm(p0)
    trace = Term(_tangential, length_power=0)
    for weight, pair in weighted:
        ds = quadrature.ds(pair.order, definedon=region)
        form += weight * _pair_integrand(trace, pair) * q * ds
    form.Assemble()
    per_face = np.zeros(len(model.mesh.triangles), dtype=complex)
    per_face[built.triangle_order] = form.vec.FV().NumPy()
    return per_face[faces]


def _transport(term: ShapeTerm, u: Any, v: Any) -> Any:
    """The transport tensor of ``term`` for fields ``u``, ``v`` (module doc)."""
    c = term.coefficient
    tensor = len(c.dims) == 2
    identity = ngs.Id(3)
    if term.kind == "curl":
        cu, cv = ngs.curl(u), ngs.curl(v)
        if tensor:
            ccu, ctcv = c * cu, c.trans * cv
            return (
                ngs.OuterProduct(ctcv, cu)
                + ngs.OuterProduct(ccu, cv)
                - (ccu * cv) * identity
            )
        return c * (
            ngs.OuterProduct(cv, cu) + ngs.OuterProduct(cu, cv) - (cu * cv) * identity
        )
    if term.kind == "gradient":
        u, v = ngs.grad(u), ngs.grad(v)
    if tensor:
        cu, ctv = c * u, c.trans * v
        return (cu * v) * identity - ngs.OuterProduct(u, ctv) - ngs.OuterProduct(v, cu)
    return c * ((u * v) * identity - ngs.OuterProduct(u, v) - ngs.OuterProduct(v, u))


def _volume_shape(
    terms: Sequence[tuple[ShapeTerm, complex]], pair: FieldPair, gw: Any
) -> Any:
    """``int Re(sum_t theta_t sum_k T_t(left_k, right_k)) : grad W``, one pair.

    On the pair's fields as they are (complex fields stay complex: no
    ``DiffShape``, no splitting). Value terms are integrated on the pinned
    rule; curl and gradient terms, products of first derivatives, on the
    exact rule of their lower degree (`quadrature.dx_derivatives`).
    """
    totals: dict[bool, Any] = {}
    for a, lefts in _parts(pair.space, pair.left):
        for b, rights in _parts(pair.space, pair.right):
            for term, theta in terms:
                weight = complex(theta * a * b)
                factor: complex | float = weight.real if weight.imag == 0 else weight
                derivatives = term.kind != "value"
                for u, v in zip(lefts, rights, strict=True):
                    part = factor * _transport(term, u, v)
                    total = totals.get(derivatives)
                    totals[derivatives] = part if total is None else total + part
    out = None
    for derivatives, total in totals.items():
        tensor = total.real if total.is_complex else total
        rule = quadrature.dx_derivatives if derivatives else quadrature.dx
        part = ngs.InnerProduct(gw, tensor) * rule(pair.order, definedon=pair.domain)
        out = part if out is None else out + part
    return out


def _trace_shape(
    terms: Sequence[tuple[ShapeTerm, complex]], pair: FieldPair, w: Any
) -> Any:
    """The surface transport of the trace terms, on the pair's fields as they are.

    With ``Gs`` the surface gradient of the displacement (``grad W`` times
    the tangential projector) a tangential trace ``u_t`` of an H(curl) field
    moves covariantly on its face and ``ds`` grows by ``tr Gs``:

    - ``int c u_t . v_t`` changes by ``int c ((u_t . v_t) tr Gs - u_t .
      Gs v_t - v_t . Gs u_t)``;
    - ``int c (n . curl u)(n . curl v)`` (a surface density) by
      ``-int c (n . curl u)(n . curl v) tr Gs``.
    """
    gs = ngs.grad(w).Trace()
    divergence = ngs.Trace(gs)
    out = None
    for a, lefts in _parts(pair.space, pair.left):
        for b, rights in _parts(pair.space, pair.right):
            for term, theta in terms:
                weight = complex(theta * a * b)
                factor: complex | float = weight.real if weight.imag == 0 else weight
                total = None
                for u, v in zip(lefts, rights, strict=True):
                    if term.kind == "trace":
                        ut, vt = u.Trace(), v.Trace()
                        part = (ut * vt) * divergence - ut * (gs * vt) - vt * (gs * ut)
                    else:
                        part = -(ngs.curl(u).Trace() * ngs.curl(v).Trace()) * divergence
                    total = part if total is None else total + part
                if total is None:  # pragma: no cover - a pair has columns
                    continue
                integrand = factor * term.coefficient * total
                if integrand.is_complex:
                    integrand = integrand.real
                ds = quadrature.ds(pair.order, definedon=term.region)
                part = integrand * ds
                out = part if out is None else out + part
    return out


def _real_parts(space: ngs.FESpace, vectors: np.ndarray) -> list[tuple[float, list]]:
    """``vectors`` as ``[(sign, real fields on space)]``: ``Re(l . s)`` pieces.

    ``Re(a(l, s)) = a(Re l, Re s) - a(Im l, Im s)`` for a real bilinear form,
    so the left fields carry the sign (+1 real part, -1 imaginary part).
    """
    out = [(1.0, _gridfunctions(space, np.ascontiguousarray(vectors.real)))]
    if np.iscomplexobj(vectors) and vectors.imag.any():
        out.append((-1.0, _gridfunctions(space, np.ascontiguousarray(vectors.imag))))
    return out


def _surface_shape(
    terms: Sequence[tuple[ShapeTerm, complex]], pair: FieldPair, w: Any
) -> Any:
    """``DiffShape`` of ``Re(sum_t theta_t sum_k a_t(left_k, right_k))``.

    Written with real fields only: theta is folded into the right fields
    (``s = theta right``) and ``Re(a(l, s)) = a(Re l, Re s) - a(Im l, Im s)``.
    """
    lagrangian = None
    lefts: dict[int, list] = {}
    for term, theta in terms:
        space = term.space
        if space is None or term.integrand is None:  # pragma: no cover - rule bug
            msg = f"The surface shape term {term.name!r} has no space or integrand."
            raise ValueError(msg)
        if id(space) not in lefts:
            lefts[id(space)] = _real_parts(space, _columns(pair.left))
        right = theta * _columns(pair.right)
        rights = [
            _gridfunctions(space, np.ascontiguousarray(right.real)),
            _gridfunctions(space, np.ascontiguousarray(right.imag)),
        ]
        for (sign, fields), others in zip(lefts[id(space)], rights, strict=False):
            for u, v in zip(fields, others, strict=True):
                part = term.integrand(u, v)
                part = part if sign == 1.0 else -1.0 * part
                lagrangian = part if lagrangian is None else lagrangian + part
    return None if lagrangian is None else lagrangian.DiffShape(w)


def _node_pass(
    model: Model, pairs: list[FieldPair], shape_terms: Sequence[ShapeTerm]
) -> np.ndarray:
    """``sum_pairs sum_t theta_t d a_t(left, right) / dX``, (n_nodes, 3), per m."""
    built = discretization(model.mesh).built
    n_nodes = len(model.mesh.imported_nodes)
    space = ngs.VectorH1(built.mesh, order=1)
    w = space.TestFunction()
    gw = ngs.grad(w)
    form = ngs.LinearForm(space)
    empty = True
    for pair in pairs:
        volume, traces, surface = [], [], []
        for term in shape_terms:
            theta = pair.weights.get(term.name)
            if theta is None or theta == 0:
                continue
            weighted = (term, theta * built.scale**term.length_power)
            if term.kind == "surface":
                surface.append(weighted)
            elif term.kind in ("trace", "curl trace"):
                traces.append(weighted)
            else:
                volume.append(weighted)
        for part in (
            _volume_shape(volume, pair, gw) if volume else None,
            _trace_shape(traces, pair, w) if traces else None,
            _surface_shape(surface, pair, w) if surface else None,
        ):
            if part is not None:
                form += part
                empty = False
    if empty:
        return np.zeros((n_nodes, 3))
    form.Assemble()
    values = form.vec.FV().NumPy().reshape(3, -1).T[:n_nodes]
    # per metre: the engine mesh is in file units of `scale` metres
    return values / built.scale


def sensitivity(
    model: Model,
    pairs: list[FieldPair],
    wants: set[Hashable],
    *,
    form: str,
    shape: Callable[[], Sequence[ShapeTerm]] | None = None,
) -> dict[Hashable, np.ndarray]:
    """The cotangent of every wanted design input, from the rule's pairs.

    Args:
        model: The design state (concrete leaves) the pairs were solved on.
        pairs: The rule's field pairs, cotangents folded in.
        wants: Input kinds, ``("material", region, property)``,
            ``("boundary", index, "fill", "fill")`` or ``("nodes",)``.
        form: The operator family, the first key of `FORM_TERMS`.
        shape: Returns the operator's `ShapeTerm` list; called only when the
            node coordinates are wanted.

    Returns:
        Per wanted material kind, one value per tetrahedron of the region
        (mesh order): the real cotangent, or for a complex dependence the
        complex one (``sum ct * dy/dp`` when holomorphic, see
        `Coefficient`). Per sheet fill, one complex value per face of
        the surface (``mesh.elements(surface)`` order), ``sum ct * dy/dfill``.
        For ``("nodes",)``, ``(n_nodes, 3)``, real, per metre.

    Raises:
        NotImplementedError: A wanted kind is not supported (see
            `unsupported`); `gfem._adjoint` checks this before tracing.
    """
    for kind in wants:
        reason = unsupported(form, kind)  # ty: ignore[invalid-argument-type]
        if reason is not None:
            raise NotImplementedError(reason)
    out: dict[Hashable, np.ndarray] = {}
    for kind in wants:
        if is_sheet(kind):  # ty: ignore[invalid-argument-type]
            out[kind] = _sheet_pass(model, pairs, kind[1])  # ty: ignore[not-subscriptable]
    if ("nodes",) in wants:
        if shape is None:  # pragma: no cover - rule bug
            msg = f"The {form} rule gives no shape terms."
            raise NotImplementedError(msg)
        out["nodes",] = _node_pass(model, pairs, shape())
    materials = {k for k in wants if k[0] == "material"}  # ty: ignore[not-subscriptable]
    if materials:
        out |= _material_sensitivity(model, pairs, materials, form)
    return out


def _material_sensitivity(
    model: Model, pairs: list[FieldPair], wants: set[Hashable], form: str
) -> dict[Hashable, np.ndarray]:
    """The material part of `sensitivity`."""
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
