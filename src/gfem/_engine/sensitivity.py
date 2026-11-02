"""The sensitivity kernel: field pairs contracted against dA/dp.

Every differentiated solve ends here. A rule's backward pass turns the
output cotangents into **field pairs** ``(left, right)``: the gradient of
``sum_k ct_k y_k`` with respect to a design value ``p`` is
``sum_pairs sum_t theta_t * sum_e (dc_t / dp)_e * a_t,e(left, right)``,
where ``a_t`` is the bilinear form of one **term** ``t``, the kernel's view
of one piece of the operator (driven and eigenmode: the curl-curl, mass and
absorbing pieces) or of the electrostatic form (``grad u . grad v``),
``theta_t`` its scalar weight in this problem (a physical constant, a
frequency factor) and ``c_t`` its coefficient per element (a function of
the material values). The kernel never sees the problem type: the rule
gives it the operator as terms (`Term`), each carrying ``dc_t / dp`` for
the properties its coefficient reads (`Derivative`), and it integrates
every term a wanted property needs once over all pairs, on the pinned
quadrature rule, in one element-wise (P0) pass.

A **face term** (an absorbing boundary, whose coefficient is read from the
tetrahedron behind each face) is integrated per triangle in a face-wise
(surface P0) pass, and each face's integral is added to its owning
tetrahedron, so its coefficient rows read like a volume term's.

**Node coordinates** (shape gradients) are a third pass over the same
pairs, against the same terms, the operator's pieces with their
coefficients. Under a node displacement ``W`` (a real P1 vector
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

A **per-axis** leaf (a `MaterialField` with `axes`, principal values
``v_k`` along fixed axes ``a_k``) moves a coefficient coaxial with the
axes: ``dC/dv_k = s_k P_k`` with ``P_k = a_k a_k^T`` and ``s_k`` the
isotropic scalar derivative on axis k. Its pass integrates the symmetric
**moment** ``M_e = int_e sym(u (x) v)`` (of the values, curls or gradients:
6 components per element) once per term, and contracts it per axis,
``s_k a_k^T M_e a_k``; ``tr M_e`` is the scalar pass's integral, so equal
principal values give per-axis cotangents that sum to the isotropic one. A
face term (the first-order absorbing admittance, which is not coaxial once
the normal is tilted) integrates ``int_f sym(u_t (x) v_t)`` per face and
contracts it with ``dc_f/dv_k`` (a 3 x 3 tensor per face and axis) into
its owner. The moment pass runs only when a per-axis leaf is wanted: an
isotropic leaf keeps the scalar pass.

A **sheet field** (`gfem.SheetField`, a fill per face of a `Conductor` or
`Impedance` surface) multiplies the coefficient of each of its boundary's
pieces face by face, ``c_p,f = fill_f c_p``, so ``dA / dfill_f = sum_p
theta_p c_p B_f`` with ``B_f`` the face's tangential trace form. Its
**face pass** integrates ``sum_p theta_p c_p left_t . right_t`` per
triangle of the surface (surface P0, the pinned rule) and returns one value
per face, with no owner tetrahedron: the fill is a property of the face.
Each sheet piece's term carries its ``c_p`` (`Term.sheet`).

Adding a problem type means giving its rule a ``terms`` hook (its operator
as `Term` values, with their derivatives) and a row of `PROPERTIES`, not a
code path. A later pass plugs in beside these: load pairs ``int lambda psi``
for problem-leaf sources.

Design inputs are named by kind tuples, the same ones `gfem._adjoint` uses:
``("material", region, property)`` is a `MaterialField` leaf (per axis when
the field has `axes`),
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
from gfem._engine import quadrature
from gfem._engine.cache import discretization
from gfem._engine.coefficients import element_axes, per_axis_elements
from gfem._engine.domain import boundary_region
from gfem._materials import MaterialField
from gfem.errors import NonDifferentiableError
from gfem.model import Model

__all__ = [
    "INTEGRANDS",
    "MOMENTS",
    "PROPERTIES",
    "SHAPE_FORMS",
    "SHEET_FORMS",
    "Derivative",
    "FieldPair",
    "Term",
    "is_sheet",
    "sensitivity",
    "unsupported",
]


@dataclasses.dataclass(frozen=True)
class Derivative:
    """``dc_t / dp``: how one material property enters one term's coefficient.

    Attributes:
        declared: ``dc_t / dp`` per tetrahedron (gfem order), or a constant;
            or a function of no arguments that returns it, called by the
            material pass only (the pieces are built on every forward, their
            derivatives are read by a backward pass only).
        holomorphic: Whether ``c_t`` is holomorphic in a complex property
            value. If not, it is a function of one real quantity ``r`` (the
            electrostatic permittivity reads ``r = Re(eps_r)``), and
            ``declared`` is ``dc_t / dr``.
        chain: For a non-holomorphic ``c_t``, ``r = Re(chain * p)`` for a
            complex value ``p`` (``None``: ``chain = 1``), as a function of
            no arguments. The cotangent is then ``chain * Re(sum ct *
            dy/dr)``, which is JAX's convention (``dJ/dRe(p) - i
            dJ/dIm(p)``); real when ``chain`` is 1.
        length_power: The power of the mesh length unit that ``declared``
            leaves out of the coefficient: the term's integrals are scaled by
            ``scale ** (term.length_power + length_power)``. 1 for the
            driven conductivity, whose coefficient ``eta0 sigma scale`` reads
            ``eta0`` per unit ``sigma``.
        per_axis: The per-axis form, for a `MaterialField` with `axes`
            (``None``: the term has none). Its ``declared`` is per principal
            value: for a volume term the scalar ``s`` of ``dC/dv_k = s_k
            a_k a_k^T`` per tetrahedron and axis, ``(n, 3)`` or a constant;
            for a face term ``dc_f/dv_k`` of the tetrahedron behind each
            face, ``(n_triangles, 3, 3, 3)`` per gfem triangle (axis k,
            then the 3 x 3 coefficient). Its ``chain`` is ``(n, 3)``.
    """

    declared: Callable[[], np.ndarray | float | complex] | float | complex
    holomorphic: bool
    chain: Callable[[], np.ndarray] | None = None
    length_power: int = 0
    per_axis: Derivative | None = None

    def evaluated(self) -> np.ndarray | float | complex:
        """``dc_t / dp`` (`declared`, evaluated if it is a function)."""
        declared = self.declared
        if isinstance(declared, int | float | complex):
            return declared
        return declared()


@dataclasses.dataclass(frozen=True, eq=False)
class Term:
    """The kernel's view of one piece of an operator, or of the electrostatic form.

    A driven or eigenmode piece, or the electrostatic ``grad u . grad v``
    form, as the passes read it: its bilinear form ``a_t(u, v)``, its
    coefficient and the coefficient's derivatives.

    Attributes:
        name: The term's key in the pairs' `FieldPair.weights`.
        kind: ``"gradient"`` (``int (C grad u) . grad v``, H1), ``"value"``
            (``int (C u) . v``, H(curl)), ``"curl"`` (``int (C curl u) .
            curl v``), ``"trace"`` (``int c u_t . v_t`` on a surface),
            ``"curl trace"`` (``int c (n . curl u)(n . curl v)``), or
            ``"surface"`` (any other surface form, differentiated by
            ``DiffShape``). The material pass integrates the kind's bare
            integrand (`INTEGRANDS`).
        coefficient: For a volume kind, ``C``: an element-wise constant
            NGSolve coefficient, scalar or 3 x 3, real or complex, the same
            the forward form reads; for a trace kind, the face-wise
            constant ``c``: a scalar, or 3 x 3 for an absorbing face next
            to an anisotropic material (no node pass).
        region: For a trace kind, the boundary region it is integrated on.
        integrand: For ``"surface"``: ``(u, v) -> integrals`` of the form
            with its coefficient and pinned quadrature (real coefficients),
            for real fields ``u`` and ``v``.
        space: For ``"surface"``: a real space with the pairs' DOF numbering.
        length_power: The power of the mesh length unit in the term: the
            integral on the mesh in file units times ``scale **
            length_power`` is the term in the units the pairs' weights
            expect (1 for ``grad . grad`` in 3D, in SI; the driven terms are
            in the driven equation's own units, file units, see
            `gfem._engine.terms`: 0).
        owner: For a face term whose coefficient is read from the
            tetrahedron behind each face (an absorbing boundary), that
            tetrahedron per gfem triangle, -1 off the term's surfaces: the
            material pass adds each face's integral to it. ``None`` for a
            volume term.
        derivatives: ``dc_t / dp`` by material property, for the properties
            the coefficient reads.
        sheet: For a piece of a sheet boundary, ``(index, c)``: the
            boundary's index in ``model.boundaries`` and the piece's
            coefficient per unit fill. ``None`` otherwise.
    """

    name: str
    kind: Literal["gradient", "value", "curl", "trace", "curl trace", "surface"]
    coefficient: Any = None
    region: Any = None
    integrand: Callable[[Any, Any], Any] | None = None
    space: ngs.FESpace | None = None
    length_power: int = 0
    owner: np.ndarray | None = None
    derivatives: Mapping[str, Derivative] = dataclasses.field(default_factory=dict)
    sheet: tuple[int, float] | None = None


def _dot(u: ngs.CoefficientFunction, v: ngs.CoefficientFunction) -> ngs.CF:
    return u * v


def _tangential(u: Any, v: Any) -> Any:
    """``u_t . v_t`` on a surface (an H(curl) field's trace)."""
    return u.Trace() * v.Trace()


#: The bare integrand (no coefficient) of each kind of `Term`, bilinear in
#: two NGSolve fields: what the material pass integrates.
INTEGRANDS: dict[str, Callable[[Any, Any], Any]] = {
    "gradient": lambda u, v: ngs.grad(u) * ngs.grad(v),
    "curl": lambda u, v: ngs.curl(u) * ngs.curl(v),
    "value": _dot,
    "trace": _tangential,
    "curl trace": lambda u, v: ngs.curl(u).Trace() * ngs.curl(v).Trace(),
}

#: The symmetric moment ``u (x) v`` of each kind of `Term` with a
#: per-axis derivative: what the moment pass integrates.
MOMENTS: dict[str, Callable[[Any, Any], Any]] = {
    "gradient": lambda u, v: ngs.OuterProduct(ngs.grad(u), ngs.grad(v)),
    "curl": lambda u, v: ngs.OuterProduct(ngs.curl(u), ngs.curl(v)),
    "value": ngs.OuterProduct,
    "trace": lambda u, v: ngs.OuterProduct(u.Trace(), v.Trace()),
}
#: The 6 components of a symmetric 3 x 3 moment, in the pass's order.
_SYMMETRIC = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

#: The material properties each operator family differentiates: the
#: properties its terms carry `Term.derivatives` for. Static, so that
#: `unsupported` answers before any operator exists (at trace time).
#: Design inputs are `MaterialField` leaves: isotropic ones read one value
#: per element, those with `axes` their principal values
#: (`Derivative.per_axis`). A `Material` (isotropic or not) is a constant:
#: its tensor enters the forward pieces only, and the values its elements
#: show to the derivatives are never differentiated.
PROPERTIES: dict[str, frozenset[str]] = {
    # a(u, v) = eps0 int Re(eps_r) grad u . grad v
    "electrostatic": frozenset({"eps_r"}),
    # driven: mass eps_r (1 - i tan_delta), curl-curl 1 / mu_r, conductivity
    # eta0 sigma scale (real sigma), absorbing c1 and c2 of the real e and m
    "driven": frozenset({"eps_r", "tan_delta", "mu_r", "conductivity"}),
    # the driven pieces (`gfem._eigen.operator`), weighted at the complex
    # eigenvalue
    "eigenmode": frozenset({"eps_r", "tan_delta", "mu_r", "conductivity"}),
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


#: Operator families whose rules' terms the node pass reads.
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
        if kind[2] in PROPERTIES.get(form, frozenset()):
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


def _pair_integrand(
    integrand: Callable[[Any, Any], Any], pair: FieldPair
) -> ngs.CoefficientFunction:
    """``sum_k a(left_k, right_k)`` as one coefficient function (bilinear)."""
    parts = [
        (a * b) * integrand(u, v)
        for a, lefts in _parts(pair.space, pair.left)
        for b, rights in _parts(pair.space, pair.right)
        for u, v in zip(lefts, rights, strict=True)
    ]
    return sum(parts[1:], parts[0])


def _element_pass(
    model: Model, pairs: list[FieldPair], term: Term, length_power: int
) -> np.ndarray:
    """``sum_pairs theta_t sum_k a_t,e(left_k, right_k)`` per gfem tetrahedron.

    Times ``scale ** length_power``. A face term's integrals per triangle
    are summed into each triangle's owning tetrahedron (`Term.owner`).
    """
    built = discretization(model.mesh).built
    integrand = INTEGRANDS[term.kind]
    name = term.name
    pairs = [p for p in pairs if name in p.weights]
    is_complex = any(
        np.iscomplexobj(p.left)
        or np.iscomplexobj(p.right)
        or np.iscomplexobj(p.weights[name])
        for p in pairs
    )
    out = np.zeros(len(model.mesh.tetrahedra), dtype=complex if is_complex else float)
    if not pairs:
        return out
    owner = term.owner
    if owner is None:
        p0 = ngs.L2(built.mesh, order=0, complex=is_complex)
        q = p0.TestFunction()
        form = ngs.LinearForm(p0)
        for pair in pairs:
            dx = quadrature.dx(pair.order, definedon=pair.domain)
            form += pair.weights[name] * _pair_integrand(integrand, pair) * q * dx
        form.Assemble()
        out[built.tetrahedron_order] = form.vec.FV().NumPy()
        return out * built.scale**length_power
    p0 = ngs.SurfaceL2(built.mesh, order=0, complex=is_complex)
    q = p0.TestFunction()
    form = ngs.LinearForm(p0)
    for pair in pairs:
        ds = quadrature.ds(pair.order, definedon=term.region)
        form += pair.weights[name] * _pair_integrand(integrand, pair) * q * ds
    form.Assemble()
    per_face = np.zeros(len(owner), dtype=out.dtype)
    per_face[built.triangle_order] = form.vec.FV().NumPy()
    on = owner >= 0
    np.add.at(out, owner[on], per_face[on])
    return out * built.scale**length_power


def _moment_pass(
    model: Model, pairs: list[FieldPair], term: Term, length_power: int
) -> np.ndarray:
    """``sum_pairs theta_t sum_k int sym(left_k (x) right_k)``, per element.

    Of the values, curls, gradients or tangential traces as the kind reads
    them (`MOMENTS`), times ``scale ** length_power``: ``(n_tetrahedra, 3,
    3)`` for a volume term, ``(n_triangles, 3, 3)`` per gfem triangle for a
    face term (not summed into the owners: its derivative is per face).
    The trace of each moment is `_element_pass`'s integral.
    """
    built = discretization(model.mesh).built
    moment = MOMENTS[term.kind]
    name = term.name
    pairs = [p for p in pairs if name in p.weights]
    is_complex = any(
        np.iscomplexobj(p.left)
        or np.iscomplexobj(p.right)
        or np.iscomplexobj(p.weights[name])
        for p in pairs
    )
    owner = term.owner
    face = owner is not None
    count = len(model.mesh.tetrahedra) if owner is None else len(owner)
    out = np.zeros((count, 3, 3), dtype=complex if is_complex else float)
    if not pairs:
        return out
    if face:
        p0 = ngs.SurfaceL2(built.mesh, order=0, complex=is_complex)
        order = built.triangle_order
    else:
        p0 = ngs.L2(built.mesh, order=0, complex=is_complex)
        order = built.tetrahedron_order
    space = ngs.FESpace([p0] * len(_SYMMETRIC))
    tests = space.TestFunction()
    form = ngs.LinearForm(space)
    for pair in pairs:
        if face:
            measure = quadrature.ds(pair.order, definedon=term.region)
        else:
            measure = quadrature.dx(pair.order, definedon=pair.domain)
        m = _pair_integrand(moment, pair)
        parts = [
            (m[i, j] if i == j else 0.5 * (m[i, j] + m[j, i])) * q
            for (i, j), q in zip(_SYMMETRIC, tests, strict=True)
        ]
        form += pair.weights[name] * sum(parts[1:], parts[0]) * measure
    form.Assemble()
    values = form.vec.FV().NumPy().reshape(len(_SYMMETRIC), -1)
    for c, (i, j) in enumerate(_SYMMETRIC):
        out[order, i, j] = values[c]
        out[order, j, i] = values[c]
    return out * built.scale**length_power


def _sheet_pass(
    model: Model, pairs: list[FieldPair], index: int, terms: Sequence[Term]
) -> np.ndarray:
    """``sum_pairs sum_p theta_p c_p int_f left_t . right_t`` per face of a sheet.

    ``p`` runs over the terms of ``model.boundaries[index]``'s pieces, in
    their order, ``c_p`` their coefficient per unit fill (`Term.sheet`).
    The terms are in the driven equation's units (length power 0).
    """
    boundary = model.boundaries[index]
    if not isinstance(boundary, (Conductor, Impedance)):  # pragma: no cover
        msg = f"model.boundaries[{index}] carries no sheet field."
        raise TypeError(msg)
    built = discretization(model.mesh).built
    factors = [
        (t.name, t.sheet[1])
        for t in terms
        if t.sheet is not None and t.sheet[0] == index
    ]
    faces = model.mesh.elements(boundary.surface)
    weighted = []
    for pair in pairs:
        weight = sum(pair.weights.get(name, 0.0) * c for name, c in factors)
        if weight != 0:
            weighted.append((complex(weight), pair))
    if not weighted:
        return np.zeros(len(faces), dtype=complex)
    region = boundary_region(built, (boundary.surface,))
    p0 = ngs.SurfaceL2(built.mesh, order=0, complex=True)
    q = p0.TestFunction()
    form = ngs.LinearForm(p0)
    for weight, pair in weighted:
        ds = quadrature.ds(pair.order, definedon=region)
        form += weight * _pair_integrand(_tangential, pair) * q * ds
    form.Assemble()
    per_face = np.zeros(len(model.mesh.triangles), dtype=complex)
    per_face[built.triangle_order] = form.vec.FV().NumPy()
    return per_face[faces]


def _transport(term: Term, u: Any, v: Any) -> Any:
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
    terms: Sequence[tuple[Term, complex]], pair: FieldPair, gw: Any
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


def _trace_shape(terms: Sequence[tuple[Term, complex]], pair: FieldPair, w: Any) -> Any:
    """The surface transport of the trace terms, on the pair's fields as they are.

    With ``Gs`` the surface gradient of the displacement (``grad W`` times
    the tangential projector) a tangential trace ``u_t`` of an H(curl) field
    moves covariantly on its face and ``ds`` grows by ``tr Gs``:

    - ``int c u_t . v_t`` changes by ``int c ((u_t . v_t) tr Gs - u_t .
      Gs v_t - v_t . Gs u_t)``;
    - ``int c (n . curl u)(n . curl v)`` (a surface density) by
      ``-int c (n . curl u)(n . curl v) tr Gs``.

    Raises:
        NonDifferentiableError: A trace term has a tensor coefficient (an
            absorbing face next to an anisotropic material).
    """
    for term, _ in terms:
        # the rules refuse the node coordinates first (unsupported_input)
        tensor = getattr(term.coefficient, "dim", 1) != 1
        if term.kind == "trace" and tensor:  # pragma: no cover
            msg = (
                f"The trace term {term.name!r} has a tensor coefficient, whose "
                "shape derivative is not available in this version."
            )
            raise NonDifferentiableError(msg)
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
    terms: Sequence[tuple[Term, complex]], pair: FieldPair, w: Any
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
    model: Model, pairs: list[FieldPair], terms: Sequence[Term]
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
        for term in terms:
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
    terms: Callable[[bool], Sequence[Term]] | None = None,
) -> dict[Hashable, np.ndarray]:
    """The cotangent of every wanted design input, from the rule's pairs.

    Args:
        model: The design state (concrete leaves) the pairs were solved on.
        pairs: The rule's field pairs, cotangents folded in.
        wants: Input kinds, ``("material", region, property)``,
            ``("boundary", index, "fill", "fill")`` or ``("nodes",)``.
        form: The operator family, a key of `PROPERTIES`.
        terms: ``terms(nodes)`` returns the operator's `Term` list, called
            once when anything is wanted; ``nodes`` is whether the node
            coordinates are.

    Returns:
        Per wanted material kind, one value per tetrahedron of the region
        (mesh order), or ``(n, 3)`` (one per principal axis) for a field
        with `axes`: the real cotangent, or for a complex dependence the
        complex one (``sum ct * dy/dp`` when holomorphic, see
        `Derivative`). Per sheet fill, one complex value per face of
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
    if not wants:
        return out
    if terms is None:  # pragma: no cover - rule bug
        msg = f"The {form} rule gives no terms."
        raise NotImplementedError(msg)
    operator = terms(("nodes",) in wants)
    for kind in wants:
        if is_sheet(kind):  # ty: ignore[invalid-argument-type]
            out[kind] = _sheet_pass(model, pairs, kind[1], operator)  # ty: ignore[not-subscriptable]
    if ("nodes",) in wants:
        out["nodes",] = _node_pass(model, pairs, operator)
    materials = {k for k in wants if k[0] == "material"}  # ty: ignore[not-subscriptable]
    if materials:
        out |= _material_sensitivity(model, pairs, materials, operator)
    return out


def _per_axis(model: Model, kind: tuple) -> bool:
    """Whether material ``kind`` is a leaf of principal values (field `axes`)."""
    material = model.materials.get(kind[1])
    return isinstance(material, MaterialField) and material.has_principal_values(
        kind[2]
    )


def _material_sensitivity(
    model: Model, pairs: list[FieldPair], wants: set[Hashable], terms: Sequence[Term]
) -> dict[Hashable, np.ndarray]:
    """The material part of `sensitivity`: each property's terms, in their order."""
    axis_kinds = {k for k in wants if _per_axis(model, k)}  # ty: ignore[invalid-argument-type]
    integrals: dict[tuple[str, int], np.ndarray] = {}
    per_property: dict[str, np.ndarray] = {}
    for prop in sorted({k[2] for k in wants - axis_kinds}):  # ty: ignore[not-subscriptable]
        total: np.ndarray = np.zeros(len(model.mesh.tetrahedra))
        for term in terms:
            derivative = term.derivatives.get(prop)
            if derivative is None:
                continue
            key = (term.name, term.length_power + derivative.length_power)
            if key not in integrals:
                integrals[key] = _element_pass(model, pairs, term, key[1])
            part = derivative.evaluated() * integrals[key]
            if not derivative.holomorphic:
                part = np.real(part)
                if derivative.chain is not None:
                    part = derivative.chain() * part
            total = total + part
        per_property[prop] = np.asarray(total)
    per_axis = _per_axis_sensitivity(
        model,
        pairs,
        {k[2] for k in axis_kinds},  # ty: ignore[not-subscriptable]
        terms,
    )
    return {
        kind: (per_axis if kind in axis_kinds else per_property)[kind[2]][  # ty: ignore[not-subscriptable]
            model.mesh.elements(kind[1])  # ty: ignore[not-subscriptable]
        ]
        for kind in wants
    }


def _per_axis_sensitivity(
    model: Model, pairs: list[FieldPair], props: set[str], terms: Sequence[Term]
) -> dict[str, np.ndarray]:
    """Each property's cotangent per tetrahedron and principal axis, ``(n, 3)``.

    From the terms' `Derivative.per_axis` forms and the moment pass, on the
    axes of each tetrahedron's field (`element_axes`).

    Raises:
        NotImplementedError: A term reads a wanted property on an element
            of a field with axes but has no per-axis derivative (a rule bug:
            its forward refuses fields with axes there).
    """
    if not props:
        return {}
    axes = element_axes(model)
    fields = per_axis_elements(model)
    moments: dict[tuple[str, int], np.ndarray] = {}
    out: dict[str, np.ndarray] = {}
    for prop in sorted(props):
        total: np.ndarray = np.zeros((len(model.mesh.tetrahedra), 3))
        for term in terms:
            derivative = term.derivatives.get(prop)
            if derivative is None:
                continue
            axis = derivative.per_axis
            if axis is None:
                # a face term none of whose owners holds principal values
                # (second-order absorbing beside isotropic elements only)
                owner = term.owner
                if owner is not None and not fields[owner[owner >= 0]].any():
                    continue
                msg = f"The term {term.name!r} has no per-axis derivative of {prop}."
                raise NotImplementedError(msg)  # pragma: no cover - rule bug
            key = (term.name, term.length_power + axis.length_power)
            if key not in moments:
                moments[key] = _moment_pass(model, pairs, term, key[1])
            moment = moments[key]
            if term.owner is None:
                # s_k a_k^T M a_k per tetrahedron and axis
                along = np.einsum("eki,eij,ekj->ek", axes, moment, axes)
                part = axis.evaluated() * along
            else:
                per_face = np.einsum("fkij,fij->fk", axis.evaluated(), moment)
                part = np.zeros((len(model.mesh.tetrahedra), 3), per_face.dtype)
                on = term.owner >= 0
                np.add.at(part, term.owner[on], per_face[on])
            if not axis.holomorphic:
                part = np.real(part)
                if axis.chain is not None:
                    part = axis.chain() * part
            total = total + part
        out[prop] = np.asarray(total)
    return out
