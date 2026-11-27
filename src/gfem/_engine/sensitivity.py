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

On a 2D mesh the elements are triangles and the faces line segments, and
the passes run unchanged on them (counts by role: volume and surface
elements). The second-order absorbing term there is a **tangential
gradient** ``int c grad_t u . grad_t v`` along each segment.

A **layer grid** of a 2D mesh (a `MaterialField` holding pixels) has its
coefficients on the pinned rule's points (`gfem._engine.points`): its terms
carry `Term.points` and a derivative per point. Its **point pass** integrates
each such term against the rule space's test function, one value per point,
and the transpose of the bilinear point map carries the sum of derivative
times integral back to the pixels. The element pass skips these terms.

**Node coordinates** (shape gradients) are a third pass over the same
pairs, against the same terms, the operator's pieces with their
coefficients. Under a node displacement ``W`` (a real P1 vector
field, one test function per node and axis) a volume term changes by
``int T(u, v) : grad W``, with hand-written **transport tensors** ``T`` on
the complex fields themselves (gradients and H(curl) values transform
covariantly, curls contravariantly; element-wise constant coefficients
travel with their elements). A coefficient that is also a fixed function
of space (a perfectly matched layer's stretch) is read at the moved
quadrature points, which adds ``int ((W . grad) C u) . v`` (curls for a
curl term), on the pinned rule like its forward form:

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
points. On a curved mesh no rule is exact (the integrands are rational in the
reference coordinates): the node pass uses an order-2 velocity and every term
on the forward's own rule, so it is the exact derivative of the discretized
forward form, in every corner and mid-edge node.

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

A **lumped value** ``p`` (a lumped port's inductance or capacitance, an
`Impedance`'s resistance, inductance or capacitance) scales the
coefficient of its pieces, ``c ∝ p`` (C) or ``1 / p`` (R, L), so ``dA / dp =
(s / p) sum_t theta_t A_t`` with ``s = ±1``. Its **lumped pass** integrates
``sum_t theta_t (s / p) c_t left_t . right_t`` on the pieces' surfaces with the
sheet pass's surface P0 form and sums the faces: one complex number per value.
The rule gives those pieces as `Term` values carrying ``Term.lumped``, through a
``lumped_terms`` hook beside ``terms``.

A **law parameter** ``p`` (a permittivity law's eps_inf, a strength or a
frequency) moves the θ of its law's pieces, not their
coefficients: the susceptibility piece (whose form, the region's indicator
times ``u . v``, is also eps_inf's part of the mass) and the law's absorbing
face pieces. The rule gives dθ/dp at each pair's frequency as a weight of
its own, under `law_key` (piece name, parameter), and its terms carry
`Term.law`. The **law pass** contracts each such term once per pair carrying
one of its keys, ``int c left . right`` (one scalar per pair), and
multiplies by each parameter's weight: one integral per pair and law
piece, whatever the parameter count. It reads the weight under its own key
only, so it accepts pairs that carry only that key (an adaptive sweep's,
weight 1 per key).

A **load** is a problem leaf that is one power density per
tetrahedron (`LOAD_KIND`, the summed heat sources of a thermal problem):
the load side of the operator, ``F_i = int q psi_i``, so ``dJ / dq_e =
int_e Lambda`` for the rule's adjoint ``Lambda`` (a transient's is the
weighted sum of its step adjoints). Its **load pass** integrates the
adjoint against the constant 1 on each element (P0), times ``scale **
length_power``; no forward field is involved. A load form's face density
(`FACE_LOAD_KIND`, one value per triangle: sheet heat) loads ``F_i = int_S
q_s psi_i``, so its **face load pass** integrates the same adjoint against
1 on each triangle, one length power below the volume pass. The rule
returns the adjoint as a `LoadField` among its pairs. The other problem
leaves of a load form (a waveform's scales, a trace's weights) are the
rule's explicit terms (``explicit_problem``), not passes.

A rule that contracts its pairs as it goes (the transient reverse sweep,
whose pairs would otherwise hold every step's fields) returns their
per-element integrals as `Integrated`, which the material pass adds to its
own integrals of each term.

A rule whose material inputs are tensors in a rotated frame (a solid's
stiffness and piezoelectric tensor, read in the crystal's frame) gives
`Tensors` instead of terms for its material leaves: the **tensor pass**
integrates **moment blocks** of each pair, the lab-frame products of strains,
potential gradients and displacements per element (`MomentBlock`), and the
rule's own contraction maps them, with the pair's weights, to the cotangent
of each leaf (through the transpose of the frame rotation).

Adding a problem type means giving its rule a ``terms`` hook (its operator
as `Term` values, with their derivatives) and a row of `PROPERTIES`, not a
code path.

Design inputs are named by kind tuples, the same ones `gfem._adjoint` uses:
``("material", region, property)`` is a `MaterialField` leaf (per axis when
the field has `axes`),
``("boundary", index, "fill", "fill")`` the fill of the sheet field of
``model.boundaries[index]``, ``("port", index, "inductance")`` and
``("boundary", index, "resistance")`` (and the like) a lumped value,
``("law", region, "eps_inf")`` and ``("law", region, term_index, name)`` a
permittivity law's parameter, ``("nodes",)`` the mesh node coordinates,
``("problem", ".density")`` (`LOAD_KIND`) a load form's power density,
and ``("problem", ".face_density")`` (`FACE_LOAD_KIND`) its face density.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Literal

import ngsolve as ngs
import numpy as np

from gfem import _mpi
from gfem._boundaries import THERMAL_VALUES, Conductor, Impedance
from gfem._engine import quadrature
from gfem._engine.cache import discretization
from gfem._engine.coefficients import element_axes, facet_space, per_axis_elements
from gfem._engine.domain import boundary_region
from gfem._engine.runtime import serial_spaces
from gfem._materials import MaterialField
from gfem.errors import NonDifferentiableError
from gfem.model import Model

__all__ = [
    "FACE_LOAD_KIND",
    "INTEGRANDS",
    "LAW_FORMS",
    "LOAD_FORMS",
    "LOAD_KIND",
    "LUMPED_FORMS",
    "MOMENTS",
    "PROPERTIES",
    "SHAPE_FORMS",
    "SHEET_FORMS",
    "Derivative",
    "FieldPair",
    "Integrated",
    "LoadField",
    "MomentBlock",
    "Tensors",
    "Term",
    "element_integrals",
    "energy_term",
    "is_load",
    "is_lumped",
    "is_sheet",
    "law_key",
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
            ``"curl trace"`` (``int c (n . curl u)(n . curl v)``),
            ``"tangential gradient"`` (``int c grad_t u . grad_t v`` on a
            line of a 2D mesh, H1; no node pass), or
            ``"surface"`` (any other surface form, differentiated by
            ``DiffShape``). The material pass integrates the kind's bare
            integrand (`INTEGRANDS`).
        coefficient: For a volume kind, ``C``: an element-wise constant
            NGSolve coefficient (times a fixed function of space when
            ``position``), scalar or 3 x 3, real or complex, the same the
            forward form reads; for a trace kind, the face-wise
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
        lumped: For a ``"trace"`` term of a piece that moves with a lumped
            value ``p`` (the lumped pass's terms), ``(kind, slope)``: the
            value's input kind and ``dc / dp = slope c``. ``None`` otherwise.
        law: For a piece of a permittivity law (its θ moves with the law's
            parameters), the law's region: the law pass reads its weights
            under `law_key`. ``None`` otherwise.
        points: For a volume term whose coefficient lives on the quadrature
            points of a layer grid's region (`gfem._engine.points`), that
            region: its derivatives are per point of the region, and the
            point pass reads them. ``None`` otherwise.
        position: Whether a volume term's ``coefficient`` also varies in
            position within an element (a perfectly matched layer's
            stretch, a fixed function of space): the node pass then adds
            the coefficient's change at the moved quadrature points.
        elements: For the node pass, the NGSolve elements a volume or trace
            term's form is restricted to (a ``BitArray`` of volume or
            surface elements, a measurement's element layer); ``None``: all.
    """

    name: str
    kind: Literal[
        "gradient",
        "value",
        "curl",
        "trace",
        "curl trace",
        "tangential gradient",
        "surface",
    ]
    coefficient: Any = None
    region: Any = None
    integrand: Callable[[Any, Any], Any] | None = None
    space: ngs.FESpace | None = None
    length_power: int = 0
    owner: np.ndarray | None = None
    derivatives: Mapping[str, Derivative] = dataclasses.field(default_factory=dict)
    sheet: tuple[int, float] | None = None
    lumped: tuple[tuple, float] | None = None
    law: str | None = None
    points: str | None = None
    position: bool = False
    elements: Any = None


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
    # on a boundary element an H1 field's gradient is its surface gradient
    "tangential gradient": lambda u, v: ngs.grad(u) * ngs.grad(v),
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
    # and the density of a blend with a law end: its mass limit
    # and its susceptibility pieces' coefficients
    "driven": frozenset({"eps_r", "tan_delta", "mu_r", "conductivity", "blend"}),
    # the driven pieces (`gfem._eigen.operator`), weighted at the complex
    # eigenvalue
    "eigenmode": frozenset({"eps_r", "tan_delta", "mu_r", "conductivity"}),
    # a(u, v) = int sigma grad u . grad v on the conductors
    "dc_conduction": frozenset({"conductivity"}),
    # a(u, v) = int k grad u . grad v (+ convection, which reads no material)
    "steady_thermal": frozenset({"thermal_conductivity"}),
    # the steady form and the heat capacity int rho c_p u v
    "transient_thermal": frozenset(
        {"thermal_conductivity", "mass_density", "specific_heat"}
    ),
    # the stiffness, coupling, dielectric and mass pieces of a piezoelectric
    # section, with the loss 1 / Q_m (the tensor pass)
    "piezoelectric": frozenset(
        {
            "stiffness",
            "piezoelectric_tensor",
            "eps_r",
            "mass_density",
            "mechanical_quality_factor",
        }
    ),
    # a wave port's cross-section forms, mapped to the owner tetrahedra of the
    # port face (`gfem._engine.port_mode_sensitivity.integrated`)
    "port mode": frozenset({"eps_r", "tan_delta", "mu_r", "conductivity"}),
}


@dataclasses.dataclass(frozen=True, eq=False)
class FieldPair:
    """Columns of two fields whose form terms are contracted per element.

    The pair contributes ``sum_k theta_t a_t(left[:, k], right[:, k])`` to
    each term ``t`` in `weights`. Rules fold their output cotangents into
    ``right`` (or ``left``) so a backward pass needs few pairs.

    Attributes:
        space: The NGSolve space of both fields: the solve space, or the
            unpaired space on a periodic model (the rule expands both
            sides, the left one with conj(P)).
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
class MomentBlock:
    """Per-element integrals of bilinear products of a pair's two fields.

    Attributes:
        name: The block's key in the moments `Tensors.contract` reads.
        size: Number of entries.
        entries: ``(left, right) -> CF`` of ``size`` components, bilinear in
            the two fields of the pair's space (NGSolve grid functions).
        domain: The NGSolve region the block is integrated on.
        terms: The weight names (`FieldPair.weights`) the block serves: a pair
            whose weights are all zero for them skips the block.
    """

    name: str
    size: int
    entries: Callable[[Any, Any], Any]
    domain: ngs.Region
    terms: tuple[str, ...]


@dataclasses.dataclass(frozen=True, eq=False)
class Tensors:
    """A rule's material leaves as moments contracted on the host.

    Attributes:
        blocks: The moment blocks the wanted leaves need.
        contract: ``(weights, moments) -> {kind: values}``: one pair's weights
            and its integrals by block name, ``(n_tetrahedra, size)`` (a
            skipped block is missing), to each wanted leaf's cotangent over the
            region's elements (mesh order), complex; linear in the moments, so
            the pass sums it over pairs.
    """

    blocks: Sequence[MomentBlock]
    contract: Callable[
        [Mapping[str, complex], Mapping[str, np.ndarray]], dict[Hashable, np.ndarray]
    ]


@dataclasses.dataclass(frozen=True, eq=False)
class LoadField:
    """The adjoint a load pass integrates per element (`LOAD_KIND`).

    Attributes:
        space: The NGSolve space of ``values``.
        values: The adjoint's DOF vector, shape ``(ndof,)`` (real).
        order: The discretization order (fixes the quadrature rule).
        domain: The NGSolve volume region the field lives on.
        length_power: The power of the mesh length unit in ``dF_i / dq_e``
            over the file-unit integral ``int_e psi_i``.
    """

    space: ngs.FESpace
    values: np.ndarray
    order: int
    domain: ngs.Region
    length_power: int = 0


@dataclasses.dataclass(frozen=True, eq=False)
class Integrated:
    """Per-element integrals of pairs a rule contracted itself.

    Attributes:
        values: Per term name, ``sum_pairs theta_t sum_k a_t,e(left_k,
            right_k)`` per gfem tetrahedron, on the mesh in file units
            (`element_integrals`; the material pass applies the term's
            ``scale ** length_power``).
        moments: Per term name, the same contraction of the symmetric
            moments (`_moment_pass`), ``(n_tetrahedra, 3, 3)``, for a
            per-axis leaf; empty when the rule has none.
    """

    values: Mapping[str, np.ndarray]
    moments: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)


def energy_term(
    name: str,
    coefficient: Any,
    elements: np.ndarray,
    *,
    axis: int | None = None,
) -> Term:
    """An explicit term of the mode energy: ``int Re(eps_r) u . v`` on ``elements``.

    The eigenmode rule's ratios divide by the electric energy ``1/2 eps0 int
    Re(eps_r) |E|^2``, and a region's or a principal axis's share
    is the same integral on part of the mesh. Their explicit design
    derivative is this term, not the operator's mass piece ``eps_r (1 - i
    tan_delta)``: its coefficient reads ``r = Re(eps_r)`` (``dc/dr = 1`` on
    ``elements``, so a complex leaf's cotangent is real), and nothing else,
    so ``tan_delta``, ``conductivity`` and ``mu_r`` give it no derivative.
    A pair ``(conj(u), u)`` weighted with ``1/2 eps0 scale`` times the
    objective's derivative by that energy contracts it.

    Args:
        name: The term's key in the pairs' weights.
        coefficient: ``Re(eps_r)`` on ``elements`` (zero elsewhere), as the
            node pass would transport it: the whole tensor, or with
            ``axis`` its part ``Re(eps_k) a_k a_k^T`` along one principal
            axis.
        elements: One boolean per tetrahedron (gfem order).
        axis: The principal axis of a share along one (``None``: the whole
            permittivity). Its derivative is along that axis only, and an
            isotropic leaf has none.
    """
    mask = np.asarray(elements, dtype=float)
    if axis is None:
        isotropic: Any = lambda: mask  # noqa: E731
        per_axis = np.repeat(mask[:, None], 3, axis=1)
    else:
        isotropic = 0.0
        per_axis = np.zeros((len(mask), 3))
        per_axis[:, axis] = mask
    return Term(
        name,
        "value",
        coefficient=coefficient,
        derivatives={
            "eps_r": Derivative(
                isotropic,
                holomorphic=False,
                per_axis=Derivative(lambda: per_axis, holomorphic=False),
            )
        },
    )


#: Operator families whose rules' terms the node pass reads.
SHAPE_FORMS = frozenset(
    {"electrostatic", "driven", "eigenmode", "piezoelectric", "port mode"}
)
#: Operator families whose pairs carry the sheet pieces' weights (by name).
SHEET_FORMS = frozenset({"driven", "eigenmode", "port mode"})
#: Operator families whose pairs carry the lumped pieces' weights (by name).
LUMPED_FORMS = frozenset({"driven", "eigenmode", "port mode"})
#: Operator families whose pairs carry the law pieces' dθ/dp (`law_key`).
LAW_FORMS = frozenset({"driven", "eigenmode", "electrostatic", "port mode"})
#: Operator families with a load pass: their problem leaves are
#: read, the power density by the pass, the others by the rule's explicit terms.
LOAD_FORMS = frozenset({"steady_thermal", "transient_thermal"})
#: The input kind of a load form's power density per tetrahedron.
LOAD_KIND = ("problem", ".density")
#: The input kind of a load form's power density per triangle (W/m^2).
FACE_LOAD_KIND = ("problem", ".face_density")
#: The input kind of a bent port mode's radius (`gfem.Bend.radius`): the port
#: mode rule differentiates it itself, as a derivative of its 2D forms.
BEND_RADIUS_KIND = ("problem", ".bend.radius")
#: The lumped values that are design inputs, by owner.
_LUMPED = {
    "port": ("inductance", "capacitance"),
    "boundary": ("resistance", "inductance", "capacitance"),
}


def is_sheet(kind: tuple) -> bool:
    """Whether input ``kind`` is the fill of a boundary's sheet field."""
    return kind[0] == "boundary" and tuple(kind[2:]) == ("fill", "fill")


def is_load(kind: tuple) -> bool:
    """Whether input ``kind`` is a load form's power density (`LOAD_KIND`)."""
    return tuple(kind) == LOAD_KIND


def is_lumped(kind: tuple) -> bool:
    """Whether input ``kind`` is a lumped value: a port's L or C, an `Impedance`'s."""
    return len(kind) == 3 and kind[2] in _LUMPED.get(kind[0], ())


def is_thermal_value(kind: tuple) -> bool:
    """Whether input ``kind`` is a thermal boundary value (a temperature, an h...)."""
    return kind[0] == "boundary" and len(kind) == 3 and kind[2] in THERMAL_VALUES


def law_key(name: str, parameter: tuple) -> str:
    """The weight key of dθ/dp of piece ``name`` by a law parameter.

    ``parameter`` is a law input kind's tail: ``("eps_inf",)`` or
    ``(term_index, name)``.
    """
    return f"{name} d/d " + ".".join(str(p) for p in parameter)


def unsupported(form: str, kind: tuple) -> str | None:  # noqa: PLR0911
    """Why the kernel cannot differentiate ``form`` by input ``kind``, or None."""
    for kinds, forms, what in (
        (is_sheet, SHEET_FORMS, "sheet fields"),
        (is_lumped, LUMPED_FORMS, "lumped values"),
    ):
        if kinds(kind):
            read = form in forms
            return (
                None if read else f"the {form} forms of this version do not read {what}"
            )
    if is_thermal_value(kind):
        if form in LOAD_FORMS:
            return None
        return (
            "gradients with respect to thermal boundary values are not available "
            f"for {form} results in this version of gfem"
        )
    if kind[0] == "material":
        if kind[2] in PROPERTIES.get(form, frozenset()):
            return None
        return f"the {form} forms of this version do not differentiate {kind[2]}"
    if kind[0] == "law":
        if form in LAW_FORMS:
            return None
        return (
            "gradients with respect to a permittivity law's parameters are not "
            "available in this version of gfem"
        )
    if kind[0] == "nodes":
        if form in SHAPE_FORMS:
            return None
        return (
            f"gradients of {form} results with respect to mesh node coordinates "
            "(shape gradients) are not available in this version of gfem"
        )
    if kind[0] == "problem" and form in LOAD_FORMS:
        return None
    if kind == BEND_RADIUS_KIND and form == "port mode":
        return None
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


def _rows(copies: int, n: int) -> Callable[[ngs.LinearForm], np.ndarray]:
    """What reads a form on ``copies`` of a space of ``n`` DOFs as ``(copies, n)``.

    With the count given, not inferred: a rank with no elements has ``n = 0``.
    """
    return lambda form: form.vec.FV().NumPy().reshape(copies, n)


def _surface_tests(
    built: Any, *, is_complex: bool, copies: int = 1
) -> tuple[ngs.FESpace, list, Callable[[ngs.LinearForm], np.ndarray]]:
    """A P0 space on the surface elements, as a pass needs it.

    Returns:
        The space (``copies`` of it, compound, above 1); its test functions, one
        per copy; and what reads an assembled form as ``(copies, n)``, one value per
        local surface element. Under ranks the space is `facet_space`, whose DOFs
        are not the elements'.
    """
    if built.comm is None:
        one = ngs.SurfaceL2(built.mesh, order=0, complex=is_complex)
        dofs = None
    else:
        one, dofs = facet_space(built, is_complex=is_complex)
    space = one if copies == 1 else ngs.FESpace([one] * copies)
    tests = space.TestFunction()
    tests = [tests] if copies == 1 else list(tests)
    if dofs is not None:
        tests = [q.Trace() for q in tests]

    rows = _rows(copies, one.ndof)
    if dofs is None:
        return space, tests, rows
    return space, tests, lambda form: rows(form)[:, dofs]


#: Columns of one pair integrated in one coefficient function: a wider pair
#: is split, so a reverse sweep's many columns keep each integrand small.
_COLUMNS_PER_INTEGRAND = 8


def _chunks(pairs: list[FieldPair]) -> list[FieldPair]:
    """``pairs`` with at most `_COLUMNS_PER_INTEGRAND` columns each."""
    out = []
    for pair in pairs:
        left, right = _columns(pair.left), _columns(pair.right)
        if left.shape[1] <= _COLUMNS_PER_INTEGRAND:
            out.append(pair)
            continue
        for start in range(0, left.shape[1], _COLUMNS_PER_INTEGRAND):
            stop = start + _COLUMNS_PER_INTEGRAND
            out.append(
                dataclasses.replace(
                    pair, left=left[:, start:stop], right=right[:, start:stop]
                )
            )
    return out


def _element_pass(
    model: Model,
    pairs: list[FieldPair],
    term: Term,
    length_power: int,
    comm: Any = None,
    *,
    integrated: Sequence[Integrated] = (),
) -> np.ndarray:
    """``sum_pairs theta_t sum_k a_t,e(left_k, right_k)`` per gfem tetrahedron.

    Times ``scale ** length_power``, with the rule's own ``integrated``
    values of the term added. A face term's integrals per triangle
    are summed into each triangle's owning tetrahedron (`Term.owner`), which may
    lie on another rank: the sum at the backward's end takes care of that.
    """
    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
    integrand = INTEGRANDS[term.kind]
    name = term.name
    pairs = _chunks([p for p in pairs if name in p.weights])
    given = [np.asarray(i.values[name]) for i in integrated if name in i.values]
    is_complex = any(
        np.iscomplexobj(p.left)
        or np.iscomplexobj(p.right)
        or np.iscomplexobj(p.weights[name])
        for p in pairs
    ) or any(np.iscomplexobj(g) for g in given)
    out = np.zeros(
        len(model.mesh.volume_elements), dtype=complex if is_complex else float
    )
    for values in given:
        out = out + values
    if not pairs:
        return out * built.scale**length_power
    owner = term.owner
    if owner is None:
        p0 = ngs.L2(built.mesh, order=0, complex=is_complex)
        q = p0.TestFunction()
        form = ngs.LinearForm(p0)
        for pair in pairs:
            dx = quadrature.dx(pair.order, definedon=pair.domain)
            form += pair.weights[name] * _pair_integrand(integrand, pair) * q * dx
        form.Assemble()
        out[built.cell_order] += form.vec.FV().NumPy()
        return out * built.scale**length_power
    p0, (q,), read = _surface_tests(built, is_complex=is_complex)
    form = ngs.LinearForm(p0)
    for pair in pairs:
        ds = quadrature.ds(pair.order, definedon=term.region)
        form += pair.weights[name] * _pair_integrand(integrand, pair) * q * ds
    form.Assemble()
    per_face = np.zeros(len(owner), dtype=out.dtype)
    per_face[built.facet_order] = read(form)[0]
    on = owner >= 0
    np.add.at(out, owner[on], per_face[on])
    return out * built.scale**length_power


def _point_pass(
    model: Model, pairs: list[FieldPair], term: Term, points: Any, comm: Any = None
) -> np.ndarray:
    """``sum_pairs theta_t sum_k a_t(left_k, right_k)`` at each point of ``points``.

    On the pinned rule's points of a layer grid's region (a `PointMap`): the
    rule space's test function gives each point its weight times the
    integrand there, times ``scale ** term.length_power``.
    """
    from gfem._engine.points import rule_space

    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
    name = term.name
    pairs = _chunks([p for p in pairs if name in p.weights])
    out = np.zeros(len(points.dofs), dtype=complex)
    if not pairs:
        return out
    (order,) = {p.order for p in pairs}
    space = rule_space(model, order, is_complex=True)
    q = space.TestFunction()
    form = ngs.LinearForm(space)
    integrand = INTEGRANDS[term.kind]
    dx = quadrature.dx(order, definedonelements=points.cells)
    for pair in pairs:
        form += pair.weights[name] * _pair_integrand(integrand, pair) * q * dx
    form.Assemble()
    out += form.vec.FV().NumPy()[points.dofs]
    return out * built.scale**term.length_power


def _point_sensitivity(
    model: Model,
    pairs: list[FieldPair],
    wants: set[Hashable],
    terms: Sequence[Term],
    comm: Any = None,
) -> dict[Hashable, np.ndarray]:
    """The pixel cotangent of each wanted layer grid's density, ``(ny * nx,)``.

    Each term on the grid's points contributes its derivative per point
    times its point pass; the bilinear map's transpose carries the sum to
    the pixels.
    """
    from gfem._engine.points import geometry_of, point_map

    out: dict[Hashable, np.ndarray] = {}
    orders = {p.order for p in pairs if isinstance(p, FieldPair)}
    for kind in wants:
        region = kind[1]  # ty: ignore[not-subscriptable]
        total: np.ndarray = np.zeros(0, dtype=complex)
        found = None
        for term in terms:
            derivative = term.derivatives.get("blend")
            if term.points != region or derivative is None:
                continue
            if found is None:
                (order,) = orders
                found = point_map(model, region, order)
                total = np.zeros(len(found.dofs), dtype=complex)
            total = total + derivative.evaluated() * _point_pass(
                model, pairs, term, found, comm
            )
        if found is None:
            shape = geometry_of(model, region).shape
            out[kind] = np.zeros(shape[0] * shape[1])
            continue
        out[kind] = found.matrix.T @ total
    return out


def _on_points(model: Model, kind: tuple) -> bool:
    """Whether material ``kind`` is a layer grid's pixels (`gfem._engine.points`)."""
    material = model.materials.get(kind[1])
    return (
        kind[2] == "blend"
        and isinstance(material, MaterialField)
        and material.blend_ends is not None
        and material.blend_ends.pixels is not None
    )


def element_integrals(
    model: Model, pairs: list[FieldPair], terms: Sequence[Term], comm: Any = None
) -> dict[str, np.ndarray]:
    """Each volume term's ``sum_pairs theta_t a_t,e(left, right)`` per tetrahedron.

    On the mesh in file units (no ``scale`` power): what a rule that
    contracts its pairs as it goes sums into an `Integrated`.
    """
    return {
        term.name: _element_pass(model, pairs, term, 0, comm)
        for term in terms
        if term.owner is None and any(term.name in p.weights for p in pairs)
    }


def _load_pass(
    model: Model, loads: Sequence[LoadField], comm: Any = None
) -> np.ndarray:
    """``sum_loads scale ** length_power int_e Lambda`` per gfem tetrahedron."""
    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
    out = np.zeros(len(model.mesh.volume_elements))
    if not loads:
        return out
    p0 = ngs.L2(built.mesh, order=0)
    q = p0.TestFunction()
    for load in loads:
        gf = ngs.GridFunction(load.space)
        gf.vec.FV().NumPy()[:] = np.asarray(load.values, dtype=float)
        form = ngs.LinearForm(p0)
        form += gf * q * quadrature.dx(load.order, definedon=load.domain)
        form.Assemble()
        out[built.cell_order] += form.vec.FV().NumPy() * built.scale**load.length_power
    return out


def _face_load_pass(
    model: Model, loads: Sequence[LoadField], comm: Any = None
) -> np.ndarray:
    """``sum_loads scale ** (length_power - 1) int_f Lambda`` per gfem triangle.

    A face load ``int_S q_s psi`` is one length power below its volume
    load ``int q psi``, so the pass reads each `LoadField` one power lower.
    """
    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
    out = np.zeros(len(model.mesh.surface_elements))
    if not loads:
        return out
    space, (q,), read = _surface_tests(built, is_complex=False)
    for load in loads:
        gf = ngs.GridFunction(load.space)
        gf.vec.FV().NumPy()[:] = np.asarray(load.values, dtype=float)
        form = ngs.LinearForm(space)
        form += gf * q * quadrature.ds(load.order)
        form.Assemble()
        scale = built.scale ** (load.length_power - 1)
        out[built.facet_order] += read(form)[0] * scale
    return out


def _moment_pass(
    model: Model,
    pairs: list[FieldPair],
    term: Term,
    length_power: int,
    comm: Any = None,
) -> np.ndarray:
    """``sum_pairs theta_t sum_k int sym(left_k (x) right_k)``, per element.

    Of the values, curls, gradients or tangential traces as the kind reads
    them (`MOMENTS`), times ``scale ** length_power``: ``(n_tetrahedra, 3,
    3)`` for a volume term, ``(n_triangles, 3, 3)`` per gfem triangle for a
    face term (not summed into the owners: its derivative is per face).
    The trace of each moment is `_element_pass`'s integral.
    """
    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
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
    count = len(model.mesh.volume_elements) if owner is None else len(owner)
    out = np.zeros((count, 3, 3), dtype=complex if is_complex else float)
    if not pairs:
        return out
    if face:
        space, tests, read = _surface_tests(
            built, is_complex=is_complex, copies=len(_SYMMETRIC)
        )
        order = built.facet_order
    else:
        p0 = ngs.L2(built.mesh, order=0, complex=is_complex)
        space = ngs.FESpace([p0] * len(_SYMMETRIC))
        tests = space.TestFunction()
        read = _rows(len(_SYMMETRIC), p0.ndof)
        order = built.cell_order
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
    values = read(form)
    for c, (i, j) in enumerate(_SYMMETRIC):
        out[order, i, j] = values[c]
        out[order, j, i] = values[c]
    return out * built.scale**length_power


def _moments(
    built: Any, pair: FieldPair, blocks: Sequence[MomentBlock]
) -> dict[str, np.ndarray]:
    r"""Each block's :math:`\sum_k \int \mathrm{entries}(l_k, r_k)` per tetrahedron."""
    blocks = [b for b in blocks if any(pair.weights.get(t, 0) != 0 for t in b.terms)]
    total = sum(b.size for b in blocks)
    if not total:
        return {}
    p0 = ngs.L2(built.mesh, order=0, complex=True)
    space = p0 if total == 1 else ngs.FESpace([p0] * total)
    tests = [space.TestFunction()] if total == 1 else list(space.TestFunction())
    form = ngs.LinearForm(space)
    start = 0
    for block in blocks:
        m = _pair_integrand(block.entries, pair)
        parts = [m[i] * tests[start + i] for i in range(block.size)]
        form += sum(parts[1:], parts[0]) * quadrature.dx(
            pair.order, definedon=block.domain
        )
        start += block.size
    form.Assemble()
    values = _rows(total, p0.ndof)(form)
    out = {}
    start = 0
    for block in blocks:
        each = np.zeros((len(built.cell_order), block.size), dtype=complex)
        each[built.cell_order] = values[start : start + block.size].T
        out[block.name] = each
        start += block.size
    return out


def _tensor_pass(
    model: Model,
    pairs: list[FieldPair],
    tensors: Tensors,
    comm: Any = None,
) -> dict[Hashable, np.ndarray]:
    """The cotangent of each tensor leaf: moments per pair, contracted, summed."""
    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
    out: dict[Hashable, np.ndarray] = {}
    for pair in _chunks(pairs):
        moments = _moments(built, pair, tensors.blocks)
        for kind, value in tensors.contract(pair.weights, moments).items():
            out[kind] = out[kind] + value if kind in out else value
    return out


def _sheet_pass(
    model: Model,
    pairs: list[FieldPair],
    index: int,
    terms: Sequence[Term],
    comm: Any = None,
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
    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
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
    p0, (q,), read = _surface_tests(built, is_complex=True)
    form = ngs.LinearForm(p0)
    for weight, pair in weighted:
        ds = quadrature.ds(pair.order, definedon=region)
        form += weight * _pair_integrand(_tangential, pair) * q * ds
    form.Assemble()
    per_face = np.zeros(len(model.mesh.triangles), dtype=complex)
    per_face[built.facet_order] = read(form)[0]
    return per_face[faces]


def _lumped_pass(
    model: Model,
    pairs: list[FieldPair],
    kind: tuple,
    terms: Sequence[Term],
    comm: Any = None,
) -> complex:
    """``sum_pairs sum_t theta_t slope_t int c_t left_t . right_t``: one lumped value.

    ``t`` runs over the terms of the pieces that move with input ``kind``
    (`Term.lumped`), one per surface of a piece; ``dA_t / dp = slope_t A_t``.
    The terms are in the driven equation's units (length power 0). Under ranks
    the value is this rank's faces' share, as the sheet pass's faces are.
    """
    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
    weighted = []
    for term in terms:
        if term.lumped is None or term.lumped[0] != kind:
            continue
        for pair in pairs:
            weight = pair.weights.get(term.name, 0.0) * term.lumped[1]
            if weight != 0:
                weighted.append((complex(weight), term, pair))
    if not weighted:
        return 0j
    p0, (q,), read = _surface_tests(built, is_complex=True)
    form = ngs.LinearForm(p0)
    for weight, term, pair in weighted:
        ds = quadrature.ds(pair.order, definedon=term.region)
        form += weight * term.coefficient * _pair_integrand(_tangential, pair) * q * ds
    form.Assemble()
    return complex(read(form)[0].sum())


def _law_pass(
    model: Model,
    pairs: list[FieldPair],
    kinds: list[tuple],
    terms: Sequence[Term],
    comm: Any = None,
) -> dict[Hashable, np.ndarray]:
    """``sum_pairs sum_t (dθ_t/dp) int c_t left . right`` per law parameter.

    ``t`` runs over the terms of each wanted parameter's law region
    (`Term.law`); dθ_t/dp is the pair's weight under `law_key`. Each term is
    integrated once per pair that carries one of its wanted keys, and the
    scalar is reused for every parameter. One complex number per parameter,
    ``sum ct * dy/dp`` (real-linear: the caller takes the real part for a real
    leaf). Under ranks it is this rank's elements' share.
    """
    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
    out: dict[Hashable, np.ndarray] = {k: np.asarray(0j) for k in kinds}
    for term in terms:
        mine = [k for k in kinds if k[1] == term.law]
        if term.law is None or not mine:
            continue
        keys = {k: law_key(term.name, tuple(k[2:])) for k in mine}
        integrand = INTEGRANDS[term.kind]
        for pair in pairs:
            if not any(key in pair.weights for key in keys.values()):
                continue
            value = _law_integral(built, term, integrand, pair)
            value *= built.scale**term.length_power
            for kind, key in keys.items():
                weight = pair.weights.get(key)
                if weight is not None:
                    out[kind] = out[kind] + complex(weight) * value
    return out


def _law_integral(
    built: Any, term: Term, integrand: Callable[[Any, Any], Any], pair: FieldPair
) -> complex:
    """``int c_t sum_k a(left_k, right_k)``: one law term on one pair, this rank's."""
    cf = term.coefficient * _pair_integrand(integrand, pair)
    if term.owner is None:
        p0 = ngs.L2(built.mesh, order=0, complex=True)
        form = ngs.LinearForm(p0)
        form += (
            cf * p0.TestFunction() * quadrature.dx(pair.order, definedon=pair.domain)
        )
        form.Assemble()
        return complex(form.vec.FV().NumPy().sum())
    p0, (q,), read = _surface_tests(built, is_complex=True)
    form = ngs.LinearForm(p0)
    form += cf * q * quadrature.ds(pair.order, definedon=term.region)
    form.Assemble()
    return complex(read(form)[0].sum())


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


#: The coordinates a position-dependent coefficient is differentiated by.
_COORDINATES = (ngs.x, ngs.y, ngs.z)


def _position_part(term: Term, u: Any, v: Any, slopes: list) -> list:
    """``(d_j C u) . v`` (curls for a curl term) per axis ``j``: the moved point."""
    if term.kind == "curl":
        u, v = ngs.curl(u), ngs.curl(v)
    return [(slope * u) * v for slope in slopes]


def _volume_shape(
    terms: Sequence[tuple[Term, complex]],
    pair: FieldPair,
    w: Any,
    elements: Any = None,
    *,
    forward_rule: bool = False,
) -> Any:
    """``int Re(sum_t theta_t sum_k T_t(left_k, right_k)) : grad W``, one pair.

    On the pair's fields as they are (complex fields stay complex: no
    ``DiffShape``, no splitting). Value terms are integrated on the pinned
    rule; curl and gradient terms, products of first derivatives, on the
    exact rule of their lower degree (`quadrature.dx_derivatives`), on the
    NGSolve ``elements`` the terms share (`Term.elements`; ``None``: all).

    A term whose coefficient varies in position (`Term.position`) adds
    ``int Re(theta_t ((W . grad) C u) . v)`` (curls for a curl term): its
    coefficient is read at the moved quadrature points. Such a term is
    integrated on the pinned rule, as its forward form is.

    ``forward_rule`` integrates every term on the pinned rule of the forward
    form: on a curved element the integrands are rational in the reference
    coordinates, no rule is exact, and the form is the derivative of the
    discretized forward form only on the forward's own rule.
    """
    gw = ngs.grad(w)
    totals: dict[bool, Any] = {}
    moved: list[Any] = [None, None, None]
    slopes: dict[int, list] = {}
    for a, lefts in _parts(pair.space, pair.left):
        for b, rights in _parts(pair.space, pair.right):
            for term, theta in terms:
                weight = complex(theta * a * b)
                factor: complex | float = weight.real if weight.imag == 0 else weight
                derivatives = term.kind != "value" and not term.position
                if term.position and id(term) not in slopes:
                    slopes[id(term)] = [term.coefficient.Diff(c) for c in _COORDINATES]
                for u, v in zip(lefts, rights, strict=True):
                    part = factor * _transport(term, u, v)
                    total = totals.get(derivatives)
                    totals[derivatives] = part if total is None else total + part
                    if not term.position:
                        continue
                    for j, at in enumerate(
                        _position_part(term, u, v, slopes[id(term)])
                    ):
                        at = factor * at  # noqa: PLW2901
                        moved[j] = at if moved[j] is None else moved[j] + at
    out = None
    for derivatives, total in totals.items():
        tensor = total.real if total.is_complex else total
        shortcut = derivatives and not forward_rule
        rule = quadrature.dx_derivatives if shortcut else quadrature.dx
        dx = rule(pair.order, definedon=pair.domain, definedonelements=elements)
        part = ngs.InnerProduct(gw, tensor) * dx
        out = part if out is None else out + part
    if moved[0] is not None:
        scalar = sum(
            (m.real if m.is_complex else m) * w[j] for j, m in enumerate(moved)
        )
        dx = quadrature.dx(
            pair.order, definedon=pair.domain, definedonelements=elements
        )
        part = scalar * dx
        out = part if out is None else out + part
    return out


def _field_shape(
    terms: Sequence[tuple[Term, complex]], pair: FieldPair, w: Any, elements: Any = None
) -> np.ndarray:
    """``Q_W e``: the shape form of `_volume_shape` as a vector on the pair's space.

    The field ``e`` is the pair's right column and the test function ``v`` of
    the pair's space stands in the left slot, so that for a pair
    ``(conj(e), e)`` of lossless terms (real weights, which makes the form
    real symmetric) ``Re(conj(e) . Q_W e)`` is `_volume_shape` assembled on
    ``w``. ``w`` is a ``VectorH1`` order-1 grid function and the terms are
    weighted (``theta_t scale**length_power``). A coefficient that varies in
    position adds no moved-point part: the caller's elements are where it is
    constant (the shell of a body, away from any stretch).
    The result is in the units of the form on the mesh: divide by the length
    unit for a per metre value, as `_node_pass` does.

    Returns:
        The complex vector of the real and imaginary parts assembled apart.
    """
    gw = ngs.grad(w)
    v = pair.space.TestFunction()
    out = np.zeros(pair.space.ndof, complex)
    for factor, fields in _parts(pair.space, pair.right):
        form = ngs.LinearForm(pair.space)
        for term, theta in terms:
            assert theta.imag == 0  # noqa: S101 - the real symmetric form
            rule = quadrature.dx if term.kind == "value" else quadrature.dx_derivatives
            dx = rule(pair.order, definedon=pair.domain, definedonelements=elements)
            # InnerProduct conjugates its second argument: the real grad W goes there
            form += (
                theta.real * ngs.InnerProduct(_transport(term, v, fields[0]), gw) * dx
            )
        form.Assemble()
        out += factor * form.vec.FV().NumPy()
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
                ds = quadrature.ds(
                    pair.order, definedon=term.region, definedonelements=term.elements
                )
                part = integrand * ds
                out = part if out is None else out + part
    return out


def _real_parts(
    space: ngs.FESpace, vectors: np.ndarray, comm: Any = None
) -> list[tuple[float, list]]:
    """``vectors`` as ``[(sign, real fields on space)]``: ``Re(l . s)`` pieces.

    ``Re(a(l, s)) = a(Re l, Re s) - a(Im l, Im s)`` for a real bilinear form,
    so the left fields carry the sign (+1 real part, -1 imaginary part).
    """
    out = [(1.0, _gridfunctions(space, np.ascontiguousarray(vectors.real)))]
    has_imaginary = np.iscomplexobj(vectors) and bool(vectors.imag.any())
    if _mpi.agree_any(comm, flag=has_imaginary):
        out.append((-1.0, _gridfunctions(space, np.ascontiguousarray(vectors.imag))))
    return out


def _surface_shape(
    terms: Sequence[tuple[Term, complex]], pair: FieldPair, w: Any, comm: Any = None
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
            lefts[id(space)] = _real_parts(space, _columns(pair.left), comm)
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


def _nodal(out: np.ndarray, values: np.ndarray, built: Any) -> None:
    r"""Add the nodal cotangents of an order-2 shape form's values to ``out``.

    ``values`` are the form's values on the hierarchical basis of NGSolve's
    order-2 ``H1``: a vertex function is the barycentric coordinate
    :math:`\lambda_v` and an edge function is :math:`-\tfrac12\lambda_a\lambda_b`.
    The nodal basis is :math:`N_v = \lambda_v + 4\sum_{e \ni v}\varphi_e` and
    :math:`N_m = -8\varphi_e`, so with :math:`L` the form's values

    .. math:: \bar X_v = L_v + 4\sum_{e \ni v} L_e, \qquad \bar X_m = -8 L_e.
    """
    nv = len(built.node_order)
    edges = built.edge_vertices
    vertex = values[:nv].copy()
    edge = values[nv : nv + len(edges)]
    np.add.at(vertex, edges[:, 0], 4 * edge)
    np.add.at(vertex, edges[:, 1], 4 * edge)
    np.add.at(out, built.node_order, vertex)
    np.add.at(out, built.edge_nodes, -8 * edge)


def _node_pass(
    model: Model, pairs: list[FieldPair], terms: Sequence[Term], comm: Any = None
) -> np.ndarray:
    """``sum_pairs sum_t theta_t d a_t(left, right) / dX``, (n_nodes, dim), per m.

    A 2D mesh gives only ``"surface"`` terms, whose ``DiffShape`` carries the
    whole transport. The form holds a load: a rank's part at each of its
    vertices, shared ones too.

    On a curved mesh the velocity is ``VectorH1`` of order 2, so that it moves
    corners and mid-edge nodes alike, every shape term is integrated on the
    forward's rule, and the hierarchical edge values map to nodal cotangents
    (`_nodal`).
    """
    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
    n_nodes = len(model.mesh.imported_nodes)
    curved = built.edge_nodes is not None
    with serial_spaces():
        space = ngs.VectorH1(built.mesh, order=2 if curved else 1)
    w = space.TestFunction()
    form = ngs.LinearForm(space)
    empty = True
    for pair in pairs:
        volume: dict[int, list] = {}
        traces, surface = [], []
        for term in terms:
            theta = pair.weights.get(term.name)
            if theta is None or theta == 0:
                continue
            weighted = (term, theta * built.scale**term.length_power)
            if term.kind == "tangential gradient":  # pragma: no cover
                # a 2D model's rule refuses the node coordinates first
                msg = f"The term {term.name!r} has no shape derivative."
                raise NonDifferentiableError(msg)
            if term.kind == "surface":
                surface.append(weighted)
            elif term.kind in ("trace", "curl trace"):
                traces.append(weighted)
            else:
                volume.setdefault(id(term.elements), []).append(weighted)
        for part in (
            *(
                _volume_shape(v, pair, w, v[0][0].elements, forward_rule=curved)
                for v in volume.values()
            ),
            _trace_shape(traces, pair, w) if traces else None,
            _surface_shape(surface, pair, w, comm) if surface else None,
        ):
            if part is not None:
                form += part
                empty = False
    dim = model.mesh.dim
    out = np.zeros((n_nodes, dim))
    if empty:
        return out
    form.Assemble()
    # a rank with no elements has no vertices: its count is not to be inferred
    values = form.vec.FV().NumPy().reshape(dim, space.ndof // dim).T
    if curved:
        _nodal(out, values, built)
        return out / built.scale
    np.add.at(out, built.node_order, values[: len(built.node_order)])
    # per metre: the engine mesh is in file units of `scale` metres
    return out / built.scale


def sensitivity(  # noqa: PLR0912
    model: Model,
    pairs: Sequence[FieldPair | LoadField | Integrated],
    wants: set[Hashable],
    *,
    form: str,
    terms: Callable[[bool], Sequence[Term]] | None = None,
    lumped: Callable[[], Sequence[Term]] | None = None,
    tensors: Callable[[], Tensors] | None = None,
    comm: Any = None,
) -> dict[Hashable, np.ndarray]:
    """The cotangent of every wanted design input, from the rule's pairs.

    Args:
        model: The design state (concrete leaves) the pairs were solved on.
        pairs: The rule's field pairs, cotangents folded in; a load form's
            adjoint (`LoadField`) and the integrals it contracted itself
            (`Integrated`) among them.
        wants: Input kinds, ``("material", region, property)``,
            ``("boundary", index, "fill", "fill")`` or ``("nodes",)``.
        form: The operator family, a key of `PROPERTIES`.
        terms: ``terms(nodes)`` returns the operator's `Term` list, called
            once when anything is wanted; ``nodes`` is whether the node
            coordinates are.
        lumped: ``lumped()`` returns the lumped pass's terms (`Term.lumped`),
            called once when a lumped value is wanted.
        tensors: ``tensors()`` returns the rule's `Tensors`, called once when a
            material leaf is wanted; it replaces the terms' material pass.
        comm: The resolved communicator, which names the pairs' discretization. The
            passes are local: under ranks each result holds this rank's elements in
            global zeros, and the caller sums over ranks.

    Returns:
        Per wanted material kind, one value per tetrahedron of the region
        (mesh order), or ``(n, 3)`` (one per principal axis) for a field
        with `axes`: the real cotangent, or for a complex dependence the
        complex one (``sum ct * dy/dp`` when holomorphic, see
        `Derivative`). Per sheet fill, one complex value per face of
        the surface (``mesh.elements(surface)`` order), ``sum ct * dy/dfill``.
        For ``("nodes",)``, ``(n_nodes, 3)``, real, per metre. Per lumped
        value (``("port", index, "inductance")``, ``("boundary", index,
        "resistance")``, ...), one complex number, ``sum ct * dy/dp``.

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
    loads = [p for p in pairs if isinstance(p, LoadField)]
    integrated = [p for p in pairs if isinstance(p, Integrated)]
    pairs = [p for p in pairs if isinstance(p, FieldPair)]
    if LOAD_KIND in wants:
        out[LOAD_KIND] = _load_pass(model, loads, comm)
    if FACE_LOAD_KIND in wants:
        out[FACE_LOAD_KIND] = _face_load_pass(model, loads, comm)
    # a load form's other problem leaves are the rule's explicit terms
    wants = {k for k in wants if k[0] != "problem"}  # ty: ignore[not-subscriptable]
    if tensors is not None:
        tensor_kinds = {k for k in wants if k[0] == "material"}  # ty: ignore[not-subscriptable]
        if tensor_kinds:
            out |= _tensor_pass(model, pairs, tensors(), comm)
        wants = wants - tensor_kinds
    if not wants:
        return out
    if terms is None:  # pragma: no cover - rule bug
        msg = f"The {form} rule gives no terms."
        raise NotImplementedError(msg)
    operator = terms(("nodes",) in wants)
    for kind in wants:
        if is_sheet(kind):  # ty: ignore[invalid-argument-type]
            out[kind] = _sheet_pass(model, pairs, kind[1], operator, comm)  # ty: ignore[not-subscriptable]
    values = [k for k in wants if is_lumped(k)]  # ty: ignore[invalid-argument-type]
    if values:
        if lumped is None:  # pragma: no cover - rule bug
            msg = f"The {form} rule gives no lumped terms."
            raise NotImplementedError(msg)
        lumped_terms = lumped()
        for kind in values:
            out[kind] = np.asarray(
                _lumped_pass(model, pairs, kind, lumped_terms, comm)  # ty: ignore[invalid-argument-type]
            )
    laws = [k for k in wants if k[0] == "law"]  # ty: ignore[not-subscriptable]
    if laws:
        out |= _law_pass(model, pairs, laws, operator, comm)  # ty: ignore[invalid-argument-type]
    if ("nodes",) in wants:
        out["nodes",] = _node_pass(model, pairs, operator, comm)
    materials = {k for k in wants if k[0] == "material"}  # ty: ignore[not-subscriptable]
    if materials:
        out |= _material_sensitivity(
            model, pairs, materials, operator, comm, integrated=integrated
        )
    return out


def _per_axis(model: Model, kind: tuple) -> bool:
    """Whether material ``kind`` is a leaf of principal values (field `axes`)."""
    material = model.materials.get(kind[1])
    return isinstance(material, MaterialField) and material.has_principal_values(
        kind[2]
    )


def _material_sensitivity(
    model: Model,
    pairs: list[FieldPair],
    wants: set[Hashable],
    terms: Sequence[Term],
    comm: Any = None,
    *,
    integrated: Sequence[Integrated] = (),
) -> dict[Hashable, np.ndarray]:
    """The material part of `sensitivity`: each property's terms, in their order."""
    point_kinds = {k for k in wants if _on_points(model, k)}  # ty: ignore[invalid-argument-type]
    on_points = _point_sensitivity(model, pairs, point_kinds, terms, comm)
    wants = wants - point_kinds
    axis_kinds = {k for k in wants if _per_axis(model, k)}  # ty: ignore[invalid-argument-type]
    integrals: dict[tuple[str, int], np.ndarray] = {}
    per_property: dict[str, np.ndarray] = {}
    for prop in sorted({k[2] for k in wants - axis_kinds}):  # ty: ignore[not-subscriptable]
        total: np.ndarray = np.zeros(len(model.mesh.volume_elements))
        for term in terms:
            derivative = term.derivatives.get(prop)
            if derivative is None or term.points is not None:
                continue
            key = (term.name, term.length_power + derivative.length_power)
            if key not in integrals:
                integrals[key] = _element_pass(
                    model, pairs, term, key[1], comm, integrated=integrated
                )
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
        comm,
        integrated=integrated,
    )
    return on_points | {
        kind: (per_axis if kind in axis_kinds else per_property)[kind[2]][  # ty: ignore[not-subscriptable]
            model.mesh.elements(kind[1])  # ty: ignore[not-subscriptable]
        ]
        for kind in wants
    }


def _per_axis_sensitivity(
    model: Model,
    pairs: list[FieldPair],
    props: set[str],
    terms: Sequence[Term],
    comm: Any = None,
    *,
    integrated: Sequence[Integrated] = (),
) -> dict[str, np.ndarray]:
    """Each property's cotangent per tetrahedron and principal axis, ``(n, 3)``.

    From the terms' `Derivative.per_axis` forms and the moment pass (plus the
    rule's own ``integrated`` moments), on the axes of each tetrahedron's
    field (`element_axes`).

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
        total: np.ndarray = np.zeros((len(model.mesh.volume_elements), 3))
        for term in terms:
            derivative = term.derivatives.get(prop)
            if derivative is None or term.points is not None:
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
                moments[key] = _moment_pass(model, pairs, term, key[1], comm)
                given = [
                    i.moments[term.name] for i in integrated if term.name in i.moments
                ]
                if given and term.owner is None:
                    built = discretization(model.mesh, model._periodic, comm).built  # noqa: SLF001
                    moments[key] = moments[key] + sum(given) * built.scale ** key[1]
            moment = moments[key]
            if term.owner is None:
                # s_k a_k^T M a_k per tetrahedron and axis
                along = np.einsum("eki,eij,ekj->ek", axes, moment, axes)
                part = axis.evaluated() * along
            else:
                per_face = np.einsum("fkij,fij->fk", axis.evaluated(), moment)
                part = np.zeros((len(model.mesh.volume_elements), 3), per_face.dtype)
                on = term.owner >= 0
                np.add.at(part, term.owner[on], per_face[on])
            if not axis.holomorphic:
                part = np.real(part)
                if axis.chain is not None:
                    part = axis.chain() * part
            total = total + part
        out[prop] = np.asarray(total)
    return out
