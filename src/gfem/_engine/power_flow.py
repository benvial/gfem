"""The power a solved driven field carries out of a set of elements.

The host kernel of `gfem.power_flow`. With A the selected elements, the
driven equations ``a(E, v) = l(v)`` tested with ``v = conj(χ_A E)`` (E on
the DOFs of A's edges, faces and cells, zero elsewhere) split into the
terms on A, whose imaginary part is minus the power A's ports feed and
its losses absorb, and the terms on the outside elements and faces that
share an edge with A. So the outside layer's terms give the power crossing
A's boundary exactly for the discrete field, with no point evaluation:

    P = Im[conj(χ_A e) · Σ_k θ_k(ω) A_k|layer e] / (2 ω mu0 scale)

(the forms are scaled by ``mu0 * scale``, see `terms`). A face term belongs
to A only when every solve-domain element beside it is in A. A port's
term needs its excitation, which a field does not record, so an outside
port face that shares an edge with A raises. A Floquet surface is no
exception: inside A its terms (the absorbing term and the rank-one
corrections, which couple that surface's DOFs only) belong to A, and outside
A it raises when it shares an edge with A, as other ports do.

On a model with `gfem.Periodic` pairs the field holds e = P x on the
unpaired space, Bloch phases included, and the identity holds there with
the unpaired operator, once "shares an edge" and "the DOFs of A" are read
across the pairs: edges are keyed by the tie classes of their nodes, so an
edge and its images are one, and χ_A also keeps the DOFs of the images of
A's edges and faces (the test function ``conj(χ_A x)`` of the periodic
space, expanded). No wave vector is needed.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import ngsolve as ngs
import numpy as np
import scipy.constants as const

from gfem import _pml
from gfem._boundaries import FloquetPort, surfaces_of
from gfem._engine import probe
from gfem._engine.driven import DrivenOperator, driven_operator
from gfem._engine.mesh import inverse_order
from gfem._engine.sides import SideSpec, selection
from gfem._linalg.layout import ONE_PROCESS, DofLayout
from gfem._periodic import tie_classes
from gfem._pml import Stretch
from gfem.mesh import _TET_EDGES, Mesh, _same_topology

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model

__all__ = [
    "TRI_EDGES",
    "Flux",
    "Surface",
    "build",
    "check_field",
    "check_nodes",
    "check_size",
    "check_support",
    "dofs",
    "edge_keys",
    "local_coefficients",
    "node_pullback",
    "power_flow",
    "surface",
    "surface_of",
]

#: The three edges of a triangle, by vertex.
TRI_EDGES = np.array([[0, 1], [0, 2], [1, 2]])


@dataclasses.dataclass(frozen=True)
class Surface:
    """The element sets of a side's surface: topology only.

    Attributes:
        tie: The tie class of every node (`tie_classes`).
        inside: The selected tetrahedra of the field's support.
        layer: The outside tetrahedra that share an edge with ``inside``:
            the elements whose terms the power reads.
        rim: The inside tetrahedra that share an edge with the outside.
        faces: The triangles whose terms the power reads (`_outside_faces`).
    """

    tie: np.ndarray
    inside: np.ndarray
    layer: np.ndarray
    rim: np.ndarray
    faces: np.ndarray

    def read_elements(self, mesh: Mesh) -> np.ndarray:
        """The tetrahedra whose materials the power reads.

        The layer, and the solve-domain tetrahedra beside a face it reads.
        """
        out = self.layer.copy()
        nb = mesh.triangle_neighbours[self.faces]
        out[nb[nb >= 0]] = True
        return out


@dataclasses.dataclass(frozen=True)
class Flux:
    """The operator and test mask of a surface, ready to contract a field.

    ``P = Im[conj(D e) . M e] / c0`` with ``D = diag(mask)``, ``M`` the sum
    of the non-port pieces at ω and ``c0 = 2 ω mu0 scale``.

    Under ranks ``e`` is a rank's piece of the state, ``M e`` its part of a
    load (its own elements' terms) and the mask agreed on every holder of a
    DOF, so the product is the layout's pairing, reduced and broadcast: the
    same number on every rank. The pullback is then a rank's part of a load.
    """

    op: DrivenOperator
    mask: np.ndarray
    layout: DofLayout = ONE_PROCESS

    def _apply(self, x: np.ndarray, omega: float) -> np.ndarray:
        """``M x`` at ``omega``, ``x`` of shape ``(ndof,)`` or ``(ndof, m)``."""
        out = np.zeros(x.shape, complex)
        for k, piece in enumerate(self.op.pieces):
            if piece.kind != "port":
                out += complex(piece.theta(omega)) * self.op.apply(k, x)
        return out

    def _scale(self, omega: float) -> float:
        return 2 * omega * const.mu_0 * self.op.built.scale

    def power(self, e: np.ndarray, omega: float) -> float:
        """The power leaving the side, W."""
        w = np.where(self.mask, e, 0.0)
        paired = self.layout.pair(self._apply(e, omega), np.conj(w))
        return float(np.imag(paired)) / self._scale(omega)

    def pullback(self, e: np.ndarray, omega: float) -> np.ndarray:
        """JAX's gradient of `power` in ``e``: ``dP/dx - i dP/dy = 2 conj(H e)``.

        ``P = e^H H e`` with ``H = (D M - M^H D) / (2i c0)`` Hermitian, and
        ``M^H D e = conj(M conj(D e))``: the unpaired pieces are complex
        symmetric.
        """
        w = np.where(self.mask, e, 0.0)
        both = self._apply(np.stack([e, np.conj(w)], axis=1), omega)
        h = (np.where(self.mask, both[:, 0], 0.0) - np.conj(both[:, 1])) / (
            2j * self._scale(omega)
        )
        return 2 * np.conj(h)


def power_flow(field: Field, side: SideSpec, model: Model) -> float:
    """Net time-averaged power leaving ``side``, W (see `gfem.power_flow`).

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: The side, the field or a port near the surface does not
            fit (see `gfem.power_flow`).
    """
    check_nodes(field, model)
    where = surface(field, side, model)
    comm = field._communicator  # noqa: SLF001
    stretch = _pml.field_stretch(model, field)
    flux = build(field.order, model, where, stretch=stretch, comm=comm)
    e = local_coefficients(np.asarray(field.coefficients), flux.op, comm)
    check_size(e, flux.op, field.order)
    return flux.power(e, float(np.asarray(field.omega).real))


def local_coefficients(e: np.ndarray, op: DrivenOperator, comm: Any) -> np.ndarray:
    """A field's coefficients on ``op``'s space: under ranks, its piece unpadded.

    A rank's piece is padded to one width on every rank; its first
    ``op.space.ndof`` entries are this rank's DOFs.
    """
    return e if comm is None else e[: op.space.ndof]


def surface(field: Field, side: SideSpec, model: Model) -> Surface:
    """The surface of ``side``, from the mesh topology.

    A predicate side reads the node coordinates too.

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: The side or the field does not fit, or the surface is
            within one element of a port (see `gfem.power_flow`).
    """
    return surface_of(
        field, selection(model.mesh, side, refuse_surfaces="power_flow"), model
    )


def surface_of(field: Field, selected: np.ndarray, model: Model) -> Surface:
    """The surface of the selected tetrahedra (one boolean each), as `surface`.

    Raises:
        ValueError: The selection or the field does not fit, or the surface
            is within one element of a port (see `gfem.power_flow`).
    """
    mesh = model.mesh
    support = check_support(field, model)
    inside = selected & support
    outside = support & ~inside
    if not inside.any():
        raise ValueError(
            "The side selects no element of the field's support, so it has no "
            "surface: select elements of the solve domain."
        )
    if not outside.any():
        raise ValueError(
            "The side selects every element of the field's support, so it has "
            "no surface: the side holds the whole solve domain."
        )
    tie = tie_classes(len(mesh.nodes), model._periodic)  # noqa: SLF001
    tet_edges = edge_keys(tie, mesh.tetrahedra, np.array(_TET_EDGES))
    closure = np.unique(tet_edges[inside])
    layer = outside & np.isin(tet_edges, closure).any(axis=1)
    rim = inside & np.isin(tet_edges, np.unique(tet_edges[outside])).any(axis=1)
    faces = _outside_faces(mesh, tie, inside, support, closure)
    for port in model.ports:
        kind = "FloquetPort" if isinstance(port, FloquetPort) else "port"
        for name in surfaces_of(port):
            if faces[mesh.elements(name)].any():
                raise ValueError(
                    f"The surface of this side is within one element of {kind} "
                    f"{port.name!r}'s surface {name!r}: power_flow cannot "
                    "count a port's excitation. Move the surface at least one "
                    "element away from the port, or put the port's elements "
                    "inside the side."
                )
    _check_point_sources(field, layer)
    return Surface(tie=tie, inside=inside, layer=layer, rim=rim, faces=faces)


def _check_point_sources(field: Field, layer: np.ndarray) -> None:
    """Raise if a point dipole's load sits in the element layer the power reads.

    Its load would enter the residual as power crossing the surface.
    """
    sources = getattr(field, "_point_sources", ())
    if not sources:
        return
    from gfem._probe import layout as probe_layout

    points = np.array([position for _, position in sources], float)
    located = probe.locate(field.mesh, probe_layout(field), points)
    for (name, _), tet in zip(sources, located.tetrahedra, strict=True):
        if layer[tet]:
            raise ValueError(
                f"The element layer just outside this side holds the dipole {name!r} "
                "whose load solved the field: its load would count as power "
                "crossing the surface. Move the side's surface "
                "at least one element away from the dipole, or put the dipole "
                "inside the side."
            )


def build(
    order: int,
    model: Model,
    where: Surface,
    *,
    stretch: Stretch | None = None,
    comm: Any = None,
) -> Flux:
    """The `Flux` of a surface on ``model`` at its node positions.

    ``stretch`` is the field's perfectly matched layer stretch
    (`gfem._pml.field_stretch`); ``comm`` the communicator of a field across
    ranks (collective).
    """
    op = driven_operator(
        model,
        order,
        elements=where.layer,
        faces=where.faces,
        paired=False,
        comm=comm,
        stretch=stretch,
    )
    mask = dofs(op, where.tie, where.rim, where.rim | where.layer)
    return Flux(op=op, mask=mask, layout=op.layout or ONE_PROCESS)


def node_pullback(
    flux: Flux,
    model: Model,
    where: Surface,
    e: np.ndarray,
    omega: float,
    *,
    comm: Any = None,
) -> np.ndarray:
    r"""The power's gradient in the node coordinates at fixed coefficients.

    :math:`P = \operatorname{Im}(\overline{De}^\mathsf{T} M e) / c_0` is a sum
    of the layer's pieces on the pair :math:`(\overline{De}, e)`, so it goes
    through the sensitivity kernel's node pass (`gfem._engine.sensitivity`)
    with weights :math:`-i\theta_k / c_0` (the pass takes real parts): the
    transport of the field and of the element-wise coefficients, a
    perfectly matched layer's stretch read at the moved quadrature points,
    and the face terms, each restricted to the layer's elements and faces
    as the flux's pieces are. The mask is topology and the field moves with
    the nodes at fixed coefficients.

    Returns:
        ``(n_nodes, 3)``, real, per metre; under ranks summed over them, the
        same on every rank.
    """
    from gfem._engine.domain import em_region
    from gfem._engine.driven import operator_terms
    from gfem._engine.sensitivity import _node_pass
    from gfem._engine.terms import _bits

    op = flux.op
    built = op.built
    part = built.comm is not None
    bits = {
        "volume": _bits(where.layer, built.cell_order, "tetrahedron", part=part),
        "surface": _bits(where.faces, built.facet_order, "triangle", part=part),
    }
    terms = [
        dataclasses.replace(
            t,
            elements=bits["surface" if t.kind in ("trace", "curl trace") else "volume"],
        )
        for t in operator_terms(model, op, nodes=False)
    ]
    scale = flux._scale(omega)  # noqa: SLF001
    weights = {
        p.name: -1j * complex(p.theta(omega)) / scale
        for p in op.pieces
        if p.kind != "port"
    }
    w = np.where(flux.mask, e, 0.0)
    pair = op.pair(np.conj(w), e, weights=weights, domain=em_region(model, built))
    out = _node_pass(model, [pair], terms, comm)
    if comm is not None:
        from gfem import _mpi

        out = np.asarray(_mpi.reduce_bcast(comm, out))
    return out


def check_size(e: np.ndarray, op: DrivenOperator, order: int) -> None:
    """Raise unless the coefficients fit the driven operator's space.

    Raises:
        ValueError: Another number of DOFs.
    """
    if e.shape[0] != op.space.ndof:
        raise ValueError(
            f"The field has {e.shape[0]} coefficients but the model's order-"
            f"{order} driven space has {op.space.ndof} DOFs: the field "
            "was not solved on this model."
        )


def check_nodes(field: Field, model: Model) -> None:
    """Raise unless the field's node positions are the model's.

    Raises:
        ValueError: They differ.
    """
    if not np.array_equal(np.asarray(field.mesh.nodes), np.asarray(model.mesh.nodes)):
        raise _other_mesh()


def _other_mesh() -> ValueError:
    return ValueError(
        "The field lives on another mesh (or other node positions) than the "
        "model: pass the model the field was solved with."
    )


def check_field(field: Field, model: Model) -> np.ndarray:
    """The field's support mask, after checking it was solved on ``model``.

    Raises:
        ValueError: The mesh, its nodes or the support differ.
    """
    check_nodes(field, model)
    return check_support(field, model)


def check_support(field: Field, model: Model) -> np.ndarray:
    """The field's support mask, after checking its topology is the model's.

    Raises:
        ValueError: The mesh topology or the support differ.
    """
    mesh = model.mesh
    if not _same_topology(field.mesh, mesh):
        raise _other_mesh()
    excluded = tuple(field._excluded)  # noqa: SLF001
    layout = probe.Layout(tuple(field.support), excluded, field.space, field.order)
    support = probe.in_support(mesh, layout)
    if not np.array_equal(support, model._domain.active):  # noqa: SLF001
        raise ValueError(
            "The field's support is not the model's solve domain (other regions "
            "or exclusions): pass the model the field was solved with."
        )
    return support


def edge_keys(tie: np.ndarray, cells: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """One integer per edge of each cell, the same for the same edge.

    Nodes are read as their tie class representatives (`tie_classes`), so
    an edge and its images across the periodic pairs share a key.
    """
    ends = np.sort(tie[cells[:, pairs]], axis=-1)
    return ends[..., 0].astype(np.int64) * len(tie) + ends[..., 1]


def _outside_faces(
    mesh: Mesh,
    tie: np.ndarray,
    inside: np.ndarray,
    support: np.ndarray,
    closure: np.ndarray,
) -> np.ndarray:
    """Triangles whose terms belong to the outside and reach A's DOFs.

    A face term belongs to A only when every solve-domain element beside
    the face is in A; the outside's reach A when the face shares an edge
    with A.
    """
    nb = mesh.triangle_neighbours
    tet = np.maximum(nb, 0)
    beside = (nb >= 0) & support[tet]
    owned = beside.any(axis=1) & ~(~beside | inside[tet]).all(axis=1)
    touching = np.isin(edge_keys(tie, mesh.triangles, TRI_EDGES), closure)
    return owned & touching.any(axis=1)


def dofs(
    op: DrivenOperator, tie: np.ndarray, elements: np.ndarray, near: np.ndarray
) -> np.ndarray:
    """The DOFs of the cells, edges and faces of the given gfem tetrahedra.

    With the images of those edges and faces across the periodic pairs
    (same `tie_classes` key), looked up on the ``near`` tetrahedra, which
    hold every image the operator reaches. Without pairs the images are
    the edges and faces themselves.

    Under ranks each rank marks the DOFs of the tetrahedra it holds, and a
    DOF shared with a rank that holds such a tetrahedron is marked too (one
    neighbour exchange, collective): the holders of a DOF agree.
    """
    space, ngmesh = op.space, op.built.mesh
    node_order = op.built.node_order
    ngs_index = inverse_order(op.built.cell_order, len(elements))
    out = np.zeros(space.ndof, bool)

    def mark(node: ngs.NodeId) -> None:
        dofs = np.asarray(space.GetDofNrs(node), dtype=np.int64)
        out[dofs[dofs >= 0]] = True

    def key(node: ngs.NodeId) -> tuple[int, ...]:
        return tuple(sorted(int(tie[node_order[v.nr]]) for v in ngmesh[node].vertices))

    keys = set()

    def held(mask: np.ndarray) -> np.ndarray:
        """The NGSolve numbers of the masked tetrahedra this rank holds."""
        local = ngs_index[np.flatnonzero(mask)]
        return local[local >= 0]

    for t in held(elements):
        element = ngmesh[ngs.ElementId(ngs.VOL, int(t))]
        keys.update(key(n) for n in (*element.edges, *element.faces))
        mark(ngs.NodeId(ngs.CELL, int(t)))
    for t in held(near):
        element = ngmesh[ngs.ElementId(ngs.VOL, int(t))]
        for n in (*element.edges, *element.faces):
            if key(n) in keys:
                mark(n)
    return out if op.layout is None else op.layout.held_by_any(out)
