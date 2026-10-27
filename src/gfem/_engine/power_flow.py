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
port face that shares an edge with A raises.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import ngsolve as ngs
import numpy as np
import scipy.constants as const

from gfem._boundaries import surfaces_of
from gfem._engine import probe
from gfem._engine.driven import DrivenOperator, driven_operator
from gfem.mesh import _TET_EDGES, Mesh

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model

__all__ = ["power_flow"]

#: The three edges of a triangle, by vertex.
_TRI_EDGES = np.array([[0, 1], [0, 2], [1, 2]])
#: The `Mesh` arrays that, with its regions, define its elements.
_TOPOLOGY_ARRAYS = (
    "tetrahedra",
    "tetrahedron_tags",
    "tetrahedron_entities",
    "triangles",
    "triangle_tags",
    "triangle_entities",
)


def power_flow(
    field: Field,
    side: tuple[str, ...] | Callable[[np.ndarray], np.ndarray],
    model: Model,
) -> float:
    """Net time-averaged power leaving ``side``, W (see `gfem.power_flow`).

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: The side, the field or a port near the surface does not
            fit (see `gfem.power_flow`).
    """
    mesh = model.mesh
    support = _check_field(field, model)
    inside = _selection(mesh, side) & support
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
    tet_edges = _edge_keys(mesh, mesh.tetrahedra, np.array(_TET_EDGES))
    closure = np.unique(tet_edges[inside])
    layer = outside & np.isin(tet_edges, closure).any(axis=1)
    rim = inside & np.isin(tet_edges, np.unique(tet_edges[outside])).any(axis=1)
    faces = _outside_faces(mesh, inside, support, closure)
    for port in model.ports:
        for surface in surfaces_of(port):
            if faces[mesh.elements(surface)].any():
                raise ValueError(
                    f"The surface of this side is within one element of port "
                    f"{port.name!r}'s surface {surface!r}: power_flow cannot "
                    "count a port's excitation. Move the surface at least one "
                    "element away from the port, or put the port's elements "
                    "inside the side."
                )

    op = driven_operator(model, field.order, elements=layer, faces=faces)
    e = np.asarray(field.coefficients)
    if e.shape[0] != op.space.ndof:
        raise ValueError(
            f"The field has {e.shape[0]} coefficients but the model's order-"
            f"{field.order} driven space has {op.space.ndof} DOFs: the field "
            "was not solved on this model."
        )
    w = np.where(_dofs(op, rim), e, 0.0)
    omega = float(np.asarray(field.omega).real)
    total = sum(
        piece.theta(omega) * np.vdot(w, op.apply(k, e))
        for k, piece in enumerate(op.pieces)
        if piece.kind != "port"
    )
    return float(np.imag(total)) / (2 * omega * const.mu_0 * op.built.scale)


def _check_field(field: Field, model: Model) -> np.ndarray:
    """The field's support mask, after checking it was solved on ``model``.

    Raises:
        ValueError: The mesh, its nodes or the support differ.
    """
    mesh = model.mesh
    same = _same_topology(field.mesh, mesh) and np.array_equal(
        np.asarray(field.mesh.nodes), np.asarray(mesh.nodes)
    )
    if not same:
        raise ValueError(
            "The field lives on another mesh (or other node positions) than the "
            "model: pass the model the field was solved with."
        )
    excluded = tuple(field._excluded)  # noqa: SLF001
    layout = probe.Layout(tuple(field.support), excluded, field.space, field.order)
    support = probe.in_support(mesh, layout)
    if not np.array_equal(support, model._domain.active):  # noqa: SLF001
        raise ValueError(
            "The field's support is not the model's solve domain (other regions "
            "or exclusions): pass the model the field was solved with."
        )
    return support


def _same_topology(a: Mesh, b: Mesh) -> bool:
    """Whether two meshes have the same elements, tags and regions.

    The same object, or equal arrays: the same file read twice (a loaded
    result's mesh and the model's) gives two objects.
    """
    if a._topology is b._topology:  # noqa: SLF001
        return True
    return a.regions == b.regions and all(
        np.array_equal(getattr(a, name), getattr(b, name)) for name in _TOPOLOGY_ARRAYS
    )


def _selection(
    mesh: Mesh, side: tuple[str, ...] | Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """The selected tetrahedra: named volume regions or a centroid predicate.

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: A name is a surface, or the predicate does not return
            one boolean per tetrahedron.
    """
    n = len(mesh.tetrahedra)
    if callable(side):
        centroids = np.asarray(mesh.nodes)[mesh.tetrahedra].mean(axis=1)
        out = np.asarray(side(centroids))
        if out.dtype != bool or out.shape != (n,):
            raise ValueError(
                f"The side predicate must return one boolean per tetrahedron "
                f"({n},) from the (n_tetrahedra, 3) element centroids, got an "
                f"array of {out.dtype} of shape {out.shape}."
            )
        return out
    out = np.zeros(n, bool)
    for name in side:
        if mesh.region(name).dim != 3:
            raise ValueError(
                f"{name!r} is a surface region; power_flow's side is a set of "
                "elements: volume regions, or a predicate on element centroids."
            )
        out[mesh.elements(name)] = True
    return out


def _edge_keys(mesh: Mesh, cells: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """One integer per edge of each cell, the same for the same edge."""
    ends = np.sort(cells[:, pairs], axis=-1)
    return ends[..., 0].astype(np.int64) * len(mesh.nodes) + ends[..., 1]


def _outside_faces(
    mesh: Mesh, inside: np.ndarray, support: np.ndarray, closure: np.ndarray
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
    touching = np.isin(_edge_keys(mesh, mesh.triangles, _TRI_EDGES), closure)
    return owned & touching.any(axis=1)


def _dofs(op: DrivenOperator, elements: np.ndarray) -> np.ndarray:
    """The DOFs of the edges, faces and cells of the given gfem tetrahedra."""
    space, built = op.space, op.built
    ngs_index = np.argsort(built.tetrahedron_order)
    out = np.zeros(space.ndof, bool)
    for t in ngs_index[np.flatnonzero(elements)]:
        dofs = np.asarray(space.GetDofNrs(ngs.ElementId(ngs.VOL, int(t))))
        out[dofs[dofs >= 0]] = True
    return out
