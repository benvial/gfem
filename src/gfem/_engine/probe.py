"""Point evaluation of fields, the host kernel behind measurements.

`Field.at` and the surface measurements evaluate fields at points.

A field's value at a point of element ``e`` is a linear map of the field's
coefficients, ``Phi c``, with ``Phi`` the element's (Piola-mapped) shape
functions at the point. `Probe` evaluates ``Phi c`` for any number of
coefficient vectors and its transpose ``Phi^T g`` (the backward pass) without
forming ``Phi``: in the discontinuous copy of the field's space every
element owns its DOFs, so setting local DOF ``k`` of every element to one and
evaluating at all points gives the ``k``-th shape function at each point in
one pass.

Points are given as (NGSolve volume element, reference coordinates), the
reference coordinates being the barycentric coordinates of the point with
respect to the element's first three vertices (gfem's tetrahedron vertex
order, which the engine mesh keeps). Points on a face shared by two
elements are therefore evaluated from the element they are given with: a
one-sided value of a field that jumps across the face.

Under ranks (ADR 0063) a tetrahedron lies on one rank, which holds the points in it:
``Phi c`` is the rank sum of what each rank evaluates, and ``Phi^T g`` is a
``DISTRIBUTED`` load, each point counted once. A point's tetrahedron is settled from the
replicated mesh, so it does not depend on the partition. Coefficients are a rank's piece
at the seam's width (`gfem._engine.fields.dof_shape`).
"""

from __future__ import annotations

import collections
import dataclasses
import threading
from typing import Any

import ngsolve as ngs
import numpy as np

from gfem import _mpi
from gfem._engine.fields import LENGTH_POWER, dof_shape, fe_space
from gfem._engine.mesh import EngineMesh, inverse_order
from gfem.mesh import Mesh

__all__ = [
    "Layout",
    "Located",
    "Probe",
    "face_points",
    "in_support",
    "locate",
    "probe",
    "triangle_rule",
]


@dataclasses.dataclass(frozen=True, eq=False)
class Located:
    """Points as (gfem tetrahedron, barycentric coordinates).

    Attributes:
        tetrahedra: ``(n,)`` gfem tetrahedron index of each point.
        barycentric: ``(n, 4)`` barycentric coordinates with respect to the
            tetrahedron's vertices, in gfem order.
    """

    tetrahedra: np.ndarray
    barycentric: np.ndarray


@dataclasses.dataclass(frozen=True)
class Layout:
    """What a field's coefficients mean (see `gfem.Field`)."""

    support: tuple[str, ...]
    excluded: tuple[tuple[int, int], ...]
    space: str
    order: int


#: Under this smallest barycentric coordinate a point counts as on a face.
ON_FACE = 1e-12


def in_support(mesh: Mesh, layout: Layout) -> np.ndarray:
    """Per gfem tetrahedron: whether the field is defined on it."""
    names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    wanted = {t for t, n in names.items() if n in set(layout.support)}
    inside = np.isin(mesh.tetrahedron_tags, list(wanted))
    if layout.excluded:
        pairs = np.stack([mesh.tetrahedron_tags, mesh.tetrahedron_entities], axis=1)
        out = (pairs[:, None, :] == np.asarray(layout.excluded)[None]).all(-1).any(-1)
        inside &= ~out
    return inside


def locate(mesh: Mesh, layout: Layout, points: np.ndarray, comm: Any = None) -> Located:
    """The tetrahedra holding ``points`` (``(n, 3)``, m).

    A point inside a tetrahedron gets that one, on any number of ranks. On a face, to
    round-off, it gets one holding it: in the support when one is, then the largest
    index. Netgen's search also accepts a point within 1e-4 outside a tetrahedron, which
    stays only for a point just outside the mesh: it gets the nearest tetrahedron.

    Under ``comm`` it is collective, and all ranks get the same location.

    Raises:
        ValueError: A point lies outside the mesh, or in a tetrahedron where
            the field is not defined (a volume outside its support, or one
            removed from the solve by the model's ``excluded=``). Under ``comm``
            on every rank.
    """
    from gfem._engine.cache import discretization

    built = discretization(mesh, (), comm).built
    p = np.asarray(points, float).reshape(-1, 3) / built.scale
    found = built.mesh(p[:, 0].copy(), p[:, 1].copy(), p[:, 2].copy())
    nr = np.asarray(found["nr"], dtype=np.int64)
    held = nr >= 0
    tets = np.full(len(nr), -1, dtype=np.int64)
    tets[held] = np.asarray(built.tetrahedron_order)[nr[held]]
    tets = _agree_location(comm, tets)
    outside = tets < 0
    if outside.any():
        worst = points.reshape(-1, 3)[int(np.flatnonzero(outside)[0])]
        raise _mpi.agreed(
            ValueError(
                f"Point {_fmt(worst)} m is outside the mesh: a field has no value "
                "there."
            )
        )
    support = in_support(mesh, layout)
    tets, bary = _settle(mesh, support, tets, np.asarray(points, float).reshape(-1, 3))
    inside = support[tets]
    if not inside.all():
        k = int(np.flatnonzero(~inside)[0])
        names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
        region = names[int(mesh.tetrahedron_tags[tets[k]])]
        raise _mpi.agreed(
            ValueError(
                f"Point {_fmt(points.reshape(-1, 3)[k])} m lies in volume region "
                f"{region!r}, where the field is not defined (its support is "
                f"{list(layout.support)}, minus the volumes the model excludes from "
                "the solve)."
            )
        )
    return Located(tets, bary)


def _agree_location(comm: Any, tets: np.ndarray) -> np.ndarray:
    """One location on all ranks: the largest gfem tetrahedron any rank found.

    Any rule held by all ranks serves. ``-1`` is a point no rank found.
    """
    return tets if comm is None else _mpi.max_bcast(comm, tets)


def _barycentric(
    nodes: np.ndarray, tetrahedra: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """``(n, 4)`` barycentric coordinates of ``points`` in ``(n, 4)`` ``tetrahedra``."""
    v = nodes[tetrahedra]
    edges = np.stack([v[:, i] - v[:, 3] for i in range(3)], axis=-1)
    head = np.linalg.solve(edges, (points - v[:, 3])[..., None])[..., 0]
    return np.concatenate([head, 1.0 - head.sum(axis=1, keepdims=True)], axis=1)


def _settle(
    mesh: Mesh, support: np.ndarray, tets: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move each point out of a tetrahedron netgen accepted but that does not hold it.

    Reads only the mesh every rank holds (ADR 0059), so the answer does not depend on
    the partition. Every tetrahedron holding a point shares a node with any other, and
    so with one netgen found holding it.

    Returns:
        The settled ``(n,)`` tetrahedra and their ``(n, 4)`` barycentric coordinates.
    """
    nodes = np.asarray(mesh.nodes, float)
    all_tets = np.asarray(mesh.tetrahedra)
    tets = tets.copy()
    bary = _barycentric(nodes, all_tets[tets], points)
    flagged = np.flatnonzero(bary.min(axis=1) < ON_FACE)
    if not flagged.size:
        return tets, bary
    of_node = np.argsort(all_tets.ravel(), kind="stable") // 4
    start = np.concatenate(([0], np.cumsum(np.bincount(all_tets.ravel()))))
    for i in flagged:
        around = np.unique(
            np.concatenate(
                [of_node[start[n] : start[n + 1]] for n in all_tets[tets[i]]]
            )
        )
        lowest = _barycentric(
            nodes, all_tets[around], np.repeat(points[i : i + 1], len(around), axis=0)
        ).min(axis=1)
        holders = around[lowest >= min(lowest.max(), 0.0) - ON_FACE]
        supported = holders[support[holders]]
        tets[i] = (supported if supported.size else holders).max()
    return tets, _barycentric(nodes, all_tets[tets], points)


def face_points(
    mesh: Mesh, triangles: np.ndarray, tetrahedra: np.ndarray, rule: np.ndarray
) -> Located:
    """Quadrature points on ``triangles``, seen from ``tetrahedra``.

    Args:
        mesh: The mesh.
        triangles: ``(n,)`` gfem triangle indices.
        tetrahedra: ``(n,)`` the tetrahedron on the side to evaluate, one of
            the triangle's neighbours.
        rule: ``(q, 3)`` barycentric points on the triangle (its vertices in
            gfem order).

    Returns:
        ``n * q`` points, triangle-major.
    """
    tris = mesh.triangles[triangles]  # (n, 3) node ids
    tets = mesh.tetrahedra[tetrahedra]  # (n, 4)
    # position of each triangle vertex in its tetrahedron's vertex list
    where = (tris[:, :, None] == tets[:, None, :]).argmax(axis=2)  # (n, 3)
    n, q = len(tris), len(rule)
    bary = np.zeros((n, q, 4))
    rows, cols = np.arange(n)[:, None], np.arange(q)[None, :]
    for j in range(3):
        bary[rows, cols, where[:, j : j + 1]] = rule[None, :, j]
    return Located(np.repeat(tetrahedra, len(rule)), bary.reshape(-1, 4))


def triangle_rule(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """NGSolve's triangle rule exact to ``degree``.

    Returns:
        Barycentric points ``(q, 3)`` and weights ``(q,)`` summing to 1
        (multiply by the triangle's area).
    """
    rule = ngs.IntegrationRule(ngs.TRIG, degree)
    xi = np.array([p.point[:2] for p in rule])
    bary = np.column_stack([xi, 1.0 - xi.sum(axis=1)])
    weights = 2.0 * np.array([p.weight for p in rule])
    return bary, weights


def _fmt(p: np.ndarray) -> str:
    return "(" + ", ".join(f"{x:.6g}" for x in p) + ")"


class Probe:
    """``Phi`` of one field layout at fixed points of one mesh state."""

    def __init__(
        self, mesh: Mesh, layout: Layout, located: Located, comm: Any = None
    ) -> None:
        """Build the engine space and the points (the mesh's node positions).

        Under ``comm`` it is collective, also on a rank that holds none of the points.
        """
        self.layout = layout
        self.comm = comm
        built, fes, _ = fe_space(
            mesh, layout.support, layout.excluded, layout.space, layout.order, comm=comm
        )
        self._built: EngineMesh = built
        self._fes = fes
        self.factor = built.scale ** LENGTH_POWER[layout.space]
        inverse = inverse_order(built.tetrahedron_order, len(mesh.tetrahedra))
        self._elements = inverse[located.tetrahedra]
        self.n_points = len(self._elements)
        self._own = np.flatnonzero(self._elements >= 0)
        self._mips = None
        if self._own.size:
            mips = built.mesh(np.zeros(1), np.zeros(1), np.zeros(1))
            mips = np.repeat(mips, self._own.size)
            mips["nr"] = self._elements[self._own]
            mips["VorB"] = 0
            bary = located.barycentric[self._own]
            mips["x"], mips["y"], mips["z"] = bary[:, 0], bary[:, 1], bary[:, 2]
            self._mips = mips
        self._local: tuple | None = None
        # a collective call, so no rank may leave it for its first point
        self._discontinuous = None if comm is None else ngs.Discontinuous(fes)
        self.width = int(fes.ndof)
        if comm is not None:
            self.width = dof_shape(
                mesh,
                layout.support,
                layout.excluded,
                layout.space,
                layout.order,
                comm=comm,
            )[0]
        self._lock = threading.Lock()

    @property
    def ndof(self) -> int:
        """The number of field coefficients this rank holds."""
        return int(self._fes.ndof)

    def _evaluate(self, fes: ngs.FESpace, values: np.ndarray) -> np.ndarray:
        """Values at this rank's points, ``(its points, dim)``."""
        gf = ngs.GridFunction(fes)
        if self._mips is None:
            return np.zeros((0, gf.dim))
        gf.vec.FV().NumPy()[:] = values
        out = np.asarray(gf(self._mips))
        return out.reshape(self._own.size, -1)

    def values(self, coefficients: np.ndarray) -> np.ndarray:
        """``Phi c`` in SI units, ``(m, n_points, dim)`` for ``(m, width)``.

        Under ranks it is collective, with the same bits on every rank.
        """
        c = np.atleast_2d(coefficients)
        if c.shape[1] != self.width:
            raise _mpi.agreed(
                ValueError(
                    f"The field has {c.shape[1]} coefficients but its "
                    f"{self.layout.space} order-{self.layout.order} space has "
                    f"{self.width} DOFs: the field does not belong to this mesh."
                )
            )
        c = c[:, : self.ndof]
        parts = [np.real(c)] + ([np.imag(c)] if np.iscomplexobj(c) else [])
        with self._lock:
            out = [
                np.stack([self._evaluate(self._fes, row) for row in part])
                for part in parts
            ]
        values = out[0] if len(out) == 1 else out[0] + 1j * out[1]
        values = values * self.factor
        if self.comm is None:
            return values
        whole = np.zeros((c.shape[0], self.n_points, values.shape[2]), values.dtype)
        whole[:, self._own] = values
        return _mpi.reduce_bcast(self.comm, whole)

    def _local_dofs(self) -> tuple:
        if self._local is None:
            fes = self._fes
            dfes = self._discontinuous
            if dfes is None:
                dfes = ngs.Discontinuous(fes)
            unique, position = np.unique(self._elements[self._own], return_inverse=True)
            cont = np.array(
                [list(fes.GetDofNrs(ngs.ElementId(ngs.VOL, int(e)))) for e in unique],
                dtype=np.int64,
            )
            disc = np.array(
                [list(dfes.GetDofNrs(ngs.ElementId(ngs.VOL, int(e)))) for e in unique],
                dtype=np.int64,
            )
            self._local = (dfes, cont[position.ravel()], disc)
        return self._local

    def transpose(self, cotangent: np.ndarray) -> np.ndarray:
        """``Phi^T g`` (SI scaling included), ``(m, width)`` for ``(m, n, dim)``.

        Under ranks a ``DISTRIBUTED`` load, and no rank communicates: ``g`` is
        replicated, and each rank writes only for the points it holds.
        """
        g = np.asarray(cotangent).reshape(-1, self.n_points, cotangent.shape[-1])
        out = np.zeros((len(g), self.width), dtype=g.dtype)
        if self._own.size == 0:
            return out
        g = g[:, self._own]
        with self._lock:
            dfes, cont, disc = self._local_dofs()
            for k in range(cont.shape[1]):
                unit = np.zeros(dfes.ndof)
                unit[disc[:, k]] = 1.0
                shape = self._evaluate(dfes, unit) * self.factor  # (n, dim)
                contribution = np.einsum("pd,mpd->mp", shape, g)
                for j in range(len(g)):
                    out[j, : self.ndof] += np.bincount(
                        cont[:, k], weights=contribution[j].real, minlength=self.ndof
                    )
                    if np.iscomplexobj(g):
                        out[j, : self.ndof] += 1j * np.bincount(
                            cont[:, k],
                            weights=contribution[j].imag,
                            minlength=self.ndof,
                        )
        return out


_PROBES: collections.OrderedDict[tuple, tuple] = collections.OrderedDict()
_MAX_PROBES = 4


def probe(mesh: Mesh, layout: Layout, located: Located, comm: Any = None) -> Probe:
    """The `Probe` of a layout at points, cached for the backward pass.

    Under ``comm`` a miss is collective, and all ranks miss together.
    """
    key = (
        id(mesh._topology),  # noqa: SLF001
        _mpi.comm_key(comm),
        _digest(np.asarray(mesh.nodes)),
        layout,
        _digest(located.tetrahedra),
        _digest(located.barycentric),
    )
    found = _PROBES.get(key)
    if found is not None and found[0] is mesh._topology:  # noqa: SLF001
        _PROBES.move_to_end(key)
        return found[1]
    made = Probe(mesh, layout, located, comm)
    _PROBES[key] = (mesh._topology, made)  # noqa: SLF001
    while len(_PROBES) > _MAX_PROBES:
        _PROBES.popitem(last=False)
    return made


def _digest(a: np.ndarray) -> int:
    a = np.ascontiguousarray(a)
    return hash((a.shape, a.dtype.str, a.tobytes()))
