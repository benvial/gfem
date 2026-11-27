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
respect to the element's first ``dim`` vertices (gfem's vertex order of a
tetrahedron, or of a 2D mesh's triangle, which the engine mesh keeps).
Points on a face shared by two elements are therefore evaluated from the
element they are given with: a one-sided value of a field that jumps across
the face.

Under ranks a tetrahedron lies on one rank, which holds the points in it:
``Phi c`` is the rank sum of what each rank evaluates, and ``Phi^T g`` is a
``DISTRIBUTED`` load, each point counted once. A point's tetrahedron is settled from the
replicated mesh, so it does not depend on the partition. Coefficients are a rank's piece
at the seam's width (`gfem._engine.fields.dof_shape`).
"""

from __future__ import annotations

import collections
import dataclasses
import threading
import weakref
from typing import Any

import ngsolve as ngs
import numpy as np

from gfem import _curved, _mpi
from gfem._engine.fields import LENGTH_POWER, dof_shape, fe_space
from gfem._engine.mesh import EngineMesh, inverse_order
from gfem.mesh import Mesh

__all__ = [
    "GradientProbe",
    "Layout",
    "Located",
    "Probe",
    "covered",
    "face_points",
    "in_support",
    "locate",
    "probe",
    "triangle_rule",
]


@dataclasses.dataclass(frozen=True, eq=False)
class Located:
    """Points as (gfem volume element, barycentric coordinates).

    Attributes:
        tetrahedra: ``(n,)`` gfem volume element (`gfem.Mesh.volume_elements`
            row) of each point: a tetrahedron, or a 2D mesh's triangle.
        barycentric: ``(n, dim + 1)`` barycentric coordinates with respect to
            the element's vertices, in gfem order.
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


#: Under this smallest barycentric coordinate a point counts as on a face: round-off
#: on a sliver tetrahedron reaches 1e-11.
ON_FACE = 1e-9

#: An element whose volume (area) is below this share of its longest edge to the
#: power ``dim`` is flat.
_FLAT = 1e-13


def in_support(mesh: Mesh, layout: Layout) -> np.ndarray:
    """Per gfem volume element: whether the field is defined on it."""
    topology = mesh._topology  # noqa: SLF001
    names = mesh._region_names()  # noqa: SLF001
    wanted = {t for t, n in names.items() if n in set(layout.support)}
    inside = np.isin(topology.volume_tags, list(wanted))
    if layout.excluded:
        pairs = np.stack([topology.volume_tags, topology.volume_entities], axis=1)
        out = (pairs[:, None, :] == np.asarray(layout.excluded)[None]).all(-1).any(-1)
        inside &= ~out
    return inside


def locate(mesh: Mesh, layout: Layout, points: np.ndarray, comm: Any = None) -> Located:
    """The volume elements holding ``points`` (``(n, dim)``, m).

    On a 2D mesh the elements are triangles and the points ``(n, 2)``; what
    follows says tetrahedron for either.

    A point inside a tetrahedron gets that one, on any number of ranks. On a face, to
    1e-9 barycentric (`ON_FACE`), it gets one holding it to that tolerance: in the
    support when one is, then the largest index. A tetrahedron of zero volume (to
    round-off) holds no point. Netgen's search also accepts a point within 1e-4
    outside a tetrahedron, which stays only for a point just outside the mesh: it gets
    the nearest tetrahedron.

    Under ``comm`` it is collective, and all ranks get the same location.

    Raises:
        ValueError: A point lies outside the mesh, or in a tetrahedron where
            the field is not defined (a volume outside its support, or one
            removed from the solve by the model's ``excluded=``). Under ``comm``
            on every rank.
    """
    from gfem._engine.cache import discretization

    built = discretization(mesh, (), comm).built
    points = np.asarray(points, float).reshape(-1, mesh.dim)
    p = points / built.scale
    z = p[:, 2].copy() if mesh.dim == 3 else np.zeros(len(p))
    found = built.mesh(p[:, 0].copy(), p[:, 1].copy(), z)
    nr = np.asarray(found["nr"], dtype=np.int64)
    held = nr >= 0
    tets = np.full(len(nr), -1, dtype=np.int64)
    tets[held] = np.asarray(built.cell_order)[nr[held]]
    tets = _agree_location(comm, tets)
    outside = tets < 0
    if outside.any():
        worst = points[int(np.flatnonzero(outside)[0])]
        raise _mpi.agreed(
            ValueError(
                f"Point {_fmt(worst)} m is outside the mesh: a field has no value "
                "there."
            )
        )
    support = in_support(mesh, layout)
    tets, bary = _settle(mesh, support, tets, points)
    inside = support[tets]
    if not inside.all():
        k = int(np.flatnonzero(~inside)[0])
        raise _mpi.agreed(outside_support(mesh, layout, int(tets[k]), points[k]))
    if mesh.curved:
        # `_settle` and the barycentric coordinates invert the affine map: a
        # point in a curved element would read the field off by up to the sagitta
        bent = _curved.curved_tetrahedra(mesh, _curved.moved_nodes(mesh))[tets]
        if bent.any():
            at = points[int(np.flatnonzero(bent)[0])]
            _curved.refuse_tets(
                mesh, "point_read", tets, f"A point read at {_fmt(at)} m"
            )
    return Located(tets, bary)


def covered(mesh: Mesh, layout: Layout, points: np.ndarray) -> np.ndarray:
    """Which of ``points`` (``(n, dim)``, m) a field of ``layout`` is defined at.

    The test `locate` applies, answered instead of raised: a point is covered
    when the mesh holds it (to the tolerance of `locate`, so a point on the
    mesh's boundary counts) and the tetrahedron settled for it is in the
    support. Not collective: for a mesh every rank holds.
    """
    from gfem._engine.cache import discretization

    built = discretization(mesh, (), None).built
    points = np.asarray(points, float).reshape(-1, mesh.dim)
    p = points / built.scale
    z = p[:, 2].copy() if mesh.dim == 3 else np.zeros(len(p))
    nr = np.asarray(built.mesh(p[:, 0].copy(), p[:, 1].copy(), z)["nr"], np.int64)
    held = nr >= 0
    out = np.zeros(len(p), bool)
    if held.any():
        tets = np.asarray(built.cell_order)[nr[held]]
        support = in_support(mesh, layout)
        settled, _ = _settle(mesh, support, tets, points[held])
        out[held] = support[settled]
    return out


def outside_support(
    mesh: Mesh, layout: Layout, tet: int, point: np.ndarray
) -> ValueError:
    """The error for ``point``, in tetrahedron ``tet`` outside the support."""
    names = mesh._region_names()  # noqa: SLF001
    region = names[int(mesh._topology.volume_tags[tet])]  # noqa: SLF001
    return ValueError(
        f"Point {_fmt(point)} m lies in volume region "
        f"{region!r}, where the field is not defined (its support is "
        f"{list(layout.support)}, minus the volumes the model excludes from "
        "the solve)."
    )


def _agree_location(comm: Any, tets: np.ndarray) -> np.ndarray:
    """One location on all ranks: the largest gfem tetrahedron any rank found.

    Any rule held by all ranks serves. ``-1`` is a point no rank found.
    """
    return tets if comm is None else _mpi.max_bcast(comm, tets)


def _barycentric(
    nodes: np.ndarray, tetrahedra: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """``(n, d + 1)`` barycentric coordinates of ``(n, d)`` ``points`` in elements.

    ``tetrahedra`` are ``(n, d + 1)`` node indices: tetrahedra, or a 2D mesh's
    triangles. A flat element (zero volume to round-off) gets ``-inf``: it holds
    no point.
    """
    v = nodes[tetrahedra]
    d = v.shape[-1]
    edges = np.stack([v[:, i] - v[:, d] for i in range(d)], axis=-1)
    longest = np.linalg.norm(edges, axis=1).max(axis=1)
    flat = np.abs(np.linalg.det(edges)) <= _FLAT * longest**d
    bary = np.full((len(v), d + 1), -np.inf)
    if (~flat).any():
        rhs = (points - v[:, d])[~flat][..., None]
        head = np.linalg.solve(edges[~flat], rhs)[..., 0]
        bary[~flat] = np.concatenate(
            [head, 1.0 - head.sum(axis=1, keepdims=True)], axis=1
        )
    return bary


_NODE_TETS: weakref.WeakKeyDictionary[Any, tuple[np.ndarray, np.ndarray]] = (
    weakref.WeakKeyDictionary()
)


def _tets_of_nodes(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """The volume elements around each node, CSR: ``tets[start[n] : start[n + 1]]``.

    Built once per mesh topology.
    """
    topology = mesh._topology  # noqa: SLF001
    found = _NODE_TETS.get(topology)
    if found is None:
        elements = np.asarray(topology.volume_elements)
        flat = elements.ravel()
        tets = np.argsort(flat, kind="stable") // elements.shape[1]
        start = np.concatenate(([0], np.cumsum(np.bincount(flat))))
        found = _NODE_TETS[topology] = (tets, start)
    return found


def _settle(
    mesh: Mesh, support: np.ndarray, tets: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move each point out of a tetrahedron netgen accepted but that does not hold it.

    Reads only the mesh every rank holds, so the answer does not depend on
    the partition. Every tetrahedron holding a point shares a node with any other, and
    so with one netgen found holding it.

    Returns:
        The settled ``(n,)`` tetrahedra and their ``(n, 4)`` barycentric coordinates.
    """
    nodes = np.asarray(mesh.nodes, float)
    all_tets = np.asarray(mesh._topology.volume_elements)  # noqa: SLF001
    tets = tets.copy()
    bary = _barycentric(nodes, all_tets[tets], points)
    flagged = np.flatnonzero(bary.min(axis=1) < ON_FACE)
    if not flagged.size:
        return tets, bary
    of_node, start = _tets_of_nodes(mesh)
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


def tetrahedron_rule(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """NGSolve's tetrahedron rule exact to ``degree``.

    Returns:
        Barycentric points ``(q, 4)`` and weights ``(q,)`` summing to 1
        (multiply by the tetrahedron's volume).
    """
    rule = ngs.IntegrationRule(ngs.TET, degree)
    xi = np.array([p.point[:3] for p in rule])
    bary = np.column_stack([xi, 1.0 - xi.sum(axis=1)])
    weights = np.array([p.weight for p in rule])
    return bary, weights / weights.sum()


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
        self.located = located
        built, fes, _ = fe_space(
            mesh, layout.support, layout.excluded, layout.space, layout.order, comm=comm
        )
        self._built: EngineMesh = built
        self._fes = fes
        self.factor = built.scale ** LENGTH_POWER[layout.space]
        inverse = inverse_order(built.cell_order, len(mesh.volume_elements))
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
            # the reference coordinates: the first dim barycentric ones
            mips["x"], mips["y"] = bary[:, 0], bary[:, 1]
            mips["z"] = bary[:, 2] if mesh.dim == 3 else 0.0
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

    def _evaluate(
        self, fes: ngs.FESpace, values: np.ndarray, *, curl: bool = False
    ) -> np.ndarray:
        """Values (or curls) at this rank's points, ``(its points, dim)``."""
        gf = ngs.GridFunction(fes)
        if self._mips is None:
            return np.zeros((0, 3 if curl else gf.dim))
        gf.vec.FV().NumPy()[:] = values
        out = np.asarray((ngs.curl(gf) if curl else gf)(self._mips))
        return out.reshape(self._own.size, -1)

    def values(self, coefficients: np.ndarray, *, curl: bool = False) -> np.ndarray:
        """``Phi c`` in SI units, ``(m, n_points, dim)`` for ``(m, width)``.

        With ``curl``, an H(curl) field's curl at the points instead, per
        metre more. Under ranks it is collective, with the same bits on every
        rank.
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
                np.stack([self._evaluate(self._fes, row, curl=curl) for row in part])
                for part in parts
            ]
        values = out[0] if len(out) == 1 else out[0] + 1j * out[1]
        values = values * (self.factor / self._built.scale if curl else self.factor)
        if self.comm is None:
            return values
        whole = np.zeros((c.shape[0], self.n_points, values.shape[2]), values.dtype)
        whole[:, self._own] = values
        return _mpi.reduce_bcast(self.comm, whole)

    def jet(self, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One H(curl) field's value and gradient at the points this rank holds.

        SI units: ``(n_points, 3)`` values and ``(n_points, 3, 3)``
        gradients, ``[p, i, j]`` the derivative along axis ``j`` of component
        ``i`` at point ``p``; zeros at the points other ranks hold (no rank
        communicates, so a sum over ranks counts each point once).
        ``coefficients`` is this rank's piece.
        """
        c = np.asarray(coefficients)[: self.ndof]
        scale = self._built.scale
        dtype = complex if np.iscomplexobj(c) else float
        values = np.zeros((self.n_points, 3), dtype)
        grads = np.zeros((self.n_points, 3, 3), dtype)
        if self._mips is None:
            return values, grads
        parts = [(1.0, np.real(c))] + ([(1j, np.imag(c))] if dtype is complex else [])
        with self._lock:
            for unit, part in parts:
                gf = ngs.GridFunction(self._fes)
                gf.vec.FV().NumPy()[:] = part
                value = np.asarray(gf(self._mips)).reshape(self._own.size, 3)
                grad = np.asarray(ngs.Grad(gf)(self._mips))
                values[self._own] += unit * value * self.factor
                grads[self._own] += (
                    unit * grad.reshape(self._own.size, 3, 3) * self.factor / scale
                )
        return values, grads

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

    def rows(self, weights: np.ndarray, rows: np.ndarray, n_rows: int) -> Any:
        """``(n_rows, ndof)`` sparse: row ``r`` sums ``weights[p] Phi_p`` of its points.

        A scalar field's weighted point sums as one matrix (``rows[p]`` is
        point ``p``'s row), SI scaling included. One process only.
        """
        import scipy.sparse

        if self.comm is not None:  # pragma: no cover - callers refuse ranks
            raise NotImplementedError("Probe.rows runs on one process")
        own = self._own
        data, r, c = [np.zeros(0)], [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
        if own.size:
            with self._lock:
                dfes, cont, disc = self._local_dofs()
                for k in range(cont.shape[1]):
                    unit = np.zeros(dfes.ndof)
                    unit[disc[:, k]] = 1.0
                    shape = self._evaluate(dfes, unit)[:, 0] * self.factor
                    data.append(shape * weights[own])
                    r.append(np.asarray(rows)[own])
                    c.append(cont[:, k])
        matrix = scipy.sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(r), np.concatenate(c))),
            shape=(n_rows, self.ndof),
        )
        return matrix.tocsr()

    def columns(self, weights: np.ndarray, curl: np.ndarray) -> Any:
        r"""``(width, n_points)`` sparse: column ``p`` is :math:`\Phi_p^\mathsf{T} w_p`.

        The point-by-point transpose of `transpose` for ``weights`` ``(n_points,
        dim)``, one vector per point: with ``curl[p]`` the curl evaluation at
        point ``p`` instead of the value. SI scaling included. The shape
        functions are evaluated once for all the points, and a column holds
        only the DOFs of its element. Under ranks a ``DISTRIBUTED`` load, as
        `transpose`: each rank fills the points it holds.
        """
        import scipy.sparse

        w = np.asarray(weights)
        curl = np.asarray(curl, bool)
        shape = (self.width, self.n_points)
        own = self._own
        if own.size == 0:
            return scipy.sparse.csc_matrix(shape, dtype=w.dtype)
        w, curl = w[own], curl[own]
        # a curl is one length derivative more than the value
        factors = [(False, self.factor), (True, self.factor / self._built.scale)]
        data, rows, cols = [], [], []
        with self._lock:
            dfes, cont, disc = self._local_dofs()
            for k in range(cont.shape[1]):
                unit = np.zeros(dfes.ndof)
                unit[disc[:, k]] = 1.0
                entry = np.zeros(own.size, w.dtype)
                for is_curl, factor in factors:
                    pick = curl == is_curl
                    if pick.any():
                        value = self._evaluate(dfes, unit, curl=is_curl)
                        pairs = np.einsum("pd,pd->p", value[pick], w[pick])
                        entry[pick] = pairs * factor
                data.append(entry)
                rows.append(cont[:, k])
                cols.append(own)
        matrix = scipy.sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=shape,
        )
        return matrix.tocsc()

    def transpose(self, cotangent: np.ndarray, *, curl: bool = False) -> np.ndarray:
        """``Phi^T g`` (SI scaling included), ``(m, width)`` for ``(m, n, dim)``.

        With ``curl``, the transpose of the curl evaluation instead (an
        H(curl) field's curl at the points, per square metre). Under ranks a
        ``DISTRIBUTED`` load, and no rank communicates: ``g`` is
        replicated, and each rank writes only for the points it holds.
        """
        g = np.asarray(cotangent).reshape(-1, self.n_points, cotangent.shape[-1])
        out = np.zeros((len(g), self.width), dtype=g.dtype)
        if self._own.size == 0:
            return out
        g = g[:, self._own]
        # a curl is one length derivative more than the value
        factor = self.factor / self._built.scale if curl else self.factor
        with self._lock:
            dfes, cont, disc = self._local_dofs()
            for k in range(cont.shape[1]):
                unit = np.zeros(dfes.ndof)
                unit[disc[:, k]] = 1.0
                shape = self._evaluate(dfes, unit, curl=curl) * factor  # (n, dim)
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


class GradientProbe(Probe):
    r"""The gradient :math:`\nabla u` of a scalar H1 layout at fixed points.

    `values` gives ``(m, n_points, dim)`` gradients per metre and `transpose`
    its transpose; the points are those of `Probe`.
    """

    def __init__(
        self, mesh: Mesh, layout: Layout, located: Located, comm: Any = None
    ) -> None:
        """Build the engine space and the points; derivatives are per mesh unit."""
        super().__init__(mesh, layout, located, comm)
        # d/dx of the engine's scaled coordinates
        self.factor = 1.0 / self._built.scale

    def hessians(self, coefficients: np.ndarray) -> np.ndarray:
        """The Hessian of one field at the points, ``(n_points, dim, dim)`` per m^2.

        Complex coefficients give complex Hessians. One process only.
        """
        c = np.asarray(coefficients)[: self.ndof]
        dim = self.located.barycentric.shape[1] - 1
        out = np.zeros((self.n_points, dim, dim), dtype=c.dtype)
        if self._mips is None:
            return out
        parts = [(1.0, np.real(c))] + ([(1j, np.imag(c))] if np.iscomplexobj(c) else [])
        with self._lock:
            for unit, part in parts:
                gf = ngs.GridFunction(self._fes)
                gf.vec.FV().NumPy()[:] = part
                h = np.asarray(gf.Operator("hesse")(self._mips))
                out[self._own] += unit * h.reshape(self._own.size, dim, dim)
        return out / self._built.scale**2

    def _evaluate(
        self, fes: ngs.FESpace, values: np.ndarray, *, curl: bool = False
    ) -> np.ndarray:
        """Gradients at this rank's points, ``(its points, dim)``."""
        if curl:
            raise ValueError("A scalar gradient probe reads no curl.")
        if self._mips is None:
            return np.zeros((0, 2))
        gf = ngs.GridFunction(fes)
        gf.vec.FV().NumPy()[:] = values
        out = np.asarray(ngs.grad(gf)(self._mips))
        return out.reshape(self._own.size, -1)


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
