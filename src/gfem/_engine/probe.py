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
"""

from __future__ import annotations

import collections
import dataclasses
import threading

import ngsolve as ngs
import numpy as np

from gfem._engine.fields import LENGTH_POWER, fe_space
from gfem._engine.mesh import EngineMesh
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


def locate(mesh: Mesh, layout: Layout, points: np.ndarray) -> Located:
    """The tetrahedra holding ``points`` (``(n, 3)``, m).

    Raises:
        ValueError: A point lies outside the mesh, or in a tetrahedron where
            the field is not defined (a volume outside its support, or one
            removed from the solve by the model's ``excluded=``).
    """
    from gfem._engine.cache import discretization

    built = discretization(mesh).built
    p = np.asarray(points, float).reshape(-1, 3) / built.scale
    found = built.mesh(p[:, 0].copy(), p[:, 1].copy(), p[:, 2].copy())
    nr = np.asarray(found["nr"], dtype=np.int64)
    outside = nr < 0
    if outside.any():
        worst = points.reshape(-1, 3)[int(np.flatnonzero(outside)[0])]
        raise ValueError(
            f"Point {_fmt(worst)} m is outside the mesh: a field has no value there."
        )
    tets = np.asarray(built.tetrahedron_order)[nr]
    inside = in_support(mesh, layout)[tets]
    if not inside.all():
        k = int(np.flatnonzero(~inside)[0])
        names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
        region = names[int(mesh.tetrahedron_tags[tets[k]])]
        raise ValueError(
            f"Point {_fmt(points.reshape(-1, 3)[k])} m lies in volume region "
            f"{region!r}, where the field is not defined (its support is "
            f"{list(layout.support)}, minus the volumes the model excludes from "
            "the solve). A point on the boundary of the support may be found in "
            "the neighbouring volume: move it inside."
        )
    ref = np.stack([found["x"], found["y"], found["z"]], axis=1)
    bary = np.concatenate([ref, 1.0 - ref.sum(axis=1, keepdims=True)], axis=1)
    return Located(tets, bary)


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

    def __init__(self, mesh: Mesh, layout: Layout, located: Located) -> None:
        """Build the engine space and the points (the mesh's node positions)."""
        self.layout = layout
        built, fes, _ = fe_space(
            mesh, layout.support, layout.excluded, layout.space, layout.order
        )
        self._built: EngineMesh = built
        self._fes = fes
        self.factor = built.scale ** LENGTH_POWER[layout.space]
        inverse = np.empty_like(built.tetrahedron_order)
        inverse[built.tetrahedron_order] = np.arange(len(inverse))
        self._elements = inverse[located.tetrahedra]
        n = len(self._elements)
        mips = built.mesh(np.zeros(1), np.zeros(1), np.zeros(1))
        mips = np.repeat(mips, n)
        mips["nr"] = self._elements
        mips["VorB"] = 0
        bary = located.barycentric
        mips["x"], mips["y"], mips["z"] = bary[:, 0], bary[:, 1], bary[:, 2]
        self._mips = mips
        self.n_points = n
        self._local: tuple | None = None
        self._lock = threading.Lock()

    @property
    def ndof(self) -> int:
        """The number of field coefficients."""
        return int(self._fes.ndof)

    def _evaluate(self, fes: ngs.FESpace, values: np.ndarray) -> np.ndarray:
        gf = ngs.GridFunction(fes)
        if self.n_points == 0:
            return np.zeros((0, gf.dim))
        gf.vec.FV().NumPy()[:] = values
        out = np.asarray(gf(self._mips))
        return out.reshape(self.n_points, -1)

    def values(self, coefficients: np.ndarray) -> np.ndarray:
        """``Phi c`` in SI units, ``(m, n_points, dim)`` for ``(m, ndof)``."""
        c = np.atleast_2d(coefficients)
        if c.shape[1] != self.ndof:
            raise ValueError(
                f"The field has {c.shape[1]} coefficients but its "
                f"{self.layout.space} order-{self.layout.order} space has "
                f"{self.ndof} DOFs: the field does not belong to this mesh."
            )
        parts = [np.real(c)] + ([np.imag(c)] if np.iscomplexobj(c) else [])
        with self._lock:
            out = [
                np.stack([self._evaluate(self._fes, row) for row in part])
                for part in parts
            ]
        values = out[0] if len(out) == 1 else out[0] + 1j * out[1]
        return values * self.factor

    def _local_dofs(self) -> tuple:
        if self._local is None:
            fes = self._fes
            dfes = ngs.Discontinuous(fes)
            unique, position = np.unique(self._elements, return_inverse=True)
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
        """``Phi^T g`` (SI scaling included), ``(m, ndof)`` for ``(m, n, dim)``."""
        g = np.asarray(cotangent).reshape(-1, self.n_points, cotangent.shape[-1])
        if self.n_points == 0:
            return np.zeros((len(g), self.ndof), dtype=g.dtype)
        with self._lock:
            dfes, cont, disc = self._local_dofs()
            out = np.zeros((len(g), self.ndof), dtype=g.dtype)
            for k in range(cont.shape[1]):
                unit = np.zeros(dfes.ndof)
                unit[disc[:, k]] = 1.0
                shape = self._evaluate(dfes, unit) * self.factor  # (n, dim)
                contribution = np.einsum("pd,mpd->mp", shape, g)
                for j in range(len(g)):
                    out[j] += np.bincount(
                        cont[:, k], weights=contribution[j].real, minlength=self.ndof
                    )
                    if np.iscomplexobj(g):
                        out[j] += 1j * np.bincount(
                            cont[:, k],
                            weights=contribution[j].imag,
                            minlength=self.ndof,
                        )
        return out


_PROBES: collections.OrderedDict[tuple, tuple] = collections.OrderedDict()
_MAX_PROBES = 4


def probe(mesh: Mesh, layout: Layout, located: Located) -> Probe:
    """The `Probe` of a layout at points, cached for the backward pass."""
    key = (
        id(mesh._topology),  # noqa: SLF001
        _digest(np.asarray(mesh.nodes)),
        layout,
        _digest(located.tetrahedra),
        _digest(located.barycentric),
    )
    found = _PROBES.get(key)
    if found is not None and found[0] is mesh._topology:  # noqa: SLF001
        _PROBES.move_to_end(key)
        return found[1]
    made = Probe(mesh, layout, located)
    _PROBES[key] = (mesh._topology, made)  # noqa: SLF001
    while len(_PROBES) > _MAX_PROBES:
        _PROBES.popitem(last=False)
    return made


def _digest(a: np.ndarray) -> int:
    a = np.ascontiguousarray(a)
    return hash((a.shape, a.dtype.str, a.tobytes()))
