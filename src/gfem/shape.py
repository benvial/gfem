"""Shape optimization: boundary morphs and mesh quality.

A shape design moves the nodes of the imported mesh. `boundary_morph` builds
a fixed linear map from normal displacements of a surface's nodes to the node
coordinates of a moved mesh, and `quality` measures how far the move has
degraded each element. There is no remeshing: when a design moves too far,
remesh at the current geometry and restart (see the "remesh and restart"
recipe in the docs).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from gfem import _morph
from gfem._boundaries import surfaces_of
from gfem._checks import is_traced
from gfem._pytree import readonly
from gfem._sparse import Coo, coo, spd_solve
from gfem.errors import MeshQualityError, MeshQualityWarning, RegionNotFoundError
from gfem.mesh import LENGTH_UNITS, Mesh
from gfem.model import Model, check_surface

__all__ = [
    "WARN_QUALITY_RATIO",
    "BoundaryMorph",
    "MeshQualityError",
    "MeshQualityWarning",
    "boundary_morph",
    "quality",
]

#: An element whose quality falls below this fraction of its imported quality
#: triggers a `MeshQualityWarning`.
WARN_QUALITY_RATIO = 0.25

_EXTENSIONS = ("elastic", "laplace")

# Corners of a tet and their three edges, ordered so that each corner's
# determinant is +6 V on a positively oriented tet.
_CORNERS = ((0, 1, 2, 3), (1, 2, 0, 3), (2, 0, 1, 3), (3, 0, 2, 1))


# --- mesh quality ------------------------------------------------------------


def _scaled_jacobian(xp: ModuleType, nodes, tets: np.ndarray, sign: np.ndarray):
    """Min over corners of sqrt(2) det(e1, e2, e3) / (|e1| |e2| |e3|), times sign."""
    q = None
    for a, b, c, d in _CORNERS:
        e1, e2, e3 = (nodes[tets[:, k]] - nodes[tets[:, a]] for k in (b, c, d))
        det = xp.sum(e1 * xp.cross(e2, e3), axis=1)
        norm = xp.prod(
            xp.stack([xp.linalg.norm(e, axis=1) for e in (e1, e2, e3)]), axis=0
        )
        corner = sign * np.sqrt(2) * det / norm
        q = corner if q is None else xp.minimum(q, corner)
    return q


def _orientation(mesh: Mesh) -> np.ndarray:
    """+1 / -1 per tet: the sign of its volume in the imported mesh."""
    x, tets = mesh.imported_nodes, mesh.tetrahedra
    e = [x[tets[:, k]] - x[tets[:, 0]] for k in (1, 2, 3)]
    det = np.einsum("ij,ij->i", e[0], np.cross(e[1], e[2]))
    return np.where(det < 0, -1.0, 1.0)


def _imported_quality(mesh: Mesh) -> np.ndarray:
    return _scaled_jacobian(
        np, mesh.imported_nodes, mesh.tetrahedra, _orientation(mesh)
    )


def quality(mesh: Mesh) -> jax.Array:
    """The scaled Jacobian of every tetrahedron, in mesh order.

    The scaled Jacobian is the minimum over a tet's four corners of
    ``sqrt(2) * det(e1, e2, e3) / (|e1| |e2| |e3|)``, with ``e1, e2, e3`` the
    corner's three edges. A regular tet scores 1, a flat one 0, and a tet
    turned inside out relative to the imported mesh scores below 0. Imported
    meshes already contain poor elements (gsim meshes reach 0.015-0.05), so
    compare against ``quality`` of the imported mesh rather than an absolute
    threshold: gfem warns when an element falls below a quarter of its
    imported value (`MeshQualityWarning`) and raises on an inverted element
    (`MeshQualityError`).

    It is a pure JAX function of the node coordinates: use it to accept or
    reject optimization steps, or, smoothly aggregated, as a constraint.

    Args:
        mesh: A mesh, typically moved by a `BoundaryMorph`.

    Returns:
        One value per tetrahedron, shape ``(n_tetrahedra,)``.
    """
    return _scaled_jacobian(
        jnp, jnp.asarray(mesh.nodes), mesh.tetrahedra, _orientation(mesh)
    )


def _check_quality(
    mesh: Mesh, active: np.ndarray | None = None, *, stacklevel: int = 2
) -> None:
    """Raise on an inverted element, warn on one far below its imported quality.

    Args:
        mesh: The mesh to check, with concrete node coordinates.
        active: Per tet, whether it counts (the solve domain); default all.
        stacklevel: Passed on to `warnings.warn`, counted from the caller.

    Raises:
        MeshQualityError: An element is inverted or flat.
    """
    nodes = np.asarray(mesh.nodes)
    if np.array_equal(nodes, mesh.imported_nodes):
        return
    tets = mesh.tetrahedra
    if active is None:
        active = np.ones(len(tets), dtype=bool)
    q = _scaled_jacobian(np, nodes, tets, _orientation(mesh))
    q0 = _imported_quality(mesh)
    inverted = active & (q <= 0)
    if inverted.any():
        worst = int(np.flatnonzero(inverted)[np.argmin(q[inverted])])
        what = (
            f"{int(inverted.sum())} element(s) of the moved mesh are inverted "
            f"(scaled Jacobian <= 0)"
        )
        msg = _quality_message(
            mesh, nodes=nodes, worst=worst, q=q, q0=q0, what=what, fatal=True
        )
        raise MeshQualityError(msg)
    ratio = q / q0
    poor = active & (ratio < WARN_QUALITY_RATIO)
    if poor.any():
        worst = int(np.flatnonzero(poor)[np.argmin(ratio[poor])])
        what = (
            f"{int(poor.sum())} element(s) of the moved mesh fell below "
            f"{WARN_QUALITY_RATIO} x their imported quality"
        )
        msg = _quality_message(
            mesh, nodes=nodes, worst=worst, q=q, q0=q0, what=what, fatal=False
        )
        warnings.warn(msg, MeshQualityWarning, stacklevel=stacklevel + 1)


def _quality_message(
    mesh: Mesh,
    *,
    nodes: np.ndarray,
    worst: int,
    q: np.ndarray,
    q0: np.ndarray,
    what: str,
    fatal: bool,
) -> str:
    unit = mesh.length_unit
    scale = LENGTH_UNITS[unit]
    tet = mesh.tetrahedra[worst]
    x0 = mesh.imported_nodes[tet]
    region = next(
        r.name
        for r in mesh.regions
        if r.dim == 3 and r.tag == mesh.tetrahedron_tags[worst]
    )
    centroid = nodes[tet].mean(axis=0) / scale
    edges = [np.linalg.norm(x0[i] - x0[j]) for i in range(4) for j in range(i + 1, 4)]
    moved = np.linalg.norm(nodes[tet] - x0, axis=1).max()
    where = ", ".join(f"{c:.4g}" for c in centroid)
    if fatal:
        effect = "The solve would be meaningless on this mesh."
    else:
        effect = "Results lose accuracy, and further steps may invert elements."
    return (
        f"{what}. The worst, element {worst} in region {region!r} at "
        f"({where}) {unit}, has quality {q[worst]:.3g} against {q0[worst]:.3g} "
        f"imported; its nodes moved by up to {moved / np.mean(edges):.2g} x its "
        f"mean edge length. {effect} Remedies: backtrack or take a smaller step; "
        "accept steps on gfem.shape.quality(mesh); parametrize the shape more "
        "smoothly; or remesh at the current geometry and restart (docs recipe "
        '"remesh and restart").'
    )


# --- boundary morph -----------------------------------------------------------


class BoundaryMorph:
    """A fixed linear map from normal displacements of surface nodes to a mesh.

    Build it with `boundary_morph`. Calling it with ``t``, one normal
    displacement in metres per parameter, returns the moved `Mesh`::

        moved = morph(t)
        model = model.replace(mesh=moved)

    The map is solved once on the imported mesh, so the same ``t`` always
    gives the same mesh, whatever was evaluated before. It is linear in
    ``t`` and differentiable with JAX (``grad``, ``jvp``, ``jit``, ``vmap``).

    Attributes:
        n_params: The number of parameters (moving nodes).
        points: Imported coordinates of the moving nodes, metres,
            shape ``(n_params, 3)``.
        normals: Unit normal of each moving node, shape ``(n_params, 3)``:
            parameter ``i`` moves node ``i`` by ``t[i] * normals[i]``.
        surface: The moving surface's name.
        extension: ``"elastic"`` or ``"laplace"``.
    """

    n_params: int
    points: np.ndarray
    normals: np.ndarray
    surface: str
    extension: str
    _treedef: Any
    _x0: np.ndarray
    _active: np.ndarray
    _factor: Any
    _reduced: Coo
    _coupling: Coo
    _free: Coo
    _prescribed: Coo
    _nodes: np.ndarray

    def __init__(
        self,
        *,
        mesh: Mesh,
        active: np.ndarray,
        surface: str,
        extension: str,
        nodes: np.ndarray,
        normals: np.ndarray,
        free: sp.csr_array,
        stiffness: sp.csr_array,
    ) -> None:
        """Internal: use `boundary_morph`."""
        # Imported here so that `import gfem` does not load NGSolve.
        from gfem._linalg import factorize

        prescribed = _morph.prescribed(len(mesh.imported_nodes), nodes, normals)
        reduced = (free.T @ stiffness @ free).tocoo()
        coupling = (-(free.T @ stiffness @ prescribed)).tocoo()
        factor = factorize(reduced)
        values = {
            "_treedef": jax.tree_util.tree_structure(mesh),
            "_x0": np.asarray(mesh.imported_nodes),
            "_active": active,
            "_factor": factor,
            "_reduced": coo(reduced),
            "_coupling": coo(coupling),
            "_free": coo(free.tocoo()),
            "_prescribed": coo(prescribed.tocoo()),
            "_nodes": readonly(nodes),
            "n_params": len(nodes),
            "points": readonly(np.asarray(mesh.imported_nodes)[nodes]),
            "normals": readonly(normals),
            "surface": surface,
            "extension": extension,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __call__(self, t: Any) -> Mesh:
        """The mesh with each moving node displaced by ``t[i] * normals[i]``.

        With a concrete ``t`` (not under a JAX transformation), the moved
        mesh's quality is checked at once; under ``jit`` or ``grad`` the
        check runs when the mesh is solved on.

        Args:
            t: Normal displacements in metres, shape ``(n_params,)``. For a
                positive ``t`` the surface moves into the solve domain.

        Returns:
            The moved mesh, which shares the imported mesh's topology.

        Raises:
            ValueError: ``t`` has the wrong shape.
            MeshQualityError: An element of the solve domain is inverted.

        Warns:
            MeshQualityWarning: An element fell below a quarter of its
                imported quality.
        """
        t = jnp.asarray(t, dtype=jnp.float64)
        if t.shape != (self.n_params,):
            msg = (
                f"This boundary morph takes {self.n_params} normal displacements "
                f"(one per moving node), got an array of shape {t.shape}."
            )
            raise ValueError(msg)

        rhs = self._coupling.matvec(t)
        z = spd_solve(self._reduced, self._factor, rhs)
        u = self._free.matvec(z) + self._prescribed.matvec(t)
        nodes = jnp.asarray(self._x0) + u.reshape(self._x0.shape)
        mesh = jax.tree_util.tree_unflatten(self._treedef, [nodes])
        if not is_traced(t):
            _check_quality(mesh, self._active, stacklevel=2)
        return mesh

    def __setattr__(self, name: str, value: Any) -> None:
        msg = "BoundaryMorph is immutable; build a new one with boundary_morph()."
        raise AttributeError(msg)

    def __repr__(self) -> str:
        return (
            f"BoundaryMorph(surface={self.surface!r}, n_params={self.n_params}, "
            f"extension={self.extension!r})"
        )


jax.tree_util.register_static(BoundaryMorph)


def boundary_morph(
    model: Model,
    moving: str,
    *,
    where: Callable[[np.ndarray], np.ndarray] | None = None,
    grows: str | None = None,
    fixed: Sequence[str] = (),
    extension: str = "elastic",
) -> BoundaryMorph:
    """Move a named surface of the mesh by normal displacements of its nodes.

    Each parameter is one node of the moving surface, displaced along its
    normal (the area-weighted normal of the selected faces around it, taken
    on the imported mesh). The rest of the mesh follows by a stiffened linear
    elastic extension, solved once on the imported mesh. Surfaces play roles:

    - the moving nodes are displaced by ``t[i] * normals[i]``;
    - nodes on port surfaces (from the model) and on ``fixed`` surfaces are
      held, and are not parameters even when they lie on the moving surface;
    - nodes on every other named surface slide within it: a planar facet
      keeps its plane, the line where two facets meet keeps its line, and a
      corner is held;
    - interior nodes, including those inside excluded volumes, follow.

    Sign: on a surface that bounds the solve domain on one side (an excluded
    conductor, a void, the mesh exterior), ``t > 0`` moves the surface into
    the domain, so the conductor or void grows. On an interface between two
    volumes of the solve domain, ``grows`` names the volume that grows for
    ``t > 0``.

    Shapes with few parameters are a JAX map from them to ``t`` built from
    `BoundaryMorph.points` and `BoundaryMorph.normals`, for example
    ``t = dw / 2 * jnp.ones(morph.n_params)`` widens a strip by ``dw``.

    Args:
        model: The model; its ports are held and its excluded volumes set the
            sign. Only its imported mesh geometry is used.
        moving: The moving surface region.
        where: Optional predicate that narrows the moving nodes. It gets the
            imported coordinates (metres) of the surface's nodes, shape
            ``(n, 3)``, and returns a boolean array of shape ``(n,)``. Mesh
            surfaces are often grouped by layer and orientation, so a name
            alone may hold every sidewall of every conductor.
        grows: The volume region that grows for ``t > 0`` where the moving
            surface separates two volumes of the solve domain.
        fixed: Further surface regions whose nodes are held.
        extension: ``"elastic"`` (default: linear elasticity, Poisson ratio
            0.3, stiffened by ``V_min / V_e``) or ``"laplace"`` (each
            coordinate harmonic, same stiffening).

    Returns:
        The morph: call it with ``t`` to get the moved mesh.

    Raises:
        RegionNotFoundError: A surface or volume name is not in the mesh.
        ValueError: The selection is empty or entirely held, ``grows`` is
            missing where it is needed (or given where it means nothing), or
            ``extension`` is unknown.
        TypeError: An argument has the wrong type.
    """
    if not isinstance(model, Model):
        msg = f"boundary_morph needs a gfem.Model, got {type(model).__name__}."
        raise TypeError(msg)
    if extension not in _EXTENSIONS:
        msg = f"Unknown extension {extension!r}; use 'elastic' or 'laplace'."
        raise ValueError(msg)
    mesh = model.mesh
    if isinstance(fixed, str) or not isinstance(fixed, (list, tuple)):
        msg = f"fixed= takes a list or tuple of surface names, got {fixed!r}."
        raise TypeError(msg)
    for name in (moving, *fixed):
        check_surface(
            mesh, name, "boundary_morph", " boundary_morph moves and holds surfaces."
        )
    if moving in fixed:
        msg = f"Surface {moving!r} is both moving and fixed=; remove it from fixed."
        raise ValueError(msg)
    x0 = np.asarray(mesh.imported_nodes)
    n_nodes = len(x0)
    held_surfaces = {s for port in model.ports for s in surfaces_of(port)}
    held_surfaces |= set(fixed)
    tag = {r.name: r.tag for r in mesh.regions if r.dim == 2}
    tri_tags = mesh.triangle_tags
    held_tri = np.isin(tri_tags, [tag[s] for s in held_surfaces])
    held = np.zeros(n_nodes, dtype=bool)
    held[mesh.triangles[held_tri].ravel()] = True
    in_tet = np.zeros(n_nodes, dtype=bool)
    in_tet[mesh.tetrahedra.ravel()] = True

    moving_tri = np.flatnonzero(tri_tags == tag[moving])
    candidates = np.unique(mesh.triangles[moving_tri])
    selected = _select(candidates, x0, where, moving)
    params = selected[~held[selected]]
    if len(params) == 0:
        msg = (
            f"Every selected node of {moving!r} ({len(selected)}) lies on a held "
            "surface (a port or fixed=), so nothing can move. Narrow fixed= or "
            "pick another surface."
        )
        raise ValueError(msg)
    normals = _node_normals(
        model,
        moving_tri=moving_tri,
        selected=selected,
        params=params,
        grows=grows,
        moving=moving,
    )

    constrained = held | ~in_tet
    constrained[params] = True
    slide = ~held_tri
    slide_tris = mesh.triangles[slide]
    free = _morph.free_directions(
        n_nodes, slide_tris, _morph.unit_normals(x0, slide_tris), constrained
    )
    stiffness = _morph.stiffness(x0, mesh.tetrahedra, extension)
    return BoundaryMorph(
        mesh=mesh,
        active=np.asarray(model._domain.active),  # noqa: SLF001
        surface=moving,
        extension=extension,
        nodes=params,
        normals=normals,
        free=free,
        stiffness=stiffness,
    )


def _select(
    candidates: np.ndarray,
    x0: np.ndarray,
    where: Callable[[np.ndarray], np.ndarray] | None,
    moving: str,
) -> np.ndarray:
    if where is None:
        return candidates
    points = x0[candidates]
    points.flags.writeable = False
    mask = np.asarray(where(points))
    if mask.shape != (len(candidates),) or mask.dtype != bool:
        msg = (
            f"where= must return a boolean array of shape ({len(candidates)},), one "
            f"value per node of {moving!r}; got {mask.dtype} of shape {mask.shape}."
        )
        raise ValueError(msg)
    if not mask.any():
        lo, hi = points.min(axis=0), points.max(axis=0)
        msg = (
            f"where= selects none of the {len(candidates)} nodes of {moving!r}. "
            f"They span x {lo[0]:.6g}..{hi[0]:.6g}, y {lo[1]:.6g}..{hi[1]:.6g}, "
            f"z {lo[2]:.6g}..{hi[2]:.6g} m (where= gets coordinates in metres)."
        )
        raise ValueError(msg)
    return candidates[mask]


def _node_normals(
    model: Model,
    *,
    moving_tri: np.ndarray,
    selected: np.ndarray,
    params: np.ndarray,
    grows: str | None,
    moving: str,
) -> np.ndarray:
    """Unit normal per parameter node, oriented by the sign convention."""
    mesh = model.mesh
    x0 = mesh.imported_nodes
    tris = mesh.triangles[moving_tri]
    is_selected = np.zeros(len(x0), dtype=bool)
    is_selected[selected] = True
    full = is_selected[tris].all(axis=1)
    # A node's normal comes from its fully selected faces; a node with none of
    # them uses every face of the surface it touches.
    covered = np.zeros(len(x0), dtype=bool)
    covered[tris[full].ravel()] = True
    is_param = np.zeros(len(x0), dtype=bool)
    is_param[params] = True
    use = full | (is_param[tris] & ~covered[tris]).any(axis=1)
    sign = np.zeros(len(tris))
    sign[use] = _face_signs(model, moving_tri[use], grows, moving)
    a, b, c = (x0[tris[:, k]] for k in range(3))
    weighted = sign[:, None] * np.cross(b - a, c - a)  # 2 x area x unit normal
    total = np.zeros((2, len(x0), 3))
    area = np.zeros((2, len(x0)))
    for k in range(3):
        for slot, faces in enumerate((full, np.ones_like(full))):
            np.add.at(total[slot], tris[faces, k], weighted[faces])
            np.add.at(
                area[slot], tris[faces, k], np.linalg.norm(weighted[faces], axis=1)
            )
    slot = np.where(covered[params], 0, 1)
    total, area = total[slot, params], area[slot, params]
    length = np.linalg.norm(total, axis=1)
    bad = length <= 1e-8 * area
    if bad.any():
        node = int(params[np.flatnonzero(bad)[0]])
        msg = (
            f"The selected faces of {moving!r} around node {node} at "
            f"{x0[node]} m have no mean normal (they face opposite ways). "
            "Narrow where= to one side of the feature."
        )
        raise ValueError(msg)
    return total / length[:, None]


def _face_signs(
    model: Model, faces: np.ndarray, grows: str | None, moving: str
) -> np.ndarray:
    """+1 where t > 0 moves along a face's right-hand normal, else -1."""
    mesh = model.mesh
    active = model._domain.active  # noqa: SLF001
    nb = mesh.triangle_neighbours[faces]
    exists = nb >= 0
    on_side = exists & active[np.maximum(nb, 0)]
    one_sided = on_side.sum(axis=1) == 1
    # The right-hand normal points from column 0 to column 1: into the domain
    # when the domain is on side 1.
    sign = np.where(on_side[:, 1], 1.0, -1.0)
    need = ~one_sided
    if not need.any():
        if grows is not None:
            msg = (
                f"grows={grows!r} has no effect: every selected face of {moving!r} "
                "bounds the solve domain on one side, where t > 0 always moves "
                "the surface into the domain. Remove grows=."
            )
            raise ValueError(msg)
        return sign
    if grows is None:
        msg = (
            f"{moving!r} separates two volumes of the solve domain "
            f"({_sides(model, faces[need])}), so the sign of t is ambiguous: pass "
            "grows='<volume>' to name the volume that grows for t > 0."
        )
        raise ValueError(msg)
    if grows not in mesh.volumes:
        raise RegionNotFoundError(grows, mesh.volumes, kind="volume")
    in_grows = exists & (
        mesh.tetrahedron_tags[np.maximum(nb, 0)] == mesh.region(grows).tag
    )
    ok = in_grows.sum(axis=1) == 1
    if not ok[need].all():
        msg = (
            f"grows={grows!r} must lie on exactly one side of every selected "
            f"interface face of {moving!r}, which separates "
            f"{_sides(model, faces[need & ~ok])}."
        )
        raise ValueError(msg)
    # The grows volume is behind the face (column 0): t > 0 moves along +n.
    return np.where(need, np.where(in_grows[:, 0], 1.0, -1.0), sign)


def _sides(model: Model, faces: np.ndarray) -> str:
    mesh = model.mesh
    name = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    nb = mesh.triangle_neighbours[faces]
    pairs = {
        tuple(name[int(mesh.tetrahedron_tags[t])] if t >= 0 else "outside" for t in row)
        for row in nb
    }
    return ", ".join(f"{a} | {b}" for a, b in sorted(pairs))
