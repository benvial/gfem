"""Shape optimization: boundary morphs and mesh quality.

A shape design moves the nodes of the imported mesh. `boundary_morph` builds
a fixed linear map from normal displacements of a surface's nodes to the node
coordinates of a moved mesh, and `quality` measures how far the move has
degraded each element. There is no remeshing: when a design moves too far,
remesh at the current geometry and restart (see the "remesh and restart"
recipe in the docs).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from gfem import _morph, _mpi, _periodic, _pml, _truncation
from gfem._boundaries import WavePort, surfaces_of
from gfem._checks import is_traced
from gfem._dim2 import refuse_entry
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

Extension = Literal["elastic", "laplace"]
_EXTENSIONS = get_args(Extension)

# Corners of a tet and their three edges, ordered so that each corner's
# determinant is +6 V on a positively oriented tet.
_CORNERS = ((0, 1, 2, 3), (1, 2, 0, 3), (2, 0, 1, 3), (3, 0, 2, 1))
# Corners of a triangle and their two edges, ordered so that each corner's
# determinant is +2 A on an anticlockwise triangle.
_TRIANGLE_CORNERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


# --- mesh quality ------------------------------------------------------------


def _det(xp: ModuleType, edges) -> Any:
    """Per element, the determinant of its corner's edges (two in 2D, three in 3D)."""
    if len(edges) == 2:
        e1, e2 = edges
        return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    e1, e2, e3 = edges
    return xp.sum(e1 * xp.cross(e2, e3), axis=1)


def _scaled_jacobian(xp: ModuleType, nodes, elements: np.ndarray, sign: np.ndarray):
    r"""The scaled Jacobian of each element, times ``sign``.

    The minimum over a tet's corners of
    :math:`\sqrt{2} \det(e_1, e_2, e_3) / (|e_1| |e_2| |e_3|)`, or over a
    triangle's corners of :math:`(2 / \sqrt{3}) \det(e_1, e_2) / (|e_1| |e_2|)`.
    """
    triangles = elements.shape[1] == 3
    corners, factor = (
        (_TRIANGLE_CORNERS, 2 / np.sqrt(3)) if triangles else (_CORNERS, np.sqrt(2))
    )
    q = None
    for a, *others in corners:
        edges = [nodes[elements[:, k]] - nodes[elements[:, a]] for k in others]
        norm = xp.prod(xp.stack([xp.linalg.norm(e, axis=1) for e in edges]), axis=0)
        corner = sign * factor * _det(xp, edges) / norm
        q = corner if q is None else xp.minimum(q, corner)
    return q


def _orientation(mesh: Mesh) -> np.ndarray:
    """+1 / -1 per element: the sign of its volume (area) in the imported mesh."""
    x, elements = mesh.imported_nodes, mesh.volume_elements
    det = _det(
        np, [x[elements[:, k]] - x[elements[:, 0]] for k in range(1, mesh.dim + 1)]
    )
    return np.where(det < 0, -1.0, 1.0)


def _imported_quality(mesh: Mesh) -> np.ndarray:
    return _scaled_jacobian(
        np, mesh.imported_nodes, mesh.volume_elements, _orientation(mesh)
    )


def quality(mesh: Mesh) -> jax.Array:
    r"""The scaled Jacobian of every volume element, in mesh order.

    The scaled Jacobian of a tetrahedron is the minimum over its four corners
    of :math:`\sqrt{2} \det(e_1, e_2, e_3) / (|e_1| |e_2| |e_3|)`, with
    :math:`e_1, e_2, e_3` the corner's three edges; that of a triangle (a 2D
    mesh) the minimum over its three corners of
    :math:`(2 / \sqrt{3}) \det(e_1, e_2) / (|e_1| |e_2|)`. A regular element
    scores 1, a flat one 0, and one turned inside out relative to the
    imported mesh scores below 0. Imported
    meshes already contain poor elements (gsim meshes reach 0.015-0.05), so
    compare against ``quality`` of the imported mesh rather than an absolute
    threshold: gfem warns when an element falls below a quarter of its
    imported value (`MeshQualityWarning`) and raises on an inverted element
    (`MeshQualityError`).

    It is a pure JAX function of the node coordinates: use it to accept or
    reject optimization steps, or, smoothly aggregated, as a constraint.

    Args:
        mesh: A mesh, typically moved by a `BoundaryMorph` or built with
            `Mesh.with_nodes`.

    Returns:
        One value per volume element (tetrahedron or triangle), shape
        ``(n_elements,)``.
    """
    return _scaled_jacobian(
        jnp, jnp.asarray(mesh.nodes), mesh.volume_elements, _orientation(mesh)
    )


def _check_quality(
    mesh: Mesh,
    active: np.ndarray | None = None,
    *,
    stacklevel: int = 2,
    comm: Any = None,
) -> None:
    """Raise on an inverted element, warn on one far below its imported quality.

    Args:
        mesh: The mesh to check, with concrete node coordinates.
        active: Per volume element, whether it counts (the solve domain);
            default all.
        stacklevel: Passed on to `warnings.warn`, counted from the caller.
        comm: The resolved communicator: rank 0 warns (`gfem._mpi.warn`).

    Raises:
        MeshQualityError: An element is inverted or flat.
    """
    nodes = np.asarray(mesh.nodes)
    if np.array_equal(nodes, mesh.imported_nodes):
        return
    elements = mesh.volume_elements
    if active is None:
        active = np.ones(len(elements), dtype=bool)
    q = _scaled_jacobian(np, nodes, elements, _orientation(mesh))
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
        _mpi.warn(comm, msg, MeshQualityWarning, stacklevel=stacklevel + 1)


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
    element = mesh.volume_elements[worst]
    x0 = mesh.imported_nodes[element]
    region = next(
        r.name
        for r in mesh.regions
        if r.dim == mesh.dim and r.tag == mesh._topology.volume_tags[worst]  # noqa: SLF001
    )
    centroid = nodes[element].mean(axis=0) / scale
    k = len(element)
    edges = [np.linalg.norm(x0[i] - x0[j]) for i in range(k) for j in range(i + 1, k)]
    moved = np.linalg.norm(nodes[element] - x0, axis=1).max()
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
    r"""A fixed linear map from normal displacements of surface nodes to a mesh.

    Build it with `boundary_morph`. Calling it with ``t``, one normal
    displacement in metres per parameter, returns the moved `Mesh`::

        moved = morph(t)
        model = model.replace(mesh=moved)

    The map is solved once on the imported mesh, so the same ``t`` always
    gives the same mesh, whatever was evaluated before. It is linear in
    ``t`` and differentiable with JAX (``grad``, ``jvp``, ``jit``, ``vmap``).

    Attributes:
        n_params: The number of parameters (moving nodes; on a periodic
            model, tie classes of paired nodes).
        points: Imported coordinates of the moving nodes, metres,
            shape ``(n_params, 3)`` (a tie class's representative, on the
            first surface of its pairs).
        normals: Unit normal of each moving node, shape ``(n_params, 3)``:
            parameter ``i`` moves node ``i`` (and its tie class) by
            ``t[i] * normals[i]``. A node on the face of a sliding wave port
            (``sliding_ports``) has the normal projected into the face's
            plane and scaled to keep :math:`n \cdot d = t`: :math:`n_t / |n_t|^2`.
        surface: The moving surface's name.
        extension: ``"elastic"`` or ``"laplace"``.
    """

    n_params: int
    points: np.ndarray
    normals: np.ndarray
    surface: str
    extension: Extension
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
        extension: Extension,
        nodes: np.ndarray,
        normals: np.ndarray,
        free: sp.csr_array,
        stiffness: sp.csr_array,
        ties: np.ndarray | None = None,
    ) -> None:
        """Internal: use `boundary_morph`."""
        # Imported here so that `import gfem` does not load NGSolve.
        from gfem._linalg import factorize

        prescribed = _morph.prescribed(len(mesh.imported_nodes), nodes, normals, ties)
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
            t: Normal displacements in metres, shape ``(n_params,)``, or a
                scalar: every moving node moves by it, the surface's uniform
                offset (its edge bias). For a positive ``t`` the surface moves
                into the solve domain.

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
        if t.ndim == 0:
            t = jnp.broadcast_to(t, (self.n_params,))
        if t.shape != (self.n_params,):
            msg = (
                f"This boundary morph takes {self.n_params} normal displacements "
                f"(one per moving node) or one scalar, got an array of shape {t.shape}."
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


def boundary_morph(  # noqa: PLR0915
    model: Model,
    moving: str,
    *,
    where: Callable[[np.ndarray], np.ndarray] | None = None,
    grows: str | None = None,
    fixed: Sequence[str] = (),
    sliding_ports: Sequence[str] = (),
    extension: Extension = "elastic",
) -> BoundaryMorph:
    r"""Move a named surface of the mesh by normal displacements of its nodes.

    Each parameter is one node of the moving surface, displaced along its
    normal (the area-weighted normal of the selected faces around it, taken
    on the imported mesh). The rest of the mesh follows by a stiffened linear
    elastic extension, solved once on the imported mesh. Surfaces play roles:

    - the moving nodes are displaced by ``t[i] * normals[i]``;
    - nodes on port surfaces (from the model; a wave port named in
      ``sliding_ports`` slides instead), on ``fixed`` surfaces, of
      the tetrahedra of a perfectly matched layer and of the elements
      touching the truncation (the physical tetrahedra with a vertex on the
      layer or on a `gfem.Absorbing` face) are held, and are not parameters
      even when they lie on the moving surface: the layer's stretch is
      fixed at the imported mesh, and the far field, the cross sections and
      the beam coupling read the field in the elements touching the
      truncation, whose geometry then stays as declared;
    - nodes on every other named surface slide within it: a planar facet
      keeps its plane, the line where two facets meet keeps its line, and a
      corner is held;
    - interior nodes, including those inside excluded volumes, follow.

    On a model with `gfem.Periodic` pairs, paired nodes are tied: each tie
    class (a node and its twins across every pair: 2 on a face, 4 on a cell
    edge, 8 at a corner of a 3-pair cell) moves by one displacement, so the
    moved mesh keeps every pair and its translation. A class is held if any
    member is held (``fixed=["x_min"]`` holds the matching "x_max" nodes
    too), and is one parameter if any member is a selected, unheld node of
    the moving surface: a ``where=`` that selects one side of the seam moves
    both twins. Its normal sums its members' faces. Paired surfaces slide,
    and may not be the moving surface (that would change the period). The
    gradient with respect to ``t`` is then exact along the tied motion; a
    raw gradient with respect to ``mesh.nodes`` is not tied (it is exact
    along every move that keeps the pairs).

    Sign: on a surface that bounds the solve domain on one side (an excluded
    conductor, a void, the mesh exterior), ``t > 0`` moves the surface into
    the domain, so the conductor or void grows. On an interface between two
    volumes of the solve domain, ``grows`` names the volume that grows for
    ``t > 0``.

    Shapes with few parameters are a JAX map from them to ``t`` built from
    `BoundaryMorph.points` and `BoundaryMorph.normals`, for example
    ``t = dw / 2 * jnp.ones(morph.n_params)`` widens a strip by ``dw``.

    Args:
        model: The model; its ports, its perfectly matched layer and the
            elements touching its truncation are held, and its excluded
            volumes set the sign. Only its imported
            mesh geometry is used.
        moving: The moving surface region.
        where: Optional predicate that narrows the moving nodes. It gets the
            imported coordinates (metres) of the surface's nodes, shape
            ``(n, 3)``, and returns a boolean array of shape ``(n,)``. Mesh
            surfaces are often grouped by layer and orientation, so a name
            alone may hold every sidewall of every conductor.
        grows: The volume region that grows for ``t > 0`` where the moving
            surface separates two volumes of the solve domain.
        fixed: Further surface regions whose nodes are held.
        sliding_ports: Wave ports whose faces slide in their plane instead of
            being held, so a design reaches the port: a taper's width at its
            port, say. The face's nodes then move as those of any other
            named surface do (in the face's plane, its edges along their
            lines). A node of the moving surface on such a face moves by
            :math:`t\,n_t/|n_t|^2`, with :math:`n_t` the moving surface's
            normal projected into the face's plane, which keeps it in the
            plane and on the moved surface. The port keeps its reference
            plane, and its mode follows the moved section. Naming one wave
            port of a shared port surface lets the whole surface slide: its
            ports share the face. Lumped and Floquet ports stay held.
        extension: ``"elastic"`` (default: linear elasticity, Poisson ratio
            0.3, stiffened by ``V_min / V_e``) or ``"laplace"`` (each
            coordinate harmonic, same stiffening).

    Returns:
        The morph: call it with ``t`` to get the moved mesh.

    Raises:
        RegionNotFoundError: A surface or volume name is not in the mesh.
        ValueError: The selection is empty or entirely held (on ports,
            ``fixed`` surfaces, in the layer or in the elements touching
            the truncation), ``grows`` is
            missing where it is needed (or given where it means nothing),
            ``extension`` is unknown, or ``moving`` is a periodic surface; a
            name in ``sliding_ports`` is not a port of the model, or is a
            lumped or Floquet port (their geometry is held by the gradient);
            a sliding port's surface is the moving surface (a port face
            moves only in its plane: move the surfaces that meet it) or is in
            ``fixed``; or a moving node on a sliding face lies where the
            moving surface is within 1 degree of the face's plane.
        TypeError: An argument has the wrong type (``fixed`` or
            ``sliding_ports`` is a string, not a sequence of names).
    """
    refuse_entry(getattr(model, "mesh", None), "boundary_morph")
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
    if isinstance(sliding_ports, str) or not isinstance(sliding_ports, (list, tuple)):
        msg = (
            f"sliding_ports= takes a list or tuple of wave port names, got "
            f"{sliding_ports!r}."
        )
        raise TypeError(msg)
    sliding_surfaces = _sliding_surfaces(model, sliding_ports, moving, fixed)
    pairings = model._periodic  # noqa: SLF001
    for p in pairings:
        if moving in (p.first, p.second):
            twin = p.second if moving == p.first else p.first
            msg = (
                f"Periodic surface {moving!r} cannot be the moving surface: moving "
                "it along its normal changes the period. Its nodes slide tied to "
                f"{twin!r}."
            )
            raise ValueError(msg)
    x0 = np.asarray(mesh.imported_nodes)
    n_nodes = len(x0)
    # paired nodes move together: one displacement per tie class
    ties = _periodic.tie_classes(n_nodes, pairings)
    held, held_tri, _ = _held(model, ties, sliding_surfaces, fixed)
    tag = {r.name: r.tag for r in mesh.regions if r.dim == mesh.dim - 1}
    tri_tags = mesh.triangle_tags
    in_tet = np.zeros(n_nodes, dtype=bool)
    in_tet[mesh.tetrahedra.ravel()] = True

    moving_tri = np.flatnonzero(tri_tags == tag[moving])
    candidates = np.unique(mesh.triangles[moving_tri])
    selected = _select(candidates, x0, where, moving)
    params = np.unique(ties[selected[~held[selected]]])
    if len(params) == 0:
        msg = (
            f"Every selected node of {moving!r} ({len(selected)}) lies on a held "
            "surface (a port or fixed=), in the perfectly matched layer, or in "
            "the elements touching the truncation (the layer or an Absorbing "
            "face), so nothing can move. Narrow fixed= or pick another surface."
        )
        raise ValueError(msg)
    # every node of the moving surface in a parameter's class moves with it
    is_param = np.zeros(n_nodes, dtype=bool)
    is_param[params] = True
    movers = candidates[is_param[ties[candidates]]]
    normals = _node_normals(
        model,
        moving_tri=moving_tri,
        selected=np.union1d(selected, movers),
        params=movers,
        grows=grows,
        moving=moving,
        groups=np.searchsorted(params, ties[movers]),
    )
    if sliding_surfaces:
        normals = _project_on_sliding_faces(
            model, normals, params, sliding_surfaces, moving
        )

    constrained = held | ~in_tet
    constrained[params] = True
    slide = ~held_tri
    slide_tris = mesh.triangles[slide]
    free = _morph.free_directions(
        n_nodes, slide_tris, _morph.unit_normals(x0, slide_tris), constrained, ties
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
        ties=ties if pairings else None,
    )


def _held(
    model: Model,
    ties: np.ndarray,
    sliding_surfaces: set[str],
    fixed: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The nodes a morph holds, the faces that hold them, and why.

    Held are the nodes of the ports' surfaces (except the sliding ones), of
    the ``fixed`` surfaces, of the tetrahedra of a perfectly matched layer
    (its stretch is fixed at the imported mesh) and of the truncation layer
    (the physical tetrahedra touching the layer or a `gfem.Absorbing` face,
    where the far field, the cross sections and the beam coupling read the
    field); a tie class is held if any member is.

    Returns:
        ``(held, held_tri, cause)``: a mask per node, a mask per surface
        triangle of the held-surface faces, and per node a label of what holds
        it (``""`` when free), the same on every member of a tie class.
    """
    mesh = model.mesh
    n_nodes = len(mesh.imported_nodes)
    holder: dict[str, str] = {}
    for port in model.ports:
        for s in surfaces_of(port):
            if s not in sliding_surfaces:
                holder.setdefault(s, f"port {port.name!r}")
    for s in fixed:
        holder.setdefault(s, f"fixed surface {s!r}")
    tag = {r.name: r.tag for r in mesh.regions if r.dim == mesh.dim - 1}
    held_tri = np.isin(mesh.triangle_tags, [tag[s] for s in holder])
    cause = np.full(n_nodes, "", dtype=object)

    def label(nodes: np.ndarray, why: str) -> None:
        nodes = np.unique(nodes).astype(np.int64)
        cause[nodes[cause[nodes] == ""]] = why  # the first cause stays

    for s, why in holder.items():
        label(mesh.triangles[mesh.triangle_tags == tag[s]].ravel(), why)
    layer = _pml.layer_tetrahedra(mesh, model.pml)
    if layer is not None:
        label(mesh.tetrahedra[layer], "the perfectly matched layer")
    label(
        _truncation.truncation(model).layer_nodes(mesh.tetrahedra),
        "the truncation layer",
    )
    # a class is held, with one cause, if any member is
    named = np.flatnonzero(cause != "")[::-1]
    by_class = np.full(n_nodes, "", dtype=object)
    by_class[ties[named]] = cause[named]  # the last write is the first member
    cause = by_class[ties]
    return cause != "", held_tri, cause


def _sliding_surfaces(
    model: Model,
    names: Sequence[str],
    moving: str | None = None,
    fixed: Sequence[str] = (),
) -> set[str]:
    """The surfaces of the wave ports named in ``sliding_ports``, after checks.

    ``moving`` and ``fixed`` are the morph's moving and held surfaces, which a
    sliding face may be neither of; a morph that names none passes neither.
    """
    ports = {p.name: p for p in model.ports}
    waves = [p.name for p in model.ports if isinstance(p, WavePort)]
    surfaces: set[str] = set()
    for name in names:
        port = ports.get(name)
        if port is None:
            msg = (
                f"sliding_ports names {name!r}, which is not a port of the model. "
                f"Its wave ports: {', '.join(map(repr, waves)) or 'none'}."
            )
            raise ValueError(msg)
        if not isinstance(port, WavePort):
            msg = (
                f"sliding_ports names {name!r}, a {type(port).__name__}; only a "
                "wave port's face can slide (a lumped or Floquet port's geometry "
                "is held by shape gradients). Its wave ports: "
                f"{', '.join(map(repr, waves)) or 'none'}."
            )
            raise ValueError(msg)  # noqa: TRY004 - a name's value, not a type
        if port.surface == moving:
            msg = (
                f"Surface {moving!r} is the face of the sliding wave port "
                f"{name!r}: a port face moves only in its plane. Move the "
                "surfaces that meet it; the face's nodes follow."
            )
            raise ValueError(msg)
        if port.surface in fixed:
            msg = (
                f"Surface {port.surface!r} is the face of the sliding wave port "
                f"{name!r} and is also in fixed=: a face cannot both slide and be "
                "held. Remove it from one of them."
            )
            raise ValueError(msg)
        surfaces |= set(surfaces_of(port))
    return surfaces


#: A node of the moving surface on a sliding face is refused where the
#: surface is within this angle of the face's plane (the projected normal
#: would be shorter than its sine and the node would travel 1 / sin times t).
_MIN_SLIDE_ANGLE = np.radians(1.0)


def _project_on_sliding_faces(
    model: Model,
    normals: np.ndarray,
    params: np.ndarray,
    surfaces: set[str],
    moving: str,
) -> np.ndarray:
    r"""``normals`` with those on a sliding face moved into the face's plane.

    A node of the moving surface on a sliding face moves by
    :math:`t\,n_t/|n_t|^2`, with :math:`n_t` the normal projected on the
    plane (or the line, on two faces) the node may move in: it stays on the
    face and on the moved surface (:math:`n\cdot d = t`).

    Raises:
        ValueError: The moving surface is within 1 degree of a face's plane
            at a node.
    """
    mesh = model.mesh
    x0 = mesh.imported_nodes
    tags = [mesh.region(s).tag for s in sorted(surfaces)]
    faces = mesh.triangles[np.isin(mesh.triangle_tags, tags)]
    face_normals = _morph.unit_normals(x0, faces)
    corner = faces.ravel()
    corner_normal = np.repeat(face_normals, 3, axis=0)
    out = normals.copy()
    for i, node in enumerate(params):
        here = corner_normal[corner == node]
        if not len(here):
            continue
        _, s, vt = np.linalg.svd(here)
        rank = int((s > 1e-8 * s[0]).sum())
        free = vt[rank:]  # the directions the node may move in
        n_t = free.T @ (free @ normals[i]) if len(free) else np.zeros(3)
        length = float(np.linalg.norm(n_t))
        if length < np.sin(_MIN_SLIDE_ANGLE):
            msg = (
                f"Moving surface {moving!r} lies within 1 degree of the plane of a "
                f"sliding wave port's face at node {int(node)} ({x0[node]} m): "
                "a node on the face moves only in the face's plane, so it would "
                "have to travel 1/sin(angle) times its displacement. Move a "
                "surface that meets the port across its plane, or narrow where= "
                "away from the face."
            )
            raise ValueError(msg)
        out[i] = n_t / length**2
    return out


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
    groups: np.ndarray,
) -> np.ndarray:
    """Unit normal per parameter, oriented by the sign convention.

    ``params`` are moving nodes, ``groups`` the parameter each moves with
    (one parameter per tie class): a parameter's normal is the normalized
    sum of its nodes' area-weighted normals (on a periodic surface, the two
    half-stars of twins form the full star).
    """
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
    n_params = int(groups.max()) + 1
    summed, summed_area = np.zeros((n_params, 3)), np.zeros(n_params)
    np.add.at(summed, groups, total[slot, params])
    np.add.at(summed_area, groups, area[slot, params])
    total, area = summed, summed_area
    length = np.linalg.norm(total, axis=1)
    bad = length <= 1e-8 * area
    if bad.any():
        node = int(params[np.flatnonzero(groups == np.flatnonzero(bad)[0])[0]])
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
    name = mesh._region_names()  # noqa: SLF001
    nb = mesh.triangle_neighbours[faces]
    pairs = {
        tuple(name[int(mesh.tetrahedron_tags[t])] if t >= 0 else "outside" for t in row)
        for row in nb
    }
    return ", ".join(f"{a} | {b}" for a, b in sorted(pairs))
