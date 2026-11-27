"""Shape optimization: boundary and CAD morphs and mesh quality.

A shape design moves the nodes of the imported mesh. `boundary_morph` builds
a fixed linear map from normal displacements of a surface's nodes to the node
coordinates of a moved mesh, and `cad_morph` one from the named parameters of
a parametric gmsh model; `quality` measures how far a move has degraded each
element. A morph is linear, so when a design moves too far the mesh must be
made again at the current geometry: a CAD morph remeshes in one call
(`CadMorph.remesh`), a boundary morph needs the remesh-and-restart recipe in
the docs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from gfem import _cad, _morph, _mpi, _periodic, _pml, _truncation
from gfem._boundaries import WavePort, surfaces_of
from gfem._checks import is_traced
from gfem._curved import gated_triangles
from gfem._dim2 import refuse_entry
from gfem._pytree import readonly
from gfem._sparse import Coo, coo, spd_solve
from gfem.errors import MeshQualityError, MeshQualityWarning, RegionNotFoundError
from gfem.mesh import _TET_EDGES, _TRI_EDGES, LENGTH_UNITS, Mesh, _check_unit
from gfem.model import Model, check_surface

__all__ = [
    "WARN_QUALITY_RATIO",
    "BoundaryMorph",
    "CadMorph",
    "MeshQualityError",
    "MeshQualityWarning",
    "boundary_morph",
    "cad_morph",
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

# The 20 points of the cubic lattice of a tetrahedron, as multi-indices
# alpha with |alpha| = 3: barycentric coordinates alpha / 3.
_LATTICE_INDEX = np.array(
    [
        (a, b, c, 3 - a - b - c)
        for a in range(4)
        for b in range(4 - a)
        for c in range(4 - a - b)
    ]
)
_LATTICE = _LATTICE_INDEX / 3


def _lattice_gradients() -> np.ndarray:
    r"""The derivatives of the quadratic map along each barycentric coordinate.

    Entry ``[p, i, n]`` is the weight of node ``n`` (the four corners, then the
    mid-edge nodes in `_TET_EDGES` order) in the derivative
    :math:`\partial x / \partial \lambda_i` at lattice point ``p``, for the map

    .. math::
        x = \sum_i \lambda_i (2 \lambda_i - 1) x_i
            + \sum_{ab} 4 \lambda_a \lambda_b x_{ab},

    with the four coordinates taken as independent.
    A difference :math:`\partial_j x - \partial_c x` is the derivative along
    the edge from corner :math:`c` to corner :math:`j`, and on a straight
    tetrahedron it is :math:`x_j - x_c`.
    """
    g = np.zeros((len(_LATTICE), 4, 10))
    for i in range(4):
        g[:, i, i] = 4 * _LATTICE[:, i] - 1
    for k, (a, b) in enumerate(_TET_EDGES):
        g[:, a, 4 + k] = 4 * _LATTICE[:, b]
        g[:, b, 4 + k] = 4 * _LATTICE[:, a]
    return g


def _bernstein_inverse() -> np.ndarray:
    r"""The map from a cubic's 20 lattice values to its Bernstein coefficients.

    The degree-3 Bernstein basis is :math:`B_\alpha = (3! / \alpha!)
    \lambda^\alpha`. A cubic's coefficients bound its minimum from below.
    """
    factorial = np.array([1.0, 1.0, 2.0, 6.0])
    weight = 6.0 / np.prod(factorial[_LATTICE_INDEX], axis=1)
    basis = weight * np.prod(_LATTICE[:, None, :] ** _LATTICE_INDEX[None, :, :], axis=2)
    return np.linalg.inv(basis)


_GRADIENTS = _lattice_gradients()
_BERNSTEIN_INVERSE = _bernstein_inverse()


def _curved_edges(xp: ModuleType, nodes, elements: np.ndarray, midnodes: np.ndarray):
    r"""The quadratic map's derivatives at the lattice, ``(n, 20, 4, 3)``.

    Entry ``[e, p, i]`` is :math:`\partial x / \partial \lambda_i` of element
    ``e`` at lattice point ``p`` (`_lattice_gradients`).
    """
    x10 = nodes[np.concatenate([elements, midnodes], axis=1)]
    return xp.einsum("pin,enx->epix", xp.asarray(_GRADIENTS), x10)


def _curved_scaled_jacobian(
    xp: ModuleType, nodes, elements: np.ndarray, midnodes: np.ndarray, sign
):
    """The scaled Jacobian of each curved tetrahedron, times ``sign``.

    The minimum over the 20 lattice points and the four corner frames
    (`_CORNERS`) of the straight element's measure on the frame's three edge
    derivatives; on a straight tetrahedron it is `_scaled_jacobian`'s value.
    """
    d = _curved_edges(xp, nodes, elements, midnodes)
    q = None
    for a, *others in _CORNERS:
        edges = [d[:, :, k] - d[:, :, a] for k in others]
        norm = xp.prod(xp.stack([xp.linalg.norm(e, axis=-1) for e in edges]), axis=0)
        e1, e2, e3 = edges
        det = xp.sum(e1 * xp.cross(e2, e3), axis=-1)
        corner = xp.min(_ratio(xp, sign[:, None] * np.sqrt(2) * det, norm), axis=1)
        q = corner if q is None else xp.minimum(q, corner)
    return q


def _bernstein_bound(nodes, elements: np.ndarray, midnodes: np.ndarray, sign):
    """A lower bound of each curved tetrahedron's Jacobian determinant, times ``sign``.

    The smallest Bernstein coefficient of the determinant, a cubic in the
    element's coordinates known at the 20 lattice points. It is at most the
    determinant's minimum, so a bound at or below zero means the element may
    fold between the sample points.
    """
    d = _curved_edges(np, nodes, elements, midnodes)
    e1, e2, e3 = (d[:, :, k] - d[:, :, 0] for k in (1, 2, 3))
    det = sign[:, None] * np.sum(e1 * np.cross(e2, e3), axis=-1)
    return (det @ _BERNSTEIN_INVERSE.T).min(axis=1)


def _ratio(xp: ModuleType, numerator, norm):
    """``numerator / norm``, 0 where ``norm`` is 0 (a collapsed edge is a flat element).

    Elsewhere it is the quotient itself, so a regular element's value is the
    plain division's. The inner ``where`` keeps the unused branch finite.
    """
    return xp.where(norm > 0, numerator / xp.where(norm > 0, norm, 1.0), 0.0)


def _det(xp: ModuleType, edges) -> Any:
    """Per element, the determinant of its corner's edges (two in 2D, three in 3D)."""
    if len(edges) == 2:
        e1, e2 = edges
        return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    e1, e2, e3 = edges
    return xp.sum(e1 * xp.cross(e2, e3), axis=1)


def _scaled_jacobian(
    xp: ModuleType,
    nodes,
    elements: np.ndarray,
    sign: np.ndarray,
    midnodes: np.ndarray | None = None,
):
    r"""The scaled Jacobian of each element, times ``sign``.

    With the mid-edge nodes of a curved mesh (``midnodes``), the minimum over
    the cubic lattice of the curved tetrahedra (`_curved_scaled_jacobian`).
    Otherwise the minimum over a tet's corners of
    :math:`\sqrt{2} \det(e_1, e_2, e_3) / (|e_1| |e_2| |e_3|)`, or over a
    triangle's corners of :math:`(2 / \sqrt{3}) \det(e_1, e_2) / (|e_1| |e_2|)`.
    """
    if midnodes is not None:
        return _curved_scaled_jacobian(xp, nodes, elements, midnodes, sign)
    triangles = elements.shape[1] == 3
    corners, factor = (
        (_TRIANGLE_CORNERS, 2 / np.sqrt(3)) if triangles else (_CORNERS, np.sqrt(2))
    )
    q = None
    for a, *others in corners:
        edges = [nodes[elements[:, k]] - nodes[elements[:, a]] for k in others]
        norm = xp.prod(xp.stack([xp.linalg.norm(e, axis=1) for e in edges]), axis=0)
        corner = _ratio(xp, sign * factor * _det(xp, edges), norm)
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
        np,
        mesh.imported_nodes,
        mesh.volume_elements,
        _orientation(mesh),
        mesh._topology.volume_midnodes,  # noqa: SLF001
    )


def quality(mesh: Mesh) -> jax.Array:
    r"""The scaled Jacobian of every volume element, in mesh order.

    The scaled Jacobian of a tetrahedron is the minimum over its four corners
    of :math:`\sqrt{2} \det(e_1, e_2, e_3) / (|e_1| |e_2| |e_3|)`, with
    :math:`e_1, e_2, e_3` the corner's three edges; that of a triangle (a 2D
    mesh) the minimum over its three corners of
    :math:`(2 / \sqrt{3}) \det(e_1, e_2) / (|e_1| |e_2|)`. A regular element
    scores 1, a flat one 0 (also one with a collapsed edge), and one turned
    inside out relative to the imported mesh scores below 0. On a curved mesh
    a tetrahedron scores the smallest such value over the 20 points of the
    cubic lattice of its quadratic map and over the four corners' frames
    (each frame's edges are the map's derivatives along them), which on a
    straight tetrahedron is the value above. Imported
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
        jnp,
        jnp.asarray(mesh.nodes),
        mesh.volume_elements,
        _orientation(mesh),
        mesh._topology.volume_midnodes,  # noqa: SLF001
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
    midnodes = mesh._topology.volume_midnodes  # noqa: SLF001
    q = _scaled_jacobian(np, nodes, elements, _orientation(mesh), midnodes)
    q0 = _imported_quality(mesh)
    inverted = active & ~(q > 0)  # not (q <= 0): a NaN is bad too
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
    if midnodes is not None:
        _warn_between_samples(mesh, nodes, active, stacklevel + 1, comm)
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


def _warn_between_samples(
    mesh: Mesh, nodes: np.ndarray, active: np.ndarray, stacklevel: int, comm: Any
) -> None:
    """Warn of curved tetrahedra whose Jacobian bound is not positive.

    Their sampled values are (the caller raised on the others), so they may fold
    only between the 20 sample points.
    """
    topology = mesh._topology  # noqa: SLF001
    midnodes = topology.volume_midnodes
    if midnodes is None:  # a straight mesh has no sample points to miss
        return
    bound = _bernstein_bound(nodes, mesh.volume_elements, midnodes, _orientation(mesh))
    unsure = active & (bound <= 0)
    if unsure.any():
        _mpi.warn(
            comm,
            f"{int(unsure.sum())} curved element(s) of the mesh may be folded "
            "between sample points: a Bernstein lower bound of their Jacobian "
            "determinant is not positive, though it is at the 20 points of the "
            "cubic lattice. Remedies: take a smaller step, or remesh with gmsh "
            "`Mesh.HighOrderOptimize 1`.",
            MeshQualityWarning,
            stacklevel=stacklevel + 1,
        )


def _check_curved_read(mesh: Mesh, path: Any) -> None:
    """The read-time check of a curved mesh: raise on a fold, warn on a doubt.

    Raises:
        MeshQualityError: A curved tetrahedron has a scaled Jacobian <= 0 at one
            of the 20 lattice points.
    """
    topology = mesh._topology  # noqa: SLF001
    q = np.asarray(_imported_quality(mesh))
    folded = ~(q > 0)  # also a NaN: an edge of zero length
    if folded.any():
        value = np.where(np.isnan(q), -np.inf, q)
        worst = int(np.flatnonzero(folded)[np.argmin(value[folded])])
        centroid = mesh.imported_nodes[topology.volume_elements[worst]].mean(axis=0)
        unit = mesh.length_unit
        where = ", ".join(f"{c / LENGTH_UNITS[unit]:.4g}" for c in centroid)
        raise MeshQualityError(
            f"{path.name}: {int(folded.sum())} curved tetrahedra are folded (the "
            "scaled Jacobian of the quadratic map is <= 0 at a sample point). "
            f"The worst, element {worst}, is at ({where}) {unit} with value "
            f"{q[worst]:.3g}. Mesh the curved geometry with gmsh "
            "`Mesh.HighOrderOptimize 1` (Python: "
            'gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)), or with a '
            "smaller mesh size on the curved walls."
        )
    _warn_between_samples(
        mesh,
        mesh.imported_nodes,
        np.ones(len(q), dtype=bool),
        stacklevel=4,
        comm=None,
    )


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
    held = element
    if mesh._topology.volume_midnodes is not None:  # noqa: SLF001
        held = np.concatenate([element, mesh._topology.volume_midnodes[worst]])  # noqa: SLF001
    moved = np.linalg.norm(nodes[held] - mesh.imported_nodes[held], axis=1).max()
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
    _chord: Coo | None
    _bubble: Coo | None

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
        chord: sp.csr_array | None = None,
        bubble: sp.csr_array | None = None,
    ) -> None:
        """Internal: use `boundary_morph`.

        ``chord`` and ``bubble`` are the mid-edge node maps of a curved mesh
        (`_midnode_maps`), ``None`` on a straight one.
        """
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
            "_chord": None if chord is None else coo(chord.tocoo()),
            "_bubble": None if bubble is None else coo(bubble.tocoo()),
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
        if self._chord is not None and self._bubble is not None:
            # mid-edge nodes: the mean of their corners' motion, plus the
            # offset that puts the moving wall's own along its normal
            u = u + self._chord.matvec(u) + self._bubble.matvec(t)
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

    On a curved mesh the parameters stay the corner nodes, and the normals
    come from the quadratic triangles. A mid-edge node on the moving surface
    whose edge has both corners moving moves by the mean of their ``t`` along
    its own normal, :math:`\tfrac12 (t_a + t_b)\, n_m`, so a uniform ``t``
    puts every wall node on the offset wall. Every other mid-edge node
    follows its chord, :math:`u_m = \tfrac12 (u_a + u_b)`: interior, sliding
    and held ones, and those of the faces a feature reads as straight (ports,
    periodic faces, sheet designs), which therefore stay straight.

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
    # The extension is solved on the corner mesh: mid-edge nodes of a curved
    # mesh are no unknowns of it (they are placed after it).
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
    normals, mid_nodes, mid_normals = _node_normals(
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
    chord = bubble = None
    if mesh.curved:
        chord, bubble = _midnode_maps(
            model,
            moving_tri=moving_tri,
            params=params,
            ties=ties,
            normals=normals,
            mid_nodes=mid_nodes,
            mid_normals=mid_normals,
            held=held,
        )
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
        chord=chord,
        bubble=bubble,
    )


def _edge_ends(topology: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each mid-edge node of a curved mesh, with the two corners of its edge."""
    corners, mids = topology.volume_elements, topology.volume_midnodes
    every = np.concatenate([mids[:, k] for k in range(len(_TET_EDGES))])
    ends = [np.concatenate([corners[:, e[end]] for e in _TET_EDGES]) for end in (0, 1)]
    m, first = np.unique(every, return_index=True)
    return m, ends[0][first], ends[1][first]


def _chord_map(
    m: np.ndarray, a: np.ndarray, b: np.ndarray, n_nodes: int
) -> sp.csr_array:
    """The sparse ``(3 n, 3 n)`` map from node displacements to chord means.

    Row ``3 m + i`` of a mid-edge node ``m`` on the edge ``a``-``b`` holds
    :math:`1/2` at columns ``3 a + i`` and ``3 b + i``; all other rows are zero.
    """
    dof = np.arange(3)
    return sp.csr_array(
        (
            np.full(6 * len(m), 0.5),
            (
                np.tile((3 * m[:, None] + dof).ravel(), 2),
                np.concatenate(
                    [(3 * a[:, None] + dof).ravel(), (3 * b[:, None] + dof).ravel()]
                ),
            ),
        ),
        shape=(3 * n_nodes, 3 * n_nodes),
    )


def _midnode_maps(
    model: Model,
    *,
    moving_tri: np.ndarray,
    params: np.ndarray,
    ties: np.ndarray,
    normals: np.ndarray,
    mid_nodes: np.ndarray,
    mid_normals: np.ndarray,
    held: np.ndarray,
) -> tuple[sp.csr_array, sp.csr_array]:
    r"""The maps that place the mid-edge nodes of a curved mesh after the corners.

    With ``u`` the corners' displacement (zero at mid-edge nodes), the mid-edge
    displacement is ``chord @ u + bubble @ t``: every mid-edge node follows its
    chord, :math:`u_m = (u_a + u_b) / 2`, and a node of the moving wall whose
    edge has both corners moving, and lies in no held or gated face, instead
    moves :math:`(t_a + t_b)\,n_m / 2` along its own normal. ``bubble`` is
    the difference of the two, per parameter.

    Returns:
        ``(chord, bubble)``, sparse ``(3 n, 3 n)`` and ``(3 n, n_params)``.
    """
    mesh = model.mesh
    topology = mesh._topology  # noqa: SLF001
    n_nodes = len(mesh.imported_nodes)
    dof = np.arange(3)
    m, a, b = _edge_ends(topology)
    chord = _chord_map(m, a, b, n_nodes)

    # the mid-edge nodes of the moving wall that take their own normal: both
    # corners moving, in no held or gated face
    wall, wall_mids = mesh.triangles[moving_tri], topology.surface_midnodes[moving_tri]
    gated = held.copy()
    gated[topology.surface_midnodes[gated_triangles(model)].ravel()] = True
    is_param = np.zeros(n_nodes, dtype=bool)
    is_param[params] = True
    has_normal = np.zeros(n_nodes, dtype=bool)
    has_normal[mid_nodes] = True
    normal_of = np.zeros((n_nodes, 3))
    normal_of[mid_nodes] = mid_normals
    column = np.full(n_nodes, -1)
    column[params] = np.arange(len(params))
    edges = [
        (wall_mids[:, k], wall[:, i], wall[:, j]) for k, (i, j) in enumerate(_TRI_EDGES)
    ]
    node = np.concatenate([e[0] for e in edges])
    end_a = np.concatenate([e[1] for e in edges])
    end_b = np.concatenate([e[2] for e in edges])
    moves = (
        is_param[ties[end_a]] & is_param[ties[end_b]] & has_normal[node] & ~gated[node]
    )
    node, end_a, end_b = node[moves], end_a[moves], end_b[moves]
    node, first = np.unique(node, return_index=True)  # an edge of two wall faces
    rows, cols, vals = [], [], []
    for end in (end_a[first], end_b[first]):
        col = column[ties[end]]
        rows.append((3 * node[:, None] + dof).ravel())
        cols.append(np.repeat(col, 3))
        vals.append((0.5 * (normal_of[node] - normals[col])).ravel())
    bubble = sp.csr_array(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * n_nodes, len(params)),
    )
    return chord, bubble


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

    topology = mesh._topology  # noqa: SLF001
    # a curved mesh holds the mid-edge nodes of its held elements too
    surface_mids, volume_mids = topology.surface_midnodes, topology.volume_midnodes

    def with_mids(corners: np.ndarray, mids: np.ndarray | None) -> np.ndarray:
        return (
            corners.ravel()
            if mids is None
            else np.concatenate([corners, mids], axis=1).ravel()
        )

    for s, why in holder.items():
        on = mesh.triangle_tags == tag[s]
        label(
            with_mids(
                mesh.triangles[on], None if surface_mids is None else surface_mids[on]
            ),
            why,
        )
    layer = _pml.layer_tetrahedra(mesh, model.pml)
    if layer is not None:
        label(
            with_mids(
                mesh.tetrahedra[layer],
                None if volume_mids is None else volume_mids[layer],
            ),
            "the perfectly matched layer",
        )
    truncation = _truncation.truncation(model)
    label(
        with_mids(
            mesh.tetrahedra[truncation.layer],
            None if volume_mids is None else volume_mids[truncation.layer],
        ),
        "the truncation layer",
    )
    # a class is held, with one cause, if any member is
    named = np.flatnonzero(cause != "")
    by_class = np.full(n_nodes, "", dtype=object)
    classes, first = np.unique(ties[named], return_index=True)
    by_class[classes] = cause[named[first]]  # the first member's cause
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


def _quadratic_normals(x: np.ndarray, tris: np.ndarray, mids: np.ndarray) -> np.ndarray:
    r"""Twice the area times the unit normal of curved triangles at their six nodes.

    The quadratic map of a triangle with corners :math:`x_i` and mid-edge nodes
    :math:`x_{ab}` is :math:`x = \sum_i \lambda_i (2 \lambda_i - 1) x_i +
    \sum_{ab} 4 \lambda_a \lambda_b x_{ab}`. At each node the vector is
    :math:`\partial_1 x \times \partial_2 x`, the tangents being the
    derivatives along the edges from corner 0 (`_lattice_gradients`'s
    construction), which on a straight triangle is :math:`(b - a) \times (c - a)`.

    Args:
        x: Node coordinates.
        tris: Corner indices, ``(n, 3)``.
        mids: Mid-edge node indices, ``(n, 3)``, column ``k`` on the edge
            opposite corner ``k``.

    Returns:
        ``(n, 6, 3)``: the three corners, then the three mid-edge nodes.
    """
    points = np.concatenate([np.eye(3), 0.5 * (1 - np.eye(3))])  # barycentric
    g = np.zeros((6, 3, 6))  # [point, coordinate, node]
    for i in range(3):
        g[:, i, i] = 4 * points[:, i] - 1
    for opposite, (a, b) in enumerate(_TRI_EDGES):
        g[:, a, 3 + opposite] = 4 * points[:, b]
        g[:, b, 3 + opposite] = 4 * points[:, a]
    d = np.einsum("pin,enx->epix", g, x[np.concatenate([tris, mids], axis=1)])
    return np.cross(d[:, :, 1] - d[:, :, 0], d[:, :, 2] - d[:, :, 0])


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
    half-stars of twins form the full star). On a curved mesh the normals come
    from the quadratic triangles, at the corners and at the mid-edge nodes of
    the faces fully inside the selection.

    Returns:
        ``(normals, mid_nodes, mid_normals)``: the parameters' unit normals,
        and on a curved mesh the mid-edge nodes that have one, with theirs
        (empty on a straight mesh).
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
    if mesh.curved:
        # the quadratic triangles' normals at their six nodes
        mids = mesh._topology.surface_midnodes[moving_tri]  # noqa: SLF001
        weighted = sign[:, None, None] * _quadratic_normals(x0, tris, mids)
        ids = np.concatenate([tris, mids], axis=1)
    else:
        a, b, c = (x0[tris[:, k]] for k in range(3))
        weighted = sign[:, None, None] * np.cross(b - a, c - a)[:, None]  # 2 A n
        ids = tris
    total = np.zeros((2, len(x0), 3))
    area = np.zeros((2, len(x0)))
    for k in range(ids.shape[1]):
        w = weighted[:, k if mesh.curved else 0]
        for slot, faces in enumerate((full, np.ones_like(full))):
            np.add.at(total[slot], ids[faces, k], w[faces])
            np.add.at(area[slot], ids[faces, k], np.linalg.norm(w[faces], axis=1))
    # mid-edge nodes: the faces fully inside the selection
    mid_nodes = np.unique(ids[:, 3:]) if mesh.curved else np.empty(0, dtype=np.int64)
    mid_total, mid_area = total[0, mid_nodes], area[0, mid_nodes]
    mid_length = np.linalg.norm(mid_total, axis=1)
    mid_ok = mid_length > 1e-8 * np.maximum(mid_area, np.finfo(float).tiny)
    mid_nodes = mid_nodes[mid_ok]
    mid_normals = mid_total[mid_ok] / mid_length[mid_ok, None]
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
    return total / length[:, None], mid_nodes, mid_normals


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


# --- CAD morph ----------------------------------------------------------------

#: A velocity below this fraction of its parameter's largest counts as zero
#: where the design must not move a node (a held node, a sliding port's
#: normal, a periodic twin). A node whose entity does not move projects to
#: itself to round-off, about eps / h, far below it; the 2e-6 relative error of
#: a closest point on an approximated curve scales with the motion, so it
#: cannot appear on an entity that does not move.
_STILL = 1e-6

#: A parameter whose largest velocity times its step is below this fraction
#: of the model's size moves nothing: its morph gradient would be zero while
#: remeshing gives mesh noise (a mesh-size parameter, say). A real motion is
#: at least the kernel's own tolerance, 1e-7 of a unit-size model; the closest
#: point of an entity that does not move differs by round-off, 1e-15 of the
#: size, so 1e-12 separates the two.
_DEAD = 1e-12


def _check_cad_arguments(
    *,
    build: Any,
    p0: Any,
    make_model: Any,
    length_unit: str,
    step: Any,
    sliding_ports: Any,
    extension: str,
) -> tuple[dict[str, float], dict[str, float], tuple[str, ...]]:
    """The checked ``p0``, ``step`` and ``sliding_ports`` of `cad_morph`."""
    if not callable(build):
        msg = f"build must be a function of the parameters, got {build!r}."
        raise TypeError(msg)
    if not callable(make_model):
        msg = (
            f"make_model must be a function from a Mesh to a Model, got {make_model!r}."
        )
        raise TypeError(msg)
    if isinstance(sliding_ports, str) or not isinstance(sliding_ports, (list, tuple)):
        msg = (
            f"sliding_ports= takes a list or tuple of wave port names, got "
            f"{sliding_ports!r}."
        )
        raise TypeError(msg)
    _check_unit(length_unit)
    if extension not in _EXTENSIONS:
        msg = f"Unknown extension {extension!r}; use 'elastic' or 'laplace'."
        raise ValueError(msg)
    if not isinstance(p0, Mapping) or len(p0) == 0:
        msg = (
            f"p0 must be a non-empty mapping from parameter name to value, got {p0!r}."
        )
        raise ValueError(msg)
    values = {}
    for name, value in p0.items():
        if not isinstance(name, str) or not name.isidentifier():
            msg = (
                f"Parameter names must be Python identifiers (build is called as "
                f"build(**params)), got {name!r}."
            )
            raise ValueError(msg)
        if is_traced(value):
            msg = (
                f"p0[{name!r}] is traced by JAX: the CAD model is built from "
                "concrete values; call cad_morph outside jax transformations."
            )
            raise TypeError(msg)
        values[name] = float(value)
        if not math.isfinite(values[name]):
            msg = f"p0[{name!r}] = {value!r} is not finite."
            raise ValueError(msg)
    steps = {}
    for name, h in (step or {}).items():
        if name not in values:
            msg = (
                f"step names {name!r}, which is not a parameter. "
                f"Known: {_names(values)}."
            )
            raise ValueError(msg)
        steps[name] = float(h)
        if not (math.isfinite(steps[name]) and steps[name] > 0):
            msg = f"step[{name!r}] = {h!r} must be a positive, finite step."
            raise ValueError(msg)
    return values, steps, tuple(sliding_ports)


def _names(values: Any) -> str:
    return ", ".join(repr(n) for n in values)


def _make_cad_morph(
    build: Callable[..., object],
    p0: Mapping[str, float],
    make_model: Callable[[Mesh], Model],
    *,
    length_unit: str,
    step: Mapping[str, float] | None,
    sliding_ports: Sequence[str],
    extension: Extension,
    curved: bool = False,
) -> CadMorph:
    """`cad_morph` and `CadMorph.remesh`: refuse ranks, check, mesh, extend."""
    # Before anything imports gmsh: it is not distributed.
    if _mpi.resolve_comm(None) is not None:
        _mpi.refuse_feature("cad_morph")
    values, steps, ports = _check_cad_arguments(
        build=build,
        p0=p0,
        make_model=make_model,
        length_unit=length_unit,
        step=step,
        sliding_ports=sliding_ports,
        extension=extension,
    )
    if not isinstance(curved, (bool, np.bool_)):
        msg = f"curved must be True or False, got {curved!r}."
        raise TypeError(msg)
    curved = bool(curved)
    if curved:
        cad = _cad.build_cad(build, values, length_unit, steps, curved=True)
    else:
        cad = _cad.build_cad(build, values, length_unit, steps)
    try:
        model = make_model(cad.mesh)
    except Exception as exc:
        exc.add_note(f"gfem.shape.cad_morph called make_model on the mesh at {values}")
        raise
    if not isinstance(model, Model):
        msg = (
            f"make_model must return a gfem.Model, got {type(model).__name__}. "
            "Build it on the mesh it is given: gfem.Model(mesh, materials=..., "
            "...)."
        )
        raise TypeError(msg)
    if not np.array_equal(model.mesh.imported_nodes, cad.mesh.imported_nodes):
        msg = (
            "make_model must build its model on the mesh it is given (the "
            "design velocities belong to its nodes); the model's mesh has other "
            "nodes."
        )
        raise ValueError(msg)
    w = _extend(model, cad, tuple(values), ports, extension)
    return CadMorph(
        model=model,
        names=tuple(values),
        p0=tuple(values.values()),
        displacement=w,
        length_unit=length_unit,
        extension=extension,
        active=np.asarray(model._domain.active),  # noqa: SLF001
        remesh_with=(build, make_model, dict(steps), ports),
        curved=curved,
    )


def _extend(
    model: Model,
    cad: _cad.CadMesh,
    names: tuple[str, ...],
    sliding_ports: tuple[str, ...],
    extension: Extension,
) -> np.ndarray:
    """``W``, ``(3 n, n_p)``, metres per parameter unit, after the design checks."""
    mesh = model.mesh
    x0 = np.asarray(mesh.imported_nodes)
    n_nodes, n_params = len(x0), len(names)
    scale = cad.scale
    pairings = model._periodic  # noqa: SLF001
    ties = _periodic.tie_classes(n_nodes, pairings)
    sliding = _sliding_surfaces(model, sliding_ports)
    held, _, cause = _held(model, ties, sliding)
    # The extension is solved on the corners: a mid-edge node of a curved mesh
    # is no unknown of it (it is placed after it), though it is in an element.
    corner_in_tet = np.zeros(n_nodes, dtype=bool)
    corner_in_tet[mesh.tetrahedra.ravel()] = True
    in_tet = corner_in_tet.copy()
    if mesh.curved:
        in_tet[mesh._topology.volume_midnodes.ravel()] = True  # noqa: SLF001
    velocity = cad.velocity.reshape(n_nodes, 3, n_params)
    speed = np.linalg.norm(velocity, axis=1)  # (n_nodes, n_params)
    top = speed.max(axis=0)
    unit = mesh.length_unit
    where = x0 / scale
    for k, name in enumerate(names):
        if top[k] * cad.steps[k] <= _DEAD * cad.size:
            msg = (
                f"Parameter {name!r} moves nothing: its largest design velocity "
                f"is {top[k]:.3g} {unit} per unit, so a step of {cad.steps[k]:.3g} "
                f"moves the model by less than {_DEAD:g} of its size. Its "
                "gradient would be zero (a mesh-size parameter, say). Remove it "
                "from p0, or change what the build draws with it."
            )
            raise ValueError(msg)
    tolerance = _STILL * top
    _check_held(
        velocity=velocity,
        speed=speed,
        tolerance=tolerance,
        held=held | ~in_tet,
        cause=np.where(in_tet, cause, "belonging to no element"),
        where=where,
        names=names,
        unit=unit,
    )
    _check_sliding(
        model=model,
        velocity=velocity,
        tolerance=tolerance,
        sliding_ports=sliding_ports,
        names=names,
        unit=unit,
    )
    _check_periodic(
        velocity=velocity,
        tolerance=tolerance,
        ties=ties,
        where=where,
        names=names,
        unit=unit,
    )
    members = np.flatnonzero(ties != np.arange(n_nodes))
    velocity[members] = velocity[ties[members]]
    velocity[held | ~in_tet] = 0.0
    if mesh.curved:
        # the mid-edge design velocities enter after the corner solve
        mid_velocity = velocity.copy()
        velocity[~corner_in_tet] = 0.0
    x_model = x0 / scale
    # A CAD vertex has no freedom, and neither has a curve node whose
    # incident surface normals are coplanar (a patch edge on a flat face): the
    # surface leaves it free in its plane, where the energy minimisation would
    # cancel the design velocity. Its closest-point tangential motion along
    # the curve is ~0 to first order, so W = V is the conservative choice.
    free = _morph.free_directions(
        n_nodes,
        cad.slide_triangles,
        _morph.unit_normals(x_model, cad.slide_triangles),
        held | ~corner_in_tet | (cad.dim == 0),
        ties,
        curves=cad.dim == 1,
    )
    stiffness = _morph.stiffness(x_model, mesh.tetrahedra, extension)
    w = _morph.vector_extension(
        stiffness, free, velocity.reshape(3 * n_nodes, n_params)
    )
    if mesh.curved:
        w = _place_cad_midnodes(
            model, cad, w, mid_velocity, held=held, x_model=x_model, n_params=n_params
        )
    return w * scale


def _place_cad_midnodes(
    model: Model,
    cad: _cad.CadMesh,
    w: np.ndarray,
    velocity: np.ndarray,
    *,
    held: np.ndarray,
    x_model: np.ndarray,
    n_params: int,
) -> np.ndarray:
    r"""``w`` with the mid-edge nodes of a curved mesh placed after the corner solve.

    Every mid-edge node follows its chord, :math:`W_m = (W_a + W_b)/2`, except
    a node classified on the CAD (a surface or a curve) in no held or gated
    face, which takes its own design velocity :math:`V_m` in the directions the
    CAD fixes: :math:`W_m = P V_m + (I - P)\,\tfrac12 (W_a + W_b)`. On a surface
    :math:`P = n n^\mathsf{T}` with :math:`n` the normal of the quadratic
    triangle at the node; on a curve :math:`P = I - \hat e \hat e^\mathsf{T}`
    with :math:`\hat e` the chord direction, which is the tangent of the
    quadratic edge at its midpoint. The corners slide along the CAD, so only
    the part across it is the CAD's.

    Args:
        model: The model, on the curved mesh.
        cad: The meshed CAD model.
        w: ``(3 n, n_params)`` corner displacements (zero at mid-edge nodes).
        velocity: ``(n, 3, n_params)`` design velocities with the held ones
            zeroed.
        held: Nodes the morph holds.
        x_model: Node coordinates, model units.
        n_params: The number of parameters.

    Returns:
        ``w`` with the mid-edge rows filled.
    """
    mesh = model.mesh
    topology = mesh._topology  # noqa: SLF001
    n_nodes = len(x_model)
    m, a, b = _edge_ends(topology)
    chord = (_chord_map(m, a, b, n_nodes) @ w).reshape(n_nodes, 3, n_params)
    w = w.reshape(n_nodes, 3, n_params).copy()
    w[m] = chord[m]
    gated = np.zeros(n_nodes, dtype=bool)
    gated[topology.surface_midnodes[gated_triangles(model)].ravel()] = True
    own = (cad.dim[m] <= 2) & ~held[m] & ~gated[m]  # classified on the CAD
    # surface nodes: the normal of the quadratic triangle at the mid-edge node
    vectors = _quadratic_normals(x_model, cad.slide_triangles, cad.slide_midnodes)
    ids = cad.slide_midnodes.ravel()
    nodes, first = np.unique(ids, return_index=True)
    vector = vectors[:, 3:].reshape(-1, 3)[first]
    length = np.linalg.norm(vector, axis=1)
    normal = np.zeros((n_nodes, 3))
    normal[nodes] = vector / np.maximum(length, np.finfo(float).tiny)[:, None]
    # curve nodes: the chord direction
    chord_direction = x_model[b] - x_model[a]
    tangent = np.zeros((n_nodes, 3))
    tangent[m] = (
        chord_direction
        / np.maximum(np.linalg.norm(chord_direction, axis=1), np.finfo(float).tiny)[
            :, None
        ]
    )
    for dim, direction in ((2, normal), (1, tangent)):
        sel = m[own & (cad.dim[m] == dim)]
        d = direction[sel][:, :, None]
        c, v = chord[sel], velocity[sel]
        if dim == 2:  # P = n n^T: the CAD's part is across the surface
            w[sel] = c + d * np.sum(d * (v - c), axis=1, keepdims=True)
        else:  # P = I - e e^T: the CAD's part is across the curve
            w[sel] = v + d * np.sum(d * (c - v), axis=1, keepdims=True)
    return w.reshape(3 * n_nodes, n_params)


def _check_held(
    *,
    velocity: np.ndarray,
    speed: np.ndarray,
    tolerance: np.ndarray,
    held: np.ndarray,
    cause: np.ndarray,
    where: np.ndarray,
    names: tuple[str, ...],
    unit: str,
) -> None:
    """Refuse a parameter that moves a node the morph holds."""
    for k, name in enumerate(names):
        moved = np.where(held, speed[:, k], 0.0)
        node = int(np.argmax(moved))
        if moved[node] > tolerance[k]:
            msg = (
                f"Parameter {name!r} moves a node held by {cause[node]}: "
                f"at ({_xyz(where[node])}) {unit} the design velocity is "
                f"({_xyz(velocity[node, :, k])}) {unit} per unit, the largest of "
                f"{int((moved > tolerance[k]).sum())} held node(s) it moves. "
                "A held node stays where the imported mesh has it, so the morph "
                "would carry out a different design than the CAD. Keep the "
                "feature off the layer or the port; keep the PML and truncation "
                "geometry independent of the parameter; for a wave port, name it "
                "in sliding_ports=."
            )
            raise ValueError(msg)


def _check_sliding(
    *,
    model: Model,
    velocity: np.ndarray,
    tolerance: np.ndarray,
    sliding_ports: tuple[str, ...],
    names: tuple[str, ...],
    unit: str,
) -> None:
    """Refuse a parameter that moves the reference plane of a sliding port."""
    mesh = model.mesh
    x0 = np.asarray(mesh.imported_nodes)
    ports = {p.name: p for p in model.ports}
    tag = {r.name: r.tag for r in mesh.regions if r.dim == mesh.dim - 1}
    for port_name in sliding_ports:
        tags = [tag[s] for s in surfaces_of(ports[port_name])]
        on = np.isin(mesh.triangle_tags, tags)
        faces = mesh.triangles[on]
        normals = _morph.unit_normals(x0, faces)
        if mesh.curved:  # the mid-edge nodes of the face are checked as well
            faces = np.concatenate([faces, mesh._topology.surface_midnodes[on]], axis=1)  # noqa: SLF001
        along = np.abs(np.einsum("td,tcdk->tck", normals, velocity[faces]))
        for k, name in enumerate(names):
            worst = int(np.argmax(along[:, :, k].max(axis=1)))
            corner = int(np.argmax(along[worst, :, k]))
            if along[worst, corner, k] > tolerance[k]:
                node = faces[worst, corner]
                msg = (
                    f"Parameter {name!r} moves the plane of sliding port "
                    f"{port_name!r}: at node ({_xyz(x0[node] / LENGTH_UNITS[unit])}) "
                    f"{unit} its normal velocity is {along[worst, corner, k]:.3g} "
                    f"{unit} per unit. A port keeps its reference plane where it "
                    "is declared. Keep the parameter from moving that face (move "
                    "the surfaces that meet it), or remove the port from "
                    "sliding_ports= to hold it, which refuses any motion at it."
                )
                raise ValueError(msg)


def _check_periodic(
    *,
    velocity: np.ndarray,
    tolerance: np.ndarray,
    ties: np.ndarray,
    where: np.ndarray,
    names: tuple[str, ...],
    unit: str,
) -> None:
    """Refuse a parameter whose velocity differs between tied nodes."""
    twins = np.flatnonzero(ties != np.arange(len(ties)))
    if not len(twins):
        return
    gap = np.linalg.norm(velocity[twins] - velocity[ties[twins]], axis=1)
    for k, name in enumerate(names):
        i = int(np.argmax(gap[:, k]))
        if gap[i, k] > tolerance[k]:
            a, b = twins[i], ties[twins[i]]
            msg = (
                f"Parameter {name!r} breaks the periodicity or changes the "
                f"period, which is static: the paired nodes at ({_xyz(where[b])}) "
                f"and ({_xyz(where[a])}) {unit} have design velocities "
                f"({_xyz(velocity[b, :, k])}) and ({_xyz(velocity[a, :, k])}) "
                f"{unit} per unit. Keep the parameter off the paired faces and "
                "off the cell's dimensions, or move both faces of a pair alike."
            )
            raise ValueError(msg)


def _xyz(v: np.ndarray) -> str:
    return ", ".join(f"{c:.4g}" for c in v)


class CadMorph:
    r"""A parametric CAD model, meshed, as a linear map from parameters to meshes.

    Build it with `cad_morph`. Calling it with a dict of parameter values
    returns the moved `Mesh`::

        moved = morph({"width": 0.55})
        model = morph.model.replace(mesh=moved)

    The map is :math:`x(p) = x_0 + W (p - p_0)`: the velocity of every node
    with respect to every parameter, :math:`W`, is solved once at the
    parameters the morph was built at, so the same ``p`` always gives the same
    mesh. It is linear in ``p`` and differentiable with JAX (``grad``, ``jit``,
    ``vmap``, ``jvp``): `jax.grad` of an objective returns a dict with the
    parameters' names, each a derivative per parameter unit.

    The geometry follows the CAD only to first order in :math:`p - p_0`, and
    the mesh keeps its elements, which distort. When a design moves too far
    (`gfem.shape.quality`, or the quality warnings), restart at the current
    values with `CadMorph.remesh`.

    Attributes:
        names: The parameter names, in the order of ``p0`` (the columns of
            :math:`W`).
        n_params: The number of parameters.
        model: The model `cad_morph`'s ``make_model`` made on the mesh at the
            first parameters.
        velocities: A read-only array ``(n_nodes, 3)`` per parameter name;
            what the parameter moves, in metres per parameter unit (a column of
            :math:`W`).
        length_unit: The unit of the build's lengths (and of length
            parameters).
        extension: ``"elastic"`` or ``"laplace"``.
        curved: Whether the mesh has second-order elements (``cad_morph``'s
            ``curved``); `remesh` keeps it.
    """

    names: tuple[str, ...]
    n_params: int
    model: Model
    length_unit: str
    extension: Extension
    curved: bool
    _p0: tuple[float, ...]
    _treedef: Any
    _x0: np.ndarray
    _w: np.ndarray
    _active: np.ndarray
    _columns: dict[str, np.ndarray]
    _build: Callable[..., object]
    _make_model: Callable[[Mesh], Model]
    _step: dict[str, float]
    _sliding_ports: tuple[str, ...]

    def __init__(
        self,
        *,
        model: Model,
        names: tuple[str, ...],
        p0: tuple[float, ...],
        displacement: np.ndarray,
        length_unit: str,
        extension: Extension,
        active: np.ndarray,
        remesh_with: tuple[Callable[..., object], Callable[[Mesh], Model], dict, tuple],
        curved: bool = False,
    ) -> None:
        """Internal: use `cad_morph`."""
        mesh = model.mesh
        x0 = np.asarray(mesh.imported_nodes)
        build, make_model, step, sliding_ports = remesh_with
        values = {
            "names": names,
            "n_params": len(names),
            "model": model,
            "length_unit": length_unit,
            "extension": extension,
            "curved": curved,
            "_p0": p0,
            "_treedef": jax.tree_util.tree_structure(mesh),
            "_x0": x0,
            "_w": readonly(displacement),
            "_active": active,
            "_columns": {
                name: readonly(displacement[:, k].reshape(-1, 3))
                for k, name in enumerate(names)
            },
            "_build": build,
            "_make_model": make_model,
            "_step": step,
            "_sliding_ports": sliding_ports,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def p0(self) -> dict[str, float]:
        """The parameter values the morph was built at, a new dict on each access."""
        return dict(zip(self.names, self._p0, strict=True))

    @property
    def velocities(self) -> dict[str, np.ndarray]:
        """Read-only motion of the mesh per parameter, metres per parameter unit.

        This is the extended field :math:`W` of every node, not only the design
        velocities: its values on the boundary entities are the design
        velocities, and the volume nodes follow them.
        """
        return dict(self._columns)

    def _vector(self, p: Mapping[str, Any]) -> list[Any]:
        """The parameter values of ``p``, the missing ones at ``p0``."""
        if not isinstance(p, Mapping):
            msg = (
                "A CAD morph takes its parameters as a dict from name to value, "
                f"got {type(p).__name__}. Known names: {_names(self.names)}."
            )
            raise TypeError(msg)
        unknown = [k for k in p if k not in self.names]
        if unknown:
            msg = (
                f"Unknown parameter(s) {_names(unknown)}. Known names: "
                f"{_names(self.names)}."
            )
            raise ValueError(msg)
        return [p.get(n, v) for n, v in zip(self.names, self._p0, strict=True)]

    def __call__(self, p: Mapping[str, Any]) -> Mesh:
        r"""The mesh at ``p``: :math:`x_0 + W (p - p_0)`.

        With concrete values (not under a JAX transformation), the moved
        mesh's quality is checked at once; under ``jit`` or ``grad`` the check
        runs when the mesh is solved on.

        Args:
            p: Parameter values by name, in the build's units. Names left out
                keep their first value.

        Returns:
            The moved mesh, which shares the imported mesh's topology.

        Raises:
            ValueError: A name is not a parameter, or a value is not a scalar.
            MeshQualityError: An element of the solve domain is inverted.

        Warns:
            MeshQualityWarning: An element fell below a quarter of its
                imported quality.
        """
        values = self._vector(p)
        for name, value in zip(self.names, values, strict=True):
            if jnp.ndim(value) != 0:
                msg = (
                    f"Parameter {name!r} must be a scalar, got an array of "
                    f"shape {jnp.shape(value)}."
                )
                raise ValueError(msg)
        t = jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in values])
        delta = t - jnp.asarray(self._p0, dtype=jnp.float64)
        nodes = jnp.asarray(self._x0) + (jnp.asarray(self._w) @ delta).reshape(
            self._x0.shape
        )
        mesh = jax.tree_util.tree_unflatten(self._treedef, [nodes])
        if not is_traced(t):
            _check_quality(mesh, self._active, stacklevel=2)
        return mesh

    def remesh(self, p: Mapping[str, float]) -> CadMorph:
        """Rebuild, remesh and re-linearize the CAD model at ``p``.

        A new morph from the same build function, ``make_model``, units,
        steps, sliding ports and extension; its ``p0`` is ``p`` (absolute, so
        parameters need no offset bookkeeping). Rebuild the objective on the
        new morph's `model`, reset an optimizer's history, and compare
        objective values only within one mesh.

        Args:
            p: Parameter values by name, concrete. Names left out keep this
                morph's ``p0``.

        Returns:
            The new morph.

        Raises:
            TypeError: ``p`` is traced by JAX: remeshing rebuilds the CAD model.
            ValueError: A name is not a parameter, or the rebuild is refused as
                `cad_morph` refuses it.
        """
        if is_traced(p):
            msg = (
                "remesh rebuilds the CAD model; call it outside jax "
                "transformations with concrete values"
            )
            raise TypeError(msg)
        values = dict(zip(self.names, self._vector(p), strict=True))
        return _make_cad_morph(
            self._build,
            values,
            self._make_model,
            length_unit=self.length_unit,
            step=self._step,
            sliding_ports=self._sliding_ports,
            extension=self.extension,
            curved=self.curved,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        msg = (
            "CadMorph is immutable; build a new one with gfem.shape.cad_morph() "
            "or CadMorph.remesh()"
        )
        raise AttributeError(msg)

    def __repr__(self) -> str:
        build = getattr(self._build, "__name__", repr(self._build))
        return (
            f"CadMorph(build={build}, p0={self.p0!r}, "
            f"length_unit={self.length_unit!r}, extension={self.extension!r}"
            f"{', curved=True' if self.curved else ''})"
        )


jax.tree_util.register_static(CadMorph)


def cad_morph(
    build: Callable[..., object],
    p0: Mapping[str, float],
    make_model: Callable[[Mesh], Model],
    *,
    length_unit: str = "um",
    step: Mapping[str, float] | None = None,
    sliding_ports: Sequence[str] = (),
    extension: Extension = "elastic",
    curved: bool = False,
) -> CadMorph:
    r"""Mesh a parametric gmsh model and morph its mesh with its CAD parameters.

    Builds the CAD model at ``p0`` with ``build(**p0)``, meshes it, reads the
    mesh (metres, in memory), makes the model with ``make_model(mesh)``, and
    takes each parameter's design velocity by rebuilding the geometry at
    :math:`p_0 \pm h`. The mesh follows by the mesh extension. The returned
    morph maps parameter values to moved meshes linearly, fixed at ``p0``:
    :math:`x(p) = x_0 + W (p - p_0)`. ::

        def build(width, height):  # micrometres
            occ = gmsh.model.occ
            ...
            occ.synchronize()
            gmsh.model.addPhysicalGroup(3, [core], name="core")


        morph = gfem.shape.cad_morph(build, {"width": 0.5, "height": 0.3}, make_model)
        grad = jax.grad(lambda p: loss(morph.model.replace(mesh=morph(p))))(morph.p0)

    The build function's contract:

    - gfem calls ``build(**params)`` with Python floats, in an empty current
      gmsh model that it created. The build draws with the OCC kernel
      (``gmsh.model.occ``; the built-in geometry kernel is not supported and
      not detected), calls ``synchronize()``, and returns nothing.
    - It names physical groups with ``addPhysicalGroup(dim, tags, name=...)``:
      3D groups become the volume regions and 2D groups the surface regions
      ``make_model`` refers to (lower dimensions are ignored). Every 3D entity
      must be in a volume group.
    - It sets its own mesh sizes (options, fields, ``setSize``): they are
      physics the user owns. Options it sets stay set in a gmsh session the
      user opened.
    - It never calls ``initialize``, ``finalize``, ``model.add``,
      ``model.remove``, ``model.setCurrent``, ``mesh.generate`` or ``write``:
      gfem meshes at ``p0`` only, and only rebuilds the geometry at
      :math:`p_0 \pm h`.
    - It is deterministic: equal parameters give equal entities and tags.
    - A model with `gfem.Periodic` pairs needs
      ``gmsh.model.mesh.setPeriodic(2, ...)`` on each pair of faces in the
      build, so the two faces mesh alike.
    - Parameters are lengths in ``length_unit``; give a parameter of another
      kind (an angle) a ``step`` in its own unit.

    Each parameter's velocity is that of the mesh nodes on points, curves and
    surfaces, taken as the central difference of their closest points on the
    same entity of the model rebuilt at :math:`p_0 \pm h` (default
    :math:`h = 2^{\operatorname{round}(\log_2(\varepsilon^{1/3} L))}`, with
    :math:`L` the model's bounding-box diagonal). Nodes in a volume get none.
    The extension carries it into the volume: the velocity is prescribed in
    the directions the CAD determines (the normal of a face, the two normals of
    an edge, all of a corner), and a node slides freely in the directions the
    CAD leaves open, as :math:`W = V + B z` with
    :math:`B^\top K B\, z = -B^\top K V` (:math:`K` the stiffened extension).
    The nodes of a perfectly matched layer, of the elements touching the
    truncation, of port surfaces (except wave ports named in
    ``sliding_ports``) and in no element are held; the CAD must not move them.
    ``W`` is a dense array of :math:`24\, n_\mathrm{nodes}\, n_\mathrm{params}` bytes.

    gmsh is an optional dependency (``pip install 'gfem[gmsh]'``). gfem uses
    the gmsh session the user opened, adding and removing models of its own and
    restoring the current one, or opens a quiet one that reads no
    configuration file and finalizes it. The velocity of an edge that gmsh
    approximates by a spline (the intersection of two curved surfaces) is
    accurate to about two parts in a million.

    With ``curved=True`` the mesh has second-order elements: gfem sets gmsh's
    ``Mesh.ElementOrder 2`` and ``Mesh.HighOrderOptimize 1`` (a level of 2 to 4
    set by the build is kept) around the meshing and restores them. gmsh puts
    each boundary mid-edge node on the CAD, so the mesh follows the curved
    walls and the gradient with it: on a straight mesh the morph moves a
    polygon and its gradient is the polygon's derivative, which falls short of
    the CAD's by about :math:`0.18\,(\ell/R)^2` at a wall of radius :math:`R`
    meshed at size :math:`\ell`; on a curved mesh it follows the CAD to the
    discretization error. A mid-edge node on a CAD surface or curve takes the
    design velocity :math:`V_m` across it and its chord's mean along it,
    :math:`W_m = P V_m + (I - P)\,\tfrac12 (W_a + W_b)` (:math:`P` the
    projector on the surface normal, or across the curve), and every other
    mid-edge node (in a volume, on a port, periodic or sheet face, or held)
    follows its chord, :math:`W_m = \tfrac12 (W_a + W_b)`. Features that read
    straight elements only still refuse a curved mesh at the solve. The mesh
    has about seven times the nodes of the straight one, and so does the dense
    ``W``.

    Args:
        build: The function of named parameters that draws the model. It must
            not set ``Mesh.ElementOrder``: gfem sets it, with ``curved``.
        p0: The parameter values to mesh at, by name. The order fixes
            :math:`W`'s columns. Names are Python identifiers.
        make_model: Makes the `Model` (materials, boundaries, ports) from the
            mesh. Its declarations do not depend on the parameters except
            through the mesh.
        length_unit: The unit of the build's coordinates, one of
            `gfem.mesh.LENGTH_UNITS`.
        step: Half-steps :math:`h` per parameter in its own unit, overriding
            the default; names left out keep it.
        sliding_ports: Wave ports whose faces slide in their plane instead of
            being held, as in `boundary_morph`. Lumped and Floquet ports stay
            held.
        extension: ``"elastic"`` (default) or ``"laplace"``, as in
            `boundary_morph`.
        curved: Mesh with second-order elements (see above); `CadMorph.remesh`
            keeps it. The default is straight.

    Returns:
        The morph: call it with parameter values to get the moved mesh.

    Raises:
        ImportError: gmsh is not installed.
        OSError: gmsh's library cannot load (the PyPI wheel needs system
            OpenGL and X11 libraries).
        ValueError: A name is not an identifier, a value or step is not
            finite or positive, the unit or extension is unknown; the build
            made no 3D entity, or left one out of every volume group; the
            topology of the build changes within :math:`p_0 \pm h` (use a
            smaller ``step``), or a node jumps there; the build sets
            ``Mesh.ElementOrder`` (use ``curved=True``); gmsh's nodes do not map
            onto the mesh read; a parameter moves nothing; moves a node a
            perfectly matched layer, the truncation or a port holds (keep the
            feature off it; for a wave port, name it in ``sliding_ports``);
            moves the plane of a sliding port; or breaks a periodic pair or
            changes the period.
        TypeError: ``make_model`` returns no `Model`, ``sliding_ports`` is a
            string, or ``curved`` is not a bool.
        DistributedUnsupportedError: Launched above one process: the CAD model
            is built with gmsh, which is not distributed.
    """
    return _make_cad_morph(
        build,
        p0,
        make_model,
        length_unit=length_unit,
        step=step,
        sliding_ports=sliding_ports,
        extension=extension,
        curved=curved,
    )
