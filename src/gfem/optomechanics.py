r"""The optomechanical overlap of a displacement with a wave port's mode.

A mechanical displacement :math:`u(x, y)` of a waveguide's cross-section,
scaled by a dimensionless amplitude :math:`\alpha`, changes the optical
mode's effective index at a fixed optical frequency :math:`\omega` by two
mechanisms, to first order in :math:`\alpha`:

.. math::

    \frac{\partial n_\mathrm{eff}}{\partial\alpha}
    = \delta n_\mathrm{MB} + \delta n_\mathrm{PE}

The moving-boundary part :math:`\delta n_\mathrm{MB}` is the displacement of
the section's nodes seen by the optical mode (Johnson's perturbation of a
shifted interface, discretized); the photoelastic part
:math:`\delta n_\mathrm{PE}` is the strain of the displacement acting through
the materials' ``photoelastic_tensor`` (as in `gfem.acousto_optic_shift`).
`optomechanical_overlap` gives both, per unit amplitude of the displacement
it is given.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._checks import is_staging, is_traced
from gfem._dim2 import refuse_entry
from gfem._optical_shift import (
    _check_model,
    _frequency,
    _mode_weights,
    _section,
    _wave_port,
)
from gfem._pytree import register_fields
from gfem.acousto_optic import (
    _Caller,
    _check_displacement,
    _check_optical_fixed,
    _check_plane,
    _contribution,
    _crystal,
    _mean_strain_of,
    _weights_rule,
)
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gfem._field import Field, FieldMap
    from gfem.acousto_optic import _Crystal
    from gfem.model import Model
    from gfem.results import MechanicalEigenmodeResult

__all__ = [
    "OptomechanicalCoupling",
    "OptomechanicalOverlap",
    "optomechanical_coupling",
    "optomechanical_overlap",
]

_OVERLAP = _Caller(
    "optomechanical_overlap",
    "the optomechanical overlap",
    "the displacement",
    "the displacement or the photoelastic_tensor",
    "mechanical",
)
_COUPLING = _Caller(
    "optomechanical_coupling",
    "the optomechanical coupling",
    "the displacement",
    "the mechanical model or the photoelastic_tensor",
    "mechanical",
)


@dataclasses.dataclass(frozen=True, eq=False)
class OptomechanicalOverlap:
    r"""The overlap of a displacement with a wave port's mode, per unit amplitude.

    With :math:`u \to \alpha u` for a dimensionless amplitude :math:`\alpha`,
    the effective index of the mode at the fixed optical frequency changes as
    :math:`\partial n_\mathrm{eff}/\partial\alpha` = ``total``: dimensionless
    (the displacement is in metres, the index per unit amplitude of that
    field). The phase of the displacement carries into the values: complex for
    a complex displacement (a lossy or a driven mode), real to round-off for a
    real one. A mechanical mode's sign follows its phase convention: an
    optimizer reads :math:`|\partial n_\mathrm{eff}/\partial\alpha|^2`.

    `optomechanical_overlap` gives one displacement's overlap, with the parts
    complex scalars. The ``overlap`` of an `OptomechanicalCoupling` holds those
    of its ``n`` mechanical modes at once: ``moving_boundary`` and
    ``photoelastic`` (so ``total``) are then arrays ``(n,)`` in the order of
    its ``modes``, and ``n_eff`` and ``n_g`` stay scalars (one optical mode
    serves them all).

    Attributes:
        port: The wave port's name.
        mode: Which mode (1 = the most propagating).
        frequency: The optical frequency, Hz.
        n_eff: The unperturbed effective index (real; the section is
            lossless), a constant.
        n_g: The mode's group index (real), a constant.
        moving_boundary: :math:`\delta n_\mathrm{MB}`, the moving-boundary
            part: a complex scalar, or ``(n,)`` over a coupling's modes.
        photoelastic: :math:`\delta n_\mathrm{PE}`, the photoelastic part:
            a complex scalar, or ``(n,)`` over a coupling's modes.
    """

    port: str
    mode: int
    frequency: float
    n_eff: jax.Array
    n_g: jax.Array
    moving_boundary: jax.Array
    photoelastic: jax.Array

    @property
    def total(self) -> jax.Array:
        r""":math:`\partial n_\mathrm{eff}/\partial\alpha`: the two parts' sum."""
        return self.moving_boundary + self.photoelastic

    def __repr__(self) -> str:
        head = f"OptomechanicalOverlap(port {self.port!r}, mode {self.mode}"
        try:
            n = complex(np.asarray(self.n_eff))
            if np.ndim(self.moving_boundary):
                return (
                    f"{head}, {self.frequency:.6g} Hz, n_eff {n.real:.6f}, "
                    f"{np.shape(self.moving_boundary)[0]} displacements)"
                )
            mb, pe = (
                complex(np.asarray(v))
                for v in (self.moving_boundary, self.photoelastic)
            )
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"{head})"
        return (
            f"{head}, {self.frequency:.6g} Hz, n_eff {n.real:.6f}, "
            f"moving_boundary {mb:.6g}, photoelastic {pe:.6g})"
        )


register_fields(
    OptomechanicalOverlap,
    ("n_eff", "n_g", "moving_boundary", "photoelastic"),
)


def _optical_only(model: Model) -> Model:
    """``model`` without the photoelastic tensors, which the mode does not read.

    A tensor traced by `jax.grad` would send the mode solve down the traced
    path; the mode is held constant, so it never needs them.
    """
    return model.replace(
        materials={
            name: dataclasses.replace(material, photoelastic_tensor=None)
            for name, material in model.materials.items()
        }
    )


def _node_gradient(
    model: Model, port: str, frequency: float, mode: int | None
) -> tuple[np.ndarray, jax.Array, jax.Array, int]:
    r"""The gradient of :math:`\mathrm{Re}\,n_\mathrm{eff}` in the mesh nodes.

    One port mode solve and its backward pass. Returns ``(n_nodes, 3)`` in
    1/m (non-zero on the port's nodes), ``n_eff``, ``n_g`` and the mode
    number.
    """
    import gfem
    from gfem.problems import PortMode

    found: list[int] = []

    def n_eff(nodes: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        moved = model.replace(mesh=model.mesh.with_nodes(nodes))
        out = gfem.solve(moved, PortMode(port, frequency, mode=mode))
        found.append(out.mode)  # a Python int, not traced
        n_g = jnp.real(jnp.asarray(out.n_g)[0])  # a PortMode solve always carries it
        value = jnp.real(out.n_eff[0])
        return value, (jax.lax.stop_gradient(value), jax.lax.stop_gradient(n_g))

    grad, (n, n_g) = jax.grad(n_eff, has_aux=True)(jnp.asarray(model.mesh.nodes))
    return np.asarray(grad), n, n_g, found[0]


def _host_nodes(mesh: Any) -> np.ndarray:
    """The node coordinates of ``mesh`` as numbers (m).

    The support is a place, not a design variable: its tests read the value of
    node coordinates a node map traces.
    """
    nodes = mesh.nodes
    if is_traced(nodes):
        nodes = jax.lax.stop_gradient(nodes)
    return np.asarray(nodes, float)


def _covered(displacement: FieldMap, xy: np.ndarray) -> np.ndarray:
    """Which points ``xy`` (``(n, 2)``, m) both in-plane components are defined at."""
    from gfem import _probe

    mask = np.ones(len(xy), bool)
    for c in "xy":
        field = displacement[c]
        mesh = field.mesh
        if is_traced(mesh.nodes):  # the place a node map traces, as numbers
            mesh = mesh.with_nodes(_host_nodes(mesh))
        mask &= _probe.engine().covered(mesh, _probe.layout(field), xy)
    return mask


def _support_boundary(field: Field) -> tuple[np.ndarray, np.ndarray]:
    """The edges of the boundary of ``field``'s support on its 2D mesh.

    Returns the end points ``(k, 2, 2)`` (m) of each edge and the centroid
    ``(k, 2)`` of the support triangle that owns it.
    """
    from gfem import _probe

    nodes = _host_nodes(field.mesh)[:, :2]
    inside = _probe.engine().in_support(field.mesh, _probe.layout(field))
    tris = np.asarray(field.mesh.volume_elements)[inside]
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    owner = np.tile(np.arange(len(tris)), 3)
    _, first, count = np.unique(
        np.sort(edges, axis=1), axis=0, return_index=True, return_counts=True
    )
    once = first[count == 1]
    return nodes[edges[once]], nodes[tris[owner[once]]].mean(axis=1)


#: Where a node off the support reads the displacement: this share of the way
#: from the nearest point of the boundary to the centroid of the triangle that
#: owns it, so the point lies inside that triangle (its barycentric coordinates
#: are at least this over 3, far above the locator's 1e-9) and moves the value
#: read by this share of the triangle's variation, far below the sagitta the
#: node stands off by.
_INWARD = 1e-6


def _nearest_on_boundary(
    points: np.ndarray, edges: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""The read point of each of ``points`` near the boundary ``edges``.

    A point within a quarter of an edge's length :math:`h` of that edge reads
    at the edge's nearest point, moved `_INWARD`. Why a quarter: a chord of
    length :math:`h` of a boundary curve with radius of curvature :math:`R`
    lies inside the curve by at most its sagitta,
    :math:`R - \sqrt{R^2 - h^2/4} \le h/4` for :math:`R \ge 5h/8`
    (:math:`\approx h^2/(8R)` when :math:`R \gg h`). So a node of the curve
    that another mesh's vertices miss is that close to one of its edges
    wherever the mesh resolves the curve with edges shorter than 1.6
    :math:`R`, while a node a few edges away, which the support does not reach,
    is not taken. Returns a mask over ``points`` and the read points ``(m,
    2)`` of the points it keeps.
    """
    from scipy.spatial import KDTree

    keep = np.zeros(len(points), bool)
    read = np.zeros((len(points), 2))
    if not len(edges):
        return keep, read[keep]
    a, b = edges[:, 0], edges[:, 1]
    length = np.linalg.norm(b - a, axis=1)
    # an edge within a quarter of its length of a point has its midpoint within
    # three quarters of it
    near = KDTree((a + b) / 2).query_ball_point(points, 0.75 * length.max())
    for i, (p, candidates) in enumerate(zip(points, near, strict=True)):
        if not candidates:
            continue
        k = np.asarray(candidates)
        ab = b[k] - a[k]
        t = np.clip(np.einsum("ij,ij->i", p - a[k], ab) / length[k] ** 2, 0.0, 1.0)
        foot = a[k] + t[:, None] * ab
        gap = np.linalg.norm(p - foot, axis=1)
        ok = np.flatnonzero(gap <= length[k] / 4)
        if not len(ok):
            continue
        j = ok[np.argmin(gap[ok])]
        keep[i] = True
        read[i] = foot[j] + _INWARD * (centroids[k[j]] - foot[j])
    return keep, read[keep]


def _sampled_nodes(
    displacement: FieldMap, xy: np.ndarray, interface: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The section nodes ``xy`` (``(n, 2)``, m) the displacement moves.

    A node in the support reads the displacement where it stands. A node on
    an interface of the optical section (``interface``) that lies just off the
    support, where a curved boundary meshed twice puts it, reads it at the
    nearest point of the support's boundary (`_nearest_on_boundary`). Every
    other node stays. Returns the mask over ``xy`` and the read points ``(m,
    2)`` of the moved nodes, in their order.
    """
    inside = _covered(displacement, xy)
    read = xy.copy()
    off = np.flatnonzero(interface & ~inside)
    if len(off):
        edges, centroids = _support_boundary(displacement["x"])
        near, points = _nearest_on_boundary(xy[off], edges, centroids)
        held = _covered(displacement, points)
        near[np.flatnonzero(near)[~held]] = False
        inside[off[near]] = True
        read[off[near]] = points[held]
    return inside, read[inside]


class _Section(NamedTuple):
    """The port's section as the moving boundary reads it.

    ``nodes`` are the section's mesh nodes; ``interface`` marks those shared
    by triangles of two volume regions; ``walls`` are the edges (pairs of
    positions in ``nodes``) of the loop around the whole section, the optical
    domain's truncation, and ``normals`` their unit normals in :math:`(x, y)`.
    """

    nodes: np.ndarray
    interface: np.ndarray
    walls: np.ndarray
    normals: np.ndarray


def _section_nodes(model: Model, faces: np.ndarray, tets: np.ndarray) -> _Section:
    """The nodes, interfaces and outer walls of the section ``faces``."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    mesh = model.mesh
    tri = np.asarray(mesh.triangles)[faces]
    nodes = np.unique(tri)
    local = np.searchsorted(nodes, tri)
    tags = np.repeat(np.asarray(mesh.tetrahedron_tags)[tets], 3)
    pairs = np.unique(np.stack([local.ravel(), tags], axis=1), axis=0)
    interface = np.bincount(pairs[:, 0], minlength=len(nodes)) > 1
    edges = np.sort(
        np.concatenate([local[:, [0, 1]], local[:, [1, 2]], local[:, [2, 0]]]), axis=1
    )
    unique, count = np.unique(edges, axis=0, return_counts=True)
    rim = unique[count == 1]
    # the loop around the whole section holds its leftmost node; the other loops
    # of the rim are holes (an excluded electrode), whose edges are surfaces
    xy = _host_nodes(mesh)[nodes, :2]
    graph = coo_matrix(
        (np.ones(len(rim)), (rim[:, 0], rim[:, 1])), shape=(len(nodes),) * 2
    )
    _, label = connected_components(graph, directed=False)
    on_rim = np.unique(rim)
    leftmost = on_rim[np.argmin(xy[on_rim, 0])]
    walls = rim[label[rim[:, 0]] == label[leftmost]]
    tangent = xy[walls[:, 1]] - xy[walls[:, 0]]
    normals = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return _Section(nodes, interface, walls, normals)


class _Setup(NamedTuple):
    """What the optical model fixes before any solve: the section and its crystal."""

    frequency: float
    faces: np.ndarray
    tets: np.ndarray
    crystal: _Crystal
    section: _Section
    photoelastic_rows: np.ndarray
    nodes_xy: np.ndarray


def _refuse_jit(caller: _Caller) -> None:
    """Raise `NonDifferentiableError` inside `jax.jit` (or another staging trace)."""
    if is_staging():
        raise NonDifferentiableError(
            f"{caller.owner} cannot run inside jax.jit: it reads the optical "
            "mode's node gradient and the displacement's mesh as numbers. Call it "
            "outside jax.jit (jax.grad works through it) and jit the rest."
        )


def _setup(
    model: Model, port: str, frequency: float, caller: _Caller, displacement: Any
) -> _Setup:
    """Check the optical model and read its section (no solve).

    Raises what `optomechanical_overlap` documents for the model, the port,
    the frequency and ``displacement``.
    """
    _refuse_jit(caller)
    refuse_entry(getattr(model, "mesh", None), caller.owner)
    _check_model(model, caller.owner)
    _check_displacement(displacement, caller.owner)
    f = _frequency(frequency, caller.owner)
    faces, tets = _section(model, _wave_port(model, port))
    crystal = _crystal(model, port, tets, caller, electro_optic=False, require=False)
    _check_optical_fixed(model, caller)
    if is_traced(model.mesh.nodes):
        raise NonDifferentiableError(
            f"{caller.owner}: gfem cannot differentiate {caller.what} "
            "with respect to the optical model's node coordinates: the mode and "
            "the points the displacement is read at are held constant. Wrap them "
            "in jax.lax.stop_gradient to use their value without their "
            "derivative."
        )
    _check_plane(model.mesh, faces, port, caller)
    return _Setup(
        f,
        faces,
        tets,
        crystal,
        _section_nodes(model, faces, tets),
        np.flatnonzero(crystal.has_p),
        np.asarray(model.mesh.nodes, float)[:, :2],
    )


def _check_support(
    setup: _Setup, displacement: FieldMap, caller: _Caller, port: str
) -> tuple[np.ndarray, np.ndarray]:
    """The section's moved nodes and their read points; raise if nothing couples."""
    section = setup.section
    xy = setup.nodes_xy[section.nodes]
    inside, points = _sampled_nodes(displacement, xy, section.interface)
    if not len(setup.photoelastic_rows) and not inside.any():
        raise ValueError(
            f"{caller.owner}: no element of wave port {port!r}'s section has a "
            "photoelastic_tensor and no node of the section lies in the "
            "displacement's support, so the overlap is zero. Set a "
            "photoelastic_tensor on the optical material, or align the "
            "displacement's mesh with the section (same x, y in m)."
        )
    return inside, points


class _Solved(NamedTuple):
    """The optical mode's constants: the node gradient, indices and tensor weights."""

    grad: np.ndarray
    n_eff: jax.Array
    n_g: jax.Array
    mode: int
    weights: jax.Array


def _solve_optical(model: Model, setup: _Setup, port: str, mode: int | None) -> _Solved:
    """The optical work, done once for every displacement.

    Two port mode solves: one with its backward pass for the node gradient,
    and, with a photoelastic element, one of the weights rule for its tensor
    weights.
    """
    optical = _optical_only(model)
    grad, n_eff, n_g, found = _node_gradient(optical, port, setup.frequency, mode)
    weights = jnp.zeros((0, 3, 3))  # no photoelastic element reads them
    if len(setup.photoelastic_rows):
        _, _, weights, _ = _mode_weights(
            optical,
            port,
            setup.frequency,
            mode,
            setup.faces,
            tensor=True,
            rule=_weights_rule(),
        )
    return _Solved(grad, n_eff, n_g, found, weights)


#: A wall node moves across its wall when the normal part of its displacement
#: exceeds this share of the largest displacement of the section. A motion along
#: a straight wall reads a normal part of exactly zero, or round-off of it
#: (1e-16 relative); a wall that moves with the structure, the case to refuse,
#: moves by a share of order one.
_ACROSS = 1e-8


def _check_walls(
    setup: _Setup, w: jax.Array, inside: np.ndarray, caller: _Caller, port: str
) -> None:
    """Raise if the displacement moves a node across the section's outer wall.

    ``w`` is the in-plane displacement ``(m, 2)`` of the moved nodes
    ``inside`` (a mask over the section's nodes). Its value is read as numbers
    (`jax.grad` passes it through); a batch of `jax.vmap` is not checked.
    """
    value = jax.lax.stop_gradient(w)
    if is_traced(value) or not len(setup.section.walls):
        return
    full = np.zeros((len(inside), 2), complex)
    full[inside] = np.asarray(value)
    scale = np.abs(full).max()
    walls, normals = setup.section.walls, setup.section.normals
    across = np.abs(np.einsum("kij,kj->ki", full[walls], normals))
    if not scale or across.max() <= _ACROSS * scale:
        return
    k, end = np.unravel_index(np.argmax(across), across.shape)
    node = setup.section.nodes[walls[k, end]]
    at = ", ".join(f"{v:.6g}" for v in setup.nodes_xy[node])
    raise ValueError(
        f"{caller.owner}: the displacement moves node ({at}) m of wave port "
        f"{port!r}'s section across the section's outer edge, the truncation "
        "of the optical domain (its walls are not a surface of the structure, "
        "and moving them changes the mode). Keep the moving parts away from the "
        "edge: enlarge the optical section, or clamp the mechanical solid where "
        "it meets the edge."
    )


def _parts(
    model: Model,
    setup: _Setup,
    solved: _Solved,
    displacement: FieldMap,
    *,
    inside: np.ndarray,
    points: np.ndarray,
    caller: _Caller,
    port: str,
) -> tuple[jax.Array, jax.Array]:
    """The moving-boundary and photoelastic overlaps of one displacement."""
    crystal, rows = setup.crystal, setup.photoelastic_rows
    mb = jnp.zeros((), dtype=jnp.complex128)
    pe = jnp.zeros((), dtype=jnp.complex128)
    if inside.any():
        w = jnp.stack([displacement[c].at(points) for c in "xy"], axis=1)
        _check_walls(setup, w, inside, caller, port)
        g = jnp.asarray(solved.grad[setup.section.nodes[inside]][:, :2])
        mb = jnp.sum(g * w).astype(jnp.complex128)
    if len(rows):
        mean_s = _mean_strain_of(
            displacement,
            model.mesh,
            setup.tets[crystal.rows[rows]],
            offset=jnp.zeros(2),
            owner=f"{caller.owner} (wave port {port!r})",
            moved={},
        )
        eta = jnp.einsum("nIJ,nJ->nI", crystal.p[rows], mean_s)
        pe = _contribution(
            solved.weights[crystal.rows[rows]], crystal.eps[rows], eta[None]
        )[0]
    return mb, pe.astype(jnp.complex128)


def optomechanical_overlap(
    model: Model,
    displacement: FieldMap,
    *,
    port: str,
    frequency: float,
    mode: int | None = None,
) -> OptomechanicalOverlap:
    r"""The optomechanical overlap of a displacement with a wave port's mode.

    Solves the port's mode on ``model`` (a `gfem.PortMode` at one optical
    frequency) and gives the change of its effective index at that frequency
    per unit amplitude :math:`\alpha` of ``displacement`` :math:`u`, to first
    order:

    .. math::

        \frac{\partial n_\mathrm{eff}}{\partial\alpha}
        = \underbrace{\sum_{i\in\mathrm{section}}
            \frac{\partial\,\mathrm{Re}\,n_\mathrm{eff}}{\partial \mathbf r_i}
            \cdot \mathbf W_i}_{\delta n_\mathrm{MB}}
        + \underbrace{\sum_t W_t : \Delta\varepsilon_t}_{\delta n_\mathrm{PE}},
        \qquad
        \Delta\varepsilon_t = -\varepsilon_t\,(p_t \bar S_t)\,\varepsilon_t

    The moving-boundary part is the gradient of the port mode's effective
    index in the nodes :math:`\mathbf r_i` of its section, contracted with the
    in-plane displacement :math:`\mathbf W_i = (u_x, u_y)` read at each node
    inside the displacement's support (zero at the others, front and back
    nodes of the slab alike: the slab moves as a prism; :math:`u_z` moves no
    node of a section that is invariant along :math:`z`). It is the discrete
    form of Johnson's interface integral, exact for the discrete problem, for
    any permittivity (anisotropic included) and without a one-sided sampling of
    the field at a corner. An interface of the optical mesh carries the term,
    and a mechanical boundary inside a homogeneous optical region contributes
    discretization error only. The two meshes need not share nodes: where a
    curved boundary is meshed twice, each mesh's boundary is the polygon
    through its own vertices, and an interface node of the section that lies
    off the mechanical polygon, within a quarter of the nearest mechanical
    boundary edge's length :math:`h` (beyond the chord's sagitta
    :math:`\approx h^2/(8R)` wherever the radius of curvature :math:`R`
    exceeds :math:`5h/8`), reads the displacement at the nearest point of that
    boundary. A node that is not on an interface, and one farther off, does
    not move.

    The section's outer edge, the truncation of the optical domain (a PEC or
    natural wall), must clear the moving parts: a displacement that moves a
    node of that edge across it is refused, as it would move the wall. A
    motion along a straight wall is not a motion of the wall and is read.

    The photoelastic part is the first-order shift of
    `gfem.acousto_optic_shift` for the strain :math:`\bar S_t` of the
    displacement (`gfem.strain`, mean over each tetrahedron :math:`t` behind a
    triangle of the port's section, read at its quadrature points) through the
    ``photoelastic_tensor`` :math:`p_t` of its material, with :math:`W_t`
    the mode's weight at 1 W (see `gfem.acousto_optic_shift`). By reciprocity
    the volume overlap of the strain already holds the electrostrictive
    pressure on the boundaries. The rotation of the crystal by the
    displacement (roto-optic) is not counted.

    The sign is that of the displacement: reversing :math:`u` reverses both
    parts, and a mechanical mode's own sign follows its phase convention.
    A positive value means the displacement raises the effective index.

    The optical mode is solved once (two port mode solves: one with its
    backward pass for the node gradient, one for the photoelastic weights)
    and held constant. `jax.grad` reaches the displacement's fields (so every
    input of the mechanical solve that made them) and the
    ``photoelastic_tensor`` of a `gfem.MaterialField`; the optical model's
    other inputs are refused. The function runs outside `jax.jit` only. A
    mode's sign follows its phase convention, which can flip between two
    nearby designs of a symmetric section: an objective reads the
    convention-free :math:`|\partial n_\mathrm{eff}/\partial\alpha|^2`.

    Args:
        model: The optical model: a 3D slab (`gfem.Mesh.from_gmsh_cross_section`)
            with a lossless section.
        displacement: A displacement ``FieldMap`` with the components
            ``x``, ``y``, ``z`` (m) at :math:`q = 0`, on a native 2D mesh with
            the same global :math:`(x, y)` as the optical section (translate a
            mesh to align them).
        port: The wave port's name; its section lies in a plane
            :math:`z = \mathrm{const}`.
        frequency: One optical frequency, Hz.
        mode: Which mode (1 = the most propagating), or ``None`` for the wave
            port's own.

    Returns:
        The two parts and the mode's effective and group index.

    Raises:
        TypeError: ``model`` is not a `gfem.Model`, or ``displacement`` is not
            a ``FieldMap``.
        ValueError: ``port`` is not a wave port of the model, a sequence of
            frequencies is given, ``displacement`` lacks a component, is not
            on a 2D mesh or has an axial wavenumber other than 0, the port's
            section is not in a plane :math:`z = \mathrm{const}`, no node of
            the section is in the displacement's support and no element has a
            ``photoelastic_tensor``, an element with one lies outside the
            displacement's support (the message names the region), or the
            displacement moves a node of the section's outer edge across it
            (the message names the node).
        NotImplementedError: The model is on a 2D mesh (a 2D `gfem.PortMode` is
            a line mode) or the section is lossy (a complex ``eps_r``, a
            ``tan_delta`` or ``conductivity`` other than 0, or a permittivity
            law).
        NonDifferentiableError: Under `jax.grad`, an input of the optical
            model other than the photoelastic tensors (its ``eps_r``,
            ``tan_delta``, ``mu_r``, ``conductivity`` or node coordinates):
            the mode is held constant. Inside `jax.jit`: the overlap reads
            the mode's node gradient and the meshes as numbers.
    """
    setup = _setup(model, port, frequency, _OVERLAP, displacement)
    inside, points = _check_support(setup, displacement, _OVERLAP, port)
    solved = _solve_optical(model, setup, port, mode)
    mb, pe = _parts(
        model,
        setup,
        solved,
        displacement,
        inside=inside,
        points=points,
        caller=_OVERLAP,
        port=port,
    )
    return OptomechanicalOverlap(
        port=port,
        mode=solved.mode,
        frequency=setup.frequency,
        n_eff=solved.n_eff,
        n_g=solved.n_g,
        moving_boundary=mb,
        photoelastic=pe,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class OptomechanicalCoupling:
    r"""The vacuum optomechanical coupling of mechanical modes to a wave port's mode.

    For a mechanical mode of angular frequency :math:`\Omega` that stores
    1 J/m, with the optical mode at the fixed angular frequency
    :math:`\omega = 2\pi f`, ``total`` is

    .. math::

        \tilde g_0 = \frac{\omega}{n_g}\,
        \frac{\partial n_\mathrm{eff}}{\partial\alpha}\,
        \sqrt{\frac{\hbar\,\mathrm{Re}\,\Omega}{4}},
        \qquad g_0(L) = \frac{\tilde g_0}{\sqrt{L}}

    in rad s\ :sup:`-1` m\ :sup:`1/2`, with ``moving_boundary`` and
    ``photoelastic`` the two parts of :math:`\partial n_\mathrm{eff}/\partial\alpha`
    scaled the same way. The values are complex: real to round-off for a
    lossless mode, complex for a lossy one (the displacement is). Their sign
    follows the mode's phase convention (largest displacement coefficient
    real and positive) and flips with it: :math:`|g_0|` is the convention-free
    number.

    Attributes:
        overlap: The `OptomechanicalOverlap` of each mode, arrays of
            shape ``(n,)`` over the modes (``n_eff``, ``n_g`` are scalars:
            one optical mode serves every mode).
        omega_m: The mechanical modes' complex angular frequencies, rad/s,
            ``(n,)``.
        moving_boundary: The moving-boundary part of :math:`\tilde g_0`, ``(n,)``.
        photoelastic: The photoelastic part of :math:`\tilde g_0`, ``(n,)``.
        modes: The numbers of the mechanical modes (``MechanicalEigenmodeResult``
            order, 0 = nearest its target), in the order of the arrays.
    """

    overlap: OptomechanicalOverlap
    omega_m: jax.Array
    moving_boundary: jax.Array
    photoelastic: jax.Array
    modes: tuple[int, ...]

    @property
    def total(self) -> jax.Array:
        r""":math:`\tilde g_0`, ``(n,)``: the two parts' sum."""
        return self.moving_boundary + self.photoelastic

    def g0(self, length: float) -> jax.Array:
        r""":math:`g_0 = \tilde g_0/\sqrt{L}` of a guide of length ``length``, rad/s.

        The parts follow by the same division (``moving_boundary`` and
        ``photoelastic`` over :math:`\sqrt{L}`). The coupling of a length
        :math:`L` treats it as a cavity: the optical mode shifts by
        :math:`\partial\omega_c/\partial\alpha = -(\omega/n_g)\,
        \partial n_\mathrm{eff}/\partial\alpha` and the mode's effective
        mass is :math:`2L/(\mathrm{Re}\,\Omega)^2`. Angular units: divide by
        :math:`2\pi` for a rate in Hz.

        Raises:
            ValueError: ``length`` is not positive.
        """
        if not is_traced(length) and not float(length) > 0:
            raise ValueError(f"g0 needs a positive length in m, got {length!r}.")
        return self.total / jnp.sqrt(length)

    def __repr__(self) -> str:
        head = f"OptomechanicalCoupling(port {self.overlap.port!r}, modes {self.modes}"
        try:
            g = np.asarray(self.total)
            f = np.real(np.asarray(self.omega_m)) / (2 * np.pi)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return f"{head})"
        rows = ", ".join(
            f"{a:.6g} Hz: |g0 sqrt(L)| {abs(b):.4g}" for a, b in zip(f, g, strict=True)
        )
        return f"{head}, [{rows}] rad/s m^1/2)"


register_fields(
    OptomechanicalCoupling, ("overlap", "omega_m", "moving_boundary", "photoelastic")
)


def _modes(mechanical: MechanicalEigenmodeResult, selection: Any) -> tuple[int, ...]:
    """The non-negative mode numbers ``selection`` names, ``None`` for every mode."""
    n = mechanical.n_modes
    if selection is None:
        return tuple(range(n))
    single = isinstance(selection, int | np.integer | bool | np.bool_)
    modes = []
    for k in [selection] if single else selection:
        if isinstance(k, bool | np.bool_) or not isinstance(k, int | np.integer):
            raise TypeError(
                f"{_COUPLING.owner}: mechanical_modes takes an int or a sequence "
                f"of ints, got {k!r}."
            )
        if not -n <= int(k) < n:
            raise IndexError(
                f"{_COUPLING.owner}: Mode {int(k)} does not exist: the result has "
                f"{n} mode(s), numbered from 0."
            )
        modes.append(int(k) % n)
    if not modes:
        raise ValueError(f"{_COUPLING.owner}: mechanical_modes is empty.")
    return tuple(modes)


def optomechanical_coupling(
    model: Model,
    mechanical: MechanicalEigenmodeResult,
    *,
    port: str,
    frequency: float,
    mode: int | None = None,
    mechanical_modes: int | Sequence[int] | None = None,
) -> OptomechanicalCoupling:
    r"""The vacuum coupling :math:`g_0` of mechanical modes to a wave port's mode.

    For each requested mode :math:`u` of a `gfem.MechanicalEigenmode` solve
    (:math:`q = 0`, stored energy 1 J/m) the overlap of
    `optomechanical_overlap` is turned into the coupling of a length
    :math:`L` of the guide treated as a cavity. A displacement amplitude
    :math:`\alpha` shifts the optical resonance by
    :math:`\partial\omega_c/\partial\alpha = -(\omega/n_g)\,
    \partial n_\mathrm{eff}/\partial\alpha` at fixed :math:`\omega`; the mode's
    effective mass is :math:`m_\alpha = L\int\rho|u|^2\,dA =
    2L/(\mathrm{Re}\,\Omega)^2`, so its zero-point amplitude is
    :math:`\alpha_\mathrm{zpf} = \sqrt{\hbar/(2 m_\alpha\,\mathrm{Re}\,\Omega)}
    = \sqrt{\hbar\,\mathrm{Re}\,\Omega/(4L)}` and

    .. math::

        \tilde g_0 = g_0\sqrt{L} = \frac{\omega}{n_g}\,
        \frac{\partial n_\mathrm{eff}}{\partial\alpha}\,
        \sqrt{\frac{\hbar\,\mathrm{Re}\,\Omega}{4}}

    with :math:`n_g` the port mode's group index and :math:`\hbar` from
    `scipy.constants`. The sign is the
    convention :math:`g_0 = -(\partial\omega_c/\partial\alpha)\,\alpha_\mathrm{zpf}`,
    positive when the mode's displacement raises the effective index, and it
    follows the mode's phase convention: read :math:`|g_0|`. The modes
    are intramodal forward couplings at :math:`q = 0`; the roto-optic term is
    not counted.

    The optical mode, its node gradient and its weights are solved once (two
    port mode solves) for every requested mechanical mode. `jax.grad`
    reaches every input of the mechanical solve (stiffness, density, quality
    factor, piezoelectric tensor, permittivity, node coordinates of the
    mechanical mesh) through :math:`\mathrm{Re}\,\Omega` and the mode field,
    and the ``photoelastic_tensor``; a clustered (degenerate) mode is refused by the
    mechanical solve's backward pass, as for any scalar of the mode. The
    optical model's other inputs are refused (the mode is held constant), and
    so is `jax.jit`. An objective reads the convention-free :math:`|g_0|^2`.

    Args:
        model: The optical model, as for `optomechanical_overlap`.
        mechanical: A `gfem.MechanicalEigenmodeResult` at :math:`q = 0` on a
            native 2D mesh with the same global :math:`(x, y)` (m) as the
            optical section.
        port: The wave port's name.
        frequency: One optical frequency, Hz.
        mode: Which optical mode (1 = the most propagating), or ``None`` for
            the wave port's own.
        mechanical_modes: Which mechanical modes (numbers in ``mechanical``,
            negative counting from the last): an int, a sequence, or ``None``
            for every mode.

    Returns:
        The coupling, parts and total, per requested mode.

    Raises:
        TypeError: ``mechanical`` is not a `gfem.MechanicalEigenmodeResult`, or
            ``model`` is not a `gfem.Model`.
        IndexError: A number of ``mechanical_modes`` is not a mode of ``mechanical``.
        ValueError: ``mechanical`` has an axial wavenumber other than 0,
            ``mechanical_modes`` is empty, and every ``ValueError`` of
            `optomechanical_overlap`.
        NotImplementedError: As `optomechanical_overlap`.
        NonDifferentiableError: As `optomechanical_overlap`.
    """
    from gfem.results import MechanicalEigenmodeResult

    _refuse_jit(_COUPLING)
    if not isinstance(mechanical, MechanicalEigenmodeResult):
        raise TypeError(
            f"{_COUPLING.owner} takes a gfem.MechanicalEigenmodeResult (the solve of a "
            f"gfem.MechanicalEigenmode), got {type(mechanical).__name__}; give a "
            "displacement FieldMap to optomechanical_overlap instead."
        )
    if mechanical.wavenumber != 0:
        raise ValueError(
            f"{_COUPLING.owner}: the mechanical modes have the axial wavenumber "
            f"q = {mechanical.wavenumber:g} rad/m; only q = 0 is read (a mode "
            "with q != 0 averages to zero against an optical mode with itself)."
        )
    modes = _modes(mechanical, mechanical_modes)
    first = mechanical.displacement(modes[0])
    setup = _setup(model, port, frequency, _COUPLING, first)
    inside, points = _check_support(setup, first, _COUPLING, port)
    solved = _solve_optical(model, setup, port, mode)
    parts = [
        _parts(
            model,
            setup,
            solved,
            mechanical.displacement(m),
            inside=inside,
            points=points,
            caller=_COUPLING,
            port=port,
        )
        for m in modes
    ]
    mb = jnp.stack([a for a, _ in parts])
    pe = jnp.stack([b for _, b in parts])
    omega_m = jnp.asarray(mechanical.omega)[jnp.asarray(modes)]
    scale = (
        2
        * jnp.pi
        * setup.frequency
        / solved.n_g
        * jnp.sqrt(scipy.constants.hbar * jnp.real(omega_m) / 4)
    )
    return OptomechanicalCoupling(
        overlap=OptomechanicalOverlap(
            port=port,
            mode=solved.mode,
            frequency=setup.frequency,
            n_eff=solved.n_eff,
            n_g=solved.n_g,
            moving_boundary=mb,
            photoelastic=pe,
        ),
        omega_m=omega_m,
        moving_boundary=scale * mb,
        photoelastic=scale * pe,
        modes=modes,
    )
