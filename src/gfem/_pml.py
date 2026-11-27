r"""Perfectly matched layers: the declaration, its geometry and its stretch.

A perfectly matched layer is a set of meshed volume regions around the air
box, each with its background material in ``Model.materials``. Its geometry
is read from the imported mesh: the interior box is the bounding box of the
active tetrahedra outside every layer region, and a layer tetrahedron beyond
the face :math:`\pm i` of the box is stretched along axis :math:`i` from
that face, over the side's thickness :math:`d`.

Under gfem's :math:`e^{+i\omega t}` the stretch factor along a stretch axis
is :math:`s = 1 - i\sigma(\xi)`, with
:math:`\sigma = \sigma_\max (\xi / d)^2` at the depth :math:`\xi` into the
layer and :math:`\sigma_\max = -3 \ln R_0 / (2 k_\text{ref} d)`; the PML
tensor is
:math:`\Lambda = \operatorname{diag}(s_y s_z / s_x,\ s_x s_z / s_y,\ s_x s_y / s_z)`.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from gfem import _mpi
from gfem._boundaries import (
    Absorbing,
    Conductor,
    FloquetPort,
    Impedance,
    LumpedPort,
    WavePort,
    surfaces_of,
)
from gfem._checks import optional_positive, static_float
from gfem._floquet import C0
from gfem._materials import Material, MaterialField, PermittivityLaw
from gfem._periodic import REL_TOL, Pairing
from gfem._pytree import readonly
from gfem.errors import (
    MeshRefinementUnsupportedError,
    RegionNotFoundError,
    ThinLayerWarning,
)
from gfem.mesh import LENGTH_UNITS, Mesh, _shortest_edges
from gfem.problems import Driven, Eigenmode, PortMode

if TYPE_CHECKING:
    from gfem._domain import EMDomain

_AXES = "xyz"
#: The faces of a tetrahedron (node positions), each opposite node ``k``.
_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
#: Largest component of a unit vector counted as zero (a face normal across
#: a stretch axis, a port normal across it, an axis-aligned principal axis).
_ALIGN_TOL = 1e-6
_FIXED = (
    "The stretch of a perfectly matched layer is fixed: give a plain number, "
    "outside the transformed function."
)
_BOX_SHELL = (
    "a perfectly matched layer must be a box shell around the interior box, "
    "with faces on the coordinate planes; radial and curved layers are not "
    "supported yet."
)


@dataclasses.dataclass(frozen=True)
class PML:
    r"""A perfectly matched layer: the volume regions that absorb outgoing waves.

    Pass one to ``gfem.Model(..., pml=gfem.PML(regions))``, naming every
    layer region; each region keeps its background material in
    ``materials`` (usually ``gfem.Material()`` for air), which solves that
    ignore the layer (electrostatic, DC, thermal) see. The layer stretches
    the coordinates into the complex plane, with the stretch factor
    :math:`s = 1 - i\sigma(\xi)` (gfem's :math:`e^{+i\omega t}`) along each
    stretch axis, :math:`\sigma = \sigma_\max (\xi / d)^2` at the depth
    :math:`\xi` and

    .. math::

        \sigma_\max = -\frac{3 \ln R_0}{2 k_\text{ref} d},

    with :math:`R_0` the target reflection, :math:`k_\text{ref}` the vacuum
    wavenumber at the reference frequency and :math:`d` the side's
    thickness. The background permittivity and permeability take the PML
    tensor
    :math:`\Lambda = \operatorname{diag}(s_y s_z/s_x,\ s_x s_z/s_y,\ s_x s_y/s_z)`;
    edges and corners stretch along two or three axes. A PEC-backed layer
    reflects :math:`|R| = R_0^{f \cos\theta / f_\text{ref}}` at the
    frequency :math:`f` and the angle :math:`\theta` from its normal, in
    vacuum.

    The geometry is read from the mesh: the interior box is the bounding
    box of the active tetrahedra outside every layer region, a layer
    tetrahedron beyond one of its faces is stretched along that face's
    normal, and :math:`d` is the layer's extent beyond the face. Mesh the
    layer as a box shell around the air box, at least
    :math:`\lambda_\max / 2` thick in its background, with about ten
    elements across at order 2; every layer region on one side must end on
    one plane. Its outer face is PMC by default, or a named `gfem.PEC`.
    The fields inside the layer are not physical.

    Attributes:
        regions: The layer's volume region names, a string or a sequence of
            strings.
        target_reflection: :math:`R_0`, the reflection the layer is designed
            for at normal incidence and at the reference frequency, in
            :math:`(0, 1)`. Smaller values need a thicker or finer layer.
        reference_frequency: The frequency (Hz) at which the layer meets its
            target reflection; above it the layer reflects less. ``None``:
            the lowest frequency of the solve, or the eigenmode target.

    Raises:
        TypeError: ``regions`` is not a name or a sequence of names.
        ValueError: No region, a region listed twice, a target reflection
            outside :math:`(0, 1)` or a reference frequency that is not > 0.
        NonDifferentiableError: A knob is traced by JAX.
    """

    regions: tuple[str, ...]
    target_reflection: float = dataclasses.field(default=1e-6, kw_only=True)
    reference_frequency: float | None = dataclasses.field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "regions", _regions(self.regions))
        r0 = static_float(self.target_reflection, "PML.target_reflection", _FIXED)
        if not 0 < r0 < 1:
            raise ValueError(
                "PML: target_reflection must be between 0 and 1 (exclusive), got "
                f"{r0!r}; 1e-6 is the default, smaller values need a thicker or "
                "finer layer."
            )
        set_(self, "target_reflection", r0)
        set_(
            self,
            "reference_frequency",
            optional_positive(
                self.reference_frequency,
                "PML: reference_frequency",
                "Hz",
                _FIXED,
            ),
        )

    def __repr__(self) -> str:
        out = f"PML({self.regions!r}, target_reflection={self.target_reflection!r}"
        if self.reference_frequency is not None:
            out += f", reference_frequency={self.reference_frequency:.12g}"
        return out + ")"


jax.tree_util.register_static(PML)


def _regions(value: Any) -> tuple[str, ...]:
    names = (value,) if isinstance(value, str) else value
    bad = not isinstance(names, Iterable)
    names = () if bad else tuple(names)
    if bad or not all(isinstance(n, str) for n in names):
        raise TypeError(
            "PML: regions must be volume region names (a string or a sequence "
            f"of strings), got {value!r}."
        )
    if not names:
        raise ValueError("PML: name at least one layer region.")
    for name in names:
        if names.count(name) > 1:
            raise ValueError(f"PML: region {name!r} is listed twice.")
    return names


# --- profile -------------------------------------------------------------------


def stretch_maximum(target_reflection: float, k_ref: float, d: float) -> float:
    r"""The stretch maximum :math:`\sigma_\max = -3\ln R_0/(2 k_\text{ref} d)`.

    The quadratic profile :math:`\sigma_\max(\xi/d)^2` then gives the
    PEC-backed layer the reflection :math:`R_0` at normal incidence and at
    :math:`k_\text{ref}`.

    Args:
        target_reflection: :math:`R_0`.
        k_ref: The reference wavenumber, in the inverse unit of ``d``.
        d: The layer thickness.
    """
    return -3.0 * math.log(target_reflection) / (2.0 * k_ref * d)


def reflection(target: float, f: float, f_ref: float, cos_theta: float) -> float:
    r"""The PML reflection :math:`|R| = R_0^{f\cos\theta/f_\text{ref}}`.

    The continuum reflection of a PEC-backed layer in vacuum, at the
    frequency :math:`f` and the angle :math:`\theta` from its normal. In a
    background of index :math:`n` the exponent takes the factor :math:`n`:
    pass :math:`n f` as ``f``.

    Args:
        target: The target reflection :math:`R_0`.
        f: The frequency, Hz.
        f_ref: The reference frequency, Hz.
        cos_theta: :math:`\cos\theta`.
    """
    return target ** (f * cos_theta / f_ref)


# --- geometry ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Side:
    """One side of the interior box that a layer covers.

    Attributes:
        axis: The stretch axis, 0, 1 or 2.
        sign: +1 beyond the upper face, -1 beyond the lower one.
        face: The face's coordinate along ``axis``, m.
        thickness: The layer's extent beyond the face, m.
        regions: The layer regions with a tetrahedron on this side.
    """

    axis: int
    sign: int
    face: float
    thickness: float
    regions: tuple[str, ...]

    @property
    def label(self) -> str:
        """``"+x"``, ``"-y"``, ..."""
        return ("+" if self.sign > 0 else "-") + _AXES[self.axis]


@dataclasses.dataclass(frozen=True, eq=False)
class Layers:
    """The geometry of a model's perfectly matched layer, from its imported mesh.

    Compares by identity: it is static pytree data of a `Model`.

    Attributes:
        lower: The interior box's lower corner, m.
        upper: The interior box's upper corner, m.
        sides: The covered sides, by axis then sign.
        elements: The active layer tetrahedra.
        nodes: Their nodes.
    """

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    sides: tuple[Side, ...]
    elements: np.ndarray
    nodes: np.ndarray

    @property
    def axes(self) -> tuple[int, ...]:
        """The stretch axes, sorted."""
        return tuple(sorted({s.axis for s in self.sides}))


def _length(value: float, mesh: Mesh) -> str:
    unit = mesh.length_unit
    return f"{value / LENGTH_UNITS[unit]:.6g} {unit}"


def derive_layers(mesh: Mesh, pml: PML, active: np.ndarray) -> Layers:
    """The interior box and the covered sides, from the imported nodes.

    Args:
        mesh: The model's mesh.
        pml: The layer, whose regions are volumes of ``mesh``.
        active: Per tetrahedron, whether it is in the electromagnetic solve
            domain.

    Raises:
        NotImplementedError: A layer tetrahedron straddles a face of the
            interior box.
        ValueError: The layer stretches along no axis, a layer tetrahedron
            lies inside the interior box, or a side does not end on one
            plane.
    """
    x = mesh.imported_nodes
    tets = mesh.tetrahedra
    names = {mesh.region(r).tag: r for r in pml.regions}
    is_layer = np.isin(mesh.tetrahedron_tags, list(names))
    inner = active & ~is_layer
    elements = np.flatnonzero(active & is_layer)
    if not inner.any():
        raise ValueError(
            f"perfectly matched layer {pml.regions!r}: every active tetrahedron "
            "is in the layer, so there is no interior box for it to surround."
        )
    pts = x[tets[inner]].reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    tol = REL_TOL * float(_shortest_edges(mesh)[elements].min())
    p = x[tets[elements]]  # (n, 4, 3)
    above = (p >= hi - tol).all(axis=1) & (p > hi + tol).any(axis=1)
    below = (p <= lo + tol).all(axis=1) & (p < lo - tol).any(axis=1)
    out = ((p > hi + tol) | (p < lo - tol)).any(axis=1)
    straddle = out & ~above & ~below
    region = [names[t] for t in mesh.tetrahedron_tags[elements]]
    if straddle.any():
        e, axis = np.argwhere(straddle)[0]
        raise NotImplementedError(
            f"perfectly matched layer {region[e]!r}: a tetrahedron straddles a "
            f"face of the interior box along {_AXES[axis]} (the box spans "
            f"{_length(lo[axis], mesh)} to {_length(hi[axis], mesh)}): " + _BOX_SHELL
        )
    beyond = above | below
    if not beyond.any():
        raise ValueError(
            f"perfectly matched layer {pml.regions!r} stretches along no axis: "
            "every layer tetrahedron lies inside the interior box (the bounding "
            "box of the active tetrahedra outside the layer). Mesh the layer "
            "outside the air box, around it."
        )
    inside = ~beyond.any(axis=1)
    if inside.any():
        e = int(np.flatnonzero(inside)[0])
        raise ValueError(
            f"perfectly matched layer {region[e]!r}: a tetrahedron lies inside "
            "the interior box (the bounding box of the active tetrahedra outside "
            "the layer): a layer must surround the air box, not sit in it."
        )
    sides = []
    for axis in range(3):
        for sign, on in ((-1, below[:, axis]), (1, above[:, axis])):
            if not on.any():
                continue
            face = float(hi[axis] if sign > 0 else lo[axis])
            side = elements[on]
            ext = (sign * (x[tets[side], axis] - face)).max(axis=1)
            d = float(ext.max())
            covered = {region[i] for i in np.flatnonzero(on)}
            regions = tuple(r for r in pml.regions if r in covered)
            label = ("+" if sign > 0 else "-") + _AXES[axis]
            _check_plane(
                mesh,
                pml,
                side=side,
                axis=axis,
                sign=sign,
                face=face,
                d=d,
                tol=tol,
                label=label,
            )
            sides.append(Side(axis, sign, face, d, regions))
    return Layers(
        lower=(float(lo[0]), float(lo[1]), float(lo[2])),
        upper=(float(hi[0]), float(hi[1]), float(hi[2])),
        sides=tuple(sides),
        elements=readonly(elements),
        nodes=readonly(np.unique(tets[elements])),
    )


def _exterior_faces(mesh: Mesh, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Faces of ``elements`` that no other tetrahedron of the mesh shares.

    Returns:
        ``(faces, owners)``: the faces' node triples, outward by the
        right-hand rule, and their tetrahedra.
    """
    tets = mesh.tetrahedra
    nodes = np.unique(tets[elements])
    near = np.flatnonzero(np.isin(tets, nodes).sum(axis=1) >= 3)
    faces = tets[near][:, _FACES].reshape(-1, 3)
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    owners = np.repeat(near, 4)
    opposite = np.tile(np.arange(4), len(near))
    keep = (counts[inverse.ravel()] == 1) & np.isin(owners, elements)
    faces, owners, opposite = faces[keep], owners[keep], opposite[keep]
    x = mesh.imported_nodes
    a, b, c = (x[faces[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    inward = np.einsum("ij,ij->i", normal, x[tets[owners, opposite]] - a) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return faces, owners


def _check_plane(
    mesh: Mesh,
    pml: PML,
    *,
    side: np.ndarray,
    axis: int,
    sign: int,
    face: float,
    d: float,
    tol: float,
    label: str,
) -> None:
    """Every exterior face of a side facing outward lies on its outer plane."""
    x = mesh.imported_nodes
    faces, owners = _exterior_faces(mesh, side)
    a, b, c = (x[faces[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    outward = sign * normal[:, axis] > _ALIGN_TOL
    depth = sign * (x[faces[outward]][..., axis] - face)  # (n, 3)
    short = (depth < d - tol).any(axis=1)
    if not short.any():
        return
    names = {mesh.region(r).tag: r for r in pml.regions}
    tags = mesh.tetrahedron_tags
    k = int(np.flatnonzero(short)[0])
    low = float(depth[k].min())
    low_region = names[tags[owners[outward][k]]]
    tets = mesh.tetrahedra
    reach = (sign * (x[tets[side], axis] - face)).max(axis=1)
    top_region = names[tags[side[int(np.argmax(reach))]]]
    raise ValueError(
        f"perfectly matched layer {pml.regions!r}: its {label} side does not end "
        f"on one plane: it is {_length(d, mesh)} thick ({top_region!r}) but "
        f"{_length(low, mesh)} ({low_region!r}) somewhere. The stretch is "
        "normalized by the side's thickness, so a thinner part would miss the "
        "target reflection: end every layer region of a side on one plane."
    )


# --- model checks --------------------------------------------------------------


def missing_background(regions: Iterable[str], pml: PML | None) -> str:
    """The sentence the missing-material error adds for layer regions."""
    if pml is None or not set(regions) & set(pml.regions):
        return ""
    return (
        " A perfectly matched layer region takes its background material, "
        "usually gfem.Material() for air."
    )


def check_regions(
    mesh: Mesh, pml: PML, em_excluded: set[str], materials: dict[str, Any]
) -> None:
    """The layer's regions and backgrounds, before its geometry.

    Raises:
        RegionNotFoundError: A region is not in the mesh.
        ValueError: A region is a surface, or is excluded from
            electromagnetic solves.
        NotImplementedError: A background is a `MaterialField`, has a
            permittivity law, or principal axes off the coordinate axes.
    """
    for name in pml.regions:
        if name in mesh.surfaces and name not in mesh.volumes:
            raise ValueError(
                f"a perfectly matched layer is a volume region; {name!r} is a surface."
            )
        if name not in mesh.volumes:
            raise RegionNotFoundError(name, mesh.volumes, kind="volume")
        if name in em_excluded:
            raise ValueError(
                f"perfectly matched layer {name!r}: the region is excluded from "
                "electromagnetic solves, so the layer would stretch nothing. "
                "Remove it from excluded= or from the layer."
            )
        _check_background(name, materials[name])


def _check_background(name: str, material: Any) -> None:
    head = (
        f"perfectly matched layer {name!r}: its background must be a Material "
        "with constant, isotropic or axis-aligned properties"
    )
    if isinstance(material, MaterialField):
        raise NotImplementedError(
            f"{head}, got a MaterialField. Give the layer region a Material."
        )
    if not isinstance(material, Material):
        return
    if isinstance(material.eps_r, PermittivityLaw):
        raise NotImplementedError(
            f"{head}, got a permittivity law. Give it the constant permittivity "
            "of the band."
        )
    if material.axes is not None:
        axes = np.abs(np.asarray(material.axes, dtype=float))
        if ((axes > _ALIGN_TOL) & (axes < 1 - _ALIGN_TOL)).any():
            raise NotImplementedError(
                f"{head}, got principal axes off the coordinate axes. Rotate the "
                "model so that the layer's principal axes are x, y and z."
            )


def _touches(mesh: Mesh, surface: str, nodes: np.ndarray) -> int:
    """How many nodes of ``surface`` are nodes of the layer."""
    tri = np.unique(mesh.triangles[mesh.elements(surface)])
    return int(np.isin(tri, nodes).sum())


def _faces_layer(mesh: Mesh, surface: str, elements: np.ndarray) -> bool:
    """Whether a triangle of ``surface`` is a face of a layer tetrahedron."""
    n = mesh.triangle_neighbours[mesh.elements(surface)]
    return bool(np.isin(n, elements).any())


def wave_ports_in_layer(model: Any) -> list[WavePort]:
    """The wave ports with a face on a layer tetrahedron, in model order.

    Their port section is stretched by the layer.
    """
    layers = model._layers  # noqa: SLF001
    if layers is None:
        return []
    return [
        p
        for p in model.ports
        if isinstance(p, WavePort)
        and _faces_layer(model.mesh, p.surface, layers.elements)
    ]


def _normal(mesh: Mesh, surface: str) -> np.ndarray:
    """The unit normal of a planar surface (its least-variance direction)."""
    x = mesh.imported_nodes[np.unique(mesh.triangles[mesh.elements(surface)])]
    _, _, vt = np.linalg.svd(x - x.mean(axis=0))
    return vt[-1]


def check_surfaces(
    mesh: Mesh,
    layers: Layers,
    *,
    boundaries: tuple,
    ports: tuple,
    domain: EMDomain,
    pairs: tuple[Pairing, ...],
) -> None:
    """Boundaries, ports and periodic pairs against the layer.

    Raises:
        ValueError: `Absorbing` on a face of a layer tetrahedron.
        NotImplementedError: An `Impedance` or `Conductor` sheet in the
            layer, a `LumpedPort` or `FloquetPort` touching it, a `WavePort`
            touching it whose normal is not across its stretch axes, or a
            stretch along a periodic axis.
    """
    elements, nodes = layers.elements, layers.nodes
    for item in boundaries:
        what = type(item).__name__
        if isinstance(item, Absorbing) and _faces_layer(mesh, item.surface, elements):
            raise ValueError(
                f"Absorbing on {item.surface!r}: this surface bounds the perfectly "
                "matched layer; the layer absorbs the outgoing wave, so end it "
                "with the default PMC or a PEC: drop the Absorbing there."
            )
        if (
            isinstance(item, (Impedance, Conductor))
            and domain.interior(item.surface)
            and _faces_layer(mesh, item.surface, elements)
        ):
            raise NotImplementedError(
                f"{what} on {item.surface!r}: a sheet inside a perfectly matched "
                "layer is not supported yet. Keep sheets in the air box."
            )
    for port in ports:
        what = f"{type(port).__name__} {port.name!r}"
        surfaces = surfaces_of(port)
        touching = [s for s in surfaces if _touches(mesh, s, nodes)]
        if not touching:
            continue
        if isinstance(port, (LumpedPort, FloquetPort)):
            raise NotImplementedError(
                f"{what} on {touching[0]!r} touches the perfectly matched layer: "
                "a lumped port or a Floquet face must keep clear of the layer "
                "(a layer opposite a Floquet port is fine). Move the port into "
                "the air box."
            )
        if isinstance(port, WavePort):
            _check_wave_port(mesh, layers, port, what)
    for pair in pairs:
        a = np.asarray(pair.translation, dtype=float)
        a = a / np.linalg.norm(a)
        for axis in layers.axes:
            if abs(a[axis]) > _ALIGN_TOL:
                raise NotImplementedError(
                    f"perfectly matched layer: {pair.label} is periodic along "
                    f"{_AXES[axis]}, which is a stretch axis of the layer. A "
                    "layer in a periodic cell must span the whole period and "
                    "stretch along a non-periodic axis only."
                )


def _check_wave_port(mesh: Mesh, layers: Layers, port: WavePort, what: str) -> None:
    tets = mesh.tetrahedra
    tri = np.unique(mesh.triangles[mesh.elements(port.surface)])
    near = np.isin(tets[layers.elements], tri).any(axis=1)
    normal = _normal(mesh, port.surface)
    for side in layers.sides:
        on = _side_mask(mesh, layers, side)
        if not (near & on).any() or abs(normal[side.axis]) <= _ALIGN_TOL:
            continue
        axis = _AXES[side.axis]
        along = abs(normal[side.axis]) >= 1 - _ALIGN_TOL
        how = (
            f"its normal is the stretch axis {axis}"
            if along
            else f"it is oblique to the stretch axis {axis}"
        )
        raise NotImplementedError(
            f"{what} on {port.surface!r} touches the perfectly matched layer's "
            f"{side.label} side, and {how}: a wave port may cross a side layer "
            "only with every stretch axis it meets in its plane. Put the port "
            "in the air box, across the layer's sides."
        )


def _side_mask(mesh: Mesh, layers: Layers, side: Side) -> np.ndarray:
    """Per layer tetrahedron, whether it lies beyond ``side``'s face."""
    x = mesh.imported_nodes[mesh.tetrahedra[layers.elements], side.axis]
    tol = REL_TOL * float(_shortest_edges(mesh)[layers.elements].min())
    return (side.sign * (x - side.face) >= -tol).all(axis=1) & (
        side.sign * (x - side.face) > tol
    ).any(axis=1)


def check_layer_nodes(model: Any, problem: Any) -> None:
    """Raise if a moved mesh puts physical tetrahedra past a stretched face.

    The stretch is fixed at the imported mesh (its interior box and
    thicknesses), so a vertex of an active tetrahedron outside the layer
    moved beyond a covered face would be stretched with the device
    material. Layer vertices may move inward: below its inner face the
    layer is its background. On a curved mesh the mid-edge nodes count as
    vertices. Only wave problems (driven, eigenmode and
    port mode) stretch: the others see the background and pass.

    Args:
        model: A model with concrete node coordinates.
        problem: The problem solved on ``model``.

    Raises:
        ValueError: A vertex of an active non-layer tetrahedron lies beyond
            a covered face by more than the node tolerance.
    """
    layers, mesh = model._layers, model.mesh  # noqa: SLF001
    if layers is None or not isinstance(problem, (Driven, Eigenmode, PortMode)):
        return
    nodes = np.asarray(mesh.nodes)
    if np.array_equal(nodes, mesh.imported_nodes):
        return
    active = model._domain.active  # noqa: SLF001
    is_layer = np.zeros(len(mesh.tetrahedra), dtype=bool)
    is_layer[layers.elements] = True
    inner = np.flatnonzero(active & ~is_layer)
    tol = REL_TOL * float(_shortest_edges(mesh)[inner].min())
    corners = mesh.tetrahedra[inner]
    midnodes = mesh._topology.volume_midnodes  # noqa: SLF001
    if midnodes is not None:  # a curved mesh: the mid-edge nodes count too
        corners = np.concatenate([corners, midnodes[inner]], axis=1)
    p = nodes[corners]  # (n, 4 or 10, 3)
    names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    for side in layers.sides:
        past = (side.sign * (p[:, :, side.axis] - side.face) > tol).any(axis=1)
        if not past.any():
            continue
        tags = mesh.tetrahedron_tags[inner[past]]
        region = names[int(tags[0])]
        n = int(np.count_nonzero(tags == tags[0]))
        raise ValueError(
            f"The moved mesh puts {n} tetrahedra of region {region!r} past the "
            f"inner face {side.label} ({_AXES[side.axis]} = {side.face:.6g} m) of "
            "the perfectly matched layer, whose stretch is fixed at the imported "
            "mesh, so the stretch would act on them. Keep the moving geometry "
            "inside the layer's interior box (gfem.shape.boundary_morph holds "
            "the layer), or remesh."
        )


def describe(layers: Layers, mesh: Mesh) -> list[str]:
    """One line per side for a model's repr: ``+x: d = 0.6 um (pml)``."""
    return [
        f"{s.label}: d = {_length(s.thickness, mesh)} ({', '.join(s.regions)})"
        for s in layers.sides
    ]


# --- readers ---------------------------------------------------------------------


def layer_tetrahedra(mesh: Mesh, pml: PML | None) -> np.ndarray | None:
    """Per tetrahedron (gfem order), whether it lies in a layer region.

    ``None`` without a layer. The energy readers zero these tetrahedra:
    their element arrays hold the background material while the field
    there is the stretched one, so the integral would depend on the
    layer's thickness and target reflection, not on the device.
    """
    if pml is None:
        return None
    tags = [mesh.region(r).tag for r in pml.regions]
    return np.isin(mesh.tetrahedron_tags, tags)


def physical_regions(regions: Iterable[str], pml: PML | None) -> tuple[str, ...]:
    """``regions`` without the layer regions, in their order."""
    layer = set() if pml is None else set(pml.regions)
    return tuple(r for r in regions if r not in layer)


def refuse_estimate(
    subject: str, pml: PML, problem: str | None = None
) -> MeshRefinementUnsupportedError:
    """The refusal of a wave estimator or refinement on a model with a layer.

    Args:
        subject: What refuses, e.g. ``"gfem.FieldEstimator"``.
        pml: The model's layer.
        problem: The problem type refined, e.g. ``"Driven"``, or ``None``.
    """
    what = "a model" if problem is None else f"a gfem.{problem} problem on a model"
    return MeshRefinementUnsupportedError(
        "perfectly matched layer",
        f"{subject} has no path for {what} with a perfectly matched layer "
        f"(regions {', '.join(map(repr, pml.regions))}): its indicators weigh "
        "the layer with its background material, not the stretched one, so "
        "they would mark the layer and its inner face. Refine the mesh in "
        "your mesher, or refine the same device closed by gfem.Absorbing and "
        "solve the refined mesh with the layer.",
    )


# --- the stretch of a solve ------------------------------------------------------

#: :math:`k_\text{ref} d` below which a side is electrically thin.
THIN = 2 * math.pi / 100


@dataclasses.dataclass(frozen=True, eq=False)
class Stretch:
    r"""A layer's stretch for one solve: its geometry at a reference frequency.

    Attributes:
        layers: The layer geometry (`derive_layers`).
        target_reflection: :math:`R_0`.
        reference_frequency: :math:`f_\text{ref}`, Hz.
    """

    layers: Layers
    target_reflection: float
    reference_frequency: float

    @property
    def k_ref(self) -> float:
        r""":math:`k_\text{ref} = 2\pi f_\text{ref}/c`, rad/m (vacuum)."""
        return 2 * math.pi * self.reference_frequency / C0

    def stretch_maximum(self, side: Side) -> float:
        r"""The stretch maximum :math:`\sigma_\max` of ``side`` (`stretch_maximum`)."""
        return stretch_maximum(self.target_reflection, self.k_ref, side.thickness)

    @functools.cached_property
    def key(self) -> tuple:
        r"""What the stretched operator depends on, for cache keys.

        Per side its axis, sign, face, thickness and :math:`\sigma_\max`,
        then the reference frequency and the target reflection.
        """
        sides = tuple(
            (s.axis, s.sign, s.face, s.thickness, self.stretch_maximum(s))
            for s in self.layers.sides
        )
        return ("pml", sides, self.reference_frequency, self.target_reflection)

    def warn_thin(self) -> None:
        r"""Warn `ThinLayerWarning` for a side with :math:`k_\text{ref}d < 2\pi/100`.

        The warning points at the user's call into gfem.
        """
        for side in self.layers.sides:
            kd = self.k_ref * side.thickness
            if kd < THIN:
                warnings.warn(
                    f"perfectly matched layer, {side.label} side "
                    f"({', '.join(side.regions)}): it is "
                    f"{side.thickness * self.reference_frequency / C0:.3g} "
                    "reference wavelengths thick (k_ref d = "
                    f"{kd:.3g} < 2 pi / 100 at {self.reference_frequency:.6g} Hz), "
                    f"so its stretch maximum {self.stretch_maximum(side):.3g} makes "
                    "the operator badly conditioned and loses digits. "
                    "Make the layer about half a wavelength thick.",
                    ThinLayerWarning,
                    stacklevel=_mpi.outside_gfem(),
                )


def reference_frequency(pml: PML, frequencies: Iterable[float]) -> float:
    """The layer's reference frequency: pinned, else the lowest of ``frequencies``.

    ``frequencies`` is the whole solve's grid (a frequency worker's parent's).
    """
    if pml.reference_frequency is not None:
        return pml.reference_frequency
    return float(min(frequencies))


def stretch(model: Any, frequencies: Iterable[float]) -> Stretch | None:
    """The stretch of ``model``'s layer for a solve over ``frequencies``.

    ``None`` without a layer.
    """
    pml, layers = model.pml, model._layers  # noqa: SLF001
    if pml is None or layers is None:
        return None
    return Stretch(layers, pml.target_reflection, reference_frequency(pml, frequencies))


def refuse_bddc(pml: PML) -> ValueError:
    """The refusal of an explicit ``linear_solver="bddc_gmres"`` with a layer."""
    return ValueError(
        f"linear_solver='bddc_gmres' cannot solve a model with a perfectly matched "
        f"layer ({', '.join(map(repr, pml.regions))}): its stretch stalls the "
        "BDDC-preconditioned GMRES. Use linear_solver='pmg_gmres', which solves "
        "the layer exactly inside an iterative solve, or 'direct' (or 'auto', "
        "which picks one of them), or end the air box with gfem.Absorbing faces "
        "instead of the layer."
    )


def field_stretch(model: Any, field: Any) -> Stretch | None:
    """The stretch a driven ``field`` of ``model`` was solved with.

    The field's own reference frequency, else the layer's pinned one, else
    the field's frequency (a field formed outside a solve).
    """
    f_ref = field._reference_frequency  # noqa: SLF001
    if f_ref is None:
        f_ref = float(np.real(np.asarray(field.omega))) / (2 * math.pi)
    return stretch(model, [f_ref])
