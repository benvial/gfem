r"""What a curved (second-order) mesh lets each feature do.

A curved mesh keeps the mid-edge node of every edge (`gfem.Mesh.curved`). NGSolve
integrates on the quadratic map, but a feature that computes geometry in numpy
from the corner nodes (port sections, layer planes, point location, incident
loads, heat densities, surface integrals of a measurement) reads the affine
element. That is the same element only where the mid-edge nodes lie on their
chord midpoints, so each such feature is checked on the elements it reads: it
runs where they are straight and raises `gfem.CurvedMeshUnsupportedError`
naming the feature where one is curved.

An element is *straight* when every mid-edge node lies within
:math:`\tau = 64\,\varepsilon\,\max_i \lVert x_i \rVert_\infty` of the midpoint
of its chord, with :math:`\varepsilon = 2^{-52}` and the maximum over the
mesh's nodes. A mesh generator places a mid node by evaluating its geometry in
its own coordinates, so its round-off scales with the coordinates, not the edge:
on edges straight by construction the deviation was at most
:math:`2.4\,\varepsilon L` (:math:`L` the largest coordinate of the edge), and
:math:`\max_i \lVert x_i \rVert_\infty \ge L` and the factor 64 leave a margin
of 27. The quadratic map of a straight element differs from its affine map by
at most :math:`1.5\,\tau`, round-off, so what a feature computes there is what
NGSolve integrates. Real curvature has no gap above round-off, so no physical
threshold is used: an element near :math:`\tau` is affine to :math:`1.5\,\tau`
either way.

The predicate is computed once per topology from the imported nodes. A solve
on moved nodes tests the elements of the model's gated features again
(`check_moved`), because a morph may have curved them.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gfem._boundaries import (
    Absorbing,
    Conductor,
    FloquetPort,
    Impedance,
    LumpedPort,
    Periodic,
    WavePort,
    surfaces_of,
)
from gfem._checks import is_traced
from gfem._materials import Material, MaterialField
from gfem.errors import CurvedMeshUnsupportedError
from gfem.mesh import _TET_EDGES, _TRI_EDGES

if TYPE_CHECKING:
    from gfem.mesh import Mesh

_EPS = 2.0**-52
#: The factor of the round-off tolerance (see the module docstring).
_FACTOR = 64.0

_REMEDY_ORDER_1 = "or read a straight mesh (gmsh `Mesh.ElementOrder 1`)"


class Feature(NamedTuple):
    """One row of the table: what a gated feature is and why it refuses.

    Attributes:
        label: The feature as the capabilities page's Limits row names it.
        reason: Why the feature cannot read a curved element.
        remedy: What to change in the mesh so that the feature runs.
    """

    label: str
    reason: str
    remedy: str


_FLAT = "Make it a flat face bounded by straight lines"
_KEEP = "keep the elements it reads off curved surfaces"

#: The gated features by key: a feature runs when the elements it reads are
#: straight and refuses with `gfem.CurvedMeshUnsupportedError(feature=key)`
#: otherwise. `optical_force` refuses any curved mesh.
FEATURES: dict[str, Feature] = {
    "wave_port": Feature(
        "wave ports",
        "Its mode is solved on straight triangles of the port face.",
        f"{_FLAT}, or place the port where the mesh is straight",
    ),
    "lumped_port": Feature(
        "lumped ports",
        "Its element width, length and tangency come from straight triangles.",
        f"{_FLAT}",
    ),
    "floquet_port": Feature(
        "Floquet ports",
        "Its areas, element size and planes come from straight triangles.",
        f"{_FLAT}",
    ),
    "pml": Feature(
        "perfectly matched layers",
        "The layer's planes are read from the corners of its boundary faces.",
        "Make the layer a box shell with flat faces",
    ),
    "periodic": Feature(
        "periodic pairs",
        "The pairing and the planes of a pair are read from straight triangles.",
        "Make both faces of the pair flat",
    ),
    "sheet_frame": Feature(
        "sheet frames",
        "A sheet's local frame comes from the chord normal of each triangle.",
        "Make the sheet flat, or drop the anisotropy or the sheet field",
    ),
    "point_read": Feature(
        "point reads",
        "A point is located and its field read on the affine element.",
        "Place the point away from curved elements",
    ),
    "incident_field": Feature(
        "incident fields",
        "The load is interpolated on straight elements, which would undo the "
        "curved mesh at the scatterer.",
        "Pose the scatterer on a straight mesh",
    ),
    "heat_input": Feature(
        "heat inputs",
        "A power becomes a density by dividing by straight volumes and areas.",
        "Give a power_density, or " + _KEEP,
    ),
    "far_field": Feature(
        "far field",
        "The surface integral reads straight owner elements.",
        "Choose a side whose layer lies off curved surfaces",
    ),
    "optical_force": Feature(
        "optical force",
        "The stress integral and the torque read straight areas and normals, "
        "and the shape pass reads the corner lift.",
        "Solve on a straight mesh",
    ),
    "interface_participation": Feature(
        "interface participation",
        "The quadrature reads straight triangles and straight-located points.",
        "Choose surfaces " + _KEEP,
    ),
    "multipoles": Feature(
        "multipoles",
        "The far-field layer reads straight owner elements.",
        "Choose a scatterer whose truncation layer is straight",
    ),
    "current_moments": Feature(
        "current moments",
        "The moments integrate the scatterer on straight Jacobians.",
        "Pose the scatterer on a straight mesh",
    ),
    "cross_sections": Feature(
        "cross sections",
        "The power balance reads the truncation layer's straight elements.",
        "Choose a truncation layer off curved surfaces",
    ),
    "beam_coupling": Feature(
        "beam coupling",
        "The beam interpolant reads straight elements.",
        "Choose a side whose layer lies off curved surfaces",
    ),
}

# --- the entries -----------------------------------------------------------------

ACCEPTED = "accepted"
GATED = "gated"
REFUSED = "refused"

#: Every public function of `gfem` and how it treats a curved mesh: ``(status,
#: key)``. ``accepted``: runs as it is (NGSolve integrals, algebra on solved
#: values, or results that inherit the solve's gate); ``gated``: checks the
#: elements of ``key`` (`FEATURES`), directly or through its model or solve;
#: ``refused``: ``key`` names the refusal upstream of any curved element.
ENTRIES: dict[str, tuple[str, str | None]] = {
    "acousto_optic_shift": (REFUSED, "cross_section"),
    "band_diagram": (GATED, "periodic"),
    "beam_coupling": (GATED, "beam_coupling"),
    "brillouin_gain": (REFUSED, "cross_section"),
    "characteristic_impedance": (GATED, "wave_port"),
    "cross_sections": (GATED, "cross_sections"),
    "cuda_available": (ACCEPTED, None),
    "current_moments": (GATED, "current_moments"),
    "dissipated_power": (ACCEPTED, None),
    "electric_energy": (ACCEPTED, None),
    "electro_optic_shift": (GATED, "point_read"),
    "electromagnetic_heating": (GATED, "heat_input"),
    "element_mean": (ACCEPTED, None),
    "enclosed_by": (ACCEPTED, None),
    "error_estimate": (ACCEPTED, None),
    "far_field": (GATED, "far_field"),
    "fit_permittivity": (ACCEPTED, None),
    "free_carriers": (ACCEPTED, None),
    "heat_flow": (ACCEPTED, None),
    "heat_flow_per_length": (ACCEPTED, None),
    "interface_participation": (GATED, "interface_participation"),
    "isotropic_stiffness": (ACCEPTED, None),
    "joule_heating": (GATED, "heat_input"),
    "junction_coupling": (ACCEPTED, None),
    "kerr_coupling": (ACCEPTED, None),
    "kerr_nonlinearity": (ACCEPTED, None),
    "load_result": (ACCEPTED, None),
    "multipoles": (GATED, "multipoles"),
    "optical_force": (GATED, "optical_force"),
    "optomechanical_coupling": (REFUSED, "cross_section"),
    "optomechanical_overlap": (REFUSED, "cross_section"),
    "power_flow": (ACCEPTED, None),
    "purcell_factor": (GATED, "point_read"),
    "raman_gain": (ACCEPTED, None),
    "refine_mesh": (ACCEPTED, None),
    "region_mean": (ACCEPTED, None),
    "release_memory": (ACCEPTED, None),
    "rise_time": (ACCEPTED, None),
    "route_phase": (GATED, "point_read"),
    "second_harmonic": (ACCEPTED, None),
    "sfwm": (ACCEPTED, None),
    "solve": (GATED, "incident_field"),
    "spdc": (ACCEPTED, None),
    "strain": (REFUSED, "2d_mesh"),
    "thermo_optic_shift": (GATED, "point_read"),
    "three_wave_coupling": (ACCEPTED, None),
    "time_constant": (ACCEPTED, None),
    "tolerance_report": (ACCEPTED, None),
    "wave_vector_path": (GATED, "periodic"),
}

#: The model items that carry a gate, by class name.
ITEMS: dict[str, str] = {
    "WavePort": "wave_port",
    "LumpedPort": "lumped_port",
    "FloquetPort": "floquet_port",
    "PML": "pml",
    "Periodic": "periodic",
    "Conductor": "sheet_frame",
    "Impedance": "sheet_frame",
    "Absorbing": "sheet_frame",
    "HeatSource": "heat_input",
}

# --- the predicate ---------------------------------------------------------------

# Per topology, the (n_tet, 6) and (n_tri, 3) arrays of mid nodes off their chord
# midpoints, on the imported nodes.
_CACHE: weakref.WeakKeyDictionary[Any, tuple[np.ndarray, np.ndarray]] = (
    weakref.WeakKeyDictionary()
)


def _bent(nodes: np.ndarray, corners: np.ndarray, mids: np.ndarray, edges: Any):
    """Per element and edge: whether the mid node leaves its chord midpoint."""
    x = np.asarray(nodes, float)
    tol = _FACTOR * _EPS * float(np.max(np.abs(x)))
    cols = []
    for k, (a, b) in enumerate(edges):
        chord = 0.5 * (x[corners[:, a]] + x[corners[:, b]])
        cols.append(np.linalg.norm(x[mids[:, k]] - chord, axis=1) > tol)
    return np.stack(cols, axis=1)


def _bent_edges(mesh: Mesh, nodes: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    t = mesh._topology  # noqa: SLF001
    if t.volume_midnodes is None or t.surface_midnodes is None:
        return (
            np.zeros((len(t.volume_elements), 6), bool),
            np.zeros((len(t.surface_elements), 3), bool),
        )
    cached = nodes is None
    if cached and t in _CACHE:
        return _CACHE[t]
    x = t.imported_nodes if cached else nodes
    out = (
        _bent(x, t.volume_elements, t.volume_midnodes, _TET_EDGES),
        _bent(
            x, t.surface_elements, t.surface_midnodes, [tuple(e) for e in _TRI_EDGES]
        ),
    )
    for a in out:
        a.setflags(write=False)
    if cached:
        _CACHE[t] = out
    return out


def curved_tetrahedra(mesh: Mesh, nodes: np.ndarray | None = None) -> np.ndarray:
    """Per tetrahedron: whether a mid-edge node leaves its chord midpoint.

    Args:
        mesh: The mesh; a straight mesh has no curved element.
        nodes: Node coordinates to test instead of the imported ones (not cached).
    """
    return _bent_edges(mesh, nodes)[0].any(axis=1)


def curved_triangles(mesh: Mesh, nodes: np.ndarray | None = None) -> np.ndarray:
    """Per surface triangle: whether a mid-edge node leaves its chord midpoint."""
    return _bent_edges(mesh, nodes)[1].any(axis=1)


# --- the sets a feature reads ------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ReadSet:
    """The elements one gated feature reads.

    Attributes:
        key: The feature key (`FEATURES`).
        what: The item as a message names it, ``WavePort 'in'``.
        tets: Tetrahedra read whole.
        triangles: Surface triangles read.
        faces: Faces of tetrahedra read: owner tetrahedra and, per face, the
            three columns of their mid nodes (`_TET_EDGES`), shape ``(m, 3)``.
    """

    key: str
    what: str
    tets: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0, int))
    triangles: np.ndarray = dataclasses.field(default_factory=lambda: np.empty(0, int))
    faces: tuple[np.ndarray, np.ndarray] | None = None


_EDGE_COLUMN = {edge: k for k, edge in enumerate(_TET_EDGES)}


def _face_columns(face: Iterable[int]) -> list[int]:
    a, b, c = sorted(face)
    return [_EDGE_COLUMN[(a, b)], _EDGE_COLUMN[(a, c)], _EDGE_COLUMN[(b, c)]]


#: Mid-node columns of the face opposite each tetrahedron vertex.
_FACE_COLUMNS = np.array([_face_columns(set(range(4)) - {j}) for j in range(4)])


def _layer_faces(mesh: Mesh, layer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Faces of the ``layer`` tetrahedra that no other layer tetrahedron shares.

    Returns ``(owners, columns)``: the tetrahedra and the mid-node columns of
    each face's three edges.
    """
    tets = np.asarray(mesh.tetrahedra)[layer]
    local = np.arange(4)
    faces = np.sort(
        np.stack([np.delete(tets, j, axis=1) for j in range(4)], axis=1), axis=2
    ).reshape(-1, 3)
    _, inverse, counts = np.unique(
        faces, axis=0, return_inverse=True, return_counts=True
    )
    single = counts[inverse.ravel()] == 1
    owner = np.repeat(layer, 4)[single]
    opposite = np.tile(local, len(layer))[single]
    return owner, _FACE_COLUMNS[opposite]


def _count(mesh: Mesh, read: ReadSet, nodes: np.ndarray | None) -> tuple[int, int, Any]:
    """``(curved, total, first)``: curved elements among a set, and one of them."""
    bent_tet, bent_tri = _bent_edges(mesh, nodes)
    curved = 0
    total = 0
    first = None
    if len(read.tets):
        hit = bent_tet[read.tets].any(axis=1)
        total += len(hit)
        curved += int(hit.sum())
        if hit.any():
            first = ("tet", read.tets[np.argmax(hit)])
    if len(read.triangles):
        hit = bent_tri[read.triangles].any(axis=1)
        total += len(hit)
        curved += int(hit.sum())
        if hit.any() and first is None:
            first = ("tri", read.triangles[np.argmax(hit)])
    if read.faces is not None and len(read.faces[0]):
        owners, columns = read.faces
        hit = np.take_along_axis(bent_tet[owners], columns, axis=1).any(axis=1)
        total += len(hit)
        curved += int(hit.sum())
        if hit.any() and first is None:
            first = ("tet", owners[np.argmax(hit)])
    return curved, total, first


def _where(mesh: Mesh, first: Any, nodes: np.ndarray | None) -> str:
    kind, index = first
    x = np.asarray(mesh.imported_nodes if nodes is None else nodes, float)
    corners = (
        mesh.volume_elements[index] if kind == "tet" else mesh.surface_elements[index]
    )
    c = x[corners].mean(axis=0)
    return "(" + ", ".join(f"{v:.3g}" for v in c) + ") m"


def _all_curved(mesh: Mesh) -> bool:
    return bool(curved_tetrahedra(mesh).all())


def _message(mesh: Mesh, key: str, read: ReadSet, count: tuple[int, int, Any]) -> str:
    feature = FEATURES[key]
    curved, total, first = count
    head = f"{read.what} is not available on a curved (second-order) mesh"
    n = f"{curved} of its {total} elements are curved" if total > 1 else "it is curved"
    msg = (
        f"{head}: {n}, the first near {_where(mesh, first, None)}. "
        f"{feature.reason} {feature.remedy}, {_REMEDY_ORDER_1}."
    )
    if _all_curved(mesh):
        msg += (
            " Every tetrahedron of this mesh is curved: gmsh `Mesh.HighOrderOptimize"
            " 2` or `3` moves every mid-edge node, `0` or `1` keeps the elements off"
            " curved surfaces straight."
        )
    return msg


def refuse(mesh: Any, read: ReadSet) -> None:
    """Raise `gfem.CurvedMeshUnsupportedError` when ``read`` holds a curved element.

    A no-op on a straight mesh (and on anything that is not a mesh).

    Raises:
        CurvedMeshUnsupportedError: An element of the set is curved; its
            ``feature`` is the set's key.
    """
    if not getattr(mesh, "curved", False):
        return
    count = _count(mesh, read, None)
    if count[0]:
        raise CurvedMeshUnsupportedError(
            read.key, _message(mesh, read.key, read, count)
        )


def refuse_tets(mesh: Any, key: str, tets: Any, what: str | None = None) -> None:
    """`refuse` for the tetrahedra ``tets`` (indices or a boolean mask) of ``key``."""
    if not getattr(mesh, "curved", False):
        return
    tets = np.asarray(tets)
    if tets.dtype == bool:
        tets = np.flatnonzero(tets)
    refuse(mesh, ReadSet(key, what or f"gfem.{key}", tets=tets.astype(int)))


def refuse_triangles(
    mesh: Any, key: str, triangles: Any, what: str | None = None
) -> None:
    """`refuse` for surface ``triangles`` (indices or a mask) of ``key``."""
    if not getattr(mesh, "curved", False):
        return
    triangles = np.asarray(triangles)
    if triangles.dtype == bool:
        triangles = np.flatnonzero(triangles)
    refuse(mesh, ReadSet(key, what or f"gfem.{key}", triangles=triangles.astype(int)))


def refuse_surface(
    mesh: Any, key: str, triangles: Any, what: str | None = None
) -> None:
    """`refuse` for surface ``triangles`` and the tetrahedra that own them."""
    if not getattr(mesh, "curved", False):
        return
    triangles = np.asarray(triangles, dtype=int)
    owners = np.asarray(mesh.triangle_neighbours)[triangles].ravel()
    tets = np.unique(owners[owners >= 0])
    refuse(mesh, ReadSet(key, what or f"gfem.{key}", tets=tets, triangles=triangles))


def refuse_entry(mesh: Any, key: str, what: str | None = None) -> None:
    """Raise for a feature that refuses every curved mesh (`optical_force`)."""
    if not getattr(mesh, "curved", False):
        return
    feature = FEATURES[key]
    raise CurvedMeshUnsupportedError(
        key,
        f"{what or 'gfem.' + key} is not available on a curved (second-order) mesh. "
        f"{feature.reason} {feature.remedy[0].upper() + feature.remedy[1:]}, "
        f"{_REMEDY_ORDER_1}.",
    )


# --- the model's items -------------------------------------------------------------


def _named(item: Any) -> str:
    name = getattr(item, "name", None)
    kind = type(item).__name__
    if name is not None:
        return f"{kind} {name!r}"
    if isinstance(item, Periodic):
        return f"Periodic({item.surfaces!r})"
    return f"{kind} on {getattr(item, 'surface', None)!r}"


def _triangles(mesh: Mesh, names: Iterable[str]) -> np.ndarray:
    known = set(mesh.surfaces)
    parts = [np.asarray(mesh.elements(n)) for n in names if n in known]
    return np.concatenate(parts).astype(int) if parts else np.empty(0, int)


def _anisotropic(material: Any) -> bool:
    if isinstance(material, Material):
        return not material.isotropic
    return isinstance(material, MaterialField) and material.axes is not None


def _sets(
    mesh: Mesh,
    boundaries: Sequence[Any],
    ports: Sequence[Any],
    pml: Any,
    materials: Any,
) -> list[ReadSet]:
    """The element sets the model's gated items read."""
    out: list[ReadSet] = []
    for item in (*boundaries, *ports):
        if isinstance(item, WavePort):
            key = "wave_port"
        elif isinstance(item, LumpedPort):
            key = "lumped_port"
        elif isinstance(item, FloquetPort):
            key = "floquet_port"
        elif isinstance(item, Periodic):
            key = "periodic"
        elif (
            isinstance(item, (Conductor, Impedance)) and item.fill is not None
        ) or (
            isinstance(item, Absorbing) and _beside_anisotropic(mesh, item, materials)
        ):
            key = "sheet_frame"
        else:
            continue
        out.append(
            ReadSet(key, _named(item), triangles=_triangles(mesh, surfaces_of(item)))
        )
    if pml is not None:
        names = [r for r in pml.regions if r in mesh.volumes]
        if names:
            layer = np.concatenate([np.asarray(mesh.elements(r)) for r in names])
            owners, columns = _layer_faces(mesh, layer.astype(int))
            regions = ", ".join(map(repr, names))
            out.append(
                ReadSet(
                    "pml",
                    f"The perfectly matched layer {regions}",
                    faces=(owners, columns),
                )
            )
    return out


def _beside_anisotropic(mesh: Mesh, item: Absorbing, materials: Any) -> bool:
    if item.surface not in mesh.surfaces:
        return False
    return any(
        _anisotropic((materials or {}).get(v))
        for v in mesh.adjacent_volumes(item.surface)
    )


def refuse_model(
    mesh: Any,
    boundaries: Sequence[Any],
    ports: Sequence[Any],
    pml: Any,
    materials: Any,
) -> None:
    """Raise when a gated item of a model sits on curved elements.

    A no-op on a straight mesh. Items and names the model refuses itself are
    skipped here.

    Raises:
        CurvedMeshUnsupportedError: An item reads a curved element; the
            ``feature`` is the item's key.
    """
    if not getattr(mesh, "curved", False):
        return
    for read in _sets(mesh, boundaries, ports, pml, materials):
        refuse(mesh, read)


def check_moved(model: Any) -> None:
    """Raise when concrete moved nodes curve an element a gated item reads.

    A morph may bend elements it was meant to keep: the gated sets of the
    model's items are tested again on the current nodes, with the tolerance
    from them.

    Raises:
        ValueError: A moved mesh curves the elements of a gated item.
    """
    mesh = model.mesh
    if not mesh.curved or is_traced(mesh.nodes):
        return
    nodes = np.asarray(mesh.nodes)
    if np.array_equal(nodes, mesh.imported_nodes):
        return
    for read in _sets(mesh, model.boundaries, model.ports, model.pml, model.materials):
        count = _count(mesh, read, nodes)
        if count[0]:
            curved, total, first = count
            raise ValueError(
                f"The moved mesh curves {curved} of the {total} elements of "
                f"{read.what} ({FEATURES[read.key].label}), the first near "
                f"{_where(mesh, first, nodes)}: its mid-edge nodes left their chord "
                "midpoints. Hold its surface in the morph (move its mid-edge nodes "
                "with the corners), or remesh."
            )


# --- solve-level inputs -------------------------------------------------------------


def refuse_incident(model: Any, problem: Any) -> None:
    """Raise when an incident-field source meets a curved physical element.

    Raises:
        CurvedMeshUnsupportedError: ``problem`` has an incident-field source
            and a tetrahedron of the solve domain outside the layer is curved.
    """
    mesh = model.mesh
    if not getattr(mesh, "curved", False):
        return
    from gfem._sources import INCIDENT

    sources = getattr(problem, "sources", ())
    sources = (sources,) if not isinstance(sources, (list, tuple)) else sources
    named = [s for s in sources if isinstance(s, INCIDENT)]
    if not named:
        return
    from gfem._pml import layer_tetrahedra

    layer = layer_tetrahedra(mesh, model.pml)
    physical = np.asarray(model._domain.active)  # noqa: SLF001
    if layer is not None:
        physical = physical & ~layer
    refuse_tets(
        mesh, "incident_field", physical, f"The incident field {named[0].name!r}"
    )
