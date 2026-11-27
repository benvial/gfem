"""The one place gfem imports gmsh.

gmsh is the optional extra ``gfem[gmsh]``: importing gfem never needs it, and
this module imports it only when a CAD feature asks, so a missing package or a
broken system library fails with the remedy instead of at ``import gfem``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import itertools
import math
import tempfile
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import numpy as np

from gfem.mesh import _TRI_MID_FROM_MESHIO, LENGTH_UNITS, Mesh, _read_gmsh

#: Debian/Ubuntu packages the PyPI gmsh wheel loads at import (OpenGL and X11).
_APT_PACKAGES = "libglu1-mesa libxcursor1 libxft2 libxinerama1 libgl1 libgomp1"

#: gmsh's element types of a 3-node and of a 6-node triangle.
_TRIANGLE = 2
_TRIANGLE6 = 9
_KINDS = {0: "point", 1: "curve", 2: "surface"}


def import_gmsh() -> ModuleType:
    """Import and return gmsh, with an actionable error when it cannot load.

    A missing package raises :class:`ImportError` (a test may skip on it); a
    package whose shared library fails to load raises :class:`OSError`, a
    different type so a runner without the system libraries fails loudly
    instead of skipping.
    """
    try:
        return importlib.import_module("gmsh")
    except ImportError as exc:
        raise ImportError(
            "gfem.shape.cad_morph needs gmsh, an optional dependency: "
            "pip install 'gfem[gmsh]'"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"gmsh is installed but its library failed to load ({exc}). "
            "The PyPI gmsh wheel needs system OpenGL/X11 libraries; on "
            f"Debian/Ubuntu: apt-get install {_APT_PACKAGES}."
        ) from exc


# --- the gmsh session --------------------------------------------------------

#: Export options set around the write and restored: MSH 4.1 keeps the node
#: numbering the reader relies on (gmsh 5 defaults to 4.2), binary keeps every
#: coordinate exact, only the grouped elements are saved, the coordinates are
#: not scaled (the node map compares them with the model's) and no parametric
#: coordinates are appended to the node block.
_EXPORT_OPTIONS = {
    "Mesh.MshFileVersion": 4.1,
    "Mesh.Binary": 1,
    "Mesh.SaveAll": 0,
    "Mesh.ScalingFactor": 1,
    "Mesh.SaveParametric": 0,
}

#: A projected node that moves by more than this many steps did not follow
#: its entity: the closest point landed elsewhere. A node on its entity moves
#: by at most the entity's speed times the step, a few steps for any design
#: whose speed is a few lengths per unit; a hop to another entity is a
#: distance of the model's features, thousands of steps of 1e-5 of its size.
_JUMP = 1e3

#: The node map is trusted when gmsh's tag - 1 lands within this fraction of
#: the model's size of the node's coordinates. The file holds gmsh's doubles
#: and the scaling adds a rounding of 1e-16 relative, so a true map is off by
#: 1e-15 of the size at most; a wrong one by a mesh spacing, 1e-2 of it.
_MAP_TOLERANCE = 1e-12

_model_names = itertools.count()


class _Models:
    """The gmsh models gfem adds, removed when the session ends."""

    def __init__(self, gmsh: ModuleType) -> None:
        self._gmsh = gmsh
        self._names: list[str] = []

    def add(self) -> str:
        """A new, empty, current model with a name no model has."""
        taken = set(self._gmsh.model.list())
        name = f"gfem_cad_{next(_model_names)}"
        while name in taken:
            name = f"gfem_cad_{next(_model_names)}"
        self._gmsh.model.add(name)
        self._names.append(name)
        return name

    def drop(self, name: str) -> None:
        """Remove one of gfem's models; the current model is then unset."""
        self._gmsh.model.setCurrent(name)
        self._gmsh.model.remove()
        self._names.remove(name)

    def clear(self) -> None:
        for name in list(self._names):
            self.drop(name)


@contextlib.contextmanager
def _session(gmsh: ModuleType) -> Generator[_Models]:
    """A gmsh session: the user's, left as found, or a quiet one of gfem's own.

    With no session open, gfem initializes one that reads no configuration
    file (the build function is then the whole recipe, whatever ``~/.gmshrc``
    says) and does not take over SIGINT (Ctrl-C stays a `KeyboardInterrupt`),
    and finalizes it. With the user's session open, gfem adds models of its
    own, removes them, restores the user's current model, and touches no
    option but the export options, which it restores.
    """
    if not gmsh.isInitialized():
        gmsh.initialize(readConfigFiles=False, interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            yield _Models(gmsh)
        finally:
            gmsh.finalize()
        return
    current = gmsh.model.getCurrent()
    models = _Models(gmsh)
    try:
        yield models
    finally:
        models.clear()
        gmsh.model.setCurrent(current)


@contextlib.contextmanager
def _export_options(gmsh: ModuleType) -> Generator[None]:
    saved = {key: gmsh.option.getNumber(key) for key in _EXPORT_OPTIONS}
    try:
        for key, value in _EXPORT_OPTIONS.items():
            gmsh.option.setNumber(key, value)
        yield
    finally:
        for key, value in saved.items():
            gmsh.option.setNumber(key, value)


@contextlib.contextmanager
def _mesh_order(gmsh: ModuleType, *, curved: bool) -> Generator[None]:
    """Second-order elements with untangling around the mesh generation, restored.

    ``Mesh.HighOrderOptimize 1`` untangles the curved elements gmsh would
    otherwise leave folded (the reader refuses those); a level the build chose
    (2 to 4) is kept.
    """
    if not curved:
        yield
        return
    keys = ("Mesh.ElementOrder", "Mesh.HighOrderOptimize")
    saved = {key: gmsh.option.getNumber(key) for key in keys}
    try:
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber(
            "Mesh.HighOrderOptimize", max(saved["Mesh.HighOrderOptimize"], 1)
        )
        yield
    finally:
        for key, value in saved.items():
            gmsh.option.setNumber(key, value)


def _check_order(gmsh: ModuleType) -> None:
    """Refuse a build that left a mesh element order other than 1.

    Raises:
        ValueError: ``Mesh.ElementOrder`` is not 1 after the build.
    """
    order = gmsh.option.getNumber("Mesh.ElementOrder")
    if order != 1:
        msg = (
            f"The build function (or the open gmsh session) left "
            f"Mesh.ElementOrder at {order:g}. gfem sets the element order "
            "itself: remove the option from the build, and pass curved=True "
            "to mesh curved (second-order) elements."
        )
        raise ValueError(msg)


def _run_build(build: Callable[..., object], params: Mapping[str, float]) -> None:
    """``build(**params)`` in the current model; a failure names the parameters."""
    try:
        build(**params)
    except Exception as exc:
        exc.add_note(f"gfem.shape.cad_morph called build with {dict(params)}")
        raise


def _check_volumes(gmsh: ModuleType, length_unit: str) -> None:
    """Refuse a model with no volume, or a volume in no volume group.

    gmsh leaves a volume outside every physical group out of the mesh, a hole
    whose walls would take the default outer condition, and a region name is
    the user's to declare.
    """
    volumes = [tag for _, tag in gmsh.model.getEntities(3)]
    if not volumes:
        msg = (
            "The build function made no 3D entity: cad_morph meshes volumes "
            "(a 2D model is not supported). Draw the model with "
            "gmsh.model.occ (addBox, extrude, fragment, ...) and call "
            "gmsh.model.occ.synchronize() before returning."
        )
        raise ValueError(msg)
    grouped: set[int] = set()
    for dim, group in gmsh.model.getPhysicalGroups(3):
        grouped.update(gmsh.model.getEntitiesForPhysicalGroup(dim, group))
    loose = [tag for tag in volumes if tag not in grouped]
    if loose:
        where = ", ".join(
            f"volume {tag} centred at ({_format(_centre(gmsh, tag))})" for tag in loose
        )
        msg = (
            f"The build function left {where} ({length_unit}) in no volume "
            "physical group. gmsh would leave it out of the mesh, a hole whose "
            "walls take the default outer condition. Add it to a group with "
            "gmsh.model.addPhysicalGroup(3, [...], name=...), or remove it from "
            "the model."
        )
        raise ValueError(msg)


def _format(point: np.ndarray) -> str:
    return ", ".join(f"{c:.4g}" for c in point)


def _centre(gmsh: ModuleType, tag: int) -> np.ndarray:
    box = np.asarray(gmsh.model.getBoundingBox(3, tag))
    return (box[:3] + box[3:]) / 2


# --- topology and classification --------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Signature:
    """What must not change between the builds: entities, boundaries, groups."""

    entities: list[tuple[int, int]]
    boundary: dict[tuple[int, int], tuple[int, ...]]
    groups: dict[tuple[int, int], tuple[str, tuple[int, ...]]]

    def difference(self, other: _Signature) -> str | None:
        """What differs from ``other``, or None."""
        if self.entities != other.entities:
            gone = sorted(set(self.entities) - set(other.entities))
            new = sorted(set(other.entities) - set(self.entities))
            return f"its entities differ (missing {gone}, new {new}) as (dim, tag)"
        for entity, boundary in self.boundary.items():
            if other.boundary[entity] != boundary:
                return (
                    f"the boundary of entity {entity} differs "
                    f"({list(boundary)} against {list(other.boundary[entity])})"
                )
        if self.groups != other.groups:
            return "the entities of a physical group differ"
        return None


def _signature(gmsh: ModuleType) -> _Signature:
    entities = sorted(gmsh.model.getEntities())
    boundary = {
        entity: tuple(
            sorted(
                tag
                for _, tag in gmsh.model.getBoundary(
                    [entity], combined=False, oriented=False
                )
            )
        )
        for entity in entities
        if entity[0] > 0
    }
    groups = {
        (dim, group): (
            gmsh.model.getPhysicalName(dim, group),
            tuple(sorted(gmsh.model.getEntitiesForPhysicalGroup(dim, group))),
        )
        for dim, group in gmsh.model.getPhysicalGroups()
    }
    return _Signature(entities, boundary, groups)


@dataclasses.dataclass(frozen=True)
class _Entity:
    """The mesh nodes classified on one geometric entity, copied out of gmsh."""

    dim: int
    tag: int
    tags: np.ndarray
    coords: np.ndarray


def _classify(gmsh: ModuleType) -> list[_Entity]:
    """Every node of dimension 0 to 2, once, on the entity it is classified on."""
    found = []
    for dim in (0, 1, 2):
        for _, tag in gmsh.model.getEntities(dim):
            tags, coords, _ = gmsh.model.mesh.getNodes(
                dim, tag, includeBoundary=False, returnParametricCoord=False
            )
            if len(tags):
                found.append(
                    _Entity(
                        dim,
                        tag,
                        np.asarray(tags, dtype=np.int64),
                        np.reshape(coords, (-1, 3)),
                    )
                )
    return found


def _surface_triangles(gmsh: ModuleType) -> tuple[np.ndarray, np.ndarray | None]:
    """Node tags of the triangles of every surface entity, grouped or not.

    Returns:
        ``(corners, midnodes)``: the corners, ``(n, 3)``, and for second-order
        triangles the mid-edge nodes in gfem's order (column ``k`` on the edge
        opposite corner ``k``), ``(n, 3)``; ``None`` for first-order ones.
    """
    corners, mids = [], []
    for _, tag in gmsh.model.getEntities(2):
        types, _, nodes = gmsh.model.mesh.getElements(2, tag)
        for kind, tags in zip(types, nodes, strict=True):
            if kind == _TRIANGLE:
                corners.append(np.asarray(tags, dtype=np.int64).reshape(-1, 3))
            elif kind == _TRIANGLE6:
                six = np.asarray(tags, dtype=np.int64).reshape(-1, 6)
                corners.append(six[:, :3])
                mids.append(six[:, 3:][:, _TRI_MID_FROM_MESHIO])
    out = np.concatenate(corners) if corners else np.empty((0, 3), np.int64)
    return out, (np.concatenate(mids) if mids else None)


def _project(gmsh: ModuleType, entities: list[_Entity]) -> list[np.ndarray]:
    """Each entity's nodes at their closest points on the same entity of the model."""
    out = []
    for e in entities:
        if e.dim == 0:
            point = np.reshape(gmsh.model.getValue(0, e.tag, []), (-1, 3))
            out.append(np.broadcast_to(point, e.coords.shape))
        else:
            closest, _ = gmsh.model.getClosestPoint(e.dim, e.tag, e.coords.ravel())
            out.append(np.reshape(closest, (-1, 3)))
    return out


def default_step(size: float) -> float:
    r"""The central-difference half-step for a parameter in model units.

    :math:`h = 2^{\operatorname{round}(\log_2(\varepsilon^{1/3} L))}` balances
    the second-order truncation of the difference against the
    :math:`\varepsilon L / h` round-off of the coordinates, and a power of two
    keeps :math:`p_0 \pm h` free of extra rounding. It is floored at
    :math:`2^{-17}` model units: the kernel treats points closer than
    :math:`10^{-7}` as one, and :math:`2^{-17} \approx 7.6\times10^{-6}` is 76
    times that, so a small device drawn in large units does not step below it.
    """
    step = 2.0 ** round(math.log2(np.finfo(float).eps ** (1 / 3) * size))
    return max(step, 2.0**-17)


# --- the node map ------------------------------------------------------------


def node_map(
    imported: np.ndarray,
    scale: float,
    tags: np.ndarray,
    coords: np.ndarray,
    size: float,
) -> np.ndarray:
    r"""The index in the read mesh of each gmsh node: tag - 1, or its coordinates.

    The reader keeps the file's node order and gmsh writes tags 1..n in order,
    so :math:`\mathrm{tag} - 1` is the index; it is checked against the coordinates, and
    when it fails the nodes are matched by exactly equal coordinates (the
    binary file holds the doubles gmsh had, and both sides are scaled the same
    way), refusing when that is ambiguous or incomplete too.

    Args:
        imported: The read mesh's node coordinates, metres.
        scale: Metres per model unit.
        tags: gmsh node tags.
        coords: Their coordinates in model units, ``(len(tags), 3)``.
        size: The model's size in model units, for the tolerance.

    Raises:
        ValueError: No consistent map exists.
    """
    index = tags - 1
    inside = (index >= 0) & (index < len(imported))
    if inside.all():
        gap = np.abs(imported[index] / scale - coords).max(initial=0.0)
        if gap <= _MAP_TOLERANCE * size:
            return index
    row = np.dtype((np.void, 3 * 8))
    # + 0.0 turns -0.0 into 0.0, which compare equal as floats but not as bytes
    keys = np.ascontiguousarray(imported + 0.0).view(row).ravel()
    wanted = np.ascontiguousarray(coords * scale + 0.0).view(row).ravel()
    order = np.argsort(keys)
    sorted_keys = keys[order]
    duplicated = bool((sorted_keys[1:] == sorted_keys[:-1]).any())
    where = np.minimum(np.searchsorted(sorted_keys, wanted), len(keys) - 1)
    if duplicated or not (sorted_keys[where] == wanted).all():
        reason = (
            "two mesh nodes share coordinates"
            if duplicated
            else ("a node of the CAD model is not in the mesh file")
        )
        msg = (
            "The nodes of the CAD model do not map onto the mesh read from "
            "gmsh: gmsh's node tags are not the mesh's indices (tag - 1 is off "
            f"by more than {_MAP_TOLERANCE:g} of the model size) and {reason}. "
            "Every surface of the model must bound or lie in a volume of a "
            "physical group, so that its nodes are written."
        )
        raise ValueError(msg)
    return order[where]


# --- the build ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CadMesh:
    """A meshed CAD model and its design velocities, as plain arrays.

    Attributes:
        mesh: The mesh at the first parameters, in memory (no source file).
        velocity: ``(3 n_nodes, n_params)``, model units per parameter unit:
            the velocity of the nodes classified on points, curves and
            surfaces, zero elsewhere.
        dim: ``(n_nodes,)``, the dimension of the entity each node is
            classified on (0 point, 1 curve, 2 surface, 3 anywhere else).
        slide_triangles: Triangles (mesh node indices) of every surface entity.
        slide_midnodes: Their mid-edge nodes, ``(n, 3)`` (column ``k`` on the
            edge opposite corner ``k``), or ``None`` on a first-order mesh.
        size: The model's bounding-box diagonal in model units.
        steps: The half-step used per parameter.
        scale: Metres per model unit.
    """

    mesh: Mesh
    velocity: np.ndarray
    dim: np.ndarray
    slide_triangles: np.ndarray
    slide_midnodes: np.ndarray | None
    size: float
    steps: tuple[float, ...]
    scale: float


def build_cad(
    build: Callable[..., object],
    p0: Mapping[str, float],
    length_unit: str,
    step: Mapping[str, float] | None = None,
    *,
    curved: bool = False,
) -> CadMesh:
    r"""Mesh ``build(**p0)`` and take each parameter's design velocity.

    Every node of the mesh is classified on a point, curve, surface or volume
    of the model. The model is rebuilt, geometry only, at
    :math:`p_0 \pm h e_k` and each classified node is projected on the same
    entity there; the central difference of the projections is the velocity.
    A node in a volume's interior gets none. Everything is copied out of gmsh
    before this returns, and the session is closed (or gfem's models removed).

    Args:
        build: Draws the model in the current gmsh model. It must not set
            ``Mesh.ElementOrder``: gfem sets it.
        p0: Parameter values, in the build's units.
        length_unit: The unit of the model's coordinates.
        step: Half-steps overriding the default, per parameter.
        curved: Mesh with second-order elements (``Mesh.ElementOrder 2`` and
            ``Mesh.HighOrderOptimize 1``, restored afterwards; a level of 2 to
            4 set by the build is kept). The mid-edge nodes are classified and
            projected on the CAD like the corners.

    Raises:
        ImportError: gmsh is not installed.
        OSError: gmsh's library cannot load.
        ValueError: No volume, a volume in no group, a topology change or a
            jump within a step, a build that sets ``Mesh.ElementOrder``, or nodes
            the reader did not keep.
    """
    gmsh = import_gmsh()
    names = tuple(p0)
    scale = LENGTH_UNITS[length_unit]
    with tempfile.TemporaryDirectory() as tmp, _session(gmsh) as models:
        first = models.add()
        _run_build(build, p0)
        _check_order(gmsh)
        _check_volumes(gmsh, length_unit)
        signature = _signature(gmsh)
        box = np.asarray(gmsh.model.getBoundingBox(-1, -1))
        size = float(np.linalg.norm(box[3:] - box[:3]))
        steps = tuple(float((step or {}).get(n, default_step(size))) for n in names)
        with _mesh_order(gmsh, curved=curved):
            gmsh.model.mesh.generate(3)
        entities = _classify(gmsh)
        reference = _Reference(gmsh, models, build, p0, first, signature, entities)
        triangles, triangle_mids = _surface_triangles(gmsh)
        path = Path(tmp) / "cad.msh"
        with _export_options(gmsh):
            gmsh.write(str(path))
        sides = [
            [_rebuilt(reference, name, sign * h) for sign in (+1, -1)]
            for name, h in zip(names, steps, strict=True)
        ]
        topology = dataclasses.replace(_read_gmsh(path, length_unit), source=None)
    mesh = Mesh(jnp.asarray(topology.imported_nodes), topology)
    imported = np.asarray(topology.imported_nodes)
    tags = np.concatenate([e.tags for e in entities])
    coords = np.concatenate([e.coords for e in entities])
    index = node_map(imported, scale, tags, coords, size)
    velocity = np.zeros((len(imported), 3, len(names)))
    for k, (name, h, (plus, minus)) in enumerate(zip(names, steps, sides, strict=True)):
        # the floating-point difference actually taken, not 2 h
        width = (p0[name] + h) - (p0[name] - h)
        velocity[index, :, k] = (np.concatenate(plus) - np.concatenate(minus)) / width
    lookup = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    lookup[tags] = index
    dim = np.full(len(imported), 3, dtype=np.int64)
    dim[index] = np.concatenate([np.full(len(e.tags), e.dim) for e in entities])
    return CadMesh(
        mesh=mesh,
        velocity=velocity.reshape(3 * len(imported), len(names)),
        dim=dim,
        slide_triangles=lookup[triangles],
        slide_midnodes=None if triangle_mids is None else lookup[triangle_mids],
        size=size,
        steps=steps,
        scale=scale,
    )


@dataclasses.dataclass(frozen=True)
class _Reference:
    """The first model (its name, topology and classified nodes) and its build."""

    gmsh: ModuleType
    models: _Models
    build: Callable[..., object]
    p0: Mapping[str, float]
    model: str
    signature: _Signature
    entities: list[_Entity]


def _rebuilt(reference: _Reference, name: str, offset: float) -> list[np.ndarray]:
    """The classified nodes projected on the model rebuilt at ``p0[name] + offset``.

    Raises:
        ValueError: The topology changed, or a node jumped.
    """
    gmsh, models = reference.gmsh, reference.models
    h = abs(offset)
    entities = reference.entities
    side = "+h" if offset > 0 else "-h"
    params = {**reference.p0, name: reference.p0[name] + offset}
    model = models.add()
    try:
        _run_build(reference.build, params)
        changed = reference.signature.difference(_signature(gmsh))
        if changed is not None:
            msg = (
                f"The build function changed topology within p0 +- h for "
                f"parameter {name!r} at {side} (h = {abs(offset):.3g}): {changed}. "
                "Pass a smaller step={...: h} for it, or use a build whose "
                "topology does not change there (no face or edge appears, "
                "vanishes or merges)."
            )
            raise ValueError(msg)
        projected = _project(gmsh, entities)
    finally:
        models.drop(model)
        gmsh.model.setCurrent(reference.model)
    for e, x in zip(entities, projected, strict=True):
        move = np.linalg.norm(x - e.coords, axis=1).max(initial=0.0)
        if move > _JUMP * h:
            msg = (
                f"A node of the CAD model moved by {move:.3g}, more than "
                f"{_JUMP:g} steps, when parameter {name!r} changed by "
                f"{side} (h = {h:.3g}). Either its closest point on "
                f"{_KINDS[e.dim]} {e.tag} is a different place, so the "
                "difference is not a velocity (pass a smaller step={...: h} "
                "for it, or check the build for a feature that jumps), or the "
                f"parameter moves the model faster than {_JUMP:g} lengths per "
                "unit (rescale it so that a unit moves the model by about a "
                "unit)."
            )
            raise ValueError(msg)
    return projected
