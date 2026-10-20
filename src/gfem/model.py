"""The immutable `Model`: mesh, materials, boundaries and ports."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Iterable, Mapping
from typing import Any

import jax

from gfem._boundaries import (
    PEC,
    PMC,
    Absorbing,
    Boundary,
    Conductor,
    Impedance,
    LumpedPort,
    Port,
    WavePort,
    surfaces_of,
)
from gfem._domain import EMDomain, em_domain, enclosed_entities
from gfem._materials import Material, MaterialField
from gfem._sheet import SheetField
from gfem.errors import RegionNotFoundError
from gfem.mesh import Mesh

__all__ = [
    "PEC",
    "PMC",
    "Absorbing",
    "Boundary",
    "Conductor",
    "EnclosedBy",
    "Impedance",
    "LumpedPort",
    "Material",
    "MaterialField",
    "Model",
    "Port",
    "SheetField",
    "WavePort",
    "enclosed_by",
]

_BOUNDARY_TYPES = (PEC, PMC, Absorbing, Conductor, Impedance)
_PORT_TYPES = (LumpedPort, WavePort)
_PHYSICS_NAMES = {"em": "electromagnetic"}


@dataclasses.dataclass(frozen=True)
class EnclosedBy:
    """An ``excluded=`` entry: every volume entity enclosed by some surfaces.

    Build it with `enclosed_by`.

    Attributes:
        surfaces: The surface names that together enclose the volumes.
    """

    surfaces: tuple[str, ...]

    def __repr__(self) -> str:
        return f"enclosed_by({', '.join(repr(s) for s in self.surfaces)})"


jax.tree_util.register_static(EnclosedBy)


def enclosed_by(*surfaces: str) -> EnclosedBy:
    """Select, for ``Model(excluded=...)``, the volumes enclosed by surfaces.

    Every gmsh volume entity (elementary volume) whose whole boundary lies on
    the named surfaces is excluded, whatever region it is tagged with; the
    rest of its region keeps its material. Use it when the mesh tags a meshed
    conductor with the name of the dielectric around it (gsim does), so the
    conductor has no region name of its own: name the conductor's shell, for
    example ``enclosed_by("metal_xy", "metal_z")``.

    Args:
        *surfaces: Surface region names.

    Raises:
        ValueError: No surface was given.
        TypeError: A surface is not a name.
    """
    if not surfaces:
        raise ValueError("enclosed_by needs at least one surface name.")
    for s in surfaces:
        if not isinstance(s, str):
            raise TypeError(f"enclosed_by takes surface names, got {s!r}.")
    return EnclosedBy(tuple(surfaces))


Excluded = str | EnclosedBy


def _check_volume(mesh: Mesh, name: str, remedy: str) -> None:
    if name in mesh.volumes:
        return
    if name in mesh.surfaces:
        raise ValueError(f"{name!r} is a surface, not a volume region. {remedy}")
    raise RegionNotFoundError(name, mesh.volumes, kind="volume")


def _check_surface(mesh: Mesh, name: str, what: str) -> None:
    if name in mesh.surfaces:
        return
    if name in mesh.volumes:
        raise ValueError(f"{what}: {name!r} is a volume, not a surface region.")
    raise RegionNotFoundError(name, mesh.surfaces, kind="surface")


def _as_tuple(value: Any, what: str) -> tuple:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise TypeError(f"{what} must be a list or tuple, got {type(value).__name__}.")


@jax.tree_util.register_pytree_with_keys_class
class Model:
    """Everything physical about a device: mesh, materials, boundaries, ports.

    A model is independent of the problem type: one model feeds driven,
    eigenmode, electrostatic and port mode problems. It is validated on
    construction and immutable; derive variants with `replace` and
    `with_material`. It is a JAX pytree whose leaves are the mesh node
    coordinates and every `MaterialField` array, so `jax.grad` of an
    objective reaches them.

    Regions are addressed by gmsh physical-group name. Every volume region
    gets a material or is excluded. Exterior surfaces of the
    electromagnetic solve domain that no boundary condition or port names
    are PMC (n x H = 0) for electromagnetic problems; `default_pmc` lists
    them.

    Args:
        mesh: The mesh.
        materials: A `Material` or `MaterialField` per volume region name.
        boundaries: Boundary conditions (`PEC`, `PMC`, `Absorbing`,
            `Conductor`, `Impedance`). At most one electromagnetic condition
            per surface.
        ports: `LumpedPort` and `WavePort` values; their order is the order
            of the network parameters.
        excluded: Volumes removed from the solve domain of electromagnetic
            problems, typically meshed conductors whose shell carries a
            `Conductor` or `PEC`: volume region names, or
            `enclosed_by(*surfaces)`. An excluded region has no material.

    Raises:
        RegionNotFoundError: A name is not a region of the mesh.
        ValueError: The model is physically inconsistent (a volume without a
            material, two conditions on one surface, an absorbing boundary on
            an interior surface, ...); the message gives the remedy.
        TypeError: An argument has the wrong type.
    """

    mesh: Mesh
    materials: Mapping[str, Material | MaterialField]
    boundaries: tuple[Boundary, ...]
    ports: tuple[Port, ...]
    excluded: tuple[Excluded, ...]
    _domain: EMDomain

    def __init__(
        self,
        mesh: Mesh,
        *,
        materials: Mapping[str, Material | MaterialField],
        boundaries: Iterable[Boundary] = (),
        ports: Iterable[Port] = (),
        excluded: Iterable[Excluded] = (),
    ) -> None:
        """Validate and build the model (see the class docstring)."""
        self._build(
            mesh=mesh,
            materials=materials,
            boundaries=boundaries,
            ports=ports,
            excluded=excluded,
            domain=None,
        )

    def _build(self, *, mesh, materials, boundaries, ports, excluded, domain) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError(
                f"Model needs a gfem.Mesh, got {type(mesh).__name__}; read one "
                "with gfem.Mesh.from_gmsh(path)."
            )
        excluded = _as_tuple(excluded, "excluded")
        if domain is None:
            domain = em_domain(mesh, _excluded_entities(mesh, excluded))
            if not domain.active.any():
                raise ValueError(
                    "Every volume is excluded, so there is nothing left to solve "
                    "on. Exclude only the meshed conductors."
                )
        set_ = object.__setattr__
        set_(self, "mesh", mesh)
        set_(self, "excluded", excluded)
        set_(self, "_domain", domain)
        set_(self, "materials", _materials(mesh, materials, self._fully_excluded()))
        set_(self, "boundaries", _items(boundaries, _BOUNDARY_TYPES, "boundaries"))
        set_(self, "ports", _items(ports, _PORT_TYPES, "ports"))
        self._check_surfaces()

    # --- validation -----------------------------------------------------------

    def _fully_excluded(self) -> set[str]:
        """Volume regions with no element left in the solve domain."""
        active = self._domain.active
        return {
            name
            for name in self.mesh.volumes
            if not active[self.mesh.elements(name)].any()
        }

    def _check_surfaces(self) -> None:
        mesh, domain = self.mesh, self._domain
        names = [p.name for p in self.ports]
        for name in names:
            if names.count(name) > 1:
                raise ValueError(
                    f"Port name {name!r} is used twice; port names must be unique."
                )
        taken: dict[tuple[str, str], list[Boundary | Port]] = {}
        for item in (*self.boundaries, *self.ports):
            what = type(item).__name__
            for surface in surfaces_of(item):
                _check_surface(mesh, surface, what)
                if not domain.touches(surface):
                    raise ValueError(
                        f"{what} on {surface!r}: the surface does not border the "
                        "electromagnetic solve domain (it lies inside excluded "
                        "volumes). Remove it, or exclude fewer volumes."
                    )
                taken.setdefault((item._physics, surface), []).append(item)  # noqa: SLF001
            if isinstance(item, (Absorbing, WavePort)) and domain.interior(
                item.surface
            ):
                raise ValueError(
                    f"{what} on {item.surface!r}: this surface is interior (it has "
                    "the solve domain on both sides), but an absorbing boundary or "
                    "a wave port must be an exterior surface of the "
                    "electromagnetic solve domain."
                )
            if isinstance(item, (Conductor, Impedance)) and item.fill is not None:
                self._check_sheet(item)
            if isinstance(item, WavePort):
                for edge in item.pec_edges:
                    _check_surface(mesh, edge, f"WavePort {item.name!r} pec_edges")
        for (physics, surface), items in taken.items():
            if len(items) > 1:
                raise ValueError(
                    f"Surface {surface!r} has more than one "
                    f"{_PHYSICS_NAMES[physics]} boundary condition or port: "
                    f"{', '.join(repr(i) for i in items)}. Keep one."
                )

    def _check_sheet(self, item: Conductor | Impedance) -> None:
        """A sheet field: on an interior surface, one fill value per face."""
        what = type(item).__name__
        surface, field = item.surface, item.fill
        if field is None:  # pragma: no cover - the caller checks
            return
        if field.surface is not None and field.surface != surface:
            raise ValueError(
                f"{what} on {surface!r} carries a SheetField built for surface "
                f"{field.surface!r}; attach it to a boundary on that surface."
            )
        n0, n1, _ = self._domain.sides[surface]
        if n0 or n1:
            raise ValueError(
                f"{what} on {surface!r}: a SheetField needs an interior surface, "
                "with the electromagnetic solve domain on both sides of every "
                f"face, but {n0 + n1} face(s) of {surface!r} are on the domain's "
                "exterior (the mesh boundary or the shell of an excluded volume). "
                "There, fill 0 would not be a void (it would leave a PMC wall or "
                "a hole in a conductor). Put the design on an interior sheet."
            )
        expected = len(self.mesh.elements(surface))
        if field.n_faces != expected:
            raise ValueError(
                f"{what} on {surface!r}: its SheetField has {field.n_faces} fill "
                f"values, but {surface!r} has {expected} faces: give one value per "
                f"face, in the order of mesh.elements({surface!r})."
            )

    # --- public views ---------------------------------------------------------

    @property
    def port_names(self) -> tuple[str, ...]:
        """Port names, in network-parameter order."""
        return tuple(p.name for p in self.ports)

    @property
    def excluded_entities(self) -> tuple[tuple[str, int], ...]:
        """``(region name, gmsh entity number)`` of each excluded volume entity."""
        names = {r.tag: r.name for r in self.mesh.regions if r.dim == 3}
        return tuple((names[tag], ent) for tag, ent in self._domain.excluded)

    @property
    def default_pmc(self) -> tuple[str, ...]:
        """Exterior surfaces that are PMC by default in electromagnetic problems.

        These are the surfaces of the electromagnetic solve domain's exterior
        (the mesh boundary and the shells of excluded volumes) that no
        boundary condition or port names.
        """
        named = {
            s for item in (*self.boundaries, *self.ports) for s in surfaces_of(item)
        }
        return tuple(
            s for s in self.mesh.surfaces if s not in named and self._domain.exterior(s)
        )

    # --- derived models -------------------------------------------------------

    def replace(self, **changes: Any) -> Model:
        """A new model with some fields changed, validated again.

        The fields are ``mesh``, ``materials``, ``boundaries``, ``ports`` and
        ``excluded``. Shape optimization passes ``mesh=moved_mesh``.

        Raises:
            TypeError: An unknown field name.
        """
        fields = ("mesh", "materials", "boundaries", "ports", "excluded")
        unknown = sorted(set(changes) - set(fields))
        if unknown:
            raise TypeError(
                f"Model.replace got unknown field(s) {', '.join(unknown)}; "
                f"use {', '.join(fields)}."
            )
        args = {f: changes.get(f, getattr(self, f)) for f in fields}
        same_domain = (
            isinstance(args["mesh"], Mesh)
            and args["mesh"]._topology is self.mesh._topology  # noqa: SLF001
            and "excluded" not in changes
        )
        new = object.__new__(Model)
        new._build(**args, domain=self._domain if same_domain else None)
        return new

    def with_material(self, region: str, material: Material | MaterialField) -> Model:
        """A new model with one volume region's material replaced.

        Raises:
            RegionNotFoundError: ``region`` is not a volume of the mesh.
        """
        _check_volume(self.mesh, region, "Materials are assigned to volumes.")
        return self.replace(materials={**self.materials, region: material})

    # --- pytree ---------------------------------------------------------------

    def tree_flatten_with_keys(self):
        """JAX pytree protocol: the mesh, materials, boundaries and ports."""
        key = jax.tree_util.GetAttrKey
        children = (
            (key("mesh"), self.mesh),
            (key("materials"), dict(self.materials)),
            (key("boundaries"), self.boundaries),
            (key("ports"), self.ports),
        )
        return children, (tuple(self.materials), self.excluded, self._domain)

    def tree_flatten(self):
        """JAX pytree protocol."""
        children, aux = self.tree_flatten_with_keys()
        return tuple(c for _, c in children), aux

    @classmethod
    def tree_unflatten(cls, aux, children) -> Model:
        """JAX pytree protocol (no validation: the structure is unchanged)."""
        order, excluded, domain = aux
        mesh, materials, boundaries, ports = children
        new = object.__new__(cls)
        set_ = object.__setattr__
        set_(new, "mesh", mesh)
        set_(new, "materials", types.MappingProxyType({k: materials[k] for k in order}))
        set_(new, "boundaries", boundaries)
        set_(new, "ports", ports)
        set_(new, "excluded", excluded)
        set_(new, "_domain", domain)
        return new

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "Model is immutable; derive a new one with model.replace(...) or "
            "model.with_material(...)."
        )

    def __repr__(self) -> str:
        lines = [f"Model({self.mesh!r}", "  materials:"]
        lines += [f"    {r}: {m!r}" for r, m in self.materials.items()]
        if self.excluded:
            ents = ", ".join(f"{r} entity {e}" for r, e in self.excluded_entities)
            spec = ", ".join(repr(x) for x in self.excluded)
            lines.append(f"  excluded (electromagnetic): {ents} (from {spec})")
        if self.boundaries:
            lines.append("  boundaries:")
            lines += [f"    {b!r}" for b in self.boundaries]
        if self.ports:
            lines.append("  ports:")
            lines += [f"    {p!r}" for p in self.ports]
        if self.default_pmc:
            lines.append(
                "  default PMC (electromagnetic problems): "
                + ", ".join(self.default_pmc)
            )
        return "\n".join(lines) + ")"


def _excluded_entities(mesh: Mesh, excluded: tuple) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for entry in excluded:
        if isinstance(entry, str):
            _check_volume(
                mesh,
                entry,
                "excluded= takes volume regions; to exclude the volume a surface "
                f"shell encloses, use gfem.enclosed_by({entry!r}, ...).",
            )
            idx = mesh.elements(entry)
            tag = mesh.region(entry).tag
            out |= {(tag, int(e)) for e in set(mesh.tetrahedron_entities[idx].tolist())}
        elif isinstance(entry, EnclosedBy):
            for s in entry.surfaces:
                _check_surface(mesh, s, repr(entry))
            found = enclosed_entities(mesh, entry.surfaces)
            if not found:
                raise ValueError(
                    f"{entry!r} encloses no volume entity: no gmsh volume has its "
                    "whole boundary on these surfaces. Name every surface of the "
                    "conductor's shell."
                )
            out |= found
        else:
            raise TypeError(
                f"excluded= takes volume region names or gfem.enclosed_by(...), "
                f"got {entry!r}."
            )
    return out


def _materials(
    mesh: Mesh, materials: Any, fully_excluded: set[str]
) -> Mapping[str, Material | MaterialField]:
    if not isinstance(materials, Mapping):
        raise TypeError(
            "materials must map volume region names to gfem.Material or "
            f"gfem.MaterialField, got {type(materials).__name__}."
        )
    for region, value in materials.items():
        _check_volume(mesh, region, "Materials are assigned to volumes.")
        if not isinstance(value, (Material, MaterialField)):
            raise TypeError(
                f"materials[{region!r}] must be a gfem.Material or MaterialField, "
                f"got {type(value).__name__}."
            )
        if region in fully_excluded:
            raise ValueError(
                f"Region {region!r} is excluded from the electromagnetic solve "
                "domain, so it cannot also have a material. Remove one of the two."
            )
        if isinstance(value, MaterialField):
            _check_field(mesh, region, value)
    missing = [
        v for v in mesh.volumes if v not in materials and v not in fully_excluded
    ]
    if missing:
        raise ValueError(
            f"No material for volume(s) {', '.join(repr(v) for v in missing)}. "
            "Give each volume a material, e.g. materials={..., "
            f"{missing[0]!r}: gfem.Material(eps_r=...)}}, or remove it from the "
            "solve with excluded=(...)."
        )
    return types.MappingProxyType(
        {v: materials[v] for v in mesh.volumes if v in materials}
    )


def _check_field(mesh: Mesh, region: str, field: MaterialField) -> None:
    if field.region is not None and field.region != region:
        raise ValueError(
            f"materials[{region!r}] is a MaterialField built for region "
            f"{field.region!r}; assign it to that region."
        )
    n = field.n_elements
    expected = len(mesh.elements(region))
    if n is not None and n != expected:
        raise ValueError(
            f"materials[{region!r}] is a MaterialField with {n} values, but region "
            f"{region!r} has {expected} tetrahedra: give one value per element, in "
            f"the order of mesh.elements({region!r})."
        )


def _items(values: Any, kinds: tuple[type, ...], what: str) -> tuple:
    items = _as_tuple(values, what)
    for item in items:
        if not isinstance(item, kinds):
            names = ", ".join(k.__name__ for k in kinds)
            raise TypeError(f"{what} takes {names}; got {item!r}.")
    return items
