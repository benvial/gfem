"""The immutable `Model`: mesh, materials, boundaries and ports."""

from __future__ import annotations

import dataclasses
import itertools
import math
import types
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from gfem._background import Background, background_of
from gfem._boundaries import (
    PEC,
    PMC,
    Absorbing,
    Boundary,
    Conductor,
    Convection,
    FixedTemperature,
    FloquetPort,
    HeatFlux,
    Impedance,
    LumpedPort,
    Periodic,
    Polarity,
    Port,
    WavePort,
    sheet_of,
    surfaces_of,
)
from gfem._checks import check_fixed, is_traced
from gfem._curved import refuse_model as refuse_curved
from gfem._dim2 import refuse_model
from gfem._domain import EMDomain, em_domain, enclosed_entities, physics_of
from gfem._floquet import (
    FloquetFace,
    floquet_face,
    floquet_surfaces,
    propagating_orders,
    surface_lead,
    wavenumber,
)
from gfem._materials import (
    Debye,
    DjordjevicSarkar,
    Drude,
    Lorentz,
    Material,
    MaterialField,
    PermittivityLaw,
    isotropic_stiffness,
)
from gfem._periodic import (
    Pairing,
    check_domain,
    check_nodes,
    check_pec,
    check_wave_vector,
    pairing,
)
from gfem._permittivity_fit import PermittivityFit, fit_permittivity
from gfem._pml import (
    PML,
    Layers,
    check_regions,
    check_surfaces,
    derive_layers,
    describe,
    missing_background,
)
from gfem._sheet import SheetField
from gfem.errors import RegionNotFoundError
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._field import Field

__all__ = [
    "PEC",
    "PMC",
    "PML",
    "Absorbing",
    "Background",
    "Boundary",
    "Conductor",
    "Convection",
    "Debye",
    "DjordjevicSarkar",
    "Drude",
    "EnclosedBy",
    "FixedTemperature",
    "FloquetPort",
    "HeatFlux",
    "Impedance",
    "Lorentz",
    "LumpedPort",
    "Material",
    "MaterialField",
    "Model",
    "Periodic",
    "PermittivityFit",
    "PermittivityLaw",
    "Port",
    "SheetField",
    "WavePort",
    "enclosed_by",
    "fit_permittivity",
    "isotropic_stiffness",
]

_BOUNDARY_TYPES = (
    PEC,
    PMC,
    Absorbing,
    Conductor,
    Impedance,
    Periodic,
    FixedTemperature,
    Convection,
    HeatFlux,
)
#: The most periodic pairs a model takes (a 3D cell).
_MAX_PERIODIC_PAIRS = 3
#: Sine of the angle below which two translations count as parallel.
_PARALLEL_SINE = 1e-6
_PORT_TYPES = (LumpedPort, WavePort, FloquetPort)
_BLOCH_ANGLES = (
    "Model.bloch_wave_vector: theta and phi must be fixed numbers: the incidence "
    "of a Floquet port is a constant of the solve, not a design variable"
)
#: Each physics' key in ``excluded=`` and its name in messages, in order.
_PHYSICS_NAMES = {
    "em": "electromagnetic",
    "thermal": "thermal",
    "dc_conduction": "DC conduction",
    "mechanical": "mechanical",
}
_PHYSICS = tuple(_PHYSICS_NAMES)
_KEYS = ", ".join(repr(k) for k in _PHYSICS)


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


def check_surface(mesh: Mesh, name: Any, what: str, hint: str = "") -> None:
    """Check that ``name`` names a surface region of ``mesh``.

    Args:
        mesh: The mesh.
        name: The user's surface name.
        what: Who names it, opening the error messages.
        hint: The remedy appended when ``name`` is a volume.

    Raises:
        TypeError: ``name`` is not a string.
        ValueError: ``name`` is a volume region.
        RegionNotFoundError: The mesh has no region ``name``.
    """
    if not isinstance(name, str):
        raise TypeError(f"{what}: surface names are strings, got {name!r}.")
    if name in mesh.surfaces:
        return
    if name in mesh.volumes:
        raise ValueError(f"{what}: {name!r} is a volume, not a surface region.{hint}")
    raise RegionNotFoundError(name, mesh.surfaces, kind="surface")


def _as_tuple(value: Any, what: str) -> tuple:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise TypeError(f"{what} must be a list or tuple, got {type(value).__name__}.")


def _check_physics(physics: Any, what: str) -> str:
    if physics not in _PHYSICS_NAMES:
        raise ValueError(
            f"{what}: unknown physics {physics!r}; the physics keys are {_KEYS}."
        )
    return physics


def _excluded_by_physics(excluded: Any) -> tuple[tuple, ...]:
    """``excluded=`` as one tuple per physics, in `_PHYSICS` order.

    A list or tuple excludes for electromagnetic problems only.
    """
    if not isinstance(excluded, Mapping):
        return (_as_tuple(excluded, "excluded"), *(() for _ in _PHYSICS[1:]))
    for key in excluded:
        if key not in _PHYSICS_NAMES:
            raise ValueError(
                f"excluded= has the unknown physics key {key!r}; the keys are "
                f"{_KEYS}. A list or tuple excludes for electromagnetic problems "
                "only."
            )
    return tuple(_as_tuple(excluded.get(k, ()), f"excluded[{k!r}]") for k in _PHYSICS)


@jax.tree_util.register_pytree_with_keys_class
class Model:
    """Everything physical about a device: mesh, materials, boundaries, ports.

    A model is independent of the problem type: one model feeds driven,
    eigenmode, electrostatic and port mode problems. It is validated on
    construction and immutable; derive variants with `replace` and
    `with_material`. A `Model` is a JAX
    `pytree <https://docs.jax.dev/en/latest/pytrees.html>`__:
    `jax.grad` of an objective reaches the mesh node coordinates and every
    `MaterialField` array; everything else is fixed.

    Regions are addressed by gmsh physical-group name. Every volume region
    of the electromagnetic solve domain gets a material. Exterior surfaces of
    the electromagnetic solve domain that no boundary condition or port names
    are PMC (n x H = 0) for electromagnetic problems; `default_pmc` lists
    them. Each physics solves on its own domain (``excluded=``), and each
    boundary condition is checked against its own physics' domain: thermal
    conditions (`FixedTemperature`, `Convection`, `HeatFlux`) against the
    thermal domain. Exterior surfaces of the thermal domain that no thermal
    condition names are insulated; `default_insulated` lists them.

    A perfectly matched layer (``pml=``) is refused with a `ValueError` when
    a region is a surface or excluded from electromagnetic solves, a layer
    tetrahedron lies inside the interior box, a side does not end on one
    plane, the layer stretches along no axis, or an `Absorbing` sits on a
    face of a layer tetrahedron; with a `NotImplementedError` for a
    background that is a `MaterialField`, a permittivity law or rotated off
    the coordinate axes, a layer that is not a box shell, a sheet in the
    layer, a lumped port or a Floquet face touching it, a wave port whose
    normal is not across the stretch axes it meets, or a stretch along a
    periodic axis.

    Example:
        A two-port microstrip on a substrate::

            model = gfem.Model(
                gfem.Mesh.from_gmsh("filter.msh"),
                materials={"substrate": gfem.Material(eps_r=4.4)},
                boundaries=[gfem.PEC("ground"), gfem.Absorbing("outer")],
                ports=[
                    gfem.LumpedPort("p1", "port1", "+z"),
                    gfem.LumpedPort("p2", "port2", "+z"),
                ],
            )

    Args:
        mesh: The mesh.
        materials: A `Material` or `MaterialField` per volume region name.
        boundaries: Boundary conditions (`PEC`, `PMC`, `Absorbing`,
            `Conductor`, `Impedance`), periodic pairs (`Periodic`, one to
            three) and thermal conditions (`FixedTemperature`, `Convection`,
            `HeatFlux`). At most one condition or port per physics and
            surface, except several wave ports (`WavePort`), one mode each (a
            shared port surface); an electromagnetic and a thermal condition
            may share a surface.
        ports: `LumpedPort`, `WavePort` and `FloquetPort` values; their
            order is the order of the network parameters. Several Floquet
            ports share a Floquet surface, one (order, polarization) each.
        excluded: Volumes removed from the solve domain of electromagnetic
            problems, typically meshed conductors whose shell carries a
            `Conductor` or `PEC`: volume region names, or
            ``enclosed_by(*surfaces)`` (see `enclosed_by`). A mapping with
            the keys ``"em"``, ``"thermal"``, ``"dc_conduction"`` and
            ``"mechanical"`` sets the volumes removed for each physics (a
            missing key removes nothing); a list or tuple is the ``"em"``
            entry. The mechanical domain is the solid a piezoelectric
            problem solves the displacement on: every volume not excluded
            from it gives a `Material.stiffness` and a ``mass_density``, so
            exclude the air and the vacuum by name. A meshed
            conductor excluded from electromagnetic problems stays in the
            thermal domain (it is often the heater) and may carry a
            material; a region excluded from every physics has none.
        pml: A `PML` naming every perfectly matched layer region, or
            ``None``. Each layer region keeps its background material in
            ``materials``; the stretch axes and thicknesses are read from the
            mesh (see `PML`).
        background: The background stack (`Background`) of the model's
            incident fields, or one `Material` for a homogeneous one;
            ``None`` (the default) for a model without incident fields.
            Point dipoles do not need one.

    Raises:
        RegionNotFoundError: A name is not a region of the mesh.
        ValueError: The model is physically inconsistent (a volume without a
            material, two conditions of one physics on one surface (except
            several wave ports, one mode each), an absorbing boundary or a
            `Convection` on an interior surface, two touching
            `FixedTemperature` surfaces with different values, ...), an
            unknown physics key in ``excluded=``, or a `Periodic` pair's
            two meshes do not match under a translation, there are more than
            three pairs, or two translations are parallel; the message gives the remedy.
        TypeError: An argument has the wrong type.
        NotImplementedError: A `WavePort` shares a mesh node with a surface
            of a `Periodic` pair (for plane-wave incidence on a periodic
            cell, use `FloquetPort`); a `FloquetPort` surface whose rim lies
            on one pair only, with several regions behind it, or with a
            medium that is not lossless, isotropic and constant. On a 2D
            mesh, a part with no 2D path (lumped and Floquet ports,
            `Periodic`, sheets, thermal boundaries, `enclosed_by`, a
            ``voltage_path``, an anisotropic material); the message names
            the remedy.
        PortGeometryError: A `FloquetPort` surface that is not planar, or
            whose rim is not on the seams of two pairs over whole periods.

    A ``background`` raises as `Background` does: `TypeError` for a layer
    that is not a `Material`, `NotImplementedError` for one that is a
    `MaterialField`, has a permittivity law or is anisotropic.
    """

    mesh: Mesh
    materials: Mapping[str, Material | MaterialField]
    boundaries: tuple[Boundary, ...]
    ports: tuple[Port, ...]
    excluded: tuple[Excluded, ...]
    pml: PML | None
    background: Background | None
    _layers: Layers | None
    _excluded_by: tuple[tuple[Excluded, ...], ...]
    _domains: tuple[EMDomain, ...]
    _domain: EMDomain
    _periodic: tuple[Pairing, ...]

    def __init__(
        self,
        mesh: Mesh,
        *,
        materials: Mapping[str, Material | MaterialField],
        boundaries: Iterable[Boundary] = (),
        ports: Iterable[Port] = (),
        excluded: Iterable[Excluded] | Mapping[str, Iterable[Excluded]] = (),
        pml: PML | None = None,
        background: Background | Material | None = None,
    ) -> None:
        """Validate and build the model (see the class docstring)."""
        self._build(
            mesh=mesh,
            materials=materials,
            boundaries=boundaries,
            ports=ports,
            excluded=excluded,
            pml=pml,
            background=background,
            domains=None,
        )

    def _build(
        self,
        *,
        mesh,
        materials,
        boundaries,
        ports,
        excluded,
        pml,
        background,
        domains,
        layers=None,
    ) -> None:
        if not isinstance(mesh, Mesh):
            raise TypeError(
                f"Model needs a gfem.Mesh, got {type(mesh).__name__}; read one "
                "with gfem.Mesh.from_gmsh(path)."
            )
        if pml is not None and not isinstance(pml, PML):
            raise TypeError(
                "Model: pml takes one gfem.PML naming every layer region, got "
                f"{type(pml).__name__}; write pml=gfem.PML(('pml_x', 'pml_y'))."
            )
        by = _excluded_by_physics(excluded)
        if mesh.curved:
            # before any check that reads corner geometry; malformed arguments
            # are left to the checks below
            listed = (list, tuple)
            refuse_curved(
                mesh,
                tuple(boundaries) if isinstance(boundaries, listed) else (),
                tuple(ports) if isinstance(ports, listed) else (),
                pml,
                materials if isinstance(materials, dict) else {},
            )
        if mesh.dim == 2:
            items = (*_as_tuple(boundaries, "boundaries"), *_as_tuple(ports, "ports"))
            refuse_model(mesh, items, by, pml)
        if domains is None:
            # physics with the same exclusions share one domain
            keys = [frozenset(_excluded_entities(mesh, e)) for e in by]
            built = {k: em_domain(mesh, set(k)) for k in dict.fromkeys(keys)}
            domains = tuple(built[k] for k in keys)
            if not any(d.active.any() for d in domains):
                raise ValueError(
                    "Every volume is excluded from every physics, so there is "
                    "nothing left to solve on. Exclude only the meshed "
                    "conductors."
                )
        set_ = object.__setattr__
        set_(self, "mesh", mesh)
        set_(self, "excluded", by[0])
        set_(self, "_excluded_by", by)
        set_(self, "_domains", domains)
        set_(self, "_domain", domains[0])
        everywhere = set.intersection(*(self._fully_excluded(p) for p in _PHYSICS))
        set_(
            self,
            "materials",
            _materials(mesh, materials, everywhere, self._fully_excluded("em"), pml),
        )
        set_(self, "pml", pml)
        set_(self, "background", background_of(background))
        if pml is not None:
            check_regions(mesh, pml, self._fully_excluded("em"), dict(self.materials))
            if layers is None:
                layers = derive_layers(mesh, pml, self._domain.active)
        set_(self, "_layers", layers if pml is not None else None)
        set_(self, "boundaries", _items(boundaries, _BOUNDARY_TYPES, "boundaries"))
        set_(self, "ports", _items(ports, _PORT_TYPES, "ports"))
        self._check_surfaces()
        set_(self, "_periodic", self._check_periodic())
        if self._layers is not None:
            check_surfaces(
                mesh,
                self._layers,
                boundaries=self.boundaries,
                ports=self.ports,
                domain=self._domain,
                pairs=self._periodic,
            )

    # --- validation -----------------------------------------------------------

    def _domain_of(self, physics: str) -> EMDomain:
        """The solve domain of ``physics`` (a key of `_PHYSICS_NAMES`)."""
        return self._domains[_PHYSICS.index(physics)]

    def _fully_excluded(self, physics: str) -> set[str]:
        """Volume regions with no element left in the solve domain of ``physics``."""
        active = self._domain_of(physics).active
        return {
            name
            for name in self.mesh.volumes
            if not active[self.mesh.elements(name)].any()
        }

    def _check_surfaces(self) -> None:
        mesh = self.mesh
        names = [p.name for p in self.ports]
        for name in names:
            if names.count(name) > 1:
                raise ValueError(
                    f"Port name {name!r} is used twice; port names must be unique."
                )
        taken: dict[tuple[str, str], list[Boundary | Port]] = {}
        for item in (*self.boundaries, *self.ports):
            what = type(item).__name__
            physics = physics_of(item)
            domain = self._domain_of(physics)
            for surface in surfaces_of(item):
                check_surface(mesh, surface, what)
                if not domain.touches(surface):
                    raise ValueError(_off_domain(what, surface, physics))
                taken.setdefault((physics, surface), []).append(item)
            if isinstance(item, Convection) and domain.interior(item.surface):
                raise ValueError(
                    f"Convection on {item.surface!r}: this surface is interior (it "
                    "has the thermal solve domain on both sides), but a "
                    "convection boundary must be an exterior surface of the "
                    "thermal solve domain: an interior face has no ambient. Use "
                    "a HeatFlux or a FixedTemperature there, or exclude the "
                    'volume on one side with excluded={"thermal": (...)}.'
                )
            if isinstance(item, (Absorbing, WavePort, FloquetPort)) and (
                domain.interior(item.surface)
            ):
                raise ValueError(
                    f"{what} on {item.surface!r}: this surface is interior (it has "
                    "the solve domain on both sides), but an absorbing boundary, "
                    "a wave port or a Floquet port must be an exterior surface of "
                    "the electromagnetic solve domain."
                )
            sheet = sheet_of(item)
            if sheet is not None:
                (surface,) = surfaces_of(item)
                self._check_sheet(what, surface, sheet)
            if isinstance(item, WavePort):
                for edge in item.pec_edges:
                    check_surface(mesh, edge, f"WavePort {item.name!r} pec_edges")
                if item.polarity is not None:
                    _check_polarity(item.name, item.polarity, mesh.dim)
        _check_one_per_surface(taken)
        self._check_fixed_temperatures()

    def _check_fixed_temperatures(self) -> None:
        """Two `FixedTemperature` surfaces with different values share no node.

        A traced value is compared on the host, on every thermal forward.
        """
        fixed = [b for b in self.boundaries if isinstance(b, FixedTemperature)]
        mesh = self.mesh
        nodes = {
            b.surface: np.unique(mesh.triangles[mesh.elements(b.surface)])
            for b in fixed
        }
        for i, b in enumerate(fixed):
            for a in fixed[:i]:
                if is_traced((a.temperature, b.temperature)):
                    continue
                if a.temperature == b.temperature:
                    continue
                shared = np.intersect1d(nodes[a.surface], nodes[b.surface]).size
                if shared:
                    raise ValueError(
                        f"FixedTemperature on {a.surface!r} ({a.temperature} K) "
                        f"and on {b.surface!r} ({b.temperature} K) share {shared} "
                        "mesh nodes, where the temperature would have two values: "
                        "merge them, or separate them in the geometry."
                    )

    def _check_periodic(self) -> tuple[Pairing, ...]:
        """Pair the nodes of every `Periodic` and check the pairs."""
        pairs = [b for b in self.boundaries if isinstance(b, Periodic)]
        if len(pairs) > _MAX_PERIODIC_PAIRS:
            raise ValueError(
                f"A cell has at most three periodic pairs, got {len(pairs)}: "
                f"{', '.join(repr(p) for p in pairs)}."
            )
        pec = tuple(
            dict.fromkeys(b.surface for b in self.boundaries if isinstance(b, PEC))
        )
        for item in pairs:
            for surface in item.surfaces:
                if self._domain.interior(surface):
                    raise ValueError(
                        f"Periodic on {surface!r}: this surface is interior (it has "
                        "the solve domain on both sides), but a periodic pair "
                        "joins two exterior surfaces of the electromagnetic solve "
                        "domain."
                    )
        out = []
        for item in pairs:
            p = pairing(self.mesh, *item.surfaces)
            check_domain(self.mesh, p, self._domain)
            check_pec(self.mesh, p, pec)
            out.append(p)
        _check_independent(out)
        self._check_wave_ports(out)
        self._check_floquet_ports(out)
        if out and not is_traced(self.mesh.nodes):
            check_nodes(self.mesh.nodes, out)
        return tuple(out)

    def _check_wave_ports(self, pairs: list[Pairing]) -> None:
        """No wave port shares a mesh node with a surface of a periodic pair.

        Raises:
            NotImplementedError: A wave port touches a pair, even at one node.
        """
        if not pairs:
            return
        mesh = self.mesh
        for port in self.ports:
            if not isinstance(port, WavePort):
                continue
            nodes = np.unique(mesh.triangles[mesh.elements(port.surface)])
            for p in pairs:
                shared = np.intersect1d(
                    nodes, np.concatenate([p.first_nodes, p.second_nodes])
                )
                if shared.size:
                    raise NotImplementedError(
                        f"WavePort {port.name!r} touches {p.label} at {shared.size} "
                        "nodes: a wave port's mode solve cannot be periodic. Move "
                        "the port off the pair's surfaces; for plane-wave incidence "
                        "on the open face of a periodic cell, use gfem.FloquetPort."
                    )

    def _check_floquet_ports(self, pairs: list[Pairing]) -> None:
        """Each Floquet surface is an open face of the cell (`floquet_face`).

        Raises:
            ValueError, PortGeometryError, NotImplementedError: As
                `gfem._floquet.floquet_face`, naming the surface's first port.
        """
        for surface, ports in floquet_surfaces(self.ports).items():
            floquet_face(self, surface, surface_lead(surface, ports), pairs)

    def _check_sheet(self, what: str, surface: str, field: SheetField) -> None:
        """A sheet field: on an interior surface, one fill value per face."""
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
    def periodic_translations(self) -> tuple[tuple[float, float, float], ...]:
        """The translation a of each `Periodic`, in m, in ``boundaries`` order.

        Measured from the imported mesh, from the pair's first surface
        (``surfaces[0]``) to its second; it stays fixed when the nodes move.
        A Bloch wave vector k (rad/m) gives the pair the phase exp(-i k·a).
        Empty without a periodic pair.
        """
        return tuple(p.translation for p in self._periodic)

    @property
    def reciprocal_lattice(self) -> tuple[tuple[float, float, float], ...]:
        r"""The reciprocal lattice vector :math:`\mathbf b_i` of each `Periodic`, rad/m.

        One per pair, in ``boundaries`` order, inside the span of the
        `periodic_translations` :math:`\mathbf a_j`, with
        :math:`\mathbf b_i \cdot \mathbf a_j = 2\pi\delta_{ij}`: the rows of
        :math:`B = 2\pi (A A^T)^{-1} A`, where the rows of :math:`A` are the
        translations. A Bloch wave vector is
        :math:`\mathbf k = \sum_i c_i \mathbf b_i` in reduced coordinates
        :math:`c_i`, and :math:`\mathbf k` and :math:`\mathbf k + \mathbf b_i`
        give every pair the same phase. Empty without a periodic pair.
        """
        if not self._periodic:
            return ()
        a = np.asarray(self.periodic_translations, dtype=float)
        b = 2 * np.pi * np.linalg.solve(a @ a.T, a)
        return tuple((float(x), float(y), float(z)) for x, y, z in b)

    @property
    def excluded_entities(self) -> tuple[tuple[str, int], ...]:
        """``(region name, gmsh entity number)`` of each EM-excluded volume entity.

        These are the entities excluded from electromagnetic problems; see
        `excluded_from` for the other physics.
        """
        return self._excluded_entities_of("em")

    def excluded_from(self, physics: str) -> tuple[Excluded, ...]:
        """The ``excluded=`` entries of one physics.

        Args:
            physics: ``"em"``, ``"thermal"``, ``"dc_conduction"`` or
                ``"mechanical"``.

        Raises:
            ValueError: An unknown physics.
        """
        _check_physics(physics, "Model.excluded_from")
        return self._excluded_by[_PHYSICS.index(physics)]

    def _excluded_entities_of(self, physics: str) -> tuple[tuple[str, int], ...]:
        """``(region name, gmsh entity number)`` excluded from ``physics``."""
        names = self.mesh._region_names()  # noqa: SLF001
        domain = self._domain_of(physics)
        return tuple((names[tag], ent) for tag, ent in domain.excluded)

    def _default(self, physics: str) -> tuple[str, ...]:
        """Exterior surfaces of the ``physics`` domain no item of it names."""
        named = {
            s
            for item in (*self.boundaries, *self.ports)
            if physics_of(item) == physics
            for s in surfaces_of(item)
        }
        domain = self._domain_of(physics)
        return tuple(
            s for s in self.mesh.surfaces if s not in named and domain.exterior(s)
        )

    @property
    def default_pmc(self) -> tuple[str, ...]:
        """Exterior surfaces that are PMC by default in electromagnetic problems.

        These are the surfaces of the electromagnetic solve domain's exterior
        (the mesh boundary and the shells of excluded volumes) that no
        electromagnetic boundary condition or port names.
        """
        return self._default("em")

    @property
    def default_insulated(self) -> tuple[str, ...]:
        """Exterior surfaces that are insulated by default in thermal problems.

        These are the surfaces of the thermal domain's exterior (the mesh
        boundary, the end faces ``front`` and ``back`` of a slab
        cross-section, and the shells of volumes excluded from thermal
        problems) that no thermal condition names: no heat crosses them.
        """
        return self._default("thermal")

    def _face(self, surface: Any, owner: str) -> FloquetFace:
        check_surface(self.mesh, surface, owner)
        if not self._domain.exterior(surface):
            raise ValueError(
                f"{owner}: {surface!r} is not an exterior surface of the "
                "electromagnetic solve domain; name an open face of the cell."
            )
        return floquet_face(self, surface, f"{owner}: surface {surface!r}")

    def bloch_wave_vector(
        self, surface: str, frequency: float, *, theta: float, phi: float = 0.0
    ) -> tuple[float, float, float]:
        r"""The Bloch wave vector of a plane wave incident through a face, rad/m.

        :math:`k_0 n \sin\theta\, (\cos\phi\, \hat e_1 + \sin\phi\, \hat e_2)`,
        with :math:`n = \sqrt{\varepsilon_r \mu_r}` of the medium behind
        ``surface`` and :math:`(\hat e_1, \hat e_2)` its lattice
        frame (`gfem.conventions`). Pass it as ``wave_vector`` of
        `gfem.Driven`: it is the specular order's in-plane wave vector, the
        same on every face of the cell.

        Args:
            surface: An open face of the cell (a Floquet surface, with or
                without ports yet).
            frequency: Hz.
            theta: The angle from the face's inward normal, rad,
                :math:`|\theta| \le \pi/2`.
            phi: The azimuth from :math:`\hat e_1` toward :math:`\hat e_2`,
                rad.

        Returns:
            ``(kx, ky, kz)`` in rad/m, in model coordinates.

        Raises:
            NonDifferentiableError: An argument is traced by JAX.
            ValueError: A non-positive frequency, an angle out of range, or
                ``surface`` is not an open face of the cell.
            PortGeometryError, NotImplementedError: As for a `FloquetPort`
                on ``surface``.
        """
        owner = "Model.bloch_wave_vector"
        check_fixed(frequency, f"{owner}: frequency must be a fixed number")
        check_fixed((theta, phi), _BLOCH_ANGLES)
        f, th, ph = (float(v) for v in (frequency, theta, phi))
        if not (math.isfinite(f) and f > 0):
            raise ValueError(f"{owner}: frequency must be > 0 Hz, got {frequency!r}.")
        if not (math.isfinite(th) and abs(th) <= math.pi / 2 and math.isfinite(ph)):
            raise ValueError(
                f"{owner}: theta must be in [-pi/2, pi/2] rad and phi finite, got "
                f"theta={theta!r}, phi={phi!r} (angles are in radians)."
            )
        face = self._face(surface, owner)
        e1, e2, _ = face.frame
        k = float(wavenumber(f, face.eps_r, face.mu_r)) * math.sin(th)
        v = k * (math.cos(ph) * e1 + math.sin(ph) * e2)
        return (float(v[0]), float(v[1]), float(v[2]))

    def propagating_orders(
        self, surface: str, frequency: Any, *, wave_vector: Any = None
    ) -> tuple[tuple[int, int], ...]:
        """The diffraction orders that propagate behind a face.

        Orders (m, n) on the reciprocal lattice of the face's two pairs
        (`gfem.FloquetPort`) whose in-plane wave vector is shorter than the
        medium's wavenumber at some of the frequencies (strictly: a grazing
        order is not listed).

        Args:
            surface: An open face of the cell.
            frequency: One frequency or several, Hz.
            wave_vector: The problem's Bloch wave vector, rad/m, or ``None``
                (normal incidence).

        Returns:
            The orders, sorted by ``|m| + |n|``, then m, then n.

        Raises:
            NonDifferentiableError: An argument is traced by JAX.
            ValueError: A non-positive frequency, a wave vector off the
                pairs' span, or ``surface`` is not an open face of the cell.
        """
        owner = "Model.propagating_orders"
        check_fixed(frequency, f"{owner}: frequency must be fixed numbers")
        check_fixed(wave_vector, f"{owner}: wave_vector must be fixed numbers")
        f = np.atleast_1d(np.asarray(frequency, float)).ravel()
        if not (np.isfinite(f).all() and (f > 0).all()) or not f.size:
            raise ValueError(f"{owner}: frequencies must be > 0 Hz, got {frequency!r}.")
        if wave_vector is not None:
            wave_vector = tuple(float(v) for v in np.asarray(wave_vector, float))
            check_wave_vector(self, wave_vector, owner)  # ty: ignore[invalid-argument-type]
        return propagating_orders(self._face(surface, owner), f, wave_vector)

    # --- derived models -------------------------------------------------------

    def replace(self, **changes: Any) -> Model:
        """A new model with some fields changed, validated again.

        The fields are ``mesh``, ``materials``, ``boundaries``, ``ports``,
        ``excluded``, ``pml`` and ``background``. Shape optimization passes
        ``mesh=moved_mesh``. A new ``excluded`` replaces every physics'
        entries (a list or tuple: the electromagnetic ones, nothing excluded
        for the others); without it, each physics keeps its entries.

        Raises:
            TypeError: An unknown field name.
        """
        fields = (
            "mesh",
            "materials",
            "boundaries",
            "ports",
            "excluded",
            "pml",
            "background",
        )
        unknown = sorted(set(changes) - set(fields))
        if unknown:
            raise TypeError(
                f"Model.replace got unknown field(s) {', '.join(unknown)}; "
                f"use {', '.join(fields)}."
            )
        args = {f: changes.get(f, getattr(self, f)) for f in fields}
        if "excluded" not in changes:
            args["excluded"] = dict(zip(_PHYSICS, self._excluded_by, strict=True))
        same_domain = (
            isinstance(args["mesh"], Mesh)
            and args["mesh"]._topology is self.mesh._topology  # noqa: SLF001
            and "excluded" not in changes
        )
        # the layer geometry is read from the imported nodes: a moved mesh
        # of the same topology keeps it
        keep = same_domain and args["pml"] is not None and args["pml"] == self.pml
        new = object.__new__(Model)
        new._build(
            **args,
            domains=self._domains if same_domain else None,
            layers=self._layers if keep else None,
        )
        return new

    def with_material(self, region: str, material: Material | MaterialField) -> Model:
        """A new model with one volume region's material replaced.

        Raises:
            RegionNotFoundError: ``region`` is not a volume of the mesh.
        """
        _check_volume(self.mesh, region, "Materials are assigned to volumes.")
        return self.replace(materials={**self.materials, region: material})

    def with_temperature_rise(self, rise: Field) -> Model:
        r"""A new model whose permittivity follows a temperature rise.

        Each region with a ``thermo_optic_coefficient`` gets a
        `MaterialField` with, per element,
        :math:`\varepsilon_r = (n_0 + (dn/dT)\, \Delta T_e)^2`:
        :math:`n_0 = \sqrt{\varepsilon_r}` of its material (the index at the
        source-free temperature) and :math:`\Delta T_e` the mean rise over the
        element (the exact element mean on the rise's own mesh, else the
        quadrature mean of the rise at points located in its mesh). Its
        other properties are kept; regions without a coefficient, and
        elements excluded from electromagnetic solves, keep their index.
        Any optical solve then reads the heated index: a `PortMode`
        re-solve checks `gfem.thermo_optic_shift`'s first order.
        Differentiable with respect to the rise and the coefficient.

        Args:
            rise: The temperature rise, a scalar `Field` on any mesh that
                holds the heated elements (``SteadyThermalResult.rise``).

        Raises:
            TypeError: ``rise`` is not a scalar `Field`.
            ValueError: An element lies outside the rise's mesh or support
                (the message names the point).
            NotImplementedError: A region with a coefficient has a
                permittivity law or an anisotropic material.
        """
        from gfem.thermo_optic import with_temperature_rise

        return with_temperature_rise(self, rise)

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
        aux = (
            tuple(self.materials),
            self._excluded_by,
            self._domains,
            self._periodic,
            self.pml,
            self._layers,
            self.background,
        )
        return children, aux

    def tree_flatten(self):
        """JAX pytree protocol."""
        children, aux = self.tree_flatten_with_keys()
        return tuple(c for _, c in children), aux

    @classmethod
    def tree_unflatten(cls, aux, children) -> Model:
        """JAX pytree protocol (no validation: the structure is unchanged)."""
        order, excluded_by, domains, periodic, pml, layers, background = aux
        mesh, materials, boundaries, ports = children
        new = object.__new__(cls)
        set_ = object.__setattr__
        set_(new, "mesh", mesh)
        set_(new, "materials", types.MappingProxyType({k: materials[k] for k in order}))
        set_(new, "boundaries", boundaries)
        set_(new, "ports", ports)
        set_(new, "excluded", excluded_by[0])
        set_(new, "_excluded_by", excluded_by)
        set_(new, "_domains", domains)
        set_(new, "_domain", domains[0])
        set_(new, "_periodic", periodic)
        set_(new, "pml", pml)
        set_(new, "_layers", layers)
        set_(new, "background", background)
        return new

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "Model is immutable; derive a new one with model.replace(...) or "
            "model.with_material(...)."
        )

    def __repr__(self) -> str:
        lines = [f"Model({self.mesh!r}", "  materials:"]
        lines += [f"    {r}: {m!r}" for r, m in self.materials.items()]
        for physics, entries in zip(_PHYSICS, self._excluded_by, strict=True):
            if not entries:
                continue
            ents = ", ".join(
                f"{r} entity {e}" for r, e in self._excluded_entities_of(physics)
            )
            spec = ", ".join(repr(x) for x in entries)
            name = _PHYSICS_NAMES[physics]
            lines.append(f"  excluded ({name}): {ents} (from {spec})")
        if self.boundaries:
            lines.append("  boundaries:")
            lines += [f"    {b!r}" for b in self.boundaries]
        if self.ports:
            lines.append("  ports:")
            lines += [f"    {p!r}" for p in self.ports]
        if self.pml is not None and self._layers is not None:
            lines.append(f"  pml: {self.pml!r}")
            lines += [f"    {line}" for line in describe(self._layers, self.mesh)]
        if self.background is not None:
            lines.append(f"  background: {self.background!r}")
        if self.default_pmc:
            lines.append(
                "  default PMC (electromagnetic problems): "
                + ", ".join(self.default_pmc)
            )
        thermal = self._excluded_by[_PHYSICS.index("thermal")] or any(
            physics_of(b) == "thermal" for b in self.boundaries
        )
        if thermal and self.default_insulated:
            lines.append(
                "  insulated by default (thermal): " + ", ".join(self.default_insulated)
            )
        return "\n".join(lines) + ")"


def _check_independent(pairs: list[Pairing]) -> None:
    """The translations of a cell's pairs are linearly independent."""
    a = [np.asarray(p.translation) for p in pairs]
    for i, j in itertools.combinations(range(len(a)), 2):
        sine = np.linalg.norm(np.cross(a[i], a[j])) / (
            np.linalg.norm(a[i]) * np.linalg.norm(a[j])
        )
        if sine < _PARALLEL_SINE:
            raise ValueError(
                f"Pairs {pairs[i].label} and {pairs[j].label} are translated along "
                "parallel directions; a cell has one pair per lattice direction."
            )
    if len(a) == _MAX_PERIODIC_PAIRS:
        volume = abs(np.linalg.det(np.stack(a))) / np.prod(
            [np.linalg.norm(v) for v in a]
        )
        if volume < _PARALLEL_SINE:
            raise ValueError(
                f"Pairs {', '.join(p.label for p in pairs)} are translated along "
                "coplanar directions; a cell has one pair per lattice direction, "
                "along independent directions."
            )


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
            entities = mesh._topology.volume_entities[idx]  # noqa: SLF001 - either dimension
            out |= {(tag, int(e)) for e in set(entities.tolist())}
        elif isinstance(entry, EnclosedBy):
            for s in entry.surfaces:
                check_surface(mesh, s, repr(entry))
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


def _off_domain(what: str, surface: str, physics: str) -> str:
    """The refusal of an item on a surface its physics' domain does not reach."""
    if physics == "em":
        return (
            f"{what} on {surface!r}: the surface does not border the "
            "electromagnetic solve domain (it lies inside excluded volumes). "
            "Remove it, or exclude fewer volumes."
        )
    name = _PHYSICS_NAMES[physics]
    return (
        f"{what} on {surface!r}: the surface does not border the {name} solve "
        f"domain (it lies inside volumes excluded from {name} problems). Remove "
        f'it, or exclude fewer volumes with excluded={{"{physics}": (...)}}.'
    )


def _materials(
    mesh: Mesh,
    materials: Any,
    everywhere: set[str],
    em_excluded: set[str],
    pml: PML | None = None,
) -> Mapping[str, Material | MaterialField]:
    """Check ``materials``: none on ``everywhere``, all on EM-active volumes.

    ``everywhere`` holds the regions excluded from every physics,
    ``em_excluded`` those excluded from electromagnetic problems: a region
    only some physics solves on may go without a material (a thermal problem
    raises `MissingMaterialPropertyError` there).
    """
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
        if region in everywhere:
            raise ValueError(
                f"Region {region!r} is excluded from every physics "
                "(electromagnetic, thermal, DC conduction and mechanical), so it "
                "cannot also have a material. Remove one of the two."
            )
        if isinstance(value, MaterialField):
            _check_field(mesh, region, value)
    missing = [v for v in mesh.volumes if v not in materials and v not in em_excluded]
    if missing:
        raise ValueError(
            f"No material for volume(s) {', '.join(repr(v) for v in missing)}. "
            "Give each volume a material, e.g. materials={..., "
            f"{missing[0]!r}: gfem.Material(eps_r=...)}}, or remove it from the "
            "solve with excluded=(...)." + missing_background(missing, pml)
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
    if field.blend_ends is not None and field.blend_ends.pixels is not None:
        # a layer grid's pixels: the solve samples them at its own points
        if mesh.dim != 2:
            raise ValueError(
                f"materials[{region!r}] is a MaterialField over the pixels of a "
                "layer grid of a 2D mesh, which a tetrahedral model cannot "
                "sample. Build the grid on this model's mesh."
            )
        return
    n = field.n_elements
    expected = len(mesh.elements(region))
    if n is not None and n != expected:
        raise ValueError(
            f"materials[{region!r}] is a MaterialField with {n} values, but region "
            f"{region!r} has {expected} elements: give one value per element, in "
            f"the order of mesh.elements({region!r})."
        )


def _items(values: Any, kinds: tuple[type, ...], what: str) -> tuple:
    items = _as_tuple(values, what)
    for item in items:
        if not isinstance(item, kinds):
            names = ", ".join(k.__name__ for k in kinds)
            raise TypeError(f"{what} takes {names}; got {item!r}.")
    return items


def _check_polarity(
    name: str,
    polarity: Polarity,
    dim: int,
) -> None:
    """A polarity's point has the mesh's dimension; a line port's arrow is ±z."""
    point, direction = polarity
    owner = f"WavePort {name!r} polarity"
    if dim == 3 and len(point) != 3:
        raise ValueError(
            f"{owner}: the mesh is 3D, so the point needs 3 coordinates (x, y, z) "
            f"in m, got {point!r}."
        )
    if dim == 2:
        if len(point) != 2:
            raise ValueError(
                f"{owner}: the mesh is 2D, so the point is (x, y) in m on the "
                f"port's line, got {point!r}."
            )
        if abs(direction[2]) != 1.0:
            raise ValueError(
                f"{owner}: on a line port the direction is '+z' or '-z' (the "
                f"solved field is normal to the plane), got {direction!r}."
            )


def _check_one_per_surface(taken: dict[tuple[str, str], list]) -> None:
    """One condition or port per surface, or only wave or only Floquet ports.

    Several wave ports share a port surface; several Floquet ports share a
    Floquet surface. The mixed-kinds error comes first on every surface,
    then each shared surface's rules.
    """
    for (physics, surface), items in taken.items():
        shared = all(isinstance(i, WavePort) for i in items) or all(
            isinstance(i, FloquetPort) for i in items
        )
        if len(items) > 1 and not shared:
            raise ValueError(
                f"Surface {surface!r} has more than one "
                f"{_PHYSICS_NAMES[physics]} boundary condition or port: "
                f"{', '.join(repr(i) for i in items)}. Keep one."
            )
    for (_, surface), items in taken.items():
        if len(items) > 1 and isinstance(items[0], FloquetPort):
            _check_floquet_surface(surface, items)
        elif len(items) > 1:
            _check_shared_port_surface(surface, items)


def _check_floquet_surface(surface: str, ports: list) -> None:
    """Several Floquet ports on one surface: distinct (order, polarization)."""
    for k, b in enumerate(ports):
        for a in ports[:k]:
            if (a.order, a.polarization) == (b.order, b.polarization):
                raise ValueError(
                    f"FloquetPorts {a.name!r} and {b.name!r} on surface {surface!r} "
                    f"both request order {a.order} in {a.polarization}; give each "
                    "Floquet port on a surface a different (order, polarization)."
                )


def _check_shared_port_surface(surface: str, ports: list) -> None:
    """Several wave ports on one surface: distinct modes, equal ``pec_edges``."""
    for k, b in enumerate(ports):
        for a in ports[:k]:
            if a.mode == b.mode:
                raise ValueError(
                    f"WavePorts {a.name!r} and {b.name!r} on surface {surface!r} "
                    f"both request mode {a.mode}; give each wave port on a surface "
                    "a different mode."
                )
    first = ports[0]
    for other in ports[1:]:
        if set(other.pec_edges) != set(first.pec_edges):
            raise ValueError(
                f"WavePorts {first.name!r} and {other.name!r} on surface "
                f"{surface!r} share one port mode problem, so they need the same "
                f"pec_edges; got {first.pec_edges!r} and {other.pec_edges!r}."
            )
