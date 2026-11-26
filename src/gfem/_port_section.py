"""The cross-section of a wave port: a planar 2D mesh with materials and edges.

Pure NumPy on the `Model` arrays; the engine turns it into an NGSolve mesh.
The port mode problem reads the material of the volume element behind each
port triangle and the boundary conditions of the 3D surfaces that meet the
port along an edge.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import numpy as np

from gfem._boundaries import (
    PEC,
    Absorbing,
    Conductor,
    Impedance,
    WavePort,
    sheet_of,
    surfaces_of,
)
from gfem._checks import is_traced
from gfem._domain import physics_of
from gfem._engine.anisotropy import CROSS_N, edge_admittance
from gfem._line_section import LineSection, line_section, line_section_key
from gfem._materials import (
    PROPERTIES,
    Material,
    MaterialField,
    PermittivityLaw,
    field_principal_values,
    lossy_permittivity,
    principal_tensor,
)
from gfem._pml import Stretch, layer_tetrahedra
from gfem._text import hertz
from gfem.conventions import EXTERIOR_THICKNESS_FACTOR
from gfem.errors import PortGeometryError
from gfem.mesh import LENGTH_UNITS
from gfem.model import Model
from gfem.problems import Bend

# Relative distance from the port plane (in port diameters) still on the plane.
PLANE_RTOL = 1e-6


@dataclasses.dataclass(frozen=True, eq=False)
class SectionBend:
    r"""A `gfem.Bend` in the frame of a port section.

    The metric factor is :math:`h = 1 + \xi/R` with
    :math:`\xi = s\,(\mathbf{x}\cdot\mathrm{outward} - \mathrm{offset})` the
    in-plane distance from the reference arc in m, :math:`\mathbf{x}` a point
    in local units and :math:`s` the section's ``scale``.

    Attributes:
        radius: The radius :math:`R` of the reference arc, m, as given.
        outward: Unit vector along ``outward`` in the local ``(u, v)`` basis.
        offset: The reference arc's coordinate along ``outward``, local units.
    """

    radius: float
    outward: tuple[float, float]
    offset: float

    @property
    def inverse_radius(self) -> float:
        """:math:`1/R` in 1/m."""
        return 1.0 / self.radius


@dataclasses.dataclass(frozen=True, eq=False)
class PortSection:
    """A wave port's planar cross-section, ready for the port mode problem.

    Coordinates are local to the port plane, in the mesh file's length unit
    (``scale`` metres). The local frame ``(u, v, n)`` is right-handed and
    ``n`` points into the electromagnetic solve domain, the direction of
    propagation of an incident wave.

    Attributes:
        port: The wave port.
        scale: Metres per local coordinate unit.
        origin: A point of the plane, in local units of the global frame.
        frame: Rows ``u``, ``v``, ``n`` (unit vectors, global frame).
        points: Node coordinates ``(n_nodes, 2)`` in the plane.
        triangles: ``(n_tri, 3)`` node indices, counter-clockwise in (u, v).
        eps_tt: Complex relative permittivity in the plane, ``(n_tri, 2, 2)``,
            without conductivity; a law triangle's eps_inf.
        eps_nn: Complex relative permittivity along ``n``, ``(n_tri,)``.
        nu_tt: ``R^T mu_tt^-1 R`` with ``R a = a x n``, ``(n_tri, 2, 2)``.
        nu_nn: ``1 / mu_nn``, ``(n_tri,)``.
        sigma_tt: Electrical conductivity in the plane, S/m, ``(n_tri, 2, 2)``.
        sigma_nn: Electrical conductivity along ``n``, S/m, ``(n_tri,)``.
        pec_edges: ``(n, 2)`` node pairs on perfect conductors.
        absorbing_edges: ``(n, 2)`` node pairs on absorbing boundaries.
        absorbing_admittance: Relative wave admittances ``(Y_tau, Y_n)`` of
            the medium beside each absorbing edge for the field along the
            edge and along ``n``, ``(n, 2)``
            (`gfem._engine.anisotropy.edge_admittance`; both ``sqrt(eps /
            mu)`` in an isotropic medium). Zero on an edge beside a law
            triangle: its admittance depends on the frequency
            (``absorbing_law``).
        faces: The mesh triangle (gfem index) of each of ``triangles``, or
            ``None`` (sections saved before it was recorded).
        impedance_edges: ``(n, 2)`` node pairs on `Conductor` or `Impedance`
            surfaces.
        impedance_group: ``(n,)`` index of each such edge's term in
            ``impedance_terms``.
        impedance_terms: ``(boundary, thickness factor)`` per group: the
            `Conductor` or `Impedance` and, for a conductor, the factor on
            its thickness (2 where the 3D surface has the solve domain on one
            side only, as in the driven problem; see `gfem.conventions`).
        impedance_fill: ``(n,)`` the sheet field's fill on the face of the
            3D surface along each impedance edge (1 for a boundary without
            one), or ``None`` when no edge's boundary has a sheet field.
        laws: ``(region, law)`` per region of the section whose eps_r is a
            `gfem.PermittivityLaw`; empty without one.
        law_index: ``(n_tri,)`` each triangle's index in ``laws``, -1 off
            them; ``None`` without laws.
        absorbing_law: ``(n,)`` the index in ``laws`` of the triangle beside
            each absorbing edge, -1 off them; ``None`` without laws.
        stretch: The perfectly matched layer's stretch of the solve the
            section serves (`gfem._pml.stretch`), or ``None`` without a
            layer. The material arrays keep the background; the port mode
            problem multiplies them by the stretch through the port frame.
        layer: ``(n_tri,)`` whether each triangle lies on a layer
            tetrahedron, or ``None`` when none does (then ``stretch`` is
            ``None`` too).
    """

    port: WavePort
    scale: float
    origin: np.ndarray
    frame: np.ndarray
    points: np.ndarray
    triangles: np.ndarray
    eps_tt: np.ndarray
    eps_nn: np.ndarray
    nu_tt: np.ndarray
    nu_nn: np.ndarray
    sigma_tt: np.ndarray
    sigma_nn: np.ndarray
    pec_edges: np.ndarray
    absorbing_edges: np.ndarray
    absorbing_admittance: np.ndarray
    faces: np.ndarray | None = None
    impedance_edges: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((0, 2), np.int64)
    )
    impedance_group: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, np.int64)
    )
    impedance_terms: tuple = ()
    impedance_fill: np.ndarray | None = None
    laws: tuple[tuple[str, PermittivityLaw], ...] = ()
    law_index: np.ndarray | None = None
    absorbing_law: np.ndarray | None = None
    stretch: Stretch | None = None
    layer: np.ndarray | None = None
    bend: SectionBend | None = None

    @property
    def normal(self) -> np.ndarray:
        """Unit inward port normal, global frame, ``(3,)``."""
        return self.frame[2]

    @property
    def diameter(self) -> float:
        """Largest extent of the cross-section, local units."""
        return float(np.ptp(self.points, axis=0).max())

    def bend_distance(self, points: np.ndarray | None = None) -> np.ndarray:
        r"""The distance :math:`\xi` of ``points`` (default the nodes) from the arc, m.

        Along the bend's ``outward``; requires a bend.
        """
        bend = self.bend
        assert bend is not None  # noqa: S101 - callers check
        pts = self.points if points is None else points
        return (pts @ np.asarray(bend.outward) - bend.offset) * self.scale

    def bend_h_range(self, *, physical: bool = False) -> tuple[float, float]:
        r"""The extremes of :math:`\mathrm{Re}\,h = 1 + \xi/R` over the section.

        With ``physical`` over the triangles off the layer only.
        """
        bend = self.bend
        assert bend is not None  # noqa: S101 - callers check
        tris = self.triangles
        if physical and self.layer is not None:
            tris = tris[~self.layer]
        xi = self.bend_distance()[np.unique(tris)]
        return 1.0 + float(xi.min()) / bend.radius, 1.0 + float(xi.max()) / bend.radius

    @property
    def max_conductivity(self) -> float:
        """The largest principal conductivity over the cross-section, S/m."""
        return float(max(np.linalg.eigvalsh(self.sigma_tt).max(), self.sigma_nn.max()))

    def index_squared(
        self, sigma_term: float = 0.0, frequency: float | None = None
    ) -> tuple[float, float]:
        """Bounds on the squared bulk index over the cross-section.

        From the principal values of each triangle's tensors (the diagonal
        entries of a tensor turned in the plane fall short of them).

        Args:
            sigma_term: ``max_conductivity / (omega eps0)`` to add to ``|eps|``.
            frequency: Hz, where the laws are read (`eps_at`); without it, a
                law triangle counts its eps_inf.

        Returns:
            ``max Re(eps) max Re(mu)`` (the shift; largest
            eigenvalues) and ``(max |eps| + sigma_term) max |mu|`` (an upper
            bound on the ``|n|^2`` of any propagating mode; spectral norms).
        """
        # mu_tt^-1 is congruent to nu_tt by a rotation: same eigenvalues
        mu_tt, mu_nn = np.linalg.inv(self.nu_tt), 1 / self.nu_nn
        eps_tt, eps_nn = (
            (self.eps_tt, self.eps_nn) if frequency is None else self.eps_at(frequency)
        )
        return (
            float(_largest(eps_tt, eps_nn) * _largest(mu_tt, mu_nn)),
            float((_norm(eps_tt, eps_nn) + sigma_term) * _norm(mu_tt, mu_nn)),
        )

    def eps_at(self, frequency: float) -> tuple[np.ndarray, np.ndarray]:
        """``(eps_tt, eps_nn)`` with each law's permittivity at ``frequency`` (Hz).

        eps_inf + chi(omega) on the law triangles (isotropic), the stored
        values elsewhere. Non-finite on a lossless pole (`check_frequencies`).
        """
        if not self.laws:
            return self.eps_tt, self.eps_nn
        eps_tt, eps_nn = self.eps_tt.copy(), self.eps_nn.copy()
        index = np.asarray(self.law_index)
        for g, chi in enumerate(self.susceptibilities(frequency)):
            rows = index == g
            eps_tt[rows] += chi * np.eye(2)
            eps_nn[rows] += chi
        return eps_tt, eps_nn

    def susceptibilities(self, frequency: float) -> list[complex]:
        """The susceptibility chi of each of ``laws`` at ``frequency`` (Hz)."""
        omega = 2 * math.pi * frequency
        from gfem._engine.dispersion import susceptibility  # lazy: no engine at import

        return [susceptibility(law, omega) for _, law in self.laws]

    def check_frequencies(self, frequencies) -> None:
        """Raise where a law of the section cannot be solved.

        Static: reads the laws and the frequencies only, before any solve.

        Raises:
            ValueError: A frequency lies on a lossless pole of a law, or a
                law region has Re eps <= 0 there (a metal in the section).
        """
        check_law_frequencies(self.port, self.laws, self.susceptibilities, frequencies)

    def to_local(self, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Plane coordinates ``(n, 2)`` and normal distances of global points (m)."""
        rel = np.asarray(points_m, float) / self.scale - self.origin
        return rel @ self.frame[:2].T, rel @ self.frame[2]

    def to_global(self, local: np.ndarray) -> np.ndarray:
        """Global points (m) of plane coordinates ``(n, 2)``."""
        return (self.origin + np.asarray(local, float) @ self.frame[:2]) * self.scale


def check_law_frequencies(port: WavePort, laws, susceptibilities, frequencies) -> None:
    """Raise where a law of a port's section cannot be solved.

    Static: reads the laws and the frequencies only, before any solve.

    Args:
        port: The wave port, for messages.
        laws: The section's ``(region, law)`` pairs.
        susceptibilities: ``frequency -> [chi of each law]``.
        frequencies: Hz.

    Raises:
        ValueError: A frequency lies on a lossless pole of a law, or a
            law region has Re eps <= 0 there (a metal in the section).
    """
    for f in frequencies:
        for (region, law), chi in zip(laws, susceptibilities(float(f)), strict=True):
            name = port.name
            if not np.isfinite(chi):
                raise ValueError(
                    f"WavePort {name!r}: frequency {hertz(f)} lies on a "
                    f"lossless pole of the permittivity law of region "
                    f"{region!r} in the port's cross-section: its permittivity "
                    "is infinite there. Give the term a damping, or leave "
                    "this frequency out."
                )
            eps = float(law.eps_inf) + chi.real
            if eps <= 0:
                raise ValueError(
                    f"WavePort {name!r}: region {region!r} has Re eps = "
                    f"{eps:.3g} at {hertz(f)} in the port's cross-section; a "
                    "port mode with a metal (Re eps <= 0) in its cross-section "
                    "is not supported. Move the port plane so that its "
                    "cross-section is dielectric, or use a LumpedPort."
                )


def _largest(tt: np.ndarray, nn: np.ndarray) -> float:
    """The largest eigenvalue of the real parts of ``(tt, nn)`` blocks."""
    return max(np.linalg.eigvalsh(tt.real).max(), nn.real.max())


def _norm(tt: np.ndarray, nn: np.ndarray) -> float:
    """The largest spectral norm of ``(tt, nn)`` blocks."""
    return max(np.linalg.norm(tt, ord=2, axis=(1, 2)).max(), np.abs(nn).max())


@dataclasses.dataclass(frozen=True, eq=False)
class _Plane:
    """What a port surface's mesh triangles say of the plane, nodes included."""

    scale: float
    sel: np.ndarray
    tets: np.ndarray
    tris: np.ndarray
    used: np.ndarray
    origin: np.ndarray
    frame: np.ndarray
    points: np.ndarray
    local_tris: np.ndarray


def _plane(model: Model, port: WavePort, node_values: np.ndarray) -> _Plane:
    """The port plane's frame and 2D triangulation at node coordinates ``node_values``.

    Raises:
        PortGeometryError: The port surface is not planar, or the solve
            domain lies on both sides of it.
    """
    mesh = model.mesh
    scale = LENGTH_UNITS[mesh.length_unit]
    nodes = node_values / scale
    active = model._domain.active  # noqa: SLF001
    tag = mesh.region(port.surface).tag
    nb = mesh.triangle_neighbours
    sel = np.flatnonzero(mesh.triangle_tags == tag)
    side_ok = (nb[sel] >= 0) & active[np.maximum(nb[sel], 0)]
    sel = sel[side_ok.sum(axis=1) == 1]
    tets = np.where(side_ok[side_ok.sum(axis=1) == 1][:, 0], nb[sel, 0], nb[sel, 1])
    tris = mesh.triangles[sel]

    used, local_tris = np.unique(tris, return_inverse=True)
    local_tris = local_tris.reshape(-1, 3)
    xyz = nodes[used]
    origin = xyz.mean(axis=0)
    *_, vt = np.linalg.svd(xyz - origin, full_matrices=False)
    normal = vt[2]
    diameter = float(np.ptp(xyz, axis=0).max())
    off = np.abs((xyz - origin) @ normal).max()
    if off > PLANE_RTOL * diameter:
        raise PortGeometryError(
            f"WavePort {port.name!r}: surface {port.surface!r} is not planar (its "
            f"nodes are up to {off * scale:.3g} m off their best-fit plane). A wave "
            "port must be one flat cross-section; split the surface or use a "
            "LumpedPort."
        )
    centroid = nodes[mesh.tetrahedra[tets]].mean(axis=1)
    inward = np.sign((centroid - nodes[tris].mean(axis=1)) @ normal)
    if not (inward == inward[0]).all():
        raise PortGeometryError(
            f"WavePort {port.name!r}: the solve domain lies on both sides of the "
            f"plane of {port.surface!r}; a wave port must face the domain from one "
            "side."
        )
    normal = normal * inward[0]
    u = vt[0]
    v = np.cross(normal, u)
    frame = np.stack([u, v, normal])
    points = (xyz - origin) @ frame[:2].T

    p = points[local_tris]
    area2 = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (
        p[:, 1, 1] - p[:, 0, 1]
    ) * (p[:, 2, 0] - p[:, 0, 0])
    flip = area2 < 0
    local_tris[flip] = local_tris[flip][:, ::-1]

    return _Plane(scale, sel, tets, tris, used, origin, frame, points, local_tris)


def _topology_edges(
    model: Model, port: WavePort, plane: _Plane
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    """The cross-section's edges by kind, from the mesh and the boundaries alone.

    Returns the perfect-conductor, absorbing and sheet (`Conductor`,
    `Impedance`) edges as local node pairs, and the triangle beside each
    absorbing edge. The same classification as `_edges`, without values.
    """
    port_edges = _edge_keys(plane.tris)
    kinds = _edge_kinds(model, port)
    local = np.searchsorted(plane.used, port_edges)
    pec: list[np.ndarray] = []
    absorbing: list[np.ndarray] = []
    sheets: list[np.ndarray] = []
    owners: list[int] = []
    seen: set[tuple[int, int]] = set()
    for k, surfaces in enumerate(_surfaces_on_edges(model, port, port_edges)):
        key = (int(local[k][0]), int(local[k][1]))
        if not surfaces or key in seen:
            continue
        seen.add(key)
        types = [kinds.get(s) for s in sorted(surfaces)]
        if "pec" in types:
            pec.append(local[k])
        elif any(isinstance(t, (Conductor, Impedance)) for t in types):
            sheets.append(local[k])
        elif any(t in ("absorbing", "absorbing2") for t in types):
            absorbing.append(local[k])
            owners.append(k // 3)
    return pec, absorbing, sheets, owners


def port_geometry(model: Model, port: WavePort) -> PortSection:
    """The cross-section of ``port`` from the mesh and the boundaries alone.

    The triangulation, frame and edge sets of `port_section`, with no
    material, law or sheet value read: a model whose values are JAX tracers
    (a design inside `jax.jit`) has one, and it locates a mode's field. The
    node coordinates are the model's when they are concrete values and the
    mesh's imported ones when they are traced (the DOF count does not read
    them). Material arrays are placeholders, and the sheet edges join the
    absorbing ones.

    Raises:
        PortGeometryError: As `port_section`.
    """
    mesh = model.mesh
    nodes = mesh.nodes
    plane = _plane(
        model, port, np.asarray(mesh.imported_nodes if is_traced(nodes) else nodes)
    )
    pec, absorbing, sheets, _ = _topology_edges(model, port, plane)
    n = len(plane.tris)
    pairs = [
        np.asarray(e, np.int64).reshape(-1, 2) for e in (pec, [*absorbing, *sheets])
    ]
    return PortSection(
        port=port,
        scale=plane.scale,
        origin=plane.origin,
        frame=plane.frame,
        points=plane.points,
        triangles=plane.local_tris,
        eps_tt=np.tile(np.eye(2, dtype=complex), (n, 1, 1)),
        eps_nn=np.ones(n, complex),
        nu_tt=np.tile(np.eye(2, dtype=complex), (n, 1, 1)),
        nu_nn=np.ones(n, complex),
        sigma_tt=np.zeros((n, 2, 2)),
        sigma_nn=np.zeros(n),
        pec_edges=pairs[0],
        absorbing_edges=pairs[1],
        absorbing_admittance=np.ones((len(pairs[1]), 2)),
        faces=plane.sel,
    )


def anisotropic_absorbing(
    model: Model, port: WavePort, region: str | None = None
) -> str | None:
    """Why a port-face design is refused beside an absorbing edge, or ``None``.

    An anisotropic material (per-axis values or axes) owning a port-face
    triangle next to an absorbing port edge has no per-axis admittance
    derivative in this version. With ``region``, only that region's design is
    in question: it is refused if its material is an anisotropic
    `MaterialField` that owns such a triangle (a constant anisotropic owner
    has no leaf, so its admittance takes no cotangent).
    """
    mesh = model.mesh
    if mesh.dim == 2:
        return None
    plane = _plane(model, port, np.asarray(mesh.imported_nodes))
    _, absorbing, _, owners = _topology_edges(model, port, plane)
    if not absorbing:
        return None
    names = mesh._region_names()  # noqa: SLF001
    for t in plane.tets[owners]:
        name = names[int(mesh.tetrahedron_tags[t])]
        material = model.materials[name]
        if region is not None and (
            name != region
            or not isinstance(material, MaterialField)
            or material.axes is None
        ):
            continue
        if isinstance(material, MaterialField):
            aniso = material.axes is not None
        else:
            aniso = not material.isotropic
        if aniso:
            return (
                "the absorbing edge's admittance beside a per-axis material at a "
                "wave port face is not differentiable in this version; make the "
                "material isotropic next to the absorbing edge or wrap it in "
                "jax.lax.stop_gradient"
            )
    return None


def port_section(
    model: Model,
    port: WavePort,
    stretch: Stretch | None = None,
    bend: Bend | None = None,
) -> PortSection | LineSection:
    """Extract the cross-section of ``port`` from ``model``.

    On a 2D mesh, the `LineSection` of a line port (`line_section`).

    ``stretch`` is the layer's stretch of the solve (`gfem._pml.stretch`),
    at its reference frequency; the section keeps it and marks its layer
    triangles. ``bend`` is the guide's `gfem.Bend`, kept in the section's
    frame once its section passes the checks of `check_bend`.

    Raises:
        PortGeometryError: The port surface is not planar (on a 2D mesh: the
            line is not straight and connected).
        ValueError: An anisotropic material beside the port has no principal
            axis along the port normal; or, with a ``bend``, as `check_bend`.
        NotImplementedError: A ``bend`` on a 2D mesh or beside an absorbing,
            impedance or conductor edge.
    """
    mesh = model.mesh
    if mesh.dim == 2:
        if bend is not None:
            raise NotImplementedError(
                f"WavePort {port.name!r}: a bend is a 3D port mode problem; a "
                "line port on a 2D mesh has none in this version."
            )
        return line_section(model, port)
    plane = _plane(model, port, np.asarray(mesh.nodes, float))
    scale, sel, tets, tris, used = (
        plane.scale,
        plane.sel,
        plane.tets,
        plane.tris,
        plane.used,
    )
    origin, frame, points, local_tris = (
        plane.origin,
        plane.frame,
        plane.points,
        plane.local_tris,
    )
    eps_tt, eps_nn, nu_tt, nu_nn, sigma_tt, sigma_nn, isotropic = _materials(
        model, port, tets, frame
    )
    laws, law_index = _laws(model, tets)
    edges = _edges(
        model,
        port,
        tris,
        used,
        points=points,
        tensors=(eps_tt, eps_nn, nu_tt, nu_nn),
        isotropic=isotropic,
        law_index=law_index,
        laws=laws,
    )
    pec, absorbing, admittance, impedance, absorbing_law = edges
    imp_edges, imp_group, imp_terms, imp_fill = impedance
    in_layer = layer_tetrahedra(mesh, model.pml)
    layer = None if stretch is None or in_layer is None else in_layer[tets]
    if layer is not None and not layer.any():
        # a port clear of the layer: the stretch is the identity on its face
        layer = None
    section = PortSection(
        port=port,
        scale=scale,
        origin=origin,
        frame=frame,
        points=points,
        triangles=local_tris,
        eps_tt=eps_tt,
        eps_nn=eps_nn,
        nu_tt=nu_tt,
        nu_nn=nu_nn,
        sigma_tt=sigma_tt,
        sigma_nn=sigma_nn,
        pec_edges=pec,
        absorbing_edges=absorbing,
        absorbing_admittance=admittance,
        faces=sel,
        impedance_edges=imp_edges,
        impedance_group=imp_group,
        impedance_terms=imp_terms,
        impedance_fill=imp_fill,
        laws=laws,
        law_index=law_index if laws else None,
        absorbing_law=absorbing_law if laws else None,
        stretch=None if layer is None else stretch,
        layer=layer,
        bend=None if bend is None else _section_bend(bend, port, plane),
    )
    if bend is not None:
        names = mesh._region_names()  # noqa: SLF001
        regions = np.array([names[int(t)] for t in mesh.tetrahedron_tags[tets]])
        check_bend(section, regions)
    return section


#: The material properties the port mode problem reads (never the thermal ones).
_PORT_PROPERTIES = ("eps_r", "mu_r", "tan_delta", "conductivity")


def port_section_key(
    model: Model,
    port: WavePort,
    stretch: Stretch | None = None,
    bend: Bend | None = None,
) -> str:
    """A digest of what `port_section` reads of ``model`` beyond its mesh.

    Two models on one discretization (mesh topology and node coordinates,
    which the caller keys on) with equal keys have equal port sections, and
    equal Robin coefficients on the port's surface
    (`gfem._engine.terms.wave_port_coefficients`). The key holds the length
    unit, the solve domain, the port, every boundary condition and port (what
    each surface is to the section's edges), the electromagnetic material
    values (eps_r, mu_r, tan_delta, conductivity, principal axes) of the
    solve-domain tetrahedra beside the port's faces, and the sheet fill of the
    faces sharing an edge with them, a layer's ``stretch`` (its sides,
    target reflection and reference frequency) and a ``bend``'s radius,
    direction and point. It builds no section: no
    plane fit and no edge classification, so it costs a fraction of
    `port_section`. On a 2D mesh, `line_section_key`.
    """
    mesh = model.mesh
    if mesh.dim == 2:
        return line_section_key(model, port)
    domain = model._domain  # noqa: SLF001
    digest = hashlib.blake2b(digest_size=16)

    def add(*parts: object) -> None:
        digest.update(repr(parts).encode())

    add(mesh.length_unit, domain.excluded, port, model.ports)
    if stretch is not None:
        add(stretch.key)
    if bend is not None:
        add(bend)
    # a sheet field's repr is not its values: its fill enters below
    add(
        tuple(
            dataclasses.replace(b, fill=None) if sheet_of(b) is not None else b
            for b in model.boundaries
            if physics_of(b) == "em"
        )
    )
    faces = mesh.elements(port.surface)
    nb = mesh.triangle_neighbours[faces]
    tets = np.unique(nb[(nb >= 0) & domain.active[np.maximum(nb, 0)]])
    names = mesh._region_names()  # noqa: SLF001
    region_of = mesh.tetrahedron_tags[tets]
    for rtag in np.unique(region_of):
        region = names[int(rtag)]
        material = model.materials[region]
        add(region, type(material).__name__, material.axes)
        if isinstance(material, Material):
            add(*(getattr(material, n) for n in _PORT_PROPERTIES))
            continue
        pos = np.searchsorted(mesh.elements(region), tets[region_of == rtag])
        for name in _PORT_PROPERTIES:
            value = getattr(material, name)
            if value is None:
                add(name, None)
                continue
            values = np.asarray(value)
            # per element: (n,) or, with principal axes, (n, 3); else uniform
            if values.ndim == (1 if material.axes is None else 2):
                values = values[pos]
            add(name, values.dtype.str, values.shape)
            digest.update(np.ascontiguousarray(values).tobytes())
    n_nodes = len(mesh.nodes)
    port_edges = _edge_keys(mesh.triangles[faces]) @ np.array([n_nodes, 1])
    for i, b in enumerate(model.boundaries):
        sheet = sheet_of(b)
        if sheet is None:
            continue
        (surface,) = surfaces_of(b)
        edges = _edge_keys(mesh.triangles[mesh.elements(surface)])
        beside = np.isin(edges @ np.array([n_nodes, 1]), port_edges)
        beside = beside.reshape(-1, 3).any(axis=1)
        add(i, np.flatnonzero(beside).tobytes())
        digest.update(np.asarray(sheet.fill, float)[beside].tobytes())
    return digest.hexdigest()


#: ``outward`` must lie in the port plane to this (the sine of its angle).
_BEND_PLANE_TOL = 1e-6
#: Relative tolerance of the rotation invariance of a bent section's materials
#: (round-off of a tensor rotated into the port frame is about 1e-16).
_BEND_INVARIANCE_RTOL = 1e-12


def _section_bend(bend: Bend, port: WavePort, plane: _Plane) -> SectionBend:
    """The bend in the port plane's frame.

    Raises:
        ValueError: ``outward`` is not in the plane of the port, or ``at`` is
            off it.
    """
    frame, scale = plane.frame, plane.scale
    outward = np.asarray(bend.outward, float)
    if abs(outward @ frame[2]) > _BEND_PLANE_TOL:
        raise ValueError(
            f"WavePort {port.name!r}: Bend.outward {bend.outward} is not in the "
            f"plane of the port: it has a component {outward @ frame[2]:.3g} along "
            f"the port normal {tuple(float(x) for x in frame[2])}. A bend turns "
            "the guide within its cross-section's plane; give an in-plane "
            "direction."
        )
    rel = np.asarray(bend.at, float) / scale - plane.origin
    diameter = float(np.ptp(plane.points, axis=0).max())
    if abs(rel @ frame[2]) > PLANE_RTOL * diameter:
        raise ValueError(
            f"WavePort {port.name!r}: Bend.at {bend.at} is "
            f"{abs(rel @ frame[2]) * scale:.3g} m off the plane of the port; give "
            "a point of the reference arc in that plane."
        )
    local = frame[:2] @ outward
    local = local / np.linalg.norm(local)
    return SectionBend(
        radius=float(bend.radius),
        outward=(float(local[0]), float(local[1])),
        offset=float(local @ (frame[:2] @ rel)),
    )


def check_bend(section: PortSection, regions: np.ndarray) -> None:
    r"""Refuse a bent section that the metric cannot represent.

    Args:
        section: A section with a ``bend``.
        regions: ``(n_tri,)`` the volume region name of each triangle.

    Raises:
        NotImplementedError: An absorbing, impedance or conductor edge (their
            edge terms carry no bend metric).
        ValueError: :math:`\mathrm{Re}\,h = 1 + \xi/R` is not positive
            everywhere, or a region's material is not invariant about the
            bend's axis.
    """
    bend, port = section.bend, section.port
    assert bend is not None  # noqa: S101 - callers check
    if len(section.absorbing_edges) or len(section.impedance_edges):
        raise NotImplementedError(
            f"WavePort {port.name!r}: a bent port mode has no Absorbing, "
            "Impedance or Conductor edge (or sheet) in this version; its "
            "edges are PEC, PMC (natural) or the layer's. Absorb the "
            "radiated field with a gfem.PML."
        )
    h_min, _ = section.bend_h_range()
    if h_min <= 0:
        lowest = float(section.bend_distance().min())
        raise ValueError(
            f"WavePort {port.name!r}: the bend of radius {bend.radius:.6g} m "
            f"reaches its own axis: the section extends {lowest:.6g} m from "
            f"the reference arc along outward and the axis lies at "
            f"{-bend.radius:.6g} m, so the minimum of h = 1 + xi/R is "
            f"{h_min:.3g} <= 0. Use a radius larger than the section's extent "
            "on the inner side."
        )
    o = np.asarray(bend.outward)
    across = np.array([-o[1], o[0]])
    # about the bend's axis (along `across`) the section must be a body of
    # revolution: the (outward, normal) block of each tensor a scalar. For
    # eps and sigma that is tt @ outward == nn * outward; nu is
    # R^T mu_tt^-1 R, whose block along `across` is the mu block along outward
    checks = (
        ("eps_r", section.eps_tt, section.eps_nn, o),
        ("sigma", section.sigma_tt, section.sigma_nn, o),
        ("mu_r", section.nu_tt, section.nu_nn, across),
    )
    for name, tt, nn, vec in checks:
        lhs = tt @ vec
        rhs = nn[:, None] * vec
        size = np.maximum(np.linalg.norm(tt, axis=(1, 2)), np.abs(nn))
        bad = np.linalg.norm(lhs - rhs, axis=1) > _BEND_INVARIANCE_RTOL * size
        if bad.any():
            names = sorted({str(r) for r in regions[bad]})
            raise ValueError(
                f"WavePort {port.name!r}: region(s) {names} "
                f"have a {name} that is not invariant about the bend's axis: "
                "the permittivity (conductivity, permeability) along outward "
                "must equal that along the propagation direction, with no "
                "cross term, as for an isotropic material or a uniaxial one "
                "whose axis is along the in-plane direction perpendicular to "
                "outward. Rotate the crystal axes or use a straight port mode."
            )


def outer_sides(section: PortSection) -> list:
    """The layer sides on the outer side of a bent section (along ``outward``).

    The sides of the section's layer whose outward normal has a positive
    component along the bend's ``outward``; empty without a layer.
    """
    bend, stretch = section.bend, section.stretch
    assert bend is not None  # noqa: S101 - callers check
    if stretch is None or section.layer is None:
        return []
    outward = np.asarray(bend.outward) @ section.frame[:2]
    return [
        side
        for side in stretch.layers.sides
        if side.sign * outward[side.axis] > _BEND_PLANE_TOL
    ]


def outer_edges_conduct(section: PortSection) -> bool:
    """Whether the outer wall of a bent section is perfectly conducting.

    The wall is the boundary edges with both nodes at the largest distance
    along the bend's ``outward``.
    """
    bend = section.bend
    assert bend is not None  # noqa: S101 - callers check
    tris = section.triangles
    edges = _edge_keys(tris)
    keys, counts = np.unique(
        edges @ np.array([len(section.points), 1]), return_counts=True
    )
    boundary = np.isin(edges @ np.array([len(section.points), 1]), keys[counts == 1])
    edges = edges[boundary]
    xi = section.points @ np.asarray(bend.outward)
    # the outer wall: both nodes within round-off of the section's extreme
    tol = 1e-9 * section.diameter
    outer = (xi[edges] >= xi.max() - tol).all(axis=1)
    if not outer.any():
        return False
    pec = np.asarray(section.pec_edges, np.int64).reshape(-1, 2)
    wall = edges[outer]
    n = len(section.points)
    return bool(
        np.isin(wall @ np.array([n, 1]), np.sort(pec, axis=1) @ np.array([n, 1])).all()
    )


def _materials(model: Model, port: WavePort, tets: np.ndarray, frame: np.ndarray):
    """Per-triangle material values in the port frame.

    Also whether each triangle's material is isotropic. A law triangle holds
    eps_inf.
    """
    mesh = model.mesh
    names = mesh._region_names()  # noqa: SLF001
    n = len(tets)
    eps_tt = np.zeros((n, 2, 2), complex)
    eps_nn = np.zeros(n, complex)
    nu_tt = np.zeros((n, 2, 2), complex)
    nu_nn = np.zeros(n, complex)
    sigma_tt = np.zeros((n, 2, 2))
    sigma_nn = np.zeros(n)
    isotropic = np.ones(n, bool)
    region_of = mesh.tetrahedron_tags[tets]
    for rtag in np.unique(region_of):
        region = names[int(rtag)]
        rows = np.flatnonzero(region_of == rtag)
        material = model.materials[region]
        if isinstance(material, Material):
            eps, eps_name = _permittivity(material)
            mu = principal_tensor(material.mu_r, material.axes).astype(complex)
            sigma = principal_tensor(material.conductivity, material.axes)
            et, en = _in_frame(eps, frame, port, region, eps_name)
            mt, mn = _in_frame(mu, frame, port, region, "mu_r")
            st, sn = _in_frame(sigma, frame, port, region, "conductivity")
            eps_tt[rows], eps_nn[rows] = et, en
            nu_tt[rows], nu_nn[rows] = _inverse_permeability(mt, mn)
            sigma_tt[rows], sigma_nn[rows] = st, sn
            isotropic[rows] = material.isotropic
        elif material.blend_ends is not None:
            raise NotImplementedError(
                f"WavePort {port.name!r}: region {region!r} has elements on the "
                "port face and is a density blend with a permittivity-law end "
                "(MaterialField.interpolate), which the port mode problem does "
                "not read. Keep the design away from wave port faces (give the "
                "elements beside the port a region of their own with a constant "
                "gfem.Material)."
            )
        elif material.axes is not None:
            pos = np.searchsorted(mesh.elements(region), tets[rows])
            blocks = _field_blocks(material, frame, port, region, pos)
            (et, en), (mt, mn), (st, sn) = blocks
            eps_tt[rows], eps_nn[rows] = et, en
            nu_tt[rows], nu_nn[rows] = _inverse_permeability(mt, mn)
            sigma_tt[rows], sigma_nn[rows] = st.real, sn.real
            isotropic[rows] = False
        else:
            pos = np.searchsorted(mesh.elements(region), tets[rows])
            eps = _field(material, "eps_r", 1.0, pos)
            tan = _field(material, "tan_delta", 0.0, pos).real
            mu = _field(material, "mu_r", 1.0, pos)
            eps = eps * (1 - 1j * tan)
            eps_tt[rows] = eps[:, None, None] * np.eye(2)
            eps_nn[rows] = eps
            nu_tt[rows] = (1 / mu)[:, None, None] * np.eye(2)
            nu_nn[rows] = 1 / mu
            sigma = _field(material, "conductivity", 0.0, pos).real
            sigma_tt[rows] = sigma[:, None, None] * np.eye(2)
            sigma_nn[rows] = sigma
    return eps_tt, eps_nn, nu_tt, nu_nn, sigma_tt, sigma_nn, isotropic


def _laws(model: Model, tets: np.ndarray):
    """The section's laws, ``((region, law), ...)``, and each triangle's index.

    ``tets`` are the volume elements behind the triangles; the index is -1
    off the law regions.
    """
    mesh = model.mesh
    names = mesh._region_names()  # noqa: SLF001
    laws: list[tuple[str, PermittivityLaw]] = []
    index = np.full(len(tets), -1, np.int64)
    region_of = mesh.tetrahedron_tags[tets]
    for rtag in np.unique(region_of):
        region = names[int(rtag)]
        material = model.materials[region]
        if isinstance(material, Material) and isinstance(
            material.eps_r, PermittivityLaw
        ):
            index[region_of == rtag] = len(laws)
            laws.append((region, material.eps_r))
    return tuple(laws), index


def _permittivity(material: Material) -> tuple[np.ndarray, str]:
    """A material's lossy permittivity tensor, and the property it comes from.

    A law gives its eps_inf (isotropic, with no tan_delta beside it): its
    susceptibility is a piece of the port mode pencil.
    """
    eps_r = material.eps_r
    if isinstance(eps_r, PermittivityLaw):
        return float(eps_r.eps_inf) * np.eye(3, dtype=complex), "eps_r"
    # tan_delta alone can make the lossy permittivity a tensor
    return lossy_permittivity(material), "tan_delta" if np.ndim(eps_r) == 0 else "eps_r"


def _field(material: MaterialField, name: str, default: float, pos: np.ndarray):
    value = getattr(material, name)
    if value is None:
        return np.full(len(pos), default, complex)
    arr = np.asarray(value).astype(complex)
    return np.broadcast_to(arr, (len(pos),)) if arr.ndim == 0 else arr[pos]


def _inverse_permeability(mu_tt: np.ndarray, mu_nn: np.ndarray):
    """``(nu_tt, nu_nn)`` of permeability blocks: ``R^T mu_tt^-1 R``, ``1 / mu_nn``."""
    return CROSS_N.T @ np.linalg.inv(mu_tt) @ CROSS_N, 1 / mu_nn


#: The properties a material may hold as tensors, in table order.
_TENSOR_PROPERTIES = tuple(p.name for p in PROPERTIES if p.may_be_tensor)


def _names(names: list[str]) -> str:
    """``"a"``, ``"a and b"``, ``"a, b and c"``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _field_blocks(
    material: MaterialField,
    frame: np.ndarray,
    port: WavePort,
    region: str,
    pos: np.ndarray,
):
    """An anisotropic material field's lossy eps, mu and sigma blocks per triangle.

    Each is ``(tt, nn)``: ``(n, 2, 2)`` and ``(n,)`` in the port frame, from
    ``sum_k v_k a_k a_k^T`` per element. The normal-axis test reads the
    field's fixed axes, never its values (structural): an axis along the
    normal is needed whatever the principal values, so every tensor property
    the field gives fails it together, and the error names them all.
    """
    axes = np.asarray(material.axes, float)
    given = [n for n in _TENSOR_PROPERTIES if getattr(material, n) is not None]
    # distinct principal values: block-diagonal in the frame iff an axis is
    # along the normal
    probe = principal_tensor((1.0, 2.0, 3.0), material.axes)
    if given:
        _in_frame(probe, frame, port, region, _names(given))
    local = axes @ frame.T  # (3 axes, 3 frame components)

    def values(name: str) -> np.ndarray:
        return np.asarray(field_principal_values(material, name, pos)).astype(complex)

    eps = values("eps_r") * (1 - 1j * values("tan_delta").real)
    out = []
    for v in (eps, values("mu_r"), values("conductivity")):
        t = np.einsum("ek,ki,kj->eij", v, local, local)
        out.append((t[:, :2, :2], t[:, 2, 2]))
    return out


def _in_frame(tensor, frame, port: WavePort, region: str, what: str):
    local = frame @ tensor @ frame.T
    scale = np.abs(local).max()
    if np.abs(local[:2, 2]).max() > 1e-9 * scale:
        raise ValueError(
            f"WavePort {port.name!r}: region {region!r} has {what} with no "
            "principal axis along the port normal (tangential-normal coupling); "
            "the port mode problem supports only transverse-anisotropic materials, "
            "with a principal axis along the normal. Rotate the principal axes or "
            "the port, or end the port in an isotropic or transverse-anisotropic "
            "material; tilted principal axes are allowed away from ports."
        )
    return local[:2, :2], local[2, 2]


def _edge_keys(tris: np.ndarray) -> np.ndarray:
    return np.sort(tris[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)


def _edge_kinds(model: Model, port: WavePort) -> dict[str, object]:
    """What each named surface is to the port mode problem's edges.

    ``"pec"``, ``"absorbing"`` (``"absorbing2"`` at order 2), or the
    `Conductor` / `Impedance` itself.
    """
    kinds: dict[str, object] = {}
    for item in (*model.boundaries, *model.ports):
        for s in surfaces_of(item):
            if isinstance(item, PEC) or (isinstance(item, WavePort) and item != port):
                kinds[s] = "pec"
            elif isinstance(item, (Conductor, Impedance)):
                kinds[s] = item
            elif isinstance(item, Absorbing):
                kinds[s] = "absorbing2" if item.order == 2 else "absorbing"
    for s in port.pec_edges:
        kinds[s] = "pec"
    return kinds


def _surfaces_on_edges(
    model: Model, port: WavePort, port_edges: np.ndarray
) -> list[dict[str, bool]]:
    """The other surfaces bordering the domain along each port edge.

    Per edge, surface name -> whether a face of it on the edge has the mesh
    on one side only.
    """
    mesh = model.mesh
    nb = mesh.triangle_neighbours
    touches = ((nb >= 0) & model._domain.active[np.maximum(nb, 0)]).any(axis=1)  # noqa: SLF001
    other = (mesh.triangle_tags != mesh.region(port.surface).tag) & touches
    names = mesh._region_names(surfaces=True)  # noqa: SLF001
    o_edges = _edge_keys(mesh.triangles[other])
    o_tags = np.repeat(mesh.triangle_tags[other], 3)
    o_one = np.repeat((nb[other] < 0).any(axis=1), 3)
    _, ids = np.unique(
        np.concatenate([port_edges, o_edges]), axis=0, return_inverse=True
    )
    ids = ids.ravel()
    p_ids, o_ids = ids[: len(port_edges)], ids[len(port_edges) :]
    on = np.isin(o_ids, p_ids)
    found: dict[int, dict[str, bool]] = {}
    for eid, t, one in zip(o_ids[on], o_tags[on], o_one[on], strict=True):
        seen = found.setdefault(int(eid), {})
        seen[names[int(t)]] = seen.get(names[int(t)], False) or bool(one)
    return [found.get(int(e), {}) for e in p_ids]


def _edges(model, port, tris, used, *, points, tensors, isotropic, law_index, laws):
    """Perfect-conductor, absorbing and impedance edges of the cross-section.

    An edge on several surfaces takes the first of: perfect conductor,
    `Conductor` / `Impedance`, absorbing. ``tensors`` are the triangles'
    ``(eps_tt, eps_nn, nu_tt, nu_nn)``. Also the law index of the triangle
    beside each absorbing edge (``law_index`` of `_materials`), whose
    admittance is left zero.

    Raises:
        NotImplementedError: A second-order absorbing edge lies beside a law
            triangle (unverified).
    """
    port_edges = _edge_keys(tris)
    kinds = _edge_kinds(model, port)
    local = np.searchsorted(used, port_edges)
    pec, absorbing, owners, names, seen = [], [], [], [], set()
    imp_edges, imp_group, terms, fills = [], [], [], []
    sheet_fill = _sheet_fill_by_edge(model)
    for k, surfaces in enumerate(_surfaces_on_edges(model, port, port_edges)):
        key = tuple(local[k])
        if not surfaces or key in seen:
            continue
        seen.add(key)
        types = [kinds.get(s) for s in sorted(surfaces)]
        if "pec" in types:
            pec.append(local[k])
            continue
        sheets = [
            (kinds[s], one)
            for s, one in sorted(surfaces.items())
            if isinstance(kinds.get(s), (Conductor, Impedance))
        ]
        if sheets:
            item, one_sided = sheets[0]
            factor = 1.0
            if isinstance(item, Conductor) and item.thickness is not None and one_sided:
                factor = EXTERIOR_THICKNESS_FACTOR
            if (item, factor) not in terms:
                terms.append((item, factor))
            imp_edges.append(local[k])
            imp_group.append(terms.index((item, factor)))
            on_sheet = sheet_fill.get(item.surface, {})  # ty: ignore[unresolved-attribute]
            fills.append(on_sheet.get(tuple(port_edges[k]), 1.0))
            continue
        absorbing_types = [t for t in types if t in ("absorbing", "absorbing2")]
        if absorbing_types:
            absorbing.append(local[k])
            owners.append(k // 3)
            name = next(
                s
                for s, t in zip(sorted(surfaces), types, strict=True)
                if t in ("absorbing", "absorbing2")
            )
            names.append(name)
            g = law_index[k // 3]
            if g >= 0 and "absorbing2" in absorbing_types:
                raise NotImplementedError(
                    f"WavePort {port.name!r}: the second-order absorbing surface "
                    f"{name!r} meets region {laws[g][0]!r}, whose eps_r is a "
                    "permittivity law, along the port's cross-section; a "
                    "second-order absorbing edge beside a dispersive region is not "
                    "supported in the port mode problem. Use Absorbing(order=1) "
                    "there."
                )
    owners = np.asarray(owners, np.int64)
    admittance = _absorbing_admittance(
        port,
        points,
        np.asarray(absorbing).reshape(-1, 2),
        owners,
        surfaces=names,
        tensors=tensors,
        isotropic=isotropic,
    )
    absorbing_law = law_index[owners]
    # beside a law: the law's own edge piece (`gfem._engine.port_mode`)
    admittance[absorbing_law >= 0] = 0.0
    pairs = [
        np.asarray(e, dtype=np.int64).reshape(-1, 2)
        for e in (pec, absorbing, imp_edges)
    ]
    fill = np.asarray(fills, float) if sheet_fill else None
    impedance = (pairs[2], np.asarray(imp_group, np.int64), tuple(terms), fill)
    return pairs[0], pairs[1], admittance, impedance, absorbing_law


def _absorbing_admittance(port, points, edges, owners, *, surfaces, tensors, isotropic):
    """``(Y_tau, Y_n)`` per absorbing edge from the triangle beside it.

    An isotropic medium keeps its one value ``sqrt(eps / mu)`` in both.
    """
    eps_tt, eps_nn, nu_tt, nu_nn = (t[owners] for t in tensors)
    out = np.zeros((len(owners), 2))
    iso = isotropic[owners]
    if iso.any():
        eps = np.trace(eps_tt[iso], axis1=1, axis2=2).real / 2
        mu = 2 / np.trace(nu_tt[iso], axis1=1, axis2=2).real
        out[iso] = np.sqrt(eps / mu)[:, None]
    for name in dict.fromkeys(np.asarray(surfaces)[~iso]):
        rows = np.flatnonzero(~iso & (np.asarray(surfaces) == name))
        tangent = points[edges[rows, 1]] - points[edges[rows, 0]]
        normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        out[rows] = edge_admittance(
            eps_tt[rows],
            eps_nn[rows],
            nu_tt[rows],
            nu_nn[rows],
            normal,
            where=f"WavePort {port.name!r}: the absorbing surface {name!r}",
        )
    return out


def _sheet_fill_by_edge(model: Model) -> dict[str, dict[tuple[int, int], float]]:
    """Per sheet-field surface, the fill of a face beside each of its edges.

    Edges are sorted global node pairs (as `_edge_keys` gives them).
    """
    mesh = model.mesh
    out: dict[str, dict[tuple[int, int], float]] = {}
    for b in model.boundaries:
        sheet = sheet_of(b)
        if sheet is None:
            continue
        fill = np.asarray(sheet.fill, float)
        (surface,) = surfaces_of(b)
        keys = _edge_keys(mesh.triangles[mesh.elements(surface)])
        values = np.repeat(fill, 3)
        out[surface] = {
            (int(a), int(c)): float(w) for (a, c), w in zip(keys, values, strict=True)
        }
    return out
