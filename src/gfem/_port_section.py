"""The cross-section of a wave port: a planar 2D mesh with materials and edges.

Pure NumPy on the `Model` arrays; the engine turns it into an NGSolve mesh.
The port mode problem reads the material of the volume element behind each
port triangle and the boundary conditions of the 3D surfaces that meet the
port along an edge.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from gfem._boundaries import (
    PEC,
    Absorbing,
    Conductor,
    Impedance,
    WavePort,
    surfaces_of,
)
from gfem._materials import (
    Material,
    MaterialField,
    lossy_permittivity,
    principal_tensor,
)
from gfem.conventions import EXTERIOR_THICKNESS_FACTOR
from gfem.errors import PortGeometryError
from gfem.mesh import LENGTH_UNITS
from gfem.model import Model

# Relative distance from the port plane (in port diameters) still on the plane.
PLANE_RTOL = 1e-6
# a x z = R a for a transverse vector a = (a_u, a_v).
_ROT = np.array([[0.0, 1.0], [-1.0, 0.0]])


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
            without conductivity.
        eps_nn: Complex relative permittivity along ``n``, ``(n_tri,)``.
        nu_tt: ``R^T mu_tt^-1 R`` with ``R a = a x n``, ``(n_tri, 2, 2)``.
        nu_nn: ``1 / mu_nn``, ``(n_tri,)``.
        sigma: Electrical conductivity in S/m, ``(n_tri,)``.
        pec_edges: ``(n, 2)`` node pairs on perfect conductors.
        absorbing_edges: ``(n, 2)`` node pairs on absorbing boundaries.
        absorbing_admittance: Relative wave admittance ``sqrt(eps / mu)`` of
            the medium beside each absorbing edge, ``(n,)``.
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
    sigma: np.ndarray
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

    @property
    def diameter(self) -> float:
        """Largest extent of the cross-section, local units."""
        return float(np.ptp(self.points, axis=0).max())

    def index_squared(self, sigma_term: float = 0.0) -> tuple[float, float]:
        """Bounds on the squared bulk index over the cross-section.

        Args:
            sigma_term: ``max conductivity / (omega eps0)`` to add to ``|eps|``.

        Returns:
            ``max Re(eps) max Re(mu)`` (the shift, as in Palace) and
            ``(max |eps| + sigma_term) max |mu|`` (an upper bound on the
            ``|n|^2`` of any propagating mode).
        """
        eps = np.concatenate([self.eps_tt[:, 0, 0], self.eps_tt[:, 1, 1], self.eps_nn])
        mu = 1 / np.concatenate([self.nu_tt[:, 0, 0], self.nu_tt[:, 1, 1], self.nu_nn])
        return (
            float(eps.real.max() * mu.real.max()),
            float((np.abs(eps).max() + sigma_term) * np.abs(mu).max()),
        )

    def to_local(self, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Plane coordinates ``(n, 2)`` and normal distances of global points (m)."""
        rel = np.asarray(points_m, float) / self.scale - self.origin
        return rel @ self.frame[:2].T, rel @ self.frame[2]

    def to_global(self, local: np.ndarray) -> np.ndarray:
        """Global points (m) of plane coordinates ``(n, 2)``."""
        return (self.origin + np.asarray(local, float) @ self.frame[:2]) * self.scale


def port_section(model: Model, port: WavePort) -> PortSection:
    """Extract the cross-section of ``port`` from ``model``.

    Raises:
        PortGeometryError: The port surface is not planar.
        ValueError: An anisotropic material beside the port has no principal
            axis along the port normal.
    """
    mesh = model.mesh
    scale = LENGTH_UNITS[mesh.length_unit]
    nodes = np.asarray(mesh.nodes, float) / scale
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

    eps_tt, eps_nn, nu_tt, nu_nn, sigma = _materials(model, port, tets, frame)
    pec, absorbing, admittance, (imp_edges, imp_group, imp_terms, imp_fill) = _edges(
        model, port, tris, used, eps_tt=eps_tt, nu_tt=nu_tt
    )
    return PortSection(
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
        sigma=sigma,
        pec_edges=pec,
        absorbing_edges=absorbing,
        absorbing_admittance=admittance,
        faces=sel,
        impedance_edges=imp_edges,
        impedance_group=imp_group,
        impedance_terms=imp_terms,
        impedance_fill=imp_fill,
    )


def _materials(model: Model, port: WavePort, tets: np.ndarray, frame: np.ndarray):
    """Per-triangle material values in the port frame."""
    mesh = model.mesh
    names = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    n = len(tets)
    eps_tt = np.zeros((n, 2, 2), complex)
    eps_nn = np.zeros(n, complex)
    nu_tt = np.zeros((n, 2, 2), complex)
    nu_nn = np.zeros(n, complex)
    sigma = np.zeros(n)
    region_of = mesh.tetrahedron_tags[tets]
    for rtag in np.unique(region_of):
        region = names[int(rtag)]
        rows = np.flatnonzero(region_of == rtag)
        material = model.materials[region]
        if isinstance(material, Material):
            eps = lossy_permittivity(material)
            mu = principal_tensor(material.mu_r, material.axes).astype(complex)
            et, en = _in_frame(eps, frame, port, region, "eps_r")
            mt, mn = _in_frame(mu, frame, port, region, "mu_r")
            eps_tt[rows], eps_nn[rows] = et, en
            nu_tt[rows], nu_nn[rows] = _ROT.T @ np.linalg.inv(mt) @ _ROT, 1 / mn
            sigma[rows] = material.conductivity
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
            sigma[rows] = _field(material, "conductivity", 0.0, pos).real
    return eps_tt, eps_nn, nu_tt, nu_nn, sigma


def _field(material: MaterialField, name: str, default: float, pos: np.ndarray):
    value = getattr(material, name)
    if value is None:
        return np.full(len(pos), default, complex)
    arr = np.asarray(value).astype(complex)
    return np.broadcast_to(arr, (len(pos),)) if arr.ndim == 0 else arr[pos]


def _in_frame(tensor, frame, port: WavePort, region: str, what: str):
    local = frame @ tensor @ frame.T
    scale = np.abs(local).max()
    if np.abs(local[:2, 2]).max() > 1e-9 * scale:
        raise ValueError(
            f"WavePort {port.name!r}: region {region!r} has an anisotropic {what} "
            "with no principal axis along the port normal; the port mode problem "
            "needs one (rotate the material axes, or move the port)."
        )
    return local[:2, :2], local[2, 2]


def _edge_keys(tris: np.ndarray) -> np.ndarray:
    return np.sort(tris[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)


def _edge_kinds(model: Model, port: WavePort) -> dict[str, object]:
    """What each named surface is to the port mode problem's edges.

    ``"pec"``, ``"absorbing"``, or the `Conductor` / `Impedance` itself.
    """
    kinds: dict[str, object] = {}
    for item in (*model.boundaries, *model.ports):
        for s in surfaces_of(item):
            if isinstance(item, PEC) or (isinstance(item, WavePort) and item != port):
                kinds[s] = "pec"
            elif isinstance(item, (Conductor, Impedance)):
                kinds[s] = item
            elif isinstance(item, Absorbing):
                kinds[s] = "absorbing"
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
    names = {r.tag: r.name for r in mesh.regions if r.dim == 2}
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


def _edges(model, port, tris, used, *, eps_tt, nu_tt):
    """Perfect-conductor, absorbing and impedance edges of the cross-section.

    An edge on several surfaces takes the first of: perfect conductor,
    `Conductor` / `Impedance`, absorbing.
    """
    port_edges = _edge_keys(tris)
    kinds = _edge_kinds(model, port)
    local = np.searchsorted(used, port_edges)
    pec, absorbing, admittance, seen = [], [], [], set()
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
        if "absorbing" in types:
            absorbing.append(local[k])
            t = k // 3
            eps = np.trace(eps_tt[t]).real / 2
            mu = 2 / np.trace(nu_tt[t]).real
            admittance.append(np.sqrt(eps / mu))
    pairs = [
        np.asarray(e, dtype=np.int64).reshape(-1, 2)
        for e in (pec, absorbing, imp_edges)
    ]
    fill = np.asarray(fills, float) if sheet_fill else None
    impedance = (pairs[2], np.asarray(imp_group, np.int64), tuple(terms), fill)
    return pairs[0], pairs[1], np.asarray(admittance, float), impedance


def _sheet_fill_by_edge(model: Model) -> dict[str, dict[tuple[int, int], float]]:
    """Per sheet-field surface, the fill of a face beside each of its edges.

    Edges are sorted global node pairs (as `_edge_keys` gives them).
    """
    mesh = model.mesh
    out: dict[str, dict[tuple[int, int], float]] = {}
    for b in model.boundaries:
        if not isinstance(b, (Conductor, Impedance)) or b.fill is None:
            continue
        fill = np.asarray(b.fill.fill, float)
        keys = _edge_keys(mesh.triangles[mesh.elements(b.surface)])
        values = np.repeat(fill, 3)
        out[b.surface] = {
            (int(a), int(c)): float(w) for (a, c), w in zip(keys, values, strict=True)
        }
    return out
