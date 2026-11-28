"""Post-hoc measurements on result fields.

A result carries every quantity that needs no extra input. A quantity that
needs extra physics input (a probe point, an interface surface with its
thickness and permittivity, a voltage path) is a measurement: a function
called on a result that carries its fields, after the solve, also on a
result reloaded with `gfem.load_result`.

- `Field.at`: a field's value at points.
- `interface_participation`: the electric energy of a mode in a thin
  dielectric layer on a surface, and its share of ``1 / Q``.
- `characteristic_impedance`: the complex impedance of a port mode from a
  voltage path, a current path or both (power-voltage, power-current or
  voltage-current).
- `junction_coupling`: the complex power coupling between two
  port modes of one section, the loss of a butt junction between a straight
  and a bent guide.
- `power_flow`: the net power a driven field carries out of a set of
  elements, exact for the solved field (a flux of point-sampled
  ``E x H*`` is not, see the measurements recipe).
- `optical_force`: the time-averaged force and torque a driven or eigenmode
  field exerts on a body, the exact derivative of the discrete Lagrangian
  under a rigid motion of the body.
- `dissipated_power`: the time-averaged power a driven field dissipates
  in each volume region and on each `Conductor` or `Impedance` sheet,
  exact for the solved field.
- `heat_flow` and `heat_flow_per_length`: the heat a steady temperature
  carries out of a set of elements or through named surfaces, exact for
  the solved temperature.
- `electric_energy`: the time-averaged electric energy a field stores in
  each volume region, with the model's materials.
- `region_mean`: a field's mean over named volume regions, exact for the
  solved field (an electric field's, or a potential's).
- `strain`: the engineering strain of a piezoelectric solve's displacement
  at points, exact for the solved field.
- `purcell_factor`: a point dipole's emitted power over its power in the
  unbounded material at its position.
- `far_field`: the far-field amplitude of a port, dipole or incident-field
  column on a grid of directions, with the IEEE 145 radiation outputs
  (`FarField`).
- `multipoles`: the spherical multipoles of the field an incident wave
  scatters, from the far field of the scattered field (`Multipoles`).
- `current_moments`: the long-wavelength electric dipole and the toroidal
  dipole of the induced current in a scatterer (`CurrentMoments`).
- `beam_coupling`: the amplitude a field couples into an incoming beam
  (a fiber), by reciprocity the S-parameter between the beam and a port.

Measurements run on the CPU and work under `jax.jit` and `jax.vmap`; under
`jax.grad` they differentiate with respect to the field coefficients (so
material-field gradients flow through the solve that made the field) and
the measurement's own numbers. That includes the normalized mode fields of
an eigenmode solve, which differentiate with respect to `gfem.MaterialField`
properties and `gfem.SheetField` fills for an isolated mode (see the
"Differentiate eigenfrequencies, Q and participation ratios" recipe).
Mesh-node gradients raise `gfem.NonDifferentiableError`: for a shape
derivative of a measurement, take finite differences of it over meshes
moved by `gfem.shape.boundary_morph`.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem import _curved as curved
from gfem import _mpi, _objective, _probe
from gfem._adjoint.measure import measure
from gfem._beam_coupling import beam_coupling
from gfem._checks import check_fixed, is_traced
from gfem._dim2 import entry_refusal, refuse_entry
from gfem._far_field import FarField, far_field
from gfem._materials import Material, _is_law, field_principal_values
from gfem._multipoles import (
    CurrentMoments,
    Multipoles,
    current_moments,
    multipoles,
)
from gfem._purcell import purcell_factor
from gfem._pytree import register_fields
from gfem._truncation import truncation
from gfem.conventions import MODE_ENERGY
from gfem.errors import MeshRefinementUnsupportedError, NonDifferentiableError
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.model import Model
    from gfem.results import (
        DrivenResult,
        EigenmodeResult,
        PortModeResult,
        SteadyThermalResult,
    )

__all__ = [
    "INTERFACE_KINDS",
    "CrossSections",
    "CurrentMoments",
    "FarField",
    "InterfaceParticipation",
    "Multipoles",
    "OpticalForce",
    "beam_coupling",
    "characteristic_impedance",
    "cross_sections",
    "current_moments",
    "dissipated_power",
    "electric_energy",
    "element_mean",
    "far_field",
    "heat_flow",
    "heat_flow_per_length",
    "interface_participation",
    "junction_coupling",
    "multipoles",
    "optical_force",
    "power_flow",
    "purcell_factor",
    "region_mean",
    "strain",
]

# --- interface participation -------------------------------------------------------

#: The thin-layer models of `interface_participation` (Wenner et al., Appl.
#: Phys. Lett. 99, 113513 (2011)).
InterfaceKind = Literal["default", "MA", "MS", "SA"]
INTERFACE_KINDS = get_args(InterfaceKind)
#: A medium is vacuum when its slowest light speed is within this of c.
_VACUUM = 1.0 - 1e-6


@dataclasses.dataclass(frozen=True, eq=False)
class InterfaceParticipation:
    """The share of each mode's electric energy in a thin interface layer.

    Attributes:
        participation: ``(n_modes,)`` the layer's electric energy over the
            mode's electric energy ``E`` (see `gfem.EigenmodeResult`).
        inverse_q: ``(n_modes,)`` the layer's dielectric-loss contribution to
            ``1 / Q``, ``participation * tan_delta``; ``None`` when no loss
            tangent was given.
    """

    participation: jax.Array
    inverse_q: jax.Array | None = None

    @property
    def q(self) -> jax.Array:
        """The Q the layer's loss alone would give, ``1 / inverse_q``.

        Raises:
            ValueError: No loss tangent was given.
        """
        if self.inverse_q is None:
            raise ValueError(
                "This interface participation has no loss: pass tan_delta= to "
                "gfem.interface_participation."
            )
        return 1.0 / self.inverse_q

    def __repr__(self) -> str:
        try:
            p = np.asarray(self.participation)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return "InterfaceParticipation(traced)"
        text = ", ".join(f"{x:.4g}" for x in p)
        if self.inverse_q is not None:
            text += "; 1/Q " + ", ".join(f"{x:.4g}" for x in np.asarray(self.inverse_q))
        return f"InterfaceParticipation([{text}])"


register_fields(InterfaceParticipation, ("participation", "inverse_q"))


def interface_participation(
    modes: EigenmodeResult,
    surface: str | Sequence[str],
    thickness: Any,
    eps_r: Any,
    *,
    kind: InterfaceKind = "default",
    tan_delta: Any = None,
) -> InterfaceParticipation:
    """Electric energy participation of a thin dielectric layer on a surface.

    The layer (an oxide on a metal, a residue on a substrate) is too thin to
    mesh: it has thickness ``t`` and relative permittivity ``eps_r`` and
    sits on ``surface``. Its electric energy follows from the mode's field
    at the surface, for each kind of interface (Wenner et al. 2011):

    - ``"MA"`` (metal-air): ``t / eps_r * |E_n|^2``, ``E`` on the vacuum
      side of the surface.
    - ``"MS"`` (metal-substrate): ``t / eps_r * |(eps_s E)_n|^2``, ``E``
      and the substrate permittivity ``eps_s`` (real part, a tensor if
      anisotropic) on the non-vacuum side: the layer carries the
      substrate's normal flux.
    - ``"SA"`` (substrate-air): ``t * (eps_r |E_t|^2 + |E_n|^2 / eps_r)``,
      ``E`` on the vacuum side.
    - ``"default"``: ``t * eps_r * |E|^2``, ``E`` averaged over the two
      sides of an interior surface. Not for a film with the same medium on
      both faces: there the two faces' normal fields are opposite and the
      mean cancels.

    each times ``eps0 / 2`` and integrated over the surface. A medium is
    vacuum when no wave in it is slower than in vacuum: its slowest light
    speed, from its largest principal ``eps_r`` and ``mu_r``, is that of
    vacuum (to 1e-6), so an anisotropic side such as ``eps_r = (1, 1, 4)`` is
    not vacuum. The layer itself is isotropic. For ``"MA"`` and ``"MS"``
    each side of the wanted kind carries its own layer and the layers'
    energies add: a film with vacuum on both faces has one oxide per face,
    and its ``"MA"`` participation is the sum of the two faces'. Asked on a
    surface that is not a film and has the same medium on both sides, they
    count the layer twice, which is a misuse. For ``"SA"`` and ``"default"``
    the field is averaged over the sides of the wanted kind. Where no side
    is of the wanted kind (``"MA"`` on a plate under substrate), the surface
    adds nothing. Sides inside volumes the model excludes from the solve
    (metal) do not count. A face with a side in a perfectly matched layer
    (`gfem.PML`) does not count either: the field there is the stretched
    one, not a physical field, so a ground plane that runs into a side
    layer counts its part in the physical regions. The participation is
    that energy over the mode's electric energy ``E`` (1 J, see
    `gfem.conventions`), and ``participation * tan_delta`` is the layer's
    contribution to ``1 / Q``.

    On a zero-thickness film the field at the film edge is singular, and the
    participation of a layer that reaches the edge grows without bound as
    the mesh is refined there: by about 6 to 7 % of its value each time the
    edge cells are halved, on a 10 um strip. It is not a mesh-converged
    number. Compare designs on meshes with the same edge cell size, and take
    that size near the real film thickness for an absolute value.

    Differentiable with respect to ``thickness``, ``eps_r``, ``tan_delta``,
    the mode fields and the substrate's `MaterialField` permittivity (the
    ``"MS"`` flux), per principal axis for a field with ``axes``. Through
    the eigenmode solve, the mode fields differentiate with respect to
    `gfem.MaterialField` properties and `gfem.SheetField` fills of an
    isolated mode (see the "Differentiate eigenfrequencies, Q and
    participation ratios" recipe; a mode of a degenerate cluster raises
    `gfem.DegenerateModeError`). Mesh-node gradients raise
    `gfem.NonDifferentiableError`: take finite differences of the
    participation over meshes moved by `gfem.shape.boundary_morph` instead.

    On a result solved across ranks the mode fields are each rank's pieces, and
    the call is collective: every rank makes it with the same arguments and gets
    the same participation.

    Args:
        modes: An eigenmode result.
        surface: A surface region, or several (their layers add up).
        thickness: The layer thickness ``t``, m, > 0.
        eps_r: The layer's relative permittivity, > 0.
        kind: One of `INTERFACE_KINDS`.
        tan_delta: The layer's loss tangent, >= 0, or ``None``.

    Returns:
        Per-mode participation and, with ``tan_delta``, its ``1 / Q``.

    Raises:
        TypeError: ``modes`` is not a `gfem.EigenmodeResult`.
        RegionNotFoundError: The mesh has no region of that name.
        TypeError: ``eps_r`` is a `gfem.PermittivityLaw` (the layer is a
            number).
        ValueError: A name is a volume region or a surface of a
            `gfem.Periodic` pair (interior to the periodic cell), a number is
            out of range, the result does not carry its materials (saved
            by an older gfem), a side of the surface lies in a region
            with a permittivity law, or every face of the surfaces has a
            side in a perfectly matched layer.
    """
    from gfem.results import EigenmodeResult

    fields = tuple(getattr(modes, "e_fields", ()))
    mesh = getattr(fields[0], "mesh", None) if fields else None
    refuse_entry(mesh, "interface_participation")
    if not isinstance(modes, EigenmodeResult):
        raise TypeError(
            "interface_participation needs a gfem.EigenmodeResult, got "
            f"{type(modes).__name__}."
        )
    if kind not in INTERFACE_KINDS:
        raise ValueError(
            f"Unknown interface kind {kind!r}; expected one of "
            f"{', '.join(repr(k) for k in INTERFACE_KINDS)}."
        )
    _positive(thickness, "thickness", strict=True)
    if _is_law(eps_r):
        raise TypeError(
            "The interface layer's eps_r must be a number; dispersive layers are "
            f"not supported (got {type(eps_r).__name__})."
        )
    _positive(eps_r, "eps_r", strict=True)
    if tan_delta is not None:
        _positive(tan_delta, "tan_delta", strict=False)
    if not modes.e_fields:
        raise ValueError("This eigenmode result holds no mode fields.")
    if not modes.materials:
        raise ValueError(
            "This eigenmode result does not carry the materials it was solved "
            "with (it was saved by an older gfem): solve the model again."
        )
    field = modes.e_fields[0]
    surfaces = (surface,) if isinstance(surface, str) else tuple(surface)
    # read from the result, never from a cache: the pairs of this model only
    for pair in modes.periodic_surfaces:
        for name in surfaces:
            if name in pair:
                raise ValueError(
                    f"{name!r} is a surface of periodic pair {pair!r}, which "
                    "is interior to the periodic cell: it holds no interface "
                    "layer."
                )
    comm = field._communicator  # noqa: SLF001
    with _mpi.section(comm):
        _mpi.check_call(
            comm,
            "interface_participation",
            {
                "the surfaces": np.array(surfaces),
                "the kind": np.array(kind),
                "the thickness": thickness,
                "the layer's eps_r": eps_r,
                "the loss tangent": np.nan if tan_delta is None else tan_delta,
            },
        )
    layer = () if modes.pml is None else modes.pml.regions
    geometry = _surface_geometry(field, surfaces, layer)
    curved.refuse_surface(
        field.mesh,
        "interface_participation",
        geometry.triangles,
        "gfem.interface_participation",
    )
    _refuse_law_sides(modes.materials, field.mesh, geometry, surfaces)
    coefficients = jnp.stack([f.coefficients for f in modes.e_fields])
    energy = _layer_energy(
        geometry,
        kind,
        _mpi.static_comm(comm),
        coefficients=coefficients,
        nodes=field.mesh.nodes,
        materials=modes.materials,
        thickness=jnp.asarray(thickness, float),
        eps_l=jnp.asarray(eps_r, float),
    )
    participation = energy / MODE_ENERGY
    inverse_q = None if tan_delta is None else participation * jnp.asarray(tan_delta)
    return InterfaceParticipation(participation, inverse_q)


class _SurfaceGeometry:
    """The topology of a measurement surface: sides, quadrature points.

    Hashed by identity: one object per (mesh topology, surfaces, field
    layout), so `_layer_energy` compiles once per surface.
    """

    def __init__(
        self,
        mesh: Mesh,
        surfaces: tuple[str, ...],
        layout: Any,
        layer: tuple[str, ...] = (),
    ) -> None:
        triangles = []
        for name in surfaces:
            region = mesh.region(name)
            if region.dim != mesh.dim - 1:
                raise ValueError(
                    f"{name!r} is a volume region; interface_participation needs "
                    "a surface region."
                )
            triangles.append(mesh.elements(name))
        engine = _probe.engine()
        tri = np.concatenate(triangles)
        sides = mesh.triangle_neighbours[tri]
        inside = engine.in_support(mesh, layout)
        if layer:
            # a face counts when no side of it lies in the perfectly matched
            # layer: there the field is the stretched one, which holds no
            # energy of the device
            in_layer = np.isin(
                mesh.tetrahedron_tags, [mesh.region(r).tag for r in layer]
            )
            held = (sides >= 0) & inside[np.maximum(sides, 0)]
            physical = ~(held & in_layer[np.maximum(sides, 0)]).any(axis=1)
            if not physical.any():
                raise ValueError(
                    "interface_participation on "
                    f"{', '.join(map(repr, surfaces))}: every face of the surface "
                    "lies in the perfectly matched layer "
                    f"({', '.join(map(repr, layer))}), where the field is not "
                    "physical. Measure a surface in the physical regions."
                )
            tri, sides = tri[physical], sides[physical]
        self.topology = mesh._topology  # noqa: SLF001
        self.layout = layout
        self.triangles = tri
        self.sides = sides
        self.exists = (sides >= 0) & inside[np.maximum(sides, 0)]
        bary, self.weights = engine.triangle_rule(2 * layout.order)
        self.n_q = len(bary)
        # the evaluation points: the quadrature points seen from each side
        self.pairs = [np.flatnonzero(self.exists[:, s]) for s in (0, 1)]
        parts = [
            engine.face_points(mesh, tri[k], sides[k, s], bary)
            for s, k in enumerate(self.pairs)
        ]
        self.located = engine.Located(
            np.concatenate([p.tetrahedra for p in parts]),
            np.concatenate([p.barycentric for p in parts]),
        )


_GEOMETRIES: dict[tuple, _SurfaceGeometry] = {}
_MAX_GEOMETRIES = 8


def _surface_geometry(
    field: Field, surfaces: tuple[str, ...], layer: tuple[str, ...] = ()
) -> _SurfaceGeometry:
    mesh = field.mesh
    layout = _probe.layout(field)
    key = (id(mesh._topology), surfaces, layout, layer)  # noqa: SLF001
    found = _GEOMETRIES.get(key)
    if found is not None and found.topology is mesh._topology:  # noqa: SLF001
        return found
    made = _SurfaceGeometry(mesh, surfaces, layout, layer)
    _GEOMETRIES[key] = made
    while len(_GEOMETRIES) > _MAX_GEOMETRIES:
        _GEOMETRIES.pop(next(iter(_GEOMETRIES)))
    return made


@functools.partial(jax.jit, static_argnames=("geometry", "kind", "comm"))
def _layer_energy(
    geometry: _SurfaceGeometry,
    kind: InterfaceKind,
    comm: _mpi.StaticComm | None,
    *,
    coefficients: jax.Array,
    nodes: jax.Array,
    materials: dict[str, Any],
    thickness: jax.Array,
    eps_l: jax.Array,
) -> jax.Array:
    """The layer's electric energy per mode, J (see `interface_participation`).

    Under ranks (``comm``) the coefficients are each rank's pieces, and the values
    at the points every rank's (`gfem._probe.evaluate`).
    """
    g = geometry
    mesh = Mesh(nodes, g.topology)
    m, n_t, n_q = coefficients.shape[0], len(g.triangles), g.n_q
    values = _probe.evaluate(
        coefficients, mesh, g.layout, g.located, None if comm is None else comm.comm
    )
    e = jnp.zeros((m, n_t, 2, n_q, 3), dtype=values.dtype)
    start = 0
    for s, k in enumerate(g.pairs):
        block = values[:, start : start + len(k) * n_q]
        e = e.at[:, k, s].set(block.reshape(m, len(k), n_q, 3))
        start += len(k) * n_q

    v = nodes[mesh.triangles[g.triangles]]
    cross = jnp.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    twice_area = jnp.linalg.norm(cross, axis=1)
    normal = cross / twice_area[:, None]
    weights = 0.5 * twice_area[:, None] * g.weights[None, :]  # (n_t, q), m^2

    eps_side, vacuum = _media(materials, mesh, g.sides, g.exists)
    ex = jnp.asarray(g.exists)
    if kind == "default":
        density = eps_l * _norm2(_average(e, ex))
    elif kind == "MS":
        # one layer per selected side: density per side, then summed
        d = jnp.einsum("tsij,mtsqj->mtsqi", eps_side, e)
        dn = jnp.einsum("mtsqi,ti->mtsq", d, normal)
        density = _sum_sides(_sq(dn), ex & ~vacuum) / eps_l
    elif kind == "MA":
        en = jnp.einsum("mtsqi,ti->mtsq", e, normal)
        density = _sum_sides(_sq(en), ex & vacuum) / eps_l
    else:  # SA
        selected = _average(e, ex & vacuum)
        en = jnp.einsum("mtqi,ti->mtq", selected, normal)
        et = selected - en[..., None] * normal[None, :, None, :]
        density = eps_l * _norm2(et) + _sq(en) / eps_l
    return 0.5 * const.epsilon_0 * thickness * jnp.einsum("mtq,tq->m", density, weights)


def _refuse_law_sides(
    materials: Any, mesh: Mesh, geometry: _SurfaceGeometry, surfaces: tuple[str, ...]
) -> None:
    """Raise if a side of a measured surface lies in a permittivity law region.

    The ``"MS"`` and ``"SA"`` layers read a side's Re eps_r, which a law does
    not have. A law region that touches no measured face is
    skipped by `_media`.

    Raises:
        ValueError: A law region has a tetrahedron on a side of ``surfaces``.
    """
    sides = geometry.sides[geometry.exists]
    for region, material in materials.items():
        if not (isinstance(material, Material) and _is_law(material.eps_r)):
            continue
        if np.isin(sides, mesh.elements(region)).any():
            raise ValueError(
                f"interface_participation on {', '.join(map(repr, surfaces))}: "
                f"region {region!r} on a side of the surface holds a permittivity "
                "law, which has no one permittivity for the layer's field. Measure "
                "an interface between constant materials."
            )


def _positive(value: Any, name: str, *, strict: bool) -> None:
    if is_traced(value):
        return
    x = np.asarray(value)
    if x.shape != () or not np.isrealobj(x) or not np.isfinite(x):
        raise ValueError(f"{name} must be one finite real number, got {value!r}.")
    if (x <= 0) if strict else (x < 0):
        bound = "> 0" if strict else ">= 0"
        raise ValueError(f"{name} must be {bound}, got {value!r}.")


def _sq(x: jax.Array) -> jax.Array:
    """``|x|^2`` elementwise."""
    return jnp.real(x) ** 2 + jnp.imag(x) ** 2


def _norm2(x: jax.Array) -> jax.Array:
    """``|x|^2`` of vectors along the last axis."""
    return _sq(x).sum(axis=-1)


def _average(x: jax.Array, selected: jax.Array) -> jax.Array:
    """Mean over the selected sides: ``(m, t, 2, q, ...)`` -> ``(m, t, q, ...)``."""
    w = selected.astype(float)
    count = jnp.maximum(w.sum(axis=1), 1.0)
    w = w / count[:, None]
    shape = (1, *w.shape, 1) + (1,) * (x.ndim - 4)
    return (x * w.reshape(shape)).sum(axis=2)


def _sum_sides(x: jax.Array, selected: jax.Array) -> jax.Array:
    """Sum over the selected sides: ``(m, t, 2, q)`` -> ``(m, t, q)``."""
    w = selected.astype(float)
    return (x * w[None, :, :, None]).sum(axis=2)


def _media(
    materials: dict[str, Any], mesh: Mesh, sides: np.ndarray, exists: np.ndarray
) -> tuple[jax.Array, jax.Array]:
    """Real permittivity tensors ``(t, 2, 3, 3)`` and vacuum flags ``(t, 2)``."""
    from gfem._engine.coefficients import _tensor

    tets = np.maximum(sides, 0)
    eps = jnp.zeros((*sides.shape, 3, 3)) + jnp.eye(3)
    # (c_min / c0)^2 >= 1 / (eps_max mu_max): no wave slower than in vacuum
    # means the side's permittivity is eps0 I, which the layers need
    speed2 = jnp.ones(sides.shape)
    for region, material in materials.items():
        members = mesh.elements(region)
        position = np.full(len(mesh.tetrahedra), -1)
        position[members] = np.arange(len(members))
        inside = jnp.asarray((position[tets] >= 0) & exists)
        if isinstance(material, Material) and _is_law(material.eps_r):
            # on no measured side (`_refuse_law_sides`): never read
            continue
        if isinstance(material, Material):
            tensor = jnp.asarray(_tensor(material).real)
            high = float(np.max(material.eps_r)) * float(np.max(material.mu_r))
            value = jnp.broadcast_to(tensor, eps.shape)
            high = jnp.full(sides.shape, high)
        elif material.axes is not None:
            # principal values along fixed axes: the tensor per element, and
            # the bound from the largest principal values
            index = np.maximum(position[tets], 0)
            e_r, m_r = (
                jnp.real(jnp.asarray(field_principal_values(material, name, index)))
                for name in ("eps_r", "mu_r")
            )
            a = jnp.asarray(material.axes)
            value = jnp.einsum("...k,ki,kj->...ij", e_r, a, a)
            value = jnp.broadcast_to(value, eps.shape)
            high = jnp.max(e_r, axis=-1) * jnp.max(m_r, axis=-1)
            high = jnp.broadcast_to(high, sides.shape)
        else:
            e_r = 1.0 if material.eps_r is None else material.eps_r
            m_r = 1.0 if material.mu_r is None else material.mu_r
            e_r = jnp.real(jnp.asarray(e_r))
            m_r = jnp.real(jnp.asarray(m_r))
            index = np.maximum(position[tets], 0)
            e_r = e_r[index] if e_r.ndim else jnp.full(sides.shape, e_r)
            m_r = m_r[index] if m_r.ndim else jnp.full(sides.shape, m_r)
            value = e_r[..., None, None] * jnp.eye(3)
            high = e_r * m_r
        eps = jnp.where(inside[..., None, None], value, eps)
        speed2 = jnp.where(inside, 1.0 / high, speed2)
    vacuum = speed2 >= _VACUUM**2
    return eps, vacuum


# --- junction coupling -------------------------------------------------------------


def junction_coupling(model: Model, a: PortModeResult, b: PortModeResult) -> jax.Array:
    r"""Power coupling :math:`\tau^2` between two port modes of one section.

    A butt junction between two guides that share a cross-section plane (a
    straight guide and a bent one, two bends of different radius or side,
    two straight modes) launches the mode ``b`` from the mode ``a`` with the
    amplitude :math:`\tau`,

    .. math::

        \tau^2 = \frac{\big(\tfrac12(\langle a, b\rangle +
        \langle b, a\rangle)\big)^2}{\langle a, a\rangle\langle b, b\rangle},
        \qquad
        \langle a, b\rangle = \tfrac12 \int (\mathbf{E}_a \times
        \mathbf{H}_b)\cdot\mathbf{n}\, \mathrm{d}A,

    over the whole section, a perfectly matched layer included, with the
    reciprocity pairing of the port mode problem: each mode's :math:`\mathbf{H}`
    comes from its own :math:`\mu^{-1}` (the bend's metric is in it), and
    the pairing is unconjugated on purpose, as the modes of a leaky guide
    are not orthogonal under a conjugated one. The two modes are normalized
    out, so :math:`\tau^2` does not depend on their phase or power; it is 1
    for a mode with itself, symmetric in ``a`` and ``b``, and 0 between two
    orthogonal straight modes. The junction loss is

    .. math:: L = -10 \log_{10} |\tau^2| \;\mathrm{dB},

    which counts the power the junction leaves out of ``b`` into the other
    modes and the radiation; the reflection at the junction is neglected.
    For a lossless guide the loss is the overlap of the two modes. Returns
    the raw complex :math:`\tau^2`, not the loss.

    Under `jax.grad` it differentiates with respect to the bend radius of
    either mode, and to the permittivity, laws and `gfem.MaterialField`
    values through the solves that made the modes. The inverse permeability
    of a mode's section enters the pairing directly: a traced ``mu_r`` (or
    principal axis) raises `gfem.NonDifferentiableError`, as the node
    coordinates do.

    Example:
        The loss of the junction between a straight guide and the same guide
        bent with radius ``R``, in dB::

            problem = gfem.PortMode("p1", (f,), bend=gfem.Bend(R, outward="+x", at=at))
            straight = gfem.solve(model, gfem.PortMode("p1", (f,)))
            bent = gfem.solve(model, problem)
            loss = -10 * jnp.log10(
                jnp.abs(gfem.junction_coupling(model, straight, bent))
            )

    Args:
        model: The model both modes were solved on.
        a: A port mode result (straight or bent).
        b: A port mode result of the same port, element order, section and
            frequency grid.

    Returns:
        ``(n_freq,)`` complex, dimensionless.

    Raises:
        TypeError: An argument is not a model or a `gfem.PortModeResult`
            with an electric field.
        ValueError: The modes are of different ports, element orders,
            sections or frequency grids, or ``model`` is not the one they
            were solved on.
        NotImplementedError: A mode of a line port.
        NonDifferentiableError: The node coordinates, or a ``mu_r`` or
            principal axis of the model, are traced.
    """
    from gfem._engine.junction import junction_tau2

    return junction_tau2(model, a, b)


# --- characteristic impedance ------------------------------------------------------


def characteristic_impedance(
    mode: PortModeResult,
    voltage_path: Any = None,
    *,
    current_path: Any = None,
) -> jax.Array:
    r"""Complex characteristic impedance of a port mode, from a path or two.

    The definition follows the paths given, with :math:`V = \int E \cdot dl`
    along ``voltage_path``, :math:`I = \int H \cdot dl` along ``current_path``
    and :math:`P = \tfrac12 \int (E \times H^*) \cdot n\, dA` the mode's
    complex power through the port:

    - ``voltage_path`` only: power-voltage,
      :math:`Z_{PV} = |V|^2 / (2 P^*)`.
    - ``current_path`` only: power-current,
      :math:`Z_{PI} = 2 P / |I|^2`.
    - both: voltage-current, :math:`Z_{VI} = V / I`.

    The three agree for a TEM mode and differ by the mode's departure from
    it. For a lossy line all three are complex; under :math:`e^{+i\omega t}`
    a line with conductor loss has :math:`\mathrm{Im}\, Z < 0`. For a wave
    port declared with a ``voltage_path`` the result already holds
    :math:`Z_{PV}` along it as ``Z_PV``; this measures any definition along
    any path, after the solve. Not differentiable; ``Z_PV`` on the result is.

    On a cross-section cut by a perfectly matched layer (`gfem.PML`),
    :math:`P` counts the section outside the layer only, so the
    impedances are low by the mode's share inside the layer, and a path
    must stay outside it.

    The current is the line integral of the in-plane magnetic field, so it
    is the current through a closed path that encircles a conductor, or
    through an open path whose ends lie on magnetic walls (natural edges of
    the cross-section, where :math:`n \times H = 0`). Close a loop by
    repeating its first point. The direction of either path does not matter
    for :math:`Z_{PI}`; for :math:`Z_{VI}` the current is taken in the
    direction that makes :math:`\mathrm{Re}\, Z_{VI} \ge 0`.

    Example:
        The power-current impedance of a coplanar waveguide's mode, around
        its centre conductor (a rectangle in the port plane ``z = 0``)::

            loop = [(-a, -b, 0), (a, -b, 0), (a, b, 0), (-a, b, 0), (-a, -b, 0)]
            z_pi = gfem.characteristic_impedance(mode, current_path=loop)

    Args:
        mode: A port mode result.
        voltage_path: Two or more points (m) on the port surface; the path
            runs through them in order, e.g. from a ground to the signal
            conductor.
        current_path: Two or more points (m) on the port surface, through
            the medium around a conductor.

    Returns:
        ``(n_freq,)`` complex, ohm.

    Raises:
        TypeError: ``mode`` is not a `gfem.PortModeResult`, or neither path
            is given.
        ValueError: A path is malformed, leaves the port plane or passes
            through a perfectly matched layer, or ``mode`` carries no
            field (a driven result computed inside `jax.jit`, `jax.grad` or
            `jax.vmap`).
        NonDifferentiableError: A path is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap` from their arguments; pass a fixed value.
            Or ``mode`` is traced (a port mode solved inside them).
    """
    from gfem._sweep.port_mode import path_current, path_voltage, planar
    from gfem.results import PortModeResult

    if not isinstance(mode, PortModeResult):
        raise TypeError(
            "characteristic_impedance needs a gfem.PortModeResult, got "
            f"{type(mode).__name__}."
        )
    field = mode.e_field if mode.e_field is not None else mode.h_field
    if field is not None and getattr(field, "_vector", None) is not None:
        raise NonDifferentiableError(
            "characteristic_impedance is not differentiable in this version. "
            "Z_PV along the port's own voltage_path is (PortModeResult.Z_PV, "
            "WavePort(voltage_path=...)). Compute it outside jax.grad, or from "
            "jax.lax.stop_gradient(mode)."
        )
    if field is not None and len(field.normal) == 2:
        # a 2-component normal (or an Hz field): the mode of a line port
        raise entry_refusal("characteristic_impedance")
    if voltage_path is None and current_path is None:
        raise TypeError(
            "characteristic_impedance needs a voltage_path, a current_path or "
            "both (Z_PV, Z_PI or Z_VI)."
        )
    paths = {
        what: _impedance_path(value, what)
        for what, value in (
            ("voltage_path", voltage_path),
            ("current_path", current_path),
        )
        if value is not None
    }
    field = mode.e_field
    if field is None:
        raise ValueError(
            f"The port mode of {mode.port!r} carries no field: the driven result "
            "was computed inside jax.jit, jax.grad or jax.vmap, where a wave "
            "port's mode field is not available. Port modes are constants of the "
            "solve: solve gfem.PortMode for the port outside the transformation."
        )
    space, basis, coords = planar(field._space), field._basis, field._coords  # noqa: SLF001
    frequency = np.asarray(mode.frequency, float)
    gamma = np.asarray(mode.gamma, complex)
    volts = amps = None
    if "voltage_path" in paths:
        volts = path_voltage(
            space,
            basis,
            coords,
            paths["voltage_path"],
            mode.port,
            subject="characteristic_impedance",
        )
    if "current_path" in paths:
        amps = path_current(
            space,
            basis,
            coords,
            gamma=gamma,
            frequency=frequency,
            path=paths["current_path"],
            name=mode.port,
            subject="characteristic_impedance",
        )
    if volts is not None and amps is not None:
        z = volts / amps
        return jnp.asarray(np.where(z.real < 0, -z, z))
    omega = 2 * np.pi * frequency
    scale = space.section.scale
    power = np.array(
        [
            space.power(basis @ c, g * scale, w)
            for c, g, w in zip(coords, gamma, omega, strict=True)
        ]
    )
    if volts is not None:
        return jnp.asarray(np.abs(volts) ** 2 / (2.0 * np.conj(power)))
    if amps is not None:
        return jnp.asarray(2.0 * power / np.abs(amps) ** 2)
    raise AssertionError  # unreachable: a path was given


def _impedance_path(value: Any, what: str) -> np.ndarray:
    """A fixed, finite ``(n >= 2, 3)`` path in metres, or raise."""
    check_fixed(value, f"{what} must be fixed numbers")
    path = np.asarray(value, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError(
            f"{what} must be two or more points (m), each a 3-vector, got "
            f"shape {path.shape}."
        )
    if not np.isfinite(path).all():
        raise ValueError(f"{what} must hold finite numbers.")
    return path


# --- power flow --------------------------------------------------------------------


def power_flow(
    field: Field,
    side: str | Sequence[str] | Callable[[np.ndarray], np.ndarray],
    *,
    model: Model,
) -> jax.Array:
    r"""Net time-averaged power leaving ``side`` through its boundary, W.

    ``side`` is a set of elements: a volume region, several (their union),
    or a predicate ``inside(centroids) -> bool`` on the ``(n_tetrahedra,
    3)`` element centroids in m, e.g. ``lambda c: c[:, 0] < x0`` for the
    power flowing in +x across ``x = x0``. Its surface is made of the
    element faces between the selected elements of the field's support and
    the unselected ones (a staircase where a plane cuts through elements);
    faces on the boundary of the solve domain are not on it, since what
    crosses them is counted as absorbed inside ``side``. Positive means net
    power leaves ``side``. A driven field is the peak phasor for 1 W
    incident at its excited port (`gfem.conventions`), so the number reads
    as a fraction of the incident power.

    The power comes from the residual of the discrete equations on the
    outside element layer, not from an integral of
    :math:`\frac{1}{2} \mathrm{Re}(E \times H^*)`, so
    it is exact for the solved field (to the solver's residual; with
    ``"bddc_gmres"`` or ``"pmg_gmres"`` that is the true residual, not the
    preconditioned one ``SolverOptions.tolerance`` bounds, and at low
    frequency it divides by a small :math:`\omega`, see
    `SolverOptions.tolerance`) and the
    same through every surface that encloses the same ports and losses.
    In lossy media it is the power entering ``side`` through its ports,
    minus what its elements and faces absorb (dielectric loss,
    conductivity, `Conductor` and `Impedance` sheets, its absorbing faces).
    A face term belongs to ``side`` only when every solve-domain element
    beside it does, so a sheet lying on the surface counts as outside.
    A side may cross or touch a `gfem.Periodic` pair.

    With a perfectly matched layer (`gfem.PML`), ``side`` set to the layer
    regions gives the power they absorb, with the stretch the field was
    solved with; a side whose surface cuts through the layer has no
    physical meaning, since the field there is the stretched one.

    A field of an adaptive sweep (`gfem.ReducedFields`) is exact at the
    solved frequencies and accurate to the sweep's tolerance in between.

    The total field of an incident-field column (``result.e_field(f,
    "wave")``) is not fed through a port: its residual holds the pairing
    :math:`\operatorname{Im}(u_S^H b)/(2\omega\mu_0)` of the incident load
    :math:`b` with the field on the side's DOFs, which is subtracted, so
    the power is again exact and, for a side of physical elements, minus
    the power absorbed inside it: ``-power_flow(total, "absorber",
    model=model)`` is one region's absorption under the wave. A side
    holding perfectly-matched-layer elements is refused (the field there is
    the scattered one), as is the incident part (``part="incident"``),
    which balances no load. The scattered part (``part="scattered"``) needs
    no correction.

    On a field of a distributed solve it is a collective call: every rank
    calls it with the same side and model, each pairs its own elements'
    terms with its piece of the field, and every rank gets the same number.

    Differentiable through the field: inside `jax.grad`, `jax.jit` or
    `jax.vmap` the kept field of a solve (lumped or wave ports) carries the
    gradient, so absorbed, radiated and transmitted power are objectives.
    The model's leaves are read only in the element layer just outside
    ``side``; a traced material field there has no gradient here and
    raises, and any other leaf gets an exact zero from this measurement
    (its influence arrives through the field). The node coordinates (a
    shape design, `gfem.shape.boundary_morph`) are differentiable too: the
    element layer moves with its nodes, and the field with them at fixed
    coefficients; the corrected power of an incident-field column is not
    (its solve refuses them first).

    Args:
        field: A driven electric field (quantity ``"E"``, in H(curl),
            complex, with a real ``omega``), e.g. ``result.e_field(f)``.
        side: The volume region name(s), or the centroid predicate.
        model: The model the field was solved with: the materials and
            boundaries the residual reads.

    Returns:
        A 0-d float64 array, W.

    Raises:
        TypeError: ``field`` is not a `gfem.Field`, ``model`` is not a
            `gfem.Model`, or ``side`` is neither names nor a callable.
        RegionNotFoundError: A named region does not exist.
        ValueError: A name is a surface region; the predicate does not
            return one boolean per tetrahedron; ``side`` selects no element
            of the support, or all of them; the field is not a driven
            electric field, or was not solved on ``model``; the
            surface is within one element of a port's surface; or, for an
            incident-field column, the field is its incident part, the side
            holds perfectly-matched-layer elements, or the side reaches a
            wave port's surface.
        NonDifferentiableError: While `jax.grad` traces it: a material
            field with an element in the element layer just outside
            ``side`` (or beside a face that layer's terms read), or the fill
            of a sheet with a face there, is differentiated (move the
            surface one element away from the design region, or wrap the
            value in `jax.lax.stop_gradient`); the field's ``omega`` is
            differentiated, or the node coordinates of an incident-field
            column or beside an anisotropic `gfem.Absorbing` face; the field
            is one of an
            adaptive sweep (`gfem.ReducedFields`) and is differentiated. And
            whenever the node coordinates are traced and ``side`` is a
            predicate: name its regions.
    """
    from gfem._field import Field
    from gfem.model import Model

    refuse_entry(getattr(field, "mesh", None), "power_flow")
    if not isinstance(field, Field):
        raise TypeError(f"power_flow needs a gfem.Field, got {type(field).__name__}.")
    if not isinstance(model, Model):
        raise TypeError(
            f"power_flow needs model= a gfem.Model, got {type(model).__name__}."
        )
    from gfem._power_flow import side_names

    names = side_names(side, "power_flow")
    _check_driven(field)
    recorder = _objective.recording()
    if recorder is not None:
        return _recorded_power_flow(recorder, field, names, model)
    incident = field._incident is not None  # noqa: SLF001
    # across ranks the kernel's collective host call checks the ranks agree
    ranked = field._communicator is not None  # noqa: SLF001
    if not (incident or ranked or is_traced(field) or is_traced(model)):
        engine = importlib.import_module("gfem._engine.power_flow")
        return jnp.asarray(engine.power_flow(field, names, model), dtype=float)
    if callable(names) and is_traced((field.mesh.nodes, model.mesh.nodes)):
        raise NonDifferentiableError(
            "power_flow's side is a predicate on element centroids, which needs "
            "the node coordinates as numbers, but they are computed inside "
            "jax.grad, jax.jit or jax.vmap here: name the side's volume regions "
            "instead, so that its surface is known before the transformation runs."
        )
    from gfem._power_flow import power_flow as traced_power_flow

    return traced_power_flow(field, names, model)


def _recorded_power_flow(
    recorder: _objective.Recorder,
    field: Field,
    names: tuple[str, ...] | Callable[[np.ndarray], np.ndarray],
    model: Model,
) -> jax.Array:
    """`power_flow` while an objective is classified: one recorded output.

    Its side is checked as the absorbed-power estimate needs it: named regions, no port
    surface in or on it, only volume losses, isotropic non-law materials.
    """
    label = recorder.label(field, "power_flow")
    if callable(names):
        raise MeshRefinementUnsupportedError(
            "power_flow side predicate",
            "The objective calls power_flow on a centroid predicate: its "
            "staircase surface changes as the mesh is refined, so the power "
            "it selects drifts with the mesh, an error the estimate does not "
            "describe. Name the side's volume regions.",
        )
    if recorder.discover:
        _check_estimated_side(names, model)
    concrete = recorder.field(label)
    return recorder.output(
        f"power_flow|{label}|{'+'.join(names)}",
        (),
        jnp.float64,
        lambda: power_flow(concrete, list(names), model=model),
        measurement="power_flow",
        label=label,
        side=names,
    )


def _check_estimated_side(names: tuple[str, ...], model: Model) -> None:
    """Raise for a `power_flow` side the absorbed-power estimate has no case for."""
    from gfem._engine import dispersion
    from gfem._engine.coefficients import anisotropic_regions
    from gfem.model import Absorbing, Conductor, Impedance, surfaces_of

    mesh = model.mesh
    tetrahedra = np.concatenate([mesh.elements(n) for n in names])
    nodes = np.unique(np.asarray(mesh.tetrahedra)[tetrahedra])

    def touches(surface: str) -> bool:
        triangles = np.asarray(mesh.triangles)[mesh.elements(surface)]
        return bool(np.isin(triangles, nodes).any())

    for port in model.ports:
        if any(touches(s) for s in surfaces_of(port)):
            raise MeshRefinementUnsupportedError(
                "power through a port",
                f"The objective's power_flow side touches port {port.name!r}: "
                "the power through a port is |S|^2, so write the objective on "
                "result.s.",
            )
    for boundary in model.boundaries:
        if isinstance(boundary, (Impedance, Conductor, Absorbing)) and any(
            touches(s) for s in surfaces_of(boundary)
        ):
            raise MeshRefinementUnsupportedError(
                "surface loss in power_flow",
                "The objective's power_flow side holds the lossy surface "
                f"{boundary.surface!r}, whose absorbed power has no estimate: "
                "use gfem.FieldEstimator().",
            )
    laws = {lr.region for lr in dispersion.law_regions(model)}
    odd = [n for n in names if n in laws] + list(anisotropic_regions(model, names))
    if odd:
        raise MeshRefinementUnsupportedError(
            "anisotropic or dispersive power_flow",
            "The objective's power_flow side holds the anisotropic or "
            f"dispersive region {odd[0]!r}, where the absorbed-power estimate "
            "was not verified: use gfem.FieldEstimator().",
        )


# --- optical force ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class OpticalForce:
    """The force and torque on a body (`gfem.optical_force`).

    Attributes:
        force: The force, N, ``(3,)``.
        torque: The torque about ``origin``, N m, ``(3,)``.
        origin: The point the torque is taken about, m, ``(3,)``.
    """

    force: jax.Array
    torque: jax.Array
    origin: jax.Array

    def __repr__(self) -> str:
        try:
            f, t, o = (np.asarray(x) for x in (self.force, self.torque, self.origin))
        except (TypeError, jax.errors.TracerArrayConversionError):
            return "OpticalForce(traced)"
        return f"OpticalForce(force={f} N, torque={t} N m, origin={o} m)"


register_fields(OpticalForce, ("force", "torque", "origin"))


def optical_force(
    field: Field,
    body: str | Sequence[str] | Callable[[np.ndarray], np.ndarray],
    *,
    model: Model,
    origin: Any = None,
) -> OpticalForce:
    r"""Time-averaged force and torque a solved field exerts on ``body``.

    ``body`` is a set of elements: a volume region, several (their union), or
    a predicate ``inside(centroids) -> bool`` on the ``(n_tetrahedra, 3)``
    element centroids in m, as `power_flow`'s side. A region held out of the
    electromagnetic solve (``Model(excluded=...)``), such as a meshed metal
    whose surface carries a `gfem.PEC`, is a body too.

    The force and torque are the derivative of the discrete time-averaged
    Lagrangian :math:`L = \frac{1}{4}\int (\varepsilon |E|^2 - |B|^2 /
    \mu)\,dV` of the solved field, at fixed coefficients, under a rigid
    motion of the body's nodes: the volume form of the Maxwell stress tensor
    on the stress shell, the solve-domain elements outside the body that share
    a vertex with it. Another shell would change the number by discretization
    error only, so there is no argument for it: refine the mesh to check
    convergence. The result converges with the mesh
    to the physical force; it is the exact derivative of the discrete
    Lagrangian, not the residual of the discrete equations (there is no
    discrete momentum balance).

    A driven field is the peak phasor for 1 W incident at its excited port
    (a plane wave at its ``amplitude``), so the force scales with the incident
    power. An eigenmode field is normalized to :math:`1\,\mathrm{J}` of
    stored energy, so its force equals :math:`-(1\,\mathrm{J}/\omega)\,
    d\omega/d\xi`, with :math:`\xi` the translation of the body (and the
    rotation, for the torque); it scales linearly with the stored energy. For
    a complex :math:`\omega` the real part is read: the identity is exact for
    a lossless, closed mode.

    The torque is taken about ``origin``, by default the body's volume
    centroid.

    The force and torque are differentiable in the field's coefficients and
    frequency, hence in the body's material and in any design of the solve
    that reaches the field (an eigenmode's frequency moves with the design).
    The node coordinates and the medium of the shell are read without a
    gradient: `jax.lax.stop_gradient` is the way out.

    The shell must be one homogeneous, isotropic, lossless medium
    (a `gfem.Material` with a real ``eps_r`` and ``mu_r``). The body itself
    may be anything, lossy, anisotropic, dispersive or a `gfem.MaterialField`.
    It may touch a `gfem.PEC` or `gfem.PMC` wall, or the domain's natural
    wall, it does not own: the motions that would slide it along the wall
    are then not defined, and the force and torque components that move it
    along that wall are NaN, the others are the force on the body held on the
    wall. Rotations are about axes through the centroid.

    The total field of an incident-field column (``result.e_field(f,
    "wave")``) is read; its incident and scattered parts are refused.

    Args:
        field: A 3D electric field (quantity ``"E"``, in H(curl), complex),
            of a `gfem.Driven` or `gfem.Eigenmode` solve.
        body: The volume region name(s), or the centroid predicate.
        model: The model the field was solved with.
        origin: The point the torque is taken about, m, ``(3,)``; ``None``
            for the body's volume centroid. It may be a traced value.

    Returns:
        An `OpticalForce`.

    Raises:
        TypeError: ``field`` is not a `gfem.Field`, ``model`` is not a
            `gfem.Model`, ``body`` is neither names nor a callable, or
            ``origin`` is not a real value of shape ``(3,)``.
        RegionNotFoundError: A named region does not exist.
        ValueError: A name is a surface region; the predicate does not return
            one boolean per tetrahedron; the body holds no element or has no
            stress shell; the field is not a 3D electric time-harmonic field
            or was not solved on ``model``; it is the incident or the
            scattered part of an incident-field column; the shell is not
            one homogeneous isotropic lossless medium; a sheet touches the
            body without lying on it; a port, an absorbing surface or a
            perfectly matched layer touches the body or its shell; a body
            node is on a periodic pair; or a point dipole whose load solved
            the field is in the shell.
        NotImplementedError: A 2D model.
        gfem.DistributedUnsupportedError: A field across ranks.
        gfem.MeshRefinementUnsupportedError: Under the refinement recorder.
        NonDifferentiableError: A differentiated input is the node
            coordinates, the medium of a region in the shell, or the
            coefficients of an adaptive sweep's field.
    """
    from gfem._field import Field
    from gfem._optical_force import evaluate
    from gfem.model import Model

    refuse_entry(getattr(field, "mesh", None), "optical_force")
    curved.refuse_entry(
        getattr(field, "mesh", None), "optical_force", "gfem.optical_force"
    )
    if not isinstance(field, Field):
        raise TypeError(
            f"optical_force needs a gfem.Field, got {type(field).__name__}."
        )
    if not isinstance(model, Model):
        raise TypeError(
            f"optical_force needs model= a gfem.Model, got {type(model).__name__}."
        )
    if field._communicator is not None:  # noqa: SLF001
        _mpi.refuse_feature("optical_force")
    names = _body_names(body)
    around = _origin(origin)
    if callable(names) and is_traced((field.mesh.nodes, model.mesh.nodes)):
        raise NonDifferentiableError(
            "optical_force's body is a predicate on element centroids, which "
            "needs the node coordinates as numbers, but they are computed "
            "inside jax.grad, jax.jit or jax.vmap here: name the body's volume "
            "regions instead, so that its elements are known before the "
            "transformation runs."
        )
    if _objective.recording() is not None:
        raise MeshRefinementUnsupportedError(
            "optical_force",
            "The objective reads gfem.optical_force, whose error is not "
            "estimated in this version: refine on the S-parameters or on "
            "gfem.power_flow instead, or use gfem.FieldEstimator().",
        )
    _check_time_harmonic(field, "optical_force", static="solve a gfem.Driven problem")
    if not field.is_complex:
        raise ValueError(
            "optical_force needs a complex phasor field, got real coefficients: "
            "solve a gfem.Driven or gfem.Eigenmode problem."
        )
    force, torque, centroid = evaluate(field, names, model)
    if around is None:
        return OpticalForce(force=force, torque=torque, origin=centroid)
    return OpticalForce(
        force=force,
        torque=torque + _shift(centroid - around, force),
        origin=around,
    )


def _shift(arm: jax.Array, force: jax.Array) -> jax.Array:
    """``arm x force``, NaN only where a NaN force component has a nonzero arm.

    A component of the force that slides on a wall is NaN: it must not reach
    a torque component it does not multiply (``0 * NaN``).
    """
    nan = jnp.isnan(force)
    shift = jnp.cross(arm, jnp.where(nan, 0.0, force))
    # component i is arm_{i+1} F_{i+2} - arm_{i+2} F_{i+1}
    nonzero = arm != 0
    reach = (jnp.roll(nonzero, -1) & jnp.roll(nan, -2)) | (
        jnp.roll(nonzero, -2) & jnp.roll(nan, -1)
    )
    return jnp.where(reach, jnp.nan, shift)


def _body_names(
    body: Any,
) -> tuple[str, ...] | Callable[[np.ndarray], np.ndarray]:
    """``body`` as a tuple of region names or a predicate, or raise."""
    if isinstance(body, str):
        return (body,)
    if callable(body):
        return body
    if isinstance(body, Sequence) and body and all(isinstance(b, str) for b in body):
        return tuple(body)
    raise TypeError(
        "optical_force's body must be a volume region name, a sequence of names "
        f"or a predicate on element centroids, got {body!r}."
    )


def _origin(origin: Any) -> jax.Array | None:
    """The torque origin as a float ``(3,)`` array, or raise."""
    if origin is None:
        return None
    try:
        out = jnp.asarray(origin)
    except (TypeError, ValueError):
        out = None
    if out is None or out.shape != (3,) or jnp.iscomplexobj(out) or out.dtype == bool:
        raise TypeError(
            "optical_force's origin must be None or three real numbers (m), got "
            f"{origin!r}."
        )
    return out.astype(float)


# --- dissipated power -------------------------------------------------------------


def dissipated_power(field: Field, *, model: Model) -> dict[str, jax.Array]:
    r"""Time-averaged power a driven field dissipates in each region and sheet, W.

    The electric loss of each volume region,
    :math:`\frac{1}{2} \omega \varepsilon_0 \int_r E^* \cdot
    (-\mathrm{Im}\, \varepsilon_\mathrm{eff}) E\, dV` with the loss tangent,
    the conductivity, a complex or anisotropic permittivity and a
    `gfem.PermittivityLaw` at the field's frequency in
    :math:`\varepsilon_\mathrm{eff}`, and the loss of each surface carrying
    a `gfem.Conductor` or `gfem.Impedance`,
    :math:`\frac{1}{2} \int_S \mathrm{Re}(Y_s) |E_t|^2 dS` times the sheet's
    fill. The power is read from the driven operator's lossy pieces, so it
    is exact for the solved field: a region's value is minus
    `power_flow` out of it when no sheet lies on its boundary, and the
    values sum to :math:`1 - \sum_j |S_{ji}|^2` on a model whose ports have
    real reference impedances and that radiates nothing.

    A driven field is the one for 1 W incident at its excited port
    (`gfem.conventions`), so each value reads as the fraction of the
    incident power dissipated there. With a complex port reference
    impedance the field is per unit :math:`|a|^2`, which is not exactly 1 W
    incident. On a slab cross-section the fraction is the same per length.

    What is not heat gets no value: a perfectly matched layer (its regions
    have no key), an `gfem.Absorbing` face, a wave or Floquet port, and a
    lumped port's resistance, whose power is the network's load.

    Differentiable: inside `jax.grad`, `jax.jit` or `jax.vmap` the field
    carries the gradient through its solve, and the loss tangent,
    conductivity and permittivity (real or complex, per axis) of a lossy
    region, a permittivity law's parameters, a `gfem.SheetField` fill and an
    `gfem.Impedance` resistance add their direct share. An adaptive sweep's
    field, the field's frequency and the node coordinates raise.

    Args:
        field: A driven electric field (quantity ``"E"``, in H(curl),
            complex, with a real ``omega``), e.g. ``result.e_field(f)``.
        model: The model the field was solved with: the materials and
            boundaries whose loss is read.

    Returns:
        A 0-d float64 array, W, per volume region of the solve domain
        outside a perfectly matched layer (lossless regions at 0.0) and per
        surface carrying a `gfem.Conductor` or `gfem.Impedance`.

    Raises:
        TypeError: ``field`` is not a `gfem.Field`, or ``model`` is not a
            `gfem.Model`.
        ValueError: The field is not a driven electric field (another
            quantity, static, an eigenmode's complex omega, real
            coefficients), or was not solved on ``model``; or a name is
            both a volume region and a sheet surface.
        NotImplementedError: A region has a complex ``mu_r`` (magnetic loss
            is not heat in this version), or the model is
            2D (pose the driven solve on the slab of
            `gfem.Mesh.from_gmsh_cross_section`).
        gfem.MeshRefinementUnsupportedError: Called while an objective is
            classified for `gfem.refine_mesh`.
        gfem.DistributedUnsupportedError: The field is a piece across ranks.
        NonDifferentiableError: While `jax.grad` traces it: the node
            coordinates or the field's ``omega`` are differentiated, or the
            field is one of an adaptive sweep (`gfem.ReducedFields`).
    """
    from gfem._engine.heat import sheet_surfaces
    from gfem._heat import heated_regions, loss_matrices

    check_heat_field(field, model, "dissipated_power")

    sheets = sheet_surfaces(model)
    mesh = model.mesh
    matrices = loss_matrices([field], model)
    tets = matrices["tetrahedra_re"][:, 0, 0]
    faces = matrices["triangles_re"][:, 0, 0]
    out = {
        region: jnp.sum(tets[elements])
        for region, elements in heated_regions(field, model).items()
    }
    for name in sheets:
        out[name] = jnp.sum(faces[mesh.elements(name)])
    return out


def check_heat_field(field: Any, model: Any, who: str) -> None:
    """Raise unless ``field`` is a driven field whose loss ``model`` can turn into heat.

    The refusals of `dissipated_power`, shared by every caller of the loss
    matrices (`gfem._heat.loss_matrices`). ``who`` is the public entry's name:
    its 2D refusal (`gfem._dim2`), its key among the distributed refusals and
    the refinement recorder's feature. Complex ``mu_r`` is refused by the
    kernel, which reads the element values.

    Raises:
        TypeError: ``field`` is not a `gfem.Field`, or ``model`` is not a
            `gfem.Model`.
        ValueError: Not a driven electric field, or a name that is both a
            volume region and a sheet surface.
        NotImplementedError: A 2D model.
        gfem.MeshRefinementUnsupportedError: Under the refinement recorder.
        gfem.DistributedUnsupportedError: A field across ranks.
    """
    from gfem._engine.heat import sheet_surfaces
    from gfem._field import Field
    from gfem.model import Model

    refuse_entry(getattr(field, "mesh", None), who)
    if not isinstance(field, Field):
        raise TypeError(f"{who} needs a gfem.Field, got {type(field).__name__}.")
    if field._communicator is not None:  # noqa: SLF001
        _mpi.refuse_feature(who)
    if not isinstance(model, Model):
        raise TypeError(f"{who} needs model= a gfem.Model, got {type(model).__name__}.")
    if _objective.recording() is not None:
        raise MeshRefinementUnsupportedError(
            who,
            f"The objective reads gfem.{who}, whose error is not estimated in "
            "this version: refine on the S-parameters or on gfem.power_flow "
            "instead.",
        )
    _check_time_harmonic(field, who, static="solve a gfem.Driven problem")
    if jnp.iscomplexobj(field.omega):
        raise ValueError(
            f"{who} needs a driven field with a real omega: an eigenmode has no "
            "drive, so its amplitude and its heat are undefined; solve a driven "
            "problem at the mode's frequency."
        )
    if not field.is_complex:
        raise ValueError(
            f"{who} needs a complex phasor field, got real coefficients: solve a "
            "gfem.Driven problem."
        )
    for name in sheet_surfaces(model):
        if name in model.mesh.volumes:
            raise ValueError(
                f"{name!r} names both a volume region and a sheet surface, so "
                f"{who} would give two values under one key: rename one of "
                "them in the mesh."
            )


# --- heat flow -------------------------------------------------------------------

#: What a heat-flow side may be: names, or a centroid predicate.
HeatSide = str | Sequence[str] | Callable[[np.ndarray], np.ndarray]


def heat_flow(
    result: SteadyThermalResult, side: HeatSide, *, model: Model
) -> jax.Array:
    r"""Net heat leaving ``side``, W: positive out.

    ``side`` takes three forms:

    - a volume region, several (their union), or a predicate
      ``inside(centroids) -> bool`` on the ``(n_tetrahedra, 3)`` element
      centroids in m (a staircase cut, e.g. ``lambda c: c[:, 1] > y0``).
      Only elements of the thermal domain count. The flow is the heat
      leaving the selected elements through the faces between them and the
      rest of the thermal domain; what crosses the thermal domain's own
      boundary there (convection, a heat flux, a fixed temperature) is
      counted inside ``side``. A cut may end on a fixed-temperature
      surface: the reaction of its nodes is split between the two sides by
      their elements, so a side's flow is minus its complement's. A face
      belongs to ``side`` when every thermal element beside it does, so
      the heat of a sheet or a `gfem.HeatFlux` lying exactly on the cut
      belongs to neither side.
    - a surface name: the heat leaving the thermal domain through it. A
      `gfem.FixedTemperature` surface gives the heat it removes (an
      interior one removes it from both sides), a `gfem.Convection`
      surface :math:`\int h (T - T_\infty)\, dS`, a `gfem.HeatFlux` surface
      minus the power it puts in, and an exterior surface with no thermal
      condition 0 (insulated).
    - several surface names: their union. The flows out of every boundary
      surface sum to ``result.power``, sheet heat included: heat on a face
      is a source, and on a fixed-temperature face its reaction takes it.

    The heat comes from the residual of the discrete heat equation, tested
    with the vertex hat functions, not from :math:`-k\, \partial T /
    \partial n`: it is exact for the solved temperature (to the solver's
    residual), conservative across any set of surfaces, and needs no
    gradient sampled at a sink, where the gradient is least accurate. It
    measures the absolute temperature, so a background flow set by unequal
    boundary temperatures is part of it.

    Differentiable: inside `jax.grad`, `jax.jit` or `jax.vmap` the flow
    differentiates with respect to the thermal conductivity (read on the
    side's elements, and through the temperature everywhere), the heat
    sources (their amounts, and through `gfem.joule_heating` the drive),
    and the temperature. A side whose closure holds no fixed-temperature
    node and no convection face measures exactly its source power, so its
    conductivity gradient is exactly zero.

    Args:
        result: A steady thermal result, e.g. ``gfem.solve(model,
            gfem.SteadyThermal(heat))``, on a 3D mesh.
        side: The volume region name(s) or predicate, or the surface
            name(s).
        model: The model the result was solved with: the conductivity and
            the boundaries the residual reads.

    Returns:
        A 0-d float64 array, W.

    Raises:
        TypeError: ``result`` is not a `gfem.SteadyThermalResult` (a
            transient result records its heat with a heat-out trace,
            `gfem.Trace.heat_out`), ``model`` is not a `gfem.Model`, or
            ``side`` is neither names nor a callable.
        RegionNotFoundError: A named region does not exist.
        ValueError: The result was solved on a cross-section mesh (use
            `heat_flow_per_length`), or not on ``model``; ``side`` mixes
            volume regions and surfaces, selects no element of the thermal
            domain or all of them, names an interior surface with no
            thermal condition or a surface away from the thermal domain,
            or names one of two fixed-temperature surfaces that share mesh
            nodes (measure their union); the predicate does not return one
            boolean per tetrahedron.
        gfem.DistributedUnsupportedError: The result's fields are pieces
            across ranks.
        MeshRefinementUnsupportedError: Called inside a refined objective.
        NonDifferentiableError: While `jax.grad` traces it, the node
            coordinates are differentiated.
    """
    names = _heat_side(result, side, model, "heat_flow")
    if result.temperature.mesh.section_thickness is not None:
        raise ValueError(
            "gfem.heat_flow measures a 3D result in W, but this result was solved "
            "on a cross-section mesh, whose slab thickness is a numerical choice: "
            "use gfem.heat_flow_per_length (W/m)."
        )
    from gfem._heat_flow import heat_flow as measured

    return measured(result, names, model, who="heat_flow")


def heat_flow_per_length(
    result: SteadyThermalResult, side: HeatSide, *, model: Model
) -> jax.Array:
    """Net heat leaving ``side`` per unit length of a cross-section, W/m.

    `heat_flow` on a result solved on a cross-section mesh
    (`gfem.Mesh.from_gmsh_cross_section`): the same sides and the same
    residual, divided by the slab thickness. The end faces ``front`` and
    ``back`` are insulated unless the model gives them a condition.

    Args:
        result: A steady thermal result on a cross-section mesh.
        side: As for `heat_flow`.
        model: The model the result was solved with.

    Returns:
        A 0-d float64 array, W/m.

    Raises:
        TypeError: As for `heat_flow`.
        RegionNotFoundError: A named region does not exist.
        ValueError: The result was solved on a 3D mesh (use `heat_flow`),
            and as for `heat_flow`.
        gfem.DistributedUnsupportedError: As for `heat_flow`.
        MeshRefinementUnsupportedError: Called inside a refined objective.
        NonDifferentiableError: As for `heat_flow`.
    """
    names = _heat_side(result, side, model, "heat_flow_per_length")
    thickness = result.temperature.mesh.section_thickness
    if thickness is None:
        raise ValueError(
            "gfem.heat_flow_per_length measures a cross-section result in W/m, but "
            "this result was solved on a 3D mesh: use gfem.heat_flow (W)."
        )
    from gfem._heat_flow import heat_flow as measured

    return measured(result, names, model, who="heat_flow_per_length") / thickness


def _heat_side(
    result: Any, side: Any, model: Any, who: str
) -> tuple[str, ...] | Callable[[np.ndarray], np.ndarray]:
    """``side`` as names or a predicate, after the checks every heat flow runs.

    Raises:
        TypeError: The result, the model or the side has the wrong type.
        ValueError: The result was not solved on ``model``.
        gfem.DistributedUnsupportedError: Fields across ranks.
        MeshRefinementUnsupportedError: Inside a refined objective.
        NonDifferentiableError: A predicate side on traced node coordinates.
    """
    from gfem.model import Model
    from gfem.results import SteadyThermalResult, TransientThermalResult

    if isinstance(result, TransientThermalResult):
        raise TypeError(
            f"gfem.{who} measures a steady result; a transient result records the "
            "heat leaving a side at every time step with a heat-out trace: pass "
            "traces={'out': gfem.Trace.heat_out(side)} to gfem.TransientThermal."
        )
    if not isinstance(result, SteadyThermalResult):
        raise TypeError(
            f"gfem.{who} needs a gfem.SteadyThermalResult, got {type(result).__name__}."
        )
    if result.temperature._communicator is not None:  # noqa: SLF001
        _mpi.refuse_feature("thermal")  # no distributed thermal path
    if not isinstance(model, Model):
        raise TypeError(f"{who} needs model= a gfem.Model, got {type(model).__name__}.")
    if _objective.recording() is not None:
        raise MeshRefinementUnsupportedError(
            who,
            f"The objective reads gfem.{who}, and thermal problems have no error "
            "estimate: refine the mesh by hand and compare.",
        )
    if isinstance(side, str):
        names: Any = (side,)
    elif callable(side):
        names = side
    elif isinstance(side, Sequence) and side and all(isinstance(s, str) for s in side):
        names = tuple(dict.fromkeys(side))
    else:
        raise TypeError(
            f"{who}'s side must be a region or surface name, a sequence of names or "
            f"a predicate on element centroids, got {side!r}."
        )
    _check_thermal_field(result.temperature, model, who)
    if callable(names) and is_traced(model.mesh.nodes):
        raise NonDifferentiableError(
            f"{who}'s side is a predicate on element centroids, which needs the "
            "node coordinates as numbers, but they are computed inside jax.grad, "
            "jax.jit or jax.vmap here: name the side's volume regions instead."
        )
    return names


def _check_thermal_field(field: Field, model: Model, who: str) -> None:
    """Raise unless the temperature was solved on ``model``'s thermal domain.

    Raises:
        ValueError: Another mesh, node positions, thermal domain or order.
    """
    from gfem.mesh import _same_topology

    other = ValueError(
        f"{who}: the result was not solved on this model (another mesh, node "
        "positions or thermal domain): pass the model the result was solved with."
    )
    if not _same_topology(field.mesh, model.mesh):
        raise other
    if not (is_traced(field.mesh.nodes) or is_traced(model.mesh.nodes)) and not (
        np.array_equal(np.asarray(field.mesh.nodes), np.asarray(model.mesh.nodes))
    ):
        raise other
    domain = model._domain_of("thermal")  # noqa: SLF001
    mesh = model.mesh
    support = tuple(r for r in mesh.volumes if domain.active[mesh.elements(r)].any())
    if (
        field.quantity != "T"
        or tuple(field.support) != support
        or tuple(field._excluded) != tuple(domain.excluded)  # noqa: SLF001
    ):
        raise other


# --- cross sections --------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class CrossSections:
    r"""The extinction, absorption and scattering cross sections of one source.

    Each is a power over the source's intensity, per kept frequency;
    ``extinction`` is ``absorption + scattering`` exactly.

    Attributes:
        frequency: The kept frequencies of the source's column, Hz,
            ``(n_freq,)``.
        extinction: The power removed from the incident field, m²,
            ``(n_freq,)``.
        absorption: The power the scatterer absorbs, m², ``(n_freq,)``.
        scattering: The power the scattered field carries out, m²,
            ``(n_freq,)``.
        intensity: The intensity :math:`I` the powers are divided by, W/m²,
            ``(n_freq,)``.
    """

    frequency: np.ndarray
    extinction: jax.Array
    absorption: jax.Array
    scattering: jax.Array
    intensity: np.ndarray

    def __repr__(self) -> str:
        try:
            values = [
                np.asarray(x)
                for x in (self.extinction, self.absorption, self.scattering)
            ]
        except (TypeError, jax.errors.TracerArrayConversionError):
            return "CrossSections(traced)"
        rows = ", ".join(
            f"{f:.6g} Hz: ext {e:.4g}, abs {a:.4g}, sca {c:.4g} m²"
            for f, e, a, c in zip(self.frequency, *values, strict=True)
        )
        return f"CrossSections({rows})"


register_fields(CrossSections, ("extinction", "absorption", "scattering"))


def cross_sections(
    result: DrivenResult,
    source: str,
    *,
    model: Model,
    intensity: Any = None,
) -> CrossSections:
    r"""The extinction, absorption and scattering cross sections of an incident field.

    Read from one exact discrete power balance on the truncation layer (the
    physical elements touching the PML layer or an `gfem.Absorbing` face),
    with :math:`u` the solved total field, :math:`b` its load,
    :math:`e_s` the scattered field and :math:`c = 2\omega\mu_0`:

    .. math::

        P_\mathrm{sca} = \operatorname{flux}(e_s), \quad
        P_\mathrm{abs} = \operatorname{Im}(u_S^H b)/c - \operatorname{flux}(u),
        \quad P_\mathrm{ext} = P_\mathrm{abs} + P_\mathrm{sca},

    where :math:`\operatorname{flux}` is `power_flow`'s residual form out
    of the elements :math:`S` inside the layer and :math:`u_S` the field on
    their DOFs. Every set of elements holding the scatterer gives the same
    numbers, so there is no surface to choose. Each cross section is the
    power over the source's intensity :math:`I`:

    - `gfem.PlaneWave`: :math:`|E_0|^2 / (2\eta)`, :math:`\eta` the wave
      impedance of the layer it comes from;
    - `gfem.GaussianBeam`: its focal intensity
      :math:`I_0 = |E(r_f)|^2 / (2\eta)` at each frequency
      (`gfem.GaussianBeam.focal_intensity`);
    - `gfem.IncidentField`: ``intensity=``, required.

    On a layered background (`gfem.Background`) the extinction is the
    power the scatterer removes from the stack's own field, the incident
    wave with its reflected and transmitted waves, not from the incident
    wave alone; the scattered power is what the scattered field carries to
    the truncation, into every layer. Every layer must be lossless.

    A model reduced by :math:`n` mirror planes (walls the incident field
    satisfies) reports the meshed part, :math:`2^{-n}` of the full model's
    cross sections: gfem cannot tell a symmetry plane from a physical wall.

    On a result of a distributed solve it is a collective call: every rank
    calls it and gets the same cross sections.

    Differentiable through the field: inside `jax.grad` the cross sections
    of a material design in the scatterer (a `gfem.MaterialField`) carry
    its gradient; the measurement reads no scatterer material, so no
    material pass is needed. The source, its intensity and the frequency are
    static.

    Args:
        result: A driven result with kept fields, solved with ``source``.
        source: The name of an incident-field source of the result.
        model: The model the result was solved with.
        intensity: W/m², a positive number or one per kept frequency: the
            intensity of a `gfem.IncidentField`; ``None`` for any other
            source.

    Returns:
        A `CrossSections`.

    Raises:
        KeyError: ``source`` is not a column of the result.
        ValueError: ``source`` is a port or a point dipole; the result keeps
            no field of it; ``intensity`` is missing for a
            `gfem.IncidentField`, given for another source, or not positive
            and finite; a scatterer region, sheet, lumped port or perfect
            conductor reaches the truncation layer; the truncation layer
            reaches a wave port; the model has no background.
        NotImplementedError: A lossy background, or a lossy layer of a
            layered one.
        NonDifferentiableError: While `jax.grad` traces it: a material design
            with an element in the truncation layer, or the node coordinates.
    """
    from gfem.model import Model
    from gfem.results import DrivenResult

    if not isinstance(result, DrivenResult):
        raise TypeError(
            f"cross_sections needs a gfem.DrivenResult, got {type(result).__name__}."
        )
    if not isinstance(model, Model):
        raise TypeError(
            f"cross_sections needs model= a gfem.Model, got {type(model).__name__}."
        )
    kind = _column_kind(result, source)
    if kind is not None:
        raise ValueError(
            f"cross_sections: {source!r} is a {kind} column; cross sections are "
            "measured for an incident field (gfem.PlaneWave, gfem.GaussianBeam, "
            "gfem.IncidentField)."
        )
    wave = result.sources[source]
    frequencies = [f for f, label in result.kept if label == source]
    if not frequencies:
        raise ValueError(
            f"cross_sections needs the kept fields of {source!r}: solve with "
            "Driven(..., keep_fields=True) or keep_fields=[frequencies]."
        )
    background = model.background
    if background is None:
        raise ValueError(
            "cross_sections needs the model's background (Model(background=...)): "
            "pass the model the result was solved with."
        )
    xs = importlib.import_module("gfem._engine.cross_sections")
    xs.check_lossless(background)
    i0 = _intensity(wave, intensity, frequencies, background)
    absorbed, scattered = [], []
    for f in frequencies:
        field = result.e_field(f, source)
        if field._incident is None:  # noqa: SLF001
            raise ValueError(
                f"The kept field of {source!r} carries no incident part: solve "
                "again to measure its cross sections."
            )
        pf = importlib.import_module("gfem._engine.power_flow")
        where = pf.surface_of(field, truncation(model).side, model)
        curved.refuse_tets(
            model.mesh,
            "cross_sections",
            where.layer,
            "The truncation layer of gfem.cross_sections",
        )
        xs.check_truncation_layer(model, background, where)
        xs.check_side(model, where, "cross_sections")
        from gfem._power_flow import balance

        out = balance(field, where, model, mode="cross_sections")
        absorbed.append(out["absorption"])
        scattered.append(out["scattering"])
    absorption = jnp.stack(absorbed) / i0
    scattering = jnp.stack(scattered) / i0
    return CrossSections(
        frequency=np.asarray(frequencies, float),
        extinction=absorption + scattering,
        absorption=absorption,
        scattering=scattering,
        intensity=i0,
    )


def _column_kind(result: DrivenResult, name: str) -> str | None:
    """``None`` for an incident-field source, else what the column is.

    Raises:
        KeyError: ``name`` is no column of the result.
    """
    from gfem._sources import INCIDENT

    if name in result.sources:
        source = result.sources[name]
        return None if isinstance(source, INCIDENT) else type(source).__name__
    if name in result.s.excited:
        return "port"
    columns = (*result.s.excited, *result.sources)
    raise KeyError(
        f"{name!r} is not a column of the result; its ports and sources are "
        f"{', '.join(map(repr, columns))}."
    )


def _intensity(
    source: Any,
    intensity: Any,
    frequencies: Sequence[float],
    background: Any,
    entry: str = "cross_sections",
) -> np.ndarray:
    """The intensity of ``source`` per kept frequency, W/m² (see `cross_sections`).

    Raises:
        ValueError: Missing for an `gfem.IncidentField`, given for another
            source, or not positive and finite.
    """
    from gfem import _incident

    n = len(frequencies)
    own = _incident.intensity(source, background, frequencies)
    if own is None:
        if intensity is None:
            raise ValueError(
                f"{entry} of the {type(source).__name__} {source.name!r} needs "
                "intensity= (W/m², a number or one per kept frequency): gfem "
                "cannot tell the intensity of a field it only evaluates."
            )
        if is_traced(intensity):
            raise NonDifferentiableError(
                f"{entry}' intensity is static: wrap it in "
                "jax.lax.stop_gradient or pass a number."
            )
        values = (
            np.broadcast_to(np.asarray(intensity, float), (n,)).copy()
            if (np.ndim(intensity) == 0 or np.shape(intensity) == (n,))
            else None
        )
        if values is None or not (np.isfinite(values).all() and (values > 0).all()):
            raise ValueError(
                f"{entry}' intensity must be a positive finite number or "
                f"{n} of them (one per kept frequency), W/m², got {intensity!r}."
            )
        return values
    if intensity is not None:
        raise ValueError(
            f"{entry} takes intensity= for an IncidentField only: the "
            f"intensity of the {type(source).__name__} {source.name!r} is its "
            "own (see gfem.cross_sections)."
        )
    return own


# --- region mean -----------------------------------------------------------------


def region_mean(field: Field, region: str | Sequence[str]) -> jax.Array:
    r"""Mean of a field over volume regions, or of a potential.

    The mean is :math:`\int_R E\, dV / |R|`.
    ``R`` is the union of the named regions' elements in the field's
    support. The mean is exact for the solved field (the integral of its
    finite-element polynomial, not a sample of it), and linear in its
    coefficients: under `jax.grad` the gradient flows back into the solve
    that made the field, and it works under `jax.jit` and `jax.vmap`. A
    weighted component ``w . mean`` is ``jnp.dot(w, region_mean(E, "R"))``.

    Regions are named, never selected by a predicate: mesh refinement
    (`gfem.refine_mesh`) keeps region tags, so a named mean is the same
    quantity on every refined mesh. Inside a perfectly matched layer
    (`gfem.PML`) the field is the stretched one, not a physical field, and
    so is its mean. It is the measurement to use in place
    of a point value (`gfem.Field.at`) in an objective: a mean over a
    small region has an error estimate (`gfem.QuantityEstimator`), a
    point value has none.

    Args:
        field: A field of a result, e.g. ``result.e_field(f)`` or
            ``result.potential["t1"]``.
        region: A volume region name, or a sequence of names (their union).

    Returns:
        ``(3,)`` for a vector field (an electric field in V/m, complex for
        a phasor), ``()`` for a scalar field (a potential in V).

    Raises:
        TypeError: ``field`` is not a `gfem.Field`, or ``region`` is
            neither a name nor a non-empty sequence of names.
        RegionNotFoundError: A named region does not exist.
        ValueError: A name is a surface region, or the regions hold no
            element of the field's support.
        gfem.DistributedUnsupportedError: The field is a piece across ranks.
        NonDifferentiableError: The mesh node coordinates are differentiated.
    """
    from gfem._field import Field

    refuse_entry(getattr(field, "mesh", None), "region_mean")
    if not isinstance(field, Field):
        raise TypeError(f"region_mean needs a gfem.Field, got {type(field).__name__}.")
    if field._communicator is not None:  # noqa: SLF001
        _mpi.refuse_feature("region_mean")
    names = _region_names(region)
    _check_mean_regions(field, names)
    recorder = _objective.recording()
    if recorder is not None:
        return _recorded_region_mean(recorder, field, names)
    kernel = _MeanKernel(field, names)
    return measure(
        kernel, {"coefficients": field.coefficients, "nodes": field.mesh.nodes}
    )["mean"]


def strain(displacement: Any, points: Any) -> jax.Array:
    r"""The engineering strain of a displacement at points.

    ``displacement`` is the kept displacement of a `gfem.Piezoelectric`
    solve (``res.displacement(f, terminal)``), per volt on the terminal, or the
    displacement of a mode of a `gfem.MechanicalEigenmode` solve
    (``res.displacement(mode)``). With :math:`u = (u_x, u_y, u_z)` the strain in
    Voigt order (xx, yy, zz, yz, xz, xy), the shear entries engineering ones
    (twice the tensor components), of a 3D body is

    .. math::
        S = \left(\partial_x u_x,\ \partial_y u_y,\ \partial_z u_z,\
        \partial_z u_y + \partial_y u_z,\ \partial_z u_x + \partial_x u_z,\
        \partial_y u_x + \partial_x u_y\right).

    A 2D section has no variation along :math:`z`, or the field of a mode has
    :math:`e^{-iqz}` with the map's ``wavenumber`` :math:`q` (0 for a
    piezoelectric solve), so the strain there is

    .. math::
        S = \left(\partial_x u_x,\ \partial_y u_y,\ -iq\,u_z,\
        \partial_y u_z - iq\,u_y,\ \partial_x u_z - iq\,u_x,\
        \partial_y u_x + \partial_x u_y\right).

    It is read in the lab frame, in the mechanical domain. The gradient of
    each component is exact for the solved field, and linear in its
    coefficients: under `jax.grad` it flows back into the solve that made the
    displacement, and it works under `jax.jit` and `jax.vmap`. The points are
    fixed. The mesh node coordinates are differentiable: the point stays where
    it is while the elements move under it (a node map of `gfem.Mesh.with_nodes`
    reaches the strain at a fixed point).

    Args:
        displacement: The displacement map ``res.displacement(f)``, with the
            keys ``"x"``, ``"y"`` and ``"z"``.
        points: One point or ``(n, 2)`` points, in metres, in the
            :math:`(x, y)` plane of a 2D section; ``(3,)`` or ``(n, 3)`` on a
            3D body.

    Returns:
        ``(n, 6)`` complex strain per volt, ``(6,)`` for one point.

    Raises:
        TypeError: ``displacement`` is not a displacement map; pass
            ``res.displacement(f)``.
        ValueError: A point is outside the mechanical domain.
    """
    from gfem._field import FieldMap

    keys = ("x", "y", "z")
    if not isinstance(displacement, FieldMap) or tuple(displacement) != keys:
        raise TypeError(
            "strain reads the displacement of a piezoelectric or mechanical-mode "
            f"solve: pass res.displacement(...), got {type(displacement).__name__}."
        )
    single = np.ndim(points) == 1
    grads = {k: _probe.gradient_at(displacement[k], points) for k in keys}
    zero = jnp.zeros_like(grads["x"][:, 0])
    if displacement["x"].mesh.dim == 3:
        out = jnp.stack(
            [
                grads["x"][:, 0],
                grads["y"][:, 1],
                grads["z"][:, 2],
                grads["y"][:, 2] + grads["z"][:, 1],
                grads["x"][:, 2] + grads["z"][:, 0],
                grads["x"][:, 1] + grads["y"][:, 0],
            ],
            axis=-1,
        )
        return out[0] if single else out
    out = jnp.stack(
        [
            grads["x"][:, 0],
            grads["y"][:, 1],
            zero,
            grads["z"][:, 1],
            grads["z"][:, 0],
            grads["x"][:, 1] + grads["y"][:, 0],
        ],
        axis=-1,
    )
    q = displacement.wavenumber
    if q:
        # the axial derivative of e^{-iqz} u is -iq u: zz, yz and xz
        values = {
            k: jnp.reshape(_probe.field_at(displacement[k], points), (-1,))
            for k in keys
        }
        along = jnp.stack(
            [zero, zero, values["z"], values["y"], values["x"], zero], axis=-1
        )
        out = out - 1j * q * along
    return out[0] if single else out


def _region_names(region: Any) -> tuple[str, ...]:
    if isinstance(region, str):
        return (region,)
    if (
        isinstance(region, Sequence)
        and region
        and all(isinstance(r, str) for r in region)
    ):
        return tuple(dict.fromkeys(region))
    raise TypeError(
        "region_mean's region must be a volume region name or a non-empty "
        f"sequence of names, got {region!r}."
    )


def _check_mean_regions(field: Field, names: tuple[str, ...]) -> None:
    """Raise for a surface name, or regions with no element in the support."""
    mesh = field.mesh
    for name in names:
        if mesh.region(name).dim != mesh.dim:
            raise ValueError(
                f"region_mean averages over volume regions, but {name!r} is a "
                "surface region of the mesh."
            )
    tags = np.asarray(mesh.tetrahedron_tags)
    wanted = [mesh.region(n).tag for n in names if n in field.support]
    selected = np.isin(tags, wanted)
    if selected.any() and field._excluded:  # noqa: SLF001
        entities = np.asarray(mesh.tetrahedron_entities)
        for tag, entity in field._excluded:  # noqa: SLF001
            selected &= ~((tags == tag) & (entities == entity))
    if not selected.any():
        raise ValueError(
            f"region_mean over {list(names)} holds no element of the field's "
            f"support {list(field.support)}: average over regions the field is "
            "defined on."
        )


class _MeanKernel:
    """A field's region mean: ``M c`` and its transpose."""

    name = "a region mean"
    pieces = frozenset({"coefficients"})

    def __init__(self, field: Field, names: tuple[str, ...]) -> None:
        self.topology = field.mesh._topology  # noqa: SLF001
        self.layout = (
            tuple(field.support),
            tuple(field._excluded),  # noqa: SLF001
            field.space,
            field.order,
        )
        self.names = names
        self.comm = None
        components = _probe.COMPONENTS[field.space]
        shape = () if components == 1 else (components,)
        dtype = field.coefficients.dtype
        self.output_specs = {"mean": jax.ShapeDtypeStruct(shape, dtype)}

    def role(self, name: str) -> str:
        del name
        return "read"

    def unsupported(self, name: str) -> str | None:
        return _probe.NODES_REASON if name == "nodes" else None

    def describe(self, name: str) -> str:
        return {"nodes": "the mesh node coordinates (field.mesh.nodes)"}.get(
            name, f"the field {name}"
        )

    def _rows(self, nodes: np.ndarray) -> np.ndarray:
        from gfem._engine.fields import mean_rows

        return mean_rows(Mesh(nodes, self.topology), self.layout, self.names)

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rows = self._rows(values["nodes"])
        mean = rows @ values["coefficients"]
        return {"mean": mean[0] if len(rows) == 1 else mean}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        if wanted != ("coefficients",):  # the measure wrapper raised for others
            raise NotImplementedError(f"no region mean gradient for {wanted}")
        rows = self._rows(values["nodes"])
        ct = np.reshape(cotangents["mean"], (len(rows),))
        return {"coefficients": rows.T @ ct}


def element_mean(field: Field, region: str | Sequence[str]) -> jax.Array:
    r"""Mean of a scalar field over each tetrahedron of volume regions.

    Each value is :math:`\int_e f\, dV / |e|`, exact for the solved field (the
    integral of its finite-element polynomial), in ``mesh.elements(region)``
    order, the order a `gfem.MaterialField` on that region takes: a
    temperature rise per element gives a per-element permittivity, for
    example. It is `region_mean` without the sum: linear in the field's
    coefficients, so differentiable like it, and the volume-weighted
    average of the element means is the region mean. Inside a perfectly
    matched layer (`gfem.PML`) the means are of the stretched field, not a
    physical one.

    Example:
        A permittivity that follows the temperature rise in the core::

            rise = gfem.element_mean(thermal.rise, "core")
            eps = (n0 + dn_dt * rise) ** 2
            core = gfem.MaterialField(eps_r=eps, region="core")

    Args:
        field: A scalar field of a result (H1 or L2), e.g.
            ``result.rise`` or ``result.potential["t1"]``.
        region: A volume region name, or a sequence of names (their
            elements in turn).

    Returns:
        ``(n_elements,)``, one mean per tetrahedron of the region(s).

    Raises:
        TypeError: ``field`` is not a scalar `gfem.Field`, or ``region``
            is neither a name nor a non-empty sequence of names.
        RegionNotFoundError: A named region does not exist.
        ValueError: A name is a surface region, or the region holds an
            element outside the field's support.
        gfem.DistributedUnsupportedError: The field is a piece across ranks.
        NonDifferentiableError: The mesh node coordinates are differentiated.
    """
    from gfem._field import Field

    refuse_entry(getattr(field, "mesh", None), "element_mean")
    if not isinstance(field, Field):
        raise TypeError(f"element_mean needs a gfem.Field, got {type(field).__name__}.")
    if _probe.COMPONENTS[field.space] != 1:
        raise TypeError(
            f"element_mean averages a scalar field, but this {field.quantity!r} "
            f"field is a vector ({field.space}); average a vector field over a "
            "region with region_mean."
        )
    if field._communicator is not None:  # noqa: SLF001
        _mpi.refuse_feature("element_mean")
    names = _region_names(region)
    _check_mean_regions(field, names)
    tets = np.concatenate([field.mesh.elements(n) for n in names])
    return _element_means(
        field,
        tets,
        f"element_mean over {list(names)}: {{n}} of their elements are outside "
        "the field's support (excluded from its solve), where it has no value. "
        "Average over regions the field is defined on.",
    )


def _element_means(field: Field, tetrahedra: np.ndarray, outside: str) -> jax.Array:
    """The exact mean of a scalar ``field`` over each of ``tetrahedra``.

    `element_mean`'s kernel on given tetrahedra of the field's own mesh,
    differentiable in the field's coefficients.

    Raises:
        ValueError: Some tetrahedra are outside the field's support;
            ``outside`` is the message, ``{n}`` their count.
    """
    mesh = field.mesh
    tets = np.asarray(tetrahedra)
    inside = np.isin(
        mesh.tetrahedron_tags[tets], [mesh.region(n).tag for n in field.support]
    )
    if field._excluded:  # noqa: SLF001
        entities = np.asarray(mesh.tetrahedron_entities)[tets]
        tags = np.asarray(mesh.tetrahedron_tags)[tets]
        for tag, entity in field._excluded:  # noqa: SLF001
            inside &= ~((tags == tag) & (entities == entity))
    if not inside.all():
        raise ValueError(outside.format(n=int((~inside).sum())))
    kernel = _ElementMeanKernel(field, tets)
    return measure(
        kernel, {"coefficients": field.coefficients, "nodes": field.mesh.nodes}
    )["mean"]


class _ElementMeanKernel(_MeanKernel):
    """A scalar field's element means: ``R c`` and its transpose."""

    name = "an element mean"

    def __init__(self, field: Field, tetrahedra: np.ndarray) -> None:
        super().__init__(field, ())
        self.tetrahedra = tetrahedra
        dtype = field.coefficients.dtype
        self.output_specs = {"mean": jax.ShapeDtypeStruct((len(tetrahedra),), dtype)}

    def _rows(self, nodes: np.ndarray) -> Any:
        from gfem._engine.fields import element_rows

        return element_rows(Mesh(nodes, self.topology), self.layout, self.tetrahedra)

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {"mean": self._rows(values["nodes"]) @ values["coefficients"]}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        if wanted != ("coefficients",):  # the measure wrapper raised for others
            raise NotImplementedError(f"no element mean gradient for {wanted}")
        rows = self._rows(values["nodes"])
        return {"coefficients": rows.T @ np.asarray(cotangents["mean"])}


def _recorded_region_mean(
    recorder: _objective.Recorder, field: Field, names: tuple[str, ...]
) -> jax.Array:
    """`region_mean` while an objective is classified: one recorded output."""
    label = recorder.label(field, "region_mean")
    concrete = recorder.field(label)
    components = _probe.COMPONENTS[field.space]
    return recorder.output(
        f"region_mean|{label}|{'+'.join(names)}",
        () if components == 1 else (components,),
        concrete.coefficients.dtype,
        lambda: region_mean(concrete, list(names)),
        measurement="region_mean",
        label=label,
        regions=names,
    )


# --- electric energy ---------------------------------------------------------------


def electric_energy(field: Field, *, model: Model) -> dict[str, jax.Array]:
    r"""Time-averaged electric energy of a field in each volume region, J.

    :math:`W_e[r] = \frac{1}{4} \varepsilon_0 \int_r E^* \cdot
    \mathrm{Re}(\varepsilon_r) E\, dV` over the region's elements
    in the solve domain, with the full real permittivity tensor in an
    anisotropic medium. In a region with a `gfem.PermittivityLaw` the
    weight is the Brillouin one,
    :math:`\mathrm{Re}\, d(\omega\varepsilon)/d\omega` at the field's
    real frequency (the time-averaged electric energy of a dispersive
    medium, exact for a lossless law and the weak-loss value otherwise).
    The field is a peak phasor, hence the 1/4. A driven field is the one for
    1 W incident at its excited port
    (`gfem.conventions`), so in a region with one loss tangent the power
    dielectric loss absorbs is :math:`2 \omega \tan\delta\, W_e[r]` W, the
    number `power_flow` gives for that region when nothing else in it
    absorbs.

    A mode field of an `gfem.EigenmodeResult` is accepted too. The mode
    normalization counts :math:`\frac{1}{2} \mathrm{Re} \int D^* \cdot E`,
    the total energy of a lossless
    mode and twice this time average: ``2 * W_e[r]`` equals
    ``bulk_participation[r] * gfem.conventions.MODE_ENERGY`` in a region
    without a permittivity law (the mode energy weighs a law region with its
    energy permittivity instead, see `gfem.conventions`).

    On a model with a `gfem.PML` the layer regions get no entry: there the
    field is the stretched one, not a physical field, and its energy would
    depend on the layer's thickness and target reflection.

    The integral is exact for the solved field and the model's element-wise
    permittivity. A field of an adaptive sweep (`gfem.ReducedFields`) is
    exact at the solved frequencies and accurate to the sweep's tolerance in
    between.

    Differentiable with respect to the field's coefficients, the
    permittivity of a region without a law (a scalar, per element or per
    axis), a permittivity law's parameters, a mode field's frequency and the
    node coordinates. The loss tangent, conductivity and permeability are
    not read: only the field carries them. The coefficients of an adaptive
    sweep's field are read but have no gradient (solve with
    ``sweep="uniform"`` at the frequency you differentiate).

    Args:
        field: A time-harmonic electric field (quantity ``"E"``, in H(curl),
            with an ``omega``), e.g. ``result.e_field(f)``.
        model: The model the field was solved with: the permittivities and
            regions the integral reads.

    Returns:
        A 0-d float64 array per volume region with elements in the solve
        domain, outside a perfectly matched layer, by region name, J.

    Raises:
        TypeError: ``field`` is not a `gfem.Field`, or ``model`` is not a
            `gfem.Model`.
        ValueError: The field is not a time-harmonic electric field (another
            quantity, or static), or was not solved on ``model``; or a
            permittivity law gives
            :math:`\mathrm{Re}\, d(\omega\varepsilon)/d\omega \le 0` in its
            region at the field's frequency (loss dominates there: a metal
            below its damping frequency, a Lorentz pole inside its
            linewidth).
        NotImplementedError: ``model`` holds a metal-dielectric blend
            (`MaterialField.interpolate` with a permittivity law), which is
            supported on the driven uniform sweep only.
        gfem.DistributedUnsupportedError: The field is a piece across ranks.
        NonDifferentiableError: The field's coefficients, formed from an
            adaptive sweep, are differentiated.
    """
    from gfem._field import Field
    from gfem.model import Model

    refuse_entry(getattr(field, "mesh", None), "electric_energy")
    if not isinstance(field, Field):
        raise TypeError(
            f"electric_energy needs a gfem.Field, got {type(field).__name__}."
        )
    if field._communicator is not None:  # noqa: SLF001
        _mpi.refuse_feature("electric_energy")
    if not isinstance(model, Model):
        raise TypeError(
            f"electric_energy needs model= a gfem.Model, got {type(model).__name__}."
        )
    from gfem._engine.dispersion import refuse_blends

    if _objective.recording() is not None:
        raise MeshRefinementUnsupportedError(
            "electric_energy",
            "The error of gfem.electric_energy is not estimated in this "
            "version. Read the energy of an electrostatic result from "
            "result.energy.",
        )
    refuse_blends(model, "electric_energy")
    _check_time_harmonic(
        field,
        "electric_energy",
        static="an electrostatic result has its energy per region in result.energy",
    )
    energy = importlib.import_module("gfem._energy")
    return {
        region: jnp.asarray(value, dtype=float)
        for region, value in energy.electric_energy(field, model).items()
    }


def _check_time_harmonic(field: Field, who: str, *, static: str) -> None:
    """Raise unless ``field`` is a time-harmonic electric field.

    ``who`` names the measurement in the messages, ``static`` is the remedy
    for a static field.

    Raises:
        ValueError: Another quantity or space, or no ``omega``.
    """
    if field.quantity != "E" or field.space != "hcurl":
        raise ValueError(
            f"{who} needs an electric field (quantity 'E' in 'hcurl'), got "
            f"quantity {field.quantity!r} in {field.space!r}: pass the E field "
            "the B field or potential came from."
        )
    if field.omega is None:
        raise ValueError(
            f"{who} needs a time-harmonic field; this one is static (omega is "
            f"None): {static}."
        )


def _check_driven(field: Field) -> None:
    """Raise unless ``field`` is a driven electric field.

    Raises:
        ValueError: Another quantity or space, real coefficients, or an
            ``omega`` that is missing or complex.
    """
    _check_time_harmonic(field, "power_flow", static="solve a gfem.Driven problem")
    if jnp.iscomplexobj(field.omega):
        raise ValueError(
            "power_flow needs a driven field with a real omega; eigenmode fields "
            "have no time-averaged power flow (complex omega, arbitrary "
            "normalization)."
        )
    if not field.is_complex:
        raise ValueError(
            "power_flow needs a complex phasor field, got real coefficients: "
            "solve a gfem.Driven problem."
        )
