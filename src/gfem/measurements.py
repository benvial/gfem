"""Post-hoc measurements on result fields.

A result carries every quantity that needs no extra input. A quantity that
needs extra physics input (a probe point, an interface surface with its
thickness and permittivity, a voltage path) is a measurement: a function
called on a result that carries its fields, after the solve, also on a
result reloaded with `gfem.load_result`.

- `Field.at`: a field's value at points.
- `interface_participation`: the electric energy of a mode in a thin
  dielectric layer on a surface, and its share of ``1 / Q``.
- `characteristic_impedance`: the power-voltage impedance of a port mode
  along a voltage path.

Measurements that evaluate fields run as host kernels behind gfem's
gradient wrapper: they work under `jax.jit` and `jax.vmap`, and under
`jax.grad` they differentiate with respect to the field coefficients (so
material-field gradients flow through the solve that made the field) and
the measurement's own numbers. Mesh-node gradients raise
`gfem.NonDifferentiableError` in this version.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem import _probe
from gfem._checks import check_fixed, is_traced
from gfem._materials import Material
from gfem._pytree import register_fields
from gfem.conventions import MODE_ENERGY
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.results import EigenmodeResult, PortModeResult

__all__ = [
    "INTERFACE_KINDS",
    "InterfaceParticipation",
    "characteristic_impedance",
    "interface_participation",
]

# --- interface participation -------------------------------------------------------

#: The thin-layer models of `interface_participation` (Wenner et al., Appl.
#: Phys. Lett. 99, 113513 (2011); Palace's ``Dielectric`` types).
InterfaceKind = Literal["default", "MA", "MS", "SA"]
INTERFACE_KINDS = get_args(InterfaceKind)
#: A medium is vacuum when its fastest light speed is within this of c
#: (Palace's test).
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
    at the surface, for each kind of interface (Wenner et al. 2011, the
    definitions of Palace's ``Dielectric`` postprocessing):

    - ``"MA"`` (metal-air): ``t / eps_r * |E_n|^2``, ``E`` on the vacuum
      side of the surface.
    - ``"MS"`` (metal-substrate): ``t / eps_r * |(eps_s E)_n|^2``, ``E``
      and the substrate permittivity ``eps_s`` (real part, a tensor if
      anisotropic) on the non-vacuum side: the layer carries the
      substrate's normal flux.
    - ``"SA"`` (substrate-air): ``t * (eps_r |E_t|^2 + |E_n|^2 / eps_r)``,
      ``E`` on the vacuum side.
    - ``"default"``: ``t * eps_r * |E|^2``, ``E`` averaged over the two
      sides of an interior surface.

    each times ``eps0 / 2`` and integrated over the surface. A medium is
    vacuum when its light speed is that of vacuum (to 1e-6). Where both
    sides are of the wanted kind, the two sides' values are averaged; where
    neither is (``"MA"`` on a plate under substrate), the surface adds
    nothing. Sides inside volumes the model excludes from the solve
    (metal) do not count. The participation is that energy over the mode's
    electric energy ``E`` (1 J, see `gfem.conventions`), and
    ``participation * tan_delta`` is the layer's contribution to ``1 / Q``
    (Palace's ``surface-Q.csv``).

    Differentiable with respect to ``thickness``, ``eps_r``, ``tan_delta``,
    the mode fields and the substrate's `MaterialField` permittivity (the
    ``"MS"`` flux). Mode fields are not differentiable in this version, so
    a gradient that reaches the eigenmode solve raises
    `gfem.NonDifferentiableError` there; mesh-node gradients raise.

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
        ValueError: A name is a volume region, a number is out of range, or
            the result does not carry its materials (saved by an older gfem).
    """
    from gfem.results import EigenmodeResult

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
    geometry = _surface_geometry(field, surfaces)
    coefficients = jnp.stack([f.coefficients for f in modes.e_fields])
    energy = _layer_energy(
        geometry,
        kind,
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

    def __init__(self, mesh: Mesh, surfaces: tuple[str, ...], layout: Any) -> None:
        triangles = []
        for name in surfaces:
            region = mesh.region(name)
            if region.dim != 2:
                raise ValueError(
                    f"{name!r} is a volume region; interface_participation needs "
                    "a surface region."
                )
            triangles.append(mesh.elements(name))
        engine = _probe.engine()
        tri = np.concatenate(triangles)
        sides = mesh.triangle_neighbours[tri]
        inside = engine.in_support(mesh, layout)
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


def _surface_geometry(field: Field, surfaces: tuple[str, ...]) -> _SurfaceGeometry:
    mesh = field.mesh
    layout = _probe.layout(field)
    key = (id(mesh._topology), surfaces, layout)  # noqa: SLF001
    found = _GEOMETRIES.get(key)
    if found is not None and found.topology is mesh._topology:  # noqa: SLF001
        return found
    made = _SurfaceGeometry(mesh, surfaces, layout)
    _GEOMETRIES[key] = made
    while len(_GEOMETRIES) > _MAX_GEOMETRIES:
        _GEOMETRIES.pop(next(iter(_GEOMETRIES)))
    return made


@functools.partial(jax.jit, static_argnames=("geometry", "kind"))
def _layer_energy(
    geometry: _SurfaceGeometry,
    kind: InterfaceKind,
    *,
    coefficients: jax.Array,
    nodes: jax.Array,
    materials: dict[str, Any],
    thickness: jax.Array,
    eps_l: jax.Array,
) -> jax.Array:
    """The layer's electric energy per mode, J (see `interface_participation`)."""
    g = geometry
    mesh = Mesh(nodes, g.topology)
    m, n_t, n_q = coefficients.shape[0], len(g.triangles), g.n_q
    values = _probe.evaluate(coefficients, mesh, g.layout, g.located)
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
        d = jnp.einsum("tsij,mtsqj->mtsqi", eps_side, e)
        dn = jnp.einsum("mtqi,ti->mtq", _average(d, ex & ~vacuum), normal)
        density = _sq(dn) / eps_l
    else:
        selected = _average(e, ex & vacuum)
        en = jnp.einsum("mtqi,ti->mtq", selected, normal)
        if kind == "MA":
            density = _sq(en) / eps_l
        else:  # SA
            et = selected - en[..., None] * normal[None, :, None, :]
            density = eps_l * _norm2(et) + _sq(en) / eps_l
    return 0.5 * const.epsilon_0 * thickness * jnp.einsum("mtq,tq->m", density, weights)


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


def _media(
    materials: dict[str, Any], mesh: Mesh, sides: np.ndarray, exists: np.ndarray
) -> tuple[jax.Array, jax.Array]:
    """Real permittivity tensors ``(t, 2, 3, 3)`` and vacuum flags ``(t, 2)``."""
    from gfem._engine.coefficients import _tensor

    tets = np.maximum(sides, 0)
    eps = jnp.zeros((*sides.shape, 3, 3)) + jnp.eye(3)
    speed2 = jnp.ones(sides.shape)  # (c_max / c0)^2 = 1 / (eps_min mu_min)
    for region, material in materials.items():
        members = mesh.elements(region)
        position = np.full(len(mesh.tetrahedra), -1)
        position[members] = np.arange(len(members))
        inside = jnp.asarray((position[tets] >= 0) & exists)
        if isinstance(material, Material):
            tensor = jnp.asarray(_tensor(material).real)
            low = float(np.min(material.eps_r)) * float(np.min(material.mu_r))
            value = jnp.broadcast_to(tensor, eps.shape)
            low = jnp.full(sides.shape, low)
        else:
            e_r = 1.0 if material.eps_r is None else material.eps_r
            m_r = 1.0 if material.mu_r is None else material.mu_r
            e_r = jnp.real(jnp.asarray(e_r))
            m_r = jnp.real(jnp.asarray(m_r))
            index = np.maximum(position[tets], 0)
            e_r = e_r[index] if e_r.ndim else jnp.full(sides.shape, e_r)
            m_r = m_r[index] if m_r.ndim else jnp.full(sides.shape, m_r)
            value = e_r[..., None, None] * jnp.eye(3)
            low = e_r * m_r
        eps = jnp.where(inside[..., None, None], value, eps)
        speed2 = jnp.where(inside, 1.0 / low, speed2)
    vacuum = speed2 >= _VACUUM**2
    return eps, vacuum


# --- characteristic impedance ------------------------------------------------------


def characteristic_impedance(mode: PortModeResult, voltage_path: Any) -> jax.Array:
    """Power-voltage characteristic impedance of a port mode along a path.

    ``Z_PV = |V|^2 / (2 P)`` with ``V`` the line integral of the mode's
    electric field along ``voltage_path`` and ``P = 1 W`` the mode's power
    (Palace's ``mode-Z``). For a wave port declared with a ``voltage_path``
    the result already holds it as ``Z_PV``; this measures it along any
    path, after the solve. A constant of the port mode solve: port modes
    are not differentiable.

    Args:
        mode: A port mode result.
        voltage_path: Two or more points (m) on the port surface; the path
            runs through them in order, e.g. from a ground to the signal
            conductor.

    Returns:
        ``(n_freq,)`` ohm.

    Raises:
        TypeError: ``mode`` is not a `gfem.PortModeResult`.
        ValueError: The path is malformed or leaves the port plane, or
            ``mode`` carries no field (a driven result without kept fields).
        NonDifferentiableError: ``voltage_path`` is traced by JAX.
    """
    from gfem._sweep.port_mode import path_voltage
    from gfem.results import PortModeResult

    if not isinstance(mode, PortModeResult):
        raise TypeError(
            "characteristic_impedance needs a gfem.PortModeResult, got "
            f"{type(mode).__name__}."
        )
    check_fixed(voltage_path, "voltage_path must be fixed numbers")
    path = np.asarray(voltage_path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError(
            "voltage_path must be two or more points (m), each a 3-vector, got "
            f"shape {path.shape}."
        )
    if not np.isfinite(path).all():
        raise ValueError("voltage_path must hold finite numbers.")
    field = mode.e_field
    if field is None:
        raise ValueError(
            f"The port mode of {mode.port!r} carries no field: a driven result "
            "keeps its wave ports' mode fields only with Driven(keep_fields=...). "
            "Solve with keep_fields, or solve gfem.PortMode for the port."
        )
    volts = path_voltage(field._space, field._basis, field._coords, path)  # noqa: SLF001
    return jnp.asarray(np.abs(volts) ** 2 / 2.0)
