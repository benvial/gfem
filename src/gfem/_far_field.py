r"""`gfem.far_field`: the far-field amplitude and the IEEE 145 radiation outputs.

The amplitude is the reciprocity integral of `gfem._engine.far_field`
over the element layer just outside a side, a linear function of the field
coefficients and a closed-form function of the layer's node coordinates.
The radiated power is `gfem.power_flow`'s residual through the same side,
so the directivity integrates to :math:`4\pi` to the discretization of
:math:`U` alone. One host kernel computes both, behind gfem's gradient
wrapper (`gfem._adjoint.measure`):

- the field coefficients: differentiable (both outputs), unless the field
  is formed from an adaptive sweep's frozen basis;
- the field's node coordinates: differentiable (both outputs; the radiated
  power through `power_flow`'s node pass);
- ``omega``: read, not differentiable;
- the model's leaves: as `gfem.power_flow` reads them (a design in its
  layer raises; the far-field layer refuses a design before that).
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem import _curved as curved
from gfem import _incident, _pml
from gfem._adjoint.measure import measure
from gfem._checks import check_fixed, is_traced
from gfem._dim2 import refuse_entry
from gfem._field_kernels import (
    Role,
    RoleKernel,
    design_key,
    engine,
    field_inputs,
    field_role,
    model_inputs,
    rebuild,
    reduced_reason,
)
from gfem._power_flow import _model_role, concrete_field, side_names
from gfem._pytree import register_fields
from gfem._sources import DIPOLES, INCIDENT
from gfem.errors import NonDifferentiableError
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._engine.far_field import Layer
    from gfem._engine.power_flow import Surface
    from gfem._field import Field
    from gfem.model import Model
    from gfem.results import DrivenResult

__all__ = ["FarField", "far_field"]


@dataclasses.dataclass(frozen=True, eq=False)
class FarField:
    r"""The far field of one excitation on a grid of directions, per kept frequency.

    Under :math:`e^{+i\omega t}` the field far from the sources is
    :math:`E(r\hat r) \approx F(\hat r)\,e^{-ikr}/r`, with :math:`k` and
    :math:`\eta` of the exterior medium and the phase referred to the
    model's origin (`gfem.conventions`, Far fields). Every array has the
    kept frequencies first, then the broadcast shape of ``theta`` and
    ``phi``.

    Attributes:
        frequency: The kept frequencies of the column, Hz, ``(n_freq,)``.
        theta: The polar angles from :math:`+z`, rad.
        phi: The azimuths from :math:`+x`, rad.
        e_theta: :math:`F_\theta`, V (m times V/m).
        e_phi: :math:`F_\phi`, V.
        radiated_power: :math:`P_\mathrm{rad}`, W, ``(n_freq,)``: the power
            flowing out of the side, from the residual of the discrete
            equations (`gfem.power_flow`).
        accepted_power: :math:`P_\mathrm{acc}`, W, ``(n_freq,)``: a port's
            :math:`1 - |S_{ii}|^2` (1 W incident), a dipole's emitted power;
            ``None`` for an incident field.
        incident_power: :math:`P_\mathrm{inc}`, W, ``(n_freq,)``: a port's
            1 W, a dipole's emitted power (no mismatch); ``None`` for an
            incident field.
        incident_amplitude: :math:`|E_0|`, V/m, ``(n_freq,)``, of a
            `gfem.PlaneWave` column, ``None`` otherwise.
        impedance: :math:`\eta`, the exterior medium's wave impedance, ohm.
    """

    frequency: tuple[float, ...]
    theta: Any
    phi: Any
    e_theta: jax.Array
    e_phi: jax.Array
    radiated_power: jax.Array
    accepted_power: jax.Array | None
    incident_power: jax.Array | None
    incident_amplitude: jax.Array | None
    impedance: float

    def _per_frequency(self, x: Any) -> Any:
        """``x`` of shape ``(n_freq,)`` broadcast against the direction grid."""
        return jnp.reshape(x, (-1,) + (1,) * (jnp.ndim(self.e_theta) - 1))

    @property
    def intensity(self) -> jax.Array:
        r"""The radiation intensity :math:`U = |F|^2 / (2\eta)`, W/sr."""
        return (jnp.abs(self.e_theta) ** 2 + jnp.abs(self.e_phi) ** 2) / (
            2 * self.impedance
        )

    @property
    def directivity(self) -> jax.Array:
        r"""The directivity :math:`D = 4\pi U / P_\mathrm{rad}`."""
        return 4 * math.pi * self.intensity / self._per_frequency(self.radiated_power)

    def _driven(self, what: str) -> tuple[jax.Array, jax.Array]:
        if self.accepted_power is None or self.incident_power is None:
            raise ValueError(
                f"The {what} is an antenna's (a port or a dipole column): an "
                "incident field's column has no accepted power; read rcs."
            )
        return self.accepted_power, self.incident_power

    @property
    def gain(self) -> jax.Array:
        r"""The gain :math:`G = 4\pi U / P_\mathrm{acc}` (IEEE 145)."""
        accepted, _ = self._driven("gain")
        return 4 * math.pi * self.intensity / self._per_frequency(accepted)

    @property
    def realized_gain(self) -> jax.Array:
        r"""The realized gain :math:`G_r = 4\pi U / P_\mathrm{inc}` (IEEE 145).

        :math:`4\pi U` for a port column, fed 1 W.
        """
        _, incident = self._driven("realized gain")
        return 4 * math.pi * self.intensity / self._per_frequency(incident)

    @property
    def radiation_efficiency(self) -> jax.Array:
        r""":math:`P_\mathrm{rad} / P_\mathrm{acc}`, ``(n_freq,)``."""
        accepted, _ = self._driven("radiation efficiency")
        return self.radiated_power / accepted

    @property
    def e_right(self) -> jax.Array:
        r"""The right-hand circular component :math:`(F_\theta + iF_\phi)/\sqrt2`, V."""
        return (self.e_theta + 1j * self.e_phi) / math.sqrt(2)

    @property
    def e_left(self) -> jax.Array:
        r"""The left-hand circular component :math:`(F_\theta - iF_\phi)/\sqrt2`, V."""
        return (self.e_theta - 1j * self.e_phi) / math.sqrt(2)

    @property
    def axial_ratio(self) -> jax.Array:
        r"""The axial ratio :math:`(|F_R| + |F_L|)/\big||F_R| - |F_L|\big|`.

        1 for circular polarization, ``inf`` for linear.
        """
        r, left = jnp.abs(self.e_right), jnp.abs(self.e_left)
        return (r + left) / jnp.abs(r - left)

    @property
    def rcs(self) -> jax.Array:
        r"""The bistatic radar cross section :math:`4\pi|F|^2/|E_0|^2`, m².

        Of a `gfem.PlaneWave` column's scattered field.

        Raises:
            ValueError: The column is not a plane wave's.
        """
        if self.incident_amplitude is None:
            raise ValueError(
                "The radar cross section is a plane wave's (a gfem.PlaneWave "
                "column): it divides by the incident amplitude |E_0|; compute "
                "4 pi |F|^2 / |E_0|^2 for another incident field."
            )
        f2 = jnp.abs(self.e_theta) ** 2 + jnp.abs(self.e_phi) ** 2
        return 4 * math.pi * f2 / self._per_frequency(self.incident_amplitude) ** 2

    def __repr__(self) -> str:
        shape = tuple(np.shape(self.e_theta))
        return f"FarField(frequency={self.frequency!r}, shape={shape})"


register_fields(
    FarField,
    (
        "theta",
        "phi",
        "e_theta",
        "e_phi",
        "radiated_power",
        "accepted_power",
        "incident_power",
        "incident_amplitude",
    ),
)


def far_field(
    result: DrivenResult,
    name: str,
    theta: Any,
    phi: Any,
    *,
    model: Model,
    side: str | Sequence[str] | Callable[[np.ndarray], np.ndarray] | None = None,
) -> FarField:
    r"""The far field of one excitation on a grid of directions.

    The far-field amplitude :math:`F(\hat r) = \lim r\,e^{+ikr}E(r\hat r)`
    (V, phase referred to the model origin, :math:`k` of the exterior
    medium; `gfem.conventions`, Far fields) of a port, point dipole or
    incident-field column, for each of its kept frequencies, with the IEEE
    145 quantities built on it (`FarField`): radiation intensity, radiated
    power, directivity, gain, realized gain, radiation efficiency, circular
    components, axial ratio, and for a plane wave the bistatic radar cross
    section of its scattered field.

    The amplitude is a reciprocity integral over the element layer just
    outside ``side``, with :math:`\chi` the side's P1 indicator and
    :math:`w = \hat q\,e^{ik\hat r\cdot r}` the plane wave arriving from
    :math:`\hat r`:

    .. math::

        \hat q\cdot F(\hat r) = \frac{1}{4\pi}\int \Big[
            E\cdot(\nabla\chi\times\nabla\times w)
            + (\nabla\times E)\cdot(\nabla\chi\times w)\Big]\,dV,

    the Stratton-Chu surface integral averaged over the layer: no surface,
    normals or face traces, and the field is read inside elements. The
    radiated power is `power_flow`'s residual through the same side, exact
    for the solved field. A flat PEC or PMC exterior face that the layer
    meets is a ground plane: the test field gets its mirror image, so the
    far field is that of the sources and their images, in the half-space
    above the plane only.

    On a result of a distributed solve it is a collective call: every rank
    calls it and gets the same far field.

    Example:
        The pattern of a dipole on a 2° grid::

            theta = np.radians(np.arange(0, 181, 2))[:, None]
            phi = np.radians(np.arange(0, 360, 2))[None, :]
            ff = gfem.far_field(result, "d", theta, phi, model=model)
            ff.directivity[0]  # (91, 180)

    Args:
        result: A `gfem.DrivenResult` that kept the column's fields
            (``Driven(..., keep_fields=...)``).
        name: The excited port or the source whose far field to read.
        theta: Polar angles from :math:`+z`, rad (fixed numbers).
        phi: Azimuths from :math:`+x`, rad, broadcast against ``theta``.
        model: The model the result was solved on, with its background as
            the exterior medium (``Model(background=...)``).
        side: ``None`` (the default): every physical element with no vertex
            on the perfectly matched layer or an `gfem.Absorbing` face, so
            the layer read is the one touching the truncation. Or volume
            region names, or a predicate ``inside(centroids) -> bool`` on the
            element centroids (m), as for `power_flow`; the side must hold
            every source, port, sheet and scatterer, one element clear of
            its outside layer.

    Returns:
        A `FarField`; its arrays are ``(n_freq, *shape)`` with ``shape`` the
        broadcast shape of ``theta`` and ``phi``.

    Raises:
        TypeError: ``result`` or ``model`` is of the wrong type, or ``side``
            is neither names, a predicate nor ``None``.
        KeyError: ``name`` is neither an excited port nor a source.
        RegionNotFoundError: A named region does not exist.
        ValueError: No field of ``name`` was kept, or ``name`` is a dipole of
            a ``Driven(source_fields=False)`` solve, whose field was not
            solved; the model has no background, no PML layer and no
            `gfem.Absorbing` face, or `gfem.Periodic` pairs; ``theta`` and
            ``phi`` do not broadcast or are not finite; the side selects nothing
            or everything; its outside layer is not the exterior medium (another
            material, a `gfem.MaterialField`, the perfectly matched layer); a dipole,
            port, sheet, wall or scatterer is in that layer or outside the
            side; the layer meets a boundary of the solve domain that is not
            one flat PEC or PMC ground plane; a direction points below the
            ground plane.
        NotImplementedError: A layered background (a dielectric or layered
            half-space) or a lossy one.
        NonDifferentiableError: ``theta`` or ``phi`` is traced; the side is
            a predicate while the node coordinates are traced; and, while
            `jax.grad` traces it, the field's ``omega`` or a design in
            `power_flow`'s layer. Every output is differentiable in the node
            coordinates.
    """
    from gfem.model import Model
    from gfem.results import DrivenResult

    refuse_entry(getattr(model, "mesh", None), "far_field")
    if not isinstance(result, DrivenResult):
        raise TypeError(
            f"far_field needs a gfem.DrivenResult, got {type(result).__name__}."
        )
    if not isinstance(model, Model):
        raise TypeError(
            f"far_field needs model= a gfem.Model, got {type(model).__name__}."
        )
    theta_b, phi_b = _angles(theta, phi)
    names = side_names(side, "far_field", default="the layer touching the truncation")
    source = result.sources.get(name)
    if source is None and name not in result.s.excited:
        columns = ", ".join(map(repr, result._excitations)) or "none"  # noqa: SLF001
        raise KeyError(
            f"{name!r} is neither an excited port nor a source of the result; its "
            f"columns are {columns}."
        )
    result._refuse_unsolved_source(  # noqa: SLF001
        name, "so it has no far field"
    )
    background = model.background
    if background is None:
        raise ValueError(
            "far_field reads the exterior medium from the model's background: "
            "give Model(background=gfem.Material(...)), the medium the field "
            "radiates into."
        )
    frequencies = [f for f, c in result.kept if c == name]
    if not frequencies:
        raise ValueError(
            f"No field of {name!r} was kept: solve with Driven(..., "
            "keep_fields=True) to read its far field."
        )
    incident = isinstance(source, INCIDENT)
    part = "scattered" if incident else "total"
    fields = [result.e_field(f, name, part=part) for f in frequencies]
    traced_nodes = is_traced((fields[0].mesh.nodes, model.mesh.nodes))
    if callable(names) and traced_nodes:
        raise NonDifferentiableError(
            "far_field's side is a predicate on element centroids, which needs "
            "the node coordinates as numbers, but they are computed inside "
            "jax.grad, jax.jit or jax.vjp here: name the side's volume regions "
            "instead, so that its surface is known before the transformation runs."
        )
    concrete = concrete_field(fields[0])
    ff = engine("far_field")
    layer = ff.exterior_layer(
        concrete,
        model,
        names,
        background,
        nodes=np.asarray(concrete.mesh.nodes),
        incident=incident,
    )
    curved.refuse_tets(
        model.mesh,
        "far_field",
        layer.tetrahedra,
        "The far-field layer of gfem.far_field",
    )
    pf = importlib.import_module("gfem._engine.power_flow")
    where = pf.surface_of(concrete, layer.inside, model)
    rhat, that, phat = ff.directions(theta_b.reshape(-1), phi_b.reshape(-1))
    if layer.ground is not None:
        _check_half_space(rhat, layer.ground)
    amplitude = []
    power = []
    for field in fields:
        a, p = _measure(field, model, layer, where, (rhat, that, phat))
        amplitude.append(a)
        power.append(p)
    amp = jnp.stack(amplitude)  # (n_freq, n_dir, 2)
    shape = (len(fields), *theta_b.shape)
    accepted, offered, e0 = _normalization(result, name, source, frequencies)
    eta = const.mu_0 * const.c * math.sqrt(layer.mu_r / layer.eps_r)
    return FarField(
        frequency=tuple(float(f) for f in frequencies),
        theta=theta_b,
        phi=phi_b,
        e_theta=amp[..., 0].reshape(shape),
        e_phi=amp[..., 1].reshape(shape),
        radiated_power=jnp.stack(power),
        accepted_power=accepted,
        incident_power=offered,
        incident_amplitude=e0,
        impedance=eta,
    )


def _angles(theta: Any, phi: Any) -> tuple[np.ndarray, np.ndarray]:
    """``theta`` and ``phi`` broadcast together, checked."""
    check_fixed(theta, "far_field's theta must be fixed numbers")
    check_fixed(phi, "far_field's phi must be fixed numbers")
    try:
        theta_b, phi_b = np.broadcast_arrays(
            np.asarray(theta, float), np.asarray(phi, float)
        )
    except ValueError:
        raise ValueError(
            f"far_field's theta {np.shape(theta)} and phi {np.shape(phi)} do not "
            "broadcast together."
        ) from None
    if not (np.isfinite(theta_b).all() and np.isfinite(phi_b).all()):
        raise ValueError("far_field's theta and phi must be finite, in rad.")
    return theta_b, phi_b


def _check_half_space(rhat: np.ndarray, ground: Any) -> None:
    """Raise for a direction below the ground plane."""
    below = rhat @ ground.normal < -1e-12
    if below.any():
        k = int(np.flatnonzero(below)[0])
        raise ValueError(
            f"The model ends on a {ground.kind} ground plane (normal "
            f"{tuple(np.round(ground.normal, 12))} into the domain), and the far "
            f"field exists on its side only, but direction {tuple(rhat[k])} points "
            "below it: give theta and phi in the half-space above the plane."
        )


def _normalization(
    result: DrivenResult, name: str, source: Any, frequencies: list[float]
) -> tuple[Any, Any, Any]:
    """``(P_acc, P_inc, |E_0|)`` of the column at the kept frequencies."""
    rows = [
        int(np.argmin(np.abs(np.asarray(result.frequency) - f))) for f in frequencies
    ]
    if source is None:  # a port fed 1 W
        s = result.s[name, name]
        accepted = 1.0 - jnp.abs(jnp.asarray(s)[np.asarray(rows)]) ** 2
        return accepted, jnp.ones(len(rows)), None
    if isinstance(source, DIPOLES):
        emitted = jnp.asarray(result.emitted_power[name])[np.asarray(rows)]
        return emitted, emitted, None
    peak = _incident.amplitude(source)
    return None, None, None if peak is None else jnp.full(len(rows), peak)


def _measure(
    field: Field,
    model: Model,
    layer: Layer,
    where: Surface,
    basis: tuple[np.ndarray, ...],
) -> tuple[jax.Array, jax.Array]:
    """``(amplitude (n_dir, 2), radiated power)`` of one field."""
    inputs: dict[str, Any] = {}
    roles: dict[str, Role] = {}
    field_def = field_inputs(
        field,
        inputs,
        roles,
        lambda name: field_role(
            name,
            field,
            reduced=reduced_reason("far field"),
            nodes=None,
        ),
    )
    model_def, _ = model_inputs(
        model,
        inputs,
        roles,
        lambda kind: (
            Role("read", None, "the mesh node coordinates (model.mesh.nodes)")
            if kind[0] == "nodes"
            else _model_role(kind, model, where)
        ),
    )
    kernel = _FarFieldKernel(
        comm=field._communicator,  # noqa: SLF001
        roles=roles,
        field_def=field_def,
        model_def=model_def,
        topology=field.mesh._topology,  # noqa: SLF001
        layer=layer,
        where=where,
        order=field.order,
        basis=basis,
    )
    out = measure(kernel, inputs)
    return out["amplitude"], out["power"]


class _FarFieldKernel(RoleKernel):
    """The far-field amplitude and the radiated power of one field.

    Under ranks (``comm``) a collective host call: the coefficients are a
    rank's piece; the field at the layer's quadrature points is gathered by the
    probe, so every rank sums the same amplitude, and the radiated power is the
    flux's pairing, reduced and broadcast. The coefficients' cotangent is a
    rank's part of a load.
    """

    name = "gfem.far_field"
    pieces: ClassVar[frozenset[str]] = frozenset({"coefficients"})

    def __init__(
        self,
        *,
        comm: Any,
        roles: dict[str, Role],
        field_def: tuple[Any, tuple[str, ...]],
        model_def: tuple[Any, tuple[str, ...]],
        topology: Any,
        layer: Layer,
        where: Surface,
        order: int,
        basis: tuple[np.ndarray, ...],
    ) -> None:
        self.comm = comm
        self.roles = roles
        self.field_def = field_def
        self.model_def = model_def
        self.topology = topology
        self.layer = layer
        self.where = where
        self.order = order
        self.basis = basis
        self.output_specs = {
            "amplitude": jax.ShapeDtypeStruct((len(basis[0]), 2), jnp.complex128),
            "power": jax.ShapeDtypeStruct((), jnp.float64),
        }
        self._flux: tuple[bytes, Any] | None = None

    # --- host -------------------------------------------------------------------

    def _probe(self, nodes: np.ndarray, field: Field) -> Any:
        from gfem._engine import probe
        from gfem._probe import layout

        ff = engine("far_field")
        bary, _ = ff.rule(self.order)
        mesh = Mesh(nodes, self.topology)
        located = ff.located(self.layer, bary)
        return probe.probe(mesh, layout(field), located, self.comm)

    def _flux_of(
        self, values: dict[str, np.ndarray], model: Model, field: Field
    ) -> Any:
        pf = importlib.import_module("gfem._engine.power_flow")
        key = design_key(values, self.model_def[1])
        if self._flux is None or self._flux[0] != key:
            stretch = _pml.field_stretch(model, field)
            flux = pf.build(
                self.order, model, self.where, stretch=stretch, comm=self.comm
            )
            self._flux = (key, flux)
        return self._flux[1]

    def _function(
        self, nodes: np.ndarray, omega: float
    ) -> tuple[Callable[..., Any], np.ndarray]:
        """``(F(corners, E, curl E) -> (n_dir, 2), corners at the nodes)``."""
        ff = engine("far_field")
        mesh = Mesh(nodes, self.topology)
        corners0 = ff.corners(mesh, self.layer, nodes)
        chi = ff.chi_corners(mesh, self.layer)
        bary, weights = ff.rule(self.order)
        rhat, that, phat = self.basis
        k = omega / const.c * self.layer.index
        layer = self.layer

        def function(corners: Any, e: Any, curl: Any) -> Any:
            e_ref, c_ref = ff.reference_values(corners0, e, curl)
            v = ff.amplitudes(
                corners,
                e_ref,
                c_ref,
                chi=chi,
                bary=bary,
                weights=weights,
                rhat=rhat,
                k=k,
                ground=layer.ground,
            )
            v = v / (4 * math.pi)
            return jnp.stack(
                [jnp.sum(v * that, axis=1), jnp.sum(v * phat, axis=1)], axis=1
            )

        return function, corners0

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        field, model = (
            rebuild(self.field_def, values),
            rebuild(self.model_def, values),
        )
        pf = importlib.import_module("gfem._engine.power_flow")
        pf.check_nodes(field, model)
        nodes = np.asarray(values["nodes"], float)
        omega = float(np.asarray(values["omega"]).real)
        e = np.asarray(values["coefficients"])
        flux = self._flux_of(values, model, field)
        local = pf.local_coefficients(e, flux.op, self.comm)
        pf.check_size(local, flux.op, self.order)
        prb = self._probe(nodes, field)
        ev = prb.values(e)[0]
        cv = prb.values(e, curl=True)[0]
        function, corners = self._function(nodes, omega)
        amplitude = np.asarray(function(corners, ev, cv))
        return {"amplitude": amplitude, "power": np.asarray(flux.power(local, omega))}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        field, model = (
            rebuild(self.field_def, values),
            rebuild(self.model_def, values),
        )
        positions = np.asarray(values["nodes"], float)
        omega = float(np.asarray(values["omega"]).real)
        e = np.asarray(values["coefficients"])
        out: dict[str, np.ndarray] = {}
        for name in wanted:
            out[name] = np.zeros_like(values[name])
        nodes = [n for n in wanted if n == "nodes" or n.startswith("model.mesh")]
        # the field's and the model's nodes are one position (the host refuses
        # others): the gradient goes to the first wanted
        grad = np.zeros(np.shape(values["nodes"]))
        ct_a = cotangents.get("amplitude")
        if ct_a is not None and ("coefficients" in wanted or nodes):
            prb = self._probe(positions, field)
            ev = prb.values(e)[0]
            cv = prb.values(e, curl=True)[0]
            function, corners = self._function(positions, omega)
            _, pull = jax.vjp(function, corners, ev, cv)
            g_corners, g_e, g_c = (np.asarray(g) for g in pull(jnp.asarray(ct_a)))
            if "coefficients" in wanted:
                g = prb.transpose(g_e[None]) + prb.transpose(g_c[None], curl=True)
                out["coefficients"] = out["coefficients"] + g[0]
            if nodes:
                tets = np.asarray(Mesh(positions, self.topology).tetrahedra)
                np.add.at(grad, tets[self.layer.tetrahedra], np.real(g_corners))
        ct_p = cotangents.get("power")
        if ct_p is not None and ("coefficients" in wanted or nodes):
            pf = importlib.import_module("gfem._engine.power_flow")
            flux = self._flux_of(values, model, field)
            local = pf.local_coefficients(e, flux.op, self.comm)
            if "coefficients" in wanted:
                # a rank's piece is padded: the pullback fills its first entries
                g = float(ct_p) * flux.pullback(local, omega)
                out["coefficients"][: len(g)] += g
            if nodes:
                grad += float(ct_p) * pf.node_pullback(
                    flux, model, self.where, local, omega, comm=self.comm
                )
        if nodes:
            out[nodes[0]] = grad.reshape(np.shape(values[nodes[0]]))
        unknown = [
            n
            for n in wanted
            if n not in ("coefficients", "nodes") and not n.startswith("model.mesh")
        ]
        if unknown:  # the measure wrapper raised for the others
            raise NotImplementedError(f"no far-field gradient for {unknown}")
        return out
