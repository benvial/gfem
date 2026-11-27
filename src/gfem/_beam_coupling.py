r"""`gfem.beam_coupling`: the amplitude a solved field couples into an incoming beam.

The coupling is the reaction of `gfem._engine.beam_coupling` over the
element layer just outside a side, with the beam scaled to 1 W through
that side: a linear function of the field coefficients and a closed-form
function of the layer's node coordinates. It runs behind gfem's gradient
wrapper (`gfem._adjoint.measure`):

- the field coefficients: differentiable, unless the field is formed from
  an adaptive sweep's frozen basis;
- the node coordinates: differentiable (a `gfem.GaussianBeam` is read at
  the moved quadrature points; an `gfem.IncidentField` is frozen on the
  imported mesh and refused once its layer moves);
- ``omega``: read, not differentiable;
- the model's materials in the layer: read, not differentiable (a design
  there raises).
"""

from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from gfem import _curved as curved
from gfem._adjoint.measure import measure
from gfem._checks import is_traced
from gfem._dim2 import refuse_entry
from gfem._field_kernels import (
    Role,
    RoleKernel,
    engine,
    field_inputs,
    field_role,
    model_inputs,
    rebuild,
    reduced_reason,
)
from gfem._power_flow import concrete_field, side_names
from gfem._sources import INCIDENT, GaussianBeam, IncidentField
from gfem.errors import NonDifferentiableError
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._engine.beam_coupling import CouplingLayer
    from gfem._field import Field
    from gfem.model import Model
    from gfem.results import DrivenResult

__all__ = ["beam_coupling"]

#: A `gfem.GaussianBeam` seen with less power than this through the side is
#: clipped by it: part of the beam misses the layer.
CLIPPED = 0.999
#: The share of the beam's power through the layer that crosses the
#: truncation where the side's surface meets it (the layer's open ends):
#: above `CLIPPING` the side clips the beam (a warning), above `CROSSING`
#: the surface runs along the truncation under the beam (an error).
CLIPPING = 1e-3
CROSSING = 0.1
#: Why a material of the layer has no gradient.
LAYER_REASON = (
    "the coupling reads the beam's medium from the element layer just outside "
    "the side, which must be a constant material: move the design one element "
    "away from the side's surface, or wrap the material in jax.lax.stop_gradient"
)


def beam_coupling(
    result: DrivenResult,
    name: str,
    beam: GaussianBeam | IncidentField,
    *,
    model: Model,
    side: str | Sequence[str] | Callable[[np.ndarray], np.ndarray] | None = None,
) -> jax.Array:
    r"""The amplitude an excitation's field couples into an incoming beam, √W.

    For a port column fed 1 W this is, by reciprocity, the wave the port
    would receive if ``beam`` lit the device: the S-parameter between the
    beam and the port, ``port_wave(port, beam)`` of the beam-lit solve,
    from the port-fed solve alone. A grating coupler's fiber coupling
    efficiency is :math:`|c|^2`, and every fiber position, angle, waist and
    polarization is read from the same solve. :math:`|c|^2` is a power on a
    real reference (a photonic wave port without a voltage path); otherwise
    :math:`c` is a pseudo-wave, like S (`gfem.conventions`, Port waves and
    beam coupling).

    With :math:`\chi` the side's P1 indicator and :math:`w` the beam
    travelling toward the side, over the element layer :math:`L` just
    outside it,

    .. math::

        c = \frac{1}{4i\omega\mu\sqrt{P}}\int_L \Big[
            E\cdot(\nabla\chi\times\nabla\times w)
            + (\nabla\times E)\cdot(\nabla\chi\times w)\Big]\,dV,
        \qquad
        P = \frac{1}{2\omega\mu}\operatorname{Im}\int_L
            \nabla\chi\cdot(w\times\overline{\nabla\times w})\,dV,

    the reaction of the far field (`gfem.far_field`) with the beam as test
    field, scaled so that the beam carries 1 W through the side. The phase
    is referred to the beam's own (a `gfem.GaussianBeam`'s zero focal
    phase). A `gfem.GaussianBeam` that shows less than 0.999 W through the
    side is clipped by it, and warns; so does any beam of which more than
    a thousandth of the power through the layer crosses the truncation where
    the side's surface meets it.

    On a result of a distributed solve it is a collective call: every rank
    calls it and gets the same coupling. An `gfem.IncidentField` beam is
    then evaluated on every rank, at that rank's points.

    Example:
        The coupling of a grating's port into a fiber 10° off the normal::

            fiber = gfem.GaussianBeam(
                "fiber",
                waist=5.2e-6,
                focus=(4e-6, 0, 1e-6),
                theta=np.pi - np.radians(10),
                polarization="TE",
            )
            c = gfem.beam_coupling(result, "wg", fiber, model=model)
            efficiency = abs(c[0]) ** 2

    Args:
        result: A `gfem.DrivenResult` that kept the column's fields
            (``Driven(..., keep_fields=...)``).
        name: The excited port, or the source, whose field couples (an
            incident-field column's scattered field).
        beam: The incoming beam, as the fiber launches it toward the device:
            a `gfem.GaussianBeam` or a `gfem.IncidentField` (its curl from
            its moment interpolant on the layer). Its own medium is the
            layer's: no background is needed.
        model: The model the result was solved on.
        side: ``None`` (the default): the physical elements touching the
            truncation face (a plane of `gfem.Absorbing` faces or of the
            perfectly matched layer's inner face) the beam brings most power
            in through. Or volume region names, or a predicate
            ``inside(centroids) -> bool`` on the element centroids (m), as
            for `power_flow`; the side holds every source and port. The
            layer drops its perfectly matched layer elements, so a plane
            through the layer ends open there.

    Returns:
        ``(n_freq,)`` complex, √W, one value per kept frequency of ``name``.

    Raises:
        TypeError: ``result``, ``beam`` or ``model`` is of the wrong type,
            or ``side`` is neither names, a predicate nor ``None``.
        KeyError: ``name`` is neither an excited port nor a source.
        RegionNotFoundError: A named region does not exist.
        ValueError: No field of ``name`` was kept; the model has no PML
            layer and no `gfem.Absorbing` face, or `gfem.Periodic` pairs;
            the side selects nothing or everything; its layer is not one
            constant, isotropic, lossless medium (the beam's), or holds a
            source, port, sheet or lumped element; it meets a boundary face
            that is neither `gfem.Absorbing` nor a PEC or PMC wall the beam
            satisfies; the beam carries no power into the side; an
            `gfem.IncidentField` beam's layer has moved from the imported
            mesh.
        NonDifferentiableError: The side is a predicate while the node
            coordinates are traced; and, while `jax.grad` traces it, the
            field's ``omega`` or a material of the layer.
    """
    from gfem.model import Model
    from gfem.results import DrivenResult

    refuse_entry(getattr(model, "mesh", None), "beam_coupling")
    if not isinstance(result, DrivenResult):
        raise TypeError(
            f"beam_coupling needs a gfem.DrivenResult, got {type(result).__name__}."
        )
    if not isinstance(model, Model):
        raise TypeError(
            f"beam_coupling needs model= a gfem.Model, got {type(model).__name__}."
        )
    if not isinstance(beam, (GaussianBeam, IncidentField)):
        raise TypeError(
            "beam_coupling's beam is the incoming beam, a gfem.GaussianBeam or a "
            f"gfem.IncidentField, got {type(beam).__name__}."
        )
    names = side_names(
        side, "beam_coupling", default="the layer at the beam's entrance"
    )
    source = result.sources.get(name)
    if source is None and name not in result.s.excited:
        columns = ", ".join(map(repr, result._excitations)) or "none"  # noqa: SLF001
        raise KeyError(
            f"{name!r} is neither an excited port nor a source of the result; its "
            f"columns are {columns}."
        )
    frequencies = [f for f, c in result.kept if c == name]
    if not frequencies:
        raise ValueError(
            f"No field of {name!r} was kept: solve with Driven(..., "
            "keep_fields=True) to read its beam coupling."
        )
    incident = isinstance(source, INCIDENT)
    part = "scattered" if incident else "total"
    fields = [result.e_field(f, name, part=part) for f in frequencies]
    if callable(names) and is_traced((fields[0].mesh.nodes, model.mesh.nodes)):
        raise NonDifferentiableError(
            "beam_coupling's side is a predicate on element centroids, which "
            "needs the node coordinates as numbers, but they are computed inside "
            "jax.grad, jax.jit or jax.vjp here: name the side's volume regions "
            "instead, so that its surface is known before the transformation runs."
        )
    out = []
    for field, frequency in zip(fields, frequencies, strict=True):
        concrete = concrete_field(field)
        nodes = np.asarray(concrete.mesh.nodes)
        layer = engine("beam_coupling").coupling_layer(
            concrete, model, names, beam, frequency, nodes=nodes, incident=incident
        )
        curved.refuse_tets(
            model.mesh,
            "beam_coupling",
            layer.tetrahedra,
            "The coupling layer of gfem.beam_coupling",
        )
        reference = None
        if isinstance(beam, IncidentField):
            reference = engine("beam_coupling").frozen_reference(
                concrete,
                model,
                layer=layer,
                beam=beam,
                frequency=frequency,
                order=field.order,
            )
        kernel, inputs = _kernel(field, model, layer, beam, reference)
        _check_power(kernel, layer, nodes, frequency, beam)
        out.append(measure(kernel, inputs)["coupling"])
    return jnp.stack(out)


def _check_power(
    kernel: _CouplingKernel,
    layer: CouplingLayer,
    nodes: np.ndarray,
    frequency: float,
    beam: GaussianBeam | IncidentField,
) -> None:
    """Raise for a beam bringing no power into the side; warn for a clipped one."""
    power = kernel.power(nodes, 2 * math.pi * frequency)
    if not power > 0:
        raise ValueError(
            f"The beam {beam.name!r} carries no power into the side ({power:.3g} "
            "W through its layer): describe it travelling toward the device, as "
            "the fiber launches it, and give a side whose surface crosses the "
            "beam once (a half-space, not a closed surface around the device, "
            "through which a beam brings no net power)."
        )
    if layer.crossing > CROSSING * power:
        raise ValueError(
            f"beam_coupling's side meets the truncation (the PML layer or an "
            f"Absorbing face) where the beam {beam.name!r} crosses it: "
            f"{layer.crossing:.3g} W of the beam crosses the truncation faces of "
            f"the layer's elements that touch the side, against {power:.3g} W "
            "through the layer. Put the side's surface at least one element away "
            "from the truncation under the beam, or leave side=None."
        )
    if isinstance(beam, GaussianBeam) and power < CLIPPED:
        _warn_clipped(
            beam, f"carries {power:.4g} W through the side's layer", frequency
        )
    elif layer.crossing > CLIPPING * power:
        _warn_clipped(
            beam,
            f"crosses the truncation with {layer.crossing / power:.3g} of its power "
            "where the side's surface meets it",
            frequency,
        )


def _warn_clipped(
    beam: GaussianBeam | IncidentField, what: str, frequency: float
) -> None:
    warnings.warn(
        f"The beam {beam.name!r} {what} at {frequency:.6g} Hz: the side "
        "clips it. The coupling is normalized to the power that crosses the "
        "layer; widen the side (about 2.2 waists on each side of the axis) to "
        "read the whole beam.",
        UserWarning,
        stacklevel=4,
    )


def _kernel(
    field: Field,
    model: Model,
    layer: CouplingLayer,
    beam: GaussianBeam | IncidentField,
    reference: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[_CouplingKernel, dict[str, Any]]:
    """The kernel of one field and its named inputs."""
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
        model, inputs, roles, lambda kind: _model_role(kind, model, layer)
    )
    kernel = _CouplingKernel(
        roles=roles,
        field_def=field_def,
        model_def=model_def,
        topology=field.mesh._topology,  # noqa: SLF001
        layer=layer,
        beam=beam,
        reference=reference,
        order=field.order,
        imported=np.asarray(model.mesh.imported_nodes, float),
        model=model,
        comm=field._communicator,  # noqa: SLF001
    )
    return kernel, inputs


def _model_role(kind: tuple[Any, ...], model: Model, layer: CouplingLayer) -> Role:
    """The role of a model leaf, by its `input_kind`."""
    if kind[0] == "nodes":
        return Role("read", None, "the mesh node coordinates (model.mesh.nodes)")
    if kind[0] == "material":
        region = kind[1]
        described = f"model.materials[{region!r}].{kind[2]}"
        mesh = model.mesh
        if np.isin(mesh.elements(region), layer.tetrahedra).any():
            return Role(
                "read",
                f"region {region!r} has elements in the element layer just "
                f"outside the side; {LAYER_REASON}",
                described,
            )
        return Role("unread", None, described)
    return Role("unread", None, "model." + ".".join(str(k) for k in kind))


class _CouplingKernel(RoleKernel):
    """The coupling of one field into one beam through one layer.

    Under ranks (``comm``) a collective host call: the coefficients are a
    rank's piece; the field at the layer's quadrature points is gathered by
    the probe, so every rank computes the same coupling, and the
    coefficients' cotangent is a rank's part of a load.
    """

    name = "gfem.beam_coupling"
    pieces: ClassVar[frozenset[str]] = frozenset({"coefficients"})
    output_specs: ClassVar[dict[str, jax.ShapeDtypeStruct]] = {
        "coupling": jax.ShapeDtypeStruct((), jnp.complex128)
    }

    def __init__(
        self,
        *,
        roles: dict[str, Role],
        field_def: tuple[Any, tuple[str, ...]],
        model_def: tuple[Any, tuple[str, ...]],
        topology: Any,
        layer: CouplingLayer,
        beam: GaussianBeam | IncidentField,
        reference: tuple[np.ndarray, np.ndarray] | None,
        order: int,
        imported: np.ndarray,
        model: Model,
        comm: Any,
    ) -> None:
        self.comm = comm
        self.roles = roles
        self.field_def = field_def
        self.model_def = model_def
        self.topology = topology
        self.layer = layer
        self.beam = beam
        self.reference = reference
        self.order = order
        self.imported = imported
        self.model = model

    # --- host -------------------------------------------------------------------

    def _probe(self, nodes: np.ndarray, field: Field) -> Any:
        from gfem._engine import far_field as ff
        from gfem._engine import probe
        from gfem._probe import layout

        bary, _ = ff.rule(self.order)
        mesh = Mesh(nodes, self.topology)
        located = ff.located(self.layer, bary)  # ty: ignore[invalid-argument-type]
        return probe.probe(mesh, layout(field), located, self.comm)

    def _test(self, omega: float) -> Callable[[Any, Any], tuple[Any, Any]]:
        bc = engine("beam_coupling")
        if self.reference is not None:
            return bc.frozen_test(*self.reference)
        return bc.gaussian_test(self.beam, self.layer, omega / (2 * math.pi))

    def _function(
        self, nodes: np.ndarray, omega: float
    ) -> tuple[Callable[..., Any], np.ndarray]:
        """``(c(corners, E, curl E), corners at the nodes)``."""
        from gfem._engine import far_field as ff

        bc = engine("beam_coupling")
        tets = np.asarray(Mesh(nodes, self.topology).tetrahedra)[self.layer.tetrahedra]
        corners0 = np.asarray(nodes, float)[tets]
        chi = self.layer.chi[tets]
        bary, weights = ff.rule(self.order)
        test = self._test(omega)
        mu_r = self.layer.mu_r

        def function(corners: Any, e: Any, curl: Any) -> Any:
            e_ref, c_ref = ff.reference_values(corners0, e, curl)
            r, flux = bc.reaction(
                corners, e_ref, c_ref, chi=chi, bary=bary, weights=weights, test=test
            )
            return bc.coupling(r, flux, omega, mu_r)[0]

        return function, corners0

    def power(self, nodes: np.ndarray, omega: float) -> float:
        """The beam's power through the side's layer, W."""
        from gfem._engine import far_field as ff

        bc = engine("beam_coupling")
        tets = np.asarray(Mesh(nodes, self.topology).tetrahedra)[self.layer.tetrahedra]
        corners = jnp.asarray(np.asarray(nodes, float)[tets])
        bary, weights = ff.rule(self.order)
        zero = jnp.zeros((len(tets), len(bary), 3), complex)
        _, flux = bc.reaction(
            corners,
            zero,
            zero,
            chi=self.layer.chi[tets],
            bary=bary,
            weights=weights,
            test=self._test(omega),
        )
        return float(bc.coupling(0.0, flux, omega, self.layer.mu_r)[1])

    def _check_frozen(self, nodes: np.ndarray) -> None:
        if self.reference is None:
            return
        n, d = engine("beam_coupling").moved_layer(self.layer, self.model, nodes)
        if n:
            raise ValueError(
                f"beam_coupling reads the IncidentField {self.beam.name!r} as "
                f"interpolated on the imported mesh, but {n} nodes of its element "
                f"layer have moved (up to {d:.3g} m), so the beam would move with "
                "them. Measure on a side whose layer boundary_morph holds (the "
                "default side, next to the truncation), or give the beam as a "
                "gfem.GaussianBeam, which is read at the moved nodes."
            )

    def _values(
        self, values: dict[str, np.ndarray]
    ) -> tuple[Field, np.ndarray, float, np.ndarray]:
        field, model = (
            rebuild(self.field_def, values),
            rebuild(self.model_def, values),
        )
        pf = importlib.import_module("gfem._engine.power_flow")
        pf.check_nodes(field, model)
        nodes = np.asarray(values["nodes"], float)
        self._check_frozen(nodes)
        omega = float(np.asarray(values["omega"]).real)
        return field, nodes, omega, np.asarray(values["coefficients"])

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        field, nodes, omega, e = self._values(values)
        prb = self._probe(nodes, field)
        ev = prb.values(e)[0]
        cv = prb.values(e, curl=True)[0]
        function, corners = self._function(nodes, omega)
        return {"coupling": np.asarray(function(corners, ev, cv))}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        field, nodes, omega, e = self._values(values)
        out = {name: np.zeros_like(values[name]) for name in wanted}
        ct = cotangents.get("coupling")
        if ct is not None and ("coefficients" in wanted or "nodes" in wanted):
            prb = self._probe(nodes, field)
            ev = prb.values(e)[0]
            cv = prb.values(e, curl=True)[0]
            function, corners = self._function(nodes, omega)
            _, pull = jax.vjp(function, corners, ev, cv)
            g_corners, g_e, g_c = (np.asarray(g) for g in pull(jnp.asarray(ct)))
            if "coefficients" in wanted:
                g = prb.transpose(g_e[None]) + prb.transpose(g_c[None], curl=True)
                out["coefficients"] = out["coefficients"] + g[0]
            if "nodes" in wanted:
                tets = np.asarray(Mesh(nodes, self.topology).tetrahedra)
                grad = np.zeros_like(nodes)
                np.add.at(grad, tets[self.layer.tetrahedra], np.real(g_corners))
                out["nodes"] = grad
        unknown = [
            n
            for n in wanted
            if n not in ("coefficients", "nodes") and not n.startswith("model.mesh")
        ]
        if unknown:  # the measure wrapper raised for the others
            raise NotImplementedError(f"no beam-coupling gradient for {unknown}")
        return out
