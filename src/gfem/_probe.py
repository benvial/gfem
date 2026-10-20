"""Field values at points: the host kernel behind `Field.at` and measurements.

`field_at` is `gfem.Field.at`; `evaluate` gives the values of coefficient
vectors at fixed points (gfem tetrahedra and barycentric coordinates), the
JAX entry point that measurements integrating fields on surfaces build on.
Both run `gfem._engine.probe` behind gfem's gradient wrapper
(`gfem._adjoint.measure`): differentiable in the field coefficients, while
mesh-node and point perturbations raise `gfem.NonDifferentiableError`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._adjoint.measure import measure
from gfem._checks import is_traced
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._field import Field

__all__ = ["COMPONENTS", "engine", "evaluate", "field_at", "layout"]

NODES_REASON = (
    "moving the mesh nodes moves the elements a measurement integrates or "
    "evaluates on, and this version has no shape gradient for measurements"
)
#: Value components per field space.
COMPONENTS = {"h1": 1, "l2": 1, "hcurl": 3, "hdiv": 3}


def engine() -> Any:
    """The NGSolve point-evaluation module, imported on first use."""
    return importlib.import_module("gfem._engine.probe")


def layout(field: Field) -> Any:
    """The engine layout (support, exclusions, space, order) of ``field``."""
    return engine().Layout(
        support=tuple(field.support),
        excluded=tuple(field._excluded),  # noqa: SLF001
        space=field.space,
        order=field.order,
    )


class _PointKernel:
    """Field values at points: ``Phi c`` and its transpose."""

    name = "a field probe"

    def __init__(
        self,
        topology: Any,
        layout: Any,
        shape: tuple[int, ...],
        dtype: Any,
        located: Any = None,
    ) -> None:
        self.topology = topology
        self.layout = layout
        self.located = located
        self.output_specs = {"values": jax.ShapeDtypeStruct(shape, dtype)}

    def role(self, name: str) -> str:
        del name
        return "read"

    def unsupported(self, name: str) -> str | None:
        if name == "nodes":
            return NODES_REASON
        if name == "points":
            return "probe positions are fixed numbers, not differentiable inputs"
        return None

    def describe(self, name: str) -> str:
        return {
            "nodes": "the mesh node coordinates (field.mesh.nodes)",
            "points": "the probe points",
        }.get(name, f"the field {name}")

    def _probe(self, values: dict[str, np.ndarray]) -> Any:
        probe = engine()
        mesh = Mesh(values["nodes"], self.topology)
        located = self.located
        if located is None:
            located = probe.locate(mesh, self.layout, values["points"])
        return probe.probe(mesh, self.layout, located)

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {"values": self._probe(values).values(values["coefficients"])}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        if wanted != ("coefficients",):  # the measure wrapper raised for others
            raise NotImplementedError(f"no field probe gradient for {wanted}")
        p = self._probe(values)
        ct = cotangents["values"]
        g = p.transpose(ct.reshape(-1, p.n_points, ct.shape[-1]))
        return {"coefficients": g.reshape(values["coefficients"].shape)}


def evaluate(
    field_coefficients: jax.Array,
    mesh: Mesh,
    layout: Any,
    located: Any,
) -> jax.Array:
    """``(m, n_points, components)`` values of ``(m, ndof)`` coefficients.

    The points are fixed (gfem tetrahedra and barycentric coordinates); a
    JAX function of the coefficients and the node coordinates.
    """
    c = jnp.asarray(field_coefficients)
    shape = (c.shape[0], len(located.tetrahedra), COMPONENTS[layout.space])
    kernel = _PointKernel(mesh._topology, layout, shape, c.dtype, located)  # noqa: SLF001
    return measure(kernel, {"coefficients": c, "nodes": mesh.nodes})["values"]


def field_at(field: Field, points: Any) -> jax.Array:
    """The values of ``field`` at ``points`` (see `gfem.Field.at`)."""
    single = np.ndim(points) == 1
    p = points if is_traced(points) else np.asarray(points, float)
    if p.shape[-1:] != (3,) or p.ndim not in (1, 2):
        raise ValueError(
            f"points must be one 3-vector or an (n, 3) array in m, got shape "
            f"{np.shape(points)}."
        )
    p = p.reshape(-1, 3)
    where = layout(field)
    mesh = field.mesh
    c = field.coefficients
    n_comp = COMPONENTS[field.space]
    located = None
    if not is_traced((mesh.nodes, p)):
        located = engine().locate(mesh, where, np.asarray(p))
    kernel = _PointKernel(
        mesh._topology,  # noqa: SLF001
        where,
        (p.shape[0], n_comp),
        c.dtype,
        located,
    )
    inputs = {"coefficients": c, "nodes": mesh.nodes}
    if located is None:
        inputs["points"] = p
    values = measure(kernel, inputs)["values"]
    if n_comp == 1:
        values = values[:, 0]
    return values[0] if single else values
