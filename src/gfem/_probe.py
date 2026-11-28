"""Field values at points: the host kernel behind `Field.at` and measurements.

`field_at` is `gfem.Field.at`; `evaluate` gives the values of coefficient
vectors at fixed points (gfem tetrahedra and barycentric coordinates), the
JAX entry point that measurements integrating fields on surfaces build on.
Both run `gfem._engine.probe` behind gfem's gradient wrapper
(`gfem._adjoint.measure`): differentiable in the field coefficients, while
mesh-node and point perturbations raise `gfem.NonDifferentiableError`.

Under ranks a field's coefficients are a rank's piece, and each of these
calls is collective.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem import _mpi
from gfem._adjoint.measure import measure
from gfem._checks import is_traced
from gfem.mesh import Mesh

if TYPE_CHECKING:
    from gfem._field import Field

__all__ = ["COMPONENTS", "engine", "evaluate", "field_at", "gradient_at", "layout"]

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
    #: A rank's own piece under ranks: the entry check reads its shape alone.
    pieces = frozenset({"coefficients"})

    def __init__(
        self,
        topology: Any,
        layout: Any,
        shape: tuple[int, ...],
        dtype: Any,
        located: Any = None,
        *,
        comm: Any = None,
        eulerian: bool = False,
    ) -> None:
        self.topology = topology
        self.eulerian = eulerian
        self.layout = layout
        self.located = located
        self.comm = comm
        self.output_specs = {"values": jax.ShapeDtypeStruct(shape, dtype)}

    def role(self, name: str) -> str:
        del name
        return "read"

    def unsupported(self, name: str) -> str | None:
        if name == "nodes":
            return None if self._moves_with_nodes() else NODES_REASON
        if name == "points":
            return "probe positions are fixed numbers, not differentiable inputs"
        return None

    def describe(self, name: str) -> str:
        return {
            "nodes": "the mesh node coordinates (field.mesh.nodes)",
            "points": "the probe points",
        }.get(name, f"the field {name}")

    def _moves_with_nodes(self) -> bool:
        """Whether the reading is differentiable by the nodes.

        Physical points (`Field.at`, `gradient_at`) of an H1 field on a 2D or 3D
        mesh: the reading at a fixed point changes as the element under it moves (see
        `_node_cotangent`). Fixed (element, barycentric) points and the other
        spaces have no node derivative.
        """
        return self.eulerian and self.layout.space == "h1" and self.comm is None

    def _probe(self, values: dict[str, np.ndarray]) -> Any:
        probe = engine()
        mesh = Mesh(values["nodes"], self.topology)
        located = self.located
        if located is None:
            located = probe.locate(mesh, self.layout, values["points"], self.comm)
        return probe.probe(mesh, self.layout, located, self.comm)

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {"values": self._probe(values).values(values["coefficients"])}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        if not set(wanted) <= {"coefficients", "nodes"}:  # the wrapper raised
            raise NotImplementedError(f"no field probe gradient for {wanted}")
        p = self._probe(values)
        ct = cotangents["values"]
        ct = ct.reshape(-1, p.n_points, ct.shape[-1])
        out = {}
        if "coefficients" in wanted:
            out["coefficients"] = p.transpose(ct).reshape(values["coefficients"].shape)
        if "nodes" in wanted:
            out["nodes"] = _node_cotangent(self, p, values, ct)
        return out


def _node_cotangent(
    kernel: Any, p: Any, values: dict[str, np.ndarray], ct: np.ndarray
) -> np.ndarray:
    r"""The nodes' cotangent of a reading at fixed physical points (Eulerian).

    The coefficients stay with the material, so a node velocity :math:`U_a` moves
    the field under a fixed point by the P1 velocity :math:`u = \sum_a \lambda_a
    U_a`. For the value :math:`f(x_q)` this changes it by
    :math:`-u\cdot\nabla f`; for the gradient by :math:`\delta\partial_i f =
    -\partial_i u_k\,\partial_k f - u_j\,\partial_j\partial_i f`. With a
    cotangent :math:`\bar g` the corners :math:`a` of the triangle holding the
    point get

    .. math::

        \bar X_a \mathrel{+}= -\lambda_a\,\bar g\,\nabla f \quad\text{(value)},
        \qquad
        \bar X_a \mathrel{+}= -(\bar g\cdot\nabla\lambda_a)\,\nabla f
            - \lambda_a\,H\bar g \quad\text{(gradient)}.

    ``ct`` is ``(m, n_points, components)``; the result is real, the nodes' shape.
    """
    probe = engine()
    nodes = np.asarray(values["nodes"], float)
    c = np.atleast_2d(values["coefficients"])
    mesh = Mesh(nodes, kernel.topology)
    gradients = probe.GradientProbe(mesh, kernel.layout, p.located)
    gradient = isinstance(kernel, _GradientKernel)
    corners = np.asarray(kernel.topology.volume_elements)[p.located.tetrahedra]
    lam = p.located.barycentric
    v = nodes[corners]
    dim = nodes.shape[1]
    edges = np.stack([v[:, i] - v[:, dim] for i in range(dim)], axis=-1)
    head = np.linalg.inv(edges)  # rows: the gradients of the first dim coordinates
    grad_lam = np.concatenate([head, -head.sum(axis=1, keepdims=True)], axis=1)
    bar = np.zeros((p.n_points, dim + 1, dim), dtype=complex)
    for j in range(len(c)):
        grad_f = gradients.values(c[j])[0]
        g = ct[j]
        if gradient:
            hessian = gradients.hessians(c[j])
            bar += (
                -np.einsum("pai,pi->pa", grad_lam, g)[:, :, None] * grad_f[:, None, :]
            )
            bar += -lam[:, :, None] * np.einsum("pik,pi->pk", hessian, g)[:, None, :]
        else:
            bar += -lam[:, :, None] * (g[:, 0, None] * grad_f)[:, None, :]
    out = np.zeros_like(nodes)
    np.add.at(out, corners, bar.real)
    return out


def evaluate(
    field_coefficients: jax.Array,
    mesh: Mesh,
    layout: Any,
    located: Any,
    comm: Any = None,
) -> jax.Array:
    """``(m, n_points, components)`` values of ``(m, ndof)`` coefficients.

    The points are fixed (gfem tetrahedra and barycentric coordinates); a
    JAX function of the coefficients and the node coordinates. Under ``comm`` it is
    collective, and the coefficients are a rank's piece.
    """
    c = jnp.asarray(field_coefficients)
    shape = (c.shape[0], len(located.tetrahedra), COMPONENTS[layout.space])
    kernel = _PointKernel(mesh._topology, layout, shape, c.dtype, located, comm=comm)  # noqa: SLF001
    return measure(kernel, {"coefficients": c, "nodes": mesh.nodes})["values"]


def field_at(field: Field, points: Any) -> jax.Array:
    """The values of ``field`` at ``points`` (see `gfem.Field.at`)."""
    single = np.ndim(points) == 1
    p = points if is_traced(points) else np.asarray(points, float)
    mesh = field.mesh
    dim = mesh.dim
    if dim == 2 and p.shape[-1:] == (3,):
        raise ValueError(
            "A 2D mesh lies in the (x, y) plane: pass (m, 2) points (or one "
            f"2-vector) in m, got shape {np.shape(points)}."
        )
    if p.shape[-1:] != (dim,) or p.ndim not in (1, 2):
        raise ValueError(
            f"points must be one {dim}-vector or an (n, {dim}) array in m, got "
            f"shape {np.shape(points)}."
        )
    p = p.reshape(-1, dim)
    where = layout(field)
    c = field.coefficients
    n_comp = COMPONENTS[field.space]
    comm = field._communicator  # noqa: SLF001
    located = None
    if not is_traced((mesh.nodes, p)):
        with _mpi.section(comm):
            _mpi.check_call(comm, "Field.at", {"the points": np.asarray(p)})
            located = engine().locate(mesh, where, np.asarray(p), comm)
    kernel = _PointKernel(
        mesh._topology,  # noqa: SLF001
        where,
        (p.shape[0], n_comp),
        c.dtype,
        located,
        comm=comm,
        eulerian=True,
    )
    inputs = {"coefficients": c, "nodes": mesh.nodes}
    if located is None:
        inputs["points"] = p
    values = measure(kernel, inputs)["values"]
    if n_comp == 1:
        values = values[:, 0]
    return values[0] if single else values


class _GradientKernel(_PointKernel):
    """Gradients of a scalar H1 field at points: ``G c`` and its transpose."""

    name = "a field gradient probe"

    def _probe(self, values: dict[str, np.ndarray]) -> Any:
        probe = engine()
        mesh = Mesh(values["nodes"], self.topology)
        located = self.located
        if located is None:
            located = probe.locate(mesh, self.layout, values["points"], self.comm)
        return probe.GradientProbe(mesh, self.layout, located, self.comm)


def gradient_at(field: Field, points: Any) -> jax.Array:
    """The gradient of a scalar H1 ``field`` at ``points``, ``(n, dim)`` per metre.

    A JAX function of the coefficients, with the rules of `field_at`.
    """
    p = points if is_traced(points) else np.asarray(points, float)
    mesh = field.mesh
    dim = mesh.dim
    if p.shape[-1:] != (dim,) or p.ndim not in (1, 2):
        raise ValueError(
            f"points must be one {dim}-vector or an (n, {dim}) array in m, got "
            f"shape {np.shape(points)}."
        )
    p = p.reshape(-1, dim)
    where = layout(field)
    c = field.coefficients
    comm = field._communicator  # noqa: SLF001
    located = None
    if not is_traced((mesh.nodes, p)):
        with _mpi.section(comm):
            _mpi.check_call(comm, "gradient_at", {"the points": np.asarray(p)})
            located = engine().locate(mesh, where, np.asarray(p), comm)
    kernel = _GradientKernel(
        mesh._topology,  # noqa: SLF001
        where,
        (p.shape[0], dim),
        c.dtype,
        located,
        comm=comm,
        eulerian=True,
    )
    inputs = {"coefficients": c, "nodes": mesh.nodes}
    if located is None:
        inputs["points"] = p
    return measure(kernel, inputs)["values"]
