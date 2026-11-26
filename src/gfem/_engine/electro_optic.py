r"""The static field and the mode weights of an electro-optic shift.

`StaticFieldKernel` reads the electrostatic field
:math:`E_\mathrm{s} = -\nabla V` of a potential at fixed physical points of
the optical model's tetrahedra (a quadrature rule of each), located in the
potential's mesh, and gives each tetrahedron's quadrature mean:

.. math::

    E_{\mathrm{s},t} = -\sum_q w_q \nabla V(x_q), \qquad \sum_q w_q = 1

It is linear in the potential's coefficients, ``E = G c``, and runs behind
gfem's gradient wrapper (`gfem._adjoint.measure`) with inputs
``coefficients`` and ``nodes``. The points stay where they are when the
potential's mesh nodes move, so a node derivative is the field's change at
fixed points (Eulerian). On a straight tetrahedron a node velocity
:math:`U_a` moves the point's reference coordinates under it by the P1
velocity :math:`u = \sum_a \lambda_a U_a`, and at fixed coefficients

.. math::

    \delta E_i = -\sum_k \partial_i u_k\,E_k - \sum_j u_j\,\partial_j E_i

The first term is the element's deformation, the second the field's
transport past the point. For a cotangent :math:`\bar g_q` of
:math:`E(x_q)` the backward pass is

.. math::

    \bar c = G^\mathsf{T} \bar g, \qquad
    \bar X_a \mathrel{+}= -(\bar g_q \cdot \nabla\lambda_a)\,E_q
        + \lambda_a(x_q)\,H_q\,\bar g_q

over the corners :math:`a` of the tetrahedron holding :math:`x_q`, with
:math:`H_q = \nabla\nabla V(x_q)` from the engine's Hessian of the H1
space (zero at order 1).

`ElectroOpticWeightsRule` is the port mode solve of the shift: the real
symmetric 3x3 weight per section triangle, its refusals naming the
electro-optic shift.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np

from gfem._engine import probe as _probe
from gfem._engine.thermo_optic import (
    ShiftWeightsRule,
    section_complex_tensor_weights,
    section_ez_boundary_weights,
)
from gfem._sweep.port_mode import PortModeRule
from gfem.mesh import Mesh
from gfem.model import Model

__all__ = [
    "AcoustoOpticWeightsRule",
    "ElectroOpticWeightsRule",
    "GradientProbe",
    "OptomechanicalPairWeightsRule",
    "StaticFieldKernel",
]


class ElectroOpticWeightsRule(ShiftWeightsRule):
    """The port mode solve of an electro-optic shift, and its 3x3 section weights."""

    def unsupported_input(self, kind: tuple) -> str | None:
        """Every input the port mode problem reads: the mode is held constant."""
        del kind
        return (
            "an electro-optic shift holds the port mode constant (first order in "
            "the static field), and the mode's derivative with respect to the optical "
            "model's permittivity, losses, permeability, node coordinates or "
            "boundary values is not wired through the shift's weights in this "
            "version (a frozen mode gives a wrong value). "
            "Differentiate through the potential or the electro_optic_tensor"
        )

    def nondifferentiable_output(self, name: str, kinds: Any = ()) -> str:
        """The error message for a cotangent on output ``name``."""
        del kinds
        return (
            f"gfem cannot differentiate the electro-optic shift's mode output "
            f"{name!r} with respect to the optical model: the mode is a constant "
            "of the shift. Wrap the input in jax.lax.stop_gradient to use its "
            "value without its derivative."
        )


class AcoustoOpticWeightsRule(ElectroOpticWeightsRule):
    """The port mode solve of an acousto-optic shift, with its own refusals."""

    def unsupported_input(self, kind: tuple) -> str | None:
        """Every input the port mode problem reads: the mode is held constant."""
        del kind
        return (
            "an acousto-optic shift holds the port mode constant (first order in "
            "the acoustic fields), and the mode's derivative with respect to the "
            "optical model's permittivity, losses, permeability, node coordinates "
            "or boundary values is not wired through the shift's weights in this "
            "version (a frozen mode gives a wrong value). "
            "Differentiate through the acoustic result, the photoelastic_tensor, "
            "the electro_optic_tensor or the offset"
        )

    def nondifferentiable_output(self, name: str, kinds: Any = ()) -> str:
        """The error message for a cotangent on output ``name``."""
        del kinds
        return (
            f"gfem cannot differentiate the acousto-optic shift's mode output "
            f"{name!r} with respect to the optical model: the mode is a constant "
            "of the shift. Wrap the input in jax.lax.stop_gradient to use its "
            "value without its derivative."
        )


class OptomechanicalPairWeightsRule(AcoustoOpticWeightsRule):
    r"""The port mode solve of a backward optomechanical pair.

    Outputs ``gamma`` (``(1,)``), the complex Hermitian section weight
    ``weights`` (``(n_tri, 3, 3)``, :math:`H_t`, global frame, in the order of
    ``faces``) and ``ez`` (``(n_vertices, 2)``, 1/m), the node weights of the
    normal field's moving-boundary correction, indexed like the unique nodes
    of the section's mesh triangles.
    """

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """The propagation constant, the complex weights and the correction."""
        n_nodes = np.unique(np.asarray(model.mesh.triangles)[self.faces]).size
        return {
            "gamma": jax.ShapeDtypeStruct((1,), jnp.complex128),
            "weights": jax.ShapeDtypeStruct((len(self.faces), 3, 3), jnp.complex128),
            "ez": jax.ShapeDtypeStruct((n_nodes, 2), jnp.float64),
        }

    def forward(
        self, model: Model, *, keep: bool, request: Any = None
    ) -> tuple[dict, None]:
        """Solve the mode and integrate its weights and the normal-field correction."""
        outputs, _ = PortModeRule.forward(self, model, keep=keep, request=request)
        sweep = outputs["sweep"]
        coefficients = sweep.basis @ sweep.coords[0]
        weights = section_complex_tensor_weights(sweep.space, coefficients)
        ez = section_ez_boundary_weights(sweep.space, coefficients)
        section = sweep.space.section
        faces = np.asarray(section.faces)
        order = np.argsort(faces)
        where = order[np.searchsorted(faces, self.faces, sorter=order)]
        if not np.array_equal(faces[where], self.faces):  # pragma: no cover
            raise RuntimeError("the port section's triangles changed")
        nodes = np.unique(np.asarray(model.mesh.triangles)[self.faces])
        mesh_xyz = np.asarray(model.mesh.nodes, float)[nodes]
        at = section.to_global(section.points)
        # the plane's points are (u, v) coordinates; their global position must be
        # the mesh nodes' for the correction to be indexed like them
        tol = 1e-9 * section.diameter * section.scale
        if len(at) != len(nodes) or np.abs(at - mesh_xyz).max() > tol:
            raise RuntimeError(
                "the port section's vertices are not the mesh nodes of its "
                "triangles in the order the correction vector is indexed by"
            )
        return {"gamma": sweep.gamma, "weights": weights[where], "ez": ez}, None


class GradientProbe(_probe.Probe):
    r"""The gradient :math:`\nabla V` of a scalar H1 layout at fixed points.

    `values` gives ``(m, n_points, 3)`` gradients in V/m and `transpose`
    its transpose, as `gfem._engine.probe.Probe` does for values;
    `hessians` gives :math:`\nabla\nabla V`.
    """

    def __init__(self, mesh: Mesh, layout: Any, located: Any, comm: Any = None) -> None:
        """Build the engine space and the points (the mesh's node positions)."""
        super().__init__(mesh, layout, located, comm)
        self.located = located
        # d/dx of the engine's scaled coordinates
        self.factor = 1.0 / self._built.scale

    def hessians(self, coefficients: np.ndarray) -> np.ndarray:
        """The Hessian of one real field at the points, ``(n_points, 3, 3)``, V/m^2."""
        if self._mips is None:
            return np.zeros((0, 3, 3))
        gf = ngs.GridFunction(self._fes)
        with self._lock:
            gf.vec.FV().NumPy()[:] = np.asarray(coefficients)[: self.ndof]
            out = np.asarray(gf.Operator("hesse")(self._mips))
        return out.reshape(self._own.size, 3, 3) * self.factor**2

    def _evaluate(
        self, fes: ngs.FESpace, values: np.ndarray, *, curl: bool = False
    ) -> np.ndarray:
        """Gradients at this rank's points, ``(its points, 3)``.

        The potential's gradient has no curl to read, so ``curl`` must stay off.
        """
        if curl:
            raise ValueError("The static potential's gradient probe reads no curl.")
        if self._mips is None:
            return np.zeros((0, 3))
        gf = ngs.GridFunction(fes)
        gf.vec.FV().NumPy()[:] = values
        out = np.asarray(ngs.grad(gf)(self._mips))
        return out.reshape(self._own.size, -1)


class StaticFieldKernel:
    r"""The mean of :math:`-\nabla V` over each optical tetrahedron, :math:`G c`.

    Its adjoint is :math:`G^T`.

    The points are fixed physical points; at given node coordinates of the
    potential's mesh they are located in it (or taken as given, ``located``,
    for the nodes they were located at).
    """

    name = "the electro-optic shift's static field"
    pieces = frozenset({"coefficients"})
    comm = None

    def __init__(
        self,
        topology: Any,
        layout: Any,
        points: np.ndarray,
        weights: np.ndarray,
        *,
        located: Any = None,
        nodes: np.ndarray | None = None,
    ) -> None:
        """The points and their rule.

        Args:
            topology: The potential's mesh topology.
            layout: The potential's `gfem._engine.probe.Layout`.
            points: ``(n * q, 3)`` points, m, element-major.
            weights: ``(q,)`` rule weights summing to 1.
            located: The points located at ``nodes``, or ``None``.
            nodes: The node coordinates ``located`` holds for.
        """
        self.topology = topology
        self.layout = layout
        self.points = np.asarray(points, float)
        self.weights = np.asarray(weights, float)
        self.n = len(self.points) // len(self.weights)
        self._located = located
        self._nodes = None if nodes is None else np.asarray(nodes)
        self._probes: dict[int, GradientProbe] = {}
        self.output_specs = {"field": jax.ShapeDtypeStruct((self.n, 3), jnp.float64)}

    def role(self, name: str) -> str:
        del name
        return "read"

    def unsupported(self, name: str) -> str | None:
        del name
        return None

    def describe(self, name: str) -> str:
        return {
            "nodes": "the electrostatic mesh node coordinates (potential.mesh.nodes)",
        }.get(name, f"the potential {name}")

    def _probe(self, nodes: np.ndarray) -> GradientProbe:
        key = _probe._digest(nodes)  # noqa: SLF001
        found = self._probes.get(key)
        if found is None:
            mesh = Mesh(nodes, self.topology)
            located = self._located
            held = self._nodes is not None and np.array_equal(nodes, self._nodes)
            if located is None or not held:
                located = _probe.locate(mesh, self.layout, self.points)
            found = self._probes[key] = GradientProbe(mesh, self.layout, located)
        return found

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        grad = self._probe(values["nodes"]).values(values["coefficients"])[0]
        grad = grad.reshape(self.n, len(self.weights), 3)
        return {"field": -np.einsum("q,nqd->nd", self.weights, grad)}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        p = self._probe(values["nodes"])
        ct = np.asarray(cotangents["field"]).reshape(self.n, 3)
        # the cotangent of E at each point
        g = np.einsum("q,nd->nqd", self.weights, ct).reshape(-1, 3)
        out = {}
        if "coefficients" in wanted:
            c = p.transpose(-g[None])
            out["coefficients"] = c.reshape(values["coefficients"].shape)
        if "nodes" in wanted:
            out["nodes"] = self._node_cotangent(p, values, g)
        return out

    def _node_cotangent(
        self, p: GradientProbe, values: dict[str, np.ndarray], g: np.ndarray
    ) -> np.ndarray:
        """The nodes' cotangent at fixed points and coefficients (Eulerian)."""
        nodes = np.asarray(values["nodes"], float)
        c = values["coefficients"]
        corners = np.asarray(self.topology.volume_elements)[p.located.tetrahedra]
        lam = p.located.barycentric
        v = nodes[corners]
        edges = np.stack([v[:, i] - v[:, 3] for i in range(3)], axis=-1)
        # rows: the gradients of the first three barycentric coordinates
        head = np.linalg.inv(edges)
        grad_lam = np.concatenate([head, -head.sum(axis=1, keepdims=True)], axis=1)
        field = -p.values(c)[0]
        hessian = p.hessians(c)
        bar = -np.einsum("pai,pi->pa", grad_lam, g)[:, :, None] * field[:, None, :]
        bar += lam[:, :, None] * np.einsum("pij,pj->pi", hessian, g)[:, None, :]
        out = np.zeros_like(nodes)
        np.add.at(out, corners, bar)
        return out
