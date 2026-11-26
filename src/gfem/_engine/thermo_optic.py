r"""The first-order weights of a thermo-optic shift on a wave port's section.

`ShiftWeightsRule` runs the port's mode solve (`gfem.PortMode`) and turns
its mode at 1 W into one weight per triangle of the section:

.. math::

    w_t = \frac{c\varepsilon_0}{4}\, 2 n_{0,t} \int_t |E_0|^2\, dA,

so that :math:`\Delta n_\mathrm{eff} = \sum_t w_t\, (dn/dT)_t\, \Delta T_t`.
:math:`n_{0,t}` is the square root of the real permittivity of the
tetrahedron behind triangle :math:`t`, as the port mode problem reads it.
With ``tensor`` it gives instead the real symmetric 3x3 weight per
triangle, in the global frame,

.. math::

    W_t = \frac{c\varepsilon_0}{4}\, \mathrm{Re} \int_t E_0 E_0^H\, dA,

whose trace is :math:`w_t / (2 n_{0,t})`, for a tensor perturbation
:math:`\Delta n_\mathrm{eff} = \sum_t W_t : \Delta\varepsilon_t`. The
weights are constants of the mode: no output has a gradient, and every
input the port mode problem reads (the optical model's ``eps_r``,
``tan_delta``, ``mu_r``, ``conductivity``, node coordinates, boundary and
port values) raises `gfem.NonDifferentiableError` under `jax.grad`; the
thermal properties and the thermo-optic coefficient are not read, so they
get exact zeros here and differentiate where the shift multiplies them in
JAX.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem._engine.solve_request import SolveRequest
from gfem._sweep.port_mode import PortModeRule
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import PortMode

__all__ = ["ShiftWeightsRule"]


class ShiftWeightsRule(PortModeRule):
    """The port mode solve of a thermo-optic shift, and its section weights.

    Outputs ``gamma`` (``(1,)``, the mode's propagation constant, 1/m) and
    ``weights`` (``(n_tri,)``, or ``(n_tri, 3, 3)`` with ``tensor``,
    dimensionless, in the order of ``faces``).
    """

    def __init__(
        self,
        model: Model,
        problem: PortMode,
        options: SolverOptions,
        faces: np.ndarray,
        *,
        tensor: bool = False,
    ) -> None:
        """The port mode rule, and the section's mesh triangles in order.

        Args:
            model: The optical model.
            problem: The port mode problem at one frequency.
            options: Solver options (the element order).
            faces: The gfem triangles of the port's section, the order of
                ``weights``.
            tensor: The 3x3 weight per triangle instead of the scalar one.
        """
        super().__init__(model, problem, options)
        self.faces = np.asarray(faces)
        self.tensor = tensor

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """The propagation constant and one weight per section triangle."""
        del model
        shape = (len(self.faces), 3, 3) if self.tensor else (len(self.faces),)
        return {
            "gamma": jax.ShapeDtypeStruct((1,), jnp.complex128),
            "weights": jax.ShapeDtypeStruct(shape, jnp.float64),
        }

    #: The shift's outputs are not the port mode rule's: every cotangent raises.
    input_aware = False
    differentiable_outputs = frozenset()

    def unsupported_input(self, kind: tuple) -> str | None:
        """Every input the port mode problem reads: the mode is held constant."""
        del kind
        return (
            "a thermo-optic shift holds the port mode constant (first order in "
            "the rise), and the mode's derivative with respect to the optical "
            "model's permittivity, losses, permeability, node coordinates or "
            "boundary values is not wired through the shift's weights in this "
            "version (a frozen mode gives a wrong value). Differentiate "
            "through the rise or the thermo_optic_coefficient"
        )

    def nondifferentiable_output(self, name: str, kinds: Any = ()) -> str:
        """The error message for a cotangent on output ``name``."""
        del kinds
        return (
            f"gfem cannot differentiate the thermo-optic shift's mode output "
            f"{name!r} with respect to the optical model: the mode is a constant "
            "of the shift. Wrap the input in jax.lax.stop_gradient to use its "
            "value without its derivative."
        )

    def forward(
        self, model: Model, *, keep: bool, request: SolveRequest | None = None
    ) -> tuple[dict, None]:
        """Solve the mode and integrate its energy on each section triangle."""
        outputs, _ = super().forward(model, keep=keep, request=request)
        sweep = outputs["sweep"]
        weigh = section_tensor_weights if self.tensor else section_weights
        weights = weigh(sweep.space, sweep.basis @ sweep.coords[0])
        faces = np.asarray(sweep.space.section.faces)
        order = np.argsort(faces)
        where = order[np.searchsorted(faces, self.faces, sorter=order)]
        if not np.array_equal(faces[where], self.faces):  # pragma: no cover
            raise RuntimeError("the port section's triangles changed")
        return {"gamma": sweep.gamma, "weights": weights[where]}, None


def section_weights(space: Any, coefficients: np.ndarray) -> np.ndarray:
    """``(c eps0 / 4) 2 n0_t int_t |E|^2 dA`` per section triangle.

    Args:
        space: The `gfem._engine.port_mode.PortModeSpace` of the section.
        coefficients: The mode's full coefficient vector of ``(E_t, E_n)``,
            V/m at 1 W.

    Returns:
        ``(n_tri,)`` in the section's triangle order (the 2D mesh's element
        order).
    """
    gf = ngs.GridFunction(space.space)
    gf.vec.FV().NumPy()[:] = coefficients
    et, en = gf.components
    density = ngs.Norm(et) ** 2 + ngs.Norm(en) ** 2
    energy = ngs.Integrate(
        density, space.mesh, order=2 * space.order, element_wise=True
    )
    section = space.section
    energy = np.asarray(energy).real * section.scale**2
    n0 = np.sqrt(np.asarray(section.eps_nn).real)
    c_eps0 = scipy.constants.c * scipy.constants.epsilon_0
    return c_eps0 / 4 * 2 * n0 * energy


def section_tensor_weights(space: Any, coefficients: np.ndarray) -> np.ndarray:
    r"""The tensor weight per section triangle, in the global frame.

    It is :math:`\tfrac{c\varepsilon_0}{4}\mathrm{Re}\int_t E E^H\,dA`.

    Real symmetric: the imaginary part is antisymmetric, so it meets nothing
    in a symmetric permittivity change. Its trace is ``section_weights``
    over :math:`2 n_{0,t}`.

    Args:
        space: The `gfem._engine.port_mode.PortModeSpace` of the section.
        coefficients: The mode's full coefficient vector of ``(E_t, E_n)``,
            V/m at 1 W.

    Returns:
        ``(n_tri, 3, 3)`` in the section's triangle order (the 2D mesh's
        element order).
    """
    gf = ngs.GridFunction(space.space)
    gf.vec.FV().NumPy()[:] = coefficients
    et, en = gf.components
    local = [et[0], et[1], en]
    section = space.section
    w = np.empty((len(section.triangles), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            product = ngs.Integrate(
                local[i] * ngs.Conj(local[j]),
                space.mesh,
                order=2 * space.order,
                element_wise=True,
            )
            w[:, i, j] = w[:, j, i] = np.asarray(product).real
    c_eps0 = scipy.constants.c * scipy.constants.epsilon_0
    w *= c_eps0 / 4 * section.scale**2
    frame = np.asarray(section.frame, float)
    w = np.einsum("ai,tab,bj->tij", frame, w, frame)
    return (w + np.swapaxes(w, 1, 2)) / 2


def section_complex_tensor_weights(space: Any, coefficients: np.ndarray) -> np.ndarray:
    r"""The complex tensor weight per section triangle, in the global frame.

    It is :math:`H_t = \tfrac{c\varepsilon_0}{4}\int_t E E^H\,dA`: Hermitian,
    not symmetrized. The pair of a mode with its counter-propagating twin
    reads its blocks separately (the real part of the in-plane block and the
    imaginary part of the cross block), so the imaginary part is kept. Its
    real part is `section_tensor_weights`.

    Args:
        space: The `gfem._engine.port_mode.PortModeSpace` of the section.
        coefficients: The mode's full coefficient vector of ``(E_t, E_n)``,
            V/m at 1 W.

    Returns:
        ``(n_tri, 3, 3)`` complex, in the section's triangle order.
    """
    gf = ngs.GridFunction(space.space)
    gf.vec.FV().NumPy()[:] = coefficients
    et, en = gf.components
    local = [et[0], et[1], en]
    section = space.section
    w = np.empty((len(section.triangles), 3, 3), complex)
    for i in range(3):
        for j in range(3):
            w[:, i, j] = np.asarray(
                ngs.Integrate(
                    local[i] * ngs.Conj(local[j]),
                    space.mesh,
                    order=2 * space.order,
                    element_wise=True,
                )
            )
    c_eps0 = scipy.constants.c * scipy.constants.epsilon_0
    w *= c_eps0 / 4 * section.scale**2
    frame = np.asarray(section.frame, float)
    return np.einsum("ai,tab,bj->tij", frame, w, frame)


def section_ez_boundary_weights(space: Any, coefficients: np.ndarray) -> np.ndarray:
    r"""The node weights of the normal field's moving-boundary correction.

    For a P1 vector field :math:`v` on the section (global :math:`x, y`), the
    linear form

    .. math::

        v \mapsto \frac{c\varepsilon_0}{4}\sum_t \varepsilon_{nn,t}
        \int_t \nabla\cdot\left(|E_n|^2 v\right)\,dA

    has, on the hat function of node :math:`i` along :math:`x` or :math:`y`,
    the weight ``c[i]``. By the divergence theorem the elements of one region
    cancel, and what is left is the flux of :math:`\varepsilon_{nn}|E_n|^2` of
    each region through its boundary: for the discrete fields exactly
    :math:`\tfrac{c\varepsilon_0}{4}\oint u_n\,\Delta\varepsilon_{nn}
    |E_n|^2\,dl` over the interfaces (zero over a wall that does not move).
    The permittivity along the normal enters piece by piece, so a jump of
    :math:`\varepsilon_{nn}` at an interface is weighed by its own size.

    Args:
        space: The `gfem._engine.port_mode.PortModeSpace` of the section.
        coefficients: The mode's full coefficient vector of ``(E_t, E_n)``,
            V/m at 1 W.

    Returns:
        ``(n_vertices, 2)`` in 1/m, rows in the order of the section's
        points, columns the global :math:`x, y` components.
    """
    section = space.section
    mesh = space.mesh
    gf = ngs.GridFunction(space.space)
    gf.vec.FV().NumPy()[:] = coefficients
    _, en = gf.components
    l2 = ngs.L2(mesh, order=0)
    eps = ngs.GridFunction(l2)
    eps.vec.FV().NumPy()[:] = np.asarray(section.eps_nn).real
    h1 = ngs.H1(mesh, order=1)
    v = h1.TestFunction()
    density = (en * ngs.Conj(en)).real
    grad_density = 2 * (ngs.Conj(en) * ngs.grad(en)).real
    frame = np.asarray(section.frame, float)
    c_eps0 = scipy.constants.c * scipy.constants.epsilon_0
    out = np.empty((len(section.points), 2))
    for a in range(2):
        # the global direction e_a in the local (u, v) components
        d = ngs.CF((float(frame[0, a]), float(frame[1, a])))
        lf = ngs.LinearForm(h1)
        lf += (
            eps
            * (
                ngs.InnerProduct(d, grad_density) * v
                + density * ngs.InnerProduct(d, ngs.grad(v))
            )
            * ngs.dx(bonus_intorder=1)
        )
        lf.Assemble()
        out[:, a] = lf.vec.FV().NumPy()[: len(section.points)]
    # the local gradient is per local unit, the area in local units squared
    return out * (c_eps0 / 4 * section.scale)
