r"""Piezoelectric rule of a 2D mesh: displacement and potential in one H1 space.

For fields that do not vary along :math:`z`, the unknowns are the three
displacement components :math:`u_x, u_y, u_z` on the mechanical domain and the
potential :math:`\varphi` on the electromagnetic one, all complex H1 of one
order, in one compound space. With the stress-charge relations
:math:`T = \hat c S - e^T E`, :math:`D = e S + \varepsilon^S E` and
:math:`E = -\nabla\varphi`, the symmetric weak form (charge rows negated) is

.. math::

    a(x, y) = \int S(v)\cdot(\hat c\,S(u) + e^T\nabla\varphi)
    + \nabla\psi\cdot(e\,S(u) - \varepsilon^S\nabla\varphi)
    - \omega^2\rho\,v\cdot u,

with :math:`\hat c = c\,(1 + i(1/Q_m + \eta\,r^2))`, the damping frame's
:math:`r` (`gfem.DampingFrame`), under :math:`e^{+i\omega t}`. The operator is
complex symmetric and affine in :math:`\omega^2`: :math:`K` and :math:`M` are
assembled once and :math:`A = K - \omega^2 M` is formed per frequency, one
sparse Cholesky factorization each, shared by every terminal.

Terminal :math:`j` is the lifting :math:`w_j` (the potential 1 on its
electrode DOFs, interpolated: a hierarchical basis has zero higher-order
values there) corrected to :math:`x_j = w_j + (\text{free part})` with
:math:`(A x_j)_f = 0`. The charge per length on terminal :math:`i` is the
reaction weighted by its lifting, :math:`q'_{ij} = -w_i^T A x_j`, and
:math:`Y'_{ij} = i\omega q'_{ij}`; it equals the quadratic form
:math:`-x_i^T A x_j`. The clamped capacitance is the quadratic form of the
potential alone with :math:`\varepsilon^S` and no displacement.

**Gradients.** Every terminal state is the solution of a system whose free rows
vanish, so :math:`\delta Y'_{ij} = -i\omega\,x_i^T\,\delta A\,x_j` needs no
adjoint solve, and neither does :math:`C'_0`; a cotangent on a kept field takes
one solve :math:`A\lambda = g` per frequency that carries one, on that
frequency's factorization. The backward hands the kernel **field pairs** whose
weights are those of the five pieces of :math:`A` (stiffness 1, loss
:math:`i`, coupling 1, dielectric 1, mass :math:`-\omega^2`), and the material
leaves come from the **tensor pass**: per-element lab-frame moments of the
strain, the potential gradient and the displacement
(`gfem._engine.sensitivity.MomentBlock`), contracted into the crystal frame
through the transpose of the rotation of each region's `axes` (the rotation is
linear in :math:`c` and :math:`e`). The loss factor enters as
:math:`1/Q_m + \eta\,r^2`, so its moments are the strain moments weighted by
the frame's :math:`\eta r^2`.

Lengths are the mesh file's unit (``scale`` metres): the strain and the
potential gradient carry :math:`1/\text{scale}` and the area
:math:`\text{scale}^2`, so the stiffness and permittivity forms are the
file-unit ones and only the mass term carries :math:`\text{scale}^2`. The
displacement is in m per V. On a 3D mesh (mechanical modes only) the volume
is :math:`\text{scale}^3`: every form carries one more :math:`\text{scale}`,
the mass term :math:`\text{scale}^3`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable
from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem._boundaries import PEC, Absorbing
from gfem._engine import quadrature
from gfem._engine.cache import Entry, discretization, retain_budget
from gfem._engine.coefficients import element_axes, p0, permittivity
from gfem._engine.domain import boundary_region, em_region, h1_space, region_domain
from gfem._engine.runtime import serial_spaces, task_manager
from gfem._engine.sensitivity import FieldPair, MomentBlock, Tensors, Term
from gfem._engine.solve_request import SolveRequest
from gfem._field import FieldMap
from gfem._linalg.base import free_mask
from gfem._linalg.direct import DirectHandle
from gfem._materials import piezoelectric_in_frame, stiffness_in_frame
from gfem.errors import MissingMaterialPropertyError
from gfem.mesh import Mesh
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Piezoelectric
from gfem.results import Field, PiezoelectricResult

__all__ = ["PiezoelectricRule", "PiezoelectricSystem"]

EPS0 = scipy.constants.epsilon_0


class PiezoelectricRule:
    """Host-side forward of a `Piezoelectric` problem on a 2D `Model`.

    Construction runs every static check (it needs no array values);
    `forward` assembles the system and solves each frequency.
    """

    physics = "em"
    #: The operator family (a key of `sensitivity.PROPERTIES`).
    form = "piezoelectric"
    #: Material properties the solve reads.
    reads = frozenset(
        {
            "stiffness",
            "piezoelectric_tensor",
            "eps_r",
            "mass_density",
            "mechanical_quality_factor",
        }
    )
    #: Outputs whose cotangents the backward pass handles.
    differentiable_outputs = frozenset(
        {"admittance", "capacitance", "displacement", "potential"}
    )

    def __init__(
        self,
        model: Model,
        problem: Piezoelectric,
        options: SolverOptions,
        *,
        comm: Any = None,
    ) -> None:
        """Check that the problem fits the model (see `gfem.Piezoelectric`).

        Raises:
            NotImplementedError: ``linear_solver="petsc"``, or
                ``frequency_workers`` above 1.
            RegionNotFoundError: A name is not a region of the mesh.
            ValueError: The problem does not fit the model; the message gives
                the remedy.
            MissingMaterialPropertyError: A region of the mechanical domain
                has no stiffness or density.
        """
        if not isinstance(problem, Piezoelectric):
            raise TypeError(f"Expected gfem.Piezoelectric, got {problem!r}.")
        self.problem = problem
        self.options = options
        self.comm = comm
        if options.linear_solver == "petsc":
            raise NotImplementedError(
                "A Piezoelectric solve has no PETSc path in this version: use "
                'linear_solver="direct" or "auto".'
            )
        if options.frequency_workers > 1:
            raise NotImplementedError(
                "A Piezoelectric solve runs its frequencies in one process, one "
                "factorization each: leave frequency_workers at 1 (threads= sets "
                "the cores)."
            )
        self.mechanical = _check_mechanical(model)
        self.electrodes = _check_electrodes(model, problem)
        _check_clamped(model, problem)
        _check_absorbing(model)
        _check_materials(model)
        self.frame = _check_frame(model, problem)
        mesh = model.mesh
        mechanical = model._domain_of("mechanical")  # noqa: SLF001
        domain = model._domain  # noqa: SLF001
        #: The areas with elements in each domain, mesh order.
        self.mechanical_regions = tuple(
            r for r in mesh.volumes if mechanical.active[mesh.elements(r)].any()
        )
        self.potential_regions = tuple(
            r for r in mesh.volumes if domain.active[mesh.elements(r)].any()
        )
        self._mesh = mesh
        #: The kept ``(frequency, terminal)`` labels, in output order.
        self.kept = tuple(
            (f, name)
            for f in problem.frequency
            if f in problem.kept_frequencies
            for name in problem.terminal_names
        )
        self._mechanical_excluded = mechanical.excluded
        self._em_excluded = domain.excluded

    def forward(
        self,
        model: Model,
        *,
        keep: bool,
        request: SolveRequest | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], Entry | None]:
        """Solve every frequency and terminal.

        Returns:
            ``{"admittance": (n_f, n_t, n_t) S/m, symmetrized,
            "admittance_raw": the same before the symmetrization,
            "capacitance": (n_t, n_t) F/m, "displacement": (n_kept, 3,
            n_dof) m/V and "potential": (n_kept, n_dof) V/V, the kept
            columns in `kept` order}``, and with ``keep`` the entry a backward
            reads: the terminals' state at each frequency and the potential
            of :math:`C'_0`.
        """
        problem = self.problem
        with task_manager(self.options.threads, self.comm):
            system = PiezoelectricSystem(model, problem, self.options, self)
            states: list[np.ndarray] | None = [] if keep else None
            # the factorizations the backward's field cotangents solve on, within
            # half the retained bytes (the rest is the entry's fields)
            factors = KeptFactors(retain_budget(self.options.retain_memory) // 2)
            raw, kept = system.admittance(
                problem.frequency,
                problem.kept_frequencies,
                states,
                factors if keep else None,
            )
            capacitance, potential = system.clamped_capacitance()
        admittance = 0.5 * (raw + np.swapaxes(raw, 1, 2))
        nu = system.offset // 3
        displacement = np.zeros((len(self.kept), 3, nu if self.kept else 0), complex)
        phi = np.zeros(
            (len(self.kept), system.ndof - system.offset if self.kept else 0), complex
        )
        for i, label in enumerate(self.kept):
            displacement[i] = [kept[label][axis] for axis in "xyz"]
            phi[i] = kept[label]["phi"]
        outputs = {
            "admittance": admittance,
            "admittance_raw": raw,
            "capacitance": 0.5 * (capacitance + capacitance.T),
            "displacement": displacement,
            "potential": phi,
        }
        if states is None:
            return outputs, None
        fields = {f"x{k}": x for k, x in enumerate(states)}
        fields["phi"] = potential
        return outputs, Entry("em", handle=factors or None, fields=fields)

    def result(self, outputs: dict[str, Any]) -> PiezoelectricResult:
        """The public result from the forward outputs."""
        problem, order = self.problem, self.options.order
        displacement, potential = [], []
        mesh = self._mesh
        mechanical = self._mechanical_excluded
        for i, (f, _) in enumerate(self.kept):
            omega = jnp.asarray(2 * np.pi * f)
            displacement.append(
                FieldMap(
                    {
                        axis: Field(
                            coefficients=outputs["displacement"][i][a],
                            mesh=mesh,
                            quantity=f"u{axis}",
                            unit="m",
                            space="h1",
                            order=order,
                            support=self.mechanical_regions,
                            omega=omega,
                            _excluded=mechanical,
                        )
                        for a, axis in enumerate("xyz")
                    },
                    label="component",
                )
            )
            potential.append(
                Field(
                    coefficients=outputs["potential"][i],
                    mesh=mesh,
                    quantity="V",
                    unit="V",
                    space="h1",
                    order=order,
                    support=self.potential_regions,
                    omega=omega,
                    _excluded=self._em_excluded,
                )
            )
        return PiezoelectricResult(
            admittance_per_length=jnp.asarray(outputs["admittance"]),
            clamped_capacitance_per_length=jnp.asarray(outputs["capacitance"]),
            terminals=problem.terminal_names,
            frequencies=problem.frequency,
            displacement_fields=displacement,
            potential_fields=potential,
            kept=self.kept,
        )

    # --- the solve_vjp interface ---------------------------------------------------

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output (from the mesh topology)."""
        problem = self.problem
        nf, nt = len(problem.frequency), len(problem.terminals)
        nu = nphi = 0
        if self.kept:
            # the sizes depend on the topology only: traced nodes are not read
            mesh = model.mesh
            fixed = model.replace(mesh=Mesh(mesh.imported_nodes, mesh._topology))  # noqa: SLF001
            with task_manager(self.options.threads, self.comm):
                system = PiezoelectricSystem(
                    fixed, problem, self.options, self, assemble=False
                )
            nu, nphi = system.offset // 3, system.ndof - system.offset
        nk = len(self.kept)
        return {
            "admittance": jax.ShapeDtypeStruct((nf, nt, nt), jnp.complex128),
            "capacitance": jax.ShapeDtypeStruct((nt, nt), jnp.float64),
            "displacement": jax.ShapeDtypeStruct((nk, 3, nu), jnp.complex128),
            "potential": jax.ShapeDtypeStruct((nk, nphi), jnp.complex128),
        }

    def role(self, kind: tuple) -> str:
        """``"read"`` if the solve depends on input ``kind``.

        The five material properties, the permittivity law's parameters (which
        `unsupported_input` refuses) and the node coordinates are read; every
        other property, boundary and port value is not, so its gradient is an
        exact zero.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        if kind[0] in ("boundary", "port"):
            return "unread"
        return "read"

    def unsupported_input(self, kind: tuple) -> str | None:
        """Why input ``kind`` cannot be differentiated, or ``None``."""
        if kind[0] == "law":
            return (
                "a permittivity law's parameters are not design inputs of a "
                "Piezoelectric solve, which reads the law's static permittivity"
            )
        return None

    def nondifferentiable_output(self, name: str) -> str:  # pragma: no cover
        """The error message for a cotangent on output ``name``."""
        return f"gfem cannot differentiate the Piezoelectric output {name!r}."

    def backward(
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        wanted: frozenset | None = None,  # noqa: ARG002
    ) -> list[FieldPair]:
        r"""Fold the output cotangents into field pairs, per frequency and for C0.

        With :math:`x_j` the terminal states (lifting included) and
        :math:`G = (W + W^T)/2` for the admittance cotangent :math:`W` at a
        frequency, the pair is left :math:`X`, right :math:`X(-i\omega G)^T`: no
        solve. A cotangent :math:`g` on kept fields adds the pair
        :math:`(-\lambda, X)` with :math:`A\lambda = g` on the free DOFs.
        The capacitance cotangent gives the pair :math:`(\Phi, \Phi G^T)` on
        the dielectric piece alone, with weight :math:`-1` (the form is
        :math:`-K_\varphi` in :math:`A`), the potentials embedded in the
        compound space.
        """
        problem, options = self.problem, self.options
        nt = len(problem.terminals)
        pairs: list[FieldPair] = []
        with task_manager(options.threads, self.comm):
            system = PiezoelectricSystem(model, problem, options, self, assemble=False)
            ndof, offset = system.ndof, system.offset
            factors = entry.handle
            ct_y = cotangents.get("admittance")
            loads = self._kept_loads(cotangents, ndof, offset)
            for k, f in enumerate(problem.frequency):
                omega = 2 * np.pi * f
                left, right = [], []
                x = entry.fields[f"x{k}"]
                if ct_y is not None and np.any(ct_y[k]):
                    ct = np.asarray(ct_y[k], dtype=complex)
                    g = 0.5 * (ct + ct.T)
                    left.append(x)
                    right.append(x @ (-1j * omega * g).T)
                if k in loads:
                    lam = system.adjoint(
                        f, loads[k], None if factors is None else factors.get(k)
                    )
                    left.append(-lam)
                    right.append(x)
                if left:
                    pairs.append(
                        FieldPair(
                            space=system.space,
                            left=np.concatenate(left, axis=1),
                            right=np.concatenate(right, axis=1),
                            weights=_weights(omega),
                            order=options.order,
                            domain=system.diel,
                        )
                    )
            ct_c = cotangents.get("capacitance")
            if ct_c is not None and np.any(ct_c):
                ct = np.asarray(ct_c, dtype=float)
                phi = np.zeros((ndof, nt), complex)
                phi[offset:] = entry.fields["phi"]
                pairs.append(
                    FieldPair(
                        space=system.space,
                        left=phi,
                        right=phi @ (0.5 * (ct + ct.T)).T,
                        weights={"dielectric": -1.0},
                        order=options.order,
                        domain=system.diel,
                    )
                )
        return pairs

    def _kept_loads(
        self, cotangents: dict[str, np.ndarray], ndof: int, offset: int
    ) -> dict[int, np.ndarray]:
        """The cotangents on kept fields as loads ``(ndof, n_t)`` by frequency index."""
        ct_u, ct_p = cotangents.get("displacement"), cotangents.get("potential")
        names = self.problem.terminal_names
        frequencies = self.problem.frequency
        loads: dict[int, np.ndarray] = {}
        for i, (f, name) in enumerate(self.kept):
            g = np.zeros(ndof, complex)
            if ct_u is not None:
                g[:offset] = np.asarray(ct_u[i]).reshape(-1)
            if ct_p is not None:
                g[offset:] = ct_p[i]
            if not g.any():
                continue
            k = frequencies.index(f)
            load = loads.setdefault(k, np.zeros((ndof, len(names)), complex))
            load[:, names.index(name)] = g
        return loads

    def terms(self, model: Model, *, nodes: bool) -> list[Term]:  # noqa: ARG002
        r"""The five pieces of :math:`A` for the node pass, as Lagrangian terms.

        Each term is the integral of its real-coefficient form on the pinned
        rule, so ``DiffShape`` at the integral level carries the measure, the
        gradients and the position-dependent loss (read from the imported
        coordinates, which ride with the material).
        """
        return _terms(self, model)

    def tensors(self, model: Model, wanted: frozenset) -> Tensors:
        """The tensor pass of the wanted material leaves (see `Tensors`)."""
        return _tensors(self, model, wanted)


def _weights(omega: float) -> dict[str, complex]:
    """The weights of the five pieces of :math:`A` at angular frequency ``omega``."""
    return {
        "stiffness": 1.0,
        "loss": 1j,
        "coupling": 1.0,
        "dielectric": 1.0,
        "mass": -(omega**2),
    }


def _terms(rule: PiezoelectricRule, model: Model) -> list[Term]:
    """The node pass's terms: stiffness, loss, coupling, dielectric and mass."""
    problem, order = rule.problem, rule.options.order
    built = discretization(model.mesh).built
    mech = em_region(model, built, "mechanical")
    diel = em_region(model, built, "em")
    vu = h1_space(model, built, order, tuple(problem.clamped), physics="mechanical")
    vp = h1_space(model, built, order, rule.electrodes.lines, physics="em")
    with serial_spaces():
        space = ngs.FESpace([vu, vu, vu, vp])
    forms = _weak_forms(
        _coefficients(model, built, rule.frame), built, order, mech, diel
    )

    def term(name: str) -> Term:
        form, dx = forms[name]

        def integrand(u: Any, v: Any) -> Any:
            return form(_parts(u), _parts(v)) * dx

        return Term(name, "surface", integrand=integrand, space=space)

    return [term(name) for name in forms]


def _parts(u: Any) -> tuple[Any, Any, Any, Any]:
    """The four components of a compound field ``u`` of the system's space."""
    return tuple(u.components[i] for i in range(4))


def _permittivity(model: Model, built: Any) -> ngs.CoefficientFunction:
    r""":math:`\varepsilon_0\varepsilon_r` as a 3x3 coefficient."""
    eps = permittivity(model, built)
    if eps.dim == 1:
        eps = eps * ngs.Id(3)
    return EPS0 * eps


def _plane(g: Any) -> ngs.CoefficientFunction:
    """A 2D mesh's gradient with its out-of-plane component zero; a 3D one whole."""
    return g if g.dim == 3 else ngs.CF((g[0], g[1], 0))


def _coefficients(model: Model, built: Any, frame: Any) -> dict[str, Any]:
    r"""The coefficients of the weak forms, shared by the system and the node pass.

    ``eps`` (F/m), ``c`` and ``e`` (lab frame), ``rho`` and ``loss``: the
    :math:`1/Q_m` plus, in a damping frame, :math:`\eta r^2`.
    """
    arrays = _element_arrays(model)
    loss = p0(built, arrays["loss"])
    if frame is not None:
        loss = loss + _frame_depth(model, built, frame)
    return {
        "eps": _permittivity(model, built),
        "c": _tensor(built, arrays["c"]),
        "e": _tensor(built, arrays["e"]),
        "rho": p0(built, arrays["rho"]),
        "loss": loss,
    }


def _weak_forms(
    co: dict[str, Any],
    built: Any,
    order: int,
    mech: Any,
    diel: Any,
    *,
    wavenumber_pieces: bool = False,
) -> dict[str, tuple[Callable[[Any, Any], Any], Any]]:
    r"""The pieces of :math:`A`: ``(integrand, measure)`` by name.

    The integrand takes ``(trial, test)`` 4-tuples ``(u_x, u_y, u_z, phi)``.
    The one place the forms are written: the system assembles them and the
    node pass differentiates them.

    With ``wavenumber_pieces`` the forms of a field :math:`e^{-iqz}` follow,
    for the stiffness, the loss, the coupling and the dielectric: with
    :math:`S_z(u)` the strain :math:`\partial_z u` per unit :math:`\partial_z`
    (:math:`u_z` in zz, :math:`u_y` in yz, :math:`u_x` in xz), the trial's
    :math:`\partial_z` is :math:`-iq` and the test's :math:`+iq`, so that the form is
    ``<piece>`` :math:`-iq` ``<piece>1`` :math:`+ q^2` ``<piece>2`` with the
    three real forms ``<piece>`` (the in-plane strains alone),
    ``<piece>1`` :math:`= S_\perp(v)\cdot c\,S_z(u) - S_z(v)\cdot
    c\,S_\perp(u)` and ``<piece>2`` :math:`= S_z(v)\cdot c\,S_z(u)`. The
    wavenumber in a weight is that of the mesh's length unit
    (:math:`q\,\text{scale}`). The coupling and the dielectric split the same
    way, with :math:`g_z(\varphi) = \varphi` for the potential in the place of
    :math:`S_z(u)` (``coupling1``, ``coupling2``, ``dielectric1``,
    ``dielectric2``); they read the potential, so they are for 4-tuples only.

    On a 3D mesh the strain has its six components and the gradient its three,
    and every form carries one more factor ``scale``: the volume is
    :math:`\text{scale}^3` where the strains (:math:`1/\text{scale}^2`) leave
    :math:`\text{scale}`, so the stiffness is in J and the mass term carries
    :math:`\text{scale}^3`. The wavenumber pieces are for 2D meshes.
    """
    c, e, eps, rho, loss = co["c"], co["e"], co["eps"], co["rho"], co["loss"]
    scale = built.scale
    dx_mech = quadrature.dx(order, definedon=mech)
    dx_diel = quadrature.dx(order, definedon=diel)

    def strain(u: Any) -> Any:
        return _strain(u[0], u[1], u[2])

    def gradient(u: Any) -> Any:
        return _plane(ngs.grad(u[3]))

    def stiffness(u: Any, v: Any) -> Any:
        return strain(v) * (c * strain(u))

    def damping(u: Any, v: Any) -> Any:
        return loss * (strain(v) * (c * strain(u)))

    def coupling(u: Any, v: Any) -> Any:
        return strain(v) * (e.trans * gradient(u)) + gradient(v) * (e * strain(u))

    def dielectric(u: Any, v: Any) -> Any:
        return -(gradient(v) * (eps * gradient(u)))

    def mass(u: Any, v: Any) -> Any:
        return scale**2 * rho * (u[0] * v[0] + u[1] * v[1] + u[2] * v[2])

    forms = {
        "stiffness": (stiffness, dx_mech),
        "loss": (damping, dx_mech),
        "coupling": (coupling, dx_mech),
        "dielectric": (dielectric, dx_diel),
        "mass": (mass, dx_mech),
    }
    if built.mesh.dim == 3:
        forms = {k: (_per_length(f, scale), dx) for k, (f, dx) in forms.items()}
    if not wavenumber_pieces:
        return forms

    def axial(u: Any) -> Any:
        return ngs.CF((0, 0, u[2], u[1], u[0], 0))

    def cross(u: Any, v: Any) -> Any:
        return strain(v) * (c * axial(u)) - axial(v) * (c * strain(u))

    def along(u: Any, v: Any) -> Any:
        return axial(v) * (c * axial(u))

    forms |= {
        "stiffness1": (cross, dx_mech),
        "stiffness2": (along, dx_mech),
        "loss1": (lambda u, v: loss * cross(u, v), dx_mech),
        "loss2": (lambda u, v: loss * along(u, v), dx_mech),
    }

    def lifted(u: Any) -> Any:
        return ngs.CF((0, 0, u[3]))

    def coupling_cross(u: Any, v: Any) -> Any:
        return (
            strain(v) * (e.trans * lifted(u))
            + gradient(v) * (e * axial(u))
            - axial(v) * (e.trans * gradient(u))
            - lifted(v) * (e * strain(u))
        )

    def coupling_along(u: Any, v: Any) -> Any:
        return axial(v) * (e.trans * lifted(u)) + lifted(v) * (e * axial(u))

    def dielectric_cross(u: Any, v: Any) -> Any:
        return -(gradient(v) * (eps * lifted(u)) - lifted(v) * (eps * gradient(u)))

    def dielectric_along(u: Any, v: Any) -> Any:
        return -(lifted(v) * (eps * lifted(u)))

    forms |= {
        "coupling1": (coupling_cross, dx_mech),
        "coupling2": (coupling_along, dx_mech),
        "dielectric1": (dielectric_cross, dx_diel),
        "dielectric2": (dielectric_along, dx_diel),
    }
    return forms


def _per_length(form: Callable[[Any, Any], Any], scale: float):
    """``form`` times ``scale``: a 3D mesh's volume carries one more length."""
    return lambda u, v: scale * form(u, v)


# --- the tensor pass ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Layout:
    """The entries of the symmetric moments, by which strains and gradients vary.

    Attributes:
        active: The strains (Voigt order) that are not zero.
        upper: The entries ``I <= J`` of a symmetric moment of the strains.
        coupled: The entries ``(a, I)`` of the coupling moment: gradient
            component, strain.
        plane: The entries ``a <= b`` of the symmetric gradient moment.
        dim: The number of gradient components.
    """

    active: tuple[int, ...]
    upper: tuple[tuple[int, int], ...]
    coupled: tuple[tuple[int, int], ...]
    plane: tuple[tuple[int, int], ...]
    dim: int

    @classmethod
    def of(cls, active: tuple[int, ...], dim: int) -> _Layout:
        """The layout of ``active`` strains and ``dim`` gradient components."""
        return cls(
            active,
            tuple((i, j) for a, i in enumerate(active) for j in active[a:]),
            tuple((a, i) for a in range(dim) for i in active),
            tuple((a, b) for a in range(dim) for b in range(a, dim)),
            dim,
        )


#: Nothing varies along z: the strains that are not zero (Voigt order), and
#: the in-plane gradient.
_PLANAR = _Layout.of((0, 1, 3, 4, 5), 2)
#: A field :math:`e^{-iqz}`: every strain and gradient component.
_AXIAL = _Layout.of((0, 1, 2, 3, 4, 5), 3)


def _strains(u: Any, shift: complex = 0.0) -> ngs.CoefficientFunction:
    r"""The strain of a compound field ``u`` of the system's space.

    With a ``shift`` the strain of a field with :math:`\partial_z =` ``shift``.
    """
    ux, uy, uz = (u.components[i] for i in range(3))
    strain = _strain(ux, uy, uz)
    if shift == 0:
        return strain
    return strain + shift * ngs.CF((0, 0, uz, uy, ux, 0))


def _gradient(u: Any, shift: complex = 0.0) -> ngs.CoefficientFunction:
    r"""The potential's gradient of a compound field, :math:`\partial_z =` ``shift``."""
    g = ngs.grad(u.components[3])
    if shift == 0:
        return g if g.dim == 3 else ngs.CF((g[0], g[1]))
    return ngs.CF((g[0], g[1], shift * u.components[3]))


def _shifts(wavenumber: float) -> tuple[complex, complex]:
    r"""The :math:`\partial_z` of the left (test) and right (trial) field.

    A field :math:`e^{-iqz}` has :math:`-iq` as trial and :math:`+iq` as test.
    """
    return 1j * wavenumber, -1j * wavenumber


def _strain_moment(
    weight: ngs.CoefficientFunction | None, layout: _Layout = _PLANAR, q: float = 0.0
) -> Any:
    r"""The moment :math:`\frac12 (S_I(l) S_J(r) + S_J(l) S_I(r))`, :math:`I \le J`.

    With a wavenumber ``q`` the left field is the test (:math:`+iq`) and the
    right one the trial (:math:`-iq`).
    """
    tl, tr = _shifts(q)

    def entries(left: Any, right: Any) -> ngs.CoefficientFunction:
        sl, sr = _strains(left, tl), _strains(right, tr)
        out = ngs.CF(
            tuple(0.5 * (sl[i] * sr[j] + sl[j] * sr[i]) for i, j in layout.upper)
        )
        return out if weight is None else weight * out

    return entries


def _coupling_moment(layout: _Layout = _PLANAR, q: float = 0.0) -> Any:
    r""":math:`g_a(l) S_I(r) + g_a(r) S_I(l)`, :math:`g` the potential's gradient.

    The left field is the test and the right one the trial (see `_strain_moment`).
    """
    tl, tr = _shifts(q)

    def entries(left: Any, right: Any) -> ngs.CoefficientFunction:
        gl, gr = _gradient(left, tl), _gradient(right, tr)
        sl, sr = _strains(left, tl), _strains(right, tr)
        return ngs.CF(tuple(gl[a] * sr[i] + gr[a] * sl[i] for a, i in layout.coupled))

    return entries


def _gradient_moment(layout: _Layout = _PLANAR, q: float = 0.0) -> Any:
    r""":math:`\frac12 (g_a(l) g_b(r) + g_b(l) g_a(r))`, :math:`a \le b`."""
    tl, tr = _shifts(q)

    def entries(left: Any, right: Any) -> ngs.CoefficientFunction:
        gl, gr = _gradient(left, tl), _gradient(right, tr)
        return ngs.CF(
            tuple(0.5 * (gl[a] * gr[b] + gl[b] * gr[a]) for a, b in layout.plane)
        )

    return entries


def _displacement_moment(left: Any, right: Any) -> ngs.CoefficientFunction:
    r""":math:`u(l) \cdot u(r)`."""
    return sum(
        (left.components[i] * right.components[i] for i in range(1, 3)),
        left.components[0] * right.components[0],
    )


def _symmetric(values: np.ndarray, pairs: tuple[tuple[int, int], ...], n: int):
    """Entries ``(e, k)`` of a symmetric moment as ``(e, n, n)`` matrices."""
    out = np.zeros((len(values), n, n), dtype=values.dtype)
    for k, (i, j) in enumerate(pairs):
        out[:, i, j] = out[:, j, i] = values[:, k]
    return out


def _rotation_rows(shape: tuple[int, int], axes: Any, rotate: Any) -> np.ndarray:
    r"""The transpose of the rotation into the lab frame, as a matrix.

    The rotation of a tensor of ``shape`` from the frame of ``axes`` to the lab
    frame is linear: :math:`\mathrm{vec}(\mathrm{rotate}(t)) = L\,\mathrm{vec}(t)`.
    This is :math:`L^T`, so a cotangent of the lab-frame tensor maps to the
    crystal frame as :math:`L^T g`.
    """
    n = shape[0] * shape[1]
    unit = np.eye(n).reshape(n, *shape)
    return np.asarray(rotate(unit, axes, None)).reshape(n, n)


def _tensors(
    rule: Any,
    model: Model,
    wanted: frozenset,
    *,
    wavenumber: float = 0.0,
    coupled: bool = True,
) -> Tensors:
    r"""The moment blocks and the contraction of the wanted material leaves.

    With a ``wavenumber`` :math:`q` (in the mesh's length unit) the moments are
    those of a field :math:`e^{-iqz}`, every strain and gradient component
    dressed with :math:`\partial_z = \mp iq` (``coupled`` False: no potential,
    so no coupling or dielectric moment).
    """
    layout = _AXIAL if model.mesh.dim == 3 or wavenumber != 0 else _PLANAR
    kinds = [k for k in wanted if k[0] == "material"]
    props = {k[2] for k in kinds}
    built = discretization(model.mesh).built
    mech = em_region(model, built, "mechanical")
    diel = em_region(model, built, "em")
    blocks = []
    if props & {"stiffness", "mechanical_quality_factor"}:
        blocks.append(
            MomentBlock(
                "SS",
                len(layout.upper),
                _strain_moment(None, layout, wavenumber),
                mech,
                ("stiffness", "loss"),
            )
        )
    if "stiffness" in props and rule.frame is not None:
        eta = _frame_depth(model, built, rule.frame)
        blocks.append(
            MomentBlock(
                "SSeta",
                len(layout.upper),
                _strain_moment(eta, layout, wavenumber),
                mech,
                ("loss",),
            )
        )
    if "piezoelectric_tensor" in props and coupled:
        blocks.append(
            MomentBlock(
                "ES",
                len(layout.coupled),
                _coupling_moment(layout, wavenumber),
                mech,
                ("coupling",),
            )
        )
    if "eps_r" in props and coupled:
        blocks.append(
            MomentBlock(
                "EE",
                len(layout.plane),
                _gradient_moment(layout, wavenumber),
                diel,
                ("dielectric",),
            )
        )
    if "mass_density" in props:
        blocks.append(MomentBlock("uu", 1, _displacement_moment, mech, ("mass",)))
    return Tensors(
        blocks, _Contraction(model, kinds, built.scale, layout, model.mesh.dim)
    )


class _Contraction:
    r"""Moments of one pair to the cotangent of each wanted material leaf.

    With the pair's weights :math:`\theta` (`_weights`), :math:`M` its moments
    and :math:`L` the rotation of a region's tensor into the lab frame:

    - :math:`\bar c_\mathrm{lab} = \theta_s M_{SS} + \theta_l (Q_m^{-1}
      M_{SS} + M_{S\eta})`, :math:`\bar c = L^T \bar c_\mathrm{lab}`;
    - :math:`\bar e_\mathrm{lab} = \theta_c M_{ES}`, :math:`\bar e = L^T
      \bar e_\mathrm{lab}`;
    - :math:`\bar\varepsilon_\mathrm{lab} = -\theta_d \varepsilon_0 M_{EE}`, read
      along each principal axis or as its trace;
    - :math:`\bar\rho = \theta_m\,\text{scale}^2 M_{uu}` (:math:`\text{scale}^3` on a
      3D mesh, where every other leaf carries one more :math:`\text{scale}`);
    - :math:`\bar Q_m = -\theta_l Q_m^{-2}\, c_\mathrm{lab} : M_{SS}`.
    """

    def __init__(
        self,
        model: Model,
        kinds: list[tuple],
        scale: float,
        layout: _Layout = _PLANAR,
        dim: int = 2,
    ) -> None:
        self.model = model
        self.layout = layout
        self.kinds = kinds
        self.scale = scale
        self.dim = dim
        #: The factor of every form but the mass: one more length in 3D.
        self.volume = scale if dim == 3 else 1.0
        self.arrays = _element_arrays(model)
        self._by_property = {
            "stiffness": self._stiffness,
            "mechanical_quality_factor": self._quality,
            "piezoelectric_tensor": self._piezoelectric,
            "eps_r": self._permittivity,
            "mass_density": self._density,
        }

    def __call__(self, weights, moments) -> dict[Hashable, np.ndarray]:
        out = {}
        for kind in self.kinds:
            _, region, prop = kind
            index = self.model.mesh.elements(region)
            material = self.model.materials[region]
            out[kind] = self._by_property[prop](weights, moments, index, material)
        return out

    def _strain_moment(self, moments, index) -> np.ndarray | None:
        if "SS" not in moments:
            return None
        return _symmetric(moments["SS"][index], self.layout.upper, 6)

    def _stiffness(self, weights, moments, index, material) -> np.ndarray:
        n = len(index)
        lab = np.zeros((n, 6, 6), complex)
        ss = self._strain_moment(moments, index)
        loss = weights.get("loss", 0)
        if ss is not None:
            factor = weights.get("stiffness", 0) + loss * self.arrays["loss"][index]
            lab += factor[:, None, None] * ss
        if "SSeta" in moments:
            lab += loss * _symmetric(moments["SSeta"][index], self.layout.upper, 6)
        rows = _rotation_rows((6, 6), material.axes, stiffness_in_frame)
        return self.volume * (lab.reshape(n, 36) @ rows.T).reshape(n, 6, 6)

    def _quality(self, weights, moments, index, material) -> np.ndarray:  # noqa: ARG002
        ss = self._strain_moment(moments, index)
        if ss is None:
            return np.zeros(len(index), complex)
        loss = self.arrays["loss"][index]
        along = np.einsum("eij,eij->e", self.arrays["c"][index], ss)
        return -weights.get("loss", 0) * self.volume * loss**2 * along

    def _piezoelectric(self, weights, moments, index, material) -> np.ndarray:
        n = len(index)
        lab = np.zeros((n, 3, 6), complex)
        if "ES" in moments:
            for k, (a, i) in enumerate(self.layout.coupled):
                lab[:, a, i] = weights.get("coupling", 0) * moments["ES"][index, k]
        rows = _rotation_rows((3, 6), material.axes, piezoelectric_in_frame)
        return self.volume * (lab.reshape(n, 18) @ rows.T).reshape(n, 3, 6)

    def _permittivity(self, weights, moments, index, material) -> np.ndarray:
        lab = np.zeros((len(index), 3, 3), complex)
        if "EE" in moments:
            dim = self.layout.dim
            ee = _symmetric(moments["EE"][index], self.layout.plane, dim)
            lab[:, :dim, :dim] = -weights.get("dielectric", 0) * self.volume * EPS0 * ee
        if material.has_principal_values("eps_r"):
            axes = element_axes(self.model)[index]
            return np.real(np.einsum("eki,eij,ekj->ek", axes, lab, axes))
        return np.real(np.trace(lab, axis1=1, axis2=2))

    def _density(self, weights, moments, index, material) -> np.ndarray:  # noqa: ARG002
        if "uu" not in moments:
            return np.zeros(len(index), complex)
        return weights.get("mass", 0) * self.scale**self.dim * moments["uu"][index, 0]


# --- checks ------------------------------------------------------------------------


def _check_mechanical(model: Model, owner: str = "Piezoelectric") -> Any:
    """The mechanical domain; raise when it is empty.

    Raises:
        ValueError: Every volume is excluded from the mechanical domain.
    """
    domain = model._domain_of("mechanical")  # noqa: SLF001
    if not domain.active.any():
        raise ValueError(
            "Every volume is excluded from mechanical problems, so the "
            f"{owner} problem has nothing to solve on. Exclude fewer "
            'volumes for mechanical problems (excluded={"mechanical": ...}).'
        )
    return domain


def _words(model: Model) -> tuple[str, str]:
    """What the mesh calls its boundaries and its cells: line and area in 2D."""
    return ("line", "area") if model.mesh.dim == 2 else ("surface", "volume")


def _lines_and_areas(model: Model, names: tuple[str, ...], what: str):
    """Split ``names`` into lines and areas of the 2D mesh, or surfaces and volumes.

    Raises:
        RegionNotFoundError: A name is not a region of the mesh.
        ValueError: An area outside the electromagnetic domain, or a line that
            does not border it.
    """
    mesh, domain = model.mesh, model._domain  # noqa: SLF001
    line, area = _words(model)
    lines, areas = [], []
    for name in names:
        region = mesh.region(name)
        if region.dim == mesh.dim:
            if not domain.active[mesh.elements(name)].any():
                raise ValueError(
                    f"The {what} {name!r} is {_article(area)} {area} outside the "
                    'electromagnetic domain (excluded from "em"): a meshed '
                    f"electrode needs its potential. Name its shell {line}s, or "
                    "keep it in the electromagnetic domain."
                )
            areas.append(name)
        else:
            if not domain.touches(name):
                raise ValueError(
                    f"The {what} {line} {name!r} borders no {area} of the "
                    "electromagnetic domain, so it has no potential to fix."
                )
            lines.append(name)
    return tuple(lines), tuple(areas)


class Electrodes:
    """The terminals' and the ground's lines and areas.

    Attributes:
        terminals: ``(name, lines, areas)`` per terminal, matrix order.
        ground_lines: The grounded lines (the ground, and every `gfem.PEC`
            line that is not a terminal).
        ground_areas: The grounded areas.
    """

    def __init__(self, terminals, ground_lines, ground_areas) -> None:
        self.terminals = terminals
        self.ground_lines = ground_lines
        self.ground_areas = ground_areas

    @property
    def lines(self) -> tuple[str, ...]:
        """Every electrode line: the Dirichlet lines of the potential."""
        out = [n for _, lines, _ in self.terminals for n in lines]
        return tuple(dict.fromkeys([*out, *self.ground_lines]))


def _check_electrodes(model: Model, problem: Any) -> Electrodes:
    """The electrodes the problem names, checked against the model.

    ``problem`` is a `Piezoelectric` or a `MechanicalEigenmode` (``ground`` only).
    """
    terminals = []
    named = set()
    for name, surfaces in getattr(problem, "terminals", ()):
        lines, areas = _lines_and_areas(model, surfaces, "terminal")
        terminals.append((name, lines, areas))
        named.update(surfaces)
    lines, areas = _lines_and_areas(model, problem.ground, "ground")
    pec = tuple(
        b.surface
        for b in model.boundaries
        if isinstance(b, PEC)
        and b.surface not in named
        and b.surface not in problem.ground
        and model._domain.touches(b.surface)  # noqa: SLF001
    )
    return Electrodes(tuple(terminals), tuple(dict.fromkeys([*lines, *pec])), areas)


def _article(word: str) -> str:
    """``a`` or ``an`` before ``word``."""
    return "an" if word[0] in "aeiou" else "a"


def _check_clamped(model: Model, problem: Any) -> None:
    """Raise for a clamped line (surface in 3D) that does not border the solid.

    ``problem.clamped`` holds names, or ``(name, components)`` pairs.
    """
    mechanical = model._domain_of("mechanical")  # noqa: SLF001
    line = _words(model)[0]
    for item in problem.clamped:
        name = item if isinstance(item, str) else item[0]
        region = model.mesh.region(name)
        if region.dim == model.mesh.dim or not mechanical.touches(name):
            raise ValueError(
                f"clamped {name!r} must be {_article(line)} {line} bordering the "
                "mechanical domain, where the displacement is fixed."
            )


def _check_absorbing(model: Model, owner: str = "Piezoelectric") -> None:
    """Raise for an `Absorbing` line that borders the mechanical domain."""
    mechanical = model._domain_of("mechanical")  # noqa: SLF001
    for b in model.boundaries:
        if isinstance(b, Absorbing) and mechanical.touches(b.surface):
            raise ValueError(
                f"gfem.Absorbing on {b.surface!r} absorbs electromagnetic waves; "
                f"a {owner} problem leaves that line traction-free. Absorb "
                "acoustic waves with damping_frame=gfem.DampingFrame(...)."
            )


def _check_materials(model: Model) -> None:
    """Raise for a mechanical region without stiffness or density.

    A piezoelectric region outside the electromagnetic domain raises too.

    Raises:
        MissingMaterialPropertyError: A region lacks ``stiffness`` or
            ``mass_density``.
        ValueError: A piezoelectric region is excluded from ``"em"``.
    """
    mesh = model.mesh
    mechanical = model._domain_of("mechanical").active  # noqa: SLF001
    domain = model._domain.active  # noqa: SLF001
    missing = []
    for region in mesh.volumes:
        elements = mesh.elements(region)
        active = elements[mechanical[elements]]
        if not active.size:
            continue
        material = model.materials.get(region)
        missing.extend(
            (region, prop)
            for prop in ("stiffness", "mass_density")
            if material is None or getattr(material, prop) is None
        )
        if (
            material is not None
            and getattr(material, "piezoelectric_tensor", None) is not None
            and not domain[active].all()
        ):
            raise ValueError(
                f'Region {region!r} is piezoelectric but excluded from "em": the '
                "potential that drives it has no domain there. Keep it in the "
                "electromagnetic domain."
            )
    if missing:
        raise MissingMaterialPropertyError(missing, physics="mechanical")


def _check_frame(model: Model, problem: Any) -> Any:
    """The damping frame's regions checked; ``None`` without a frame.

    Raises:
        TypeError: A frame region is not an area.
        ValueError: A frame region is outside the mechanical domain, or no
            mechanical element lies outside the frame.
    """
    frame = problem.damping_frame
    if frame is None:
        return None
    mesh = model.mesh
    line, area = _words(model)
    mechanical = model._domain_of("mechanical").active  # noqa: SLF001
    inside = mechanical.copy()
    for name in frame.regions:
        region = mesh.region(name)
        if region.dim != mesh.dim:
            raise TypeError(
                f"DampingFrame region {name!r} is {_article(line)} {line}: name "
                f"{area}s of the mesh."
            )
        elements = mesh.elements(name)
        if not mechanical[elements].any():
            raise ValueError(
                f"DampingFrame region {name!r} is outside the mechanical domain: "
                "a frame damps the stiffness, so it must be a solid."
            )
        inside[elements] = False
    if not inside.any():
        raise ValueError(
            "DampingFrame names every mechanical region: nothing is left inside "
            "the frame, so it has no interior box to measure its depth from."
        )
    return frame


# --- material arrays ---------------------------------------------------------------


def _element_arrays(model: Model) -> dict[str, np.ndarray]:
    """Lab-frame stiffness, piezoelectric tensor, density and loss per element.

    Zero outside the regions that give them: ``c`` ``(n, 6, 6)`` Pa, ``e``
    ``(n, 3, 6)`` C/m^2, ``rho`` ``(n,)`` kg/m^3, ``loss`` ``(n,)`` the
    :math:`1/Q_m` (0: lossless).
    """
    mesh = model.mesh
    n = len(mesh.volume_elements)
    c = np.zeros((n, 6, 6))
    e = np.zeros((n, 3, 6))
    rho = np.zeros(n)
    loss = np.zeros(n)
    for region, material in model.materials.items():
        if material.stiffness is None:
            continue
        index = mesh.elements(region)
        shape = (len(index),)
        axes = material.axes
        c[index] = stiffness_in_frame(
            np.broadcast_to(np.asarray(material.stiffness, float), (*shape, 6, 6)),
            axes,
            None,
        )
        if material.piezoelectric_tensor is not None:
            e[index] = piezoelectric_in_frame(
                np.broadcast_to(
                    np.asarray(material.piezoelectric_tensor, float), (*shape, 3, 6)
                ),
                axes,
                None,
            )
        if material.mass_density is not None:
            rho[index] = np.asarray(material.mass_density, float)
        q = material.mechanical_quality_factor
        if q is not None:
            loss[index] = 1.0 / np.asarray(q, float)
    return {"c": c, "e": e, "rho": rho, "loss": loss}


def _tensor(built: Any, values: np.ndarray) -> ngs.CoefficientFunction:
    """Per-element ``(n, a, b)`` values as a P0 matrix coefficient."""
    _, a, b = values.shape
    parts = [p0(built, values[:, i, j]) for i in range(a) for j in range(b)]
    return ngs.CF(tuple(parts), dims=(a, b))


def _frame_depth(model: Model, built: Any, frame: Any) -> ngs.CoefficientFunction:
    r""":math:`\eta\,r^2` on the frame regions, zero elsewhere.

    The interior box is the bounding box of the mechanical triangles outside
    the frame, read from the imported nodes (the depth moves with the material
    points); a frame triangle beyond a side of the box has :math:`r` = the
    distance past it over the frame's extent on that side, the depths of a
    corner adding as squares.
    """
    mesh = model.mesh
    scale = built.scale
    dim = mesh.dim
    nodes = np.asarray(mesh.imported_nodes, float)[:, :dim] / scale
    cells = np.asarray(mesh.volume_elements)
    in_frame = np.zeros(len(cells), bool)
    for name in frame.regions:
        in_frame[mesh.elements(name)] = True
    mechanical = model._domain_of("mechanical").active  # noqa: SLF001
    interior = mechanical & ~in_frame
    box = nodes[np.unique(cells[interior])]
    lo, hi = box.min(axis=0), box.max(axis=0)
    ring = nodes[np.unique(cells[in_frame & mechanical])]
    low = lo - ring.min(axis=0)
    high = ring.max(axis=0) - hi
    with serial_spaces():
        space = ngs.H1(built.mesh, order=1)
    coordinates = []
    for axis in range(dim):
        gf = ngs.GridFunction(space)
        gf.vec.FV().NumPy()[:] = nodes[built.node_order, axis]
        coordinates.append(gf)
    depth: Any = 0.0
    for axis, x in enumerate(coordinates):
        side = ngs.CF(0.0)
        if high[axis] > 0:
            side = ngs.IfPos(x - hi[axis], (x - hi[axis]) / high[axis], side)
        if low[axis] > 0:
            side = ngs.IfPos(lo[axis] - x, (lo[axis] - x) / low[axis], side)
        depth = depth + side * side
    return p0(built, in_frame.astype(float)) * frame.loss_factor * depth


# --- the system --------------------------------------------------------------------


class KeptFactors:
    """The factorizations a forward keeps for its backward's field cotangents.

    One per kept frequency, by frequency index, within a byte budget: a
    factorization that does not fit is freed, and the backward factorizes that
    frequency again. It is an `Entry`'s handle, so the entry's bytes count
    them and the cache frees them with it.
    """

    def __init__(self, budget: int) -> None:
        """An empty store of at most ``budget`` bytes."""
        self.budget = budget
        self._handles: dict[int, DirectHandle] = {}

    def __bool__(self) -> bool:
        return bool(self._handles)

    @property
    def nbytes(self) -> int:
        """Host memory the kept factorizations hold, in bytes."""
        return sum(int(h.nbytes) for h in self._handles.values())

    def offer(self, index: int, handle: DirectHandle) -> bool:
        """Keep ``handle`` for frequency ``index`` if it fits the budget."""
        if self.nbytes + int(handle.nbytes) > self.budget:
            return False
        self._handles[index] = handle
        return True

    def get(self, index: int) -> DirectHandle | None:
        """The factorization of frequency ``index``, or ``None`` if not kept."""
        return self._handles.get(index)

    def free(self) -> None:
        """Free every kept factorization."""
        for handle in self._handles.values():
            handle.free()
        self._handles = {}


class PiezoelectricSystem:
    r"""The assembled operator, the electrode liftings and the per-frequency solves.

    Attributes:
        space: The compound space ``[u_x, u_y, u_z, phi]``.
        ndof: Its size.
        offset: Where the potential block starts (3 displacement blocks of
            ``offset // 3`` DOFs).
        free: Boolean mask of the free DOFs.
        lifts: ``(ndof, n_terminals)`` real liftings.
        free_bits: The free mask as an NGSolve bit array.
        matrix: The matrix of :math:`A = K - \omega^2 M`, rewritten per
            frequency (after assembly).
        loss_matrix: The real symmetric matrix :math:`L` with
            :math:`K = K_r + iL` (the damping and :math:`Q_m` part),
            assembled on first use.
    """

    def __init__(
        self,
        model: Model,
        problem: Piezoelectric,
        options: SolverOptions,
        rule: PiezoelectricRule,
        *,
        assemble: bool = True,
    ) -> None:
        self.rule = rule
        self.model = model
        self.options = options
        order = options.order
        self.order = order
        disc = discretization(model.mesh)
        built = self.built = disc.built
        electrodes = rule.electrodes
        mech = em_region(model, built, "mechanical")
        diel = em_region(model, built, "em")
        self.mech, self.diel = mech, diel

        vu = h1_space(
            model,
            built,
            order,
            tuple(problem.clamped),
            physics="mechanical",
            complex_valued=True,
        )
        vp = h1_space(
            model, built, order, electrodes.lines, physics="em", complex_valued=True
        )
        self.vu, self.vp = vu, vp
        with serial_spaces():
            self.space = space = ngs.FESpace([vu, vu, vu, vp])
        self.offset = 3 * vu.ndof
        self.ndof = space.ndof

        if assemble:
            self._assemble(model, rule)
        self.lifts, self.free = self._electrodes(vp, electrodes)
        self.potential_free = self._potential_free(vp, electrodes)
        self.free_bits = ngs.BitArray(self.free.tolist())

    def _assemble(self, model: Model, rule: PiezoelectricRule) -> None:
        """Assemble :math:`K` and :math:`M` (one pattern)."""
        built, space, order = self.built, self.space, self.order
        self._coefficients = _coefficients(model, built, rule.frame)
        forms = _weak_forms(self._coefficients, built, order, self.mech, self.diel)
        trial, test = space.TnT()
        u, v = tuple(trial), tuple(test)
        # one form, switched between K and M by two parameters: the two
        # matrices share one pattern, and neither is a difference of the other
        on, off = ngs.Parameter(1.0), ngs.Parameter(0.0)
        with ngs.TaskManager():
            k = ngs.BilinearForm(space, symmetric=True)
            for name, switch, factor in (
                ("stiffness", on, 1.0),
                ("loss", on, 1j),
                ("coupling", on, 1.0),
                ("dielectric", on, 1.0),
                ("mass", off, -1.0),
            ):
                form, dx = forms[name]
                k += switch * (factor * form(u, v)) * dx
            k.Assemble()
            self._k = k.mat.AsVector().FV().NumPy().copy()
            on.Set(0.0)
            off.Set(-1.0)
            k.Assemble()
            self._m = k.mat.AsVector().FV().NumPy().copy()
        self.matrix = k.mat.CreateMatrix()

    @property
    def loss_matrix(self) -> ngs.BaseMatrix:
        r"""The real symmetric :math:`L` with :math:`K = K_r + iL`, assembled on use."""
        if not hasattr(self, "_loss"):
            if not hasattr(self, "_coefficients"):
                self._assemble(self.model, self.rule)
            forms = _weak_forms(
                self._coefficients, self.built, self.order, self.mech, self.diel
            )
            trial, test = self.space.TnT()
            with ngs.TaskManager():
                form = ngs.BilinearForm(self.space, symmetric=True)
                damping, dx = forms["loss"]
                form += damping(tuple(trial), tuple(test)) * dx
                form.Assemble()
            self._loss = form.mat
        return self._loss

    # --- electrodes ----------------------------------------------------------------

    def _fixed_area_dofs(
        self, vp: ngs.FESpace, electrodes: Electrodes
    ) -> list[np.ndarray]:
        """The potential DOFs on each electrode area, ``(terminals..., ground)``."""
        out = []
        areas = [a for _, _, a in electrodes.terminals] + [electrodes.ground_areas]
        for names in areas:
            mask = np.zeros(vp.ndof, bool)
            for name in names:
                region = region_domain(self.model, self.built, name)
                mask |= free_mask(vp.GetDofs(region), vp.ndof)
            out.append(mask)
        return out

    def _electrodes(
        self, vp: ngs.FESpace, electrodes: Electrodes
    ) -> tuple[np.ndarray, np.ndarray]:
        """The liftings and the free mask of the compound space."""
        free = free_mask(self.space.FreeDofs(), self.ndof)
        areas = self._fixed_area_dofs(vp, electrodes)
        for mask in areas:
            free[self.offset :][mask] = False
        lifts = np.zeros((self.ndof, len(electrodes.terminals)))
        for j, (_, lines, names) in enumerate(electrodes.terminals):
            g = ngs.GridFunction(vp)
            one = ngs.CF(1.0)
            for name in lines:
                g.Set(one, definedon=boundary_region(self.built, (name,)))
            for name in names:
                g.Set(one, definedon=region_domain(self.model, self.built, name))
            w = np.real(g.vec.FV().NumPy()).copy()
            w[np.abs(w) < 1e-10] = 0.0
            lifts[self.offset :, j] = w
            if free[lifts[:, j] != 0].any():
                raise RuntimeError("internal: a lifting is nonzero on a free DOF.")
        return lifts, free

    def _potential_free(self, vp: ngs.FESpace, electrodes: Electrodes) -> np.ndarray:
        """The free mask of the potential alone (real space, same numbering)."""
        free = free_mask(vp.FreeDofs(), vp.ndof)
        for mask in self._fixed_area_dofs(vp, electrodes):
            free[mask] = False
        return free

    # --- solves --------------------------------------------------------------------

    def _factor(self, frequency: float, *, own: bool = False) -> DirectHandle:
        r"""The factorization of :math:`A = K - \omega^2 M` at ``frequency``.

        With ``own`` the handle gets a matrix of its own, to outlive the next
        factorization.
        """
        if not hasattr(self, "_k"):
            self._assemble(self.model, self.rule)
        omega = 2 * np.pi * frequency
        matrix = self.matrix.CreateMatrix() if own else self.matrix
        matrix.AsVector().FV().NumPy()[:] = self._k - omega**2 * self._m
        return DirectHandle(
            matrix,
            self.free_bits,
            "complex_symmetric",
            is_complex=True,
            cholesky=True,
        )

    def state(
        self,
        frequency: float,
        keep: tuple[KeptFactors, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(x, A x)``: the terminals' states at ``frequency``, ``(ndof, n_t)``.

        With ``keep = (factors, index)`` the factorization is offered to
        ``factors`` under ``index`` instead of freed, and freed if it is not kept.
        """
        handle = self._factor(frequency, own=keep is not None)
        try:
            lifts = self.lifts.astype(complex)
            x = lifts + handle.solve(-handle.apply(lifts))
            ax = handle.apply(x)
        except BaseException:
            handle.free()
            raise
        if keep is None or not keep[0].offer(keep[1], handle):
            handle.free()
        return x, ax

    def adjoint(
        self,
        frequency: float,
        loads: np.ndarray,
        kept: DirectHandle | None = None,
    ) -> np.ndarray:
        """:math:`A^{-1} g` on the free DOFs for each column of ``loads``.

        :math:`A` is complex symmetric, so the adjoint of a state is solved
        without a transpose. The result is zero on the fixed DOFs. With
        ``kept``, the forward's factorization at ``frequency``, nothing is
        assembled or factorized.
        """
        rhs = np.where(self.free[:, None], loads, 0.0)
        if kept is not None:
            return kept.solve(rhs)
        handle = self._factor(frequency)
        try:
            return handle.solve(rhs)
        finally:
            handle.free()

    def admittance(
        self,
        frequencies: tuple[float, ...],
        kept: tuple[float, ...],
        states: list[np.ndarray] | None = None,
        factors: KeptFactors | None = None,
    ) -> tuple[np.ndarray, dict[tuple[float, str], dict[str, np.ndarray]]]:
        """:math:`Y'` over ``frequencies`` (before symmetrizing) and the kept fields.

        With ``states`` a list, each frequency's terminal states are appended.
        With ``factors``, the factorizations of the kept frequencies are offered
        to it (`state`).
        """
        nt = self.lifts.shape[1]
        out = np.zeros((len(frequencies), nt, nt), complex)
        names = self.rule.problem.terminal_names
        fields: dict[tuple[float, str], dict[str, np.ndarray]] = {}
        nu = self.offset // 3
        for k, f in enumerate(frequencies):
            x, ax = self.state(
                f, (factors, k) if factors is not None and f in kept else None
            )
            if states is not None:
                states.append(x)
            out[k] = -1j * 2 * np.pi * f * (self.lifts.T @ ax)
            if f in kept:
                for j, name in enumerate(names):
                    parts = {
                        axis: x[i * nu : (i + 1) * nu, j].copy()
                        for i, axis in enumerate("xyz")
                    }
                    parts["phi"] = x[self.offset :, j].copy()
                    fields[(f, name)] = parts
        return out, fields

    def clamped_capacitance(self) -> tuple[np.ndarray, np.ndarray]:
        r""":math:`C'_0` and the potentials: :math:`\varepsilon^S` alone, no motion.

        The potentials are ``(n_potential_dofs, n_t)``, real.
        """
        model, built, order = self.model, self.built, self.order
        electrodes = self.rule.electrodes
        vr = h1_space(model, built, order, electrodes.lines, physics="em")
        eps = _permittivity(model, built)
        p, q = vr.TnT()
        gp, gq = _plane(ngs.grad(p)), _plane(ngs.grad(q))
        with ngs.TaskManager():
            form = ngs.BilinearForm(vr, symmetric=True)
            form += (gq * (eps * gp)) * quadrature.dx(order, definedon=self.diel)
            form.Assemble()
        free = ngs.BitArray(self.potential_free.tolist())
        handle = DirectHandle(
            form.mat, free, "hermitian", is_complex=False, cholesky=True
        )
        try:
            lifts = np.real(self.lifts[self.offset :])
            x = lifts + handle.solve(-handle.apply(lifts))
            return x.T @ handle.apply(x), x
        finally:
            handle.free()


def _strain(gx: Any, gy: Any, gz: Any) -> ngs.CoefficientFunction:
    """Engineering strain (xx, yy, zz, yz, xz, xy); none varies along z on a 2D mesh."""
    ax, ay, az = ngs.grad(gx), ngs.grad(gy), ngs.grad(gz)
    if ax.dim == 3:
        return ngs.CF(
            (ax[0], ay[1], az[2], az[1] + ay[2], az[0] + ax[2], ax[1] + ay[0])
        )
    return ngs.CF((ax[0], ay[1], 0, az[1], az[0], ax[1] + ay[0]))
