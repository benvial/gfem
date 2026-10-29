"""Electrostatic rule: terminal potentials in H1 and the Maxwell capacitance.

For terminal i, the potential phi_i is 1 on terminal i, 0 on the other
terminals and the ground, and solves the charge-free Laplace problem
``a(phi_i, v) = 0`` with ``a(u, v) = int eps0 eps_r grad u . grad v`` for
every v vanishing on the fixed surfaces. Then ``C_ij = a(phi_i, phi_j)``,
which equals Palace's energy formula ``E(phi_i + phi_j) - (C_ii + C_jj) / 2``
and is symmetric. All terminals share one prepared operator: the right-hand
sides are the liftings, solved together: by CG + BDDC by default, or by
one sparse Cholesky factorization (``direct``). ``phi_i^T K phi_j`` is
quadratic in the solver error (the error vanishes on the fixed DOFs, the
residual on the free ones), so the capacitance is far more accurate than
the solver tolerance.

The terminal potentials are their own adjoints (``dC_ij = phi_i^T dK phi_j``),
so a capacitance gradient needs no solve: a kept entry holds the potentials
only, and the forward frees its factorization or preconditioner. A potential
cotangent prepares the operator again in the backward pass (once per entry,
at the adjoint tolerance) for its adjoint solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem import _petsc
from gfem._adjoint import adjoint_tolerance
from gfem._boundaries import PEC, Conductor, sheet_of
from gfem._engine import quadrature
from gfem._engine.cache import Entry, discretization
from gfem._engine.coefficients import permittivity
from gfem._engine.domain import boundary_region, domain_key, em_region, h1_space
from gfem._engine.fields import fe_space
from gfem._engine.mesh import EngineMesh
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import FieldPair, ShapeTerm
from gfem._field import FieldMap
from gfem._linalg import KINDS, Handle, Operator, prepare
from gfem.mesh import Mesh
from gfem.model import Model, check_surface
from gfem.options import SolverOptions
from gfem.problems import Electrostatic
from gfem.results import ElectrostaticResult, Field

__all__ = ["ElectrostaticRule"]

#: The linear solver for ``linear_solver="auto"``: CG + BDDC.
DEFAULT_SOLVER = "bddc_gmres"


class ElectrostaticRule:
    """Host-side forward of an `Electrostatic` problem on a `Model`.

    Construction runs every static check (it needs no array values);
    `forward` reads the model's leaves (node coordinates, material values)
    and solves.
    """

    physics = "em"
    #: The operator family in the property-to-form-term table.
    form = "electrostatic"
    #: Material properties the electrostatic forms read.
    reads = frozenset({"eps_r"})
    #: Outputs whose cotangents the backward pass handles (not ``energy``).
    differentiable_outputs = frozenset({"capacitance", "potential"})

    def __init__(
        self, model: Model, problem: Electrostatic, options: SolverOptions
    ) -> None:
        """Check that the problem fits the model (see `gfem.Electrostatic`).

        Raises:
            gfem.SolverUnavailableError: ``linear_solver="petsc"`` and PETSc
                cannot be used.
        """
        if not isinstance(problem, Electrostatic):
            raise TypeError(f"Expected gfem.Electrostatic, got {problem!r}.")
        self.problem = problem
        self.options = options
        kind = options.linear_solver
        self.kind = DEFAULT_SOLVER if kind == "auto" else kind
        if self.kind not in KINDS:
            raise NotImplementedError(
                f"linear_solver={kind!r} is not available for electrostatic "
                "problems in this version of gfem; use one of "
                f"{', '.join(repr(k) for k in KINDS)}."
            )
        if self.kind == "petsc":  # before any assembly
            _petsc.require_petsc("linear_solver")
        for b in model.boundaries:
            if sheet_of(b) is not None:
                raise NotImplementedError(
                    f"{type(b).__name__} on {b.surface!r} carries a sheet field "
                    "(fill=), but a partly filled conducting sheet is not an "
                    "electrostatic object: electrostatic conductors are terminals "
                    "at a fixed potential. Fix the metal pattern (a PEC or "
                    "terminal on the metal's own surface), or optimize the "
                    "capacitance through a gfem.Driven or gfem.Eigenmode "
                    "surrogate, where the sheet field is differentiable."
                )
        in_terminal = {s for _, surfaces in problem.terminals for s in surfaces}
        conductors = tuple(
            b.surface
            for b in model.boundaries
            if isinstance(b, (PEC, Conductor)) and b.surface not in in_terminal
        )
        self.ground = tuple(dict.fromkeys(problem.ground + conductors))
        for _, surfaces in problem.terminals:
            for s in surfaces:
                _check_surface(model, s, "terminal")
        for s in problem.ground:
            _check_surface(model, s, "ground")
        _check_apart(model, problem.terminals, self.ground)
        #: Surfaces at a fixed potential: the terminals and the ground.
        self.fixed = tuple(sorted(in_terminal.union(self.ground)))
        mesh, active = model.mesh, model._domain.active  # noqa: SLF001
        #: Volume regions with tetrahedra in the solve domain, mesh order.
        self.regions = tuple(r for r in mesh.volumes if active[mesh.elements(r)].any())
        # The potentials live on the mesh as given (traced or not).
        self._mesh = model.mesh
        self._excluded = model._domain.excluded  # noqa: SLF001

    def forward(
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, np.ndarray], Entry | None]:
        """Solve every terminal; return the outputs and, if ``keep``, an entry.

        Args:
            model: The model, with concrete leaves.
            keep: Keep the potentials for a backward pass.

        Returns:
            ``{"capacitance": (n, n) array in F, "energy": (len(regions), n)
            array in J, "potential": (ndof, n) array in V, one column per
            terminal}``, and the cache entry (``fields["potential"]``; no
            handle) or ``None``.

        Raises:
            ValueError: Part of the solve domain touches no terminal or
                ground, so its potential is undetermined.
        """
        with task_manager(self.options.threads):
            space, built = self._space(model)
            regions = [
                boundary_region(built, surfaces)
                for _, surfaces in self.problem.terminals
            ]
            eps = permittivity(model, built)
            handle = self._prepare(space, eps, self.options.tolerance)
            lift = np.zeros((space.ndof, len(regions)))
            for i, region in enumerate(regions):
                gf = ngs.GridFunction(space)
                gf.Set(ngs.CF(1.0), definedon=region)
                lift[:, i] = gf.vec.FV().NumPy()
            phi = lift + handle.solve(-handle.apply(lift))
            scale = scipy.constants.epsilon_0 * built.scale
            c = phi.T @ handle.apply(phi) * scale
            energy = 0.5 * scale * self._region_energies(model, built, space, eps, phi)
        handle.free()
        c = 0.5 * (c + c.T)
        if not np.isfinite(c).all():
            raise ValueError(
                "The electrostatic potential is undetermined: some part of the "
                "solve domain touches no terminal and no ground. Add a ground "
                "surface there, or exclude that volume."
            )
        entry = None
        if keep:
            entry = Entry(physics=self.physics, fields={"potential": phi})
        return {"capacitance": c, "energy": energy, "potential": phi}, entry

    def _prepare(
        self, space: ngs.FESpace, eps: ngs.CoefficientFunction, tolerance: float
    ) -> Handle:
        """The prepared operator ``int eps_r grad u . grad v`` (file units)."""
        order = self.options.order
        u, v = space.TnT()
        form = ngs.BilinearForm(space, symmetric=True)
        form += eps * ngs.grad(u) * ngs.grad(v) * quadrature.dx(order)
        return prepare(
            Operator(form, space.FreeDofs()),
            symmetry="hermitian",
            kind=self.kind,
            device=self.options.device,
            threads=self.options.threads,
            tolerance=tolerance,
            width=len(self.problem.terminals),
            petsc_options=self.options.petsc_options,
        )

    def _space(self, model: Model) -> tuple[ngs.FESpace, EngineMesh]:
        order = self.options.order
        disc = discretization(model.mesh)
        built = disc.built
        fixed = self.fixed
        space = disc.space(
            (domain_key(model), "h1", order, fixed),
            lambda: h1_space(model, built, order, fixed),
        )
        return space, built

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output (from the mesh topology)."""
        n = len(self.problem.terminals)
        mesh = model.mesh
        still = Mesh(mesh.imported_nodes, mesh._topology)  # noqa: SLF001
        with task_manager(self.options.threads):
            _, space, _ = fe_space(
                still, self.regions, self._excluded, "h1", self.options.order
            )
        return {
            "capacitance": jax.ShapeDtypeStruct((n, n), jnp.float64),
            "energy": jax.ShapeDtypeStruct((len(self.regions), n), jnp.float64),
            "potential": jax.ShapeDtypeStruct((space.ndof, n), jnp.float64),
        }

    def role(self, kind: tuple) -> str:
        """``"read"`` if the electrostatic solve depends on input ``kind``.

        The permittivity and the node coordinates are read; every other
        material property (losses, ``mu_r``, thermal values), boundary and
        port value is not, so its gradient is an exact zero.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        if kind[0] in ("boundary", "port"):
            return "unread"
        return "read"

    def nondifferentiable_output(self, name: str) -> str:
        """The error message for a cotangent on output ``name``."""
        if name == "energy":
            # dE_r = phi^T dK_r phi / 2 + phi^T K_r dphi: unlike C, the region
            # energy is not stationary in phi, so it needs one adjoint solve
            # per terminal; not implemented.
            return (
                "gfem cannot differentiate the electrostatic region energies "
                "(result.energy) with respect to eps_r or the mesh nodes. "
                "Differentiate the capacitance instead (the "
                "total energy of excitation i is capacitance[i, i] / 2), or wrap "
                "result.energy in jax.lax.stop_gradient to use its value without "
                "its derivative."
            )
        return (
            f"gfem cannot differentiate the electrostatic output {name!r}. Wrap "
            "it in jax.lax.stop_gradient to use its value without its derivative."
        )

    def backward(
        self, model: Model, entry: Entry, cotangents: dict[str, np.ndarray]
    ) -> list[FieldPair]:
        """Fold the capacitance and potential cotangents into field pairs.

        The terminal potentials are their own adjoints for the capacitance:
        ``sum_ij ct_ij dC_ij = sum_i phi_i^T dK (sum_j G_ij phi_j)`` with
        ``G`` the symmetric part of ``ct`` (C is symmetrized), so no solve.
        A potential cotangent ``ct_i`` takes one adjoint solve per terminal,
        all columns at once: ``K phi_i = 0`` on the free DOFs gives
        ``ct_i . dphi_i = -lambda_i^T dK phi_i`` with ``K lambda_i = ct_i``
        (zero on the fixed DOFs), K the SI operator. Its operator is
        prepared on first use at the adjoint tolerance and kept in the entry
        for later pullbacks.
        """
        phi = entry.fields["potential"]
        eps0 = scipy.constants.epsilon_0
        with task_manager(self.options.threads):
            space, built = self._space(model)
            domain = em_region(model, built)
            left, right = [], []
            if "capacitance" in cotangents:
                ct = np.asarray(cotangents["capacitance"], dtype=float)
                left.append(phi)
                right.append(phi @ (0.5 * (ct + ct.T)).T)
            if "potential" in cotangents:
                ct = np.asarray(cotangents["potential"], dtype=float)
                tolerance = adjoint_tolerance(self.options)
                if entry.handle is None:
                    eps = permittivity(model, built)
                    entry.handle = self._prepare(space, eps, tolerance)
                # the handle solves the form without eps0 * scale
                adjoint = entry.handle.solve(ct, tolerance=tolerance)
                adjoint /= eps0 * built.scale
                left.append(-adjoint)
                right.append(phi)
        return [
            FieldPair(
                space=space,
                left=np.concatenate(left, axis=1),
                right=np.concatenate(right, axis=1),
                weights={"grad_grad": eps0},
                order=self.options.order,
                domain=domain,
            )
        ]

    def shape_terms(self, model: Model) -> list[ShapeTerm]:
        """The operator ``int Re(eps_r) grad u . grad v`` as a shape term.

        The pairs' weight ``eps0`` and the length power 1 make it SI, as for
        the permittivity. The terminal liftings are DOF values (1 on a
        terminal's DOFs, 0 elsewhere), so they do not move with the nodes.
        """
        _, built = self._space(model)
        return [
            ShapeTerm(
                "grad_grad",
                "gradient",
                coefficient=permittivity(model, built),
                length_power=1,
            )
        ]

    def result(self, outputs: dict) -> ElectrostaticResult:
        """The public result from the forward outputs."""
        names = self.problem.terminal_names
        potential = FieldMap(
            {
                name: Field(
                    coefficients=jnp.asarray(outputs["potential"][:, i]),
                    mesh=self._mesh,
                    quantity="V",
                    unit="V",
                    space="h1",
                    order=self.options.order,
                    support=self.regions,
                    _excluded=self._excluded,
                )
                for i, name in enumerate(names)
            },
            label="terminal",
        )
        return ElectrostaticResult(
            capacitance=jnp.asarray(outputs["capacitance"]),
            terminals=names,
            energy={
                r: jnp.asarray(e)
                for r, e in zip(self.regions, outputs["energy"], strict=True)
            },
            potential=potential,
        )

    def _region_energies(
        self,
        model: Model,
        built: EngineMesh,
        space: ngs.FESpace,
        eps: ngs.CoefficientFunction,
        phi: np.ndarray,
    ) -> np.ndarray:
        """``phi_i^T K_r phi_i``, shape ``(len(self.regions), n_terminals)``.

        One element pass per terminal (a P0 linear form on the pinned rule),
        summed per region over its active tetrahedra.
        """
        order = self.options.order
        mesh = model.mesh
        p0 = ngs.L2(built.mesh, order=0)
        q = p0.TestFunction()
        gf = ngs.GridFunction(space)
        grad = ngs.grad(gf)
        form = ngs.LinearForm(p0)
        form += (eps * grad) * grad * q * quadrature.dx(order)
        per_tet = np.zeros((len(mesh.tetrahedra), phi.shape[1]))
        for i in range(phi.shape[1]):
            gf.vec.FV().NumPy()[:] = phi[:, i]
            form.Assemble()
            per_tet[built.tetrahedron_order, i] = form.vec.FV().NumPy()
        active = model._domain.active  # noqa: SLF001
        out = np.zeros((len(self.regions), phi.shape[1]))
        for k, region in enumerate(self.regions):
            tets = mesh.elements(region)
            out[k] = per_tet[tets[active[tets]]].sum(axis=0)
        return out


def _check_surface(model: Model, name: str, role: str) -> None:
    check_surface(
        model.mesh,
        name,
        f"Electrostatic {role}",
        " Give the surfaces of the conductor's shell instead.",
    )
    if not model._domain.touches(name):  # noqa: SLF001
        raise ValueError(
            f"Electrostatic {role} surface {name!r} does not border the solve "
            "domain (it lies inside excluded volumes). Name a surface of the "
            "domain, or exclude fewer volumes."
        )


def _nodes(model: Model, surfaces: tuple[str, ...]) -> np.ndarray:
    mesh = model.mesh
    tags = [mesh.region(s).tag for s in surfaces]
    return np.unique(mesh.triangles[np.isin(mesh.triangle_tags, tags)])


def _check_apart(
    model: Model,
    terminals: tuple[tuple[str, tuple[str, ...]], ...],
    ground: tuple[str, ...],
) -> None:
    """Terminals and the ground must not touch: touching conductors are one."""
    groups = [(f"terminal {n!r}", _nodes(model, s)) for n, s in terminals]
    if ground:
        groups.append(("the ground", _nodes(model, ground)))
    for i, (a, na) in enumerate(groups):
        for b, nb in groups[i + 1 :]:
            shared = np.intersect1d(na, nb).size
            if shared:
                raise ValueError(
                    f"Electrostatic: {a} and {b} touch ({shared} shared mesh "
                    "nodes), so they are one conductor at one potential. Merge "
                    "them into one terminal, or separate them in the geometry."
                )
