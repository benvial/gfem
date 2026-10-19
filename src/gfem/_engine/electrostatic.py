"""Electrostatic rule: terminal potentials in H1 and the Maxwell capacitance.

For terminal i, the potential phi_i is 1 on terminal i, 0 on the other
terminals and the ground, and solves the charge-free Laplace problem
``a(phi_i, v) = 0`` with ``a(u, v) = int eps0 eps_r grad u . grad v`` for
every v vanishing on the fixed surfaces. Then ``C_ij = a(phi_i, phi_j)``,
which equals Palace's energy formula ``E(phi_i + phi_j) - (C_ii + C_jj) / 2``
and is symmetric. All terminals share one prepared operator: the right-hand
sides are the liftings, solved together.

The terminal potentials are their own adjoints (``dC_ij = phi_i^T dK phi_j``),
so a capacitance gradient needs no solve; the handle is kept only for
potential functionals.
"""

from __future__ import annotations

import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem._boundaries import PEC, Conductor
from gfem._engine import quadrature
from gfem._engine.cache import Entry, discretization
from gfem._engine.coefficients import permittivity
from gfem._engine.domain import boundary_region, domain_key, h1_space
from gfem._engine.runtime import task_manager
from gfem._linalg import KINDS, Operator, prepare
from gfem.errors import RegionNotFoundError
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Electrostatic
from gfem.results import ElectrostaticResult

__all__ = ["ElectrostaticRule"]

#: The linear solver for ``linear_solver="auto"``.
DEFAULT_SOLVER = "direct"


class ElectrostaticRule:
    """Host-side forward of an `Electrostatic` problem on a `Model`.

    Construction runs every static check (it needs no array values);
    `forward` reads the model's leaves (node coordinates, material values)
    and solves.
    """

    physics = "em"
    #: Material properties the electrostatic forms read.
    reads = frozenset({"eps_r"})

    def __init__(
        self, model: Model, problem: Electrostatic, options: SolverOptions
    ) -> None:
        """Check that the problem fits the model (see `gfem.Electrostatic`)."""
        if not isinstance(problem, Electrostatic):
            raise TypeError(f"Expected gfem.Electrostatic, got {problem!r}.")
        self.problem = problem
        self.options = options
        kind = options.linear_solver
        self.kind = DEFAULT_SOLVER if kind == "auto" else kind
        if self.kind not in KINDS:
            raise NotImplementedError(
                f"linear_solver={kind!r} is not available for electrostatic "
                "problems in this version of gfem; use linear_solver='direct'."
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

    def forward(
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, np.ndarray], Entry | None]:
        """Solve every terminal; return the outputs and, if ``keep``, an entry.

        Args:
            model: The model, with concrete leaves.
            keep: Keep the solver handle and the potentials for a backward
                pass.

        Returns:
            ``{"capacitance": (n, n) array in F}``, and the cache entry (the
            handle and ``fields["potential"]``, one DOF column per terminal)
            or ``None``.

        Raises:
            ValueError: Part of the solve domain touches no terminal or
                ground, so its potential is undetermined.
        """
        order = self.options.order
        with task_manager(self.options.threads):
            disc = discretization(model.mesh)
            built = disc.built
            space = disc.space(
                (domain_key(model), "h1", order),
                lambda: h1_space(model, built, order),
            )
            regions = [
                boundary_region(built, surfaces)
                for _, surfaces in self.problem.terminals
            ]
            fixed = [space.GetDofs(r) for r in regions]
            if self.ground:
                fixed.append(space.GetDofs(boundary_region(built, self.ground)))
            free = space.FreeDofs() & ~fixed[0]
            for dofs in fixed[1:]:
                free &= ~dofs

            u, v = space.TnT()
            form = ngs.BilinearForm(space, symmetric=True)
            eps = permittivity(model, built)
            form += eps * ngs.grad(u) * ngs.grad(v) * quadrature.dx(order)
            handle = prepare(
                Operator(form, free),
                symmetry="hermitian",
                kind=self.kind,
                device=self.options.device,
            )
            lift = np.zeros((space.ndof, len(regions)))
            for i, region in enumerate(regions):
                gf = ngs.GridFunction(space)
                gf.Set(ngs.CF(1.0), definedon=region)
                lift[:, i] = gf.vec.FV().NumPy()
            phi = lift + handle.solve(-handle.apply(lift))
            scale = scipy.constants.epsilon_0 * built.scale
            c = phi.T @ handle.apply(phi) * scale
        c = 0.5 * (c + c.T)
        if not np.isfinite(c).all():
            handle.free()
            raise ValueError(
                "The electrostatic potential is undetermined: some part of the "
                "solve domain touches no terminal and no ground. Add a ground "
                "surface there, or exclude that volume."
            )
        entry = None
        if keep:
            entry = Entry(
                physics=self.physics, handle=handle, fields={"potential": phi}
            )
        else:
            handle.free()
        return {"capacitance": c}, entry

    def result(self, outputs: dict) -> ElectrostaticResult:
        """The public result from the forward outputs."""
        return ElectrostaticResult(
            capacitance=jnp.asarray(outputs["capacitance"]),
            terminals=self.problem.terminal_names,
        )


def _check_surface(model: Model, name: str, role: str) -> None:
    mesh = model.mesh
    if name not in mesh.surfaces:
        if name in mesh.volumes:
            raise ValueError(
                f"Electrostatic {role} {name!r} is a volume; give the surfaces of "
                "the conductor's shell instead."
            )
        raise RegionNotFoundError(name, mesh.surfaces, kind="surface")
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
