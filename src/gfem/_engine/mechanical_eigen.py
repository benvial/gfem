r"""Mechanical eigenmodes of a 2D section or a 3D body: :math:`\hat K(q) - \omega^2 M`.

The unknowns are the three displacement components of a field
:math:`u(x, y)\,e^{i\omega t - iqz}` on the mechanical domain, in one compound
space of three H1 spaces (each with its own clamped lines). The trial's
:math:`\partial_z` is :math:`-iq` and the test's :math:`+iq`, so the lossless
:math:`\hat K` is Hermitian and :math:`M` real symmetric; the weak form splits
into real forms by power of :math:`q` (`gfem._engine.piezoelectric._weak_forms`
with ``wavenumber_pieces``),

.. math::

    \hat K(q) = K_0 - i q\,K_1 + q^2 K_2, \qquad
    \partial_q \hat K = -i K_1 + 2 q\,K_2,

each :math:`K_p` the stiffness plus :math:`i` times the loss piece of power
:math:`p` (:math:`q` in the mesh's length unit). At :math:`q = 0` only
:math:`K_0` and :math:`M` are assembled into the matrix that is factorized.
A 3D body (tetrahedra, fields :math:`u(x, y, z)\,e^{i\omega t}`) is that
case, with the full six-component strain and the whole potential gradient: it
has no wavenumber, and its forms carry one more length of the mesh's unit, so
the pencil is in SI (a mode of 1 J).

The modes nearest ``target`` are found by shift-invert Arnoldi (SciPy ARPACK)
on :math:`A(\sigma)^{-1} M` with :math:`A(\sigma) = \hat K - \sigma M`,
:math:`\sigma = (2\pi\,\text{target})^2`, restricted to the free DOFs, on one
`DirectHandle` (sparse :math:`LDL^T` at :math:`q = 0`, LU otherwise):
:math:`\lambda = \sigma + 1/\nu`. A mode's left vector :math:`y`
(:math:`y^T A(\lambda) = 0`) is :math:`\bar x` for a lossless section (a
Hermitian :math:`\hat K` and a real :math:`M`), :math:`x` at :math:`q = 0` (a
complex symmetric pencil), and otherwise comes from inverse iteration on the
transposed LU of :math:`A` at a nearby shift. The group velocity is
:math:`\mathrm{Re}\,[y^T (\partial_q \hat K) x / (2\omega\,y^T M x)]`.

Gradients reach the pencil's pieces through field pairs of the
piezoelectric kernel: :math:`\omega` through :math:`y` and :math:`x` with no
solve, a mode's field (displacement and potential, as normalized and gauged)
through one bordered solve with :math:`A(\lambda)^T` at the converged
eigenvalue (`MechanicalSystem.field_adjoint`).
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph  # noqa: F401
import scipy.sparse.linalg as spla

from gfem import _mpi
from gfem._checks import is_traced
from gfem._eigen.reach import clusters
from gfem._engine.cache import Entry, discretization
from gfem._engine.domain import em_region, h1_space, region_domain
from gfem._engine.elastic_pml import ElasticStretch, check_layer
from gfem._engine.piezoelectric import (
    _check_absorbing,
    _check_clamped,
    _check_electrodes,
    _check_frame,
    _check_materials,
    _check_mechanical,
    _coefficients,
    _element_arrays,
    _tensors,
    _weak_forms,
)
from gfem._engine.runtime import serial_spaces, task_manager
from gfem._engine.sensitivity import FieldPair, Tensors, Term
from gfem._engine.solve_request import SolveRequest
from gfem._field import FieldMap
from gfem._linalg import KRYLOV_KINDS
from gfem._linalg.base import free_mask
from gfem._linalg.direct import DirectHandle
from gfem._pml import derive_layers
from gfem.errors import (
    EigenmodeConvergenceError,
    LayerClearanceWarning,
    NonDifferentiableError,
)
from gfem.mesh import Mesh
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import MechanicalEigenmode
from gfem.results import Field, MechanicalEigenmodeResult

__all__ = ["MechanicalEigenmodeRule", "MechanicalSystem"]

#: ARPACK's relative tolerance on the shift-inverted eigenvalues.
_ARPACK_TOLERANCE = 1e-12
#: Modes with :math:`|\lambda| <` this times the shift are rigid motions.
_STATIC = 1e-8
#: The relative shift of a left vector's inverse iteration, and its step bound.
_LEFT_SHIFT = 1e-8
_LEFT_STEPS = 6
#: A left-vector step that lowers the backward error by less than this factor has
#: met the round-off floor of the factorization: the iteration stops there.
_LEFT_STALL = 0.1
#: The inverse-iteration steps that refine a coupled mode's right vector.
_REFINE_STEPS = 2
#: A mode whose stretch sensitivity (`MechanicalSystem.sensitivities`) is above
#: this is a mode of the layer, not of the device. The decay of a mode of the
#: layer is made by the stretch, so its sensitivity is about one or more; that
#: of a mode of the device is the discretization error of its field in the layer
#: (below :math:`10^{-4}` on the leaky columns). A decade below one.
_LAYER_SENSITIVITY = 0.1
#: The decay rate below which a mode counts as undamped: its stretch sensitivity
#: is relative to this times :math:`|\lambda|` instead of to its decay rate. A
#: trapped mode is then kept while the stretch moves its eigenvalue by less than
#: `_LAYER_SENSITIVITY` times this, relatively (measured :math:`8 \cdot 10^{-8}` for
#: a mode uniform through the layer); a mode of the layer on a branch of the
#: substrate, whose sensitivity is about one, is dropped up to substrates of about
#: a hundred wavelengths.
_UNDAMPED = 1e-4
#: The most modes the eigensolver is asked for when it widens its search.
_NEV_CAP = 200
#: ``Im omega / |omega|`` below minus this is a gain-sign mode of a layer
#: (round-off of a trapped mode is far below it).
_GAIN = 1e-6
#: Two eigenvalues of one mode (right and left search) agree to this, relatively.
_MATCH = 1e-8

#: The assembled groups: real forms by power of the wavenumber, and the mass.
_GROUPS = {
    "k0": (("stiffness", 1.0), ("loss", 1j)),
    "k1": (("stiffness1", 1.0), ("loss1", 1j)),
    "k2": (("stiffness2", 1.0), ("loss2", 1j)),
    "m": (("mass", 1.0),),
}
#: What a stretched layer adds to each group: the imaginary-coefficient forms
#: (weight ``i``, ``i * i`` for the loss' own ``i``) and, group ``mp``, the
#: physical mass the modes are normalized by.
_STRETCHED = {
    "k0": (("stiffness_im", 1j), ("loss_im", -1.0)),
    "k1": (("stiffness1_im", 1j), ("loss1_im", -1.0)),
    "k2": (("stiffness2_im", 1j), ("loss2_im", -1.0)),
    "m": (("mass_im", 1j),),
    "mp": (("mass_phys", 1.0),),
}
#: The stretch derivatives of the stiffness and the mass, and the loss.
_StretchPencil = tuple[sp.csr_array, sp.csr_array, sp.csr_array | None]
#: The loss of a stretched solve's stiffness, by power of the wavenumber: the
#: material part of the eigenvalue, which the stretch sensitivity leaves out.
_LOSS = {
    "l0": (("loss", 1.0), ("loss_im", 1j)),
    "l1": (("loss1", 1.0), ("loss1_im", 1j)),
    "l2": (("loss2", 1.0), ("loss2_im", 1j)),
}
_STRETCHED_COUPLED = {
    "k0": (("coupling_im", 1j), ("dielectric_im", 1j)),
    "k1": (("coupling1_im", 1j), ("dielectric1_im", 1j)),
    "k2": (("coupling2_im", 1j), ("dielectric2_im", 1j)),
}
#: What a piezoelectric section adds to each group (the potential's forms).
_COUPLED = {
    "k0": (("coupling", 1.0), ("dielectric", 1.0)),
    "k1": (("coupling1", 1.0), ("dielectric1", 1.0)),
    "k2": (("coupling2", 1.0), ("dielectric2", 1.0)),
}


class MechanicalEigenmodeRule:
    """Host-side forward of a `MechanicalEigenmode` problem on a 2D or 3D `Model`.

    Construction runs every static check (it needs no array values);
    `forward` assembles the pencil and finds the modes.
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
    differentiable_outputs: frozenset[str] = frozenset(
        {"omega", "cluster", "displacement", "potential"}
    )

    def __init__(
        self,
        model: Model,
        problem: MechanicalEigenmode,
        options: SolverOptions,
        *,
        comm: Any = None,
    ) -> None:
        """Check that the problem fits the model (see `gfem.MechanicalEigenmode`).

        Raises:
            NotImplementedError: ``linear_solver="petsc"``, or a slab mesh; on a
                3D body also ``comm``, a Krylov ``linear_solver`` and a GPU
                ``device``.
            RegionNotFoundError: A name is not a region of the mesh.
            ValueError: The problem does not fit the model (a ``wavenumber`` on
                a 3D body, a ``model.pml`` meeting the mechanical domain
                included); the message gives the remedy.
            MissingMaterialPropertyError: A region of the mechanical domain
                has no stiffness or density.
        """
        if not isinstance(problem, MechanicalEigenmode):
            raise TypeError(f"Expected gfem.MechanicalEigenmode, got {problem!r}.")
        self.problem = problem
        self.options = options
        self.comm = comm
        if options.linear_solver == "petsc":
            raise NotImplementedError(
                "A MechanicalEigenmode solve has no PETSc path in this version: use "
                'linear_solver="direct" or "auto".'
            )
        owner = "MechanicalEigenmode"
        self.dim = model.mesh.dim
        _refuse_body(model, problem, options, comm)
        self.mechanical = _check_mechanical(model, owner)
        _check_clamped(model, problem)
        _check_absorbing(model, owner)
        _check_materials(model)
        self.frame = _check_frame(model, problem)
        self.coupled = _coupled(model)
        if problem.ground and not self.coupled:
            raise ValueError(
                f"MechanicalEigenmode.ground {list(problem.ground)} needs a "
                "potential, and no region of the mechanical domain has a "
                "piezoelectric tensor: drop ground=, or give the material its "
                "piezoelectric_tensor."
            )
        #: The grounded lines and areas of the potential (coupled models).
        self.electrodes = _check_electrodes(model, problem) if self.coupled else None
        mesh = model.mesh
        mechanical = model._domain_of("mechanical")  # noqa: SLF001
        #: The areas with elements in the mechanical domain, mesh order.
        self.mechanical_regions = tuple(
            r for r in mesh.volumes if mechanical.active[mesh.elements(r)].any()
        )
        domain = model._domain  # noqa: SLF001
        self.potential_regions = tuple(
            r for r in mesh.volumes if domain.active[mesh.elements(r)].any()
        )
        _refuse_layer(model, mechanical)
        #: The elastic layer (`gfem.PML`) and its geometry; ``None`` without one.
        self.pml = problem.pml
        self.layers = None
        if self.pml is not None:
            check_layer(model, self.pml, mechanical.active)
            active = mechanical.active | (domain.active if self.coupled else False)
            self.layers = derive_layers(mesh, self.pml, active)
        self._mesh = mesh
        self._mechanical_excluded = mechanical.excluded
        self._em_excluded = domain.excluded

    def forward(
        self,
        model: Model,
        *,
        keep: bool,
        request: SolveRequest | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], Entry | None]:
        """Find the modes nearest the target.

        Returns:
            ``{"omega": (n,) rad/s, "error": (n,), "group_velocity": (n,) m/s,
            "cluster": (n,), "displacement": (n, 3, n_dof) m, "potential":
            (n, n_dof) V, empty without coupling}``, the modes nearest first,
            and with ``keep`` the entry a backward reads: the right and left
            vectors of each mode on the whole space, ``y^T M x``, the
            eigenvalue ``omega**2`` and whether each mode shares its cluster.

        Raises:
            EigenmodeConvergenceError: Fewer than ``n_modes`` modes meet
                ``SolverOptions.eigen_tolerance``.
        """
        with task_manager(self.options.threads, self.comm):
            system = MechanicalSystem(model, self.problem, self.options, self)
            try:
                found = system.modes()
                fields = system.kept
            finally:
                system.release()
        return found, Entry("em", fields=fields) if keep else None

    def result(self, outputs: dict[str, Any]) -> MechanicalEigenmodeResult:
        """The public result from the forward outputs."""
        order = self.options.order
        omegas = jnp.asarray(outputs["omega"])
        displacement = [
            FieldMap(
                {
                    axis: Field(
                        coefficients=outputs["displacement"][i][a],
                        mesh=self._mesh,
                        quantity=f"u{axis}",
                        unit="m",
                        space="h1",
                        order=order,
                        support=self.mechanical_regions,
                        omega=omegas[i],
                        _excluded=self._mechanical_excluded,
                    )
                    for a, axis in enumerate("xyz")
                },
                label="component",
                wavenumber=self.problem.wavenumber,
            )
            for i in range(omegas.shape[0])
        ]
        potential = []
        if self.coupled:
            potential = [
                Field(
                    coefficients=outputs["potential"][i],
                    mesh=self._mesh,
                    quantity="V",
                    unit="V",
                    space="h1",
                    order=order,
                    support=self.potential_regions,
                    omega=omegas[i],
                    _excluded=self._em_excluded,
                )
                for i in range(omegas.shape[0])
            ]
        return MechanicalEigenmodeResult(
            omega=jnp.asarray(outputs["omega"]),
            error=jnp.asarray(outputs["error"]),
            group_velocity=jnp.asarray(outputs["group_velocity"]),
            wavenumber=self.problem.wavenumber,
            cluster=jnp.asarray(outputs["cluster"]),
            displacement_fields=displacement,
            potential_fields=potential,
        )

    # --- the solve_vjp interface ---------------------------------------------------

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output (from the mesh topology)."""
        n = self.problem.n_modes
        mesh = model.mesh
        # the sizes depend on the topology only: traced nodes are not read
        fixed = model.replace(mesh=Mesh(mesh.imported_nodes, mesh._topology))  # noqa: SLF001
        with task_manager(self.options.threads, self.comm):
            system = MechanicalSystem(
                fixed, self.problem, self.options, self, assemble=False
            )
        return {
            "omega": jax.ShapeDtypeStruct((n,), jnp.complex128),
            "error": jax.ShapeDtypeStruct((n,), jnp.float64),
            "group_velocity": jax.ShapeDtypeStruct((n,), jnp.float64),
            "cluster": jax.ShapeDtypeStruct((n,), jnp.int64),
            "displacement": jax.ShapeDtypeStruct((n, 3, system.nu), jnp.complex128),
            "potential": jax.ShapeDtypeStruct((n, system.nphi), jnp.complex128),
        }

    def role(self, kind: tuple) -> str:
        """``"read"`` if the solve depends on input ``kind``.

        The material properties of `reads` and the node coordinates are read;
        every other property, boundary and port value is not, so its gradient
        is an exact zero.
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
                "MechanicalEigenmode solve"
            )
        return None

    def nondifferentiable_output(self, name: str) -> str:
        """The error message for a cotangent on output ``name``."""
        why = {
            "group_velocity": (
                "it is NaN for a 3D body, which has no axial band"
                if self._mesh.dim == 3
                else "it is read from the modes and their derivative in q"
            ),
            "error": "it is the backward error of the solve",
        }[name]
        return (
            f"gfem does not differentiate the MechanicalEigenmode output {name!r}: "
            f"{why}. The eigenfrequencies (omega, so frequency and Q) and the "
            "mode fields keep their gradients; wrap the value in "
            "jax.lax.stop_gradient to use it without its derivative."
        )

    def backward(
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        wanted: frozenset | None = None,  # noqa: ARG002
    ) -> list[FieldPair]:
        r"""Fold the cotangents of ``omega`` and the mode fields into field pairs.

        With :math:`\lambda = \omega^2`, the left vector :math:`y`, the right
        one :math:`x` and :math:`g = y^T M x`, :math:`d\lambda = y^T (d\hat K -
        \lambda\,dM)\,x / g` and :math:`d\omega = d\lambda / (2\omega)`: for a
        cotangent :math:`\bar\omega` alone the pair is left :math:`y`, right
        :math:`x\,\bar\omega / (2\omega g)`, with the weights of every piece of
        :math:`\hat K` at :math:`q` and :math:`-\lambda` on the mass, and no
        solve. A cotangent on a mode's displacement or potential adds one
        solve at the converged eigenvalue (`MechanicalSystem.field_adjoint`).
        The left field
        is the test side of each form (`_terms`, `tensors`).

        Raises:
            NonDifferentiableError: A mode with a cotangent shares its cluster
                (its vectors depend on a choice of basis).
        """
        if self.pml is not None:
            raise NotImplementedError(
                "gfem does not differentiate a MechanicalEigenmode solve with a "
                "perfectly matched layer (pml=) in this version: solve the "
                "gradient case with damping_frame= or without a layer."
            )
        fields = entry.fields
        n = len(fields["lambda"])
        omega_ct = np.zeros(n, complex)
        if cotangents.get("omega") is not None:
            omega_ct += np.asarray(cotangents["omega"])
        field_ct = _field_cotangent(cotangents, n, len(fields["x"]))
        has_field = np.any(field_ct != 0, axis=1)
        live = np.flatnonzero((omega_ct != 0) | has_field)
        if not len(live):
            return []
        for i in live:
            if fields["shared"][i]:
                omega = np.sqrt(fields["lambda"][i])
                raise NonDifferentiableError(
                    "gfem cannot differentiate the frequency or the field of the "
                    f"MechanicalEigenmode mode {i} at {omega / (2 * np.pi):.6g} Hz: "
                    "it is degenerate with another mode (to "
                    "SolverOptions.cluster_rtol), so its eigenvector, and its "
                    "derivative, depend on an arbitrary choice of basis. Break the "
                    f"symmetry of the {'body' if self._mesh.dim == 3 else 'section'}, "
                    "or wrap the value in "
                    "jax.lax.stop_gradient to use it without its derivative."
                )
        pairs: list[FieldPair] = []
        with task_manager(self.options.threads, self.comm):
            # the field adjoint solves on the assembled pencil; omega alone reads
            # nothing of it
            system = MechanicalSystem(
                model, self.problem, self.options, self, assemble=bool(has_field.any())
            )
            try:
                domain = em_region(model, system.built, "em")
                weights = functools.partial(_weights, system.q, coupled=self.coupled)
                for i in live:
                    lam = fields["lambda"][i]
                    omega = np.sqrt(lam)
                    x = fields["x"][:, i]
                    y = fields["y"][:, i]
                    left = y * (omega_ct[i] / (2 * omega * fields["gram"][i]))
                    if has_field[i]:
                        adjoint = system.field_adjoint(
                            lam,
                            x,
                            y,
                            fields["gram"][i],
                            field_ct[i],
                            gauge=int(fields["gauge"][i]),
                        )
                        left = left + adjoint.left
                        pairs.append(
                            FieldPair(
                                space=system.space,
                                left=np.conj(x)[:, None],
                                right=x[:, None] * adjoint.mass,
                                weights={"mass": 1.0},
                                order=self.options.order,
                                domain=domain,
                            )
                        )
                    pairs.append(
                        FieldPair(
                            space=system.space,
                            left=left[:, None],
                            right=x[:, None],
                            weights=weights(lam),
                            order=self.options.order,
                            domain=domain,
                        )
                    )
            finally:
                system.release()
        return pairs

    def terms(self, model: Model, *, nodes: bool) -> list[Term]:  # noqa: ARG002
        r"""The pieces of :math:`\hat K - \lambda M` for the node pass, as terms."""
        return _terms(self, model)

    def tensors(self, model: Model, wanted: frozenset) -> Tensors:
        """The tensor pass of the wanted material leaves (see `Tensors`)."""
        return _tensors(
            self,
            model,
            wanted,
            wavenumber=self.problem.wavenumber * discretization(model.mesh).built.scale,
            coupled=self.coupled,
        )


def _refuse_body(
    model: Model, problem: MechanicalEigenmode, options: SolverOptions, comm: Any
) -> None:
    """Raise for the slab, and for what a 3D body does not have.

    The 2D refusals of `gfem._dim2.refuse_solve` do not reach a mechanical 3D
    solve: a communicator, a Krylov solver and a GPU have no path here.
    """
    mesh = model.mesh
    if mesh.section_thickness is not None:
        raise NotImplementedError(
            "A MechanicalEigenmode problem solves a 2D mesh "
            "(gfem.Mesh.from_gmsh on a triangle-only file) or a tetrahedral "
            "body; a slab cross-section has no mechanicaleigenmode solve in "
            "this version."
        )
    if mesh.dim != 3:
        return
    if problem.wavenumber != 0:
        raise ValueError(
            "a 3D body has no axial wavenumber; wavenumber is the phase along z "
            "of a 2D section: leave it at 0."
        )
    refused = (
        (comm is not None, "A solve under several processes (comm=)"),
        (
            options.linear_solver in KRYLOV_KINDS,
            f"linear_solver={options.linear_solver!r}",
        ),
        (options.device != "cpu", f"device={options.device!r}"),
    )
    for found, what in refused:
        if found:
            raise NotImplementedError(
                f"{what} is not available for a 3D MechanicalEigenmode: solve on "
                'one process, with linear_solver="direct" or "auto", on '
                'device="cpu".'
            )


def _refuse_layer(model: Model, mechanical: Any) -> None:
    """Raise when the model's electromagnetic layer meets the mechanical domain."""
    if model.pml is None:
        return
    mesh = model.mesh
    for name in model.pml.regions:
        if mechanical.active[mesh.elements(name)].any():
            raise ValueError(
                f"The model's PML region {name!r} meets the mechanical domain, "
                "and the mechanical solve does not read the model's "
                "electromagnetic layer: pass it as MechanicalEigenmode(pml=...) "
                "to absorb the radiated waves, or exclude the region from "
                'mechanical problems (excluded={"mechanical": ...}).'
            )


def _field_cotangent(
    cotangents: dict[str, np.ndarray], n: int, ndof: int
) -> np.ndarray:
    """The cotangents of the ``n`` modes' displacement and potential, ``(n, ndof)``.

    In the order of the whole space: the three displacement blocks, then the
    potential's. An output without a cotangent is zero.
    """
    blocks = [
        None if cotangents.get(name) is None else np.reshape(cotangents[name], (n, -1))
        for name in ("displacement", "potential")
    ]
    sizes = [0 if b is None else b.shape[1] for b in blocks]
    out = np.zeros((n, ndof), complex)
    if blocks[0] is not None:
        out[:, : sizes[0]] = blocks[0]
    if blocks[1] is not None and sizes[1]:
        out[:, ndof - sizes[1] :] = blocks[1]
    return out


def _weights(q: float, lam: complex, *, coupled: bool) -> dict[str, complex]:
    r"""The weights of the pieces of :math:`\hat K - \lambda M`.

    ``q`` is the wavenumber in the mesh's length unit: the first-order pieces
    carry :math:`-iq` (:math:`q` for the loss, which has an extra :math:`i`),
    the second-order ones :math:`q^2`. Those at :math:`q = 0` are left out.
    """
    out: dict[str, complex] = {"stiffness": 1.0, "loss": 1j, "mass": -lam}
    if coupled:
        out |= {"coupling": 1.0, "dielectric": 1.0}
    if q != 0:
        out |= {
            "stiffness1": -1j * q,
            "loss1": q,
            "stiffness2": q**2,
            "loss2": 1j * q**2,
        }
        if coupled:
            out |= {
                "coupling1": -1j * q,
                "dielectric1": -1j * q,
                "coupling2": q**2,
                "dielectric2": q**2,
            }
    return out


def _terms(rule: MechanicalEigenmodeRule, model: Model) -> list[Term]:
    """The node pass's terms: the pieces of the pencil, as `MechanicalSystem` has them.

    Each is the integral of its real-coefficient form on the pinned rule. The
    first field of the pair is the test (left) and the second the trial: the
    forms take ``(trial, test)``.
    """
    problem, options = rule.problem, rule.options
    built = discretization(model.mesh).built
    mech = em_region(model, built, "mechanical")
    diel = em_region(model, built, "em")
    spaces = [
        h1_space(
            model,
            built,
            options.order,
            tuple(name for name, letters in problem.clamped if axis in letters),
            physics="mechanical",
        )
        for axis in "xyz"
    ]
    if rule.electrodes is not None:
        spaces.append(
            h1_space(model, built, options.order, rule.electrodes.lines, physics="em")
        )
    with serial_spaces():
        space = ngs.FESpace(spaces)
    forms = _weak_forms(
        _coefficients(model, built, rule.frame),
        built,
        options.order,
        mech,
        diel,
        wavenumber_pieces=True,
    )
    names = {piece for g in _GROUPS for piece, _ in _GROUPS[g]}
    if rule.coupled:
        names |= {piece for g in _COUPLED for piece, _ in _COUPLED[g]}
    n = len(spaces)

    def term(name: str) -> Term:
        form, dx = forms[name]

        def integrand(left: Any, right: Any) -> Any:
            return form(_parts(right, n), _parts(left, n)) * dx

        return Term(name, "surface", integrand=integrand, space=space)

    return [term(name) for name in forms if name in names]


def _parts(u: Any, n: int) -> tuple[Any, ...]:
    """The ``n`` components of a compound field ``u`` of the system's space."""
    return tuple(u.components[i] for i in range(n))


def _coupled(model: Model) -> bool:
    """Whether a mechanical region has a nonzero piezoelectric tensor."""
    mesh = model.mesh
    active = model._domain_of("mechanical").active  # noqa: SLF001
    for region in mesh.volumes:
        if not active[mesh.elements(region)].any():
            continue
        tensor = getattr(model.materials.get(region), "piezoelectric_tensor", None)
        if tensor is None:
            continue
        if is_traced(tensor) or np.any(np.asarray(tensor) != 0):
            return True
    return False


@dataclasses.dataclass(frozen=True)
class FieldAdjoint:
    """What a cotangent of a mode's field adds to the backward pass.

    Attributes:
        left: The left field of the pair on the pieces of the pencil (right
            the mode), on the whole space.
        mass: The factor of the pair (conjugate mode, mode) on the mass.
    """

    left: np.ndarray
    mass: float


class _Operator(spla.LinearOperator):
    """A complex ``size x size`` operator from its matrix-vector product."""

    def __init__(self, size: int, apply: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__(dtype=np.dtype(complex), shape=(size, size))
        self._apply = apply

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._apply(x)


def _values(
    form: ngs.BilinearForm, *, symmetric: bool
) -> tuple[np.ndarray, sp.csr_array]:
    """A form's stored values (the matrix's own order) and its full CSR matrix."""
    stored = form.mat.AsVector().FV().NumPy().copy()
    data, cols, rowptr = form.mat.CSR()
    n = form.mat.height
    matrix = sp.csr_array(
        (np.array(data, dtype=complex), np.array(cols), np.array(rowptr)),
        shape=(n, n),
    )
    if symmetric and sp.triu(matrix, 1).nnz == 0:
        # a symmetric form may store its lower triangle only
        matrix = (matrix + matrix.T - sp.diags_array(matrix.diagonal())).tocsr()
    return stored, matrix


class MechanicalSystem:
    r"""The assembled pencil and its eigensolver.

    Attributes:
        space: The compound space ``[u_x, u_y, u_z]``, plus the potential
            ``phi`` for a piezoelectric section.
        ndof: Its size.
        nu: The size of one displacement block.
        nphi: The size of the potential block (0 without coupling).
        free: Boolean mask of the free DOFs (the grounded potential and the
            gauge pin are not free).
        lossless: Whether :math:`\hat K` is Hermitian (no loss anywhere).
        matrix: The NGSolve matrix a factorization is written into.
    """

    def __init__(
        self,
        model: Model,
        problem: MechanicalEigenmode,
        options: SolverOptions,
        rule: MechanicalEigenmodeRule,
        *,
        assemble: bool = True,
    ) -> None:
        self.rule = rule
        self.problem = problem
        self.options = options
        order = options.order
        built = self.built = discretization(model.mesh).built
        mech = em_region(model, built, "mechanical")
        diel = em_region(model, built, "em")
        spaces = [
            h1_space(
                model,
                built,
                order,
                tuple(name for name, letters in problem.clamped if axis in letters),
                physics="mechanical",
                complex_valued=True,
            )
            for axis in "xyz"
        ]
        self.coupled = rule.coupled
        electrodes = rule.electrodes
        self.vp = None
        if electrodes is not None:
            self.vp = h1_space(
                model,
                built,
                order,
                electrodes.lines,
                physics="em",
                complex_valued=True,
            )
            spaces.append(self.vp)
        with serial_spaces():
            self.space = ngs.FESpace(spaces)
        self.nu = spaces[0].ndof
        self.nphi = self.vp.ndof if self.vp is not None else 0
        self.ndof = self.space.ndof
        self.free = free_mask(self.space.FreeDofs(), self.ndof)
        if electrodes is not None:
            self._ground_areas(model, electrodes.ground_areas)
        self._fix(self.free)
        #: The wavenumber in the mesh's length unit.
        self.q = problem.wavenumber * built.scale
        self.lossless = False
        self._model = model
        #: The elastic layer's stretch (`ElasticStretch`), or ``None``.
        self.stretch: ElasticStretch | None = None
        if rule.pml is not None and rule.layers is not None:
            f_ref = rule.pml.reference_frequency or problem.target
            self.stretch = ElasticStretch(model, built, rule.layers, rule.pml, f_ref)
            if assemble:
                self.stretch.warn_thin()
        self._dpencil: _StretchPencil | None = None
        self.matrix: ngs.BaseMatrix | None = None
        #: What a backward reads of the modes found (`modes`).
        self.kept: dict[str, np.ndarray] = {}
        self._handle: DirectHandle | None = None
        if assemble:
            self._assemble(model, rule, mech, diel)

    def _ground_areas(self, model: Model, names: tuple[str, ...]) -> None:
        """Hold the potential at zero on the grounded areas."""
        assert self.vp is not None  # noqa: S101
        for name in names:
            region = region_domain(model, self.built, name)
            mask = free_mask(self.vp.GetDofs(region), self.nphi)
            self.free[3 * self.nu :][mask] = False

    def _fix(self, free: np.ndarray) -> None:
        """Set the free DOFs, their bit array and their indices."""
        self.free = free
        self.free_bits = ngs.BitArray(free.tolist())
        #: The free DOFs' indices; the vectors of the eigensolver live there.
        self.index = np.flatnonzero(free)

    def release(self) -> None:
        """Free the factorization, if any."""
        if self._handle is not None:
            self._handle.free()
            self._handle = None

    # --- assembly ------------------------------------------------------------------

    def _pieces(self, group: str) -> tuple[tuple[str, complex], ...]:
        """The ``(piece, factor)`` pairs of ``group``, plus the potential's."""
        if group in _LOSS:
            return _LOSS[group]
        out = _GROUPS.get(group, ())
        if self.coupled:
            out += _COUPLED.get(group, ())
        if self.stretch is not None:
            out += _STRETCHED.get(group, ())
            if self.coupled:
                out += _STRETCHED_COUPLED.get(group, ())
        return out

    def _form(
        self, forms: dict[str, Any], names: tuple[str, ...], *, symmetric: bool
    ) -> tuple[ngs.BilinearForm, dict[str, tuple[np.ndarray, sp.csr_array]]]:
        """One form per group of ``names``, each assembled once.

        Returns the last form and each group's stored values and CSR matrix.
        A group's form also holds every other group's regions with a zero
        weight, so all of them share one matrix pattern (a symmetric form is
        for groups that are symmetric forms).
        """
        trial, test = self.space.TnT()
        u, v = tuple(trial), tuple(test)
        every: dict[int, tuple[Any, bool]] = {}
        for g in names:
            for piece, _ in self._pieces(g):
                if piece not in forms:
                    continue
                dx = forms[piece][1]
                every[id(dx)] = (dx, piece.startswith("dielectric"))
        out = {}
        pattern = None
        with ngs.TaskManager():
            for g in names:
                k = ngs.BilinearForm(self.space, symmetric=symmetric)
                held = set()
                for piece, factor in self._pieces(g):
                    if piece not in forms:
                        continue  # a piece the stretch leaves out (no loss)
                    form, dx = forms[piece]
                    k += (factor * form(u, v)) * dx
                    held.add(id(dx))
                # a zero-weight term on the regions the others hold, so the
                # patterns agree; Parameter keeps it from being folded away
                zero = ngs.Parameter(0.0)
                for key, (dx, potential) in every.items():
                    if key not in held:
                        j = 3 if potential else 0
                        k += (zero * (u[j] * v[j])) * dx
                k.Assemble()
                out[g] = _values(k, symmetric=symmetric)
                columns = np.array(k.mat.CSR()[1])
                if pattern is None:
                    pattern = columns
                elif not np.array_equal(pattern, columns):
                    raise RuntimeError(
                        "MechanicalEigenmode: the groups of the pencil have "
                        "different matrix patterns (internal error)."
                    )
        return k, out

    def _assemble(
        self, model: Model, rule: MechanicalEigenmodeRule, mech, diel
    ) -> None:
        """Assemble :math:`K_0`, :math:`K_1`, :math:`K_2` and :math:`M`."""
        arrays = _element_arrays(model)
        self.lossless = (
            rule.frame is None and self.stretch is None and not arrays["loss"].any()
        )
        self._co = _coefficients(model, self.built, rule.frame)
        self._lossy = bool(arrays["loss"].any())
        if self.stretch is None:
            forms = _weak_forms(
                self._co,
                self.built,
                self.options.order,
                mech,
                diel,
                wavenumber_pieces=True,
            )
        else:
            forms = self.stretch.forms(
                model,
                self._co,
                self.options.order,
                wavenumber_pieces=True,
                coupled=self.coupled,
                lossy=self._lossy,
            )
        extra = () if self.stretch is None else ("mp",)
        q = self.q
        d: sp.csr_array | None
        if q == 0:
            # only the order-0 forms enter the matrix that is factorized; the
            # group velocity is 0 there by reciprocity, so no first-order form
            form, a = self._form(forms, ("k0", "m", *extra), symmetric=True)
            stored = a["k0"][0]
            k = a["k0"][1]
            d = None
        else:
            form, a = self._form(forms, (*_GROUPS, *extra), symmetric=False)
            stored = a["k0"][0] - 1j * q * a["k1"][0] + q**2 * a["k2"][0]
            k = a["k0"][1] - 1j * q * a["k1"][1] + q**2 * a["k2"][1]
            d = -1j * a["k1"][1] + 2 * q * a["k2"][1]
        self._k, self._m = stored, a["m"][0]
        self.matrix = form.mat.CreateMatrix()
        if self.coupled and q == 0:
            self._pin(k)
        i = self.index
        self.k = k[i][:, i].tocsr()
        self.mass = a["m"][1][i][:, i].tocsr()
        #: The matrix the modes are normalized by: the physical regions' mass
        #: (the layer's is stretched, not energy).
        self.mass_norm = self.mass
        #: The displacement DOFs that touch a physical element.
        self.physical = np.ones(self.ndof, bool)
        if self.stretch is not None:
            full = a["mp"][1]
            self.mass_norm = full[i][:, i].tocsr()
            self.physical = full.diagonal() != 0
        # d/dq of the wavenumber in the mesh's unit is the mesh's unit per metre
        self.dk = None if d is None else (d * self.built.scale)[i][:, i].tocsr()
        self.k_norm = float(spla.norm(self.k, 1))
        self.m_norm = float(spla.norm(self.mass, 1))

    def _pin(self, k: sp.csr_array) -> None:
        r"""Pin one potential DOF in each ungrounded connected component.

        At :math:`q = 0` a constant potential is a null vector of
        :math:`K_0` and of :math:`M`. The components are those of the graph of
        the potential block of ``k``; one holding a fixed DOF is grounded.
        """
        start = 3 * self.nu
        block = k[start:, start:]
        count, label = sp.csgraph.connected_components(block, directed=False)
        free = self.free.copy()
        for c in range(count):
            dofs = np.flatnonzero(label == c)
            if free[start + dofs].all():
                free[start + dofs[0]] = False
        self._fix(free)

    # --- solves --------------------------------------------------------------------

    def _factor(
        self, shift: complex, *, own: bool = False, free: np.ndarray | None = None
    ) -> DirectHandle:
        r"""The factorization of :math:`A = \hat K - \text{shift}\,M`.

        With ``own`` the handle gets a matrix of its own, to outlive the next
        factorization; ``free`` replaces the free DOFs (a mask of the whole
        space).
        """
        assert self.matrix is not None  # noqa: S101
        matrix = self.matrix.CreateMatrix() if own else self.matrix
        matrix.AsVector().FV().NumPy()[:] = self._k - shift * self._m
        symmetric = self.q == 0
        return DirectHandle(
            matrix,
            self.free_bits if free is None else ngs.BitArray(free.tolist()),
            "complex_symmetric" if symmetric else "general",
            is_complex=True,
            cholesky=symmetric,
        )

    def _embed(self, x: np.ndarray) -> np.ndarray:
        """Free-DOF vector(s) ``x`` as vector(s) of the whole space."""
        out = np.zeros((self.ndof, *x.shape[1:]), complex)
        out[self.index] = x
        return out

    def _backward_error(self, lam: complex, x: np.ndarray, *, left: bool) -> float:
        r"""The backward error :math:`|A(\lambda) x| / ((|\hat K| + |\lambda||M|)|x|)`.

        With ``left``, that of :math:`A^T`.
        """
        a = self.k - lam * self.mass
        r = (a.T if left else a) @ x
        return float(
            np.linalg.norm(r)
            / ((self.k_norm + abs(lam) * self.m_norm) * np.linalg.norm(x))
        )

    def _search(
        self,
        handle: DirectHandle,
        sigma: complex,
        nev: int,
        *,
        transpose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Up to ``nev`` pairs :math:`(\lambda, x)` nearest ``sigma``, free DOFs only.

        ARPACK on :math:`A(\sigma)^{-1} M`, from a fixed start vector. With
        ``transpose`` the pairs of :math:`A(\sigma)^{-T} M^T`: the same
        eigenvalues, with the left vectors :math:`y`.
        """
        n = len(self.index)
        mass = self.mass.T if transpose else self.mass

        def apply(x: np.ndarray) -> np.ndarray:
            rhs = self._embed(mass @ np.asarray(x).reshape(-1))
            return handle.solve(rhs, transpose=transpose)[self.index]

        op = _Operator(n, apply)
        rng = np.random.default_rng(0)
        v0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        ncv = min(max(2 * nev + 1, 20), n - 1)
        try:
            theta, vectors = spla.eigs(
                op, k=nev, which="LM", tol=_ARPACK_TOLERANCE, ncv=ncv, v0=v0
            )
        except spla.ArpackNoConvergence as err:
            theta, vectors = err.eigenvalues, err.eigenvectors
        return sigma + 1 / theta, vectors

    def _stretch_pencil(self) -> _StretchPencil:
        r""":math:`\partial_g \hat K`, :math:`\partial_g M` and the loss :math:`L`.

        On the free DOFs. :math:`g` scales every stretch maximum: one assembly
        of the layer's pieces with the stretch factors replaced by their
        derivative. :math:`L` is the stretched loss of the stiffness,
        :math:`\hat K = \hat K_e + iL` (``None`` without a loss), one more
        assembly.
        """
        if self._dpencil is None:
            q = self.q
            assert self.stretch is not None  # noqa: S101
            i = self.index

            def assembled(names: tuple[str, ...], derivative: bool) -> list:
                assert self.stretch is not None  # noqa: S101
                forms = self.stretch.forms(
                    self._model,
                    self._co,
                    self.options.order,
                    wavenumber_pieces=q != 0,
                    coupled=self.coupled,
                    lossy=self._lossy,
                    derivative=derivative,
                )
                _, a = self._form(forms, names, symmetric=q == 0)
                k = a[names[0]][1]
                if q != 0:
                    k = k - 1j * q * a[names[1]][1] + q**2 * a[names[2]][1]
                out = [k[i][:, i].tocsr()]
                if "m" in a:
                    out.append(a["m"][1][i][:, i].tocsr())
                return out

            dk, dm = assembled(("k0", "m") if q == 0 else tuple(_GROUPS), True)
            loss = None
            if self._lossy:
                (loss,) = assembled(("l0",) if q == 0 else tuple(_LOSS), False)
            self._dpencil = (dk, dm, loss)
        return self._dpencil

    def sensitivities(
        self,
        found: list[tuple[complex, np.ndarray, float]],
        sigma: complex,
        nev: int,
    ) -> list[float]:
        r"""The stretch sensitivity of each mode of ``found`` (NaN: no left vector).

        .. math::

            \eta = \frac{|\partial_g\lambda|}
                {\max(|\mathrm{Im}(\lambda - \lambda_m)|,\
                \epsilon\,|\lambda|)}, \qquad
            \partial_g\lambda = \frac{y^T (\partial_g\hat K - \lambda\,
                \partial_g M)\,x}{y^T M x}, \qquad
            \lambda_m = \frac{i\,y^T L x}{y^T M x},

        where :math:`g` scales every stretch, :math:`\lambda_m` is the
        material loss' part of the eigenvalue (`_stretch_pencil`) and
        :math:`\epsilon` is `_UNDAMPED`. :math:`y` is :math:`x` at :math:`q = 0`
        and otherwise the left vector of a transposed search matched by its
        eigenvalue.

        The sensitivity is relative to the mode's decay rate without the
        material loss, not to :math:`|\lambda|`. The decay of a mode of the
        layer is made by the stretch: its eigenvalue is real at :math:`g = 0`
        and moves with :math:`g`, so :math:`|\partial_g\lambda| \gtrsim
        |\mathrm{Im}\,\lambda|` at :math:`g = 1` (a mode on a branch of the
        substrate with the complex length :math:`\ell = a + d(1 - ig\sigma_\max/3)`
        has :math:`\lambda - \lambda_c \propto \ell^{-2}`, so exactly
        :math:`|\ell|/\mathrm{Re}\,\ell \ge 1`), however close it is to the
        branch's cutoff :math:`\lambda_c`, where :math:`|\partial_g\lambda| /
        |\lambda|` is small. The radiation of a mode of the device does not depend
        on the stretch.
        """
        dk, dm, loss = self._stretch_pencil()
        if self.q == 0:
            lefts: list[np.ndarray | None] = [x for _, x, _ in found]
        else:
            assert self._handle is not None  # noqa: S101
            lam_t, vec_t = self._search(self._handle, sigma, nev, transpose=True)
            lefts = []
            for k, _, _ in found:
                j = int(np.argmin(np.abs(lam_t - k))) if len(lam_t) else -1
                near = j >= 0 and abs(lam_t[j] - k) <= _MATCH * abs(k)
                lefts.append(vec_t[:, j] if near else None)
        out = []
        for (k, x, _), y in zip(found, lefts, strict=True):
            if y is None:
                out.append(float("nan"))
                continue
            g = y @ (self.mass @ x)
            moved = abs((y @ (dk @ x) - k * (y @ (dm @ x))) / g)
            material = 0.0 if loss is None else 1j * (y @ (loss @ x)) / g
            decay = abs((k - material).imag)
            out.append(moved / max(decay, _UNDAMPED * abs(k)))
        return out

    def _layer_modes(
        self,
        found: list[tuple[complex, np.ndarray, float]],
        sigma: complex,
        nev: int,
    ) -> list[tuple[complex, np.ndarray, float]]:
        r"""The modes of ``found`` that the layer's stretch hardly moves.

        The sensitivity of a mode of the device is far below
        `_LAYER_SENSITIVITY`; that of a mode of the layer is about one or more.
        A mode with no left vector is dropped: the next, wider search matches
        it.
        """
        etas = self.sensitivities(found, sigma, nev)
        return [m for m, e in zip(found, etas, strict=True) if e <= _LAYER_SENSITIVITY]

    def _shifted(self, lam: complex) -> DirectHandle:
        r"""The factorization of :math:`A` at a shift next to :math:`\lambda`."""
        return self._factor(lam * (1 + _LEFT_SHIFT), own=True)

    def _refined(self, handle: DirectHandle, x: np.ndarray) -> np.ndarray:
        r"""``x`` after steps of inverse iteration on ``handle`` (`_shifted`).

        The backward error ARPACK is stopped on is relative to the norm of the
        whole pencil, which a piezoelectric section's stiffness dominates by
        many orders: the potential's rows of a coupled mode are left with an error
        that a gradient, first order in it, reads through the coupling. A step
        brings the error down by the ratio of the shift's distance to the
        eigenvalue's gap.
        """
        for _ in range(_REFINE_STEPS):
            x = handle.solve(self._embed(self.mass @ x))[self.index]
            x = x / np.linalg.norm(x)
        return x

    def _left(
        self,
        lam: complex,
        x: np.ndarray,
        tolerance: float,
        handle: DirectHandle | None,
        mode: int,
    ) -> np.ndarray:
        r"""The left vector :math:`y`, :math:`y^T A(\lambda) = 0`, of mode ``mode``.

        ``handle`` is the factorization of `_shifted`, for a lossy section at a
        wavenumber.

        The iteration runs to the round-off floor, not just to ``tolerance``:
        the group velocity and the gradient of the mode are first order in the
        left vector's error, and ``tolerance`` bounds a backward error relative
        to the whole pencil, whose stiffness norm (set by the mesh's highest
        modes) dwarfs :math:`|\lambda|\,|M|` at the mode. A vector at
        ``tolerance`` can still hold a part of a neighbouring mode's left
        vector that a gradient reads: a mode 1 % from its neighbours, stopped
        after one step at a backward error of :math:`2 \cdot 10^{-12}`, left a
        gradient :math:`5 \cdot 10^{-6}` off. Each step shrinks that part by
        the shift's distance over the gap (here :math:`10^{-8} / 10^{-2}`), so
        the next step removes it. The iteration stops once a step lowers the
        error by less than a factor 1/`_LEFT_STALL` (10), or after `_LEFT_STEPS`
        steps. It returns the iterate of least error, so a later step that drifts
        above ``tolerance`` does not discard an earlier one within it.

        Raises:
            EigenmodeConvergenceError: The iteration ends above ``tolerance``.
        """
        if self.lossless:
            return np.conj(x)
        if self.q == 0:
            return x
        assert handle is not None  # noqa: S101
        y = np.conj(x)
        error = np.inf
        best_error, best = np.inf, y
        for _ in range(_LEFT_STEPS):
            y = handle.solve(self._embed(self.mass.T @ y), transpose=True)[self.index]
            y = y / np.linalg.norm(y)
            previous, error = error, self._backward_error(lam, y, left=True)
            if error < best_error:
                best_error, best = error, y
            # Step 1 never stops: previous = inf, so the error always drops by
            # more than the stall factor and every lossy q != 0 mode takes two
            # solves at least.
            if error <= tolerance and error > _LEFT_STALL * previous:
                return best
        if best_error <= tolerance:
            return best
        error = best_error
        raise EigenmodeConvergenceError(
            f"MechanicalEigenmode: the left vector of mode {mode} did not reach "
            f"eigen_tolerance={tolerance:g} in {_LEFT_STEPS} steps (backward error "
            f"{error:.2g}), so its group velocity and gradient would be inexact. "
            "A neighbouring mode is closer than SolverOptions.cluster_rtol "
            "separates, or the section is badly scaled: loosen "
            "SolverOptions.eigen_tolerance or raise cluster_rtol."
        )

    def _vectors(
        self, lam: complex, x: np.ndarray, omega: complex, tolerance: float, mode: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The normalized right vector of a mode and its left vector.

        The refinement of a coupled mode and the left vector of a lossy mode at a
        wavenumber share one factorization.
        """
        iterated = not (self.lossless or self.q == 0)
        handle = self._shifted(lam) if self.coupled or iterated else None
        try:
            if handle is not None and self.coupled:
                x = self._refined(handle, x)
            v, _ = self._normalized(x, omega)
            return v, self._left(lam, v, tolerance, handle, mode)
        finally:
            if handle is not None:
                handle.free()

    def _converged(
        self, sigma: complex
    ) -> tuple[list[tuple[complex, np.ndarray, float]], float, int]:
        """The modes that meet the tolerance (and, with a layer, the filter).

        Searches with a growing ``nev`` until ``n_modes`` are found. Returns
        the last rung's modes, its best backward error and the number of
        converged modes the layer's stretch sensitivity dropped.
        """
        want, tolerance = self.problem.n_modes, self.options.eigen_tolerance
        handle = self._handle
        assert handle is not None  # noqa: S101
        found: list[tuple[complex, np.ndarray, float]] = []
        best = np.inf
        dropped = 0
        rungs: tuple[int, ...] = (want + 2, 2 * want + 8)
        if self.stretch is not None:
            n = len(self.index)
            rungs = (*rungs, 8 * want + 40)
            rungs = tuple(min(r, _NEV_CAP, n - 2) for r in rungs)
        for nev in rungs:
            lam, vectors = self._search(handle, sigma, nev)
            found, best = [], np.inf
            for k, x in zip(lam, vectors.T, strict=True):
                if abs(k) < _STATIC * abs(sigma):
                    continue  # a rigid motion
                error = self._backward_error(k, x, left=False)
                best = min(best, error)
                if error <= tolerance:
                    found.append((k, x, error))
            if self.stretch is not None:
                converged = len(found)
                found = self._layer_modes(found, sigma, nev)
                dropped = converged - len(found)
            if len(found) >= want:
                break
        return found, best, dropped

    def modes(self) -> dict[str, np.ndarray]:
        """The ``n_modes`` modes nearest the target, and their outputs.

        Raises:
            EigenmodeConvergenceError: Fewer than ``n_modes`` modes meet
                ``SolverOptions.eigen_tolerance`` after one widened retry.
        """
        problem, options = self.problem, self.options
        tolerance = options.eigen_tolerance
        target = problem.target
        sigma: complex = (2 * np.pi * target) ** 2
        if problem.target_quality_factor is not None:
            sigma *= 1 + 1j / problem.target_quality_factor
        want = problem.n_modes
        # the factorization of A(sigma): one, shared by the search and its retries
        self._handle = self._factor(sigma)
        found, best, dropped = self._converged(sigma)
        if len(found) < want:
            layer = (
                ""
                if self.stretch is None
                else f" and the layer's stretch sensitivity ({dropped} converged "
                "modes were modes of the layer and were dropped)"
            )
            raise EigenmodeConvergenceError(
                f"MechanicalEigenmode: {len(found)} of {want} modes near "
                f"{target:g} Hz met eigen_tolerance={tolerance:g}{layer} (the best "
                f"backward error was {best:.2g}). Move target closer to the wanted "
                "modes, or loosen SolverOptions.eigen_tolerance."
            )
        omega_all = np.sqrt(np.array([k for k, _, _ in found]))
        order = np.lexsort(
            (omega_all.imag, np.abs(omega_all.real / (2 * np.pi) - target))
        )
        found = [found[i] for i in order]
        omega_all = omega_all[order]
        # the labels run over the kept modes and the guard modes; a partner of a
        # kept mode among the guard modes makes it shared, and a partner the
        # eigensolver did not return, or dropped for its error, is not seen
        every = clusters(list(omega_all), options.cluster_rtol)
        labels = every[:want]
        shared = np.array([np.sum(every == label) > 1 for label in labels])
        out = {
            "omega": np.zeros(want, complex),
            "error": np.zeros(want),
            # a 3D body has no axial band: NaN, a 2D section's is 0 at q = 0
            "group_velocity": np.full(
                want, np.nan if self.built.mesh.dim == 3 else 0.0
            ),
            "cluster": labels,
            "displacement": np.zeros((want, 3, self.nu), complex),
            "potential": np.zeros((want, self.nphi), complex),
        }
        right = np.zeros((self.ndof, want), complex)
        left = np.zeros((self.ndof, want), complex)
        gram = np.zeros(want, complex)
        gauge = np.zeros(want, int)
        lams = np.zeros(want, complex)
        for i, (lam, x, _) in enumerate(found[:want]):
            v, y = self._vectors(lam, x, omega_all[i], tolerance, i)
            # the two-sided Rayleigh quotient: second order in the vectors' errors,
            # where the Ritz value is first order in its residual
            lams[i] = (y @ (self.k @ v)) / (y @ (self.mass @ v))
            if self.lossless:
                # Im of the quotient is round-off of either sign: a lossless
                # section has a real frequency, so its quality factor is inf
                lams[i] = lams[i].real
            omega = np.sqrt(lams[i])
            v, gauge[i] = self._normalized(v, omega)
            right[self.index, i], left[self.index, i] = v, y
            gram[i] = y @ (self.mass @ v)
            out["omega"][i] = omega
            # the returned vector's, not the search's
            out["error"][i] = self._backward_error(lams[i], v, left=False)
            if self.dk is not None:
                out["group_velocity"][i] = np.real(
                    (y @ (self.dk @ v)) / (2 * omega * (y @ (self.mass @ v)))
                )
            self._warn_gain(i, omega)
            full = self._embed(v)
            out["displacement"][i] = full[: 3 * self.nu].reshape(3, self.nu)
            out["potential"][i] = full[3 * self.nu :]
        self.kept = {
            "x": right,
            "y": left,
            "gram": gram,
            "gauge": gauge,
            "lambda": lams,
            "shared": shared,
        }
        return out

    def _warn_gain(self, mode: int, omega: complex) -> None:
        """Warn `LayerClearanceWarning` for a kept mode that grows in time."""
        if self.stretch is None or omega.imag >= -_GAIN * abs(omega):
            return
        warnings.warn(
            f"MechanicalEigenmode: mode {mode} at {omega.real / (2 * np.pi):.6g} Hz "
            "grows in time (Im omega < 0, Q = "
            f"{omega.real / (2 * omega.imag):.4g} with the gain sign) in a "
            "perfectly matched layer: its evanescent tail most likely reaches "
            "the layer, which then pumps it instead of absorbing it. Its Q is "
            "not physical. Move the layer farther from the structure.",
            LayerClearanceWarning,
            stacklevel=_mpi.outside_gfem(),
        )

    # --- the eigenvector adjoint -----------------------------------------------------

    def field_adjoint(
        self,
        lam: complex,
        x: np.ndarray,
        y: np.ndarray,
        gram: complex,
        x_bar: np.ndarray,
        *,
        gauge: int,
    ) -> FieldAdjoint:
        r"""Carry a cotangent of a normalized mode to the pencil.

        ``x`` is the mode as returned (`_normalized`), ``y`` its left vector,
        ``gram`` :math:`= y^T M x`, ``x_bar`` the cotangent of ``x``, all on
        the whole space, in the convention :math:`dL = \mathrm{Re}\,\bar
        x^T dx`, and ``gauge`` the position (among the free coefficients) of
        the coefficient `_normalized` made real. With :math:`A = \hat K -
        \lambda M`, a perturbation of the pencil moves the mode as
        :math:`dx = w + \beta x`, :math:`A w = -P\,dA\,x`,
        :math:`P = I - M x\,y^T/g` and :math:`dA = d\hat K - \lambda\,dM`;
        the complex :math:`\beta` holds the normalization
        :math:`\tfrac12 (\mathrm{Re}\,\omega)^2 x^H M x = 1` and the phase
        gauge :math:`\mathrm{Im}\,x_j = 0` at the coefficient :math:`j` =
        ``gauge``, the largest displacement coefficient. With
        :math:`c = \bar x^T x` and :math:`E = x^H M x`, eliminating :math:`\beta` gives

        .. math::

            dL = \mathrm{Re}\,\Big[\Big(\frac{\bar\omega_x\,y}{2\omega g}
            - P^T z\Big)^T dA\,x\Big] - \frac{\mathrm{Re}\,c}{2E}\,
            \mathrm{Re}\,(x^H dM\,x),

        with :math:`\bar\omega_x = -\mathrm{Re}\,c / \mathrm{Re}\,\omega`
        and :math:`A^T z = r`, :math:`r = \bar x - (\mathrm{Re}\,c/E)\,M\bar
        x^{*} - i\,(\mathrm{Im}\,c/x_j)\,e_j` (:math:`\bar x^{*}` the
        conjugate of the mode :math:`x`). That :math:`r^T x = 0` makes the
        singular system consistent, and :math:`P^T z` does not depend on the
        null vector :math:`y` added to :math:`z`: `_bordered` solves it.

        Returns:
            The left field of the pair on the pieces of :math:`dA` (right
            :math:`x`), and the factor of the pair :math:`(x^{*}, x)` on the
            mass alone.
        """
        i = self.index
        v, w, v_bar = x[i], y[i], x_bar[i]
        m = self.mass
        energy = float(np.real(np.vdot(v, m @ v)))
        c = complex(v_bar @ v)
        j = gauge
        r = v_bar - (c.real / energy) * (m @ np.conj(v))
        r[j] -= 1j * c.imag / v[j].real
        z = self._bordered(lam, v, w, r)
        z = z - w * ((v @ (m @ z)) / gram)
        omega = np.sqrt(lam)
        left = w * (-c.real / omega.real / (2 * omega * gram)) - z
        return FieldAdjoint(left=self._embed(left), mass=-c.real / (2 * energy))

    def _bordered(
        self, lam: complex, x: np.ndarray, y: np.ndarray, r: np.ndarray
    ) -> np.ndarray:
        r"""A solution of :math:`A(\lambda)^T z = r` for :math:`r^T x = 0`, free DOFs.

        The bordered system :math:`[A^T, e_k; e_k^T, 0]` is regular when
        :math:`x_k y_k \ne 0`: :math:`k` maximizes :math:`|x_k y_k|`, and the
        border holds :math:`z_k = 0` and drops row :math:`k`, whose equation
        holds by :math:`r^T x = 0` (its multiplier is :math:`x^T r / x_k`).
        One factorization of :math:`A` on the other free DOFs.
        """
        k = int(np.argmax(np.abs(x) * np.abs(y)))
        free = self.free.copy()
        free[self.index[k]] = False
        handle = self._factor(lam, own=True, free=free)
        try:
            return handle.solve(self._embed(r), transpose=True)[self.index]
        finally:
            handle.free()

    def _normalized(self, x: np.ndarray, omega: complex) -> tuple[np.ndarray, int]:
        r"""``x`` scaled to :math:`\tfrac12 (\mathrm{Re}\,\omega)^2 x^H M x = 1`.

        The largest displacement coefficient is real and positive; the
        potential is scaled with it. Returns the vector and the position of
        that coefficient (the phase gauge's), which the eigenvector adjoint
        reads rather than seek it again: on a symmetric section mirror
        coefficients tie to round-off, and another search could pick the
        other one.
        """
        energy = np.real(np.vdot(x, self.mass_norm @ x))
        x = x * np.sqrt(2 / (omega.real**2 * energy))
        shown = np.flatnonzero((self.index < 3 * self.nu) & self.physical[self.index])
        j = int(shown[np.argmax(np.abs(x[shown]))])
        largest = x[j]
        return x * (np.conj(largest) / abs(largest)), j
