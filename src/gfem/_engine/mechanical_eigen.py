r"""Mechanical eigenmodes of a 2D section: the pencil :math:`\hat K(q) - \omega^2 M`.

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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from gfem._checks import is_traced
from gfem._eigen.reach import clusters
from gfem._engine.cache import Entry, discretization
from gfem._engine.domain import em_region, h1_space
from gfem._engine.piezoelectric import (
    _check_absorbing,
    _check_clamped,
    _check_frame,
    _check_materials,
    _check_mechanical,
    _coefficients,
    _element_arrays,
    _weak_forms,
)
from gfem._engine.runtime import serial_spaces, task_manager
from gfem._engine.sensitivity import FieldPair
from gfem._engine.solve_request import SolveRequest
from gfem._field import FieldMap
from gfem._linalg.base import free_mask
from gfem._linalg.direct import DirectHandle
from gfem.errors import EigenmodeConvergenceError
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

#: The assembled groups: real forms by power of the wavenumber, and the mass.
_GROUPS = {
    "k0": (("stiffness", 1.0), ("loss", 1j)),
    "k1": (("stiffness1", 1.0), ("loss1", 1j)),
    "k2": (("stiffness2", 1.0), ("loss2", 1j)),
    "m": (("mass", 1.0),),
}


class MechanicalEigenmodeRule:
    """Host-side forward of a `MechanicalEigenmode` problem on a 2D `Model`.

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
    differentiable_outputs: frozenset[str] = frozenset()

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
            NotImplementedError: ``linear_solver="petsc"``, or a model with a
                piezoelectric tensor.
            RegionNotFoundError: A name is not a region of the mesh.
            ValueError: The problem does not fit the model; the message gives
                the remedy.
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
        if self.coupled:
            raise NotImplementedError(
                "A MechanicalEigenmode solve of a model with a piezoelectric "
                "tensor is not available in this version: leave the "
                "piezoelectric_tensor out of the materials of the mechanical "
                "domain, or solve the admittance with gfem.Piezoelectric."
            )
        mesh = model.mesh
        mechanical = model._domain_of("mechanical")  # noqa: SLF001
        #: The areas with elements in the mechanical domain, mesh order.
        self.mechanical_regions = tuple(
            r for r in mesh.volumes if mechanical.active[mesh.elements(r)].any()
        )
        self._mesh = mesh
        self._mechanical_excluded = mechanical.excluded

    def forward(
        self,
        model: Model,
        *,
        keep: bool,  # noqa: ARG002
        request: SolveRequest | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], Entry | None]:
        """Find the modes nearest the target.

        Returns:
            ``{"omega": (n,) rad/s, "error": (n,), "group_velocity": (n,) m/s,
            "cluster": (n,), "displacement": (n, 3, n_dof) m, "potential":
            (n, 0) V}``, the modes nearest first, and no entry (nothing is
            differentiated yet).

        Raises:
            EigenmodeConvergenceError: Fewer than ``n_modes`` modes meet
                ``SolverOptions.eigen_tolerance``.
        """
        with task_manager(self.options.threads, self.comm):
            system = MechanicalSystem(model, self.problem, self.options, self)
            try:
                found = system.modes()
            finally:
                system.release()
        return found, None

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
            potential_fields=(),
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
            "potential": jax.ShapeDtypeStruct((n, 0), jnp.complex128),
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
        return (
            f"gfem does not differentiate the MechanicalEigenmode output {name!r} "
            "in this version."
        )

    def backward(  # pragma: no cover
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        wanted: frozenset | None = None,
    ) -> list[FieldPair]:
        """No output is differentiable yet: `differentiable_outputs` is empty."""
        raise NotImplementedError("A MechanicalEigenmode solve has no backward yet.")


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
        space: The compound space ``[u_x, u_y, u_z]``.
        ndof: Its size.
        nu: The size of one displacement block.
        free: Boolean mask of the free DOFs.
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
        with serial_spaces():
            self.space = ngs.FESpace(spaces)
        self.nu = spaces[0].ndof
        self.ndof = self.space.ndof
        self.free = free_mask(self.space.FreeDofs(), self.ndof)
        self.free_bits = ngs.BitArray(self.free.tolist())
        #: The free DOFs' indices; the vectors of the eigensolver live there.
        self.index = np.flatnonzero(self.free)
        #: The wavenumber in the mesh's length unit.
        self.q = problem.wavenumber * built.scale
        self.lossless = False
        self.matrix: ngs.BaseMatrix | None = None
        self._handle: DirectHandle | None = None
        if assemble:
            self._assemble(model, rule, mech, diel)

    def release(self) -> None:
        """Free the factorization, if any."""
        if self._handle is not None:
            self._handle.free()
            self._handle = None

    # --- assembly ------------------------------------------------------------------

    def _form(
        self, forms: dict[str, Any], names: tuple[str, ...], *, symmetric: bool
    ) -> tuple[ngs.BilinearForm, dict[str, tuple[np.ndarray, sp.csr_array]]]:
        """One form of the groups ``names``, assembled once per group.

        Returns the form and each group's stored values and CSR matrix. One
        form keeps the groups on one pattern; a symmetric one is for
        the groups that are symmetric forms.
        """
        trial, test = self.space.TnT()
        u, v = tuple(trial), tuple(test)
        switches = {g: ngs.Parameter(0.0) for g in names}
        out = {}
        with ngs.TaskManager():
            k = ngs.BilinearForm(self.space, symmetric=symmetric)
            for g in names:
                for piece, factor in _GROUPS[g]:
                    form, dx = forms[piece]
                    k += switches[g] * (factor * form(u, v)) * dx
            for g in names:
                for h, switch in switches.items():
                    switch.Set(1.0 if h == g else 0.0)
                k.Assemble()
                out[g] = _values(k, symmetric=symmetric)
        return k, out

    def _assemble(
        self, model: Model, rule: MechanicalEigenmodeRule, mech, diel
    ) -> None:
        """Assemble :math:`K_0`, :math:`K_1`, :math:`K_2` and :math:`M`."""
        arrays = _element_arrays(model)
        self.lossless = rule.frame is None and not arrays["loss"].any()
        forms = _weak_forms(
            _coefficients(model, self.built, rule.frame),
            self.built,
            self.options.order,
            mech,
            diel,
            wavenumber_pieces=True,
        )
        q = self.q
        if q == 0:
            # only the order-0 forms enter the matrix that is factorized; the
            # first-order form is for the group velocity alone
            form, a = self._form(forms, ("k0", "m"), symmetric=True)
            _, b = self._form(forms, ("k1",), symmetric=False)
            stored = a["k0"][0]
            k = a["k0"][1]
            d = -1j * b["k1"][1]
        else:
            form, a = self._form(forms, tuple(_GROUPS), symmetric=False)
            stored = a["k0"][0] - 1j * q * a["k1"][0] + q**2 * a["k2"][0]
            k = a["k0"][1] - 1j * q * a["k1"][1] + q**2 * a["k2"][1]
            d = -1j * a["k1"][1] + 2 * q * a["k2"][1]
        self._k, self._m = stored, a["m"][0]
        self.matrix = form.mat.CreateMatrix()
        i = self.index
        self.k = k[i][:, i].tocsr()
        self.mass = a["m"][1][i][:, i].tocsr()
        # d/dq of the wavenumber in the mesh's unit is the mesh's unit per metre
        self.dk = (self.built.scale * d)[i][:, i].tocsr()
        self.k_norm = float(spla.norm(self.k, 1))
        self.m_norm = float(spla.norm(self.mass, 1))

    # --- solves --------------------------------------------------------------------

    def _factor(self, shift: complex, *, own: bool = False) -> DirectHandle:
        r"""The factorization of :math:`A = \hat K - \text{shift}\,M`.

        With ``own`` the handle gets a matrix of its own, to outlive the next
        factorization.
        """
        assert self.matrix is not None  # noqa: S101
        matrix = self.matrix.CreateMatrix() if own else self.matrix
        matrix.AsVector().FV().NumPy()[:] = self._k - shift * self._m
        symmetric = self.q == 0
        return DirectHandle(
            matrix,
            self.free_bits,
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
        self, handle: DirectHandle, sigma: float, nev: int
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Up to ``nev`` pairs :math:`(\lambda, x)` nearest ``sigma``, free DOFs only.

        ARPACK on :math:`A(\sigma)^{-1} M`, from a fixed start vector.
        """
        n = len(self.index)

        def apply(x: np.ndarray) -> np.ndarray:
            rhs = self._embed(self.mass @ np.asarray(x).reshape(-1))
            return handle.solve(rhs)[self.index]

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

    def _left(self, lam: complex, x: np.ndarray, tolerance: float) -> np.ndarray:
        r"""The left vector :math:`y`, :math:`y^T A(\lambda) = 0`, of a mode."""
        if self.lossless:
            return np.conj(x)
        if self.q == 0:
            return x
        handle = self._factor(lam * (1 + _LEFT_SHIFT), own=True)
        try:
            y = np.conj(x)
            for _ in range(_LEFT_STEPS):
                y = handle.solve(self._embed(y), transpose=True)[self.index]
                y = y / np.linalg.norm(y)
                if self._backward_error(lam, y, left=True) <= tolerance:
                    break
        finally:
            handle.free()
        return y

    def modes(self) -> dict[str, np.ndarray]:
        """The ``n_modes`` modes nearest the target, and their outputs.

        Raises:
            EigenmodeConvergenceError: Fewer than ``n_modes`` modes meet
                ``SolverOptions.eigen_tolerance`` after one widened retry.
        """
        problem, options = self.problem, self.options
        tolerance = options.eigen_tolerance
        target = problem.target
        sigma = (2 * np.pi * target) ** 2
        want = problem.n_modes
        # the factorization of A(sigma): one, shared by the search and its retry
        self._handle = self._factor(sigma)
        found: list[tuple[complex, np.ndarray, float]] = []
        best = np.inf
        for nev in (want + 2, 2 * want + 8):
            lam, vectors = self._search(self._handle, sigma, nev)
            found, best = [], np.inf
            for k, x in zip(lam, vectors.T, strict=True):
                if abs(k) < _STATIC * sigma:
                    continue  # a rigid motion
                error = self._backward_error(k, x, left=False)
                best = min(best, error)
                if error <= tolerance:
                    found.append((k, x, error))
            if len(found) >= want:
                break
        if len(found) < want:
            raise EigenmodeConvergenceError(
                f"MechanicalEigenmode: {len(found)} of {want} modes near "
                f"{target:g} Hz met eigen_tolerance={tolerance:g} (the best "
                f"backward error was {best:.2g}). Move target closer to the wanted "
                "modes, or loosen SolverOptions.eigen_tolerance."
            )
        omega_all = np.sqrt(np.array([k for k, _, _ in found]))
        order = np.lexsort(
            (omega_all.imag, np.abs(omega_all.real / (2 * np.pi) - target))
        )
        found = [found[i] for i in order]
        omega_all = omega_all[order]
        labels = clusters(list(omega_all), options.cluster_rtol)[:want]
        out = {
            "omega": np.zeros(want, complex),
            "error": np.zeros(want),
            "group_velocity": np.zeros(want),
            "cluster": labels,
            "displacement": np.zeros((want, 3, self.nu), complex),
            "potential": np.zeros((want, 0), complex),
        }
        for i, (lam, x, error) in enumerate(found[:want]):
            omega = omega_all[i]
            v = self._normalized(x, omega)
            y = self._left(lam, v, tolerance)
            out["omega"][i] = omega
            out["error"][i] = error
            out["group_velocity"][i] = np.real(
                (y @ (self.dk @ v)) / (2 * omega * (y @ (self.mass @ v)))
            )
            out["displacement"][i] = self._embed(v).reshape(3, self.nu)
        return out

    def _normalized(self, x: np.ndarray, omega: complex) -> np.ndarray:
        r"""``x`` scaled to :math:`\tfrac12 (\mathrm{Re}\,\omega)^2 x^H M x = 1`.

        The largest coefficient is real and positive.
        """
        energy = np.real(np.vdot(x, self.mass @ x))
        x = x * np.sqrt(2 / (omega.real**2 * energy))
        largest = x[np.argmax(np.abs(x))]
        return x * (np.conj(largest) / abs(largest))
