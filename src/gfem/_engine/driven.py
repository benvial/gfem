r"""Driven rule: time-harmonic E field in complex Nédélec elements, lumped ports.

At each frequency the operator ``A(ω) = Σ θ_k(ω) A_k`` (see
`gfem._engine.terms`) is assembled and prepared once, and every excited
port's right-hand side is solved in one handle call: GMRES + BDDC by
default, each port starting from its solution at the previous frequency
(at a uniform sweep's first, on one process, from the previous design's
there), or one LDL^T factorization (``direct``: MKL PARDISO with
the ``mkl`` extra, NGSolve sparse Cholesky otherwise). The system operator
(`gfem._engine.system_operator.SystemOperator`) answers the forward and
adjoint solves. A sweep keeps one base handle: a direct solver plans at the
first frequency and refactorizes at the others (on the plan pool, one base per
frequency met before, kept across designs), and sums each frequency's
matrix from the pieces ``A_k`` assembled once per forward (the piece sum,
kept with the system operator for the backward pass) instead of assembling
it.
With ``SolverOptions.frequency_workers`` above 1, a uniform sweep splits
into contiguous blocks of frequencies, each swept the same way in a
frequency worker (`gfem._engine.workers`), joined in frequency order; the
backward pass stays in this process. Without a Bloch wave vector ``A``
is complex symmetric (A^T = A), and a port's excitation vector is
proportional to its voltage functional, so S is symmetric to solver
precision. With one (``Driven.wave_vector``) the operator is general:
A(k)^T = A(-k), so S(k)^T = S(-k), and the direct solver factors it by LU
(MKL PARDISO, or NGSolve's UMFPACK without the ``mkl`` extra).
At a fixed angle (``Driven.wave_vector_frequency``) the wave vector scales
with frequency, k(f) = k f / f_ref: each frequency is solved by a
single-frequency rule of its own, with its Bloch space, operator,
factorization and transposed system at ±k(f).

S is read from each port's outgoing wave: a lumped port's is the power wave
:math:`V_i / \sqrt{2 R_i}` with reference impedance :math:`R_i`; a wave port's is its
pseudo-wave, the biorthogonal modal amplitude on its surface's modes times
the port's pseudo-wave factor (`gfem._engine.wave_ports`), at the complex
``Z_PV`` when it has a ``voltage_path``. The excited port is fed an incident
wave of amplitude 1 (1 W on a lossless line), and ``S_ij`` is port i's
outgoing wave minus :math:`\delta_{ij}`. The solution
itself (volts per mesh file unit) is that 1 W field's peak phasor in the
`gfem.Field` coefficient convention, so kept fields are the solution
columns as they are.

Gradients (`DrivenRule.backward`): with ``A x_j = b_j`` and port i's S
functional ``f_i`` parallel to its excitation ``b_i`` (every lumped port and every
wave port, lossy or not, lone or on a shared surface: by reciprocity the
unconjugated reactions between a surface's modes vanish, so the biorthogonal
functional is the port's own load scaled), port i's solution is (a
multiple of) the adjoint of its own S row, ``A^-T f_i = A^-1 f_i = c x_i``
(A^T = A, no transpose, no conjugate). An S cotangent therefore needs no
solve when every port it reads is excited and parallel: per frequency it
folds into one pair ``(-sum_i W_ij A^-T f_i, x_j)``. A kept-field
cotangent or an S row of a port that was not excited takes one
multi-column adjoint solve per frequency with the system operator
(wave-port modal corrections included); that frequency's operator is
prepared on first use at the adjoint tolerance and kept in the cache entry
for later pullbacks. With a Bloch wave vector A(k)^T = A(-k): there is no
reciprocity shortcut, and every adjoint load is solved on the transposed
system, the system operator at -k (`SystemOperator.solve_adjoint`). On a
periodic model the pairs go to the sensitivity kernel on the unpaired
space (`DrivenOperator.pair`): ``A = P^H A_plain P``, so ``lambda^T (dA)
x = (conj(P) lambda)^T (dA_plain) (P x)``. Wave
ports' modes follow a design at their faces: the excitations, the S
functionals, the modal corrections and the Robin term read the modes, so
the held pairs miss their derivative, which a port pass adds per surface
whose section reads a wanted input (`gfem._engine.port_pass`).

An adaptive sweep differentiates with its reduced basis V and sample set
held fixed (`DrivenRule._backward_adaptive`): the cache entry keeps only V
and the reduced model, the backward solves n x n reduced adjoints and folds
them into at most n field pairs ``(V a, V b)`` per material term. Each
sample also solves the S functionals whose adjoint the solutions miss
(`PortNetwork.adjoint_loads`: wave ports, and lumped ports that are not
excited) into V, so the frozen-basis gradient equals the full solve's at
the samples. Under a JAX
transformation the sample set must be frozen (``Driven.samples``), so
every output shape is known before the solve.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import functools
import math
from collections.abc import Callable, Collection, Generator, Hashable, Mapping, Sequence
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.sparse

from gfem import _floquet, _mpi, _pml, _progress, _timing
from gfem._adjoint import adjoint_tolerance
from gfem._boundaries import (
    Absorbing,
    FloquetPort,
    Impedance,
    LumpedPort,
    Periodic,
    WavePort,
    sheet_of,
    surfaces_of,
    wave_port_label,
    wave_port_surfaces,
)
from gfem._checks import is_traced
from gfem._domain import nedelec_unknowns
from gfem._engine import dispersion, sweep_outputs, terms
from gfem._engine.cache import (
    ENTRIES,
    TEMPLATE_TOL,
    Discretization,
    Entry,
    default_retain_memory,
    discretization,
    memory_limit,
    retain_budget,
)
from gfem._engine.dipoles import check_positions
from gfem._engine.domain import (
    element_dofs,
    em_region,
    hcurl_space,
    pec_surfaces,
    space_key,
)
from gfem._engine.driven_sweep import (
    FixedAngleChildren,
    ReducedKept,
    SweepSplit,
    raise_unsolved,
)
from gfem._engine.driven_system import DrivenSystemBuilder
from gfem._engine.fields import dof_shape, rank_piece
from gfem._engine.floquet_ports import check_excitable
from gfem._engine.gauge import GaugeChoice, resolve_gauge
from gfem._engine.incident import (
    check_contact,
    check_incident_model,
    layer_regions,
    truncation_designs,
)
from gfem._engine.mesh import EngineMesh
from gfem._engine.port_network import (
    PortNetwork,
    PortSolutions,
    complex_reference,
    parallel_scale,
    resistance,
    split_cotangent,
)
from gfem._engine.port_pass import PortPass, PortResult
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import (
    FieldPair,
    Integrated,
    Term,
    is_lumped,
    is_sheet,
    is_thermal_value,
    law_key,
)
from gfem._engine.solve_request import SolveRequest
from gfem._engine.source_columns import (
    ColumnContext,
    SourceColumns,
    real_up_to_phase,
)
from gfem._engine.sweep_outputs import SweepOutputs
from gfem._engine.system_operator import FixedPieces, SystemOperator, SystemSweep
from gfem._engine.wave_ports import PortSurfaceCoupling
from gfem._incident import IncidentPart
from gfem._linalg import (
    PMG_BYTES_PER_UNKNOWN,
    Kind,
    PmgInputs,
    ResolvedLinearSolver,
    auto_linear_solver,
    report,
)
from gfem._linalg.layout import DofLayout, RankLayout
from gfem._linalg.pieces import KeptParts
from gfem._linalg.pmg import Cells, DirectSetCount, direct_set_count
from gfem._materials import MaterialField
from gfem._periodic import (
    bloch_phases,
    check_bloch,
    check_floquet,
    check_wave_vector,
)
from gfem._pml import Stretch
from gfem._sources import DIPOLES, INCIDENT, MagneticDipole
from gfem._sweep.adaptive import AffineSystem, AffineTerm, Sweep, adaptive_sweep
from gfem._text import hertz
from gfem.errors import (
    FloquetOrderCapWarning,
    LinearSolverConvergenceError,
    LowFrequencyWarning,
    ObliqueAbsorberWarning,
    PortGeometryError,
    SkippedPortModeWarning,
    ThinLayerWarning,
    UnrequestedOrderWarning,
)
from gfem.mesh import Mesh, _at_imported_nodes, _shortest_edges
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Driven
from gfem.results import (
    DiffractionOrderResult,
    DrivenResult,
    Field,
    NetworkParameters,
    PortModeResult,
    ReducedFields,
    SweepInfo,
)

__all__ = [
    "DEFAULT_SOLVER",
    "FORWARD_WARNINGS",
    "LOW_FREQUENCY_FLOOR",
    "DrivenOperator",
    "DrivenRule",
    "Expansion",
    "LowFrequencyFloor",
    "check_low_frequency",
    "check_ports_held",
    "driven_operator",
    "low_frequency_deferred",
    "operator_terms",
    "real_space",
]

#: Whether A^T = A (no Bloch phase) or not: a driven operator is never
#: Hermitian.
DrivenSymmetry = Literal["complex_symmetric", "general"]

#: The linear solver for ``linear_solver="auto"`` when neither direct nor the
#: p-multigrid fits, or the p-multigrid does not solve the model
#: (`gfem._linalg.auto_linear_solver`).
DEFAULT_SOLVER: Kind = "bddc_gmres"

#: k0 * h_min below which a driven solve warns (gfem.LowFrequencyWarning).
LOW_FREQUENCY_FLOOR = 1e-6
#: k0 * h_min below which MKL PARDISO becomes unreliable (with iparm(10) =
#: 16; MUMPS and NGSolve's sparse Cholesky hold to 1e-8, the lowest point
#: measured), named in the low-frequency warning.
_PARDISO_FLOOR = 3e-8


@dataclasses.dataclass(frozen=True, eq=False)
class Expansion:
    """A periodic space's fields on its unpaired space: e = P x.

    Attributes:
        matrix: P, the periodic space's coefficients to the unpaired
            space's (`expansion_matrix`).
        space: The complex unpaired space (P's rows): where the backward
            pass's field pairs live.
        field_rows: Under ranks, the rows of ``space`` that a local volume
            element uses (`gfem._engine.domain.element_dofs`): the plain
            mesh's field layout, in its order, so results keep ``P x`` on
            those rows. ``None`` on one process: every row.
        lattice: Builds `lattice_vectors`; ``None`` without a Bloch phase
            and under ranks.
    """

    matrix: scipy.sparse.csr_matrix
    space: ngs.FESpace
    field_rows: np.ndarray | None = None
    lattice: Callable[[], np.ndarray | None] | None = None

    @functools.cached_property
    def lattice_vectors(self) -> np.ndarray | None:
        r"""The lattice vector :math:`\mathbf{m}` of each stored entry of P, (nnz, 3).

        In metres, in ``matrix``'s CSR order: the entry is :math:`g\,
        e^{-i\,\mathbf{m}\cdot\mathbf{k}}` with :math:`g` real, so
        :math:`\partial P/\partial k_d = -i\,m_d \circ P` entry by entry.
        From the discretization's phase template of P
        (`gfem._engine.cache.Discretization.phase_template`: one more
        conversion the first time a mesh needs it). ``None`` without a
        Bloch phase, under ranks, and where the template does not stand for
        ``matrix`` (`gfem._engine.cache.TEMPLATE_TOL`).
        """
        return None if self.lattice is None else self.lattice()

    @functools.cached_property
    def field(self) -> scipy.sparse.csr_matrix:
        """P on the field rows: solve-space coefficients to a result field's."""
        if self.field_rows is None:
            return self.matrix
        return self.matrix[self.field_rows]


@dataclasses.dataclass(eq=False)
class DrivenOperator:
    """The driven operator of a model, as pieces over one H(curl) space.

    Attributes:
        built: The engine mesh.
        space: Complex first-kind Nédélec space, `PEC` DOFs fixed.
        pieces: The affine pieces (`gfem._engine.terms.Piece`).
        ports: Lumped port geometry, in model port order.
        order: Polynomial order.
        expansion: For a model with periodic pairs, the `Expansion` of
            the solve space onto the unpaired space; ``None`` without pairs,
            where the two spaces are one.
        symmetry: ``"general"`` on a Bloch-phased space (A^T != A),
            ``"complex_symmetric"`` otherwise: declared by the problem's
            wave vector, never read from its value.
        layout: How ``space`` lies across ranks; ``None`` on one process.
        fixed: How the volume pieces split at the material-field regions,
            for a piece sum that keeps their other part; ``None``
            when nothing is fixed, restricted, or under ranks.
        topology: The mesh topology ``space`` lies on; with ``space_key``,
            what a solver plan of the operator is pooled under.
        space_key: ``space``'s key in its discretization (order, `PEC`
            surfaces, periodic pairs and phases).
        latent: Pieces whose matrix is zero at this design, so that no
            forward assembles them, but whose derivative is not: the
            conductivity piece where the conductivity is zero everywhere.
            Only a backward pass reads them (`weights`, `operator_terms`).
        cells: Builds the cells ``space`` is defined on, with the
            perfectly matched layer's marked (`solve_cells`), for the
            ``pmg_gmres`` linear solver; ``None`` under ranks.
    """

    built: EngineMesh
    space: ngs.FESpace
    pieces: list[terms.Piece]
    ports: tuple[terms.LumpedPortData, ...]
    order: int
    expansion: Expansion | None = None
    symmetry: DrivenSymmetry = "complex_symmetric"
    layout: DofLayout | None = None
    fixed: FixedPieces | None = None
    topology: Any = None
    space_key: tuple = ()
    latent: tuple[terms.Piece, ...] = ()
    cells: Callable[[], Cells] | None = None

    def expand(self, x: np.ndarray) -> np.ndarray:
        """``P x``: solve-space coefficients ``(ndof, ...)`` as a result field's.

        Under ranks, the rows of P a local volume element uses
        (`Expansion.field_rows`).
        """
        return x if self.expansion is None else self.expansion.field @ x

    @property
    def field_ndof(self) -> int:
        """The rows of a result field on this rank: the rows `expand` gives."""
        if self.expansion is None:
            return int(self.space.ndof)
        return int(self.expansion.field.shape[0])

    def pair(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        weights: Mapping[str, complex],
        domain: ngs.Region,
    ) -> FieldPair:
        """The `FieldPair` of solve-space columns ``(left, right)``.

        ``left`` holds adjoints (on the transposed system's space under a
        Bloch wave vector), ``right`` solutions. Without periodic pairs both
        stay on the solve space. With pairs both go to the unpaired space,
        ``(conj(P) left, P right)``: the periodic operator is ``P^H A_plain
        P`` and ``conj(P(k)) = P(-k)``, so ``left^T (dA) right`` is the
        plain form of the expanded fields, which the kernel contracts as it
        does without pairs (and a real space can hold under a Bloch phase,
        `real_space`).
        """
        if self.expansion is None:
            return FieldPair(
                space=self.space,
                left=left,
                right=right,
                weights=weights,
                order=self.order,
                domain=domain,
            )
        p = self.expansion.matrix
        return FieldPair(
            space=self.expansion.space,
            left=p.conj() @ left,
            right=p @ right,
            weights=weights,
            order=self.order,
            domain=domain,
        )

    def form(
        self, omega: complex, extra: Sequence[terms.Piece] = ()
    ) -> ngs.BilinearForm:
        """``Σ θ_k(ω) a_k``, not assembled (a backend assembles it).

        With the pieces ``extra`` beside `pieces` (an eigen operator's wave
        port Robin pieces), at a complex ``omega`` there.
        """
        u, v = self.space.TnT()
        form = ngs.BilinearForm(self.space, symmetric=self.symmetric)
        for piece in (*self.pieces, *extra):
            form += complex(piece.theta(omega)) * piece.integrand(u, v)
        return form

    @property
    def symmetric(self) -> bool:
        """Whether A^T = A (no Bloch phase): the forms' ``symmetric`` flag."""
        return self.symmetry != "general"

    def weights(self, omega: complex, *, laws: bool = True) -> dict[str, complex]:
        """θ(ω) of each piece a gradient moves, by piece name (`gfem._engine.terms`).

        The `FieldPair.weights` of a backward pass, driven (real ω) or
        eigenmode (the complex eigenvalue's ω). The sensitivity kernel looks
        them up by name: its terms are the pieces, under their names
        (`operator_terms`, and `lumped_terms` for the lumped values). Of the
        ports' pieces only a lumped port's L and C pieces move (with their
        values); its R piece, its surface and a wave port's mode
        are fixed, so no pair carries them. The `latent` pieces are
        included: their derivative is not zero. A permittivity law's pieces
        add dθ/dp by each law parameter, each under its own key
        (`gfem._engine.sensitivity.law_key`, the law pass), unless
        ``laws`` is false (no law parameter is a wanted input).
        """
        out = {
            p.name: complex(p.theta(omega))
            for p in (*self.pieces, *self.latent)
            if p.kind != "port" or p.lumped is not None
        }
        for p in self.pieces if laws else ():
            if p.law is not None:
                for parameter, value in p.law[1](omega).items():
                    out[law_key(p.name, parameter)] = complex(value)
        return out

    def wavenumber(self, omega: float) -> float:
        """k0 per mesh file unit."""
        return terms.wavenumber(omega, self.built.scale)

    @functools.cached_property
    def _piece_forms(self) -> list[ngs.BilinearForm]:
        u, v = self.space.TnT()
        forms = []
        for piece in self.pieces:
            form = ngs.BilinearForm(
                self.space, symmetric=self.symmetric, nonassemble=True
            )
            form += piece.integrand(u, v)
            forms.append(form)
        return forms

    def apply(self, k: int, x: np.ndarray) -> np.ndarray:
        """``A_k X`` of piece ``k`` (its form without θ), matrix-free.

        ``X`` is ``(ndof,)`` or ``(ndof, m)``; fixed DOFs are not special.
        Under ranks ``X`` is a state (every holder of a DOF holds its value)
        and the result this rank's part of a load.
        """
        form = self._piece_forms[k]
        # no reshape to -1: it raises on a rank with no DOFs
        columns = x[:, None] if x.ndim == 1 else x
        source = ngs.GridFunction(self.space).vec
        target = source.CreateVector()
        out = np.empty(columns.shape, complex)
        for j, column in enumerate(columns.T):
            source.FV().NumPy()[:] = column
            if self.layout is not None:
                # a state in, this rank's part of the load out
                self.layout.mark_state(source)
            form.Apply(source, target)
            out[:, j] = target.FV().NumPy()
        return out.reshape(x.shape)


def driven_operator(
    model: Model,
    order: int,
    *,
    wave_vector: tuple[float, float, float] | None = None,
    elements: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    paired: bool = True,
    comm: Any = None,
    swept: bool = False,
    stretch: Stretch | None = None,
) -> DrivenOperator:
    """The driven operator of ``model`` at its current node positions.

    Every `PEC` surface is a Dirichlet constraint of the space; every
    other boundary and lumped port contributes pieces. ``elements`` and
    ``faces`` restrict the pieces to those gfem tetrahedra and triangles
    (boolean masks, ``None`` for all; see `terms.pieces`); the space is the
    unrestricted one. A model with `gfem.Periodic` pairs solves on the
    periodic space, Bloch-phased by ``wave_vector`` (rad/m; ``None``: plain
    periodicity), with its `expansion`; a wave vector makes the operator
    ``"general"`` (`DrivenOperator.symmetry`). With ``paired=False`` it
    acts on the unpaired space instead, where result fields live (no
    expansion). ``comm`` is the resolved communicator of a distributed solve.
    With ``swept`` the wave vector is one of a fixed-angle sweep's: its
    spaces sit in the discretization's swept group
    (`gfem._engine.cache.Discretization.space`). ``stretch`` is the
    perfectly matched layer's stretch for this solve (`gfem._pml.stretch`),
    composed into the volume pieces (`terms.pieces`).
    """
    periodic = model._periodic  # noqa: SLF001
    disc = discretization(model.mesh, periodic, comm)
    built = disc.built
    pec = pec_surfaces(model)
    unpaired_key = solve_key = space_key(model, "hcurl", order)
    unpaired = disc.space(
        solve_key, lambda: hcurl_space(model, built, order, dirichlet=pec)
    )
    space, expansion, phases = unpaired, None, None
    if periodic and paired:
        # the pairs as distributed: a swapped pair takes the inverse phase
        phases = bloch_phases(model, wave_vector, pairs=built.pairs)
        key = solve_key = space_key(model, "hcurl", order, periodic=True, phases=phases)
        space = disc.space(
            key,
            lambda: hcurl_space(
                model, built, order, dirichlet=pec, periodic=True, phases=phases
            ),
            group=phases,
            swept=swept,
        )

        def direct() -> scipy.sparse.csr_matrix:
            return expansion_matrix(space, unpaired, real=phases is None, comm=comm)

        template_key, convert = _expansion_template(model, built, order, unpaired)

        def build() -> scipy.sparse.csr_matrix:
            if phases is None or comm is not None:
                return direct()
            return disc.phased(template_key, phases, direct, convert)

        matrix = disc.space((*key, "expansion"), build, group=phases, swept=swept)
        rows = None
        if comm is not None:
            rows = disc.space(
                (*unpaired_key, "field rows"), lambda: element_dofs(unpaired)
            )
        lattice = None
        if phases is not None and comm is None:
            lattice = functools.partial(
                _lattice_vectors,
                disc,
                template_key,
                convert,
                matrix=matrix,
                phases=phases,
                translations=tuple(p.translation for p in built.pairs),
            )
        expansion = Expansion(matrix, unpaired, rows, lattice)
    ports = terms.lumped_ports(model, built.scale)
    pieces, latent = terms.pieces(
        model, built, order, ports, elements=elements, faces=faces, stretch=stretch
    )
    symmetry: DrivenSymmetry = "complex_symmetric" if wave_vector is None else "general"
    layout = None
    if comm is not None:
        # under the solve space's key: a periodic or Bloch space has its own
        layout = disc.space(
            (*solve_key, "layout"),
            lambda: RankLayout(space, comm),
            group=phases,
            swept=swept,
        )
    fixed = None
    if comm is None and elements is None and faces is None:
        fixed = _fixed_pieces(
            model,
            disc,
            solve_key,
            order,
            ports,
            phases=phases,
            swept=swept,
            stretch=stretch,
        )
    return DrivenOperator(
        built,
        space,
        pieces,
        ports,
        order,
        expansion,
        symmetry,
        layout,
        fixed,
        topology=model.mesh._topology,  # noqa: SLF001
        space_key=solve_key,
        latent=latent,
        cells=None if comm is not None else _solve_cells(model, built),
    )


def _expansion_template(
    model: Model, built: EngineMesh, order: int, unpaired: ngs.FESpace
) -> tuple[tuple, Callable[[tuple[complex, ...]], scipy.sparse.csr_matrix]]:
    """The key of the periodic expansion's phase template, and its conversion.

    The conversion takes any Bloch phases, on a periodic space of its own:
    the probe phase set never enters the discretization's phase sets
    (`gfem._engine.cache.Discretization.phased`).
    """
    pec = pec_surfaces(model)

    def convert(phases: tuple[complex, ...]) -> scipy.sparse.csr_matrix:
        space = hcurl_space(
            model, built, order, dirichlet=pec, periodic=True, phases=phases
        )
        return expansion_matrix(space, unpaired, real=False)

    key = (*space_key(model, "hcurl", order, periodic=True), "expansion template")
    return key, convert


def _lattice_vectors(
    disc: Discretization,
    key: tuple,
    convert: Callable[[tuple[complex, ...]], scipy.sparse.csr_matrix],
    *,
    matrix: scipy.sparse.csr_matrix,
    phases: tuple[complex, ...],
    translations: tuple[tuple[float, float, float], ...],
) -> np.ndarray | None:
    """`Expansion.lattice_vectors` of ``matrix``, P at ``phases``."""
    template = disc.phase_template(key, convert)
    if max(template.fit, template.gap(matrix, phases)) > TEMPLATE_TOL:
        return None
    return template.lattice_vectors(translations)


def _solve_cells(model: Model, built: EngineMesh) -> Callable[[], Cells]:
    """Builds the tetrahedra of ``model``'s electromagnetic domain (`solve_cells`).

    It holds the mesh and the masks, not the model.
    """
    mesh = model.mesh
    active = model._domain_of("em").active  # noqa: SLF001
    layer = () if model.pml is None else model.pml.regions
    tags = tuple(mesh.region(name).tag for name in layer)
    return functools.partial(solve_cells, mesh, built, active, tags)


def solve_cells(
    mesh: Mesh, built: EngineMesh, active: np.ndarray, pml_tags: tuple[int, ...]
) -> Cells:
    """The tetrahedra of a solve domain, in NGSolve's order.

    Args:
        mesh: The model's mesh, from which ``built`` was built.
        built: The engine mesh.
        active: Per tetrahedron of ``mesh``, whether it is in the domain.
        pml_tags: The volume tags of the perfectly matched layer's regions;
            its tetrahedra are marked.
    """
    order = built.cell_order
    inside = active[order]
    pml = None
    if pml_tags:
        pml = np.isin(mesh.tetrahedron_tags[order][inside], pml_tags)
    return Cells(
        np.asarray(mesh.nodes),
        mesh.tetrahedra[order][inside],
        np.flatnonzero(inside),
        pml,
    )


def owned_dofs(op: DrivenOperator) -> int:
    """The DOFs of ``op``'s solve space this rank owns: all of them in one process.

    Summed over the ranks, the space's count (each DOF has one owner).
    """
    owners = None if op.layout is None else op.layout.owners
    if owners is None:
        return int(op.space.ndof)
    return int(owners.sum())


def _fixed_pieces(
    model: Model,
    disc: Any,
    key: tuple,
    order: int,
    ports: tuple[terms.LumpedPortData, ...],
    *,
    phases: tuple[complex, ...] | None = None,
    swept: bool = False,
    stretch: Stretch | None = None,
) -> FixedPieces | None:
    """The split of the volume pieces at ``model``'s material-field regions.

    ``None`` when no element, or every element, lies in one (nothing to keep).
    ``key`` is the solve space's: the store lives with it, and leaves with
    its Bloch ``phases`` (`gfem._engine.cache.Discretization.space`). A
    layer's ``stretch`` enters the pieces and the fixed parts' keys.
    """
    mesh = model.mesh
    inside = np.zeros(len(mesh.tetrahedra), bool)
    for region, material in model.materials.items():
        if isinstance(material, MaterialField):
            inside[mesh.elements(region)] = True
    if not inside.any() or inside.all():
        return None
    built, no_faces = disc.built, np.zeros(len(mesh.triangles), bool)
    return FixedPieces(
        disc.space((*key, "fixed pieces"), KeptParts, group=phases, swept=swept),
        ~inside,
        lambda: terms.pieces(
            model,
            built,
            order,
            ports,
            elements=inside,
            faces=no_faces,
            stretch=stretch,
        )[0],
        stretch=() if stretch is None else stretch.key,
    )


#: Entries of the expansion P below this are the round-off of NGSolve's
#: local projection (exact entries are 0, ±1 and Bloch phases).
EXPANSION_FLOOR = 1e-12


#: The largest element-wise gap between a periodic field and its expansion
#: P x that a distributed P may have (`element_gap`; 8.4e-16 measured).
ELEMENT_GAP_LIMIT = 1e-10


def expansion_matrix(
    periodic: ngs.FESpace, unpaired: ngs.FESpace, *, real: bool, comm: Any = None
) -> scipy.sparse.csr_matrix:
    """P, the periodic space's coefficients to the unpaired space's (e = P x).

    From NGSolve's ``ConvertOperator``, exact on every field of the
    periodic space; entries below `EXPANSION_FLOOR` dropped, real for plain
    periodicity (``real``). Under ranks (``comm``) P is the conversion this
    rank computes alone, between its local spaces: NGSolve's parallel
    conversion is wrong from a periodic space. Each such P is checked
    element by element (`element_gap`), with no exchange.

    Raises:
        gfem._mpi.AgreedError: On every rank, when a rank's P is not the
            periodic field on its elements.
    """
    if comm is None:
        convert = ngs.ConvertOperator(periodic, unpaired)
    else:
        convert = ngs.ConvertOperator(periodic, unpaired, localop=True, parmat=False)
    rows, cols, values = convert.COO()
    data = np.asarray(values)
    if real:
        data = data.real
    out = scipy.sparse.csr_matrix(
        (data, (np.asarray(rows), np.asarray(cols))),
        shape=(unpaired.ndof, periodic.ndof),
    )
    out.data[np.abs(out.data) < EXPANSION_FLOOR] = 0
    out.eliminate_zeros()
    if comm is not None:
        gaps = np.zeros(comm.size)
        gaps[comm.rank] = element_gap(periodic, unpaired, out)
        gaps = _mpi.max_bcast(comm, gaps)
        if gaps.max() > ELEMENT_GAP_LIMIT:
            ranks = np.flatnonzero(gaps > ELEMENT_GAP_LIMIT).tolist()
            raise _mpi.AgreedError(
                f"The periodic expansion differs from the periodic field by up to "
                f"{gaps.max():.1e} (element-wise, relative) on ranks {ranks}: "
                "netgen or NGSolve now behaves differently from the version gfem "
                "was checked against. Report it, and run one-process copies with "
                "comm=MPI.COMM_SELF meanwhile."
            )
    return out


def element_gap(
    periodic: ngs.FESpace,
    unpaired: ngs.FESpace,
    matrix: scipy.sparse.csr_matrix,
    trials: int = 2,
) -> float:
    r"""The largest element-wise gap between a periodic field and its ``matrix`` image.

    For ``trials`` seeded random coefficient vectors :math:`x` of ``periodic``,
    the largest over this rank's elements :math:`K` of
    :math:`\|P x - x\|_{L^2(K)} / \|x\|_{L^2(K)}`, with :math:`P` = ``matrix``
    onto ``unpaired``. Local: element-wise integrals only, no exchange.
    """
    rng = np.random.default_rng(7)
    mesh = periodic.mesh
    worst = 0.0
    for _ in range(trials):
        x = rng.standard_normal(periodic.ndof)
        if periodic.is_complex:
            x = x + 1j * rng.standard_normal(periodic.ndof)
        field, image = ngs.GridFunction(periodic), ngs.GridFunction(unpaired)
        field.vec.FV().NumPy()[:] = x
        image.vec.FV().NumPy()[:] = matrix @ x
        gap, size = (
            np.asarray(
                ngs.Integrate(ngs.InnerProduct(u, u).real, mesh, element_wise=True)
            )
            for u in (image - field, field)
        )
        if len(gap):
            ratio = gap / np.maximum(size, np.finfo(float).tiny)
            worst = max(worst, float(np.sqrt(ratio.max())))
    return worst


@dataclasses.dataclass(frozen=True, eq=False)
class AdjointContext:
    """What an `AdjointLoads` callback sees at one frequency.

    Attributes:
        frequency: The frequency, Hz.
        solution: ``(ndof, n_excited)`` complex, the excited ports' solutions
            at ``frequency`` on the fields' (unpaired) space, columns in
            ``excite`` order; ``None`` where the caller has no forward
            solution (`DrivenRule.solve_adjoints` without ``solutions``).
        excite: The excited ports' names.
        model: The model solved.
        layout: The fields' (support, excluded entities, order): with
            ``model.mesh``, their space (`gfem._engine.fields.fe_space`).
    """

    frequency: float
    solution: np.ndarray | None
    excite: tuple[str, ...]
    model: Model
    layout: tuple[tuple[str, ...], tuple[tuple[int, int], ...], int]
    _operator: DrivenOperator
    _ports: PortNetwork
    _port_names: tuple[str, ...]
    _rhs: np.ndarray

    def restrict(self, g: np.ndarray) -> np.ndarray:
        """``P^T g``: the load, on the solve space, of a functional ``g`` of the field.

        ``g`` ``(ndof, ...)`` is on the fields' space; with periodic pairs
        the field is ``P x``, so its functional loads ``P^T g`` (as a field
        cotangent does in the backward pass).
        """
        expansion = self._operator.expansion
        return g if expansion is None else expansion.field.T @ g

    def s_functional(self, port: str) -> np.ndarray:
        """``(N,)``: port ``port``'s S functional on the solve space (its load)."""
        i = self._port_names.index(port)
        return np.array(self._ports.functional(2 * math.pi * self.frequency)[:, i])


#: A per-frequency adjoint load callback: given one frequency's
#: `AdjointContext`, the loads ``(N,)`` on the solve space to solve
#: ``A z = load`` for, by key. An empty mapping solves nothing at that
#: frequency. Each key's adjoint comes back in the request's
#: `SolveRequest.adjoints` per ``(frequency, key)``, on the fields' space.
AdjointLoads = Callable[[AdjointContext], Mapping[Hashable, np.ndarray]]


def port_loads(names: Sequence[str]) -> AdjointLoads:
    """The S-functional adjoint loads of ports ``names``.

    Keyed by port name, and only where a port's adjoint is not its own
    solution scaled: a port that is not excited, or whose S functional is
    not parallel to its excitation (`PortNetwork.nonparallel`; a guard for
    a pair that is not biorthogonal in floating point: every wave port
    measured is parallel).
    """
    names = tuple(names)

    def loads(context: AdjointContext) -> dict[Hashable, np.ndarray]:
        indices = [context._port_names.index(n) for n in names]  # noqa: SLF001
        solved, columns = context._ports.nonparallel(  # noqa: SLF001
            2 * math.pi * context.frequency,
            indices,
            context._rhs,  # noqa: SLF001
        )
        return {
            context._port_names[i]: columns[:, k]  # noqa: SLF001
            for k, i in enumerate(solved)
        }

    return loads


class DrivenRule:
    """Host-side forward of a `Driven` problem on a `Model`.

    Construction runs every static check; `forward` reads the model's
    leaves (node coordinates, material values) and solves.
    """

    physics = "em"
    #: The operator family (a key of `sensitivity.PROPERTIES`).
    form = "driven"
    #: Material properties the driven forms read, and a law blend's density.
    reads = frozenset({"eps_r", "mu_r", "tan_delta", "conductivity", "blend"})
    #: Outputs whose cotangents the backward pass handles.
    differentiable_outputs = sweep_outputs.DIFFERENTIABLE
    #: `nondifferentiable_output` reads the wanted input kinds: the port
    #: modes' ``residual`` is refused only beside a design at a port face.
    input_aware = True
    #: Outputs that are a rank's own piece under ranks: the ranks
    #: compare their cotangents' shapes, not their values.
    pieces = frozenset({"e_field"})

    def __init__(  # noqa: PLR0915
        self,
        model: Model,
        problem: Driven,
        options: SolverOptions,
        *,
        comm: Any = None,
        order_frequency: Sequence[float] | None = None,
        gauge: GaugeChoice | None = None,
    ) -> None:
        """Check that the problem fits the model (see `gfem.Driven`).

        ``comm`` is the resolved communicator of a distributed solve.
        ``order_frequency`` is the grid the Floquet order table is computed
        over: a frequency worker's whole sweep, so that its block models the
        sweep's orders; the problem's by default.
        ``gauge`` is the parent rule's gauge choice (a frequency worker, a
        fixed-angle sweep's frequency), taken as it is; by default the rule
        resolves ``SolverOptions.gauge`` (`resolve_gauge`) and logs it.

        Raises:
            NotImplementedError: A feature this version cannot solve.
            ValueError: The problem does not fit the model (among others, a
                wave vector without a periodic pair, or outside the span of
                the pairs' translations), or ``gauge="tree_cotree"`` where
                the tree-cotree gauge does not apply.
            gfem.DistributedUnsupportedError: ``gauge="tree_cotree"`` under
                ``comm``.
            gfem.SolverUnavailableError: ``linear_solver="petsc"`` and PETSc
                cannot be used, or ``"direct"`` or ``"petsc"`` under ``comm``
                without the MPI PETSc stack.
        """
        if not isinstance(problem, Driven):
            raise TypeError(f"Expected gfem.Driven, got {problem!r}.")
        self.problem = problem
        self.options = options
        self.comm = comm
        #: Whether the sweep solves several wave vectors, one per frequency
        #: (a fixed-angle sweep): `forward` and `backward` then
        #: run one single-frequency rule per frequency (`FixedAngleChildren.child`).
        self.swept = (
            problem.wave_vector_frequency is not None and len(problem.frequency) > 1
        )
        #: The Bloch wave vector of the operator, rad/m: the problem's, or
        #: at a fixed angle its first frequency's (every frequency's space
        #: has the same DOFs).
        self.wave_vector = wave_vector_at(problem, problem.frequency[0])
        # before the Bloch refusals: a wave vector without a pair is the
        # first thing to fix
        check_wave_vector(model, problem.wave_vector, "Driven")
        if problem.wave_vector is not None:
            check_bloch(
                "Driven",
                device=options.device,
                linear_solver=options.linear_solver,
                solvers=("direct", "bddc_gmres"),
                reason="factors symmetric operators only",
                sweep=problem.sweep,
            )
        check_floquet(
            "Driven",
            [p.name for p in _floquet.floquet_ports_in(model.ports)],
            device=options.device,
            linear_solver=options.linear_solver,
            sweep=problem.sweep,
        )
        # the model's own refusals before the options'
        _check_supported(model)
        # before any assembly, and before any worker; `gauge` (whether the
        # solve runs through the tree-cotree gauge) is
        # resolved next to the linear solver, or the parent rule's
        self.linear, self.gauge = _resolve(
            model, problem, options, comm=comm, gauge=gauge
        )
        # a frequency on a lossless pole, before any assembly or worker
        # (and warns for a law read outside its fitted range)
        dispersion.check_frequencies(
            model, problem.frequency, reader="driven solve", comm=comm
        )
        if not model.ports and not problem.sources:
            raise ValueError(
                "The model has no ports, so a driven problem has nothing to "
                "excite. Add a gfem.LumpedPort to the model's ports, or give "
                "Driven(sources=...)."
            )
        names = model.port_names
        #: The model's port names: the rows of S.
        self.port_names = names
        #: The excited port names: the columns of S.
        self.excite = names if problem.excite is None else problem.excite
        for name in self.excite:
            if name not in names:
                raise ValueError(
                    f"Driven.excite names port {name!r}, but the model's ports are "
                    f"{', '.join(map(repr, names))}."
                )
            port = model.ports[names.index(name)]
            if isinstance(port, LumpedPort) and math.isnan(resistance(port)):
                raise ValueError(
                    f"LumpedPort {name!r} has no resistance, so it has no reference "
                    "impedance to feed 1 W through. Give it a resistance, or leave "
                    "it out of Driven.excite."
                )
        self._init_sources(model, problem)
        #: The grid of the Floquet order table: the whole sweep's, in a
        #: frequency worker too.
        self.order_frequency = tuple(
            problem.frequency if order_frequency is None else order_frequency
        )
        #: The perfectly matched layer's stretch, at the reference frequency
        #: of the whole sweep (in a frequency worker too); ``None`` without a
        #: layer.
        self.stretch = _layer_stretch(model, self.order_frequency)
        #: The layer, which the port modes carry.
        self.pml = model.pml
        #: Whether this rule's Bloch spaces sit in the discretization's
        #: swept group (`gfem._engine.cache.Discretization.space`): a
        #: fixed-angle sweep of several frequencies, read on the whole grid
        #: so that each frequency's single-frequency rule (`FixedAngleChildren.child`)
        #: and each frequency worker of it agree with `swept`.
        self.swept_spaces = (
            problem.wave_vector_frequency is not None and len(self.order_frequency) > 1
        )
        #: Per Floquet surface, its order table over the whole grid
        #: (`gfem._floquet.order_table`): the parent's pre-pass, before any
        #: assembly; empty without Floquet ports.
        self.orders = _floquet_prepass(
            model, problem, options, self.order_frequency, self.excite
        )
        # the parent's warning, from the same grid
        _floquet.warn_oblique_absorbers(
            model,
            problem.frequency,
            wave_vectors(problem, problem.frequency),
            stacklevel=4,
        )
        #: Indices of the frequencies whose fields are kept.
        self.kept = tuple(problem.frequency.index(f) for f in problem.kept_frequencies)
        #: Whether the sweep is adaptive (a reduced model).
        self.adaptive = problem.sweep == "adaptive"
        mesh, active = model.mesh, model._domain.active  # noqa: SLF001
        #: Volume regions with tetrahedra in the solve domain: the field support.
        self.regions = tuple(r for r in mesh.volumes if active[mesh.elements(r)].any())
        #: The wave ports' names, in model port order.
        self.wave_ports = tuple(p.name for p in model.ports if isinstance(p, WavePort))
        #: The wave ports by name, for the fields a port mode result echoes.
        self._wave_port_specs = {
            p.name: p for p in model.ports if isinstance(p, WavePort)
        }
        #: The lumped ports an adaptive sweep solves as adjoint loads: not
        #: excited, with a resistance (`PortNetwork.adjoint_loads`).
        self.lumped_loads = tuple(
            p.name
            for p in model.ports
            if isinstance(p, LumpedPort)
            and p.name not in self.excite
            and not math.isnan(resistance(p))
        )
        #: The wave ports' modes.
        self.modes = {p.name: p.mode for p in model.ports if isinstance(p, WavePort)}
        #: The wave and Floquet ports' de-embedding offsets, m.
        self.offsets = tuple(
            (p.name, p.offset)
            for p in model.ports
            if isinstance(p, (WavePort, FloquetPort))
        )
        #: What the wave ports' modes read, which the design must stay off.
        self.port_design = PortDesign.of(model)
        # The fields live on the mesh as given (traced or not).
        self._mesh = model.mesh
        self._materials = dict(model.materials)
        self._excluded = model._domain.excluded  # noqa: SLF001
        self._check_dipoles(model)
        #: The system operator and port network builder, and the state its
        #: solves keep on a discretization across designs.
        self.systems = DrivenSystemBuilder(
            frequency=problem.frequency,
            columns=tuple(names.index(n) for n in self.excite),
            excite=tuple(self.excite),
            n_sources=self._extra_width(),
            port_mode_tolerance=options.port_mode_tolerance,
            floquet_order_tolerance=options.floquet_order_tolerance,
            orders=self.orders,
            order_frequency=self.order_frequency,
            wave_vector=self.wave_vector,
            problem_wave_vector=problem.wave_vector,
            wave_vector_frequency=problem.wave_vector_frequency,
            swept_spaces=self.swept_spaces,
            comm=comm,
            linear=self.linear,
            adjoint_tolerance=adjoint_tolerance(options),
            retain_memory=options.retain_memory,
            gauge=self.gauge.on,
            polarization=problem.polarization,
            stretch=self.stretch,
        )
        #: A fixed-angle sweep's single-frequency rules, or ``None``.
        self.children = FixedAngleChildren(self) if self.swept else None
        #: How the sweep splits its grid into blocks, or ``None``.
        self.split = SweepSplit.of(self)
        _check_low_frequency(
            model, problem.frequency, self.gauge, stacklevel=4, comm=comm
        )
        _check_skipped_port_modes(model, stacklevel=4, comm=comm)

    def _extra_width(self) -> int:
        """The right-hand sides beyond the excited ports' in one solve.

        The system builder's ``n_sources``: a width hint for its solves.
        """
        if self.reciprocal:
            return len(self.wave_ports) + len(self.lumped_loads)
        return len(self.sources)

    def _init_sources(self, model: Model, problem: Driven) -> None:
        """The sources' attributes and the refusals the model shows of them."""
        #: The problem's sources, solved after the excited ports.
        self.sources = problem.sources
        #: The incident-field sources among them.
        self.incident = tuple(s for s in problem.sources if isinstance(s, INCIDENT))
        #: The point dipoles among them, in source order.
        self.dipoles = tuple(s for s in problem.sources if isinstance(s, DIPOLES))
        #: Whether the sources' port waves are read from the ports' own
        #: solutions (`gfem.Driven.source_fields` ``False``): no source column.
        self.reciprocal = not problem.source_fields
        #: The dipoles whose own field is solved: their emitted power.
        self.emitting = () if self.reciprocal else self.dipoles
        #: The labels of the solve's columns: the excited ports, then the
        #: sources solved.
        self.columns = (
            *self.excite,
            *(() if self.reciprocal else (s.name for s in problem.sources)),
        )
        check_incident_model(model, self.incident)
        #: The model's background stack and its layer's regions, which the
        #: kept fields of incident columns carry.
        self.background = model.background
        self.layers = layer_regions(model)
        #: Per design region a truncation would make the incident load read,
        #: why it cannot be differentiated (with incident columns only).
        self.truncation_designs = truncation_designs(model) if self.incident else {}
        #: Why the node coordinates cannot be differentiated, or None.
        self.nodes_unsupported = self._nodes_unsupported(model)

    @property
    def _reference_frequency(self) -> float | None:
        """What the result fields carry for readers of the stretched operator."""
        return None if self.stretch is None else self.stretch.reference_frequency

    def operator(self, model: Model) -> DrivenOperator:
        """The model's driven operator at its current node positions.

        The last model's operator is kept, so that a backward pass and its
        terms build it once.
        """
        last = getattr(self, "_last_operator", None)
        if last is not None and last[0] is model:
            return last[1]
        with _timing.phase("discretize") as span:
            op = driven_operator(
                model,
                self.options.order,
                wave_vector=self.wave_vector,
                comm=self.comm,
                swept=self.swept_spaces,
                stretch=self.stretch,
            )
            span.set(ndof=owned_dofs(op))
        self._last_operator = (model, op)
        return op

    def forget_operator(self) -> None:
        """Drop the last model's operator (`operator`), so its memory goes."""
        self.__dict__.pop("_last_operator", None)

    def _check_forward(
        self, model: Model, *, keep: bool, request: SolveRequest
    ) -> None:
        """The checks every forward runs before any assembly.

        A law traced at construction is checked here, on concrete values.
        """
        dispersion.check_frequencies(model, self.problem.frequency)
        self._check_adjoint_ports(keep=keep, loads=request.adjoint_loads)

    def forward(  # noqa: PLR0912, PLR0915 - one frequency loop with its adjoint columns
        self, model: Model, *, keep: bool, request: SolveRequest | None = None
    ) -> tuple[dict[str, object], Entry | None]:
        """Solve every excited port at every frequency.

        With ``frequency_workers`` above 1 the frequencies are split over
        workers (`SweepSplit`); the outputs are the same.

        Args:
            model: The model, with concrete leaves.
            keep: Keep every port solution for a backward pass (fields
                only: the backward needs no solver for S).
            request: The per-frequency adjoint loads a uniform forward also
                solves on its own factorization (their adjoints, per
                ``(frequency, key)`` on the unpaired space, land in its
                ``adjoints``), whether a partial result is kept, and a
                frequency block's rows of its parent's port mode sweep.

        Returns:
            ``{"s": (n_freq, n_ports, n_excited) complex, "z_ref":
            (n_freq, n_ports) ohm, "iterations": Krylov iterations per
            frequency and excited port (empty for direct solves, including
            linear_solver='petsc' with ksp_type preonly), "reused": per
            frequency, whether its solve ran on a preconditioner built for
            another matrix}``, plus
            ``"e_field"``, ``(ndof, n_kept * n_excited)`` complex, when
            fields are kept (frequency-major), the wave ports' modes (see
            `SweepOutputs`) when there are wave ports, and the cache entry
            (``fields["solution"]``, ``(ndof, n_freq * n_excited)``,
            frequency-major) or ``None``.

        Raises:
            LinearSolverConvergenceError: A linear solve did not converge.
                A uniform sweep stops at that frequency (each frequency
                worker's block at its own first one), sets the error's
                ``frequency`` and ``unsolved``, and keeps the outputs at
                the solved frequencies on the error for `partial_result`
                (with the request's ``keeps_partial``). An adaptive sweep sets
                ``frequency`` to the sample's.
        """
        request = SolveRequest() if request is None else request
        self._check_forward(model, keep=keep, request=request)
        other = self._forward_other(model, keep=keep, request=request)
        if other is not None:
            return other
        frequencies = self.problem.frequency
        width = len(self.excite)
        n_ports = len(self.port_names)
        iterations: list[tuple[int, ...]] = []
        reused: list[bool] = []
        fields: list[np.ndarray] = []
        solutions: list[np.ndarray] = []
        miss: LinearSolverConvergenceError | None = None
        n_solved = len(frequencies)
        # one factorization at a time under ranks: the last design's goes first
        ENTRIES.free_bases(self.comm)
        with task_manager(self.options.threads, self.comm):
            # the incident fields' rules read the source alone: before the ports
            check_contact(model, self.incident, frequencies)
            system, ports = self.system(model, handed=request.port_modes)
            sources = self._source_columns(model, system)
            s = np.zeros((len(frequencies), n_ports, width), complex)
            waves = np.zeros((len(frequencies), n_ports, len(self.sources)), complex)
            emitted = np.zeros((len(frequencies), len(self.emitting)))
            # the first frequency starts from the previous design
            previous, design = None, self.systems.design_guess(system)
            with (
                system.sweep(
                    solves=len(frequencies),
                    pool_budget=retain_budget(self.options.retain_memory),
                ) as sweep,
                _progress.task(_progress.UNIFORM_SWEEP, total=len(frequencies)) as row,
            ):
                for n, f in enumerate(frequencies):
                    omega = 2 * math.pi * f
                    reading = None
                    if self.reciprocal:
                        excited = _rhs(ports, None, omega, system, width)
                        reading = ports.solution_loads(omega, excited)
                        rhs = np.concatenate([excited, reading.loads], axis=1)
                    else:
                        rhs = _rhs(ports, sources, omega, system, width)
                    try:
                        x, its = design.solve(sweep, omega, rhs, previous)
                    except LinearSolverConvergenceError as err:
                        miss, n_solved = err, n
                        break
                    if request.adjoint_loads is not None:
                        try:
                            self._solve_adjoints(
                                sweep,
                                system,
                                ports,
                                f,
                                rhs,
                                model=model,
                                solution=system.operator.expand(x),
                                loads=cast("AdjointLoads", request.adjoint_loads),
                                into=request.adjoints,
                            )
                        except LinearSolverConvergenceError:
                            system.free()
                            raise
                    iterations.append(its)
                    reused.append(sweep.reused)
                    previous = x
                    if keep:
                        solutions.append(np.array(x, dtype=complex))
                    if n in self.kept:
                        # results keep fields on the unpaired space
                        fields.append(
                            np.array(system.operator.expand(x[:, :width]), complex)
                            if self.reciprocal
                            else np.array(system.operator.expand(x), complex)
                        )
                    if reading is not None:
                        if n_ports and sources is not None:
                            s[n] = ports.outgoing(omega, x[:, :width])
                            z = reading.solutions(ports.columns, x)
                            waves[n] = ports.deembedding[n][:, None] * (
                                sources.port_waves(omega, z)
                            )
                    elif n_ports:
                        s[n], waves[n] = _outgoing(ports, n, omega, x, width)
                    if sources is not None and self.emitting:
                        emitted[n] = sources.emitted_power(omega, x[:, width:])
                    row.advance(frequency=f, iterations=its)
                if keep and miss is None:
                    system.keep_base(sweep)
        # no backward pass will prepare from the pieces
        if not keep or miss is not None:
            system.free()
            # no entry walks the kept port state down: this solve does
            ENTRIES.trim(retain_budget(self.options.retain_memory), self.comm)
        out = SweepOutputs.uniform(
            tuple(range(n_solved)),
            s=ports.scattering(s),
            z_ref=ports.z_ref,
            iterations=iterations,
            reused=reused,
            e_field=rank_piece(
                np.concatenate(fields, axis=1),
                lambda: self._field_shape(model)[0],
                self.comm,
            )
            if fields
            else None,
            waves=ports.waves,
            port_waves=waves if self.sources else None,
            emitted_power=emitted if self.emitting else None,
        )
        if miss is not None:
            raise_unsolved(
                frequencies,
                out,
                [(n_solved, str(miss))],
                miss,
                keeps_partial=request.keeps_partial,
            )
        entry = None
        if keep:
            entry = Entry(
                physics=self.physics,
                handle=_Kept(system, ports),
                fields={"solution": np.concatenate(solutions, axis=1)},
            )
        return out.outputs(), entry

    def _check_dipoles(self, model: Model) -> None:
        """The dipoles' position refusals, before any assembly.

        A traced model's are checked by its forward, on concrete values.
        """
        if self.dipoles and not is_traced(
            (model.mesh.nodes, list(model.materials.values()))
        ):
            check_positions(
                model,
                self.dipoles,
                self._field_layout(),
                self.problem.frequency,
                self.comm,
            )

    def _field_layout(self) -> Any:
        """The kept fields' probe layout (`gfem._engine.probe.Layout`)."""
        from gfem._engine.probe import Layout

        return Layout(
            support=self.regions,
            excluded=tuple(self._excluded),
            space="hcurl",
            order=self.options.order,
        )

    def _source_columns(
        self, model: Model, system: SystemOperator
    ) -> SourceColumns | None:
        """The source columns of this forward; ``None`` without sources."""
        if not self.sources:
            return None
        with _timing.phase("discretize"):
            return SourceColumns.build(
                self.sources,
                ColumnContext(
                    model=model,
                    system=system,
                    order=self.options.order,
                    stretch=self.stretch,
                    frequencies=tuple(self.problem.frequency),
                    field_layout=self._field_layout(),
                    comm=self.comm,
                ),
            )

    def solve_adjoints(
        self,
        model: Model,
        adjoint_loads: AdjointLoads,
        solutions: Mapping[float, np.ndarray] | None = None,
    ) -> dict[tuple[float, Hashable], np.ndarray]:
        """The adjoints of ``adjoint_loads`` at every frequency, without the forward.

        For `gfem.error_estimate` on a finished result: ``solutions`` holds
        its fields per frequency, ``(ndof, n_excited)`` on the unpaired
        space, for loads formed from them (`AdjointContext.solution`). Each
        frequency with a load is factorized once; the others are not
        factorized at all. Returns the adjoints per ``(frequency, key)``, on
        the unpaired space.
        """
        self._check_adjoint_ports(keep=False, loads=adjoint_loads)
        adjoints: dict[tuple[float, Hashable], np.ndarray] = {}
        with task_manager(self.options.threads, self.comm):
            system, ports = self.system(model)
            try:
                frequencies = self.problem.frequency
                with (
                    system.sweep(solves=len(frequencies)) as sweep,
                    _progress.task(
                        _progress.UNIFORM_SWEEP_ADJOINT, total=len(frequencies)
                    ) as row,
                ):
                    for f in frequencies:
                        rhs = ports.rhs(2 * math.pi * f)
                        x = None if solutions is None else solutions[f]
                        self._solve_adjoints(
                            sweep,
                            system,
                            ports,
                            f,
                            rhs,
                            model=model,
                            solution=x,
                            loads=adjoint_loads,
                            into=adjoints,
                        )
                        row.advance(frequency=f)
            finally:
                system.free()
        return adjoints

    def _check_adjoint_ports(self, *, keep: bool, loads: object) -> None:
        """Raise if adjoint ``loads`` reach a forward that would not solve them.

        Only a uniform sweep on one worker, never differentiated, adds the
        adjoint columns (`gfem.refine_mesh` asks for no other).
        """
        if loads is None:
            return
        workers_run = min(self.options.frequency_workers, len(self.problem.frequency))
        where = None
        if keep:
            where = "a differentiated solve"
        elif self.adaptive:
            where = "an adaptive sweep"
        elif workers_run > 1:
            where = "frequency workers"
        if where is not None:
            raise RuntimeError(
                f"adjoint_loads reached {where}: only an "
                "undifferentiated uniform sweep on one worker solves them."
            )

    def _solve_adjoints(
        self,
        sweep: SystemSweep,
        system: SystemOperator,
        ports: PortNetwork,
        f: float,
        rhs: np.ndarray,
        *,
        model: Model,
        solution: np.ndarray | None,
        loads: AdjointLoads | None,
        into: dict[tuple[float, Hashable], np.ndarray],
    ) -> None:
        """Solve the adjoint ``loads`` at ``f`` on the sweep's handle, into ``into``.

        ``solution`` is the forward's at ``f`` on the unpaired space, or
        ``None``. The handle at ``f`` is the forward's if the sweep is there
        already (one more back-substitution), else prepared here; a
        frequency without loads prepares nothing.
        """
        if loads is None:
            return
        omega = 2 * math.pi * f
        context = AdjointContext(
            frequency=f,
            solution=solution,
            excite=tuple(self.excite),
            model=model,
            layout=(self.regions, tuple(self._excluded), self.options.order),
            _operator=system.operator,
            _ports=ports,
            _port_names=self.port_names,
            _rhs=rhs,
        )
        keyed = dict(loads(context))
        if not keyed:
            return
        keys = list(keyed)
        columns = np.stack([np.asarray(keyed[key], complex) for key in keys], axis=1)
        held = sweep.frequency == omega
        try:
            z, its = sweep.reuse_at(omega).solve_counted(columns)
        except LinearSolverConvergenceError as err:
            names = ", ".join(map(repr, keys))
            raise _estimate_adjoint_error(f, names, err) from err
        report.solve_step(
            sweep.base,
            problem="driven",
            direction="forward",
            iterations=its,
            frequency=f,
            reused=held,
            columns="adjoint load",
        )
        z = system.operator.expand(z)
        for k, key in enumerate(keys):
            into[f, key] = np.array(z[:, k], complex)

    def _forward_other(
        self, model: Model, *, keep: bool, request: SolveRequest
    ) -> tuple[dict[str, object], Entry | None] | None:
        """`forward` of an adaptive sweep, over frequency workers or at a fixed angle.

        ``None`` when the uniform sweep solves its frequencies in one pass here.
        """
        if self.adaptive:
            return self._forward_adaptive(model, keep=keep)
        if self.split is not None:
            return self.split.forward(model, keep=keep, request=request)
        return None

    def partial_result(self, err: LinearSolverConvergenceError) -> DrivenResult | None:
        """The partial result ``err`` carries, if a uniform sweep raised it.

        `gfem.solve`'s eager path sets it as the error's ``result``, outside
        any JAX host callback. ``None`` when the forward was adaptive, solved
        none, or did not keep it (`SolveRequest.keeps_partial`). It is read once: the
        outputs it was built from are dropped from ``err``.
        """
        partial, err._partial = err._partial, None  # noqa: SLF001
        if partial is None or not partial.outputs.rows:
            return None
        return self._uniform_result(partial.outputs)

    def system(
        self, model: Model, *, handed: list[dict[str, np.ndarray]] | None = None
    ) -> tuple[SystemOperator, PortNetwork]:
        """The system operator and the port network of one driven solve.

        The model's operator (`operator`) with its wave and Floquet port
        couplings (`DrivenSystemBuilder.build`), kept on the discretization
        and reused while their key holds. A frequency block's couplings
        stand on ``handed``, its rows of its parent's sweep per wave port
        surface (`SolveRequest.port_modes`).
        """
        with _timing.phase("discretize") as span:
            op = self.operator(model)
            out = self.systems.build(model, op, handed=handed)
            span.set(ndof=owned_dofs(op))
        return out

    def _forward_adaptive(
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, object], Entry | None]:
        """The adaptive sweep of `forward` (see `gfem._sweep.adaptive`).

        Outputs as `forward`'s, except that ``"iterations"`` and
        ``"reused"`` are per sample, kept fields come as ``"e_basis"``
        ``(ndof, n)`` real and ``"e_coordinates"`` ``(n_kept * n_excited,
        n)`` (frequency-major), and ``"sweep"`` holds ``{"samples": grid
        indices in solve order, "errors", "converged", "basis_size"}``. With
        frozen samples, ``n`` is `_padded_size` (zero columns after the
        basis), so the kept fields' shapes are known before the solve.

        The entry (``keep``) holds the reduced model only: the basis
        ``"basis"`` ``(ndof, n)``, the reduced pieces ``"operators"``
        ``(n_pieces, n, n)``, the reduced port voltages ``"voltage"``
        ``(n_ports, n)`` and the reduced solutions ``"coordinates"``
        ``(n_freq, n, n_excited)``; no solver.
        """
        frequencies = self.problem.frequency
        options = self.options
        iterations: list[tuple[int, ...]] = []
        reused: list[bool] = []
        # Dimensionless frequency: omega times the largest extent / c
        extent = np.ptp(np.asarray(model.mesh.nodes), axis=0).max()
        z_scale = float(extent) / scipy.constants.c
        with task_manager(options.threads):
            system, ports = self.system(model)
            sources = self._source_columns(model, system)
            size = int(system.operator.space.ndof)
            samples = self.problem.samples
            # samples lie far apart: a reused BDDC would mostly hit its cap
            with system.sweep(
                reuse_preconditioner=False,
                pool_budget=retain_budget(options.retain_memory),
            ) as system_sweep:

                def solve(
                    omega: float, rhs: np.ndarray, x0: np.ndarray | None
                ) -> np.ndarray:
                    try:
                        x, its = system_sweep.solve(omega, rhs, x0=x0)
                    except LinearSolverConvergenceError as err:
                        raise _sample_error(frequencies, omega, err) from err
                    iterations.append(its)
                    reused.append(system_sweep.reused)
                    return x

                affine = AffineSystem(
                    thetas=[piece.theta for piece in system.operator.pieces],
                    apply=system.operator.apply,
                    excitation=ports.excitation
                    if sources is None or self.reciprocal
                    else sources.excitation(ports.excitation, len(self.excite)),
                    outputs=ports.outputs,
                    solve=solve,
                    corrections=[
                        AffineTerm(c.vectors, c.correction_coefficients)
                        for c in system.surfaces
                        if isinstance(c, PortSurfaceCoupling)
                    ],
                    adjoint_loads=_adaptive_loads(
                        ports, None if self.reciprocal else sources, size
                    ),
                    measure_loads=self.reciprocal,
                    size=size,
                )
                sweep = adaptive_sweep(
                    affine,
                    2 * np.pi * np.asarray(frequencies),
                    tolerance=options.adaptive_tolerance,
                    max_samples=options.adaptive_max_samples,
                    memory=options.adaptive_memory,
                    samples=None
                    if samples is None
                    else [frequencies.index(f) for f in samples],
                    seeds=()
                    if samples is not None
                    else dispersion.resonance_seeds(model, frequencies),
                    z_scale=z_scale,
                )
        system.free()  # the backward pass solves the reduced model only
        if not keep:  # no entry walks the kept port state down: this solve does
            ENTRIES.trim(retain_budget(options.retain_memory))
        kept_fields = None
        size = sweep.model.size
        if self.kept:
            coordinates = sweep.coordinates[list(self.kept)]  # (n_kept, n, n_exc)
            coordinates = np.swapaxes(coordinates, 1, 2).reshape(-1, size)
            width = size if samples is None else self._padded_size()
            # results keep fields on the unpaired space
            expanded = system.operator.expand(sweep.basis)
            basis = np.zeros((expanded.shape[0], width))
            basis[:, :size] = expanded
            padded = np.zeros((len(coordinates), width), complex)
            padded[:, :size] = coordinates
            kept_fields = (basis, padded)
        n_exc, n_freq = len(self.excite), len(frequencies)
        waves = sweep.outputs if ports.outputs else np.zeros((n_freq, 0, 0))
        waves = np.reshape(waves, (n_freq, len(self.port_names), len(self.columns)))
        port_waves, emitted = self._source_outputs(ports, sources, sweep, waves)
        out = SweepOutputs.adaptive(
            s=ports.scattering(waves[:, :, :n_exc]),
            # (n_freq, 0, n_sources) without ports: the traced outputs' width
            port_waves=port_waves,
            emitted_power=emitted,
            z_ref=ports.z_ref,
            iterations=iterations,
            reused=reused,
            sweep={
                "samples": sweep.samples,
                "errors": sweep.errors,
                "converged": sweep.converged,
                "basis_size": sweep.model.size,
            },
            reduced=kept_fields,
            waves=ports.waves,
        )
        entry = None
        if keep:
            omegas = 2 * np.pi * np.asarray(frequencies)
            reduced = sweep.model
            entry = ReducedKept(
                basis=sweep.basis,
                matrices=np.stack([reduced.matrix(w) for w in omegas]),
                functionals=np.stack(
                    [ports.reduced_functional(w, reduced.outputs) for w in omegas]
                ),
                deembedding=ports.deembedding,
                coordinates=sweep.coordinates,
            ).entry(self.physics)
        return out.outputs(), entry

    def _source_outputs(
        self,
        ports: PortNetwork,
        sources: SourceColumns | None,
        sweep: Sweep,
        waves: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """The adaptive sweep's sources' port waves and emitted powers.

        Read from the reduced solutions of the sources' columns, or, with
        ``source_fields=False``, from the ports' own reduced solutions.
        ``waves`` is the sweep's outputs ``(n_freq, n_ports, n_columns)``.
        """
        n_exc = len(self.excite)
        if sources is None:
            return None, None
        if self.reciprocal:
            return self._reciprocal_waves(ports, sources, sweep), None
        # (n_freq, 0, n_sources) without ports: the traced outputs' width
        port_waves = waves[:, :, n_exc:] * ports.deembedding[:, :, None]
        if sources.dipoles is None:
            return port_waves, None
        projected = (sources.dipoles[1].vectors.T @ sweep.basis).T
        emitted = np.stack(
            [
                sources.reduced_power(2 * np.pi * f, projected, y[:, n_exc:])
                for f, y in zip(self.problem.frequency, sweep.coordinates, strict=True)
            ]
        )
        return port_waves, emitted

    def _reciprocal_waves(
        self, ports: PortNetwork, sources: SourceColumns, sweep: Sweep
    ) -> np.ndarray:
        """``(n_freq, n_ports, n_sources)`` port waves read from the reduced solutions.

        Each port's own solution on the basis is the excitation's reduced
        solution scaled, or the reduced solve of its load; the dipoles' loads
        are projected on the basis once (`SourceColumns.reduced_port_waves`).
        """
        model, basis = sweep.model, sweep.basis
        found = sources.dipoles
        assert found is not None  # noqa: S101 - refused otherwise at Driven
        projected = found[1].vectors.T @ basis
        out = np.zeros(
            (len(self.problem.frequency), len(self.port_names), len(self.sources)),
            complex,
        )
        for n, (f, y) in enumerate(
            zip(self.problem.frequency, sweep.coordinates, strict=True)
        ):
            omega = 2 * math.pi * f
            reading = ports.solution_loads(omega, ports.rhs(omega))
            zeta = model.load_coordinates(omega, reading.loads)
            z = reading.solutions(ports.columns, np.concatenate([y, zeta], axis=1))
            out[n] = ports.deembedding[n][:, None] * sources.reduced_port_waves(
                omega, projected, z
            )
        return out

    def _field_shape(self, model: Model) -> tuple[int, int]:
        """``(width, ndof)`` of a kept field: `gfem._engine.fields.dof_shape`.

        On a periodic model under ranks the paired mesh is distributed first:
        its search picks the partition, and the plain mesh the field lives on
        follows it. Distributed plain first, it would hold a seed
        the search may not keep.
        """
        periodic = model._periodic  # noqa: SLF001
        if self.comm is not None and periodic:
            still = _at_imported_nodes(model.mesh)
            discretization(still, periodic, self.comm)
        return dof_shape(
            model.mesh,
            self.regions,
            self._excluded,
            "hcurl",
            self.options.order,
            comm=self.comm,
        )

    def _padded_size(self) -> int:
        """Most basis vectors frozen samples give.

        Two per sample and column (excited port or source), and two per
        sample and adjoint load (`PortNetwork.adjoint_loads`, and
        `SourceColumns.adjoint_loads`): every wave port, every lumped port
        that is not excited and has a resistance, and every dipole whose
        moment is not real up to one phase.
        """
        samples = self.problem.samples or ()
        loads = len(self.wave_ports) + len(self.lumped_loads)
        if not self.reciprocal:
            loads += sum(
                not real_up_to_phase(np.asarray(d.moment)) for d in self.dipoles
            )
        return 2 * len(samples) * (len(self.columns) + loads)

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output."""
        if self.adaptive and self.problem.samples is None:
            raise NotImplementedError(
                "Driven(sweep='adaptive') runs under jax.grad, jax.jit or jax.vmap "
                "only with a frozen sample set: the greedy's samples and basis size "
                "are only known after the solve. Solve once outside the "
                "transformation and pass Driven(..., samples="
                "result.sweep.sampled_frequencies), or use sweep='uniform'."
            )
        ndof = None
        if self.kept:
            with task_manager(self.options.threads, self.comm):
                ndof = self._field_shape(model)[0]
        return SweepOutputs.specs(
            n_freq=len(self.problem.frequency),
            n_ports=len(model.ports),
            n_excited=len(self.excite),
            n_wave=len(self.wave_ports),
            ndof=ndof,
            n_fields=len(self.kept) * len(self.columns),
            reduced=self._padded_size() if self.adaptive else None,
            complex_z_ref=complex_reference(model.ports),
            n_sources=len(self.sources),
            n_dipoles=len(self.emitting),
        )

    def role(self, kind: tuple) -> str:
        """``"read"`` if the driven solve depends on input ``kind``.

        The electromagnetic material properties, the node coordinates and
        any boundary or port value are read; thermal properties and thermal
        boundary values are not, so their gradient is an exact zero.
        """
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        return "unread" if is_thermal_value(kind) else "read"

    def terms(self, model: Model, *, nodes: bool) -> list[Term]:
        """The operator's pieces as the kernel sees them: one `Term` per piece.

        See `operator_terms`.

        Raises:
            PortGeometryError: ``nodes`` and a node of a lumped or Floquet
                port surface has moved (or of a wave port's, where the port
                pass does not differentiate its face).
        """
        return operator_terms(
            model,
            self.operator(model),
            nodes=nodes,
            waves=self.wave_faces_held,
        )

    @property
    def wave_faces_held(self) -> str | None:
        """Who holds the wave ports' faces in the node pass, or ``None``.

        The port pass differentiates a moved face on the uniform and the
        adaptive sweep without a Bloch wave vector; under one the faces stay
        held, and so they do under ``source_fields=False``, whose port waves
        have no gradient through the port modes
        (`_refuse_reciprocal_port_pass`).
        """
        if self.problem.wave_vector is not None:
            return "bloch"
        return "reciprocal" if self.reciprocal else None

    def lumped_terms(self, model: Model) -> list[Term]:
        """The pieces that move with a lumped value, as the lumped pass reads them.

        See `lumped_terms`.
        """
        return lumped_terms(self.operator(model))

    def real_space(self, model: Model) -> ngs.FESpace:
        """A real H(curl) space with the field pairs' DOF numbering (``DiffShape``)."""
        return real_space(model, self.operator(model))

    def unsupported_input(self, kind: tuple) -> str | None:
        """Why a read input cannot be differentiated here, or ``None``.

        An input a wave port's mode reads (the materials of the regions on its
        face, a permittivity law's parameters, a sheet field or `Impedance`
        beside it) is differentiated through the port pass, except under a
        Bloch wave vector or at a fixed angle.

        The node coordinates cannot be differentiated next to a tensor
        absorbing admittance (`terms.absorbing_shape_unsupported`). A
        permittivity law's parameter is differentiated on the uniform and
        adaptive sweeps without a Bloch wave vector (`_law_unsupported`).
        """
        if kind[0] == "law":
            reason = self._law_unsupported()
            if reason is not None:
                return reason
        reason = self._port_face_unsupported(kind)
        if reason is not None:
            return reason
        if kind == ("nodes",) or kind[1:2] in [(r,) for r in self.truncation_designs]:
            # nodes, or a design an incident load would read at the truncation
            return self._unsupported_with_sources(kind)
        return None

    def _port_face_unsupported(self, kind: tuple) -> str | None:
        """Why a wave port's mode cannot follow input ``kind`` here, or ``None``.

        ``None`` for an input no wave port's mode reads.
        """
        ports = self.port_design.reading(kind)
        if not ports:
            return None
        problem = self.problem
        if problem.wave_vector is not None:
            where = (
                "on a fixed-angle sweep"
                if problem.wave_vector_frequency is not None
                else "under a Bloch wave vector"
            )
            return (
                f"the mode of wave port {ports[0]!r} reads it, and gradients with "
                f"respect to a design at a wave port face are not verified {where} "
                "in this version. Keep the design away from wave port faces"
            )
        absorbing = self.port_design.absorbing.intersection(ports)
        if kind[0] == "material" and absorbing and self._anisotropic(kind[1]):
            return (
                "the absorbing edge's admittance beside a per-axis material at a "
                "wave port face is not differentiable in this version: make the "
                "material isotropic next to the absorbing edge or wrap it in "
                "jax.lax.stop_gradient"
            )
        return None

    def _anisotropic(self, region: str) -> bool:
        """Whether the material of ``region`` is a field with principal axes."""
        material = self._materials.get(region)
        return isinstance(material, MaterialField) and material.axes is not None

    def _nodes_unsupported(self, model: Model) -> str | None:
        """Why the node coordinates cannot be differentiated, or ``None``."""
        if self.incident:
            return (
                f"the incident field {self.incident[0].name!r} enters the load "
                "through its interpolant on the mesh, whose derivative by the "
                "node coordinates is not available; use a material design (a "
                "gfem.MaterialField)"
            )
        magnetic = [d for d in self.dipoles if isinstance(d, MagneticDipole)]
        if magnetic:
            return (
                f"the load of the MagneticDipole {magnetic[0].name!r} evaluates "
                "the curl of the basis at its position, and this version has no "
                "node derivative of it. Use an ElectricDipole, a density "
                "(gfem.MaterialField) design, or wrap the node coordinates in "
                "jax.lax.stop_gradient"
            )
        return terms.absorbing_shape_unsupported(model)

    def _unsupported_with_sources(self, kind: tuple) -> str | None:
        """`unsupported_input` of the nodes, or of a region at the truncation."""
        if kind == ("nodes",):
            return self.nodes_unsupported
        if kind[0] in ("material", "law"):
            return self.truncation_designs[kind[1]]
        return None

    def _law_unsupported(self) -> str | None:
        """Why a law parameter cannot be differentiated on this sweep, or None.

        The law pass is verified on the uniform and adaptive sweeps (the
        adaptive backward's per-key pairs carry d theta/dp
        under the law's own keys), without a Bloch wave vector: a
        Bloch-periodic cell has no verification cell.
        """
        if self.problem.wave_vector is not None:
            return (
                "gradients with respect to a permittivity law's parameters are not "
                "verified under a Bloch wave vector in this version"
            )
        return None

    def nondifferentiable_output(
        self, name: str, kinds: Collection[tuple] = frozenset()
    ) -> str | None:
        """Why a cotangent on output ``name`` cannot be pulled back, or ``None``.

        Input-aware: ``kinds`` are the wanted input kinds. The wave ports'
        ``residual`` is the backward error of the mode solve, a diagnostic: it
        has no derivative where a wanted input moves a mode (one a wave port
        reads), and an exact zero elsewhere.
        """
        if name == "port_residual":
            ports = [p for k in kinds for p in self.port_design.reading(k)]
            if ("nodes",) in kinds:  # a face's nodes are inputs of every mode
                ports += self.wave_ports
            if not ports:
                return None
            return (
                f"gfem cannot differentiate the port mode output 'residual' of "
                f"wave port {ports[0]!r}: it is the backward error of the mode "
                "solve, a diagnostic, not a physical quantity. Wrap it in "
                "jax.lax.stop_gradient to use its value without its derivative."
            )
        if name in self.differentiable_outputs:
            return None
        return (
            f"gfem cannot differentiate the driven output {name!r} in this "
            "version. Wrap it in jax.lax.stop_gradient to use its value without "
            "its derivative."
        )

    def backward(  # noqa: PLR0915 - one frequency loop with its adjoint columns
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        wanted: frozenset | None = None,
        port_modes: list[dict[str, np.ndarray]] | None = None,
    ) -> list[FieldPair | Integrated]:
        r"""Fold the S and kept-field cotangents into one field pair per frequency.

        At frequency n, with ``X`` the excited ports' solutions, ``f_i`` port
        i's S functional (``S_ij = f_iᵀ x_j``, de-embedding aside) and ``W``
        the S cotangent, the pair is ``(-Lambda, X)`` with ``Lambda_j =
        sum_i W_ij A^-T f_i + A^-T g_j``, ``g_j`` the cotangent of a kept
        field of port j (on the solve space: ``P^T`` times the field's
        cotangent on a periodic model) and ``A`` the system operator
        (wave-port Robin terms and modal corrections included): ``sum ct
        dy/dp = -sum_j Lambda_j^T (dA/dp) x_j`` (the excitations and the
        modal corrections depend on the port modes only, which the port pass
        below differentiates). Without a Bloch
        wave vector ``A`` is complex symmetric, and when ``f_i`` is parallel
        to port i's excitation (every lumped port and every wave port, lossy
        or not), :math:`A^{-T} f_i` is port i's own solution scaled (no solve).
        The remaining loads (kept-field cotangents, rows of ports that were
        not excited, and every row under a Bloch wave vector) take one
        multi-column adjoint solve, on the transposed system (-k) under a
        Bloch wave vector. On a periodic model the pair
        is on the unpaired space (`DrivenOperator.pair`).

        A design a wave port's mode reads (`PortDesign.reading`) moves the
        excitations, the S functionals, the modal corrections, the Robin term
        and the outputs ``port_gamma``, ``port_z_pv`` and ``z_ref`` through the
        mode, which the held pairs miss: ``PortPass`` collects, per frequency,
        the traces of ``Lambda`` and ``X`` on each such surface with the row's
        cotangents, and runs one 2D pass per surface after the loop. Its
        material cotangents join the pairs (`Integrated`), the rest
        (a law's parameters, a sheet's fill, an `Impedance`'s values) go
        through `explicit_inputs`.

        Port waves read by reciprocity (``source_fields=False``,
        :math:`W_{ic} = d_i\, b_c^\mathsf{T} z_i` with :math:`z_i` port i's
        own solution) load the ports' own columns instead: one adjoint column
        per port, the dipoles' loads weighted by its cotangent row, whatever
        the number of dipoles (`_reciprocal_loads`); columns without a load
        are not solved. Their node term reads the ports' own solutions at the
        dipoles (`PointLoads.port_wave_nodes`).

        The frequencies are solved from the adjoint base's frequency
        (`SystemOperator.adjoint_order`): a base handle kept from the forward
        sits at the last (`SystemOperator.keep_base`), and a backward leaves it
        at the frequency it ended on, so each pass refactorizes once fewer. On
        the plan pool each frequency's base still holds the design, and a pass
        refactorizes nothing. The pairs stay in frequency order.

        ``wanted`` holds the wanted input kinds (``None``: every kind), and
        ``port_modes`` the port mode rows a rebuilt system stands on, the
        ones its forward was handed (`SolveRequest.port_modes`).
        """
        # the pairs carry the law pieces' dθ/dp only when a law parameter is
        # wanted, so a gradient beside a law computes and factors none of them
        laws = wanted is None or any(k[0] == "law" for k in wanted)
        if self.adaptive:
            return self._backward_adaptive(
                model, entry, cotangents, laws=laws, wanted=wanted
            )
        if self.children is not None:
            return self.children.backward(model, entry, cotangents, wanted=wanted)
        solution = entry.fields["solution"]
        width = len(self.excite)
        n_columns = len(self.columns)
        ct_e = cotangents.get("e_field")
        ct_p = cotangents.get("emitted_power")
        ct_w = cotangents.get("port_waves")
        frequencies = self.problem.frequency
        nodes = bool(self.dipoles) and (wanted is None or ("nodes",) in wanted)
        self._explicit_nodes = None
        self._explicit_inputs = {}
        self._port_nodes_report = {}
        pairs: list[FieldPair | None] = [None] * len(frequencies)
        port_terms: list[Integrated] = []
        rebuilt = not isinstance(entry.handle, _Kept)  # frequency workers keep none
        with task_manager(self.options.threads, self.comm):
            if rebuilt:
                entry.handle = _Kept(*self.system(model, handed=port_modes))
            system, ports = entry.handle.system, entry.handle.ports
            op = system.operator
            # the wave ports' modes follow a design at their faces: one 2D pass
            # per surface that reads a wanted input, none elsewhere
            port_pass = PortPass(self, model, system, ports, wanted)
            self._refuse_reciprocal_port_pass(port_pass)
            sources = None
            if system is not None and (
                ct_p is not None or nodes or (self.reciprocal and ct_w is not None)
            ):
                sources = self._source_columns(model, system)
            domain = em_region(model, op.built)
            omegas = [2 * math.pi * f for f in frequencies]
            readings = self._port_readings(ports, system) if self.reciprocal else None
            starts = self._column_starts(readings)
            with _progress.task(
                _progress.UNIFORM_SWEEP_ADJOINT, total=len(frequencies)
            ) as row:
                for n in system.adjoint_order(omegas):
                    f = frequencies[n]
                    omega = 2 * math.pi * f
                    x = solution[:, starts[n] : starts[n + 1]]
                    reading = None if readings is None else readings[n]
                    adjoint, load, undefined = self._output_loads(
                        n, ports, sources, x, cotangents, reading=reading
                    )
                    if ct_e is not None and n in self.kept:
                        k = self.kept.index(n)
                        g = np.asarray(ct_e)[:, k * n_columns : (k + 1) * n_columns]
                        if self.comm is not None:  # a rank's rows, then its padding
                            g = g[: op.field_ndof]
                        # the field is P x: its cotangent loads P^T g
                        load[:, :n_columns] += (
                            g if op.expansion is None else op.expansion.field.T @ g
                        )
                    self._solve_loads(system, n, f, adjoint, load)
                    adjoint[:, undefined] = np.nan
                    pairs[n] = op.pair(
                        -adjoint,
                        x,
                        weights=op.weights(omega, laws=laws),
                        domain=domain,
                    )
                    if port_pass.active:
                        port_pass.collect(n, adjoint, x, cotangents)
                    if nodes and sources is not None and reading is not None:
                        self._add_reciprocal_nodes(
                            model, sources, omega, x, ports, n, reading, ct_w
                        )
                    elif nodes and sources is not None:
                        self._add_explicit_nodes(
                            model,
                            sources,
                            omega,
                            adjoint=adjoint[:, width:],
                            x=x[:, width:],
                            ct=None if ct_p is None else ct_p[n],
                        )
                    row.advance(frequency=f)
            port_terms = self._finish_ports(port_pass if port_pass.active else None)
        if rebuilt:
            # as a rerun forward's (`gfem._adjoint.solve_vjp`): the kept state
            # this backward built goes down to the budget, no entry evicted
            ENTRIES.trim(retain_budget(self.options.retain_memory), self.comm)
        return [*(p for p in pairs if p is not None), *port_terms]

    def explicit_inputs(self) -> dict[tuple, Any]:
        """The cotangents by input kind the backward computed itself.

        The port pass's: a permittivity law's parameters, a sheet field's fill
        and an `Impedance`'s values, which a wave port's mode reads and no
        pass over the 3D pairs gives. Rank 0 alone has them (the ranks' sum
        counts the 2D work once).
        """
        return getattr(self, "_explicit_inputs", {})

    def _port_readings(
        self, ports: PortNetwork, system: SystemOperator
    ) -> list[PortSolutions]:
        """Each frequency's `PortSolutions`, as its forward solved them.

        Which ports' own solutions are the excited columns scaled and which
        are solved columns after them; the loads themselves are dropped
        (``(N, 0)``): the backward reads the solved columns.
        """
        width = len(self.excite)
        out = []
        for f in self.problem.frequency:
            omega = 2 * math.pi * f
            reading = ports.solution_loads(
                omega, _rhs(ports, None, omega, system, width)
            )
            out.append(dataclasses.replace(reading, loads=reading.loads[:, :0]))
        return out

    def _column_starts(self, readings: list[PortSolutions] | None) -> np.ndarray:
        """Where each frequency's columns start in a uniform entry's solution.

        The excited ports' and the sources' columns; with ``source_fields=False``
        (``readings``) the excited ports', then the other ports' own solutions.
        """
        n = len(self.columns)
        if readings is None:
            widths = [n] * len(self.problem.frequency)
        else:
            widths = [n + len(r.ports) for r in readings]
        return np.cumsum([0, *widths])

    def _solve_loads(
        self,
        system: SystemOperator,
        n: int,
        f: float,
        adjoint: np.ndarray,
        load: np.ndarray,
    ) -> None:
        """Add the adjoint solve of frequency ``n`` (``f`` Hz) to ``adjoint``, in place.

        Raises:
            LinearSolverConvergenceError: The adjoint solve did not converge
                (`_adjoint_error`).
        """
        loaded = self._loaded_columns(load)
        if not loaded.any():
            return
        try:
            adjoint[:, loaded] += system.solve_adjoint(
                n, 2 * math.pi * f, load[:, loaded]
            )
        except LinearSolverConvergenceError as err:
            raise _adjoint_error(f, err) from err

    def _loaded_columns(self, load: np.ndarray) -> np.ndarray:
        """The columns of ``load`` the adjoint solve takes, the same on every rank.

        Every column when any is loaded, else none; with ``source_fields=False``
        only the loaded ones, so a port whose port waves carry no cotangent
        costs no solve.
        """
        # a rank without port DOFs would skip a collective solve
        if not self.reciprocal:
            flag = _mpi.agree_any(self.comm, flag=bool(load.any()))
            return np.full(load.shape[1], flag)
        loaded = load.any(axis=0).astype(np.intp)
        return np.asarray(_mpi.reduce_bcast(self.comm, loaded)) > 0

    def _output_loads(
        self,
        n: int,
        ports: PortNetwork,
        sources: SourceColumns | None,
        x: np.ndarray,
        cotangents: Mapping[str, np.ndarray],
        *,
        reading: PortSolutions | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Frequency ``n``'s adjoint columns from the outputs' cotangents.

        Returns ``(free, load, undefined)``: the adjoint columns known
        without a solve (S's reciprocity, a dipole's self-adjoint shortcut),
        the loads left to solve, and the columns meeting an undefined S row.
        S's cotangent loads the excited ports' columns, the port waves' and
        the emitted power's the sources'. With ``source_fields=False``
        (``reading``) the port waves' cotangent loads the ports' own
        columns instead (`_reciprocal_loads`).
        """
        omega = 2 * math.pi * self.problem.frequency[n]
        width = len(self.excite)
        adjoint, load = np.zeros_like(x), np.zeros_like(x)
        undefined = np.zeros(x.shape[1], bool)
        ct_s, ct_w = cotangents.get("s"), cotangents.get("port_waves")
        ct_p = cotangents.get("emitted_power")
        if ct_s is not None and width:
            split = ports.pullback(n, omega, ct_s[n], x[:, :width])
            adjoint[:, :width], load[:, :width] = split.free, split.loads
            undefined[:width] = split.undefined
        if ct_w is not None and self.port_names:
            # port wave (i, j) = d_i f_i^T x_j: loads f_i d_i ct_ij
            d = ports.deembedding[n]
            ct = np.asarray(ct_w[n]) * d[:, None]
            if reading is not None and sources is not None:
                _reciprocal_loads(
                    reading, ports.columns, sources.port_wave_loads, omega, ct, load
                )
                if not _reciprocal_defined(reading, ct):
                    undefined[:] = True
            else:
                load[:, width:] += ports.functional(omega) @ ct
        if ct_p is not None and sources is not None:
            _power_adjoint(
                sources, omega, ct_p[n], ports.layout, x, adjoint, load, width
            )
        return adjoint, load, undefined

    def _backward_adaptive(  # noqa: PLR0912, PLR0915 - one frequency loop, two adjoints
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        *,
        laws: bool,
        wanted: frozenset | None = None,
    ) -> list[FieldPair | Integrated]:
        r"""The adaptive backward: reduced adjoints, full-size ones for reciprocity.

        With the basis V and the sample set held fixed, the reduced solution
        at ω solves ``A_r(ω) y = b_r(ω)``, ``A_r = Vᵀ A(ω) V`` complex
        symmetric (wave-port Robin terms and modal corrections included).
        A cotangent ``g`` on ``y`` (from S through the reduced S
        functionals, the biorthogonal modal projection for wave ports, and from
        kept fields' coordinates) gives the reduced adjoint ``λ = A_r^-T g``
        (n x n) and ``sum ct dy/dp = -Σ_t Σ_ab G_t,ab a_t'(V_a, V_b)`` with
        ``G_t = Σ_ω θ_t(ω) Λ(ω) Y(ω)ᵀ``. Each ``G_t`` (n x n) factors by
        SVD into at most n pairs ``(V a, V b)``, so the element passes do
        not grow with the number of frequencies (on the unpaired space on a
        periodic model, `DrivenOperator.pair`: plain periodicity only, P is
        real). A cotangent on the basis itself is refused by the wrapper.
        A source column's port waves load like S's columns; a dipole's
        emitted power loads :math:`V^\mathsf{T}` of its power's load
        (`SourceColumns.reduced_power_loads`), and with the node coordinates
        wanted, its load's node term reads the adjoint and the solution
        on the basis (`_add_explicit_nodes`).

        Port waves read by reciprocity (``source_fields=False``) take no
        reduced adjoint: at each frequency where they carry a cotangent, one
        adjoint column per port is solved at full size and paired with the
        port's reduced solution (`_reciprocal_adaptive_pair`). A dipole's
        near field is in no basis of port solutions, so a reduced adjoint
        would miss it for a design near a dipole, at the samples too.
        """
        kept = ReducedKept.of(entry)
        basis, matrices = kept.basis, kept.matrices
        functionals, coordinates = kept.functionals, kept.coordinates
        deembedding = kept.deembedding
        columns = [self.port_names.index(p) for p in self.excite]
        n, n_exc, width = basis.shape[1], len(self.excite), len(self.columns)
        ct_c = cotangents.get("e_coordinates")
        ct_p = cotangents.get("emitted_power")
        ct_w = cotangents.get("port_waves") if self.reciprocal else None
        nodes = bool(self.dipoles) and (wanted is None or ("nodes",) in wanted)
        self._explicit_nodes = None
        self._port_nodes_report = {}
        accumulated: dict[str, np.ndarray] = {}
        full: list[FieldPair] = []
        with task_manager(self.options.threads), contextlib.ExitStack() as stack:
            op = self.operator(model)
            sources, projected, sweep = None, None, None
            system, port_pass = self._adaptive_ports(
                model, wanted, sources=ct_p is not None or nodes or ct_w is not None
            )
            self._refuse_reciprocal_port_pass(port_pass)
            if system is not None and (ct_p is not None or nodes or ct_w is not None):
                sources, sweep = self._adaptive_sources(
                    model, system, stack, full_size=ct_w is not None
                )
                if sources is not None and sources.dipoles is not None:
                    projected = (sources.dipoles[1].vectors.T @ basis).T
            domain = em_region(model, op.built)
            for i, f in enumerate(self.problem.frequency):
                omega = 2 * math.pi * f
                if sweep is not None and sources is not None and ct_w is not None:
                    pair = self._reciprocal_adaptive_pair(
                        model,
                        op,
                        kept,
                        sweep,
                        sources,
                        i,
                        np.asarray(ct_w[i]) * deembedding[i][:, None],
                        nodes=nodes,
                        weights=op.weights(omega, laws=laws),
                        domain=domain,
                    )
                    if pair is not None:
                        full.append(pair)
                g = np.zeros((n, width), complex)
                ct = self._reduced_cotangent(i, cotangents)
                if ct.any():
                    split = split_cotangent(
                        ct, deembedding[i], columns, functionals[i].T
                    )
                    g += split.loads
                    g[:, split.undefined] = np.nan
                if ct_p is not None and sources is not None and projected is not None:
                    g[:, n_exc:] += sources.reduced_power_loads(
                        omega, projected, ct_p[i]
                    )
                if ct_c is not None and i in self.kept:
                    k = self.kept.index(i)
                    g += np.asarray(ct_c)[k * width : (k + 1) * width, :n].T
                # a row whose only cotangents are on gamma or Z_PV loads no
                # reduced solve, and still runs the port pass
                adjoint = np.linalg.solve(matrices[i].T, g) if g.any() else None
                if port_pass is not None:
                    port_pass.collect_reduced(
                        i,
                        basis,
                        adjoint,
                        coordinates=coordinates[i],
                        functional=functionals[i],
                        cotangents=cotangents,
                    )
                if adjoint is None:
                    continue
                # under the flag the dipoles have no column: their node term
                # is the full-size pair's
                if nodes and sources is not None and not self.reciprocal:
                    self._add_explicit_nodes(
                        model,
                        sources,
                        omega,
                        adjoint=basis @ adjoint[:, n_exc:],
                        x=basis @ coordinates[i][:, n_exc:],
                        ct=None if ct_p is None else ct_p[i],
                    )
                outer = adjoint @ coordinates[i].T
                for term, theta in op.weights(omega, laws=laws).items():
                    accumulated[term] = accumulated.get(term, 0) + theta * outer
            port_terms = self._finish_ports(port_pass)
            pairs = []
            for term, g in accumulated.items():
                left, right = _factor(g)
                pairs.append(
                    op.pair(
                        -(basis @ left),
                        basis @ right,
                        weights={term: 1.0},
                        domain=domain,
                    )
                )
        return [*pairs, *full, *port_terms]

    def _refuse_reciprocal_port_pass(self, port_pass: PortPass | None) -> None:
        """Refuse a design at a wave port face under ``source_fields=False``.

        The port pass pairs each solved column with its adjoint; under the
        flag the columns are the ports' own solutions and the adjoints are
        loaded by the dipoles' cotangent rows, a pairing no case verifies.
        """
        if self.reciprocal and port_pass is not None and port_pass.active:
            raise NotImplementedError(
                "Driven(source_fields=False) does not differentiate with respect "
                "to a design at a wave port face: the port waves read by "
                "reciprocity have no gradient through the port modes in this "
                "version. Solve with source_fields=True (the default)."
            )

    def _finish_ports(self, port_pass: PortPass | None) -> list[Integrated]:
        """Run the port pass's 2D work: its material terms; the rest go explicit."""
        self._explicit_inputs = {}
        if port_pass is None:
            return []
        found = port_pass.finish()
        self._explicit_inputs = found.explicit
        self._add_port_nodes(found)
        return found.integrated

    def _add_port_nodes(self, found: PortResult) -> None:
        """Add the port pass's face node cotangent to `explicit_nodes`."""
        self._port_nodes_report = found.report
        if found.nodes is None:
            return
        held = self._explicit_nodes
        self._explicit_nodes = found.nodes if held is None else held + found.nodes

    def _adaptive_ports(
        self, model: Model, wanted: frozenset | None, *, sources: bool
    ) -> tuple[SystemOperator | None, PortPass | None]:
        """The system a frozen-basis backward reads, and its port pass.

        The system (its port couplings come from the kept port state, else a
        deterministic re-sweep of the port modes) is built only when the
        sources or a wanted input a wave port's mode reads need it; the pass is
        ``None`` where no surface reads a wanted input.
        """
        groups = bool(self.port_design.groups)
        if wanted is None:
            reads = groups
        else:  # a face's nodes are inputs of its port's mode too
            reads = (("nodes",) in wanted and groups) or any(
                self.port_design.reading(k) for k in wanted
            )
        if not (reads or sources):
            return None, None
        system, ports = self.system(model)
        port_pass = PortPass(self, model, system, ports, wanted) if reads else None
        return system, (port_pass if port_pass and port_pass.active else None)

    def _adaptive_sources(
        self,
        model: Model,
        system: SystemOperator,
        stack: contextlib.ExitStack,
        *,
        full_size: bool,
    ) -> tuple[SourceColumns | None, SystemSweep | None]:
        """The adaptive backward's source columns, and its full-size sweep.

        ``system`` is the one `_adaptive_ports` built. With ``full_size``
        (port waves read by reciprocity carry a cotangent) a sweep at the
        adjoint tolerance, one base refactorized at each frequency with a
        cotangent, held by ``stack`` with the system.
        """
        sources = self._source_columns(model, system)
        if not full_size:
            return sources, None
        stack.callback(system.free)
        sweep = stack.enter_context(
            system.sweep(
                reuse_preconditioner=False,
                tolerance=adjoint_tolerance(self.options),
                pool_budget=retain_budget(self.options.retain_memory),
            )
        )
        return sources, sweep

    def _reduced_cotangent(
        self, i: int, cotangents: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Frequency ``i``'s cotangent on the reduced columns' waves.

        ``(n_ports, n_columns)``: S's on the excited ports' columns, a source
        column's port waves' on its own; port waves read by reciprocity have
        no column (`_reciprocal_adaptive_pair`).
        """
        n_exc = len(self.excite)
        ct = np.zeros((len(self.port_names), len(self.columns)), complex)
        if cotangents.get("s") is not None:
            ct[:, :n_exc] = cotangents["s"][i]
        if cotangents.get("port_waves") is not None and not self.reciprocal:
            ct[:, n_exc:] = cotangents["port_waves"][i]
        return ct

    def _reciprocal_adaptive_pair(  # noqa: PLR0917 - one frequency's adjoint
        self,
        model: Model,
        op: DrivenOperator,
        kept: ReducedKept,
        sweep: SystemSweep,
        sources: SourceColumns,
        i: int,
        ct: np.ndarray,
        *,
        nodes: bool,
        weights: dict[str, complex],
        domain: ngs.Region,
    ) -> FieldPair | None:
        r"""Frequency ``i``'s pair of port waves read by reciprocity, adaptive sweep.

        ``ct`` ``(n_ports, n_sources)`` is their cotangent, de-embedded. Port
        :math:`i`'s solution is its reduced one, :math:`z_i = V\zeta_i` with
        :math:`\zeta_i = A_r^{-1} V^\mathsf{T} f_i` (the kept reduced matrix
        and S functional), and its adjoint column :math:`y_i = A^{-1}\sum_c
        \bar W_{ic} d_i b_c` is solved at full size on ``sweep``: the pair
        :math:`(-y_i, z_i)` is exact at the samples, where :math:`z_i` lies in
        the basis, and accurate to the sweep's field error between them.
        ``None`` without a cotangent at ``i``; ``weights`` and ``domain`` are
        the pair's (`DrivenOperator.pair`).
        """
        rows = np.flatnonzero(np.any(ct != 0, axis=1))
        if not rows.size:
            return None
        f = self.problem.frequency[i]
        omega = 2 * math.pi * f
        zeta = np.linalg.solve(kept.matrices[i], kept.functionals[i][rows].T)
        z = kept.basis @ zeta
        try:
            y, _ = sweep.solve(omega, sources.port_wave_loads(omega, ct[rows]))
        except LinearSolverConvergenceError as err:
            raise _adjoint_error(f, err) from err
        if nodes:
            self._add_port_wave_nodes(
                model, sources, omega, solutions=z, weights=ct[rows]
            )
        return op.pair(-np.asarray(y), z, weights=weights, domain=domain)

    def _add_explicit_nodes(
        self,
        model: Model,
        sources: SourceColumns,
        omega: float,
        *,
        adjoint: np.ndarray,
        x: np.ndarray,
        ct: np.ndarray | None,
    ) -> None:
        """Add the dipoles' node term at ω to `explicit_nodes`.

        ``adjoint`` and ``x`` are the source columns' ``(N, n_sources)``.
        """
        found = sources.dipoles
        if found is None:
            return
        positions, loads = found
        term = loads.nodes(
            model.mesh,
            self._field_layout(),
            omega,
            adjoint=adjoint[:, positions],
            solution=x[:, positions],
            power=np.zeros(len(positions)) if ct is None else np.asarray(ct),
            comm=self.comm,
        )
        held = getattr(self, "_explicit_nodes", None)
        self._explicit_nodes = term if held is None else held + term

    def _add_reciprocal_nodes(  # noqa: PLR0917 - one frequency's columns
        self,
        model: Model,
        sources: SourceColumns,
        omega: float,
        x: np.ndarray,
        ports: PortNetwork,
        n: int,
        reading: PortSolutions,
        ct: np.ndarray | None,
    ) -> None:
        """Add frequency ``n``'s node term of reciprocal port waves to `explicit_nodes`.

        ``x`` holds the frequency's columns, from which `PortSolutions`
        ``reading`` takes the ports' own solutions; ``ct`` is the port waves'
        cotangent ``(n_freq, n_ports, n_sources)`` (`PointLoads.port_wave_nodes`).
        """
        if ct is None:
            return
        self._add_port_wave_nodes(
            model,
            sources,
            omega,
            solutions=reading.solutions(ports.columns, x),
            weights=np.asarray(ct[n]) * ports.deembedding[n][:, None],
        )

    def _add_port_wave_nodes(
        self,
        model: Model,
        sources: SourceColumns,
        omega: float,
        *,
        solutions: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Add the node term at ω of port waves read by reciprocity to `explicit_nodes`.

        ``solutions`` ``(N, n_ports)`` are the ports' own, ``weights``
        ``(n_ports, n_sources)`` the port waves' cotangent, de-embedded
        (`PointLoads.port_wave_nodes`).
        """
        found = sources.dipoles
        if found is None:
            return
        positions, loads = found
        term = loads.port_wave_nodes(
            model.mesh,
            self._field_layout(),
            omega,
            solutions=solutions,
            weights=np.asarray(weights)[:, positions],
            comm=self.comm,
        )
        held = getattr(self, "_explicit_nodes", None)
        self._explicit_nodes = term if held is None else held + term

    def explicit_nodes(self) -> np.ndarray | None:
        """The node cotangent the backward computed itself: the dipoles' loads.

        ``(n_nodes, 3)``, this rank's part, or ``None`` (no dipole moved).
        """
        return getattr(self, "_explicit_nodes", None)

    def result(self, outputs: dict) -> DrivenResult:
        """The public result from the forward outputs."""
        frequencies = self.problem.frequency
        kept = [(frequencies[n], label) for n in self.kept for label in self.columns]
        out = SweepOutputs.read(outputs)
        if self.adaptive:
            return self._adaptive_result(out, kept)
        return self._uniform_result(out)

    def _uniform_result(self, out: SweepOutputs) -> DrivenResult:
        """`result` of a uniform sweep over the grid indices ``out.rows``."""
        grid = self.problem.frequency
        frequencies = tuple(grid[i] for i in out.rows)
        kept = [
            (grid[n], label)
            for n in self.kept
            if n in out.rows
            for label in self.columns
        ]
        incident = {
            s.name: IncidentPart(s, self.background, self.layers)
            for s in self.incident
            if self.background is not None
        }
        # NumPy rows, which Field takes as they are
        coefficients: Any = out.e_field
        e_fields = []
        if coefficients is not None:  # every kept frequency's fields are there
            e_fields = [
                Field(
                    coefficients=coefficients[:, k],
                    mesh=self._mesh,
                    quantity="E",
                    unit="V/m",
                    space="hcurl",
                    order=self.options.order,
                    support=self.regions,
                    omega=jnp.asarray(2 * math.pi * f),
                    _excluded=self._excluded,
                    _comm=_mpi.static_comm(self.comm),
                    _reference_frequency=self._reference_frequency,
                    _incident=incident.get(label),
                    _point_sources=self._point_sources(label),
                )
                for k, (f, label) in enumerate(kept)
            ]
        return DrivenResult(
            s=NetworkParameters(
                values=jnp.asarray(out.s),
                z_ref=jnp.asarray(out.z_ref),
                frequency=frequencies,
                ports=self.port_names,
                excited=tuple(self.excite),
            ),
            e_fields=tuple(e_fields),
            kept=tuple(kept),
            sweep=SweepInfo.uniform(
                frequencies,
                iterations=out.iterations,
                preconditioner_reused=out.reused,
            ),
            port_modes=self._port_modes(out),
            offsets=self.offsets,
            diffraction_orders=self._diffraction_orders(out.rows),
            **self._source_leaves(out),
        )

    def _source_leaves(self, out: SweepOutputs) -> dict[str, Any]:
        """The result's leaves of the sources: names, port waves, emitted power."""
        power = out.emitted_power
        return {
            "sources": {s.name: s for s in self.sources},
            "port_waves": None
            if out.port_waves is None
            else jnp.asarray(out.port_waves),
            "emitted_power": {}
            if power is None
            else {
                d.name: jnp.asarray(power[:, k]) for k, d in enumerate(self.emitting)
            },
        }

    def _point_sources(
        self, label: str
    ) -> tuple[tuple[str, tuple[float, float, float]], ...]:
        """The point dipole whose load solves column ``label``, if it is one."""
        return tuple((d.name, d.position) for d in self.dipoles if d.name == label)

    def _diffraction_orders(
        self, rows: Sequence[int]
    ) -> dict[str, DiffractionOrderResult]:
        """Each Floquet port's `DiffractionOrderResult` at the grid indices ``rows``."""
        grid = [self.problem.frequency[i] for i in rows]
        frequency = np.asarray(grid)
        rows = [self.order_frequency.index(f) for f in grid]  # the table's rows
        out = {}
        for surface, table in self.orders.items():
            r = table.rows
            for port in table.ports:
                o = table.orders.index(port.order)
                out[port.name] = DiffractionOrderResult(
                    port=port.name,
                    frequency=frequency,
                    order=port.order,
                    polarization=port.polarization,
                    surface=surface,
                    index=table.face.index,
                    wave_vector=np.asarray(r.wave_vector[rows, o]),
                    gamma=np.asarray(r.gamma[rows, o]),
                    propagating=np.asarray(r.propagating[rows, o]),
                    theta=np.asarray(r.theta[rows, o]),
                    phi=np.asarray(r.phi[rows, o]),
                )
        return out

    def _port_modes(self, out: SweepOutputs) -> dict[str, PortModeResult]:
        """The wave ports' `PortModeResult` at the grid indices ``out.rows``.

        With their mode fields on the eager path; under a JAX transformation
        the port mode sweeps do not cross the seam, so the fields are ``None``.
        """
        ports = out.ports
        if not self.wave_ports or ports is None:
            return {}
        grid = self.problem.frequency
        frequency = jnp.asarray([grid[i] for i in out.rows])
        modes = {}
        for k, name in enumerate(self.wave_ports):
            modes[name] = PortModeResult(
                port=name,
                mode=self.modes[name],
                frequency=frequency,
                gamma=jnp.asarray(ports.gamma[:, k]),
                Z_PV=jnp.asarray(ports.z_pv[:, k]),
                residual=jnp.asarray(ports.residual[:, k]),
                solved=jnp.asarray(ports.solved[:, k]),
                e_field=ports.field(k),
                pml=self.pml,
                allow_evanescent=self._wave_port_specs[name].allow_evanescent,
                polarity=self._wave_port_specs[name].polarity,
            )
        return modes

    def _adaptive_result(
        self, out: SweepOutputs, kept: list[tuple[float, str]]
    ) -> DrivenResult:
        """`result` of an adaptive sweep: basis-backed fields, `SweepInfo`."""
        frequencies = self.problem.frequency
        # Under a JAX transformation only the frozen sample set is known.
        info = out.sweep
        if not info:
            frozen: dict[str, Any] = {
                "samples": [frequencies.index(f) for f in self.problem.samples or ()],
                "errors": None,
                "converged": None,
                "basis_size": None,
            }
            info = frozen
        sampled = set(info["samples"])
        sweep = SweepInfo(
            kind="adaptive",
            solved=tuple(i in sampled for i in range(len(frequencies))),
            sampled_frequencies=tuple(frequencies[i] for i in info["samples"]),
            sample_errors=info["errors"],
            converged=info["converged"],
            tolerance=self.options.adaptive_tolerance,
            basis_size=info["basis_size"],
            iterations=None if out.iterations is None else tuple(out.iterations),
            preconditioner_reused=None if out.reused is None else tuple(out.reused),
        )
        e_fields: ReducedFields | tuple = ()
        if kept and out.reduced is not None:
            # NumPy arrays, which ReducedFields takes as they are
            reduced: Any = out.reduced
            basis, coordinates = reduced
            e_fields = ReducedFields(
                basis=basis,
                coordinates=coordinates,
                omega=jnp.asarray([2 * math.pi * f for f, _ in kept]),
                mesh=self._mesh,
                quantity="E",
                unit="V/m",
                space="hcurl",
                order=self.options.order,
                support=self.regions,
                _excluded=self._excluded,
                _reference_frequency=self._reference_frequency,
                _point_sources=tuple(self._point_sources(label) for _, label in kept)
                if self.dipoles
                else (),
            )
        return DrivenResult(
            s=NetworkParameters(
                values=jnp.asarray(out.s),
                z_ref=jnp.asarray(out.z_ref),
                frequency=frequencies,
                ports=self.port_names,
                excited=tuple(self.excite),
            ),
            e_fields=e_fields,
            kept=tuple(kept),
            sweep=sweep,
            port_modes=self._port_modes(out),
            offsets=self.offsets,
            **self._source_leaves(out),
        )


def operator_terms(
    model: Model,
    op: DrivenOperator,
    *,
    nodes: bool,
    waves: str | None = "eigenmode",
) -> list[Term]:
    """The pieces of ``model``'s driven operator as the kernel sees them.

    One `Term` per piece, latent ones included, but the ports': a port's
    pieces, right-hand side and voltage read only its own surface (shape
    gradients hold port surfaces in place) and its lumped values, whose
    gradient is the lumped pass's (`lumped_terms`). Each term
    carries its piece's material derivatives, face owners and sheet factor.
    The eigenmode rule's operator is the same pieces, so it shares these.

    Args:
        model: The model ``op`` was built on.
        op: Its driven operator.
        nodes: Whether the node pass reads the terms: the ports' nodes must
            then be held (a material or fill gradient checks no port
            geometry).
        waves: Who holds the wave ports' faces (see
            `check_ports_held`): the driven backward's port pass
            carries their node derivative (`gfem._engine.port_pass`), so it
            holds none; the eigenmode rule holds them.

    Raises:
        PortGeometryError: ``nodes`` and a node of a port surface has moved
            (of a wave port's only where ``waves`` holds it).
    """
    if nodes:
        check_ports_held(model, waves=waves)
    out = []
    for piece in (*op.pieces, *op.latent):
        if piece.kind == "port":
            continue
        common: dict[str, Any] = {
            "derivatives": piece.derivatives,
            "owner": piece.owner,
            "sheet": piece.sheet,
            "law": None if piece.law is None else piece.law[0],
            "points": piece.points,
        }
        if piece.kind == "surface":
            term = Term(
                piece.name,
                "surface",
                integrand=piece.integrand,
                space=real_space(model, op),
                **common,
            )
        else:
            term = Term(
                piece.name,
                piece.kind,
                coefficient=piece.coefficient,
                region=piece.region,
                position=piece.position,
                **common,
            )
        out.append(term)
    return out


def lumped_terms(op: DrivenOperator) -> list[Term]:
    """The terms of the lumped pass: the pieces that move with a lumped value.

    One ``"trace"`` `Term` per surface of each piece with a `LumpedValue` (a
    lumped port's L and C pieces, every piece of an `Impedance`), under the
    piece's name, carrying its coefficient there and ``(kind, slope)``. The
    eigenmode rule's operator is the same pieces, so it shares these.
    """
    out = []
    for piece in op.pieces:
        value = piece.lumped
        if value is None:
            continue
        out += [
            Term(
                piece.name,
                "trace",
                coefficient=c,
                region=region,
                lumped=(value.kind, value.slope),
            )
            for region, c in value.faces
        ]
    return out


def real_space(model: Model, op: DrivenOperator) -> ngs.FESpace:
    """A real H(curl) space with the DOF numbering of ``op``'s field pairs.

    For ``DiffShape``: ``op``'s solve space's numbering without periodic
    pairs; with them the real unpaired space (on the identified mesh), where
    `DrivenOperator.pair` puts the fields, since no real periodic space can
    hold a Bloch phase (the kernel splits complex fields into real and
    imaginary parts).
    """
    pec = pec_surfaces(model)
    disc = discretization(model.mesh, model._periodic, op.built.comm)  # noqa: SLF001
    return disc.space(
        space_key(model, "hcurl", op.order, real=True),
        lambda: hcurl_space(
            model, op.built, op.order, complex_valued=False, dirichlet=pec
        ),
    )


#: Why a moved lumped or Floquet port surface is refused.
_HELD_REASON = (
    "shape gradients hold lumped and Floquet port surfaces fixed. Keep the "
    "displacement zero there: gfem.shape.boundary_morph holds them for you."
)
#: Who holds the wave port faces, by the refusal's context, and the remedy.
_WAVE_HOLDERS = {
    "eigenmode": (
        "an Eigenmode solve",
        (
            "A gfem.Driven solve differentiates a moved wave port face "
            "(gfem.shape.boundary_morph(..., sliding_ports=...))."
        ),
    ),
    "bloch": (
        "a Bloch wave vector",
        (
            "A gfem.Driven solve without one differentiates a moved wave port face "
            "(gfem.shape.boundary_morph(..., sliding_ports=...))."
        ),
    ),
    "reciprocal": (
        "Driven(source_fields=False)",
        (
            "A gfem.Driven solve with source_fields=True (the default) "
            "differentiates a moved wave port face "
            "(gfem.shape.boundary_morph(..., sliding_ports=...))."
        ),
    ),
}


def _reason(waves: str | None) -> str:
    """Why a moved wave port face is refused, held by ``waves``."""
    holder, remedy = _WAVE_HOLDERS[waves or "eigenmode"]
    return f"{holder} holds wave port faces. {remedy}"


def check_ports_held(model: Model, *, waves: str | None = "eigenmode") -> None:
    """Raise unless the port surfaces' nodes sit at their imported positions.

    Shape gradients treat a lumped or Floquet port's geometry (its terms,
    excitation and voltage) as frozen: exact for every displacement that
    holds the ports. A wave port's face is held only where ``waves`` names
    who holds it (``"eigenmode"``, ``"bloch"`` or ``"reciprocal"``): the
    driven port pass differentiates a moved face (``None``).

    Raises:
        PortGeometryError: A node of a held port surface has moved.
    """
    mesh = model.mesh
    nodes = np.asarray(mesh.nodes)
    moved = np.any(nodes != mesh.imported_nodes, axis=1)
    if not moved.any():
        return
    for port in model.ports:
        wave = isinstance(port, WavePort)
        if wave and waves is None:
            continue
        for surface in surfaces_of(port):
            on = mesh.triangles[mesh.triangle_tags == mesh.region(surface).tag]
            off = np.unique(on)[moved[np.unique(on)]]
            if len(off):
                distance = np.linalg.norm(
                    nodes[off] - mesh.imported_nodes[off], axis=1
                ).max()
                raise PortGeometryError(
                    f"{type(port).__name__} {port.name!r}: {len(off)} node(s) of its "
                    f"surface {surface!r} moved (up to {distance:.3g} m), so gfem "
                    "cannot differentiate the solve with respect to the node "
                    f"coordinates: {_reason(waves) if wave else _HELD_REASON}"
                )


@dataclasses.dataclass
class DeferredCheck:
    """A deferred low-frequency check (`low_frequency_deferred`).

    Attributes:
        gauge: The gauge choice of the last driven rule built in the
            context, for `check_low_frequency`; ``None`` before any.
    """

    gauge: GaugeChoice | None = None


#: The deferred check of the driven solves in this context, or ``None``
#: (`low_frequency_deferred`).
_LOW_FREQUENCY_DEFERRED: contextvars.ContextVar[DeferredCheck | None] = (
    contextvars.ContextVar("low_frequency_deferred", default=None)
)


@contextlib.contextmanager
def low_frequency_deferred() -> Generator[DeferredCheck]:
    """Skip the low-frequency check of the driven solves in this context.

    Mesh refinement solves on a new mesh each cycle and checks the floor once,
    on the mesh it returns, with `check_low_frequency`. Each rule
    built in the context records its gauge choice on the yielded check
    (`_check_low_frequency`), so that the deferred check words its warning
    as the last solve was gauged.
    """
    deferred = DeferredCheck()
    token = _LOW_FREQUENCY_DEFERRED.set(deferred)
    try:
        yield deferred
    finally:
        _LOW_FREQUENCY_DEFERRED.reset(token)


@dataclasses.dataclass(frozen=True)
class LowFrequencyFloor:
    """A mesh's low-frequency floor, ``k0 h_min = LOW_FREQUENCY_FLOOR``.

    Shared by the driven and the eigenmode checks.

    Attributes:
        h_min: The shortest edge of the solve domain on the imported nodes, m.
    """

    h_min: float

    @classmethod
    def of(cls, model: Model) -> LowFrequencyFloor:
        """The floor of ``model``'s solve domain."""
        active = model._domain.active  # noqa: SLF001
        return cls(float(_shortest_edges(model.mesh)[active].min()))

    @property
    def frequency(self) -> float:
        """The floor's frequency, Hz."""
        return LOW_FREQUENCY_FLOOR * scipy.constants.c / (2 * math.pi * self.h_min)

    def k0h(self, f: Any) -> Any:
        """``k0 h_min`` at the frequencies ``f`` (Hz), a float or an array."""
        return 2 * math.pi * f / scipy.constants.c * self.h_min

    def describe(self, k0h: float) -> str:
        """The message's clause on the floor, at the reported ``k0 h_min``."""
        return (
            f"this mesh's low-frequency floor of {hertz(self.frequency, 4)} "
            f"(k0*h_min = {k0h:.2g} < {LOW_FREQUENCY_FLOOR:g}, with h_min = "
            f"{self.h_min:.3g} m the shortest edge in the solve domain)"
        )


def check_low_frequency(
    model: Model,
    frequencies: tuple[float, ...],
    gauge: GaugeChoice | None,
    *,
    stacklevel: int,
) -> None:
    """`_check_low_frequency` on one process, for a deferred check.

    ``gauge`` is the last solve's gauge choice (`DeferredCheck.gauge`).
    """
    _check_low_frequency(
        model, frequencies, gauge, stacklevel=stacklevel + 1, force=True
    )


def _check_low_frequency(
    model: Model,
    frequencies: tuple[float, ...],
    gauge: GaugeChoice | None,
    *,
    stacklevel: int,
    comm: Any = None,
    force: bool = False,
) -> None:
    """Warn once if a frequency is below the mesh's low-frequency floor.

    h_min is the shortest edge of the solve domain on the imported nodes,
    so the message is the same at every step of a shape optimization.
    Skipped inside `low_frequency_deferred` unless ``force``. A gauged
    solve warns only with wave ports, about their modes; an
    ungauged one ends with the gauge choice's clause on why.

    Args:
        model: The model, its solve domain, its mesh and its ports.
        frequencies: The output frequencies (Hz).
        gauge: The solve's gauge choice; ``None`` reads as ungauged, with
            no clause.
        stacklevel: Passed on to `warnings.warn`, counted from the caller.
        force: Check inside `low_frequency_deferred` too; without it,
            the check there records ``gauge`` (`DeferredCheck.gauge`).
    """
    deferred = _LOW_FREQUENCY_DEFERRED.get()
    if deferred is not None and not force:
        deferred.gauge = gauge
        return
    gauged = gauge is not None and gauge.on
    wave_ports = any(isinstance(p, WavePort) for p in model.ports)
    if gauged and not wave_ports:
        return
    floor = LowFrequencyFloor.of(model)
    f = np.asarray(frequencies)
    k0h = floor.k0h(f)
    low = k0h < LOW_FREQUENCY_FLOOR
    if not low.any():
        return
    lowest = int(np.argmin(f))
    below = (
        f"Driven: {int(low.sum())} of {len(f)} frequencies, the lowest "
        f"{hertz(f[lowest], 4)}, are below {floor.describe(k0h[lowest])}. "
    )
    if gauged:
        msg = below + (
            "The volume solve is gauged and keeps its digits, but the wave ports' "
            "mode solve is not: their modes, and S through them, carry an error "
            "that grows as the inverse square of the frequency below it, and a "
            "mode solve may report a TEM mode as evanescent (gfem.PortModeError). "
            "Use lumped ports at these "
            "frequencies, or raise the lowest frequency."
        )
        _mpi.warn(comm, msg, LowFrequencyWarning, stacklevel=stacklevel + 1)
        return
    msg = below + (
        "Below it the fields and their gradients, and "
        "with linear_solver='direct', or 'petsc' with its default "
        "factorization, the capacitive part of S, "
        "carry rounding errors up to about 1e-16/(k0*h_min)^2, and MKL "
        "PARDISO ('direct' with the mkl extra) becomes unreliable below "
        f"k0*h_min = {_PARDISO_FLOOR:g}. For capacitance, solve "
        "gfem.Electrostatic instead of reading it from S; otherwise raise the "
        "lowest frequency or coarsen the smallest elements."
    )
    if gauge is not None and gauge.clause is not None:
        msg = f"{msg} {gauge.clause}"
    _mpi.warn(comm, msg, LowFrequencyWarning, stacklevel=stacklevel + 1)


def _modes(ranks: list[int]) -> str:
    """``"mode 2"`` or ``"modes 1 and 2"``."""
    if len(ranks) == 1:
        return f"mode {ranks[0]}"
    return f"modes {', '.join(map(str, ranks[:-1]))} and {ranks[-1]}"


def _check_skipped_port_modes(
    model: Model, *, stacklevel: int, comm: Any = None
) -> None:
    """Warn once per wave port surface that skips a mode below its highest.

    A rank below the surface's highest requested ``mode`` propagates
    wherever that mode does (the port mode sweep raises otherwise), so the
    check needs no numerics. Ranks count the modes the surface's port mode
    problem admits: on a symmetry plane, those of the plane's parity.

    Args:
        model: The model and its wave ports.
        stacklevel: Passed on to `warnings.warn`, counted from the caller.
    """
    for surface, ports in wave_port_surfaces(model.ports).items():
        requested = {p.mode for p in ports}
        skipped = [k for k in range(1, max(requested)) if k not in requested]
        if not skipped:
            continue
        label = wave_port_label(ports, lone_surface=True, modes=True)
        msg = (
            f"{label}: the surface skips {_modes(skipped)}, which no wave port "
            "on it requests. A skipped port mode propagates wherever the highest "
            "requested mode does, and it is partly reflected at the port instead "
            "of absorbed, so S misses the power the device sends into it. Add a "
            f"WavePort on surface {surface!r} for each skipped mode, or filter "
            "this warning if the reflection is acceptable."
        )
        _mpi.warn(comm, msg, SkippedPortModeWarning, stacklevel=stacklevel + 1)


#: The warnings a driven solve gives about its model and ports, whatever
#: it solves: a second solve of the same sweep (a frequency worker, an
#: estimate's adjoint) ignores them, since the first already gave them.
FORWARD_WARNINGS = (
    FloquetOrderCapWarning,
    UnrequestedOrderWarning,
    ObliqueAbsorberWarning,
    LowFrequencyWarning,
    SkippedPortModeWarning,
    ThinLayerWarning,
)


def _layer_stretch(model: Model, grid: Sequence[float]) -> Stretch | None:
    """The layer's stretch over the sweep's ``grid``; warns a thin layer."""
    stretch = _pml.stretch(model, grid)
    if stretch is not None:
        stretch.warn_thin()
    return stretch


def _floquet_prepass(
    model: Model,
    problem: Driven,
    options: SolverOptions,
    grid: tuple[float, ...],
    excite: Sequence[str],
) -> dict[str, _floquet.SurfaceOrders]:
    """The Floquet order table over ``grid``, its warnings and the excitation check.

    The parent's pre-pass, before any assembly: at a fixed angle
    each frequency's row reads its own k(f). Empty without
    Floquet ports.
    """
    orders = _floquet.order_table(model, grid, wave_vectors(problem, grid), options)
    if orders:
        _floquet.warn_order_cap(orders, options.floquet_order_tolerance)
        _floquet.warn_unrequested(orders)
        check_excitable(orders, excite, grid)
    return orders


def wave_vector_at(
    problem: Driven, frequency: float
) -> tuple[float, float, float] | None:
    """The Bloch wave vector, rad/m, of ``problem``'s solve at ``frequency``.

    ``problem.wave_vector`` itself, or at a fixed angle that
    vector times ``frequency / wave_vector_frequency``; ``None`` for plain
    periodicity.
    """
    k, reference = problem.wave_vector, problem.wave_vector_frequency
    if k is None or reference is None:
        return k
    return (
        k[0] * frequency / reference,
        k[1] * frequency / reference,
        k[2] * frequency / reference,
    )


def wave_vectors(problem: Driven, frequency: Sequence[float]) -> Any:
    """``problem``'s wave vector over ``frequency``: one, None, or one per frequency.

    The form `gfem._floquet.order_table` takes: one per frequency only at a
    fixed angle.
    """
    if problem.wave_vector_frequency is None:
        return problem.wave_vector
    return [wave_vector_at(problem, f) for f in frequency]


def _auto_inputs(
    model: Model, problem: Driven, options: SolverOptions, comm: Any
) -> dict[str, Any]:
    """What ``linear_solver="auto"`` reads on one CPU process.

    Empty when the rule does not apply, so the unknowns are counted only
    when it does.
    """
    if options.linear_solver != "auto" or options.device != "cpu" or comm is not None:
        return {}
    workers = 1 if problem.sweep == "adaptive" else options.frequency_workers
    return {
        "unknowns": nedelec_unknowns(model.mesh, model._domain, options.order),  # noqa: SLF001
        # each frequency worker holds its own factorization
        "in_flight": min(workers, len(problem.frequency)),
        "general": problem.wave_vector is not None,
        "budget": default_retain_memory(),
        "pmg": _pmg_inputs(model, problem, options),
    }


def _pmg_inputs(
    model: Model, problem: Driven, options: SolverOptions
) -> PmgInputs | None:
    """The p-multigrid's inputs to the rule, or ``None`` where it does not apply.

    ``pmg_gmres`` solves order 2 and 3, an operator equal to its transpose
    (no Bloch wave vector), no periodic pair or Floquet port, and an
    ungauged solve (``gauge="tree_cotree"`` needs ``direct``). The direct
    set is counted on first call, once.
    """
    if (
        options.order not in PMG_BYTES_PER_UNKNOWN
        or problem.wave_vector is not None
        or model._periodic  # noqa: SLF001
        or _floquet.floquet_ports_in(model.ports)
        or options.gauge == "tree_cotree"
    ):
        return None
    count = functools.cache(functools.partial(_direct_set, model, options.order))
    return PmgInputs(options.order, memory_limit(), count)


def _direct_set(model: Model, order: int) -> DirectSetCount:
    """The direct set of ``model``'s electromagnetic domain at ``order``, counted.

    The cells `solve_cells` hands the builder, in the mesh's order, so the
    count is the built set's.
    """
    mesh = model.mesh
    active = model._domain_of("em").active  # noqa: SLF001
    pml = None
    if model.pml is not None:
        tags = [mesh.region(name).tag for name in model.pml.regions]
        pml = np.isin(mesh.tetrahedron_tags[active], tags)
    return direct_set_count(mesh.nodes, mesh.tetrahedra[active], order, pml)


def _resolve(
    model: Model,
    problem: Driven,
    options: SolverOptions,
    *,
    comm: Any,
    gauge: GaugeChoice | None,
) -> tuple[ResolvedLinearSolver, GaugeChoice]:
    """The rule's linear solver and gauge choice, before any assembly.

    The gauge choice is ``gauge`` when given (a parent rule's); otherwise
    ``options.gauge`` resolved (`resolve_gauge`) and logged. A refusal of
    ``linear_solver="auto"``'s iterative choice states the rule's numbers
    (`_auto_inputs`). Where the linear solver is known without the rule (an
    explicit one, or any under ``comm``), the gauge resolves first, so its
    refusal comes before the linear solver's (a missing PETSc stack).

    A perfectly matched layer stalls ``bddc_gmres``: an explicit one raises,
    and ``"auto"`` resolves to ``direct`` where neither direct's nor the
    p-multigrid's estimate fits.

    Raises:
        ValueError: ``linear_solver="bddc_gmres"`` with a perfectly matched
            layer, or ``"pmg_gmres"`` with periodic pairs or Floquet ports.
    """
    if options.linear_solver == "pmg_gmres":
        _refuse_pmg(model)
    layer = model.pml is not None
    if layer and options.linear_solver == "bddc_gmres":
        raise _pml.refuse_bddc(model.pml)
    inputs = _auto_inputs(model, problem, options, comm)
    known = _known_kind(options, comm, layer=layer)
    if gauge is None and known is not None:
        gauge = _resolve_gauge(model, problem, options, comm=comm, kind=known)
    linear = ResolvedLinearSolver.resolve(
        options,
        default="direct" if layer else DEFAULT_SOLVER,
        problem="driven",
        comm=comm,
        reason=_LAYER_DIRECT if layer else None,
        **inputs,
    )
    if gauge is not None:
        return linear, gauge
    auto_reason = None
    if inputs and linear.kind != "direct":
        auto_reason = auto_linear_solver(DEFAULT_SOLVER, **inputs).reason
    choice = _resolve_gauge(
        model, problem, options, comm=comm, kind=linear.kind, auto_reason=auto_reason
    )
    return linear, choice


def _refuse_pmg(model: Model) -> None:
    """Raise where ``pmg_gmres`` meets periodic pairs or Floquet ports.

    They have no path through it in this version. A Bloch wave vector is
    refused earlier, with the other solvers that factor symmetric operators
    only (`check_bloch`).
    """
    floquet = _floquet.floquet_ports_in(model.ports)
    periodic = model._periodic  # noqa: SLF001
    if floquet or periodic:
        label = f"FloquetPort {floquet[0].name!r}" if floquet else periodic[0].label
        raise ValueError(
            "linear_solver='pmg_gmres' does not solve models with periodic pairs "
            f"or Floquet ports in this version of gfem ({label}). Use "
            "linear_solver='direct' (or 'auto', which picks it while it fits)."
        )


#: Why ``linear_solver="auto"`` falls back on ``direct`` for a model with a
#: layer.
_LAYER_DIRECT = (
    "the perfectly matched layer excludes 'bddc_gmres', whose iterations stall "
    "on its stretch"
)


def _known_kind(
    options: SolverOptions, comm: Any, *, layer: bool = False
) -> str | None:
    """The linear solver ``options`` resolve to without the rule, or ``None``.

    ``None`` for ``"auto"`` on one CPU process, which `auto_linear_solver`
    decides from the system size; across ranks, ``direct`` with a perfectly
    matched ``layer``.
    """
    if options.linear_solver != "auto":
        return options.linear_solver
    if options.device != "cpu":
        return "direct"
    if comm is None:
        return None
    return "direct" if layer else DEFAULT_SOLVER


def _resolve_gauge(
    model: Model,
    problem: Driven,
    options: SolverOptions,
    *,
    comm: Any,
    kind: str,
    auto_reason: str | None = None,
) -> GaugeChoice:
    """``options.gauge`` resolved for a ``kind`` solve (`resolve_gauge`), logged."""
    choice = resolve_gauge(
        options.gauge,
        linear_solver=options.linear_solver,
        kind=kind,
        device=options.device,
        auto_reason=auto_reason,
        distributed=comm is not None,
        periodic=any(isinstance(b, Periodic) for b in model.boundaries),
        bloch=problem.wave_vector is not None,
        floquet=bool(_floquet.floquet_ports_in(model.ports)),
    )
    report.gauge_choice(problem="driven", on=choice.on, reason=choice.reason)
    return choice


def rule_class(model: Model) -> type[DrivenRule]:
    """The driven rule of ``model``'s mesh: `DrivenRule`, or a 2D mesh's."""
    if model.mesh.dim == 2:
        from gfem._engine.driven2d import Driven2DRule

        return Driven2DRule
    return DrivenRule


def _sample_error(
    frequencies: tuple[float, ...], omega: float, err: LinearSolverConvergenceError
) -> LinearSolverConvergenceError:
    """An adaptive sweep's error where a sample's linear solve did not converge.

    No partial result: every output of the reduced model depends on every
    sample. ``frequency`` is the sample's grid frequency.
    """
    f = min(frequencies, key=lambda g: abs(2 * math.pi * g - omega))
    return LinearSolverConvergenceError(
        f"At the adaptive sweep's sample {f:g} Hz, {err} The reduced model needs "
        "every sample, so no frequency has a result: use linear_solver='direct', "
        "or a uniform sweep to find the frequencies that converge.",
        frequency=f,
    )


def _adjoint_error(
    f: float, err: LinearSolverConvergenceError
) -> LinearSolverConvergenceError:
    """A backward pass's error where the adjoint solve at ``f`` Hz did not converge.

    No partial result: a gradient needs every frequency's adjoint.
    """
    return LinearSolverConvergenceError(
        f"At {f:g} Hz, the adjoint solve of the gradient's backward pass: {err}",
        frequency=f,
    )


def _estimate_adjoint_error(
    f: float, keys: str, err: LinearSolverConvergenceError
) -> LinearSolverConvergenceError:
    """An error estimate's error where its adjoint solve at ``f`` Hz did not converge.

    The forward solve converged there; no partial result, since the estimate
    needs the adjoint at every frequency. ``keys`` names the loads (a port's
    name for its S read-out).
    """
    return LinearSolverConvergenceError(
        f"At {f:g} Hz, the adjoint solve of {keys} for the error estimate: {err}",
        frequency=f,
    )


#: Singular values of a reduced cotangent matrix below this fraction of the
#: largest one are dropped (their pairs add nothing).
FACTOR_RANK = 1e-14


def _factor(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(a, b)`` with ``g = a @ b.T`` and few columns (truncated SVD).

    A matrix with a non-finite entry (an S row of a port without a
    resistance) gives a NaN column, so the gradient is NaN like the value.
    """
    if not np.isfinite(g).all():
        # complex, like the factors of a finite matrix: the pair's fields
        # lie in the operator's complex space
        column = np.full((g.shape[0], 1), np.nan, dtype=complex)
        return column, column
    u, sigma, vh = np.linalg.svd(g)
    rank = max(1, int(np.count_nonzero(sigma > FACTOR_RANK * sigma[0])))
    return u[:, :rank] * sigma[:rank], vh[:rank].T


@dataclasses.dataclass(eq=False)
class _Kept:
    """A forward's system operator and port network, kept for its backward pass.

    The cache entry's ``handle`` (`gfem._engine.cache.Entry`): a backward
    pass reuses the adjoint handles and the pieces, and runs no port mode
    sweep again.
    """

    system: SystemOperator
    ports: PortNetwork
    _freed: bool = dataclasses.field(default=False, init=False)

    @property
    def nbytes(self) -> int:
        """Host memory held, in bytes (0 once freed), but the shared objects.

        Wave port couplings, the port network, kept fixed parts and patterns
        are in `shared`: a discretization or another entry may hold them, so
        the entry cache counts them once, and an entry's bytes do
        not depend on what the discretization still keeps.
        """
        return 0 if self._freed else self.system.nbytes

    def shared(self) -> dict[int, int]:
        """The shared objects' bytes, by identity (empty once freed)."""
        if self._freed:
            return {}
        return {**self.system.shared(), **self.ports.shared()}

    def free_base(self, comm: Any, *, walk: bool = True) -> None:
        """Give up the base handle kept for a backward pass, if it is on ``comm``.

        A base from the plan pool returns there, with its factors,
        and the pool walks unless not to ``walk`` (`EntryCache.give_plan`);
        any other is freed. Freeing may be collective on a backend, so an entry
        of another communicator stays: its ranks are not here.
        """
        if not self._freed and _mpi.comm_key(
            self.system.operator.built.comm
        ) == _mpi.comm_key(comm):
            self.system.free_base(return_plan=True, walk=walk)

    def free(self, *, return_plans: bool = False) -> None:
        """Release the system operator's handles and pieces, and drop both.

        With ``return_plans`` (a consumed entry) the kept base handle returns
        to the plan pool, where it came from one. A second call does nothing.
        """
        if self._freed:
            return
        self.system.free(return_plans=return_plans)
        del self.system, self.ports
        self._freed = True


@dataclasses.dataclass(frozen=True)
class PortDesign:
    """What the wave ports' modes read.

    A wave port's mode follows the inputs its cross-section reads: the
    materials of the regions on its face, a permittivity law's parameters, and
    a sheet field or `Impedance` sharing an edge with the face. `reading` is the
    map of those inputs: the driven rule differentiates them through the
    port pass (`gfem._engine.port_pass`) and only gates on it, while the
    eigenmode rule, whose wave port term holds its mode, refuses them with
    `check`.
    """

    #: Per wave port, the volume regions with an element on its face: the
    #: port mode reads their materials.
    regions: dict[str, tuple[str, ...]]
    #: Per wave port, the wave ports of its surface (several on a shared port
    #: surface).
    groups: dict[str, Sequence[WavePort]]
    #: Per sheet field or `Impedance` (index into ``model.boundaries``), the
    #: wave ports whose face shares an edge with its surface: the port mode
    #: reads it there.
    sheets: dict[int, tuple[str, ...]]
    #: The wave ports whose face shares an edge with an `Absorbing` surface.
    absorbing: frozenset[str] = frozenset()

    @classmethod
    def of(cls, model: Model, *, sheets: bool = True) -> PortDesign:
        """The port design of ``model``; without ``sheets``, no sheet map."""
        return cls(
            regions=_port_regions(model),
            groups={
                p.name: group
                for group in wave_port_surfaces(model.ports).values()
                for p in group
            },
            sheets=_sheet_ports(model) if sheets else {},
            absorbing=_absorbing_ports(model) if sheets else frozenset(),
        )

    def reading(self, kind: tuple) -> tuple[str, ...]:
        """The wave ports whose mode reads input ``kind`` (none: empty)."""
        if (is_lumped(kind) and kind[0] == "boundary") or is_sheet(kind):
            return self.sheets.get(kind[1], ())
        if kind[0] not in ("material", "law"):
            return ()
        return tuple(p for p, regions in self.regions.items() if kind[1] in regions)

    def reads(self, wanted: Collection[tuple] | None, ports: Sequence[str]) -> bool:
        """Whether a wanted input is read by a mode of ``ports`` (``None``: all)."""
        if wanted is None:
            return True
        return any(set(self.reading(k)) & set(ports) for k in wanted)

    def check(self, kind: tuple) -> None:
        """Raise when input ``kind`` is one a wave port's mode reads.

        The eigenmode rule's refusal: its wave port term at a complex
        eigenfrequency holds the mode, so the mode's own derivative is
        missing.

        Raises:
            PortGeometryError: ``kind`` is the material, or a law parameter,
                of a region that touches a wave port face, or the fill of a
                sheet or a lumped value of an `Impedance` beside one.
        """
        ports = self.reading(kind)
        if not ports:
            return
        who = "an Eigenmode solve"
        why = "the wave port's term at a complex eigenfrequency holds its mode"
        keep = (
            "A gfem.Driven solve differentiates a design at a wave port face. "
            "For an eigenmode problem, keep the design away from wave port faces"
        )
        label = self._label(ports[0])
        if is_lumped(kind) and kind[0] == "boundary":
            raise PortGeometryError(
                f"{label}: the Impedance model.boundaries[{kind[1]}] shares an "
                f"edge with the port face, so {who} cannot differentiate with "
                f"respect to its {kind[2]}: {why}. {keep} (split the surface so "
                "the part touching the port is a boundary of its own), or wrap "
                "the value in jax.lax.stop_gradient."
            )
        if is_sheet(kind):
            raise PortGeometryError(
                f"{label}: the sheet field of model.boundaries[{kind[1]}] shares "
                f"an edge with the port face, so {who} cannot differentiate with "
                f"respect to its fill: {why}. {keep} (split the sheet so the part "
                "touching the port is a boundary of its own without fill), or "
                "wrap the fill in jax.lax.stop_gradient."
            )
        region = kind[1]
        leaf = (
            f"the permittivity law of model.materials[{region!r}]"
            if kind[0] == "law"
            else f"model.materials[{region!r}].{kind[2]}"
        )
        raise PortGeometryError(
            f"{label}: region {region!r} has elements on the port face, so {who} "
            f"cannot differentiate with respect to {leaf}: {why}. {keep} (give "
            "the elements beside the port a region of their own with a constant "
            "gfem.Material), or wrap the value in jax.lax.stop_gradient."
        )

    def _label(self, port: str) -> str:
        """``WavePort 'p1'``, or a shared port surface's ports and surface."""
        return wave_port_label(self.groups[port])


def _sheet_ports(model: Model) -> dict[int, tuple[str, ...]]:
    """Per sheet-field or `Impedance` boundary index, the wave ports beside it.

    The wave ports whose face shares an edge with the boundary's surface.
    """
    mesh = model.mesh
    out = {}
    for i, b in enumerate(model.boundaries):
        if sheet_of(b) is None and not isinstance(b, Impedance):
            continue
        (surface,) = surfaces_of(b)
        sheet = _edges(mesh.triangles[mesh.elements(surface)])
        ports = tuple(
            p.name
            for p in model.ports
            if isinstance(p, WavePort)
            and sheet & _edges(mesh.triangles[mesh.elements(p.surface)])
        )
        if ports:
            out[i] = ports
    return out


def _absorbing_ports(model: Model) -> frozenset[str]:
    """The wave ports whose face shares an edge with an `Absorbing` surface."""
    mesh = model.mesh
    edges: set[tuple[int, int]] = set()
    for b in model.boundaries:
        if isinstance(b, Absorbing):
            for surface in surfaces_of(b):
                edges |= _edges(mesh.triangles[mesh.elements(surface)])
    if not edges:
        return frozenset()
    return frozenset(
        p.name
        for p in model.ports
        if isinstance(p, WavePort)
        and edges & _edges(mesh.triangles[mesh.elements(p.surface)])
    )


def _edges(triangles: np.ndarray) -> set[tuple[int, int]]:
    pairs = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    return {(int(a), int(b)) for a, b in pairs}


def _port_regions(model: Model) -> dict[str, tuple[str, ...]]:
    """Per wave port, the solve-domain volume regions beside its face."""
    mesh = model.mesh
    active = model._domain.active  # noqa: SLF001
    names = mesh._region_names()  # noqa: SLF001
    out = {}
    for port in model.ports:
        if not isinstance(port, WavePort):
            continue
        topology = mesh._topology  # noqa: SLF001 - a 2D mesh's lines too
        on = topology.surface_tags == mesh.region(port.surface).tag
        nb = topology.surface_neighbours[on]
        tets = nb[nb >= 0]
        tags = np.unique(topology.volume_tags[tets[active[tets]]])
        out[port.name] = tuple(names[int(t)] for t in tags)
    return out


def _outgoing(
    ports: PortNetwork, n: int, omega: float, x: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """The outgoing waves of the port columns and of the source columns of ``x``.

    A source column's are de-embedded at their port (its row), as S is.
    """
    waves = ports.outgoing(omega, x[:, width:]) if x.shape[1] > width else 0
    return (
        ports.outgoing(omega, x[:, :width]),
        ports.deembedding[n][:, None] * waves,
    )


def _rhs(
    ports: PortNetwork,
    sources: SourceColumns | None,
    omega: float,
    system: SystemOperator,
    width: int,
) -> np.ndarray:
    """The right-hand sides at ω: the excited ports', then the sources'."""
    if width == 0:
        rhs = np.zeros((system.operator.space.ndof, 0), complex)
    else:
        rhs = ports.rhs(omega)
    if sources is None:
        return rhs
    return np.concatenate([rhs, sources.rhs(omega)], axis=1)


def _power_adjoint(  # noqa: PLR0917 - one frequency's columns, in place
    sources: SourceColumns,
    omega: float,
    ct: np.ndarray,
    layout: DofLayout,
    x: np.ndarray,
    adjoint: np.ndarray,
    load: np.ndarray,
    width: int,
) -> None:
    """Fold an emitted power cotangent into ``adjoint`` and ``load``, in place.

    Its load on dipole k's column is parallel to the dipole's own load when
    the moment is real up to one phase: the adjoint is then the dipole's
    solution scaled (the self-adjoint shortcut), no solve. Otherwise (a
    circular moment) it is one more adjoint load. ``width`` is the excited
    ports' columns, before the sources'.
    """
    power = sources.power_loads(omega, ct)
    rhs = sources.rhs(omega)
    for k in range(power.shape[1]):
        # a dipole's load is on the rank holding its element: decide together
        if not _mpi.agree_any(layout.comm, flag=bool(power[:, k].any())):
            continue
        j = width + k
        scale = parallel_scale(power[:, k], rhs[:, k], layout)
        if scale is not None:
            adjoint[:, j] += scale * x[:, j]
        else:
            load[:, j] += power[:, k]


def _reciprocal_loads(  # noqa: PLR0917 - one frequency's columns, in place
    reading: PortSolutions,
    columns: Sequence[int],
    weighted: Callable[[float, np.ndarray], np.ndarray],
    omega: float,
    ct: np.ndarray,
    load: np.ndarray,
) -> None:
    r"""Fold the cotangent of port waves read by reciprocity into ``load``, in place.

    With :math:`W_{ic} = d_i\, b_c^\mathsf{T} z_i` and ``ct`` the cotangent
    :math:`\bar W_{ic} d_i`, the derivative through the operator is
    :math:`-\sum_i z_i^\mathsf{T}\, \partial A\, y_i` with :math:`y_i =
    A^{-1} \sum_c \bar W_{ic} d_i b_c` (``weighted``, one column per port
    with a nonzero row): it loads the column that holds :math:`z_i`, a
    solved load column, or an excited port's column :math:`x_j` with
    :math:`z_i = s\, x_j`, scaled by :math:`s` (`PortSolutions`). A port
    with neither has no own solution (`_reciprocal_defined`).
    """
    rows = np.flatnonzero(np.any(ct != 0, axis=1))
    if not rows.size:
        return
    width = len(columns)
    for row, b in zip(rows, weighted(omega, ct[rows]).T, strict=True):
        i = int(row)
        scale = reading.own.get(i)
        if i in reading.ports:
            load[:, width + reading.ports.index(i)] += b
        elif scale is not None:
            load[:, list(columns).index(i)] += scale * b


def _reciprocal_defined(reading: PortSolutions, ct: np.ndarray) -> bool:
    """Whether every port with a nonzero cotangent row has an own solution.

    A port without a defined wave (a lumped port without a resistance) has a
    NaN row of port waves: a cotangent on it gives a NaN gradient.
    """
    rows = np.flatnonzero(np.any(ct != 0, axis=1))
    return all(i in reading.ports or reading.own.get(int(i)) is not None for i in rows)


def _adaptive_loads(
    ports: PortNetwork, sources: SourceColumns | None, size: int
) -> Callable[[float], np.ndarray]:
    """The adaptive sweep's adjoint loads: the ports', then the dipoles'."""

    def loads(omega: float) -> np.ndarray:
        parts = [
            ports.adjoint_loads(omega)
            if ports.outputs
            else np.zeros((size, 0), complex)
        ]
        if sources is not None:
            parts.append(sources.adjoint_loads(omega))
        return np.concatenate(parts, axis=1)

    return loads


def _check_supported(model: Model) -> None:
    """Raise for ports and absorbing boundaries this version cannot solve."""
    for port in model.ports:
        if not isinstance(port, (LumpedPort, WavePort, FloquetPort)):
            raise TypeError(f"Unknown port {port!r}.")  # pragma: no cover
    # a law region behind an absorbing face takes its face pieces
    terms.check_absorbing(model, dispersive=True)
