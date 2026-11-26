"""Solver options shared by every problem type."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from typing import Any, Literal

__all__ = ["SolverOptions"]

LinearSolver = Literal["auto", "bddc_gmres", "direct", "petsc", "pmg_gmres"]
_LINEAR_SOLVERS = ("auto", "bddc_gmres", "direct", "petsc", "pmg_gmres")
EigenSolver = Literal["auto", "arpack", "slepc"]
_EIGEN_SOLVERS = ("auto", "arpack", "slepc")
TimeScheme = Literal["tr_bdf2", "theta"]
_TIME_SCHEMES = ("tr_bdf2", "theta")
Gauge = Literal["auto", "tree_cotree", "none"]
_GAUGES = ("auto", "tree_cotree", "none")
#: A PETSc option's value: ``None`` is a flag with no value.
PetscValue = str | int | float | bool | None
#: ``SolverOptions.petsc_options`` as stored: pairs sorted by name.
PetscOptions = tuple[tuple[str, PetscValue], ...]
#: ``"cpu"``, ``"cuda"``, or ``"cuda:N"`` for the N-th GPU (hence ``| str``).
Device = Literal["cpu", "cuda"] | str  # noqa: PYI051 - names the choices
_DEVICE = re.compile(r"cpu|cuda(:\d+)?")


def _one_of(value: object, name: str, choices: tuple[str, ...]) -> None:
    """Raise `ValueError` unless ``value`` is one of ``choices``."""
    if value not in choices:
        raise ValueError(
            f"SolverOptions.{name} must be one of "
            f"{', '.join(repr(c) for c in choices)}, got {value!r}."
        )


def _pmg_order(order: int) -> None:
    """Raise `ValueError` unless ``order`` is one the p-multigrid solves: 2 or 3."""
    if order == 1:
        raise ValueError(
            "linear_solver='pmg_gmres' needs order 2 or 3, got order 1: at order 1 "
            "its direct level is the whole system. Use linear_solver='direct'."
        )
    if order >= 4:
        raise ValueError(
            f"linear_solver='pmg_gmres' needs order 2 or 3, got order {order}: its "
            "vertex-star blocks above order 3 are not verified in this version of "
            "gfem. Use linear_solver='direct' or 'bddc_gmres'."
        )


def _int(
    value: object, name: str, low: int, high: int | None = None, why: str = ""
) -> int:
    ok = isinstance(value, int) and not isinstance(value, bool) and value >= low
    if not ok or (high is not None and value > high):
        span = f">= {low}" if high is None else f"from {low} to {high}"
        raise ValueError(
            f"SolverOptions.{name} must be an integer {span}, got {value!r}.{why}"
        )
    return value


def _tol(value: object, name: str) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not ok or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"SolverOptions.{name} must be in (0, 1), got {value!r}.")
    return float(value)


_PETSC_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
#: Option names that ``eigen_solver="slepc"`` reads (SLEPc's objects); every
#: other name is read by ``linear_solver="petsc"``.
SLEPC_NAMESPACES = ("eps_", "bv_", "ds_", "rg_", "fn_")
#: The seed solve's EPS types: the others apply the shift-inverted operator
#: once per DOF (``lapack``) or swap its spectral transform (``gd``).
_EPS_TYPES = ("krylovschur", "arnoldi")
_SELECTION = (
    "is set by gfem: Eigenmode.target picks the modes (gfem keeps the "
    "eigenvalues nearest it); set that instead."
)
_PROBLEM_TYPE = (
    "is set by gfem: the eigen solver works on a non-Hermitian shift-inverted operator."
)
_CONVERGENCE = (
    "is set by gfem: eigen_tolerance is relative to each shift-inverted "
    "eigenvalue, as with ARPACK; set SolverOptions.eigen_tolerance instead."
)
_EXTRACTION = (
    "needs the operator's transpose or a target, which gfem's shift-inverted "
    "operator does not give SLEPc."
)
#: Options gfem sets itself, and what to set instead (the message's tail).
_RESERVED: dict[str, str] = {
    "ksp_rtol": (
        "is set from SolverOptions.tolerance (adjoint_tolerance for adjoint "
        "solves); set that instead."
    ),
    "ksp_atol": (
        "is reserved: gfem stops a Krylov solve on the relative residual only, "
        "SolverOptions.tolerance relative to the right-hand side; set that "
        "instead."
    ),
    "ksp_initial_guess_nonzero": (
        "is set per solve from the initial guess gfem passes; remove it."
    ),
    "eps_tol": "is set from SolverOptions.eigen_tolerance; set that instead.",
    "eps_nev": "is set from Eigenmode.n_modes; set that instead.",
    "eps_target": "is set from Eigenmode.target; set that instead.",
    **dict.fromkeys(
        (
            "eps_largest_magnitude",
            "eps_smallest_magnitude",
            "eps_largest_real",
            "eps_smallest_real",
            "eps_largest_imaginary",
            "eps_smallest_imaginary",
            "eps_target_magnitude",
            "eps_target_real",
            "eps_target_imaginary",
            "eps_all",
            "eps_which_user",
            "eps_interval",
            "eps_threshold_absolute",
            "eps_threshold_relative",
        ),
        _SELECTION,
    ),
    **dict.fromkeys(
        (
            "eps_hermitian",
            "eps_gen_hermitian",
            "eps_non_hermitian",
            "eps_gen_non_hermitian",
            "eps_pos_gen_non_hermitian",
            "eps_gen_indefinite",
            "eps_bse",
            "eps_hamiltonian",
            "eps_lrep",
        ),
        _PROBLEM_TYPE,
    ),
    **dict.fromkeys(
        ("eps_conv_abs", "eps_conv_rel", "eps_conv_norm", "eps_conv_user"),
        _CONVERGENCE,
    ),
    "eps_two_sided": _EXTRACTION,
    "eps_ritz": _EXTRACTION,
}
#: Reserved option families, by name prefix.
_RESERVED_PREFIXES: dict[str, str] = {
    "eps_harmonic": _EXTRACTION,
    "eps_refined": _EXTRACTION,
    "st_": (
        "has no effect: gfem applies T(σ)⁻¹ through linear_solver, not "  # noqa: RUF001
        "through a SLEPc spectral transform; configure that solver instead "
        "(with linear_solver='petsc', its ksp_ / pc_ options)."
    ),
}


def _is_slepc(name: str) -> bool:
    return name.startswith(SLEPC_NAMESPACES)


def split_petsc_options(pairs: PetscOptions) -> tuple[PetscOptions, PetscOptions]:
    """``SolverOptions.petsc_options`` split into its linear and eigen options.

    Names in `SLEPC_NAMESPACES` go to the eigen solver (SLEPc), every other
    name to the linear solver (PETSc).

    Returns:
        ``(linear, eigen)``, each in the order of ``pairs``.
    """
    linear = tuple(p for p in pairs if not _is_slepc(p[0]))
    eigen = tuple(p for p in pairs if _is_slepc(p[0]))
    return linear, eigen


def _petsc_pairs(value: object) -> list[tuple[Any, Any]]:
    """The ``(name, value)`` pairs of a mapping or of the stored pairs."""
    if isinstance(value, Mapping):
        return list(value.items())
    if isinstance(value, tuple) and all(
        isinstance(p, tuple) and len(p) == 2 for p in value
    ):
        return list(value)
    raise ValueError(
        "SolverOptions.petsc_options must map PETSc option names to values, e.g. "
        f"{{'ksp_type': 'gmres'}}, got {type(value).__name__}."
    )


def _petsc_options(options: SolverOptions) -> PetscOptions:
    """Check ``options.petsc_options`` and return its stored form."""
    pairs = _petsc_pairs(options.petsc_options)
    seen: set[str] = set()
    for name, value in pairs:
        field = f"SolverOptions.petsc_options[{name!r}]"
        if not isinstance(name, str) or not _PETSC_NAME.fullmatch(name):
            raise ValueError(
                "SolverOptions.petsc_options keys are PETSc option names without "
                f"the leading '-', e.g. 'ksp_type', got {name!r}."
            )
        if name in seen:
            raise ValueError(f"SolverOptions.petsc_options names {name!r} twice.")
        seen.add(name)
        # exact types: NumPy scalars (float64 is a float) would not stay plain JSON
        if value is not None and type(value) not in (str, int, float, bool):
            raise ValueError(
                f"{field} must be a str, int, float, bool or None, got "
                f"{type(value).__name__}."
            )
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"{field} must be finite, got {value!r}.")
        reserved = _RESERVED.get(name) or next(
            (t for p, t in _RESERVED_PREFIXES.items() if name.startswith(p)), None
        )
        if reserved is not None:
            raise ValueError(f"{field} {reserved}")
        if _is_slepc(name):
            if options.eigen_solver != "slepc":
                raise ValueError(f"{field} is read only by eigen_solver='slepc'.")
            if name == "eps_ncv" and (type(value) is not int or value < 1):
                # gfem reads it (floored at nev + 1 at the solve), not SLEPc
                raise ValueError(f"{field} must be an integer >= 1, got {value!r}.")
            if name == "eps_type" and value not in _EPS_TYPES:
                raise ValueError(
                    f"{field} must be 'krylovschur' or 'arnoldi' on gfem's "
                    f"shift-inverted operator, got {value!r}."
                )
        elif options.linear_solver != "petsc":
            slepc = (
                "; eigen_solver='slepc' applies the solves of linear_solver="
                f"{options.linear_solver!r}"
                if options.eigen_solver == "slepc"
                else ""
            )
            raise ValueError(f"{field} is read only by linear_solver='petsc'{slepc}.")
    return tuple(sorted(pairs, key=lambda p: p[0]))


@dataclasses.dataclass(frozen=True, kw_only=True)
class SolverOptions:
    r"""Every numerical knob of a solve, kept apart from the model and problem.

    Frozen and keyword-only. The defaults suit most models; `preset` trades
    speed for accuracy in one word. Most solves need only ``order``,
    ``linear_solver`` and ``tolerance``, or a preset. The adaptive, eigen
    and PETSc fields matter only on those paths.

    Example:
        Finer settings in one word, or a direct solve at order 3::

            options = gfem.SolverOptions.preset("accurate")
            options = gfem.SolverOptions(linear_solver="direct", order=3)
            result = gfem.solve(model, problem, options=options)

    Attributes:
        order: Polynomial order of the Nédélec (H(curl)) and H1 elements, from
            1 to 12. Unknowns per tetrahedron grow as order^3 and matrix memory
            as order^6: order 12 already takes about 28 MiB and 3 s of assembly
            per tetrahedron.
        linear_solver: ``"auto"`` (the default) picks per problem type. For
            a driven problem on the CPU in one process, ``"direct"`` when its
            factorization (estimated at about 48 KiB per unknown, per
            frequency worker, twice that with a Bloch wave vector) fits in
            half the memory the process may use; else ``"pmg_gmres"`` for a
            3D model at order 2 or 3 with no Bloch wave vector, periodic
            pair, Floquet port or ``gauge="tree_cotree"``, when its estimate
            (about 8 KiB per unknown at order 2, 16 KiB at order 3, plus
            48 KiB per higher-order unknown on its direct set, per frequency
            worker) fits in all of that memory; else ``"bddc_gmres"``, or
            ``"direct"`` with a perfectly matched layer; ``"bddc_gmres"`` for
            electrostatic; ``"direct"`` for eigenmode and for every problem
            type on a GPU (``device``); ``"bddc_gmres"`` for a driven problem
            across MPI ranks (eigenmode keeps ``"direct"``). The choice is
            logged on the ``gfem.solver`` logger. ``"direct"`` suits small
            and medium models on one machine: one factorization per frequency serves
            every port, and the ``mkl`` extra makes it faster (MKL PARDISO);
            for two-port lumped-port lines without the ``mkl`` extra,
            ``"bddc_gmres"`` can be faster. Pick ``"bddc_gmres"`` for large
            models or limited memory: an iterative solve to ``tolerance``,
            across MPI ranks and on periodic and Bloch models too. Pick
            ``"pmg_gmres"`` for driven models several wavelengths across
            that the direct solver cannot hold: photonic devices in
            dielectric cladding, with absorbing faces or a perfectly matched
            layer, and metal circuits. Its iterations stay near 10 as the
            model grows in wavelengths, where ``"bddc_gmres"``'s grow with
            the size. At order 2 it needs about a quarter of the direct
            solver's memory, and it is faster than the direct solver from a
            few million unknowns on. It solves driven problems at order 2 or
            3 on one process; a Bloch wave vector, periodic pairs, Floquet
            ports, a GPU and MPI ranks raise. See the "Solve a driven model
            too large for the direct solver" recipe.
            Pick ``"direct"`` explicitly to lift ``"auto"``'s memory guard.
            Pick ``"petsc"`` to use an
            existing PETSc build (the "Solve with PETSc and SLEPc" recipe):
            an exact MUMPS factorization by default, or the PETSc solver
            ``petsc_options`` configures; ``"auto"`` never picks it. A solver
            this Python cannot use raises `gfem.SolverUnavailableError` when
            the solve starts. With a Bloch wave vector (``wave_vector=`` on
            the problem), use ``"direct"``, or ``"bddc_gmres"`` for driven
            problems only; ``"petsc"`` raises. Under rank copies (one
            complete solve per process, ``comm=MPI.COMM_SELF``), use PETSc on
            every copy or on none: a job that mixes them has not been tried.
            With ``"bddc_gmres"`` across ranks, gfem factors BDDC's coarse
            problem on a share of the ranks, the same for the same script,
            ranks and stack, and not an option: S stays that of the coarse
            problem factored on one rank, to round-off. A singular coarse
            problem raises at once; only a MUMPS workspace error is retried,
            with more workspace.
        tolerance: Relative residual at which an iterative linear solve
            stops (see Notes for its norm). It is not the error of S or of
            the fields: with ``"bddc_gmres"`` at the default 1e-6, S was
            within 7e-7 of the exact solve on a short line and within 5e-5
            on a 100-frequency microstrip sweep. To judge the accuracy,
            solve again at a tighter ``tolerance`` and compare S. At low
            frequency the magnetic flux density of a field (`Field.b`) and
            `gfem.power_flow` divide by the angular frequency
            :math:`\omega`, so a residual left at the tolerance is
            amplified by :math:`1/\omega` and they lose more digits than
            S. On a 10 mm parallel-plate line at 500 kHz, the default left
            B 1.7 % and the power through a cut 2.8e-3 W (per W incident)
            off with ``"bddc_gmres"``, and 4e-4 and 1.2e-5 W with
            ``"pmg_gmres"``. Each error falls with the tolerance: to read
            them, solve again at 1e-8 or 1e-10 and compare the quantity
            read, not only S. A direct
            solve, the default for a driven problem that fits in memory, is
            exact, except in an optimization of a model above the reuse
            floor, where a design may be solved on the factorization of an
            earlier design (when ``retain_memory`` keeps it and the counted
            costs say that is cheaper) to a twentieth of this tolerance.
        device: Where the direct solver runs: ``"cpu"``, ``"cuda"`` or
            ``"cuda:N"`` (the N-th GPU). A GPU needs
            ``linear_solver="direct"`` or ``"auto"``, which is ``"direct"``
            on a GPU (NVIDIA cuDSS, from the ``cuda12`` or ``cuda13``
            extra). Its availability is checked when solving, so
            options built on a laptop can run on a GPU machine; a GPU that
            cannot be used raises `gfem.DeviceUnavailableError`, never a
            silent CPU fallback. See `gfem.cuda_available`.
        threads: CPU threads, or ``None`` for the solve threads:
            ``NGS_NUM_THREADS`` when set to a positive integer, else the physical
            cores the process may run on (its affinity mask), not every hardware
            thread of the machine. Inside your own ``ngs.TaskManager()`` a solve
            runs on that TaskManager's threads, whatever ``threads`` says:
            NGSolve cannot change the count while one is active. Run one solve
            at a time in a process: NGSolve's threads are shared by the whole
            process, so solves started from several Python threads are not
            supported. Use ``frequency_workers``, or separate processes.
            ``linear_solver="petsc"`` does not read it: MUMPS's BLAS follows
            ``OPENBLAS_NUM_THREADS``, which frequency workers set to their
            share.
            Across MPI ranks ``None`` takes ``NGS_NUM_THREADS`` if set, else
            shares each physical core of the rank's CPU mask with the other
            ranks of the node holding it; every rank then uses the smallest
            count over all ranks, so one narrow mask slows the job
            (``mpirun -n 2`` binds each rank to one core: add
            ``--bind-to none`` or pass ``threads``). The count is read at the
            first solve on a communicator, and gfem sets the BLAS threads
            itself without reading ``OMP_NUM_THREADS``.
        adaptive_tolerance: Error target of the adaptive frequency sweep;
            at least 10x ``tolerance``.
        adaptive_max_samples: Most frequencies the adaptive sweep solves.
        adaptive_memory: Consecutive samples under ``adaptive_tolerance``
            that end the adaptive sweep.
        port_mode_tolerance: Accuracy of each wave port's mode across the
            sweep: the backward error each mode reaches at every frequency
            of the sweep, and at each complex frequency an eigenmode solve
            evaluates (``PortModeResult.residual``).
        floquet_order_tolerance: Which evanescent diffraction orders a
            Floquet surface models: those whose round trip to the nearest
            structure decays by less than this (:math:`e^{-2|\gamma| d}`
            above it),
            besides every propagating order, up to what the face mesh
            resolves.
        retain_memory: Bytes gfem keeps between solves in one process: each
            differentiated solve's fields and solver for its gradients (its
            solution fields and, for a uniform driven sweep, its assembled
            system with its linear solver), and what later designs of a
            driven model reuse: solver plans, with one factorization per
            frequency while they fit (a later design at the same frequency
            may solve on it, see ``tolerance``), the assembled parts that do
            not depend on the design, the wave ports' couplings, and the
            iterative solvers' initial guesses (the previous design's
            solution, from which the next design's first frequency starts).
            Over budget, gfem drops the iterative solvers' initial guesses
            first, then kept factorizations, then those parts, then solver
            plans,
            then the oldest entries, which are recomputed when their gradient
            runs, and port couplings last. ``None`` is half of the memory the
            process may use (the physical memory, or a smaller job memory
            limit), except that the last factorization stays held, even over
            it, while the process's peak memory leaves a tenth of that memory
            free; an explicit value is a hard bound. 0 keeps nothing between
            solves and always recomputes.
            The budget bounds what gfem keeps, not the process's memory:
            meshes and spaces of the last four discretizations, JAX's own
            caches and memory the allocator has not returned are outside it.
            `gfem.release_memory` drops everything kept.
            Across MPI ranks the default compares the memory held by a node's
            ranks with half of the smallest node's memory; BDDC's coarse
            factorizations kept between solves count against it (with
            ``retain_memory=0`` each solve sets them up anew), and only what
            the solving communicator holds is evicted, so solves on a second
            communicator (rank copies) can keep memory past it.
        adjoint_tolerance: Relative residual of adjoint solves, in the norm
            ``tolerance`` names for each backend; ``None`` is ``tolerance``.
        eigen_tolerance: Backward error accepted for an eigenmode (the
            residual over :math:`|K| + |\omega| |C| + |\omega|^2 |M|`), in the
            eigen solver's first pass (ARPACK's, or SLEPc's ``eps_tol`` with
            ``eigen_solver="slepc"``) and in the refinement. For frequency
            and Q to many digits, and for eigenmode gradients, keep the
            default 1e-10 or go lower: 1e-6 can leave Q about 0.5 % off on
            a port-loaded resonator. The norms are set by the highest
            curl-curl modes, so frequency and Q can be much less accurate
            than this value. It is distinct from ``tolerance``, the residual
            of each linear solve.
        cluster_rtol: Relative distance under which eigenfrequencies form a
            degenerate cluster.
        max_refine_iterations: Most iterations of the eigenmode refinement.
        frequency_workers: Processes a uniform `gfem.Driven` sweep spreads
            its frequencies over, in contiguous blocks, each with
            ``threads // frequency_workers`` threads (at least 1; with
            ``threads=None``, of the solve threads). It applies to uniform
            driven sweeps only: adaptive sweeps, eigenmode and electrostatic
            solves ignore it. Memory grows about ``frequency_workers`` times,
            one solve per worker. S agrees with one worker's to the solver
            precision (each block's iterative solves start cold). Wave ports'
            modes are solved once, over the whole grid, before the workers
            start. Not available on a GPU device. The workers are
            spawned processes, so a script that sets it must guard its
            entry point with ``if __name__ == "__main__":``.
        eigen_solver: What finds an `gfem.Eigenmode` problem's modes near
            ``target`` from shift-invert solves: ``"arpack"`` (SciPy ARPACK,
            one process only), ``"slepc"`` (SLEPc's Krylov-Schur, through
            slepc4py, from the PETSc stack the "Solve with PETSc and SLEPc"
            recipe builds), or ``"auto"``: ARPACK on one process, SLEPc
            across MPI ranks, where ``"arpack"`` raises
            `gfem.DistributedUnsupportedError`. gfem refines and
            differentiates the modes the same way after either. Driven and
            electrostatic solves ignore it. SLEPc this Python cannot use
            raises `gfem.SolverUnavailableError` when the solve starts.
        petsc_options: PETSc and SLEPc option names, without the leading
            ``-`` (``-ksp_type`` raises), and their values, read by
            ``linear_solver="petsc"`` and ``eigen_solver="slepc"``; a value of ``None`` is a flag, a
            ``bool`` is ``true`` / ``false``. They override gfem's defaults
            (the MUMPS factorization is ``ksp_type preonly``, ``pc_type
            cholesky``), under each solver's own options prefix: never write
            one. ``ksp_type preonly`` measures no residual, so it takes only
            ``pc_type`` ``lu`` or ``cholesky``, without a pivot shift; any
            other preconditioner needs a Krylov ``ksp_type``; to try PETSc's
            ILU, also set ``pc_factor_mat_solver_type`` to ``petsc``, since
            gfem's default MUMPS has no ILU and ``pc_type`` ``ilu`` alone is
            refused. Names starting ``eps_``, ``bv_``, ``ds_``, ``rg_`` or
            ``fn_`` need
            ``eigen_solver="slepc"``, every other name needs
            ``linear_solver="petsc"``, so an option never goes to a solver
            that is not selected (``ksp_type`` with ``linear_solver="direct"``
            raises: select ``"petsc"``). Options gfem sets from
            its own fields (``ksp_rtol`` from ``tolerance``, ``eps_tol`` from
            ``eigen_tolerance``, ...) raise `ValueError`, and so does
            ``ksp_atol``: gfem stops on the relative residual only. An option the
            PETSc linear solver does not read (a typo, or an option of
            another configuration) raises `ValueError` when the solve
            starts; one SLEPc does not read raises it after the eigen
            solver's first pass. With ``ksp_reuse_preconditioner``, a reused
            preconditioner is set up again at the frequency or shift where
            a Krylov solve on it does not converge (for example at
            ``ksp_max_it``). Stored as ``(name, value)`` pairs sorted by
            name.
        time_scheme: How a `gfem.TransientThermal` problem steps in time:
            ``"tr_bdf2"`` (TR-BDF2, second order and L-stable: no ringing
            on stiff thin-metal modes; both stages on one factorization)
            or ``"theta"`` (the theta scheme with `theta`). Other problems
            ignore it.
        theta: The implicitness of ``time_scheme="theta"``, from 0.5
            (Crank-Nicolson) to 1 (backward Euler, the default). Ignored
            by ``"tr_bdf2"``.
        gauge: How a `gfem.Driven` solve treats the null space of the curl,
            the gradients :math:`\nabla \phi`, on which the curl-curl term
            vanishes and the system degenerates as
            :math:`k_0 h_{\min} \to 0`. ``"auto"`` (the default) solves
            through the tree-cotree gauge wherever it applies: the linear
            solver resolves to ``"direct"``, ``device="cpu"``, one process,
            and the model has no `gfem.Periodic` pair, Bloch wave vector or
            `gfem.FloquetPort`; any other driven solve runs ungauged, without
            a word. ``"tree_cotree"`` asks for the gauge and raises
            `ValueError` when the solve starts wherever it does not apply
            (`gfem.DistributedUnsupportedError` across MPI ranks), including
            an ``"auto"`` linear solver that chose an iterative solver; the
            message names every reason.
            ``"none"`` solves ungauged. The gauge writes the field as
            :math:`x = T y`, :math:`T` square and nonsingular: node potentials
            along a spanning tree, one floating potential per conductor other
            than its region's reference, and the remaining (cotree) edge
            unknowns, so the direct solve keeps its digits below the
            low-frequency floor (:math:`k_0 h_{\min} < 10^{-6}`) and
            `gfem.LowFrequencyWarning` does not fire. The wave ports' mode
            problem is not gauged and keeps its own floor: a gauged solve
            with a `gfem.WavePort` still warns there. Above the floor, S
            moves by :math:`10^{-13}` to :math:`10^{-10}` towards the exact
            answer; ``"none"`` restores the ungauged S bit for bit. Measured
            costs: a setup of about one factorization per new mesh topology
            (reused across shape steps) and about 10 % more per
            refactorization. Eigenmode,
            electrostatic, DC conduction, thermal and `gfem.PortMode` solves
            ignore it, even ``"tree_cotree"``.

    Notes:
        ``tolerance`` is a different norm per backend, so the same value
        does not give the same accuracy everywhere. With ``"bddc_gmres"``
        and ``"pmg_gmres"`` it is the relative preconditioned residual:
        :math:`|P(b - Ax)| / |P b|` as GMRES estimates it, or for conjugate
        gradients (electrostatic) the P-norm of the residual over its first
        value. The
        unpreconditioned residual :math:`|b - Ax| / |b|` can be orders larger at
        low frequency and does not measure the error. With ``"petsc"`` and
        a Krylov ``ksp_type`` it is PETSc's ``ksp_rtol`` on the
        unpreconditioned residual (unless ``petsc_options`` sets
        ``ksp_norm_type``). When you switch backend, solve once more at a
        tighter ``tolerance`` and check that S does not change.
    """

    order: int = 2
    linear_solver: LinearSolver = "auto"
    tolerance: float = 1e-6
    device: Device = "cpu"
    threads: int | None = None
    adaptive_tolerance: float = 1e-3
    adaptive_max_samples: int = 20
    adaptive_memory: int = 2
    port_mode_tolerance: float = 1e-6
    floquet_order_tolerance: float = 1e-4
    retain_memory: int | None = None
    adjoint_tolerance: float | None = None
    eigen_tolerance: float = 1e-10
    cluster_rtol: float = 1e-6
    max_refine_iterations: int = 50
    frequency_workers: int = 1
    eigen_solver: EigenSolver = "auto"
    petsc_options: Mapping[str, PetscValue] | PetscOptions = ()
    time_scheme: TimeScheme = "tr_bdf2"
    theta: float = 1.0
    gauge: Gauge = "auto"

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        _one_of(self.gauge, "gauge", _GAUGES)
        _one_of(self.time_scheme, "time_scheme", _TIME_SCHEMES)
        theta = self.theta
        ok = isinstance(theta, (int, float)) and not isinstance(theta, bool)
        if not ok or not 0.5 <= theta <= 1.0:
            raise ValueError(
                "SolverOptions.theta must be a number from 0.5 (Crank-Nicolson) to "
                f"1 (backward Euler), got {theta!r}: below 0.5 the theta scheme is "
                "unstable."
            )
        set_(self, "theta", float(theta))
        _int(
            self.order,
            "order",
            1,
            high=12,
            why=" Unknowns per tetrahedron grow as order^3 and matrix memory as "
            "order^6: order 12 already takes about 28 MiB and 3 s of assembly per "
            "tetrahedron.",
        )
        _one_of(self.linear_solver, "linear_solver", _LINEAR_SOLVERS)
        _one_of(self.eigen_solver, "eigen_solver", _EIGEN_SOLVERS)
        set_(self, "petsc_options", _petsc_options(self))
        for name in (
            "tolerance",
            "adaptive_tolerance",
            "port_mode_tolerance",
            "floquet_order_tolerance",
            "eigen_tolerance",
        ):
            set_(self, name, _tol(getattr(self, name), name))
        if self.adjoint_tolerance is not None:
            set_(
                self,
                "adjoint_tolerance",
                _tol(self.adjoint_tolerance, "adjoint_tolerance"),
            )
        set_(self, "cluster_rtol", _tol(self.cluster_rtol, "cluster_rtol"))
        if not isinstance(self.device, str) or not _DEVICE.fullmatch(self.device):
            raise ValueError(
                "SolverOptions.device must be 'cpu', 'cuda' or 'cuda:N', got "
                f"{self.device!r}."
            )
        if self.device.startswith("cuda") and self.linear_solver not in (
            "direct",
            "auto",
        ):
            raise ValueError(
                f"device={self.device!r} needs linear_solver='direct' or 'auto': "
                "only the direct solver runs on a GPU, not "
                f"{self.linear_solver!r}."
            )
        if self.linear_solver == "pmg_gmres":
            _pmg_order(self.order)
        if self.threads is not None:
            _int(self.threads, "threads", 1)
        _int(self.adaptive_max_samples, "adaptive_max_samples", 2)
        _int(self.adaptive_memory, "adaptive_memory", 1)
        if self.retain_memory is not None:
            _int(self.retain_memory, "retain_memory", 0)
        _int(self.max_refine_iterations, "max_refine_iterations", 1)
        _int(self.frequency_workers, "frequency_workers", 1)
        if self.device.startswith("cuda") and self.frequency_workers > 1:
            raise ValueError(
                f"device={self.device!r} needs frequency_workers=1: frequency "
                "workers would share one GPU's memory."
            )
        if self.adaptive_tolerance < 10 * self.tolerance:
            raise ValueError(
                f"SolverOptions.adaptive_tolerance={self.adaptive_tolerance:g} must be "
                f"at least 10 x tolerance={self.tolerance:g}: the sweep cannot be "
                "more accurate than the solves it samples."
            )

    @classmethod
    def preset(cls, precision: Literal["fast", "default", "accurate"]) -> SolverOptions:
        """Options in one word.

        ``"fast"``: order 1. ``"default"``: the defaults. ``"accurate"``:
        order 3, tolerance 1e-10, adaptive tolerance 1e-5.

        Raises:
            ValueError: Unknown preset.
        """
        presets: dict[str, dict[str, Any]] = {
            "fast": {"order": 1},
            "default": {},
            "accurate": {"order": 3, "tolerance": 1e-10, "adaptive_tolerance": 1e-5},
        }
        if precision not in presets:
            raise ValueError(
                f"Unknown preset {precision!r}; use one of "
                f"{', '.join(repr(p) for p in presets)}."
            )
        return cls(**presets[precision])
