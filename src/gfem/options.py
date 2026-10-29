"""Solver options shared by every problem type."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from typing import Any, Literal

__all__ = ["SolverOptions"]

LinearSolver = Literal["auto", "bddc_gmres", "direct", "petsc"]
_LINEAR_SOLVERS = ("auto", "bddc_gmres", "direct", "petsc")
EigenSolver = Literal["auto", "arpack", "slepc"]
_EIGEN_SOLVERS = ("auto", "arpack", "slepc")
#: A PETSc option's value: ``None`` is a flag with no value.
PetscValue = str | int | float | bool | None
#: ``SolverOptions.petsc_options`` as stored: pairs sorted by name.
PetscOptions = tuple[tuple[str, PetscValue], ...]
#: ``"cpu"``, ``"cuda"``, or ``"cuda:N"`` for the N-th GPU (hence ``| str``).
Device = Literal["cpu", "cuda"] | str  # noqa: PYI051 - names the choices
_DEVICE = re.compile(r"cpu|cuda(:\d+)?")


def _int(value: object, name: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(
            f"SolverOptions.{name} must be an integer >= {low}, got {value!r}."
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
    "is set by gfem: Eigenmode.target picks the modes (the seed keeps the "
    "eigenvalues nearest it); set that instead."
)
_PROBLEM_TYPE = (
    "is set by gfem: the seed operator is a non-Hermitian shift-inverted operator."
)
_CONVERGENCE = (
    "is set by gfem: eigen_tolerance is relative to each shift-inverted "
    "eigenvalue, as with ARPACK; set SolverOptions.eigen_tolerance instead."
)
_EXTRACTION = (
    "needs the operator's transpose or a target, which gfem's seed operator "
    "does not give SLEPc."
)
#: Options gfem sets itself, and what to set instead (the message's tail).
_RESERVED: dict[str, str] = {
    "ksp_rtol": (
        "is set from SolverOptions.tolerance (adjoint_tolerance for adjoint "
        "solves); set that instead."
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
        "has no effect: gfem's seed applies T(σ)⁻¹ through linear_solver, not "  # noqa: RUF001
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
                    f"shift-inverted seed operator, got {value!r}."
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
    """Every numerical knob of a solve, kept apart from the model and problem.

    Frozen and keyword-only. The defaults converge on the reference problems;
    `preset` gives coarser or finer settings in one word.

    Attributes:
        order: Polynomial order of the Nédélec (H(curl)) and H1 elements.
        linear_solver: ``"auto"`` (each problem type's default: BDDC + GMRES
            for driven, BDDC + conjugate gradients for electrostatic, direct
            for eigenmode), ``"bddc_gmres"`` (BDDC with the Krylov method
            that fits the operator: conjugate gradients for electrostatic),
            ``"direct"`` (one LDL^T factorization reused for every
            right-hand side: MKL PARDISO with METIS ordering when the ``mkl``
            extra is installed, NGSolve's sparse Cholesky otherwise) or
            ``"petsc"`` (PETSc: an exact MUMPS LDL^T factorization unless
            ``petsc_options`` configures another PETSc solver, from the
            PETSc stack the "Solve with PETSc and SLEPc" recipe builds; never
            chosen by ``"auto"``). A solver this Python cannot use raises
            `gfem.SolverUnavailableError` when the solve starts.
        tolerance: With BDDC, the relative preconditioned residual of the
            iterative solve: for BDDC + GMRES ``|P(b - Ax)| / |P b|`` as
            GMRES estimates it, for BDDC + conjugate gradients the P-norm of
            the residual over its first value. S and fields are accurate to
            about this size; the unpreconditioned residual ``|b - Ax| / |b|``
            can be orders larger at low frequency and does not measure the
            error. With ``"petsc"`` and a Krylov ``ksp_type`` it is PETSc's
            ``ksp_rtol`` on the unpreconditioned residual ``|b - Ax| / |b|``
            (unless ``petsc_options`` sets ``ksp_norm_type``), so the same
            value is a different norm per backend; PETSc's default exact
            factorization ignores it.
        device: Where the direct solver runs: ``"cpu"``, ``"cuda"`` or
            ``"cuda:N"`` (the N-th GPU). A GPU needs
            ``linear_solver="direct"`` (NVIDIA cuDSS, from the ``cuda12`` or
            ``cuda13`` extra). Its availability is checked when solving, so
            options built on a laptop can run on a GPU machine; a GPU that
            cannot be used raises `gfem.DeviceUnavailableError`, never a
            silent CPU fallback. See `gfem.cuda_available`.
        threads: CPU threads, or ``None`` for the machine's default.
            ``linear_solver="petsc"`` does not read it: MUMPS's BLAS follows
            ``OPENBLAS_NUM_THREADS``, which frequency workers set to their
            share.
        adaptive_tolerance: Error target of the adaptive frequency sweep;
            at least 10x ``tolerance``.
        adaptive_max_samples: Most frequencies the adaptive sweep solves.
        adaptive_memory: Consecutive samples under ``adaptive_tolerance``
            that end the adaptive sweep.
        port_mode_tolerance: Residual accepted for a reduced port mode before
            an exact port mode solve replaces it.
        retain_memory: Bytes of solver factorizations kept for gradients;
            ``None`` is half of the physical memory, 0 always recomputes.
        adjoint_tolerance: Relative residual of adjoint solves, in the norm
            ``tolerance`` names for each backend; ``None`` is ``tolerance``.
        eigen_tolerance: Backward error accepted for an eigenmode (the
            residual over ``|K| + |omega| |C| + |omega|^2 |M|``), shared by
            the seed iteration (ARPACK's, or SLEPc's ``eps_tol`` with
            ``eigen_solver="slepc"``) and the refinement; distinct from
            ``tolerance``, the residual of each linear solve. The norms are
            set by the highest curl-curl modes, so a resonance's frequency
            and Q can be much less accurate than this value: on a port-loaded
            CPW resonator, gsim's default 1e-6 leaves f off by 1e-5 and Q by
            0.5 %, while 1e-10 brings both errors far below that.
        cluster_rtol: Relative distance under which eigenfrequencies form a
            degenerate cluster.
        max_refine_iterations: Most iterations of the eigenmode refinement.
        frequency_workers: Processes a uniform `gfem.Driven` sweep spreads
            its frequencies over, in contiguous blocks, each with
            ``threads // frequency_workers`` threads (at least 1; with
            ``threads=None``, of the logical CPUs the process may run
            on). It applies to uniform driven sweeps only: adaptive
            sweeps, eigenmode and electrostatic solves ignore it. Memory
            grows about ``frequency_workers`` times, one solve per worker.
            S agrees with one worker's to the solver precision (each
            block's iterative solves start cold). Not available with wave
            ports on more than one frequency (`NotImplementedError`) or a
            GPU device. The workers are
            spawned processes, so a script that sets it must guard its
            entry point with ``if __name__ == "__main__":``.
        eigen_solver: What finds an `gfem.Eigenmode` problem's modes near
            ``target`` from shift-invert solves: ``"auto"`` or ``"arpack"``
            (SciPy ARPACK), or ``"slepc"`` (SLEPc's Krylov-Schur, through
            slepc4py, from the PETSc stack the "Solve with PETSc and SLEPc"
            recipe builds; never chosen by ``"auto"``). gfem refines and
            differentiates the modes the same way after either. Other
            problem types ignore it. SLEPc this Python cannot use raises
            `gfem.SolverUnavailableError` when the solve starts.
        petsc_options: PETSc and SLEPc option names, without the leading
            ``-``, and their values, read by ``linear_solver="petsc"`` and
            ``eigen_solver="slepc"``; a value of ``None`` is a flag, a
            ``bool`` is ``true`` / ``false``. They override gfem's defaults
            (``pc_type cholesky``, ...), under each solver's own options
            prefix: never write one. Names starting ``eps_``, ``bv_``,
            ``ds_``, ``rg_`` or ``fn_`` need ``eigen_solver="slepc"``, every
            other name needs ``linear_solver="petsc"``, so an option never
            goes to a solver that is not selected. Options gfem sets from
            its own fields (``ksp_rtol`` from ``tolerance``, ``eps_tol``
            from ``eigen_tolerance``, ...) raise `ValueError`. An option the
            PETSc linear solver does not read (a typo, or an option of
            another configuration) raises `ValueError` when the solve
            starts; one SLEPc does not read raises it after SLEPc's seed
            solve. Stored as ``(name, value)`` pairs sorted by name.
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
    retain_memory: int | None = None
    adjoint_tolerance: float | None = None
    eigen_tolerance: float = 1e-10
    cluster_rtol: float = 1e-6
    max_refine_iterations: int = 50
    frequency_workers: int = 1
    eigen_solver: EigenSolver = "auto"
    petsc_options: Mapping[str, PetscValue] | PetscOptions = ()

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        _int(self.order, "order", 1)
        if self.linear_solver not in _LINEAR_SOLVERS:
            raise ValueError(
                f"SolverOptions.linear_solver must be one of "
                f"{', '.join(repr(s) for s in _LINEAR_SOLVERS)}, got "
                f"{self.linear_solver!r}."
            )
        if self.eigen_solver not in _EIGEN_SOLVERS:
            raise ValueError(
                f"SolverOptions.eigen_solver must be one of "
                f"{', '.join(repr(s) for s in _EIGEN_SOLVERS)}, got "
                f"{self.eigen_solver!r}."
            )
        set_(self, "petsc_options", _petsc_options(self))
        for name in (
            "tolerance",
            "adaptive_tolerance",
            "port_mode_tolerance",
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
        if self.device.startswith("cuda") and self.linear_solver != "direct":
            raise ValueError(
                f"device={self.device!r} needs linear_solver='direct': only the "
                "direct solver runs on a GPU."
            )
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
