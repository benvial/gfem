"""Solver options shared by every problem type."""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Literal

__all__ = ["SolverOptions"]

LinearSolver = Literal["auto", "bddc_gmres", "direct"]
_LINEAR_SOLVERS = ("auto", "bddc_gmres", "direct")
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
            that fits the operator: conjugate gradients for electrostatic)
            or ``"direct"`` (sparse Cholesky, one factorization reused for
            every right-hand side).
        tolerance: Relative residual of iterative solves.
        device: Where the direct solver runs: ``"cpu"``, ``"cuda"`` or
            ``"cuda:N"`` (the N-th GPU). A GPU needs
            ``linear_solver="direct"`` (NVIDIA cuDSS, from the ``cuda12`` or
            ``cuda13`` extra). Its availability is checked when solving, so
            options built on a laptop can run on a GPU machine; a GPU that
            cannot be used raises `gfem.DeviceUnavailableError`, never a
            silent CPU fallback. See `gfem.cuda_available`.
        threads: CPU threads, or ``None`` for the machine's default.
        adaptive_tolerance: Error target of the adaptive frequency sweep;
            at least 10x ``tolerance``.
        adaptive_max_samples: Most frequencies the adaptive sweep solves.
        adaptive_memory: Consecutive samples under ``adaptive_tolerance``
            that end the adaptive sweep.
        port_mode_tolerance: Residual accepted for a reduced port mode before
            an exact port mode solve replaces it.
        retain_memory: Bytes of solver factorizations kept for gradients;
            ``None`` is half of the physical memory, 0 always recomputes.
        adjoint_tolerance: Relative residual of adjoint solves; ``None`` is
            ``tolerance``.
        eigen_tolerance: Backward error accepted for an eigenmode, as
            Palace defines it (``Eigenmode.Tol``: the residual over
            ``|K| + |omega| |C| + |omega|^2 |M|``), shared by the Arnoldi
            iteration and the refinement; distinct from ``tolerance``, the
            residual of each linear solve. The norms are set by the highest
            curl-curl modes, so a resonance's frequency and Q can be much
            less accurate than this value: on a port-loaded CPW resonator,
            Palace's (and gsim's) default 1e-6 leaves f off by 1e-5 and Q
            by 0.5 %, while 1e-10 matches Palace run at 1e-10 far below
            both.
        cluster_rtol: Relative distance under which eigenfrequencies form a
            degenerate cluster.
        max_refine_iterations: Most iterations of the eigenmode refinement.
    """

    order: int = 2
    linear_solver: LinearSolver = "auto"
    tolerance: float = 1e-6
    device: str = "cpu"
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

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        _int(self.order, "order", 1)
        if self.linear_solver not in _LINEAR_SOLVERS:
            raise ValueError(
                f"SolverOptions.linear_solver must be one of "
                f"{', '.join(repr(s) for s in _LINEAR_SOLVERS)}, got "
                f"{self.linear_solver!r}."
            )
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
