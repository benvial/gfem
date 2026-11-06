"""A run on a gsim case through the real `gfem.solve`, its record, its agreement gates.

Run it under the launcher, once per node count and solver, or with no launcher for the
one-process reference (ADR 0073). Layout and limits belong to the job script; only the
agreement gates can fail, and no time is ever a gate:

    srun --mpi=pmix --cpus-per-task=4 python scale_run.py run CASE OUT/NAME \
        --solver bddc_gmres --ghz 50
    python scale_run.py run CASE OUT/ref --solver bddc_gmres --ghz 50 --threads 112
    python scale_run.py check OUT/NAME OUT/ref --gate bddc_gmres
    python scale_run.py table OUT/*.json --base 4
    python scale_run.py close OUT/NAME --log LOG --state OUT_OF_MEMORY --rc 137

``run`` writes ``NAME.json``, the record: versions, nodes, ranks and threads, DOFs per
rank, phases, Krylov vector share, reduction wait, memory, true residual. ``NAME.npz``
holds S, iterations and gradients. S, iterations and gradients are compared bit for bit
across ranks, and a difference ends the run with exit code 1 once the record is written.

``--design`` replaces the plain solve by two steps of `jax.value_and_grad` on the
permittivity of the highest-permittivity volume and the fill of a resistive sheet on the
largest interface between two volumes, since a sheet needs the domain on both sides.
``--spaces-only`` stops once the space is built, for the DOFs per rank.

``check`` compares a run with its reference at ADR 0070's gates: 1e-11 on ``direct``,
1e-9 and the exact iteration count on ``bddc_gmres``, S entries relative to the largest
of the reference's. Gradient entries read `GRADIENT_GATE` on a real mesh. The result
goes into the run's record. ``--observe`` reports the same difference with no gate, for
a run on another mesh.

``close`` is the job script's, after every step, and by hand after a job that died
with its script. A run that left no record gets a stub, ``failed: true``, with Slurm's
state, the phases its ``--trace`` left open and the figures MUMPS printed. A finished
run keeps its record, with those figures added, since the coarse's cannot be read
behind ``PCTELESCOPE``. ``check`` reports a stub as not passed, and ``table`` shows it.
``close --state STARTED``, before a run, first moves any record and sidecar of that
name aside to ``NAME.superseded-N.json`` and ``.npz``, so a killed rerun is not
reported by an earlier run's record; ``table`` and ``check`` ignore superseded files.

``table`` gives one row per rung: median and range over the repeats, DOF per rank and
per node, and with ``--base NODES`` an efficiency against that named rung, strong within
one mesh or ``--weak`` across meshes. It reads times only to print them.

A phase is read through a wrapper around gfem's own functions. These are private names:
this script is pinned to the commit in its record.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import re
import socket
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

#: ADR 0070's gates, relative to the largest entry of the reference.
GATES = {"direct": 1e-11, "bddc_gmres": 1e-9}
#: Gradient entries on a real mesh, either solver, relative to the reference's largest.
#: ADR 0070's 1e-11 for ``direct`` holds on the launched tier's tiny meshes. On
#: r10M_void gradients differ beyond it with no ranks at all, against one-process
#: PARDISO at 112 threads: PARDISO at 56 threads by 6.5e-11 in eps_r (job 48473913),
#: one-process MUMPS by 9.9e-11 (48473915). MUMPS on 112 ranks differs by 1.4e-10
#: (48473421), while S agrees to 7e-13.
GRADIENT_GATE = 1e-9
#: Phases too frequent for the trace: several per GMRES iteration.
UNTRACED = {"gmres_vector", "gmres_reduction"}
#: The packages a record names.
PACKAGES = ("gfem", "ngsolve", "petsc", "petsc4py", "mpi4py", "numpy", "scipy", "jax")
#: PETSc's log events of a Cholesky factorization: analysis, numeric.
FACTOR_EVENTS = {"analysis": "MatCholFctrSym", "factor": "MatCholFctrNum"}
#: Weights of the design objective on S with one excited port, whose S21 asks for an
#: adjoint solve on the other port.
WEIGHTS = np.array([[0.7], [1.1]])
#: The design's sheet: S/m and m, a film of 1 kilo-ohm per square at fill 1.
SHEET_SIGMA, SHEET_THICKNESS = 1e3, 1e-6


class Phases:
    """Seconds and calls per phase on this rank; the record keeps the slowest rank."""

    def __init__(self, *, trace: bool = False) -> None:
        """No phase has run yet; ``trace`` prints each phase's start and end."""
        self.trace = trace
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    @contextlib.contextmanager
    def timed(self, name: str):
        """Count the block as phase ``name``."""
        start = time.perf_counter()
        trace = self.trace and name not in UNTRACED
        if trace:
            print(f"{name} starts at {_peak_gib():.1f} GiB peak", flush=True)
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1
            if trace:
                print(f"{name} ends at {_peak_gib():.1f} GiB peak", flush=True)

    def wrap(self, owner: Any, attribute: str, name: str) -> None:
        """Count every call of ``owner.attribute`` as phase ``name``."""
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def timed_call(*args, **kwargs):
            with self.timed(name):
                return original(*args, **kwargs)

        setattr(owner, attribute, timed_call)

    def gathered(self, comm: Any) -> dict[str, dict[str, float]]:
        """Each phase's seconds, slowest rank and rank 0, and its calls on rank 0."""
        here = {k: (self.seconds[k], self.calls[k]) for k in sorted(self.seconds)}
        every = [here] if comm is None else comm.allgather(here)
        names = sorted({k for rank in every for k in rank})
        return {
            k: {
                "max_s": max(rank.get(k, (0.0, 0))[0] for rank in every),
                "rank0_s": every[0].get(k, (0.0, 0))[0],
                "calls": every[0].get(k, (0.0, 0))[1],
            }
            for k in names
        }


class Facts:
    """What the wrappers read on the way: DOF counts, factors, residuals, one apply."""

    def __init__(self) -> None:
        """Nothing is read yet."""
        self.dofs: dict[str, Any] = {}
        self.factors: list[dict[str, Any]] = []
        self.coarse: list[dict[str, Any]] = []
        self.residuals: list[dict[str, Any]] = []
        self.apply: dict[str, float] = {}
        self.layout: Any = None


def _record_dofs(comm: Any, facts: Facts, layout: Any, space: Any) -> None:
    owned = comm.allgather(int(np.count_nonzero(layout.owners)))
    local = comm.allgather(int(space.ndof))
    workers = np.array(owned[1:] or owned)
    facts.dofs = {
        "owned_sum": int(sum(owned)),
        "owned_rank0": int(owned[0]),
        "owned_min": int(workers.min()),
        "owned_mean": float(workers.mean()),
        "owned_max": int(workers.max()),
        "owned_zero_ranks": int(np.count_nonzero(workers == 0)),
        "local_min": int(min(local[1:] or local)),
        "local_max": int(max(local)),
    }
    facts.layout = layout


def _instrument(phases: Phases, facts: Facts, comm: Any) -> None:
    """Put the wrappers on gfem's functions: see the module's docstring."""
    _instrument_phases(phases, facts, comm)
    _instrument_solve(phases, facts)
    _instrument_gmres(phases, facts)


def _instrument_phases(phases: Phases, facts: Facts, comm: Any) -> None:
    """The plain phases, the DOFs of a rank's layout, and PETSc's under ranks."""
    import ngsolve as ngs

    import gfem
    from gfem._engine import driven, mesh, system_operator
    from gfem._linalg import ResolvedLinearSolver, bddc, layout

    phases.wrap(gfem.Mesh, "from_gmsh", "mesh_read")
    phases.wrap(mesh, "_distributed", "partition")
    phases.wrap(driven, "driven_operator", "spaces")
    phases.wrap(driven.DrivenRule, "_ports", "ports")
    phases.wrap(driven.DrivenRule, "forward", "forward")
    phases.wrap(driven.DrivenRule, "backward", "backward")
    phases.wrap(system_operator, "refactorize_or_prepare", "prepare_or_refactor")
    phases.wrap(system_operator.SystemOperator, "solve_adjoint", "adjoint_solve")
    phases.wrap(ResolvedLinearSolver, "prepare", "prepare")
    phases.wrap(bddc, "_set_up", "bddc_setup")
    with contextlib.suppress(TypeError, AttributeError):  # an NGSolve type may refuse
        phases.wrap(ngs.BilinearForm, "Assemble", "assemble")

    if comm is not None:
        init = layout.RankLayout.__init__

        def layout_init(self, space, comm_) -> None:
            init(self, space, comm_)
            _record_dofs(comm_, facts, self, space)

        layout.RankLayout.__init__ = layout_init

        from gfem import _petsc

        if _petsc.has_mpi():
            _instrument_petsc(phases, facts)


def _instrument_solve(phases: Phases, facts: Facts) -> None:
    """Each sweep solve, with a true residual per column."""
    from gfem._engine import system_operator

    sweep_solve = system_operator.SystemSweep.solve

    def solve(self, omega, rhs, *, x0=None):
        with phases.timed("sweep_solve"):
            x, iterations = sweep_solve(self, omega, rhs, x0=x0)
        handle = self._last[1]
        with phases.timed("true_residual"):
            residual = handle.apply(x) - rhs
            norm = facts.layout.norm if facts.layout else np.linalg.norm
            facts.residuals.append(
                {
                    "omega": float(omega),
                    "relative": [
                        float(norm(residual[:, j]) / norm(rhs[:, j]))
                        for j in range(rhs.shape[1])
                    ],
                }
            )
        return x, iterations

    system_operator.SystemSweep.solve = solve


def _instrument_gmres(phases: Phases, facts: Facts) -> None:
    """GMRES: the vector work, the reductions inside it, one apply and one matvec."""
    from gfem import _mpi
    from gfem._linalg import gmres

    depth = {"vector": 0, "gmres": 0}

    def vector_work(owner, attribute):
        original = getattr(owner, attribute)

        @functools.wraps(original)
        def call(*args, **kwargs):
            depth["vector"] += 1
            try:
                if depth["vector"] > 1:
                    return original(*args, **kwargs)
                with phases.timed("gmres_vector"):
                    return original(*args, **kwargs)
            finally:
                depth["vector"] -= 1

        setattr(owner, attribute, call)

    for attribute in ("append", "norm", "scale", "orthogonalize", "add"):
        vector_work(gmres._Basis, attribute)  # noqa: SLF001
    vector_work(gmres._OwnedBasis, "norm")  # noqa: SLF001

    reduce_bcast = _mpi.reduce_bcast

    def timed_reduction(comm_, values):
        if not depth["gmres"]:
            return reduce_bcast(comm_, values)
        with phases.timed("gmres_reduction"):
            return reduce_bcast(comm_, values)

    _mpi.reduce_bcast = timed_reduction

    iteration = gmres.Gmres._gmres  # noqa: SLF001

    def timed_gmres(self, rhs, sol, cols, kernel):
        depth["gmres"] += 1
        try:
            with phases.timed("gmres"):
                out = iteration(self, rhs, sol, cols, kernel)
        finally:
            depth["gmres"] -= 1
        if not facts.apply:
            scratch = rhs.CreateVector()
            start = time.perf_counter()
            scratch.data = self.pre * rhs
            middle = time.perf_counter()
            scratch.data = self.mat * sol
            facts.apply = {
                "apply_s": middle - start,
                "matvec_s": time.perf_counter() - middle,
            }
        return out

    gmres.Gmres._gmres = timed_gmres  # noqa: SLF001


def _instrument_petsc(phases: Phases, facts: Facts) -> None:
    """The factorizations' sizes, and their set-ups, on the MPI PETSc stack."""
    from gfem import _petsc
    from gfem._linalg import coarse, petsc

    PETSc = _petsc.require_petsc()
    with contextlib.suppress(Exception):
        PETSc.Log.begin()

    set_up_now = petsc.RankPetscHandle._set_up_now  # noqa: SLF001

    def timed_set_up(self):
        with phases.timed("mumps_setup"):
            set_up_now(self)
        factor = self._ksp.getPC().getFactorMatrix()
        facts.factors.append(
            {
                "infog29": int(factor.getMumpsInfog(29)),
                "infog22_mb": int(factor.getMumpsInfog(22)),
                "info22_mb": int(factor.getMumpsInfo(22)),
            }
        )

    petsc.RankPetscHandle._set_up_now = timed_set_up  # noqa: SLF001

    phases.wrap(petsc.RankPetscHandle, "_retry_with_more_workspace", "mumps_retry")
    phases.wrap(coarse.CoarseMumps, "_options", "coarse_attempt")
    set_up = coarse.CoarseMumps.set_up

    def timed_coarse(self):
        with phases.timed("coarse_setup"):
            set_up(self)
        fact = {
            "dofs": int(self._numbering.total),
            "reduction": int(self._factor),
            "ranks": math.ceil(self._comm.size / self._factor),
        }
        with contextlib.suppress(Exception):  # the inner KSP is absent on some ranks
            pc = self._ksp.getPC()
            matrix = (
                pc.getFactorMatrix()
                if self._factor == 1
                else pc.getTelescopeSubKSP().getPC().getFactorMatrix()
            )
            fact["infog29"] = int(matrix.getMumpsInfog(29))
            fact["infog22_mb"] = int(matrix.getMumpsInfog(22))
        facts.coarse.append(fact)

    coarse.CoarseMumps.set_up = timed_coarse
    phases.wrap(coarse.CoarseMumps, "Mult", "coarse_apply")


def _print_mumps(comm: Any, say: Callable[[str], None]) -> None:
    """MUMPS's own statistics, ``ICNTL(4)=2``: the whole system's and the coarse's.

    Without ranks it only says so: one process's ``direct`` is PARDISO or sparse
    Cholesky and its BDDC has no MUMPS coarse, so neither reaches MUMPS.
    """
    if comm is None:
        say("--mumps-print does nothing without ranks")
        return
    from gfem._linalg import coarse, petsc

    defaults = petsc.RankPetscHandle._defaults  # noqa: SLF001

    def with_statistics(self):
        return (*defaults(self), ("mat_mumps_icntl_4", 2))

    petsc.RankPetscHandle._defaults = with_statistics  # noqa: SLF001

    options = coarse.CoarseMumps._options  # noqa: SLF001

    def with_printing(self, workspace):
        prefix = "telescope_" if self._factor > 1 else ""
        return (*options(self, workspace), (f"{prefix}mat_mumps_icntl_4", 2))

    coarse.CoarseMumps._options = with_printing  # noqa: SLF001


def _factor_events() -> dict[str, float]:
    """PETSc's own seconds in MUMPS's analysis and numeric phases on this rank."""
    out: dict[str, float] = {}
    with contextlib.suppress(Exception):
        from gfem import _petsc

        PETSc = _petsc.require_petsc()
        for name, event in FACTOR_EVENTS.items():
            info = PETSc.Log.Event(event).getPerfInfo()
            out[name] = float(info["time"])
    return out


#: A phase line of ``--trace``.
TRACE_LINE = re.compile(r"^(\w+) (starts|ends) at [\d.]+ GiB peak$", re.MULTILINE)
#: MUMPS's line opening a call, ``ICNTL(4)=2``: the job and the system's size.
MUMPS_CALL = re.compile(r"Entering [SDCZ]MUMPS \S+ .*JOB, N =\s*(\d+)\s+(\d+)")
#: MUMPS's lines worth keeping, in the words it prints them.
MUMPS_FIGURES = {
    "entries_estimated": r"\(20\) Number of entries in factors \(estim\.\)\s*=\s*(\S+)",
    "front_estimated": r"\(5\) Maximum frontal size\s+\(estimated\)\s*=\s*(\S+)",
    "flops_estimated": r"RINFOG\(1\) Operations during elimination \(estim\)\s*=\s*"
    r"(\S+)",
    "max_ic_mb": r"\(INFOG\(16\)\):\s*(\S+)",
    "total_ic_mb": r"\(INFOG\(17\)\):\s*(\S+)",
    "total_used_mb": r"\(INFOG\(22\)\):\s*(\S+)",
    "entries": r"INFOG \(29\) Number of entries in factors\s*=\s*(\S+)",
    "front": r"INFOG \(11\) Maximum front size\s*=\s*(\S+)",
}


def _open_phases(log: str) -> list[str]:
    """The phases a trace started and never ended, outermost first."""
    opened: list[str] = []
    for name, event in TRACE_LINE.findall(log):
        if event == "starts":
            opened.append(name)
        elif name in opened:
            while opened.pop() != name:
                pass
    return opened


def _mumps_number(text: str) -> float | int:
    """A figure as MUMPS prints it: ``7316285143`` or ``2.592D+13``."""
    try:
        return int(text)
    except ValueError:
        return float(text.replace("D", "E"))


def _mumps_printed(log: str) -> list[dict[str, Any]]:
    """MUMPS's printed figures, one entry per analysis, in the order they ran."""
    entries: list[dict[str, Any]] = []
    figures = {key: re.compile(pattern) for key, pattern in MUMPS_FIGURES.items()}
    for line in log.splitlines():
        call = MUMPS_CALL.search(line)
        if call:
            if call[1] == "1":
                entries.append({"n": int(call[2])})
        elif entries:
            for key, pattern in figures.items():
                found = pattern.search(line)
                if found:
                    entries[-1][key] = _mumps_number(found[1])
    return entries


def _direction(text: str) -> str:
    text = text.strip().lower()
    return text if text[0] in "+-" else "+" + text


def _is_voided(case: Path) -> bool:
    build = case / "build.json"
    if not build.exists():
        return False
    return "excluded_entities" in json.loads(build.read_text())


class Case:
    """A gsim case as `tests/reference/test_driven.py` builds it."""

    def __init__(self, directory: Path) -> None:
        """Read ``config.json`` and the mesh it names."""
        import gfem

        self.gfem = gfem
        self.config = json.loads((directory / "config.json").read_text())
        self.unit = self.config["Model"]["L0"]
        self.mesh = gfem.Mesh.from_gmsh(
            directory / self.config["Model"]["Mesh"], length_unit="um"
        )
        self.voided = _is_voided(directory)
        self.name = {(r.dim, r.tag): r.name for r in self.mesh.regions}
        boundaries = self.config["Boundaries"]
        self.shells = [
            (self.name[2, tag], c["Conductivity"], c["Thickness"] * self.unit)
            for c in boundaries["Conductivity"]
            for tag in c["Attributes"]
        ]
        self.materials = {
            self.name[3, tag]: (m["Permittivity"], m.get("LossTan", 0.0))
            for m in self.config["Domains"]["Materials"]
            for tag in m["Attributes"]
        }
        order = boundaries["Absorbing"]["Order"]
        self.absorbing = [
            gfem.Absorbing(self.name[2, tag], order=order)
            for tag in boundaries["Absorbing"]["Attributes"]
        ]
        self.ports = []
        for port in sorted(boundaries.get("LumpedPort", []), key=lambda p: p["Index"]):
            elements = port["Elements"]
            surfaces = tuple(self.name[2, e["Attributes"][0]] for e in elements)
            directions = tuple(_direction(e["Direction"]) for e in elements)
            self.ports.append(
                gfem.LumpedPort(
                    f"p{port['Index']}", surfaces, directions, impedance=port["R"]
                )
            )

        # gsim's WavePortPEC: perfect conductors in the port mode problem only
        pec_edges = tuple(
            self.name[2, t]
            for t in boundaries.get("WavePortPEC", {}).get("Attributes", [])
        )
        self.ports += [
            gfem.WavePort(
                f"p{port['Index']}",
                self.name[2, port["Attributes"][0]],
                mode=port["Mode"],
                offset=port["Offset"] * self.unit,
                pec_edges=pec_edges,
            )
            for port in sorted(boundaries.get("WavePort", []), key=lambda p: p["Index"])
        ]
        # with wave ports the metal reaches the line's end faces, which absorb
        self.ends = (
            tuple(a.surface for a in self.absorbing)
            if boundaries.get("WavePort")
            else ()
        )

    @property
    def design_region(self) -> str:
        """The volume of highest permittivity."""
        return max(self.materials, key=lambda k: self.materials[k][0])

    @property
    def design_surface(self) -> str:
        """The largest surface between two different volumes."""

        def is_interface(name: str) -> bool:
            sides = self.mesh.adjacency(name)
            return all(a and b and a != b for a, b in sides)

        names = [
            r.name for r in self.mesh.regions if r.dim == 2 and is_interface(r.name)
        ]
        return max(names, key=lambda n: (len(self.mesh.elements(n)), n))

    def model(self, eps=None, fill=None):
        """The case's model, with a design's permittivity and sheet if given."""
        gfem = self.gfem
        materials = {
            name: gfem.Material(eps_r=eps_r, tan_delta=tan_delta)
            for name, (eps_r, tan_delta) in self.materials.items()
        }
        if eps is not None:
            region = self.design_region
            materials[region] = gfem.MaterialField(
                eps_r=eps, tan_delta=float(self.materials[region][1]), region=region
            )
        shells = [gfem.Conductor(*shell) for shell in self.shells]
        if fill is not None:
            shells.append(
                gfem.Conductor(
                    self.design_surface,
                    SHEET_SIGMA,
                    SHEET_THICKNESS,
                    fill=gfem.SheetField(fill=fill),
                )
            )
        excluded = ()
        if not self.voided:
            excluded = tuple(
                gfem.enclosed_by(s[0], s[0].replace("_xy", "_z"), *self.ends)
                for s in self.shells
                if s[0].endswith("_xy")
            )
        return gfem.Model(
            self.mesh,
            materials=materials,
            boundaries=[*shells, *self.absorbing],
            ports=self.ports,
            excluded=excluded,
        )


def _options(args: argparse.Namespace):
    import gfem

    return gfem.SolverOptions(
        linear_solver=args.solver,
        tolerance=args.tol,
        threads=args.threads,
        order=args.order,
    )


def _capture_iterations(store: list) -> None:
    """Keep each forward's iterations: `gfem.solve` does not return them."""
    from gfem._engine.driven import DrivenRule

    forward = DrivenRule.forward

    def spy(self, model, *, keep):
        outputs, entry = forward(self, model, keep=keep)
        store.append(np.asarray(outputs["iterations"], dtype=np.int64))
        return outputs, entry

    DrivenRule.forward = spy


def _plain(case: Case, args, frequencies, *, arrays, phases, iterations) -> dict:
    import gfem

    with phases.timed("solve"):
        result = gfem.solve(
            case.model(), gfem.Driven(list(frequencies)), options=_options(args)
        )
    arrays["s"] = np.asarray(result.s.values)
    arrays["iterations"] = iterations[-1]
    return {}


def _design(case: Case, args, frequencies, *, arrays, phases, iterations, comm) -> dict:
    """Two steps of `value_and_grad`, each design from the last gradient."""
    import jax
    import jax.numpy as jnp

    import gfem

    rng = np.random.default_rng(args.seed)
    base = case.materials[case.design_region][0]
    n_eps = len(case.mesh.elements(case.design_region))
    n_fill = len(case.mesh.elements(case.design_surface))
    design = {
        "eps": jnp.asarray(base * rng.uniform(0.99, 1.01, n_eps)),
        "fill": jnp.asarray(rng.uniform(0.2, 0.8, n_fill)),
    }
    driven = gfem.Driven(list(frequencies), excite=[case.ports[0].name])
    options = _options(args)
    weights = jnp.asarray(WEIGHTS)

    def objective(p):
        model = case.model(eps=p["eps"], fill=p["fill"])
        s = gfem.solve(model, driven, options=options).s.values
        return jnp.sum(weights * jnp.abs(s) ** 2) + jnp.sum(
            jnp.real((0.3 - 0.7j) * s[:, 1, 0])
        )

    steps = []
    for step in range(2):
        before = dict(phases.seconds)
        with phases.timed("value_and_grad"):
            value, gradient = jax.value_and_grad(objective)(design)
        arrays[f"value_{step}"] = np.asarray(value)
        arrays[f"grad_eps_{step}"] = np.asarray(gradient["eps"])
        arrays[f"grad_fill_{step}"] = np.asarray(gradient["fill"])
        arrays[f"iterations_{step}"] = iterations[-1]
        spent = {
            k: phases.seconds[k] - before.get(k, 0.0)
            for k in phases.seconds
            if phases.seconds[k] != before.get(k, 0.0)
        }
        steps.append(spent if comm is None or comm.rank == 0 else {})
        design = {
            "eps": jnp.maximum(
                design["eps"] - 0.01 * gradient["eps"] / jnp.abs(gradient["eps"]).max(),
                1.0,
            ),
            "fill": jnp.clip(
                design["fill"]
                - 0.01 * gradient["fill"] / jnp.abs(gradient["fill"]).max(),
                0.0,
                1.0,
            ),
        }
    return {
        "steps_rank0": steps,
        "design": {
            "volume": case.design_region,
            "volume_elements": n_eps,
            "sheet": case.design_surface,
            "sheet_faces": n_fill,
        },
    }


def _spaces(case: Case, args, frequencies, comm) -> None:
    import gfem
    from gfem._engine.driven import DrivenRule
    from gfem._engine.runtime import task_manager

    model = case.model()
    rule = DrivenRule(model, gfem.Driven(list(frequencies)), _options(args), comm=comm)
    with task_manager(args.threads, comm):
        rule.operator(model)


def _peak_gib() -> float:
    import resource  # Unix only; the tool runs on Linux, its tests import it anywhere

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _memory(comm: Any) -> dict[str, Any]:
    here = (socket.gethostname(), _peak_gib())
    every = [here] if comm is None else comm.allgather(here)
    nodes: dict[str, float] = defaultdict(float)
    for host, peak in every:
        nodes[host] += peak
    return {
        "peak_rss_rank0_gib": every[0][1],
        "peak_rss_rank_max_gib": max(peak for _, peak in every),
        "peak_rss_largest_node_gib": max(nodes.values()),
        "peak_rss_sum_gib": sum(nodes.values()),
    }


def _digest(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
        for name, value in sorted(arrays.items())
    }


def _mem_total_kb() -> int:
    """What the node reports as ``MemTotal``, which tells highmem from standard."""
    with contextlib.suppress(OSError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    return 0


def _job_facts() -> dict[str, Any]:
    """What the job script knows and a run does not: see ``scale_job.sbatch``."""
    return {
        "commit": os.environ.get("SCALE_COMMIT"),
        "job": os.environ.get("SLURM_JOB_ID"),
        "qos": os.environ.get("SCALE_QOS") or None,
        "constraint": os.environ.get("SCALE_CONSTRAINT") or None,
    }


def _environment(comm: Any) -> dict[str, Any]:
    versions = {}
    for package in PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    here = (socket.gethostname(), _mem_total_kb())
    every = [here] if comm is None else comm.allgather(here)
    hosts = [host for host, _ in every]
    library = None
    if comm is not None:
        from mpi4py import MPI

        library = MPI.Get_library_version().splitlines()[0]
    model = None
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {
        **_job_facts(),
        "mem_total_kb": sorted({kb for _, kb in every}),
        "versions": versions,
        "mpi_library": library,
        "modules": os.environ.get("LOADEDMODULES"),
        "machine": os.environ.get("SCALE_MACHINE", "MareNostrum 5 GPP"),
        "cpu": model,
        "nodes": sorted(set(hosts)),
        "node_count": len(set(hosts)),
        "ranks_per_node": max(hosts.count(h) for h in set(hosts)),
        "ranks": len(hosts),
    }


def _observed(args, phases, facts, comm) -> dict[str, Any]:
    """What the record reads across ranks once the solve is done: collective."""
    gathered = phases.gathered(comm)
    gmres = gathered.get("gmres", {}).get("max_s", 0.0)
    krylov = None
    if gmres:
        vector = gathered.get("gmres_vector", {}).get("max_s", 0.0)
        reduction = gathered.get("gmres_reduction", {}).get("max_s", 0.0)
        krylov = {
            "vector_share": vector / gmres,
            "reduction_wait_s": reduction,
            "reduction_share": reduction / gmres,
        }
    apply_time = dict(facts.apply)
    events = _factor_events() if args.solver == "direct" and comm is not None else {}
    if comm is not None:
        apply_time = max(comm.allgather(apply_time), key=lambda d: d.get("apply_s", 0))
        if events:
            everyone = comm.allgather(events)
            events = {k: max(e.get(k, 0.0) for e in everyone) for k in events}
    factors: dict[str, Any] = {"factors": facts.factors}
    if comm is not None and facts.factors:
        info = comm.allgather([f["info22_mb"] for f in facts.factors])
        factors["info22_mb_sum"] = [int(sum(r)) for r in zip(*info, strict=True)]
        factors["info22_mb_max"] = [int(max(r)) for r in zip(*info, strict=True)]
    return {
        "dofs": facts.dofs,
        "phases": gathered,
        "krylov": krylov,
        "one_apply": apply_time,
        "petsc_events": events,
        "factors": factors,
        "coarse": facts.coarse,
        "residuals": facts.residuals,
        "memory": _memory(comm),
    }


def run(args: argparse.Namespace) -> int:
    """One solve, design or space on this launch's ranks: see the module's docstring."""
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    started = time.perf_counter()

    import gfem  # noqa: F401
    from gfem import _mpi

    comm = _mpi.resolve_comm(None)
    if comm is not None:
        _mpi.install_abort_hook()
    rank = 0 if comm is None else comm.rank

    def say(text: str) -> None:
        if rank == 0:
            print(f"{time.perf_counter() - started:9.1f} s  {text}", flush=True)

    if args.dump_after and rank < 2:
        faulthandler.dump_traceback_later(args.dump_after, exit=False)

    phases, facts = Phases(trace=args.trace and rank == 0), Facts()
    _instrument(phases, facts, comm)
    if args.mumps_print:
        _print_mumps(comm, say)
    iterations: list[np.ndarray] = []
    if not args.spaces_only:
        _capture_iterations(iterations)
    environment = _environment(comm)
    say(f"{environment['ranks']} ranks on {environment['node_count']} nodes")

    frequencies = [float(g) * 1e9 for g in args.ghz.split(",")]
    case = Case(args.case)
    say(f"mesh: {len(case.mesh.tetrahedra)} tetrahedra, voided={case.voided}")

    arrays: dict[str, np.ndarray] = {}
    extra: dict[str, Any] = {}
    if args.spaces_only:
        with phases.timed("spaces_only"):
            _spaces(case, args, frequencies, comm)
    elif args.design:
        extra = _design(
            case,
            args,
            frequencies,
            arrays=arrays,
            phases=phases,
            iterations=iterations,
            comm=comm,
        )
    else:
        extra = _plain(
            case, args, frequencies, arrays=arrays, phases=phases, iterations=iterations
        )
    say("solved")

    digests = _digest(arrays)
    every = [digests] if comm is None else comm.allgather(digests)
    differing = sorted({k for d in every for k in d if d.get(k) != digests.get(k)})
    bitwise = not differing

    observed = _observed(args, phases, facts, comm)
    threads_read = args.threads if comm is None else _mpi.default_threads(comm)
    if rank == 0:
        record = {
            "name": args.name.name,
            "case": str(args.case),
            "tetrahedra": len(case.mesh.tetrahedra),
            "solver": args.solver,
            "options": {
                "tolerance": args.tol,
                "threads": args.threads,
                "order": args.order,
            },
            "frequencies_ghz": [f / 1e9 for f in frequencies],
            "repeat": args.repeat,
            "mode": "spaces"
            if args.spaces_only
            else "design"
            if args.design
            else "solve",
            **environment,
            "threads_read": threads_read,
            **observed,
            "iterations": {k: arrays[k].tolist() for k in arrays if "iterations" in k},
            "s": _printable(arrays.get("s")),
            "ranks_bitwise": bitwise,
            "ranks_differ_in": differing,
            "wall_s": time.perf_counter() - started,
            **extra,
        }
        args.name.parent.mkdir(parents=True, exist_ok=True)
        _file(args.name, ".json").write_text(json.dumps(record, indent=1) + "\n")
        if arrays:
            np.savez(_file(args.name, ".npz"), **arrays)
    say(f"record written; ranks bitwise: {bitwise} {differing or ''}")
    return 0 if bitwise else 1


def _file(name: Path, suffix: str) -> Path:
    """The record's file: a name may hold a dot (``x_r0.5_void``), so never replace."""
    return name.with_name(name.name + suffix)


#: A record a rerun moved aside (`close`), with no suffix or with ``.json``.
SUPERSEDED = re.compile(r"\.superseded-\d+(\.json)?$")


def _superseded(name: Path) -> bool:
    return SUPERSEDED.search(name.name) is not None


def _supersede(name: Path) -> None:
    """Move ``name``'s record and sidecar aside, to the first free ``superseded-N``."""
    suffixes = (".json", ".npz")
    if not any(_file(name, s).exists() for s in suffixes):
        return
    n = 1
    while any(_file(name, f".superseded-{n}{s}").exists() for s in suffixes):
        n += 1
    for suffix in suffixes:
        if _file(name, suffix).exists():
            _file(name, suffix).rename(_file(name, f".superseded-{n}{suffix}"))


def _printable(s: np.ndarray | None) -> list | None:
    if s is None:
        return None
    return [[repr(complex(v)) for v in row] for row in np.asarray(s).reshape(-1, 2)]


def _difference(a: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(a - reference).max() / np.abs(reference).max())


def _nothing_to_compare(run_path: Path, reference_path: Path) -> str | None:
    """Why a run cannot be compared, a killed run's or a reference's absence."""
    for label, name in (("run", run_path), ("reference", reference_path)):
        if _file(name, ".npz").exists():
            continue
        path = _file(name, ".json")
        record = json.loads(path.read_text()) if path.exists() else {}
        state = record.get("state", "no record")
        return f"the {label} {name} holds no solution: {state}"
    return None


def _not_compared(args: argparse.Namespace, reason: str) -> int:
    """A check with nothing to compare: not passed, in the record if there is one."""
    result = {"reference": str(args.reference), "passed": False, "reason": reason}
    path = _file(args.name, ".json")
    if path.exists():
        record = json.loads(path.read_text())
        record.setdefault("checks", []).append(result)
        path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"run": args.name.name, **result}))
    return 1


def check(args: argparse.Namespace) -> int:
    """Compare a run with a reference; write the result into the run's record."""
    run_path, reference_path = args.name, args.reference
    for label, name in (("run", run_path), ("reference", reference_path)):
        if _superseded(name):
            reason = f"the {label} {name} is superseded by a rerun"
            print(json.dumps({"run": run_path.name, "passed": False, "reason": reason}))
            return 1
    reason = _nothing_to_compare(run_path, reference_path)
    if reason:
        return _not_compared(args, reason)
    here = np.load(_file(run_path, ".npz"))
    there = np.load(_file(reference_path, ".npz"))
    record = json.loads(_file(run_path, ".json").read_text())
    gate = GATES[args.gate] if args.gate else None
    names = [k for k in here.files if k == "s" or k.startswith(("grad_", "value_"))]
    differences = {k: _difference(here[k], there[k]) for k in names if k in there.files}
    passed = bitwise = bool(record["ranks_bitwise"])
    same_iterations = None
    if gate is not None:
        passed = passed and all(
            d < (GRADIENT_GATE if k.startswith("grad_") else gate)
            for k, d in differences.items()
        )
        passed = passed and set(names) <= set(there.files)
        if args.gate == "bddc_gmres":
            # a direct reference has no iterations: only its S compares
            keys = [k for k in here.files if "iterations" in k and there[k].size > 0]
            if keys:
                same_iterations = all(np.array_equal(here[k], there[k]) for k in keys)
                passed = passed and same_iterations
    result = {
        "reference": str(reference_path),
        "gate": gate,
        "gradient_gate": GRADIENT_GATE if gate is not None else None,
        "differences": differences,
        "same_iterations": same_iterations,
        "ranks_bitwise": bitwise,
        "passed": passed if gate is not None else None,
    }
    record.setdefault("checks", []).append(result)
    _file(run_path, ".json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"run": record["name"], **result}))
    return 0 if passed else 1


#: Printed under every table: what the time column holds.
SOLVE_NOTE = (
    "`gfem.solve` is the slowest rank's whole call, with the instrumentation's true "
    "residual and one extra apply and matvec inside it."
)


def _solve_seconds(record: dict[str, Any]) -> float:
    """The solve's time; NaN for a record with none (``--spaces-only``)."""
    return record.get("phases", {}).get("solve", {}).get("max_s", float("nan"))


def _rungs(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The records by rung, in the order given: repeats of one mesh, solver, layout."""
    rungs: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (
            Path(r["case"]).name,
            r["solver"],
            r["node_count"],
            r["ranks"],
            r["threads_read"],
        )
        rungs[key].append(r)
    return list(rungs.values())


def _median(rung: list[dict[str, Any]]) -> float | None:
    seconds = [_solve_seconds(r) for r in rung if not r.get("failed")]
    return float(np.median(seconds)) if seconds else None


def _efficiency(
    rung: list[dict[str, Any]],
    rungs: list[list[dict[str, Any]]],
    base: int | None,
    *,
    weak: bool,
) -> str:
    """Against the rung on ``base`` nodes: this solver's, and this mesh's if strong."""
    seconds, first = _median(rung), rung[0]
    if base is None or seconds is None:
        return "-"
    for other in rungs:
        one = other[0]
        same = one["solver"] == first["solver"] and (
            weak or Path(one["case"]).name == Path(first["case"]).name
        )
        if same and one["node_count"] == base:
            reference = _median(other)
            if reference is None:
                return "-"
            nodes = 1 if weak else first["node_count"] / base
            return f"{reference / (seconds * nodes):.2f}"
    return "-"


def _rung_row(
    rung: list[dict[str, Any]],
    rungs: list[list[dict[str, Any]]],
    base: int | None,
    *,
    weak: bool,
) -> str:
    """One row of the table: failed repeats are counted, a failed rung says how."""
    first = rung[0]
    done = [r for r in rung if not r.get("failed")]
    failed = [r for r in rung if r.get("failed")]
    repeats = " + ".join(
        [
            *([str(len(done))] if done or not failed else []),
            *(f"1 {r['state']}" for r in failed),
        ]
    )
    dofs = max((r["dofs"].get("owned_sum", 0) for r in done), default=0)
    per_rank = f"{dofs / first['ranks'] / 1e3:.0f} k" if dofs else "-"
    per_node = f"{dofs / first['node_count'] / 1e6:.2f} M" if dofs else "-"
    its = sorted(
        {i for r in done for row in r["iterations"].get("iterations", []) for i in row}
    )
    if done:
        seconds = [_solve_seconds(r) for r in done]
        median = f"{_median(rung):.0f}"
        spread = f"{min(seconds):.0f}-{max(seconds):.0f}"
        memory = max(r["memory"]["peak_rss_largest_node_gib"] for r in done)
        largest = f"{memory:.0f}"
    else:
        last = failed[0]
        median = f"{last['state']} in {last.get('last_phase') or 'no phase'}"
        spread = largest = "-"
    jobs = ", ".join(dict.fromkeys(str(r.get("job")) for r in rung))
    cells = (
        Path(first["case"]).name,
        f"`{first['solver']}`",
        first["node_count"],
        f"{first['ranks']} x {first['threads_read']}",
        f"{dofs:,}" if dofs else "-",
        per_rank,
        per_node,
        repeats,
        len(first.get("frequencies_ghz", [])),
        ", ".join(map(str, its)) or "-",
        median,
        spread,
        _efficiency(rung, rungs, base, weak=weak),
        largest,
        jobs,
    )
    return "| " + " | ".join(map(str, cells)) + " |"


def table(args: argparse.Namespace) -> int:
    """A Markdown table of records, one row per rung, in the order given."""
    paths = [path for path in args.records if not _superseded(path)]
    records = [json.loads(path.read_text()) for path in paths]
    rungs = _rungs(records)
    titles = (
        "Mesh",
        "Solver",
        "Nodes",
        "Ranks x threads",
        "DOF",
        "DOF per rank",
        "DOF per node",
        "Repeats",
        "Frequencies",
        "Iterations",
        "`gfem.solve` median, s",
        "Range, s",
        "Efficiency",
        "Largest node, GiB",
        "Job",
    )
    print("| " + " | ".join(titles) + " |")
    print("|" + "---|" * len(titles))
    for rung in rungs:
        print(_rung_row(rung, rungs, args.base, weak=args.weak))
    print()
    print(SOLVE_NOTE)
    if args.base is not None:
        kind = "weak" if args.weak else "strong"
        print(f"Efficiency against {args.base} nodes ({kind}).")
    return 0


#: A stub closed by hand with no step line: what the table still needs.
STUB_DEFAULTS = {
    "state": "UNKNOWN",
    "case": "unknown",
    "solver": "unknown",
    "node_count": 0,
    "ranks": 0,
    "threads_read": 0,
}


def _stub_facts(args: argparse.Namespace) -> dict[str, Any]:
    """What a step's line and its job know about a run that wrote nothing itself."""
    nodes = os.environ.get("SLURM_JOB_NUM_NODES")
    node_count = int(nodes) if nodes else None
    facts = {
        **_job_facts(),
        "node_count": node_count,
        "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "case": str(args.case) if args.case else None,
        "solver": args.solver,
        "repeat": args.repeat,
        "threads_read": args.threads_per_rank,
        "ranks_per_node": args.ranks_per_node,
        "state": args.state,
        "rc": args.rc,
    }
    if args.ranks_per_node == 0:
        facts["ranks"] = 1  # python with no launcher, as its own record says
    elif node_count and args.ranks_per_node:
        facts["ranks"] = node_count * args.ranks_per_node
    if args.ghz:
        facts["frequencies_ghz"] = [float(g) for g in args.ghz.split(",")]
    return {key: value for key, value in facts.items() if value is not None}


def close(args: argparse.Namespace) -> int:
    """After a step: the printed figures on its record, or a stub if it has none."""
    if args.state == "STARTED":
        _supersede(args.name)
    path = _file(args.name, ".json")
    record = json.loads(path.read_text()) if path.exists() else {}
    log = args.log.read_text(errors="replace") if args.log and args.log.exists() else ""
    figures = {"mumps_printed": _mumps_printed(log)} if log else {}
    if record and not record.get("failed"):
        record |= figures
    else:
        opened = _open_phases(log)
        trace = {"open_phases": opened, "last_phase": opened[-1]} if opened else {}
        record = {
            **STUB_DEFAULTS,
            **record,
            "name": args.name.name,
            "mode": "failed",
            "failed": True,
            **_stub_facts(args),
            **trace,
            **figures,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return 0


def main() -> int:
    """Parse the command and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    one = commands.add_parser("run")
    one.add_argument("case", type=Path)
    one.add_argument("name", type=Path, help="the record's path, with no suffix")
    one.add_argument("--solver", choices=(*GATES, "petsc"), required=True)
    one.add_argument("--ghz", default="50")
    one.add_argument("--tol", type=float, default=1e-8)
    one.add_argument("--order", type=int, default=2)
    one.add_argument("--threads", type=int, default=None)
    one.add_argument("--design", action="store_true")
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--repeat", type=int, default=1, help="this run's repeat index")
    one.add_argument("--spaces-only", action="store_true")
    one.add_argument("--trace", action="store_true", help="print phases on rank 0")
    one.add_argument(
        "--mumps-print",
        action="store_true",
        help="MUMPS statistics, of the whole system on direct and of BDDC's coarse",
    )
    one.add_argument(
        "--dump-after",
        type=float,
        default=0.0,
        help="print rank 0's and 1's stacks after this many seconds",
    )
    one.set_defaults(call=run)

    against = commands.add_parser("check")
    against.add_argument("name", type=Path)
    against.add_argument("reference", type=Path)
    group = against.add_mutually_exclusive_group(required=True)
    group.add_argument("--gate", choices=tuple(GATES))
    group.add_argument("--observe", action="store_true")
    against.set_defaults(call=check)

    rows = commands.add_parser("table")
    rows.add_argument("records", type=Path, nargs="+")
    rows.add_argument(
        "--base", type=int, default=None, help="efficiency against this node count"
    )
    rows.add_argument(
        "--weak", action="store_true", help="the base's time over this rung's"
    )
    rows.set_defaults(call=table)

    closing = commands.add_parser("close")
    closing.add_argument("name", type=Path)
    closing.add_argument("--log", type=Path, default=None)
    closing.add_argument("--state", default=None, help="Slurm's state of the step")
    closing.add_argument("--rc", type=int, default=None)
    closing.add_argument("--case", type=Path, default=None)
    closing.add_argument("--solver", default=None)
    closing.add_argument("--ghz", default=None)
    closing.add_argument("--ranks-per-node", type=int, default=None)
    closing.add_argument("--threads-per-rank", type=int, default=None)
    closing.add_argument("--repeat", type=int, default=None)
    closing.set_defaults(call=close)

    args = parser.parse_args()
    return args.call(args)


if __name__ == "__main__":
    sys.exit(main())
