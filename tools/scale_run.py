"""A run on a gsim case through the real `gfem.solve`, its record, its agreement gates.

Run it under the launcher, once per node count and solver, or with no launcher for the
one-process reference (ADR 0073). Layout and limits belong to the job script; only the
agreement gates can fail, and no time is ever a gate:

    srun --mpi=pmix --cpus-per-task=4 python scale_run.py run CASE OUT/NAME \
        --solver bddc_gmres --ghz 50
    python scale_run.py run CASE OUT/ref --solver bddc_gmres --ghz 50 --threads 112
    python scale_run.py check OUT/NAME OUT/ref --gate bddc_gmres
    python scale_run.py table OUT/*.json

``run`` writes ``NAME.json``, the record: versions, nodes, ranks and threads, DOFs per
rank, phases, Krylov vector share, reduction wait, memory, true residual. ``NAME.npz``
holds S, iterations and gradients. S, iterations and gradients are compared bit for bit
across ranks, and a difference ends the run with exit code 1 once the record is written.

``--design`` replaces the plain solve by two steps of `jax.value_and_grad` on the
permittivity of the highest-permittivity volume and the fill of a resistive sheet on the
largest interface between two volumes, since a sheet needs the domain on both sides.
``--spaces-only`` stops once the space is built, for the DOFs per rank.

``check`` compares a run with its reference at ADR 0070's gates: 1e-11 on ``direct``,
1e-9 and the exact iteration count on ``bddc_gmres``, S and gradient entries relative to
the largest of the reference's. The result goes into the run's record. ``--observe``
reports the same difference with no gate, for a run on another mesh.

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
import os
import resource
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

#: ADR 0070's gates, relative to the largest entry of the reference.
GATES = {"direct": 1e-11, "bddc_gmres": 1e-9}
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

    def __init__(self) -> None:
        """No phase has run yet."""
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    @contextlib.contextmanager
    def timed(self, name: str):
        """Count the block as phase ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1

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

    set_up = coarse.CoarseMumps.set_up

    def timed_coarse(self):
        with phases.timed("coarse_setup"):
            set_up(self)
        fact = {"dofs": int(self._numbering.total), "reduction": int(self._factor)}
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
        for port in sorted(boundaries["LumpedPort"], key=lambda p: p["Index"]):
            elements = port["Elements"]
            surfaces = tuple(self.name[2, e["Attributes"][0]] for e in elements)
            directions = tuple(_direction(e["Direction"]) for e in elements)
            self.ports.append(
                gfem.LumpedPort(
                    f"p{port['Index']}", surfaces, directions, impedance=port["R"]
                )
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
                gfem.enclosed_by(s[0], s[0].replace("_xy", "_z"))
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


def _environment(comm: Any) -> dict[str, Any]:
    versions = {}
    for package in PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    hosts = (
        [socket.gethostname()] if comm is None else comm.allgather(socket.gethostname())
    )
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
        "commit": os.environ.get("SCALE_COMMIT"),
        "versions": versions,
        "mpi_library": library,
        "modules": os.environ.get("LOADEDMODULES"),
        "machine": os.environ.get("SCALE_MACHINE", "MareNostrum 5 GPP"),
        "cpu": model,
        "job": os.environ.get("SLURM_JOB_ID"),
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

    phases, facts = Phases(), Facts()
    _instrument(phases, facts, comm)
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
        args.name.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
        if arrays:
            np.savez(args.name.with_suffix(".npz"), **arrays)
    say(f"record written; ranks bitwise: {bitwise} {differing or ''}")
    return 0 if bitwise else 1


def _printable(s: np.ndarray | None) -> list | None:
    if s is None:
        return None
    return [[repr(complex(v)) for v in row] for row in np.asarray(s).reshape(-1, 2)]


def _difference(a: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(a - reference).max() / np.abs(reference).max())


def check(args: argparse.Namespace) -> int:
    """Compare a run with a reference; write the result into the run's record."""
    run_path, reference_path = args.name, args.reference
    here = np.load(run_path.with_suffix(".npz"))
    there = np.load(reference_path.with_suffix(".npz"))
    record = json.loads(run_path.with_suffix(".json").read_text())
    gate = GATES[args.gate] if args.gate else None
    names = [k for k in here.files if k == "s" or k.startswith(("grad_", "value_"))]
    differences = {k: _difference(here[k], there[k]) for k in names if k in there.files}
    passed = bitwise = bool(record["ranks_bitwise"])
    same_iterations = None
    if gate is not None:
        passed = passed and all(d < gate for d in differences.values())
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
        "differences": differences,
        "same_iterations": same_iterations,
        "ranks_bitwise": bitwise,
        "passed": passed if gate is not None else None,
    }
    record.setdefault("checks", []).append(result)
    run_path.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"run": record["name"], **result}))
    return 0 if passed else 1


def table(args: argparse.Namespace) -> int:
    """A Markdown table of records, in the order given."""
    print(
        "| Nodes | Ranks x threads | DOF per rank | Solver | Frequencies | "
        "Iterations | `gfem.solve`, s | Largest node, GiB | Job |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for path in args.records:
        r = json.loads(path.read_text())
        dofs = r["dofs"].get("owned_sum", 0)
        per_rank = f"{dofs / r['ranks'] / 1e3:.0f} k" if dofs else "all"
        solve = r["phases"].get("solve", {}).get("max_s", float("nan"))
        its = sorted({i for row in r["iterations"].get("iterations", []) for i in row})
        print(
            f"| {r['node_count']} | {r['ranks']} x {r['threads_read']} | "
            f"{per_rank} | "
            f"`{r['solver']}` | {len(r['frequencies_ghz'])} | "
            f"{'-' if not its else ', '.join(map(str, its))} | {solve:.0f} | "
            f"{r['memory']['peak_rss_largest_node_gib']:.0f} | {r['job']} |"
        )
    return 0


def main() -> int:
    """Parse the command and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    one = commands.add_parser("run")
    one.add_argument("case", type=Path)
    one.add_argument("name", type=Path, help="the record's path, with no suffix")
    one.add_argument("--solver", choices=tuple(GATES), required=True)
    one.add_argument("--ghz", default="50")
    one.add_argument("--tol", type=float, default=1e-8)
    one.add_argument("--order", type=int, default=2)
    one.add_argument("--threads", type=int, default=None)
    one.add_argument("--design", action="store_true")
    one.add_argument("--seed", type=int, default=0)
    one.add_argument("--spaces-only", action="store_true")
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
    rows.set_defaults(call=table)

    args = parser.parse_args()
    return args.call(args)


if __name__ == "__main__":
    sys.exit(main())
