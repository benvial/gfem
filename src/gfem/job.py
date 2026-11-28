"""Serialized runs: `Job` (``job.json``) and `load_result` (``result.json``).

A job bundle is ``job.json``, the mesh file it references (by path and
sha256) and, when the job has arrays, a ``.npz`` sidecar next to it. The
sidecar stores every array of the job (material fields, moved mesh nodes, a
refined mesh, which has no file, ...), keyed by where it sits in the job, e.g.
``.model.materials['oxide'].eps_r``. A result bundle is a directory with
``result.json`` and ``result.npz``. A uniform sweep that stops at an unsolved
frequency leaves a partial bundle there instead, ``partial.json`` and
``partial.npz``: the same layout, holding the partial result, never under the
name ``result.json``. Both documents carry ``schema_version``
and ``gfem_version``: gfem reads older schema versions and refuses newer
ones, and it refuses by name any entry it does not know (one a newer gfem
wrote, or a misspelt one).

Run a job bundle with ``python -m gfem run job.json --out results/`` (or
`Job.run`), and read the result back with `load_result`.

Across ranks rank 0 reads and writes the files, and a result bundle holds its fields
in one-process numbering, so a bundle from any rank count is an ordinary one that a
laptop reads.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from gfem import _codec, _mpi
from gfem._background import Background
from gfem._boundaries import (
    PEC,
    PMC,
    Absorbing,
    Conductor,
    Convection,
    FixedTemperature,
    FloquetPort,
    HeatFlux,
    Impedance,
    LumpedPort,
    Periodic,
    WavePort,
)
from gfem._checks import is_traced
from gfem._incident import IncidentPart
from gfem._line_section import LineSection
from gfem._materials import (
    Blend,
    Debye,
    DjordjevicSarkar,
    Drude,
    Lorentz,
    Material,
    MaterialField,
    PermittivityLaw,
    PixelGeometry,
)
from gfem._optical_shift import _Sampling
from gfem._pml import PML, Layers, Side, Stretch
from gfem._port_section import PortSection, SectionBend
from gfem._pytree import register_fields
from gfem._raman_spectrum import RamanSpectrum
from gfem._sheet import SheetField
from gfem._sources import (
    ElectricDipole,
    GaussianBeam,
    IncidentField,
    MagneticDipole,
    PlaneWave,
    qualified,
)
from gfem.errors import LinearSolverConvergenceError
from gfem.measurements import InterfaceParticipation
from gfem.mesh import (
    _MIDNODE_ARRAYS,
    _ROLE_ARRAYS,
    _TOPOLOGY_ARRAYS,
    Mesh,
    Region,
    _array_names,
    _is_model_mesh,
    _mesh_sha256,
    _topology_arrays,
    _topology_from_arrays,
)
from gfem.model import EnclosedBy, Model
from gfem.options import SolverOptions
from gfem.problems import (
    Bend,
    DampingFrame,
    DCConduction,
    Driven,
    Eigenmode,
    Electrostatic,
    MechanicalEigenmode,
    Piezoelectric,
    PortMode,
    SteadyThermal,
    TransientThermal,
)
from gfem.results import (
    DCConductionResult,
    DiffractionOrderResult,
    DrivenResult,
    EigenmodeResult,
    ElectrostaticResult,
    Field,
    MechanicalEigenmodeResult,
    NetworkParameters,
    PiezoelectricResult,
    PortModeField,
    PortModeResult,
    ReducedFields,
    SteadyThermalResult,
    SweepInfo,
    TransientThermalResult,
)
from gfem.solve import _solve, _solve_cache, solve
from gfem.thermal import HeatSource, _FieldTrace
from gfem.thermo_optic import _RouteSampler, _RouteTrace, _ShiftTrace

__all__ = ["Job", "load_result"]

_JOB_FORMAT = "gfem.job"
_RESULT_FORMAT = "gfem.result"
RESULT_JSON = "result.json"
RESULT_NPZ = "result.npz"
PARTIAL_JSON = "partial.json"
PARTIAL_NPZ = "partial.npz"
#: A job's JSON among its arrays on the way across ranks; leaf paths start with ".".
_DOCUMENT = "$document"


@dataclasses.dataclass(frozen=True, eq=False)
class Job:
    """A self-contained run: a model, a problem and solver options.

    The unit gsim and cloud runners exchange. `to_json` writes it as a job
    bundle; ``python -m gfem run job.json --out DIR`` or `run` solves it.
    A job is a JAX
    `pytree <https://docs.jax.dev/en/latest/pytrees.html>`__
    whose arrays are the model's and the problem's.

    Attributes:
        model: The device.
        problem: What to solve, e.g. `gfem.Electrostatic`.
        options: The numerics; ``None`` is ``SolverOptions()``.

    Raises:
        TypeError: An argument has the wrong type.
    """

    model: Model
    problem: Any
    options: SolverOptions = None  # ty: ignore[invalid-assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.model, Model):
            raise TypeError(f"Job needs a gfem.Model, got {type(self.model).__name__}.")
        kind = type(self.problem)
        if not any(
            isinstance(getattr(kind, a, None), str) for a in ("_rule", "_rule_2d")
        ):
            raise TypeError(
                "Job needs a problem type such as gfem.Electrostatic(...), got "
                f"{type(self.problem).__name__}."
            )
        if self.options is None:
            object.__setattr__(self, "options", SolverOptions())
        elif not isinstance(self.options, SolverOptions):
            raise TypeError(
                "Job options must be a gfem.SolverOptions, got "
                f"{type(self.options).__name__}."
            )

    def run(
        self,
        out: str | Path | None = None,
        *,
        device: str | None = None,
        comm: Any = None,
        cache: str | os.PathLike | None = None,
    ) -> Any:
        """Solve the job: ``gfem.solve(model, problem, options=options)``.

        Args:
            out: If given, a directory to write the result bundle to
                (``result.json`` and ``result.npz``, read by `load_result`).
                Across ranks this is collective and rank 0 writes. A partial
                bundle left there by an earlier run is removed.
            device: Overrides ``options.device`` (e.g. a job written for a
                GPU run on a CPU machine). The bundle records the device used.
            comm: The communicator, as for `gfem.solve`; not part of the job,
                which does not say where it runs.
            cache: A directory for the solve cache, as for `gfem.solve`: the
                result is read from it when this job was solved and stored
                there before, and stored there otherwise. Independent of
                ``out``: the bundle is written either way. With a cache the
                bundle's ``run`` entry has a ``cache`` item: ``status``
                (``"hit"``, ``"miss"``, or ``"bypass"`` when the cache was
                not used), ``key`` (the solve key, or None), and on a hit
                ``solved``, the ``device``, ``ranks`` and ``wall_time_s`` of
                the run that solved it; ``wall_time_s`` of the ``run`` entry
                is this call's. ``None`` (the default) solves without a cache
                and adds no ``cache`` item.

        Returns:
            The problem type's result.

        Raises:
            OSError: Rank 0 cannot write the bundle; on every rank.
            gfem.LinearSolverConvergenceError: A linear solve did not
                converge. When a uniform sweep stopped with a partial result
                and ``out`` is given, the partial result is written there
                first as a partial bundle, ``partial.json`` and
                ``partial.npz`` (never ``result.json``, and a ``result.json``
                of an earlier run is left alone), and the error carries a
                note saying where it is. Read it with
                ``gfem.load_result(out / "partial.json")``; its ``run`` entry
                lists the unsolved frequencies.
        """
        options = self.options
        if device is not None:
            try:
                options = dataclasses.replace(options, device=device)
            except (TypeError, ValueError) as err:
                # one argv on every rank, so every rank raises it together
                raise _mpi.agreed(err) from None
        use = _solve_cache(cache)
        resolved = _mpi.resolve_comm(comm)
        start = time.perf_counter()

        def run() -> dict[str, Any]:
            entry = {
                "device": options.device,
                "device_override": device,
                "options": options,
                "ranks": 1 if resolved is None else resolved.size,
                "wall_time_s": time.perf_counter() - start,
            }
            if use is not None:
                entry["cache"] = use.report()
            return entry

        try:
            if use is None:
                result = solve(self.model, self.problem, options=options, comm=comm)
            else:
                result = _solve(
                    self.model, self.problem, options=options, comm=comm, cache=use
                )[0]
        except LinearSolverConvergenceError as err:
            if out is not None and err.result is not None:
                _keep_partial(Path(out), err, err.result, run(), resolved)
            raise
        if out is not None:
            entry = run()
            with _mpi.section(resolved):
                _write_result(Path(out), result, entry, resolved)
        return result

    def to_json(self, path: str | Path) -> Path:
        """Write the job bundle's ``job.json`` (and its ``.npz`` sidecar).

        The mesh is referenced by its file path and sha256; the path is
        stored relative to the JSON file's directory when the mesh lies
        inside it, so a directory holding the job and its mesh can be moved
        or uploaded as one bundle. A mesh with no file (a refined mesh) is
        stored whole in the sidecar instead. The sidecar, ``<stem>.npz`` next to the
        JSON file, holds every array of the job (material fields, moved mesh
        nodes), keyed by where it sits in the job, e.g.
        ``.model.materials['oxide'].eps_r``; it is written only when there
        are arrays.

        Not collective: across ranks a script calls it on rank 0.

        Args:
            path: The JSON file to write; its directory is created.

        Returns:
            The JSON file's path.

        Raises:
            TypeError: The job holds values computed inside `jax.grad`,
                `jax.jit` or `jax.vmap`; write it from fixed values.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoder = _codec.Encoder(_codec.Context(path.parent.resolve()))
        body = encoder.encode(self)
        sidecar = path.with_suffix(".npz")
        doc = _codec.header(_JOB_FORMAT) | {
            "arrays": sidecar.name if encoder.arrays else None,
            "job": body,
        }
        if encoder.arrays:
            _save_npz(sidecar, encoder.arrays)
        path.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
        return path

    @classmethod
    def from_json(cls, path: str | Path, *, comm: Any = None) -> Job:
        """Read a job bundle written by `to_json`.

        The mesh is read again and its sha256 checked. Across ranks this is
        collective: rank 0 reads the files, and every rank gets the same job or
        the same error.

        Args:
            path: The JSON file.
            comm: The communicator, as for `gfem.solve`. Ranks reading job files
                of their own pass ``MPI.COMM_SELF``.

        Raises:
            FileNotFoundError: The JSON file, its sidecar or its mesh is
                missing.
            ValueError: Not a job file, a newer schema version, a type or an
                entry this gfem does not know, or a mesh file that changed.
            RuntimeError: Ranks asked for different paths.
        """
        path = Path(path)
        resolved = _mpi.resolve_comm(comm)
        with _mpi.section(resolved):
            _mpi.check_call(resolved, "Job.from_json", {"the path": str(path)})
            doc, arrays = _read_job(path, resolved)
            try:
                job = _decode(doc, arrays, "job", path, comm)
                if not isinstance(job, cls):
                    raise ValueError(f"{path} is not a gfem job file (no job in it).")  # noqa: TRY004, TRY301
            except (OSError, TypeError, ValueError, KeyError) as err:
                # all ranks decode one document, and the mesh's errors were agreed
                _mpi.agreed(err)
                raise
        return job


# The problem may hold arrays one day; the options never do.
register_fields(Job, ("model", "problem"))


def load_result(path: str | Path, *, comm: Any = None) -> Any:
    """Read a result bundle written by ``python -m gfem run`` or `Job.run`.

    A bundle from any rank count reads on one process. Under a launcher a bundle
    with no fields holds no mesh and is read with no communication, on any rank, so
    a read that fails is one rank's failure: it ends the job, and the
    errors below raise only on one process.

    Args:
        path: The result directory, or its ``result.json``. A partial bundle
            is read by its file, ``DIR/partial.json``: it returns the partial
            result, over the solved frequencies only.
        comm: The communicator, as for `gfem.solve`. ``MPI.COMM_SELF`` under a
            launcher reads a bundle with fields as one-process fields.

    Returns:
        The result, e.g. a `gfem.ElectrostaticResult`, with the same values
        as the in-process solve.

    Raises:
        FileNotFoundError: No ``result.json`` (the message says when the
            directory holds a partial bundle instead), or its sidecar is
            missing.
        ValueError: Not a result file, a newer schema version, or an entry
            this gfem does not know.
        gfem.DistributedUnsupportedError: A bundle with fields, above one rank.
    """
    path = Path(path)
    if path.is_dir():
        partial = path / PARTIAL_JSON
        if not (path / RESULT_JSON).is_file() and partial.is_file():
            raise FileNotFoundError(
                f"No {RESULT_JSON} in {path}, only a partial result: the run "
                "stopped at an unsolved frequency. Read it with "
                f'gfem.load_result("{partial}").'
            )
        path /= RESULT_JSON
    resolved = _mpi.resolve_comm(comm)
    with _mpi.section(resolved):
        doc = _codec.check_header(
            _read_json(path), _RESULT_FORMAT, "gfem result file", path
        )
        if resolved is not None and _holds_a_mesh(doc.get("result")):
            _mpi.refuse_feature("load_result_fields")
        return _decode(doc, _read_arrays(doc, path), "result", path, comm)


def _holds_a_mesh(value: Any) -> bool:
    """Whether a JSON form holds a mesh: a field's, the only one a result has."""
    if isinstance(value, dict):
        return value.get("$type") == "Mesh" or any(map(_holds_a_mesh, value.values()))
    return isinstance(value, list) and any(map(_holds_a_mesh, value))


def _keep_partial(
    out: Path,
    err: LinearSolverConvergenceError,
    result: DrivenResult,
    run: dict[str, Any],
    comm: Any,
) -> None:
    """Write ``err``'s partial ``result`` as a partial bundle and note it on ``err``.

    A write that fails is noted too: the solver's error stays the one raised.
    """
    run = run | {"unsolved": err.unsolved, "error": f"{type(err).__name__}: {err}"}
    try:
        with _mpi.section(comm):
            _write_result(out, result, run, comm, partial=True)
    except (OSError, TypeError, ValueError) as failure:
        err.add_note(
            f"The partial result could not be written to {out}: "
            f"{type(failure).__name__}: {failure}"
        )
        return
    solved = len(result.s.frequency)
    path = out / PARTIAL_JSON
    note = (
        f"The partial result ({solved} of {solved + len(err.unsolved or ())} "
        f'frequencies) is in {path}: read it with gfem.load_result("{path}"). '
        'Its "run" entry lists the unsolved frequencies.'
    )
    if (out / RESULT_JSON).is_file():
        note += f" {out / RESULT_JSON} is the complete result of an earlier run."
    err.add_note(note)


def _write_result(
    out: Path, result: Any, run: dict[str, Any], comm: Any, *, partial: bool = False
) -> None:
    """Write the bundle on rank 0, each field gathered in one-process numbering.

    A partial result is written as a partial bundle, next to any result bundle;
    a complete one removes a partial bundle first.
    """
    json_name, npz_name = (
        (PARTIAL_JSON, PARTIAL_NPZ) if partial else (RESULT_JSON, RESULT_NPZ)
    )
    if comm is not None:
        if is_traced(result):
            raise _mpi.agreed(
                TypeError(
                    "Cannot write a result computed inside jax.grad, jax.jit or "
                    "jax.vmap. Write results from concrete values."
                )
            )
        _mpi.check_call(comm, "Job.run's bundle", {"the directory": str(out)})
        if isinstance(result, DrivenResult):
            result = dataclasses.replace(
                result,
                e_fields=tuple(field._gathered() for field in result.e_fields),  # noqa: SLF001
                h_fields=tuple(field._gathered() for field in result.h_fields),  # noqa: SLF001
            )
        elif isinstance(result, ElectrostaticResult):
            result = dataclasses.replace(
                result,
                potential={
                    name: field._gathered()  # noqa: SLF001
                    for name, field in result.potential.items()
                },
            )

    def write() -> dict[str, np.ndarray]:
        out.mkdir(parents=True, exist_ok=True)
        encoder = _codec.Encoder(_codec.Context(out.resolve()))
        body = encoder.encode(result)
        run_json = encoder.encode(run, ".run")
        doc = _codec.header(_RESULT_FORMAT) | {
            "arrays": npz_name if encoder.arrays else None,
            "run": run_json,
            "result": body,
        }
        if encoder.arrays:
            _save_npz(out / npz_name, encoder.arrays)
        if not partial:
            # a stale partial bundle would contradict the result
            (out / PARTIAL_JSON).unlink(missing_ok=True)
            (out / PARTIAL_NPZ).unlink(missing_ok=True)
        # result.json last: its presence means the bundle is complete.
        (out / json_name).write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
        return {}

    _mpi.on_root(comm, write, results=(OSError, TypeError, ValueError))


#: What rank 0's read of a job's files can raise: a truncated ``.npz`` is a
#: ``BadZipFile`` or an ``EOFError``, not an ``OSError``.
_READ_ERRORS = (OSError, ValueError, EOFError, zipfile.BadZipFile)


def _read_job(path: Path, comm: Any) -> tuple[dict, dict[str, np.ndarray]]:
    """A job's document and arrays, read by rank 0 alone."""

    def read() -> dict[str, np.ndarray]:
        doc = _codec.check_header(_read_json(path), _JOB_FORMAT, "gfem job file", path)
        arrays = _read_arrays(doc, path)
        text = np.frombuffer(json.dumps(doc).encode(), dtype=np.uint8)
        return {_DOCUMENT: text, **arrays}

    shared = _mpi.on_root(comm, read, results=_READ_ERRORS)
    return json.loads(shared.pop(_DOCUMENT).tobytes()), shared


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"{path} is not valid JSON: {err}.") from None


def _read_arrays(doc: dict, path: Path) -> dict[str, np.ndarray]:
    if not doc.get("arrays"):
        return {}
    sidecar = path.parent / doc["arrays"]
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"{path} keeps its arrays in {sidecar}, which is missing."
        )
    with np.load(sidecar, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _decode(
    doc: dict, arrays: dict[str, np.ndarray], key: str, path: Path, comm: Any
) -> Any:
    if key not in doc:
        raise ValueError(f"{path} has no {key!r} entry.")
    written_by = doc.get("gfem_version")
    decoder = _codec.Decoder(
        _codec.Context(path.parent.resolve(), comm),
        arrays,
        path=path,
        written_by=written_by if isinstance(written_by, str) else None,
    )
    return decoder.decode(doc[key])


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as f:
        np.savez(f, allow_pickle=False, **arrays)


# --- the registered value classes -----------------------------------------------
# A new model, problem, options or result class becomes serializable by one
# _codec.register call here (dataclasses need no more arguments).


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mesh_args(mesh: Mesh, context: _codec.Context) -> dict[str, Any]:
    if context.key:
        return _key_mesh_args(mesh)
    if context.model_mesh is not None:
        # a solve cache entry: never a file, so it outlives any edit to one
        if _is_model_mesh(mesh, context.model_mesh):
            return {"model_mesh": _mesh_sha256(mesh)}
        return _in_memory_mesh_args(mesh)
    source = mesh.source
    if source is None:
        return _in_memory_mesh_args(mesh)
    if not source.is_file():
        raise ValueError(
            f"Cannot write the mesh: its file {source} is gone. A job references "
            "the mesh file, so keep it next to the job."
        )
    path = (
        source.relative_to(context.base)
        if source.is_relative_to(context.base)
        else source
    )
    args: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": _sha256(source),
        "length_unit": mesh.length_unit,
    }
    return args | _section(mesh) | _moved_nodes(mesh)


def _key_mesh_args(mesh: Mesh) -> dict[str, Any]:
    """A mesh as the solve key sees it: the arrays in memory, never a file.

    A mesh read from a file, the same file elsewhere and the same topology
    rebuilt in memory all give these arguments.
    """
    return (
        {
            "length_unit": mesh.length_unit,
            "regions": [[r.name, r.tag, r.dim] for r in mesh.regions],
        }
        | _section(mesh)
        | {"topology_sha256": _mesh_sha256(mesh)}
        | _moved_nodes(mesh)
    )


def _model_mesh_from_args(digest: str, context: _codec.Context) -> Mesh:
    """The model's mesh, which a solve cache entry's fields refer to."""
    mesh = context.model_mesh
    if mesh is None:
        raise ValueError(
            "The fields refer to the model's mesh, but no model was given."
        )
    own = _mesh_sha256(mesh)
    if digest != own:
        raise ValueError(
            f"The fields were solved on a mesh of topology {digest}, but the "
            f"model's mesh has topology {own}."
        )
    return mesh


def _section(mesh: Mesh) -> dict[str, Any]:
    """A cross-section's thickness; no key for any other mesh."""
    thickness = mesh.section_thickness
    return {} if thickness is None else {"section_thickness": thickness}


def _moved_nodes(mesh: Mesh) -> dict[str, Any]:
    nodes = mesh.nodes
    if is_traced(nodes) or not np.array_equal(np.asarray(nodes), mesh.imported_nodes):
        return {"nodes": nodes}  # moved (shape optimization): keep them
    return {}


def _in_memory_mesh_args(mesh: Mesh) -> dict[str, Any]:
    """A mesh with no file (a refined one): its topology's arrays, in the sidecar."""
    topology = mesh._topology  # noqa: SLF001
    return (
        {
            "source": None,
            "length_unit": mesh.length_unit,
            "regions": mesh.regions,
        }
        | _section(mesh)
        | ({} if topology.dim == 3 else {"dim": topology.dim})
        | {name: np.asarray(a) for name, a in _topology_arrays(topology).items()}
        | _moved_nodes(mesh)
    )


def _in_memory_mesh_from_args(args: dict[str, Any]) -> Mesh:
    # a tetrahedral mesh writes no dimension and its arrays' historical names
    dim = int(args.get("dim", 3))
    arrays = {}
    names = (*_array_names(dim), *(n for n, _ in _MIDNODE_ARRAYS if n in args))
    for name in names:
        array = np.array(args[name])
        array.flags.writeable = False
        arrays[name] = array
    topology = _topology_from_arrays(
        arrays,
        length_unit=args["length_unit"],
        regions=tuple(args["regions"]),
        source=None,
        section_thickness=args.get("section_thickness"),
        dim=dim,
    )
    return _with_nodes(Mesh(jnp.asarray(topology.imported_nodes), topology), args)


def _with_nodes(mesh: Mesh, args: dict[str, Any]) -> Mesh:
    """The mesh with the job's moved nodes, if it has any."""
    if args.get("nodes") is None:
        return mesh
    nodes = jnp.asarray(args["nodes"])
    if nodes.shape != mesh.imported_nodes.shape:
        where = "the job's mesh" if mesh.source is None else f"the mesh {mesh.source}"
        raise ValueError(
            f"The job's moved nodes have shape {nodes.shape}, but {where} has "
            f"{mesh.imported_nodes.shape}."
        )
    return Mesh(nodes, mesh._topology)  # noqa: SLF001


def _mesh_from_args(args: dict[str, Any], context: _codec.Context) -> Mesh:
    if "model_mesh" in args:
        return _model_mesh_from_args(args["model_mesh"], context)
    if "path" not in args:
        return _in_memory_mesh_from_args(args)
    path = context.base / args["path"]

    def check() -> dict[str, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(
                f"The mesh file {path} is missing; a job bundle needs its mesh "
                f"({args['path']}) next to the job file."
            )
        digest = _sha256(path)
        if digest != args["sha256"]:
            raise ValueError(
                f"The mesh file {path} has changed since the job was written: its "
                f"sha256 is {digest}, the job expects {args['sha256']}. Restore the "
                "mesh, or build the model again and rewrite the job."
            )
        return {}

    _mpi.on_root(_mpi.resolve_comm(context.comm), check, results=(OSError, ValueError))
    if "section_thickness" in args:
        mesh = Mesh.from_gmsh_cross_section(
            path,
            length_unit=args["length_unit"],
            thickness=args["section_thickness"],
            comm=context.comm,
        )
    else:
        mesh = Mesh.from_gmsh(path, length_unit=args["length_unit"], comm=context.comm)
    return _with_nodes(mesh, args)


@dataclasses.dataclass(frozen=True)
class _Excluded:
    """A model's ``excluded=`` mapping as written (tag ``Excluded``).

    Only a model that excludes volumes from thermal, DC conduction or
    mechanical problems writes it; an electromagnetic-only model writes its
    list, as before. Being a tagged object, it stops an older gfem at the
    unknown type by name. An empty mechanical entry is left out, so a file
    without one is written as before; reading one, it defaults to empty.
    """

    em: tuple
    thermal: tuple
    dc_conduction: tuple
    mechanical: tuple = ()


def _excluded_args(model: Model) -> tuple | _Excluded:
    em, *others = (model.excluded_from(k) for k in _Excluded.__dataclass_fields__)
    if not any(others):
        return em
    return _Excluded(em, *others)


def _model_args(model: Model, _: _codec.Context) -> dict[str, Any]:
    return {
        "mesh": model.mesh,
        "materials": dict(model.materials),
        "boundaries": model.boundaries,
        "ports": model.ports,
        "excluded": _excluded_args(model),
        # written only when set: a model without a layer encodes as before
        **({"pml": model.pml} if model.pml is not None else {}),
        **({"background": model.background} if model.background is not None else {}),
    }


def _model_from_args(args: dict[str, Any], _: _codec.Context) -> Model:
    mesh = args.pop("mesh")
    return Model(mesh, **args)


def _terminal_args(
    problem: Electrostatic | DCConduction, _: _codec.Context
) -> dict[str, Any]:
    return {"terminals": dict(problem.terminals), "ground": problem.ground}


def _piezoelectric_args(problem: Piezoelectric, _: _codec.Context) -> dict[str, Any]:
    return {
        "frequency": problem.frequency,
        "terminals": dict(problem.terminals),
        "ground": problem.ground,
        "clamped": problem.clamped,
        "damping_frame": problem.damping_frame,
        "keep_fields": problem.keep_fields,
    }


def _mechanical_eigenmode_args(
    problem: MechanicalEigenmode, _: _codec.Context
) -> dict[str, Any]:
    return {
        "target": problem.target,
        "n_modes": problem.n_modes,
        "wavenumber": problem.wavenumber,
        "clamped": dict(problem.clamped),
        "ground": problem.ground,
        "damping_frame": problem.damping_frame,
        # written only when set: a problem without them encodes as before
        **({"pml": problem.pml} if problem.pml is not None else {}),
        **(
            {"target_quality_factor": problem.target_quality_factor}
            if problem.target_quality_factor is not None
            else {}
        ),
    }


#: A transient's arguments: its waveform is written as the pytree holds it,
#: start times and scales apart, so the scales' sidecar key is their leaf path.
_TRANSIENT_NAMES = (
    "heat",
    "duration",
    "steps",
    "initial_temperature",
    "traces",
    "keep_fields",
    "_switches",
    "_scales",
)


def _transient_args(problem: TransientThermal, _: _codec.Context) -> dict[str, Any]:
    return {name: getattr(problem, name) for name in _TRANSIENT_NAMES}


def _transient_from_args(args: dict[str, Any], _: _codec.Context) -> TransientThermal:
    switches, scales = args.pop("_switches"), args.pop("_scales")
    return TransientThermal(**args, waveform=tuple(zip(switches, scales, strict=True)))


def _field_args(field: Any, _: _codec.Context) -> dict[str, Any]:
    """A field's arguments: a communicator is not part of what it is.

    A layer's reference frequency, an incident column's incident part, the
    point dipoles and the scattered flag are written only when set, so that
    a field without them encodes as before.
    """
    out = {
        f.name: getattr(field, f.name)
        for f in dataclasses.fields(field)
        if f.name != "_comm"
    }
    for name in ("_reference_frequency", "_incident"):
        if out.get(name, 0) is None:
            del out[name]
    if not out.get("_point_sources", 0):
        out.pop("_point_sources", None)
    if not out.get("_scattered", 1):
        del out["_scattered"]
    return out


def _with_layer_args(result: Any, _: _codec.Context) -> dict[str, Any]:
    """A result's arguments, its layer and group velocity only when set.

    A result without them then encodes as before.
    """
    out = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
    for name in ("pml", "v_g"):
        if name in out and out[name] is None:
            del out[name]
    return out


def _port_mode_field_args(field: PortModeField, _: _codec.Context) -> dict[str, Any]:
    space = field._space  # noqa: SLF001
    # a line port's polarization; a planar port's field is written as before
    polarization = getattr(space, "polarization", None)
    return (
        {
            "section": space.section,
            "order": space.order,
            "basis": field._basis,  # noqa: SLF001
            "coords": field._coords,  # noqa: SLF001
        }
        | ({} if polarization is None else {"polarization": polarization})
        | (
            {} if field._bend is None else {"bend": field._bend}  # noqa: SLF001
        )
    )


def _port_mode_field_from_args(args: dict[str, Any], _: _codec.Context) -> Any:
    from gfem._engine.line_mode import mode_space

    space = mode_space(args["section"], args["order"], args.get("polarization"))
    return PortModeField(space, args["basis"], args["coords"], args.get("bend"))


def _port_section_from_args(args: dict[str, Any], _: _codec.Context) -> PortSection:
    # Sections saved by 0.1.0 hold one conductivity per triangle and one
    # admittance per absorbing edge: an isotropic medium's values.
    if "sigma" in args:
        sigma = np.asarray(args.pop("sigma"), float)
        args["sigma_tt"] = sigma[:, None, None] * np.eye(2)
        args["sigma_nn"] = sigma
    admittance = np.asarray(args["absorbing_admittance"], float)
    if admittance.ndim == 1:
        args["absorbing_admittance"] = np.repeat(admittance[:, None], 2, axis=1)
    return PortSection(**args)


def _options_args(options: SolverOptions, _: _codec.Context) -> dict[str, Any]:
    args = {f.name: getattr(options, f.name) for f in dataclasses.fields(options)}
    # a JSON object, as a user writes it; the constructor sorts it back
    args["petsc_options"] = dict(options.petsc_options)
    # the time-stepping options and the gauge are written only when set, so a
    # job that leaves them at their defaults reads in a gfem that predates them
    for name, default in (
        ("time_scheme", "tr_bdf2"),
        ("theta", 1.0),
        ("gauge", "auto"),
    ):
        if args[name] == default:
            del args[name]
    return args


def _unset_omitted(cls: Any, **defaults: Any) -> _codec.ToArgs:
    """Write ``cls``'s init fields, leaving out each of ``defaults`` at its default.

    It serves any optional field added after files were written (the fields
    of 2D meshes, a material's electro-optic tensor, a permittivity law's
    fitted range): left at its default, the field is absent from the file,
    which stays as an older gfem wrote it and reads there, and its solve key
    is unchanged.
    """
    names = [f.name for f in dataclasses.fields(cls) if f.init]

    def to_args(obj: Any, _: _codec.Context) -> dict[str, Any]:
        args = {n: getattr(obj, n) for n in names}
        for name, default in defaults.items():
            # a field sequence or mapping equals no default: only an empty one
            # is left out; an array is never compared with None
            value = args[name]
            if default is None:  # an array field is unset only when None
                unset = value is None
            elif isinstance(default, (tuple, dict)):
                unset = not len(value)
            else:
                unset = value == default
            if unset:
                del args[name]
        return args

    return to_args


def _job_from_args(args: dict[str, Any], _: _codec.Context) -> Job:
    return Job(args["model"], args["problem"], args.get("options", SolverOptions()))


#: What a job file's mesh may hold: a file reference, or a mesh with no file;
#: a solve cache entry's, a reference to the model's mesh.
_MESH_NAMES = (
    "model_mesh",
    "path",
    "sha256",
    "length_unit",
    "nodes",
    "source",
    "regions",
    "section_thickness",
    "dim",
    *dict.fromkeys((*_TOPOLOGY_ARRAYS, *_ROLE_ARRAYS)),
    *(name for name, _ in _MIDNODE_ARRAYS),
)

_codec.register(Mesh, to_args=_mesh_args, from_args=_mesh_from_args, names=_MESH_NAMES)
_codec.register(Region)
_codec.register(
    Model,
    to_args=_model_args,
    from_args=_model_from_args,
    names=("mesh", "materials", "boundaries", "ports", "excluded", "pml", "background"),
)
_codec.register(
    _Excluded,
    tag="Excluded",
    to_args=_unset_omitted(_Excluded, mechanical=()),
    from_args=lambda a, _: dict(a),
    names=("em", "thermal", "dc_conduction", "mechanical"),
)
_codec.register(
    EnclosedBy,
    from_args=lambda a, _: EnclosedBy(tuple(a["surfaces"])),
    names=("surfaces",),
)


def _renamed_density_from_args(cls: type) -> _codec.FromArgs:
    """Read an older file's ``density`` key as ``mass_density``."""

    def from_args(args: dict[str, Any], _: _codec.Context) -> Any:
        if "density" in args:
            old = args.pop("density")
            args.setdefault("mass_density", old)
        return cls(**args)

    return from_args


for _cls in (Material, MaterialField):
    _codec.register(
        _cls,
        # no electro-optic tensor, no solid and no Kerr property: written as
        # an older gfem wrote it, so the file and its solve key are unchanged
        to_args=_unset_omitted(
            _cls,
            nonlinear_index=None,
            two_photon_absorption=None,
            electro_optic_tensor=None,
            stiffness=None,
            piezoelectric_tensor=None,
            mechanical_quality_factor=None,
            photoelastic_tensor=None,
            second_order_nonlinear_tensor=None,
            # only a Material carries a Raman spectrum or a Kerr anisotropy
            **(
                {"raman_spectrum": None, "nonlinear_anisotropy": None}
                if _cls is Material
                else {}
            ),
        ),
        from_args=_renamed_density_from_args(_cls),
        # "density": the mass density as gfem 0.1 wrote it (a read alias)
        names=[f.name for f in dataclasses.fields(_cls) if f.init] + ["density"],
    )
# a Material's eps_r may hold a law: tags of their own, no Material key; a law
# without a fitted range is written as an older gfem wrote it, so the file
# and its solve key are unchanged
_codec.register(
    PermittivityLaw,
    to_args=_unset_omitted(PermittivityLaw, frequency_range=None),
)
for _cls in (
    Drude,
    Lorentz,
    Debye,
    DjordjevicSarkar,
    SheetField,
    RamanSpectrum,
    PEC,
    PMC,
    Absorbing,
    Conductor,
    Impedance,
    Periodic,
    PML,
    FixedTemperature,
    Convection,
    HeatFlux,
    # a joule source's power density goes to the sidecar by its leaf path
    HeatSource,
    SteadyThermal,
    LumpedPort,
    FloquetPort,
    DiffractionOrderResult,
    NetworkParameters,
    SweepInfo,
    Eigenmode,
    InterfaceParticipation,
    DCConductionResult,
    SteadyThermalResult,
    TransientThermalResult,
):
    _codec.register(_cls)
_codec.register(EigenmodeResult, to_args=_with_layer_args)
# a polarity only when set, so a port without one is written as before
_codec.register(
    WavePort, to_args=_unset_omitted(WavePort, polarity=None, allow_evanescent=False)
)
_codec.register(
    Field,
    to_args=_field_args,
    names=[f.name for f in dataclasses.fields(Field) if f.name != "_comm"],
)
_codec.register(
    ReducedFields,
    to_args=_field_args,
    names=[f.name for f in dataclasses.fields(ReducedFields)],
)
# a blend's fixed ends (MaterialField.blend_ends); a layer density's grid and
# other properties only when set, so a 3D blend is written as before
_codec.register(Blend, to_args=_unset_omitted(Blend, others=(), pixels=None))
_codec.register(PixelGeometry)
_codec.register(SolverOptions, to_args=_options_args)
_codec.register(
    Driven,
    to_args=_unset_omitted(Driven, polarization=None, sources=(), source_fields=True),
)
# a bend only when set, so a straight port mode is written as before
_codec.register(Bend)
_codec.register(
    PortMode, to_args=_unset_omitted(PortMode, polarization=None, bend=None)
)
_codec.register(
    PortModeResult,
    to_args=_unset_omitted(
        PortModeResult,
        h_field=None,
        pml=None,
        n_g=None,
        beta_2=None,
        beta_3=None,
        bend=None,
        allow_evanescent=False,
        polarity=None,
    ),
)
_codec.register(
    DrivenResult,
    to_args=_unset_omitted(
        DrivenResult,
        h_fields=(),
        polarization=None,
        sources={},
        port_waves=None,
        emitted_power={},
    ),
)
for _cls in (PlaneWave, GaussianBeam, ElectricDipole, MagneticDipole):
    _codec.register(_cls)
_codec.register(
    Background, to_args=_unset_omitted(Background, interfaces=(), ground=None)
)
_codec.register(
    IncidentPart, to_args=_unset_omitted(IncidentPart, layers=(), is_incident=False)
)


def _incident_field_args(field: IncidentField, _: _codec.Context) -> dict[str, Any]:
    raise _codec.UnencodableError(
        f"the IncidentField {field.name!r} (function {qualified(field.function)}) "
        "cannot be written: its function is code, not data, and a file without "
        "it would solve another problem"
    )


def _incident_field_from_args(_args: dict[str, Any], _: _codec.Context) -> Any:
    raise ValueError(
        "The file holds an IncidentField, which gfem never writes: its function "
        "cannot be read from a file."
    )


_codec.register(
    IncidentField,
    to_args=_incident_field_args,
    from_args=_incident_field_from_args,
    names=("name", "function"),
)
# a cross-section's slab thickness only when set, so a 3D result is written as
# before
_codec.register(
    ElectrostaticResult, to_args=_unset_omitted(ElectrostaticResult, _thickness=None)
)
# a bend only when set, so a straight section is written as before
_codec.register(
    PortSection,
    to_args=_unset_omitted(PortSection, bend=None),
    from_args=_port_section_from_args,
    # "sigma": the scalar conductivity of sections saved by 0.1.0
    names=[f.name for f in dataclasses.fields(PortSection) if f.init] + ["sigma"],
)
# a section's perfectly matched layer and bend, written with it (a result's
# mode field holds its section)
for _cls in (Side, Layers, Stretch, SectionBend):
    _codec.register(_cls)
_codec.register(Piezoelectric, to_args=_piezoelectric_args)
_codec.register(DampingFrame)
_codec.register(PiezoelectricResult)
_codec.register(MechanicalEigenmode, to_args=_mechanical_eigenmode_args)
_codec.register(MechanicalEigenmodeResult)
_codec.register(Electrostatic, to_args=_terminal_args)
_codec.register(DCConduction, to_args=_terminal_args)
_codec.register(
    TransientThermal,
    to_args=_transient_args,
    from_args=_transient_from_args,
    names=_TRANSIENT_NAMES,
)
# a Trace.point or Trace.mean
_codec.register(_FieldTrace, tag="Trace")
# ThermoOpticShift.trace(): its weights, and the optical mesh's tetrahedra
# that sample the rise
_codec.register(_ShiftTrace, tag="ShiftTrace")
_codec.register(_Sampling, tag="ShiftSampling")
# RoutePhase.trace(name): its weights, and the window placed along the route
_codec.register(_RouteTrace, tag="RouteTrace")
_ROUTE_SAMPLER_NAMES = tuple(
    f.name for f in dataclasses.fields(_RouteSampler) if f.name != "_cache"
)
_codec.register(
    _RouteSampler,
    tag="RouteSampler",
    # the located matrices are a cache, rebuilt on the first sample
    to_args=lambda r, _: {n: getattr(r, n) for n in _ROUTE_SAMPLER_NAMES},
    from_args=lambda a, _: _RouteSampler(**a),
    names=_ROUTE_SAMPLER_NAMES,
)
_codec.register(
    PortModeField,
    to_args=_port_mode_field_args,
    from_args=_port_mode_field_from_args,
    names=("section", "order", "basis", "coords", "polarization", "bend"),
)
_codec.register(LineSection)
_codec.register(Job, from_args=_job_from_args, names=("model", "problem", "options"))
