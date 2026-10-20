"""Serialized runs: `Job` (``job.json``) and `load_result` (``result.json``).

A job bundle is ``job.json``, the mesh file it references (by path and
sha256) and, when the job has arrays, a ``.npz`` sidecar next to it with
every array leaf (material fields, moved mesh nodes, ...) under its pytree
leaf path. A result bundle is a directory with ``result.json`` and
``result.npz``. Both documents carry ``schema_version`` and
``gfem_version``: gfem reads older schema versions and refuses newer ones.

Run a job bundle with ``python -m gfem run job.json --out results/`` (or
`Job.run`), and read the result back with `load_result`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from gfem import _codec
from gfem._boundaries import (
    PEC,
    PMC,
    Absorbing,
    Conductor,
    Impedance,
    LumpedPort,
    WavePort,
)
from gfem._checks import is_traced
from gfem._materials import Material, MaterialField
from gfem._port_section import PortSection
from gfem._pytree import register_fields
from gfem._sheet import SheetField
from gfem.measurements import InterfaceParticipation
from gfem.mesh import Mesh
from gfem.model import EnclosedBy, Model
from gfem.options import SolverOptions
from gfem.problems import Driven, Eigenmode, Electrostatic, PortMode
from gfem.results import (
    DrivenResult,
    EigenmodeResult,
    ElectrostaticResult,
    Field,
    NetworkParameters,
    PortModeField,
    PortModeResult,
    ReducedFields,
    SweepInfo,
)
from gfem.solve import solve

__all__ = ["Job", "load_result"]

_JOB_FORMAT = "gfem.job"
_RESULT_FORMAT = "gfem.result"
RESULT_JSON = "result.json"
RESULT_NPZ = "result.npz"


@dataclasses.dataclass(frozen=True, eq=False)
class Job:
    """A self-contained run: a model, a problem and solver options.

    The unit gsim and cloud runners exchange. `to_json` writes it as a job
    bundle; ``python -m gfem run job.json --out DIR`` or `run` solves it.
    A job is a JAX pytree whose leaves are the model's and the problem's.

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
        if not isinstance(getattr(type(self.problem), "_rule", None), str):
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

    def run(self, out: str | Path | None = None, *, device: str | None = None) -> Any:
        """Solve the job: ``gfem.solve(model, problem, options=options)``.

        Args:
            out: If given, a directory to write the result bundle to
                (``result.json`` and ``result.npz``, read by `load_result`).
            device: Overrides ``options.device`` (e.g. a job written for a
                GPU run on a CPU machine). The bundle records the device used.

        Returns:
            The problem type's result.
        """
        options = self.options
        if device is not None:
            options = dataclasses.replace(options, device=device)
        start = time.perf_counter()
        result = solve(self.model, self.problem, options=options)
        elapsed = time.perf_counter() - start
        if out is not None:
            _write_result(
                Path(out),
                result,
                {
                    "device": options.device,
                    "device_override": device,
                    "options": options,
                    "wall_time_s": elapsed,
                },
            )
        return result

    def to_json(self, path: str | Path) -> Path:
        """Write the job bundle's ``job.json`` (and its ``.npz`` sidecar).

        The mesh is referenced by its file path and sha256; the path is
        stored relative to the JSON file's directory when the mesh lies
        inside it, so a directory holding the job and its mesh can be moved
        or uploaded as one bundle. The sidecar, ``<stem>.npz`` next to the
        JSON file, holds every array leaf of the job (material fields,
        moved mesh nodes) under its pytree leaf path; it is written only
        when there are arrays.

        Args:
            path: The JSON file to write; its directory is created.

        Returns:
            The JSON file's path.

        Raises:
            TypeError: The job holds values traced by JAX.
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
    def from_json(cls, path: str | Path) -> Job:
        """Read a job bundle written by `to_json`.

        The mesh is read again and its sha256 checked.

        Raises:
            FileNotFoundError: The JSON file, its sidecar or its mesh is
                missing.
            ValueError: Not a job file, a newer schema version, a type this
                gfem does not know, or a mesh file that changed.
        """
        path = Path(path)
        doc = _codec.check_header(_read_json(path), _JOB_FORMAT, "gfem job file", path)
        job = _decode(doc, "job", path)
        if not isinstance(job, cls):
            raise ValueError(f"{path} is not a gfem job file (no job in it).")  # noqa: TRY004
        return job


# The problem may hold arrays one day; the options never do.
register_fields(Job, ("model", "problem"))


def load_result(path: str | Path) -> Any:
    """Read a result bundle written by ``python -m gfem run`` or `Job.run`.

    Args:
        path: The result directory, or its ``result.json``.

    Returns:
        The result, e.g. a `gfem.ElectrostaticResult`, with the same values
        as the in-process solve.

    Raises:
        FileNotFoundError: No ``result.json``, or its sidecar is missing.
        ValueError: Not a result file, or a newer schema version.
    """
    path = Path(path)
    if path.is_dir():
        path /= RESULT_JSON
    doc = _codec.check_header(
        _read_json(path), _RESULT_FORMAT, "gfem result file", path
    )
    return _decode(doc, "result", path)


def _write_result(out: Path, result: Any, run: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    encoder = _codec.Encoder(_codec.Context(out.resolve()))
    body = encoder.encode(result)
    run_json = encoder.encode(run, ".run")
    doc = _codec.header(_RESULT_FORMAT) | {
        "arrays": RESULT_NPZ if encoder.arrays else None,
        "run": run_json,
        "result": body,
    }
    if encoder.arrays:
        _save_npz(out / RESULT_NPZ, encoder.arrays)
    # result.json last: its presence means the bundle is complete.
    (out / RESULT_JSON).write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}.")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise ValueError(f"{path} is not valid JSON: {err}.") from None


def _decode(doc: dict, key: str, path: Path) -> Any:
    arrays: dict[str, np.ndarray] = {}
    if doc.get("arrays"):
        sidecar = path.parent / doc["arrays"]
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"{path} keeps its arrays in {sidecar}, which is missing."
            )
        with np.load(sidecar, allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
    if key not in doc:
        raise ValueError(f"{path} has no {key!r} entry.")
    decoder = _codec.Decoder(_codec.Context(path.parent.resolve()), arrays)
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
    source = mesh.source
    if source is None or not source.is_file():
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
    nodes = mesh.nodes
    if is_traced(nodes) or not np.array_equal(np.asarray(nodes), mesh.imported_nodes):
        args["nodes"] = nodes  # moved (shape optimization): keep them
    return args


def _mesh_from_args(args: dict[str, Any], context: _codec.Context) -> Mesh:
    path = context.base / args["path"]
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
    mesh = Mesh.from_gmsh(path, length_unit=args["length_unit"])
    if args.get("nodes") is not None:
        nodes = jnp.asarray(args["nodes"])
        if nodes.shape != mesh.imported_nodes.shape:
            raise ValueError(
                f"The job's moved nodes have shape {nodes.shape}, but the mesh "
                f"{path} has {mesh.imported_nodes.shape}."
            )
        mesh = Mesh(nodes, mesh._topology)  # noqa: SLF001
    return mesh


def _model_args(model: Model, _: _codec.Context) -> dict[str, Any]:
    return {
        "mesh": model.mesh,
        "materials": dict(model.materials),
        "boundaries": model.boundaries,
        "ports": model.ports,
        "excluded": model.excluded,
    }


def _model_from_args(args: dict[str, Any], _: _codec.Context) -> Model:
    mesh = args.pop("mesh")
    return Model(mesh, **args)


def _electrostatic_args(problem: Electrostatic, _: _codec.Context) -> dict[str, Any]:
    return {"terminals": dict(problem.terminals), "ground": problem.ground}


def _port_mode_field_args(field: PortModeField, _: _codec.Context) -> dict[str, Any]:
    space = field._space  # noqa: SLF001
    return {
        "section": space.section,
        "order": space.order,
        "basis": field._basis,  # noqa: SLF001
        "coords": field._coords,  # noqa: SLF001
    }


def _port_mode_field_from_args(args: dict[str, Any], _: _codec.Context) -> Any:
    from gfem._engine.port_mode import PortModeSpace

    space = PortModeSpace(args["section"], args["order"])
    return PortModeField(space, args["basis"], args["coords"])


def _job_from_args(args: dict[str, Any], _: _codec.Context) -> Job:
    return Job(args["model"], args["problem"], args.get("options", SolverOptions()))


_codec.register(Mesh, to_args=_mesh_args, from_args=_mesh_from_args)
_codec.register(Model, to_args=_model_args, from_args=_model_from_args)
_codec.register(EnclosedBy, from_args=lambda a, _: EnclosedBy(tuple(a["surfaces"])))
for _cls in (
    Material,
    MaterialField,
    SheetField,
    PEC,
    PMC,
    Absorbing,
    Conductor,
    Impedance,
    LumpedPort,
    WavePort,
    SolverOptions,
    PortMode,
    ElectrostaticResult,
    Field,
    PortModeResult,
    Driven,
    DrivenResult,
    NetworkParameters,
    PortSection,
    SweepInfo,
    ReducedFields,
    Eigenmode,
    EigenmodeResult,
    InterfaceParticipation,
):
    _codec.register(_cls)
_codec.register(Electrostatic, to_args=_electrostatic_args)
_codec.register(
    PortModeField, to_args=_port_mode_field_args, from_args=_port_mode_field_from_args
)
_codec.register(Job, from_args=_job_from_args)
