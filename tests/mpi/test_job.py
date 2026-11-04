"""Files across ranks: bundles, the command line, one `.vtu` (ADR 0064).

A bundle written by any rank count is an ordinary one-process bundle, so each test
writes under ranks and reads in pytest's one process, against the same solve there.
"""

import json
import sys
from pathlib import Path

import gradient_cases as cases
import jax
import jax.numpy as jnp
import meshio
import mpi_cases
import ngsolve as ngs
import numpy as np
import pytest
from ranks import RanksFailed, on_ranks, on_ranks_argv

import gfem
from gfem import _mpi, _petsc
from gfem._engine import fields as engine_fields

RANKS = (3, 4, 8)
#: Relative to the largest one-process value: ticket 08's gate on a direct solve,
#: ADR 0070's on BDDC.
GATES = {"bddc_gmres": 1e-9, "direct": 1e-13}
SOLVERS = (
    "bddc_gmres",
    pytest.param(
        "direct",
        marks=pytest.mark.skipif(not _petsc.has_mpi(), reason="needs the MPI stack"),
    ),
)
FREQUENCIES = (18e9, 27e9)
#: The block's volume entity, removed as a model's ``excluded=`` removes it.
SUPPORT = ("air", "diel", "metal")
EXCLUDED = ((3, 3),)


def _world():
    from mpi4py import MPI

    return MPI.COMM_WORLD


# --- the map to one-process numbering -------------------------------------------------


def _walk(fes, mesh, node_order, tetrahedron_order):
    """``(kind, a, b, c, position) -> DOF`` by `GetDofNrs`, in gfem's numbers."""
    rows = {}
    for kind, nodes, node_type in (
        (1, mesh.edges, ngs.EDGE),
        (2, mesh.faces, ngs.FACE),
    ):
        for node in nodes:
            dofs = fes.GetDofNrs(ngs.NodeId(node_type, node.nr))
            vertices = sorted(int(node_order[v.nr]) for v in node.vertices)
            key = (kind, *vertices, *[-1] * (3 - len(vertices)))
            rows |= {(*key, j): d for j, d in enumerate(dofs) if d >= 0}
    for element in range(mesh.ne):
        dofs = fes.GetDofNrs(ngs.NodeId(ngs.CELL, element))
        key = (3, int(tetrahedron_order[element]), -1, -1)
        rows |= {(*key, j): d for j, d in enumerate(dofs) if d >= 0}
    return rows


def _walked(mesh, order, comm):
    built, fes, _ = engine_fields.fe_space(
        mesh, SUPPORT, EXCLUDED, "hcurl", order, comm=comm
    )
    return (
        _walk(fes, built.mesh, built.node_order, built.tetrahedron_order),
        int(fes.ndof),
    )


def _box():
    return mpi_cases.permuted(mpi_cases.box(4))


def _map_against_the_walk(order, serial):
    comm = _world()
    mesh = _box()
    table, ndof = _walked(mesh, order, comm)
    walked = np.full(ndof, -1, dtype=np.int64)
    for key, dof in table.items():
        walked[dof] = serial[key]
    numpy_map = engine_fields.one_process_dofs(mesh, SUPPORT, EXCLUDED, order, comm)
    same = np.array_equal(numpy_map, walked) and (walked >= 0).all()
    return {
        "same": _mpi.agree_all(comm, flag=bool(same)),
        "held": comm.allreduce(ndof),
    }


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("ranks", RANKS)
def test_the_numpy_map_is_the_walk(ranks, order):
    serial, ndof = _walked(_box(), order, None)
    got, *_ = on_ranks(_map_against_the_walk, order, serial, ranks=ranks, limit=120)
    assert got["same"]
    assert got["held"] >= ndof


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_map_at_one_process_is_the_identity(order):
    mesh = _box()
    _, ndof = _walked(mesh, order, None)
    numbers = engine_fields.one_process_dofs(mesh, SUPPORT, EXCLUDED, order)
    assert np.array_equal(numbers, np.arange(ndof))


def _column_naming_its_numbers(message):
    comm = _world()
    mesh = _box()
    _mpi.GATHER_MESSAGE = message
    support, order = ("air", "diel"), 2
    numbers = engine_fields.one_process_dofs(mesh, support, (), order, comm)
    width, ndof = engine_fields.dof_shape(mesh, support, (), "hcurl", order, comm=comm)
    piece = np.zeros(width, complex)
    piece[:ndof] = numbers * (1 + 2j)
    field = gfem.Field(
        coefficients=piece,
        mesh=mesh,
        quantity="E",
        unit="V/m",
        space="hcurl",
        order=order,
        support=support,
        omega=1.0,
        _comm=_mpi.static_comm(comm),
    )
    whole = field._gathered()
    if comm.rank == 0:
        exact = np.arange(len(whole.coefficients)) * (1 + 2j)
        return float(np.abs(np.asarray(whole.coefficients) - exact).max())
    return None


@pytest.mark.parametrize("message", [2**30, 7])
def test_a_gathered_column_is_exact_in_any_number_of_messages(message):
    error, *rest = on_ranks(
        _column_naming_its_numbers, message, ranks=4, limit=120, replicated=False
    )
    print(f"gather message={message} max error={error:.3g}")  # noqa: T201
    assert error == 0.0
    assert rest == [None] * 3


# --- a bundle ---------------------------------------------------------------------


def _options(solver):
    return cases.tight(solver, threads=1)


def _job(directory, solver):
    line = cases.layered_line(Path(directory))
    values, _ = cases.line_design(line, seed=2)
    return gfem.Job(
        cases.with_fields(line, values),
        gfem.Driven(frequency=FREQUENCIES, keep_fields=[FREQUENCIES[1]]),
        _options(solver),
    )


def _relative(a, b):
    b = np.asarray(b)
    return float(np.abs(np.asarray(a) - b).max() / np.abs(b).max())


def _assert_the_same_result(loaded, one, gate, what):
    assert loaded.kept == one.kept
    assert loaded.e_fields
    for got, expected, (frequency, port) in zip(
        loaded.e_fields, one.e_fields, one.kept, strict=True
    ):
        assert got.coefficients.shape == expected.coefficients.shape
        error = _relative(got.coefficients, expected.coefficients)
        print(f"{what} {frequency:g} Hz {port} field error={error:.3g}")  # noqa: T201
        assert error < gate
    error = _relative(loaded.s.values, one.s.values)
    print(f"{what} S error={error:.3g}")  # noqa: T201
    assert error < gate * 100


@pytest.mark.parametrize("solver", SOLVERS)
def test_a_bundle_from_the_command_line_reads_on_one_process(tmp_path, solver):
    job = _job(tmp_path, solver)
    path = job.to_json(tmp_path / "job.json")
    out = tmp_path / "out"
    code, log = on_ranks_argv(
        [sys.executable, "-m", "gfem", "run", str(path), "--out", str(out)],
        ranks=4,
        limit=300,
    )
    assert code == 0, log
    loaded = gfem.load_result(out)
    assert json.loads((out / "result.json").read_text())["run"]["ranks"] == 4
    _assert_the_same_result(loaded, job.run(), GATES[solver], f"cli {solver}")
    points = _inside(loaded.e_fields[0].mesh)
    error = _relative(loaded.e_fields[0].at(points), job.run().e_fields[0].at(points))
    print(f"cli {solver} at error={error:.3g}")  # noqa: T201
    assert error < GATES[solver] * 100


def _inside(mesh, n=16, seed=3):
    """Points well inside ``n`` tets, off every face."""
    chosen = np.random.default_rng(seed).choice(len(mesh.tetrahedra), n, replace=False)
    return np.asarray(mesh.nodes)[np.asarray(mesh.tetrahedra)[chosen]].mean(axis=1)


def _run_bundle(directory, out):
    job = gfem.Job.from_json(Path(directory) / "job.json")
    result = job.run(out)
    return {
        "s": np.asarray(result.s.values),
        "shape": result.e_fields[0].coefficients.shape,
    }


@pytest.mark.parametrize("ranks", [3, 8])
def test_any_rank_count_writes_the_same_kind_of_bundle(tmp_path, ranks):
    job = _job(tmp_path, "bddc_gmres")
    job.to_json(tmp_path / "job.json")
    out = tmp_path / "out"
    on_ranks(_run_bundle, tmp_path, out, ranks=ranks, limit=300)
    loaded = gfem.load_result(out)
    _assert_the_same_result(loaded, job.run(), GATES["bddc_gmres"], f"ranks={ranks}")


def test_a_moved_mesh_writes_its_bundle_in_the_same_numbering(tmp_path):
    job = _job(tmp_path, "bddc_gmres")
    line = job.model
    nodes = np.asarray(line.mesh.nodes).copy()
    nodes[:, 0] += 1e-4 * np.sin(np.pi * nodes[:, 0] / 4e-3)
    moved = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(line.mesh), [jnp.asarray(nodes)]
    )
    job = gfem.Job(line.replace(mesh=moved), job.problem, job.options)
    job.to_json(tmp_path / "job.json")
    out = tmp_path / "out"
    on_ranks(_run_bundle, tmp_path, out, ranks=3, limit=300)
    loaded = gfem.load_result(out)
    np.testing.assert_array_equal(loaded.e_fields[0].mesh.nodes, nodes)
    _assert_the_same_result(loaded, job.run(), GATES["bddc_gmres"], "moved")


def _wave_job(directory):
    mesh = cases.wave_mesh(Path(directory))
    return gfem.Job(
        cases.wave_line(mesh, "lossy"),
        gfem.Driven(frequency=(12e9,), keep_fields=True),
        _options("bddc_gmres"),
    )


def _run_wave_bundle(directory, out):
    job = gfem.Job.from_json(Path(directory) / "job.json")
    return np.asarray(job.run(out).s.values)


def test_a_bundle_with_wave_ports_holds_their_modes(tmp_path):
    job = _wave_job(tmp_path)
    job.to_json(tmp_path / "job.json")
    out = tmp_path / "out"
    on_ranks(_run_wave_bundle, tmp_path, out, ranks=3, limit=300)
    loaded, one = gfem.load_result(out), job.run()
    _assert_the_same_result(loaded, one, GATES["bddc_gmres"], "wave")
    assert list(loaded.port_modes) == list(one.port_modes)
    for port, modes in one.port_modes.items():
        np.testing.assert_allclose(
            np.asarray(loaded.port_modes[port].gamma),
            np.asarray(modes.gamma),
            rtol=1e-10,
        )


# --- what raises on every rank ----------------------------------------------------


def _error_then_a_collective(call):
    """``call()``'s error by its type and text, and proof that the job goes on."""
    try:
        call()
    except Exception as err:  # noqa: BLE001 - the test reads it
        return type(err).__name__, str(err), _world().allreduce(1)
    return None, None, _world().allreduce(1)


def _write_under_a_file(directory, out):
    job = gfem.Job.from_json(Path(directory) / "job.json")
    return _error_then_a_collective(lambda: job.run(out))


def test_rank_0_failing_to_write_raises_on_every_rank(tmp_path):
    job = _job(tmp_path, "bddc_gmres")
    job.to_json(tmp_path / "job.json")
    in_the_way = tmp_path / "file"
    in_the_way.write_text("")
    got, *_ = on_ranks(
        _write_under_a_file, tmp_path, in_the_way / "out", ranks=3, limit=300
    )
    print(f"rank 0 write failure: {got}")  # noqa: T201
    assert got[0] == "NotADirectoryError"
    assert got[2] == 3


def _read_job(path):
    return _error_then_a_collective(lambda: gfem.Job.from_json(path))


def test_a_changed_mesh_raises_on_every_rank(tmp_path):
    path = _job(tmp_path, "bddc_gmres").to_json(tmp_path / "job.json")
    mesh = next(tmp_path.glob("*.msh"))
    mesh.write_text(mesh.read_text() + "\n")
    got, *_ = on_ranks(_read_job, path, ranks=3, limit=120)
    assert got[0] == "ValueError"
    assert "has changed" in got[1]
    assert got[2] == 3


def test_a_missing_sidecar_raises_on_every_rank(tmp_path):
    path = _job(tmp_path, "bddc_gmres").to_json(tmp_path / "job.json")
    path.with_suffix(".npz").unlink()
    got, *_ = on_ranks(_read_job, path, ranks=3, limit=120)
    assert got[0] == "FileNotFoundError"
    assert got[2] == 3


def _read_on_every_rank(directory):
    comm = _world()
    job = gfem.Job.from_json(Path(directory) / "job.json")
    return {
        "nodes": np.asarray(job.model.mesh.nodes),
        "design": np.asarray(job.model.materials["sub"].eps_r),
        "held": comm.allreduce(1),
    }


def test_a_job_reads_to_the_same_bits_on_every_rank(tmp_path):
    job = _job(tmp_path, "bddc_gmres")
    job.to_json(tmp_path / "job.json")
    got, *_ = on_ranks(_read_on_every_rank, tmp_path, ranks=3, limit=120)
    np.testing.assert_array_equal(got["nodes"], job.model.mesh.nodes)
    np.testing.assert_array_equal(got["design"], job.model.materials["sub"].eps_r)


def _a_value_with_no_axis():
    shared = _mpi.on_root(_world(), lambda: {"value": np.asarray(2.5)})
    return shared["value"]


def test_an_array_with_no_axis_crosses_ranks_as_it_is():
    value, *_ = on_ranks(_a_value_with_no_axis, ranks=3, limit=60)
    assert value.shape == ()
    assert value == 2.5


def _copies_with_jobs_of_their_own(directory):
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    own = Path(directory) / f"rank-{rank}"
    job = gfem.Job.from_json(own / "job.json", comm=MPI.COMM_SELF)
    result = job.run(own / "out", comm=MPI.COMM_SELF)
    loaded = gfem.load_result(own / "out", comm=MPI.COMM_SELF)
    assert np.array_equal(loaded.s.values, result.s.values)
    return np.asarray(result.s.values)


def test_copies_read_and_write_files_of_their_own(tmp_path):
    for rank in range(3):
        own = tmp_path / f"rank-{rank}"
        own.mkdir()
        gfem.Job(
            cases.layered_line(own), gfem.Driven(FREQUENCIES[1]), _options("bddc_gmres")
        ).to_json(own / "job.json")
    values = on_ranks(
        _copies_with_jobs_of_their_own, tmp_path, ranks=3, limit=300, replicated=False
    )
    for value in values[1:]:
        np.testing.assert_allclose(value, values[0], rtol=1e-9)
    assert all(
        (tmp_path / f"rank-{r}" / "out" / "result.json").exists() for r in range(3)
    )


# --- load_result ------------------------------------------------------------------


def _load_on_ranks(out):
    comm = _world()
    try:
        gfem.load_result(out)
    except gfem.DistributedUnsupportedError as err:
        return err.feature, str(err), comm.allreduce(1)
    return None, None, comm.allreduce(1)


def _load_as_copies(out, points):
    from mpi4py import MPI

    loaded = gfem.load_result(out, comm=MPI.COMM_SELF)
    return np.asarray(loaded.e_fields[0].at(points))


def test_a_bundle_with_fields_raises_on_ranks_and_loads_on_copies(tmp_path):
    out = tmp_path / "out"
    one = _job(tmp_path, "bddc_gmres").run(out)
    feature, text, held = on_ranks(_load_on_ranks, out, ranks=3, limit=120)[0]
    print(f"load_result above one rank: {feature}: {text}")  # noqa: T201
    assert feature == "load_result_fields"
    assert "COMM_SELF" in text
    assert held == 3
    points = _inside(one.e_fields[0].mesh)
    values, *_ = on_ranks(_load_as_copies, out, points, ranks=3, limit=180)
    error = _relative(values, one.e_fields[0].at(points))
    print(f"load_result COMM_SELF at error={error:.3g}")  # noqa: T201
    assert error == 0.0


def _load_on_rank_0_alone(out):
    comm = _world()
    s = None
    if comm.rank == 0:
        s = np.asarray(gfem.load_result(out).s.values)
    comm.Barrier()
    return comm.bcast(s, root=0)


def _load_missing_on_rank_1(out):
    path = Path(out) / "missing" if _world().rank == 1 else out
    gfem.load_result(path)
    _world().Barrier()


def test_a_bundle_missing_on_one_rank_ends_the_job(tmp_path):
    out = tmp_path / "out"
    line = cases.layered_line(tmp_path)
    gfem.Job(line, gfem.Driven(FREQUENCIES[1]), _options("bddc_gmres")).run(out)
    with pytest.raises(RanksFailed) as caught:
        on_ranks(_load_missing_on_rank_1, out, ranks=3, limit=120)
    # the section ended the job: nothing reached the worker's own handler
    assert caught.value.tracebacks == {}
    assert "gfem: rank 1:" in caught.value.log
    assert "FileNotFoundError" in caught.value.log


def test_a_bundle_without_fields_reads_on_one_rank_alone(tmp_path):
    out = tmp_path / "out"
    line = cases.layered_line(tmp_path)
    one = gfem.Job(line, gfem.Driven(FREQUENCIES[1]), _options("bddc_gmres")).run(out)
    s, *_ = on_ranks(_load_on_rank_0_alone, out, ranks=3, limit=120)
    np.testing.assert_array_equal(s, one.s.values)


# --- one .vtu ---------------------------------------------------------------------


def _write_vtus(directory, solver, e_path, b_path, bad_path):
    job = gfem.Job.from_json(Path(directory) / "job.json")
    field = job.run().e_fields[0]
    field.to_vtu(e_path)
    field.b().to_vtu(b_path)
    return _error_then_a_collective(lambda: field.to_vtu(bad_path))[:2], solver


def _assert_matches_sample(path, field, gate, what):
    points, cells, point_data, cell_data = engine_fields.sample(field)
    read = meshio.read(path)
    assert path.exists()
    position = float(np.abs(read.points - points).max() / np.abs(points).max())
    print(f"vtu {what} points error={position:.3g}")  # noqa: T201
    assert position == 0.0
    np.testing.assert_array_equal(read.cells_dict["tetra"], cells)
    for name, values in cell_data.items():
        np.testing.assert_array_equal(read.cell_data[name][0], values)
    assert read.point_data.keys() == point_data.keys()
    for name, values in point_data.items():
        error = _relative(read.point_data[name], values)
        print(f"vtu {what} {name} error={error:.3g}")  # noqa: T201
        assert error < gate


@pytest.mark.parametrize("solver", SOLVERS)
def test_to_vtu_gathers_one_file_matching_sample(tmp_path, solver):
    job = _job(tmp_path, solver)
    job.to_json(tmp_path / "job.json")
    e_path, b_path = tmp_path / "e.vtu", tmp_path / "b.vtu"
    in_the_way = tmp_path / "file"
    in_the_way.write_text("")
    got, *_ = on_ranks(
        _write_vtus,
        tmp_path,
        solver,
        e_path,
        b_path,
        in_the_way / "x.vtu",
        ranks=4,
        limit=300,
    )
    assert got[0][0] == "NotADirectoryError"
    field = job.run().e_fields[0]
    gate = GATES[solver] * 100
    _assert_matches_sample(e_path, field, gate, f"E {solver}")
    _assert_matches_sample(b_path, field.b(), gate, f"B {solver}")
