"""gfem.Job and gfem.load_result: the job.json / result.json bundles."""

import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem

DATA = Path(__file__).parent / "data"
LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}
PROBLEM = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})


@pytest.fixture
def mesh_path(tmp_path):
    return sm.write_msh22(tmp_path / "stack.msh", sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))


@pytest.fixture
def model(mesh_path):
    return gfem.Model(gfem.Mesh.from_gmsh(mesh_path), materials=MATERIALS)


def _round_trip(job, tmp_path):
    path = job.to_json(tmp_path / "job.json")
    return gfem.Job.from_json(path)


def _assert_same_model(a, b):
    assert a.mesh.source == b.mesh.source
    assert a.mesh.length_unit == b.mesh.length_unit
    np.testing.assert_array_equal(a.mesh.nodes, b.mesh.nodes)
    assert list(a.materials) == list(b.materials)
    for region, m in a.materials.items():
        n = b.materials[region]
        assert type(m) is type(n)
        if isinstance(m, gfem.MaterialField):
            assert m.region == n.region
            for prop in gfem.model.MaterialField.__dataclass_fields__:
                if prop == "region":
                    continue
                x, y = getattr(m, prop), getattr(n, prop)
                assert (x is None) == (y is None)
                if x is not None:
                    assert x.dtype == y.dtype
                    np.testing.assert_array_equal(x, y)
        else:
            assert m == n
    assert a.boundaries == b.boundaries
    assert a.ports == b.ports
    assert a.excluded == b.excluded


# --- the job file -----------------------------------------------------------------


def test_job_round_trips_model_problem_and_options(model, tmp_path):
    options = gfem.SolverOptions(order=1, linear_solver="direct", threads=2)
    job = gfem.Job(model, PROBLEM, options)
    back = _round_trip(job, tmp_path)
    _assert_same_model(job.model, back.model)
    assert back.problem == PROBLEM
    assert back.options == options


def test_options_default_to_solver_defaults(model, tmp_path):
    job = gfem.Job(model, PROBLEM)
    assert job.options == gfem.SolverOptions()
    assert _round_trip(job, tmp_path).options == gfem.SolverOptions()


def test_job_json_is_versioned_and_hashes_the_mesh(model, mesh_path, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    assert doc["schema_version"] == 1
    assert doc["gfem_version"] == gfem.__version__
    mesh = doc["job"]["model"]["mesh"]
    assert mesh["path"] == "stack.msh"  # next to job.json: stored relative
    import hashlib

    assert mesh["sha256"] == hashlib.sha256(mesh_path.read_bytes()).hexdigest()


def test_bundle_moves_with_its_directory(model, tmp_path):
    gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    moved = shutil.copytree(tmp_path, tmp_path.parent / (tmp_path.name + "_moved"))
    back = gfem.Job.from_json(moved / "job.json")
    assert back.model.mesh.source == (moved / "stack.msh").resolve()


def test_mesh_outside_the_job_directory_is_stored_absolute(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "sub" / "job.json")
    doc = json.loads(path.read_text())
    assert doc["job"]["model"]["mesh"]["path"] == model.mesh.source.as_posix()
    _assert_same_model(model, gfem.Job.from_json(path).model)


def test_a_changed_mesh_file_is_refused(model, mesh_path, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    mesh_path.write_text(mesh_path.read_text() + "\n")
    with pytest.raises(ValueError, match="sha256"):
        gfem.Job.from_json(path)


def test_a_missing_mesh_file_is_reported(model, mesh_path, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    mesh_path.unlink()
    with pytest.raises(FileNotFoundError, match=r"stack\.msh"):
        gfem.Job.from_json(path)


def test_material_fields_go_to_the_sidecar_by_leaf_path(model, tmp_path):
    n = len(model.mesh.elements("oxide"))
    field = gfem.MaterialField(
        eps_r=jnp.linspace(2.0, 4.0, n) - 0.01j, conductivity=jnp.asarray(0.5)
    )
    uniform = gfem.MaterialField.uniform("nitride", gfem.Material(eps_r=7.5))
    job = gfem.Job(
        model.replace(materials={"oxide": field, "nitride": uniform}), PROBLEM
    )
    path = job.to_json(tmp_path / "job.json")

    with np.load(tmp_path / "job.npz", allow_pickle=False) as npz:
        keys = set(npz.files)
    array_paths = {
        jax.tree_util.keystr(p)
        for p, leaf in jax.tree_util.tree_flatten_with_path(job)[0]
        if leaf is not None
    }
    # Every array leaf but the (unmoved) mesh nodes, which the mesh file holds.
    assert keys == array_paths - {".model.mesh.nodes"}
    assert ".model.materials['oxide'].eps_r" in keys
    _assert_same_model(job.model, gfem.Job.from_json(path).model)


def test_moved_mesh_nodes_are_kept(model, tmp_path):
    nodes = model.mesh.nodes.at[0, 2].add(1e-8)
    moved = model.replace(mesh=gfem.Mesh(nodes, model.mesh._topology))
    path = gfem.Job(moved, PROBLEM).to_json(tmp_path / "job.json")
    with np.load(tmp_path / "job.npz") as npz:
        assert ".model.mesh.nodes" in npz.files
    _assert_same_model(moved, gfem.Job.from_json(path).model)


def test_boundaries_ports_and_exclusions_round_trip(tmp_path):
    path = sm.write_msh22(
        tmp_path / "three.msh",
        sm.plate(4.0, 3.0, (("a", 0.5), ("b", 0.25), ("c", 0.25)), (1, 1, 1)),
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={
            "a": gfem.Material(
                eps_r=(2.0, 3.0, 4.0),
                axes=((0, 1, 0), (1, 0, 0), (0, 0, 1)),
                tan_delta=(1e-3, 2e-3, 3e-5),
                name="sapphire",
            ),
            "b": gfem.Material(eps_r=1.0, conductivity=2.5, thermal_conductivity=1.4),
        },
        boundaries=[
            gfem.PEC("bottom"),
            gfem.Absorbing("side", order=2),
            gfem.Conductor("a__b", conductivity=5.8e7, thickness=1e-6),
        ],
        ports=[
            gfem.LumpedPort("p1", "b__c", "+z", resistance=25.0, inductance=1e-9),
        ],
        excluded=("c",),
    )
    back = _round_trip(gfem.Job(model, PROBLEM), tmp_path)
    _assert_same_model(model, back.model)


def test_wave_ports_and_enclosed_by_round_trip(tmp_path):
    path = sm.write_msh22(tmp_path / "stack.msh", sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path)
    model = gfem.Model(
        mesh,
        materials={"oxide": gfem.Material(eps_r=3.9)},
        ports=[
            gfem.WavePort(
                "w1",
                "oxide__nitride",
                mode=2,
                offset=1e-6,
                voltage_path=((0, 0, 0), (0, 0, 1e-6)),
                pec_edges=("side",),
            ),
            gfem.LumpedPort("p2", ["bottom"], ["+x"]),
        ],
        excluded=(gfem.enclosed_by("oxide__nitride", "top", "side"),),
    )
    back = _round_trip(gfem.Job(model, PROBLEM), tmp_path)
    _assert_same_model(model, back.model)


def test_job_checks_its_inputs(model):
    with pytest.raises(TypeError, match="Model"):
        gfem.Job("model", PROBLEM)
    with pytest.raises(TypeError, match="problem"):
        gfem.Job(model, "problem")
    with pytest.raises(TypeError, match="SolverOptions"):
        gfem.Job(model, PROBLEM, {"order": 1})


def test_traced_values_cannot_be_written(model, tmp_path):
    n = len(model.mesh.elements("oxide"))

    def write(eps):
        field = gfem.MaterialField(eps_r=eps)
        job = gfem.Job(model.with_material("oxide", field), PROBLEM)
        job.to_json(tmp_path / "job.json")
        return eps.sum()

    with pytest.raises(TypeError, match="traced"):
        jax.grad(write)(jnp.full(n, 2.0))


# --- versioning ---------------------------------------------------------------------


def _edit(path, **changes):
    doc = json.loads(path.read_text())
    doc.update(changes)
    path.write_text(json.dumps(doc))


def test_newer_schema_is_refused_with_an_upgrade_message(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    _edit(path, schema_version=2, gfem_version="9.0.0")
    with pytest.raises(ValueError, match=r"schema_version 2.*gfem 9\.0\.0.*upgrade"):
        gfem.Job.from_json(path)


def test_older_schema_is_migrated(model, tmp_path, monkeypatch):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    codec = __import__("gfem._codec", fromlist=["_"])

    def v1_to_v2(doc):
        doc["job"]["options"]["order"] = 1
        return doc

    monkeypatch.setattr(codec, "SCHEMA_VERSION", 2)
    monkeypatch.setitem(codec.MIGRATIONS, 1, v1_to_v2)
    assert gfem.Job.from_json(path).options.order == 1


@pytest.mark.parametrize(
    "changes",
    [{"schema_version": "1"}, {"schema_version": 0}, {"format": "gfem.result"}],
)
def test_not_a_job_file_is_refused(model, tmp_path, changes):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    _edit(path, **changes)
    with pytest.raises(ValueError, match="gfem job file"):
        gfem.Job.from_json(path)


def test_unknown_type_asks_for_a_newer_gfem(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    doc["job"]["problem"]["$type"] = "Magnetostatic"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=r"'Magnetostatic'.*upgrade gfem"):
        gfem.Job.from_json(path)


# --- running and results -------------------------------------------------------------


def test_run_equals_solve_and_result_round_trips(model, tmp_path):
    options = gfem.SolverOptions(order=1)
    expected = gfem.solve(model, PROBLEM, options=options)
    job = gfem.Job.from_json(
        gfem.Job(model, PROBLEM, options).to_json(tmp_path / "j.json")
    )
    result = job.run(out=tmp_path / "results")
    # The same to rounding: threaded assembly does not fix the summation order.
    np.testing.assert_allclose(result.capacitance, expected.capacitance, rtol=1e-12)

    loaded = gfem.load_result(tmp_path / "results")
    assert type(loaded) is gfem.ElectrostaticResult
    assert loaded.terminals == expected.terminals
    assert loaded.capacitance.dtype == expected.capacitance.dtype
    np.testing.assert_array_equal(loaded.capacitance, result.capacitance)
    assert list(loaded.energy) == list(result.energy)
    for region, energy in result.energy.items():
        np.testing.assert_array_equal(loaded.energy[region], energy)

    doc = json.loads((tmp_path / "results" / "result.json").read_text())
    assert doc["schema_version"] == 1
    assert doc["gfem_version"] == gfem.__version__
    assert doc["run"]["device"] == "cpu"
    assert gfem.load_result(tmp_path / "results" / "result.json").terminals == (
        "top",
        "mid",
    )


def test_load_result_refuses_a_job_file(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    with pytest.raises(ValueError, match="gfem result file"):
        gfem.load_result(path)


def test_load_result_needs_a_result(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"result\.json"):
        gfem.load_result(tmp_path)


@pytest.mark.parametrize(
    "walls",
    [gfem.PEC("walls"), gfem.Conductor("walls", conductivity=5.8e7, thickness=1e-6)],
    ids=["pec", "conductor"],
)
def test_port_mode_result_round_trips_with_its_field(tmp_path, walls):
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[walls],
        ports=[gfem.WavePort("p1", "port1")],
    )
    problem = gfem.PortMode("p1", [8e9, 10e9])
    job = gfem.Job(model, problem, gfem.SolverOptions(order=1))
    job = gfem.Job.from_json(job.to_json(tmp_path / "job.json"))
    assert job.problem == problem
    result = job.run(out=tmp_path / "results")
    loaded = gfem.load_result(tmp_path / "results")
    assert (loaded.port, loaded.mode) == (result.port, result.mode)
    for name in ("frequency", "gamma", "Z_PV", "residual", "solved"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
    points = np.asarray(mesh.nodes)[mesh.triangles[mesh.elements("port1")][:, 0]]
    np.testing.assert_array_equal(loaded.e_field.at(points), result.e_field.at(points))
