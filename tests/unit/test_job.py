"""gfem.Job and gfem.load_result: the job.json / result.json bundles."""

import dataclasses
import json
import shutil
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from cascade_cases import box_mesh, lines
from petsc_recipe import SOLVER_OPTIONS
from solver_misses import cannot_converge_at

import gfem
from gfem._refine import refine_model
from gfem.mesh import _topology_arrays

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
            assert m.axes == n.axes
            for prop in gfem.model.MaterialField.__dataclass_fields__:
                if prop in ("region", "axes"):
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


def test_job_round_trips_petsc_options(model, tmp_path):
    # PETSc is not needed to write or read the job, only to run it; the
    # options are the job file the PETSc and SLEPc recipe shows
    options = gfem.SolverOptions(**SOLVER_OPTIONS["job"])
    path = gfem.Job(model, PROBLEM, options).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    written = doc["job"]["options"]
    assert written["petsc_options"] == {"ksp_view": None, "pc_type": "lu"}
    assert (written["linear_solver"], written["eigen_solver"]) == ("petsc", "slepc")
    assert gfem.Job.from_json(path).options == options


def test_a_job_written_before_the_petsc_fields_reads_their_defaults(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    del doc["job"]["options"]["eigen_solver"]
    del doc["job"]["options"]["petsc_options"]
    path.write_text(json.dumps(doc))
    assert gfem.Job.from_json(path).options == gfem.SolverOptions()


def test_time_stepping_options_are_written_only_when_set(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    written = json.loads(path.read_text())["job"]["options"]
    assert "time_scheme" not in written
    assert "theta" not in written
    options = gfem.SolverOptions(time_scheme="theta", theta=0.5)
    path = gfem.Job(model, PROBLEM, options).to_json(tmp_path / "theta.json")
    written = json.loads(path.read_text())["job"]["options"]
    assert (written["time_scheme"], written["theta"]) == ("theta", 0.5)
    assert gfem.Job.from_json(path).options == options


def test_the_gauge_is_written_only_when_set(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    assert "gauge" not in json.loads(path.read_text())["job"]["options"]
    options = gfem.SolverOptions(gauge="none")
    path = gfem.Job(model, PROBLEM, options).to_json(tmp_path / "none.json")
    assert json.loads(path.read_text())["job"]["options"]["gauge"] == "none"
    assert gfem.Job.from_json(path).options == options


def test_options_default_to_solver_defaults(model, tmp_path):
    job = gfem.Job(model, PROBLEM)
    assert job.options == gfem.SolverOptions()
    assert _round_trip(job, tmp_path).options == gfem.SolverOptions()


def test_job_json_is_versioned_and_hashes_the_mesh(model, mesh_path, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    assert doc["schema_version"] == 2
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


def test_a_mesh_whose_file_is_gone_cannot_be_written(model, mesh_path, tmp_path):
    mesh_path.unlink()
    with pytest.raises(ValueError, match=r"its file .*stack\.msh is gone"):
        gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")


def _refined(model):
    mesh = model.mesh
    marked = np.zeros(len(mesh.tetrahedra), bool)
    marked[::3] = True
    return refine_model(model, marked)[0]


def _assert_same_topology(a, b):
    assert a.source is None
    assert b.source is None
    assert a.length_unit == b.length_unit
    assert a.regions == b.regions
    ya = _topology_arrays(b._topology)
    for name, x in _topology_arrays(a._topology).items():
        y = ya[name]
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)
        assert not y.flags.writeable
    assert np.asarray(a.nodes).dtype == np.asarray(b.nodes).dtype
    np.testing.assert_array_equal(a.nodes, b.nodes)


def test_a_refined_mesh_round_trips_through_the_sidecar(model, tmp_path):
    refined = _refined(model)
    path = gfem.Job(refined, PROBLEM).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    assert doc["schema_version"] == 2
    assert doc["job"]["model"]["mesh"]["source"] is None
    with np.load(tmp_path / "job.npz") as npz:
        assert ".model.mesh.tetrahedra" in npz.files
        assert ".model.mesh.nodes" not in npz.files
    back = gfem.Job.from_json(path).model
    _assert_same_topology(refined.mesh, back.mesh)
    _assert_same_model(refined, back)


def test_a_refined_moved_mesh_round_trips_with_its_nodes(model, tmp_path):
    refined = _refined(model)
    nodes = refined.mesh.nodes.at[0, 2].add(1e-8)
    moved = refined.replace(mesh=gfem.Mesh(nodes, refined.mesh._topology))
    path = gfem.Job(moved, PROBLEM).to_json(tmp_path / "job.json")
    with np.load(tmp_path / "job.npz") as npz:
        assert ".model.mesh.nodes" in npz.files
    back = gfem.Job.from_json(path).model
    _assert_same_topology(moved.mesh, back.mesh)


def test_a_refined_moved_mesh_is_refined_again_and_round_trips(model, tmp_path):
    moved = model.replace(mesh=gfem.Mesh(model.mesh.nodes * 1.01, model.mesh._topology))
    refined = _refined(moved)
    back = _round_trip(gfem.Job(refined, PROBLEM), tmp_path).model
    _assert_same_topology(refined.mesh, back.mesh)


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


def test_matrix_built_and_per_axis_conductivity_materials_round_trip(
    mesh_path, tmp_path
):
    c, s = np.cos(0.4), np.sin(0.4)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    tilted = gfem.Material(
        eps_r=r @ np.diag([2.0, 3.0, 5.0]) @ r.T,
        conductivity=r @ np.diag([0.0, 0.5, 1.0]) @ r.T,
        name="tilted",
    )
    wire = gfem.Material(eps_r=7.5, conductivity=(0.0, 0.0, 1.0))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(mesh_path), materials={"oxide": tilted, "nitride": wire}
    )
    back = _round_trip(gfem.Job(model, PROBLEM), tmp_path)
    _assert_same_model(model, back.model)


def test_material_fields_with_axes_round_trip(model, tmp_path):
    n = len(model.mesh.elements("oxide"))
    c, s = np.cos(0.4), np.sin(0.4)
    axes = ((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0))
    field = gfem.MaterialField(
        eps_r=jnp.tile(jnp.array([2.0, 3.0, 5.0]), (n, 1)),
        conductivity=jnp.array([0.0, 0.5, 1.0]),
        thermal_conductivity=jnp.ones(n),
        axes=axes,
    )
    crystal = gfem.Material(eps_r=(2.0, 3.0, 4.0), axes=axes)
    uniform = gfem.MaterialField.uniform("nitride", crystal)
    changed = model.replace(materials={"oxide": field, "nitride": uniform})
    back = _round_trip(gfem.Job(changed, PROBLEM), tmp_path)
    _assert_same_model(changed, back.model)
    assert back.model.materials["oxide"].axes == field.axes


def test_an_old_material_field_reads_without_axes(model, tmp_path):
    n = len(model.mesh.elements("oxide"))
    field = gfem.MaterialField(eps_r=jnp.linspace(2.0, 4.0, n))
    path = gfem.Job(model.with_material("oxide", field), PROBLEM).to_json(
        tmp_path / "job.json"
    )
    doc = json.loads(path.read_text())
    fields = list(_typed_in(doc, "MaterialField"))
    assert fields
    for f in fields:
        f.pop("axes")  # as 0.1.0 wrote it
    path.write_text(json.dumps(doc))
    back = gfem.Job.from_json(path).model.materials["oxide"]
    assert back.axes is None
    np.testing.assert_array_equal(back.eps_r, field.eps_r)


def _typed_in(node, tag):
    if isinstance(node, dict):
        if node.get("$type") == tag:
            yield node
        for v in node.values():
            yield from _typed_in(v, tag)
    elif isinstance(node, list):
        for v in node:
            yield from _typed_in(v, tag)


def _materials_in(node):
    if isinstance(node, dict):
        if node.get("$type") == "Material":
            yield node
        for v in node.values():
            yield from _materials_in(v)
    elif isinstance(node, list):
        for v in node:
            yield from _materials_in(v)


def test_an_old_job_with_equal_principal_values_reads_them_as_one(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    (oxide,) = (m for m in _materials_in(doc) if m["eps_r"] == 3.9)
    oxide["eps_r"] = [2.0, 2.0, 2.0]  # as 0.1.0 wrote Material(eps_r=(2, 2, 2))
    path.write_text(json.dumps(doc))
    back = gfem.Job.from_json(path).model.materials["oxide"]
    assert back == gfem.Material(eps_r=2.0)
    assert back.isotropic


def test_an_old_density_key_reads_as_mass_density_density_alias(model, tmp_path):
    n = len(model.mesh.elements("oxide"))
    field = gfem.MaterialField(mass_density=jnp.linspace(2000.0, 2400.0, n))
    nitride = gfem.Material(
        eps_r=7.5, mass_density=3100.0, thermo_optic_coefficient=4e-5
    )
    changed = model.replace(materials={"oxide": field, "nitride": nitride})
    path = gfem.Job(changed, PROBLEM).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    for tag in ("Material", "MaterialField"):
        nodes = list(_typed_in(doc, tag))
        assert nodes
        for node in nodes:  # as a gfem before the rename wrote it
            node["density"] = node.pop("mass_density")
            node.pop("thermo_optic_coefficient")
    path.write_text(json.dumps(doc))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # a read is silent
        back = gfem.Job.from_json(path).model.materials
    np.testing.assert_array_equal(back["oxide"].mass_density, field.mass_density)
    assert back["nitride"] == gfem.Material(eps_r=7.5, mass_density=3100.0)


def test_sheet_fields_round_trip(tmp_path):
    path = sm.write_msh22(
        tmp_path / "two.msh", sm.plate(4.0, 3.0, (("a", 0.5), ("b", 0.5)), (2, 1, 1))
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    fill = gfem.SheetField(
        fill=jnp.linspace(0.0, 1.0, len(mesh.elements("a__b"))), surface="a__b"
    )
    model = gfem.Model(
        mesh,
        materials={"a": gfem.Material(), "b": gfem.Material(eps_r=2.0)},
        boundaries=[gfem.Impedance("a__b", resistance=10.0, fill=fill)],
    )
    back = _round_trip(gfem.Job(model, PROBLEM), tmp_path)
    (boundary,) = back.model.boundaries
    assert (boundary.surface, boundary.resistance) == ("a__b", 10.0)
    sheet = boundary.fill
    assert isinstance(sheet, gfem.SheetField)
    assert sheet.surface == "a__b"
    np.testing.assert_array_equal(sheet.fill, fill.fill)


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

    with pytest.raises(
        TypeError, match=r"computed inside jax\.grad, jax\.jit or jax\.vmap"
    ):
        jax.grad(write)(jnp.full(n, 2.0))


# --- versioning ---------------------------------------------------------------------


def _edit(path, **changes):
    doc = json.loads(path.read_text())
    doc.update(changes)
    path.write_text(json.dumps(doc))


def test_newer_schema_is_refused_with_an_upgrade_message(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    _edit(path, schema_version=3, gfem_version="9.0.0")
    with pytest.raises(ValueError, match=r"schema_version 3.*gfem 9\.0\.0.*upgrade"):
        gfem.Job.from_json(path)


def test_older_schema_is_migrated(model, tmp_path, monkeypatch):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    codec = __import__("gfem._codec", fromlist=["_"])

    def v2_to_v3(doc):
        doc["job"]["options"]["order"] = 1
        return doc

    monkeypatch.setattr(codec, "SCHEMA_VERSION", 3)
    monkeypatch.setitem(codec.MIGRATIONS, 2, v2_to_v3)
    assert gfem.Job.from_json(path).options.order == 1


def test_a_schema_1_job_still_loads(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    _edit(path, schema_version=1)
    _assert_same_model(model, gfem.Job.from_json(path).model)


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


def test_principal_and_sheet_participation_round_trip_and_an_old_result_reads_none(
    tmp_path,
):
    from gfem import _codec

    result = gfem.EigenmodeResult(
        omega=jnp.array([1e10 + 1e6j, 2e10 + 3e6j]),
        error=jnp.array([1e-12, 2e-12]),
        bulk_participation={
            "crystal": jnp.array([0.6, 0.5]),
            "air": jnp.array([0.4, 0.5]),
        },
        principal_participation={
            "crystal": jnp.array([[0.1, 0.2, 0.3], [0.05, 0.25, 0.2]])
        },
        sheet_participation={"film": jnp.array([0.3, 0.05])},
    )
    context = _codec.Context(tmp_path)
    encoder = _codec.Encoder(context)
    doc = json.loads(json.dumps(encoder.encode(result)))
    back = _codec.Decoder(context, encoder.arrays).decode(doc)
    assert list(back.principal_participation) == ["crystal"]
    np.testing.assert_array_equal(
        back.principal_participation["crystal"],
        result.principal_participation["crystal"],
    )
    np.testing.assert_array_equal(
        back.sheet_participation["film"], result.sheet_participation["film"]
    )
    (stored,) = _typed_in(doc, "EigenmodeResult")
    del stored["principal_participation"]  # as 0.1.0 wrote it
    del stored["sheet_participation"]
    old = _codec.Decoder(context, encoder.arrays).decode(doc)
    assert old.principal_participation == {}
    assert old.sheet_participation == {}
    np.testing.assert_array_equal(old.omega, result.omega)


def _tagged(node, seen=None):
    """The first object of each ``$type`` tag in a JSON form, by tag."""
    seen = {} if seen is None else seen
    if isinstance(node, dict):
        if "$type" in node:
            seen.setdefault(node["$type"], node)
        for value in node.values():
            _tagged(value, seen)
    elif isinstance(node, list):
        for value in node:
            _tagged(value, seen)
    return seen


def _with_an_unknown_key(source, tag, target, **head):
    """``source`` rewritten at ``target``, one object of ``tag`` given a new key."""
    doc = json.loads(source.read_text()) | head
    _tagged(doc)[tag]["field_of_a_newer_gfem"] = 1
    target.write_text(json.dumps(doc))
    return target


def _driven_jobs(tmp_path):
    """Driven jobs that hold, between them, every kind of model entry."""
    edges = tuple(
        np.linspace(0, size, n + 1)
        for size, n in zip((10.0, 7.0, 1.0), (4, 3, 1), strict=True)
    )
    cell = gfem.Mesh.from_gmsh(
        sm.write_msh22(tmp_path / "cell.msh", sm.periodic_cell(edges)),
        length_unit="mm",
    )
    eps_r = jnp.linspace(2.0, 4.0, len(cell.elements("cell")))
    periodic = gfem.Model(
        cell,
        materials={"cell": gfem.MaterialField(eps_r=eps_r)},
        boundaries=[gfem.Periodic(("x_min", "x_max")), gfem.PEC("y_min")],
        ports=[gfem.LumpedPort("p1", "z_max", "+y")],
    )
    plate = gfem.Mesh.from_gmsh(
        sm.write_msh22(tmp_path / "stack.msh", sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))
    )
    guided = gfem.Model(
        plate,
        materials={"oxide": gfem.Material(eps_r=3.9)},
        ports=[gfem.WavePort("w1", "oxide__nitride")],
        excluded=(gfem.enclosed_by("oxide__nitride", "top", "side"),),
    )
    problem = gfem.Driven(frequency=(1e9,))
    return [
        gfem.Job(model, problem).to_json(tmp_path / f"{name}.json")
        for name, model in (("periodic", periodic), ("guided", guided))
    ]


def test_a_plain_wave_port_is_written_without_the_evanescent_opt_in(tmp_path):
    # files written without the opt-in stay readable by gfem that lacks it;
    # a port that opts in writes it and reads back
    guided = _driven_jobs(tmp_path)[1]
    assert '"allow_evanescent"' not in guided.read_text(encoding="utf-8")
    mesh = box_mesh(
        tmp_path / "box.msh",
        lines([0.0, 1.0], 0.2),
        lines([0.0, 1.0], 0.2),
        lambda *_: "fill",
    )
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=1.0)},
        boundaries=[gfem.PEC("side")],
        ports=[
            gfem.WavePort("p", "left", allow_evanescent=True),
            gfem.WavePort("q", "right"),
        ],
    )
    job = gfem.Job(model, gfem.PortMode("p", 1e14, polarization="Ez"))
    path = job.to_json(tmp_path / "opted.json")
    assert '"allow_evanescent"' in path.read_text(encoding="utf-8")
    first, second = gfem.Job.from_json(path).model.ports
    assert (first.allow_evanescent, second.allow_evanescent) == (True, False)


def test_an_unknown_key_is_refused_by_name_on_every_class(tmp_path):
    seen = set()
    for source in _driven_jobs(tmp_path):
        for tag in _tagged(json.loads(source.read_text())):
            seen.add(tag)
            path = _with_an_unknown_key(source, tag, source.with_stem("changed"))
            with pytest.raises(ValueError) as err:
                gfem.Job.from_json(path)
            message = str(err.value)
            assert message.startswith(f"{path}: {tag} has no 'field_of_a_newer_gfem'")
    assert seen >= {
        "Job",
        "Model",
        "Mesh",
        "MaterialField",
        "Material",
        "Periodic",
        "PEC",
        "LumpedPort",
        "WavePort",
        "EnclosedBy",
        "Driven",
        "SolverOptions",
    }


def test_an_unknown_key_says_which_gfem_wrote_the_file(model, tmp_path):
    source = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    path = tmp_path / "changed.json"
    _with_an_unknown_key(source, "SolverOptions", path, gfem_version="99.0")
    with pytest.raises(ValueError) as err:
        gfem.Job.from_json(path)
    assert str(err.value) == (
        f"{path}: SolverOptions has no 'field_of_a_newer_gfem' in this gfem "
        f"({gfem.__version__}). The file was written by gfem 99.0: upgrade gfem "
        "(pip install -U gfem)."
    )
    _with_an_unknown_key(source, "SolverOptions", path)
    with pytest.raises(ValueError) as err:
        gfem.Job.from_json(path)
    assert str(err.value) == (
        f"{path}: SolverOptions has no 'field_of_a_newer_gfem' in this gfem "
        f"({gfem.__version__}): remove the entry or check its spelling."
    )


def test_several_unknown_keys_are_listed_sorted(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    doc["job"]["model"]["mesh"] |= {"zeta": 1, "alpha": 2}
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Mesh has no 'alpha', 'zeta' in this gfem"):
        gfem.Job.from_json(path)


def test_an_unknown_key_in_a_result_bundle_is_refused_by_name(model, tmp_path):
    out = tmp_path / "results"
    gfem.Job(model, PROBLEM, gfem.SolverOptions(order=1)).run(out)
    path = out / "result.json"
    _with_an_unknown_key(path, "ElectrostaticResult", path, gfem_version="99.0")
    with pytest.raises(ValueError) as err:
        gfem.load_result(out)
    message = str(err.value)
    assert message.startswith(
        f"{path}: ElectrostaticResult has no 'field_of_a_newer_gfem'"
    )
    assert "written by gfem 99.0: upgrade gfem" in message
    _with_an_unknown_key(
        path, "ElectrostaticResult", path, gfem_version=gfem.__version__
    )
    with pytest.raises(ValueError, match="check its spelling"):
        gfem.load_result(out)


def _with_a_value(source, tag, field, value, target, **head):
    """``source`` rewritten at ``target``, ``field`` of one ``tag`` object set."""
    doc = json.loads(source.read_text()) | head
    _tagged(doc)[tag][field] = value
    target.write_text(json.dumps(doc))
    return target


def _newer_note(path):
    return (
        f"{path} was written by gfem 99.0; this is gfem {gfem.__version__}. "
        "If the refused value is one gfem 99.0 takes, install that version or "
        "newer (pip install -U gfem)."
    )


def test_a_refused_value_names_both_gfem_versions(model, tmp_path):
    source = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    path = tmp_path / "changed.json"
    _with_a_value(
        source, "SolverOptions", "linear_solver", "amg_cg", path, gfem_version="99.0"
    )
    with pytest.raises(ValueError, match="linear_solver must be one of") as err:
        gfem.Job.from_json(path)
    assert str(err.value).startswith("SolverOptions.linear_solver must be one of")
    assert err.value.__notes__ == [_newer_note(path)]
    _with_a_value(
        source,
        "SolverOptions",
        "linear_solver",
        "amg_cg",
        path,
        gfem_version=gfem.__version__,
    )
    with pytest.raises(ValueError, match="linear_solver must be one of") as err:
        gfem.Job.from_json(path)
    assert not getattr(err.value, "__notes__", [])


def test_a_refused_problem_value_is_noted_too(model, tmp_path):
    source = gfem.Job(model, gfem.Driven(frequency=1e9)).to_json(tmp_path / "job.json")
    path = tmp_path / "changed.json"
    _with_a_value(source, "Driven", "sweep", "newer", path, gfem_version="99.0")
    with pytest.raises(ValueError) as err:
        gfem.Job.from_json(path)
    assert err.value.__notes__ == [_newer_note(path)]


def test_a_refused_value_in_a_result_bundle_is_noted(model, tmp_path):
    out = tmp_path / "results"
    gfem.Job(model, PROBLEM, gfem.SolverOptions(order=1)).run(out)
    path = out / "result.json"
    _with_a_value(path, "Mesh", "length_unit", "furlong", path, gfem_version="99.0")
    with pytest.raises(ValueError, match="Unknown length_unit") as err:
        gfem.load_result(out)
    assert err.value.__notes__ == [_newer_note(path)]


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
    assert doc["schema_version"] == 2
    assert doc["gfem_version"] == gfem.__version__
    assert doc["run"]["device"] == "cpu"
    assert gfem.load_result(tmp_path / "results" / "result.json").terminals == (
        "top",
        "mid",
    )


def test_a_cached_run_records_the_hit_and_writes_the_same_bundle(model, tmp_path):
    job = gfem.Job(model, PROBLEM, gfem.SolverOptions(order=1))
    cache = tmp_path / "cache"
    job.run(tmp_path / "first", cache=cache)
    first = json.loads((tmp_path / "first" / "result.json").read_text())["run"]
    assert first["cache"]["status"] == "miss"
    assert len(first["cache"]["key"]) == 64
    assert "solved" not in first["cache"]

    job.run(tmp_path / "second", cache=cache, device="cpu")
    second = json.loads((tmp_path / "second" / "result.json").read_text())["run"]
    assert second["cache"]["status"] == "hit"
    assert second["cache"]["key"] == first["cache"]["key"]
    assert second["cache"]["solved"] == {
        "device": "cpu",
        "ranks": 1,
        "wall_time_s": second["cache"]["solved"]["wall_time_s"],
    }
    assert second["cache"]["solved"]["wall_time_s"] > 0
    a = gfem.load_result(tmp_path / "first")
    b = gfem.load_result(tmp_path / "second")
    np.testing.assert_array_equal(b.capacitance, a.capacitance)
    for region, energy in a.energy.items():
        np.testing.assert_array_equal(b.energy[region], energy)


def test_a_run_without_a_cache_records_no_cache_item(model, tmp_path):
    gfem.Job(model, PROBLEM, gfem.SolverOptions(order=1)).run(tmp_path / "out")
    run = json.loads((tmp_path / "out" / "result.json").read_text())["run"]
    assert "cache" not in run


def test_load_result_refuses_a_job_file(model, tmp_path):
    path = gfem.Job(model, PROBLEM).to_json(tmp_path / "job.json")
    with pytest.raises(ValueError, match="gfem result file"):
        gfem.load_result(path)


def test_load_result_needs_a_result(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"result\.json"):
        gfem.load_result(tmp_path)


# A driven result.json as gfem wrote it at 9f004a7, before SweepInfo had
# iterations and preconditioner_reused (two frequencies, both ports excited).
DRIVEN_RESULT_9F004A7 = (
    '{"format":"gfem.result","schema_version":2,"gfem_version":"0.1.0",'
    '"arrays":"result.npz","run":{"device":"cpu","device_override":null,'
    '"options":{"$type":"SolverOptions","order":2,"linear_solver":"direct",'
    '"tolerance":1e-06,"device":"cpu","threads":null,"adaptive_tolerance":0.001,'
    '"adaptive_max_samples":20,"adaptive_memory":2,"port_mode_tolerance":1e-06,'
    '"retain_memory":null,"adjoint_tolerance":null,"eigen_tolerance":1e-10,'
    '"cluster_rtol":1e-06,"max_refine_iterations":50,"frequency_workers":1,'
    '"eigen_solver":"auto","petsc_options":{}},"ranks":1,"wall_time_s":0.75},'
    '"result":{"$type":"DrivenResult","s":{"$type":"NetworkParameters",'
    '"values":{"$array":".s.values"},"z_ref":{"$array":".s.z_ref"},'
    '"frequency":[4000000000.0,5000000000.0],"ports":["p1","p2"],'
    '"excited":["p1","p2"]},"e_fields":[],"kept":[],"sweep":{"$type":"SweepInfo",'
    '"kind":"uniform","solved":[true,true],'
    '"sampled_frequencies":[4000000000.0,5000000000.0],'
    '"sample_errors":[{"$float":"inf"},{"$float":"inf"}],"converged":true,'
    '"tolerance":null,"basis_size":0},"port_modes":{},"offsets":[]}}'
)


def test_a_driven_result_saved_before_the_solve_record_loads_without_it(tmp_path):
    (tmp_path / "result.json").write_text(DRIVEN_RESULT_9F004A7)
    arrays = {".s.values": np.zeros((2, 2, 2), complex), ".s.z_ref": np.ones((2, 2))}
    np.savez(tmp_path / "result.npz", **arrays)
    sweep = gfem.load_result(tmp_path).sweep
    assert sweep.solved == (True, True)
    assert sweep.iterations is None
    assert sweep.preconditioner_reused is None


def test_a_driven_result_saved_before_2d_models_reads_as_a_3d_result(tmp_path):
    (tmp_path / "result.json").write_text(DRIVEN_RESULT_9F004A7)
    arrays = {".s.values": np.zeros((2, 2, 2), complex), ".s.z_ref": np.ones((2, 2))}
    np.savez(tmp_path / "result.npz", **arrays)
    result = gfem.load_result(tmp_path)
    assert result.h_fields == ()
    assert result.polarization is None


# --- a sweep that stops part-way ----------------------------------------------------

SWEEP = (2e9, 3e9, 4e9, 5e9, 6e9)


@pytest.fixture
def sweep_job(tmp_path):
    """A uniform sweep of a two-port strip line on the iterative solver."""
    path = sm.write_msh22(
        tmp_path / "line.msh", sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1))
    )
    line = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    return gfem.Job(
        line,
        gfem.Driven(frequency=SWEEP),
        gfem.SolverOptions(linear_solver="bddc_gmres"),
    )


def _stopped(job, out):
    with pytest.raises(gfem.LinearSolverConvergenceError) as caught:
        job.run(out)
    return caught.value


def test_a_sweep_that_stops_leaves_a_partial_bundle_and_a_note(
    sweep_job, tmp_path, monkeypatch
):
    out = tmp_path / "results"
    cannot_converge_at(monkeypatch, (SWEEP[2],))
    err = _stopped(sweep_job, out)

    assert sorted(p.name for p in out.iterdir()) == ["partial.json", "partial.npz"]
    partial = gfem.load_result(out / "partial.json")
    assert partial.s.frequency == SWEEP[:2]
    np.testing.assert_array_equal(partial.s.values, err.result.s.values)
    run = json.loads((out / "partial.json").read_text())["run"]
    assert run["unsolved"] == list(SWEEP[2:])
    assert run["error"] == f"LinearSolverConvergenceError: {err}"
    assert run["device"] == "cpu"
    assert run["ranks"] == 1
    assert run["options"]["linear_solver"] == "bddc_gmres"
    (note,) = err.__notes__
    assert note == (
        f"The partial result (2 of 5 frequencies) is in {out / 'partial.json'}: "
        f'read it with gfem.load_result("{out / "partial.json"}"). Its "run" '
        "entry lists the unsolved frequencies."
    )

    with pytest.raises(FileNotFoundError) as missing:
        gfem.load_result(out)
    assert str(missing.value) == (
        f"No result.json in {out}, only a partial result: the run stopped at an "
        f'unsolved frequency. Read it with gfem.load_result("{out / "partial.json"}").'
    )


def test_a_cached_sweep_that_stops_records_a_miss_and_stores_nothing(
    sweep_job, tmp_path, monkeypatch
):
    out = tmp_path / "results"
    cache = tmp_path / "cache"
    cannot_converge_at(monkeypatch, (SWEEP[2],))
    with pytest.raises(gfem.LinearSolverConvergenceError):
        sweep_job.run(out, cache=cache)
    run = json.loads((out / "partial.json").read_text())["run"]
    assert run["cache"]["status"] == "miss"
    assert len(run["cache"]["key"]) == 64
    assert "solved" not in run["cache"]
    with pytest.raises(gfem.LinearSolverConvergenceError):
        sweep_job.run(out, cache=cache)


def test_a_sweep_that_solves_nothing_writes_nothing(sweep_job, tmp_path, monkeypatch):
    out = tmp_path / "results"
    cannot_converge_at(monkeypatch, (SWEEP[0],))
    err = _stopped(sweep_job, out)
    assert err.result is None
    assert not out.exists()
    assert not hasattr(err, "__notes__")


def test_a_complete_run_removes_a_stale_partial_bundle(sweep_job, tmp_path):
    out = tmp_path / "results"
    with pytest.MonkeyPatch.context() as patch:
        cannot_converge_at(patch, (SWEEP[2],))
        _stopped(sweep_job, out)
    result = sweep_job.run(out)
    assert sorted(p.name for p in out.iterdir()) == ["result.json", "result.npz"]
    assert gfem.load_result(out).s.frequency == result.s.frequency == SWEEP


def test_a_partial_run_keeps_an_older_result_and_its_note_says_so(
    sweep_job, tmp_path, monkeypatch
):
    out = tmp_path / "results"
    sweep_job.run(out)
    complete = (out / "result.json").read_bytes()
    cannot_converge_at(monkeypatch, (SWEEP[2],))
    err = _stopped(sweep_job, out)
    assert (out / "result.json").read_bytes() == complete
    assert (out / "partial.json").is_file()
    (note,) = err.__notes__
    assert note.endswith(
        f"{out / 'result.json'} is the complete result of an earlier run."
    )
    assert gfem.load_result(out).s.frequency == SWEEP


def test_a_partial_bundle_that_cannot_be_written_leaves_the_solver_error(
    sweep_job, tmp_path, monkeypatch
):
    blocked = tmp_path / "a_file"
    blocked.write_text("")
    out = blocked / "results"
    cannot_converge_at(monkeypatch, (SWEEP[2],))
    err = _stopped(sweep_job, out)
    assert err.result.s.frequency == SWEEP[:2]
    (note,) = err.__notes__
    # NotADirectoryError on POSIX; Windows names another OSError
    assert note.startswith(f"The partial result could not be written to {out}: ")
    assert "Error: " in note


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
    names = ("frequency", "gamma", "Z_PV", "residual", "solved")
    for name in (*names, "n_g", "beta_2", "beta_3"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
    points = np.asarray(mesh.nodes)[mesh.triangles[mesh.elements("port1")][:, 0]]
    np.testing.assert_array_equal(loaded.e_field.at(points), result.e_field.at(points))


def test_an_old_port_mode_bundle_reads_no_dispersion(tmp_path):
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    job = gfem.Job(model, gfem.PortMode("p1", [8e9, 10e9]), gfem.SolverOptions(order=1))
    result = job.run(out=tmp_path / "results")
    path = tmp_path / "results" / "result.json"
    doc = json.loads(path.read_text())
    (stored,) = _typed_in(doc, "PortModeResult")
    for name in ("n_g", "beta_2", "beta_3"):  # as a gfem before them wrote it
        del stored[name]
    path.write_text(json.dumps(doc))
    old = gfem.load_result(tmp_path / "results")
    assert (old.n_g, old.beta_2, old.beta_3, old.D) == (None, None, None, None)
    np.testing.assert_array_equal(old.gamma, result.gamma)


def test_a_port_section_with_a_law_round_trips_with_its_field(tmp_path):
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    law = gfem.PermittivityLaw(
        eps_inf=2.0,
        terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=30e9, damping=1e9)],
    )
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material(eps_r=law)},
        boundaries=[gfem.Absorbing("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    job = gfem.Job(model, gfem.PortMode("p1", [8e9, 10e9]), gfem.SolverOptions(order=1))
    result = job.run(out=tmp_path / "results")
    loaded = gfem.load_result(tmp_path / "results")
    section, back = result.e_field._section, loaded.e_field._section
    assert back.laws == section.laws == (("air", law),)
    np.testing.assert_array_equal(back.law_index, section.law_index)
    np.testing.assert_array_equal(back.absorbing_law, section.absorbing_law)
    points = np.asarray(mesh.nodes)[mesh.triangles[mesh.elements("port1")][:, 0]]
    np.testing.assert_array_equal(loaded.e_field.at(points), result.e_field.at(points))


def _of_type(node, tag):
    if isinstance(node, dict):
        if node.get("$type") == tag:
            yield node
        for v in node.values():
            yield from _of_type(v, tag)
    elif isinstance(node, list):
        for v in node:
            yield from _of_type(v, tag)


def test_an_old_port_section_reads_its_scalar_conductivity_and_admittance(tmp_path):
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material(eps_r=2.0, conductivity=0.01)},
        boundaries=[gfem.Absorbing("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    job = gfem.Job(model, gfem.PortMode("p1", 10e9), gfem.SolverOptions(order=1))
    result = job.run(out=tmp_path / "results")
    new = result.e_field._section
    assert len(new.absorbing_edges) > 0

    # rewrite the stored section as 0.1.0 wrote it: one conductivity per
    # triangle and one admittance per absorbing edge
    path = tmp_path / "results" / "result.json"
    doc = json.loads(path.read_text())
    with np.load(path.with_name(doc["arrays"])) as npz:
        arrays = {k: npz[k] for k in npz.files}
    (section,) = _of_type(doc, "PortSection")
    tt, nn = section.pop("sigma_tt")["$array"], section.pop("sigma_nn")["$array"]
    old = nn.removesuffix("sigma_nn") + "sigma"
    arrays[old] = arrays.pop(nn)
    del arrays[tt]
    section["sigma"] = {"$array": old, "numpy": True}
    admittance = section["absorbing_admittance"]["$array"]
    arrays[admittance] = arrays[admittance][:, 0]
    path.write_text(json.dumps(doc))
    with path.with_name(doc["arrays"]).open("wb") as f:
        np.savez(f, **arrays)

    back = gfem.load_result(tmp_path / "results").e_field._section
    np.testing.assert_array_equal(back.sigma_nn, new.sigma_nn)
    # sigma I_2; the new section's block carries the frame's round-off
    np.testing.assert_array_equal(
        back.sigma_tt, back.sigma_nn[:, None, None] * np.eye(2)
    )
    np.testing.assert_allclose(back.sigma_tt, new.sigma_tt, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(back.absorbing_admittance, new.absorbing_admittance)


def test_a_shared_port_surface_round_trips_through_a_bundle(tmp_path):
    # two modes per end of a parallel plate (W = 40, D = 10, L = 20 um)
    mesh = gfem.Mesh.from_gmsh(DATA / "parallel_plate.msh", length_unit="um")
    w, d, length = 40e-6, 10e-6, 20e-6

    def path(x, z):
        return ((x, 0.0, z), (x, d, z))

    model = gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("plates")],
        ports=[
            gfem.WavePort("p1_TEM", "port1", 1, 1e-6, path(w / 2, 0.0)),
            gfem.WavePort("p1_m2", "port1", 2, 2e-6, path(w / 4, 0.0)),
            gfem.WavePort("p2_TEM", "port2", 1),
            gfem.WavePort("p2_m2", "port2", 2, voltage_path=path(w / 4, length)),
        ],
    )
    job = gfem.Job(model, gfem.Driven((2.0e12, 2.5e12)), gfem.SolverOptions(order=2))
    job = gfem.Job.from_json(job.to_json(tmp_path / "job.json"))
    assert job.model.ports == model.ports
    result = job.run(out=tmp_path / "out")
    loaded = gfem.load_result(tmp_path / "out")
    np.testing.assert_array_equal(loaded.s.values, result.s.values)
    np.testing.assert_array_equal(loaded.s.z_ref, result.s.z_ref)
    assert loaded.offsets == result.offsets
    assert dict(loaded.offsets) == {
        "p1_TEM": 1e-6,
        "p1_m2": 2e-6,
        "p2_TEM": 0.0,
        "p2_m2": 0.0,
    }
    assert set(loaded.port_modes) == set(model.port_names)
    for name, mode in result.port_modes.items():
        for field in ("frequency", "gamma", "Z_PV", "residual", "solved"):
            np.testing.assert_array_equal(
                getattr(loaded.port_modes[name], field), getattr(mode, field)
            )
        assert loaded.port_modes[name].n_g is None
    # a driven result's port modes are written as before they had dispersion
    assert '"n_g"' not in (tmp_path / "out" / "result.json").read_text()


# --- thermal and DC conduction ------------------------------------------------------

#: The slab section's layers (name, thickness in um), up y.
SLAB = (("si", 1.0), ("oxide", 1.0), ("metal", 0.2))
SLAB_MATERIALS = {
    "si": gfem.Material(
        eps_r=11.7, thermal_conductivity=148.0, mass_density=2330.0, specific_heat=711.0
    ),
    "oxide": gfem.Material(
        eps_r=2.1,
        thermal_conductivity=1.4,
        mass_density=2203.0,
        specific_heat=709.0,
        thermo_optic_coefficient=1e-5,
    ),
    "metal": gfem.Material(
        conductivity=1e6,
        thermal_conductivity=28.0,
        mass_density=5240.0,
        specific_heat=598.0,
    ),
}
SLAB_OPTIONS = gfem.SolverOptions(order=1, linear_solver="direct")


@pytest.fixture(scope="module")
def slab_model(tmp_path_factory):
    """A 4 um wide cross-section of SLAB with every thermal boundary type."""
    import thermal_cases as tc

    rects, y = [], 0.0
    for name, t in SLAB:
        rects.append((0.0, y, 4.0, y + t, name, None))
        y += t
    top = y
    edges = [
        (lambda p: abs(p[1]) < 1e-9, "bottom"),
        (lambda p: abs(p[1] - top) < 1e-9, "top"),
        (lambda p: abs(p[0]) < 1e-9 and p[1] < 1.0, "left"),
    ]
    path = tmp_path_factory.mktemp("slab") / "slab.msh"
    mesh = tc.section(path, rects, 0.6, edges, thickness=3e-7)
    return gfem.Model(
        mesh,
        materials=SLAB_MATERIALS,
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=293.15),
            gfem.Convection("top", coefficient=10.0, ambient=300.0),
            gfem.HeatFlux("left", flux=50.0),
        ],
        excluded={"em": ("metal",), "dc_conduction": ("si",)},
    )


def _thermal_problems():
    heat = gfem.HeatSource.uniform("metal", power_per_length=2.0)
    return {
        "steady_thermal": gfem.SteadyThermal(heat),
        "transient_thermal": gfem.TransientThermal(
            [heat, gfem.HeatSource.uniform(("si", "oxide"), power_density=1e9)],
            duration=4e-6,
            steps=4,
            waveform=[(0.0, 1.0), (2e-6, 0.5)],
            initial_temperature=295.0,
            traces={
                "core": gfem.Trace.mean("oxide"),
                "spot": gfem.Trace.point((2e-6, 1.5e-6, 1.5e-7)),
            },
            keep_fields=[1e-6, 3e-6],
        ),
        "dc_conduction": gfem.DCConduction(terminals={"a": "front"}, ground="back"),
    }


def _assert_same_heat(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert type(y) is gfem.HeatSource
        assert repr(x) == repr(y)
        for name in ("regions", "_kind", "_quantity", "_total", "_thickness"):
            assert getattr(x, name) == getattr(y, name)
        for name in ("_amount", "_density"):
            u, v = getattr(x, name), getattr(y, name)
            assert (u is None) == (v is None)
            if u is not None:
                np.testing.assert_array_equal(u, v)


def _assert_same_problem(a, b):
    assert type(a) is type(b)
    assert repr(a) == repr(b)
    if isinstance(a, gfem.DCConduction):
        assert a == b
        return
    _assert_same_heat(a.heat, b.heat)
    if isinstance(a, gfem.TransientThermal):
        for name in (
            "duration",
            "steps",
            "initial_temperature",
            "keep_fields",
            "_switches",
        ):
            assert getattr(a, name) == getattr(b, name)
        np.testing.assert_array_equal(a._scales, b._scales)
        assert {k: repr(t) for k, t in a.traces.items()} == {
            k: repr(t) for k, t in b.traces.items()
        }


@pytest.mark.parametrize("name", list(_thermal_problems()))
def test_thermal_and_dc_conduction_jobs_round_trip_with_section_thickness_and_mapping(
    slab_model, name, tmp_path
):
    problem = _thermal_problems()[name]
    options = gfem.SolverOptions(order=1, time_scheme="theta", theta=0.5)
    job = _round_trip(gfem.Job(slab_model, problem, options), tmp_path)
    _assert_same_problem(problem, job.problem)
    assert job.options == options
    assert job.model.mesh.section_thickness == 3e-7
    assert job.model.boundaries == slab_model.boundaries
    for physics in ("em", "thermal", "dc_conduction"):
        assert job.model.excluded_from(physics) == slab_model.excluded_from(physics)
    doc = json.loads((tmp_path / "job.json").read_text())
    assert doc["job"]["model"]["excluded"]["$type"] == "Excluded"


def test_a_thermal_job_refuses_an_unknown_key_on_every_new_class(slab_model, tmp_path):
    seen = set()
    for name, problem in _thermal_problems().items():
        source = gfem.Job(slab_model, problem).to_json(tmp_path / f"{name}.json")
        for tag in _tagged(json.loads(source.read_text())):
            seen.add(tag)
            path = _with_an_unknown_key(source, tag, source.with_stem("changed"))
            with pytest.raises(ValueError, match=f"{tag} has no 'field_of_a_newer"):
                gfem.Job.from_json(path)
    assert seen >= {
        "SteadyThermal",
        "TransientThermal",
        "DCConduction",
        "HeatSource",
        "Trace",
        "FixedTemperature",
        "Convection",
        "HeatFlux",
        "Excluded",
    }


def test_a_joule_heat_source_stores_its_array_by_leaf_path(slab_model, tmp_path):
    dc = gfem.solve(
        slab_model, _thermal_problems()["dc_conduction"], options=SLAB_OPTIONS
    )
    joule = gfem.joule_heating(dc, power_per_length=2.0)
    uniform = gfem.HeatSource.uniform("si", power_density=1e8)
    problem = gfem.SteadyThermal([joule, uniform])
    job = gfem.Job(slab_model, problem, SLAB_OPTIONS)
    path = job.to_json(tmp_path / "job.json")
    with np.load(path.with_suffix(".npz")) as npz:
        stored = dict(npz)
    leaves = {
        jax.tree_util.keystr(p): leaf
        for p, leaf in jax.tree_util.tree_flatten_with_path(job)[0]
    }
    key = ".problem.heat[0]._density"
    assert key in leaves
    np.testing.assert_array_equal(stored[key], joule._density)
    back = gfem.Job.from_json(path)
    _assert_same_problem(problem, back.problem)
    # the stored source heats as the original does
    expected = gfem.solve(slab_model, problem, options=SLAB_OPTIONS)
    result = back.run(out=tmp_path / "out")
    np.testing.assert_allclose(result.power, expected.power, rtol=1e-12)


@pytest.mark.parametrize("name", list(_thermal_problems()))
def test_thermal_and_dc_conduction_results_round_trip(slab_model, name, tmp_path):
    problem = _thermal_problems()[name]
    result = gfem.Job(slab_model, problem, SLAB_OPTIONS).run(out=tmp_path / "out")
    loaded = gfem.load_result(tmp_path / "out")
    assert type(loaded) is type(result)
    assert repr(loaded) == repr(result)
    # the result's leaves and their paths are those of the solve
    a, _ = jax.tree_util.tree_flatten_with_path(result)
    b, _ = jax.tree_util.tree_flatten_with_path(loaded)
    assert [p for p, _ in a] == [p for p, _ in b]
    for (_, x), (_, y) in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    if name == "dc_conduction":
        assert loaded.terminals == result.terminals
        np.testing.assert_array_equal(
            loaded.resistance_per_length, result.resistance_per_length
        )
        assert loaded.potential["a"].mesh.section_thickness == 3e-7
    elif name == "steady_thermal":
        np.testing.assert_array_equal(loaded.power_per_length, result.power_per_length)
        assert loaded.temperature.mesh.section_thickness == 3e-7
    else:
        assert loaded.kept_times == result.kept_times
        assert loaded.switch_times == result.switch_times
        assert list(loaded.traces) == ["core", "spot"]
        for t in loaded.kept_times:
            np.testing.assert_array_equal(
                loaded.temperature(t).coefficients, result.temperature(t).coefficients
            )


def test_thermo_optic_shift_and_route_traces_round_trip_in_a_thermal_job(tmp_path):
    import scipy.constants
    import thermal_cases as tc

    n_si, n_ox = 3.4777, 1.444
    rects = [
        (-1.75, -1.0, 1.75, 1.8, "clad", None),
        (-0.25, 0.0, 0.25, 0.22, "core", 0.05),
    ]
    bottom = [(lambda p: abs(p[1] + 1.0) < 1e-9, "bottom")]
    mesh = tc.section(tmp_path / "section.msh", rects, 0.4, bottom)
    thermal = {"mass_density": 2200.0, "specific_heat": 700.0}
    model = gfem.Model(
        mesh,
        materials={
            "clad": gfem.Material(
                eps_r=n_ox**2,
                thermal_conductivity=1.38,
                thermo_optic_coefficient=1e-5,
                **thermal,
            ),
            "core": gfem.Material(
                eps_r=n_si**2,
                thermal_conductivity=90.0,
                thermo_optic_coefficient=1.86e-4,
                **thermal,
            ),
        },
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
        ports=[gfem.WavePort("p", "front")],
    )
    frequency = scipy.constants.c / 1.55e-6
    shift = gfem.thermo_optic_shift(model, None, port="p", frequency=frequency)
    line = np.array([(4.0, 0.5, 1.5), (4.0, 3.5, 1.5)]) * 1e-6
    route = gfem.route_phase(
        model, None, routes={"line": line}, port="p", frequency=frequency
    )
    heat = gfem.HeatSource.uniform("core", power_per_length=1.0)
    problem = gfem.TransientThermal(
        heat,
        duration=1e-6,
        steps=2,
        traces={"shift": shift.trace(), "route": route.trace("line")},
    )
    job = _round_trip(gfem.Job(model, problem), tmp_path)
    _assert_same_problem(problem, job.problem)
    a, b = problem.traces["route"]._sampler, job.problem.traces["route"]._sampler
    assert b.name == a.name
    for name in ("arc_length", "points", "left", "up", "trapezoid", "corners"):
        np.testing.assert_array_equal(getattr(b, name), getattr(a, name))
    for key in ("shift", "route"):
        np.testing.assert_array_equal(
            job.problem.traces[key]._weights, problem.traces[key]._weights
        )
    with np.load(tmp_path / "job.npz") as npz:
        assert ".problem.traces['shift']._weights" in npz.files
        assert ".problem.traces['route']._weights" in npz.files
    # the read shift trace samples a rise as the written one does
    steady = gfem.solve(model, gfem.SteadyThermal(heat))
    np.testing.assert_allclose(
        job.problem.traces["shift"]._sample(None, steady.rise),
        problem.traces["shift"]._sample(None, steady.rise),
        rtol=1e-13,  # the same weights; the sum's order is not fixed
    )


def test_a_field_writes_its_scattered_flag_only_when_set(model):
    from gfem.job import _field_args

    field = gfem.Field(
        coefficients=np.zeros(3, complex),
        mesh=model.mesh,
        quantity="E",
        unit="V/m",
        space="hcurl",
        order=2,
        support=(),
        omega=1.0,
    )
    # a field of 0.2.0 has no such key: it must encode as it did
    assert "_scattered" not in _field_args(field, None)
    assert _field_args(dataclasses.replace(field, _scattered=True), None)["_scattered"]
