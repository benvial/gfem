"""The scaling campaign's scripts in `tools/`, on what runs without a machine."""

import importlib.metadata
import json
from pathlib import Path

import pytest
import scale_mesh
import scale_steps
import scale_void

import gfem

#: A cube of two tetrahedra around the shared face (1 2 3): one belongs to the excluded
#: entity 20, and a triangle lies on each.
MSH = """\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
2 7 "shell"
3 8 "air"
3 9 "metal"
$EndPhysicalNames
$Nodes
5
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 0 0 -1
$EndNodes
$Elements
4
1 2 2 7 100 1 2 4
2 2 2 7 101 1 2 5
3 4 2 8 10 1 2 3 4
4 4 2 8 20 1 3 2 5
$EndElements
"""


def test_voiding_removes_the_entity_and_what_only_it_touched(tmp_path: Path) -> None:
    source, target = tmp_path / "in.msh", tmp_path / "out.msh"
    source.write_text(MSH)
    counts = scale_void.void_mesh(source, target, {(8, 20)})
    assert counts == {
        "nd2_dof": 2 * 6 + 2 * 4,
        "tetrahedra": [2, 1],
        "tetrahedra_removed": 1,
        "triangles": [2, 1],
        "nodes": [5, 4],
    }
    mesh = gfem.Mesh.from_gmsh(target, length_unit="um")
    assert len(mesh.tetrahedra) == 1
    assert len(mesh.triangles) == 1
    assert {r.name for r in mesh.regions} >= {"shell", "air"}


def test_voiding_a_case_records_it_so_a_run_excludes_nothing(tmp_path: Path) -> None:
    case = tmp_path / "case"
    case.mkdir()
    (case / "palace.msh").write_text(MSH)
    (case / "config.json").write_text(
        json.dumps(
            {
                "Problem": {"Output": "elsewhere"},
                "Model": {"L0": 1e-6, "Mesh": "palace.msh"},
                "Domains": {
                    "Materials": [{"Attributes": [8], "Permittivity": 2.0}],
                },
                "Boundaries": {
                    "Conductivity": [],
                    "Absorbing": {"Order": 1, "Attributes": [7]},
                },
            }
        )
    )
    out = tmp_path / "void"
    info = scale_void.void_case(case, out)
    assert info["excluded_entities"] == []
    assert info["source"] == "case"
    assert json.loads((out / "build.json").read_text()) == info
    config = json.loads((out / "config.json").read_text())
    assert config["Problem"]["Output"] == "output/palace"


def test_a_mesh_is_built_on_the_pinned_packages_only() -> None:
    scale_mesh.check_pins(scale_mesh.PINS.__getitem__)  # the pins themselves pass
    other = {**scale_mesh.PINS, "gsim": "0.8.0"}
    with pytest.raises(SystemExit, match="another config"):
        scale_mesh.check_pins(other.__getitem__)


def test_a_missing_pinned_package_is_refused() -> None:
    def missing(package: str) -> str:
        raise importlib.metadata.PackageNotFoundError(package)

    with pytest.raises(SystemExit, match="not installed"):
        scale_mesh.check_pins(missing)


def test_the_family_sizes_read_by_name_or_number() -> None:
    assert scale_mesh.refined_size("B2") == 0.024
    assert scale_mesh.refined_size("0.5") == 0.5


@pytest.fixture
def steps(tmp_path: Path) -> Path:
    scale_steps.write(Path("/m"), Path("/ref/r10M_void"), tmp_path)
    return tmp_path


def _lines(steps: Path, name: str) -> list[str]:
    return (steps / f"{name}.txt").read_text().splitlines()


def _plan(steps: Path) -> dict[str, dict[str, str]]:
    lines = (steps / "plan.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, line.split("\t"), strict=True)) for line in lines[1:]]
    return {row["file"]: row for row in rows}


def test_each_plan_line_names_a_step_file(steps: Path) -> None:
    for name, row in _plan(steps).items():
        assert (steps / f"{name}.txt").exists()
        assert int(row["time_min"]) > 0


def test_ms3_is_repeat_one_at_32_nodes_on_the_largest_mesh(steps: Path) -> None:
    lines = _lines(steps, "r2_bddc_32")
    assert lines[0].split()[:6] == [
        "r2_bddc_32_r1",
        "28",
        "4",
        "/m/B2_void_split",
        "bddc_gmres",
        "50",
    ]
    assert lines[1] == "check r2_bddc_32_r1 ref_bddc observe"
    assert sum(line.startswith("r2_bddc_32_r") for line in lines) == 3


def test_rows_run_on_the_nodes_and_qos_of_the_plan(steps: Path) -> None:
    plan = _plan(steps)
    assert {n for n, r in plan.items() if r["row"] == "2"} == {
        f"r2_bddc_{n}" for n in (4, 8, 16, 32)
    }
    assert {plan[f"r4_bddc_{n}"]["nodes"] for n in (1, 2, 4, 8, 16, 32)} == {
        str(n) for n in (1, 2, 4, 8, 16, 32)
    }
    assert plan["r2_bddc_32"]["qos"] == "gp_debug"
    assert plan["r4_bddc_1"]["qos"] == "gp_ehpc"
    assert plan["r3_direct_32"]["time_min"] == "45"


def test_direct_at_47_m_runs_one_repeat_per_job(steps: Path) -> None:
    for k in (1, 2, 3):
        lines = _lines(steps, f"r7_direct_16_r{k}")
        assert len(lines) == 1
        assert f"--repeat {k}" in lines[0]
    assert "/m/B1_void_split" in lines[0]


def test_direct_strong_above_8_nodes_keeps_more_repeats_apart(steps: Path) -> None:
    assert len(_lines(steps, "r5_direct_32")) == 1
    assert len(_lines(steps, "r5_direct_32_more")) == 2
    assert len(_lines(steps, "r5_direct_8")) == 3


def test_a_cross_check_runs_tighter_and_reads_direct(steps: Path) -> None:
    run, check = _lines(steps, "r8_B2_void_split")
    assert "--tol 1e-10" in run
    assert check == "check r8_B2_void_split r3_direct_32 bddc_gmres"
