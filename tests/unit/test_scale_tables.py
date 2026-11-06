"""`tools/scale_tables.py`: run records to the scaling page's committed snippets."""

import json
from pathlib import Path

import pytest
import scale_tables

COMMIT = "9a57680c0db3f8bbf179aff83db9a9e0805efcc3"


def _record(name, *, nodes, seconds, dofs=2_800_000, **extra):
    """A finished run on 28 ranks per node at 4 threads, the fields a table reads."""
    ranks = 28 * nodes
    return {
        "name": name,
        "case": f"/scratch/meshes/{extra.pop('mesh', 'm_void')}",
        "solver": extra.pop("solver", "bddc_gmres"),
        "options": {"tolerance": 1e-8, "order": 2},
        "frequencies_ghz": [50.0],
        "commit": extra.pop("commit", COMMIT),
        "job": extra.pop("job", "100"),
        "node_count": nodes,
        "ranks": ranks,
        "threads_read": 4,
        "dofs": {
            "owned_sum": dofs,
            "owned_mean": dofs / ranks,
            "owned_min": dofs / ranks - 1000,
            "owned_max": dofs / ranks + 1000,
        },
        "phases": {
            "solve": {"max_s": seconds},
            "gmres": {"max_s": seconds / 2},
            "coarse_apply": {"max_s": seconds / 4},
        },
        "krylov": {"vector_share": 0.1, "reduction_wait_s": 1.0},
        "coarse": [{"dofs": 9000, "ranks": 2}],
        "iterations": {"iterations": [[40, 41]]},
        "memory": {"peak_rss_largest_node_gib": 50.0},
        "ranks_bitwise": True,
        "checks": extra.pop("checks", []),
        **extra,
    }


def _stub(name, *, nodes, state, job, **extra):
    """A run that wrote nothing itself, as the job script closes it."""
    return {
        "name": name,
        "failed": True,
        "mode": "failed",
        "state": state,
        "case": "/scratch/meshes/m_void",
        "solver": "bddc_gmres",
        "node_count": nodes,
        "ranks": 28 * nodes,
        "threads_read": 4,
        "commit": COMMIT,
        "job": job,
        "last_phase": "gmres",
        **extra,
    }


def _write(directory: Path, *records: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for r in records:
        (directory / f"{r['name']}.json").write_text(json.dumps(r))
    return directory


LADDER = {
    "strong": {
        "kind": "ladder",
        "records": ["s_1_r*", "s_2_r*"],
        "base": 1,
    }
}


@pytest.fixture
def ladder(tmp_path):
    return _write(
        tmp_path / "records",
        _record("s_1_r1", nodes=1, seconds=100.0, job="11"),
        _record("s_1_r2", nodes=1, seconds=120.0, job="11"),
        _record("s_1_r3", nodes=1, seconds=110.0, job="11"),
        _record("s_2_r1", nodes=2, seconds=50.0, job="12"),
        _record("s_2_r2", nodes=2, seconds=70.0, job="12"),
        _stub("s_2_r3", nodes=2, state="TIMEOUT", job="12"),
    )


def _table_rows(text: str) -> list[list[str]]:
    rows = [line for line in text.splitlines() if line.startswith("| ")]
    return [[c.strip() for c in row.strip("|").split("|")] for row in rows[1:]]


def test_a_ladder_gives_each_rung_its_median_dof_per_rank_and_tolerance(
    ladder, tmp_path
):
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "out", commit="9a57680", spec=LADDER
    )
    rows = _table_rows(path.read_text())
    one, two = rows[0], rows[1]
    assert one[:2] == ["1", "28 x 4"]
    assert "100 k" in one  # 2.8 M DOF on 28 ranks
    assert "1e-08" in one
    assert "110" in one  # the median of 100, 110, 120
    assert "100-120" in one
    assert two[:2] == ["2", "56 x 4"]
    assert "50 k" in two
    assert "60" in two  # the median of the two that finished
    assert "0.92" in two  # 110 / (60 x 2), against 1 node


def test_a_failed_repeat_is_a_row_of_its_own_with_its_state(ladder, tmp_path):
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "out", commit="9a57680", spec=LADDER
    )
    rows = _table_rows(path.read_text())
    assert len(rows) == 3
    failed = rows[2]
    assert failed[:2] == ["2", "56 x 4"]
    assert "TIMEOUT" in " ".join(failed)
    assert "50 k" in failed  # the mesh's DOF, from the rung's finished runs


def test_a_record_of_another_commit_is_refused(ladder, tmp_path):
    _write(ladder, _record("s_2_r2", nodes=2, seconds=70.0, commit="0" * 40))
    with pytest.raises(SystemExit, match="s_2_r2"):
        scale_tables.write_snippets(
            ladder, tmp_path / "out", commit="9a57680", spec=LADDER
        )


def test_a_missing_record_is_refused(ladder, tmp_path):
    spec = {"strong": {**LADDER["strong"], "records": ["s_1_r*", "s_4_r*"]}}
    with pytest.raises(SystemExit, match="s_4_r"):
        scale_tables.write_snippets(
            ladder, tmp_path / "out", commit="9a57680", spec=spec
        )


def test_a_failed_agreement_gate_is_refused(ladder, tmp_path):
    check = {"gate": 1e-9, "differences": {"s": 2e-9}, "passed": False}
    _write(ladder, _record("s_1_r1", nodes=1, seconds=100.0, checks=[check]))
    with pytest.raises(SystemExit, match="s_1_r1"):
        scale_tables.write_snippets(
            ladder, tmp_path / "out", commit="9a57680", spec=LADDER
        )


def test_the_header_names_the_commit_and_changes_with_any_record(ladder, tmp_path):
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "a", commit="9a57680", spec=LADDER
    )
    first = path.read_text().splitlines()[0]
    assert first.startswith("%")
    assert "9a57680" in first
    _write(ladder, _record("s_1_r1", nodes=1, seconds=100.5, job="11"))
    (again,) = scale_tables.write_snippets(
        ladder, tmp_path / "b", commit="9a57680", spec=LADDER
    )
    assert again.read_text().splitlines()[0] != first


def test_a_superseded_record_is_not_read(ladder, tmp_path):
    old = _record("s_1_r1", nodes=1, seconds=999.0, commit="0" * 40)
    (ladder / "s_1_r1.superseded-1.json").write_text(json.dumps(old))
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "out", commit="9a57680", spec=LADDER
    )
    assert "999" not in path.read_text()


def test_a_table_names_the_machine_and_nodes_its_runs_name(ladder, tmp_path):
    for path in ladder.glob("*.json"):
        record = json.loads(path.read_text())
        record.update(machine="Some Cluster", constraint="highmem")
        path.write_text(json.dumps(record))
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "out", commit="9a57680", spec=LADDER
    )
    where = path.read_text().splitlines()[2]
    assert "Some Cluster" in where
    assert "highmem" in where


def test_what_renders_names_no_commit_and_no_job(ladder, tmp_path):
    for path in ladder.glob("*.json"):
        record = json.loads(path.read_text())
        record.update(machine="Some Cluster", job="48598488")
        path.write_text(json.dumps(record))
    (path,) = scale_tables.write_snippets(
        ladder, tmp_path / "out", commit="9a57680", spec=LADDER
    )
    rendered = "\n".join(
        line for line in path.read_text().splitlines() if not line.startswith("%")
    )
    assert "9a57680" not in rendered
    assert "48598488" not in rendered
    assert "Job" not in rendered
