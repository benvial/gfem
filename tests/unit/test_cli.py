"""The command-line runner: python -m gfem run job.json --out DIR."""

import json
import subprocess
import sys

import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem import _cuda
from gfem.__main__ import main

LAYERS = (("oxide", 0.5), ("nitride", 0.25))
MATERIALS = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}
PROBLEM = gfem.Electrostatic(terminals={"top": "top", "mid": "oxide__nitride"})


@pytest.fixture
def model(tmp_path):
    path = sm.write_msh22(tmp_path / "stack.msh", sm.plate(4.0, 3.0, LAYERS, (1, 1, 1)))
    return gfem.Model(gfem.Mesh.from_gmsh(path), materials=MATERIALS)


def _job(model, tmp_path, **options):
    job = gfem.Job(model, PROBLEM, gfem.SolverOptions(order=1, **options))
    return job.to_json(tmp_path / "job.json")


def test_cli_round_trip_equals_the_in_process_result(model, tmp_path):
    expected = gfem.solve(model, PROBLEM, options=gfem.SolverOptions(order=1))
    job = _job(model, tmp_path)
    out = tmp_path / "results"
    proc = subprocess.run(
        [sys.executable, "-m", "gfem", "run", str(job), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = gfem.load_result(out)
    assert result.terminals == expected.terminals
    # The same to rounding: threaded assembly does not fix the summation order.
    np.testing.assert_allclose(result.capacitance, expected.capacitance, rtol=1e-12)


def test_device_override_is_recorded(model, tmp_path):
    job = _job(model, tmp_path, linear_solver="direct", device="cuda")
    out = tmp_path / "results"
    assert main(["run", str(job), "--out", str(out), "--device", "cpu"]) == 0
    run = json.loads((out / "result.json").read_text())["run"]
    assert run["device"] == "cpu"
    assert run["device_override"] == "cpu"
    assert run["options"]["device"] == "cpu"


def test_no_override_records_the_job_device(model, tmp_path):
    out = tmp_path / "results"
    assert main(["run", str(_job(model, tmp_path)), "--out", str(out)]) == 0
    run = json.loads((out / "result.json").read_text())["run"]
    assert run["device"] == "cpu"
    assert run["device_override"] is None


@pytest.mark.parametrize("device", ["cuda", "tpu"])
def test_a_device_that_cannot_run_exits_1(model, tmp_path, capsys, monkeypatch, device):
    monkeypatch.setattr(_cuda, "_query", lambda: _cuda.Probe(0, "CUDA_ERROR_NO_DEVICE"))
    _cuda.probe.cache_clear()
    job = _job(model, tmp_path, linear_solver="direct")
    out = tmp_path / "results"
    assert main(["run", str(job), "--out", str(out), "--device", device]) == 1
    err = capsys.readouterr().err
    assert err.startswith("gfem: error:")
    assert "device" in err
    assert not (out / "result.json").exists()
    _cuda.probe.cache_clear()


def test_a_gpu_job_without_a_gpu_names_the_cuda_error(
    model, tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(_cuda, "_query", lambda: _cuda.Probe(0, "CUDA_ERROR_NO_DEVICE"))
    _cuda.probe.cache_clear()
    job = _job(model, tmp_path, linear_solver="direct", device="cuda")
    assert main(["run", str(job), "--out", str(tmp_path / "r")]) == 1
    err = capsys.readouterr().err
    assert "DeviceUnavailableError" in err
    assert "CUDA_ERROR_NO_DEVICE" in err
    assert "--device cpu" in err
    _cuda.probe.cache_clear()


def test_a_broken_job_exits_1(model, tmp_path, capsys):
    job = _job(model, tmp_path)
    (tmp_path / "stack.msh").write_text("changed")
    assert main(["run", str(job), "--out", str(tmp_path / "r")]) == 1
    assert "sha256" in capsys.readouterr().err


def test_a_newer_job_exits_1_asking_for_an_upgrade(model, tmp_path, capsys):
    job = _job(model, tmp_path)
    doc = json.loads(job.read_text())
    doc["schema_version"] = 99
    job.write_text(json.dumps(doc))
    assert main(["run", str(job), "--out", str(tmp_path / "r")]) == 1
    assert "upgrade gfem" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [[], ["solve", "job.json"], ["run"], ["run", "job.json"], ["run", "--out", "r"]],
)
def test_usage_errors_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_a_missing_job_file_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", str(tmp_path / "nope.json"), "--out", str(tmp_path / "r")])
    assert exc.value.code == 2
    assert "nope.json" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert gfem.__version__ in capsys.readouterr().out
