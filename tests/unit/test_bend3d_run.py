"""The 3D bend run script's pure parts (tools/bend3d_run.py): no solve."""

import json

import bend3d_run as run
import numpy as np
import pytest

# The run record's solver outputs, typed here, not read back
# from the script, so a drift in either is a failure.
NPZ_KEYS = [
    *("t_history", "core_history", "beta_history", "seconds_history"),
    *("rho_start", "rho_final", "snapshots", "snapshot_iterations", "last", "final"),
    *("eroded", "nominal", "dilated", "arc", "corner"),
    *("s_last", "s_final", "s_final_full", "s_final_h007", "s_final_box7"),
    *("s_eroded", "s_nominal", "s_dilated", "s_arc", "s_corner", "s_straight"),
    *("n_eff", "spectrum_wavelengths", "spectrum_s", "flux_in", "flux_up"),
    *("cut_top_final", "cut_top_arc", "cut_out_final", "cut_out_arc"),
    *("cut_in_final", "cut_x", "cut_y", "cut_z"),
]


def _record():
    """A record with every key the schema asks for, and small arrays."""
    record = {
        "profile": "smoke",
        "provenance": dict.fromkeys(run.PROVENANCE_KEYS, ""),
        "settings": run.settings(run.PROFILES["smoke"], {}),
        "observations": {
            stage: dict.fromkeys(run.observation_keys(stage), 0.0)
            for stage in run.STAGES
        },
    }
    arrays = {key: np.zeros(3) for key in NPZ_KEYS}
    return record, arrays


def test_record_holds_exactly_the_listed_keys(tmp_path):
    record, arrays = _record()
    run.write_record(record, arrays, tmp_path)
    loaded = json.loads((tmp_path / "bend3d.json").read_text())
    assert set(loaded) == {"profile", "provenance", "settings", "observations"}
    assert set(loaded["observations"]) == {
        "optimize",
        "half",
        "spectrum",
        "full",
        "box",
        "fine",
    }
    assert "gmres_iterations" in loaded["observations"]["fine"]
    with np.load(tmp_path / "bend3d.npz") as npz:
        assert sorted(npz.files) == sorted(NPZ_KEYS)


def test_settings_give_the_plane_positions_as_numbers():
    settings = run.settings(run.PROFILES["smoke"], {})
    assert set(settings) == set(run.SETTINGS_KEYS)
    assert settings["geometry"]["port_window_half_width_um"] == 2.0
    assert settings["cuts"]["cut_in_x_um"] == pytest.approx(-2.0)
    assert settings["cuts"]["cut_out_y_um"] == pytest.approx(-2.0)
    assert settings["fluxes"]["flux_in_x_um"] == pytest.approx(-2.0)
    assert settings["fluxes"]["flux_up_z_um"] == pytest.approx(1.7)
    json.dumps(settings)  # the record's JSON holds them


@pytest.mark.parametrize(
    ("where", "key"),
    [
        ("geometry", "port_window_half_width_um"),
        ("cuts", "cut_in_x_um"),
        ("cuts", "cut_out_y_um"),
        ("fluxes", "flux_in_x_um"),
        ("fluxes", "flux_up_z_um"),
    ],
)
def test_record_without_a_plane_position_is_refused(tmp_path, where, key):
    record, arrays = _record()
    del record["settings"][where][key]
    with pytest.raises(ValueError, match=key):
        run.write_record(record, arrays, tmp_path)
    assert not list(tmp_path.iterdir())


def test_record_with_a_missing_or_extra_array_is_refused(tmp_path):
    record, arrays = _record()
    del arrays["s_final_box7"]
    arrays["s_final_box8"] = np.zeros(3)
    with pytest.raises(ValueError, match=r"s_final_box7.*s_final_box8"):
        run.write_record(record, arrays, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("cap", ["MAX_RECORD_BYTES", "MAX_FILE_BYTES"])
def test_record_over_a_size_cap_is_refused(tmp_path, monkeypatch, cap):
    record, arrays = _record()
    monkeypatch.setattr(run, cap, 1000)
    with pytest.raises(ValueError, match="bytes"):
        run.write_record(record, arrays, tmp_path)
    assert not list(tmp_path.iterdir())


def _git(root, *args):
    import shutil
    import subprocess

    git = shutil.which("git")
    subprocess.run([git, "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path):
    """A git checkout with one committed file under src/ and one under tools/."""
    for name in ("src/a.py", "tools/b.py", "docs/c.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")
    return tmp_path


def test_clean_checkout_gives_its_commit(checkout, monkeypatch):
    monkeypatch.setenv("GFEM_RUN_COMMIT", "not this")
    (checkout / "docs/c.md").write_text("edited\n")  # outside src/ and tools/
    commit = run.clean_commit(checkout)
    assert len(commit) == 40
    assert commit != "not this"


@pytest.mark.parametrize("edit", ["src/a.py", "tools/new.py"])
def test_dirty_src_or_tools_is_refused(checkout, edit):
    (checkout / edit).write_text("x = 2\n")
    with pytest.raises(RuntimeError, match=edit):
        run.clean_commit(checkout)


def test_outside_a_checkout_the_commit_comes_from_the_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GFEM_RUN_COMMIT", "0123abcd")
    assert run.clean_commit(tmp_path) == "0123abcd"
    monkeypatch.delenv("GFEM_RUN_COMMIT")
    with pytest.raises(RuntimeError, match="GFEM_RUN_COMMIT"):
        run.clean_commit(tmp_path)


@pytest.fixture
def grid():
    """The design's layer grid as the designs read it: 60 x 60 pixels of 0.05 um."""
    from types import SimpleNamespace

    centres = (-1.5 + 0.05 * (np.arange(60) + 0.5)) * 1e-6
    return SimpleNamespace(x=centres, y=centres, pitch=0.05e-6, shape=(60, 60))


def test_hand_drawn_designs_cover_their_drawn_area(grid):
    # Areas in um^2 by hand: a quarter ring of radii 1.1 and 1.9 about
    # (-1.5, -1.5) um; an L of two 0.8 um arms, 1.9 um long, sharing a square.
    arc, corner = run.arc_design(grid), run.corner_design(grid)
    pixel = 0.05**2
    assert arc.shape == corner.shape == (60, 60)
    assert arc.sum() * pixel == pytest.approx(np.pi / 4 * (1.9**2 - 1.1**2), rel=2e-3)
    assert corner.sum() * pixel == pytest.approx(2 * 1.9 * 0.8 - 0.8**2, rel=1e-9)
    # rows are y, columns x: both designs start at both arms
    for design in (arc, corner):
        assert design[30, 0] == design[0, 30] == 1.0
        assert design[59, 59] == 0.0


def test_bddc_applies_to_the_box_only(tmp_path, monkeypatch):
    box = run.Stage("box", tmp_path, "smoke", bddc=True)
    assert box.solver().linear_solver == "bddc_gmres"
    assert box.options["linear_solver"] == "bddc_gmres"
    with pytest.raises(SystemExit):
        run.main(["optimize", "--out", str(tmp_path), "--bddc"])
    commands = []
    monkeypatch.setattr(run.subprocess, "run", lambda c, **_: commands.append(c))
    run.main(["all", "--out", str(tmp_path), "--profile", "smoke", "--bddc"])
    assert [c[2] for c in commands] == [*run.STAGES, "merge"]
    assert [c[2] for c in commands if "--bddc" in c] == ["box"]


def _check(default_pmc=(), shape=(60, 60), unsampled=0, *, all_sampled=True):
    from types import SimpleNamespace

    sampled = np.ones(shape, bool)
    sampled.flat[:unsampled] = False
    run.check_mesh(
        "m",
        SimpleNamespace(default_pmc=default_pmc),
        SimpleNamespace(shape=shape, sampled=sampled),
        all_sampled=all_sampled,
    )


def test_mesh_checks_refuse_a_default_pmc_a_wrong_grid_or_unsampled_pixels():
    _check()
    _check(unsampled=5, all_sampled=False)  # the smoke profile's coarse meshes
    with pytest.raises(RuntimeError, match="PMC"):
        _check(default_pmc=("side",))
    with pytest.raises(RuntimeError, match="60, 60"):
        _check(shape=(59, 60))
    with pytest.raises(RuntimeError, match="5 of 3600"):
        _check(unsampled=5)


def _info(**environment):
    env = {
        "versions": {"gfem": "1.0"},
        "pardiso_available": True,
        "os": "Linux",
        "kernel": "6.1",
        "cpu_model": "cpu",
        "physical_cores": 16,
        "threads": 16,
        "memory_limit_bytes": 64,
        "host": "a",
    }
    return {
        "commit": "c",
        "dirty": [],
        "sha256": {"s": "h"},
        "profile": "full",
        "environment": env | environment,
    }


def test_a_stage_rerun_in_another_container_of_the_same_machine_merges():
    first, hashes = _info(), {"s": "h"}
    assert run.mismatches(_info(host="b", kernel="6.2"), first, "c", hashes) == []
    assert run.mismatches(first, first, "d", hashes) == ["commit 'c', not 'd'"]
    (problem,) = run.mismatches(_info(threads=8), first, "c", hashes)
    assert problem.startswith("machine")


def test_command_line_is_the_rerun_command():
    argv = ["/abs/checkout/tools/bend3d_run.py", "fine", "--out", "/w d"]
    assert run.command_line(argv) == "python tools/bend3d_run.py fine --out '/w d'"
