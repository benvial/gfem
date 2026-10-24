"""The arc spectrum script's pure parts (tools/bend3d_arc_spectrum.py): no solve."""

import json

import bend3d_arc_spectrum as arc
import numpy as np


def test_hashes_cover_the_three_scripts():
    hashes = arc.sha256s()
    assert set(hashes) == {
        "tools/bend3d_arc_spectrum.py",
        "tools/bend3d_run.py",
        "tools/bend3d_mesh.py",
    }
    assert all(len(h) == 64 for h in hashes.values())


def test_record_round_trips(tmp_path):
    s = np.ones((3, 2, 2), complex)
    record = {"profile": "smoke", "provenance": {}, "settings": {}}
    arc.write(
        record, {"spectrum_wavelengths": np.arange(3.0), "spectrum_s": s}, tmp_path
    )
    assert json.loads((tmp_path / "bend3d_arc.json").read_text()) == record
    with np.load(tmp_path / "bend3d_arc.npz") as npz:
        assert set(npz.files) == {"spectrum_wavelengths", "spectrum_s"}
        assert npz["spectrum_s"].dtype == np.complex128
