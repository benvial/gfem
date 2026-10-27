"""Write the front-page hero data, ``docs/_data/front-page.json`` (committed).

The data comes from the 3D bend tutorial's committed run record,
``docs/tutorials/data/bend3d.json`` and ``bend3d.npz`` (written by
``tools/bend3d_run.py``): this script solves nothing, it only resamples and
quotes the record. The JSON holds:

- |S21|^2 and beta of every iteration, for the hero's history strip under
  the optimization video (``tools/bend3d_animation.py``);
- the physical density at the tutorial's four snapshot iterations, inside the
  design square, quantised to 0..255 and base64-encoded, rows from
  y = +1.5 um down;
- the numbers the front page quotes: |S21|^2 of ``final``, the schedule, the
  mesh, the solve time, reciprocity, power balance and the spectrum's
  minimum.

``docs/conf.py`` inlines the JSON into the front page; regenerate it whenever
the run record changes::

    just front-page-data
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "tutorials" / "data"
OUT = ROOT / "docs" / "_data" / "front-page.json"
um = 1e-6


def _bytes(values: np.ndarray, vmax: float) -> str:
    """``values`` / ``vmax`` as base64 uint8, rows top (max y) first."""
    scaled = np.clip(np.asarray(values, float) / vmax, 0, 1)[::-1]
    return base64.b64encode(np.round(255 * scaled).astype(np.uint8).tobytes()).decode()


def _t(s: np.ndarray) -> np.ndarray:
    """|S21|^2 of an S-matrix (..., 2, 2)."""
    return np.abs(np.asarray(s)[..., 1, 0]) ** 2


def _db(t: float) -> float:
    return float(-10 * np.log10(t))


def build() -> dict:
    """The hero JSON from the run record."""
    record = json.loads((DATA / "bend3d.json").read_text())
    with np.load(DATA / "bend3d.npz") as npz:
        out = {key: npz[key] for key in npz.files}
    settings, geometry = record["settings"], record["settings"]["geometry"]
    half, square = geometry["box_half_width_um"], geometry["design_half_width_um"]
    guide, pitch = geometry["core_width_um"] / 2, settings["pitch_um"]
    schedule = settings["schedule"]

    t_final = float(_t(out["s_final"]))
    s = out["s_final"]
    r11 = float(np.abs(s[0, 0]) ** 2)
    wavelengths = np.round(out["spectrum_wavelengths"] / um, 3)
    t_band = _t(out["spectrum_s"])
    low = int(np.argmin(t_band))
    seconds = out["seconds_history"][1:]  # iteration 0 includes the JIT compilation
    mesh = next(m for m in settings["meshes"].values() if "optimize" in m["stages"])
    snapshot_at = out["snapshot_iterations"]

    return {
        "source": "docs/tutorials/data/bend3d.{json,npz}, by tools/front_page_data.py",
        "extent_um": [-half, half, -half, half],
        "design_um": square,
        "guide_um": guide,
        "history": {
            "t": out["t_history"].round(5).tolist(),
            "beta": out["beta_history"].tolist(),
        },
        "snapshots": {
            "shape": list(out["snapshots"].shape[1:]),
            "iterations": snapshot_at.tolist(),
            "t": [round(float(out["t_history"][k]), 4) for k in snapshot_at],
            "density": [_bytes(rho, 1.0) for rho in out["snapshots"]],
        },
        "params": {
            "wavelength_um": settings["wavelength_um"],
            "core_thickness_um": geometry["core_thickness_um"],
            "core_width_um": geometry["core_width_um"],
            "square_um": 2 * square,
            "pixels": list(out["rho_final"].shape),
            "pitch_um": pitch,
            "filter_radius_um": settings["filter_radius_um"],
            "iterations": schedule["iterations"],
            "steps_per_beta": schedule["steps_per_beta"],
            "beta0": schedule["betas"][0],
            "start": schedule["start"],
            "dof": mesh["dof"],
            "tets": mesh["tets"],
            "order": settings["order"],
            "median_s": float(np.median(seconds)),
            "cores": record["provenance"]["physical_cores"],
            "n_si": settings["indices"]["core"]["n"],
            "n_sio2": settings["indices"]["cladding"]["n"],
        },
        "results": {
            "t_final": t_final,
            "db_final": _db(t_final),
            "core_fraction": float(out["final"].mean()),
            "reciprocity": float(np.abs(s[0, 1] - s[1, 0])),
            "balance": r11 + t_final,
            "band_min": {"um": float(wavelengths[low]), "t": float(t_band[low])},
            "band_um": [float(wavelengths.min()), float(wavelengths.max())],
        },
    }


def main() -> None:
    """Read the run record and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT, help="output JSON path")
    args = parser.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")  # never leave a half-written JSON
    tmp.write_text(json.dumps(build(), separators=(",", ":")) + "\n")
    tmp.replace(out)
    print("wrote", out.relative_to(ROOT) if out.is_relative_to(ROOT) else out)


if __name__ == "__main__":
    main()
