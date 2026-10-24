"""The arc design's spectrum, beside the 3D bend's run record.

Usage::

    python tools/bend3d_arc_spectrum.py --out DIR [--profile full|smoke]

``tools/bend3d_run.py``'s ``spectrum`` stage sweeps ``final`` over the band; this
script sweeps the hand-drawn arc (``arc_design``) the same way: the same mesh
(h = 0.07 um, mirror half model), solver (direct), wavelengths (11, 1.45 to
1.65 um) and constant indices, all imported from ``tools/bend3d_run.py``
unchanged. It writes a record of its own, ``bend3d_arc.json`` (provenance and
settings, no result numbers) and ``bend3d_arc.npz`` (the S-matrices), which
``docs/tutorials/bend3d.ipynb`` reads beside ``bend3d.json``.

The full profile takes about 20 min and 53 GiB on 16 cores with the ``mkl``
extra (PARDISO): one factorization per wavelength. ``--profile smoke`` runs at
h = 0.4 um with 3 wavelengths in under a minute, for development only.
"""

import argparse
import dataclasses
import hashlib
import json
import shlex
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import bend3d_run as run
import numpy as np
from scipy.constants import c

RECORD = ("bend3d_arc.json", "bend3d_arc.npz")
SCRIPTS = ("tools/bend3d_arc_spectrum.py", *run.SCRIPTS)


def _utc():
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256s():
    """The sha256 of this script, the run script and the mesh builder, by path."""
    return {
        path: hashlib.sha256((run.ROOT / path).read_bytes()).hexdigest()
        for path in SCRIPTS
    }


def sweep(stage):
    """The arc over the profile's wavelengths: ``(record settings, arrays)``."""
    profile = stage.profile
    bend, grid, name = stage.design_mesh(profile.h)
    result = run.solve_design(
        bend,
        grid,
        run.arc_design(grid),
        run.DIRECT,
        frequency=c / profile.wavelengths,
    )
    settings = {
        "design": "arc_design of tools/bend3d_run.py",
        "wavelengths_um": [round(w / 1e-6, 9) for w in profile.wavelengths],
        "mesh": {"name": name, **stage.meshes[name]},
        "solver_options": dataclasses.asdict(run.DIRECT),
    }
    arrays = {
        "spectrum_wavelengths": profile.wavelengths,
        "spectrum_s": run.s_matrix(result.s),
    }
    return settings, arrays


def write(record, arrays, dest):
    """``bend3d_arc.json`` and ``bend3d_arc.npz`` in ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / RECORD[0]).write_text(json.dumps(record, indent=1) + "\n")
    np.savez_compressed(dest / RECORD[1], **arrays)


def main(argv=None):
    """Sweep the arc and write its record into ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="record directory")
    parser.add_argument("--profile", default="full", choices=tuple(run.PROFILES))
    args = parser.parse_args(argv)
    commit = run.clean_commit(run.ROOT)
    memory, start, tic = run.PeakMemory(), _utc(), time.perf_counter()
    stage = run.Stage("spectrum", args.out / "work", args.profile)
    settings, arrays = sweep(stage)
    provenance = {
        "commit": commit,
        "sha256": sha256s(),
        **run.environment(),
        "start_utc": start,
        "end_utc": _utc(),
        "command": shlex.join(["python", *sys.argv]),
        "wall_s": round(time.perf_counter() - tic, 1),
        "peak_rss_bytes": memory.peak_bytes,
    }
    record = {"profile": args.profile, "provenance": provenance, "settings": settings}
    write(record, arrays, args.out)
    print(f"wrote {', '.join(RECORD)} in {args.out}", flush=True)


if __name__ == "__main__":
    main()
