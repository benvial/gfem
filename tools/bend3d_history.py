"""The 3D bend's optimization, recorded at every iteration, for animations.

Usage::

    python tools/bend3d_history.py PHASE --work DIR [--out DIR] [--profile full|smoke]

``tools/bend3d_run.py``'s ``optimize`` stage keeps |S21|^2 and the core
fraction of every iteration but the design at four only, and no field. This
script reruns the same optimization, with every model value, the mesh, the
solver and the Adam loop imported from ``tools/bend3d_run.py`` unchanged, and
keeps everything it computes. It writes a record of its own,
``bend3d_history.json`` (provenance, settings, observations) and
``bend3d_history.npz``, which is not committed: it is some 70 MB, and the
animations rendered from it are.

Phases, each in its own process so a failed one reruns alone:

- ``optimize``: the 42 Adam iterations, keeping per iteration the raw pixels
  ``rho``, the filtered and the physical densities, the gradient dJ/d rho,
  |S21|^2, the core fraction, beta and the wall time (``DIR/optimize.npz``).
  On the full profile its |S21|^2 history is compared with the committed run
  record's (``docs/tutorials/data/bend3d.npz``).
- ``frames``: one forward solve per iteration of that iteration's physical
  density (the jitted gradient keeps no field: wave ports), and one of
  ``final``, with p1 fed (1 W incident). Each frame keeps the S-matrix, the
  tutorial's cuts and fluxes (`bend3d_run.field_maps`), complex E and the
  in-plane time-averaged Poynting vector on the core mid-plane z = 0, and
  |E| on a 0.05 um grid over the mirror half model. ``final`` also keeps
  complex E on that grid. Each frame is saved on its own
  (``DIR/frames/NN.npz``) and skipped when present, so a rerun resumes.
- ``merge``: the record into ``--out`` (default ``DIR``).
- ``all``: the three in order, each as a subprocess.

The full profile takes about 3.2 h on 16 cores / 96 GiB with the ``mkl``
extra: about 1.7 h for ``optimize`` and 42 x 130 s for ``frames``.
``--profile smoke`` (h = 0.4 um, 2 iterations) runs in minutes, for
development only.
"""

# The run script's own sampling helpers, reused so the frames sample as its cuts do.
# ruff: noqa: SLF001

import argparse
import gc
import json
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import bend3d_mesh as bm
import bend3d_run as run
import jax
import jax.numpy as jnp
import numpy as np

RECORD = ("bend3d_history.json", "bend3d_history.npz")
SCRIPTS = ("tools/bend3d_history.py", *run.SCRIPTS)
PHASES = ("optimize", "frames", "merge")
um = run.um
#: The |E| volume's step: every other point of the tutorial's cut grid.
VOLUME_STEP = 2 * run.CUT_STEP
VOLUME_ALONG = run.CUT_ALONG[::2] + run.CUT_STEP / 2
VOLUME_UP = VOLUME_STEP * (np.arange(round(bm.Z_TOP * um / VOLUME_STEP)) + 0.5)


def _utc():
    return datetime.now(UTC).isoformat(timespec="seconds")


def optimize(stage):
    """`bend3d_run.optimize`'s loop, keeping every iteration's arrays."""
    import optax  # the docs group's, as in the run script

    profile = stage.profile
    bend, grid, _ = stage.design_mesh(profile.h)
    blur = run.density_filter(grid)
    stage.solver()
    value_and_grad = jax.jit(
        jax.value_and_grad(run.missed_power(bend, grid, blur), has_aux=True)
    )
    rho = jnp.full(grid.shape, 0.5)
    optimizer = optax.adam(run.LEARNING_RATE)
    state = optimizer.init(rho)
    keep = {key: [] for key in ("rho", "filtered", "density", "grad")}
    history = []
    for beta in profile.betas:
        for _ in range(profile.steps_per_beta):
            k, tic = len(history), time.perf_counter()
            (_, (t, core)), g = value_and_grad(rho, beta)
            keep["rho"].append(np.asarray(rho))
            keep["filtered"].append(np.asarray(run.filtered(rho, blur)))
            keep["density"].append(np.asarray(run.physical(rho, beta, blur)))
            keep["grad"].append(np.asarray(g))
            updates, state = optimizer.update(g, state)
            rho = jnp.clip(optax.apply_updates(rho, updates), 0.0, 1.0)
            t, core = float(t), float(core)
            history.append((t, core, beta, time.perf_counter() - tic))
            print(
                f"iteration {k}: beta = {beta:g}, |S21|^2 = {t:.4f}, "
                f"core fraction = {core:.3f}, {history[-1][3]:.1f} s",
                flush=True,
            )
    last = np.asarray(run.physical(rho, profile.betas[-1], blur))
    t_history, core_history, beta_history, seconds_history = np.array(history).T
    return {
        "t_history": t_history,
        "core_history": core_history,
        "beta_history": beta_history,
        "seconds_history": seconds_history,
        **{f"{key}_history": np.stack(v).astype(np.float32) for key, v in keep.items()},
        "rho_final": np.asarray(rho, np.float32),
        "last": last.astype(np.float32),
        "final": (last > run.THRESHOLD).astype(np.uint8),
    }


def _mid_plane(field):
    """Complex E and B on z = 0 (rows y, columns x, 3)."""
    top = run._plane(2, 0.0, run.CUT_ALONG, run.CUT_ALONG)[..., [1, 0, 2]]
    return run._sample(field, top), run._sample(field.b(), top)


def _volume(field):
    """Complex E on the volume grid (rows y, columns x, z, 3), z >= 0."""
    y, x, z = np.meshgrid(VOLUME_ALONG, VOLUME_ALONG, VOLUME_UP, indexing="ij")
    return run._sample(field, np.stack([x, y, z], axis=-1))


def frame(bend, grid, density, cut_planes, *, volume_e=False):
    """One forward solve of ``density``, p1 fed: the frame's arrays."""
    result = run.solve_design(bend, grid, density, run.DIRECT, keep=True)
    e = result.e_field(run.F0, excite="p1")
    e_top, b_top = _mid_plane(e)
    s_top = 0.5 * np.real(np.cross(e_top, np.conj(b_top / run.mu_0)))[..., :2]
    e_volume = _volume(e)
    arrays = {
        "s": run.s_matrix(result.s)[0],
        **run.field_maps(result, cut_planes),
        "e_top": e_top.astype(np.complex64),
        "poynting_top": s_top.astype(np.float32),
        "volume": run._magnitude(e_volume),
    }
    if volume_e:
        arrays["e_volume"] = e_volume.astype(np.complex64)
    return arrays


def frames(stage, work):
    """Every iteration's density and ``final``, one saved frame each."""
    opt = _load(work / "optimize.npz")
    bend, grid, _ = stage.design_mesh(stage.profile.h)
    cut_planes = run.planes(stage.profile.h)
    folder = work / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    designs = [*opt["density_history"], opt["final"].astype(float)]
    for k, density in enumerate(designs):
        path = folder / f"{k:02d}.npz"
        if path.exists():
            continue
        tic = time.perf_counter()
        last = k == len(designs) - 1
        arrays = frame(
            bend, grid, np.asarray(density, float), cut_planes, volume_e=last
        )
        partial = path.with_suffix(".part.npz")
        np.savez_compressed(partial, **arrays)
        partial.replace(path)
        name = "final" if last else f"iteration {k}"
        t = float(np.abs(arrays["s"][1, 0]) ** 2)
        print(
            f"{name}: |S21|^2 = {t:.4f}, {time.perf_counter() - tic:.1f} s", flush=True
        )
        del arrays
        gc.collect()
    return len(designs)


def _load(path):
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def _phase_info(work, phase):
    return work / f"{phase}.json"


def run_phase(phase, work, profile):
    """Run ``optimize`` or ``frames`` and note its provenance in ``work``."""
    memory, start, tic = run.PeakMemory(), _utc(), time.perf_counter()
    commit, dirty = run.run_commit(run.ROOT)
    stage = run.Stage(phase, work, profile)
    if phase == "optimize":
        arrays = optimize(stage)
        partial = work / "optimize.part.npz"
        np.savez_compressed(partial, **arrays)
        partial.replace(work / "optimize.npz")
    else:
        frames(stage, work)
    info = {
        "phase": phase,
        "profile": profile,
        "command": shlex.join(["python", *sys.argv]),
        "start_utc": start,
        "end_utc": _utc(),
        "commit": commit,
        "dirty": dirty,
        "sha256": {
            path: run.hashlib.sha256((run.ROOT / path).read_bytes()).hexdigest()
            for path in SCRIPTS
        },
        "environment": run.environment(),
        "meshes": stage.meshes,
        "solver_options": stage.options,
        "observations": {
            "wall_s": round(time.perf_counter() - tic, 1),
            "peak_rss_bytes": memory.peak_bytes,
        },
    }
    _phase_info(work, phase).write_text(json.dumps(info, indent=1) + "\n")
    print(f"{phase}: {info['observations']['wall_s']} s", flush=True)


def merge(work, out, profile):
    """The record from the phases' outputs, into ``out``."""
    infos = {p: json.loads(_phase_info(work, p).read_text()) for p in PHASES[:2]}
    commits = {info["commit"] for info in infos.values()}
    dirty = any(info["dirty"] for info in infos.values())
    if len(commits) != 1 or (dirty and profile == "full"):  # smoke: development
        raise RuntimeError(f"phases ran at {commits}, or with uncommitted changes")
    arrays = _load(work / "optimize.npz")
    paths = sorted((work / "frames").glob("[0-9][0-9].npz"))
    if len(paths) != len(arrays["t_history"]) + 1:
        raise RuntimeError(f"{len(paths)} frames in {work / 'frames'}: rerun frames")
    loaded = [_load(path) for path in paths]
    *iterations, final = loaded
    for key in iterations[0]:
        arrays[f"frame_{key}"] = np.stack([f[key] for f in iterations])
    arrays |= {f"final_{key}": value for key, value in final.items()}
    arrays["cut_x"] = arrays["cut_y"] = run.CUT_ALONG.astype(np.float32)
    arrays["cut_z"] = run.CUT_HEIGHT.astype(np.float32)
    arrays["volume_xy"] = VOLUME_ALONG.astype(np.float32)
    arrays["volume_z"] = VOLUME_UP.astype(np.float32)
    check = {}
    committed = run.ROOT / "docs" / "tutorials" / "data" / "bend3d.npz"
    if profile == "full" and committed.exists():
        with np.load(committed) as npz:
            diff = np.abs(arrays["t_history"] - npz["t_history"])
            check = {
                "t_history_max_abs_diff": float(diff.max()),
                "final_pixels_differing": int((arrays["final"] != npz["final"]).sum()),
            }
        print(f"against the committed run record: {check}", flush=True)
    profile_values = run.PROFILES[profile]
    record = {
        "profile": profile,
        "provenance": {
            "commit": commits.pop(),
            "phases": {
                p: {k: info[k] for k in ("start_utc", "end_utc", "command")}
                for p, info in infos.items()
            },
            "sha256": infos["optimize"]["sha256"],
            **infos["optimize"]["environment"],
        },
        "settings": {
            "h_um": profile_values.h,
            "betas": list(profile_values.betas),
            "steps_per_beta": profile_values.steps_per_beta,
            "meshes": infos["frames"]["meshes"],
            "solver_options": infos["frames"]["solver_options"],
            "frames": "every iteration's physical density, then final (final_*)",
            "fed": "p1, 1 W incident in the fundamental mode, mirror half model z >= 0",
            "e_top": "complex E (V/m) on z = 0, rows y, columns x (cut_y, cut_x), 3",
            "poynting_top": "0.5 Re(E x conj(B) / mu_0), x and y (W/m^2), on z = 0",
            "volume": "|E| (MV/m, float16) at (volume_xy y, volume_xy x, volume_z)",
            "final_e_volume": "complex E (V/m) of final on the volume grid, 3",
            "cuts": "frame_cut_top, frame_cut_out, frame_cut_in, fluxes, te1: as "
            "bend3d_run.field_maps",
        },
        "observations": {p: info["observations"] for p, info in infos.items()}
        | {"against_record": check},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / RECORD[0]).write_text(json.dumps(record, indent=1) + "\n")
    np.savez_compressed(out / RECORD[1], **arrays)
    size = (out / RECORD[1]).stat().st_size
    print(f"wrote {', '.join(RECORD)} in {out} ({size / 2**20:.1f} MiB)", flush=True)


def main(argv=None):
    """Run one phase, or ``all`` of them as subprocesses."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=(*PHASES, "all"))
    parser.add_argument("--work", type=Path, required=True, help="work directory")
    parser.add_argument("--out", type=Path, help="record directory (default: --work)")
    parser.add_argument("--profile", default="full", choices=tuple(run.PROFILES))
    args = parser.parse_args(argv)
    work, out = args.work.resolve(), (args.out or args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.phase == "all":
        for phase in PHASES:
            cmd = [sys.executable, __file__, phase, "--work", str(work)]
            cmd += ["--out", str(out), "--profile", args.profile]
            subprocess.run(cmd, check=True)
    elif args.phase == "merge":
        merge(work, out, args.profile)
    else:
        run_phase(args.phase, work, args.profile)


if __name__ == "__main__":
    main()
