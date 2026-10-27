"""Render the 3D bend's optimization as a video, for the front page.

Usage::

    python tools/bend3d_animation.py RECORD_DIR [--out DIR] [--size W H] [--fps N]

Reads ``bend3d_history.json`` and ``bend3d_history.npz`` (written by
``tools/bend3d_history.py``, not committed) and writes, for the site's light
and dark themes, ``bend3d-optimization-{theme}.webm`` and ``.mp4`` and a
poster ``.png`` of the last frame, plus ``bend3d-optimization.json``: the
time of each iteration in the video and its |S21|^2, which the front page
shows beside it. Default ``--out``: ``docs/_static/front``.

Each video frame is one point of the optimization, iterations linearly
blended between saved iterations:

- the silicon: the guides and the design pixels whose physical density is
  above 0.5, as blocks through the 0.22 um core;
- the field: a translucent isosurface of |E| at a fixed fraction of its peak
  in the input guide, mirrored about the PMC plane z = 0 to the full stack,
  and |E| on the core mid-plane drawn on the floor of the oxide box;
- the box's outline, and a camera that turns slowly round the bend.

The last iteration is followed by ``final`` (the thresholded design), held
while the camera keeps turning. Rendering needs an off-screen VTK and
``ffmpeg``: with a GPU, ``VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow``
(about 5 min per theme at 1600 x 900); ``vtkOSMesaRenderWindow`` renders on
the CPU, some 15 times slower.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "_static" / "front"
NAME = "bend3d-optimization"
um = 1e-6

# The site's colour tokens (docs/_static/gfem.css): the panel behind the
# figure, its ink, the hairline and the signal red of the sensitivity field.
THEMES = {
    "light": {
        "panel": "#fafafa",
        "ink": "#262626",
        "hair": "#c9c9c9",
        "field": (181, 38, 47),
        "silicon": "#8a8a8a",
    },
    "dark": {
        "panel": "#232323",
        "ink": "#ececea",
        "hair": "#4a4a4a",
        "field": (214, 48, 56),
        "silicon": "#b8b8b4",
    },
}
#: The isosurface level, as a fraction of |E|'s peak in the input guide.
ISO = 0.35
#: Video frames between two saved iterations, and of ``final`` at the end.
BLEND, HOLD = 6, 72
#: The camera's turn over the whole video (degrees) and its elevation.
TURN, ELEVATION = 70.0, 36.0


def load(record_dir):
    """The record's arrays and settings."""
    record_dir = Path(record_dir)
    record = json.loads((record_dir / "bend3d_history.json").read_text())
    with np.load(record_dir / "bend3d_history.npz") as npz:
        out = {key: npz[key] for key in npz.files}
    return record, out


def geometry():
    """The stack's dimensions (um), from the mesh builder."""
    import bend3d_mesh as bm

    return {
        "half": bm.BOX,
        "design": bm.DESIGN,
        "guide": bm.CORE_W / 2,
        "core": bm.CORE_T / 2,
        "top": bm.Z_TOP,
    }


def states(out):
    """Per video frame: (density, |E| volume, mid-plane |E|, iteration index).

    The index is fractional between saved iterations; ``final`` frames carry
    the last iteration's index plus one.
    """
    densities = out["density_history"].astype(float)
    volumes = out["frame_volume"].astype(float)
    tops = out["frame_cut_top"].astype(float)
    n = len(densities)
    frames = []
    for k in range(n - 1):
        for j in range(BLEND):
            a = j / BLEND
            frames.append(
                (
                    (1 - a) * densities[k] + a * densities[k + 1],
                    (1 - a) * volumes[k] + a * volumes[k + 1],
                    (1 - a) * tops[k] + a * tops[k + 1],
                    k + a,
                )
            )
    final = (
        out["final"].astype(float),
        out["final_volume"].astype(float),
        out["final_cut_top"].astype(float),
        float(n),
    )
    last = (densities[-1], volumes[-1], tops[-1], float(n - 1))
    frames += [last] * BLEND + [final] * HOLD
    return frames


class Scene:
    """The PyVista scene of one theme, updated frame by frame."""

    def __init__(self, theme, size, g, out):
        """The box outline and lights of ``theme`` at ``size``; ``out``: the record."""
        import pyvista as pv

        self.pv, self.g, self.colors = pv, g, THEMES[theme]
        xy = out["volume_xy"] / um
        z = out["volume_z"] / um
        self.z_full = np.concatenate([-z[::-1], z])
        self.volume_grid = pv.RectilinearGrid(xy, xy, self.z_full)
        cut = out["cut_x"] / um
        self.cut = cut
        self.in_guide = cut < -g["design"]
        from matplotlib.colors import LinearSegmentedColormap

        # density 0 is the oxide (the panel), 1 the silicon
        self.cmap = LinearSegmentedColormap.from_list(
            "rho", [self.colors["panel"], self.colors["silicon"]]
        )
        p = pv.Plotter(off_screen=True, window_size=size)
        p.set_background(self.colors["panel"])
        p.enable_anti_aliasing("ssaa")
        p.enable_lightkit()
        h, top = g["half"], g["top"]
        p.add_mesh(
            pv.Box((-h, h, -h, h, -top, top)).outline(),
            color=self.colors["hair"],
            line_width=1.5,
        )
        p.enable_depth_peeling(number_of_peels=8)
        self.plotter = p
        self.actors = []

    def _blocks(self, density):
        pv, g = self.pv, self.g
        ny, nx = density.shape
        pitch = 2 * g["design"] / nx
        blocks = pv.ImageData(
            dimensions=(nx + 1, ny + 1, 2),
            spacing=(pitch, pitch, 2 * g["core"]),
            origin=(-g["design"], -g["design"], -g["core"]),
        )
        blocks.cell_data["rho"] = density.ravel()  # rows y, x fastest
        silicon = [blocks.threshold(0.03, scalars="rho")]
        h, w, c = g["half"], g["guide"], g["core"]
        for box in ((-h, -g["design"], -w, w, -c, c), (-w, w, -h, -g["design"], -c, c)):
            arm = pv.Box(box).cast_to_unstructured_grid()
            arm.cell_data["rho"] = np.ones(arm.n_cells)
            silicon.append(arm)
        return [s for s in silicon if s.n_cells]

    def _iso(self, volume, scale):
        # |E| over z >= 0 (rows y, columns x, z) mirrored to the full stack
        full = np.concatenate([volume[..., ::-1], volume], axis=-1) / scale
        grid = self.volume_grid.copy()
        grid.point_data["E"] = np.transpose(full, (2, 0, 1)).ravel()  # x fastest
        return grid.contour([ISO], scalars="E")

    def _floor(self, top, scale):
        pv, g = self.pv, self.g
        r, gr, b = self.colors["field"]
        panel = np.array(pv.Color(self.colors["panel"]).int_rgb, float)
        v = np.clip(top / scale, 0, 1)
        alpha = np.minimum(0.92, 0.95 * v)[..., None]
        rgb = (1 - alpha) * panel + alpha * np.array([r, gr, b], float)
        n = len(self.cut)
        step = self.cut[1] - self.cut[0]
        floor = pv.ImageData(
            dimensions=(n + 1, n + 1, 1),
            spacing=(step, step, 1),
            origin=(self.cut[0] - step / 2, self.cut[0] - step / 2, -g["top"]),
        )
        floor.cell_data["rgb"] = rgb.reshape(-1, 3).astype(np.uint8)
        return floor

    def draw(self, density, volume, top, azimuth):
        """Replace the frame's actors and point the camera; returns the plotter."""
        p = self.plotter
        for actor in self.actors:
            p.remove_actor(actor)
        scale = top[:, self.in_guide].max()  # |E| peak in the input guide
        self.actors = [
            p.add_mesh(
                self._floor(top, scale),
                scalars="rgb",
                rgb=True,
                show_scalar_bar=False,
                lighting=False,
            ),
        ]
        for part in self._blocks(density):
            self.actors.append(
                p.add_mesh(
                    part,
                    scalars="rho",
                    cmap=self.cmap,
                    clim=(0, 1),
                    show_scalar_bar=False,
                    specular=0.2,
                )
            )
        iso = self._iso(volume, scale)
        if iso.n_points:
            self.actors.append(
                p.add_mesh(
                    iso,
                    color=self.colors["field"],
                    opacity=0.45,
                    smooth_shading=True,
                    specular=0.4,
                )
            )
        a, e = np.radians(azimuth), np.radians(ELEVATION)
        distance = 11.2
        p.camera.focal_point = (0.0, 0.0, -0.3)
        p.camera.position = (
            distance * np.cos(e) * np.cos(a),
            distance * np.cos(e) * np.sin(a),
            distance * np.sin(e) - 0.3,
        )
        p.camera.up = (0, 0, 1)
        p.camera.view_angle = 30
        return p


def encode(frames_dir, fps, dest):
    """``dest`` .webm (VP9) and .mp4 (H.264) from the numbered PNGs."""
    pattern = str(frames_dir / "%04d.png")
    common = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps)]
    common += ["-i", pattern, "-pix_fmt", "yuv420p"]
    vp9 = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36", "-row-mt", "1"]
    h264 = [
        "-c:v",
        "libx264",
        "-crf",
        "26",
        "-preset",
        "slow",
        "-movflags",
        "+faststart",
    ]
    subprocess.run([*common, *vp9, str(dest.with_suffix(".webm"))], check=True)
    subprocess.run([*common, *h264, str(dest.with_suffix(".mp4"))], check=True)


def main(argv=None):
    """Render both themes and write the timing JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("record", type=Path, help="directory of bend3d_history.*")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--size", type=int, nargs=2, default=(1600, 900))
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--themes", nargs="+", default=list(THEMES))
    args = parser.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH")
    record, out = load(args.record)
    g = geometry()
    frames = states(out)
    args.out.mkdir(parents=True, exist_ok=True)
    for theme in args.themes:
        scene = Scene(theme, tuple(args.size), g, out)
        with tempfile.TemporaryDirectory() as folder:
            pngs = Path(folder)
            for i, (density, volume, top, _) in enumerate(frames):
                azimuth = -135 + TURN * i / len(frames)
                scene.draw(density, volume, top, azimuth).screenshot(
                    pngs / f"{i:04d}.png"
                )
            dest = args.out / f"{NAME}-{theme}"
            encode(pngs, args.fps, dest)
            shutil.copy(pngs / f"{len(frames) - 1:04d}.png", dest.with_suffix(".png"))
        scene.plotter.close()
        print(f"{theme}: {len(frames)} frames -> {dest}.webm, .mp4, .png", flush=True)
    t = out["t_history"]
    n = len(t)
    timing = {
        "source": "bend3d_history record, by tools/bend3d_animation.py",
        "fps": args.fps,
        "frames": len(frames),
        "iterations": [
            {
                "time_s": round(k * BLEND / args.fps, 4),
                "t": round(float(t[k]), 4),
                "beta": float(out["beta_history"][k]),
            }
            for k in range(n)
        ],
        "final": {
            "time_s": round((n - 1) * BLEND / args.fps + BLEND / args.fps, 4),
            "t": round(float(np.abs(out["final_s"][1, 0]) ** 2), 4),
        },
        "profile": record["profile"],
        "commit": record["provenance"]["commit"],
    }
    (args.out / f"{NAME}.json").write_text(json.dumps(timing, indent=1) + "\n")


if __name__ == "__main__":
    main()
