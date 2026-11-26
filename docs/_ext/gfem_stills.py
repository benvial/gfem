"""Card stills: one light/dark PNG pair per physics section, from the tutorials.

A still is a 3D view of one device, drawn in one house style so that the cards
of the site's grid read as a set: the camera, the neutral grey bodies, one
colormap per kind of quantity and a flat ground, with no axes, scalar bars or
text. Each tutorial tags one executed cell with the Jupyter tag ``card`` (cell
metadata only, so a downloaded notebook runs unchanged). ``docs/_stills/<stem>.py``
defines ``card(ns)``, which builds an off-screen PyVista plotter from the
notebook's namespace and calls `save_still`. The docs startup hook
(`gfem_scenes.install`) runs that script right after the tagged cell, in the
kernel; a section without a tutorial has a script whose ``card(None)`` builds
its own model, run by ``tools/stills.py``.

`save_still` restyles the plotter once per theme (every colour, lookup table
and opacity is set again, so one plotter serves both), renders at twice the
final size, crops both renders to one box around the subject and writes
``docs/_static/cards/<still>-{light,dark}.png`` and ``<still>.json``, which
records what was drawn so that a test can check the pair without rendering.

The top of this module needs only the standard library and NumPy, so that the
tests import the pure helpers; PyVista, Pillow and matplotlib load inside the
functions that draw.
"""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

DOCS = Path(__file__).resolve().parents[1]
CARDS = DOCS / "_static" / "cards"
TUTORIALS = DOCS / "tutorials"
SCRIPTS = DOCS / "_stills"
TAG = "card"

SIZE = (1280, 800)  # px, 16:10: twice the widest card (640 CSS px) for HiDPI screens
RENDER = (2560, 1600)  # px, twice the final size, downsampled for anti-aliasing
AZIMUTH, ELEVATION = -60.0, 30.0  # deg: camera on the +x, -y side, 30 deg up
VIEW_ANGLE = 20.0  # deg, perspective
ZOOM = 1.25  # reset_camera fits the bounding sphere, which is loose
MARGIN = 0.06  # of the crop's width, on each side of the subject
GROUND_TOLERANCE = 12  # sum of the RGB differences below which a pixel is ground
DROP_OPACITY = 0.2  # solid actors below this opacity (cladding, air boxes) are dropped
SEE_THROUGH = (
    0.25  # translucent bodies are redrawn at this opacity: a core shows its field
)

KINDS = ("magnitude", "signed", "temperature")
# The bodies' grey on each ground. The light grey is the site's mesh token; the
# dark mesh token is the field's zero, so the dark bodies take the light muted
# grey, or they would vanish against it.
BODIES = {"light": "#c4c4c4", "dark": "#6b6b6b"}
THEMES = ("light", "dark")
PRESET = {
    "azimuth_deg": AZIMUTH,
    "elevation_deg": ELEVATION,
    "view_angle_deg": VIEW_ANGLE,
    "up": [0, 0, 1],
}


def _ext(name):
    """Load ``docs/_ext/<name>.py`` by path (the docs are not a package)."""
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            name, DOCS / "_ext" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def ground(theme):
    """The theme's ground colour, the one the scenes and the site use."""
    return _ext("gfem_scenes").THEMES[theme]["ground"]


def ramp_name(kind, theme):
    """The name in ``gfem_figures.RAMPS`` of the kind's colormap on a theme."""
    return f"gfem_{kind}" + ("_dark" if theme == "dark" else "")


def hex_rgb(colour):
    """``#rrggbb`` as an integer triple."""
    return tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))


# --- pure helpers: the anchor index, the content box and the crop -------------


def anchors(notebooks=TUTORIALS):
    """Map the sha256 of each ``card`` cell's source to its notebook stem.

    The kernel gets no cell id or metadata, only the source, so the hook
    recognises the tagged cell by this hash, taken over the text the kernel
    sees: the lines of the notebook's ``source`` joined.
    """
    found = {}
    for path in sorted(Path(notebooks).glob("*.ipynb")):
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            if TAG in cell.get("metadata", {}).get("tags", ()):
                source = cell["source"]
                text = source if isinstance(source, str) else "".join(source)
                found[hashlib.sha256(text.encode()).hexdigest()] = path.stem
    return found


def content_box(image, ground_rgb, tolerance=GROUND_TOLERANCE):
    """The pixel box (x0, y0, x1, y1) of what differs from the ground.

    ``image`` is (rows, columns, 3 or more channels) of 8-bit values.
    """
    pixels = np.asarray(image)[..., :3].astype(int)
    diff = np.abs(pixels - np.array(ground_rgb, dtype=int)).sum(axis=-1)
    ys, xs = np.nonzero(diff > tolerance)
    if xs.size == 0:
        msg = "the render shows nothing but the ground"
        raise ValueError(msg)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_box(box, size=SIZE, margin=MARGIN):
    """Grow a content box to the size's aspect, with ``margin`` of the width per side.

    The box may leave the image: the crop pads with the ground. Its width is a
    multiple of the aspect's numerator, so the height is an integer and the
    ratio exact.
    """
    x0, y0, x1, y1 = box
    unit = size[0] // math.gcd(*size)  # 8 for 8:5
    aspect = size[0] / size[1]
    width = max(x1 - x0, (y1 - y0) * aspect) / (1 - 2 * margin)
    width = unit * math.ceil(width / unit)
    height = width * size[1] // size[0]
    left = round((x0 + x1) / 2 - width / 2)
    top = round((y0 + y1) / 2 - height / 2)
    return left, top, left + width, top + height


def sidecar(name, kind, records, camera):
    """The JSON record of a still: what a test checks without rendering it."""
    record = {
        "still": name,
        "kind": kind,
        "size": list(SIZE),
        "camera": camera,
    }
    for theme in THEMES:
        record[theme] = {
            "ground": ground(theme),
            "colormaps": sorted(records[theme]["colormaps"]),
            "solids": sorted(records[theme]["solids"]),
            "dropped": records[theme]["dropped"],
        }
    return record


# --- the card script hook -------------------------------------------------------


def run_card(stem, namespace):
    """Import ``docs/_stills/<stem>.py`` by path and call its ``card(namespace)``."""
    spec = importlib.util.spec_from_file_location(
        f"gfem_still_{stem.replace('-', '_')}", SCRIPTS / f"{stem}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.card(namespace)


# --- the restyle and the camera (PyVista) ---------------------------------------


def _classify(plotter):
    """Decide each actor's role once; hide the dropped; return (roles, dropped).

    Roles are decided on the plotter as the card script built it, before any
    theme repaints it, so a second theme sees the same decisions.
    """
    import pyvista as pv

    dropped = []
    for bar in list(plotter.scalar_bars.keys()):
        dropped.append("scalar bar")
        plotter.remove_scalar_bar(bar)
    roles = []
    for actor in plotter.renderer.actors.values():
        if not isinstance(actor, pv.Actor):  # text, axes, cube axes
            actor.SetVisibility(False)
            dropped.append(type(actor).__name__)
            continue
        prop, mapper = actor.prop, actor.mapper
        dataset = mapper.dataset
        prop.show_edges = False
        # Lines and wireframes first: a colour-mapped line set is still a
        # line set, and the cards keep none.
        if (
            isinstance(dataset, pv.PolyData)
            and dataset.n_lines > 0
            and dataset.n_cells == dataset.n_lines
        ) or prop.style == "wireframe":
            actor.SetVisibility(False)
            dropped.append("lines")
        elif mapper.scalar_visibility:  # a colour-mapped actor: the field
            # Direct colours (rgb=True) would bypass the lookup table, and the
            # still would carry colours of no ramp: map the scalars.
            mapper.SetColorModeToMapScalars()
            bounds = np.reshape(actor.GetBounds(), (3, 2))
            extent = np.ptp(bounds, axis=1)
            # A flat cut is drawn unlit, so that its colour is the value; a
            # curved surface (a warped body, a lobe) is lit, or its shape is lost.
            flat = extent.min() <= 1e-6 * extent.max()
            light = (
                {"ambient": 1.0, "diffuse": 0.0, "specular": 0.0}
                if flat
                else {"ambient": 0.45, "diffuse": 0.6, "specular": 0.0}
            )
            roles.append(
                ("field", actor, {"range": mapper.scalar_range, "light": light})
            )
        elif prop.opacity < DROP_OPACITY:
            actor.SetVisibility(False)
            dropped.append(f"opacity {prop.opacity:.2f}")
        else:
            if prop.opacity < 1:
                prop.opacity = SEE_THROUGH
            roles.append(("body", actor, {}))
    return roles, dropped


def _paint(plotter, roles, kind, theme):
    """Colour every kept actor for ``theme``; return the colormaps and solids used."""
    import pyvista as pv

    cmaps, solids = set(), set()
    for role, actor, info in roles:
        prop = actor.prop
        if role == "field":
            name = ramp_name(kind, theme)
            actor.mapper.lookup_table = pv.LookupTable(
                cmap=_ext("gfem_figures").ramp(name)
            )
            actor.mapper.scalar_range = info["range"]
            for key, value in info["light"].items():
                setattr(prop, key, value)
            cmaps.add(name)
        else:
            prop.color = BODIES[theme]
            prop.ambient, prop.diffuse, prop.specular = 0.25, 0.75, 0.1
            solids.add(BODIES[theme])
    plotter.renderer.set_background(ground(theme))
    return cmaps, solids


def _visible_bounds(plotter):
    bounds = np.array([np.inf, -np.inf] * 3)
    for actor in plotter.renderer.actors.values():
        if hasattr(actor, "GetBounds") and actor.GetVisibility():
            ab = np.array(actor.GetBounds())
            bounds[0::2] = np.minimum(bounds[0::2], ab[0::2])
            bounds[1::2] = np.maximum(bounds[1::2], ab[1::2])
    return bounds


def _direction(azimuth, elevation):
    az, el = math.radians(azimuth), math.radians(elevation)
    return np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )


def _camera(plotter, azimuth, elevation):
    """The camera preset, framed on the visible actors, with its two lights."""
    import pyvista as pv

    bounds = _visible_bounds(plotter)
    centre = (bounds[0::2] + bounds[1::2]) / 2
    camera = plotter.camera
    camera.focal_point = centre
    camera.position = centre + _direction(azimuth, elevation)  # distance: reset_camera
    camera.up = (0, 0, 1)
    camera.view_angle = VIEW_ANGLE
    plotter.reset_camera(bounds=bounds)
    camera.zoom(ZOOM)
    # A key light from above and left of the camera and a dimmer headlight: the
    # top, the front and the side of a box take three different greys.
    plotter.remove_all_lights()
    span = float(np.ptp(np.reshape(bounds, (3, 2)), axis=1).max())
    for (az, el), intensity in (
        ((azimuth - 60.0, 60.0), 0.6),
        ((azimuth, elevation), 0.5),
    ):
        position = centre + 10 * span * _direction(az, el)
        plotter.add_light(
            pv.Light(
                position=tuple(position),
                focal_point=tuple(centre),
                intensity=intensity,
                light_type="scene light",
            )
        )


def _crop(image, box, ground_colour):
    from PIL import Image

    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), ground_colour)
    canvas.paste(
        Image.fromarray(np.ascontiguousarray(image[..., :3])), (-box[0], -box[1])
    )
    return canvas.resize(SIZE, Image.LANCZOS)


def _quantized(image, ground_colour):
    """256 colours, dithered, the ground exact: the corner pixel matches the sidecar."""
    from PIL import Image

    quantized = image.quantize(colors=256, dither=Image.Dither.FLOYDSTEINBERG)
    palette = quantized.getpalette()
    index = quantized.getpixel((0, 0))
    palette[3 * index : 3 * index + 3] = hex_rgb(ground_colour)
    quantized.putpalette(palette)
    return quantized


def save_still(
    plotter, name, kind, *, azimuth=AZIMUTH, elevation=ELEVATION, directory=CARDS
):
    """Write the light and dark stills of ``plotter`` and their sidecar.

    ``plotter`` is an off-screen `pyvista.Plotter` the caller built; it is
    restyled, rendered twice and closed. ``kind`` (``"magnitude"``, ``"signed"``
    or ``"temperature"``) picks the colormap of the field actors. A script may
    override the camera's azimuth and elevation, and nothing else; the sidecar
    then says so.
    """
    if kind not in KINDS:
        msg = f"unknown still kind {kind!r}; choose one of {KINDS}"
        raise ValueError(msg)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plotter.window_size = list(RENDER)
    roles, dropped = _classify(plotter)
    _camera(plotter, azimuth, elevation)
    shots, records = {}, {}
    for theme in THEMES:
        cmaps, solids = _paint(plotter, roles, kind, theme)
        records[theme] = {"colormaps": cmaps, "solids": solids, "dropped": dropped}
        plotter.render()
        shots[theme] = plotter.screenshot(return_img=True)
    plotter.close()
    box = crop_box(content_box(shots["light"], hex_rgb(ground("light"))))
    for theme, shot in shots.items():
        image = _crop(shot, box, ground(theme))
        _quantized(image, ground(theme)).save(
            directory / f"{name}-{theme}.png", optimize=True
        )
    camera = {**PRESET, "azimuth_deg": azimuth, "elevation_deg": elevation}
    record = sidecar(name, kind, records, camera)
    if (azimuth, elevation) != (AZIMUTH, ELEVATION):
        record["camera_override"] = True
    (directory / f"{name}.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8"
    )
    return record
