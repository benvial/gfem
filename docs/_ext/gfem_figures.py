"""Theme-paired matplotlib figures for the docs build.

When the docs build executes the tutorials, every kernel runs
``docs/_ipython/profile_default/startup/00-gfem-figures.py`` (``conf.py``
points ``IPYTHONDIR`` at ``docs/_ipython``), which calls `install`. From then
on a displayed figure is drawn in the gfem style and comes out as a light and
a dark PNG, swapped by Shibuya's ``light-only`` / ``dark-only`` classes. The
dark render is the same figure recoloured by role: every colour is looked up
in a light-to-dark table and replaced, the PNG is drawn, and the figure is
restored. The notebooks themselves stay plain matplotlib.

The recolour table pairs each colour of ``docs/_mpl/gfem.mplstyle`` with the
same key of ``docs/_mpl/gfem-dark.mplstyle`` (the plot cycle pairs entry by
entry), plus the site's colour tokens (``gfem.css``) and plain colours the
styles do not name.
A colour or colormap outside the table is written to stderr, which the docs
build turns into a warning (``nb_output_stderr = "warn"``) and ``-W`` into a
failure: an off-palette figure cannot reach the site silently.
"""

import base64
import html
import io
import logging
import struct
import sys
from pathlib import Path

import matplotlib as mpl
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib import style as mstyle
from matplotlib.artist import Artist
from matplotlib.collections import Collection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

MPL_DIR = Path(__file__).resolve().parents[1] / "_mpl"
LIGHT_STYLE = MPL_DIR / "gfem.mplstyle"
DARK_STYLE = MPL_DIR / "gfem-dark.mplstyle"
FONTS = MPL_DIR / "fonts"

DPI = 192  # twice the CSS pixel density, so the PNGs stay sharp on HiDPI screens
CSS_DPI = 96

# The site's colour tokens (docs/_static/gfem.css, light -> dark) that the
# styles do not name.
TOKENS = {
    "#ffffff": "#111317",  # ground; also white, as tutorials hard-code it
    "#cdd0d5": "#353a42",  # mesh
    "#007c6e": "#3cc4b0",  # accent
    "#e2f2ef": "#12302b",  # accent-soft
}
# Achromatic colours the tutorials hard-code: color="k" and color="gray".
PLAIN = {
    "#000000": "#e6e8eb",  # "k" -> ink-dark
    "#808080": "#9aa1aa",  # "gray" -> muted-dark
}

# The sensitivity ramp: one hue, from the ground to 0.92 * teal + 0.08 * ground,
# per theme.
RAMPS = {
    "gfem_sensitivity": ("#ffffff", "#007c6e"),
    "gfem_sensitivity_dark": ("#111317", "#3cc4b0"),
}
CMAP_SWAP = {"gfem_sensitivity": "gfem_sensitivity_dark"}
# Greyscale density maps are data: the same in both themes.
DATA_CMAPS = {"gray", "gray_r", "Greys", "Greys_r"}

COLOUR_PROPS = {
    Line2D: ("color", "markerfacecolor", "markeredgecolor", "markerfacecoloralt"),
    Text: ("color",),
    Patch: ("facecolor", "edgecolor"),
    Collection: ("facecolor", "edgecolor"),
}


def _style_colours(path):
    """The colours of a style file: {key: hex} with the cycle as ``cycle[i]``."""
    rc = mpl.rc_params_from_file(path, use_default_template=False)
    out = {}
    for key, value in rc.items():
        if key == "axes.prop_cycle":
            for i, colour in enumerate(value.by_key()["color"]):
                out[f"cycle[{i}]"] = mcolors.to_hex(colour)
        elif (
            isinstance(value, str) and value != "none" and mcolors.is_color_like(value)
        ):
            out[key] = mcolors.to_hex(value)
    return out


def _recolour_table():
    light, dark = _style_colours(LIGHT_STYLE), _style_colours(DARK_STYLE)
    if light.keys() != dark.keys():
        raise ValueError(
            f"{LIGHT_STYLE.name} and {DARK_STYLE.name} name different colours: "
            f"{sorted(light.keys() ^ dark.keys())}"
        )
    table = {**TOKENS, **PLAIN}
    for key, colour in light.items():
        if table.setdefault(colour, dark[key]) != dark[key]:
            raise ValueError(
                f"{colour} ({key}) has two dark counterparts: "
                f"{table[colour]} and {dark[key]}"
            )
    return table


RECOLOUR = _recolour_table()


def _ramp(name, ground, accent):
    ground_rgb = mcolors.to_rgb(ground)
    top = [
        0.92 * a + 0.08 * g
        for a, g in zip(mcolors.to_rgb(accent), ground_rgb, strict=True)
    ]
    return LinearSegmentedColormap.from_list(name, [ground_rgb, top])


def _not_a_weight_fallback(record):
    # Matplotlib looks up every font.family entry for glyph fallback and logs
    # each weight it lacks (DejaVu Sans has no 600). In a kernel that log line
    # is stderr, which the docs build treats as a warning.
    return not record.getMessage().startswith("findfont: Failed to find font weight")


def apply_style():
    """Register Barlow and the sensitivity colormaps, then apply the light style."""
    logging.getLogger("matplotlib.font_manager").addFilter(_not_a_weight_fallback)
    for font in sorted(FONTS.glob("*.ttf")):
        font_manager.fontManager.addfont(font)
    for name, (ground, accent) in RAMPS.items():
        if name not in mpl.colormaps:
            mpl.colormaps.register(_ramp(name, ground, accent))
    mstyle.use(LIGHT_STYLE)


def _dark(colour, unknown, where):
    """The dark counterpart of one colour, alpha kept; None leaves it as is."""
    rgba = mcolors.to_rgba(colour)
    if rgba[3] == 0:
        return None
    dark = RECOLOUR.get(mcolors.to_hex(rgba))
    if dark is None:
        unknown.add(f"{mcolors.to_hex(rgba)} ({where})")
        return None
    return (*mcolors.to_rgb(dark), rgba[3])


def _recolour(artist, undo, unknown):
    # Read every colour before setting any: a marker colour left on "auto"
    # follows the line colour, so it must be read while still light.
    current = [
        (prop, getattr(artist, f"get_{prop}")())
        for kind, props in COLOUR_PROPS.items()
        if isinstance(artist, kind)
        for prop in props
        if not (
            prop == "facecolor"
            and isinstance(artist, Collection)
            and artist.get_array() is not None  # colormapped: swapped below
        )
    ]
    for prop, value in current:
        where = f"{type(artist).__name__} {prop}"
        if isinstance(artist, Collection):
            rows = [_dark(c, unknown, where) for c in value]
            if all(r is None for r in rows):
                continue
            new = [
                tuple(c) if r is None else r for r, c in zip(rows, value, strict=True)
            ]
        else:
            new = _dark(value, unknown, where)
            if new is None:
                continue
        setter = getattr(artist, f"set_{prop}")
        undo.append((setter, value))
        setter(new)
    get_cmap = getattr(artist, "get_cmap", None)
    if get_cmap is None or artist.get_array() is None:
        return
    name = get_cmap().name
    if name in CMAP_SWAP:
        undo.append((artist.set_cmap, get_cmap()))
        artist.set_cmap(CMAP_SWAP[name])
    elif name not in DATA_CMAPS:
        unknown.add(f"colormap {name!r} ({type(artist).__name__})")


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", transparent=True)
    return buf.getvalue()


def _dark_png(fig):
    undo, unknown = [], set()
    artists = fig.findobj(lambda a: isinstance(a, Artist) and a.get_visible())
    try:
        for artist in artists:
            _recolour(artist, undo, unknown)
        return _png(fig)
    finally:
        for setter, value in reversed(undo):
            setter(value)
        for item in sorted(unknown):
            sys.stderr.write(
                f"gfem_figures: no dark counterpart for {item}; add it to the "
                "recolour table in docs/_ext/gfem_figures.py\n"
            )


def _alt(fig):
    parts = []
    for ax in fig.axes:
        # The gfem style puts titles on the left; read whichever slot is set.
        title = next(
            (t for loc in ("left", "center", "right") if (t := ax.get_title(loc))), ""
        )
        x, y = ax.get_xlabel(), ax.get_ylabel()
        if y and x:
            parts.append(f"{title + ': ' if title else ''}{y} against {x}")
        elif title or y:
            parts.append(title or y)
    return "Plot of " + "; ".join(parts) if parts else "Plot"


def _img(theme, png, alt):
    width = struct.unpack(">I", png[16:20])[0] * CSS_DPI // DPI  # PNG IHDR width
    data = base64.b64encode(png).decode("ascii")
    return (
        f'<img class="{theme}-only" src="data:image/png;base64,{data}" '
        f'width="{width}" alt="{html.escape(alt)}">'
    )


def _no_grid_on_images(fig):
    # The style's grid is for plots; drawn over an image it hides the data.
    for ax in fig.axes:
        if ax.images:
            ax.grid(False)


def figure_html(fig: Figure) -> str:
    """The light and dark renders of ``fig`` as two theme-switched ``<img>``."""
    _no_grid_on_images(fig)
    alt = _alt(fig)
    light = _png(fig)
    dark = _dark_png(fig)
    return _img("light", light, alt) + "\n" + _img("dark", dark, alt)


def install(shell):
    """Apply the gfem style and display figures as light/dark pairs in ``shell``."""
    apply_style()
    html_formatter = shell.display_formatter.formatters["text/html"]

    def register(*_):
        html_formatter.for_type(Figure, figure_html)

    # Importing pyplot selects the inline backend's formats, which drops every
    # Figure formatter (IPython's select_figure_formats). Registering again
    # after each cell, before the inline backend's own post_execute hook
    # displays the cell's figures, keeps the pair.
    register()
    shell.events.register("pre_run_cell", register)
    shell.events.register("post_execute", register)
