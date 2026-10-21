# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fonttools",
#   "numpy",
#   "pillow",
#   "scipy",
#   "uharfbuzz",
# ]
# ///
"""Build the gfem logo asset set and social card into ``docs/_static/logo/``.

The mark is made the way gfem's optimizer regularizes a design: draw the members (a
nabla frame and a web of struts), apply a density filter (Gaussian blur), project
(threshold), then trace. The rounded corners and the fillets where struts meet the
frame are real filter-radius fillets. The wordmark is lowercase ``gfem`` outlined from
Barlow SemiBold; the social card adds the positioning line in Barlow Regular.

This script is the only source of the logo: nothing is hand-edited after generation,
and it is not run in CI. ``site.webmanifest`` next to the outputs is hand-written and
left untouched.

Run from anywhere::

    uv run tools/build_logo.py [--font PATH] [--text-font PATH] [--out DIR]

Python dependencies are declared inline (PEP 723). The command-line tools must be on
``PATH``: potrace, rsvg-convert (librsvg), magick (ImageMagick 7) and optipng.

Fonts: Barlow by The Barlow Project Authors, SIL Open Font License 1.1
(https://github.com/jpt/barlow, also on Google Fonts). The TTFs and their ``OFL.txt``
are committed under ``docs/_mpl/fonts/``; ``--font`` / ``GFEM_LOGO_FONT`` and
``--text-font`` / ``GFEM_LOGO_TEXT_FONT`` override them.
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import uharfbuzz as hb
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "_static" / "logo"
FONTS = REPO / "docs" / "_mpl" / "fonts"
FONT = FONTS / "Barlow-SemiBold.ttf"  # wordmark and lockup
TEXT_FONT = FONTS / "Barlow-Regular.ttf"  # social card headline (Regular Display Rule)

RATIO = 0.78  # height / width of the nabla

# Ink colours. Ink only: the accent belongs to the site, not the logo.
INK_LIGHT_BG = "#16181d"  # mark on light backgrounds
INK_DARK_BG = "#e6e8eb"  # mark on dark backgrounds
GROUND = "#ffffff"

# Social card (Open Graph, GitHub social preview).
OG_SIZE = (1280, 640)
OG_SAFE_WIDTH = 1100  # inside the 1222 px a 1.91:1 crop keeps
OG_HEADLINE = (
    "Gradients of full-wave FEM results with respect to your geometry and materials, "
    "in Python."
)


# --- geometry ---------------------------------------------------------------


def seg_dist(X, Y, a, b):
    """Return the distance from grid points to the segment a-b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = np.clip(((X - ax) * dx + (Y - ay) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.hypot(X - ax - t * dx, Y - ay - t * dy)


def truss_density(N, side, frame_w, strut_w, junction=0.45):
    """Return the material density of the drawn members on an N x N grid (100 box).

    Pixel centres are symmetric about x = 50, so the design is exactly
    mirror-symmetric.
    """
    c = (np.arange(N) + 0.5) * 100 / N
    X, Y = np.meshgrid(c, c)
    h = side * RATIO
    top = 50 - h / 2
    P = [(50 - side / 2, top), (50 + side / 2, top), (50, top + h)]
    d = np.minimum.reduce([seg_dist(X, Y, P[i], P[(i + 1) % 3]) for i in range(3)])
    rho = d <= frame_w / 2
    J = (50, top + h * junction)
    for a, b in [(P[0], J), (P[1], J), (J, P[2])]:
        rho |= seg_dist(X, Y, a, b) <= strut_w / 2
    return rho.astype(float)


def filter_project(rho, radius, eta=0.42):
    """Apply a density filter (Gaussian, radius in box units), then project."""
    N = rho.shape[0]
    return gaussian_filter(rho, radius * N / 100) > eta


# --- external tools ---------------------------------------------------------


def tool(name):
    """Return the absolute path of a required command-line tool."""
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"build_logo.py needs `{name}` on PATH")
    return path


def run(*args, **kwargs):
    """Run a command-line tool, failing loudly."""
    return subprocess.run([tool(args[0]), *args[1:]], check=True, **kwargs)


# --- tracing ----------------------------------------------------------------

_TOKEN = re.compile(r"[MmCcLlZz]|-?\d+(?:\.\d+)?")
_ARITY = {"M": 2, "m": 2, "L": 2, "l": 2, "C": 6, "c": 6}


def _absolute(dstr, xf):
    """Rewrite one potrace path (M/L/C, relative or absolute) as absolute commands."""
    toks = _TOKEN.findall(dstr)
    parts = []
    i, cmd, cur, start = 0, "M", (0.0, 0.0), (0.0, 0.0)
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                parts.append("Z")
                cur = start
            continue
        n = _ARITY[cmd]
        v = [float(x) for x in toks[i : i + n]]
        i += n
        if cmd.islower():
            v = [v[j] + cur[j % 2] for j in range(n)]
        pts = " ".join(
            f"{x:.2f} {y:.2f}" for x, y in (xf(v[j], v[j + 1]) for j in range(0, n, 2))
        )
        cur = (v[-2], v[-1])
        if cmd in "Mm":
            start = cur
            parts.append("M" + pts)
            cmd = "l" if cmd == "m" else "L"
        else:
            parts.append(cmd.upper() + pts)
    return "".join(parts)


def trace(mask):
    """Trace a boolean mask with potrace; return an absolute path in 100-box units."""
    N = mask.shape[0]
    with tempfile.TemporaryDirectory() as d:
        pbm, out = Path(d) / "m.pbm", Path(d) / "m.svg"
        img = Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).convert("1")
        img.save(pbm)
        run(
            "potrace", str(pbm), "-s", "-o", str(out), "--alphamax", "1.0",
            "--opttolerance", "0.2", "--turdsize", "8",
        )  # fmt: skip
        s = out.read_text()
    # potrace: <g transform="translate(0,H) scale(0.1,-0.1)">, path in 0.1 px, y up
    k = 100 / N

    def xf(x, y):
        return x * 0.1 * k, (N - y * 0.1) * k

    return "".join(_absolute(dstr, xf) for dstr in re.findall(r'<path d="([^"]+)"', s))


def path_bounds(d):
    """Return the control-point bounds (x0, y0, x1, y1) of an absolute path."""
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", d)]
    xs, ys = nums[0::2], nums[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def shift(d, dx, dy):
    """Translate an absolute path (M/L/C only) by (dx, dy)."""
    k = 0

    def sub(mt):
        nonlocal k
        v = float(mt.group(0)) + (dx if k % 2 == 0 else dy)
        k += 1
        return f"{v:.2f}"

    return re.sub(r"-?\d+(?:\.\d+)?", sub, d)


# --- type -------------------------------------------------------------------


def _ntos(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def wordmark(font, text="gfem"):
    """Outline ``text`` with plain advances, no kerning (the wordmark).

    Returns the path in font units with y down and the baseline at 0, its bounds,
    and the cap height.
    """
    f = TTFont(font)
    gs, cmap = f.getGlyphSet(), f.getBestCmap()
    pen = SVGPathPen(gs, ntos=_ntos)
    bp = BoundsPen(gs)
    x = 0
    for ch in text:
        g = gs[cmap[ord(ch)]]
        g.draw(TransformPen(pen, (1, 0, 0, -1, x, 0)))  # y down, baseline at 0
        g.draw(TransformPen(bp, (1, 0, 0, -1, x, 0)))
        x += g.width
    return pen.getCommands(), bp.bounds, f["OS/2"].sCapHeight


class Typesetter:
    """Shape running text with HarfBuzz (kerning on) and outline it."""

    def __init__(self, font):
        """Load ``font`` for both shaping and outlines."""
        self.tt = TTFont(font)
        self.gs = self.tt.getGlyphSet()
        self.upem = self.tt["head"].unitsPerEm
        self.cap = self.tt["OS/2"].sCapHeight
        self.hb = hb.Font(hb.Face(hb.Blob.from_file_path(str(font))))

    def _shape(self, text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb, buf, {"kern": True, "liga": True})
        return buf.glyph_infos, buf.glyph_positions

    def width(self, text):
        """Return the advance width of ``text`` in font units."""
        return sum(p.x_advance for p in self._shape(text)[1])

    def outline(self, text):
        """Return ``text`` as one path in font units, y down, baseline at 0."""
        pen = SVGPathPen(self.gs, ntos=_ntos)
        x = 0
        for info, pos in zip(*self._shape(text), strict=True):
            g = self.gs[self.tt.getGlyphName(info.codepoint)]
            g.draw(TransformPen(pen, (1, 0, 0, -1, x + pos.x_offset, -pos.y_offset)))
            x += pos.x_advance
        return pen.getCommands()

    def balance(self, text):
        """Split ``text`` into the two lines whose longer line is shortest."""
        words = text.split()
        splits = [
            (" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))
        ]
        return min(splits, key=lambda ab: max(self.width(ab[0]), self.width(ab[1])))


# --- svg and png ------------------------------------------------------------


def svg(view, body, title="gfem", fill="currentColor"):
    """Wrap ``body`` in an accessible, tight SVG."""
    vx, vy, vw, vh = view
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vx:g} {vy:g} {vw:g} {vh:g}"'
        f' role="img" aria-label="{title}"><title>{title}</title>'
        f'<g fill="{fill}">{body}</g></svg>\n'
    )


def fmt(v):
    """Round to the 2 decimals the paths use."""
    return float(f"{v:.2f}")


def rasterize(svg_text, w, h=None, bg=None):
    """Render SVG text to PNG bytes with rsvg-convert."""
    args = ["rsvg-convert", "-w", str(w)]
    args += ["-h", str(h)] if h else []
    args += ["-b", bg] if bg else []
    return run(*args, "-f", "png", input=svg_text.encode(), capture_output=True).stdout


def png(path, data):
    """Write PNG bytes and optimize them losslessly."""
    path.write_bytes(data)
    run("optipng", "-quiet", "-o5", str(path))


def alpha_png(a, color):
    """Return an RGBA image of ``color`` with coverage ``a`` as alpha."""
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    h, w = a.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, g, b
    rgba[..., 3] = np.round(a * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


# --- masters ----------------------------------------------------------------

# Large master: the converged truss chosen in the logo-directions round (R6).
# Used for the mark, the lockup and every raster from 48 px up.
LARGE = dict(side=86, frame_w=9, strut_w=6, radius=2.0, junction=0.45)

# Small master for favicon.svg (drawn by the browser at 16 and 32 px): the same
# construction with heavier members, the same filter radius so the fillets survive,
# and the junction raised. Fills the 100 box edge to edge; the top edge is shifted to
# land on a pixel row at 16 and 32 px.
SMALL = dict(side=97, frame_w=13, strut_w=7, radius=2.0, junction=0.42)
SMALL_TOP = 6.25  # one 16 px pixel

# 16 px master for favicon.ico, hand-tuned pixel by pixel. Started from the micro
# geometry (frame 13, struts 6, filter 1.0, junction 0.38 on a 16 px grid, interior
# voids snapped to whole pixels), then cleaned: the stray top row dropped, the outer
# top corners softened to echo the filter fillets. Digits are ink coverage in tenths
# ('#' full, ' ' none).
FAVICON_16 = [
    "                ",
    "6##############6",
    "################",
    "####        ####",
    "3#####    #####3",
    " 7## ##  ## ##7 ",
    " 1##   ##   ##1 ",
    "  4##  ##  ##4  ",
    "   8## ## ##8   ",
    "   2## ## ##2   ",
    "    5######5    ",
    "     8####8     ",
    "     2####2     ",
    "      6##6      ",
    "       88       ",
    "                ",
]


def alpha_map(rows):
    """Turn the hand-tuned coverage map into an alpha array."""
    level = {" ": 0.0, "#": 1.0}
    return np.array(
        [[level[c] if c in level else int(c) / 10 for c in r] for r in rows]
    )


def master(p, N=2400):
    """Draw, filter, project and trace one master."""
    rho = truss_density(N, p["side"], p["frame_w"], p["strut_w"], p["junction"])
    return trace(filter_project(rho, p["radius"]))


# --- build ------------------------------------------------------------------

THEMES = {  # file suffix: fill
    "": "currentColor",  # one colour, inherits; for inline SVG and CSS mask use
    "-light": INK_LIGHT_BG,  # dark ink, for light backgrounds
    "-dark": INK_DARK_BG,  # light ink, for dark backgrounds
}


def social_card(lockup_body, lockup_view, text_font):
    """Return the 1280 x 640 social card SVG: light lockup over the headline."""
    W, H = OG_SIZE
    ts = Typesetter(text_font)
    lines = ts.balance(OG_HEADLINE)
    size = 40  # headline px
    k_text = size / ts.upem
    widest = max(ts.width(line) for line in lines) * k_text
    if widest > OG_SAFE_WIDTH:
        raise SystemExit(f"headline is {widest:.0f} px wide, over {OG_SAFE_WIDTH}")
    lead = 1.35 * size
    lockup_h = 132  # px
    gap = 84  # lockup bottom to the first headline cap line
    _, vy, vw, vh = lockup_view
    k_lock = lockup_h / vh
    block = lockup_h + gap + ts.cap * k_text + lead * (len(lines) - 1)
    top = (H - block) / 2
    parts = [
        f'<rect width="{W}" height="{H}" fill="{GROUND}"/>',
    ]
    lx, ly = (W - vw * k_lock) / 2, top - vy * k_lock
    parts.append(
        f'<g fill="{INK_LIGHT_BG}" transform="translate({lx:.2f} {ly:.2f})'
        f' scale({k_lock:.5f})">{lockup_body}</g>'
    )
    baseline = top + lockup_h + gap + ts.cap * k_text
    for line in lines:
        x = (W - ts.width(line) * k_text) / 2
        parts.append(
            f'<path fill="{INK_LIGHT_BG}" transform="translate({x:.2f} {baseline:.2f})'
            f' scale({k_text:.5f})" d="{ts.outline(line)}"/>'
        )
        baseline += lead
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
        + "".join(parts)
        + "</svg>\n"
    )


def build(out, font, text_font):
    """Write the whole asset set into ``out``."""
    out.mkdir(parents=True, exist_ok=True)

    def write(name, text):
        (out / name).write_text(text)

    # mark: tight viewBox
    d = master(LARGE)
    x0, y0, x1, y1 = path_bounds(d)
    d = shift(d, -x0, -y0)
    mw, mh = fmt(x1 - x0), fmt(y1 - y0)
    for suf, fill in THEMES.items():
        write(f"gfem-mark{suf}.svg", svg((0, 0, mw, mh), f'<path d="{d}"/>', fill=fill))

    # wordmark: font units, y down, baseline at 0; tight viewBox
    wd, (wx0, wy0, wx1, wy1), cap = wordmark(font)
    for suf, fill in THEMES.items():
        view = (wx0, wy0, wx1 - wx0, wy1 - wy0)
        write(f"gfem-wordmark{suf}.svg", svg(view, f'<path d="{wd}"/>', fill=fill))

    # lockup, in font units: the mark's flat top sits on the f ascender (cap height),
    # its tip overshoots the baseline like a round letter, and the gap is a little
    # over one stem.
    over = 12
    s = (cap + over) / mh
    gap = 150
    lw = mw * s + gap
    body = (
        f'<path transform="translate(0 {-cap}) scale({s:.4f})" d="{d}"/>'
        f'<path transform="translate({lw:.1f} 0)" d="{wd}"/>'
    )
    lockup_view = (0, -cap, round(lw + wx1), round(wy1 + cap))
    for suf, fill in THEMES.items():
        write(f"gfem-lockup{suf}.svg", svg(lockup_view, body, fill=fill))

    # favicon.svg: small master, adapts to the browser's colour scheme
    fd = master(SMALL)
    _, fy0, _, _ = path_bounds(fd)
    fd = shift(fd, 0, SMALL_TOP - fy0)
    write(
        "favicon.svg",
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        f"<style>path{{fill:{INK_LIGHT_BG}}}@media (prefers-color-scheme:dark)"
        f"{{path{{fill:{INK_DARK_BG}}}}}</style>"
        f'<path d="{fd}"/></svg>\n',
    )
    small = svg((0, 0, 100, 100), f'<path d="{fd}"/>', fill=INK_LIGHT_BG)
    centred = f"translate({(100 - mw) / 2:.2f} {(100 - mh) / 2:.2f})"
    big = svg(
        (0, 0, 100, 100), f'<path transform="{centred}" d="{d}"/>', fill=INK_LIGHT_BG
    )

    # rasters: 16 px hand-tuned, 32 and 48 from the small master, larger from the
    # large master. The 16/32/48 PNGs only feed the ICO, so they stay temporary.
    for z in (192, 512):
        png(out / f"icon-{z}.png", rasterize(big, z, z))
    # apple-touch-icon: opaque ink tile, light mark at about 62 % width, raised
    k = 62 / mw
    tile = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        f'<rect width="100" height="100" fill="{INK_LIGHT_BG}"/>'
        f'<path fill="{INK_DARK_BG}" transform="translate({(100 - mw * k) / 2:.2f} '
        f'{(100 - mh * k) / 2 - 2:.2f}) scale({k:.4f})" d="{d}"/></svg>'
    )
    png(out / "apple-touch-icon.png", rasterize(tile, 180, 180))
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        alpha_png(alpha_map(FAVICON_16), INK_LIGHT_BG).save(t / "favicon-16.png")
        png(t / "favicon-32.png", rasterize(small, 32, 32))
        png(t / "favicon-48.png", rasterize(small, 48, 48))
        sizes = [str(t / f"favicon-{z}.png") for z in (16, 32, 48)]
        run("magick", *sizes, str(out / "favicon.ico"))

    # social card: opaque, 2:1
    W, H = OG_SIZE
    card = social_card(body, lockup_view, text_font)
    png(out / "gfem-og.png", rasterize(card, W, H, bg=GROUND))


def main(argv=None):
    """Parse the command line and build the assets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--font",
        type=Path,
        default=Path(os.environ.get("GFEM_LOGO_FONT", FONT)),
        help="Barlow SemiBold TTF for the wordmark (default: %(default)s)",
    )
    parser.add_argument(
        "--text-font",
        type=Path,
        default=Path(os.environ.get("GFEM_LOGO_TEXT_FONT", TEXT_FONT)),
        help="Barlow Regular TTF for the social card headline (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT, help="output directory (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    for f in (args.font, args.text_font):
        if not f.is_file():
            parser.error(f"font not found: {f} (Barlow, SIL OFL 1.1, Google Fonts)")
    build(args.out, args.font, args.text_font)


if __name__ == "__main__":
    main()
