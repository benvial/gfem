"""Read the card stills' PNGs and check their palette, with the standard library.

A still is a palette PNG whose saturated entries must lie on the hues of its
colormap's ``gfem_figures.RAMPS`` stops, and whose first pixel is the ground.
These helpers decode the header, the corner pixel and the PLTE entries and
compare hues, so a test can check a committed PNG without rendering it. They use
the standard library only (the test environment has no Pillow, and
``gfem_figures`` imports matplotlib, so the ramps are read from its source with
``ast.literal_eval``).
"""

import ast
import struct
import zlib
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _ramps():
    # gfem_figures imports matplotlib, which the test environment lacks: read
    # the literal table from its source.
    tree = ast.parse((DOCS / "_ext" / "gfem_figures.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "RAMPS"
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("gfem_figures.RAMPS not found")


RAMPS = _ramps()


# --- a PNG read with the standard library --------------------------------------


def _chunks(data):
    assert data[:8] == SIGNATURE, "not a PNG"
    at = 8
    while at < len(data):
        length, kind = struct.unpack(">I4s", data[at : at + 8])
        yield kind, data[at + 8 : at + 8 + length]
        at += 12 + length


def header(path):
    """(width, height) from the IHDR chunk."""
    kind, body = next(iter(_chunks(Path(path).read_bytes())))
    assert kind == b"IHDR"
    return struct.unpack(">II", body[:8])


def corner(path):
    """The first pixel of the first row as an RGB triple.

    Whatever the row's filter, the first pixel of the first row is stored raw
    (the left and up neighbours it is predicted from are zero), so it is the
    bytes after the filter byte.
    """
    chunks = list(_chunks(Path(path).read_bytes()))
    depth, colour = struct.unpack(">BB", chunks[0][1][8:10])
    assert chunks[0][0] == b"IHDR"
    data = zlib.decompress(b"".join(body for kind, body in chunks if kind == b"IDAT"))
    first = data[1:]
    if colour == 3:  # palette
        palette = next(body for kind, body in chunks if kind == b"PLTE")
        index = first[0] >> (8 - depth) if depth < 8 else first[0]
        return tuple(palette[3 * index : 3 * index + 3])
    assert depth == 8, "only 8-bit RGB, RGBA and palette PNGs are read"
    if colour in (2, 6):
        return tuple(first[:3])
    raise AssertionError(f"PNG colour type {colour} is not read")


def palette(path):
    """The PLTE entries of a palette PNG as RGB triples."""
    chunks = list(_chunks(Path(path).read_bytes()))
    assert struct.unpack(">B", chunks[0][1][9:10])[0] == 3, (
        "a card still is a palette PNG"
    )
    body = next(body for kind, body in chunks if kind == b"PLTE")
    return [tuple(body[i : i + 3]) for i in range(0, len(body), 3)]


# An entry is a colour (not a grey: the ground, the bodies and the anti-aliased
# edges between them) when its channels span more than 40 of 255. The palette's
# greys are exact (span 0) and the quantiser moves a channel by a few units at
# most, so 40 is far above a grey's span and below any mix of a stop with
# enough ground to keep its hue.
SATURATED = 40
# An antialiased or dithered pixel between a stop and the grey ground is a
# mix of two colours one of which has no hue, so it keeps the stop's hue; the
# quantiser's rounding to 8 bits moves a hue by well under 1 deg at saturation
# 40. The measured entries of the five committed stills sit within 15 deg of a
# stop; 15 deg is also far below the 90 deg between the red and the blue stops.
HUE_TOLERANCE = 15.0  # deg


def hue(rgb):
    """The hue in degrees of an RGB triple (0-255), from the standard hexcone."""
    r, g, b = (float(c) for c in rgb)
    top, bottom = max(r, g, b), min(r, g, b)
    span = top - bottom
    if top == r:
        angle = ((g - b) / span) % 6
    elif top == g:
        angle = (b - r) / span + 2
    else:
        angle = (r - g) / span + 4
    return 60.0 * angle


def ramp_hues(name):
    """The hues of the saturated stops of a ``RAMPS`` entry."""
    stops = [tuple(int(h[i : i + 2], 16) for i in (1, 3, 5)) for h in RAMPS[name]]
    return [hue(c) for c in stops if max(c) - min(c) > SATURATED]


def off_ramp(entries, hues):
    """The saturated palette entries farther than the tolerance from every hue."""
    stray = []
    for rgb in entries:
        if max(rgb) - min(rgb) <= SATURATED:
            continue
        gaps = [min(abs(hue(rgb) - h), 360 - abs(hue(rgb) - h)) for h in hues]
        if min(gaps) > HUE_TOLERANCE:
            stray.append(rgb)
    return stray
