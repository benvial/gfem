"""The card stills on disk are the ones their sidecars record.

``docs/_static/cards/<still>-{light,dark}.png`` and ``<still>.json`` are
written by ``just stills`` (docs/_ext/gfem_stills.py). This reads files only:
it renders nothing and needs neither PyVista nor Pillow, so the PNG header and
the corner pixel are decoded with the standard library. It checks that the
files come in complete sets, that each sidecar draws from the site's palette
(the ramps of ``gfem_figures.RAMPS``, the bodies' grey and the ground of the
theme), that the PNG is the size and the ground the sidecar says, that every
tutorial has exactly one ``card`` cell and a script, and the pure helpers that
frame a still.
"""

import ast
import importlib.util
import json
import re
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
CARDS = DOCS / "_static" / "cards"
TUTORIALS = sorted((DOCS / "tutorials").glob("*.ipynb"))
SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, DOCS / "_ext" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stills = _module("gfem_stills")
scenes = _module("gfem_scenes")


def _ramps():
    # gfem_figures imports matplotlib, which the test environment lacks: read
    # the literal table from its source.
    tree = ast.parse((DOCS / "_ext" / "gfem_figures.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "RAMPS":
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


def _png(rows, width, colour, palette=None, *, depth=8, filter_byte=0):
    raw = b"".join(
        bytes([filter_byte if i == 0 else 0]) + row for i, row in enumerate(rows)
    )
    ihdr = struct.pack(">IIBBBBB", width, len(rows), depth, colour, 0, 0, 0)

    def chunk(kind, body):
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body))
        )

    out = SIGNATURE + chunk(b"IHDR", ihdr)
    if palette:
        out += chunk(b"PLTE", palette)
    return out + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


@pytest.mark.parametrize("filter_byte", range(5))
def test_the_corner_reader_ignores_the_filter_of_the_first_row(tmp_path, filter_byte):
    # Every filter predicts from neighbours that are zero on the first pixel of
    # the first row, so its stored byte is the pixel.
    palette = bytes([10, 20, 30, 242, 242, 242])
    path = tmp_path / "p.png"
    path.write_bytes(
        _png([bytes([1, 0]), bytes([0, 0])], 2, 3, palette, filter_byte=filter_byte)
    )
    assert corner(path) == (242, 242, 242)
    path.write_bytes(_png([bytes([5, 6, 7, 8, 9, 10])], 2, 2, filter_byte=filter_byte))
    assert corner(path) == (5, 6, 7)
    path.write_bytes(
        _png([bytes([5, 6, 7, 255, 9, 10, 11, 255])], 2, 6, filter_byte=filter_byte)
    )
    assert corner(path) == (5, 6, 7)
    assert header(path) == (2, 1)


def test_the_corner_reader_unpacks_a_palette_of_few_colours(tmp_path):
    # 2-bit pixels: the first pixel is the top two bits of the first byte.
    palette = bytes([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    path = tmp_path / "p.png"
    path.write_bytes(_png([bytes([0b10_00_01_11])], 4, 3, palette, depth=2))
    assert corner(path) == (4, 5, 6)


# --- the frame of a still ----------------------------------------------------------


def _scene(shape=(160, 300), box=(40, 30, 120, 90), ground=(242, 242, 242)):
    image = np.empty((*shape, 3), dtype=np.uint8)
    image[:] = ground
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1] = (181, 38, 47)
    return image


def test_the_content_box_is_the_pixels_that_differ_from_the_ground():
    image = _scene()
    image[0, 0] = (
        244,
        242,
        242,
    )  # within the tolerance of the ground: round-off of a render
    assert stills.content_box(image, (242, 242, 242)) == (40, 30, 120, 90)


def test_a_render_of_nothing_but_the_ground_is_an_error():
    with pytest.raises(ValueError, match="nothing"):
        stills.content_box(_scene(box=(0, 0, 0, 0)), (242, 242, 242))


@pytest.mark.parametrize(
    "box",
    [(40, 30, 120, 90), (0, 0, 1000, 100), (10, 10, 20, 400), (5, 5, 6, 6)],
)
def test_the_crop_box_is_16_by_10_with_the_margin_and_contains_the_content(box):
    left, top, right, bottom = stills.crop_box(box)
    width, height = right - left, bottom - top
    assert width * 10 == height * 16  # integers: the ratio is exact
    x0, y0, x1, y1 = box
    # The content fits the frame shrunk by the margin on each side. Rounding
    # the crop's corners to whole pixels moves each edge by at most half a
    # pixel, so the check allows one pixel in all.
    shrink = 1 - 2 * stills.MARGIN
    assert x1 - x0 <= shrink * width + 1
    assert y1 - y0 <= shrink * height + 1
    assert (left, top) <= (x0, y0)
    assert (right, bottom) >= (x1, y1)
    assert (left + right) / 2 == pytest.approx((x0 + x1) / 2, abs=1)
    assert (top + bottom) / 2 == pytest.approx((y0 + y1) / 2, abs=1)


def test_the_tall_content_sets_the_crop_height_and_the_wide_content_its_width():
    wide = stills.crop_box((0, 0, 1600, 100))
    tall = stills.crop_box((0, 0, 100, 1000))
    assert wide[2] - wide[0] == pytest.approx(1600 / (1 - 2 * stills.MARGIN), abs=8)
    assert tall[3] - tall[1] == pytest.approx(1000 / (1 - 2 * stills.MARGIN), abs=8)


def test_the_anchor_index_hashes_the_source_the_kernel_sees(tmp_path):
    import hashlib

    def cell(source, tags=()):
        return {"cell_type": "code", "source": source, "metadata": {"tags": list(tags)}}

    notebook = {
        "cells": [cell(["a = 1\n", "b = 2"]), cell(["plot()\n", "show()"], ["card"])]
    }
    (tmp_path / "one.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    expected = hashlib.sha256(b"plot()\nshow()").hexdigest()
    assert stills.anchors(tmp_path) == {expected: "one"}


# --- the files -----------------------------------------------------------------------


def _stems():
    files = {"light": set(), "dark": set(), "json": set()}
    for path in CARDS.glob("*"):
        if path.suffix == ".json":
            files["json"].add(path.stem)
        elif path.name.endswith(("-light.png", "-dark.png")):
            files[path.stem.rsplit("-", 1)[1]].add(path.stem.rsplit("-", 1)[0])
        else:
            pytest.fail(f"{path.name} is neither a still nor its sidecar")
    return files


def test_every_still_has_both_pngs_and_a_sidecar_and_nothing_is_left_over():
    files = _stems()
    assert files["json"], "no still found"
    assert files["light"] == files["json"]
    assert files["dark"] == files["json"]


SIDECARS = sorted(CARDS.glob("*.json"))


@pytest.mark.parametrize("path", SIDECARS, ids=lambda p: p.stem)
def test_a_sidecar_draws_from_the_sites_palette(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["still"] == path.stem
    assert record["kind"] in stills.KINDS
    assert record["size"] == list(stills.SIZE)
    for theme in stills.THEMES:
        side = record[theme]
        suffix = "_dark" if theme == "dark" else ""
        ramp = f"gfem_{record['kind']}{suffix}"
        assert ramp in RAMPS, f"{ramp} is not in gfem_figures.RAMPS"
        assert side["colormaps"], f"the {theme} still draws no field"
        assert set(side["colormaps"]) <= {ramp}
        assert set(side["solids"]) <= {stills.BODIES[theme]}
        assert side["ground"] == scenes.THEMES[theme]["ground"]


@pytest.mark.parametrize("path", SIDECARS, ids=lambda p: p.stem)
def test_a_sidecar_camera_is_the_preset_unless_it_declares_an_override(path):
    camera = json.loads(path.read_text(encoding="utf-8"))["camera"]
    record = json.loads(path.read_text(encoding="utf-8"))
    preset = stills.PRESET
    assert camera["view_angle_deg"] == preset["view_angle_deg"]
    assert camera["up"] == preset["up"]
    if record.get("camera_override"):  # only the direction may change
        return
    assert camera == preset


@pytest.mark.parametrize("path", SIDECARS, ids=lambda p: p.stem)
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_png_is_the_size_and_ground_its_sidecar_records(path, theme):
    record = json.loads(path.read_text(encoding="utf-8"))
    png = CARDS / f"{path.stem}-{theme}.png"
    assert list(header(png)) == record["size"] == list(stills.SIZE)
    ground = record[theme]["ground"]
    assert corner(png) == tuple(int(ground[i : i + 2], 16) for i in (1, 3, 5))


# --- the scripts and anchors ---------------------------------------------------


def _anchors(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return [c for c in cells if "card" in c.get("metadata", {}).get("tags", ())]


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda p: p.stem)
def test_every_tutorial_has_exactly_one_card_anchor_that_is_an_executed_code_cell(path):
    anchors = _anchors(path)
    assert len(anchors) == 1, f"{path.name} has {len(anchors)} cells tagged card"
    assert anchors[0]["cell_type"] == "code"
    assert anchors[0]["execution_count"] is not None


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda p: p.stem)
def test_every_tutorial_has_a_card_script_defining_card(path):
    script = DOCS / "_stills" / f"{path.stem}.py"
    assert script.is_file(), f"docs/_stills/{path.stem}.py is missing"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "card" in functions, f"{script.name} defines no card(ns)"
    assert [a.arg for a in functions["card"].args.args] in (["ns"], ["_ns"])


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda p: p.stem)
def test_every_tutorial_has_a_still(path):
    for suffix in ("-light.png", "-dark.png", ".json"):
        assert (CARDS / f"{path.stem}{suffix}").is_file(), (
            f"{path.stem}{suffix} is missing"
        )


def test_no_two_tutorials_tag_the_same_source():
    # The kernel recognises the card cell by the hash of its source alone.
    sources = [_anchors(p)[0]["source"] for p in TUTORIALS if _anchors(p)]
    assert len({"".join(s) for s in sources}) == len(sources)


def _section_stills():
    stills_named = set()
    for page in sorted((DOCS / "physics").glob("*.md")):
        found = re.search(
            r"^still:\s*(\S+)\s*$", page.read_text(encoding="utf-8"), re.MULTILINE
        )
        if found:
            stills_named.add(found.group(1).strip("'\""))
    return stills_named


def test_every_card_script_is_anchored_or_a_section_still():
    stems = {p.stem for p in TUTORIALS}
    scripts = sorted(p.stem for p in (DOCS / "_stills").glob("*.py"))
    assert scripts
    standalone = [s for s in scripts if s not in stems]
    named = _section_stills()
    for stem in standalone:
        assert (CARDS / f"{stem}.json").is_file(), f"{stem}.py has no still"
        if (DOCS / "physics").is_dir():  # the section pages name the standalone stills
            assert stem in named, (
                f"{stem}.py is neither anchored by a tutorial nor a section still"
            )
