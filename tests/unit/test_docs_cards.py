"""The physics cards: the section pages, the landing grid's data and the stills.

The cards are derived from the site tree by docs/_ext/gfem_sections.py,
imported without Sphinx. A card can only go dead through a bad toctree entry
or a missing file, so these checks read text and PNG headers only (stdlib,
no image library).
"""

import importlib.util
import struct
import sys
from pathlib import Path

import docs_pages as dp
import pytest

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "docs" / "_ext"))
_spec = importlib.util.spec_from_file_location(
    "gfem_sections", REPO / "docs" / "_ext" / "gfem_sections.py"
)
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)

import gfem_capabilities as caps  # noqa: E402  (docs/_ext is on sys.path above)

SECTIONS = gs.sections(dp.DOCS)
LINE_WORDS = 20
PAGE_WORDS = 40


@pytest.fixture(scope="module")
def derived():
    return caps.derive(REPO)


def _png_size(path: Path) -> tuple[int, int]:
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name}: not a PNG"
    return struct.unpack(">II", head[16:24])


def test_there_are_nine_sections():
    assert len(SECTIONS) == 9


def test_every_card_target_is_a_page_of_the_site_tree():
    tree = {doc for docs in dp.root_tree().values() for doc in docs}
    for section in SECTIONS:
        targets = [
            section["doc"],
            *(c["doc"] for c in section["cards"]),
            *(r["doc"] for r in section["recipes"]),
        ]
        for doc in targets:
            assert doc in tree, f"{section['name']}: {doc} is not in a root toctree"
            assert dp.source(doc).is_file(), f"{doc}: no source file"


def test_every_card_still_is_committed_as_a_16_by_10_pair():
    missing = gs.missing_stills(dp.DOCS, SECTIONS)
    assert not missing, f"missing card stills in docs/_static/cards: {missing}"
    for name in gs.still_names(SECTIONS):
        for mode in ("light", "dark"):
            width, height = _png_size(
                dp.DOCS / "_static" / "cards" / f"{name}-{mode}.png"
            )
            assert width * 10 == height * 16, f"{name}-{mode}.png is {width}x{height}"


@pytest.mark.parametrize("section", SECTIONS, ids=lambda s: s["doc"])
def test_a_physics_page_is_the_first_entry_of_its_toctree(section):
    caption, entries = next(
        (c, e) for c, e in dp.root_tree().items() if section["doc"] in e
    )
    assert entries[0] == section["doc"]
    assert (
        caption == section["name"] == dp.title(dp.page_text(dp.source(section["doc"])))
    )
    text = dp.source(section["doc"]).read_text(encoding="utf-8")
    meta = gs._front_matter(text)
    assert meta["layout"] == "gfem-section"
    assert meta["still"]
    assert "gradient_rows" in meta


def test_every_physics_page_is_a_section():
    on_disk = {f"physics/{p.stem}" for p in (dp.DOCS / "physics").glob("*.md")}
    assert on_disk == {s["doc"] for s in SECTIONS}


def test_every_gradient_row_is_a_row_of_the_capabilities_grid():
    for section in SECTIONS:
        assert set(section["rows"]) <= set(caps.ROWS), section["doc"]


def test_every_card_line_is_one_plain_sentence():
    lines = [(s["doc"], s["line"]) for s in SECTIONS]
    lines += [(c["doc"], c["line"]) for s in SECTIONS for c in s["cards"] if c["line"]]
    for doc, line in lines:
        assert line, f"{doc}: no line"
        assert len(line.split()) <= LINE_WORDS, f"{doc}: {len(line.split())} words"
        assert not set("${`") & set(line), f"{doc}: markup in the card line {line!r}"


@pytest.mark.parametrize("section", SECTIONS, ids=lambda s: s["doc"])
def test_a_section_page_is_at_most_40_prose_words(section):
    words = dp.prose_words(dp.page_text(dp.source(section["doc"])))
    assert words <= PAGE_WORDS, f"{section['doc']}: {words} words"


def test_a_section_without_a_tutorial_promotes_its_first_recipe():
    for section in SECTIONS:
        assert section["cards"], section["doc"]
        if not section["cards"][0]["doc"].startswith("tutorials/"):
            assert section["cards"][0]["doc"].startswith("recipes/")
            assert section["cards"][0]["still"] == section["still"]
            assert not section["cards"][0]["line"]


def test_the_tags_are_derived_from_the_matrix_rows(derived):
    tagged = gs.sections(dp.DOCS, derived)
    for section in tagged:
        assert section["tag"] == gs.tag(section["rows"], derived), section["doc"]
        assert bool(section["tag"]) == bool(section["rows"]), section["doc"]
        assert section["tag"] == "" or section["tag"].startswith("Gradients: ")


def test_the_class_table_covers_every_matrix_input():
    covered = {name for names in gs.CLASSES.values() for name in names}
    assert covered == set(caps.COLUMNS) | set(caps.OFF_GRID)
