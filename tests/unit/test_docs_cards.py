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

PAGES = sorted(f"physics/{p.stem}" for p in (dp.DOCS / "physics").glob("*.md"))
LINE_WORDS = 20
PAGE_WORDS = 40


@pytest.fixture(scope="module")
def sections():
    return gs.sections(dp.DOCS)


@pytest.fixture(scope="module")
def derived():
    return caps.derive(REPO)


def _by_doc(sections, doc):
    return next(s for s in sections if s["doc"] == doc)


def _png_size(path: Path) -> tuple[int, int]:
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name}: not a PNG"
    return struct.unpack(">II", head[16:24])


def test_there_are_nine_sections(sections):
    assert len(sections) == 9


def test_every_card_target_is_a_page_of_the_site_tree(sections):
    tree = {doc for docs in dp.root_tree().values() for doc in docs}
    for section in sections:
        targets = [
            section["doc"],
            *(c["doc"] for c in section["cards"]),
            *(r["doc"] for r in section["recipes"]),
        ]
        for doc in targets:
            assert doc in tree, f"{section['name']}: {doc} is not in a root toctree"
            assert dp.source(doc).is_file(), f"{doc}: no source file"


def test_every_card_still_is_committed_as_a_16_by_10_pair(sections):
    missing = gs.missing_stills(dp.DOCS, sections)
    assert not missing, f"missing card stills in docs/_static/cards: {missing}"
    for name in gs.still_names(sections):
        for mode in ("light", "dark"):
            width, height = _png_size(
                dp.DOCS / "_static" / "cards" / f"{name}-{mode}.png"
            )
            assert width * 10 == height * 16, f"{name}-{mode}.png is {width}x{height}"


@pytest.mark.parametrize("doc", PAGES)
def test_a_physics_page_is_the_first_entry_of_its_toctree(doc, sections):
    section = _by_doc(sections, doc)
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


def test_every_physics_page_is_a_section(sections):
    on_disk = {f"physics/{p.stem}" for p in (dp.DOCS / "physics").glob("*.md")}
    assert on_disk == {s["doc"] for s in sections}


def test_every_gradient_row_is_a_row_of_the_capabilities_grid(sections):
    for section in sections:
        assert set(section["rows"]) <= set(caps.ROWS), section["doc"]


def test_every_card_line_is_one_plain_sentence(sections):
    lines = [(s["doc"], s["line"]) for s in sections]
    # A promoted recipe has no line of its own; a tutorial card must have one.
    lines += [
        (c["doc"], c["line"])
        for s in sections
        for c in s["cards"]
        if not c["doc"].startswith("recipes/")
    ]
    for doc, line in lines:
        assert line, f"{doc}: no line"
        assert len(line.split()) <= LINE_WORDS, f"{doc}: {len(line.split())} words"
        assert not set("${`") & set(line), f"{doc}: markup in the card line {line!r}"


@pytest.mark.parametrize("doc", PAGES)
def test_a_section_page_is_at_most_40_prose_words(doc, sections):
    section = _by_doc(sections, doc)
    words = dp.prose_words(dp.page_text(dp.source(section["doc"])))
    assert words <= PAGE_WORDS, f"{section['doc']}: {words} words"


def test_a_section_without_a_tutorial_promotes_its_first_recipe(sections):
    for section in sections:
        assert section["cards"], section["doc"]
        if not section["cards"][0]["doc"].startswith("tutorials/"):
            assert section["cards"][0]["doc"].startswith("recipes/")
            assert section["cards"][0]["still"] == section["still"]
            assert not section["cards"][0]["line"]


# The tag of each section on the committed matrix, written out: a change of
# the matrix, the class table or a section's rows must be a visible edit here.
TAGS = {
    "physics/integrated-photonics": "Gradients: shape, materials, sheet fill",
    "physics/nonlinear-and-quantum-optics": "",
    "physics/electro-optics": "Gradients: shape, materials",
    "physics/heat-and-thermo-optics": "Gradients: materials, sheet fill, sources",
    "physics/mechanics-and-acoustics": "Gradients: shape, materials",
    "physics/free-space-scattering": "Gradients: shape, materials, sheet fill",
    "physics/periodic-surfaces": "Gradients: shape, materials, sheet fill",
    "physics/rf-and-microwave": "Gradients: shape, materials, sheet fill",
    "physics/resonators-and-qubits": "Gradients: shape, materials, sheet fill",
}


def test_the_nine_tags_are_the_pinned_ones(sections, derived):
    tagged = {s["doc"]: s["tag"] for s in gs.sections(dp.DOCS, derived)}
    assert tagged == TAGS
    assert set(TAGS) == {s["doc"] for s in sections}


def test_the_tag_of_a_section_is_the_classes_verified_on_its_rows():
    derived = {
        "rows": [
            {
                "key": "a",
                "cells": [
                    {"column": "nodes", "status": "verified"},
                    {"column": "eps_r", "status": "verified"},
                    {"column": "fill", "status": "unverified"},
                ],
            },
            {
                "key": "b",
                "cells": [{"column": "fill", "status": "verified"}],
            },
        ],
        "also": [
            {"key": "a", "inputs": ["lumped values"]},  # the hidden class
            {"key": "b", "inputs": ["loss tangent"]},  # materials, off-grid
            {"key": "d", "inputs": ["thermal conductivity"]},  # another row
        ],
    }
    assert gs.tag(["a"], derived) == "Gradients: shape, materials"
    # A verified cell of another row does not count; a row set adds up in
    # class order; the "unverified" cell and the hidden class add nothing.
    assert gs.tag(["b"], derived) == "Gradients: materials, sheet fill"
    assert gs.tag(["a", "b"], derived) == "Gradients: shape, materials, sheet fill"
    derived["rows"][1]["cells"][0]["status"] = "unverified"
    assert gs.tag(["b"], derived) == "Gradients: materials"
    assert gs.tag(["a"], derived) == "Gradients: shape, materials"
    # A row whose only result is the hidden class has no tag at all.
    only_hidden = {"rows": [], "also": [derived["also"][0]]}
    assert gs.tag(["a"], only_hidden) == ""
    assert gs.tag([], derived) == ""
    assert gs.tag(["c"], derived) == ""


def test_the_class_table_covers_every_matrix_input():
    covered = {name for names in gs.CLASSES.values() for name in names}
    assert covered == set(caps.COLUMNS) | set(caps.OFF_GRID)


# --- sections() on a synthetic site: the errors a bad card must raise -----------


def _site(tmp_path, *, tutorial_md="# Bend\n\nA bend in a waveguide.\n", extra=""):
    """A minimal site: one physics page, one tutorial, one recipe, an index."""
    import json

    (tmp_path / "physics").mkdir()
    (tmp_path / "tutorials").mkdir()
    (tmp_path / "recipes").mkdir()
    (tmp_path / "index.md").write_text(
        "```{toctree}\n:caption: Optics\n\nphysics/optics\ntutorials/bend\n"
        f"recipes/rec\n{extra}```\n",
        encoding="utf-8",
    )
    (tmp_path / "physics" / "optics.md").write_text(
        "---\nlayout: gfem-section\nstill: optics\ngradient_rows:\n---\n"
        "# Optics\n\nLight in glass.\n",
        encoding="utf-8",
    )
    cell = {"cell_type": "markdown", "source": [tutorial_md]}
    (tmp_path / "tutorials" / "bend.ipynb").write_text(
        json.dumps({"cells": [cell]}), encoding="utf-8"
    )
    (tmp_path / "recipes" / "rec.md").write_text(
        "# Recipe\n\nText.\n", encoding="utf-8"
    )
    return tmp_path


def test_a_synthetic_site_builds(tmp_path):
    (section,) = gs.sections(_site(tmp_path))
    assert section["cards"][0]["line"] == "A bend in a waveguide."
    assert section["recipes"] == [{"doc": "recipes/rec", "name": "Recipe"}]


def test_a_tutorial_without_a_first_paragraph_is_an_error(tmp_path):
    site = _site(tmp_path, tutorial_md="# Bend\n\n## Next\n")
    with pytest.raises(gs.SectionsError, match="tutorials/bend"):
        gs.sections(site)


def test_a_tutorial_or_recipe_without_a_heading_is_an_error(tmp_path):
    with pytest.raises(gs.SectionsError, match="tutorials/bend"):
        gs.sections(_site(tmp_path, tutorial_md="no heading\n"))


def test_a_recipe_without_a_heading_is_an_error(tmp_path):
    site = _site(tmp_path)
    (site / "recipes" / "rec.md").write_text("text only\n", encoding="utf-8")
    with pytest.raises(gs.SectionsError, match="recipes/rec"):
        gs.sections(site)


def test_a_physics_entry_outside_tutorials_and_recipes_is_an_error(tmp_path):
    site = _site(tmp_path, extra="theory/forward\n")
    with pytest.raises(gs.SectionsError, match="theory/forward"):
        gs.sections(site)
