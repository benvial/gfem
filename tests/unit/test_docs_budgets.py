"""Every user page stays within the docs budgets.

A docs budget is a ceiling on one kind of user-page content: a sidebar
section, a page title, a recipe, a tutorial, a release-note entry, a glossary
entry. A page over budget is cut, then split by page type; the budget is not
raised and there is no allowlist. The checks read text only, so they run in
milliseconds and import nothing of gfem.
"""

import json
import re

import docs_pages as dp
import pytest

TREE = dp.root_tree()
DOCS = [doc for docs in TREE.values() for doc in docs]
CUT = "cut, then split by page type; do not raise the budget"


@pytest.mark.parametrize("caption", list(TREE), ids=str)
def test_a_sidebar_section_holds_at_most_ten_pages(caption):
    assert caption, "a root toctree without a caption is not a sidebar section"
    assert len(TREE[caption]) <= 10, f"{caption}: {len(TREE[caption])} pages; {CUT}"


@pytest.mark.parametrize("doc", DOCS)
def test_a_page_title_is_at_most_35_characters(doc):
    title = dp.title(dp.page_text(dp.source(doc)))
    assert title, f"{doc}: no first-level heading"
    assert len(title) <= 35, f"{doc}: {title!r} is {len(title)} characters; {CUT}"


@pytest.mark.parametrize("doc", [d for d in DOCS if d.startswith("recipes/")])
def test_a_recipe_is_at_most_850_prose_words(doc):
    words = dp.prose_words(dp.page_text(dp.source(doc)))
    assert words <= 850, f"{doc}: {words} words; {CUT}"


@pytest.mark.parametrize("doc", [d for d in DOCS if d.startswith("tutorials/")])
def test_a_tutorial_is_at_most_2500_words(doc):
    cells = json.loads(dp.source(doc).read_text(encoding="utf-8"))["cells"]
    words = sum(
        len("".join(cell["source"]).split())
        for cell in cells
        if cell["cell_type"] == "markdown"
    )
    assert words <= 2500, f"{doc}: {words} words; {CUT}"


#: A trailing link to the page that says more, not counted in the entry.
_LINK = re.compile(r"\s*\(\{(?:doc|py:\w+)\}`[^`]+`\)\s*$")


def _release_notes():
    notes = dp._prose(
        (dp.DOCS / "project" / "release-notes.md").read_text(encoding="utf-8")
    )
    return [m[1] for m in re.finditer(r"^- (.*(?:\n[ \t]+\S.*)*)", notes, re.MULTILINE)]


@pytest.mark.parametrize("entry", _release_notes(), ids=lambda e: e[:40])
def test_a_release_note_is_one_short_entry(entry):
    assert not re.search(r"\n[ \t]+[-*] ", entry), "no sub-bullets under an entry"
    assert "$" not in entry, "no math in a release note; the linked page has it"
    words = len(_LINK.sub("", " ".join(entry.split())).split())
    assert words <= 40, f"{words} words; {CUT}"


#: The closing link of a glossary entry to the page where the term lives.
_HOME = re.compile(r"\s*(See \{doc\}`[^`]+`\.|\(\{doc\}`[^`]+`\)\.?)\s*$")


def _glossary():
    page = (dp.DOCS / "glossary.md").read_text(encoding="utf-8")
    entries = []
    for body in dp.blocks(page, "{glossary}"):
        current = None
        for line in body.splitlines():
            if line.startswith(":"):
                continue
            if line and not line.startswith(" "):
                if current and not current[1]:
                    current[0].append(line.strip())  # an alias of the same entry
                    continue
                current = ([line.strip()], [])
                entries.append(current)
            elif current:
                current[1].append(line.strip())
    return [(terms[0], " ".join(" ".join(text).split())) for terms, text in entries]


@pytest.mark.parametrize(
    ("term", "text"), _glossary(), ids=lambda x: x if len(x) < 40 else ""
)
def test_a_glossary_entry_is_short_and_links_its_home(term, text):
    assert _HOME.search(text), f"{term}: no closing {{doc}} link to its home page"
    words = len(_HOME.sub("", text).split())
    assert words <= 50, f"{term}: {words} words; {CUT}"
