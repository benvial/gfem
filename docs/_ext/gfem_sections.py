"""The nine physics sections: their pages, cards and gradient tags.

A section is a root toctree of ``docs/index.md`` whose first entry is a page
``physics/<slug>``. The page carries what belongs to the section alone: its
name (the H1, equal to the toctree caption), one physics line (its first
paragraph) and two front-matter fields, ``still`` and ``gradient_rows``.
Everything else is read from the site tree, so a recipe added to a toctree
needs no other edit: the large cards are the section's tutorials (name = the
notebook's H1, line = its first paragraph), or the first recipe when the
section has none; the other recipes are listed by title.

The gradient tag is never typed. It lists the input classes with an
FD-verified result on the section's gradient-matrix rows, from the data
`gfem_capabilities` derives from the verification tests.

`sections` and `tag` are pure: no Sphinx import, no build. `setup` registers
the build hooks, which need ``gfem_capabilities`` registered first.
"""

import json
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
CONTEXT_KEY = "gfem_sections"
PHYSICS_DIR = "physics"
CARDS_DIR = Path("_static") / "cards"
#: Pages that render other files' data and so are read again on every build.
INDEX_DOC = "index"

#: The input classes of the tag, in tag order, and the inputs of each (keys
#: of ``gfem_capabilities.COLUMNS`` and ``OFF_GRID``). "Sheet fill" is the
#: matrix column's own label. A class of None hides its inputs from the tag.
CLASSES = {
    "shape": ("nodes",),
    "materials": (
        "eps_r",
        "conductivity",
        "mu_r",
        "tan_delta",
        "law",
        "blend",
        "thermal_conductivity",
        "mass_density",
        "specific_heat",
        "thermo_optic_coefficient",
    ),
    "sheet fill": ("fill",),
    "sources": ("heat", "drive"),
    None: ("lumped", "boundary_value"),
}


class SectionsError(Exception):
    """A section page, card or tag cannot be built from the site tree."""


def _blocks(text: str) -> list[tuple[dict, list[str]]]:
    """The ``{toctree}`` blocks of a page: (options, entries)."""
    found = []
    for body in re.findall(
        r"```\{toctree\}\n(.*?)^```", text, re.MULTILINE | re.DOTALL
    ):
        options = dict(re.findall(r"^:(\w+):[ \t]*(.*)$", body, re.MULTILINE))
        entries = [
            line.strip()
            for line in body.splitlines()
            if line.strip() and not line.startswith(":")
        ]
        found.append((options, entries))
    return found


def _front_matter(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {}
    return {
        key: value.split("#")[0].strip()
        for key, value in re.findall(r"^(\w+):[ \t]*(.*)$", match[1], re.MULTILINE)
    }


def _body(text: str) -> str:
    """A Markdown page without its front matter and fenced blocks."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    return re.sub(r"^```.*?^```[ \t]*$", "", text, flags=re.MULTILINE | re.DOTALL)


def _heading_and_line(markdown: str) -> tuple[str, str]:
    """The first-level heading and the first paragraph after it."""
    text = re.sub(r"<!--.*?-->", "", _body(markdown), flags=re.DOTALL)
    match = re.search(r"^# (.+)$", text, re.MULTILINE)
    if not match:
        return "", ""
    paragraph = re.split(r"\n\s*\n", text[match.end() :].strip(), maxsplit=1)[0]
    if paragraph.startswith(("#", "```", ":::")):
        return match[1].strip(), ""
    return match[1].strip(), " ".join(paragraph.split())


def _read(docs: Path, doc: str) -> tuple[str, str]:
    """(H1, first paragraph) of a document: Markdown or a notebook."""
    path = docs / f"{doc}.md"
    if path.is_file():
        found = _heading_and_line(path.read_text(encoding="utf-8"))
    else:
        path = docs / f"{doc}.ipynb"
        if not path.is_file():
            raise SectionsError(f"{doc}: no source file")
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"] == "markdown":
                found = _heading_and_line("".join(cell["source"]))
                break
        else:
            raise SectionsError(f"{doc}: no markdown cell")
    if not found[0]:
        raise SectionsError(f"{doc}: no first-level heading for the card name")
    return found


def sections(docs: Path = DOCS, derived: dict | None = None) -> list[dict]:
    """The sections in site-tree order, with their cards and recipes.

    Args:
        docs: the docs directory.
        derived: ``gfem_capabilities.derive()``'s data; with it, each
            section's ``tag`` is filled (else it is empty).

    Returns:
        Per section: ``name``, ``doc``, ``line``, ``still``, ``rows``,
        ``tag``, ``cards`` (``doc``, ``name``, ``line``, ``still``)
        and ``recipes`` (``doc``, ``name``).

    Raises:
        SectionsError: unknown row, H1 other than the caption, a physics
            page that is not the first entry of its toctree, or a class-table
            gap.
    """
    from gfem_capabilities import ROWS

    index = (docs / f"{INDEX_DOC}.md").read_text(encoding="utf-8")
    out = []
    placed = set()
    for options, entries in _blocks(index):
        caption = options.get("caption", "")
        physics = [d for d in entries if d.startswith(f"{PHYSICS_DIR}/")]
        placed.update(physics)
        if not physics:
            continue
        if entries[0] != physics[0] or len(physics) > 1:
            raise SectionsError(
                f"{caption}: the physics page must be the only physics entry and "
                f"the first of its toctree, not {entries}"
            )
        doc = physics[0]
        path = docs / f"{doc}.md"
        if not path.is_file():
            raise SectionsError(f"{doc}: no source file")
        text = path.read_text(encoding="utf-8")
        meta = _front_matter(text)
        name, line = _heading_and_line(text)
        if name != caption:
            raise SectionsError(f"{doc}: H1 {name!r} is not the caption {caption!r}")
        if meta.get("layout") != "gfem-section" or not meta.get("still"):
            raise SectionsError(f"{doc}: front matter needs layout and still")
        rows = [
            r.strip() for r in meta.get("gradient_rows", "").split(",") if r.strip()
        ]
        unknown = [r for r in rows if r not in ROWS]
        if unknown:
            raise SectionsError(f"{doc}: unknown gradient rows {unknown}")
        tutorials = [d for d in entries[1:] if d.startswith("tutorials/")]
        recipes = [d for d in entries[1:] if d.startswith("recipes/")]
        other = [d for d in entries[1:] if d not in tutorials and d not in recipes]
        if other:
            raise SectionsError(
                f"{caption}: entries that are neither tutorials/ nor recipes/ "
                f"would vanish from the cards: {other}"
            )
        cards = []
        for tutorial in tutorials:
            title, first = _read(docs, tutorial)
            if not first:
                raise SectionsError(
                    f"{tutorial}: no first paragraph after the H1 for the card line"
                )
            cards.append(
                {
                    "doc": tutorial,
                    "name": title,
                    "line": first,
                    "still": tutorial.split("/")[-1],
                }
            )
        if not cards and recipes:
            promoted = recipes.pop(0)
            cards.append(
                {
                    "doc": promoted,
                    "name": _read(docs, promoted)[0],
                    "line": "",
                    "still": meta["still"],
                }
            )
        out.append(
            {
                "name": name,
                "doc": doc,
                "line": line,
                "still": meta["still"],
                "rows": rows,
                "tag": tag(rows, derived) if derived is not None else "",
                "cards": cards,
                "recipes": [{"doc": r, "name": _read(docs, r)[0]} for r in recipes],
            }
        )
    stray = sorted(
        f"{PHYSICS_DIR}/{p.stem}"
        for p in (docs / PHYSICS_DIR).glob("*.md")
        if f"{PHYSICS_DIR}/{p.stem}" not in placed
    )
    if stray:
        raise SectionsError(f"physics pages outside a root toctree: {stray}")
    return out


def tag(rows: list[str], derived: dict) -> str:
    """``Gradients: ...`` for the classes FD-verified on ``rows``; "" for none.

    Raises:
        SectionsError: a matrix column or off-grid input missing from `CLASSES`.
    """
    from gfem_capabilities import COLUMNS, OFF_GRID

    class_of = {name: cls for cls, names in CLASSES.items() for name in names}
    missing = sorted((set(COLUMNS) | set(OFF_GRID)) - set(class_of))
    if missing:
        raise SectionsError(
            f"inputs without a gradient class in docs/_ext/gfem_sections.py: {missing}"
        )
    by_words = {words: key for key, words in OFF_GRID.items()}
    found = set()
    for row in derived["rows"]:
        if row["key"] not in rows:
            continue
        found |= {
            class_of[cell["column"]]
            for cell in row["cells"]
            if cell["status"] == "verified"
        }
    for entry in derived["also"]:
        if entry["key"] in rows:
            found |= {class_of[by_words[words]] for words in entry["inputs"]}
    found.discard(None)
    ordered = [cls for cls in CLASSES if cls in found]
    return f"Gradients: {', '.join(ordered)}" if ordered else ""


def still_names(secs: list[dict]) -> list[str]:
    """Every still the cards use, once, in page order."""
    return list(dict.fromkeys(c["still"] for s in secs for c in [s, *s["cards"]]))


def missing_stills(docs: Path, secs: list[dict]) -> list[str]:
    """The stills with a missing light or dark PNG, as file names."""
    return [
        f"{name}-{mode}.png"
        for name in still_names(secs)
        for mode in ("light", "dark")
        if not (docs / CARDS_DIR / f"{name}-{mode}.png").is_file()
    ]


def checked(docs: Path, derived: dict) -> list[dict]:
    """The tagged sections, once every still they use is committed."""
    secs = sections(docs, derived)
    missing = missing_stills(docs, secs)
    if missing:
        raise SectionsError(f"missing card stills in _static/cards: {missing}")
    return secs


# --- Sphinx --------------------------------------------------------------------


def setup(app):
    """Compute the sections once per build and give every page them."""
    from sphinx.errors import ExtensionError

    def build(app) -> None:
        try:
            secs = checked(Path(app.srcdir), app.env.gfem_capabilities)
        except SectionsError as exc:
            raise ExtensionError(f"Sections: {exc}", modname="gfem_sections") from exc
        app.env.gfem_sections = secs

    def add_context(app, _pagename, _templatename, context, _doctree) -> None:
        context[CONTEXT_KEY] = app.env.gfem_sections

    def reread(_app, env, *_changes) -> list[str]:
        # Cards depend on other files (tutorial text, recipe titles, the
        # tree): read the pages that show them again on every build.
        return [d for d in env.found_docs if d == INDEX_DOC or d.startswith("physics/")]

    app.connect("builder-inited", build)
    app.connect("env-get-outdated", reread)
    app.connect("html-page-context", add_context)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
