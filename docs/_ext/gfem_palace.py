"""Sphinx extension: fail the build when the site or the README names Palace.

User-facing docs never name Palace (ADR 0080). After the HTML is written, the
``build-finished`` hook scans every text file the build publishes (HTML pages,
their ``_sources`` copies, ``searchindex.js``, JSON, downloaded scripts and
notebooks) and ``README.md`` for "palace" in any case. Each match is a warning
with its file and line, so the strict build (``-W``) fails. Binary files are
skipped. The pagefind index is built afterwards from the checked HTML.

The one allowed match is gsim's import path ``gsim.palace`` inside code: gsim
exports ``DrivenSim`` only from there, so user code must type it. Code is a
``<pre>`` block in HTML, read with its tags stripped (Pygments may split
``gsim.palace`` into spans), a fenced block in Markdown and ``_sources``, a
code cell in a notebook, and a whole downloaded script.
"""

import html
import json
import re
from pathlib import Path

_WORD = re.compile("palace", re.IGNORECASE)
_ALLOWED = "gsim.palace"
_HTML_CODE = re.compile(r"<pre\b.*?</pre>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]*>")
_MD_CODE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
_WHOLE = re.compile(r"\A.*\Z", re.DOTALL)
#: Code finders by suffix; any other text file has no code.
_CODE = {
    ".html": _HTML_CODE,
    ".txt": _MD_CODE,
    ".md": _MD_CODE,
    ".py": _WHOLE,
    ".sh": _WHOLE,
}
#: What a code block reads as: HTML loses its tags and entities.
_STRIP = {".html": lambda block: html.unescape(_TAG.sub("", block))}


def _allowed(text, match):
    return text[match.start() - len("gsim.") : match.end()] == _ALLOWED


def _hit(text, start, end, offset=0):
    line = offset + text.count("\n", 0, start) + 1
    return line, text[max(0, start - 40) : end + 40].replace("\n", " ")


def _hits(text, code_blocks, strip=lambda block: block):
    """(line, excerpt) of each match in ``text`` outside the allowed import."""
    spans = [m.span() for m in code_blocks.finditer(text)] if code_blocks else []
    found = [
        _hit(text, *match.span())
        for match in _WORD.finditer(text)
        if not any(a <= match.start() < b for a, b in spans)
    ]
    for a, b in spans:
        block = strip(text[a:b])
        offset = text.count("\n", 0, a)
        found += [
            _hit(block, *m.span(), offset)
            for m in _WORD.finditer(block)
            if not _allowed(block, m)
        ]
    return sorted(found)


def _notebook(text):
    """A notebook as Markdown: code cells fenced, so only they hold code."""
    try:
        cells = json.loads(text).get("cells", [])
    except (json.JSONDecodeError, AttributeError):
        return text
    parts = []
    for cell in cells:
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        if cell.get("cell_type") == "code":
            source = f"```\n{source}\n```"
        parts.append(source)
    return "\n".join(parts)


def _text(path):
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        return None


def palace_hits(html_dir, readme):
    """Every forbidden Palace mention, as (path, line, excerpt), in file order.

    In a notebook the line counts in its cell text, not in its JSON.
    """
    found = []
    files = [
        path
        for path in sorted(Path(html_dir).rglob("*"))
        if path.is_file() and ".doctrees" not in path.parts
    ]
    for path in [*files, Path(readme)]:
        text = _text(path)
        if text is None:
            continue
        suffix = path.suffix.lower()
        if suffix == ".ipynb" or path.name.endswith(".ipynb.txt"):
            text, suffix = _notebook(text), ".md"
        hits = _hits(text, _CODE.get(suffix), _STRIP.get(suffix, lambda b: b))
        found += [(path, line, excerpt) for line, excerpt in hits]
    return found


def _check(app, exception):
    from sphinx.util import logging

    if exception is not None or app.builder.format != "html":
        return
    readme = Path(app.srcdir).parent / "README.md"
    for path, line, excerpt in palace_hits(app.outdir, readme):
        logging.getLogger(__name__).warning(
            "Palace named in user-facing docs (ADR 0080): ...%s...",
            excerpt,
            location=f"{path}:{line}",
        )


def setup(app):
    """Register the post-build Palace check."""
    app.connect("build-finished", _check)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
