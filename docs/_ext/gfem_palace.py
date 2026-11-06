"""Sphinx extension: fail the build when the site or the README names Palace.

User-facing docs never name Palace (ADR 0080). After the HTML is written, the
``build-finished`` hook scans every built ``.html`` file (prose, generated
snippets, notebook outputs, API pages, attributes) and ``README.md`` for
"palace" in any case. Each match is a warning with its file and line, so the
strict build (``-W``) fails. The one allowed match is gsim's import path
``gsim.palace`` inside a code block (``<pre>`` in HTML, a fenced block in the
README): gsim exports ``DrivenSim`` only from there, so user code must type it.
"""

import re
from pathlib import Path

_WORD = re.compile("palace", re.IGNORECASE)
_ALLOWED = "gsim.palace"
_HTML_CODE = re.compile(r"<pre\b.*?</pre>", re.DOTALL | re.IGNORECASE)
_MD_CODE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)


def _hits(text, code_blocks):
    """(line, excerpt) of each match in ``text`` outside the allowed import."""
    spans = [m.span() for m in code_blocks.finditer(text)]
    found = []
    for match in _WORD.finditer(text):
        start, end = match.span()
        in_code = any(a <= start and end <= b for a, b in spans)
        if in_code and text[start - len("gsim.") : end] == _ALLOWED:
            continue
        line = text.count("\n", 0, start) + 1
        excerpt = text[max(0, start - 40) : end + 40].replace("\n", " ")
        found.append((line, excerpt))
    return found


def palace_hits(html_dir, readme):
    """Every forbidden Palace mention, as (path, line, excerpt), in file order."""
    found = []
    pages = [(path, _HTML_CODE) for path in sorted(Path(html_dir).rglob("*.html"))]
    for path, code_blocks in [*pages, (Path(readme), _MD_CODE)]:
        text = path.read_text(encoding="utf-8")
        found += [(path, line, excerpt) for line, excerpt in _hits(text, code_blocks)]
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
