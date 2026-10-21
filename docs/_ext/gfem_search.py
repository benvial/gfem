"""Sphinx extension: keep the permalink "#" out of the Pagefind index.

Sphinx appends ``<a class="headerlink">#</a>`` to every heading and API
signature. Pagefind takes a page's title from its first ``h1`` and indexes all
text inside ``data-pagefind-body``, so without this every result title and
excerpt would read "Shape gradients#". The ``html-page-context`` hook marks
those links ``data-pagefind-ignore``; the rendered page is unchanged.
"""

_HEADERLINK = 'class="headerlink"'
_IGNORED = _HEADERLINK + " data-pagefind-ignore"


def _ignore_headerlinks(app, pagename, templatename, context, doctree):  # noqa: ARG001
    body = context.get("body")
    if body:
        context["body"] = body.replace(_HEADERLINK, _IGNORED)


def setup(app):
    """Register the Pagefind index hook."""
    app.connect("html-page-context", _ignore_headerlinks)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
