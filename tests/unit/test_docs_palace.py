"""The docs build's Palace check (docs/_ext/gfem_palace.py).

User-facing docs never name Palace (ADR 0080). The only allowed string is the
gsim import path ``gsim.palace`` inside a code block, because user code must
type it.
"""

import importlib.util
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"

GSIM_IMPORT = (
    '<div class="highlight"><pre><span></span><span class="kn">from</span>'
    '<span class="w"> </span><span class="nn">gsim.palace</span>'
    '<span class="w"> </span><span class="kn">import</span> '
    '<span class="n">DrivenSim</span>\n</pre></div>\n'
)


@pytest.fixture(scope="module")
def gfem_palace():
    spec = importlib.util.spec_from_file_location(
        "gfem_palace", DOCS / "_ext" / "gfem_palace.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def site(tmp_path, pages, readme=""):
    html = tmp_path / "html"
    for name, text in pages.items():
        page = html / name
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(text, encoding="utf-8")
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return html, tmp_path / "README.md"


def test_clean_site_has_no_hits(gfem_palace, tmp_path):
    html, readme = site(tmp_path, {"index.html": "<p>gfem</p>\n"}, "# gfem\n")
    assert gfem_palace.palace_hits(html, readme) == []


def test_gsim_import_in_a_code_block_is_allowed(gfem_palace, tmp_path):
    readme = "```python\nfrom gsim.palace import DrivenSim\n```\n"
    html, readme = site(tmp_path, {"recipes/gsim.html": GSIM_IMPORT}, readme)
    assert gfem_palace.palace_hits(html, readme) == []


def test_prose_mention_is_reported_with_file_and_line(gfem_palace, tmp_path):
    page = "<p>gfem</p>\n<p>As in Palace, the sign is +.</p>\n"
    html, readme = site(tmp_path, {"recipes/a.html": page})
    [(path, line, excerpt)] = gfem_palace.palace_hits(html, readme)
    assert (path, line) == (html / "recipes" / "a.html", 2)
    assert "Palace" in excerpt


@pytest.mark.parametrize(
    "page",
    [
        "<p>gfem reads <code>gsim.palace</code> output</p>\n",  # inline, not a block
        "<pre>mesh = 'sim/palace.msh'</pre>\n",  # code block, not the import
        '<a href="https://github.com/awslabs/PALACE">x</a>\n',  # attribute
        "<title>Coming from palace</title>\n",
    ],
)
def test_every_other_mention_is_a_hit(gfem_palace, tmp_path, page):
    html, readme = site(tmp_path, {"p.html": page})
    assert len(gfem_palace.palace_hits(html, readme)) == 1


def test_readme_prose_is_a_hit(gfem_palace, tmp_path):
    readme = "# gfem\n\nSee `gsim.palace` and Palace.\n"
    html, readme = site(tmp_path, {}, readme)
    assert [line for _, line, _ in gfem_palace.palace_hits(html, readme)] == [3, 3]


SPLIT_IMPORT = (
    '<div class="highlight"><pre><span></span><span class="kn">import</span>'
    '<span class="w"> </span><span class="nn">gsim</span><span class="o">.</span>'
    '<span class="nn">palace</span>\n</pre></div>\n'
)
SPLIT_ATTRIBUTE = (
    '<div class="highlight"><pre><span></span><span class="n">sim</span> '
    '<span class="o">=</span> <span class="n">gsim</span><span class="o">.</span>'
    '<span class="n">palace</span><span class="o">.</span>'
    '<span class="n">DrivenSim</span><span class="p">()</span>\n</pre></div>\n'
)


@pytest.mark.parametrize("page", [SPLIT_IMPORT, SPLIT_ATTRIBUTE])
def test_gsim_import_split_into_spans_is_allowed(gfem_palace, tmp_path, page):
    html, readme = site(tmp_path, {"recipes/gsim.html": page})
    assert gfem_palace.palace_hits(html, readme) == []


def test_a_split_code_block_still_reports_another_mention(gfem_palace, tmp_path):
    page = SPLIT_ATTRIBUTE.replace("DrivenSim", "Palace")
    html, readme = site(tmp_path, {"p.html": "<p>x</p>\n" + page})
    [(_, line, excerpt)] = gfem_palace.palace_hits(html, readme)
    assert line == 2
    assert "Palace" in excerpt


@pytest.mark.parametrize(
    "name",
    [
        "_sources/recipes/a.md.txt",
        "searchindex.js",
        "_static/data.json",
        "_downloads/abc/run.py",
        "tutorials/a.ipynb",
    ],
)
def test_every_published_text_file_is_scanned(gfem_palace, tmp_path, name):
    text = "Palace\n"
    if name.endswith(".ipynb"):
        text = '{"cells": [{"cell_type": "markdown", "source": ["As in Palace"]}]}'
    html, readme = site(tmp_path, {name: text})
    assert len(gfem_palace.palace_hits(html, readme)) == 1


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("_sources/recipes/gsim.md.txt", "```python\nfrom gsim.palace import X\n```\n"),
        ("_downloads/abc/run.py", "from gsim.palace import DrivenSim\n"),
        (
            "tutorials/a.ipynb",
            '{"cells": [{"cell_type": "code", "source": ["import gsim.palace"]}]}',
        ),
        (
            "_sources/tutorials/a.ipynb.txt",
            '{"cells": [{"cell_type": "code", "source": ["import gsim.palace"]}]}',
        ),
    ],
)
def test_gsim_import_is_allowed_in_the_code_of_every_file_type(
    gfem_palace, tmp_path, name, text
):
    html, readme = site(tmp_path, {name: text})
    assert gfem_palace.palace_hits(html, readme) == []


def test_binary_files_are_not_scanned(gfem_palace, tmp_path):
    html, readme = site(tmp_path, {"index.html": "<p>gfem</p>\n"})
    (html / "font.woff2").write_bytes(b"\xff\xfePalace")
    assert gfem_palace.palace_hits(html, readme) == []
