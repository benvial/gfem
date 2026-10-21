"""Sphinx configuration for the gfem docs site.

Build with ``just docs`` (or ``uv run --python 3.13 --group docs --group site
sphinx-build docs docs/_build/html``). The build executes no notebook: the
tutorials' outputs and the front page's hero data are committed, and
``just notebooks`` / ``just front-page-data`` regenerate them.
"""

import json
import os
import sys
from pathlib import Path

from sphinx.errors import ConfigError

DOCS = Path(__file__).resolve().parent

project = "gfem"
html_title = "gfem"

extensions = [
    "myst_nb",  # also loads myst_parser for the .md pages
    "sphinx.ext.napoleon",
]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "jupyter_execute",
    "tutorials/data",
]

# --- MyST / MyST-NB -----------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]

# The tutorials render their committed outputs; the build never executes them
# (CI's tutorials job does, on the plain path).
nb_execution_mode = "off"
nb_merge_streams = True  # one output box per cell, not one per print()
nb_output_stderr = "warn"  # stray stderr is a warning, so -W fails on it
# gfem is imported by the build (Capabilities from the tests); JAX on CPU keeps
# it from probing for CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# --- Figure pipeline -----------------------------------------------------------
# `just notebooks` executes the tutorials in place with IPYTHONDIR=docs/_ipython,
# whose startup file applies the gfem matplotlib style (docs/_mpl) and gives
# every figure a text/html light/dark pair next to its plain image/png
# (docs/_ext/gfem_figures.py). MyST-NB's HTML priority puts text/html before
# image/png, so the site shows the pair and Jupyter or GitHub the plain PNG.
# --- end figure pipeline -------------------------------------------------------

# --- Docstrings: gfem uses Google style ----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_use_ivar = True  # "Attributes:" as :ivar: fields

# --- Theme: the gfem "Mesh" world on Shibuya ------------------------------------
# Every visual rule is carried by one of: these settings, the seven Shibuya
# template overrides in _templates/ (five partials and components here, plus
# layout/default.html and searchbox.html for search below; each Shibuya
# 2026.7.12 original sits in _templates/upstream/, so an upgrade check is a
# diff), _static/gfem.css or _static/gfem.js. The front page adds its own
# layout, _templates/layout/gfem-home.html, with _static/gfem-home.css and
# _static/front/front.js.
extensions += [
    "sphinx_copybutton",  # the worded "copy" button on code blocks
    "sphinxext.opengraph",
]
# Status lists and tags are plain Markdown plus classes ({.gfem-status}, [..]{.tag}).
myst_enable_extensions += ["attrs_block", "attrs_inline"]
exclude_patterns += ["_templates"]  # upstream/ copies are for diffing, not pages

html_theme = "shibuya"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["gfem.css"]
html_js_files = [("gfem.js", {"defer": "defer"})]
html_favicon = "_static/logo/favicon.svg"  # ICO, apple-touch, manifest: extra-head.html
html_permalinks_icon = "#"
html_theme_options = {
    "accent_color": "teal",  # nearest Radix scale; gfem.css pins --accent-* to gfem
    "color_mode": "auto",  # light is home; dark follows the OS until the switch is used
    "light_logo": "_static/logo/gfem-lockup-light.svg",
    "dark_logo": "_static/logo/gfem-lockup-dark.svg",
    "nav_links": [
        {"title": "Get started", "url": "get-started"},
        {"title": "Tutorials", "url": "tutorials/topology"},
        {"title": "Recipes", "url": "recipes/index"},
        {"title": "Capabilities", "url": "capabilities"},
        {"title": "API", "url": "api/index"},
    ],
    "github_url": "",  # an empty slot until a remote exists
    "show_ai_links": False,  # no "Open in ChatGPT / Claude" menu
    "ethical_ads_publisher": "",  # no EthicalAds div (the default is "readthedocs")
    "globaltoc_expand_depth": 1,
    "toctree_maxdepth": 2,
}
# Only the on-page TOC: drops repo-stats, edit-this-page and the ad slots.
html_sidebars = {"**": ["sidebars/localtoc.html"]}
html_context = {
    # The footer's Project group (partials/site-foot.html): (title, docname).
    "gfem_project_links": [
        ("Licence", "project/licence"),
        ("Development", "project/development"),
        ("Release notes", "project/release-notes"),
    ],
}
html_show_sphinx = False
html_show_copyright = False
html_show_sourcelink = False

copybutton_exclude = ".linenos, .gp, .go"  # never copy prompts or output
copybutton_selector = "div.highlight pre"

# Open Graph: one static social card. Per-page generated cards are off: they put
# the Sphinx logo on the card and warn on an SVG logo, which fails -W.
ogp_image = "_static/logo/gfem-og.png"
ogp_image_alt = (
    "gfem: gradients of full-wave FEM results with respect to your geometry "
    "and materials, in Python"
)
ogp_social_cards = {"enable": False}
ogp_custom_meta_tags = ['<meta name="twitter:card" content="summary_large_image">']
# ogp_site_url and html_baseurl: set once the public remote exists.
# --- end theme ---------------------------------------------------------------------

# --- Search: Pagefind, with Sphinx's search page as the fallback ------------------
# Pagefind runs on the built HTML after sphinx-build,
#   uv run --python 3.13 --group site python -m pagefind --site docs/_build/html
# and two more template overrides carry it (Shibuya originals in _templates/upstream/):
# layout/default.html marks the article data-pagefind-body, so only page content is
# indexed; searchbox.html opens Pagefind's Component UI modal and keeps the form to
# Sphinx's search.html, which stays built (the default) for exact API names and for
# builds without the Pagefind step. Styles: gfem.css "3m. search".
# _ext/gfem_search.py keeps the permalink "#" out of result titles (_ext is put on
# sys.path in the API block below).
extensions += ["gfem_search"]
# --- end search -------------------------------------------------------------------

# --- API reference: sphinx-autoapi and docs/_ext/gfem_api.py --------------------
# One page per public module under api/, each public name documented once on
# its home module; gfem_api.py resolves the gfem.X and module spellings.
sys.path.insert(0, str(DOCS / "_ext"))
extensions += ["autoapi.extension", "sphinx.ext.intersphinx", "gfem_api"]
exclude_patterns += ["_autoapi_templates"]

autoapi_type = "python"
autoapi_dirs = ["../src/gfem"]
autoapi_root = "api"
autoapi_add_toctree_entry = False  # own toctree in api/index.md (workflow order)
autoapi_keep_files = True  # docs/api/gfem/ is generated and git-ignored
autoapi_template_dir = "_autoapi_templates"  # package.rst: no Submodules toctree
autoapi_own_page_level = "module"  # one page per public module
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",  # names defined in private modules (_boundaries, _field)
]  # no "private-members", no "special-members": every _* module stays hidden
autoapi_member_order = "groupwise"
autoapi_python_class_content = "class"
autodoc_typehints = "description"  # signatures show names; types go to fields
python_use_unqualified_type_names = True
add_module_names = False  # "solve(...)" on the gfem.solve page
python_display_short_literal_types = True

default_role = "py:obj"  # single-backtick names in docstrings become links
# Every unresolved cross-reference is a warning; the CI build runs -n -W.
nitpicky = True
nitpick_ignore = [
    # Literal / union aliases naming a parameter's choices: not exported, and
    # the Args text lists the choices.
    *[
        ("py:class", name)
        for name in (
            "Sweep",
            "Excluded",
            "Direction",
            "Extension",
            "InterfaceKind",
            "LinearSolver",
            "Device",
        )
    ],
    # meshio publishes no objects.inv.
    ("py:class", "meshio.Mesh"),
    ("py:obj", "meshio.Mesh"),
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}
# --- end API reference ----------------------------------------------------------

# --- Capabilities: claims derived from the tests at build time -----------------
# docs/_ext/gfem_capabilities.py imports the verification tests' CELLS,
# EXPECTED and OUT_OF_SCOPE and AST-reads the reference tolerances, for the
# gfem-gradient-matrix, gfem-raises and gfem-palace-agreement directives on
# capabilities.md. A claim that cannot be derived fails the build. Every page
# template also gets the same dict as `gfem_capabilities` (the front page's
# compact matrix and Palace rows); no html_context entry is needed.
# docs/_ext is on sys.path (the API reference block above).
extensions.append("gfem_capabilities")
# Base URL the test paths are appended to (cells link to their test file);
# empty until a remote exists, e.g. f"{REPO_URL}/blob/main/".
gfem_capabilities_source_url = ""
# --- end Capabilities ------------------------------------------------------------

# --- Front page: layout/gfem-home.html (index.md: `layout: gfem-home`) -----------
# The copy is static in the template; every number or quoted line reaches it here,
# or through gfem_capabilities (the compact matrix and the Palace rows, above).

# The repository URL; empty until a public remote exists. Setting it also needs
# html_baseurl, ogp_site_url and gfem_capabilities_source_url.
REPO_URL = ""
# The install line, in one place: the front page shows it and get-started.md uses
# it through the {{ install }} substitution. At PyPI publishing it becomes
# {"line": "pip install gfem", "note": ""} and the front page's "not on PyPI yet"
# sentence goes.
if REPO_URL:
    gfem_install = {"line": f'pip install "gfem @ git+{REPO_URL}"', "note": ""}
else:
    gfem_install = {
        "line": "uv build --wheel && pip install dist/gfem-*.whl",
        "note": "Build the wheel in a checkout of the repository.",
    }
myst_enable_extensions += ["substitution"]
myst_substitutions = {"install": f"```bash\n{gfem_install['line']}\n```"}


def _code_line(notebook: Path) -> dict:
    """The tutorial's value_and_grad line, split around ``jax.value_and_grad(``."""
    cells = json.loads(notebook.read_text())["cells"]
    lines = [
        line.rstrip("\n")
        for cell in cells
        if "decisive" in cell.get("metadata", {}).get("tags", [])
        for line in cell["source"]
        if line.startswith("value_and_grad = ")
    ]
    call = "jax.value_and_grad("
    if len(lines) != 1 or call not in lines[0]:
        msg = (
            f"front page: expected one `value_and_grad = ...{call}...` line in a "
            f"`decisive` cell of {notebook.name}, found {lines}"
        )
        raise ConfigError(msg)
    head, args = lines[0].split(call, 1)
    return {"head": head, "args": args}


def _front(path: Path) -> dict | None:
    """The committed hero data (tools/front_page_data.py), or None when absent."""
    if not path.exists():
        if os.environ.get("GFEM_DOCS_STRICT") == "1":
            msg = (
                f"front page: {path.relative_to(DOCS.parent)} is missing; "
                "regenerate it with `just front-page-data` (GFEM_DOCS_STRICT=1)"
            )
            raise ConfigError(msg)
        return None  # a local build: the hero frame says how to generate it
    data = json.loads(path.read_text())
    final = data["final"]  # J of each etch variant after the optimisation
    return {
        "steps": len(data["history"]),
        "final": {name: f"{value:.3f}" for name, value in final.items()},
        "worst": f"{max(final.values()):.3f}",
        "balance": f"{data['balance']:.1e}",
        # inlined into a <script> (no extra fetch); "</" cannot close the element
        "topo_json": json.dumps(data, separators=(",", ":")).replace("</", "<\\/"),
    }


html_context["gfem_front"] = _front(DOCS / "_data" / "front-page.json")
html_context["gfem_code_line"] = _code_line(DOCS / "tutorials" / "topology.ipynb")
html_context["gfem_install"] = gfem_install
# The front-page header: Get started, Tutorials, Capabilities (no Recipes, no API).
html_context["gfem_nav_home"] = [
    link
    for link in html_theme_options["nav_links"]
    if link["title"] in {"Get started", "Tutorials", "Capabilities"}
]
# --- end front page ---------------------------------------------------------------
