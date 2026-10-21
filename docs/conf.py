"""Sphinx configuration for the gfem docs site.

Build with ``uv run --python 3.13 --group docs --group site sphinx-build docs
docs/_build/html``.
"""

import os
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent

project = "gfem"
html_title = "gfem"

extensions = [
    "myst_nb",  # also loads myst_parser for the .md pages
    "sphinx.ext.napoleon",
]

exclude_patterns = [
    "_build",
    "_generated",
    "**.ipynb_checkpoints",
    "jupyter_execute",
    "tutorials/data",
]

# --- MyST / MyST-NB -----------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]

# The notebooks keep their committed outputs. "off" (the local default) renders
# them; docs.yml sets GFEM_NB_EXECUTION=cache to execute them through
# jupyter-cache.
nb_execution_mode = os.environ.get("GFEM_NB_EXECUTION", "off")
nb_execution_cache_path = str(DOCS / "_build" / ".jupyter_cache")
nb_execution_timeout = 1800  # as CI's tutorials job
nb_execution_raise_on_error = True
nb_merge_streams = True  # one output box per cell, not one per print()
nb_output_stderr = "warn"  # stray stderr is a warning, so -W fails on it
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # no "CUDA jaxlib" stderr line

# --- Docstrings: gfem uses Google style ----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_use_ivar = True  # "Attributes:" as :ivar: fields

# --- Theme: the gfem "Mesh" world on Shibuya ------------------------------------
# Every visual rule is carried by one of: these settings, the five partial
# overrides in _templates/ (Shibuya 2026.7.12 originals in _templates/upstream/,
# so an upgrade check is a diff), _static/gfem.css or _static/gfem.js.
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
