"""Sphinx configuration for the gfem docs site.

Build with ``just docs`` (or ``uv run --python 3.13 --group docs --group site
sphinx-build docs docs/_build/html``). The build executes no notebook: the
tutorials' outputs and the front page's hero data are committed, and
``just notebooks`` / ``just front-page-data`` regenerate them.
"""

import ast
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from sphinx.errors import ConfigError

DOCS = Path(__file__).resolve().parent

# gfem is imported by the build (the version here, Capabilities from the tests);
# JAX on CPU keeps it from probing for CUDA.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import gfem  # noqa: E402

project = "gfem"
html_title = "gfem"
version = release = gfem.__version__

# The repository and the published site. REPO_URL also switches the install
# line below to a pip-from-git install.
REPO_URL = "https://github.com/benvial/gfem"
html_baseurl = "https://benvial.github.io/gfem/"

extensions = [
    "myst_nb",  # also loads myst_parser for the .md pages
    "sphinx.ext.napoleon",
]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "jupyter_execute",
    "tutorials/data",
    # never published, though a local checkout may keep them under docs/
    "adr",
    "agents",
]

# --- MyST / MyST-NB -----------------------------------------------------------
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist"]

# The tutorials render their committed outputs; the build never executes them
# (CI's tutorials job does, on the plain path).
nb_execution_mode = "off"
nb_merge_streams = True  # one output box per cell, not one per print()
nb_output_stderr = "warn"  # stray stderr is a warning, so -W fails on it

# --- Figure pipeline -----------------------------------------------------------
# `just notebooks` executes the tutorials in place with IPYTHONDIR=docs/_ipython,
# whose startup file applies the gfem matplotlib style (docs/_mpl) and gives
# every figure a text/html light/dark pair next to its plain image/png
# (docs/_ext/gfem_figures.py). MyST-NB's HTML priority puts text/html before
# image/png, so the site shows the pair and Jupyter or GitHub the plain PNG.
# PyVista plotters take the same route (docs/_ext/gfem_scenes.py): their
# text/html is an iframe of trame-vtk's offline vtk.js viewer on a committed
# scene, _static/scenes/*.vtksz; setup() below copies the viewer to _static/.
# --- end figure pipeline -------------------------------------------------------

# --- Docstrings: gfem uses Google style ----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_use_ivar = True  # "Attributes:" as :ivar: fields

# --- Theme: the gfem "Instrument" world on Shibuya ------------------------------
# Every visual rule is carried by one of: these settings, the seven Shibuya
# template overrides in _templates/ (five partials and components here, plus
# layout/default.html and searchbox.html for search below; each Shibuya
# 2026.7.12 original sits in _templates/upstream/, so an upgrade check is a
# diff), _static/gfem.css (icons from _static/gfem-icons.css) or
# _static/gfem.js. The front page adds its own layout,
# _templates/layout/gfem-home.html, with _static/gfem-home.css and
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
# icons: Lucide masks, painted by gfem.css
html_css_files = ["gfem-icons.css", "gfem.css"]
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
ogp_site_url = html_baseurl
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
# EXPECTED and OUT_OF_SCOPE for the gfem-gradient-matrix and gfem-raises
# directives on capabilities.md. A claim that cannot be derived fails the
# build. Every page template also gets the same dict as `gfem_capabilities`
# (the front page's compact matrix); no html_context entry is needed.
# docs/_ext is on sys.path (the API reference block above).
extensions.append("gfem_capabilities")
# Base URL the test paths are appended to (cells link to their test file).
gfem_capabilities_source_url = f"{REPO_URL}/blob/main/"
# --- end Capabilities ------------------------------------------------------------

# --- Front page: layout/gfem-home.html (index.md: `layout: gfem-home`) -----------
# Static in the template: the headline, lead, scope and section prose, the
# chain of the differentiable pipeline, the button and link labels, the
# accessibility labels and the platform list ("Linux x86_64, macOS arm64 and
# Windows x86_64"). Every number and quoted line reaches it from here (the hero
# data and its model values, the code line, the version status, the Python
# floor, the install line and its note) or through gfem_capabilities (the
# compact matrix and the provenance line, above).

# The install line and its note, in one place: the front page and get-started.md
# (the {{ install }} substitution) show both. At PyPI publishing this becomes
# {"line": "pip install gfem", "note": ""}.
if REPO_URL:
    gfem_install = {
        "line": f'pip install "gfem @ git+{REPO_URL}"',
        "note": "gfem is not on PyPI yet: pip installs it from the repository.",
    }
else:
    gfem_install = {
        "line": "uv build --wheel && pip install dist/gfem-*.whl",
        "note": "gfem is not on PyPI yet: build the wheel in a checkout of the "
        "repository and install it.",
    }


def _status(version: str) -> str:
    """The version label: "under development towards v0.1" before a release."""
    v = Version(version)
    if v.is_prerelease:
        return f"under development towards v{v.major}.{v.minor}"
    return f"v{v.public}"


def _python_floor(pyproject: Path) -> str:
    """The lowest supported Python, from ``requires-python`` (">=3.12" -> "3.12")."""
    spec = tomllib.loads(pyproject.read_text())["project"]["requires-python"]
    floors = [s.version for s in SpecifierSet(spec) if s.operator == ">="]
    if len(floors) != 1:
        msg = f"front page: cannot read one >= floor from requires-python {spec!r}"
        raise ConfigError(msg)
    return floors[0]


gfem_status = _status(release)
gfem_python = _python_floor(DOCS.parent / "pyproject.toml")
myst_enable_extensions += ["substitution"]
myst_substitutions = {
    "status": gfem_status,
    "python": gfem_python,
    "install": "\n\n".join(
        part
        for part in (gfem_install["note"], f"```bash\n{gfem_install['line']}\n```")
        if part
    ),
}


def _code_line(script: Path) -> dict:
    """The run's value_and_grad assignment, split around ``jax.value_and_grad(``."""
    tree = ast.parse(script.read_text())
    lines = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and [ast.unparse(target) for target in node.targets] == ["value_and_grad"]
    ]
    call = "jax.value_and_grad("
    if len(lines) != 1 or call not in lines[0]:
        msg = (
            f"front page: expected one `value_and_grad = ...{call}...` assignment "
            f"in {script.name}, found {lines}"
        )
        raise ConfigError(msg)
    head, args = lines[0].split(call, 1)
    return {"head": head, "args": args}


def _num(value: float) -> str:
    return f"{value:g}"


def _front(path: Path, video: Path) -> dict | None:
    """The committed hero data (tools/front_page_data.py), or None when absent.

    ``video`` is the optimization video's timing (tools/bend3d_animation.py),
    inlined beside the data; the hero needs both.
    """
    if not (path.exists() and video.exists()):
        return None  # the hero frame says how to generate them; strict builds fail
    data = json.loads(path.read_text())
    data["video"] = json.loads(video.read_text())
    params, results = data["params"], data["results"]  # the run record's values
    snapshots = data["snapshots"]
    return {
        "wavelength": _num(params["wavelength_um"]),
        "thickness": _num(params["core_thickness_um"] * 1e3),  # nm
        "width": _num(params["core_width_um"]),
        "square": _num(params["square_um"]),
        "box": _num(2 * data["extent_um"][1]),
        "design": _num(data["design_um"]),
        "pixels": params["pixels"],
        "pitch": _num(params["pitch_um"]),
        "steps": params["iterations"],
        "steps_per_beta": params["steps_per_beta"],
        "last": params["iterations"] - 1,
        "start": _num(params["start"]),
        "dof": f"{params['dof'] / 1e6:.2f} M",
        "order": params["order"],
        "median_s": f"{params['median_s']:.0f}",
        "cores": params["cores"],
        "n_si": f"{params['n_si']:.2f}",
        "n_sio2": f"{params['n_sio2']:.2f}",
        "t_final": f"{results['t_final']:.4f}",
        "db_final": f"{results['db_final']:.3f}",
        "reciprocity": f"{results['reciprocity']:.1e}",
        "balance": f"{results['balance']:.4f}",
        "band_min": {
            "um": _num(results["band_min"]["um"]),
            "t": f"{results['band_min']['t']:.4f}",
        },
        "band": [_num(v) for v in results["band_um"]],
        "snapshots": [
            {"iteration": k, "t": f"{t:.3f}"}
            for k, t in zip(snapshots["iterations"], snapshots["t"], strict=True)
        ],
        # inlined into a <script> (no extra fetch); "</" cannot close the element
        "data_json": json.dumps(data, separators=(",", ":")).replace("</", "<\\/"),
    }


FRONT_DATA = DOCS / "_data" / "front-page.json"
FRONT_VIDEO = DOCS / "_static" / "front" / "bend3d-optimization.json"
html_context["gfem_front"] = _front(FRONT_DATA, FRONT_VIDEO)
html_context["gfem_code_line"] = _code_line(DOCS.parent / "tools" / "bend3d_run.py")
html_context["gfem_install"] = gfem_install
html_context["gfem_status"] = gfem_status  # the footer and the front page
html_context["gfem_python"] = gfem_python
# The front-page header: Get started, Tutorials, Capabilities (no Recipes, no API).
html_context["gfem_nav_home"] = [
    link
    for link in html_theme_options["nav_links"]
    if link["title"] in {"Get started", "Tutorials", "Capabilities"}
]
# --- end front page ---------------------------------------------------------------


def setup(app):
    """Fail a strict build (GFEM_DOCS_STRICT=1) that lacks the hero data.

    Raised from config-inited, not while conf.py runs, so Sphinx prints the
    message instead of a crash report. After an HTML build, also copy the
    offline viewer of the tutorials' 3D scenes to _static/.
    """

    def require_front(_app, config) -> None:
        if os.environ.get("GFEM_DOCS_STRICT") == "1" and not config.html_context.get(
            "gfem_front"
        ):
            msg = (
                f"front page: {FRONT_DATA.relative_to(DOCS.parent)} or "
                f"{FRONT_VIDEO.relative_to(DOCS.parent)} is missing; regenerate them "
                "with `just front-page-data` and tools/bend3d_animation.py "
                "(GFEM_DOCS_STRICT=1)"
            )
            raise ConfigError(msg)

    app.connect("config-inited", require_front)

    def copy_scene_viewer(app, exception) -> None:
        """Put trame-vtk's offline viewer next to the scenes it opens."""
        if exception is not None or app.builder.format != "html":
            return
        from trame_vtk.tools.vtksz2html import HTML_VIEWER_PATH

        shutil.copy(HTML_VIEWER_PATH, Path(app.outdir) / "_static")

    app.connect("build-finished", copy_scene_viewer)
