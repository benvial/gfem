"""Sphinx configuration for the gfem docs site.

Build with ``uv run --python 3.13 --group docs --group site sphinx-build docs
docs/_build/html``.
"""

import os
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

# --- Capabilities: claims derived from the tests (ADR 0017) ---------------------
# docs/_ext/gfem_capabilities.py imports the verification tests' CELLS,
# EXPECTED and OUT_OF_SCOPE and AST-reads the reference tolerances, for the
# gfem-gradient-matrix, gfem-raises and gfem-palace-agreement directives on
# capabilities.md. A claim that cannot be derived fails the
# build. Every page template also gets the same dict as `gfem_capabilities`
# (the front page's compact matrix and Palace rows); no html_context entry is
# needed.
import sys  # noqa: E402 - kept in this block so parallel conf.py edits merge

sys.path.insert(0, str(DOCS / "_ext"))
extensions.append("gfem_capabilities")
# Base URL the test paths are appended to (cells link to their test file);
# empty until a remote exists, e.g. f"{REPO_URL}/blob/main/".
gfem_capabilities_source_url = ""
# --- end Capabilities ------------------------------------------------------------
