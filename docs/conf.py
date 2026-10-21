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

# --- Figure pipeline -----------------------------------------------------------
# Executing kernels read docs/_ipython, whose startup file applies the gfem
# matplotlib style (docs/_mpl) and displays every figure as a light/dark pair
# (docs/_ext/gfem_figures.py). With "off" the committed outputs are shown as is.
os.environ.setdefault("IPYTHONDIR", str(DOCS / "_ipython"))
# --- end figure pipeline -------------------------------------------------------

# --- Docstrings: gfem uses Google style ----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
napoleon_use_ivar = True  # "Attributes:" as :ivar: fields
