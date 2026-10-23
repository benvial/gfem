# gfem maintenance recipes (https://just.systems); `just` lists them.
# The docs build executes no notebook: the tutorials' outputs and the front
# page's hero data are committed, and these recipes regenerate them.

set shell := ["bash", "-eo", "pipefail", "-c"]

docs_env := "uv run --python 3.13 --group docs --group site"

# The tutorials' 3D views (PyVista) render off-screen, with no display server.
export VTK_DEFAULT_OPENGL_WINDOW := "vtkOSOpenGLRenderWindow"

# List the recipes.
default:
    @just --list

# Re-execute docs/tutorials/*.ipynb in place with the docs IPython profile (gfem style, light/dark figure pairs).
notebooks:
    rm -rf docs/_static/scenes  # the 3D scenes the run exports (gfem_scenes.py)
    IPYTHONDIR="{{ justfile_directory() }}/docs/_ipython" JAX_PLATFORMS=cpu \
    {{ docs_env }} jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.record_timing=False \
        docs/tutorials/*.ipynb

# Regenerate the front page's hero data, docs/_data/front-page.json, from the topology tutorial.
front-page-data:
    {{ docs_env }} python tools/front_page_data.py

# Rerun the 3D bend and rewrite its run record, docs/tutorials/data/bend3d.{json,npz} (tools/bend3d_run.py): about 1 h on 16 cores / 64 GiB with the mkl extra.
bend3d-data work="/tmp/bend3d-run":
    JAX_PLATFORMS=cpu uv run --extra mkl --group docs python tools/bend3d_run.py all --out "{{ work }}"
    JAX_PLATFORMS=cpu uv run --extra mkl --group docs python tools/bend3d_run.py merge --out "{{ work }}" --record docs/tutorials/data

# Build the site as docs.yml does (strict, then the Pagefind index) into docs/_build/html.
docs:
    GFEM_DOCS_STRICT=1 {{ docs_env }} sphinx-build -n -W --keep-going docs docs/_build/html
    {{ docs_env }} python -m pagefind --site docs/_build/html

# Build the site, then serve it at http://localhost:8000.
serve: docs
    {{ docs_env }} python -m http.server --directory docs/_build/html 8000
