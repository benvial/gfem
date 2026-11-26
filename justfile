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

# Re-execute the tutorials in place with the docs IPython profile (gfem style, light/dark figure pairs): all of docs/tutorials/*.ipynb, or only those named (`just notebooks topology bend`).
notebooks *names:
    files=(); for name in {{ names }}; do files+=("docs/tutorials/$(basename "$name" .ipynb).ipynb"); done; \
    if [ "${#files[@]}" -eq 0 ]; then files=(docs/tutorials/*.ipynb); fi; \
    IPYTHONDIR="{{ justfile_directory() }}/docs/_ipython" JAX_PLATFORMS=cpu \
    {{ docs_env }} jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.record_timing=False \
        "${files[@]}"
    # A rerun overwrites its 3D scenes in place (gfem_scenes.py); drop those no notebook shows any more.
    {{ docs_env }} python tools/prune_scenes.py

# Save the card stills of the physics sections (docs/_static/cards): all of them, or only those named (`just stills bend3d nanosphere waveguide-nonlinearity`). Runs each tutorial in memory up to its `card` cell: no notebook, output or scene changes.
stills *names:
    JAX_PLATFORMS=cpu {{ docs_env }} python tools/stills.py {{ names }}

# Rebuild the transmon tutorial's meshes, docs/tutorials/data/transmon{,_coarse}.msh (tools/transmon_mesh.py, gmsh pinned; a new mesh means re-measuring the tutorial's values).
transmon-mesh:
    uv run --with gmsh==4.15.2 --no-project python tools/transmon_mesh.py docs/tutorials/data

# Rebuild the metagrating tutorial's meshes, docs/tutorials/data/metagrating{,_fine}.msh (tools/metagrating_mesh.py, gmsh pinned; a new mesh means re-measuring the tutorial's values).
metagrating-mesh:
    uv run --with gmsh==4.15.2 --with numpy --no-project python tools/metagrating_mesh.py docs/tutorials/data

# Rebuild the photonic-crystal tutorial's meshes, docs/tutorials/data/rods{,_fine}.msh (tools/rods_mesh.py, gmsh pinned; a new mesh means re-measuring the tutorial's values).
rods-mesh:
    uv run --with gmsh==4.15.2 --with numpy --no-project python tools/rods_mesh.py docs/tutorials/data

# Rebuild the grating-coupler tutorial's mesh, docs/tutorials/data/grating.msh (tools/grating_mesh.py, gmsh pinned; a new mesh means re-measuring the tutorial's values).
grating-mesh:
    uv run --with gmsh==4.15.2 --no-project python tools/grating_mesh.py docs/tutorials/data

# Rebuild the incident-field and cross-section recipes' sphere mesh, docs/tutorials/data/sphere.msh (tools/sphere_mesh.py, netgen; a new mesh means re-measuring the recipes' values).
sphere-mesh:
    PYTHONPATH=tools uv run python tools/sphere_mesh.py docs/tutorials/data

# Rebuild the dipole tutorial's nanosphere mesh, docs/tutorials/data/nanosphere.msh (tools/nanosphere_mesh.py, netgen; a new mesh means re-measuring the tutorial's values).
nanosphere-mesh:
    PYTHONPATH=tools uv run python tools/nanosphere_mesh.py docs/tutorials/data

# Rebuild the meta-atom tutorial's mesh, docs/tutorials/data/meta_atom.msh (tools/meta_atom_mesh.py, netgen; a new mesh means re-measuring the tutorial's values).
meta-atom-mesh:
    PYTHONPATH=tools uv run python tools/meta_atom_mesh.py docs/tutorials/data

# Rebuild the heater tutorial's cross-sections, docs/tutorials/data/heater_w{2,7p5}.msh, and the thermo-optic recipe's extruded heater_straight.msh (tools/heater_section_mesh.py, gmsh pinned; a new mesh means re-measuring the tutorial's and the recipes' values).
heater-section-mesh:
    uv run --with gmsh==4.15.2 --no-project python tools/heater_section_mesh.py docs/tutorials/data

# Regenerate the front page's hero data, docs/_data/front-page.json, from the 3D bend's run record (solves nothing).
front-page-data:
    {{ docs_env }} python tools/front_page_data.py

# Rerun the 3D bend and rewrite its run record, docs/tutorials/data/bend3d.{json,npz} (tools/bend3d_run.py): about 2.5 h on 16 cores / 96 GiB with the mkl extra.
bend3d-data work="/tmp/bend3d-run":
    JAX_PLATFORMS=cpu uv run --extra mkl --group docs python tools/bend3d_run.py all --out "{{ work }}"
    JAX_PLATFORMS=cpu uv run --extra mkl --group docs python tools/bend3d_run.py merge --out "{{ work }}" --record docs/tutorials/data

# Rerun the meta-atom design and rewrite its run record, docs/tutorials/data/meta_atom.{json,npz} (tools/meta_atom_run.py): about 10 min on 8 threads, 4 GiB.
meta-atom-data:
    JAX_PLATFORMS=cpu uv run --group docs python tools/meta_atom_run.py --out docs/tutorials/data

# Rerun the 3D spiral heater and rewrite its run record, docs/tutorials/data/heater3d.{json,npz} (tools/heater3d_run.py, gmsh pinned): about 30 min on 16 cores; the ladder's finest mesh needs about 14 GiB.
heater3d-data work="/tmp/heater3d-run":
    JAX_PLATFORMS=cpu uv run --with gmsh==4.15.2 --group docs python tools/heater3d_run.py all --out "{{ work }}"
    JAX_PLATFORMS=cpu uv run --with gmsh==4.15.2 --group docs python tools/heater3d_run.py merge --out "{{ work }}" --record docs/tutorials/data

# Rerun the arc design's spectrum beside the 3D bend's run record, docs/tutorials/data/bend3d_arc.{json,npz} (tools/bend3d_arc_spectrum.py): about 20 min on 16 cores / 96 GiB with the mkl extra.
bend3d-arc-data work="/tmp/bend3d-arc":
    JAX_PLATFORMS=cpu uv run --extra mkl --group docs python tools/bend3d_arc_spectrum.py --out docs/tutorials/data --work "{{ work }}"

# Rerun the 3D bend's optimization keeping every iteration's design, gradient and fields, for animations (tools/bend3d_history.py; not committed): about 3.2 h on 16 cores / 96 GiB with the mkl extra.
bend3d-history work="/tmp/bend3d-history":
    JAX_PLATFORMS=cpu uv run --extra mkl --group docs python tools/bend3d_history.py all --work "{{ work }}"

# Render the front page's optimization video, docs/_static/front/bend3d-optimization-*, from that record (tools/bend3d_animation.py; GPU via EGL, about 10 min).
bend3d-animation record="/tmp/bend3d-history":
    cd tools && VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow {{ docs_env }} python bend3d_animation.py "{{ record }}"

# Build the whitepaper, whitepaper/gfem-whitepaper.pdf: its data from the docs and tests (tools/whitepaper_data.py), then typst >= 0.14 (not a Python dependency: install it yourself).
whitepaper:
    {{ docs_env }} python tools/whitepaper_data.py
    typst compile --font-path whitepaper/fonts whitepaper/whitepaper.typ whitepaper/gfem-whitepaper.pdf

# Build the site as docs.yml does (strict, then the Pagefind index) into docs/_build/html.
docs:
    GFEM_DOCS_STRICT=1 {{ docs_env }} sphinx-build -n -W --keep-going docs docs/_build/html
    {{ docs_env }} python -m pagefind --site docs/_build/html

# Build the site, then serve it at http://localhost:8000.
serve: docs
    {{ docs_env }} python -m http.server --directory docs/_build/html 8000
