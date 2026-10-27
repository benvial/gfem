# Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run ty check src
uv run pytest -m "unit or verification"
```

Test tiers are pytest markers set by directory: `tests/unit`,
`tests/verification`, `tests/reference`, `tests/benchmark` and `tests/cuda`.
The `reference`, `benchmark` and `cuda` tiers are deselected by default; run
them with `uv run pytest -m reference` (and so on). Reference tests read their
data from `GFEM_REFERENCE_DIR` and skip when it is unset.

## Docs

The docs site needs Python 3.13 and [just](https://just.systems):

```bash
uv sync --python 3.13 --group docs --group site
just docs   # strict build and search index, into docs/_build/html
just serve  # the same, then serve it at http://localhost:8000
```

The build executes no notebook. The tutorials' outputs and the front page's
hero data (`docs/_data/front-page.json`) are committed; after changing a
tutorial, regenerate them and commit the result:

```bash
just notebooks        # re-execute docs/tutorials/*.ipynb in place, docs figure style
just front-page-data  # resample the 3D bend's run record for the front page's hero data
```

The front page's video, `docs/_static/front/bend3d-optimization-*`, is rendered
from a second, uncommitted record of the 3D bend that keeps every iteration's
design and fields (`tools/bend3d_history.py`, about 3.2 h on 16 cores /
96 GiB). Rerender it from that record on a machine with a GPU:

```bash
just bend3d-history WORK      # the record, into WORK
just bend3d-animation WORK    # the light and dark videos, posters and timing
```

The 3D bend tutorial solves nothing: it reads a committed run record,
`docs/tutorials/data/bend3d.json` and `bend3d.npz`, written by
`tools/bend3d_run.py`. The page stops when that script or `tools/bend3d_mesh.py`
no longer matches the hashes in the record. After changing either, rerun the
script and commit the new record:

```bash
just bend3d-data      # about 2.7 h on 16 cores / 96 GiB, with the mkl extra
```
