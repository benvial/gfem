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
just front-page-data  # rerun the topology tutorial for the front page's hero data
```
