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

The docs site needs Python 3.13:

```bash
uv sync --python 3.13 --group docs --group site
uv run --python 3.13 --group docs --group site sphinx-build docs docs/_build/html
```

The tutorials render their committed outputs. Set `GFEM_NB_EXECUTION=cache` to
execute them instead.
