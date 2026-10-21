<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/gfem-lockup-dark.svg">
    <img alt="gfem" src="docs/_static/logo/gfem-lockup-light.svg" height="64">
  </picture>
</h1>

Differentiable finite-element electromagnetics for RF, microwave, mm-wave and superconducting quantum devices.
gfem runs on [NGSolve](https://ngsolve.org) for discretization and [JAX](https://jax.readthedocs.io) for
everything differentiable.

gfem is under development towards v0.1. Nothing here is a stable API yet.

## Install

```bash
pip install gfem            # CPU
pip install "gfem[cuda12]"  # optional cuDSS GPU solver (Linux x86_64, Windows)
pip install "gfem[mkl]"     # optional MKL PARDISO CPU direct solver (Linux x86_64, Windows)
pip install "gfem[xarray]"  # optional: results as xarray datasets (.to_xarray())
pip install "gfem[viz]"     # optional: fields as pyvista grids (.to_pyvista())
```

Supported platforms: Linux x86_64 (glibc >= 2.28), macOS arm64 and Windows x86_64, Python >= 3.12.

Importing gfem turns on JAX double precision (`jax_enable_x64`), which every gfem solve requires.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run ty check src
uv run pytest -m "unit or verification"
```

Test tiers are pytest markers set by directory: `tests/unit`, `tests/verification`, `tests/reference`,
`tests/benchmark` and `tests/cuda`. The `reference`, `benchmark` and `cuda` tiers are deselected by default; run
them with `uv run pytest -m reference` (and so on). Reference tests read their data from `GFEM_REFERENCE_DIR` and
skip when it is unset.

## Licence

MIT. gfem depends on NGSolve (LGPL-2.1), whose PyPI wheels also contain GPL-2.0+ SuiteSparse code (UMFPACK, CHOLMOD
Supernodal); keep that in mind when redistributing a whole gfem environment. The CI licence gate
(`tools/licence_gate.py`) fails on non-free or non-commercial licences in the installed dependency tree.
