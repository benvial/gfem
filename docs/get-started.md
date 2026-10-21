# Get started

gfem is under development towards v0.1. Nothing here is a stable API yet.

## Install

gfem is not on PyPI yet. Build the wheel from a checkout of the repository and
install it:

```bash
uv build --wheel             # writes dist/gfem-<version>-py3-none-any.whl
pip install dist/gfem-*.whl  # CPU
```

Optional extras go after the wheel path:

```bash
pip install "$(ls dist/gfem-*.whl)[cuda12]"  # cuDSS GPU solver (Linux x86_64, Windows)
pip install "$(ls dist/gfem-*.whl)[xarray]"  # results as xarray datasets (.to_xarray())
pip install "$(ls dist/gfem-*.whl)[viz]"     # fields as pyvista grids (.to_pyvista())
```

`cuda13` replaces `cuda12` for CUDA 13; the two exclude each other.

Supported platforms: Linux x86_64 (glibc >= 2.28), macOS arm64 and Windows
x86_64, Python >= 3.12.

Importing gfem turns on JAX double precision (`jax_enable_x64`), which every
gfem solve requires.

## Next

- The {doc}`topology tutorial <tutorials/topology>` runs a first solve and
  differentiates it.
- {doc}`capabilities` lists what gfem solves and differentiates.
