# Get started

gfem is {{ status }}. Nothing here is a stable API yet.

## Install

{{ install }}

`uv build` writes `dist/gfem-<version>-py3-none-any.whl`; the plain install
is the CPU build.

Optional extras go after the wheel path:

```bash
pip install "$(ls dist/gfem-*.whl)[cuda12]"  # cuDSS GPU solver (Linux x86_64, Windows)
pip install "$(ls dist/gfem-*.whl)[mkl]"     # MKL PARDISO CPU direct solver (Linux x86_64, Windows)
pip install "$(ls dist/gfem-*.whl)[xarray]"  # results as xarray datasets (.to_xarray())
pip install "$(ls dist/gfem-*.whl)[viz]"     # fields as pyvista grids (.to_pyvista())
```

`cuda13` replaces `cuda12` for CUDA 13; the two exclude each other.

`mkl` makes `linear_solver="direct"` on the CPU factor with Intel MKL PARDISO
and a METIS nested-dissection ordering instead of NGSolve's sparse Cholesky:
a smaller factor and a faster factorization, with the same results to the
direct solve's accuracy. Intel publishes MKL wheels for Linux x86_64 and
Windows x86_64 only; elsewhere the extra installs nothing. MKL and its runtime
are proprietary Intel software, and Intel's OpenMP runtime comes under a
licence that forbids SaaS and service-bureau use: read {doc}`project/licence`
before installing the extra.

PETSc and SLEPc are not an extra: gfem needs them built from source with
complex scalars, which {doc}`recipes/petsc-slepc` does with one script. Then
select them with `linear_solver="petsc"` and `eigen_solver="slepc"`.

Supported platforms: Linux x86_64 (glibc >= 2.28), macOS arm64 and Windows
x86_64, Python >= {{ python }}.

Importing gfem turns on JAX double precision (`jax_enable_x64`), which every
gfem solve requires.

## Next

- The {doc}`topology tutorial <tutorials/topology>` runs a first solve and
  differentiates it.
- {doc}`capabilities` lists what gfem solves and differentiates.
