<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/gfem-lockup-dark.svg">
    <img alt="gfem" src="docs/_static/logo/gfem-lockup-light.svg" height="64">
  </picture>
</h1>

Differentiable finite-element electromagnetics for RF, microwave, mm-wave and superconducting quantum devices.
gfem runs on [NGSolve](https://ngsolve.org) for discretization and [JAX](https://jax.readthedocs.io) for
everything differentiable.

gfem 0.1.0 is the first release. The API may still change before 1.0.

## Install

gfem is not on PyPI yet: pip installs it from the repository.

```bash
pip install "gfem @ git+https://github.com/benvial/gfem"
```

Each optional extra goes in brackets after `gfem`:

```bash
pip install "gfem[mkl] @ git+https://github.com/benvial/gfem"     # a faster CPU direct solver, MKL PARDISO (Linux x86_64, Windows); read the licence first
pip install "gfem[cuda12] @ git+https://github.com/benvial/gfem"  # the direct solver on an NVIDIA GPU, cuDSS (Linux x86_64, Windows); cuda13 for CUDA 13
pip install "gfem[xarray] @ git+https://github.com/benvial/gfem"  # results as xarray datasets (.to_xarray())
pip install "gfem[viz] @ git+https://github.com/benvial/gfem"     # fields as PyVista grids (.to_pyvista())
pip install "gfem[mpi] @ git+https://github.com/benvial/gfem"     # one driven solve across MPI ranks (Linux x86_64; Open MPI from your machine)
```

Supported platforms: Linux x86_64 (glibc >= 2.28), macOS arm64 and Windows x86_64, Python >= 3.12.

Importing gfem turns on JAX double precision (`jax_enable_x64`), which every gfem solve requires.

## A first solve

A parallel-plate line 40 mm long, 6 mm wide and 1 mm high joins two lumped ports matched to it. Its middle 20 mm
is filled with a ceramic of relative permittivity 10. How much of the incident power reaches port 2 at 2, 5 and
8 GHz?

```python
import urllib.request

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
urllib.request.urlretrieve(url, "line.msh")
mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")

z_line = 376.73 / 6  # ohm: the 6 mm wide, 1 mm high parallel-plate line in air
model = gfem.Model(
    mesh,
    materials={"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)},
    boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    ports=[
        gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
        gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
    ],
)
result = gfem.solve(model, gfem.Driven(frequency=(2e9, 5e9, 8e9), excite="p1"))
print(abs(result.s["p2", "p1"]))  # |S21| at 2, 5 and 8 GHz
```

```text
[0.8306221  0.90133232 0.60517623]
```

[Get started](https://benvial.github.io/gfem/get-started.html) explains each line, checks the numbers by hand and differentiates the
result.

## Documentation

The documentation is at <https://benvial.github.io/gfem/>. Good places to start:

- [Get started](https://benvial.github.io/gfem/get-started.html): install, a first solve and its gradient.
- [Tutorials](https://benvial.github.io/gfem/tutorials/topology.html): a band-stop filter, a coplanar line matched to 50 Ω and photonic
  waveguide bends, each designed with gradients.
- [PETSc and SLEPc](https://benvial.github.io/gfem/recipes/petsc-slepc.html): the PETSc and SLEPc solvers, which need a source build.
- [Distributed solve](https://benvial.github.io/gfem/recipes/distributed-solve.html): one driven solve across the ranks of an MPI job.

## Licence

MIT. gfem depends on NGSolve (LGPL-2.1), whose PyPI wheels also contain GPL-2.0+ SuiteSparse code (UMFPACK, CHOLMOD
Supernodal); keep that in mind when redistributing a whole gfem environment. The `mkl` and `cuda12`/`cuda13`
extras pull proprietary vendor libraries: see [Licence](https://benvial.github.io/gfem/project/licence.html).

## Contributing

To build gfem, run its tests or rebuild the docs, see [Contributing](https://benvial.github.io/gfem/project/development.html).
