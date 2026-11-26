<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/benvial/gfem/main/docs/_static/logo/gfem-lockup-dark.svg">
    <img alt="gfem" src="https://raw.githubusercontent.com/benvial/gfem/main/docs/_static/logo/gfem-lockup-light.svg" height="64">
  </picture>
</h1>

Differentiable finite-element electromagnetics, heat and mechanics for photonic, RF, microwave and superconducting quantum devices.
gfem runs on [NGSolve](https://ngsolve.org) for discretization and [JAX](https://jax.readthedocs.io) for
everything differentiable.

gfem 0.2.0 is the current release. The API may still change before 1.0.

## Install

```bash
pip install gfem
```

Each optional extra goes in brackets after `gfem`:

```bash
pip install "gfem[mkl]"     # a faster CPU direct solver, MKL PARDISO (Linux x86_64, Windows); read the licence first
pip install "gfem[cuda12]"  # the direct solver on an NVIDIA GPU, cuDSS (Linux x86_64, Windows); cuda13 for CUDA 13
pip install "gfem[xarray]"  # results as xarray datasets (.to_xarray())
pip install "gfem[viz]"     # fields as PyVista grids (.to_pyvista())
pip install "gfem[mpi]"     # one solve across MPI ranks (Linux x86_64; Open MPI from your machine)
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
[0.83062238 0.90133217 0.60517471]
```

[Get started](https://bvial.info/gfem/get-started.html) explains each line, checks the numbers by hand and differentiates the
result.

## Documentation

The documentation is at <https://bvial.info/gfem/>. [Get started](https://bvial.info/gfem/get-started.html)
installs gfem, runs a first solve and differentiates it. Each physics domain opens with its tutorials:

- [Integrated photonics](https://bvial.info/gfem/tutorials/bend.html): waveguide bends in 2D and 3D, a grating
  coupler, dispersion and multimode ports.
- [Nonlinear and quantum optics](https://bvial.info/gfem/recipes/waveguide-nonlinearity.html): Kerr nonlinearity,
  second-harmonic generation and photon-pair sources.
- [Electro-optics](https://bvial.info/gfem/tutorials/modulator.html): a lithium niobate modulator and the phase a
  voltage puts on a waveguide.
- [Heat and thermo-optics](https://bvial.info/gfem/tutorials/heater.html): heaters, temperature over time, heat
  from electromagnetic loss and the phase a heater puts on a waveguide, in 2D and 3D.
- [Mechanics and acoustics](https://bvial.info/gfem/tutorials/transducer.html): mechanical modes, piezoelectric
  transducers, and acousto-optic and optomechanical coupling.
- [Free-space scattering](https://bvial.info/gfem/tutorials/nanosphere.html): cross sections, far fields,
  multipoles, dipole emission and optical force.
- [Periodic surfaces](https://bvial.info/gfem/tutorials/metagrating.html): a metagrating, a photonic band gap, a
  two-colour metalens, diffraction efficiencies and emissivity.
- [RF and microwave](https://bvial.info/gfem/tutorials/topology.html): a band-stop line, a matched attenuator, a
  coplanar line matched to 50 Ω, time-domain reflectometry and SPICE export.
- [Resonators and qubits](https://bvial.info/gfem/tutorials/transmon.html): a transmon's dispersive shift,
  quantum metrics and loss budgets.
- Infrastructure: [meshes, materials and boundaries](https://bvial.info/gfem/recipes/prepare-a-mesh.html),
  [design workflow](https://bvial.info/gfem/recipes/shape-gradient.html),
  [solvers and scale-out](https://bvial.info/gfem/recipes/watch-a-solve.html) (GPU, PETSc and SLEPc from a source
  build, MPI) and [results and files](https://bvial.info/gfem/recipes/measurements.html).
- [Theory](https://bvial.info/gfem/conventions.html): the sign, unit and normalization conventions, and modes and
  ports.
- [Capabilities](https://bvial.info/gfem/capabilities.html): what each problem type gives and which gradients are
  supported; [verification](https://bvial.info/gfem/verification.html) shows how each claim is checked.

## Licence

MIT. gfem depends on NGSolve (LGPL-2.1), whose PyPI wheels also contain GPL-2.0+ SuiteSparse code (UMFPACK, CHOLMOD
Supernodal); keep that in mind when redistributing a whole gfem environment. The `mkl` and `cuda12`/`cuda13`
extras pull proprietary vendor libraries: see [Licence](https://bvial.info/gfem/project/licence.html).

## Contributing

To build gfem, run its tests or rebuild the docs, see [Contributing](https://bvial.info/gfem/project/development.html).
