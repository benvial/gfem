# Design a patterned metal sheet

Optimize the pattern of a thin metal film on a dielectric stack (an RF film,
a superconducting chip layer), pixel by pixel, with `jax.grad`. The film is a
sheet: a zero-thickness interior surface of the mesh, as gsim meshes it. A
{py:obj}`gfem.SheetField` holds one real **fill** per face of that surface and
scales the admittance of the {py:obj}`gfem.Conductor` or
{py:obj}`gfem.Impedance` on it, `Y_f(ω) = fill_f · Y_solid(ω)`. Fill 0 is no
metal (plain dielectric continuity), 1 the solid sheet.

## The chain

```python
import jax
import jax.numpy as jnp

import gfem
from gfem import design

grid = design.LayerGrid(mesh, "M1_design", pitch=2e-6)  # pixels over the sheet
blur = design.helmholtz_filter(grid, radius=20e-6)  # minimum length scale
problem = gfem.Driven(frequency=(5e9, 8e9))
options = gfem.SolverOptions(linear_solver="direct")


def objective(rho, beta):
    density = grid.to_faces(design.project(blur(rho), beta=beta))
    fill = gfem.SheetField.interpolate(density)  # exponential, contrast 1e10
    metal = gfem.Conductor("M1_design", conductivity=5.8e7, thickness=2e-6, fill=fill)
    s = gfem.solve(
        model.replace(boundaries=[*fixed, metal]), problem, options=options
    ).s
    return jnp.abs(s["p2", "p1"][1]) ** 2


rho = jnp.full(grid.shape, 0.5)
value, grad = jax.value_and_grad(objective)(rho, 4.0)
```

`LayerGrid` on a surface needs every face horizontal (normal along mesh z) and
maps pixels to faces by bilinear interpolation at the face centroids
(`grid.to_faces`; `to_elements` is the volume map). `SheetField.interpolate`
also takes any per-face density of your own.

## Which surfaces

- **Interior surfaces only**: the solve domain must be on both sides of every
  face, so that fill 0 is a true void. A sheet field on an exterior surface or
  on the shell of an excluded volume raises when the model is built.
- **A `Conductor` or `Impedance` solid**: a `PEC` has no admittance to scale.
  For a superconductor use its kinetic inductance, `Impedance(inductance=L_k)`
  (0.1-10 pH per square is PEC-like for the fields at GHz); for a normal metal
  `Conductor(sigma, thickness)`. On an interior surface a thin film of
  thickness `t` is a sheet admittance of about `sigma t / 2` (one face term;
  see `gfem.conventions`).
- **Driven (uniform and adaptive) and eigenmode problems.** An electrostatic
  problem raises: a partly filled sheet is not a fixed-potential conductor.

## The interpolation

The fields feel a sheet over a window of admittances: between about
`|Y| = 0.1 / eta` and `10 / eta` (eta the wave impedance around the sheet), it
goes from transparent to nearly a short. A fill linear in the admittance would
put that whole window in the first fraction of a percent of the density. So
`interpolate` is exponential by default, `w = (r^rho - 1) / (r - 1)`: log-scale
in the admittance, with exact 0 and 1 at the ends. The window's centre is at
about

    rho_c ≈ 1 - log10(eta |Y_solid|) / log10(contrast)

which is 0.4-0.8 for metal films with the default `contrast = 1e10`. Lower the
contrast for a weak solid (a resistive film of `eta |Y| ~ 1`). Grey resistive
sheets absorb (up to about 40 % of a plane wave), so the optimizer will use
grey unless the design is projected: raise `beta` of `design.project` in steps
as for any density method. Grey inductive sheets stay lossless.

## Solver

Use `direct` for inductive sheet designs, and for the small, repeated solves
of an optimization loop: it factorizes once per frequency and is insensitive
to the contrast. `bddc_gmres`, the driven default, suits large models of resistive
or conductor sheets. On any sheet design, check its iteration counts at your
design's extremes (all void, all solid) and at a grey fill.

A design sweeps the sheet's admittance over many orders of magnitude, and a
solid metal sheet is PEC-like (`eta |Y|` of 1e4 for 2 µm of copper). That
contrast alone does not slow BDDC + GMRES: solid, checkerboard and random
fills of a copper film or of a kinetic inductance take about the iterations of
an empty sheet.

What slows it is a faint **inductive** face. A face of an inductive sheet (an
`Impedance` with an `inductance`, a kinetic-inductance film) at fill `w` has a
reactance per square `omega L / w`. Where that is about

    X ≈ eta0 lambda0 / (2 h (eps1 + eps2))

the sheet resonates at the drive frequency in a quasi-static mode about `2 h`
long (a sheet plasmon at the mesh scale). Here `h` is the mesh size along the
sheet and `eps1`, `eps2` the relative permittivities on its two sides, so `X`
is tens of `eta0` on usual meshes. Grey and exponentially interpolated fills
of an inductive sheet put some faces there, as does a uniform fill tuned onto
the resonance. The GMRES iterations then grow several times, and more on a
finer mesh, which resonates over a wider range of fills. A solve that reaches
the iteration limit raises {py:obj}`gfem.LinearSolverConvergenceError`, naming
`direct`; switch to it. Resistive films and conductors are barely affected.

## Gradients

`d S / d fill` costs no extra solve beyond the adjoint solve of the other
inputs, and an eigenfrequency's `d omega / d fill` needs no extra solve at
all. An adaptive sweep differentiates the fill like a material, with frozen
samples ({doc}`adaptive-sweep-gradient`). Shape gradients see the sheet too:
node coordinates differentiate through it like any other surface term, and the
fill stays attached to its face.

Three cases raise:

- A perturbed fill on a sheet that shares an edge with a **wave port** face
  raises {py:obj}`gfem.PortGeometryError`: the port mode reads the sheet there
  and is a constant of the solve. Split the sheet so the part touching the
  port is a boundary of its own, or `jax.lax.stop_gradient` the fill. The
  forward solve works: the port mode applies the fill on its edges.
- A negative fill raises `ValueError` in the forward solve: the sheet would be
  active. `SheetField.interpolate` of a density in [0, 1] keeps it passive.
- The solid's own values (conductivity, thickness, R, L, C) are fixed numbers:
  differentiating them raises {py:obj}`gfem.NonDifferentiableError`. Keep them
  plain numbers and differentiate the fill.

## Next

{py:obj}`gfem.SheetField`, {py:obj}`gfem.design.LayerGrid`,
{py:obj}`gfem.design.project`, {doc}`s-parameter-gradient`,
{doc}`robust-optimization`, {doc}`../capabilities`.
