# Patterned metal sheets

Optimize the pattern of a thin metal film on a dielectric stack (an RF film,
a superconducting chip layer), pixel by pixel, with `jax.grad`. The film is a
sheet: a zero-thickness interior surface of the mesh, as gsim meshes it. A
{py:obj}`gfem.SheetField` holds one real **fill** per face of that surface and
scales the admittance of the {py:obj}`gfem.Conductor` or
{py:obj}`gfem.Impedance` on it, `Y_f(ω) = fill_f · Y_solid(ω)`. Fill 0 is no
metal (plain dielectric continuity), 1 the solid sheet.

The [resistive film tutorial](../tutorials/sheet.ipynb) runs a whole sheet
optimization, from pixels to a binary film and its checks.

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

The fields feel a sheet over a window of admittances, between about
`|Y| = 0.1 / eta` and `10 / eta` (eta the wave impedance around the sheet),
where it goes from transparent to nearly a short. A fill linear in the
admittance would put that whole window in the first fraction of a percent of
the density. So `interpolate` is exponential by default,
`w = (r^rho - 1) / (r - 1)`: log-scale in the admittance, with exact 0 and 1
at the ends. The window's centre is at about

    rho_c ≈ 1 - log10(eta |Y_solid|) / log10(contrast)

which is 0.4-0.8 for metal films with the default `contrast = 1e10`. Lower the
contrast for a weak solid. Grey resistive sheets absorb (up to about 40 % of a
plane wave), so raise `beta` of `design.project` in steps as for any density
method; grey inductive sheets stay lossless.

## Solver

Use `direct` for inductive sheet designs and for the small, repeated solves
of an optimization loop; `auto` picks it while the factorization fits in half
the memory. `bddc_gmres`, which `auto` picks above that, suits large models of
resistive or conductor sheets, but it slows on a faint **inductive** face whose
reactance puts a quasi-static resonance at the drive frequency (grey or
exponentially interpolated fills of a kinetic-inductance film can do that; the
condition is in {py:obj}`gfem.SheetField`). A solve that reaches the iteration
limit raises {py:obj}`gfem.LinearSolverConvergenceError`, naming `direct`:
switch to it. Check the iteration counts at all void, all solid and a grey fill.

## Gradients

`d S / d fill` costs no extra solve beyond the adjoint solve of the other
inputs, and an eigenfrequency's `d omega / d fill` needs no extra solve at
all. An adaptive sweep differentiates the fill like a material, with frozen
samples ({doc}`adaptive-sweep-gradient`). Shape gradients see the sheet too:
node coordinates differentiate through it like any other surface term, and the
fill stays attached to its face. A fill on a sheet that shares an edge with a
wave port's face differentiates too: the port mode reads the fill on its
edges and follows it.

## Limits

- **Negative fill.** A negative fill raises `ValueError` in the forward solve: the sheet would be
  active. `SheetField.interpolate` of a density in [0, 1] keeps it passive.
- **The solid's values.** The solid's own values (conductivity, thickness, R, L, C) are fixed numbers:
  differentiating them raises {py:obj}`gfem.NonDifferentiableError`. Keep them
  plain numbers and differentiate the fill.

## Next

{py:obj}`gfem.SheetField`, {py:obj}`gfem.design.LayerGrid`,
{py:obj}`gfem.design.project`, {doc}`s-parameter-gradient`,
{doc}`robust-optimization`, {doc}`../capabilities`, {doc}`../verification`.
For the kinetic inductance fraction of a superconducting film, see
{ref}`sheet participation <quantum-sheet-participation>`.
