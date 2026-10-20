# Design a patterned metal sheet

Thin metal on a dielectric stack (RF films, superconducting chips) is a sheet:
gsim meshes it as a zero-thickness interior surface. A `gfem.SheetField`
patterns such a sheet face by face: it holds one real **fill** per face of the
surface and scales the admittance of the `Conductor` or `Impedance` on it,
`Y_f(ω) = fill_f · Y_solid(ω)`. Fill 0 is no metal (plain dielectric
continuity), 1 the solid sheet. The fill is a pytree leaf, so `jax.grad` of a
driven or eigenmode objective reaches it, and through it a pixel design.

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
  thickness `t` is a sheet admittance of about `sigma t / 2` (one face term,
  Palace's convention; see `gfem.conventions`).
- **Driven (uniform and adaptive) and eigenmode problems.** An electrostatic
  problem raises: a partly filled sheet is not a fixed-potential conductor.

## The interpolation

The fields feel a sheet over a window of admittances: between about
`|Y| = 0.1 / eta` and `10 / eta` (eta the wave impedance around the sheet), it
goes from transparent to nearly a short. A fill linear in the admittance would
put that whole window in the first fraction of a percent of the density, so
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

A design sweeps the sheet's admittance over many orders of magnitude, and a
solid metal sheet is PEC-like (`eta |Y|` of 1e4 for 2 µm of copper). Both
solvers are verified on sheet designs: the gradient checks pass on `direct`
and on `bddc_gmres`, and on a test line BDDC + GMRES took the same ~51
iterations (tolerance 1e-10) for a solid, a random grey and a binary fill of
a 2 µm copper film, a 1 pH kinetic-inductance sheet and a 50 ohm resistive
film, matching the direct solve to 2e-11. Prefer `direct` for the small,
repeated solves of an optimization loop (it factorizes once per frequency and
is insensitive to the contrast); keep `bddc_gmres` for large models, and check
its iteration counts on your own design's extremes (all void, all solid).

## Gradients

`d S / d fill_f` takes no extra solve beyond the adjoint of the other inputs:
one face pass over the sheet, `-θ(ω) ∫_f λ_t · u_t`, per field pair. An
eigenfrequency's `d omega / d fill_f` needs no solve at all; the frequency
derivative of a `Conductor`'s `Z_s(ω)` is part of `T'(k)`. An adaptive sweep
treats the sheet as one more design-dependent affine piece (frozen samples, as
for materials).

What raises:

- a perturbed fill on a sheet that shares an edge with a **wave port** face
  (`gfem.PortGeometryError`): the port mode reads the sheet there and is a
  constant of the solve. Split the sheet so the part touching the port is a
  boundary of its own, or `jax.lax.stop_gradient` the fill. The forward works:
  the port mode applies the fill on its edges.
- a negative fill, in the forward (passivity).
- the solid's own values (conductivity, thickness, R, L, C) are fixed numbers.

The face pass and the shape gradient both see the sheet: node coordinates
differentiate through a sheet term like any other surface term (the fill
stays attached to its face).
