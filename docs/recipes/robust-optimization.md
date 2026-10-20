# Optimize a design that tolerates the etch

A density design is fabricated with an etch bias: edges move outwards
(under-etch, more material) or inwards (over-etch, less material).
`gfem.design.robust` gives the three variants of one filtered density, and
the optimizer works on the worst of them. The result keeps its performance
across the bias, and both the solid and the void get a minimum feature size
set by the filter radius. The full worked example is the
[topology tutorial](../tutorials/topology.ipynb).

## The chain

```python
import jax
import jax.numpy as jnp

import gfem
from gfem import design

grid = design.LayerGrid(mesh, "design", pitch=0.5e-3)  # pixels over the layer
blur = design.helmholtz_filter(grid, radius=1.5e-3)  # minimum length scale
AIR, CERAMIC = gfem.Material(), gfem.Material(eps_r=10.0)


def transmission(density):
    """|S21|^2 of one design (one density per pixel)."""
    eps = gfem.MaterialField.interpolate(grid.to_elements(density), AIR, CERAMIC)
    s = gfem.solve(model.with_material("design", eps), problem).s
    return jnp.abs(s["p2", "p1"][0]) ** 2


def worst_loss(rho, beta):
    variants = design.robust(blur(rho), beta=beta, delta=0.15)
    losses = jax.vmap(lambda d: 1 - transmission(d))(jnp.stack(variants))
    return jnp.max(losses)  # eroded, nominal, dilated: the worst one


value, grad = jax.value_and_grad(worst_loss)(rho, 4.0)
```

- **Eroded** is the projection at the higher threshold `eta + delta` (less
  solid), **dilated** at `eta - delta` (more solid). `delta` sets the etch
  tolerance and has no default.
- `jax.vmap` over `gfem.solve` runs the three solves one after the other:
  each optimizer step costs three forward solves and three backward passes.
- `jnp.max` is not smooth where two variants tie. Adam copes; for a
  gradient-based constrained optimizer (MMA, SLSQP) use the epigraph form:
  minimize `t` subject to `loss_i(rho) <= t` for the three variants.
- A volume constraint goes on the **dilated** design, the one with the most
  material.
- Raise `beta` in steps (2, 4, 8, 16, ...) as the design converges: a soft
  projection lets material move, a sharp one makes the design binary.

## Fixed regions and symmetry

Both are one line on the pixel array, before the filter:

```python
rho = jnp.where(feed_mask, 1.0, rho)  # pin solid feed connections
rho = (rho + rho[:, ::-1]) / 2  # mirror symmetry about the grid's centre line in x
rho = jnp.where(grid.sampled, rho, 0.0)  # void outside a non-rectangular footprint
```
