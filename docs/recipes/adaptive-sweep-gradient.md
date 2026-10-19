# Differentiate an adaptive frequency sweep

An adaptive sweep (`Driven(sweep="adaptive")`) solves a few frequencies in
full and reads S and the kept fields at every frequency of the grid from a
reduced model built on those solutions. Its gradients hold the reduced
basis, and the sample frequencies it came from, fixed:

- **At a sample frequency** the gradient equals the full solve's gradient to
  the solver tolerance (both the field and its adjoint lie in the basis, when
  every port the objective reads is excited).
- **Between samples** its error follows the reduced model's field error,
  about `SolverOptions.adaptive_tolerance` or better.
- **It is not the exact derivative of the reduced model**: how the basis
  itself moves with the design is left out. Finite differences of an adaptive
  objective therefore agree with it only to the sweep's accuracy, except at
  the sample frequencies.

The backward pass solves no full-size system and keeps no solver: one small
reduced adjoint per frequency, then a few element-wise integrals (at most one
per basis vector and material term, however many frequencies the grid has).

## Freeze the samples across optimizer iterations

Under `jax.grad`, `jax.jit` or `jax.vmap` an adaptive sweep needs a fixed
sample set, `Driven(samples=...)`: the greedy's choice is only known after
the solve. Let the greedy pick the samples once, outside the transformation,
then reuse them:

```python
import jax
import jax.numpy as jnp
import numpy as np

import gfem

mesh = gfem.Mesh.from_gmsh("filter.msh")
model = gfem.Model(
    mesh,
    materials={"substrate": gfem.Material(eps_r=4.4)},
    boundaries=[gfem.PEC("ground"), gfem.Absorbing("outer")],
    ports=[
        gfem.LumpedPort("p1", "port1", "+z"),
        gfem.LumpedPort("p2", "port2", "+z"),
    ],
)
grid = np.linspace(1e9, 20e9, 300)
first = gfem.solve(model, gfem.Driven(frequency=grid, sweep="adaptive"))
problem = gfem.Driven(
    frequency=grid, sweep="adaptive", samples=first.sweep.sampled_frequencies
)


def loss(eps):
    m = model.with_material("substrate", gfem.MaterialField(eps_r=eps))
    s = gfem.solve(m, problem).s
    return jnp.mean(1 - jnp.abs(s["p2", "p1"]) ** 2)


eps = jnp.full(len(mesh.elements("substrate")), 4.4)
for _ in range(20):
    value, grad = jax.value_and_grad(loss)(eps)
    eps = eps - 0.1 * grad
```

With the samples frozen the objective is a smooth function of the design. If
the greedy re-ran at every iteration, a small design step could change the
sample set and make the objective jump by about the sweep tolerance, which
confuses line searches.

## Re-sample with a tight tolerance

When the design has moved far from where the samples were picked (the
response has shifted, so the frozen samples no longer resolve it), pick new
ones the same way: a greedy solve outside the gradient, then frozen samples.
Run that greedy at `adaptive_tolerance <= 1e-6`, so that the objective's jump
at the switch stays below the optimizer's resolution:

```python
tight = gfem.SolverOptions(adaptive_tolerance=1e-6)
m = model.with_material("substrate", gfem.MaterialField(eps_r=eps))
fresh = gfem.solve(m, gfem.Driven(frequency=grid, sweep="adaptive"), options=tight)
problem = gfem.Driven(
    frequency=grid, sweep="adaptive", samples=fresh.sweep.sampled_frequencies
)
```

A tight tolerance costs a few more samples (each a full solve) and makes the
gradient accurate between samples too. Check `fresh.sweep.converged` before
freezing its samples.

## What a traced solve reports

Under a transformation `result.sweep` gives the frozen `sampled_frequencies`
and `solved` mask, while `sample_errors`, `converged` and `basis_size` are
`None` (known only after the solve). The kept fields' basis is padded with
zero columns to its largest size, two per sample and excited port, so its
shape is known before the solve. Gradients reach the fields through their
reduced coordinates; differentiating `result.e_fields.basis` itself raises
`gfem.NonDifferentiableError`.
