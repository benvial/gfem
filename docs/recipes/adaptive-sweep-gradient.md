# Differentiate an adaptive frequency sweep

Optimize over a broadband response without a full solve at every frequency:
differentiate an adaptive sweep with its samples frozen. An adaptive sweep
(`Driven(sweep="adaptive")`, see {doc}`frequency-sweep`) solves a few
frequencies in full and reads S and the kept fields at every frequency of the
grid from a reduced model built on those solutions. Its gradient holds the
reduced basis, and the sample frequencies it came from, fixed.

## Freeze the samples, then optimize

Under `jax.grad`, `jax.jit` or `jax.vmap` an adaptive sweep needs a fixed
sample set, `Driven(samples=...)`, because the sweep chooses its samples only
during the solve. Let the sweep pick the samples once, outside the gradient,
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
    ports=[gfem.LumpedPort("p1", "port1", "+z"), gfem.LumpedPort("p2", "port2", "+z")],
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

- With the samples frozen the objective is a smooth function of the design.
  If the samples were chosen again at every iteration, a small design step
  could change them. The objective would then jump by about the sweep
  tolerance, which confuses line searches.
- Under `jax.grad`, `jax.jit` or `jax.vmap`, `result.sweep` gives the frozen
  `sampled_frequencies` and the `solved` mask. `sample_errors`, `converged`
  and `basis_size` are `None`, since they are known only after the solve; read
  them from the solve outside the gradient.

## How accurate the gradient is

- **At a sample frequency** the gradient equals the full solve's gradient to
  the solver tolerance, when every port the objective reads is excited.
- **Between samples** its error follows the reduced model's field error,
  about `SolverOptions.adaptive_tolerance` or better.
- **It is not the exact derivative of the reduced model**: how the basis
  itself moves with the design is left out. Finite differences of an adaptive
  objective therefore agree with it only to the sweep's accuracy, except at
  the sample frequencies. For a tighter gradient between samples, re-sample
  with a tight tolerance (next section).

The backward pass (the part of `jax.grad` that runs after the solve) solves
no full-size system and keeps no solver. It solves one small reduced adjoint
per frequency, then a few element-wise integrals: at most one per basis vector
and material term, however many frequencies the grid has. A lossy wave port
in the objective adds one right-hand side per sample to the sweep, which keeps
the gradient exact at the samples.

## Re-sample with a tight tolerance

When the design has moved far from where the samples were picked, the
response has shifted and the frozen samples no longer resolve it. Pick new
ones the same way: an adaptive solve outside the gradient, then frozen
samples. Run that solve at `adaptive_tolerance <= 1e-6`, so that the
objective's jump at the switch stays below the optimizer's resolution:

```python
tight = gfem.SolverOptions(adaptive_tolerance=1e-6)
m = model.with_material("substrate", gfem.MaterialField(eps_r=eps))
fresh = gfem.solve(m, gfem.Driven(frequency=grid, sweep="adaptive"), options=tight)
problem = gfem.Driven(
    frequency=grid, sweep="adaptive", samples=fresh.sweep.sampled_frequencies
)
```

- A tight tolerance costs a few more samples, each a full solve, and makes the
  gradient accurate between samples too.
- Check `fresh.sweep.converged` before freezing its samples.

## Check your results

- Compare the gradient with a central difference of an objective read at one
  sample frequency: there the two agree to the solver tolerance. Between
  samples, expect agreement only to the sweep's accuracy.

How gfem itself is checked: {doc}`../verification`.

## Limits

- Differentiating the kept fields' reduced basis, `result.e_fields.basis`,
  raises `gfem.NonDifferentiableError`. Differentiate the fields themselves
  (`result.e_field(f, port)`), whose gradient reaches the design through
  their reduced coordinates.

## Next

{py:obj}`gfem.Driven`, {py:obj}`gfem.SolverOptions`; {doc}`frequency-sweep`,
{doc}`s-parameter-gradient`; {doc}`../capabilities`.
