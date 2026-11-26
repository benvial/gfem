# Adaptive-sweep gradients

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
  the solver tolerance, for every S entry that is defined, whether its port
  is excited or not.
- **Between samples** its error follows the reduced model's field error,
  about `SolverOptions.adaptive_tolerance` or better. This holds for samples
  taken from a converged adaptive solve (`result.sweep.converged`). A
  hand-picked or short sample set gives a gradient as accurate as its own
  reduced model: on a two-port line, three hand-picked samples left S off by
  up to 3.5e-3 and the gradient between samples off by 0.5 % to 27 %. To check
  a sample set, solve once outside the gradient with the same `samples=` and
  compare S with a uniform sweep at a few frequencies. Under frozen samples
  `result.sweep.converged` reads the errors in the order the samples are
  given, so it is meaningful only for `sampled_frequencies` passed unchanged,
  at the design they were chosen for.
- **The S row of a lumped port without a resistance is undefined** (NaN), and
  so is the gradient of an objective that reads it. The other rows of the
  same solve keep their gradient.
- **It is not the exact derivative of the reduced model**: how the basis
  itself moves with the design is left out. Finite differences of an adaptive
  objective therefore agree with it only to the sweep's accuracy, except at
  the sample frequencies. For a tighter gradient between samples, re-sample
  with a tight tolerance (next section).

The backward pass (the part of `jax.grad` that runs after the solve) solves
no full-size system and keeps no solver. It solves one small reduced adjoint
per frequency, then a few element-wise integrals: at most one per basis vector
and material term, however many frequencies the grid has. Each port that is
not excited adds one right-hand side per sample to
the sweep, which keeps the gradient exact at the samples. An adaptive sweep
with `Driven(excite=...)` therefore solves as many right-hand sides per
sample as one that excites every port; these run in every adaptive solve,
under `jax.grad` or not, so a value does not depend on whether its gradient
is taken.

A design at a wave port's face (the material of a region that touches it) moves
the port's mode too. The backward then also differentiates each row's mode
exactly, on the traces of the reduced solution and adjoint, for every row whose
outputs have a cotangent; it holds the basis and the samples like the rest, so
the gradient is exact at the samples. The mode's 2D factors are built during the
backward and freed after it. It adds, at each frequency whose output the
objective reads, a small 2D solve per port mode on that face; on a long
broadband grid this can cost as much as a few full factorizations.

## Re-sample with a tight tolerance

When the design has moved far from where the samples were picked, the
response has shifted and the frozen samples no longer resolve it. Pick new
ones the same way: an adaptive solve outside the gradient, then frozen
samples. Run that solve at `adaptive_tolerance <= 1e-6`, with `tolerance` at
most a tenth of it, so that the objective's jump at the switch stays below the
optimizer's resolution:

```python
tight = gfem.SolverOptions(tolerance=1e-7, adaptive_tolerance=1e-6)
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
