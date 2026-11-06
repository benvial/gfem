# Differentiate a capacitance

Get the derivative of a capacitance with respect to the permittivity of every
element, without writing an adjoint. `jax.grad` of any objective built from
an electrostatic result reaches the per-element values of every
{py:obj}`gfem.MaterialField` in the model. This page also holds what the other
gradient recipes share: batches of designs, the memory kept for gradients,
`jax.lax.stop_gradient`, and what to do when a gradient raises.

## A mutual capacitance against a permittivity field

The mesh `stack.msh` has an oxide and a nitride layer, terminals on its top
face and on the oxide-nitride interface, and the ground on its bottom face.

```python
import jax
import jax.numpy as jnp

import gfem

mesh = gfem.Mesh.from_gmsh("stack.msh")
model = gfem.Model(
    mesh,
    materials={"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)},
)
problem = gfem.Electrostatic(
    terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
)


def coupling(eps):
    """Mutual capacitance top-mid in fF, for a permittivity per oxide element."""
    m = model.with_material("oxide", gfem.MaterialField(eps_r=eps))
    return gfem.solve(m, problem).mutual[0, 1] * 1e15


n = len(mesh.elements("oxide"))
value, grad = jax.value_and_grad(coupling)(jnp.full(n, 3.9))  # grad: (n,), fF
```

- `grad[i]` is the change of the mutual capacitance, in fF, per unit change of
  the relative permittivity of oxide element `i`.
- `result.mutual` and `result.inverse` differentiate like
  `result.capacitance`.
- The region energies `result.energy` have no gradient yet: a gradient
  through them raises {py:obj}`gfem.NonDifferentiableError`. Use
  `capacitance[i, i] / 2`, the total energy of excitation `i`, instead.
- The gradient costs no extra solve. Each terminal's potential is its own
  adjoint, so the backward pass (the part of `jax.grad` that runs after the
  solve) is one element-wise integral.

## Differentiate a whole region's permittivity

For the sensitivity to one constant permittivity of a whole region, assign
`gfem.MaterialField.uniform("oxide", gfem.Material(eps_r=3.9))`. Its values
are scalars, so the gradient has one value per property. A
{py:obj}`gfem.Material`'s values are fixed constants: differentiating with
respect to one raises `gfem.NonDifferentiableError`, whose message gives this
remedy.

## Several designs at once

`jax.jit` and `jax.vmap` work on the same functions. Under `jax.vmap` the
solves run one after the other, so `jax.vmap(jax.grad(...))` over a batch of
designs (an eroded, nominal and dilated triple, say) gives what a Python loop
over them gives:

```python
designs = jnp.stack([eps_eroded, eps_nominal, eps_dilated])  # (3, n)
grads = jax.vmap(jax.grad(coupling))(designs)  # (3, n)
robust = jax.grad(lambda ds: jnp.max(jax.vmap(coupling)(ds)))(designs)
```

## Several gradients from one solve

`jax.jacrev` of the whole capacitance matrix, and the function `jax.vjp`
returns called several times, give the same numbers as separate `jax.grad`
calls. Between the solve and its gradients gfem keeps what the gradients need
(for a capacitance, the terminal potentials):

- It keeps up to `SolverOptions(retain_memory=...)` bytes, half of the
  physical memory by default, and drops the oldest entries first.
- A dropped entry is recomputed from the design when its gradient runs. This
  costs time, never accuracy.
- `retain_memory=0` keeps nothing and always recomputes. Use it when memory,
  not time, is short.

## What every other input gives

Every input of the model has one of three outcomes:

- **a gradient**, for what the problem reads and gfem differentiates (for
  electrostatics: `eps_r`, the real part when it is complex, and the mesh
  node coordinates, see {doc}`shape-gradient`);
- **an exact zero**, for what the problem never reads. Electrostatics does not
  read `conductivity`, `tan_delta`, `mu_r`, the thermal properties or the
  elements of excluded volumes, so one `MaterialField` can serve an
  electrostatic and a driven objective together;
- **`gfem.NonDifferentiableError`**, raised when `jax.grad` is called, before
  any solve, for what the problem reads but gfem cannot differentiate yet (an
  output such as `result.energy`). The message names the input or output and
  the remedy.

gfem never returns a silent zero for an input the physics reads.

## Use a value without its derivative

`jax.lax.stop_gradient` is the one way to use a value without its derivative.
For example, `jax.grad` over a whole model also differentiates the mesh
nodes, a shape gradient you may not need (it costs one more pass over the
elements). Freeze them first:

```python
def objective(model):
    frozen = model.replace(mesh=jax.lax.stop_gradient(model.mesh))
    return gfem.solve(frozen, problem).capacitance[1, 1]


grads = jax.grad(objective)(model)  # grads.materials[...] holds the field gradients
```

The same works on an output that has no gradient: wrap it in
`stop_gradient` where the objective uses it.

(gradient-errors)=

## When a gradient raises

Some errors can only be found while `jax.grad` computes the gradient, after
the solve has succeeded. Examples are a port node moved by a shape update
({py:obj}`gfem.PortGeometryError`) and a degenerate mode cluster your objective
does not treat symmetrically ({py:obj}`gfem.DegenerateModeError`). JAX does not
pass gfem's error class through at that stage. You get a
`jax.errors.JaxRuntimeError` or, from a jitted function that has already run
successfully, a `ValueError`. The message is gfem's own: it names the error
and how to fix it. Catch those two classes and read the message:

```python
try:
    grad = jax.grad(coupling)(eps)
except (jax.errors.JaxRuntimeError, ValueError) as error:
    print(error)  # names the gfem error and its remedy
```

## Check your results

- Compare the gradient with a central difference of your objective along a
  random direction `v`: `(f(p + h v) - f(p - h v)) / (2 h)` against
  `grad @ v`. Their difference shrinks as `h` shrinks, until the solver
  tolerance limits it.

How gfem itself is checked: {doc}`../verification`.

## Limits

- Only reverse mode is supported: `jax.jvp` and `jax.jacfwd` of a solve raise
  JAX's own `TypeError`. Use `jax.grad`, `jax.vjp` or `jax.jacrev`.

## Next

{py:obj}`gfem.Electrostatic`, {py:obj}`gfem.MaterialField`,
{py:obj}`gfem.SolverOptions`; {doc}`s-parameter-gradient`,
{doc}`eigenmode-gradient`, {doc}`shape-gradient`; {doc}`../capabilities`.
