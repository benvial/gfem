# Differentiate a capacitance

`gfem.solve` is a JAX function of the model's array leaves. `jax.grad` of any
objective built from a result reaches the per-element values of every
`MaterialField` in the model, without you writing an adjoint.

## A gradient with respect to a permittivity field

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


def mutual(eps):
    """C(top, mid) in fF, for a permittivity per oxide element."""
    m = model.with_material("oxide", gfem.MaterialField(eps_r=eps))
    return gfem.solve(m, problem).capacitance[0, 1] * 1e15


n = len(mesh.elements("oxide"))
value, grad = jax.value_and_grad(mutual)(jnp.full(n, 3.9))  # grad: (n,)
```

The capacitance gradient costs no extra solve: each terminal's potential is its
own adjoint, so the backward pass is one element-wise integral. To get the
sensitivity to a whole region's constant permittivity, assign
`gfem.MaterialField.uniform("oxide", gfem.Material(eps_r=3.9))`: its values are
scalar leaves. A `Material`'s values are fixed constants; tracing one raises
`gfem.NonDifferentiableError` with that remedy.

`jax.jit` and `jax.vmap` work on the same functions. Under `vmap` the solves
run one after the other.

## What every other input gives

Every input of the model has one of three outcomes:

- **a gradient**, for what the problem reads and gfem differentiates (for
  electrostatics: `eps_r`, the real part when it is complex);
- **an exact zero**, for what the problem never reads. Electrostatics does not
  read `conductivity`, `tan_delta`, `mu_r`, the thermal properties or the
  elements of excluded volumes, so one `MaterialField` can serve an
  electrostatic and a driven objective together;
- **`gfem.NonDifferentiableError`**, raised while JAX traces the gradient,
  before any solve, for what the problem reads but gfem cannot differentiate
  yet (in this version: the mesh node coordinates). The message names the input
  and the remedy.

gfem never returns a silent zero for an input the physics reads.

## Opting out: `jax.lax.stop_gradient`

`jax.lax.stop_gradient` is the one way to use a value without its derivative.
For example, `jax.grad` over a whole model also perturbs the mesh nodes, which
raises; freeze them first:

```python
def objective(model):
    frozen = model.replace(mesh=jax.lax.stop_gradient(model.mesh))
    return gfem.solve(frozen, problem).capacitance[1, 1]


grads = jax.grad(objective)(model)  # grads.materials[...] holds the field gradients
```

The same works on an output that has no gradient: wrap it in
`stop_gradient` where the objective uses it. Of the electrostatic outputs,
`capacitance` and the matrices derived from it (`mutual`, `inverse`)
differentiate; the per-region `energy` does not yet, and raises when a
gradient through it reaches `eps_r`.

Errors that only the data can reveal are raised on the host during the
backward pass. JAX re-raises them as `jax.errors.JaxRuntimeError`; the message
keeps the gfem error name and its remedy.
