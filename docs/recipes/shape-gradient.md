# Shape gradients

Optimize the shape of a device by differentiating S, a capacitance or an
eigenfrequency with respect to a few geometric parameters that move the mesh.
`gfem.solve` differentiates its results with respect to the mesh node
coordinates, `model.mesh.nodes` (metres, shape `(n_nodes, 3)`), for
electrostatic, driven (uniform sweeps, and adaptive sweeps with frozen
samples) and eigenmode problems (the eigenfrequencies, as for material
gradients). A shape design is a JAX function from a few parameters to moved
node coordinates, usually through {py:obj}`gfem.shape.boundary_morph`.

## Widen a strip

`model` is a driven model with a lumped port `p1`, whose signal strip has the
side walls `signal_wall` and whose ground is `ground`.

```python
import jax
import jax.numpy as jnp

import gfem

morph = gfem.shape.boundary_morph(model, "signal_wall", fixed=("ground",))
problem = gfem.Driven(frequency=(20e9, 40e9, 60e9), excite="p1")


def reflection(dw):
    """Mean |S11|^2 when the strip widens by dw metres (dw / 2 per wall)."""
    moved = model.replace(mesh=morph(dw / 2 * jnp.ones(morph.n_params)))
    s11 = gfem.solve(moved, problem).s["p1", "p1"]
    return jnp.mean(jnp.abs(s11) ** 2)


grad = jax.grad(reflection)(0.0)
```

- `grad` is the change of the mean `|S11|²` per metre of extra width.
- The gradient needs no extra solve: the backward pass (the part of
  `jax.grad` that runs after the solve) contracts the forward fields against
  the derivative of the operator with respect to every node. A kept field or
  potential in the objective adds the same adjoint solve as for a material
  gradient.
- Per excited port and frequency the backward pass costs well under a tenth
  of a forward linear solve. Under an adaptive sweep it costs one such pass
  per reduced basis vector and operator term, whatever the number of
  frequencies.

The tutorial {doc}`../tutorials/shape` runs a whole gap-width optimization
this way, accepting steps on `gfem.shape.quality`.

## Keep the ports where they are

A port's geometry (its surface, the lumped element's length and width, its
excitation and voltage) is held fixed by the gradient. The node gradient is
exact for every displacement that leaves the port surfaces where they were
imported, and `boundary_morph` holds the model's ports for you.

- A moved port node raises {py:obj}`gfem.PortGeometryError` while the gradient
  is computed; hold the port surfaces fixed in your morph (see
  {ref}`gradient-errors`).
- Wave ports follow the same rule: their modes come from the port faces'
  geometry, which stays as imported.

## Check your results

- Compare the gradient with central finite differences of your objective
  along a direction that holds the ports. The shape gradient is the exact
  derivative of the discrete objective, to solver precision.
- Each moved mesh is a different discretization. The gradient therefore
  differs from the derivative of the continuous problem by the sensitivity of
  the discretization error; refine the mesh if the two must agree closely.

How gfem itself is checked: {doc}`../verification`.

## Limits

- Eigenmode quantities of the mode fields (anything but `omega`) have no
  gradient, as for material gradients: they raise
  `gfem.NonDifferentiableError` when `jax.grad` is called. Wrap them in
  `jax.lax.stop_gradient`.
- Measurements on result fields (`Field.at`, `gfem.interface_participation`)
  raise with respect to the node coordinates, because the field's mesh is the
  solve's mesh. To differentiate through the field values only, with the
  measurement's own geometry held fixed, freeze it:
  `dataclasses.replace(field, mesh=jax.lax.stop_gradient(field.mesh))`.
- `result.energy` of an electrostatic problem raises, as for any input the
  solve reads. Use `capacitance[i, i] / 2` instead.

## Next

{py:obj}`gfem.shape.boundary_morph`, {py:obj}`gfem.shape.quality`;
{doc}`s-parameter-gradient`, {doc}`capacitance-gradient`,
{doc}`eigenmode-gradient`; {doc}`../capabilities`.
