# Shape gradients

`gfem.solve` differentiates its results with respect to the mesh node
coordinates, `model.mesh.nodes` (metres, shape `(n_nodes, 3)`), for
electrostatic, driven (uniform sweeps, and adaptive sweeps with frozen
samples) and eigenmode problems (the eigenfrequencies, as for material
gradients). A shape design is a JAX function from a few parameters to
moved node coordinates, usually through `gfem.shape.boundary_morph`:

```python
import gfem
import jax
import jax.numpy as jnp

morph = gfem.shape.boundary_morph(model, "signal_wall", fixed=("ground",))
problem = gfem.Driven(frequency=(20e9, 40e9, 60e9), excite="p1")


def reflection(dw):
    """Mean |S11|^2 when the strip widens by dw metres (dw / 2 per wall)."""
    moved = model.replace(mesh=morph(dw / 2 * jnp.ones(morph.n_params)))
    s11 = gfem.solve(moved, problem).s["p1", "p1"]
    return jnp.mean(jnp.abs(s11) ** 2)


grad = jax.grad(reflection)(0.0)
```

The tutorial `docs/tutorials/shape.ipynb` runs a whole gap-width optimization
this way, accepting steps on `gfem.shape.quality`.

## What it costs

No extra solve. The backward pass contracts the forward fields (and, for
field or potential cotangents, the same adjoint solve as a material
gradient) against the derivative of the operator with respect to every node:

- volume terms (the curl-curl, mass and conductivity terms, the
  electrostatic `eps grad . grad`) through closed-form transport tensors on
  the fields, element by element, integrated exactly;
- surface terms (absorbing boundaries, `Conductor`, `Impedance`) through
  the same kind of closed form on their faces (checked against NGSolve's
  own shape derivative).

Per excited port and frequency this costs well under a tenth of a forward
linear solve; under an adaptive sweep it costs one such pass per reduced
basis vector and operator term, whatever the number of frequencies.

## Ports stay put

A port's geometry (its surface, the lumped element's length and width, its
excitation and voltage) is held fixed by the gradient: the node gradient
is exact for every displacement that leaves the port surfaces where they
were imported. `boundary_morph` holds the model's ports for you. A moved
port node raises `gfem.PortGeometryError` in the backward pass (reaching you
as `jax.errors.JaxRuntimeError` with the message kept). The same holds for
wave ports: their modes come from the port faces' geometry, which stays as
imported.

## What raises

- Eigenmode quantities of the mode fields (anything but `omega`, as for
  material gradients: `gfem.NonDifferentiableError` while JAX traces).
- Measurements on result fields (`Field.at`, `gfem.interface_participation`)
  with respect to the node coordinates: the field's mesh is the solve's
  mesh. To differentiate through the field values only (the measurement's
  own geometry held fixed), freeze it:
  `dataclasses.replace(field, mesh=jax.lax.stop_gradient(field.mesh))`.
- `result.energy` of an electrostatic problem, as for any read input.

## Accuracy

The shape gradient is the exact derivative of the discrete objective, to
solver precision (the moved mesh is a different discretization, so it
differs from the derivative of the continuous problem by the
discretization error's own sensitivity). Check it with central finite
differences along a direction that holds the ports.
