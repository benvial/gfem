# Shape gradients

Optimize the shape of a device by differentiating S, a capacitance or an
eigenfrequency with respect to a few geometric parameters that move the mesh.
`gfem.solve` differentiates its results with respect to the mesh node
coordinates, `model.mesh.nodes` (metres, shape `(n_nodes, 3)`), for
electrostatic, driven (uniform sweeps, and adaptive sweeps with frozen
samples) and eigenmode problems (eigenfrequencies and participation
ratios). A shape design is a JAX function from a few parameters to moved
node coordinates, usually through {py:obj}`gfem.shape.boundary_morph`. A wave
port's face moves too when the morph names the port in `sliding_ports`.

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

A scalar `morph(t)` moves every moving node by `t`: the surface's edge bias, the
parameter of {doc}`tolerance-report`.

The tutorial {doc}`../tutorials/shape` runs a whole gap-width optimization
this way, accepting steps on `gfem.shape.quality`.

## Keep the ports where they are

A lumped or Floquet port's geometry (its surface, the lumped element's length
and width, its excitation and voltage) is held fixed by the gradient, and so
is a wave port's face unless you let it slide. `boundary_morph` holds the
model's ports for you.

- A moved node of a lumped or Floquet port raises
  {py:obj}`gfem.PortGeometryError` while the gradient is computed; hold those
  surfaces fixed in your morph (see {ref}`gradient-errors`).
- Wave ports are held too unless named in `sliding_ports=`, where their faces
  slide in their plane and the mode follows: a taper's width at its port is a
  design ({doc}`port-mode-gradient`). Keep the conductors a port's
  `voltage_path` joins where its ends are.
- A junction port between two moving pads keeps its imported voltage path,
  width and capacitance. The pad edges move beside it, and the leads that
  join it stretch: the gradient is that of a gap that changes beside a fixed
  junction.
- Give every face that must stay flat a name in the mesh, such as a chip
  surface, even when it carries no boundary condition. `boundary_morph`
  slides the nodes of every named surface in its plane; an unnamed face
  moves with the volume around it and warps.

## Check your results

- Compare the gradient with central finite differences of your objective
  along a direction that holds the ports. The shape gradient is the exact
  derivative of the discrete objective, to solver precision.
- Each moved mesh is a different discretization. The gradient therefore
  differs from the derivative of the continuous problem by the sensitivity of
  the discretization error; refine the mesh if the two must agree closely.

How gfem itself is checked: {doc}`../verification`.

## Limits

- Participation ratios and port Q of an eigenmode differentiate with
  respect to the node coordinates, for an isolated mode (see
  {doc}`eigenmode-gradient`). The mode field itself (`e_field`, `field.at`,
  `gfem.interface_participation`) does not: it raises
  `gfem.NonDifferentiableError`; take finite differences over moved meshes.
- Measurements on result fields (`Field.at`, `gfem.interface_participation`)
  raise with respect to the node coordinates, because the field's mesh is the
  solve's mesh. To differentiate through the field values only, with the
  measurement's own geometry held fixed, freeze it:
  `dataclasses.replace(field, mesh=jax.lax.stop_gradient(field.mesh))`.
- `result.energy` of an electrostatic problem differentiates with respect to
  the node coordinates, like the capacitance.
- {py:obj}`gfem.power_flow`, {py:obj}`gfem.far_field` and
  {py:obj}`gfem.beam_coupling` differentiate with respect to the node
  coordinates: the element layer they read moves with its nodes.
  `boundary_morph` holds the elements touching the truncation (the
  perfectly matched layer or an `Absorbing` face), where the far field and
  the beam coupling read by default. A `beam_coupling` of an
  `gfem.IncidentField` raises once a node of its layer has moved.

## Next

{py:obj}`gfem.shape.boundary_morph`, {py:obj}`gfem.shape.quality`;
{py:meth}`gfem.Mesh.with_nodes` for a node map of your own, on any mesh
(the {doc}`transducer recipe <piezoelectric-gradient>` differentiates a transducer through one);
{doc}`s-parameter-gradient`, {doc}`capacitance-gradient`,
{doc}`eigenmode-gradient`, {doc}`quantum-metrics`; {doc}`../tutorials/transmon`;
{doc}`../capabilities`.
