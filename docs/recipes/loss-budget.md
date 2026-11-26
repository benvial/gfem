# Loss budget and T1 limits

Add up the loss channels of a resonator mode or a qubit's linear mode into
one 1/Q, read the T1 limit each channel sets, and differentiate the total. A
{term}`loss budget` sums three kinds of term: the loss already in the model,
thin layers too thin to mesh, and regions you solved without their loss
tangent.

## A budget of one mode

`model` is a resonator on a substrate, solved lossless, with metal on the
surface `"metal"` and bare substrate on `"gaps"`. The layer numbers (3 nm,
relative permittivity 10, loss tangent 2e-3) and the substrate's loss tangent
are yours to supply.

```python
import jax.numpy as jnp

import gfem

modes = gfem.solve(model, gfem.Eigenmode(target=7.8e9, n_modes=2))

layer = {"thickness": 3e-9, "eps_r": 10.0, "tan_delta": 2e-3}
ma = gfem.interface_participation(modes, "metal", **layer, kind="MA")
ms = gfem.interface_participation(modes, "metal", **layer, kind="MS")
sa = gfem.interface_participation(modes, "gaps", **layer, kind="SA")
substrate = modes.bulk_participation["substrate"] * 5e-7  # its tan_delta

inverse_q = 1 / modes.q + ma.inverse_q + ms.inverse_q + sa.inverse_q + substrate
t1 = 1 / (jnp.real(modes.omega) * inverse_q)  # s, per mode
```

- `1 / modes.q` is the loss in the model: materials with a loss, `Conductor`
  walls, resistive sheets and ports, `Absorbing` boundaries, and the
  radiation a {term}`perfectly matched layer` absorbs
  ({doc}`perfectly-matched-layer`). With none it is 0: leave it out of a
  differentiated objective, since `|Im omega|` has a kink at zero loss.
- Each {py:obj}`gfem.interface_participation` adds one unmeshed layer: metal
  to air (`"MA"`), metal to substrate (`"MS"`) or substrate to air (`"SA"`).
  The {doc}`measurements` recipe explains the three kinds.
- A region solved lossless adds its `bulk_participation` times its loss
  tangent. For a lossy anisotropic region use `principal_participation`, one
  share per principal axis.

## Count each loss once

- Loss that is in the model is in `1 / modes.q` already. `1 / q_port[name]`
  and `bulk_participation * tan_delta` of a region solved with its loss are
  then splits of `1 / q`: read them to see where the loss goes, and never add
  them to it.
- Loss that is not in the model is added through its participation. A
  substrate solved lossless takes `bulk_participation * tan_delta`; solved
  with `tan_delta`, it is in `1 / q`. Do one or the other.
- Build objectives from sums of `1 / Q` terms, never from `q` or `q_port`
  themselves: both are infinite for a lossless mode or a port the mode does
  not reach.
- A driven device's loss split by region, in W per 1 W incident, is
  {py:obj}`gfem.dissipated_power` ({doc}`measurements`).

## T1 limits

A channel with contribution `1 / Q_i` limits the energy relaxation time to
`T1_i = Q_i / Re(omega)`. Channels add as rates:
`1 / T1 = Re(omega) * sum_i 1 / Q_i`. For the modelled loss alone this is
`1 / (2 Im omega)`, since the energy decays twice as fast as the field.

- The Purcell limit through a readout line is `q_port` of the qubit's linear
  mode on the line's resistive port, or all of `1 / q` when that port is the
  only loss in the model.
- It is the T1 limit of the linear mode, from the channels you list.
  Quasiparticles, flux noise and two-level-system saturation are not in it: add
  them as further `1 / Q_i` terms.

## Gradients

The whole budget differentiates with respect to the materials, the sheet
fill and the {term}`lumped values <lumped value>`. Every term but `1 / q` is
read from the mode fields, so it needs an isolated mode (see
{doc}`eigenmode-gradient`). With respect to the mesh node coordinates,
everything differentiates except the unmeshed layers:

- `1 / q`, `1 / q_port`, `bulk_participation` and `principal_participation`
  differentiate with respect to the node coordinates too.
- `interface_participation` raises `gfem.NonDifferentiableError` with respect
  to the node coordinates: take finite differences over re-solves on meshes
  moved by {py:obj}`gfem.shape.boundary_morph`, which keep the topology.
- The layer's own thickness, permittivity and loss tangent differentiate.
- A gradient through an infinite `q_port` raises
  `gfem.NonDifferentiableError`; differentiate `1 / q_port` instead.

## Film edges

On a zero-thickness film the field at the edge is singular, and the
participation of a layer that reaches it grows as the mesh is refined there: on
a 10 µm strip by about 6 to 7 % each time the edge cells were halved. It is not
a mesh-converged number:

- quote a film-edge interface participation together with its edge cell
  size;
- compare designs on meshes with the same edge cell size;
- for an absolute value, take that size near the real film thickness, and
  expect a factor, not a percent.

Gradients are exact derivatives of the value on that mesh and inherit its
dependence. A layer away from film edges, such as a cavity wall, converges
normally.

## Check your results

- On a model with its losses in it, `1 / q` should be close to the sum of
  its splits: `bulk_participation * tan_delta` over the lossy regions plus
  `1 / q_port` over the resistive ports, plus the wall and sheet loss. They
  agree to first order in the losses. On a port of several elements that see
  different voltages, `1 / q_port` is a lower bound on the port's share.

How gfem itself is checked: {doc}`../verification`.

## Limits

- No budget function: the result does not record which losses are in the
  model, so gfem cannot tell a split from an addition. Write the sum
  yourself, as above.
- Mode-field gradients need an isolated mode on the direct solver, on the
  CPU, in one process: {doc}`eigenmode-gradient` lists them.

## Next

{doc}`measurements`, {doc}`quantum-metrics`, {doc}`eigenmode-gradient`,
{doc}`shape-gradient`; {py:obj}`gfem.interface_participation`,
{py:obj}`gfem.EigenmodeResult`.
