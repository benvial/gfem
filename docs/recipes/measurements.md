# Measure fields after the solve

Get a resonator's surface-loss Q, field values at points, a line's
characteristic impedance or the power through a surface from a solved
result. A result already carries every quantity that needs no extra input.
Anything that needs more input (a probe point, a thin interface layer, a
voltage path) is a measurement: a function you call on the result, also on a
result reloaded with {py:obj}`gfem.load_result`.

## Interface (surface) loss of a resonator

The thin oxides and residues on a superconducting resonator are too thin to
mesh. `gfem.interface_participation` gives the share of each mode's electric
energy in a layer of thickness `t` and permittivity `eps_r` on a surface,
with the metal-air (`"MA"`), metal-substrate (`"MS"`) and substrate-air
(`"SA"`) models of Wenner et al.:

```python
import gfem

modes = gfem.solve(model, gfem.Eigenmode(target=7.8e9, n_modes=2))
ma = gfem.interface_participation(modes, "metal", 3e-9, 10.0, kind="MA", tan_delta=2e-3)
ms = gfem.interface_participation(modes, "metal", 3e-9, 10.0, kind="MS", tan_delta=2e-3)
sa = gfem.interface_participation(modes, "gaps", 3e-9, 10.0, kind="SA", tan_delta=2e-3)
ma.participation  # (n_modes,)
1 / (ma.inverse_q + ms.inverse_q + sa.inverse_q)  # the surface-loss Q
```

On a metal sheet between substrate and vacuum, `"MA"` reads the vacuum face
and `"MS"` the substrate face (its normal flux, with the substrate's
permittivity tensor), so one surface serves both. A side counts as vacuum
only if no wave in it is slower than in vacuum, so an anisotropic substrate is
never taken for vacuum. The thin layer itself is isotropic.

## Probes

`field.at(points)` evaluates any result field at points in metres: kept
driven fields, mode fields, terminal potentials, `field.b()`:

```python
result = gfem.solve(model, gfem.Driven(frequency=28e9, keep_fields=True))
e = result.e_field(28e9, excite="p1").at([[0.0, 0.0, 5e-6], [1e-4, 0.0, 5e-6]])
```

- A point inside an element reads that element's value.
- A point on a face reads the value from one of the elements that share it.
  A component that jumps there, such as the normal component of E at a
  material interface, has no single value: to read one side, offset the
  point into it. A point on an excluded conductor's surface reads the side
  the field covers.
- A point inside a volume the field does not cover (an excluded metal volume)
  raises `ValueError`: move it into the solved region.
- A point outside the mesh by less than 1e-4 of an element's size reads the
  nearest element. Farther out raises `ValueError`: move the point inside.

## Characteristic impedance

`gfem.characteristic_impedance(mode, voltage_path)` is the power-voltage
impedance `|V|^2 / (2 P)` of a `gfem.PortMode` result along any path on the
port, when the wave port declares none:

```python
mode = gfem.solve(model, gfem.PortMode("p1", [10e9, 20e9]))
z = gfem.characteristic_impedance(mode, [(0.0, 0.0, 0.0), (0.0, 10e-6, 0.0)])
```

## Power through a surface

`gfem.power_flow(field, side, model=model)` is the net time-averaged power, in
W, that a driven field carries out of `side`: a volume region, several, or a
predicate on the element centroids (m). How much of the power launched at
port 1 crosses a plane, leaves through the top of the stack, or leaves the
design region:

```python
r = gfem.solve(model, gfem.Driven(frequency=f, keep_fields=True))
e = r.e_field(f, excite="p1")
# W crossing x = -2 um in +x, W leaving through z = 1 um, W leaving the design
p_in = gfem.power_flow(e, lambda c: c[:, 0] < -2.0e-6, model=model)
p_up = gfem.power_flow(e, lambda c: c[:, 2] < 1.0e-6, model=model)
p_box = gfem.power_flow(e, "design", model=model)
```

- **Sign and scale.** Positive means power leaves `side`, so `c[:, 0] < x0`
  gives the flow in +x. A driven field carries 1 W incident at its excited
  port, so the number is a fraction of the incident power.
- **The surface is element faces.** It is the set of faces between the
  selected elements and the rest of the solve domain: a grid plane exactly,
  or the staircase of faces nearest a plane that cuts through elements. Faces
  on the domain boundary are not part of it: what leaves through them
  (absorbing walls, ports) is counted inside `side`.
- **Exact, in lossy media too.** The number comes from the residual of the
  discrete equations on the element layer next to `side`, so it is
  exact for the solved field. It equals the power `side`'s ports feed minus
  what its elements, sheets and absorbing faces absorb, so the difference
  between two surfaces is exactly the power absorbed or radiated between
  them. A `Conductor` or `Impedance` sheet lying on the surface counts as
  outside.
- **Keep the surface off the ports.** A port's term needs its excitation,
  which a field does not record: a surface within one element of a port's
  surface raises `ValueError`. Move it one element away, or put the port's
  elements inside `side`.
- **The model the field was solved with** is a required input: the residual
  reads its materials and boundaries. Pass the same model: gfem does not
  detect a design solved with other `MaterialField` values.
- **Not differentiable.** `power_flow` inside `jax.grad` raises
  `gfem.NonDifferentiableError`. Build a gradient objective from S or from
  probes instead.

:::{note}
**Why not integrate E × H\* over a plane.** On a finite-element field the
pointwise Poynting flux is not conserved from plane to plane: it oscillates
with the mesh period along the propagation direction, highest on element faces
and lowest at mid-cell. On a 220 nm SOI strip guide it is off by about ±1.5-2 %
at roughly ten cells per guided wavelength and falls about as h^1.8; exact
face quadrature does not remove it. `gfem.power_flow` is exact for the solved
field.
:::

## Gradients

Probes and interface participation work inside `jax.jit` and `jax.vmap`,
and `jax.grad` flows through the field into the solve. A probe objective on
a kept driven field therefore has material-field gradients:

```python
def field_at_probe(eps):
    m = model.with_material("substrate", gfem.MaterialField(eps_r=eps))
    result = gfem.solve(m, gfem.Driven(frequency=28e9, keep_fields=True))
    return jnp.sum(jnp.abs(result.e_field(28e9).at(points)) ** 2)


grad = jax.grad(field_at_probe)(eps)
```

- Mesh-node coordinates and probe points are fixed: differentiating with
  respect to them raises `gfem.NonDifferentiableError`. Wrap them in
  `jax.lax.stop_gradient` to hold them fixed.
- Mode fields are not differentiable in this version, so a gradient of an
  interface participation that reaches the eigenmode solve raises there.
  The layer's thickness, permittivity and loss tangent differentiate: take
  the gradient with respect to those, or build the objective from the
  eigenmode's `frequency` and `q`.

## Check your results

- On a lossless device driven at port 1, `gfem.power_flow` through a plane
  in an output lead should equal the sum of `|S[out, "p1"]|²` over the
  output ports at that end.
- Between two surfaces in a lossless region with no port or sheet, the two
  `power_flow` values should agree.

How gfem itself is checked: {doc}`../verification`.

## Next

{py:obj}`gfem.interface_participation`, {py:obj}`gfem.characteristic_impedance`,
{py:obj}`gfem.power_flow`, {py:obj}`gfem.Field`, {doc}`shared-port-surface`,
{doc}`s-parameter-gradient`, {doc}`eigenmode-gradient`.

