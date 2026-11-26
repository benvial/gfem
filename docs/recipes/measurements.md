# Field measurements

Get a resonator's surface-loss Q, field values at points or their mean over
a region, a line's characteristic impedance, the power through a surface or
the power each region dissipates from a solved result. A result already carries every quantity that needs no extra input.
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
never taken for vacuum. The thin layer itself is isotropic. A selected side
in a region with a permittivity law raises `ValueError`, since the layer's
field reads one permittivity there: measure a surface away from that region.
Faces with a side in a {term}`perfectly matched layer` do not count, and a
surface wholly in the layer raises `ValueError`.

On a film with the same medium on both faces, such as an airbridge or a
suspended film, `"MA"` and `"MS"` count one layer on each face and add the
two. On a surface that is not a film but has the same medium on both sides,
they would count the layer twice: use `"default"` or `"SA"` there. Those two take the mean of the
fields on the two sides, so they are not for a film.

At the edge of a zero-thickness film the field is singular, and the
participation of a layer that reaches the edge grows as the mesh is refined
there. Compare designs on meshes with the same edge cell size; the
{doc}`loss-budget` recipe gives the rule for an absolute value and adds the
layers to the mode's other losses.

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
- A point inside a {term}`perfectly matched layer` reads the stretched
  field, which is not physical; so do means and exports there
  ({doc}`perfectly-matched-layer`).

## Mean of a field over a region

`gfem.region_mean(field, region)` is the field's mean over named volume
regions, the integral of the solved field divided by the regions' volume:

```python
result = gfem.solve(model, gfem.Driven(frequency=28e9, keep_fields=True))
e_core = gfem.region_mean(result.e_field(28e9, excite="p1"), "core")  # (3,), V/m
static = gfem.solve(model, gfem.Electrostatic(terminals=["t1"]))
v_gap = gfem.region_mean(static.potential["t1"], ["gap", "gap_top"])  # (), V
```

- It returns shape `(3,)` for an electric field and `()` for a potential,
  complex for a phasor. A sequence of names takes the mean over their
  union.
- It is differentiable: `jax.grad` of an objective that reads it flows
  back into the solve.
- Regions are named, never selected by coordinates, so the mean is the
  same quantity on every mesh {doc}`mesh-refinement` produces.
- Prefer it to `Field.at` in an objective you refine for: a mean over a
  small region has an error estimate, a point value has none.

## Characteristic impedance

`gfem.characteristic_impedance` gives the complex impedance of a
`gfem.PortMode` result, by the definition the paths you give select:

- `voltage_path` only: power-voltage, $Z_{PV} = |V|^2 / (2 P^*)$.
- `current_path` only: power-current, $Z_{PI} = 2 P / |I|^2$.
- both: voltage-current, $Z_{VI} = V / I$.

Here $V = \int E \cdot dl$, $I = \int H \cdot dl$ and $P$ is the mode's complex
power. A current path encircles a conductor (repeat its first point to close
it), or ends on magnetic walls of the cross-section. The three agree for a TEM
mode; for a lossy line all three are complex, with $\mathrm{Im}\, Z < 0$ for
conductor loss under $e^{+i\omega t}$.

```python
mode = gfem.solve(model, gfem.PortMode("p1", [10e9, 20e9]))
v_path = [(0.0, 0.0, 0.0), (0.0, 10e-6, 0.0)]
loop = [
    (-5e-6, 2e-6, 0.0),
    (5e-6, 2e-6, 0.0),
    (5e-6, 8e-6, 0.0),
    (-5e-6, 8e-6, 0.0),
    (-5e-6, 2e-6, 0.0),
]
z_pv = gfem.characteristic_impedance(mode, v_path)
z_pi = gfem.characteristic_impedance(mode, current_path=loop)
z_vi = gfem.characteristic_impedance(mode, v_path, current_path=loop)
```

A wave port with a `voltage_path` already holds $Z_{PV}$ as `mode.Z_PV`.

Where a port's cross-section cuts a {term}`perfectly matched layer`, $P$
counts the {term}`physical regions <physical region>` of the section only,
so $Z_{PV}$ is low by the mode's share inside the layer (S is unchanged),
and a path through the layer raises `ValueError`.

A driven result holds the mode of each wave port too, so
`result.port_modes["p1"]` takes the place of `mode`, with or without kept
fields. Inside `jax.jit`, `jax.grad` or `jax.vmap` a driven result's port
modes carry no field and the call raises `ValueError`.
`characteristic_impedance` is not differentiable: on a traced `PortMode`
result it raises `gfem.NonDifferentiableError`. Differentiate `mode.Z_PV`,
along the port's own `voltage_path`, which follows a design at the port face;
or call `characteristic_impedance` outside the transformation.

## Power through a surface

`gfem.power_flow(field, side, model=model)` is the net time-averaged power, in
W, that a driven field carries out of `side`: a volume region, several, or a
predicate on the element centroids (m). Its thermal twin,
{py:obj}`gfem.heat_flow`, reads the heat leaving a side of a steady
temperature in the same way ({doc}`heater-temperature`). How much of the power launched at
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
- **With a perfectly matched layer**, `side` set to the layer regions gives
  the power they absorb; a surface that cuts through the layer has no
  physical meaning ({doc}`perfectly-matched-layer`).
- **Under an incident field.** On the total field of an incident-field
  column (`r.e_field(f, "wave")`, {doc}`incident-fields`) the incident load
  is not a port, and its share of the residual is subtracted, so the number
  is again exact: minus the power leaving a side of physical elements is
  what that side absorbs, one region at a time. A side holding perfectly
  matched layer elements raises (the field there is the scattered one), and
  so does `part="incident"`, which balances no load; the scattered part
  (`part="scattered"`) is the power the {term}`scattered field` carries out.
- **Keep the surface off the ports.** A port's term needs its excitation,
  which a field does not record: a surface within one element of a port's
  surface raises `ValueError`. Move it one element away, or put the port's
  elements inside `side`.
- **The model the field was solved with** is a required input: the residual
  reads its materials and boundaries. Pass the same model: gfem does not
  detect a design solved with other `MaterialField` values.
- **Across ranks.** On a field of a {term}`distributed solve` it is a
  collective call: call it on every rank with the same side and model, and
  every rank gets the same number ({doc}`distributed-solve`).
- **Several ports fed at once.** `power_flow` is quadratic in the field, and
  the kept fields add up: for incident waves `a1` and `a2` (in sqrt(W)) at two
  ports, at one frequency,

  ```python
  e1, e2 = r.e_field(f, excite="p1"), r.e_field(f, excite="p2")
  both = dataclasses.replace(e1, coefficients=a1 * e1.coefficients + a2 * e2.coefficients)
  p = gfem.power_flow(both, side, model=model)
  ```

  is the net power for that excitation. Through a plane in the lead of port 1
  it is `abs(a1)**2 - abs(b1)**2`, with `b = S @ a` the outgoing waves.
- **Differentiable through the field.** Absorbed, radiated or transmitted
  power is an objective: inside `jax.grad`, `jax.jit` or `jax.vmap`, pass the
  kept field and the model of the same design (lumped or wave ports alike):

  ```python
  def absorbed(eps):
      m = model.with_material("design", gfem.MaterialField(eps_r=eps))
      r = gfem.solve(m, gfem.Driven(frequency=f, keep_fields=True))
      return -gfem.power_flow(r.e_field(f, excite="p1"), "design", model=m)


  g = jax.grad(absorbed)(eps)
  ```

  The power reads the materials of the element layer just outside `side`, and
  their gradient there is not available: a material field with an element in
  that layer, traced, raises `gfem.NonDifferentiableError`. Keep the surface
  one element away from the design region (inside `side` is fine), or wrap
  the value in `jax.lax.stop_gradient`. The power is differentiable with
  respect to the node coordinates too (a shape design,
  {doc}`shape-gradient`): the element layer moves with its nodes. When they
  are traced (a `jax.jit` over the model), name the side's regions instead
  of a predicate. A field of an adaptive sweep raises: its
  frozen basis gives no exact gradient of the power; use `sweep="uniform"`.

:::{note}
**Why not integrate $\mathbf{E} \times \mathbf{H}^*$ over a plane.** On a finite-element field the
pointwise Poynting flux is not conserved from plane to plane: it oscillates
with the mesh period along the propagation direction, highest on element faces
and lowest at mid-cell. On a 220 nm SOI strip guide it is off by about ±1.5-2 %
at roughly ten cells per guided wavelength and falls about as $h^{1.8}$; exact
face quadrature does not remove it. `gfem.power_flow` is exact for the solved
field.
:::

## Power dissipated per region

`gfem.dissipated_power(field, model=model)` is the time-averaged power a
driven field dissipates in each volume region and on each
{py:obj}`gfem.Conductor` or {py:obj}`gfem.Impedance` surface, in W, by name
({term}`dissipated power`): the loss tangent, the conductivity, a complex
permittivity or a permittivity law in a region, and the sheet loss of a
metal surface.

```python
r = gfem.solve(model, gfem.Driven(frequency=f, keep_fields=True))
lost = gfem.dissipated_power(r.e_field(f, excite="p1"), model=model)
total = sum(lost.values())  # W lost inside the model, for 1 W incident at p1
```

- **Per 1 W incident.** A driven field carries 1 W incident at its excited
  port, so each value is a fraction of the incident power. A region's value
  is minus `power_flow` out of it when no sheet lies on its surface.
- **What S does not account for.** With real reference impedances on every
  port, each on a uniform stretch of line, and nothing radiating, the values
  sum to $1 - \sum_j |S_{j1}|^2$. A wave port on a lossy line has a complex
  reference: its field is per unit $|a|^2$, which is not exactly 1 W
  incident, and the sum does not match S; compare with `power_flow` into
  the lossy regions instead.
- **Not heat, so not listed.** A {term}`perfectly matched layer`, an
  absorbing face, a port and a lumped port's resistance get no key.
- **Differentiable** through the field and directly through the lossy
  materials, sheets' fills and an `Impedance`'s resistance; an adaptive
  sweep's field raises under `jax.grad`.
- The loss of a drive turns into a heat source with
  {py:obj}`gfem.electromagnetic_heating` ({doc}`electromagnetic-heating`).

## Electric energy per region

`gfem.electric_energy(field, model=model)` is the time-averaged electric energy,
in J, that a field stores in each volume region of the solve domain,
`1/4 eps0 ∫ E* · Re(eps_r) E dV`, by region name:

```python
r = gfem.solve(model, gfem.Driven(frequency=f, keep_fields=True))
e = r.e_field(f, excite="p1")
w = gfem.electric_energy(e, model=model)
# W absorbed by dielectric loss in the substrate, for 1 W incident at p1
p_sub = 2 * (2 * np.pi * f) * tan_delta * w["substrate"]
```

- **Loss per region.** In a region with one loss tangent, dielectric loss
  absorbs `2 * omega * tan_delta * w[region]`. For a region with nothing else
  that absorbs, it is the number `-gfem.power_flow(e, region, model=model)`
  gives, and `gfem.dissipated_power` gives it for every region at once.
- **Time-averaged, on a peak phasor.** Hence the 1/4. A mode of an eigenmode
  result is normalized to its total energy, twice the time average of the
  electric part: `2 * w[region]` of `modes.e_field(k)` is
  `bulk_participation[region][k]` times 1 J.
- **The model the field was solved with** is a required input, as for
  `power_flow`: a driven result does not carry its materials.
- **Physical regions only.** The regions of a {term}`perfectly matched layer`
  get no entry: the field there is not physical.
- **Differentiable.** `electric_energy` inside `jax.grad`, `jax.jit` or
  `jax.vmap` has a gradient with respect to the field (a driven field or a
  mode's), the permittivity of a region without a law, a law's parameters,
  a mode's frequency and the node coordinates. The loss tangent, the
  conductivity and the permeability enter through the field only. The field
  of an adaptive sweep raises `gfem.NonDifferentiableError`: solve with
  `sweep="uniform"` at the frequency you differentiate.
- **A region with a permittivity law** is weighed with $\operatorname{Re} \mathrm{d}(\omega\varepsilon)/\mathrm{d}\omega$ at the
  field's frequency, the energy of a dispersive medium. Where that weight is
  not positive (loss dominates the law) it raises `ValueError`; model such a
  metal as `gfem.Conductor` or `gfem.PEC` ({doc}`dispersive-materials`).

## Move a wave port's reference plane

`result.with_offsets({"p1": 50e-6})` returns the result with S referred to a
plane 50 um inside the domain at wave port `p1`, from the propagation constants
the result already holds: no new solve. Distances are absolute, as in
`WavePort(offset=...)`, so the shifted result equals a solve with that offset,
and `result.offsets` records the planes S is referred to.

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
- Mode fields differentiate with respect to the materials, the sheet fill
  and the lumped values, for an isolated mode: probes and interface
  participation of an eigenmode result are objectives too. With respect to
  the node coordinates they raise `gfem.NonDifferentiableError`; take finite
  differences over moved meshes instead. The conditions are in
  {doc}`eigenmode-gradient`.
- The layer's thickness, permittivity and loss tangent differentiate.

## Check your results

- On a lossless device driven at port 1, `gfem.power_flow` through a plane
  in an output lead should equal the sum of `|S[out, "p1"]|²` over the
  output ports at that end.
- Between two surfaces in a lossless region with no port or sheet, the two
  `power_flow` values should agree.

How gfem itself is checked: {doc}`../verification`.

## Next

{py:obj}`gfem.interface_participation`, {py:obj}`gfem.characteristic_impedance`,
{py:obj}`gfem.power_flow`, {py:obj}`gfem.dissipated_power`, {py:obj}`gfem.electric_energy`, {py:obj}`gfem.region_mean`,
{py:obj}`gfem.Field`, {doc}`shared-port-surface`, {doc}`mesh-refinement`,
{doc}`s-parameter-gradient`, {doc}`eigenmode-gradient`, {doc}`loss-budget`,
{doc}`electromagnetic-heating`.

