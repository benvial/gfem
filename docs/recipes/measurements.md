# Field measurements

Get a resonator's surface-loss Q, field values at points or their mean over
a region, a line's characteristic impedance, the power through a surface or
the power each region dissipates from a solved result. Each is a function you
call on the result, also on one reloaded with {py:obj}`gfem.load_result`.

## Interface (surface) loss of a resonator

`gfem.interface_participation` gives the share of each mode's electric energy
in a thin oxide or residue layer of thickness `t` and permittivity `eps_r` on a
surface, with the metal-air (`"MA"`), metal-substrate (`"MS"`) and
substrate-air (`"SA"`) models of Wenner et al.:

```python
import gfem

modes = gfem.solve(model, gfem.Eigenmode(target=7.8e9, n_modes=2))
ma = gfem.interface_participation(modes, "metal", 3e-9, 10.0, kind="MA", tan_delta=2e-3)
ms = gfem.interface_participation(modes, "metal", 3e-9, 10.0, kind="MS", tan_delta=2e-3)
sa = gfem.interface_participation(modes, "gaps", 3e-9, 10.0, kind="SA", tan_delta=2e-3)
ma.participation  # (n_modes,)
1 / (ma.inverse_q + ms.inverse_q + sa.inverse_q)  # the surface-loss Q
```

- On a metal sheet between substrate and vacuum, `"MA"` reads the vacuum face
  and `"MS"` the substrate face; for a film with the same medium on both faces
  they add one layer each.
- At a zero-thickness film's edge the participation grows with refinement:
  compare meshes with the same edge cell size ({doc}`loss-budget`).

## Probes

`field.at(points)` evaluates any result field at points in metres: kept
driven fields, mode fields, terminal potentials, `field.b()`:

```python
result = gfem.solve(model, gfem.Driven(frequency=28e9, keep_fields=True))
e = result.e_field(28e9, excite="p1").at([[0.0, 0.0, 5e-6], [1e-4, 0.0, 5e-6]])
```

- A point on a face reads one of its elements: offset it into the side you
  want. A point outside the solved region raises `ValueError`.

## Mean of a field over a region

`gfem.region_mean(field, region)` is the field's mean over named volume
regions:

```python
result = gfem.solve(model, gfem.Driven(frequency=28e9, keep_fields=True))
e_core = gfem.region_mean(result.e_field(28e9, excite="p1"), "core")  # (3,), V/m
static = gfem.solve(model, gfem.Electrostatic(terminals=["t1"]))
v_gap = gfem.region_mean(static.potential["t1"], ["gap", "gap_top"])  # (), V
```

- It returns shape `(3,)` for an electric field and `()` for a potential, and
  is differentiable. Prefer it to `Field.at` in an objective you refine for:
  it is the same quantity on every mesh {doc}`mesh-refinement` produces.

## Characteristic impedance

`gfem.characteristic_impedance` gives the complex impedance of a
`gfem.PortMode` result, by the definition the paths you give select:

- `voltage_path` only: power-voltage, $Z_{PV} = |V|^2 / (2 P^*)$.
- `current_path` only: power-current, $Z_{PI} = 2 P / |I|^2$.
- both: voltage-current, $Z_{VI} = V / I$.

A current path encircles a conductor (repeat its first point to close it). The
three agree for a TEM mode.

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

A wave port with a `voltage_path` already holds $Z_{PV}$ as `mode.Z_PV`, and
`result.port_modes["p1"]` takes the place of `mode`.
`characteristic_impedance` raises `gfem.NonDifferentiableError` on a traced
`PortMode`: differentiate `mode.Z_PV` instead.

## Power through a surface

`gfem.power_flow(field, side, model=model)` is the net time-averaged power, in
W, that a driven field carries out of `side`: a volume region, several, or a
predicate on the element centroids (m). The power launched at port 1 that
crosses a plane, leaves through the top, or leaves the design region:

```python
r = gfem.solve(model, gfem.Driven(frequency=f, keep_fields=True))
e = r.e_field(f, excite="p1")
# W crossing x = -2 um in +x, W leaving through z = 1 um, W leaving the design
p_in = gfem.power_flow(e, lambda c: c[:, 0] < -2.0e-6, model=model)
p_up = gfem.power_flow(e, lambda c: c[:, 2] < 1.0e-6, model=model)
p_box = gfem.power_flow(e, "design", model=model)
```

- **Sign and scale.** Positive means power leaves `side`, as a fraction of the
  1 W incident at the excited port.
- **Exact, in lossy media too.** The number is the power `side`'s ports feed
  minus what its elements, sheets and absorbing faces absorb, so two surfaces
  differ by exactly what is absorbed between them; under an incident field
  ({doc}`incident-fields`), minus it is what `side` absorbs.
- **Inputs.** Keep the surface one element off the ports, and pass the model
  the field was solved with.
- **Several ports fed at once.** Kept fields add up: for waves `a1`, `a2` (in
  sqrt(W)) at one frequency,

  ```python
  e1, e2 = r.e_field(f, excite="p1"), r.e_field(f, excite="p2")
  both = dataclasses.replace(e1, coefficients=a1 * e1.coefficients + a2 * e2.coefficients)
  p = gfem.power_flow(both, side, model=model)
  ```

  is the net power for that excitation.
- **Differentiable.** Pass the kept field and the model of the same design:

  ```python
  def absorbed(eps):
      m = model.with_material("design", gfem.MaterialField(eps_r=eps))
      r = gfem.solve(m, gfem.Driven(frequency=f, keep_fields=True))
      return -gfem.power_flow(r.e_field(f, excite="p1"), "design", model=m)


  g = jax.grad(absorbed)(eps)
  ```

  A traced material in the element layer just outside `side` raises
  `gfem.NonDifferentiableError`: keep the surface one element from the design
  ({doc}`shape-gradient` for node coordinates).

:::{note}
**Why not integrate $\mathbf{E} \times \mathbf{H}^*$ over a plane.** The
pointwise Poynting flux of a finite-element field oscillates with the mesh
period, by about ±1.5-2 % at ten cells per wavelength, falling as $h^{1.8}$.
`gfem.power_flow` is exact for the solved field.
:::

## Power dissipated per region

`gfem.dissipated_power(field, model=model)` is the power a driven field
dissipates in each volume region and on each {py:obj}`gfem.Conductor` or
{py:obj}`gfem.Impedance` surface, in W, by name ({term}`dissipated power`):

```python
r = gfem.solve(model, gfem.Driven(frequency=f, keep_fields=True))
lost = gfem.dissipated_power(r.e_field(f, excite="p1"), model=model)
total = sum(lost.values())  # W lost inside the model, for 1 W incident at p1
```

- **Per 1 W incident.** With real references on every port and nothing
  radiating, the values sum to $1 - \sum_j |S_{j1}|^2$.
- It is differentiable through the field and the lossy materials. The loss
  turns into a heat source with {py:obj}`gfem.electromagnetic_heating`.

## Electric energy per region

`gfem.electric_energy(field, model=model)` is the time-averaged electric energy,
in J, that a field stores in each volume region, `1/4 eps0 ∫ E* · Re(eps_r) E dV`:

```python
r = gfem.solve(model, gfem.Driven(frequency=f, keep_fields=True))
e = r.e_field(f, excite="p1")
w = gfem.electric_energy(e, model=model)
# W absorbed by dielectric loss in the substrate, for 1 W incident at p1
p_sub = 2 * (2 * np.pi * f) * tan_delta * w["substrate"]
```

- **Loss per region.** Dielectric loss is `2 * omega * tan_delta * w[region]`,
  and `-gfem.power_flow(e, region, model=model)` where nothing else absorbs.
- **Peak phasor.** Hence the 1/4; for an eigenmode `2 * w[region]` is
  `bulk_participation[region][k]` times 1 J.
- **Differentiable** with respect to the field, permittivities, a law's
  parameters, a mode's frequency and node coordinates. A region with a law is
  weighed with $\operatorname{Re} \mathrm{d}(\omega\varepsilon)/\mathrm{d}\omega$.

## Move a wave port's reference plane

`result.with_offsets({"p1": 50e-6})` refers S to a plane 50 um inside the
domain at wave port `p1`, with no new solve; it equals a solve with
`WavePort(offset=...)`.

## Gradients

Probes and interface participation work inside `jax.jit`, `jax.vmap` and
`jax.grad`, which flows through the field into the solve:

```python
def field_at_probe(eps):
    m = model.with_material("substrate", gfem.MaterialField(eps_r=eps))
    result = gfem.solve(m, gfem.Driven(frequency=28e9, keep_fields=True))
    return jnp.sum(jnp.abs(result.e_field(28e9).at(points)) ** 2)


grad = jax.grad(field_at_probe)(eps)
```

- Node coordinates and probe points are fixed: use `jax.lax.stop_gradient`.
  Mode fields differentiate with respect to materials and lumped values
  ({doc}`eigenmode-gradient`).

## Check your results

- On a lossless device driven at port 1, `gfem.power_flow` through a plane in
  an output lead equals the sum of `|S[out, "p1"]|²` over the ports at that end.
- Between two lossless surfaces, the two values agree.

How gfem itself is checked: {doc}`../verification`.

## Limits

- A point in an excluded metal volume, a surface wholly in a
  {term}`perfectly matched layer`, or a side in a region with a permittivity
  law raises `ValueError`; model such a metal as `gfem.Conductor`
  ({doc}`dispersive-materials`).
- An adaptive sweep's field raises under `jax.grad` for the power
  measurements: solve with `sweep="uniform"`.

## Next

{py:obj}`gfem.interface_participation`, {py:obj}`gfem.power_flow`,
{py:obj}`gfem.dissipated_power`, {py:obj}`gfem.electric_energy`,
{py:obj}`gfem.region_mean`, {py:obj}`gfem.Field`;
{doc}`shared-port-surface`, {doc}`s-parameter-gradient`,
{doc}`electromagnetic-heating`.

