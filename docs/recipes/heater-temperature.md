# Heater temperature

Find the temperature of a device heated by a current: the resistance of the
heater, the power it dissipates, and how much each part of the device warms
up. {py:obj}`gfem.DCConduction` solves the current, {py:obj}`gfem.joule_heating` turns it into heat and {py:obj}`gfem.SteadyThermal` solves the heat.

The device is the cross-section of the {doc}`heater tutorial <../tutorials/heater>`. Extensive quantities are {term}`per length <per-length quantity>`; temperatures are absolute ({doc}`../conventions`, "Temperature and thermo-optic phase").

## Materials and boundaries

The mesh is a {term}`slab cross-section` with {term}`end faces <End faces>` `"front"` and `"back"`. Only the heater has a `conductivity`, so only it carries current.

```python
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/heater_w2.msh"
urllib.request.urlretrieve(url, "heater_w2.msh")
mesh = gfem.Mesh.from_gmsh_cross_section("heater_w2.msh", length_unit="um")

oxide = gfem.Material(eps_r=1.444**2, thermal_conductivity=1.38)  # W/m/K
silicon = gfem.Material(eps_r=3.4777**2, thermal_conductivity=90.0)
substrate = gfem.Material(thermal_conductivity=148.0)
tin = gfem.Material(thermal_conductivity=28.0, conductivity=2.3e6)  # S/m
materials = {
    "substrate": substrate,
    "box": oxide,
    "cladding": oxide,
    "window": oxide,
    "neighbour_window": oxide,
    "core": silicon,
    "neighbour": silicon,
    "heater": tin,
}
boundaries = [
    gfem.FixedTemperature("bottom", temperature=300.0),  # K, the heat sink
    gfem.Convection("top", coefficient=10.0, ambient=300.0),  # W/m^2/K, K
]
model = gfem.Model(
    mesh,
    materials=materials,
    boundaries=boundaries,
    excluded={"em": ["substrate", "heater"]},
)
print(mesh.volumes)
print(model.default_insulated)
print(model.excluded_from("thermal"))
```


```text
('core', 'neighbour', 'heater', 'window', 'neighbour_window', 'box', 'cladding', 'substrate')
('front', 'back')
()
```


- A {py:obj}`gfem.FixedTemperature` or {py:obj}`gfem.Convection` surface is a {term}`heat sink <Heat sink>`. Every connected piece of the {term}`thermal domain` needs one in a steady solve; a piece without one raises, naming a region of it.
- A surface no thermal boundary names is an {term}`insulated surface`: `model.default_insulated` lists the named ones, here the end faces.
- `excluded=` maps a physics (`"em"`, `"thermal"`, `"dc_conduction"`) to the regions left out of it. Here the substrate and heater leave electromagnetic solves only ({doc}`thermo-optic-phase`).
- A thermal solve needs `thermal_conductivity` in every region, and a transient one also `mass_density` and `specific_heat` ({doc}`thermal-step-response`). A missing property raises {py:obj}`gfem.errors.MissingMaterialPropertyError`, listing every region and property.

## The heater's resistance

{py:obj}`gfem.DCConduction` holds each terminal surface at 1 V against the grounded ones. On a cross-section the terminals are the end faces and the current flows along the guide:

```python
dc = gfem.solve(model, gfem.DCConduction({"heater": "front"}, ground="back"))
r_per_length = float(dc.resistance_per_length[0, 0])  # ohm/m
print(f"R' = {r_per_length:.6e} ohm/m")
print(f"1 / (sigma w t) = {1 / (2.3e6 * 2e-6 * 0.14e-6):.6e} ohm/m")
length = 320e-6  # m, the heater's length along the guide
print(f"R over 320 um: {r_per_length * length:.2f} ohm")
```


```text
R' = 1.552795e+06 ohm/m
1 / (sigma w t) = 1.552795e+06 ohm/m
R over 320 um: 496.89 ohm
```


- `dc.conductance` is the {term}`conductance matrix` in siemens, in the order of `dc.terminals`; on a cross-section read `resistance_per_length`.
- A uniform strip has $1/(\sigma A)$ per metre: the solve agrees to rounding.
- One DC result serves every drive, because the conductance is linear.

## The temperature at a drive

{py:obj}`gfem.joule_heating` turns the DC result and a drive into a {py:obj}`gfem.HeatSource` on the conductors. On a cross-section the drive is the current through the heater, in A, or the power per metre:

```python
heat = gfem.joule_heating(dc, current=6.74e-3)  # A
steady = gfem.solve(model, gfem.SteadyThermal(heat))
print(heat)
print(f"power: {float(steady.power_per_length):.4f} W/m")
print(f"I^2 R': {6.74e-3**2 * r_per_length:.4f} W/m")
core = float(gfem.region_mean(steady.rise, "core"))  # K
heater = float(gfem.region_mean(steady.rise, "heater"))
print(f"mean rise: core {core:.3f} K, heater {heater:.3f} K")
print(f"core temperature: {float(gfem.region_mean(steady.temperature, 'core')):.3f} K")
```


```text
HeatSource(joule, regions=['heater'], total 70.5398 W/m)
power: 70.5398 W/m
I^2 R': 70.5398 W/m
mean rise: core 12.074 K, heater 31.508 K
core temperature: 312.074 K
```


- `steady.temperature` is absolute, in K. `steady.rise` is the {term}`temperature rise` above the {term}`source-free temperature`, the temperature with every source off.
- Both are fields: read them with `.at(points)`, {py:obj}`gfem.region_mean`, {py:obj}`gfem.element_mean` or `to_xarray`.
- On a 3D mesh, drive by `voltage=` (V), `current=` (A) or `power=` (W). On a cross-section `voltage=` and `power=` raise, because they fix nothing without the heater's length; the message gives the per-length form.
- With several terminals, give `voltage=` or `current=` as a mapping from terminal to value.

## Where the heat goes

In steady state every watt leaves through a {term}`heat sink`. {py:obj}`gfem.heat_flow_per_length` gives the heat leaving through named surfaces, or out of a set of elements, in W/m, positive out ({term}`heat flow`):

```python
sink = float(gfem.heat_flow_per_length(steady, "bottom", model=model))  # W/m
up = float(gfem.heat_flow_per_length(steady, "top", model=model))
above = gfem.heat_flow_per_length(steady, lambda c: c[:, 1] > -12e-6, model=model)
down = float(above)  # W/m out of the elements above y = -12 um
power = float(steady.power_per_length)
print(f"into the sink: {sink:.4f} W/m, up into the air: {up:.2e} W/m")
print(f"balance: |sink + up - power| / power = {abs(sink + up - power) / power:.0e}")
print(f"down across y = -12 um: {down:.4f} W/m")
```


```text
into the sink: 70.5375 W/m, up into the air: 2.23e-03 W/m
balance: |sink + up - power| / power = 1e-11
down across y = -12 um: 70.5375 W/m
```


- A surface name gives the heat leaving through it: a {py:obj}`gfem.FixedTemperature` surface the heat it removes, a {py:obj}`gfem.Convection` surface $\int h (T - T_\infty)\, dS$, an insulated one 0.
- A region, several, or a predicate on the element centroids (m) gives the heat leaving those elements into the rest of the section. The cut here is the element faces nearest $y = -12$ µm, so all the sink's heat crosses it.
- The heat comes from the discrete heat balance, not from the gradient, so the flows close to round-off. On a 3D mesh, {py:obj}`gfem.heat_flow` gives the same in W. A transient run records it as {py:meth}`gfem.Trace.heat_out` ({doc}`thermal-step-response`).

## Heat without a current

When you know the power a region dissipates, skip the DC solve: {py:meth}`gfem.HeatSource.uniform` spreads it evenly over one or more regions. A straight strip carries a uniform current, so the two sources give the same temperature:

```python
uniform = gfem.HeatSource.uniform("heater", power_per_length=steady.power_per_length)
same = gfem.solve(model, gfem.SteadyThermal(uniform))
print(uniform)
print(f"core rise: {float(gfem.region_mean(same.rise, 'core')):.3f} K")
```


```text
HeatSource(uniform, regions=['heater'], power_per_length=70.5398 W/m)
core rise: 12.074 K
```


- {py:obj}`gfem.electromagnetic_heating` gives a driven solve's loss as heat ({doc}`electromagnetic-heating`).
- `HeatSource.uniform` takes exactly one of `power=` (W, 3D), `power_per_length=` (W/m, cross-section) or `power_density=` (W/m³).

## A boundary that sets the temperature, not the rise

The rise follows the heat sources only. A boundary temperature, an ambient or a {py:obj}`gfem.HeatFlux` moves the temperature and leaves the rise unchanged. Warm the air above the chip to 320 K:

```python
warm_air = model.replace(
    boundaries=[
        gfem.FixedTemperature("bottom", temperature=300.0),
        gfem.Convection("top", coefficient=10.0, ambient=320.0),
    ]
)
warm = gfem.solve(warm_air, gfem.SteadyThermal(heat))
warmer = gfem.region_mean(warm.temperature, "core") - core - 300.0
print(f"core rise: {float(gfem.region_mean(warm.rise, 'core')):.3f} K")
print(f"core warmer by {1e3 * float(warmer):.2f} mK")
```


```text
core rise: 12.074 K
core warmer by 0.32 mK
```


The rise is the same. The core warms by a third of a millikelvin: the air's extra 20 K reach it only through $h$ = 10 W/m²/K.

## Check your results

- **The resistance.** For a strip, compare `resistance_per_length` with $1/(\sigma A)$.
- **The heat balance.** The heat through every sink adds up to `steady.power_per_length`.
- **Convergence.** Raise `gfem.SolverOptions(order=3)`, or refine where the temperature bends most: around the heater and across thin low-conductivity layers.

## Limits

- Conductivities are constant and isotropic: no $k(T)$, no $\sigma(T)$, no radiation, no contact resistance. The heat does not change the current.
- A cross-section is infinitely long: the heater's ends and the spreading along the guide need a 3D mesh, read with {py:meth}`gfem.Mesh.from_gmsh`.

## Next

- {doc}`thermo-optic-phase`: the phase this temperature puts on the waveguide.
- {doc}`thermal-step-response`: how fast the temperature follows the current.
- {doc}`thermal-gradient`: the gradient with respect to the materials and the drive.
- {doc}`../tutorials/heater`: the whole phase shifter, against a published device.
- {doc}`prepare-a-mesh`: build a cross-section of your own device.
