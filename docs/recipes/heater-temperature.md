# Heater temperature

Find the temperature of a device heated by a current: the resistance of the
heater, the power it dissipates, and how much each part of the device warms
up. gfem solves the current with {py:obj}`gfem.DCConduction`, turns it into
heat with {py:obj}`gfem.joule_heating`, and solves the heat with
{py:obj}`gfem.SteadyThermal`.

The device is the cross-section of the
{doc}`heater tutorial <../tutorials/heater>`: a titanium nitride strip,
2 µm wide and 0.14 µm thick, 2 µm above a silicon waveguide in oxide, on a
silicon substrate whose bottom is held at 300 K. The section is 40 µm wide
and 18 µm high. Heat conduction is linear: every conductivity is constant,
and nothing radiates. The section is a slice of a long device, so nothing
varies along the guide: the current and the heat flow out of the section
plane, through the strip, and every extensive quantity is per metre of guide
({term}`per-length quantity`). Temperatures are absolute, in kelvin
({doc}`../conventions`, "Temperature and thermo-optic phase").

## Materials and boundaries

The mesh is a triangulated section, read as a {term}`slab cross-section`: one
element thick along z, with the two {term}`end faces <End faces>` `"front"`
and `"back"`. A material takes its thermal properties beside its optical
ones. Only the heater has an electrical `conductivity`, so only it carries
current.

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

- A {py:obj}`gfem.FixedTemperature` surface and a {py:obj}`gfem.Convection`
  surface are {term}`heat sinks <Heat sink>`. Every connected piece of the
  {term}`thermal domain` needs one in a steady solve; a piece without one
  raises, naming a region of it.
- A surface no thermal boundary names is an {term}`insulated surface`: no
  heat crosses it. `model.default_insulated` lists the named ones; here the
  end faces, which keep the solution the same all along the guide. The
  section's unnamed sides are insulated too.
- {py:obj}`gfem.HeatFlux` puts a heat flux through a surface, positive into
  the device, in W/m².
- `excluded=` takes a mapping from physics (`"em"`, `"thermal"`,
  `"dc_conduction"`) to the regions left out of it. Here the substrate and
  the heater are left out of electromagnetic solves only, so the same model
  can solve the waveguide's mode ({doc}`thermo-optic-phase`) while both still
  conduct heat. A plain list or tuple keeps its old meaning: electromagnetic
  solves only.
- A thermal solve needs `thermal_conductivity` in every region of the thermal
  domain, and a transient one also `mass_density` and `specific_heat`
  ({doc}`thermal-step-response`). A missing property raises
  {py:obj}`gfem.errors.MissingMaterialPropertyError`, listing every region
  and property at once.

## The heater's resistance

{py:obj}`gfem.DCConduction` holds each terminal surface at 1 V against the
grounded ones, and solves for the current in the conductors. On a
cross-section the terminals are the end faces, and the current flows along
the guide:

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

- `dc.conductance` is the {term}`conductance matrix` in siemens, in the order
  of `dc.terminals`, and `dc.resistance` its inverse. On a cross-section the
  slab's values are for its thickness; read `resistance_per_length` instead.
- A uniform strip carrying a current along its length has the resistance
  $1/(\sigma A)$ per metre: the solve agrees to rounding.
- One DC result serves every drive: the conductance is linear, so no drive
  needs a new solve.

## The temperature at a drive

{py:obj}`gfem.joule_heating` turns the DC result and a drive into a
{py:obj}`gfem.HeatSource` on the conductors. On a cross-section the drive is
the current through the heater (the same number as in the device, in A), or
the power per metre of heater:

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

- `steady.temperature` is the absolute temperature in K, and `steady.rise`
  the {term}`temperature rise`: the temperature minus the
  {term}`source-free temperature`, the temperature with every heat source
  off. Here every boundary is at 300 K, so the rise is the temperature above
  300 K.
- Both are fields: read them anywhere with `.at(points)`, as means over a
  region with {py:obj}`gfem.region_mean`, per element with
  {py:obj}`gfem.element_mean`, or export them with `to_xarray`.
- `steady.power` is the heat the problem's sources put in, in W;
  `steady.power_per_length` the same per metre, on a cross-section.
- On a 3D mesh, drive by `voltage=` (V), `current=` (A) or `power=` (W).
  On a cross-section `voltage=` and `power=` raise, because a voltage or a
  power fixes nothing without the heater's length: the message gives the
  per-length form. `power_per_length=` raises on a 3D mesh.
- With several terminals, give `voltage=` or `current=` as a mapping from
  terminal name to value; terminals left out are at 0 V or 0 A.

## Where the heat goes

In steady state every watt the heater dissipates leaves the section through a
{term}`heat sink`: down into the substrate's bottom, or up through the top
into the air. {py:obj}`gfem.heat_flow_per_length` gives the heat leaving
through named surfaces, or out of a set of elements, in W/m, positive out
({term}`heat flow`):

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

- A surface name gives the heat leaving through it: a
  {py:obj}`gfem.FixedTemperature` surface the heat it removes, a
  {py:obj}`gfem.Convection` surface $\int h (T - T_\infty)\, dS$, a
  {py:obj}`gfem.HeatFlux` surface minus the power it puts in, an insulated
  one 0. Several names give their union.
- A volume region, several, or a predicate on the element centroids (m)
  gives the heat leaving those elements into the rest of the section; what
  crosses the section's own boundary there counts inside. The cut above is
  the staircase of element faces nearest $y = -12$ µm, 3.14 µm above the
  sink. No heat is made below it, so all of the sink's heat crosses it. A
  cut whose elements reach a fixed-temperature surface takes their share of
  its heat inside: keep it an element away to read the flow across it.
- The heat is read from the discrete heat balance of the solved
  temperature, not from its gradient, so the flows close to round-off, and
  a set of elements gives minus what its complement gives. A face carrying
  heat that lies exactly on a cut, a sheet's or a
  {py:obj}`gfem.HeatFlux`'s, belongs to neither side.
- On a 3D mesh, {py:obj}`gfem.heat_flow` gives the same in W; each raises
  on the other kind of mesh, naming its twin. A transient run records it as
  {py:meth}`gfem.Trace.heat_out` ({doc}`thermal-step-response`).

## Heat without a current

When you know the power a region dissipates, skip the DC solve:
{py:meth}`gfem.HeatSource.uniform` spreads a power evenly over the volume of
one or more regions. A straight strip carries a uniform current, so its Joule
heat is uniform, and the two sources give the same temperature:

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

- The heat of a driven electromagnetic solve's loss comes from
  {py:obj}`gfem.electromagnetic_heating`, in the same form
  ({doc}`electromagnetic-heating`).
- `HeatSource.uniform` takes exactly one amount: `power=` (W, on a 3D mesh),
  `power_per_length=` (W/m, on a cross-section) or `power_density=` (W/m³,
  on either).
- Sources superpose: `gfem.SteadyThermal([heat, uniform])` solves both
  together, and doubling a source doubles the rise.

## A boundary that sets the temperature, not the rise

The rise follows the heat sources only. A boundary temperature, an ambient or
a {py:obj}`gfem.HeatFlux` moves the source-free temperature, and the
temperature with it, but leaves the rise unchanged. Warm the air above the
chip to 320 K:

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

The rise is the same. The core warms by a third of a millikelvin: the air's
extra 20 K reach it only through $h$ = 10 W/m²/K, against the 18 µm of oxide
and silicon below, which conduct the heat to the sink far better.

## Check your results

- **The resistance.** For a strip, compare `resistance_per_length` with
  $1/(\sigma A)$, as above. For your own heater, the number of squares between its
  contacts times the sheet resistance $1/(\sigma t)$ is a close estimate.
- **The heat balance.** In steady state, every watt the sources put in leaves
  through the heat sinks: the heat through every sink adds up to
  `steady.power_per_length`, as in "Where the heat goes".
- **Convergence.** Raise `gfem.SolverOptions(order=3)` (the default is 2), or
  refine the mesh where the temperature bends most: around the heater and
  across thin layers of low conductivity.

## Limits

- Conductivities are constant and isotropic: no $k(T)$, no $\sigma(T)$, no radiation,
  no thermal contact resistance between layers. The current and the heat are
  solved one after the other; the heat does not change the current.
- A cross-section is infinitely long: the heater's ends, its contacts and
  the heat spreading along the guide need a 3D mesh, read with
  {py:meth}`gfem.Mesh.from_gmsh`, where the same calls take `power=` in W.
- The conductors' ends are the terminals: a contact pad, a via or a face of
  the mesh, each a named surface.

## Next

- {doc}`thermo-optic-phase`: the phase this temperature puts on the waveguide.
- {doc}`thermal-step-response`: how fast the temperature follows the current.
- {doc}`thermal-gradient`: the gradient of a temperature or a phase with
  respect to the materials and the drive.
- {doc}`../tutorials/heater`: the whole phase shifter, against a published
  device.
- {doc}`prepare-a-mesh`: build a cross-section of your own device.
