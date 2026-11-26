# Thermo-optic phase shift

Find how much a heater shifts the phase of the light in a waveguide: the
phase per watt, the power for a $\pi$ shift ($P_\pi$), how much of it a neighbouring
waveguide picks up, and the phase along a guide through a 3D temperature
map. The temperature comes from a thermal solve ({doc}`heater-temperature`);
{py:obj}`gfem.thermo_optic_shift` turns it into a phase on a cross-section,
and {py:obj}`gfem.route_phase` along a route through a 3D device.

The device is the heater section of {doc}`heater-temperature` and of the
{doc}`heater tutorial <../tutorials/heater>`: a TiN strip 2 µm above a
500 × 220 nm silicon waveguide, at $\lambda$ = 1.55 µm. A material's
`thermo_optic_coefficient` ($\mathrm{d}n/\mathrm{d}T$, in 1/K) sets how its index moves with
temperature. The phase is first order in the temperature rise: one mode
solve gives the weights of every element, and the phase is their sum
weighted by the rise, with the mode held constant. The section must be
lossless. The {term}`thermo-optic phase` is a delay, positive for a positive
dn/dT and a positive rise; a transmission's argument moves by minus it
({doc}`../conventions`, "Temperature and thermo-optic phase").

## The phase per watt on a cross-section

The same mesh holds the heat and the light. The optical solve keeps only the
core and the oxide around it, its {term}`optical window`: the other regions
are left out of electromagnetic solves with `excluded={"em": [...]}`, and
still conduct heat. A wave port on the end face `"front"` carries the core's
mode.

```python
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/heater_w2.msh"
urllib.request.urlretrieve(url, "heater_w2.msh")
section = gfem.Mesh.from_gmsh_cross_section("heater_w2.msh", length_unit="um")

oxide = gfem.Material(
    eps_r=1.444**2, thermal_conductivity=1.38, thermo_optic_coefficient=1e-5
)
silicon = gfem.Material(
    eps_r=3.4777**2, thermal_conductivity=90.0, thermo_optic_coefficient=1.86e-4
)
materials = {
    "substrate": gfem.Material(thermal_conductivity=148.0),
    "box": oxide,
    "cladding": oxide,
    "window": oxide,
    "neighbour_window": oxide,
    "core": silicon,
    "neighbour": silicon,
    "heater": gfem.Material(thermal_conductivity=28.0, conductivity=2.3e6),
}
boundaries = [
    gfem.FixedTemperature("bottom", temperature=300.0),
    gfem.Convection("top", coefficient=10.0, ambient=300.0),
]
thermal_only = ["substrate", "box", "cladding", "heater"]
core = gfem.Model(
    section,
    materials=materials,
    boundaries=boundaries,
    ports=[gfem.WavePort("core", "front")],
    excluded={"em": [*thermal_only, "neighbour", "neighbour_window"]},
)

frequency = 299_792_458.0 / 1.55e-6  # Hz
dc = gfem.solve(core, gfem.DCConduction({"heater": "front"}, ground="back"))
heat = gfem.joule_heating(dc, power_per_length=1.0)  # W/m
unit = gfem.solve(core, gfem.SteadyThermal(heat))
shift = gfem.thermo_optic_shift(core, unit.rise, port="core", frequency=frequency)
p_pi = np.pi * unit.power_per_length / shift.phase_per_length  # W
print(shift)
per_watt = shift.phase_per_length / unit.power_per_length  # rad/W
print(f"phase per watt: {float(per_watt):.1f} rad/W")
print(f"P_pi = {1e3 * float(p_pi):.2f} mW")
```

```text
ThermoOpticShift(port 'core', mode 1, 1.93414e+14 Hz, n_eff 2.447236, delta_n_eff 3.4375e-05, phase_per_length 139.345 rad/m)
phase per watt: 139.3 rad/W
P_pi = 22.55 mW
```

- A heater of length $L$ dissipating $P$ gives $P' = P / L$ per metre and the
  phase $\varphi = \varphi' L$, with $\varphi'$ = `shift.phase_per_length`. So $\pi$ needs
  $P_\pi = \pi P' / \varphi'$, the same for every length. The rise is linear in the power,
  and the phase in the rise, so one solve at 1 W/m serves every drive.
- `shift.delta_n_eff` is the change of the effective index and
  `shift.n_eff` the index without heat; $\varphi' = k_0 \Delta n_\mathrm{eff}$.
- The weights need a mode at the frequency asked, and a section without loss:
  a complex `eps_r`, a `tan_delta` or a `conductivity` in the optical solve's
  regions raises. Leave the heater out of the optical solve, as here.
- `mode=` picks another mode of the port, numbered as in
  {py:obj}`gfem.PortMode`.

## Thermal crosstalk on a second waveguide

A second core 10 µm away warms up too. Read the same rise through that core's
own window: a second model on the same mesh, with its port's optical solve
on the neighbour. The ratio of the two phases is the
{term}`thermal crosstalk`, the same at every power:

```python
neighbour = core.replace(
    ports=[gfem.WavePort("neighbour", "front")],
    excluded={"em": [*thermal_only, "core", "window"]},
)
far = gfem.thermo_optic_shift(
    neighbour, unit.rise, port="neighbour", frequency=frequency
)
print(f"crosstalk: {float(far.phase_per_length / shift.phase_per_length):.4f}")
```

```text
crosstalk: 0.0787
```

The window may also be a mesh of its own, around one core in the thermal
mesh's coordinates: the rise is read at its elements wherever they sit in the
thermal mesh. A window much smaller than the thermal section keeps the mode
solve small; it must still hold the mode's whole field.

## Along a route through a 3D heater

A real heater has ends, bends and pads, and its temperature varies along the
guide. {py:obj}`gfem.route_phase` sweeps the window's weights along a
{term}`route`, a polyline through a 3D rise, and sums the phase over its
length: one mode solve, however long the route.

The 3D mesh here is the same section extruded 20 µm along z, with the heater
along its whole length. The end faces `"front"` and `"back"` are insulated,
so the rise does not vary along z, and the route phase must equal the
section's phase per length times 20 µm:

```python
url = url.replace("heater_w2.msh", "heater_straight.msh")
urllib.request.urlretrieve(url, "heater_straight.msh")
straight = gfem.Model(
    gfem.Mesh.from_gmsh("heater_straight.msh", length_unit="um"),
    materials=materials,
    boundaries=boundaries,
)
length = 20e-6  # m
dc_3d = gfem.solve(straight, gfem.DCConduction({"heater": "front"}, ground="back"))
heat_3d = gfem.joule_heating(dc_3d, power=1e-3)  # W
heated = gfem.solve(straight, gfem.SteadyThermal(heat_3d))
route = gfem.route_phase(
    core,
    heated.rise,
    routes={"core": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, length]])},
    port="core",
    frequency=frequency,
    up=(0.0, 1.0, 0.0),
)
r_section = float(dc.resistance_per_length[0, 0]) * length
print(f"R: {float(dc_3d.resistance[0, 0]):.3f} ohm, R' L: {r_section:.3f} ohm")
print(f"route phase at 1 mW: {float(route.phase['core']):.5f} rad")
per_metre = shift.phase_per_length * 1e-3 / length  # rad/m, the section at 1 mW / L
print(f"section phase x L:   {float(per_metre) * length:.5f} rad")
print(f"P_pi = {1e3 * float(np.pi * heated.power / route.phase['core']):.2f} mW")
```

```text
R: 31.056 ohm, R' L: 31.056 ohm
route phase at 1 mW: 0.13932 rad
section phase x L:   0.13934 rad
P_pi = 22.55 mW
```

The two agree to 2 × 10⁻⁴, the difference between the 3D mesh, coarser
across the section, and the section's own.

- A route is the guide's reference line, an (N, 3) array in metres, through
  as many points as its shape needs. Name several routes to read each part
  of a guide (a {term}`pass` under the heater, the one beside it) from the
  same rise.
- The window lives in the route's frame. Its section is its port's surface,
  a plane $z$ = constant, read in its own $x$ and $y$. At each {term}`station` the
  section's origin sits on the route, its $x$ points to the left of travel
  (`up` × tangent) and its $y$ along `up`, the wafer normal. Here the core's
  bottom centre is the section's origin and the route runs along it.
- Stations are at most `spacing` apart (0.5 µm by default) and the phase is
  the trapezoid sum over them. `route.phase_per_length[name]` gives the phase
  per metre at each station, at `route.arc_length[name]`.
- $P_\pi = \pi P / \varphi$, with $P$ the heater's total power in W (`heated.power`).
- The window's own `thermo_optic_coefficient` weighs the rise; the 3D model's
  is not read.
- In a bend, the weights stay the straight guide's: the route phase does not
  include the change of the mode in the bend.

## Check your results

- **The sanity limit.** If only the core's index moved, a $\pi$ shift over a
  length $L$ would need a core rise of $\lambda / (2 L\, \mathrm{d}n/\mathrm{d}T)$. $P_\pi$ should sit within
  a few tens of percent of the power that gives that rise, since part of the
  mode runs in the warm oxide.
- **First order.** {py:meth}`gfem.Model.with_temperature_rise` gives each
  element the index $n_0 + (\mathrm{d}n/\mathrm{d}T)\, \Delta T$. Solve a {py:obj}`gfem.PortMode` on it at
  the drive you use, and compare the change of `n_eff` with `delta_n_eff`.
  The difference is the second-order term, small for rises of tens of
  kelvin.
- **Convergence.** The phase reads the rise near the core: refine the
  thermal mesh there, and raise `gfem.SolverOptions(order=...)` for both
  solves.

## Limits

- First order in the rise, with the mode held constant: the mode's own
  change with temperature is left out.
- A lossless optical section only: no metal, absorber or `tan_delta` in the
  regions of the optical solve.
- The thermo-optic coefficient is a constant per material: no dn/dT that
  depends on temperature, and no strain-optic effect.
- In 3D, the window is one straight guide's section: in a tight bend the
  mode shifts outwards, which the route phase does not see.

## Next

- {doc}`heater-temperature`: the materials, boundaries and drives behind the
  rise.
- {doc}`thermal-step-response`: the phase over time, as a trace.
- {doc}`thermal-gradient`: the gradient of the phase with respect to the
  materials and the drive.
- {doc}`../tutorials/heater` and {doc}`../tutorials/heater3d`: a phase
  shifter in section, and a spiral heater in 3D.
