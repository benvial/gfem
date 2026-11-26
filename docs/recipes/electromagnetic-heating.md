# Electromagnetic heating

Find how much of the power sent into a device turns into heat, where it
lands, and how warm the device gets: the loss of each region and metal
surface, the temperature under a drive, and the heat reaching the sink.
{py:obj}`gfem.dissipated_power` splits a driven field's loss by region and
sheet ({term}`dissipated power`), {py:obj}`gfem.electromagnetic_heating`
turns a driven result and a drive into a {py:obj}`gfem.HeatSource`
({term}`electromagnetic heating`), and {py:obj}`gfem.SteadyThermal` solves
the heat.

The device is the line of {doc}`../get-started`: two parallel plates 40 mm
long, 6 mm wide and 1 mm apart, with a ceramic of relative permittivity 10
between them over the middle 20 mm. Here the ceramic has a loss tangent of
0.02, the top plate is copper, and the bottom plate is held at 300 K. The
electromagnetic solve is full-wave and linear, at one frequency; the heat
conduction is linear and steady, and the heat does not change the
materials. Units are SI, phasors follow $\exp(+i\omega t)$, and temperatures
are in kelvin ({doc}`../conventions`).

## Loss split

A kept driven field is the field for 1 W incident at its excited port, with
every other port matched. {py:obj}`gfem.dissipated_power` gives the power
that field dissipates in each volume region and on each surface carrying a
{py:obj}`gfem.Conductor` or {py:obj}`gfem.Impedance`, in W: the fraction of
the incident power each one turns into heat.

```python
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
urllib.request.urlretrieve(url, "line.msh")
mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")

z_line = 376.73 / 6  # ohm: the empty line's impedance
ports = [
    gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
    gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
]
air = gfem.Material(thermal_conductivity=0.026)  # W/m/K
ceramic = gfem.Material(eps_r=10.0, tan_delta=0.02, thermal_conductivity=30.0)
boundaries = [
    gfem.PEC("bottom"),
    gfem.Conductor("top", 5.8e7),  # S/m: a copper top plate
    gfem.FixedTemperature("bottom", temperature=300.0),  # K: the heat sink
]
model = gfem.Model(
    mesh,
    materials={"feed": air, "design": ceramic},
    boundaries=boundaries,
    ports=ports,
)

f = 5e9  # Hz
result = gfem.solve(model, gfem.Driven(frequency=f, keep_fields=True))
lost = gfem.dissipated_power(result.e_field(f, "p1"), model=model)  # W per 1 W
for name, value in lost.items():
    print(f"{name:>6}: {float(value):.6f}")
s11, s21 = result.s["p1", "p1"][0], result.s["p2", "p1"][0]
print(f"   sum: {sum(float(value) for value in lost.values()):.6f}")
print(f"1 - |S11|^2 - |S21|^2: {1 - abs(s11) ** 2 - abs(s21) ** 2:.6f}")
```

```text
  feed: 0.000000
design: 0.166443
   top: 0.004820
   sum: 0.171263
1 - |S11|^2 - |S21|^2: 0.171263
```

- The keys are every volume region of the solve domain, a lossless one at
  0, and every {py:obj}`gfem.Conductor` and {py:obj}`gfem.Impedance`
  surface. A name that is both a region and such a surface raises.
- The loss is read from the solved field and the model's materials: the
  loss tangent, the conductivity, a complex or anisotropic permittivity, a
  {py:obj}`gfem.MaterialField` and a permittivity law at the field's
  frequency. A sheet dissipates $\mathrm{Re}(1/Z_s)\, |E_t|^2/2$ per area
  for a `Conductor` and $|E_t|^2/(2R)$ for an `Impedance`, times its fill.
- Pass the model the field was solved with: the field does not carry its
  materials.
- **The sum is what S does not account for.** On a closed device whose
  ports all have a real reference impedance, the losses add up to
  $1 - \sum_j |S_{j1}|^2$, the power that neither returns nor reaches another
  port. That needs every port to have a real reference impedance and to
  sit on a uniform stretch of line: here both lumped ports have a real
  resistance, in air. To round-off it also needs element order 3 or
  more: at the default order 2, S read on a lumped port's faces is off by
  about $10^{-6}$, below the six digits printed here, and a port on a
  nonuniform stretch is off by about $10^{-5}$. On a wave port on a lossy
  line, whose reference impedance is complex, the field is per unit
  $|a|^2$, which is not exactly 1 W incident, and S does not close the
  balance: read the power entering a region with {py:obj}`gfem.power_flow`
  instead ({doc}`measurements`).
- A region's value is minus `gfem.power_flow` out of it when no sheet lies
  on its surface.

## Heat and temperature

The heat of a drive needs no new solve: {py:obj}`gfem.electromagnetic_heating`
scales the kept fields at one frequency by the incident power of each port,
in W. The result is a heat source on the regions, with the sheet's heat on
its faces ({term}`sheet heat`):

```python
heat = gfem.electromagnetic_heating(
    result, model=model, frequency=f, incident_power={"p1": 1.0}
)
steady = gfem.solve(model, gfem.SteadyThermal(heat))
print(heat)
print(f"power: {float(steady.power):.6f} W")
print(f"into the sink: {float(gfem.heat_flow(steady, 'bottom', model=model)):.6f} W")
for region in ("design", "feed"):
    rise = float(gfem.region_mean(steady.rise, region))
    print(f"mean rise, {region}: {rise:.4f} K")
```

```text
HeatSource(electromagnetic, regions=['feed', 'design'], total 0.171263 W: volume 0.166443 W, sheet 0.00481981 W)
power: 0.171263 W
into the sink: 0.171263 W
mean rise, design: 0.0159 K
mean rise, feed: 0.1838 K
```

- `frequency` must be one the result kept fields at. `incident_power` is a
  mapping from port name to W; a number is enough when the result excited
  one port only. A port the solve did not excite raises `KeyError`, naming
  `Driven(excite=)`.
- The heat is the drive times the dissipated power: 1 W in at `p1` puts
  0.171 W into the device. The feed's air is lossless, yet it warms most:
  the copper's heat over it reaches the sink through 1 mm of air.
- {py:obj}`gfem.heat_flow` is the heat leaving the device through a
  surface, in W, read from the solved temperature
  ({doc}`heater-temperature`, "Where the heat goes"). In steady state all
  of it leaves through the sink.
- A heat source from a DC current comes from {py:obj}`gfem.joule_heating`
  instead; {py:obj}`gfem.joule_heating` of a driven result raises
  `TypeError`, naming `electromagnetic_heating`.

## What is not heat

The power a driven field loses inside the model is heat. Power that leaves
it is not, and gets none:

- what radiates out through an {py:obj}`gfem.Absorbing` face, or into a
  {term}`perfectly matched layer`;
- what leaves through a wave port or a Floquet port;
- what a {py:obj}`gfem.LumpedPort`'s resistance takes: it is the load of the
  network the port stands for, outside the model.

A resistor that should heat the device is a part of the model: an
{py:obj}`gfem.Impedance` sheet with its resistance per square, or a
conducting volume.

## Where sheet heat goes

A sheet's heat enters the thermal problem at its faces, and the face's
thermal boundary decides where it goes. On an exterior face:

- an {term}`insulated surface` sends all of it into the device, as here: the
  copper's heat crosses the air to the sink;
- a {py:obj}`gfem.FixedTemperature` face takes all of it, and the device
  never sees it;
- a {py:obj}`gfem.Convection` face shares it with the device.

Cool the top plate with air at 10 W/m²/K:

```python
cooled = model.replace(
    boundaries=[*boundaries, gfem.Convection("top", coefficient=10.0, ambient=300.0)]
)
cool = gfem.solve(cooled, gfem.SteadyThermal(heat))
for surface in ("top", "bottom"):
    out = float(gfem.heat_flow(cool, surface, model=cooled))
    print(f"out through {surface}: {out:.6f} W")
```

```text
out through top: 0.000353 W
out through bottom: 0.170910 W
```

The air above takes a fraction of a percent: 10 W/m²/K is weak beside the
1 mm of ceramic and air below. A metal enclosure modelled as a lossy wall
with nothing outside it is therefore "insulated outside": all of its heat
flows into the model. Give its outside a convection, or a fixed
temperature, when the enclosure is cooled.

## Coherent and incoherent drives

Ports fed at one frequency in one call are a {term}`coherent drive`: their
fields add before they dissipate, so the heat depends on their relative
phase, set at each port's reference plane by `incident_phase` (rad). Feed
both ends with 0.5 W each, in phase and in opposition:

```python
for phase in (0.0, np.pi):
    both = gfem.electromagnetic_heating(
        result,
        model=model,
        frequency=f,
        incident_power={"p1": 0.5, "p2": 0.5},
        incident_phase={"p2": phase},
    )
    print(f"p2 at {phase:.4f} rad: {both}")
tones = [
    gfem.electromagnetic_heating(
        result, model=model, frequency=f, incident_power={port: 0.5}
    )
    for port in ("p1", "p2")
]
apart = gfem.solve(model, gfem.SteadyThermal(tones))
print(f"separate tones: {float(apart.power):.6f} W")
```

```text
p2 at 0.0000 rad: HeatSource(electromagnetic, regions=['feed', 'design'], total 0.300242 W: volume 0.292912 W, sheet 0.00732965 W)
p2 at 3.1416 rad: HeatSource(electromagnetic, regions=['feed', 'design'], total 0.0422843 W: volume 0.0399743 W, sheet 0.00230998 W)
separate tones: 0.171263 W
```

- The line is symmetric: in phase, the two waves reach the middle of the
  ceramic with the same phase and their fields add there; in opposition
  they cancel there. The heat goes from 0.30 W to 0.04 W.
- Drives that are not coherent (other tones, unrelated sources) are separate
  heat sources: a thermal problem takes a sequence and adds them. Their sum
  is the mean of the coherent heat over the phase.
- A reference plane moved with `WavePort(offset=)` or
  `result.with_offsets` moves the phase with it, as it does for S.

## Cross-sections

A device that does not vary along one axis is solved on a
{term}`slab cross-section`, read with
{py:meth}`gfem.Mesh.from_gmsh_cross_section`. Pose the driven solve on the
slab, with its ports and walls, and give the drive per metre:
`gfem.electromagnetic_heating(slab_result, model=slab, frequency=f, incident_power_per_length=p)`
with `p` in W/m. The heat is then per metre too, and the thermal solve on
the same slab reads it as `power_per_length`. On a slab `incident_power=`
raises, and on a 3D mesh `incident_power_per_length=` does.

A {term}`2D model` has no heat: `dissipated_power` and
`electromagnetic_heating` of a native 2D result raise
`NotImplementedError`, naming the slab.

## Gradients

The heat is differentiable, so a temperature is an objective of the
electromagnetic design: through the driven solve, and directly through the
loss. The mean rise of the ceramic with respect to its loss tangent, per
element:

```python
import jax
import jax.numpy as jnp

n = len(mesh.elements("design"))


def ceramic_rise(tan_delta):
    """The ceramic's mean rise, K, for 1 W into p1."""
    lossy = gfem.MaterialField(
        eps_r=jnp.full(n, 10.0),
        tan_delta=tan_delta,
        thermal_conductivity=jnp.full(n, 30.0),
    )
    m = model.with_material("design", lossy)
    r = gfem.solve(m, gfem.Driven(frequency=f, excite="p1", keep_fields=True))
    heat = gfem.electromagnetic_heating(r, model=m, frequency=f, incident_power=1.0)
    return gfem.region_mean(gfem.solve(m, gfem.SteadyThermal(heat)).rise, "design")


tan_delta = jnp.full(n, 0.02)
rise, d_tan = jax.value_and_grad(ceramic_rise)(tan_delta)
step = 1e-4
fd = (ceramic_rise(tan_delta + step) - ceramic_rise(tan_delta - step)) / (2 * step)
print(f"mean rise: {float(rise):.4f} K")
print(f"sum of the gradient: {float(jnp.sum(d_tan)):.6f} K")
print(f"central difference:  {float(fd):.6f} K")
```

```text
mean rise: 0.0159 K
sum of the gradient: 0.654832 K
central difference:  0.654833 K
```

- The sum of the per-element gradient is the derivative along a uniform
  change of the loss tangent, which the central difference measures.
- **What differentiates.** The permittivity, real or complex and per
  axis, the loss tangent, the conductivity, a permittivity law's
  parameters, a {py:obj}`gfem.SheetField` fill, an `Impedance`'s
  resistance, inductance and capacitance, the thermal conductivity, and the
  drive: the incident powers and phases.
- **A port at exactly 0 W next to a coherent partner.** The heat's cross
  term grows as the square root of the two powers, so its derivative with
  respect to a power of exactly 0 W is unbounded (`inf`). Start such a port
  at a small power.
- **A metal's conductivity is fixed.** A `Conductor`'s conductivity and
  thickness raise under `jax.grad`; to design a lossy metal, vary a sheet's
  fill or an `Impedance`'s resistance.
- The frequency and the mesh node coordinates raise
  {py:obj}`gfem.NonDifferentiableError`.

The full table is on {doc}`../capabilities`.

## Check your results

- **The loss balance.** On a device with real port references, compare the
  sum of `dissipated_power` with $1 - \sum_j |S_{j1}|^2$, as above.
- **The heat balance.** In steady state, `heat_flow` through every heat
  sink adds up to `steady.power`.
- **Convergence.** Raise `gfem.SolverOptions(order=3)` or refine the mesh
  where the field is strong in a lossy region: the heat follows the square
  of the field, so it converges no faster than the field itself. A metal
  modelled as a conducting volume needs elements smaller than its skin
  depth; a `Conductor` does not.

## Limits

- **Magnetic loss** is not heat in this version: a region with a complex
  `mu_r` raises `NotImplementedError`, naming it.
- **Eigenmodes** have no drive: an eigenmode result raises `TypeError`.
- **A port mode** is a line, not a device: a port mode result raises. For the
  loss of a line per length, use
  {py:meth}`gfem.HeatSource.uniform` with `power_per_length=`.
- **An adaptive sweep's** fields heat, but their gradients raise: use
  `sweep="uniform"`.
- **Several MPI ranks** and **mesh refinement** objectives raise, naming the
  call.
- The materials are constant: the heat does not change the loss, and there
  is no electro-thermal iteration.

## Next

- {doc}`heater-temperature`: thermal boundaries, drives and where the heat
  goes.
- {doc}`thermal-step-response`: the same heat switched on and off.
- {doc}`thermal-gradient`: the heat through a sink as an objective.
- {doc}`measurements`: power through a surface and dissipated power.
- {doc}`../conventions`: incident power, reference impedances and the sign
  of loss.
