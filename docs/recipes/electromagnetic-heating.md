# Electromagnetic heating

Find how much of the power sent into a device turns into heat, where it
lands, and how warm the device gets: the loss of each region and metal
surface, the temperature under a drive, and the heat reaching the sink.
{py:obj}`gfem.dissipated_power` ({term}`dissipated power`) splits a driven
field's loss, and {py:obj}`gfem.electromagnetic_heating`
({term}`electromagnetic heating`) makes a {py:obj}`gfem.HeatSource` of it. The
device is the line of {doc}`../get-started` with a lossy ceramic
({doc}`../conventions`).

## Loss split

{py:obj}`gfem.dissipated_power` gives, in W per 1 W incident at the excited
port, the loss in each volume region and on each sheet surface.

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

- The keys are every volume region and every sheet surface; a name that is both raises.
- Pass the model the field was solved with. A sheet dissipates
  $\mathrm{Re}(1/Z_s)\, |E_t|^2/2$ per area for a `Conductor` and
  $|E_t|^2/(2R)$ for an `Impedance`, times its fill.
- **The sum is what S does not account for.** With real port reference
  impedances on uniform stretches of line, the losses add up to
  $1 - \sum_j |S_{j1}|^2$, to round-off at element order 3 or more. On a wave
  port on a lossy line S does not close the balance: read the power entering a
  region with {py:obj}`gfem.power_flow` ({doc}`measurements`).

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

- `incident_power` maps port name to W, at a frequency the result kept. A
  port the solve did not excite raises `KeyError`, naming `Driven(excite=)`.
- The feed's air is lossless, yet it warms most: the copper's heat over it
  reaches the sink through 1 mm of air.

## Where sheet heat goes

A sheet's heat enters at its faces. On an exterior face an
{term}`insulated surface` sends all of it into the device, a
{py:obj}`gfem.FixedTemperature` face takes all of it, a
{py:obj}`gfem.Convection` face shares it. Cool the top plate with air at 10 W/m²/K:

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

The air takes a fraction of a percent. A lossy wall with nothing outside is
insulated: give it a convection or a fixed temperature when it is cooled.

## Coherent and incoherent drives

Ports fed at one frequency in one call are a {term}`coherent drive`: their
fields add before they dissipate, so the heat depends on their relative phase,
`incident_phase` (rad). Feed both ends with 0.5 W, in phase and in opposition:

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

- In phase the waves add in the ceramic, in opposition they cancel.
- Incoherent drives are separate heat sources: pass a sequence.

## Cross-sections

On a {term}`slab cross-section` give the drive per metre:
`gfem.electromagnetic_heating(slab_result, model=slab, frequency=f, incident_power_per_length=p)`
with `p` in W/m; the heat is per metre too. On a slab `incident_power=` raises,
and on a 3D mesh `incident_power_per_length=` does. A {term}`2D model` has no
heat: its results raise `NotImplementedError`, naming the slab.

## Gradients

The heat is differentiable, so a temperature is an objective of the
electromagnetic design. The mean rise of the ceramic with respect to its loss
tangent, per element:

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

- The gradient's sum is the derivative along a uniform change of the loss tangent.
- **What differentiates.** The permittivity, loss tangent, conductivity, a
  law's parameters, a {py:obj}`gfem.SheetField` fill, an `Impedance`'s
  resistance, inductance and capacitance, the thermal conductivity, and the
  incident powers and phases.
- **A port at exactly 0 W next to a coherent partner** has an `inf` derivative
  (the cross term goes as the square root of the powers). Start it at a small power.
- **A metal's conductivity is fixed.** A `Conductor`'s conductivity and
  thickness raise under `jax.grad`; vary a sheet's fill or an `Impedance`'s
  resistance.
- The frequency and node coordinates raise {py:obj}`gfem.NonDifferentiableError`.

The full table is on {doc}`../capabilities`.

## Check your results

- **The loss balance.** Compare the sum of `dissipated_power` with
  $1 - \sum_j |S_{j1}|^2$, as above.
- **The heat balance.** `heat_flow` through every heat sink adds up to
  `steady.power` ({doc}`heater-temperature`, "Where the heat goes").
- **Convergence.** Raise `gfem.SolverOptions(order=3)` or refine where the
  field is strong in a lossy region. A conducting volume needs elements
  smaller than its skin depth; a `Conductor` does not.

## Limits

- **Power that leaves the model** gets no heat: what radiates through an
  {py:obj}`gfem.Absorbing` face or a {term}`perfectly matched layer`, leaves
  through a wave or Floquet port, or a {py:obj}`gfem.LumpedPort`'s resistance
  takes. Model a resistor that should heat the device as an
  {py:obj}`gfem.Impedance` sheet or a conducting volume.
- **Magnetic loss** is not heat in this version: a region with a complex
  `mu_r` raises `NotImplementedError`, naming it.
- **A port mode** is a line, not a device: its result raises. For a line's loss
  per length use {py:meth}`gfem.HeatSource.uniform` with `power_per_length=`.
- **An adaptive sweep's** fields heat, but their gradients raise: use
  `sweep="uniform"`.
- **The wrong result type** raises `TypeError`: an eigenmode result has no
  drive, and {py:obj}`gfem.joule_heating` of a driven result names
  `electromagnetic_heating`.
- **Several MPI ranks**, **eigenmodes** and **mesh refinement** objectives raise, naming the call.
- The heat does not change the loss: there is no electro-thermal iteration.

## Next

- {doc}`heater-temperature`: thermal boundaries and drives.
- {doc}`thermal-step-response`: the same heat switched on and off.
- {doc}`thermal-gradient`: the heat through a sink as an objective.
- {doc}`measurements`: power through a surface.
