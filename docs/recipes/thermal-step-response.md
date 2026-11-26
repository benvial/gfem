# Thermal step response

Find how fast a device heats up and cools down when its heater switches on
and off: the temperature and the phase over time, the rise time and the time
constant. {py:obj}`gfem.TransientThermal` steps the heat equation
$\rho c_p\, \partial T/\partial t - \nabla\cdot(k \nabla T) = s(t)\, q$ on a uniform time grid, and records
{term}`traces <Trace>` of the temperature at every step.

The device is the section of {doc}`heater-temperature`; every material now also
needs `mass_density` and `specific_heat` ({doc}`../conventions`, "Temperature and
thermo-optic phase").

## A step on and off

The heater dissipates 70 W/m, about a $\pi$ phase ({doc}`thermo-optic-phase`),
from $t = 0$ to 80 µs; the run lasts 160 µs in 320 steps.

```python
import urllib.request
import warnings

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/heater_w2.msh"
urllib.request.urlretrieve(url, "heater_w2.msh")
section = gfem.Mesh.from_gmsh_cross_section("heater_w2.msh", length_unit="um")

oxide = gfem.Material(
    eps_r=1.444**2,
    thermal_conductivity=1.38,  # W/m/K
    mass_density=2203.0,  # kg/m^3
    specific_heat=709.0,  # J/kg/K
    thermo_optic_coefficient=1e-5,  # 1/K
)
silicon = gfem.Material(
    eps_r=3.4777**2,
    thermal_conductivity=90.0,
    mass_density=2330.0,
    specific_heat=711.0,
    thermo_optic_coefficient=1.86e-4,
)
materials = {
    "substrate": gfem.Material(
        thermal_conductivity=148.0, mass_density=2330.0, specific_heat=711.0
    ),
    "box": oxide,
    "cladding": oxide,
    "window": oxide,
    "neighbour_window": oxide,
    "core": silicon,
    "neighbour": silicon,
    "heater": gfem.Material(
        thermal_conductivity=28.0, mass_density=5240.0, specific_heat=598.0
    ),
}
thermal_only = ["substrate", "box", "cladding", "heater"]  # no optics there
model = gfem.Model(
    section,
    materials=materials,
    boundaries=[
        gfem.FixedTemperature("bottom", temperature=300.0),
        gfem.Convection("top", coefficient=10.0, ambient=300.0),
    ],
    ports=[gfem.WavePort("core", "front")],
    excluded={"em": [*thermal_only, "neighbour", "neighbour_window"]},
)
frequency = 299_792_458.0 / 1.55e-6  # Hz
phase = gfem.thermo_optic_shift(model, None, port="core", frequency=frequency).trace()

t_off = 80e-6  # s
step = gfem.TransientThermal(
    gfem.HeatSource.uniform("heater", power_per_length=70.0),  # W/m
    duration=160e-6,  # s
    steps=320,
    waveform=[(0.0, 1.0), (t_off, 0.0)],  # (start time in s, scale)
    traces={
        "core": gfem.Trace.mean("core"),
        "heater": gfem.Trace.point((0.0, 2.29e-6, section.section_thickness / 2)),
        "phase": phase,
        "sink": gfem.Trace.heat_out_per_length("bottom"),
    },
    keep_fields=[40e-6],
)
run = gfem.solve(model, step)
t = np.asarray(run.time)
core = np.asarray(run.traces["core"])  # K, the core's mean temperature
on = t <= t_off
print(t.shape, run.switch_times)
print(f"core at switch-off: {core[on][-1]:.3f} K, at the end: {core[-1]:.4f} K")
heater = np.asarray(run.traces["heater"])  # K
print(f"heater centre at switch-off: {heater[on][-1]:.2f} K")
phase_320 = np.asarray(run.traces["phase"]) * 320e-6  # rad, over 320 um of guide
print(f"phase at switch-off: {phase_320[on][-1]:.4f} rad")
```

```text
(321,) (0.0, 8e-05)
core at switch-off: 311.978 K, at the end: 300.0038 K
heater centre at switch-off: 331.80 K
phase at switch-off: 3.1203 rad
```

- `run.time` has `steps + 1` times from 0 to `duration`. A switch time must be
  on the grid; one off it raises, naming the nearest grid times.
- The {term}`waveform` scales every heat source: off before its first start
  time, then each scale until the next start time.
- {py:meth}`gfem.Trace.mean` and {py:meth}`gfem.Trace.point` record absolute
  temperatures, in K. `thermo_optic_shift(...).trace()` and
  `route_phase(...).trace(name)` record the phase; their mode is solved once,
  before the run.
- With no `initial_temperature`, the run starts from the
  {term}`source-free temperature`, so every rise starts at 0.

## Fields at chosen times

The run keeps the temperature field at the final time and at the times
named in `keep_fields` (`True` keeps every step):

```python
print(np.round(np.asarray(run.kept_times) * 1e6, 6))  # us
halfway = run.rise(40e-6)  # a Field, K
print(f"mean core rise at 40 us: {float(gfem.region_mean(halfway, 'core')):.3f} K")
```

```text
[ 40. 160.]
mean core rise at 40 us: 11.818 K
```

`run.temperature(time)` gives the absolute temperature. A time that was not
kept raises, naming the kept ones; on a large mesh keep only the times you need.

## Rise time and time constant

{py:func}`gfem.rise_time` is the 10–90 % time of a trace, and
{py:func}`gfem.time_constant` the time to $1 - 1/e$ = 63.2 % of its swing.
Both normalise between the first and last samples, so pass the part of the
trace that goes with one switch, and its time:

```python
tau = gfem.time_constant(t[on], core[on], switch_times=(0.0,))
rise = gfem.rise_time(t[on], core[on], switch_times=(0.0,))
off = t >= t_off
tau_off = gfem.time_constant(t[off], core[off], switch_times=(t_off,))
print(f"time constant: {tau * 1e6:.2f} us on, {tau_off * 1e6:.2f} us off")
print(f"10-90 % rise time: {rise * 1e6:.2f} us, ratio {rise / tau:.2f}")
```

```text
time constant: 8.56 us on, 8.56 us off
10-90 % rise time: 18.56 us, ratio 2.17
```

- The model is linear, so the fall mirrors the rise.
- A single exponential has rise time / time constant = $\ln 9 = 2.20$; a ratio
  near it means one slow thermal mode dominates.

## The heat that has left

{py:meth}`gfem.Trace.heat_out_per_length` records the {term}`heat out` of a
side: the heat that has left it since $t = 0$, in J/m (J on a 3D mesh, with
{py:meth}`gfem.Trace.heat_out`). Its step difference over `dt` is the mean
{term}`heat flow` over each step, which tends to the steady flow:

```python
heat_out = np.asarray(run.traces["sink"])  # J/m since t = 0
dt = t[1] - t[0]
flow = np.diff(heat_out) / dt  # W/m, the mean over each step
steady = gfem.solve(model, gfem.SteadyThermal(step.heat))
held = float(gfem.heat_flow_per_length(steady, "bottom", model=model))
put_in, out = 70.0 * t_off * 1e3, heat_out[-1] * 1e3  # mJ/m
print(f"heat in: {put_in:.4f} mJ/m, out by the end: {out:.4f} mJ/m")
print(f"mean flow over the last step on: {flow[on[1:]][-1]:.4f} W/m")
print(f"steady flow into the sink: {held:.4f} W/m")
```

```text
heat in: 5.6000 mJ/m, out by the end: 5.5988 mJ/m
mean flow over the last step on: 69.9032 W/m
steady flow into the sink: 69.9978 W/m
```

- The heat out of every surface plus the heat stored equals the heat the
  sources put in, at every time and for every time scheme.
- At switch-off the flow into the sink is 0.14 % short of the steady one:
  slower thermal modes than the core's are still filling. By the end the
  device has given back all but 0.02 % of its heat.
- The trace is a cumulative heat: take `np.diff` before
  {py:func}`gfem.rise_time` or {py:func}`gfem.time_constant`.

## A trace that has not settled

When the last samples still change by more than 1 % of the extrapolated swing,
the {term}`settled trace` test fails and both functions warn with
{py:obj}`gfem.UnsettledTraceWarning`. The first 15 µs of the step are not settled:

```python
early = t <= 15e-6
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    gfem.time_constant(t[early], core[early], switch_times=(0.0,))
print([w.category.__name__ for w in caught])
```

```text
['UnsettledTraceWarning']
```

Run longer: about five time constants after the last switch settle a
single exponential to 1 %.

## The time scheme

The default TR-BDF2 is second order and damps the jump at a switch.
`SolverOptions(time_scheme="theta", theta=1.0)` is backward Euler, first
order; `theta=0.5` is Crank-Nicolson.

```python
euler = gfem.solve(
    model,
    gfem.TransientThermal(
        step.heat,
        duration=80e-6,
        steps=160,
        traces={"core": gfem.Trace.mean("core")},
    ),
    options=gfem.SolverOptions(time_scheme="theta", theta=1.0),
)
core_euler = np.asarray(euler.traces["core"])
t_euler = np.asarray(euler.time)
tau_euler = gfem.time_constant(t_euler, core_euler, switch_times=(0.0,))
print(f"time constant, backward Euler: {tau_euler * 1e6:.2f} us")
```

```text
time constant, backward Euler: 8.86 us
```

Backward Euler lags by 3.5 % at 0.5 µs steps; halving the step roughly halves the lag.

## Check your results

- **Steps.** Halve `duration / steps` and compare the time constant: with
  TR-BDF2 the change falls about four times per halving.
- **The final value.** Run long after a step on and compare the trace's last
  value with a {py:obj}`gfem.SteadyThermal` solve of the same source.
- **Linearity.** Twice the power gives twice every rise, at every time.

## Limits

- The time grid is uniform; there is no adaptive stepping. A short pulse needs
  steps much shorter than it.
- Sources switch between constant levels at grid times; a ramp is a staircase.
- Materials are constant and linear. A gradient through a long run keeps or
  checkpoints the trajectory ({doc}`thermal-gradient`).

## Next

- {doc}`heater-temperature`: the steady temperature, boundaries and drives.
- {doc}`thermo-optic-phase`: phases on a section and along a route.
- {doc}`thermal-gradient`: the gradient of a trace or a time constant's
  inputs.
- {doc}`../tutorials/heater`: the step response of a phase shifter, against
  a published device.
