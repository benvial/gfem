# TDR and time responses

What would a time-domain reflectometer show on your device: the reflected
step at a port (TDR), the impedance along the line behind it, the
transmitted step (TDT) and the impulse response? gfem reads them from S on a
{term}`harmonic grid` by an inverse Fourier transform, with no time-domain
solve. Each is a {term}`time response`, a dimensionless wave ratio over time
({doc}`../conventions`, "Time responses"). The device is the line of
{doc}`solve-a-device`: air, 20 mm of $\varepsilon_r = 10$ ceramic, air.

## A step and an impedance profile

Solve on the harmonic grid $f_k = k\,\Delta f$, $k = 1, \dots, N$: the
record lasts $1/\Delta f$ and the top frequency sets how sharp the edges are.

```python
import math
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
urllib.request.urlretrieve(url, "line.msh")
mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")

z_line = 376.73 / 6  # ohm: the 6 mm wide, 1 mm high parallel-plate line in air
plates = [gfem.PEC("bottom"), gfem.PEC("top")]
ports = [
    gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
    gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
]
ceramic = {"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)}
model = gfem.Model(mesh, materials=ceramic, boundaries=plates, ports=ports)

duration = 20e-9  # s: the record, 1 / df
f_max = 20e9  # Hz: the top of the grid
df = 1 / duration
n = math.ceil(f_max / df)
freqs = df * np.arange(1, n + 1)  # harmonic grid: k df, k = 1..n
result = gfem.solve(model, gfem.Driven(frequency=freqs, sweep="adaptive"))
s = result.s

step = s.step_response()  # TimeResponse, Kaiser window beta = 6
z = step.impedance("p1")  # impedance profile seen from port 1, ohm
ceramic_z, late_z = np.interp([0.28e-9, 9e-9], step.time, z)
print(f"on the ceramic {ceramic_z:.2f} ohm, late {late_z:.2f} ohm")
print(f"by hand {z_line / np.sqrt(10):.2f} ohm and {z_line:.2f} ohm")
print(len(result.sweep.sampled_frequencies), "full solves")
```

```text
on the ceramic 19.86 ohm, late 62.78 ohm
by hand 19.86 ohm and 62.79 ohm
11 full solves
```

- `step.values[m, i, j]` is the wave leaving port `ports[i]` at time
  `step.time[m]` for a unit step of incident wave at port `excited[j]`, every
  other port ending on its reference impedance. `step["p1", "p1"]` is the
  reflected step $\rho(t)$ (TDR).
- The {term}`impedance profile` is
  $Z(t) = Z_0 \frac{1 + \rho(t)}{1 - \rho(t)}$, with $Z_0$ the port's
  `s.z_ref`, 62.79 Ω here. The ceramic reads
  $Z_0/\sqrt{10}$ from the round trip to its front face,
  $2 \times 10\ \mathrm{mm}/c = 66.7$ ps, to the round trip to its back
  face, 489 ps; after that the far air line, ended on port 2's 62.79 Ω.
- `step.time` is the whole two-sided record, $-T/2$ to $T/2$ with
  $T = 1/\Delta f$: slice it for a plot.
- The adaptive sweep solves 11 of the 400 frequencies ({doc}`frequency-sweep`).
- `step.to_xarray()` labels the responses by time and port name.

## Choose the grid

- **Record.** The record $1/\Delta f$ must outlast the response: make it at
  least 5 times the time the device takes to settle: on this line, whose
  ceramic rings for about 3 ns, a 5 ns record left a 4 % error at DC.
- **Resolution.** The span and the window set the 10-90 % rise of an edge:

  | `kaiser_beta` | Rise time, about | Edges |
  | --- | --- | --- |
  | 0 (no window) | $0.45/f_\mathrm{max}$ | sharpest, with ringing (13 dB sidelobes) |
  | 6 (default) | $0.99/f_\mathrm{max}$ | the usual compromise |
  | 13 | $1.44/f_\mathrm{max}$ | smoothest |

  `n_time=` only interpolates between samples.
- **First frequency.** Lumped ports solved through the tree-cotree gauge
  (`SolverOptions.gauge`) can start at $\Delta f$, however low. Wave ports lose
  digits below the mesh's low-frequency floor: choose $\Delta f$ at or above
  the floor {py:obj}`gfem.LowFrequencyWarning` names.
- **Long lines.** An adaptive sweep of a line $W$ phase wraps long needs about
  $W + 10$ samples. Above 10 wraps, raise `SolverOptions(adaptive_max_samples=...)`
  from 20, or the sweep stops with {py:obj}`gfem.SweepNotConvergedWarning`.

## The transmitted step and the rise time

`step["p2", "p1"]` is the TDT: the wave reaching port 2 for a unit step sent
into port 1. {py:obj}`gfem.rise_time` reads the 10-90 % time of one edge;
give it a slice around that edge.

```python
tdt = np.asarray(step["p2", "p1"])
print(f"the TDT settles at {tdt[-1]:.4f}")  # S21 at DC

edge = (step.time > 0) & (step.time < 0.3e-9)  # the first reflection only
rise = gfem.rise_time(step.time[edge], np.asarray(step["p1", "p1"])[edge])
print(f"rise {rise * 1e12:.1f} ps, 0.99 / f_max = {0.99 / f_max * 1e12:.1f} ps")
```

```text
the TDT settles at 1.0000
rise 49.2 ps, 0.99 / f_max = 49.5 ps
```

- The TDT is a ratio of power waves; with different reference impedances the
  voltage ratio is $S_{21}\sqrt{Z_{0,2}/Z_{0,1}}$.
- `rise_time` normalizes to the slice's first and last samples: on the whole
  TDT of an echoing line it gives a meaningless number, so slice around one
  edge.

## The impulse response

`s.impulse_response()` is the derivative of the step, scaled so that an
isolated reflection $\Gamma$ peaks at $\Gamma$. The front face of the
ceramic reflects $\Gamma = (Z_0/\sqrt{10} - Z_0)/(Z_0/\sqrt{10} + Z_0)$:

```python
impulse = s.impulse_response()
h = np.asarray(impulse["p1", "p1"])
i = int(np.argmin(h))
print(f"peak {h[i]:.3f} at {impulse.time[i] * 1e12:.1f} ps")
gamma = (1 / np.sqrt(10) - 1) / (1 / np.sqrt(10) + 1)
print(f"by hand {gamma:.3f} at {2 * 10e-3 / 299792458.0 * 1e12:.1f} ps")
```

```text
peak -0.519 at 65.9 ps
by hand -0.519 at 66.7 ps
```

- The impulse is dimensionless, like the step, which ends at $S(0)$.

## Data with the other sign of time

gfem's S follows $e^{+i\omega t}$, and the inverse transform has no
conjugation, so a delay is a positive time. Data written as $e^{-i\omega t}$
are the complex conjugate: conjugate them before building a
{py:obj}`gfem.NetworkParameters`, or every reflection lands at a negative time.

## Guides with a cutoff

A wave port on a waveguide has no S below its cutoff, so no harmonic grid
reaches DC. `s.impulse_response(mode="band_pass")` takes any evenly spaced grid
across the band and returns the magnitude $|h(t)|$, without sign. There is no
band-pass step: a step needs DC.

## S not on a harmonic grid

A log grid, or one that starts well above $\Delta f$, raises `ValueError`
naming the grid to solve on. To use such S anyway, resample it through a {term}`rational model`, explicitly:
`s.fit_rational().network(df * np.arange(0, n + 1)).step_response()`
({doc}`rational-model`); the fit has no gradient.

## Under `jax.grad`

The time views are written in `jax.numpy`: an objective on $Z(t)$ or on a
TDT differentiates through S ({doc}`adaptive-sweep-gradient` for an adaptive
sweep). When the reference impedance is traced too (a wave port's $Z_{PV}$),
pass `z0=`: inside `jax.grad` gfem cannot check that `z_ref` is real.

## Check your results

- **The late impedance is the far end's load.** $Z(t)$ near the end of the
  record is port 2's 62.79 Ω here.
- **The transmitted step ends at $S_{21}(0)$.** 1 for this lossless line.
- **A longer record changes nothing.** Halve $\Delta f$: a change in the
  plateaus means the record was too short.

How gfem itself is checked: {doc}`../verification`.

## Limits

- No gating and no peeling: the impedance profile is exact up to the first
  round trip past the first discontinuity; after that, multiple reflections
  mix in (a 30 Ω load behind a 75 Ω line, seen from a 50 Ω port, reads
  32.55 Ω).
- Anything later than $1/\Delta f$ aliases into the record, and DC comes from
  the first three frequencies unless S has a 0 Hz point: use a long record.
- `impedance` needs a step, an excited port and a real reference impedance; a
  wave port without a `voltage_path` has none.

## Next

- {doc}`rational-model`: the same S as a SPICE subcircuit, and its exact step.
- {doc}`frequency-sweep`: the adaptive sweep behind the dense grid.
- {py:obj}`gfem.TimeResponse` and {py:obj}`gfem.NetworkParameters`.
