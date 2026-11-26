# SPICE and Verilog-A export

Hand a solved device to a circuit simulator, for a transient or an AC
analysis beside the rest of a circuit. gfem fits S with a
{term}`rational model`, makes it passive, and writes it as a SPICE subcircuit
or a Verilog-A module, which runs at any frequency and in time, where a
Touchstone file ({doc}`save-and-export`) holds S only where it was solved. The
device is the line of {doc}`solve-a-device`.

## Fit a rational model

Fit on a dense grid ({doc}`frequency-sweep` gives one for few solves); here
the {term}`harmonic grid` of {doc}`time-responses`.

```python
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

df = 50e6  # Hz
freqs = df * np.arange(1, 401)  # 50 MHz to 20 GHz
result = gfem.solve(model, gfem.Driven(frequency=freqs, sweep="adaptive"))
s = result.s

rational = s.fit_rational()  # passive, tolerance 1e-3
print(rational)
print(rational.z_ref)  # ohm: the reference S is fitted on
```

```text
RationalModel(ports=['p1', 'p2'], n_poles=24, 0.05-20 GHz, error=0.000449, passive=True)
[62.78833333 62.78833333]
```

- The model is poles shared by every element of S plus a constant,
  $S(f) = D + \sum_n R_n / (2\pi i f - p_n)$, with $p_n$ in rad/s,
  $\mathrm{Re}\,p_n < 0$, and phases $e^{+i\omega t}$
  ({doc}`../conventions`).
- The order rises 4 poles at a time until every element of S is fitted to
  `tolerance` (an RMS error, $10^{-3}$ by default), then poles that add less
  than a tenth of it are removed.
- `rational.error` is the largest RMS error over the elements of S after
  passivity enforcement; `rational.fit_error` is the same before it.
- `rational.poles`, `rational.residues` and `rational.d` are the model,
  `rational.frequency_range` its fitted band. The fit never solves again.

## The reference impedance

The model holds one real reference impedance per port, constant over
frequency and fixed at the fit: the netlist's ports see power waves on it.

- Lumped ports have a real, constant reference, their resistance: the fit
  keeps it, as above.
- A complex or frequency-dependent reference (the $Z_{PV}$ of a wave port on a
  lossy or dispersive line) raises `ValueError` asking for `z0=`:
  `s.fit_rational(z0=50.0)` renormalizes S to 50 Ω first.
- A network without reference impedances (wave ports without a
  `voltage_path`) needs `z0=` as a nominal label only: S is fitted unchanged
  and `rational.nominal_reference` is `True`.

## Write the netlist

```python
rational.to_spice("line.sp")
rational.to_verilog_a("line.va")
print(next(line for line in open("line.sp") if line.startswith(".subckt")).strip())
print(next(line for line in open("line.va") if line.startswith("module")).strip())
```

```text
.subckt line p1 p1_ref p2 p2_ref
module line(p1, p1_ref, p2, p2_ref);
```

- Each port is a pair of pins, `<port>` and `<port>_ref`, in port order; tie
  each `*_ref` pin to the port's return. Pin names are the port names in lower
  case, with characters outside `a-z`, `0-9` and `_` replaced by `_`; two
  ports with one name, or a pin named `gnd`, raise `ValueError`: rename the
  port.
- The subcircuit or module is named after the file, or `name=`. A model that
  is not passive is refused: its netlist could make a transient diverge.

A deck that sends a 50 ps edge into port 1 of the subcircuit, both ports
ending on 62.79 Ω, in ngspice:

```text
* line.sp driven through port 1
.include line.sp
xline in 0 out 0 line
vsrc src 0 pulse(0 1 0 50p)
rsrc src in 62.79
rload out 0 62.79
.tran 1p 5n
.end
```

The Verilog-A module compiles with OpenVAF-Reloaded:

```bash
openvaf-r line.va -o line.osdi
```

and ngspice loads the compiled file:

```text
.control
pre_osdi line.osdi
.endc
.model line_va line
nline in 0 out 0 line_va
```

The SPICE file uses only `R`, `C`, `V`, `G` and `F` elements; the Verilog-A
module integrates each pole's states with explicit `ddt` statements. Both are
checked in ngspice and VACASK, in a small common subset other simulators read.

## The step a transient must reproduce

`rational.step_response(time)` is the model's exact step response,
$D + \sum_n \frac{R_n}{p_n}\left(e^{p_n t} - 1\right)$ for $t \ge 0$. A
transient simulation of the netlist, driven by a step of incident wave, gives
the same curve. The windowed step of the data ({doc}`time-responses`) is
band-limited, so the two differ while the edges pass; after about 3 ns, once
the ceramic's echoes have died down, they agree to within the fit's error:

```python
step = s.step_response()  # from S, band-limited
later = step.time > 3e-9  # s
exact = rational.step_response(step.time[later])
gap = abs(np.asarray(exact.values) - np.asarray(step.values)[later]).max()
print(gap < rational.error)  # about 3.5e-4 here
```

```text
True
```

## Long lines

A line $W$ phase wraps long at the top of the band needs about 2.3 poles per
wrap when matched, 4.4 when reflections bounce on it. The default `max_poles`,
100, reaches about 40 wraps matched, 22 mismatched; beyond that the fit warns
with {py:obj}`gfem.RationalFitNotConvergedWarning` and returns its most
accurate model. Then:

- remove a known delay first, with `result.with_offsets(...)`
  ({py:obj}`gfem.DrivenResult.with_offsets`) moving a wave port's reference plane;
- fit a narrower band, or raise `max_poles=` (up to the frequencies minus one).

## Passivity

A passive device has the largest singular value of S at most 1 at every
frequency. A fit can break that, in a {term}`passivity violation`; by default
it removes every one, changing the residues by the least it can, and
`rational.passivity_violations` lists the intervals it had before (Hz).

- When removing them changes S by more than `tolerance`, the fit warns with
  {py:obj}`gfem.NonPassiveDataWarning`: the data are not passive on this
  reference (a plane moved past a lossy line, a nominal reference on a lossy
  mode, a coarse mesh).
- `s.fit_rational(enforce_passivity=False)` fits only, to look at the
  violations; such a model cannot be exported. Violations that remain raise
  {py:obj}`gfem.PassivityEnforcementError`.

## From a stored run

A {term}`result bundle` holds S, so a netlist of a stored run is
`gfem.load_result("result").s.fit_rational().to_spice("line.sp")`. gfem does
not store models; keep their netlists.

## Under `jax.grad`

The fit is not differentiable: under `jax.grad`, `jax.jit` or `jax.vmap` it
raises {py:obj}`gfem.NonDifferentiableError`. Fit outside the transformation,
or wrap S in `jax.lax.stop_gradient` first. To optimize a time-domain
quantity, use the time responses ({doc}`time-responses`).

## Check your results

- **The model against S.** `rational.network(s.frequency)` against `s.values`
  differs by `rational.error` (RMS); `rational.passive` is `True`.
- **The netlist against the model.** An AC analysis of the subcircuit gives
  $S_{ij} = V_i / \sqrt{Z_{0,i}} - \delta_{ij}$ for a source of $2\sqrt{Z_{0,j}}$
  behind $Z_{0,j}$; a transient gives `rational.step_response`.

How gfem itself is checked: {doc}`../verification`.

## Limits

- No term proportional to frequency: S tends to $D$ above the band.
- One constant real reference impedance per port.
- The passivity test's cost grows as $(2NP)^3$ for $N$ poles and $P$ ports.
- `rational.network` refuses frequencies above the fitted range; it accepts
  0 Hz, the limit $D - \sum_n R_n / p_n$.

## Next

- {doc}`time-responses` and {doc}`save-and-export`;
  {py:obj}`gfem.RationalModel`, {py:obj}`gfem.NetworkParameters.fit_rational`.
