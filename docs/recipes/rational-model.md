# SPICE and Verilog-A export

Hand a solved device to a circuit simulator, for a transient or an AC
analysis beside the rest of a circuit. A Touchstone file holds S only at the
solved frequencies ({doc}`save-and-export`); a circuit simulator needs a
model it can run at any frequency and in time. gfem fits S with a
{term}`rational model`, poles shared by every element of S plus a constant,

$$
S(f) = D + \sum_n \frac{R_n}{2\pi i f - p_n},
$$

makes it passive, and writes it as a SPICE subcircuit or a Verilog-A module.
The poles $p_n$ are in rad/s with $\mathrm{Re}\,p_n < 0$, and phases follow
$e^{+i\omega t}$ ({doc}`../conventions`). The device is the line of
{doc}`solve-a-device`: 10 mm of air line, 20 mm of $\varepsilon_r = 10$
ceramic, 10 mm of air line.

## Fit a rational model

Fit on a dense grid: the fit sees only the frequencies it is given. An
adaptive sweep gives a dense grid for few solves ({doc}`frequency-sweep`).
The grid below is the {term}`harmonic grid` of {doc}`time-responses`, so the
same S also gives the time responses; any grid works for the fit.

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

- The order rises 4 poles at a time until every element of S is fitted to
  `tolerance` (an RMS error over the data, $10^{-3}$ by default), then the
  poles that add less than a tenth of it are removed.
- `rational.error` is that RMS error, the largest over the elements of S,
  after passivity enforcement; `rational.fit_error` is the same before it.
- `rational.poles` (rad/s), `rational.residues` and `rational.d` are the
  model itself, `rational.frequency_range` the band it was fitted on.
- The fit takes about a second here, on the host; it never solves again.

## The reference impedance

The model holds one real reference impedance per port, constant over
frequency, fixed at the fit: power waves on it are what the netlist's ports
see. It is chosen at the fit:

- Lumped ports have a real, constant reference, their resistance: the fit
  keeps it, as above.
- When a port's reference is complex or changes with frequency, as the
  $Z_{PV}$ of a wave port with a `voltage_path` does on a lossy or
  dispersive line, the fit raises `ValueError` and asks for `z0=`:
  `s.fit_rational(z0=50.0)` fits S renormalized to 50 Ω first. One value
  per port, in port order, also works.
- A network without reference impedances (wave ports without a
  `voltage_path`, S on unit modal power) needs `z0=` too, as a nominal label
  only: S is fitted unchanged, `rational.nominal_reference` is `True`, and the
  netlists say so in a comment.

The model's reference cannot be changed afterwards: to export on another
reference, fit again with `z0=`.

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

- Each port is a pair of pins, `<port>` and `<port>_ref`, in port order.
  Tie each `*_ref` pin to the port's return, `0` for ground. Pin names are
  the port names in lower case, with characters outside `a-z`, `0-9` and `_`
  replaced by `_`; two ports that end up with one name, or a pin named
  `gnd`, raise `ValueError`: rename the port.
- The subcircuit or module is named after the file, or `name=`.
- Comment lines at the top give the port names, the reference impedances,
  the fitted range, `error`, the gfem version and the time convention.
- A model that is not passive is refused: its netlist could make a transient
  simulation diverge.

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

The SPICE file uses only resistors, capacitors, 0 V sources and linear
controlled sources (`R`, `C`, `V`, `G`, `F`), with plain exponent numbers.
The Verilog-A module integrates each pole's states with explicit `ddt`
statements, with no Laplace filters, arrays or parameters. Both are checked
in ngspice and VACASK, and written in this small common subset so that other
SPICE and Verilog-A simulators read them too.

## The step a transient must reproduce

`rational.step_response(time)` is the model's exact step response, from its
poles and residues:
$D + \sum_n \frac{R_n}{p_n}\left(e^{p_n t} - 1\right)$ for $t \ge 0$. A
transient simulation of the netlist, driven by a step of incident wave,
gives the same curve. The windowed step of the data ({doc}`time-responses`)
is band-limited, so the two differ while the edges pass. Once the ceramic's
echoes have died down, after about 3 ns, they agree to within the fit's
error:

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

A line $W$ phase wraps long at the top of the band needs many poles: about
2.3 per wrap on a matched line, about 4.4 when reflections bounce on it,
since the round trip doubles the delay. The default `max_poles`, 100,
reaches about 40 wraps on a matched line, 22 on a mismatched one. Beyond
that the fit warns with {py:obj}`gfem.RationalFitNotConvergedWarning` and
returns its most accurate model. Then:

- remove a known delay first: move a wave port's reference plane along the
  line with `result.with_offsets(...)` ({py:obj}`gfem.DrivenResult.with_offsets`);
- fit a narrower band;
- raise `max_poles=`, up to the number of frequencies minus one.

The sweep must resolve the line too: an adaptive sweep of a line $W$ wraps
long needs about $W + 10$ samples, so raise
`gfem.SolverOptions(adaptive_max_samples=...)` above 10 wraps.

## Passivity

A passive device returns at most the power sent into it: the largest
singular value of S is at most 1 at every frequency. A fit can break that
between or beyond the data frequencies, in a {term}`passivity violation`,
and a non-passive model can make a transient simulation grow without
bound. By default the fit finds every violation and removes it, changing
the residues by the least it can. `rational.passivity_violations` lists the
intervals the fit had before (Hz), and `rational.passive` tests the model as
it is.

When removing the violations changes S by more than `tolerance`, the fit
warns with {py:obj}`gfem.NonPassiveDataWarning`: the data themselves are not
passive on this reference. Usual causes:

- a reference plane moved past the end of a lossy line;
- a nominal reference on a lossy mode;
- a mesh too coarse for the losses.

`s.fit_rational(enforce_passivity=False)` fits only, to look at the
violations; a model that is not passive cannot be exported. Violations that remain after
enforcement raise {py:obj}`gfem.PassivityEnforcementError`, naming them.

## From a stored run

A {term}`result bundle` holds S, so a netlist of a stored run is
`gfem.load_result("result").s.fit_rational().to_spice("line.sp")`. The fit
is deterministic and takes seconds, so gfem does not store models: keep a
model as its netlists, or as S on a grid of your choice,
`rational.network(f).to_touchstone("line_fit.s2p")`.

## Under `jax.grad`

The fit is not differentiable: it relocates poles iteratively and picks
its order in steps. Under `jax.grad`, `jax.jit` or `jax.vmap` it raises
{py:obj}`gfem.NonDifferentiableError`. Fit outside the transformation, or,
under `jax.grad`, wrap S in `jax.lax.stop_gradient` first: the model is then
a constant. To optimize a time-domain quantity, use
the time responses, which differentiate ({doc}`time-responses`).

## Check your results

- **The model against S.** `rational.network(s.frequency)` gives the
  model's S at the data frequencies. Compare it with `s.values`, element by
  element: `rational.error` is the RMS of that difference.
- **Passive.** `rational.passive` is `True` for every model the fit returns
  with `enforce_passivity=True`.
- **The netlist against the model.** An AC analysis of the subcircuit,
  driven by a unit incident wave at one port, gives the model's S at every
  frequency: $S_{ij} = V_i / \sqrt{Z_{0,i}} - \delta_{ij}$ with a source of
  $2\sqrt{Z_{0,j}}$ behind $Z_{0,j}$. A transient gives
  `rational.step_response`.

How gfem itself is checked: {doc}`../verification`.

## Limits

- No term proportional to frequency: S tends to $D$ above the band.
- One constant real reference impedance per port.
- The passivity test's cost grows as $(2NP)^3$ for $N$ poles and $P$ ports:
  seconds at 100 poles and 2 ports, more with many ports.
- `rational.network` refuses frequencies above the fitted range: a rational
  model is never extrapolated above its data. It accepts 0 Hz: the DC limit
  $D - \sum_n R_n / p_n$.
- All $P^2$ elements of S are fitted; a reciprocal device's model is as
  symmetric as its data.

## Next

- {doc}`time-responses`: TDR, TDT and the impedance profile of the same S.
- {doc}`save-and-export`: S as a Touchstone file, and result bundles.
- {py:obj}`gfem.RationalModel` and {py:obj}`gfem.NetworkParameters.fit_rational`
  in the API reference.
