# First device solve

Get the three answers gfem gives for one device: what it transmits between its
ports, where it resonates and how fast each resonance decays, and the
capacitance between its conductors. Each is a problem type solved on the same
{term}`model`: {py:obj}`gfem.Driven`, {py:obj}`gfem.Eigenmode` and
{py:obj}`gfem.Electrostatic`.

The device is the line of {doc}`../get-started`: two plates 40 mm long, 6 mm
wide and 1 mm apart, with ceramic of relative permittivity 10 over the middle
20 mm. Values are SI ({doc}`../conventions`).

## S-parameters over a band

How much power sent into port 1 reaches port 2 between 2 and 8 GHz, through a
{term}`lumped port` at each end, matched to the empty line?

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

result = gfem.solve(model, gfem.Driven(frequency=np.linspace(2e9, 8e9, 7)))
s = result.s
print(result.frequency / 1e9)  # GHz
print(abs(s["p2", "p1"]))  # |S21|
print(s.db("p2", "p1"))  # 20 log10 |S21|, dB
print(s.z_ref[0])  # ohm, the reference impedance of each port at 2 GHz
```

```text
[2. 3. 4. 5. 6. 7. 8.]
[0.8306221  0.68797765 0.64567246 0.90133223 0.57685516 0.97922016
 0.60517486]
[-1.61193031 -3.24851336 -3.79975481 -0.90230198 -4.7786644  -0.18239307
 -4.36238245]
[62.78833333 62.78833333]
```

- `frequency` is in Hz, one full solve each (hundreds: {doc}`frequency-sweep`).
  Every port is excited in turn unless you pass `excite="p1"`.
- `result.s["p2", "p1"]` is S21, one complex entry per frequency; `s.db(to,
  source)` is its dB and `s.values` the array, indexed frequency, output port,
  excited port.
- S is in {term}`pseudo-waves` normalized to `s.z_ref`, here each port's own
  resistance; `s.renormalize(50.0)` refers it to 50 Ω.
- The closed form for the 20 mm ceramic line agrees within 0.00002.

## Walls and metals

A surface the model does not name is a magnetic wall ({py:obj}`gfem.PMC` says
so in writing). With air everywhere the line is matched and $S_{21}$ is the
phase of 40 mm of free space, $\exp(-i k L)$:

```python
air = {"feed": gfem.Material(), "design": gfem.Material()}
at_5ghz = gfem.Driven(frequency=5e9, excite="p1")

walls = [*plates, gfem.PMC("side")]
unnamed = gfem.Model(mesh, materials=air, boundaries=plates, ports=ports)
named = gfem.Model(mesh, materials=air, boundaries=walls, ports=ports)
for empty_line in (unnamed, named):
    s21 = gfem.solve(empty_line, at_5ghz).s["p2", "p1"][0]
    print(abs(s21), np.angle(s21))  # |S21| and its phase, rad

k = 2 * np.pi * 5e9 / 299792458.0  # rad/m
print(np.angle(np.exp(-1j * k * 40e-3)))  # the phase of exp(-i k L), rad
```

```text
0.9999999272390561 2.091495024608073
0.9999999272390561 2.091495024608073
2.0914952632762236
```

The plates are perfect conductors ({py:obj}`gfem.PEC`). A real metal is a
{py:obj}`gfem.Conductor` (conductivity in S/m, optional thickness) and absorbs:
how much power is left after 40 mm with a copper top plate?

```python
copper = [gfem.PEC("bottom"), gfem.Conductor("top", 5.8e7)]
lossy_line = gfem.Model(mesh, materials=air, boundaries=copper, ports=ports)
s21 = gfem.solve(lossy_line, at_5ghz).s["p2", "p1"][0]
print(abs(s21) ** 2)  # fraction of the incident power that reaches port 2

r_s = np.sqrt(np.pi * 5e9 * 4e-7 * np.pi / 5.8e7)  # ohm: copper's surface resistance
alpha = r_s / (2 * 376.73 * 1e-3)  # Np/m: one lossy plate of a 1 mm high line
print(np.exp(-2 * alpha * 40e-3))  # the same fraction from the attenuation formula
```

```text
0.9980422163744983
0.9980431596715874
```

Other boundaries are a class and a surface name: {py:obj}`gfem.Impedance`,
{py:obj}`gfem.Absorbing` and {py:obj}`gfem.Periodic` ({doc}`periodic-cell`).

## Ports

A {term}`lumped port` ({py:obj}`gfem.LumpedPort`) is a resistor across a face
between two conductors; a {term}`wave port` ({py:obj}`gfem.WavePort`) feeds a
mode of an exterior face's cross-section ({doc}`shared-port-surface`). The
order of `ports` is the order of the rows and columns of `s.values`.

## Resonances and Q

With the ports gone both ends are open: a resonator. Where are its first three
resonances, and what Q does a {term}`loss tangent` of 1e-3 leave?

```python
lossy = {"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0, tan_delta=1e-3)}
resonator = gfem.Model(mesh, materials=lossy, boundaries=plates)

modes = gfem.solve(resonator, gfem.Eigenmode(target=3e9, n_modes=3))
print(modes.frequency / 1e9)  # GHz
print(modes.q)
p = modes.bulk_participation["design"]
print(p)  # share of each mode's electric energy in the ceramic
print(1 / (1e-3 * p))  # the Q that this share and the loss tangent give
print(modes.error.max() < 1e-9)
```

```text
[2.14197116 4.19062391 5.96307245]
[1251.68048682 1379.50404589 1748.29991075]
[0.79892612 0.72489834 0.57198431]
[1251.68018722 1379.50377406 1748.29969626]
True
```

- `target` (Hz) is where to search, `n_modes` how many modes to return,
  nearest first; `modes.q` comes from the complex `modes.omega`.
- `modes.bulk_participation` times the loss tangent adds to $1/Q$, the fourth
  printed line.
- The open-ended line resonates where $\tan(k a) \tan(n k a) = n$ (modes 1 and
  3) or $\tan(k a) = -n \tan(n k a)$ (mode 2), $a$ = 10 mm, $n = \sqrt{10}$:
  within 5 kHz of the printed values.

## Coupling Q of a port

How strongly do two 10 kΩ ports couple to the first two modes of the lossless
line?

```python
weak_ports = [
    gfem.LumpedPort("p1", "port1", "+z", impedance=1e4),
    gfem.LumpedPort("p2", "port2", "+z", impedance=1e4),
]
coupled = gfem.Model(mesh, materials=ceramic, boundaries=plates, ports=weak_ports)

modes = gfem.solve(coupled, gfem.Eigenmode(target=3e9, n_modes=2))
print(modes.q_port["p1"], modes.q_port["p2"])
print(modes.q)
print(1 / (1 / modes.q_port["p1"] + 1 / modes.q_port["p2"]))
```

```text
[665.27717644 792.96160529] [665.27717644 792.96160529]
[332.63896398 396.48111753]
[332.63858822 396.48080264]
```

`modes.q_port` holds one coupling Q per resistive lumped port. The device is
symmetric, so the total Q is the two in parallel, half of each: the last two
lines.

## A resonator loaded by a guide

Let a guide carry power away from one end: the air line continues past `port1`
without reflection, the far end stays open. Where are the resonances near
4.4 GHz, and what Q does the guide alone give?

```python
guide = [gfem.WavePort("p1", "port1")]
loaded = gfem.Model(mesh, materials=ceramic, boundaries=plates, ports=guide)

modes = gfem.solve(loaded, gfem.Eigenmode(target=4.4e9, n_modes=2))
print(modes.frequency / 1e9)  # GHz
print(modes.q)
print(modes.q_port["p1"])  # the guide's coupling Q
print(modes.requested_power_share["port1"])  # what the requested mode carries
```

```text
[4.4386642  6.40626756]
[10.05955574 17.40900762]
[10.04712206 17.40182595]
[1. 1.]
```

- The {term}`wave port` absorbs the guide's mode on the {term}`outgoing
  branch`: a {term}`quasi-normal mode`.
- `modes.q_port["p1"]` is the guide's {term}`coupling Q`;
  `modes.requested_power_share["port1"]` is the share of the port's absorbed
  power its requested mode carries, near 1 here. A low share makes the Q the
  port's, not the device's: the solve warns `gfem.UnrequestedPortModeWarning`
  below one half.
- Below the port's cutoff the solve raises `gfem.PortModeError`. The
  transmission-line roots, $4.43867 + 0.22089i$ and $6.40627 + 0.18407i$ GHz,
  are within 10 kHz of the printed frequencies.

## Capacitance

What is the capacitance between the plates? An electrostatic problem takes no
ports and no walls: it holds each terminal at a potential and grounds the rest.

```python
plates_only = gfem.Model(mesh, materials=ceramic)
problem = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
static = gfem.solve(plates_only, problem)
print(static.capacitance[0, 0] * 1e12)  # pF

eps_0 = 8.8541878128e-12  # F/m
print(eps_0 * (6e-3 / 1e-3) * (20e-3 + 10.0 * 20e-3) * 1e12)  # two plate capacitors, pF
```

```text
11.6875279208162
11.687527912896002
```

- `terminals` maps a name to the surfaces of one conductor, `ground` names
  those held at 0 V; `static.capacitance` is the Maxwell matrix in F.
- The hand value is $\varepsilon_0 (w / h) (L_\mathrm{air} + 10 L_\mathrm{ceramic})$:
  plate capacitors in parallel.

## Time each phase

gfem logs each phase's duration on the logger `"gfem.timing"`, silent by
default. This handler sums the seconds of each phase and keeps the unknowns:

```python
import collections
import logging


class PhaseTimes(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.seconds = collections.Counter()
        self.ndof = 0

    def emit(self, record):
        self.seconds[record.phase] += record.seconds
        if record.phase == "discretize":
            self.ndof += record.ndof


times = PhaseTimes()
timing = logging.getLogger("gfem.timing")
timing.addHandler(times)
timing.setLevel(logging.INFO)
gfem.solve(model, at_5ghz)
timing.removeHandler(times)
print(sorted(times.seconds), times.ndof)
```

```text
['assemble', 'discretize', 'setup', 'solve'] 24074
```

- The phases `discretize`, `assemble`, `setup` and `solve` do not overlap, so
  their seconds add up; records also carry `ndof`, `iterations`, `frequency`
  and `rank`. Under MPI sum `ndof` and take the slowest rank per phase.
- `gfem.log.enable(timing=True)` prints one line per record
  ({doc}`watch-a-solve`).

## Check your results

A lossless reciprocal device conserves power, $|S_{11}|^2 + |S_{21}|^2 = 1$,
and has a symmetric $S$; a resolved result does not move at a higher
{term}`element order`:

```python
s11, s21 = s["p1", "p1"], s["p2", "p1"]
print(abs(s11) ** 2 + abs(s21) ** 2)  # 1 for a lossless device
print(abs(s["p1", "p2"] - s21).max() < 1e-10)  # reciprocity

order_3 = gfem.SolverOptions(order=3)
finer = gfem.solve(model, at_5ghz, options=order_3)
print(abs(s21[3]), abs(finer.s["p2", "p1"][0]))  # |S21| at 5 GHz, orders 2 and 3
```

```text
[1. 1. 1. 1. 1. 1. 1.]
True
0.9013321719282609 0.9013341565407665
```

- The balance holds to 1e-6; a larger gap in a model you believe lossless
  points at an absorbing boundary or a wrong port.
- The orders differ in the sixth digit: the mesh resolves the line; if not,
  refine ({doc}`mesh-refinement`).
- A true resonance returns at the same frequency for another `target`.

## Limits

- **Low frequency.** Wave ports and iterative solvers lose digits below the
  floor and warn with {py:obj}`gfem.LowFrequencyWarning`; a device far smaller
  than the wavelength wants {py:obj}`gfem.Electrostatic`.
- **Scope.** Straight-sided tetrahedra, linear time-invariant materials, no
  time-domain solve ({doc}`../capabilities`).
- **Eigenmodes.** Only the `n_modes` nearest `target`: ask for more and check
  the farthest lies outside your band. A lossy region with about 10 modes per
  linewidth raises `gfem.EigenmodeConvergenceError`: sweep instead.

## Next

{py:obj}`gfem.SolverOptions`, {doc}`frequency-sweep`, {doc}`../modes-and-ports`, {doc}`measurements`,
{doc}`prepare-a-mesh`, {doc}`save-and-export`, {doc}`mesh-refinement`,
{doc}`s-parameter-gradient`, {doc}`../conventions`.
