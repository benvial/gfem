# SAX circuits with gfem blocks

Put a finite-element solve inside a circuit of analytic models, and take
`jax.grad` of the circuit with respect to the solve's design and to the
analytic models' parameters at once. A {py:obj}`gfem.SaxComponent` turns a
{py:obj}`gfem.Driven` solve into a model that a
[SAX](https://github.com/flaport/sax) netlist places like its own splitters,
waveguides and capacitors. It works for photonic circuits (wavelength in µm)
and RF circuits (frequency in Hz) alike. Install the extra with
`pip install "gfem[sax]"`; gfem itself never imports SAX, a component returns
a plain dictionary that SAX reads.

Two examples follow, both on the meshes of the tutorials: an unbalanced
Mach-Zehnder interferometer whose arms turn through a finite-element bend, and
the ceramic line section of {doc}`../get-started` placed behind a series
capacitor in a 50 Ω system.

## A photonic circuit

The arms of the interferometer each turn through the 2D waveguide bend of the
{doc}`../tutorials/bend` (core index 2.25 in a cladding of 1.444, $E_z$
polarization), whose design square holds a quarter ring of core. Ideal 50/50
couplers split and join the light, and analytic waveguides carry it between.
The question: how does the power at the bar output at 1.55 µm change with the
bend's core pattern and with the length difference of the arms?

The bend is a function of its design: the core fraction of each 0.05 µm pixel
of the design square, which `bend_at` lays into the model. The component takes
that function and the frequencies of the band it must answer in:

```python
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import sax
from scipy.constants import c

import gfem
from gfem import design

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/bend2d.msh"
)
urllib.request.urlretrieve(url, "bend2d.msh")
mesh = gfem.Mesh.from_gmsh("bend2d.msh", length_unit="um")

um = 1e-6
CLADDING, CORE = gfem.Material(eps_r=1.444**2), gfem.Material(eps_r=2.25**2)
bend = gfem.Model(
    mesh,
    materials={"cladding": CLADDING, "core": CORE, "design": CLADDING},
    boundaries=[gfem.Absorbing("outer", order=2)],
    ports=[  # E_z of the guide's even mode is positive at its centre, at both ends
        gfem.WavePort(
            "p1", "port1", pec_edges=("outer",), polarity=((-2.5 * um, 0.0), "+z")
        ),
        gfem.WavePort(
            "p2", "port2", pec_edges=("outer",), polarity=((0.0, -2.5 * um), "+z")
        ),
    ],
)
grid = design.LayerGrid(mesh, "design", pitch=0.05 * um)
x, y = np.meshgrid(grid.x, grid.y)  # pixel centres, m
r = np.hypot(x + 1.5 * um, y + 1.5 * um)
arc = ((abs(r - 1.5 * um) <= 0.275 * um) & (x > -1.5 * um) & (y > -1.5 * um)).astype(
    float
)


def bend_at(*, density=arc):
    """The bend with this core fraction (0 cladding, 1 core) per pixel of its square."""
    fill = gfem.MaterialField.interpolate(grid.to_elements(density), CLADDING, CORE)
    return bend.with_material("design", fill)


wl = np.linspace(1.50, 1.60, 11)  # um: the band the component answers in
block = gfem.SaxComponent(
    bend_at,
    gfem.Driven(frequency=c / (wl * um), polarization="Ez"),
    circuit="optical",  # wl in um, S in SAX's photonic phase convention
    ports={"p1": "o1", "p2": "o2"},
    options=gfem.SolverOptions(linear_solver="direct"),
)
print(list(block.settings))
```

```text
['density', 'wl']
```

- The component's settings are the keyword defaults of the model function,
  here `density`, plus `wl`, the circuit's wavelength in µm. The constructor
  solves once, at the defaults.
- `ports` maps the model's ports to the names the netlist connects: the
  {term}`circuit port` `o1` is the model's `p1`. A port left out still
  terminates the device as in a solve.
- SAX hands `density` over as a JAX array when it is set on an instance, and
  unchanged otherwise. `arc` is a pixel image, as in the bend tutorial.

The arms' straight guides are analytic. SAX's `straight` takes an effective
index and a group index, so take both from the bend's own port mode, the
index at three neighbouring wavelengths giving the group index
$n_g = n_\mathrm{eff} - \lambda\,\mathrm{d}n_\mathrm{eff}/\mathrm{d}\lambda$:

```python
modes = gfem.solve(
    bend,
    gfem.PortMode("p1", c / (np.array([1.54, 1.55, 1.56]) * um), polarization="Ez"),
)
n = np.real(np.asarray(modes.n_eff))
guide = {"wl0": 1.55, "neff": n[1], "ng": n[1] - 1.55 * (n[2] - n[0]) / 0.02}
print(f"n_eff {guide['neff']:.4f}, n_g {guide['ng']:.4f}")
```

```text
n_eff 2.0575, n_g 2.3076
```

The netlist joins the pieces. Build the circuit once, outside `jax.jit` and
`jax.grad`, and pass designs as the settings of its instances: SAX's default
backend cannot build a circuit inside `jax.jit`, and a circuit built once is
analysed once.

```python
mzi, _ = sax.circuit(
    netlist={
        "instances": {
            "split": "coupler",
            "top": "bend",
            "bottom": "bend",
            "top_wg": "guide",
            "bottom_wg": "guide",
            "join": "coupler",
        },
        "connections": {
            "split,out0": "top,o1",
            "top,o2": "top_wg,in0",
            "top_wg,out0": "join,in0",
            "split,out1": "bottom,o1",
            "bottom,o2": "bottom_wg,in0",
            "bottom_wg,out0": "join,in1",
        },
        "ports": {"in": "split,in0", "bar": "join,out0", "cross": "join,out1"},
    },
    models={
        "coupler": sax.models.coupler_ideal,
        "bend": block,
        "guide": sax.models.straight,
    },
)


def bar_power(density, dl):
    """Power at the bar output at 1.55 um; the bottom arm is dl um longer."""
    s = mzi(
        wl=1.55,
        top={"density": density},
        bottom={"density": density},
        top_wg={**guide, "length": 10.0},
        bottom_wg={**guide, "length": 10.0 + dl},
    )
    return jnp.abs(s["in", "bar"]) ** 2


power, (g_density, g_dl) = jax.value_and_grad(bar_power, argnums=(0, 1))(arc, 20.0)
print(f"{power:.4f}", g_density.shape, f"{g_dl:.4f}")
```

```text
0.6739 (60, 60) -0.8696
```

One `jax.grad` returns the derivative with respect to the bend's 3600 pixels
(`g_density`, a gfem input) and to the arms' length difference in µm (`g_dl`,
a SAX input). Both bends are fed the same design, so one solve and one
backward pass serve them. A 20 µm length difference gives a free spectral
range near 50 nm, so the bar output has a fringe across the band.

### The polarity reference

Each wave port a component exposes needs a {term}`polarity reference`. A
port mode is defined up to a sign, and the sign of the port's row and column
of S is otherwise set by the mesh (and in 3D by the model's orientation). Within
one block it changes no power. Around a loop of the netlist it swaps an
interferometer's bar and cross outputs, so it must be physical.

- On a 2D model the reference is a point and `"+z"` or `"-z"`: the field
  normal to the plane is real and positive at that point, at every frequency.
  Here the guide's even mode is positive at its centre.
- On a 3D model it is a point and an in-plane direction: the electric field
  along that direction is real and positive at the point.
- A lumped port, or a wave port with a `voltage_path`, needs none: its voltage
  fixes the sign. A port left out of `ports` needs none.

### Conventions of the photonic block

gfem's fields go as $e^{+i\omega t}$ and SAX's photonic models as
$e^{-i\omega t}$ (a straight guide of length $L$ is $e^{+ikL}$). A photonic
component therefore hands SAX $\overline{S}$, the complex conjugate of gfem's
S, with unit modal power on each wave port. The couplers are ideal and the
guides analytic with the port mode's index, so the example is a model of
the interferometer, not a layout. The conventions page lists the rules:
{doc}`../conventions`.

## The band and the circuit's wavelengths

The frequencies of the `Driven` template are the component's
{term}`component band`. At each evaluation the component solves the whole band
for the design, and answers any wavelength inside it: exactly at the band's
frequencies, and by a cubic interpolation in frequency between them. The
interpolation error is about $\theta^3/16$, with $\theta$ the largest phase
change of an entry of S between neighbouring band frequencies, so about 25
band frequencies per $2\pi$ of phase keep it near $10^{-3}$. The bend's 11
frequencies give steps of 0.29 rad.

A spectrum is then free: no solve runs while the design is unchanged.

```python
bar_spectrum = jax.jit(
    lambda w: (
        jnp.abs(
            mzi(
                wl=w,
                top={"density": arc},
                bottom={"density": arc},
                top_wg={**guide, "length": 10.0},
                bottom_wg={**guide, "length": 30.0},
            )["in", "bar"]
        )
        ** 2
    )
)
spectrum = bar_spectrum(np.linspace(1.50, 1.60, 1001))
print(spectrum.shape, f"{spectrum.min():.2f} to {spectrum.max():.2f}")
```

```text
(1001,) 0.00 to 0.70
```

- A wavelength outside the band raises `ValueError`; under `jax.jit` or
  `jax.grad`, where it may be unknown, the value and the gradient are NaN.
  Without `wl`, the component reads the band frequency nearest its middle.
- The derivative with respect to `wl` is that of the interpolant, not a group
  delay of the physics beyond the interpolation's accuracy.
- The template's frequencies may come in any order: the component sorts them.
- A long or resonant block wants a dense band. A 3D model can use
  `gfem.Driven(..., sweep="adaptive")` over hundreds of frequencies (2D
  models refuse the adaptive sweep): the component runs it once when it is
  built and keeps its samples for every later design, so a value never
  depends on whether it is differentiated ({doc}`adaptive-sweep-gradient`). To
  compare such a component's gradient with finite differences, build it with
  `options=gfem.SolverOptions(tolerance=1e-7, adaptive_tolerance=1e-6)`: the
  adaptive tolerance must be at least ten times `tolerance`, so a tighter sweep
  needs tighter solves too.

## An RF circuit

The RF example is the 20 mm ceramic section of {doc}`../get-started`, a
parallel-plate line fed by two lumped ports of $\eta_0 h / w \approx 62.8$ Ω,
placed in a 50 Ω system in series with an ideal capacitor. How does its
reflection at 5 GHz change with the ceramic's permittivity and the
capacitance?

```python
url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/line.msh"
urllib.request.urlretrieve(url, "line.msh")
line_mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")
z_line = 376.73 / 6  # ohm: the empty line's impedance, the ports' resistance
n_design = len(line_mesh.elements("design"))


def line_at(*, eps_r=10.0):
    """The line with its middle 20 mm filled with a ceramic of relative permittivity eps_r."""
    return gfem.Model(
        line_mesh,
        materials={
            "feed": gfem.Material(),
            "design": gfem.MaterialField(eps_r=eps_r * jnp.ones(n_design)),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
            gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
        ],
    )


section = gfem.SaxComponent(
    line_at,
    gfem.Driven(frequency=np.linspace(4.5e9, 5.5e9, 11)),  # Hz
    circuit="rf",  # f in Hz, S under e^{+i omega t}, renormalized to the setting z0
    ports={"p1": "o1", "p2": "o2"},
)
matched, _ = sax.circuit(
    netlist={
        "instances": {"dut": "section", "series": "capacitor"},
        "connections": {"dut,o2": "series,o1"},
        "ports": {"in": "dut,o1", "out": "series,o2"},
    },
    models={"section": section, "capacitor": sax.models.rf.capacitor},
)


def reflection(eps_r, capacitance):
    """|S11|^2 at 5 GHz, every model on one 50 ohm reference."""
    s = matched(
        f=5e9, z0=50.0, dut={"eps_r": eps_r}, series={"capacitance": capacitance}
    )
    return jnp.abs(s["in", "in"]) ** 2


value, (g_eps, g_cap) = jax.value_and_grad(reflection, argnums=(0, 1))(10.0, 2e-12)
print(f"{value:.4f} {g_eps:.4f} {g_cap:.4e}")
```

```text
0.0597 0.1966 1.5793e+10
```

The ceramic's permittivity is a scalar setting; the model function turns it
into the material field the solve needs. The reflection grows by 0.20 per unit
of permittivity and by $1.6 \times 10^{10}$ per farad of capacitance.

- `f` in Hz and `z0` in Ω are settings of the component, with `z0` 50 Ω by
  default. Set `z0` once for the whole circuit, as above, so SAX passes the
  same value to every model. The component renormalizes its ports from their own
  reference impedances to `z0` after the solve, so a new `z0` costs no solve.
- gfem, SAX's RF models and scikit-rf share $e^{+j\omega t}$, so S is not
  conjugated.
- An exposed lumped port needs a resistance (`resistance=` or `impedance=`).
  A lumped port with only an inductance or a capacitance stays inside the
  device as an {term}`internal lumped load`: leave it out of `ports`, and its
  value is still a design setting. An exposed wave port needs a
  `voltage_path`, which is also its polarity reference. A 2D model, or a
  Floquet port, cannot be an RF component. Each is refused when the component
  is built, naming the port and the remedy.
- The band is 4.5 to 5.5 GHz in 11 frequencies: 0.27 rad of phase between
  neighbours, so the interpolation between them stays near $10^{-3}$.
- The common reference is the one real impedance every RF model of the circuit
  refers its S to ({term}`common reference impedance`).

### SAX's RF models beside a gfem block

gfem cannot check any of the following, because it never sees the netlist.

1. Give every model one reference: pass `z0` to the circuit call, never per
   instance. A single mismatched instance gives a wrong circuit with no error.
2. `sax.models.rf.coplanar_waveguide` and `microstrip` are matched to their own
   characteristic impedance and take no `z0`, so a circuit-wide `z0` does not
   reach them. Beside a gfem block they are right only when that impedance
   equals `z0`. Write a small model around
   `sax.models.rf.transmission_line_s_params(gamma, z_c, length, z_ref=z0)`
   instead.
3. In released SAX, `sax.models.rf.admittance` has no `z0` and is normalized to
   0.5 Ω. Use `impedance(z=1 / y, z0=z0)`.
4. With SAX 0.16, the coplanar and microstrip helpers need the `jaxellip`
   package (`pip install "sax[rf]"`); the lumped models and
   `transmission_line_s_params` need nothing extra.

A quasi-TEM line's impedance has several definitions. To join a wave-port block
to an analytic line without a spurious reflection, build the line from the
block's own port mode: pass the `gamma` and `Z_PV` of a
`gfem.solve(model, gfem.PortMode(...))` result, interpolated on the band, to
`transmission_line_s_params`.

## What a circuit evaluation solves

- Each component solves once when it is built, at its defaults. An evaluation
  solves each component once per distinct design, over its whole band:
  instances with equal settings, or fed the same traced value in the same trace,
  share one solve,
  and a component whose settings do not change (a fixed block) costs no solve
  during an optimization. `cache=` passes gfem's disk solve cache
  ({doc}`save-and-export`) to every solve.
- Between evaluations gfem keeps the solver state (mesh, plan, factorizations)
  of up to four distinct meshes. A circuit whose components use more distinct
  meshes rebuilds some of that state at every evaluation and runs slower, with
  the same results. Instances of one component, and components built on equal
  meshes, share one.
- `jax.vmap` over a design setting runs its solves one after another; over `wl`
  or `f` it runs none. The in-process reuse of an optimization loop applies
  ({doc}`s-parameter-gradient`, "In an optimization loop").

## A solved result as a circuit block

A block that is solved once and never differentiated needs no component:
`result.s.to_sdict(circuit=..., ports=..., z0=...)` gives the same dictionary
for a solved result, and a model function that returns it is a model for SAX.

```python
result = gfem.solve(line_at(), gfem.Driven(frequency=5e9))
solved = result.s.to_sdict(circuit="rf", ports={"p1": "o1", "p2": "o2"}, z0=50.0)


def fixed_section(*, f=5e9):
    """The section solved above, on 50 ohm."""
    return {pair: values[0] for pair, values in solved.items()}


fixed, _ = sax.circuit(
    netlist={
        "instances": {"dut": "section", "series": "capacitor"},
        "connections": {"dut,o2": "series,o1"},
        "ports": {"in": "dut,o1", "out": "series,o2"},
    },
    models={"section": fixed_section, "capacitor": sax.models.rf.capacitor},
)
s_fixed = fixed(f=5e9, z0=50.0, series={"capacitance": 2e-12})
print(f"{abs(s_fixed['in', 'in']) ** 2:.4f}")
```

```text
0.0597
```

- Every port pair is in the dictionary, zeros included, because SAX reads a
  missing pair as zero. Every exposed port must have been excited.
- For RF the dictionary is always on one `z0`, 50 Ω if none is given.
- The same S as a scikit-rf network is `result.s.to_skrf()`:
  {doc}`save-and-export`.

## Check your results

The first two checks use what the page already holds. The gradient of the
interferometer's arm length is SAX's alone and needs no solve; compare it, and
the gradient with respect to the design along a random direction, with central
differences. `gfem.release_memory()` before each evaluation keeps one solve's
memory from piling on the last, as in {doc}`s-parameter-gradient`.

```python
step = 1e-3
fd_dl = (bar_power(arc, 20.0 + step) - bar_power(arc, 20.0 - step)) / (2 * step)
print(f"{g_dl:.4f} {fd_dl:.4f}")

mid = 0.2 + 0.6 * arc  # a design inside [0, 1], so both steps stay in range
direction = np.random.default_rng(0).normal(size=arc.shape)
direction /= np.linalg.norm(direction)
slope = jnp.vdot(jax.grad(bar_power)(mid, 20.0), direction)
h = 1e-3
values = []
for sign in (1, -1):
    gfem.release_memory()
    values.append(bar_power(mid + sign * h * direction, 20.0))
print(f"{slope:.3e} {(values[0] - values[1]) / (2 * h):.3e}")
```

```text
-0.8696 -0.8696
-1.813e-04 -1.813e-04
```

The component at a band frequency is the solve's own S, and the two outputs
together carry no more power than goes in, since the couplers are lossless and
the bends lose some:

```python
direct = gfem.solve(
    bend_at(), gfem.Driven(frequency=c / (1.55 * um), polarization="Ez")
).s.to_sdict(circuit="optical", ports={"p1": "o1", "p2": "o2"})
print(abs(block(wl=1.55)["o1", "o2"] - direct["o1", "o2"][0]) < 1e-10)

s = mzi(
    wl=1.55,
    top={"density": arc},
    bottom={"density": arc},
    top_wg={**guide, "length": 10.0},
    bottom_wg={**guide, "length": 30.0},
)
print(f"{abs(s['in', 'bar']) ** 2 + abs(s['in', 'cross']) ** 2:.4f}")
```

```text
True
0.6900
```

Between band frequencies, solve a few frequencies directly and compare: the
difference is the interpolation error above. For the RF circuit, the same
two-port in scikit-rf's cascade gives the same reflection:

```python
import skrf

freq = skrf.Frequency(5, 5, 1, unit="GHz")
network = gfem.solve(line_at(), gfem.Driven(frequency=5e9)).s.to_skrf(name="line")
capacitor = skrf.media.DefinedGammaZ0(freq, z0=50).capacitor(2e-12)
network.renormalize(50.0)  # in place, to one real reference
cascade = network**capacitor
print(f"{abs(cascade.s[0, 0, 0]) ** 2:.4f}")
```

```text
0.0597
```

How gfem itself is checked: {doc}`../verification`.

## Limits

- The frequency of a solve is not differentiable, only the interpolant's.
- No forward-mode derivative (`jax.jvp`) with respect to a design setting.
  `jax.vmap` over designs is sequential.
- Build the circuit outside `jax.jit`.
- A design that touches a wave port's face raises `gfem.PortGeometryError` at
  the first gradient. A fixed block is fine.
- RF components need a 3D model and one reference impedance for the whole
  netlist.
- Distributed (MPI) solves and Floquet ports inside a component are not tested.
- gfem has no group index on a port mode: take it from the effective index at
  neighbouring wavelengths, as above.

## Next

- {py:obj}`gfem.SaxComponent`, {py:obj}`gfem.NetworkParameters.to_sdict` and
  {py:obj}`gfem.WavePort` in the reference.
- {doc}`s-parameter-gradient`, {doc}`adaptive-sweep-gradient` and
  {doc}`save-and-export`.
- {doc}`../conventions` for what a circuit receives.
- [SAX](https://github.com/flaport/sax) for the analytic models.
- {term}`SAX component` in the glossary.
