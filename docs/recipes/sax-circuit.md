# SAX circuits with gfem blocks

Put a finite-element solve inside a circuit of analytic models, and take
`jax.grad` of the circuit with respect to the solve's design and to the
analytic models' parameters at once. A {py:obj}`gfem.SaxComponent` turns a
{py:obj}`gfem.Driven` solve into a model that a
[SAX](https://github.com/flaport/sax) netlist places like its own
(`pip install "gfem[sax]"`).

## A photonic circuit

In an unbalanced Mach-Zehnder interferometer each arm turns through the 2D
bend of the {doc}`../tutorials/bend`. How does the power at the bar output at
1.55 µm change with the bend's core pattern and the arms' length difference?
The component takes the bend as a function of its design, and a band:

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

- The settings are the keyword defaults of the model function, here `density`,
  plus `wl` in µm. `ports` maps the model's ports to the
  {term}`circuit port` names the netlist connects.

The arms' analytic guides take their indices from the bend's own port mode:
$n_g = n_\mathrm{eff} - \lambda\,\mathrm{d}n_\mathrm{eff}/\mathrm{d}\lambda$.

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
`jax.grad`, and pass designs as the settings of its instances:

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
(`g_density`) and the arms' length difference in µm (`g_dl`, a SAX input); both
bends share one solve and one backward pass.

### The polarity reference

Each exposed wave port needs a {term}`polarity reference` (a point and a
direction): a port mode's sign changes no power inside a block but swaps an
interferometer's bar and cross outputs around a loop. A lumped port or a wave
port with a `voltage_path` needs none. A photonic component hands SAX the
complex conjugate of gfem's S, since SAX's models go as $e^{-i\omega t}$
({doc}`../conventions`).

## The band and the circuit's wavelengths

The frequencies of the `Driven` template are the component's
{term}`component band`. Each evaluation solves the whole band for the design
and answers any wavelength inside it by a cubic interpolation in frequency,
with an error near $10^{-3}$ at 25 band frequencies per $2\pi$ of phase. A
spectrum then costs no solve:

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
  `jax.grad` the value and the gradient are NaN. The derivative with respect
  to `wl` is that of the interpolant.
- A 3D model can use `gfem.Driven(..., sweep="adaptive")` over hundreds of
  frequencies, run once at build ({doc}`adaptive-sweep-gradient`).

## An RF circuit

The ceramic section of {doc}`../get-started` sits in a 50 Ω system in series
with an ideal capacitor. How does its reflection at 5 GHz change with the
ceramic's permittivity and the capacitance?

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

The reflection grows by 0.20 per unit of permittivity and by
$1.6 \times 10^{10}$ per farad of capacitance.

- `f` (Hz) and `z0` (Ω, 50 by default) are settings; a new `z0` costs no
  solve. Set `z0` once for the whole circuit so SAX passes it to every model.
- A lumped port with only an inductance or a capacitance stays inside the
  device as an {term}`internal lumped load`: leave it out of `ports`.
- SAX's RF `coplanar_waveguide`, `microstrip` and `admittance` ignore `z0`;
  beside a gfem block use `transmission_line_s_params(..., z_ref=z0)` and
  `impedance(z=1 / y, z0=z0)`. gfem cannot check the netlist.

## What a circuit evaluation solves

An evaluation solves each component once per distinct design over its whole
band; instances with equal settings share a solve, and a fixed block costs
none. `cache=` passes gfem's disk solve cache ({doc}`save-and-export`) to every
solve. `jax.vmap` over a design setting runs its solves one after another.

## A solved result as a circuit block

A block that is solved once and never differentiated needs no component:
`result.s.to_sdict(circuit=..., ports=..., z0=...)` gives the dictionary for a
solved result, and a function that returns it is a model for SAX.

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

Every port pair is in the dictionary, zeros included, and every exposed port
must have been excited. `result.s.to_skrf()` is the same S as a scikit-rf
network ({doc}`save-and-export`).

## Check your results

Compare the arm-length gradient (SAX's alone) and the design gradient along a
random direction with central differences, calling `gfem.release_memory()`
before each evaluation ({doc}`s-parameter-gradient`).

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

The component at a band frequency is the solve's own S, and the outputs carry
no more power than goes in:

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

Between band frequencies a direct solve differs by the interpolation error. The
RF circuit's reflection matches scikit-rf's cascade:

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

- **Frequency.** Only the interpolant's derivative exists, not the solve's.
- **Design settings.** No `jax.jvp`; `jax.vmap` is sequential.
- **Circuit.** Build it outside `jax.jit`; RF components need a 3D model.
- **Port faces.** A design touching a wave port's face raises
  `gfem.PortGeometryError` at the first gradient.

## Next

{py:obj}`gfem.SaxComponent`, {py:obj}`gfem.NetworkParameters.to_sdict`,
{doc}`s-parameter-gradient`, {doc}`adaptive-sweep-gradient`,
{doc}`save-and-export`, {doc}`../conventions`, {term}`SAX component`.
