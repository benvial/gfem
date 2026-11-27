# Length sweeps by mode cascade

How long should a multimode interference coupler be? A full solve per length
meshes and solves the whole device each time. The field in a uniform section
is a sum of its guide modes, each carried as $e^{-\gamma\ell}$ over a length
$\ell$, so only the discontinuities need the finite element method. You solve
the two ends of the coupler once, with
{py:obj}`gfem.WavePort.kept_modes` ports at planes inside the section, and
{py:obj}`gfem.cascade` joins them through a section of any length: a
{term}`mode cascade`. The length is a plain number, so a sweep is a
{py:obj}`jax.vmap` and the length has a gradient.

## Solve the two ends

The coupler is a 2 µm multimode section in a perfectly conducting box, with one
0.5 µm input guide and two output guides ({doc}`2d-models` has the 2D setup).
{download}`mmi_blocks.py <mmi_blocks.py>` meshes each end with its plane
`DEPTH` inside the section, and the whole device for the check. The planes
carry {term}`kept mode`s: the six propagating modes of the section and
{term}`evanescent port mode`s beyond them. Every port of a block is excited:

```python
import functools
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import c

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/mmi_blocks.py"
urllib.request.urlretrieve(url, "mmi_blocks.py")
from mmi_blocks import full_device, input_block, output_block  # noqa: E402

FREQUENCY = c / 1.55e-6  # Hz
DEPTH = 0.5  # um from each discontinuity to its plane
materials = {"core": gfem.Material(eps_r=2.0**2), "clad": gfem.Material(eps_r=1.444**2)}
options = gfem.SolverOptions(order=4, linear_solver="direct")


def solve(mesh, ports):
    model = gfem.Model(
        mesh, materials=materials, boundaries=[gfem.PEC("side")], ports=ports
    )
    return gfem.solve(model, gfem.Driven(FREQUENCY, polarization="Ez"), options=options)


@functools.cache
def ends(kept):
    """The input and output ends, each with ``kept`` modes at its plane."""
    first = solve(
        input_block(DEPTH),
        [gfem.WavePort("in1", "left"), *gfem.WavePort.kept_modes("mmi", "right", kept)],
    )
    second = solve(
        output_block(DEPTH),
        [
            *gfem.WavePort.kept_modes("mmi", "left", kept),
            gfem.WavePort("out1", "right", mode=1),  # the even pair of the two guides
            gfem.WavePort("out2", "right", mode=2),
        ],
    )
    return first, second


blocks = ends(12)
gamma = np.array([blocks[0].port_modes[f"mmi@{m}"].gamma[0] for m in range(1, 13)])
propagating = int(np.sum(np.abs(gamma.imag) > np.abs(gamma.real)))
print(f"{propagating} of 12 kept modes propagate")
print(f"first evanescent decay length {1e6 / gamma[propagating].real:.2f} um")
```

```text
6 of 12 kept modes propagate
first evanescent decay length 1.73 um
```

## Sweep the length

The section's length $\ell$ runs between the two planes, so for a coupler of
length $L$ it is $\ell = L - 2\,\mathrm{DEPTH}$. The power at the first output
is $T = |S_{\mathrm{out1,in1}}|^2$. A sweep of 1101 lengths is one
{py:obj}`jax.vmap`, and Newton steps on $\mathrm{d}T/\mathrm{d}L$ refine its
maximum:

```python
def transmission(length, blocks=blocks):
    """T at the first output for a coupler ``length`` um long."""
    s = gfem.cascade(blocks, sections={"mmi": (length - 2 * DEPTH) * 1e-6}).s
    return jnp.abs(s.values[0, s.ports.index("out1"), s.ports.index("in1")]) ** 2


lengths = jnp.linspace(1.0, 12.0, 1101)
power = jax.vmap(transmission)(lengths)
best = lengths[jnp.argmax(power)]
slope = jax.grad(transmission)
for _ in range(3):  # Newton steps on dT/dL
    best = best - slope(best) / jax.grad(slope)(best)
print(f"L* = {best:.3f} um, T(L*) = {transmission(best):.5f}")
table = jnp.array([1.5, 2, 3, 4, 5, 6, 8, 10])
for length, value in zip(table, jax.vmap(transmission)(table)):
    print(f"{length:5.1f} um  {value:.4f}")
```

```text
L* = 3.195 um, T(L*) = 0.95248
  1.5 um  0.6425
  2.0 um  0.6915
  3.0 um  0.9189
  4.0 um  0.7610
  5.0 um  0.7641
  6.0 um  0.4882
  8.0 um  0.5538
 10.0 um  0.8853
```

To plot the sweep:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots()
ax.plot(lengths, power)
ax.set_xlabel("coupler length (µm)")
ax.set_ylabel("$T$ at the first output")
fig.savefig("mmi_sweep.png", dpi=150)
```

## Kept modes

How many modes must the planes keep? Each dropped mode is a reflection the
joined blocks lose: it decays as $e^{-\alpha d}$ from the discontinuity to the
plane and back, so the error falls as $e^{-2\alpha_{N+1} d}$ with $\alpha_{N+1}$
the decay rate of the first mode dropped and $d$ the distance from the
discontinuity. Keep every propagating mode and the near-cutoff evanescent
ones. The boxed section has some propagating modes that the guides do not
carry: they are an artifact of the closed box. To check $N$, re-solve the
blocks at a larger $N$ and compare. Dropping modes from one solve looks
converged when it is not: the planes then hold only the modes they were
solved with. Against one full solve at $L^*$:

```python
full = solve(
    full_device(float(best)),
    [
        gfem.WavePort("in1", "left"),
        gfem.WavePort("out1", "right", mode=1),
        gfem.WavePort("out2", "right", mode=2),
    ],
)
names = ["in1", "out1", "out2"]


def external(s):
    index = [s.ports.index(name) for name in names]
    return np.asarray(s.values)[0][np.ix_(index, index)]


for kept in (7, 8, 10, 12):
    s = gfem.cascade(ends(kept), sections={"mmi": (best - 2 * DEPTH) * 1e-6}).s
    print(
        f"N = {kept:2d}  max |S - S_full| = {np.abs(external(s) - external(full.s)).max():.1e}"
    )
```

```text
N =  7  max |S - S_full| = 1.9e-04
N =  8  max |S - S_full| = 4.4e-06
N = 10  max |S - S_full| = 9.5e-08
N = 12  max |S - S_full| = 5.6e-08
```

From $N = 10$ the difference stops falling near $5\times10^{-8}$: the blocks and
the full device are meshed separately, so what remains is the difference
between the two meshes, not a dropped mode; it falls with the mesh size or the
element order.

## Differentiate in the length

`jax.grad` of $T$ in the length goes through the cascade's linear solve:

```python
slope_at_3 = jax.grad(transmission)(3.0)
step = 1e-3  # um
central = (transmission(3.0 + step) - transmission(3.0 - step)) / (2 * step)
print(f"dT/dL at 3 um: {slope_at_3:.5f} /um, central difference {central:.5f} /um")
```

```text
dT/dL at 3 um: 0.12934 /um, central difference 0.12934 /um
```

To differentiate in the blocks too, a permittivity or a shape, solve them
inside the traced function. Their port fields are then traced and the sign of
each mode cannot be read from them, so give every kept mode a `polarity` point
in `WavePort.kept_modes(..., polarity={m: (point, "+z")})`; without it
{py:obj}`gfem.NonDifferentiableError` says which mode.

## Check your results

The full solve at $L^*$ has the same power at the first output as the cascade;
the rest of the incident power is reflected or goes to box modes that neither
output port carries:

```python
t_cascade = float(transmission(best))
t_full = abs(np.asarray(full.s.values)[0, 1, 0]) ** 2
print(f"T cascade {t_cascade:.5f}, T full solve {t_full:.5f}")
```

```text
T cascade 0.95248, T full solve 0.95248
```

## Limits

- **Ports.** 2D line ports only; a 3D section raises.
- **Walls.** Closed or {py:obj}`gfem.Absorbing` sections; no cut by a perfectly
  matched layer.
- **Geometry.** No bends: a section is straight and uniform.
- **Frequencies.** All blocks share one frequency grid.
- **Evanescent rows.** An evanescent port mode's row of S is a pseudo-wave that
  carries no power; read S only at the external ports.

## Next

{py:obj}`gfem.cascade`, {py:obj}`gfem.CascadeResult`,
{py:obj}`gfem.WavePort.kept_modes`, {doc}`shared-port-surface`,
{doc}`port-mode-gradient`, {doc}`sax-circuit`.
