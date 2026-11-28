# Second harmonic from a metasurface

Find how much second-harmonic light a patterned lithium niobate film sends into
each diffraction order, for a pump at 1550 nm at normal incidence: the pump is
solved once, its second-order polarization drives a second solve at 775 nm, and
the power in each order is read from the Floquet ports. The pump's field is not
reduced to a mode, so the pattern's near field counts
({doc}`second-harmonic-generation` does the same for a guide).

## The grating cell

A lamellar grating of x-cut lithium niobate on silica: period 0.9 µm along $x$
and 0.1 µm along $y$ (two cells), ridges 0.45 µm wide and 0.35 µm tall, 0.4 µm of
silica below and 0.5 µm of air above. The optic axis lies along $x$, so $d_{33}$
acts on a pump polarized across the ridges (TM). Each frequency has its own
model of the same mesh, with constant permittivities at that frequency. The
tensor and the indices are those of {doc}`second-harmonic-generation`.

At 1550 nm every order but the specular one is evanescent; at 775 nm the orders
$(\pm1, 0)$ propagate in air and in silica. The harmonic model finds them, and
both models take the same ports:

```python
import dataclasses
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

import gfem
from gfem import diffraction

c = scipy.constants.c  # m/s
url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/recipes/grating_cell.py"
urllib.request.urlretrieve(url, "grating_cell.py")
from grating_cell import grating_cell  # noqa: E402

d33, d31, d22 = 25e-12, 4.6e-12, 2.2e-12  # m/V, as in the guide recipe
rows = ((0, 0, 0, 0, d31, -d22), (-d22, d22, 0, d31, 0, 0), (d31, d31, d33, 0, 0, 0))
mesh = grating_cell("grating_cell.msh")  # period 0.9 um, 0.1 um deep


def silica_index(frequency):
    """Malitson's index of fused silica."""
    um2 = (c / frequency * 1e6) ** 2
    b, root = (0.6961663, 0.4079426, 0.8974794), (0.0684043, 0.1162414, 9.896161)
    return np.sqrt(1 + sum(bi * um2 / (um2 - ri**2) for bi, ri in zip(b, root)))


def cell(frequency, gap=1.0, tensor=rows, ports=()):
    """The cell at ``frequency``; ``gap`` is the permittivity between the ridges."""
    crystal = gfem.materials.lithium_niobate(
        frequency=frequency, axes=((0, 1, 0), (0, 0, 1), (1, 0, 0))
    )
    return gfem.Model(
        mesh,
        materials={
            "ridge": dataclasses.replace(crystal, second_order_nonlinear_tensor=tensor),
            "substrate": gfem.Material(eps_r=float(silica_index(frequency)) ** 2),
            "gap": gfem.MaterialField(eps_r=gap),
            "cover": gfem.Material(eps_r=1.0),
        },
        boundaries=[
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
        ports=list(ports),
    )


f0 = c / 1.55e-6
harmonic = cell(2 * f0)
ports = [
    *diffraction.floquet_ports(harmonic, "bottom", 2 * f0),
    *diffraction.floquet_ports(harmonic, "top", 2 * f0),
]
orders = sorted({p.name.split(" (")[1].split(")")[0] for p in ports})
print(len(ports), "ports, orders", orders)
```

```text
12 ports, orders ['-1, 0', '0, 0', '1, 0']
```

## The pump and the generated light

The pump is the TM wave of order $(0, 0)$ from the substrate at 1 GW/cm², which
is $1\,\mathrm{GW/cm^2} \times 0.9 \times 0.1\,\mu\mathrm{m}^2 = 0.9$ W per cell.
A `gfem.Pump` names the column and its power; the polarization it drives is a
source of the generated solve, which excites no port:

```python
options = gfem.SolverOptions(order=2, linear_solver="direct")
power = 1e13 * 0.9e-6 * 0.1e-6  # W per cell


def solve_pump(gap=1.0):
    """The pump's solve, for a permittivity ``gap`` between the ridges."""
    return gfem.solve(
        cell(f0, gap, ports=ports),
        gfem.Driven(f0, excite="bottom (0, 0) TM", keep_fields=True),
        options=options,
    )


def generate(pump, gap=1.0, tensor=rows):
    """The harmonic source and its solve, from a pump solved at the same ``gap``."""
    source = gfem.second_order_polarization(
        cell(f0, gap, tensor, ports),
        gfem.Pump(pump, f0, incident_power=power),
    )
    drive = gfem.Driven(2 * f0, excite=(), sources=source)
    return source, gfem.solve(cell(2 * f0, gap, ports=ports), drive, options=options)


pump = solve_pump()
source, generated = generate(pump)
print(source)
```

```text
PolarizationSource('second harmonic', frequency=3.868e+14, process='second-harmonic', regions=('ridge',), points=16128)
```

The source is a polarization on the ridge's quadrature points, not a
port: it is built from the measured pump field and enters the second solve as a
current $J = i\omega_3 P$ ({doc}`../nonlinear-and-quantum-optics`).

## The power in each order

`port_wave` of a port and the source's name is the generated wave there, in
$\sqrt{\mathrm W}$ per cell on a Floquet port. Add the TE and TM waves of an
order:

```python
def order_powers(result, name):
    """W per cell leaving each face, by order, TE and TM added."""
    return {
        face: {
            o: sum(
                jnp.abs(result.port_wave(f"{face} ({o}) {pol}", name)[0]) ** 2
                for pol in ("TE", "TM")
            )
            for o in orders
        }
        for face in ("top", "bottom")
    }


powers = order_powers(generated, source.name)
for face in ("top", "bottom"):
    row = ", ".join(f"({o}) {1e9 * powers[face][o]:.3g}" for o in orders)
    print(f"{face}: {row} nW")
up = sum(powers["top"].values())
print(f"upward {1e9 * up:.3g} nW of {power:.2g} W, eta/I = {up / power:.3g} cm^2/GW")
```

```text
top: (-1, 0) 74.6, (0, 0) 167, (1, 0) 71.5 nW
bottom: (-1, 0) 6.22, (0, 0) 1.49, (1, 0) 7.06 nW
upward 313 nW of 0.9 W, eta/I = 3.48e-07 cm^2/GW
```

- The harmonic power grows as $I^2$, so $\eta/I$, the upward power over the
  pump's power per cell, in cm²/GW for a pump of 1 GW/cm², is the figure to
  quote.
- The orders $(\pm1, 0)$ leave at 59° from the normal in air. Upward, the
  specular order carries half the power; downward, into the silica, the
  light is about a twentieth of that.
- $(+1, 0)$ and $(-1, 0)$ differ: the polar axis lies along the grating
  vector, so the mirror $x \to -x$ is not a symmetry of the crystal.

## Differentiate through both solves

The permittivity of the gaps between the ridges, one parameter set in the pump
model and in the harmonic model, changes the near field of both. `jax.grad`
runs the adjoint through the generated solve, the polarization and the pump's
solve; a central difference checks it:

```python
def upward(gap):
    src, result = generate(solve_pump(gap), gap)
    return sum(order_powers(result, src.name)["top"].values())


grad = jax.grad(upward)(1.0)
step = 1e-3
difference = (upward(1.0 + step) - upward(1.0 - step)) / (2 * step)
print(f"d(upward)/d(eps_gap) = {1e9 * grad:.4g} nW")
print(f"central difference agrees to 1e-5: {abs(difference / grad - 1) < 1e-5}")
```

```text
d(upward)/d(eps_gap) = 85.36 nW
central difference agrees to 1e-5: True
```

Raising the gaps' permittivity by 1 raises the upward power by about a quarter.

## Check your results

- **The orders account for the power the source emits.** The emitted power
  equals the sum over the ports, up to discretization:

  ```python
  total = sum(sum(face.values()) for face in powers.values())
  emitted = float(generated.emitted_power[source.name][0])
  print(f"emitted {1e9 * emitted:.4g} nW, orders {1e9 * total:.4g} nW")
  ```

  ```text
  emitted 327.8 nW, orders 327.8 nW
  ```

- **A tensor with $d_{33}$ alone gives equal $\pm1$ orders.** The mirror
  $x \to -x$ then maps the cell onto itself, with the sign of $d_{33}$ absorbed
  by the square of the pump, and the mesh is mirror-symmetric:

  ```python
  only33 = ((0,) * 6, (0,) * 6, (0, 0, d33, 0, 0, 0))
  src33, gen33 = generate(pump, tensor=only33)
  top33 = order_powers(gen33, src33.name)["top"]
  print(f"top (-1, 0) / (1, 0) = {top33['-1, 0'] / top33['1, 0']:.6f}")
  ```

  ```text
  top (-1, 0) / (1, 0) = 1.000000
  ```

- **A coarser mesh.** `grating_cell(..., h=0.07)` lowers the upward power by
  4 % and the gradient by 8 %: this mesh shows the method and is not converged.
  Refine with `h` and `gfem.SolverOptions(order=3)` before quoting a figure.

## Limits

- **Normal incidence.** An oblique pump gives the source a `wave_vector`, which
  the generated solve takes in `gfem.Driven`
  ({doc}`../nonlinear-and-quantum-optics`).
- **Undepleted pump,** at a single frequency of each solve; the harmonic does
  not act back on the pump.
- **Not a measured efficiency:** the tensor is the bulk value, with no surface
  contribution, and the mesh is coarse.
- **Gradients** reach materials, not node coordinates: remesh and difference.
- **No far field** for the generated column: read the Floquet ports.

## Next

{doc}`second-harmonic-generation`, {doc}`floquet-ports`,
{doc}`diffraction-efficiency`, {doc}`../conventions`,
{py:obj}`gfem.second_order_polarization`, {py:obj}`gfem.Pump`.
