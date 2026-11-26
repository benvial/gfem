# Port-mode gradients

Tune a guide's cross-section for a target effective index or impedance, and
optimize a device whose design reaches its wave ports: a taper's width at its
port, a filling that runs into the feed lines. Under {py:obj}`jax.grad`, a
{py:obj}`gfem.PortMode` result's propagation constant $\gamma$ (so `n_eff`),
impedance $Z_{PV}$ and field differentiate with respect to the section's
materials, sheets and nodes, and so does a {py:obj}`gfem.Driven` solve whose
design reaches a wave port face.

## The effective index of a strip waveguide

How does the effective index of a 500 × 220 nm silicon strip's TE mode at
1.55 µm depend on the permittivities of the silicon and of the oxide
({doc}`thermo-optic-phase` has the window)?

```python
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/heater_w2.msh"
urllib.request.urlretrieve(url, "heater_w2.msh")
section = gfem.Mesh.from_gmsh_cross_section("heater_w2.msh", length_unit="um")
frequency = 299_792_458.0 / 1.55e-6  # Hz
outside = ["substrate", "box", "cladding", "heater", "neighbour", "neighbour_window"]


def strip(eps):
    """The strip's optical model, for the silicon and oxide permittivities."""
    materials = {
        "core": gfem.MaterialField(eps_r=eps[0], region="core"),
        "window": gfem.MaterialField(eps_r=eps[1], region="window"),
    }
    ports = [gfem.WavePort("core", "front")]
    return gfem.Model(section, materials=materials, ports=ports, excluded=outside)


def n_eff(eps, f=frequency):
    mode = gfem.solve(strip(eps), gfem.PortMode("core", f))
    return mode.n_eff[0].real


eps = jnp.array([3.4777**2, 1.444**2])  # silicon, oxide
value, grad = jax.value_and_grad(n_eff)(eps)
print(f"n_eff = {float(value):.6f}")
print(f"dn_eff/deps_r: silicon {float(grad[0]):.5f}, oxide {float(grad[1]):.5f}")
```

```text
n_eff = 2.447236
dn_eff/deps_r: silicon 0.15339, oxide 0.08275
```

- `n_eff` is $\gamma / (i k_0)$: its real part is the effective index, its
  imaginary part $-\alpha / k_0$ ({doc}`../conventions`).
- A gradient of $\gamma$ needs no extra solve: the backward pass is one sum
  over the section's elements per mode and frequency. A
  {py:obj}`gfem.MaterialField` on the window gives one derivative per element.
- The field differentiates too: an objective on `mode.e_field.at(points)`
  costs one 2D solve per mode and frequency. The gradient follows gfem's fixed
  phase ({py:obj}`gfem.PortModeResult`); a physical overlap is read as
  $|\cdot|^2$.

### Half the group index

Scaling every permittivity of the section by $s$ scales the frequency by
$\sqrt{s}$ (with $\mu_r = 1$, no frequency-dependent boundary), so

$$
\sum_r \varepsilon_r \frac{\partial n_\mathrm{eff}}{\partial \varepsilon_r}
= \frac{n_g}{2}, \qquad n_g = n_\mathrm{eff} + f\,\frac{\partial n_\mathrm{eff}}{\partial f},
$$

for the discrete mode as well. Check it with a group index from a frequency
difference:

```python
step = 1e-4
up, down = n_eff(eps, frequency * (1 + step)), n_eff(eps, frequency * (1 - step))
n_g = value + (up - down) / (2 * step)
print(f"sum eps dn_eff/deps = {float(eps @ grad):.6f}, n_g / 2 = {float(n_g) / 2:.6f}")
```

```text
sum eps dn_eff/deps = 2.027680, n_g / 2 = 2.027680
```

## Size a line for 50 Ω

The line of the {doc}`get started page <../get-started>` is two plates 1 mm
apart and 6 mm wide, between magnetic side walls `"side"`, with a wave port at
each end. Which width gives 50 Ω? The width moves the port faces, so both
ports are named in `sliding_ports`: their faces slide in their plane and each
port's mode follows.

```python
urllib.request.urlretrieve(url.replace("heater_w2.msh", "line.msh"), "line.msh")
mesh = gfem.Mesh.from_gmsh("line.msh", length_unit="mm")
air = gfem.Material()
line = gfem.Model(
    mesh,
    materials={"feed": air, "design": air},
    boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    ports=[
        gfem.WavePort("p1", "port1", voltage_path=[(0, 3e-3, 0), (0, 3e-3, 1e-3)]),
        gfem.WavePort(
            "p2", "port2", voltage_path=[(40e-3, 3e-3, 0), (40e-3, 3e-3, 1e-3)]
        ),
    ],
)
morph = gfem.shape.boundary_morph(line, "side", sliding_ports=("p1", "p2"))


def impedance(dw):
    """Z_PV of port p1 when the line widens by dw metres (dw / 2 per wall)."""
    moved = line.replace(mesh=morph(-dw / 2 * jnp.ones(morph.n_params)))
    return gfem.solve(moved, gfem.PortMode("p1", 5e9)).Z_PV[0].real


dw = 0.0
for _ in range(4):
    z, dz = jax.value_and_grad(impedance)(dw)
    print(f"w = {6 + 1e3 * dw:.5f} mm: Z_PV = {float(z):.5f} ohm")
    dw -= float((z - 50.0) / dz)  # a Newton step
print(f"w = {6 + 1e3 * dw:.5f} mm")
```

```text
w = 6.00000 mm: Z_PV = 62.78839 ohm
w = 7.22205 mm: Z_PV = 52.16393 ohm
w = 7.52164 mm: Z_PV = 50.08619 ohm
w = 7.53458 mm: Z_PV = 50.00015 ohm
w = 7.53461 mm
```

- Without `sliding_ports`, `boundary_morph` holds every port's face; a lumped
  or Floquet port's face is always held.
- The line's TEM mode has $Z_{PV} = \eta_0 h / w$ on any mesh: 50 Ω at
  $w$ = 7.5346 mm, the last width printed.
- The voltage path is a fixed set of points: keep the conductors it joins
  where its ends are (here the side walls move, the plates stay).

## A design that reaches the ports

Put the ceramic of {doc}`solve-a-device` back in the middle of the line and
make the feed lines' filling the design. It runs into both port faces. How
does the reflection into 50 Ω terminations depend on it?

```python
problem = gfem.Driven(frequency=np.linspace(2e9, 8e9, 4))


def reflection(eps_feed):
    """Mean |S11|^2 into 50 ohm terminations."""
    feed = gfem.MaterialField(eps_r=eps_feed, region="feed")
    model = line.replace(materials={"feed": feed, "design": gfem.Material(eps_r=10.0)})
    s = gfem.solve(model, problem).s.renormalize(50.0)
    return jnp.mean(jnp.abs(s["p1", "p1"]) ** 2)


value, grad = jax.value_and_grad(reflection)(1.0)
h = 1e-4
gfem.release_memory()
up = reflection(1.0 + h)
gfem.release_memory()
down = reflection(1.0 - h)
print(f"mean |S11|^2 at 50 ohm: {float(value):.6f}")
print(
    f"gradient {float(grad):.6f}, central difference {float((up - down) / (2 * h)):.6f}"
)
```

```text
mean |S11|^2 at 50 ohm: 0.625562
gradient -0.332319, central difference -0.332319
```

- S is normalized to each port's own mode, which follows the design:
  `s.z_ref` and `result.port_modes` move with the filling. S in the line's
  own pseudo-waves does not see a mismatch to the terminations, so renormalize
  to a fixed external line with `s.renormalize(50.0)`. For a guide with no
  impedance, keep a stretch of the external guide's section between the design
  and the port.
- A wave port's S entry, `s["out", "in"]`, is the projection of the field on
  the port's mode, exact on lossy lines too; on a
  {doc}`shared port surface <shared-port-surface>` each requested mode is a
  port of its own.
- The gradient adds a 2D pass on each wave port face at each frequency, only
  when the design reaches a face.

## Check your results

- Compare the gradient with a central difference along one direction, as
  above, calling {py:obj}`gfem.release_memory` before each evaluation
  ({doc}`s-parameter-gradient`).
- Between the frequencies a sweep solves exactly, the mode comes from a
  reduced model accurate to `SolverOptions(port_mode_tolerance=...)`; the
  gradient is that of the exact mode. For a difference over many frequencies,
  use `port_mode_tolerance=1e-12`.
- A node gradient is the exact derivative of the discrete objective
  ({doc}`shape-gradient`); how gfem itself is checked is in
  {doc}`../verification`.

## Limits

- **Eigenmode.** A design at a wave port face raises
  {py:obj}`gfem.PortGeometryError` in an {py:obj}`gfem.Eigenmode` solve (a
  driven solve differentiates it); so does a move of a face out of its plane,
  which `sliding_ports` otherwise allows within it.
- **Bloch and fixed-angle sweeps.** A design at a wave port face raises
  {py:obj}`gfem.NonDifferentiableError`.
- **Diagnostics.** `residual` raises {py:obj}`gfem.NonDifferentiableError`
  with a design at the face (wrap it in `jax.lax.stop_gradient`), and so does
  `mode.e_field.at(points)` for node coordinates, whose points stay put;
  {py:obj}`gfem.characteristic_impedance` is not differentiable, `Z_PV` along
  the port's `voltage_path` is.
- **Voltage path.** An end on a moving conductor, or a path along a moving
  interface, gives $Z_{PV}$ a kink (a few percent); place it inside elements,
  across the gap.
- **Fixed modes.** The thermo-optic and electro-optic shifts hold their mode
  constant: their gradient with respect to the optical model raises
  ({doc}`thermal-gradient`, {doc}`electro-optic-gradient`).

## Next

{py:obj}`gfem.PortMode`, {py:obj}`gfem.PortModeResult`,
{py:obj}`gfem.shape.boundary_morph`; {doc}`s-parameter-gradient`,
{doc}`shape-gradient`, {doc}`shared-port-surface`, {doc}`sheet-design`,
{doc}`../modes-and-ports`; {doc}`../capabilities`.
