# Port-mode gradients

Tune a guide's cross-section for a target effective index or impedance, and
optimize a device whose design reaches its wave ports: a taper's width at its
port, a filling that runs into the feed lines. Under {py:obj}`jax.grad`, a
{py:obj}`gfem.PortMode` result's propagation constant $\gamma$ (so
`n_eff`), its impedance $Z_{PV}$ and its field differentiate with respect to
everything the port's cross-section is made of: the materials of the regions
with elements on the port face (a {py:obj}`gfem.MaterialField` and a
permittivity law included), the fill of a sheet and the values of an
`Impedance` on an edge of the face, and the node coordinates of the face. A
{py:obj}`gfem.Driven` solve differentiates the same way when its design
reaches a wave port face: S, the port waves, kept fields,
`result.port_modes` and `s.z_ref` follow the port's mode as it changes.
Fields follow $e^{+i\omega t}$ and a port mode varies as $e^{-\gamma s}$
along its guide ({doc}`../conventions`).

## The effective index of a strip waveguide

How does the effective index of a 500 × 220 nm silicon strip's TE mode at
1.55 µm depend on the silicon and on the oxide around it? The section is the
optical window of {doc}`thermo-optic-phase`; the two permittivities are the
inputs.

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
  imaginary part $-\alpha / k_0$.
- A gradient of $\gamma$ alone (so of `n_eff`) needs no extra solve: the
  backward pass (the part of `jax.grad` that runs after the solve) is one
  sum over the section's elements per mode and frequency.
- A {py:obj}`gfem.MaterialField` on the window gives one derivative per
  element: where added index raises $n_\mathrm{eff}$ most.
- The mode's field differentiates too: `mode.e_field.at(points)` is
  linear in the field, and an objective on it costs one 2D solve per mode
  and frequency in the backward pass. gfem fixes the field's phase (see
  {py:obj}`gfem.PortModeResult`), and the gradient follows that phase
  exactly, but a physical overlap is read as $|\cdot|^2$.

### Half the group index

Scaling every permittivity of the section by $s$ is the same as scaling the
frequency by $\sqrt{s}$, when $\mu_r = 1$ and no boundary of the section
depends on frequency. The gradient therefore obeys

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

The line of the {doc}`get started page <../get-started>` is two plates 1 mm apart and 6 mm wide,
between magnetic side walls `"side"`, with a wave port at each end. Which
width gives 50 Ω? The width is a shape design that moves the port faces, so
both ports are named in `sliding_ports`: their faces slide in their plane,
and each port's mode follows the moved section.

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

- `sliding_ports` names the wave ports whose faces slide in their plane
  instead of being held; the port keeps its reference plane. Without it,
  `boundary_morph` holds every port's face, and a lumped or Floquet port's
  face is always held.
- The line's TEM mode has $Z_{PV} = \eta_0 h / w$, exactly on any mesh of
  the section: 50 Ω at $w$ = 7.5346 mm, the last width printed.
- The voltage path is a fixed set of points. Keep the conductors it joins
  where its ends are: here the side walls move and the plates stay, so the
  path still spans the gap.

## A design that reaches the ports

Put the ceramic of {doc}`solve-a-device` back in the middle of the line, and
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

- S is normalized to each port's own mode, which now follows the design:
  `s.z_ref` and `result.port_modes` move with the filling
  ($Z_{PV} = \eta_0 h / (w \sqrt{\varepsilon_r})$ here), and S in the line's
  own pseudo-waves does not see a mismatch to the terminations. Where a port
  meets a fixed external line, as here, renormalize S to it
  (`s.renormalize(50.0)`). For a guide with no impedance, keep a stretch of
  the external guide's own section between the design and the port, so the
  port's mode is that guide's.
- An objective on how much of an output field is in a port's mode is that
  port's S entry, `s["out", "in"]`: a wave port's S is the field's
  projection on the port's mode, exact on lossy lines too. On a
  {doc}`shared port surface <shared-port-surface>` each requested mode is a
  port of its own.
- The gradient costs what it costs with the design away from the ports,
  plus a 2D pass on each wave port face at each frequency (a small part of
  the 3D solve). That pass runs only when the design reaches a port face.

## Check your results

- Compare the gradient with a central difference of your objective along
  one direction, as above. Call {py:obj}`gfem.release_memory` before each
  evaluation ({doc}`s-parameter-gradient`).
- Between the frequencies a sweep solves exactly, a port mode comes from a
  reduced model accurate to `SolverOptions(port_mode_tolerance=...)`. The
  gradient is that of the exact mode. For a difference over many
  frequencies, solve with `port_mode_tolerance=1e-12`.
- Each moved mesh is a different discretization: a node gradient is the
  exact derivative of the discrete objective ({doc}`shape-gradient`).

How gfem itself is checked: {doc}`../verification`.

## Limits

- An {py:obj}`gfem.Eigenmode` solve with a design at a wave port face raises
  {py:obj}`gfem.PortGeometryError`; a driven solve differentiates it.
- Under a Bloch wave vector or on a fixed-angle sweep, a design at a wave
  port face raises {py:obj}`gfem.NonDifferentiableError`.
- `residual` (of a `PortMode` result or of `result.port_modes`) is the mode
  solve's backward error, a diagnostic: with a design at the face it raises
  {py:obj}`gfem.NonDifferentiableError`. Wrap it in `jax.lax.stop_gradient`.
- `mode.e_field.at(points)` raises {py:obj}`gfem.NonDifferentiableError`
  with respect to node coordinates (its points stay put while the section
  moves); `gamma`, `n_eff` and `Z_PV` differentiate.
- {py:obj}`gfem.characteristic_impedance` is not differentiable; `Z_PV`,
  along the port's own `voltage_path`, is.
- A voltage path whose end sits on a conductor that the design moves, or
  that runs along a material interface or mesh edge that moves, gives
  $Z_{PV}$ and `z_ref` a kink: the derivatives from the two sides differ
  (by a few percent along a moving interface). Place the path inside
  elements and across the gap, between conductors that stay.
- The thermo-optic and electro-optic shifts hold their mode constant: their
  gradient with respect to the optical model's permittivity or nodes raises
  ({doc}`thermal-gradient`, {doc}`electro-optic-gradient`).
- A port face stays planar: `sliding_ports` keeps it in its plane. A move
  out of the plane beyond the forward's tolerance raises
  {py:obj}`gfem.PortGeometryError` in the solve.

## Next

{py:obj}`gfem.PortMode`, {py:obj}`gfem.PortModeResult`,
{py:obj}`gfem.shape.boundary_morph`; {doc}`s-parameter-gradient`,
{doc}`shape-gradient`, {doc}`shared-port-surface`, {doc}`sheet-design`,
{doc}`../modes-and-ports`; {doc}`../capabilities`.
