# Heater phase gradients

Find how a heater's phase depends on everything that sets it: the drive
current, the heater's electrical conductivity, the thermal conductivity of
every element, the thermo-optic coefficient. One `jax.grad` call through the
chain {py:obj}`gfem.DCConduction` → {py:obj}`gfem.joule_heating` →
{py:obj}`gfem.SteadyThermal` → {py:obj}`gfem.thermo_optic_shift` gives all of
them, for one adjoint solve per forward solve, however many inputs there
are: the input of an optimizer that shapes a heater's materials, or of a
tolerance study.

The device is the heater section of {doc}`heater-temperature` and
{doc}`thermo-optic-phase`, 320 µm long, driven at 6.74 mA: close to a $\pi$
shift. The chain is linear in the rise, and the rise in the heat, so each
input enters in a way that has a closed form to check the gradient against.
Units are SI: the gradient of a phase in rad with respect to a current in A
is in rad/A ({doc}`../conventions`).

## The phase as a function of its inputs

Per-element values go in a {py:obj}`gfem.MaterialField`, one value per
element of a region, in the order of `mesh.elements(region)`. The thermal
model carries them; the optical model, with only the core and its window in
its optical solve, carries the permittivity and the thermo-optic
coefficient, scaled here by one number:

```python
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/heater_w2.msh"
urllib.request.urlretrieve(url, "heater_w2.msh")
section = gfem.Mesh.from_gmsh_cross_section("heater_w2.msh", length_unit="um")

k_bulk = {  # W/m/K
    "substrate": 148.0,
    "box": 1.38,
    "cladding": 1.38,
    "window": 1.38,
    "neighbour_window": 1.38,
    "core": 90.0,
    "neighbour": 90.0,
    "heater": 28.0,
}
k0 = {r: jnp.full(len(section.elements(r)), k) for r, k in k_bulk.items()}
boundaries = [
    gfem.FixedTemperature("bottom", temperature=300.0),
    gfem.Convection("top", coefficient=10.0, ambient=300.0),
]
frequency = 299_792_458.0 / 1.55e-6  # Hz
length = 320e-6  # m


def field(region, **values):
    """A material field on one region: each value spread over its elements."""
    n = len(section.elements(region))
    return gfem.MaterialField(**{key: v * jnp.ones(n) for key, v in values.items()})


def phase(current, sigma, k, dn_dt_scale):
    """The core's thermo-optic phase over the heater's length, rad."""
    materials = {r: gfem.MaterialField(thermal_conductivity=k[r]) for r in k}
    materials["heater"] = gfem.MaterialField(
        thermal_conductivity=k["heater"],
        conductivity=sigma * jnp.ones_like(k["heater"]),
    )
    thermal = gfem.Model(section, materials=materials, boundaries=boundaries)
    optical = gfem.Model(
        section,
        materials={
            "window": field(
                "window", eps_r=1.444**2, thermo_optic_coefficient=1e-5 * dn_dt_scale
            ),
            "core": field(
                "core", eps_r=3.4777**2, thermo_optic_coefficient=1.86e-4 * dn_dt_scale
            ),
        },
        ports=[gfem.WavePort("core", "front")],
        excluded={"em": [r for r in k if r not in ("window", "core")]},
    )
    dc = gfem.solve(thermal, gfem.DCConduction({"heater": "front"}, ground="back"))
    heat = gfem.joule_heating(dc, current=current)
    steady = gfem.solve(thermal, gfem.SteadyThermal(heat))
    shift = gfem.thermo_optic_shift(
        optical, steady.rise, port="core", frequency=frequency
    )
    return shift.phase_per_length * length


inputs = (6.74e-3, 2.3e6, k0, 1.0)  # A, S/m, W/m/K per element, 1
gradient = jax.value_and_grad(phase, argnums=(0, 1, 2, 3))
value, (d_current, d_sigma, d_k, d_scale) = gradient(*inputs)
print(f"phase: {float(value):.4f} rad")
```

```text
phase: 3.1454 rad
```

- `jax.grad` differentiates with respect to any argument that reaches a
  material, a source or a drive; `argnums` picks them, and a mapping of
  arrays comes back as a mapping of arrays.
- The mode is solved once per call and held constant: the gradient goes
  through the weights of each element, not through the mode. So the optical
  model's permittivity cannot be an input (see "What raises").
- The heater is out of the optical solve: a model whose solved regions hold a
  traced electrical conductivity cannot give a mode.

## Check the gradient against closed forms

Each input enters the phase by a power law, so each gradient times its input
is a whole multiple of the phase:

- the Joule power is $I^2 R'$, so $I\, \partial\varphi/\partial I = 2\varphi$;
- at a fixed current the power is $I^2 / (\sigma A)$, so $\sigma\, \partial\varphi/\partial\sigma = -\varphi$;
- scaling every thermal conductivity by a factor divides the rise by it, so
  $\sum_e k_e\, \partial\varphi/\partial k_e = -\varphi$, apart from the convection on top, which does not
  scale;
- the phase is linear in $\mathrm{d}n/\mathrm{d}T$, so its scale has $\partial\varphi/\partial c = \varphi$.

```python
weighted = {r: float(jnp.sum(k0[r] * d_k[r])) for r in k0}
print(f"I dphi/dI / phi:         {float(6.74e-3 * d_current / value):.6f}")
print(f"sigma dphi/dsigma / phi: {float(2.3e6 * d_sigma / value):.6f}")
print(f"sum k dphi/dk / phi:     {sum(weighted.values()) / float(value):.6f}")
print(f"dphi/dc / phi:           {float(d_scale / value):.6f}")
for region in ("box", "cladding", "window", "heater", "core", "substrate"):
    print(f"  {region:>9}: {weighted[region] / float(value):+.4f}")
```

```text
I dphi/dI / phi:         2.000000
sigma dphi/dsigma / phi: -1.000000
sum k dphi/dk / phi:     -0.999979
dphi/dc / phi:           1.000000
        box: -0.4393
   cladding: -0.3006
     window: -0.2320
     heater: -0.0035
       core: -0.0002
  substrate: -0.0226
```

The sum over the thermal conductivities misses −1 by 2 × 10⁻⁵, of the order
of the share of the heat that leaves through the top: the convection
coefficient was not scaled. The split by region says where the phase is made: the oxide between
the heater and the core (the cladding and the window) and the buried oxide
under it hold almost all of the sensitivity; the silicon substrate, a
hundred times more conductive, a little; the heater and the core almost
none.

For any other objective, check against finite differences: perturb one
input, or a random direction of many, by a small step both ways and compare
the difference quotient with the gradient's projection.

## The heat through a surface as an objective

{py:obj}`gfem.heat_flow_per_length` ({py:obj}`gfem.heat_flow` on a 3D mesh)
is differentiable too. The {term}`heat flow` up through the top, under a
fixed 70 W/m in the heater, as a function of every element's thermal
conductivity, and the heat out of the heater itself:

```python
heater_heat = gfem.HeatSource.uniform("heater", power_per_length=70.0)  # W/m


def flow(k, side):
    """The heat leaving ``side``, W/m, for the conductivities ``k``."""
    materials = {r: gfem.MaterialField(thermal_conductivity=k[r]) for r in k}
    thermal = gfem.Model(section, materials=materials, boundaries=boundaries)
    steady = gfem.solve(thermal, gfem.SteadyThermal(heater_heat))
    return gfem.heat_flow_per_length(steady, side, model=thermal)


up, d_up = jax.value_and_grad(flow)(k0, "top")
out, d_out = jax.value_and_grad(flow)(k0, "heater")
scaled = sum(float(jnp.sum(k0[r] * d_up[r])) for r in k0)
step = 1e-3  # a relative change of every conductivity
fd = flow({r: k * (1 + step) for r, k in k0.items()}, "top")
fd = (fd - flow({r: k * (1 - step) for r, k in k0.items()}, "top")) / (2 * step)
print(f"up through the top: {float(up):.6e} W/m")
print(f"sum k d(up)/dk: {scaled:.6e} W/m, central difference {float(fd):.6e} W/m")
print(f"out of the heater: {float(out):.4f} W/m")
print(f"largest |d(out)/dk|: {max(float(jnp.max(jnp.abs(d_out[r]))) for r in k0)}")
```

```text
up through the top: 2.211500e-03 W/m
sum k d(up)/dk: -2.211421e-03 W/m, central difference -2.211423e-03 W/m
out of the heater: 70.0000 W/m
largest |d(out)/dk|: 0.0
```

- The flow up is the top's convection over the rise there, and scaling
  every conductivity by a factor divides the rise nearly by it: the
  gradient times the conductivities is nearly minus the flow, as the
  central difference of that scaling confirms.
- A side with no heat sink in it, no fixed-temperature node and no
  convection face, such as the heater, gives off exactly the heat it makes,
  whatever the conductivities: its gradient with respect to them is exactly
  zero. The heat's own amount, or the drive of `joule_heating`, still
  differentiates.
- The flow is read from the discrete heat balance, so it is exact for the
  solved temperature, and its gradient is the gradient of that number
  ({doc}`heater-temperature`, "Where the heat goes").
- In a transient run, the {term}`heat out` trace
  ({py:meth}`gfem.Trace.heat_out`) differentiates with respect to the same
  inputs, and the mass density and specific heat besides.

## What differentiates

- **Inputs**: a fixed temperature, a convection's coefficient and ambient and
  a heat flux (`FixedTemperature.temperature`, `Convection.coefficient`,
  `Convection.ambient`, `HeatFlux.flux`; the rise reads only the
  coefficient, so its gradient with respect to the others is exactly zero);
  per-element `thermal_conductivity`, `conductivity` (through
  the DC solve), `thermo_optic_coefficient`, and, in a transient solve,
  `mass_density` and `specific_heat`; the amount of a
  {py:meth}`gfem.HeatSource.uniform`; the drive of `joule_heating`
  (voltage, current or power); the scales of a transient's waveform. Through
  {py:obj}`gfem.electromagnetic_heating`, the driven solve's materials,
  sheets and drive ({doc}`electromagnetic-heating`, "Gradients").
- **Outputs**: the temperature, rise, heat and power of a steady solve, and
  its heat flow; the traces (heat out included) and kept fields of a
  transient one; the conductance, dissipation
  and potentials of a DC solve; the shift's `delta_n_eff` and
  `phase_per_length`; a route's phase.
- A steady solve's gradient with respect to `mass_density` and
  `specific_heat` is exactly zero.
- A transient gradient keeps the run's trajectory for the reverse sweep, or,
  when it does not fit in memory, checkpoints and recomputes it.

The full table, with how each mark is checked, is on {doc}`../capabilities`.

## What raises

Rather than return a wrong value, these raise:

- the optical model's `eps_r`, `tan_delta`, `mu_r`, `conductivity` and node
  coordinates under `thermo_optic_shift` or `route_phase`
  ({py:obj}`gfem.NonDifferentiableError`): the shift holds its mode constant
  (a {py:obj}`gfem.PortMode` result differentiates on its own, but not
  through the shift in this version);
- the node coordinates of a thermal or DC conduction model (a shape
  gradient);
- {py:func}`gfem.rise_time` and {py:func}`gfem.time_constant` of a traced
  trace: they are crossing times of a sampled curve; differentiate the trace
  itself, or a value of it, instead;
- `duration`, `steps`, the waveform's start times and `initial_temperature`:
  they are fixed numbers, and a traced one raises when the problem is built.

## Limits

- The gradient is of the first-order phase, with the mode held constant: it
  does not include how the mode itself moves with the permittivity.
- The materials are constant: there is no temperature-dependent property to
  differentiate through.
- MPI ranks and the GPU solver do not run thermal problems.

## Next

- {doc}`heater-temperature`, {doc}`thermo-optic-phase` and
  {doc}`thermal-step-response`: the solves this chain is made of.
- {doc}`electromagnetic-heating`: the gradient of a temperature heated by
  an electromagnetic loss, through the driven solve.
- {doc}`s-parameter-gradient`: the same pattern for an electromagnetic
  objective, with a design loop.
- {doc}`../capabilities`: every gradient gfem computes, and what it refuses.
