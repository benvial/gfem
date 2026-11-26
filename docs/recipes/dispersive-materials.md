# Dispersive materials

Give a region a permittivity that follows Drude, Lorentz, Debye or
Djordjevic-Sarkar terms, solve over a band, and differentiate through it. A
{term}`dispersive material` is a {py:obj}`gfem.Material` whose `eps_r` is a
{term}`permittivity law`; every frequency of a sweep sees the law's own value.

## Write a law

A law is a high-frequency permittivity `eps_inf` plus one or more
{term}`dispersion terms <dispersion term>`. Every frequency is in Hz:

```python
import numpy as np

import gfem

# a gold-like Drude metal: plasma frequency 2.183e15 Hz (9.03 eV / h)
gold = gfem.Material(
    eps_r=gfem.PermittivityLaw(
        eps_inf=1.0,
        terms=[gfem.Drude(plasma_frequency=2.183e15, damping=1.28e13)],
    ),
    name="gold",
)

# a polar crystal: one lossy Lorentz resonance at 12 THz
crystal = gfem.Material(
    eps_r=gfem.PermittivityLaw(
        eps_inf=2.5,
        terms=[gfem.Lorentz(delta_eps=4.0, resonance_frequency=12e12, damping=0.1e12)],
    )
)

# water-like relaxation, and an FR-4 datasheet value: 4.3 and tan δ = 0.02 at 1 GHz
water = gfem.Material(
    eps_r=gfem.PermittivityLaw(
        eps_inf=5.2, terms=[gfem.Debye(delta_eps=73.0, relaxation_frequency=19e9)]
    )
)
fr4 = gfem.Material(
    eps_r=gfem.PermittivityLaw.djordjevic_sarkar(4.3, 0.02, frequency=1e9)
)
# the same law written out: eps_inf about 3.92 and one wideband term
written = gfem.PermittivityLaw(
    eps_inf=3.92,
    terms=[
        gfem.DjordjevicSarkar(delta_eps=1.135, low_frequency=1e3, high_frequency=1e12)
    ],
)

# a law is callable: the complex relative permittivity at f, Hz
f = np.linspace(1e9, 20e9, 200)
eps = fr4.eps_r(f)  # complex, Im eps <= 0
static = crystal.eps_r(0.0)  # the static permittivity, eps_inf + delta_eps
```

- **The terms** ({py:obj}`gfem.Drude`, {py:obj}`gfem.Lorentz`,
  {py:obj}`gfem.Debye`, {py:obj}`gfem.DjordjevicSarkar`) take the parameters
  their docstrings list. Strengths and frequencies must be positive: a gain
  medium is a constant permittivity.
- **A datasheet.**
  {py:obj}`PermittivityLaw.djordjevic_sarkar <gfem.PermittivityLaw.djordjevic_sarkar>`
  fits `eps_inf` and one term to a permittivity and a loss tangent at one
  frequency.
- **A published fit.** Optics papers use $e^{-i\omega t}$, gfem
  $e^{+i\omega t}$: read `−i γ` as `+i γ` ({py:obj}`gfem.conventions`). Convert
  eV as $f = E / h$ and a relaxation time as $f = 1 / (2\pi \tau)$.
- **Loss beside a law.** `conductivity` adds to it; a nonzero `tan_delta`
  raises `ValueError`.
- **Isotropic only** ({doc}`anisotropic-materials`): `axes` or a tensor `mu_r`
  or `conductivity` beside a law raises `ValueError`.

## Take a law from cited data

A silicon nitride guide in oxide, with gold electrodes. Each
{term}`library material` in {py:mod}`gfem.materials` returns a plain
{py:obj}`gfem.Material` built from one cited source:

```python
from scipy.constants import c

from gfem import materials

core = materials.silicon_nitride()
cladding = materials.silica()
electrode = materials.gold()
oxide_1550 = materials.silica(frequency=c / 1.55e-6)  # a constant, at 1.55 µm
```

- **Sources.** The {py:mod}`gfem.materials` page lists each source, specimen,
  accuracy and range. Cite the paper.
- **The fitted range.** A law holds over its {term}`fitted range`; a solve
  outside it warns `gfem.FittedRangeWarning` and runs on.
- **No static or RF values.** Silica's law gives $\varepsilon_r(0) = 3.00$,
  silicon nitride's 265: for electrostatic or RF models give a constant from
  your process.
- **A constant at one frequency.** `frequency=` gives
  $\mathrm{Re}\,\varepsilon_r$ there, for the electro-optic and thermo-optic
  shifts; it carries no fitted range.
- **Lithium niobate.** {py:obj}`gfem.materials.lithium_niobate` and
  {py:obj}`gfem.materials.lithium_niobate_static` ({doc}`electro-optic-phase`).
- **Your own data.** {py:obj}`gfem.fit_permittivity` ({doc}`fit-measured-data`).

## A board over a wide band

A microstrip filter on FR-4, swept from 1 to 20 GHz with the adaptive sweep.
`microstrip.msh` holds the substrate, a design region and a lumped port at each
end:

```python
import jax
import jax.numpy as jnp

mesh = gfem.Mesh.from_gmsh("microstrip.msh", length_unit="mm")
model = gfem.Model(
    mesh,
    materials={"substrate": fr4, "design": gfem.Material(eps_r=1.0)},
    boundaries=[gfem.PEC("ground"), gfem.PEC("trace"), gfem.Absorbing("outer")],
    ports=[gfem.LumpedPort("p1", "port1", "+z"), gfem.LumpedPort("p2", "port2", "+z")],
)
grid = np.linspace(1e9, 20e9, 300)
first = gfem.solve(model, gfem.Driven(frequency=grid, sweep="adaptive"))
problem = gfem.Driven(
    frequency=grid, sweep="adaptive", samples=first.sweep.sampled_frequencies
)


def passband_loss(eps):
    """Mean |S21|^2 deficit over the band, for a permittivity per design element."""
    m = model.with_material("design", gfem.MaterialField(eps_r=eps))
    s = gfem.solve(m, problem).s
    return jnp.mean(1 - jnp.abs(s["p2", "p1"]) ** 2)


n = len(mesh.elements("design"))
value, grad = jax.value_and_grad(passband_loss)(jnp.full(n, 2.0))
```

- The adaptive sweep needs no setting for a law
  ({doc}`adaptive-sweep-gradient`). A weakly damped resonance in the band, such
  as one of a library or fitted metal law, costs samples.
- A grid frequency exactly on a lossless resonance raises `ValueError`.

## Metals in a topology optimization

To place a metal by gradients, blend it with a dielectric through
{term}`index interpolation`:

```python
density = jnp.asarray(start)  # one value per design element, 0 or 1 (a binary start)
cladding = gfem.Material(eps_r=1.444**2)
field = gfem.MaterialField.interpolate(density, cladding, gold, scheme="index")
blended = model.with_material("design", field)
```

- **Why the index.** A linear blend crosses
  $\operatorname{Re} \varepsilon \approx -2 \varepsilon_d$, where a grey
  element is a lossless plasmonic resonator; `scheme="index"` blends
  $n - i\kappa$, so a grey element absorbs.
- **Start from a binary design**: at density 0.5 everywhere the gradient
  vanishes.
- **Driven uniform sweep only.** A blend with a law end raises elsewhere,
  behind an `Absorbing` face and on a wave port's face.

The tutorial {doc}`../tutorials/metal-bend` optimizes a gold bend this way.

## Which problems take a law

| Problem | Takes a law | Refuses, with the remedy |
| --- | --- | --- |
| {py:obj}`gfem.Driven` | both sweeps, metals included | `Absorbing` face on a law region where $\operatorname{Re} \varepsilon \le 0$: move it into a dielectric |
| {py:obj}`gfem.Eigenmode` | modes nearest the target | target in the {term}`unresolved band`: move or refine; a mode with non-positive {term}`energy permittivity`: use `gfem.Conductor`; `Absorbing` face beside a law: add a dielectric layer |
| {py:obj}`gfem.PortMode`, wave ports | a {term}`dispersive cross-section` with $\operatorname{Re} \varepsilon > 0$ | a metal in the section: use lumped ports |
| {py:obj}`gfem.Electrostatic` | the {term}`static permittivity` `law(0)` | a Drude law has none: use a `PEC` terminal |

{py:obj}`gfem.interface_participation` raises on a side in a law region
({doc}`measurements`).

## Gradients

Design inputs differentiate on a model with a law: a
{py:obj}`gfem.MaterialField` elsewhere, mesh nodes, sheet fills, lumped values,
and the law's own numbers. Build the law inside the differentiated function and
scale each frequency by a reference:

```python
f_ref = 12e12  # Hz


def loss(x):
    law = gfem.PermittivityLaw(
        eps_inf=2.5,
        terms=[
            gfem.Lorentz(
                delta_eps=x[0], resonance_frequency=f_ref * x[1], damping=0.1e12
            )
        ],
    )
    m = model.with_material("substrate", gfem.Material(eps_r=law))
    s = gfem.solve(m, gfem.Driven(frequency=grid)).s
    return jnp.mean(jnp.abs(s["p1", "p1"]) ** 2)


grad = jax.grad(loss)(jnp.array([4.0, 1.0]))  # by the strength and by f0 / f_ref
```

- The derivative by a frequency comes per hertz, about 1e-11 against order-one
  strengths: scale by `f_ref` or write `jnp.exp(log_f)`.
- Driven S, kept fields, eigenfrequency and Q differentiate by the law, on both
  sweeps.

What raises (the full list is on {doc}`../capabilities`):

- An eigenmode's participations, `q_port` and field by a law's numbers, and a
  law's numbers on a Bloch cell, raise `gfem.NonDifferentiableError`:
  differentiate the eigenfrequency, Q or the plain cell.
- So does a law built inside the function and passed to `interpolate`.

## Check your results

- **Against a constant material.** At one frequency where
  $\operatorname{Re} \varepsilon > 0$ a law region solves as a constant set to
  the law's value there:

  ```python
  f0 = 5e9
  eps = complex(fr4.eps_r(f0))
  constant = gfem.Material(
      eps_r=eps.real, tan_delta=-eps.imag / eps.real, conductivity=fr4.conductivity
  )
  check = model.with_material("substrate", constant)
  # S of `check` at f0 equals S of `model` at f0 to the solver tolerance
  ```

- **Where the power goes.** With every damping at 0, the drop in
  $1 - |S_{11}|^2 - |S_{21}|^2$ is what the law absorbed.
- **The gradient.** Compare with a central difference of the scaled inputs.

How gfem itself is checked: {doc}`../verification`.

## Limits

- **Only the permittivity disperses.** A law as `mu_r` raises `ValueError`.
- **Measured n and k.** Fit them with {py:obj}`gfem.fit_permittivity`
  ({doc}`fit-measured-data`).
- **Outside the {term}`fitted range`** a solve warns `gfem.FittedRangeWarning`
  with an accuracy nothing states.
- **Eigenmodes near a resonance.** Where a lossy region's {term}`modal overlap`
  is above about 10 the first pass stops after 100 restarts and raises
  `gfem.EigenmodeConvergenceError` naming the region and term: move the
  target or find resonances from a `gfem.Driven` sweep.
- **Crowded lossless branches.** A solve can return a mode one step farther
  than a nearer one: ask for more modes.
- **The skin depth** is tens of nanometres at optical frequencies: refine at
  the metal walls.
- **No law in a perfectly matched layer** ({doc}`perfectly-matched-layer`).

## Next

{doc}`../tutorials/metal-bend`, {py:obj}`gfem.PermittivityLaw`,
{doc}`fit-measured-data`, {py:mod}`gfem.materials`,
{py:obj}`gfem.conventions`, {doc}`../capabilities`,
{doc}`s-parameter-gradient`, {doc}`eigenmode-gradient`,
{doc}`adaptive-sweep-gradient`, {doc}`frequency-sweep`.
