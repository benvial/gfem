# Dispersive materials

Give a region a permittivity that follows Drude, Lorentz, Debye or
Djordjevic-Sarkar terms, solve over a band, and differentiate through it. A
gold mirror at 1.55 µm, an FR-4 board from 1 to 20 GHz and a polar crystal
near its phonon resonance are each one {term}`dispersive material`: a
{py:obj}`gfem.Material` whose `eps_r` is a {term}`permittivity law`. Every
frequency of a sweep then sees the law's own permittivity there.

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

- **The terms.** {py:obj}`gfem.Drude` takes a `plasma_frequency` and a
  `damping`. {py:obj}`gfem.Lorentz` takes a strength `delta_eps`, a
  `resonance_frequency` and a `damping`. {py:obj}`gfem.Debye` takes
  `delta_eps` and a `relaxation_frequency`. {py:obj}`gfem.DjordjevicSarkar`
  takes `delta_eps` and two corner frequencies, `low_frequency` and
  `high_frequency`.
- **A datasheet.**
  {py:obj}`PermittivityLaw.djordjevic_sarkar <gfem.PermittivityLaw.djordjevic_sarkar>`
  fits `eps_inf` and one Djordjevic-Sarkar term to a permittivity and a loss
  tangent at one frequency. Its loss tangent is then nearly constant between
  the corners (1 kHz and 1 THz by default).
- **Passive by construction.** Strengths and frequencies must be positive
  and damping at least 0, so each term is passive alone. A negative strength
  raises `ValueError`: write a gain medium as a constant permittivity instead.
- **A published fit.** Most optics papers write their laws under
  $e^{-i\omega t}$. gfem uses $e^{+i\omega t}$, so its law is the complex conjugate: keep the
  parameter values and read the paper's `−i γ` as `+i γ`
  ({py:obj}`gfem.conventions`). A fit in eV converts as $f = E / h$, and a
  relaxation time $\tau$ as $f = 1 / (2\pi \tau)$.
- **Loss beside a law.** `conductivity` adds to the law. A nonzero
  `tan_delta` beside a law raises `ValueError`: give a datasheet loss tangent
  through `djordjevic_sarkar`, or add damping to a term.
- **Isotropic only.** A law is one value for every direction, beside one
  `mu_r` and one `conductivity`. A law with `axes`, or beside a tensor
  `mu_r` or `conductivity`, raises `ValueError`
  ({doc}`anisotropic-materials`).

## Take a law from cited data

A silicon nitride guide in oxide, with gold electrodes, from published
optical constants. Each {term}`library material` in {py:mod}`gfem.materials`
is a function that returns a plain {py:obj}`gfem.Material`, built from one
cited source:

```python
from scipy.constants import c

from gfem import materials

core = materials.silicon_nitride()
cladding = materials.silica()
electrode = materials.gold()
oxide_1550 = materials.silica(frequency=c / 1.55e-6)  # a constant, at 1.55 µm
```

- **Sources.** The {py:mod}`gfem.materials` page lists each material's
  source, specimen, accuracy and range. Each `name` carries a short source
  tag, `"silica (Malitson 1965)"`, which a saved job keeps. Cite the source
  paper when you publish.
- **The fitted range.** Each law holds over its {term}`fitted range`, the
  law's `frequency_range`. A solve that reads it outside warns
  `gfem.FittedRangeWarning` and runs on.
- **No static or RF values.** Silica's law gives $\varepsilon_r(0) = 3.00$
  and silicon nitride's 265, far from real films, which also vary by
  process. An electrostatic or RF model takes a constant from your own
  process: `gfem.Material(eps_r=<your oxide's static permittivity>)`, for
  example 3.9 for a thermal oxide.
- **A constant at one frequency.** `frequency=` gives the dielectrics'
  $\mathrm{Re}\,\varepsilon_r$ there as a constant, for the electro-optic
  and thermo-optic shifts, which take no law. It holds at that frequency
  only (silica at 1.55 µm gives 2.085) and carries no fitted range, so later
  solves read it without a warning.
- **Lithium niobate.** {py:obj}`gfem.materials.lithium_niobate` builds the
  crystal at one frequency on the principal axes you give, with its clamped
  electro-optic tensor, and {py:obj}`gfem.materials.lithium_niobate_static`
  gives its static permittivity for the electrostatic model
  ({doc}`electro-optic-phase`).
- **Your own data.** Fit a measured $n$, $k$ table with
  {py:obj}`gfem.fit_permittivity` ({doc}`fit-measured-data`).

## A board over a wide band

A microstrip filter on the FR-4 board, swept from 1 to 20 GHz with the
adaptive sweep. The mesh `microstrip.msh` holds the substrate, a design
region beside the trace and a lumped port at each end:

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

- The adaptive sweep needs no setting for a law. Its samples are frozen for
  the gradient, as for any adaptive sweep ({doc}`adaptive-sweep-gradient`).
- A lossless or weakly damped Lorentz resonance among the swept
  frequencies, such as one of a library or fitted metal law, costs samples:
  the sweep solves the two grid frequencies on either side of it first.
  Expect a few more full solves than for a constant material.
- A grid frequency exactly on a lossless resonance raises `ValueError`, since
  the permittivity is infinite there. Move the grid point, or give the term
  a damping.

## Metals in a topology optimization

To place a metal by gradients, blend it with a dielectric through
{term}`index interpolation`:

```python
density = jnp.asarray(start)  # one value per design element, 0 or 1 (a binary start)
cladding = gfem.Material(eps_r=1.444**2)
field = gfem.MaterialField.interpolate(density, cladding, gold, scheme="index")
blended = model.with_material("design", field)
```

- **Why the index.** A linear blend of the permittivity between a dielectric
  and a metal crosses $\operatorname{Re} \varepsilon \approx -2 \varepsilon_d$, with $\varepsilon_d$ the dielectric's
  permittivity. There a grey element is a small,
  almost lossless plasmonic resonator, and an optimizer finds such
  artificial resonances. `scheme="index"` blends $n - i\kappa$ instead, so a grey
  element absorbs.
- **Start from a binary design.** At a density of 0.5 everywhere, the
  blended square absorbs almost everything and its gradient vanishes. Start
  from a binary design close to what you want, such as a mirror.
- **The driven uniform sweep only.** A blend with a law end raises on the
  adaptive sweep and on the eigenmode, port mode and electrostatic problems.
  It also raises behind an `Absorbing` face or on a wave port's face. Solve
  it on a uniform list of frequencies, away from both.

The tutorial {doc}`../tutorials/metal-bend` optimizes a gold waveguide bend
this way over 1.5 to 1.6 µm.

## Which problems take a law

- **{py:obj}`gfem.Driven`**: uniform and adaptive sweeps, on every linear
  solver, with frequency workers, across MPI ranks and on plain and Bloch
  periodic cells. A metal ($\operatorname{Re} \varepsilon < 0$) is allowed. An `Absorbing` face on a law
  region needs $\operatorname{Re} \varepsilon > 0$ at every frequency of the grid. Where it is not, the
  solve raises `ValueError`: move the boundary into a dielectric.
- **{py:obj}`gfem.Eigenmode`**: modes nearest the target, as for a constant
  material, on one process and across MPI ranks. A target or a mode in a law region's {term}`unresolved band`, where
  its wavelength is shorter than two of its longest edges, raises
  `ValueError`. Move the target away from the resonance, or refine the mesh
  in that region. The mode energy weighs a law region with its
  {term}`energy permittivity`. A mode where it is not positive raises
  `ValueError`: model such a lossy metal as {py:obj}`gfem.Conductor` or
  {py:obj}`gfem.PEC`. An `Absorbing` face on a law region raises
  `NotImplementedError`; put a dielectric layer between them. An
  absorbing edge beside a law region in a wave port's cross-section raises
  `NotImplementedError` too: that edge takes $\operatorname{Re} \varepsilon$, which has no value at the
  complex eigenfrequency. Move the edge off the law's region.
- **{py:obj}`gfem.PortMode`** and wave ports: a law may sit in a port's
  cross-section (a {term}`dispersive cross-section`) when $\operatorname{Re} \varepsilon > 0$ there at
  every frequency. A metal in the cross-section raises `ValueError` before
  any solve. Keep the metal away from the port faces, or use lumped ports.
- **{py:obj}`gfem.Electrostatic`**: the law's {term}`static permittivity`,
  `law(0)`. A Drude term has none, and a Drude law raises `ValueError`:
  give the metal as a `PEC` terminal instead.
- **Measurements.** {py:obj}`gfem.electric_energy` weighs a law region with
  $\operatorname{Re} \mathrm{d}(\omega\varepsilon)/\mathrm{d}\omega$. {py:obj}`gfem.interface_participation` raises when a selected
  side lies in a law region ({doc}`measurements`).

## Gradients

The design inputs you already use differentiate on a model with a law: a
{py:obj}`gfem.MaterialField` in another region, the mesh nodes, a sheet's
fill and the lumped values. The law's own numbers are design
inputs too. Build the law inside the differentiated function, and scale each
frequency by a reference:

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

- The derivative by a frequency comes per hertz. Differentiated directly,
  it is about 1e-11 against order-one strengths, which stalls a step-size
  rule. Scaling by `f_ref`, or writing `jnp.exp(log_f)`, keeps every input of
  order one.
- `jax.grad` of an objective by a whole law gives a law of derivatives, one
  per number, each in that number's unit.
- Driven S and kept fields, and eigenfrequency and Q, differentiate by the
  law on uniform and adaptive sweeps, through wave ports, on a plain periodic
  cell and behind absorbing faces. The capacitance differentiates by
  `eps_inf` and the strengths, and its derivative by every frequency is an
  exact zero.

What raises, each with its remedy (the full list is on
{doc}`../capabilities`):

- An eigenmode's participations, `q_port` and mode field by a law's numbers
  raise `gfem.NonDifferentiableError`. Differentiate the eigenfrequency or
  Q instead.
- A law's numbers on a Bloch cell or a periodic eigenmode problem raise
  `gfem.NonDifferentiableError`. Differentiate on the plain periodic cell.
- A law region with elements on a wave port's face differentiates as
  elsewhere: the port mode reads the law and follows it (not under a Bloch
  wave vector).
- The ends of a blend are not design inputs: a law built inside the
  differentiated function and passed to `interpolate` raises
  `gfem.NonDifferentiableError`. Differentiate the law on a `Material` region.
- The port mode problem is not differentiable.

## Check your results

- **Against a constant material.** At one frequency where $\operatorname{Re} \varepsilon > 0$, a law
  region solves as a constant material set to the law's value there:

  ```python
  f0 = 5e9
  eps = complex(fr4.eps_r(f0))
  constant = gfem.Material(
      eps_r=eps.real, tan_delta=-eps.imag / eps.real, conductivity=fr4.conductivity
  )
  check = model.with_material("substrate", constant)
  # S of `check` at f0 equals S of `model` at f0 to the solver tolerance
  ```

- **Where the power goes.** Solve again with every damping set to 0. The drop
  in $1 - |S_{11}|^2 - |S_{21}|^2$ is what the law absorbed.
- **The gradient.** Compare it with a central difference along a random
  direction of the scaled inputs, with a relative step of about 0.1 %.

How gfem itself is checked: {doc}`../verification`.

## Limits

- **Only the permittivity disperses.** A law as `mu_r` raises `ValueError`,
  and a law cannot be anisotropic. Solve a dispersive permeability or a
  dispersive crystal one band at a time with constant values.
- **Measured n and k.** Fit them with {py:obj}`gfem.fit_permittivity`
  ({doc}`fit-measured-data`).
- **Outside the fitted range.** A library or fitted law is accurate within
  its {term}`fitted range` only. A solve outside it warns
  `gfem.FittedRangeWarning` and runs on, with an accuracy nothing states.
- **No metal in a port's cross-section.** Plasmonic port modes raise; end the
  guide in a dielectric or use lumped ports.
- **Eigenmodes near a resonance.** "Nearest the target" stops being well
  defined where a lossy region's {term}`modal overlap` is above about 10. The
  eigen solver's first pass then stops after 100 restarts and raises
  `gfem.EigenmodeConvergenceError`, naming the region and its term nearest
  the target. Move the target away from the resonance, or find the
  resonances from a `gfem.Driven` sweep.
- **Crowded lossless branches.** Close to a lossless resonance, the modes of
  a law region crowd, and a solve can return a mode one step farther than a
  nearer one. Ask for more modes than you need, and compare two targets.
- **The skin depth.** A metal's field decays within its skin depth, tens of
  nanometres at optical frequencies. Refine the mesh at the metal walls before
  trusting the second digit.
- **No law in a perfectly matched layer.** A layer region's background is a
  constant `Material`: give it the permittivity of the band, or keep the
  law region inside the air box ({doc}`perfectly-matched-layer`).

## Next

{doc}`../tutorials/metal-bend`, {py:obj}`gfem.PermittivityLaw`,
{doc}`fit-measured-data`, {py:mod}`gfem.materials`,
{py:obj}`gfem.conventions`, {doc}`../capabilities`,
{doc}`s-parameter-gradient`, {doc}`eigenmode-gradient`,
{doc}`adaptive-sweep-gradient`, {doc}`frequency-sweep`.
