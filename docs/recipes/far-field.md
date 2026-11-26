# Far fields and directivity

{py:func}`gfem.far_field` turns a solved field into its
{term}`far-field amplitude`: the field far from every source, per direction,
with the outgoing spherical factor removed. It works on any column of a
driven solve, an excited port (an antenna), a {term}`point dipole` or an
{term}`incident field` (a scatterer), and returns a {py:obj}`gfem.FarField`
with the antenna quantities of IEEE 145 built on it: the {term}`radiation
intensity`, the radiated power, the {term}`directivity`, gain and realized
gain, the radiation efficiency, the circular components and the axial
ratio, and a plane wave's bistatic radar cross section.

## Read the pattern of an antenna

A strip dipole two thirds of a wavelength long, fed by a lumped port in its
gap, in a box of air wrapped in a {term}`perfectly matched layer`, at a 1 µm
wavelength. The model's background is the {term}`exterior medium` the field
radiates into:

```python
import urllib.request

import numpy as np

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/antenna_box.msh"
urllib.request.urlretrieve(url, "antenna_box.msh")
mesh = gfem.Mesh.from_gmsh("antenna_box.msh", length_unit="um")

air = gfem.Material()
model = gfem.Model(
    mesh,
    materials={"left": air, "right": air, "pml": air},
    boundaries=[gfem.PEC("wire")],
    ports=[gfem.LumpedPort("p1", "feed", "+z", resistance=50.0)],
    pml=gfem.PML("pml"),
    background=air,
)
f0 = 299_792_458.0 / 1e-6  # Hz, a wavelength of 1 um
problem = gfem.Driven(f0, keep_fields=True)
result = gfem.solve(model, problem, options=gfem.SolverOptions(order=2))

# Gauss-Legendre in cos(theta) times a uniform phi: a quadrature of the sphere
x, w = np.polynomial.legendre.leggauss(16)
theta = np.arccos(x)[:, None]
phi = np.arange(32)[None, :] * np.pi / 16
weights = w[:, None] * np.full(32, np.pi / 16)[None, :]

ff = gfem.far_field(result, "p1", theta, phi, model=model)
print(f"P_rad = {ff.radiated_power[0]:.4f} W, P_acc = {ff.accepted_power[0]:.4f} W")
print(f"integral of U = {np.sum(ff.intensity[0] * weights):.4f} W")
print(f"peak directivity {ff.directivity[0].max():.3f}")
print(f"peak realized gain {ff.realized_gain[0].max():.3f}")
```

```text
P_rad = 0.6023 W, P_acc = 0.6229 W
integral of U = 0.6036 W
peak directivity 2.302
peak realized gain 1.386
```

- `theta` and `phi` are any arrays that broadcast together, in rad, with the
  angles of an incident field's direction: $\theta$ from $+z$, $\phi$ from
  $+x$. Every array of the result has the kept frequencies first, then their
  broadcast shape.
- `e_theta` and `e_phi` are $F_\theta$ and $F_\phi$ in V, with
  $\mathbf{E} \approx \mathbf{F}\,e^{-ikr}/r$ far away and the phase
  referred to the model origin (`gfem.conventions`, Far fields).
- `intensity` is $U = |\mathbf{F}|^2/(2\eta)$ in W/sr, and `radiated_power`
  the power flowing out of the side the far field is read on, from the
  residual of the discrete equations, as {py:func}`gfem.power_flow` computes
  it. The intensity integrated over the sphere agrees with it to the
  discretization; it is a check, not how $P_\text{rad}$ is computed.
- `directivity` is $4\pi U / P_\text{rad}$, `gain` $4\pi U / P_\text{acc}$,
  `realized_gain` $4\pi U / P_\text{inc}$ and `radiation_efficiency`
  $P_\text{rad}/P_\text{acc}$. A port is fed 1 W, so its realized gain is
  $4\pi U$ and its accepted power $1 - |S_{11}|^2$. The lumped port's $S$
  comes from its voltage and current: on this coarse mesh it meets the
  radiated power to 3 %, so the efficiency of this lossless strip reads
  0.97, not 1.
- For a point dipole, the accepted and incident powers are both its
  {term}`emitted power`: its gain equals its realized gain.

## Circular polarization

`e_right` and `e_left` are the components along right- and left-hand
circular polarization, $(F_\theta \pm iF_\phi)/\sqrt{2}$ (IEEE handedness,
as `"RCP"` and `"LCP"`), and `axial_ratio` is
$(|F_R| + |F_L|)/\big||F_R| - |F_L|\big|$: 1 for circular, infinite for
linear polarization. Two crossed dipoles in quadrature, $p(\hat x - i\hat y)$,
radiate right-hand circular polarization along $+z$:

```python
r0 = np.array([0.47, 0.53, 0.56]) * 1e-6  # m
pair = gfem.ElectricDipole("c", position=r0, moment=(1e-30, -1e-30j, 0))
free = model.replace(boundaries=[], ports=[])  # the strip removed
quadrature = gfem.solve(
    free,
    gfem.Driven(f0, sources=pair, keep_fields=True),
    options=gfem.SolverOptions(order=2),
)
up = gfem.far_field(quadrature, "c", 0.0, 0.0, model=free)
print(f"|F_L| / |F_R| = {abs(up.e_left[0]) / abs(up.e_right[0]):.3f}")
```

```text
|F_L| / |F_R| = 0.014
```

## Radar cross section

For a {py:obj}`gfem.PlaneWave` column the far field is the
{term}`scattered field`'s, and `rcs` is its bistatic radar cross section
$\sigma = 4\pi|\mathbf{F}|^2/|E_0|^2$ in m². The strip without its feed, lit
face-on along $+x$ with the field along it ($\hat{\boldsymbol{\theta}} = -\hat{\mathbf{z}}$):

```python
wave = gfem.PlaneWave("w", theta=np.pi / 2, phi=0.0, polarization="TM")
lit = model.replace(ports=[])
scattered = gfem.solve(
    lit,
    gfem.Driven(f0, excite=(), sources=wave, keep_fields=True),
    options=gfem.SolverOptions(order=2),
)
back = gfem.far_field(scattered, "w", np.pi / 2, np.pi, model=lit)
print(f"backscatter {back.rcs[0] / 1e-12:.4f} um^2")
```

```text
backscatter 0.9333 um^2
```

The forward amplitude gives the extinction by the optical theorem,
$\sigma_\text{ext} = -\tfrac{4\pi}{k}\operatorname{Im}\big(\hat{\mathbf{e}}^*
\cdot\mathbf{F}(\hat{\mathbf{k}})\big)/E_0$ for a wave of amplitude $E_0$
along $\hat{\mathbf{e}}$: a check of the extinction cross section.

## Where the far field is read

The amplitude is a reciprocity integral over one element layer, the
{term}`far-field layer`: with $\chi$ the side's P1 indicator and
$\mathbf{w}$ the plane wave arriving from $\hat{\mathbf{r}}$,

$$
\hat{\mathbf{q}}\cdot\mathbf{F}(\hat{\mathbf{r}}) = \frac{1}{4\pi}\int
\Big[\mathbf{E}\cdot(\nabla\chi\times\nabla\times\mathbf{w})
+ (\nabla\times\mathbf{E})\cdot(\nabla\chi\times\mathbf{w})\Big]\,dV,
$$

the Stratton-Chu surface integral averaged across the layer. It needs no
surface, and reads the field inside elements, where it is smooth: two to
five times more accurate than a surface integral at orders 2 and 3.

- By default (`side=None`) the side is every physical element with no
  vertex on the perfectly matched layer or an absorbing face, so the layer
  is the one touching the truncation, and no name is needed.
- `side=` takes volume region names or a predicate on element centroids, as
  for `power_flow`. The side must hold every source, port, sheet and
  scatterer, one element clear of its outside layer.
- The layer must be the exterior medium: one constant, isotropic, lossless
  material, the model's background.

## Ground planes

A flat PEC or PMC exterior face that the layer meets is a
{term}`ground plane`: the model ends on it instead of a layer, and the far
field is that of the sources and their mirror images, in the half-space
above it (image theory). gfem finds it from the model's walls; directions
below the plane raise. The same dipole $0.21\lambda$ above a PEC ground
gives, at order 3, the pattern of the dipole and its image to 0.2 %.

## Couple into a beam

The same reaction with a beam as the test field gives the
{term}`beam coupling`: {py:func}`gfem.beam_coupling` reads how much of a
solved field goes into an incoming beam, such as a fiber's above a grating
coupler. For a port column fed 1 W it is, by reciprocity, the wave the port
would receive if the beam lit the device, so the fiber-to-port efficiency
of every fiber position, angle, waist and polarization comes from one
port-fed solve: `abs(gfem.beam_coupling(result, "wg", fiber, model=model)) ** 2`.

- The beam is a `gfem.GaussianBeam`, or a `gfem.IncidentField` (a
  z-invariant fiber on a slab, say), described travelling toward the
  device. It is scaled to 1 W through the side, so neither its own power
  nor a slab's thickness enters; the phase is the beam's.
- By default the side's surface is the truncation face the beam brings most
  power in through: the physical elements touching it are the layer. A
  named side or a predicate works too, if its surface crosses the beam once
  (a half-space; through a closed surface a beam brings no net power) and
  stays an element away from the truncation under the beam. Perfectly
  matched layer elements are dropped from the layer, so a plane through
  the layer ends open there.
- The layer must be the beam's medium, one constant, isotropic, lossless
  material, and may end on absorbing faces or on PEC or PMC walls the beam
  satisfies (a TE beam on a slab's walls); otherwise `beam_coupling` raises
  a `ValueError`. A `GaussianBeam` seen with less than 0.999 W through the
  side, or any beam of which more than a thousandth crosses the truncation
  where the side's surface meets it, is clipped and warns.
- On a grating slab the coupling equals the beam-lit port wave to 0.14 %,
  and two planes in the cladding agree to $10^{-5}$.
- It is differentiable through the field with respect to material fields
  and the node coordinates. An `IncidentField` beam is interpolated on the
  imported mesh: once a node of its layer moves, `beam_coupling` raises a
  `ValueError`; read it on the default side, whose layer
  `boundary_morph` holds, or give a `GaussianBeam`, which is read at the
  moved nodes.

## Cost and accuracy

- One far field costs one evaluation of the field and its curl at the
  layer's quadrature points and one dense product per chunk of directions:
  a second or two for a few hundred directions on 50 000 unknowns. Columns
  and frequencies reuse nothing but the solve.
- At order 3 with elements of $\lambda/4$ and half a wavelength of layer,
  a Hertzian dipole's pattern is within 0.2-1 % of its closed form, its
  directivity within 0.1 %. With a point source the error follows the
  emitted power's; it is the field's discretization, not the transform's.
- On a result of a {term}`distributed solve`, `far_field` is a collective
  call: call it on every rank, and every rank gets the same far field
  ({doc}`distributed-solve`).

## Gradients

- `e_theta`, `e_phi`, `intensity`, `directivity`, `gain` and `rcs` are
  differentiable with respect to material fields through the solved field;
  a design inside the side costs one adjoint column.
- Every output is differentiable with respect to the node coordinates
  ({py:func}`gfem.shape.boundary_morph`): the layer's test field is read at
  the moved quadrature points, and the radiated power follows the moved
  layer as `gfem.power_flow` does. `boundary_morph` holds the elements
  touching the truncation, so the default side's layer stays as meshed.
- Directions, the side and the frequency are fixed values.

## Limits

- **A radiating truncation**: a model with no perfectly matched layer and
  no absorbing face, or with periodic pairs, raises a `ValueError`.
- **One lossless exterior medium**: `far_field` needs `Model(background=)`;
  its outside layer must be that material, with no design, no perfectly
  matched layer element and no other material, or it raises a `ValueError`.
  A layered or dielectric half-space, and a lossy background, raise a
  `NotImplementedError`.
- **Every source inside the side**: a dipole, port, sheet or scatterer in
  the layer or outside the side raises a `ValueError`.
- **Ground planes flat, PEC or PMC**: the layer may end on one flat PEC or
  PMC wall only; directions below it raise a `ValueError`.
