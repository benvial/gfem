# Far fields and directivity

{py:func}`gfem.far_field` turns a solved field into its
{term}`far-field amplitude`, per direction. It works on any column of a driven
solve: an excited port (an antenna), a {term}`point dipole` or an
{term}`incident field` (a scatterer). The {py:obj}`gfem.FarField` it returns
holds the {term}`radiation intensity`, the {term}`directivity`, gain, the
circular components and a plane wave's bistatic radar cross section.

## Read the pattern of an antenna

A strip dipole two thirds of a wavelength long, fed by a lumped port in its
gap, in air wrapped in a {term}`perfectly matched layer`, at 1 µm. The
model's background is the {term}`exterior medium` the field radiates into:

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

- `theta` and `phi` are broadcast arrays in rad, like an incident field's
  angles.
- `e_theta` and `e_phi` are $F_\theta$ and $F_\phi$ in V, with
  $\mathbf{E} \approx \mathbf{F}\,e^{-ikr}/r$ and the phase referred to the
  origin (`gfem.conventions`, Far fields).
- `intensity` is $U = |\mathbf{F}|^2/(2\eta)$ in W/sr; `radiated_power` is the
  power out of the side, from the residual as {py:func}`gfem.power_flow`
  computes it. The integral of $U$ agrees with it to the discretization.
- `directivity` is $4\pi U / P_\text{rad}$, `gain` $4\pi U / P_\text{acc}$,
  `realized_gain` $4\pi U / P_\text{inc}$ and `radiation_efficiency`
  $P_\text{rad}/P_\text{acc}$. A port is fed 1 W; on this coarse mesh its $S$
  meets the radiated power to 3 %, so the lossless strip reads 0.97.
- For a point dipole, the accepted and incident powers are its
  {term}`emitted power`.

## Circular polarization

`e_right` and `e_left` are the components $(F_\theta \pm iF_\phi)/\sqrt{2}$
along right- and left-hand circular polarization (IEEE), and `axial_ratio` is
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
\cdot\mathbf{F}(\hat{\mathbf{k}})\big)/E_0$.

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
surface and is two to five times more accurate than a surface integral at
orders 2 and 3.

- By default (`side=None`) the side is every physical element with no vertex on
  the perfectly matched layer or an absorbing face.
- `side=` takes volume region names or a predicate on centroids and must hold
  every source, port and scatterer, one element clear of its outside layer.
- The layer must be the exterior medium: constant, isotropic and lossless.

## Ground planes

A flat PEC or PMC exterior face that the layer meets is a
{term}`ground plane`: the model ends on it instead of a layer, and the far
field is that of the sources and their images above it. The same dipole
$0.21\lambda$ above a PEC ground gives, at order 3, the pattern of the dipole
and its image to 0.2 %.

## Couple into a beam

The same reaction with a beam as the test field gives the
{term}`beam coupling`: {py:func}`gfem.beam_coupling` reads how much of a solved
field goes into an incoming beam, such as a fiber's above a grating coupler.
For a port column fed 1 W it is, by reciprocity, the wave the port would
receive if the beam lit the device, so every fiber position, angle and waist
comes from one port-fed solve:
`abs(gfem.beam_coupling(result, "wg", fiber, model=model)) ** 2`.

- The beam is a `gfem.GaussianBeam` or a `gfem.IncidentField` travelling
  toward the device, scaled to 1 W through the side.
- The default side's surface is the truncation face the beam brings most power
  in through; a named side must cross the beam once.
- The layer must be the beam's medium, lossless; otherwise `beam_coupling`
  raises a `ValueError`. A beam clipped by the truncation warns.
- It is differentiable with respect to material fields and node coordinates;
  an `IncidentField` beam raises once a node of its layer moves, so give a
  `GaussianBeam`.

## Cost and accuracy

- At order 3 with elements of $\lambda/4$ and half a wavelength of layer, a
  Hertzian dipole's pattern is within 0.2-1 % of its closed form, its
  directivity within 0.1 %; a few hundred directions take a second or two.
- On a {term}`distributed solve`, `far_field` is collective: call it on every
  rank ({doc}`distributed-solve`).

## Gradients

- `e_theta`, `e_phi`, `intensity`, `directivity`, `gain` and `rcs` are
  differentiable with respect to material fields; a design inside the side
  costs one adjoint column.
- Every output is differentiable with respect to the node coordinates
  ({py:func}`gfem.shape.boundary_morph`), which holds the default side's layer
  as meshed.
- Directions, side and frequency are fixed.

## Limits

- **A radiating truncation**: a model with no perfectly matched layer or
  absorbing face, or with periodic pairs, raises a `ValueError`.
- **One lossless exterior medium**: `far_field` needs `Model(background=)`
  as its outside layer, or raises a `ValueError`; a layered half-space or a
  lossy background raises a `NotImplementedError`.
- **Every source inside the side**: a source in the layer or outside the side
  raises a `ValueError`.
- **Ground planes flat, PEC or PMC**: one flat wall only; directions below it
  raise a `ValueError`.
