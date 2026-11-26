# Incident fields

An {term}`incident field` is the field a {term}`source` would have in the
{term}`background stack` alone: a plane wave ({py:obj}`gfem.PlaneWave`), a
focused Gaussian beam ({py:obj}`gfem.GaussianBeam`) or a field you give as a
function ({py:obj}`gfem.IncidentField`). Pass it to
{py:obj}`gfem.Driven` as `sources=`; each source adds a column to the solve
(an {term}`excitation`), beside the excited ports, on the same factorization. Use it for anything lit
from outside: a particle, an antenna receiving a wave, an absorber.

gfem solves for the {term}`total field` (incident plus {term}`scattered
field`) everywhere outside the {term}`perfectly matched layer` and for the
scattered field inside it, so the layer absorbs only what the
{term}`scatterer` sends out. The incident field enters as a load computed from
the source: it is never meshed, never stored, and it costs one more column.

## Give the model a background

The background is the medium the incident field travels in, set with
`Model(background=...)`: one {py:obj}`gfem.Material`, or a
{py:obj}`gfem.Background`. The {term}`scatterer` is then every region whose
material is not the background's, chosen by region, not by value: a
{py:obj}`gfem.MaterialField` is always part of it, whatever its values.

A sphere of radius 0.25 µm ($\varepsilon = 4 - i$) in a cube of air wrapped in
a box layer, at a 1 µm wavelength:

```python
import urllib.request

import numpy as np

import gfem

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/sphere.msh"
)
urllib.request.urlretrieve(url, "sphere.msh")
mesh = gfem.Mesh.from_gmsh("sphere.msh", length_unit="um")

air = gfem.Material()
model = gfem.Model(
    mesh,
    materials={
        "sphere": gfem.Material(eps_r=4.0, tan_delta=0.25),
        "air": air,
        "pml": air,
    },
    pml=gfem.PML("pml"),
    background=air,
)
```

## Light it with plane waves

A {py:obj}`gfem.PlaneWave` travels along
$\hat k = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$: `theta` is
measured from $+z$, so a wave coming down at 30° from the vertical is
`theta=np.pi - np.radians(30)`. Its polarization is `"TE"` ($\hat\phi$,
$E$ normal to the plane of incidence), `"TM"` ($\hat\theta$), `"RCP"` or
`"LCP"` (IEEE circular), or a Jones pair on $(\hat\theta, \hat\phi)$.
`amplitude` is the peak $E_0$ in V/m, with zero phase at the model's origin
({doc}`../conventions`, Incident fields). Two waves at 30° from $+z$ in the
$xz$ plane, solved as two columns:

```python
theta = np.radians(30)
waves = (
    gfem.PlaneWave("te", theta=theta, phi=0.0, polarization="TE"),
    gfem.PlaneWave("tm", theta=theta, phi=0.0, polarization="TM"),
)
f = 299792458.0 / 1e-6
problem = gfem.Driven(f, sources=waves, keep_fields=True)
result = gfem.solve(model, problem, options=gfem.SolverOptions(order=2))
print(result)
```

```text
DrivenResult(1 frequencies 299.8 THz, ports=[], excited=[], sources=['te', 'tm'], 2 kept fields, SweepInfo(uniform, 1 frequencies solved))
```

A model with no port takes sources alone; `result.s` then has no rows. With
ports, `excite=()` solves the sources only, and `result.port_wave(port,
source)` is the wave each source sends into each port.

## Focus a Gaussian beam

A {term}`Gaussian beam` ({py:obj}`gfem.GaussianBeam`) takes the plane wave's
angles and polarizations, a {term}`waist` $w_0$ (m, the same at every frequency) and a focus
(m). It is an exact solution of Maxwell's equations whose paraxial limit is
the fundamental Gaussian beam, so it leaves no spurious source however tight
it is. Like an excited port it carries 1 W through any plane across its axis,
with a real, positive field at its focus; its focal field and
{term}`focal intensity` depend on the background, so they take the frequency
and the model:

```python
beam = gfem.GaussianBeam("beam", waist=3e-6, theta=theta, phi=0.0, polarization="TE")
print(f"focal field {beam.focal_field(f, model=model):.4g} V/m")
print(f"focal intensity {beam.focal_intensity(f, model=model):.4g} W/m²")
```

```text
focal field 7.27e+06 V/m
focal intensity 7.014e+10 W/m²
```

Pass it in `sources=` beside the plane waves: each is one more column on the
same factorization. The focus may sit anywhere, inside the scatterer too.

## Read the total, scattered and incident fields

`result.e_field(f, source)` is the total field. `part="scattered"` subtracts
the incident field, and `part="incident"` is the incident field on the same
space, zero in the layer; the three add exactly:

```python
total = result.e_field(f, "te")
scattered = result.e_field(f, "te", part="scattered")
incident = result.e_field(f, "te", part="incident")
parts = scattered.coefficients + incident.coefficients
print("the parts add up:", np.allclose(parts, total.coefficients, rtol=1e-14, atol=0))
```

```text
the parts add up: True
```

## Measure the absorption of one region

{py:obj}`gfem.power_flow` on a source's total field is the net power leaving
a side, exact: the incident load is not a port, so its share is subtracted
from the residual. Minus it is the power the side absorbs, so one region's
absorption is:

```python
absorbed = -gfem.power_flow(total, "sphere", model=model)
print(f"sphere absorbs {float(absorbed):.4e} W")
```

```text
sphere absorbs 2.6990e-16 W
```

That is the absorption a {term}`cross sections` measurement gives
({doc}`cross-sections`), in watts for a 1 V/m wave, and it splits per
region where that one cannot. The side must stay outside the layer; read the
scattered part instead for the power the scattered field carries.

## Give the field as a function

{py:obj}`gfem.IncidentField` takes `function(points, frequency) -> E`:
`points` is an `(n, 3)` array in metres, `frequency` in Hz, and it returns
the `(n, 3)` complex peak field in V/m. gfem evaluates it on the host with
NumPy arrays at the points it needs, so it must be a pure function of its
arguments. The same wave as `"te"` above:

```python
def te_wave(points, frequency):
    k = 2 * np.pi * frequency / 299792458.0
    k_hat = np.array([np.sin(theta), 0.0, np.cos(theta)])
    return np.exp(-1j * k * points @ k_hat)[:, None] * np.array([0.0, 1.0, 0.0])


wave = gfem.IncidentField("te_function", te_wave)
```

The field should solve Maxwell's equations in the background: gfem does not
check it, and a field that does not is a source spread over the whole model.
A result holding an {py:obj}`gfem.IncidentField` cannot be written to a
file, since its function is code; define the function in a module (not in
`__main__`) so that frequency workers can load it.

## Put the scatterer on a substrate

A {py:obj}`gfem.Background` of several layers is a planar stack along $z$,
bottom to top, separated by the planes `interfaces` (m); `ground=` ends it
below on a perfect conductor. gfem solves the stack for each plane wave,
frequency and polarization, so the incident field already holds the
substrate's reflection and transmission, and the {term}`scatterer` is only
what differs from the stack:

```python
glass, air = gfem.Material(eps_r=2.25), gfem.Material()
film = gfem.Material(eps_r=4.0, tan_delta=0.01)
stack = gfem.Background([glass, film, air], interfaces=[0.0, 0.12e-6])
```

Mesh the stack's planes: no tetrahedron may cross one, and each element
belongs to the background when its region's material is the stack's layer
at its height, so pass the same {py:obj}`gfem.Material` to `materials` and
to the stack. Split the {term}`perfectly matched layer`'s regions at the
planes too, each carrying the stack's material at its height: a layer
region that does not is a scatterer reaching the layer, refused under a
plane wave.

`theta < pi/2` comes up from the bottom layer, `theta > pi/2` down from the
top one; that {term}`incidence layer` must be lossless, the others may be lossy.
$E_0$ and its phase are those of the incident wave in its own layer, at the
model origin. From the substrate past the critical angle the light reaches
the air side as an evanescent field, with no special setting. With
`ground=z`, the model ends on a {py:obj}`gfem.PEC` face on that plane and
the wave comes from above.

## Choose the truncation

The incident field must leave the model without meeting what absorbs or
reflects it:

- A {py:obj}`gfem.PML` layer or {py:obj}`gfem.Absorbing` faces truncate the
  model. gfem drops the incident field in the layer, so where a scatterer
  reaches the layer or an absorbing face the incident field there must be
  negligible: at most $10^{-3}$ of its peak, the {term}`incident contact`.
  A waveguide that runs into the layer is fine under a narrow beam that
  misses its end, refused under a plane wave that lights it.
- An exterior wall must be satisfied by the incident field: a
  {py:obj}`gfem.PEC` wall where its $E$ is normal, a magnetic wall (the
  default for an unnamed face) where its $H$ is normal. Such a wall is a
  symmetry plane: a model cut by $n$ of them reports $2^{-n}$ of the full
  model's cross sections and powers.
- Ports read the scattered field and must lie outside the incident field.

Each rule is checked before any factorization and names the face, the ratio
it found and the remedy.

## Limits

- **A radiating truncation**: an incident field needs a perfectly matched
  layer or an absorbing face, and refuses exterior impedance, conductor and
  lumped port faces.
- **Scatterers clear of the truncation**: where a scatterer meets the layer
  or an absorbing face, the incident field must be negligible.
- **Walls the incident field satisfies**: an exterior PEC or magnetic wall it
  crosses is refused.
- **Ports outside the incident field**: a wave port face it reaches is
  refused.
- **Lossless incidence**: a plane wave comes from a lossless material, or a
  lossless incidence layer of a stack; a beam travels in one lossless
  material: on a stack of several layers or a ground plane it is refused,
  so give its field in the stack as an {py:obj}`gfem.IncidentField`.
- **Stacks matched to the mesh**: a stack plane may not cut a tetrahedron,
  a plane wave may not graze the planes or come up from below a ground
  plane, a ground plane needs a PEC face on it and no element below it, and
  a layer region off the stack is a scatterer reaching the layer.
- **Beams not too tight**: a beam with $k w_0 < 0.2$ (a waist under about
  $\lambda/30$) at a solved frequency cannot be normalized to 1 W and is
  refused.
- **Material gradients**: gradients flow to the scatterer's material fields;
  the node coordinates, and a design region touching the truncation, are not
  differentiable with an incident field.
- **Functions are not written**: a result or job with an
  {py:obj}`gfem.IncidentField` cannot be saved.
