# Incident fields

An {term}`incident field` is the field a {term}`source` would have in the
{term}`background stack` alone: a plane wave ({py:obj}`gfem.PlaneWave`), a
focused Gaussian beam ({py:obj}`gfem.GaussianBeam`) or a function
({py:obj}`gfem.IncidentField`). Pass it to {py:obj}`gfem.Driven` as
`sources=`; each source is one more {term}`excitation` on the same factorization. gfem
solves for the {term}`total field`, and for the {term}`scattered field` inside
the {term}`perfectly matched layer`.

## Give the model a background

`Model(background=...)` sets the medium: a {py:obj}`gfem.Material` or a
{py:obj}`gfem.Background`. The {term}`scatterer` is every region whose
material is not the background's. A sphere of radius 0.25 µm
($\varepsilon = 4 - i$) in air, at 1 µm:

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
$\hat k = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$, with
`theta` from $+z$: a wave coming down at 30° from the vertical is
`theta=np.pi - np.radians(30)`. Polarization is `"TE"`, `"TM"`, `"RCP"`,
`"LCP"` or a Jones pair; `amplitude` is the peak $E_0$ in V/m, zero phase at
the origin ({doc}`../conventions`, Incident fields). Two waves at 30° from
$+z$, solved as two columns:

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

With ports, `excite=()` solves the sources only, and `result.port_wave(port,
source)` is the wave each source sends into each port.

## Focus a Gaussian beam

A {term}`Gaussian beam` ({py:obj}`gfem.GaussianBeam`) takes the angles and
polarizations of the plane wave, a {term}`waist` $w_0$ (m) and a focus (m). It
is an exact solution of Maxwell's equations and carries 1 W through any plane
across its axis. Its focal field and {term}`focal intensity` take the
frequency and the model:

```python
beam = gfem.GaussianBeam("beam", waist=3e-6, theta=theta, phi=0.0, polarization="TE")
print(f"focal field {beam.focal_field(f, model=model):.4g} V/m")
print(f"focal intensity {beam.focal_intensity(f, model=model):.4g} W/m²")
```

```text
focal field 7.27e+06 V/m
focal intensity 7.014e+10 W/m²
```

## Read the total, scattered and incident fields

`result.e_field(f, source)` is the total field; `part="scattered"` and
`part="incident"` give the other two, which add exactly:

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

{py:obj}`gfem.power_flow` on a source's total field is the net power leaving a
side, exact; minus it is the power the side absorbs:

```python
absorbed = -gfem.power_flow(total, "sphere", model=model)
print(f"sphere absorbs {float(absorbed):.4e} W")
```

```text
sphere absorbs 2.6990e-16 W
```

That is the {term}`cross sections` absorption ({doc}`cross-sections`) in
watts, split per region. The side must stay outside the layer.

## Give the field as a function

{py:obj}`gfem.IncidentField` takes `function(points, frequency) -> E`: `points`
is `(n, 3)` in metres, `frequency` in Hz, and it returns the `(n, 3)` complex
peak field in V/m. It is evaluated on the host with NumPy arrays, so it must
be pure. The same wave as `"te"` above:

```python
def te_wave(points, frequency):
    k = 2 * np.pi * frequency / 299792458.0
    k_hat = np.array([np.sin(theta), 0.0, np.cos(theta)])
    return np.exp(-1j * k * points @ k_hat)[:, None] * np.array([0.0, 1.0, 0.0])


wave = gfem.IncidentField("te_function", te_wave)
```

gfem does not check that it solves Maxwell's equations. Define it in a module,
not in `__main__`, so frequency workers can load it.

## Put the scatterer on a substrate

A {py:obj}`gfem.Background` of several layers is a planar stack along $z$,
bottom to top, separated by `interfaces` (m); `ground=` ends it below on a
perfect conductor. gfem solves the stack for each wave, so the incident field
already holds the substrate's reflection and transmission and the scatterer is
only what differs from it:

```python
glass, air = gfem.Material(eps_r=2.25), gfem.Material()
film = gfem.Material(eps_r=4.0, tan_delta=0.01)
stack = gfem.Background([glass, film, air], interfaces=[0.0, 0.12e-6])
```

Mesh the stack's planes so no tetrahedron crosses one, and pass the same
{py:obj}`gfem.Material` to `materials` and to the stack. Split the layer's
regions at the planes too, each with the stack's material at its height.
`theta < pi/2` comes up from the bottom layer, `theta > pi/2` down from the
top one; that {term}`incidence layer` must be lossless. With `ground=z` the
model ends on a {py:obj}`gfem.PEC` face on that plane and the wave comes from
above.

## Choose the truncation

The incident field must leave the model without meeting what absorbs or
reflects it:

- A {py:obj}`gfem.PML` layer or {py:obj}`gfem.Absorbing` faces truncate the
  model and drop the incident field there, so where a scatterer reaches them
  the field must be at most $10^{-3}$ of its peak, the {term}`incident
  contact`.
- An exterior wall must be satisfied by the incident field: a
  {py:obj}`gfem.PEC` wall where its $E$ is normal, a magnetic wall (the default)
  where its $H$ is normal. Such a wall is a symmetry plane: a model cut by $n$
  of them reports $2^{-n}$ of the full model's cross sections and powers.
- Ports read the scattered field and must lie outside the incident field.

Each rule is checked before factorization and names the face and the remedy.

## Limits

- **A radiating truncation**: needs a perfectly matched layer or absorbing
  face; exterior impedance, conductor and lumped port faces are refused.
- **Scatterers clear of the truncation**: a scatterer meeting the layer under
  a plane wave is refused; use a narrow beam that misses it.
- **Walls the incident field satisfies**: an exterior PEC or magnetic wall it
  crosses is refused.
- **Ports outside the incident field**: a wave port face it reaches is
  refused.
- **Lossless incidence**: a plane wave needs a lossless incidence layer; a
  beam on a stack or ground plane is refused, so give its field as an
  {py:obj}`gfem.IncidentField`.
- **Stacks matched to the mesh**: a plane cutting a tetrahedron, grazing
  incidence, a ground without a PEC face and a layer region off the stack are
  refused.
- **Beams not too tight**: $k w_0 < 0.2$ (a waist under about $\lambda/30$)
  is refused.
- **Material gradients**: gradients flow to the scatterer's material fields,
  not to node coordinates or a design region touching the truncation.
- **Functions are not written**: a result or job with an
  {py:obj}`gfem.IncidentField` cannot be saved.
