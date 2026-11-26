# Perfectly matched layer

A {term}`perfectly matched layer` ({py:obj}`gfem.PML`) is a meshed shell of
volume regions around the {term}`air box` that absorbs the waves leaving it.
Choose it over {py:obj}`gfem.Absorbing` for oblique waves, a resonator's
radiation Q, or an open face where a {py:obj}`gfem.FloquetPort` refuses the
half-space. It costs about half a wavelength of shell per side.

## Close a line with a layer

A TEM line along $z$: 1.5 µm of air closed by 0.5 µm of layer (`"pml"`), with
a {term}`wave port` on the open end. The {term}`target reflection` is
$R_0 = 10^{-2}$; the solve is at a 1 µm wavelength and 1.5 times that
frequency:

```python
import urllib.request

import gfem

url = "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/pml_line.msh"
urllib.request.urlretrieve(url, "pml_line.msh")
mesh = gfem.Mesh.from_gmsh("pml_line.msh", length_unit="um")

air = gfem.Material()
model = gfem.Model(
    mesh,
    materials={"air": air, "pml": air},  # the layer's background: air
    boundaries=[gfem.PEC("plates")],
    ports=[gfem.WavePort("p1", "port")],
    pml=gfem.PML("pml", target_reflection=1e-2),
)
f0 = 299_792_458.0 / 1e-6  # Hz, a wavelength of 1 um
options = gfem.SolverOptions(order=3)
result = gfem.solve(model, gfem.Driven(frequency=[f0, 1.5 * f0]), options=options)
for f, s11 in zip(result.frequency, abs(result.s["p1", "p1"]), strict=True):
    print(f"f = {f / f0:.1f} f0: |S11| = {s11:.3e}, R0^(f/f0) = {1e-2 ** (f / f0):.3e}")
```

```text
f = 1.0 f0: |S11| = 1.000e-02, R0^(f/f0) = 1.000e-02
f = 1.5 f0: |S11| = 1.005e-03, R0^(f/f0) = 1.000e-03
```

- The reflection is the layer's own, $R_0$ at the lowest solve frequency (the
  reference frequency when none is given), and less above it.
- The mesh, not $R_0$, sets how small a reflection you can see.

## Choose the target reflection and the reference frequency

Along each axis it stretches, the layer damps the wave with the profile
$\sigma(\xi) = \sigma_\max (\xi/d)^2$ at the depth $\xi$ into it, with

$$
\sigma_\max = -\frac{3 \ln R_0}{2 k_\text{ref} d},
\qquad
|R| = R_0^{\,f \cos\theta / f_\text{ref}},
$$

$k_\text{ref} = 2\pi f_\text{ref}/c$ at the reference frequency
$f_\text{ref}$, and $|R|$ the reflection at the frequency $f$ and the angle
$\theta$ from the normal. A background of index $n$ multiplies the exponent
by $n$.

```python
import math


def reflection(target, f, f_ref, cos_theta):
    """|R| of a layer backed by a wall, in vacuum, at f and the angle theta."""
    return target ** (f * cos_theta / f_ref)


def stretch_maximum(target, f_ref, thickness):
    """The stretch maximum, at the layer's outer face, 1/m."""
    k_ref = 2 * math.pi * f_ref / 299_792_458.0
    return -3 * math.log(target) / (2 * k_ref * thickness)


print(f"{reflection(1e-6, 1.0, 1.0, 0.36):.1e}")  # cos(theta) = 0.36
```

```text
6.9e-03
```

- The default `target_reflection` is $R_0 = 10^{-6}$, below what a usual mesh
  resolves. A grazing wave reflects more ($7\times10^{-3}$ at
  $\cos\theta = 0.36$, above).
- `reference_frequency` (Hz) pins $f_\text{ref}$. Without it a driven solve
  takes its lowest frequency and an eigenmode solve its target, so two
  sub-bands get two layers. Pin it to use one:

```python
pml = gfem.PML("pml", target_reflection=1e-4, reference_frequency=f0)
low = gfem.solve(model.replace(pml=pml), gfem.Driven([f0, 1.2 * f0]), options=options)
high = gfem.solve(
    model.replace(pml=pml), gfem.Driven([1.3 * f0, 1.5 * f0]), options=options
)
```

A pinned value above a solve frequency is accepted; the layer reflects more
than $R_0$ there.

## Oblique incidence on a periodic cell

A grating on a lossy substrate, lit at 60°: a {py:obj}`gfem.FloquetPort` reads
the orders and a layer under the substrate absorbs what goes through, where an
`Absorbing` face would reflect a third of it:

```python
import numpy as np

substrate = gfem.Material(eps_r=12.0, tan_delta=1e-3)
cell = gfem.Model(
    gfem.Mesh.from_gmsh("grating_cell.msh", length_unit="um"),
    materials={
        "above": air,
        "grating": gfem.Material(eps_r=4.0),
        "substrate": substrate,
        "pml": substrate,
    },
    boundaries=[gfem.Periodic(("x_min", "x_max")), gfem.Periodic(("y_min", "y_max"))],
    ports=[gfem.FloquetPort(f"R {p}", "top", polarization=p) for p in ("TE", "TM")],
    pml=gfem.PML("pml"),
)
f = 299_792_458.0 / 1.55e-6
k = 2 * np.pi * f / 299_792_458.0
kt = (k * np.sin(np.radians(60.0)), 0.0, 0.0)
s = gfem.solve(cell, gfem.Driven(f, wave_vector=kt)).s
```

This is the remedy the {py:obj}`gfem.ObliqueAbsorberWarning` names:

```text
Absorbing 'bottom' meets a plane wave at 60.0° (order (0, 0) at 299.792 THz),
and its first-order condition reflects about 33.3% of the amplitude of such
a wave; a gfem.FloquetPort on 'bottom' absorbs every order at any angle, and
so does a perfectly matched layer (gfem.PML) meshed beyond it, which also
takes a lossy or layered substrate.
```

The layer spans the whole period, stretches along the non-periodic axis only
and keeps a cell from a Floquet surface.

## Wave ports through side layers

A strip guide whose cladding runs into side layers; the port faces may cut
them on their sides, never a layer stretched along the port's normal.

```python
oxide = gfem.Material(eps_r=2.07)
strip = gfem.Model(
    gfem.Mesh.from_gmsh("strip.msh", length_unit="um"),
    materials={
        "core": gfem.Material(eps_r=12.1),
        "clad": air,
        "box": oxide,
        "pml_clad": air,
        "pml_box": oxide,
    },
    ports=[gfem.WavePort("in", "port1"), gfem.WavePort("out", "port2")],
    pml=gfem.PML(("pml_clad", "pml_box")),
)
result = gfem.solve(strip, gfem.Driven(frequency=np.linspace(185e12, 200e12, 31)))
```

- **{term}`Clearance`.** The layer perturbs the guided mode by about
  $|\Delta n_\text{eff}| \approx 10^{-2} e^{-2\gamma_t c}$ ($\gamma_t$ the
  cladding decay rate, $c$ the core-to-layer distance). A mode with over 1 %
  of its energy in the layer warns {py:obj}`gfem.LayerClearanceWarning`: move
  the layer out.
- **{term}`PML modes <PML mode>`** of the port's section are dropped. Port
  power counts the {term}`physical regions <physical region>` only; S is
  unchanged.
- **Hollow metal guides near cutoff** reflect more than $R_0$ at
  $\cos\theta = \beta/k_0$: set `reference_frequency` to $f\beta/k_0$.

## Radiation Q of a resonator

An {py:obj}`gfem.Eigenmode` solve finds modes that radiate into the layer. A
silicon disk in air, wrapped in a layer on every side:

```python
disk = gfem.Model(
    gfem.Mesh.from_gmsh("disk.msh", length_unit="um"),
    materials={"disk": gfem.Material(eps_r=12.1), "air": air, "pml": air},
    pml=gfem.PML("pml"),
)
modes = gfem.solve(disk, gfem.Eigenmode(target=193e12, n_modes=4))
q = modes.q  # the radiation Q of a lossless disk
```

- Q is the total Q: radiation plus material, sheet and port losses. The
  reference frequency is the target unless pinned.
- {term}`PML modes <PML mode>` are dropped; a growing mode warns
  {py:obj}`gfem.LayerClearanceWarning`. In a {term}`loss budget`, what $1/Q$
  holds beyond the listed losses is the radiation.

## Mesh the layer

Mesh a box shell of volume regions around the air box, one per background,
edges and corners included, each keeping the material it continues. The
{term}`interior box` bounds every tetrahedron outside the layer; the side's
thickness $d$ is the layer's extent beyond a face. Make it at least
$\lambda/2$ at the {term}`reference frequency`, about ten elements across at
order 2. Unnamed outer faces are magnetic walls.

## Light a scatterer from outside

Under an {term}`incident field` ({doc}`incident-fields`) the layer truncates
the {term}`scattered field` and absorbs what the {term}`scatterer` sends out.
Mesh it as above, with the background's material.

- **The incident field is dropped in the layer.** It must be at most
  $10^{-3}$ of its peak there (the {term}`incident contact`) or the solve
  raises, naming the face and the remedy.
- **Keep one background element next to the layer**, the
  {term}`truncation layer`, where {py:obj}`gfem.cross_sections` and
  {py:obj}`gfem.far_field` read.

## What you read inside the layer

Fields there are the stretched ones: mask them by the export's `region`
array. {py:obj}`gfem.power_flow` over the layer is the power it absorbs.

## Gradients

Design and shape gradients are exact outside and next to the layer. Its profile
is fixed at the imported mesh; `target_reflection` and `reference_frequency`
are fixed numbers (a traced one raises).

## Check your results

- Thicken the layer or lower $R_0$: S or the eigenfrequency stays put.
- A straight guide closed by the same layer gives
  $|S_{11}| = R_0^{\,f\cos\theta/f_\text{ref}}$; the gap is your mesh's.
- {py:obj}`gfem.LayerClearanceWarning` does not fire.

How gfem itself is checked: {doc}`../verification`.

## Limits

- **No error estimates or mesh refinement for wave problems**: they raise;
  electrostatic refinement works.
- **Box layers only**: faces on the coordinate planes, none inside the interior
  box, no curved layer.
- **One thickness per side**: a side's regions end on one plane.
- **Constant backgrounds**: a plain `Material`; no law, `MaterialField` or
  tilted axes.
- **No absorber or sheet on the layer**: no `Absorbing`, `Impedance` or `Conductor` in it.
- **Ports clear of the layer**: no lumped port or Floquet surface touches it;
  a wave port may not lie along a stretch axis or obliquely.
- **No stretch along a periodic axis**: the layer spans the period.
- **Paths and surfaces in the physical part**: one through the layer raises.
- **Moved meshes**: device elements may not pass a stretched face.
- **Iterative solver**: `"pmg_gmres"` at order 2 or 3 on one process;
  `"bddc_gmres"` raises.
- **Thin layers**: under a hundredth of the reference wavelength warns
  {py:obj}`gfem.ThinLayerWarning`.

## Next

{doc}`periodic-cell`, {doc}`floquet-ports`, {doc}`loss-budget`,
{doc}`eigenmode-gradient`, {doc}`large-driven-model`.
