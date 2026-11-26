# Perfectly matched layer

A {term}`perfectly matched layer` ({py:obj}`gfem.PML`) is a meshed shell of
volume regions around the {term}`air box` that absorbs the waves leaving it.
Choose it over {py:obj}`gfem.Absorbing` when:

- waves meet the boundary obliquely: a periodic cell lit at an angle, the
  radiation of a bend;
- you want the radiation Q of a resonator, which a layer gives without
  being a small perturbation;
- an open face sits on a lossy or layered substrate, where a
  {py:obj}`gfem.FloquetPort` refuses the half-space.

The cost is the shell: its elements and their unknowns, about half a
wavelength thick on every side it covers.

## Mesh the layer

Mesh a box shell of volume regions around the air box, one region per
background (`"pml_clad"` over the cladding, `"pml_box"` under it beside the
oxide, for example). Edges and corners of the shell are layer regions too:
there the layer stretches along two or three axes. Each region keeps its
background material in `materials`, the material it continues; solves that
ignore the layer (electrostatic, DC conduction, thermal) see that material.

gfem reads the geometry from the mesh. The {term}`interior box` is the
bounding box of every tetrahedron outside the layer regions; a layer
tetrahedron beyond one of its faces stretches along that face's normal, and
the side's thickness $d$ is the layer's extent beyond the face. Every layer
region of one side must end on one plane, so each side has one thickness.
`repr(model)` shows one line per side with its thickness and regions,
such as `+z: d = 0.5 um (pml)` for the line below.

Make the layer at least $\lambda/2$ thick in its background at the
{term}`reference frequency`, with about ten elements across at order 2. Its
outer faces need no boundary: an unnamed face is a magnetic wall, and a
{py:obj}`gfem.PEC` there works the same.

## Close a line with a layer

A TEM line along $z$, 0.2 µm × 0.2 µm: 1.5 µm of air (`"air"`) closed by
0.5 µm of layer (`"pml"`), with metal plates on its $y$ walls (`"plates"`)
and a {term}`wave port` on its open end (`"port"`). Its other walls and the
layer's back face are left unnamed. The layer is designed for a
{term}`target reflection` $R_0 = 10^{-2}$ and solved at a 1 µm wavelength
and 1.5 times that frequency:

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

- The reflection is the layer's own, $R_0$ at the lowest frequency of the
  solve, which is the reference frequency when none is given, and less
  above it.
- The layer regions take a `Material` like any region; here air, as the
  line they continue.
- Above the reference frequency the line resolves the wave less well
  ($\lambda/6.7$ elements), and the reflection departs from its formula by
  a few parts in $10^{6}$: the mesh, not $R_0$, sets how small a reflection
  you can see.

## Choose the target reflection and the reference frequency

Along each axis it stretches, the layer damps the wave with the profile
$\sigma(\xi) = \sigma_\max (\xi/d)^2$ at the depth $\xi$ into it, with

$$
\sigma_\max = -\frac{3 \ln R_0}{2 k_\text{ref} d},
\qquad
|R| = R_0^{\,f \cos\theta / f_\text{ref}},
$$

$k_\text{ref} = 2\pi f_\text{ref}/c$ the vacuum wavenumber at the reference
frequency $f_\text{ref}$, and $|R|$ the reflection of a wave at the frequency
$f$ and the angle $\theta$ from the layer's normal. In a background of index
$n$ the exponent takes the factor $n$: a denser background reflects less
than $R_0$.

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

- The default `target_reflection` is $R_0 = 10^{-6}$, below what a usual
  mesh resolves: the mesh sets the accuracy, and $R_0$ only has to sit
  below it. A smaller $R_0$ needs a thicker or finer layer.
- A wave at a grazing angle reflects more: at $\cos\theta = 0.36$ the
  default layer reflects $7\times10^{-3}$, above.
- `reference_frequency` (Hz) pins $f_\text{ref}$. Without it, a driven solve
  takes its lowest frequency and an eigenmode solve its target, so two
  solves over two sub-bands get two different layers. Pin it to solve both
  with one layer:

```python
pml = gfem.PML("pml", target_reflection=1e-4, reference_frequency=f0)
low = gfem.solve(model.replace(pml=pml), gfem.Driven([f0, 1.2 * f0]), options=options)
high = gfem.solve(
    model.replace(pml=pml), gfem.Driven([1.3 * f0, 1.5 * f0]), options=options
)
```

A pinned value above a solve frequency is accepted: the layer then
reflects more than $R_0$ there, as the formula says.

## Oblique incidence on a periodic cell

A grating on a lossy substrate, lit at 60°: a {py:obj}`gfem.FloquetPort`
on the air face above reads the reflected orders, and a layer under the
substrate absorbs what goes through at any angle, where an `Absorbing` face
would reflect a third of it and a Floquet port would refuse the lossy
half-space. The layer region `"pml"` continues the substrate:

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

This is the remedy the {py:obj}`gfem.ObliqueAbsorberWarning` names when an
`Absorbing` face meets a wave obliquely:

```text
Absorbing 'bottom' meets a plane wave at 60.0° (order (0, 0) at 299.792 THz),
and its first-order condition reflects about 33.3% of the amplitude of such
a wave; a gfem.FloquetPort on 'bottom' absorbs every order at any angle, and
so does a perfectly matched layer (gfem.PML) meshed beyond it, which also
takes a lossy or layered substrate.
```

The layer must span the whole period and stretch along the non-periodic
axis only, and it may not touch a Floquet surface: keep some cell between
them. A layer opposite a Floquet port is fine.

## Wave ports through side layers

A strip guide whose cladding runs into side layers: the port faces cut
the layers on their sides, which is allowed as long as no layer stretches
along the port's normal.

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
  $|\Delta n_\text{eff}| \approx 10^{-2} e^{-2\gamma_t c}$, with $\gamma_t$
  the mode's transverse decay rate in the cladding and $c$ the distance from
  the core to the layer. On a high-contrast guide, about $\lambda_0/1.5$ of
  cladding keeps it near $10^{-5}$. A mode with more than about 1 % of its
  energy in the layer warns {py:obj}`gfem.LayerClearanceWarning`: move the
  layer out.
- **{term}`PML modes <PML mode>`** of the port's cross-section, which live
  in the layer, are dropped; the port's modes are the guided ones.
- **Port power.** A mode's 1 W counts the {term}`physical regions
  <physical region>` of the section only, so its power and its $Z_{PV}$ are
  low by the mode's share inside the layer. S is unchanged.
- **Hollow metal guides near cutoff.** Their modes meet the layer at a
  grazing angle, $\cos\theta = \beta/k_0$, and reflect more than $R_0$ (the
  formula above). Set `reference_frequency` to $f\beta/k_0$, where
  $k_0 = \beta$, to restore the target.
- A port face may cut a side layer in a driven solve, never a layer
  stretched along its normal, and never obliquely. An eigenmode solve
  refuses a wave port face that crosses the layer (`NotImplementedError`).

## Radiation Q of a resonator

An {py:obj}`gfem.Eigenmode` solve of a model with a layer finds modes that
radiate into it. A silicon disk in air, wrapped in a layer on every side:

```python
disk = gfem.Model(
    gfem.Mesh.from_gmsh("disk.msh", length_unit="um"),
    materials={"disk": gfem.Material(eps_r=12.1), "air": air, "pml": air},
    pml=gfem.PML("pml"),
)
modes = gfem.solve(disk, gfem.Eigenmode(target=193e12, n_modes=4))
q = modes.q  # the radiation Q of a lossless disk
```

- The reference frequency is the target, unless you pin it.
- Q is the total Q: radiation into the layer and the losses of the
  materials, sheets and ports.
- {term}`PML modes <PML mode>` are dropped, and the message counts them
  when fewer modes than asked are left.
- A mode that grows in time, past round-off, warns
  {py:obj}`gfem.LayerClearanceWarning`: an evanescent tail reaches the
  layer. Move the layer away.
- The 1 J normalization and the participations count the physical regions
  only. In a {term}`loss budget`, what $1/Q$ holds beyond the listed losses
  is the radiation, plus any loss of the layer's background.

## Light a scatterer from outside

A layer also truncates a model lit by an {term}`incident field`
(`Model(background=...)`, `Driven(sources=...)`, {doc}`incident-fields`).
gfem then solves for the {term}`total field` outside the layer and for the
{term}`scattered field` inside it, so the layer absorbs only what the
{term}`scatterer` sends out. Mesh it as above, with the background's
material in its regions (on a {term}`background stack`, split at the stack's
planes, each region carrying the stack's material at its height).

- **The incident field is dropped in the layer.** Where a scatterer reaches
  the layer, the incident field there must be negligible, at most $10^{-3}$
  of its peak (the {term}`incident contact`); otherwise the solve raises
  before any factorization, naming the face, the ratio it found and the
  remedy. A waveguide running into the layer is fine under a narrow beam
  that misses its end.
- **Keep one background element next to the layer.** The physical elements
  touching it, the {term}`truncation layer`, are where
  {py:obj}`gfem.cross_sections` reads its power balance and where
  {py:obj}`gfem.far_field` and {py:obj}`gfem.beam_coupling` read by default
  ({doc}`cross-sections`, {doc}`far-field`): they must hold the background
  only.
- **Shape designs hold it.** `gfem.shape.boundary_morph` holds the layer's
  nodes and those of the truncation layer, so the measurements read there
  keep their elements while the device moves ({doc}`shape-gradient`).

## What you read inside the layer

- **Fields.** Inside the layer the field is the stretched one, not a
  physical field: `Field.at`, `region_mean`, exports and plots show it as
  solved. Mask it out of a plot by the `region` cell array of the export.
- **Energies.** {py:obj}`gfem.electric_energy` and the participations skip
  the layer regions.
- **Power.** {py:obj}`gfem.power_flow` over the layer regions is the power
  they absorb. A side that cuts through the layer has no physical meaning.
- **Paths and surfaces.** {py:obj}`gfem.characteristic_impedance` paths and
  {py:obj}`gfem.interface_participation` surfaces stay in the physical part:
  a path through the layer raises, and faces in the layer do not count.

## Gradients

- Design and shape gradients are exact outside and next to the layer.
- The layer's profile is fixed at the imported mesh:
  `gfem.shape.boundary_morph` holds its nodes and those of the physical
  elements touching it, and a moved mesh that pushes device elements past a
  stretched face raises.
- The layer's background is not a gradient input: it is a constant
  `Material`. `target_reflection` and `reference_frequency` are fixed
  numbers, and a traced one raises.

## Check your results

- **The layer.** Thicken it, or lower $R_0$: S or the eigenfrequency should
  stay put to the accuracy you want.
- **The mesh.** Close a straight guide with the same layer and compare its
  $|S_{11}|$ with $R_0^{\,f\cos\theta/f_\text{ref}}$, as above; the gap is
  what your mesh resolves.
- **The clearance.** Watch {py:obj}`gfem.LayerClearanceWarning`.

How gfem itself is checked: {doc}`../verification`.

## Limits

- **No error estimates or mesh refinement for wave problems**: driven and
  eigenmode estimators and `gfem.refine_mesh` raise on a model with a layer.
  Electrostatic refinement works.
- **Box layers only**: a shell around a box with faces on the coordinate
  planes. A layer cell inside the interior box or across one of its faces
  raises; radial and curved layers are not supported.
- **One thickness per side**: every layer region of a side ends on one
  plane.
- **Constant backgrounds**: a `Material`, isotropic or with principal axes
  along $x$, $y$ and $z$; no permittivity law, no `MaterialField`, no tilted
  axes.
- **No absorber or sheet on the layer**: no `Absorbing` on a face of the
  layer, no `Impedance` or `Conductor` inside it.
- **Ports clear of the layer**: a lumped port or a Floquet surface may not
  touch it; a wave port may not lie along a stretch axis or obliquely in it.
- **No stretch along a periodic axis**: in a periodic cell the layer spans
  the period.
- **Paths and surfaces in the physical part**: a characteristic impedance
  path through the layer, or an interface surface wholly in it, raises.
- **Moved meshes**: a moved mesh may not push device elements past a
  stretched face, and a morph of a surface wholly in the layer, or in the
  elements touching it, raises.
- **Iterative solver**: `linear_solver="pmg_gmres"` solves the layer exactly
  inside its iterations, at order 2 or 3 on one process; its memory grows with
  the share of the unknowns in the layer. `"bddc_gmres"` raises, because the
  stretch stalls its iterations. `"auto"` picks the direct solver while it
  fits, then `"pmg_gmres"`, then the direct solver whatever the memory (MUMPS
  across MPI ranks).
- **Thin layers**: a side thinner than about a hundredth of the reference
  wavelength warns {py:obj}`gfem.ThinLayerWarning`: the stretch is then
  steep, and the solve loses digits.

## Next

- {doc}`periodic-cell` and {doc}`floquet-ports` for the cells.
- {doc}`loss-budget` for what the radiation Q joins.
- {doc}`eigenmode-gradient` for gradients of the modes.
- {doc}`large-driven-model` for a model too large for the direct solver.
