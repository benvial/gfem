# 2D models

A waveguide bend seen from above in effective indices, a parallel-plate line, a
metal grating's cross-section at normal incidence: when neither the structure nor
the field varies along z, gfem solves the plane alone, as a {term}`2D model`. The
field is then one component normal to the plane, $E_z$ or $H_z$, and every power
and field is per metre along z.

## A 2D model, a slab or a 3D model

- **A 2D model** fits when the structure is uniform along z, waves travel in the
  $(x, y)$ plane and you want one polarization: E along z
  ({term}`Ez polarization`) or H along z ({term}`Hz polarization`). The two do not
  mix in a uniform structure, so each is solved on its own.
- **A slab**, the same plane read with {py:meth}`gfem.Mesh.from_gmsh_cross_section`
  and extruded one element thick between plates ({term}`slab model`), poses what
  a 2D model refuses: oblique or conical incidence, periodic cells and Floquet
  ports, lumped ports, metal sheets, thermal problems and eigenmodes.
- **A 3D model** is needed as soon as anything varies along z: a vertical stack,
  a finite height, leakage into a substrate.

A 2D model solves a far smaller system than a slab of the same plane, so a
design study runs faster and on a finer mesh.

## The mesh

Draw the plane in gmsh, mesh it with triangles only, all at one height z, and
read it with {py:meth}`gfem.Mesh.from_gmsh`, with the length unit of the file:

- each physical surface is an area, a region that takes a material;
- each physical curve is a line, which takes a boundary condition or a port.

```python
import numpy as np
from scipy.constants import c

import gfem

mesh = gfem.Mesh.from_gmsh("bend2d.msh", length_unit="um")
print(mesh.dim)  # 2: a plane of triangles
print(mesh.volumes, mesh.surfaces)  # the areas, then the lines
```

`mesh.nodes` holds $(x, y)$ in m, `mesh.volume_elements` the triangles and
`mesh.surface_elements` the line segments.

## Solve one polarization

The waveguide bend of {doc}`../tutorials/bend`: core and cladding areas, an
absorbing outer line and two {term}`line ports <Line port>` across the guides.
The polarization is stated on the problem, and every driven problem on a 2D
model must state it:

```python
core = gfem.Material(eps_r=2.25**2)
cladding = gfem.Material(eps_r=1.444**2)
bend = gfem.Model(
    mesh,
    materials={"core": core, "cladding": cladding, "design": cladding},
    boundaries=[gfem.Absorbing("outer", order=2)],
    ports=[
        gfem.WavePort("p1", "port1", pec_edges=("outer",)),
        gfem.WavePort("p2", "port2", pec_edges=("outer",)),
    ],
)
f0 = c / 1.55e-6  # Hz
problem = gfem.Driven(frequency=f0, excite="p1", polarization="Ez", keep_fields=True)
result = gfem.solve(bend, problem)

t = abs(result.s["p2", "p1"][0]) ** 2  # the power fraction into p2's mode
ez = result.e_field(f0, excite="p1")  # E_z in V/m for 1 W/m incident
values = ez.at(np.array([[0.0, 0.0], [-2e-6, 0.0]]))  # (m, 2) points, m
```

- `ez.at` takes $(x, y)$ points and returns one complex value per point.
- With `polarization="Hz"` the field is $H_z$ in A/m: read it with
  `result.h_field(f0, excite="p1")`. Asking a result for the other component
  raises `ValueError` naming the right one.
- `ez.to_vtu("bend.vtu")` writes the field on the triangles for ParaView.
- The S-parameters follow the electric field in both polarizations, as on every
  other gfem model.

## Line ports

A {py:obj}`gfem.WavePort` on a straight exterior line is a line port. Its mode
is that of the line's cross-section: a guide's profile across its width, or the
uniform TEM mode of a parallel-plate line. {py:obj}`gfem.PortMode` solves it on
its own, again with the polarization:

```python
mode = gfem.solve(bend, gfem.PortMode("p1", f0, polarization="Ez"))
print(mode.n_eff)  # the effective index of the port's fundamental mode
```

- A port line must be straight and on the outside of the mesh.
- `pec_edges` makes the ends of a port line conducting, as on a 3D port.
  In Hz a `pec_edges` line must be a `PEC`, `Absorbing` or port line of the
  model: on a PMC line, the default, the field is fixed while the mode is free,
  and the solve raises.
- A line port carries 1 W/m and needs no reference impedance: a `voltage_path`
  raises, and the mode's $Z_{PV}$ is NaN.
- Lumped ports raise on a 2D model: feed a line with a line port across it, as the
  {doc}`topology tutorial <../tutorials/topology>` does.

## Per metre along z

Every power and field of a 2D model is per metre along z
({py:obj}`gfem.conventions`, "2D models"):

- each excited port is fed 1 W/m of mode power;
- S is a ratio of powers, the same as for any height of the structure, and the
  same as a slab's;
- a field is the field for 1 W/m. A slab of thickness $h$ fed 1 W carries
  $1/h$ W/m, so its field relates to the 2D model's by
  $E^{\mathrm{2D}} = \sqrt{h}\,E^{\mathrm{slab}}$.

## Conductors in each polarization

A line named {py:obj}`gfem.PEC` or {py:obj}`gfem.PMC` acts on the field normal to
the plane as follows. An exterior line no boundary names is a PMC.

| Line | Ez polarization | Hz polarization |
| --- | --- | --- |
| `PEC` | $E_z = 0$ | $\partial_n H_z = 0$ |
| `PMC`, or unnamed | $\partial_n E_z = 0$ | $H_z = 0$ |

- **Ez.** A metal is a `PEC` line, inside the mesh as well as on its outside (a
  thin strip), or an area of its own with a conducting or Drude
  {py:obj}`gfem.Material`.
- **Hz.** A metal is a hole in the plane: leave it out of the mesh, or name its
  area in `excluded=`, and name the hole's edge `PEC`. An unnamed edge is a PMC,
  $H_z = 0$, which is a different and wrong wall. A `PEC` on an interior line
  raises: in this polarization a thin strip would let the wave through
  unchanged. Cut the mesh along the strip, or mesh the metal as an excluded
  area.
- A `PMC` on an interior line raises in Ez, for the same reason.
- {py:obj}`gfem.Conductor` and {py:obj}`gfem.Impedance` surfaces raise on a 2D
  model.

## Absorbing boundaries

{py:obj}`gfem.Absorbing` works in both polarizations, at first or second order.
The second order absorbs waves that reach the line at an angle better than the
first, in Ez and in Hz. A slab in Hz has no such gain: its second order acts as
its first for this polarization, so a 2D model in Hz is the better-closed
problem.

## Designs and gradients

A design is written as on a 3D model ({doc}`robust-optimization`,
{doc}`s-parameter-gradient`):

```python
from gfem import design

grid = design.LayerGrid(mesh, "design", pitch=0.05e-6)
drude = gfem.Drude(plasma_frequency=2.2e15, damping=1.3e13)  # Hz
gold = gfem.Material(eps_r=gfem.PermittivityLaw(terms=[drude]))


def material(rho):
    density = grid.to_elements(rho)  # the pixels, bound to the grid
    return gfem.MaterialField.interpolate(density, cladding, gold, scheme="index")
```

- The solve reads the pixels' bilinear interpolation at several points inside
  each triangle, more of them at a higher order, so a material edge between
  pixels is not rounded to whole triangles.
- `grid.to_elements(rho)` is the pixels bound to the grid, not one value per
  triangle: `np.asarray` of it raises. Plot the pixels themselves, `rho` on
  `grid.x` and `grid.y`, with their 0.5 contour as the material edge.
- `jax.grad` of any S functional goes through the solve to the pixels, in both
  polarizations.
- A design at a line port differentiates too: a `gfem.MaterialField` region
  with elements on a port's line moves the port's mode, and `jax.grad` of
  an S functional or of `PortMode`'s `n_eff` and field values follows it, in
  both polarizations, lossless or lossy. The node coordinates, law
  parameters and boundary values are still not differentiated.
- Hz takes blends of constant permittivities only: a permittivity law or a
  conductivity in a blend raises. A {py:obj}`gfem.Material` region with a law
  or a conductivity is accepted.

## Check your results

- **The port mode.** For a guide in a uniform cladding, compare
  `PortMode(...).n_eff` with the symmetric slab guide's closed form, as the
  {doc}`bend tutorial <../tutorials/bend>` does.
- **Power balance and reciprocity.** For lossless media, $|S_{11}|^2 + |S_{21}|^2 \le 1$
  and $S_{12} = S_{21}$.
- **The element order.** Solve again with `gfem.SolverOptions(order=3)`; the
  change in S bounds the discretization error, the pixel sampling included.
- **The slab.** In Ez, the same plane posed as a slab between PEC plates gives
  the same S. In Hz the slab's port absorbs its mode with a different boundary
  term, so expect agreement to a few $10^{-3}$ only ({doc}`../verification`).

## What raises

A 2D model raises `NotImplementedError` for what it does not solve, and the
message names the alternative, most often the slab:

- eigenmode, electrostatic, DC conduction and thermal problems;
- oblique incidence (`wave_vector=`), periodic pairs and Floquet ports;
- lumped ports, conductor and impedance surfaces, sheet designs;
- anisotropic materials and PMLs (a {doc}`piezoelectric problem
  <piezoelectric-admittance>` takes anisotropic materials);
- the adaptive sweep (use `sweep="uniform"`), iterative and GPU solvers, and
  solves over several processes;
- power flow, energies, participations, characteristic impedance and region or
  element means (read the field with `Field.at` instead);
- mesh refinement, error estimates and moved nodes (change the mesh in gmsh;
  a piezoelectric problem takes node maps, {doc}`piezoelectric-gradient`).

## Next

- {doc}`../tutorials/bend`, {doc}`../tutorials/metal-bend` and
  {doc}`../tutorials/topology`: three designs on 2D models.
- {py:obj}`gfem.conventions`: the equations of each polarization and the
  per-metre normalization.
- {doc}`prepare-a-mesh`: reading gmsh files, the cross-section reader included.
- {doc}`../verification`: how 2D models are checked.
