# 2D models

When nothing varies along z (a bend in effective indices, a parallel-plate
line, a grating's cross-section), gfem solves the plane alone as a
{term}`2D model`: one field component, $E_z$ or $H_z$, per metre along z.

## A 2D model, a slab or a 3D model

- **2D model:** waves in the $(x, y)$ plane, one polarization,
  {term}`Ez polarization` or {term}`Hz polarization`; they do not mix.
- **Slab:** the plane extruded one element thick
  ({py:meth}`gfem.Mesh.from_gmsh_cross_section`, {term}`slab model`); it takes
  what a 2D model refuses.
- **3D model:** anything varies along z.

## The mesh

Mesh the plane in gmsh with triangles only, read with
{py:meth}`gfem.Mesh.from_gmsh`:

- a physical surface is an area (a material);
- a physical curve is a line (a boundary condition or a port).

```python
import numpy as np
from scipy.constants import c

import gfem

mesh = gfem.Mesh.from_gmsh("bend2d.msh", length_unit="um")
print(mesh.dim)  # 2: a plane of triangles
print(mesh.volumes, mesh.surfaces)  # the areas, then the lines
```

`mesh.nodes` holds $(x, y)$ in m.

## Solve one polarization

The bend of {doc}`../tutorials/bend`: core and cladding areas, an absorbing
outer line and two {term}`line ports <Line port>`. Every driven problem on a
2D model states its polarization:

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
  `result.h_field(f0, excite="p1")`. The other component raises `ValueError`
  naming the right one.
- `ez.to_vtu("bend.vtu")` writes the field for ParaView.
- S follows the electric field in both polarizations.

## Line ports

A {py:obj}`gfem.WavePort` on a straight exterior line is a line port, with the
mode of the line's cross-section. {py:obj}`gfem.PortMode` solves it alone:

```python
mode = gfem.solve(bend, gfem.PortMode("p1", f0, polarization="Ez"))
print(mode.n_eff)  # the effective index of the port's fundamental mode
```

- A port line is straight and on the outside of the mesh.
- `pec_edges` makes the ends of a port line conducting. In Hz such a line must
  be a `PEC`, `Absorbing` or port line: on the default PMC the solve raises.
- A line port carries 1 W/m and needs no reference impedance: a `voltage_path`
  raises and the mode's $Z_{PV}$ is NaN.
- Lumped ports raise: feed a line with a line port, as the
  {doc}`topology tutorial <../tutorials/topology>` does.

## Per metre along z

Each excited port is fed 1 W/m of mode power. S is a ratio of powers, the same
as a slab's; a field is the field for 1 W/m, so a slab of thickness $h$ fed
1 W relates to it by $E^{\mathrm{2D}} = \sqrt{h}\,E^{\mathrm{slab}}$
({py:obj}`gfem.conventions`, "2D models").

## Conductors in each polarization

An exterior line no boundary names is a PMC.

| Line | Ez polarization | Hz polarization |
| --- | --- | --- |
| `PEC` | $E_z = 0$ | $\partial_n H_z = 0$ |
| `PMC`, or unnamed | $\partial_n E_z = 0$ | $H_z = 0$ |

- **Ez.** A metal is a `PEC` line, interior or exterior, or an area with a
  conducting or Drude {py:obj}`gfem.Material`.
- **Hz.** A metal is a hole in the plane: leave it out of the mesh or name its
  area in `excluded=`, and name the hole's edge `PEC`; an unnamed edge is a
  PMC, a wrong wall. A `PEC` on an interior line raises, since a thin strip
  would pass the wave: cut the mesh along it or exclude the area.
- A `PMC` on an interior line raises in Ez, for the same reason.
- {py:obj}`gfem.Conductor` and {py:obj}`gfem.Impedance` surfaces raise.

## Absorbing boundaries

{py:obj}`gfem.Absorbing` works in both polarizations at first or second order;
the second absorbs oblique waves better. A slab in Hz has no such gain, so a
2D model in Hz is the better-closed problem.

## Designs and gradients

A design is written as on a 3D model ({doc}`robust-optimization`):

```python
from gfem import design

grid = design.LayerGrid(mesh, "design", pitch=0.05e-6)
drude = gfem.Drude(plasma_frequency=2.2e15, damping=1.3e13)  # Hz
gold = gfem.Material(eps_r=gfem.PermittivityLaw(terms=[drude]))


def material(rho):
    density = grid.to_elements(rho)  # the pixels, bound to the grid
    return gfem.MaterialField.interpolate(density, cladding, gold, scheme="index")
```

- The solve samples the pixels' bilinear interpolation inside each triangle,
  so a material edge between pixels is not rounded to whole triangles.
- `grid.to_elements(rho)` is bound to the grid, not one value per triangle:
  `np.asarray` of it raises. Plot `rho` on `grid.x` and `grid.y` with the 0.5
  contour as the material edge.
- `jax.grad` of any S functional goes through the solve to the pixels, in both
  polarizations. A design at a line port differentiates too, through the
  port's mode, for `PortMode`'s `n_eff` and field values as well. Node
  coordinates, law parameters and boundary values are not differentiated.
- Hz takes blends of constant permittivities only: a law or a conductivity in
  a blend raises. A {py:obj}`gfem.Material` region with one is accepted.

## Check your results

- **Port mode.** `PortMode(...).n_eff` matches the symmetric slab guide's
  closed form, as in the {doc}`bend tutorial <../tutorials/bend>`.
- **Power balance.** Lossless: $|S_{11}|^2 + |S_{21}|^2 \le 1$, $S_{12} = S_{21}$.
- **Element order.** `gfem.SolverOptions(order=3)` again; the change in S bounds
  the discretization error.
- **Slab.** In Ez the same plane as a slab gives the same S; in Hz expect a few
  $10^{-3}$ ({doc}`../verification`).

## What raises

A 2D model raises `NotImplementedError` for these, naming the alternative,
most often the slab:

- eigenmode, electrostatic, DC conduction and thermal problems;
- oblique incidence (`wave_vector=`), periodic pairs, Floquet ports;
- lumped ports, conductor and impedance surfaces, sheet designs;
- anisotropic materials and PMLs ({doc}`piezoelectric-admittance` takes the
  former);
- the adaptive sweep (use `sweep="uniform"`), iterative and GPU solvers,
  multi-process solves;
- power flow, energies, participations, characteristic impedance, region means
  (read `Field.at` instead);
- mesh refinement, error estimates, moved nodes (change the mesh in gmsh;
  {doc}`piezoelectric-gradient` takes node maps).

## Next

- {doc}`../tutorials/bend`, {doc}`../tutorials/metal-bend` and
  {doc}`../tutorials/topology`: three designs on 2D models.
- {py:obj}`gfem.conventions`: the equations of each polarization and the
  per-metre normalization.
- {doc}`prepare-a-mesh`: reading gmsh files, the cross-section reader included.
- {doc}`../verification`: how 2D models are checked.
