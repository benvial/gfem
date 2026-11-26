# GDS export of a design

An optimized layer-grid design is a pixel image. A foundry takes polygons: the
{term}`layer pattern` where the layer is solid, kept above its
{term}`minimum width` and {term}`minimum spacing`, in a GDS file. This page goes
from the filtered density to that file and solves the pattern again to confirm
it transmits what was promised, on the 2D bend of the
[bend tutorial](../tutorials/bend.ipynb) with a hand-drawn arc.

## The design

Core index 2.25 in a cladding of 1.444, two 0.55 µm guides at a right angle and
a 3 µm square between them (0.05 µm pixels, 0.3 µm filter radius). `rho` holds
a quarter ring joining the guides.

```python
import urllib.request

import numpy as np

import gfem
from gfem import design

url = (
    "https://raw.githubusercontent.com/benvial/gfem/main/docs/tutorials/data/bend2d.msh"
)
urllib.request.urlretrieve(url, "bend2d.msh")
mesh = gfem.Mesh.from_gmsh("bend2d.msh", length_unit="um")
um = 1e-6

grid = design.LayerGrid(mesh, "design", pitch=0.05 * um)
blur = design.helmholtz_filter(grid, radius=0.3 * um)

# A quarter ring 0.55 um wide of radius 1.5 um about the corner (-1.5, -1.5) um.
angle = np.linspace(0, np.pi / 2, 65)
unit = np.stack([np.cos(angle), np.sin(angle)], axis=1)
centre = np.array([-1.5, -1.5]) * um
ring = np.concatenate([centre + 1.775 * um * unit, centre + 1.225 * um * unit[::-1]])
rho = grid.area_fraction(design.LayerPattern.from_polygons([ring]))
```

## The pattern

```python
pattern = grid.pattern(blur(rho), eta=0.5)
print(pattern)
```

```text
LayerPattern('design', 1 polygon, 0 holes, 122 vertices, solid 14.4 % of 9e-12 m^2, eta=0.5)
```

- `grid.pattern` contours the **filtered** density `blur(rho)` at the projection
  threshold, the $\beta \to \infty$ edge of `design.project(blur(rho), beta)`,
  to a few hundredths of a pixel; a projected density only to about half a pixel.
- The polygons are in metres in the mesh's $(x, y)$ frame; solid reaching the
  footprint closes along `grid.bounds`, and a periodic pattern is one period.
- Each item is a {py:obj}`gfem.design.Polygon`, `(exterior, holes)`.
- Not differentiable: call it on the final design, outside `jax.grad`.

## Check the width and spacing

```python
check = pattern.check(min_width=0.15 * um, min_spacing=0.15 * um)
print(check)
print(check.passed)
```

```text
PatternCheck('design', min_width 150 nm: passed, narrowest 548.8 nm; min_spacing 150 nm: passed, no facing edges)
True
```

`check` measures facing edges perpendicular to an edge, as a foundry's rules do:
across solid for a width, across void for a spacing. It reports the narrowest
width and spacing even when the pattern passes, as the margin. The ring is
0.549 µm wide, against the 0.55 µm drawn; the void inside the square faces no
other edge, so it has no spacing. A broken rule lists each place, narrowest
first, with segments to draw (`violation.markers`):

```python
strict = pattern.check(min_width=0.6 * um)
for violation in strict.violations[:3]:
    x, y = violation.location / um
    print(f"{violation.kind} {violation.distance / um:.3f} um at ({x:.2f}, {y:.2f}) um")
```

```text
width 0.549 um at (-1.10, -0.06) um
```

The check is a pre-check, not foundry sign-off. Edges on the footprint and cut
edges at a periodic seam are not measured, and there is no minimum-area rule. A
foundry's checks may measure differently, and snapping to the GDS grid moves a
distance by up to $\sqrt2$ database units.

## Fixed geometry

Parts that are not optimized, such as feed guides or alignment marks, are
polygons too:

```python
guide_in = np.array([[-2.5, -0.275], [-1.5, -0.275], [-1.5, 0.275], [-2.5, 0.275]]) * um
guide_out = guide_in[:, ::-1]  # its mirror about y = x
guides = design.LayerPattern.from_polygons([guide_in, guide_out])
print(guides)
```

```text
LayerPattern(2 polygons, 0 holes, 8 vertices, area 1.1e-12 m^2)
```

`from_polygons` takes `(n, 2)` arrays in metres, or `design.Polygon(exterior,
holes)`. It closes and orients rings, and refuses, naming the ring, one that
crosses itself or another, a hole outside its polygon, and a vertex outside
`bounds`. A periodic pattern needs `bounds` and polygons cut at the seam.

## Write the GDS file

```python
path = design.write_gds("bend.gds", {(1, 0): (pattern, guides)})
print(path, path.stat().st_size)
```

```text
bend.gds 1178
```

- Each `(layer, datatype)` key holds a pattern or a sequence of them; layer
  numbers come from the foundry's layer map, so there is no default.
- One flat cell named after the file (`cell=`), in the mesh's frame.
- The user unit is 1 µm; vertices snap to a 1 nm grid (`database_unit=`).
- GDS has no holes: each joins its outer ring through a zero-width cut.
- The same arguments give the same bytes (stamped 1970-01-01 unless
  `timestamp=`). Under MPI, call it on rank 0.

Open the file in any GDS viewer; KLayout is a free one.

### Printed variants for review

A {term}`fabrication model` predicts what a process prints: a Gaussian blur of
standard deviation $\sigma$ and an {term}`edge bias` $b$ ($b < 0$ for an
over-etch). The printed patterns at $b \pm t$, for a tolerance $t$, can go into
the file on review datatypes beside the drawn layer:

```python
fab = design.FabricationModel(grid, sigma=0.1 * um, bias=-10e-9)  # an example
drawn = grid.area_fraction(pattern)  # the drawn design, exact, as pixels
eroded = fab.pattern(drawn, bias=fab.bias - 10e-9)  # b - t = -20 nm
dilated = fab.pattern(drawn, bias=fab.bias + 10e-9)  # b + t = 0
design.write_gds(
    "bend_review.gds", {(1, 0): (pattern, guides), (1, 10): eroded, (1, 11): dilated}
)
print(eroded.check(min_width=0.15 * um))
print(dilated.check(min_spacing=0.15 * um))
```

```text
PatternCheck('design', min_width 150 nm: passed, narrowest 511.3 nm)
PatternCheck('design', min_spacing 150 nm: passed, no facing edges)
```

- The review datatypes (10 and 11) are for the layout tool; the foundry
  receives only the drawn layer. The process numbers are examples to agree with
  the foundry.
- Check the width on the **eroded** pattern and the spacing on the **dilated**
  one. The eroded ring prints 511 nm wide, against 549 nm drawn and
  $2 \times 20$ nm of bias: each edge sits within about $0.01\sigma$ of where
  the bias puts it.
- The pitch here is $\sigma/2$; use $\sigma/4$ or finer for edges within a few
  hundredths of $\sigma$. The guides are not biased.

## Solve the pattern again

The pattern's area fractions go back to the mesh as any density does, and one
solve gives the transmission of the design as drawn:

```python
cladding = gfem.Material(eps_r=1.444**2)
core = gfem.Material(eps_r=2.25**2)
bend = gfem.Model(
    mesh,
    materials={"cladding": cladding, "core": core, "design": cladding},
    boundaries=[gfem.Absorbing("outer", order=2)],
    ports=[
        gfem.WavePort("p1", "port1", pec_edges=("outer",)),
        gfem.WavePort("p2", "port2", pec_edges=("outer",)),
    ],
)
eps = gfem.MaterialField.interpolate(
    grid.to_elements(grid.area_fraction(pattern)), cladding, core
)
s = gfem.solve(
    bend.with_material("design", eps),
    gfem.Driven(frequency=299792458.0 / 1.55e-6, polarization="Ez"),
    options=gfem.SolverOptions(linear_solver="direct"),
).s
print(f"|S21|^2 = {abs(complex(s['p2', 'p1'][0])) ** 2:.4f}")
```

```text
|S21|^2 = 0.6890
```

Compare it with the optimizer's last value. In the bend tutorial the drawn
pattern of the optimized design transmits 0.9762 against 0.9777 for the last
iterate at $\beta = 32$. They differ because the iterate is a finite-$\beta$
projection and the solve reads the pattern's pixels bilinearly; here the
difference stays within 0.02. A larger one means the chain from pixels to
polygons and back is broken. For the third digit, use a finer mesh or grid.

## Non-rectangular footprints, 2D models and sheets

- A pattern closes on the footprint's bounding rectangle. Set pixels outside a
  non-rectangular footprint void before the filter
  (`jnp.where(grid.sampled, rho, 0.0)`); the pattern can still reach past the
  outline by a fraction of the filter radius, so clip it in the layout tool.
- On a 2D model the pattern is the wafer's layer pattern when the model's plane
  is the wafer's; on a cross-section model it is the cross-section's profile.
- On a patterned metal sheet it is the film's layer pattern: take it from the
  pixels before `to_faces`.

## Next

{doc}`robust-optimization` (a fabrication model and minimum width and spacing
inside the optimization), the [bend tutorial](../tutorials/bend.ipynb),
{py:obj}`gfem.design.LayerPattern`, {py:obj}`gfem.design.write_gds`,
{py:obj}`gfem.design.FabricationModel`.
