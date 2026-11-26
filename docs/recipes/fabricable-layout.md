# GDS export of a design

An optimized layer-grid design is a pixel image. A foundry takes polygons: the
{term}`layer pattern` where the layer is solid, kept above its
{term}`minimum width` and {term}`minimum spacing`, in a GDS file. This page goes
from the filtered density to that file, and solves the pattern again to confirm
it transmits what the optimization promised. The example is the 2D waveguide
bend of the [bend tutorial](../tutorials/bend.ipynb), with its hand-drawn arc
standing in for an optimized design.

## The design

A 2D model in the wafer plane: core index 2.25 in a cladding of 1.444, two guides
0.55 µm wide meeting at a right angle, and a 3 µm × 3 µm design square between
them, laid with 0.05 µm pixels and filtered at a 0.3 µm radius. `rho` holds the
pixels of a quarter ring joining the two guides.

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

- `grid.pattern` contours the **filtered** density, `blur(rho)`, at the
  projection threshold: the edge of the $\beta \to \infty$ limit of
  `design.project(blur(rho), beta)`, placed between pixel centres to a few
  hundredths of a pixel. A projected density works too, but only to about half
  a pixel, since its values saturate at the pixel centres.
- The polygons are in metres in the mesh's $(x, y)$ frame. Solid that reaches the
  footprint closes along `grid.bounds`, the rectangle of the design region;
  on a periodic axis the pattern is one period, cut at the seam.
- Each item is a {py:obj}`gfem.design.Polygon`, `(exterior, holes)`: closed
  `(n, 2)` rings, the outer one counter-clockwise and the holes clockwise, so
  `ax.plot(*exterior.T)` draws one whole.
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

`check` measures facing edges of the polygons perpendicular to an edge, as a
foundry's width and space rules do: across solid for a width, across void for a
spacing. It reports the narrowest width and spacing even when the pattern
passes, as the margin. The ring is 0.549 µm wide, against the 0.55 µm drawn; the
void inside the design square faces no other edge of the pattern, so it has no
spacing to measure. A rule the pattern breaks lists each place, narrowest
first, with segments to draw (`ax.plot(*marker.T)` for each of
`violation.markers`):

```python
strict = pattern.check(min_width=0.6 * um)
for violation in strict.violations[:3]:
    x, y = violation.location / um
    print(f"{violation.kind} {violation.distance / um:.3f} um at ({x:.2f}, {y:.2f}) um")
```

```text
width 0.549 um at (-1.10, -0.06) um
```

The check is a pre-check, not foundry sign-off. Edges on the footprint are not
measured, since what lies beyond them is the rest of the layout; nor are the
cut edges at a periodic seam, across which features are measured whole. There is
no minimum-area rule. A foundry's own checks may measure differently (along the shortest line on
Manhattan layouts), and snapping to the GDS grid moves a distance by up to
$\sqrt2$ database units.

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
holes)` for a polygon with holes. It closes open rings, orients them and drops
repeated vertices, and refuses, naming the polygon and the ring, a ring that
crosses itself or another, a hole outside its polygon, a polygon inside
another's solid, and a vertex outside `bounds` when given. A periodic pattern
needs `bounds` and polygons already cut at the seam.

## Write the GDS file

```python
path = design.write_gds("bend.gds", {(1, 0): (pattern, guides)})
print(path, path.stat().st_size)
```

```text
bend.gds 1178
```

- Each `(layer, datatype)` key holds a pattern or a sequence of them. Layer
  numbers come from the foundry's layer map, so there is no default.
- One flat cell, named after the file (`cell=` to change it). The polygons keep
  the mesh's frame, so the cell placed at the origin of the layout that built the
  mesh abuts the parts drawn there; here the guides end on the design square's
  edge.
- The user unit is 1 µm and every vertex snaps to a 1 nm database grid
  (`database_unit=` for a foundry's manufacturing grid).
- GDS has no holes: each hole joins its outer ring through a zero-width cut,
  which a layout tool's merge removes.
- The same arguments give the same bytes: the file is stamped 1970-01-01
  unless `timestamp=` says otherwise.
- Not collective: under MPI, call it on rank 0.

Open the file in any GDS viewer; KLayout is a free one.

### Printed variants for review

A {term}`fabrication model` predicts what a process prints: a Gaussian blur of
standard deviation $\sigma$ and an {term}`edge bias` $b$ that moves straight
edges outwards ($b < 0$ for an over-etch). The printed patterns at $b \pm t$,
for a bias tolerance $t$, can go into the file on review datatypes beside the
drawn layer:

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

- The review datatypes (10 and 11 here) are for looking at in the layout tool;
  the foundry receives only the drawn layer, datatype 0. The process numbers and
  the datatypes are examples to agree with the foundry.
- Check the width on the **eroded** pattern and the spacing on the **dilated**
  one: each is where its rule is tightest. The eroded ring prints 511 nm wide,
  against 549 nm drawn and $2 \times 20$ nm of bias: each edge sits within about
  $0.01\sigma$ of where the bias puts it.
- `fab.pattern` reads the drawn design as pixels. The area fractions of the drawn
  pattern place its edges to about 0.015 pixel; a $\beta = 32$ projection of
  `blur(rho)` would do it to about 0.2 pixel.
- The pitch here is $\sigma/2$. For printed edges within a few hundredths of
  $\sigma$, lay the grid at $\sigma/4$ or finer.
- The guides are not biased: a printed variant steps by $b$ per side against
  them.

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
iterate at $\beta = 32$. The two differ because the iterate is a finite-$\beta$
projection, not a sharp edge, and the solve reads the pattern's pixels
bilinearly, so the permittivity stays fractional along the edges; on this bend
such differences stay within 0.02. A larger difference means the chain from pixels to polygons
and back is broken. To confirm the third digit, solve on a finer mesh or a finer
grid.

## Non-rectangular footprints, 2D models and sheets

- A pattern closes on the footprint's bounding rectangle. Set pixels outside a
  non-rectangular footprint void before the filter
  (`jnp.where(grid.sampled, rho, 0.0)`); where solid meets the real outline the
  pattern can still reach past it by a fraction of the filter radius, so draw such
  a region only where nothing fixed lies beyond it, or clip it in the layout tool.
- On a 2D model the pattern is the wafer's layer pattern when the model's plane
  is the wafer's, as for this bend. On a cross-section model it is the profile of the cross-section.
- On a patterned metal sheet the pattern is the film's layer pattern: take it
  from the pixels before `to_faces`.

## Next

{doc}`robust-optimization` (a fabrication model and minimum width and spacing
inside the optimization), the [bend tutorial](../tutorials/bend.ipynb),
{py:obj}`gfem.design.LayerPattern`, {py:obj}`gfem.design.write_gds`,
{py:obj}`gfem.design.FabricationModel`.
