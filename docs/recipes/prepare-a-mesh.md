# Mesh preparation

Build the mesh of your own device and solve on it. gfem reads a
[gmsh](https://gmsh.info) `.msh` file of tetrahedra whose
volumes and surfaces are grouped into named {term}`regions <region>` (gmsh's
physical groups); the model refers to the geometry only by those names.

The script rebuilds the line of {doc}`../get-started` (plates 40 mm long,
6 mm wide, 1 mm apart; ceramic $\varepsilon_r = 10$ over the middle 20 mm)
with the Python API from `pip install gmsh`. File coordinates are in mm
({doc}`../conventions`).

## Build and name the regions

Three boxes along x: feed, ceramic, feed. `fragment` makes neighbouring
volumes share one surface mesh, which gfem needs:

```python
import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.NumThreads", 1)

occ = gmsh.model.occ
feed_in = occ.addBox(0, 0, 0, 10, 6, 1)  # x, y, z, then dx, dy, dz in mm
design = occ.addBox(10, 0, 0, 20, 6, 1)
feed_out = occ.addBox(30, 0, 0, 10, 6, 1)
occ.fragment([(3, feed_in)], [(3, design), (3, feed_out)])
occ.synchronize()
```

Each volume that takes its own material is one physical group, named. Both
feeds are air, so they form one region, `feed`:

```python
gmsh.model.addPhysicalGroup(3, [feed_in, feed_out], name="feed")
gmsh.model.addPhysicalGroup(3, [design], name="design")
```

Name every surface the model refers to: the two plates and the two port
ends, found here by the plane they lie in:

```python
def faces_in_plane(axis, value):
    """The surfaces that lie in the plane x, y or z (axis 0, 1, 2) = value."""
    found = []
    for _, tag in gmsh.model.getEntities(2):
        box = gmsh.model.getBoundingBox(2, tag)  # xmin, ymin, zmin, xmax, ...
        if abs(box[axis] - value) < 1e-6 and abs(box[axis + 3] - value) < 1e-6:
            found.append(tag)
    return found


gmsh.model.addPhysicalGroup(2, faces_in_plane(2, 0), name="bottom")
gmsh.model.addPhysicalGroup(2, faces_in_plane(2, 1), name="top")
gmsh.model.addPhysicalGroup(2, faces_in_plane(0, 0), name="port1")
gmsh.model.addPhysicalGroup(2, faces_in_plane(0, 40), name="port2")
```

The side walls ($y = 0$ and $y$ = 6 mm) stay unnamed: an unnamed outer surface
is a magnetic wall, the side condition of the get-started line. Name a surface
to give it another condition.

Mesh and write the file. The largest edge is the plate spacing, 1 mm, about a
twelfth of the shortest wavelength in the ceramic (11.9 mm at 8 GHz):

```python
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)  # mm
gmsh.model.mesh.generate(3)
gmsh.write("my_line.msh")
gmsh.finalize()
```

gmsh writes first-order tetrahedra by default; formats 4.1 and 2.2 both read.

## Read it and check the names

```python
import gfem

mesh = gfem.Mesh.from_gmsh("my_line.msh", length_unit="mm")
print(mesh.volumes)
print(mesh.surfaces)
print(len(mesh.tetrahedra), "tetrahedra")
print(mesh.imported_nodes.max(axis=0))  # m: the far corner of the line
```

```text
('feed', 'design')
('bottom', 'top', 'port1', 'port2')
2220 tetrahedra
[0.04  0.006 0.001]
```

- `length_unit` is the unit of the file's coordinates, one of `gfem.mesh.LENGTH_UNITS`
  (default `"um"`, what gsim writes). A wrong unit shifts every length by a
  power of ten: the far corner above should read 40 mm, 6 mm, 1 mm.
- `mesh.volumes` and `mesh.surfaces` are the only names the model can use. A
  misspelt name raises {py:obj}`gfem.RegionNotFoundError` with the closest
  names.

## Solve on it

The model of {doc}`../get-started`, word for word, runs on this mesh:

```python
z_line = 376.73 / 6  # ohm: the 6 mm wide, 1 mm high parallel-plate line in air
model = gfem.Model(
    mesh,
    materials={"feed": gfem.Material(), "design": gfem.Material(eps_r=10.0)},
    boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
    ports=[
        gfem.LumpedPort("p1", "port1", "+z", impedance=z_line),
        gfem.LumpedPort("p2", "port2", "+z", impedance=z_line),
    ],
)
at_3 = gfem.Driven(frequency=(2e9, 5e9, 8e9), excite="p1")
s21 = gfem.solve(model, at_3).s["p2", "p1"]
print(abs(s21))  # |S21| at 2, 5 and 8 GHz
```

```text
[0.83062229 0.90133319 0.6051741 ]
```

These are the get-started values, 0.8306, 0.9013 and 0.6052, to the fourth
digit, on a different mesh of the same line.

## Is it fine enough?

Raise the {term}`element order` on the same mesh and solve again. A result
the mesh resolves does not move:

```python
finer = gfem.solve(model, at_3, options=gfem.SolverOptions(order=3))
print(abs(finer.s["p2", "p1"]))
print(abs(finer.s["p2", "p1"] - s21).max())  # the largest change of S21
```

```text
[0.83062279 0.90133387 0.60518538]
1.1851208817813629e-05
```

S21 changes by about 1e-5, most at 8 GHz, where the wavelength is shortest.
If that is too much, refine where the field varies fastest: in gmsh
(`"Mesh.MeshSizeMax"`, or a size field near edges and gaps) or with gfem's
own refinement ({doc}`mesh-refinement`).

## Curved walls

Straight tetrahedra facet a curved wall, so a capacitance or a resonance
improves only as $h^2$ with the mesh size $h$, whatever the
{term}`element order`: the {term}`geometry floor`. Set
`gmsh.option.setNumber("Mesh.ElementOrder", 2)` before `generate`, and gfem
reads a {term}`curved mesh`, whose mid-edge nodes lie on the wall: at orders 2
and 3 the error falls as $h^4$ ({ref}`verification <verification-curved-walls>`).

```python
coax = gfem.Mesh.from_gmsh("coax.msh", length_unit="mm")
print(coax.curved)
```

```text
True
```

Features that compute geometry from element corners (ports, periodic walls,
layers, incident fields, field reads; {doc}`../capabilities`, Limits) run where
their own elements are straight, as on a flat face with straight edges, and
raise {py:obj}`gfem.CurvedMeshUnsupportedError` elsewhere; so does a curved 2D
mesh. Keep gmsh's `"Mesh.HighOrderOptimize"` option at 0 or 1 so that elements off the wall
stay straight.

## Metals: a surface or an excluded volume

A thin metal whose thickness you do not resolve is a named surface with a
{py:obj}`gfem.PEC` or {py:obj}`gfem.Conductor` condition, as the plates
above. A thick metal meshed as a volume leaves the solve domain with
`Model(excluded=...)` and its outer faces carry the condition. Name the
volume (`excluded=("trace",)`) or, when the mesher tags it with the
surrounding dielectric (gsim does), take the volume enclosed by its faces,
{py:obj}`gfem.enclosed_by`. Here `chip.msh` is a substrate `sub` under `air`
with a trace inside the substrate whose faces are `metal_xy` (top and bottom)
and `metal_z` (sides):

```python
chip = gfem.Mesh.from_gmsh("chip.msh", length_unit="um")
print(chip.volumes)
print(chip.surfaces)

shell = ["metal_xy", "metal_z"]
on_chip = gfem.Model(
    chip,
    materials={"sub": gfem.Material(eps_r=11.9), "air": gfem.Material()},
    boundaries=[gfem.Conductor(face, 5.8e7) for face in shell],
    excluded=[gfem.enclosed_by(*shell)],
)
```

```text
('sub', 'air')
('metal_xy', 'metal_z', 'air__sub', 'sub__None', 'air__None')
```

A {term}`lumped port` sits on a surface that spans from one conductor to the
other, as `port1` does. Give it a face and a name of its own: a port cannot
share a surface with another condition.

## Look at the mesh

`mesh.to_vtu` writes the mesh, with each element's region number, for
ParaView:

```python
mesh.to_vtu("my_line.vtu")
```

Colour by region in ParaView to see each volume where you meant it; `gmsh
my_line.msh` shows the physical groups too.

## What a bad mesh raises

{py:obj}`gfem.Mesh.from_gmsh` checks the file before any model is built and
raises `ValueError` naming the file and the remedy:

- No physical groups at all: group and name the volumes and surfaces, then
  export again.
- Order 3 and above, or mixed orders: set `"Mesh.ElementOrder"` to 1 or 2.
- Hexahedra, prisms or pyramids: gfem reads tetrahedra only, with
  triangles on surfaces.
- Tetrahedra outside any volume group: add every volume to a group.
- A physical group without a name: name every group.
- An element in two groups of the same dimension, a surface triangle that is
  not a face of any tetrahedron, or overlapping tetrahedra: split the groups
  or remesh so the surface and volume meshes conform.

## Limits

- gfem reads no CAD file: geometry, meshing and region
  names are gmsh's, or gsim's ({doc}`gsim`).
- The whole mesh is read into memory on every process.

## Next

- {doc}`solve-a-device`: S-parameters over a band, resonances and Q, and
  capacitance on this line.
- {doc}`mesh-refinement`: refine the mesh where the error is.
- {doc}`gsim`: a mesh and its configuration from a GDS layout.
- {doc}`shape-gradient`: move the mesh's nodes to change the shape, and
  keep its elements well shaped.
- {doc}`cad-parameter-gradient`: to differentiate with respect to the script's
  dimensions, write it as a build function.
