# Mesh preparation

Build the mesh of your own device and solve on it. gfem builds no geometry: it
reads a [gmsh](https://gmsh.info) `.msh` file of straight-sided tetrahedra
whose volumes and surfaces are grouped into named
{term}`regions <region>` (gmsh's physical groups). The model refers to the
geometry only by those names: a material per volume, a boundary condition or
a port per surface.

This page builds the line of {doc}`../get-started` from scratch: two parallel
plates 40 mm long, 6 mm wide and 1 mm apart, with a ceramic of relative
permittivity 10 over the middle 20 mm. It then reads the mesh, solves the
get-started model on it unchanged and checks that the mesh resolves the
field. The coordinates in the file are in mm; gfem converts them to metres
once, when it reads the file, and works in SI units from there on
({doc}`../conventions`).

gfem does not install gmsh: `pip install gmsh` gives its Python API, which the
script below uses.

## Build and name the regions

Draw the line as three boxes along x, in mm: a feed, the ceramic and a second
feed. `fragment` splits them along their common faces, so neighbouring
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

Name every surface the model will refer to: the two plates, which will be
perfect conductors, and the two ends, which will carry the ports. Here the
surfaces are found by the plane they lie in:

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

The two side walls ($y = 0$ and $y$ = 6 mm) stay unnamed. An outer surface that
the model does not name is a magnetic wall, where the tangential magnetic
field vanishes. That is the side condition of the get-started line, so the
sides need no name. Name a surface when you want another condition on it.

Then mesh and write the file. The largest element edge is set to the plate
spacing, 1 mm, about a twelfth of the shortest wavelength in the ceramic
(11.9 mm at 8 GHz):

```python
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)  # mm
gmsh.model.mesh.generate(3)
gmsh.write("my_line.msh")
gmsh.finalize()
```

gmsh writes first-order (straight-sided) tetrahedra unless you ask for more,
and its default `.msh` format 4.1 is one gfem reads; format 2.2 works too.

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

- `length_unit` is the unit of the file's coordinates: one of
  `"m"`, `"cm"`, `"mm"`, `"um"` or `"nm"` (`gfem.mesh.LENGTH_UNITS`). The
  default is `"um"`, the unit gsim writes. Get it wrong and every length is
  off by a power of ten: the far corner above should read 40 mm, 6 mm, 1 mm.
- `mesh.volumes` and `mesh.surfaces` are the names the model can use, and
  only those. A misspelt name in the model raises
  {py:obj}`gfem.RegionNotFoundError`, which suggests the closest names the
  mesh has.

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
digit, on a different mesh of the same line. The two meshes agree on what
the model uses: the names it refers to, the unit and the geometry.

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

S21 changes by about 1e-5, most at 8 GHz, where the wavelength is
shortest. When the change is larger than you can accept, make the mesh finer
where the field varies fastest: in gmsh (`"Mesh.MeshSizeMax"`, or a size
field near edges and gaps), or with gfem's own refinement, which starts from
this mesh ({doc}`mesh-refinement`).

## Metals: a surface or an excluded volume

A thin metal, a plate or a trace whose thickness you do not need to resolve,
is a named surface with a {py:obj}`gfem.PEC` or {py:obj}`gfem.Conductor`
condition, as the plates above. A thick metal that you mesh as a volume is
taken out of the solve domain with `Model(excluded=...)`, and its outer
faces carry the condition. The volume is either a region of its own,
`excluded=("trace",)`, or, when the mesher tags it with the name of the
dielectric around it (gsim does), the volume enclosed by its named faces,
{py:obj}`gfem.enclosed_by`. Say `chip.msh` is your own mesh of a substrate
`sub` under a volume of `air`, with a meshed trace inside the substrate whose
faces are `metal_xy` (top and bottom) and `metal_z` (sides):

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
other, as `port1` spans from the bottom plate to the top one. Make that
surface a face of its own in gmsh and give it a name: a port cannot share a
surface with another condition.

## Look at the mesh

`mesh.to_vtu` writes the mesh, with each element's region number, for
ParaView:

```python
mesh.to_vtu("my_line.vtu")
```

Open the file and colour by region to see that each volume is where you meant
it. gmsh's own viewer shows the physical groups too (`gmsh my_line.msh`).

## What a bad mesh raises

{py:obj}`gfem.Mesh.from_gmsh` checks the file before any model is built and
raises `ValueError` naming the file and the remedy:

- No physical groups at all: group and name the volumes and surfaces, then
  export again.
- A second-order (curved) mesh: regenerate it with
  `gmsh.option.setNumber("Mesh.ElementOrder", 1)`.
- Hexahedra, prisms or pyramids: gfem reads tetrahedra only, with
  triangles on surfaces.
- Tetrahedra outside any volume group: add every volume to a group.
- A physical group without a name: name every group.
- An element in two groups of the same dimension, a surface triangle that is
  not a face of any tetrahedron, or overlapping tetrahedra: split the groups
  or remesh so the surface and volume meshes conform.

## Limits

- Straight-sided tetrahedra only: no curved elements, no hexahedra or
  prisms. A curved surface is as round as its facets; refinement in gfem
  subdivides them and never moves them onto the curve, so mesh curved
  surfaces finer in gmsh ({doc}`mesh-refinement`, Limits).
- gfem generates no mesh and reads no CAD file: geometry, meshing and region
  names are gmsh's, or gsim's ({doc}`gsim`).
- The whole mesh is read into memory on every process.

## Next

- {doc}`solve-a-device`: S-parameters over a band, resonances and Q, and
  capacitance on this line.
- {doc}`mesh-refinement`: refine the mesh where the error is.
- {doc}`gsim`: a mesh and its configuration from a GDS layout.
- {doc}`shape-gradient`: move the mesh's nodes to change the shape, and
  keep its elements well shaped.
