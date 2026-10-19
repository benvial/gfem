"""Build the tiny gmsh fixtures for gfem's mesh-import unit tests.

Usage: python tools/make_mesh_fixtures.py tests/unit/data   (needs the gmsh
Python package, which gfem itself does not depend on; tested with 4.15.2).

Geometry (um): substrate box [0,2]^2 x [0,1] ("sub", tag 1) holding a small
conductor box [0.5,1.5]^2 x [0.3,0.7] that is ALSO tagged "sub" (gsim tags
meshed conductor volumes with a dielectric name), and an air box
[0,2]^2 x [1,2] ("air", tag 2). Surfaces: conductor top/bottom "metal_xy" (4),
conductor sides "metal_z" (5), interface "air__sub" (7), exterior
"sub__None" (8) and "air__None" (9). Tags 3 and 6 are unused on purpose.
"""

import sys

import gmsh

out = sys.argv[1]
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.NumThreads", 1)
gmsh.option.setNumber("Mesh.RandomSeed", 1)
occ = gmsh.model.occ
sub = occ.addBox(0, 0, 0, 2, 2, 1)
air = occ.addBox(0, 0, 1, 2, 2, 1)
met = occ.addBox(0.5, 0.5, 0.3, 1, 1, 0.4)
occ.fragment([(3, sub), (3, air)], [(3, met)])
occ.synchronize()

vol_of = {}
for _, t in gmsh.model.getEntities(3):
    c = occ.getCenterOfMass(3, t)
    mass = occ.getMass(3, t)
    if c[2] > 1:
        vol_of["air"] = t
    elif mass < 0.5:
        vol_of["metal"] = t
    else:
        vol_of["sub"] = t
gmsh.model.addPhysicalGroup(3, [vol_of["sub"], vol_of["metal"]], 1, name="sub")
gmsh.model.addPhysicalGroup(3, [vol_of["air"]], 2, name="air")


def bnd(v):
    """Surface entities bounding volume entity ``v``."""
    return {t for _, t in gmsh.model.getBoundary([(3, v)], oriented=False)}


bs, ba, bm = bnd(vol_of["sub"]), bnd(vol_of["air"]), bnd(vol_of["metal"])
metal_xy, metal_z = [], []
for t in bm:
    n = gmsh.model.getNormal(t, [0.5, 0.5])
    (metal_xy if abs(abs(n[2]) - 1) < 1e-9 else metal_z).append(t)
gmsh.model.addPhysicalGroup(2, sorted(metal_xy), 4, name="metal_xy")
gmsh.model.addPhysicalGroup(2, sorted(metal_z), 5, name="metal_z")
gmsh.model.addPhysicalGroup(2, sorted(bs & ba), 7, name="air__sub")
gmsh.model.addPhysicalGroup(2, sorted(bs - ba - bm), 8, name="sub__None")
gmsh.model.addPhysicalGroup(2, sorted(ba - bs), 9, name="air__None")

gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.4)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.Binary", 0)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write(f"{out}/two_box_msh22.msh")
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.write(f"{out}/two_box_msh41.msh")
gmsh.finalize()

# Second-order fixture: one coarse unit cube, "box" (1) bounded by "wall" (2).
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("Mesh.RandomSeed", 1)
box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [box], 1, name="box")
walls = [t for _, t in gmsh.model.getBoundary([(3, box)], oriented=False)]
gmsh.model.addPhysicalGroup(2, walls, 2, name="wall")
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeMin", 1.0)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.setOrder(2)
gmsh.option.setNumber("Mesh.Binary", 0)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write(f"{out}/cube_order2_msh22.msh")
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.write(f"{out}/cube_order2_msh41.msh")
gmsh.finalize()

# A face in two physical groups: "wall" (2) holds every face, "port" (3) one.
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("Mesh.RandomSeed", 1)
box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [box], 1, name="box")
walls = [t for _, t in gmsh.model.getBoundary([(3, box)], oriented=False)]
gmsh.model.addPhysicalGroup(2, walls, 2, name="wall")
gmsh.model.addPhysicalGroup(2, walls[:1], 3, name="port")
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeMin", 1.0)
gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.Binary", 0)
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write(f"{out}/cube_overlap_msh22.msh")
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.write(f"{out}/cube_overlap_msh41.msh")
gmsh.finalize()
