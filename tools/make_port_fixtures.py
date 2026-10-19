"""Build the small gmsh fixtures for gfem's port mode tests.

Usage: python tools/make_port_fixtures.py tests/unit/data   (needs the gmsh
Python package, which gfem itself does not depend on; tested with 4.15.2).

All three are straight waveguide sections along z with a port face at each
end, so their port modes are known in closed form:

- ``waveguide.msh`` (mm): empty rectangular guide [0, 22.86] x [0, 10.16] x
  [0, 6] (WR-90). Volume "air" (1); surfaces "port1" (z = 0, 2), "port2"
  (z = 6, 3), "walls" (the four side walls, 4).
- ``slab_waveguide.msh`` (mm): the same guide with a dielectric slab
  x in [0, 8] filling the full height. Volumes "slab" (1) and "air" (2);
  surfaces "port1" (3), "port2" (4), "walls" (5).
- ``parallel_plate.msh`` (um): plates y = 0 and y = 10 ("plates", 3), open
  sides x = 0 and x = 40 ("sides", 4), ports at z = 0 ("port1", 1) and
  z = 20 ("port2", 2). Volume "fill" (5).
"""

import sys

import gmsh

out = sys.argv[1]


def start():
    """Open a fresh deterministic gmsh model."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.RandomSeed", 1)


def finish(name, size):
    """Mesh uniformly at ``size`` and write ``name`` (msh 4.1 ASCII)."""
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(f"{out}/{name}")
    gmsh.finalize()


def faces_by_side(length):
    """Split the outer faces into z = 0, z = length and the rest."""
    lo, hi, rest = [], [], []
    for _, t in gmsh.model.getEntities(2):
        z = gmsh.model.occ.getCenterOfMass(2, t)[2]
        if abs(z) < 1e-9:
            lo.append(t)
        elif abs(z - length) < 1e-9:
            hi.append(t)
        else:
            rest.append(t)
    return lo, hi, rest


a, b, length = 22.86, 10.16, 6.0

start()
box = gmsh.model.occ.addBox(0, 0, 0, a, b, length)
gmsh.model.occ.synchronize()
lo, hi, rest = faces_by_side(length)
gmsh.model.addPhysicalGroup(3, [box], 1, name="air")
gmsh.model.addPhysicalGroup(2, lo, 2, name="port1")
gmsh.model.addPhysicalGroup(2, hi, 3, name="port2")
gmsh.model.addPhysicalGroup(2, rest, 4, name="walls")
finish("waveguide.msh", 4.0)

start()
slab = gmsh.model.occ.addBox(0, 0, 0, 8.0, b, length)
air = gmsh.model.occ.addBox(8.0, 0, 0, a - 8.0, b, length)
gmsh.model.occ.fragment([(3, slab)], [(3, air)])
gmsh.model.occ.synchronize()
inner = set()
for _, t in gmsh.model.getEntities(2):
    up, _ = gmsh.model.getAdjacencies(2, t)
    if len(up) == 2:
        inner.add(t)
lo, hi, rest = faces_by_side(length)
gmsh.model.addPhysicalGroup(3, [slab], 1, name="slab")
gmsh.model.addPhysicalGroup(3, [air], 2, name="air")
gmsh.model.addPhysicalGroup(2, lo, 3, name="port1")
gmsh.model.addPhysicalGroup(2, hi, 4, name="port2")
gmsh.model.addPhysicalGroup(2, [t for t in rest if t not in inner], 5, name="walls")
finish("slab_waveguide.msh", 2.5)

w, d, length = 40.0, 10.0, 20.0
start()
box = gmsh.model.occ.addBox(0, 0, 0, w, d, length)
gmsh.model.occ.synchronize()
lo, hi, rest = faces_by_side(length)
plates, sides = [], []
for t in rest:
    y = gmsh.model.occ.getCenterOfMass(2, t)[1]
    (plates if abs(y) < 1e-9 or abs(y - d) < 1e-9 else sides).append(t)
gmsh.model.addPhysicalGroup(2, lo, 1, name="port1")
gmsh.model.addPhysicalGroup(2, hi, 2, name="port2")
gmsh.model.addPhysicalGroup(2, plates, 3, name="plates")
gmsh.model.addPhysicalGroup(2, sides, 4, name="sides")
gmsh.model.addPhysicalGroup(3, [box], 5, name="fill")
finish("parallel_plate.msh", 10.0)
