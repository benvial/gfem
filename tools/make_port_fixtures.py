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


def extrude_section(faces, length):
    """Extrude 2D faces at z = 0 by one element layer; return the volumes."""
    out = gmsh.model.occ.extrude([(2, f) for f in faces], 0, 0, length, [1])
    gmsh.model.occ.synchronize()
    return [t for d, t in out if d == 3]


def mesh_section(name, size_of):
    """Mesh with element size ``size_of(x, y)`` and write ``name``."""
    gmsh.model.mesh.setSizeCallback(lambda _dim, _tag, x, y, _z, _lc: size_of(x, y))
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.write(f"{out}/{name}")
    gmsh.finalize()


def group(volumes_by_name, length):
    """Volume groups, then "port1" (z = 0) and "port2" (z = length).

    Returns the outer side faces (one adjacent volume) and the next free tag.
    """
    tag = 1
    for name, vols in volumes_by_name.items():
        gmsh.model.addPhysicalGroup(3, vols, tag, name=name)
        tag += 1
    lo, hi, rest = faces_by_side(length)
    gmsh.model.addPhysicalGroup(2, lo, tag, name="port1")
    gmsh.model.addPhysicalGroup(2, hi, tag + 1, name="port2")
    outer = [t for t in rest if len(gmsh.model.getAdjacencies(2, t)[0]) == 1]
    return outer, tag + 2


# hybrid_waveguide.msh (mm): WR-90 with an eps-6 block (x 8-15, y 0-4) on the
# bottom wall, one element layer long: hybrid, strongly dispersive modes.
# Volumes "block" (1), "air" (2); surfaces "port1" (3), "port2" (4), "walls".
a, b, length = 22.86, 10.16, 2.0
start()
blk = gmsh.model.occ.addRectangle(8.0, 0, 0, 7.0, 4.0)
box = gmsh.model.occ.addRectangle(0, 0, 0, a, b)
gmsh.model.occ.fragment([(2, box)], [(2, blk)])
gmsh.model.occ.synchronize()
vols = extrude_section([t for _, t in gmsh.model.getEntities(2)], length)
block = [v for v in vols if gmsh.model.occ.getCenterOfMass(3, v)[1] < 2.5]
walls, tag = group({"block": block, "air": [v for v in vols if v not in block]}, length)
gmsh.model.addPhysicalGroup(2, walls, tag, name="walls")
mesh_section("hybrid_waveguide.msh", lambda _x, _y: 1.5)

# cpw_section.msh (um): coplanar waveguide (signal 10, gaps 6, metal 2 thick
# and not meshed) on silicon (300 x 150) under air (300 x 150) in a closed
# box, one element layer long. Volumes "si" (1), "air" (2); surfaces "port1"
# (3), "port2" (4), "walls" (the box, 5), "metal" (the metal sides, 6).
wd, hs, ha, s, g, t, length = 300.0, 150.0, 150.0, 10.0, 6.0, 2.0, 20.0
x0 = wd / 2
start()
sub = gmsh.model.occ.addRectangle(0, 0, 0, wd, hs)
air = gmsh.model.occ.addRectangle(0, hs, 0, wd, ha)
side = wd / 2 - s / 2 - g
metals = [
    gmsh.model.occ.addRectangle(x0 - s / 2, hs, 0, s, t),
    gmsh.model.occ.addRectangle(0, hs, 0, side, t),
    gmsh.model.occ.addRectangle(x0 + s / 2 + g, hs, 0, side, t),
]
gmsh.model.occ.cut([(2, air)], [(2, m) for m in metals])
gmsh.model.occ.fragment(gmsh.model.occ.getEntities(2), [])
gmsh.model.occ.synchronize()
vols = extrude_section([t for _, t in gmsh.model.getEntities(2)], length)
si = [v for v in vols if gmsh.model.occ.getCenterOfMass(3, v)[1] < hs]
outer, tag = group({"si": si, "air": [v for v in vols if v not in si]}, length)
walls, metal = [], []
for f in outer:
    cx, cy, _ = gmsh.model.occ.getCenterOfMass(2, f)
    (walls if min(cx, wd - cx, cy, hs + ha - cy) < 1e-6 else metal).append(f)
gmsh.model.addPhysicalGroup(2, walls, tag, name="walls")
gmsh.model.addPhysicalGroup(2, metal, tag + 1, name="metal")


def cpw_size(x, y):
    """1 um around the slots, growing linearly to 40 um."""
    d = max(abs(abs(x - x0) - (s + g) / 2) - g / 2 - 2.0, abs(y - hs - t / 2) - 4.0)
    return min(40.0, 1.0 + 0.25 * max(d, 0.0))


mesh_section("cpw_section.msh", cpw_size)

# square_waveguide.msh (mm): 10 x 10 square guide, one element layer long,
# its port meshed with alternating diagonals on 6 x 6 cells. The mesh keeps
# the square's 90 degree rotation symmetry, so TE10 and TE01 stay exactly
# degenerate. Volume "air" (1); surfaces "port1" (2), "port2" (3), "walls".
start()
sq = gmsh.model.occ.addRectangle(0, 0, 0, 10.0, 10.0)
gmsh.model.occ.synchronize()
for _, c in gmsh.model.getEntities(1):
    gmsh.model.mesh.setTransfiniteCurve(c, 7)
gmsh.model.mesh.setTransfiniteSurface(sq, "Alternate")
vols = extrude_section([sq], 2.0)
walls, tag = group({"air": vols}, 2.0)
gmsh.model.addPhysicalGroup(2, walls, tag, name="walls")
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.Binary", 0)
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.write(f"{out}/square_waveguide.msh")
gmsh.finalize()
