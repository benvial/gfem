r"""Parametric CAD geometries shared by the CAD morph tests.

A dielectric core in a $2 \times 2 \times 1$ um box (micrometres, the build's unit), the
terminal on the top face and the ground on the bottom face; the objective is
the self-capacitance of the top terminal in aF.

- strip: the core spans the box along $x$, ``width`` wide about $y = 1$,
  ``height`` tall from $z = Z_0$.
- disk: a cylinder of ``radius`` about $(1, 1)$, from $z = Z_0$ to $Z_0 + T_0$.

The mesh size is the factory's argument, so one module can mesh the same
geometry coarsely or finely. gmsh is imported when a build runs, never at
import, so collecting a test file needs no gmsh.
"""

import numpy as np

import gfem
from gfem import _cad

#: Core width, height and z of its base, and the disk's radius, micrometres.
#: 0.5 and 0.3 leave the core 0.2 and 0.4 um from the side walls and the
#: terminal, a few elements at the sizes the tests mesh with.
W0, T0, Z0, R0 = 0.5, 0.3, 0.3, 0.4

#: gmsh's bounding boxes of OCC entities are padded by the kernel's tolerance
#: (1e-7 um), so a face is "at z" within this.
PAD = 1e-6

#: Nodes are matched to the design's entities within this, micrometres: the
#: coordinates are exact to about 1e-15 and a projected node is off by under
#: 1e-11, while the finest spacing the tests mesh with is 0.0375 um, so 1e-9
#: neither misses a node on an entity nor catches one off it.
MATCH = 1e-9

#: The box, micrometres.
BOX = (2.0, 2.0, 1.0)

PROBLEM = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")
OPTIONS = gfem.SolverOptions(threads=1, linear_solver="direct")


def _finish(lc):
    """Group the volumes and the top and bottom faces; set the mesh sizes."""
    gmsh = _cad.import_gmsh()
    gmsh.model.occ.synchronize()
    volumes = [tag for _, tag in gmsh.model.getEntities(3)]
    # the core is the smaller of the two volumes the fragment leaves
    mass = {tag: gmsh.model.occ.getMass(3, tag) for tag in volumes}
    core = min(mass, key=mass.get)
    gmsh.model.addPhysicalGroup(3, [core], name="core")
    gmsh.model.addPhysicalGroup(3, [t for t in volumes if t != core], name="clad")
    top, bottom = [], []
    for _, tag in gmsh.model.getEntities(2):
        box = gmsh.model.getBoundingBox(2, tag)
        if abs(box[2] - BOX[2]) < PAD and abs(box[5] - BOX[2]) < PAD:
            top.append(tag)
        if abs(box[2]) < PAD and abs(box[5]) < PAD:
            bottom.append(tag)
    gmsh.model.addPhysicalGroup(2, top, name="top")
    gmsh.model.addPhysicalGroup(2, bottom, name="bottom")
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc / 4)


def strip_build(lc):
    """The strip's build function, ``build(width, height)``, meshed at ``lc``."""

    def build(width, height):
        occ = _cad.import_gmsh().model.occ
        box = occ.addBox(0, 0, 0, *BOX)
        core = occ.addBox(0, 1 - width / 2, Z0, BOX[0], width, height)
        occ.fragment([(3, box)], [(3, core)])
        _finish(lc)

    return build


def disk_build(lc):
    """The disk's build function, ``build(radius)``, meshed at ``lc``."""

    def build(radius):
        occ = _cad.import_gmsh().model.occ
        box = occ.addBox(0, 0, 0, *BOX)
        core = occ.addCylinder(1, 1, Z0, 0, 0, T0, radius)
        occ.fragment([(3, box)], [(3, core)])
        _finish(lc)

    return build


def make_model(mesh):
    return gfem.Model(
        mesh,
        materials={"core": gfem.Material(eps_r=3.9), "clad": gfem.Material(eps_r=1.0)},
    )


def capacitance(model):
    """The top terminal's self-capacitance, aF."""
    return gfem.solve(model, PROBLEM, options=OPTIONS).capacitance[0, 0] * 1e18


def strip_velocity(x, width, height):
    r"""The strip's design velocity of ``width`` at nodes ``x`` (um), ``(n, 3)``.

    The side walls $y = 1 \pm w / 2$ (and their edges and corners, which
    lie on them) move by half a unit each way. ``height`` is the build's.
    """
    v = np.zeros_like(x)
    wall = (np.abs(np.abs(x[:, 1] - 1) - width / 2) < MATCH) & (
        (x[:, 2] > Z0 - MATCH) & (x[:, 2] < Z0 + height + MATCH)
    )
    v[wall, 1] = 0.5 * np.sign(x[wall, 1] - 1)
    return v


def disk_velocity(x, radius):
    """The disk's design velocity of ``radius`` at nodes ``x`` (um), ``(n, 3)``.

    The wall and the two circles bounding it move radially, a unit per unit;
    the end caps' interior nodes stay in their planes.
    """
    r = x[:, :2] - 1
    on = np.abs(np.linalg.norm(r, axis=1) - radius) < MATCH
    v = np.zeros_like(x)
    v[on, :2] = r[on] / radius
    return v
