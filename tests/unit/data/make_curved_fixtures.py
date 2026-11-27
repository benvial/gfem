"""Regenerate the curved-mesh fixtures next to this file (needs gmsh).

    python make_curved_fixtures.py

Writes, in gmsh format 4.1:

- ``octant_order1.msh`` / ``octant_order2.msh``: the octant of the unit sphere,
  mesh size 0.2, physical groups volume ``ball``, surfaces ``wall`` (the
  sphere) and ``sym`` (the three flat faces). The order-2 file is the
  order-1 mesh with mid-edge nodes put on the sphere, with ``HighOrderOptimize 1``.
- ``coax_order1.msh`` / ``coax_order2.msh``: a quarter of a coaxial line, radii 1
  and 2, length 1, mesh size 0.5, volume ``fill``, surfaces ``inner``,
  ``outer``, ``sym``.
- ``octant_order3.msh``: the octant at order 3, which gfem refuses.
- ``octant_mixed.msh``: tetra10 volumes with first-order ``triangle`` faces,
  which gfem refuses.
"""

from pathlib import Path

import gmsh

HERE = Path(__file__).parent


def _mesh(build, name: str, order: int, size: float) -> None:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        build()
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        if order > 1:
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(HERE / f"{name}.msh"))
    finally:
        gmsh.finalize()


def _faces_by_normal():
    """Surfaces of the model split into curved ones and flat ones (by bounding box)."""
    curved, flat = [], []
    for dim, tag in gmsh.model.getEntities(2):
        kind = gmsh.model.getType(dim, tag)
        (flat if kind == "Plane" else curved).append(tag)
    return curved, flat


def _name_octant() -> None:
    curved, flat = _faces_by_normal()
    gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], 1, "ball")
    gmsh.model.addPhysicalGroup(2, curved, 2, "wall")
    gmsh.model.addPhysicalGroup(2, flat, 3, "sym")


def _octant() -> None:
    gmsh.model.occ.addSphere(
        0,
        0,
        0,
        1.0,
        angle1=0,
        angle2=3.141592653589793 / 2,
        angle3=3.141592653589793 / 2,
    )
    gmsh.model.occ.synchronize()
    _name_octant()


def _coax() -> None:
    outer = gmsh.model.occ.addCylinder(
        0, 0, 0, 0, 0, 1, 2.0, angle=3.141592653589793 / 2
    )
    inner = gmsh.model.occ.addCylinder(
        0, 0, 0, 0, 0, 1, 1.0, angle=3.141592653589793 / 2
    )
    gmsh.model.occ.cut([(3, outer)], [(3, inner)])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], 1, "fill")
    inner_s, outer_s, sym = [], [], []
    for _, tag in gmsh.model.getEntities(2):
        if gmsh.model.getType(2, tag) == "Plane":
            sym.append(tag)
            continue
        x, y, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        (inner_s if (x * x + y * y) ** 0.5 < 1.5 else outer_s).append(tag)
    gmsh.model.addPhysicalGroup(2, inner_s, 2, "inner")
    gmsh.model.addPhysicalGroup(2, outer_s, 3, "outer")
    gmsh.model.addPhysicalGroup(2, sym, 4, "sym")


def main() -> None:
    _mesh(_octant, "octant_order1", 1, 0.2)
    _mesh(_octant, "octant_order2", 2, 0.2)
    _mesh(_coax, "coax_order1", 1, 0.5)
    _mesh(_coax, "coax_order2", 2, 0.5)
    _mesh(_octant, "octant_order3", 3, 0.5)


if __name__ == "__main__":
    main()
