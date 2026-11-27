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
- ``cell_order1.msh`` / ``cell_order2.msh``: a cube of side 3 (a shell of 26
  cells around the core) with a sphere of radius 0.3 at (1.2, 1.5, 1.5).
  Volumes ``ball``, ``air`` (the core [0.5, 2.5]^3 outside the ball and the
  heater), ``heater`` (x in 1.9..2.5 of the core) and ``pml`` (the shell).
  Surfaces ``xlo`` / ``xhi`` (x = 0 and 3), ``ylo`` / ``yhi`` (a periodic
  pair), ``zwall`` (z = 0 and 3), ``sheet`` (the interior face x = 1.9
  between air and heater) and ``wall`` (the sphere).
  Mesh size 0.4 in the core and 0.2 on the sphere (`CELL_CORE`, `CELL_FINE`),
  and 2 at the corners of the shell (`CELL_COARSE`), which needs no
  resolution: a third the size of a uniform 0.4 mesh. No high-order
  optimization, which would spread curvature to far elements. Only the
  elements touching the sphere are curved, so every flat outer face, the
  ``heater`` and the ``pml`` shell are straight.
- ``coax_split_order2.msh``: the same line at order 2 with the plane x = 0
  split off ``sym`` as surface ``port``: a flat face whose edges are straight
  lines, two of them on the curved walls ``inner`` and ``outer``.
- ``octant_order3.msh``: the octant at order 3, which gfem refuses.
- ``octant_mixed.msh``: tetra10 volumes with first-order ``triangle`` faces,
  which gfem refuses.
"""

from pathlib import Path

import gmsh

HERE = Path(__file__).parent
#: The mesh size (m) at the cell's corners away from the sphere.
CELL_COARSE = 2.0
#: The sphere's centre and radius (m).
BALL_CENTRE = (1.2, 1.5, 1.5)
BALL_RADIUS = 0.3
#: The mesh size (m) at the core's corners.
CELL_CORE = 0.4
#: The mesh size (m) on the sphere.
CELL_FINE = 0.2


def _mesh(
    build,
    name: str,
    order: int,
    size: float,
    *,
    periodic=None,
    fine: float | None = None,
    optimize: int = 1,
) -> None:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        build()
        gmsh.model.occ.synchronize()
        if periodic is not None:
            periodic()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size if fine is None else fine)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        if order > 1:
            gmsh.option.setNumber("Mesh.HighOrderOptimize", optimize)
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


def _coax(split: bool = False) -> None:
    outer = gmsh.model.occ.addCylinder(
        0, 0, 0, 0, 0, 1, 2.0, angle=3.141592653589793 / 2
    )
    inner = gmsh.model.occ.addCylinder(
        0, 0, 0, 0, 0, 1, 1.0, angle=3.141592653589793 / 2
    )
    gmsh.model.occ.cut([(3, outer)], [(3, inner)])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [t for _, t in gmsh.model.getEntities(3)], 1, "fill")
    inner_s, outer_s, sym, port = [], [], [], []
    for _, tag in gmsh.model.getEntities(2):
        if gmsh.model.getType(2, tag) == "Plane":
            x, _, z = gmsh.model.occ.getCenterOfMass(2, tag)
            # split: the plane x = 0 is a port face whose rim lines are straight
            (port if split and x < 1e-9 and 1e-9 < z < 1 - 1e-9 else sym).append(tag)
            continue
        x, y, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        (inner_s if (x * x + y * y) ** 0.5 < 1.5 else outer_s).append(tag)
    gmsh.model.addPhysicalGroup(2, inner_s, 2, "inner")
    gmsh.model.addPhysicalGroup(2, outer_s, 3, "outer")
    gmsh.model.addPhysicalGroup(2, sym, 4, "sym")
    if split:
        gmsh.model.addPhysicalGroup(2, port, 5, "port")


def _coax_split() -> None:
    _coax(split=True)


def _name_cell_surfaces() -> list[int]:
    """Name the cell's surfaces; return the sphere's surface tags."""
    faces: dict[str, list[int]] = {
        n: [] for n in ("xlo", "xhi", "ylo", "yhi", "zwall", "sheet")
    }
    wall = []
    for _, tag in gmsh.model.getEntities(2):
        box = gmsh.model.getBoundingBox(2, tag)
        flat = [i for i in range(3) if abs(box[3 + i] - box[i]) < 1e-6]
        if gmsh.model.getType(2, tag) != "Plane":
            wall.append(tag)
        elif flat == [2] and (box[2] < 1e-6 or box[2] > 3 - 1e-6):
            faces["zwall"].append(tag)
        elif flat == [0] and abs(box[0] - 1.9) < 1e-6:
            faces["sheet"].append(tag)
        elif flat == [0] and box[0] < 1e-6:
            faces["xlo"].append(tag)
        elif flat == [0] and box[0] > 3 - 1e-6:
            faces["xhi"].append(tag)
        elif flat == [1] and box[1] < 1e-6:
            faces["ylo"].append(tag)
        elif flat == [1] and box[1] > 3 - 1e-6:
            faces["yhi"].append(tag)
    for k, (name, tags) in enumerate({**faces, "wall": wall}.items(), start=1):
        gmsh.model.addPhysicalGroup(2, tags, 10 + k, name)
    return wall


def _cell() -> None:
    occ = gmsh.model.occ
    # 27 boxes cut by the planes 0.5 and 2.5: the shell's cells end on the core's
    # faces, as a perfectly matched layer needs
    cuts = (0.0, 0.5, 2.5, 3.0)
    cells = [
        occ.addBox(
            cuts[i],
            cuts[j],
            cuts[k],
            cuts[i + 1] - cuts[i],
            cuts[j + 1] - cuts[j],
            cuts[k + 1] - cuts[k],
        )
        for i in range(3)
        for j in range(3)
        for k in range(3)
    ]
    heater = occ.addBox(1.9, 0.5, 0.5, 0.6, 2, 2)
    ball = occ.addSphere(*BALL_CENTRE, BALL_RADIUS)
    occ.fragment([(3, c) for c in cells], [(3, heater), (3, ball)])
    occ.synchronize()
    groups: dict[str, list[int]] = {"ball": [], "air": [], "heater": [], "pml": []}
    for _, tag in gmsh.model.getEntities(3):
        box = gmsh.model.getBoundingBox(3, tag)
        if all(
            box[i] > BALL_CENTRE[i] - BALL_RADIUS - 1e-6
            and box[3 + i] < BALL_CENTRE[i] + BALL_RADIUS + 1e-6
            for i in range(3)
        ):
            groups["ball"].append(tag)
        elif min(box[:3]) < 0.4 or max(box[3:]) > 2.6:
            groups["pml"].append(tag)
        else:
            groups["heater" if box[0] > 1.89 else "air"].append(tag)
    for k, (name, tags) in enumerate(groups.items(), start=1):
        gmsh.model.addPhysicalGroup(3, tags, k, name)
    wall = _name_cell_surfaces()
    # a fine mesh on the sphere, coarse elsewhere (the shell dominates the count)
    for _, tag in gmsh.model.getEntities(0):
        x = gmsh.model.getValue(0, tag, [])
        core = all(0.5 - 1e-9 <= c <= 2.5 + 1e-9 for c in x)
        gmsh.model.mesh.setSize([(0, tag)], CELL_CORE if core else CELL_COARSE)
    for _, tag in gmsh.model.getBoundary(
        [(2, t) for t in wall], combined=False, oriented=False, recursive=True
    ):
        gmsh.model.mesh.setSize([(0, tag)], CELL_FINE)


def _cell_periodic() -> None:
    """Pair each y = 3 patch with the y = 0 patch under it (translation by -3)."""
    lo, hi = [], []
    for _, tag in gmsh.model.getEntities(2):
        box = gmsh.model.getBoundingBox(2, tag)
        if gmsh.model.getType(2, tag) == "Plane" and abs(box[1] - box[4]) < 1e-6:
            if box[1] < 1e-6:
                lo.append((tag, box))
            elif box[1] > 3 - 1e-6:
                hi.append((tag, box))
    for slave, sbox in hi:
        master = next(
            m
            for m, mbox in lo
            if all(abs(mbox[i] - sbox[i]) < 1e-6 for i in (0, 2, 3, 5))
        )
        gmsh.model.mesh.setPeriodic(
            2, [slave], [master], [1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1, 0, 0, 0, 0, 1]
        )


def main() -> None:
    _mesh(_octant, "octant_order1", 1, 0.2)
    _mesh(_octant, "octant_order2", 2, 0.2)
    _mesh(_coax, "coax_order1", 1, 0.5)
    _mesh(_coax, "coax_order2", 2, 0.5)
    _mesh(_coax_split, "coax_split_order2", 2, 0.5)
    _mesh(_octant, "octant_order3", 3, 0.5)
    for order in (1, 2):
        _mesh(
            _cell,
            f"cell_order{order}",
            order,
            CELL_COARSE,
            periodic=_cell_periodic,
            fine=CELL_FINE,
            optimize=0,
        )


if __name__ == "__main__":
    main()
