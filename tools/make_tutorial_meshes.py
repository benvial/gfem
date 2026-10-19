"""Build the meshes of the docs tutorials (no gmsh needed).

Usage: python tools/make_tutorial_meshes.py docs/tutorials/data

``line.msh`` (mm): a parallel-plate line along x, 40 mm long, 6 mm wide and
1 mm high. Volumes "feed" (x < 10 and x > 30) and "design" (10 < x < 30);
surfaces "bottom" / "top" (the plates, z = 0 / 1), "port1" (x = 0), "port2"
(x = 40) and "side" (y walls).
"""

import sys
from pathlib import Path

import numpy as np
import structured_meshes as sm

LENGTH, WIDTH, HEIGHT = 40.0, 6.0, 1.0
DESIGN = (10.0, 30.0)


def line():
    """The line mesh as `structured_meshes.structured` returns it."""
    edges = (
        np.linspace(0, LENGTH, 81),
        np.linspace(0, WIDTH, 7),
        np.linspace(0, HEIGHT, 2),
    )
    eps = 1e-9 * LENGTH

    def volume(c):
        return "design" if DESIGN[0] < c[0] < DESIGN[1] else "feed"

    def surface(c):
        if abs(c[0]) < eps:
            return "port1"
        if abs(c[0] - LENGTH) < eps:
            return "port2"
        if abs(c[2]) < eps:
            return "bottom"
        if abs(c[2] - HEIGHT) < eps:
            return "top"
        if abs(c[1]) < eps or abs(c[1] - WIDTH) < eps:
            return "side"
        return ""  # the design / feed interfaces carry no boundary

    return sm.structured(edges, volume, surface)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    print(sm.write_msh22(out / "line.msh", line()))
