"""Build the meshes of the docs tutorials (no gmsh needed).

Usage: python tools/make_tutorial_meshes.py docs/tutorials/data

``line.msh`` (mm): a parallel-plate line along x, 40 mm long, 6 mm wide and
1 mm high. Volumes "feed" (x < 10 and x > 30) and "design" (10 < x < 30);
surfaces "bottom" / "top" (the plates, z = 0 / 1), "port1" (x = 0), "port2"
(x = 40) and "side" (y walls).

``cpw.msh`` (um): a conductor-backed coplanar line along x, 1500 um long.
Substrate 0 < z < 100 (volume "substrate"), metal 100 < z < 105 (volumes
"signal_metal", |y| < 25, and "ground_metal", |y| > 55, to be excluded),
air above and in the 30 um gaps (volume "air", up to z = 300), |y| < 300.
Surfaces: "signal_wall" (the signal's two sidewalls, y = +-25), "signal"
(its other faces), "ground" (the ground planes' faces), "substrate_top"
(substrate under the gaps), "backside" (z = 0), "top" (z = 300),
"side" (y = +-300), "ends" (x = 0 and 1500 outside the gaps), and the
port gaps "p1a" / "p1b" (x = 0, y > 0 / y < 0) and "p2a" / "p2b"
(x = 1500).
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


CPW_LENGTH = 1500.0
CPW_SIGNAL, CPW_GAP = 25.0, 30.0  # signal half-width, gap (um)
CPW_METAL = (100.0, 105.0)


def cpw():
    """The coplanar line mesh as `structured_meshes.structured` returns it."""
    s, g = CPW_SIGNAL, CPW_SIGNAL + CPW_GAP
    half = (0.0, 12.5, s, 32.5, 40.0, 47.5, g, 100.0, 180.0, 300.0)
    edges = (
        np.linspace(0, CPW_LENGTH, 13),
        np.array(sorted({-y for y in half} | set(half))),
        np.array([0.0, 50.0, 100.0, 105.0, 140.0, 200.0, 300.0]),
    )
    eps = 1e-6
    z0, z1 = CPW_METAL

    def volume(c):
        if c[2] < z0:
            return "substrate"
        if c[2] < z1:
            if abs(c[1]) < s:
                return "signal_metal"
            if abs(c[1]) > g:
                return "ground_metal"
        return "air"

    def surface(c):  # noqa: PLR0911
        x, y, z = c
        in_metal = z0 - eps < z < z1 + eps
        if in_metal and abs(abs(y) - s) < eps and z0 + eps < z < z1 - eps:
            return "signal_wall"
        if in_metal and abs(y) < s + eps:
            return "signal"
        if (
            (in_metal
            and abs(y) > g - eps
            and abs(z - z0) > eps)
            or (abs(z - z0) < eps and abs(y) > g - eps)
        ):
            return "ground"
        if abs(z - z0) < eps:
            return "substrate_top"
        if abs(x) < eps or abs(x - CPW_LENGTH) < eps:
            if z0 < z < z1 and s < abs(y) < g:
                end = "p1" if x < eps else "p2"
                return end + ("a" if y > 0 else "b")
            return "ends"
        if abs(z) < eps:
            return "backside"
        if abs(z - 300.0) < eps:
            return "top"
        return "side"

    return sm.structured(edges, volume, surface)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    print(sm.write_msh22(out / "line.msh", line()))
    print(sm.write_msh22(out / "cpw.msh", cpw()))
