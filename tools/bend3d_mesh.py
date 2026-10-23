"""Build the meshes of the 3D waveguide bend (no gmsh needed).

Usage: python tools/bend3d_mesh.py OUTDIR --h H [--half] [--kind bend|straight]
[--box B] [--ymirror] [--length L]

The stack (um): a 0.8 x 0.4 Si3N4 strip in SiO2, with 1.6 of oxide above and
below it (|z| <= 1.8), in the 2D tutorial's 5 x 5 layout (x and y in
[-2.5, 2.5]). The meshes are 4.5 to 56 MB, so none is committed; the run
builds them in seconds.

Volumes: "design" (|x|, |y| < 1.5 through the core, |z| < 0.2), "core" (the
input arm x < -1.5, |y| < 0.4, and the output arm y < -1.5, |x| < 0.4, both
|z| < 0.2) and "cladding" (the rest). Surfaces: "port1" (x = -2.5,
|y| < 2.0, the full height), "port2" (y = -2.5, |x| < 2.0), "outer" (every
other box face, the top and bottom z = +-1.8 included) and, on the mirror
half model only, "mirror" (z = 0, for ``PMC("mirror")``).

``kind="straight"``: the core runs straight along x through the box, x in
[-length / 2, length / 2] (default the box), and "port2" is x = +length / 2,
|y| < 2.0. "design" keeps its footprint (fill it with core along |y| < 0.4
for the straight-guide check). ``ymirror`` keeps y >= 0 with the surface
"ywall" at y = 0 (the TE0 y mirror, for ``PEC("ywall")``).

The grid: in-plane lines at {+-0.4, +-1.5, +-2.0, +-box} um (and +-2.5 for a
larger box, so its mesh holds the 5 x 5 um one), each segment cut into
ceil(L / h) equal cells, the same in x and y. In z, ``n_core`` (even) cells
through the core, then cells growing geometrically by ``growth`` through the
oxide up to ``hz_max``, then equal cells to 1.8 um. The full stack is
symmetric under z -> -z; the half is exactly its z >= 0 part.
"""

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import structured_meshes as sm

CORE_W, CORE_T = 0.8, 0.4  # um
DESIGN, PORT, BOX = 1.5, 2.0, 2.5  # half-widths (um)
Z_TOP = CORE_T / 2 + 1.6  # 1.8 um


def axis(lines, h):
    """Grid along one axis: every line a node, each segment cut at <= ``h``."""
    pts = sorted(set(lines))
    out = [np.array([pts[0]])]
    for a, b in itertools.pairwise(pts):
        out.append(np.linspace(a, b, math.ceil((b - a) / h - 1e-9) + 1)[1:])
    return np.concatenate(out)


def z_edges(h, *, half, n_core=None, hz_max=None, growth=1.25):
    """The z grid: ``n_core`` (even) core cells, then graded oxide to 1.8 um.

    Args:
        h: The in-plane cell size (um), which sets the defaults.
        half: Return z >= 0 only.
        n_core: Cells through the full core, even; default
            max(4, 2 ceil(0.2 / h)).
        hz_max: The largest oxide cell (um); default min(0.2, 2 h).
        growth: The ratio of successive oxide cells while they grow.
    """
    if n_core is None:
        n_core = max(4, 2 * math.ceil(CORE_T / 2 / h - 1e-9))
    if n_core % 2:
        raise ValueError(f"n_core must be even on the full stack, got {n_core}")
    hz_max = min(0.2, 2 * h) if hz_max is None else hz_max
    up = list(np.linspace(0.0, CORE_T / 2, n_core // 2 + 1))
    size, z, sizes = CORE_T / n_core, CORE_T / 2, []
    while size * growth < hz_max and z + size * growth < Z_TOP:
        size *= growth
        sizes.append(size)
        z += size
    rest = Z_TOP - CORE_T / 2 - sum(sizes)
    n_rest = math.ceil(rest / hz_max - 1e-9)
    sizes += [rest / n_rest] * n_rest
    up = np.array(up + list(CORE_T / 2 + np.cumsum(sizes)))
    up[-1] = Z_TOP
    return up if half else np.concatenate([-up[::-1][:-1], up])


def bend3d(
    h,
    *,
    half,
    kind="bend",
    box=BOX,
    ymirror=False,
    length=None,
    n_core=None,
    hz_max=None,
    growth=1.25,
):
    """The 3D bend (or straight guide) mesh, as `structured` returns it.

    Args:
        h: The in-plane cell size (um).
        half: Keep z >= 0 with the surface "mirror" at z = 0.
        kind: ``"bend"`` or ``"straight"`` (the core straight along x).
        box: The half-width of the box in x and y (um).
        ymirror: Straight guide only: keep y >= 0 with the surface "ywall"
            at y = 0.
        length: Straight guide only: the port-to-port length (um), x in
            [-length / 2, length / 2]; default 2 ``box``.
        n_core, hz_max, growth: The z grid, as in `z_edges`.
    """
    if kind not in ("bend", "straight"):
        raise ValueError(f"kind must be 'bend' or 'straight', got {kind!r}")
    if length is not None and kind != "straight":
        raise ValueError("length is a straight-guide option only")
    if ymirror and kind != "straight":
        raise ValueError("ymirror is a straight-guide reduction only")
    # +-2.5 stays a grid line so a larger box keeps the 5 x 5 um mesh inside
    lines = {CORE_W / 2, DESIGN, PORT, BOX, box}
    xh = box if length is None else length / 2
    xl = lines if length is None else {v for v in lines if v < xh - 1e-9} | {xh}
    x = axis({-v for v in xl} | xl, h)
    y = axis({0.0} | lines, h) if ymirror else axis({-v for v in lines} | lines, h)
    z = z_edges(h, half=half, n_core=n_core, hz_max=hz_max, growth=growth)
    eps = 1e-9
    zc = CORE_T / 2

    def volume(c):
        x, y, z = c
        if abs(z) >= zc:
            return "cladding"
        if abs(x) < DESIGN and abs(y) < DESIGN:
            return "design"
        if kind == "straight" and abs(y) < CORE_W / 2:
            return "core"
        if kind == "bend" and (
            (x < -DESIGN and abs(y) < CORE_W / 2)
            or (y < -DESIGN and abs(x) < CORE_W / 2)
        ):
            return "core"
        return "cladding"

    def surface(c):  # noqa: PLR0911
        x, y, z = c
        if half and abs(z) < eps:
            return "mirror"
        if ymirror and abs(y) < eps:
            return "ywall"
        if abs(abs(z) - Z_TOP) < eps:
            return "outer"
        if abs(x + xh) < eps:
            return "port1" if abs(y) < PORT else "outer"
        if kind == "bend" and abs(y + box) < eps:
            return "port2" if abs(x) < PORT else "outer"
        if kind == "straight" and abs(x - xh) < eps:
            return "port2" if abs(y) < PORT else "outer"
        if abs(abs(x) - xh) < eps or abs(abs(y) - box) < eps:
            return "outer"
        if abs(x) < xh - eps and abs(y) < box - eps and abs(z) < Z_TOP - eps:
            return ""  # the region interfaces carry no boundary
        raise ValueError(f"unclassified exterior face at {c}")

    if half:
        # built on z in [-1.8, 0] and reflected, so the half is exactly the
        # z >= 0 part of the full mesh (built with the z mirror)
        return sm.structured(
            (x, y, -z[::-1]), volume, surface, lambda p: p * [1.0, 1.0, -1.0]
        )
    return sm.structured((x, y, z), volume, surface, mirror=(False, False, True))


def mesh_name(h, *, half, kind="bend", box=BOX, ymirror=False, length=None):
    """The file name of a `bend3d` mesh, e.g. ``bend3d_h0.1_half.msh``."""
    name = f"{kind}3d_h{h:g}_{'half' if half else 'full'}"
    name += "_ymirror" if ymirror else ""
    name += "" if length is None else f"_L{length:g}"
    name += "" if box == BOX else f"_box{box:g}"
    return name + ".msh"


def main(argv=None):
    """Write one `bend3d` mesh into OUTDIR and print its path."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path, help="the directory to write into")
    ap.add_argument("--h", type=float, required=True, help="in-plane cell (um)")
    ap.add_argument("--half", action="store_true", help="z >= 0, mirror at z = 0")
    ap.add_argument("--kind", default="bend", choices=("bend", "straight"))
    ap.add_argument("--box", type=float, default=BOX, help="box half-width (um)")
    ap.add_argument("--ymirror", action="store_true", help="straight: y >= 0")
    ap.add_argument("--length", type=float, help="straight: port to port (um)")
    a = ap.parse_args(argv)
    opts = dict(half=a.half, kind=a.kind, box=a.box, ymirror=a.ymirror, length=a.length)
    a.outdir.mkdir(parents=True, exist_ok=True)
    path = sm.write_msh22(a.outdir / mesh_name(a.h, **opts), bend3d(a.h, **opts))
    print(path)
    return path


if __name__ == "__main__":
    main()
