"""The linear taper of the mode cascade recipe, as steps, in netgen.

The materials are those of `mmi_blocks`, the box ``WIDTH`` tall, perfectly
conducting ``"side"`` walls, a ``"core"`` that starts at ``y = Y0`` and widens
linearly from ``W0`` to ``W1`` over ``LENGTH`` um between two ``LEAD`` um
guides. The end lines are ``"left"`` and ``"right"``:

- `widths` the guide width and the ``m`` midpoint widths of a staircase;
- `step_block` the width step ``k`` of ``m``, planes ``depth`` either side;
- `full_taper` the smooth taper.
"""

from __future__ import annotations

import numpy as np
from mmi_blocks import mesh_of
from netgen.occ import Glue, WorkPlane

import gfem

WIDTH, Y0, W0, W1, LENGTH, LEAD = 2.0, 0.75, 0.5, 0.9, 16.0, 1.0  # um
H_CORE, H_CLAD = 0.08, 0.2


def width(x):
    """The core width of the smooth taper at ``x`` um."""
    return W0 + (W1 - W0) * np.clip((x - LEAD) / LENGTH, 0.0, 1.0)


def widths(m: int) -> np.ndarray:
    """The first guide's width, the ``m`` midpoint widths, the last guide's width."""
    return np.r_[W0, width(LEAD + (np.arange(m) + 0.5) * LENGTH / m), W1]


def _device(x0: float, x1: float, upper: list[tuple[float, float]]) -> gfem.Mesh:
    """The box on ``[x0, x1]`` with a core whose top edge goes through ``upper``."""
    pts = [(x0, Y0), (x1, Y0), *reversed([(x, Y0 + w) for x, w in upper])]
    work = WorkPlane().MoveTo(*pts[0])
    for p in pts[1:]:
        work = work.LineTo(*p)
    core = work.Close().Face()
    core.faces.name = "core"
    core.faces.maxh = H_CORE
    box = WorkPlane().MoveTo(x0, 0.0).Rectangle(x1 - x0, WIDTH).Face()
    box.faces.maxh = H_CLAD
    cladding = box - core
    cladding.faces.name = "clad"
    return mesh_of(Glue([cladding, core]), x0, x1, WIDTH)


def step_block(k: int, m: int, depth: float) -> gfem.Mesh:
    """Width step ``k`` (0 to ``m``): its end is a guide for the first and last."""
    w = widths(m)
    x = LEAD + k * LENGTH / m
    x0, x1 = (0.0 if k == 0 else x - depth), (x + LEAD if k == m else x + depth)
    return _device(x0, x1, [(x0, w[k]), (x, w[k]), (x, w[k + 1]), (x1, w[k + 1])])


def full_taper() -> gfem.Mesh:
    """The smooth taper between its two guides."""
    x1 = 2 * LEAD + LENGTH
    return _device(0.0, x1, [(0.0, W0), (LEAD, W0), (LEAD + LENGTH, W1), (x1, W1)])
