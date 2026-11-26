"""Card still of the heater3d tutorial: the temperature rise over the heated spiral."""

import numpy as np

CORE_HEIGHT = 0.22  # um, the waveguide core's thickness
HEATER_Z, HEATER_HEIGHT = 2.2, 0.1  # um: the heater's level and thickness
PAD_HEIGHT = 0.1  # um


def _prism(polygon, z0, height, pv):
    ring = np.c_[polygon, np.full(len(polygon), z0)]
    face = pv.PolyData(ring, faces=[len(ring), *range(len(ring))]).triangulate()
    return face.extrude((0, 0, height), capping=True)


def card(ns):
    """Redraw the tutorial's 3D cell (`plane`, `out`) with the core spiral as a body."""
    import pyvista as pv
    from gfem_stills import save_still

    out = ns["out"]
    view = pv.Plotter(off_screen=True)
    view.add_mesh(ns["plane"], scalars="rise (K/W)", show_scalar_bar=False)
    view.add_mesh(_prism(out["core_polygon"], 0.0, CORE_HEIGHT, pv), color="grey")
    view.add_mesh(
        _prism(out["heater_polygon"], HEATER_Z, HEATER_HEIGHT, pv), color="tan"
    )
    for pad in out["pad_polygons"]:
        (x0, y0), (x1, y1) = pad.min(axis=0), pad.max(axis=0)
        view.add_mesh(
            pv.Box((x0, x1, y0, y1, HEATER_Z, HEATER_Z + PAD_HEIGHT)), color="tan"
        )
    view.set_scale(zscale=4)  # the stack is thin: stretch it, as the tutorial does
    save_still(view, "heater3d", "temperature")
