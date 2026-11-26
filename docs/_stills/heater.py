"""Card still of the heater tutorial: the temperature rise under the TiN strip."""

import numpy as np

X_SPAN = (-8.0, 15.0)  # um of the section drawn
Y_SPAN = (-7.0, 2.86)  # um: from the oxide down towards the sink, up to the air
DEPTH = 10.0  # um of guide drawn along the invariant axis
# um, from the section: (x0, x1, y0, y1) of the heater, the core and the neighbour
HEATER = (-1.0, 1.0, 2.22, 2.36)
CORE = (-0.25, 0.25, 0.0, 0.22)
NEIGHBOUR = (9.75, 10.25, 0.0, 0.22)


def card(ns):
    """Sweep the section along the guide; the rise at the heater's P_pi on its faces."""
    import pyvista as pv
    from gfem_stills import save_still

    um = ns["um"]
    x = np.linspace(*X_SPAN, 300)
    y = np.linspace(*Y_SPAN, 200)
    xx, yy = np.meshgrid(x, y)
    depth = ns["model"].mesh.section_thickness / 2
    points = np.c_[xx.ravel() * um, yy.ravel() * um, np.full(xx.size, depth)]
    face = pv.StructuredGrid(xx, np.zeros_like(xx), yy)
    rise = np.asarray(ns["steady"].rise.at(points)).reshape(xx.shape)
    face["rise (K)"] = rise.ravel(order="F")  # a structured grid runs x fastest

    view = pv.Plotter(off_screen=True)
    # The section swept along the guide: the rise shows on its front, top and side.
    block = face.extract_surface().extrude((0, DEPTH, 0), capping=True)
    view.add_mesh(block, scalars="rise (K)", show_scalar_bar=False)
    for x0, x1, y0, y1 in (HEATER, CORE, NEIGHBOUR):
        view.add_mesh(pv.Box((x0, x1, -0.05, DEPTH, y0, y1)), color="grey")
    save_still(view, "heater", "temperature")
