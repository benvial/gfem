"""Card still of the metal-bend tutorial: the field around the gold pattern."""

import numpy as np

CORE_HEIGHT = 0.66  # um, the guides' thickness drawn: three times 220 nm
POINTS = 250  # per side of the field plane


def card(ns):
    """Extrude the feed guides (see-through) and the gold (solid); lay |E_z| between."""
    import pyvista as pv
    from gfem_stills import save_still

    um = ns["um"]
    half = ns["HALF"] / um
    pitch = ns["grid"].pitch / um
    n = round(2 * half / pitch)
    gold = np.zeros((n, n))
    gold[ns["inside_design"]] = ns["final"]

    def body(pixels, name):
        image = pv.ImageData(
            dimensions=(n + 1, n + 1, 2),
            spacing=(pitch, pitch, CORE_HEIGHT),
            origin=(-half, -half, 0.0),
        )
        image.cell_data[name] = pixels.ravel()  # rows are y: x runs fastest, as here
        return image.threshold(0.5, scalars=name).extract_surface()

    along = -half + (np.arange(POINTS) + 0.5) * (2 * half / POINTS)
    x, y = np.meshgrid(along, along)
    xy = np.c_[x.ravel(), y.ravel()]
    field = ns["final_result"].e_field(ns["F0"], excite="p1").at(xy * um)
    plane = pv.StructuredGrid(x, y, np.full_like(x, CORE_HEIGHT / 2))
    plane["|E|"] = np.abs(np.asarray(field)).reshape(x.shape).ravel(order="F")
    view = pv.Plotter(off_screen=True)
    view.add_mesh(body(ns["ARMS"], "arms"), color="tan", opacity=0.5)
    view.add_mesh(body(gold, "gold"), color="grey")
    view.add_mesh(plane, scalars="|E|", show_scalar_bar=False)
    save_still(view, "metal-bend", "magnitude")
