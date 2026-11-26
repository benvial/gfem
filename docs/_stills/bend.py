"""Card still of the bend tutorial: the field of the optimized bend in its core."""

import numpy as np

CORE_HEIGHT = 0.66  # um, the core's thickness drawn: three times the 220 nm of a guide
POINTS = 250  # per side of the field plane


def card(ns):
    """Extrude the drawn bend as a body and lay its field through the core's middle."""
    import pyvista as pv
    from gfem_stills import save_still

    um = ns["um"]
    half = ns["HALF"] / um
    pitch = ns["grid"].pitch / um
    n = round(2 * half / pitch)
    # The box as pixels, the feed guides and the drawn pattern: core where > 0.5.
    pixels = ns["window"](ns["final"])
    image = pv.ImageData(
        dimensions=(n + 1, n + 1, 2),
        spacing=(pitch, pitch, CORE_HEIGHT),
        origin=(-half, -half, 0.0),
    )
    image.cell_data["core"] = pixels.ravel()  # rows are y: x runs fastest, as here
    core = image.threshold(0.5, scalars="core").extract_surface()

    along = -half + (np.arange(POINTS) + 0.5) * (2 * half / POINTS)
    x, y = np.meshgrid(along, along)
    xy = np.c_[x.ravel(), y.ravel()]
    field = ns["trust"].e_field(ns["F0"], excite="p1").at(xy * um)
    plane = pv.StructuredGrid(
        x, y, np.full_like(x, CORE_HEIGHT / 2)
    )  # the field's points, rows y
    plane["|E|"] = np.abs(np.asarray(field)).reshape(x.shape).ravel(order="F")
    view = pv.Plotter(off_screen=True)
    view.add_mesh(core, color="tan", opacity=0.5)  # see-through, so the field shows
    view.add_mesh(plane, scalars="|E|", show_scalar_bar=False)
    save_still(view, "bend", "magnitude")
