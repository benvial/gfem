"""Card still of the topology tutorial: the optimized layer, coloured by $|E_z|$."""

import numpy as np

THICKNESS = 0.8  # mm, the layer is drawn as a slab this thick
BOARD = 0.4  # mm, the grey board under the whole line
NUDGE = 1e-3  # mm: probe points are moved this far inside the line's rectangle
LENGTH, WIDTH = 40.0, 6.0  # mm, the line's rectangle


def card(ns):
    """Draw the nominal design as a thin slab with the pass-band field on its face."""
    import pyvista as pv
    from gfem_stills import save_still

    mm, grid = ns["mm"], ns["grid"]
    density = np.asarray(ns["final"].nominal)  # (rows, columns): the pixels
    rows, columns = density.shape
    x0, y0 = np.asarray(grid.origin) / mm
    pitch = grid.pitch / mm
    pixels = pv.ImageData(
        dimensions=(columns + 1, rows + 1, 2),
        spacing=(pitch, pitch, THICKNESS),
        origin=(x0, y0, 0.0),
    )
    pixels.cell_data["density"] = density.ravel()  # x fastest, as the grid is laid out
    layer = pixels.threshold(0.5, scalars="density").extract_surface()
    xy = np.clip(layer.points[:, :2], (NUDGE, NUDGE), (LENGTH - NUDGE, WIDTH - NUDGE))
    field = ns["fields"].e_fields[0]  # the pass frequency
    layer["|E_z|"] = np.abs(np.asarray(field.at(xy * mm))).reshape(len(xy), -1)[:, -1]
    board = pv.Box((0, LENGTH, 0, WIDTH, -BOARD, 0))
    view = pv.Plotter(off_screen=True)
    view.add_mesh(layer, scalars="|E_z|", show_scalar_bar=False)
    view.add_mesh(board, color="grey")
    save_still(view, "topology", "magnitude")
