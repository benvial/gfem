"""Card still of the transmon tutorial: the pads and the qubit mode's field."""

NEAR = (
    -1.0,
    1.0,
    -0.7,
    0.7,
    -0.4,
    0.4,
)  # mm: the pads, the junction and the chip around
CLIP = 0.5  # of the peak field: the colour scale tops out here, so the pad edges read


def card(ns):
    """Draw |E| of the qubit mode on the pads' plane, over the chip."""
    import numpy as np
    import pyvista as pv
    from gfem_stills import save_still

    mm, part, modes = ns["mm"], ns["part"], ns["modes"]
    field = modes.e_field(ns["QUBIT"]).to_pyvista()
    field.points /= mm
    field["|E|"] = np.linalg.norm(field["E_real"] + 1j * field["E_imag"], axis=1)
    plane = field.slice(normal="z", origin=(0, 0, 0.025)).clip_box(NEAR, invert=False)
    view = pv.Plotter(off_screen=True)
    view.add_mesh(
        plane,
        scalars="|E|",
        clim=(0, CLIP * float(plane["|E|"].max())),
        show_scalar_bar=False,
    )
    # The chip as a box of its bounds inside the window, not its jagged mesh surface.
    low, high = np.reshape(part("chip").bounds, (3, 2)).T
    window = np.reshape(NEAR, (3, 2)).T
    box = np.c_[np.maximum(low, window[0]), np.minimum(high, window[1])].ravel()
    view.add_mesh(pv.Box(tuple(box)), color="grey", opacity=0.3)
    for surface in ns["METAL"]:
        view.add_mesh(part(surface).clip_box(NEAR, invert=False), color="grey")
    save_still(view, "transmon", "magnitude")
