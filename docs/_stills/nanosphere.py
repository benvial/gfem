"""Card still of the nanosphere tutorial: a dipole beside a silicon sphere."""

import numpy as np

HALF = 0.45  # um, half the side of the box the field is cut from
LOBE = "tangential 0"  # the dipole 30 nm from the surface


def card(ns):
    """Draw Re E_x of the silicon case on the y = 0 cut, from the kept fields."""
    import pyvista as pv
    from gfem_stills import save_still

    um, radius = ns["um"], ns["RADIUS"] / ns["um"]
    field = ns["results"]["silicon"].e_field(ns["F0"], LOBE).to_pyvista()
    field.points /= um
    field["Re Ex"] = field["E_real"][:, 0]
    box = field.clip_box((-HALF, HALF, -HALF, HALF, -HALF, HALF), invert=False)
    cut = box.slice(normal="y", origin=(0, 0, 0))
    # Symmetric limits at the 75th percentile: the dipole's near field would
    # otherwise saturate the map and wash out the radiated wave.
    top = np.percentile(np.abs(cut["Re Ex"]), 75)
    view = pv.Plotter(off_screen=True)
    view.add_mesh(cut, scalars="Re Ex", clim=(-top, top), show_scalar_bar=False)
    view.add_mesh(
        pv.Sphere(radius=radius, theta_resolution=96, phi_resolution=96), color="tan"
    )
    arrow = pv.Arrow(start=(-0.05, 0, radius + 0.03), direction=(1, 0, 0), scale=0.1)
    view.add_mesh(arrow, color="red")
    save_still(view, "nanosphere", "signed")
