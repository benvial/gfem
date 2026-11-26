"""Card still of the sheet tutorial: the optimized film, coloured by $|J|$."""

import numpy as np

THICKNESS = 0.12  # mm, the film is drawn as a slab this thick, below the sheet's plane
MARGIN = 0.5  # mm of plate drawn beyond the film, along x
LIFT = 0.005  # mm, the coloured face rides just above the slab so they do not z-fight


def card(ns):
    """Draw the film as a thin slab with its surface current density on the face."""
    import pyvista as pv
    from gfem_stills import save_still

    part, faces = ns["part"], ns["faces"]
    e_t = np.sqrt(ns["e_t2"].mean(axis=1))  # V/m, the tangential field of each face
    faces.cell_data["|J| (A/m)"] = ns["fill"] * e_t / ns["FILM"]  # Ohm: J = E / R_s
    film = faces.threshold(0.5, scalars="film").extract_surface()
    view = pv.Plotter(off_screen=True)
    view.add_mesh(
        film.translate((0, 0, LIFT)), scalars="|J| (A/m)", show_scalar_bar=False
    )
    view.add_mesh(film.extrude((0, 0, -THICKNESS), capping=True), color="grey")
    x0, x1 = film.bounds[:2]  # the plates, only as far as the film and a margin
    window = (x0 - MARGIN, x1 + MARGIN, -1.0, 7.0, -1.0, 3.0)
    view.add_mesh(
        part("plate1").clip_box(window, invert=False), color="grey", opacity=0.3
    )
    view.add_mesh(part("plate2").clip_box(window, invert=False), color="grey")
    save_still(view, "sheet", "magnitude")
