"""Card still of the modulator tutorial: the signal electrode's potential."""

import numpy as np

SLAB, RIB = 0.3, 0.4  # um: the film left after the etch, and the etch depth
ELECTRODE = 0.2  # um, the electrodes' thickness, on the slab
SIGNAL_WIDTH, GROUND_WIDTH = 4.0, 4.0  # um
BOX = -1.2  # um, the oxide drawn below the film
TOP = 1.2  # um, the cladding drawn above it
DEPTH = 6.0  # um of guide drawn along the invariant axis
LIFT = 3.0  # the section's heights are stretched by this, or the film is a line
NUDGE = 1e-3  # um: probes in the box and cladding stay this far from the film


def card(ns):
    """Extrude the film, ridges and electrodes; the potential at 1 V on the front."""
    import pyvista as pv
    from gfem_stills import save_still

    um = ns["um"]
    section = ns["section"]
    signal = section.signal_edge / um
    ground = section.ground_edge / um
    centre = section.ridge_centre / um
    width = section.width / um
    reach = ground + GROUND_WIDTH + 1.0  # um drawn beyond the grounds' outer walls

    # Front face: the potential everywhere but in the metal, which is drawn solid.
    x = np.linspace(-reach, reach, 400)
    y = np.linspace(BOX + NUDGE, TOP - NUDGE, 120)
    xx, yy = np.meshgrid(x, y)
    in_metal = (
        (yy > SLAB - NUDGE)
        & (yy < SLAB + ELECTRODE + NUDGE)
        & ((np.abs(xx) < signal + NUDGE) | (np.abs(xx) > ground - NUDGE))
        & (np.abs(xx) < ground + GROUND_WIDTH + NUDGE)
    )
    points = np.c_[xx.ravel() * um, yy.ravel() * um, np.full(xx.size, 0.5e-7)]
    face = pv.StructuredGrid(xx, np.zeros_like(xx), LIFT * yy)
    potential = np.zeros(xx.size)  # the metal's cells are dropped below: no probe there
    outside = ~in_metal.ravel()
    potential[outside] = np.asarray(ns["potential"].at(points[outside]))
    face["V"] = potential.reshape(xx.shape).ravel(
        order="F"
    )  # a structured grid runs x fastest
    face["metal"] = in_metal.ravel(order="F")
    face = face.threshold(0.5, scalars="metal", invert=True).extract_surface()

    def box(x0, x1, y0, y1):
        return pv.Box((x0, x1, -0.05, DEPTH, LIFT * y0, LIFT * y1))

    film = [box(-reach, reach, 0.0, SLAB)]
    film += [
        box(s * centre - width / 2, s * centre + width / 2, SLAB, SLAB + RIB)
        for s in (1, -1)
    ]
    metal = [
        box(-signal, signal, SLAB, SLAB + ELECTRODE),
        *(
            box(s * ground, s * (ground + GROUND_WIDTH), SLAB, SLAB + ELECTRODE)
            for s in (1, -1)
        ),
    ]
    view = pv.Plotter(off_screen=True)
    view.add_mesh(face, scalars="V", clim=(-1, 1), show_scalar_bar=False)
    for body in film:
        view.add_mesh(
            body, color="grey", opacity=0.5
        )  # see-through: the film shows its V
    for body in metal:
        view.add_mesh(body, color="grey")
    save_still(view, "modulator", "signed")
