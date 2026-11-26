"""Card still of the grating tutorial: light leaving the grating coupler."""

import numpy as np

X_SPAN = (-2.0, 17.0)  # um of the guide drawn
Y_ABOVE, Y_BELOW = 1.2, -0.9  # um, how far the field is drawn over and under the film
LIFT = 3.0  # the section's heights are stretched by this, or the film is a line
DEPTH = 3.0  # um of the grating drawn along the invariant axis
NUDGE = 1e-3  # um: field probes stay this far inside their region


def card(ns):
    """Extrude the grating as meshed; lay |E_z| over and under it."""
    import pyvista as pv
    from gfem_stills import save_still

    um = ns["um"]
    period, fill, teeth = ns["PERIOD"], ns["FILL"], ns["TEETH"]
    t_si, etch = ns["T_SI"], ns["ETCH"]
    field = ns["initial"].e_field(ns["F0"], "input")

    def front(y0, y1, rows):
        # |E_z| on a face drawn in front of the section, x along the guide, z up.
        x = np.linspace(*X_SPAN, 600)
        y = np.linspace(y0 + NUDGE, y1 - NUDGE, rows)
        xx, yy = np.meshgrid(x, y)
        xy = np.c_[
            xx.ravel(), yy.ravel(), np.full(xx.size, ns["mesh"].section_thickness / 2)
        ]
        face = pv.StructuredGrid(xx, np.full_like(xx, -0.01), LIFT * yy)
        face["|E|"] = (
            np.abs(np.asarray(field.at(xy * [um, um, 1.0]))[:, 2])
            .reshape(xx.shape)
            .ravel(order="F")
        )
        return face

    film = pv.Box((*X_SPAN, 0, DEPTH, 0, LIFT * (t_si - etch)))
    teeth_boxes = [
        pv.Box(
            (
                n * period,
                (n + fill) * period,
                0,
                DEPTH,
                LIFT * (t_si - etch),
                LIFT * t_si,
            )
        )
        for n in range(teeth)
    ]
    guide = pv.Box((X_SPAN[0], 0, 0, DEPTH, LIFT * (t_si - etch), LIFT * t_si))
    faces = [front(t_si, Y_ABOVE, 120), front(Y_BELOW, 0.0, 90)]
    peak = np.percentile(np.concatenate([f["|E|"] for f in faces]), 99)  # not the ends
    view = pv.Plotter(off_screen=True)
    for face in faces:
        view.add_mesh(face, scalars="|E|", clim=(0, peak), show_scalar_bar=False)
    for body in (film, guide, *teeth_boxes):
        view.add_mesh(body, color="grey")
    save_still(view, "grating", "magnitude")
