"""Card still of the shape tutorial: the optimized coplanar line and its gap field."""

NEAR = (0, 650, -120, 120, 0, 110)  # um: the line's first stretch, from the p1 end
CLIP = 0.8  # of the peak field: the colour scale tops out here, so the gaps read


def card(ns):
    """Draw |E| on the optimized line's metal mid-plane, with its conductors."""
    import pyvista as pv
    from gfem_stills import save_still

    plane = ns["planes"]["optimized"]
    part, designed = ns["part"], ns["designed"]
    peak = float(plane["E (kV/m)"].max())
    view = pv.Plotter(off_screen=True)
    view.add_mesh(
        plane.clip_box(NEAR, invert=False),
        scalars="E (kV/m)",
        clim=(0, CLIP * peak),
        show_scalar_bar=False,
    )
    view.add_mesh(
        part("substrate", on=designed).clip_box(NEAR, invert=False),
        color="grey",
        opacity=0.3,
    )
    for region in ("signal_metal", "ground_metal"):
        view.add_mesh(
            part(region, on=designed).clip_box(NEAR, invert=False), color="grey"
        )
    save_still(view, "shape", "magnitude")
