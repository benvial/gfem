"""Card still of the bend3d tutorial: the optimized bend's field on its two cuts."""


def card(ns):
    """Redraw the tutorial's 3D cell (`surfaces`, `core`, `oxide`) in the card style."""
    import pyvista as pv
    from gfem_stills import save_still

    view = pv.Plotter(off_screen=True)
    view.add_mesh(ns["oxide"], color="lightgrey", opacity=0.15)  # the cladding: dropped
    for part in ns["core"]:
        view.add_mesh(part, color="tan", opacity=0.5)  # see-through, so the field shows
    for surface in ns["surfaces"]:
        view.add_mesh(
            surface, scalars="|E|", clim=(0, ns["peak"]), show_scalar_bar=False
        )
    save_still(view, "bend3d", "magnitude")
