"""Card still of the metagrating tutorial: the optimized grating and its field."""

import numpy as np

PERIODS = 3  # cells drawn side by side along x
DEPTH = 1.2  # um of grating drawn along y, the invariant axis
SAMPLES = (150, 140)  # field samples per cell: along x, along z


def _runs(row):
    """The (first, last + 1) pixel of each run of silicon in a row of 0/1 pixels."""
    padded = np.r_[0, np.asarray(row) > 0.5, 0]
    edges = np.flatnonzero(np.diff(padded))
    return edges.reshape(-1, 2)


def card(ns):
    """Draw Re E_y of the incident wave through boxes built from the design."""
    import pyvista as pv
    from gfem_stills import save_still

    um, period, pitch, top = (
        ns["um"],
        ns["PERIOD"] / ns["um"],
        ns["PITCH"] / ns["um"],
        ns["TOP"] / ns["um"],
    )
    heights = {k: v / um for k, v in ns["HEIGHTS"].items()}
    field = ns["trust"].e_field(ns["F0"], excite="in")
    xs = (np.arange(SAMPLES[0]) + 0.5) / SAMPLES[0] * period
    zs = (np.arange(SAMPLES[1]) + 0.5) / SAMPLES[1] * top
    x, z = np.meshgrid(xs, zs)
    # y = 0.04 um, inside the mesh's thin section, like the tutorial's own cut
    points = np.c_[x.ravel(), np.full(x.size, 0.04), z.ravel()] * um
    values = np.asarray(field.at(points))[:, 1].real
    clim = float(np.percentile(np.abs(values), 99.5))
    # A StructuredGrid orders its points with x fastest: the scalars follow.
    values = values.reshape(x.shape).ravel(order="F")

    view = pv.Plotter(off_screen=True)
    z0 = heights["glass"]
    layers = [
        (z0, z0 + heights["lower"]),
        (z0 + heights["lower"], z0 + heights["lower"] + heights["upper"]),
    ]
    for i in range(PERIODS):
        shift = i * period
        face = pv.StructuredGrid(x + shift, np.full_like(x, DEPTH / 2), z)
        face["Re E_y"] = values
        view.add_mesh(face, scalars="Re E_y", clim=(-clim, clim), show_scalar_bar=False)
        for row, (z_a, z_b) in zip(np.asarray(ns["final"]), layers, strict=True):
            for first, last in _runs(row):
                box = (shift + first * pitch, shift + last * pitch, 0, DEPTH, z_a, z_b)
                view.add_mesh(pv.Box(box), color="grey", opacity=0.3)
    glass = (0, PERIODS * period, 0, DEPTH, 0, z0)
    view.add_mesh(pv.Box(glass), color="grey", opacity=0.3)
    save_still(view, "metagrating", "signed")
