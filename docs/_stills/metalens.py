"""Card still of the metalens tutorial: a row of ridge cells and its field."""

import numpy as np

PERIODS = 7  # cells drawn side by side along x
DEPTH = 0.9  # um of ridge drawn behind the cell's thin section, along y
SAMPLES = (24, 90)  # field samples per cell: along x, along z
NUDGE = 1e-3  # of the cell's depth: the probe plane sits this far inside the cell


def card(ns):
    """Solve the cell at 633 nm; draw Re E of the TE wave across a row of ridges."""
    import pyvista as pv
    from gfem_stills import save_still

    import gfem

    um, mesh, cell = ns["um"], ns["mesh"], ns["cell"]
    solved = gfem.solve(
        cell,
        gfem.Driven(
            frequency=ns["FREQS"][1], excite=["bottom (0, 0) TE"], keep_fields=True
        ),
        options=ns["options"],
    )
    field = solved.e_field(ns["FREQS"][1], excite="bottom (0, 0) TE")
    nodes = np.asarray(mesh.nodes) / um
    low, high = nodes.min(axis=0), nodes.max(axis=0)  # the cell's box, um
    period, thin = high[0] - low[0], high[1] - low[1]

    def extent(region):
        corners = nodes[np.asarray(mesh.volume_elements)[mesh.elements(region)]]
        return corners.reshape(-1, 3).min(axis=0), corners.reshape(-1, 3).max(axis=0)

    r_low, r_high = extent("ridge")
    g_high = extent("glass")[1][2]

    xs = low[0] + (np.arange(SAMPLES[0]) + 0.5) / SAMPLES[0] * period
    zs = low[2] + (np.arange(SAMPLES[1]) + 0.5) / SAMPLES[1] * (high[2] - low[2])
    x, z = np.meshgrid(xs, zs)
    y = low[1] + NUDGE * thin
    points = np.c_[x.ravel(), np.full(x.size, y), z.ravel()] * um
    values = np.asarray(field.at(points)).real  # (n, 3)
    component = np.argmax(np.abs(values).sum(axis=0))  # the wave's polarization
    section = pv.StructuredGrid(x, np.zeros_like(x), z)
    # A StructuredGrid orders its points with x fastest: the scalars follow.
    section["Re E"] = values[:, component].reshape(x.shape).ravel(order="F")
    clim = float(np.percentile(np.abs(values[:, component]), 99))

    view = pv.Plotter(off_screen=True)
    for i in range(PERIODS):
        shift = (i - PERIODS / 2) * period
        tile = section.copy()
        tile.points = tile.points + np.array([shift, low[1] + DEPTH / 2, 0])
        view.add_mesh(tile, scalars="Re E", clim=(-clim, clim), show_scalar_bar=False)
        box = (
            r_low[0] + shift,
            r_high[0] + shift,
            low[1],
            low[1] + DEPTH,
            r_low[2],
            r_high[2],
        )
        view.add_mesh(pv.Box(box), color="grey", opacity=0.3)
    glass = (low[0] - PERIODS / 2 * period, high[0] + (PERIODS / 2 - 1) * period)
    view.add_mesh(
        pv.Box((*glass, low[1], low[1] + DEPTH, low[2], g_high)),
        color="grey",
        opacity=0.3,
    )
    save_still(view, "metalens", "signed")
