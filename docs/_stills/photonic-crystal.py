"""Card still of the photonic-crystal tutorial: the dielectric-band mode."""

import numpy as np

TILES = 3  # cells drawn along each axis
SAMPLES = 60  # field samples per cell side
ROD_HEIGHT = 0.8  # of the lattice constant: the rods are drawn this tall


def card(ns):
    """Draw Re E_z of band 0 at M, Bloch-tiled, under the rods (grey cylinders)."""
    import pyvista as pv
    from gfem_stills import save_still

    a, mm, mesh = ns["A"] / ns["mm"], ns["mm"], ns["mesh"]
    index, band = ns["i_m"], 0  # the dielectric band at M
    xs = (np.arange(SAMPLES) + 0.5) / SAMPLES * a  # mm, inside the cell
    x, y = np.meshgrid(xs, xs)
    points = np.c_[
        x.ravel(), y.ravel(), np.full(x.size, mesh.section_thickness / 2 / mm)
    ]
    ez = np.asarray(ns["ez_bands"].e_field(index, band).at(points * mm))[:, 2]
    ez = ez * np.exp(
        -1j * np.angle(ez[np.argmax(np.abs(ez))])
    )  # the global phase: real
    # M is k = (pi, pi) / a: from one cell to the next the Bloch phase is -1.
    values = ez.real
    clim = float(np.percentile(np.abs(values), 99))

    view = pv.Plotter(off_screen=True)
    for i in range(TILES):
        for j in range(TILES):
            tile = pv.StructuredGrid(x + i * a, y + j * a, np.zeros_like(x))
            tile["Re E_z"] = (-1) ** (i + j) * values.reshape(x.shape).ravel(order="F")
            view.add_mesh(
                tile, scalars="Re E_z", clim=(-clim, clim), show_scalar_bar=False
            )
            rod = pv.Cylinder(
                center=((i + 0.5) * a, (j + 0.5) * a, ROD_HEIGHT * a / 2),
                direction=(0, 0, 1),
                radius=ns["R_ROD"] / mm,
                height=ROD_HEIGHT * a,
                resolution=64,
            )
            view.add_mesh(rod, color="grey", opacity=0.3)
    save_still(view, "photonic-crystal", "signed")
