"""Card still of the meta-atom tutorial: the optimized cube cut open at y = 0."""

import numpy as np

CUBE = 0.35  # um, the design cube's side
VOXEL = 0.0125  # um, the grid the element densities are resampled on
SOLID = 0.5  # the density above which a voxel is drawn as material


def card(ns):
    """Resample the final element densities on a voxel grid; cut the cube at y = 0.

    The tutorial keeps the design, not a field, so the cut face carries the
    physical density where the plane crosses material.
    """
    import pyvista as pv
    from gfem_stills import save_still
    from scipy.spatial import cKDTree

    out = ns["out"]
    centres = out["centres"] / 1e-6  # m to um
    density = out["density_final"]
    n = round(CUBE / VOXEL)
    axis = -CUBE / 2 + (np.arange(n) + 0.5) * VOXEL
    voxels = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    # Each voxel takes the density of the nearest element centroid.
    _, nearest = cKDTree(centres).query(voxels.reshape(-1, 3))
    grid = pv.ImageData(
        dimensions=(n + 1, n + 1, n + 1),
        spacing=(VOXEL,) * 3,
        origin=(-CUBE / 2,) * 3,
    )
    grid.cell_data["density"] = density[nearest].reshape(n, n, n).ravel(order="F")
    solid = grid.threshold(SOLID, scalars="density")
    # The cube's back half is a grey body; the cut face shows the density.
    back = solid.clip_box((-CUBE, CUBE, VOXEL, CUBE, -CUBE, CUBE), invert=False)
    cut = solid.slice("y", origin=(0, VOXEL - 1e-4, 0))
    view = pv.Plotter(off_screen=True)
    view.add_mesh(back.extract_surface(), color="grey")
    view.add_mesh(cut, scalars="density", clim=(0, 1), show_scalar_bar=False)
    save_still(view, "meta-atom", "magnitude")
