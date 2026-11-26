"""Card still of the nonlinear section: a ridge guide and its pump mode."""

import sys
from pathlib import Path

import numpy as np

RECIPES = Path(__file__).resolve().parents[1] / "recipes"
WIDTH, FILM, ETCH = 1.4, 0.6, 0.3  # um, the second-harmonic recipe's ridge
LENGTH, LIFT, HALF = (
    7.0,
    2.0,
    1.6,
)  # um of guide behind the face; heights stretched; half width


    """Solve the recipe's ridge at 1550 nm; draw its pump mode on an extruded guide."""
    """Solve the recipe's ridge at 1550 nm, its pump mode on the end of an extruded guide."""
    import pyvista as pv
    from gfem_stills import save_still
    from scipy.constants import c

    import gfem

    sys.path.insert(0, str(RECIPES))
    try:
        from ridge_section import ridge_section
    finally:
        sys.path.remove(str(RECIPES))

    um = 1e-6
    mesh = ridge_section(WIDTH * um)
    frequency = c / 1.55e-6
    x_cut = ((0, 1, 0), (0, 0, 1), (1, 0, 0))
    film = gfem.materials.lithium_niobate(frequency=frequency, axes=x_cut)
    model = gfem.Model(
        mesh,
        materials={
            "film": film,
            "box": gfem.Material(eps_r=1.444**2),
            "air": gfem.Material(),
        },
        ports=[gfem.WavePort("wg", "front")],
    )
    pump = gfem.solve(
        model,
        gfem.PortMode("wg", np.array([frequency])),
        options=gfem.SolverOptions(order=3),
    )
    # The end face: the section's surface cells, y up, the guide along +y.
    grid = pv.from_meshio(mesh.to_meshio())
    cells = mesh.elements("front") + len(mesh.tetrahedra)
    face = grid.extract_cells(cells).extract_surface(algorithm="dataset_surface")
    xy = np.asarray(face.points)[:, :2]
    face.points = np.c_[xy[:, 0] / um, np.zeros(len(xy)), LIFT * xy[:, 1] / um]
    field = np.asarray(pump.e_field.at(np.c_[xy, np.zeros(len(xy))]))[0]
    face["|E|"] = np.linalg.norm(field, axis=1)
    face = face.clip_box((-HALF, HALF, -1, 1, -0.4 * LIFT, 1.0 * LIFT), invert=False)
    face = face.translate((0, -0.01, 0))  # just in front of the guide's end
    slab = pv.Box((-HALF, HALF, 0, LENGTH, 0, LIFT * (FILM - ETCH)))
    ridge = pv.Box(
        (-WIDTH / 2, WIDTH / 2, 0, LENGTH, LIFT * (FILM - ETCH), LIFT * FILM)
    )
    oxide = pv.Box((-HALF, HALF, 0, LENGTH, -0.4 * LIFT, 0))
    view = pv.Plotter(off_screen=True)
    view.add_mesh(face, scalars="|E|", show_scalar_bar=False)
    view.add_mesh(slab, color="lightgrey")
    view.add_mesh(ridge, color="lightgrey")
    view.add_mesh(oxide, color="lightblue")
    save_still(view, "waveguide-nonlinearity", "magnitude")
