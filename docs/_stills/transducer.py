"""Card still of the transducer tutorial: the film warped by the acoustic wave."""

import numpy as np

APERTURE = 12.0  # um of finger drawn along the invariant axis
LIFT = 5.0  # the section's heights are stretched by this, or the film is a line
DRAWN_DISPLACEMENT = 0.5  # um: the largest warped displacement, so the wave shows


def card(ns):
    """Extrude the section along the fingers, warped by the real displacement."""
    import pyvista as pv
    from gfem_stills import save_still

    um = ns["um"]
    section = ns["section"]
    mesh = section.mesh
    nodes = np.asarray(mesh.nodes)
    elements = np.asarray(mesh.volume_elements)
    disp = ns["res"].displacement(ns["f_wave"])

    def slab(regions, span, *, field=False):
        # The triangles of `regions` inside `span`, extruded along y. Each triangle
        # keeps its own corners, nudged inwards, so a probe on an interface
        # stays in its own region.
        rows = np.concatenate([mesh.elements(r) for r in regions])
        centre = nodes[elements[rows]].mean(axis=1)[:, 0] / um
        rows = rows[(centre > span[0]) & (centre < span[1])]
        corners = nodes[elements[rows]]
        corners = corners + 1e-3 * (corners.mean(axis=1, keepdims=True) - corners)
        xy = corners.reshape(-1, 2)
        points = np.c_[xy[:, 0] / um, np.zeros(len(xy)), LIFT * xy[:, 1] / um]
        faces = np.c_[
            np.full(len(rows), 3), np.arange(3 * len(rows)).reshape(-1, 3)
        ].ravel()
        surface = pv.PolyData(points, faces=faces)
        u = {k: np.asarray(v.at(xy)).real for k, v in disp.items()}
        surface.point_data["u"] = np.c_[u["x"], u["z"], LIFT * u["y"]]
        surface.point_data["Re u_z"] = u["z"]
        if not field:  # a body: one welded surface, so no inner walls show through
            surface = surface.clean(tolerance=1e-3)
        return surface.extrude((0, APERTURE, 0), capping=True)

    x_a, x_b = np.array(section.interior) / um
    span = (x_a - 1, x_b + 1)
    film = slab(["ln"], span, field=True)
    peak = np.abs(film["Re u_z"]).max()
    factor = DRAWN_DISPLACEMENT / np.abs(film["u"]).max()
    film = film.warp_by_vector("u", factor=factor)
    fingers = slab(["signal", "ground"], span).warp_by_vector("u", factor=factor)
    oxide = pv.Box((span[0], span[1], 0, APERTURE, LIFT * -0.8, LIFT * -0.18))
    view = pv.Plotter(off_screen=True)
    view.add_mesh(film, scalars="Re u_z", clim=(-peak, peak), show_scalar_bar=False)
    view.add_mesh(fingers, color="silver")
    view.add_mesh(oxide, color="lightblue")
    save_still(view, "transducer", "signed")
