"""Periodic solves on a curved mesh agree with the same cell meshed straight.

``cell_order2`` and ``cell_order1`` (tests/unit/data) share their corner nodes;
the order-2 file adds the mid-edge nodes and bends the elements on its sphere.
Every material is vacuum, so the sphere is invisible to the physics and both
discretise the same periodic box: their eigenfrequencies differ only by the
discretisation error of two meshes with identical corners.
"""

import numpy as np

import gfem

DATA = __import__("pathlib").Path(__file__).parent / "data"


def _frequencies(name):
    mesh = gfem.Mesh.from_gmsh(DATA / f"{name}.msh", length_unit="m")
    model = gfem.Model(
        mesh,
        materials={r: gfem.Material() for r in mesh.volumes},
        boundaries=[gfem.Periodic(("ylo", "yhi"))],
    )
    result = gfem.solve(model, gfem.Eigenmode(target=5e7, n_modes=4))
    return np.sort(np.asarray(result.omega).real)


def test_a_periodic_eigenmode_solve_matches_the_straight_twin():
    curved, straight = _frequencies("cell_order2"), _frequencies("cell_order1")
    # Identifying the wrong nodes moves the modes by 10 to 20 percent (the first
    # pair of modes lands at 25.9e7 instead of 31.4e7 rad/s); the same corners
    # on a vacuum box agree to the discretisation difference of the curved
    # sphere elements, measured below 4e-7.
    # 1e-4 leaves a margin of 250 over that and 1e3 under
    # the error.
    np.testing.assert_allclose(curved, straight, rtol=1e-4, atol=0)


def _power_flows(name):
    mesh = gfem.Mesh.from_gmsh(DATA / f"{name}.msh", length_unit="m")
    lossy = gfem.Material(eps_r=2.0, tan_delta=0.5)
    model = gfem.Model(
        mesh,
        materials={
            r: lossy if r == "heater" else gfem.Material() for r in mesh.volumes
        },
        boundaries=[gfem.Periodic(("ylo", "yhi"))],
        ports=[gfem.LumpedPort("p", "xlo", direction="+z")],
    )
    problem = gfem.Driven(frequency=5e7, excite="p", keep_fields=True)
    e = gfem.solve(model, problem).e_field(5e7)
    # the heater's own losses, and the slab x > 1 m, y < 1.5 m, whose surface
    # reaches the periodic pair at y = 0 m (tie classes key its faces)
    sides = ("heater", lambda c: (c[:, 0] > 1.0) & (c[:, 1] < 1.5))
    return np.array([float(gfem.power_flow(e, s, model=model)) for s in sides])


def test_power_flow_across_a_periodic_pair_matches_the_straight_twin():
    curved, straight = _power_flows("cell_order2"), _power_flows("cell_order1")
    # Same corners, same vacuum: the two meshes differ by the sphere's element
    # curvature only, measured 3e-6 apart. Keying the faces by NGSolve vertex
    # numbers where gfem node numbers are meant moves the slab's power by 2.3e-5,
    # so 1e-5 sits between the two (margins 3x and 2x).
    np.testing.assert_allclose(curved, straight, rtol=1e-5, atol=0)
