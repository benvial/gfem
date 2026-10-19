"""Boundary morphs: a linear map from normal displacements to a moved mesh."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem.shape import boundary_morph

DATA = Path(__file__).parent / "data"
UM = 1e-6
METAL = gfem.enclosed_by("metal_xy", "metal_z")


@pytest.fixture(scope="module")
def mesh():
    return gfem.Mesh.from_gmsh(DATA / "two_box_msh41.msh")


@pytest.fixture(scope="module")
def model(mesh):
    return gfem.Model(
        mesh,
        materials={"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()},
        excluded=(METAL,),
    )


def plus_x_wall(p):
    """The conductor's +x sidewall, at x = 1.5 um."""
    return np.abs(p[:, 0] - 1.5 * UM) < 1e-12


@pytest.fixture(scope="module")
def wall(model):
    return boundary_morph(model, "metal_z", where=plus_x_wall)


def test_parameters_are_the_selected_nodes_with_normals_into_the_domain(wall):
    assert wall.n_params == len(wall.points) == len(wall.normals) > 0
    assert wall.points.shape == (wall.n_params, 3)
    np.testing.assert_allclose(wall.points[:, 0], 1.5 * UM, rtol=0, atol=1e-18)
    # the conductor is excluded, so t > 0 moves its wall into the dielectric (+x)
    np.testing.assert_allclose(wall.normals, np.tile([1.0, 0, 0], (wall.n_params, 1)))


def test_zero_displacement_gives_the_imported_mesh(wall, mesh):
    moved = wall(jnp.zeros(wall.n_params))
    assert isinstance(moved, gfem.Mesh)
    np.testing.assert_array_equal(np.asarray(moved.nodes), mesh.imported_nodes)
    np.testing.assert_array_equal(moved.tetrahedra, mesh.tetrahedra)


def _bump(morph):
    """A smooth bump over the wall, peak 1, zero at its edges."""
    y, z = morph.points[:, 1] / UM, morph.points[:, 2] / UM
    return np.sin(np.pi * (y - 0.5)) * np.sin(np.pi * (z - 0.3) / 0.4)


def test_moving_nodes_move_along_their_normals(wall, mesh):
    t = 0.1 * UM * _bump(wall)
    moved = np.asarray(wall(t).nodes)
    params = _node_index(mesh, wall.points)
    np.testing.assert_allclose(
        moved[params] - mesh.imported_nodes[params],
        t[:, None] * wall.normals,
        rtol=0,
        atol=1e-20,
    )


def _node_index(mesh, points):
    d = np.linalg.norm(mesh.imported_nodes[None] - points[:, None], axis=2)
    return d.argmin(axis=1)


@pytest.mark.parametrize("extension", ["elastic", "laplace"])
def test_every_other_surface_slides_in_its_plane(model, mesh, extension):
    morph = boundary_morph(model, "metal_z", where=plus_x_wall, extension=extension)
    moved = np.asarray(morph(0.2 * UM * _bump(morph)).nodes)
    shift = moved - mesh.imported_nodes
    assert np.abs(shift).max() > 0.05 * UM
    # every surface of this mesh is a union of axis-aligned planes
    for name in ("air__sub", "sub__None", "air__None", "metal_xy"):
        tris = mesh.triangles[mesh.elements(name)]
        x = mesh.imported_nodes
        n = np.cross(x[tris[:, 1]] - x[tris[:, 0]], x[tris[:, 2]] - x[tris[:, 0]])
        axis = np.abs(n).argmax(axis=1)
        for k in range(3):
            nodes = np.unique(tris[axis == k])
            moving_here = np.isin(nodes, _node_index(mesh, morph.points))
            off_plane = np.abs(shift[nodes[~moving_here], k])
            assert off_plane.max(initial=0) <= 1e-12 * np.abs(shift).max(), name
    # interior nodes do follow
    assert np.count_nonzero(np.linalg.norm(shift, axis=1)) > morph.n_params


def test_the_map_is_linear_and_path_independent(wall, mesh):
    t = 0.05 * UM * _bump(wall)
    x0 = mesh.imported_nodes
    once = np.asarray(wall(t).nodes) - x0
    wall(3 * t)  # evaluating elsewhere first changes nothing
    twice = np.asarray(wall(2 * t).nodes) - x0
    # atol: rounding of x0 + u, x0 ~ 2 um
    np.testing.assert_allclose(twice, 2 * once, rtol=1e-12, atol=1e-21)


def _objective(morph):
    def j(t):
        return jnp.sum(jnp.sin(1e6 * morph(t).nodes) ** 2)

    return j


def test_gradient_matches_central_differences(wall):
    j = _objective(wall)
    t = 0.05 * UM * _bump(wall)
    g = np.asarray(jax.grad(j)(t))
    rng = np.random.default_rng(0)
    for _ in range(3):
        dt = rng.standard_normal(wall.n_params) * 1e-3 * UM
        fd = (j(t + dt) - j(t - dt)) / 2
        assert g @ dt == pytest.approx(float(fd), rel=1e-6)


def test_jvp_jit_and_vmap_agree_with_grad(wall):
    j = _objective(wall)
    t = 0.05 * UM * _bump(wall)
    g = jax.grad(j)(t)
    dt = jnp.asarray(np.random.default_rng(1).standard_normal(wall.n_params)) * UM
    _, tangent = jax.jvp(j, (t,), (dt,))
    assert float(tangent) == pytest.approx(float(g @ dt), rel=1e-10)
    np.testing.assert_allclose(jax.jit(jax.grad(j))(t), g, rtol=1e-10)
    ts = jnp.stack([t, 2 * t, -t])
    batched = jax.vmap(jax.grad(j))(ts)
    looped = np.stack([jax.grad(j)(ti) for ti in ts])
    np.testing.assert_allclose(batched, looped, rtol=1e-12)
    jitted = jax.jit(lambda t: wall(t).nodes)(t)
    np.testing.assert_allclose(jitted, wall(t).nodes, rtol=1e-14)


def _on(mesh, surface):
    return np.unique(mesh.triangles[mesh.elements(surface)])


def test_fixed_surfaces_are_held_and_hold_beats_moving(model, mesh, wall):
    held = boundary_morph(model, "metal_z", where=plus_x_wall, fixed=("metal_xy",))
    # the wall's top and bottom edges lie on metal_xy: held, so not parameters
    assert 0 < held.n_params < wall.n_params
    params = _node_index(mesh, held.points)
    assert not np.isin(params, _on(mesh, "metal_xy")).any()
    t = 0.1 * UM * np.ones(held.n_params)
    shift = np.asarray(held(t).nodes) - mesh.imported_nodes
    assert np.abs(shift[_on(mesh, "metal_xy")]).max() == 0


@pytest.fixture(scope="module")
def ported(mesh):
    return gfem.Model(
        mesh,
        materials={"sub": gfem.Material(eps_r=11.7), "air": gfem.Material()},
        ports=[gfem.LumpedPort("p1", "sub__None", direction="+z")],
    )


def test_interfaces_need_grows_to_fix_the_sign(ported):
    with pytest.raises(ValueError, match="grows="):
        boundary_morph(ported, "air__sub")
    up = boundary_morph(ported, "air__sub", grows="sub")
    down = boundary_morph(ported, "air__sub", grows="air")
    np.testing.assert_allclose(up.normals, np.tile([0, 0, 1.0], (up.n_params, 1)))
    np.testing.assert_allclose(down.normals, -up.normals)


def test_port_surfaces_are_held(ported, mesh):
    morph = boundary_morph(ported, "air__sub", grows="sub")
    port = _on(mesh, "sub__None")
    # the interface's rim lies on the port: those nodes are not parameters
    interface = _on(mesh, "air__sub")
    assert morph.n_params == np.count_nonzero(~np.isin(interface, port))
    moved = np.asarray(morph(0.1 * UM * np.ones(morph.n_params)).nodes)
    np.testing.assert_array_equal(moved[port], mesh.imported_nodes[port])


def test_grows_on_a_one_sided_surface_is_refused(model):
    with pytest.raises(ValueError, match="grows='sub' has no effect"):
        boundary_morph(model, "metal_z", where=plus_x_wall, grows="sub")


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"moving": "metal_q"}, gfem.RegionNotFoundError, "metal_z"),
        ({"moving": "sub"}, ValueError, "is a volume"),
        ({"moving": "metal_z", "where": lambda p: p[:, 0] > 1}, ValueError, "none of"),
        ({"moving": "metal_z", "where": lambda p: p[:, 0]}, ValueError, "boolean"),
        ({"moving": "metal_z", "extension": "spring"}, ValueError, "elastic"),
        ({"moving": "metal_z", "fixed": "metal_xy"}, TypeError, "tuple"),
        ({"moving": "metal_z", "fixed": ("metal_z",)}, ValueError, "both moving"),
    ],
)
def test_bad_arguments_raise_with_a_remedy(model, kwargs, error, match):
    with pytest.raises(error, match=match):
        boundary_morph(model, **kwargs)


def _wall_top_edge(p):
    return plus_x_wall(p) & (np.abs(p[:, 2] - 0.7 * UM) < 1e-12)


def test_a_selection_that_is_entirely_held_raises(model):
    with pytest.raises(ValueError, match="held"):
        boundary_morph(model, "metal_z", where=_wall_top_edge, fixed=("metal_xy",))


def test_a_morph_is_immutable(wall):
    with pytest.raises(AttributeError, match="immutable"):
        wall.n_params = 3
    with pytest.raises(ValueError, match="read-only"):
        wall.points[0, 0] = 0.0


def test_wrong_number_of_displacements_raises(wall):
    with pytest.raises(ValueError, match=f"takes {wall.n_params} normal"):
        wall(np.zeros(wall.n_params + 1))
