"""Design helpers: projection, the robust triple, filters and the layer grid."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from gfem import design

# --- project -----------------------------------------------------------------------


def test_project_keeps_void_solid_and_threshold():
    rho = jnp.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(design.project(rho, beta=8.0), [0.0, 0.5, 1.0])
    # an off-centre threshold still maps 0 -> 0 and 1 -> 1
    np.testing.assert_allclose(design.project(rho[::2], beta=8.0, eta=0.3), [0, 1])


def test_project_sharpens_towards_a_step_as_beta_grows():
    rho = jnp.array([0.4, 0.6])
    soft = design.project(rho, beta=1.0)
    hard = design.project(rho, beta=64.0)
    assert 0.39 < soft[0] < 0.41  # nearly linear at beta = 1 (tanh x ~ x)
    np.testing.assert_allclose(hard, [0.0, 1.0], atol=1e-5)


def test_project_rejects_a_threshold_outside_the_unit_interval():
    with pytest.raises(ValueError, match="eta"):
        design.project(jnp.zeros(2), beta=8.0, eta=1.2)
    with pytest.raises(ValueError, match="beta"):
        design.project(jnp.zeros(2), beta=0.0)


def test_project_gradients():
    rho = jnp.linspace(0.05, 0.95, 7)
    check_grads(lambda r: design.project(r, beta=6.0, eta=0.4), (rho,), order=2)
    check_grads(lambda b: design.project(rho, beta=b), (4.0,), order=1)


# --- robust ------------------------------------------------------------------------


def test_robust_eroded_uses_the_higher_threshold():
    rho = jnp.linspace(0, 1, 11)
    r = design.robust(rho, beta=8.0, delta=0.2)
    np.testing.assert_allclose(r.eroded, design.project(rho, beta=8.0, eta=0.7))
    np.testing.assert_allclose(r.nominal, design.project(rho, beta=8.0, eta=0.5))
    np.testing.assert_allclose(r.dilated, design.project(rho, beta=8.0, eta=0.3))
    # eroded has less solid than nominal, dilated more
    assert jnp.sum(r.eroded) < jnp.sum(r.nominal) < jnp.sum(r.dilated)


def test_robust_triple_stacks_for_vmap():
    rho = jnp.linspace(0, 1, 5)
    stacked = jnp.stack(design.robust(rho, beta=4.0, delta=0.1))
    assert stacked.shape == (3, 5)
    sums = jax.vmap(jnp.sum)(stacked)
    assert sums.shape == (3,)


def test_robust_rejects_a_delta_that_leaves_the_unit_interval():
    with pytest.raises(ValueError, match="delta"):
        design.robust(jnp.zeros(3), beta=4.0, delta=0.6)
    with pytest.raises(ValueError, match="delta"):
        design.robust(jnp.zeros(3), beta=4.0, delta=0.0)


def test_robust_gradients():
    rho = jnp.linspace(0.05, 0.95, 6)

    def worst(r):
        return jnp.max(jnp.stack(design.robust(r, beta=5.0, delta=0.1)) @ r)

    check_grads(lambda r: design.robust(r, beta=5.0, delta=0.1), (rho,), order=2)
    check_grads(worst, (rho,), order=1, modes=["rev"])


# --- LayerGrid ---------------------------------------------------------------------

MM = 1e-3


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """A 1 mm x 0.5 mm stack: substrate, a two-cell-thick design layer, air."""
    import structured_meshes as sm

    import gfem

    layers = (("substrate", 0.2), ("design", 0.1), ("air", 0.3))
    path = tmp_path_factory.mktemp("stack") / "stack.msh"
    sm.write_msh22(path, sm.plate(1.0, 0.5, layers, (8, 4, 2)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _centroids(mesh, region):
    nodes = np.asarray(mesh.nodes)
    return nodes[mesh.tetrahedra[mesh.elements(region)]].mean(axis=1)


def test_layer_grid_covers_the_region_footprint(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    assert grid.shape == (2, 4)  # (ny, nx)
    assert grid.region == "design"
    assert grid.pitch == pytest.approx(0.25 * MM)
    np.testing.assert_allclose(grid.origin, [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(grid.x, np.array([0.125, 0.375, 0.625, 0.875]) * MM)
    np.testing.assert_allclose(grid.y, np.array([0.125, 0.375]) * MM)
    assert grid.n_elements == len(stack.elements("design"))


def test_layer_grid_reproduces_a_linear_density_inside_the_pixel_centres(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    X, Y = np.meshgrid(grid.x, grid.y)
    rho = 0.2 + 0.4 * X / MM + 0.6 * Y / MM  # bilinear interpolation is exact
    values = np.asarray(grid.to_elements(jnp.asarray(rho)))
    c = _centroids(stack, "design")
    inside = (
        (c[:, 0] >= grid.x[0]) & (c[:, 0] <= grid.x[-1])
        & (c[:, 1] >= grid.y[0]) & (c[:, 1] <= grid.y[-1])
    )  # fmt: skip
    assert inside.sum() > 20
    np.testing.assert_allclose(
        values[inside], 0.2 + 0.4 * c[inside, 0] / MM + 0.6 * c[inside, 1] / MM
    )
    # outside the span of pixel centres the edge pixels' values hold
    assert values.min() >= rho.min() - 1e-12
    assert values.max() <= rho.max() + 1e-12


def test_layer_grid_is_constant_through_the_thickness(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    rho = jax.random.uniform(jax.random.key(0), grid.shape)
    values = np.asarray(grid.to_elements(rho))
    c = _centroids(stack, "design")
    lower = c[:, 2] < 0.25 * MM  # the design layer is two cells thick
    key = np.round(c[:, :2] / MM, 9)
    by_xy = {tuple(k): v for k, v in zip(key[lower], values[lower], strict=True)}
    above = [by_xy[tuple(k)] for k in key[~lower]]
    np.testing.assert_array_equal(values[~lower], above)


def test_layer_grid_warns_about_pixels_no_element_samples(stack):
    with pytest.warns(UserWarning, match=r"(?s)\d+ of 800 pixels.*coarsen"):
        grid = design.LayerGrid(stack, "design", pitch=0.025 * MM)
    assert grid.shape == (20, 40)
    assert grid.sampled.shape == grid.shape
    assert 0 < grid.sampled.sum() < grid.sampled.size
    g = jax.grad(lambda r: jnp.sum(grid.to_elements(r) ** 2))(jnp.ones(grid.shape))
    np.testing.assert_array_equal(np.asarray(g)[~grid.sampled], 0.0)
    assert (np.asarray(g)[grid.sampled] > 0).all()


def test_layer_grid_rejects_a_density_of_the_wrong_shape(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    with pytest.raises(ValueError, match=r"\(2, 4\)"):
        grid.to_elements(jnp.zeros(8))


def test_layer_grid_on_an_unknown_region_or_bad_pitch_raises(stack):
    import gfem

    with pytest.raises(gfem.RegionNotFoundError):
        design.LayerGrid(stack, "desing", pitch=0.25 * MM)
    with pytest.raises(ValueError, match="pitch"):
        design.LayerGrid(stack, "design", pitch=-1.0)


# --- LayerGrid on a sheet (a horizontal surface) ----------------------------------


def _face_centroids(mesh, surface):
    nodes = np.asarray(mesh.nodes)
    return nodes[mesh.triangles[mesh.elements(surface)]].mean(axis=1)


def test_surface_grid_covers_the_sheet_and_maps_pixels_to_faces(stack):
    sheet = "substrate__design"
    grid = design.LayerGrid(stack, sheet, pitch=0.25 * MM)
    assert grid.shape == (2, 4)
    assert grid.n_faces == len(stack.elements(sheet))
    assert grid.n_elements is None
    X, Y = np.meshgrid(grid.x, grid.y)
    rho = 0.2 + 0.4 * X / MM + 0.6 * Y / MM
    values = np.asarray(grid.to_faces(jnp.asarray(rho)))
    c = _face_centroids(stack, sheet)
    inside = (
        (c[:, 0] >= grid.x[0]) & (c[:, 0] <= grid.x[-1])
        & (c[:, 1] >= grid.y[0]) & (c[:, 1] <= grid.y[-1])
    )  # fmt: skip
    assert inside.sum() > 20
    np.testing.assert_allclose(
        values[inside], 0.2 + 0.4 * c[inside, 0] / MM + 0.6 * c[inside, 1] / MM
    )
    check_grads(grid.to_faces, (jnp.asarray(rho),), order=2)
    assert "faces" in repr(grid)


def test_surface_and_volume_grids_name_each_others_map(stack):
    faces = design.LayerGrid(stack, "substrate__design", pitch=0.25 * MM)
    volume = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    with pytest.raises(ValueError, match="to_faces"):
        faces.to_elements(jnp.zeros(faces.shape))
    with pytest.raises(ValueError, match="to_elements"):
        volume.to_faces(jnp.zeros(volume.shape))
    with pytest.raises(ValueError, match=r"\(2, 4\)"):
        faces.to_faces(jnp.zeros(8))


def test_surface_grid_needs_a_horizontal_surface(stack):
    with pytest.raises(ValueError, match=r"(?s)not horizontal.*\d+ of"):
        design.LayerGrid(stack, "side", pitch=0.25 * MM)


def test_surface_grid_filters_on_its_pixels(stack):
    grid = design.LayerGrid(stack, "substrate__design", pitch=0.25 * MM)
    blur = design.helmholtz_filter(grid, radius=0.5 * MM)
    rho = jax.random.uniform(jax.random.key(2), grid.shape)
    assert blur(rho).shape == grid.shape


def test_layer_grid_gradients(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    rho = jax.random.uniform(jax.random.key(1), grid.shape)
    check_grads(grid.to_elements, (rho,), order=2)
    jitted = jax.jit(grid.to_elements)
    np.testing.assert_allclose(jitted(rho), grid.to_elements(rho))


# --- helmholtz_filter --------------------------------------------------------------
# The filter solves -r^2 lap(u) + u = rho with Neumann edges, r = radius / (2
# sqrt(3)) (the cone filter of the same radius has the same second moment). On a
# cosine of wavenumber k that fits the Neumann edges the exact attenuation is
# 1 / (1 + r^2 k^2).


def _attenuation(radius, k):
    return 1 / (1 + (radius / (2 * np.sqrt(3))) ** 2 * k**2)


@pytest.fixture(scope="module")
def fine(tmp_path_factory):
    """A 100 x 50 pixel grid on a 1 mm x 0.5 mm single-layer plate."""
    import structured_meshes as sm

    import gfem

    path = sm.write_msh22(
        tmp_path_factory.mktemp("fine") / "fine.msh",
        sm.plate(1.0, 0.5, (("design", 0.1),), (100, 50, 1)),
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return design.LayerGrid(mesh, "design", pitch=0.01 * MM)


def test_grid_filter_attenuates_a_cosine_as_the_pde_says(fine):
    k = np.pi / (1.0 * MM)  # half a period across the 1 mm footprint
    rho = np.cos(k * fine.x)[None, :] * np.ones((fine.shape[0], 1))
    radius = 0.3 * MM
    out = np.asarray(design.helmholtz_filter(fine, radius=radius)(jnp.asarray(rho)))
    np.testing.assert_allclose(out, _attenuation(radius, k) * rho, atol=2e-3)
    np.testing.assert_allclose(
        design.helmholtz_filter(fine, radius=radius)(jnp.ones(fine.shape)), 1.0
    )  # constants pass


def test_grid_filter_conserves_the_total_and_is_symmetric(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    blur = design.helmholtz_filter(grid, radius=0.4 * MM)
    a, b = jax.random.uniform(jax.random.key(2), (2, *grid.shape))
    np.testing.assert_allclose(jnp.sum(blur(a)), jnp.sum(a))
    np.testing.assert_allclose(jnp.vdot(a, blur(b)), jnp.vdot(blur(a), b))
    assert jnp.ptp(blur(a)) < jnp.ptp(a)


def test_mesh_filter_attenuates_a_cosine_as_the_pde_says(tmp_path):
    import structured_meshes as sm

    import gfem

    path = sm.write_msh22(
        tmp_path / "bar.msh", sm.plate(1.0, 0.2, (("design", 0.2),), (24, 2, 2))
    )
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    k = np.pi / (1.0 * MM)
    c = _centroids(mesh, "design")
    radius = 0.3 * MM
    out = design.helmholtz_filter(mesh, "design", radius=radius)(np.cos(k * c[:, 0]))
    np.testing.assert_allclose(
        out, _attenuation(radius, k) * np.cos(k * c[:, 0]), atol=1e-2
    )


def test_mesh_filter_keeps_constants_and_the_volume_weighted_total(stack):
    blur = design.helmholtz_filter(stack, "design", radius=0.2 * MM)
    n = len(stack.elements("design"))
    np.testing.assert_allclose(blur(jnp.ones(n)), 1.0)
    nodes = np.asarray(stack.nodes)
    p = nodes[stack.tetrahedra[stack.elements("design")]]
    vol = np.abs(np.linalg.det(p[:, 1:] - p[:, :1])) / 6
    rho = jax.random.uniform(jax.random.key(3), (n,))
    np.testing.assert_allclose(vol @ blur(rho), vol @ rho)
    assert jnp.ptp(blur(rho)) < jnp.ptp(rho)


def test_filter_arguments_are_checked(stack):
    grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
    with pytest.raises(ValueError, match="LayerGrid"):
        design.helmholtz_filter(stack, "substrate__design", radius=1e-4)
    with pytest.raises(ValueError, match="region"):
        design.helmholtz_filter(stack, radius=1e-4)
    with pytest.raises(ValueError, match="region"):
        design.helmholtz_filter(grid, "design", radius=1e-4)
    with pytest.raises(ValueError, match="radius"):
        design.helmholtz_filter(grid, radius=0.0)
    with pytest.raises(ValueError, match=r"\(2, 4\)"):
        design.helmholtz_filter(grid, radius=1e-4)(jnp.zeros(3))


@pytest.mark.parametrize("carrier", ["mesh", "grid"])
def test_filter_gradients_and_transforms(stack, carrier):
    if carrier == "grid":
        grid = design.LayerGrid(stack, "design", pitch=0.25 * MM)
        blur, shape = design.helmholtz_filter(grid, radius=0.3 * MM), grid.shape
    else:
        blur = design.helmholtz_filter(stack, "design", radius=0.3 * MM)
        shape = (len(stack.elements("design")),)
    rho = jax.random.uniform(jax.random.key(4), shape)
    check_grads(blur, (rho,), order=2)
    np.testing.assert_allclose(jax.jit(blur)(rho), blur(rho), rtol=1e-12)
    batch = jax.random.uniform(jax.random.key(5), (3, *shape))
    np.testing.assert_allclose(
        jax.vmap(blur)(batch), np.stack([blur(b) for b in batch]), rtol=1e-12
    )
