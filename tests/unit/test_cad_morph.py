"""The lazy gmsh import of the CAD layer."""

import importlib
import sys

import pytest

from gfem import _cad


def test_a_missing_gmsh_says_which_extra_to_install(monkeypatch):
    # A None entry in sys.modules makes `import gmsh` raise ImportError.
    monkeypatch.setitem(sys.modules, "gmsh", None)
    with pytest.raises(ImportError) as info:
        _cad.import_gmsh()
    assert str(info.value) == (
        "gfem.shape.cad_morph needs gmsh, an optional dependency: "
        "pip install 'gfem[gmsh]'"
    )
    assert isinstance(info.value.__cause__, ImportError)


def test_a_gmsh_library_that_fails_to_load_names_the_apt_packages(monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("libGLU.so.1: cannot open")

    monkeypatch.setattr(_cad.importlib, "import_module", broken)
    with pytest.raises(OSError) as info:
        _cad.import_gmsh()
    message = str(info.value)
    assert message == (
        "gmsh is installed but its library failed to load "
        "(libGLU.so.1: cannot open). The PyPI gmsh wheel needs system "
        "OpenGL/X11 libraries; on Debian/Ubuntu: apt-get install libglu1-mesa "
        "libxcursor1 libxft2 libxinerama1 libgl1 libgomp1."
    )
    assert not isinstance(info.value, ImportError)
    assert isinstance(info.value.__cause__, OSError)


@pytest.fixture(scope="module")
def gmsh():
    # Inside a fixture, not at module level: collecting this file must not
    # import gmsh in jobs that have no system libraries for it.
    pytest.importorskip("gmsh")
    return _cad.import_gmsh()


def test_import_gmsh_returns_the_module_and_an_occ_box_meshes(gmsh):
    assert gmsh is importlib.import_module("gmsh")
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("smoke_box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
        gmsh.model.mesh.generate(3)
        tags, _, _ = gmsh.model.mesh.getNodes()
        assert len(tags) >= 8
        types, _, _ = gmsh.model.mesh.getElements(3)
        assert 4 in types  # linear tetrahedra
    finally:
        gmsh.finalize()


# --- the CAD morph -------------------------------------------------------------

import signal  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import cad_cases as cc  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import gfem  # noqa: E402
from gfem import _morph, _mpi  # noqa: E402
from gfem.mesh import LENGTH_UNITS  # noqa: E402

UM = LENGTH_UNITS["um"]
P0 = {"width": cc.W0, "height": cc.T0}
EPS = np.finfo(float).eps

#: The mesh size of the shared strip: a few elements across the 0.5 um core.
LC = 0.15


def strip_morph(lc=LC, **kwargs):
    return gfem.shape.cad_morph(cc.strip_build(lc), P0, cc.make_model, **kwargs)


@pytest.fixture(scope="module")
def morph(gmsh):  # noqa: ARG001 - the fixture skips the file without gmsh
    return strip_morph()


@pytest.fixture
def no_gmsh(monkeypatch):
    """Make any attempt to build a CAD model fail: argument checks come first."""

    def reached(*_args, **_kwargs):
        raise AssertionError("the CAD model must not be built")

    monkeypatch.setattr(_cad, "build_cad", reached)


# --- design velocities ---------------------------------------------------------


@pytest.mark.usefixtures("gmsh")
def test_the_strip_velocities_are_the_side_walls_moving_by_half_a_unit():
    cad = _cad.build_cad(cc.strip_build(LC), P0, "um")
    x = np.asarray(cad.mesh.imported_nodes) / UM
    v = cad.velocity.reshape(-1, 3, 2)
    # Both parameters are exact on planes: the closest point on a plane moves
    # with it, so the only error is the round-off of the two coordinates,
    # half an ulp of a coordinate below 2 um (2.2e-16 each) over the step
    # difference 2 h = 2^-15 um: 2 * 2.2e-16 / 3.1e-5 = 1.4e-11. The
    # closest-point search of the kernel adds its own tolerance, measured
    # below 1e-11 on planes and cylinders; 1e-9 leaves two orders of margin and
    # is far below the 0.5 that tells a wall from a node at rest.
    atol = 1e-9
    np.testing.assert_allclose(
        v[:, :, 0], cc.strip_velocity(x, cc.W0), rtol=0, atol=atol
    )
    top = np.zeros_like(x)
    on_top = (np.abs(x[:, 2] - (cc.Z0 + cc.T0)) < 1e-9) & (
        np.abs(x[:, 1] - 1) < cc.W0 / 2 + 1e-9
    )
    top[on_top, 2] = 1.0
    np.testing.assert_allclose(v[:, :, 1], top, rtol=0, atol=atol)
    assert np.count_nonzero(v[:, :, 0].any(axis=1)) > 100


@pytest.mark.usefixtures("gmsh")
def test_the_disk_velocity_is_radial_on_the_wall_and_its_circles():
    cad = _cad.build_cad(cc.disk_build(0.2), {"radius": cc.R0}, "um")
    x = np.asarray(cad.mesh.imported_nodes) / UM
    # The wall is a cylinder and its circles arcs of one: the closest point
    # of a node moves radially with no truncation error, so only the
    # round-off above (1.4e-11) and the kernel's projection tolerance
    # remain, measured below 1e-11; 1e-9 as for the strip.
    np.testing.assert_allclose(
        cad.velocity.reshape(-1, 3), cc.disk_velocity(x, cc.R0), rtol=0, atol=1e-9
    )


@pytest.mark.usefixtures("gmsh")
def test_the_default_step_is_a_power_of_two_near_the_cube_root_of_epsilon():
    cad = _cad.build_cad(cc.strip_build(LC), P0, "um")
    # |x| <= L = 3.0 (um) and the bounding box is padded by 1e-7 um by the kernel:
    # eps^(1/3) L = 1.8e-5, whose nearest power of two is 2^-16.
    assert cad.steps == (2.0**-16, 2.0**-16)
    assert cad.size == pytest.approx(3.0, abs=1e-6)


@pytest.mark.usefixtures("gmsh")
def test_a_step_override_is_used_for_that_parameter_only():
    cad = _cad.build_cad(cc.strip_build(LC), P0, "um", {"height": 2.0**-10})
    assert cad.steps == (2.0**-16, 2.0**-10)


def test_the_velocity_is_in_metres_per_unit_and_a_wall_keeps_its_normal_part(morph):
    x = np.asarray(morph.model.mesh.imported_nodes) / UM
    wall = cc.strip_velocity(x, cc.W0)[:, 1] != 0
    # A wall slides in its plane, so its y component is the design velocity
    # (half a unit, in m per um) to the projection's round-off (1e-16 of
    # the velocity scale 1e-6 m/um) and the slide basis's orthonormality
    # (1e-16 relative).
    y = morph.velocities["width"][:, 1]
    np.testing.assert_allclose(
        y[wall], UM * cc.strip_velocity(x, cc.W0)[wall, 1], rtol=1e-9, atol=0
    )


# --- the morph -----------------------------------------------------------------


def test_zero_displacement_returns_the_imported_nodes_bitwise(morph):
    x0 = np.asarray(morph.model.mesh.imported_nodes)
    for p in (morph.p0, {}, {"width": cc.W0}):
        assert np.array_equal(np.asarray(morph(p).nodes), x0)


def test_the_morph_is_linear_in_the_parameters(morph):
    x0 = np.asarray(morph.model.mesh.imported_nodes)

    def moved(w, t):
        return np.asarray(morph({"width": cc.W0 + w, "height": cc.T0 + t}).nodes) - x0

    a, b = moved(0.02, 0.0), moved(0.0, 0.015)
    both = moved(0.02, 0.015)
    # Each displacement is a difference of two coordinates below 2e-6 m, each
    # rounded by half an ulp (eps/2 max|x0|), and the sum adds three more
    # roundings: 4 eps max|x0| bounds it; the products W dp agree to rtol 1e-12.
    atol = 4 * EPS * np.abs(x0).max()
    np.testing.assert_allclose(both, a + b, rtol=1e-12, atol=atol)


def test_jit_vmap_and_a_loop_give_the_same_nodes(morph):
    widths = jnp.array([0.45, 0.5, 0.55])
    x0 = np.asarray(morph.model.mesh.imported_nodes)

    def nodes(w):
        return morph({"width": w}).nodes

    looped = np.stack([np.asarray(nodes(w)) for w in widths])
    # The three evaluate the same matrix product; the compiler may fuse or
    # reorder its sums, a few ulps of the largest coordinate each.
    atol = 16 * EPS * np.abs(x0).max()
    np.testing.assert_allclose(
        np.asarray(jax.jit(nodes)(widths[2])), looped[2], rtol=1e-10, atol=atol
    )
    np.testing.assert_allclose(
        np.asarray(jax.vmap(nodes)(widths)), looped, rtol=1e-10, atol=atol
    )


def test_a_partial_dict_keeps_the_other_parameters_at_p0(morph):
    full = np.asarray(morph({"width": 0.52, "height": cc.T0}).nodes)
    part = np.asarray(morph({"width": 0.52}).nodes)
    np.testing.assert_allclose(part, full, rtol=1e-12, atol=0)


def test_the_gradient_is_a_dict_with_the_names_of_p0(morph):
    x0 = np.asarray(morph.model.mesh.imported_nodes)
    grad = jax.grad(lambda p: jnp.sum(morph(p).nodes ** 2))(morph.p0)
    assert set(grad) == set(morph.p0)
    for name in morph.names:
        w = morph.velocities[name]
        exact = 2 * np.sum(x0 * w)
        # The sums of 5e3 products differ in order, n eps of their absolute sum.
        atol = 1e-10 * 2 * np.sum(np.abs(x0 * w))
        np.testing.assert_allclose(float(grad[name]), exact, rtol=0, atol=atol)


def test_a_capacitance_gradient_through_the_morph_has_the_physical_sign(morph):
    # A wider and a taller dielectric core raise the capacitance.
    grad = jax.grad(lambda p: cc.capacitance(morph.model.replace(mesh=morph(p))))(
        morph.p0
    )
    assert float(grad["width"]) > 0
    assert float(grad["height"]) > 0


def test_a_concrete_move_that_inverts_an_element_raises_at_once(morph):
    with pytest.raises(gfem.shape.MeshQualityError):
        morph({"width": 5.0})
    # traced, the check waits for the solve
    jax.jit(lambda w: morph({"width": w}).nodes)(5.0)


def test_the_morph_is_immutable_and_has_a_repr(morph):
    with pytest.raises(AttributeError, match="CadMorph is immutable") as info:
        morph.names = ()
    assert str(info.value) == (
        "CadMorph is immutable; build a new one with gfem.shape.cad_morph() "
        "or CadMorph.remesh()"
    )
    assert repr(morph) == (
        "CadMorph(build=build, p0={'width': 0.5, 'height': 0.3}, "
        "length_unit='um', extension='elastic')"
    )
    assert jax.tree_util.tree_leaves(morph) == []


def test_the_attributes(morph):
    assert morph.names == ("width", "height")
    assert morph.n_params == 2
    assert morph.length_unit == "um"
    assert morph.extension == "elastic"
    assert isinstance(morph.model, gfem.Model)
    first = morph.p0
    first["width"] = 9.0
    assert morph.p0 == P0  # a fresh dict each time


def test_the_velocities_are_read_only_columns_of_the_displacement(morph):
    assert set(morph.velocities) == {"width", "height"}
    n = len(morph.model.mesh.imported_nodes)
    for v in morph.velocities.values():
        assert v.shape == (n, 3)
        assert not v.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            v[0, 0] = 1.0
    # one unit of width moves the wall by half a unit, 0.5 um
    assert np.abs(morph.velocities["width"]).max() == pytest.approx(0.5 * UM, rel=1e-6)


def test_unknown_parameter_names_list_the_known_ones(morph):
    with pytest.raises(
        ValueError, match=r"Unknown parameter.*'widht'.*'width', 'height'"
    ):
        morph({"widht": 0.5})
    with pytest.raises(ValueError, match=r"'widht'.*'width', 'height'"):
        morph.remesh({"widht": 0.5})


# --- remesh --------------------------------------------------------------------


def test_remesh_restarts_at_the_new_values_and_keeps_them_absolute(morph):
    new = morph.remesh({"width": 0.55})
    assert new is not morph
    assert new.p0 == {"width": 0.55, "height": cc.T0}
    assert morph.p0 == P0
    assert new.names == morph.names
    assert new.length_unit == morph.length_unit
    assert new.extension == morph.extension
    mesh = new.model.mesh
    core = mesh.region("core").tag
    nodes = np.unique(mesh.tetrahedra[mesh.tetrahedron_tags == core])
    y = mesh.imported_nodes[nodes, 1]
    # the core is 0.55 um wide: its y extent is the difference of two
    # coordinates near 1.2e-6 m, accurate to an ulp (2e-22 m).
    assert np.ptp(y) == pytest.approx(0.55 * UM, abs=1e-18)
    assert np.array_equal(np.asarray(new(new.p0).nodes), mesh.imported_nodes)


def test_remesh_fills_a_partial_dict_from_the_current_p0(morph):
    new = morph.remesh({"height": 0.35})
    assert new.p0 == {"width": cc.W0, "height": 0.35}


def test_remesh_is_concrete_only(morph):
    with pytest.raises(TypeError, match="remesh rebuilds the CAD model") as info:
        jax.jit(lambda w: morph.remesh({"width": w}))(0.5)
    assert str(info.value) == (
        "remesh rebuilds the CAD model; call it outside jax transformations "
        "with concrete values"
    )


@pytest.mark.usefixtures("gmsh")
def test_remesh_reuses_the_arguments_of_the_morph(monkeypatch):
    first = gfem.shape.cad_morph(
        cc.strip_build(0.25),
        P0,
        cc.make_model,
        step={"width": 2.0**-10},
        extension="laplace",
    )
    seen = []
    build_cad = _cad.build_cad

    def spy(build, p0, length_unit, step=None):
        seen.append((dict(p0), length_unit, dict(step)))
        return build_cad(build, p0, length_unit, step)

    monkeypatch.setattr(_cad, "build_cad", spy)
    again = first.remesh({"width": 0.52})
    assert seen == [({"width": 0.52, "height": cc.T0}, "um", {"width": 2.0**-10})]
    assert again.extension == "laplace"


# --- the build contract --------------------------------------------------------


def stub_comm(size):
    class Comm:
        rank = 0

    comm = Comm()
    comm.size = size
    return comm


@pytest.mark.usefixtures("no_gmsh")
def test_above_one_rank_cad_morph_refuses_before_gmsh_is_imported(monkeypatch):
    monkeypatch.setattr(_mpi, "resolve_comm", lambda _comm=None: stub_comm(4))
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        gfem.shape.cad_morph(cc.strip_build(LC), P0, cc.make_model)
    assert caught.value.feature == "cad_morph"
    assert str(caught.value).endswith(
        "gfem.shape.cad_morph builds the CAD model with gmsh, which is not "
        "distributed; build and optimize on one process"
    )


@pytest.mark.usefixtures("no_gmsh")
def test_above_one_rank_remesh_refuses_too(morph, monkeypatch):
    monkeypatch.setattr(_mpi, "resolve_comm", lambda _comm=None: stub_comm(2))
    with pytest.raises(gfem.DistributedUnsupportedError):
        morph.remesh({"width": 0.55})


@pytest.mark.usefixtures("no_gmsh")
@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"p0": {"width": float("nan")}}, ValueError, "'width'.*not finite"),
        ({"p0": {"width": float("inf")}}, ValueError, "not finite"),
        ({"p0": {"1width": 0.5}}, ValueError, "Python identifiers.*'1width'"),
        ({"p0": {}}, ValueError, "non-empty mapping"),
        ({"p0": [0.5]}, ValueError, "non-empty mapping"),
        ({"step": {"depth": 0.1}}, ValueError, "step names 'depth'.*'width', 'height'"),
        ({"step": {"width": 0.0}}, ValueError, "step\\['width'\\].*positive"),
        ({"step": {"width": -1e-3}}, ValueError, "positive, finite"),
        ({"step": {"width": float("nan")}}, ValueError, "positive, finite"),
        ({"step": {"width": float("inf")}}, ValueError, "positive, finite"),
        ({"length_unit": "inch"}, ValueError, "inch"),
        ({"extension": "rubber"}, ValueError, "Unknown extension 'rubber'"),
        ({"sliding_ports": "p1"}, TypeError, "sliding_ports= takes a list or tuple"),
        ({"build": 3}, TypeError, "build must be a function"),
        ({"make_model": None}, TypeError, "make_model must be a function"),
    ],
)
def test_bad_arguments_are_refused_before_any_gmsh_work(kwargs, error, match):
    arguments = {
        "build": cc.strip_build(LC),
        "p0": P0,
        "make_model": cc.make_model,
    } | kwargs
    with pytest.raises(error, match=match):
        gfem.shape.cad_morph(**arguments)


@pytest.mark.usefixtures("no_gmsh")
def test_a_traced_p0_is_refused():
    def f(w):
        return gfem.shape.cad_morph(cc.strip_build(LC), {"width": w}, cc.make_model)

    with pytest.raises(TypeError, match="traced by JAX"):
        jax.jit(f)(0.5)


@pytest.mark.usefixtures("gmsh")
def test_make_model_must_return_a_model():
    with pytest.raises(TypeError, match=r"make_model must return a gfem\.Model"):
        gfem.shape.cad_morph(cc.strip_build(0.3), P0, lambda mesh: mesh)


@pytest.mark.usefixtures("gmsh")
def test_make_model_must_use_the_mesh_it_is_given():
    other = gfem.shape.cad_morph(cc.strip_build(0.25), P0, cc.make_model).model

    with pytest.raises(ValueError, match="on the mesh it is given"):
        gfem.shape.cad_morph(cc.strip_build(0.3), P0, lambda _mesh: other)


def test_an_exception_in_make_model_propagates_with_the_parameters(gmsh):
    def make_model(_mesh):
        raise RuntimeError("no materials")

    with pytest.raises(RuntimeError, match="no materials") as info:
        gfem.shape.cad_morph(cc.strip_build(0.3), P0, make_model)
    assert any("make_model" in n and "'width': 0.5" in n for n in info.value.__notes__)
    assert not gmsh.isInitialized()


def stretch_build(lc, *, periodic=False):
    """A strip along x of ``width`` in a box of ``length`` (the cell's period)."""

    def build(width, length=2.0):
        gm = _cad.import_gmsh()
        occ = gm.model.occ
        box = occ.addBox(0, 0, 0, length, 2, 1)
        core = occ.addBox(0, 1 - width / 2, cc.Z0, length, width, cc.T0)
        occ.fragment([(3, box)], [(3, core)])
        occ.synchronize()
        volumes = [t for _, t in gm.model.getEntities(3)]
        mass = {t: occ.getMass(3, t) for t in volumes}
        small = min(mass, key=mass.get)
        gm.model.addPhysicalGroup(3, [small], name="core")
        gm.model.addPhysicalGroup(3, [t for t in volumes if t != small], name="clad")
        lows, highs = [], []
        for _, t in gm.model.getEntities(2):
            b = gm.model.getBoundingBox(2, t)
            if abs(b[0]) < cc.PAD and abs(b[3]) < cc.PAD:
                lows.append((t, b))
            if abs(b[0] - length) < cc.PAD and abs(b[3] - length) < cc.PAD:
                highs.append((t, b))
        gm.model.addPhysicalGroup(2, [t for t, _ in lows], name="xmin")
        gm.model.addPhysicalGroup(2, [t for t, _ in highs], name="xmax")
        if periodic:
            # each face of x = length is the image of the face of x = 0 at the
            # same (y, z); the translation takes the master to the slave
            def centre(b):
                return (round((b[1] + b[4]) / 2, 6), round((b[2] + b[5]) / 2, 6))

            master = {centre(b): t for t, b in lows}
            gm.model.mesh.setPeriodic(
                2,
                [t for t, _ in highs],
                [master[centre(b)] for _, b in highs],
                [1, 0, 0, length, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            )
        gm.option.setNumber("Mesh.MeshSizeMax", lc)
        gm.option.setNumber("Mesh.MeshSizeMin", lc / 4)

    return build


STRETCH = {"width": cc.W0, "length": 2.0}


def with_port(mesh):
    return gfem.Model(
        mesh,
        materials={"core": gfem.Material(eps_r=3.9), "clad": gfem.Material()},
        ports=[gfem.WavePort("p", "xmax")],
    )


def with_pair(mesh):
    return gfem.Model(
        mesh,
        materials={"core": gfem.Material(eps_r=3.9), "clad": gfem.Material()},
        boundaries=[gfem.Periodic(("xmin", "xmax"))],
    )


@pytest.mark.usefixtures("gmsh")
def test_a_parameter_that_moves_a_held_port_node_is_refused_with_its_holder():
    with pytest.raises(ValueError, match=r"Parameter 'width'.*port 'p' holds") as info:
        gfem.shape.cad_morph(stretch_build(0.3), STRETCH, with_port)
    message = str(info.value)
    assert "sliding_ports=" in message
    assert "(2, 0.75, 0.6) um" in message  # the worst node, in model units
    assert "(0, -0.5, 0) um per unit" in message


@pytest.mark.usefixtures("gmsh")
def test_a_sliding_port_keeps_its_plane_while_the_walls_that_meet_it_move():
    morph = gfem.shape.cad_morph(
        stretch_build(0.3), {"width": cc.W0}, with_port, sliding_ports=("p",)
    )
    mesh = morph.model.mesh
    face = np.unique(mesh.triangles[mesh.triangle_tags == mesh.region("xmax").tag])
    w = morph.velocities["width"]
    # The face's nodes slide in its plane: no x component. The plane x = 2
    # um is exact in the CAD, so the slide basis (orthonormal to the
    # face's normal e_x to 1e-16) leaves x at rounding of the velocity scale
    # 1e-6 m/um.
    assert np.abs(w[face, 0]).max() <= 1e-9 * UM
    assert np.abs(w[face, 1]).max() > 0.1 * UM  # the walls' edges do move
    moved = np.asarray(morph({"width": 0.55}).nodes)
    assert (
        np.abs(moved[face, 0] - mesh.imported_nodes[face, 0]).max() <= 1e-9 * 0.05 * UM
    )


@pytest.mark.usefixtures("gmsh")
def test_a_parameter_that_moves_a_sliding_ports_plane_is_refused():
    with pytest.raises(ValueError, match=r"Parameter 'length'.*sliding port 'p'"):
        gfem.shape.cad_morph(
            stretch_build(0.3), STRETCH, with_port, sliding_ports=("p",)
        )


@pytest.mark.usefixtures("gmsh")
def test_a_parameter_that_changes_the_period_is_refused_naming_the_pair():
    with pytest.raises(
        ValueError, match="Parameter 'length' breaks the periodicity"
    ) as info:
        gfem.shape.cad_morph(stretch_build(0.3, periodic=True), STRETCH, with_pair)
    assert "(0, 0.75, 0.6) and (2, 0.75, 0.6) um" in str(info.value)


@pytest.mark.usefixtures("gmsh")
def test_a_parameter_that_keeps_the_pairs_moves_each_twin_alike():
    morph = gfem.shape.cad_morph(
        stretch_build(0.3, periodic=True), {"width": cc.W0}, with_pair
    )
    mesh = morph.model.mesh
    ties = gfem._periodic.tie_classes(len(mesh.imported_nodes), morph.model._periodic)
    twins = np.flatnonzero(ties != np.arange(len(ties)))
    assert len(twins) > 50
    w = morph.velocities["width"]
    # twins take the representative's column of the extension: equal up to
    # the solve's round-off, a few ulps of the velocity scale (0.5e-6 m/um)
    np.testing.assert_allclose(w[twins], w[ties[twins]], rtol=0, atol=1e-12 * UM)
    # the pair keeps its translation: 2 um along x
    moved = np.asarray(morph({"width": 0.56}).nodes)
    shift = moved[twins] - moved[ties[twins]]
    np.testing.assert_allclose(
        shift,
        mesh.imported_nodes[twins] - mesh.imported_nodes[ties[twins]],
        rtol=0,
        atol=4 * EPS * np.abs(moved).max(),
    )


@pytest.mark.usefixtures("gmsh")
def test_a_topology_change_inside_the_step_is_refused_with_the_parameter_and_side():
    # 1.9 um is 0.1 um from the 2 um box wall, a mesh size away at 0.25 um;
    # the 0.25 um step makes the + build 2.15 um wide, wider than the box,
    # so the fragment gains faces and volumes.
    with pytest.raises(ValueError, match="'width' at \\+h") as info:
        gfem.shape.cad_morph(
            cc.strip_build(0.25),
            {"width": 1.9, "height": cc.T0},
            cc.make_model,
            step={"width": 0.25},
        )
    assert "topology" in str(info.value)
    assert "step=" in str(info.value)


@pytest.mark.usefixtures("gmsh")
def test_a_node_that_jumps_is_refused_with_the_parameter():
    inner = cc.strip_build(0.3)

    def build(scaled, height):
        # 5000 um per unit of `scaled`: a velocity of 2500, past the 1e3 steps
        # a node may move per step
        inner(5000 * scaled, height)

    with pytest.raises(ValueError, match=r"more than 1000 steps.*'scaled'"):
        gfem.shape.cad_morph(build, {"scaled": 1e-4, "height": cc.T0}, cc.make_model)


@pytest.mark.usefixtures("gmsh")
def test_a_parameter_that_moves_nothing_is_refused():
    inner = cc.strip_build(0.3)

    def build(width, height, grain):
        inner(width, height)
        _cad.import_gmsh().option.setNumber("Mesh.MeshSizeMax", 0.3 + 0.01 * grain)

    with pytest.raises(ValueError, match="Parameter 'grain' moves nothing"):
        gfem.shape.cad_morph(build, {**P0, "grain": 1.0}, cc.make_model)


def test_a_volume_in_no_group_is_refused_before_meshing(gmsh):
    def build(width, height):
        cc.strip_build(0.3)(width, height)
        occ = gmsh.model.occ
        occ.addBox(3, 3, 3, 1, 1, 1)
        occ.synchronize()

    with pytest.raises(
        ValueError, match=r"volume.*centred at \(3\.5, 3\.5, 3\.5\) \(um\)"
    ) as info:
        gfem.shape.cad_morph(build, P0, cc.make_model)
    assert "addPhysicalGroup(3" in str(info.value)
    assert not gmsh.isInitialized()


def test_a_model_with_no_volume_is_refused(gmsh):
    def build(width):
        gmsh.model.occ.addRectangle(0, 0, 0, width, 1)
        gmsh.model.occ.synchronize()

    with pytest.raises(ValueError, match="no 3D entity"):
        gfem.shape.cad_morph(build, {"width": 1.0}, cc.make_model)


# --- the gmsh session ----------------------------------------------------------


def test_an_own_session_is_quiet_reads_no_config_and_is_finalized(gmsh, monkeypatch):
    assert not gmsh.isInitialized()
    handler = signal.getsignal(signal.SIGINT)
    kwargs = []
    initialize = gmsh.initialize

    def spy(*args, **options):
        kwargs.append(options)
        return initialize(*args, **options)

    monkeypatch.setattr(gmsh, "initialize", spy)
    inside = []
    inner = cc.strip_build(0.3)

    def build(width, height):
        inside.append(
            (
                gmsh.option.getNumber("General.Terminal"),
                gmsh.model.getCurrent(),
                signal.getsignal(signal.SIGINT),
            )
        )
        inner(width, height)

    gfem.shape.cad_morph(build, P0, cc.make_model)
    assert kwargs == [{"readConfigFiles": False, "interruptible": False}]
    assert not gmsh.isInitialized()
    assert signal.getsignal(signal.SIGINT) is handler
    # three builds: p0, p0 + h, p0 - h of each of the two parameters
    assert len(inside) == 1 + 2 * 2
    assert all(
        t == 0 and m.startswith("gfem_cad_") and h is handler for t, m, h in inside
    )
    assert len({m for _, m, _ in inside}) == len(inside)


def test_a_failing_build_leaves_no_session_and_names_the_parameters(gmsh):
    def build(**_params):
        raise RuntimeError("bad drawing")

    with pytest.raises(RuntimeError, match="bad drawing") as info:
        gfem.shape.cad_morph(build, P0, cc.make_model)
    assert info.value.__notes__ == [
        "gfem.shape.cad_morph called build with {'width': 0.5, 'height': 0.3}"
    ]
    assert not gmsh.isInitialized()


def test_a_build_that_fails_at_a_step_names_that_rebuild(gmsh):
    inner = cc.strip_build(0.3)

    def build(width, height):
        if width != cc.W0:
            raise RuntimeError("only p0 works")
        inner(width, height)

    with pytest.raises(RuntimeError, match="only p0 works") as info:
        gfem.shape.cad_morph(build, P0, cc.make_model)
    assert f"'width': {cc.W0 + 2.0**-16}" in info.value.__notes__[0]
    assert not gmsh.isInitialized()


def test_a_users_session_is_left_as_found(gmsh):
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.add("mine")
        gmsh.model.occ.addSphere(0, 0, 0, 1)
        gmsh.model.occ.synchronize()
        before = (gmsh.model.list(), gmsh.model.getEntities())
        gfem.shape.cad_morph(cc.strip_build(0.3), P0, cc.make_model)
        assert gmsh.isInitialized()
        assert gmsh.model.getCurrent() == "mine"
        assert (gmsh.model.list(), gmsh.model.getEntities()) == before
        assert gmsh.option.getNumber("General.Terminal") == 1
        assert gmsh.option.getNumber("Mesh.MshFileVersion") == 2.2
        assert gmsh.option.getNumber("Mesh.Binary") == 0
        assert gmsh.option.getNumber("Mesh.SaveAll") == 1
        # a failing build and a refused one clean up as well
        with pytest.raises(ValueError, match="no 3D entity"):
            gfem.shape.cad_morph(
                lambda **_params: gmsh.model.occ.addRectangle(0, 0, 0, 1, 1),
                {"width": 1.0},
                cc.make_model,
            )
        assert gmsh.model.getCurrent() == "mine"
        assert gmsh.model.list() == before[0]
    finally:
        gmsh.finalize()


@pytest.mark.usefixtures("gmsh")
def test_the_mesh_is_written_as_binary_msh_41_in_a_temporary_directory(
    monkeypatch, tmp_path
):
    seen = []
    read = _cad._read_gmsh

    def spy(path, length_unit):
        seen.append((Path(path), Path(path).read_bytes()[:20]))
        return read(path, length_unit)

    monkeypatch.setattr(_cad, "_read_gmsh", spy)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    morph = gfem.shape.cad_morph(cc.strip_build(0.3), P0, cc.make_model)
    ((path, head),) = seen
    assert head == b"$MeshFormat\n4.1 1 8\n"  # version 4.1, binary, 8-byte sizes
    assert path.parent.parent == tmp_path  # TMPDIR is honoured
    assert not path.exists()  # the directory is gone
    assert morph.model.mesh._topology.source is None


# --- the node map and the extension --------------------------------------------


def test_the_node_map_is_tag_minus_one_when_the_coordinates_agree():
    rng = np.random.default_rng(1)
    x = rng.random((20, 3))
    chosen = np.array([3, 4, 9, 19])
    index = _cad.node_map(x * UM, UM, chosen + 1, x[chosen], 1.0)
    assert np.array_equal(index, chosen)


def test_the_node_map_falls_back_to_exact_coordinates_on_a_permuted_mesh():
    rng = np.random.default_rng(2)
    x = rng.random((20, 3))
    order = rng.permutation(20)  # the mesh lists the nodes in another order
    imported = x[order] * UM
    chosen = np.array([3, 4, 9, 19])
    index = _cad.node_map(imported, UM, chosen + 1, x[chosen], 1.0)
    assert np.array_equal(imported[index], x[chosen] * UM)
    assert np.array_equal(order[index], chosen)


def test_the_node_map_refuses_a_node_that_is_not_in_the_mesh_or_is_ambiguous():
    rng = np.random.default_rng(3)
    x = rng.random((20, 3))
    missing = x[[3]] + 1e-3
    with pytest.raises(ValueError, match="do not map onto the mesh"):
        _cad.node_map(x[::-1] * UM, UM, np.array([4]), missing, 1.0)
    twin = x.copy()
    twin[5] = twin[7]
    with pytest.raises(ValueError, match="share coordinates"):
        _cad.node_map(twin[::-1] * UM, UM, np.array([8]), twin[[7]], 1.0)


def test_the_vector_extension_solves_for_the_free_directions_only():
    rng = np.random.default_rng(4)
    n, k, p = 30, 12, 3
    a = rng.standard_normal((n, n))
    stiffness = sp.csr_array(a @ a.T + n * np.eye(n))
    free = sp.csr_array(rng.standard_normal((n, k)))
    velocity = rng.standard_normal((n, p))
    w = _morph.vector_extension(stiffness, free, velocity)
    # W - V lies in the span of B, and the energy is stationary over it:
    # B^T K W = 0. The reduced matrix has a condition number below 1e2 for
    # these sizes, so the residual is at round-off times that, 1e-12 of the
    # scale of K W (about 1e2).
    resid = free.T @ (stiffness @ w)
    assert np.abs(resid).max() <= 1e-12 * np.abs(stiffness @ w).max() * 1e2
    coefficients, *_ = np.linalg.lstsq(free.toarray(), w - velocity, rcond=None)
    np.testing.assert_allclose(free @ coefficients, w - velocity, rtol=0, atol=1e-10)
