"""Bent port modes: the `Bend`, its result, refusals, warnings and p = 3 references."""

from __future__ import annotations

import math
import re
import warnings

import bend_cases as bc
import jax
import numpy as np
import pytest
from line_port_cases import FREQUENCY, guide, write_guide

import gfem

C0 = bc.C0
# $10\pi/\ln 10$: dB per quarter turn per unit $R\,\mathrm{Re}\,\gamma$
DB90 = 10 * math.pi / math.log(10)


# --- the value ---------------------------------------------------------------------


def test_a_bend_has_the_radius_as_its_only_leaf():
    bend = gfem.Bend(0.02, outward="+x", at=(1e-3, 0.0, 0.0))
    assert jax.tree.leaves(bend) == [0.02]
    scaled = jax.tree.map(lambda r: 2 * r, bend)
    assert (scaled.radius, scaled.outward, scaled.at) == (0.04, bend.outward, bend.at)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("+x", (1.0, 0.0, 0.0)),
        ("-y", (0.0, -1.0, 0.0)),
        ("z", (0.0, 0.0, 1.0)),
        ((3.0, 0.0, 4.0), (0.6, 0.0, 0.8)),
    ],
)
def test_outward_is_an_axis_or_a_vector_stored_as_a_unit_vector(value, expected):
    bend = gfem.Bend(1.0, outward=value, at=(0, 0, 0))
    assert bend.outward == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("radius", [0.0, -1.0, math.inf, math.nan, "r", None])
def test_a_radius_must_be_positive_and_finite(radius):
    with pytest.raises(ValueError, match=r"Bend\.radius must be positive and finite"):
        gfem.Bend(radius, outward="+x", at=(0, 0, 0))


def test_a_bad_outward_or_point_is_refused():
    with pytest.raises(ValueError, match="not an axis"):
        gfem.Bend(1.0, outward="up", at=(0, 0, 0))
    with pytest.raises(ValueError, match="non-zero 3-vector"):
        gfem.Bend(1.0, outward=(0, 0, 0), at=(0, 0, 0))
    with pytest.raises(ValueError, match="finite 3-vector"):
        gfem.Bend(1.0, outward="x", at=(0, 0))
    with pytest.raises(ValueError, match="finite 3-vector"):
        gfem.Bend(1.0, outward="x", at=(0, math.nan, 0))


def test_a_traced_direction_or_point_is_not_differentiable_but_the_radius_is():
    def build_outward(x):
        return gfem.Bend(1.0, outward=(1.0, x, 0.0), at=(0, 0, 0)).radius

    def build_at(x):
        return gfem.Bend(1.0, outward="x", at=(x, 0.0, 0.0)).radius

    with pytest.raises(gfem.NonDifferentiableError):
        jax.grad(build_outward)(0.1)
    with pytest.raises(gfem.NonDifferentiableError):
        jax.grad(build_at)(0.1)
    # the radius may be traced (its leaf is the one input a bend differentiates)
    grad = jax.grad(lambda r: gfem.Bend(r, outward="x", at=(0, 0, 0)).radius)(2.0)
    assert float(grad) == 1.0


def test_a_port_mode_takes_a_bend_only_as_a_bend():
    with pytest.raises(TypeError, match=r"gfem\.Bend"):
        gfem.PortMode("p1", 10e9, bend=0.02)
    assert gfem.PortMode("p1", 10e9).bend is None


def test_the_repr_names_the_radius():
    bend = gfem.Bend(0.02, outward="+x", at=(0.0, 0.0, 0.0))
    assert "radius=0.02 m" in repr(bend)


def test_a_port_mode_with_and_without_a_bend_round_trips_a_job(tmp_path):
    mesh = gfem.Mesh.from_gmsh(bc.wr90_mesh(tmp_path / "w.msh"), length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "p1")],
    )
    options = gfem.SolverOptions(order=1)
    bend = bc.wr90_bend(20.0)
    for problem in (gfem.PortMode("p1", 10e9), gfem.PortMode("p1", 10e9, bend=bend)):
        text = gfem.Job(model, problem, options).to_json(tmp_path / "job.json")
        job = gfem.Job.from_json(tmp_path / "job.json")
        assert job.problem == problem
        assert job.problem.bend == problem.bend
        # a straight port mode is written as before: no bend key
        assert ('"bend"' in (tmp_path / "job.json").read_text()) == (
            problem.bend is not None
        )
        del text


# --- meshes and models, shared by the solves below ---------------------------------


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    d = tmp_path_factory.mktemp("bend")
    return d


@pytest.fixture(scope="module")
def wr90(meshes):
    return bc.wr90_model(bc.wr90_mesh(meshes / "wr90.msh"))


@pytest.fixture(scope="module")
def h20(wr90):
    """WR-90 H-plane at R = 20 mm, p = 3."""
    return bc.solve_mode(wr90, bc.WR90_F, bc.wr90_bend(20.0))


# --- the result --------------------------------------------------------------------


def test_the_bend_loss_is_the_loss_per_quarter_turn(h20):
    gamma = np.asarray(h20.gamma)
    # the same formula evaluated by the test, so only round-off apart (rtol 1e-13 is
    # a few ulp of the product of three factors)
    np.testing.assert_allclose(h20.bend_loss, DB90 * 20e-3 * gamma.real, rtol=1e-13)
    assert h20.bend == bc.wr90_bend(20.0)
    assert "bend radius 0.02 m" in repr(h20)


def test_a_straight_result_has_no_bend_loss(wr90):
    straight = bc.solve_mode(wr90, bc.WR90_F)
    assert straight.bend is None
    assert straight.bend_loss is None
    assert "bend" not in repr(straight)


def test_the_bend_loss_enters_the_dataset(h20):
    xr = pytest.importorskip("xarray")
    ds = h20.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert ds["bend_loss"].attrs["units"] == "dB"
    assert ds.attrs["bend_radius"] == 0.02
    np.testing.assert_array_equal(ds["bend_loss"].values, np.asarray(h20.bend_loss))


def test_the_bend_round_trips_through_a_result_pytree(h20):
    leaves, tree = jax.tree.flatten(h20)
    back = jax.tree.unflatten(tree, leaves)
    assert back.bend == h20.bend


def test_the_straight_limit_of_a_huge_radius(wr90):
    straight = bc.solve_mode(wr90, bc.WR90_F)
    bent = bc.solve_mode(wr90, bc.WR90_F, bc.wr90_bend(1e6 * bc.WR90_A))
    ratio = np.asarray(bent.gamma)[0] / np.asarray(straight.gamma)[0]
    # physical: (a/R)^2 = 1e-12; mesh asymmetry of the TE10 mode (the bend leans
    # on one side) <= ~1e-11; round-off of two eigensolves <= ~1e-11; the gate is
    # a margin of 100 over their sum
    assert abs(ratio - 1) <= 1e-9


# --- exact references at p = 3 -------------------------------------------------------


def test_h_plane_bend_matches_the_bessel_root(h20):
    # measured 7.2e-7 at p = 3 (6x2 mesh), gate 2e-6 (2.8x)
    nu = bc.nu_of(h20, 20e-3)
    assert abs(nu / bc.WR90[20.0][0] - 1) <= 2e-6
    assert abs(nu.imag) <= 1e-9 * abs(nu)  # closed and lossless: real


def test_e_in_plane_bend_matches_the_bessel_root(meshes):
    # it finds no mode without the h_max bound on the propagation constant;
    # measured 1.95e-8 at p = 3 (8x1 mesh), gate 5e-8 (2.6x)
    model = bc.line_model(bc.line_mesh(meshes / "line.msh"))
    result = bc.solve_mode(model, bc.LINE_F, bc.line_bend(10.0))
    assert abs(bc.nu_of(result, 10e-3) / bc.LINE[10.0] - 1) <= 5e-8


@pytest.mark.parametrize("pol", ["Ey", "Hy"])
def test_a_leaky_slab_bend_matches_the_wall_root(meshes, pol):
    # measured at p = 3 on h = 0.1 um: |nu/ref - 1| = 3.7e-9 (Ey), 1.2e-9 (Hy), so
    # the 1e-8 gate is 2.7x; Im nu (5.3e-3 for Ey) carries the same absolute
    # error scale (~1e-7 at most, 3.7e-9 |nu| = 1.4e-7), gate 2e-7
    model = bc.slab_model(bc.slab_mesh(meshes / f"slab_{pol}.msh", pol))
    result = bc.solve_mode(model, bc.SLAB_F, bc.slab_bend(5.0))
    nu = bc.nu_of(result, 5e-6)
    ref = bc.SLAB_NU[(5, pol)]
    assert abs(nu - ref) <= 1e-8 * abs(ref)
    assert abs(nu.imag - ref.imag) <= 2e-7
    np.testing.assert_allclose(result.bend_loss, -DB90 * ref.imag, rtol=1e-4)


# --- refusals ------------------------------------------------------------------------


def _wr90_with(meshes, boundaries, materials=None):
    mesh = gfem.Mesh.from_gmsh(bc.wr90_mesh(meshes / "wr90.msh"), length_unit="mm")
    return gfem.Model(
        mesh,
        materials=materials or {"air": gfem.Material()},
        boundaries=boundaries,
        ports=[gfem.WavePort("p1", "p1")],
    )


def test_a_line_port_cannot_be_bent(meshes):
    mesh = gfem.Mesh.from_gmsh(write_guide(meshes / "guide.msh"), length_unit="mm")
    model = guide(mesh, 4.0, 4.0)
    bend = gfem.Bend(1e-2, outward="+y", at=(0.0, 1.5e-3, 0.0))
    with pytest.raises(NotImplementedError, match="3D port mode problem"):
        gfem.solve(model, gfem.PortMode("p1", FREQUENCY, polarization="Ez", bend=bend))


def test_a_bend_that_reaches_its_axis_names_the_minimum_h(wr90):
    # the guide extends 11.43 mm inward of the centre line: R = 10 mm puts the
    # axis inside it
    with pytest.raises(ValueError, match=r"minimum of h = 1 \+ xi/R is -0\.143"):
        bc.solve_mode(wr90, bc.WR90_F, bc.wr90_bend(10.0))


def test_outward_and_the_point_must_lie_in_the_port_plane(wr90):
    with pytest.raises(ValueError, match="not in the plane of the port"):
        bc.solve_mode(wr90, bc.WR90_F, gfem.Bend(0.02, outward="z", at=(0.0, 0, 0)))
    with pytest.raises(ValueError, match="off the plane of the port"):
        bc.solve_mode(
            wr90, bc.WR90_F, gfem.Bend(0.02, outward="x", at=(0.0114, 0.0, 1e-3))
        )


def _uniaxial(axis):
    eps = [2.0, 2.0, 2.0]
    eps[axis] = 3.0
    return {"air": gfem.Material(eps_r=tuple(eps))}


def test_a_material_must_be_invariant_about_the_bend_axis(meshes):
    boundaries = [gfem.PEC("walls")]
    # optic axis along y, out of the bend plane: invariant (z-cut)
    ok = _wr90_with(meshes, boundaries, _uniaxial(1))
    result = bc.solve_mode(ok, bc.WR90_F, bc.wr90_bend(60.0))
    assert np.isfinite(np.asarray(result.gamma)).all()
    # optic axis along x, the bend's outward direction: not a body of revolution
    bad = _wr90_with(meshes, boundaries, _uniaxial(0))
    with pytest.raises(
        ValueError, match=r"region\(s\) \['air'\].*eps_r.*not invariant"
    ):
        bc.solve_mode(bad, bc.WR90_F, bc.wr90_bend(60.0))
    # a permeability with its axis along x (nu is read across the bend plane)
    mu = {"air": gfem.Material(mu_r=(3.0, 2.0, 2.0))}
    with pytest.raises(ValueError, match=r"mu_r.*not invariant"):
        bc.solve_mode(_wr90_with(meshes, boundaries, mu), bc.WR90_F, bc.wr90_bend(60.0))


@pytest.mark.parametrize(
    "boundary",
    [
        gfem.Absorbing("walls"),
        gfem.Impedance("walls", resistance=1.0),
        gfem.Conductor("walls", conductivity=5.8e7),
        gfem.Conductor("walls", conductivity=5.8e7, thickness=1e-6),
    ],
    ids=["absorbing", "impedance", "conductor", "sheet"],
)
def test_edge_terms_without_a_bend_metric_are_refused(meshes, boundary):
    model = _wr90_with(meshes, [boundary])
    with pytest.raises(NotImplementedError, match="Absorbing, Impedance or Conductor"):
        bc.solve_mode(model, bc.WR90_F, bc.wr90_bend(60.0))


def test_pec_and_natural_edges_are_solved(meshes):
    # PEC all round is the reference case (the TE10-like mode of the exact
    # nu); with every edge natural (PMC) the section is another valid bent one
    mesh = gfem.Mesh.from_gmsh(bc.wr90_mesh(meshes / "wr90.msh"), length_unit="mm")
    pec = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "p1")],
    )
    result = bc.solve_mode(pec, bc.WR90_F, bc.wr90_bend(60.0))
    # measured 1.6e-7 at p = 3 on the 6x2 mesh (p = 5 reaches 1e-9)
    assert abs(bc.nu_of(result, 60e-3) / bc.WR90[60.0][0] - 1) <= 1e-6
    natural = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[],
        ports=[gfem.WavePort("p1", "p1")],
    )
    result = bc.solve_mode(natural, bc.WR90_F, bc.wr90_bend(60.0))
    assert np.isfinite(np.asarray(result.gamma)).all()


# --- the unabsorbed-bend warning -----------------------------------------------------


def _solve_slab(meshes, name, pol, radius, *, bend=None, **mesh_options):
    model = bc.slab_model(bc.slab_mesh(meshes / f"{name}.msh", pol, **mesh_options))
    problem = gfem.PortMode("p1", bc.SLAB_F, bend=bend or bc.slab_bend(radius))
    return gfem.solve(model, problem, options=gfem.SolverOptions(order=2, threads=1))


@pytest.mark.parametrize(
    "mesh_options",
    [
        {"d": 0.0, "outer_wall": False},
        {"d": 0.0, "outer_wall": False, "inner_d": 1.0},
    ],
    ids=["no layer", "inner layer only"],
)
def test_a_bend_with_a_free_outer_edge_and_no_outer_layer_warns(meshes, mesh_options):
    with pytest.warns(gfem.UnabsorbedBendWarning, match="outer side"):
        _solve_slab(meshes, "free", "Ey", 8.0, **mesh_options)


@pytest.mark.parametrize(
    "mesh_options",
    [{"d": 0.0}, {}, {"outer_wall": False}],
    ids=["PEC outer wall", "outer layer", "outer layer, free edge"],
)
def test_a_bend_with_an_absorbed_or_conducting_outer_side_is_silent(
    meshes, mesh_options
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.UnabsorbedBendWarning)
        _solve_slab(meshes, "quiet", "Ey", 8.0, **mesh_options)


# --- the caustic ---------------------------------------------------------------------


def _caustic_warnings(meshes, pol, radius, xp):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _solve_slab(meshes, f"caustic_{pol}_{xp}", pol, radius, xp=xp, d=1.0)
    return [
        str(w.message)
        for w in record
        if issubclass(w.category, gfem.LayerClearanceWarning)
    ]


def test_a_layer_before_the_caustic_warns_with_its_position(meshes):
    # x_c = R (Re n_eff / n_cl - 1) = 2.20 um for the Ey mode at R = 8 um
    (message,) = _caustic_warnings(meshes, "Ey", 8.0, 1.0)
    match = re.search(r"x_c = ([0-9.e+-]+) m", message)
    assert match is not None
    assert float(match[1]) == pytest.approx(2.20e-6, abs=0.01e-6)


def test_a_layer_beyond_the_caustic_or_a_tight_bend_is_silent(meshes):
    assert _caustic_warnings(meshes, "Ey", 8.0, 2.5) == []
    # R = 3 um Hy: x_c = 1.40 um (Ey) and less for Hy, below the layer at 2.5 um
    assert _caustic_warnings(meshes, "Hy", 3.0, 2.5) == []


# --- isolation -----------------------------------------------------------------------


def test_a_straight_solve_after_a_bent_one_is_the_fresh_straight_solve(meshes):
    first = bc.wr90_model(bc.wr90_mesh(meshes / "iso.msh"))
    fresh = bc.wr90_model(bc.wr90_mesh(meshes / "iso2.msh"))
    bc.solve_mode(first, bc.WR90_F, bc.wr90_bend(20.0))
    after = bc.solve_mode(first, bc.WR90_F)
    alone = bc.solve_mode(fresh, bc.WR90_F)
    # the same operations on the same mesh: only the thread-independent
    # eigensolver's round-off (a few 1e-16) apart, far below the gate
    np.testing.assert_allclose(after.gamma, alone.gamma, rtol=1e-12)
    np.testing.assert_allclose(after.n_g, alone.n_g, rtol=1e-12)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_bent_mode_on_a_shared_surface_is_the_single_port_solve(meshes):
    mesh = gfem.Mesh.from_gmsh(bc.wr90_mesh(meshes / "shared.msh"), length_unit="mm")

    def model(*ports):
        return gfem.Model(
            mesh,
            materials={"air": gfem.Material()},
            boundaries=[gfem.PEC("walls")],
            ports=list(ports),
        )

    p1 = gfem.WavePort("p1", "p1", mode=1)
    p2 = gfem.WavePort("p2", "p1", mode=2)
    bend = bc.wr90_bend(30.0)
    f = 16e9  # TE20 propagates (cutoff 13.1 GHz)
    shared = bc.solve_mode(model(p1, p2), f, bend)
    single = bc.solve_mode(model(p1), f, bend)
    # same section, same sweep: the eigensolver's round-off apart
    np.testing.assert_allclose(shared.gamma, single.gamma, rtol=1e-12)
    np.testing.assert_allclose(shared.Z_PV, single.Z_PV, rtol=1e-12)


# --- jit -----------------------------------------------------------------------------


def test_a_bent_port_mode_runs_under_jit(wr90, h20):
    problem = gfem.PortMode("p1", (bc.WR90_F,), bend=bc.wr90_bend(20.0))
    options = gfem.SolverOptions(order=3, threads=1)

    @jax.jit
    def solved(scale):
        # a traced value makes the solve stage; its value leaves the operator alone
        result = gfem.solve(wr90, problem, options=options)
        return result.gamma * scale, result.bend_loss * scale

    gamma, loss = solved(1.0)
    # both run the same host solve; only thread scheduling in the eigensolver
    # differs, round-off (1e-13) well under the gate
    np.testing.assert_allclose(gamma, h20.gamma, rtol=1e-10)
    np.testing.assert_allclose(loss, h20.bend_loss, rtol=1e-10)
