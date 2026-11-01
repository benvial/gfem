"""`gfem.Periodic` on a model: node pairing, build checks, wave vectors, job files.

The meshes are the periodic cells of tests/periodic_cases.py (mm), or
`structured_meshes.periodic_cell` variants that break one condition of a
periodic pair each.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
import structured_meshes as sm

import gfem

MM = 1e-3


@pytest.fixture(scope="module")
def c2_mesh(tmp_path_factory):
    return pc.c2_mesh(tmp_path_factory.mktemp("c2"))


@pytest.fixture(scope="module")
def c4_mesh(tmp_path_factory):
    return pc.c4_mesh(tmp_path_factory.mktemp("c4"))


def _cell(tmp_path, cell, name="cell"):
    path = sm.write_msh22(tmp_path / f"{name}.msh", cell)
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _box(size=(10.0, 7.0, 1.0), cells=(4, 3, 1), **kwargs):
    edges = tuple(np.linspace(0, s, n + 1) for s, n in zip(size, cells, strict=True))
    return sm.periodic_cell(edges, **kwargs)


def _model(mesh, boundaries, **kwargs):
    materials = kwargs.pop("materials", {"cell": gfem.Material()})
    return gfem.Model(mesh, materials=materials, boundaries=boundaries, **kwargs)


def _x_pair():
    return gfem.Periodic(("x_min", "x_max"))


# --- pairing on the C2 and C4 cells ------------------------------------------------


def test_two_pairs_measure_their_translations_in_metres(c2_mesh):
    model = pc.c2(c2_mesh)
    np.testing.assert_allclose(
        model.periodic_translations,
        [(pc.C2_AX * MM, 0, 0), (0, pc.C2_AY * MM, 0)],
        rtol=0,
        atol=1e-15,
    )
    assert all(isinstance(v, float) for a in model.periodic_translations for v in a)


def test_three_pairs_follow_boundary_order(c4_mesh):
    model = gfem.Model(
        c4_mesh,
        materials={"cell": gfem.Material()},
        boundaries=[
            gfem.Periodic(("z_max", "z_min")),
            gfem.Periodic(("x_min", "x_max")),
            gfem.Periodic(("y_min", "y_max")),
        ],
    )
    L = pc.C4_L * MM
    np.testing.assert_allclose(
        model.periodic_translations,
        [(0, 0, -L), (L, 0, 0), (0, L, 0)],
        rtol=0,
        atol=1e-15,
    )


def test_a_model_without_pairs_has_no_translations(c4_mesh):
    model = gfem.Model(c4_mesh, materials={"cell": gfem.Material()})
    assert model.periodic_translations == ()


def test_paired_surfaces_are_not_default_pmc(c2_mesh):
    model = pc.c2(c2_mesh)
    assert model.default_pmc == ()
    one = _model(c2_mesh, [_x_pair()], materials={"cell": gfem.Material(eps_r=2)})
    assert set(one.default_pmc) == {"y_min", "y_max", "z_min", "z_max"}


def test_a_surface_list_is_read_as_a_tuple():
    assert gfem.Periodic(surfaces=["x_min", "x_max"]).surfaces == ("x_min", "x_max")


# --- Periodic values ----------------------------------------------------------------


@pytest.mark.parametrize("surfaces", ["x_min", ("x_min",), ("x_min", 1), None])
def test_periodic_takes_two_surface_names(surfaces):
    with pytest.raises(TypeError, match="two surface names"):
        gfem.Periodic(surfaces)


def test_periodic_surfaces_must_differ():
    with pytest.raises(ValueError, match="'x_min' twice"):
        gfem.Periodic(("x_min", "x_min"))


# --- model checks (ticket 03) ------------------------------------------------------


def _split_x(c):
    """Cut the x faces at y = 3.5 into "<face>_a" (below) and "<face>_b"."""
    if abs(c[0]) < 1e-9 or abs(c[0] - 10) < 1e-9:
        side = "x_min" if abs(c[0]) < 1e-9 else "x_max"
        return f"{side}_{'a' if c[1] < 3.5 else 'b'}"
    return ""


def test_a_paired_surface_must_be_exterior(tmp_path):
    mesh = pc.c1_mesh(tmp_path)
    with pytest.raises(ValueError, match="Periodic on 'sheet1': this surface is int"):
        _model(mesh, [gfem.Periodic(("sheet1", "sheet2"))])


def test_a_cell_has_at_most_three_pairs(tmp_path):
    mesh = _cell(tmp_path, _box(cells=(4, 2, 1), surface=_split_x))
    pairs = [
        gfem.Periodic(("x_min_a", "x_max_a")),
        gfem.Periodic(("x_min_b", "x_max_b")),
        gfem.Periodic(("y_min", "y_max")),
        gfem.Periodic(("z_min", "z_max")),
    ]
    with pytest.raises(ValueError, match="at most three periodic pairs, got 4"):
        _model(mesh, pairs)


def test_parallel_translations_are_refused(tmp_path):
    mesh = _cell(tmp_path, _box(cells=(4, 2, 1), surface=_split_x))
    pairs = [
        gfem.Periodic(("x_min_a", "x_max_a")),
        gfem.Periodic(("x_min_b", "x_max_b")),
    ]
    with pytest.raises(ValueError, match="parallel directions; a cell has one pair"):
        _model(mesh, pairs)


@pytest.mark.parametrize(
    "other",
    [gfem.PEC("x_max"), gfem.Absorbing("x_min"), gfem.Periodic(("x_max", "y_min"))],
)
def test_a_paired_surface_takes_no_other_condition(c4_mesh, other):
    with pytest.raises(ValueError, match="more than one electromagnetic boundary"):
        _model(c4_mesh, [_x_pair(), other])


def test_a_paired_surface_cannot_be_a_port(tmp_path):
    mesh = pc.c1_mesh(tmp_path)
    port = gfem.LumpedPort("p1", "x_max", "+z")
    with pytest.raises(ValueError, match="more than one electromagnetic boundary"):
        _model(mesh, [_x_pair()], ports=[port])


# --- mesh checks (ticket 04) -------------------------------------------------------


def test_node_counts_must_match(tmp_path):
    def cap(c):
        return "cap" if abs(c[0] - 10) < 1e-9 and c[1] > 5 else ""

    mesh = _cell(tmp_path, _box(surface=cap))
    with pytest.raises(
        ValueError,
        match=r"Periodic\(\('x_min', 'x_max'\)\): 'x_min' has 8 nodes and 'x_max' "
        r"has 6, so their meshes cannot match under a translation\. .*setPeriodic",
    ):
        _model(mesh, [_x_pair()])


def test_a_rotated_pair_is_not_a_translation(c4_mesh):
    with pytest.raises(
        ValueError,
        match=r"the surfaces are not parallel \(angle 90\.0°\); gfem pairs "
        "surfaces by a translation only",
    ):
        _model(c4_mesh, [gfem.Periodic(("x_min", "y_min"))])


def test_nodes_must_match_under_the_translation(tmp_path):
    nodes, *rest = _box()
    nodes = nodes.copy()
    on = np.abs(nodes[:, 0] - 10) < 1e-9
    nodes[on, 1] += 0.5 * np.sin(np.pi * nodes[on, 1] / 7)
    mesh = _cell(tmp_path, (nodes, *rest))
    with pytest.raises(
        ValueError,
        match=r"no translation maps 'x_min' onto 'x_max': with a = \(.*\) m, 8 "
        r"nodes of 'x_max' are farther than .* m from any node of 'x_min' "
        r"\(worst .* m at \(.*\) m\)\. .*setPeriodic",
    ):
        _model(mesh, [_x_pair()])


def _five_tet_row(n):
    """``n`` unit hexahedra along x, each cut into five tetrahedra.

    The cuts alternate from one hexahedron to the next (conforming), so with
    an odd ``n`` the x faces are triangulated along different diagonals.
    """
    grid = np.array([(x, y, z) for x in range(n + 1) for y in (0, 1) for z in (0, 1)])
    node = {tuple(p): i for i, p in enumerate(grid.tolist())}
    tets = []
    for i in range(n):
        even = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        centre = even if i % 2 == 0 else [(1 - a, b, c) for a, b, c in even]
        corners = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        tips = [c for c in corners if c not in centre]
        cells = [centre] + [
            [t]
            + [
                c
                for c in centre
                if sum(abs(u - v) for u, v in zip(t, c, strict=True)) == 1
            ]
            for t in tips
        ]
        tets += [[node[(i + a, b, c)] for a, b, c in tet] for tet in cells]
    tets = np.array(tets)
    p = grid[tets].astype(float)
    det = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    tets[det < 0] = tets[det < 0][:, [0, 2, 1, 3]]
    faces = tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3)
    uniq, first, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    tris, names = [], []
    for f in np.flatnonzero(counts == 1):
        c = grid[uniq[f]].mean(axis=0)
        names.append("x_min" if c[0] == 0 else "x_max" if c[0] == n else "wall")
        tris.append(faces[first[f]])
    return grid.astype(float), tets, ["cell"] * len(tets), np.array(tris), names


def test_triangulations_must_match(tmp_path):
    mesh = _cell(tmp_path, _five_tet_row(3))
    with pytest.raises(
        ValueError,
        match=r"the nodes match under a = \(0\.003, 0, 0\) m but 2 triangles of "
        "'x_max' have no translated twin on 'x_min'",
    ):
        _model(mesh, [_x_pair()])


def test_no_element_touches_both_surfaces(tmp_path):
    mesh = _cell(tmp_path, _box(cells=(1, 3, 1)))
    with pytest.raises(
        ValueError,
        match=r"18 elements touch both 'x_min' and 'x_max'; a periodic cell needs "
        "at least two elements across each period",
    ):
        _model(mesh, [_x_pair()])


def test_both_sides_border_the_same_solve_domain(tmp_path):
    def volume(c):
        return "metal" if c[0] > 7.5 and c[1] > 14 / 3 else "cell"

    mesh = _cell(tmp_path, _box(volume=volume))
    with pytest.raises(
        ValueError,
        match=r"2 faces of 'x_min' border the solve domain where their twins on "
        r"'x_max' border excluded volume 'metal' \(or the reverse\)",
    ):
        _model(mesh, [_x_pair()], excluded=["metal"])
    # excluding the metal on both sides is a periodic cell
    both = _cell(
        tmp_path,
        _box(
            volume=lambda c: (
                "metal" if (c[0] > 7.5 or c[0] < 2.5) and c[1] > 14 / 3 else "cell"
            )
        ),
        name="both",
    )
    _model(both, [_x_pair()], excluded=["metal"])


def _strip(c):
    return "strip" if abs(c[2]) < 1e-9 and c[0] > 5 else ""


def test_pec_edges_on_a_pair_must_be_periodic(tmp_path):
    mesh = _cell(tmp_path, _box(surface=_strip))
    with pytest.raises(
        ValueError,
        match=r"PEC 'strip' meets 'x_max' along 3 edges that have no PEC twin on "
        "'x_min'; NGSolve would impose PEC on both sides",
    ):
        _model(mesh, [_x_pair(), gfem.PEC("strip")])
    _model(mesh, [_x_pair(), gfem.PEC("strip"), gfem.PEC("z_min")])
    _model(mesh, [_x_pair(), gfem.Conductor("strip", 5.8e7)])


def _moved(mesh, displacement):
    return jax.tree_util.tree_map(lambda x: x + displacement, mesh)


def test_moved_nodes_must_keep_every_pair(c2_mesh):
    model = pc.c2(c2_mesh)
    shift = np.zeros_like(c2_mesh.imported_nodes)
    on = np.abs(c2_mesh.imported_nodes[:, 0] - pc.C2_AX * MM) < 1e-12
    shift[on, 1] = 1e-6
    with pytest.raises(
        ValueError,
        match=r"The moved mesh breaks Periodic\(\('x_min', 'x_max'\)\): 18 node pairs "
        r"are no longer related by the translation a = \(0\.01, 0, 0\) m "
        r"\(worst 1e-06 m at .*\)\. Move paired nodes by the same displacement: "
        "build the shape with gfem.boundary_morph",
    ):
        model.replace(mesh=_moved(c2_mesh, shift))


def test_tied_and_rigid_moves_keep_the_pairs(c2_mesh):
    model = pc.c2(c2_mesh)
    x = c2_mesh.imported_nodes
    # a periodic displacement: the same at x and x + a
    tied = np.zeros_like(x)
    tied[:, 2] = 1e-5 * np.sin(2 * np.pi * x[:, 0] / (pc.C2_AX * MM)) * x[:, 2]
    moved = model.replace(mesh=_moved(c2_mesh, tied))
    assert moved.periodic_translations == model.periodic_translations
    rigid = model.replace(mesh=_moved(c2_mesh, np.array([1e-3, -2e-3, 5e-4])))
    assert rigid.periodic_translations == model.periodic_translations


def test_traced_nodes_are_not_checked(c2_mesh):
    model = pc.c2(c2_mesh)

    def objective(nodes):
        mesh = jax.tree_util.tree_map(lambda _: nodes, c2_mesh)
        return jnp.sum(model.replace(mesh=mesh).mesh.nodes)

    grad = jax.grad(objective)(jnp.asarray(c2_mesh.nodes))
    np.testing.assert_array_equal(grad, 1.0)


def test_the_pairs_survive_flattening(c2_mesh):
    model = pc.c2(c2_mesh)
    leaves, tree = jax.tree_util.tree_flatten(model)
    back = jax.tree_util.tree_unflatten(tree, leaves)
    assert back.periodic_translations == model.periodic_translations


# --- the Bloch wave vector on problems --------------------------------------------

PROBLEMS = {
    "Driven": lambda **kw: gfem.Driven(5e9, **kw),
    "Eigenmode": lambda **kw: gfem.Eigenmode(5e9, **kw),
}


@pytest.mark.parametrize("problem", list(PROBLEMS))
def test_wave_vector_is_three_floats_in_rad_per_metre(problem):
    build = PROBLEMS[problem]
    assert build().wave_vector is None
    k = build(wave_vector=[1, -2.5, np.float32(3)]).wave_vector
    assert k == (1.0, -2.5, 3.0)
    assert all(type(v) is float for v in k)
    assert build(wave_vector=jnp.zeros(3)).wave_vector == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("value", [(1.0, 2.0), (1.0, np.nan, 0.0), "x", 5.0])
def test_wave_vector_must_be_a_finite_three_vector(problem, value):
    with pytest.raises(ValueError, match=r"wave_vector must be a 3-vector in rad/m"):
        PROBLEMS[problem](wave_vector=value)


@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("transform", [jax.jit, jax.grad])
def test_a_traced_wave_vector_is_not_differentiable(problem, transform):
    def objective(k):
        PROBLEMS[problem](wave_vector=(k, 0.0, 0.0))
        return k

    with pytest.raises(
        gfem.NonDifferentiableError, match=f"{problem}.wave_vector must be fixed"
    ):
        transform(objective)(1.0)


def test_the_bloch_phase_convention_is_stated():
    assert gfem.conventions.BLOCH_PHASE == "E(r + a) = exp(-i k·a) E(r)"
    assert "BLOCH_PHASE" in gfem.conventions.__all__


# --- job files ----------------------------------------------------------------------


def _round_trip(job, tmp_path):
    return gfem.Job.from_json(job.to_json(tmp_path / "job.json"))


@pytest.mark.parametrize(
    "problem",
    [
        gfem.Driven(pc.C1_FREQUENCY),
        gfem.Driven(pc.C1_FREQUENCY, wave_vector=pc.C1_K),
        gfem.Eigenmode(5e9, n_modes=2),
        gfem.Eigenmode(5e9, n_modes=2, wave_vector=pc.C1_K),
    ],
)
def test_one_pair_and_a_wave_vector_round_trip(tmp_path, problem):
    model = pc.c1(pc.c1_mesh(tmp_path))
    back = _round_trip(gfem.Job(model, problem), tmp_path)
    assert back.model.boundaries == model.boundaries
    assert back.model.periodic_translations == model.periodic_translations
    assert back.problem == problem


def test_three_pairs_round_trip(tmp_path, c4_mesh):
    model = pc.c4(c4_mesh)
    problem = gfem.Eigenmode(5e9, wave_vector=pc.C4_K)
    back = _round_trip(gfem.Job(model, problem), tmp_path)
    assert back.model.boundaries == model.boundaries
    assert back.model.periodic_translations == model.periodic_translations
    assert back.problem.wave_vector == pc.C4_K


def test_the_job_file_lists_the_pair_surfaces(tmp_path, c4_mesh):
    job = gfem.Job(pc.c4(c4_mesh), gfem.Eigenmode(5e9))
    doc = json.loads(job.to_json(tmp_path / "job.json").read_text())
    boundaries = doc["job"]["model"]["boundaries"]
    assert boundaries[0] == {"$type": "Periodic", "surfaces": ["x_min", "x_max"]}
    assert doc["job"]["problem"]["wave_vector"] is None


@pytest.mark.parametrize(
    "problem", [gfem.Driven(pc.C1_FREQUENCY), gfem.Eigenmode(5e9, n_modes=2)]
)
def test_a_job_written_before_wave_vectors_reads_plain(tmp_path, problem):
    model = pc.c1(pc.c1_mesh(tmp_path))
    path = gfem.Job(model, problem).to_json(tmp_path / "job.json")
    doc = json.loads(path.read_text())
    del doc["job"]["problem"]["wave_vector"]
    path.write_text(json.dumps(doc))
    assert gfem.Job.from_json(path).problem.wave_vector is None


# --- the test mesh tool -------------------------------------------------------------


def test_periodic_cell_refuses_mirror_on_a_paired_axis():
    with pytest.raises(ValueError, match="mirror on the paired axis 'y'"):
        _box(cells=(4, 4, 1), mirror=(False, True, False))
    nodes, *_ = _box(cells=(4, 4, 1), paired="xz", mirror=(False, True, False))
    assert len(nodes) == 5 * 5 * 2
