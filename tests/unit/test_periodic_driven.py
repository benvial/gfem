"""Plain periodic driven solves, and the rule refusals of periodic models.

A plain periodic pair (no wave vector) whose field is uniform across the
pair acts as a PMC wall (U2); a plain periodic two-port line (C1 at k = 0)
gives a symmetric S equal to the closed form, and fields on the unpaired
space, equal on twin points of the pair. A mesh move that breaks a pair
raises where the solve first sees concrete nodes.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
import scipy.constants as const
import structured_meshes as sm

import gfem

MM = 1e-3


@pytest.fixture(scope="module")
def c1_mesh(tmp_path_factory):
    return pc.c1_mesh(tmp_path_factory.mktemp("c1"))


def _s_from_z(z, r=50.0):
    eye = np.eye(len(z))
    return (z - r * eye) @ np.linalg.inv(z + r * eye)


# --- U2: plain periodic y walls = PMC y walls -----------------------------------------

L, W, H = 10.0, 2.0, 1.0  # mm
LINE_EPS = 4.0
LINE_FREQUENCIES = (3e9, 8e9)
#: |S_periodic - S_pmc|: the two walls are the same condition here.
WALLS_TOL = 1e-6
#: |S - S_exact| of the order 2 line on (16, 2, 1) cells.
LINE_S_TOL = 5e-5


def _line(tmp_path, periodic):
    edges = tuple(
        np.linspace(0, s, n + 1) for s, n in zip((L, W, H), (16, 2, 1), strict=True)
    )
    path = sm.write_msh22(tmp_path / "line.msh", sm.periodic_cell(edges))
    boundaries = [gfem.PEC("z_min"), gfem.PEC("z_max")]
    if periodic:
        boundaries.append(gfem.Periodic(("y_min", "y_max")))
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"cell": gfem.Material(eps_r=LINE_EPS)},
        boundaries=boundaries,
        ports=[
            gfem.LumpedPort("p1", "x_min", "+z"),
            gfem.LumpedPort("p2", "x_max", "+z"),
        ],
    )
    result = gfem.solve(
        model,
        gfem.Driven(frequency=list(LINE_FREQUENCIES)),
        options=gfem.SolverOptions(order=2, linear_solver="direct"),
    )
    return np.asarray(result.s.values)


def _line_exact(f):
    """S of a matched-reference TEM line of length L (50 ohm ports)."""
    beta = 2 * math.pi * f / const.c * math.sqrt(LINE_EPS)
    z0 = const.mu_0 * const.c * H / (W * math.sqrt(LINE_EPS))
    c, s = math.cos(beta * L * MM), math.sin(beta * L * MM)
    a, b, cc, d, r = c, 1j * z0 * s, 1j * s / z0, c, 50.0
    den = a * r + b + cc * r * r + d * r
    s11 = (a * r + b - cc * r * r - d * r) / den
    s21 = 2 * r * (a * d - b * cc) / den
    return np.array([[s11, s21], [s21, s11]])


def test_plain_periodic_walls_equal_pmc_walls(tmp_path):
    periodic = _line(tmp_path, periodic=True)
    pmc = _line(tmp_path, periodic=False)
    exact = np.array([_line_exact(f) for f in LINE_FREQUENCIES])
    assert np.abs(periodic - pmc).max() < WALLS_TOL
    assert np.abs(pmc - exact).max() < LINE_S_TOL
    assert np.abs(periodic - exact).max() < LINE_S_TOL


# --- C1 without a wave vector ---------------------------------------------------------


#: |S - S^T|: round-off of a factorization; GMRES's stopping tolerance.
SYMMETRY_TOL = {"direct": 1e-12, "bddc_gmres": 1e-6}


@pytest.mark.parametrize("linear_solver", ["direct", "bddc_gmres"])
def test_plain_periodic_line_is_reciprocal_and_matches_k_zero(c1_mesh, linear_solver):
    result = gfem.solve(
        pc.c1(c1_mesh),
        gfem.Driven(pc.C1_FREQUENCY),
        options=gfem.SolverOptions(order=pc.C1_ORDER, linear_solver=linear_solver),
    )
    (s,) = np.asarray(result.s.values)
    eps = pc.C1_EPS * (1 - 1j * pc.C1_TAN_DELTA)
    exact = _s_from_z(pc.bloch_line_z(pc.C1_FREQUENCY, 0.0, eps))
    np.testing.assert_allclose(s, s.T, rtol=0, atol=SYMMETRY_TOL[linear_solver])
    assert np.abs(s - exact).max() < pc.C1_S_TOL


def test_plain_periodic_fields_live_on_the_unpaired_space(c1_mesh):
    problem = gfem.Driven(pc.C1_FREQUENCY, excite=["p1"], keep_fields=True)
    options = gfem.SolverOptions(order=pc.C1_ORDER, linear_solver="direct")
    periodic = gfem.solve(pc.c1(c1_mesh), problem, options=options)
    unpaired = pc.c1(c1_mesh).replace(boundaries=[gfem.PEC("z_min"), gfem.PEC("z_max")])
    walls = gfem.solve(unpaired, problem, options=options)
    (field,) = periodic.e_fields
    assert field.coefficients.shape == walls.e_fields[0].coefficients.shape
    # twin points of the pair: the tangential components are one field
    y = np.array([0.3, 0.9, 1.7]) * MM
    z = np.array([0.2, 0.5, 0.8]) * MM
    first = np.stack([np.zeros(3), y, z], axis=1)
    second = first + np.array([pc.C1_A * MM, 0, 0])
    e1, e2 = np.asarray(field.at(first)), np.asarray(field.at(second))
    scale = np.abs(e1).max()
    assert scale > 0
    np.testing.assert_allclose(e2[:, 1:], e1[:, 1:], rtol=0, atol=1e-10 * scale)


# --- a move that breaks a pair --------------------------------------------------------


def _broken(mesh):
    """Displacements moving the x_max nodes off their twins by 1 um in y."""
    shift = np.zeros_like(mesh.imported_nodes)
    on = np.abs(mesh.imported_nodes[:, 0] - pc.C1_A * MM) < 1e-12
    shift[on, 1] = 1e-6
    return shift


BROKEN = (
    r"The moved mesh breaks Periodic\(\('x_min', 'x_max'\)\): \d+ node pairs are "
    r"no longer related by the translation a = \(0\.01, 0, 0\) m"
)


def test_a_pair_breaking_move_raises_at_the_solve(c1_mesh):
    model = pc.c1(c1_mesh)
    leaves, tree = jax.tree_util.tree_flatten(model)
    nodes = np.asarray(c1_mesh.nodes)
    moved = [x + _broken(c1_mesh) if np.shape(x) == nodes.shape else x for x in leaves]
    # unflattening does not validate: the solve's checkpoint must
    broken = jax.tree_util.tree_unflatten(tree, moved)
    with pytest.raises(ValueError, match=BROKEN) as caught:
        gfem.solve(broken, gfem.Driven(pc.C1_FREQUENCY))
    assert not isinstance(caught.value, gfem.MeshQualityError)


def test_a_pair_breaking_move_raises_under_grad(c1_mesh):
    model = pc.c1(c1_mesh)
    shift = jnp.asarray(_broken(c1_mesh))

    def objective(t):
        mesh = jax.tree_util.tree_map(lambda x: x + t * shift, c1_mesh)
        result = gfem.solve(model.replace(mesh=mesh), gfem.Driven(pc.C1_FREQUENCY))
        return jnp.abs(result.s.values[0, 1, 0]) ** 2

    with pytest.raises(jax.errors.JaxRuntimeError, match=BROKEN):
        jax.grad(objective)(1.0)


# --- refusals before assembly ---------------------------------------------------------


def test_electrostatic_refuses_a_periodic_pair(c1_mesh):
    with pytest.raises(
        NotImplementedError,
        match=r"Electrostatic does not support periodic pairs .*Periodic\(\('x_min', "
        r"'x_max'\)\).*use a gfem\.Driven surrogate",
    ):
        gfem.solve(pc.c1(c1_mesh), gfem.Electrostatic(terminals=["z_max"]))


def test_a_wave_vector_needs_a_periodic_pair(c1_mesh):
    model = pc.c1(c1_mesh).replace(boundaries=[gfem.PEC("z_min"), gfem.PEC("z_max")])
    with pytest.raises(
        ValueError,
        match=r"Driven\.wave_vector sets a Bloch phase, but the model has no "
        r"gfem\.Periodic pair",
    ):
        gfem.solve(model, gfem.Driven(pc.C1_FREQUENCY, wave_vector=pc.C1_K))


def test_a_wave_vector_off_the_translations_is_refused(c1_mesh):
    with pytest.raises(
        ValueError,
        match=r"Driven\.wave_vector \(94\.2478, 50, 0\) rad/m has a component of "
        r"50 rad/m outside the span of the periodic translations \(0\.01, 0, 0\) m"
        r".*e\.g\. \(94\.2478, 0, 0\) rad/m",
    ):
        gfem.solve(
            pc.c1(c1_mesh),
            gfem.Driven(pc.C1_FREQUENCY, wave_vector=(pc.C1_K[0], 50.0, 0.0)),
        )
