"""Driven boundaries with closed forms: Impedance, Conductor sheets, Absorbing.

A parallel-plate line along x (PEC plates at z = 0 and z = h, PMC side walls)
carries an exact TEM wave, so a surface across it (a y-z plane of width W and
height h) loads the line with the lumped admittance ``Y_s W / h`` of its
per-square admittance ``Y_s``, and S has a closed form (ABCD matrices).

With PEC side walls the same box is a rectangular waveguide whose TE_m0 modes
are orthogonal on the end face under both the mass and the curl-trace forms,
so a second-order absorbing end face acts on TE10 exactly like a surface
admittance ``k_eff / (omega mu0)``, with ``k_eff = k - k_t^2 / (2 k)`` in the
textbook form.

Next to an anisotropic material the first-order absorbing boundary applies
the normal-incidence admittance tensor (ADR 0051): checked against the 6x6
plane-wave reference of ``tests/anisotropy.py``, and on a parallel-plate
line filled with a crystal and ended by the boundary, which then reflects
like a half-space of the crystal, at every length.
"""

import math

import anisotropy as anisotropy_reference
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as const
import scipy.linalg
import structured_meshes as sm

import gfem
from gfem import conventions
from gfem._engine import anisotropy, terms
from gfem._engine.driven import DrivenRule

L, W, H = 10.0, 2.0, 1.0  # mm
EPS = 4.0
F = 5e9
R = 50.0
ETA0 = const.mu_0 * const.c


def _abcd_line(f, length_m, eps=EPS):
    beta = 2 * math.pi * f / const.c * math.sqrt(eps)
    z0 = ETA0 * H / (W * math.sqrt(eps))
    c, s = math.cos(beta * length_m), math.sin(beta * length_m)
    return np.array([[c, 1j * z0 * s], [1j * s / z0, c]])


def _shunt(y):
    return np.array([[1, 0], [y, 1]], complex)


def _s_from_abcd(m, r=R):
    (a, b), (c, d) = m
    den = a + b / r + c * r + d
    return np.array(
        [
            [(a + b / r - c * r - d) / den, 2 * (a * d - b * c) / den],
            [2 / den, (-a + b / r - c * r + d) / den],
        ]
    )


def _s11_loaded(m, y_load, r=R):
    """Input reflection of a two-port ABCD ``m`` closed by admittance ``y_load``."""
    (a, b), (c, d) = m
    z_in = (a + b * y_load) / (c + d * y_load)
    return (z_in - r) / (z_in + r)


def _s(model, frequency=F, **options):
    options.setdefault("linear_solver", "direct")
    result = gfem.solve(
        model, gfem.Driven(frequency=frequency), options=gfem.SolverOptions(**options)
    )
    return np.asarray(result.s.values)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (10, 1, 1)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _one_port(mesh, end):
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), end],
        ports=[gfem.LumpedPort("p1", "port1", "+z", impedance=R)],
    )


# --- Impedance -------------------------------------------------------------------


@pytest.mark.parametrize(
    "rlc",
    [
        {"resistance": 30.0},
        {"resistance": 80.0, "inductance": 1e-9},
        {"resistance": 120.0, "capacitance": 0.2e-12},
        {"inductance": 2e-9, "capacitance": 0.1e-12},
    ],
)
def test_impedance_end_face_loads_the_line_with_its_parallel_rlc(mesh, rlc):
    model = _one_port(mesh, gfem.Impedance("port2", **rlc))
    s11 = _s(model)[0, 0, 0]
    w = 2 * math.pi * F
    y_square = (
        1 / rlc.get("resistance", math.inf)
        + (1 / (1j * w * rlc["inductance"]) if "inductance" in rlc else 0)
        + 1j * w * rlc.get("capacitance", 0.0)
    )
    expected = _s11_loaded(_abcd_line(F, L * 1e-3), y_square * W / H)
    assert abs(s11 - expected) < 2e-3


def test_new_boundary_terms_are_affine_pieces(mesh):
    rlc = gfem.Impedance("port2", resistance=1.0, inductance=1e-9, capacitance=1e-12)
    abc = gfem.Absorbing("side", order=2)
    model = _one_port(mesh, rlc).replace(
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), rlc, abc]
    )
    rule = DrivenRule(model, gfem.Driven(frequency=F), gfem.SolverOptions())
    op = rule.operator(model)
    theta = {p.name: p.theta for p in op.pieces}
    w = 2 * math.pi * F
    k0 = op.wavenumber(w)
    assert theta["absorbing order 2"](w) == pytest.approx(-0.5j / k0)
    assert theta["impedance port2 R"](w) == pytest.approx(1j * k0)
    assert theta["impedance port2 L"](w) == 1.0
    assert theta["impedance port2 C"](w) == pytest.approx(-(k0**2))


def test_matched_impedance_is_the_first_order_absorbing_boundary(mesh):
    matched = gfem.Impedance("port2", resistance=ETA0 / math.sqrt(EPS))
    s_imp = _s(_one_port(mesh, matched), frequency=[2e9, 6e9])
    s_abc = _s(_one_port(mesh, gfem.Absorbing("port2")), frequency=[2e9, 6e9])
    np.testing.assert_allclose(s_imp, s_abc, atol=1e-12)


# --- Conductor sheets ------------------------------------------------------------

#: A resistive film far thinner than its skin depth (225 um at 5 GHz).
SIGMA, THICKNESS = 1e3, 5e-6


def _sheet_line(tmp):
    """The line cut at x = L / 2 by the interior surface "sheet"."""
    eps = 1e-9 * L

    def volume(c):
        return "left" if c[0] < L / 2 else "right"

    def surface(c):
        for x, name in ((0.0, "port1"), (L, "port2"), (L / 2, "sheet")):
            if abs(c[0] - x) < eps:
                return name
        if abs(c[2]) < eps:
            return "bottom"
        if abs(c[2] - H) < eps:
            return "top"
        return "side"

    edges = (np.linspace(0, L, 11), np.linspace(0, W, 2), np.linspace(0, H, 2))
    path = sm.write_msh22(tmp / "sheet.msh", sm.structured(edges, volume, surface))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def test_the_interior_sheet_convention_is_stated():
    assert conventions.INTERIOR_SHEET_ADMITTANCE == 0.5
    assert conventions.EXTERIOR_THICKNESS_FACTOR == 2.0


def test_thin_interior_conductor_is_a_shunt_of_half_its_film_conductance(tmp_path):
    material = gfem.Material(eps_r=EPS)
    model = gfem.Model(
        _sheet_line(tmp_path),
        materials={"left": material, "right": material},
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Conductor("sheet", SIGMA, THICKNESS),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    s = _s(model)[0]
    # one interior term of a thin film: sheet admittance sigma t / 2 per square
    half = _abcd_line(F, L / 2 * 1e-3)
    sheet = _shunt(0.5 * SIGMA * THICKNESS * W / H)
    np.testing.assert_allclose(s, _s_from_abcd(half @ sheet @ half), atol=2e-3)
    without = _s_from_abcd(half @ half)
    assert abs(s[1, 0] - without[1, 0]) > 0.1


def test_thin_exterior_conductor_counts_its_thickness_twice(mesh):
    s11 = _s(_one_port(mesh, gfem.Conductor("port2", SIGMA, THICKNESS)))[0, 0, 0]
    expected = _s11_loaded(_abcd_line(F, L * 1e-3), SIGMA * THICKNESS * W / H)
    assert abs(s11 - expected) < 2e-3


def test_thin_film_surface_impedance_tends_to_two_over_its_conductance():
    w = 2 * math.pi * F
    z = terms.surface_impedance(SIGMA, THICKNESS, w)
    assert 1 / z == pytest.approx(0.5 * SIGMA * THICKNESS, rel=1e-4)


# --- second-order absorbing boundary ---------------------------------------------

#: Rectangular waveguide along x (mm): TE10 varies across y, cutoff 7.5 GHz.
GUIDE = (40.0, 20.0, 5.0)
F_GUIDE = 10e9


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    path = tmp_path_factory.mktemp("guide") / "guide.msh"
    sm.write_msh22(path, sm.strip_line(*GUIDE, (16, 8, 1)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _guide_s11(mesh, end):
    model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=1.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.PEC("side"), end],
        ports=[gfem.LumpedPort("p1", "port1", "+z", impedance=R)],
    )
    return _s(model, frequency=F_GUIDE, order=3)[0, 0, 0]


def test_second_order_absorbing_boundary_is_the_textbook_condition(guide):
    k = 2 * math.pi * F_GUIDE / const.c
    kt = math.pi / (GUIDE[1] * 1e-3)
    # On TE10 the condition is a surface admittance k_eff / (omega mu0):
    # textbook k_eff = k - kt^2 / (2k) (beta = 1 / (2ik), Jin 9.3); Palace
    # <= 0.18 has the opposite sign of the curl term, k + kt^2 / (2k).
    textbook = gfem.Impedance("port2", resistance=ETA0 * k / (k - kt**2 / (2 * k)))
    palace = gfem.Impedance("port2", resistance=ETA0 * k / (k + kt**2 / (2 * k)))
    s_abc2 = _guide_s11(guide, gfem.Absorbing("port2", order=2))
    s_text = _guide_s11(guide, textbook)
    s_palace = _guide_s11(guide, palace)
    s_abc1 = _guide_s11(guide, gfem.Absorbing("port2", order=1))
    # the forms differ by 0.1-0.2 in S11; the two discretizations of the
    # textbook condition agree to ~4e-9
    assert abs(s_text - s_palace) > 0.1
    assert abs(s_text - s_abc1) > 0.05
    assert abs(s_abc2 - s_text) < 1e-6


# --- absorbing boundary next to an anisotropic material (ADR 0051) ---------------

#: A biaxial, magnetic crystal on the coordinate axes.
CRYSTAL = gfem.Material(eps_r=(2.0, 3.6, 2.5), mu_r=(1.5, 1.0, 1.2))


def _admittance_cases(count):
    """Random symmetric eps, mu (every other one lossy) and unit normals."""
    rng = np.random.default_rng(0)
    for trial in range(count):
        eps = anisotropy_reference.symmetric(rng, 1.0, 12.0)
        mu = anisotropy_reference.symmetric(rng, 0.5, 3.0)
        if trial % 2:  # e^{+iωt}: a lossy medium has Im < 0
            eps = eps - 1j * anisotropy_reference.symmetric(rng, 0.0, 1.0)
            mu = mu - 1j * anisotropy_reference.symmetric(rng, 0.0, 0.3)
        n = rng.normal(size=3)
        yield eps, mu, n / np.linalg.norm(n)


def _helper_admittance(eps, mu, n):
    return anisotropy.admittance(eps[None], mu[None], n[None], where="test")[0]


def test_admittance_is_the_plane_wave_admittance_at_normal_incidence():
    worst = 0.0
    for eps, mu, n in _admittance_cases(200):
        y = _helper_admittance(eps, mu, n)
        ref = anisotropy_reference.admittance(eps, mu, n)
        worst = max(worst, np.abs(y - ref).max() / np.abs(ref).max())
        assert np.abs(y - y.T).max() <= 1e-14 * np.abs(y).max()
        assert np.abs(y @ n).max() <= 1e-14 * np.abs(y).max()
    assert worst < 1e-12


def test_admittance_closed_form_is_the_matrix_geometric_mean():
    j = np.array([[0.0, -1.0], [1.0, 0.0]])  # n x in the frame (t1, t2)
    for eps, mu, n in _admittance_cases(20):
        t = anisotropy_reference.frame(n)
        a = j @ (t @ np.linalg.inv(mu) @ t.T) @ j.T
        b = np.linalg.inv(t @ np.linalg.inv(eps) @ t.T)
        half = scipy.linalg.sqrtm(a)
        inverse_half = np.linalg.inv(half)
        mean = half @ scipy.linalg.sqrtm(inverse_half @ b @ inverse_half) @ half
        y = t @ _helper_admittance(eps, mu, n) @ t.T
        np.testing.assert_allclose(y, mean, rtol=0, atol=1e-12 * np.abs(mean).max())


def test_admittance_limits():
    n = np.array([0.0, 0.0, 1.0])
    y = _helper_admittance(4.0 * np.eye(3), 2.0 * np.eye(3), n)
    np.testing.assert_allclose(y, math.sqrt(2.0) * np.diag([1, 1, 0]), atol=1e-15)
    # the diagonal crystal pairs eps_x with mu_y and eps_y with mu_x
    y = _helper_admittance(np.diag([2.0, 3.6, 2.5]), np.diag([1.5, 1.0, 1.2]), n)
    expected = np.diag([math.sqrt(2.0 / 1.0), math.sqrt(3.6 / 1.5), 0.0])
    np.testing.assert_allclose(y, expected, atol=1e-15)


def test_admittance_without_a_real_geometric_mean_names_the_surface():
    n = np.array([[0.0, 0.0, 1.0]])
    indefinite = np.diag([2.0, -1.0, 1.0])[None]
    with pytest.raises(
        ValueError, match=r"Absorbing 'box'.*eps_eff.*positive definite"
    ):
        anisotropy.admittance(indefinite, np.eye(3)[None], n, where="Absorbing 'box'")
    with pytest.raises(ValueError, match=r"Absorbing 'box'.*\bA\b.*positive definite"):
        anisotropy.admittance(np.eye(3)[None], indefinite, n, where="Absorbing 'box'")


#: A square parallel-plate section (mm) along z: the slab fills it.
SIDE, LENGTH = 2.0, 10.0


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    """Box ``SIDE x SIDE x LENGTH`` mm: volumes "a" (x < SIDE/2) and "b".

    Surfaces "port" (z = 0), "end" (z = LENGTH), "xwalls" and "ywalls".
    """
    eps = 1e-9 * LENGTH

    def volume(c):
        return "a" if c[0] < SIDE / 2 else "b"

    def surface(c):
        if abs(c[2]) < eps:
            return "port"
        if abs(c[2] - LENGTH) < eps:
            return "end"
        if abs(c[0]) < eps or abs(c[0] - SIDE) < eps:
            return "xwalls"
        if abs(c[1]) < eps or abs(c[1] - SIDE) < eps:
            return "ywalls"
        return ""

    edges = (
        np.linspace(0, SIDE, 3),
        np.linspace(0, SIDE, 3),
        np.linspace(0, LENGTH, 31),
    )
    path = tmp_path_factory.mktemp("slab") / "slab.msh"
    sm.write_msh22(path, sm.structured(edges, volume, surface))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _slab_model(mesh, a, b=None, *, polarization="y", order=1):
    """The slab fed at "port" along ``polarization``, absorbing at "end".

    PEC on the walls normal to the field, PMC on the others: a TEM line of
    impedance eta0 in air, matched by the port.
    """
    return gfem.Model(
        mesh,
        materials={"a": a, "b": a if b is None else b},
        boundaries=[gfem.PEC(f"{polarization}walls"), gfem.Absorbing("end", order)],
        ports=[gfem.LumpedPort("p1", "port", f"+{polarization}", impedance=ETA0)],
    )


def test_isotropic_faces_keep_the_scalar_admittance(slab):
    iso = gfem.Material(eps_r=4.0, tan_delta=0.01, mu_r=2.0)
    surfaces, owner = terms.absorbing_faces(_slab_model(slab, iso))
    first, _ = terms.absorbing_coefficients(_slab_model(slab, iso), surfaces, owner)
    on = slab.triangle_tags == slab.region("end").tag
    assert first.ndim == 1
    assert (first[on] == math.sqrt(4.0 / 2.0)).all()
    mixed = _slab_model(slab, iso, CRYSTAL)
    tensor, _ = terms.absorbing_coefficients(mixed, surfaces, owner)
    behind_a = on & np.isin(owner, slab.elements("a"))
    assert behind_a.any()
    assert (on & ~behind_a).any()
    assert (tensor[behind_a] == math.sqrt(4.0 / 2.0) * np.eye(3)).all()
    crystal = np.diag([math.sqrt(2.0 / 1.0), math.sqrt(3.6 / 1.5), 0.0])
    np.testing.assert_allclose(tensor[on & ~behind_a], crystal[None], atol=1e-14)


@pytest.mark.parametrize(
    ("polarization", "half_space"),
    # (1 - Y) / (1 + Y), Y = sqrt(eps_y / mu_x) and sqrt(eps_x / mu_y)
    [("y", 0.2154), ("x", 0.1716)],
)
def test_first_order_absorbing_boundary_ends_an_anisotropic_slab_as_a_half_space(
    slab, polarization, half_space
):
    # k0 d = 1 and 4: with the exact admittance S11 does not depend on d
    frequencies = [k0d * const.c / (2 * math.pi * LENGTH * 1e-3) for k0d in (1.0, 4.0)]
    model = _slab_model(slab, CRYSTAL, polarization=polarization)
    s11 = _s(model, frequency=frequencies, order=3)[:, 0, 0]
    # the mean diagonal reflected 0.022 off, Palace's coefficient 0.098
    np.testing.assert_allclose(np.abs(s11), half_space, atol=0.005)
    assert abs(s11[0] - s11[1]) <= 0.005


def _eigenmode(model):
    gfem.solve(model, gfem.Eigenmode(target=10e9, n_modes=1))


def _driven(model):
    gfem.solve(model, gfem.Driven(frequency=5e9))


@pytest.mark.parametrize("solve", [_driven, _eigenmode])
def test_second_order_absorbing_boundary_refuses_an_anisotropic_material(slab, solve):
    model = _slab_model(slab, gfem.Material(eps_r=4.0), CRYSTAL, order=2)
    match = (
        r"Absorbing 'end': order 2 needs an isotropic material behind the surface; "
        r"an anisotropic eps_r / mu_r touches it\. Use order=1, or put an isotropic "
        r"layer between the material and the boundary\."
    )
    with pytest.raises(NotImplementedError, match=match):
        solve(model)


def test_second_order_absorbing_boundary_takes_a_per_axis_loss_tangent(slab):
    lossy = gfem.Material(eps_r=4.0, tan_delta=(0.01, 0.0, 0.02))
    model = _slab_model(slab, lossy, order=2)
    op = DrivenRule(model, gfem.Driven(frequency=F), gfem.SolverOptions()).operator(
        model
    )
    assert "absorbing order 2" in {p.name for p in op.pieces}


@pytest.mark.parametrize(
    "problem",
    [gfem.Driven(frequency=5e9), gfem.Eigenmode(target=10e9, n_modes=1)],
    ids=["driven", "eigenmode"],
)
def test_shape_gradients_refuse_a_tensor_absorbing_admittance(slab, problem):
    model = _slab_model(slab, gfem.Material(eps_r=4.0), CRYSTAL)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(model.mesh), [nodes]
        )
        result = gfem.solve(model.replace(mesh=mesh), problem)
        if isinstance(problem, gfem.Driven):
            return jnp.sum(jnp.abs(result.s.values) ** 2)
        return jnp.sum(jnp.real(result.omega))

    with pytest.raises(
        gfem.NonDifferentiableError,
        match=r"Absorbing surface 'end' has an anisotropic material.*stop_gradient",
    ):
        jax.make_jaxpr(jax.grad(objective))(jnp.asarray(model.mesh.nodes))
