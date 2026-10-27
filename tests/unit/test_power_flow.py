"""`gfem.power_flow`: the net power a solved driven field carries out of a side.

A parallel-plate TEM line with wave ports carries an exact TEM mode, so the
power through any surface equals the network power 1 - |S11|^2 = |S21|^2
to solver precision, whatever the mesh error of the field itself. The line
is structured so that its permittivity step is a plane: on the unstructured
tests/unit/data/parallel_plate.msh the step is a staircase that scatters
into evanescent modes reaching the ports, whose terms absorb that non-modal
power (1 - |S11|^2 - |S21|^2 = 9e-4 there, and power_flow is plane-
independent but 3e-4 off 1 - |S11|^2).
"""

import dataclasses

import jax
import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm

import gfem

LENGTH, WIDTH, HEIGHT = 20.0, 4.0, 2.0  # um, the line along x
CELLS = (8, 1, 1)
F0 = 2.0e12
EPS = (4.0, 9.0)  # before and after the step at mid-length
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
# um: the grid planes x = 2.5, 7.5, 12.5, 15 and two staircases through cells
BEFORE = (2.5, 4.0, 7.5)
AFTER = (12.5, 13.6, 15.0)


@pytest.fixture(scope="module")
def plate_mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    sm.write_msh22(path, sm.strip_line(LENGTH, WIDTH, HEIGHT, CELLS))
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def _centroids(mesh, region="dielectric"):
    return mesh.nodes[mesh.tetrahedra[mesh.elements(region)]].mean(axis=1)


def _line(mesh, tan_delta=None):
    x = _centroids(mesh)[:, 0]
    fill = {"eps_r": np.where(x < LENGTH / 2 * 1e-6, *EPS)}
    if tan_delta is not None:
        fill["tan_delta"] = tan_delta
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.MaterialField(**fill)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )


def _solve(model):
    problem = gfem.Driven(frequency=F0, excite="p1", keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    s = np.asarray(result.s.values)[0]
    return result.e_field(F0), abs(s[0, 0]) ** 2, abs(s[1, 0]) ** 2


@pytest.fixture(scope="module")
def step(plate_mesh):
    model = _line(plate_mesh)
    return model, *_solve(model)


def _below(x_um):
    return lambda c: c[:, 0] < x_um * 1e-6


def test_power_through_a_plane_is_the_network_power(step):
    model, e, s11, s21 = step
    assert 0.03 < s11 < 0.05  # a non-trivial step
    for x in BEFORE:
        p = float(gfem.power_flow(e, _below(x), model=model))
        assert p == pytest.approx(1 - s11, rel=1e-9), x
    for x in AFTER:
        p = float(gfem.power_flow(e, _below(x), model=model))
        assert p == pytest.approx(s21, rel=1e-9), x


def test_power_flow_is_the_same_through_every_plane(step):
    model, e, _, _ = step
    p = [float(gfem.power_flow(e, _below(x), model=model)) for x in BEFORE + AFTER]
    assert np.ptp(p) <= 1e-11


def test_power_flow_is_positive_out_of_the_side(step):
    model, e, _, _ = step
    forward = float(gfem.power_flow(e, _below(7.5), model=model))
    backward = float(gfem.power_flow(e, lambda c: c[:, 0] > 7.5e-6, model=model))
    assert forward > 0.9
    assert backward == pytest.approx(-forward, abs=1e-12)


@pytest.fixture(scope="module")
def open_line(tmp_path_factory):
    # a two-layer line with lumped ports and absorbing side walls, a "core"
    # region across both layers for 1 < x < 3 mm
    path = tmp_path_factory.mktemp("open") / "open.msh"
    layers = (("sub", 0.5), ("cover", 0.5))
    raw = sm.strip_line(4.0, 2.0, 1.0, (8, 2, 1), layers=layers, core=(1.0, 3.0))
    sm.write_msh22(path, raw)
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={
            "sub": gfem.Material(eps_r=4.0),
            "cover": gfem.Material(eps_r=2.0),
            "core": gfem.Material(eps_r=3.0),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.Absorbing("side")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )
    problem = gfem.Driven(frequency=25e9, excite="p1", keep_fields=True)
    options = gfem.SolverOptions(order=2, linear_solver="direct")
    return model, gfem.solve(model, problem, options=options).e_field(25e9)


def test_a_side_and_its_complement_carry_opposite_power(open_line):
    # the absorbing faces on the surface's edge are split once, not dropped
    model, e = open_line
    for x in (1.7e-3, 2.0e-3, 2.5e-3):
        out = float(gfem.power_flow(e, lambda c, x=x: c[:, 0] < x, model=model))
        back = float(gfem.power_flow(e, lambda c, x=x: c[:, 0] >= x, model=model))
        assert abs(out) > 0.01
        assert abs(out + back) <= 1e-12, x


def test_a_region_and_its_predicate_give_the_same_power(open_line):
    model, e = open_line

    def core(c):
        return (c[:, 0] > 1e-3) & (c[:, 0] < 3e-3)

    by_name = float(gfem.power_flow(e, "core", model=model))
    assert by_name < 0  # the core's absorbing walls take power in
    assert float(gfem.power_flow(e, core, model=model)) == by_name
    rest = float(gfem.power_flow(e, ["sub", "cover"], model=model))
    assert float(gfem.power_flow(e, lambda c: ~core(c), model=model)) == rest


def _tetrahedron_rule(n):
    """Collapsed Gauss rule on the unit tetrahedron, exact to degree 2n - 3."""
    t, w = np.polynomial.legendre.leggauss(n)
    t, w = (t + 1) / 2, w / 2
    u, v, s = (a.ravel() for a in np.meshgrid(t, t, t, indexing="ij"))
    weight = np.einsum("i,j,k->ijk", w, w, w).ravel() * (1 - u) ** 2 * (1 - v)
    points = np.stack([u, v * (1 - u), s * (1 - u) * (1 - v)], axis=1)
    return points, weight


def _band_loss(e, mesh, band, tan_delta):
    """``1/2 ω eps0 eps_r tanδ ∫ |E|^2`` over the band, from point values."""
    ref, weight = _tetrahedron_rule(5)  # |E|^2 of order 3: degree 6
    total = 0.0
    eps_r = np.where(_centroids(mesh)[:, 0] < LENGTH / 2 * 1e-6, *EPS)
    tets = mesh.elements("dielectric")
    for k in np.flatnonzero(band):
        p = mesh.nodes[mesh.tetrahedra[tets[k]]]
        jac = (p[1:] - p[0]).T
        values = np.asarray(e.at(p[0] + ref @ jac.T))
        density = (np.abs(values) ** 2).sum(axis=1)
        total += eps_r[k] * abs(np.linalg.det(jac)) * (weight @ density)
    omega = 2 * np.pi * F0
    return 0.5 * omega * const.epsilon_0 * tan_delta * total


def test_a_lossy_band_absorbs_the_power_between_two_planes(plate_mesh):
    tan_delta = 0.05
    x = _centroids(plate_mesh)[:, 0]
    band = (x > 7.5e-6) & (x < 12.5e-6)
    model = _line(plate_mesh, tan_delta=np.where(band, tan_delta, 0.0))
    e, s11, s21 = _solve(model)
    before = float(gfem.power_flow(e, _below(5.0), model=model))
    after = float(gfem.power_flow(e, _below(15.0), model=model))
    absorbed = _band_loss(e, plate_mesh, band, tan_delta)
    assert absorbed > 0.01
    assert before - after == pytest.approx(absorbed, rel=1e-9)
    assert before - after == pytest.approx(1 - s11 - s21, rel=1e-9)


def test_power_flow_names_what_it_needs(step):
    model, e, _, _ = step
    side = _below(7.5)
    with pytest.raises(TypeError, match=r"gfem\.Field"):
        gfem.power_flow(np.asarray(e.coefficients), side, model=model)
    with pytest.raises(TypeError, match=r"model= a gfem\.Model"):
        gfem.power_flow(e, side, model=model.mesh)
    for bad in (3, [], ["dielectric", 2]):
        with pytest.raises(TypeError, match="side must be"):
            gfem.power_flow(e, bad, model=model)


def test_power_flow_rejects_a_port_mode_field(plate_mesh):
    problem = gfem.Driven(frequency=F0, excite="p1", keep_fields=True)
    result = gfem.solve(_line(plate_mesh), problem, options=OPTIONS)
    mode = result.port_modes["p1"].e_field
    with pytest.raises(TypeError, match="PortModeField"):
        gfem.power_flow(mode, _below(7.5), model=_line(plate_mesh))


def test_power_flow_checks_the_side(step):
    model, e, _, _ = step
    with pytest.raises(gfem.RegionNotFoundError):
        gfem.power_flow(e, "core", model=model)
    with pytest.raises(ValueError, match="'top' is a surface region"):
        gfem.power_flow(e, ["dielectric", "top"], model=model)
    with pytest.raises(ValueError, match="one boolean per tetrahedron"):
        gfem.power_flow(e, lambda c: c[:, 0], model=model)
    with pytest.raises(ValueError, match="one boolean per tetrahedron"):
        gfem.power_flow(e, lambda c: c[:-1, 0] < 0, model=model)
    with pytest.raises(ValueError, match="selects no element"):
        gfem.power_flow(e, _below(-1.0), model=model)
    with pytest.raises(ValueError, match="the whole solve domain"):
        gfem.power_flow(e, "dielectric", model=model)


def test_power_flow_raises_next_to_a_port(step):
    # the Kuhn tetrahedra of an end cell with centroids at mid-cell share an
    # edge with the port face, but not the face itself: its term is outside
    model, e, s11, s21 = step
    for x, port in ((1.25, "p1"), (18.75, "p2")):
        mid = lambda c, x=x: np.abs(c[:, 0] - x * 1e-6) < 0.1e-6  # noqa: E731
        with pytest.raises(ValueError, match=f"port {port!r}"):
            gfem.power_flow(e, mid, model=model)
    # the elements on a port face may be outside, if A has no edge on it
    out = float(gfem.power_flow(e, _below(17.5), model=model))
    back = float(gfem.power_flow(e, lambda c: c[:, 0] > 2.5e-6, model=model))
    assert out == pytest.approx(s21, rel=1e-9)
    assert back == pytest.approx(s11 - 1, rel=1e-9)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: e.b(), "quantity 'B' in 'hdiv'"),
        (lambda e: dataclasses.replace(e, omega=None), "static"),
        (
            lambda e: dataclasses.replace(e, omega=e.omega * (1 + 1e-3j)),
            "eigenmode fields have no time-averaged power flow",
        ),
        (
            lambda e: dataclasses.replace(e, coefficients=e.coefficients.real),
            "real coefficients",
        ),
        (lambda e: dataclasses.replace(e, order=2), "not solved on this model"),
    ],
)
def test_power_flow_needs_a_driven_e_field(step, change, message):
    model, e, _, _ = step
    with pytest.raises(ValueError, match=message):
        gfem.power_flow(change(e), _below(7.5), model=model)


def test_power_flow_needs_the_model_the_field_was_solved_on(step, open_line):
    model, e, _, _ = step
    other, _ = open_line
    with pytest.raises(ValueError, match="another mesh"):
        gfem.power_flow(e, _below(7.5), model=other)
    moved = model.replace(mesh=gfem.Mesh(model.mesh.nodes * 1.01, model.mesh._topology))
    with pytest.raises(ValueError, match="another mesh"):
        gfem.power_flow(e, _below(7.5), model=moved)


def test_power_flow_is_not_differentiable_yet(step):
    model, e, _, _ = step

    def through(coefficients):
        field = dataclasses.replace(e, coefficients=coefficients)
        return gfem.power_flow(field, _below(7.5), model=model)

    with pytest.raises(gfem.NonDifferentiableError, match="power_flow"):
        jax.grad(through)(e.coefficients.real)
    with pytest.raises(gfem.NonDifferentiableError, match="power_flow"):
        jax.jit(lambda m: gfem.power_flow(e, _below(7.5), model=m))(model)
