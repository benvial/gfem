"""Interface participation of eigenmodes on parallel-plate line resonators.

The lines run along x between PEC plates at z = 0 and z = H, open (PMC)
at both ends and on the sides, so their first mode is E = E_z z-hat with
E_z = cos(pi x / L) g(y), independent of z: exact closed forms for the
energy of a thin layer on a plate or an interface.

- ``side``: substrate (eps) for y < W1 next to vacuum up to y = W1 + W2.
  The cross-section is inhomogeneous, so the mode is not TEM: g solves
  g'' = -(eps(y) k0^2 - kx^2) g with g' = 0 on the side walls and g, g'
  continuous at y = W1 (one transcendental equation for k0). With
  I_sub, I_air the integrals of g^2 over each part and D = eps I_sub +
  I_air: MA on the vacuum plate is t I_air / (eps_MA H D), MS on the
  substrate plate t eps^2 I_sub / (eps_MS H D), SA and default on the
  interface (E tangential) t eps_l g(W1)^2 / D.
- ``sheet``: a PEC sheet at z = H1 splits the line into a substrate line
  under a vacuum line, two homogeneous (TEM) resonators with one-sided
  fields on each face of the sheet. The vacuum line's mode has MA
  t / (eps_MA H2) on the sheet (vacuum side) and no MS; the substrate
  line's mode has MS t eps / (eps_MS H1) (substrate side) and no MA; the
  default kind averages the two sides, so it sees half the vacuum field.

Order-3 elements resolve g to about 1e-6.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm
from scipy.integrate import quad
from scipy.optimize import brentq

import gfem

L, H = 10.0, 1.0  # mm
W1, W2 = 1.0, 1.5  # mm
EPS = 9.0
T = 3e-9  # m
F1 = const.c / (2 * L * 1e-3)  # vacuum half-wave, 15 GHz
OPTIONS = gfem.SolverOptions(order=3)
KX = math.pi / L  # 1/mm


def _exact_side():
    """k0 (1/mm), g(y) and the energy integrals of the side-by-side line."""

    def wavenumbers(k0):
        return math.sqrt(EPS * k0**2 - KX**2), k0**2 - KX**2

    def air(q2, u):  # the air part of g, normalized at y = W1 (u = W - y)
        k2 = math.sqrt(abs(q2))
        if q2 >= 0:
            return math.cos(k2 * u) / math.cos(k2 * W2), k2 * math.tan(k2 * W2)
        return math.cosh(k2 * u) / math.cosh(k2 * W2), -k2 * math.tanh(k2 * W2)

    def mismatch(k0):  # g'/g at y = W1 from both sides
        k1, q2 = wavenumbers(k0)
        return -k1 * math.tan(k1 * W1) - air(q2, W2)[1]

    # the fundamental lies between the substrate and vacuum TEM wavenumbers
    k0 = brentq(mismatch, KX / math.sqrt(EPS) * 1.0001, KX, xtol=1e-15)
    k1, q2 = wavenumbers(k0)

    def g(y):
        if y <= W1:
            return math.cos(k1 * y) / math.cos(k1 * W1)
        return air(q2, W1 + W2 - y)[0]

    i_sub = quad(lambda y: g(y) ** 2, 0, W1, epsabs=0, epsrel=1e-13)[0]
    i_air = quad(lambda y: g(y) ** 2, W1, W1 + W2, epsabs=0, epsrel=1e-13)[0]
    return k0, g, i_sub, i_air


K0, G, I_SUB, I_AIR = _exact_side()
D = EPS * I_SUB + I_AIR  # mm


def _side_by_side():
    width = W1 + W2
    edges = (
        np.linspace(0, L, 21),
        np.concatenate([np.linspace(0, W1, 3), np.linspace(W1, width, 4)[1:]]),
        np.linspace(0, H, 2),
    )
    tol = 1e-9

    def volume(c):
        return "sub" if c[1] < W1 else "air"

    def surface(c):
        if abs(c[0]) < tol or abs(c[0] - L) < tol:
            return "ends"
        if abs(c[2]) < tol:
            return "bottom"
        if abs(c[2] - H) < tol:
            return "top_sub" if c[1] < W1 else "top_air"
        if abs(c[1] - W1) < tol:
            return "sub__air"
        return "side"

    return sm.structured(edges, volume, surface)


@pytest.fixture(scope="module")
def side(tmp_path_factory):
    path = tmp_path_factory.mktemp("side") / "side.msh"
    sm.write_msh22(path, _side_by_side())
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"sub": gfem.Material(eps_r=EPS), "air": gfem.Material()},
        boundaries=[gfem.PEC(s) for s in ("bottom", "top_sub", "top_air")],
    )
    modes = gfem.solve(model, gfem.Eigenmode(target=F1 * K0 / KX), options=OPTIONS)
    np.testing.assert_allclose(modes.frequency, [F1 * K0 / KX], rtol=1e-6)
    return modes


H1, H2 = 0.4, 0.6  # mm


@pytest.fixture(scope="module")
def sheet(tmp_path_factory):
    path = tmp_path_factory.mktemp("sheet") / "sheet.msh"
    layers = (("sub", H1), ("air", H2))
    sm.write_msh22(path, sm.strip_line(L, 2.0, H, (20, 1, 1), layers=layers))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"sub": gfem.Material(eps_r=EPS), "air": gfem.Material()},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.PEC("sub__air")],
    )
    # the vacuum line at F1, the substrate line at F1 / 3: one mode each
    vacuum = gfem.solve(model, gfem.Eigenmode(target=F1), options=OPTIONS)
    substrate = gfem.solve(
        model, gfem.Eigenmode(target=F1 / math.sqrt(EPS)), options=OPTIONS
    )
    return vacuum, substrate


# --- closed forms ------------------------------------------------------------------


def test_metal_air_participation_reads_the_vacuum_plate(side):
    p = gfem.interface_participation(side, "top_air", T, 10.0, kind="MA")
    assert isinstance(p, gfem.InterfaceParticipation)
    assert p.participation.shape == (1,)
    expected = T * I_AIR / (10.0 * H * 1e-3 * D)
    np.testing.assert_allclose(p.participation, [expected], rtol=1e-5)
    assert p.inverse_q is None
    # the substrate plate has no vacuum side: no metal-air layer
    none = gfem.interface_participation(side, "top_sub", T, 10.0, kind="MA")
    np.testing.assert_allclose(none.participation, [0.0], atol=1e-20)


def test_metal_substrate_participation_scales_with_the_substrate_permittivity(side):
    p = gfem.interface_participation(side, "top_sub", T, 5.0, kind="MS")
    expected = T * EPS**2 * I_SUB / (5.0 * H * 1e-3 * D)
    np.testing.assert_allclose(p.participation, [expected], rtol=1e-5)
    none = gfem.interface_participation(side, "top_air", T, 5.0, kind="MS")
    np.testing.assert_allclose(none.participation, [0.0], atol=1e-20)


@pytest.mark.parametrize("kind", ["SA", "default"])
def test_substrate_air_and_default_layers_read_the_tangential_field(side, kind):
    p = gfem.interface_participation(side, "sub__air", T, 4.0, kind=kind)
    expected = T * 4.0 * G(W1) ** 2 / (D * 1e-3)
    np.testing.assert_allclose(p.participation, [expected], rtol=1e-5)


def test_loss_tangent_gives_the_inverse_q_contribution(side):
    p = gfem.interface_participation(
        side, "top_air", T, 10.0, kind="MA", tan_delta=2e-3
    )
    np.testing.assert_allclose(p.inverse_q, p.participation * 2e-3, rtol=1e-15)
    np.testing.assert_allclose(p.q, 1 / p.inverse_q, rtol=1e-15)


def test_a_surface_without_field_sides_adds_nothing(tmp_path):
    # a plate on a volume excluded from the solve: no side carries the field
    path = tmp_path / "plate.msh"
    layers = (("metal", 0.5), ("oxide", 0.5))
    sm.write_msh22(path, sm.plate(4.0, 3.0, layers, (1, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"oxide": gfem.Material(eps_r=2.0)},
        boundaries=[gfem.PEC("metal__oxide"), gfem.PEC("top")],
        excluded=("metal",),
    )
    modes = gfem.solve(model, gfem.Eigenmode(target=30e9))
    p = gfem.interface_participation(modes, "bottom", T, 10.0)
    np.testing.assert_array_equal(p.participation, [0.0])


def test_several_surfaces_add_up(side):
    both = gfem.interface_participation(side, ["top_sub", "top_air"], T, 5.0, kind="MS")
    one = gfem.interface_participation(side, "top_sub", T, 5.0, kind="MS")
    np.testing.assert_allclose(both.participation, one.participation, rtol=1e-12)


def test_two_sided_sheet_picks_the_vacuum_or_substrate_side(sheet):
    vacuum, substrate = sheet
    ma = gfem.interface_participation(vacuum, "sub__air", T, 10.0, kind="MA")
    np.testing.assert_allclose(ma.participation, [T / (10.0 * H2 * 1e-3)], rtol=1e-5)
    ms = gfem.interface_participation(vacuum, "sub__air", T, 10.0, kind="MS")
    assert abs(float(ms.participation[0])) < 1e-8 * float(ma.participation[0])
    ms = gfem.interface_participation(substrate, "sub__air", T, 10.0, kind="MS")
    np.testing.assert_allclose(
        ms.participation, [T * EPS / (10.0 * H1 * 1e-3)], rtol=1e-4
    )
    ma = gfem.interface_participation(substrate, "sub__air", T, 10.0, kind="MA")
    assert abs(float(ma.participation[0])) < 1e-8 * float(ms.participation[0])
    # the default kind averages both faces: half the vacuum field
    default = gfem.interface_participation(vacuum, "sub__air", T, 3.0)
    np.testing.assert_allclose(
        default.participation, [T * 3.0 / (4 * H2 * 1e-3)], rtol=1e-4
    )


# --- validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "kwargs", "error", "match"),
    [
        (("nowhere", T, 10.0), {}, gfem.RegionNotFoundError, "nowhere"),
        (("sub", T, 10.0), {}, ValueError, "volume region"),
        (("top_air", -T, 10.0), {}, ValueError, "thickness"),
        (("top_air", T, 0.0), {}, ValueError, "eps_r"),
        (("top_air", T, 10.0), {"kind": "XY"}, ValueError, "'MA'"),
        (("top_air", T, 10.0), {"tan_delta": -1.0}, ValueError, "tan_delta"),
    ],
)
def test_bad_inputs_raise(side, args, kwargs, error, match):
    with pytest.raises(error, match=match):
        gfem.interface_participation(side, *args, **kwargs)


def test_needs_an_eigenmode_result(side):
    with pytest.raises(TypeError, match="EigenmodeResult"):
        gfem.interface_participation(side.e_field(0), "top_air", T, 10.0)


# --- JAX ---------------------------------------------------------------------------


def test_layer_parameters_differentiate(side):
    def f(t, eps):
        return gfem.interface_participation(
            side, "sub__air", t, eps, kind="SA"
        ).participation[0]

    p = f(T, 4.0)
    dt, deps = jax.grad(f, argnums=(0, 1))(T, 4.0)
    # E_n = 0 on the interface: p = t eps |E_t|^2 ...
    np.testing.assert_allclose(dt, p / T, rtol=1e-6)
    np.testing.assert_allclose(deps, p / 4.0, rtol=1e-5)


def test_substrate_permittivity_differentiates_the_metal_substrate_layer(side):
    # holding the mode fixed, p_MS grows as the substrate permittivity squared
    def f(eps):
        materials = {"sub": gfem.MaterialField(eps_r=eps), "air": gfem.Material()}
        modes = dataclasses.replace(side, materials=materials)
        return gfem.interface_participation(modes, "top_sub", T, 5.0, kind="MS")

    p = f(jnp.asarray(EPS)).participation[0]
    g = jax.grad(lambda e: f(e).participation[0])(jnp.asarray(EPS))
    np.testing.assert_allclose(g, 2 * p / EPS, rtol=1e-10)


def test_through_the_eigenmode_solve_the_gradient_raises(tmp_path):
    # mode fields are not differentiable in this version: never a silent zero
    path = tmp_path / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, 2.0, H, (8, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")

    def f(eps):
        model = gfem.Model(
            mesh,
            materials={"dielectric": gfem.MaterialField(eps_r=eps)},
            boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        )
        modes = gfem.solve(model, gfem.Eigenmode(target=F1 / 2))
        return gfem.interface_participation(
            modes, "top", T, 5.0, kind="MS"
        ).participation[0]

    with pytest.raises(gfem.NonDifferentiableError):
        jax.grad(f)(jnp.asarray(4.0))


def test_runs_under_jit(side):
    def f(modes):
        return gfem.interface_participation(
            modes, "top_sub", T, 5.0, kind="MS"
        ).participation

    np.testing.assert_allclose(
        jax.jit(f)(side),
        gfem.interface_participation(side, "top_sub", T, 5.0, kind="MS").participation,
        rtol=1e-13,
    )


def test_eigenmode_result_carries_its_materials(side):
    assert set(side.materials) == {"sub", "air"}
    assert side.materials["sub"].eps_r == EPS


def test_measures_a_result_reloaded_from_its_bundle(tmp_path):
    path = tmp_path / "side.msh"
    sm.write_msh22(path, _side_by_side())
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    n = len(mesh.elements("sub"))
    model = gfem.Model(
        mesh,
        materials={
            "sub": gfem.MaterialField(eps_r=np.full(n, EPS)),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC(s) for s in ("bottom", "top_sub", "top_air")],
    )
    job = gfem.Job(model, gfem.Eigenmode(target=F1 * K0 / KX))
    result = job.run(out=tmp_path / "out")
    loaded = gfem.load_result(tmp_path / "out")
    assert isinstance(loaded.materials["sub"], gfem.MaterialField)
    for kind in ("MS", "MA"):
        a = gfem.interface_participation(
            result, "top_sub", T, 5.0, kind=kind, tan_delta=1e-3
        )
        b = gfem.interface_participation(
            loaded, "top_sub", T, 5.0, kind=kind, tan_delta=1e-3
        )
        np.testing.assert_array_equal(a.participation, b.participation)
        np.testing.assert_array_equal(a.inverse_q, b.inverse_q)
    np.testing.assert_array_equal(
        loaded.e_field(0).at([[5e-3, 0.5e-3, 0.5e-3]]),
        result.e_field(0).at([[5e-3, 0.5e-3, 0.5e-3]]),
    )


def test_interface_participation_round_trips_through_the_codec(side, tmp_path):
    from gfem import _codec

    p = gfem.interface_participation(
        side, "top_air", T, 10.0, kind="MA", tan_delta=2e-3
    )
    context = _codec.Context(tmp_path)
    encoder = _codec.Encoder(context)
    doc = encoder.encode(p)
    back = _codec.Decoder(context, encoder.arrays).decode(doc)
    assert isinstance(back, gfem.InterfaceParticipation)
    np.testing.assert_array_equal(back.participation, p.participation)
    np.testing.assert_array_equal(back.inverse_q, p.inverse_q)
