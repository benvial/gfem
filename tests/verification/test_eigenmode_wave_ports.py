r"""Resonances of guides closed by wave ports, against exact roots and the driven sweep.

The guides are those of `wave_port_cases` (full-cross-section layers, so
each mode family is an exact transmission line): a resonance's complex
frequency converges with the order to the line's root. Gates are pinned
at 3x to 30x the measured errors, written beside each.

- A (Q 5.5) and A4 (Q 869), one port: the Robin term at a complex
  wavenumber, orders 2 to 4.
- C, a port surface carrying TE10 and TE20: the TE20 family needs the
  surface's modal correction; each family is told by the field's parity
  about :math:`x = a/2`.
- B3, two port surfaces: the root, and the ports' coupling Q in the ratio of
  their shares of the outgoing power.
- L, a lossy anisotropic port section; D, a coax (TEM); A3, a root just
  above the port's cutoff; E, a circular guide on TM01, a port mode with
  :math:`E_n \ne 0`.
- The requested power share and its warning: the share of the port's loss
  that its requested modes carry is near 1 on a requested family's root, near
  0 on modes trapped against the port or carried by an unrequested family
  (warned, in two forms), and 0.87 on the hybrid guide, whose window 3 mm
  from the port puts an evanescent tail on the port (reported, not warned).
- The driven cross-check: eigen and driven solves share the mesh, the port
  modes and the operator, so the pole of a one-pole fit of the driven S
  (degree-6 numerator, 41 points over one half-width each side) matches the
  eigen solve's root to the port mode tolerance (1e-10 on both sides), not
  to the discretization error. On A4, C and H, whose hybrid, dispersive
  port mode has no exact value.
"""

import warnings

import numpy as np
import pytest
import wave_port_cases as w
from resonance import fit_resonance

import gfem
from gfem.errors import PortModeError, UnrequestedPortModeWarning


def direct(order, **options):
    return gfem.SolverOptions(order=order, linear_solver="direct", **options)


def eigen(model, target, n_modes, order, **options):
    problem = gfem.Eigenmode(target=target, n_modes=n_modes)
    return gfem.solve(model, problem, options=direct(order, **options))


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    directory = tmp_path_factory.mktemp("guides")
    return lambda name, build, *args, **kw: build(
        directory / f"{name}.msh", *args, **kw
    )


def lone_port(meshes, name, layers):
    mesh = meshes(name, w.ported_guide, layers, left_port=False)
    return w.ported_model(mesh, layers, [gfem.WavePort("p1", "port")])


# --- a lone port's Robin term -------------------------------------------------

# name: layers, root, target, (e_3, e_4, Im e_3) gates
LONE = {
    # measured e 1.2e-3 / 7.5e-6 / 3.3e-8, Im e_3 4.7e-5
    "A": (w.CASE_A, w.ROOT_A, 11.67e9, (1e-4, 3e-7, 1e-3)),
    # measured e 3.7e-4 / 3.6e-6 / 3.5e-8, Im e_3 4.5e-4
    "A4": (w.CASE_A4, w.ROOT_A4, 9.97e9, (1e-4, 3e-7, 1e-3)),
}


@pytest.mark.parametrize("name", list(LONE))
def test_a_lone_port_s_resonance_converges_with_order(meshes, name):
    layers, root, target, (gate_3, gate_4, gate_im) = LONE[name]
    model = lone_port(meshes, name, layers)
    errors, im = [], None
    for order in (2, 3, 4):
        result = eigen(model, target, 3 if name == "A" else 1, order)
        error, j = w.nearest_error(result, root)
        errors.append(error)
        if order == 3:
            im = w.im_error(result, j, root)
    e = errors
    assert e[0] > 10 * e[1] > 100 * e[2], e
    assert e[1] <= gate_3, e
    assert e[2] <= gate_4, e
    assert im <= gate_im, im


# --- a shared port surface (the modal correction at a complex wavenumber) --------


@pytest.fixture(scope="module")
def case_c(meshes):
    mesh = meshes("c", w.ported_guide, w.CASE_A3, left_port=False, b_mm=w.C_B_MM)
    return w.ported_model(mesh, w.CASE_A3, w.SHARED_PORTS)


def parity(result, j):
    """``E_y(3a/4) / E_y(a/4)`` of mode ``j`` in C's dielectric: +1 TE10, -1 TE20."""
    a, b = w.A_MM * 1e-3, w.C_B_MM * 1e-3
    points = [[a / 4, b / 2, 15e-3], [3 * a / 4, b / 2, 15e-3]]
    ey = np.asarray(result.e_field(j).at(points))[:, 1]
    return ey[1] / ey[0]


# family: root, target, parity, gates on e and Im e
FAMILIES = {
    # measured e 3.4e-5, Im 2.3e-4
    "te20": (w.ROOT_C20, 15.2e9, -1, (1e-4, 3e-3)),
    # from its own target: measured e 4.6e-6, Im 2.9e-5
    "te10": (w.ROOT_C10, 14.12e9, 1, (1e-4, 1e-3)),
}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_each_family_of_a_shared_port_surface_meets_its_root(case_c, family):
    root, target, sign, (gate, gate_im) = FAMILIES[family]
    result = eigen(case_c, target, 3, 3)
    error, j = w.nearest_error(result, root)
    assert error <= gate
    assert w.im_error(result, j, root) <= gate_im
    assert abs(parity(result, j) - sign) <= 1e-2


def test_a_shared_port_surface_s_te20_root_at_order_4(case_c):
    result = eigen(case_c, 15.2e9, 3, 4)
    error, j = w.nearest_error(result, w.ROOT_C20)
    assert error <= 3e-6  # 5.6e-7
    assert w.im_error(result, j, w.ROOT_C20) <= 1e-4  # 4.7e-6


# --- two port surfaces, a lossy port section, TEM, near cutoff ------------------


def test_two_port_surfaces_meet_the_root_and_split_the_coupling_q(meshes):
    mesh = meshes("b3", w.ported_guide, w.CASE_B3, left_port=True)
    result = eigen(w.ported_model(mesh, w.CASE_B3, w.TWO_PORTS), 9.97e9, 3, 3)
    error, j = w.nearest_error(result, w.ROOT_B3)
    assert error <= 1e-4  # 6.9e-6
    q = {name: float(v[j]) for name, v in result.q_port.items()}
    # the shares are exact at the port planes; the coupling Q follows them to
    # O(1/Q) (Q 52): measured 2.4e-4 off
    share = w.SHARE_B3["left"] / w.SHARE_B3["right"]
    assert q["right"] / q["left"] == pytest.approx(share, rel=2e-2)


def test_a_lossy_anisotropic_port_section_meets_its_root(meshes):
    # Q 2.8: twelve modes, since higher-Q modes of other families lie nearer
    # the real target
    result = eigen(lone_port(meshes, "l", w.CASE_L), 11.67e9, 12, 3)
    error, j = w.nearest_error(result, w.ROOT_L)
    assert error <= 1e-4  # 7.9e-6
    assert w.im_error(result, j, w.ROOT_L) <= 1e-3  # 4.4e-5


def test_a_coax_s_tem_roots_meet_the_closed_form(meshes):
    mesh = meshes("d", w.coax_guide, w.CASE_A, n_r=4, n_theta=8)
    model = w.ported_model(
        mesh,
        w.CASE_A,
        [gfem.WavePort("p1", "port")],
        boundaries=(gfem.PEC("walls"), gfem.PMC("symmetry")),
    )
    result = eigen(model, 7.5e9, 2, 3)
    roots = w.coax_roots(4.0, 10.0, (0, 1))
    found = []
    for root in roots:
        error, j = w.nearest_error(result, root)
        assert error <= 1e-4  # 4.7e-6 and 1.7e-6
        found.append(w.ghz(result.omega[j]))
    # every root of the line has the same Im f: 8.7e-6 apart
    assert abs(found[0].imag - found[1].imag) <= 1e-4 * roots[0].imag


def test_a_root_just_above_the_port_s_cutoff_meets_the_exact_root(meshes):
    result = eigen(lone_port(meshes, "a3", w.CASE_A3), 7.03e9, 1, 3)
    error, _ = w.nearest_error(result, w.ROOT_A3)
    assert error <= 3e-6  # 3.5e-7


# The port absorbs TM01 only, so the guide's other families ring at high Q
# around the target, nearer it than the low-Q TM01 root (Q 3.7): the eigen
# solve returns one of them (13.65 GHz, Q 8000) instead of the root.
@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="the eigen solve's seeds miss a low-Q root among high-Q resonances "
    "of the families the port does not absorb",
)
def test_a_port_mode_with_a_normal_field_meets_the_line_root(meshes):
    # straight-sided mesh of the disk (16 x 16): the reference takes the
    # port's discrete cutoff, so only the 3D discretization error remains
    mesh = meshes("e", w.circular_guide, w.CASE_A, n=16)
    model = w.ported_model(mesh, w.CASE_A, [gfem.WavePort("p1", "port", mode=3)])
    f0 = 13.4e9
    modes = gfem.solve(model, gfem.PortMode("p1", f0), options=direct(3))
    k0 = 2 * np.pi * f0 / w.C0
    gamma = complex(np.asarray(modes.gamma).ravel()[0])
    kc = float(np.sqrt((gamma**2 + k0**2).real))
    root = w.line_root("TM", kc, [(4.0, 10.0), (1.0, 5.0)], 13.4 + 1.8j)
    result = eigen(model, f0, 1, 3)
    error, _ = w.nearest_error(result, root)
    assert error <= 3e-4  # 2.3e-6 when found


# --- the driven cross-check -----------------------------------------------------

TIGHT = 1e-10  # port mode tolerance of both solves


def driven_pole(model, pole, entry, order):
    """The pole (Hz) of a fit of S[entry, entry] over ``pole`` +- one half-width."""
    half = pole.imag
    f = np.linspace(pole.real - half, pole.real + half, 41)
    driven = gfem.Driven(tuple(f))
    s = np.asarray(
        gfem.solve(
            model, driven, options=direct(order, port_mode_tolerance=TIGHT)
        ).s.values
    )
    fit = fit_resonance(f, s[:, entry, entry], degree=6)
    return fit.pole if fit.pole.imag > 0 else fit.pole.conjugate()


def cross_check(model, target, n_modes, order, *, root=None, entry=0):
    result = eigen(model, target, n_modes, order, port_mode_tolerance=TIGHT)
    j = 0 if root is None else w.nearest_error(result, root)[1]
    pole = complex(result.omega[j]) / (2 * np.pi)
    fitted = driven_pole(model, pole, entry, order)
    return abs(fitted - pole) / abs(pole), abs(fitted.imag - pole.imag) / pole.imag


def test_a_lone_port_s_root_is_the_driven_pole(meshes):
    # A4 at order 3: measured 1.1e-13, Im 1.9e-10 (fit bias 2e-14)
    error, im = cross_check(lone_port(meshes, "a4", w.CASE_A4), 9.97e9, 1, 3)
    assert error <= 1e-12
    assert im <= 2e-9


def test_a_shared_port_surface_s_root_is_the_driven_pole(meshes):
    # mirrored in x and y, so families of other parities do not couple to
    # TE20: on the plain mesh two of them (15.173 and 15.237 GHz, Q 5e4 and
    # 200) lie within the window and bias the one-pole fit to 1.5e-5.
    # Measured 4.8e-9, Im 3.5e-7: the fit's own bias on the exact S
    mesh = meshes(
        "c_mirrored",
        w.ported_guide,
        w.CASE_A3,
        left_port=False,
        b_mm=w.C_B_MM,
        mirror=(True, True, False),
    )
    model = w.ported_model(mesh, w.CASE_A3, w.SHARED_PORTS)
    error, im = cross_check(model, 15.2e9, 3, 3, root=w.ROOT_C20, entry=1)
    assert error <= 5e-8
    assert im <= 5e-6


@pytest.fixture(scope="module")
def case_h(meshes):
    return w.hybrid_model(meshes("h", w.hybrid_guide))


def test_the_hybrid_guide_carries_one_propagating_mode_at_its_pole(case_h):
    # its second mode's cutoff is near 9.65 GHz, so up to there S11 of the
    # lossless guide has modulus one and the fit sees a single port mode
    with pytest.raises(PortModeError, match=r"carries 1 propagating mode"):
        gfem.solve(case_h, gfem.PortMode("p", 7.3e9, mode=2), options=direct(2))


def test_a_hybrid_dispersive_port_s_root_is_the_driven_pole(case_h):
    # order 2, pole 7.2964 + 0.2146i GHz: measured 5.1e-8, Im 1.5e-7
    error, im = cross_check(case_h, 7.3e9, 1, 2)
    assert error <= 3e-7
    assert im <= 1.5e-6


# --- the requested power share and its warning ----------------------------------

#: E's root (TM01, Q 3.7) from its line, 13.4 GHz target; the probe's value.
ROOT_E = 13.3815 + 1.8108j


def share_warnings(model, target, n_modes, order=3):
    """The result of an eigen solve and its `UnrequestedPortModeWarning` texts."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = eigen(model, target, n_modes, order)
    texts = [str(c.message) for c in caught if c.category is UnrequestedPortModeWarning]
    return result, texts


def test_a_requested_root_carries_the_port_s_loss_on_a_lone_te_line(meshes):
    # A: measured 1 - eta = -3.7e-3; L (lossy, anisotropic section): 3.9e-3
    for name, layers, root, target, n in (
        ("a", w.CASE_A, w.ROOT_A, 11.67e9, 3),
        ("l", w.CASE_L, w.ROOT_L, 11.67e9, 12),
    ):
        result, _ = share_warnings(lone_port(meshes, name, layers), target, n)
        _, j = w.nearest_error(result, root)
        eta = float(result.requested_power_share["port"][j])
        assert abs(1 - eta) <= 2e-2, (name, eta)


def test_the_tm_root_of_a_circular_guide_carries_most_of_the_port_s_loss(meshes):
    # E (Q 3.7): measured eta 0.962, the O(1/Q^2) floor reaching 0.04; the
    # search finds it among the guide's other families from six modes
    mesh = meshes("e", w.circular_guide, w.CASE_A, n=16)
    model = w.ported_model(mesh, w.CASE_A, [gfem.WavePort("p1", "port", mode=3)])
    result, _ = share_warnings(model, 13.4e9, 6)
    error, j = w.nearest_error(result, ROOT_E)
    assert error <= 1e-4
    assert float(result.requested_power_share["port"][j]) >= 0.9


def test_modes_trapped_against_a_port_are_warned_in_the_trapped_form(meshes):
    # A at three modes: the two modes beside the root are TE30 and TE21-like,
    # below cutoff at the port, whose Q (190) is the port's Robin scalar on
    # their evanescent tail. Measured eta 2e-8 and 2e-12
    result, texts = share_warnings(lone_port(meshes, "a", w.CASE_A), 11.67e9, 3)
    f = w.ghz(result.omega)
    root = int(np.argmin(np.abs(f - w.ROOT_A)))
    trapped = [j for j in range(3) if j != root]
    share = np.asarray(result.requested_power_share["port"])
    assert all(share[j] <= 1e-3 for j in trapped)
    assert len(texts) == 1
    assert "'port'" in texts[0]
    assert "trapped" in texts[0]
    assert "propagates" not in texts[0]
    assert all(f"{j} at" in texts[0] for j in trapped)
    assert f"{root} at" not in texts[0]


def test_a_surface_that_takes_little_of_a_mode_s_loss_is_not_named(meshes):
    # B3, two ports: the trapped modes (Q 420) lose their power at "port" (Q/q_s
    # 0.999); at "portl", far from the structure's side, their share is a
    # ratio of two tiny powers (Q/q_s 7.3e-4, 14x below the 0.01 gate)
    mesh = meshes("b3", w.ported_guide, w.CASE_B3, left_port=True)
    result, texts = share_warnings(
        w.ported_model(mesh, w.CASE_B3, w.TWO_PORTS), 9.97e9, 3
    )
    assert list(result.requested_power_share) == ["portl", "port"]
    _, j = w.nearest_error(result, w.ROOT_B3)
    for surface in ("portl", "port"):
        assert abs(1 - float(result.requested_power_share[surface][j])) <= 1e-3
    assert len(texts) == 1
    assert "'port'" in texts[0]
    assert "'portl'" not in texts[0]


def test_unrequested_modes_of_a_lossy_multimode_section_are_warned_in_the_open_form(
    meshes,
):
    # L's section carries seven unrequested ranks at the target: the modes at
    # 11.32, 11.27 and 11.13 GHz (Q 4 to 47, beside the root's 2.8) are warned
    # with the conditional wording, and the root, whenever the four modes
    # include it, is not named. Measured eta 1.3e-9 and below, Q/q_s 0.95 to 1.0
    result, texts = share_warnings(lone_port(meshes, "l", w.CASE_L), 11.67e9, 4)
    share = np.asarray(result.requested_power_share["port"])
    others = [j for j in range(4) if share[j] <= 1e-3]
    assert len(others) >= 3
    assert len(texts) == 1
    assert "propagates at that frequency" in texts[0]
    assert "mode=" in texts[0]
    assert all(f"{j} at" in texts[0] for j in others)
    for j in set(range(4)) - set(others):
        assert abs(1 - share[j]) <= 2e-2
        assert f"{j} at" not in texts[0].split("on wave port")[0]


def test_a_share_between_one_half_and_one_is_reported_and_not_warned(case_h):
    # H: a window 3 mm from the port puts an evanescent tail on it; measured
    # 0.870 at order 3, 0.869 at order 2 (this fixture)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnrequestedPortModeWarning)
        result = eigen(case_h, 7.3e9, 1, 2)
    eta = float(result.requested_power_share["port"][0])
    assert 0.5 < eta < 1
    assert 0.8 <= eta <= 0.92  # measured 0.869


def test_the_share_of_the_hybrid_guide_is_the_q_error_of_its_near_port(meshes):
    # moving the port from 3 mm to 15 mm behind the window raises the share
    # toward 1 and the Q to 20.2; Q / eta at 3 mm (19.6) is within 3 % of it.
    # The other two modes are trapped ones, warned
    q, eta = {}, {}
    for gap in (3.0, 15.0):
        mesh = meshes(f"h{gap:g}", w.hybrid_guide, gap_mm=gap)
        result, _ = share_warnings(w.hybrid_model(mesh), 7.3e9, 3)
        share = np.asarray(result.requested_power_share["port"])
        j = int(np.argmax(share))
        q[gap], eta[gap] = float(result.q[j]), float(share[j])
    assert 0.5 < eta[3.0] < eta[15.0] <= 1.01
    assert abs(q[3.0] / eta[3.0] - q[15.0]) <= 0.1 * q[15.0]


def test_a_hybrid_guide_s_single_mode_root_is_the_three_mode_root(case_h):
    # order 3: the guard seed's refinement wanders to a complex frequency
    # where the port term cannot follow its mode; losing that seed must not
    # lose the solve (it raised PortModeError with one mode)
    one = eigen(case_h, 7.3e9, 1, 3)
    three = eigen(case_h, 7.3e9, 3, 3)
    assert abs(one.omega[0] - three.omega[0]) <= 1e-7 * abs(three.omega[0])
