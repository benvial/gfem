r"""Wave ports that keep a mode below cutoff, on a 2D line port.

The closed forms: a PEC box of width :math:`A` filled with permittivity
:math:`\varepsilon` carries :math:`E_z \propto \sin(m\pi y/A)` with
:math:`\gamma_m^2 = (m\pi/A)^2 - \varepsilon k_0^2`, so a mode below cutoff has
:math:`\gamma = \alpha > 0` and :math:`n_\mathrm{eff} = -i\alpha/k_0`. The block
is a width step in that box (the lead core widens to the section core), whose
kept modes couple, solved at order 2 on a structured triangle mesh.
"""

import cmath
import math
import warnings

import numpy as np
import pytest
import structured_meshes as sm
from cascade_cases import box_mesh, lines
from scipy.constants import c

import gfem
from gfem._sweep.port_mode import _branch_gamma
from gfem.errors import PortModeError

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", threads=1)
EPS_CLAD, EPS_CORE = 1.444**2, 2.0**2


# --- the uniform box: closed forms ------------------------------------------------

WIDTH = 1.0  # um


@pytest.fixture(scope="module")
def uniform(tmp_path_factory):
    path = tmp_path_factory.mktemp("uniform") / "box.msh"
    mesh = box_mesh(
        path, lines([0.0, 1.0], 0.1), lines([0.0, WIDTH], 0.025), lambda *_: "fill"
    )
    return mesh


def uniform_model(mesh, *, evanescent, mode=1):
    return gfem.Model(
        mesh,
        materials={"fill": gfem.Material(eps_r=1.0)},
        boundaries=[gfem.PEC("side")],
        ports=[
            gfem.WavePort("p", "left", mode=mode, allow_evanescent=evanescent),
            gfem.WavePort("q", "right"),
        ],
    )


def gamma_exact(m, f, eps=1.0):
    k0 = 2 * math.pi * f / c * 1e-6  # per um
    return cmath.sqrt((m * math.pi / WIDTH) ** 2 - eps * k0**2) * 1e6  # per m


F_ABOVE = 1.2 * c / (2 * WIDTH * 1e-6)  # mode 1 propagates, mode 2 below cutoff
FC2 = c / (WIDTH * 1e-6)  # cutoff of mode 2 (eps 1)


def port_mode(mesh, f, mode, *, evanescent=True):
    problem = gfem.PortMode("p", f, mode=mode, polarization="Ez")
    return gfem.solve(
        uniform_model(mesh, evanescent=evanescent, mode=mode), problem, options=OPTIONS
    )


def test_an_opted_in_port_solves_modes_beyond_the_propagating_count(uniform):
    f = [0.6 * FC2]  # only mode 1 propagates (cutoff of mode 1: FC2 / 2)
    for m in (2, 3):
        r = port_mode(uniform, f, m)
        assert r.allow_evanescent
        assert complex(np.asarray(r.gamma)[0]).real > 0


@pytest.mark.parametrize("m", [2, 3])
def test_gamma_is_decaying_below_cutoff_and_n_eff_is_minus_i_alpha_over_k0(uniform, m):
    f = 0.6 * FC2
    r = port_mode(uniform, [f], m)
    gamma = complex(np.asarray(r.gamma)[0])
    n_eff = complex(np.asarray(r.n_eff)[0])
    k0 = 2 * math.pi * f / c
    exact = gamma_exact(m, f)
    assert exact.real > 0
    # order 2 on h = 0.025 um: the discretization error of gamma, measured
    # 6.6e-7 (m = 2) and 2.5e-6 (m = 3) relative; gate 1e-5, four times the larger
    assert abs(gamma - exact) <= 1e-5 * abs(exact)
    assert gamma.real > 0
    # n_eff = gamma / (i k0) is a division by a constant: 8 eps; it is purely
    # imaginary as gamma is real (the real part carries round-off only)
    assert n_eff == pytest.approx(gamma / (1j * k0), rel=8 * np.finfo(float).eps)
    assert abs(n_eff.real) <= 8 * np.finfo(float).eps * abs(n_eff)
    assert n_eff.imag == pytest.approx(-gamma.real / k0, rel=8 * np.finfo(float).eps)


def test_the_first_ranks_equal_those_of_a_sweep_without_the_opt_in(uniform):
    f = [1.1 * FC2, 1.3 * FC2]  # modes 1 and 2 propagate
    for m in (1, 2):
        kept = np.asarray(port_mode(uniform, f, m).gamma)
        plain = np.asarray(port_mode(uniform, f, m, evanescent=False).gamma)
        # the same eigenpairs from the same factorizations, ranked by the same
        # key: measured identical; 1e-12 leaves room for a threaded factor's
        # summation order
        np.testing.assert_allclose(kept, plain, rtol=1e-12, atol=0)


def test_a_mode_that_crosses_cutoff_has_nan_group_index_only_below_it(uniform):
    f = np.array([0.8, 0.95, 1.05, 1.2]) * FC2
    r = port_mode(uniform, f, 2)
    n_g = np.asarray(r.n_g)
    gamma = np.asarray(r.gamma)
    below = f < FC2
    assert np.isnan(n_g[below]).all()
    assert np.isfinite(n_g[~below]).all()
    assert (gamma[below].real > 0).all()
    assert (gamma[~below].imag > 0).all()
    assert np.isnan(np.asarray(r.beta_2)[below]).all()
    assert np.isfinite(np.asarray(r.beta_2)[~below]).all()


def test_a_port_that_has_not_opted_in_names_the_remedy_below_cutoff(uniform):
    with pytest.raises(PortModeError, match=r"allow_evanescent=True"):
        port_mode(uniform, [0.6 * FC2], 2, evanescent=False)
    with pytest.raises(PortModeError, match=r"WavePort.kept_modes"):
        port_mode(uniform, [0.6 * FC2, 1.3 * FC2], 2, evanescent=False)


def test_a_kept_mode_beyond_the_modes_the_line_yields_is_not_a_cutoff(uniform):
    # the line has about 80 DOFs: a mode of rank 500 is not a matter of cutoff
    # and the port already has the opt-in, so the message must not suggest it
    with pytest.raises(PortModeError, match=r"yields only .*keep fewer") as err:
        port_mode(uniform, [F_ABOVE], 500)
    assert "allow_evanescent" not in str(err.value)
    assert "cutoff" not in str(err.value)


# --- the branch helper -------------------------------------------------------------


def test_the_branch_helper_is_the_quarter_turn_root_on_random_eigenvalues():
    rng = np.random.default_rng(0)
    lam = (rng.normal(size=4000) + 1j * rng.normal(size=4000)) * 10.0 ** rng.integers(
        -3, 8, size=4000
    )
    # round-off zeros of either sign on either part, and the negative real axis
    # (the ray lambda = -i t is the cut of the reference: either root, by rounding)
    lam = np.concatenate(
        [lam, [0.0, -0.0, 1e-30, 1e-30j, 1j * -0.0, complex(-0.0, 0.0), -4.0, 4.0]]
    )
    gamma = _branch_gamma(lam)
    # e^{i pi/4} sqrt(-i gamma^2) with gamma^2 = lam: the root in the half plane
    # Re(gamma e^{-i pi/4}) > 0; the principal root has a relative error of a few
    # epsilon, the rotation by e^{i pi/4} one more: 10 eps
    ref = np.exp(0.25j * np.pi) * np.sqrt(-1j * lam)
    rtol = 10 * np.finfo(float).eps
    np.testing.assert_allclose(gamma, ref, rtol=rtol, atol=0)
    np.testing.assert_allclose(gamma**2, lam, rtol=4 * rtol, atol=0)
    # a propagating mode keeps today's root bit for bit
    prop = np.array([-1.0, -4.0, -9.0 - 1e-3j, -2.0 + 0.5j])
    assert np.array_equal(_branch_gamma(prop), 1j * np.sqrt(-prop + 0j))


# --- refusals and the constructor --------------------------------------------------


def test_kept_modes_builds_the_section_ports():
    ports = gfem.WavePort.kept_modes(
        "s", "plane", 3, pec_edges=("side",), polarity={2: ((0.0, 1e-6), "+z")}
    )
    assert [p.name for p in ports] == ["s@1", "s@2", "s@3"]
    assert [p.mode for p in ports] == [1, 2, 3]
    assert all(p.allow_evanescent and p.surface == "plane" for p in ports)
    assert all(p.pec_edges == ("side",) for p in ports)
    assert [p.polarity is not None for p in ports] == [False, True, False]


@pytest.mark.parametrize(
    "call",
    [
        lambda: gfem.WavePort.kept_modes("s", "plane", 0),
        lambda: gfem.WavePort.kept_modes("s", "plane", 1.5),
        lambda: gfem.WavePort.kept_modes("s@a", "plane", 2),
        lambda: gfem.WavePort.kept_modes(
            "s", "plane", 2, polarity={3: ((0.0, 0.0), "+z")}
        ),
        lambda: gfem.WavePort.kept_modes(
            "s", "plane", 2, polarity={0: ((0.0, 0.0), "+z")}
        ),
        lambda: gfem.WavePort(
            "p",
            "left",
            allow_evanescent=True,
            voltage_path=((0.0, 0.0, 0.0), (0.0, 1e-6, 0.0)),
        ),
        lambda: gfem.WavePort("p", "left", allow_evanescent=True, offset=1e-6),
    ],
    ids=[
        "modes0",
        "modes-float",
        "at-sign",
        "key-high",
        "key-low",
        "voltage_path",
        "offset",
    ],
)
def test_the_value_refusals(call):
    with pytest.raises(ValueError):
        call()


def test_an_opted_in_port_on_a_3d_mesh_is_refused_before_the_solve(tmp_path):
    mesh = gfem.Mesh.from_gmsh(
        sm.write_msh22(tmp_path / "line.msh", sm.step_line(2)), length_unit="mm"
    )
    ports = [
        gfem.WavePort("p1", "port1", allow_evanescent=True),
        gfem.WavePort("p2", "port2"),
    ]
    with pytest.raises(NotImplementedError, match=r"2D"):
        gfem.Model(
            mesh,
            materials={"dielectric": gfem.Material(eps_r=2.0)},
            boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
            ports=ports,
        )


def test_a_port_that_has_not_opted_in_still_refuses_in_a_sweep_that_keeps_modes(
    uniform,
):
    model = gfem.Model(
        uniform,
        materials={"fill": gfem.Material(eps_r=1.0)},
        boundaries=[gfem.PEC("side")],
        ports=[
            gfem.WavePort("p1", "left", mode=1),
            gfem.WavePort("p2", "left", mode=2),
            gfem.WavePort("p3", "left", mode=3, allow_evanescent=True),
            gfem.WavePort("q", "right"),
        ],
    )
    # mode 2 is below cutoff at 0.6 FC2: its port did not opt in
    with pytest.raises(PortModeError, match=r"allow_evanescent=True"):
        gfem.solve(model, gfem.Driven(0.6 * FC2, polarization="Ez"), options=OPTIONS)


# --- a width step: the block of a mode cascade -------------------------------------

LAMBDA = 1.55  # um
F_STEP = c / (LAMBDA * 1e-6)
A, Y0, Y1, Y2 = 2.0, 0.75, 1.25, 1.65
N_KEPT = 6


@pytest.fixture(scope="module")
def step_block(tmp_path_factory):
    """Lead core [Y0, Y1] for x < 1, section core [Y0, Y2] after; N_KEPT per end."""
    h = 0.1
    xs = lines([0.0, 1.0, 2.0], h)
    ys = lines([0.0, Y0, Y1, Y2, A], h)

    def region(x, y):
        top = Y1 if x < 1.0 else Y2
        return "core" if Y0 < y < top else "clad"

    mesh = box_mesh(tmp_path_factory.mktemp("step") / "step.msh", xs, ys, region)
    ports = [
        *gfem.WavePort.kept_modes("a", "left", N_KEPT),
        *gfem.WavePort.kept_modes("b", "right", N_KEPT),
    ]
    model = gfem.Model(
        mesh,
        materials={
            "core": gfem.Material(eps_r=EPS_CORE),
            "clad": gfem.Material(eps_r=EPS_CLAD),
        },
        boundaries=[gfem.PEC("side")],
        ports=ports,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an opted-in port raises no warning
        result = gfem.solve(
            model, gfem.Driven(F_STEP, polarization="Ez"), options=OPTIONS
        )
    return model, result


def test_the_block_solves_modes_beyond_the_propagating_count(step_block):
    _, result = step_block
    for end in "ab":
        gamma = np.array(
            [
                np.asarray(result.port_modes[f"{end}@{m}"].gamma)[0]
                for m in range(1, N_KEPT + 1)
            ]
        )
        # propagating modes first (Im gamma > 0, Re gamma = 0), then the modes
        # below cutoff (Re gamma > 0), whose decay constant grows with the rank
        prop = gamma.imag > abs(gamma.real)
        assert prop[0]
        assert not prop[-1]
        assert np.all(np.diff(prop.astype(int)) <= 0)
        assert np.all(gamma[~prop].real > 0)
        assert np.all(np.diff(gamma[~prop].real) > 0)
        assert np.all(np.diff(gamma[prop].imag) < 0)


def test_the_block_is_reciprocal(step_block):
    _, result = step_block
    s = np.asarray(result.s.values)[0]
    asym = np.max(np.abs(s - s.T))
    # measured 2.5e-15 here (about 11 eps of a solve whose condition number is
    # modest); the same block at order 4 measured 1.6e-13, which bounds what a
    # platform's summation order can add: gate 3e-14, twelve times this
    # mesh's measurement and well below the order 4 one
    assert asym <= 3e-14
    assert np.isfinite(s).all()


def test_with_offsets_refuses_to_shift_an_opted_in_port(step_block):
    _, result = step_block
    with pytest.raises(ValueError, match=r"allow_evanescent"):
        result.with_offsets({"a@1": 0.1e-6})
    # the plane where it already is is no shift
    assert result.with_offsets({"a@1": 0.0}).offsets == result.offsets
