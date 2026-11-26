r"""The Kerr coefficient of a port mode converges to closed forms.

Two guides have a closed form for the overlap of the mode's fourth power: a
220 nm silicon slab in silica crossed by an Ez line port with perfectly
conducting ends 3 um from the core, and the filled WR-90 guide. The slab's
literals come from the dispersion relation solved at 30 digits (``mpmath``),
the field integrals of its fundamental TE mode done in closed form,

.. math::

    \gamma_K = \frac{k_0}{n_\mathrm{eff}^2}\,
    \frac{\int n^2 n_2\,u^4\,\mathrm{d}y}{\left(\int u^2\,\mathrm{d}y\right)^2},

and the WR-90's from :math:`\tfrac32 k_0 n_2/(ab)\,(n/n_\mathrm{eff})^2` with
the converged :math:`n_\mathrm{eff}`. The element order refines the mesh's
solution towards them. A silicon wire has no closed form, so its three-wave
tensor is held to the symmetries of the tensor; the gradient through the
mode's field is held to finite differences of exact solves; and the rows a
sweep reduces are held to exact solves at their frequency.
"""

import itertools
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as sc
import thermal_cases as tc
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import PORT_MODE_FREQUENCIES, port_mode_line, ramp, wave_mesh
from slab2d import write_slab

import gfem

ORDERS = (2, 3, 4, 5)
DATA = Path(__file__).parents[1] / "unit" / "data"

# --- the Ez slab ------------------------------------------------------------------

N_SI, N_OX = 3.476, 1.444
N2_SI, N2_OX = 6e-18, 2.6e-20
LAMBDA = 1.55e-6
#: gamma_K W (1/W) and n_eff of the slab's fundamental TE mode (mpmath, 30 digits)
SLAB_GAMMA, SLAB_N_EFF = 1.14457082596761e-4, 2.84778224344627

# --- the filled WR-90 -------------------------------------------------------------

#: WR-90 (m): the guide's section, not the extent of its points
A, B = 22.86e-3, 10.16e-3
N_FILL, N2_FILL = 1.5, 2.6e-20
F_GUIDE = 8e9
#: n_eff of the converged TE10 at 8 GHz, and gamma_K = 3/2 k0 n2/(ab) (n/n_eff)^2
GUIDE_N_EFF, GUIDE_GAMMA = 1.256258769165717, 4.013914625089216e-14


def _converges(errors, floor):
    """Each order 5x below the last (measured 90x), the last under ``floor``."""
    assert all(e2 < e1 / 5 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < floor, errors


@pytest.fixture(scope="module")
def slab_model(tmp_path_factory):
    path = write_slab(
        tmp_path_factory.mktemp("slab") / "slab.msh",
        thick=0.22,
        clear=3.0,
        h_core=0.0275,
        h_clad=0.05,
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={
            "core": gfem.Material(eps_r=N_SI**2, nonlinear_index=N2_SI),
            "clad": gfem.Material(eps_r=N_OX**2, nonlinear_index=N2_OX),
        },
        ports=[gfem.WavePort("p1", "port1", pec_edges=("outer",))],
    )


def test_the_slab_coefficient_converges_to_its_closed_form(slab_model):
    # measured relative errors of gamma 6.5e-5, 1.6e-7, 2.1e-10, 2.2e-12 at
    # orders 2-5 (the field's mesh error, quartic in it), of n_eff far below
    f = sc.c / LAMBDA
    gamma_errors, n_errors = [], []
    for p in ORDERS:
        mode = gfem.solve(
            slab_model,
            gfem.PortMode("p1", f, polarization="Ez"),
            options=gfem.SolverOptions(order=p),
        )
        kn = gfem.kerr_nonlinearity(slab_model, mode)
        gamma = complex(kn.gamma_nl[0])
        assert abs(gamma.imag) <= 1e-12 * abs(gamma.real)
        gamma_errors.append(abs(gamma.real - SLAB_GAMMA) / SLAB_GAMMA)
        n_errors.append(abs(complex(mode.n_eff[0]).real - SLAB_N_EFF) / SLAB_N_EFF)
    _converges(gamma_errors, 1e-10)
    assert n_errors[-1] < 1e-10, n_errors


@pytest.fixture(scope="module")
def guide_model():
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    material = gfem.Material(eps_r=N_FILL**2, nonlinear_index=N2_FILL)
    return gfem.Model(
        mesh,
        materials={"air": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def test_the_filled_wr90_coefficient_converges_to_its_closed_form(guide_model):
    # measured relative errors of gamma 6.7e-6, 1.1e-7, 6.9e-10, 1.4e-12
    errors, n_errors = [], []
    for p in ORDERS:
        mode = gfem.solve(
            guide_model,
            gfem.PortMode("p1", [F_GUIDE]),
            options=gfem.SolverOptions(order=p),
        )
        gamma = complex(gfem.kerr_nonlinearity(guide_model, mode).gamma_nl[0])
        errors.append(abs(gamma.real - GUIDE_GAMMA) / GUIDE_GAMMA)
        n_errors.append(abs(complex(mode.n_eff[0]).real - GUIDE_N_EFF) / GUIDE_N_EFF)
    _converges(errors, 1e-10)
    assert n_errors[-1] < 1e-10, n_errors


def test_the_filled_wr90_with_a_cubic_anisotropy_follows_its_factor(guide_model):
    # E is y alone in the continuum TE10, so the Kerr coefficient is the
    # isotropic one times rho + (1 - rho) sum_s (a_s . y)^4, which is
    # (1 + rho) / 2 for a frame turned 45 degrees about z; the mesh's x
    # component is the error, converging with the order like the isotropic one
    # (measured 3.1e-6, 1.3e-7, 7.9e-10, 1.5e-12 at orders 2-5)
    rho, c = 1.27, np.cos(np.pi / 4)
    axes = ((c, c, 0.0), (-c, c, 0.0), (0.0, 0.0, 1.0))
    material = gfem.Material(
        eps_r=N_FILL**2, nonlinear_index=N2_FILL, nonlinear_anisotropy=rho, axes=axes
    )
    model = gfem.Model(
        guide_model.mesh,
        materials={"air": material},
        boundaries=guide_model.boundaries,
        ports=guide_model.ports,
    )
    errors = []
    for p in ORDERS:
        mode = gfem.solve(
            model,
            gfem.PortMode("p1", [F_GUIDE]),
            options=gfem.SolverOptions(order=p),
        )
        gamma = complex(gfem.kerr_nonlinearity(model, mode).gamma_nl[0])
        errors.append(abs(gamma.real - (1 + rho) / 2 * GUIDE_GAMMA) / GUIDE_GAMMA)
    _converges(errors, 1e-10)


# --- a silicon wire ---------------------------------------------------------------


@pytest.fixture(scope="module")
def wire_tensor(tmp_path_factory):
    """Three waves at 1530, 1550 and 1570 nm of a 500 x 220 nm wire, order 2."""
    rects = [
        (-1.75, -1.0, 1.75, 1.8, "clad", None),
        (-0.25, 0.0, 0.25, 0.22, "core", 0.05),
    ]
    bottom = [(lambda p: abs(p[1] + 1.0) < 1e-9, "bottom")]
    mesh = tc.section(tmp_path_factory.mktemp("wire") / "s.msh", rects, 0.3, bottom)
    model = gfem.Model(
        mesh,
        materials={
            "clad": gfem.Material(eps_r=1.444**2, nonlinear_index=N2_OX),
            "core": gfem.Material(eps_r=3.4777**2, nonlinear_index=N2_SI),
        },
        ports=[gfem.WavePort("p", "front")],
    )
    f = sc.c / np.array([1.53e-6, 1.55e-6, 1.57e-6])
    mode = gfem.solve(model, gfem.PortMode("p", f), options=gfem.SolverOptions(order=2))
    return model, mode, gfem.kerr_coupling(model, mode)


def test_the_wire_tensor_is_real_and_obeys_its_identities(wire_tensor):
    model, mode, tensor = wire_tensor
    g = np.asarray(tensor.gamma_nl)
    scale = np.abs(g).max()
    # the lossless gauge makes E_t real and E_n imaginary: every product in the
    # integrand is real; observed Im = 0, the gate is round-off of 1e4 terms
    assert np.abs(g.imag).max() <= 1e-12 * scale
    # the three frequencies share a section: the value is the literature's
    # order (265 1/(W m) measured at order 2 with n2 = 6e-18 and 3.4777)
    assert 200 < g[0, 0, 0, 0].real < 350
    # swapping the two unconjugated factors; round-off of the einsum, 1e-16
    np.testing.assert_allclose(g, g.transpose(0, 2, 1, 3), rtol=0, atol=1e-13 * scale)
    # energy: entry abcd over omega_a is the conjugate of badc over omega_b
    omega = 2 * np.pi * np.asarray(mode.frequency)
    left = g / omega[:, None, None, None]
    right = np.conj(g.transpose(1, 0, 3, 2)) / omega[None, :, None, None]
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-13 * scale / omega.min())
    # the diagonal is the coefficient of each row
    kn = np.asarray(gfem.kerr_nonlinearity(model, mode).gamma_nl)
    diagonal = np.array([g[i, i, i, i] for i in range(3)])
    np.testing.assert_allclose(diagonal, kn, rtol=1e-12)


# --- the gradient ----------------------------------------------------------------


def test_the_coefficient_differentiates_through_the_lower_layer_of_a_line(tmp_path):
    # the 3D line's port p1 has a design "lo" on its face: eps_r and both Kerr
    # coefficients per element. The objective sums Re + Im of gamma_NL over
    # three exact rows, each over its value at the start so that it is O(1).
    mesh = wave_mesh(tmp_path)
    n = len(mesh.elements("lo"))
    ones = jnp.ones(n)
    n2_unit, beta_unit = 1e-20, 1e-12
    problem = gfem.PortMode("p1", PORT_MODE_FREQUENCIES)
    options = gfem.SolverOptions(port_mode_tolerance=1e-12)

    def gamma(eps, n2, beta):
        field = gfem.MaterialField(
            eps_r=eps,
            nonlinear_index=n2 * n2_unit,
            two_photon_absorption=beta * beta_unit,
        )
        model = port_mode_line(mesh, field)
        result = gfem.solve(model, problem, options=options)
        return gfem.kerr_nonlinearity(model, result).gamma_nl

    eps0, n20, beta0 = ramp(mesh, "lo", 3.5, 4.5), ramp(mesh, "lo", 1.0, 3.0), ones
    g0 = np.asarray(gamma(eps0, n20, beta0))
    assert np.all(g0.real > 0)
    assert np.all(g0.imag > 0)

    def objective(which):
        def f(p):
            args = [eps0, n20, beta0]
            args[which] = p
            g = gamma(*args) / g0
            return jnp.sum(g.real + g.imag)

        return f

    rng = np.random.default_rng(0)
    # eps_r: step 1e-2, the fourth-order truncation of the module guidance
    # (about 1e-8 relative). The two coefficients enter exactly linearly, so
    # their differences have no truncation: the step only has to lift the
    # difference over the round-off (1e-15 / h), unit directions on O(1) values
    for which, p, step in ((0, eps0, 1e-2), (1, n20, 0.5), (2, beta0, 0.5)):
        assert_gradient_matches(
            objective(which), p, unit_directions(rng, p.shape, 2), step=step
        )


# --- the rows of a sweep ----------------------------------------------------------


def test_a_reduced_row_matches_an_exact_solve_on_a_shape_changing_guide():
    # the half-filled guide's mode changes shape with frequency, so the reduced
    # rows' field is an interpolation held to the sweep's residual (1e-6 by
    # default), and gamma is quartic in the field
    mesh = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={
            "slab": gfem.Material(eps_r=4.0, nonlinear_index=N2_FILL),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    grid = np.linspace(10e9, 20e9, 50)
    swept = gfem.solve(model, gfem.PortMode("p1", grid))
    solved = np.asarray(swept.solved)
    assert 2 < solved.sum() < len(grid)
    gamma = np.asarray(gfem.kerr_nonlinearity(model, swept).gamma_nl)
    errors = []
    for i in (3, 17, 30, 44):
        assert not solved[i]
        exact = gfem.solve(model, gfem.PortMode("p1", grid[i]))
        ref = complex(gfem.kerr_nonlinearity(model, exact).gamma_nl[0])
        errors.append(abs(gamma[i] - ref) / abs(ref))
    # measured 1.0e-7 at most over these rows (swept residual 6.8e-7, 6 exact
    # solves of 50): the gate is 10x that, about the field residual
    assert max(errors) < 1e-6, errors
