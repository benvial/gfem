r"""The modal Raman gain converges to closed forms.

A guide filled with a constant index $n$ and PEC walls has TE modes
$E \propto \sin(\pi x/a)$ with $n_\mathrm{eff} = \sqrt{n^2 - (\pi/(a k_0))^2}$,
and at 1 W the gain of a pump and a Stokes wave of constant weight $W$ is

.. math::

    g = c_\mathrm{int}\,\frac{G W n^2}{n_{e,p} n_{e,s}\,a b},

with $c_\mathrm{int} = 3/2$ for two TE10 waves ($\int\sin^4$) and $1$ for a TE10
and a TE01 wave ($\int\sin^2\sin^2$ over $ab/4$ each). A 220 nm silicon slab
in silica along [110] has no elementary closed form: its literal is the
dispersion relation solved with quadrature of the fields. The element order
refines the mesh's solution towards each; the gradient through the mode field
and the permittivity is held to fourth-order differences of exact solves.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import raman_cases as rc
from finite_differences import assert_gradient_matches, unit_directions
from slab2d import write_slab

import gfem

ORDERS = (2, 3, 4, 5)
N_SILICA, G_SILICA = 1.5, 6e-14
N_SI, G_SI = 3.476, 2e-10
F_SI_PUMP = rc.F_S + 15.6e12
F_SILICA_PUMP = rc.F_S + 13.2e12
S = 2**-0.5
#: crystal axes (rows, in the frame x = propagation [110], y = [001], z = [1-10])
AXES_110 = np.array([[S, 0, S], [S, 0, -S], [0, 1, 0]])
#: crystal axes turned 45 degrees about z
AXES_45 = np.array([[S, S, 0], [-S, S, 0], [0, 0, 1]])
#: the slab's closed-form g w (1/W), TE0 along [110], 7 digits
SLAB_GW = 9.551756e-4


def _converges(errors, floor):
    """Each order 5x below the last, the last under ``floor``."""
    assert all(e2 < e1 / 5 for e1, e2 in itertools.pairwise(errors)), errors
    assert errors[-1] < floor, errors


def _gains(model, grid, order, modes=(1, 1)):
    options = gfem.SolverOptions(order=order)
    results = [
        gfem.solve(model, gfem.PortMode("p1", f, mode=m), options=options)
        for f, m in zip(grid, modes, strict=True)
    ]
    return np.asarray(gfem.raman_gain(model, results).gain)


def _model(tmp_path, a, b, n, **material):
    return rc.guide_model(
        tmp_path / "g.msh", a, b, gfem.Material(eps_r=n**2, **material), nx=6, ny=4
    )


def test_te10_pump_and_stokes_in_a_filled_guide(tmp_path):
    # measured relative errors 6.8e-5, 6.3e-8, 6.1e-11, 2.3e-12; the Manley-Rowe
    # partner is the same sum times f_p / f_s (round-off 1e-15, gate 1e-13)
    a, b = 2.0, 1.2
    spectrum = rc.flat_spectrum(G_SILICA, symmetry="isotropic", perpendicular=3.6e-15)
    model = _model(tmp_path, a, b, N_SILICA, raman_spectrum=spectrum)
    n_p, n_s = rc.te_index(F_SILICA_PUMP, N_SILICA, a), rc.te_index(rc.F_S, N_SILICA, a)
    closed = 1.5 * G_SILICA * N_SILICA**2 / (n_p * n_s * a * b * 1e-12)
    errors = []
    for order in ORDERS:
        g = _gains(model, [F_SILICA_PUMP, rc.F_S], order)
        errors.append(abs(g[1, 0] / closed - 1))
        assert g[0, 1] == pytest.approx(-F_SILICA_PUMP / rc.F_S * g[1, 0], rel=1e-13)
    _converges(errors, 1e-10)


def test_a_te10_pump_and_a_te01_stokes_wave_see_the_perpendicular_gain(tmp_path):
    # measured relative errors 4.2e-3, 1.4e-5, 3.1e-8, 4.7e-11
    a, b = 2.0, 1.2
    g_perp = 3.6e-15
    spectrum = rc.flat_spectrum(G_SILICA, symmetry="isotropic", perpendicular=g_perp)
    model = _model(tmp_path, a, b, N_SILICA, raman_spectrum=spectrum)
    n_p = rc.te_index(F_SILICA_PUMP, N_SILICA, a)
    n_s = rc.te_index(rc.F_S, N_SILICA, b)
    closed = g_perp * N_SILICA**2 / (n_p * n_s * a * b * 1e-12)
    errors = []
    for order in ORDERS:
        g = _gains(model, [F_SILICA_PUMP, rc.F_S], order, modes=(1, 2))
        errors.append(abs(g[1, 0] / closed - 1))
    _converges(errors, 1e-9)


def test_silicon_filled_guide_selection_rule(tmp_path):
    # TE10 along y: [010] co-polarized has weight 0 with the crystal axes
    # aligned, 1 along <110> (turned 45 degrees). Measured errors of the turned
    # guide 9.2e-5, 1.7e-7, 1.8e-10, 2.3e-12; of the aligned one (a zero, scaled
    # by the turned value) 1.6e-4, 8.7e-8, 2.4e-10, 5.6e-14
    a, b = 1.0, 0.5
    spectrum = rc.flat_spectrum(G_SI, symmetry="cubic")
    n_p, n_s = rc.te_index(F_SI_PUMP, N_SI, a), rc.te_index(rc.F_S, N_SI, a)
    full = 1.5 * G_SI * N_SI**2 / (n_p * n_s * a * b * 1e-12)
    assert full == pytest.approx(6.290264241e2, rel=1e-10)
    turned = _model(tmp_path, a, b, N_SI, raman_spectrum=spectrum, axes=AXES_45)
    aligned = _model(tmp_path, a, b, N_SI, raman_spectrum=spectrum, axes=np.eye(3))
    turned_err, aligned_err = [], []
    for order in ORDERS:
        turned_err.append(
            abs(_gains(turned, [F_SI_PUMP, rc.F_S], order)[1, 0] / full - 1)
        )
        aligned_err.append(
            abs(_gains(aligned, [F_SI_PUMP, rc.F_S], order)[1, 0] / full)
        )
    _converges(turned_err, 1e-10)
    _converges(aligned_err, 1e-12)


def test_crossed_fields_swap_the_selection_rule(tmp_path):
    # TE10 pump along y, TE01 Stokes along x: weight 1 aligned, 0 turned
    a, b = 1.0, 0.6
    spectrum = rc.flat_spectrum(G_SI, symmetry="cubic")
    n_p, n_s = rc.te_index(F_SI_PUMP, N_SI, a), rc.te_index(rc.F_S, N_SI, b)
    full = G_SI * N_SI**2 / (n_p * n_s * a * b * 1e-12)
    for axes, weight, gate in ((np.eye(3), 1.0, 1e-9), (AXES_45, 0.0, 1e-9)):
        model = _model(tmp_path, a, b, N_SI, raman_spectrum=spectrum, axes=axes)
        g = _gains(model, [F_SI_PUMP, rc.F_S], 5, modes=(1, 2))
        # order 5 meshes errors of the cross-polarized pair: measured below 5e-11
        assert g[1, 0] / full == pytest.approx(weight, abs=gate)


def test_the_slab_gain_along_110_converges_and_vanishes_on_the_axes(tmp_path):
    # TE0 (Ez line port) at order 5 has the 7 digits of SLAB_GW (measured
    # 3.1e-8, the literal's round-off); aligned axes make E_z the [001] axis
    path = write_slab(
        tmp_path / "s.msh", thick=0.22, clear=3.0, h_core=0.0275, h_clad=0.05
    )
    spectrum = rc.flat_spectrum(G_SI, symmetry="cubic")
    values = {}
    for name, axes in (("110", AXES_110), ("100", np.eye(3))):
        model = gfem.Model(
            gfem.Mesh.from_gmsh(path, length_unit="um"),
            materials={
                "core": gfem.Material(
                    eps_r=N_SI**2, axes=axes, raman_spectrum=spectrum
                ),
                "clad": gfem.Material(eps_r=1.444**2),
            },
            ports=[gfem.WavePort("p1", "port1", pec_edges=("outer",))],
        )
        grid = [F_SI_PUMP, rc.F_S]
        results = [
            gfem.solve(
                model,
                gfem.PortMode("p1", f, polarization="Ez"),
                options=gfem.SolverOptions(order=5),
            )
            for f in grid
        ]
        values[name] = np.asarray(gfem.raman_gain(model, results).gain)
    assert values["110"][1, 0] == pytest.approx(SLAB_GW, rel=1e-7)
    assert values["100"][1, 0] == 0.0


def test_the_gain_differentiates_through_the_mode_field_and_the_permittivity(tmp_path):
    # The spectrum lives on a constant Material, which cannot be traced, so the
    # modes are solved on a model whose region is a MaterialField with a traced
    # eps_r and the overlap reads the constant model of the same geometry (the
    # route for a design). The derivative is that of the mode field (one
    # adjoint per row) at the index of the constant model. Step 1e-2 on eps_r
    # of 12: fourth-order truncation of this smooth response about 1e-9
    # relative, round-off 1e-15 / h = 1e-13, both far under the 1e-6 gate.
    a, b = 1.0, 0.5
    spectrum = rc.flat_spectrum(G_SI, symmetry="cubic")
    mesh = gfem.Mesh.from_gmsh(
        rc.guide_mesh(tmp_path / "g.msh", a, b, nx=4, ny=3), length_unit="um"
    )
    options = gfem.SolverOptions(order=3)

    def build(material):
        return gfem.Model(
            mesh,
            materials={"air": material},
            boundaries=[gfem.PEC("walls")],
            ports=[gfem.WavePort("p1", "port1")],
        )

    constant = build(
        gfem.Material(eps_r=N_SI**2, axes=AXES_45, raman_spectrum=spectrum)
    )

    def gain(p):
        solved = build(gfem.MaterialField(eps_r=p[0], region="air"))
        modes = [
            gfem.solve(solved, gfem.PortMode("p1", f), options=options)
            for f in (F_SI_PUMP, rc.F_S)
        ]
        return gfem.raman_gain(constant, modes).gain[1, 0]

    p0 = jnp.array([N_SI**2])
    g0 = float(gain(p0))
    rng = np.random.default_rng(0)
    assert_gradient_matches(
        lambda p: gain(p) / g0, p0, unit_directions(rng, p0.shape, 1), step=1e-2
    )
    assert jax.grad(gain)(p0).shape == (1,)
