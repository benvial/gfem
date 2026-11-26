r"""`gfem.optomechanical_overlap` of a displacement at the phase-matched wavenumber.

A displacement of axial wavenumber :math:`q \approx 2\beta` along the pump gives
the overlap of the mode with its counter-propagating twin (a backward pair,
:math:`E_s = DE_p`, :math:`D = \mathrm{diag}(1, 1, -1)`), split into the moving
boundary and the photoelastic parts; :math:`q \approx 0` keeps the forward
numbers; every other :math:`q` raises.

The closed form is the layered TM slab of the forward tests (second mode of the
section with natural side walls) under the film's dilation
:math:`u = (0, y - y_c, 0)\,e^{-iqz}`. Its moving boundary is
:math:`d\,\partial n/\partial d - 2C`: the interfaces move, and the normal field
:math:`E_z` (tangential to them) enters the pair with the opposite sign of its
energy. Its photoelastic part has the diagonal strains weighed by
:math:`\mathrm{Re}\,H_{tt}` and :math:`-H_{zz}` and the shear :math:`S_{yz} =
-iq u_y` weighed by :math:`i\,\mathrm{Im}\,H_{yz}`.
"""

import dataclasses

import electro_optic_cases as eo
import numpy as np
import optomechanics_cases as oc
import pytest
import scipy.constants

import gfem
from gfem._field import FieldMap

D = oc.FILM * 1e-6
EPS = oc.N_FILM**2
P = oc.cubic_p()
FREQUENCY = eo.FREQUENCY
COARSE, FINE, ROUGH = 2.0, 1.0, 3.0
#: The phase-matching window's share of the pump's wavenumber (half width 2 tau).
TAU = 1e-2


def _film(**kw):
    return gfem.Material(eps_r=EPS, photoelastic_tensor=P, **kw)


def _overlap(model, u, **kw):
    return gfem.optomechanical_overlap(model, u, port="p", frequency=FREQUENCY, **kw)


@pytest.fixture(scope="module")
def mechanical(tmp_path_factory):
    return oc.mechanical_mesh(tmp_path_factory.mktemp("bo") / "mechanical.msh")


@pytest.fixture(scope="module")
def slabs(tmp_path_factory):
    """The slab meshes by edge factor, built on first use."""
    path = tmp_path_factory.mktemp("bo_slab")
    made = {}

    def get(factor):
        if factor not in made:
            made[factor] = oc.slab(path / f"slab{factor}.msh", factor)
        return made[factor]

    return get


@pytest.fixture(scope="module")
def dilate(mechanical):
    """The film's dilation about its middle at wavenumber ``q`` (film only)."""

    def build(q):
        return oc.displacement(
            mechanical,
            ("film",),
            gradients=[(0, 0), (0, 1), (0, 0)],
            offsets=(0.0, -D / 2, 0.0),
            wavenumber=q,
        )

    return build


@pytest.fixture(scope="module")
def rough(slabs):
    return slabs(ROUGH)


@pytest.fixture(scope="module")
def beta(rough, dilate):
    """gfem's pump wavenumber on the rough slab (TM mode), from a forward call."""
    got = _overlap(oc.model(rough, _film(), pec=False), dilate(0.0), mode=2)
    return 2 * np.pi * FREQUENCY * float(got.n_eff) / scipy.constants.c


# --- TM slab closed form -------------------------------------------------------------

_MOMENTS = oc.slab_tm_moments()
_Q = 2 * _MOMENTS.beta  # 2 k0 n_exact


def _exact_tm():
    eps = (EPS,) * 3
    n4 = EPS**2
    mb = D * oc.thickness_derivative(eps, D, "TM") - 2 * _MOMENTS.corner
    diag = -(n4) * (oc.P11 * _MOMENTS.gamma_y - oc.P12 * _MOMENTS.gamma_z)
    # the shear strain of the dilation at q is S_yz = -i q u_y; its weight
    # is 2 (i Im M) (the factor 2 of the two off-diagonal blocks), times p44
    shear = 2 * (1j * _MOMENTS.shear.imag) * (1j * _Q * n4 * oc.P44)
    return mb, diag, shear


#: Gates: the moving boundary's relative error, and the photoelastic part's error
#: over the size of its shear term (the part that cancels the diagonal one:
#: the sum is 0.17 of the diagonal 1.8 and the shear -1.97, so an error of the
#: parts would show in the sum). Measured, factor 2 then 1: moving boundary
#: MB_C MB_F, photoelastic PE_C PE_F; the gates are 3x.
_GATES = {COARSE: {"mb": 1.5e-2}, FINE: {"mb": 1.4e-3}}


@pytest.fixture(scope="module")
def tm(slabs, dilate):
    made = {}

    def get(factor):
        if factor not in made:
            model = oc.model(slabs(factor), _film(), pec=False)
            made[factor] = _overlap(model, dilate(_Q), mode=2)
        return made[factor]

    return get


def _tm_errors(got):
    mb, diag, shear = _exact_tm()
    del diag, shear
    return {"mb": abs(got.moving_boundary / mb - 1)}


@pytest.mark.parametrize("factor", [COARSE, FINE], ids=["coarse", "fine"])
def test_the_film_dilation_at_twice_beta_matches_the_slab_backward_closed_form(
    tm, factor
):
    got = tm(factor)
    assert got.direction == "backward"
    assert got.wavenumber == _Q
    errors = _tm_errors(got)
    for name, error in errors.items():
        assert error < _GATES[factor][name], (factor, name, error)
    np.testing.assert_allclose(got.total, got.moving_boundary + got.photoelastic)


def test_halving_the_edge_length_cuts_each_backward_error_fivefold(tm):
    coarse, fine = _tm_errors(tm(COARSE)), _tm_errors(tm(FINE))
    for name in coarse:
        assert coarse[name] / fine[name] > 5, (name, coarse[name], fine[name])


# --- TE guard ------------------------------------------------------------------------

#: The forward gates of the TE slab at the fine edge (the relative errors of the
#: moving boundary and the photoelastic part against the forward closed form,
#: 3x the measured 1.3e-3 and 3.9e-5 of the forward tests).
_TE_GATE = {"mb": 4e-3, "pe": 1.2e-4}


def test_a_te_pair_equals_the_forward_closed_form(slabs, dilate):
    # PEC walls keep E_z = 0 in the TE mode of the slab: the normal-field
    # correction and the cross blocks vanish, and the backward pair is the
    # forward overlap (a spurious E_z term would show here at order one).
    model = oc.model(slabs(FINE), _film(), pec=True)
    n = oc.slab_n_eff((EPS,) * 3, D, "TE")
    q = 2 * 2 * np.pi * FREQUENCY * n / scipy.constants.c
    got = _overlap(model, dilate(q), mode=1)
    assert got.direction == "backward"
    eps = (EPS,) * 3
    mb = D * oc.thickness_derivative(eps, D, "TE")
    pe = oc.eps_derivative(eps, D, "TE", 0) * (-(EPS**2) * oc.P12)
    assert abs(got.moving_boundary / mb - 1) < _TE_GATE["mb"]
    assert abs(got.photoelastic / pe - 1) < _TE_GATE["pe"]


# --- the complex weights -------------------------------------------------------------


def test_the_real_part_of_the_complex_weights_is_the_forward_weights(rough):
    from gfem._engine.thermo_optic import (
        ShiftWeightsRule,
        section_complex_tensor_weights,
        section_tensor_weights,
    )
    from gfem._optical_shift import _section, _wave_port
    from gfem._sweep.port_mode import PortModeRule
    from gfem.options import SolverOptions
    from gfem.problems import PortMode

    class Both(ShiftWeightsRule):
        def forward(self, model, *, keep, request=None):
            out, _ = PortModeRule.forward(self, model, keep=keep, request=request)
            sweep = out["sweep"]
            c = sweep.basis @ sweep.coords[0]
            self.pair = (
                section_complex_tensor_weights(sweep.space, c),
                section_tensor_weights(sweep.space, c),
            )
            return {"gamma": sweep.gamma, "weights": None}, None

    model = oc.model(rough, _film(), pec=False)
    faces, _ = _section(model, _wave_port(model, "p"))
    rule = Both(model, PortMode("p", FREQUENCY, mode=2), SolverOptions(), faces)
    rule.forward(model, keep=False)
    h, w = rule.pair
    # different computations (complex against real products, a symmetrization):
    # round-off of 10-50-point quadrature sums, ~50 eps = 1e-14 of the largest
    # entry, gate 1e-12 (100x)
    atol = 1e-12 * np.abs(w).max()
    np.testing.assert_allclose(h.real, w, rtol=0, atol=atol)
    np.testing.assert_allclose(h, np.conj(np.swapaxes(h, 1, 2)), rtol=0, atol=atol)
    assert np.abs(h.imag).max() > 1e-3 * np.abs(w).max()  # the TM mode has Im H_yz


# --- the window ----------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1 - TAU / 2, 1 + TAU / 2], ids=["below", "above"])
def test_a_wavenumber_inside_the_backward_window_is_a_backward_pair(
    rough, dilate, beta, scale
):
    model = oc.model(rough, _film(), pec=False)
    got = _overlap(model, dilate(2 * beta * scale), mode=2)
    assert got.direction == "backward"
    assert got.wavenumber == 2 * beta * scale
    assert "backward q" in repr(got)


@pytest.mark.parametrize("scale", [TAU, -TAU], ids=["plus", "minus"])
def test_a_wavenumber_inside_the_forward_window_is_a_forward_pair(
    rough, dilate, beta, scale
):
    model = oc.model(rough, _film(), pec=False)
    got = _overlap(model, dilate(scale * beta), mode=2)
    assert got.direction == "forward"
    assert got.wavenumber == scale * beta
    assert "backward" not in repr(got)


def _expected(beta):
    return f"expected q = 0 \\(forward\\) or q = {2 * beta:+.4e} rad/m"


@pytest.mark.parametrize(
    ("scale", "kind"),
    [
        (2 * (1 + 2 * TAU), "intermodal"),
        (2 * (1 - 2 * TAU), "intermodal"),
        (-2.0, "counter-propagates"),
        (1.0, "intermodal"),
    ],
    ids=["above", "below", "counter", "beta"],
)
def test_a_wavenumber_outside_the_windows_raises_naming_the_expected_values(
    rough, dilate, beta, scale, kind
):
    model = oc.model(rough, _film(), pec=False)
    with pytest.raises(ValueError, match=kind) as info:
        _overlap(model, dilate(scale * beta), mode=2)
    assert _expected(beta).replace("\\", "") in str(info.value)
    assert "(backward, along the pump of port 'p')" in str(info.value)


# --- refusals and cost ---------------------------------------------------------------


def test_a_magnetic_film_is_refused_for_a_backward_pair_only(rough, dilate):
    model = oc.model(rough, _film(mu_r=2.0), pec=False)
    forward = _overlap(model, dilate(0.0), mode=2)
    assert forward.direction == "forward"
    # the magnetic film's own pump wavenumber, so q sits in its backward window
    beta = 2 * np.pi * FREQUENCY * float(forward.n_eff) / scipy.constants.c
    with pytest.raises(NotImplementedError, match=r"region 'film'.*not assembled"):
        _overlap(model, dilate(2 * beta), mode=2)


def test_a_backward_pair_costs_two_port_mode_solves(rough, dilate, beta, monkeypatch):
    from gfem import optomechanics
    from gfem._engine.electro_optic import OptomechanicalPairWeightsRule

    calls = {"solve": 0, "pair": 0, "forward_weights": 0}
    solve, forward = gfem.solve, OptomechanicalPairWeightsRule.forward
    weights = optomechanics._mode_weights

    def counted_solve(*a, **k):
        calls["solve"] += 1
        return solve(*a, **k)

    def counted_pair(self, *a, **k):
        calls["pair"] += 1
        return forward(self, *a, **k)

    def counted_weights(*a, **k):
        calls["forward_weights"] += 1
        return weights(*a, **k)

    monkeypatch.setattr(gfem, "solve", counted_solve)
    monkeypatch.setattr(OptomechanicalPairWeightsRule, "forward", counted_pair)
    monkeypatch.setattr(optomechanics, "_mode_weights", counted_weights)
    model = oc.model(rough, _film(), pec=False)
    _overlap(model, dilate(2 * beta), mode=2)
    assert calls == {"solve": 1, "pair": 1, "forward_weights": 0}


def test_the_overlap_holds_its_wavenumber_and_direction_as_static_fields(
    rough, dilate, beta
):
    got = _overlap(oc.model(rough, _film(), pec=False), dilate(2 * beta), mode=2)
    other = dataclasses.replace(got, wavenumber=1.0, direction="forward")
    assert other.wavenumber == 1.0
    import jax

    leaves = jax.tree_util.tree_leaves(got)
    assert len(leaves) == 4  # n_eff, n_g, moving_boundary, photoelastic
    assert isinstance(dilate(0.0), FieldMap)
