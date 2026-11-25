r"""Resonant modes of a guide closed by a wave port.

A rectangular WR-90 guide with layers that fill its cross-section, closed
by a short at one end and a wave port at the other: each guide mode is an
exact transmission line along the guide, so its complex resonances solve a
scalar equation (the reference values below, residual 1e-13). The port's
TE10 mode leaves without reflection, so the resonance's Q is its radiation
through the port.

The port enters the eigen operator as a Robin term with
:math:`\theta = \gamma_0(k)` and a modal correction
:math:`U C(k) U^\mathsf{T}` (zero for a lone TE mode, not for the hybrid
guide's), both from the port mode sweep's reduced model at the complex
wavenumber.
"""

import dataclasses
import math
import warnings
from pathlib import Path

import jax
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from wave_port_cases import (
    ANTI_BOUND_A3,
    C0,
    CASE_A,
    CASE_A3,
    CASE_A4,
    CASE_B3,
    ROOT_A,
    ROOT_A3,
    ROOT_A4,
    ROOT_B3,
    ROOT_C10,
    ROOT_C20,
    SHARE_B3,
    SHARED_PORTS,
    TWO_PORTS,
    ghz,
    hybrid_guide,
    hybrid_model,
    nearest_error,
    ported_guide,
    ported_model,
)

import gfem
from gfem._eigen import rule as eigen_rule
from gfem._eigen.operator import split
from gfem._engine import cache
from gfem._engine.driven import driven_operator, operator_terms
from gfem._engine.eigen import eigen_ports
from gfem._engine.port_term import PortTerm, PortTermRefusalError
from gfem.errors import AntiBoundRootWarning

DATA = Path(__file__).parent / "data"
ORDER_3 = gfem.SolverOptions(order=3, linear_solver="direct")
LONE_PORT = (gfem.WavePort("p1", "port"),)


@pytest.fixture(scope="module")
def case_a(tmp_path_factory):
    mesh = ported_guide(tmp_path_factory.mktemp("a") / "a.msh", CASE_A, left_port=False)
    return ported_model(mesh, CASE_A, LONE_PORT)


@pytest.fixture(scope="module")
def case_a3(tmp_path_factory):
    mesh = ported_guide(
        tmp_path_factory.mktemp("a3") / "a3.msh", CASE_A3, left_port=False
    )
    return ported_model(mesh, CASE_A3, LONE_PORT)


# --- the solve ------------------------------------------------------------------


def test_lone_wave_port_resonance_matches_the_exact_root(case_a):
    # the two modes nearer the target than the port's resonance are of other
    # guide families (TE30/TE21-like), which the TE10 port only partly absorbs
    result = gfem.solve(
        case_a, gfem.Eigenmode(target=11.67e9, n_modes=3), options=ORDER_3
    )
    f = ghz(result.omega)
    j = int(np.argmin(np.abs(f - ROOT_A)))
    assert abs(f[j] - ROOT_A) / abs(ROOT_A) <= 1e-4
    assert result.error[j] <= 1e-12


def test_anti_bound_root_below_cutoff_is_dropped_with_a_warning(case_a3):
    # on the outgoing branch the guide has a real root at 4.619 GHz, below
    # the air guide's cutoff (6.557 GHz): its field grows into the guide, so
    # it is not a resonance; the search drops it and carries on
    with pytest.warns(AntiBoundRootWarning, match=r"4\.619"):
        result = gfem.solve(
            case_a3, gfem.Eigenmode(target=6.7e9, n_modes=8), options=ORDER_3
        )
    f = ghz(result.omega)
    assert len(f) == 8
    assert np.min(np.abs(f - ANTI_BOUND_A3)) > 1e-2
    assert np.min(np.abs(f - ROOT_A3)) / abs(ROOT_A3) <= 1e-4


# --- the operator ---------------------------------------------------------------


@pytest.fixture(scope="module")
def hybrid():
    # WR-90 with an eps_r = 6 block: a hybrid port mode (E_n != 0), whose
    # modal correction is not zero; the far end is a short
    mesh = gfem.Mesh.from_gmsh(DATA / "hybrid_waveguide.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"block": gfem.Material(eps_r=6.0), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls"), gfem.PEC("port2")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    target = 12e9
    rule = eigen_rule.EigenmodeRule(
        model, gfem.Eigenmode(target=target), gfem.SolverOptions(order=2)
    )
    op = driven_operator(model, 2)
    ports = eigen_ports(model, op, target=target, tolerance=1e-6)
    return model, rule, op, ports


def free_vector(eop, seed=0):
    rng = np.random.default_rng(seed)
    n = len(eop.free)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return np.where(eop.free, x, 0)


def test_port_piece_is_nonpolynomial_and_never_an_operator_term(hybrid):
    model, _, op, (port,) = hybrid
    piece = port.piece
    assert piece.kind == "port"
    with_port = dataclasses.replace(op, pieces=[*op.pieces, piece])
    k = 2 * math.pi * 12e9 / C0 * op.built.scale
    assert piece in split(with_port.pieces, op.built.scale, k).nonpolynomial
    names = {t.name for t in operator_terms(model, with_port, nodes=False)}
    assert piece.name not in names
    assert all(not n.startswith("waveport") for n in names)


def test_operator_derivative_matches_a_central_difference(hybrid):
    _, rule, op, ports = hybrid
    sigma = 2 * math.pi * 12e9 / C0 * op.built.scale
    eop = rule.operator(op, sigma, ports=ports)
    try:
        (low_rank,) = eop.low_rank
        k = sigma * (1.02 + 0.05j)
        _, c = low_rank.at(k)
        assert np.linalg.norm(c) > 1e-6  # the hybrid mode's correction is live
        x = free_vector(eop)
        h = 1e-5 * abs(k)
        difference = (eop.apply(k + h, x) - eop.apply(k - h, x)) / (2 * h)
        slope = eop.derivative(k, x)
        assert np.linalg.norm(slope - difference) <= 1e-8 * np.linalg.norm(slope)
    finally:
        eop.free_handle()


def test_shift_holds_the_low_rank_term_across_a_reshift(hybrid):
    _, rule, op, ports = hybrid
    sigma = 2 * math.pi * 12e9 / C0 * op.built.scale
    eop = rule.operator(op, sigma, ports=ports)
    try:
        x = free_vector(eop, 1)
        # the prepared T(sigma) solves the operator apply gives, port terms in
        back = eop.solve(eop.apply(sigma, x))
        assert np.linalg.norm(back - x) <= 1e-10 * np.linalg.norm(x)
        kx = eop.parts(x)[0]
        k = sigma * (1.01 + 0.03j)
        eop.reshift(k)
        back = eop.solve(eop.apply(k, x))
        assert np.linalg.norm(back - x) <= 1e-10 * np.linalg.norm(x)
        # K is the same whatever the shift
        assert np.linalg.norm(eop.parts(x)[0] - kx) <= 1e-10 * np.linalg.norm(kx)
    finally:
        eop.free_handle()


def test_k_holds_across_an_enrichment_of_the_port_basis(hybrid):
    # an enrichment between preparing T(s) and recovering K moves gamma_0(s),
    # but not the factorized operator K is read from
    model, rule, op, _ = hybrid
    target = 12e9
    ports = eigen_ports(model, op, target=target, tolerance=1e-2)  # own basis
    (port,) = ports
    sigma = 2 * math.pi * target / C0 * op.built.scale
    eop = rule.operator(op, sigma, ports=ports)
    try:
        s = sigma * (1.05 + 0.1j)
        eop.reshift(s)
        x = free_vector(eop, 2)
        kx = eop.parts(x)[0]
        before = port.term.at(s).gamma0
        k = sigma * (1.05 + 0.12j)
        pairs, _ = port.term._pairs(k)
        port.term._enrich(k, pairs, list(range(len(pairs))))
        assert abs(port.term.at(s).gamma0 - before) > 1e-10 * abs(before)  # 2e-8
        # 8e-10 when K subtracts gamma_0(s) recomputed after the enrichment
        assert np.linalg.norm(eop.parts(x)[0] - kx) <= 1e-14 * np.linalg.norm(kx)
    finally:
        eop.free_handle()


@pytest.fixture(scope="module")
def case_h(tmp_path_factory):
    return hybrid_model(hybrid_guide(tmp_path_factory.mktemp("h") / "h.msh"))


@pytest.mark.parametrize(
    ("case", "tolerance", "f_ghz"),
    [("hybrid", 1e-6, 0.2), ("hybrid", 1e-6, 3.0), ("case_h", 1e-3, 0.2)],
)
def test_a_root_far_below_cutoff_is_classified_without_touching_the_port_basis(
    request, case, tolerance, f_ghz
):
    # the reduced model holds the band near the target: far below it, the
    # port term at the real wavenumber needs exact solves (an enrichment of
    # the forward's basis) and may give up after them; the classification
    # touches neither
    model = request.getfixturevalue(case)
    model = model[0] if isinstance(model, tuple) else model
    op = driven_operator(model, 2)
    ports = eigen_ports(model, op, target=12e9, tolerance=tolerance)
    below = eigen_rule._below_cutoff(ports)
    per_hz = 2 * math.pi / C0 * op.built.scale
    assert below(f_ghz * 1e9 * per_hz)
    assert not below(12e9 * per_hz)
    assert all(p.term.version == 0 for p in ports)


# --- several ports and shared surfaces ------------------------------------------

# relative error at order 3 on these meshes: B3 6.9e-6, C 3.4e-5, N 3.9e-7

# N's exact root, GHz, on the outgoing branch: Re f at 0.977 of the TE10
# cutoff (6.557 GHz)
CASE_N = (("dielectric", 4.0, 6.0), ("air", 1.0, 5.0))
ROOT_N = 6.403885446223 + 0.564967925392j


def test_two_port_surfaces_resonance_matches_the_exact_root(tmp_path):
    mesh = ported_guide(tmp_path / "b3.msh", CASE_B3, left_port=True)
    model = ported_model(mesh, CASE_B3, TWO_PORTS)
    result = gfem.solve(
        model, gfem.Eigenmode(target=9.97e9, n_modes=3), options=ORDER_3
    )
    error, j = nearest_error(result, ROOT_B3)
    assert error <= 1e-4
    assert result.error[j] <= 1e-12


def test_shared_port_surface_resonance_matches_the_exact_root(tmp_path):
    # the TE20 family needs the surface's modal correction: its propagation
    # constant is not the lead (TE10) mode's of the Robin term
    mesh = ported_guide(tmp_path / "c.msh", CASE_A3, left_port=False, b_mm=5.0)
    model = ported_model(mesh, CASE_A3, SHARED_PORTS)
    result = gfem.solve(
        model, gfem.Eigenmode(target=15.2e9, n_modes=3), options=ORDER_3
    )
    error, j = nearest_error(result, ROOT_C20)
    assert error <= 1e-4
    assert result.error[j] <= 1e-12


def test_decaying_root_below_cutoff_is_kept_and_its_mirror_dropped(tmp_path):
    # a low-Q root with Re f below the port's cutoff decays in time: it is a
    # resonance; its mirror (Im f < 0) grows in time and never comes back
    mesh = ported_guide(tmp_path / "n.msh", CASE_N, left_port=False)
    model = ported_model(mesh, CASE_N, LONE_PORT)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntiBoundRootWarning)
        result = gfem.solve(
            model, gfem.Eigenmode(target=6.7e9, n_modes=4), options=ORDER_3
        )
    error, j = nearest_error(result, ROOT_N)
    assert error <= 1e-4
    assert result.error[j] <= 1e-12
    f = ghz(result.omega)
    assert np.all(f.imag > 0)


# --- the coupling Q -------------------------------------------------------------


def coupling_sum_error(result, j):
    """``|sum 1/q_p - 1/Q| Q`` of mode ``j``, ``Q = result.q``."""
    q = float(result.q[j])
    total = sum(1 / float(v[j]) for v in result.q_port.values())
    return abs(total - 1 / q) * q


def test_a_wave_port_s_coupling_q_sums_to_the_mode_s_q_at_high_q(tmp_path):
    # the outgoing power is the mode's loss, so 1/q_port is 1/Q up to the
    # O(1/Q^2) of the electric energy against the stored energy
    mesh = ported_guide(tmp_path / "a4.msh", CASE_A4, left_port=False)
    result = gfem.solve(
        ported_model(mesh, CASE_A4, LONE_PORT),
        gfem.Eigenmode(target=9.97e9, n_modes=1),
        options=ORDER_3,
    )
    assert abs(ghz(result.omega[0]) - ROOT_A4) / abs(ROOT_A4) <= 1e-4
    assert list(result.q_port) == ["p1"]
    assert coupling_sum_error(result, 0) <= 1e-6  # 2.8e-7


def test_a_wave_port_s_coupling_q_is_within_a_percent_of_q_at_q_5(case_a):
    result = gfem.solve(
        case_a, gfem.Eigenmode(target=11.67e9, n_modes=3), options=ORDER_3
    )
    j = int(np.argmin(np.abs(ghz(result.omega) - ROOT_A)))
    assert coupling_sum_error(result, j) <= 1e-2  # 7.9e-3


# port, air 5, eps_r = 10 4, air 5, port (mm): each port carries half the power
CASE_B = (("air", 1.0, 5.0), ("window", 10.0, 4.0), ("air", 1.0, 5.0))
ROOT_B = 11.884342073581 + 2.083440824738j


def test_the_two_ports_of_a_mirrored_guide_share_its_coupling_q_equally(tmp_path):
    # many high-Q modes of other guide families lie nearer the real target
    # than this Q 2.9 one
    mesh = ported_guide(tmp_path / "b.msh", CASE_B, left_port=True, mirror=True)
    result = gfem.solve(
        ported_model(mesh, CASE_B, TWO_PORTS),
        gfem.Eigenmode(target=11.88e9, n_modes=16),
        options=gfem.SolverOptions(order=2, linear_solver="direct"),
    )
    error, j = nearest_error(result, ROOT_B)
    assert error <= 2e-2
    q1, q2 = (float(result.q_port[name][j]) for name in ("left", "right"))
    assert abs(q1 - q2) <= 1e-8 * q1  # 8e-15


def test_two_ports_split_the_coupling_q_as_their_shares_of_the_power(tmp_path):
    mesh = ported_guide(tmp_path / "b3.msh", CASE_B3, left_port=True)
    result = gfem.solve(
        ported_model(mesh, CASE_B3, TWO_PORTS),
        gfem.Eigenmode(target=9.97e9, n_modes=3),
        options=ORDER_3,
    )
    error, j = nearest_error(result, ROOT_B3)
    assert error <= 1e-4
    q = float(result.q[j])
    # B3's exact power shares: in the high-Q limit q_p = Q / share
    for name, share in SHARE_B3.items():
        # 1.0e-4 (left) and 3.5e-4 (right) at order 3
        assert float(result.q_port[name][j]) == pytest.approx(q / share, rel=2e-3)


def test_each_port_of_a_shared_surface_reports_its_own_mode_s_coupling_q(tmp_path):
    mesh = ported_guide(tmp_path / "c.msh", CASE_A3, left_port=False, b_mm=5.0)
    model = ported_model(mesh, CASE_A3, SHARED_PORTS)
    result = gfem.solve(
        model, gfem.Eigenmode(target=14.6e9, n_modes=4), options=ORDER_3
    )
    for root, own, other in ((ROOT_C10, "te10", "te20"), (ROOT_C20, "te20", "te10")):
        error, j = nearest_error(result, root)
        assert error <= 1e-4
        # the mode leaves through its own port mode alone (6e-4 at Q 19 and 37)
        assert coupling_sum_error(result, j) <= 2e-3
        assert float(result.q_port[other][j]) >= 1e6 * float(result.q_port[own][j])


def _case_a_field(mesh, eps):
    return gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.MaterialField(eps_r=eps),
            "air": gfem.Material(eps_r=1.0),
        },
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port")],
    )


GRADIENT_OPTIONS = gfem.SolverOptions(
    order=2, linear_solver="direct", eigen_tolerance=1e-12
)


def test_a_wave_port_s_coupling_q_gradient_matches_a_central_difference(tmp_path):
    # the load reads the port mode's amplitude and power at the mode's
    # complex k through the reduced model's derivatives
    mesh = ported_guide(tmp_path / "a.msh", CASE_A, left_port=False, nx=4)

    problem = gfem.Eigenmode(target=11.67e9, n_modes=3)
    eps = jax.numpy.full(len(mesh.elements("dielectric")), 4.0)
    first = gfem.solve(_case_a_field(mesh, eps), problem, options=GRADIENT_OPTIONS)
    j = nearest_error(first, ROOT_A)[1]

    def objective(eps):
        result = gfem.solve(_case_a_field(mesh, eps), problem, options=GRADIENT_OPTIONS)
        return 1 / result.q_port["p1"][j]

    rng = np.random.default_rng(3)
    assert_gradient_matches(
        objective, eps, unit_directions(rng, eps.shape, 2), step=1e-2
    )


# --- the kept port model --------------------------------------------------------


@pytest.fixture
def lone_a(tmp_path_factory):
    mesh = ported_guide(
        tmp_path_factory.mktemp("a4x") / "a.msh", CASE_A, left_port=False, nx=4
    )
    eps = jax.numpy.full(len(mesh.elements("dielectric")), 4.0)
    return mesh, eps


@pytest.fixture
def sweeps(monkeypatch):
    """How many port mode sweeps have run, as a one-item list."""
    from gfem._engine import port_term

    count = [0]
    sweep = port_term.port_mode_sweep

    def counted(*args, **kwargs):
        count[0] += 1
        return sweep(*args, **kwargs)

    monkeypatch.setattr(port_term, "port_mode_sweep", counted)
    cache.ENTRIES.clear()
    yield count
    cache.ENTRIES.clear()


def _budget(model, problem):
    """A ``retain_memory`` that keeps a forward's fields and port terms only."""
    rule = eigen_rule.EigenmodeRule(model, problem, GRADIENT_OPTIONS)
    _, entry = rule.forward(model, keep=True)
    holder = entry.handle
    fields = sum(f.nbytes for f in entry.fields.values())
    ports = holder.nbytes - holder.operator.nbytes
    assert ports > 0  # the port terms count in the entry's bytes
    budget = fields + ports + holder.operator.nbytes // 2
    entry.free()
    return budget


def _objective(mesh, problem, read, **options):
    def objective(eps):
        result = gfem.solve(
            _case_a_field(mesh, eps),
            problem,
            options=dataclasses.replace(GRADIENT_OPTIONS, **options),
        )
        return read(result)

    return objective


def test_frequency_gradient_after_free_base_is_the_kept_one_bit_for_bit(lone_a, sweeps):
    # one entry's backward before and after its operator is freed: the
    # frequency pairs read the entry's fields alone, and no port is swept
    mesh, eps = lone_a
    model = _case_a_field(mesh, eps)
    rule = eigen_rule.EigenmodeRule(
        model, gfem.Eigenmode(target=11.67e9), GRADIENT_OPTIONS
    )
    _, entry = rule.forward(model, keep=True)
    cotangents = {"omega": np.array([1.0 - 30j])}
    before = rule.backward(model, entry, cotangents)
    swept = sweeps[0]
    entry.handle.free_base(None)
    assert entry.handle.operator is None
    assert entry.handle.ports  # the port terms stay
    after = rule.backward(model, entry, cotangents)
    assert sweeps[0] == swept
    assert len(after) == len(before) == 1
    for a, b in zip(after, before, strict=True):
        np.testing.assert_array_equal(a.left, b.left)
        np.testing.assert_array_equal(a.right, b.right)
        assert a.weights == b.weights
    entry.free()


def test_coupling_q_gradient_without_the_operator_reads_the_kept_ports(lone_a, sweeps):
    # the backward prepares its own eigen operator, with the forward's port
    # terms, and sweeps no port
    mesh, eps = lone_a
    problem = gfem.Eigenmode(target=11.67e9)
    budget = _budget(_case_a_field(mesh, eps), problem)

    def read(result):
        return 1 / result.q_port["p1"][0]

    kept = jax.grad(_objective(mesh, problem, read))(eps)
    value, pullback = jax.vjp(
        _objective(mesh, problem, read, retain_memory=budget), eps
    )
    before = sweeps[0]
    (grad,) = pullback(jax.numpy.ones_like(value))
    assert sweeps[0] == before
    np.testing.assert_allclose(grad, kept, rtol=1e-7, atol=1e-7 * abs(kept).max())


# --- refusals of the coupling Q's gradient --------------------------------------


def _q_port_gradient(mesh, eps):
    objective = _objective(
        mesh, gfem.Eigenmode(target=11.67e9), lambda r: 1 / r.q_port["p1"][0]
    )
    return jax.grad(objective)(eps)


def test_a_mode_without_power_through_the_port_refuses_its_coupling_q_gradient(
    lone_a, monkeypatch
):
    # a mode with no amplitude on the port's mode (a symmetry) has an
    # infinite q_port; its power is set to 0 here, which no mesh reaches
    # exactly
    mesh, eps = lone_a
    slopes = PortTerm.power_slopes

    def no_power(self, k, x, pair=None):
        power, d_x, d_k = slopes(self, k, x, pair)
        return 0 * power, d_x, d_k

    monkeypatch.setattr(PortTerm, "power_slopes", no_power)
    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match=r"NonDifferentiableError: q_port\['p1'\] of mode 0 is infinite",
    ):
        _q_port_gradient(mesh, eps)


def test_a_degenerate_port_mode_refuses_its_coupling_q_gradient(lone_a, monkeypatch):
    # a requested port mode degenerate with another at the resonance has no
    # shape derivative; the sweep refuses one degenerate on the real band, so
    # the degeneracy is forced here
    mesh, eps = lone_a
    monkeypatch.setattr(PortTerm, "degenerate", lambda *_: [0])
    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match=r"NonDifferentiableError: q_port\['p1'\] of mode 0 has no derivative: "
        r"the port's mode is degenerate.*Differentiate omega or q",
    ):
        _q_port_gradient(mesh, eps)


def test_a_port_mode_is_degenerate_within_the_relative_gap(hybrid):
    # the hybrid guide's lead mode and the next one of its reduced pencil
    _, _, op, (port,) = hybrid
    k = 2 * math.pi * 12e9 / C0 * op.built.scale * (1 + 0.02j)
    assert port.term.degenerate(k, 1e-6) == []
    assert port.term.degenerate(k, 10.0) == [0]


# --- the requested power share and its warning ----------------------------------


@pytest.fixture(scope="module")
def line(case_a):
    """Case A at three modes: the root and two modes trapped against the port."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gfem.solve(
            case_a, gfem.Eigenmode(target=11.67e9, n_modes=3), options=ORDER_3
        )
    texts = [
        str(c.message) for c in caught if c.category is gfem.UnrequestedPortModeWarning
    ]
    return result, texts


def test_the_requested_power_share_is_one_row_per_port_surface(line):
    result, _ = line
    share = result.requested_power_share
    assert list(share) == ["port"]
    assert share["port"].shape == (3,)
    assert share["port"].dtype == np.float64
    j = int(np.argmin(np.abs(ghz(result.omega) - ROOT_A)))
    assert abs(1 - float(share["port"][j])) <= 2e-2


def test_a_model_without_wave_ports_has_no_requested_power_share():
    result = gfem.EigenmodeResult(omega=np.array([1.0]), error=np.zeros(1))
    assert result.requested_power_share == {}


def test_a_surface_that_absorbs_no_power_has_a_nan_share(hybrid):
    from gfem._eigen.rule import _layout
    from gfem._eigen.share import requested_power_shares
    from gfem._engine import terms

    _, _, op, ports = hybrid
    k = terms.wavenumber(2 * math.pi * 12e9, op.built.scale)
    x = np.zeros((op.space.ndof, 1), complex)
    share, loss = requested_power_shares(
        ports, op.space, x, [k], scale=op.built.scale, pair=_layout(op).pair
    )
    assert share.shape == (1, 1)
    assert np.isnan(share).all()
    assert np.isnan(loss).all()


def test_the_requested_power_share_is_exported_over_mode_and_port_surface(line):
    pytest.importorskip("xarray")
    result, _ = line
    data = result.to_xarray()
    variable = data["requested_power_share"]
    assert variable.dims == ("mode", "port_surface")
    assert list(data["port_surface"].values) == ["port"]
    np.testing.assert_array_equal(
        variable.values[:, 0], np.asarray(result.requested_power_share["port"])
    )


def test_a_result_with_its_share_survives_pickle(line):
    import pickle

    result, _ = line
    back = pickle.loads(pickle.dumps(result))  # noqa: S301
    np.testing.assert_array_equal(
        back.requested_power_share["port"], result.requested_power_share["port"]
    )


def test_the_requested_power_share_takes_no_gradient(lone_a):
    mesh, eps = lone_a

    def objective(read):
        return _objective(mesh, gfem.Eigenmode(target=11.67e9), read)

    with pytest.raises(gfem.NonDifferentiableError, match="requested_power_share"):
        jax.grad(objective(lambda r: r.requested_power_share["port"][0]))(eps)
    # the opt-out reads the value, and omega keeps its gradient
    grad = jax.grad(
        objective(
            lambda r: (
                jax.lax.stop_gradient(r.requested_power_share["port"][0])
                + r.frequency[0]
            )
        )
    )(eps)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.abs(np.asarray(grad)).max() > 0


# --- the warning's forms -----------------------------------------------------------


def _surface(requested, share, loss=None):
    """A stand-in surface of ``requested`` ranks and its flagged-mode rows."""
    from types import SimpleNamespace

    from gfem._eigen.share import Unrequested

    ports = [SimpleNamespace(mode=r) for r in requested]
    port = SimpleNamespace(term=SimpleNamespace(ports=ports, modes=object()))
    share = np.asarray(share, float)
    loss = np.ones_like(share) if loss is None else np.asarray(loss, float)
    return Unrequested(port, share, loss)


KS = [0.5 + 0.01j, 0.6 + 0.02j, 0.7 + 0.03j]


def test_a_mode_with_an_unrequested_rank_propagating_gets_the_open_form(monkeypatch):
    from gfem._eigen import share as module

    monkeypatch.setattr(module, "propagating_modes", lambda _modes, _f, rank: rank)
    flagged = _surface([1], [1.0, 1e-9, 0.2])
    (text,) = flagged.messages("s", KS, 1.0, np.array([5.0, 190.0, 40.0]))
    assert "mode(s) 1 at" in text
    assert "2 at" in text
    assert "Mode 2 of the surface propagates" in text
    assert "mode=2" in text
    assert "If the warning persists" in text
    assert "0 at" not in text.split("on wave port")[0]


def test_a_skipped_rank_always_propagates_and_gets_the_open_form(monkeypatch):
    from gfem._eigen import share as module

    def never(*_args):
        raise AssertionError("a skipped rank needs no port mode solve")

    monkeypatch.setattr(module, "propagating_modes", never)
    flagged = _surface([2], [1.0, 1e-9, 1.0])
    (text,) = flagged.messages("s", KS, 1.0, np.array([5.0, 190.0, 5.0]))
    assert "Mode 1 of the surface propagates" in text
    assert "mode=1" in text


def test_a_mode_with_no_unrequested_rank_propagating_gets_the_trapped_form(
    monkeypatch,
):
    from gfem._eigen import share as module

    monkeypatch.setattr(module, "propagating_modes", lambda _modes, _f, rank: rank - 1)
    flagged = _surface([1], [1.0, 1e-9, 1e-12])
    (text,) = flagged.messages("s", KS, 1.0, np.array([5.0, 190.0, 191.0]))
    assert "trapped" in text
    assert "propagates" not in text
    assert "1 at" in text
    assert "2 at" in text


def test_the_two_forms_are_two_warnings_and_a_loss_gate_drops_a_mode(monkeypatch):
    from gfem._eigen import share as module

    # mode 1's frequency has rank 2 propagating, mode 2's does not; mode 0 is a
    # requested one, and mode 3 has a tiny share of a surface that takes
    # little of its loss
    monkeypatch.setattr(
        module, "propagating_modes", lambda _modes, f, rank: rank if f < 3e7 else 0
    )
    ks = [0.5, 0.6, 0.7, 0.8]
    flagged = _surface([1], [1.0, 1e-9, 1e-9, 1e-9], [1.0, 1.0, 1.0, 7e-4])
    assert list(flagged.flagged) == [1, 2]
    texts = flagged.messages("s", ks, 1.0, np.ones(4))
    assert len(texts) == 2
    assert "propagates" in texts[0]
    assert "propagates" not in texts[1]


def test_the_warning_is_public_and_replayed_by_the_solve_cache():
    from gfem import _solve_cache

    assert "UnrequestedPortModeWarning" in gfem.__all__
    assert issubclass(gfem.UnrequestedPortModeWarning, UserWarning)
    assert gfem.UnrequestedPortModeWarning in _solve_cache.REPLAYED


# --- a refusal of the port term at an iterate -----------------------------------


def refuse_off_axis(monkeypatch, share):
    """The port term refuses where Im k exceeds ``share`` of Re k, but at the
    search's seed shift (a stand-in for a complex frequency its port mode
    cannot be followed to)."""
    real = PortTerm.at

    def at(self, k):
        z = complex(k)
        if z.imag > share * z.real and abs(z.imag / z.real - 0.125) > 1e-9:
            raise PortTermRefusalError(self.label, z, "forced")
        return real(self, k)

    monkeypatch.setattr(PortTerm, "at", at)


@pytest.fixture
def refuses_off_axis(monkeypatch):
    refuse_off_axis(monkeypatch, 0.15)


@pytest.mark.usefixtures("refuses_off_axis")
def test_a_guard_seed_the_port_term_refuses_is_dropped_silently(case_h):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = gfem.solve(
            case_h, gfem.Eigenmode(target=7.3e9, n_modes=1), options=ORDER_3
        )
    assert len(result.omega) == 1
    assert abs(ghz(result.omega[0]) - (7.30912 + 0.21415j)) <= 1e-3


def test_a_requested_seed_the_port_term_refuses_is_reported_on_a_lost_line(
    case_h, monkeypatch
):
    refuse_off_axis(monkeypatch, 0.0)
    problem = gfem.Eigenmode(target=7.3e9, n_modes=2)
    with pytest.raises(gfem.EigenmodeConvergenceError) as raised:
        gfem.solve(case_h, problem, options=ORDER_3)
    lines = str(raised.value).splitlines()
    lost = [x for x in lines if x.lstrip().startswith("lost:")]
    assert lost
    assert "GHz" in lost[0]
    assert "WavePort 'p' on surface" in lost[0]
    assert "can follow its port mode" in lost[0]


def test_a_refusal_in_a_reshift_leaves_the_operator_usable(hybrid, monkeypatch):
    _, rule, op, ports = hybrid
    sigma = 2 * math.pi * 12e9 / C0 * op.built.scale
    eop = rule.operator(op, sigma, ports=ports)
    try:
        x = free_vector(eop, 3)
        eop.solve(x)  # prepared at sigma
        far = sigma * (1.01 + 0.03j)
        real = PortTerm.at

        def at(self, k):
            if complex(k) == far:
                raise PortTermRefusalError(self.label, complex(k), "forced")
            return real(self, k)

        monkeypatch.setattr(PortTerm, "at", at)
        with pytest.raises(PortTermRefusalError):
            eop.reshift(far)
        assert eop.shift == sigma
        back = eop.solve(eop.apply(sigma, x))  # the old factorization lives
        assert np.linalg.norm(back - x) <= 1e-10 * np.linalg.norm(x)
        near = sigma * (1.01 + 0.02j)
        eop.reshift(near)
        back = eop.solve(eop.apply(near, x))
        assert np.linalg.norm(back - x) <= 1e-10 * np.linalg.norm(x)
    finally:
        eop.free_handle()


# --- the coverage search ---------------------------------------------------------


def test_a_strip_the_attempts_cannot_cover_raises_instead_of_returning(
    case_a, monkeypatch
):
    # a first seed solve of two pairs does not reach the certificate's radius,
    # and one attempt cannot ask for more
    from gfem._eigen import search

    monkeypatch.setattr(search, "_COVER_NEV", 2)
    monkeypatch.setattr(search, "_COVER_ATTEMPTS", 1)
    with pytest.raises(gfem.EigenmodeConvergenceError, match="could not cover") as e:
        gfem.solve(case_a, gfem.Eigenmode(target=11.67e9, n_modes=1), options=ORDER_3)
    assert "after 1 attempts" in str(e.value)
    assert "lower n_modes or move target" in str(e.value)


def test_the_nearest_mode_does_not_depend_on_n_modes(case_a):
    one = gfem.solve(case_a, gfem.Eigenmode(target=11.67e9, n_modes=1), options=ORDER_3)
    two = gfem.solve(case_a, gfem.Eigenmode(target=11.67e9, n_modes=2), options=ORDER_3)
    assert abs(one.omega[0] - two.omega[0]) <= 1e-7 * abs(two.omega[0])
    f = ghz(one.omega)
    assert abs(f[0] - ROOT_A) <= 1e-3 * ROOT_A


def test_a_seed_shift_the_port_term_refuses_raises_a_convergence_error(
    case_h, monkeypatch
):
    real = PortTerm.at

    def at(self, k):
        if complex(k).imag > 0.1 * complex(k).real:
            raise PortTermRefusalError(self.label, complex(k), "forced")
        return real(self, k)

    monkeypatch.setattr(PortTerm, "at", at)
    with pytest.raises(gfem.EigenmodeConvergenceError, match="cannot follow"):
        gfem.solve(case_h, gfem.Eigenmode(target=7.3e9, n_modes=1), options=ORDER_3)
