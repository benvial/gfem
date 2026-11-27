"""`gfem.cascade` on synthetic blocks: wiring, algebra, refusals, transformations.

No FEM: blocks are `CascadeResult` instances built from an S matrix, the
propagation constants of their section ports and polarity arrows
(`cascade_cases`). Tolerances. The algebra is a handful of complex
operations on matrices of O(1) entries and a solve of a 2-10 order system
whose condition number stays below 10 (reflections at most 0.9), so a result
differs from an independent evaluation by a few hundred rounding errors at
most: 1e-13 (450 eps) is the working gate, and each test states the
measured error beside it. A phase :math:`\\gamma \\ell` of about 20 rad enters
through an exponential whose relative error is :math:`|\\gamma \\ell|\\,\\epsilon`,
2e-15.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cascade_cases import (
    DOWN,
    FREQUENCY,
    UP,
    airy,
    evanescent,
    fresnel_pair,
    k0,
    random_block,
    section_ports,
    synthetic_block,
)

import gfem

TOL = 1e-13  # see the module docstring
BETA_OUTER, BETA_SLAB = 1.0, 3.2  # refractive indices of the two guides


def vars_of(mode):
    return {f.name: getattr(mode, f.name) for f in dataclasses.fields(mode)}


def _gammas(loss: float):
    """Outer and slab :math:`\\gamma` (1/m); ``loss`` is the slab's alpha/beta."""
    return 1j * BETA_OUTER * k0(), (1j + loss) * BETA_SLAB * k0()


def _s(result, to, source):
    return np.asarray(result.s[to, source])


# --- closed form ----------------------------------------------------------------


@pytest.mark.parametrize("loss", [0.0, 0.05], ids=["lossless", "lossy"])
def test_two_fresnel_junctions_give_the_airy_slab(loss):
    g1, g2 = _gammas(loss)
    a, b = fresnel_pair(g1, g2)
    length = 3.1e-6  # about 20 rad of phase in the slab
    result = gfem.cascade([a, b], {"s": length})
    rho, tau = airy(g1, g2, length)
    # measured below 1e-15; two exponentials and a 2x2 solve, 1e-13 gate
    np.testing.assert_allclose(_s(result, "a", "a"), rho, rtol=0, atol=TOL)
    np.testing.assert_allclose(_s(result, "b", "a"), tau, rtol=0, atol=TOL)
    np.testing.assert_allclose(_s(result, "a", "b"), tau, rtol=0, atol=TOL)
    if loss == 0.0:
        total = np.abs(rho) ** 2 + np.abs(tau) ** 2
        np.testing.assert_allclose(total, 1.0, rtol=0, atol=TOL)


def test_the_section_gamma_is_the_mean_and_the_mismatch_is_in_k0():
    g1, g2 = _gammas(0.0)
    a, b = fresnel_pair(g1, g2)
    # a second discretization of the same guide: gamma off by 1e-3 k0
    shift = 1e-3 * k0()
    modes = dict(b.port_modes)
    modes["s@1"] = type(modes["s@1"])(
        **{**vars_of(modes["s@1"]), "gamma": modes["s@1"].gamma + shift}
    )
    b = gfem.CascadeResult(s=b.s, port_modes=modes, polarization="Ez")
    result = gfem.cascade([a, b], {"s": 1e-6})
    np.testing.assert_allclose(result.gamma["s"][:, 0], g2 + shift / 2, rtol=1e-14)
    np.testing.assert_allclose(result.gamma_mismatch["s"][:, 0], 1e-3, rtol=1e-12)
    assert result.gamma["s"].shape == (len(FREQUENCY), 2)


# --- scikit-rf ------------------------------------------------------------------


def _network(rf, s):
    freq = rf.Frequency.from_f(np.asarray(FREQUENCY), unit="hz")
    return rf.Network(frequency=freq, s=np.asarray(s), z0=50.0)


def _line(rf, gamma, length):
    """A 2N-port of N lossy lines: ports (A side 1..N, B side 1..N)."""
    n = gamma.shape[1]
    e = np.exp(-gamma * length)
    s = np.zeros((len(FREQUENCY), 2 * n, 2 * n), complex)
    for m in range(n):
        s[:, m, n + m] = s[:, n + m, m] = e[:, m]
    return _network(rf, s)


def test_random_reciprocal_blocks_agree_with_scikit_rf():
    rf = pytest.importorskip("skrf")
    rng = np.random.default_rng(7)
    gamma = {
        "s@1": 1j * 3.0 * k0(),
        "s@2": 4.0 * k0(),
    }
    a = random_block(rng, ["a1", "a2", "s@1", "s@2"], gamma=gamma)
    b = random_block(rng, ["s@1", "s@2", "b1"], gamma=gamma)
    length = 1.7e-6
    result = gfem.cascade([a, b], {"s": length})
    g = np.stack([gamma["s@1"] * np.ones(2), gamma["s@2"] * np.ones(2)], axis=-1)
    line = _line(rf, g, length)
    left = rf.network.connect(_network(rf, a.s.values), 2, line, 0, num=2)
    both = rf.network.connect(left, 2, _network(rf, b.s.values), 0, num=2)
    assert result.s.ports == ("a1", "a2", "b1")
    # measured below 1e-15 on 3x3 O(1) entries; 1e-13 gate
    np.testing.assert_allclose(result.s.values, both.s, rtol=0, atol=TOL)


def test_two_ports_cascade_as_scikit_rf_power_operator():
    rf = pytest.importorskip("skrf")
    rng = np.random.default_rng(11)
    gamma = {"s@1": 5.0 * k0()}
    a = random_block(rng, ["a", "s@1"], gamma=gamma)
    b = random_block(rng, ["s@1", "b"], gamma=gamma)
    length = 0.9e-6
    result = gfem.cascade([a, b], {"s": length})
    line = _line(rf, np.broadcast_to(gamma["s@1"][:, None], (2, 1)), length)
    chained = _network(rf, a.s.values) ** line ** _network(rf, b.s.values)
    np.testing.assert_allclose(result.s.values, chained.s, rtol=0, atol=TOL)


# --- identities -----------------------------------------------------------------


def _pair(rng, n=2, section="s"):
    """Two random passive blocks: modes 1..n-1 propagate, mode n does not."""
    names = section_ports(section, n)
    gamma = {nm: 1j * (2.0 + m) * k0() for m, nm in enumerate(names[:-1])}
    gamma[names[-1]] = evanescent(n)
    a = random_block(rng, ["a1", "a2", *names], gamma=gamma)
    b = random_block(rng, [*names, "b1"], gamma=gamma)
    return a, b, gamma


def test_length_zero_is_the_direct_connection_of_two_ports():
    rng = np.random.default_rng(3)
    gamma = {"s@1": evanescent(1)}
    a = random_block(rng, ["a", "s@1"], gamma=gamma)
    b = random_block(rng, ["s@1", "b"], gamma=gamma)
    result = gfem.cascade([a, b], {"s": 0.0})
    sa, sb = np.asarray(a.s.values), np.asarray(b.s.values)
    # S11 = A11 + A12 B11 A21 / (1 - A22 B11), written out for two ports
    direct = np.empty_like(sa)
    den = 1 - sa[:, 1, 1] * sb[:, 0, 0]
    direct[:, 0, 0] = sa[:, 0, 0] + sa[:, 0, 1] * sb[:, 0, 0] * sa[:, 1, 0] / den
    direct[:, 0, 1] = sa[:, 0, 1] * sb[:, 0, 1] / den
    direct[:, 1, 0] = sb[:, 1, 0] * sa[:, 1, 0] / den
    direct[:, 1, 1] = sb[:, 1, 1] + sb[:, 1, 0] * sa[:, 1, 1] * sb[:, 0, 1] / den
    np.testing.assert_allclose(result.s.values, direct, rtol=0, atol=TOL)


def _thru(n, names_a, names_b, gamma_a, gamma_b):
    """The block (0 I; I 0) between two sections of N modes."""
    s = np.zeros((len(FREQUENCY), 2 * n, 2 * n), complex)
    for m in range(n):
        s[:, m, n + m] = s[:, n + m, m] = 1.0
    return synthetic_block([*names_a, *names_b], s, gamma={**gamma_a, **gamma_b})


def test_a_thru_block_is_neutral_and_lengths_add():
    rng = np.random.default_rng(5)
    gamma_p = {"p@1": 1j * 3.0 * k0(), "p@2": evanescent(2)}
    gamma_q = {"q@1": 1j * 3.0 * k0(), "q@2": evanescent(2)}
    a = random_block(rng, ["a1", "a2", "p@1", "p@2"], gamma=gamma_p)
    c = random_block(rng, ["q@1", "q@2", "b1"], gamma=gamma_q)
    t = _thru(2, list(gamma_p), list(gamma_q), gamma_p, gamma_q)
    # joined at length 0, the thru block leaves A as it is, its ports renamed
    neutral = gfem.cascade([a, t], {"p": 0.0})
    np.testing.assert_allclose(neutral.s.values, a.s.values, rtol=0, atol=TOL)
    # additivity: l1, thru, l2 equals l1 + l2 (the same gamma on every end)
    l1, l2 = 0.8e-6, 1.3e-6
    split = gfem.cascade([a, t, c], {"p": l1, "q": l2})
    renamed = synthetic_block(["a1", "a2", "q@1", "q@2"], a.s.values, gamma=gamma_q)
    whole = gfem.cascade([renamed, c], {"q": l1 + l2})
    # measured below 1e-15; 1e-13 gate
    np.testing.assert_allclose(split.s.values, whole.s.values, rtol=0, atol=TOL)


def test_the_one_solve_form_equals_the_pairwise_form_in_either_order():
    rng = np.random.default_rng(13)
    p = {"p@1": 1j * 2.5 * k0(), "p@2": evanescent(2)}
    q = {"q@1": 1j * 2.5 * k0(), "q@2": evanescent(2)}
    a = random_block(rng, ["a1", "p@1", "p@2"], gamma=p)
    m = random_block(rng, ["p@1", "p@2", "q@1", "q@2"], gamma={**p, **q})
    c = random_block(rng, ["q@1", "q@2", "c1", "c2"], gamma=q)
    lengths = {"p": 1.1e-6, "q": 0.6e-6}
    flat = gfem.cascade([a, m, c], lengths)
    left = gfem.cascade(
        [gfem.cascade([a, m], {"p": lengths["p"]}), c], {"q": lengths["q"]}
    )
    right = gfem.cascade(
        [a, gfem.cascade([m, c], {"q": lengths["q"]})], {"p": lengths["p"]}
    )
    assert flat.s.ports == left.s.ports == right.s.ports == ("a1", "c1", "c2")
    # measured below 1e-15; three solves of 2-4 unknowns, 1e-13 gate
    np.testing.assert_allclose(left.s.values, flat.s.values, rtol=0, atol=TOL)
    np.testing.assert_allclose(right.s.values, flat.s.values, rtol=0, atol=TOL)
    assert isinstance(left, gfem.CascadeResult)


# --- physics --------------------------------------------------------------------


def test_a_cascade_is_reciprocal_and_passive():
    rng = np.random.default_rng(17)
    a, b, _ = _pair(rng, n=3)
    s = np.asarray(gfem.cascade([a, b], {"s": 0.7e-6}).s.values)
    np.testing.assert_allclose(s, np.swapaxes(s, 1, 2), rtol=0, atol=TOL)
    worst = max(np.linalg.eigvalsh(x.conj().T @ x).max() for x in s)
    # the blocks are contractions (norm 0.9) and a section never amplifies
    assert worst <= 1 + TOL, worst


def test_lossless_blocks_with_propagating_modes_give_a_unitary_result():
    rng = np.random.default_rng(19)
    # mode 1 propagates through unitary blocks; the evanescent mode 2 is dead
    gamma = {"s@1": 1j * 3.0 * k0(), "s@2": evanescent(2)}
    a = random_block(
        rng, ["a1", "a2", "s@1", "s@2"], kind="unitary", dead=("s@2",), gamma=gamma
    )
    b = random_block(
        rng, ["s@1", "s@2", "b1", "b2"], kind="unitary", dead=("s@2",), gamma=gamma
    )
    s = np.asarray(gfem.cascade([a, b], {"s": 2.3e-6}).s.values)
    eye = np.eye(s.shape[1])
    # measured between 1e-15 and 1e-14 (a 4x4 unitary product, 1e-13 gate)
    for x in s:
        np.testing.assert_allclose(x.conj().T @ x, eye, rtol=0, atol=TOL)


def test_a_long_evanescent_section_is_the_decoupled_limit_and_stays_finite():
    rng = np.random.default_rng(23)
    gamma = {"s@1": evanescent(1), "s@2": evanescent(2)}
    a = random_block(rng, ["a1", "s@1", "s@2"], gamma=gamma)
    b = random_block(rng, ["s@1", "s@2", "b1"], gamma=gamma)
    # exp(-3 k0 * 0.1 m) = exp(-6e4) underflows to exactly 0: the blocks decouple
    result = gfem.cascade([a, b], {"s": 0.1})
    expected = np.zeros((len(FREQUENCY), 2, 2), complex)
    expected[:, 0, 0] = np.asarray(a.s.values)[:, 0, 0]
    expected[:, 1, 1] = np.asarray(b.s.values)[:, 2, 2]
    assert np.isfinite(np.asarray(result.s.values)).all()
    np.testing.assert_allclose(result.s.values, expected, rtol=0, atol=1e-15)
    grad = jax.grad(
        lambda x: jnp.abs(gfem.cascade([a, b], {"s": x}).s.values[0, 0, 1])
    )(0.1)
    assert np.isfinite(float(grad))


def test_the_exposed_ports_keep_block_order_excitation_z_ref_and_offsets():
    rng = np.random.default_rng(29)
    gamma = {"s@1": evanescent(1)}
    a = random_block(
        rng, ["a1", "s@1", "a2"], gamma=gamma, z_ref=50.0, offsets=(("a1", 1e-6),)
    )
    b = random_block(rng, ["b1", "s@1"], gamma=gamma, z_ref=75.0)
    result = gfem.cascade([a, b], {"s": 1e-6})
    assert result.s.ports == result.s.excited == ("a1", "a2", "b1")
    np.testing.assert_array_equal(result.s.z_ref[0], [50.0, 50.0, 75.0])
    assert result.offsets == (("a1", 1e-6),)
    assert result.polarization == "Ez"
    assert result.s.frequency == FREQUENCY
    text = repr(result)
    assert "'a1', 'a2', 'b1'" in text
    assert "'s' (1 modes, 1e-06 m)" in text
    assert "THz" in text


def test_the_polarity_arrows_set_the_gauge():
    g1, g2 = _gammas(0.0)
    a, b = fresnel_pair(g1, g2)
    reference = gfem.cascade([a, b], {"s": 3.1e-6})
    # block B's modes both point down: the gauge t = -1 flips B's row and
    # column of every mode, so S of its exposed port from A changes sign and
    # the reflections, flipped twice, do not
    flipped = {
        k: type(v)(**{**vars_of(v), "polarity": DOWN}) for k, v in b.port_modes.items()
    }
    down = gfem.CascadeResult(s=b.s, port_modes=flipped, polarization="Ez")
    result = np.asarray(gfem.cascade([a, down], {"s": 3.1e-6}).s.values)
    expected = np.asarray(reference.s.values) * np.array([[1, -1], [-1, 1]])
    np.testing.assert_allclose(result, expected, rtol=0, atol=TOL)
    # the same block written in the opposite sign of mode 1 (its row and
    # column negated) with the arrow reversed joins as the original does
    values = np.asarray(b.s.values).copy()
    values[:, 0, :] *= -1
    values[:, :, 0] *= -1
    gamma = {k: v.gamma for k, v in b.port_modes.items()}
    negated = synthetic_block(
        b.s.ports, values, gamma=gamma, polarity={"s@1": DOWN, "s@2": UP}
    )
    again = gfem.cascade([a, negated], {"s": 3.1e-6})
    np.testing.assert_allclose(again.s.values, reference.s.values, rtol=0, atol=TOL)


# --- refusals -------------------------------------------------------------------


def _fresnel():
    g1, g2 = _gammas(0.0)
    return fresnel_pair(g1, g2)


def _remake(block, **changes):
    """``block`` with some `CascadeResult` fields replaced."""
    return dataclasses.replace(block, **changes)


def _with_mode(block, port, **changes):
    modes = dict(block.port_modes)
    modes[port] = type(modes[port])(**{**vars_of(modes[port]), **changes})
    return _remake(block, port_modes=modes)


def _with_s(block, **changes):
    return _remake(block, s=dataclasses.replace(block.s, **changes))


class _Plane:
    def __init__(self, normal):
        self.normal = normal


REFUSALS = {
    "frequencies": (
        lambda a, b: (a, _with_s(b, frequency=(1.0e14, 1.2e14))),
        {"s": 1e-6},
        "same frequencies",
    ),
    "not excited": (
        lambda a, b: (
            a,
            _with_s(b, excited=b.s.ports[:2], values=b.s.values[:, :, :2]),
        ),
        {"s": 1e-6},
        r"Driven\(excite=None\)",
    ),
    "section name": (lambda a, b: (a, b), {"s@": 1e-6}, "contains '@'"),
    "no block": (lambda a, b: (a, b), {"s": 1e-6, "t": 1e-6}, "no block"),
    "one block": (
        lambda a, b: (
            a,
            _with_s(b, ports=("t@1", "t@2", "b"), excited=("t@1", "t@2", "b")),
        ),
        {"s": 1e-6},
        "1 block",
    ),
    "unequal N": (
        lambda a, b: (
            a,
            _remake(
                _with_s(
                    b,
                    ports=("s@1", "b"),
                    excited=("s@1", "b"),
                    values=b.s.values[:, [0, 2]][:, :, [0, 2]],
                ),
                port_modes={"s@1": b.port_modes["s@1"]},
            ),
        ),
        {"s": 1e-6},
        "same number",
    ),
    "ranks": (
        lambda a, b: (
            a,
            _with_s(b, ports=("s@1", "s@3", "b"), excited=("s@1", "s@3", "b")),
        ),
        {"s": 1e-6},
        r"ranks .* must be 1\.\.N",
    ),
    "mode rank": (
        lambda a, b: (a, _with_mode(b, "s@2", mode=1)),
        {"s": 1e-6},
        "carries mode 1, not 2",
    ),
    "no port mode": (
        lambda a, b: (a, _remake(b, port_modes={"s@1": b.port_modes["s@1"]})),
        {"s": 1e-6},
        "not a wave port mode",
    ),
    "shared port": (
        lambda a, b: (
            a,
            _with_s(b, ports=("a", "s@1", "s@2"), excited=("a", "s@1", "s@2")),
        ),
        {"s": 1e-6},
        r"port 'a' occurs in blocks 0 and 1.*sections=\{'name': 0\.0\}",
    ),
    "offset": (
        lambda a, b: (a, _remake(b, offsets=(("s@1", 1e-6),))),
        {"s": 1e-6},
        "offset 1e-06 m",
    ),
    "pml": (
        lambda a, b: (a, _with_mode(b, "s@1", pml="layer")),
        {"s": 1e-6},
        "perfectly matched layer",
    ),
    "polarization": (
        lambda a, b: (a, _remake(b, polarization="Hz")),
        {"s": 1e-6},
        "different polarizations",
    ),
    "length shape": (lambda a, b: (a, b), {"s": [1e-6, 2e-6]}, "must be a scalar"),
    "voltage path": (
        lambda a, b: (a, _with_mode(b, "s@1", Z_PV=jnp.full(2, 50.0 + 0j))),
        {"s": 1e-6},
        "voltage path",
    ),
    "normals": (
        lambda a, b: (
            _with_mode(a, "s@1", e_field=_Plane((1.0, 0.0))),
            _with_mode(b, "s@1", e_field=_Plane((1.0, 0.0))),
        ),
        {"s": 1e-6},
        "do not face each other",
    ),
    "negative": (lambda a, b: (a, b), {"s": -1e-6}, r"real, finite and >= 0"),
    "infinite": (lambda a, b: (a, b), {"s": np.inf}, r"real, finite and >= 0"),
    "complex": (lambda a, b: (a, b), {"s": 1e-6 + 0j}, r"real, finite and >= 0"),
    "propagating last mode": (
        lambda a, b: (a, _with_mode(b, "s@2", gamma=jnp.full(2, 1j * 2.0e6))),
        {"s": 1e-6},
        "kept mode 2 of section 's' propagates",
    ),
    "mismatch": (
        lambda a, b: (a, _with_mode(b, "s@1", gamma=b.port_modes["s@1"].gamma * 1.1)),
        {"s": 1e-6},
        "do not model the same guide",
    ),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_every_refusal_names_its_cause(case):
    build, sections, message = REFUSALS[case]
    a, b = _fresnel()
    blocks = build(a, b)
    with pytest.raises(ValueError, match=message):
        gfem.cascade(list(blocks), sections)


def test_a_section_in_three_blocks_is_refused():
    a, b = _fresnel()
    with pytest.raises(ValueError, match=r"3 blocks \(0, 1, 2\)"):
        gfem.cascade([a, b, b], {"s": 1e-6})


def test_a_section_in_no_block_and_mixed_modes_refuse():
    a, b = _fresnel()
    with pytest.raises(ValueError, match="at least one block"):
        gfem.cascade([], {})
    with pytest.raises(ValueError, match="map each section name"):
        gfem.cascade([a, b], ["s"])  # ty: ignore[invalid-argument-type]


def test_a_missing_polarity_is_a_value_error_and_non_differentiable_under_grad():
    g1, g2 = _gammas(0.0)
    a, b = fresnel_pair(g1, g2)
    bare = _with_mode(b, "s@1", polarity=None)
    with pytest.raises(ValueError, match=r"kept_modes\(\.\.\., polarity=") as plain:
        gfem.cascade([a, bare], {"s": 1e-6})
    assert type(plain.value) is ValueError

    def objective(length):
        return jnp.abs(gfem.cascade([a, bare], {"s": length}).s.values[0, 0, 1])

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"kept_modes\(\.\.\., polarity="
    ):
        jax.grad(objective)(1e-6)


def test_the_field_gauge_is_a_later_step():
    a, b = _fresnel()
    a = _with_mode(a, "s@1", polarity=None, e_field=_Plane((1.0, 0.0)))
    b = _with_mode(b, "s@1", polarity=None, e_field=_Plane((-1.0, 0.0)))
    with pytest.raises(NotImplementedError, match="pairing-sign gauge"):
        gfem.cascade([a, b], {"s": 1e-6})


# --- transformations ------------------------------------------------------------


def _slab(loss=0.02):
    g1, g2 = _gammas(loss)
    a, b = fresnel_pair(g1, g2)
    return a, b, g1, g2


def _transmission(a, b, length):
    return jnp.abs(gfem.cascade([a, b], {"s": length}).s["b", "a"][0]) ** 2


def test_jit_equals_eager():
    a, b, *_ = _slab()
    length = 2.2e-6
    eager = gfem.cascade([a, b], {"s": length}).s.values
    jitted = jax.jit(lambda x: gfem.cascade([a, b], {"s": x}).s.values)(length)
    # the same operations compiled differently: a few eps; measured below 1e-15
    np.testing.assert_allclose(jitted, eager, rtol=0, atol=TOL)


def test_vmap_over_the_length_equals_a_loop():
    a, b, *_ = _slab()
    lengths = jnp.linspace(0.0, 4e-6, 7)
    batched = jax.vmap(lambda x: gfem.cascade([a, b], {"s": x}).s.values)(lengths)
    looped = np.stack([gfem.cascade([a, b], {"s": x}).s.values for x in lengths])
    # a batched solve against single ones: a few eps; measured below 1e-15
    np.testing.assert_allclose(batched, looped, rtol=0, atol=TOL)
    result = jax.vmap(lambda x: gfem.cascade([a, b], {"s": x}))(lengths)
    assert result.sections["s"].shape == (7,)


def _airy_slope(g1, g2, length):
    """d|tau|^2/dl at the first frequency, tau' = -gamma tau (1+r^2 p)/(1-r^2 p)."""
    g1, g2 = np.asarray(g1)[0], np.asarray(g2)[0]
    r = (g1 - g2) / (g1 + g2)
    p = np.exp(-2 * g2 * length)
    tau = (1 - r**2) * np.exp(-g2 * length) / (1 - r**2 * p)
    dtau = -g2 * tau * (1 + r**2 * p) / (1 - r**2 * p)
    return 2 * (np.conj(tau) * dtau).real


def test_the_length_gradient_is_the_airy_derivative_and_a_fourth_order_difference():
    a, b, g1, g2 = _slab()
    length = 2.2e-6
    grad = float(jax.grad(lambda x: _transmission(a, b, x))(length))
    exact = _airy_slope(g1, g2, length)
    # the derivative is about 1e6 /m, through the same exponentials:
    # a few hundred eps relative; measured below 1e-15
    assert grad == pytest.approx(exact, rel=1e-13)
    # fourth-order central difference with a step of 0.02/beta_max (the
    # truncation (2 beta h)^4/30 ~ 8e-8 of the slope, round-off 1.5 eps/h
    # ~ 1e-14 relative), beta_max the largest |gamma| of the kept modes
    beta_max = float(max(np.abs(g2).max(), np.abs(evanescent(2)).max()))
    h = 0.02 / beta_max

    def f(x):
        return float(_transmission(a, b, x))

    fd4 = (
        -f(length + 2 * h) + 8 * f(length + h) - 8 * f(length - h) + f(length - 2 * h)
    ) / (12 * h)
    # measured 7.4e-8 (the truncation estimate), gate 4x
    assert fd4 == pytest.approx(grad, rel=3e-7)


def test_the_cascade_is_a_pytree_of_its_arrays():
    a, b, *_ = _slab()
    result = gfem.cascade([a, b], {"s": 1e-6})
    leaves, tree = jax.tree.flatten(result)
    back = jax.tree.unflatten(tree, leaves)
    np.testing.assert_array_equal(back.s.values, result.s.values)
    assert back.s.ports == result.s.ports
    assert back.polarization == "Ez"
    with pytest.raises(Exception, match="cannot assign"):
        result.sections = {}  # ty: ignore[invalid-assignment]
