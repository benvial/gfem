r"""The pairing-sign gauge of `gfem.cascade` on blocks solved with FEM.

The blocks are the two halves of a width step in a box (`cascade_cases`),
solved at order 2 on a coarse structured mesh with :math:`N = 6` kept modes
per plane, the last below cutoff. A block's mode carries an arbitrary sign:
the cascade fixes B's against A's, from the polarity arrows or, without them,
from the port fields and the pseudo-wave scale :math:`\kappa`.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cascade_cases import (
    F_STEP,
    X_A,
    X_B,
    polarity_points,
    solve_step,
    step_meshes,
    step_models,
)

import gfem
from gfem import mode_cascade

N = 6
RANKS = range(1, N + 1)
LENGTH = 0.5e-6  # m, the section between the two planes of the step
OMEGA = 2 * np.pi * F_STEP


@pytest.fixture(scope="module")
def conforming(tmp_path_factory):
    return step_meshes(tmp_path_factory.mktemp("conforming"))


@pytest.fixture(scope="module")
def shifted(tmp_path_factory):
    return step_meshes(tmp_path_factory.mktemp("shifted"), shift=0.3)


def _unpolarized(block):
    """The same block without its polarity arrows: the gauge needs the fields."""
    modes = {
        k: dataclasses.replace(v, polarity=None) for k, v in block.port_modes.items()
    }
    return dataclasses.replace(block, port_modes=modes)


def _s(blocks):
    return np.asarray(gfem.cascade(list(blocks), {"s": LENGTH}).s.values)


# --- kappa --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("polarization", "lossy"),
    [("Ez", False), ("Hz", False), ("Ez", True)],
    ids=["Ez", "Hz", "Ez-lossy-open"],
)
def test_kappa_squared_is_the_pairing_of_the_solve(conforming, polarization, lossy):
    models = step_models(
        conforming, n=N, absorbing=lossy, tan_delta=0.05 if lossy else 0.0
    )
    block, _ = solve_step(models, polarization)
    for m in RANKS:
        mode = block.port_modes[f"s@{m}"]
        field = mode.e_field if mode.e_field is not None else mode.h_field
        space, basis, coords = field._space, field._basis, field._coords
        gamma = np.asarray(mode.gamma) * space.section.scale
        # the solve's expression: pairing factor times r_kk = phi reaction c
        reaction = space.port_columns(basis)["reaction"]
        phi = space.phi(coords, gamma, np.array([OMEGA]))
        r_kk = np.einsum("fi,ij,fj->f", phi, reaction, coords)
        engine = space.pairing_factor(np.array([OMEGA]))[:, 0] * r_kk
        kappa = mode_cascade._kappa(mode)
        # both sum the same few hundred products, in another association;
        # measured relative differences are 0 to 7e-16 (3 eps) over the 18
        # modes, and the sum does not cancel (it is a power of a mode's
        # modulus), so the gate is 1e-14, 45 eps
        np.testing.assert_allclose(kappa**2, engine, rtol=1e-14, atol=0)


# --- the gauge ----------------------------------------------------------------


@pytest.mark.parametrize("which", ["conforming", "shifted"])
def test_the_pairing_gauge_equals_the_polarity_gauge_and_the_flips_are_real(
    request, which
):
    meshes = request.getfixturevalue(which)
    # B's modes 2, 3 and 5 point down: their raw fields are minus A's
    flipped = {m: (-1 if m in (2, 3, 5) else 1) for m in RANKS}
    up = polarity_points(X_A, RANKS)
    down = {
        m: (pt, "+z" if flipped[m] > 0 else "-z")
        for m, (pt, _) in polarity_points(X_B, RANKS).items()
    }
    blocks = solve_step(step_models(meshes, n=N, polarity=(up, down)))
    reference = solve_step(step_models(meshes, n=N))
    by_polarity = _s(blocks)
    by_pairing = _s([_unpolarized(b) for b in blocks])
    baseline = _s(reference)
    signs = [
        mode_cascade._pairing_sign(
            "s",
            m,
            _unpolarized(blocks[0]).port_modes[f"s@{m}"],
            _unpolarized(blocks[1]).port_modes[f"s@{m}"],
        )
        for m in RANKS
    ]
    assert signs == [flipped[m] for m in RANKS]
    # the three differ by the rounding of the solves of one linear system (a
    # port mode's sign enters only through its load and functional): measured
    # 1e-16 to 3e-16 on S of modulus up to 0.88; the reciprocity asymmetry of
    # the same kind of solve measures 2.5e-15 (12 eps), so the gate is 1e-13,
    # 40 times that
    np.testing.assert_allclose(by_pairing, baseline, rtol=0, atol=1e-13)
    np.testing.assert_allclose(by_polarity, baseline, rtol=0, atol=1e-13)


@pytest.fixture(scope="module")
def plain(conforming):
    """The two blocks without polarity, solved once."""
    return solve_step(step_models(conforming, n=N))


def _swap_fields(block, i, j):
    """Block with the port fields of modes ``i`` and ``j`` exchanged (labels stay)."""
    modes = dict(block.port_modes)
    a, b = modes[f"s@{i}"], modes[f"s@{j}"]
    modes[f"s@{i}"] = dataclasses.replace(a, e_field=b.e_field)
    modes[f"s@{j}"] = dataclasses.replace(b, e_field=a.e_field)
    return dataclasses.replace(block, port_modes=modes)


def test_swapped_modes_are_refused(plain):
    a, b = plain
    # modes 2 and 3 of B swapped: an Ez guide's modes solve a Sturm-Liouville
    # problem with unit weight, so different modes are orthogonal under the
    # unconjugated integral (measured overlap 2e-16), far from the 0.95 of one mode
    with pytest.raises(ValueError, match="do not carry the same mode 2") as err:
        gfem.cascade([a, _swap_fields(b, 2, 3)], {"s": LENGTH})
    assert "section 's'" in str(err.value)


def test_polarity_points_at_different_positions_across_a_node_disagree(conforming):
    # mode 2 has its node near y = 1.2 (measured on the field): read at 0.5 in A
    # and at 1.5 in B, the polarity says the fields agree and the pairing
    # says they are opposite
    up = polarity_points(X_A, RANKS)
    up[2] = ((X_A * 1e-6, 0.5e-6), "+z")
    also_up = polarity_points(X_B, RANKS)
    also_up[2] = ((X_B * 1e-6, 1.5e-6), "+z")
    a, b = solve_step(step_models(conforming, n=N, polarity=(up, also_up)))
    with pytest.raises(ValueError, match="same transverse position") as err:
        gfem.cascade([a, b], {"s": LENGTH})
    assert "mode 2" in str(err.value)
    assert "section 's'" in str(err.value)
    # the same position on both ends agrees
    aligned = polarity_points(X_B, RANKS)
    aligned[2] = ((X_B * 1e-6, 0.5e-6), "+z")
    a, b = solve_step(step_models(conforming, n=N, polarity=(up, aligned)))
    gfem.cascade([a, b], {"s": LENGTH})


def test_a_gauge_that_flips_across_the_band_or_is_undetermined_is_refused(
    plain, monkeypatch
):
    a, b = plain
    real = mode_cascade._kappa

    def quarter_turn(mode):
        k = real(mode)
        # B's scale turned by 90 degrees: r kappa_A / kappa_B is imaginary
        return k * 1j if mode is b.port_modes["s@1"] else k

    monkeypatch.setattr(mode_cascade, "_kappa", quarter_turn)
    with pytest.raises(ValueError, match="gauge undetermined"):
        gfem.cascade([a, b], {"s": LENGTH})

    def flipping(mode):
        k = real(mode)
        if mode is b.port_modes["s@1"]:
            return k * np.array([1.0, -1.0])  # opposite signs at the two frequencies
        return k

    monkeypatch.setattr(mode_cascade, "_kappa", flipping)
    with pytest.raises(ValueError, match="gauge changes sign across the band"):
        gfem.cascade([a, b], {"s": LENGTH})


# --- a lossy open step under jax.value_and_grad ----------------------------------

EPS_DESIGN = 4.0
# kept modes of the open guide: the same six, the last below cutoff
LOSSY = {"n": N, "tan_delta": 0.05, "absorbing": True}


def _transmission(meshes, eps, *, polarity):
    pol = (
        (polarity_points(X_A, RANKS), polarity_points(X_B, RANKS))
        if polarity
        else (None, None)
    )
    blocks = solve_step(step_models(meshes, polarity=pol, eps_core=eps, **LOSSY))
    return jnp.abs(gfem.cascade(list(blocks), {"s": LENGTH}).s["out", "in"][0]) ** 2


def test_a_lossy_open_step_with_polarity_differentiates_like_the_pairing_gauge(
    conforming,
):
    value, grad = jax.value_and_grad(
        lambda eps: _transmission(conforming, eps, polarity=True)
    )(EPS_DESIGN)
    concrete = float(_transmission(conforming, EPS_DESIGN, polarity=False))
    # the traced solve and the concrete one are the same linear systems: the
    # values differ by their rounding, measured 3e-16 on 0.17 (1.5 eps); the
    # gate is the 1e-13 of the other solves here
    assert abs(float(value) - concrete) <= 1e-13

    # FD4 of the concrete cascade (the pairing gauge, no polarity): the
    # error is (h^4/30) f + 1.5 eta / h with eta the rounding of one
    # evaluation, 1e-15 here (the value differences above); at h = 1e-3 the
    # truncation (f varies on the scale of eps_r ~ 1: 3e-15) and the rounding
    # (1.5e-12) leave 2e-12; the measured difference is 2e-13
    h = 1e-3

    def f(eps):
        return float(_transmission(conforming, eps, polarity=False))

    fd = (
        -f(EPS_DESIGN + 2 * h)
        + 8 * f(EPS_DESIGN + h)
        - 8 * f(EPS_DESIGN - h)
        + f(EPS_DESIGN - 2 * h)
    ) / (12 * h)
    assert abs(float(grad) - fd) <= 2e-12
    assert abs(fd) > 1e-2  # the check is not of two zeros
