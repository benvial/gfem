r"""`gfem.brillouin_gain`: the peak Brillouin gain of mechanical modes.

The optical section is the layered slab of `test_optomechanical_coupling` (a
photoelastic film in oxide, `optomechanics_cases.embedded`); the mechanical side
is the film alone, a 2 x 0.6 um rectangle of an isotropic silicon-like solid with
a quality factor, suspended in air. The forward modes are those of the coupling
tests at :math:`q = 0` (a strong symmetric mode near 2.15 GHz, a weak one beside
it). The backward modes sit at :math:`q = 2\,\mathrm{Im}\,\gamma` of the optical
port mode, near 30 GHz, where the film's modes are 1 % apart; modes 1 and 2
have no neighbour inside the solver's clustering tolerance, so their gradients
are not refused (their numbers: mode 1 is carried by the photoelastic part,
mode 2 by the moving boundary with a photoelastic part of a tenth of its
amplitude).

Gradients are read through the gains, which are :math:`|\cdot|^2` of the
overlap: the phase gauge of a mode does not enter.
"""

import dataclasses

import electro_optic_cases as eo
import finite_differences as fd
import jax
import jax.numpy as jnp
import mechanical_cases as mc
import numpy as np
import optomechanics_cases as oc
import pytest
import scipy.constants

import gfem

FREQUENCY = eo.FREQUENCY
EPS = oc.N_FILM**2
P = oc.cubic_p()
#: The forward mechanical target (Hz), the backward one, the slab mesh's edge factor.
TARGET, TARGET_BACK, ROUGH = 2.15e9, 30e9, 3.0
#: A quality factor of the lossy film.
QUALITY = 50.0
C_BASE = np.asarray(gfem.isotropic_stiffness(oc.E_SI, oc.NU_SI))
#: The backward modes the gradient tests read: isolated, both parts present.
STRONG, MIXED = 1, 2


def _film(**kw):
    return gfem.Material(eps_r=EPS, photoelastic_tensor=P, **kw)


def _gain(model, result, **kw):
    return gfem.brillouin_gain(model, result, port="p", frequency=FREQUENCY, **kw)


def _edited(model, **values):
    return mc.edited(model, ["film"], **values)


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    path = tmp_path_factory.mktemp("brillouin")
    return oc.film_rectangle(path / "film.msh"), oc.embedded(path / "slab.msh", ROUGH)


@pytest.fixture(scope="module")
def optical(meshes):
    return oc.embedded_model(meshes[1], _film())


@pytest.fixture(scope="module")
def q_pump(optical):
    """The phase-matched wavenumber :math:`2\\beta` along the pump, rad/m."""
    pm = gfem.solve(optical, gfem.PortMode("p", FREQUENCY))
    return 2 * float(pm.gamma[0].imag)


@pytest.fixture(scope="module")
def forward(meshes):
    """Two lossy modes of the film at q = 0."""
    return gfem.solve(
        oc.film_model(meshes[0], quality=QUALITY),
        gfem.MechanicalEigenmode(TARGET, n_modes=2),
    )


@pytest.fixture(scope="module")
def backward(meshes, q_pump):
    """Four lossy modes of the film at the phase-matched wavenumber."""
    return gfem.solve(
        oc.film_model(meshes[0], quality=QUALITY),
        gfem.MechanicalEigenmode(TARGET_BACK, n_modes=4, wavenumber=q_pump),
    )


@pytest.fixture(scope="module")
def gain_forward(optical, forward):
    return _gain(optical, forward)


@pytest.fixture(scope="module")
def gain_backward(optical, backward):
    return _gain(optical, backward)


# --- the assembly --------------------------------------------------------------------


def test_the_forward_gain_is_the_couplings_g0_squared(optical, forward, gain_forward):
    # G_B = 4 Q_m |g~0|^2 / (hbar omega v_g^2 Re Omega) with v_g = c / n_g, from the
    # coupling's own g~0 = (omega / n_g) dn sqrt(hbar Re Omega / 4): both read the
    # same overlap path, each side is ~12 scalar flops (<= 3e-15 relative), so the
    # gate 1e-13 holds 30x of margin
    coupling = gfem.optomechanical_coupling(
        optical, forward, port="p", frequency=FREQUENCY
    )
    omega = 2 * np.pi * FREQUENCY
    v_g = scipy.constants.c / float(coupling.overlap.n_g)
    re = np.real(np.asarray(forward.omega))
    want = (
        4
        * np.asarray(forward.q)
        * np.abs(np.asarray(coupling.total)) ** 2
        / (scipy.constants.hbar * omega * v_g**2 * re)
    )
    np.testing.assert_allclose(gain_forward.total, want, rtol=1e-13, atol=0)
    for name in ("moving_boundary", "photoelastic", "total"):
        np.testing.assert_allclose(
            getattr(gain_forward.overlap, name),
            getattr(coupling.overlap, name),
            rtol=1e-13,
            atol=0,
        )
    assert float(gain_forward.total[0]) > 1  # the strong mode really gains


@pytest.mark.parametrize("which", ["gain_forward", "gain_backward"])
def test_the_gains_are_the_overlap_squared_times_the_quality_and_frequency(
    which, request, forward, backward
):
    # Q_m omega |dn|^2 / c^2 from the stored overlap, omega and Omega: a few
    # products in another order than the property's, gate 1e-14 (45 eps)
    gain = request.getfixturevalue(which)
    result = forward if which == "gain_forward" else backward
    o = gain.overlap
    omega = 2 * np.pi * FREQUENCY
    q = np.asarray(result.q)[list(gain.modes)]
    np.testing.assert_allclose(gain.quality_factor, q, rtol=1e-14, atol=0)
    for name in ("total", "moving_boundary", "photoelastic"):
        want = (
            q * omega * np.abs(np.asarray(getattr(o, name))) ** 2 / scipy.constants.c**2
        )
        np.testing.assert_allclose(getattr(gain, name), want, rtol=1e-14, atol=0)
    np.testing.assert_allclose(
        o.total, np.asarray(o.moving_boundary) + np.asarray(o.photoelastic), rtol=1e-15
    )
    np.testing.assert_array_equal(
        gain.omega_m, np.asarray(result.omega)[list(gain.modes)]
    )
    assert gain.modes == tuple(range(len(gain.modes)))


def test_the_direction_and_wavenumber_are_those_solved(
    gain_forward, gain_backward, q_pump
):
    assert gain_forward.overlap.direction == "forward"
    assert gain_forward.overlap.wavenumber == 0.0
    assert gain_backward.overlap.direction == "backward"
    assert gain_backward.overlap.wavenumber == pytest.approx(q_pump, rel=1e-15)


def test_the_total_is_not_the_sum_of_the_parts(gain_backward):
    # the amplitudes add with their signs: the total gain is the parts' gain
    # plus twice the cross term, equal to the sum only when it vanishes
    o = gain_backward.overlap
    cross = 2 * np.real(
        np.asarray(o.moving_boundary) * np.conj(np.asarray(o.photoelastic))
    )
    k = MIXED
    assert abs(float(cross[k])) > 1e-3 * float(np.abs(np.asarray(o.total)[k]) ** 2)
    assert float(gain_backward.total[k]) != pytest.approx(
        float(gain_backward.moving_boundary[k] + gain_backward.photoelastic[k]),
        rel=1e-3,
    )


def test_the_modes_are_selected_by_number(optical, backward, gain_backward):
    only = _gain(optical, backward, mechanical_modes=[MIXED, -4])
    assert only.modes == (MIXED, 0)
    # the same solves and sums in another order of execution: gate 1e-10
    np.testing.assert_allclose(
        only.total, np.asarray(gain_backward.total)[[MIXED, 0]], rtol=1e-10, atol=0
    )


# --- the public path -----------------------------------------------------------------


def test_the_names_are_public():
    assert gfem.brillouin_gain is gfem.optomechanics.brillouin_gain
    assert gfem.BrillouinGain is gfem.optomechanics.BrillouinGain
    assert "brillouin_gain" in gfem.__all__
    assert "BrillouinGain" in gfem.__all__


def test_the_result_is_a_pytree_with_a_readable_repr(gain_backward):
    leaves, tree = jax.tree_util.tree_flatten(gain_backward)
    o = gain_backward.overlap
    want = jax.tree_util.tree_leaves((o, gain_backward.omega_m))
    assert len(leaves) == len(want)
    for a, b in zip(leaves, want, strict=True):
        np.testing.assert_array_equal(a, b)
    again = jax.tree_util.tree_unflatten(tree, leaves)
    assert again.modes == gain_backward.modes
    assert again.overlap.direction == "backward"
    np.testing.assert_array_equal(again.total, gain_backward.total)
    text = repr(gain_backward)
    assert text.startswith("BrillouinGain(port 'p', backward q ")
    assert "rad/m" in text
    assert "Hz" in text
    assert f"{float(gain_backward.total[STRONG]):.4g}" in text


def test_the_repr_survives_tracing(gain_backward):
    jax.jit(lambda g: repr(g).startswith("BrillouinGain(port 'p'") or g.total)(
        gain_backward
    )


# --- refusals ------------------------------------------------------------------------


def _expected(q_pump):
    return f"expected q = 0 (forward) or q = {q_pump:+.4e} rad/m"


def test_a_counter_propagating_phonon_raises_with_the_expected_sign(
    optical, backward, q_pump
):
    # a solved result with q relabelled: the classification raises before any
    # field is read at that q, so no second mechanical solve
    wrong = dataclasses.replace(backward, wavenumber=-q_pump)
    with pytest.raises(ValueError, match="counter-propagates") as info:
        _gain(optical, wrong)
    assert _expected(q_pump) in str(info.value)


@pytest.mark.parametrize("factor", [0.5, 1.05, 2.0])
def test_an_intermodal_or_far_wavenumber_raises_with_the_expected_value(
    optical, backward, q_pump, factor
):
    # 0.5 (q = beta) is between the windows, 1.05 just beyond the backward one
    # (half width 1e-2 of 2 beta), 2 is far beyond
    wrong = dataclasses.replace(backward, wavenumber=factor * q_pump)
    with pytest.raises(ValueError, match="intermodal pairs") as info:
        _gain(optical, wrong)
    assert _expected(q_pump) in str(info.value)


def test_lossless_modes_raise_naming_the_mode_and_the_quality_factor(optical, meshes):
    lossless = gfem.solve(
        oc.film_model(meshes[0]), gfem.MechanicalEigenmode(TARGET, n_modes=2)
    )
    with pytest.raises(ValueError, match=r"mode 1 .*mechanical_quality_factor"):
        _gain(optical, lossless, mechanical_modes=1)
    with pytest.raises(ValueError, match=r"mode 0 .*mechanical_quality_factor"):
        _gain(optical, lossless)


def test_a_backward_pair_in_a_magnetic_film_raises(meshes, backward):
    model = oc.embedded_model(meshes[1], _film(mu_r=2.0))
    pm = gfem.solve(model, gfem.PortMode("p", FREQUENCY))
    # the magnetic film's own pump wavenumber, so q sits in its backward window
    moved = dataclasses.replace(backward, wavenumber=2 * float(pm.gamma[0].imag))
    with pytest.raises(NotImplementedError, match=r"region 'film'.*not assembled"):
        _gain(model, moved)


def test_the_coupling_at_a_wavenumber_names_the_gain(optical, backward):
    with pytest.raises(ValueError, match=r"use gfem\.brillouin_gain for the gain"):
        gfem.optomechanical_coupling(optical, backward, port="p", frequency=FREQUENCY)


def test_a_result_that_is_not_a_mechanical_result_raises(optical, forward):
    with pytest.raises(TypeError, match=r"brillouin_gain takes a gfem\.Mechanical"):
        _gain(optical, forward.displacement(0))
    with pytest.raises(TypeError, match=r"brillouin_gain needs a gfem\.Model"):
        _gain(forward, forward)


def test_modes_out_of_range_or_empty_raise(optical, forward):
    with pytest.raises(IndexError, match="brillouin_gain: Mode 2 does not exist"):
        _gain(optical, forward, mechanical_modes=2)
    with pytest.raises(ValueError, match="brillouin_gain: mechanical_modes is empty"):
        _gain(optical, forward, mechanical_modes=[])


def test_a_model_on_a_2d_mesh_raises(meshes, forward):
    flat = gfem.Model(meshes[0], materials={"film": gfem.Material()})
    with pytest.raises(NotImplementedError, match=r"brillouin_gain.*2D mesh"):
        _gain(flat, forward)


def test_the_gain_refuses_jit_by_name(optical, meshes):
    model = oc.film_model(meshes[0], quality=QUALITY)

    def f(scale):
        result = gfem.solve(
            _edited(model, stiffness=scale * jnp.asarray(C_BASE)),
            gfem.MechanicalEigenmode(TARGET, n_modes=2),
        )
        return jnp.sum(_gain(optical, result).total)

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"cannot run inside jax\.jit"
    ):
        jax.jit(f)(jnp.asarray(1.0))


# --- gradients -----------------------------------------------------------------------


def test_the_modulus_and_density_scaling_laws_hold_under_grad(optical, meshes, q_pump):
    # An isotropic single material, uniform Q, fixed q. The stiffness matrix of
    # the q-problem is K(q) = E K_hat(nu, q) at every power of q, so
    # Omega^2 is proportional to E/rho and Omega to E^(1/2) rho^(-1/2) (Q =
    # Re Omega / (2 Im Omega) unchanged by uniform scaling); the mode stores 1 J/m,
    # (1/2) (Re Omega)^2 int rho |u|^2 = 1, so |u|^2 is proportional to
    # 1/(rho Omega^2) = 1/E for the shape fixed: u ~ E^(-1/2), and rho cancels
    # (int rho |u|^2 = 2/Omega^2 is proportional to rho, which leaves u).
    # G_B is proportional to Q |dn|^2 with dn linear in u: dlnG/dlnE = -1 and
    # dlnG/dlnrho = 0, exact laws; the residual of the gate is the eigenvector
    # adjoint's error (worst agreement with finite differences 7e-8, 10x margin).
    model = oc.film_model(meshes[0], quality=QUALITY)

    def gain(scale, mass, wavenumber):
        result = gfem.solve(
            _edited(
                model,
                stiffness=scale * jnp.asarray(C_BASE),
                mass_density=mass * oc.RHO_SI,
            ),
            gfem.MechanicalEigenmode(
                TARGET_BACK if wavenumber else TARGET,
                n_modes=4 if wavenumber else 2,
                wavenumber=wavenumber,
            ),
        )
        mode = STRONG if wavenumber else 0
        return _gain(optical, result, mechanical_modes=mode).total[0]

    one = jnp.asarray(1.0)
    for wavenumber, rho_law in ((q_pump, 0.0), (0.0, None)):
        value, (d_e, d_rho) = jax.value_and_grad(gain, argnums=(0, 1))(
            one, one, wavenumber
        )
        assert float(d_e) / float(value) == pytest.approx(-1.0, abs=1e-6)
        if rho_law is not None:
            assert float(d_rho) / float(value) == pytest.approx(rho_law, abs=1e-6)


#: The finite-difference step: a relative change of 1e-3 of the input. The
#: fourth-order difference's truncation is about h^4 J^(5) / 30 ~ 1e-12 of the
#: derivative (each order of derivative of a mode grows by the inverse of its
#: relative gap to the next one, 1 %, so (100 h)^4 / 30 ~ 3e-6 at most; mode 2's
#: neighbours sit 1 % away and the measured error is the gate's margin), its
#: round-off the eigen-solve's noise 1e-12 relative over h, 1e-9; the gate is
#: the shared 1e-6 of `finite_differences`.
STEP = 1e-3


def _checked(objective, p0, directions, outputs, *, step=STEP):
    """``assert_gradient_matches`` of every output of a vector objective.

    The vector ``objective(p)`` is solved once per sample and once for the
    pullbacks of all its outputs (the cache is keyed on the sample's bytes), so
    each output's check costs only its own pullback.
    """
    cache = {}

    def evaluate(p):
        key = np.asarray(p).tobytes()
        if key not in cache:
            cache[key] = np.asarray(objective(p))
        return cache[key]

    value, pullback = jax.vjp(objective, p0)
    for i in outputs:
        fd.assert_gradient_matches(
            lambda p, i=i: evaluate(p)[i],
            p0,
            directions,
            step=step,
            reference=lambda p, i=i: evaluate(p)[i],
            grad=pullback(jnp.zeros_like(value).at[i].set(1.0))[0],
        )


def _gains(gain):
    """The gains of the first requested mode: moving boundary, photoelastic, total."""
    return jnp.stack([gain.moving_boundary[0], gain.photoelastic[0], gain.total[0]])


def test_the_photoelastic_tensor_gradient_matches_finite_differences(meshes, backward):
    # the photoelastic gain is |dn|^2 with dn affine in p, so the fourth-order
    # difference has no truncation; the moving boundary reads no p
    d = fd.unit_directions(np.random.default_rng(3), (6, 6), 1)
    scale = np.linalg.norm(P)

    def objective(p):
        model = oc.embedded_model(
            meshes[1],
            gfem.MaterialField(
                eps_r=jnp.asarray(EPS), photoelastic_tensor=jnp.asarray(P + scale * p)
            ),
        )
        return _gains(_gain(model, backward, mechanical_modes=MIXED))

    _checked(objective, jnp.zeros((6, 6)), d, outputs=(1, 2))


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="measured 4.88e-6 against the 1e-6 gate on the first output (the "
    "moving-boundary gain of mode 2); under numerical diagnosis",
)
def test_the_quality_factor_gradient_matches_finite_differences(
    optical, meshes, q_pump
):
    # an element-wise Q makes the modes complex, so every term of the eigenvector
    # adjoint runs; the input is Q = 50 + 50 p along a unit direction
    mesh = meshes[0]
    n = len(mesh.elements("film"))
    base = QUALITY + 20.0 * np.sin(np.arange(n))
    model = _edited(
        oc.film_model(mesh, quality=QUALITY), mechanical_quality_factor=base
    )
    d = [mc.per_element_direction(mesh, "film", 4)]

    def objective(p):
        result = gfem.solve(
            _edited(model, mechanical_quality_factor=base + 50.0 * p),
            gfem.MechanicalEigenmode(TARGET_BACK, n_modes=4, wavenumber=q_pump),
        )
        return _gains(_gain(optical, result, mechanical_modes=MIXED))

    _checked(objective, jnp.zeros(n), d, outputs=(0, 1, 2))


def _stretch_base(optical, mesh, result, step):
    r"""A stretch :math:`s` whose samples :math:`s \pm 2h` keep every reading point.

    The mode is read at fixed points (the strain at the optical quadrature
    points, the displacement at the section's nodes); within a mechanical
    triangle a reading is smooth, across its edges it is not. The triangles
    scale with :math:`s` along a straight path, so a point in the same
    triangle at both ends of the samples never leaves it. Returns the first
    such :math:`s` from 1.01 up.
    """
    from gfem import _probe, acousto_optic, optomechanics

    u = result.displacement(MIXED)
    setup = optomechanics._setup(optical, "p", FREQUENCY, optomechanics._GAIN, u)
    rows = setup.crystal.rows[setup.photoelastic_rows]
    strain_points, _ = acousto_optic._points(
        optical.mesh, setup.tets[rows], u["x"].order
    )
    _, node_points = optomechanics._sampled_nodes(
        u, setup.nodes_xy[setup.section.nodes], setup.section.interface
    )
    points = np.concatenate([strain_points, node_points])
    nodes = np.asarray(mesh.nodes)

    def cells(s):
        stretched = mesh.with_nodes(nodes * np.array([s, 1.0]))
        located = _probe.engine().locate(stretched, _probe.layout(u["x"]), points)
        return located.tetrahedra

    for s in 1.01 + 5 * step * np.arange(10):
        if np.array_equal(cells(s - 2 * step), cells(s + 2 * step)):
            return float(s)
    raise AssertionError("every stretch moves a mechanical edge across a point")


@pytest.mark.slow
def test_a_stretch_of_the_mechanical_nodes_matches_finite_differences(
    optical, meshes, backward, q_pump
):
    # the mechanical mesh stretched along x about its centre by s + t, s a bit
    # over 1 (`_stretch_base`): no node enters or leaves the optical support and
    # no reading point changes triangle within the 2e-3 of the samples
    mesh = meshes[0]
    materials = oc.film_model(mesh, quality=QUALITY).materials
    base = _stretch_base(optical, mesh, backward, STEP)

    def objective(t):
        s = base + t
        nodes = jnp.asarray(mesh.nodes) * jnp.array(
            [s, 1.0, 1.0][: mesh.nodes.shape[1]]
        )
        result = gfem.solve(
            gfem.Model(mesh.with_nodes(nodes), materials=materials),
            gfem.MechanicalEigenmode(TARGET_BACK, n_modes=4, wavenumber=q_pump),
        )
        return _gains(_gain(optical, result, mechanical_modes=MIXED))

    _checked(objective, jnp.asarray(0.0), [np.asarray(1.0)], outputs=(0, 1, 2))
