r"""`gfem.optomechanical_coupling`: the vacuum coupling of mechanical modes.

The optical section is the layered slab of `test_optomechanics` with a
photoelastic film and oxide beside it (`optomechanics_cases.embedded`: the
side walls clear the film, whose free ends move along x); the mechanical side
is the film alone, a 2 x 0.6 um rectangle of an isotropic silicon-like solid
suspended in air (every side free), on a native 2D mesh of the same (x, y),
solved by `gfem.MechanicalEigenmode` at :math:`q = 0`. Its modes near 2.15 GHz
are a strong symmetric mode (index 0, :math:`|\tilde g_0|` of order
:math:`10^3`) and a weak one 5 % above it (index 1).

Gradients are read through :math:`|\cdot|^2` of the parts and the total: the
phase gauge of a mode (largest displacement coefficient real and positive)
ties on this mirror-symmetric section, so the sign of :math:`g_0` can flip
between finite difference samples, and :math:`|g_0|^2` is gauge-free.
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
from gfem._field import FieldMap

FREQUENCY = eo.FREQUENCY
EPS = oc.N_FILM**2
P = oc.cubic_p()
#: The mechanical target (Hz) and the slab mesh's edge factor.
TARGET, ROUGH = 2.15e9, 3.0
#: A quality factor of the lossy film.
QUALITY = 50.0
C_BASE = np.asarray(gfem.isotropic_stiffness(oc.E_SI, oc.NU_SI))


#: A film quality factor that varies from element to element, so the lossy
#: mode is complex (a uniform one scales the stiffness and keeps a real shape).
def _quality_field(mesh):
    n = len(mesh.elements("film"))
    return QUALITY + 20.0 * np.sin(np.arange(n))


def _film(**kw):
    return gfem.Material(eps_r=EPS, photoelastic_tensor=P, **kw)


def _coupling(model, result, **kw):
    return gfem.optomechanical_coupling(
        model, result, port="p", frequency=FREQUENCY, **kw
    )


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    path = tmp_path_factory.mktemp("coupling")
    return oc.film_rectangle(path / "film.msh"), oc.embedded(path / "slab.msh", ROUGH)


@pytest.fixture(scope="module")
def optical(meshes):
    return oc.embedded_model(meshes[1], _film())


@pytest.fixture(scope="module")
def modes(meshes):
    """Two mechanical modes of the lossless film at q = 0."""
    return gfem.solve(
        oc.film_model(meshes[0]), gfem.MechanicalEigenmode(TARGET, n_modes=2)
    )


@pytest.fixture(scope="module")
def coupling(optical, modes):
    return _coupling(optical, modes)


# --- the assembly --------------------------------------------------------------------


def test_the_coupling_is_the_overlap_scaled_by_the_zero_point_amplitude(
    coupling, modes
):
    # g~0 = (omega / n_g) dn/dalpha sqrt(hbar Re(Omega) / 4), recomputed here in
    # float arithmetic from the stored overlap, n_g and Omega: the same few
    # products in another order, so equal to a few ulps (gate 1e-14 = 45 eps)
    o = coupling.overlap
    omega = 2 * np.pi * FREQUENCY
    zpf = np.sqrt(scipy.constants.hbar * np.real(np.asarray(modes.omega)) / 4)
    for name in ("moving_boundary", "photoelastic"):
        want = omega / float(o.n_g) * np.asarray(getattr(o, name)) * zpf
        np.testing.assert_allclose(getattr(coupling, name), want, rtol=1e-14, atol=0)
    np.testing.assert_allclose(
        coupling.total, coupling.moving_boundary + coupling.photoelastic, rtol=1e-15
    )
    np.testing.assert_array_equal(coupling.omega_m, modes.omega)
    assert coupling.modes == (0, 1)


@pytest.mark.parametrize("length", [1e-6, 3.7e-4, 1.0])
def test_g0_of_a_length_divides_the_total_by_its_square_root(coupling, length):
    np.testing.assert_allclose(
        coupling.g0(length), np.asarray(coupling.total) / np.sqrt(length), rtol=1e-14
    )


def test_the_overlap_of_each_mode_is_that_of_its_displacement(optical, modes, coupling):
    # the same solves and sums as optomechanical_overlap, in another order of
    # execution: the eigen-solve's thread-order noise is 6e-14 relative, gate 1e-10
    for k in (0, 1):
        single = gfem.optomechanical_overlap(
            optical, modes.displacement(k), port="p", frequency=FREQUENCY
        )
        for name in ("moving_boundary", "photoelastic"):
            np.testing.assert_allclose(
                getattr(coupling.overlap, name)[k],
                getattr(single, name),
                rtol=1e-10,
                atol=0,
            )
        assert float(coupling.overlap.n_g) == pytest.approx(
            float(single.n_g), rel=1e-12
        )
    assert coupling.overlap.mode == single.mode


def test_the_modes_are_selected_by_number(optical, modes, coupling):
    only = _coupling(optical, modes, mechanical_modes=1)
    np.testing.assert_allclose(only.total, coupling.total[1:], rtol=1e-10, atol=0)
    assert only.modes == (1,)
    swapped = _coupling(optical, modes, mechanical_modes=[1, -2])
    assert swapped.modes == (1, 0)
    np.testing.assert_allclose(swapped.total, coupling.total[::-1], rtol=1e-10, atol=0)


def test_one_optical_solve_set_serves_every_mechanical_mode(
    optical, modes, monkeypatch
):
    # the optical work is two port mode solves: the node gradient (one
    # gfem.solve and its backward pass) and the weights rule (one _mode_weights,
    # its own solve); the same count for one mechanical mode as for two
    from gfem import optomechanics

    calls = {"solve": 0, "weights": 0}
    solve, weights = gfem.solve, optomechanics._mode_weights

    def counted_solve(*a, **k):
        calls["solve"] += 1
        return solve(*a, **k)

    def counted_weights(*a, **k):
        calls["weights"] += 1
        return weights(*a, **k)

    monkeypatch.setattr(gfem, "solve", counted_solve)
    monkeypatch.setattr(optomechanics, "_mode_weights", counted_weights)
    _coupling(optical, modes, mechanical_modes=0)
    one = dict(calls)
    calls.update(solve=0, weights=0)
    _coupling(optical, modes)
    assert calls == one
    assert one == {"solve": 1, "weights": 1}


def test_a_mode_stores_one_joule_per_metre_which_the_formula_assumes(modes, meshes):
    # 1/2 (Re Omega)^2 int rho |u|^2 dA = 1 J/m, the stored energy m_alpha =
    # 2 / (Re Omega)^2 of the zero-point amplitude. The integral by the
    # engine's own quadrature on the solve's space; the eigen tolerance (1e-10
    # backward error) bounds the normalization, gate 1e-9
    import ngsolve as ngs

    from gfem._engine.cache import discretization
    from gfem._engine.domain import em_region, h1_space

    model = oc.film_model(meshes[0])
    built = discretization(model.mesh).built
    space = h1_space(model, built, 2, (), physics="mechanical", complex_valued=True)
    for k in (0, 1):
        total = 0.0
        with ngs.TaskManager():
            for axis in "xyz":
                g = ngs.GridFunction(space)
                g.vec.FV().NumPy()[:] = np.asarray(
                    modes.displacement(k)[axis].coefficients
                )
                total += ngs.Integrate(
                    oc.RHO_SI * (g * ngs.Conj(g)).real * built.scale**2,
                    built.mesh,
                    definedon=em_region(model, built, "mechanical"),
                    order=6,
                )
        omega = float(np.real(modes.omega[k]))
        assert 0.5 * omega**2 * total == pytest.approx(1.0, rel=1e-9)


# --- values: reality, sign, loss -----------------------------------------------------


def test_a_lossless_mode_gives_real_values_and_a_flipped_mode_flips_the_sign(
    optical, modes, coupling
):
    # the imaginary part of a lossless mode is the eigenvector's: refined to a
    # backward error of 1e-12 and conditioned by the 5 % gap to the next mode,
    # ~2e-11 relative; the gate 1e-9 is 50x that (measured 1.5e-11 at most)
    total = np.asarray(coupling.total)
    assert np.abs(total.imag).max() < 1e-9 * np.abs(total).max()
    for name in ("moving_boundary", "photoelastic"):
        v = np.asarray(getattr(coupling, name))
        assert np.abs(v.imag).max() < 1e-9 * np.abs(v).max()
    # a mode's sign is its phase convention: -u gives -g0 (linear in u, exactly)
    flipped = dataclasses.replace(
        modes,
        displacement_fields=tuple(
            FieldMap(
                {
                    c: dataclasses.replace(f[c], coefficients=-f[c].coefficients)
                    for c in "xyz"
                },
                label="component",
            )
            for f in modes.displacement_fields
        ),
    )
    other = _coupling(optical, flipped)
    np.testing.assert_allclose(other.total, -np.asarray(coupling.total), rtol=1e-10)
    np.testing.assert_allclose(
        other.g0(1e-3), -np.asarray(coupling.g0(1e-3)), rtol=1e-10
    )
    assert np.abs(total[0]) > 100  # the strong mode really couples


def test_a_lossy_mode_gives_complex_values_with_the_real_part_of_omega(optical, meshes):
    lossy = gfem.solve(
        _lossy(meshes[0]),
        gfem.MechanicalEigenmode(TARGET, n_modes=1),
    )
    assert float(jnp.imag(lossy.omega[0])) > 0
    got = _coupling(optical, lossy)
    # complex displacement: a complex overlap, and the zero-point factor reads
    # Re(Omega) only (the assembly identity of the lossless test)
    assert abs(complex(got.total[0]).imag) > 1e-6 * abs(complex(got.total[0]))
    o = got.overlap
    want = (
        2
        * np.pi
        * FREQUENCY
        / float(o.n_g)
        * np.asarray(o.moving_boundary)
        * np.sqrt(scipy.constants.hbar * float(jnp.real(lossy.omega[0])) / 4)
    )
    np.testing.assert_allclose(got.moving_boundary, want, rtol=1e-14, atol=0)


def test_the_result_is_a_pytree_with_a_readable_repr(coupling):
    leaves, tree = jax.tree_util.tree_flatten(coupling)
    again = jax.tree_util.tree_unflatten(tree, leaves)
    assert again.modes == (0, 1)
    np.testing.assert_array_equal(again.total, coupling.total)
    text = repr(coupling)
    assert text.startswith("OptomechanicalCoupling(port 'p', modes (0, 1)")
    assert "Hz" in text
    assert repr(coupling.overlap).startswith("OptomechanicalOverlap(port 'p', mode 1")


# --- refusals ------------------------------------------------------------------------


def test_a_result_that_is_not_a_mechanical_result_raises(optical, modes):
    with pytest.raises(TypeError, match=r"takes a gfem\.MechanicalEigenmodeResult"):
        _coupling(optical, modes.displacement(0))
    with pytest.raises(TypeError, match=r"optomechanical_coupling needs a gfem\.Model"):
        _coupling(modes, modes)


def test_an_axial_wavenumber_other_than_zero_raises(optical, meshes):
    guided = gfem.solve(
        oc.film_model(meshes[0]), gfem.MechanicalEigenmode(TARGET, wavenumber=1e5)
    )
    with pytest.raises(ValueError, match=r"optomechanical_coupling.*q = 100000"):
        _coupling(optical, guided)


def test_modes_out_of_range_or_empty_raise(optical, modes):
    with pytest.raises(IndexError, match="Mode 2 does not exist"):
        _coupling(optical, modes, mechanical_modes=2)
    with pytest.raises(IndexError, match="Mode -3 does not exist"):
        _coupling(optical, modes, mechanical_modes=[0, -3])
    with pytest.raises(ValueError, match="mechanical_modes is empty"):
        _coupling(optical, modes, mechanical_modes=[])
    for bad in ([0.5], True, [False]):
        # a bool is no mode number, though Python counts it an int
        with pytest.raises(TypeError, match="takes an int or a sequence"):
            _coupling(optical, modes, mechanical_modes=bad)


@pytest.mark.parametrize("length", [0.0, -1e-3])
def test_a_length_that_is_not_positive_raises(coupling, length):
    with pytest.raises(ValueError, match="positive length"):
        coupling.g0(length)


def test_a_model_on_a_2d_mesh_raises(meshes, modes):
    flat = gfem.Model(meshes[0], materials={"film": gfem.Material()})
    with pytest.raises(NotImplementedError, match=r"optomechanical_coupling.*2D mesh"):
        _coupling(flat, modes)


def test_no_photoelastic_tensor_and_no_node_in_the_support_raises(meshes, modes):
    away = dataclasses.replace(
        modes,
        displacement_fields=tuple(
            FieldMap(
                {
                    c: dataclasses.replace(
                        f[c],
                        mesh=f[c].mesh.with_nodes(
                            np.asarray(f[c].mesh.nodes) + np.array([1e-3, 0.0])
                        ),
                    )
                    for c in "xyz"
                },
                label="component",
            )
            for f in modes.displacement_fields
        ),
    )
    plain = oc.embedded_model(meshes[1], gfem.Material(eps_r=EPS))
    with pytest.raises(ValueError, match=r"optomechanical_coupling: no element"):
        _coupling(plain, away)


def test_a_traced_optical_permittivity_raises_with_the_stop_gradient_advice(
    meshes, modes
):
    def f(eps):
        film = gfem.MaterialField(eps_r=eps, photoelastic_tensor=jnp.asarray(P))
        return jnp.real(_coupling(oc.embedded_model(meshes[1], film), modes).total[0])

    with pytest.raises(gfem.NonDifferentiableError, match="optomechanical coupling"):
        jax.grad(f)(jnp.asarray(EPS))


def test_the_coupling_refuses_jit_by_name(optical, meshes):
    # the coupling reads the optical node gradient and the meshes as numbers
    model = oc.film_model(meshes[0])

    def f(scale):
        result = gfem.solve(
            mc.edited(model, ["film"], stiffness=scale * jnp.asarray(C_BASE)),
            gfem.MechanicalEigenmode(TARGET, n_modes=2),
        )
        return jnp.abs(_coupling(optical, result).total[0])

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"cannot run inside jax\.jit"
    ):
        jax.jit(f)(jnp.asarray(1.0))


def test_a_clustered_mode_raises_under_grad(optical, meshes):
    # modes 5 % apart form a cluster at a cluster_rtol of 0.1: the mechanical
    # solve's backward pass refuses the field of a mode it cannot separate
    options = gfem.SolverOptions(cluster_rtol=0.1)
    model = oc.film_model(meshes[0])

    def f(scale):
        result = gfem.solve(
            _edited(model, stiffness=scale * jnp.asarray(C_BASE)),
            gfem.MechanicalEigenmode(TARGET, n_modes=2),
            options=options,
        )
        assert int(result.cluster[0]) == int(result.cluster[1])
        return jnp.abs(_coupling(optical, result, mechanical_modes=0).total[0])

    # raised in the backward's host callback: jax wraps it
    with pytest.raises(jax.errors.JaxRuntimeError, match="degenerate with another"):
        jax.grad(f)(jnp.asarray(1.0))


# --- gradients -----------------------------------------------------------------------

#: Output groups of `_outputs`: |mb|^2, |pe|^2, |total|^2 per mode (two modes).
GROUPS = {"moving_boundary": [0, 3], "photoelastic": [1, 4], "total": [2, 5]}


def _outputs(c):
    """:math:`|\\cdot|^2` of both parts and the total, per mode: ``(6,)`` of 2 modes."""
    parts = (c.moving_boundary, c.photoelastic, c.total)
    return jnp.stack([jnp.abs(x[k]) ** 2 for k in (0, 1) for x in parts])


def _check(objective, p0, d, step, *, groups=None, inert=()):
    """``jacobian . d`` equals the fourth-order difference of every output.

    As in the mechanical field gradients: an output whose derivative is under
    :math:`10^{-3}` of its group's largest is read against that floor, so
    round-off on a near zero is not a failure.
    """
    fd.assert_jacobian_matches_by_group(
        objective, p0, [d], step, groups or GROUPS, inert=inert
    )


def _edited(model, **values):
    """``model`` with the leaves ``values`` of its film replaced."""
    return mc.edited(model, ["film"], **values)


def _lossy(mesh):
    """The film as a mechanical model with the element-wise quality factor."""
    return _edited(
        oc.film_model(mesh, quality=QUALITY),
        mechanical_quality_factor=_quality_field(mesh),
    )


def test_the_photoelastic_tensor_gradient_matches_finite_differences(meshes, modes):
    # the total is affine in p (the moving-boundary part has none): |.|^2 is
    # quadratic, the fourth-order difference has no truncation; the round-off is
    # the optical mode solve's, 6e-14 relative carried as 1.5 * 6e-14 / h with
    # h = 1 against a perturbation 0.03 q of the tensor (30 % of p12)
    d = fd.unit_directions(np.random.default_rng(3), (6, 6), 1)[0]

    def objective(q):
        model = oc.embedded_model(
            meshes[1],
            gfem.MaterialField(
                eps_r=jnp.asarray(EPS), photoelastic_tensor=jnp.asarray(P + 0.03 * q)
            ),
        )
        return _outputs(_coupling(model, modes))

    # the moving-boundary part does not read p
    _check(
        objective,
        jnp.zeros((6, 6)),
        d,
        1.0,
        groups={"photoelastic": [1, 4], "total": [2, 5]},
        inert=[0, 3],
    )


#: name -> (edit(film model, p), direction(mesh), lossy): a unit of p is 1e10 Pa
#: of the 1.65e11 Pa stiffness scale, 100 kg/m^3 of 2329 or 5 of Q = 50; the
#: step 1e-2 is a relative change of 1e-4 to 1e-3 of the leaf. The truncation
#: of the fourth-order difference is about (10 h)^4 / 30 ~ 3e-10 (the next mode
#: is 5 % away, each order of derivative of the mode grows by ~10), the
#: round-off of the eigenvectors (backward error 1e-12) well below the gate.
LEAVES = {
    "stiffness": (
        lambda m, p: _edited(m, stiffness=C_BASE + 1e10 * p),
        lambda _: mc.symmetric_direction(5),
        False,
    ),
    "mass_density": (
        lambda m, p: _edited(m, mass_density=oc.RHO_SI + 100.0 * p),
        lambda mesh: mc.per_element_direction(mesh, "film", 3),
        False,
    ),
    "mechanical_quality_factor": (
        lambda m, p: _edited(
            m, mechanical_quality_factor=_quality_field(m.mesh) + 5.0 * p
        ),
        lambda mesh: mc.per_element_direction(mesh, "film", 4),
        True,
    ),
}


#: Each case below solves the mechanical modes and the coupling at four samples
#: and runs a jacrev of six backward passes; the unit tier keeps one per path of
#: the gradient (the quality factor of the lossy film, whose complex mode runs
#: every term of the eigenvector adjoint; the photoelastic tensor; the node
#: stretch) and the slow tier the other leaves.
SLOW_LEAVES = ("stiffness", "mass_density")


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=(pytest.mark.slow,) if n in SLOW_LEAVES else ())
        for n in LEAVES
    ],
)
def test_a_mechanical_input_gradient_matches_finite_differences(optical, meshes, name):
    edit, direction, lossy = LEAVES[name]
    model = _lossy(meshes[0]) if lossy else oc.film_model(meshes[0])
    d = direction(meshes[0])

    def objective(p):
        result = gfem.solve(edit(model, p), gfem.MechanicalEigenmode(TARGET, n_modes=2))
        return _outputs(_coupling(optical, result))

    _check(objective, jnp.zeros_like(d), d, 1e-2)


@pytest.mark.slow
def test_the_stiffness_gradient_of_a_lossy_film_matches_finite_differences(
    optical, meshes
):
    model = _lossy(meshes[0])
    d = mc.symmetric_direction(6)

    def objective(p):
        result = gfem.solve(
            _edited(model, stiffness=C_BASE + 1e10 * p),
            gfem.MechanicalEigenmode(TARGET, n_modes=2),
        )
        return _outputs(_coupling(optical, result))

    _check(objective, jnp.zeros_like(d), d, 1e-2)


#: The stretch's finite-difference step: a change of 1e-3 of the mesh's width.
STRETCH_STEP = 1e-3


def _stretch_base(optical, mesh, modes):
    r"""A stretch :math:`s` whose samples :math:`s \pm 2h` keep every reading point.

    The mode is read at fixed points: the strain at the optical quadrature
    points, the displacement at the section's nodes. Within a mechanical
    triangle a reading is smooth; across its edges the strain jumps and the
    displacement's slope does (the field is :math:`H^1`), so a sample that
    moves an edge across a reading point breaks the difference. The points
    are fixed and the triangles scale with :math:`s` along a straight path, so
    a point in the same triangle at both ends of the samples never leaves it.
    Returns the first such :math:`s` from 1.01 up.
    """
    from gfem import _probe, acousto_optic, optomechanics

    u = modes.displacement(0)
    setup = optomechanics._setup(optical, "p", FREQUENCY, optomechanics._COUPLING, u)
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

    for s in 1.01 + 5 * STRETCH_STEP * np.arange(10):
        if np.array_equal(cells(s - 2 * STRETCH_STEP), cells(s + 2 * STRETCH_STEP)):
            return float(s)
    raise AssertionError("every stretch moves a mechanical edge across a point")


def test_a_stretch_of_the_mechanical_nodes_matches_finite_differences(
    optical, meshes, modes
):
    # the mechanical mesh stretched along x about its centre by s + t, s a bit
    # over 1: it keeps covering the optical film (and every optical node that
    # was in the support), so no node enters or leaves it within the 2e-3 of
    # the samples, and no reading point changes triangle (`_stretch_base`)
    mesh = meshes[0]
    materials = oc.film_model(mesh).materials
    base = _stretch_base(optical, mesh, modes)

    def objective(t):
        s = base + t
        nodes = jnp.asarray(mesh.nodes) * jnp.array(
            [s, 1.0, 1.0][: mesh.nodes.shape[1]]
        )
        result = gfem.solve(
            gfem.Model(mesh.with_nodes(nodes), materials=materials),
            gfem.MechanicalEigenmode(TARGET, n_modes=2),
        )
        return _outputs(_coupling(optical, result))

    _check(objective, jnp.asarray(0.0), np.asarray(1.0), STRETCH_STEP)


def test_a_mode_of_a_3d_body_has_no_cross_section_to_overlap(optical, tmp_path):
    body = gfem.solve(
        gfem.Model(
            mc.box(tmp_path / "body.msh", (1.0, 0.8, 0.6), 3),
            materials={"body": mc.material()},
        ),
        gfem.MechanicalEigenmode(3.2e9, clamped=mc.ROLLERS),
    )
    match = (
        "a mode of a 3D body has no cross-section to overlap with a waveguide "
        "mode; these helpers read 2D section modes"
    )
    with pytest.raises(ValueError, match=match):
        _coupling(optical, body)
    # the lossless 3D result is refused for its dimension, not for its missing Q_m
    assert np.imag(body.omega[0]) == 0
    with pytest.raises(ValueError, match=match):
        gfem.brillouin_gain(optical, body, port="p", frequency=FREQUENCY)
