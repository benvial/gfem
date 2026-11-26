r"""`gfem.optomechanical_overlap`: the optomechanical overlap of a displacement.

The optical section is a layered slab (a 0.6 um film of index 3.48 between
oxide, layers along y, side walls ``left`` and ``right``); the displacement is an
exact field on a native 2D mesh of the same extent, built by
`optomechanics_cases.displacement` (the test helper of this file: an analytic
field on a 2D mesh as a `gfem.FieldMap`). The film's dilation
:math:`u = (0, y - y_c, 0)` has a closed form from the exact slab dispersion: it
moves the two interfaces, :math:`\delta n_\mathrm{MB} = d\,\partial
n_\mathrm{eff}/\partial d`, and strains the film by :math:`S_{yy} = 1`,
:math:`\delta n_\mathrm{PE} = \sum_i \partial n_\mathrm{eff}/\partial
\varepsilon_{ii}\,\Delta\varepsilon_{ii}`, :math:`\Delta\varepsilon_{ii} =
-n^4 p_{iy}`. PEC side walls keep the TE mode the 1D slab's; natural walls leave
the TM mode (the second mode of that section, the first being a hybrid with
:math:`x` structure) the 1D slab's.
"""

import dataclasses

import electro_optic_cases as eo
import finite_differences as fd
import jax
import jax.numpy as jnp
import numpy as np
import optomechanics_cases as oc
import pytest

import gfem
from gfem._field import FieldMap

D = oc.FILM * 1e-6
EPS = oc.N_FILM**2
P = oc.cubic_p()
FREQUENCY = eo.FREQUENCY
#: The slab mesh's edge-length factors of the closed-form checks (film edge
#: 0.08 factor um) and the rough one of the identities and refusals.
COARSE, FINE, ROUGH = 2.0, 1.0, 3.0


def _film(**kw):
    return gfem.Material(eps_r=EPS, photoelastic_tensor=P, **kw)


def _overlap(model, u, **kw):
    return gfem.optomechanical_overlap(model, u, port="p", frequency=FREQUENCY, **kw)


@pytest.fixture(scope="module")
def mechanical(tmp_path_factory):
    return oc.mechanical_mesh(tmp_path_factory.mktemp("om") / "mechanical.msh")


@pytest.fixture(scope="module")
def slabs(tmp_path_factory):
    """The slab meshes by edge factor, built on first use."""
    path = tmp_path_factory.mktemp("om_slab")
    made = {}

    def get(factor):
        if factor not in made:
            made[factor] = oc.slab(path / f"slab{factor}.msh", factor)
        return made[factor]

    return get


@pytest.fixture(scope="module")
def dilation(mechanical):
    """The film's dilation about its middle: u_y = y - y_c on the film alone."""
    return oc.displacement(
        mechanical,
        ("film",),
        gradients=[(0, 0), (0, 1), (0, 0)],
        offsets=(0.0, -D / 2, 0.0),
    )


@pytest.fixture(scope="module")
def rough(slabs):
    return slabs(ROUGH)


# --- closed form ---------------------------------------------------------------------

#: Polarization -> (which mode of the section, the film's strained permittivity
#: components with the photoelastic constant that acts on S_yy = 1, whether the
#: side walls are PEC).
_POLARIZATIONS = {
    "TE": (1, ((0, oc.P12),), True),
    "TM": (2, ((1, oc.P11), (2, oc.P12)), False),
}
#: Gates, 2x the errors measured on this mesh pair (a relative error of the
#: overlap's parts, an absolute one of n_eff), per polarization and edge factor.
#: Measured, factor 2 then 1: TE n_eff 4.6e-4, 3.1e-5; moving boundary 1.3e-2,
#: 1.3e-3; photoelastic 5.7e-4, 3.9e-5. TM n_eff 5.0e-5, 4.4e-6; moving boundary
#: 2.6e-3, 2.7e-4; photoelastic 1.6e-4, 1.2e-5.
_GATES = {
    ("TE", COARSE): {"n_eff": 1e-3, "moving_boundary": 3e-2, "photoelastic": 1.2e-3},
    ("TE", FINE): {"n_eff": 7e-5, "moving_boundary": 3e-3, "photoelastic": 1e-4},
    ("TM", COARSE): {"n_eff": 1e-4, "moving_boundary": 6e-3, "photoelastic": 4e-4},
    ("TM", FINE): {"n_eff": 1e-5, "moving_boundary": 6e-4, "photoelastic": 3e-5},
}


def _exact(polarization):
    eps = (EPS,) * 3
    _, strained, _ = _POLARIZATIONS[polarization]
    return {
        "n_eff": oc.slab_n_eff(eps, D, polarization),
        "moving_boundary": D * oc.thickness_derivative(eps, D, polarization),
        "photoelastic": sum(
            oc.eps_derivative(eps, D, polarization, axis) * (-(EPS**2) * p)
            for axis, p in strained
        ),
    }


@pytest.fixture(scope="module")
def measured(slabs, dilation):
    """The overlaps of the film dilation on the closed-form meshes, by (pol, factor)."""
    made = {}

    def get(polarization, factor):
        key = polarization, factor
        if key not in made:
            mode, _, pec = _POLARIZATIONS[polarization]
            model = oc.model(slabs(factor), _film(), pec=pec)
            made[key] = _overlap(model, dilation, mode=mode)
        return made[key]

    return get


def _errors(got, polarization):
    exact = _exact(polarization)
    return {
        "n_eff": abs(float(got.n_eff) - exact["n_eff"]),
        "moving_boundary": abs(got.moving_boundary / exact["moving_boundary"] - 1),
        "photoelastic": abs(got.photoelastic / exact["photoelastic"] - 1),
    }


@pytest.mark.parametrize("factor", [COARSE, FINE], ids=["coarse", "fine"])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_the_film_dilation_matches_the_slab_closed_form(measured, polarization, factor):
    # The film (index 3.48, 0.6 um) dilated about its middle: d -> d (1 + alpha),
    # the oxide fixed, and S_yy = alpha. First order in alpha at the fixed
    # frequency, so dn/dalpha = d dn/dd + sum_i dn/d(eps_ii) (-n^4 p_iy). The
    # closed form is the exact dispersion (a root of the transcendental
    # equation), differentiated by fourth-order central differences (truncation
    # ~1e-12, round-off ~1e-13); the gap is the mode's discretization. Gates
    # are 2x the measured errors of `_GATES`.
    got = measured(polarization, factor)
    gate = _GATES[polarization, factor]
    for name, error in _errors(got, polarization).items():
        assert error < gate[name], (polarization, factor, name, error)
    # the parts add, and both are real for a real displacement
    np.testing.assert_allclose(got.total, got.moving_boundary + got.photoelastic)
    assert abs(got.moving_boundary.imag) < 1e-12 * abs(got.moving_boundary)
    assert abs(got.photoelastic.imag) < 1e-12 * abs(got.photoelastic)
    assert got.mode == _POLARIZATIONS[polarization][0]


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_halving_the_edge_length_cuts_every_error_at_least_fivefold(
    measured, polarization
):
    # Measured ratios (factor 2 over factor 1): TE 15 (n_eff), 9.8 (moving
    # boundary), 15 (photoelastic); TM 11, 9.6, 13. The mesh-limited errors
    # fall as a power of the edge, ~h^3 here; the gate is about half the
    # smallest measured ratio.
    coarse = _errors(measured(polarization, COARSE), polarization)
    fine = _errors(measured(polarization, FINE), polarization)
    for name in coarse:
        assert coarse[name] / fine[name] > 5, (polarization, name)


# --- identities ----------------------------------------------------------------------


def test_the_moving_boundary_part_equals_the_gradient_through_the_node_map(
    slabs, dilation
):
    # The public pattern: jax.grad of n_eff through mesh.with_nodes(nodes + a W),
    # W the film's dilation written on every node from the analytic field (the
    # nodes with y in the film, front and back alike), not through FieldMap
    # sampling. Same solve and same adjoint, contracted in another order (the
    # chain through `with_nodes` against one gradient dotted with W), so the two
    # differ by round-off of sums of ~1e3 terms: measured 1e-16 relative on one
    # thread; the gate 1e-10 covers the eigen-solve's thread-order noise (6e-14
    # relative on 8 threads) with margin.
    mesh = slabs(COARSE)
    model = oc.model(mesh, _film())
    nodes = np.asarray(mesh.nodes)
    y = nodes[:, 1]
    in_film = np.abs(y - D / 2) <= D / 2 + 1e-12
    w = np.zeros_like(nodes)
    w[in_film, 1] = y[in_film] - D / 2

    def n_eff(a):
        moved = model.replace(mesh=mesh.with_nodes(jnp.asarray(nodes + a * w)))
        out = gfem.solve(moved, gfem.PortMode("p", FREQUENCY))
        return jnp.real(out.n_eff[0])

    expected = float(jax.grad(n_eff)(0.0))
    got = _overlap(model, dilation).moving_boundary
    np.testing.assert_allclose(got.real, expected, rtol=1e-10, atol=0)
    assert abs(got.imag) < 1e-12 * abs(got.real)


def test_a_dilation_of_the_whole_nondispersive_section_gives_n_g_minus_n_eff(
    slabs, mechanical
):
    # u = (x, y, 0) everywhere scales the section about the origin, the walls
    # included: the discrete problem on the scaled mesh at omega is the problem
    # on the original at omega (1 + alpha), so d n_eff / d alpha = omega d n_eff /
    # d omega = n_g - n_eff for a section of constant permittivities, with n_g
    # the exact derivative of the discrete gamma. Holds to the two derivatives'
    # round-off (a shape adjoint and a frequency derivative), measured 6e-15
    # relative; the gate 1e-10 is the thread-order noise of the eigen-solve (6e-14)
    # with margin.
    everywhere = oc.displacement(
        mechanical,
        ("below", "film", "above"),
        gradients=[(1, 0), (0, 1), (0, 0)],
    )
    model = oc.model(slabs(COARSE), gfem.Material(eps_r=EPS))
    got = _overlap(model, everywhere)
    assert got.photoelastic == 0
    np.testing.assert_allclose(
        got.moving_boundary.real, float(got.n_g) - float(got.n_eff), rtol=1e-10, atol=0
    )


def _linear(u, v, a, b):
    """``a u + b v`` of two displacements on the same mesh (complex scalars)."""
    return FieldMap(
        {
            c: dataclasses.replace(
                u[c], coefficients=a * u[c].coefficients + b * v[c].coefficients
            )
            for c in "xyz"
        },
        label="component",
    )


def test_the_overlap_is_linear_in_the_displacement(rough, mechanical, dilation):
    # A second displacement with in-plane motion and shear strain on the film
    # (the dilation's support, so the coefficients add); complex weights. Every
    # step is linear in the coefficients, so the sides differ by round-off only
    # (measured 8e-16 relative); the gate 1e-10 is the eigen-solve's thread-order
    # noise (6e-14 relative) with margin.
    other = oc.displacement(
        mechanical,
        ("film",),
        gradients=[(0.3, 2.0), (-0.4, 0.5), (1.0, 0.2)],
        offsets=(1e-9, -2e-9, 3e-9),
    )
    model = oc.model(rough, _film())
    a, b = 0.7 - 0.2j, -1.3 + 0.9j
    first, second = _overlap(model, dilation), _overlap(model, other)
    both = _overlap(model, _linear(dilation, other, a, b))
    for name in ("moving_boundary", "photoelastic"):
        want = a * getattr(first, name) + b * getattr(second, name)
        np.testing.assert_allclose(getattr(both, name), want, rtol=1e-10, atol=0)
        assert abs(getattr(second, name)) > 0


def test_a_displacement_of_the_cladding_alone_moves_the_boundary_and_strains_nothing(
    rough, mechanical
):
    # u_y = const on the cladding below: no strain in the film, and only the
    # interface nodes at y = 0 are in the support (the support's boundary counts),
    # which move the lower interface
    below = oc.displacement(
        mechanical, ("below",), gradients=[(0, 0)] * 3, offsets=(0.0, 1e-9, 0.0)
    )
    got = _overlap(oc.model(rough, gfem.Material(eps_r=EPS)), below)
    assert got.photoelastic == 0
    assert got.moving_boundary != 0


# --- gradients -----------------------------------------------------------------------


def _field_model(mesh, p):
    return oc.model(
        mesh,
        gfem.MaterialField(eps_r=jnp.asarray(EPS), photoelastic_tensor=jnp.asarray(p)),
    )


def _objective(total):
    """``|total|^2``: real, and it reads both parts of the complex overlap."""
    return jnp.abs(total) ** 2


def test_the_photoelastic_tensor_gradient_matches_finite_differences(rough, dilation):
    # The overlap is affine in the tensor (the moving-boundary part has none), so
    # the objective is quadratic in the perturbation and the fourth-order
    # difference has no truncation (J^(5) = 0); what is left is the mode solve's
    # round-off, 6e-14 relative, carried as 1.5 * 6e-14 / h. The perturbation
    # 0.03 q moves p by ~30% of p12 and the objective by order 1 over h = 1, so
    # the round-off sits at ~1e-13 against the gate of 1e-6 (measured 2e-16).
    rng = np.random.default_rng(3)

    def objective(q):
        model = _field_model(rough, P + 0.03 * q)
        return _objective(_overlap(model, dilation).total)

    q0 = jnp.zeros((6, 6))
    (d,) = fd.unit_directions(rng, (6, 6), 1)
    grad = jax.grad(objective)(q0)
    assert float(jnp.abs(grad).max()) > 0
    fd.assert_gradient_matches(objective, q0, [d], step=1.0, grad=grad)


def test_the_gradient_in_the_displacement_matches_finite_differences(rough, dilation):
    # The displacement's coefficients are inputs of the overlap (the mechanical
    # solve's output). The overlap is linear in them, |total|^2 quadratic: no
    # truncation in the fourth-order difference; the round-off is the mode
    # solve's, as above (measured 2e-15), h = 1 against a perturbation of 5e-8
    # of the coefficients.
    rng = np.random.default_rng(8)
    base = {c: dilation[c].coefficients for c in "xyz"}
    model = oc.model(rough, _film())
    noise = {c: jnp.asarray(rng.standard_normal(len(base[c]))) for c in "xyz"}

    def objective(t):
        u = FieldMap(
            {
                c: dataclasses.replace(
                    dilation[c], coefficients=base[c] + 0.05 * t * noise[c] * 1e-6
                )
                for c in "xyz"
            },
            label="component",
        )
        return _objective(_overlap(model, u).total)

    t0 = jnp.zeros(())
    grad = jax.grad(objective)(t0)
    assert abs(float(grad)) > 0
    fd.assert_gradient_matches(objective, t0, [jnp.ones(())], step=1.0, grad=grad)


def test_the_forward_under_grad_equals_the_plain_forward(rough, dilation):
    # the same solves and sums, traced or not: only the order of the floating-point
    # sums differs (the eigen-solve's thread noise is 6e-14 relative)
    def objective(q):
        return _objective(_overlap(_field_model(rough, P + 0.03 * q), dilation).total)

    q = jnp.asarray(np.random.default_rng(4).standard_normal((6, 6)))
    value = jax.value_and_grad(objective)(q)[0]
    np.testing.assert_allclose(value, objective(q), rtol=1e-12, atol=0)


def test_a_traced_optical_permittivity_raises_with_the_stop_gradient_advice(
    rough, dilation
):
    def f(eps):
        film = gfem.MaterialField(eps_r=eps, photoelastic_tensor=jnp.asarray(P))
        return jnp.real(_overlap(oc.model(rough, film), dilation).total)

    with pytest.raises(gfem.NonDifferentiableError, match="optomechanical overlap"):
        jax.grad(f)(jnp.asarray(EPS))


def test_the_optical_node_coordinates_raise_under_grad(rough, dilation):
    def f(nodes):
        moved = gfem.mesh.Mesh(nodes, rough._topology)
        return jnp.real(_overlap(oc.model(moved, _film()), dilation).total)

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"node coordinates.*stop_gradient"
    ):
        jax.grad(f)(jnp.asarray(rough.nodes))


# --- the result ----------------------------------------------------------------------


def test_the_overlap_is_exported_a_pytree_and_has_a_readable_repr(rough, dilation):
    got = _overlap(oc.model(rough, _film()), dilation)
    assert isinstance(got, gfem.OptomechanicalOverlap)
    assert got.port == "p"
    assert got.frequency == FREQUENCY
    leaves, tree = jax.tree_util.tree_flatten(got)
    assert len(leaves) == 4
    again = jax.tree_util.tree_unflatten(tree, leaves)
    assert again.port == "p"
    assert again.total == got.total
    assert float(got.n_g) > float(got.n_eff) > eo.N_OX
    text = repr(got)
    assert text.startswith("OptomechanicalOverlap(port 'p', mode 1")
    assert "moving_boundary" in text


# --- refusals ------------------------------------------------------------------------


def test_a_model_that_is_no_model_or_a_displacement_that_is_no_field_map_raises(
    rough, dilation
):
    model = oc.model(rough, _film())
    with pytest.raises(TypeError, match=r"optomechanical_overlap needs a gfem\.Model"):
        _overlap(dilation, dilation)
    for bad in (None, dilation["x"], 3.0):
        with pytest.raises(TypeError, match=r"takes a displacement gfem\.FieldMap"):
            _overlap(model, bad)


def test_a_displacement_without_a_component_or_on_a_3d_mesh_raises(rough, dilation):
    model = oc.model(rough, _film())
    short = FieldMap({c: dilation[c] for c in "xy"}, label="component")
    with pytest.raises(ValueError, match=r"needs 'x', 'y' and 'z'"):
        _overlap(model, short)
    bad = dataclasses.replace(dilation["x"], mesh=rough)
    solid = FieldMap({**dict(dilation), "x": bad}, label="component")
    with pytest.raises(ValueError, match="native 2D mesh"):
        _overlap(model, solid)


def test_an_axial_wavenumber_other_than_zero_raises(rough, dilation):
    model = oc.model(rough, _film())
    guided = FieldMap(dict(dilation), label="component", wavenumber=1e5)
    with pytest.raises(ValueError, match=r"optomechanical_overlap.*q = 100000"):
        _overlap(model, guided)


def test_a_model_on_a_2d_mesh_raises(mechanical, dilation):
    flat = gfem.Model(
        mechanical, materials={r: gfem.Material() for r in ("below", "film", "above")}
    )
    with pytest.raises(NotImplementedError, match=r"optomechanical_overlap.*2D mesh"):
        _overlap(flat, dilation)


def test_a_port_that_is_not_a_wave_port_or_a_sequence_of_frequencies_raises(
    rough, dilation
):
    model = oc.model(rough, _film())
    with pytest.raises(ValueError, match="no port named 'q'"):
        gfem.optomechanical_overlap(model, dilation, port="q", frequency=FREQUENCY)
    with pytest.raises(ValueError, match="one frequency"):
        gfem.optomechanical_overlap(
            model, dilation, port="p", frequency=[FREQUENCY, 2 * FREQUENCY]
        )


@pytest.mark.parametrize(
    ("loss", "match"),
    [({"tan_delta": 1e-3}, "tan_delta"), ({"conductivity": 1.0}, "conductivity")],
)
def test_a_lossy_section_raises(rough, dilation, loss, match):
    # a lossy region without any photoelastic tensor still refuses: the weights
    # of the first-order shift hold for a lossless section only
    model = oc.model(rough, gfem.Material(eps_r=EPS, **loss))
    with pytest.raises(NotImplementedError, match=rf"optomechanical_overlap.*{match}"):
        _overlap(model, dilation)


def _translated(u, shift):
    """``u`` on its mesh translated by ``shift`` m along x."""
    mesh = u["x"].mesh
    moved = mesh.with_nodes(np.asarray(mesh.nodes) + np.array([shift, 0.0]))
    return FieldMap(
        {c: dataclasses.replace(u[c], mesh=moved) for c in "xyz"}, label="component"
    )


def test_neither_a_photoelastic_tensor_nor_a_node_in_the_support_raises_naming_the_port(
    rough, dilation
):
    # the film's displacement moved 1 mm away from the section: no node of the
    # section is in its support, and the model has no photoelastic tensor
    away = _translated(dilation, 1e-3)
    plain = oc.model(rough, gfem.Material(eps_r=EPS))
    with pytest.raises(
        ValueError,
        match=r"optomechanical_overlap: no element of wave port 'p'.*photoelastic",
    ):
        _overlap(plain, away)


def test_an_element_with_a_tensor_outside_the_support_raises_naming_the_region(
    rough, mechanical
):
    below = oc.displacement(
        mechanical, ("below",), gradients=[(0, 0)] * 3, offsets=(0.0, 1e-9, 0.0)
    )
    with pytest.raises(
        ValueError, match=r"wave port 'p'.*reading the strain.*region 'film'"
    ):
        _overlap(oc.model(rough, _film()), below)
    # and one off the mechanical mesh altogether
    with pytest.raises(ValueError, match=r"reading the strain.*outside the mesh"):
        _overlap(oc.model(rough, _film()), _translated(below, 1e-3))


def test_a_port_section_off_the_plane_z_const_raises_before_the_mode_solve(
    rough, dilation
):
    # the guide along x: the "left" face is normal to x
    oxide = gfem.Material(eps_r=eo.N_OX**2)
    sideways = gfem.Model(
        rough,
        materials={"below": oxide, "film": _film(), "above": oxide},
        boundaries=[gfem.PEC("front")],
        ports=[gfem.WavePort("p", "left")],
    )
    with pytest.raises(ValueError, match=r"optomechanical_overlap.*plane z = const"):
        _overlap(sideways, dilation)
