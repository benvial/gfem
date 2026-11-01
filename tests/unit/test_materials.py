"""`Material` and `MaterialField`: keyword-only physics values with checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem


def test_material_is_keyword_only():
    with pytest.raises(TypeError):
        gfem.Material(11.7)  # ty: ignore[too-many-positional-arguments]


def test_material_defaults_are_vacuum_and_thermal_properties_are_reserved():
    m = gfem.Material()
    assert (m.eps_r, m.mu_r, m.tan_delta, m.conductivity) == (1.0, 1.0, 0.0, 0.0)
    assert m.axes is None
    assert m.thermal_conductivity is None
    assert m.density is None
    assert m.specific_heat is None


@pytest.mark.parametrize(
    ("kwargs", "word"),
    [
        ({"eps_r": -2.0}, "eps_r"),
        ({"eps_r": 0.0}, "eps_r"),
        ({"eps_r": (2.0, -1.0, 2.0)}, "eps_r"),
        ({"mu_r": 0.0}, "mu_r"),
        ({"tan_delta": -1e-3}, "tan_delta"),
        ({"conductivity": -1.0}, "conductivity"),
    ],
)
def test_material_rejects_active_media(kwargs, word):
    with pytest.raises(ValueError, match="passive") as err:
        gfem.Material(**kwargs)
    assert word in str(err.value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_material_rejects_non_finite_values(bad):
    with pytest.raises(ValueError, match="finite"):
        gfem.Material(eps_r=bad)


def test_material_rejects_non_positive_thermal_values():
    with pytest.raises(ValueError, match="thermal_conductivity"):
        gfem.Material(thermal_conductivity=0.0)
    assert gfem.Material(density=2330.0, specific_heat=700.0).density == 2330.0


def test_anisotropic_material_takes_three_principal_values_and_orthonormal_axes():
    m = gfem.Material(
        eps_r=np.array([9.3, 9.3, 11.5]),
        axes=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    assert m.eps_r == (9.3, 9.3, 11.5)
    assert m.axes == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    hash(m)  # static pytree data must be hashable
    with pytest.raises(ValueError, match="three"):
        gfem.Material(eps_r=(1.0, 2.0))
    with pytest.raises(ValueError, match="orthonormal"):
        gfem.Material(eps_r=(1.0, 2.0, 3.0), axes=((1, 0, 0), (1, 1, 0), (0, 0, 1)))


def test_loss_tangent_takes_three_values_along_the_axes():
    # sapphire: Palace's LossTan is one value per principal axis, like eps_r
    m = gfem.Material(
        eps_r=(9.3, 9.3, 11.5),
        tan_delta=np.array([3e-5, 3e-5, 8.6e-5]),
        axes=((0.8, 0.6, 0), (-0.6, 0.8, 0), (0, 0, 1)),
    )
    assert m.tan_delta == (3e-5, 3e-5, 8.6e-5)
    assert not m.isotropic
    assert not gfem.Material(eps_r=4.0, tan_delta=(0.0, 0.0, 1e-3)).isotropic
    hash(m)
    with pytest.raises(ValueError, match="three"):
        gfem.Material(tan_delta=(1e-3, 2e-3))
    with pytest.raises(ValueError, match="passive") as err:
        gfem.Material(tan_delta=(1e-3, -1e-4, 0.0))
    assert "tan_delta" in str(err.value)
    with pytest.raises(ValueError, match="isotropic"):
        gfem.MaterialField.uniform("r", gfem.Material(tan_delta=(0.0, 0.0, 1e-3)))


def _rotation(axis, angle):
    """The rotation by ``angle`` about ``axis`` (Rodrigues)."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _tensor(values, axes):
    a = np.eye(3) if axes is None else np.asarray(axes)
    return sum(v * np.outer(a_k, a_k) for v, a_k in zip(values, a, strict=True))


def test_a_symmetric_matrix_becomes_principal_values_along_axes():
    r = _rotation((1, 2, 3), 0.4)
    matrix = r @ np.diag([2.0, 3.0, 5.0]) @ r.T
    m = gfem.Material(eps_r=matrix)
    # r is close to the identity: column k of r is the axis closest to e_k
    np.testing.assert_allclose(m.eps_r, (2.0, 3.0, 5.0), rtol=1e-12)
    np.testing.assert_allclose(m.axes, r.T, atol=1e-12)
    np.testing.assert_allclose(_tensor(m.eps_r, m.axes), matrix, atol=1e-12)
    # the same matrix from permuted, negated eigenvectors: the same normal form
    q = r[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])
    same = gfem.Material(eps_r=q @ np.diag([5.0, 2.0, 3.0]) @ q.T)
    np.testing.assert_allclose(same.eps_r, m.eps_r, rtol=1e-12)
    np.testing.assert_allclose(same.axes, m.axes, atol=1e-12)
    hash(m)
    assert repr(m).startswith("Material(eps_r=(")
    assert "axes=((" in repr(m)
    assert "array" not in repr(m)


@pytest.mark.parametrize(
    ("angle", "values", "axes"),
    [
        # Rz(2.5): both in-plane eigenvectors point away from +x / +y, so the
        # dominant component of each is made positive
        (
            2.5,
            (2.0, 3.0, 5.0),
            ((-np.cos(2.5), -np.sin(2.5), 0), (np.sin(2.5), -np.cos(2.5), 0)),
        ),
        # Rz(pi/2 + 0.2): the first eigenvector lies closest to y, so it is
        # axis 2 and its value moves with it
        (
            np.pi / 2 + 0.2,
            (3.0, 2.0, 5.0),
            (
                (np.sin(np.pi / 2 + 0.2), -np.cos(np.pi / 2 + 0.2), 0),
                (np.cos(np.pi / 2 + 0.2), np.sin(np.pi / 2 + 0.2), 0),
            ),
        ),
    ],
)
def test_derived_axes_are_ordered_by_the_closest_coordinate_axis_and_signed(
    angle, values, axes
):
    r = _rotation((0, 0, 1), angle)
    m = gfem.Material(eps_r=r @ np.diag([2.0, 3.0, 5.0]) @ r.T)
    np.testing.assert_allclose(m.eps_r, values, rtol=1e-12)
    np.testing.assert_allclose(m.axes, (*axes, (0, 0, 1)), atol=1e-12)


def test_a_diagonal_matrix_equals_its_principal_values_on_the_coordinate_axes():
    m = gfem.Material(eps_r=np.diag([2.0, 2.0, 3.0]))
    assert m == gfem.Material(eps_r=(2.0, 2.0, 3.0))
    assert hash(m) == hash(gfem.Material(eps_r=(2.0, 2.0, 3.0)))
    assert m.axes is None
    assert repr(m) == "Material(eps_r=(2.0, 2.0, 3.0))"
    off = np.diag([2.0, 3.0, 4.0]) + 1e-12 * (np.ones((3, 3)) - np.eye(3))
    assert gfem.Material(mu_r=off).axes is None


def test_a_degenerate_eigenspace_takes_the_projected_coordinate_axes():
    r = _rotation((1, 2, 3), 0.4)
    u = r[:, 2]  # the eigenvector of 5, closest to z
    p = np.eye(3) - np.outer(u, u)  # the eigenspace of 2
    a0 = p[:, 0] / np.linalg.norm(p[:, 0])
    a1 = p[:, 1] - (a0 @ p[:, 1]) * a0
    a1 /= np.linalg.norm(a1)
    expected = (a0, a1, u)
    for spin in (0.0, 1.0, 2.0):  # any basis of the eigenspace: one normal form
        q = r @ _rotation((0, 0, 1), spin)
        m = gfem.Material(eps_r=q @ np.diag([2.0, 2.0, 5.0]) @ q.T)
        np.testing.assert_allclose(m.eps_r, (2.0, 2.0, 5.0), rtol=1e-12)
        np.testing.assert_allclose(m.axes, expected, atol=1e-12)


def _message(**kwargs):
    with pytest.raises(ValueError) as err:
        gfem.Material(**kwargs)
    return str(err.value)


def test_matrix_input_rules():
    r = _rotation((1, 2, 3), 0.4)
    tilted = r @ np.diag([2.0, 3.0, 5.0]) @ r.T
    assert _message(eps_r=tilted, axes=np.eye(3)) == (
        "Material: eps_r is a 3x3 matrix, which carries its own principal axes; "
        "leave `axes` as None."
    )
    assert _message(eps_r=tilted, tan_delta=(1e-3, 1e-3, 2e-3)) == (
        "Material: eps_r is a 3x3 matrix but tan_delta gives three principal "
        "values along `axes`, which the matrix derives itself; give tan_delta "
        "as one value or a 3x3 matrix."
    )
    assert _message(eps_r=tilted * (1 - 1e-3j)) == (
        "Material: eps_r must be real; give losses with tan_delta and conductivity."
    )
    bad = tilted.copy()
    bad[0, 1] = np.nan
    assert "Material: eps_r must be finite" in _message(eps_r=bad)
    assert _message(eps_r=np.eye(2)) == (
        "Material: eps_r takes one value, three principal values (along `axes`) "
        "or a 3x3 matrix, got shape (2, 2)."
    )
    skew = tilted + 1e-6 * np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    assert _message(eps_r=skew, name="ferrite") == (
        "Material 'ferrite': eps_r is not symmetric; non-symmetric (gyrotropic) "
        "tensors are not supported."
    )
    # an asymmetry at round-off is accepted and symmetrized
    nearly = tilted + 1e-11 * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    m = gfem.Material(eps_r=nearly)
    np.testing.assert_allclose(_tensor(m.eps_r, m.axes), tilted, atol=1e-11)


def test_a_material_s_matrices_share_one_set_of_principal_axes():
    r = _rotation((1, 2, 3), 0.4)
    assert _message(
        eps_r=r @ np.diag([2.0, 3.0, 5.0]) @ r.T,
        conductivity=np.diag([1.0, 2.0, 3.0]),
        name="x",
    ) == (
        "eps_r and conductivity of Material 'x' have different principal axes; "
        "a material's tensors share one set of principal axes."
    )
    # eps_r is degenerate; conductivity, which commutes with it, splits the
    # eigenspace and so fixes the axes
    eps = r @ np.diag([2.0, 2.0, 5.0]) @ r.T
    sigma = r @ np.diag([1.0, 3.0, 0.5]) @ r.T
    m = gfem.Material(eps_r=eps, conductivity=sigma)
    np.testing.assert_allclose(m.axes, r.T, atol=1e-12)
    np.testing.assert_allclose(m.eps_r, (2.0, 2.0, 5.0), rtol=1e-12)
    np.testing.assert_allclose(m.conductivity, (1.0, 3.0, 0.5), rtol=1e-12)
    np.testing.assert_allclose(_tensor(m.eps_r, m.axes), eps, atol=1e-12)
    np.testing.assert_allclose(_tensor(m.conductivity, m.axes), sigma, atol=1e-12)


def test_passivity_is_checked_per_principal_value():
    assert _message(conductivity=(1.0, -1.0, 0.0), name="x") == (
        "Material 'x' is not passive: conductivity has principal value -1.0 "
        "(along axis 2) but every principal value of conductivity must be >= 0. "
        "A passive medium has eps_r > 0, mu_r > 0 and losses (tan_delta, "
        "conductivity) >= 0."
    )
    axes = np.array([[0.6, 0.8, 0.0], [-0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    # the eigenvector of -1 lies closest to x, so it is axis 1, signed +x
    assert _message(eps_r=axes.T @ np.diag([2.0, -1.0, 3.0]) @ axes) == (
        "Material is not passive: eps_r has principal value -1.0 (along axis "
        "(0.8, -0.6, 0.0)) but every principal value of eps_r must be > 0. A "
        "passive medium has eps_r > 0, mu_r > 0 and losses (tan_delta, "
        "conductivity) >= 0."
    )
    wire = gfem.Material(conductivity=(0.0, 0.0, 1.0))  # conducts along z only
    assert wire.conductivity == (0.0, 0.0, 1.0)
    assert not wire.isotropic
    assert gfem.Material(conductivity=(0.0, 0.0, 0.0)).conductivity == 0.0


def test_equal_principal_values_are_one_value():
    m = gfem.Material(eps_r=(2.0, 2.0, 2.0))
    assert m.eps_r == 2.0
    assert m.isotropic
    assert m == gfem.Material(eps_r=2)
    assert hash(m) == hash(gfem.Material(eps_r=2))
    assert repr(m) == "Material(eps_r=2.0)"
    r = _rotation((1, 2, 3), 0.4)
    near = r @ (2 * np.diag([1 + 1e-10, 1.0, 1 - 1e-10])) @ r.T
    m = gfem.Material(eps_r=near)
    assert m.isotropic
    assert m.axes is None
    assert m.eps_r == pytest.approx(2.0, rel=1e-12)
    assert not gfem.Material(conductivity=(1.0, 1.0, 2.0)).isotropic
    f = gfem.MaterialField.uniform("r", gfem.Material(eps_r=(2.0, 2.0, 2.0)))
    assert float(f.eps_r) == 2.0


def test_material_conductivity_is_documented_as_electrical():
    assert "electrical" in gfem.Material.__doc__
    assert "S/m" in gfem.Material.__doc__


def test_material_under_a_gradient_raises_with_the_material_field_remedy():
    def objective(eps):
        return gfem.Material(eps_r=eps).eps_r

    with pytest.raises(gfem.NonDifferentiableError, match=r"MaterialField\.uniform"):
        jax.grad(objective)(2.0)


# --- MaterialField -------------------------------------------------------------


def test_material_field_is_keyword_only_with_absent_properties_none():
    with pytest.raises(TypeError):
        gfem.MaterialField(jnp.ones(3))  # ty: ignore[too-many-positional-arguments]
    f = gfem.MaterialField(eps_r=jnp.ones(3))
    assert f.conductivity is None
    assert f.thermal_conductivity is None
    assert jax.tree.leaves(f)[0].shape == (3,)
    assert len(jax.tree.leaves(f)) == 1


def test_material_field_values_are_float64_one_per_element():
    f = gfem.MaterialField(eps_r=[1, 2, 3], conductivity=np.zeros(3))
    assert f.eps_r.dtype == jnp.float64
    assert f.n_elements == 3
    with pytest.raises(ValueError, match="one value per element"):
        gfem.MaterialField(eps_r=jnp.ones((3, 2)))
    with pytest.raises(ValueError, match="same number"):
        gfem.MaterialField(eps_r=jnp.ones(3), conductivity=jnp.ones(4))
    with pytest.raises(ValueError, match="at least one"):
        gfem.MaterialField()


def test_material_field_rejects_active_concrete_values():
    with pytest.raises(ValueError, match="passive"):
        gfem.MaterialField(eps_r=jnp.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="passive"):
        # e^{+iwt}: a lossy permittivity has a negative imaginary part
        gfem.MaterialField(eps_r=jnp.array([4.0 + 0.1j]))
    assert gfem.MaterialField(eps_r=jnp.array([4.0 - 0.1j])).eps_r.dtype == (
        jnp.complex128
    )


def test_interpolate_blends_every_property_both_materials_define():
    void = gfem.Material()
    solid = gfem.Material(eps_r=10.0, conductivity=2.0, thermal_conductivity=150.0)
    rho = jnp.array([0.0, 0.5, 1.0])
    f = gfem.MaterialField.interpolate(rho, void, solid)
    np.testing.assert_allclose(f.eps_r, [1.0, 5.5, 10.0])
    np.testing.assert_allclose(f.conductivity, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(f.mu_r, [1.0, 1.0, 1.0])
    assert f.thermal_conductivity is None  # the void defines no thermal value


@pytest.mark.parametrize(
    ("scheme", "expected"),
    # rho = 0.5, penalty 3: simp 0.5**3 = 0.125; ramp 0.5 / (1 + 3 * 0.5) = 0.2
    [("linear", 0.5), ("simp", 0.125), ("ramp", 0.2)],
)
def test_interpolation_schemes(scheme, expected):
    f = gfem.MaterialField.interpolate(
        jnp.array([0.5]), gfem.Material(), gfem.Material(eps_r=2.0), scheme=scheme
    )
    np.testing.assert_allclose(f.eps_r, [1.0 + expected])


def test_interpolate_is_differentiable_in_the_density():
    void, solid = gfem.Material(), gfem.Material(eps_r=10.0)

    def total(rho):
        return jnp.sum(gfem.MaterialField.interpolate(rho, void, solid).eps_r)

    np.testing.assert_allclose(jax.grad(total)(jnp.full(4, 0.3)), np.full(4, 9.0))


def test_interpolate_rejects_unknown_scheme_and_anisotropic_materials():
    with pytest.raises(ValueError, match="scheme"):
        gfem.MaterialField.interpolate(
            jnp.ones(2), gfem.Material(), gfem.Material(), scheme="cubic"
        )
    with pytest.raises(ValueError, match="isotropic"):
        gfem.MaterialField.interpolate(
            jnp.ones(2), gfem.Material(), gfem.Material(eps_r=(1.0, 2.0, 3.0))
        )


def test_uniform_field_holds_one_differentiable_value_per_property():
    sub = gfem.Material(eps_r=11.7, tan_delta=1e-4)
    f = gfem.MaterialField.uniform("sub", sub)
    assert f.region == "sub"
    assert f.eps_r.shape == ()
    assert float(f.eps_r) == 11.7
    assert float(f.tan_delta) == 1e-4
    assert f.thermal_conductivity is None

    def eps(field):
        return field.eps_r * 2.0

    assert float(jax.grad(eps)(f).eps_r) == 2.0


def test_material_field_is_a_pytree_with_named_leaf_paths():
    f = gfem.MaterialField(eps_r=jnp.ones(2), conductivity=jnp.zeros(2))
    paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(f)]
    assert paths == [".eps_r", ".conductivity"]
    g = jax.tree.map(lambda x: x * 2, f)
    assert isinstance(g, gfem.MaterialField)
    np.testing.assert_allclose(g.eps_r, [2.0, 2.0])


def test_material_repr_shows_only_what_differs_from_vacuum():
    assert repr(gfem.Material()) == "Material()"
    m = gfem.Material(eps_r=11.7, tan_delta=1e-4, name="si")
    assert repr(m) == "Material(eps_r=11.7, tan_delta=0.0001, name='si')"
