r"""The Kerr nonlinear coefficient of a port mode: closed forms and discrete identities.

On a homogeneously filled PEC guide the discrete modes obey
:math:`\beta_h^2 = \varepsilon k_0^2 - k_{c,h}^2` with a cutoff fixed by the
mesh, and the mode's shape does not change with frequency. The coefficient is
then a closed form in the computed index, which removes the method error in
:math:`\beta`; what is left is the field's own mesh error (measured per gate).
"""

import dataclasses
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases
import pytest
import scipy.constants as sc
from slab2d import write_slab

import gfem
from gfem import kerr
from gfem.errors import NonDifferentiableError

DATA = Path(__file__).parent / "data"
#: WR-90 (m): the guide's section, not the extent of its points
A, B = 22.86e-3, 10.16e-3
N, N2, TPA = 1.5, 2.6e-20, 5e-12
F = 8e9
ORDER = gfem.SolverOptions(order=2)
LORENTZ = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=40e9, damping=0.0)],
)


def k0_of(f):
    return 2 * np.pi * np.asarray(f) / sc.c


def guide_model(mesh, material):
    return gfem.Model(
        mesh,
        materials={"air": material},
        boundaries=[gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )


def mode_of(model, f=F, order=ORDER, **kwargs):
    return gfem.solve(model, gfem.PortMode("p1", f, **kwargs), options=order)


@pytest.fixture(scope="module")
def guide_mesh():
    return gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")


@pytest.fixture(scope="module")
def filled(guide_mesh):
    """The filled guide at 8 GHz with n2 and TPA, its mode and coefficient."""
    material = gfem.Material(eps_r=N**2, nonlinear_index=N2, two_photon_absorption=TPA)
    model = guide_model(guide_mesh, material)
    mode = mode_of(model, [F])
    return model, mode, gfem.kerr_nonlinearity(model, mode)


# --- closed forms --------------------------------------------------------------


def test_the_filled_te10_follows_its_closed_form(filled):
    _, mode, kn = filled
    k0 = k0_of(F)
    n_eff = float(mode.gamma.imag[0] / k0)
    reference = 1.5 * k0 * N2 / (A * B) * (N / n_eff) ** 2
    # the field's own mesh error: 7.0e-6 measured at order 2 (3.4e-8 at order 3)
    assert kn.gamma_nl[0].real == pytest.approx(reference, rel=1e-5, abs=0)


def test_the_result_names_its_port_mode_and_frequencies(filled):
    _, _, kn = filled
    assert (kn.port, kn.mode) == ("p1", 1)
    np.testing.assert_array_equal(kn.frequency, [F])
    assert kn.gamma_nl.shape == (1,)
    assert kn.gamma_nl.dtype == jnp.complex128


@pytest.mark.parametrize("fill", ["constant", "lorentz"])
def test_every_row_of_a_grid_follows_the_row_scaling(guide_mesh, fill):
    # 41 rows, two exact: the others are reduced rows. The homogeneous fill's
    # discrete shape does not depend on frequency and a reduced beta is exact
    # to 2.5e-14, so the ratio of rows is a closed form of the computed indices
    # (and, for the law, of Re eps_r); round-off over a few operations is 1e-15,
    # the gate 1e-10 leaves the margin for solver noise in beta.
    f = np.linspace(7e9, 8.5e9, 41)
    eps = LORENTZ if fill == "lorentz" else N**2
    model = guide_model(guide_mesh, gfem.Material(eps_r=eps, nonlinear_index=N2))
    mode = mode_of(model, f)
    assert (~np.asarray(mode.solved)).any()
    gamma = np.asarray(gfem.kerr_nonlinearity(model, mode).gamma_nl)
    n_eff = np.asarray(mode.gamma.imag) / k0_of(f)
    re_eps = np.real(np.asarray(LORENTZ(f))) if fill == "lorentz" else np.full(41, N**2)
    expected = gamma[0] * (f / f[0]) * (n_eff[0] / n_eff) ** 2 * re_eps / re_eps[0]
    np.testing.assert_allclose(gamma, expected, rtol=1e-10)


def test_two_photon_absorption_is_the_imaginary_part(filled):
    _, _, kn = filled
    g = complex(kn.gamma_nl[0])
    # the integrand is shared: the ratio is exact up to two roundings
    assert g.imag >= 0
    assert g.imag / g.real == pytest.approx(TPA / (2 * k0_of(F) * N2), rel=1e-12)


def test_a_material_with_two_photon_absorption_only_has_no_kerr_part(guide_mesh):
    model = guide_model(
        guide_mesh, gfem.Material(eps_r=N**2, two_photon_absorption=TPA)
    )
    g = complex(gfem.kerr_nonlinearity(model, mode_of(model)).gamma_nl[0])
    assert g.real == 0.0
    assert g.imag > 0


def test_a_lossy_fill_changes_gamma_through_the_mode_normalization_only(
    guide_mesh, filled
):
    # tan_delta adds an imaginary part to eps_r only, and the integrand stays
    # real for a real chi. The 1 W normalization of a lossy mode is by the
    # modulus of its complex power, so the shape is the lossless one and
    # gamma scales as (n_eff / |n_eff,lossy|)^2; both identities hold to
    # round-off over a few operations, the gate 1e-10 leaves margin.
    _, mode, _ = filled
    lossy = guide_model(
        guide_mesh,
        gfem.Material(eps_r=N**2, tan_delta=0.05, nonlinear_index=N2),
    )
    lossy_mode = mode_of(lossy, [F])
    g = complex(gfem.kerr_nonlinearity(lossy, lossy_mode).gamma_nl[0])
    assert abs(g.imag) <= 1e-14 * abs(g.real)
    lossless = guide_model(guide_mesh, gfem.Material(eps_r=N**2, nonlinear_index=N2))
    g0 = complex(gfem.kerr_nonlinearity(lossless, mode_of(lossless, [F])).gamma_nl[0])
    n0 = complex(mode.n_eff[0]).real
    n_lossy = complex(lossy_mode.n_eff[0])
    assert abs(n_lossy.imag) > 1e-2  # the case is lossy
    assert g.real / g0.real == pytest.approx((n0 / abs(n_lossy)) ** 2, rel=1e-10)


# --- gradients -----------------------------------------------------------------


def _field_model(mesh, eps, n2, beta):
    material = gfem.MaterialField(
        eps_r=eps,
        nonlinear_index=n2,
        two_photon_absorption=beta,
        region="air",
    )
    return guide_model(mesh, material)


def test_gamma_differentiates_in_the_permittivity_and_the_two_coefficients(
    guide_mesh, filled
):
    _, mode, kn = filled
    g = complex(kn.gamma_nl[0])
    n_eff = float(np.real(mode.n_eff[0]))

    def gamma(eps, n2, beta):
        model = _field_model(guide_mesh, eps, n2, beta)
        return gfem.kerr_nonlinearity(model, mode_of(model, [F])).gamma_nl[0]

    args = (jnp.asarray(N**2), jnp.asarray(N2), jnp.asarray(TPA))

    def real(*a):
        return jnp.real(gamma(*a))

    def imag(*a):
        return jnp.imag(gamma(*a))

    value, d_real = jax.value_and_grad(real, argnums=(0, 1, 2))(*args)
    d_imag = jax.grad(imag, argnums=(0, 1, 2))(*args)
    # the traced vector gives the eager value: the same operations, up to the
    # solve's own round-off (1e-14), the gate 1e-12
    assert float(value) == pytest.approx(g.real, rel=1e-12, abs=0)
    # on a homogeneous fill the discrete cutoff is fixed, so
    # d ln(gamma) / d eps = 1 / eps - 1 / n_eff^2; the derivative of the
    # discrete problem is exact up to the adjoint solve's round-off (1e-10)
    slope = 1 / N**2 - 1 / n_eff**2
    assert float(d_real[0]) == pytest.approx(g.real * slope, rel=1e-8, abs=0)
    assert float(d_imag[0]) == pytest.approx(g.imag * slope, rel=1e-8, abs=0)
    # exact: gamma is linear in each coefficient
    assert float(d_real[1]) == pytest.approx(g.real / N2, rel=1e-12)
    assert float(d_imag[2]) == pytest.approx(g.imag / TPA, rel=1e-12)
    assert float(d_real[2]) == pytest.approx(0.0, abs=1e-12 * g.real / TPA)
    assert float(d_imag[1]) == pytest.approx(0.0, abs=1e-12 * g.imag / N2)


# --- a perfectly matched layer -------------------------------------------------


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    # a coarse strip with a wide cladding: the layer holds under 0.1 of the mode's
    # energy (no LayerClearanceWarning) and the solve takes seconds
    path = pml_cases.strip_guide(
        tmp_path_factory.mktemp("strip") / "strip.msh",
        0.8,
        quarter=False,
        length=0.15,
        core_h=0.2,
        clad_h=0.3,
        thickness=0.5,
    )
    model = pml_cases.strip_model(path)
    materials = dict(model.materials)
    materials["core"] = dataclasses.replace(materials["core"], nonlinear_index=6e-18)
    model = model.replace(materials=materials)
    mode = gfem.solve(model, gfem.PortMode("p1", pml_cases.STRIP_F), options=ORDER)
    return model, mode


def test_a_property_on_the_layer_alone_changes_nothing(strip):
    model, mode = strip
    base = complex(gfem.kerr_nonlinearity(model, mode).gamma_nl[0])
    materials = dict(model.materials)
    materials["pml"] = dataclasses.replace(materials["pml"], nonlinear_index=1e-17)
    on_layer = model.replace(materials=materials)
    # the layer's triangles carry weight 0: every term of the sum is the same
    got = complex(gfem.kerr_nonlinearity(on_layer, mode).gamma_nl[0])
    assert got == pytest.approx(base, rel=1e-14)


def test_a_property_only_on_the_layer_is_refused(strip):
    model, mode = strip
    materials = {
        name: dataclasses.replace(m, nonlinear_index=None)
        for name, m in model.materials.items()
    }
    materials["pml"] = dataclasses.replace(materials["pml"], nonlinear_index=1e-17)
    with pytest.raises(ValueError, match=r"wave port 'p1'.*layer"):
        gfem.kerr_nonlinearity(model.replace(materials=materials), mode)


def _read_from_the_mesh(model, mode):
    """``mode`` as a solve traced by JAX returns it: the DOF vector alone.

    The space is then located from the mesh (`port_geometry`), whose section
    has no layer; the values are the same concrete ones.
    """
    from gfem._engine.port_mode import PortModeSpace
    from gfem._port_section import port_geometry
    from gfem.results import PortModeField

    space = mode.e_field._space
    section = port_geometry(model, space.section.port)
    field = PortModeField._on_vector(
        mode.port,
        lambda: PortModeSpace(section, space.order, geometry=True),
        kerr._vector(mode.e_field),
    )
    return dataclasses.replace(mode, e_field=field)


def test_a_traced_field_skips_the_layer_as_a_concrete_one_does(strip):
    model, mode = strip
    materials = dict(model.materials)
    materials["pml"] = dataclasses.replace(materials["pml"], nonlinear_index=1e-16)
    on_layer = model.replace(materials=materials)
    eager = complex(gfem.kerr_nonlinearity(on_layer, mode).gamma_nl[0])
    traced = _read_from_the_mesh(on_layer, mode)
    assert traced.e_field._space.section.layer is None
    got = complex(gfem.kerr_nonlinearity(on_layer, traced).gamma_nl[0])
    # the same vector, columns and sum of the same terms: round-off of the
    # reordered sum (1e-15), the gate 1e-12; counting the layer's triangles
    # changed the value by 9e-3
    assert got == pytest.approx(eager, rel=1e-12)


def test_a_property_only_on_the_layer_is_refused_under_a_trace_too(strip):
    model, mode = strip
    materials = {
        name: dataclasses.replace(m, nonlinear_index=None)
        for name, m in model.materials.items()
    }
    materials["pml"] = dataclasses.replace(materials["pml"], nonlinear_index=1e-17)
    only = model.replace(materials=materials)
    with pytest.raises(ValueError, match=r"wave port 'p1'.*layer"):
        gfem.kerr_nonlinearity(only, _read_from_the_mesh(only, mode))


# --- the columns -----------------------------------------------------------------


def _rule_points(space, raised=0):
    """Global points ``(n_el, n_q, 2 or 3)`` of the rule's points, in section order.

    NGSolve itself maps the points: a coordinate function set into an
    integration rule space holds the position of each of its points.
    """
    import ngsolve as ngs

    section = space.section
    planar = len(section.normal) == 3
    irs = ngs.comp.IntegrationRuleSpace(space.mesh, order=2 * space.order + raised)
    local = []
    for coordinate in (ngs.x, ngs.y) if planar else (ngs.x,):
        gf = ngs.GridFunction(irs)
        gf.Set(coordinate, ngs.VOL)
        local.append(gf.vec.FV().NumPy().copy())
    owner = np.array(
        [list(irs.GetDofNrs(ngs.ElementId(ngs.VOL, e))) for e in range(space.mesh.ne)]
    )
    local = np.stack(local, axis=-1)[owner]  # (n_el, n_q, 2 or 1)
    if not planar:
        order = np.empty(len(owner), int)
        order[np.asarray(space.element_segment)] = np.arange(len(owner))
        local = local[order]
    shape = local.shape[:2]
    if planar:
        return section.to_global(local.reshape(-1, 2)).reshape(*shape, 3)
    return section.to_global(local.reshape(-1)).reshape(*shape, 2)


@pytest.mark.parametrize("raised", [0, 2])
def test_the_planar_columns_are_the_field_at_the_rule_points(filled, raised):
    _, mode, _ = filled
    field = mode.e_field
    space = field._space
    cols = kerr._columns(space, raised)
    points = _rule_points(space, raised)
    x = np.asarray(kerr._vector(field))
    waves = kerr._Waves(
        "p1", (1,), np.array([F]), jnp.asarray(x), space.section, cols, None
    )
    got = np.asarray(waves.field())[0]
    expected = np.asarray(field.at(points.reshape(-1, 3)))[0].reshape(got.shape)
    # point values of a polynomial field through two paths: each is a sum of at
    # most 45 products of O(1) weights and DOFs, so round-off is about 45 eps
    # = 1e-14 of the largest value, measured 1.3e-15; the gate 1e-12 keeps the
    # margin of a platform's BLAS and FMA ordering
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )
    area = cols.measure.sum()
    assert area == pytest.approx(A * B, rel=1e-12, abs=0)


def test_the_integral_is_exact_at_the_rule_degree(filled, monkeypatch):
    model, mode, kn = filled
    original = kerr._columns
    # the columns of the default rule are kept per mesh: start from none
    monkeypatch.setattr(kerr, "_COLUMNS", type(kerr._COLUMNS)())
    monkeypatch.setattr(kerr, "_columns", lambda space: original(space, 2))
    higher = gfem.kerr_nonlinearity(model, mode)
    # degree 4p is exact for a degree-p field and a material constant per
    # element: the two rules differ by summation order only (1e-16 relative)
    np.testing.assert_allclose(higher.gamma_nl, kn.gamma_nl, rtol=1e-13)


def test_a_traced_result_reads_the_columns_of_its_section_once(guide_mesh, monkeypatch):
    # every traced result rebuilds its space; the columns are kept per mesh
    monkeypatch.setattr(kerr, "_COLUMNS", type(kerr._COLUMNS)())
    built = []
    original = kerr._columns
    monkeypatch.setattr(
        kerr, "_columns", lambda space: built.append(space) or original(space)
    )

    def objective(n2):
        model = _field_model(guide_mesh, jnp.asarray(N**2), n2, jnp.asarray(0.0))
        return jnp.real(gfem.kerr_nonlinearity(model, mode_of(model, [F])).gamma_nl[0])

    first = jax.grad(objective)(jnp.asarray(N2))
    second = jax.grad(objective)(jnp.asarray(N2 * 2))
    assert len(built) == 1
    # linear in n2: the gradient is the coefficient per unit n2, whatever n2
    assert float(second) == pytest.approx(float(first), rel=1e-12)


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    path = write_slab(
        tmp_path_factory.mktemp("slab") / "slab.msh",
        thick=0.22,
        clear=1.0,
        h_core=0.055,
        h_clad=0.2,
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="um"),
        materials={
            "core": gfem.Material(eps_r=3.476**2, nonlinear_index=6e-18),
            "clad": gfem.Material(eps_r=1.444**2, nonlinear_index=2.6e-20),
        },
        ports=[gfem.WavePort("p1", "port1", pec_edges=("outer",))],
    )
    mode = mode_of(
        model, sc.c / 1.55e-6, polarization="Ez", order=gfem.SolverOptions(order=3)
    )
    return model, mode


@pytest.mark.parametrize("raised", [0, 2])
def test_the_line_columns_are_the_field_at_the_rule_points(slab, raised):
    _, mode = slab
    field = mode.e_field
    space = field._space
    cols = kerr._columns(space, raised)
    points = _rule_points(space, raised)
    section = space.section
    waves = kerr._Waves(
        "p1", (1,), np.array([1.0]), kerr._vector(field), section, cols, None
    )
    got = np.asarray(waves.field())[0, ..., 0]
    expected = np.asarray(field.at(points.reshape(-1, 2)))[0].reshape(got.shape)
    # as the planar columns: a sum of at most 4 (order 3) products per point,
    # round-off about 1e-15 of the largest value (measured 6e-16), gate 1e-12
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )
    length = np.ptp(section.points) * section.scale
    assert cols.measure.sum() == pytest.approx(length, rel=1e-12, abs=0)


def test_an_ez_line_port_gives_its_slab_value_per_watt(slab):
    model, mode = slab
    kn = gfem.kerr_nonlinearity(model, mode)
    g = complex(kn.gamma_nl[0])
    assert g.imag == 0.0
    # the order of magnitude of a 220 nm silicon slab TE0: 1e-4 1/W, and the
    # exact closed form is the verification tier's
    assert 5e-5 < g.real < 2e-4


# --- refusals ----------------------------------------------------------------------


def test_a_model_without_a_property_names_the_port(guide_mesh):
    model = guide_model(guide_mesh, gfem.Material(eps_r=N**2))
    with pytest.raises(ValueError, match=r"wave port 'p1'.*layer"):
        gfem.kerr_nonlinearity(model, mode_of(model))


def test_an_anisotropic_field_with_a_property_is_refused(tmp_path):
    path = pml_cases.strip_guide(
        tmp_path / "strip.msh", 0.4, thickness=0, quarter=False, length=0.15
    )
    base = pml_cases.strip_model(path)
    aniso = gfem.Material(eps_r=(2.1, 2.1, 2.3))
    core = gfem.Material(eps_r=pml_cases.STRIP_CORE**2, nonlinear_index=6e-18)
    f = pml_cases.STRIP_F

    def materials(clad):
        return base.replace(materials={"core": core, "clad": clad})

    works = materials(aniso)  # without the property it contributes nothing
    mode = gfem.solve(works, gfem.PortMode("p1", f), options=ORDER)
    assert gfem.kerr_nonlinearity(works, mode).gamma_nl[0].real > 0
    field = gfem.MaterialField.uniform(
        "clad", dataclasses.replace(aniso, nonlinear_index=2.6e-20)
    )
    with pytest.raises(NotImplementedError, match=r"region 'clad'.*MaterialField"):
        gfem.kerr_nonlinearity(materials(field), mode)


def test_a_layer_grid_design_blending_a_kerr_property_is_refused(slab):
    # the solve refuses a layer grid in a port mode problem, so the mode is the
    # slab's and the model is the same mesh with the core as a pixel design
    model, mode = slab
    mesh = model.mesh
    grid = gfem.design.LayerGrid(mesh, "core", pitch=0.05e-6)
    rho = grid.to_elements(np.ones(grid.shape))
    clad = model.materials["clad"]
    core = gfem.MaterialField.interpolate(rho, clad, model.materials["core"])
    assert core.blend_ends.pixels is not None
    assert "nonlinear_index" in [n for n, _, _ in core.blend_ends.others]
    pixels = model.replace(materials={"core": core, "clad": clad})
    with pytest.raises(
        NotImplementedError, match=r"region 'core'.*layer-grid.*'nonlinear_index'"
    ):
        gfem.kerr_nonlinearity(pixels, mode)


def test_a_model_of_another_mesh_or_dimension_is_refused(guide_mesh, filled, slab):
    model, mode, _ = filled
    nodes = jnp.asarray(np.asarray(guide_mesh.imported_nodes) * 1.01)
    mesh = jax.tree.unflatten(jax.tree.structure(guide_mesh), [nodes])
    with pytest.raises(ValueError, match=r"not the one wave port 'p1'"):
        gfem.kerr_nonlinearity(model.replace(mesh=mesh), mode)
    with pytest.raises(ValueError, match=r"not the one wave port 'p1'"):
        gfem.kerr_nonlinearity(slab[0], mode)
    with pytest.raises(ValueError, match=r"not the one wave port 'p1'"):
        gfem.kerr_nonlinearity(model, slab[1])


def test_an_hz_line_port_is_refused(slab):
    model, _ = slab
    hz = mode_of(model, sc.c / 1.55e-6, polarization="Hz")
    with pytest.raises(NotImplementedError, match="Hz line port"):
        gfem.kerr_nonlinearity(model, hz)


def test_a_result_without_a_field_or_a_non_result_is_a_type_error(filled):
    model, mode, _ = filled
    with pytest.raises(TypeError, match=r"gfem\.PortMode"):
        gfem.kerr_nonlinearity(model, dataclasses.replace(mode, e_field=None))
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.kerr_nonlinearity(model, "mode")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"gfem\.Model"):
        gfem.kerr_nonlinearity(mode, mode)  # ty: ignore[invalid-argument-type]


def test_traced_node_coordinates_differentiate_the_filled_guide(guide_mesh, filled):
    model, _, _ = filled
    nodes = np.array(guide_mesh.imported_nodes)

    def objective(scale):
        mesh = jax.tree.unflatten(
            jax.tree.structure(guide_mesh), [jnp.asarray(nodes) * scale]
        )
        moved = model.replace(mesh=mesh)
        mode = mode_of(moved, [F])
        return jnp.real(gfem.kerr_nonlinearity(moved, mode).gamma_nl[0]), mode.n_eff[0]

    # the whole mesh grows as s, the section as s: d ln(gamma)/ds = -2 - 2 (N^2 -
    # n_eff^2) / n_eff^2 of the discrete modes. The gradient is the explicit and
    # the field part, which cancel to O(1) of their size: round-off a few 1e-16
    # of each, gate 1e-12 (6.6e-16 measured on the same closed form)
    grad = float(jax.grad(lambda s: objective(s)[0])(1.0))
    value, n_eff = (float(np.real(x)) for x in objective(1.0))
    closed = -2 - 2 * (N**2 - n_eff**2) / n_eff**2
    assert grad / value == pytest.approx(closed, rel=1e-12)


def test_a_line_port_has_no_node_derivative(slab):
    model, mode = slab
    nodes = jnp.asarray(model.mesh.nodes)

    def objective(n):
        moved = model.replace(mesh=model.mesh.with_nodes(n))
        return jnp.real(gfem.kerr_nonlinearity(moved, mode).gamma_nl[0])

    with pytest.raises(NonDifferentiableError, match="kerr_nonlinearity"):
        jax.grad(objective)(nodes)


# --- repr and pytree ---------------------------------------------------------------


def test_the_repr_gives_the_ranges_and_a_short_form_under_a_trace(guide_mesh, filled):
    _, _, kn = filled
    text = repr(kn)
    assert text.startswith("KerrNonlinearity(port 'p1', mode 1, 1 frequencies")
    assert "Re " in text
    assert "Im " in text
    assert "1/(W m)" in text
    seen = []

    def objective(n2):
        model = _field_model(guide_mesh, jnp.asarray(N**2), n2, jnp.asarray(0.0))
        result = gfem.kerr_nonlinearity(model, mode_of(model, [F]))
        seen.append(repr(result))
        return jnp.real(result.gamma_nl[0])

    jax.grad(objective)(jnp.asarray(N2))
    assert seen == ["KerrNonlinearity(port 'p1', mode 1, 1 frequencies)"]


def test_the_result_is_a_pytree_with_gamma_as_its_only_leaf(filled):
    _, _, kn = filled
    leaves, tree = jax.tree.flatten(kn)
    assert len(leaves) == 1
    again = jax.tree.unflatten(tree, leaves)
    assert (again.port, again.mode) == ("p1", 1)
    np.testing.assert_array_equal(again.frequency, kn.frequency)
    np.testing.assert_array_equal(again.gamma_nl, kn.gamma_nl)
    doubled = jax.tree.map(lambda x: 2 * x, kn)
    np.testing.assert_allclose(doubled.gamma_nl, 2 * kn.gamma_nl, rtol=1e-15)
    assert math.isfinite(float(jnp.abs(doubled.gamma_nl[0])))


# --- the coupling tensor ---------------------------------------------------------


@pytest.fixture(scope="module")
def waves(guide_mesh):
    """Signal, pump and idler of one mode on the filled guide, TPA 0."""
    model = guide_model(guide_mesh, gfem.Material(eps_r=N**2, nonlinear_index=N2))
    f = np.array([7.5e9, 8e9, 8.5e9])
    mode = mode_of(model, f)
    return model, mode, gfem.kerr_coupling(model, mode)


def test_the_tensor_names_its_waves_and_has_one_entry_per_tuple(waves):
    _, _, tensor = waves
    assert tensor.port == "p1"
    assert tensor.waves == ((1, 7.5e9), (1, 8e9), (1, 8.5e9))
    assert tensor.gamma_nl.shape == (3, 3, 3, 3)
    assert tensor.gamma_nl.dtype == jnp.complex128


def test_four_wave_mixing_follows_the_indices_of_its_waves(waves):
    # the homogeneous fill's discrete shape is the same at every frequency and
    # E_j scales as n_j^(-1/2) at 1 W: gamma_sppi / gamma_ssss =
    # n_s^2 / (n_p sqrt(n_s n_i)); round-off over a few operations is 1e-15,
    # the gate 1e-10 leaves margin for solver noise in the indices
    _, mode, tensor = waves
    n = np.asarray(mode.n_eff).real
    g = np.asarray(tensor.gamma_nl)
    s, p, i = 0, 1, 2
    expected = n[s] ** 2 / (n[p] * np.sqrt(n[s] * n[i]))
    assert g[s, p, p, i] / g[s, s, s, s] == pytest.approx(expected, rel=1e-10)


def test_the_diagonal_is_the_nonlinear_coefficient_of_each_row(waves):
    model, mode, tensor = waves
    kn = gfem.kerr_nonlinearity(model, mode)
    diagonal = np.asarray(tensor.gamma_nl)[range(3), range(3), range(3), range(3)]
    # two sums of the same terms in a different order: round-off, 1e-16
    np.testing.assert_allclose(diagonal, kn.gamma_nl, rtol=1e-13)


def test_the_tensor_is_symmetric_in_its_two_amplitudes_and_conserves_energy(waves):
    _, mode, tensor = waves
    g = np.asarray(tensor.gamma_nl)
    omega = 2 * np.pi * np.asarray(mode.frequency)
    scale = np.abs(g).max()
    # gamma_abcd = gamma_acbd: the product of A_b and A_c is symmetric
    np.testing.assert_allclose(g, g.transpose(0, 2, 1, 3), rtol=0, atol=1e-13 * scale)
    # the energy identity gamma_abcd / omega_a = conj(gamma_badc) / omega_b holds
    # for the real chi of a lossless fill; round-off of a few operations, 1e-16
    left = g / omega[:, None, None, None]
    right = np.conj(g.transpose(1, 0, 3, 2)) / omega[None, :, None, None]
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-13 * scale / omega.min())


def test_a_mode_twice_gives_every_entry_the_nonlinear_coefficient(filled):
    model, mode, kn = filled
    tensor = gfem.kerr_coupling(model, [mode, mode])
    g = np.asarray(tensor.gamma_nl)
    assert g.shape == (2, 2, 2, 2)
    # every entry sums the same terms as the diagonal in another order: a few
    # eps per partial sum over the 1e3 points, 2 log2(n) eps = 2e-15 at worst,
    # measured 6e-16; the gate 1e-13 leaves 50 times that
    np.testing.assert_allclose(g, complex(kn.gamma_nl[0]), rtol=1e-13)
    # the cross-phase sum of the amplitude equation counts the other wave twice
    assert g[0, 0, 1, 1] + g[0, 1, 0, 1] == pytest.approx(
        2 * complex(kn.gamma_nl[0]), rel=1e-13, abs=0
    )


def test_a_single_row_gives_one_wave(filled):
    model, mode, kn = filled
    tensor = gfem.kerr_coupling(model, mode)
    assert tensor.gamma_nl.shape == (1, 1, 1, 1)
    # the diagonal entry sums the same terms in another order: measured 3e-16,
    # the gate 1e-13 as for the doubled mode
    assert complex(tensor.gamma_nl[0, 0, 0, 0]) == pytest.approx(
        complex(kn.gamma_nl[0]), rel=1e-13, abs=0
    )


def test_two_photon_absorption_enters_with_the_frequency_of_the_equation(guide_mesh):
    # chi_a = 4/3 n^2 c eps0 (n2 - i beta c / (2 omega_a)): the ratio of the
    # imaginary to the real part of the entries of wave a is the diagonal's
    model = guide_model(
        guide_mesh,
        gfem.Material(eps_r=N**2, nonlinear_index=N2, two_photon_absorption=TPA),
    )
    f = np.array([7.5e9, 8.5e9])
    tensor = np.asarray(gfem.kerr_coupling(model, mode_of(model, f)).gamma_nl)
    ratio = tensor.imag / tensor.real
    for a in range(2):
        assert ratio[a] == pytest.approx(TPA / (2 * k0_of(f[a]) * N2), rel=1e-12)


@pytest.fixture(scope="module")
def cross_phase(guide_mesh):
    """TE10 and TE01 of the filled guide at 12 GHz, order 3."""
    model = guide_model(guide_mesh, gfem.Material(eps_r=N**2, nonlinear_index=N2))
    order = gfem.SolverOptions(order=3)
    modes = [mode_of(model, [12e9], order=order, mode=k) for k in (1, 3)]
    return model, modes, gfem.kerr_coupling(model, modes)


def test_cross_phase_of_te10_by_te01_is_four_ninths_of_its_self_phase(cross_phase):
    _, modes, tensor = cross_phase
    n_m, n_n = (float(np.real(m.n_eff[0])) for m in modes)
    g = np.asarray(tensor.gamma_nl)
    m, n = 0, 1
    ratio = (g[m, m, n, n] + g[m, n, m, n]) / g[m, m, m, m]
    # the modes' fields are the guide's TE10 and TE01 up to the mesh error of
    # their shapes: 5.8e-4, 1.4e-5 and 4.5e-8 measured at orders 2, 3 and 4;
    # the gate is 3.5 times the order-3 error
    assert ratio.real == pytest.approx(4 / 9 * n_m / n_n, rel=5e-5)
    # real fields in the gauge: the two orderings of the conjugated pair agree
    assert g[m, n, n, m] == pytest.approx(g[m, m, n, n], rel=1e-14, abs=0)


def test_waves_of_different_ports_or_orders_are_refused(guide_mesh, filled):
    model, mode, _ = filled
    ports = gfem.Model(
        guide_mesh,
        materials=model.materials,
        boundaries=model.boundaries,
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
    other = gfem.solve(ports, gfem.PortMode("p2", [F]), options=ORDER)
    with pytest.raises(ValueError, match=r"share one port.*'p1'.*'p2'"):
        gfem.kerr_coupling(ports, [mode, other])
    third = mode_of(model, [F], order=gfem.SolverOptions(order=3))
    with pytest.raises(ValueError, match=r"share one port, element order"):
        gfem.kerr_coupling(model, [mode, third])


def test_the_coupling_refuses_what_the_coefficient_refuses(filled, slab):
    model, mode, _ = filled
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.kerr_coupling(model, "mode")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="PortModeResult"):
        gfem.kerr_coupling(model, [])
    with pytest.raises(TypeError, match=r"gfem\.PortMode"):
        gfem.kerr_coupling(model, [dataclasses.replace(mode, e_field=None)])
    bare = guide_model(model.mesh, gfem.Material(eps_r=N**2))
    with pytest.raises(ValueError, match=r"wave port 'p1'"):
        gfem.kerr_coupling(bare, mode)
    slab_model, _ = slab
    hz = mode_of(slab_model, sc.c / 1.55e-6, polarization="Hz")
    with pytest.raises(NotImplementedError, match="Hz line port"):
        gfem.kerr_coupling(slab_model, hz)


def test_the_coupling_is_linear_in_the_nonlinear_index(guide_mesh):
    f = np.array([7.5e9, 8.5e9])

    def entry(n2):
        model = _field_model(guide_mesh, jnp.asarray(N**2), n2, jnp.asarray(0.0))
        tensor = gfem.kerr_coupling(model, mode_of(model, f))
        return jnp.real(tensor.gamma_nl[0, 1, 1, 0])

    value, slope = jax.value_and_grad(entry)(jnp.asarray(N2))
    # exact: every term of the integral carries chi once
    assert float(slope) == pytest.approx(float(value) / N2, rel=1e-12)


def test_the_repr_gives_the_waves_and_a_short_form_under_a_trace(guide_mesh, waves):
    _, _, tensor = waves
    assert repr(tensor).startswith("KerrCoupling(port 'p1', 3 waves, |gamma_nl| up to ")
    seen = []

    def objective(n2):
        model = _field_model(guide_mesh, jnp.asarray(N**2), n2, jnp.asarray(0.0))
        result = gfem.kerr_coupling(model, mode_of(model, [F]))
        seen.append(repr(result))
        return jnp.real(result.gamma_nl[0, 0, 0, 0])

    jax.grad(objective)(jnp.asarray(N2))
    assert seen == ["KerrCoupling(port 'p1', 1 waves)"]


def test_the_tensor_is_a_pytree_with_gamma_as_its_only_leaf(waves):
    _, _, tensor = waves
    leaves, tree = jax.tree.flatten(tensor)
    assert len(leaves) == 1
    again = jax.tree.unflatten(tree, leaves)
    assert (again.port, again.waves) == (tensor.port, tensor.waves)
    np.testing.assert_array_equal(again.gamma_nl, tensor.gamma_nl)


# --- the cubic anisotropy ---------------------------------------------------------

RHO = 1.27  # silicon, 1.2 to 2.4 um (Zhang et al. 2007): rho = 3 / 2.36 = 1.271


def _axes(rotation):
    return tuple(tuple(float(v) for v in row) for row in rotation)


def _random_axes(seed):
    from scipy.spatial.transform import Rotation

    return _axes(Rotation.random(random_state=seed).as_matrix())


def _cubic(rho, axes=None, **kwargs):
    return gfem.Material(
        eps_r=N**2, nonlinear_index=N2, nonlinear_anisotropy=rho, axes=axes, **kwargs
    )


def _quartic_factor(axes, direction):
    """:math:`\\rho`-free part of the factor :math:`\\sum_s (a_s\\cdot u)^4`."""
    return float(np.sum((np.asarray(axes) @ np.asarray(direction)) ** 4))


class _Uniform:
    """Stand-in waves: one point per wave with a given field, weight 1."""

    def __init__(self, fields, material):
        self.fields = jnp.asarray(fields, dtype=complex)[:, None, None, :]
        n = len(self.fields)
        self.port = "p1"
        self.modes = (1,) * n
        self.frequency = np.full(n, 8e9)
        self.vectors = jnp.arange(n)
        self.weights = jnp.ones((1, 1))
        self.materials = material

    def row(self, vector):
        return self.fields[vector]

    def field(self):
        return self.fields


@pytest.fixture
def uniform(monkeypatch, filled):
    """``measure(fields, rho, axes, what)``: the overlap of given uniform fields."""
    model, mode, _ = filled

    def measure(fields, rho, axes=None, coupling=False, index=None):
        axes = np.eye(3) if axes is None else axes
        fields = np.asarray(fields, complex)
        n = len(fields)
        has = rho is not None
        kerr_materials = kerr._Kerr(
            n2=jnp.asarray([N2]),
            beta=jnp.zeros(1),
            eps=jnp.full((n, 1), N**2),
            keep=np.ones(1, bool),
            rho=jnp.asarray([rho]) if has else None,
            axes=np.asarray(axes)[None] if has else None,
            index=None
            if index is None
            else jnp.asarray(index)[None, None] * jnp.ones((n, 1, 1, 1)),
        )
        waves = _Uniform(fields, kerr_materials)
        monkeypatch.setattr(kerr, "_waves", lambda *_, **__: waves)
        if coupling:
            return np.asarray(gfem.kerr_coupling(model, mode).gamma_nl)
        return np.asarray(gfem.kerr_nonlinearity(model, mode).gamma_nl)

    return measure


X, Y, Z = np.eye(3)


def test_rho_one_is_the_isotropic_overlap_whatever_the_axes(guide_mesh, filled):
    _, mode, kn = filled
    base = gfem.Material(eps_r=N**2, nonlinear_index=N2, two_photon_absorption=TPA)
    reference = np.asarray(
        gfem.kerr_coupling(guide_model(guide_mesh, base), mode).gamma_nl
    )
    for seed in range(3):
        material = _cubic(1.0, _random_axes(seed), two_photon_absorption=TPA)
        other = guide_model(guide_mesh, material)
        # 1 * q + 3 (1 - 1) * c is q in floating point, up to the compiler's
        # reordering: round-off 1e-16, gate 1e-12
        got = gfem.kerr_nonlinearity(other, mode)
        np.testing.assert_allclose(got.gamma_nl, kn.gamma_nl, rtol=1e-12)
        tensor = np.asarray(gfem.kerr_coupling(other, mode).gamma_nl)
        np.testing.assert_allclose(tensor, reference, rtol=1e-12, atol=0)


def test_a_material_with_axes_and_one_permittivity_is_accepted_without_rho(
    guide_mesh, filled
):
    _, mode, kn = filled
    material = gfem.Material(
        eps_r=N**2,
        nonlinear_index=N2,
        two_photon_absorption=TPA,
        axes=_random_axes(5),
    )
    got = gfem.kerr_nonlinearity(guide_model(guide_mesh, material), mode)
    np.testing.assert_allclose(got.gamma_nl, kn.gamma_nl, rtol=1e-12)


def test_uniform_field_ratios_follow_rho(uniform):
    # each ratio is a closed form of rho; the operations are a few products of
    # numbers of order 1, so round-off is 1e-16 and the gate 1e-13
    def q(field):
        return uniform([field], RHO)[0].real

    lin = q(X)
    assert q((X + Y) / np.sqrt(2)) / lin == pytest.approx((1 + RHO) / 2, rel=1e-13)
    assert q((X + Y + Z) / np.sqrt(3)) / lin == pytest.approx(
        (1 + 2 * RHO) / 3, rel=1e-13
    )
    circular = q((X + 1j * Y) / np.sqrt(2)) / lin
    assert circular == pytest.approx(0.5 + RHO / 6, rel=1e-13)
    assert circular == pytest.approx(0.711667, abs=1e-6)
    assert q((X + Y) / np.sqrt(2)) / lin == pytest.approx(1.135, abs=1e-12)
    assert q((X + Y + Z) / np.sqrt(3)) / lin == pytest.approx(1.18, abs=1e-12)


def test_the_cubic_form_reads_the_crystal_frame_not_the_global_one(uniform):
    # a linear field along the first row of a rotated frame is its [100]
    axes = np.asarray(_random_axes(7))
    along = uniform([axes[0]], RHO, axes)[0]
    diagonal = uniform([(axes[0] + axes[1]) / np.sqrt(2)], RHO, axes)[0]
    assert diagonal / along == pytest.approx((1 + RHO) / 2, rel=1e-13)
    # the same field in the global frame is not a cube axis
    assert abs(uniform([axes[0]], RHO)[0].real / along.real - 1) > 0.01


def test_orthogonal_polarizations_cross_phase_follows_rho(uniform):
    def ratios(a, b):
        g = uniform([a, b], RHO, coupling=True)
        spm = g[0, 0, 0, 0]
        return (g[0, 0, 1, 1] + g[0, 1, 0, 1]) / spm, g[0, 1, 1, 0] / spm

    xpm, coherent = ratios(X, Y)
    assert xpm == pytest.approx(2 * RHO / 3, rel=1e-13)
    assert coherent == pytest.approx(RHO / 3, rel=1e-13)
    rho = 3 / 2.36
    g = uniform([(X + Y) / np.sqrt(2), (X - Y) / np.sqrt(2)], rho, coupling=True)
    spm = g[0, 0, 0, 0]
    assert (g[0, 0, 1, 1] + g[0, 1, 0, 1]) / spm == pytest.approx(0.507463, abs=1e-6)
    assert g[0, 1, 1, 0] / spm == pytest.approx(0.253731, abs=1e-6)


def test_the_coupling_diagonal_is_the_coefficient_with_rho(uniform):
    fields = [(X + 2 * Y + 0.5j * Z) / 2]
    diagonal = uniform(fields, RHO, _random_axes(3), coupling=True)[0, 0, 0, 0]
    single = uniform(fields, RHO, _random_axes(3))[0]
    assert diagonal == pytest.approx(single, rel=1e-13, abs=0)


@pytest.mark.parametrize(
    "axes",
    [
        np.eye(3),
        np.diag([1.0, -1.0, 1.0]),
        np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]]),
        np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]]),
    ],
    ids=["identity", "sign", "quarter turn", "cyclic"],
)
def test_te10_is_unchanged_by_the_cube_axes_along_its_field(guide_mesh, filled, axes):
    _, mode, kn = filled
    other = guide_model(guide_mesh, _cubic(RHO, _axes(axes), two_photon_absorption=TPA))
    got = gfem.kerr_nonlinearity(other, mode).gamma_nl
    # E is y up to the mesh's x component (measured 1.07e-5 relative in gamma at
    # order 2, gate 2e-5); a signed permutation sends y to an axis, the factor is 1
    np.testing.assert_allclose(got, kn.gamma_nl, rtol=2e-5)


def test_te10_at_45_degrees_has_the_factor_one_plus_rho_over_two(guide_mesh, filled):
    _, mode, _ = filled
    c = np.cos(np.pi / 4)
    about_z = [[c, c, 0], [-c, c, 0], [0, 0, 1]]
    about_x = [[1, 0, 0], [0, c, c], [0, -c, c]]
    for axes in (about_z, about_x):
        other = guide_model(guide_mesh, _cubic(RHO, _axes(axes)))
        plain = guide_model(guide_mesh, gfem.Material(eps_r=N**2, nonlinear_index=N2))
        got = complex(gfem.kerr_nonlinearity(other, mode).gamma_nl[0])
        ref = complex(gfem.kerr_nonlinearity(plain, mode).gamma_nl[0])
        factor = RHO + (1 - RHO) * _quartic_factor(axes, Y)
        assert factor == pytest.approx((1 + RHO) / 2, rel=1e-15)
        assert got.real / ref.real == pytest.approx(factor, rel=2e-5)


def test_gamma_is_affine_in_rho_exactly(guide_mesh, filled):
    _, mode, _ = filled
    axes = _random_axes(11)

    def gamma(rho):
        other = guide_model(guide_mesh, _cubic(rho, axes, two_photon_absorption=TPA))
        return complex(gfem.kerr_nonlinearity(other, mode).gamma_nl[0])

    g1, g2, g3 = gamma(1.0), gamma(2.0), gamma(RHO)
    # the third is the affine combination of the first two: 1e-16 round-off
    # of a sum of a few terms, gate 1e-12
    assert g3 == pytest.approx(g1 + (RHO - 1) * (g2 - g1), rel=1e-12, abs=0)


def test_an_ez_line_port_has_the_factor_of_the_z_axis(slab):
    model, mode = slab
    axes = _random_axes(4)
    base = complex(gfem.kerr_nonlinearity(model, mode).gamma_nl[0])
    cubic = {
        name: dataclasses.replace(m, nonlinear_anisotropy=RHO, axes=axes)
        for name, m in model.materials.items()
    }
    got = complex(
        gfem.kerr_nonlinearity(model.replace(materials=cubic), mode).gamma_nl[0]
    )
    # the field has one component, along z: every term of the integral is
    # scaled by the same factor; round-off of the sum 1e-16, gate 1e-12
    factor = RHO + (1 - RHO) * _quartic_factor(axes, Z)
    assert abs(factor - 1) > 0.05  # the frame is not a cube axis along z
    assert got.real == pytest.approx(factor * base.real, rel=1e-12, abs=0)


def test_a_field_along_a_cube_axis_sees_the_axis_value_for_every_rho(uniform):
    base = uniform([Y], None)[0]
    assert uniform([Y], 1.7)[0].real == pytest.approx(base.real, rel=1e-13, abs=0)


# --- the refusals of the anisotropy -------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), [1.0, 1.0], 1j, "1.2"])
def test_a_nonlinear_anisotropy_must_be_one_finite_real_number(bad):
    with pytest.raises(ValueError, match="nonlinear_anisotropy"):
        gfem.Material(eps_r=N**2, nonlinear_index=N2, nonlinear_anisotropy=bad)


def test_a_nonlinear_anisotropy_needs_a_kerr_property():
    with pytest.raises(ValueError, match=r"nonlinear_anisotropy.*no nonlinear_index"):
        gfem.Material(eps_r=N**2, nonlinear_anisotropy=1.27)
    gfem.Material(eps_r=N**2, two_photon_absorption=TPA, nonlinear_anisotropy=1.27)


def test_a_nonlinear_anisotropy_beside_an_anisotropic_permittivity_is_refused():
    with pytest.raises(ValueError, match=r"nonlinear_anisotropy.*optically isotropic"):
        gfem.Material(
            eps_r=(2.0, 2.0, 2.3), nonlinear_index=N2, nonlinear_anisotropy=1.27
        )


def test_a_nonlinear_anisotropy_is_refused_under_a_trace():
    def build(rho):
        gfem.Material(eps_r=N**2, nonlinear_index=N2, nonlinear_anisotropy=rho)
        return rho

    with pytest.raises(NonDifferentiableError, match="nonlinear_anisotropy"):
        jax.grad(build)(1.27)


def test_a_field_blending_an_anisotropy_is_refused():
    kerr_end = _cubic(1.27)
    other = gfem.Material(eps_r=1.0, nonlinear_index=0.0)
    with pytest.raises(ValueError, match="nonlinear_anisotropy"):
        gfem.MaterialField.interpolate(jnp.asarray([0.5]), other, kerr_end)


def test_the_anisotropy_survives_a_pytree_and_a_job_round_trip():
    material = _cubic(RHO, _random_axes(2))
    leaves, tree = jax.tree.flatten(material)
    assert jax.tree.unflatten(tree, leaves).nonlinear_anisotropy == RHO
    assert "nonlinear_anisotropy=1.27" in repr(material)


# --- birefringent regions: the index matrix N ----------------------------------------

NO, NE = 2.21111101, 2.13755965  # lithium niobate at 1550 nm (ordinary, extraordinary)
LN_N2 = 1.8e-19


def _index_matrix(values, axes=None):
    axes = np.eye(3) if axes is None else np.asarray(axes)
    return axes.T @ np.diag(values) @ axes


def _ln(n_y, axes=None):
    # the guide's TE10 field is along y: the second principal value is its index
    values = (NO, n_y, NO) if n_y != NO else (NE, NO, NE)
    return gfem.Material(
        eps_r=tuple(v**2 for v in values), nonlinear_index=LN_N2, axes=axes
    )


@pytest.mark.parametrize("n_y", [NO, NE], ids=["ordinary", "extraordinary"])
def test_a_birefringent_filled_te10_follows_its_closed_form(guide_mesh, n_y):
    model = guide_model(guide_mesh, _ln(n_y))
    mode = mode_of(model, [F], order=gfem.SolverOptions(order=3))
    got = gfem.kerr_nonlinearity(model, mode).gamma_nl[0]
    k0 = k0_of(F)
    n_eff = float(mode.gamma.imag[0] / k0)
    reference = 1.5 * k0 * LN_N2 / (A * B) * (n_y / n_eff) ** 2
    # the field's own mesh error against the continuum form: 2.8e-6 and 1.1e-5
    # at order 2, 1.5e-7 and 1.2e-7 measured at order 3 (ordinary, extraordinary);
    # gate 1e-6 at order 3
    assert got.real == pytest.approx(reference, rel=1e-6, abs=0)
    assert got.imag == 0.0


def test_the_ln_guide_of_the_decision_numbers_is_the_same_closed_form():
    # the micron guide of the design notes (a = 2 um, 1550 nm) has no mesh here: the
    # closed form itself, evaluated at its two indices; the quoted digits are the
    # formula's to 1e-9
    a, b, wavelength = 2e-6, 1e-6, 1.55e-6
    k0 = 2 * np.pi / wavelength

    def gamma(n_y):
        n_eff = math.sqrt(n_y**2 - (wavelength / (2 * a)) ** 2)
        return 1.5 * k0 * LN_N2 / (a * b) * (n_y / n_eff) ** 2

    assert gamma(NO) == pytest.approx(0.5645852829, rel=1e-9)
    assert gamma(NE) == pytest.approx(0.5658403721, rel=1e-9)


def test_an_isotropic_index_matrix_is_the_squared_index_overlap(uniform):
    rng = np.random.default_rng(3)
    fields = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    matrix = N * np.eye(3)
    # N = n I gives the operations of n^2 S with the order of the products changed:
    # round-off 1e-16 on numbers of order 1, gate 1e-13
    np.testing.assert_allclose(
        uniform(fields, None, index=matrix),
        uniform(fields, None),
        rtol=1e-13,
        atol=0,
    )
    np.testing.assert_allclose(
        uniform(fields, None, coupling=True, index=matrix),
        uniform(fields, None, coupling=True),
        rtol=1e-13,
        atol=1e-13 * np.abs(uniform(fields, None, coupling=True)).max(),
    )


def test_the_index_matrix_overlap_is_exact_along_each_principal_axis(uniform):
    axes = np.asarray(_random_axes(9))
    values = (2.0, 2.2, 2.4)
    matrix = _index_matrix(values, axes)
    for k, n_k in enumerate(values):
        got = uniform([axes[k]], None, index=matrix)[0]
        # the scalar n2 is the index change along each principal axis: n_k^2 S
        np.testing.assert_allclose(
            got, uniform([axes[k]], None, index=n_k * np.eye(3))[0], rtol=1e-13
        )
    # halfway between two axes the quadratic form is the mean index (n_0 + n_2) / 2
    mixed = uniform([(axes[0] + axes[2]) / np.sqrt(2)], None, index=matrix)[0]
    along = uniform([axes[0]], None, index=matrix)[0]
    ratio = ((values[0] + values[2]) / 2 / values[0]) ** 2
    assert mixed.real / along.real == pytest.approx(ratio, rel=1e-13)


def test_the_index_matrix_diagonal_of_the_coupling_is_the_coefficient(uniform):
    rng = np.random.default_rng(5)
    fields = rng.normal(size=(1, 3)) + 1j * rng.normal(size=(1, 3))
    matrix = _index_matrix((2.0, 2.2, 2.4), _random_axes(2))
    tensor = uniform(fields, None, coupling=True, index=matrix)[0, 0, 0, 0]
    single = uniform(fields, None, index=matrix)[0]
    assert tensor == pytest.approx(single, rel=1e-13, abs=0)


def test_the_index_matrix_coupling_has_the_symmetries_of_s(uniform):
    # S is symmetric in its first pair and in (c, d): gamma_abcd = conj(T) with T
    # symmetric in b <-> c at one frequency (here N is the same for every wave)
    rng = np.random.default_rng(8)
    fields = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    matrix = _index_matrix((2.0, 2.2, 2.4), _random_axes(6))
    g = uniform(fields, None, coupling=True, index=matrix)
    # the entries are ~1e-22: a fixed atol would pass anything, so it scales
    np.testing.assert_allclose(
        g, np.swapaxes(g, 1, 2), rtol=1e-13, atol=1e-13 * np.abs(g).max()
    )


def test_a_law_beside_a_rho_is_refused_and_the_constant_route_works(guide_mesh):
    # a dispersive core cannot carry axes, so a rho beside a law would lose its
    # frame: refused. The route is the band centre's constant permittivity with
    # axes, in a second model of the same mesh, whose centre row differs from the
    # law's by the factor alone (the same permittivity there)
    eps = complex(LORENTZ(jnp.asarray(F)))
    axes = _random_axes(12)
    with pytest.raises(ValueError, match=r"nonlinear_anisotropy.*constant eps_r"):
        gfem.Material(eps_r=LORENTZ, nonlinear_index=N2, nonlinear_anisotropy=RHO)
    with pytest.raises(ValueError, match="axes"):
        gfem.Material(eps_r=LORENTZ, nonlinear_index=N2, axes=axes)
    law = gfem.Material(eps_r=LORENTZ, nonlinear_index=N2)
    constant = gfem.Material(
        eps_r=eps.real, nonlinear_index=N2, nonlinear_anisotropy=RHO, axes=axes
    )
    model = guide_model(guide_mesh, law)
    mode = mode_of(model, [F])
    a = gfem.kerr_nonlinearity(model, mode).gamma_nl
    b = gfem.kerr_nonlinearity(model.with_material("air", constant), mode).gamma_nl
    factor = RHO + (1 - RHO) * _quartic_factor(axes, Y)
    assert abs(factor - 1) > 0.05
    # the field is y up to the mesh's x component (measured 1.07e-5 at order 2)
    assert b.real[0] / a.real[0] == pytest.approx(factor, rel=2e-5)


def test_a_birefringent_region_beside_a_rho_region_is_refused(tmp_path):
    path = pml_cases.strip_guide(
        tmp_path / "strip.msh", 0.4, thickness=0, quarter=False, length=0.15
    )
    base = pml_cases.strip_model(path)
    core = gfem.Material(
        eps_r=pml_cases.STRIP_CORE**2, nonlinear_index=6e-18, nonlinear_anisotropy=RHO
    )
    clad = gfem.Material(eps_r=(2.1, 2.1, 2.3), nonlinear_index=2.6e-20)
    model = base.replace(materials={"core": core, "clad": clad})
    mode = gfem.solve(model, gfem.PortMode("p1", pml_cases.STRIP_F), options=ORDER)
    with pytest.raises(
        NotImplementedError, match=r"nonlinear_anisotropy.*birefringent"
    ):
        gfem.kerr_nonlinearity(model, mode)
    # alone, the birefringent cladding is accepted
    alone = base.replace(
        materials={
            "core": dataclasses.replace(core, nonlinear_anisotropy=None),
            "clad": clad,
        }
    )
    assert gfem.kerr_nonlinearity(alone, mode).gamma_nl[0].real > 0


def test_uniform_field_refuses_a_material_with_dropped_properties():
    cubic = _cubic(RHO)
    with pytest.raises(ValueError, match="nonlinear_anisotropy"):
        gfem.MaterialField.uniform("air", cubic)
    spectrum = gfem.RamanSpectrum(
        [0.0, 1e12, 2e12], [0.0, 1.0, 0.0], 2e14, "isotropic", [0.0, 0.5, 0.0]
    )
    with pytest.raises(ValueError, match="raman_spectrum"):
        gfem.MaterialField.uniform(
            "air", gfem.Material(eps_r=N**2, raman_spectrum=spectrum)
        )
