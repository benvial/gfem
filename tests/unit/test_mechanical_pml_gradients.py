r"""Gradients of the mechanical eigenmodes behind an elastic layer, against differences.

A `gfem.PML` of the mechanical eigenmode does not change what differentiates:
:math:`\omega` (so frequency and Q) and the mode fields, by the stiffness,
density and quality factor of the physical regions and by the node coordinates,
in a 2D section at any wavenumber and in a 3D body. The derivatives are those of
the forward as defined: the layer's stretch is a coefficient of the imported
coordinates, so it rides with the nodes, and a node map may move the layer's.
Each is checked against fourth-order central differences at the gate of
`finite_differences`.

Noise. The 2D gradient files budget a round-off of :math:`\delta = 10^{-12}`
relative on a sample, a 3D body one of :math:`\delta = 1.2 \cdot 10^{-10}` (see
`test_mechanical_3d_gradients`). A step :math:`h` on an input of relative slope
:math:`s` (the relative change of the output per unit of the input) rounds at
:math:`1.5\,\delta / (h s)` and truncates the fourth-order difference of
:math:`\omega \propto (1+p)^a` at :math:`0.22\,h^4` for :math:`a = 1/2` (the
modulus), :math:`2\,h^4` for :math:`a = -1/2` (the density) and :math:`4\,h^4`
for :math:`a = -1` (the quality factor on :math:`\mathrm{Im}\,\omega`, a stretch
of the whole mesh, which scales :math:`\omega` as :math:`1/s`). At the step
:math:`h = 10^{-2}` truncation is at most :math:`4 \cdot 10^{-8}`; round-off is at
most :math:`1.5 \cdot 10^{-12} / (10^{-2} s) = 1.5 \cdot 10^{-10} / s` in a
section, under :math:`10^{-7}` for :math:`s \ge 1.5 \cdot 10^{-3}`, and
:math:`1.5 \cdot 1.2 \cdot 10^{-10} / (10^{-2} s) = 1.8 \cdot 10^{-8} / s` in a
body, under :math:`10^{-7}` for :math:`s \ge 0.18` at the step :math:`10^{-2}` (the
body's step is smaller, see its cases). The slopes measured are in the comments of
each case. An output that an input barely moves (the real part of :math:`\omega`
under the quality factor) is read against :math:`10^{-3}` of the largest derivative
of its group, as in the other gradient files.
"""

import dataclasses

import elastic_pml_cases as ec
import finite_differences as fd
import jax
import jax.numpy as jnp
import mechanical_cases as mc
import numpy as np
import pytest

import gfem

#: The step of the leaves and of a stretch of the whole mesh.
STEP = 1e-2
#: The rough Q estimate the complex shift of a leaky search expects.
Q_SHIFT = 150.0
#: The quality factor of every region of the lossy columns, layer included, so that
#: the layer's loss pieces (a stretched loss) are in the pencil.
QUALITY = 200.0
OMEGA = {"omega": [0, 1]}
#: The stiffness the film's leaf starts from.
C_FILM = np.asarray(
    gfem.isotropic_stiffness(2 * ec.MU_LAYER * (1 + ec.NU), ec.NU), float
)
#: One unit of the stiffness direction, Pa: the Young modulus of the film.
UNIT = 2 * ec.MU_LAYER * (1 + ec.NU)


def _omega(result):
    """``[Re omega, Im omega]`` of the nearest mode."""
    omega = result.omega[0]
    return jnp.stack([omega.real, omega.imag])


def _solve(column, order, model=None, **kwargs):
    if column.wavenumber:
        kwargs["wavenumber"] = column.wavenumber
    problem = gfem.MechanicalEigenmode(
        column.target, pml=column.pml, clamped=column.clamped, **kwargs
    )
    return gfem.solve(
        model or column.model, problem, options=gfem.SolverOptions(order=order)
    )


def _film(model, **values):
    """``model`` with the leaves ``values`` of the film as a `gfem.MaterialField`.

    The layer's regions stay plain `gfem.Material` (a layer takes no field).
    """
    fields = dict(model.materials)
    fields["layer"] = dataclasses.replace(
        gfem.MaterialField.uniform("layer", model.materials["layer"]), **values
    )
    return model.replace(materials=fields)


def _stretched(mesh, s, axis):
    """``mesh`` with its coordinate ``axis`` scaled by ``s``, layer included."""
    nodes = jnp.asarray(mesh.nodes)
    factor = jnp.ones(nodes.shape[1]).at[axis].set(s)
    return mesh.with_nodes(nodes * factor)


def _direction(seed, rows):
    """A unit symmetric 6 x 6 direction that weighs the Voigt ``rows`` the mode reads.

    A random symmetric matrix, with the diagonal of ``rows`` raised to a
    component of the direction at least 0.5, so that the stiffness moves the mode
    with a relative slope far above the budget's floor.
    """
    d = mc.symmetric_direction(seed)
    for r in rows:
        d[r, r] = np.sign(d[r, r]) * max(abs(d[r, r]), 0.5)
    return d / np.linalg.norm(d)


# --- the leaky antiplane column: a real shift -----------------------------------------


@pytest.fixture(scope="module")
def antiplane(tmp_path_factory):
    path = tmp_path_factory.mktemp("antiplane") / "column.msh"
    return ec.leaky_column(path, 0.2, quality=QUALITY)


#: The antiplane mode u_z(y) reads c_44 (the yz shear modulus) of the film, whose
#: share of the mode's energy is about a half.
ANTIPLANE_DIRECTION = _direction(3, [3])
#: Where the fields are read (m): in the film (0 < y < 1 um).
POINT = np.array([0.23e-6, 0.37e-6])


def _antiplane_stiffness(p):
    return C_FILM + UNIT * p


@pytest.mark.parametrize("name", ["stiffness", "mass_density", "quality"])
def test_the_frequency_of_a_leaky_column_matches_finite_differences(antiplane, name):
    # Measured relative slopes of (Re omega, Im omega) per unit of the leaf: 0.58 and
    # 0.25 for the stiffness direction, -0.50 and -0.78 for the density, about 0 (read
    # against its floor) and -0.44 for the quality factor. Round-off at most
    # 1.5 * 1e-12 / (1e-2 * 0.25) = 6e-10; truncation 0.22 h^4 = 2.2e-9 (modulus),
    # 2 h^4 = 2e-8 (density) and 4 h^4 = 4e-8 (quality factor on Im omega). Measured
    # errors at h = 1e-2: 6.3e-9 and 8.0e-9 (stiffness), 2.0e-8 and 3.4e-8 (density),
    # 1.2e-7 against the floor and 4.0e-8 (quality factor): the first of these is
    # the round-off of Re omega over 1e-3 of the derivative of Im omega, 1.5 * 1e-12
    # / (1e-2 * 1e-3 * 0.44) = 3.4e-7 at most
    fields = {
        "stiffness": lambda p: {"stiffness": _antiplane_stiffness(p)},
        "mass_density": lambda p: {"mass_density": ec.RHO_LAYER * (1 + p)},
        "quality": lambda p: {"mechanical_quality_factor": QUALITY * (1 + p)},
    }[name]
    shape = (6, 6) if name == "stiffness" else ()

    def objective(p):
        return _omega(_solve(antiplane, 3, _film(antiplane.model, **fields(p))))

    direction = ANTIPLANE_DIRECTION if name == "stiffness" else np.asarray(1.0)
    fd.assert_jacobian_matches_by_group(
        objective, jnp.zeros(shape), [direction], STEP, OMEGA
    )


def test_a_stretch_of_the_whole_mesh_through_the_layer_matches_finite_differences(
    antiplane,
):
    # the layer's nodes move with the film's, and its stretch rides with them. Every
    # length scales by s, K by 1/s and M by s: omega scales as 1/s (relative slope
    # -1), the truncation of the fourth-order difference is 4 h^4 = 4e-8 at h = 1e-2
    # (measured 4.00e-8 on both parts) and the round-off 1.5 delta / (h s) = 1.5e-10
    materials = antiplane.model.materials

    def objective(s):
        mesh = _stretched(antiplane.model.mesh, s, 1)
        return _omega(_solve(antiplane, 3, gfem.Model(mesh, materials=materials)))

    fd.assert_jacobian_matches_by_group(
        objective, jnp.asarray(1.0), [np.asarray(1.0)], STEP, OMEGA
    )


def _readings(result):
    r""":math:`|u(p)|^2` and :math:`\mathrm{Re}\,S_{yz}` of mode 0 at ``POINT``.

    The first is invariant under the phase gauge; the strain is not: the gauge
    leaves the largest physical displacement coefficient real, and the
    normalization is that of the physical mass.
    """
    u = result.displacement(0)
    intensity = sum(jnp.abs(u[k].at(POINT)) ** 2 for k in "xyz")
    strain = gfem.strain(u, POINT[None])[0, 3]
    return jnp.stack([intensity, strain.real])


def test_the_field_readings_of_a_leaky_column_match_finite_differences(antiplane):
    # |u|^2 and Re S_yz of the mode at a film point, against the film's stiffness: the
    # normalization (over the physical mass, not the stretched layer's) and the gauge
    # (the largest physical coefficient) are in the derivative. Measured slopes -1.23
    # and -0.58: round-off at most 1.5e-12 / (1e-2 * 0.58) = 2.6e-10 plus the
    # eigenvector's own noise, truncation a few h^4 times the growth of the
    # derivatives (a reading moves by more than the frequency does). Measured errors
    # at h = 1e-2: 1.1e-7 and 7.7e-8, falling as h^4 (8.6e-10 and 8.4e-10 at 3e-3)
    def objective(p):
        model = _film(antiplane.model, stiffness=_antiplane_stiffness(p))
        return _readings(_solve(antiplane, 3, model))

    fd.assert_jacobian_matches_by_group(
        objective,
        jnp.zeros((6, 6)),
        [ANTIPLANE_DIRECTION],
        STEP,
        {"|u|^2": [0], "S_yz": [1]},
    )


# --- the Love mode at q = 2 / um: a complex shift -------------------------------------


@pytest.fixture(scope="module")
def love(tmp_path_factory):
    path = tmp_path_factory.mktemp("love") / "column.msh"
    return ec.leaky_column(path, 0.2, case="love")


#: The Love mode u_x(y) e^{-iqz} reads c_55 (xz) and c_66 (xy) of the film.
LOVE_DIRECTION = _direction(5, [4, 5])


def test_the_frequency_of_a_leaky_love_mode_matches_finite_differences(love):
    # q != 0: the pencil is a general matrix, the left vector comes from a transposed
    # search and the field adjoint from an LU, with the stretched q pieces. Measured
    # relative slopes of (Re omega, Im omega) per unit of the direction -0.52 and
    # -0.21: round-off at most 1.5e-12 / (1e-2 * 0.21) = 7e-10, truncation at most
    # 4 h^4 = 4e-8; measured errors 7.5e-9 and 1.0e-7 at h = 1e-2 (6e-11 and 8e-10 at
    # 3e-3: the h^4 fall of truncation, so round-off sits below those)
    def objective(p):
        model = _film(love.model, stiffness=C_FILM + UNIT * p)
        return _omega(_solve(love, 3, model, target_quality_factor=Q_SHIFT))

    fd.assert_jacobian_matches_by_group(
        objective, jnp.zeros((6, 6)), [LOVE_DIRECTION], STEP, OMEGA
    )


# --- a 3D column stretched along z ----------------------------------------------------


@pytest.fixture(scope="module")
def column3d(tmp_path_factory):
    path = tmp_path_factory.mktemp("body") / "body.msh"
    return ec.leaky_column(path, BODY_MAXH, dim=3)


#: The body's element size, um: 78 tetrahedra and 243 nodes. At 0.2 um (1200 tetrahedra,
#: 2700 nodes) one solve takes 18 s, 17 of them the assembly of the stretched
#: pieces, and a check of seven solves is 100 s; 0.4 um takes 1.2 s. The gradient
#: is that of the discrete problem, so the coarse mesh only changes the frequency
#: by 0.2 %.
BODY_MAXH = 0.4
#: The body's step. The mode has neighbours at +3.3 % and -19 % of its frequency at
#: this mesh; a fixed target selects the nearest mode, so the samples (omega moves
#: with a slope of -0.6 to -1 per unit) must stay within half the gap: 2 h * 1 <
#: 1.6 %, which h = 1e-2 breaks (2 %: the nearest mode changes, a jump of 20 % in
#: the difference) and h = 3e-3 keeps (0.6 %).
BODY_STEP = 3e-3
#: The thickness-shear mode u_x(z) reads c_55 (xz) of the film.
BODY_DIRECTION = _direction(11, [4])
#: A film point (m) at least 0.03 um from the planes of the Kuhn grid, more than
#: the 0.0086 um a stretch of 1e-2 moves it.
BODY_POINT = np.array([0.07e-6, 0.13e-6, 0.5e-6])


def test_the_frequency_of_a_leaky_body_matches_finite_differences_in_the_stiffness(
    column3d,
):
    # a 3D body, delta = 1.2e-10 (see the module docstring), step BODY_STEP = 3e-3.
    # Measured slopes (Re omega, Im omega) -0.60 and -0.0024 (the leak's impedance
    # barely depends on the film's stiffness). Round-off on Re: 1.5 * 1.2e-10 / (3e-3 *
    # 0.6) = 1.0e-7; on Im, read against 1e-3 of the derivative of Re: 1.5 delta
    # (Im omega / Re omega) / (h * 1e-3 * 0.6) = 1.5 * 1.2e-10 * 3.2e-3 / (3e-3 * 6e-4)
    # = 3.2e-7. Truncation of Re: 0.22 h^4 = 2e-11; of Im, measured 3.7e-9 against the
    # floor (4.2e-7 at h = 1e-2, a fall of 114 for the 123 of h^4). Measured errors:
    # 4.9e-11 and 3.7e-9
    def objective(p):
        model = _film(column3d.model, stiffness=C_FILM + UNIT * p)
        return _omega(_solve(column3d, 2, model))

    fd.assert_jacobian_matches_by_group(
        objective, jnp.zeros((6, 6)), [BODY_DIRECTION], BODY_STEP, OMEGA
    )


def test_the_frequency_of_a_leaky_body_matches_finite_differences_in_a_z_stretch(
    column3d,
):
    # the stretch of the whole z axis, layer included: omega scales as 1/s (relative
    # slope -1, measured -0.9993 and -0.9969): truncation 4 h^4 = 3.2e-10 at h = 3e-3
    # (measured 3.2e-10 on both parts), round-off 1.5 * 1.2e-10 / 3e-3 = 6e-8
    materials = column3d.model.materials

    def objective(s):
        mesh = _stretched(column3d.model.mesh, s, 2)
        return _omega(_solve(column3d, 2, gfem.Model(mesh, materials=materials)))

    fd.assert_jacobian_matches_by_group(
        objective, jnp.asarray(1.0), [np.asarray(1.0)], BODY_STEP, OMEGA
    )


def test_a_field_reading_of_a_leaky_body_matches_finite_differences(column3d):
    # |u(p)|^2 at a film point against the film's stiffness: measured slope 26.9 (the
    # normalization over the physical mass makes it move far more than the frequency),
    # so round-off is at most 1.5 * 1.2e-10 / (3e-3 * 26.9) = 2.2e-9; truncation scales
    # as the growth of the derivatives, measured 4.6e-9 at h = 3e-3 (5.7e-7 at 1e-2,
    # a fall of 123 for the h^4 of 123)
    def objective(p):
        model = _film(column3d.model, stiffness=C_FILM + UNIT * p)
        u = _solve(column3d, 2, model).displacement(0)
        return jnp.stack([sum(jnp.abs(u[k].at(BODY_POINT)) ** 2 for k in "xyz")])

    fd.assert_jacobian_matches_by_group(
        objective,
        jnp.zeros((6, 6)),
        [BODY_DIRECTION],
        BODY_STEP,
        {"|u|^2": [0]},
    )


# --- what raises, and what it costs ---------------------------------------------------


def test_a_traced_target_quality_factor_refuses_at_construction():
    def build(x):
        problem = gfem.MechanicalEigenmode(
            1e9, pml=gfem.PML(["pml"]), target_quality_factor=x
        )
        return problem.target_quality_factor

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"target_quality_factor.*fixed"
    ):
        jax.grad(build)(100.0)


def test_a_field_cotangent_costs_one_factorization_per_mode(tmp_path, monkeypatch):
    # with a layer the forward of a lossless-or-lossy solve at q = 0 factorizes once
    # (the shift); the filter's left vectors are the right ones there, so the
    # backward of omega alone adds none, and each mode with a field cotangent one
    from gfem._engine import mechanical_eigen

    count = [0]
    real = mechanical_eigen.DirectHandle

    def counted(*args, **kwargs):
        count[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(mechanical_eigen, "DirectHandle", counted)
    column = ec.leaky_column(tmp_path / "column.msh", 0.2, quality=QUALITY)
    problem = gfem.MechanicalEigenmode(
        column.target, n_modes=2, pml=column.pml, clamped=column.clamped
    )
    options = gfem.SolverOptions(order=2)

    def factorizations(objective):
        count[0] = 0
        fd._alone(jax.grad(objective), 1.0)
        return count[0]

    def solved(scale):
        model = _film(column.model, mass_density=ec.RHO_LAYER * scale)
        return gfem.solve(model, problem, options=options)

    def omega(scale):
        return jnp.real(solved(scale).omega).sum()

    def one_field(scale):
        u = solved(scale).displacement(0)
        return sum(jnp.abs(u[k].at(POINT)) ** 2 for k in "xyz")

    def both_fields(scale):
        result = solved(scale)
        u = [result.displacement(i)["z"].at(POINT) for i in range(2)]
        return sum(jnp.abs(v) ** 2 for v in u)

    assert factorizations(omega) == 1
    assert factorizations(one_field) == 2
    assert factorizations(both_fields) == 3


def test_a_thin_layer_warns_once_in_a_forward_and_its_backward(tmp_path):
    # the layer is 3.46e-3 um thick at the reference frequency target / 200, so
    # k_ref d = 0.031 is under 2 pi / 100; the backward builds the system again
    # and must not repeat the warning
    column = ec.leaky_column(tmp_path / "column.msh", 0.2)
    pml = gfem.PML(
        column.pml.regions,
        target_reflection=column.pml.target_reflection,
        reference_frequency=column.target / 200,
    )
    problem = gfem.MechanicalEigenmode(column.target, pml=pml, clamped=column.clamped)

    def omega(rho):
        model = _film(column.model, mass_density=rho)
        result = gfem.solve(model, problem, options=gfem.SolverOptions(order=2))
        return jnp.real(result.omega[0])

    with pytest.warns(gfem.ThinLayerWarning) as record:
        jax.grad(omega)(ec.RHO_LAYER)
    assert sum(issubclass(w.category, gfem.ThinLayerWarning) for w in record) == 1
