r"""Gradients of the mechanical eigenmodes of 3D bodies, against finite differences.

The 3D rule uses the backward of the 2D sections unchanged, with the tensor pass laid
out by dimension and the node pass on tetrahedra: the frequencies and the readings of
the mode fields differentiate by the stiffness, density and quality factor, the
piezoelectric tensor and permittivity, the damping frame (through the node
coordinates its depth rides with) and the node coordinates. Each is checked against
fourth-order central differences at the gate of `finite_differences`.

Noise. The 2D gradient files budget a round-off of :math:`\delta = 10^{-12}`
relative on a sample. A 3D flexural mode has about :math:`\delta = 1.2 \cdot
10^{-10}`: the quadratic forms cancel by :math:`\epsilon \lVert K \rVert / \lambda`,
large for a slender mode (measured on a 100 x 10 x 2 um cantilever, where the
fourth-order difference of the frequency is flat from :math:`h = 10^{-2}` to
:math:`5 \cdot 10^{-3}` and grows as :math:`1/h` below; a second ARPACK start vector
moves the frequency by :math:`1.1 \cdot 10^{-11}`). The stubby bodies here sit
below that, so the steps keep the budget of the slender one. A step :math:`h` on a
leaf of relative slope :math:`s` (the relative change of the output per unit of
:math:`p`) rounds at :math:`1.5\,\delta / (h\,s)`; with :math:`\delta = 1.2 \cdot
10^{-10}` this stays under :math:`10^{-7}` for :math:`h\,s \ge 1.8 \cdot 10^{-3}`:
the material directions below are random over both regions (or the clamped one) with
:math:`s \ge 0.15` (the slopes are in the comments), so the step :math:`10^{-2}` gives
:math:`h\,s \ge 1.5 \cdot 10^{-3}` (the measured slopes are at least 0.29: round-off
at most :math:`6 \cdot 10^{-8}`). The coupled plate and the framed box are small and
smooth: their measured :math:`\delta` is under :math:`10^{-14}`, which is what their
comments rest on, since the coupling moves :math:`\omega` by only :math:`k^2` of the
relative change of a tensor (slopes of :math:`10^{-3}`).
The truncation of the fourth-order difference of :math:`\omega \propto (1+p)^a` is
:math:`0.22\,h^4` for :math:`a = 1/2` (the modulus), :math:`4\,h^4` for :math:`a = -1`
(the quality factor on :math:`\mathrm{Im}\,\omega`, a transverse stretch) and
:math:`12\,h^4` for :math:`a = -2` (a stretch along the length of a beam): at
:math:`h = 10^{-2}` at most :math:`4 \cdot 10^{-8}`, and at :math:`h = 5 \cdot 10^{-3}`
for the length, :math:`7.5 \cdot 10^{-9}`. An output a leaf barely moves (the real part
of :math:`\omega` under the quality factor) is read against :math:`10^{-3}` of the
largest derivative of its group, as in the 2D files.
"""

import finite_differences as fd
import jax
import jax.numpy as jnp
import mechanical_cases as mc
import numpy as np
import piezo_cases as pc
import pytest

import gfem

#: The stiffness of the isotropic solid.
C_BASE = np.asarray(gfem.isotropic_stiffness(mc.E, mc.NU))
#: The quality factor of the lossy solids, and the step of the material leaves.
QUALITY, STEP = 50.0, 1e-2
OPTIONS = gfem.SolverOptions(order=2)
OMEGA = {"omega": [0, 1]}


def _omega(result):
    """``[Re omega, Im omega]`` of the nearest mode."""
    omega = result.omega[0]
    return jnp.stack([omega.real, omega.imag])


def _stretched(mesh, s, axes):
    """``mesh`` with its coordinates ``axes`` (an index or list) scaled by ``s``."""
    nodes = jnp.asarray(mesh.nodes)
    factor = jnp.ones(nodes.shape[1]).at[np.asarray(axes)].set(s)
    return mesh.with_nodes(nodes * factor)


# --- a stubby cantilever --------------------------------------------------------------

#: The cantilever, 20 x 4 x 2 um, clamped at x = 0 and glued at x = 10 um into the
#: regions ``a`` and ``b``: P2 on one cell per um (960 tetrahedra), lossy so that the
#: left vector is the right one.
CANTILEVER = (20.0, 4.0, 2.0)
#: Where the fields are read (m), near the clamp where the strain is largest: at
#: least 0.13 um from every face of the Kuhn grid (the planes of integer x, y, z and
#: x - y, y - z, x - z), because a stretch of the nodes by 1e-2 moves a face by up to
#: 2.3 * 1e-2 = 0.023 um at this point and the reading must stay in its element.
POINT = np.array([2.3e-6, 1.71e-6, 0.43e-6])
XX = 0


@pytest.fixture(scope="module")
def cantilever(tmp_path_factory):
    path = tmp_path_factory.mktemp("cantilever") / "cantilever.msh"
    mesh = mc.box(path, CANTILEVER, 1, regions=lambda c: "a" if c[0] < 10.0 else "b")
    material = mc.material(quality=QUALITY)
    return gfem.Model(mesh, materials={"a": material, "b": material})


#: The first flexural mode (in z, the thin side) near 4.27 MHz.
CANTILEVER_PROBLEM = gfem.MechanicalEigenmode(4.3e6, clamped={"x0": "xyz"})


#: name -> the leaf's value at the relative change ``x`` of the region.
LEAF = {
    "stiffness": lambda x: C_BASE * (1 + x),
    "mass_density": lambda x: mc.RHO * (1 + x),
    "mechanical_quality_factor": lambda x: QUALITY * (1 + x),
}


def _edit(model, name, p):
    """``model`` with the leaf ``name`` of regions ``a`` and ``b`` times ``1 + p``."""
    for k, region in enumerate("ab"):
        model = mc.edited(model, [region], **{name: LEAF[name](p[k])})
    return model


#: One direction over the three leaves and both regions, ``(leaf, region)``: the
#: leaves in turn (stiffness, density, quality factor) with the signs that make their
#: slopes add on both parts of omega (measured relative slopes of Re omega and Im
#: omega per unit of a leaf, both regions moved: 0.305 and 0.305 for the stiffness,
#: -0.395 and -0.395 for the density, -6e-5 and -0.611 for the quality factor; the
#: readings |u|^2 and S_xx move by -0.59 and -0.29 per unit of the stiffness, and
#: little by the others, so their gate is the floor of the group). The unit vector
#: has 1/sqrt(3) in each leaf, split 0.6 : 0.8 over the two regions (so that the
#: regions add), signs (+, -, -): the slope of the joint step is at least
#: (0.305 + 0.395) 0.577 = 0.40 on Re omega and 0.76 on Im omega.
JOINT = np.array([[0.6, 0.8], [-0.6, -0.8], [-0.6, -0.8]]) / np.sqrt(3.0)


def _joint(model, p):
    """``model`` with the leaves of ``LEAF`` moved by the rows of ``p`` (regions a, b)."""
    for row, name in zip(p, LEAF, strict=True):
        model = _edit(model, name, row)
    return model


def _readings(result):
    r""":math:`|u(p)|^2` and :math:`\mathrm{Re}\,S_{xx}` of mode 0.

    The first is invariant under the phase gauge; the strain is not. The gauge
    leaves the largest displacement coefficient real: the three largest, at the
    free tip, lead each other by only :math:`2.5 \cdot 10^{-6}` and :math:`5.3 \cdot
    10^{-6}`, but their phases agree to :math:`10^{-10}` (the lossy mode is real
    up to a global phase) and the gaps move by less than :math:`10^{-8}` over the
    samples, so no sample changes the coefficient or the phase it fixes.
    """
    u = result.displacement(0)
    intensity = sum(jnp.abs(u[k].at(POINT)) ** 2 for k in "xyz")
    strain = gfem.strain(u, POINT[None])[0, XX]
    return jnp.stack([intensity, strain.real])


def test_the_frequency_and_readings_of_a_cantilever_match_finite_differences_in_its_leaves(
    cantilever,
):
    # the step 1e-2 on a unit vector (0.577 per leaf): round-off 1.5 * 1.2e-10 / (1e-2
    # * 0.40) = 4.5e-8 on Re omega, 2.4e-8 on Im omega (the readings, of slopes of
    # 0.3 to 0.6 by the modulus, are read at the same budget); the truncation of the
    # fourth-order difference of omega ~ (1+p1)^(1/2) (1+p2)^(-1/2) (1+p3)^(-1) is
    # at most 4 h^4 = 4e-8 per unit of a leaf and (0.577)^4 of it here
    def objective(p):
        result = gfem.solve(_joint(cantilever, p), CANTILEVER_PROBLEM, options=OPTIONS)
        return jnp.concatenate([_omega(result), _readings(result)])

    groups = {"omega": [0, 1], "|u|^2": [2], "S_xx": [3]}
    fd.assert_jacobian_matches_by_group(
        objective, jnp.zeros((3, 2)), [JOINT], STEP, groups
    )


def test_a_stretch_of_a_cantilever_along_its_length_matches_finite_differences(
    cantilever,
):
    # omega scales as s^-2 (relative slope -2): the truncation of the fourth-order
    # difference is 12 h^4 = 7.5e-9 at h = 5e-3 (measured on the slender beam: 1.9e-6
    # at h = 0.02, 12 h^4 = 1.9e-6), the round-off 1.5 delta / (h s) = 1.5 * 1.2e-10 /
    # (5e-3 * 2) = 1.8e-8: 2.6e-8 together. The readings stay at a fixed point while
    # the elements stretch under it (measured slopes -0.97 for |u|^2 and -0.50 for
    # S_xx: round-off 1.8e-8 / 0.5 = 3.7e-8)
    materials = cantilever.materials

    def objective(s):
        model = gfem.Model(_stretched(cantilever.mesh, s, 0), materials=materials)
        result = gfem.solve(model, CANTILEVER_PROBLEM, options=OPTIONS)
        return jnp.concatenate([_omega(result), _readings(result)])

    groups = {"omega": [0, 1], "|u|^2": [2], "S_xx": [3]}
    fd.assert_jacobian_matches_by_group(
        objective, jnp.asarray(1.0), [np.asarray(1.0)], 5e-3, groups
    )


# --- a coupled plate ------------------------------------------------------------------

PLATE_CLAMPED = {"left": "x", "right": "x"}
PLATE_OPTIONS = gfem.SolverOptions(order=3)
#: An interior point of the plate (m), off every face of the grid.
PLATE_POINT = np.array([0.13e-6, 0.31e-6, 0.11e-6])


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    """The 6mm TE plate box of 4 cells across y (about 1.4 k DOFs at order 3)."""
    mesh = mc.te_plate_box(tmp_path_factory.mktemp("te") / "te.msh", 4)
    return mc.piezo_plate_model(mesh, "TE", quality=100.0)


def _plate_problem():
    # the shorted plate resonates a little below the stiffened half wave
    target = float(np.real(pc.plate_form("TE")["v"])) / (2 * pc.PLATE_THICKNESS * mc.UM)
    return gfem.MechanicalEigenmode(
        0.9 * target, clamped=PLATE_CLAMPED, ground=["bot", "top"]
    )


def _plate_edit(name, model, p):
    _, e, eps = pc.hexagonal("TE")
    if name == "piezoelectric_tensor":
        return mc.edited(model, ["pz"], piezoelectric_tensor=e + p)
    return mc.edited(model, ["pz"], eps_r=jnp.asarray(eps) + 10.0 * p)


PLATE_SHAPES = {"piezoelectric_tensor": (3, 6), "eps_r": (3,)}


def _plate_direction(name):
    """A unit direction with the weight of the thickness component (e33, eps33)."""
    d = np.random.default_rng(7).standard_normal(PLATE_SHAPES[name])
    d[(2, 2) if name == "piezoelectric_tensor" else 2] = 1.0
    return d / np.linalg.norm(d)


def _plate_potential(result):
    """``[|V|^2, Re, Im of V conj(u_y)]`` at the plate's point.

    |V|^2 is invariant under the phase gauge (the plate's mode is uniform along x
    and z, so its largest displacement coefficients tie and the gauge is not
    defined); the product with the displacement reads the potential's phase.
    """
    v = result.potential(0).at(PLATE_POINT)
    cross = v * jnp.conj(result.displacement(0)["y"].at(PLATE_POINT))
    return jnp.stack([jnp.abs(v) ** 2, cross.real, cross.imag])


@pytest.mark.parametrize("name", list(PLATE_SHAPES))
def test_the_frequency_of_a_3d_plate_matches_finite_differences(plate, name):
    # a unit of p is 1 C/m^2 of e (e33 = 1.3) or 10 of eps_r (eps33 = 29). The
    # coupling moves omega by k^2 of the relative change of e (or eps): the measured
    # relative slopes per unit of p are -6.9e-3 and 2.3e-4 (Re, Im omega) for e,
    # -2.6e-3 and -1.1e-4 for eps_r, far under the 0.15 of the cantilever, so the
    # budget 1.5 delta / (h s) with delta = 1.2e-10 would not hold at h = 1e-2. It is
    # not needed: this plate (1.4 k DOFs, a smooth thickness mode) has a measured
    # delta of 1e-15 (the difference's error against the gradient falls as h^4 from
    # 7e-8 at h = 0.1 to 3e-12 at 3e-3, and rises again as 1/h only below that:
    # 3e-11 at 1e-3, so 1.5 delta / (h s) = 3e-11 gives delta = 1.4e-16 for e). At
    # h = 1e-2 truncation (about 7e-12) and round-off (about 1e-11) both sit far
    # under 1e-6; the imaginary part, a hundred times smaller, is read against 1e-3 of
    # the larger derivative. The tensor's case also reads the potential at a point
    # (the same step and solves): |V|^2 and V conj(u_y), each against the floor of
    # its group
    def objective(p):
        result = gfem.solve(
            _plate_edit(name, plate, p), _plate_problem(), options=PLATE_OPTIONS
        )
        if name == "piezoelectric_tensor":
            return jnp.concatenate([_omega(result), _plate_potential(result)])
        return _omega(result)

    groups = dict(OMEGA)
    if name == "piezoelectric_tensor":
        groups |= {"|V|^2": [2], "V conj(u_y)": [3, 4]}
    shape = PLATE_SHAPES[name]
    fd.assert_jacobian_matches_by_group(
        objective, jnp.zeros(shape), [_plate_direction(name)], STEP, groups
    )


# --- a damping frame ------------------------------------------------------------------

BOX = (1.0, 0.8, 0.6)


@pytest.fixture(scope="module")
def framed(tmp_path_factory):
    """The 1 x 0.8 x 0.6 um box with a frame cell layer (0.2 um) on every side."""

    def region(c):
        inside = all(0.2 < c[i] < BOX[i] - 0.2 for i in range(3))
        return "core" if inside else "frame"

    path = tmp_path_factory.mktemp("frame") / "frame.msh"
    mesh = mc.box(path, BOX, 5, regions=region)
    material = mc.material()
    return gfem.Model(mesh, materials={"core": material, "frame": material})


#: The stretch of the framed box along each axis, as one direction (unit vector): the
#: slopes of (Re omega, Im omega) below add for it.
FRAME_DIRECTION = np.array([0.6, 0.7, 0.4]) / np.linalg.norm([0.6, 0.7, 0.4])


def test_the_frequency_of_a_framed_box_matches_finite_differences_in_a_stretch(framed):
    # a mode near 3.0 GHz of the roller box, damped by the frame (Q 1.7), whose depth
    # rides with the nodes: the stretch changes the depth along each axis it moves.
    # One anisotropic stretch, the factors (s_x, s_y, s_z) moved along a direction
    # with all three components. Measured relative slopes of (Re omega, Im omega)
    # per unit of one factor: x -0.43 and -0.71, y -0.58 and -0.27, z 0.010 and
    # -0.021 (the mode does not vary along z), all three together about -1; they
    # add along (0.6, 0.7, 0.4) / 1.005: -0.66 and -0.62 per unit step, so the
    # derivative reads the x and y depths and, through the z component, the z
    # depth (a sign or index error there moves the derivative by under 1 %, the
    # same share the all-axes case had). Re omega scales as s^-1 at most, so the
    # truncation is 4 h^4 = 4e-8 at h = 1e-2, and the round-off 1.5 delta / (h s) of
    # the measured delta of this box (1.6 k DOFs, under 1e-13 as the plate's) is
    # under 1e-9
    problem = gfem.MechanicalEigenmode(
        2.9e9, clamped=mc.ROLLERS, damping_frame=gfem.DampingFrame("frame")
    )
    materials = framed.materials
    nodes = jnp.asarray(framed.mesh.nodes)

    def objective(s):
        model = gfem.Model(framed.mesh.with_nodes(nodes * s), materials=materials)
        return _omega(gfem.solve(model, problem, options=OPTIONS))

    fd.assert_jacobian_matches_by_group(
        objective, jnp.ones(3), [FRAME_DIRECTION], STEP, OMEGA
    )


# --- what raises, and what it costs ---------------------------------------------------


@pytest.fixture(scope="module")
def beam(tmp_path_factory):
    """A 6 x 2 x 2 um beam (W = T) on a Kuhn grid symmetric under y <-> z."""
    mesh = mc.box(tmp_path_factory.mktemp("beam") / "beam.msh", (6.0, 2.0, 2.0), 2)
    return gfem.Model(mesh, materials={"body": mc.material()})


def test_a_degenerate_pair_of_a_3d_body_refuses_by_name(beam):
    # the Kuhn grid is not symmetric under y <-> z: the pair (44.4415 and 44.4520 MHz)
    # is split by 2.4e-4, far under the 1e-3 cluster tolerance given here
    options = gfem.SolverOptions(order=2, cluster_rtol=1e-3)
    problem = gfem.MechanicalEigenmode(4.4e7, n_modes=2, clamped={"x0": "xyz"})
    result = gfem.solve(beam, problem, options=options)
    assert np.sum(np.asarray(result.cluster) == int(result.cluster[0])) == 2

    def objective(rho):
        moved = mc.edited(beam, ["body"], mass_density=rho)
        return jnp.real(gfem.solve(moved, problem, options=options).omega[0])

    # raised in the backward's host callback: jax wraps it
    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match=r"degenerate with another(?s:.*)symmetry of the body",
    ):
        jax.grad(objective)(mc.RHO)


def test_a_group_velocity_cotangent_of_a_3d_body_refuses_by_name(cantilever):
    def objective(rho):
        moved = mc.edited(cantilever, ["a", "b"], mass_density=rho)
        return gfem.solve(moved, CANTILEVER_PROBLEM, options=OPTIONS).group_velocity[0]

    with pytest.raises(gfem.NonDifferentiableError, match="NaN for a 3D body"):
        jax.grad(objective)(mc.RHO)


@pytest.mark.parametrize("argument", ["target", "wavenumber"])
def test_a_traced_target_or_wavenumber_refuses_by_name(argument):
    def build(x):
        kwargs = {"target": 4.3e6, "wavenumber": 0.0, argument: x}
        return gfem.MechanicalEigenmode(**kwargs).target

    with pytest.raises(gfem.NonDifferentiableError, match=f"{argument}.*fixed"):
        jax.grad(build)(4.3e6)


def test_a_field_cotangent_costs_one_factorization_per_mode(cantilever, monkeypatch):
    # the forward of a lossless body at q = 0 factorizes once (the shift); a backward
    # of omega alone adds none, one of a mode's field one per mode
    from gfem._engine import mechanical_eigen

    count = [0]
    real = mechanical_eigen.DirectHandle

    def counted(*args, **kwargs):
        count[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(mechanical_eigen, "DirectHandle", counted)
    model = gfem.Model(
        cantilever.mesh, materials={r: mc.material() for r in cantilever.materials}
    )
    problem = gfem.MechanicalEigenmode(4.3e6, n_modes=2, clamped={"x0": "xyz"})

    def solved(rho):
        return gfem.solve(
            mc.edited(model, ["a", "b"], mass_density=rho), problem, options=OPTIONS
        )

    def outputs(rho):
        result = solved(rho)
        one = result.displacement(0)
        both = [result.displacement(i)["z"].at(POINT) for i in range(2)]
        return (
            jnp.real(result.omega).sum(),
            sum(jnp.abs(one[k].at(POINT)) ** 2 for k in "xyz"),
            sum(jnp.abs(v) ** 2 for v in both),
        )

    # one forward, then three backwards of it, each with a cotangent on one output
    # only; the factorizations are counted between them
    count[0] = 0
    _, pullback = jax.vjp(outputs, jnp.asarray(mc.RHO))
    assert count[0] == 1
    totals = []
    for cotangent in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        count[0] = 0
        pullback(tuple(jnp.asarray(c) for c in cotangent))
        totals.append(count[0])
    # omega alone adds none, the field of one mode one, those of two modes two
    assert totals == [0, 1, 2]
