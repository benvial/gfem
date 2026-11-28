r"""Gradients of the mechanical mode fields by the design inputs.

A scalar of a mode's displacement (or potential) differentiates through the
eigenvector adjoint of the solve: a bordered solve at the converged
eigenvalue, through the 1 J/m normalization and the phase gauge (the largest
displacement coefficient real and positive). Each scalar is checked against
fourth-order central differences at the gate of `finite_differences`.

The scalars read the field at interior points: :math:`|u(p)|^2`, invariant
under the phase gauge, and the real and imaginary parts of a strain
component, which the gauge fixes.

Steps. Each step changes its leaf by :math:`10^{-5}` to :math:`10^{-3}`
relative. The fourth-order difference truncates at about :math:`h^4
|f^{(5)}| / (30 |f'|)`; the mode moves with the leaf through a pencil whose
next mode is some 10 % away in frequency, so each order of derivative grows by
about 10 and the truncation stays near :math:`(10 \cdot 10^{-3})^4 / 30
\approx 3 \cdot 10^{-10}`. The round-off is the fields' noise between samples:
eigenvectors from a fixed ARPACK start, refined to a backward error of
:math:`10^{-12}`, whose error moves smoothly with the design; the measured
errors are below :math:`10^{-9}` but on a reading the leaf barely moves (the
quality factor on :math:`|u|^2`, a derivative :math:`10^{-5}` of the value),
where they reach :math:`7 \cdot 10^{-8}`, still under the gate.

The round-off of a sample of a 2D section is :math:`\delta \approx 10^{-12}`
relative (a 3D body's is :math:`\delta \approx 1.2 \cdot 10^{-10}`, see
`test_mechanical_3d_gradients`); the fields' extra noise above is on top of it.

An output whose derivative is zero (the imaginary part of a lossless mode at
:math:`q = 0`, real to round-off) is read against :math:`10^{-3}` of the
largest derivative of its group, so that round-off on a zero is not a
failure.
"""

import finite_differences as fd
import jax
import jax.numpy as jnp
import mechanical_cases as mc
import numpy as np
import piezo_cases as pc
import pytest

import gfem

#: Interior points of the 1.7 x 1 um rectangle, in m.
POINT = np.array([0.6e-6, 0.4e-6])


def _check(objective, p0, directions, step, groups):
    """``jacobian . d`` equals the fourth-order difference of every output.

    ``groups`` maps a label to the output indices that share a derivative
    floor (see the module docstring).
    """
    fd.assert_jacobian_matches_by_group(objective, p0, directions, step, groups)


# --- an uncoupled rectangle -----------------------------------------------------------


@pytest.fixture(scope="module")
def rectangle(tmp_path_factory):
    """1.7 x 1 um, maxh 0.3: a free solid with a standing wave near 1.2 GHz."""
    path = tmp_path_factory.mktemp("rect") / "rect.msh"
    return mc.rectangle(path, 1.7, 1.0, 0.3)


def _intensity(result):
    r""":math:`|u(p)|^2` of mode 0 at `POINT`, in m^2."""
    u = result.displacement(0)
    return sum(jnp.abs(u[k].at(POINT)) ** 2 for k in "xyz")


#: A wavenumber of the rectangle's modes at q != 0, rad/m, and a quality factor.
Q_RECT, QUALITY = 3e5, 50.0
#: The stiffness of the isotropic solid.
C_BASE = np.asarray(gfem.isotropic_stiffness(mc.E, mc.NU))
#: The Voigt index of the in-plane shear strain, the largest at `POINT`.
XY = 5


def _readings(result):
    r""":math:`|u(p)|^2` and the real and imaginary parts of :math:`S_{xy}(p)`.

    The first is invariant under the phase gauge of the mode, the strain is not.
    The gauge is read at the largest displacement coefficient, which leads the
    next one by about 1 % here: the steps below change the mode by less than
    :math:`10^{-3}`, so the gauge's coefficient never changes between samples.
    """
    u = result.displacement(0)
    strain = gfem.strain(u, POINT[None])[0, XY]
    return jnp.stack([_intensity(result), strain.real, strain.imag])


READINGS = {"|u|^2": [0], "S_xy": [1, 2]}


#: name -> (edit(model, p), direction(mesh), step): a unit of p is 1e10 Pa of the
#: 1.2e11 Pa stiffness (8 %), 100 kg/m^3 of 2500 (4 %) or 5 of the quality factor
#: 50 (10 %), and the per-element directions are ~0.13 per element: the steps of
#: 1e-2 are relative changes of 1e-5 to 1e-3 of the leaf.
LEAVES = {
    "stiffness": (
        lambda m, p: mc.edited(m, ["film"], stiffness=C_BASE + 1e10 * p),
        lambda _: mc.symmetric_direction(5),
        1e-2,
    ),
    "mass_density": (
        lambda m, p: mc.edited(m, ["film"], mass_density=mc.RHO + 100.0 * p),
        lambda mesh: mc.per_element_direction(mesh, "film", 3),
        1e-2,
    ),
    "mechanical_quality_factor": (
        lambda m, p: mc.edited(
            m, ["film"], mechanical_quality_factor=QUALITY + 5.0 * p
        ),
        lambda mesh: mc.per_element_direction(mesh, "film", 4),
        1e-2,
    ),
}


#: The grid of the rectangle's cases: (q, quality factor). Each finite difference
#: takes 8-14 s on one thread (four samples and a jacrev of three backward
#: passes), so the unit tier keeps one case per path of the adjoint (the
#: lossy mode at q != 0, which runs every term, per leaf, and the lossless one
#: at q = 0 for the stiffness) and the slow tier runs the rest of the grid.
GRID = {
    "q0-lossless": (0.0, None),
    "q0-lossy": (0.0, QUALITY),
    "q-lossless": (Q_RECT, None),
    "q-lossy": (Q_RECT, QUALITY),
}
#: The (leaf, case) pairs of the unit tier.
UNIT = {
    ("stiffness", "q-lossy"),
    ("stiffness", "q0-lossless"),
    ("mass_density", "q-lossy"),
    ("mechanical_quality_factor", "q-lossy"),
}


def _leaf_cases():
    for name in LEAVES:
        for case, (q, quality) in GRID.items():
            if name == "mechanical_quality_factor" and quality is None:
                continue  # a lossless section does not read the quality factor
            marks = () if (name, case) in UNIT else (pytest.mark.slow,)
            yield pytest.param(name, q, quality, id=f"{name}-{case}", marks=marks)


@pytest.mark.parametrize(("name", "q", "quality"), list(_leaf_cases()))
def test_a_reading_of_a_mode_field_matches_finite_differences(
    rectangle, name, q, quality
):
    edit, direction, step = LEAVES[name]
    model = mc.model(rectangle, quality=quality)
    problem = gfem.MechanicalEigenmode(1.2e9, wavenumber=q)
    d = direction(rectangle)

    def objective(p):
        return _readings(gfem.solve(edit(model, p), problem))

    _check(objective, jnp.zeros_like(d), [d], step, READINGS)


@pytest.mark.parametrize(
    ("q", "quality"),
    [(0.0, None), (0.0, QUALITY), (Q_RECT, None), (Q_RECT, QUALITY)],
    ids=["q0-lossless", "q0-lossy", "q-lossless", "q-lossy"],
)
def test_the_gauge_of_the_rectangles_mode_has_one_coefficient(rectangle, q, quality):
    # the strain readings above are gauge-dependent: the largest displacement
    # coefficient must lead the next by far more than the 1e-3 the steps move the
    # mode, or the samples would straddle a change of gauge
    result = gfem.solve(
        mc.model(rectangle, quality=quality),
        gfem.MechanicalEigenmode(1.2e9, wavenumber=q),
    )
    u = result.displacement(0)
    size = np.sort(np.abs(np.concatenate([np.asarray(u[k].coefficients) for k in u])))
    assert size[-2] < (1 - 5e-3) * size[-1]


def _stretched(mesh, s, axis):
    """``mesh`` with its coordinate ``axis`` multiplied by ``s``."""
    nodes = jnp.asarray(mesh.nodes)
    factor = jnp.ones(nodes.shape[1]).at[axis].set(s)
    return mesh.with_nodes(nodes * factor)


def _stretch_cases():
    # one case in the unit tier, the rest of the grid in the slow one
    for axis, label in ((0, "x"), (1, "y")):
        for case, (q, quality) in GRID.items():
            unit = (label, case) == ("x", "q-lossy")
            marks = () if unit else (pytest.mark.slow,)
            yield pytest.param(axis, q, quality, id=f"{label}-{case}", marks=marks)


@pytest.mark.parametrize(("axis", "q", "quality"), list(_stretch_cases()))
def test_a_stretch_of_the_nodes_moves_a_reading_as_finite_differences(
    rectangle, axis, q, quality
):
    # the readings stay at a fixed point while the elements move under it: the
    # node pass of the pencil, the normalization's mass and the transport of the
    # reading. A step of 1e-3 on the stretch s changes the mode by about 1e-3
    materials = mc.model(rectangle, quality=quality).materials
    problem = gfem.MechanicalEigenmode(1.2e9, wavenumber=q)

    def objective(s):
        model = gfem.Model(_stretched(rectangle, s, axis), materials=materials)
        return _readings(gfem.solve(model, problem))

    _check(objective, jnp.asarray(1.0), [np.asarray(1.0)], 1e-3, READINGS)


# --- a piezoelectric plate ------------------------------------------------------------

PLATE_CLAMPED = {"left": "x", "right": "x"}
PLATE_OPTIONS = gfem.SolverOptions(order=3)
Q_PLATE = 3e6
#: An interior point of the 2 x 1 um plate, in m.
PLATE_POINT = np.array([0.74e-6, 0.31e-6])


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = tmp_path_factory.mktemp("te") / "te.msh"
    mesh = mc.piezo_plate_mesh(path, pc.PLATE_WIDTH, pc.PLATE_THICKNESS, 0.5)
    return mc.piezo_plate_model(mesh, "TE", quality=100.0)


def _plate_edit(name, model, p):
    c, e, eps = pc.hexagonal("TE")
    if name == "piezoelectric_tensor":
        return mc.edited(model, ["pz"], piezoelectric_tensor=e + 0.1 * p)
    if name == "eps_r":
        return mc.edited(model, ["pz"], eps_r=jnp.asarray(eps) + p)
    if name == "mass_density":
        rho = model.materials["pz"].mass_density
        return mc.edited(model, ["pz"], mass_density=rho + 100.0 * p)
    return mc.edited(model, ["pz"], stiffness=c + 1e10 * p)


PLATE_SHAPES = {
    "piezoelectric_tensor": (3, 6),
    "eps_r": (3,),
    "stiffness": (6, 6),
    "mass_density": None,
}


def _plate_cases():
    # one case in the unit tier (the coupling tensor at q != 0), the rest slow
    for name in PLATE_SHAPES:
        for q in (0.0, Q_PLATE):
            unit = (name, q) == ("piezoelectric_tensor", Q_PLATE)
            marks = () if unit else (pytest.mark.slow,)
            yield pytest.param(name, q, id=f"{name}-{q:g}", marks=marks)


def _plate_readings(result):
    r""":math:`|u(p)|^2`, :math:`|V(p)|^2` and :math:`V(p)\,\overline{u_y(p)}`.

    All three are invariant under the phase gauge: the plate's mode is uniform
    along x, so its largest displacement coefficients tie and the gauge is not
    defined. The product reads the phase of the potential against the
    displacement's.
    """
    u = result.displacement(0)
    v = result.potential(0).at(PLATE_POINT)
    cross = v * jnp.conj(u["y"].at(PLATE_POINT))
    return jnp.stack(
        [
            sum(jnp.abs(u[k].at(PLATE_POINT)) ** 2 for k in "xyz"),
            jnp.abs(v) ** 2,
            cross.real,
            cross.imag,
        ]
    )


@pytest.mark.parametrize(("name", "q"), list(_plate_cases()))
def test_a_reading_of_a_coupled_mode_matches_finite_differences(plate, name, q):
    # the open plate resonates near the stiffened half wave of its thickness; a
    # unit of p is 0.1 C/m^2 of the piezoelectric tensor, 1 of eps_r, 1e10 Pa of
    # the stiffness or 100 kg/m^3 of the density (per element), so a step of
    # 1e-2 is a relative change of 1e-4 to 1e-3
    target = float(np.real(pc.plate_form("TE")["v"])) / (2 * pc.PLATE_THICKNESS * mc.UM)
    problem = gfem.MechanicalEigenmode(target, wavenumber=q, clamped=PLATE_CLAMPED)
    if name == "stiffness":
        d = mc.symmetric_direction(7)
    elif name == "mass_density":
        d = mc.per_element_direction(plate.mesh, "pz", 7)
    else:
        d = fd.unit_directions(np.random.default_rng(7), PLATE_SHAPES[name], 1)[0]

    def objective(p):
        return _plate_readings(
            gfem.solve(_plate_edit(name, plate, p), problem, options=PLATE_OPTIONS)
        )

    _check(
        objective,
        jnp.zeros(d.shape),
        [d],
        1e-2,
        {"|u|^2": [0], "|V|^2": [1], "V conj(u_y)": [2, 3]},
    )


# --- with omega, and what one cotangent costs -----------------------------------------


def test_a_scalar_of_omega_and_the_field_matches_finite_differences(rectangle):
    # one cotangent on omega and on the field of the same mode, lossy at q != 0
    # (every term of the adjoint); the step as in the stiffness case above
    model = mc.model(rectangle, quality=QUALITY)
    problem = gfem.MechanicalEigenmode(1.2e9, wavenumber=Q_RECT)
    d = mc.symmetric_direction(11)

    def objective(p):
        result = gfem.solve(
            mc.edited(model, ["film"], stiffness=C_BASE + 1e10 * p), problem
        )
        reading = _readings(result)
        omega = result.omega[0] / 7e9
        return jnp.stack(
            [reading[0] * 1e11 * omega.real + reading[1] * omega.imag * 1e2]
        )

    _check(objective, jnp.zeros_like(d), [d], 1e-2, {"f": [0]})


def test_a_field_cotangent_costs_one_factorization_per_mode(rectangle, monkeypatch):
    # the forward of a lossless section at q = 0 factorizes once (the shift); a
    # backward of omega alone adds none, one of a mode's field one per mode
    from gfem._engine import mechanical_eigen

    count = [0]
    real = mechanical_eigen.DirectHandle

    def counted(*args, **kwargs):
        count[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(mechanical_eigen, "DirectHandle", counted)
    model = mc.model(rectangle)
    problem = gfem.MechanicalEigenmode(1.2e9, n_modes=2)

    def factorizations(objective):
        count[0] = 0
        fd._alone(jax.grad(objective), mc.RHO)
        return count[0]

    def solved(rho):
        return gfem.solve(mc.edited(model, ["film"], mass_density=rho), problem)

    def omega(rho):
        return jnp.real(solved(rho).omega).sum()

    def one_field(rho):
        return _intensity(solved(rho))

    def both_fields(rho):
        result = solved(rho)
        u = [result.displacement(i)["x"].at(POINT) for i in range(2)]
        return sum(jnp.abs(v) ** 2 for v in u)

    assert factorizations(omega) == 1
    assert factorizations(one_field) == 2
    assert factorizations(both_fields) == 3


def test_a_field_of_a_degenerate_mode_refuses_by_name(tmp_path):
    # the anti-plane waves (1, 0) and (0, 1) of a square share one frequency; the
    # mesh splits them by far less than the 1e-2 cluster tolerance given here
    mesh = mc.rectangle(tmp_path / "square.msh", 1.0, 1.0, 0.25)
    model = mc.model(mesh)
    options = gfem.SolverOptions(cluster_rtol=1e-2)
    problem = gfem.MechanicalEigenmode(mc.sh_standing(1, 0, 1.0, 1.0), n_modes=2)

    def objective(rho):
        moved = mc.edited(model, ["film"], mass_density=rho)
        u = gfem.solve(moved, problem, options=options).displacement(0)
        return jnp.abs(u["z"].at(np.array([0.3e-6, 0.2e-6]))) ** 2

    # raised in the backward's host callback: jax wraps it
    with pytest.raises(jax.errors.JaxRuntimeError, match="degenerate with another"):
        jax.grad(objective)(mc.RHO)
