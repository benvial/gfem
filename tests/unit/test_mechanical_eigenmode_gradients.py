r"""Gradients of the mechanical eigenfrequencies by the design inputs.

:math:`\omega` is differentiated through the left and right eigenvectors of the
solve, with no further solve: the stiffness, piezoelectric tensor, permittivity,
density and quality factor, the damping frame, and the node coordinates, at
:math:`q = 0` and :math:`q \ne 0`, lossless (left vector
:math:`\bar x`), lossy at :math:`q = 0` (:math:`x`) and lossy at :math:`q \ne 0`
(inverse iteration), coupled and uncoupled. Each is checked against
fourth-order central differences of :math:`(\mathrm{Re}\,\omega,
\mathrm{Im}\,\omega)` at the gate of `finite_differences`.

The differenced objective is the eigenfrequency, a two-sided Rayleigh quotient of
vectors converged to the ARPACK tolerance of :math:`10^{-12}`: round-off
:math:`\delta \approx 10^{-12}` relative. A relative perturbation :math:`h` of a
leaf that scales :math:`\omega` as :math:`(1 + p)^{\pm 1/2}` or :math:`(1 + p)^{-1}`
(the density, the modulus, the stretch) has a fourth-order truncation of at most
:math:`4 h^4` of the derivative and a round-off of :math:`1.5\,\delta / (h\,s)`, with
:math:`s` the relative slope: for :math:`h = 10^{-2}` and :math:`s \sim 1` both sit
under :math:`10^{-7}`. A leaf that moves :math:`\omega` less (the quality factor on
:math:`\mathrm{Re}\,\omega`, a tensor entry, the loss of a lossless section) keeps
the step but reads an error against the larger of its derivative and
:math:`10^{-3}` of the biggest one of the two outputs, so that round-off on a
derivative that is zero (or a thousand times below the other) is not a failure.
"""

import dataclasses

import finite_differences as fd
import jax
import jax.numpy as jnp
import mechanical_cases as mc
import numpy as np
import piezo_cases as pc
import pytest

import gfem

#: The sliding walls of a strip: the faces of constant x keep u_y = u_z = 0.
SLIDING = {"left": "yz", "right": "yz"}
#: A wavenumber of the rectangle's lossy modes, and of the plate's, rad/m.
Q_RECT, Q_PLATE, Q_SANDWICH = 3e5, 3e6, 3e6
#: The quality factor of a lossy section.
QUALITY = 50.0
#: The stiffness of the isotropic solid.
C_BASE = np.asarray(gfem.isotropic_stiffness(mc.E, mc.NU))


def _fields(model):
    """``model`` with every material a `gfem.MaterialField`."""
    return {
        region: gfem.MaterialField.uniform(region, material)
        for region, material in model.materials.items()
    }


def _edited(model, regions, **values):
    """``model`` with the leaves ``values`` of ``regions`` replaced."""
    fields = _fields(model)
    for region in regions:
        fields[region] = dataclasses.replace(fields[region], **values)
    return model.replace(materials=fields)


def _omega(model, problem, options=None):
    """``[Re omega, Im omega]`` of the nearest mode."""
    omega = gfem.solve(model, problem, options=options).omega[0]
    return jnp.stack([omega.real, omega.imag])


def _check(objective, p0, directions, step, labels=("Re omega", "Im omega")):
    """``jacobian . d`` equals the fourth-order difference of every output."""
    jacobian = np.asarray(fd._alone(jax.jacrev(objective), p0))
    for d in directions:
        samples = np.array(
            [
                np.asarray(fd._alone(objective, p0 + k * step * d))
                for k in (-2, -1, 1, 2)
            ]
        )
        difference = (8 * (samples[2] - samples[1]) - (samples[3] - samples[0])) / (
            12 * step
        )
        derivative = jacobian.reshape(len(labels), -1) @ np.reshape(d, -1)
        for i, label in enumerate(labels):
            scale = max(abs(difference[i]), 1e-3 * abs(derivative).max())
            error = abs(derivative[i] - difference[i]) / scale
            assert error < fd.RTOL, (
                f"{label}: gradient {derivative[i]:.12e} vs difference "
                f"{difference[i]:.12e} (relative error {error:.2e})"
            )


def _directions(seed, shape, n=1, *, symmetric=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = rng.standard_normal(shape)
        d = d + d.T if symmetric else d
        out.append(d / np.linalg.norm(d))
    return out


# --- an uncoupled rectangle -----------------------------------------------------------


@pytest.fixture(scope="module")
def rectangle(tmp_path_factory):
    """1.7 x 1 um, maxh 0.3: a free solid with a standing wave near 1.2 GHz."""
    path = tmp_path_factory.mktemp("rect") / "rect.msh"
    return mc.rectangle(path, 1.7, 1.0, 0.3)


#: name -> (edit(model, p), shape of p, step): a unit of p is 1e10 Pa of the 1.2e11 Pa
#: stiffness (8 %), 100 kg/m^3 of 2500 (4 %) or 5 of the quality factor 50 (10 %), so
#: the step of 1e-2 is a relative change of 4e-4 to 1e-3 of the leaf.
LEAVES = {
    "stiffness": (
        lambda m, p: _edited(m, ["film"], stiffness=C_BASE + 1e10 * p),
        (6, 6),
        1e-2,
    ),
    "mass_density": (
        lambda m, p: _edited(m, ["film"], mass_density=mc.RHO + 100.0 * p),
        (),
        1e-2,
    ),
    "mechanical_quality_factor": (
        lambda m, p: _edited(m, ["film"], mechanical_quality_factor=QUALITY + 5.0 * p),
        (),
        1e-2,
    ),
}


@pytest.mark.parametrize("name", LEAVES)
@pytest.mark.parametrize(
    ("q", "quality"),
    [(0.0, None), (0.0, QUALITY), (Q_RECT, None), (Q_RECT, QUALITY)],
    ids=["q0-lossless", "q0-lossy", "q-lossless", "q-lossy"],
)
def test_a_material_gradient_matches_finite_differences(rectangle, name, q, quality):
    if name == "mechanical_quality_factor" and quality is None:
        pytest.skip("a lossless section does not read the quality factor")
    edit, shape, step = LEAVES[name]
    model = mc.model(rectangle, quality=quality)
    problem = gfem.MechanicalEigenmode(1.2e9, wavenumber=q)

    def objective(p):
        return _omega(edit(model, p), problem)

    _check(
        objective,
        jnp.zeros(shape),
        _directions(5, shape, symmetric=name == "stiffness"),
        step,
    )


def _stretched(mesh, s, axis):
    """``mesh`` with its coordinate ``axis`` multiplied by ``s``."""
    nodes = jnp.asarray(mesh.nodes)
    factor = jnp.ones(nodes.shape[1]).at[axis].set(s)
    return mesh.with_nodes(nodes * factor)


@pytest.mark.parametrize("axis", [0, 1], ids=["x", "y"])
@pytest.mark.parametrize(
    ("q", "quality"),
    [(0.0, None), (Q_RECT, QUALITY)],
    ids=["q0-lossless", "q-lossy"],
)
def test_a_stretch_of_the_nodes_matches_finite_differences(rectangle, axis, q, quality):
    # omega scales as 1/s along the stretched axis: a step of 1e-2 on s truncates
    # at about 4 h^4 = 4e-8 relative (see the module docstring)
    materials = mc.model(rectangle, quality=quality).materials
    problem = gfem.MechanicalEigenmode(1.2e9, wavenumber=q)

    def objective(s):
        model = gfem.Model(_stretched(rectangle, s, axis), materials=materials)
        return _omega(model, problem)

    _check(objective, jnp.asarray(1.0), [np.asarray(1.0)], 1e-2)


# --- a damping frame ------------------------------------------------------------------


@pytest.fixture(scope="module")
def sandwich(tmp_path_factory):
    return mc.sandwich_model(tmp_path_factory.mktemp("sandwich") / "sandwich.msh")


def _frame_problem(q):
    return gfem.MechanicalEigenmode(
        mc.sh_strip(1, Q_SANDWICH, 1.0),
        wavenumber=q,
        clamped={**SLIDING, "bottom": "xyz", "top": "xyz"},
        damping_frame=gfem.DampingFrame(["clad_lo", "clad_hi"], loss_factor=0.5),
    )


@pytest.mark.parametrize("q", [0.0, Q_SANDWICH])
@pytest.mark.parametrize("leaf", ["core stiffness", "frame stiffness", "stretch"])
def test_a_damping_frame_mode_matches_finite_differences(sandwich, q, leaf):
    # the frame's loss reads the stiffness of its regions (the eta r^2 moment) and
    # the node coordinates (its depth): the loss path of the pair and of the nodes
    problem = _frame_problem(q)
    mesh = sandwich.mesh

    def objective(p):
        if leaf == "stretch":
            model = gfem.Model(_stretched(mesh, p, 1), materials=sandwich.materials)
        else:
            regions = ["core"] if leaf == "core stiffness" else ["clad_lo", "clad_hi"]
            model = _edited(
                sandwich,
                regions,
                stiffness=gfem.isotropic_stiffness(
                    mc.E * (1 + 4 * (leaf == "frame stiffness")) * (1 + 1e-1 * p), mc.NU
                ),
            )
        return _omega(model, problem)

    p0 = jnp.asarray(1.0 if leaf == "stretch" else 0.0)
    _check(objective, p0, [np.asarray(1.0)], 1e-2)


# --- a piezoelectric plate ------------------------------------------------------------

PLATE_CLAMPED = {"left": "x", "right": "x"}
PLATE_OPTIONS = gfem.SolverOptions(order=3)


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    path = tmp_path_factory.mktemp("te") / "te.msh"
    mesh = mc.piezo_plate_mesh(path, pc.PLATE_WIDTH, pc.PLATE_THICKNESS, 0.5)
    return mc.piezo_plate_model(mesh, "TE", quality=100.0)


def _plate_edit(name, model, p):
    c, e, eps = pc.hexagonal("TE")
    if name == "piezoelectric_tensor":
        return _edited(model, ["pz"], piezoelectric_tensor=e + 0.1 * p)
    if name == "eps_r":
        return _edited(model, ["pz"], eps_r=jnp.asarray(eps) + p)
    if name == "stiffness":
        return _edited(model, ["pz"], stiffness=c + 1e10 * p)
    return _edited(model, ["pz"], mass_density=pc.HEX["rho"] + 100.0 * p)


PLATE_SHAPES = {
    "piezoelectric_tensor": (3, 6),
    "eps_r": (3,),
    "stiffness": (6, 6),
    "mass_density": (),
}


@pytest.mark.parametrize("name", PLATE_SHAPES)
@pytest.mark.parametrize("q", [0.0, Q_PLATE])
def test_a_coupled_plate_gradient_matches_finite_differences(plate, name, q):
    # the open plate resonates near the stiffened half wave of its thickness
    target = float(np.real(pc.plate_form("TE")["v"])) / (2 * pc.PLATE_THICKNESS * mc.UM)
    problem = gfem.MechanicalEigenmode(target, wavenumber=q, clamped=PLATE_CLAMPED)
    shape = PLATE_SHAPES[name]

    def objective(p):
        return _omega(_plate_edit(name, plate, p), problem, PLATE_OPTIONS)

    _check(
        objective,
        jnp.zeros(shape),
        _directions(7, shape, symmetric=name == "stiffness"),
        1e-2,
    )


@pytest.mark.parametrize("axis", [0, 1], ids=["x", "y"])
def test_a_stretch_of_a_coupled_plate_matches_finite_differences(plate, axis):
    # the node pass of the coupling and dielectric pieces at q > 0. The step is 1e-3
    # on the stretch s: a thickness stretch of 1e-2 moves the plate between its
    # nearest modes (the samples jump by 3 %), 1e-3 does not. The fourth-order
    # difference truncates at about 4 h^4 = 4e-12 relative and rounds at
    # eps / h = 2e-13 (omega is smooth to the 1e-13 backward error of the solve).
    target = float(np.real(pc.plate_form("TE")["v"])) / (2 * pc.PLATE_THICKNESS * mc.UM)
    problem = gfem.MechanicalEigenmode(
        target, wavenumber=Q_PLATE, clamped=PLATE_CLAMPED
    )

    def objective(s):
        model = gfem.Model(_stretched(plate.mesh, s, axis), materials=plate.materials)
        return _omega(model, problem, PLATE_OPTIONS)

    _check(objective, jnp.asarray(1.0), [np.asarray(1.0)], 1e-3)


# --- what raises ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "match"),
    [
        ("group_velocity", "group_velocity.*read from the modes"),
        ("error", "error.*backward error"),
        ("displacement", "displacement.*bordered eigenvector adjoint"),
        ("potential", "potential.*bordered eigenvector adjoint"),
    ],
)
def test_an_output_the_backward_does_not_have_refuses_by_name(plate, output, match):
    problem = gfem.MechanicalEigenmode(2.2e9, wavenumber=Q_PLATE, clamped=PLATE_CLAMPED)

    def objective(rho):
        model = _edited(plate, ["pz"], mass_density=rho)
        result = gfem.solve(model, problem, options=PLATE_OPTIONS)
        if output == "displacement":
            return jnp.real(result.displacement(0)["z"].coefficients[0])
        if output == "potential":
            return jnp.real(result.potential_fields[0].coefficients[0])
        return getattr(result, output)[0]

    with pytest.raises(gfem.NonDifferentiableError, match=match):
        jax.grad(objective)(pc.HEX["rho"])


def test_a_degenerate_mode_refuses_by_name(tmp_path):
    # the anti-plane waves (1, 0) and (0, 1) of a square share one frequency; the
    # mesh splits them by far less than the 1e-2 cluster tolerance given here
    mesh = mc.rectangle(tmp_path / "square.msh", 1.0, 1.0, 0.25)
    model = mc.model(mesh)
    options = gfem.SolverOptions(cluster_rtol=1e-2)
    problem = gfem.MechanicalEigenmode(mc.sh_standing(1, 0, 1.0, 1.0), n_modes=2)
    result = gfem.solve(model, problem, options=options)
    assert np.sum(np.asarray(result.cluster) == int(result.cluster[0])) == 2

    def objective(rho):
        moved = _edited(model, ["film"], mass_density=rho)
        return jnp.real(gfem.solve(moved, problem, options=options).omega[0])

    # raised in the backward's host callback: jax wraps it
    with pytest.raises(jax.errors.JaxRuntimeError, match="degenerate with another"):
        jax.grad(objective)(mc.RHO)


def test_a_left_vector_above_the_tolerance_raises_by_mode(rectangle, monkeypatch):
    # no left iteration at all leaves the start vector conj(x), whose backward
    # error is far above the 1e-10 tolerance of a lossy section at q
    monkeypatch.setattr("gfem._engine.mechanical_eigen._LEFT_STEPS", 0)
    model = mc.model(rectangle, quality=QUALITY)
    problem = gfem.MechanicalEigenmode(1.2e9, wavenumber=Q_RECT)
    with pytest.raises(gfem.EigenmodeConvergenceError, match="left vector of mode 0"):
        gfem.solve(model, problem)


# --- the strain of a mode ---------------------------------------------------------


def test_a_field_map_keeps_its_wavenumber_through_the_pytree(rectangle):
    model = mc.model(rectangle)
    result = gfem.solve(model, gfem.MechanicalEigenmode(1.2e9, wavenumber=Q_RECT))
    displacement = result.displacement(0)
    assert displacement.wavenumber == Q_RECT
    assert f"wavenumber={Q_RECT:g}" in repr(displacement)
    leaves, tree = jax.tree.flatten(displacement)
    again = jax.tree.unflatten(tree, leaves)
    assert again.wavenumber == Q_RECT
    assert tuple(again) == ("x", "y", "z")
    # a field that does not vary along z says nothing of it
    flat = gfem.solve(model, gfem.MechanicalEigenmode(1.2e9)).displacement(0)
    assert flat.wavenumber == 0.0
    assert "wavenumber" not in repr(flat)


def test_the_job_codec_keeps_the_wavenumber_of_the_displacement(rectangle, tmp_path):
    model = mc.model(rectangle)
    problem = gfem.MechanicalEigenmode(1.2e9, wavenumber=Q_RECT)
    job = gfem.Job.from_json(gfem.Job(model, problem).to_json(tmp_path / "job.json"))
    result = job.run(out=tmp_path / "out")
    assert result.displacement(0).wavenumber == Q_RECT
    # and from the files: the displacement is still a FieldMap at its wavenumber,
    # so gfem.strain reads it
    back = gfem.load_result(tmp_path / "out")
    loaded = back.displacement(0)
    assert type(loaded).__name__ == "FieldMap"
    assert loaded.wavenumber == Q_RECT
    points = np.array([[0.6e-6, 0.4e-6]])
    assert np.asarray(gfem.strain(loaded, points)) == pytest.approx(
        np.asarray(gfem.strain(result.displacement(0), points)), rel=1e-12, abs=1e-30
    )


def test_the_strain_of_a_mode_adds_the_axial_derivative(rectangle):
    # the strain at points against central differences of the fields themselves, and
    # -iq u in zz, yz and xz. The fields are quadratic inside an element, so a
    # central difference of step d is exact up to round-off 1e-16 |u| / d; with
    # d = 1e-8 m (10 nm of a 0.3 um mesh) that is 1e-8 relative to |u|/d.
    model = mc.model(rectangle)
    result = gfem.solve(model, gfem.MechanicalEigenmode(1.2e9, wavenumber=Q_RECT))
    u = result.displacement(0)
    points = np.array([[0.6e-6, 0.4e-6], [1.1e-6, 0.7e-6], [0.3e-6, 0.2e-6]])
    got = np.asarray(gfem.strain(u, points))
    d = 1e-8

    def derivative(key, axis):
        step = np.zeros(2)
        step[axis] = d
        return (
            np.asarray(u[key].at(points + step)) - np.asarray(u[key].at(points - step))
        ) / (2 * d)

    values = {k: np.asarray(u[k].at(points)) for k in "xyz"}
    expected = np.stack(
        [
            derivative("x", 0),
            derivative("y", 1),
            -1j * Q_RECT * values["z"],
            derivative("z", 1) - 1j * Q_RECT * values["y"],
            derivative("z", 0) - 1j * Q_RECT * values["x"],
            derivative("x", 1) + derivative("y", 0),
        ],
        axis=-1,
    )
    scale = np.abs(expected).max()
    assert np.abs(got - expected).max() < 1e-6 * scale
