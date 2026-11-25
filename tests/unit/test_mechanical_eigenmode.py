"""The mechanical eigenmodes of a 2D section: closed forms, identities, refusals."""

import mechanical_cases as mc
import numpy as np
import pytest
import thermal_cases as tc

import gfem

#: The sliding walls of a strip: the faces of constant x keep u_y = u_z = 0.
SLIDING = {"left": "yz", "right": "yz"}
#: A plate in plane strain along (y, z): the faces of constant x keep u_x = 0.
PLANE = {"left": "x", "right": "x"}


@pytest.fixture(scope="module")
def free_rectangle(tmp_path_factory):
    """1.7 x 1 um, maxh 0.3: the standing waves, and the strip with sliding walls."""
    path = tmp_path_factory.mktemp("rect") / "rect.msh"
    return mc.model(mc.rectangle(path, 1.7, 1.0, 0.3))


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    """0.5 x 1 um, maxh 0.25: the Lamb modes of a 1 um plate."""
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    return mc.model(mc.rectangle(path, 0.5, 1.0, 0.25))


@pytest.fixture(scope="module")
def half_space(tmp_path_factory):
    """0.3 x 4 um, maxh 0.2: four wavelengths deep at 1 um."""
    path = tmp_path_factory.mktemp("deep") / "deep.msh"
    return mc.model(mc.rectangle(path, 0.3, 4.0, 0.2))


def _relative(found, exact):
    return abs(found - exact) / exact


# The closed-form tolerances come from one measurement on two meshes (maxh and
# maxh / 2, P2 elements): the eigenfrequency error falls like h^3.6 to h^5, and
# each gate is at least twice the coarse mesh's error, the number in its comment.


@pytest.mark.parametrize(("n", "m"), [(1, 0), (0, 1), (1, 1)])
def test_the_standing_shear_waves_of_a_free_rectangle(free_rectangle, n, m):
    # coarse errors 3.3e-5, 2.8e-4 and 4.5e-4 (fine: 2.5e-6, 2.3e-5, 3.7e-5)
    exact = mc.sh_standing(n, m, 1.7, 1.0)
    result = mc.solve(free_rectangle, exact)
    assert _relative(float(result.frequency[0]), exact) < 1e-3
    u = result.displacement(0)
    peak = {a: float(np.abs(np.asarray(u[a].coefficients)).max()) for a in "xyz"}
    # anti-plane motion decouples exactly: the in-plane parts are round-off
    assert max(peak["x"], peak["y"]) < 1e-12 * peak["z"]
    assert result.n_modes == 1
    assert result.wavenumber == 0.0


@pytest.mark.parametrize(("m", "tolerance"), [(0, 1e-9), (1, 1e-3)])
def test_the_shear_horizontal_modes_of_a_strip_at_q(free_rectangle, m, tolerance):
    # SH0 is exact in the discrete space (u_x uniform); SH1's coarse error is
    # 2.3e-4 at q = 2e6 rad/m (1.9e-5 on the fine mesh)
    q = 2e6
    exact = mc.sh_strip(m, q, 1.0)
    result = mc.solve(free_rectangle, exact, wavenumber=q, clamped=SLIDING)
    assert _relative(float(result.frequency[0]), exact) < tolerance
    # the group velocity of SH0 is v_T, SH1's is v_T^2 q / omega
    omega = 2 * np.pi * exact
    assert float(result.group_velocity[0]) == pytest.approx(
        mc.V_T**2 * q / omega, rel=max(tolerance, 1e-9)
    )


@pytest.mark.parametrize("kind", ["S", "A"])
def test_the_lamb_modes_of_a_plate(plate, kind):
    # coarse errors at q = 2e6 rad/m (maxh 0.25): S0 2.2e-5, A0 2.3e-4; the fine
    # mesh's (maxh 0.125) are 7e-7 and 7.4e-6
    q = 2e6
    (exact,) = mc.lamb_roots(q, 1.0, kind)
    result = mc.solve(plate, exact, wavenumber=q, clamped=PLANE)
    assert _relative(float(result.frequency[0]), exact) < 1e-3


def test_the_rayleigh_wave(half_space):
    # coarse error 8.9e-4 at maxh 0.2 (3.1e-5 at 0.1); the wave decays by
    # exp(-2 q D sqrt(1 - xi^2)) ~ 1e-30 over four wavelengths, and its group
    # velocity is v_R (no dispersion)
    q = 2 * np.pi / (1.0 * mc.UM)
    v_r = mc.rayleigh_ratio() * mc.V_T
    clamped = {**PLANE, "bottom": "xyz"}
    result = mc.solve(half_space, v_r * q / (2 * np.pi), wavenumber=q, clamped=clamped)
    assert _relative(float(result.frequency[0]), v_r * q / (2 * np.pi)) < 2e-3
    assert float(result.group_velocity[0]) == pytest.approx(v_r, rel=2e-2)


# --- identities ------------------------------------------------------------------


def test_a_uniform_quality_factor_scales_the_frequency_exactly(free_rectangle):
    # the stiffness is (1 + i/Q) K on the same mesh, so omega scales by
    # sqrt(1 + i/Q); both solves are held to the eigen tolerance (1e-10
    # backward error, a normal pencil: the eigenvalue is as accurate)
    rect = free_rectangle
    target = mc.sh_standing(1, 1, 1.7, 1.0)
    quality = 40.0
    lossless = mc.solve(rect, target, wavenumber=1e6, clamped=SLIDING)
    lossy = mc.solve(
        mc.model(rect.mesh, quality=quality), target, wavenumber=1e6, clamped=SLIDING
    )
    ratio = np.asarray(lossy.omega) / np.asarray(lossless.omega)
    assert abs(ratio[0] - np.sqrt(1 + 1j / quality)) < 1e-9
    # a lossless mode is real
    assert np.abs(np.imag(lossless.omega)).max() <= 1e-10 * abs(lossless.omega[0])
    assert float(lossy.q[0]) == pytest.approx(
        float(np.real(lossy.omega[0]) / (2 * np.imag(lossy.omega[0])))
    )


def test_an_isotropic_section_has_the_same_frequency_at_minus_q(free_rectangle):
    target = mc.sh_strip(1, 3e6, 1.0)
    up = mc.solve(free_rectangle, target, 2, wavenumber=3e6, clamped=SLIDING)
    down = mc.solve(free_rectangle, target, 2, wavenumber=-3e6, clamped=SLIDING)
    assert np.abs(np.asarray(up.omega) - np.asarray(down.omega)).max() < 1e-9 * float(
        np.abs(up.omega[0])
    )
    # and the opposite group velocity
    assert np.asarray(down.group_velocity) == pytest.approx(
        -np.asarray(up.group_velocity), rel=1e-6
    )


def test_a_mode_is_normalized_to_one_joule_per_metre_with_a_real_largest_coefficient(
    free_rectangle,
):
    result = mc.solve(
        free_rectangle, mc.sh_strip(1, 2e6, 1.0), 2, wavenumber=2e6, clamped=SLIDING
    )
    for mode in range(2):
        u = result.displacement(mode)
        # the kinetic energy per metre: the mass form of the displacement DOFs
        coefficients = np.concatenate([np.asarray(u[a].coefficients) for a in "xyz"])
        largest = coefficients[np.argmax(np.abs(coefficients))]
        # real to round-off (a symmetric mode has several equal peaks)
        assert abs(largest.imag) <= 1e-14 * abs(largest)
        assert largest.real > 0
        omega = float(np.real(result.omega[mode]))
        energy = 0.5 * omega**2 * _mass_form(free_rectangle, u)
        # the eigen tolerance 1e-10 on a vector scaled in floating point
        assert energy == pytest.approx(1.0, rel=1e-10)


def _mass_form(model, u):
    """:math:`\\int \\rho |u|^2 dA` of a displacement `FieldMap`, in SI."""
    import ngsolve as ngs

    from gfem._engine.cache import discretization
    from gfem._engine.domain import em_region, h1_space

    built = discretization(model.mesh).built
    space = h1_space(model, built, 2, (), physics="mechanical", complex_valued=True)
    total = 0.0
    with ngs.TaskManager():
        for axis in "xyz":
            g = ngs.GridFunction(space)
            g.vec.FV().NumPy()[:] = np.asarray(u[axis].coefficients)
            region = em_region(model, built, "mechanical")
            total += ngs.Integrate(
                mc.RHO * (g * ngs.Conj(g)).real * built.scale**2,
                built.mesh,
                definedon=region,
                order=6,
            )
    return total


def test_the_group_velocity_matches_a_central_difference_in_q(free_rectangle):
    # a central difference with step d = 1e-2 q has the truncation error
    # (d/q)^2 / 6 * q^2 |w'''| / w', at most (d/q)^2 / 8 = 1.3e-5 for
    # w = v sqrt(q^2 + k^2); the solves agree to 1e-10, so rounding adds
    # about 1e-10 / 1e-2 = 1e-8
    q = 2e6
    d = 1e-2 * q
    target = mc.sh_strip(1, q, 1.0)
    f = {
        s: float(
            mc.solve(free_rectangle, target, wavenumber=q + s * d, clamped=SLIDING)
            .omega[0]
            .real
        )
        for s in (-1, 1)
    }
    exact = mc.solve(free_rectangle, target, wavenumber=q, clamped=SLIDING)
    assert float(exact.group_velocity[0]) == pytest.approx(
        (f[1] - f[-1]) / (2 * d), rel=5e-5
    )


# --- a damping frame -------------------------------------------------------------


@pytest.fixture(scope="module")
def sandwich(tmp_path_factory):
    """A 1 um soft core between two 1.5 um stiffer claddings (4x the modulus).

    Shear horizontal modes at q = 3e6 rad/m are confined between the core's and the
    cladding's shear velocity (v_T q < omega < 2 v_T q) and leak above it.
    """
    w, core, clad = 0.4, 1.0, 1.5
    rects = [
        (0, -clad, w, 0, "clad_lo", None),
        (0, 0, w, core, "core", None),
        (0, core, w, core + clad, "clad_hi", None),
    ]
    edges = [
        (lambda p: abs(p[0]) < 1e-9, "left"),
        (lambda p: abs(p[0] - w) < 1e-9, "right"),
        (lambda p: abs(p[1] + clad) < 1e-9, "bottom"),
        (lambda p: abs(p[1] - core - clad) < 1e-9, "top"),
    ]
    nodes, tris, lines = tc._mesh2d(rects, 0.25, edges)
    path = tmp_path_factory.mktemp("sandwich") / "sandwich.msh"
    tc._write_section(path, nodes, tris, lines)
    stiff = gfem.Material(
        stiffness=gfem.isotropic_stiffness(4 * mc.E, mc.NU), mass_density=mc.RHO
    )
    materials = {"core": mc.material(), "clad_lo": stiff, "clad_hi": stiff}
    return gfem.Model(gfem.Mesh.from_gmsh(path, length_unit="um"), materials=materials)


def _sandwich(model, loss_factor=None):
    frame = (
        None
        if loss_factor is None
        else gfem.DampingFrame(["clad_lo", "clad_hi"], loss_factor=loss_factor)
    )
    return mc.solve(
        model,
        mc.sh_strip(1, 3e6, 1.0),
        3,
        wavenumber=3e6,
        clamped={**SLIDING, "bottom": "xyz", "top": "xyz"},
        damping_frame=frame,
    )


def test_a_damping_frame_lowers_the_q_of_a_leaky_mode_and_spares_a_confined_one(
    sandwich,
):
    # measured (Q): eta 1.2 gives 114, 22 and 7.8 for the modes at 2.11, 2.98 and
    # 3.58 GHz (the first confined, the last above the cladding's cutoff
    # 2 v_T q / 2 pi = 3.2 GHz), eta 2.4 gives 65, 15 and 5.7; the gates keep a
    # factor of two or more
    bare = _sandwich(sandwich)
    assert np.abs(np.imag(bare.omega)).max() <= 1e-10 * float(np.abs(bare.omega[0]))
    framed = _sandwich(sandwich, 1.2)
    stronger = _sandwich(sandwich, 2.4)
    q1, q2 = np.asarray(framed.q), np.asarray(stronger.q)
    assert q1[0] > 50
    assert q1[2] < 15
    assert (np.diff(q1) < 0).all()
    assert (q2 < q1).all()
    # the lossy left vector at q != 0 gives the group velocity of the complex band
    assert np.isfinite(np.asarray(framed.group_velocity)).all()


def test_the_group_velocity_of_a_lossy_band_matches_a_central_difference(sandwich):
    # the mode has complex omega(q): Re d omega / dq against a central difference of
    # Re omega with step 1e-2 q (truncation (d/q)^2 / 6 times a dispersion
    # factor of order 1 to 10: 2e-4 allowed)
    q = 3e6
    d = 1e-2 * q
    values = {}
    for s in (-1, 0, 1):
        r = mc.solve(
            sandwich,
            mc.sh_strip(1, q, 1.0),
            wavenumber=q + s * d,
            clamped={**SLIDING, "bottom": "xyz", "top": "xyz"},
            damping_frame=gfem.DampingFrame(["clad_lo", "clad_hi"]),
        )
        values[s] = r
    slope = (float(values[1].omega[0].real) - float(values[-1].omega[0].real)) / (2 * d)
    assert float(values[0].group_velocity[0]) == pytest.approx(slope, rel=2e-4)


# --- the problem, the result, the refusals -----------------------------------------


def test_the_problem_validates_and_normalizes():
    p = gfem.MechanicalEigenmode(
        2e9, n_modes=3, wavenumber=1e5, clamped={"left": "yx", "bottom": "xyz"}
    )
    assert p.clamped == (("left", "xy"), ("bottom", "xyz"))
    assert gfem.MechanicalEigenmode(1e9, clamped="a").clamped == (("a", "xyz"),)
    assert gfem.MechanicalEigenmode(1e9, clamped=["a", "b", "a"]).clamped == (
        ("a", "xyz"),
        ("b", "xyz"),
    )
    assert gfem.MechanicalEigenmode(1e9).wavenumber == 0.0
    assert "wavenumber=100000" in repr(p)
    assert "n_modes=3" in repr(p)
    for kwargs, error, match in [
        ({"target": 0.0}, ValueError, "finite frequency > 0"),
        ({"target": float("nan")}, ValueError, "finite frequency > 0"),
        ({"target": 1e9, "n_modes": 0}, ValueError, "integer >= 1"),
        ({"target": 1e9, "n_modes": 1.5}, ValueError, "integer >= 1"),
        ({"target": 1e9, "wavenumber": float("inf")}, ValueError, "must be finite"),
        ({"target": 1e9, "wavenumber": 1j}, ValueError, "real number"),
        ({"target": 1e9, "clamped": {"a": ""}}, ValueError, "non-empty string"),
        ({"target": 1e9, "clamped": {"a": "xw"}}, ValueError, "components"),
        ({"target": 1e9, "clamped": {"a": ["x"]}}, TypeError, "component strings"),
        ({"target": 1e9, "clamped": 3}, TypeError, "surface name"),
        ({"target": 1e9, "damping_frame": "frame"}, TypeError, "DampingFrame"),
    ]:
        with pytest.raises(error, match=match):
            gfem.MechanicalEigenmode(**kwargs)


def test_a_traced_target_or_wavenumber_raises():
    import jax

    for argument in ("target", "wavenumber"):

        def build(x, argument=argument):
            kwargs = {"target": 1e9, "wavenumber": 1.0, argument: x}
            return gfem.MechanicalEigenmode(**kwargs).target

        with pytest.raises(gfem.NonDifferentiableError):
            jax.grad(build)(1e9)


def test_the_result_reads_by_mode_and_has_no_potential_without_coupling(
    free_rectangle,
):
    result = mc.solve(free_rectangle, 1.0e9, 2)
    assert isinstance(result, gfem.MechanicalEigenmodeResult)
    assert result.n_modes == 2
    assert len(result.displacement_fields) == 2
    assert result.potential_fields == ()
    assert "MechanicalEigenmodeResult([" in repr(result)
    assert float(result.frequency[0]) == pytest.approx(
        float(np.real(result.omega[0])) / (2 * np.pi)
    )
    with pytest.raises(ValueError, match=r"no potential.*piezoelectric tensor"):
        result.potential(0)
    with pytest.raises(IndexError, match="Mode 2 does not exist"):
        result.displacement(2)
    assert result.displacement(-1) is result.displacement_fields[1]
    # nearest the target first
    frequencies = np.asarray(result.frequency)
    assert abs(frequencies[0] - 1.0e9) <= abs(frequencies[1] - 1.0e9)


def test_the_job_codec_round_trips_the_problem_and_the_result(free_rectangle, tmp_path):
    problem = gfem.MechanicalEigenmode(
        1.6e9, n_modes=2, wavenumber=1e6, clamped=SLIDING
    )
    job = gfem.Job.from_json(
        gfem.Job(free_rectangle, problem).to_json(tmp_path / "job.json")
    )
    assert job.problem == problem
    out = tmp_path / "results"
    result = job.run(out=out)
    direct = gfem.solve(free_rectangle, problem)
    assert np.asarray(result.omega) == pytest.approx(np.asarray(direct.omega), rel=1e-9)
    assert result.wavenumber == 1e6


def test_a_mesh_without_a_2d_path_or_a_petsc_solve_refuses_by_name(tmp_path):
    problem = gfem.MechanicalEigenmode(1e9)
    tets = tc.brick(
        tmp_path / "brick.msh",
        (1.0, 1.0),
        [("a", 1.0)],
        0.5,
        [(lambda p: p[2] < 1e-9, "hot")],
    )
    slab = tc.section(
        tmp_path / "slab.msh",
        [(0, 0, 1, 1, "a", None)],
        0.5,
        [(lambda p: p[1] < 1e-9, "hot")],
    )
    for mesh in (tets, slab):
        model = gfem.Model(mesh, materials={"a": mc.material()})
        with pytest.raises(
            NotImplementedError, match=r"MechanicalEigenmode problem solves a 2D mesh"
        ):
            gfem.solve(model, problem)


def test_the_numerics_the_rule_does_not_have_refuse_by_name(free_rectangle):
    petsc = gfem.SolverOptions(linear_solver="petsc")
    with pytest.raises(NotImplementedError, match=r"no PETSc path.*\"direct\""):
        gfem.solve(free_rectangle, gfem.MechanicalEigenmode(1e9), options=petsc)


@pytest.mark.parametrize(
    ("problem", "error", "match"),
    [
        (
            gfem.MechanicalEigenmode(1e9, ground=["bottom"]),
            ValueError,
            "ground.*no region of the mechanical domain has a piezoelectric tensor",
        ),
        (
            gfem.MechanicalEigenmode(1e9, clamped=["nowhere"]),
            gfem.RegionNotFoundError,
            "nowhere",
        ),
        (
            gfem.MechanicalEigenmode(1e9, clamped={"film": "x"}),
            ValueError,
            "must be a line bordering the mechanical domain",
        ),
    ],
)
def test_a_problem_that_does_not_fit_the_model_raises(
    free_rectangle, problem, error, match
):
    with pytest.raises(error, match=match):
        gfem.solve(free_rectangle, problem)


def test_an_absorbing_line_a_missing_property_and_a_coupled_model_raise(free_rectangle):
    rect = free_rectangle
    absorbing = gfem.Model(
        rect.mesh, materials=rect.materials, boundaries=(gfem.Absorbing("bottom"),)
    )
    with pytest.raises(ValueError, match=r"Absorbing on 'bottom'.*damping_frame"):
        gfem.solve(absorbing, gfem.MechanicalEigenmode(1e9))
    bare = gfem.Model(rect.mesh, materials={"film": gfem.Material(mass_density=1.0)})
    with pytest.raises(gfem.MissingMaterialPropertyError, match="stiffness"):
        gfem.solve(bare, gfem.MechanicalEigenmode(1e9))
    coupled = gfem.Model(
        rect.mesh,
        materials={
            "film": gfem.Material(
                stiffness=gfem.isotropic_stiffness(mc.E, mc.NU),
                mass_density=mc.RHO,
                piezoelectric_tensor=np.ones((3, 6)),
                eps_r=4.0,
            )
        },
    )
    with pytest.raises(NotImplementedError, match="piezoelectric"):
        gfem.solve(coupled, gfem.MechanicalEigenmode(1e9))


def test_too_few_converged_modes_raise_the_convergence_error(free_rectangle):
    options = gfem.SolverOptions(eigen_tolerance=1e-30)
    with pytest.raises(gfem.EigenmodeConvergenceError, match="of 2 modes near"):
        gfem.solve(
            free_rectangle, gfem.MechanicalEigenmode(1e9, n_modes=2), options=options
        )
