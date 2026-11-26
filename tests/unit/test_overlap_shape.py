r"""Shape gradients of the Kerr, Raman and free-carrier overlaps.

The overlaps read the mode at the quadrature points of its section; with the
node coordinates traced they map the section's columns by the element
Jacobians (the field's own node cotangent is the engine's). Two references:

- a closed form: scaling the port face of a filled guide, :math:`X \to sX`,
  gives :math:`\mathrm{d}\ln\gamma_\mathrm{NL}/\mathrm{d}s = -2 - 2(N^2 -
  n_\mathrm{eff}^2)/n_\mathrm{eff}^2` in the discrete problem, exactly;
- a fourth-order central difference over moved face nodes and a fresh solve.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import raman_cases as rc

import gfem

DATA = Path(__file__).parent / "data"
ORDER = gfem.SolverOptions(order=2)
N, N2, TPA = 1.5, 2.6e-20, 5e-12
F_WR90 = 8e9
F_SLAB = 16e9


def _model(mesh, nodes, materials, boundaries=None):
    model = gfem.Model(
        mesh,
        materials=materials,
        boundaries=boundaries or [gfem.PEC("walls")],
        ports=[gfem.WavePort("p1", "port1")],
    )
    return model.replace(mesh=model.mesh.with_nodes(nodes))


def _face(model, mode):
    """The port face's nodes, in-plane frame rows and shortest edge (m)."""
    section = mode.e_field._space.section
    used = np.unique(np.asarray(model.mesh.triangles)[section.faces])
    pts = section.points[np.asarray(section.triangles)]
    edge = np.min(np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1)) * section.scale
    return used, section.frame[:2], edge


def _directional(objective, mesh, make, mode_of, *, seed=3, reference=None):
    r"""The adjoint and fourth-order difference derivatives along a random motion.

    ``objective(model)`` is a vector of figures; ``make(nodes)`` the model at
    ``nodes``. The motion is an O(1) random in-plane displacement of every face
    node, in units of ``h = 1e-3`` of the shortest section edge: a truncation
    of about :math:`(h/\ell)^4 = 10^{-12}` of the figure's scale against
    solve noise of about :math:`10^{-13}/10^{-3} = 10^{-10}`. ``reference``,
    if given, is differenced instead of ``objective``.
    """
    nodes0 = np.asarray(mesh.nodes, float)
    used, frame, edge = _face(make(jnp.asarray(nodes0)), mode_of(make(nodes0)))
    v = np.zeros_like(nodes0)
    v[used] = np.random.default_rng(seed).standard_normal((len(used), 2)) @ frame
    h = 1e-3 * edge

    def figures(nodes):
        return objective(make(nodes))

    jac = jax.jacrev(figures)(jnp.asarray(nodes0))
    got = np.tensordot(np.asarray(jac), v, axes=([-2, -1], [0, 1]))
    differenced = figures if reference is None else lambda n: reference(make(n))
    f = [
        np.asarray(differenced(jnp.asarray(nodes0 + t * h * v))) for t in (2, 1, -1, -2)
    ]
    fd = (-f[0] + 8 * f[1] - 8 * f[2] + f[3]) / (12 * h)
    return got, fd


# --- filled WR-90: the scaling closed form ---------------------------------------


@pytest.fixture(scope="module")
def guide():
    mesh = gfem.Mesh.from_gmsh(DATA / "waveguide.msh", length_unit="mm")
    material = gfem.Material(eps_r=N**2, nonlinear_index=N2, two_photon_absorption=TPA)

    def make(nodes):
        return _model(mesh, nodes, {"air": material})

    return mesh, make


def _mode(model, f, **kwargs):
    return gfem.solve(model, gfem.PortMode("p1", f, **kwargs), options=ORDER)


def test_scaling_the_port_face_follows_the_closed_form(guide):
    mesh, make = guide
    nodes0 = jnp.asarray(mesh.nodes)
    base = make(nodes0)
    mode = _mode(base, [F_WR90])
    used, frame, _ = _face(base, mode)
    # X -> (1 + s) X about the face centroid, in the face's plane
    v = np.zeros(nodes0.shape)
    local = np.asarray(nodes0)[used] @ frame.T
    v[used] = (local - local.mean(axis=0)) @ frame

    def gamma(nodes):
        model = make(nodes)
        return jnp.real(
            gfem.kerr_nonlinearity(model, _mode(model, [F_WR90])).gamma_nl[0]
        )

    grad = np.asarray(jax.grad(gamma)(nodes0))
    n_eff = float(np.real(mode.n_eff[0]))
    value = float(gamma(nodes0))
    # d ln(gamma)/ds = -2 - 2 (N^2 - n_eff^2) / n_eff^2 of the discrete modes;
    # measured 6.6e-16, gate 1e-12 (the solve's backward error is 1e-14)
    closed = -2 - 2 * (N**2 - n_eff**2) / n_eff**2
    assert np.sum(grad * v) / value == pytest.approx(closed, rel=1e-12)


def test_free_carriers_scale_with_the_face(guide):
    mesh, make = guide
    nodes0 = jnp.asarray(mesh.nodes)
    base = make(nodes0)
    mode = _mode(base, [F_WR90])
    used, frame, _ = _face(base, mode)
    v = np.zeros(nodes0.shape)
    local = np.asarray(nodes0)[used] @ frame.T
    v[used] = (local - local.mean(axis=0)) @ frame

    def area(nodes):
        model = make(nodes)
        return gfem.free_carriers(model, _mode(model, [F_WR90])).area

    # the area of the one region is the face's: it grows as s^2, so d ln A/ds = 2
    # (round-off of a few sums of 1e-16, gate 1e-12)
    grad = np.asarray(jax.grad(area)(nodes0))
    assert np.sum(grad * v) / float(area(nodes0)) == pytest.approx(2.0, rel=1e-12)


# --- finite differences over moved nodes -----------------------------------------


@pytest.fixture(scope="module")
def slab():
    mesh = gfem.Mesh.from_gmsh(DATA / "slab_waveguide.msh", length_unit="mm")
    materials = {
        "slab": gfem.Material(eps_r=4.0, nonlinear_index=N2, two_photon_absorption=TPA),
        "air": gfem.Material(),
    }

    def make(nodes):
        return _model(mesh, nodes, materials)

    return mesh, make


def test_gamma_nl_matches_finite_differences_over_moved_nodes(slab):
    mesh, make = slab

    def gamma(model):
        mode = _mode(model, [F_SLAB])
        return jnp.real(gfem.kerr_nonlinearity(model, mode).gamma_nl)

    got, fd = _directional(gamma, mesh, make, lambda m: _mode(m, [F_SLAB]))
    # measured 7.8e-10 (truncation 1e-12, noise 1e-10..1e-8); gate 1e-6
    assert got == pytest.approx(fd, rel=1e-6, abs=0)


def test_the_loss_part_and_free_carriers_match_too(slab):
    mesh, make = slab

    def figures(model):
        mode = _mode(model, [F_SLAB])
        fc = gfem.free_carriers(model, mode)
        gamma = gfem.kerr_nonlinearity(model, mode).gamma_nl[0]
        return jnp.stack([jnp.imag(gamma), fc.overlap[0], fc.area, fc.generation[0]])

    got, fd = _directional(figures, mesh, make, lambda m: _mode(m, [F_SLAB]))
    assert got == pytest.approx(fd, rel=1e-6, abs=0)


def test_a_grid_with_reduced_rows_and_the_coupling_modulus_match(slab):
    mesh, make = slab
    grid = np.linspace(15e9, 17e9, 7)
    # a reduced row's gradient is the exact mode's (it is polished first), so
    # the differences solve every row exactly
    tight = gfem.SolverOptions(order=2, port_mode_tolerance=1e-9)
    exact = gfem.SolverOptions(order=2, port_mode_tolerance=1e-12)
    base = gfem.solve(
        make(jnp.asarray(mesh.nodes)), gfem.PortMode("p1", grid), options=tight
    )
    solved = np.asarray(base.solved)
    assert solved.sum() >= 2
    (reduced, *_) = np.flatnonzero(~solved)  # a reduced row stands in the grid
    other = int(np.flatnonzero(solved)[0])

    def figures_with(options):
        def figures(model):
            mode = gfem.solve(model, gfem.PortMode("p1", grid), options=options)
            g = gfem.kerr_coupling(model, mode).gamma_nl
            # the modulus does not depend on each row's phase: the guard passes
            entry = jnp.abs(g[other, other, reduced, reduced]) ** 2
            return jnp.stack([jnp.sum(jnp.abs(g) ** 2), entry])

        return figures

    got, fd = _directional(
        figures_with(tight),
        mesh,
        make,
        lambda m: gfem.solve(m, gfem.PortMode("p1", grid), options=tight),
        reference=figures_with(exact),
    )
    # The overlap reads the reduced row's own field, off the exact mode by
    # $\delta \approx 1.2 r$ to $1.3 r$ in its DOF vector ($r$ the row's
    # residual, measured), and each node's term is linear in each of the row's
    # $m \le 4$ factors in an entry $|\gamma_{aarr}|^2$: an error of at most
    # $m\delta$ of the node terms' scale $S = \sum_i |g_i \cdot v_i|$. The
    # random motion's derivative cancels to $|\mathrm{d}F| = S/\kappa$, with
    # $\kappa = 23$ measured, so the relative error is at most
    # $\kappa m \delta$. At the default tolerance $10^{-6}$ a five-row grid's
    # reduced row has $r = 2.8\times10^{-7}$ ($\delta = 3.7\times10^{-7}$):
    # bound $3.4\times10^{-5}$, measured $4.4\times10^{-6}$, so a gate of
    # $10^{-6}$ cannot hold there (the row is only promised to the tolerance).
    # At $10^{-9}$ the reduced row has $r = 2.4\times10^{-11}$
    # ($\delta = 2.9\times10^{-11}$): bound $2.7\times10^{-9}$ beside the
    # differences' noise of $10^{-9}$; measured $8.3\times10^{-10}$ and
    # $1.0\times10^{-9}$, gate $10^{-6}$.
    assert got == pytest.approx(fd, rel=1e-6, abs=0)


def test_a_material_field_moves_with_its_elements(slab):
    mesh, _ = slab
    k = len(mesh.elements("slab"))
    rng = np.random.default_rng(5)
    field = gfem.MaterialField(
        region="slab",
        eps_r=4.0 + 0.4 * rng.random(k),
        nonlinear_index=N2 * (1 + rng.random(k)),
        two_photon_absorption=TPA,
    )
    materials = {"slab": field, "air": gfem.Material()}

    def make(nodes):
        return _model(mesh, nodes, materials)

    def gamma(model):
        mode = _mode(model, [F_SLAB])
        g = gfem.kerr_nonlinearity(model, mode).gamma_nl[0]
        return jnp.stack([jnp.real(g), jnp.imag(g)])

    # the differences hold each element's values while its nodes move: the
    # convected derivative, which the solve and the overlaps both give; the
    # budget is the uniform slab's, measured 3.1e-10 and 2.5e-10, gate 1e-6
    got, fd = _directional(gamma, mesh, make, lambda m: _mode(m, [F_SLAB]))
    assert got == pytest.approx(fd, rel=1e-6, abs=0)


def test_the_raman_gain_matches_finite_differences(tmp_path):
    material = gfem.Material(
        eps_r=N**2,
        raman_spectrum=rc.flat_spectrum(
            6e-14, symmetry="isotropic", perpendicular=3.6e-15
        ),
    )
    ref = rc.guide_model(tmp_path / "g.msh", 2.0, 1.2, material, nx=4, ny=3)
    pump = rc.F_S + 13.2e12

    def make(nodes):
        return ref.replace(mesh=ref.mesh.with_nodes(nodes))

    def figures(model):
        mode = _mode(model, [pump, rc.F_S])
        return gfem.raman_gain(model, mode).gain[1, 0][None]

    got, fd = _directional(figures, ref.mesh, make, lambda m: _mode(m, [pump, rc.F_S]))
    assert got == pytest.approx(fd, rel=1e-6, abs=0)


# --- refusals ---------------------------------------------------------------------


def test_the_overlaps_refuse_staged_node_coordinates(guide):
    mesh, make = guide
    nodes0 = jnp.asarray(mesh.nodes)
    mode = _mode(make(nodes0), [F_WR90])

    @jax.jit
    def staged(nodes):
        return jnp.real(gfem.kerr_nonlinearity(make(nodes), mode).gamma_nl[0])

    with pytest.raises(TypeError, match=r"jax\.jit"):
        staged(nodes0)


def test_a_mode_solved_outside_the_gradient_is_refused(guide):
    # the mode's field holds no node derivative: the overlap would get only the
    # motion of its quadrature points
    mesh, make = guide
    nodes0 = jnp.asarray(mesh.nodes)
    mode = _mode(make(nodes0), [F_WR90])
    figures = {
        "kerr_nonlinearity": lambda m: gfem.kerr_nonlinearity(m, mode).gamma_nl[0],
        "free_carriers": lambda m: gfem.free_carriers(m, mode).overlap[0],
    }
    for owner, figure in figures.items():
        with pytest.raises(gfem.NonDifferentiableError, match=rf"{owner}.*constants"):
            jax.grad(lambda n, f=figure: jnp.real(f(make(n))))(nodes0)


def test_a_traced_mode_on_a_constant_model_is_refused(guide):
    # the mode moves with the nodes, the constant model's section does not
    mesh, make = guide
    nodes0 = jnp.asarray(mesh.nodes)
    base = make(nodes0)
    figures = {
        "kerr_nonlinearity": lambda m: gfem.kerr_nonlinearity(base, m).gamma_nl[0],
        "free_carriers": lambda m: gfem.free_carriers(base, m).overlap[0],
        "three_wave_coupling": lambda m: gfem.three_wave_coupling(base, [m, m]).kappa[
            0, 0, 0
        ],
        "spdc": lambda m: gfem.spdc(base, m, m, m).coupling,
    }
    for owner, figure in figures.items():

        def objective(nodes, f=figure):
            return jnp.real(f(_mode(make(nodes), [F_WR90])))

        with pytest.raises(gfem.NonDifferentiableError, match=rf"gfem\.{owner}"):
            jax.grad(objective)(nodes0)
