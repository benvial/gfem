"""S-parameter gradients through wave ports against finite differences.

A two-layer parallel-plate line fed by a wave port at each end, with a
"core" section in the middle that carries a random per-element lossy
permittivity and conductivity. The design stays away from the port faces,
so the port modes are constants of the solve. The layers differ in
permittivity, so the port mode is hybrid (``E_n != 0``) and the modal
correction is active:

- **lossless** ports: the mode is real up to one phase, so a port's S
  functional is its excitation scaled and the S adjoint is the port's own
  solution (no solve);
- **lossy** ports (a lossy lower layer in the leads, offset reference
  planes): the S functional projects on the conjugated mode, which is not
  parallel to the excitation, so the adjoint takes a solve with the port
  operator (Robin terms and modal corrections).

Each cell of the verification matrix (port x input x solver x sweep)
checks the gradient along random directions against fourth-order
differences. Adaptive cells read S at the frozen sample frequencies, where
the reduced model reproduces the full solve of the perturbed design too,
and its basis holds the S functionals' adjoints, so the frozen-basis
gradient is the full solve's there.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from finite_differences import assert_gradient_matches, unit_directions

import gfem

SOLVERS = ["direct", "bddc_gmres"]
FREQUENCIES = (12e9, 24e9)
LAYERS = (("lo", 0.5), ("hi", 0.5))
GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 11))
SAMPLES = (GRID[0], GRID[10], GRID[5], GRID[2])
ON = [GRID.index(f) for f in SAMPLES]
ADAPTIVE = gfem.Driven(frequency=GRID, sweep="adaptive", samples=SAMPLES)

PORTS = {
    "lossless": {
        "lo": gfem.Material(eps_r=4.0),
        "hi": gfem.Material(eps_r=2.0),
    },
    "lossy": {
        "lo": gfem.Material(eps_r=4.0, tan_delta=0.05),
        "hi": gfem.Material(eps_r=2.0),
    },
}
OFFSETS = {"lossless": (0.0, 0.0), "lossy": (0.3e-3, 0.1e-3)}


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(
        path, sm.strip_line(4.0, 2.0, 1.0, (8, 2, 2), layers=LAYERS, core=(1.0, 3.0))
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line(mesh, port):
    """Wave ports at both ends; ``port="mixed"``: lossy p1, lumped p2."""
    if port == "mixed":
        return _line(mesh, "lossy").replace(
            ports=[
                gfem.WavePort("p1", "port1", offset=OFFSETS["lossy"][0]),
                gfem.LumpedPort("p2", "port2", "+z", impedance=60.0),
            ]
        )
    d1, d2 = OFFSETS[port]
    return gfem.Model(
        mesh,
        materials={**PORTS[port], "core": gfem.Material(eps_r=3.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.WavePort("p1", "port1", offset=d1),
            gfem.WavePort("p2", "port2", offset=d2),
        ],
    )


def _design(mesh, seed=0):
    """Random P0 permittivity (lossy) and conductivity in the core."""
    rng = np.random.default_rng(seed)
    n = len(mesh.elements("core"))
    eps = rng.uniform(2.0, 6.0, n) * (1 - 0.02j)
    sigma = rng.uniform(0.5, 2.0, n)
    return {"eps_r": jnp.asarray(eps), "conductivity": jnp.asarray(sigma)}, rng


def _options(solver):
    return gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12)


WEIGHTS = np.array([[1.0, 0.4], [-0.6, 1.5]])


def _s_objective(model, values, name, options, problem, rows):  # noqa: PLR0917
    """<W, |S|^2> + Re(c S21) summed over the grid ``rows`` of ``problem``."""

    def objective(p):
        core = gfem.MaterialField(**{**values, name: p})
        m = model.with_material("core", core)
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        s = gfem.solve(m, problem, options=options).s.values[jnp.asarray(rows)]
        return jnp.sum(WEIGHTS * jnp.abs(s) ** 2) + jnp.sum(
            jnp.real((0.3 - 0.7j) * s[:, 1, 0])
        )

    return objective


# Fourth-order differences (see tests/finite_differences.py).
STEPS = {"eps_r": 1e-2, "conductivity": 5e-2}


# --- the verification matrix: wave port x {eps, sigma} x solver x sweep ----------


# Every port x input on the direct solver; GMRES + BDDC (the adjoint solve
# through the port operator's Woodbury update, the adjoint loads' extra
# columns at the samples) on the permittivity cells, which take the same
# code path as the conductivity ones.
UNIFORM_CELLS = [
    *((port, name, "direct") for port in PORTS for name in STEPS),
    ("lossless", "eps_r", "bddc_gmres"),
    ("lossy", "eps_r", "bddc_gmres"),
]
ADAPTIVE_CELLS = [
    *((port, name, "direct") for port in PORTS for name in STEPS),
    ("lossy", "eps_r", "bddc_gmres"),
]


@pytest.mark.parametrize(("port", "name", "solver"), UNIFORM_CELLS)
def test_uniform_s_gradient_matches_central_differences(mesh, port, name, solver):
    values, rng = _design(mesh)
    problem = gfem.Driven(frequency=FREQUENCIES)
    objective = _s_objective(
        _line(mesh, port), values, name, _options(solver), problem, [0, 1]
    )
    p = values[name]
    assert_gradient_matches(
        objective, p, unit_directions(rng, p.shape, 2), step=STEPS[name]
    )


@pytest.mark.parametrize(("port", "name", "solver"), ADAPTIVE_CELLS)
def test_adaptive_s_gradient_matches_central_differences(mesh, port, name, solver):
    # The lossy cells need the S functionals' adjoints in the basis: without
    # them the frozen-basis gradient is off by 1e-4..1e-3 even at the samples.
    values, rng = _design(mesh, seed=1)
    objective = _s_objective(
        _line(mesh, port), values, name, _options(solver), ADAPTIVE, ON
    )
    p = values[name]
    assert_gradient_matches(
        objective, p, unit_directions(rng, p.shape, 2), step=STEPS[name]
    )


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
def test_a_wave_port_and_a_lumped_port_mix(mesh, sweep):
    values, rng = _design(mesh, seed=4)
    problem, rows = (
        (gfem.Driven(frequency=FREQUENCIES), [0, 1])
        if sweep == "uniform"
        else (ADAPTIVE, ON)
    )
    objective = _s_objective(
        _line(mesh, "mixed"), values, "eps_r", _options("direct"), problem, rows
    )
    p = values["eps_r"]
    directions = unit_directions(rng, p.shape, 2)
    assert_gradient_matches(objective, p, directions, step=STEPS["eps_r"])


def test_complex_permittivity_gradient_has_both_parts(mesh):
    values, rng = _design(mesh, seed=2)
    problem = gfem.Driven(frequency=FREQUENCIES)
    objective = _s_objective(
        _line(mesh, "lossy"), values, "eps_r", _options("direct"), problem, [0, 1]
    )
    eps = values["eps_r"]
    (d,) = unit_directions(rng, eps.shape, 1)
    assert_gradient_matches(objective, eps, [d, 1j * d], step=STEPS["eps_r"])


# --- consistency: the port's own solution vs an explicit adjoint solve -----------


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("port", ["lossless", "lossy"])
def test_the_s21_gradient_does_not_depend_on_which_ports_are_excited(
    mesh, port, solver
):
    # Exciting p1 only leaves S21's adjoint to a solve with the port
    # operator; with both ports excited a lossless port reuses p2's own
    # solution and a lossy one solves too.
    values, _ = _design(mesh, seed=3)
    model = _line(mesh, port)

    def s21(problem):
        def objective(eps):
            m = model.with_material(
                "core", gfem.MaterialField(**{**values, "eps_r": eps})
            )
            m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
            s = gfem.solve(m, problem, options=_options(solver)).s
            return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

        return jax.grad(objective)(values["eps_r"])

    full = s21(gfem.Driven(frequency=FREQUENCIES))
    adjoint = s21(gfem.Driven(frequency=FREQUENCIES, excite="p1"))
    np.testing.assert_allclose(adjoint, full, rtol=1e-8, atol=1e-10 * abs(full).max())


@pytest.mark.parametrize(("port", "solves"), [("lossless", 0), ("lossy", 2)])
def test_only_lossy_wave_ports_take_an_adjoint_solve(mesh, port, solves, monkeypatch):
    from gfem._engine import driven

    values, _ = _design(mesh, seed=5)
    objective = _s_objective(
        _line(mesh, port),
        values,
        "eps_r",
        _options("direct"),
        gfem.Driven(frequency=FREQUENCIES),
        [0, 1],
    )
    value, pullback = jax.vjp(objective, values["eps_r"])
    prepared = []
    original = driven.prepare

    def counting(*args, **kwargs):
        prepared.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(driven, "prepare", counting)
    pullback(jnp.ones_like(value))
    assert len(prepared) == solves  # one operator per frequency
