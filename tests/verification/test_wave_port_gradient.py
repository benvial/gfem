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

The cells of the verification matrix (port x input x solver x sweep) are in
test_gradient_matrix.py (the line is `gradient_cases.wave_line`); here a
wave port beside a lumped one, the complex permittivity and the adjoint
paths. Adaptive checks read S at the frozen sample frequencies, where the
reduced model reproduces the full solve of the perturbed design too, and
its basis holds the S functionals' adjoints, so the frozen-basis gradient
is the full solve's there.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from finite_differences import assert_gradient_matches, unit_directions
from gradient_cases import core_design, s_functional, tight, wave_line, wave_mesh

import gfem

SOLVERS = ["direct", "bddc_gmres"]
FREQUENCIES = (12e9, 24e9)
GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 11))
SAMPLES = (GRID[0], GRID[10], GRID[5], GRID[2])
ON = [GRID.index(f) for f in SAMPLES]
ADAPTIVE = gfem.Driven(frequency=GRID, sweep="adaptive", samples=SAMPLES)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return wave_mesh(tmp_path_factory.mktemp("line"))


def _s_objective(model, values, name, options, problem, rows):  # noqa: PLR0917
    """<W, |S|^2> + Re(c S21) summed over the grid ``rows`` of ``problem``."""

    def objective(p):
        core = gfem.MaterialField(**{**values, name: p})
        m = model.with_material("core", core)
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        s = gfem.solve(m, problem, options=options).s.values[jnp.asarray(rows)]
        return s_functional(s)

    return objective


# Fourth-order differences (see tests/finite_differences.py).
STEPS = {"eps_r": 1e-2, "conductivity": 5e-2}


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
def test_a_wave_port_and_a_lumped_port_mix(mesh, sweep):
    values, rng = core_design(mesh, seed=4)
    problem, rows = (
        (gfem.Driven(frequency=FREQUENCIES), [0, 1])
        if sweep == "uniform"
        else (ADAPTIVE, ON)
    )
    objective = _s_objective(
        wave_line(mesh, "mixed"), values, "eps_r", tight("direct"), problem, rows
    )
    p = values["eps_r"]
    directions = unit_directions(rng, p.shape, 2)
    assert_gradient_matches(objective, p, directions, step=STEPS["eps_r"])


def test_complex_permittivity_gradient_has_both_parts(mesh):
    values, rng = core_design(mesh, seed=2)
    problem = gfem.Driven(frequency=FREQUENCIES)
    objective = _s_objective(
        wave_line(mesh, "lossy"), values, "eps_r", tight("direct"), problem, [0, 1]
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
    values, _ = core_design(mesh, seed=3)
    model = wave_line(mesh, port)

    def s21(problem):
        def objective(eps):
            m = model.with_material(
                "core", gfem.MaterialField(**{**values, "eps_r": eps})
            )
            m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
            s = gfem.solve(m, problem, options=tight(solver)).s
            return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

        return jax.grad(objective)(values["eps_r"])

    full = s21(gfem.Driven(frequency=FREQUENCIES))
    adjoint = s21(gfem.Driven(frequency=FREQUENCIES, excite="p1"))
    np.testing.assert_allclose(adjoint, full, rtol=1e-8, atol=1e-10 * abs(full).max())


@pytest.mark.parametrize(("port", "solves"), [("lossless", 0), ("lossy", 2)])
def test_only_lossy_wave_ports_take_an_adjoint_solve(mesh, port, solves, monkeypatch):
    from gfem._engine import driven

    values, _ = core_design(mesh, seed=5)
    objective = _s_objective(
        wave_line(mesh, port),
        values,
        "eps_r",
        tight("direct"),
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
