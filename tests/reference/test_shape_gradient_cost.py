"""Cost of shape gradients on the driven timing case.

On ``driven/r8`` (the IHP coplanar line at gsim defaults, order 2, both
ports excited) the +y sidewall of the signal conductor moves by a smooth
bump, through `gfem.shape.boundary_morph` with the ports held. Gates:

- the node-coordinate kernel costs at most a tenth of one forward linear
  solve per field pair (one excited port at one frequency: one pair of
  one column each), at the same mesh and threads;
- the whole backward of `jax.vjp` (S cotangents: no solve, the shape
  kernel over every field pair, the morph's transposed extension solve)
  takes at most a quarter of the forward wall time, for a uniform sweep
  and for an adaptive sweep with frozen samples, both warm (as in every
  iteration of an optimization after the first).

Run under the heavy lock with 4 threads.
"""

import importlib.util
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem._engine.driven import DrivenRule
from gfem._engine.sensitivity import sensitivity
from gfem.shape import boundary_morph

THREADS = 4
UM = 1e-6
#: Per field pair, the kernel's wall per forward linear solve.
MAX_PAIR_RATIO = 0.1
#: The whole backward's wall per forward wall.
MAX_RATIO = 0.25
WEIGHTS = jnp.array([[1.0, 0.5], [2.0, -1.0]])


@pytest.fixture(scope="module")
def r8():
    ref = os.environ.get("GFEM_REFERENCE_DIR")
    if not ref:
        pytest.skip("GFEM_REFERENCE_DIR not set (needs the driven Palace runs)")
    path = Path(__file__).with_name("test_driven.py")
    spec = importlib.util.spec_from_file_location("_reference_driven", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._model(Path(ref) / "driven" / "r8")


def _signal_wall(p):
    """The signal conductor's +y sidewall: y = 10 um, z in the metal layer."""
    return (
        (np.abs(p[:, 1] - 10 * UM) < 1e-12)
        & (p[:, 2] > 11 * UM)
        & (p[:, 2] < 14.1 * UM)
    )


def _bump(morph):
    x = morph.points[:, 0]
    return np.where(np.abs(x) < 300 * UM, np.cos(np.pi * x / (600 * UM)) ** 2, 0.0)


def test_shape_kernel_costs_a_tenth_of_a_forward_solve_per_pair(r8):
    problem = gfem.Driven(frequency=50e9, excite="p1")
    options = gfem.SolverOptions(threads=THREADS)
    rule = DrivenRule(r8, problem, options)
    rule.forward(r8, keep=False)  # warm: the discretization and its spaces
    start = time.perf_counter()
    _, entry = rule.forward(r8, keep=True)
    forward = time.perf_counter() - start
    ct = np.zeros((1, 2, 1), complex)
    ct[0, 1, 0] = 1.0
    pairs = rule.backward(r8, entry, {"s": ct})
    assert len(pairs) == 1
    assert pairs[0].left.shape[1] == 1

    def kernel():
        return sensitivity(
            r8,
            pairs,
            {("nodes",)},
            form=rule.form,
            shape=lambda: rule.shape_terms(r8),
        )[("nodes",)]

    kernel()  # warm
    start = time.perf_counter()
    grad = kernel()
    pair = time.perf_counter() - start
    entry.free()
    print(  # noqa: T201
        f"r8 shape kernel: forward solve {forward:.2f} s, one pair {pair:.3f} s = "
        f"{pair / forward:.4f} forward solves; load {os.getloadavg()}"
    )
    assert np.isfinite(grad).all()
    assert np.abs(grad).max() > 0
    assert pair <= MAX_PAIR_RATIO * forward


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
def test_shape_backward_is_a_quarter_of_the_forward(r8, sweep):
    morph = boundary_morph(r8, "topmetal2_z", where=_signal_wall)
    bump = jnp.asarray(_bump(morph))
    if sweep == "uniform":
        problem = gfem.Driven(frequency=(20e9, 80e9))
    else:
        grid = tuple(float(f) for f in np.linspace(1e9, 100e9, 100))
        samples = (grid[0], grid[99], grid[49], grid[24], grid[74])
        problem = gfem.Driven(frequency=grid, sweep="adaptive", samples=samples)
    options = gfem.SolverOptions(threads=THREADS)

    def objective(a):
        model = r8.replace(mesh=morph(a * UM * bump))
        s = gfem.solve(model, problem, options=options).s.values
        return jnp.sum(WEIGHTS * jnp.abs(s) ** 2) / s.shape[0]

    a = jnp.asarray(2.0)
    # warm up: the discretization cache, and JAX's compilation of the backward
    value, pullback = jax.vjp(objective, a)
    jax.block_until_ready(pullback(jnp.ones_like(value)))
    start = time.perf_counter()
    jax.block_until_ready(objective(a))
    forward = time.perf_counter() - start

    start = time.perf_counter()
    value, pullback = jax.vjp(objective, a)
    jax.block_until_ready(value)
    forward_kept = time.perf_counter() - start
    start = time.perf_counter()
    (grad,) = pullback(jnp.ones_like(value))
    jax.block_until_ready(grad)
    backward = time.perf_counter() - start

    print(  # noqa: T201
        f"r8 shape gradient ({sweep}, {morph.n_params} moving nodes): forward "
        f"{forward:.2f} s, forward with kept fields {forward_kept:.2f} s, backward "
        f"{backward:.3f} s = {backward / forward:.4f}x forward; dJ/da = "
        f"{float(grad):.6g} per um; load {os.getloadavg()}"
    )
    assert math.isfinite(float(grad))
    assert float(grad) != 0
    assert backward <= MAX_RATIO * forward
