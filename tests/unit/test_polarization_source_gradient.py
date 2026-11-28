r"""Gradients through the pump solve and the generated solve: structure and identities.

The cell is the slab of `second_order_cases` at 10 cells/um along z (336
tetrahedra) solved at order 2. $J = |b|^2$ of the generated TM wave at the top port
is a smooth function of the tensor's scale $s$, the pump's powers $p$ and phases
$\varphi$, because the polarization is bilinear in the pump amplitudes
$a = \sqrt p\,e^{i\varphi}$ and linear in $s$ and the generated wave is linear in it:

- second harmonic, $P \propto s\,p\,e^{2i\varphi}$: $\partial J/\partial s = 2J/s$,
  $\partial J/\partial p = 2J/p$, $\partial|b|/\partial\varphi = 0$,
  $\partial\arg b/\partial\varphi = 2$;
- sum frequency, $P \propto s\,\sqrt{p_1 p_2}\,e^{i(\varphi_1 + \varphi_2)}$, so
  $J\propto p_1 p_2$: $\partial J/\partial p_k = J/p_k$ and
  $\partial\arg b/\partial\varphi_k = 1$.

Tolerance. Each identity holds exactly for the exact solves, so the gradient
and the value differ by round-off only: the forward and the adjoint solves of one
direct factorization carry a relative error of at most (condition number of the
cell's matrix, below $10^{6}$) times $10^{-16}$, that is $10^{-10}$, and the
identities measured below $10^{-14}$ (the test tolerance is the bound). The jit
and the eager gradient run the same host solves on pump coefficients that differ
by fusion round-off, so they agree to the same bound.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from second_order_cases import FREQUENCY, D, slab_mesh, slab_model

import gfem
from gfem._engine import volume_loads
from gfem._solve_cache import solve_key

OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
F2 = 1.2 * FREQUENCY
#: Identities and jit against eager, relative to the value (see the module docstring).
REL = 1e-10


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return slab_mesh(tmp_path_factory.mktemp("cell") / "cell.msh", 10)


@pytest.fixture(scope="module")
def pump(mesh):
    return gfem.solve(
        slab_model(mesh, 2.2, chi2=False),
        gfem.Driven([FREQUENCY, F2], excite="z_min TM", keep_fields=True),
        options=OPTIONS,
    )


def _wave(mesh, pump, *, scale=1.0, powers=(1.0,), phases=(0.0,)):
    """The generated TM wave at the top port for one or two pumps."""
    d = np.zeros((3, 6))
    d[0, 0] = D
    model = slab_model(mesh, 2.2, chi2=False).with_material(
        "slab",
        gfem.MaterialField(eps_r=2.2**2, second_order_nonlinear_tensor=scale * d),
    )
    pumps = [
        gfem.Pump(pump, f, incident_power=p, incident_phase=phi)
        for f, p, phi in zip((FREQUENCY, F2), powers, phases, strict=False)
    ]
    if len(pumps) == 1:
        source = gfem.second_order_polarization(model, pumps[0])
    else:
        source = gfem.second_order_polarization(
            model, tuple(pumps), process="sum-frequency"
        )
    generated = gfem.solve(
        slab_model(mesh, 2.2, chi2=False),
        gfem.Driven(source.frequency, excite=(), sources=source),
        options=OPTIONS,
    )
    return generated.port_wave("z_max TM", source.name)[0]


def _power(mesh, pump, **kwargs):
    return jnp.abs(_wave(mesh, pump, **kwargs)) ** 2


def test_a_driven_without_a_polarization_has_no_leaves_and_keeps_its_key(mesh):
    model = slab_model(mesh, 2.2, chi2=False)
    plain = gfem.Driven(FREQUENCY, excite="z_min TM")
    again = gfem.Driven(FREQUENCY, excite="z_min TM")
    assert jax.tree_util.tree_leaves(plain) == []
    leaves, treedef = jax.tree_util.tree_flatten(plain)
    assert jax.tree_util.tree_unflatten(treedef, leaves) == plain
    assert solve_key(model, plain, OPTIONS)[0] == solve_key(model, again, OPTIONS)[0]


def test_a_driven_with_a_polarization_has_two_leaves(mesh, pump):
    model = slab_model(mesh, 2.2, chi2=True)
    source = gfem.second_order_polarization(
        model, gfem.Pump(pump, FREQUENCY, incident_power=1.0)
    )
    problem = gfem.Driven(source.frequency, excite=(), sources=source)
    assert len(jax.tree_util.tree_leaves(problem)) == 2


@pytest.fixture(scope="module")
def shg(mesh, pump):
    value = _power(mesh, pump)
    grads = jax.grad(
        lambda s, p, phi: _power(mesh, pump, scale=s, powers=(p,), phases=(phi,)),
        argnums=(0, 1, 2),
    )(1.0, 1.0, 0.3)
    return value, grads


def test_the_tensor_scale_and_the_pump_power_scale_the_power_by_two(mesh, pump, shg):
    value = _power(mesh, pump)
    d_scale, d_power, _ = shg[1]
    assert float(d_scale) == pytest.approx(2 * float(value), rel=REL)
    assert float(d_power) == pytest.approx(2 * float(value), rel=REL)


def test_the_pump_phase_turns_the_wave_by_twice_its_value(mesh, pump):
    def modulus(phi):
        return jnp.abs(_wave(mesh, pump, phases=(phi,)))

    def argument(phi):
        return jnp.angle(_wave(mesh, pump, phases=(phi,)))

    b = float(modulus(0.3))
    assert abs(float(jax.grad(modulus)(0.3))) <= REL * b
    assert float(jax.grad(argument)(0.3)) == pytest.approx(2.0, rel=REL)


def test_the_sum_frequency_power_is_the_product_of_the_pump_powers(mesh, pump):
    value = float(_power(mesh, pump, powers=(1.0, 1.0), phases=(0.0, 0.0)))
    g = jax.grad(
        lambda p1, p2, f1, f2: _power(mesh, pump, powers=(p1, p2), phases=(f1, f2)),
        argnums=(0, 1, 2, 3),
    )(1.0, 1.0, 0.0, 0.0)
    assert float(g[0]) == pytest.approx(value, rel=REL)
    assert float(g[1]) == pytest.approx(value, rel=REL)
    # the power does not depend on the phases
    assert abs(float(g[2])) <= REL * value
    assert abs(float(g[3])) <= REL * value


def test_a_chunk_of_one_element_gives_the_same_gradient(mesh, pump, shg, monkeypatch):
    # the adjoint assembles the maps again, a group of points at a time: with
    # one element per group it reads the same entries (the `REL` bound)
    monkeypatch.setattr(volume_loads, "_CHUNK_ENTRIES", 1)
    grads = jax.grad(
        lambda s, p, phi: _power(mesh, pump, scale=s, powers=(p,), phases=(phi,)),
        argnums=(0, 1),
    )(1.0, 1.0, 0.3)
    for got, want in zip(grads, shg[1][:2], strict=True):
        assert abs(float(got) - float(want)) <= REL * float(shg[0])


def test_jit_equals_eager(mesh, pump, shg):
    def objective(s, p, phi):
        return _power(mesh, pump, scale=s, powers=(p,), phases=(phi,))

    value, grads = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))(
        1.0, 1.0, 0.3
    )
    assert float(value) == pytest.approx(float(shg[0]), rel=REL)
    for got, want in zip(grads, shg[1], strict=True):
        scale = max(abs(float(want)), float(shg[0]))
        assert abs(float(got) - float(want)) <= REL * scale


@pytest.fixture(scope="module")
def source(mesh, pump):
    return gfem.second_order_polarization(
        slab_model(mesh, 2.2, chi2=True),
        gfem.Pump(pump, FREQUENCY, incident_power=1.0),
    )


def test_the_node_coordinates_are_refused_with_a_source(mesh, source):
    def objective(t):
        moved = mesh.with_nodes(mesh.nodes + t * np.array([0.0, 0.0, 1e-9]))
        model = slab_model(mesh, 2.2, chi2=False).replace(mesh=moved)
        problem = gfem.Driven(source.frequency, excite=(), sources=source)
        result = gfem.solve(model, problem, options=OPTIONS)
        return jnp.abs(result.port_wave("z_max TM", source.name)[0]) ** 2

    with pytest.raises(gfem.NonDifferentiableError, match="PolarizationSource"):
        jax.grad(objective)(0.0)


def test_the_emitted_power_of_the_source_is_refused(mesh, source):
    def objective(s):
        scaled = gfem.PolarizationSource(
            **{
                **{f: getattr(source, f) for f in source.__dataclass_fields__},
                "re": s * source.re,
            }
        )
        problem = gfem.Driven(source.frequency, excite=(), sources=scaled)
        model = slab_model(mesh, 2.2, chi2=False)
        result = gfem.solve(model, problem, options=OPTIONS)
        return result.emitted_power[source.name][0]

    with pytest.raises(gfem.NonDifferentiableError, match="emitted power"):
        jax.grad(objective)(1.0)


def test_a_non_finite_polarization_is_refused_when_it_is_traced(mesh, source):
    def objective(s):
        bad = gfem.PolarizationSource(
            **{
                **{f: getattr(source, f) for f in source.__dataclass_fields__},
                "re": s * source.re,
            }
        )
        problem = gfem.Driven(source.frequency, excite=(), sources=bad)
        model = slab_model(mesh, 2.2, chi2=False)
        result = gfem.solve(model, problem, options=OPTIONS)
        return jnp.abs(result.port_wave("z_max TM", source.name)[0]) ** 2

    # the check runs in the host forward of the traced solve, which JAX reports
    # as a runtime error carrying the message
    with pytest.raises(jax.errors.JaxRuntimeError, match="non-finite"):
        jax.grad(objective)(jnp.nan)
