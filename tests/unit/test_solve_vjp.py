"""`solve_vjp` hands every host call the design state it was traced with."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._adjoint import adjoint_tolerance, solve_vjp
from gfem._pytree import register_fields


@dataclasses.dataclass(frozen=True)
class Scaled:
    """A problem with one array leaf."""

    scale: object


register_fields(Scaled, ("scale",))


class ScaledRule:
    """Returns ``scale * sum(eps_r)``, reading the problem from ``self.problem``."""

    physics = "em"
    form = "electrostatic"
    differentiable_outputs = frozenset()

    def __init__(self, problem):
        self.problem = problem
        self.options = gfem.SolverOptions()

    def output_specs(self, model):  # noqa: ARG002
        return {"y": jax.ShapeDtypeStruct((), jnp.float64)}

    def role(self, kind):  # noqa: ARG002
        return "unread"

    def forward(self, model, *, keep):  # noqa: ARG002
        eps = np.asarray(model.materials["box"].eps_r)
        return {"y": float(np.asarray(self.problem.scale)) * eps.sum()}, None


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "box.msh"
    sm.write_msh22(path, sm.plate(1.0, 1.0, (("box", 1.0),), (1, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path)
    n = len(mesh.elements("box"))
    return gfem.Model(mesh, materials={"box": gfem.MaterialField(eps_r=np.ones(n))})


def _solve(model, scale):
    problem = Scaled(scale)
    return solve_vjp(ScaledRule(problem), model, problem)["y"]


def test_host_forward_reads_problem_leaves_from_the_traced_values(model):
    n = len(model.mesh.elements("box"))
    assert float(jax.jit(lambda s: _solve(model, s))(2.0)) == 2.0 * n
    scales = jnp.asarray([1.0, 3.0, 5.0])
    np.testing.assert_allclose(jax.vmap(lambda s: _solve(model, s))(scales), scales * n)


def test_adjoint_tolerance_defaults_to_the_forward_tolerance():
    assert adjoint_tolerance(gfem.SolverOptions()) == 1e-6
    assert adjoint_tolerance(gfem.SolverOptions(tolerance=1e-9)) == 1e-9
    options = gfem.SolverOptions(tolerance=1e-9, adjoint_tolerance=1e-11)
    assert adjoint_tolerance(options) == 1e-11
