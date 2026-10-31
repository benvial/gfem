"""The engine sensitivity kernel: field pairs contracted per element."""

import numpy as np
import pytest
import scipy.constants
import structured_meshes as sm

import gfem
from gfem._engine.domain import em_region
from gfem._engine.electrostatic import ElectrostaticRule
from gfem._engine.sensitivity import PROPERTIES, FieldPair, sensitivity, unsupported

KIND = ("material", "oxide", "eps_r")


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    path = tmp_path_factory.mktemp("mesh") / "stack.msh"
    layers = (("oxide", 0.5), ("nitride", 0.25))
    sm.write_msh22(path, sm.plate(4.0, 3.0, layers, (2, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path)
    eps = np.random.default_rng(3).uniform(2.0, 6.0, len(mesh.elements("oxide")))
    model = gfem.Model(
        mesh,
        materials={
            "oxide": gfem.MaterialField(eps_r=eps),
            "nitride": gfem.Material(eps_r=7.5),
        },
    )
    problem = gfem.Electrostatic(terminals={"mid": "oxide__nitride"}, ground="bottom")
    rule = ElectrostaticRule(model, problem, gfem.SolverOptions(order=2))
    outputs, entry = rule.forward(model, keep=True)
    space, built = rule._space(model)
    # phi^T K phi in m (K assembled in file units, um)
    energy = outputs["capacitance"][0, 0] / scipy.constants.epsilon_0
    yield model, entry, space, em_region(model, built), eps, energy, rule
    entry.free()


def _pair(solved, left, right, weight=1.0):
    _, _, space, domain, _, _, _ = solved
    return FieldPair(space, left, right, {"grad_grad": weight}, 2, domain)


def _grads(solved, pairs):
    model, rule = solved[0], solved[-1]
    return sensitivity(
        model,
        pairs,
        {KIND},
        form="electrostatic",
        terms=lambda nodes: rule.terms(model, nodes=nodes),
    )


def test_element_contractions_weighted_by_eps_rebuild_the_energy(solved):
    # phi^T K phi = sum_e eps_e int_e |grad phi|^2 (oxide only: nitride is 0)
    _, entry, _, _, eps, energy, _ = solved
    phi = entry.fields["potential"][:, 0]
    grads = _grads(solved, [_pair(solved, phi, phi)])
    assert grads[KIND].shape == eps.shape
    assert float(grads[KIND] @ eps) == pytest.approx(energy, rel=1e-12)


def test_pairs_are_bilinear_in_complex_columns_on_a_real_space(solved):
    _, entry, _, _, _, _, _ = solved
    phi = entry.fields["potential"][:, 0]
    rng = np.random.default_rng(0)
    other = rng.standard_normal(phi.shape)
    left = phi + 2j * other
    right = 3 * phi - 1j * other
    got = _grads(solved, [_pair(solved, left, right)])

    def real(a, b):
        return _grads(solved, [_pair(solved, a, b)])[KIND]

    # electrostatics reads Re(eps_r): the cotangent is the real part
    expected = 3 * real(phi, phi) + 2 * real(other, other)
    np.testing.assert_allclose(got[KIND], expected, rtol=1e-12, atol=0)


def test_electrostatic_term_derivatives_are_the_differentiable_properties(solved):
    model, rule = solved[0], solved[-1]
    (term,) = rule.terms(model, nodes=False)
    assert set(term.derivatives) == PROPERTIES["electrostatic"] == {"eps_r"}


def test_unsupported_inputs_are_named():
    assert unsupported("electrostatic", KIND) is None
    assert unsupported("electrostatic", ("nodes",)) is None
    assert unsupported("driven", ("nodes",)) is None
    assert unsupported("eigenmode", ("nodes",)) is None
    assert "shape" in unsupported("port_mode", ("nodes",))
    assert "conductivity" in unsupported(
        "electrostatic", ("material", "oxide", "conductivity")
    )
