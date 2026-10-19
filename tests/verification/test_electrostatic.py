"""Electrostatic capacitance against analytic values.

A parallel-plate stack has a potential linear in z, which every H1 space
holds exactly: gfem must return eps A / d to round-off at any mesh size and
order. A coaxial line has a logarithmic potential; on straight-sided meshes
with nodes on the circles its capacitance converges at second order in h
(the polygonal conductors cost O(h^2) whatever the order), and on one mesh
the discretization error falls fast with p to that geometric floor.
"""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants
import structured_meshes as sm

import gfem
from gfem._linalg import KINDS

EPS0 = scipy.constants.epsilon_0
UM = 1e-6


def _plate_model(tmp_path, layers, n, materials):
    path = sm.write_msh22(tmp_path / "plate.msh", sm.plate(40.0, 30.0, layers, n))
    return gfem.Model(gfem.Mesh.from_gmsh(path), materials=materials)


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("n", [(1, 1, 1), (3, 2, 2)])
def test_parallel_plate_is_exact_at_every_mesh_size_and_order(tmp_path, n, order):
    model = _plate_model(
        tmp_path, (("oxide", 0.5),), n, {"oxide": gfem.Material(eps_r=3.9)}
    )
    cap = gfem.solve(
        model,
        gfem.Electrostatic(terminals={"plate": "top"}, ground="bottom"),
        options=gfem.SolverOptions(order=order, linear_solver="direct"),
    )
    expected = EPS0 * 3.9 * (40 * UM) * (30 * UM) / (0.5 * UM)
    assert cap.terminals == ("plate",)
    assert cap.capacitance.shape == (1, 1)
    assert float(cap.capacitance[0, 0]) == pytest.approx(expected, rel=1e-11)


def _plate_c(model, order=2):
    cap = gfem.solve(
        model,
        gfem.Electrostatic(terminals={"plate": "top"}, ground="bottom"),
        options=gfem.SolverOptions(order=order),
    )
    return float(cap.capacitance[0, 0])


def test_two_dielectric_layers_are_capacitors_in_series(tmp_path):
    layers = (("oxide", 0.5), ("nitride", 0.25))
    model = _plate_model(
        tmp_path,
        layers,
        (2, 2, 2),
        {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)},
    )
    area = (40 * UM) * (30 * UM)
    expected = EPS0 * area / (0.5 * UM / 3.9 + 0.25 * UM / 7.5)
    assert _plate_c(model) == pytest.approx(expected, rel=1e-11)


def test_lossy_material_field_reads_the_real_permittivity(tmp_path):
    model = _plate_model(
        tmp_path, (("oxide", 0.5),), (2, 2, 2), {"oxide": gfem.Material(eps_r=3.9)}
    )
    n = len(model.mesh.elements("oxide"))
    field = gfem.MaterialField(eps_r=np.full(n, 3.9) * (1 - 0.01j))
    lossy = model.with_material("oxide", field)
    assert _plate_c(lossy) == pytest.approx(_plate_c(model), rel=1e-13)


def test_anisotropic_permittivity_reads_the_component_across_the_plates(tmp_path):
    rotated = gfem.Material(
        eps_r=(2.0, 5.0, 9.0), axes=((0, 0, 1), (1, 0, 0), (0, 1, 0))
    )
    model = _plate_model(tmp_path, (("crystal", 0.5),), (2, 2, 2), {"crystal": rotated})
    expected = EPS0 * 2.0 * (40 * UM) * (30 * UM) / (0.5 * UM)
    assert _plate_c(model) == pytest.approx(expected, rel=1e-11)


def test_two_terminal_stack_gives_the_maxwell_matrix_of_two_capacitors(tmp_path):
    # ground at z = 0, terminal "mid" at the layer interface, "top" on top
    layers = (("oxide", 0.5), ("nitride", 0.25))
    model = _plate_model(
        tmp_path,
        layers,
        (2, 1, 2),
        {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)},
    )
    cap = gfem.solve(
        model,
        gfem.Electrostatic(
            terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
        ),
    )
    area = (40 * UM) * (30 * UM)
    c1, c2 = EPS0 * 3.9 * area / (0.5 * UM), EPS0 * 7.5 * area / (0.25 * UM)
    expected = np.array([[c2, -c2], [-c2, c1 + c2]])
    assert cap.terminals == ("top", "mid")
    np.testing.assert_allclose(cap.capacitance, expected, rtol=1e-11)


# --- coaxial line ----------------------------------------------------------------

A, B, LENGTH = 1.0, 2.0, 0.5  # um
# a quarter of the line: symmetry planes and end faces carry no charge
COAX = math.pi * EPS0 * LENGTH * UM / (2 * math.log(B / A))


def _coax_error(tmp_path, k, order):
    """Relative error on a mesh with 2k radial and 4k angular cells."""
    mesh = sm.coax_quarter(A, B, LENGTH, n_r=2 * k, n_theta=4 * k)
    path = sm.write_msh22(tmp_path / f"coax{k}.msh", mesh)
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"dielectric": gfem.Material()}
    )
    cap = gfem.solve(
        model,
        gfem.Electrostatic(terminals={"inner": "inner"}, ground="outer"),
        options=gfem.SolverOptions(order=order),
    )
    return abs(float(cap.capacitance[0, 0]) - COAX) / COAX


def _rates(errors):
    return [math.log2(e0 / e1) for e0, e1 in itertools.pairwise(errors)]


def test_coax_order_1_converges_at_second_order_in_h(tmp_path):
    # energy error h^(2p): the capacitance is the energy
    errors = [_coax_error(tmp_path, k, order=1) for k in (1, 2, 4, 8)]
    assert errors[-1] < 1e-3
    assert _rates(errors)[-2:] == pytest.approx([2.0, 2.0], abs=0.1)


@pytest.mark.parametrize("order", [2, 3])
def test_coax_higher_orders_converge_at_the_straight_sided_geometry_rate(
    tmp_path, order
):
    # The straight-sided conductors, not the order, bound the rate for
    # p >= 2; with the nodes on the circles it measures 3.0 here.
    errors = [_coax_error(tmp_path, k, order) for k in (1, 2, 4, 8)]
    assert errors[-1] < 2e-5
    assert min(_rates(errors)) > 2.8


def test_coax_error_falls_with_order_on_one_mesh(tmp_path):
    errors = [_coax_error(tmp_path, 4, order) for order in (1, 2, 3)]
    assert errors[0] > 10 * errors[1] > errors[2]


# --- gradients w.r.t. the permittivity -------------------------------------------

SOLVERS = [
    "direct",
    pytest.param(
        "bddc_gmres",
        marks=pytest.mark.skipif(
            "bddc_gmres" not in KINDS, reason="the bddc backend is not available yet"
        ),
    ),
]


def _gradient_case(tmp_path, solver):
    """J(eps) = <W, C(eps)> on a two-terminal stack with random P0 fields."""
    layers = (("oxide", 0.5), ("nitride", 0.25))
    path = sm.write_msh22(
        tmp_path / "grad.msh", sm.plate(40.0, 30.0, layers, (3, 2, 2))
    )
    mesh = gfem.Mesh.from_gmsh(path)
    model = gfem.Model(
        mesh, materials={"oxide": gfem.Material(), "nitride": gfem.Material()}
    )
    problem = gfem.Electrostatic(
        terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
    )
    options = gfem.SolverOptions(order=2, linear_solver=solver, tolerance=1e-12)
    weights = np.array([[1.0, 0.3], [-0.7, 2.0]])  # not symmetric on purpose
    n_ox = len(mesh.elements("oxide"))

    def objective(eps):
        m = model.replace(
            materials={
                "oxide": gfem.MaterialField(eps_r=eps[:n_ox]),
                "nitride": gfem.MaterialField(eps_r=eps[n_ox:]),
            }
        )
        c = gfem.solve(m, problem, options=options).capacitance
        return jnp.sum(weights * c) / (EPS0 * UM)  # O(1) values

    rng = np.random.default_rng(42)
    eps = jnp.asarray(rng.uniform(2.0, 9.0, len(mesh.tetrahedra)))
    return objective, eps, rng


@pytest.mark.parametrize("solver", SOLVERS)
def test_capacitance_gradient_matches_central_differences(tmp_path, solver):
    objective, eps, rng = _gradient_case(tmp_path, solver)
    grad = jax.grad(objective)(eps)
    h = 1e-4
    for _ in range(3):
        d = rng.standard_normal(eps.shape)
        d /= np.linalg.norm(d)
        fd = (objective(eps + h * d) - objective(eps - h * d)) / (2 * h)
        assert float(grad @ d) == pytest.approx(float(fd), rel=1e-6)


@pytest.mark.parametrize("solver", SOLVERS)
def test_capacitance_taylor_remainder_is_second_order(tmp_path, solver):
    objective, eps, rng = _gradient_case(tmp_path, solver)
    j0, grad = jax.value_and_grad(objective)(eps)
    d = rng.standard_normal(eps.shape)
    hs = np.array([1e-2, 1e-3, 1e-4])
    remainders = [abs(float(objective(eps + h * d) - j0 - h * (grad @ d))) for h in hs]
    slopes = np.diff(np.log(remainders)) / np.diff(np.log(hs))
    assert slopes == pytest.approx([2.0, 2.0], abs=0.1)


def test_complex_permittivity_gradient_is_the_real_part_derivative(tmp_path):
    # Electrostatics reads Re(eps_r): the cotangent of a complex leaf is
    # dJ/dRe(eps) (JAX's convention: dJ/dx - i dJ/dy, and dJ/dy = 0).
    objective, eps, rng = _gradient_case(tmp_path, "direct")
    lossy = eps * (1 - 0.05j)
    grad = jax.grad(objective)(lossy)
    assert grad.dtype == jnp.complex128
    assert not np.asarray(grad.imag).any()
    d = rng.standard_normal(eps.shape)
    h = 1e-4
    fd = (objective(lossy + h * d) - objective(lossy - h * d)) / (2 * h)
    assert float(grad.real @ d) == pytest.approx(float(fd), rel=1e-6)
