"""The pieces of eigenmode gradients: T'(k), clusters, the error type.

The eigenvalue derivative ``dk/dp = -x^T (dT/dp) x / x^T T'(k) x`` holds
at a complex eigenvalue, so ``T'(k)`` must be the derivative of the
analytic continuation of ``T``: for the polynomial part ``D + 2 k M`` and
for each nonpolynomial term ``θ_p'(k) A_p`` (a conductor's surface
impedance, the second-order absorbing wall). Checked against complex-step
(imaginary-axis) central differences of ``T(k) x`` at a complex ``k``,
which only an analytic ``T`` matches.
"""

import math

import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm

import gfem
from gfem._eigen.rule import EigenmodeRule, clusters
from gfem._engine import terms
from gfem._engine.driven import driven_operator
from gfem._engine.runtime import task_manager

L, W, H = 10.0, 2.0, 1.0  # mm
F = 7.5e9


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(L, W, H, (6, 1, 1)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _boundaries(kind):
    plates = [gfem.PEC("bottom")]
    if kind == "conductor":
        return [*plates, gfem.Conductor("top", 1e5)]
    if kind == "thin conductor":
        return [*plates, gfem.Conductor("top", 1e5, 2e-6)]
    if kind == "absorbing order 2":
        return [*plates, gfem.PEC("top"), gfem.Absorbing("port1", order=2)]
    return [*plates, gfem.PEC("top")]  # damped by the lumped resistor


@pytest.mark.parametrize(
    "kind", ["conductor", "thin conductor", "absorbing order 2", "lumped"]
)
def test_t_prime_is_the_complex_step_derivative_of_t(mesh, kind):
    model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0, tan_delta=1e-3)},
        boundaries=_boundaries(kind),
        ports=[gfem.LumpedPort("p1", "port2", "+z", impedance=50.0)],
    )
    problem = gfem.Eigenmode(target=F)
    rule = EigenmodeRule(model, problem, gfem.SolverOptions(order=1))
    with task_manager(1):
        op = driven_operator(model, 1)
        sigma = terms.wavenumber(2 * math.pi * F, op.built.scale)
        eop = rule.operator(op, sigma)
        rng = np.random.default_rng(0)
        n = len(eop.free)
        x = np.where(eop.free, rng.standard_normal(n) + 1j * rng.standard_normal(n), 0)
        k = sigma * (1.1 + 0.05j)  # a decaying mode's complex wavenumber
        h = 1e-4 * abs(k)
        step = (eop.apply(k + 1j * h, x) - eop.apply(k - 1j * h, x)) / (2j * h)
        exact = eop.derivative(k, x)
        eop.free_handle()
    assert np.linalg.norm(exact - step) <= 1e-7 * np.linalg.norm(exact)
    # and the scalar the gradient divides by
    np.testing.assert_allclose(x @ exact, x @ step, rtol=1e-7)


def test_nonlinear_terms_enter_t_prime(mesh):
    # T'(k) of a conductor plate differs from the polynomial part alone
    model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=_boundaries("conductor"),
    )
    rule = EigenmodeRule(model, gfem.Eigenmode(target=F), gfem.SolverOptions(order=1))
    with task_manager(1):
        op = driven_operator(model, 1)
        k = terms.wavenumber(2 * math.pi * F, op.built.scale)
        eop = rule.operator(op, k)
        x = np.where(eop.free, 1.0 + 0j, 0)
        full = x @ eop.derivative(k, x)
        polynomial = 2 * k * (x @ eop.seed_mass(x))
    assert abs(full - polynomial) > 1e-6 * abs(polynomial)


# --- cluster labels ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ks", "labels"),
    [
        ([1.0, 2.0, 3.0], [0, 1, 2]),
        ([1.0, 1.0 + 1e-9, 2.0], [0, 0, 1]),
        ([2.0, 1.0, 2.0 * (1 + 5e-7)], [0, 1, 0]),
        # chains: each neighbour within rtol joins the cluster
        ([1.0, 1.0 + 8e-7, 1.0 + 1.6e-6, 3.0], [0, 0, 0, 1]),
        ([1.0 + 1e-3j, 1.0 + 1e-3j + 1e-9], [0, 0]),
        ([1.0, 1.0 + 1e-5], [0, 1]),
    ],
)
def test_modes_within_cluster_rtol_share_a_label(ks, labels):
    np.testing.assert_array_equal(clusters(ks, 1e-6), labels)


def test_simple_modes_get_labels_of_their_own(mesh):
    model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=_boundaries("lumped"),
    )
    f1 = const.c / (2 * L * 1e-3 * 2.0)
    result = gfem.solve(model, gfem.Eigenmode(target=1.5 * f1, n_modes=2))
    np.testing.assert_array_equal(np.asarray(result.cluster), [0, 1])
    assert np.asarray(result.cluster).dtype == np.int64


def test_result_without_labels_has_simple_modes():
    # results written before cluster labels existed: every mode simple
    result = gfem.EigenmodeResult(omega=np.array([1.0, 2.0]), error=np.zeros(2))
    np.testing.assert_array_equal(np.asarray(result.cluster), [0, 1])


def test_degenerate_mode_error_is_public():
    assert issubclass(gfem.DegenerateModeError, ValueError)
    assert "sum" in gfem.DegenerateModeError.__doc__
