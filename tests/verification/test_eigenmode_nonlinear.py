"""The nonlinear eigenmode refinement against independent references.

A 20 x 10 x 20 mm cavity, PEC except its top wall (z = 20 mm), a thick
`Conductor` of 1e5 S/m: the operator ``K - k^2 M + i k eta0 / Z_s(omega) B``
with ``Z_s = (1 + i) sqrt(omega mu0 / (2 sigma))`` is not a polynomial in
the frequency. Its TE101 mode (10.599 GHz) is found from targets 25 %
below, 6 % below and 23 % above the mode.

- The refined root equals the root of the same discrete operator found
  here by a Rayleigh-functional iteration that refactorizes ``T(omega)``
  at every step (assembled independently with NGSolve and solved with
  SciPy's LU): to 1e-12 relative, from every target. The seed alone, frozen
  at the target, would have a Q off by +15 %, +3 % and -10 %.
- Its Q matches first-order perturbation theory for the lossy wall,
  ``Q = omega W / P``, to 0.5 %.
"""

import cmath
import math

import ngsolve as ngs
import numpy as np
import pytest
import scipy.constants as const
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import structured_meshes as sm

import gfem
from gfem._engine.mesh import to_ngsolve

A, B, D = 20.0, 10.0, 20.0  # mm
SCALE = 1e-3
SIGMA = 1e5
CELLS = (6, 3, 6)
ETA0 = const.mu_0 * const.c
OMEGA_101 = const.c * math.pi * math.hypot(1 / (A * SCALE), 1 / (D * SCALE))


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    edges = tuple(
        np.linspace(0, s, n + 1) for s, n in zip((A, B, D), CELLS, strict=True)
    )

    def surface(c):
        if abs(c[2] - D) < 1e-9:
            return "lossy"
        on = (abs(c[0]), abs(c[0] - A), abs(c[1]), abs(c[1] - B), abs(c[2]))
        return "pec" if min(on) < 1e-9 else ""

    path = tmp_path_factory.mktemp("cavity") / "cavity.msh"
    sm.write_msh22(path, sm.structured(edges, lambda _c: "air", surface))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _inverse_impedance(omega):
    return 1 / ((1 + 1j) * cmath.sqrt(omega * const.mu_0 / (2 * SIGMA)))


def _reference_root(mesh) -> complex:
    """The TE101 root of the discrete operator, refactorizing at every step."""
    built = to_ngsolve(mesh)
    space = ngs.HCurl(built.mesh, order=2, type1=True, dirichlet="pec", complex=True)
    u, v = space.TnT()

    def assemble(integrand):
        form = ngs.BilinearForm(space)
        form += integrand
        form.Assemble()
        rows, cols, values = form.mat.COO()
        shape = (space.ndof, space.ndof)
        return sp.csr_matrix(
            (np.asarray(values), (np.asarray(rows), np.asarray(cols))), shape
        )

    free = np.flatnonzero(np.array(list(space.FreeDofs()), dtype=bool))
    stiffness, mass, wall = (
        assemble(f)[free][:, free].tocsc()
        for f in (
            ngs.curl(u) * ngs.curl(v) * ngs.dx,
            u * v * ngs.dx,
            u.Trace() * v.Trace() * ngs.ds("lossy"),
        )
    )

    def k_of(omega):
        return omega * SCALE / const.c

    def operator(omega):
        k = k_of(omega)
        g = 1j * k * ETA0 * _inverse_impedance(omega)
        return stiffness - k * k * mass + g * wall

    def derivative(omega):  # dT/domega
        k, dk = k_of(omega), SCALE / const.c
        g = 1j * k * ETA0 * _inverse_impedance(omega)
        return -2 * k * dk * mass + 0.5 * g / omega * wall

    def rayleigh(omega, x):
        for _ in range(30):
            step = (x @ (operator(omega) @ x)) / (x @ (derivative(omega) @ x))
            omega -= step
            if abs(step) < 1e-16 * abs(omega):
                break
        return omega

    # the start: TE101 of the cavity with a (penalized) PEC top wall
    k2, vectors = spla.eigsh(
        (stiffness + 1e6 * wall).real, k=1, M=mass.real, sigma=k_of(OMEGA_101) ** 2
    )
    omega = math.sqrt(k2[0]) * const.c / SCALE
    x = vectors[:, 0].astype(complex)
    for _ in range(10):
        omega = rayleigh(omega, x)
        x = spla.splu(operator(omega)).solve(derivative(omega) @ x)
        x /= np.linalg.norm(x)
    return rayleigh(omega, x)


def _perturbation_q() -> float:
    """TE101 with one lossy wall at z = d: Q = omega W / P."""
    a, b, d = A * SCALE, B * SCALE, D * SCALE
    w = OMEGA_101
    rs = math.sqrt(w * const.mu_0 / (2 * SIGMA))
    energy = 2 * (const.epsilon_0 / 4) * (a * b * d / 4)  # E0 = 1
    h_wall = math.pi / d / (w * const.mu_0)  # |H_x| on z = d
    loss = rs / 2 * h_wall**2 * (a / 2) * b
    return w * energy / loss


@pytest.fixture(scope="module")
def modes(cavity):
    model = gfem.Model(
        cavity,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("pec"), gfem.Conductor("lossy", SIGMA)],
    )
    options = gfem.SolverOptions(eigen_tolerance=1e-12)
    return [
        gfem.solve(model, gfem.Eigenmode(target=f), options=options)
        for f in (8e9, 10e9, 13e9)
    ]


def test_refined_root_matches_a_refactorizing_reference(cavity, modes):
    exact = _reference_root(cavity)
    assert abs(exact.real / (2 * math.pi) - 10.6e9) < 0.05e9
    for result in modes:
        omega = complex(result.omega[0])
        assert abs(omega - exact) <= 1e-12 * abs(exact)


def test_refined_q_matches_perturbation_theory(modes):
    q = _perturbation_q()
    assert q == pytest.approx(2587.5, rel=1e-4)
    for result in modes:
        assert float(result.q[0]) == pytest.approx(q, rel=5e-3)
