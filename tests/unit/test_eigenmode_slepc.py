"""eigen_solver="slepc": SLEPc finds the eigenmode seeds, as ARPACK does.

SLEPc's Krylov-Schur runs on the same shift-inverted seed operator as
ARPACK (`gfem._eigen.seed`): ``T(sigma)^-1`` through the rule's linear
solver, the divergence-free projection and, when the seed is damped, the
companion form. Every seed is then refined on the true operator, so the
modes, their Q and their gradients are ARPACK's to the refinement's
precision, on the eigen unit tests' designs (`eigen_cases`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eigen_cases import EPS, F1, conductor_line, line, line_mesh, terminated_line
from gradient_cases import MIRRORED_TARGET, mirrored_cavity
from petsc_recipe import EPS_VIEW, SLEPC_SUBSPACE

import gfem

pytestmark = pytest.mark.slepc


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return line_mesh(tmp_path_factory.mktemp("line"))


@pytest.fixture(scope="module")
def thick_mesh(tmp_path_factory):
    return line_mesh(tmp_path_factory.mktemp("thick"), (20, 1, 2))


@pytest.fixture(scope="module")
def cavity_mesh(tmp_path_factory):
    return mirrored_cavity(tmp_path_factory.mktemp("cavity"))


def _cavity(mesh):
    return gfem.Model(
        mesh,
        materials={"fill": gfem.Material(tan_delta=1e-3)},
        boundaries=[gfem.PEC("wall")],
    )


#: (mesh fixture, model builder, problem, SolverOptions keywords)
CASES = {
    "open-line": ("mesh", line, gfem.Eigenmode(1.6 * F1, n_modes=2), {}),
    "terminated": (
        "mesh",
        lambda m: terminated_line(m, 1000.0),
        gfem.Eigenmode(8e9, n_modes=2),
        {},
    ),
    "projection": (
        "mesh",
        lambda m: terminated_line(m, 1000.0),
        gfem.Eigenmode(1e9),
        {},
    ),
    "conductivity": (
        "mesh",
        lambda m: line(m, conductivity=0.05),
        gfem.Eigenmode(8e9),
        {},
    ),
    "conductor-below": (
        "thick_mesh",
        conductor_line,
        gfem.Eigenmode(0.75 * F1),
        {"eigen_tolerance": 1e-10},
    ),
    "conductor-above": (
        "thick_mesh",
        conductor_line,
        gfem.Eigenmode(1.3 * F1),
        {"eigen_tolerance": 1e-10},
    ),
    "degenerate": (
        "cavity_mesh",
        _cavity,
        gfem.Eigenmode(MIRRORED_TARGET, n_modes=3),
        {"eigen_tolerance": 1e-12},
    ),
}


def _both(model, problem, **options):
    """The solves with ``eigen_solver="arpack"`` and ``"slepc"``."""
    return [
        gfem.solve(
            model, problem, options=gfem.SolverOptions(eigen_solver=e, **options)
        )
        for e in ("arpack", "slepc")
    ]


def _assert_same_modes(arpack, slepc, tolerance, *, fields=True):
    """Same frequencies, Q and (``fields``: simple modes only) mode fields."""
    np.testing.assert_allclose(slepc.omega, arpack.omega, rtol=1e-9)
    np.testing.assert_allclose(slepc.q, arpack.q, rtol=1e-9)
    assert (np.asarray(slepc.error) < tolerance).all()
    # the Rayleigh functional makes omega's error second order in the
    # field's, so a field agrees to about the square root of omega's bar (the
    # target far below the mode gives 2e-8)
    for i in range(len(arpack.omega) if fields else 0):
        a, s = (np.asarray(r.e_field(i).coefficients) for r in (arpack, slepc))
        assert np.linalg.norm(s - a) <= 1e-6 * np.linalg.norm(a)


@pytest.mark.parametrize("case", list(CASES))
def test_slepc_seed_gives_the_arpack_modes(request, case):
    fixture, build, problem, options = CASES[case]
    model = build(request.getfixturevalue(fixture))
    arpack, slepc = _both(model, problem, linear_solver="direct", **options)
    tolerance = gfem.SolverOptions(**options).eigen_tolerance
    # a degenerate pair's fields are an arbitrary basis of its eigenspace
    _assert_same_modes(arpack, slepc, tolerance, fields=case != "degenerate")
    if case == "degenerate":
        np.testing.assert_array_equal(np.asarray(slepc.cluster), [0, 0, 1])


@pytest.mark.parametrize(
    "linear_solver",
    ["direct", "bddc_gmres", pytest.param("petsc", marks=pytest.mark.petsc)],
)
def test_slepc_seed_runs_on_each_linear_solver(mesh, linear_solver):
    model = terminated_line(mesh, 1000.0)
    arpack, slepc = _both(model, gfem.Eigenmode(8e9), linear_solver=linear_solver)
    _assert_same_modes(arpack, slepc, gfem.SolverOptions().eigen_tolerance)


@pytest.fixture
def solved_eps(monkeypatch):
    """What each SLEPc EPS gfem solves looks like right after its solve.

    One dict per ``EPS.solve``: the ST's type, ``(nev, ncv, mpd)``, and the
    number of linear solves (`EigenOperator.solve`) the solve made.
    """
    from gfem import _petsc
    from gfem._eigen.operator import EigenOperator

    SLEPc = _petsc.require_slepc()
    solves = [0]
    solve = EigenOperator.solve

    def counted(self, b):
        solves[0] += 1
        return solve(self, b)

    seen = []

    class EPS(SLEPc.EPS):
        def solve(self):
            before = solves[0]
            super().solve()
            seen.append(
                {
                    "st": self.getST().getType(),
                    "dimensions": self.getDimensions(),
                    "solves": solves[0] - before,
                }
            )

    monkeypatch.setattr(EigenOperator, "solve", counted)
    monkeypatch.setattr(SLEPc, "EPS", EPS)
    return seen


def test_slepc_seed_solves_through_the_rule_handle(mesh, solved_eps):
    options = gfem.SolverOptions(eigen_solver="slepc")
    gfem.solve(terminated_line(mesh, 1000.0), gfem.Eigenmode(8e9), options=options)
    assert solved_eps
    for eps in solved_eps:
        assert eps["solves"] > 0  # T(sigma)^-1 through the rule's handle
        assert eps["st"] == "shift"  # no KSP of SLEPc's own


def test_eps_ncv_reaches_the_eps(mesh, solved_eps, capfd):
    # the recipe's options: a larger subspace, and SLEPc's monitor
    options = gfem.SolverOptions(eigen_solver="slepc", petsc_options=SLEPC_SUBSPACE)
    capfd.readouterr()
    gfem.solve(terminated_line(mesh, 1000.0), gfem.Eigenmode(8e9), options=options)
    assert [eps["dimensions"][1] for eps in solved_eps] == [40]
    assert "EPS nconv=" in capfd.readouterr().out


class ShiftSolveError(Exception):
    """Raised by a shift solve."""


def _gfem_keys() -> list[str]:
    """The names in PETSc's options database under a gfem prefix."""
    from gfem import _petsc

    database = _petsc.require_petsc().Options().getAll()
    return [name for name in database if name.startswith("gfem")]


# with a misspelled option too: the shift solve's error, not the typo's
@pytest.mark.parametrize("petsc_options", [{}, {"eps_mpdd": 20}])
def test_a_failed_shift_solve_raises_its_own_error(mesh, monkeypatch, petsc_options):
    from gfem._eigen.operator import EigenOperator

    solve = EigenOperator.solve
    calls = [0]

    def failing(self, b):
        calls[0] += 1
        if calls[0] == 5:
            raise ShiftSolveError("the shift solve failed")
        return solve(self, b)

    monkeypatch.setattr(EigenOperator, "solve", failing)
    options = gfem.SolverOptions(eigen_solver="slepc", petsc_options=petsc_options)
    with pytest.raises(ShiftSolveError, match="the shift solve failed"):
        gfem.solve(terminated_line(mesh, 1000.0), gfem.Eigenmode(8e9), options=options)
    assert not _gfem_keys()


def test_the_options_database_holds_no_gfem_key_after_a_solve(mesh, solved_eps):
    options = gfem.SolverOptions(eigen_solver="slepc", petsc_options={"eps_mpd": 20})
    gfem.solve(terminated_line(mesh, 1000.0), gfem.Eigenmode(8e9), options=options)
    assert [eps["dimensions"][2] for eps in solved_eps] == [20]  # it was read
    assert not _gfem_keys()


def test_eps_view_prints_the_eps(mesh, capfd):
    # SLEPc reads its viewer options at the end of the solve
    options = gfem.SolverOptions(eigen_solver="slepc", petsc_options=EPS_VIEW)
    capfd.readouterr()
    gfem.solve(terminated_line(mesh, 1000.0), gfem.Eigenmode(8e9), options=options)
    out = capfd.readouterr().out
    assert "EPS Object" in out
    assert "krylovschur" in out


def test_a_misspelled_slepc_option_raises_naming_it(mesh):
    options = gfem.SolverOptions(eigen_solver="slepc", petsc_options={"eps_mpdd": 20})
    with pytest.raises(ValueError, match=r"petsc_options\['eps_mpdd'\]") as error:
        gfem.solve(terminated_line(mesh, 1000.0), gfem.Eigenmode(8e9), options=options)
    assert "SLEPc's eigen solver (eps_type krylovschur)" in str(error.value)
    assert not _gfem_keys()


def test_a_slepc_seeded_gradient_matches_arpack(mesh):
    eps = jnp.full(len(mesh.elements("dielectric")), EPS)
    ports = [
        gfem.LumpedPort("p1", "port1", "+z", impedance=1000.0),
        gfem.LumpedPort("p2", "port2", "+z", impedance=1000.0),
    ]

    def frequency(eps_r, eigen_solver):
        model = gfem.Model(
            mesh,
            materials={"dielectric": gfem.MaterialField(eps_r=eps_r)},
            boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
            ports=ports,
        )
        options = gfem.SolverOptions(eigen_solver=eigen_solver)
        return jnp.real(
            gfem.solve(model, gfem.Eigenmode(8e9), options=options).omega[0]
        )

    arpack = jax.grad(frequency)(eps, "arpack")
    slepc = jax.grad(frequency)(eps, "slepc")
    assert np.abs(np.asarray(arpack)).max() > 0
    np.testing.assert_allclose(slepc, arpack, rtol=1e-9)
