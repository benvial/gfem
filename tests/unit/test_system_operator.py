"""The system operator of a driven solve, at its interface.

Each test builds a `SystemOperator` of a parallel-plate line's driven
operator, without wave ports and without a rule: its adjoint handles, its
pieces and its sweeps' base handles.
"""

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg
import structured_meshes as sm
from solver_misses import cannot_converge_at

import gfem
from gfem._engine import driven
from gfem._engine.system_operator import SystemOperator, SystemSweep
from gfem._linalg import ResolvedLinearSolver, prepare
from gfem._linalg.bddc import BddcHandle

OMEGAS = (2 * math.pi * 4e9, 2 * math.pi * 5e9)
ADJOINT_TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1)))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _system(line, kind, width=2):
    linear = ResolvedLinearSolver.resolve(
        gfem.SolverOptions(linear_solver=kind), default="direct", problem="driven"
    )
    op = driven.driven_operator(line, 1)
    system = SystemOperator(
        op, (), linear, width=width, adjoint_tolerance=ADJOINT_TOLERANCE
    )
    return system, linear


def _loads(system, width=2):
    rng = np.random.default_rng(0)
    shape = (system.operator.space.ndof, width)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _piece_values(system):
    form = system.operator.form(1.0)
    form.Assemble()
    return form.mat.AsVector().FV().NumPy().nbytes


def test_cpu_adjoint_solves_keep_one_handle_per_frequency(line, monkeypatch):
    system, linear = _system(line, "direct")
    loads = _loads(system)
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return prepare(*args, **kwargs)

    monkeypatch.setattr("gfem._linalg.prepare", spy)
    assert system.nbytes == 0
    x0 = system.solve_adjoint(0, OMEGAS[0], loads)
    held = system.nbytes
    assert held > 0
    again = system.solve_adjoint(0, OMEGAS[0], loads)
    assert system.nbytes == held
    x1 = system.solve_adjoint(1, OMEGAS[1], loads)
    assert system.nbytes > held
    assert len(calls) == 2
    monkeypatch.undo()
    for x, omega in ((x0, OMEGAS[0]), (again, OMEGAS[0]), (x1, OMEGAS[1])):
        fresh = linear.prepare(
            system.operator_at(omega),
            symmetry="complex_symmetric",
            width=2,
            tolerance=ADJOINT_TOLERANCE,
        )
        expected = fresh.solve(loads)
        fresh.free()
        np.testing.assert_allclose(x, expected, rtol=1e-12, atol=0)
    system.free()
    assert system.nbytes == 0  # no wave port: no W vectors


@pytest.mark.parametrize(
    "kind", ["direct", "bddc_gmres", pytest.param("petsc", marks=pytest.mark.petsc)]
)
def test_pieces_are_held_only_for_the_piece_sum_linear_solvers(line, kind):
    system, _ = _system(line, kind)
    loads = _loads(system)
    with system.sweep() as sweep:
        x, _ = sweep.solve(OMEGAS[0], loads)
    assert np.abs(x).max() > 0
    if kind == "bddc_gmres":  # BDDC assembles the form
        assert system.nbytes == 0
    else:
        # at least the values of the curl-curl and mass pieces
        assert system.nbytes >= 2 * _piece_values(system)
    system.free()
    assert system.nbytes == 0


def test_a_sweep_frees_its_base_when_it_raises(line, monkeypatch):
    system, _ = _system(line, "bddc_gmres")
    loads = _loads(system)
    cannot_converge_at(monkeypatch, (OMEGAS[1] / (2 * math.pi),))
    bases = []
    at = SystemSweep.at

    def spy(self, omega):
        handle = at(self, omega)
        bases.append(self._base)
        return handle

    monkeypatch.setattr(SystemSweep, "at", spy)
    freed = []
    free = BddcHandle.free

    def spy_free(self):
        freed.append(self)
        return free(self)

    monkeypatch.setattr(BddcHandle, "free", spy_free)

    def sweep():
        with system.sweep() as swept:
            x, _ = swept.solve(OMEGAS[0], loads)
            assert not freed
            swept.solve(OMEGAS[1], loads, x0=x)  # raises

    with pytest.raises(gfem.LinearSolverConvergenceError):
        sweep()
    assert any(h is bases[-1] for h in freed)


def test_corrections_are_block_diagonal_over_wave_port_surfaces(line):
    linear = ResolvedLinearSolver.resolve(
        gfem.SolverOptions(), default="direct", problem="driven"
    )
    op = driven.driven_operator(line, 1)
    rng = np.random.default_rng(1)
    ndof = op.space.ndof
    pairs = (
        (rng.standard_normal((ndof, 6)), rng.standard_normal((6, 4))),  # n = 2
        (rng.standard_normal((ndof, 4)), rng.standard_normal((4, 2))),  # n = 1
    )
    blocks = [rng.standard_normal((4, 4)), rng.standard_normal((2, 2))]
    surfaces = [
        SimpleNamespace(
            piece=dataclasses.replace(op.pieces[0], name=f"waveport s{k}"),
            vectors=v,
            correction=lambda _, p=p, d=d: (p, d),
        )
        for k, ((v, p), d) in enumerate(zip(pairs, blocks, strict=True))
    ]
    system = SystemOperator(
        op, surfaces, linear, width=1, adjoint_tolerance=ADJOINT_TOLERANCE
    )
    u, c = system.corrections(OMEGAS[0])
    np.testing.assert_array_equal(u, np.concatenate([v @ p for v, p in pairs], 1))
    np.testing.assert_array_equal(c, scipy.linalg.block_diag(*blocks))
