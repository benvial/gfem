"""Periodic cells on PETSc and SLEPc.

- Plain periodicity keeps the driven operator complex symmetric, so
  ``linear_solver="petsc"`` (MUMPS LDL^T) solves the C1 line two-port as
  ``direct`` does.
- SLEPc seeds the plain Bragg line C3 at Γ: its bands are the
  transfer-matrix roots.
- A Bloch wave vector makes the operator general (A^T != A), which PETSc's
  LDL^T cannot factor: the driven and eigenmode rules refuse it before any
  assembly, and so does the PETSc handle itself.
"""

import math

import numpy as np
import periodic_cases as pc
import pytest

import gfem


@pytest.fixture(scope="module")
def c1_mesh(tmp_path_factory):
    return pc.c1_mesh(tmp_path_factory.mktemp("c1"))


@pytest.fixture(scope="module")
def c3_mesh(tmp_path_factory):
    return pc.c3_mesh(tmp_path_factory.mktemp("c3"))


@pytest.mark.petsc
def test_plain_line_s_on_petsc_matches_direct(c1_mesh):
    model = pc.c1(c1_mesh)
    problem = gfem.Driven(pc.C1_FREQUENCY)
    s = {
        solver: np.asarray(
            gfem.solve(
                model,
                problem,
                options=gfem.SolverOptions(order=pc.C1_ORDER, linear_solver=solver),
            ).s.values
        )
        for solver in ("direct", "petsc")
    }
    np.testing.assert_allclose(s["petsc"], s["direct"], rtol=0, atol=1e-8)


@pytest.mark.slepc
@pytest.mark.parametrize("tan_delta", [0.0, pc.C3_TAN_DELTA])
def test_slepc_plain_bragg_line_bands_match_the_transfer_matrix(c3_mesh, tan_delta):
    problem = gfem.Eigenmode(target=pc.C3_PLAIN_TARGET, n_modes=2)
    options = gfem.SolverOptions(order=pc.C3_ORDER, eigen_solver="slepc")
    result = gfem.solve(pc.c3(c3_mesh, tan_delta), problem, options=options)
    f = np.asarray(result.omega) / (2 * math.pi)
    assert len(f) == 2
    for value in f:
        exact = pc.tmm_frequency(0.0, complex(value), tan_delta)
        assert abs(value - exact) <= pc.C3_FREQ_RTOL * abs(exact)
    if tan_delta == 0:
        assert not np.asarray(result.omega).imag.any()


@pytest.mark.petsc
def test_bloch_driven_on_petsc_is_refused_with_petsc_installed(c1_mesh):
    problem = gfem.Driven(pc.C1_FREQUENCY, wave_vector=pc.C1_K)
    with pytest.raises(ValueError, match=r"linear_solver='petsc' factors symmetric"):
        gfem.solve(
            pc.c1(c1_mesh), problem, options=gfem.SolverOptions(linear_solver="petsc")
        )


@pytest.mark.petsc
def test_bloch_eigenmode_on_petsc_is_refused_with_petsc_installed(c3_mesh):
    k = (math.pi / (pc.C3_A * pc.MM), 0.0, 0.0)
    problem = gfem.Eigenmode(target=4e9, wave_vector=k)
    with pytest.raises(ValueError, match=r"linear_solver='petsc' cannot"):
        gfem.solve(
            pc.c3(c3_mesh), problem, options=gfem.SolverOptions(linear_solver="petsc")
        )


@pytest.mark.petsc
def test_the_petsc_handle_refuses_a_bloch_operator_before_assembly(c1_mesh):
    from gfem._engine.driven import driven_operator
    from gfem._linalg import Operator, prepare

    op = driven_operator(pc.c1(c1_mesh), pc.C1_ORDER, wave_vector=pc.C1_K)
    assert op.symmetry == "general"
    form = op.form(2 * math.pi * pc.C1_FREQUENCY)
    with pytest.raises(ValueError, match=r"needs A\^T = A"):
        prepare(Operator(form, op.space.FreeDofs()), symmetry=op.symmetry, kind="petsc")
    with pytest.raises(TypeError, match="assemble"):
        _ = form.mat  # refused before the assembly
