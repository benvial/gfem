"""The options that docs/recipes/petsc-slepc.md shows, one name per snippet.

The page runs nothing at docs build time, so each snippet is backed by a
test that runs it: `test_docs_petsc_recipe` checks that every dict here is
on the page as written, `test_options` builds a `gfem.SolverOptions` from
each, and the ``petsc`` / ``slepc`` tests named below solve with them.
"""

#: "MUMPS LU, more workspace" (test_linalg_petsc.py)
MUMPS_LU = {"pc_type": "lu", "mat_mumps_icntl_14": 50}
#: "See what PETSc does" (test_linalg_petsc.py)
KSP_VIEW = {"ksp_view": None}
#: A larger SLEPc subspace and its monitor (test_eigenmode_slepc.py)
SLEPC_SUBSPACE = {"eps_ncv": 40, "eps_monitor": None}
#: "See what SLEPc did" (test_eigenmode_slepc.py)
EPS_VIEW = {"eps_view": None}
#: GMRES on the first frequency's factors over a uniform sweep (test_driven.py)
KRYLOV_REUSE = {"ksp_type": "gmres", "ksp_reuse_preconditioner": True}
#: The job file's options, shown as JSON (test_job.py)
JOB = {"ksp_view": None, "pc_type": "lu"}

#: Every `gfem.SolverOptions` the page builds, as keyword arguments.
SOLVER_OPTIONS = {
    "canonical": {"linear_solver": "petsc", "eigen_solver": "slepc"},
    "mumps_lu": {"linear_solver": "petsc", "petsc_options": MUMPS_LU},
    "ksp_view": {"linear_solver": "petsc", "petsc_options": KSP_VIEW},
    "slepc_subspace": {"eigen_solver": "slepc", "petsc_options": SLEPC_SUBSPACE},
    "eps_view": {"eigen_solver": "slepc", "petsc_options": EPS_VIEW},
    "krylov_reuse": {"linear_solver": "petsc", "petsc_options": KRYLOV_REUSE},
    "job": {"linear_solver": "petsc", "eigen_solver": "slepc", "petsc_options": JOB},
}
