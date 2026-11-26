# PETSc and SLEPc

gfem can solve its linear systems with PETSc and find eigenmodes with SLEPc
in place of its built-in solvers. `linear_solver="petsc"` gives driven,
electrostatic and eigenmode problems an exact MUMPS factorization, or with
`petsc_options` any PETSc Krylov method and preconditioner.
`eigen_solver="slepc"` finds the modes near the target with Krylov-Schur
instead of ARPACK; gfem refines, differentiates and gives Q as after ARPACK.
Both need the {term}`PETSc stack`, built once from source: `"auto"` never
picks either, and neither is a pip extra. This page builds the
{term}`no-MPI PETSc stack`; for ranks see {ref}`petsc-with-mpi`.

(petsc-when-to-choose)=
## When to choose PETSc

- For speed, keep `"direct"` with the `mkl` extra: PARDISO beats MUMPS through
  PETSc, with the same S. Without `mkl`, `"petsc"` beats NGSolve's Cholesky.
- Choose `"petsc"` for MUMPS and its options, or your own Krylov method
  ({ref}`petsc-krylov-methods`); `"slepc"` for SLEPc's controls
  (`eps_ncv`, `eps_monitor`), not speed: modes and Q match ARPACK's.

## Build the PETSc stack

You need Linux, uv, curl, make, cmake, gcc, g++, gfortran and a Python not
built by conda (conda-forge's PETSc depends on ParMETIS,
{doc}`../project/licence`). `petsc_wheelhouse.sh`
({download}`download it <../../tools/petsc_wheelhouse.sh>`, or read it below)
compiles PETSc, SLEPc, petsc4py and slepc4py with complex scalars, MUMPS,
METIS and OpenBLAS into four wheels, once per Python version, in about 22
minutes. Run it with gfem's environment active:

```bash
bash petsc_wheelhouse.sh --check      # check the prerequisites and the target Python
bash petsc_wheelhouse.sh wheelhouse   # build the four wheels into ./wheelhouse
pip install --no-index --find-links wheelhouse petsc petsc4py slepc slepc4py
```

- `PYTHON=3.12` picks another target Python; `MAKE_NP` sets the compile jobs.
- In a uv project use `uv pip install`; a plain `uv sync` removes the
  packages (use `--inexact`).
- Rebuild for another Python, a newer PETSc version, and a stack from gfem
  0.1.0's script, which gfem refuses ({doc}`../project/release-notes`): rerun
  the script into a new folder, then the install line with `--force-reinstall`.

(petsc-with-mpi)=
### With MPI

`bash petsc_wheelhouse.sh --mpi wheelhouse` builds the {term}`MPI PETSc stack`
beside the no-MPI one, for the {doc}`distributed solve recipe <distributed-solve>`,
on the Open MPI whose `mpicc`, `mpicxx` and `mpifort` are on `PATH`; it also
needs bison 3+, flex and cmake 3.26+ (`--check` names what is missing). It has
complex scalars, 64-bit indices, MUMPS, METIS, ScaLAPACK and PT-Scotch.

- An eigenmode solve across ranks uses SLEPc from this stack.
- A stack runs under the Open MPI it was built on. The no-MPI stack under a
  launcher serves {term}`rank copies` only; a {term}`distributed solve`
  naming PETSc raises `gfem.SolverUnavailableError`.
- `--mpi --with-parmetis` adds ParMETIS, which may not be redistributed.

## Check the build

```python
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

np.dtype(PETSc.ScalarType)  # dtype('complex128')
PETSc.Sys.hasExternalPackage("mumps")  # True
```

A stack gfem cannot use raises `gfem.SolverUnavailableError` with the fix;
gfem never falls back to another solver.

## Choose the solvers

```python
import gfem

options = gfem.SolverOptions(linear_solver="petsc", eigen_solver="slepc")
modes = gfem.solve(model, gfem.Eigenmode(target=7.8e9, n_modes=2), options=options)
modes.frequency, modes.q
```

- The two fields are independent; `device="cuda"` with `"petsc"` is refused.
  `frequency_workers` work.
- gfem checks the stack when the solve starts, so a job naming PETSc can be
  written anywhere:

```json
{
  "linear_solver": "petsc",
  "eigen_solver": "slepc",
  "petsc_options": {"ksp_view": null, "pc_type": "lu"}
}
```

## Pass PETSc and SLEPc options

`petsc_options` maps PETSc option names, without the leading `-`, to values;
`None` is a flag. Names starting `eps_`, `bv_`, `ds_`, `rg_` or `fn_` go to
SLEPc and need `eigen_solver="slepc"`; the rest go to the linear solver and
need `linear_solver="petsc"`.

```python
options = gfem.SolverOptions(
    linear_solver="petsc",
    petsc_options={"pc_type": "lu", "mat_mumps_icntl_14": 50},  # more MUMPS workspace
)
```

- `{"ksp_view": None}` prints the linear solver PETSc set up;
  `{"eps_ncv": 40, "eps_monitor": None}` enlarges SLEPc's subspace and prints
  its progress; `{"eps_view": None}` prints the eigensolver SLEPc ran.
- Write no options prefix; command-line and petsc4py options do not reach
  gfem.
- An option the solver does not read, or that PETSc or SLEPc refuses, raises
  `ValueError`.

(petsc-krylov-methods)=
### Krylov methods

With only `ksp_type` set, gfem's MUMPS factorization is the preconditioner.
Keep the first frequency's factors, and GMRES runs a whole uniform sweep on
them:

```python
options = gfem.SolverOptions(
    linear_solver="petsc",
    petsc_options={"ksp_type": "gmres", "ksp_reuse_preconditioner": True},
)
```

This pays off over a narrow band, where the iterations stay few. Where GMRES
does not converge (`ksp_max_it`, default 400), gfem factorizes again there and
reuses the new factors; lower `ksp_max_it` to refactorize sooner. A solve that
still fails raises `gfem.LinearSolverConvergenceError`.

- `tolerance` (`adjoint_tolerance` for adjoint solves) becomes `ksp_rtol`, the
  true residual relative to the right-hand side; `bddc_gmres` and `pmg_gmres`
  measure the preconditioned one. `ksp_norm_type` sets PETSc's own norm.
- Driven and eigenmode operators are complex symmetric, not Hermitian: use
  `gmres`, `fgmres`, `bcgs` or `tfqmr`; CG suits only electrostatics.
- No PETSc Krylov preconditioner suits curl-curl operators: algebraic
  multigrid does not know the curl's null space (no hypre) and ILU does not
  converge.
- Below the low-frequency floor ({py:obj}`gfem.LowFrequencyWarning`) PETSc
  solves are not gauged and lose the capacitive digits of S; `"direct"` on
  one process keeps them.

## What gfem refuses, and what to set instead

`SolverOptions` raises `ValueError` for these options and names the setting:

| Option | Set instead |
|---|---|
| `ksp_rtol`, `ksp_atol` | `tolerance`, `adjoint_tolerance` |
| `ksp_initial_guess_nonzero` | nothing: gfem sets it |
| `eps_tol`, `eps_nev` | `eigen_tolerance`, `Eigenmode.n_modes` |
| `eps_target`, `eps_largest_real`, other selections | `Eigenmode.target` |
| any `st_` option | `linear_solver` and its `ksp_`, `pc_` options |
| `eps_gen_hermitian`, `eps_conv_abs`, `eps_two_sided`, `eps_harmonic` | `eigen_tolerance` |
| `eps_type` other than `krylovschur` or `arnoldi` | `krylovschur` (default) or `arnoldi` |

At the solve start gfem also refuses `ksp_reuse_preconditioner` with
`preonly`, `preonly` with a `pc_type` other than `lu` or `cholesky`, and
`ksp_convergence_test` `skip`.

## Check your results

- **Two linear solvers.** `"petsc"` without `petsc_options` and `"direct"`
  give S to solver precision.
- **Two eigen solvers.** `"slepc"` and `"arpack"` give the same modes, Q,
  fields and gradients to solver precision.

How gfem itself is checked on PETSc and SLEPc:
{ref}`verification <verification-petsc>`.

## Limits

- CPU and Linux only. The no-MPI stack runs one process ({doc}`distributed-solve`
  for ranks).
- An eigenmode solve with `"petsc"` differentiates eigenfrequencies only; a
  gradient through a mode's field raises `gfem.NonDifferentiableError`: use
  `"direct"` ({doc}`eigenmode-gradient`).
- `"petsc"` ignores `SolverOptions.threads`: set `OPENBLAS_NUM_THREADS`.
- Do not call PETSc from another thread during a gfem solve.

## Next

{py:obj}`gfem.SolverOptions`, {py:obj}`gfem.SolverUnavailableError`,
{doc}`eigenmode-gradient`, {doc}`gpu-direct-solver`, {doc}`distributed-solve`,
{doc}`../project/licence`.

## The build script

```{literalinclude} ../../tools/petsc_wheelhouse.sh
:language: bash
```
