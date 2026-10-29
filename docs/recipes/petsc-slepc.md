# Solve with PETSc and SLEPc

gfem can solve its linear systems with PETSc and find eigenmodes with SLEPc
in place of its built-in solvers. `linear_solver="petsc"` runs the linear
solves of driven, electrostatic and eigenmode problems through PETSc: with no
`petsc_options` an exact MUMPS factorization, like `"direct"`, and otherwise
any PETSc Krylov method and preconditioner. `eigen_solver="slepc"` finds an
eigenmode problem's modes near the target with SLEPc's Krylov-Schur instead of
ARPACK; gfem refines them, computes their Q and differentiates them as it does
after ARPACK, so you get the same modes.

Both run on the CPU, in one process, without MPI. `"auto"` never picks either,
and neither is a pip extra: you build the {term}`PETSc stack` once, from source.

(petsc-when-to-choose)=
## When to choose PETSc

- For speed, keep `"direct"` with the `mkl` extra: on gfem's driven problems
  its PARDISO factorization is faster than MUMPS through PETSc, at every
  frequency of a sweep, in about the same memory. Both give the same S to
  solver precision.
- Without the `mkl` extra, `"direct"` uses NGSolve's sparse Cholesky.
  `"petsc"` with no `petsc_options` is faster than it and needs less memory,
  but installing the `mkl` extra gains more, for less effort.
- Choose `"petsc"` for what only PETSc gives: MUMPS and its options, or your
  own Krylov method and preconditioner through `petsc_options`.
- For a uniform sweep over a narrow band, GMRES on one reused MUMPS
  factorization ({ref}`petsc-krylov-methods`) skips the refactorization at
  every later frequency. Over a wide band, keep the default factorization.
- For eigenmodes, `eigen_solver="slepc"` finds the same modes and Q as ARPACK,
  damped and nonlinear problems included, after about as many shift-invert
  solves, in about the same time and memory. Choose it for SLEPc's own
  controls (`eps_ncv`, `eps_monitor`, `eps_view` and their kin), not for
  speed. An eigenmode solve's time goes into its linear solves, whichever
  eigen solver runs, so for speed pair it with `"direct"` and the `mkl` extra.

## Build the PETSc stack

You need Linux (macOS is not tested, Windows is not supported), uv, curl,
make, cmake, the system's gcc, g++ and gfortran compilers, and a Python not
built by conda (for example `uv python install 3.13` and a uv virtual
environment). The script takes an active conda environment off `PATH`. gfem
needs PETSc with complex scalars and without MPI, and conda-forge's PETSc
depends on ParMETIS, which is not free for commercial use
({doc}`../project/licence`).

The script `tools/petsc_wheelhouse.sh` in the repository
({download}`download it <../../tools/petsc_wheelhouse.sh>`, or read it at the
end of this page) builds PETSc, SLEPc, petsc4py and slepc4py, with MUMPS, METIS
and OpenBLAS, into a wheelhouse: a folder of four wheels. It compiles all of
them from source, a long build done once per Python version. Run it with the
environment that runs gfem active, since the wheels are built for its Python:

```bash
bash petsc_wheelhouse.sh --check      # check the prerequisites and the target Python
bash petsc_wheelhouse.sh wheelhouse   # build the four wheels into ./wheelhouse
pip install --no-index --find-links wheelhouse petsc petsc4py slepc slepc4py
```

- `PYTHON` picks another target Python, as a path or a version
  (`PYTHON=3.12`); `MAKE_NP` sets the number of parallel compile jobs.
- In a uv project, install with `uv pip install`. A plain `uv sync` removes
  packages its lock file does not list, these four included: sync with
  `uv sync --inexact`, or rerun the install line after an exact sync. `uv run`
  leaves them in place.
- Rebuild for another Python version, and when gfem raises the PETSc version
  it supports: the error message names the oldest one it accepts.

## Check the build

```python
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

np.dtype(PETSc.ScalarType)  # dtype('complex128')
PETSc.Sys.hasExternalPackage("mumps")  # True
```

If gfem cannot use the stack, the solve raises `gfem.SolverUnavailableError`
with the reason (a package that does not import, real scalars, a version older
than gfem supports) and the fix. gfem never falls back to another solver.

## Choose the solvers

```python
import gfem

options = gfem.SolverOptions(linear_solver="petsc", eigen_solver="slepc")
modes = gfem.solve(model, gfem.Eigenmode(target=7.8e9, n_modes=2), options=options)
modes.frequency, modes.q
```

- The two fields are independent: SLEPc works with every linear solver,
  `"direct"` on a GPU included, and `"petsc"` solves every problem type.
- `SolverOptions` refuses `device="cuda"` with `"petsc"`. Wave port modes are
  2D problems that always run SuperLU and SciPy on the CPU, so a standalone
  `gfem.PortMode` solve refuses both fields.
- `frequency_workers` work: each worker process starts PETSc itself.
- gfem checks the stack when the solve starts, never when the options are
  built, so a job naming PETSc can be written anywhere and run where the stack
  is installed. The options of such a job file:

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
SLEPc and need `eigen_solver="slepc"`; every other name goes to the linear
solver and needs `linear_solver="petsc"`.

```python
options = gfem.SolverOptions(
    linear_solver="petsc",
    petsc_options={"pc_type": "lu", "mat_mumps_icntl_14": 50},  # more MUMPS workspace
)
```

- `{"ksp_view": None}` prints the linear solver PETSc set up.
- `{"eps_ncv": 40, "eps_monitor": None}` gives SLEPc a larger subspace and
  prints its progress; `eps_mpd` and `eps_max_it` also reach SLEPc.
- `{"eps_view": None}` prints the eigensolver SLEPc ran, at the end of each
  seed solve; so do SLEPc's other end-of-solve viewers, `eps_converged_reason`
  and `eps_error_*`.
- Never write an options prefix: gfem gives each solver its own and holds its
  options in PETSc's database only while that solver reads them. Command-line
  PETSc options and your own petsc4py options do not reach gfem's solvers.
- An option the linear solver does not read raises `ValueError` when the solve
  starts: a typo, or an option of another configuration, such as a MUMPS
  option with `pc_type` `jacobi`. So does an option PETSc or SLEPc refuses,
  with its reason. An option SLEPc does not read, such as `eps_mpdd`, raises
  `ValueError` after its seed solve.

(petsc-krylov-methods)=
### Krylov methods

With only `ksp_type` set, gfem's MUMPS factorization is the preconditioner.
Keep the first frequency's factors, and GMRES runs a whole uniform sweep on
one factorization:

```python
options = gfem.SolverOptions(
    linear_solver="petsc",
    petsc_options={"ksp_type": "gmres", "ksp_reuse_preconditioner": True},
)
```

This pays off over a narrow band, where GMRES needs a few iterations per
frequency in place of a refactorization. Further from the first frequency the
iterations grow: over a wide band GMRES reaches `ksp_max_it` after a few
frequencies, and the sweep raises a `RuntimeError` naming the residual it
reached, without returning the frequencies it solved. Unlike `bddc_gmres`'s
preconditioner, which gfem builds again when reusing it stops paying, this one
is never set up again on its own: split a wide band into narrow sweeps, or keep
the default factorization. Adaptive sweeps set the preconditioner up at every
sample.

- `tolerance` (`adjoint_tolerance` for adjoint solves) becomes `ksp_rtol`: the
  true, unpreconditioned residual relative to the right-hand side.
  `bddc_gmres` measures its preconditioned residual instead, so the same
  `tolerance` is a different norm per backend. `ksp_max_it` defaults to 400.
  Set `ksp_norm_type` for PETSc's own norm.
- Driven and eigenmode operators are complex symmetric, not Hermitian: use
  GMRES-type methods (`gmres`, `fgmres`, `bcgs`, `tfqmr`). PETSc's CG assumes
  a Hermitian operator, and its `ksp_cg_type symmetric` variant stops when the
  real part of $p^T A p$ changes sign, which these operators allow. CG suits
  the electrostatic problem, whose operator is real symmetric positive definite.
- gfem brings no Krylov preconditioner for curl-curl operators: PETSc's
  algebraic multigrid does not know the curl's null space, and the build has
  no hypre. Incomplete factorizations fail too: GMRES with PETSc's ILU, with or
  without fill levels, does not converge on gfem's driven problems. To try
  one anyway, also set `pc_factor_mat_solver_type` to `petsc`: gfem's default
  MUMPS has no ILU, so `pc_type` `ilu` alone is refused.

## What gfem refuses, and what to set instead

`SolverOptions` raises `ValueError` for these options, naming what to set:

| Option | Why | Set instead |
|---|---|---|
| `-ksp_type` | names are written without `-` | `ksp_type` |
| `ksp_rtol` | gfem sets it | `tolerance`, `adjoint_tolerance` |
| `ksp_initial_guess_nonzero` | gfem sets it per solve | nothing: remove it |
| `eps_tol` | gfem sets it | `eigen_tolerance` |
| `eps_nev` | gfem sets it | `Eigenmode.n_modes` |
| `eps_target`, `eps_largest_real`, other selections | gfem keeps the modes nearest the target | `Eigenmode.target` |
| `st_ksp_type`, any `st_` option | gfem applies the shift-invert itself | `linear_solver`; with `"petsc"`, its `ksp_` and `pc_` options |
| `eps_gen_hermitian`, `eps_conv_abs`, `eps_two_sided`, `eps_harmonic` and their kin | SLEPc sees a non-Hermitian shift-inverted operator | `eigen_tolerance` for convergence |
| `eps_type` other than `krylovschur` or `arnoldi` | the others do not suit that operator | `krylovschur` (the default) or `arnoldi` |
| `ksp_type` with `linear_solver="direct"` | the option's solver is not selected | select it |

When the solve starts, gfem also refuses an option the linear solver did not
read, an option PETSc refuses, and `ksp_reuse_preconditioner` with `ksp_type`
`preonly`, which would solve with a stale operator.

## Threads

`linear_solver="petsc"` does not read `SolverOptions.threads`. MUMPS runs its
dense kernels on the threaded OpenBLAS the script builds: set
`OPENBLAS_NUM_THREADS` before Python starts. Frequency workers already give
each worker its share of the threads.

## How to trust the results

- With no `petsc_options`, `"petsc"` is an exact factorization: its results
  agree with `"direct"` to solver precision. SLEPc only finds first
  approximations of the modes, which gfem refines as it does ARPACK's: modes,
  Q, fields and gradients agree with ARPACK's to solver precision.
- The tests marked `petsc` and `slepc` check both:
  `tests/unit/test_linalg_petsc.py`, `tests/unit/test_eigenmode_slepc.py` and
  the PETSc cells of `tests/verification/test_gradient_matrix.py`. They run
  only where the PETSc stack is installed, so the {doc}`../capabilities`
  gradient matrix does not list PETSc.
- For speed, see {ref}`petsc-when-to-choose`: `"petsc"` is slower than
  `"direct"` with the `mkl` extra, and SLEPc is about as fast as ARPACK.

## Limits

- CPU only, one process: no MPI; spread a uniform sweep with
  `frequency_workers`. Tested on Linux only.
- SLEPc's polynomial and nonlinear eigensolvers are not used: damped and
  nonlinear eigenmode problems go through the same shift-invert iteration as
  with ARPACK.
- Your own petsc4py code in the same process: gfem starts PETSc without
  command-line options (or uses the PETSc you started), keeps its options to
  itself, and serializes only its own PETSc calls. Do not call PETSc from
  another thread during a gfem solve.

## Next

{py:obj}`gfem.SolverOptions`, {py:obj}`gfem.SolverUnavailableError`,
{doc}`eigenmode-gradient`, {doc}`gpu-direct-solver`, {doc}`../project/licence`.

## The build script

```{literalinclude} ../../tools/petsc_wheelhouse.sh
:language: bash
```
