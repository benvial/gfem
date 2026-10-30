# Release notes

gfem has no release yet. The notes start with the v0.1 tag.

## Unreleased

The direct and `bddc_gmres` solvers now keep getting faster past 16 CPU
threads on large problems. Results change only to the solver's precision.

- **BDDC solves with gfem's own GMRES.** `bddc_gmres` on a driven or eigenmode
  problem orthogonalizes its Krylov basis on `SolverOptions.threads` threads,
  where NGSolve's GMRES ran that step on one. Iteration counts match NGSolve's
  GMRES and S agrees to rounding. While such a solve runs, other SciPy BLAS
  calls in the same process run on one thread; SciPy's thread count is
  restored when the solve ends.
- **With the `mkl` extra, BDDC factors its coarse problem with MKL PARDISO.**
  This replaces NGSolve's sparse Cholesky and its serial ordering of the
  coarse problem, so BDDC's setup is faster. Without the extra, the coarse
  problem keeps sparse Cholesky. Results with and without the extra agree to
  the `tolerance` of the solve, not bit for bit.
- **On 16 or more threads, a BDDC setup that serves many solves factors
  its coarse problem with sparse Cholesky.** MKL PARDISO's solve does not
  speed up past 16 threads, while sparse Cholesky's does, so once one setup
  serves enough solves, sparse Cholesky is faster overall and uses less
  memory. This applies when one BDDC setup expects 16 or more right-hand
  sides on 16 or more threads, or 6 or more on 56 or more threads: a uniform
  driven sweep whose excited ports times frequencies reach that count, and
  an eigenmode solve's seed shift with `linear_solver="bddc_gmres"`; its
  refinement reshifts count a few solves, so they join only the 56-thread
  tier. The
  threads are `SolverOptions.threads` (or the machine's default), except
  that a {term}`frequency worker` counts its own share of them and its own
  frequencies. A single-frequency solve or an {term}`adaptive sweep` counts
  its excited ports only, an electrostatic solve its terminals. Below 16
  threads, or without the `mkl` extra, nothing changes. The choice depends
  on these counts only, never on timings, so a run repeats exactly.
- **MKL PARDISO no longer leaves its threads spinning after each call.** Every
  PARDISO call gfem makes (the direct solver's, and the solves with BDDC's
  coarse factorization) runs with Intel OpenMP's block time at 0, so MKL's
  threads stop waiting for work as soon as the call returns and free the
  cores for the rest of the solve. The setting covers gfem's calls only and is restored after
  each, so a `KMP_BLOCKTIME` you set applies everywhere but inside them.
- **Single-frequency direct solves do less work before factorizing.** When
  every unknown of the system is free, `linear_solver="direct"` hands the
  assembled matrix to the factorization without copying it first. The index
  that lets gfem refactorize a matrix with new values, such as a sweep's next
  frequency, is now built at the first refactorization, so a single-frequency
  solve never builds it.
