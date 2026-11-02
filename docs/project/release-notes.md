# Release notes

## Unreleased

- **Several modes on one wave port surface.** Several
  {py:obj}`gfem.WavePort`s on one surface, one `mode` each, form a
  {term}`shared port surface`: each mode is its own port, with its own row
  and column of S, so S holds the coupling between modes (for example TE0 to
  TM0 conversion in a silicon strip). Gradients and adaptive sweeps work as
  for one mode per port, and {py:obj}`gfem.PortMode` on such a port runs the
  surface's sweep, so it gives the mode the driven solve uses. See
  {doc}`../recipes/shared-port-surface`.
- **New warnings.** {py:obj}`gfem.SkippedPortModeWarning`: a surface leaves
  out a port mode ranked below a requested one (a lone `mode=2`, or modes 1
  and 3), which the port then partly reflects. A lone `mode=1` never warns.
  {py:obj}`gfem.PortModeProjectionWarning`: on a shared port surface with
  loss in the cross-section, one port's S reads part of another port's mode
  above 1e-3.
- **New errors at `gfem.Model`.** Two wave ports on one surface that request
  the same `mode`, or give different `pec_edges`, raise `ValueError`. A lumped
  port or a boundary condition beside a wave port on one surface still does.
- **Behaviour change for a lone wave port.** A port mode whose reaction or
  Gram matrix on the port face is singular now raises
  {py:obj}`gfem.PortModeError` before the 3D solve, where 0.1.0 silently
  dropped the port's modal correction. Lone-port S is otherwise unchanged.
- **`gfem.Eigenmode`'s message** for a wave port on a shared port surface
  names the surface and its wave ports and points to a `gfem.Driven` sweep.

## 0.1.0 (2026-11-01)

First release: driven problems (uniform and adaptive sweeps), eigenmode problems
(including nonlinear ones), electrostatic and port mode problems, with lumped
and wave ports. Gradients with respect to permittivity, conductivity, sheet
fill and mesh node coordinates are checked against finite differences.

Changes since the release candidate of 2026-10-21:

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
- **On 16 or more threads, a BDDC setup that serves many solves factors its
  coarse problem with sparse Cholesky.** MKL PARDISO's solve does not speed up
  past 16 threads, while sparse Cholesky's does, so once one setup serves
  enough solves, sparse Cholesky is faster overall and uses less memory. This
  applies when one BDDC setup expects 16 or more right-hand sides on 16 or
  more threads, or 6 or more on 56 or more threads: a uniform driven sweep
  whose excited ports times frequencies reach that count, and an eigenmode
  solve's seed shift with `linear_solver="bddc_gmres"`; its refinement
  reshifts count a few solves, so they join only the 56-thread tier. The
  threads are `SolverOptions.threads` (or the machine's default), except that
  a {term}`frequency worker` counts its own share of them and its own
  frequencies. A single-frequency solve or an {term}`adaptive sweep` counts
  its excited ports only, an electrostatic solve its terminals. Below 16
  threads, or without the `mkl` extra, nothing changes. The choice depends on
  these counts only, never on timings, so a run repeats exactly.
- **MKL PARDISO no longer leaves its threads spinning after each call.** Every
  PARDISO call gfem makes (the direct solver's, and the solves with BDDC's
  coarse factorization) runs with Intel OpenMP's block time at 0, so MKL's
  threads stop waiting for work as soon as the call returns and free the
  cores for the rest of the solve. The setting covers gfem's calls only and is restored after
  each, so a `KMP_BLOCKTIME` you set applies everywhere but inside them.
- **A PARDISO factorization keeps solving on its own thread count.** With
  the `mkl` extra, every PARDISO factorization gfem makes (the direct
  solver's, and BDDC's coarse factorization) keeps solving on the threads it
  was factored with when it is reused outside the TaskManager it was made in,
  or under a different thread count, instead of falling back to one thread.
  MKL PARDISO used to run every later solve on that factorization
  sequentially, about 5 times slower on a 10-million-unknown problem. This
  can happen when a later gradient reuses a cached adjoint factorization
  under `threads=None` after another solve changed the thread count. The
  reverse holds too: a factorization made outside any TaskManager, such as
  `gfem.design.helmholtz_filter`'s and `gfem.shape.boundary_morph`'s, keeps
  solving on one thread inside one. The results do not change.
- **Single-frequency direct solves do less work before factorizing.** When
  every unknown of the system is free, `linear_solver="direct"` hands the
  assembled matrix to the factorization without copying it first. The index
  that lets gfem refactorize a matrix with new values, such as a sweep's next
  frequency, is now built at the first refactorization, so a single-frequency
  solve never builds it.
