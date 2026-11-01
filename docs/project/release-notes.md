# Release notes

## Unreleased

- **Anisotropic material fields.** A `MaterialField` with `axes` holds three
  principal values per element, shape `(n_elements, 3)`, or for the whole
  region, shape `(3,)`, along fixed axes. `MaterialField.uniform` now takes
  an anisotropic `Material`, and `MaterialField.interpolate` blends two
  materials per principal value when their tensors share principal axes (air
  and a crystal, for example); materials with different axes raise. Driven,
  eigenmode, electrostatic and port mode solves read the field's tensor on
  each element. Gradients with respect to such a field's values, and material
  gradients of a model that holds one, raise `NonDifferentiableError` in this
  version.
- **Wave ports accept anisotropic permeability.** A driven solve with a wave
  port on an anisotropic `mu_r` (with a principal axis along the port normal)
  no longer raises `NotImplementedError`: the port's boundary term uses the
  port's tangential inverse-permeability tensor. Fixed with it: the port's
  modal coupling used the rotated tangential permeability instead of that
  tensor; no 0.1.0 result changes, because 0.1.0 refused anisotropic `mu_r` at
  a wave port before reaching it.
- **Behaviour change: equal principal values are one value.**
  `Material(eps_r=(2, 2, 2))` is now stored as `eps_r=2.0`: it equals
  `Material(eps_r=2)`, `isotropic` is `True`, and its repr changes. Results are
  unchanged. Old job files holding such a tuple read back as the scalar.
- **Behaviour change: absorbing boundaries next to anisotropic materials.**
  - A first-order `Absorbing` surface touching an anisotropic material now
    uses that material's normal-incidence admittance for each face. It
    replaced the mean of the tensor diagonals, which reflected up to about
    13 % at normal incidence next to a uniaxial dielectric (ADR 0051).
  - S, fields and Q of such models change; faces next to isotropic materials
    are bit for bit unchanged.
  - The port mode problem's absorbing edges change the same way: each edge
    next to an anisotropic material takes its exact admittance for the two
    polarizations.
  - A second-order surface touching an anisotropic material now raises
    `NotImplementedError`.
  - Node-coordinate gradients of a model whose absorbing face has an
    anisotropic material behind it now raise `NonDifferentiableError`; 0.1.0
    differentiated the approximate coefficient.
- **Behaviour change: port modes of rotated transverse tensors.** A port
  cross-section holding a tensor rotated about the port normal no longer drops
  its propagating modes (or raises `PortModeError`): the effective-index bound
  now uses the tensor's largest principal value, not its largest diagonal
  entry. Models without such a tensor are unchanged.

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
