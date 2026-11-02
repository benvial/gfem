# Release notes

## Unreleased

- **Anisotropic materials.** Driven, eigenmode, electrostatic and port mode
  solves, wave ports, first-order absorbing boundaries, `MaterialField` and
  its gradients, `EigenmodeResult.principal_participation`,
  `interface_participation` and job files now take an anisotropic
  `Material`. The cases listed below as unsupported raise. See
  {doc}`../recipes/anisotropic-materials`.
  - `eps_r`, `mu_r`, `tan_delta` and now `conductivity` each take one value,
    three principal values along `axes`, or a real symmetric 3×3 matrix in
    the global frame, stored as principal values plus `axes`. Driven and
    eigenmode solves read a conductivity given as principal values along its
    principal axes.
  - Wave ports accept anisotropic `mu_r`, and anisotropic materials in a
    {term}`transverse-anisotropic` cross-section (every tensor with a
    principal axis along the port normal, at any orientation in the port
    plane). Fixed with it: the port's modal
    coupling used the rotated tangential permeability instead of the port's
    inverse-permeability tensor; no 0.1.0 result changes, because 0.1.0
    refused anisotropic `mu_r` at a wave port before reaching it.
  - First-order absorbing boundaries use the material's exact admittance
    (see the behaviour change below).
  - A `MaterialField` with `axes` holds three principal values per element,
    shape `(n_elements, 3)`, or for the whole region, shape `(3,)`, along
    fixed axes. `MaterialField.uniform` takes an anisotropic `Material`, and
    `MaterialField.interpolate` blends two materials per principal value when
    their tensors share principal axes (air and an anisotropic material, for
    example);
    materials with different axes raise. Driven, eigenmode, electrostatic and
    port mode solves read the field's tensor on each element. `jax.grad`
    gives one cotangent per principal value, checked against finite
    differences in the gradient verification matrix. The axes are fixed: the
    orientation is not a design input.
  - `EigenmodeResult.principal_participation` splits each anisotropic
    region's participation along its principal axes, which gives the exact
    per-region loss for `tan_delta` and `conductivity` given as principal
    values.
  - Errors: passivity errors name the failing principal value and its axis;
    the wave-port error names the property (`eps_r`, `mu_r`, `conductivity`)
    with no principal axis along the port normal, and gives the workarounds;
    non-symmetric (gyrotropic) tensors and tensors with different principal
    axes raise `ValueError` at `Material`.

  Job files and saved results from 0.1.0 read unchanged.
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
- **Behaviour change: `interface_participation`'s vacuum test.** A side is
  vacuum only when no wave in it is slower than in vacuum (largest principal
  `eps_r` × largest `mu_r` equal to 1). A side such as `eps_r = (1, 1, 4)` was
  taken as vacuum, and `"MA"`/`"SA"` read its normal field with ε0 (a 4×
  error); it no longer is. Isotropic sides are unchanged.
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
- **Behaviour change: lossy modes below cutoff.** A requested port mode whose
  attenuation is at least its phase constant (Re β ≤ |Im β|, for example an
  RC-like line at low frequency) now counts as below cutoff: the solve raises
  the port mode sweep's cutoff {py:obj}`gfem.PortModeError`, naming the port
  (and its surface when shared), lone wave ports included, where 0.1.0
  accepted it as a propagating mode.
- **`gfem.Eigenmode`'s message** for a wave port on a shared port surface
  names the surface and its wave ports and points to a `gfem.Driven` sweep.

Periodic cells: `gfem.Periodic` pairs two surfaces of a model related by a
translation (one to three pairs). Eigenmode and uniform driven problems take a
Bloch wave vector, `wave_vector=(kx, ky, kz)` in rad/m, with
E(r + a) = exp(−i k·a) E(r); without one the fields are plain periodic. See
{doc}`../recipes/periodic-cell`.

- **Gradients.** Permittivity, conductivity, sheet fill and node coordinates
  differentiate on periodic models, plain and Bloch, and are checked against
  finite differences (four new rows on {doc}`../capabilities`). A boundary
  morph moves the two nodes of a pair together. `wave_vector` is not
  differentiable.
- **Solvers under a Bloch wave vector.** The operator is no longer complex
  symmetric: the direct solver factors it with MKL PARDISO LU (with the `mkl`
  extra) or UMFPACK, and BDDC factors its coarse problem likewise. Eigenmode
  problems need the direct solver. PETSc, cuDSS and the adaptive sweep refuse
  a wave vector. Models without a periodic pair are unchanged.
- **Results.** Fields of a periodic model hold values on both surfaces of
  every pair. `gfem.power_flow` works on sides that cross a pair.
  `gfem.interface_participation` refuses a paired surface.
- **Job files** carry `Periodic` and `wave_vector`; older job files read
  unchanged.

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
