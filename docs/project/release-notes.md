# Release notes

## Unreleased

- **Wrong S from a PETSc stack built by 0.1.0's script: rebuild it.** A stack built by
  0.1.0's [`tools/petsc_wheelhouse.sh`](https://github.com/benvial/gfem/blob/main/tools/petsc_wheelhouse.sh)
  can return wrong S, with no error, on a large direct solve with
  `linear_solver="petsc"`. A 10 M DOF mesh solved in one process stayed below the size
  where this happens. Rebuild the stack with the current script, as
  {doc}`../recipes/petsc-slepc` says. The cause was an integer overflow in an OpenBLAS
  routine that MUMPS calls; the script now builds MUMPS without it.
  - **Behaviour change.** gfem refuses a stack built without this fix, at any size, with
    `gfem.SolverUnavailableError`. `linear_solver="bddc_gmres"` above one rank still
    runs on such a stack, without using it for its coarse problem.
- **`Field.at` reads the tetrahedron that holds the point.** A point near a face, within
  1e-4 of an element's size, could get the neighbouring element's value, off by the
  field's full jump there, and the value could change with the rank count. It now gets
  the tetrahedron it lies in, on any number of ranks. Within 1e-9 of an element's size
  from a face, a point counts as on the face. A point on or just above an excluded
  conductor no longer raises a false "lies in volume region"; a point just inside one,
  which used to read the air's value, now raises. See {doc}`../recipes/measurements`.
- **Behaviour change: default thread count.** With `threads=None`, NGSolve now runs on
  `NGS_NUM_THREADS`, else on the physical cores the process may run on, instead of every
  hardware thread of the machine. Without an affinity mask (macOS, Windows) the count is
  the logical CPUs. Your own `ngs.SetNumThreads` no longer carries into such a solve:
  set `SolverOptions.threads` instead. Inside your own `ngs.TaskManager()`, your thread
  count is kept.
- **Fixed: periodic gradients of a `MaterialField` with `axes`.** Recompute them if
  your periodic model's fields touch a first-order `Absorbing` surface. There, the
  gradients with respect to `eps_r` and `mu_r` principal values were wrong: off by 0.5 %
  to 10 % in the cases measured, more for eigenmodes. They now agree with finite
  differences. Periodic models without such a surface got the right gradients and now
  get them faster. Non-periodic models were not affected.
- **Driven solves across MPI ranks.** `mpirun -n N python script.py` or
  `srun --mpi=pmix` now runs a uniform driven sweep and its gradients across the
  ranks, with `comm=` on `gfem.solve`, `Job.run`, `Mesh.from_gmsh`, `Job.from_json`
  and `load_result`. Every other problem and some options raise
  {py:obj}`gfem.DistributedUnsupportedError`. See {doc}`../recipes/distributed-solve`.
  - The `mpi` extra is mpi4py; `import gfem` needs neither it nor MPI. Only Open MPI
    is accepted above one rank.
  - [`tools/petsc_wheelhouse.sh`](https://github.com/benvial/gfem/blob/main/tools/petsc_wheelhouse.sh)
    gains `--mpi`, `--sources` and `--fetch-sources`, for the {term}`MPI PETSc stack`,
    which sits beside the unchanged no-MPI stack.
  - **Behaviour change.** A launched solve used to run N complete one-process solves
    with no warning. It now runs distributed or raises, and a launched install
    without mpi4py raises `gfem.SolverUnavailableError`.
    `comm=MPI.COMM_SELF` asks for the copies by name.
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
    {term}`transverse-anisotropic` cross-section: every tensor has a
    principal axis along the port normal, at any orientation in the port
    plane.
  - First-order absorbing boundaries use the material's exact admittance
    (see the behaviour change below).
  - A `MaterialField` with `axes` holds three principal values along fixed axes:
    shape `(n_elements, 3)` per element, or `(3,)` for the whole region.
    `MaterialField.uniform` takes an anisotropic `Material`.
    `MaterialField.interpolate` blends two materials per principal value when
    their tensors share principal axes (air and an anisotropic material, for
    example); materials with different axes raise. Driven, eigenmode,
    electrostatic and port mode solves read the field's tensor on each element.
    `jax.grad` gives one derivative per principal value, checked against finite
    differences. The axes are fixed: the orientation is not a design input.
  - `EigenmodeResult.principal_participation` splits each anisotropic
    region's participation along its principal axes. It gives the exact
    per-region loss for `tan_delta` and `conductivity` given as principal
    values.
  - Errors: passivity errors name the failing principal value and its axis.
    The wave-port error names the property (`eps_r`, `mu_r`, `conductivity`)
    with no principal axis along the port normal, and gives the workarounds.
    Non-symmetric (gyrotropic) tensors and tensors with different principal
    axes raise `ValueError` at `Material`.

  Job files and saved results from 0.1.0 read unchanged.
- **Behaviour change: equal principal values are one value.**
  `Material(eps_r=(2, 2, 2))` is now stored as `eps_r=2.0`: it equals
  `Material(eps_r=2)`, `isotropic` is `True`, and its repr changes. Results are
  unchanged. Old job files holding such a tuple read back as the scalar.
- **Behaviour change: absorbing boundaries next to anisotropic materials.**
  - S, fields and Q change for a model whose first-order `Absorbing` surface
    touches an anisotropic material. Each such face now uses that material's
    normal-incidence admittance. The mean of the tensor diagonals, used before,
    reflected up to about 13 % at normal incidence next to a uniaxial dielectric.
    Faces next to isotropic materials are bit for bit unchanged.
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
  its propagating modes (or raises `PortModeError`). The effective-index bound
  now uses the tensor's largest principal value, not its largest diagonal
  entry. Models without such a tensor are unchanged.
- **Behaviour change: `interface_participation`'s vacuum test.** A side is
  vacuum only when no wave in it is slower than in vacuum (largest principal
  `eps_r` × largest `mu_r` equal to 1). A side such as `eps_r = (1, 1, 4)` was
  taken as vacuum, and `"MA"`/`"SA"` read its normal field with ε0 (a 4×
  error); it no longer is. Isotropic sides are unchanged.
- **Several modes on one wave port surface.** Several
  {py:obj}`gfem.WavePort`s on one surface, one `mode` each, form a
  {term}`shared port surface`. Each mode is its own port, with its own row
  and column of S, so S holds the coupling between modes (for example TE0 to
  TM0 conversion in a silicon strip). Gradients and adaptive sweeps work as
  for one mode per port. {py:obj}`gfem.PortMode` on such a port gives the mode
  the driven solve uses. See {doc}`../recipes/shared-port-surface`.
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
  {py:obj}`gfem.PortModeError` before the 3D solve, where 0.1.0 went on
  without warning. Lone-port S is otherwise unchanged.
- **Behaviour change: lossy modes below cutoff.** A requested port mode whose
  attenuation is at least its phase constant (Re β ≤ |Im β|, for example an
  RC-like line at low frequency) now counts as below cutoff, where 0.1.0
  accepted it as a propagating mode. The solve raises the port mode sweep's
  cutoff {py:obj}`gfem.PortModeError`, naming the port (and its surface when
  shared), lone wave ports included.
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
- **Solvers under a Bloch wave vector.** Eigenmode problems need the direct
  solver. PETSc, cuDSS and the adaptive sweep refuse a wave vector. The direct
  solver factors the problem with MKL PARDISO (with the `mkl` extra) or
  UMFPACK, and BDDC factors its coarse problem likewise. Models without a
  periodic pair are unchanged.
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

- **`bddc_gmres` runs its GMRES on all its threads.** A driven or eigenmode solve
  with `bddc_gmres` now uses `SolverOptions.threads` threads for the step where
  NGSolve's GMRES used one. Iteration counts and S are unchanged, to rounding.
  While such a solve runs, other SciPy BLAS calls in the same process run on one
  thread; SciPy's thread count is restored when the solve ends.
- **With the `mkl` extra, BDDC's setup is faster.** BDDC factors its coarse
  problem with MKL PARDISO instead of NGSolve's sparse Cholesky. Without the
  extra, nothing changes. Results with and without the extra agree to the
  `tolerance` of the solve, not bit for bit.
- **On 16 or more threads, a BDDC setup that serves many solves uses sparse
  Cholesky.** Once one setup serves enough solves, sparse Cholesky is faster
  overall than MKL PARDISO there, and uses less memory. This applies when one
  BDDC setup expects 16 or more right-hand sides on 16 or more threads, or 6 or
  more on 56 or more threads. A uniform driven sweep counts its excited ports
  times its frequencies; a single-frequency solve or an {term}`adaptive sweep`
  counts its excited ports only, and an electrostatic solve its terminals. A
  {term}`frequency worker` counts its own share of the threads and its own
  frequencies. Below 16 threads, or without the `mkl` extra, nothing changes.
  The choice depends on these counts only, never on timings, so a run repeats
  exactly.
- **MKL PARDISO frees its cores as soon as each call returns.** Every PARDISO
  call gfem makes runs with Intel OpenMP's block time at 0, so the rest of the
  solve gets the cores back. The setting is restored after each call, so a
  `KMP_BLOCKTIME` you set applies everywhere else.
- **A reused PARDISO factorization keeps its thread count.** With the `mkl`
  extra, a factorization reused under a different thread count, or outside the
  `TaskManager` it was made in, used to solve on one thread: about 5 times
  slower on a 10-million-unknown problem. A gradient that reused a cached
  adjoint factorization under `threads=None` could hit this. It now solves on
  the threads it was factored with. A factorization made outside any
  `TaskManager`, such as `gfem.design.helmholtz_filter`'s, likewise keeps
  solving on one thread. Results do not change.
- **Single-frequency direct solves start factorizing sooner.**
  `linear_solver="direct"` no longer copies the assembled matrix when every
  unknown is free. It builds the index it needs to refactorize with new values
  only when a second frequency needs it.
