# Capabilities

What gfem solves, which results it differentiates, and what it refuses, at
v0.1.

## What you can solve

- **Problems**: {py:obj}`gfem.Driven` (the response of a model to its ports over a
  frequency grid, uniform or adaptive sweep), {py:obj}`gfem.Eigenmode` (resonant
  modes nearest a target frequency, with their Q; lumped ports only: a wave
  port raises), {py:obj}`gfem.Electrostatic` (the
  Maxwell capacitance matrix between named terminals) and {py:obj}`gfem.PortMode`
  (the 2D mode of a wave port). Eigenmode and uniform driven problems take a
  Bloch wave vector (`wave_vector=`, rad/m) on a periodic model.
- **Ports**: {py:obj}`gfem.LumpedPort` on one or several surfaces driven together,
  and {py:obj}`gfem.WavePort` on an exterior surface, fed by its port mode;
  several wave ports on one surface each request a different mode of it,
  each its own row and column of S ({doc}`recipes/shared-port-surface`).
- **Boundaries**: {py:obj}`gfem.PEC`, {py:obj}`gfem.PMC` (symmetry planes), {py:obj}`gfem.Absorbing`
  (first or second order), {py:obj}`gfem.Conductor` (a finite-conductivity metal, by
  its surface impedance), {py:obj}`gfem.Impedance` (per-square R, L and C),
  and {py:obj}`gfem.Periodic` (a periodic pair of surfaces, with plain
  periodicity or a Bloch wave vector set on the problem;
  {doc}`recipes/periodic-cell`).
- **Materials and designs**: {py:obj}`gfem.Material` per region (isotropic, or
  anisotropic: real symmetric tensors on shared principal axes,
  {doc}`recipes/anisotropic-materials`), {py:obj}`gfem.MaterialField` per
  element (isotropic, or principal values on fixed axes), and {py:obj}`gfem.SheetField` for the fill of a patterned metal sheet
  ({doc}`recipes/sheet-design`).
- **Measurements**: {py:obj}`gfem.characteristic_impedance`,
  {py:obj}`gfem.interface_participation` and {py:obj}`gfem.power_flow`
  ({doc}`recipes/measurements`).

## Solvers

- **`linear_solver="direct"`**: pick it for a model whose factorization fits
  in one machine's memory, for eigenmode problems, and for the repeated solves
  of an optimization loop on a small model. It factorizes once per frequency
  and reuses the factorization for every port. Install the `mkl` extra for its fastest
  factorization.
- **`linear_solver="bddc_gmres"`**: pick it for a large model, or when the
  direct factorization does not fit in memory. It is the default
  (`"auto"`) for driven problems.
- **`linear_solver="petsc"`** and **`eigen_solver="slepc"`**: pick them for
  MUMPS and its options, your own PETSc Krylov method and preconditioner, or
  SLEPc's own controls. With no options, `"petsc"` gives the same results as
  `"direct"`, and SLEPc the same modes and Q as ARPACK, the default eigen
  solver. You build them from
  source ({doc}`recipes/petsc-slepc`).
- **`device="cuda"`**: the direct solver on one NVIDIA GPU through cuDSS,
  experimental ({doc}`recipes/gpu-direct-solver`).
- **Across MPI ranks**: a driven uniform sweep and its gradients run split
  across the ranks of one job, on Open MPI and Linux x86_64. Everything else
  raises {py:obj}`gfem.DistributedUnsupportedError` there; the full table is
  in {doc}`recipes/distributed-solve`.

How large a model each solver has finished, and in what time:
{doc}`project/scaling`.

## What you can differentiate

```{gfem-gradient-matrix}
```

- **FD-verified**: supported; the gradient is checked against finite
  differences.
- **Exact zero**: the result does not depend on this input; `jax.grad`
  returns zero, correctly.
- **Raises**: asking for this gradient raises
  {py:obj}`gfem.NonDifferentiableError` (or the error named) instead of
  returning a wrong value.

How each mark is checked: {doc}`verification`.

### Gradients that raise

Off the matrix, asking for these gradients raises the named error instead of
returning a wrong value:

```{gfem-raises}
```

## Limits

- **Low frequency**: below the low-frequency floor, k0·h_min = 1e-6 (f·h_min ≈
  48 Hz·m, with h_min the shortest edge: 48 MHz with 1 µm elements, 5 GHz with
  10 nm elements), a driven solve's fields, gradients and, with the direct
  solver or PETSc's default MUMPS factorization, the capacitive part of S
  lose digits, and it warns with {py:obj}`gfem.LowFrequencyWarning`. For
  capacitance at low frequency, solve {py:obj}`gfem.Electrostatic`; the
  details are under {py:obj}`gfem.Driven`.
- **Gyrotropic materials**: non-symmetric (non-reciprocal) tensors, such as a
  magnetized ferrite, are not supported. Neither is loss whose principal axes
  differ from the permittivity's. Both raise `ValueError`.
- **Anisotropic materials at wave ports**: a port's cross-section may contain
  anisotropic materials when every tensor has a principal axis along the port
  normal. Elsewhere in the model the principal axes may point in any
  direction: a model with tilted principal axes and its rigidly rotated copy
  give the same S, eigenfrequency and capacitance.
- **Second-order absorbing boundaries** raise next to an anisotropic
  material; use order 1, which is exact at normal incidence for any
  anisotropic material.
- **Periodic models**: a Bloch wave vector needs the direct solver for
  eigenmode problems, and the direct or BDDC + GMRES solver for driven
  problems. PETSc and cuDSS refuse it, as does an adaptive sweep.
  Electrostatic problems refuse any periodic pair, and a wave port may not
  touch one. There are no Floquet ports. Details:
  {doc}`recipes/periodic-cell`.
- **Unrequested port modes**: a propagating mode that no wave port on its
  surface requests is partly reflected at the port, not converted; request
  it to see where its power goes ({py:obj}`gfem.SkippedPortModeWarning`
  flags a mode left out below a requested one).
- **Lossy port lines**: wave-port S is the conjugated modal projection, exact
  for a lossless port line. On a shared port surface with loss in the
  cross-section the modes do not fully separate, and
  {py:obj}`gfem.PortModeProjectionWarning` states the error.

## Platforms

Linux x86_64 (glibc >= 2.28), macOS arm64 and Windows x86_64, Python >= 3.12.
The cuDSS GPU solver installs on Linux x86_64 and Windows x86_64. Importing
gfem turns on JAX double precision (`jax_enable_x64`), which every gfem solve
requires. See {doc}`get-started`.

## Licence

MIT. gfem depends on NGSolve (LGPL-2.1), whose PyPI wheels also contain
GPL-2.0+ SuiteSparse code; see {doc}`project/licence`.
