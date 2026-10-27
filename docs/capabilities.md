# Capabilities

What gfem solves and differentiates at v0.1. The gradient matrix, the raises
list and the Palace tolerances on this page are generated from the test suite
when the docs are built: none of them is typed by hand.

## What differentiates

Every supported gradient is checked against finite differences. An
unsupported one raises an error instead of returning zero; an exact zero
means the physics never reads that input.

```{gfem-gradient-matrix}
```

**FD-verified**: a cell of the gradient verification matrix in
`tests/verification/test_gradient_matrix.py`, which CI's verification job runs
on every pull request and push to `main`. **Exact zero**: the problem never reads the input.
**Raises**: {py:obj}`gfem.NonDifferentiableError` at trace time, unless another error
is named; the outcomes are pinned in
`tests/verification/test_gradient_matrix_raise_table.py`.

## What raises

Outside the matrix, these uses raise a named error rather than return a
gradient that would be wrong:

```{gfem-raises}
```

## Problem types, ports and boundaries

- **Problems**: {py:obj}`gfem.Driven` (the response of a model to its ports over a
  frequency grid, uniform or adaptive sweep), {py:obj}`gfem.Eigenmode` (resonant
  modes nearest a target frequency, with their Q), {py:obj}`gfem.Electrostatic` (the
  Maxwell capacitance matrix between named terminals) and {py:obj}`gfem.PortMode`
  (the 2D mode of a wave port).
- **Ports**: {py:obj}`gfem.LumpedPort` on one or several surfaces driven together,
  and {py:obj}`gfem.WavePort` on an exterior surface, fed by its port mode.
- **Boundaries**: {py:obj}`gfem.PEC`, {py:obj}`gfem.PMC` (symmetry planes), {py:obj}`gfem.Absorbing`
  (first or second order), {py:obj}`gfem.Conductor` (a finite-conductivity metal, by
  its surface impedance) and {py:obj}`gfem.Impedance` (per-square R, L and C).
- **Materials and designs**: {py:obj}`gfem.Material` per region, {py:obj}`gfem.MaterialField`
  per element, and {py:obj}`gfem.SheetField` for the fill of a patterned metal sheet
  ({doc}`recipes/sheet-design`).
- **Measurements**: {py:obj}`gfem.characteristic_impedance`,
  {py:obj}`gfem.interface_participation` and {py:obj}`gfem.power_flow`
  ({doc}`recipes/measurements`).
- **Linear solvers**: direct, BDDC + GMRES, and the experimental cuDSS
  direct solver on an NVIDIA GPU ({doc}`recipes/gpu-direct-solver`).

## Limits

- **Low frequency**: below the low-frequency floor, k0·h_min = 1e-6 (f·h_min ≈
  48 Hz·m, with h_min the shortest edge: 48 MHz with 1 µm elements, 5 GHz with
  10 nm elements), a driven solve's fields, gradients and, with the direct
  solver, the capacitive part of S lose digits, and it warns with
  {py:obj}`gfem.LowFrequencyWarning`. For capacitance at low frequency, solve
  {py:obj}`gfem.Electrostatic`; the details are under {py:obj}`gfem.Driven`.

## Platforms

Linux x86_64 (glibc >= 2.28), macOS arm64 and Windows x86_64, Python >= 3.12.
The cuDSS GPU solver installs on Linux x86_64 and Windows x86_64. Importing
gfem turns on JAX double precision (`jax_enable_x64`), which every gfem solve
requires. See {doc}`get-started`.

## Licence

MIT. gfem depends on NGSolve (LGPL-2.1), whose PyPI wheels also contain
GPL-2.0+ SuiteSparse code; see {doc}`project/licence`.

## Agreement with Palace

On the reference problems gfem and Palace discretize the same mesh at the same
order. Each row is the tolerance its test asserts.

```{gfem-palace-agreement}
```

The reference tier, `tests/reference/`, runs outside the default test
selection: it needs the Palace baselines in `GFEM_REFERENCE_DIR`.
