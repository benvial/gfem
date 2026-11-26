# Frequency sweeps

Get S and the fields of a driven model over a broadband grid without paying
one full solve per frequency at a time. A {py:obj}`gfem.Driven` sweep is
uniform by default. Two options speed a long one up: the {term}`adaptive sweep`, which solves only a
few frequencies, and {term}`frequency workers <frequency worker>`, which solve
several at once on a many-core machine.

## S at 300 frequencies from a few solves

`model` is any driven model with two ports `p1` and `p2`, for example the one the
{doc}`gsim recipe <gsim>` builds. Ask for the adaptive sweep with `sweep="adaptive"`:

```python
import numpy as np

import gfem

problem = gfem.Driven(frequency=np.linspace(1e9, 100e9, 300), sweep="adaptive")
result = gfem.solve(model, problem)
s21 = result.s["p2", "p1"]  # all 300 frequencies
result.sweep.sampled_frequencies  # the frequencies solved in full, Hz
```

- The sweep solves a few sample frequencies in full, every excited port at
  once, and reads S and the kept fields at every grid frequency from a
  {term}`reduced model` built on those solutions.
- Each sample also solves one right-hand side per port left out of
  `Driven(excite=...)`, so gradients are exact at the samples
  ({doc}`adaptive-sweep-gradient`).
- Point dipoles (`Driven(sources=...)`, {doc}`point-dipoles`) sweep too, with
  or without ports. `Driven(source_fields=False)` keeps the basis from growing
  with the number of dipoles. An incident field does not sweep: `Driven`
  raises `NotImplementedError`.
- It pays off above about 20 frequencies; below that, keep the uniform sweep.
- `result.sweep.solved` marks the frequencies solved in full.
  `result.sweep.converged` says whether the last samples agreed with the
  reduced model to `SolverOptions.adaptive_tolerance`.
- A sweep that reaches `SolverOptions.adaptive_max_samples` first warns with
  {py:obj}`gfem.SweepNotConvergedWarning` and reports `converged=False`; raise
  `adaptive_max_samples`.
- A sweep near the accuracy of its full solves warns with
  {py:obj}`gfem.SweepRankDeficientWarning` and goes on; tighten
  `SolverOptions.tolerance` or loosen `adaptive_tolerance` if it then fails.
- On a short grid it solves every frequency in full and reports
  `converged=True`.
- For time responses, sweep a {term}`harmonic grid`,
  `df * np.arange(1, n + 1)` ({doc}`time-responses`).

## Set the accuracy of an adaptive sweep

The speed costs field accuracy. A uniform sweep's fields are as accurate as
the linear solve (`SolverOptions.tolerance`). Between samples an adaptive
sweep's fields have a relative error of about `adaptive_tolerance` (`1e-3` by
default) or less, and S about its square. Measurements read the fields
({doc}`measurements`) and carry the same error.

Tighten it at the cost of more samples:

```python
options = gfem.SolverOptions(adaptive_tolerance=1e-5)
result = gfem.solve(model, problem, options=options)
```

- `gfem.SolverOptions.preset("accurate")` also sets `adaptive_tolerance=1e-5`,
  with a higher element order and a tighter linear solve.
- `adaptive_tolerance` below 10 times `tolerance` raises `ValueError`; tighten
  `tolerance` with it.
- Where the fields matter most, keep the uniform sweep.
- Under `jax.grad`, `jax.jit` or `jax.vmap` the samples must be frozen
  (`Driven(samples=...)`): {doc}`adaptive-sweep-gradient`.
- A material resonance inside the band costs samples: the sweep first solves
  the two grid frequencies either side of each Lorentz term
  ({doc}`dispersive-materials`).

## Solve several frequencies at once

A uniform sweep can run in several {term}`frequency workers <frequency
worker>`. `SolverOptions(frequency_workers=k)` splits the grid into `k`
contiguous blocks and solves each block in a separate process with
`threads // k` of the {term}`solve threads`:

```python
if __name__ == "__main__":
    options = gfem.SolverOptions(frequency_workers=4, threads=16)
    problem = gfem.Driven(frequency=np.linspace(1e9, 100e9, 64))
    result = gfem.solve(model, problem, options=options)
```

- Workers are spawned processes: guard the entry point with
  `if __name__ == "__main__":`, as above.
- One small solve uses many threads poorly, so workers pay off on a many-core
  machine; on a few cores the gain is small.
- Memory grows about `k` times. A worker killed for lack of memory raises
  `RuntimeError`; use fewer workers.
- Each frequency is a full solve, so S agrees with a one-worker run to the
  solver tolerance. Gradients work as usual; the adjoint solves run in the
  main process.
- Sources work (`Driven(sources=...)`). The function of a
  {py:obj}`gfem.IncidentField` must load in a worker: define it at module
  level. A lambda or notebook function raises `ValueError` before any worker
  starts; use `frequency_workers=1` there.
- Wave ports work. The main process solves their port modes once over the
  grid first, so S matches a one-worker run and a {py:obj}`gfem.PortModeError`
  is raised before any worker starts.

## Choose between them

| | Adaptive sweep | Frequency workers |
|---|---|---|
| Pick it for | many frequencies on one machine | a uniform sweep on many cores |
| Accuracy | fields to about `adaptive_tolerance` between samples | every frequency to the solver tolerance |
| Memory | one solve at a time, plus up to two field vectors per excitation and sample | about `k` solves at a time |
| Under `jax.grad` | frozen samples ({doc}`adaptive-sweep-gradient`) | as a uniform sweep |

Workers apply to uniform driven sweeps only. An adaptive sweep, an
{py:obj}`gfem.Eigenmode` or an {py:obj}`gfem.Electrostatic` solve ignores
`frequency_workers`.

## Check your results

- **Adaptive against uniform.** Solve a few frequencies of the grid that are
  not in `result.sweep.sampled_frequencies` with a uniform sweep. S should
  agree to about the square of `adaptive_tolerance`.
- **Workers against one worker.** Solve a short grid with
  `frequency_workers=1` and with your `k`. S should agree to the solver
  tolerance.

How gfem itself is checked: {doc}`../verification`.

## Limits

- A GPU device needs `frequency_workers=1` (`ValueError` otherwise): the
  workers would share one GPU's memory. Run the sweep in one process there.
- A Bloch wave vector takes only the uniform sweep: the adaptive sweep raises
  `NotImplementedError` ({doc}`periodic-cell`). Plain periodicity takes both.
- Across MPI ranks, frequency workers raise
  {py:obj}`gfem.DistributedUnsupportedError`. Use `frequency_workers=1` and
  split the frequencies over communicators ({doc}`distributed-solve`).

## Next

{py:obj}`gfem.Driven`, {py:obj}`gfem.SolverOptions`, {py:obj}`gfem.SweepInfo`,
{doc}`adaptive-sweep-gradient`, {doc}`measurements`, {doc}`gsim`,
{doc}`../capabilities`.
