# Sweep many frequencies

Get S and the fields of a driven model over a broadband grid without paying
one full solve per frequency at a time. A {py:obj}`gfem.Driven` sweep is
uniform by default: every frequency is a full solve. Two options make a long
sweep faster: the {term}`adaptive sweep`, which solves only a few frequencies,
and {term}`frequency workers <frequency worker>`, which solve several
frequencies at once on a many-core machine.

## S at 300 frequencies from a few solves

`model` is any driven model with two ports `p1` and `p2`, for example the one
{doc}`gsim` builds. Ask for the adaptive sweep with `sweep="adaptive"`:

```python
import numpy as np

import gfem

problem = gfem.Driven(frequency=np.linspace(1e9, 100e9, 300), sweep="adaptive")
result = gfem.solve(model, problem)
s21 = result.s["p2", "p1"]  # all 300 frequencies
result.sweep.sampled_frequencies  # the frequencies solved in full, Hz
```

- The sweep solves a few sample frequencies in full, every excited port at
  once. It reads S and the kept fields at every frequency of the grid from a
  {term}`reduced model` built on those solutions.
- It pays off above about 20 frequencies. Below that, keep the uniform sweep.
- `result.sweep.solved` marks, frequency by frequency, the ones solved in
  full. The rest come from the reduced model.
- `result.sweep.converged` says whether the last samples agreed with the
  reduced model to `SolverOptions.adaptive_tolerance`.
- A sweep that reaches `SolverOptions.adaptive_max_samples` first warns with
  {py:obj}`gfem.SweepNotConvergedWarning` and reports `converged=False`. Raise
  `adaptive_max_samples`.
- On a short grid the sweep may run out of frequencies. It then solves every
  one in full, as a uniform sweep does, and reports `converged=True`.

## Set the accuracy of an adaptive sweep

The speed costs field accuracy. A uniform sweep's fields are as accurate as
the linear solve (`SolverOptions.tolerance`). At the frequencies an adaptive
sweep did not solve, its fields have a relative error of about
`adaptive_tolerance` or less (`1e-3` by default), and S has about the square of
that. Measurements read the fields ({doc}`measurements`), so a probe or an
interface participation on an adaptive result carries the same error.

Tighten it at the cost of more samples:

```python
options = gfem.SolverOptions(adaptive_tolerance=1e-5)
result = gfem.solve(model, problem, options=options)
```

- `gfem.SolverOptions.preset("accurate")` also sets `adaptive_tolerance=1e-5`,
  with a higher element order and a tighter linear solve.
- `adaptive_tolerance` below 10 times `tolerance` raises `ValueError`: the
  sweep cannot be more accurate than the solves it samples. Tighten `tolerance`
  with it.
- Where the fields matter most, keep the uniform sweep.
- Under `jax.grad`, `jax.jit` or `jax.vmap` an adaptive sweep needs its
  samples frozen (`Driven(samples=...)`): see {doc}`adaptive-sweep-gradient`.

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

- The workers are spawned processes, so a script that sets
  `frequency_workers` guards its entry point with
  `if __name__ == "__main__":`, as above.
- One small solve uses many threads poorly, so workers pay off on a many-core
  machine. On a few cores the gain is small.
- Memory grows about `k` times: each worker holds one solve. A worker killed
  for lack of memory raises `RuntimeError`; use fewer workers.
- Each frequency is still a full solve, so S agrees with a one-worker run to
  the solver tolerance.
- Gradients work as usual. The adjoint solves run in the main process.

## Choose between them

| | Adaptive sweep | Frequency workers |
|---|---|---|
| Pick it for | many frequencies on one machine | a uniform sweep on many cores |
| Accuracy | fields to about `adaptive_tolerance` between samples | every frequency to the solver tolerance |
| Memory | one solve at a time | about `k` solves at a time |
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
- A uniform sweep with wave ports over more than one frequency raises
  `NotImplementedError` with `frequency_workers` above 1. Use
  `frequency_workers=1` or the adaptive sweep.
- A Bloch wave vector takes only the uniform sweep: the adaptive sweep raises
  `NotImplementedError` ({doc}`periodic-cell`). Plain periodicity takes both.
- Across MPI ranks, frequency workers raise
  {py:obj}`gfem.DistributedUnsupportedError`. Use `frequency_workers=1` and
  split the frequencies over communicators ({doc}`distributed-solve`).

## Next

{py:obj}`gfem.Driven`, {py:obj}`gfem.SolverOptions`, {py:obj}`gfem.SweepInfo`,
{doc}`adaptive-sweep-gradient`, {doc}`measurements`, {doc}`gsim`,
{doc}`../capabilities`.
