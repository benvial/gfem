# S-parameter gradients

Optimize a device's transmission, reflection or a field quantity by taking
`jax.grad` of an objective built from S or from kept fields, with respect to
the material in each element of a design region. The gradient reaches the
per-element `eps_r` (complex where lossy), `mu_r`, `tan_delta` and
`conductivity` of every {py:obj}`gfem.MaterialField`. Batches, memory,
`stop_gradient` and errors work as for a capacitance: see
{doc}`capacitance-gradient`.

## A passband objective

The mesh `filter.msh` holds a filter on a substrate, with a lumped port at
each end.

```python
import jax
import jax.numpy as jnp

import gfem

mesh = gfem.Mesh.from_gmsh("filter.msh")
model = gfem.Model(
    mesh,
    materials={"substrate": gfem.Material(eps_r=4.4)},
    boundaries=[gfem.PEC("ground"), gfem.Absorbing("outer")],
    ports=[
        gfem.LumpedPort("p1", "port1", "+z"),
        gfem.LumpedPort("p2", "port2", "+z"),
    ],
)
problem = gfem.Driven(frequency=[4e9 + 0.1e9 * k for k in range(21)])


def passband_loss(eps):
    """Mean |S21|^2 deficit over the band, for a permittivity per element."""
    m = model.with_material("substrate", gfem.MaterialField(eps_r=eps))
    s = gfem.solve(m, problem).s
    return jnp.mean(1 - jnp.abs(s["p2", "p1"]) ** 2)


n = len(mesh.elements("substrate"))
value, grad = jax.value_and_grad(passband_loss)(jnp.full(n, 4.4))  # grad: (n,)
```

- Anything computed from `result.s` in `jnp` differentiates: `s.db()`,
  `s.y()`, `s.z()`, `s.renormalize(...)`. The reference impedances `s.z_ref`
  are the port resistances, constants of the materials.
- The gradient needs no extra solve when every port the objective reads is
  excited (the default). By reciprocity each port's solution is its own
  adjoint, and the backward pass (the part of `jax.grad` that runs after the
  solve) is one element-wise integral per frequency.
- Reading the row of a port left out of `Driven(excite=...)` costs one
  adjoint solve per frequency.
- For a field with `axes` (an {term}`anisotropic material field`) the
  gradient has the field's shape, one value per principal axis; see
  {doc}`anisotropic-materials`.

## Differentiate a kept field

A kept field (`Driven(keep_fields=...)`, then `result.e_field(f, port)`)
differentiates too, with lumped ports and with wave ports:

- Its gradient costs one adjoint solve at that frequency, with all the
  right-hand sides it needs in one call.
- The adjoint solve runs at `SolverOptions(adjoint_tolerance=...)`, by
  default the forward `tolerance`. With an iterative solver, an adjoint solve
  that does not converge raises {py:obj}`gfem.LinearSolverConvergenceError`
  while the gradient is computed, naming the frequency. Loosen
  `adjoint_tolerance` or use `linear_solver="direct"` (see
  {ref}`gradient-errors`).
- gfem keeps the adjoint's operator, within `retain_memory`, for each later
  `jax.vjp` or `jax.grad` call on the same solve.

## Designs at wave ports

S differentiates through {py:obj}`gfem.WavePort`s, lossy ones included,
wherever the design is. A wave port's mode comes from its cross-section, the
materials of the elements beside the face. A design with elements on the face
changes the mode, and the port follows it: `result.port_modes` and the
reference impedances `s.z_ref` (the ports' `Z_PV`) differentiate with the
design, and S stays in the pseudo-waves of the moved mode. Where a port meets
a fixed external line, renormalize S to it; see {doc}`port-mode-gradient`.
With the design away from every port face, the port modes and `s.z_ref` do
not depend on it: their gradient is an exact zero, and no 2D work runs.

- An excited wave port reuses its own solution as the adjoint, as a lumped
  port does, lossy or not: by reciprocity the biorthogonal S functional is
  the port's own load scaled. A port that is not excited has no field of its
  own; each frequency with such a port in the objective takes one adjoint
  solve, all such rows in one call.
- Several wave ports on one surface, one mode each, give S between modes. A
  cross-mode entry such as `s["out_TM0", "in_TE0"]` differentiates like any
  other S entry, and reading the row of a mode left out of
  `Driven(excite=...)` costs one adjoint solve per frequency. See
  {doc}`shared-port-surface`.

## Other inputs

- `Driven(sweep="adaptive")` differentiates too, with its reduced basis held
  fixed and its samples frozen (`Driven(samples=...)`): see
  {doc}`adaptive-sweep-gradient`.
- The mesh node coordinates differentiate, a wave port's face included when
  it is named in `sliding_ports`: see {doc}`shape-gradient`.
- The fill of a {py:obj}`gfem.SheetField` on a `Conductor` or `Impedance`
  differentiates: see {doc}`sheet-design`.
- The {term}`lumped values <lumped value>` that enter only the operator
  differentiate: a `LumpedPort`'s `inductance` and `capacitance`, and an
  `Impedance`'s `resistance`, `inductance` and `capacitance`, on both sweeps.
  Build the port or the sheet inside the differentiated function, for example
  `gfem.LumpedPort("p1", "port1", "+y", resistance=50.0, inductance=l)`. An
  `Impedance` beside a wave port's face differentiates too: the port mode
  reads it there, and follows it.
- The numbers of a permittivity law differentiate, on both sweeps: see
  {doc}`dispersive-materials`.
- Thermal properties are not read: their gradient is an exact zero.

## In an optimization loop

Run every design of a loop in one Python process. Between solves gfem keeps what
the next design with the same mesh, ports and frequencies can use, and a new
process starts without it:

- with any linear solver, the wave ports' couplings;
- with `linear_solver="direct"` on the CPU in one process, the solver plan of the
  last design (its ordering and analysis), the assembled parts that do not
  depend on the design and, while `SolverOptions(retain_memory=...)` has room,
  one factorization per frequency (per sample of an adaptive sweep with frozen
  samples) from the second design on, with or without the `mkl` extra (a single
  solve or gradient keeps one). The next design at that frequency
  solves on it, with GMRES to a twentieth of `tolerance`, while the counted
  costs (the factorization's size and work against the GMRES iterations) say
  that is cheaper than factorizing again. Small models, where factorizing is
  cheap, factorize every design. A periodic cell with a nonzero wave vector
  always factorizes again;
- with `linear_solver="bddc_gmres"` or `"pmg_gmres"` in one process, each design's first
  frequency of a uniform sweep starts its iterations from the previous
  design's solution there.

On the CPU in one process, the default `linear_solver="auto"` already picks
`"direct"` for a driven problem whose factorization fits in half the memory the process may use, by an estimate
of about 48 KiB per unknown. Above it picks `"pmg_gmres"` where that solver's
own estimate fits all of the memory, else `"bddc_gmres"` (see
{doc}`large-driven-model`). The estimate is
cautious: on a machine where a larger model still fits, an explicit
`linear_solver="direct"` lifts that guard.

gfem decides what to reuse from counts (factor sizes, factorization work,
iterations), never from times, so a script run again with the same thread count
and an explicit `retain_memory` reuses at the same designs. S and gradients
depend on the design history only within `tolerance`: a design solved on an
earlier factorization, or started from the previous design's solution, agrees
with one solved afresh to within `tolerance`. `retain_memory=0` keeps nothing
and factorizes every design.

A finite-difference check across designs (see below) must start each
evaluation afresh: call {py:obj}`gfem.release_memory` before each one, or set
`retain_memory=0`. Otherwise each evaluation's solve error depends on the
design solved before it, and that error does not cancel in the difference.

The default `retain_memory` is half of the memory the process may use. It also
keeps a large model's last factorization, even over that half, while the
process's peak memory leaves a tenth of that memory free, so the decision can
change with the measured peak. An explicit `retain_memory` is a hard bound and
the decision no longer depends on the peak. To keep the factorizations of a
large model under an explicit value, set it at least as large as one solve's
peak memory: the peak barely grows, but that memory stays held between
designs. On a machine shared with other memory-heavy processes, set it
explicitly. With less room, gfem drops the iterative solvers' initial guesses
first, then the factorizations, then the parts, then the plans. Only the time changes: S and
the gradient agree with a solve that keeps nothing to within `tolerance`.

To see what each design reused, read `result.sweep.iterations` (the Krylov
iterations of each excited port, per frequency; empty for a solve on fresh
factors) and `result.sweep.preconditioner_reused` (per frequency) on a result
solved outside JAX transformations; under them both are `None`. To see every
solve, those under `jax.grad` included, turn on the {term}`log display`
({doc}`watch-a-solve`):

```python
gfem.log.enable()
```

It prints one line per record of the logger `"gfem.solver"` on stderr. Each
record names the frequency, the direction (`forward` or `backward`), the
solver and what it did (`factorized`, `refactorized`, `reused`), and, for a
design that factorized again, its `reason`. When `"auto"` resolves, one more
record says which solver it chose and why.

{py:obj}`gfem.release_memory` frees everything gfem keeps, at once: call it after
a loop, before another study in the same process. Results you hold are not
touched.

## Check your results

- Compare the gradient with a central difference of your objective along a
  random direction `v`: `(f(p + h v) - f(p - h v)) / (2 h)` against
  `grad @ v`. Their difference shrinks as `h` shrinks, until the solver
  tolerance limits it. Call {py:obj}`gfem.release_memory` before each
  evaluation of `f`, as in an optimization loop above.

How gfem itself is checked: {doc}`../verification`.

## Limits

- A lumped port's resistance (its reference impedance), a `Conductor`'s
  conductivity and thickness, a wave port's `offset` and the frequencies are
  read but not differentiable: a traced one raises
  `gfem.NonDifferentiableError`. Keep them constant inside the differentiated
  function, and compare separate solves to see their effect.

## Next

{py:obj}`gfem.Driven`, {py:obj}`gfem.SolverOptions`; {doc}`capacitance-gradient`,
{doc}`adaptive-sweep-gradient`, {doc}`shape-gradient`; {doc}`../capabilities`.
