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
differentiates too, with lumped ports and with wave ports. Its gradient costs
one adjoint solve at that frequency, run at `SolverOptions(adjoint_tolerance=...)`
(default the forward `tolerance`). An adjoint solve that does not converge
raises {py:obj}`gfem.LinearSolverConvergenceError` naming the frequency: loosen
`adjoint_tolerance` or use `linear_solver="direct"` (see {ref}`gradient-errors`).

## Designs at wave ports

S differentiates through {py:obj}`gfem.WavePort`s, lossy ones included,
wherever the design is. A design with elements on the port face changes the
port's mode, and the port follows it: `result.port_modes` and the reference
impedances `s.z_ref` differentiate with the design, and S stays in the
pseudo-waves of the moved mode. Where a port meets a fixed external line,
renormalize S to it ({doc}`port-mode-gradient`). With the design away from
every port face, `s.z_ref` does not depend on it and its gradient is an exact
zero.

- An excited wave port reuses its own solution as the adjoint; a port that is
  not excited takes one adjoint solve per frequency, all such rows in one call.
- Several wave ports on one surface give S between modes; a cross-mode entry
  such as `s["out_TM0", "in_TE0"]` differentiates like any other
  ({doc}`shared-port-surface`).

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
  `Impedance`'s `resistance`, `inductance` and `capacitance`. Build the port
  or the sheet inside the differentiated function, for example
  `gfem.LumpedPort("p1", "port1", "+y", resistance=50.0, inductance=l)`.
- The numbers of a permittivity law differentiate, on both sweeps: see
  {doc}`dispersive-materials`.
- Thermal properties are not read: their gradient is an exact zero.

## In an optimization loop

Run every design of a loop in one Python process: between solves gfem keeps
the wave ports' couplings, the solver plan, the design-independent parts and,
within `SolverOptions(retain_memory=...)`, one factorization per frequency
that a later design may solve on, with GMRES to a twentieth of `tolerance`. A
new process starts without them. Reuse is decided from counts, never from
times; S and gradients depend on the design history only within `tolerance`.
`retain_memory=0` keeps nothing. See {py:obj}`gfem.SolverOptions` for the
budget and what is dropped first.

A finite-difference check across designs (see below) must start each
evaluation afresh: call {py:obj}`gfem.release_memory` before each one, or set
`retain_memory=0`. Otherwise each evaluation's solve error depends on the
design solved before it, and that error does not cancel in the difference.

To see every solve, those under `jax.grad` included, turn on the {term}`log display`
({doc}`watch-a-solve`):

```python
gfem.log.enable()
```

It prints one line per solve: frequency, direction, solver and what it did
(`factorized`, `refactorized`, `reused`). On a result solved outside JAX
transformations, `result.sweep.iterations` and `result.sweep.preconditioner_reused`
show the same per frequency. {py:obj}`gfem.release_memory` frees everything
gfem keeps; call it after a loop. Results you hold are not touched.

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
