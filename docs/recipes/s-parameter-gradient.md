# Differentiate S-parameters and driven fields

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
differentiates too:

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

## Keep the design off wave ports

S differentiates through {py:obj}`gfem.WavePort`s, lossy ones included, as
long as the design stays off the port faces. A wave port's mode comes from
its cross-section (the materials of the elements beside the face), and gfem
treats it as a constant of the solve. A perturbed material whose region has
an element on a wave port face therefore raises
{py:obj}`gfem.PortGeometryError` when `jax.grad` is called; a forward solve of
the same model is fine. Give the elements next to each port a region of their
own with a constant {py:obj}`gfem.Material`, and put the design further in:

```python
model = gfem.Model(
    mesh,
    materials={"lead": gfem.Material(eps_r=4.0), "core": gfem.MaterialField(eps_r=eps)},
    boundaries=[gfem.PEC("ground"), gfem.PEC("signal")],
    ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
)
```

- The port modes in `result.port_modes`, and the reference impedances
  `s.z_ref` (the ports' `Z_PV`), are then constants: their gradient is an
  exact zero.
- A lossless wave port reuses its own solution as the adjoint, as a lumped
  port does.
- A lossy port's mode is complex (conductor or dielectric loss in the
  cross-section), so its S row is not its own adjoint. Each frequency with a
  lossy port in the objective takes one adjoint solve, all such rows in one
  call.
- Several wave ports on one surface, one mode each, give S between modes. A
  cross-mode entry such as `s["out_TM0", "in_TE0"]` differentiates like any
  other S entry, and reading the row of a mode left out of
  `Driven(excite=...)` costs one adjoint solve per frequency. See
  {doc}`shared-port-surface`.

## Other inputs

- `Driven(sweep="adaptive")` differentiates too, with its reduced basis held
  fixed and its samples frozen (`Driven(samples=...)`): see
  {doc}`adaptive-sweep-gradient`.
- The mesh node coordinates differentiate as long as no port moves: see
  {doc}`shape-gradient`.
- The fill of a {py:obj}`gfem.SheetField` on a `Conductor` or `Impedance`
  differentiates: see {doc}`sheet-design`.
- Thermal properties are not read: their gradient is an exact zero.

## Check your results

- Compare the gradient with a central difference of your objective along a
  random direction `v`: `(f(p + h v) - f(p - h v)) / (2 h)` against
  `grad @ v`. Their difference shrinks as `h` shrinks, until the solver
  tolerance limits it.

How gfem itself is checked: {doc}`../verification`.

## Limits

- Port and boundary parameters (lumped `R`, `L`, `C`, `Impedance`,
  `Conductor` conductivity) and the frequencies are read but not
  differentiable: a perturbed one raises `gfem.NonDifferentiableError` when
  `jax.grad` is called. Keep them constant inside the differentiated
  function, and compare separate solves to see their effect.

## Next

{py:obj}`gfem.Driven`, {py:obj}`gfem.SolverOptions`; {doc}`capacitance-gradient`,
{doc}`adaptive-sweep-gradient`, {doc}`shape-gradient`; {doc}`../capabilities`.
