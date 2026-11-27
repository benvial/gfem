# Adaptive mesh refinement

Get a converged capacitance, S-parameter, resonance, Q, or the value of your
optimization objective without meshing the whole device finely.
{py:obj}`gfem.refine_mesh` solves, estimates the error in each tetrahedron,
subdivides the tetrahedra holding most of it, and solves again until the
{term}`error estimate` meets a tolerance. It returns a new model on the
refined mesh, its result and the history of the run. Imported physical groups
and walls are kept: gfem only subdivides them.

## Refine a capacitance solve

`stack.msh` has oxide and nitride layers, terminals on its top face and on the
oxide-nitride interface, and the ground on its bottom face.

```python
import gfem

mesh = gfem.Mesh.from_gmsh("stack.msh")
model = gfem.Model(
    mesh,
    materials={"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)},
)
problem = gfem.Electrostatic(
    terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
)

ref = gfem.refine_mesh(model, problem, tolerance=1e-2, max_tetrahedra=200_000)
ref.stop  # "tolerance": the estimate met 1e-2
ref.result.capacitance  # F, solved on ref.model.mesh
for cycle in ref.cycles:  # cycle 0 is the imported mesh
    print(cycle.tetrahedra, cycle.dofs, cycle.estimate)
```

- Each {term}`refinement cycle` solves, estimates, and stops at or below
  `tolerance`; otherwise it splits the tetrahedra holding half the squared
  error and carries the model's design over.
- The default {py:obj}`gfem.FieldEstimator` is relative to the field's energy
  norm: `tolerance=1e-2` is a field about 1 % off its value on an infinitely
  fine mesh of the same geometry.
- `ref.model` is a new model: use it for any later solve.
- On a cap with the estimate above `tolerance`, the loop warns with
  {py:obj}`gfem.MeshRefinementNotConvergedWarning`, `ref.converged` is `False`
  and `ref.stop` names the cause.
- Uniform driven sweeps and eigenmode problems refine the same way, on the
  worst frequency or mode.

## Read the estimate

`ref.estimate.value` is compared with `tolerance`; it is the largest of
`ref.estimate.values`. `ref.estimate.indicators` holds each tetrahedron's
share. {py:obj}`gfem.error_estimate` does the same for a result you solved:

```python
result = gfem.solve(model, problem)
estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
estimate.value, estimate.indicators  # relative energy error, per tetrahedron
```

A driven result needs `keep_fields=True`. `ref.estimate.kind` names what was
estimated; its measured range is on the verification page
({ref}`verification-mesh-refinement`).

## Refine for one quantity

When one number matters, a {py:obj}`gfem.QuantityEstimator` weights the error
by its effect. Its keyword is the result attribute, indexed the same way:

```python
problem = gfem.Driven([3e9, 8e9])  # every port excited
estimator = gfem.QuantityEstimator(s=("p2", "p1"))  # S21: result.s["p2", "p1"]
ref = gfem.refine_mesh(
    model, problem, estimator=estimator, tolerance=1e-3, max_tetrahedra=200_000
)
s21 = ref.result.s["p2", "p1"]
for f, s, e in zip(ref.result.frequency, s21, ref.estimate.values, strict=True):
    print(f"{f / 1e9:.0f} GHz: S21 = {complex(s):.4f} ± {e:.1e}")
```

`ref.estimate.values[i]` is the error at `ref.result.frequency[i]`: absolute
for an S entry, relative for a capacitance or a frequency (multiply by `abs(C)`
or `abs(f)`). It has no sign or phase.

The other quantities:

```python
gfem.QuantityEstimator(capacitance=("top", "top"))  # result.capacitance[0, 0]
gfem.QuantityEstimator(omega=0)  # result.omega[0], an eigenmode problem
gfem.QuantityEstimator(q=0)  # result.q[0]
gfem.QuantityEstimator(omega=(0, 1))  # the mean of result.omega[0] and [1]
```

- **Capacitance**: a diagonal entry never lies below its value on the meshed
  geometry, $C_h \ge C$.
- **S entry**: the source port must be excited; an unexcited read port solves
  its {term}`adjoint dual`, one more right-hand side per solve.
- **Mode**: a mode within `SolverOptions.cluster_rtol` of another raises
  {py:obj}`gfem.DegenerateModeError`: estimate the cluster mean `omega=(i, j)`
  and ask for one mode more than the cluster holds.
- **Q**: a lossless mode raises `ValueError`.
- `RefinementCycle.value` holds the quantity on each mesh, to plot against
  `cycle.dofs`.

(refine-for-an-objective)=
## Refine for an objective

`QuantityEstimator(objective=J)` takes `J(result, model)`, the function you
differentiate, and estimates its error in its own units:

```python
import jax.numpy as jnp

import gfem


def loss(result, model):  # the same function the gradient uses
    return jnp.mean(1 - jnp.abs(result.s["p2", "p1"]) ** 2)


estimator = gfem.QuantityEstimator(objective=loss)
ref = gfem.refine_mesh(
    model,
    gfem.Driven([3e9, 8e9]),
    estimator=estimator,
    tolerance=1e-3,
    max_tetrahedra=200_000,
)
ref.estimate.value  # in loss's units
ref.estimate.kind  # "objective of s" (lossless lumped ports: ", lossless rewrite")
ref.cycles[-1].value  # loss on the last mesh
```

- **What `J` may read.** S and its networks, electrostatic and eigenmode
  results, and fields only through {py:obj}`gfem.power_flow` and
  {py:obj}`gfem.region_mean`. Anything else raises
  {py:obj}`gfem.MeshRefinementUnsupportedError`, whose `feature` names it;
  `Field.at` has no estimate ({doc}`measurements`).
- **Tolerance in your units.** `tolerance=` is required and absolute.
- **Modes.** `J` sees the first mesh's mode order; `RefinementCycle.modes`
  maps it to each mesh's.

## Refine for several quantities

A tuple of estimators refines one mesh for all of them, each to its own
tolerance:

```python
estimators = (
    gfem.QuantityEstimator(capacitance=("a", "a")),
    gfem.QuantityEstimator(capacitance=("a", "b")),
)
ref = gfem.refine_mesh(
    model,
    problem,
    estimator=estimators,
    tolerance=(1e-3, 5e-3),
    max_tetrahedra=200_000,
)
c_aa, c_ab = ref.estimate  # one ErrorEstimate per estimator, in order
ref.cycles[-1].value  # (C_aa, C_ab) on the last mesh
```

- `ref.estimate` and `cycle.value` are tuples in the estimators' order;
  `ref.converged` means every one met its tolerance. A float `tolerance`
  applies to all.

## Refine between optimization stages

`refine_mesh` is not differentiable. Refine between stages on the current
design, then differentiate {py:obj}`gfem.solve` on the refined model: the
steps of a density design's `beta`, the points where a shape design would
remesh ({doc}`remesh-and-restart`).

```python
import jax
import jax.numpy as jnp

from gfem import design

# material_of, loss_of_result, make_loss, make_optimizer and run_stage are your
# own code; loss_of_result(result, model) is the objective make_loss calls.
for beta in (2, 4, 8, 16):
    # 1. The model holding the current physical design, with concrete values.
    current = model.with_material("design", material_of(rho, beta))
    # 2. Refine it, with a cap.
    ref = gfem.refine_mesh(
        current,
        problem,
        estimator=gfem.QuantityEstimator(objective=loss_of_result),
        tolerance=TOL,
        max_cycles=3,
        max_tetrahedra=200_000,
    )
    model = ref.model
    # 3. Rebuild every helper on the refined mesh.
    grid = design.LayerGrid(model.mesh, "design", pitch=0.5e-3)
    blur = design.helmholtz_filter(grid, radius=1.5e-3)
    # 4. Rebuild the objective and its jitted gradient on the refined model.
    loss = make_loss(model, grid, blur, beta)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    # 5. Restart the optimizer from the current parameters, its state reset.
    optimizer = make_optimizer(x0=rho)
    rho = run_stage(optimizer, value_and_grad, STEPS_PER_BETA)
```

- **Rebuild the jitted objective**: a closure keeps the model it captured,
  silently. **Reset the optimizer**: the objective jumps at a new mesh.
- **Carry your parameters.** A `LayerGrid` keeps its pixels; a density per
  element goes through `ref.transfer(values, "design")`.
- **Shape designs.** Rebuild `gfem.shape.boundary_morph(ref.model, surface,
  where=...)` and map parameters relative to the current ones. A helper left
  on the old mesh raises `ValueError`; a mesh from an old morph is not caught.
- Never call `refine_mesh` inside the objective: it raises
  {py:obj}`gfem.NonDifferentiableError`. Refine once more on the final design.

## Options

- **`max_tetrahedra`**: set it on every run; 10 cycles can grow the mesh 10 to
  30 times.
- **`tolerance`**: at least 10 times the solve's own
  (`SolverOptions.tolerance`, `eigen_tolerance`), or `ValueError`.
- **Memory.** Cycles keep every port's field at every frequency: refine on a
  few, then sweep on `ref.model`.

## Check your results

- `ref.cycles[k].value` settles as the estimate falls.
- Solve `ref.model` with `SolverOptions(order=3)`: the quantity moves by the
  estimate or less.

How gfem checks the estimates against known answers:
{ref}`verification-mesh-refinement`.

## Limits

- **An estimate, not a bound.** The field estimator measured between
  {gfem-effectivity}`field low` and {gfem-effectivity}`field high` times the
  true error, the quantity estimator between {gfem-effectivity}`quantity low`
  and {gfem-effectivity}`quantity high` at `order=2`. An objective at its
  target measured up to {gfem-effectivity}`objective at its target high`:
  refinement then stops late, never early. Ranges per kind, and the
  configurations that fall outside them (a lumped port across a loss jump, a
  permittivity law with $-20 < \operatorname{Re} \varepsilon < 0$), are on the
  verification page; mesh those regions finer by hand.
- **Walls stay.** Facets never move onto a curve, and a {term}`curved mesh`
  keeps its own quadratic walls: mesh curved surfaces finer in your mesher.
- **Refusals.** `refine_mesh` raises
  {py:obj}`gfem.MeshRefinementUnsupportedError` before any solve on an
  adaptive sweep, a port mode, periodic, thermal or sourced problem, a model
  with a {term}`perfectly matched layer`, and `order=1`. It runs on one
  process: refine, then solve `ref.model` on ranks
  ({doc}`distributed-solve`).

## Next

{py:obj}`gfem.refine_mesh`, {py:obj}`gfem.error_estimate`,
{py:obj}`gfem.FieldEstimator`, {py:obj}`gfem.QuantityEstimator`,
{py:obj}`gfem.MeshRefinement`, {py:obj}`gfem.RefinementCycle`,
{py:obj}`gfem.region_mean`, {doc}`measurements`, {doc}`remesh-and-restart`,
{doc}`robust-optimization`, {doc}`frequency-sweep`.
