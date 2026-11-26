# Adaptive mesh refinement

Get a converged capacitance, S-parameter, resonance, Q, or the value of your
optimization objective without meshing the whole device finely. {py:obj}`gfem.refine_mesh` solves, estimates the error
of the solution in each tetrahedron, subdivides the tetrahedra holding most
of it, and solves again, until the {term}`error estimate` meets a tolerance.
It returns a new model on the refined mesh, its result, and the history of
the run. The mesh you import keeps its physical groups and its facets: gfem
only subdivides them.

## Refine a capacitance solve

The mesh `stack.msh` has an oxide and a nitride layer, terminals on its top
face and on the oxide-nitride interface, and the ground on its bottom face.

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

- Each {term}`refinement cycle` solves, estimates the error, stops if the
  estimate is at or below `tolerance`, marks the fewest tetrahedra holding
  half the squared error (`error_fraction=0.5`), splits them at the
  midpoint of their longest edge, splits their neighbours until no hanging
  node remains, and carries the model's design to the new tetrahedra.
- The default estimator, {py:obj}`gfem.FieldEstimator`, estimates the
  field's error in the energy norm, relative to the field's energy norm:
  `tolerance=1e-2` asks for a field about 1 % off its value on an infinitely
  fine mesh of the same geometry. No extra solve is needed.
- `ref.model` is a new model: use it, not `model`, for any later solve.
  `ref.result` is its result, and `ref.estimate` that result's estimate.
- When the loop stops on `max_cycles` or `max_tetrahedra` with the estimate
  above `tolerance`, it warns with
  {py:obj}`gfem.MeshRefinementNotConvergedWarning`, `ref.converged` is
  `False`, and `ref.stop` says which cap stopped it. A solve that fails
  after the first mesh stops the loop with `stop="solve failed"` and keeps
  the last solved mesh. An estimate that is NaN or infinite counts as above
  `tolerance`: the loop stops with `stop="non-finite estimate"` and warns.
- A driven problem on a uniform sweep (lumped or wave ports) and an
  eigenmode problem refine the same way. The estimate of a sweep is its
  worst frequency's, of an eigenmode solve its worst mode's.

## Read the estimate

`ref.estimate.value` is the number compared with `tolerance`, the largest
of `ref.estimate.values`: one value per frequency of a driven sweep, per mode
of an eigenmode solve, and one otherwise. `ref.estimate.indicators` holds one
value per tetrahedron of `ref.model.mesh`, the share of the error that
marking reads. {py:obj}`gfem.error_estimate` gives the same estimate for a
result you solved yourself, without refining:

```python
result = gfem.solve(model, problem)
estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
estimate.value, estimate.indicators  # relative energy error, per tetrahedron
```

A driven result must keep its fields for this: solve it with
`gfem.Driven(..., keep_fields=True)`. `refine_mesh` keeps them itself.

`ref.estimate.kind` names what was estimated (`"field"`, `"s"`, `"q"`,
`"objective of absorbed power"`, ...); the verification page gives the
range measured for each kind ({ref}`verification-mesh-refinement`).

## Refine for one quantity

When one number matters, a capacitance entry, an S entry or one mode's
frequency, refine for it with a {py:obj}`gfem.QuantityEstimator`. It weights
each tetrahedron's field error by its effect on that quantity, from the
fields already solved, so the mesh grows where the quantity is sensitive
and the estimate is that quantity's error. Its keyword is the result
attribute that holds the quantity, indexed the same way:

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

`ref.estimate.values[i]` is the estimated error of the quantity at
`ref.result.frequency[i]`. For an S entry it is absolute, so
`S21 = {s:.4f} ± {e:.1e}`. For a capacitance or a frequency it is
relative: multiply by `abs(C)` or `abs(f)`. It has no sign or phase: it
tells you how far the quantity may be from the converged answer, not in
which direction. gfem does not add it to the quantity.

The other quantities:

```python
gfem.QuantityEstimator(capacitance=("top", "top"))  # result.capacitance[0, 0]
gfem.QuantityEstimator(omega=0)  # result.omega[0], an eigenmode problem
gfem.QuantityEstimator(q=0)  # result.q[0]
gfem.QuantityEstimator(omega=(0, 1))  # the mean of result.omega[0] and [1]
```

- **Capacitance**: the entry `capacitance[i, j]`, named by its two
  terminals, or by one terminal twice for a diagonal entry. The estimate is
  relative to that entry. A diagonal entry never lies below its value on
  the meshed geometry, `C_h >= C`: the difference is the energy of the
  potential's error, which is never negative, so mesh refinement lowers it.
  An off-diagonal entry may lie on either side.
- **S entry**: the source port must be excited (`gfem.Driven(excite=...)`);
  the other may be any port: when it is not excited, its {term}`adjoint dual`
  is solved, one more right-hand side on each of `refine_mesh`'s solves. The
  estimate is absolute.
- **Mode**: `omega=0` names the first mode of the input mesh's solve; on
  each refined mesh the estimate follows the mode whose `omega` is nearest
  the previous one. A mode with another within `SolverOptions.cluster_rtol`
  of it raises {py:obj}`gfem.DegenerateModeError`: estimate the cluster's
  mean instead, `omega=(i, j)`. The estimate is relative to `omega`. On a
  model with loss it covers the complex `omega`.
- **Q**: `q=0` is the mode's Q, followed as `omega=`; the estimate is
  relative to Q. A lossless mode raises `ValueError`.
- **Degenerate modes**: `omega=(0, 1)` names a whole degenerate cluster
  (`result.cluster`); the quantity is the members' mean `omega`, followed
  on each refined mesh as the modes nearest the previous mean. Ask for at
  least one mode more than the cluster holds (`gfem.Eigenmode(n_modes=...)`).
- `RefinementCycle.value` holds the quantity on each mesh (a tuple per
  frequency for an S entry), so the quantity against `cycle.dofs` plots
  without another solve.

(refine-for-an-objective)=
## Refine for an objective

Refine for the function you optimize: `QuantityEstimator(objective=J)`
takes `J(result, model)`, the same function you differentiate, and
estimates its error.

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

- **What `J` may read.** S and its networks (`db`, `y()`, `z()`,
  `renormalize`, `with_offsets`); `capacitance`, `mutual`, `inverse` and
  `energy` (per region) of an electrostatic result; `omega`, `frequency`,
  `q` and `bulk_participation` of an eigenmode result; and fields only through
  {py:obj}`gfem.power_flow` (the power absorbed in a side with no port and
  only volume losses) and {py:obj}`gfem.region_mean`. gfem traces `J` once
  to find what it reads, and anything else raises
  {py:obj}`gfem.MeshRefinementUnsupportedError`, whose `feature` names it: a
  point value, a branch on a value it reads (`max`, `min`, `where`), a
  mixture of kinds, a custom derivative of your own.
- **Tolerance in your units.** The estimate is absolute, in the objective's
  units: give `tolerance=` in them. `refine_mesh` raises without it.
- **Point values.** `Field.at` has no estimate: use
  `gfem.region_mean(field, "small_region")` over a small region instead
  ({doc}`measurements`).
- **Lossless lumped-port models.** On a lossless model with lumped ports, an
  objective of S is estimated through its {term}`lossless rewrite`, which
  is tighter; `ref.estimate.kind` says so (`"objective of s, lossless
  rewrite"`).
- **Modes.** On refined meshes, `J` sees the modes in the first mesh's
  order: each mode it reads is followed, and `RefinementCycle.modes` maps
  each first-mesh index to the index followed on that mesh.
- **One-argument objectives.** `lambda result, model: J(result)` adapts an
  objective that does not need the model.

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

- The record mirrors the tuple: `ref.estimate`, `cycle.estimate` and
  `cycle.value` are tuples in the estimators' order.
- Marking weighs each quantity's indicators by its tolerance, over the
  quantities still above it, and `ref.converged` means every one met its
  tolerance.
- A float `tolerance` applies to every estimator, an objective included;
  `None` gives each 1e-2 and raises when the tuple holds an objective.

## Refine between optimization stages

`refine_mesh` is not differentiable: it builds a new mesh, which `jax.grad`
cannot follow. Refine between stages of an optimization, on the current
design, then differentiate {py:obj}`gfem.solve` on the refined model. A
density design already has stage boundaries, the steps of its projection
sharpness `beta`; a shape design has the points where you would otherwise
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

- **Rebuild the jitted objective.** A function jitted as a closure over a
  model keeps the model it captured when it was traced. Rebinding the name
  `model` leaves the old mesh in use, with no error.
- **Reset the optimizer.** The discrete objective jumps a little at a new
  mesh, so L-BFGS curvature pairs and MMA asymptotes from the old mesh no
  longer apply. Compare objective values only within one mesh.
- **Carry your design parameters.** A `LayerGrid` pixel array keeps its
  shape: bisection only adds midpoints, so the rebuilt grid has the same
  pixels. A density with one value per element goes through
  `ref.transfer(values, "design")`, which gives each refined element the
  value of the input element it lies in; rebuild
  `design.helmholtz_filter(ref.model.mesh, "design", radius=r)` on it. The
  model's own `MaterialField` values and sheet fills are carried for you.
- **Shape designs.** Rebuild `gfem.shape.boundary_morph(ref.model, surface,
  where=...)` with the same selection, and map your parameters relative to
  the current ones `q_k`: the refined mesh already has the geometry of
  `q_k`. A helper left on the old mesh gives arrays of the old length,
  which the model rejects with `ValueError`. A mesh from an old morph passed
  to `ref.model.replace(mesh=...)` is not caught: always build meshes from
  the morph rebuilt on `ref.model`.
- **Never call `refine_mesh` inside the objective.** Under `jax.jit` it
  raises {py:obj}`gfem.NonDifferentiableError`, and so does any traced
  value in its model or problem.
- **Refine once more on the final design** before you report its
  quantity, so the reported value comes with an error estimate on the
  design you deliver.
- **Refine for the objective itself**, `QuantityEstimator(objective=...)`,
  written as a function of the result and model that your loss also calls,
  so the mesh grows where the objective needs it and the loop stops when
  its error is below a tolerance in its units. Where it raises
  {py:obj}`gfem.MeshRefinementUnsupportedError` (see Limits), use
  {py:obj}`gfem.FieldEstimator`.
- An objective near its optimum, or a reflection near zero, is estimated
  pessimistically: refinement stops late, never early (see Limits).

## Options

- **`max_tetrahedra`**: set it on every run. Each cycle grows the mesh by
  about 1.6 times on the first cycle and 1.25 times on later ones, so the
  default 10 cycles can grow a mesh 10 to 30 times. A refined mesh above the
  cap is discarded and the loop stops on the last solved one.
- **`max_cycles`** (default 10): how many times the mesh may be refined; 0
  solves and estimates only.
- **`error_fraction`** (default 0.5): the share of the summed squared
  indicators the marked tetrahedra hold. A larger fraction refines more per
  cycle, in fewer cycles.
- **`tolerance`**: `None` (the default) is 1e-2 for every estimator but an
  objective, which needs its own; a tuple gives one per estimator. Each
  must be at least 10 times the solve's tolerance carried into its units
  (`SolverOptions.tolerance`, `eigen_tolerance` for an eigenmode problem),
  or `refine_mesh` raises `ValueError`.
- **Memory.** Every cycle of a driven problem keeps the field of every
  excited port at every frequency, and `ref.result` keeps those of the last
  mesh: memory grows with ports × frequencies × DOF. Refine on the few
  frequencies that matter, then solve the sweep on `ref.model`.

## Check your results

- The quantity on each cycle, `ref.cycles[k].value`, should settle as the
  estimate falls, with each change of the order of the estimate or below.
- Solve `ref.model` again with `SolverOptions(order=3)`: the change in the
  quantity should be about the estimate or smaller.
- On a lossless device, S on the refined model should stay passive and
  reciprocal, `SᴴS ≈ I` and `S ≈ Sᵀ`.

How gfem itself checks the estimates against known answers:
{ref}`verification-mesh-refinement`.

## Limits

- The error estimate is an estimate, not a bound. On gfem's verification
  cases the field estimator's was between {gfem-effectivity}`field low` and
  {gfem-effectivity}`field high` times the true error, at `order=2` and
  `order=3`, and the quantity estimator's on a capacitance, S entry or
  frequency between {gfem-effectivity}`quantity low` and
  {gfem-effectivity}`quantity high` at `order=2`; Q, a cluster mean and each
  kind of objective have their own measured ranges, listed on the
  verification page. The quantity estimator's higher ratios came from
  fields with no sharp metal edge (a uniform line, an empty cavity), where
  its curl and mass terms are never allowed to cancel. At `order=3` the
  quantity estimate is not checked against its range; at `order=1` both
  estimators refuse.
- Some estimates are much more pessimistic. An objective at its target
  measured up to {gfem-effectivity}`objective at its target high` times its
  error; a reflection objective near a reflection zero, up to
  {gfem-effectivity}`reflection objective near a reflection zero high`;
  several modes' frequencies at their target, up to
  {gfem-effectivity}`several modes at their target high`; a participation
  times its frequency, up to
  {gfem-effectivity}`participation times frequency high`. S on a wave port
  whose face meets a microstrip's edge is also overestimated far beyond its
  range. Refinement then stops late, never early.
- Where an error changes sign between meshes (a mode of a cavity with a
  dielectric insert, the mean of a degenerate pair), the estimate cannot
  follow the crossing and can overestimate by far more than its range.
- A lumped port spanning materials of different loss (a lossy substrate
  and air, say) is not covered by the field estimator's range: there it
  measured {gfem-effectivity}`lumped port across a loss jump low` to
  {gfem-effectivity}`lumped port across a loss jump high` times the true
  error on coarse meshes.
- A lumped port whose face crosses a permittivity interface, meshed with
  cells stretched 10:1 or more along the line, measured
  {gfem-effectivity}`stretched cells at a lumped port across a permittivity contrast low`
  to
  {gfem-effectivity}`stretched cells at a lumped port across a permittivity contrast high`
  times the S error. Mesh such ports with cells of moderate aspect, or let
  refinement split them first.
- Next to an `Impedance` or `Absorbing` surface that carries the field's
  current, the field estimate measured
  {gfem-effectivity}`impedance surface low` to
  {gfem-effectivity}`impedance surface high` times the error; S,
  objectives of S and field objectives raise there.
- Next to a lumped port spanning a permeability contrast, the field
  estimate measured
  {gfem-effectivity}`lumped port across a permeability contrast low` to
  {gfem-effectivity}`lumped port across a permeability contrast high` and
  does not improve under refinement; S and objectives raise there.
- On a permittivity law with $-20 < \operatorname{Re} \varepsilon < 0$ at a solve frequency, metal
  edges and port junctions make the field itself mesh-dependent: S and
  objectives raise, and the field estimate measured as low as
  {gfem-effectivity}`plasmonic window low`. A lumped port's edge on a metal
  interface converges slowly even at $\operatorname{Re} \varepsilon = -30$. The -20 bound assumes
  metal wedges of 17° or more and a damping of about 2 % of the frequency.
- Below a guide's cutoff, a mesh coarser than the field's decay length
  underestimates both S and the field: resolve the decay length.
- It measures the error against your mesh's own facets. Mesh refinement
  subdivides the facets you imported and never moves them onto a curved
  surface, so the faceting error is neither estimated nor reduced. On a
  circular coaxial line meshed with flat facets, both estimates were below
  the error against the exact circle. Mesh curved surfaces finer in your
  mesher.
- A wave port's own mode is solved on the port's 2D mesh, and the error of
  its propagation constant is not estimated; it enters S through
  de-embedding offsets. On a dielectric-strip guide de-embedded over the
  whole port-to-port distance it stayed below
  {gfem-effectivity}`port mode share high` times the S estimate. Keep offsets
  shorter than the leads you model. A wave port in a lossy medium also
  leaves the attenuation along the port out of its boundary condition, so
  even a matched lossy port reflects a little; the estimate does not see
  that reflection.
- Some estimates need one more solve, of an {term}`adjoint dual`: an S
  entry whose read port is not excited, or is a wave port with a hybrid
  mode in a lossy cross-section (a lossy dielectric slab, say), and an
  objective of a field measurement, of electrostatic energy or a potential
  mean, or of a participation. `refine_mesh` rides it on its own solves,
  with your solver options, except on the first mesh of an objective, whose
  reads are known only once the objective has been traced on a result.
  There, and in `error_estimate` on a finished result, the system is
  factorized once more (per frequency, or per mode), with the direct
  solver at the result's order, which on a large model may not fit in
  memory: estimate such a model through `refine_mesh` (`max_cycles=0`
  estimates without refining). A participation's dual also needs an
  isolated mode, the direct solver on the CPU and `eigen_tolerance` at most
  1e-10.
- It does not include the linear solver's error. `refine_mesh` requires
  `tolerance` to be at least 10 times `SolverOptions.tolerance`
  (`eigen_tolerance` for an eigenmode problem); for $Q$, times $\cot(\arg \omega)$;
  for an objective, times its sensitivity.
- `refine_mesh` raises {py:obj}`gfem.MeshRefinementUnsupportedError`, before
  any solve, on an adaptive sweep, a port mode problem, a periodic model, a
  Bloch wave vector, a thermal or DC conduction problem (`feature`
  `"thermal"` or `"dc_conduction"`), a driven problem with sources
  (`feature` `"incident field"` or `"point dipole"`: the estimators know
  nothing of a source's load; `error_estimate` refuses its result too), a
  driven or eigenmode model with a
  {term}`perfectly matched layer` (the estimators weigh the layer with its
  background material; electrostatic refinement of such a model runs) and
  `SolverOptions(order=1)` (the `"fast"` preset); both
  estimators refuse order 1. An estimate of a mode bound to a permittivity
  law region where $\operatorname{Re} \varepsilon < 0$ at the mode (a mode guided along a metal
  surface) raises it when it estimates: refine such a model with the
  `FieldEstimator`. Permittivity laws where $\operatorname{Re} \varepsilon > 0$, second-order
  absorbing boundaries, a complex `mu_r` and loss inside an anisotropic
  material are estimated for a mode's `omega`; Q and objectives of a mode
  still raise on each of them. The dispersive shifts of
  `gfem.quantum.dressed_modes` are not estimated: they read the junction's
  participation, which raises. An objective raises by name on anything it
  may not read (see {ref}`refine-for-an-objective`).
- Mesh refinement runs on one process. Above one rank it raises
  {py:obj}`gfem.DistributedUnsupportedError`: refine on one process, then
  solve `ref.model` on ranks ({doc}`distributed-solve`).

## Next

{py:obj}`gfem.refine_mesh`, {py:obj}`gfem.error_estimate`,
{py:obj}`gfem.FieldEstimator`, {py:obj}`gfem.QuantityEstimator`,
{py:obj}`gfem.MeshRefinement`, {py:obj}`gfem.RefinementCycle`,
{py:obj}`gfem.region_mean`, {doc}`measurements`, {doc}`remesh-and-restart`,
{doc}`robust-optimization`, {doc}`frequency-sweep`.
