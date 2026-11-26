# Large driven models

A photonic device several wavelengths across, a guide inside a
{term}`perfectly matched layer`, or a circuit on a fine mesh can need more
memory than the direct solver's factorization allows. An iterative solver
trades that memory for iterations. This page picks one and checks it.

## When the direct solver does not fit

The default `linear_solver="auto"` already moves off `"direct"` when the
direct solver's estimate, about 48 KiB per unknown, passes half the memory the
process may use. It says so on the `"gfem.solver"` logger. Two iterative solvers
take over:

- **`"pmg_gmres"`** for a driven problem in 3D at order 2 or 3, in one
  process, on an operator equal to its transpose: no Bloch wave vector, no
  periodic pair, no Floquet port.
- **`"bddc_gmres"`** across MPI ranks, on periodic and Bloch models, and for
  electrostatic, thermal and DC conduction problems.

Pick `"pmg_gmres"` for a model several wavelengths across. Its iteration count
stays near 10 as the model grows, where `"bddc_gmres"`'s grows with the size
in wavelengths.

## Solve with the p-multigrid solver

```python
import gfem

options = gfem.SolverOptions(linear_solver="pmg_gmres")
result = gfem.solve(model, problem, options=options)
result.sweep.iterations  # Krylov iterations per excited port, per frequency
```

The model and problem are those of {doc}`solve-a-device`. The S, fields and
gradients come back as from any other solver.

The solver splits the field in two. It solves one part exactly: the
{term}`direct set`. That part holds the lowest-order (Whitney) unknown of every
mesh edge, which carries the long-range propagation and the static fields,
and every unknown of a cell in a perfectly matched layer or of a
{term}`poor cell`. The higher-order rest it smooths patch by patch, around
each mesh vertex. Together these make the {term}`p-multigrid`, and GMRES runs
on it. `tolerance` bounds the preconditioned residual, as for
`"bddc_gmres"`: it is not the error of S (see {py:obj}`gfem.SolverOptions`).

## How much memory it needs

`"auto"` estimates the p-multigrid's memory as

$$M \approx c_p N + c_d \,(D - W),$$

where:

- $N$ is the number of unknowns;
- $D$ is the size of the direct set and $W$ the number of mesh edges, so
  $D - W$ counts the higher-order unknowns solved exactly;
- $c_p \approx 8$ KiB at order 2 and 16 KiB at order 3;
- $c_d \approx 48$ KiB, the direct solver's figure per unknown.

The estimate is multiplied by `frequency_workers`.

On a mesh without a layer or poor cells, the direct set is about a fifth of
the unknowns at order 2, and the solver needs about a quarter of the direct
solver's memory. In a model with a perfectly matched layer, the layer's
unknowns are on the direct set: when the layer holds most of the unknowns,
the memory approaches the direct solver's. `"auto"` picks `"pmg_gmres"` when
this estimate fits the memory the process may use.

## Read what the solver built

Turn on the solver logger:

```python
import logging

logging.basicConfig()
logging.getLogger("gfem.solver").setLevel(logging.INFO)
```

Each build of the p-multigrid logs one record, such as

```text
p-multigrid at 193414 GHz: 1,250,312 of 6,570,104 DOFs on the direct set (19.0 %; PML DOF share 0.0 %, 0 poor cells)
```

The direct-set share is the fraction of the unknowns solved exactly. Above
about 40 % the solver's memory nears the direct solver's: the record's PML
share and poor-cell count tell which one is responsible. When `"auto"`
chooses the solver, one more record gives the direct solver's estimate, its
budget and the p-multigrid's estimate.

## Check your results

- **A tighter tolerance.** Solve again with `tolerance=1e-8` and compare the
  quantity you read (S, or B or power flow at low frequency), not only S.
  A change well below what you need says the default was enough.
- **Against the direct solver on a shorter model.** Cut the guide to a length
  that `linear_solver="direct"` solves, solve it with both, and compare S.
- **Iterations per port.** `result.sweep.iterations` stays near 10. A count in
  the dozens, or a {py:obj}`gfem.LinearSolverConvergenceError`, points to the
  mesh: check it for layers of thin cells. Otherwise use
  `linear_solver="direct"` if it fits, or a looser tolerance.

How gfem itself is checked: {doc}`../verification`.

## Limits

- **Order 2 or 3 only**: `order=1` and order 4 and above raise when
  {py:obj}`gfem.SolverOptions` is built.
- **One process**: across MPI ranks use `"bddc_gmres"`.
- **No Bloch wave vector, periodic pair or Floquet port**: use `"direct"` or
  `"bddc_gmres"`.
- **No GPU and no 2D model.**
- **Driven problems only**: eigenmode, electrostatic, thermal and DC
  conduction problems raise and name their own default.
- **Not gauged**: below the low-frequency floor it warns like every ungauged
  solve ({py:obj}`gfem.LowFrequencyWarning`), and `gauge="tree_cotree"`
  raises.
- **Optimization loops**: each design builds the solver again, starting its
  iterations from the previous design's solution.
- **Small models**: on a model of a few hundred thousand unknowns the direct
  solver is faster; keep `"auto"`.

## Next

- {doc}`perfectly-matched-layer` for the layer the solver takes on its direct
  set.
- {doc}`distributed-solve` for `"bddc_gmres"` across MPI ranks.
- {doc}`../project/scaling` for times and memory on one node.
- {doc}`../capabilities` for every solver at a glance.
