# Distributed solves

Split one large solve across the cores and nodes of a cluster. You start
your script with `mpirun` or `srun`, and gfem divides the mesh and the linear
system between the ranks. Every rank gets the same result and the same
gradients, from the same calls you make on one process. Driven uniform
sweeps, electrostatic and eigenmode solves, and periodic cells on one axis
run this way; anything else raises at once ({ref}`distributed-what-runs`).

## Install

{{ install_mpi }}

The extra adds mpi4py, on Linux x86_64. The MPI library comes from your
machine and must be **Open MPI**.

- On a cluster, load its Open MPI module. Do not install the `openmpi` wheel
  into the environment: mpi4py would load it before the module and miss the
  cluster's interconnect.
- On a workstation, install your distribution's Open MPI.
- The `direct` and `petsc` solvers, and every eigenmode solve, also need the
  {term}`MPI PETSc stack` ({ref}`petsc-with-mpi`); driven and electrostatic
  solves on `bddc_gmres` or `auto` do not. Run the stack under the Open MPI
  module it was built with: gfem does not check this.

## Run the script

This script solves a two-port line, keeps one field and takes the gradient of
$|S_{21}|^2$ with respect to the permittivity of each design element. Download
{download}`the script <../../tools/distributed_line.py>` and
{download}`its mesh <../tutorials/data/line.msh>`, then start it on every rank:

```bash
mpirun -n 4 python distributed_line.py bddc_gmres   # a workstation
srun --mpi=pmix python distributed_line.py direct   # under Slurm
```

```{literalinclude} ../../tools/distributed_line.py
:language: python
:start-at: mesh = gfem.Mesh.from_gmsh
```

- Rank 0 prints $S_{21}$ at 5 GHz, $|S_{21}|^2$ and the norm of the gradient. They are
  the numbers one process prints for the same script.
- Rank 0 writes `line-<solver>.s2p`; every rank takes part in writing the
  field to `e-<solver>.vtu`.
- gfem reads the rank count from the launcher's environment. To choose the
  communicator yourself, pass `comm=` to {py:obj}`gfem.solve`, `Job.run`,
  `Mesh.from_gmsh`, `Job.from_json` or `load_result`.
- A job file runs the same way: `srun --mpi=pmix python -m gfem run job.json --out result`.
- `gfem.log.enable()` shows rank 0's log lines, tagged `r0/4`, and every
  rank's warnings; `rank0_only=False` shows every rank's lines
  ({doc}`watch-a-solve`).
- On a machine with fewer cores than ranks, add `--oversubscribe --bind-to
  none --mca mpi_yield_when_idle 1` to `mpirun`.
- The script is a driven solve; an electrostatic or eigenmode script runs the
  same way, with the calls you make on one process.

(distributed-what-runs)=
## What runs across ranks

A {term}`distributed solve` runs these, with their gradients:

- a `gfem.Driven` uniform sweep, with lumped ports with a resistance, and wave
  ports, several on one surface included;
- a `gfem.Electrostatic` solve: capacitance, potentials and region energies;
- a `gfem.Eigenmode` solve without wave ports, nonlinear and dispersive
  ones included:
  frequencies, Q, mode fields, participations and
  `gfem.interface_participation`. Frequency and Q differentiate; a gradient
  through a mode's field raises;
- a model with one `gfem.Periodic` pair: driven and eigenmode solves without a
  wave vector, and a driven solve with a {term}`Bloch wave vector` on `direct`
  or `bddc_gmres`;
- `PEC`, `PMC` and `Absorbing` (orders 1 and 2) boundaries, excluded volumes,
  `Conductor` and `Impedance` sheets, `SheetField`, anisotropic materials and
  `MaterialField`;
- gradients in permittivity, conductivity, `mu_r`, `tan_delta`, sheet fill and
  node coordinates, a design at a wave port face included, also under `jax.jit`, `jax.vmap` and `jax.jacrev`;
- the solvers `auto`, `direct` and `petsc`, and `bddc_gmres` at order 2 and
  above. `auto` is `bddc_gmres` for driven and electrostatic solves and
  `direct` for eigenmode. The eigen solver is SLEPc: `eigen_solver="auto"`
  names it across ranks;
- `gfem.power_flow`, `gfem.cross_sections`, `gfem.far_field` and
  `gfem.beam_coupling`, collective calls: every rank calls them and gets the
  same numbers;
- kept fields with `Field.at`, `Field.b` and `Field.to_vtu`, `Job`,
  `python -m gfem run`, `gfem.PortMode`, and the partial result of a
  `gfem.LinearSolverConvergenceError`.

Everything else raises {py:obj}`gfem.DistributedUnsupportedError` on every
rank before any mesh is built, so nothing runs on fewer ranks than you asked
for. Catch it and read `feature`, or run the feature as {term}`rank copies`
(`comm=MPI.COMM_SELF`), where every feature runs, one complete solve per
process:

```{gfem-distributed-refusals}
```

A periodic model is partitioned so that each pair's two surfaces fall on the
ranks together. gfem tries up to 8 partitions. When none passes, every rank
raises `RuntimeError` naming the tries: run on another number of ranks, or as
{term}`rank copies`.

{py:obj}`gfem.SolverUnavailableError` means the ranks cannot be used: mpi4py
or MPI does not load, the MPI is not Open MPI, the launcher started no MPI
(`srun --mpi=none`), or `direct` or `petsc` finds no MPI PETSc stack. Its
message says which.

## Write files

S, gradients, the mesh and the job are the same on every rank, so write them
from one rank. A kept field's methods need every rank, so call them on all:

```python
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
result = gfem.solve(model, problem, options=options)
if rank == 0:
    result.s.to_touchstone("device.s2p")
    job.to_json("job.json")
result.e_field(frequency, "p1").to_vtu("field.vtu")  # every rank calls this
```

- `Field.coefficients` holds only this rank's part of the field, not the whole
  array. Use `Field.at` or `Field.to_vtu` for the whole field.
- `Job.run(out=)` and `python -m gfem run` gather the fields on rank 0, so a
  saved result reads on one process: `load_result(path, comm=MPI.COMM_SELF)`.
- Put the mesh and the job file on a filesystem every rank sees.

## Write your own script

Every rank runs the whole script and must call gfem with the same values in
the same order.

- Any value that decides what your script does next must be identical, bit for
  bit, on every rank. S, gradients and the mesh are. A clock, a file listing or
  a random number without a shared seed is not: compute it on rank 0 and send
  it to the others with `MPI.COMM_WORLD.bcast`. gfem compares its inputs at the
  start of each solve and raises on every rank, naming the input that differs.
  A rank that is late prints one line after 60 s and keeps waiting.
- An error that gfem detects on all ranks together raises on every rank, and
  you can catch it. Any other error on one rank prints its traceback with the
  rank number and ends the job. This holds for your own code from the first
  solve on. For errors before it, launch with `python -m mpi4py script.py`,
  or the other ranks wait forever.
- An error found while `jax.grad` computes the gradient loses its class across
  ranks, as on one process: catch `jax.errors.JaxRuntimeError` and `ValueError`
  and read the message (see {ref}`gradient-errors`).
- On the MPI PETSc stack, use PETSc on every rank copy or on none: a job that
  mixes them has not been tried.

(distributed-layout)=
## Choose ranks and threads

Give every rank the same number of cores, and size the layout by the solver:

- **With the MPI PETSc stack** (`direct`, `petsc`, or `bddc_gmres` with the
  stack installed): about 4 threads per rank, for example 28 ranks × 4 threads
  on a 112-core node (`srun --ntasks-per-node=28 --cpus-per-task=4`).
- **Without PETSc**: one step of every BDDC iteration runs on the first rank,
  so use fewer ranks with more threads, for example 8 ranks × 14 threads on a
  112-core node.
- Use fewer ranks when a rank would hold under about 90 k DOF.
- With `bddc_gmres` on a large model, on a PETSc stack built by the current
  `petsc_wheelhouse.sh`, a single solve and each later design of a loop on
  one mesh in one process took less time on 16 and 32 nodes than on 8: see
  {doc}`../project/scaling`. On a stack built by an earlier script these
  times do not apply; {doc}`petsc-slepc` says how to rebuild it. Keep the
  designs in one script on one set of ranks, since what BDDC keeps between
  them ends with the process.
- Do not pass `--cpu-bind`. Rank 0 needs no node or threads of its own.

With `threads=None`, each rank takes `NGS_NUM_THREADS` if set. Otherwise it
shares each physical core of its CPU mask with the other ranks of the node
whose mask holds that core.
Every rank then uses the smallest of these counts over all ranks, so one
narrow mask slows the whole job. `mpirun -n 2` binds each rank to one core:
add `--bind-to none` or pass `threads`. This default count is read at the
first solve on a communicator: a later change of `NGS_NUM_THREADS` is not
seen, so pass `threads` to change it. gfem sets the BLAS threads itself
during the solve and does not read `OMP_NUM_THREADS`. Under ranks, the
default of `retain_memory` compares the memory held by a node's ranks with
half of the smallest node's memory. `retain_memory` evicts only what the
solving communicator holds, so solves on a second communicator
({term}`rank copies`, `comm.Split`) can keep memory past it. Under ranks,
BDDC's coarse factorizations kept between solves count against
`retain_memory`; with `retain_memory=0` every solve sets them up anew.

With `bddc_gmres`, gfem factors BDDC's coarse problem on a share of the ranks.
The same script on the same ranks and stack always takes the same route, and
the route is not an option: S stays that of the coarse problem factored on one
rank, to round-off. A singular coarse problem raises at once; only a MUMPS
workspace error is retried, with more workspace.

Measured times for each layout, up to 108.6 M DOF on 32 nodes, are on
{doc}`../project/scaling`.

## Check your results

- **One process against many.** Run the same script once without a launcher
  and once on your ranks. S and the gradient should agree to the solver
  tolerance, and on `direct` close to rounding error.
- **Two solvers.** Run with `direct` and with `bddc_gmres`. S should agree to
  about `SolverOptions.tolerance`.
- **Eigenmode.** Compare with one process on `eigen_solver="slepc"`. The
  number of SLEPc iterations can change with the rank count; the frequencies,
  Q and participations should agree to the eigen tolerance.

How gfem itself is checked across ranks: {doc}`../verification`.

## Limits

- Open MPI on Linux x86_64 only. Under another MPI gfem raises
  `gfem.SolverUnavailableError` above one rank, because NGSolve's wheel gives
  wrong numbers under MPICH and has no Intel MPI support. Load an Open MPI
  module.
- `direct` across ranks is MUMPS through PETSc, not PARDISO: the `mkl` extra
  does not apply to it. Install the extra for one-process solves.
- Periodic pairs on one axis only: two or three pairs, and a Bloch eigenmode
  solve, raise `gfem.DistributedUnsupportedError`. Solve them on one process
  or as rank copies.
- An eigenmode solve with a wave port raises
  `gfem.DistributedUnsupportedError` across ranks: solve it on one process.
- A lumped port terminated by an inductance or a capacitance,
  `gfem.characteristic_impedance`, `linear_solver="petsc"` on an
  electrostatic, eigenmode or periodic solve, and a periodic eigenmode solve
  on `bddc_gmres` run across ranks but have not been compared with one process
  yet. Compare them once on your model.

## Next

{py:obj}`gfem.solve`, {py:obj}`gfem.Eigenmode`,
{py:obj}`gfem.DistributedUnsupportedError`,
{py:obj}`gfem.SolverUnavailableError`, {doc}`petsc-slepc`, {doc}`periodic-cell`,
{doc}`../project/scaling`.
