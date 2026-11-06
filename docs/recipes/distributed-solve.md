# Run a distributed solve

Split one large driven solve across the cores and nodes of a cluster. You
start your script with `mpirun` or `srun`, and gfem divides the mesh and the
linear system between the ranks. Every rank gets the same S and the same
gradients, from the same calls you make on one process. A driven uniform
sweep and its gradients run this way; anything else raises at once
({ref}`distributed-what-runs`).

## Install

```bash
pip install "gfem[mpi]"
```

The extra adds mpi4py, on Linux x86_64. The MPI library comes from your
machine and must be **Open MPI**.

- On a cluster, load its Open MPI module. Do not install the `openmpi` wheel
  into the environment: mpi4py would load it before the module and miss the
  cluster's interconnect.
- On a workstation, install your distribution's Open MPI.
- The `direct` and `petsc` solvers also need the {term}`MPI PETSc stack`
  ({ref}`petsc-with-mpi`); `bddc_gmres` and `auto` do not. Run the stack under
  the Open MPI module it was built with: gfem does not check this.
- On a cluster whose compute nodes have no internet, follow
  {doc}`offline-cluster`.

## Run the script

This script solves a two-port line, keeps one field and takes the gradient of
|S21|² with respect to the permittivity of each design element. Download
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

- Rank 0 prints S21 at 5 GHz, |S21|² and the norm of the gradient. They are
  the numbers one process prints for the same script.
- Rank 0 writes `line-<solver>.s2p`; every rank takes part in writing the
  field to `e-<solver>.vtu`.
- gfem reads the rank count from the launcher's environment. To choose the
  communicator yourself, pass `comm=` to {py:obj}`gfem.solve`, `Job.run`,
  `Mesh.from_gmsh`, `Job.from_json` or `load_result`.
- A job file runs the same way: `srun --mpi=pmix python -m gfem run job.json --out result`.
- On a machine with fewer cores than ranks, add `--oversubscribe --bind-to
  none --mca mpi_yield_when_idle 1` to `mpirun`.

(distributed-what-runs)=
## What runs across ranks

A {term}`distributed solve` runs a `gfem.Driven` uniform sweep and its
gradients with:

- lumped ports with a resistance, and wave ports, several on one surface
  included;
- `PEC`, `PMC` and `Absorbing` (orders 1 and 2) boundaries, excluded volumes,
  `Conductor` and `Impedance` sheets, `SheetField`, anisotropic materials and
  `MaterialField`;
- gradients in permittivity, conductivity, `mu_r`, `tan_delta`, sheet fill and
  node coordinates, also under `jax.jit`, `jax.vmap` and `jax.jacrev`;
- the solvers `auto` and `bddc_gmres` at order 2 and above, `direct` and
  `petsc`;
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
  and read the message (see {doc}`capacitance-gradient`).
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
- Do not pass `--cpu-bind`. Rank 0 needs no node or threads of its own.

With `threads=None`, each rank takes `NGS_NUM_THREADS` if set. Otherwise it
shares the physical cores of its CPU mask with the other ranks on that mask.
Every rank then uses the smallest of these counts over all ranks, so one
narrow mask slows the whole job. `mpirun -n 2` binds each rank to one core:
add `--bind-to none` or pass `threads`. gfem sets the BLAS threads itself
during the solve and does not read `OMP_NUM_THREADS`. Under ranks, the
default of `retain_memory` compares the memory held by a node's ranks with
half of the smallest node's memory.

Measured times for each layout, up to 108.6 M DOF on 32 nodes, are on
{doc}`../project/scaling`.

## Check your results

- **One process against many.** Run the same script once without a launcher
  and once on your ranks. S and the gradient should agree to the solver
  tolerance, and on `direct` close to rounding error.
- **Two solvers.** Run with `direct` and with `bddc_gmres`. S should agree to
  about `SolverOptions.tolerance`.

How gfem itself is checked across ranks: {doc}`../verification`.

## Limits

- Open MPI on Linux x86_64 only. Under another MPI gfem raises
  `gfem.SolverUnavailableError` above one rank, because NGSolve's wheel gives
  wrong numbers under MPICH and has no Intel MPI support. Load an Open MPI
  module.
- `direct` across ranks is MUMPS through PETSc, not PARDISO: the `mkl` extra
  does not apply to it. Install the extra for one-process solves.
- Periodic models raise `gfem.DistributedUnsupportedError`, one periodic axis
  included. Solve them on one process or as rank copies.
- A lumped port terminated by an inductance or a capacitance, and
  `gfem.characteristic_impedance`, run across ranks but have not been compared
  with one process yet. Compare them once on your model.

## Next

{py:obj}`gfem.solve`, {py:obj}`gfem.DistributedUnsupportedError`,
{py:obj}`gfem.SolverUnavailableError`, {doc}`petsc-slepc`,
{doc}`offline-cluster`, {doc}`../project/scaling`.
