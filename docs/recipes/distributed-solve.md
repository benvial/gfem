# Distributed solves

Split one large solve across the cores and nodes of a cluster. You start
your script with `mpirun` or `srun`, and gfem divides the mesh and the linear
system between the ranks. Every rank gets the same result and gradients, from
the same calls as on one process. Driven uniform sweeps, electrostatic and
eigenmode solves, and periodic cells on one axis run this way; anything else
raises at once ({ref}`distributed-what-runs`).

## Install

{{ install_mpi }}

The extra adds mpi4py, on Linux x86_64. The MPI library comes from your
machine and must be **Open MPI**: load the cluster's module (never the
`openmpi` wheel) or install your distribution's. `direct`, `petsc` and every
eigenmode solve also need the {term}`MPI PETSc stack`
({ref}`petsc-with-mpi`), run under the Open MPI it was built with.

## Run the script

This script solves a two-port line and takes the gradient of $|S_{21}|^2$ with
respect to each design element's permittivity. Download
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

- Rank 0 prints $S_{21}$ at 5 GHz, $|S_{21}|^2$ and the gradient norm, as one
  process does, and writes `line-<solver>.s2p` and (with every rank)
  `e-<solver>.vtu`.
- gfem reads the rank count from the launcher; `comm=` on
  {py:obj}`gfem.solve`, `Job.run`, `Mesh.from_gmsh`, `Job.from_json` and
  `load_result` chooses the communicator.
- A job runs the same way: `srun --mpi=pmix python -m gfem run job.json --out result`.
- Fewer cores than ranks: add `--oversubscribe --bind-to none --mca
  mpi_yield_when_idle 1` to `mpirun`. Log lines: {doc}`watch-a-solve`.

(distributed-what-runs)=
## What runs across ranks

A {term}`distributed solve` runs these, with their gradients:

- a `gfem.Driven` uniform sweep, with resistive lumped ports and wave ports;
- a `gfem.Electrostatic` solve;
- a `gfem.Eigenmode` solve without wave ports: frequencies, Q, mode fields and
  participations. A gradient through a mode's field raises;
- a model with one `gfem.Periodic` pair: driven and eigenmode solves, and a
  driven solve with a {term}`Bloch wave vector` on `direct` or `bddc_gmres`;
- `PEC`, `PMC` and `Absorbing` boundaries, `Conductor` and `Impedance`
  sheets, anisotropic materials and `MaterialField`, with gradients, also
  under `jax.jit` and `jax.vmap`;
- the solvers `auto` (`bddc_gmres` for driven and electrostatic, `direct` for
  eigenmode), `direct`, `petsc` and `bddc_gmres` (order 2 and above);
- the collective calls `gfem.power_flow`, `gfem.cross_sections`,
  `gfem.far_field` and `gfem.beam_coupling`, and `Job`.

Everything else raises {py:obj}`gfem.DistributedUnsupportedError` on every
rank before any mesh is built. Catch it, or run the feature as
{term}`rank copies` (`comm=MPI.COMM_SELF`), one complete solve per process:

```{gfem-distributed-refusals}
```

A periodic model is partitioned in up to 8 tries so each pair's surfaces
share ranks; when none passes, every rank raises `RuntimeError`: change the
rank count. {py:obj}`gfem.SolverUnavailableError` means the ranks cannot be
used (no mpi4py or MPI, not Open MPI, no PETSc stack for `direct` or `petsc`);
its message says which.

## Write files

S, gradients, the mesh and the job are the same on every rank: write them from
one rank. A kept field's methods need every rank:

```python
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
result = gfem.solve(model, problem, options=options)
if rank == 0:
    result.s.to_touchstone("device.s2p")
    job.to_json("job.json")
result.e_field(frequency, "p1").to_vtu("field.vtu")  # every rank calls this
```

- `Field.coefficients` holds only this rank's part; `Field.at` and
  `Field.to_vtu` give the whole field.
- `Job.run(out=)` gathers the fields on rank 0, so a saved result reads on one
  process: `load_result(path, comm=MPI.COMM_SELF)`. Mesh and job file sit on a
  filesystem every rank sees.

## Write your own script

Every rank runs the whole script and calls gfem with the same values in the
same order.

- A value that decides what your script does next must be identical on every
  rank. Compute a clock, a file listing or an unseeded random number on rank 0
  and send it with `MPI.COMM_WORLD.bcast`. gfem compares its inputs at each
  solve and raises on every rank, naming the input that differs.
- An error gfem detects on all ranks raises on every rank and can be caught;
  any other error on one rank ends the job with its traceback (launch with
  `python -m mpi4py script.py` for errors before the first solve).
- An error during `jax.grad` loses its class across ranks
  ({ref}`gradient-errors`).

(distributed-layout)=
## Choose ranks and threads

Give every rank the same number of cores, and size the layout by the solver:

- **With the MPI PETSc stack** (`direct`, `petsc`, or `bddc_gmres` with the
  stack installed): about 4 threads per rank, for example 28 ranks × 4 threads
  on a 112-core node (`srun --ntasks-per-node=28 --cpus-per-task=4`).
- **Without PETSc**: one BDDC step runs on the first rank, so use fewer ranks
  with more threads, for example 8 ranks × 14 threads.
- Use fewer ranks when a rank would hold under about 90 k DOF. No `--cpu-bind`.
- `bddc_gmres` on a large model ran faster on 16 and 32 nodes than on 8
  ({doc}`../project/scaling`; stack from the current `petsc_wheelhouse.sh`,
  {doc}`petsc-slepc`). Keep a loop's designs in one script on the same ranks.

With `threads=None`, one narrow CPU mask slows every rank; that and BDDC's
coarse-problem ranks: {py:obj}`gfem.SolverOptions`. Times: {doc}`../project/scaling`.

## Check your results

- **One process against many.** S and the gradient agree to the solver
  tolerance.
- **Eigenmode.** Frequencies, Q and participations agree with one process on
  `"slepc"` to the eigen tolerance.

How gfem itself is checked across ranks: {doc}`../verification`.

## Limits

- Open MPI on Linux x86_64 only: NGSolve's wheel is wrong under
  MPICH, so another MPI raises `gfem.SolverUnavailableError` above one rank.
- `direct` across ranks is MUMPS through PETSc, not the `mkl` extra.
- Reactive lumped ports, `gfem.characteristic_impedance`, `"petsc"` on an
  electrostatic, eigenmode or periodic solve, and a periodic eigenmode solve
  on `bddc_gmres` run across ranks but are not yet compared with one process:
  compare them once on your model.

## Next

{py:obj}`gfem.solve`, {py:obj}`gfem.Eigenmode`,
{py:obj}`gfem.DistributedUnsupportedError`,
{py:obj}`gfem.SolverUnavailableError`, {doc}`petsc-slepc`, {doc}`periodic-cell`,
{doc}`../project/scaling`.
