# Run a distributed solve

A driven solve can run across the ranks of an MPI job: `mpirun` or `srun` starts
your script on every rank, the mesh and the linear system split across them, and
every rank gets the same S and the same gradients. The script makes the same
calls as on one process. Without a launcher nothing changes: gfem runs on one
process, and needs neither mpi4py nor MPI.

## What runs, and what raises

A {term}`distributed solve` runs a `gfem.Driven` uniform sweep, and its gradients,
on these:

- lumped ports with a resistance, and wave ports, several on one surface included;
- `PEC`, `PMC` and `Absorbing` (orders 1 and 2) boundaries, excluded volumes,
  `Conductor` and `Impedance` sheets, `SheetField`, anisotropic materials and
  `MaterialField`;
- gradients in permittivity, conductivity, `mu_r`, `tan_delta`, sheet fill and node
  coordinates, under `jax.jit`, `jax.vmap` and `jax.jacrev`;
- `linear_solver` `"auto"` and `"bddc_gmres"` at order 2 and above, `"direct"`, and
  `"petsc"`;
- kept fields with `Field.at`, `Field.b` and `Field.to_vtu`, `Job`,
  `python -m gfem run`, and the partial result of a
  `gfem.LinearSolverConvergenceError`.

`gfem.PortMode` runs too, once on rank 0, and every rank gets its result. Each item
has launched tests in `tests/mpi/`, which compare it with one process: to 1e-11 on
`"direct"`, to 1e-9 and the exact iteration count on `"bddc_gmres"`, and bit for bit
across ranks. Two things have no such test yet: a lumped port terminated by an
inductance or a capacitance, and `gfem.characteristic_impedance`. They read the same
code, and they do not raise.

Everything else raises `gfem.DistributedUnsupportedError` on every rank, before any
mesh is built, so nothing runs on fewer ranks than you asked for and a script can
catch it. `feature` holds the key:

```{gfem-distributed-refusals}
```

This table is generated from the test cases of these refusals and gfem's own
messages. {py:obj}`gfem.interface_participation` takes an eigenmode result, which no
distributed solve produces. Nothing refuses at one rank: {term}`rank copies` run every
feature, as one process each.

`gfem.SolverUnavailableError` is what a launched solve raises when it cannot use the
ranks: mpi4py or the MPI library does not load, the MPI is not Open MPI, the launcher
did not start MPI (`srun --mpi=none`), or `"direct"` and `"petsc"` find no
{term}`MPI PETSc stack`.

## Install

```bash
pip install "gfem[mpi]"
```

The extra is mpi4py alone, on Linux x86_64 (elsewhere it installs nothing). It brings
no MPI library, which comes from the machine and must be **Open MPI**: under MPICH's
ABI NGSolve's wheel gives wrong numbers with no error, and it has no Intel MPI
wrapper, so gfem raises `gfem.SolverUnavailableError` above one rank on anything else.

- **On a cluster**, load its Open MPI module, and do not install the `openmpi` wheel
  in the environment: mpi4py looks in the environment's `lib/` first, so the wheel
  would likely win over the module, away from the machine's interconnect.
  The MareNostrum 5 section below ran this way, on Open MPI 4.1.6.
- **On a workstation**, install your distribution's Open MPI, or
  `pip install openmpi` into the environment for a small test. The launched tests
  without PETSc ran on the wheel, Open MPI 5.0.11. The wheel is not in the extra.
  Its licence file is one the licence gate cannot read, so check it yourself
  ({doc}`../project/licence`).

`"bddc_gmres"` and `"auto"` need no PETSc: BDDC's coarse problem then runs on
{term}`rank 0 <Rank-0 coarse factorization>`. `"direct"` and `"petsc"` need the
{term}`MPI PETSc stack`, and with it BDDC's coarse problem
{term}`runs across ranks too <Distributed coarse factorization>`. Build it with
`tools/petsc_wheelhouse.sh --mpi`: {ref}`petsc-with-mpi` says what it holds. A stack
runs under the Open MPI it was built on. gfem does not check this, so load the same
module when you run. The build's online form, without `--sources`, is written but has
not run yet: its first run is the `mpi-petsc` job of `.github/workflows/petsc.yml`.
The offline form below ran on MareNostrum 5.

## The script

Every rank runs the whole script, and calls gfem with the same values in the same
order:

```bash
mpirun -n 4 python script.py                # a workstation
srun --mpi=pmix python script.py            # under Slurm
srun --mpi=pmix python -m gfem run job.json --out result
```

Here is a whole one. It needs the {download}`mesh <../tutorials/data/line.msh>`, and
{download}`the script itself <../../tools/distributed_line.py>`:

```{literalinclude} ../../tools/distributed_line.py
:language: python
```

- gfem reads the launcher's task count from the environment, without importing
  mpi4py. `comm=` on `gfem.solve`, `Job.run`, `Mesh.from_gmsh`, `Job.from_json` and
  `load_result` overrides it. A communicator of size 1 is one process.
- A script of your own may do what it likes between calls, but a value that decides
  what happens next must be an {term}`agreed value`: the same bits on every rank.
  S, gradients and everything the mesh holds are. A clock, a file listing or a
  random number without a shared seed is not. gfem checks its inputs when a solve
  starts: a difference raises on every rank, naming the input. If a rank is late, it
  prints one line after 60 s and goes on waiting.
- `comm=MPI.COMM_SELF` asks for {term}`rank copies`: one complete solve per process,
  each with every feature. On the MPI PETSc stack, PETSc starts on the whole job, so
  use PETSc on every copy or on none: a job that mixes them was not tested.
  Frequency workers run in rank copies, and importing mpi4py at the script's top is
  fine with them.
- `mpirun -n 2` gives each rank one core, since Open MPI binds to a core at that
  count. On a machine with fewer cores than ranks, add `--oversubscribe --bind-to none
  --mca mpi_yield_when_idle 1`, as the tests do.

## Files

S, gradients, the mesh and the job are the same on every rank, so one rank writes
them, and they need no communication. A kept field's methods are collective, so every
rank calls them:

```python
from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
result = gfem.solve(model, problem, options=options)
if rank == 0:
    result.s.to_touchstone("device.s2p")
    result.s.to_xarray().to_netcdf("device.nc")
    job.to_json("job.json")
    mesh.to_vtu("mesh.vtu")
result.e_field(frequency, "p1").to_vtu("field.vtu")  # every rank calls this
```

- `Field.coefficients` is this rank's padded {term}`field piece`, not a global array.
- `Job.run(out=)` and `python -m gfem run` gather the fields to rank 0, in the order
  they have on one process, so a bundle reads on one process whatever rank count wrote
  it. Read it with `load_result(path, comm=MPI.COMM_SELF)`.
- The mesh and the job file must be on a filesystem every rank sees. Rank 0 reads the
  mesh and the job and broadcasts them, but `python -m gfem run` checks on every rank
  that the job file exists.

## Errors

A failure every rank decides on together, an {term}`agreed error`, raises on every
rank, and the script may catch it. Any other failure on one rank prints its traceback
with the rank and ends the job: its peers could not go on. From the first distributed
solve on, this holds for your own code too. Before it, launch with
`python -m mpi4py script.py`, or an exception on one rank leaves the others waiting.

`gfem.solve` itself raises gfem's error classes. Errors in a backward pass cross a JAX
callback, as they do on one process, and lose their class, so read the message:

```python
try:
    value, grad = jax.value_and_grad(objective)(design)
except (jax.errors.JaxRuntimeError, ValueError) as err:
    print(err)
```

## Threads and layout

On MareNostrum 5 use 28 ranks per node with 4 threads each:
`srun --ntasks-per-node=28 --cpus-per-task=4`, with no `--cpu-bind`. Use fewer ranks
when a rank would hold under about 90 k DOF, and without PETSc, where the coarse
problem stays on rank 0, use 8 ranks with 14 threads. Give every rank the same core
count, since unequal masks can fail the check at the solve's start. Rank 0 needs no
node of its own and no extra threads. This comes from layout trials of a prototype on
that machine; {ref}`distributed-scale` is where runs through `gfem.solve` go.

- `threads=None` under ranks is `NGS_NUM_THREADS` if set, otherwise the physical cores
  in the rank's mask, divided by the ranks on the node sharing that mask, at least 1,
  and the smallest count over all ranks. On one process the usual rule stands.
- gfem sets the BLAS threads itself: the rank's count in a factorization, 1 in BDDC's
  coarse solve. `OMP_NUM_THREADS` is not read. Your own later setting stands outside
  those phases.
- `retain_memory` and its default count per node: the bytes held on a node's ranks
  against the smallest node's half.

## On MareNostrum 5

The commands below ran as written, with `<account>` and `<user>` filled in, on the
GPP partition. The compute nodes have no internet, and a login node ends any process
after 300 s of CPU, so you fetch elsewhere and build in a job. The default modules
include Intel MPI, which gfem refuses, so every job starts with `module purge`.
gfem is not on PyPI yet, so this works from a clone of its repository.

On a machine with internet, with uv, in the clone:

```bash
uv build
uv python install 3.12
cp -rL "$(dirname "$(dirname "$(uv python find --managed-python 3.12)")")" python
PYTHON=3.12 bash tools/petsc_wheelhouse.sh --fetch-sources sources
uvx pip download -d wheels --python-version 3.12 --implementation cp \
  --only-binary=:all: --platform manylinux_2_28_x86_64 \
  --platform manylinux_2_27_x86_64 --platform manylinux_2_17_x86_64 \
  --platform manylinux2014_x86_64 --platform manylinux_2_5_x86_64 \
  --platform manylinux1_x86_64 "$(ls dist/gfem-*.whl)[mpi]"
rsync -a python sources wheels dist tools/petsc_wheelhouse.sh tools/distributed_line.py \
  docs/tutorials/data/line.msh \
  <user>@transfer1.bsc.es:/gpfs/scratch/<account>/<user>/gfem/
```

On a login node, in that folder, make the environment. The Python is one not built
by conda, which the stack needs:

```bash
python/bin/python3.12 -m venv venv
venv/bin/python -m pip install --no-index --find-links wheels "$(ls dist/gfem-*.whl)[mpi]"
venv/bin/python -m pip install --no-index --find-links sources/wheels uv
```

Then build the stack on a compute node, with `sbatch build.sh`. It took about a
quarter of an hour on one node, and 4.1.6 is the Open MPI to use: OpenBLAS did not build
under the Intel wrappers of Open MPI 5.0.10.

```bash
#!/bin/bash
#SBATCH --account=<account> --qos=gp_debug --nodes=1 --exclusive
#SBATCH --ntasks=1 --cpus-per-task=112 --time=00:45:00
module purge
module load cmake/3.30.5 openmpi/4.1.6-gcc
export MAKE_NP=$SLURM_CPUS_PER_TASK PYTHON=$PWD/venv/bin/python PATH=$PWD/venv/bin:$PATH
export TMPDIR=$PWD/tmp-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
bash petsc_wheelhouse.sh --mpi --sources sources wheelhouse-mpi
venv/bin/python -m pip install --no-index --find-links wheelhouse-mpi \
  petsc petsc4py slepc slepc4py
```

And run, with `sbatch run.sh`, across two nodes, on each solver in turn:

```bash
#!/bin/bash
#SBATCH --account=<account> --qos=gp_debug --nodes=2 --exclusive
#SBATCH --ntasks-per-node=28 --cpus-per-task=4 --time=00:15:00
module purge
module load openmpi/4.1.6-gcc
srun --mpi=pmix --cpus-per-task=$SLURM_CPUS_PER_TASK venv/bin/python distributed_line.py direct
srun --mpi=pmix --cpus-per-task=$SLURM_CPUS_PER_TASK venv/bin/python distributed_line.py bddc_gmres
```

This mesh is far too small for 56 ranks, and the job is here to check the install;
with your own mesh, follow the layout above. A multi-node job holds its nodes whole
and bills every core of them. Do not time anything under `srun --overlap`.

(distributed-scale)=
## Scale

% The 78 M DOF rehearsal (D4) has not run: its result goes here when it has. Keep the
% anchor: the layout section links to it.

These are observations from one run each, on MareNostrum 5 GPP (Xeon Platinum 8480+),
Open MPI 4.1.6, PETSc 3.26.0, gfem at commit ec608cd; `tools/scale_run.py` makes the runs.
They gate nothing: a time here is no promise about another mesh or layout. The mesh is
a coplanar line with 10,248,856 DOF at order 2 and 1,464,909 tetrahedra, `direct` on two
frequencies, `bddc_gmres` on one at `tolerance=1e-8`, two ports. A multi-node job holds
112 cores per node, whatever it uses.

| Nodes | Ranks x threads | DOF per rank | `direct`, s | Largest node, GiB | `bddc_gmres`, s | Largest node, GiB |
|---|---|---|---|---|---|---|
| 1 | 1 x 112 | all | 109 | 109 | 121 | 51 |
| 1 | 28 x 4 | 366 k | 101 | 153 | 67 | 64 |
| 2 | 56 x 4 | 183 k | 69 | 97 | 54 | 43 |
| 4 | 112 x 4 | 92 k | 57 | 69 | 49 | 28 |
| 8 | 224 x 4 | 46 k | 97 | 89 | 46 | 22 |
| 16 | 448 x 4 | 23 k | 181 | 96 | 58 | 20 |
| 32 | 896 x 4 | 11 k | 178 | 95 | 51 | 19 |

Seconds are inside `gfem.solve`, on the slowest rank. Memory is the peak resident size
summed over a node's ranks. Against the first row, at every rank count, S was the same
bit for bit on all ranks and differed by at most 2.8e-12 on `direct` and 7.6e-13 on
`bddc_gmres`, relative to S's largest entry, with the same 64 iterations per port. The
rows below 90 k DOF per rank are under the layout above, there to test agreement.

The largest wave port mesh shown across ranks is `r8_wp`, 168,022 DOF, on 28 ranks of
one node at 50 GHz (job 48473792): S to 1.7e-12 on `direct`, and 8.9e-13 on `bddc_gmres`
with the same iterations.

What has run so far, on MareNostrum 5:

- the launched tests, at 3, 4 and 8 ranks, and 16 for the direct solve, on one node;
- the two-node job above, at 56 ranks on the 2,880 tetrahedra of `line.msh`, Open MPI
  4.1.6, gfem at commit 11ae3c6: on each solver, what the script prints agrees with
  one process to within 3e-12, relative.

## Limits

- Open MPI only, Linux x86_64 only. Run so far on Open MPI 4.1.6 (with the MPI PETSc
  stack) and 5.0.11 (without it).
- Above one rank, `"direct"` and `"petsc"` need the MPI PETSc stack.
- A distributed `"direct"` is PETSc's MUMPS, and not PARDISO or NGSolve's Cholesky.
- Periodic models raise, one axis included.

## Next

{py:obj}`gfem.solve`, {py:obj}`gfem.DistributedUnsupportedError`,
{py:obj}`gfem.SolverUnavailableError`, {doc}`petsc-slepc`, {doc}`../project/licence`.
