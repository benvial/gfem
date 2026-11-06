# Install on a cluster without internet

Install gfem and the {term}`MPI PETSc stack` on a cluster whose compute nodes
have no internet. You download everything on a machine that has internet,
copy it to the cluster, make the environment on a login node and build the
stack in a batch job. The worked example is MareNostrum 5's general-purpose
partition; on another Slurm cluster, change the module names, the account
and the transfer host.

## Fetch on a machine with internet

gfem is not on PyPI yet, so you build its wheel from a clone of its
repository. In the clone, with uv:

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

- `python` is a copy of a Python not built by conda, which the stack needs.
  Use the same minor version everywhere: the wheels are built for it.
- `sources` holds the PETSc stack's sources and build tools; `wheels` holds
  gfem's dependencies with the `mpi` extra.
- The last two files are a check script and its mesh, from
  {doc}`distributed-solve`.

## Make the environment on a login node

In the copied folder:

```bash
python/bin/python3.12 -m venv venv
venv/bin/python -m pip install --no-index --find-links wheels "$(ls dist/gfem-*.whl)[mpi]"
venv/bin/python -m pip install --no-index --find-links sources/wheels uv
```

Do not build the stack here: a MareNostrum 5 login node ends any process after
300 s of CPU time.

## Build the PETSc stack in a job

Submit this script with `sbatch build.sh`. It took about a quarter of an hour
on one 112-core node.

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

- `module purge` comes first because the default modules load Intel MPI,
  which gfem refuses.
- Use Open MPI 4.1.6: the stack's OpenBLAS did not build under the Intel
  compiler wrappers of Open MPI 5.0.10.
- The stack runs only under the Open MPI it was built on, so every job that
  runs gfem loads the same module.

## Run a check job

Submit this script with `sbatch run.sh`. It runs the check script on two
nodes, with each solver in turn:

```bash
#!/bin/bash
#SBATCH --account=<account> --qos=gp_debug --nodes=2 --exclusive
#SBATCH --ntasks-per-node=28 --cpus-per-task=4 --time=00:15:00
module purge
module load openmpi/4.1.6-gcc
srun --mpi=pmix --cpus-per-task=$SLURM_CPUS_PER_TASK venv/bin/python distributed_line.py direct
srun --mpi=pmix --cpus-per-task=$SLURM_CPUS_PER_TASK venv/bin/python distributed_line.py bddc_gmres
```

Both runs print the same S21, |S21|² and gradient norm as the script run on
one process, to the solver tolerance. The mesh is far too small for 56 ranks:
this job only checks the install. For your own mesh, choose the layout as in
{ref}`distributed-layout`.

## Limits

- A multi-node job holds its nodes whole and bills every core of them. Test
  on one node first.
- Timings taken under `srun --overlap` share cores with another step. Time
  your runs in their own job.

## Next

{doc}`distributed-solve`, {doc}`petsc-slepc`, {doc}`../project/scaling`.
