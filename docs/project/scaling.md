# Performance data

These are observations of gfem's distributed driven solve on one machine at one commit:
MareNostrum 5 GPP highmem nodes, gfem at commit 9a57680, Open MPI 4.1.6, PETSc 3.26.0,
NGSolve 6.2.2608, 28 ranks of 4 threads per node, order 2, one frequency of 50 GHz unless
a row says otherwise, both ports excited. No time here is a pass or fail criterion, and a
time here is no promise about another mesh, layout or machine. Each row names its Slurm
job; the run records, logs, job scripts and meshes are in the scaling record (Zenodo, DOI
pending).

The two solvers are `linear_solver="direct"`, PETSc's MUMPS with PT-Scotch ordering, and
`linear_solver="bddc_gmres"`, NGSolve's BDDC under gfem's GMRES at the
`SolverOptions(tolerance=...)` named in each row's `tol` column (gfem's own default is
`1e-6`). {doc}`../recipes/distributed-solve`
shows how to run them. A table counts repeats of a run inside one Slurm job, so a rung's
range is run noise, not placement.

The tables come from the run records through `tools/scale_tables.py`, by hand, and are
committed; building these docs reads no record. Each table's first line, a comment in the
page source, holds the SHA-256 of the records manifest it was made from.

## The machine

What the hardware gives apart from any solver: MKL's dense multiply and a STREAM triad on
one node against thread count, then a ping-pong and `Allreduce` between nodes.

```{include} _scaling/machine.md
```

## The meshes

One family: a coplanar line on a TopMetal2 layer with its conductors removed from the
volume ("voided"), at three refinements, and the uniform split of each, which replaces
every tetrahedron by eight.

```{include} _scaling/meshes.md
```

## Agreement

Three checks, as the milestone of at least 100 M DOF asks:

- **Every rank count against one process.** On the 10,248,856 DOF line, from 1 to 32
  nodes, each solver against its own one-process run at 112 threads: the first twelve
  rows below.
- **The two solvers against each other.** `direct` did not finish at 108,570,046 DOF
  ([below](scaling-direct-108m)), so this check ran on the largest mesh both solvers
  finished, 47,277,960 DOF on 16 nodes: `bddc_gmres` at `tol=1e-10` against `direct`. It
  did not run at 108,570,046 DOF.
- **The largest mesh against the 10,248,856 DOF line**, observed and not gated, since the
  meshes differ: the last row.

The largest run:

```{include} _scaling/ms3.md
```

```{include} _scaling/agreement.md
```

## Strong scaling, `bddc_gmres`, 108,570,046 DOF

From 4 to 32 nodes. The `gfem.solve` median does not fall past 8 nodes: it is lowest at
8 nodes and higher at 16 and 32. BDDC's coarse problem runs on at most 64 ranks whatever
the node count (56 at 4 and 8 nodes), and its apply takes about the same time at every
node count; the per-phase table [below](scaling-phases) gives its share of `gfem.solve`.
`direct` on 32 nodes is a row of its own.

```{include} _scaling/strong-108m.md
```

## Strong scaling, both solvers, 23,148,138 DOF

From 1 to 32 nodes. Above 8 nodes a rank holds fewer than about 90 k DOF, under the
layout the recipe recommends; the rows are there to show it. `direct` at 16 nodes ran out
of its time limit in `mumps_setup`, so 32 nodes was not run.

```{include} _scaling/strong-23m.md
```

## Weak scaling, about 3 M DOF per node

From 1 to 32 nodes, each rung its own mesh of the family; the DOF per node column gives
each rung's size. The 4-node rung holds more than the others.

```{include} _scaling/weak.md
```

(scaling-direct-108m)=
## `direct` at 108,570,046 DOF

One attempt on 32 nodes. MUMPS's analysis finished and printed its estimates; the
factorization reported too little memory (`INFO(1) = -9`), gfem's retry analysed again
with a larger memory relaxation (`ICNTL(14)` from 60 to 120), and Slurm then ended the step
on an out-of-memory kill. No S came out of it.

```{include} _scaling/direct-108m.md
```

(scaling-phases)=
## Where the time goes

The phases of the largest rungs, and GMRES's own share: its vector work (the basis's
orthogonalization, norms, scaling and updates) and the time its ranks wait in reductions.

```{include} _scaling/phases.md
```

At 108,570,046 DOF the coarse apply is 42 % to 53 % of `gfem.solve` at every node count,
and the coarse setup grows from 8 to 32 nodes. At 32 nodes the reduction wait is 4.4 s,
1.9 % of GMRES and 0.8 % of `gfem.solve`; the Krylov vector share is 1.9 %. A
communication-reducing GMRES would act on that share only.

## Rerun

The scaling record holds the job scripts and `tools/` at the commit: `scale_env.sh` makes
the environment, `scale_steps.py` writes each job's steps, `scale_job.sbatch` runs one job
per rung per solver and `scale_run.py` makes each run and its record. `scale_bench.py` is
the machine benchmark. The voided meshes are in the record, by md5; `split_mesh.py` makes
the splits. `scale_mesh.py` rebuilds a voided mesh's source with `gsim==0.7.0`,
`ihp-gdsfactory==0.2.8` and `gmsh==4.15.2` only: a current gsim writes a different
conductivity, so S would move for a reason unrelated to the mesh. `scale_void.py` then
removes the conductors. To redo the tables:

```bash
python tools/scale_tables.py RECORDS docs/project/_scaling --commit 9a57680 \
  --logs LOGS --mesh-md5 meshes/md5.txt --mesh-md5 meshes/splits.md5
```
