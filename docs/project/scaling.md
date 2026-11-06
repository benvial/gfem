# Performance data

## What to expect

How large a driven model gfem has solved across MPI ranks, and how fast, on one
cluster. Every number in this summary comes from the tables below.

- **Largest model.** `linear_solver="bddc_gmres"` solved 108,570,046 DOF on 4 to 32
  nodes. Its fastest {py:obj}`gfem.solve` was 423 s, on 8 nodes
  ([table](scaling-strong-108m)).
- **Largest model with `direct`.** `linear_solver="direct"` solved 47,277,960 DOF on
  16 nodes in 832 s ([table](scaling-weak)). At 108,570,046 DOF on 32 nodes it ran
  out of memory ([details](scaling-direct-108m)).
- **Which solver.** Up to 14,217,194 DOF on 1 to 4 nodes, `direct` was faster: 21 s
  against 36 s at 3,061,188 DOF on one node, 54 s against 76 s at 14,217,194 DOF on
  4 nodes ([table](scaling-weak)). From 23,148,138 DOF up, `bddc_gmres` was faster
  at every node count tried: 199 s against 410 s on one node, 128 s against 161 s
  on 8 nodes ([table](scaling-strong-23m)). It also needed less memory: 175 GiB
  against 716 GiB on the largest node at 23,148,138 DOF on one node.
- **How many nodes.** Past a point, more nodes made `bddc_gmres` slower. At
  108,570,046 DOF its time was lowest on 8 nodes; at 23,148,138 DOF, on 4 to 8
  nodes.
- **Before the solve.** Reading and partitioning the mesh come before
  `gfem.solve` and are not in its time: about 200 s at 108,570,046 DOF
  ([table](scaling-phases)).

All of this is one machine, one gfem version, one family of meshes (a coplanar line),
order 2 and one frequency. A time here is an observation, not a promise for another
mesh, layout or machine: time a small run of your own model before you size a large
job. {doc}`../recipes/distributed-solve` shows how to run both solvers and how to
choose ranks and threads.

The S of these runs were checked against one process and against the other solver:
see {doc}`../verification`.

## The machine

MareNostrum 5 general-purpose nodes, two Intel Xeon Platinum 8480+ per node, 112
cores, run as 28 ranks of 4 threads per node. What the hardware gives apart from any
solver: MKL's dense multiply and a STREAM triad on one node against thread count,
then a ping-pong and `Allreduce` between nodes.

```{include} _scaling/machine.md
```

## The meshes

One family: a coplanar line on a TopMetal2 layer with its conductors removed from the
volume ("voided"), at three refinements, and the uniform split of each, which replaces
every tetrahedron by eight.

```{include} _scaling/meshes.md
```

(scaling-strong-108m)=
## Strong scaling, `bddc_gmres`, 108,570,046 DOF

The largest run:

```{include} _scaling/ms3.md
```

From 4 to 32 nodes. The `gfem.solve` median is lowest on 8 nodes and higher on 16
and 32. `direct` on 32 nodes is a row of its own.

```{include} _scaling/strong-108m.md
```

(scaling-strong-23m)=
## Strong scaling, both solvers, 23,148,138 DOF

From 1 to 32 nodes. Above 8 nodes a rank holds fewer than about 90 k DOF, under the
layout the recipe recommends ({ref}`distributed-layout`); the rows are there to show it.
`direct` on 16 nodes ran out of its time limit, so 32 nodes was not run.

```{include} _scaling/strong-23m.md
```

(scaling-weak)=
## Weak scaling, about 3 M DOF per node

From 1 to 32 nodes, each rung its own mesh of the family; the DOF per node column gives
each rung's size. The 4-node rung holds more than the others.

```{include} _scaling/weak.md
```

(scaling-direct-108m)=
## `direct` at 108,570,046 DOF

One attempt on 32 nodes. The factorization reported too little memory, and gfem's
retry with a larger memory allowance was then stopped by the cluster for running out
of memory. No S came out of it. The estimates below are what MUMPS printed before the
factorization.

```{include} _scaling/direct-108m.md
```

(scaling-phases)=
## Where the time goes

The phases of the largest `bddc_gmres` rungs, and GMRES's own share: its vector work
(the basis's orthogonalization, norms, scaling and updates) and the time its ranks
wait in reductions.

```{include} _scaling/phases.md
```

At 108,570,046 DOF the coarse apply is 42 % to 53 % of `gfem.solve` at every node
count, and the coarse setup grows from 8 to 32 nodes. At 32 nodes the reduction wait
is 0.8 % of `gfem.solve`.

## Ranks and threads, 10,248,856 DOF

These are separate runs, not part of the tables above: one run per layout of a
prototype script on the 10,248,856 DOF coplanar line, on the same machine with an
earlier gfem version. The advice in {ref}`distributed-layout` rests on them.

| Nodes | Solver | Ranks x threads per node | Time, s |
|---|---|---|---|
| 4 | `direct` | 28 x 4 | 54 |
| 4 | `direct` | 16 x 7 | 61 |
| 4 | `direct` | 8 x 14 | 79 |
| 4 | `bddc_gmres`, with PETSc | 28 x 4 | 57 |
| 4 | `bddc_gmres`, with PETSc | 16 x 7 | 54 |
| 1 | `bddc_gmres`, without PETSc | 8 x 14 | 118 |
| 1 | `bddc_gmres`, without PETSc | 28 x 4 | 145 |

At 28 x 4 on 4 nodes a rank holds about 92 k DOF. S did not depend on the layout.
Layouts were compared up to 4 nodes and 10,248,856 DOF; past that the advice is an
extrapolation.

## How these were measured

- **gfem**: the development version of 5 November 2026, after 0.1.0, with Open MPI
  4.1.6, PETSc 3.26.0 and NGSolve 6.2.2608.
- **Machine**: MareNostrum 5 general-purpose partition, `highmem` nodes, 28 ranks of 4
  threads per node.
- **Runs**: order 2, one frequency of 50 GHz unless a row says otherwise, both ports
  excited. `bddc_gmres` ran at the {py:obj}`gfem.SolverOptions` `tolerance` in each row's `tol`
  column; gfem's own default is `1e-6`. `direct` is PETSc's MUMPS with PT-Scotch
  ordering. The repeats of a row ran back to back on the same nodes, so a row's range
  is run-to-run noise, not placement.

The scaling record (Zenodo, DOI pending) holds the run records, logs, job scripts,
meshes and the `tools/` scripts at the commit it names: `scale_env.sh` makes the
environment, `scale_steps.py` writes each job's steps, `scale_job.sbatch` runs one job
per rung per solver and `scale_run.py` makes each run and its record. `scale_bench.py`
is the machine benchmark. The voided meshes are in the record, by md5; `split_mesh.py`
makes the splits. `scale_mesh.py` rebuilds a voided mesh's source with `gsim==0.7.0`,
`ihp-gdsfactory==0.2.8` and `gmsh==4.15.2` only: a current gsim writes a different
conductivity, so S would move for a reason unrelated to the mesh. `scale_void.py` then
removes the conductors.

The tables are not computed when these docs build. To redo them from the record, with
`COMMIT` the commit the record names:

```bash
python tools/scale_tables.py RECORDS docs/project/_scaling --commit COMMIT \
  --logs LOGS --mesh-md5 meshes/md5.txt --mesh-md5 meshes/splits.md5
```
