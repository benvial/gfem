# Performance data

## What to expect

How large a driven model gfem has solved across MPI ranks, and on one node
with `pmg_gmres`, and how fast, on one cluster. These are observations from two sets of runs, each of one gfem version;
each table says which. They are not limits or promises for your model. Every
number in this summary comes from the raw tables linked below.

For gfem and another solver on the same meshes, on one node from 8 to 112 cores,
see {doc}`benchmarks`.

| Solver | Largest model, DOF | Nodes | `gfem.solve`, s | Largest node, GiB | Set |
|---|---|---|---|---|---|
| `bddc_gmres` | 108,570,046 | 8 | 423 | 143 | earlier version |
| `bddc_gmres` | 108,570,046 | 32 | 291 | 119 | later version |
| `direct` | 47,277,960 | 16 | 832 | 401 | earlier version |
| `pmg_gmres` | 17,211,038 | 1 | 298 | 115.1 | one-node set |

The `bddc_gmres` rows are the fastest solve of the largest mesh in each version
([earlier](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/strong-108m.md), [later](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/coarse-strong-108m.md)). `direct` at
108,570,046 DOF on 32 nodes ran out of memory ([details](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/direct-108m.md)).

- **Which solver.** Up to 14,217,194 DOF on 1 to 4 nodes, `direct` was faster: 21 s
  against 36 s at 3,061,188 DOF on one node ([table](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/weak.md)). From
  23,148,138 DOF up, `bddc_gmres` was faster at every node count tried, and
  needed less memory: 175 GiB against 716 GiB on the largest node on one node
  ([table](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/strong-23m.md)).
- **How many nodes.** In the earlier version, past a point more nodes made
  `bddc_gmres` slower, because it factored BDDC's coarse problem on 64 ranks. In
  the later version, on a PETSc stack built by its build script, 108,570,046 DOF
  took 440, 323 and 291 s on 8, 16 and 32 nodes ([table](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/coarse-strong-108m.md)).
  Each time is one run, and it does not apply to a stack built by an earlier script.
- **Repeated solves.** When one script solves the same mesh again, as a loop over
  designs does, gfem keeps the ordering and symbolic factorization of the coarse
  problem, and GMRES starts from the previous solution. The second and third
  designs took 301 and 300 s on 8 nodes against 440 s for the first
  ([table](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/coarse-design-loop.md)).
- **One node, photonic models.** `linear_solver="pmg_gmres"` took about 10
  iterations per port on silicon guides a few to a dozen wavelengths across, and
  needed about a quarter of `direct`'s memory. It was faster than `direct` from a
  few million unknowns on ([table](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/pmg-ladder.md)). These are rounded
  observations; the table holds the measured values.
- **Before the solve.** Reading and partitioning the mesh come before
  `gfem.solve` and are not in its time: about 200 s at 108,570,046 DOF
  ([table](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/phases.md), [coarse set](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/coarse-phases.md)).

All of this is one machine, three gfem versions (each table says which), two
families of meshes (a coplanar line across ranks, silicon guides on one node),
order 2 unless a row says otherwise, and one frequency per solve. A time here is an
observation, not a promise for another mesh, layout or machine: time a small run of
your own model before you size a large job. {doc}`../recipes/distributed-solve`
shows how to run both solvers and how to choose ranks and threads.

The S of these runs were checked against one process and against the other solver:
see {doc}`../verification`. The later version's one-frequency S differ from the
earlier version's, on the same mesh and node count, by at most
$2.4 \times 10^{-12}$.

## Ranks and threads, 10,248,856 DOF

These are separate runs, not part of the raw tables: one run per layout of a
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

The one-node rows come from an earlier gfem and a prototype script, and time the
whole run. A later measurement on one node, with the benchmark suite and gfem
`771da66`, timed the work after startup and found 28 x 4 fastest for
`bddc_gmres`: 60.5 s against 154 s for one process of 112 threads
({doc}`benchmarks`). Layouts were compared up to 4 nodes and 10,248,856 DOF; past
that the advice is an extrapolation.

## How to reproduce

- **gfem**: the earlier version is the development version of 5 November 2026,
  between the preview and 0.1.0, with Open MPI 4.1.6, PETSc 3.26.0 and NGSolve
  6.2.2608. The later version is the development version of 21 November 2026,
  with the same Open MPI, PETSc and NGSolve, on a PETSc stack built by that
  version's `petsc_wheelhouse.sh`, whose MUMPS caps PT-Scotch's parallel ordering
  at 32 processes.
- **Machine**: MareNostrum 5 general-purpose partition, `highmem` nodes, two Intel
  Xeon Platinum 8480+ per node, 112 cores, 28 ranks of 4 threads per node.
- **Runs**: order 2, one frequency of 50 GHz unless a row says otherwise, both ports
  excited. `bddc_gmres` ran at the {py:obj}`gfem.SolverOptions` `tolerance` in each
  row's `tol` column; gfem's own default is `1e-6`. `direct` is PETSc's MUMPS with
  PT-Scotch ordering. The repeats of a row ran back to back on the same nodes, so a
  row's range is run-to-run noise, not placement.
- **Design loop**: design $i$, from 0, scales the permittivity of the
  highest-permittivity volume by $1 + 10^{-3} i$.

The scaling record (Zenodo, DOI pending) holds both sets, each with the commit it
names: the run records, logs, job scripts, meshes and the `tools/` scripts at that
commit. `scale_mesh.py` rebuilds a voided mesh's source with `gsim==0.7.0`,
`ihp-gdsfactory==0.2.8` and `gmsh==4.15.2` only: a current gsim writes a different
conductivity, so S would move for a reason unrelated to the mesh.

The tables are not computed when these docs build. To redo them from the record, with
`COMMIT` the commit the record names:

```bash
python tools/scale_tables.py RECORDS docs/project/_scaling --commit COMMIT \
  --logs LOGS --mesh-md5 meshes/md5.txt --mesh-md5 meshes/splits.md5
python tools/scale_tables.py RECORDS2 docs/project/_scaling --set coarse \
  --commit COMMIT2 --logs LOGS2
```

`RECORDS`, `LOGS` and `COMMIT` are the earlier version's; `RECORDS2`, `LOGS2` and
`COMMIT2` the later one's.

The raw tables, on GitHub:

- Machine: [hardware benchmark](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/machine.md), [meshes](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/meshes.md).
- Earlier version: [largest run](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/ms3.md), [strong scaling at 108,570,046 DOF](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/strong-108m.md),
  [strong scaling at 23,148,138 DOF](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/strong-23m.md), [weak scaling](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/weak.md),
  [`direct` at 108,570,046 DOF](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/direct-108m.md), [phases](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/phases.md).
- Later version: [one solve](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/coarse-strong-108m.md), [loop of 3 designs](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/coarse-design-loop.md),
  [phases](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/coarse-phases.md).
- One node: [`pmg_gmres` against `direct`](https://github.com/benvial/gfem/blob/main/docs/project/_scaling/pmg-ladder.md).
