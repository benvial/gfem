(benchmarks)=
# Benchmarks

Measured in November 2026 with gfem `771da66` and Palace 0.19.0; gfem has changed
since. The numbers are observations of one machine.

What was compared: gfem (main `771da66`, run from the benchmark harness commit
`e7d58a9`) and [Palace](https://github.com/awslabs/palace) 0.19.0, on MareNostrum 5
general-purpose nodes, one node per run. Five problems, from 10,620 to 10,248,856
DOF, each solved by both codes on the same mesh with the same number of unknowns,
at 8, 16, 32, 56 and 112 cores. Every number on this page
comes from one campaign report, [`20261120-37c723a`](https://github.com/benvial/gfem/blob/434e6e6d0adc538e0cc1d9661d1f90bfa24d1dc4/reports/20261120-37c723a/index.md), which holds every
per-case table. These are observations from one
set of runs on one machine, not limits or promises for your model.
For gfem alone across nodes, up to 108,570,046 DOF, see {doc}`scaling`.

## What we observed

"Ratio" below is Palace 0.19.0 over gfem `771da66`, as the report prints it: work
time, or the sum of per-rank peak memory. A ratio above 1 means the Palace run took
longer or used more memory. Each solver is at its fastest layout of ranks and
threads for that core count, unless a line says otherwise.

- **Direct solvers.** On the three driven problems, `palace:mumps` took longer than
  `gfem:direct` at every core count. The ratio fell as cores were added: 14 at 8
  cores to 4.64 at 112 on microstrip, 5.38 to 2.66 on r10M_void, 20.8 to 5.93 on
  microstrip_p3. gfem's own MUMPS kind, `gfem:petsc-mumps`, also took less time
  than `palace:mumps` on the driven problems (ratios 1.95 to 11.1). On the lcavity
  eigenproblem it did not: from 16 cores on `palace:mumps` took less time, down to
  a ratio of 0.19 at 112 cores.
- **Iterative solvers, at equal achieved error.** At the same solver error,
  1.00e-08, `palace:pmg` took less time than `gfem:bddc` from 16 cores on, on
  microstrip and microstrip_p3 (ratio 0.268 and 0.259 at 112 cores), and at 112
  cores on r10M_void (0.685). At 8 cores it took longer on all three (1.63, 1.13,
  2.71). On r10M_void at 56 cores the answer depends on the error level: 1.03 at
  1.00e-08, 0.954 at 1.42e-09.
- **gfem's best layout.** The equal-error ladder ran gfem at one process of $c$
  threads, which is not its fastest layout on r10M_void from 32 cores on. At its
  best layout and the nominal tolerance $10^{-8}$, `gfem:bddc` took less time than
  `palace:pmg` at every core count (ratio 1.38 to 2.22), a comparison at equal
  tolerance, not at equal achieved error.
- **Memory.** `palace:mumps` used more memory than `gfem:direct` at nearly every
  core count on every problem; on r10M_void at 112 cores, 298 against 103 GiB.
  Between the iterative solvers it depends on the problem.
- **Adding cores.** From 8 to 112 cores, `palace:pmg` became 5.71, 5.65, 4.29 and
  9 times faster on lcavity, microstrip, r10M_void and microstrip_p3;
  `gfem:direct` 1.33, 1.79, 1.53 and 2.7 times. gfem's default layout (one process,
  $c$ threads) stayed nearly flat on the large problems; its iterative and MUMPS
  kinds scaled through several ranks of 4 threads (`gfem:bddc` on r10M_void at 112
  cores: 60.5 s at 28 ranks of 4 threads, 154 s at its default).
- **Accuracy.** On the driven problems both codes solve the same discrete problem:
  Palace's direct solve differs from gfem's by 8.86e-11 on microstrip and
  microstrip_p3, and by 1.06e-12 on r10M_void. So every run of an order sits at the
  same distance from the converged reference, and only the solver error separates
  the iterative runs. On bragg, where the reference is closed form, `gfem:direct`
  reached 2.92e-11 and `palace:mumps` 4.78e-09.

The figure shows the largest problem, r10M_void: work time and memory against cores
for every solver kind, at each one's best layout.

```{figure} _benchmarks/fig-r10M_void-scaling-37c723a.svg
:alt: r10M_void, work time and memory against core count per solver.

r10M_void: work time and memory against cores.
```

## Cases

| case | problem | DOF (order) | reference | nodes |
|---|---|---|---|---|
| lcavity | eigenmode, PEC L-shaped cavity, 1 mode above 100 GHz | 246,416 (2) | closed form `lcavity-20261117-058b2f5`, bound 1.00e-12 | standard |
| microstrip | driven, 2 lumped 50 $\Omega$ ports, first-order absorbing boundary, 5 frequencies from 1 to 100 GHz, both ports excited | 315,870 (2) | Richardson over uniform splits `microstrip-20261120-7a078ba`, bound 1.07e-02 | standard |
| bragg | eigenmode, Bloch-periodic cell, $k = \pi/a$ along $x$, 2 modes | 10,620 (3) | closed form `bragg-20261118-a0ff134`, bound 1.00e-12 | standard |
| r10M_void | driven, coplanar line, 50 GHz, port 1 excited | 10,248,856 (2) | none (no converged reference fits the budget) | high-memory |
| microstrip_p3 | microstrip's problem and mesh at order 3 | 907,791 (3) | as microstrip | high-memory |


Errors are $\max |S_{ij} - S_{\mathrm{ref},ij}|$ over entries and frequencies for the
driven problems, and the largest relative eigenfrequency error for the eigenproblems.

## Method

The suite's `benchmarks/README.md` at commit `e7d58a9`, section
[Method and fairness rules](https://github.com/benvial/gfem/blob/e7d58a9a9db93af772fb234c29c2ecc238f7d279/benchmarks/README.md#method-and-fairness-rules),
holds the rules. In short:

- **Same problem.** One case file writes both codes' inputs; both must report the
  same DOF count on the same mesh, or the run is refused.
- **Each code at its own defaults**, except a short list of pinned settings.
  Structural differences are recorded, never removed.
- **Five solver kinds.** `gfem:direct` (PARDISO, one process), `gfem:petsc-mumps`
  and `palace:mumps` (MUMPS), `gfem:bddc` (GMRES with BDDC) and `palace:pmg`
  (GMRES with p-multigrid, Palace's default iterative solver).
- **Time** is work time: wall time minus process startup. **Memory** is the sum of
  each rank's peak resident memory, an upper bound. Three repeats per cell; the
  report prints the median with the min-max range.
- **Layouts.** At each core count $c$, every layout of $R$ ranks by $T$ threads with
  $R \times T = c$ on the small problems, and the code's default plus
  $(c/4) \times 4$ on the large ones.
- **Equal achieved error.** Both iterative kinds ran a tolerance ladder at their
  default layout, and are compared at the same achieved solver error, interpolated
  log-log between rungs and never extrapolated.

## Caveats

- **One machine, one version of each code, one set of runs.** MareNostrum 5
  general-purpose nodes only. Nothing here says how the codes compare on other
  hardware, other builds or other versions.
- **The Palace build is the palace-solver wheel, packaged by the gfem author.**
  Palace ran from the palace-solver 0.19.0 wheel on PyPI, which the author of gfem
  packages. It was chosen because, among the Palace builds measured on this machine
  before the campaign (the wheel and two builds from source), it was the fastest or
  within 1 % on the test problems. A reader may still see a conflict of interest.
- **Different MPI and libraries.** Both codes used MUMPS 5.9.1, from different
  builds: gfem's through the PETSc 3.26.0 wheel on the cluster's Open MPI, Palace's
  bundled in its wheel with MPICH 4.3.2. The wheel's MPICH has no InfiniBand
  transport (TCP between nodes); this does not affect the single-node runs here.
- **Single node only.** Runs over several nodes were tried outside the suite and
  left out: the suite's memory sampler and placement check see the first node only.
- **Nominal tolerances differ.** The codes stop on different residuals, so the
  iterative kinds are compared only at equal achieved solver error, on the default
  layouts, which are not gfem's fastest on r10M_void from 32 cores on.
- **Structural differences are kept.** Palace sets up and solves once per excited
  port and always runs its error estimator (lightened here); gfem solves all excited
  ports at once. `palace:pmg` has a matrix-free fine level; gfem assembles.
- **The microstrip reference cannot separate the codes.** Every run sits at the
  discretization floor of its order and the reference bound (1.07e-02 at 100 GHz)
  is about a third of that floor. r10M_void has no converged reference.
- **Bragg and the iterative solvers.** `palace:pmg` took 2.85e+04 GMRES iterations
  on this Bloch-periodic eigenproblem at the suite's one configuration per solver
  kind; gfem has no iterative eigensolver for it.
- **Failed, refused and timed-out runs.** `palace:mumps` crashed after the
  eigensolve on bragg at 56 and 112 cores at thread-heavy layouts (24 runs) and ran
  out of its time limit at some layouts on three problems (8 runs). gfem refused by
  design BDDC and PETSc MUMPS on Bloch eigenproblems, and PARDISO on more than one
  rank. These outcomes are not rerun and are excluded from every table.
- **Noise and small cells.** microstrip_p3 `palace:mumps` at 8 cores rests on one run;
  rows spreading over 10 % are flagged in the report, mostly the second-long bragg
  runs.
- **Known gfem limits seen here.** gfem reads and prepares the mesh in one process,
  and its default layout was up to 2.55 times slower than its best for `gfem:bddc`
  on r10M_void. Memory is a sum of per-rank peaks, an upper bound.

## Reproduce and raw data

Everything below is at fixed commits.

- **Suite and method:** [`benchmarks/`](https://github.com/benvial/gfem/tree/e7d58a9a9db93af772fb234c29c2ecc238f7d279/benchmarks)
  at commit `e7d58a9`, its
  [README](https://github.com/benvial/gfem/blob/e7d58a9a9db93af772fb234c29c2ecc238f7d279/benchmarks/README.md)
  and the campaign files under `benchmarks/campaigns/` (`full-c8` to `full-c112`,
  `full-bragg-c8` to `full-bragg-c112`, `full-bragg-pmg-c16`, `full-scale-c8` to
  `full-scale-c112`).
- **Raw data:** the `bench-data` branch at commit
  [`434e6e6`](https://github.com/benvial/gfem/tree/434e6e6d0adc538e0cc1d9661d1f90bfa24d1dc4):
  one record per run under `campaigns/20261120-e7d58a9-<file>/`, the converged
  references under `references/`, and this page's report under
  `reports/20261120-37c723a/` (`index.md`, `report.json`, every figure).
- **Check the report** from a checkout of `e7d58a9`, with the records branch as the
  store:

  ```bash
  python -m benchmarks report --check gfem-bench-data/reports/20261120-37c723a
  ```

- **Run a campaign** on a Slurm cluster: `python -m benchmarks ship`, `plan`,
  `submit` and `close`, then `report`, as the README's
  [Running a campaign](https://github.com/benvial/gfem/blob/e7d58a9a9db93af772fb234c29c2ecc238f7d279/benchmarks/README.md#running-a-campaign)
  section shows.

## Versions and date

- **gfem**: the development revision `771da66` of main, run from the harness commit
  `e7d58a9` (2026-11-20). NGSolve 6.2.2608, PETSc 3.26.0 and petsc4py 3.26.0 wheels,
  MKL 2026.2.0, mpi4py 4.1.2, on the cluster's Open MPI.
- **Palace**: 0.19.0, from the palace-solver 0.19.0 wheel (PETSc 3.26.0, MUMPS
  5.9.1, MPICH 4.3.2, all bundled).
- **Machine**: MareNostrum 5 general-purpose partition, two Intel Xeon Platinum
  8480+ per node, 112 cores; exclusive nodes, one node per run; standard nodes for
  lcavity, microstrip and bragg, high-memory nodes for r10M_void and microstrip_p3.
- **Measured**: 2026-11-20, 16 campaign files at one commit, 3 repeats per cell.
- **Report**: `20261120-37c723a`.

## Acknowledgements

[Palace](https://github.com/awslabs/palace) is an open-source 3D finite-element
solver for computational electromagnetics, developed by AWS Labs and distributed
under the Apache License 2.0. This page used Palace 0.19.0. Thanks to its
developers for making it openly available.

We acknowledge EuroHPC Joint Undertaking for awarding us access to MareNostrum5 at BSC, Spain.
