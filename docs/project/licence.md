# Licence

MIT.

gfem depends on [NGSolve](https://ngsolve.org) (LGPL-2.1), whose PyPI wheels
also contain GPL-2.0+ SuiteSparse code (UMFPACK, CHOLMOD Supernodal); keep that
in mind when redistributing a whole gfem environment. The CI licence gate
(`tools/licence_gate.py`) fails on non-free or non-commercial licences in the
installed dependency tree; it admits proprietary licences only for the vendor
wheels of the optional extras. The base install is OSI-licensed.

## Optional extras

- `cuda12` and `cuda13` pull NVIDIA's proprietary runtime wheels (cuDSS and
  the CUDA runtime, under NVIDIA's licences).
- `mkl` pulls Intel MKL, TBB and TCM under the Intel Simplified Software
  License, and Intel's OpenMP runtime (`intel-openmp`, `intel-cmplr-lib-ur`)
  under the **Intel End User License Agreement for Developer Tools**. That
  agreement is stricter: its clause 3.1(xi) forbids using the software
  "directly or indirectly for SaaS services or service bureau purposes", that
  is, in a service that gives third parties access to it. Read it (installed
  as `LICENSE.txt` in the `intel_openmp` distribution) before installing
  `gfem[mkl]`, and leave the extra out of hosted deployments; without it,
  `linear_solver="direct"` uses NGSolve's sparse Cholesky and gives the same
  results, more slowly.

## PETSc and SLEPc

`linear_solver="petsc"` and `eigen_solver="slepc"` use libraries you build
with {doc}`../recipes/petsc-slepc`; gfem neither ships nor declares them, so
the licence gate does not see them. The script builds PETSc, SLEPc and their
Python bindings (BSD-2-Clause) with MUMPS (CeCILL-C), METIS (Apache-2.0) and
OpenBLAS (BSD-3-Clause), without MPI, so ParMETIS (not free for commercial
use) is never built; a `petsc`-marked test checks that ParMETIS, PT-Scotch and
ScaLAPACK are absent. conda-forge's PETSc depends on ParMETIS, which is why the
recipe builds from source.
