# Licence

MIT.

gfem depends on [NGSolve](https://ngsolve.org) (LGPL-2.1), whose PyPI wheels
also contain GPL-2.0+ SuiteSparse code (UMFPACK, CHOLMOD Supernodal); keep that
in mind when redistributing a whole gfem environment. The base install is
OSI-licensed: only the optional extras below pull proprietary vendor wheels.

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

- `mpi` pulls mpi4py (BSD-3-Clause). It brings no MPI library: Open MPI comes from
  the machine, and is not gfem's to ship. PyPI's `openmpi` wheel is not part of
  the extra.

## PETSc and SLEPc

`linear_solver="petsc"` and `eigen_solver="slepc"` use libraries you build
with {doc}`../recipes/petsc-slepc`; gfem neither ships nor declares them.
The script builds PETSc, SLEPc and their Python bindings (BSD-2-Clause) with:

- on the {term}`no-MPI PETSc stack`, MUMPS (CeCILL-C), METIS (Apache-2.0) and
  OpenBLAS (BSD-3-Clause);
- on the {term}`MPI PETSc stack`, the same, with ScaLAPACK (BSD-3-Clause) and
  PT-Scotch (CeCILL-C), on the machine's Open MPI.

ParMETIS (not free for commercial use) is not part of either stack, and the
no-MPI stack contains neither PT-Scotch nor ScaLAPACK. The MPI stack's
`libptscotchparmetisv3` is Scotch's own ParMETIS-compatible interface, not
ParMETIS. conda-forge's PETSc depends on ParMETIS, which is why the recipe builds
from source.
