# Licence

MIT.

gfem depends on [NGSolve](https://ngsolve.org) (LGPL-2.1), whose PyPI wheels
also contain GPL-2.0+ SuiteSparse code (UMFPACK, CHOLMOD Supernodal); keep that
in mind when redistributing a whole gfem environment. The base install is
OSI-licensed: only the optional extras below pull proprietary vendor wheels.

## Optional extras

- `cuda12` and `cuda13` pull NVIDIA's proprietary runtime wheels (cuDSS and
  the CUDA runtime, under NVIDIA's licences).
- `gmsh` pulls gmsh (GPL-2.0-or-later, with an exception for Netgen, METIS,
  OpenCASCADE and ParaView only). gfem stays MIT and imports gmsh only when
  `gfem.shape.cad_morph` asks; shipping an environment or a frozen application
  that contains gmsh is redistribution under the GPL's terms.
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
- `sax` pulls SAX (Apache-2.0) and klujax (LGPL-2.1), which statically links
  SuiteSparse KLU and BTF (LGPL-2.1 or later) and AMD and COLAMD
  (BSD-3-Clause) into its binary; the rest of its tree is under permissive
  licences or MPL-2.0. gfem itself never imports SAX or klujax.
- `skrf` pulls scikit-rf (BSD-3-Clause).

## PETSc and SLEPc

`linear_solver="petsc"` and `eigen_solver="slepc"` use libraries you build
with {doc}`../recipes/petsc-slepc`; gfem neither ships nor declares them.
The script builds PETSc, SLEPc and their Python bindings (BSD-2-Clause) with:

- on the {term}`no-MPI PETSc stack`, MUMPS (CeCILL-C), METIS (Apache-2.0) and
  OpenBLAS (BSD-3-Clause);
- on the {term}`MPI PETSc stack`, the same, with ScaLAPACK (BSD-3-Clause) and
  PT-Scotch (CeCILL-C), on the machine's Open MPI.

ParMETIS is part of neither stack: it is free only for educational and research
use by non-profit institutions and US government agencies, and for evaluation by
others; it may not be redistributed. The script adds it to the MPI stack with
`--with-parmetis`, for users its terms allow; that stack is for its builder
alone and must not be shared. gfem then orders BDDC's coarse problem with it.
The no-MPI stack contains neither PT-Scotch nor ScaLAPACK. The MPI stack's
`libptscotchparmetisv3` is Scotch's own ParMETIS-compatible interface, not
ParMETIS. conda-forge's PETSc depends on ParMETIS, which is why the recipe
builds from source.

## Material data

{py:mod}`gfem.materials` ships the parameters of published formulas and
models, each with its citation, and no data tables. Cite the source paper of
each material you use when you publish; the {py:mod}`gfem.materials` page
lists them.

The tables in the tests and in the fit recipe (measured gold, aluminium and
silicon, and tabulations of the metals' published model) are excerpts of the
refractiveindex.info database, dedicated to the public domain under CC0 1.0
(M. N. Polyanskiy, Sci. Data 11, 94 (2024), doi:10.1038/s41597-023-02898-2).
Each file's header cites the paper it comes from.
