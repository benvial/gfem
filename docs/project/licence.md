# Licence

MIT.

gfem depends on [NGSolve](https://ngsolve.org) (LGPL-2.1), whose PyPI wheels
also contain GPL-2.0+ SuiteSparse code (UMFPACK, CHOLMOD Supernodal); keep that
in mind when redistributing a whole gfem environment. The CI licence gate
(`tools/licence_gate.py`) fails on non-free or non-commercial licences in the
installed dependency tree.
