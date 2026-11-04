"""Check that gfem refuses an MPI that is not Open MPI, on real libraries.

Launched by CI's ``mpi`` job with the mpiexec of a throwaway environment holding PyPI's
``mpich``. Exits 0 on every rank only if resolving the communicator raises
`gfem.SolverUnavailableError` naming Open MPI, and mpi4py really runs on MPICH.
"""

import sys

from mpi4py import MPI

import gfem
from gfem import _mpi

vendor = MPI.get_vendor()[0]
if vendor == "Open MPI":
    sys.exit("this check needs another MPI, and mpi4py found Open MPI")
try:
    _mpi.resolve_comm()
except gfem.SolverUnavailableError as err:
    if "needs Open MPI" not in str(err):
        raise
    print(f"rank {MPI.COMM_WORLD.rank} on {vendor}: {err}")
else:
    sys.exit(f"gfem did not refuse {vendor}")
