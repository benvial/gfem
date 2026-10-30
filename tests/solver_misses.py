"""Force a driven sweep's linear solve to miss at chosen frequencies.

Usage::

    from solver_misses import cannot_converge_at

    cannot_converge_at(monkeypatch, (f2,))
    ...  # a driven solve; its Krylov solve at f2 does not converge

At a chosen frequency, after the sweep's handle is prepared or refactorized
there, the base handle's iteration cap drops below what any solve needs (BDDC:
``maxiter = 1``; PETSc: the KSP's ``max_it = 0``, its tolerances kept). The
real linear solver then misses through its own code, its rebuild of a reused
preconditioner included, and raises. The cap is an iteration count, so the
miss is deterministic. The seam reaches only this process (frequency
workers run in spawned processes). `adjoint_cannot_converge_at` does the same
to the backward pass's adjoint solves only.
"""

import math

from gfem._engine import driven
from gfem._linalg.bddc import BddcHandle


def cannot_converge_at(monkeypatch, frequencies) -> None:
    """Make every sweep's linear solve miss at each of ``frequencies`` (Hz)."""
    at = driven._SweepHandle.at

    def capped(self, omega):
        handle = at(self, omega)
        if _chosen(omega, frequencies):
            _cap(self._base)
        return handle

    monkeypatch.setattr(driven._SweepHandle, "at", capped)


def adjoint_cannot_converge_at(monkeypatch, frequencies) -> None:
    """Make every adjoint solve miss at each of ``frequencies`` (Hz).

    The forward sweep is left as it is. Lumped ports only: with wave ports
    the adjoint handle wraps its base.
    """
    prepare = driven._Ports.prepare

    def capped(self, omega, **options):
        handle = prepare(self, omega, **options)
        if _chosen(omega, frequencies):
            _cap(handle)
        return handle

    monkeypatch.setattr(driven._Ports, "prepare", capped)


def _chosen(omega, frequencies) -> bool:
    f = omega / (2 * math.pi)
    return any(math.isclose(f, g, rel_tol=1e-12) for g in frequencies)


def _cap(base) -> None:
    """Drop ``base``'s iteration cap below what any solve needs."""
    if isinstance(base, BddcHandle):
        # warm solves read the handle's cap, a cold one its solver's;
        # maxiter = 0 would count as converged
        base.maxiter = base._solver.maxiter = 1
    else:
        ksp = base._ksp
        rtol, atol, divtol, _ = ksp.getTolerances()
        ksp.setTolerances(rtol=rtol, atol=atol, divtol=divtol, max_it=0)
