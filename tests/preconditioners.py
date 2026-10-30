"""Count BDDC setups, their coarse columns and Krylov iterations in a test.

Usage::

    from preconditioners import coarse_columns, count_iterations, count_preconditioners

    builds = count_preconditioners(monkeypatch)
    iterations = count_iterations(monkeypatch)
    ...  # solves
    assert len(builds) == 2
    assert max(iterations) < 60
"""

import ngsolve as ngs

from gfem._linalg import bddc
from gfem._linalg.bddc import BddcHandle


def count_preconditioners(monkeypatch) -> list[int]:
    """Spy on BDDC setups: one entry per preconditioner built."""
    calls = []
    build = ngs.Preconditioner

    def spy(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(ngs, "Preconditioner", spy)
    return calls


def count_iterations(monkeypatch) -> list[int]:
    """Spy on BDDC Krylov solves: one entry per converged right-hand side."""
    calls = []
    finish = BddcHandle._finish

    def spy(self, *args, **kwargs):
        finish(self, *args, **kwargs)
        calls.append(self.iterations[-1])

    monkeypatch.setattr(BddcHandle, "_finish", spy)
    return calls


def coarse_columns(monkeypatch) -> list[int]:
    """Spy on BDDC setups: the columns each one is expected to serve.

    One entry per coarse factorization chosen (`bddc._coarse_inverse`).
    """
    calls = []
    choose = bddc._coarse_inverse

    def spy(*args, **kwargs):
        calls.append(kwargs["columns"])
        return choose(*args, **kwargs)

    monkeypatch.setattr(bddc, "_coarse_inverse", spy)
    return calls
