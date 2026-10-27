"""Count BDDC setups and Krylov iterations in a test.

Usage::

    from preconditioners import count_iterations, count_preconditioners

    builds = count_preconditioners(monkeypatch)
    iterations = count_iterations(monkeypatch)
    ...  # solves
    assert len(builds) == 2
    assert max(iterations) < 60
"""

import ngsolve as ngs

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
    check = BddcHandle._check

    def spy(self, *args, **kwargs):
        check(self, *args, **kwargs)
        calls.append(self.iterations[-1])

    monkeypatch.setattr(BddcHandle, "_check", spy)
    return calls
