"""Count BDDC setups in a test.

Usage::

    from preconditioners import count_preconditioners

    builds = count_preconditioners(monkeypatch)
    ...  # solves
    assert len(builds) == 2
"""

import ngsolve as ngs


def count_preconditioners(monkeypatch) -> list[int]:
    """Spy on BDDC setups: one entry per preconditioner built."""
    calls = []
    build = ngs.Preconditioner

    def spy(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(ngs, "Preconditioner", spy)
    return calls
