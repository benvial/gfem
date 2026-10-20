"""tests/isolated.py: peak RSS of one call, in a fresh process.

The timing tests compare gfem's peak memory with Palace's whole-process
number, so the call they gate must not inherit the high-water mark of the
tests that ran before it in the same pytest process.
"""

import resource

import numpy as np
from isolated import isolated

MIB = 2**20


def _touch(mib):
    """Allocate and write ``mib`` MiB, so the pages count in the RSS."""
    block = np.ones(mib * MIB, dtype=np.uint8)
    return int(block[::4096].sum())


def test_returns_the_value_of_the_call():
    value, _ = isolated(_touch, 1)
    assert value == MIB // 4096


def test_peak_counts_what_the_call_allocates():
    _, rss = isolated(_touch, 300)
    assert rss >= 300 / 1024


def test_peak_ignores_the_callers_high_water_mark():
    _touch(400)
    parent = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MIB  # GiB
    assert parent >= 400 / 1024
    _, rss = isolated(_touch, 1)
    assert rss < parent - 300 / 1024
