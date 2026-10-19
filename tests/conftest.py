"""Mark every test with its tier, taken from its directory under tests/.

tests/unit -> unit, tests/verification -> verification, tests/reference ->
reference, tests/benchmark -> benchmark, tests/cuda -> cuda. A test file
outside these directories is a collection error, so each test has one tier.
"""

from pathlib import Path

import pytest

TIERS = ("unit", "verification", "reference", "benchmark", "cuda")
TESTS_DIR = Path(__file__).parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add the tier marker before pytest's `-m` deselection runs."""
    for item in items:
        tier = item.path.relative_to(TESTS_DIR).parts[0]
        if tier not in TIERS:
            raise pytest.UsageError(
                f"{item.nodeid}: tests must live in tests/<tier>/ with tier in {TIERS}"
            )
        item.add_marker(tier)
