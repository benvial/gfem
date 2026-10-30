"""Mark every test with its tier, taken from its directory under tests/.

tests/unit -> unit, tests/verification -> verification, tests/reference ->
reference, tests/benchmark -> benchmark, tests/cuda -> cuda. A test file
outside these directories is a collection error, so each test has one tier.

The reference data (Palace runs, imported meshes) is found through the
``GFEM_REFERENCE_DIR`` environment variable: the `reference_dir` fixture
gives its path and skips the test when it is unset, and a test taking a
``palace_case`` argument runs once per case directory holding a
``palace.msh``.

Tests marked ``petsc`` or ``slepc`` need gfem's PETSc or SLEPc solver: they
are skipped, with the user-facing reason, where `gfem._petsc` cannot use it.

A test that sets NGSolve's thread count uses the `ngsolve_threads` fixture,
which puts the count back afterwards.
"""

import os
from pathlib import Path

import pytest

TIERS = ("unit", "verification", "reference", "benchmark", "cuda")
TESTS_DIR = Path(__file__).parent
REFERENCE_ENV = "GFEM_REFERENCE_DIR"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add the tier marker before pytest's `-m` deselection runs.

    Then skip the `petsc` / `slepc` tests where gfem cannot use PETSc /
    SLEPc.
    """
    for item in items:
        tier = item.path.relative_to(TESTS_DIR).parts[0]
        if tier not in TIERS:
            raise pytest.UsageError(
                f"{item.nodeid}: tests must live in tests/<tier>/ with tier in {TIERS}"
            )
        item.add_marker(tier)
    _skip_without_petsc(items)


def _skip_without_petsc(items: list[pytest.Item]) -> None:
    """Skip `petsc` / `slepc` tests with the `gfem.SolverUnavailableError` message.

    The probe is gfem's own, which initialises PETSc without pytest's command
    line, and it runs only when some test carries either marker.
    """
    marked = [
        (item, name)
        for item in items
        for name in ("petsc", "slepc")
        if item.get_closest_marker(name) is not None
    ]
    if not marked:
        return
    from gfem import _petsc
    from gfem.errors import SolverUnavailableError

    probes = {"petsc": _petsc.require_petsc, "slepc": _petsc.require_slepc}
    for item, name in marked:
        try:
            probes[name]()
        except SolverUnavailableError as err:
            item.add_marker(pytest.mark.skip(reason=str(err)))


@pytest.fixture
def ngsolve_threads():
    """``SetNumThreads`` is process-wide: put NGSolve's thread count back after."""
    import ngsolve as ngs

    with ngs.TaskManager():
        threads = ngs.GetNumThreads()
    yield
    ngs.SetNumThreads(threads)


def _reference_root() -> Path | None:
    ref = os.environ.get(REFERENCE_ENV)
    return Path(ref) if ref else None


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    """The reference data directory; skips the test when it is not set."""
    root = _reference_root()
    if root is None:
        pytest.skip(f"{REFERENCE_ENV} not set (point it at the reference data)")
    return root


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize ``palace_case`` over the reference cases with a mesh."""
    if "palace_case" not in metafunc.fixturenames:
        return
    root = _reference_root()
    if root is None:
        metafunc.parametrize(
            "palace_case",
            [
                pytest.param(
                    None,
                    id="no-reference-dir",
                    marks=pytest.mark.skip(reason=f"{REFERENCE_ENV} not set"),
                )
            ],
        )
        return
    cases = sorted(p.parent for p in root.glob("*/*/palace.msh"))
    metafunc.parametrize(
        "palace_case", cases, ids=[str(c.relative_to(root)) for c in cases]
    )
