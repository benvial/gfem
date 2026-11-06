# Contributing

This page is for working on gfem itself: building it, running its tests and
rebuilding the docs. To use gfem, start at {doc}`../get-started`.

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run ty check src
uv run pytest -m "unit or verification"
```

Test tiers are pytest markers set by directory: `tests/unit`,
`tests/verification`, `tests/reference`, `tests/benchmark`, `tests/cuda` and
`tests/mpi`. The `reference`, `benchmark`, `cuda` and `mpi` tiers are deselected by
default; run them with `uv run pytest -m reference` (and so on). Reference tests read their
data from `GFEM_REFERENCE_DIR` and skip when it is unset.
{doc}`../verification` describes each verification case in physical terms and
links the test file that runs it.

Tests marked `petsc` and `slepc` sit in the unit and verification tiers and
skip unless the {term}`PETSc stack` is installed. To run them, add the
wheelhouse that {doc}`../recipes/petsc-slepc` builds:

```bash
uv sync
uv pip install --no-index --find-links wheelhouse petsc petsc4py slepc slepc4py
uv run pytest -m "unit or (verification and (petsc or slepc))" -rs
```

`uv run` leaves the four wheels in place. A plain `uv sync`, the docs one
below included, removes them again, since the lock file does not list them.
Sync with `uv sync --inexact` to keep them, or rerun the `uv pip install` line
after an exact sync.

Launched tests, in `tests/mpi`, start each test's function on several ranks from a
pytest that stays one process. They need Open MPI on `PATH` and the `mpi` extra, and
skip without them:

```bash
uv sync --extra mpi
uv run pytest -m "mpi and not reference" -rs
```

`GFEM_TEST_MPI_LAUNCH` replaces the launcher (`srun --mpi=pmix`), and
`GFEM_TEST_MPI_MAX_RANKS` skips launches above a count, for a machine with few cores.
The direct cases need the {term}`MPI PETSc stack`, which the script builds with
`--mpi`, and they skip without it:

```bash
uv sync --extra mpi
uv pip install --no-index --find-links wheelhouse-mpi petsc petsc4py slepc slepc4py
uv run pytest -m "unit or (verification and (petsc or slepc))" -rs
uv run pytest -m "mpi and not reference" -rs
```

Keep these two selections in separate pytest commands: a selected `petsc` or `slepc`
test loads MPI in pytest's own process, and a launched job under `srun` then fails.
Collection refuses launched tests that carry those markers; they skip on
`_petsc.has_mpi()` instead.

## Licence gate

CI's licence gate, `tools/licence_gate.py`, audits the installed dependency tree
and gfem's wheel. It fails on non-free and non-commercial licences. GPL and LGPL
dependencies pass. Proprietary licences pass only where `tools/licences.toml`
allows them: the NVIDIA runtime wheels of the `cuda12` and `cuda13` extras, and
the recorded Intel licence of each distribution of the `mkl` extra. A distribution
whose licence cannot be read from its metadata fails until you record it in
`tools/licences.toml`; PyPI's `openmpi` wheel is one, so the `mpi` extra leaves it
out. PETSc and SLEPc are built from source and not declared by gfem, so the gate
does not see them. A `petsc`-marked test checks that ParMETIS is absent from both
stacks, and PT-Scotch and ScaLAPACK from the no-MPI stack.

## Docs

The docs site needs Python 3.13 and [just](https://just.systems):

```bash
uv sync --python 3.13 --group docs --group site
just docs   # strict build and search index, into docs/_build/html
just serve  # the same, then serve it at http://localhost:8000
```

The build executes no notebook. The tutorials' outputs and the front page's
hero data (`docs/_data/front-page.json`) are committed; after changing a
tutorial, regenerate them and commit the result:

```bash
just notebooks        # re-execute docs/tutorials/*.ipynb in place, docs figure style
just front-page-data  # resample the 3D bend's run record for the front page's hero data
```

The front page's video, `docs/_static/front/bend3d-optimization-*`, is rendered
from a second, uncommitted record of the 3D bend that keeps every iteration's
design and fields (`tools/bend3d_history.py`, about 3.2 h on 16 cores /
96 GiB). Rerender it from that record on a machine with a GPU:

```bash
just bend3d-history WORK      # the record, into WORK
just bend3d-animation WORK    # the light and dark videos, posters and timing
```

The 3D bend tutorial solves nothing: it reads a committed run record,
`docs/tutorials/data/bend3d.json` and `bend3d.npz`, written by
`tools/bend3d_run.py`. The page stops when that script or `tools/bend3d_mesh.py`
no longer matches the hashes in the record. After changing either, rerun the
script and commit the new record:

```bash
just bend3d-data      # about 2.7 h on 16 cores / 96 GiB, with the mkl extra
```
