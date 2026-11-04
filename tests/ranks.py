"""Run a test's function on every rank of a launched job (ADR 0070).

pytest stays one process: a hang or a one-rank failure under ranks would take it
with it, and nothing could time it out. `on_ranks` is the sibling of
`isolated.isolated`: the function is a module-level function of a Python file,
loaded by path on each rank; arguments and values travel by pickle, so they
must not use classes defined in that file.

Every returned value is compared bit by bit across ranks, unless told that it is
a rank's own piece, because a missing rank sum raises nothing and exits 0. Each
launch holds its own limit, a liveness bound and no gate on timing: it signals
the process group, sends SIGKILL after a grace, then sweeps what is left by a
per-launch tag, because a rank can outlive its launcher.

The launcher is ``GFEM_TEST_MPI_LAUNCH`` (for example ``srun --mpi=pmix``), or
`DEFAULT_LAUNCH`; ``-n N`` follows it either way. ``GFEM_TEST_MPI_MAX_RANKS`` skips
a launch above that many ranks, for a machine with few cores (GitHub's runners).
"""

import contextlib
import dataclasses
import os
import pickle
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np

#: ``--oversubscribe``: a ``--ntasks=1`` job has no slots otherwise. ``--bind-to
#: none``: a bound rank could not use ``threads`` above 1. Yielding: 8 ranks on 4
#: cores take 34 s with it and pass a 120 s limit without.
DEFAULT_LAUNCH = "mpirun --oversubscribe --bind-to none --mca mpi_yield_when_idle 1"
#: The variable that caps a launch's ranks.
MAX_RANKS = "GFEM_TEST_MPI_MAX_RANKS"
#: Seconds between SIGTERM and SIGKILL.
GRACE = 15
#: The environment variable that marks a launch's processes.
TAG = "GFEM_RANKS_TAG"
WORKER = Path(__file__).with_name("rank_main.py")
THREAD_VARIABLES = (
    "NGS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


class RanksFailed(AssertionError):  # noqa: N818
    """A launched job failed.

    Attributes:
        returncode: The launcher's return code.
        tracebacks: Each traceback a rank wrote, by rank.
        log: The tail of the job's merged output.
    """

    def __init__(self, returncode: int, tracebacks: dict[int, str], log: str) -> None:
        self.returncode = returncode
        self.tracebacks = tracebacks
        self.log = log
        shown = "\n".join(f"rank {r}:\n{t}" for r, t in sorted(tracebacks.items()))
        super().__init__(f"launcher exited {returncode}\n{shown}\n--- log ---\n{log}")


class RanksHung(AssertionError):  # noqa: N818
    """A launched job passed its limit.

    Attributes:
        log: The tail of the job's merged output.
        tag: The launch's tag, for `tagged`.
    """

    def __init__(self, limit: float, log: str, tag: str) -> None:
        self.log = log
        self.tag = tag
        super().__init__(f"still running after {limit} s\n--- log ---\n{log}")


class RanksDiffer(AssertionError):  # noqa: N818
    """A value that must be replicated differs across ranks."""


def launcher() -> list[str]:
    """The launcher's words, up to its ``-n``."""
    return shlex.split(os.environ.get("GFEM_TEST_MPI_LAUNCH", DEFAULT_LAUNCH))


def _find(program: str) -> str | None:
    """A program on ``PATH``, or beside this Python (a venv's ``openmpi``)."""
    return shutil.which(program) or shutil.which(
        program, path=str(Path(sys.executable).parent)
    )


def unavailable() -> str | None:
    """Why launched tests cannot run here, or ``None``.

    mpi4py loads in a child, so that pytest's process holds no MPI.
    """
    words = launcher()
    if _find(words[0]) is None:
        return f"no {words[0]} to launch ranks with"
    proc = subprocess.run(
        [sys.executable, "-c", "import mpi4py.MPI"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        return f"mpi4py cannot load MPI: {lines[-1] if lines else proc.returncode}"
    return None


def on_ranks(
    function, *args, ranks=4, threads=1, limit=120, replicated=True, **kwargs
) -> list:
    """``function(*args, **kwargs)`` on every rank of a launched job.

    Args:
        function: A module-level function of a Python file.
        *args: Its positional arguments, by pickle.
        ranks: The number of ranks.
        threads: The thread count each rank's libraries are told.
        limit: Seconds before the job is ended as hung.
        replicated: Compare each value bit by bit across ranks. A test of a
            rank's own piece passes ``False``.
        **kwargs: Its keyword arguments, by pickle.

    Returns:
        One value per rank, in rank order.

    Raises:
        RanksFailed: A rank raised, or the launcher exited nonzero.
        RanksHung: The job passed ``limit``.
        RanksDiffer: A value differs across ranks.
    """
    name = function.__qualname__
    if "<locals>" in name:
        raise ValueError(f"{name} is not a module-level function")
    with _Workdir() as work:
        call = (function.__code__.co_filename, name, args, kwargs)
        (work.path / "call.pkl").write_bytes(pickle.dumps(call))
        argv = [sys.executable, "-u", str(WORKER), str(work.path)]
        returncode, log = work.launch(argv, ranks, threads, limit)
        tracebacks = {
            int(p.stem.removeprefix("rank-")): p.read_text()
            for p in work.path.glob("rank-*.err")
        }
        missing = [
            r for r in range(ranks) if not (work.path / f"rank-{r}.pkl").exists()
        ]
        if returncode != 0 or tracebacks or missing:
            raise RanksFailed(returncode, tracebacks, log)
        values = [
            pickle.loads((work.path / f"rank-{r}.pkl").read_bytes())  # noqa: S301
            for r in range(ranks)
        ]
    if replicated:
        assert_same_bits(values)
    return values


def on_ranks_argv(
    argv, *, ranks=4, threads=1, limit=120, env=None, cwd=None
) -> tuple[int, str]:
    """Launch a command as it is, for a script or ``python -m gfem run``.

    Args:
        argv: The command each rank runs.
        ranks: The number of ranks.
        threads: The thread count each rank's libraries are told.
        limit: Seconds before the job is ended as hung.
        env: Extra environment variables.
        cwd: The folder the command runs in; ``None`` is this process's.

    Returns:
        The launcher's return code and the job's merged output.

    Raises:
        RanksHung: The job passed ``limit``.
    """
    with _Workdir() as work:
        return work.launch(list(argv), ranks, threads, limit, env=env, cwd=cwd)


def assert_same_bits(values: list) -> None:
    """Raise `RanksDiffer` naming the first leaf whose bits differ across ranks."""
    leaves = [dict(_leaves(value, "value")) for value in values]
    for r, other in enumerate(leaves[1:], start=1):
        if other.keys() != leaves[0].keys():
            raise RanksDiffer(f"rank {r} holds another shape of value than rank 0")
        for path, bits in leaves[0].items():
            if other[path] != bits:
                raise RanksDiffer(f"{path} differs between rank 0 and rank {r}")


def _leaves(value, path: str):
    """``(path, comparable bits)`` of every leaf of a nested value."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        for key in value:
            yield from _leaves(value[key], f"{path}[{key!r}]")
    elif isinstance(value, list | tuple):
        for i, item in enumerate(value):
            yield from _leaves(item, f"{path}[{i}]")
    elif value is None or isinstance(value, bool | int | str | bytes):
        yield path, repr(value)
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = None
        if array is None or array.dtype == object:
            yield path, pickle.dumps(value)
        else:
            yield path, (array.dtype.str, array.shape, array.tobytes())


class _Workdir:
    """A launch's directory and tag; its name is the tag."""

    def __enter__(self):
        self.tag = uuid.uuid4().hex
        self.path = Path(tempfile.mkdtemp(prefix=f"gfem-ranks-{self.tag}-"))
        return self

    def __exit__(self, *_):
        shutil.rmtree(self.path, ignore_errors=True)

    def launch(
        self, argv, ranks, threads, limit, *, env=None, cwd=None
    ) -> tuple[int, str]:
        cap = os.environ.get(MAX_RANKS)
        if cap is not None and ranks > int(cap):
            import pytest  # not on the ranks, which import this module

            pytest.skip(f"{ranks} ranks is above {MAX_RANKS}={cap}")
        words = launcher()
        words[0] = _find(words[0]) or words[0]
        command = [*words, "-n", str(ranks), *argv]
        environment = os.environ | {name: str(threads) for name in THREAD_VARIABLES}
        environment |= {TAG: self.tag, **(env or {})}
        # the venv's openmpi finds its daemon and helpers on PATH
        bin_dir = str(Path(command[0]).parent)
        environment["PATH"] = f"{bin_dir}{os.pathsep}{environment.get('PATH', '')}"
        log_path = self.path / "log"
        with log_path.open("wb") as log:
            job = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=environment,
                cwd=cwd,
                start_new_session=True,
            )
            try:
                returncode = job.wait(limit)
            except subprocess.TimeoutExpired:
                _end(job, self.tag)
                raise RanksHung(limit, _tail(log_path), self.tag) from None
            finally:
                _sweep(self.tag)
        return returncode, _tail(log_path)


def _end(job: subprocess.Popen, tag: str) -> None:
    """Signal the job's process group, then SIGKILL it after the grace."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(job.pid, signal.SIGTERM)
    try:
        job.wait(GRACE)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(job.pid, signal.SIGKILL)
        job.wait()
    _sweep(tag)


def tagged(tag: str) -> list[int]:
    """The processes of this user carrying a launch's tag."""
    needle = f"{TAG}={tag}".encode()
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            try:
                if needle in (entry / "environ").read_bytes():
                    found.append(int(entry.name))
            except OSError:
                continue
    return found


def _sweep(tag: str) -> None:
    """SIGKILL what outlived its launcher."""
    for pid in tagged(tag):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def _tail(path: Path, size: int = 8000) -> str:
    data = path.read_bytes()
    return data[-size:].decode(errors="replace")
