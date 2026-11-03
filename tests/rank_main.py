"""What each rank of `ranks.on_ranks` runs: one call, its value written by rank.

A rank that raises writes its traceback and aborts the job, so a failed assertion
on one rank cannot hang its peers. No collective sits behind a test on the rank.

mpi4py is imported inside `main`: a frequency worker re-imports this file, and on
Open MPI 4.1.6 ``MPI_Init`` in that child kills it.
"""

import importlib.util
import pickle
import sys
import traceback
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def main(work: Path) -> None:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    try:
        sys.path[:0] = [str(TESTS), str(TESTS.parent / "tools")]
        path, name, args, kwargs = pickle.loads((work / "call.pkl").read_bytes())  # noqa: S301
        spec = importlib.util.spec_from_file_location("_ranks_target", path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        value = getattr(module, name)(*args, **kwargs)
        # by rename, so that the parent never reads half a file
        partial = work / f"rank-{rank}.part"
        partial.write_bytes(pickle.dumps(value))
        partial.rename(work / f"rank-{rank}.pkl")
    except BaseException:  # noqa: BLE001 - reported to the parent, which raises
        (work / f"rank-{rank}.err").write_text(traceback.format_exc())
        print(traceback.format_exc(), file=sys.stderr, flush=True)  # noqa: T201
        MPI.COMM_WORLD.Abort(1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
