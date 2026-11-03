"""The command-line runner: ``python -m gfem run job.json --out DIR``.

Exit codes: 0 on success; 1 when the job cannot be read or solved (the
error message goes to stderr: a changed mesh file, a newer schema version,
a device that cannot run, a physics error, ...); 2 for command-line usage
errors, including a job file that does not exist.

Under a launcher rank 0 reads the job and writes the result, all ranks exit together,
with rank 0 reporting; a failure that only one rank sees ends the whole job, because
its peers may never reach an exit.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    from gfem import __version__

    parser = argparse.ArgumentParser(
        prog="gfem",
        description="Run serialized gfem jobs.",
    )
    parser.add_argument("--version", action="version", version=f"gfem {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    run = commands.add_parser(
        "run",
        help="solve a job bundle and write a result bundle",
        description=(
            "Solve the job in JOB (a job.json written by gfem.Job.to_json) and "
            "write result.json and result.npz to the --out directory; read them "
            "back with gfem.load_result(DIR)."
        ),
    )
    run.add_argument("job", type=Path, metavar="JOB", help="the job.json file")
    run.add_argument(
        "--out", type=Path, required=True, metavar="DIR", help="the result directory"
    )
    run.add_argument(
        "--device",
        metavar="DEVICE",
        help="override the job's SolverOptions.device: cpu, cuda or cuda:N "
        "(recorded in result.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code.

    Args:
        argv: The arguments without the program name; ``None`` reads
            ``sys.argv``.
    """
    parser = _parser()
    args = parser.parse_args(argv)

    from gfem import _mpi
    from gfem.errors import SolverUnavailableError
    from gfem.job import Job

    try:
        comm = _mpi.resolve_comm()
    except SolverUnavailableError as err:  # no communicator, so each rank reports
        print(f"gfem: error: {type(err).__name__}: {err}", file=sys.stderr)  # noqa: T201
        return 1
    if _mpi.agree_any(comm, flag=not args.job.is_file()):
        if _mpi.rank(comm) == 0:
            parser.error(f"job file {args.job} does not exist")
        return 2
    try:
        with _mpi.section(comm):
            job = Job.from_json(args.job, comm=comm)
            job.run(args.out, device=args.device, comm=comm)
    except Exception as err:  # noqa: BLE001 (report any failure as exit code 1)
        if _mpi.rank(comm) == 0:
            print(f"gfem: error: {type(err).__name__}: {err}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
