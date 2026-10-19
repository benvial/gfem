"""The command-line runner: ``python -m gfem run job.json --out DIR``.

Exit codes: 0 on success; 1 when the job cannot be read or solved (the
error message goes to stderr: a changed mesh file, a newer schema version,
a device that cannot run, a physics error, ...); 2 for command-line usage
errors, including a job file that does not exist.
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
    if not args.job.is_file():
        parser.error(f"job file {args.job} does not exist")

    from gfem.job import Job

    try:
        Job.from_json(args.job).run(args.out, device=args.device)
    except Exception as err:  # noqa: BLE001 (report any failure as exit code 1)
        print(f"gfem: error: {type(err).__name__}: {err}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
