"""The scaling campaign's step files, one per Slurm job, and the plan that submits them.

    python scale_steps.py --meshes DIR --reference DIR/r10M_void --out STEPS

``--meshes`` holds the six cases `scale_void.py` and `split_mesh.py` made. Each
``STEPS/NAME.txt`` is what ``scale_job.sbatch`` reads. ``STEPS/plan.tsv`` gives each
file's row, nodes, QoS and time limit, 1.5 times the estimate over all its repeats; it
never names an account or a user. The rows, in the order they run:

    0  machine benchmark                     5  `direct` strong, r0.5_void_split
    1  r10M_void at the gates, references    6  BDDC weak, the rungs 2 and 4 lack
    2  BDDC strong, B2_void_split            7  `direct` weak, the same rungs
    3  `direct` attempt, B2_void_split       8  cross-check at tol=1e-10
    4  BDDC strong, r0.5_void_split

MS3 is repeat 1 of ``r2_bddc_32``. A row's ``*_more`` file or a second attempt is
submitted only if the stop rule asks: `direct` strong at 16 and 32 nodes takes more
repeats only when its first is faster than the rung below it. A ``r8_*`` file is a
choice: run the one for the largest mesh both solvers finished.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

#: Every ladder's layout: ranks per node, threads per rank.
RPN, CPT = 28, 4
REPEATS = 3
#: The QoS of rows 0 to 3, which run one after another, and of the rest.
SERIAL_QOS, OVERLAP_QOS = "gp_debug", "gp_ehpc"
LADDER = (1, 2, 4, 8, 16, 32)
#: One repeat's estimated wall, minutes: the plan's arithmetic, never a gate.
BDDC_108M = {32: 15, 16: 17, 8: 20, 4: 27}
BDDC_23M = dict.fromkeys(LADDER, 4)
DIRECT_23M = {1: 8, 2: 6, 4: 4, 8: 3, 16: 5, 32: 8}
WEAK_CASES = {1: "r0.5_void", 2: "B1_void", 4: "B2_void", 16: "B1_void_split"}
BDDC_WEAK = {1: 2, 2: 3, 4: 5, 16: 7}
DIRECT_WEAK = {1: 3, 2: 5, 4: 8, 16: 15}
#: The `direct` attempt at 108.6 M, ending in a kill or an answer: a fixed limit.
ATTEMPT_LIMIT = 45
#: The gates' jobs: one run's limit per solver and node count, minutes.
GATE_DIRECT = {1: 15, 2: 10, 4: 8, 8: 8, 16: 8, 32: 8}
GATE_BDDC = {1: 10, 2: 8, 4: 6, 8: 6, 16: 6, 32: 6}
REFERENCE_LIMIT = 45
#: A cross-check's limit, one BDDC run at the tighter tolerance.
CROSS_LIMIT = 25
#: The machine benchmark's limits: all thread counts on one node, then the network.
BENCH_LIMITS = {1: 15, 32: 5}
#: Every distributed run prints MUMPS's figures, whose coarse fill no API can read.
RUN_OPTIONS = "--mumps-print"


class Plan:
    """The step files being written and the plan's lines."""

    def __init__(self, out: Path) -> None:
        """Nothing is written yet, except the directory."""
        self.out = out
        self.lines = ["file\trow\tnodes\tqos\ttime_min"]
        out.mkdir(parents=True, exist_ok=True)

    def job(self, name: str, row: int, nodes: int, limit: float, *steps: str) -> None:
        """One Slurm job's file, and its line of the plan."""
        (self.out / f"{name}.txt").write_text("\n".join(steps) + "\n")
        qos = SERIAL_QOS if row <= 3 else OVERLAP_QOS
        self.lines.append(f"{name}\t{row}\t{nodes}\t{qos}\t{math.ceil(limit)}")

    def finish(self) -> None:
        """Write the plan."""
        (self.out / "plan.tsv").write_text("\n".join(self.lines) + "\n")


def _step(name: str, case: Path, solver: str, ghz: str = "50", extra: str = "") -> str:
    return f"{name} {RPN} {CPT} {case} {solver} {ghz} {RUN_OPTIONS} {extra}".rstrip()


def _repeats(name: str, case: Path, solver: str) -> list[str]:
    return [
        _step(f"{name}_r{k}", case, solver, extra=f"--repeat {k}")
        for k in range(1, REPEATS + 1)
    ]


def _bench(plan: Plan) -> None:
    plan.job(
        "r0_bench_1",
        0,
        1,
        BENCH_LIMITS[1],
        "bench r0_node 0 0 node --threads 1 4 28 56 112 --concurrent 28x4",
        f"bench r0_net_1 {RPN} {CPT} net",
    )
    plan.job("r0_bench_32", 0, 32, BENCH_LIMITS[32], f"bench r0_net_32 {RPN} {CPT} net")


def _gates(plan: Plan, reference: Path) -> None:
    plan.job(
        "r1_ref_direct",
        1,
        1,
        REFERENCE_LIMIT,
        f"ref_direct 0 112 {reference} direct 50,51",
    )
    plan.job(
        "r1_ref_bddc",
        1,
        1,
        REFERENCE_LIMIT,
        f"ref_bddc 0 112 {reference} bddc_gmres 50",
    )
    for nodes in LADDER:
        plan.job(
            f"r1_direct_{nodes}",
            1,
            nodes,
            GATE_DIRECT[nodes],
            _step(f"r1_direct_{nodes}", reference, "direct", "50,51"),
            f"check r1_direct_{nodes} ref_direct direct",
        )
        plan.job(
            f"r1_bddc_{nodes}",
            1,
            nodes,
            GATE_BDDC[nodes],
            _step(f"r1_bddc_{nodes}", reference, "bddc_gmres"),
            f"check r1_bddc_{nodes} ref_bddc bddc_gmres",
        )


def _strong(plan: Plan, meshes: Path) -> None:
    b2 = meshes / "B2_void_split"
    r05 = meshes / "r0.5_void_split"
    for nodes, minutes in BDDC_108M.items():
        name = f"r2_bddc_{nodes}"
        steps = _repeats(name, b2, "bddc_gmres")
        if nodes == max(BDDC_108M):  # MS3 (c): observed, no gate
            steps.insert(1, f"check {name}_r1 ref_bddc observe")
        plan.job(name, 2, nodes, 1.5 * REPEATS * minutes, *steps)
    plan.job(
        "r3_direct_32",
        3,
        32,
        ATTEMPT_LIMIT,
        _step("r3_direct_32", b2, "direct", extra="--repeat 1"),
    )
    plan.job(
        "r3_direct_32_8x14",
        3,
        32,
        ATTEMPT_LIMIT,
        f"r3_direct_32_8x14 8 14 {b2} direct 50 {RUN_OPTIONS} --repeat 1",
    )
    for nodes, minutes in BDDC_23M.items():
        name = f"r4_bddc_{nodes}"
        plan.job(
            name, 4, nodes, 1.5 * REPEATS * minutes, *_repeats(name, r05, "bddc_gmres")
        )
    for nodes, minutes in DIRECT_23M.items():
        name = f"r5_direct_{nodes}"
        steps = _repeats(name, r05, "direct")
        if nodes <= 8:
            plan.job(name, 5, nodes, 1.5 * REPEATS * minutes, *steps)
        else:
            plan.job(name, 5, nodes, 1.5 * minutes, steps[0])
            plan.job(f"{name}_more", 5, nodes, 1.5 * 2 * minutes, *steps[1:])


def _weak(plan: Plan, meshes: Path) -> None:
    for nodes, case in WEAK_CASES.items():
        name = f"r6_bddc_{nodes}"
        plan.job(
            name,
            6,
            nodes,
            1.5 * REPEATS * BDDC_WEAK[nodes],
            *_repeats(name, meshes / case, "bddc_gmres"),
        )
    for nodes, case in WEAK_CASES.items():
        name = f"r7_direct_{nodes}"
        steps = _repeats(name, meshes / case, "direct")
        minutes = DIRECT_WEAK[nodes]
        if case.endswith("_split"):  # 47 M: a step that may be killed runs alone
            for k, step in enumerate(steps, start=1):
                plan.job(f"{name}_r{k}", 7, nodes, 1.5 * minutes, step)
        else:
            plan.job(name, 7, nodes, 1.5 * REPEATS * minutes, *steps)


def _cross_checks(plan: Plan, meshes: Path) -> None:
    candidates = (
        ("B2_void_split", 32, "r3_direct_32"),
        ("B1_void_split", 16, "r7_direct_16_r1"),
        ("r0.5_void_split", 32, "r5_direct_32_r1"),
    )
    for case, nodes, direct in candidates:
        name = f"r8_{case}"
        plan.job(
            name,
            8,
            nodes,
            CROSS_LIMIT,
            _step(name, meshes / case, "bddc_gmres", extra="--tol 1e-10"),
            f"check {name} {direct} bddc_gmres",
        )


def write(meshes: Path, reference: Path, out: Path) -> None:
    """Every row's step files and the plan."""
    plan = Plan(out)
    _bench(plan)
    _gates(plan, reference)
    _strong(plan, meshes)
    _weak(plan, meshes)
    _cross_checks(plan, meshes)
    plan.finish()


def main() -> None:
    """Parse the directories and write."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meshes", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write(args.meshes, args.reference, args.out)


if __name__ == "__main__":
    main()
