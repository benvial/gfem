"""The scaling page's tables, from the scaling set's run records, as committed snippets.

    python scale_tables.py RECORDS OUT --commit 9a57680 [--logs LOGS] [--mesh-md5 FILE]

Reads every ``*.json`` in ``RECORDS`` that ``scale_run.py`` and ``scale_bench.py`` wrote
(superseded ones aside) and writes one Markdown snippet per table of `CAMPAIGN` into
``OUT``, which ``docs/project/scaling.md`` includes. The docs build never runs this
script, opens a record or computes a number (ADR 0017's amendment); a person reruns it
and diffs. Each snippet's first line names the commit and the SHA-256 of the records
manifest: one line per record file, its SHA-256 and its name, sorted by name.

It reads times only to print them, and asserts nothing about a time. It refuses, naming
the record: a record of another commit, a table's record that is missing, a ``check``
whose gate did not pass, and a finished run whose ranks did not agree bit for bit.

``--logs`` is the directory of the steps' logs (``NAME-JOB.log``), read for a run stub's
MUMPS error code and Slurm's ``oom_kill``. ``--mesh-md5``, repeatable, is ``md5sum``
output over the meshes' ``palace.msh`` (``#`` lines skipped), matched to a record's mesh
by the file's parent directory.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

Record = dict[str, Any]

#: The tables of the scaling page, rows 0 to 8 of the campaign's plan, in its names.
CAMPAIGN: dict[str, dict[str, Any]] = {
    "machine": {"kind": "bench", "records": ["r0_node", "r0_net_1", "r0_net_32"]},
    "meshes": {"kind": "meshes", "records": ["r6_bddc_*", "r4_bddc_*", "r2_bddc_*"]},
    "ms3": {"kind": "ms3", "records": ["r2_bddc_32_r1"]},
    "agreement": {
        "kind": "agreement",
        "records": ["r1_direct_*", "r1_bddc_*", "r8_*", "r2_bddc_32_r1"],
    },
    "strong-108m": {
        "kind": "ladder",
        "records": ["r2_bddc_*", "r3_direct_32"],
        "base": 4,
    },
    "strong-23m": {
        "kind": "ladder",
        "records": ["r4_bddc_*", "r5_direct_*"],
        "base": 1,
    },
    "weak": {
        "kind": "ladder",
        "weak": True,
        "base": 1,
        "records": [
            "r6_bddc_1_*",
            "r6_bddc_2_*",
            "r6_bddc_4_*",
            "r4_bddc_8_*",
            "r6_bddc_16_*",
            "r2_bddc_32_*",
            "r7_direct_1_*",
            "r7_direct_2_*",
            "r7_direct_4_*",
            "r5_direct_8_*",
            "r7_direct_16_*",
            "r3_direct_32",
        ],
    },
    "direct-108m": {"kind": "attempt", "records": ["r3_direct_32"]},
    "phases": {
        "kind": "phases",
        "records": ["r2_bddc_*", "r6_bddc_16_*", "r4_bddc_32_*", "r1_bddc_32"],
    },
}

TOL_NOTE = "`tol` is GMRES's relative residual; it does not apply to `direct`."
SOLVE_NOTE = (
    "`gfem.solve` is the slowest rank's whole call: spaces, ports, preconditioner or "
    "factorization, and GMRES, with the instrumentation's true residual and one extra "
    "apply and matvec inside it. The mesh read and the partition come before it."
)


def _refuse(message: str) -> None:
    raise SystemExit(f"scale_tables.py: {message}")


def _superseded(path: Path) -> bool:
    return ".superseded-" in path.name


def load(directory: Path, commit: str) -> dict[str, Record]:
    """Every record of the directory by name, after the checks no table may skip."""
    records = {}
    for path in sorted(directory.glob("*.json")):
        if _superseded(path):
            continue
        record = json.loads(path.read_text())
        name = path.name.removesuffix(".json")
        if not str(record.get("commit", "")).startswith(commit):
            _refuse(f"{name} is of commit {record.get('commit')}, not {commit}")
        for check in record.get("checks") or []:
            if check.get("gate") is not None and check.get("passed") is not True:
                _refuse(f"{name} did not pass its gate {check['gate']}: {check}")
        if not record.get("failed") and record.get("ranks_bitwise") is False:
            _refuse(f"{name}: the ranks differ in {record.get('ranks_differ_in')}")
        records[name] = record
    return records


def manifest(directory: Path) -> str:
    """The records manifest: SHA-256 and name of each record file, sorted by name."""
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(directory.glob("*.json"))
        if not _superseded(path)
    ]
    return "".join(f"{line}\n" for line in lines)


def _select(records: dict[str, Record], patterns: Iterable[str]) -> list[Record]:
    chosen: dict[str, Record] = {}
    for pattern in patterns:
        names = sorted(n for n in records if fnmatch.fnmatchcase(n, pattern))
        if not names:
            _refuse(f"no record matches {pattern}")
        chosen.update((n, records[n]) for n in names)
    return list(chosen.values())


# --- cells ---------------------------------------------------------------------------


def _mesh(record: Record) -> str:
    return Path(record.get("case") or "unknown").name


def _seconds(record: Record, phase: str) -> float | None:
    return (record.get("phases") or {}).get(phase, {}).get("max_s")


def _median(rung: list[Record], read: Callable[[Record], float | None]) -> float | None:
    values = [v for v in map(read, rung) if v is not None]
    return float(statistics.median(values)) if values else None


def _s(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _k(value: float) -> str:
    return f"{value / 1e3:.0f} k"


def _millions(value: float) -> str:
    return f"{value / 1e6:.1f} M"


def _tol(record: Record) -> str:
    if record.get("solver") != "bddc_gmres":
        return "-"
    tol = (record.get("options") or {}).get("tolerance")
    return "-" if tol is None else f"{tol:g}"


def _layout(record: Record) -> str:
    return f"{record['ranks']} x {record['threads_read']}"


def _jobs(rung: list[Record]) -> str:
    return ", ".join(dict.fromkeys(str(r.get("job")) for r in rung))


def _iterations(rung: list[Record]) -> str:
    seen = dict.fromkeys(
        "/".join(map(str, row))
        for r in rung
        for row in (r.get("iterations") or {}).get("iterations", [])
        if row
    )
    return ", ".join(seen) or "-"


def _dof_of_meshes(records: dict[str, Record]) -> dict[str, int]:
    """Each mesh's DOF, read from its finished runs across ranks."""
    dofs: dict[str, int] = {}
    for r in records.values():
        total = (r.get("dofs") or {}).get("owned_sum")
        if total and not r.get("failed"):
            dofs.setdefault(_mesh(r), int(total))
    return dofs


def _table(titles: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    titles = list(titles)
    lines = ["| " + " | ".join(titles) + " |", "|" + "---|" * len(titles)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines) + "\n"


# --- ladders -------------------------------------------------------------------------


def _rung_key(record: Record) -> tuple:
    return (
        _mesh(record),
        record["solver"],
        record["node_count"],
        record["ranks"],
        record["threads_read"],
    )


def _rungs(chosen: list[Record]) -> list[list[Record]]:
    """Finished repeats by rung, and each failed run alone, by solver and size."""
    solvers = list(dict.fromkeys(r["solver"] for r in chosen))
    groups: dict[tuple, list[Record]] = {}
    for r in chosen:
        key = (
            *_rung_key(r),
            bool(r.get("failed")),
            r["name"] if r.get("failed") else "",
        )
        groups.setdefault(key, []).append(r)
    return [
        groups[key]
        for key in sorted(groups, key=lambda k: (solvers.index(k[1]), k[2], k[5], k[6]))
    ]


def _efficiency(
    rung: list[Record], rungs: list[list[Record]], base: int | None, weak: bool
) -> str:
    first, seconds = rung[0], _median(rung, lambda r: _seconds(r, "solve"))
    if base is None or seconds is None:
        return "-"
    if first["node_count"] == base:
        return "base"
    for other in rungs:
        one = other[0]
        if one.get("failed") or one["node_count"] != base:
            continue
        if one["solver"] != first["solver"] or (
            not weak and _mesh(one) != _mesh(first)
        ):
            continue
        reference = _median(other, lambda r: _seconds(r, "solve"))
        if reference is None:
            return "-"
        factor = 1 if weak else first["node_count"] / base
        return f"{reference / (seconds * factor):.2f}"
    return "-"


def _failure(record: Record, logs: Path | None) -> str:
    """Slurm's state, the phase the trace left open, and what the log adds."""
    text = (
        f"{record.get('state', 'FAILED')} in `{record.get('last_phase') or 'no phase'}`"
    )
    log = _log(record, logs)
    if log is not None:
        codes = sorted(set(re.findall(r"INFO\(1\)=\s*(-\d+)", log)) - {"-1"})
        if codes:
            text += f", MUMPS INFO(1) = {', '.join(codes)}"
        if "oom_kill" in log:
            text += ", then Slurm's oom_kill"
    return text


def _log(record: Record, logs: Path | None) -> str | None:
    if logs is None:
        return None
    path = logs / f"{record['name']}-{record.get('job')}.log"
    return path.read_text(errors="replace") if path.is_file() else None


def ladder(
    chosen: list[Record],
    records: dict[str, Record],
    *,
    base: int | None,
    weak: bool = False,
    logs: Path | None = None,
) -> str:
    """One row per rung, with each failed run as a row of its own."""
    dofs = _dof_of_meshes(records)
    rungs = _rungs(chosen)
    rows = []
    for rung in rungs:
        first = rung[0]
        total = dofs.get(_mesh(first))
        nodes, ranks = first["node_count"], first["ranks"]
        per_rank = _k(total / ranks) if total else "-"
        spread = "-"
        if not first.get("failed"):
            low = min(r["dofs"]["owned_min"] for r in rung)
            high = max(r["dofs"]["owned_max"] for r in rung)
            spread = f"{low / 1e3:.0f}-{high / 1e3:.0f} k"
        if first.get("failed"):
            repeats, median, span, coarse, memory = (
                "1",
                _failure(first, logs),
                "-",
                "-",
                "-",
            )
        else:
            solve = [_seconds(r, "solve") for r in rung]
            repeats = str(len(rung))
            median = _s(_median(rung, lambda r: _seconds(r, "solve")))
            span = f"{min(solve):.0f}-{max(solve):.0f}" if len(rung) > 1 else "-"
            coarse = _s(_median(rung, lambda r: _seconds(r, "coarse_apply")))
            memory = (
                f"{max(r['memory']['peak_rss_largest_node_gib'] for r in rung):.0f}"
            )
        rows.append(
            (
                nodes,
                _layout(first),
                f"`{first['solver']}`",
                _mesh(first),
                f"{total:,}" if total else "-",
                per_rank,
                spread,
                _millions(total / nodes) if total else "-",
                _tol(first),
                repeats,
                "-" if first.get("failed") else _iterations(rung),
                median,
                span,
                coarse,
                _efficiency(rung, rungs, base, weak),
                memory,
                _jobs(rung),
            )
        )
    titles = (
        "Nodes",
        "Ranks x threads",
        "Solver",
        "Mesh",
        "DOF",
        "DOF per rank",
        "Owned per rank, min-max",
        "DOF per node",
        "`tol`",
        "Repeats",
        "Iterations, ports 1/2",
        "`gfem.solve` median, s",
        "Range, s",
        "Coarse apply median, s",
        f"Efficiency against {base} node{'s' if base != 1 else ''}"
        + (" (weak)" if weak else ""),
        "Largest node, GiB",
        "Job",
    )
    kind = (
        "time at the base over time here"
        if weak
        else "time at the base times its nodes, over time here times these nodes"
    )
    notes = (
        f"\n{SOLVE_NOTE} Efficiency is the median's {kind}, same solver"
        + ("" if weak else " and mesh")
        + f". {TOL_NOTE} A failed run is its own row, with Slurm's state and the phase "
        "its trace left open; its DOF are the mesh's.\n"
    )
    return _table(titles, rows) + notes


# --- the other tables ----------------------------------------------------------------


def agreement(chosen: list[Record], records: dict[str, Record]) -> str:
    """Each run's S against its reference, at the gate or observed."""
    dofs = _dof_of_meshes(records)
    solvers = list(dict.fromkeys(r["solver"] for r in chosen))
    chosen = sorted(
        chosen,
        key=lambda r: (
            dofs.get(_mesh(r), 0),
            solvers.index(r["solver"]),
            r["node_count"],
        ),
    )
    rows = []
    for r in chosen:
        for check in r.get("checks") or []:
            ref_name = Path(check["reference"]).name
            ref = records.get(ref_name, {})
            gate = check.get("gate")
            same = check.get("same_iterations")
            rows.append(
                (
                    _mesh(r),
                    f"{dofs.get(_mesh(r), 0):,}",
                    f"`{r['solver']}`",
                    r["node_count"],
                    _layout(r),
                    _k(dofs[_mesh(r)] / r["ranks"]) if _mesh(r) in dofs else "-",
                    _tol(r),
                    ", ".join(f"{f:g}" for f in r.get("frequencies_ghz", [])),
                    (
                        f"`{ref.get('solver', '?')}` on {_mesh(ref) if ref else '?'}, "
                        f"{_layout(ref) if ref else '?'}, "
                        f"{_tol(ref) if ref else '?'}, job {ref.get('job', '?')}"
                    ),
                    f"{check['differences']['s']:.2g}",
                    "observed, no gate" if gate is None else f"{gate:g}",
                    "-" if same is None else ("same" if same else "differ"),
                    "yes" if r.get("ranks_bitwise") else "no",
                    r.get("job"),
                )
            )
    titles = (
        "Mesh",
        "DOF",
        "Solver",
        "Nodes",
        "Ranks x threads",
        "DOF per rank",
        "`tol`",
        "GHz",
        "Reference: solver, mesh, ranks x threads, `tol`, job",
        "Difference in S",
        "Gate",
        "Iterations",
        "S bitwise on all ranks",
        "Job",
    )
    note = (
        "\nThe difference is the largest entry of S's difference over the largest "
        "entry of the reference's S. A reference's `tol` of `-` is `direct`. "
        + TOL_NOTE
        + "\n"
    )
    return _table(titles, rows) + note


def ms3(chosen: list[Record]) -> str:
    """The milestone run, in one line of facts."""
    (r,) = chosen
    solve = _seconds(r, "solve")
    return (
        f"`{r['solver']}` on {_mesh(r)}: {r['dofs']['owned_sum']:,} DOF, "
        f"{r['node_count']} nodes, {r['ranks']} ranks of {r['threads_read']} threads, "
        f"`tol={_tol(r)}`, iterations {_iterations([r])} for ports 1/2, "
        f"`gfem.solve` {_s(solve)} s, S "
        + ("bit for bit the same on all ranks" if r.get("ranks_bitwise") else "differs")
        + f", job {r['job']}.\n"
    )


def attempt(chosen: list[Record], logs: Path | None) -> str:
    """A run stub's MUMPS analyses, one row per printed analysis."""
    rows = []
    for r in chosen:
        for i, printed in enumerate(r.get("mumps_printed") or [], 1):
            rows.append(
                (
                    _mesh(r),
                    r["node_count"],
                    _layout(r),
                    i,
                    f"{printed['n']:,}",
                    f"{printed['entries_estimated']:.3g}",
                    f"{printed['front_estimated']:,}",
                    f"{printed['flops_estimated']:.3g}",
                    f"{printed['total_ic_mb'] / 1e6:.2f}",
                    f"{printed['max_ic_mb'] / 1e3:.1f}",
                    _failure(r, logs) if r.get("failed") else "finished",
                    r.get("job"),
                )
            )
    titles = (
        "Mesh",
        "Nodes",
        "Ranks x threads",
        "Analysis",
        "Order n",
        "Factor entries, estimated",
        "Largest front, estimated",
        "Flops, estimated",
        "Total memory, TB (INFOG(17))",
        "Largest rank, GB",
        "End",
        "Job",
    )
    note = (
        "\nMUMPS's own printed estimates, in-core, at `ICNTL(14)=60`. A second "
        "analysis is the retry gfem makes after MUMPS reports too little memory.\n"
    )
    return _table(titles, rows) + note


def phases(chosen: list[Record], records: dict[str, Record]) -> str:
    """Each phase's median over a rung's finished repeats, and the Krylov shares."""
    dofs = _dof_of_meshes(records)
    rows = []
    finished = _rungs([r for r in chosen if not r.get("failed")])
    finished.sort(key=lambda rung: (-dofs[_mesh(rung[0])], rung[0]["node_count"]))
    for rung in finished:
        first = rung[0]

        def median(read: Callable[[Record], float | None], rung=rung) -> float | None:
            return _median(rung, read)

        solve = median(lambda r: _seconds(r, "solve"))
        gmres = median(lambda r: _seconds(r, "gmres"))
        coarse_apply = median(lambda r: _seconds(r, "coarse_apply"))
        wait = median(lambda r: (r.get("krylov") or {}).get("reduction_wait_s"))
        vector = median(lambda r: (r.get("krylov") or {}).get("vector_share"))
        coarse = (first.get("coarse") or [{}])[0]
        printed = [
            p["entries"]
            for r in rung
            for p in r.get("mumps_printed") or []
            if "entries" in p
        ]
        rows.append(
            (
                _mesh(first),
                first["node_count"],
                _layout(first),
                _k(dofs[_mesh(first)] / first["ranks"]),
                _tol(first),
                _s(median(lambda r: _seconds(r, "mesh_read"))),
                _s(median(lambda r: _seconds(r, "partition"))),
                _s(median(lambda r: _seconds(r, "spaces"))),
                _s(median(lambda r: _seconds(r, "coarse_setup"))),
                _s(gmres),
                _s(coarse_apply),
                _s(solve),
                f"{coarse_apply / solve:.0%}" if coarse_apply and solve else "-",
                "-" if vector is None else f"{vector:.1%}",
                "-" if wait is None else f"{wait:.1f}",
                f"{wait / gmres:.1%}" if wait is not None and gmres else "-",
                f"{wait / solve:.1%}" if wait is not None and solve else "-",
                f"{coarse['dofs']:,}" if coarse.get("dofs") else "-",
                coarse.get("ranks", "-"),
                f"{max(printed):.3g}" if printed else "-",
                len(rung),
                _jobs(rung),
            )
        )
    titles = (
        "Mesh",
        "Nodes",
        "Ranks x threads",
        "DOF per rank",
        "`tol`",
        "Mesh read, s",
        "Partition, s",
        "Spaces, s",
        "Coarse setup, s",
        "GMRES, s",
        "Coarse apply, s",
        "`gfem.solve`, s",
        "Coarse apply in `gfem.solve`",
        "Krylov vector share of GMRES",
        "Reduction wait, s",
        "Reduction wait in GMRES",
        "Reduction wait in `gfem.solve`",
        "Coarse DOF",
        "Coarse ranks",
        "Coarse factor entries",
        "Repeats",
        "Job",
    )
    note = (
        "\nMedians over the finished repeats, and shares of those medians. Phases are "
        "the slowest rank's. GMRES holds the coarse apply; the coarse setup is its "
        "factorization. The Krylov vector share is the time in GMRES's vector work, "
        "reductions included, over GMRES's time, the median of the repeats' shares; "
        "the reduction wait is the time in its `Allreduce` calls. Coarse factor "
        "entries are MUMPS's printed count. " + TOL_NOTE + "\n"
    )
    return _table(titles, rows) + note


def meshes(
    chosen: list[Record], records: dict[str, Record], md5: dict[str, str]
) -> str:
    """Each mesh by its tetrahedra, DOF and checksum."""
    dofs = _dof_of_meshes(records)
    seen: dict[str, Record] = {}
    for r in chosen:
        if not r.get("failed"):
            seen.setdefault(_mesh(r), r)
    rows = [
        (
            name,
            f"{r['tetrahedra']:,}" if r.get("tetrahedra") else "-",
            f"{dofs[name]:,}",
            f"`{md5[name]}`" if name in md5 else "-",
        )
        for name, r in sorted(seen.items(), key=lambda item: dofs[item[0]])
    ]
    note = "\nOrder 2 Nedelec DOF as the runs counted them; md5 of `palace.msh`.\n"
    return _table(("Mesh", "Tetrahedra", "DOF", "md5"), rows) + note


def bench(chosen: list[Record]) -> str:
    """The machine benchmark: one node by threads, then the network."""
    node = next(r for r in chosen if r.get("mode") == "node")
    nets = [r for r in chosen if r.get("mode") == "net"]
    rows = [
        (
            row["threads"],
            f"{row['dgemm']['gflops_median']:.0f}",
            f"{row['zgemm']['gflops_median']:.0f}",
            f"{row['triad']['gbs_median']:.0f}",
        )
        for row in node["rows"]
    ]
    together = node.get("concurrent")
    if together:
        rows.append(
            (
                f"{together['layout']}, side by side",
                "-",
                "-",
                f"{together['gbs_sum_of_medians']:.0f}",
            )
        )
    text = (
        f"One node ({node['host']}, {node['cpu']}), job {node['job']}: MKL `dgemm` "
        f"(n={node['rows'][0]['dgemm']['n']}), `zgemm` "
        f"(n={node['rows'][0]['zgemm']['n']}) and a STREAM triad, medians.\n\n"
    )
    text += _table(("Threads", "dgemm, GFLOP/s", "zgemm, GFLOP/s", "Triad, GB/s"), rows)
    big = max(nets, key=lambda r: r["node_count"])
    sizes = [p["bytes"] for p in big["ping_same_node"]]
    other = {p["bytes"]: p for p in big["ping_other_node"]}
    same = {p["bytes"]: p for p in big["ping_same_node"]}
    rows = [
        (
            f"{size:,}",
            f"{same[size]['median_s'] * 1e6:.1f}",
            f"{same[size]['gbs']:.2f}",
            f"{other[size]['median_s'] * 1e6:.1f}" if size in other else "-",
            f"{other[size]['gbs']:.2f}" if size in other else "-",
        )
        for size in sizes
    ]
    text += (
        f"\nPing-pong from rank 0, {big['ranks']} ranks on {big['node_count']} nodes, "
        f"job {big['job']}, {big['mpi_library'].split(',')[0]}, medians.\n\n"
    )
    text += _table(
        (
            "Bytes",
            "Same node, us",
            "Same node, GB/s",
            "Other node, us",
            "Other node, GB/s",
        ),
        rows,
    )
    nets = sorted(nets, key=lambda r: r["ranks"])
    counts = [a["doubles"] for a in nets[0]["allreduce"]]
    rows = [
        (
            f"{r['ranks']} on {r['node_count']}",
            *(
                f"{a['median_s_slowest_rank'] * 1e6:.1f}"
                for a in sorted(r["allreduce"], key=lambda a: a["doubles"])
            ),
            r["job"],
        )
        for r in nets
    ]
    text += "\n`Allreduce`, the slowest rank's median.\n\n"
    text += _table(
        ("Ranks on nodes", *(f"{n:,} doubles, us" for n in counts), "Job"),
        rows,
    )
    return text


# --- the snippets --------------------------------------------------------------------


def _md5s(paths: Iterable[Path]) -> dict[str, str]:
    found = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                digest, name = line.split(maxsplit=1)
                found[Path(name.strip()).parent.name] = digest
    return found


def write_snippets(
    records_dir: Path,
    out: Path,
    *,
    commit: str,
    spec: dict[str, dict[str, Any]] = CAMPAIGN,
    logs: Path | None = None,
    mesh_md5: Iterable[Path] = (),
) -> list[Path]:
    """Write one snippet per table of ``spec`` into ``out``; return their paths."""
    records = load(records_dir, commit)
    digest = hashlib.sha256(manifest(records_dir).encode()).hexdigest()
    header = (
        "% Generated by tools/scale_tables.py from the run records of commit "
        f"{commit}, records manifest sha256 {digest}. Do not edit: rerun the "
        "script.\n\n"
    )
    md5 = _md5s(mesh_md5)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in spec.items():
        chosen = _select(records, table["records"])
        kind = table["kind"]
        if kind == "ladder":
            body = ladder(
                chosen,
                records,
                base=table.get("base"),
                weak=table.get("weak", False),
                logs=logs,
            )
        elif kind == "agreement":
            body = agreement(chosen, records)
        elif kind == "ms3":
            body = ms3(chosen)
        elif kind == "attempt":
            body = attempt(chosen, logs)
        elif kind == "phases":
            body = phases(chosen, records)
        elif kind == "meshes":
            body = meshes(chosen, records, md5)
        elif kind == "bench":
            body = bench(chosen)
        else:
            _refuse(f"unknown table kind {kind} for {name}")
        path = out / f"{name}.md"
        path.write_text(header + body)
        written.append(path)
    return written


def main() -> int:
    """Write the snippets and print the records manifest."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--commit", required=True, help="SCALE_COMMIT, short or full")
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--mesh-md5", type=Path, action="append", default=[])
    args = parser.parse_args()
    for path in write_snippets(
        args.records,
        args.out,
        commit=args.commit,
        logs=args.logs,
        mesh_md5=args.mesh_md5,
    ):
        print(path)
    print(manifest(args.records), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
