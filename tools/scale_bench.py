"""The machine benchmark: what the hardware allows, apart from any solver.

    python scale_bench.py node --threads 1 4 28 56 112 --concurrent 28x4 --out NODE.json
    srun --mpi=pmix --ntasks-per-node=28 python scale_bench.py net --out NET.json

``node`` runs on one node with no launcher: MKL's ``cblas_dgemm`` and ``cblas_zgemm``
through `ctypes`, on the `mkl` distribution NGSolve opens, and a C OpenMP STREAM triad,
compiled here from `TRIAD_C` with ``$CC`` (default gcc) into ``--build-dir`` (default
beside ``--out``, since a compute node may have no ``/tmp``). Each thread count runs
in its own process, threads on physical cores (``OMP_PLACES=cores``,
``OMP_PROC_BIND=close``).
``--concurrent 28x4`` runs that many triads side by side from a common start on 4
physical cores each, which is a layout's real bandwidth.

``net`` runs under the launcher: a ping-pong from rank 0 to a rank on its node and to a
rank on another, and ``Allreduce`` of a few sizes across all ranks.

Bandwidth counts 24 bytes per element (STREAM's convention, no write-allocate). Rates
are observations and no rate is a gate. The exit code is 1 only when a thread count did
not hold or a measurement failed; the record is written either way.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "scale-bench/1"
DGEMM_N, ZGEMM_N = 4000, 2000
#: Doubles per bandwidth array: at least 4 times a 2-socket node's last level cache.
ARRAY_LOG2 = 27
BANDWIDTH_REPS = 10
BYTES_PER_ELEMENT = 24
#: Gemm repeats until this much timed wall, after one warm-up.
MIN_GEMM_S = 0.5
CBLAS_ROW_MAJOR, CBLAS_NO_TRANS = 101, 111
#: Ping-pong message sizes in bytes, and Allreduce sizes in doubles.
PING_BYTES = (8, 4096, 1 << 20, 1 << 24)
REDUCE_COUNTS = (1, 64, 4096)
PING_REPS, REDUCE_REPS = 100, 500
TRIAD_FLAGS = ("-O3", "-march=native", "-fopenmp")
TRIAD_C = r"""
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
static int by_value(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
/* triad N REPS [wait]: with "wait", prints ready after first touch, reads a line. */
int main(int argc, char **argv) {
  long n = atol(argv[1]);
  int reps = atoi(argv[2]);
  double *a = malloc(n * sizeof(double)), *b = malloc(n * sizeof(double));
  double *c = malloc(n * sizeof(double)), *t = malloc(reps * sizeof(double));
  int threads = omp_get_max_threads();
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; i++) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }
  if (argc > 3) {
    puts("ready");
    fflush(stdout);
    if (getchar() == EOF) return 2;
  }
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
  for (int r = 0; r < reps; r++) {
    double start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
    t[r] = omp_get_wtime() - start;
  }
  qsort(t, reps, sizeof(double), by_value);
  printf("{\"n\": %ld, \"omp_threads\": %d, \"best_s\": %.9f, \"median_s\": %.9f,"
         " \"check\": %.1f}\n", n, threads, t[0], t[reps / 2], a[n / 2]);
  return 0;
}
"""


def _facts() -> dict[str, Any]:
    """What every record says about where it ran."""
    model = None
    meminfo = None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            meminfo = int(line.split()[1])
            break
    return {
        "schema": SCHEMA,
        "commit": os.environ.get("SCALE_COMMIT"),
        "job": os.environ.get("SLURM_JOB_ID"),
        "qos": os.environ.get("SCALE_QOS"),
        "constraint": os.environ.get("SCALE_CONSTRAINT"),
        "modules": os.environ.get("LOADEDMODULES"),
        "host": socket.gethostname(),
        "cpu": model,
        "mem_total_kb": meminfo,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write(out: Path, record: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")


def _thread_env(threads: int) -> dict[str, str]:
    return os.environ | {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "close",
    }


def _physical_cores() -> list[list[int]]:
    """The affinity mask's CPUs by physical core, one list per core, in CPU order."""
    cores: dict[tuple[str, str], list[int]] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = (topology / "physical_package_id").read_text().strip()
        core = (topology / "core_id").read_text().strip()
        cores.setdefault((package, core), []).append(cpu)
    return list(cores.values())


# ---------------------------------------------------------------- the MKL worker


def _mkl() -> ctypes.CDLL:
    """The `mkl` distribution's libmkl_rt, found as NGSolve's PARDISO wrapper does."""
    distribution = importlib.metadata.distribution("mkl")
    path = next(
        str(distribution.locate_file(f))
        for f in distribution.files or ()
        if "mkl_rt" in str(f)
    )
    lib = ctypes.CDLL(path)
    lib.MKL_Get_Max_Threads.restype = ctypes.c_int
    lib.MKL_Set_Num_Threads_Local.restype = ctypes.c_int
    lib.MKL_Set_Num_Threads_Local.argtypes = [ctypes.c_int]
    p, i, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    lib.cblas_dgemm.argtypes = [i] * 6 + [d, p, i, p, i, d, p, i]
    lib.cblas_zgemm.argtypes = [i] * 6 + [p, p, i, p, i, p, p, i]
    return lib


def _gemm(lib: ctypes.CDLL, kind: str, n: int) -> tuple[Any, float]:
    """A call of ``C = A B``, row-major and no transposes, and its flops."""
    import numpy as np

    rng = np.random.default_rng(0)
    fixed = (CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS, n, n, n)
    if kind == "zgemm":
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        b = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        c = np.zeros((n, n), np.complex128)
        one, zero = np.array([1.0 + 0j]), np.array([0j])
        return (
            lambda: lib.cblas_zgemm(
                *fixed,
                one.ctypes.data,
                a.ctypes.data,
                n,
                b.ctypes.data,
                n,
                zero.ctypes.data,
                c.ctypes.data,
                n,
            ),
            8.0 * n**3,
        )
    a, b = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    c = np.zeros((n, n))
    return (
        lambda: lib.cblas_dgemm(
            *fixed, 1.0, a.ctypes.data, n, b.ctypes.data, n, 0.0, c.ctypes.data, n
        ),
        2.0 * n**3,
    )


def _worker(threads: int, sizes: dict[str, int]) -> dict[str, Any]:
    """One thread count's gemms, in its own process."""
    lib = _mkl()
    lib.MKL_Set_Num_Threads_Local(threads)
    out: dict[str, Any] = {"mkl_threads": lib.MKL_Get_Max_Threads()}
    for kind, n in sizes.items():
        call, flops = _gemm(lib, kind, n)
        call()  # warm-up
        seconds: list[float] = []
        while not seconds or sum(seconds) < MIN_GEMM_S:
            start = time.perf_counter()
            call()
            seconds.append(time.perf_counter() - start)
        out[kind] = {
            "n": n,
            "reps": len(seconds),
            "gflops_best": flops / min(seconds) / 1e9,
            "gflops_median": flops / statistics.median(seconds) / 1e9,
        }
    return out


# ---------------------------------------------------------------- the node


def _build_triad(build_dir: Path) -> dict[str, Any]:
    build_dir.mkdir(parents=True, exist_ok=True)
    source, binary = build_dir / "triad.c", build_dir / "triad"
    source.write_text(TRIAD_C)
    compiler = os.environ.get("CC", "gcc")
    subprocess.run([compiler, *TRIAD_FLAGS, "-o", str(binary), str(source)], check=True)
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True
    )
    return {
        "path": str(binary),
        "compiler": version.stdout.splitlines()[0],
        "flags": TRIAD_FLAGS,
    }


def _gbs(n: int, seconds: float) -> float:
    return BYTES_PER_ELEMENT * n / seconds / 1e9


def _triad(binary: str, threads: int, n: int) -> dict[str, Any]:
    done = subprocess.run(
        [binary, str(n), str(BANDWIDTH_REPS)],
        env=_thread_env(threads),
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(done.stdout)
    return {
        "omp_threads": result["omp_threads"],
        "n": n,
        "gbs_best": _gbs(n, result["best_s"]),
        "gbs_median": _gbs(n, result["median_s"]),
    }


def _row(binary: str, threads: int, n: int, sizes: dict[str, int]) -> dict[str, Any]:
    row: dict[str, Any] = {"threads": threads}
    try:
        worker = subprocess.run(
            [
                sys.executable,
                __file__,
                "worker",
                str(threads),
                json.dumps(sizes),
            ],
            env=_thread_env(threads),
            capture_output=True,
            text=True,
            check=True,
        )
        row |= json.loads(worker.stdout.splitlines()[-1])
        row["triad"] = _triad(binary, threads, n)
        row["held"] = (
            row["mkl_threads"] == threads and row["triad"]["omp_threads"] == threads
        )
    except subprocess.CalledProcessError as error:
        row["error"] = f"{error.cmd[0]}: {str(error.stderr)[-2000:]}"
    return row


def _concurrent(binary: str, layout: str, n: int) -> dict[str, Any]:
    """Triads side by side from a common start, each on its own physical cores."""
    processes, threads = (int(x) for x in layout.split("x"))
    cores = _physical_cores()
    if processes * threads > len(cores):
        return {
            "layout": layout,
            "error": f"{len(cores)} physical cores in the affinity mask",
        }
    each = n // processes
    children = []
    for k in range(processes):
        cpus = {c for core in cores[k * threads : (k + 1) * threads] for c in core}
        children.append(
            subprocess.Popen(
                [binary, str(each), str(BANDWIDTH_REPS), "wait"],
                env=_thread_env(threads),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                preexec_fn=lambda cpus=cpus: os.sched_setaffinity(0, cpus),  # noqa: PLW1509
            )
        )
    for child in children:
        child.stdout.readline()  # ready: its arrays are first-touched on its cores
    for child in children:
        child.stdin.write("go\n")
        child.stdin.flush()
    results = [json.loads(child.communicate()[0]) for child in children]
    return {
        "layout": layout,
        "n_per_process": each,
        "omp_threads": sorted({r["omp_threads"] for r in results}),
        "gbs_sum_of_medians": sum(_gbs(each, r["median_s"]) for r in results),
        "gbs_median_per_process": statistics.median(
            _gbs(each, r["median_s"]) for r in results
        ),
        "held": all(r["omp_threads"] == threads for r in results),
    }


def node(args: argparse.Namespace) -> int:
    """Gemm and triad against thread count, and a layout's concurrent triads."""
    build_dir = args.build_dir or args.out.with_name(args.out.name + ".build")
    triad = _build_triad(build_dir)
    n = 2**args.array_log2
    sizes = {"dgemm": args.dgemm_n, "zgemm": args.zgemm_n}
    record: dict[str, Any] = {
        **_facts(),
        "mode": "node",
        "triad_binary": triad,
        "array_log2": args.array_log2,
        "physical_cores": len(_physical_cores()),
        "rows": [],
    }
    for threads in args.threads:
        row = _row(triad["path"], threads, n, sizes)
        record["rows"].append(row)
        print(json.dumps(row), flush=True)
    if args.concurrent:
        record["concurrent"] = _concurrent(triad["path"], args.concurrent, n)
        print(json.dumps(record["concurrent"]), flush=True)
    parts = [*record["rows"], *([record["concurrent"]] if args.concurrent else [])]
    record["held"] = all(part.get("held") for part in parts)
    _write(args.out, record)
    return 0 if record["held"] else 1


# ---------------------------------------------------------------- the network


def _ping_pong(comm: Any, partner: int, size: int, reps: int) -> dict[str, float]:
    """Half a round trip from rank 0 to ``partner``; every other rank waits it out."""
    import numpy as np

    buffer = np.zeros(size, dtype=np.uint8)
    seconds = []
    for _ in range(reps):
        if comm.rank == 0:
            start = time.perf_counter()
            comm.Send(buffer, dest=partner)
            comm.Recv(buffer, source=partner)
            seconds.append((time.perf_counter() - start) / 2)
        elif comm.rank == partner:
            comm.Recv(buffer, source=0)
            comm.Send(buffer, dest=0)
    comm.Barrier()
    if comm.rank != 0:
        return {}
    median = statistics.median(seconds)
    return {"bytes": size, "median_s": median, "gbs": size / median / 1e9}


def _reduce(comm: Any, count: int, reps: int) -> dict[str, float]:
    """``Allreduce`` of ``count`` doubles, the slowest rank's median, from a barrier."""
    import numpy as np
    from mpi4py import MPI

    send, receive = np.ones(count), np.zeros(count)
    seconds = []
    for _ in range(reps):
        comm.Barrier()
        start = time.perf_counter()
        comm.Allreduce(send, receive, op=MPI.SUM)
        seconds.append(time.perf_counter() - start)
    slowest = comm.allreduce(statistics.median(seconds), op=MPI.MAX)
    return {"doubles": count, "median_s_slowest_rank": slowest}


def net(args: argparse.Namespace) -> int:
    """Ping-pong within and between nodes, and ``Allreduce`` across all ranks."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    hosts = comm.allgather(socket.gethostname())
    same = next((r for r in range(1, len(hosts)) if hosts[r] == hosts[0]), None)
    other = next((r for r in range(1, len(hosts)) if hosts[r] != hosts[0]), None)
    results: dict[str, Any] = {"ping_same_node": [], "ping_other_node": []}
    for key, partner in (("ping_same_node", same), ("ping_other_node", other)):
        if partner is None:
            continue
        for size in args.ping_bytes:
            results[key].append(_ping_pong(comm, partner, size, args.ping_reps))
    results["allreduce"] = [
        _reduce(comm, count, args.reduce_reps) for count in REDUCE_COUNTS
    ]
    if comm.rank == 0:
        record = {
            **_facts(),
            "mode": "net",
            "mpi_library": MPI.Get_library_version().splitlines()[0],
            "ranks": comm.size,
            "nodes": sorted(set(hosts)),
            "node_count": len(set(hosts)),
            **results,
        }
        _write(args.out, record)
        print(json.dumps(results), flush=True)
    return 0


def main() -> int:
    """Parse the mode and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)

    one = modes.add_parser("node")
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--threads", type=int, nargs="+", required=True)
    one.add_argument("--concurrent", default=None, help="e.g. 28x4")
    one.add_argument("--array-log2", type=int, default=ARRAY_LOG2)
    one.add_argument("--dgemm-n", type=int, default=DGEMM_N)
    one.add_argument("--zgemm-n", type=int, default=ZGEMM_N)
    one.add_argument("--build-dir", type=Path, default=None)
    one.set_defaults(call=node)

    between = modes.add_parser("net")
    between.add_argument("--out", type=Path, required=True)
    between.add_argument("--ping-bytes", type=int, nargs="+", default=list(PING_BYTES))
    between.add_argument("--ping-reps", type=int, default=PING_REPS)
    between.add_argument("--reduce-reps", type=int, default=REDUCE_REPS)
    between.set_defaults(call=net)

    worker = modes.add_parser("worker")
    worker.add_argument("threads", type=int)
    worker.add_argument("sizes", type=json.loads)
    worker.set_defaults(
        call=lambda a: print(json.dumps(_worker(a.threads, a.sizes))) or 0
    )

    args = parser.parse_args()
    return args.call(args)


if __name__ == "__main__":
    sys.exit(main())
