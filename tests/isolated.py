"""Run one call in a fresh process and measure its peak RSS.

``RUSAGE_SELF`` max RSS never goes down, so inside a pytest process it is the
high-water mark of every test that ran before, not of the solve a test gates.
`isolated` runs the call in a spawned interpreter and returns its value with
the peak RSS of that process: what a standalone run (and Palace's
whole-process figure) measures.

Linux and gVisor carry ``ru_maxrss`` over fork and exec, so the spawned child
would still report the caller's high-water mark. Instead:

- where the kernel keeps ``VmHWM`` (Linux; per address space, fresh after
  exec) the child reports it;
- elsewhere (gVisor, Modal's sandbox) the caller, idle while it waits and
  outside the child's GIL, samples the child's ``VmRSS`` every
  `SAMPLE_INTERVAL`; the child adds its own reading just before it returns.
  A peak held for less than about one interval can be missed;
- without ``/proc`` (macOS, Windows) the child reports `peak_rss_gib`.

The call must be a module-level function of a Python file (a test module
works: the child loads it by path); arguments, return value and exception
travel by pickle, so they must not use classes defined in that file.
"""

import importlib.util
import multiprocessing
import sys
from pathlib import Path

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
else:
    import resource

#: Seconds between the caller's samples of the child's RSS.
SAMPLE_INTERVAL = 0.001
#: Ignore VmHWM (tests run the gVisor path on Linux with it).
SAMPLED_ONLY = False


def isolated(function, *args, **kwargs):
    """``(function(*args, **kwargs), peak RSS in GiB)`` from a fresh process."""
    name = function.__qualname__
    if "<locals>" in name:
        raise ValueError(f"{name} is not a module-level function")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    path = function.__code__.co_filename
    child = context.Process(
        target=_child, args=(send, (path, name), args, kwargs, SAMPLED_ONLY)
    )
    child.start()
    send.close()
    sampled = 0
    try:
        while not receive.poll(SAMPLE_INTERVAL):
            sampled = max(sampled, _status(child.pid).get("VmRSS", 0))
        try:
            ok, value, kib = receive.recv()
        except EOFError:
            child.join()
            raise RuntimeError(
                f"{name} died in its process (exit code {child.exitcode})"
            ) from None
    finally:
        receive.close()
        child.join()
    if not ok:
        raise value
    return value, max(kib, sampled) / 2**20


def _child(send, target, args, kwargs, sampled_only):
    path, name = target
    try:
        spec = importlib.util.spec_from_file_location("_isolated_target", path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        value = getattr(module, name)(*args, **kwargs)
        status = _status("self")
        if not status:  # no /proc
            kib = peak_rss_gib() * 2**20
        elif "VmHWM" in status and not sampled_only:
            kib = status["VmHWM"]
        else:
            kib = status.get("VmRSS", 0)
        reply = (True, value, kib)
    except BaseException as error:  # noqa: BLE001 - re-raised in the caller
        reply = (False, error, 0)
    send.send(reply)
    send.close()


def peak_rss_gib() -> float:
    """This process's peak RSS so far, in GiB."""
    if sys.platform == "win32":
        return _peak_working_set() / 2**30
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss / 2**30 if sys.platform == "darwin" else maxrss / 2**20  # B, KiB


def _peak_working_set() -> int:
    """``PeakWorkingSetSize`` of this process, in bytes (Windows)."""

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE  # a 64-bit pseudo-handle, not an int
    info = ctypes.windll.psapi.GetProcessMemoryInfo
    info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters(cb=ctypes.sizeof(Counters))
    if not info(current(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.PeakWorkingSetSize


def _status(pid) -> dict[str, int]:
    """The kB fields of ``/proc/<pid>/status`` (empty without it)."""
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except OSError:
        return {}
    fields = {}
    for line in lines:
        key, _, value = line.partition(":")
        if value.strip().endswith("kB"):
            fields[key] = int(value.split()[0])
    return fields
