"""Run one call in a fresh process and measure its peak RSS.

``RUSAGE_SELF`` max RSS never goes down, so inside a pytest process it is the
high-water mark of every test that ran before, not of the solve a test gates.
`isolated` runs the call in a spawned interpreter and returns its value with
the peak RSS of that process: what a standalone run (and Palace's
whole-process figure) measures.

Linux carries ``ru_maxrss`` over fork and exec, so a spawned child would still
report the caller's high-water mark. The child reads its own ``VmHWM``
(``/proc/self/status``, per address space, fresh after exec) and, where the
kernel has no ``VmHWM`` (gVisor), the largest ``VmRSS`` a sampling thread
saw; ``ru_maxrss`` only without ``/proc`` (macOS).

The call must be a module-level function of a Python file (a test module
works: the child loads it by path); arguments and return value travel by
pickle, so the return value must not use classes defined in that file.
"""

import importlib.util
import multiprocessing
import resource
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

#: Seconds between RSS samples where the kernel keeps no VmHWM.
SAMPLE_INTERVAL = 0.01


def isolated(function, *args, **kwargs):
    """``(function(*args, **kwargs), peak RSS in GiB)`` from a fresh process."""
    name = function.__qualname__
    if "<locals>" in name:
        raise ValueError(f"{name} is not a module-level function")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        job = pool.submit(_call, function.__code__.co_filename, name, args, kwargs)
        return job.result()


def _call(path, name, args, kwargs):
    spec = importlib.util.spec_from_file_location("_isolated_target", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sampled = [0]
    done = threading.Event()

    def sample():
        while True:
            sampled[0] = max(sampled[0], _status().get("VmRSS", 0))
            if done.wait(SAMPLE_INTERVAL):
                return

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        value = getattr(module, name)(*args, **kwargs)
    finally:
        done.set()
        sampler.join()
    status = _status()
    if not status:  # no /proc
        kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":  # bytes there
            kib /= 1024
    else:
        kib = max(status.get("VmHWM", 0), status.get("VmRSS", 0), sampled[0])
    return value, kib / 2**20


def _status() -> dict[str, int]:
    """The kB fields of ``/proc/self/status`` (empty without ``/proc``)."""
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
    except OSError:
        return {}
    fields = {}
    for line in lines:
        key, _, value = line.partition(":")
        if value.strip().endswith("kB"):
            fields[key] = int(value.split()[0])
    return fields
