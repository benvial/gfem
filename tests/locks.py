"""A lock that reports a waiter, to free a handle while a call waits for it.

Usage::

    from locks import Gate

    gate = handle._lock = Gate()
    with gate:
        worker.start()  # a call that takes the handle's lock
        assert gate.waiting.wait(timeout=30)
        ...  # the worker waits for the lock
"""

import threading


class Gate:
    """A reentrant lock that reports a thread waiting for it."""

    def __init__(self):
        self.lock = threading.RLock()
        self.waiting = threading.Event()

    def __enter__(self):
        if not self.lock.acquire(blocking=False):
            self.waiting.set()
            self.lock.acquire()
        return self

    def __exit__(self, *exc):
        self.lock.release()
