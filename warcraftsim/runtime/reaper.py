"""Child processes (Xvfb servers, games) that must not outlive the process that started them.

Each runs in its own session, so nothing kills it when its owner exits. Owners close them
normally; `reap()` catches the rest, e.g. a video render thread still running when a bridge
worker exits (multiprocessing children skip atexit, so workers call it themselves).
"""

from __future__ import annotations

import atexit
import subprocess
import threading
from typing import Callable

_lock = threading.Lock()
_live: dict[int, tuple[subprocess.Popen, Callable[[], None] | None]] = {}


def track(proc: subprocess.Popen, kill: Callable[[], None] | None = None) -> None:
    """Remember `proc`; `kill` (default proc.kill) stops it and whatever it started."""
    with _lock:
        _live[id(proc)] = (proc, kill)


def untrack(proc: subprocess.Popen | None) -> None:
    with _lock:
        _live.pop(id(proc), None)


def reap() -> int:
    """Stop every tracked process that is still running; returns how many."""
    with _lock:
        items = list(_live.values())
        _live.clear()
    n = 0
    for proc, kill in items:
        if proc.poll() is not None:
            continue
        n += 1
        try:
            (kill or proc.kill)()
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
    return n


atexit.register(reap)
