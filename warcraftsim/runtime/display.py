"""Headless X displays (Xvfb) for game instances."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

_pick_lock = threading.Lock()


def _listening(num: int) -> bool:
    """True once an X server listens on the abstract socket for display `num`."""
    needle = f"@/tmp/.X11-unix/X{num}\n"
    with open("/proc/net/unix") as f:
        return any(line.endswith(needle) for line in f)


class Xvfb:
    """A private Xvfb server on a free display number.

    Under WSLg /tmp/.X11-unix is a read-only mount: Xvfb cannot create its socket file there
    and serves only the abstract socket (which X clients try first), so readiness is checked
    in /proc/net/unix rather than by looking for the socket file.
    """

    def __init__(self, width: int = 1024, height: int = 768, first: int = 100, timeout: float = 10.0):
        if not shutil.which("Xvfb"):
            raise FileNotFoundError("Xvfb not installed (sudo bash scripts/setup_system.sh)")
        self.width, self.height = width, height
        self.proc: subprocess.Popen | None = None
        self.display = ""
        with _pick_lock:
            for num in range(first, first + 500):
                lock = Path(f"/tmp/.X{num}-lock")
                if lock.exists():
                    if self._stale(lock):
                        lock.unlink(missing_ok=True)
                    else:
                        continue
                proc = subprocess.Popen(
                    ["Xvfb", f":{num}", "-screen", "0", f"{width}x{height}x24", "-nolisten", "tcp", "-noreset"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                )
                deadline = time.time() + timeout
                while time.time() < deadline and proc.poll() is None:
                    if _listening(num):
                        self.proc, self.display = proc, f":{num}"
                        return
                    time.sleep(0.05)
                proc.kill()
                proc.wait()
        raise RuntimeError("could not start Xvfb")

    @staticmethod
    def _stale(lock: Path) -> bool:
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
            return False
        except (ValueError, ProcessLookupError):
            return True
        except PermissionError:
            return False

    def screenshot(self, path: str | os.PathLike) -> None:
        """Save the screen as PNG (needs x11-apps + imagemagick)."""
        xwd = subprocess.run(["xwd", "-root", "-silent", "-display", self.display], capture_output=True, check=True)
        subprocess.run(["convert", "xwd:-", str(path)], input=xwd.stdout, check=True)

    def close(self) -> None:
        proc = self.proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.proc = None

    def __enter__(self) -> "Xvfb":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
