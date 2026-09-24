import subprocess
import sys

from warcraftsim.runtime import reaper


def test_reap_kills_tracked_processes_and_skips_untracked():
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    kept, lost = subprocess.Popen(sleeper), subprocess.Popen(sleeper)
    calls = []
    reaper.track(kept)
    reaper.track(lost, lambda: (calls.append("kill"), lost.kill()))
    reaper.untrack(kept)
    try:
        assert reaper.reap() == 1
        assert calls == ["kill"]
        assert lost.wait(5) is not None
        assert kept.poll() is None
        assert reaper.reap() == 0
    finally:
        kept.kill()
        lost.kill()
