"""Where a self-play run's machine time goes (the dashboard's System tab): every `every` seconds, the cores each part
of the run used (the games' main threads and their other threads, Wine's wineserver and services, Xvfb, the actors,
the learner's process with its inference loop, the video renderer), the cores used outside the run and the idle ones,
and the GPU's utilization, memory and power: a row each in runs/<run>/system.jsonl. A thread in the learner's process;
a scan reads /proc (~0.1 s of CPU)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

TICK = os.sysconf("SC_CLK_TCK")
PARTS = ("games", "games (other threads)", "Wine", "Xvfb", "actors", "learner", "video renderer", "outside the run",
         "unattributed")  # (the last: processes that ended within the interval, interrupts: busy, but no thread's)


def _tree(root: int) -> tuple[set[int], dict[int, int]]:
    """The process `root` and all below it; every process's parent."""
    kids: dict[int, list[int]] = {}
    parent: dict[int, int] = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            st = open(f"/proc/{p}/stat").read()
        except OSError:
            continue
        ppid = int(st[st.rindex(")") + 2:].split()[1])
        kids.setdefault(ppid, []).append(int(p))
        parent[int(p)] = ppid
    out, todo = set(), [root]
    while todo:
        q = todo.pop()
        out.add(q)
        todo += kids.get(q, [])
    return out, parent


def _threads() -> dict[tuple[int, int], float]:
    """Every thread's CPU seconds (user + kernel), by (process, thread)."""
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                st = open(f"/proc/{p}/task/{t}/stat").read()
                f = st[st.rindex(")") + 2:].split()
                out[(int(p), int(t))] = (int(f[11]) + int(f[12])) / TICK
        except OSError:
            continue
    return out


def _machine() -> list[int]:
    return [int(x) for x in open("/proc/stat").readline().split()[1:]]


def _gpu() -> dict:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        u, m, w = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return {"gpu_util": u, "gpu_mem_gb": round(m / 1024, 2), "gpu_watts": w}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


class Sampler:
    """The run's machine time by part, every `every` seconds, into `path` (the dashboard places the rows on the
    steps' axis by their times)."""

    def __init__(self, path: Path, every: float = 30.0, root: int | None = None):
        self.path, self.every, self.root = path, every, root  # (root: the learner's process; None: this one)
        self.kinds: dict[tuple[int, str], str] = {}  # (pid, its start) -> part (kept while the process lives)
        self.stop = threading.Event()

    def _kind(self, pid: int, mine: set[int], parent: dict[int, int], learner: int) -> str:
        try:
            start = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19]
        except OSError:
            return "outside the run"
        key = (pid, start)
        if key in self.kinds:
            return self.kinds[key]
        try:
            comm = open(f"/proc/{pid}/comm").read().strip()
            args = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
            env = open(f"/proc/{pid}/environ", "rb").read() if pid not in mine else b""
        except OSError:
            return "outside the run"
        ours = pid in mine or b"/instances/fgsp" in env or b"/instances/fgvid" in env  # (Wine's processes: not children)
        if not ours:
            kind = "outside the run"
        elif "Warcraft III" in args:
            kind = "video renderer" if "replay.w3g" in args or "fgvid" in args else "games"
        elif comm == "wineserver" or ".exe" in args or "C:\\windows" in args:
            kind = "Wine"
        elif comm == "Xvfb":
            kind = "Xvfb"
        elif comm == "ffmpeg":
            kind = "video renderer"
        elif pid == learner or comm == "pt_data_worker":
            kind = "learner"
        else:  # (a python process below the learner: an actor, or the renderer, the parent of the replays' games)
            kind = "actors"
        self.kinds[key] = kind
        return kind

    def run(self) -> None:
        learner = self.root or os.getpid()
        a, ma, ta = _threads(), _machine(), time.time()
        while not self.stop.wait(self.every):
            try:
                b, mb, tb = _threads(), _machine(), time.time()
                mine, parent = _tree(learner)
                dt = max(tb - ta, 1e-6)
                cores = {k: 0.0 for k in PARTS}
                main: dict[int, float] = {}
                for (pid, tid), t in b.items():
                    d = t - a.get((pid, tid), t)
                    if d <= 0:
                        continue
                    kind = self._kind(pid, mine, parent, learner)
                    if kind == "games":  # (each game's busiest thread its main one: the simulation)
                        main[pid] = max(main.get(pid, 0.0), d)
                    cores[kind] += d / dt
                games = sum(main.values()) / dt
                cores["games (other threads)"] = max(0.0, cores["games"] - games)
                cores["games"] = games
                d = [y - x for x, y in zip(ma, mb)]
                ncpu = os.cpu_count() or 1
                idle = (d[3] + d[4]) / max(1, sum(d)) * ncpu
                cores["unattributed"] = max(0.0, ncpu - idle - sum(cores.values()))
                row = {"time": round(tb, 1), "ncpu": ncpu, "idle": round(idle, 2),
                       "cores": {k: round(v, 2) for k, v in cores.items()}, **_gpu()}
                with self.path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                self.kinds = {k: v for k, v in self.kinds.items() if os.path.exists(f"/proc/{k[0]}")}
                a, ma, ta = b, mb, tb
            except Exception as e:  # noqa: BLE001 (a measurement only: never the run's end)
                print(f"sysstat: {type(e).__name__}: {e}", flush=True)

    def start(self) -> "Sampler":
        threading.Thread(target=self.run, name="sysstat", daemon=True).start()
        return self


def main() -> int:
    """A run started before the learner sampled itself: `python -m warcraftsim.fullgame.sysstat <run>` samples it
    from outside until its learner is gone."""
    import argparse
    import sys
    from .. import paths
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("run")
    ap.add_argument("--every", type=float, default=30.0)
    a = ap.parse_args()
    pids = [int(p) for p in os.listdir("/proc") if p.isdigit() and _learner_of(int(p), a.run)]
    if not pids:
        print(f"no learner of {a.run}", file=sys.stderr)
        return 1
    s = Sampler(paths.REPO_ROOT / "runs" / a.run / "system.jsonl", a.every, root=min(pids)).start()
    while os.path.exists(f"/proc/{s.root}"):
        time.sleep(5)
    s.stop.set()
    return 0


def _learner_of(pid: int, run: str) -> bool:
    try:
        args = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        return f"selfplay --name {run} " in args + " " and "spawn_main" not in args and "selfplay_" not in args
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
