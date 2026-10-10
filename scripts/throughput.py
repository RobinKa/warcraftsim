"""A self-play run's throughput over a window: agent steps/s (train.jsonl), the inference server's rounds
(inference.jsonl), the learner's update and wait times, and the CPU that processes outside the run
took meanwhile (other work on the machine, e.g. a headless browser, skews a comparison).

    python3 scripts/throughput.py fgself-12 --since 18:32 [--until 18:47]     # one window
    python3 scripts/throughput.py fgself-12 --watch 600                         # the CPU outside the run, sampled
                                                                                # every 10 s for 600 s (-> outside.jsonl)
    python3 scripts/throughput.py fgself-12 --cpu 30                            # the cores by part of the run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
TICK = os.sysconf("SC_CLK_TCK")


def when(text: str) -> float:
    """HH:MM[:SS] today, or a full 'YYYY-mm-dd HH:MM'."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if t.year == 1900:
            now = datetime.now()
            t = t.replace(year=now.year, month=now.month, day=now.day)
        return t.timestamp()
    raise SystemExit(f"not a time: {text}")


def rows(path: Path, lo: float, hi: float) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if lo < r.get("time", 0) <= hi:
            out.append(r)
    return out


def run_pids(name: str) -> set[int]:
    """The run's processes: its learner and everything below it (actors, games, Wine, the server)."""
    kids: dict[int, list[int]] = {}
    parent, matches = {}, set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = open(f"/proc/{pid}/stat").read()
            args = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        ppid = int(stat[stat.rindex(")") + 2:].split()[1])
        kids.setdefault(ppid, []).append(int(pid))
        parent[int(pid)] = ppid
        if f"selfplay --name {name} " in args + " " and "selfplay_restart" not in args:
            matches.add(int(pid))
    # (its data loader workers carry the same command line: the learner is the one whose parent doesn't)
    out, todo = set(), [p for p in matches if parent.get(p) not in matches]
    while todo:
        p = todo.pop()
        out.add(p)
        todo += kids.get(p, [])
    return out


def wine_pids(name: str) -> set[int]:
    """Wine processes of the run's games (wineserver and services are not children of the learner)."""
    out = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            env = open(f"/proc/{pid}/environ", "rb").read()
        except OSError:
            continue
        if b"/instances/fgsp" in env or b"/instances/fgvid" in env:
            out.add(int(pid))
    return out


def cpu_by_pid() -> dict[int, tuple[str, float]]:
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = open(f"/proc/{pid}/stat").read()
        except OSError:
            continue
        f = stat[stat.rindex(")") + 2:].split()
        out[int(pid)] = (stat[stat.index("(") + 1:stat.rindex(")")], (int(f[11]) + int(f[12])) / TICK)
    return out


def watch(name: str, seconds: float, path: Path) -> None:
    """Every 10 s: cores used by processes outside the run (top three by name)."""
    end = time.time() + seconds
    a, ta = cpu_by_pid(), time.time()
    while time.time() < end:
        time.sleep(10)
        b, tb = cpu_by_pid(), time.time()
        mine = run_pids(name) | wine_pids(name)
        other: dict[str, float] = {}
        for pid, (comm, t) in b.items():
            if pid in mine:
                continue
            d = t - a.get(pid, (comm, t if pid not in a else 0.0))[1] if pid in a else 0.0
            if d > 0:
                other[comm] = other.get(comm, 0.0) + d / (tb - ta)
        total = sum(other.values())
        with path.open("a") as f:
            f.write(json.dumps({"time": tb, "cores": round(total, 2),
                                "top": dict(sorted(other.items(), key=lambda kv: -kv[1])[:3])}) + "\n")
        a, ta = b, tb


def cpu_parts(name: str, seconds: float) -> None:
    """Cores by part of the run over a window (user + kernel), the games' threads split into the busiest
    (the main thread) and the rest; then the machine's busy and idle cores."""
    def threads() -> dict:
        out = {}
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                for tid in os.listdir(f"/proc/{pid}/task"):
                    st = open(f"/proc/{pid}/task/{tid}/stat").read()
                    f = st[st.rindex(")") + 2:].split()
                    out[(int(pid), int(tid))] = (int(f[11]) / TICK, int(f[12]) / TICK)
            except OSError:
                continue
        return out

    def machine() -> list[int]:
        return [int(x) for x in open("/proc/stat").readline().split()[1:]]

    def kind(pid: int, mine: set, server: int | None, learner: int | None) -> str:
        try:
            comm = open(f"/proc/{pid}/comm").read().strip()
            args = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return "gone"
        if "Warcraft III" in args:
            return "video game" if "replay.w3g" in args else "game"
        if comm == "wineserver":
            return "wineserver" if pid in mine else "outside"
        if pid in mine and (args.endswith(".exe") or ".exe " in args or "C:\\windows" in args):
            return "Wine services"
        if pid == learner:
            return "learner"
        if pid == server:
            return "inference server"
        if comm == "pt_data_worker":
            return "learner data workers"
        if comm == "ffmpeg":
            return "video encoder"
        if comm == "Xvfb":
            return "Xvfb"
        if pid in mine and comm.startswith("python"):
            return "actors"
        return "outside"

    run = run_pids(name)
    mine = run | wine_pids(name)
    learner = min((p for p in run if open(f"/proc/{p}/comm").read().strip().startswith("python")), default=None)
    server = None  # the one process below the learner with the GPU open (/dev/dxg on WSL)
    for p in run:
        try:
            if any("dxg" in os.readlink(f"/proc/{p}/fd/{fd}") for fd in os.listdir(f"/proc/{p}/fd")) and p != learner \
                    and open(f"/proc/{p}/comm").read().strip() != "pt_data_worker":
                server = p
        except OSError:
            continue
    a, ma, ta = threads(), machine(), time.time()
    time.sleep(seconds)
    b, mb, tb = threads(), machine(), time.time()
    dt = tb - ta
    parts: dict[str, list[float]] = {}
    game_threads: dict[int, list[tuple]] = {}
    kinds: dict[int, str] = {}
    for (pid, tid), (u, k) in b.items():
        u0, k0 = a.get((pid, tid), (0.0, 0.0))
        du, dk = u - u0, k - k0
        if du + dk <= 0:
            continue
        kd = kinds.setdefault(pid, kind(pid, mine, server, learner))
        p = parts.setdefault(kd, [0.0, 0.0])
        p[0] += du
        p[1] += dk
        if kd == "game":
            game_threads.setdefault(pid, []).append((du + dk, du, dk))
    d = [y - x for x, y in zip(ma, mb)]
    ncpu = os.cpu_count() or 1
    share = lambda i: d[i] / max(1, sum(d)) * ncpu  # noqa: E731
    idle = share(3) + share(4)
    print(f"{dt:.0f} s: {ncpu - idle:.1f} of {ncpu} threads busy (kernel {share(2):.1f}, user {share(0) + share(1):.1f})")
    for kd, (u, k) in sorted(parts.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {kd:22s} {(u + k) / dt:5.2f} cores (kernel {k / dt:4.2f})")
    if game_threads:
        main = [max(ts) for ts in game_threads.values()]
        rest = sum(sum(t[0] for t in ts) for ts in game_threads.values()) - sum(m[0] for m in main)
        print(f"  ({len(main)} games: main threads {sum(m[0] for m in main) / dt:.2f} cores, the others {rest / dt:.2f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name")
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--watch", type=float, help="seconds to sample the CPU outside the run for")
    ap.add_argument("--cpu", type=float, help="seconds to measure the cores by part of the run")
    a = ap.parse_args()
    run = RUNS / a.name
    if a.cpu:
        cpu_parts(a.name, a.cpu)
        return 0
    outside = run / "outside.jsonl"
    if a.watch:
        watch(a.name, a.watch, outside)
        return 0
    lo = when(a.since) if a.since else time.time() - 900
    hi = when(a.until) if a.until else time.time()
    train = rows(run / "train.jsonl", lo, hi)
    infer = rows(run / "inference.jsonl", lo, hi)
    other = rows(outside, lo, hi)
    if not train:
        print("no updates in the window")
        return 1
    steps = train[-1]["agent_steps"] - train[0]["agent_steps"]
    secs = train[-1]["time"] - train[0]["time"]
    print(f"{datetime.fromtimestamp(lo):%H:%M:%S}-{datetime.fromtimestamp(hi):%H:%M:%S}: {len(train)} updates, "
          f"{steps / max(secs, 1e-9):.0f} agent steps/s (mean of the updates' SPS {sum(r['SPS'] for r in train) / len(train):.0f})")
    mean = lambda rs, k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
    print(f"  learner: update {mean(train, 'perf/train'):.2f} s, waiting for data {mean(train, 'perf/rollout'):.2f} s, "
          f"staleness {mean(train, 'staleness'):.2f}")
    if infer:
        print("  inference: " + ", ".join(f"{k} {mean(infer, k):.2f}" for k in
                                         ("round_ms", "wait_ms", "busy", "rows_per_s", "rounds_per_s", "past_calls_per_s")))
    if other:
        worst = max(other, key=lambda r: r["cores"])
        print(f"  outside the run: {mean(other, 'cores'):.2f} cores on average, at most {worst['cores']:.1f} "
              f"({worst['top']}) [{len(other)} samples]")
    else:
        print("  outside the run: not sampled (--watch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
