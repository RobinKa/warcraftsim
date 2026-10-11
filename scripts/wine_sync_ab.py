"""Wine's synchronization modes against each other: games of each variant at the same time (built-in AI
against built-in AI on duelfast, drawing off, each waiting --wait-ms for every step's orders like the
run's games), CPU a step of the game and of its wineserver from /proc, and the wall time a step.

    python3 scripts/wine_sync_ab.py --wine ~/wc3/ge/GE-Proton10-34/files/bin \\
        --variant fsync:WINEFSYNC=1,WINENTSYNC=0 --variant ntsync:WINEFSYNC=1,WINENTSYNC=1 [-n 4] [--steps 300]

Each variant's games are launched with its environment variables (a prefix's wineserver starts with its
first game's); every game warms up, then all step --steps steps from one start.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warcraftsim.runtime import wine  # noqa: E402

TICK = os.sysconf("SC_CLK_TCK")


def prefix_cpu(prefix: Path) -> dict[str, float]:
    """CPU seconds so far of the processes running in this prefix: wineserver's, everything else's."""
    want = f"WINEPREFIX={prefix}".encode()
    out = {"wineserver": 0.0, "game": 0.0}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if want not in (d / "environ").read_bytes().split(b"\0"):
                continue
            stat = (d / "stat").read_text()
        except OSError:
            continue
        comm = stat[stat.index("(") + 1:stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2:].split()
        out["wineserver" if comm == "wineserver" else "game"] += (int(fields[11]) + int(fields[12])) / TICK
    return out


def wineserver_pid(prefix: Path) -> int | None:
    want = f"WINEPREFIX={prefix}".encode()
    for d in Path("/proc").iterdir():
        try:
            if d.name.isdigit() and (d / "comm").read_text().strip() == "wineserver" \
                    and want in (d / "environ").read_bytes().split(b"\0"):
                return int(d.name)
        except OSError:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wine", required=True, help="a Proton build's files/bin")
    ap.add_argument("--variant", action="append", required=True, help="name:VAR=value,VAR=value")
    ap.add_argument("-n", type=int, default=4, help="games per variant")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--wait-ms", type=float, default=35.0)
    a = ap.parse_args()
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    from warcraftsim.runtime.instance import BuiltinAI, GameInstance, GameSetup
    races = ["human", "orc", "undead", "nightelf"]
    variants = []
    for spec in a.variant:
        name, _, kv = spec.partition(":")
        variants.append((name, dict(x.split("=", 1) for x in kv.split(",") if x)))
    games = []  # (variant, instance)
    base = dict(os.environ)
    for k in range(a.n):
        for name, env in variants:  # (launched alternately: the same load for both)
            setup = GameSetup(map="duelfast", slots=[BuiltinAI(races[k % 4], "insane", handicap=50),
                                                      BuiltinAI(races[(k + 1) % 4], "insane", handicap=50)],
                              step_seconds=0.5, max_game_seconds=1200, victory="decisive", window=(320, 240),
                              draw=False, native_obs=True, d3d_thread=False, render_threads=0, nice=10)
            os.environ.clear()
            os.environ.update(base)
            os.environ.update(env)
            g = GameInstance(setup, name=f"syncab_{name}_{k}", timeout=180)
            g.start()
            games.append((name, g))
    os.environ.clear()
    os.environ.update(base)
    for name, g in games:
        log = (g.inst_dir / "wine.log").read_text(errors="replace") if (g.inst_dir / "wine.log").exists() else ""
        sync = "ntsync" if "NTSync up and running" in log else "fsync" if "fsync: up and running" in log else "server"
        print(f"{g.name}: {sync}", flush=True)
    barrier = threading.Barrier(len(games))
    res: dict[str, list] = {name: [] for name, _ in variants}

    def play(name: str, g) -> None:
        for _ in range(a.warmup):
            g.step([])
            time.sleep(a.wait_ms / 1000)
        barrier.wait()
        c0, t0, n = prefix_cpu(g.prefix), time.time(), 0
        for _ in range(a.steps):
            obs = g.step([])
            n += 1
            time.sleep(a.wait_ms / 1000)
            if obs.game_over:
                break
        c1, t1 = prefix_cpu(g.prefix), time.time()
        res[name].append((n, c1["game"] - c0["game"], c1["wineserver"] - c0["wineserver"], t1 - t0))

    threads = [threading.Thread(target=play, args=(name, g)) for name, g in games]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for _, g in games:
        g.close()
    print("| variant | the game | wineserver | CPU a step | wall a step |")
    print("|---|---|---|---|---|")
    for name, _ in variants:
        steps = sum(r[0] for r in res[name])
        game, server = (1000 * sum(r[i] for r in res[name]) / steps for i in (1, 2))
        wall = 1000 * sum(r[3] / r[0] for r in res[name]) / len(res[name])
        print(f"| {name} | {game:.2f} ms | {server:.2f} ms | {game + server:.2f} ms | {wall:.1f} ms |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
