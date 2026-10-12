"""Equivalent-speedup experiments on the game's own code (shim W3SIM_CUTS and harness changes): does a
variant leave the simulation (and the observation) exactly as it was, and what does it save?

    # a live game, its replay and every step's full observation digest
    python scripts/cut_ab.py record --out DIR [--kind ai|agents] [--map duelfast] [--races human,orc]
        [--steps 1200] [--pairs 1] [--games 1] [--shim build/shim]
    # the replay played back once per variant, all at the same time: every unit's and player's state at
    # every step compared with the live game (and the first variant), and the main thread's user
    # instructions, cycles and CPU time per step (perf stat -t) over steps [--warmup, --steps)
    python scripts/cut_ab.py play --replay DIR/game.w3g --variant base:W3SIM_CUTS=0 --variant cuts:W3SIM_CUTS=13 \
        [--variant name:VAR=v,VAR=v@shim_dir] [--steps 1200] [--warmup 100] [--repeat 1]

Instance names start with "cutprobe" (never "fgsp"/"fgvid"); games are niced (+10), drawing off.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("WAYLAND_DISPLAY", None)  # (nothing of ours on the user's desktop: games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"  # (no such display: anything Wine runs without its Xvfb fails instead of showing)

import warcraftsim.runtime.instance as instance  # noqa: E402
from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup  # noqa: E402

TICK = os.sysconf("SC_CLK_TCK")
_launch_lock = threading.Lock()


def unit_row(u) -> tuple:
    return tuple(getattr(u, f) for f in u.__slots__) if hasattr(u, "__slots__") else dataclasses.astuple(u)


def digest(obs) -> dict:
    """Everything the observation says about the game at this step."""
    units = {u.id: unit_row(u) for u in obs.units}
    players = {p: tuple(dataclasses.astuple(s)) for p, s in sorted(obs.players.items())}
    events = [(e.kind, e.a, e.b, e.c) for e in obs.events]
    issued = [dataclasses.astuple(i) for i in obs.issued]
    return {"ms": obs.game_ms, "over": obs.game_over, "units": units, "players": players, "events": events,
            "removed": sorted(obs.removed), "issued": issued, "cres": list(obs.command_results)}


def step_hash(d: dict) -> str:
    return hashlib.sha1(repr((d["ms"], d["over"], sorted(d["units"].items()), d["players"], d["events"], d["removed"],
                              d["issued"], d["cres"])).encode()).hexdigest()[:16]


def compare(a: list[dict], b: list[dict]) -> str:
    n = min(len(a), len(b))
    for t in range(n):
        x, y = a[t], b[t]
        if step_hash(x) != step_hash(y):
            ux, uy = x["units"], y["units"]
            diff = sorted(set(ux) ^ set(uy))[:3] or [k for k in ux if ux[k] != uy.get(k)][:3]
            what = [(k, ux.get(k), uy.get(k)) for k in diff]
            other = [k for k in ("ms", "over", "players", "events", "removed", "issued", "cres") if x[k] != y[k]]
            return f"DIFFER at step {t} ({len(ux)} vs {len(uy)} units; fields {other}; e.g. {what})"
    return f"identical over {n} steps" + ("" if len(a) == len(b) else f" (lengths {len(a)} vs {len(b)})")


# ---- processes and counters ----------------------------------------------------------------------------

def game_pid(prefix: Path) -> int | None:
    want = f"WINEPREFIX={prefix}".encode()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if not (d / "comm").read_text().startswith("Warcraft III"):
                continue
            if want in (d / "environ").read_bytes().split(b"\0"):
                return int(d.name)
        except OSError:
            continue
    return None


def thread_cpu(pid: int) -> dict[int, float]:
    out = {}
    for t in Path(f"/proc/{pid}/task").iterdir():
        try:
            stat = (t / "stat").read_text()
        except OSError:
            continue
        f = stat[stat.rindex(")") + 2:].split()
        out[int(t.name)] = (int(f[11]) + int(f[12])) / TICK
    return out


def proc_cpu(pid: int) -> float:
    stat = Path(f"/proc/{pid}/stat").read_text()
    f = stat[stat.rindex(")") + 2:].split()
    return (int(f[11]) + int(f[12])) / TICK


class Perf:
    """perf stat on one thread: user instructions and cycles, task-clock (ms)."""

    def __init__(self, tid: int, out: Path):
        self.out = out
        self.p = subprocess.Popen(["perf", "stat", "-t", str(tid), "-e", "instructions:u,cycles:u,task-clock",
                                   "-x", ",", "-o", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.3)

    def stop(self) -> dict[str, float]:
        self.p.send_signal(signal.SIGINT)
        self.p.wait(10)
        res = {}
        for line in self.out.read_text().splitlines():
            parts = line.split(",")
            if len(parts) >= 3 and parts[0] and parts[0][0].isdigit():
                name = parts[2].split(":")[0]
                try:
                    res[name] = float(parts[0])
                except ValueError:
                    pass
        return res


# ---- a simple agent: workers harvest, halls train workers, units attack-move now and then -----------------

class Scripted:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def commands(self, g: GameInstance, obs, players: list[int]) -> list:
        from warcraftsim.protocol import ImmediateOrder, PointOrder, TargetOrder, UnitFlags, fourcc
        out = []
        mines = [u for u in obs.units if u.resource > 0]
        smart, attack, move = 851971, 851983, 851986  # (the game's order ids)
        for p in players:
            mine_units = [u for u in obs.units if u.owner == p and not (u.flags & UnitFlags.DEAD)]
            enemy_start = next(((s.start_x, s.start_y) for q, s in obs.players.items() if q != p), (0, 0))
            for u in mine_units:
                if u.flags & UnitFlags.WORKER and u.order == 0 and mines:
                    m = min(mines, key=lambda m: (m.x - u.x) ** 2 + (m.y - u.y) ** 2)
                    out.append(TargetOrder(u.id, smart, m.id))
                elif u.flags & UnitFlags.STRUCTURE and u.order == 0 and self.rng.random() < 0.1:
                    t = {1: "hpea", 2: "opeo", 3: "uaco", 4: "ewsp"}.get(int(obs.players[p].race), "hpea")
                    if u.type_id in (fourcc("htow"), fourcc("ogre"), fourcc("unpl"), fourcc("etol")):
                        out.append(ImmediateOrder(u.id, fourcc(t)))
                elif not (u.flags & (UnitFlags.WORKER | UnitFlags.STRUCTURE)) and self.rng.random() < 0.05:
                    if self.rng.random() < 0.5:
                        out.append(PointOrder(u.id, attack, enemy_start[0], enemy_start[1]))
                    else:
                        out.append(PointOrder(u.id, move, u.x + self.rng.uniform(-800, 800), u.y + self.rng.uniform(-800, 800)))
        return out


# ---- record ---------------------------------------------------------------------------------------------

def make_setup(spec: dict) -> GameSetup:
    r0, r1 = spec["races"]
    if spec["kind"] == "ai":
        slots = [BuiltinAI(r0, spec.get("difficulty", "insane")), BuiltinAI(r1, spec.get("difficulty", "insane"))]
    elif spec["kind"] == "mixed":
        slots = [Agent(r0), BuiltinAI(r1, spec.get("difficulty", "insane"))]
    else:
        slots = [Agent(r0), Agent(r1)]
    return GameSetup(map=spec["map"], slots=slots, step_seconds=spec.get("step_seconds", 0.5),
                     max_game_seconds=spec.get("max_game_seconds", 1200), victory="decisive", window=(320, 240),
                     draw=False, native_obs=True, d3d_thread=False, render_threads=0, nice=10, wait_floor_ms=5,
                     pairs=spec.get("pairs", 1), record_ai_orders=spec.get("record_ai_orders", False), warm_spare=False)


def launch(setup: GameSetup, name: str, env: dict[str, str], shim: Path | None, replay: Path | None):
    with _launch_lock:
        saved_env, saved_shim = dict(os.environ), instance.SHIM_DIR
        try:
            os.environ.update(env)
            if shim:
                instance.SHIM_DIR = shim.resolve()
            g = GameInstance(setup, name=name, timeout=300)
            obs = g.play_replay(replay) if replay else g.start()
        finally:
            os.environ.clear()
            os.environ.update(saved_env)
            instance.SHIM_DIR = saved_shim
    return g, obs


def cmd_record(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = {"kind": a.kind, "map": a.map, "races": a.races.split(","), "pairs": a.pairs, "steps": a.steps,
            "games": a.games, "record_ai_orders": a.record_ai_orders, "difficulty": a.difficulty}
    setup = make_setup(spec)
    env = dict(x.split("=", 1) for x in a.env.split(",") if x) if a.env else {}
    g, obs = launch(setup, f"cutprobe_rec_{out.name}", env, Path(a.shim) if a.shim else None, None)
    agents = list(setup.agent_players)
    pol = Scripted(a.seed)
    digests = []
    try:
        for game in range(a.games):
            n = 0
            while True:
                digests.append(digest(obs))
                n += 1
                if obs.game_over or n >= a.steps:
                    break
                cmds = pol.commands(g, obs, [p for p in obs.players if obs.players[p].is_agent]) if agents else []
                obs = g.step(cmds)
            print(f"game {game}: {n} steps, {len(obs.units)} units, over={obs.game_over}", flush=True)
            if game + 1 < a.games:
                obs = g.restart()
        replay = g.save_replay(out / "game.w3g")
    finally:
        g.close()
    (out / "live.json").write_text(json.dumps({"spec": spec, "digests": [step_hash(d) for d in digests]}))
    import pickle
    with open(out / "live.pkl", "wb") as f:
        pickle.dump(digests, f)
    print(f"recorded {len(digests)} steps -> {replay}", flush=True)
    return 0


# ---- play -----------------------------------------------------------------------------------------------

def setup_for(replay: Path) -> tuple[GameSetup, list[dict] | None]:
    live = replay.parent / "live.json"
    if live.exists():
        spec = json.loads(live.read_text())["spec"]
        import pickle
        pkl = replay.parent / "live.pkl"
        return make_setup(spec), (pickle.load(open(pkl, "rb")) if pkl.exists() else None)
    trace = replay.with_suffix(".trace.json")  # a self-play run's replay: its sides
    races = ["human", "orc"]
    kinds = ["agent", "agent"]
    if trace.exists():
        t = json.loads(trace.read_text())
        for s in t["sides"]:
            races[s["player"]] = s["name"].rsplit("(", 1)[1].rstrip(")")
            kinds[s["player"]] = s["kind"]
    slots = [Agent(r) if k == "agent" else BuiltinAI(r, "insane") for r, k in zip(races, kinds)]
    setup = GameSetup(map="duelfast", slots=slots, step_seconds=0.5, max_game_seconds=1200, victory="decisive",
                      window=(320, 240), draw=False, native_obs=True, d3d_thread=False, render_threads=0, nice=10,
                      wait_floor_ms=5, pairs=5, record_ai_orders=True, warm_spare=False)
    return setup, None


def cmd_play(a) -> int:
    replay = Path(a.replay).resolve()
    setup, live = setup_for(replay)
    if live:  # (a recording of ours: exactly its steps, past each game's end when it had several)
        a.steps = min(a.steps, len(live))
        a.min_steps = max(a.min_steps, len(live) - 1)
    variants = []
    for spec in a.variant:
        name, _, rest = spec.partition(":")
        rest, _, shim = rest.partition("@")
        variants.append((name, dict(x.split("=", 1) for x in rest.split(",") if x), Path(shim) if shim else None))
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    runs = []  # (variant, repeat, game, obs)
    for r in range(a.repeat):
        for vi, (name, env, shim) in enumerate(variants):
            g, obs = launch(setup, f"cutprobe{a.tag}_{vi}_{r}", env, shim, replay)
            runs.append({"variant": name, "rep": r, "g": g, "obs": obs, "digests": [], "perf": None})
    barrier = threading.Barrier(len(runs))

    def play(run: dict) -> None:
        g, obs = run["g"], run["obs"]
        perf = None
        try:
            pid = game_pid(g.prefix)
            c0 = thread_cpu(pid) if pid else {}
            for t in range(a.steps):
                run["digests"].append(digest(obs))
                if t == a.warmup:
                    if t == 0:
                        g.step([])  # (to find the main thread)
                    c1 = thread_cpu(pid)
                    tid = max(c1, key=lambda k: c1[k] - c0.get(k, 0.0))
                    run["tid"] = tid
                    try:
                        barrier.wait(timeout=600)
                    except threading.BrokenBarrierError:
                        pass
                    perf = Perf(tid, tmp / f"perf_{run['variant']}_{run['rep']}.txt")
                    run["cpu0"], run["pcpu0"], run["t0"], run["s0"] = thread_cpu(pid)[tid], proc_cpu(pid), time.time(), t
                if a.window and perf and t > a.warmup and (t - a.warmup) % a.window == 0:
                    w = perf.stop()
                    run.setdefault("windows", []).append(w.get("instructions", 0) / a.window / 1e6)
                    run.setdefault("perf_parts", []).append(w)
                    perf = Perf(run["tid"], tmp / f"perf_{run['variant']}_{run['rep']}.txt")
                if (obs.game_over and t >= a.min_steps) or t == a.steps - 1:
                    break
                obs = g.step([])
                run["obs"] = obs
            if perf:
                run["perf"] = perf.stop()
                for w in run.get("perf_parts", []):  # (the windows' counts add up)
                    for k, v in w.items():
                        run["perf"][k] = run["perf"].get(k, 0) + v
                run["steps_measured"] = len(run["digests"]) - run["s0"]
                run["cpu"] = thread_cpu(pid)[run["tid"]] - run["cpu0"]
                run["pcpu"] = proc_cpu(pid) - run["pcpu0"]
                run["wall"] = time.time() - run["t0"]
        except Exception as e:  # noqa: BLE001
            run["error"] = f"{type(e).__name__}: {e}"
            try:
                barrier.abort()
            except Exception:
                pass
        finally:
            try:
                g.close()
            except Exception:
                pass

    threads = [threading.Thread(target=play, args=(r,)) for r in runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = []
    print(f"| variant | rep | steps | M instr/step | M cycles/step | task-clock ms/step | main CPU ms/step | process CPU ms/step | parity vs live | vs {runs[0]['variant']} |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        p = run.get("perf") or {}
        n = max(run.get("steps_measured", 0), 1)
        live_cmp = compare(live, run["digests"]) if live else "-"
        base_cmp = compare(runs[0]["digests"], run["digests"]) if run is not runs[0] else "-"
        row = {"variant": run["variant"], "rep": run["rep"], "steps": n, "error": run.get("error"),
               "instr": p.get("instructions", 0) / n / 1e6, "cycles": p.get("cycles", 0) / n / 1e6,
               "task_ms": p.get("task-clock", 0) / n, "cpu_ms": 1000 * run.get("cpu", 0) / n,
               "pcpu_ms": 1000 * run.get("pcpu", 0) / n, "live": live_cmp, "base": base_cmp,
               "hashes": [step_hash(d) for d in run["digests"]]}
        rows.append(row)
        row["windows"] = run.get("windows", [])
        print(f"| {row['variant']} | {row['rep']} | {n} | {row['instr']:.2f} | {row['cycles']:.2f} | {row['task_ms']:.2f} | "
              f"{row['cpu_ms']:.2f} | {row['pcpu_ms']:.2f} | {live_cmp} | {base_cmp} |" + (f" ERROR {row['error']}" if row["error"] else ""),
              flush=True)
    if a.window:
        for row in rows:
            print(f"{row['variant']} M instr/step by {a.window}-step window: " + " ".join(f"{w:.1f}" for w in row["windows"]))
    if a.json:
        Path(a.json).write_text(json.dumps({"replay": str(replay), "rows": rows}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--kind", default="ai", choices=["ai", "agents", "mixed"])
    r.add_argument("--map", default="duelfast")
    r.add_argument("--races", default="human,orc")
    r.add_argument("--difficulty", default="insane")
    r.add_argument("--steps", type=int, default=1200)
    r.add_argument("--games", type=int, default=1)
    r.add_argument("--pairs", type=int, default=1)
    r.add_argument("--record-ai-orders", action="store_true")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--env", default="")
    r.add_argument("--shim")
    p = sub.add_parser("play")
    p.add_argument("--replay", required=True)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--min-steps", type=int, default=0)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--window", type=int, default=0, help="also the instructions per step in windows of this many steps")
    p.add_argument("--json")
    p.add_argument("--tmp", default=str(Path.home() / "tools/w3analysis/perf"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    return cmd_record(a) if a.cmd == "record" else cmd_play(a)


if __name__ == "__main__":
    raise SystemExit(main())
