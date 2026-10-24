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
# perf counters need kernel.perf_event_paranoid <= 2 (Ubuntu's default 4 refuses them): CPU time only then
try:
    NO_PERF = int(Path("/proc/sys/kernel/perf_event_paranoid").read_text()) > 2
except (OSError, ValueError):
    NO_PERF = True
_launch_lock = threading.Lock()
NICE = int(os.environ.get("CUT_AB_NICE", "15"))


def unit_row(u) -> tuple:
    return tuple(getattr(u, f) for f in u.__slots__) if hasattr(u, "__slots__") else dataclasses.astuple(u)


def digest(obs) -> dict:
    """Everything the observation says about the game at this step."""
    units = {u.id: unit_row(u) for u in obs.units}
    players = {p: tuple(dataclasses.astuple(s)) for p, s in sorted(obs.players.items())}
    events = [(e.kind, e.a, e.b, e.c) for e in obs.events]
    issued = [dataclasses.astuple(i) for i in obs.issued]
    return {"ms": obs.game_ms, "over": obs.game_over, "units": units, "players": players, "events": events,
            "removed": sorted(obs.removed), "issued": issued, "cres": list(obs.command_results),
            "ground": [tuple(x) for x in getattr(obs, "ground", [])], "damaged": obs.damaged_records}


def step_hash(d: dict) -> str:
    return hashlib.sha1(repr((d["ms"], d["over"], sorted(d["units"].items()), d["players"], d["events"], d["removed"],
                              d["issued"], d["cres"]) + ((d["ground"],) if d.get("ground") else ())
                             ).encode()).hexdigest()[:16]


def raw_tap(g) -> list:
    """The raw observation payloads' hashes (byte for byte, as the shim sent them), in order."""
    out: list = []
    nat = getattr(g, "_native", None)
    if nat is None:
        return out
    parse = nat.parse

    def tapped(payload, *a, **k):
        out.append(hashlib.sha1(bytes(payload)).hexdigest()[:16])
        return parse(payload, *a, **k)
    nat.parse = tapped
    return out


def compare_raw(a: list, b: list) -> str:
    n = min(len(a), len(b))
    bad = [t for t in range(n) if a[t] != b[t]]
    return f"raw identical over {n}" if not bad else f"raw DIFFER at {len(bad)} of {n} (first {bad[0]})"


def compare(a: list[dict], b: list[dict]) -> str:
    n = min(len(a), len(b))
    for t in range(n):
        x, y = a[t], b[t]
        if step_hash(x) != step_hash(y):
            ux, uy = x["units"], y["units"]
            diff = sorted(set(ux) ^ set(uy))[:3] or [k for k in ux if ux[k] != uy.get(k)][:3]
            what = [(k, ux.get(k), uy.get(k)) for k in diff]
            other = [k for k in ("ms", "over", "players", "events", "removed", "issued", "cres", "ground", "damaged") if x.get(k) != y.get(k)]
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


def thread_ns(pid: int, tid: int) -> int:
    """A thread's CPU time in ns (schedstat: exact, unlike the 10 ms ticks of stat)."""
    try:
        return int(Path(f"/proc/{pid}/task/{tid}/schedstat").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return 0


def prefix_pids(prefix: Path) -> dict[str, list[int]]:
    """The processes of a game's Wine prefix: its wineserver and the rest (the game, services, explorer...)."""
    want = f"WINEPREFIX={prefix}".encode()
    out: dict[str, list[int]] = {"wineserver": [], "other": []}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if want not in (d / "environ").read_bytes().split(b"\0"):
                continue
            comm = (d / "comm").read_text().strip()
        except OSError:
            continue
        if comm.startswith("Warcraft III"):
            continue
        out["wineserver" if comm == "wineserver" else "other"].append(int(d.name))
    return out


def pids_cpu(pids: list[int]) -> float:
    t = 0.0
    for p in pids:
        try:
            t += proc_cpu(p)
        except OSError:
            pass
    return t


class Perf:
    """perf stat on one thread: user instructions and cycles, task-clock (ms)."""

    def __init__(self, tid: int, out: Path):
        self.out = out
        if NO_PERF:
            self.p = None
            return
        self.p = subprocess.Popen(["perf", "stat", "-t", str(tid), "-e",
                                   os.environ.get("CUTAB_PERF_EVENTS", "instructions:u,cycles:u,task-clock"),
                                   "-x", ",", "-o", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.3)

    def stop(self) -> dict[str, float]:
        if self.p is None:
            return {}
        self.p.send_signal(signal.SIGINT)
        self.p.wait(10)
        res = {}
        if not self.out.exists():
            return res
        for line in self.out.read_text().splitlines():
            parts = line.split(",")
            if len(parts) >= 3 and parts[0] and parts[0][0].isdigit():
                name = parts[2].split(":")[0].split("/")[0]
                try:
                    res[name] = float(parts[0])
                except ValueError:
                    pass
        return res


class PerfRecord:
    """perf record on one thread (CUTAB_PERF_RECORD="-e cycles:u -c 200000 ..." in the variant's env)."""

    def __init__(self, tid: int, out: Path, args: str):
        self.p = subprocess.Popen(["perf", "record", "-t", str(tid), "-o", str(out)] + args.split(),
                                  stdout=subprocess.DEVNULL, stderr=open(str(out) + ".err", "w"))
        time.sleep(0.3)

    def stop(self) -> None:
        self.p.send_signal(signal.SIGINT)
        try:
            self.p.wait(60)
        except subprocess.TimeoutExpired:
            self.p.kill()


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
                     draw=False, native_obs=True, d3d_thread=False, render_threads=0, nice=NICE, wait_floor_ms=5,
                     pairs=spec.get("pairs", 1), record_ai_orders=spec.get("record_ai_orders", False), warm_spare=False,
                     seed=spec.get("seed", 0))


def launch(setup: GameSetup, name: str, env: dict[str, str], shim: Path | None, replay: Path | None):
    """env CUTAB_HARNESS=current (not passed to the game): a replay plays back on a map built with this checkout's
    harness instead of the map it was recorded on (a harness change's parity)."""
    env = dict(env)
    harness = env.pop("CUTAB_HARNESS", "")
    current = harness == "current"
    import warcraftsim.data.mapbuild as mapbuild
    with _launch_lock:
        saved_env, saved_shim, saved_rec = dict(os.environ), instance.SHIM_DIR, instance.recorded_map
        saved_src = mapbuild.harness_source
        try:
            os.environ.update(env)
            if shim:
                instance.SHIM_DIR = shim.resolve()
            if current:
                instance.recorded_map = lambda *a, **k: None
            elif harness:  # (live games: this harness source for the variant's map, e.g. git's)
                text = Path(harness).read_text()
                mapbuild.harness_source = lambda: text
            g = GameInstance(setup, name=name, timeout=300)
            obs = g.play_replay(replay) if replay else g.start()
        finally:
            os.environ.clear()
            os.environ.update(saved_env)
            instance.SHIM_DIR = saved_shim
            instance.recorded_map = saved_rec
            mapbuild.harness_source = saved_src
    return g, obs


def cmd_record(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = {"kind": a.kind, "map": a.map, "races": a.races.split(","), "pairs": a.pairs, "steps": a.steps,
            "games": a.games, "record_ai_orders": a.record_ai_orders, "difficulty": a.difficulty}
    if a.max_seconds:
        spec["max_game_seconds"] = a.max_seconds
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
    # the run's own settings (run.json next to replays/): its map, pairs, AI orders recorded or not
    cfg = {"map": "duelfast", "pairs": 5, "opd_share": 1.0, "max_minutes": 20.0, "step_seconds": 0.5}
    run = replay.parent.parent / "run.json"
    if run.exists():
        rc = json.loads(run.read_text()).get("config", {})
        for k in cfg:
            if k in rc:
                cfg[k] = type(cfg[k])(rc[k]) if k != "map" else rc[k]
    slots = [Agent(r) if k == "agent" else BuiltinAI(r, "insane") for r, k in zip(races, kinds)]
    setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"],
                      max_game_seconds=cfg["max_minutes"] * 60, victory="decisive",
                      window=(320, 240), draw=False, native_obs=True, d3d_thread=False, render_threads=0, nice=NICE,
                      wait_floor_ms=5, pairs=cfg["pairs"], record_ai_orders=cfg["opd_share"] > 0, warm_spare=False)
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
            env = dict(env)
            prec = env.pop("CUTAB_PERF_RECORD", "")
            g, obs = launch(setup, f"cutprobe{a.tag}_{vi}_{r}", env, shim, replay)
            runs.append({"variant": name, "rep": r, "g": g, "obs": obs, "digests": [], "perf": None, "raw": raw_tap(g),
                         "prec": prec})
    barrier = threading.Barrier(len(runs))

    def play(run: dict) -> None:
        g, obs = run["g"], run["obs"]
        perf = None
        me = threading.get_native_id()
        try:
            pid = game_pid(g.prefix)
            c0 = thread_cpu(pid) if pid else {}
            for t in range(a.steps):
                run["digests"].append(digest(obs) if not a.light else {"ms": obs.game_ms})
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
                    if run["prec"]:
                        run["precp"] = PerfRecord(tid, tmp / f"perf_{a.tag}_{run['variant']}_{run['rep']}.data", run["prec"])
                    run["cpu0"], run["pcpu0"], run["t0"], run["s0"] = thread_cpu(pid)[tid], proc_cpu(pid), time.time(), t
                    run["pp"] = prefix_pids(g.prefix)
                    run["ns0"], run["ws0"], run["oth0"] = thread_ns(pid, tid), pids_cpu(run["pp"]["wineserver"]), pids_cpu(run["pp"]["other"])
                    run["py0"] = thread_ns(os.getpid(), me)
                if a.window and perf and t > a.warmup and (t - a.warmup) % a.window == 0:
                    w = perf.stop()
                    run.setdefault("windows", []).append(w.get("instructions", 0) / a.window / 1e6)
                    run.setdefault("perf_parts", []).append(w)
                    perf = Perf(run["tid"], tmp / f"perf_{run['variant']}_{run['rep']}.txt")
                if (obs.game_over and t >= a.min_steps) or t == a.steps - 1:
                    break
                obs = g.step([])
                run["obs"] = obs
            if run.get("precp"):
                run["precp"].stop()
            if perf:
                run["perf"] = perf.stop()
                for w in run.get("perf_parts", []):  # (the windows' counts add up)
                    for k, v in w.items():
                        run["perf"][k] = run["perf"].get(k, 0) + v
                run["steps_measured"] = len(run["digests"]) - run["s0"]
                run["cpu"] = thread_cpu(pid)[run["tid"]] - run["cpu0"]
                run["ns"] = thread_ns(pid, run["tid"]) - run["ns0"]
                run["py"] = thread_ns(os.getpid(), me) - run["py0"]
                run["ws"] = pids_cpu(run["pp"]["wineserver"]) - run["ws0"]
                run["oth"] = pids_cpu(run["pp"]["other"]) - run["oth0"]
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
    print(f"| variant | rep | steps | M instr/step | main CPU ms/step | process CPU ms/step | wineserver ms/step | other wine ms/step | parity vs live | vs {runs[0]['variant']} |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        p = run.get("perf") or {}
        n = max(run.get("steps_measured", 0), 1)
        live_cmp = compare(live, run["digests"]) if live and not a.light else "-"
        base_cmp = (compare(runs[0]["digests"], run["digests"]) if not a.light else "light") if run is not runs[0] else "-"
        if run is not runs[0]:
            base_cmp += "; " + compare_raw(runs[0]["raw"], run["raw"])
        row = {"variant": run["variant"], "rep": run["rep"], "steps": n, "error": run.get("error"),
               "instr": p.get("instructions", 0) / n / 1e6, "cycles": p.get("cycles", 0) / n / 1e6,
               "task_ms": p.get("task-clock", 0) / n, "cpu_ms": run.get("ns", 0) / 1e6 / n,
               "ws_ms": 1000 * run.get("ws", 0) / n, "oth_ms": 1000 * run.get("oth", 0) / n,
               "py_ms": run.get("py", 0) / 1e6 / n,
               "pcpu_ms": 1000 * run.get("pcpu", 0) / n, "live": live_cmp, "base": base_cmp,
               "hashes": [step_hash(d) for d in run["digests"]] if not a.light else []}
        rows.append(row)
        row["windows"] = run.get("windows", [])
        print(f"| {row['variant']} | {row['rep']} | {n} | {row['instr']:.2f} | "
              f"{row['cpu_ms']:.3f} | {row['pcpu_ms']:.2f} | {row['ws_ms']:.2f} | {row['oth_ms']:.2f} | {live_cmp} | {base_cmp} | py {row['py_ms']:.3f} |" + (f" ERROR {row['error']}" if row["error"] else ""),
              flush=True)
    extra = [k for k in (runs[0].get("perf") or {}) if k not in ("instructions", "cycles", "task-clock")]
    for run, row in zip(runs, rows):
        p = run.get("perf") or {}
        if extra:
            print(f"{row['variant']} per step: " + ", ".join(f"{k} {p.get(k, 0) / row['steps']:.0f}" for k in extra))
    if a.window:
        for row in rows:
            print(f"{row['variant']} M instr/step by {a.window}-step window: " + " ".join(f"{w:.1f}" for w in row["windows"]))
    if a.json:
        Path(a.json).write_text(json.dumps({"replay": str(replay), "rows": rows}, indent=1))
    return 0


# ---- live: seeded games, one per variant, stepped in lockstep with the same commands ------------------------

def cmd_live(a) -> int:
    """A harness change's parity: seeded live games (GameSetup.seed: two games with one seed and the same orders
    play alike) with the variants' harnesses / shims, the scripted agents' commands computed from the first
    variant's observation and sent to all; every step compared, and the main thread's CPU."""
    spec = {"kind": a.kind, "map": a.map, "races": a.races.split(","), "pairs": a.pairs, "seed": a.seed,
            "record_ai_orders": a.record_ai_orders, "difficulty": a.difficulty, "max_game_seconds": a.steps * 0.5 + 60}
    setup = make_setup(spec)
    variants = []
    for v in a.variant:
        name, _, rest = v.partition(":")
        rest, _, shim = rest.partition("@")
        variants.append((name, dict(x.split("=", 1) for x in rest.split(",") if x), Path(shim) if shim else None))
    runs = []
    for r in range(a.repeat):
        for vi, (name, env, shim) in enumerate(variants):
            g, obs = launch(setup, f"cutprobe{a.tag}_{vi}_{r}", env, shim, None)
            runs.append({"variant": name, "rep": r, "g": g, "obs": obs, "digests": [], "raw": raw_tap(g)})
    pol = Scripted(a.seed or 1)
    try:
        for run in runs:
            run["pid"] = game_pid(run["g"].prefix)
            run["c0"] = thread_cpu(run["pid"])
        for t in range(a.steps):
            for run in runs:
                run["digests"].append(digest(run["obs"]))
            lead = runs[0]["obs"]
            if t == a.warmup:
                for run in runs:
                    c1 = thread_cpu(run["pid"])
                    run["tid"] = max(c1, key=lambda k: c1[k] - run["c0"].get(k, 0.0))
                    run["ns0"], run["pc0"], run["s0"] = thread_ns(run["pid"], run["tid"]), proc_cpu(run["pid"]), t
                    run["pp"] = prefix_pids(run["g"].prefix)
                    run["ws0"] = pids_cpu(run["pp"]["wineserver"])
            if any(r_["obs"].game_over for r_ in runs):
                break
            cmds = pol.commands(runs[0]["g"], lead, [p for p in lead.players if lead.players[p].is_agent])
            for run in runs:
                run["g"].send(cmds)
            for run in runs:
                run["obs"] = run["g"].receive()
        for run in runs:
            if "tid" in run:
                n = len(run["digests"]) - run["s0"]
                run["cpu_ms"] = (thread_ns(run["pid"], run["tid"]) - run["ns0"]) / 1e6 / max(n, 1)
                run["pcpu_ms"] = 1000 * (proc_cpu(run["pid"]) - run["pc0"]) / max(n, 1)
                run["ws_ms"] = 1000 * (pids_cpu(run["pp"]["wineserver"]) - run["ws0"]) / max(n, 1)
                run["n"] = n
    finally:
        for run in runs:
            try:
                run["g"].close()
            except Exception:
                pass
    print(f"| variant | rep | steps measured | main CPU ms/step | process CPU ms/step | wineserver ms/step | vs {runs[0]['variant']} |")
    print("|---|---|---|---|---|---|---|")
    for run in runs:
        cmp_ = "-" if run is runs[0] else compare(runs[0]["digests"], run["digests"]) + "; " + compare_raw(runs[0]["raw"], run["raw"])
        print(f"| {run['variant']} | {run['rep']} | {run.get('n', 0)} | {run.get('cpu_ms', 0):.3f} | {run.get('pcpu_ms', 0):.2f} | "
              f"{run.get('ws_ms', 0):.2f} | {cmp_} |", flush=True)
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
    r.add_argument("--max-seconds", type=float, default=0.0, help="the game's time limit (a tie), default 1200 s")
    r.add_argument("--env", default="")
    r.add_argument("--shim")
    lv = sub.add_parser("live")
    lv.add_argument("--variant", action="append", required=True)
    lv.add_argument("--kind", default="mixed", choices=["ai", "agents", "mixed"])
    lv.add_argument("--map", default="echofast")
    lv.add_argument("--races", default="human,orc")
    lv.add_argument("--difficulty", default="insane")
    lv.add_argument("--steps", type=int, default=1200)
    lv.add_argument("--warmup", type=int, default=50)
    lv.add_argument("--pairs", type=int, default=1)
    lv.add_argument("--seed", type=int, default=7)
    lv.add_argument("--repeat", type=int, default=1)
    lv.add_argument("--tag", default="L")
    lv.add_argument("--record-ai-orders", action="store_true")
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
    p.add_argument("--light", action="store_true", help="no digests (the raw observation bytes still compared): the "
                   "controller thread's own CPU a step is then mostly the step sync's")
    p.add_argument("--tmp", default=str(Path.home() / "tools/wt-cuts2-out/perf"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    return cmd_record(a) if a.cmd == "record" else cmd_live(a) if a.cmd == "live" else cmd_play(a)


if __name__ == "__main__":
    raise SystemExit(main())
