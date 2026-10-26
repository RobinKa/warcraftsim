"""A game rendered on request (the dashboard's Render button; the user's idea, 2026-10-24: "a button on the dashboard
to request a render with specific settings (e.g. races, targets, which players)"): one game between a run's policy
and the built-in AI or another policy, with the races, target strategies, length and map asked for, played as the
run's evaluation games are (selfplay.play_one with the run's settings: evalworker.run_config; no trajectories, no
tax, no teacher), then rendered with the followed policy's panel by the run's own renderer (selfplay.render_video).

Everything lives in runs/renders/, which the Videos page lists as a run of its own (media.jsonl, videos/,
replays/, episodes.jsonl for the videos' info): requests/<id>.json (each request and its state: queued, playing,
rendering, done, failed) with its log, and current.pt, the followed policy's weights (Nets reads them there: the
run's own current.pt is only read). One request at a time: the dashboard starts the runner when a request comes and
none runs (runner.lock); it plays each queued request in a process of its own (a run's inputs and map are module
settings) and exits when none is left. Inference on the CPU by default (the run owns the GPU), all of it niced like
the run's renderer.

    python -m warcraftsim.fullgame.render_request              # the runner: the queued requests, oldest first
    python -m warcraftsim.fullgame.render_request --one 3      # request 3 alone (what the runner starts)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from .. import paths

RUNS = paths.REPO_ROOT / "runs"
RACES = ("human", "orc", "undead", "nightelf")
DIFFICULTIES = ("easy", "normal", "insane")
MAX_MINUTES = 40.0
ACTIVE = ("queued", "playing", "rendering")
THREADS = 4  # (the CPU inference's: a game's one or two sides a step)
_NAMES: dict[str, str] | None = None
_SUBMIT = threading.Lock()


def renders(runs: Path | None = None) -> Path:
    return Path(runs or RUNS) / "renders"


def policy_file(spec: str, run: str, runs: Path | None = None) -> Path | None:
    """A policy's file: "current" (the run's live weights) or a checkpoint's path relative to the repository
    (runs/<run>/checkpoints/<steps>.pt, any run's); None when there is no such file under runs/."""
    runs = Path(runs or RUNS)
    p = (runs / run / "current.pt" if spec == "current" else runs.parent / str(spec)).resolve()
    return p if p.suffix == ".pt" and p.is_file() and runs.resolve() in p.parents else None


def label(spec: str, run: str) -> str:
    """'current' -> 'fgself-142 current'; 'runs/fgself-140/checkpoints/0000000002631004.pt' -> 'fgself-140/2631004'."""
    if spec == "current":
        return f"{run} current"
    p = Path(spec)
    return f"{p.parent.parent.name}/{int(p.stem)}" if p.stem.isdigit() else p.stem


def pool_path(run_dir: Path, where: str = "real") -> Path | None:
    """The run's strategies on the map asked for: the evaluation map's (--eval-z-pool, else --z-pool) or the
    training map's (--z-pool)."""
    c = json.loads((run_dir / "run.json").read_text())["config"]
    path = (c.get("eval_z_pool") or c.get("z_pool")) if where == "real" else c.get("z_pool")
    if not path or path == "None":
        return None
    p = Path(path) if Path(path).is_absolute() else run_dir.parent.parent / path
    return p if p.exists() else None


def validate(req: dict, runs: Path | None = None) -> tuple[dict | None, str]:
    """A request as the dashboard sends it -> (the request to queue, "") or (None, what is wrong). Fields: run; a
    (A's policy: "current" or a checkpoint); b ({"kind": "ai", "difficulty"} or {"kind": "policy", "policy"});
    races [A's, B's] (a race or "random"); targets [A's, B's] ("none", "random": one drawn as the evaluation games
    draw them, or "sig:<signature>": one of the pool's strategies of that build); follow ("a" or "b": the policy
    the panel and camera follow); minutes (the game's length limit); map ("real": the run's evaluation map, or
    "training")."""
    runs = Path(runs or RUNS)
    if not isinstance(req, dict):
        return None, "a request is a JSON object"
    run = str(req.get("run", ""))
    if not run or "/" in run or run.startswith(".") or not (runs / run / "run.json").exists() \
            or not (runs / run / "vocab.json").exists():
        return None, f"no run {run!r}"
    a = str(req.get("a") or "current")
    if policy_file(a, run, runs) is None:
        return None, f"no policy {a!r}"
    b = req.get("b") or {"kind": "ai", "difficulty": "insane"}
    if not isinstance(b, dict):
        return None, "the opponent: the built-in AI or a policy"
    if b.get("kind") == "ai":
        if b.get("difficulty") not in DIFFICULTIES:
            return None, "the AI: easy, normal or insane"
        b = {"kind": "ai", "difficulty": b["difficulty"]}
    elif b.get("kind") == "policy":
        if policy_file(str(b.get("policy", "")), run, runs) is None:
            return None, f"no policy {b.get('policy')!r}"
        b = {"kind": "policy", "policy": str(b["policy"])}
    else:
        return None, "the opponent: the built-in AI or a policy"
    races = req.get("races") or ["random", "random"]
    if not isinstance(races, list) or len(races) != 2 or any(r not in (*RACES, "random") for r in races):
        return None, "races: human, orc, undead, nightelf or random"
    targets = req.get("targets") or ["none", "none"]
    if not isinstance(targets, list) or len(targets) != 2 \
            or any(not (t in ("none", "random") or (isinstance(t, str) and t.startswith("sig:") and len(t) > 4)) for t in targets):
        return None, "targets: none, random or a build (sig:...)"
    targets = list(targets)
    if b["kind"] == "ai":
        targets[1] = "none"  # (the built-in AI plays its own)
    where = str(req.get("map") or "real")
    if where not in ("real", "training"):
        return None, "map: real or training"
    if any(t != "none" for t in targets) and pool_path(runs / run, where) is None:
        return None, "the run has no strategies for targets"
    follow = str(req.get("follow") or "a")
    if follow not in ("a", "b") or (follow == "b" and b["kind"] != "policy"):
        return None, "the panel follows a policy: a, or b when it is one"
    try:
        minutes = float(req.get("minutes", 15))
    except (TypeError, ValueError):
        return None, "minutes: a number"
    if not 1 <= minutes <= MAX_MINUTES:
        return None, f"minutes: 1 to {MAX_MINUTES:g}"
    return {"run": run, "a": a, "b": b, "races": [str(r) for r in races], "targets": targets, "follow": follow,
            "minutes": minutes, "map": where}, ""


# ---- the queue: requests/<id>.json, played one at a time by the runner ------------------------------------------
def requests(runs: Path | None = None) -> list[dict]:
    out = []
    for f in (renders(runs) / "requests").glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(out, key=lambda r: r["id"])


def save(r: dict, runs: Path | None = None) -> None:
    d = renders(runs) / "requests"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{r['id']}.json.tmp"
    tmp.write_text(json.dumps(r))
    tmp.replace(d / f"{r['id']}.json")


def _lock(runs: Path | None = None):
    """The runner's lock (held while the file stays open) or None when another process holds it."""
    d = renders(runs)
    d.mkdir(parents=True, exist_ok=True)
    f = open(d / "runner.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except OSError:
        f.close()
        return None


def runner_alive(runs: Path | None = None) -> bool:
    f = _lock(runs)
    if f is None:
        return True
    f.close()
    return False


def submit(req: dict, runs: Path | None = None, start: bool = True) -> tuple[bool, str]:
    """Queue a request (validate's fields) and start the runner when none runs -> (queued, its id or what is wrong)."""
    q, err = validate(req, runs)
    if q is None:
        return False, err
    with _SUBMIT:  # (the dashboard's threads: one id each)
        rs = requests(runs)
        r = {"id": rs[-1]["id"] + 1 if rs else 1, "time": time.time(), "state": "queued", "request": q}
        save(r, runs)  # (before the runner's test: a runner about to exit looks again once it lets go of its lock)
        if start and not runner_alive(runs):
            with open(renders(runs) / "runner.log", "a") as log:
                subprocess.Popen([sys.executable, "-m", "warcraftsim.fullgame.render_request", "--runs", str(Path(runs or RUNS))],
                                 cwd=Path(runs or RUNS).parent, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
    return True, str(r["id"])


def listing(runs: Path | None = None, n: int = 30) -> dict:
    """The newest requests, their states (one under way whose runner is gone: failed) and how many wait."""
    rs = requests(runs)
    alive = runner_alive(runs)
    out = []
    for r in rs[-n:][::-1]:
        if r["state"] in ACTIVE[1:] and not alive:
            r = {**r, "state": "failed", "error": r.get("error") or "the runner stopped"}
        out.append(r)
    return {"requests": out, "queued": sum(r["state"] == "queued" for r in rs), "runner": alive}


# ---- what the form offers ---------------------------------------------------------------------------------------
def _names() -> dict[str, str]:
    global _NAMES
    if _NAMES is None:
        try:
            from ..data.objects import unit_names, upgrade_names
            _NAMES = {**unit_names(), **upgrade_names()}
        except Exception:  # noqa: BLE001 (no game files: the codes)
            _NAMES = {}
    return _NAMES


def builds(pool: dict, names: dict | None = None, per_race: int = 80) -> dict[str, list]:
    """A pool's builds (narrow strategies' signatures: heroes | main army) per race, the most common first:
    [signature, its strategies, of them won, its heroes by name, its army by name]."""
    names = _names() if names is None else names
    count: dict[tuple, list] = {}
    for s in pool.get("strategies", []):
        if s.get("race") in RACES and s.get("sig"):
            c = count.setdefault((s["race"], s["sig"]), [0, 0])
            c[0] += 1
            c[1] += bool(s.get("won"))
    out: dict[str, list] = {r: [] for r in RACES}
    for (race, sig), (n, won) in sorted(count.items(), key=lambda kv: (-kv[1][0], kv[0][1])):
        if len(out[race]) < per_race:
            heroes, _, army = sig.partition("|")
            nm = lambda part: [names.get(c, c) for c in part.split("+") if c and c != "-"]  # noqa: E731
            out[race].append([sig, n, won, nm(heroes), nm(army)])
    return out


def options(run: str, runs: Path | None = None, where: str = "real") -> dict:
    """What the form offers for a run: its policies (its live weights, its newest checkpoints, other recent runs'
    last) and the builds its strategies hold per race on the map asked for."""
    runs = Path(runs or RUNS)
    run_dir = runs / run
    if not run or "/" in run or run.startswith(".") or not (run_dir / "run.json").exists():
        return {"error": f"no run {run!r}"}
    pol = [{"value": "current", "label": f"{run} · current"}]
    cks = sorted((run_dir / "checkpoints").glob("[0-9]*.pt"))[::-1][:12]
    pol += [{"value": str(p.relative_to(runs.parent)), "label": label(str(p), run)} for p in cks]
    others = sorted((r for r in runs.glob("fgself-*") if r.name != run and (r / "checkpoints").is_dir()),
                    key=lambda r: r.stat().st_mtime, reverse=True)[:8]
    for r in others:
        for p in sorted(r.glob("checkpoints/[0-9]*.pt"))[-1:]:
            pol.append({"value": str(p.relative_to(runs.parent)), "label": f"{label(str(p), run)} · its last"})
    pp = pool_path(run_dir, where if where in ("real", "training") else "real")
    c = json.loads((run_dir / "run.json").read_text())["config"]
    return {"run": run, "policies": pol, "builds": builds(json.loads(pp.read_text())) if pp else {},
            "maps": {"real": c.get("eval_map") or c.get("map"), "training": c.get("map")}, "max_minutes": MAX_MINUTES}


# ---- the game and its video -------------------------------------------------------------------------------------
def pick_target(pool: dict, spec: str, race: str, opp_race: str, rng: random.Random, src: str = "") -> dict | None:
    """A side's target strategy: "none", "random" (as the evaluation games draw one: the matchup's winners first)
    or "sig:<signature>" (one of the pool's strategies of that build for the race: won against the opponent's
    race first, as strategy.sample orders them)."""
    from . import strategy as strat
    if spec == "none":
        return None
    if spec == "random":
        return strat.sample(pool, race, opp_race, rng, True, src)
    mine = [s for s in pool["strategies"] if s.get("race") == race and s.get("sig") == spec[4:]]
    tiers = ([s for s in mine if s["opp"] == opp_race and s["won"]], [s for s in mine if s["won"]],
             [s for s in mine if s["opp"] == opp_race], mine)
    c = next((c for c in tiers if c), None)
    if c is None:
        raise ValueError(f"no {race} strategy of the build {spec[4:]} on this map")
    return rng.choice(c)


def play_and_render(r: dict, runs: Path, update) -> dict:
    """Play the request's game and render its video -> what the request's row gets (its video under runs/renders,
    the outcome, the races and targets played). `update(state)` marks its progress."""
    import numpy as np
    import torch

    from ..runtime import reaper, wine
    from ..runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
    from . import features as fx
    from .collect import claim_slot
    from .evalworker import newest_checkpoint, run_config
    from .model import load
    from .selfplay import (Inference, Nets, agent_difficulty, film_game, play_one, publish, render_video,
                           strategy_pool)
    q, out_dir = r["request"], renders(runs)
    run_dir = runs / q["run"]
    rng = random.Random(f"render/{r['id']}/{r['time']}")
    cfg, _ = run_config(run_dir, "render", 1, r["id"])
    w = {k[1:]: cfg.pop(k) for k in list(cfg) if k.startswith("_")}
    if w["wine"]:  # (before any game: they read the environment)
        wine.use_wine(w["wine"], w["fsync"], w["ntsync"])
    wine.ensure_template()
    if q["map"] == "training":  # (run_config's are the evaluation map's)
        c = json.loads((run_dir / "run.json").read_text())["config"]
        cfg["map"], cfg["z_pool"] = c["map"], c.get("z_pool") or ""
    cfg["run_dir"] = str(out_dir)  # (Nets' current.pt, the replays and the video: here, never in the run)
    cfg["max_minutes"] = q["minutes"]
    cfg["max_steps"] = int(q["minutes"] * 60 / cfg["step_seconds"]) + 20
    cfg["eval_checkpoints"] = []
    fx.ACTIVE_F = cfg["features"]
    os.environ["WARCRAFTSIM_FEATURES"] = str(cfg["features"])
    fx.set_inputs(cfg["inputs"], bool(cfg["privileged"]))  # (before the networks: as evalworker.main)
    fx.set_masks(cfg["tech_mask"], cfg["valid_mask"])
    fx.set_neutral(cfg["neutral"])
    fx.set_macros(int(cfg.get("macros", 0) or 0))
    fx.set_spells(int(cfg.get("spells", 0) or 0))
    fx.set_fine(int(cfg.get("fine_points", 0) or 0))  # (--fine-points: the point indices)
    fx.set_macros_off(str(cfg.get("macros_off") or "").split(","))
    from ..data.duelmap import parse_duel_name
    if parse_duel_name(cfg["map"]) is None:
        from ..data.mapbuild import stock_map_path
        fx.set_geometry(*fx.map_geometry(stock_map_path(cfg["map"])))
    # the followed policy plays as the run's "current" (play_one's learner: the panel's values and target), the
    # other policy as a snapshot
    a, b = q["a"], q["b"]
    lead, other = (a, b.get("policy")) if q["follow"] == "a" else (b["policy"], a)
    ck = torch.load(policy_file(lead, q["run"], runs), map_location="cpu", weights_only=False)
    version = int(ck.get("version", ck.get("update", 0)) or 0)
    newest = newest_checkpoint(run_dir)  # (the run's strategy block, as evalworker's: other runs' grown to it)
    cfg["z_dim"] = (torch.load(newest, map_location="cpu", weights_only=False) if newest else ck)["config"].get("z", 0)
    fx.set_z(cfg["z_dim"])
    del ck
    vocab = json.loads(Path(cfg["vocab"]).read_text())
    # (as a listed checkpoint is in the run's evaluation games, Nets.get: another run's grown to this run's inputs,
    # its orders in this run's vocabulary's order)
    out_dir.mkdir(parents=True, exist_ok=True)
    publish(load(policy_file(lead, q["run"], runs), "cpu", vocab=vocab if fx.NEUTRAL else None)[0], version, out_dir)
    from .costs import order_costs
    from .trace import unit_values
    cfg["values"] = unit_values()
    cfg["costs"] = order_costs(vocab, cfg["map"]).tolist() if cfg["avail_mask"] else None
    if cfg["avail_mask"]:
        cfg["costs_array"] = np.asarray(cfg["costs"], np.int64)
    cfg["order_names"] = vocab.get("order_names") or {}
    cfg["device"] = os.environ.get("WARCRAFTSIM_RENDER_DEVICE", "cpu")
    cfg["slot"], slot_lock = claim_slot(runs)
    cfg["learner_pid"] = os.getpid()
    torch.set_num_threads(THREADS)
    device = torch.device(cfg["device"])
    nets = Nets(cfg, device)
    infer = Inference(nets, device, 2, compile_=False, buckets=(1, 2), entities=tuple(cfg["infer_entities"]))
    # the slots: A's drawn (the start locations differ), the random races drawn
    played = [x if x != "random" else rng.choice(RACES) for x in q["races"]]
    slot_a = rng.randrange(2)
    slot_of = {"a": slot_a, "b": 1 - slot_a}
    races = [""] * 2
    races[slot_a], races[1 - slot_a] = played
    side = slot_of[q["follow"]]
    adv = agent_difficulty(cfg)  # (a person's income)
    slots = [Agent(races[s], handicap=cfg["handicap"], difficulty=adv) for s in (0, 1)]
    if b["kind"] == "ai":
        slots[slot_of["b"]] = BuiltinAI(races[slot_of["b"]], b["difficulty"], handicap=cfg["handicap"])
        opp = {"name": f"script:ai-{b['difficulty']} (real)", "kind": "ai"}
    else:
        opp = {"name": label(other, q["run"]), "kind": "past", "path": str(policy_file(other, q["run"], runs))}
    pool = strategy_pool(cfg) if any(t != "none" for t in q["targets"]) else None
    targets = {}
    for who, spec in zip("ab", q["targets"]):
        s = slot_of[who]
        st = pick_target(pool, spec, races[s], races[1 - s], rng, cfg.get("z_src", "")) if pool else None
        if st is not None:
            targets[s] = st
    opp["targets"] = targets  # (play_one: each policy side's target)
    setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"],
                      max_game_seconds=q["minutes"] * 60, victory="decisive", window=(320, 240), draw=False,
                      wait_floor_ms=cfg["wait_floor_ms"], native_obs=cfg["native_obs"], d3d_thread=False,
                      render_threads=0, cuts=cfg.get("game_cuts") or None)  # (niced: this process is)
    jobs: list[tuple] = []

    class Rows:  # (play_one's out_q: the episode row only, no learner to send it to)
        def put(self, msg):
            pass

    class Jobs:  # (film_game's render queue: rendered here, after the game)
        def put(self, job):
            jobs.append(job)
    them = f"{b['difficulty']} AI" if b["kind"] == "ai" else label(b["policy"], q["run"])
    tg = ["none" if slot_of[w] not in targets else targets[slot_of[w]].get("sig") or "a target" for w in "ab"]
    try:
        update("playing")
        with GameInstance(setup, name=f"fgrq{cfg['slot']}", timeout=120) as g:
            obs = g.start()
            ep = play_one(g, obs, cfg, vocab, races, side, opp, infer, Rows(), r["id"], record=True, train=False)
            tr = ep.get("trace")
            if tr:  # (the panel's: the policies by name, not the training run's words)
                tr["title"] = f"{q['run']} · {label(lead, q['run'])} (v{version}) · vs {them if q['follow'] == 'a' else label(a, q['run'])}"
                tr["sides"][0]["name"] = f"{label(lead, q['run'])} ({races[side]})"
                if b["kind"] == "policy":
                    tr["sides"][1]["name"] = f"{label(other, q['run'])} ({races[1 - side]})"
            film_game(g, ep, cfg, r["id"], Jobs())
        if not jobs:
            raise RuntimeError("the game's replay was not saved")
        update("rendering")
        setup_, replay, stem, trace, row, t = jobs[0]
        row = {**row, "request": r["id"],
               "sub": f"request {r['id']} · {label(a, q['run'])} ({races[slot_a]}) vs {them} ({races[1 - slot_a]}) · "
                      f"targets {tg[0]} / {tg[1]} · {ep['game_time'] / 60:.1f} of {q['minutes']:g} minutes · {cfg['map']}"}
        with open(out_dir / "episodes.jsonl", "a") as f:  # (the video's info: videos.VideoIndex matches it by time)
            f.write(json.dumps({**{k: v for k, v in ep.items() if k not in ("eval", "eval_key")}, "request": r["id"]}) + "\n")
        video = render_video(cfg, (setup_, replay, stem, trace, row, t), nice=0)
        # (the list shows "A vs B": A's result; the game's row is the followed policy's, b's when it follows b; audit 7)
        return {"video": str(video.relative_to(out_dir)), "outcome": ep["outcome"] if q["follow"] == "a" else -ep["outcome"],
                "game_time": round(ep["game_time"], 1),
                "races": [races[slot_a], races[1 - slot_a]], "targets": tg, "version": version, "slot_a": slot_a}
    finally:
        slot_lock.close()
        reaper.reap()


def run_one(rid: int, runs: Path) -> int:
    """One request, in this process (the runner's child): its state kept in its file."""
    from .selfplay import RENDER_NICE
    os.nice(RENDER_NICE)  # (the game and the renderer inherit it: the run's games' and renderer's level)
    r = next(x for x in requests(runs) if x["id"] == rid)

    def update(state):
        r.update(state=state, **({"started": time.time()} if state == "playing" else {}))
        save(r, runs)
    try:
        r.update(play_and_render(r, runs, update), state="done", finished=time.time())
        print(f"request {rid}: done in {r['finished'] - r['started']:.0f} s: {r['video']}", flush=True)
    except Exception:  # noqa: BLE001 (the request failed: its error's last lines)
        tb = traceback.format_exc()
        print(tb, flush=True)
        r.update(state="failed", finished=time.time(), error="\n".join(tb.strip().splitlines()[-6:]))
    save(r, runs)
    return 0 if r["state"] == "done" else 1


def runner(runs: Path) -> int:
    """The queued requests one at a time, oldest first, each in a process of its own; exits when none is left."""
    lock = _lock(runs)
    if lock is None:
        return 0  # (another runner plays them)
    for r in requests(runs):  # (one under way when a runner stopped: it is not taken up again)
        if r["state"] in ACTIVE[1:]:
            save({**r, "state": "failed", "error": r.get("error") or "the runner stopped"}, runs)
    logs = renders(runs) / "requests"
    while True:
        queued = [r for r in requests(runs) if r["state"] == "queued"]
        if not queued:
            lock.close()  # (then a last look: a request saved meanwhile found the lock held and started none)
            if not any(r["state"] == "queued" for r in requests(runs)) or (lock := _lock(runs)) is None:
                return 0
            continue
        rid = queued[0]["id"]
        print(f"request {rid}: {json.dumps(queued[0]['request'])}", flush=True)
        with open(logs / f"{rid}.log", "a") as log:
            code = subprocess.call([sys.executable, "-m", "warcraftsim.fullgame.render_request", "--runs", str(runs),
                                    "--one", str(rid)], cwd=runs.parent, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        r = next(x for x in requests(runs) if x["id"] == rid)
        if r["state"] in ACTIVE:  # (the process died before it could say)
            tail = (logs / f"{rid}.log").read_text(errors="replace").strip().splitlines()[-6:]
            r = {**r, "state": "failed", "finished": time.time(), "error": "\n".join(tail) or f"exit code {code}"}
            save(r, runs)
        print(f"request {rid}: {r['state']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=RUNS, help="the runs' directory (runs/renders holds the requests)")
    ap.add_argument("--one", type=int, help="play and render this request only")
    a = ap.parse_args(argv)
    return run_one(a.one, a.runs) if a.one is not None else runner(a.runs)


if __name__ == "__main__":
    raise SystemExit(main())
