"""Self-play reinforcement learning on whole games, from a behavior-cloned policy (after AlphaStar).

    python3 -m warcraftsim.fullgame.selfplay --name fgself-1 --init runs/bc/fullgame-1/policy.pt

(the torch Python). Actor processes play games on the duel map and send trajectories to the
learner (this process), which trains the policy and a value head with PPO:
* Actors: each runs a few games (threads) and batches its agents' network calls on the GPU. An
  agent sees and orders as the clone does in fullgame/play.py (play.BCAgent's view and commands).
  Games of one setup (races, built-in AI or not) run one after another in one process
  (GameInstance.restart reloads the map).
* The league (league.json): the learner plays itself (both sides train), past snapshots
  (prioritized fictitious self-play: the ones it beats less, more often) and the built-in AI (a
  fixed anchor: the win rate against it is the run's yardstick, "script:ai-<difficulty>").
* Reward: +1 for a win, -1 for a loss, 0 for a tie (the time limit).
* PPO per unit: each own unit's decision (its order, and the targets that order uses) has its own
  clipped ratio; they share the step's advantage (GAE, computed by the actors with the values
  they acted with).
* A KL term towards the behavior-cloned policy (AlphaStar's towards its supervised policy) keeps
  the policy near what it learned from the AI; the value head starts untrained, so the first
  updates train the value only (--value-warmup).

Writes runs/<name>/ as the dashboard reads it: run.json, train.jsonl, episodes.jsonl,
league.json, checkpoints/<agent steps>.pt (play.py and further runs can load them).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import features as fx
from .collect import claim_slot
from .model import FullGameNet, act, evaluate, load
from ..rl.league import Member, pfsp_weight

RUNS = Path(__file__).resolve().parents[2] / "runs"


# ---- actors ---------------------------------------------------------------------------------------

class Nets:
    """An actor's networks: the learner's current weights (reloaded when the learner publishes
    new ones) and past snapshots (a few, most recently used)."""

    def __init__(self, cfg: dict, device):
        self.cfg, self.device = cfg, device
        self.current_path = Path(cfg["run_dir"]) / "current.pt"
        self.current: FullGameNet | None = None
        self.version = -1
        self.mtime = 0.0
        self.checked = 0.0
        self.snapshots: dict[str, FullGameNet] = {}
        self.reload()

    def reload(self) -> None:
        self.checked = time.time()
        try:
            mtime = self.current_path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self.mtime:
            return
        for _ in range(3):
            try:
                ck = torch.load(self.current_path, map_location=self.device, weights_only=False)
                break
            except (EOFError, RuntimeError, OSError):  # being replaced
                time.sleep(0.2)
        else:
            return
        if self.current is None:
            self.current = FullGameNet(**ck["config"]).to(self.device).eval()
        self.current.load_state_dict(ck["model"])
        self.version, self.mtime = ck["version"], mtime

    def get(self, key: str) -> FullGameNet:
        if key == "current":
            if time.time() - self.checked > 5.0:
                self.reload()
            return self.current
        net = self.snapshots.pop(key, None)
        if net is None:
            net, _ = load(key, self.device)
        self.snapshots[key] = net  # most recently used last
        while len(self.snapshots) > 4:
            self.snapshots.pop(next(iter(self.snapshots)))
        return net


class Inference:
    """Batches the network calls of an actor's agents (one per game side) on the GPU."""

    def __init__(self, nets: Nets, device):
        self.nets, self.device = nets, device
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def request(self, items: list[tuple[str, dict]]) -> list[dict]:
        """[(net key, a view step)] -> [{order, tgt, bx, by, logp (per own unit), value, version}]."""
        done = threading.Event()
        out: list = [None] * len(items)
        for k, (key, st) in enumerate(items):
            self.q.put((key, st, out, k, done, len(items)))
        done.wait()
        return out

    def _run(self) -> None:
        pending: dict[int, int] = {}
        while True:
            batch = [self.q.get()]
            deadline = time.time() + 0.003  # a few ms for the other games' agents
            while True:
                try:
                    batch.append(self.q.get(timeout=max(0.0, deadline - time.time())))
                except queue.Empty:
                    break
            groups: dict[str, list] = {}
            for item in batch:
                groups.setdefault(item[0], []).append(item)
            for key, items in groups.items():
                try:
                    results = self._forward(self.nets.get(key), [it[1] for it in items])
                except Exception:  # noqa: BLE001 (the games must not hang on it)
                    traceback.print_exc()
                    results = [None] * len(items)
                version = self.nets.version if key == "current" else -1
                for (key_, st, out, k, done, n), res in zip(items, results):
                    if res is not None:
                        res["version"] = version
                    out[k] = res
                    pending[id(out)] = pending.get(id(out), 0) + 1
                    if pending[id(out)] == n:
                        del pending[id(out)]
                        done.set()

    @torch.no_grad()
    def _forward(self, net: FullGameNet, sts: list[dict]) -> list[dict]:
        B, E = len(sts), max(st["n"] for st in sts)
        ent = np.zeros((B, E, fx.F), np.float32)
        typ, cur = np.zeros((B, E), np.int64), np.zeros((B, E), np.int64)
        mask = np.zeros((B, E), bool)
        glob = np.stack([st["glob"] for st in sts])
        n_own = np.array([min(st["n_own"], fx.MAX_OWN) for st in sts])
        for i, st in enumerate(sts):
            n = st["n"]
            ent[i, :n], typ[i, :n], cur[i, :n], mask[i, :n] = st["ent"], st["type"], st["cur"], True
        t = lambda a: torch.from_numpy(a).to(self.device, non_blocking=True)  # noqa: E731
        out = act(net, t(ent), t(typ), t(cur), t(mask), t(glob), t(n_own))
        cpu = {k: v.cpu().numpy() for k, v in out.items()}
        res = []
        for i in range(B):
            o = n_own[i]
            res.append({"order": cpu["order"][i, :o], "tgt": cpu["tgt"][i, :o], "bx": cpu["bx"][i, :o],
                        "by": cpu["by"][i, :o], "logp": cpu["logp"][i, :o], "value": float(cpu["value"][i])})
        return res


class Trajectory:
    """One trained side's steps; complete chunks (GAE with the values it acted with) go to the learner."""

    def __init__(self, cfg: dict, out_q, info: dict):
        self.cfg, self.out_q, self.info = cfg, out_q, info
        self.steps: list[dict] = []

    def add(self, st: dict, res: dict) -> None:
        n = st["n"]
        self.steps.append({"ent": st["ent"].astype(np.float16), "type": st["type"].astype(np.int16),
                           "cur": st["cur"].astype(np.int16), "glob": st["glob"], "n": n,
                           "n_own": min(st["n_own"], fx.MAX_OWN),
                           "order": res["order"].astype(np.int16), "tgt": res["tgt"].astype(np.int16),
                           "bx": res["bx"].astype(np.int16), "by": res["by"].astype(np.int16),
                           "logp": res["logp"].astype(np.float32), "value": res["value"], "reward": 0.0,
                           "done": False, "version": res["version"]})
        if len(self.steps) > self.cfg["chunk"]:
            self.flush(bootstrap=True)

    def end(self, reward: float) -> None:
        if self.steps:
            self.steps[-1]["reward"] = reward
            self.steps[-1]["done"] = True
            self.flush(bootstrap=False)

    def flush(self, bootstrap: bool) -> None:
        """GAE over the steps (the last one only bootstraps the value when the game goes on)."""
        gamma, lam = self.cfg["gamma"], self.cfg["lam"]
        steps = self.steps[:-1] if bootstrap else self.steps
        next_v = self.steps[-1]["value"] if bootstrap else 0.0
        adv = 0.0
        for s in reversed(steps):
            nonterminal = 0.0 if s["done"] else 1.0
            delta = s["reward"] + gamma * next_v * nonterminal - s["value"]
            adv = delta + gamma * lam * nonterminal * adv
            s["adv"], s["ret"] = adv, adv + s["value"]
            next_v = s["value"]
        if steps:
            self.out_q.put({"steps": steps, "info": self.info})
        self.steps = self.steps[-1:] if bootstrap else []


def _choose(rng: random.Random, options: list[dict]) -> dict:
    total = sum(o["p"] for o in options)
    x = rng.random() * total
    for o in options:
        x -= o["p"]
        if x <= 0:
            return o
    return options[-1]


def game_loop(wid: int, k: int, cfg: dict, infer: Inference, out_q, stop) -> None:
    from ..runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
    from .play import BCAgent

    rng = random.Random(cfg["seed"] * 1000 + wid * 16 + k)
    films = wid == 0 and k == 0 and cfg["video_every"] > 0  # this game slot records the videos
    next_video = time.time() + 120
    vocab = json.loads(Path(cfg["vocab"]).read_text())
    spec_path = Path(cfg["run_dir"]) / "league_spec.json"
    name = f"fgsp{cfg['slot']}_{wid}_{k}"
    while not stop.is_set():
        try:
            spec = json.loads(spec_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
            continue
        races = [rng.choice(cfg["races"]), rng.choice(cfg["races"])]
        if cfg["mirror"]:  # both sides the same race (balanced by construction)
            races[1] = races[0]
        side = rng.randrange(2)  # the learner's
        ai = _choose(rng, spec["launch"])  # {"kind": "agents"} or {"kind": "ai", "difficulty"}
        slots = [Agent(races[0], handicap=cfg["handicap"]), Agent(races[1], handicap=cfg["handicap"])]
        if ai["kind"] == "ai":
            slots[1 - side] = BuiltinAI(races[1 - side], ai["difficulty"], handicap=cfg["handicap"])
        agents_only = ai["kind"] != "ai"
        # without the built-in AI a restart resets the game by script (0.1 s; the AI does not survive
        # that, so games against it reload the map: ~5 s)
        setup = GameSetup(map=cfg["map"], slots=slots, step_seconds=cfg["step_seconds"],
                          max_game_seconds=cfg["max_minutes"] * 60, victory="decisive", window=(320, 240),
                          wait_floor_ms=cfg["wait_floor_ms"], melee_reset=agents_only and cfg["scripted_reset"])
        per_launch = cfg["games_per_process"] * (cfg["agent_games_factor"] if agents_only and cfg["scripted_reset"] else 1)
        try:
            with GameInstance(setup, name=name, timeout=120) as g:
                obs = g.start()
                for n in range(per_launch):
                    if stop.is_set():
                        return
                    try:
                        spec = json.loads(spec_path.read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass
                    opp = ({"name": f"script:ai-{ai['difficulty']}", "kind": "ai"} if ai["kind"] == "ai"
                           else _choose(rng, spec["agents"]))
                    # a video: the first game of a launch (its replay then holds just this game)
                    film = films and n == 0 and time.time() >= next_video
                    ep = play_one(g, obs, cfg, vocab, races, side, opp, infer, out_q, wid)
                    if film:
                        next_video = time.time() + cfg["video_every"] * 60
                        film_game(g, ep, cfg)
                    if n + 1 < per_launch:
                        obs = g.restart()  # after a video: a new launch
        except Exception as e:  # noqa: BLE001 (a new launch)
            print(f"actor {wid}/{k}: {type(e).__name__}: {e}", flush=True)
            time.sleep(2)


def play_one(g, obs, cfg: dict, vocab: dict, races: list[str], side: int, opp: dict, infer: Inference,
             out_q, wid: int) -> None:
    from .play import BCAgent

    race_ix = [fx.RACES.index(r) if r in fx.RACES else 0 for r in races]
    bots: dict[int, BCAgent] = {}
    keys: dict[int, str] = {}
    trajs: dict[int, Trajectory] = {}
    for s in (0, 1):
        if g.setup.slots[s].kind != "agent":
            continue
        bot = BCAgent(None, vocab, s, None)
        bot.begin(obs, race_ix)
        bots[s] = bot
        keys[s] = "current" if s == side or opp["kind"] == "self" else opp["path"]
        if keys[s] == "current":
            trajs[s] = Trajectory(cfg, out_q, {"worker": wid, "opponent": opp["name"]})
    t, t0 = 0, time.time()
    while True:
        sts = {s: bot.observe(obs, t) for s, bot in bots.items()}
        live = [s for s in bots if sts[s] is not None]
        results = infer.request([(keys[s], sts[s]) for s in live]) if live else []
        cmds, spans = [], {}
        for s, res in zip(live, results):
            if res is None:
                continue
            c = bots[s].commands(sts[s], res["order"], res["tgt"], res["bx"], res["by"])
            spans[s] = (len(cmds), len(cmds) + len(c))
            cmds += c
            if s in trajs:
                trajs[s].add(sts[s], res)
        obs = g.step(cmds)
        for s, (a, b) in spans.items():
            bots[s].accepted(cmds[a:b], obs.command_results[a:b])
        t += 1
        if obs.game_over or t >= cfg["max_steps"]:
            break
    outcome = {}
    for s in (0, 1):
        r = obs.players.get(s)
        res = r.result.name if r is not None else "TIE"
        outcome[s] = 1.0 if res == "VICTORY" else -1.0 if res == "DEFEAT" else 0.0
    for s, tr in trajs.items():
        tr.end(outcome[s])
    me, other = obs.players.get(side), obs.players.get(1 - side)
    ep = {
        "time": time.time(), "worker": wid, "opponent": opp["name"], "outcome": outcome[side],
        "return": outcome[side], "length": t, "game_time": obs.game_time, "wall_seconds": round(time.time() - t0, 1),
        "races": races, "side": side, "race": races[side], "opponent_race": races[1 - side],
        "gold": me.gold_gathered if me else 0, "opponent_gold": other.gold_gathered if other else 0,
        "orders": bots[side].issued if side in bots else 0}
    out_q.put({"episode": ep})
    return ep


def film_game(g, ep: dict, cfg: dict) -> None:
    """Save the game's replay and render it to a video in the background (runs/<name>/videos,
    listed in media.jsonl for the dashboard)."""
    run_dir = Path(cfg["run_dir"])
    stem = f"game-{int(ep['time'])}"
    try:
        replay = g.save_replay(run_dir / "replays" / f"{stem}.w3g")
    except Exception as e:  # noqa: BLE001
        print(f"video: replay not saved: {e}", flush=True)
        return
    setup = g.setup

    def render() -> None:
        from ..video import render_replay
        try:
            out = render_replay(setup, replay, run_dir / "videos" / f"{stem}.mp4", name=f"fgvid{cfg['slot']}",
                                fit_all=True, max_steps=cfg["max_steps"] + 40)
            with open(run_dir / "media.jsonl", "a") as f:
                f.write(json.dumps({"time": time.time(), "kind": "video", "file": str(out.relative_to(run_dir)),
                                    "episode": f"{ep['race']} vs {ep['opponent_race']} ({ep['opponent']})",
                                    "outcome": ep["outcome"], "return": ep["return"], "opponent": ep["opponent"],
                                    "game_time": ep["game_time"]}) + "\n")
        except Exception as e:  # noqa: BLE001 (one video fewer)
            print(f"video: {stem} not rendered: {e}", flush=True)
    threading.Thread(target=render, daemon=True).start()


def _exit(*_):
    raise SystemExit(0)


def actor_main(wid: int, cfg: dict, out_q, stop) -> None:
    import signal

    from ..runtime import reaper
    signal.signal(signal.SIGTERM, _exit)
    torch.set_num_threads(1)
    try:
        device = torch.device(cfg["device"])
        nets = Nets(cfg, device)
        while nets.current is None and not stop.is_set():
            time.sleep(1)
            nets.reload()
        infer = Inference(nets, device)
        threads = [threading.Thread(target=game_loop, args=(wid, k, cfg, infer, out_q, stop), daemon=True)
                   for k in range(cfg["games_per_actor"])]
        for th in threads:
            th.start()
            time.sleep(2)  # launches spread out
        while not stop.is_set():
            time.sleep(1)
    finally:
        reaper.reap()  # the games (multiprocessing children skip atexit)


# ---- the league -----------------------------------------------------------------------------------

class League:
    def __init__(self, run_dir: Path, ai: list[str], shares: dict, max_past: int, pfsp: str):
        self.run_dir, self.shares, self.max_past, self.pfsp = run_dir, shares, max_past, pfsp
        self.scripts = {f"script:ai-{d}": Member(f"script:ai-{d}") for d in ai}
        self.past: list[Member] = []
        self.self_member = Member("self")

    def member(self, name: str) -> Member | None:
        if name == "self":
            return self.self_member
        return self.scripts.get(name) or next((m for m in self.past if m.name == name), None)

    def add_snapshot(self, path: Path, steps: int) -> None:
        self.past.append(Member(f"past:{steps}", path=str(path), steps=steps))
        if len(self.past) > self.max_past:  # keep the first (the clone) and the newest
            del self.past[1]

    def spec(self) -> dict:
        """What the actors draw from: per launch built-in AI or agents; per agent game an opponent."""
        ai_share = self.shares["ai"] if self.scripts else 0.0
        launch = [{"kind": "agents", "p": 1.0 - ai_share}]
        launch += [{"kind": "ai", "difficulty": n.split("-", 1)[1], "p": ai_share / len(self.scripts)}
                   for n in self.scripts]
        agents = [{"name": "self", "kind": "self", "p": self.shares["self"] if self.past else 1.0}]
        if self.past:
            w = [pfsp_weight(m.win_rate(), self.pfsp) for m in self.past]
            total = sum(w) or 1.0
            agents += [{"name": m.name, "kind": "past", "path": m.path, "p": self.shares["past"] * x / total}
                       for m, x in zip(self.past, w)]
        return {"launch": launch, "agents": agents}

    def write(self) -> None:
        spec = self.run_dir / "league_spec.json"
        tmp = spec.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.spec()))
        tmp.replace(spec)

        def row(m: Member) -> dict:
            p = m.win_rate()
            return {"name": m.name, "steps": m.steps, "games": m.games, "wins": m.wins, "losses": m.losses,
                    "draws": m.draws, "win_rate": p, "weight": pfsp_weight(p, self.pfsp) if m.path else None,
                    "path": m.path, "recent": m.recent}
        summary = {"pfsp": self.pfsp, "members": [row(m) for m in [*self.scripts.values(), *self.past]],
                   "self": row(self.self_member)}
        tmp = self.run_dir / "league.json.tmp"
        tmp.write_text(json.dumps(summary))
        tmp.replace(self.run_dir / "league.json")

    def train_keys(self) -> dict:
        out = {f"league/{n}": m.win_rate() for n, m in self.scripts.items() if m.win_rate() is not None}
        past = [x for m in self.past for x in m.recent[-50:]]
        if past:
            out["league/past"] = sum(past) / len(past)
        if self.self_member.win_rate() is not None:
            out["league/self"] = self.self_member.win_rate()
        out["league/members"] = len(self.past)
        return out


# ---- the learner ----------------------------------------------------------------------------------

def collate(steps: list[dict], device) -> dict:
    B, E = len(steps), max(s["n"] for s in steps)
    O = min(fx.MAX_OWN, E)
    G = len(steps[0]["glob"])
    ent = np.zeros((B, E, fx.F), np.float32)
    typ, cur = np.zeros((B, E), np.int64), np.zeros((B, E), np.int64)
    mask = np.zeros((B, E), bool)
    glob = np.zeros((B, G), np.float32)
    n_own = np.zeros(B, np.int64)
    acts = {k: np.zeros((B, O), np.int64) for k in ("order", "tgt", "bx", "by")}
    logp = np.zeros((B, O), np.float32)
    for i, s in enumerate(steps):
        n, o = s["n"], s["n_own"]
        ent[i, :n], typ[i, :n], cur[i, :n], mask[i, :n], glob[i], n_own[i] = s["ent"], s["type"], s["cur"], True, s["glob"], o
        for k in acts:
            acts[k][i, :o] = s[k]
        logp[i, :o] = s["logp"]
    t = lambda a: torch.from_numpy(a).to(device, non_blocking=True)  # noqa: E731
    return {"ent": t(ent), "type": t(typ), "cur": t(cur), "mask": t(mask), "glob": t(glob), "n_own": t(n_own),
            **{k: t(v) for k, v in acts.items()}, "logp": t(logp),
            "adv": torch.tensor([s["adv"] for s in steps], device=device, dtype=torch.float32),
            "ret": torch.tensor([s["ret"] for s in steps], device=device, dtype=torch.float32),
            "own": torch.arange(O, device=device)[None] < t(n_own)[:, None]}


def ppo_update(net: FullGameNet, ref: FullGameNet | None, opt, steps: list[dict], args, warmup: bool, device) -> dict:
    advs = np.array([s["adv"] for s in steps], np.float32)
    mean, std = float(advs.mean()), float(advs.std()) + 1e-8
    stats: dict[str, list[float]] = {}
    net.train()
    for _ in range(args.epochs):
        order = np.random.permutation(len(steps))
        for a in range(0, len(order), args.minibatch):
            mb = collate([steps[i] for i in order[a:a + args.minibatch]], device)
            ev = evaluate(net, mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"], mb["n_own"],
                          mb["order"], mb["tgt"], mb["bx"], mb["by"])
            own = mb["own"].float()
            n_units = own.sum().clamp(min=1)
            adv = ((mb["adv"] - mean) / std)[:, None]
            log_ratio = (ev["logp"] - mb["logp"]) * own
            ratio = log_ratio.exp()
            pg = -(torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv) * own).sum() / n_units
            v_loss = 0.5 * ((ev["value"] - mb["ret"]) ** 2).mean()
            entropy = (ev["entropy"] * own).sum() / n_units
            loss = args.vf_coef * v_loss - args.ent_coef * entropy
            if not warmup:
                loss = loss + pg
            ref_kl = torch.zeros((), device=device)
            if ref is not None and args.ref_kl > 0:
                with torch.no_grad():
                    ev_ref = evaluate(ref, mb["ent"], mb["type"], mb["cur"], mb["mask"], mb["glob"], mb["n_own"],
                                      mb["order"], mb["tgt"], mb["bx"], mb["by"])
                lp, lp_ref = torch.log_softmax(ev["logits"].float(), -1), torch.log_softmax(ev_ref["logits"].float(), -1)
                kl = (lp.exp() * (lp - lp_ref)).sum(-1)
                ref_kl = (kl * own).sum() / n_units
                if not warmup:
                    loss = loss + args.ref_kl * ref_kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            opt.step()
            with torch.no_grad():
                approx_kl = (((ratio - 1) - log_ratio) * own).sum() / n_units
                clipfrac = (((ratio - 1).abs() > args.clip).float() * own).sum() / n_units
            for k, v in (("loss/policy", pg), ("loss/value", v_loss), ("loss/entropy", entropy), ("loss/kl", approx_kl),
                         ("loss/clipfrac", clipfrac), ("loss/ref_kl", ref_kl), ("grad_norm", gn)):
                stats.setdefault(k, []).append(float(v))
    net.eval()
    return {k: sum(v) / len(v) for k, v in stats.items()}


def publish(net: FullGameNet, version: int, run_dir: Path) -> None:
    tmp = run_dir / "current.pt.tmp"
    torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "config": net.config,
                "version": version}, tmp)
    tmp.replace(run_dir / "current.pt")


def git_info() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        subject = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "warcraftsim"], cwd=root, capture_output=True,
                                    text=True, timeout=10).stdout.strip())
        return {"commit": commit, "subject": subject, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", type=Path, required=True, help="a behavior-cloned policy (fullgame/bc.py) or checkpoint")
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--timesteps", type=float, default=50e6, help="agent steps to train on")
    ap.add_argument("--actors", type=int, default=6)
    ap.add_argument("--games-per-actor", type=int, default=4)
    ap.add_argument("--games-per-process", type=int, default=6, help="games of one setup per launch")
    ap.add_argument("--scripted-reset", type=int, default=1, help="games without the built-in AI restart by script")
    ap.add_argument("--agent-games-factor", type=int, default=4, help="those run this many times as many games per launch")
    ap.add_argument("--map", default="duelrush")
    ap.add_argument("--races", default="all")
    ap.add_argument("--mirror", type=int, default=1, help="both sides play the same race (on duelrush the races "
                                                           "are far from balanced: night elf beat the others 96%%)")
    ap.add_argument("--handicap", type=int, default=50)
    ap.add_argument("--step-seconds", type=float, default=0.5)
    ap.add_argument("--max-minutes", type=float, default=4.0)
    ap.add_argument("--wait-floor-ms", type=int, default=5)
    ap.add_argument("--ai", default="normal", help="built-in AI anchors: difficulties (comma-separated; '' for none)")
    ap.add_argument("--ai-share", type=float, default=0.25, help="share of launches against the built-in AI")
    ap.add_argument("--self-share", type=float, default=0.5, help="of the agent games: against itself")
    ap.add_argument("--snapshot-every", type=int, default=20, help="updates between league snapshots")
    ap.add_argument("--max-past", type=int, default=12)
    ap.add_argument("--pfsp", default="hard")
    ap.add_argument("--batch-steps", type=int, default=8192, help="agent steps per update")
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--ref-kl", type=float, default=0.05, help="the KL term towards the initial (cloned) policy")
    ap.add_argument("--value-warmup", type=int, default=10, help="first updates: the value head only")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=64, help="steps per trajectory piece an actor sends")
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--video-every", type=float, default=15.0, help="minutes between game videos (0: none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", help="default: cuda if available (asking opens the GPU driver)")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = args.runs / args.name
    for sub_dir in ("checkpoints", "replays", "videos"):
        (run_dir / sub_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)  # the games need the cores
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net, ck = load(args.init, device)
    vocab = ck["vocab"]
    (run_dir / "vocab.json").write_text(json.dumps(vocab))
    ref = copy.deepcopy(net).eval() if args.ref_kl > 0 else None
    if ref is not None:
        for p in ref.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    slot, slot_lock = claim_slot(args.runs)
    races = list(fx.RACES) if args.races == "all" else args.races.split(",")
    ai = [d for d in args.ai.split(",") if d]
    init_rel = os.path.relpath(args.init.resolve(), args.runs.resolve().parent)
    info = {"name": args.name, "kind": "run", "trainer": "fullgame", "task": f"fullgame_{args.map}",
            "status": "launching", "created": time.time(), "timesteps": int(args.timesteps),
            "envs": args.actors * args.games_per_actor, "init_from": init_rel, "init_steps": 0,
            "description": (f"Self-play on whole games ({args.map}) from a behavior-cloned policy: PPO per unit, "
                            f"a league of itself, past snapshots and the built-in AI ({', '.join(ai) or 'none'})"),
            "config": {k: str(v) for k, v in vars(args).items()},
            "launch": {"command": "python3 -m warcraftsim.fullgame.selfplay " + " ".join(sys.argv[1:] if argv is None else argv),
                       "git": git_info()}}

    def save_info():
        tmp = run_dir / "run.json.tmp"
        tmp.write_text(json.dumps(info, indent=1))
        tmp.replace(run_dir / "run.json")
    save_info()
    if args.note:
        (run_dir / "notes.md").write_text(args.note + "\n")

    league = League(run_dir, ai, {"ai": args.ai_share, "self": args.self_share, "past": 1.0 - args.self_share},
                    args.max_past, args.pfsp)
    first = run_dir / "checkpoints" / f"{0:016d}.pt"
    torch.save({**ck, "model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": 0}, first)
    league.add_snapshot(first, 0)  # the clone itself: the first past opponent
    league.write()
    publish(net, 0, run_dir)
    cfg = {"run_dir": str(run_dir), "device": str(device), "vocab": str(run_dir / "vocab.json"), "races": races,
           "map": args.map, "handicap": args.handicap, "step_seconds": args.step_seconds,
           "max_minutes": args.max_minutes, "max_steps": int(args.max_minutes * 60 / args.step_seconds) + 20,
           "wait_floor_ms": args.wait_floor_ms, "games_per_actor": args.games_per_actor,
           "games_per_process": args.games_per_process, "chunk": args.chunk, "gamma": args.gamma, "lam": args.lam,
           "seed": args.seed, "slot": slot, "video_every": args.video_every, "scripted_reset": bool(args.scripted_reset),
           "agent_games_factor": args.agent_games_factor, "mirror": bool(args.mirror)}
    ctx = torch.multiprocessing.get_context("spawn")
    out_q, stop = ctx.Queue(maxsize=4096), ctx.Event()
    actors = [ctx.Process(target=actor_main, args=(w, cfg, out_q, stop), daemon=True) for w in range(args.actors)]
    for p in actors:
        p.start()
    info.update(status="training", started=time.time())
    save_info()
    t0 = time.time()
    agent_steps, update, episodes = 0, 0, 0
    buf: list[dict] = []
    stale = []
    try:
        while agent_steps < args.timesteps:
            t_collect = time.time()
            while len(buf) < args.batch_steps:
                try:
                    msg = out_q.get(timeout=600)
                except queue.Empty:
                    raise RuntimeError("no trajectories from the actors for 10 minutes")
                if "episode" in msg:
                    e = msg["episode"]
                    episodes += 1
                    e["episode"] = episodes
                    m = league.member(e["opponent"])
                    if m is not None:
                        m.record(e["outcome"])
                    with open(run_dir / "episodes.jsonl", "a") as f:
                        f.write(json.dumps(e) + "\n")
                    continue
                buf += msg["steps"]
                stale += [update - s["version"] for s in msg["steps"] if s["version"] >= 0]
            t_train = time.time()
            steps, buf = buf, []
            warmup = update < args.value_warmup
            stats = ppo_update(net, ref, opt, steps, args, warmup, device)
            update += 1
            agent_steps += len(steps)
            publish(net, update, run_dir)
            if update % args.snapshot_every == 0:
                path = run_dir / "checkpoints" / f"{agent_steps:016d}.pt"
                torch.save({"model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": agent_steps,
                            "update": update}, path)
                league.add_snapshot(path, agent_steps)
            elif update % args.checkpoint_every == 0:
                torch.save({"model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": agent_steps,
                            "update": update}, run_dir / "checkpoints" / f"{agent_steps:016d}.pt")
            league.write()
            now = time.time()
            row = {"agent_steps": agent_steps, "epoch": update, "time": now, "uptime": now - t0,
                   "SPS": len(steps) / max(now - t_collect, 1e-9), "lr": args.lr, "value_warmup": warmup,
                   "perf/rollout": t_train - t_collect, "perf/train": now - t_train,
                   "staleness": sum(stale) / len(stale) if stale else 0.0, **stats, **league.train_keys()}
            stale = []
            with open(run_dir / "train.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"update {update}: {agent_steps} steps, {row['SPS']:.0f} steps/s, {episodes} games; "
                  f"policy {stats.get('loss/policy', 0):.4f} value {stats.get('loss/value', 0):.4f} "
                  f"kl {stats.get('loss/kl', 0):.4f} ref_kl {stats.get('loss/ref_kl', 0):.4f} "
                  f"{json.dumps(league.train_keys())}", flush=True)
        info["status"] = "finished"
    except KeyboardInterrupt:
        info["status"] = "stopped"
    except Exception as e:  # noqa: BLE001
        info["status"] = f"failed: {type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        info["finished"] = time.time()
        save_info()
        torch.save({"model": net.state_dict(), "config": net.config, "vocab": vocab, "agent_steps": agent_steps,
                    "update": update}, run_dir / "checkpoints" / f"{agent_steps:016d}.pt")
        stop.set()
        for p in actors:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
