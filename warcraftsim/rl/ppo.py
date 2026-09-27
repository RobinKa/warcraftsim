"""PPO for EntityNet on the bridge's games (run as a script by train.py --trainer torch).

    python warcraftsim/rl/ppo.py --spec runs/NAME/spec.json --run-dir runs/NAME --envs 24 --timesteps 3e6

$WC3_BRIDGE lists the bridge workers' sockets (';'-separated), as for the PufferLib trainer. Each
epoch collects `horizon` steps from every environment, computes GAE, and updates each learner on
sequence chunks (the core state at each chunk's start comes from the rollout). Advantages are
normalized per minibatch. Logs one JSON line per epoch to $PUFFER_JSONL (the dashboard's
train.jsonl keys) and saves checkpoints to <run-dir>/checkpoints/<steps>.pt.

Self-play tasks (2 agents per game, `_self`) have a league (league.py): the main learner plays side 0
of every game but the exploiter's; side 1 is itself (both sides train it), a past snapshot (PFSP) or a
script. With --exploiters 1 some games belong to a main exploiter (as in AlphaStar): a second learner
that plays only against the main learner's current policy, joins the league as a snapshot every
--snapshot-every epochs, and restarts from the initial policy once it beats the main learner often.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import BridgeEnvs  # noqa: E402
from league import League, Member, Scripts  # noqa: E402
from model import KINDS, EntityNet  # noqa: E402

STOP = False


def _stop(signum, frame):
    global STOP
    STOP = True


def gpu_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"util/vram_used_gb": (total - free) / 2 ** 30, "util/vram_total_gb": total / 2 ** 30}


def load_checkpoint(path: str | Path, spec: dict, device) -> tuple[EntityNet, dict]:
    """A checkpoint's network for `spec`. When the task's unit features differ from the ones it
    was trained with (e.g. a task with abilities after one without), the unit encoder's input
    weights are matched by feature name: features it never saw start at zero weight, so the
    network acts exactly as before until training finds a use for them."""
    ck = torch.load(path, map_location=device, weights_only=False)
    net = EntityNet(spec, **ck.get("config", {})).to(device)
    state = dict(ck["model"])
    old = (ck.get("spec") or {}).get("spaces", {}).get("observation", {}).get("blocks", [{}])[0].get("features")
    if old is not None and list(old) != list(net.feat):
        missing = [f for f in old if f not in net.feat]
        if missing:
            raise ValueError(f"{path}: features the task no longer has: {missing[:5]}")
        w_old = state["unit_mlp.0.weight"]
        w = torch.zeros(w_old.shape[0], net.F, dtype=w_old.dtype, device=w_old.device)
        for j, f in enumerate(old):
            w[:, net.feat.index(f)] = w_old[:, j]
        state["unit_mlp.0.weight"] = w
        print(f"{path}: unit features {len(old)} -> {net.F} (new ones start at zero weight)", flush=True)
    net.load_state_dict(state)
    return net, ck


def _finished() -> dict:
    return {"n": 0, "wins": 0, "losses": 0, "ret": 0.0, "len": 0.0}


class Learner:
    """A policy being trained: its network and optimizer, the rows it plays (environment e, agent a
    -> row e * agents + a), and its rollout buffers."""

    def __init__(self, name: str, net: EntityNet, rows: np.ndarray, args, device, ref: EntityNet | None = None):
        self.name, self.net, self.rows, self.args, self.device, self.ref = name, net, rows, args, device, ref
        self.B = len(rows)
        self.opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
        # launch overhead dominates a step: CUDA graphs (~2 ms instead of ~20)
        self.step_fn = (torch.compile(net.step, mode="reduce-overhead")
                        if device.type == "cuda" and not args.no_compile else net.step)
        self.h = net.initial_state(self.B, device)
        self.ref_h = net.initial_state(self.B, device)
        self.start = torch.ones(self.B, device=device)  # the next observation begins an episode
        self.epoch = 0
        self.finished = _finished()

    def observe(self, obs: np.ndarray, masks: np.ndarray) -> None:
        self.obs_t = torch.as_tensor(obs[self.rows], device=self.device)
        self.masks_t = torch.as_tensor(masks[self.rows], device=self.device)

    def begin(self, T: int) -> None:
        B, dev, k = self.B, self.device, self.net.k
        self.b_obs = torch.zeros(T, B, self.obs_t.shape[1], device=dev)
        self.b_masks = torch.zeros(T, B, self.masks_t.shape[1], dtype=torch.uint8, device=dev)
        self.b_act = torch.zeros(T, B, k, 5, dtype=torch.long, device=dev)
        self.b_logp, self.b_val, self.b_rew, self.b_done, self.b_start, self.b_ref = (
            torch.zeros(T, B, device=dev) for _ in range(6))
        self.b_h = torch.zeros(T, B, self.net.core_size, device=dev)

    def act(self, t: int) -> np.ndarray:
        with torch.no_grad():
            self.h = self.h * (1 - self.start).unsqueeze(-1)
            self.b_h[t] = self.h
            acts, logp, _, v, h = self.step_fn(self.obs_t, self.h, self.masks_t)
            # a compiled graph reuses its output buffers on the next call: keep copies
            acts, logp, v, self.h = acts.clone(), logp.clone(), v.clone(), h.clone()
            if self.ref is not None:  # the reference's log-probability of the same orders, with its own memory
                self.ref_h = self.ref_h * (1 - self.start).unsqueeze(-1)
                u_r, x_r = self.ref.encode(self.obs_t)
                self.ref_h = self.ref.gru(x_r, self.ref_h)
                self.b_ref[t] = self.ref.heads(self.obs_t, u_r, self.ref_h, self.masks_t, actions=acts)[1]
        self.b_obs[t], self.b_masks[t], self.b_act[t] = self.obs_t, self.masks_t, acts
        self.b_logp[t], self.b_val[t], self.b_start[t] = logp, v, self.start
        return acts.view(self.B, -1).cpu().numpy()

    def after_step(self, t: int, obs, masks, rew, term, reward_scale: float) -> None:
        self.observe(obs, masks)
        self.b_rew[t] = torch.as_tensor(rew[self.rows], device=self.device) * reward_scale
        self.b_done[t] = torch.as_tensor(term[self.rows], device=self.device)
        self.start = self.b_done[t].clone()

    def update(self, T: int, progress: float) -> dict:
        """GAE and PPO epochs on the rollout; progress (0..1) anneals the learning rate."""
        args, net, dev = self.args, self.net, self.device
        with torch.no_grad():
            _, x = net.encode(self.obs_t)
            last_v = net.value(net.gru(x, self.h * (1 - self.start).unsqueeze(-1))).squeeze(-1)
        adv = torch.zeros(T, self.B, device=dev)
        gae = torch.zeros(self.B, device=dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else self.b_val[t + 1]
            nonterm = 1 - self.b_done[t]
            delta = self.b_rew[t] + args.gamma * nv * nonterm - self.b_val[t]
            gae = delta + args.gamma * args.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + self.b_val
        t1 = time.time()
        lr = args.lr * max(0.0, 1 - progress)
        for g in self.opt.param_groups:
            g["lr"] = lr
        L = args.chunk
        chunks = [(t, b) for t in range(0, T, L) for b in range(self.B)]
        per_mb = max(1, args.minibatch // L)
        acc = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0, "ref_kl": 0.0, "n": 0}
        warm = self.epoch < args.vf_warmup
        stop_kl = False
        for _ in range(0 if args.eval_only else args.epochs):
            if stop_kl:
                break
            order = np.random.permutation(len(chunks))
            for i in range(0, len(order), per_mb):
                sel = [chunks[j] for j in order[i:i + per_mb]]
                ts = torch.tensor([c[0] for c in sel], device=dev)
                bs = torch.tensor([c[1] for c in sel], device=dev)
                it = ts.unsqueeze(0) + torch.arange(L, device=dev).unsqueeze(1)  # [L, M]
                ib = bs.unsqueeze(0).expand(L, -1)
                logp, ent, v = net.evaluate(self.b_obs[it, ib], self.b_masks[it, ib], self.b_act[it, ib],
                                            self.b_h[ts, bs], self.b_start[it, ib])
                old = self.b_logp[it, ib]
                a = adv[it, ib]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - old).exp()
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vl = 0.5 * ((v - ret[it, ib]) ** 2).mean()
                el = ent.mean()
                loss = args.vf_coef * vl if warm else pg + args.vf_coef * vl - args.ent_coef * el
                if self.ref is not None and not warm:  # KL(policy || reference) on the rollout's actions (k3)
                    lr_ = self.b_ref[it, ib] - logp
                    ref_kl = (lr_.exp() - 1 - lr_).mean()
                    loss = loss + args.ref_kl * ref_kl
                    acc["ref_kl"] += ref_kl.item()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - old)).mean().item()
                    acc["policy"] += pg.item()
                    acc["value"] += vl.item()
                    acc["entropy"] += el.item()
                    acc["kl"] += kl
                    acc["clipfrac"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                    acc["n"] += 1
                if args.target_kl and not warm and kl > args.target_kl:
                    stop_kl = True
                    break
        self.epoch += 1
        n = max(acc.pop("n"), 1)
        out = {f"loss/{k}": v / n for k, v in acc.items() if k != "ref_kl" or self.ref is not None}
        out.update({"lr": lr, "perf/train": time.time() - t1})
        return out

    def win_stats(self) -> dict:
        f, self.finished = self.finished, _finished()
        if not f["n"]:
            return {}
        return {"env/win_rate": f["wins"] / f["n"], "env/loss_rate": f["losses"] / f["n"],
                "env/episode_return": f["ret"] / f["n"], "env/episode_length": f["len"] / f["n"], "env/n": f["n"]}

    def save(self, path: Path, spec: dict, steps: int) -> None:
        torch.save({"model": self.net.state_dict(), "config": self.net.config, "steps": steps, "spec": spec,
                    "args": vars(self.args), "learner": self.name}, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--timesteps", type=float, default=3e6, help="the main learner's steps")
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16, help="sequence length of an update's samples")
    ap.add_argument("--minibatch", type=int, default=512, help="samples per update (chunks x chunk length)")
    ap.add_argument("--epochs", type=int, default=4, help="passes over each rollout")
    ap.add_argument("--reward-scale", type=float, default=0.0,
                    help="multiplies the bridge's rewards (default: undo the task's reward_scale: that is for "
                         "PufferLib, which doesn't normalize advantages; large returns destabilize a shared trunk)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--core", type=int, default=256)
    ap.add_argument("--init-from", help="a checkpoint (.pt) to start from")
    ap.add_argument("--cast-bias", type=float, default=0.0,
                    help="added to the cast order's logit at the start: a checkpoint trained without "
                         "abilities never had cast possible, and would almost never try it")
    ap.add_argument("--checkpoint-interval", type=int, default=20, help="epochs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-share", type=float, default=0.5, help="self-play: share of games against itself")
    ap.add_argument("--script-share", type=float, default=0.25, help="self-play: share against scripts")
    ap.add_argument("--scripts", default="noop,focus,pull35,amove", help="self-play: the scripted opponents")
    ap.add_argument("--pfsp", default="hard", choices=("hard", "variance", "uniform"),
                    help="self-play: how past snapshots are chosen (by the learner's win rate p against them: "
                         "hard (1-p)^2, variance p(1-p), uniform)")
    ap.add_argument("--snapshot-every", type=int, default=20, help="self-play: epochs between league snapshots")
    ap.add_argument("--exploiters", type=int, default=0, help="self-play: 1 = a main exploiter (AlphaStar)")
    ap.add_argument("--exploit-share", type=float, default=0.25, help="self-play: the exploiter's share of games")
    ap.add_argument("--exploiter-reset", type=float, default=0.7,
                    help="the exploiter restarts from the initial policy once it wins this often against the "
                         "main learner (its last 100 games; it joins the league first)")
    ap.add_argument("--ref", help="a reference policy (.pt, e.g. a fitted script) to stay near: as AlphaStar's KL "
                                  "to its supervised policy, which keeps what the demonstrations knew while RL explores")
    ap.add_argument("--ref-kl", type=float, default=0.0, help="weight of the KL to --ref")
    ap.add_argument("--vf-warmup", type=int, default=0,
                    help="epochs that train only the value (a cloned policy's value is barely trained: its first "
                         "advantages are noise, and PPO would follow them away from the clone)")
    ap.add_argument("--target-kl", type=float, default=0.0,
                    help="stop an epoch's updates once the policy moved this far (approximate KL; 0: off)")
    ap.add_argument("--eval-only", type=int, default=0, help="1: play without updating (win rates in train.jsonl)")
    ap.add_argument("--no-compile", type=int, default=0,
                    help="skip torch.compile of the rollout step (CUDA graphs: ~2 ms instead of ~20 per step)")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = json.loads(Path(args.spec).read_text())
    run_dir = Path(args.run_dir)
    ck_dir = run_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    log = open(os.environ.get("PUFFER_JSONL", run_dir / "train.jsonl"), "a")

    def new_net() -> EntityNet:
        n = (load_checkpoint(args.init_from, spec, device)[0] if args.init_from
             else EntityNet(spec, d=args.d, layers=args.layers, core=args.core).to(device))
        if args.cast_bias:
            with torch.no_grad():
                n.kind.bias[KINDS.index("cast")] += args.cast_bias
        return n

    net = new_net()
    if args.init_from:
        print(f"init from {args.init_from}", flush=True)
    ref = load_checkpoint(args.ref, spec, device)[0].eval() if args.ref and args.ref_kl > 0 else None
    if ref is not None:
        for p_ in ref.parameters():
            p_.requires_grad_(False)
    print(f"EntityNet: {sum(p.numel() for p in net.parameters()) / 1e3:.0f}k parameters on {device}", flush=True)

    sockets = os.environ["WC3_BRIDGE"].split(";")
    envs = BridgeEnvs(sockets, args.envs, spec["task"], spec["obs_size"], spec["num_atns"])
    n_env, A = len(envs.conns), envs.agents
    reward_scale = args.reward_scale or 1.0 / spec.get("reward_scale", 1.0)

    # ---- who plays which seat -------------------------------------------------------------------
    league = scripts = None
    groups = ["solo"] * n_env
    if A == 2:
        league = League(run_dir, [s_ for s_ in args.scripts.split(",") if s_], args.pfsp)
        scripts = Scripts(spec)
        n_exp = round(n_env * args.exploit_share) if args.exploiters else 0
        n_self = round(n_env * args.self_share)
        n_script = round(n_env * args.script_share) if league.scripts else 0
        n_past = n_env - n_self - n_script - n_exp
        assert n_past >= 0, "the shares add up to more than all games"
        groups = ["self"] * n_self + ["past"] * n_past + ["script"] * n_script + ["exploit"] * n_exp
        first = ck_dir / f"{0:016d}.pt"  # the start is the league's first member
        torch.save({"model": net.state_dict(), "config": net.config, "steps": 0, "spec": spec, "args": vars(args)}, first)
        league.add_snapshot(str(first), 0)
        print("league: " + ", ".join(f"{g} {groups.count(g)}" for g in ("self", "past", "script", "exploit")) + " games",
              flush=True)
    elif A != 1:
        raise SystemExit(f"{A} agents per game: not supported")
    main_rows = np.array([e * A for e in range(n_env) if groups[e] != "exploit"] +
                         [e * A + 1 for e in range(n_env) if groups[e] == "self"])
    main = Learner("main", net, main_rows, args, device, ref)
    learners = [main]
    exploiter = None
    if "exploit" in groups:
        exploiter = Learner("exploiter", new_net(), np.array([e * A for e in range(n_env) if groups[e] == "exploit"]),
                            args, device)
        learners.append(exploiter)
    init_state = {k_: v_.clone() for k_, v_ in net.state_dict().items()}
    exploiter_member = Member("exploiter")  # the current exploiter's record against the main learner
    exploiter_resets = 0
    # side 1 of past / script games (league members) and of exploiter games (the main learner's policy)
    opp_rows = np.array([e * A + 1 for e in range(n_env) if groups[e] in ("past", "script", "exploit")], dtype=np.int64)
    opponent = {e: (league.sample_past() if g == "past" else league.sample_script() if g == "script" else None)
                for e, g in enumerate(groups) if g in ("past", "script", "exploit")}
    opp_nets: dict[str, EntityNet] = {}

    def publish_seats() -> None:
        """Who plays side 1 of game 0, the one the bridge records (its videos name both sides)."""
        if league is None:
            return
        g, m = groups[0], opponent.get(0)
        seat = ({"name": "itself", "kind": "self"} if g == "self" else
                {"name": "the exploiter", "kind": "self"} if g == "exploit" else
                {"name": m.name, "kind": "script", "script": m.name.split(":", 1)[1]} if m.path is None else
                {"name": m.name, "kind": "past", "path": m.path})
        tmp = run_dir / "seats.json.tmp"
        tmp.write_text(json.dumps({"0": {"learner": run_dir.name, "opponent": seat}}))
        tmp.replace(run_dir / "seats.json")

    publish_seats()
    opp_h = net.initial_state(len(opp_rows), device)
    opp_start = torch.ones(len(opp_rows), device=device)

    def opponent_actions(obs, masks) -> np.ndarray:
        nonlocal opp_h
        out = np.zeros((len(opp_rows), spec["num_atns"]), np.int64)
        if not len(opp_rows):
            return out
        opp_h = opp_h * (1 - opp_start).unsqueeze(-1)
        by: dict[str, list[int]] = {}
        for i, r in enumerate(opp_rows):
            m = opponent[int(r) // A]
            by.setdefault(m.name if m is not None else "main", []).append(i)
        for name, idx in by.items():
            rows = opp_rows[idx]
            m = opponent[int(rows[0]) // A]
            if m is not None and m.path is None:  # a script
                out[idx] = scripts.act(name.split(":", 1)[1], obs[rows])
                continue
            if m is None:
                player = main.net  # exploiter games: the main learner's current policy
            else:
                if name not in opp_nets:
                    opp_nets[name] = load_checkpoint(m.path, spec, device)[0].eval()
                player = opp_nets[name]
            with torch.no_grad():
                a, _, _, _, h_new = player.step(torch.as_tensor(obs[rows], device=device), opp_h[idx],
                                                torch.as_tensor(masks[rows], device=device))
            opp_h[idx] = h_new
            out[idx] = a.view(len(idx), -1).cpu().numpy()
        return out

    obs, masks = envs.reset()
    for L_ in learners:
        L_.observe(obs, masks)
    T = args.horizon
    total = int(args.timesteps)
    steps, epoch, t_begin = 0, 0, time.time()
    league_results: dict[str, list] = {}

    while steps < total and not STOP:
        t0 = time.time()
        for L_ in learners:
            L_.begin(T)
        t_env = t_model = 0.0
        for t in range(T):
            tm = time.time()
            flat = np.zeros((envs.n, spec["num_atns"]), np.int64)
            for L_ in learners:
                flat[L_.rows] = L_.act(t)
            if len(opp_rows):
                flat[opp_rows] = opponent_actions(obs, masks)
            te = time.time()
            t_model += te - tm
            obs, masks, rew, term, stats = envs.step(flat)
            t_env += time.time() - te
            for L_ in learners:
                L_.after_step(t, obs, masks, rew, term, reward_scale)
            if len(opp_rows):
                opp_start = torch.as_tensor(term[opp_rows], device=device)
            for e in range(n_env):
                s_ = stats[e * A]
                if s_[0] < 0.5:
                    continue
                outcome = float(s_[3])
                f = (exploiter if groups[e] == "exploit" else main).finished
                f["n"] += 1
                f["wins"] += outcome > 0.5
                f["losses"] += outcome < -0.5
                f["ret"] += float(s_[1])
                f["len"] += float(s_[2])
                if league is None:
                    continue
                if groups[e] == "exploit":  # the exploiter's result against the main learner
                    exploiter_member.record(outcome)
                    league_results.setdefault("exploiter_vs_main", []).append(outcome)
                    continue
                m = league.self_member if groups[e] == "self" else opponent[e]
                m.record(outcome)
                key = "self" if groups[e] == "self" else m.name.split(":")[0] if m.path else m.name
                league_results.setdefault(key, []).append(outcome)
                if groups[e] == "past":
                    opponent[e] = league.sample_past()
                elif groups[e] == "script":
                    opponent[e] = league.sample_script()
                if e == 0:
                    publish_seats()
        steps += main.B * T
        t_rollout = time.time() - t0

        row = {"agent_steps": steps, "epoch": epoch + 1, "uptime": time.time() - t_begin}
        progress = epoch / max(1, total // (main.B * T))
        for L_ in learners:
            stats_ = L_.update(T, progress)
            if L_ is main:
                row.update(stats_)
                row.update(L_.win_stats())
            else:
                row.update({f"{L_.name}/{k_}": v_ for k_, v_ in stats_.items() if k_.startswith("loss/")})
                L_.win_stats()
        epoch += 1
        row["SPS"] = main.B * T / (time.time() - t0)
        row.update({"perf/rollout": t_rollout, "perf/eval_env": t_env, "perf/eval_model": t_model, "time": time.time(),
                    **gpu_stats()})
        if league is not None:  # win rates this epoch by opponent kind (draws count half)
            for key, res in league_results.items():
                row[f"league/{key}"] = sum(1.0 if r > 0.5 else 0.5 if r > -0.5 else 0.0 for r in res) / len(res)
                row[f"league/{key}_n"] = len(res)
            league_results = {}
            row["league/members"] = len(league.past)
            if exploiter is not None:
                row["league/exploiter_resets"] = exploiter_resets
        log.write(json.dumps(row) + "\n")
        log.flush()
        if epoch % 10 == 1:
            print(f"epoch {epoch} steps {steps} SPS {row['SPS']:.0f} win {row.get('env/win_rate', float('nan')):.2f} "
                  f"kl {row.get('loss/kl', 0):.4f} ent {row.get('loss/entropy', 0):.2f}", flush=True)
        snapshot = league is not None and epoch % args.snapshot_every == 0
        if epoch % args.checkpoint_interval == 0 or steps >= total or STOP or snapshot:
            path = ck_dir / f"{steps:016d}.pt"
            main.save(path, spec, steps)
            if snapshot:
                league.add_snapshot(str(path), steps)
        if exploiter is not None and snapshot:
            path = ck_dir / f"exploiter-{steps:016d}.pt"
            exploiter.save(path, spec, steps)
            league.add_snapshot(str(path), steps).name = f"exploiter:{steps}"
            p = exploiter_member.win_rate(100)
            if p is not None and len(exploiter_member.recent) >= 100 and p >= args.exploiter_reset:
                # it has found what it can: back to the start, to look for the next weakness
                exploiter.net.load_state_dict(init_state)
                exploiter.opt = torch.optim.Adam(exploiter.net.parameters(), lr=args.lr, eps=1e-5)
                exploiter.epoch = 0
                exploiter_member.recent.clear()
                exploiter_resets += 1
                print(f"exploiter reset ({exploiter_resets}) after winning {p:.0%} against the main learner", flush=True)
        if league is not None:
            league.save()
    envs.close()
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
