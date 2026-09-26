"""PPO for EntityNet on the bridge's games (run as a script by train.py --trainer torch).

    python warcraftsim/rl/ppo.py --spec runs/NAME/spec.json --run-dir runs/NAME --envs 24 --timesteps 3e6

$WC3_BRIDGE lists the bridge workers' sockets (';'-separated), as for the PufferLib trainer. Each
epoch collects `horizon` steps from every environment, computes GAE, and updates the policy on
sequence chunks (the core state at each chunk's start comes from the rollout). Advantages are
normalized per minibatch. Logs one JSON line per epoch to $PUFFER_JSONL (the dashboard's
train.jsonl keys) and saves checkpoints to <run-dir>/checkpoints/<steps>.pt.

Self-play tasks (2 agents per game, `_self`): the learner plays side 0; side 1 is played by an
opponent from the league (league.py), and when that opponent is the learner itself its experience
trains too.
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
from league import League, Scripts  # noqa: E402
from model import EntityNet  # noqa: E402

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
    ck = torch.load(path, map_location=device, weights_only=False)
    net = EntityNet(spec, **ck.get("config", {})).to(device)
    net.load_state_dict(ck["model"])
    return net, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--timesteps", type=float, default=3e6)
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
    ap.add_argument("--checkpoint-interval", type=int, default=20, help="epochs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-share", type=float, default=0.5, help="self-play: share of games against itself")
    ap.add_argument("--script-share", type=float, default=0.25, help="self-play: share against scripts")
    ap.add_argument("--scripts", default="noop,focus,pull35", help="self-play: the scripted opponents")
    ap.add_argument("--pfsp", default="hard", choices=("hard", "variance", "uniform"),
                    help="self-play: how past snapshots are chosen (by the learner's win rate p against them: "
                         "hard (1-p)^2, variance p(1-p), uniform)")
    ap.add_argument("--snapshot-every", type=int, default=20, help="self-play: epochs between league snapshots")
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

    if args.init_from:
        net, ck = load_checkpoint(args.init_from, spec, device)
        print(f"init from {args.init_from} ({ck.get('steps', 0)} steps)", flush=True)
    else:
        net = EntityNet(spec, d=args.d, layers=args.layers, core=args.core).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    ref = load_checkpoint(args.ref, spec, device)[0].eval() if args.ref and args.ref_kl > 0 else None
    if ref is not None:
        for p_ in ref.parameters():
            p_.requires_grad_(False)
    print(f"EntityNet: {sum(p.numel() for p in net.parameters()) / 1e3:.0f}k parameters on {device}", flush=True)

    sockets = os.environ["WC3_BRIDGE"].split(";")
    envs = BridgeEnvs(sockets, args.envs, spec["task"], spec["obs_size"], spec["num_atns"])
    n_env, A = len(envs.conns), envs.agents
    # which rows (environment e, agent a -> row e * A + a) the learner plays; the rest: opponents
    league = scripts = None
    groups = ["solo"] * n_env
    if A == 2:
        league = League(run_dir, [s_ for s_ in args.scripts.split(",") if s_], args.pfsp)
        scripts = Scripts(spec)
        n_self = round(n_env * args.self_share)
        n_script = round(n_env * args.script_share) if league.scripts else 0
        groups = ["self"] * n_self + ["past"] * (n_env - n_self - n_script) + ["script"] * n_script
        first = ck_dir / f"{0:016d}.pt"  # the start is the league's first member
        torch.save({"model": net.state_dict(), "config": net.config, "steps": 0, "spec": spec, "args": vars(args)}, first)
        league.add_snapshot(str(first), 0)
        print("league: " + ", ".join(f"{g} {groups.count(g)}" for g in ("self", "past", "script")) + " games", flush=True)
    elif A != 1:
        raise SystemExit(f"{A} agents per game: not supported")
    learner_rows = np.array([e * A for e in range(n_env)] + [e * A + 1 for e in range(n_env) if groups[e] == "self"])
    opp_rows = np.array([e * A + 1 for e in range(n_env) if groups[e] in ("past", "script")], dtype=np.int64)
    opponent = {e: (league.sample_past() if g == "past" else league.sample_script()) for e, g in enumerate(groups)
                if g in ("past", "script")}
    opp_nets: dict[str, EntityNet] = {}
    opp_h = net.initial_state(len(opp_rows), device)
    opp_start = torch.ones(len(opp_rows), device=device)
    opp_index = {int(r): i for i, r in enumerate(opp_rows)}
    B, T, k = len(learner_rows), args.horizon, net.k
    step_fn = net.step
    if device.type == "cuda" and not args.no_compile:  # launch overhead dominates a step: CUDA graphs
        step_fn = torch.compile(net.step, mode="reduce-overhead")
    reward_scale = args.reward_scale or 1.0 / spec.get("reward_scale", 1.0)
    obs, masks = envs.reset()
    obs_t = torch.as_tensor(obs[learner_rows], device=device)
    masks_t = torch.as_tensor(masks[learner_rows], device=device)
    league_results: dict[str, list] = {}

    def opponent_actions(obs, masks) -> np.ndarray:
        """Side 1's actions in the past and script games."""
        nonlocal opp_h
        out = np.zeros((len(opp_rows), spec["num_atns"]), np.int64)
        if not len(opp_rows):
            return out
        opp_h = opp_h * (1 - opp_start).unsqueeze(-1)
        by_member: dict[str, list[int]] = {}
        for i, r in enumerate(opp_rows):
            by_member.setdefault(opponent[int(r) // A].name, []).append(i)
        for name, idx in by_member.items():
            m = opponent[int(opp_rows[idx[0]]) // A]
            rows = opp_rows[idx]
            if m.path is None:  # a script
                out[idx] = scripts.act(name.split(":", 1)[1], obs[rows])
                continue
            if name not in opp_nets:
                opp_nets[name] = load_checkpoint(m.path, spec, device)[0].eval()
            with torch.no_grad():
                a, _, _, _, h_new = opp_nets[name].step(torch.as_tensor(obs[rows], device=device), opp_h[idx],
                                                        torch.as_tensor(masks[rows], device=device))
            opp_h[idx] = h_new
            out[idx] = a.view(len(idx), -1).cpu().numpy()
        return out
    h = net.initial_state(B, device)
    ref_h = net.initial_state(B, device)
    start = torch.ones(B, device=device)  # the next observation begins an episode
    total = int(args.timesteps)
    steps, epoch, t_begin = 0, 0, time.time()
    updates_total = max(1, total // (B * T))
    finished = {"n": 0, "wins": 0, "losses": 0, "ret": 0.0, "len": 0.0}

    while steps < total and not STOP:
        t0 = time.time()
        # ---- rollout ---------------------------------------------------------------------------
        b_obs = torch.zeros(T, B, obs_t.shape[1], device=device)
        b_masks = torch.zeros(T, B, masks_t.shape[1], dtype=torch.uint8, device=device)
        b_act = torch.zeros(T, B, k, 5, dtype=torch.long, device=device)
        b_logp, b_val, b_rew, b_done, b_start = (torch.zeros(T, B, device=device) for _ in range(5))
        b_h = torch.zeros(T, B, net.core_size, device=device)
        b_ref = torch.zeros(T, B, device=device)
        t_env = t_model = 0.0
        for t in range(T):
            tm = time.time()
            with torch.no_grad():
                h = h * (1 - start).unsqueeze(-1)
                b_h[t] = h
                acts, logp, _, v, h = step_fn(obs_t, h, masks_t)
                # a compiled graph reuses its output buffers on the next call: keep copies
                acts, logp, v, h = acts.clone(), logp.clone(), v.clone(), h.clone()
            b_obs[t], b_masks[t], b_act[t], b_logp[t], b_val[t], b_start[t] = obs_t, masks_t, acts, logp, v, start
            if ref is not None:  # the reference's log-probability of the same orders, with its own memory
                with torch.no_grad():
                    ref_h = ref_h * (1 - start).unsqueeze(-1)
                    u_r, x_r = ref.encode(obs_t)
                    ref_h = ref.gru(x_r, ref_h)
                    b_ref[t] = ref.heads(obs_t, u_r, ref_h, masks_t, actions=acts)[1]
            flat = np.zeros((envs.n, spec["num_atns"]), np.int64)
            flat[learner_rows] = acts.view(B, -1).cpu().numpy()
            if len(opp_rows):
                flat[opp_rows] = opponent_actions(obs, masks)
            te = time.time()
            t_model += te - tm
            obs, masks, rew, term, stats = envs.step(flat)
            t_env += time.time() - te
            obs_t = torch.as_tensor(obs[learner_rows], device=device)
            masks_t = torch.as_tensor(masks[learner_rows], device=device)
            b_rew[t] = torch.as_tensor(rew[learner_rows], device=device) * reward_scale
            b_done[t] = torch.as_tensor(term[learner_rows], device=device)
            start = b_done[t].clone()
            if len(opp_rows):
                opp_start = torch.as_tensor(term[opp_rows], device=device)
            for e in range(n_env):
                s_ = stats[e * A]
                if s_[0] < 0.5:
                    continue
                finished["n"] += 1
                finished["wins"] += s_[3] > 0.5
                finished["losses"] += s_[3] < -0.5
                finished["ret"] += float(s_[1])
                finished["len"] += float(s_[2])
                if league is not None:  # the learner's result against this game's opponent; the next one
                    m = league.self_member if groups[e] == "self" else opponent[e]
                    m.record(float(s_[3]))
                    key = "self" if groups[e] == "self" else m.name.split(":")[0] if m.path else m.name
                    league_results.setdefault(key, []).append(float(s_[3]))
                    if groups[e] == "past":
                        opponent[e] = league.sample_past()
                    elif groups[e] == "script":
                        opponent[e] = league.sample_script()
        with torch.no_grad():
            u, x = net.encode(obs_t)
            last_v = net.value(net.gru(x, h * (1 - start).unsqueeze(-1))).squeeze(-1)
        steps += B * T
        t_rollout = time.time() - t0

        # ---- advantages ------------------------------------------------------------------------
        adv = torch.zeros(T, B, device=device)
        gae = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else b_val[t + 1]
            nonterm = 1 - b_done[t]
            delta = b_rew[t] + args.gamma * nv * nonterm - b_val[t]
            gae = delta + args.gamma * args.gae_lambda * nonterm * gae
            adv[t] = gae
        ret = adv + b_val

        # ---- update ----------------------------------------------------------------------------
        t1 = time.time()
        if args.eval_only:
            args.epochs = 0
        lr = args.lr * max(0.0, 1 - epoch / updates_total)
        for g_ in opt.param_groups:
            g_["lr"] = lr
        L = args.chunk
        chunks = [(t, b) for t in range(0, T, L) for b in range(B)]
        per_mb = max(1, args.minibatch // L)
        stats_acc = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}
        warm = epoch < args.vf_warmup
        stop_kl = False
        for _ in range(args.epochs):
            if stop_kl:
                break
            order = np.random.permutation(len(chunks))
            for i in range(0, len(order), per_mb):
                sel = [chunks[j] for j in order[i:i + per_mb]]
                ts = torch.tensor([c[0] for c in sel], device=device)
                bs = torch.tensor([c[1] for c in sel], device=device)
                idx_t = (ts.unsqueeze(0) + torch.arange(L, device=device).unsqueeze(1))  # [L, M]
                idx_b = bs.unsqueeze(0).expand(L, -1)
                o = b_obs[idx_t, idx_b]
                logp, ent, v = net.evaluate(o, b_masks[idx_t, idx_b], b_act[idx_t, idx_b], b_h[ts, bs],
                                            b_start[idx_t, idx_b])
                old = b_logp[idx_t, idx_b]
                a = adv[idx_t, idx_b]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - old).exp()
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vl = 0.5 * ((v - ret[idx_t, idx_b]) ** 2).mean()
                el = ent.mean()
                loss = args.vf_coef * vl if warm else pg + args.vf_coef * vl - args.ent_coef * el
                if ref is not None and not warm:  # KL(policy || reference) on the rollout's actions (k3)
                    lr_ = b_ref[idx_t, idx_b] - logp
                    ref_kl = (lr_.exp() - 1 - lr_).mean()
                    loss = loss + args.ref_kl * ref_kl
                    stats_acc["ref_kl"] = stats_acc.get("ref_kl", 0.0) + ref_kl.item()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    stats_acc["policy"] += pg.item()
                    stats_acc["value"] += vl.item()
                    stats_acc["entropy"] += el.item()
                    stats_acc["kl"] += ((ratio - 1) - (logp - old)).mean().item()
                    stats_acc["clipfrac"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                    stats_acc["n"] += 1
                    if args.target_kl and not warm and ((ratio - 1) - (logp - old)).mean().item() > args.target_kl:
                        stop_kl = True
                        break
        t_train = time.time() - t1
        epoch += 1
        n = max(stats_acc.pop("n"), 1)
        if "ref_kl" in stats_acc:
            stats_acc["ref_kl"] /= n
        row = {"agent_steps": steps, "SPS": B * T / (time.time() - t0), "epoch": epoch, "uptime": time.time() - t_begin,
               **({"loss/ref_kl": stats_acc.pop("ref_kl")} if "ref_kl" in stats_acc else {}),
               "loss/policy": stats_acc["policy"] / n, "loss/value": stats_acc["value"] / n,
               "loss/entropy": stats_acc["entropy"] / n, "loss/kl": stats_acc["kl"] / n,
               "loss/clipfrac": stats_acc["clipfrac"] / n, "perf/rollout": t_rollout, "perf/eval_env": t_env,
               "perf/eval_model": t_model, "perf/train": t_train, "lr": lr, "time": time.time(), **gpu_stats()}
        if finished["n"]:
            row.update({"env/win_rate": finished["wins"] / finished["n"], "env/loss_rate": finished["losses"] / finished["n"],
                        "env/episode_return": finished["ret"] / finished["n"], "env/episode_length": finished["len"] / finished["n"],
                        "env/n": finished["n"]})
            finished = {"n": 0, "wins": 0, "losses": 0, "ret": 0.0, "len": 0.0}
        if league is not None:  # win rates this epoch by opponent kind (draws count half)
            for key, res in league_results.items():
                row[f"league/{key}"] = sum(1.0 if r > 0.5 else 0.5 if r > -0.5 else 0.0 for r in res) / len(res)
                row[f"league/{key}_n"] = len(res)
            league_results = {}
            row["league/members"] = len(league.past)
        log.write(json.dumps(row) + "\n")
        log.flush()
        if epoch % 10 == 1:
            print(f"epoch {epoch} steps {steps} SPS {row['SPS']:.0f} win {row.get('env/win_rate', float('nan')):.2f} "
                  f"kl {row['loss/kl']:.4f} ent {row['loss/entropy']:.2f}", flush=True)
        if epoch % args.checkpoint_interval == 0 or steps >= total or STOP or (
                league is not None and epoch % args.snapshot_every == 0):
            path = ck_dir / f"{steps:016d}.pt"
            torch.save({"model": net.state_dict(), "config": net.config, "steps": steps, "spec": spec,
                        "args": vars(args)}, path)
            if league is not None and epoch % args.snapshot_every == 0:
                league.add_snapshot(str(path), steps)
        if league is not None:
            league.save()
    envs.close()
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
