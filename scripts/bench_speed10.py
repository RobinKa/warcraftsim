"""The 10-layer recipe's GPU costs, measured between runs (the GPU to itself), for the speed-ups of 2026-11-28.

    python3 scripts/bench_speed10.py flops   --run runs/fgself-271            # (CPU, meta tensors: FLOPs by part)
    python3 scripts/bench_speed10.py update  --run runs/fgself-271 [--profile 25] [--variants "base:;aligned:update_widths=32|40|48|56|63|80|96|127;labelrows:update_label_rows=32"]
    python3 scripts/bench_speed10.py past    --run runs/fgself-271 [--slots 10 --stack-rows 1,2,4]
    python3 scripts/bench_speed10.py buckets --run runs/fgself-271 [--sets "32,48,64,96;31,47,63,95,127;32,48,64,96,112"]
    python3 scripts/bench_speed10.py concurrent --run runs/fgself-271 [--seconds 60]

update: the learner's update (selfplay.GraphedUpdate, the run's settings: minibatch, sequences, epochs, bf16, the
anchor's forward-all KL, the labels' distillation, UPGO, Adam fused, --update-widths; no pauses) on synthetic steps of the
run's shapes (entity counts drawn around --ent-mean, own units ~60% of them, labels on --label-share of the steps);
seconds an update and each shape's graph replayed alone (GPU ms a minibatch), per variant (each a fresh network, the
same steps). The anchor is a fresh network of --ref-layers (the run's is fgself-100's 5) with the policy's heads.

past: a round's past-snapshot calls, as the C++ loop makes them: --infer-past-slots' network copies (a call per snapshot
on its slot's stream, rows padded to 4 or 16) against --infer-past-stack (one stacked call; paststack.PastStack). Rounds
drawn as the run's: ~6 snapshots of 1-4 rows, entity counts from its inference.jsonl, --snapshots distinct snapshots
over --slots slots (swaps as in the run). Prints ms a round (wall and GPU), past calls a second each could serve, and
the two paths' values and entropies on the same rows (they must agree to bf16-mixed's rounding).

buckets: the current network's calls per (rows, entities) shape, replayed alone; with the run's inference.jsonl
histograms (calls by rows, by widest entity count) the GPU ms a second each --infer-entities set would cost.

concurrent: an update loop and a past-round loop at once (two threads; the past rounds on high-priority streams as the
loop's): updates a second and rounds a second, unbatched against stacked.

Needs the GPU to itself (never beside a run): it allocates several GB and takes every SM it can. The first run compiles
every shape (the per-slot path ~10 graphs a slot: `--slots 4` for a quicker look); the inductor's cache serves later runs.
The update's synthetic actions are random (its losses are not a policy's), which changes nothing in the work timed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import selfplay as S  # noqa: E402
from warcraftsim.fullgame.model import FullGameNet  # noqa: E402
from warcraftsim.fullgame.paststack import PastStack, blank_rows, pack_rows, stack_rows  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------- setup
def setup(run: Path) -> tuple[dict, dict]:
    """The run's feature settings as its inference server sets them (selfplay.inference_main) -> (run.json's config, the
    current network's checkpoint: current.pt)."""
    c = json.loads((run / "run.json").read_text())["config"]

    def num(k: str) -> int:
        return int(float(c.get(k) or 0))
    fx.set_inputs(num("inputs") or 1, False)
    fx.set_masks(num("tech_mask"), num("valid_mask"))
    fx.set_neutral(num("neutral"))
    fx.set_macros(num("macros"))
    fx.set_spells(num("spells"))
    fx.set_shop_items(num("shop_items"))
    fx.set_full_vocab(num("full_vocab"))
    fx.set_obs_full(num("obs_full"))
    fx.set_fine(num("fine_points"))
    fx.set_macros_off(str(c.get("macros_off") or "").split(","))
    fx.set_econ_masks(num("econ_masks"))
    fx.set_order_masks(num("order_masks"))
    fx.set_value_tax(num("value_tax"))
    ck = torch.load(run / "current.pt", map_location="cpu", weights_only=False)
    fx.set_z(ck["config"].get("z", 0))
    return c, ck


def network(ck: dict, device, **over) -> FullGameNet:
    """The run's current network (its config, `over` changed: e.g. layers; a changed shape starts fresh)."""
    cfg = {**ck["config"], **over}
    net = FullGameNet(**cfg)
    if not over:
        net.load_state_dict(ck["model"])
    else:
        own = net.state_dict()
        net.load_state_dict({k: v for k, v in ck["model"].items() if k in own and own[k].shape == v.shape}, strict=False)
    return net.to(device)


def inference_entities(run: Path, past: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """The run's last ~100 inference.jsonl records: calls by widest entity count (8 a bin), by rows (8 a bin), and the
    calls a second, of the current network's calls (past False) or the past snapshots'."""
    path = run / "inference.jsonl"
    wide, rows, rate, n = np.zeros(20), np.zeros(8), 0.0, 0
    if path.exists():
        for line in path.read_text().splitlines()[-100:]:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "calls_by_entities" not in r:
                continue
            k = 1 if past else 0
            wide += np.asarray(r["calls_by_entities"][k], float)
            rows += np.asarray(r["calls_by_rows"][k], float)
            rate += r["past_calls_per_s"] if past else r["calls_per_s"] - r["past_calls_per_s"]
            n += 1
    if not n:  # (fgself-271's, 2026-11-28)
        wide = np.array([0, 537, 375, 362, 336, 756, 1175, 1274, 1130, 441, 324, 235, 192, 45, 68, 3, 0, 0, 0, 0] if past else
                        [2, 9, 9, 16, 22, 23, 26, 24, 45, 83, 169, 412, 291, 8, 0, 0, 0, 0, 0, 0], float)
        rows = np.array([7230, 23, 0, 0, 0, 0, 0, 0] if past else [188, 147, 248, 189, 90, 70, 65, 142], float)
        rate, n = (690.0 if past else 115.0), 1
    return wide / wide.sum(), rows / max(rows.sum(), 1), rate / n


# ---------------------------------------------------------------------------------------------------------------- steps
def make_chunks(net: FullGameNet, n_steps: int, rng, ent_mean: float, ent_sd: float, label_share: float,
                chunk: int = 64) -> list[list[dict]]:
    """Trajectory chunks of the run's shapes: each chunk's entity count around a draw of its own (slowly changing within
    it, as a game side's), own units 50-75% of them (at most features.MAX_OWN), actions within the masks' ranges."""
    c = net.config
    F, G, NO, d, T = fx.F, c["G"], c["n_orders"], c["d"], c["n_types"]
    pool = rng.standard_normal((64, fx.MAX_ENT, F)).astype(np.float16)
    k_shape = (ent_mean / ent_sd) ** 2
    chunks, total = [], 0
    while total < n_steps:
        n0 = float(np.clip(rng.gamma(k_shape, ent_mean / k_shape), 3, fx.MAX_ENT - 8))
        ch = []
        for t in range(chunk):
            n = int(np.clip(round(n0 + rng.normal(0, 3)), 2, fx.MAX_ENT))
            o = int(np.clip(round(n * rng.uniform(0.5, 0.75)), 1, min(n, fx.MAX_OWN)))
            avail = rng.random(NO) < 0.9
            avail[0] = True
            st = {"ent": pool[int(rng.integers(64)), :n], "type": rng.integers(1, T, n).astype(np.int16),
                  "cur": rng.integers(0, c["n_cur"], n).astype(np.int16), "glob": rng.normal(size=G).astype(np.float32),
                  "n": n, "n_own": o, "avail": avail, "order": rng.integers(0, NO, o).astype(np.int16),
                  "tgt": rng.integers(0, n, o).astype(np.int16), "bx": rng.integers(0, net.bins, o).astype(np.int16),
                  "by": rng.integers(0, net.bins, o).astype(np.int16), "logp": rng.uniform(-3, 0, o).astype(np.float32),
                  "adv": float(rng.normal()), "ret": float(rng.normal()), "uadv": float(rng.normal()), "version": 0,
                  "h": rng.normal(size=d).astype(np.float32) if t % 16 == 0 else None}
            if rng.random() < label_share:
                y = rng.integers(0, NO, o)
                y[rng.random(o) < 0.8] = 0  # (most units unlabelled: "none")
                st.update({"y_order": y.astype(np.int16), "y_ptr": np.where(rng.random(o) < 0.5, rng.integers(0, n, o), -1).astype(np.int16),
                           "y_x": np.where(rng.random(o) < 0.5, rng.integers(0, net.bins, o), -1).astype(np.int16),
                           "y_teach": rng.random(o) < 0.3})
                st["y_y"] = np.where(st["y_x"] >= 0, rng.integers(0, net.bins, o), -1).astype(np.int16)
            ch.append(st)
        chunks.append(ch)
        total += chunk
    return chunks


def update_args(c: dict, over: dict) -> types.SimpleNamespace:
    """GraphedUpdate's settings from the run's config (`over` changed), without pauses."""
    def f(k, d):
        try:
            return float(c.get(k, d))
        except (TypeError, ValueError):
            return float(d)
    a = types.SimpleNamespace(
        epochs=int(f("epochs", 1)), minibatch=int(f("minibatch", 256)), pad_groups=int(f("pad_groups", 8)),
        seq_len=int(f("seq_len", 16)), bf16=int(f("bf16", 1)), clip=f("clip", 0.2), vf_coef=f("vf_coef", 0.5),
        ent_coef=f("ent_coef", 0.0), ref_kl=f("ref_kl", 0.01), ref_kl_mode=c.get("ref_kl_mode", "forward-all"),
        opd_coef=f("opd_coef", 0.05), teach_coef=f("teach_coef", 0.5), label_norm=c.get("label_norm", "units"),
        upgo=f("upgo", 1.0), unit_credit=f("unit_credit", 0.0), max_grad_norm=f("max_grad_norm", 1.0),
        update_fuse=c.get("update_fuse", "default") or "", update_grad_fuse=int(f("update_grad_fuse", 0)),
        update_widths=c.get("update_widths", ""), update_pause_ms=0.0, bc_coef=0.0, opsd_coef=0.0, ref2_kl=0.0,
        label_w=1.0, adam_fused=int(f("adam_fused", 1)), update_label_rows=int(f("update_label_rows", 0)), ref_layers=5,
        layers=0)
    for k, v in over.items():
        old = getattr(a, k, None)
        setattr(a, k, type(old)(v) if isinstance(old, (int, float)) and not isinstance(old, bool) else v)
    return a


def floor_globals(net: FullGameNet, device) -> None:
    """The run's label floors (--teach-floor, its moves' place floor): class masks of the right size (which classes does
    not change the work)."""
    NO = net.config["n_orders"]
    g = torch.Generator().manual_seed(0)
    S.TEACH_FLOOR = ((torch.rand(NO, generator=g) < 0.05).to(device), math.log(0.005))
    S.TEACH_FLOOR_PLACE = ((torch.rand(NO, generator=g) < 0.02).to(device), math.log(0.25))
    S.TEACH_FLOOR_POINTER = None
    S.TEACH_FLOOR_LESSONS = None
    S.ANCHOR2.clear()


def ms(fn, reps: int = 20) -> float:
    """GPU ms of fn() (CUDA events around `reps` calls, after one)."""
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps


# --------------------------------------------------------------------------------------------------------------- update
def bench_update(a, c: dict, ck: dict, device) -> None:
    variants = [v.split(":", 1) for v in a.variants.split(";") if v.strip()]
    rng = np.random.default_rng(0)
    base = network(ck, "cpu")
    chunks = make_chunks(base, a.steps, rng, a.ent_mean, a.ent_sd, a.label_share)
    steps = [s for ch in chunks for s in ch]
    n = np.array([s["n"] for s in steps])
    print(f"{len(steps)} steps in {len(chunks)} chunks: entities mean {n.mean():.1f}, p90 {np.percentile(n, 90):.0f}; "
          f"labelled {np.mean(['y_order' in s for s in steps]):.3f}", flush=True)
    rows = []
    for name, spec in variants:
        over = dict(kv.split("=", 1) for kv in spec.split(",") if "=" in kv) if spec else {}
        # (a comma-separated value: --variants "aligned:update_widths=31|39|47": | for its commas)
        over = {k: v.replace("|", ",") for k, v in over.items()}
        args = update_args(c, over)
        torch.manual_seed(0)
        net = network(ck, device, **({"layers": int(args.layers)} if int(args.layers) else {})).train()
        ref = None
        if args.ref_kl > 0 and int(args.ref_layers) > 0:
            ref = network(ck, device, layers=int(args.ref_layers)).eval()
            for p in ref.parameters():
                p.requires_grad_(False)
        net.allowed[:] = True
        floor_globals(net, device)
        opt = torch.optim.Adam(net.parameters(), lr=3e-5, eps=1e-5, capturable=True,
                               **({"fused": True} if args.adam_fused else {}))
        up = S.GraphedUpdate(net, ref, opt, args, device)
        t0 = time.time()
        up.update(steps, chunks, False, None)  # (the first: every shape compiled and captured)
        torch.cuda.synchronize()
        first = time.time() - t0
        times, stats = [], {}
        for _ in range(a.updates):
            t = time.time()
            stats = up.update(steps, chunks, False, None)
            torch.cuda.synchronize()
            times.append(time.time() - t)
        keys = [k for k, _, _ in up._batches(chunks, None, args.epochs)]
        counts = {k: keys.count(k) for k in dict.fromkeys(keys)}
        per = {}
        for k, (graph, s, sb, out) in up.graphs.items():
            per[k] = ms(graph.replay, reps=10)
        gpu = sum(counts.get(k, 0) * per[k] for k in per) / 1000
        mem = torch.cuda.max_memory_reserved() / 2 ** 30
        print(f"[{name}] {spec or '(the run)'}: first update {first:.0f} s; {np.mean(times):.3f} s an update "
              f"(min {min(times):.3f}); the graphs alone {gpu:.3f} s ({1000 * gpu / max(1, len(keys)):.1f} ms a minibatch, "
              f"{len(keys)} minibatches); GPU memory {mem:.1f} GB; losses " +
              " ".join(f"{k} {v:.4f}" for k, v in sorted(stats.items()) if k.startswith("loss/")), flush=True)
        for k in sorted(per):
            print(f"    width {k[0]:3d}: {counts.get(k, 0):3d} minibatches x {per[k]:6.2f} ms", flush=True)
        if a.profile and not rows:  # (the first variant's kernels: one minibatch's step outside its graph, compiled)
            profile_step(up, a.profile)
        rows.append((name, np.mean(times), gpu))
        del up, opt, net, ref
        torch._dynamo.reset()  # (its compiled code held the variant's network and graphs)
        torch.cuda.empty_cache()
    if len(rows) > 1:
        b = rows[0]
        print("against", b[0] + ":", ", ".join(f"{r[0]} {100 * (r[1] / b[1] - 1):+.1f}% an update ({100 * (r[2] / b[2] - 1):+.1f}% GPU)"
                                          for r in rows[1:]))


def profile_step(up, top: int) -> None:
    """Where a minibatch's GPU time goes: its step (the compiled forward passes, losses, backward, clip and Adam) run
    outside its CUDA graph on the commonest width's static inputs, under torch.profiler: the top kernels by GPU time, and
    the shares of attention, matrix multiplies and the rest (Triton's fused elementwise and reductions among it)."""
    from torch.profiler import ProfilerActivity, profile
    key = max(up.graphs, key=lambda k: k[0] if k[0] <= 64 else 0)
    graph, s, sb, out = up.graphs[key]
    for _ in range(2):
        up.opt.zero_grad(set_to_none=True)
        up._step(s, sb)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            up.opt.zero_grad(set_to_none=True)
            up._step(s, sb)
        torch.cuda.synchronize()
    ka = [e for e in prof.key_averages() if e.device_time_total > 0]
    total = sum(e.self_device_time_total for e in ka) or 1.0
    kinds = {"attention": 0.0, "matmul": 0.0, "other": 0.0}
    for e in ka:
        n = e.key.lower()
        kind = "attention" if any(w in n for w in ("fmha", "attention", "flash", "cutlassb", "cutlassf")) else \
            "matmul" if any(w in n for w in ("gemm", "cutlass", "sm80_xmma", "ampere", "cublas", "s16816", "wgmma")) else "other"
        kinds[kind] += e.self_device_time_total
    print(f"    profile, width {key[0]}: {total / 3 / 1000:.2f} ms of kernels a step; " +
          ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in kinds.items()), flush=True)
    for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:top]:
        print(f"      {100 * e.self_device_time_total / total:5.1f}%  x{e.count // 3:<4d} {e.key[:110]}", flush=True)


# ----------------------------------------------------------------------------------------------------------------- past
class Rounds:
    """Past-snapshot rounds as the run's: each a few snapshots (keys) with 1-4 rows each, entity counts from the run's
    histogram of past calls' widest rows."""

    def __init__(self, run: Path, net: FullGameNet, n: int, keys: int, per_round: float, rng):
        wide, _, rate = inference_entities(run, past=True)
        self.rate = rate
        G = net.config
        out = []
        for _ in range(n):
            k = int(np.clip(rng.poisson(per_round), 1, 2 * keys))
            chosen = rng.choice(keys, size=min(k, keys), replace=False)
            rnd = []
            for key in chosen:
                r = int(rng.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05]))
                b = int(rng.choice(len(wide), p=wide))
                widest = int(rng.integers(8 * b + 1, 8 * b + 9))
                sts = []
                for j in range(r):
                    m = widest if j == 0 else int(rng.integers(max(2, widest // 2), widest + 1))
                    avail = rng.random(G["n_orders"]) < 0.9
                    avail[0] = True
                    ent = rng.random((m, fx.F)).astype(np.float32)
                    if fx.OBS_FULL:
                        ent[:, fx.A_OBS:fx.A_OBS + 3] = 0.0
                    sts.append({"n": m, "n_own": int(m * 0.6), "ent": ent, "type": rng.integers(1, G["n_types"], m),
                                "cur": rng.integers(0, G["n_cur"], m), "glob": rng.normal(size=G["G"]).astype(np.float32),
                                "avail": avail, "h": rng.normal(size=G["d"]).astype(np.float32), "shared_noise": False})
                rnd.append((f"snap{int(key)}", sts))
            out.append(rnd)
        self.rounds = out


def snapshots(net: FullGameNet, n: int, device) -> dict[str, FullGameNet]:
    """n networks of the current one's architecture, its weights a little moved (the speed does not depend on them)."""
    out = {}
    g = torch.Generator().manual_seed(1)
    for k in range(n):
        s = FullGameNet(**net.config)
        s.load_state_dict(net.state_dict())
        with torch.no_grad():
            for p in s.parameters():
                p.add_(torch.randn(p.shape, generator=g) * 1e-3)
        out[f"snap{k}"] = s.to(device).eval()
    return out


class Unbatched:
    """--infer-past-slots: a call per snapshot on its slot's stream, as the C++ loop (and launch_round) makes it."""

    def __init__(self, net, snaps, slots: int, entities, prec: dict, device, gen, priority: int = 0):
        self.snaps, self.device = snaps, device
        self.swap = S.PastNet(device, slots=slots)
        self.past = S.Inference(None, device, 16, True, buckets=(4, 16), thread=False, entities=tuple(entities),
                                fuse=True, generator=gen, capture_mode="global", **prec)
        self.streams = [torch.cuda.Stream(device=device, priority=priority) for _ in range(slots)]
        first = next(iter(snaps.values()))
        for k in range(slots):  # (as warm_inference: each slot's graphs)
            slot_net = self.swap.get(f"warm{k}", first, k)
            for B in self.past.buckets:
                for E in self.past.entities:
                    self.past.begin(0)
                    self.past._launch(slot_net, [{"n": E, "n_own": 1, "ent": np.zeros((E, fx.F), np.float32),
                                                  "type": np.zeros(E, np.int64), "cur": np.zeros(E, np.int64),
                                                  "glob": np.zeros(net.config["G"], np.float32), "avail": None,
                                                  "h": None}] * B)
                    self.past.wait()
            self.swap.holding.pop(k, None)
        self.swaps = 0

    def round(self, rnd) -> tuple[list, list]:
        base = torch.cuda.current_stream()
        self.past.begin(0)
        taken, used, handles = set(), [], []
        for key, sts in rnd:
            slot = self.swap.slot(key, taken)
            taken.add(slot)
            st = self.streams[slot]
            if st not in used:
                st.wait_stream(base)
                used.append(st)
            with torch.cuda.stream(st):
                if self.swap.holding.get(slot) != key:
                    self.swaps += 1
                net = self.swap.get(key, self.snaps[key], slot)
                for a in range(0, len(sts), 16):
                    handles.append((key, a, self.past._launch(net, sts[a:a + 16])))
        return handles, used

    def finish(self, handles, used) -> dict:
        self.past.wait(used)
        out = {}
        for key, a, h in handles:
            for i, r in enumerate(self.past._finish(h)):
                out[(key, a + i)] = (r["value"], r["entropy"])
        return out


class Stacked:
    """--infer-past-stack: the round's snapshots seated in the stack's slots (held first), one call per chunk."""

    def __init__(self, net, snaps, slots: int, rows, entities, prec: dict, device, gen, priority: int = 0):
        self.snaps, self.device = snaps, device
        first = next(iter(snaps.values()))
        self.stack = PastStack(first, slots, device, rows=rows, entities=entities, fuse=True, generator=gen, **prec)
        self.stack.warm()
        self.stream = torch.cuda.Stream(device=device, priority=priority)
        self.flats = {k: (S.pack_flat(s.parameters()), list(s.buffers())) for k, s in snaps.items()}
        self.host: dict = {}
        self.swaps = 0

    def round(self, rnd) -> tuple[list, list]:
        ps, Sn = self.stack, self.stack.S
        base = torch.cuda.current_stream()
        self.stream.wait_stream(base)
        handles, todo, bank = [], list(rnd), 0
        with torch.cuda.stream(self.stream):
            while todo:
                seat, taken = [-1] * len(todo), [False] * Sn
                for j, (key, _) in enumerate(todo):
                    for s in range(Sn):
                        if not taken[s] and ps.holding[s] == key:
                            seat[j], taken[s] = s, True
                            break
                for j, (key, _) in enumerate(todo):
                    if seat[j] >= 0:
                        continue
                    free = [s for s in range(Sn) if not taken[s]]
                    if not free:
                        continue
                    s = next((s for s in free if ps.holding[s] is None), free[0])
                    seat[j], taken[s] = s, True
                    ps.copy_in(s, *self.flats[key], key)
                    self.swaps += 1
                most = max(len(todo[j][1]) for j in range(len(todo)) if seat[j] >= 0)
                bmax = ps.rows[-1]
                for a in range(0, most, bmax):
                    parts = [(seat[j], todo[j][0], todo[j][1][a:a + bmax]) for j in range(len(todo))
                             if seat[j] >= 0 and len(todo[j][1]) > a]
                    B, E = ps.bucket(max(len(p) for _, _, p in parts), max(st["n"] for _, _, p in parts for st in p))
                    x, graph, out = ps.graph(B, E)
                    key = (B, E, bank)
                    if key not in self.host:
                        self.host[key] = (blank_rows(ps.net, Sn * B, E).pin_memory(), torch.zeros(out.shape).pin_memory())
                    hin, hout = self.host[key]
                    bank += 1
                    for s, _, p in parts:
                        pack_rows(ps.net, hin, p, E, s * B)
                    x.copy_(hin, non_blocking=True)
                    graph.replay()
                    hout.copy_(out, non_blocking=True)
                    handles.append((B, hout, [(s, k, a, len(p)) for s, k, p in parts]))
                todo = [todo[j] for j in range(len(todo)) if seat[j] < 0]
        return handles, [self.stream]

    def finish(self, handles, used) -> dict:
        used[0].synchronize()
        out, O = {}, None
        for B, hout, parts in handles:
            E = (hout.shape[1] - 2 - (self.stack.config["d"] if self.stack.config.get("memory") else 0)) // 5
            O = E
            for s, key, a, n in parts:
                for i in range(n):
                    r = hout[s * B + i]
                    out[(key, a + i)] = (float(r[5 * O]), float(r[5 * O + 1]))
        return out


def bench_past(a, c: dict, ck: dict, device) -> None:
    prec = S.precision_flags(c.get("infer_precision") or "fp32")
    entities = [int(x) for x in (a.past_entities or str(c.get("infer_past_entities", "32,48,64,96")).strip("[]")).split(",")]
    net = network(ck, device).eval()
    snaps = snapshots(net, a.snapshots, device)
    rng = np.random.default_rng(0)
    rounds = Rounds(a.run, net, a.rounds, a.snapshots, a.per_round, rng)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    t0 = time.time()
    paths = {"unbatched": Unbatched(net, snaps, a.slots, entities, prec, device, gen),
             "stacked": Stacked(net, snaps, a.slots, stack_rows(a.stack_rows), entities, prec, device, gen)}
    print(f"graphs captured in {time.time() - t0:.0f} s; {a.rounds} rounds of ~{a.per_round} snapshots (of {a.snapshots}) "
          f"over {a.slots} slots; the run's past calls {rounds.rate:.0f} a second", flush=True)
    answers = {}
    for name, p in paths.items():
        for rnd in rounds.rounds[:5]:  # (warm)
            p.finish(*p.round(rnd))
        torch.cuda.synchronize()
        p.swaps = 0
        spans, t = [], time.time()
        got = {}
        for rnd in rounds.rounds:
            st = torch.cuda.Event(enable_timing=True)
            st.record()
            h, used = p.round(rnd)
            ends = []
            for u in used:
                e = torch.cuda.Event(enable_timing=True)
                e.record(u)
                ends.append(e)
            got.update({(id(rnd),) + k: v for k, v in p.finish(h, used).items()})
            spans.append(max(st.elapsed_time(e) for e in ends))
        wall = (time.time() - t) / len(rounds.rounds)
        calls = sum(len(r) for r in rounds.rounds) / len(rounds.rounds)
        answers[name] = got
        print(f"{name:9s}: {1000 * wall:6.2f} ms a round (wall, one at a time), GPU span {np.mean(spans):6.2f} ms "
              f"(p90 {np.percentile(spans, 90):.2f}); {calls / wall:6.0f} past calls a second at most; "
              f"{p.swaps / len(rounds.rounds):.2f} swaps a round", flush=True)
    u, s_ = answers["unbatched"], answers["stacked"]
    common = [k for k in u if k in s_]
    dv = np.array([abs(u[k][0] - s_[k][0]) for k in common])
    de = np.array([abs(u[k][1] - s_[k][1]) for k in common])
    print(f"stacked against unbatched, {len(common)} rows: value |diff| mean {dv.mean():.1e} p99 {np.percentile(dv, 99):.1e} "
          f"max {dv.max():.1e}; entropy mean {de.mean():.1e} p99 {np.percentile(de, 99):.1e} max {de.max():.1e} "
          f"(bf16-mixed: expect ~1e-3, max ~1e-2)", flush=True)


# -------------------------------------------------------------------------------------------------------------- buckets
def bench_buckets(a, c: dict, ck: dict, device) -> None:
    prec = S.precision_flags(c.get("infer_precision") or "fp32")
    rows_set = S.infer_rows(str(c.get("infer_rows") or "").strip("[]"), 64)
    sets = [tuple(sorted({int(x) for x in s.split(",")} | {fx.MAX_ENT})) for s in a.sets.split(";")]
    every = sorted({e for s in sets for e in s})
    net = network(ck, device).eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    inf = S.Inference(None, device, 64, True, buckets=rows_set, thread=False, entities=tuple(every), fuse=True,
                      generator=gen, capture_mode="global", **prec)
    G = net.config["G"]
    t = {}
    for B in rows_set:
        for E in every:
            dummy = [{"n": E, "n_own": min(E, fx.MAX_OWN) // 2, "ent": np.random.rand(E, fx.F).astype(np.float32) * 0.1,
                      "type": np.ones(E, np.int64), "cur": np.zeros(E, np.int64), "glob": np.zeros(G, np.float32),
                      "avail": None, "h": None}] * B
            inf.begin(0)
            inf._launch(net, dummy)  # (its graph captured for this shape)
            inf.wait()
            x, graph, out = inf._graphs[net][(B, x_width(net, E))]
            t[(B, E)] = ms(graph.replay, reps=20)
            print(f"rows {B:2d} entities {E:3d}: {t[(B, E)]:.3f} ms", flush=True)
    wide, rows, rate = inference_entities(a.run, past=False)
    print(f"the run's current-network calls: {rate:.0f} a second", flush=True)
    for s in sets:
        tot = 0.0
        for rb, pr in enumerate(rows):
            if pr <= 0:
                continue
            B = next(b for b in rows_set if b >= min(8 * rb + 8, rows_set[-1]))
            for eb, pe in enumerate(wide):
                if pe <= 0:
                    continue
                E = next(e for e in s if e >= min(8 * eb + 8, fx.MAX_ENT))
                tot += pr * pe * t[(B, E)]
        print(f"--infer-entities {','.join(str(e) for e in s if e != fx.MAX_ENT)}: {tot:.3f} ms a call, "
              f"{rate * tot:.0f} ms of GPU a second", flush=True)


def x_width(net, E: int) -> int:
    d = net.config["d"] if net.memory else 0
    return E * (fx.F + 3) + net.config["G"] + 1 + net.config["n_orders"] + d


# ----------------------------------------------------------------------------------------------------------- concurrent
def bench_concurrent(a, c: dict, ck: dict, device) -> None:
    prec = S.precision_flags(c.get("infer_precision") or "fp32")
    entities = [int(x) for x in (a.past_entities or str(c.get("infer_past_entities", "32,48,64,96")).strip("[]")).split(",")]
    rng = np.random.default_rng(0)
    cpu_net = network(ck, "cpu")
    chunks = make_chunks(cpu_net, a.steps, rng, a.ent_mean, a.ent_sd, a.label_share)
    steps = [s for ch in chunks for s in ch]
    args = update_args(c, {})
    torch.manual_seed(0)
    net = network(ck, device).train()
    ref = network(ck, device, layers=int(args.ref_layers)).eval() if args.ref_kl > 0 else None
    if ref is not None:
        for p in ref.parameters():
            p.requires_grad_(False)
    net.allowed[:] = True
    floor_globals(net, device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-5, eps=1e-5, capturable=True, **({"fused": True} if args.adam_fused else {}))
    up = S.GraphedUpdate(net, ref, opt, args, device)
    up.update(steps, chunks, False, None)
    torch.cuda.synchronize()
    play = network(ck, device).eval()
    snaps = snapshots(play, a.snapshots, device)
    rounds = Rounds(a.run, play, a.rounds, a.snapshots, a.per_round, rng)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    hi = torch.cuda.Stream.priority_range()[1]
    paths = {"unbatched": Unbatched(play, snaps, a.slots, entities, prec, device, gen, hi),
             "stacked": Stacked(play, snaps, a.slots, stack_rows(a.stack_rows), entities, prec, device, gen, hi)}
    for name, p in paths.items():
        stop, counts = threading.Event(), {"updates": 0, "rounds": 0}

        def learner():
            while not stop.is_set():
                up.update(steps, chunks, False, None)
                torch.cuda.current_stream().synchronize()  # (its own stream: not the rounds')
                counts["updates"] += 1

        def rounds_loop():
            k = 0
            # (the rounds' base stream their own, as the C++ loop's: not the learner's, whose kernels they would wait for)
            with torch.cuda.stream(torch.cuda.Stream(device=device, priority=hi)):
                while not stop.is_set():
                    h, used = p.round(rounds.rounds[k % len(rounds.rounds)])
                    p.finish(h, used)
                    counts["rounds"] += 1
                    k += 1
        threads = [threading.Thread(target=learner), threading.Thread(target=rounds_loop)]
        t = time.time()
        [x.start() for x in threads]
        time.sleep(a.seconds)
        stop.set()
        [x.join() for x in threads]
        dt = time.time() - t
        calls = sum(len(r) for r in rounds.rounds) / len(rounds.rounds)
        print(f"{name:9s}: {counts['updates'] / dt * a.steps:6.0f} learner steps a second ({dt / max(1, counts['updates']):.2f} s "
              f"an update) beside {counts['rounds'] / dt:5.0f} rounds a second ({calls * counts['rounds'] / dt:5.0f} past calls)",
              flush=True)


# ---------------------------------------------------------------------------------------------------------------- flops
def bench_flops(a, c: dict, ck: dict) -> None:
    from torch.utils.flop_counter import FlopCounterMode

    from warcraftsim.fullgame.model import anchor_kl, evaluate
    meta = torch.device("meta")
    cfg = dict(ck["config"])
    for W in [int(x) for x in a.widths.split(",")]:
        with meta:
            net = FullGameNet(**cfg)
            ref = FullGameNet(**{**cfg, "layers": a.ref_layers_flops})
        for p in ref.parameters():
            p.requires_grad_(False)
        N, T = 256, 16
        O = min(fx.MAX_OWN, W)
        NO, G, d = cfg["n_orders"], cfg["G"], cfg["d"]
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=meta)  # noqa: E731
        s = {"ent": z(N, W, fx.F), "type": z(N, W, dt=torch.long), "cur": z(N, W, dt=torch.long),
             "mask": z(N, W, dt=torch.bool), "glob": z(N, G), "n_own": z(N, dt=torch.long), "avail": z(N, NO, dt=torch.bool),
             "own": z(N, O, dt=torch.bool), **{k: z(N, O, dt=torch.long) for k in ("order", "tgt", "bx", "by", "y_order",
                                                                                  "y_ptr", "y_x", "y_y")},
             "y_has": z(N, dt=torch.bool), "y_only": z(N, dt=torch.bool), "y_teach": z(N, O, dt=torch.bool)}
        S.TEACH_FLOOR = (z(NO, dt=torch.bool), -5.0)
        S.TEACH_FLOOR_PLACE = (z(NO, dt=torch.bool), -1.4)
        seq = (N // T, T, z(N // T, d), None)
        parts = {}

        def count(name, fn):
            with FlopCounterMode(display=False) as fc:
                out = fn()
            parts[name] = fc.get_total_flops() / 1e9
            return out
        ev = count("policy forward", lambda: evaluate(net, s["ent"], s["type"], s["cur"], s["mask"], s["glob"], s["n_own"],
                                                      s["order"], s["tgt"], s["bx"], s["by"], s["avail"], seq=seq))
        with torch.no_grad():
            er = count(f"anchor forward ({a.ref_layers_flops} layers)", lambda: evaluate(
                ref, s["ent"], s["type"], s["cur"], s["mask"], s["glob"], s["n_own"], s["order"], s["tgt"], s["bx"], s["by"],
                s["avail"], seq=seq))
        kl = count("KL", lambda: anchor_kl(ev, er, "forward-all"))
        dist = count("labels' heads forward", lambda: S.distill_loss(net, ev, s, None, 10.0, "units"))
        loss = ev["value"].sum() + ev["logp"].sum() + kl.sum() + dist[0]
        count("backward", lambda: loss.backward())
        tot = sum(parts.values())
        trunk = 2 * N * (W + 1) * 12 * d * d * cfg["layers"] / 1e9
        L = W + 1
        tiles = math.ceil(L / 64) ** 2 * 64 * 64 / (L * L)
        print(f"width {W} (own slots {O}), a minibatch of {N} steps: {tot:.0f} GFLOP: " +
              ", ".join(f"{k} {v:.0f} ({100 * v / tot:.0f}%)" for k, v in parts.items()) +
              f"; the policy's transformer matmuls ~{trunk:.0f} fwd; attention tiles (64x64) do {tiles:.2f}x its "
              f"{L}x{L} scores", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("flops", "update", "past", "buckets", "concurrent"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=8192, help="update: steps a batch")
    ap.add_argument("--updates", type=int, default=5, help="update: timed updates (after the first)")
    ap.add_argument("--variants", default="base:;aligned:update_widths=32|40|48|56|63|80|96|127;labelrows:update_label_rows=32",
                    help="update: name:setting=value,... ; separated (| for a value's commas): e.g. noref:ref_layers=0, "
                         "nolabels:opd_coef=0, five:layers=5, both:update_widths=32|40|48|56|63|80|96|127,update_label_rows=32")
    ap.add_argument("--profile", type=int, default=0, help="update: the first variant's top N kernels (torch.profiler)")
    ap.add_argument("--ent-mean", type=float, default=53.0, help="steps' entity count (fgself-271's rows: ~53)")
    ap.add_argument("--ent-sd", type=float, default=16.0)
    ap.add_argument("--label-share", type=float, default=0.01, help="steps with labels (fgself-271's opd/steps: ~0.009)")
    ap.add_argument("--slots", type=int, default=10, help="past: --infer-past-slots, and the stack's slots")
    ap.add_argument("--snapshots", type=int, default=16, help="past: distinct snapshots the rounds draw from")
    ap.add_argument("--per-round", type=float, default=6.3, help="past: snapshots a round (fgself-271: ~6.3)")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--stack-rows", default="1,2,4", help="past: --infer-past-stack-rows")
    ap.add_argument("--past-entities", default="", help="past: --infer-past-entities (default the run's)")
    ap.add_argument("--sets", default="32,48,64,96;32,48,63,96;32,48,63,96,112;31,47,63,95,127",
                    help="buckets: --infer-entities sets to price, ; separated")
    ap.add_argument("--seconds", type=float, default=60.0, help="concurrent: seconds each")
    ap.add_argument("--widths", default="32,48,57,63,64,96,127,128", help="flops: minibatch widths")
    ap.add_argument("--ref-layers-flops", type=int, default=5, help="flops: the anchor's layers")
    a = ap.parse_args()
    c, ck = setup(a.run)
    print(f"{a.run}: {ck['config']['layers']} layers, d {ck['config']['d']}, F {fx.F}, G {ck['config']['G']}, "
          f"{ck['config']['n_orders']} orders; torch {torch.__version__}", flush=True)
    if a.mode == "flops":
        bench_flops(a, c, ck)
        return 0
    device = torch.device(a.device)
    if device.type != "cuda":
        raise SystemExit("the GPU modes need --device cuda (and the GPU to themselves)")
    {"update": bench_update, "past": bench_past, "buckets": bench_buckets, "concurrent": bench_concurrent}[a.mode](a, c, ck, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
