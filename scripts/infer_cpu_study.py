"""What --infer-precision and --infer-rows would change, measured on the CPU (no GPU: beside a live run).

    python3 scripts/infer_cpu_study.py precision            # fgself-148's network on its recorded steps
    python3 scripts/infer_cpu_study.py flops --run fgself-182

precision: how far the policy's numbers move if the inference calls ran their matmuls in TF32 or bf16 instead of fp32.
Real steps (the learner's chunks recorded by scripts/bench_actor.py, runs/bench/ref5-*.pkl, fgself-148's games) through
fgself-148's network with model.evaluate (the encode, heads and log-probabilities act() uses): fp32 the reference; TF32
emulated (every matmul's float32 operands rounded to 10 bits of mantissa, sums in fp32: cuBLAS's TF32 mode); bf16 by the
CPU's autocast. The bf16 row is also the gap between today's fp32 acting and the learner's bf16 evaluation (--bf16 1),
which PPO's ratio carries already. Actions this checkpoint masks (log-probability ~ -1e9) are left out.

flops: the inference calls' matmul FLOPs by shape (FlopCounterMode, the run's network shape, every unit slot its own),
weighted by the run's calls (inference.jsonl: by rows and by widest row, taken as independent), as GPU-ms of matmuls a
second at this GPU's rates for our layer shapes (runs/bench/gpu-after-unlock-400w.json: fp32 ~17-23, tf32 ~34-52, bf16
~64-103 TFLOPS), today and with finer row and entity buckets. Matmuls only: a call's small kernels' latency is not in it.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs"  # (--runs)

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame.model import FullGameNet, configure_for, evaluate, load  # noqa: E402


def tf32(t):
    """float32 rounded to TF32 (10 mantissa bits, to nearest, ties away: cvt.rna.tf32.f32)."""
    if not isinstance(t, torch.Tensor) or t.dtype != torch.float32:
        return t
    r = (t.contiguous().view(torch.int32) + 0x1000) & ~0x1FFF
    return torch.where(torch.isfinite(t), r.view(torch.float32), t)


MM = {F.linear: (0, 1), torch.matmul: (0, 1), torch.bmm: (0, 1), torch.mm: (0, 1), torch.Tensor.__matmul__: (0, 1),
      torch.baddbmm: (1, 2), torch.addmm: (1, 2), F.scaled_dot_product_attention: (0, 1, 2)}


class TF32(TorchFunctionMode):
    """Every matmul's operands rounded to TF32 (the attention's too: an upper bound)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in MM:
            args = tuple(tf32(a) if i in MM[func] else a for i, a in enumerate(args))
            for k in ("input", "weight", "query", "key", "value", "mat1", "mat2", "batch1", "batch2"):
                if k in kwargs:
                    kwargs[k] = tf32(kwargs[k])
        elif func is torch.einsum:
            args = (args[0],) + tuple(tf32(a) for a in args[1:])
        return func(*args, **kwargs)


def precision(a) -> int:
    torch.backends.mha.set_fastpath_enabled(False)  # (the CPU's fused encoder layer would hide its matmuls)
    ck = sorted(glob.glob(str(RUNS / a.run / "checkpoints" / "*.pt")))[-1]
    configure_for(ck)
    net, _ = load(ck, "cpu")
    net.eval()
    steps = []
    for f in sorted(glob.glob(str(RUNS / "bench" / "ref5-*.pkl"))):
        if "uol" in f:  # (the long one: 160 MB)
            continue
        ref = pickle.loads(Path(f).read_bytes())
        steps += [s for m in ref["puts"] if isinstance(m, dict) and "steps" in m for s in m["steps"][::a.every] if s["n_own"] > 0]
        del ref
    print(f"{Path(ck).name}: {len(steps)} steps", flush=True)

    def batch(sts):
        N, E = len(sts), max(s["n"] for s in sts)
        O = min(fx.MAX_OWN, E)
        ent, typ, cur = torch.zeros(N, E, fx.F), torch.zeros(N, E, dtype=torch.long), torch.zeros(N, E, dtype=torch.long)
        mask, acts = torch.zeros(N, E, dtype=torch.bool), torch.zeros(4, N, O, dtype=torch.long)
        for i, s in enumerate(sts):
            n, o = s["n"], s["n_own"]
            ent[i, :n], mask[i, :n] = torch.from_numpy(s["ent"].astype(np.float32)), True
            typ[i, :n], cur[i, :n] = torch.from_numpy(s["type"].astype(np.int64)), torch.from_numpy(s["cur"].astype(np.int64))
            for k, f in enumerate(("order", "tgt", "bx", "by")):
                acts[k, i, :o] = torch.from_numpy(s[f].astype(np.int64))
        glob_ = torch.from_numpy(np.stack([s["glob"] for s in sts])).float()
        avail = torch.from_numpy(np.stack([s["avail"] for s in sts]))
        n_own = torch.tensor([s["n_own"] for s in sts])
        h0 = torch.from_numpy(np.stack([s["h"] if s.get("h") is not None else np.zeros(net.d, np.float32) for s in sts])).float()
        return (ent, typ, cur, mask, glob_, n_own, *acts, avail), (N, 1, h0, None), n_own, O

    def run(mode, args, seq):
        with torch.no_grad():
            if mode == "tf32":
                with TF32():
                    return evaluate(net, *args, seq=seq)
            if mode == "bf16":
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    return evaluate(net, *args, seq=seq)
            return evaluate(net, *args, seq=seq)

    res = {m: {k: [] for k in ("logp", "kl", "value", "agree")} for m in ("tf32", "bf16")}
    for at in range(0, len(steps), 48):
        args, seq, n_own, O = batch(steps[at:at + 48])
        own = torch.arange(O)[None] < n_own[:, None]
        ref = run("fp32", args, seq)
        possible = own & (ref["logp"].float() > -50)  # (an action this checkpoint masks: ~ -1e9, only rounding noise)
        lr = torch.log_softmax(ref["logits"].float(), -1)
        for m, r in res.items():
            ev = run(m, args, seq)
            lv = torch.log_softmax(ev["logits"].float(), -1)
            r["logp"].append((ev["logp"].float() - ref["logp"].float()).abs()[possible])
            r["kl"].append((lr.exp() * (lr - lv)).sum(-1)[own])
            r["value"].append((ev["value"].float() - ref["value"].float()).abs())
            r["agree"].append((lv.argmax(-1) == lr.argmax(-1))[own].float())
    for m, r in res.items():
        lp, kl, v, ag = (torch.cat(r[k]).double().numpy() for k in ("logp", "kl", "value", "agree"))
        print(f"{m} against fp32: |d logp| a unit mean {lp.mean():.1e} p99 {np.percentile(lp, 99):.1e} max {lp.max():.1e}; "
              f"KL(order) mean {kl.mean():.1e} p99 {np.percentile(kl, 99):.1e} max {kl.max():.1e}; |d value| mean {v.mean():.1e} "
              f"p99 {np.percentile(v, 99):.1e} max {v.max():.1e}; most likely order the same {100 * ag.mean():.2f}% "
              f"({len(lp)} units, {len(v)} steps)", flush=True)
    return 0


def flops(a) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    from warcraftsim.fullgame.selfplay import Inference
    torch.backends.mha.set_fastpath_enabled(False)
    ck = RUNS / a.run / "current.pt"
    configure_for(ck)
    cfg = torch.load(ck, map_location="cpu", weights_only=False)["config"]
    net = FullGameNet(**cfg).eval()
    net.allowed[:] = True
    G, NO, d = cfg["G"], cfg["n_orders"], cfg["d"]
    fn = Inference(None, torch.device("cpu"), 64, compile_=False, thread=False)._fn(net)
    cache: dict = {}

    def call(B, E):
        if (B, E) not in cache:
            x = torch.zeros(B, E * (fx.F + 3) + G + 1 + NO + d)
            at = E * fx.F + G
            x[:, at + 2 * E:at + 3 * E] = 1.0  # (the mask)
            x[:, at + 3 * E] = min(E, fx.MAX_OWN)  # (n_own: every unit slot, the heads' most)
            x[:, at + 3 * E + 1:at + 3 * E + 1 + NO] = 1.0
            with torch.no_grad(), FlopCounterMode(display=False) as fc:
                fn(x)
            cache[(B, E)] = fc.get_total_flops()
        return cache[(B, E)]

    recs = [json.loads(line) for line in open(RUNS / a.run / "inference.jsonl")]
    recs = [r for r in recs if "calls_by_rows" in r]
    by_rows = np.sum([r["calls_by_rows"][0] for r in recs], 0)
    by_ents = np.sum([r["calls_by_entities"][0] for r in recs], 0)
    past_ents = np.sum([r["calls_by_entities"][1] for r in recs], 0)
    current = float(np.mean([r["calls_per_s"] - r["past_calls_per_s"] for r in recs]))
    past = float(np.mean([r["past_calls_per_s"] for r in recs]))
    by_rows, by_ents, past_ents = by_rows / by_rows.sum(), by_ents / by_ents.sum(), past_ents / max(past_ents.sum(), 1)
    print(f"{a.run}: {len(recs)} records; current calls {current:.0f}/s, past {past:.0f}/s", flush=True)

    def per_s(rows, ents):
        cur = sum(current * pr * pe * call(next(b for b in rows if b >= 8 * i + 4.5), next(e for e in ents if e >= 8 * j + 4.5))
                  for i, pr in enumerate(by_rows) for j, pe in enumerate(by_ents))
        pst = sum(past * pe * call(4, next(e for e in (32, 48, 64, 96, 160) if e >= 8 * j + 4.5)) for j, pe in enumerate(past_ents))
        return cur, pst
    base = per_s((8, 24, 64), (32, 48, 64, 96, 160))[0]
    for name, rows, ents in (("today: rows 8,24,64, entities 32,48,64,96", (8, 24, 64), (32, 48, 64, 96, 160)),
                             ("rows 8,16,24,32,48,64", (8, 16, 24, 32, 48, 64), (32, 48, 64, 96, 160)),
                             ("and entities 80,112 too", (8, 16, 24, 32, 48, 64), (32, 48, 64, 80, 96, 112, 160))):
        cur, pst = per_s(rows, ents)
        tot = cur + pst
        print(f"{name}: current calls {cur / 1e12:.2f} TFLOP/s ({100 * (1 - cur / base):.0f}% less), past {pst / 1e12:.2f}; "
              f"matmul GPU-ms a second at fp32 / tf32 / bf16 (18 / 40 / 80 TFLOPS): {tot / 18e9:.0f} / {tot / 40e9:.0f} / "
              f"{tot / 80e9:.0f}", flush=True)
    for (B, E), f in sorted(cache.items()):
        if B in (8, 24, 32, 64) and E in (48, 96, 160):
            print(f"  a call of {B} rows x {E} entities: {f / 1e9:.1f} GFLOP: {f / 18e9:.2f} ms fp32, {f / 40e9:.2f} tf32, "
                  f"{f / 80e9:.2f} bf16")
    return 0


def main() -> int:
    global RUNS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["precision", "flops"])
    ap.add_argument("--run", help="precision: the run whose network made runs/bench/ref5-* (fgself-148); flops: a run with "
                                  "inference.jsonl (fgself-182)")
    ap.add_argument("--every", type=int, default=3, help="precision: every this many recorded steps")
    ap.add_argument("--runs", type=Path, default=RUNS, help="the runs folder (a worktree has none: the main tree's)")
    a = ap.parse_args()
    RUNS = a.runs
    a.run = a.run or ("fgself-148" if a.what == "precision" else "fgself-182")
    torch.set_num_threads(4)
    return precision(a) if a.what == "precision" else flops(a)


if __name__ == "__main__":
    raise SystemExit(main())
