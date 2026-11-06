"""What --infer-precision and --infer-rows would change, measured on the CPU (no GPU: beside a live run).

    python3 scripts/infer_cpu_study.py precision            # fgself-148's network on its recorded steps
    python3 scripts/infer_cpu_study.py flops --run fgself-182
    python3 scripts/infer_cpu_study.py mixed --runs /home/tora/warcraftsim/runs [--learner-heads] [--jitter 2.4e-7]

mixed: each inference precision (and split of it) against fp32 acting and against the learner's own bf16 pass, on the
same steps and actions, with CUDA's bf16 autocast emulated on the CPU (CudaAutocast, sdpa_cuda: the tensor cores' fp32
sums of bf16 products rounded once; layer norms, softmax and sums in fp32; the attention's memory-efficient kernel).
Game sides of recorded games (runs/fullgame/demos-echo-1) through the actors' View.step, fgself-182's network. The
inference side as the server's calls: the sides' rows of a step in one call (rows and entities padded to the buckets),
model.act carrying each candidate's own memory state; the fp32 call draws the actions all are scored on. The learner's
side as the update: model.evaluate on sequences of --seq steps from the candidate's recorded state, its entities float16
(raw_step_value), minibatches of 16 sequences padded to --update-widths. Prints per head (order, pointer, point) the
log-probability's difference, KL and entropy against fp32 and the value's; then PPO's log ratio at the first minibatch
without lag (|log r|, the k3 KL estimate, the share past the clip), and the value's difference from the learner's.

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
import contextlib
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

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


BF = torch.bfloat16
_A = torch.ops.aten
# the matrix multiplies (aten level: linear, matmul and einsum come apart into these) and their matrix operands
_MM = {_A.mm: (0, 1), _A.addmm: (1, 2), _A.bmm: (0, 1), _A.baddbmm: (1, 2), _A.mv: (0, 1), _A.addmv: (1, 2), _A.dot: (0, 1)}
# what CUDA's autocast runs in float32 whatever its inputs (the CPU's runs them in their own dtype: a layer norm or a sum
# of bf16 stays bf16 there)
_FP32_OPS = {_A.native_layer_norm, _A._softmax, _A._log_softmax, _A.sum, _A.exp, _A.log, _A.pow, _A.rsqrt, _A.reciprocal,
        _A.prod, _A.cumsum, _A.cumprod, _A.logsumexp, _A.log1p, _A.expm1, _A.softplus, _A.linalg_vector_norm}


def _flt(x):
    return x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x


class CudaAutocast(TorchDispatchMode):
    """torch.autocast("cuda", torch.bfloat16) as the GPU computes it, on the CPU: entered inside torch.autocast("cpu",
    torch.bfloat16) (cudaish(), which casts the same matrix multiplies' inputs to bf16), it makes (1) every bf16 matrix
    multiply a tensor core's: the bf16 operands' products (exact in fp32) summed in fp32 and the result rounded to bf16
    once, whatever the CPU's own bf16 kernels do; (2) the ops CUDA's autocast runs in fp32 (layer norm, softmax, sum,
    exp, log, ...) fp32 on bf16 inputs too. The attention is sdpa_cuda's. In a region with autocast off (the heads
    in fp32) nothing is cast. `tf32`: float32 multiplies with their matrix operands rounded to TF32 (cuBLAS's TF32
    mode; the attention's own kernel not). `jitter`: each bf16 multiply's fp32 sums moved by up to this much (relative,
    at random) before their rounding: another kernel's order of summation (another shape, split-K, another graph's)."""

    def __init__(self, tf32_: bool = False, jitter: float = 0.0):
        super().__init__()
        self.tf32, self.jitter = tf32_, jitter
        self.gen = torch.Generator().manual_seed(7)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        p = func.overloadpacket
        if p in _MM:
            if any(isinstance(x, torch.Tensor) and x.dtype == BF for x in args):
                with torch.autocast("cpu", enabled=False):
                    out = func(*map(_flt, args), **kwargs)
                    if self.jitter:
                        out = out * (1.0 + self.jitter * (2.0 * torch.rand(out.shape, generator=self.gen) - 1.0))
                    return out.to(BF)
            if self.tf32:
                args = tuple(tf32(x) if i in _MM[p] else x for i, x in enumerate(args))
        elif p in _FP32_OPS:  # (by the inputs' dtype: autocast's state reads off down here; the model's fp32 regions cast
            # their inputs to fp32 themselves, so a bf16 input here is autocast's)
            args = tuple(x.float() if isinstance(x, torch.Tensor) and x.dtype == BF else x for x in args)
            kwargs = {k: (v.float() if isinstance(v, torch.Tensor) and v.dtype == BF else v) for k, v in kwargs.items()}
        return func(*args, **kwargs)


_SDPA = F.scaled_dot_product_attention


def sdpa_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """scaled_dot_product_attention on bf16 inputs as CUDA's memory-efficient kernel computes it (autocast casts the
    mask to bf16 too): the scores and their softmax in fp32, the unnormalized probabilities rounded to bf16 for the
    product with V (fp32 sums), divided by their fp32 sum, the output rounded to bf16. Other inputs: torch's own."""
    if q.dtype != BF:
        return _SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                     enable_gqa=enable_gqa)
    with torch.autocast("cpu", enabled=False):
        s = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (scale if scale is not None else q.shape[-1] ** -0.5)
        if attn_mask is not None:
            s = s + attn_mask.to(BF).float()
        e = torch.exp(s - s.amax(-1, keepdim=True))
        return (torch.matmul(e.to(BF).float(), v.float()) / e.sum(-1, keepdim=True)).to(BF)


@contextlib.contextmanager
def cudaish(kind: str, jitter: float = 0.0):
    """A region computed as the GPU would with this precision: "fp32" (as is), "tf32" (CudaAutocast's TF32 rounding),
    "bf16" (CUDA's bf16 autocast: torch.autocast("cpu") with CudaAutocast and sdpa_cuda; the model's own autocast-off
    regions, torch.autocast(device, enabled=False), stay fp32), "bf16+tf32" (and those regions' multiplies in TF32;
    --infer-precision bf16-mixed). `jitter`: CudaAutocast's."""
    if kind == "fp32":
        yield
        return
    if kind == "tf32":
        with CudaAutocast(tf32_=True):
            yield
        return
    F.scaled_dot_product_attention = sdpa_cuda  # (F.multi_head_attention_forward calls the module's name)
    try:  # (sdpa_cuda's own fp32 multiplies are of bf16 values: exact in TF32 too)
        with torch.autocast("cpu", dtype=BF), CudaAutocast(tf32_=kind == "bf16+tf32", jitter=jitter):
            yield
    finally:
        F.scaled_dot_product_attention = _SDPA


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


def view_windows(a, vocab: dict) -> list[list[dict]]:
    """Windows of consecutive steps of recorded games' sides (a.warm + a.steps each, at random places), as the actors'
    View.step makes them (the inference calls' float32 entities; the learner gets them as float16, raw_step_value)."""
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame.costs import order_costs
    fx.set_geometry(*fx.map_geometry(stock_map_path("(2)EchoIsles")))
    fx.set_econ_masks(255)  # (fgself-194's; the masks only decide which orders are open)
    enc = fx.Encoder(vocab, order_costs(vocab, "echofast2"))
    rng = np.random.default_rng(a.seed)
    games = sorted(Path(a.demos).glob("game*.npz"))
    need, out = a.warm + a.steps, []
    for gi in rng.permutation(len(games)):
        if len(out) >= a.streams:
            break
        g = fx.load_game(games[gi])
        meta = g["meta"]
        T = meta["steps"] + 1
        if T < need + 50:
            continue
        u, pl, ev = g["units"], g["players"], g["events"]
        if fx.F > fx.F_BASE and "heroes" in g:
            u = fx.hero_rows(u, g["heroes"])
        us, ps, es = (fx._step_slices(x[:, 0], T) for x in (u, pl, ev))
        races = [fx.RACES.index(r) if r in fx.RACES else 0 for r in meta["races"]]
        for player in (0, 1):
            t0 = int(rng.integers(50, T - need + 1))  # (past the first steps: a few units only)
            view = enc.view(player, enc.side(u[us[0]:us[1]], player), races)
            view.step_seconds, view.start = float(meta.get("step_seconds", 0.5)), None
            win = []
            for t in range(t0 + need):
                prow = pl[ps[t]:ps[t + 1]]
                me, opp = prow[prow[:, 1] == player], prow[prow[:, 1] == 1 - player]
                st = view.step(u[us[t]:us[t + 1]], me[0] if len(me) else None, ev[es[t]:es[t + 1]], t,
                               opp[0] if len(opp) else None)
                if t >= t0:
                    n = st["n"]
                    win.append({"n": n, "n_own": min(int(st["n_own"]), fx.MAX_OWN),
                                "ent": np.array(st["ent"][:n], np.float32), "type": np.array(st["type"][:n], np.int64),
                                "cur": np.array(st["cur"][:n], np.int64), "glob": np.array(st["glob"], np.float32),
                                "avail": None if st["avail"] is None else np.array(st["avail"], bool)})
            out.append(win)
        del g, u, pl, ev
    return out[:a.streams]


def _batch(sts: list[dict], B: int, E: int, G: int, NO: int, half: bool = False) -> tuple:
    """Steps as an inference call's padded tensors (Inference._launch's layout: padding rows one entity, every order
    open), or the learner's (`half`: the entities as the learner gets them, float16)."""
    ent = torch.zeros(B, E, fx.F)
    typ, cur = torch.zeros(B, E, dtype=torch.long), torch.zeros(B, E, dtype=torch.long)
    mask = torch.zeros(B, E, dtype=torch.bool)
    mask[:, 0] = True
    glob_, n_own, avail = torch.zeros(B, G), torch.zeros(B, dtype=torch.long), torch.ones(B, NO, dtype=torch.bool)
    for i, st in enumerate(sts):
        n = st["n"]
        e = st["ent"].astype(np.float16).astype(np.float32) if half else st["ent"]
        ent[i, :n], typ[i, :n], cur[i, :n], mask[i, :n] = torch.from_numpy(e), torch.from_numpy(st["type"]), \
            torch.from_numpy(st["cur"]), True
        glob_[i], n_own[i] = torch.from_numpy(st["glob"]), st["n_own"]
        if st["avail"] is not None:
            avail[i] = torch.from_numpy(st["avail"])
    return ent, typ, cur, mask, glob_, n_own, avail


class _F32(torch.nn.Module):
    """A layer in fp32 inside the bf16 region (autocast off, its input cast up)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return self.m(x.float())


def _f32_method(fn):
    def run(*args, **kw):
        with torch.autocast("cpu", enabled=False):
            return fn(*map(_flt, args), **{k: _flt(v) for k, v in kw.items()})
    return run


def split_variant(net: FullGameNet, kind: str) -> FullGameNet:
    """A copy of `net` with parts of it in fp32 inside a bf16 region (prototypes of the precision splits): "value" its
    value head; "out" each head's last layer (the order, x and y logits, the value) and the pointer's product; "heads"
    every head (the order MLP, the target's conditioning, query, key and pointer, x, y, the value); "" none."""
    import copy
    import math
    net = copy.deepcopy(net)
    if kind in ("value", "heads"):
        net.value, net.value_parts = _f32_method(net.value), _f32_method(net.value_parts)
    if kind == "heads":
        net.order_logits, net.target_logits = _f32_method(net.order_logits), _f32_method(net.target_logits)
        net.y_logits = _f32_method(net.y_logits)
    if kind == "out":
        net.order[2], net.point_x, net.point_y[2], net.value_head[2] = (_F32(m) for m in (net.order[2], net.point_x,
                                                                                         net.point_y[2], net.value_head[2]))
        from warcraftsim.fullgame.model import NEG

        def target_logits(g, u, mask, order):
            O = order.shape[1]
            own = u[:, :O]
            z = net.cond(torch.cat([own + net.order_emb(order.long()), g.unsqueeze(1).expand_as(own)], -1))
            t = net.target_temp
            q, k = net.query(z), net.key(u)
            with torch.autocast("cpu", enabled=False):
                ptr = torch.einsum("nod,ned->noe", q.float(), k.float()) / (math.sqrt(net.d) * t)
            return ptr.masked_fill(~mask.unsqueeze(1), NEG), net.point_x(z) / t, z
        net.target_logits = target_logits
    return net


def _q(x) -> str:
    x = np.asarray(x, np.float64)
    if not x.size:
        return "-"
    return f"{x.mean():.1e} / {np.percentile(x, 99):.1e} / {x.max():.1e}"


def mixed(a) -> int:
    """Each inference precision (and split) against fp32 and against the learner's own bf16 pass on the same steps;
    see the docstring above."""
    from warcraftsim.fullgame import model as M
    torch.backends.mha.set_fastpath_enabled(False)  # (the learner trains: no fused encoder layer; nor here)
    configure_for(a.ck)
    base, ck = load(a.ck, "cpu")
    base.eval()
    G, NO, d = base.config["G"], base.config["n_orders"], base.config["d"]
    streams = view_windows(a, ck["vocab"])
    R = len(streams)
    print(f"{Path(a.ck).name}: {R} game sides x {a.steps} steps (after {a.warm} to warm the memory); entities per step "
          f"mean {np.mean([s['n'] for w in streams for s in w]):.0f}, own units "
          f"{np.mean([s['n_own'] for w in streams for s in w]):.1f}", flush=True)
    # the inference candidates: name -> (precision, a prototype split, act's heads_fp32, the entities rounded to float16
    # first as the learner gets them); "bf16-mixed" is --infer-precision bf16-mixed's own path (model.act's heads_fp32)
    specs = {"fp32": ("fp32", "", False, False), "tf32": ("tf32", "", False, False), "bf16": ("bf16", "", False, False),
             "bf16+value32": ("bf16", "value", False, False), "bf16+out32": ("bf16", "out", False, False),
             "bf16+heads32": ("bf16", "heads", False, False), "bf16+heads-tf32": ("bf16+tf32", "heads", False, False),
             "bf16-mixed": ("bf16+tf32", "", True, False), "bf16/f16ent": ("bf16", "", False, True),
             "bf16+value32/f16ent": ("bf16", "value", False, True), "bf16-mixed/f16ent": ("bf16+tf32", "", True, True)}
    cands = [(c, *specs[c][:2]) for c in a.cands.split(",")]
    nets = {name: split_variant(base, split) for name, _, split in cands}
    rows = next(b for b in (8, 16, 24, 32, 48, 64) if b >= R)
    orig_sample = M._gumbel_argmax
    seen: list = []
    force: list = [None]

    def capture(logits, generator=None, shared=None):  # (act's draws: their logits kept, the reference's actions given)
        k = len(seen)
        seen.append(logits.float())
        return orig_sample(logits, generator, shared) if force[0] is None else force[0][k]
    M._gumbel_argmax = capture
    H = {name: torch.zeros(rows, d) for name, _, _ in cands}
    keep = {name: [] for name, _, _ in cands}  # per measured step: (h before it, per-head log-probabilities, value)
    acts_kept = []
    stat = {name: {k: [] for k in ("lp", "lp_o", "lp_p", "lp_x", "kl_o", "kl_p", "kl_x", "kl_y", "dH", "dv", "agree")}
            for name, _, _ in cands}
    big = {"order": [], "ptr": [], "x": []}
    t_start = time.time()
    with torch.no_grad():
        for t in range(a.warm + a.steps):
            sts = [w[t] for w in streams]
            E = next(e for e in (32, 48, 64, 96, fx.MAX_ENT) if e >= max(s["n"] for s in sts))
            O = min(fx.MAX_OWN, E)
            batch = _batch(sts, rows, E, G, NO)
            half = _batch(sts, rows, E, G, NO, half=True) if any(specs[c[0]][3] for c in cands) else None
            gen = torch.Generator().manual_seed(1000 + t)
            ref = None
            for name, prec, _ in cands:
                seen.clear()
                with cudaish(prec):
                    out = M.act(nets[name], *(half if specs[name][3] else batch), H[name], generator=gen,
                                heads_fp32=specs[name][2])
                lg = list(seen)
                if ref is None:  # (the first candidate, fp32: the actions every candidate is scored on)
                    ref = {"out": out, "lg": lg, "acts": [out[k].clone() for k in ("order", "tgt", "bx", "by")]}
                    force[0] = ref["acts"]
                    order = out["order"]
                    up, upt = nets[name].uses(order)
                    own = (torch.arange(O)[None] < batch[5][:, None])[:R]
                if t >= a.warm:
                    lo = [torch.log_softmax(x[:R], -1) for x in lg]
                    lr = [torch.log_softmax(x[:R], -1) for x in ref["lg"]]
                    ac = [x[:R] for x in ref["acts"]]
                    lpo, lro = (z[0].gather(-1, ac[0][..., None])[..., 0] for z in (lo, lr))
                    lpp, lrp = (z[1].gather(-1, ac[1][..., None])[..., 0] for z in (lo, lr))
                    lpx, lrx = (z[2].gather(-1, ac[2][..., None])[..., 0] + z[3].gather(-1, ac[3][..., None])[..., 0]
                                for z in (lo, lr))
                    keep[name].append((H[name][:R].clone(), out["logp"][:R].float(), out["value"][:R].float(), lpo, lpp,
                                       lpx))
                    ok = own & (lro > -50)
                    s = stat[name]
                    s["lp"].append((out["logp"][:R] - ref["out"]["logp"][:R]).abs()[ok])
                    s["lp_o"].append((lpo - lro).abs()[ok])
                    s["lp_p"].append((lpp - lrp).abs()[ok & up[:R]])
                    s["lp_x"].append((lpx - lrx).abs()[ok & upt[:R]])

                    def kl(i):
                        p = lr[i].exp()
                        return (p * (lr[i] - lo[i])).nan_to_num(0.0).sum(-1)
                    s["kl_o"].append(kl(0)[own])
                    s["kl_p"].append(kl(1)[own & up[:R]])
                    s["kl_x"].append(kl(2)[own & upt[:R]])
                    s["kl_y"].append(kl(3)[own & upt[:R]])
                    ent = lambda z: -(z.exp() * z).nan_to_num(0.0).sum(-1)  # noqa: E731
                    s["dH"].append((ent(lo[0]) - ent(lr[0])).abs()[own])
                    s["dv"].append((out["value"][:R].float() - ref["out"]["value"][:R].float()).abs())
                    s["agree"].append((lo[0].argmax(-1) == lr[0].argmax(-1))[own].float())
                    if name == cands[0][0]:
                        allowed = lr[0] > -1e6
                        big["order"].append(lg[0][:R][own[..., None].expand_as(allowed) & allowed].abs())
                        big["ptr"].append(lg[1][:R][(own & up[:R])[..., None] & batch[3][:R, None, :]].abs())
                        big["x"].append(lg[2][:R][own & upt[:R]].abs().reshape(-1))
                H[name][:R] = out["h"][:R].float()
            force[0] = None
            if t >= a.warm:
                acts_kept.append(([x[:R].clone() for x in ref["acts"]], O))
            if t % 50 == 0:
                print(f"  step {t}: {time.time() - t_start:.0f} s", flush=True)
    M._gumbel_argmax = orig_sample
    print(f"\ninference candidates against fp32 acting (the same actions; {time.time() - t_start:.0f} s)", flush=True)
    print("candidate | |d logp| a unit, mean / p99 / max | order | pointer | point (x+y) | KL order mean / p99 | KL pointer"
          " | KL x | KL y | |d entropy| order | |d value| mean / p99 / max | top order the same")
    for name, _, _ in cands[1:]:
        s = {k: torch.cat(v).double().numpy() for k, v in stat[name].items()}
        print(f"{name} | {_q(s['lp'])} | {_q(s['lp_o'])} | {_q(s['lp_p'])} | {_q(s['lp_x'])} | {s['kl_o'].mean():.1e} / "
              f"{np.percentile(s['kl_o'], 99):.1e} | {s['kl_p'].mean():.1e} | {s['kl_x'].mean():.1e} | {s['kl_y'].mean():.1e} | "
              f"{_q(s['dH'])} | {_q(s['dv'])} | {100 * s['agree'].mean():.3f}%", flush=True)
    for k, v in big.items():
        v = torch.cat(v).double().numpy()
        print(f"fp32 {k} logits (allowed, own units): |logit| p50 {np.percentile(v, 50):.1f} p99 {np.percentile(v, 99):.1f} "
              f"max {v.max():.1f}; >= 8: {100 * (v >= 8).mean():.2f}%, >= 16: {100 * (v >= 16).mean():.3f}% "
              f"(bf16's spacing there: 0.0625, 0.125)")
    # the learner's pass on the same steps and actions: evaluate() on sequences of seq_len steps from each candidate's
    # own recorded memory state (the actors record it), its entities float16, minibatches of 16 sequences, padded
    # to the update's widths (--update-widths)
    widths = (32, 40, 48, 56, 64, 80, 96, 128, fx.MAX_ENT)
    learners = [("learner bf16", base, "bf16", True, 0.0)]  # (name, network, precision, float16 entities, jitter)
    if a.learner_heads:
        learners += [("learner bf16, float32 entities (diagnostic)", base, "bf16", False, 0.0),
                     ("learner bf16, heads fp32 (hypothetical)", split_variant(base, "heads"), "bf16", True, 0.0)]
    if a.jitter:
        learners += [(f"learner bf16, its sums moved by {a.jitter:.0e} (another kernel's order)", base, "bf16", True,
                      a.jitter)]
    seqs = [(r, s0) for s0 in range(0, a.steps, a.seq) for r in range(R)]
    print(f"\nPPO's log ratio at the update's first minibatch, numbers alone (no lag): the learner's log-probability "
          f"against the candidate's, own units, {len(seqs)} sequences of {a.seq}", flush=True)
    print("learner | candidate | |log ratio| mean / p99 / max | order | pointer | point | KL estimate ((r-1)-log r) | "
          "|r-1| > 0.2 | masked one side only | |value learner - candidate| mean / p99")
    for lname, lnet, lprec, half, jitter in learners:
        lnet.train()
        for name, _, _ in cands:
            if lname != learners[0][0] and name not in a.second.split(","):
                continue
            got = {k: [] for k in ("lr", "lo", "lp", "lx", "k3", "clip", "dv")}
            masked = 0
            for m0 in range(0, len(seqs), 16):
                part = seqs[m0:m0 + 16]
                steps = [streams[r][a.warm + s0 + i] for r, s0 in part for i in range(a.seq)]
                E = next(w for w in widths if w >= max(s["n"] for s in steps))
                O = min(fx.MAX_OWN, E)
                batch = _batch(steps, len(steps), E, G, NO, half=half)
                acts = [torch.zeros(len(steps), O, dtype=torch.long) for _ in range(4)]
                lpc, vc = torch.zeros(len(steps), O), torch.zeros(len(steps))
                heads = [torch.zeros(len(steps), O) for _ in range(3)]  # (the candidate's order, pointer, point)
                for j, (r, s0) in enumerate(part):
                    for i in range(a.seq):
                        (ac, Oc), (_, lp_, v_, *hd) = acts_kept[s0 + i], keep[name][s0 + i]
                        n, k = steps[j * a.seq + i]["n_own"], j * a.seq + i
                        for x, y in zip(acts + heads, ac + hd):
                            x[k, :n] = y[r, :n]
                        lpc[k, :n], vc[k] = lp_[r, :n], v_[r]
                h0 = torch.stack([keep[name][s0][0][r] for r, s0 in part])
                with torch.no_grad(), cudaish(lprec, jitter):
                    ev = evaluate(lnet, *batch[:6], *acts, batch[6], seq=(len(part), a.seq, h0, None))
                own = torch.arange(O)[None] < batch[5][:, None]
                lpl = ev["logp"].float()
                both = own & (lpl > -1e6) & (lpc > -1e6)
                masked += int((own & ((lpl > -1e6) != (lpc > -1e6))).sum())
                lr_ = (lpl - lpc)[both]
                got["lr"].append(lr_.abs())
                up, upt = lnet.uses(acts[0])
                lo = M._logp(ev["logits"], acts[0])
                lp_p = M._logp(ev["ptr"], acts[1])
                lp_x = M._logp(ev["xl"], acts[2]) + M._logp(ev["yl"], acts[3])
                got["lo"].append((lo - heads[0])[both].abs())
                got["lp"].append((lp_p - heads[1])[both & up].abs())
                got["lx"].append((lp_x - heads[2])[both & upt].abs())
                r_ = lr_.double().exp()
                got["k3"].append((r_ - 1) - lr_.double())
                got["clip"].append(((r_ - 1).abs() > 0.2).double())
                got["dv"].append((ev["value"].float() - vc).abs())
            s = {k: torch.cat(v).double().numpy() for k, v in got.items()}
            print(f"{lname} | {name} | {_q(s['lr'])} | {_q(s['lo'])} | {_q(s['lp'])} | {_q(s['lx'])} | {s['k3'].mean():.1e} | "
                  f"{100 * s['clip'].mean():.3f}% | "
                  f"{masked} | {s['dv'].mean():.1e} / {np.percentile(s['dv'], 99):.1e}", flush=True)
    return 0


def main() -> int:
    global RUNS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["precision", "flops", "mixed"])
    ap.add_argument("--run", help="precision: the run whose network made runs/bench/ref5-* (fgself-148); flops: a run with "
                                  "inference.jsonl (fgself-182)")
    ap.add_argument("--every", type=int, default=3, help="precision: every this many recorded steps")
    ap.add_argument("--runs", type=Path, default=RUNS, help="the runs folder (a worktree has none: the main tree's)")
    ap.add_argument("--ck", help="mixed: the checkpoint (default: fgself-182's 0000000050150172.pt)")
    ap.add_argument("--demos", help="mixed: recorded games (default: runs/fullgame/demos-echo-1)")
    ap.add_argument("--streams", type=int, default=12, help="mixed: game sides (an inference call's rows)")
    ap.add_argument("--steps", type=int, default=256, help="mixed: measured steps of each")
    ap.add_argument("--warm", type=int, default=32, help="mixed: steps before them (the memory warms up)")
    ap.add_argument("--seq", type=int, default=16, help="mixed: the learner's sequence length (--seq-len)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cands", default="fp32,tf32,bf16,bf16+value32,bf16+out32,bf16+heads32,bf16+heads-tf32",
                    help="mixed: the inference candidates (fp32 first: the reference and the actions)")
    ap.add_argument("--learner-heads", action="store_true", help="mixed: also the learner with float32 entities, and "
                                                                   "with its heads in fp32")
    ap.add_argument("--second", default="fp32,bf16,bf16+heads32,bf16+heads-tf32",
                    help="mixed: the candidates those other learners are compared with")
    ap.add_argument("--jitter", type=float, default=0.0, help="mixed: also a learner whose bf16 multiplies' fp32 sums move "
                                                              "by up to this much (relative) before rounding: the GPU's "
                                                              "kernels sum in other orders for other shapes (e.g. 2.4e-7)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    RUNS = a.runs
    a.run = a.run or ("fgself-148" if a.what == "precision" else "fgself-182")
    a.ck = a.ck or str(RUNS / "fgself-182" / "checkpoints" / "0000000050150172.pt")
    a.demos = a.demos or str(RUNS / "fullgame" / "demos-echo-1")
    torch.set_num_threads(a.threads)
    return precision(a) if a.what == "precision" else flops(a) if a.what == "flops" else mixed(a)


if __name__ == "__main__":
    raise SystemExit(main())
