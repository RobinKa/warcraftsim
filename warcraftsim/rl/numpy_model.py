"""EntityNet in numpy (no torch): for evaluating and playing torch-trainer checkpoints in the venv
(bc eval, video panels, opponents in games).

A checkpoint (.pt) is converted once to .npz next to it by warcraftsim/rl/export.py (run with the
torch Python); `load` does that on demand. The forward pass mirrors model.py exactly (pre-norm
transformer layers, a GRU cell, the autoregressive heads); tests compare the two.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

KINDS = ("noop", "stop", "hold", "move", "attack", "attack_move", "cast")
K_MOVE, K_ATTACK, K_AMOVE, K_CAST = (KINDS.index(k) for k in ("move", "attack", "attack_move", "cast"))
NEG = -1e9


def _sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _log_softmax(x):
    m = x.max(-1, keepdims=True)
    y = x - m
    return y - np.log(np.exp(y).sum(-1, keepdims=True))


def load(path: str | Path) -> "NumpyEntityNet":
    """A checkpoint (.pt, converted to .npz on first use, or .npz)."""
    path = Path(path)
    npz = path.with_suffix(".npz")
    if not npz.exists() or npz.stat().st_mtime < path.stat().st_mtime:
        from ..puffer.bc import _torch_python
        script = Path(__file__).with_name("export.py")
        subprocess.run([_torch_python(), str(script), str(path), str(npz)], check=True, capture_output=True)
    return NumpyEntityNet(npz)


class NumpyEntityNet:
    def __init__(self, npz: str | Path):
        d = np.load(npz, allow_pickle=False)
        self.w = {k: d[k].astype(np.float64) for k in d.files if not k.startswith("__")}
        self.spec = json.loads(str(d["__spec__"]))
        self.config = json.loads(str(d["__config__"]))
        self.steps = int(d["__steps__"])
        self.value_scale = float(d["__value_scale__"]) if "__value_scale__" in d.files else 1.0
        blocks = self.spec["spaces"]["observation"]["blocks"]
        heads = self.spec["spaces"]["actions"]["heads"]
        self.k, self.feat = blocks[0]["rows"], blocks[0]["features"]
        self.F = len(self.feat)
        self.sizes = [h["size"] for h in heads]
        self.per = sum(self.sizes)
        self.d, self.heads_n, self.layers = self.config["d"], self.config["heads"], self.config["layers"]
        self.core_size = self.config["core"]
        n_abil = self.sizes[4]
        self.abil_enemy = [self.feat.index(f"ability {s + 1}: for enemies") for s in range(n_abil)
                           if f"ability {s + 1}: for enemies" in self.feat]
        self.abil_instant = [self.feat.index(f"ability {s + 1}: cast instantly") for s in range(n_abil)
                             if f"ability {s + 1}: cast instantly" in self.feat]
        self.in_F, self.cols = self.F, None  # the observation's unit features (adapt)

    def adapt(self, task_spec: dict) -> "NumpyEntityNet":
        """Play a task whose unit features include this network's (e.g. a policy trained without
        abilities on a task with them): its columns are taken from the task's by name."""
        feat = task_spec["spaces"]["observation"]["blocks"][0]["features"]
        if list(feat) != list(self.feat):
            missing = [f for f in self.feat if f not in feat]
            if missing:
                raise ValueError(f"the task lacks features this network uses: {missing[:4]}")
            self.in_F, self.cols = len(feat), np.array([feat.index(f) for f in self.feat])
        return self

    def _lin(self, name, x):
        y = x @ self.w[name + ".weight"].T
        return y + self.w[name + ".bias"] if name + ".bias" in self.w else y

    def _mlp(self, name, x, n):  # Sequential(Linear, ReLU, Linear[, LayerNorm])
        x = np.maximum(self._lin(f"{name}.0", x), 0)
        x = self._lin(f"{name}.2", x)
        if n == 4:
            x = _ln(x, self.w[f"{name}.3.weight"], self.w[f"{name}.3.bias"])
        return x

    def initial_state(self, n: int = 1) -> np.ndarray:
        return np.zeros((n, self.core_size))

    # ---- encoder -------------------------------------------------------------------------------
    def _attention(self, x, pad, i):
        p = f"transformer.layers.{i}.self_attn"
        N, L, D = x.shape
        H = self.heads_n
        qkv = x @ self.w[p + ".in_proj_weight"].T + self.w[p + ".in_proj_bias"]
        q, k, v = (qkv[..., j * D:(j + 1) * D].reshape(N, L, H, D // H).transpose(0, 2, 1, 3) for j in range(3))
        s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(D // H)
        s = np.where(pad[:, None, None, :], -np.inf, s)
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(N, L, D)
        return o @ self.w[p + ".out_proj.weight"].T + self.w[p + ".out_proj.bias"]

    def encode(self, obs: np.ndarray):
        k, F = self.k, self.in_F
        obs = np.asarray(obs, np.float64)
        N = obs.shape[0]
        own = obs[:, :k * F].reshape(N, k, F)
        own_alive = obs[:, k * F:k * F + k] > 0.5
        at = k * F + k
        enemy = obs[:, at:at + k * F].reshape(N, k, F)
        enemy_alive = obs[:, at + k * F:at + k * F + k] > 0.5
        time = obs[:, at + k * F + k:at + k * F + k + 1]
        if self.cols is not None:  # the task's features -> this network's
            own, enemy = own[..., self.cols], enemy[..., self.cols]
        alive = np.concatenate([own_alive, enemy_alive], 1)
        x = np.concatenate([self._mlp("time_mlp", time, 3)[:, None], self._mlp("unit_mlp", np.concatenate([own, enemy], 1), 3)], 1)
        pad = np.concatenate([np.zeros((N, 1), bool), ~alive], 1)
        for i in range(self.layers):
            p = f"transformer.layers.{i}"
            x = x + self._attention(_ln(x, self.w[p + ".norm1.weight"], self.w[p + ".norm1.bias"]), pad, i)
            y = _ln(x, self.w[p + ".norm2.weight"], self.w[p + ".norm2.bias"])
            x = x + self._lin(p + ".linear2", np.maximum(self._lin(p + ".linear1", y), 0))
        x = _ln(x, self.w["transformer.norm.weight"], self.w["transformer.norm.bias"])
        g, u = x[:, 0], x[:, 1:]
        w_ = alive[..., None].astype(np.float64)
        mo = (u[:, :k] * w_[:, :k]).sum(1) / np.maximum(w_[:, :k].sum(1), 1)
        me = (u[:, k:] * w_[:, k:]).sum(1) / np.maximum(w_[:, k:].sum(1), 1)
        return u, np.concatenate([g, mo, me], -1), own

    def gru(self, x, h):
        H = self.core_size
        gi = x @ self.w["gru.weight_ih"].T + self.w["gru.bias_ih"]
        gh = h @ self.w["gru.weight_hh"].T + self.w["gru.bias_hh"]
        r = _sigmoid(gi[:, :H] + gh[:, :H])
        z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        return (1 - z) * n + z * h

    # ---- orders --------------------------------------------------------------------------------
    def heads(self, u, h, own, masks, actions=None, rng=None, greedy=False):
        """-> actions [N, k, 5], logp [N], value [N], dists (per head [N, k, n])."""
        N, k = u.shape[0], self.k
        z = self._mlp("unit_head", np.concatenate([u[:, :k], np.repeat(h[:, None], k, 1)], -1), 4)
        m = np.asarray(masks).reshape(N, k, self.per) > 0
        cuts = np.cumsum(self.sizes)[:-1]
        mk, md, mdist, mt, ma = np.split(m, cuts, axis=-1)
        rng = rng or np.random.default_rng()
        dists = {}

        def pick(logits, mask, head, given):
            mask = mask | ~mask.any(-1, keepdims=True)
            lp = _log_softmax(np.where(mask, logits, NEG))
            p = np.exp(lp)
            dists[head] = p
            if given is not None:
                a = given
            elif greedy:
                a = lp.argmax(-1)
            else:
                c = p.cumsum(-1)
                a = (rng.random(p.shape[:-1] + (1,)) * c[..., -1:] > c).sum(-1)
            return a, np.take_along_axis(lp, a[..., None], -1)[..., 0]

        g = (lambda i: actions[..., i]) if actions is not None else (lambda i: None)
        kind, lp_k = pick(self._lin("kind", z), mk, 0, g(0))
        z1 = z + self.w["kind_emb.weight"][kind]
        ability, lp_a = pick(self._lin("ability", z1), ma, 4, g(4))
        z2 = z1 + self.w["ability_emb.weight"][ability]
        logits_t = np.einsum("nkd,njd->nkj", self._lin("query", z2), self._lin("key", u)) / math.sqrt(self.d)
        enemy_slot = np.arange(2 * k) >= k
        tmask = np.where((kind == K_ATTACK)[..., None], mt & enemy_slot, mt)
        instant = np.zeros_like(kind, bool)
        if self.abil_enemy:
            for_enemy = np.take_along_axis(own[..., self.abil_enemy], ability[..., None], -1)[..., 0] > 0.5
            side = np.where(for_enemy[..., None], enemy_slot, ~enemy_slot)
            tmask = np.where((kind == K_CAST)[..., None], tmask & side, tmask)
            instant = np.take_along_axis(own[..., self.abil_instant], ability[..., None], -1)[..., 0] > 0.5
        target, lp_t = pick(logits_t, tmask, 3, g(3))
        direction, lp_d = pick(self._lin("direction", z1), md, 1, g(1))
        distance, lp_s = pick(self._lin("distance", z1), mdist, 2, g(2))
        moves = (kind == K_MOVE) | (kind == K_AMOVE)
        aims = (kind == K_ATTACK) | ((kind == K_CAST) & ~instant)
        logp = lp_k + moves * (lp_d + lp_s) + aims * lp_t + (kind == K_CAST) * lp_a
        value = self._mlp("value", h, 3)[:, 0]
        return np.stack([kind, direction, distance, target, ability], -1), logp.sum(-1), value, dists

    def step(self, obs, h, masks, rng=None, greedy=False, actions=None):
        """One step: -> actions [N, k*5], logp, value, next core state, dists."""
        u, x, own = self.encode(obs)
        h = self.gru(x, h)
        acts, logp, v, dists = self.heads(u, h, own, masks, actions=actions, rng=rng, greedy=greedy)
        return acts.reshape(acts.shape[0], -1), logp, v, h, dists


def episode_outputs(net: NumpyEntityNet, obs: np.ndarray, masks: np.ndarray | None, actions: np.ndarray):
    """What the policy thought over an episode, for one agent: values [T] (the bridge's reward
    scale), per flat head (unit-major, as the task's heads) the distribution [T, n] given the
    orders taken before it, and entropies [T, heads]."""
    T = len(obs)
    k, H = net.k, len(net.sizes)
    h = net.initial_state(1)
    values = np.zeros(T)
    probs = [np.zeros((T, n)) for _ in range(k) for n in net.sizes]
    ent = np.zeros((T, k * H))
    ones = np.ones((1, k * net.per), np.uint8)
    for t in range(T):
        m = masks[t][None] if masks is not None else ones
        _, _, v, h, dists = net.step(obs[t][None], h, m, actions=np.asarray(actions[t], np.int64).reshape(1, k, -1))
        values[t] = v[0] * net.value_scale
        for i in range(k):
            for hd in range(H):
                p = dists[hd][0, i]
                probs[i * H + hd][t] = p
                ent[t, i * H + hd] = -(p * np.log(np.maximum(p, 1e-12))).sum()
    return values, probs, ent
