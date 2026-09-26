"""Entity policy for general orders (MicroEnv targeting="general"; tasks mirror_mix_gen*).

Units are tokens: a shared MLP embeds each unit's features, a transformer lets them attend to each
other (dead slots masked), and a GRU core carries memory across steps. Per own unit the orders are
sampled autoregressively, as in AlphaStar:

    kind (noop, stop, hold, move, attack, attack_move, cast)
    -> ability (for cast)
    -> target: a pointer: the unit's query against every unit's key (own slots, then enemy slots)
    -> direction and distance (for move and attack-move)

Each choice is embedded and added to the unit's state before the next one. Masks come from the
environment (what is possible now) plus structure: attacks point at enemies, a cast at the side its
ability is for. A head counts in the action's probability only when the chosen kind uses it.

Standalone: numpy + torch; the task comes as a spec (warcraftsim.rl.spec).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("noop", "stop", "hold", "move", "attack", "attack_move", "cast")
K_MOVE, K_ATTACK, K_AMOVE, K_CAST = KINDS.index("move"), KINDS.index("attack"), KINDS.index("attack_move"), KINDS.index("cast")
NEG = -1e9


class EntityNet(nn.Module):
    def __init__(self, spec: dict, d: int = 128, heads: int = 4, layers: int = 2, core: int = 256):
        super().__init__()
        blocks = spec["spaces"]["observation"]["blocks"]
        names = [b["name"] for b in blocks]
        assert names[:2] == ["own units (slots A0..)", "own slot alive"] and names[2:4] == [
            "enemy units (slots E0..)", "enemy slot alive"], f"not a micro layout: {names}"
        heads_spec = spec["spaces"]["actions"]["heads"]
        assert [h["name"] for h in heads_spec] == ["order", "direction", "distance", "target", "ability"], \
            "EntityNet needs general orders (a _gen task)"
        assert tuple(heads_spec[0]["options"]) == KINDS
        self.k = blocks[0]["rows"]
        self.feat = blocks[0]["features"]
        self.F = len(self.feat)
        self.n_dir, self.n_dist, self.n_tgt, self.n_abil = (h["size"] for h in heads_spec[1:])
        assert self.n_tgt == 2 * self.k
        self.sizes = [h["size"] for h in heads_spec]
        self.per = sum(self.sizes)
        # the ability features that decide where a cast can point (ability k: for enemies / instant)
        self.abil_enemy = [self.feat.index(f"ability {s + 1}: for enemies") for s in range(self.n_abil)
                           if f"ability {s + 1}: for enemies" in self.feat]
        self.abil_instant = [self.feat.index(f"ability {s + 1}: cast instantly") for s in range(self.n_abil)
                             if f"ability {s + 1}: cast instantly" in self.feat]
        self.d, self.core_size = d, core
        self.config = dict(d=d, heads=heads, layers=layers, core=core)

        self.unit_mlp = nn.Sequential(nn.Linear(self.F, d), nn.ReLU(), nn.Linear(d, d))
        self.time_mlp = nn.Sequential(nn.Linear(1, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.gru = nn.GRUCell(3 * d, core)
        self.unit_head = nn.Sequential(nn.Linear(d + core, d), nn.ReLU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.kind = nn.Linear(d, len(KINDS))
        self.kind_emb = nn.Embedding(len(KINDS), d)
        self.ability = nn.Linear(d, self.n_abil)
        self.ability_emb = nn.Embedding(self.n_abil, d)
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.direction = nn.Linear(d, self.n_dir)
        self.distance = nn.Linear(d, self.n_dist)
        self.value = nn.Sequential(nn.Linear(core, d), nn.ReLU(), nn.Linear(d, 1))
        self._keep_dists = False  # heads() records each head's distribution in last_dists (videos)
        self.last_dists: dict = {}

    # ---- observation -> unit states and core ---------------------------------------------------
    def parse(self, obs: torch.Tensor):
        k, F_ = self.k, self.F
        own = obs[:, :k * F_].view(-1, k, F_)
        at = k * F_
        own_alive = obs[:, at:at + k]
        at += k
        enemy = obs[:, at:at + k * F_].view(-1, k, F_)
        at += k * F_
        enemy_alive = obs[:, at:at + k]
        at += k
        time = obs[:, at:at + 1]
        return own, own_alive, enemy, enemy_alive, time

    def encode(self, obs: torch.Tensor):
        """obs [N, obs] -> unit tokens [N, 2k, d], the core's input [N, 3d], alive [N, 2k]."""
        own, own_alive, enemy, enemy_alive, time = self.parse(obs)
        units = torch.cat([own, enemy], 1)
        alive = torch.cat([own_alive, enemy_alive], 1) > 0.5
        seq = torch.cat([self.time_mlp(time).unsqueeze(1), self.unit_mlp(units)], 1)
        pad = torch.cat([torch.zeros_like(alive[:, :1]), ~alive], 1)
        out = self.transformer(seq, src_key_padding_mask=pad)
        g, u = out[:, 0], out[:, 1:]
        w = alive.float().unsqueeze(-1)
        k = self.k
        mean_own = (u[:, :k] * w[:, :k]).sum(1) / w[:, :k].sum(1).clamp(min=1)
        mean_enemy = (u[:, k:] * w[:, k:]).sum(1) / w[:, k:].sum(1).clamp(min=1)
        return u, torch.cat([g, mean_own, mean_enemy], -1)

    def initial_state(self, n: int, device) -> torch.Tensor:
        return torch.zeros(n, self.core_size, device=device)

    # ---- orders --------------------------------------------------------------------------------
    def heads(self, obs, u, h, masks, actions=None, greedy: bool = False):
        """Samples (or scores `actions` [N, k, 5]) the orders of every own unit. Returns the
        actions [N, k, 5], their log-probability [N], entropy [N] and the value [N]."""
        N, k = u.shape[0], self.k
        z = self.unit_head(torch.cat([u[:, :k], h.unsqueeze(1).expand(-1, k, -1)], -1))
        m = masks.view(N, k, self.per) > 0
        mk, md, mdist, mt, ma = torch.split(m, self.sizes, dim=-1)

        def pick(logits, mask, given):
            mask = mask | ~mask.any(-1, keepdim=True)  # never all impossible
            logp = torch.log_softmax(logits.masked_fill(~mask, NEG), -1)
            if given is not None:
                a = given
            elif greedy:
                a = logp.argmax(-1)
            else:
                a = torch.distributions.Categorical(logits=logp).sample()
            ent = -(logp.exp() * logp.masked_fill(~mask, 0)).sum(-1)
            return a, logp.gather(-1, a.unsqueeze(-1)).squeeze(-1), ent

        g = (lambda i: actions[..., i]) if actions is not None else (lambda i: None)
        dists = {}  # head -> its distribution [N, k, n] (conditioned on the earlier choices)
        _pick = pick

        def pick(logits, mask, given, head=None):  # noqa: F811 (records each head's distribution)
            out = _pick(logits, mask, given)
            if head is not None and self._keep_dists:
                m_ = mask | ~mask.any(-1, keepdim=True)
                dists[head] = torch.softmax(logits.masked_fill(~m_, NEG), -1)
            return out
        kind, lp_k, en_k = pick(self.kind(z), mk, g(0), 0)
        z1 = z + self.kind_emb(kind)
        ability, lp_a, en_a = pick(self.ability(z1), ma, g(4), 4)
        z2 = z1 + self.ability_emb(ability)
        logits_t = torch.einsum("nkd,njd->nkj", self.query(z2), self.key(u)) / math.sqrt(self.d)
        # structure: attacks at enemies; a cast at the side its ability is for
        own_units, _, _, _, _ = self.parse(obs)
        enemy_slot = torch.arange(2 * k, device=u.device) >= k
        tmask = mt.clone()
        is_attack, is_cast = kind == K_ATTACK, kind == K_CAST
        tmask = torch.where(is_attack.unsqueeze(-1), tmask & enemy_slot, tmask)
        instant = torch.zeros_like(is_cast)
        if self.abil_enemy:
            for_enemy = own_units[..., self.abil_enemy].gather(-1, ability.unsqueeze(-1)).squeeze(-1) > 0.5
            side = torch.where(for_enemy.unsqueeze(-1), enemy_slot, ~enemy_slot)
            tmask = torch.where(is_cast.unsqueeze(-1), tmask & side, tmask)
            instant = own_units[..., self.abil_instant].gather(-1, ability.unsqueeze(-1)).squeeze(-1) > 0.5
        target, lp_t, en_t = pick(logits_t, tmask, g(3), 3)
        direction, lp_d, en_d = pick(self.direction(z1), md, g(1), 1)
        distance, lp_s, en_s = pick(self.distance(z1), mdist, g(2), 2)

        moves = ((kind == K_MOVE) | (kind == K_AMOVE)).float()
        aims = (is_attack | (is_cast & ~instant)).float()
        casts = is_cast.float()
        logp = lp_k + moves * (lp_d + lp_s) + aims * lp_t + casts * lp_a
        ent = en_k + moves * (en_d + en_s) + aims * en_t + casts * en_a
        acts = torch.stack([kind, direction, distance, target, ability], -1)
        self.last_dists = dists
        return acts, logp.sum(-1), ent.sum(-1), self.value(h).squeeze(-1)

    def step(self, obs, h, masks, greedy: bool = False):
        """One step for a batch: -> actions [N, k, 5], logp, entropy, value, next core state."""
        u, x = self.encode(obs)
        h = self.gru(x, h)
        acts, logp, ent, v = self.heads(obs, u, h, masks, greedy=greedy)
        return acts, logp, ent, v, h

    def evaluate(self, obs, masks, actions, h0, starts):
        """Sequences [T, B, ...] from core states h0 [B] (zeroed where starts [T, B] mark a new
        episode): -> logp, entropy, value [T, B]."""
        T, B = obs.shape[:2]
        u, x = self.encode(obs.reshape(T * B, -1))
        x = x.view(T, B, -1)
        hs, h = [], h0
        for t in range(T):
            h = self.gru(x[t], h * (1 - starts[t]).unsqueeze(-1))
            hs.append(h)
        h = torch.stack(hs).view(T * B, -1)
        _, logp, ent, v = self.heads(obs.reshape(T * B, -1), u, h, masks.reshape(T * B, -1),
                                     actions.reshape(T * B, self.k, -1))
        return logp.view(T, B), ent.view(T, B), v.view(T, B)
