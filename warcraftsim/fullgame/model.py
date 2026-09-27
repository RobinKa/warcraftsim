"""The whole-game policy: a transformer over what a player sees, orders for each own unit.

Tokens: a global token (resources, supply, time, races, upgrades) and one per entity (its
features, unit type and current order). For each own unit the order head picks an order class
(0: none), restricted to the orders its unit type gave in the demonstrations; given the order, a
pointer picks the target entity, and two heads pick the target point's x and y bins.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import features as fx

NEG = -1e9


class FullGameNet(nn.Module):
    def __init__(self, n_types: int, n_cur: int, n_orders: int, G: int, d: int = 192, heads: int = 4,
                 layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.config = dict(n_types=n_types, n_cur=n_cur, n_orders=n_orders, G=G, d=d, heads=heads, layers=layers,
                           dropout=dropout)
        self.d = d
        self.ent = nn.Linear(fx.F, d)
        self.type_emb = nn.Embedding(n_types, d)
        self.cur_emb = nn.Embedding(n_cur, d)
        self.glob = nn.Sequential(nn.Linear(G, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.order = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, n_orders))
        self.order_emb = nn.Embedding(n_orders, d)
        self.cond = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.LayerNorm(d))
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.point_x = nn.Linear(d, fx.BINS)
        self.x_emb = nn.Embedding(fx.BINS, d)  # y is chosen given x (sampled apart, they paired the
        self.point_y = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, fx.BINS))  # x of one spot with the y of another)
        # which orders each unit type got in the demonstrations (filled while training; row 0:
        # unknown types, any)
        self.register_buffer("allowed", torch.zeros(n_types, n_orders, dtype=torch.bool))
        # the value of the state for reinforcement learning (fullgame/selfplay.py; behavior cloning
        # leaves it untrained)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        # each order class's kind (features.IMMEDIATE, POINT, ...: which target it takes)
        self.register_buffer("order_kind", torch.zeros(n_orders, dtype=torch.long))

    def encode(self, ent, typ, cur, mask, glob):
        """-> global token [N, d], entity tokens [N, E, d]."""
        x = self.ent(ent) + self.type_emb(typ) + self.cur_emb(cur)
        tok = torch.cat([self.glob(glob).unsqueeze(1), x], 1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        out = self.transformer(tok, src_key_padding_mask=pad)
        return out[:, 0], out[:, 1:]

    def order_logits(self, g, u, typ, n_own, by_type: bool = True):
        """[N, O, n_orders] for the first O entities (own units first; others can only get none).
        `by_type`: only the orders each unit type got in the demonstrations (playing; training
        learns it: a mask there made unseen pairs, e.g. of upgraded buildings, infinitely wrong)."""
        O = u.shape[1]
        own = u[:, :O]
        logits = self.order(torch.cat([own, g.unsqueeze(1).expand_as(own)], -1))
        is_own = torch.arange(O, device=u.device)[None] < n_own[:, None]
        allowed = is_own.unsqueeze(-1).expand_as(logits).clone()
        if by_type:
            allowed &= self.allowed[typ[:, :O].long()]
        allowed[..., 0] = True  # no order: always
        return logits.masked_fill(~allowed, NEG)

    def target_logits(self, g, u, mask, order):
        """Given each own unit's order [N, O]: pointer logits [N, O, E], the point's x logits
        [N, O, BINS], and z (for y_logits)."""
        O = order.shape[1]
        own = u[:, :O]
        z = self.cond(torch.cat([own + self.order_emb(order.long()), g.unsqueeze(1).expand_as(own)], -1))
        ptr = torch.einsum("nod,ned->noe", self.query(z), self.key(u)) / math.sqrt(self.d)
        ptr = ptr.masked_fill(~mask.unsqueeze(1), NEG)
        return ptr, self.point_x(z), z

    def y_logits(self, z, x):
        """The point's y logits [N, O, BINS] given its x bins [N, O]."""
        return self.point_y(z + self.x_emb(x.clamp(min=0).long()))

    def value(self, g):
        return self.value_head(g).squeeze(-1)

    def uses(self, order):
        """Which targets each order uses: (pointer [N, O], point [N, O]) booleans."""
        kind = self.order_kind[order.long()]
        return kind == fx.UNIT, (kind == fx.POINT) | (kind == fx.TREE)


def _logp(logits, x):
    return torch.log_softmax(logits.float(), -1).gather(-1, x.long().unsqueeze(-1)).squeeze(-1)


def _entropy(logits):
    lp = torch.log_softmax(logits.float(), -1)
    return -(lp.exp() * lp).sum(-1)


def act(net: FullGameNet, ent, typ, cur, mask, glob, n_own) -> dict:
    """Sample every own unit's order and targets (the first O = min(MAX_OWN, E) entities; the
    orders each unit type got in the demonstrations). -> order, tgt, bx, by, logp [N, O] (the
    action's log-probability per unit: order, plus the targets that order uses; 0 for padding)
    and value [N]."""
    g, u = net.encode(ent, typ, cur, mask, glob)
    O = min(fx.MAX_OWN, u.shape[1])
    logits = net.order_logits(g, u[:, :O], typ, n_own)
    sample = lambda lg: torch.distributions.Categorical(logits=lg.float()).sample()  # noqa: E731
    order = sample(logits)
    ptr, xl, z = net.target_logits(g, u, mask, order)
    tgt, bx = sample(ptr), sample(xl)
    yl = net.y_logits(z, bx)
    by = sample(yl)
    uses_ptr, uses_pt = net.uses(order)
    own = torch.arange(O, device=ent.device)[None] < n_own[:, None]
    logp = _logp(logits, order) + uses_ptr * _logp(ptr, tgt) + uses_pt * (_logp(xl, bx) + _logp(yl, by))
    return {"order": order, "tgt": tgt, "bx": bx, "by": by, "logp": logp * own, "value": net.value(g)}


def evaluate(net: FullGameNet, ent, typ, cur, mask, glob, n_own, order, tgt, bx, by) -> dict:
    """Given actions [N, O]: logp [N, O] (as act()), the order distribution's entropy [N, O],
    order logits [N, O, C] and value [N]."""
    g, u = net.encode(ent, typ, cur, mask, glob)
    O = order.shape[1]
    logits = net.order_logits(g, u[:, :O], typ, n_own)
    ptr, xl, z = net.target_logits(g, u, mask, order)
    yl = net.y_logits(z, bx)
    uses_ptr, uses_pt = net.uses(order)
    logp = _logp(logits, order) + uses_ptr * _logp(ptr, tgt) + uses_pt * (_logp(xl, bx) + _logp(yl, by))
    return {"logp": logp, "entropy": _entropy(logits), "logits": logits, "value": net.value(g)}


def load(path, device="cpu") -> tuple[FullGameNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = FullGameNet(**ck["config"]).to(device)
    missing, unexpected = net.load_state_dict(ck["model"], strict=False)
    if unexpected or any(not (k.startswith("value_head.") or k == "order_kind") for k in missing):
        raise RuntimeError(f"{path}: missing {missing}, unexpected {unexpected}")
    if "order_kind" in missing and "vocab" in ck:  # fits from before it: from the vocabulary
        net.order_kind.copy_(torch.as_tensor([0] + [k[1] for k in ck["vocab"]["orders"]], device=net.order_kind.device))
    return net.eval(), ck
