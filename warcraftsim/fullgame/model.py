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
                 layers: int = 3):
        super().__init__()
        self.config = dict(n_types=n_types, n_cur=n_cur, n_orders=n_orders, G=G, d=d, heads=heads, layers=layers)
        self.d = d
        self.ent = nn.Linear(fx.F, d)
        self.type_emb = nn.Embedding(n_types, d)
        self.cur_emb = nn.Embedding(n_cur, d)
        self.glob = nn.Sequential(nn.Linear(G, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.order = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, n_orders))
        self.order_emb = nn.Embedding(n_orders, d)
        self.cond = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.LayerNorm(d))
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.point = nn.Linear(d, 2 * fx.BINS)
        # which orders each unit type may get (from the demonstrations; row 0: unknown types, any)
        self.register_buffer("allowed", torch.ones(n_types, n_orders, dtype=torch.bool))

    def encode(self, ent, typ, cur, mask, glob):
        """-> global token [N, d], entity tokens [N, E, d]."""
        x = self.ent(ent) + self.type_emb(typ) + self.cur_emb(cur)
        tok = torch.cat([self.glob(glob).unsqueeze(1), x], 1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        out = self.transformer(tok, src_key_padding_mask=pad)
        return out[:, 0], out[:, 1:]

    def order_logits(self, g, u, typ, n_own):
        """[N, O, n_orders] for the first O entities (own units first; others masked to none)."""
        O = u.shape[1]
        own = u[:, :O]
        logits = self.order(torch.cat([own, g.unsqueeze(1).expand_as(own)], -1))
        allowed = self.allowed[typ[:, :O].long()]
        is_own = torch.arange(O, device=u.device)[None] < n_own[:, None]
        allowed = allowed & is_own.unsqueeze(-1)
        allowed[..., 0] = True  # no order: always
        return logits.masked_fill(~allowed, NEG)

    def target_logits(self, g, u, mask, order):
        """Given each own unit's order [N, O]: pointer logits [N, O, E], x and y logits [N, O, BINS]."""
        O = order.shape[1]
        own = u[:, :O]
        z = self.cond(torch.cat([own + self.order_emb(order.long()), g.unsqueeze(1).expand_as(own)], -1))
        ptr = torch.einsum("nod,ned->noe", self.query(z), self.key(u)) / math.sqrt(self.d)
        ptr = ptr.masked_fill(~mask.unsqueeze(1), NEG)
        xy = self.point(z)
        return ptr, xy[..., :fx.BINS], xy[..., fx.BINS:]


def load(path, device="cpu") -> tuple[FullGameNet, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = FullGameNet(**ck["config"]).to(device)
    net.load_state_dict(ck["model"])
    return net.eval(), ck
