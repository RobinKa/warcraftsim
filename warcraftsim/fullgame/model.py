"""The whole-game policy: a transformer over what a player sees, orders for each own unit.

Tokens: a global token (resources, supply, time, races, upgrades) and one per entity (its
features, unit type and current order). For each own unit the order head picks an order class
(0: none), restricted to the orders its unit type gave in the demonstrations; given the order, a
pointer picks the target entity, and two heads pick the target point's x and y bins.

With `memory` a minGRU core (Feng et al. 2024, as PufferLib's) carries a state across steps (what
it saw that is out of view now, what it did): h_t = (1 - z_t) h_{t-1} + z_t W x_t, with its gate z_t
from the step's global token x_t alone, so a sequence's states come from a parallel scan (a GRU
stepped in Python took as long as the transformer: its kernels are tiny). The heads read the token
plus a projection of the state (zero at first: a memory network starts as one without).
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn

from . import features as fx

NEG = -1e9


def _order_temp_default(state, prefix, *_):
    state.setdefault(prefix + "order_temp", torch.tensor(1.0))


class FullGameNet(nn.Module):
    def __init__(self, n_types: int, n_cur: int, n_orders: int, G: int, d: int = 192, heads: int = 4,
                 layers: int = 3, dropout: float = 0.0, memory: bool = False, target_temperature: float = 1.0,
                 priv: int = 0, order_temperature: float = 1.0, bins: int = 128, z: int = 0, gx: int = 0,
                 valid: int = 0, vparts: int = 0, ghost: bool = False, privileged: bool = False):
        super().__init__()
        self.config = dict(n_types=n_types, n_cur=n_cur, n_orders=n_orders, G=G, d=d, heads=heads, layers=layers,
                           dropout=dropout, memory=memory, target_temperature=target_temperature, priv=priv,
                           order_temperature=order_temperature, bins=bins, **({"z": z} if z else {}),
                           **({"gx": gx} if gx else {}), **({"valid": valid} if valid else {}),
                           **({"vparts": vparts} if vparts else {}), **({"ghost": True} if ghost else {}),
                           **({"privileged": True} if privileged else {}))
        # the global vector's last `priv` numbers: the whole game (features.View.priv), for the value alone
        self.priv = int(priv)
        # the target heads' logits (unit, x, y) divided by it, acting and training alike: < 1 samples a unit's
        # target closer to its most likely one (AlphaStar samples at 0.8; a clone of a deterministic script
        # spreads its targets from uncertainty, and a group's points scattered)
        self.register_buffer("order_temp", torch.tensor(float(order_temperature)))  # (the order head's, alike;
        self._register_load_state_dict_pre_hook(_order_temp_default)                # checkpoints from before it: 1)
        self.register_buffer("target_temp", torch.tensor(float(target_temperature)))  # (a buffer: CUDA graphs read
        # it from memory, and a past snapshot copied into an inference slot brings its own)
        self.memory = memory
        if memory:  # the minGRU core: its gate and candidate from the token, the state into the context
            self.mem_in = nn.Linear(d, 2 * d)
            self.mem_out = nn.Linear(d, d)
            nn.init.zeros_(self.mem_out.weight)
            nn.init.zeros_(self.mem_out.bias)
        self.d = d
        self.ent = nn.Linear(fx.F, d)
        self.type_emb = nn.Embedding(n_types, d)
        self.cur_emb = nn.Embedding(n_cur, d)
        self.glob = nn.Sequential(nn.Linear(G - self.priv, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.order = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, n_orders))
        self.order_emb = nn.Embedding(n_orders, d)
        self.cond = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.LayerNorm(d))
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.bins = bins  # point targets' bins (features.BINS of the map it plays: scripts/grow_bins.py)
        self.point_x = nn.Linear(d, bins)
        self.x_emb = nn.Embedding(bins, d)  # y is chosen given x (sampled apart, they paired the
        self.point_y = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, bins))  # x of one spot with the y of another)
        # which orders each unit type got in the demonstrations (filled while training; row 0:
        # unknown types, any)
        self.register_buffer("allowed", torch.zeros(n_types, n_orders, dtype=torch.bool))
        # the value of the state for reinforcement learning (fullgame/selfplay.py; behavior cloning
        # leaves it untrained)
        # (--reward-parts: one value per reward, outcome, shaping, strategy; the value is their sum)
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, max(1, int(vparts))))
        if privileged:  # (a privileged view, the global vector's last flag: the policy reads the privileged block too;
            # zero at first)
            self.priv_pol = nn.Sequential(nn.Linear(self.priv, d), nn.ReLU(), nn.Linear(d, d))
            nn.init.zeros_(self.priv_pol[2].weight)
            nn.init.zeros_(self.priv_pol[2].bias)
        if self.priv:  # (added to the value head's input; zero at first: the value as before)
            self.priv_proj = nn.Sequential(nn.Linear(self.priv, d), nn.ReLU(), nn.Linear(d, d))
            nn.init.zeros_(self.priv_proj[2].weight)
            nn.init.zeros_(self.priv_proj[2].bias)
        # each order class's kind (features.IMMEDIATE, POINT, ...: which target it takes)
        self.register_buffer("order_kind", torch.zeros(n_orders, dtype=torch.long))
        # (inputs 3) the attention's bias on the enemies it remembers (features.A_MEM): -30 hides them (a network from
        # before them plays as it did); --memory-fade raises the learner's to 0 over its first updates
        if ghost:
            self.register_buffer("ghost_bias", torch.tensor(-30.0))
        if valid:  # (--valid-mask) what each order needs of the unit (herodata.order_needs; set_valid fills them):
            self._valid_buffers(n_types, valid)  # `valid` the number of order classes that need something

    def _valid_buffers(self, n_types: int, k: int) -> None:
        self.register_buffer("valid_cols", torch.zeros(k, dtype=torch.long))  # the order classes that need something
        self.register_buffer("valid_need", torch.full((n_types, k), -1, dtype=torch.int8))  # per unit type: what

    def set_valid(self, vocab: dict) -> None:
        """The mask of a hero's orders it can't carry out now (--valid-mask; features.A_HERO's bits, inputs 2):
        from the game's tables (herodata.order_needs)."""
        from .herodata import order_needs
        needs = order_needs(vocab)
        cols = sorted({c for _, c in needs})
        n_types = self.type_emb.num_embeddings
        dev = self.order_kind.device
        need = torch.full((n_types, max(len(cols), 1)), -1, dtype=torch.int8)
        at = {c: i for i, c in enumerate(cols)}
        for (t, c), v in needs.items():
            if t < 0:
                need[:, at[c]] = v
            elif t < n_types:
                need[t, at[c]] = v
        self._valid_buffers(n_types, need.shape[1])
        self.valid_cols.copy_(torch.as_tensor(cols or [0]))
        self.valid_need.copy_(need)
        self.valid_cols, self.valid_need = self.valid_cols.to(dev), self.valid_need.to(dev)
        self.config["valid"] = need.shape[1]

    def unit_valid(self, ent, typ):
        """[N, O, K] bool: whether each of the first O units can carry out each of the order classes that need
        something of it (valid_cols): the bits its A_HERO features give (ready per ability slot, learned, a skill
        point it may spend there, a usable item per inventory slot)."""
        h, af, it = fx.A_HERO, fx.HERO_ABIL_F, fx.A_HERO + 4 * fx.HERO_ABIL_F
        e = ent[:, :typ.shape[1]] if ent.shape[1] > typ.shape[1] else ent
        hidden = 9 + fx.FLAG_BITS.index(16)  # (UnitFlags.HIDDEN: a worker inside a gold mine)
        bits = torch.cat([e[..., h + 1:h + 4 * af:af] > 0.5, e[..., h:h + 4 * af:af] > 0.1,
                          e[..., h + 3:h + 4 * af:af] > 0.5, e[..., it + 1:it + 6 * fx.HERO_ITEM_F:fx.HERO_ITEM_F] > 0.5,
                          e[..., hidden:hidden + 1] < 0.5], -1)
        need = self.valid_need[typ.long()]
        ok = torch.ones(need.shape, dtype=torch.bool, device=need.device)
        for j in range(bits.shape[-1]):
            ok &= ~((need == j) & ~bits[..., j:j + 1])
        return ok

    def encode(self, ent, typ, cur, mask, glob):
        """-> global token [N, d], entity tokens [N, E, d]."""
        x = self.ent(ent) + self.type_emb(typ) + self.cur_emb(cur)
        # (what the player knows, and the strategy block of a network with one: config G less the privileged
        # block; one without reads only what it knows when a run's rows have a strategy block)
        g0 = self.glob(glob[..., :self.config["G"] - self.priv])
        if self.config.get("privileged"):  # (the flag: the last number before the privileged block)
            at = self.config["G"] - self.priv
            g0 = g0 + glob[..., at - 1:at].to(g0.dtype) * self.priv_pol(glob[..., at:].to(g0.dtype))
        tok = torch.cat([g0.unsqueeze(1), x], 1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        if self.config.get("ghost") and ent.shape[-1] > fx.A_MEM:  # (the remembered enemies: their attention bias)
            ghost = torch.cat([torch.zeros_like(mask[:, :1]), ent[..., fx.A_MEM] > 0.5], 1)
            zero = torch.zeros((), dtype=x.dtype, device=x.device)
            pad = torch.where(pad, torch.full_like(zero, float("-inf")),
                              torch.where(ghost, self.ghost_bias.to(x.dtype), zero))
        out = self.transformer(tok, src_key_padding_mask=pad)
        return out[:, 0], out[:, 1:]

    def context(self, g, h=None):
        """One step: the heads' context from its global token g [N, d] and, with memory, the state
        after the last step h ([N, d]; None: an episode's start) -> (context [N, d], new state)."""
        if not self.memory:
            return g, None
        z, cand = self.mem_in(g).float().chunk(2, -1)
        z = torch.sigmoid(z)
        h = z * cand if h is None else (1 - z) * h.float() + z * cand
        return g + self.mem_out(h.to(g.dtype)), h

    def context_seq(self, g, h0=None, starts=None):
        """Sequences: tokens g [B, T, d] -> (contexts [B, T, d], states [B, T, d]), from states h0
        [B, d] (None: zeros); where starts [B, T] marks an episode's start the state resets."""
        if not self.memory:
            return g, None
        z, cand = self.mem_in(g).float().chunk(2, -1)
        z = torch.sigmoid(z)
        a, b = 1 - z, z * cand  # h_t = a_t h_{t-1} + b_t
        if starts is not None:
            a = a.masked_fill(starts[..., None], 0.0)
        if h0 is not None:
            b = torch.cat([b[:, :1] + a[:, :1] * h0.float()[:, None], b[:, 1:]], 1)
        k, T = 1, g.shape[1]
        while k < T:  # Hillis-Steele: after the pass for k, each (a, b) spans the 2k steps up to it
            b = torch.cat([b[:, :k], b[:, k:] + a[:, k:] * b[:, :-k]], 1)
            a = torch.cat([a[:, :k], a[:, k:] * a[:, :-k]], 1)
            k *= 2
        return g + self.mem_out(b.to(g.dtype)), b

    def order_logits(self, g, u, typ, n_own, by_type: bool = True, avail=None, ent=None):
        """[N, O, n_orders] for the first O entities (own units first; others can only get none).
        `by_type`: only the orders each unit type got in the demonstrations (playing; training
        learns it: a mask there made unseen pairs, e.g. of upgraded buildings, infinitely wrong).
        `ent` (a network with "valid"): its units' features, for their hero orders (unit_valid)."""
        O = u.shape[1]
        own = u[:, :O]
        logits = self.order(torch.cat([own, g.unsqueeze(1).expand_as(own)], -1)) / self.order_temp
        is_own = torch.arange(O, device=u.device)[None] < n_own[:, None]
        allowed = is_own.unsqueeze(-1).expand_as(logits).clone()
        if by_type:
            allowed &= self.allowed[typ[:, :O].long()]
        if avail is not None:  # [N, n_orders]: what the player can pay for now
            allowed &= avail[:, None, :].bool()
        if ent is not None and self.config.get("valid"):
            allowed[..., self.valid_cols] &= self.unit_valid(ent[:, :O], typ[:, :O])
        allowed[..., 0] = True  # no order: always
        return logits.masked_fill(~allowed, NEG)

    def target_logits(self, g, u, mask, order):
        """Given each own unit's order [N, O]: pointer logits [N, O, E], the point's x logits
        [N, O, BINS], and z (for y_logits)."""
        O = order.shape[1]
        own = u[:, :O]
        z = self.cond(torch.cat([own + self.order_emb(order.long()), g.unsqueeze(1).expand_as(own)], -1))
        t = self.target_temp
        ptr = torch.einsum("nod,ned->noe", self.query(z), self.key(u)) / (math.sqrt(self.d) * t)
        ptr = ptr.masked_fill(~mask.unsqueeze(1), NEG)
        return ptr, self.point_x(z) / t, z

    def y_logits(self, z, x):
        """The point's y logits [N, O, BINS] given its x bins [N, O]."""
        return self.point_y(z + self.x_emb(x.clamp(min=0).long())) / self.target_temp

    def value_parts(self, g, glob=None):
        """[N, parts] (one: the value)."""
        if self.priv and glob is not None:  # (the privileged block: what the player does not see)
            g = g + self.priv_proj(glob[..., -self.priv:].to(g.dtype))
        return self.value_head(g)

    def value(self, g, glob=None):
        return self.value_parts(g, glob).sum(-1)

    def uses(self, order):
        """Which targets each order uses: (pointer [N, O], point [N, O]) booleans."""
        kind = self.order_kind[order.long()]
        return kind == fx.UNIT, (kind == fx.POINT) | (kind == fx.TREE)


def arch_key(config: dict) -> str:
    """Networks that can share an inference slot (their weights copied in): the same shapes; the target
    temperature is a buffer and comes with the weights, and dropout does nothing at inference."""
    return json.dumps({k: v for k, v in config.items() if k not in ("target_temperature", "order_temperature", "dropout")},
                      sort_keys=True,
                      default=str)


def _logp(logits, x):
    return torch.log_softmax(logits.float(), -1).gather(-1, x.long().unsqueeze(-1)).squeeze(-1)


def _gumbel_argmax(logits, generator=None, shared=None):
    """A sample from softmax(logits): the argmax of logits plus Gumbel noise (fewer kernels and no
    Python bookkeeping, unlike torch.distributions; the network calls are latency-bound). `generator`:
    a random generator of the caller's (an inference thread next to a learner's: graph captures).
    `shared` [N] (bool): those rows' units [N, O, C] draw with one noise vector for all of them (common
    random numbers): each unit's own distribution is unchanged, but units that want about the same
    choose the same (one target point for a group: with a draw each, a group's points scattered)."""
    u = torch.rand(logits.shape, device=logits.device, generator=generator)
    if shared is not None:
        one = torch.rand((logits.shape[0], 1) + tuple(logits.shape[2:]), device=logits.device, generator=generator)
        u = torch.where(shared.view((-1,) + (1,) * (logits.dim() - 1)), one.expand_as(u), u)
    u = u.clamp_(1e-10, 1.0 - 1e-7)
    return (logits.float() - torch.log(-torch.log(u))).argmax(-1)


def _entropy(logits):
    lp = torch.log_softmax(logits.float(), -1)
    return -(lp.exp() * lp).sum(-1)


def act(net: FullGameNet, ent, typ, cur, mask, glob, n_own, avail=None, h=None, generator=None, shared=None,
        guide=None) -> dict:
    """Sample every own unit's order and targets (the first O = min(MAX_OWN, E) entities; the
    orders each unit type got in the demonstrations). -> order, tgt, bx, by, logp [N, O] (the
    action's log-probability per unit: order, plus the targets that order uses; 0 for padding)
    and value [N]; with memory the core's new state "h" [N, d] (`h`: the last one, None at a start).
    `shared` [N] (bool): those rows sample their units' targets (unit, x, y) with one noise vector for all of
    them (_gumbel_argmax); the order still with one each.
    `guide` [N] (1: none), classifier-free guidance by the target strategy (as in image generation): the order's
    logits also without the strategy (its block of the global vector zeroed: the policy trains both ways, a
    quarter of its games with no target) and mixed as free + guide * (with - free); > 1 follows the target
    harder. The targets (unit, x, y), the value and the memory come from the pass with the strategy."""
    h_in = h
    g, u = net.encode(ent, typ, cur, mask, glob)
    g, h = net.context(g, h)
    O = min(fx.MAX_OWN, u.shape[1])
    logits = net.order_logits(g, u[:, :O], typ, n_own, avail=avail, ent=ent)
    if guide is not None and net.config.get("z"):
        z0 = net.config["G"] - net.priv - net.config.get("gx", 0) - net.config["z"]
        blank = glob.clone()
        blank[:, z0:z0 + net.config["z"]] = 0.0
        g0, u0 = net.encode(ent, typ, cur, mask, blank)
        g0, _ = net.context(g0, h_in)
        free = net.order_logits(g0, u0[:, :O], typ, n_own, avail=avail, ent=ent)
        w = guide.to(logits.dtype)[:, None, None]
        logits = torch.where(w != 1, free + w * (logits - free), logits)
    def sample(lg):
        return _gumbel_argmax(lg, generator, shared)
    # (the order itself with a draw of its own per unit: one draw for all made a side's similar units take a rare
    # order together or not at all: every worker builds a farm, or none does; the A/B of 2026-10-12)
    order = _gumbel_argmax(logits, generator)
    ptr, xl, z = net.target_logits(g, u, mask, order)
    tgt, bx = sample(ptr), sample(xl)
    yl = net.y_logits(z, bx)
    by = sample(yl)
    uses_ptr, uses_pt = net.uses(order)
    own = torch.arange(O, device=ent.device)[None] < n_own[:, None]
    logp = _logp(logits, order) + uses_ptr * _logp(ptr, tgt) + uses_pt * (_logp(xl, bx) + _logp(yl, by))
    ent = (_entropy(logits) * own).sum(-1) / own.sum(-1).clamp(min=1)  # the order's, mean over own units
    out = {"order": order, "tgt": tgt, "bx": bx, "by": by, "logp": logp * own, "value": net.value(g, glob), "entropy": ent}
    if net.memory:
        out["h"] = h
    return out


def evaluate(net: FullGameNet, ent, typ, cur, mask, glob, n_own, order, tgt, bx, by, avail=None, seq=None) -> dict:
    """Given actions [N, O]: logp [N, O] (as act()), the order distribution's entropy [N, O],
    order logits [N, O, C] and value [N]."""
    g, u = net.encode(ent, typ, cur, mask, glob)
    if net.memory:  # seq: (B, T, h0 [B, d] or None, starts [B, T] or None): the N = B * T steps in order
        B, T, h0, starts = seq
        g = net.context_seq(g.view(B, T, -1), h0, starts)[0].reshape(B * T, -1)
    O = order.shape[1]
    logits = net.order_logits(g, u[:, :O], typ, n_own, avail=avail, ent=ent)
    ptr, xl, z = net.target_logits(g, u, mask, order)
    yl = net.y_logits(z, bx)
    uses_ptr, uses_pt = net.uses(order)
    logp = _logp(logits, order) + uses_ptr * _logp(ptr, tgt) + uses_pt * (_logp(xl, bx) + _logp(yl, by))
    vp = net.value_parts(g, glob)
    return {"logp": logp, "entropy": _entropy(logits), "logits": logits, "value": vp.sum(-1), "g": g, "u": u,
            "ptr": ptr, "xl": xl, "yl": yl, "uses_ptr": uses_ptr, "uses_pt": uses_pt,
            **({"vparts": vp} if net.config.get("vparts") else {})}


def anchor_kl(ev: dict, ev_ref: dict, mode: str = "reverse"):
    """The KL term towards the anchor per unit [N, O]. "reverse": KL(policy || anchor) over the orders (as
    before: its pull on an order is in proportion to the policy's own probability of it, so an order RL
    drives towards 0 is no longer pulled back); "forward": KL(anchor || policy) over the orders (AlphaStar's
    direction: a rare order the anchor gives keeps its weight); "forward-all": and over the target unit
    and the point's x and y (y given the step's x), for the units whose order uses them."""
    def kl(a, b):  # KL(softmax a || softmax b) over the last axis
        la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
        return (la.exp() * (la - lb)).sum(-1)
    if mode == "reverse":
        return kl(ev["logits"], ev_ref["logits"])
    out = kl(ev_ref["logits"], ev["logits"])
    if mode == "forward-all":
        out = out + ev["uses_ptr"] * kl(ev_ref["ptr"], ev["ptr"]) \
            + ev["uses_pt"] * (kl(ev_ref["xl"], ev["xl"]) + kl(ev_ref["yl"], ev["yl"]))
    return out


DERIVED = ("allowed", "valid_cols", "valid_need")  # tables from the game's data and the vocabulary, never trained


def trained(state: dict, net: FullGameNet) -> dict:
    """A checkpoint's weights for a network built since (a resumed run): the tables it derives from the game's
    data (the orders each type may give, neutral.who_buys; the hero orders' needs, set_valid) the network's
    own, as load gives them now, not the checkpoint's."""
    own = net.state_dict()
    return {**state, **{k: own[k] for k in DERIVED if k in own}}


def adapt_config(config: dict) -> dict:
    """A checkpoint's config as this run's inputs read it (features.set_inputs): the clock's numbers in the
    global vector (inputs 2: features.GX, between the strategy block and the privileged one) added."""
    config = dict(config)
    if fx.F > fx.A_MEM and fx.INPUTS >= 3 and not config.get("ghost"):  # (inputs 3: the remembered enemies hidden from
        config["ghost"] = True                                            # a network before them)
    if fx.PRIVILEGED and not config.get("privileged"):  # (the privileged path, zero at first)
        config["privileged"] = True
    gx = config.get("gx", 0)
    if fx.GX > gx:
        config["G"] += fx.GX - gx
        config["gx"] = fx.GX
    z = config.get("z", 0)
    if z and fx.ZDIM > z:  # (--z-modes: the strategy block's modes after its build target's)
        config["G"] += fx.ZDIM - z
        config["z"] = fx.ZDIM
    return config


def configure_for(path) -> None:
    """The features' settings a checkpoint's network was trained with (features.set_inputs, the masks, the strategy
    block, the neutral wares), for a process that plays it alone (versus.py): its config and vocabulary tell."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c, vocab = ck["config"], ck.get("vocab") or {}
    fx.set_inputs(3 if c.get("ghost") else 2 if c.get("gx", 0) >= 3 else 1, bool(c.get("privileged")))
    fx.set_masks(bool(c.get("valid")), bool(c.get("valid")))
    neutral = int(bool(vocab.get("neutral")))
    if neutral and "allowed" in ck.get("model", {}):  # (--neutral 2: no unit type may buy a mercenary)
        from .neutral import MERCENARIES, fourcc
        oix = {tuple(o): k for k, o in enumerate(vocab["orders"], start=1)}
        mercs = [oix[(fourcc(m), fx.UNIT)] for m in MERCENARIES if (fourcc(m), fx.UNIT) in oix]
        if mercs and not bool(ck["model"]["allowed"][:, mercs].any()):
            neutral = 2
    fx.set_neutral(neutral)
    fx.set_macros(int(vocab.get("macros") or 0))
    fx.set_z(c.get("z", 0))
    if c.get("gx", 0) >= 3:  # (the unit channels it reads: all of them, a run's --features zeros only its weights)
        fx.ACTIVE_F = fx.F


def load(path, device="cpu", memory: bool | None = None, vocab: dict | None = None) -> tuple[FullGameNet, dict]:
    """A checkpoint's network. `memory`: with (a new core, which does nothing at first) or without one. Inputs
    added since (features.set_inputs: unit features, the clock) start with zero weights, and with --valid-mask
    (features.VALID_MASK) its hero orders' mask is set (set_valid). `vocab` (--neutral: the run's): its orders in
    that vocabulary's order (neutral.reorder; one extended in another process may list a toggle's two the other
    way round)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    config = dict(ck["config"])
    if memory is not None:
        config["memory"] = memory
    if "vocab" in ck:  # global features added since (the privileged block at the end): the value's, zero at first
        G_now = fx.Encoder(ck["vocab"]).G - fx.GX + config.get("gx", 0)
        if config["G"] < G_now and not config.get("priv"):
            config["priv"], config["G"] = G_now - config["G"], G_now
    gx_old, z_old, G_old = config.get("gx", 0), config.get("z", 0), config["G"]
    config = adapt_config(config)
    valid = bool(config.get("valid"))
    net = FullGameNet(**config).to(device)
    state = ck["model"] if net.memory else {k: v for k, v in ck["model"].items() if not k.startswith("mem_")}
    w = state.get("ent.weight")
    if w is not None and w.shape[1] < fx.F:  # features added since (at the end): their weights start at 0
        state = {**state, "ent.weight": torch.cat([w, w.new_zeros(w.shape[0], fx.F - w.shape[1])], 1)}
    elif w is not None and w.shape[1] > fx.F:  # (a run with fewer inputs: those it has)
        state = {**state, "ent.weight": w[:, :fx.F]}
    w = state.get("glob.0.weight")
    if w is not None and config["G"] > G_old:  # what the policy reads, [known | strategy block | clock]: grown parts zeros
        at = G_old - config.get("priv", 0) - gx_old  # (the strategy block's end)
        dz, dg = config.get("z", 0) - z_old, config.get("gx", 0) - gx_old
        state = {**state, "glob.0.weight": torch.cat([w[:, :at], w.new_zeros(w.shape[0], dz), w[:, at:],
                                                      w.new_zeros(w.shape[0], dg)], 1)}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected or any(not (k.startswith(("value_head.", "mem_", "priv_proj.", "priv_pol.")) or k in ("order_kind", "target_temp", "ghost_bias"))
                         for k in missing):
        raise RuntimeError(f"{path}: missing {missing}, unexpected {unexpected}")
    if "order_kind" in missing and "vocab" in ck:  # fits from before it: from the vocabulary
        net.order_kind.copy_(torch.as_tensor([0] + [k[1] for k in ck["vocab"]["orders"]], device=net.order_kind.device))
    if fx.VALID_MASK and not valid and "vocab" in ck:
        net.set_valid(ck["vocab"])
    if fx.NEUTRAL and "vocab" in ck:  # (--neutral: the neutral buildings' wares as orders, neutral.py)
        from . import neutral
        if not neutral.extended(ck["vocab"]):
            new = neutral.extend_vocab(ck["vocab"])
            net = neutral.grow(net, ck["vocab"], new)
            ck = {**ck, "vocab": new}
        if fx.MACROS and "vocab" in ck:  # (--macros: the agent's own orders, macros.py)
            from . import macros
            if not macros.extended(ck["vocab"]):
                new = macros.extend(ck["vocab"])
                net = macros.grow(net, ck["vocab"], new)
                ck = {**ck, "vocab": new}
            macros.allow(net, ck["vocab"])
        if vocab is not None and ck["vocab"]["orders"] != vocab["orders"]:
            if neutral.reorder(net, ck["vocab"], vocab) is None:
                raise RuntimeError(f"{path}: its vocabulary has other orders than the run's")
            ck = {**ck, "vocab": vocab}
        neutral.who_buys(net, ck["vocab"])  # (and its tables as they are now: networks grown before them too)
        if net.config.get("valid"):
            net.set_valid(ck["vocab"])
    if "vocab" in ck and "allowed" in dict(net.named_buffers()):
        prune_allowed(net, ck["vocab"])
    return net.eval(), ck


@torch.no_grad()
def prune_allowed(net: FullGameNet, vocab: dict) -> int:
    """The orders no unit of a type can give in the game, out of `net`'s allowed table (in place) -> how many: a type
    trains, researches or upgrades to only what the game's tables say it makes (a town hall no castle, an altar no
    research), and a worker builds only its own race's buildings. The table came from the demonstrations, where a
    recycled unit id gave its order to another type (77 such pairs, ~0.3-0.5% of the policy's orders; audit 4)."""
    from .techtree import tree
    makers = tree().makers
    types = {i: fx.rawcode(int(t)) for i, t in enumerate(vocab["types"], start=1)}
    n = 0
    for c, (o, k) in enumerate(vocab["orders"], start=1):
        if o < fx.TYPE_CODE or c >= net.allowed.shape[1]:
            continue
        code = fx.rawcode(int(o))
        if k == fx.IMMEDIATE and code in makers:
            bad = [i for i, t in types.items() if t not in makers[code]]
        elif k == fx.POINT:
            bad = [i for i, t in types.items() if t[:1] in "houe" and t[:1] != code[:1]]
        else:
            continue
        bad = [i for i in bad if i < net.allowed.shape[0] and bool(net.allowed[i, c])]
        if bad:
            net.allowed[bad, c] = False
            n += len(bad)
    return n


def split_value(net: FullGameNet, parts: int) -> FullGameNet:
    """--reward-parts: `net`'s value head with one output per reward (the first is the old value, the others zero:
    their sum plays as before) -> a new network."""
    if net.config.get("vparts", 0) >= parts:
        return net
    config = {**net.config, "vparts": parts}
    new = FullGameNet(**config).to(net.order_kind.device)
    state = net.state_dict()
    w, b = state["value_head.2.weight"], state["value_head.2.bias"]
    state["value_head.2.weight"] = torch.cat([w, w.new_zeros(parts - w.shape[0], w.shape[1])])
    state["value_head.2.bias"] = torch.cat([b, b.new_zeros(parts - b.shape[0])])
    new.load_state_dict(state)
    return new.train(net.training)
