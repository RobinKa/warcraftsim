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
from . import ordermasks as om

NEG = -1e9


def _order_temp_default(state, prefix, *_):
    state.setdefault(prefix + "order_temp", torch.tensor(1.0))


VALID_BITS = 20  # (unit_valid's bits: what herodata.order_needs can ask of a unit, 0..19)
# --value-bins' defaults: the returns' range and the Gaussian's width in bins. A state's return is about
# gamma^(T-t) * outcome - shaping * phi_t (the shaping telescopes: its potential, the material lead over
# --shaping-scale, comes back at the end); fgself-267/268's 51,000 games ended with leads of -10.6..+10.4, values
# -5.2..+4.9 (99.8% within -3.3..+3.5) at --shaping 0.5: [-6, 6] clips none of them
VALUE_MIN, VALUE_MAX, VALUE_SIGMA = -6.0, 6.0, 0.75


class FullGameNet(nn.Module):
    valid_mana: tuple = ()  # (--full-vocab: the mana steps of unit_valid's bits from fullvocab.MANA0 on; config "vmana")

    def __init__(self, n_types: int, n_cur: int, n_orders: int, G: int, d: int = 192, heads: int = 4,
                 layers: int = 3, dropout: float = 0.0, memory: bool = False, target_temperature: float = 1.0,
                 priv: int = 0, order_temperature: float = 1.0, bins: int = 128, z: int = 0, gx: int = 0,
                 valid: int = 0, vparts: int = 0, ghost: bool = False, privileged: bool = False, fine: int = 0,
                 vtax: int = 0, vmana: int = 0, obsfull: int = 0, omask: int = 0, vbins: int = 0,
                 vmin: float = VALUE_MIN, vmax: float = VALUE_MAX, vsigma: float = VALUE_SIGMA):
        super().__init__()
        if obsfull and fx.OBS_FULL != int(obsfull):  # (--obs-full: its block travels in the config, so the features'
            fx.set_obs_full(int(obsfull))             # width is the network's in any process that builds it)
        self.config = dict(n_types=n_types, n_cur=n_cur, n_orders=n_orders, G=G, d=d, heads=heads, layers=layers,
                           dropout=dropout, memory=memory, target_temperature=target_temperature, priv=priv,
                           order_temperature=order_temperature, bins=bins, **({"z": z} if z else {}),
                           **({"gx": gx} if gx else {}), **({"valid": valid} if valid else {}),
                           **({"vparts": vparts} if vparts else {}), **({"ghost": True} if ghost else {}),
                           **({"privileged": True} if privileged else {}), **({"fine": int(fine)} if fine > 1 else {}),
                           **({"vtax": int(vtax)} if vtax else {}), **({"vmana": 1} if vmana else {}),
                           **({"obsfull": int(obsfull)} if obsfull else {}), **({"omask": int(omask)} if omask else {}),
                           **({"vbins": int(vbins), "vmin": float(vmin), "vmax": float(vmax), "vsigma": float(vsigma)}
                              if vbins else {}))
        if vmana:  # (--full-vocab's mask bits travel in the config: the actors' networks are built from it and a state
            # dict, never by set_valid, and a bit only the learner checks is a -1e9 log-probability in the update)
            from .fullvocab import mana_steps
            self.valid_mana = tuple(mana_steps())
        # the global vector's last `priv` numbers: the whole game (features.View.priv), for the value alone; with
        # --value-tax its last `vtax` of them the game's taxes (features.View.tax), read by tax_proj alone
        self.priv = int(priv)
        self.vtax = int(vtax)
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
        # (--value-bins K: HL-Gauss, Farebrother et al. 2024, "Stop Regressing": K logits per part over K bins uniform on
        # [vmin, vmax], trained by cross-entropy on the return's Gaussian-smoothed histogram (hl_gauss_target); the value
        # every consumer reads, the actors' GAE and UPGO among them, is its expectation over the bins' centres, as a
        # scalar per part as before. The centres and edges are derived from the config: buffers, not saved)
        self.vbins = int(vbins)
        if self.vbins == 1:
            raise ValueError("vbins: 0 (a scalar value) or at least 2 bins")
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, max(1, int(vparts)) * max(1, self.vbins)))
        if self.vbins:
            if not float(vmax) > float(vmin) or not float(vsigma) > 0:
                raise ValueError(f"value bins over [{vmin}, {vmax}] with sigma {vsigma}: need vmax > vmin, sigma > 0")
            edges = torch.linspace(float(vmin), float(vmax), self.vbins + 1)
            self.register_buffer("value_edges", edges, persistent=False)
            self.register_buffer("value_centers", (edges[:-1] + edges[1:]) / 2, persistent=False)
            self.value_sigma = float(vsigma) * (float(vmax) - float(vmin)) / self.vbins  # (in the returns' units)
        if privileged:  # (a privileged view, the global vector's last flag: the policy reads the privileged block too;
            # zero at first)
            self.priv_pol = nn.Sequential(nn.Linear(self.priv - self.vtax, d), nn.ReLU(), nn.Linear(d, d))
            nn.init.zeros_(self.priv_pol[2].weight)
            nn.init.zeros_(self.priv_pol[2].bias)
        if self.priv:  # (added to the value head's input; zero at first: the value as before)
            self.priv_proj = nn.Sequential(nn.Linear(self.priv - self.vtax, d), nn.ReLU(), nn.Linear(d, d))
            nn.init.zeros_(self.priv_proj[2].weight)
            nn.init.zeros_(self.priv_proj[2].bias)
        # each order class's kind (features.IMMEDIATE, POINT, ...: which target it takes)
        self.register_buffer("order_kind", torch.zeros(n_orders, dtype=torch.long))
        # (inputs 3) the attention's bias on the enemies it remembers (features.A_MEM): -30 hides them (a network from
        # before them plays as it did); --memory-fade raises the learner's to 0 over its first updates
        if ghost:
            self.register_buffer("ghost_bias", torch.tensor(-30.0))
        # (--obs-full) the attention's bias on each new entity kind (obsfull.KINDS: corpses, items on the ground, area
        # spells): -30 hides them (a network from before them plays as it did); --obs-fade raises the learner's to 0 over
        # its first updates. A buffer: the actors' networks (FullGameNet(**config) and a state dict) get it with the weights
        self.obsfull = int(obsfull)
        if self.obsfull:
            self.register_buffer("kind_bias", torch.full((3,), -30.0))
        if valid:  # (--valid-mask) what each order needs of the unit (herodata.order_needs; set_valid fills them):
            self._valid_buffers(n_types, valid)  # `valid` the number of order classes that need something
        # (--fine-points S: the point's sub-cell within its x and y bins, one of S x S, given both bins: hierarchical
        # refinement, as precise anywhere as the bins are coarse there. Registered last: the other parameters keep
        # their order, Adam's state by position too. grow_fine: the last layer's weights zero, its bias a prior)
        self.fine = int(fine) if fine > 1 else 0
        if self.fine:
            self.fine_x = nn.Embedding(bins, d)
            self.fine_y = nn.Embedding(bins, d)
            self.point_f = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, self.fine * self.fine))
            nn.init.zeros_(self.point_f[2].weight)
            nn.init.zeros_(self.point_f[2].bias)
        if self.vtax:  # (--value-tax: the game's taxes into the value head's input, never the policy's; zero at first,
            self.tax_proj = nn.Linear(self.vtax, d)  # the value as before. Registered last, as the fine head)
            nn.init.zeros_(self.tax_proj.weight)
            nn.init.zeros_(self.tax_proj.bias)
        # (--order-masks 1: each order's pointer rule, the unit types' kinds and the ruled classes (set_order_masks
        # fills them; `omask` their number). The buffers last, as set_order_masks registers them on a network built
        # without: a past snapshot's slot copies its buffers by position)
        self.omask = int(omask)
        if self.omask:
            self._omask_buffers(n_types, n_orders, self.omask)

    def _omask_buffers(self, n_types: int, n_orders: int, k: int) -> None:
        dev = self.order_kind.device
        self.register_buffer("ptr_rule", torch.zeros(n_orders, dtype=torch.int32, device=dev))  # (ordermasks' bits)
        self.register_buffer("type_kind", torch.zeros(n_types, dtype=torch.int32, device=dev))  # (MECH, ANCIENT)
        self.register_buffer("ptr_cols", torch.zeros(k, dtype=torch.long, device=dev))  # (the classes with a rule)

    def set_order_masks(self, vocab: dict) -> None:
        """--order-masks 1 (ordermasks.py): each unit-target spell's and ability's pointer rule from the game's tables,
        and the unit types' kinds, for `vocab` (the network's own orders and types)."""
        T, C = self.type_emb.num_embeddings, self.order_kind.shape[0]
        rules = (om.pointer_rules(vocab) + [0] * C)[:C]
        kinds = (om.type_kinds(vocab) + [0] * T)[:T]
        cols = [c for c, r in enumerate(rules) if r] or [0]
        self._omask_buffers(T, C, len(cols))
        self.ptr_rule.copy_(torch.as_tensor(rules, dtype=torch.int32))
        self.type_kind.copy_(torch.as_tensor(kinds, dtype=torch.int32))
        self.ptr_cols.copy_(torch.as_tensor(cols, dtype=torch.long))
        self.omask = len(cols)
        self.config["omask"] = self.omask

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
        top = VALID_BITS
        if vocab.get("full_vocab"):  # (--full-vocab: an item held per slot, a unit's mana at each step: fullvocab)
            from .fullvocab import MANA0, mana_steps
            self.valid_mana = tuple(mana_steps())
            self.config["vmana"] = 1
            top = MANA0 + len(self.valid_mana)
        if int(need.max()) >= top:
            raise ValueError(f"an order needs bit {int(need.max())}: unit_valid has {top}")

    def unit_valid(self, ent, typ):
        """[N, O, K] bool: whether each of the first O units can carry out each of the order classes that need
        something of it (valid_cols): the bits its A_HERO features give (ready per ability slot, learned, a skill
        point it may spend there, a usable item per inventory slot, not hidden, a free inventory slot).
        Every bit, whatever the network's valid_need asks: the free slot's had been checked only by a network that ran
        set_valid itself (an instance attribute), and the actors' networks are built from a published config and
        state dict (selfplay.Nets, PastNet), so they let a hero with six items buy a race-shop item the learner's
        network masks: its log-probability there -1e9, the update's approximate KL ~1e3-1e5 (fgself-179 on)."""
        h, af, it = fx.A_HERO, fx.HERO_ABIL_F, fx.A_HERO + 4 * fx.HERO_ABIL_F
        e = ent[:, :typ.shape[1]] if ent.shape[1] > typ.shape[1] else ent
        hidden = 9 + fx.FLAG_BITS.index(16)  # (UnitFlags.HIDDEN: a worker inside a gold mine)
        bits = torch.cat([e[..., h + 1:h + 4 * af:af] > 0.5, e[..., h:h + 4 * af:af] > 0.1,
                          e[..., h + 3:h + 4 * af:af] > 0.5, e[..., it + 1:it + 6 * fx.HERO_ITEM_F:fx.HERO_ITEM_F] > 0.5,
                          e[..., hidden:hidden + 1] < 0.5,
                          (e[..., it:it + 6 * fx.HERO_ITEM_F:fx.HERO_ITEM_F] < 0.5).any(-1, keepdim=True)], -1)
        if self.valid_mana:  # (--full-vocab: bits 20-25 slot k holds an item, 26 on the unit's mana at least each of
            # fullvocab.mana_steps: mana / max mana times max mana / 1000, the entity's columns 7 and 8)
            mana = e[..., 7].float() * e[..., 8].float() * 1000.0  # (compared with Python's numbers: no tensor made here,
            bits = torch.cat([bits, e[..., it:it + 6 * fx.HERO_ITEM_F:fx.HERO_ITEM_F] > 0.5]  # inside a graph's capture)
                             + [(mana >= s - 0.5).unsqueeze(-1) for s in self.valid_mana], -1)
        need = self.valid_need[typ.long()]
        ok = torch.ones_like(need, dtype=torch.bool)  # (like `need`: under torch.func.vmap a stacked slot's, --infer-past-stack)
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
            p = glob[..., at:at + self.priv - self.vtax]  # (not the taxes: the value's alone)
            g0 = g0 + glob[..., at - 1:at].to(g0.dtype) * self.priv_pol(p.to(g0.dtype))
        tok = torch.cat([g0.unsqueeze(1), x], 1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        if self.config.get("ghost") and ent.shape[-1] > fx.A_MEM:  # (the remembered enemies: their attention bias)
            ghost = torch.cat([torch.zeros_like(mask[:, :1]), ent[..., fx.A_MEM] > 0.5], 1)
            zero = torch.zeros((), dtype=x.dtype, device=x.device)
            pad = torch.where(pad, torch.full_like(zero, float("-inf")),
                              torch.where(ghost, self.ghost_bias.to(x.dtype), zero))
        if self.obsfull:  # (--obs-full: each new entity kind's attention bias, kind_bias)
            kb = (ent[..., fx.A_OBS:fx.A_OBS + 3].to(x.dtype) * self.kind_bias.to(x.dtype)).sum(-1)
            kb = torch.cat([torch.zeros_like(kb[:, :1]), kb], 1)
            if pad.dtype == torch.bool:
                pad = torch.where(pad, torch.full_like(kb, float("-inf")), kb)
            else:
                pad = pad + kb
        if self.transformer.layers[0].self_attn.batch_first:
            out = self.transformer(tok, src_key_padding_mask=pad)
        else:  # (paststack.PastStack's network: the tokens [L, N, d] in memory. Under torch.func.vmap a linear layer folds
            # a contiguous input's leading dimensions into rows; the attention's transposed [L, N, d] view it does not, and
            # the batched weights were copied once per token: S x L copies of each in-projection's)
            out = self.transformer(tok.transpose(0, 1).contiguous(), src_key_padding_mask=pad).transpose(0, 1)
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

    def order_logits(self, g, u, typ, n_own, by_type: bool = True, avail=None, ent=None, emask=None):
        """[N, O, n_orders] for the first O entities (own units first; others can only get none).
        `by_type`: only the orders each unit type got in the demonstrations (playing; training
        learns it: a mask there made unseen pairs, e.g. of upgraded buildings, infinitely wrong).
        `ent` (a network with "valid"): its units' features, for their hero orders (unit_valid).
        `emask` [N, E] (a network with "omask", with `ent` and `typ` over all the entities): the entities in view; a
        unit-target spell only while one of them takes it (target_any)."""
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
            allowed = _and_cols(allowed, self.valid_cols, self.unit_valid(ent[:, :O], typ[:, :O]))
        if self.omask and ent is not None and emask is not None:
            allowed = _and_cols(allowed, self.ptr_cols, self.target_any(ent, typ, self._targetable(emask, ent), O))
        allowed[..., 0] = True  # no order: always
        return logits.masked_fill(~allowed, NEG)

    def _targetable(self, mask, ent):
        """The entities a pointer may pick [N, E]: those in view, not obs-full's new kinds (no order of the vocabulary
        takes a corpse, an item lying or an area)."""
        if self.obsfull:
            if ent is None:
                raise ValueError("target_logits: a network with obsfull needs the entities' features (ent)")
            mask = mask & (ent[..., fx.A_OBS:fx.A_OBS + 3].sum(-1) < 0.5)
        return mask

    def entity_bits(self, ent, typ):
        """--order-masks 1: each entity's properties [N, E] (int32, ordermasks' bits) from its features and type: its
        side, a building, flying or ground, a hero, a machine or an ancient (by type), alive (all of them), and out of
        the player's sight (a privileged view's hidden enemies: the order mask of opsd's teacher reads only what the
        player sees, as the policy's)."""
        flying, struct = ent[..., om.CH_FLYING] > 0.5, ent[..., om.CH_STRUCT] > 0.5
        i32 = torch.int32
        bits = ((ent[..., om.CH_OWN] > 0.5).to(i32) * om.OWN + (ent[..., om.CH_ENEMY] > 0.5).to(i32) * om.ENEMY
                + (ent[..., om.CH_NEUTRAL] > 0.5).to(i32) * om.NEUTRAL + struct.to(i32) * om.STRUCT
                + (flying & ~struct).to(i32) * om.AIR + (~flying & ~struct).to(i32) * om.GROUND
                + (ent[..., om.CH_HERO] > 0.5).to(i32) * om.HERO)
        if self.config.get("privileged") and 0 <= fx.A_HID < ent.shape[-1]:
            bits = bits + (ent[..., fx.A_HID] > 0.5).to(i32) * om.UNSEEN
        return bits | self.type_kind[typ.long()] | om.ALIVE

    def pointer_ok(self, ent, typ, mask, order):
        """--order-masks 1: [N, O, E] the entities each unit's order may aim at: those in `mask` its rule takes (and
        not the unit itself where the rule refuses it). A unit whose order no entity takes keeps `mask` (as without the
        rule: its log-probability as before; the order mask, target_any, keeps such an order from being drawn)."""
        eb = self.entity_bits(ent, typ)  # [N, E]
        rule = self.ptr_rule[order.long()]  # [N, O]
        bad = (rule.unsqueeze(-1) & eb.unsqueeze(1)) != 0
        O, E = order.shape[1], eb.shape[1]
        itself = torch.arange(O, device=eb.device)[:, None] == torch.arange(E, device=eb.device)[None, :]
        bad = bad | (itself & ((rule & om.SELF) != 0).unsqueeze(-1))
        ok = mask.unsqueeze(1) & ~bad
        return torch.where(ok.any(-1, keepdim=True), ok, mask.unsqueeze(1))

    def target_any(self, ent, typ, mask, O: int):
        """--order-masks 1: [N, O, len(ptr_cols)] whether some entity in `mask` takes each ruled order of each of the
        first O units (the unit itself out where the rule refuses it)."""
        eb = self.entity_bits(ent, typ)  # [N, E]
        rule = self.ptr_rule[self.ptr_cols]  # [R]
        ok = ((rule[None, :, None] & eb[:, None, :]) == 0) & mask[:, None, :]  # [N, R, E]
        n = ok.sum(-1)  # [N, R]
        E = eb.shape[1]
        mine = ok[..., :min(O, E)].transpose(1, 2)  # [N, O', R]: unit i is entity i
        if O > E:
            mine = torch.cat([mine, mine.new_zeros(mine.shape[0], O - E, mine.shape[2])], 1)
        n = n.unsqueeze(1) - (mine & ((rule & om.SELF) != 0)).to(n.dtype)  # [N, O, R]
        return n > 0

    def target_logits(self, g, u, mask, order, ent=None, typ=None):
        """Given each own unit's order [N, O]: pointer logits [N, O, E], the point's x logits
        [N, O, BINS], and z (for y_logits). `ent` (a network with --obs-full: its entities' features): the new entity
        kinds are no targets (_targetable). `typ` [N, E] (a network with "omask", with `ent`): each order's pointer only
        on the entities it takes (pointer_ok)."""
        mask = self._targetable(mask, ent)
        O = order.shape[1]
        own = u[:, :O]
        z = self.cond(torch.cat([own + self.order_emb(order.long()), g.unsqueeze(1).expand_as(own)], -1))
        t = self.target_temp
        ptr = torch.einsum("nod,ned->noe", self.query(z), self.key(u)) / (math.sqrt(self.d) * t)
        if self.omask:
            if ent is None or typ is None:
                raise ValueError("target_logits: a network with omask needs the entities' features and types")
            ptr = ptr.masked_fill(~self.pointer_ok(ent, typ, mask, order), NEG)
        else:
            ptr = ptr.masked_fill(~mask.unsqueeze(1), NEG)
        return ptr, self.point_x(z) / t, z

    def y_logits(self, z, x):
        """The point's y logits [N, O, BINS] given its x bins [N, O]."""
        return self.point_y(z + self.x_emb(x.clamp(min=0).long())) / self.target_temp

    def fine_logits(self, z, x, y):
        """(--fine-points) The point's sub-cell logits [N, O, fine * fine] (x's sub-cell * fine + y's) given its x and
        y bins [N, O]."""
        h = z + self.fine_x(x.clamp(min=0).long()) + self.fine_y(y.clamp(min=0).long())
        return self.point_f(h) / self.target_temp

    def value_out(self, g, glob=None):
        """(the value per part [N, parts], with --value-bins the logits [N, parts, bins] it is the expectation of, else
        None). The bins' head in fp32 (autocast off: a bf16 logit near 30 rounds to steps of 0.125, and the expectation
        weighs bins whose logits differ by less); without bins the head as before."""
        if self.priv and glob is not None:  # (the privileged block: what the player does not see)
            p = glob[..., -self.priv:].to(g.dtype)
            g = g + self.priv_proj(p[..., :self.priv - self.vtax])
            if self.vtax:  # (--value-tax: and the game's taxes)
                g = g + self.tax_proj(p[..., self.priv - self.vtax:])
        if not self.vbins:
            return self.value_head(g), None
        with torch.autocast(g.device.type, enabled=False):
            lg = self.value_head(g.float()).float().unflatten(-1, (-1, self.vbins))
            return (torch.softmax(lg, -1) * self.value_centers).sum(-1), lg

    def value_parts(self, g, glob=None):
        """[N, parts] (one: the value; --value-bins: the expectation over the bins)."""
        return self.value_out(g, glob)[0]

    def value_logits(self, g, glob=None):
        """--value-bins: the value's logits [N, parts, bins] (None without bins)."""
        return self.value_out(g, glob)[1]

    def value(self, g, glob=None):
        return self.value_parts(g, glob).sum(-1)

    def uses(self, order):
        """Which targets each order uses: (pointer [N, O], point [N, O]) booleans."""
        kind = self.order_kind[order.long()]
        return kind == fx.UNIT, (kind == fx.POINT) | (kind == fx.TREE)


def _and_cols(allowed, cols, ok):
    """allowed [..., C] (bool) with its columns `cols` [K] (distinct) and-ed with ok [..., K]: `allowed[..., cols] &= ok`
    as a scatter, the same booleans. (The indexed in-place form fails under torch.func.vmap with a stacked slot's `cols`,
    --infer-past-stack: advanced indexing after an ellipsis puts the indexed dimension first there.)"""
    return allowed & torch.ones_like(allowed).scatter(-1, cols.expand(ok.shape), ok)


def arch_key(config: dict) -> str:
    """Networks that can share an inference slot (their weights copied in): the same shapes; the target
    temperature is a buffer and comes with the weights, and dropout does nothing at inference."""
    return json.dumps({k: v for k, v in config.items() if k not in ("target_temperature", "order_temperature", "dropout")},
                      sort_keys=True,
                      default=str)


def pointer_label_ok(net: FullGameNet, ptr, y_ptr, weight):
    """A pointer loss's `weight` [N, O] without the labels `y_ptr` (-1: none) its order's pointer may not pick
    (--order-masks 1: a teacher's or a demonstration's target the order's rule refuses, a -1e9 logit, gets no loss);
    `weight` itself for a network without the rule (as before)."""
    if not net.omask:
        return weight
    got = ptr.gather(-1, y_ptr.clamp(min=0).long().unsqueeze(-1)).squeeze(-1)
    return weight * (got.float() > NEG / 2).to(weight.dtype)


def _logp(logits, x):
    return torch.log_softmax(logits.float(), -1).gather(-1, x.long().unsqueeze(-1)).squeeze(-1)


def _gumbel_argmax(logits, generator=None, shared=None, noise=None):
    """A sample from softmax(logits): the argmax of logits plus Gumbel noise (fewer kernels and no
    Python bookkeeping, unlike torch.distributions; the network calls are latency-bound). `generator`:
    a random generator of the caller's (an inference thread next to a learner's: graph captures).
    `shared` [N] (bool): those rows' units [N, O, C] draw with one noise vector for all of them (common
    random numbers): each unit's own distribution is unchanged, but units that want about the same
    choose the same (one target point for a group: with a draw each, a group's points scattered).
    `noise` (u [N, O, C], one [N, 1, C] or None): the uniform draws, made by the caller (noise_shapes; --infer-past-stack
    draws them outside torch.func.vmap, where torch.compile cannot trace a draw from a generator), used as these would be."""
    if noise is None:
        u = torch.rand(logits.shape, device=logits.device, generator=generator)
        if shared is not None:
            one = torch.rand((logits.shape[0], 1) + tuple(logits.shape[2:]), device=logits.device, generator=generator)
            u = torch.where(shared.view((-1,) + (1,) * (logits.dim() - 1)), one.expand_as(u), u)
        u = u.clamp_(1e-10, 1.0 - 1e-7)
    else:
        u, one = noise
        if shared is not None:
            u = torch.where(shared.view((-1,) + (1,) * (logits.dim() - 1)), one.expand_as(u), u)
        u = u.clamp(1e-10, 1.0 - 1e-7)
    return (logits.float() - torch.log(-torch.log(u))).argmax(-1)


NOISE_ROLES = ("order", "tgt", "x", "y", "fine")  # (act's draws, in its order: noise_shapes)


def noise_shapes(net: "FullGameNet", N: int, E: int) -> dict[str, tuple]:
    """act's draws for N rows of E entities -> {role: (u's shape, one's shape)}: the order's (one: unused, there is no
    shared draw for it), the target unit's, the point's x and y, and with a fine head of features.FINE's size its
    sub-cell (fine_head). For act(noise=...): each role's (u, one) drawn uniform on [0, 1)."""
    O = min(fx.MAX_OWN, E)
    out = {"order": ((N, O, net.config["n_orders"]), (N, 1, net.config["n_orders"])), "tgt": ((N, O, E), (N, 1, E)),
           "x": ((N, O, net.bins), (N, 1, net.bins)), "y": ((N, O, net.bins), (N, 1, net.bins))}
    if fine_head(net):
        out["fine"] = ((N, O, fx.FINE * fx.FINE), (N, 1, fx.FINE * fx.FINE))
    return out


def _entropy(logits):
    lp = torch.log_softmax(logits.float(), -1)
    return -(lp.exp() * lp).sum(-1)


def sample_points(net: FullGameNet, z, xl, sample):
    """A point target for each unit: its x bin from `xl`, then y given x, and with features.FINE (--fine-points) the
    sub-cell given both -> x and y in the run's point indices [N, O] (bin * FINE + sub-cell) and their log-probability
    [N, O]. `sample`: logits -> a draw. A network without a fine head of FINE's size (an older one: a past snapshot,
    the anchor) takes the middle sub-cell, its bins' centres as before, and draws nothing more."""
    bx = sample(xl)
    yl = net.y_logits(z, bx)
    by = sample(yl)
    lp = _logp(xl, bx) + _logp(yl, by)
    S = fx.FINE
    if S > 1:
        if net.fine == S:
            fl = net.fine_logits(z, bx, by)
            f = sample(fl)
            lp = lp + _logp(fl, f)
        else:
            f = torch.full_like(bx, S * S // 2)
        bx, by = bx * S + torch.div(f, S, rounding_mode="floor"), by * S + f % S
    return bx, by, lp


def point_parts(bx, by):
    """Point indices [N, O] (actions, or labels with -1 for none) -> (x bins, y bins, the sub-cell's class, -1 for
    none; None without features.FINE)."""
    S = fx.FINE
    if S <= 1:
        return bx, by, None
    f = (bx % S) * S + by % S
    return (torch.div(bx, S, rounding_mode="floor"), torch.div(by, S, rounding_mode="floor"),
            torch.where(bx >= 0, f, torch.full_like(f, -1)))


def fine_head(net: FullGameNet) -> bool:
    """Whether `net` picks the sub-cells of this run's point indices (a fine head of features.FINE's size)."""
    return fx.FINE > 1 and net.fine == fx.FINE


def _in_fp32(device_type: str | None, fn, *args, **kw):
    """fn(*args, **kw); with a device type (act's heads_fp32) in fp32 inside a bf16 autocast region: that device's
    autocast off and the floating inputs cast up (the trunk's outputs are fp32 already: its last layer norm, the memory's
    sum). None: the plain call, nothing added."""
    if device_type is None:
        return fn(*args, **kw)
    up = lambda a: a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a  # noqa: E731
    with torch.autocast(device_type, enabled=False):
        return fn(*[up(a) for a in args], **{k: up(v) for k, v in kw.items()})


def act(net: FullGameNet, ent, typ, cur, mask, glob, n_own, avail=None, h=None, generator=None, shared=None,
        guide=None, heads_fp32: bool = False, noise: dict | None = None) -> dict:
    """Sample every own unit's order and targets (the first O = min(MAX_OWN, E) entities; the
    orders each unit type got in the demonstrations). -> order, tgt, bx, by (point indices: sample_points), logp [N, O] (the
    action's log-probability per unit: order, plus the targets that order uses; 0 for padding)
    and value [N]; with memory the core's new state "h" [N, d] (`h`: the last one, None at a start).
    `shared` [N] (bool): those rows sample their units' targets (unit, x, y) with one noise vector for all of
    them (_gumbel_argmax); the order still with one each.
    `guide` [N] (1: none), classifier-free guidance by the target strategy (as in image generation): the order's
    logits also without the strategy (its block of the global vector zeroed: the policy trains both ways, a
    quarter of its games with no target) and mixed as free + guide * (with - free); > 1 follows the target
    harder. The targets (unit, x, y), the value and the memory come from the pass with the strategy.
    `heads_fp32` (--infer-precision bf16-mixed): in a bf16 autocast region only the trunk (the encoders, the transformer,
    the memory core) in bf16, every head (the order; the target's conditioning, query, key and pointer; the point's x
    and y; the value) in fp32 (_in_fp32): bf16 rounds its outputs, a logit near 30 (the point and pointer heads' are) to
    steps of 0.125-0.25 and a value near 0.5 to 0.004. Without autocast it changes nothing.
    `noise` ({role: (u, one)} of noise_shapes' shapes; None: drawn here from `generator`): the sampling's uniform draws
    made by the caller (--infer-past-stack: outside torch.func.vmap); the same samples as these draws would give here."""
    h_in = h
    g, u = net.encode(ent, typ, cur, mask, glob)
    g, h = net.context(g, h)
    O = min(fx.MAX_OWN, u.shape[1])
    heads32 = g.device.type if heads_fp32 else None
    emask = mask if net.omask else None  # (--order-masks 1: a unit-target spell only while an entity in view takes it)
    logits = _in_fp32(heads32, net.order_logits, g, u[:, :O], typ, n_own, avail=avail, ent=ent, emask=emask)
    if guide is not None and net.config.get("z"):
        z0 = net.config["G"] - net.priv - net.config.get("gx", 0) - net.config["z"]
        blank = glob.clone()
        blank[:, z0:z0 + net.config["z"]] = 0.0
        g0, u0 = net.encode(ent, typ, cur, mask, blank)
        g0, _ = net.context(g0, h_in)
        free = _in_fp32(heads32, net.order_logits, g0, u0[:, :O], typ, n_own, avail=avail, ent=ent, emask=emask)
        w = guide.to(logits.dtype)[:, None, None]
        logits = torch.where(w != 1, free + w * (logits - free), logits)
    roles = iter(NOISE_ROLES[1:])  # (the targets' draws in order: the unit, x, y, the sub-cell)

    def sample(lg):  # (without `noise` the call as before: a sampler swapped in by a study or a test takes three)
        if noise is None:
            return _gumbel_argmax(lg, generator, shared)
        return _gumbel_argmax(lg, generator, shared, noise[next(roles)])
    # (the order itself with a draw of its own per unit: one draw for all made a side's similar units take a rare
    # order together or not at all: every worker builds a farm, or none does; the A/B of 2026-10-12)
    order = _gumbel_argmax(logits, generator) if noise is None else _gumbel_argmax(logits, generator, None, noise["order"])
    ptr, xl, z = _in_fp32(heads32, net.target_logits, g, u, mask, order, ent, typ)
    tgt = sample(ptr)
    bx, by, lp_pt = _in_fp32(heads32, sample_points, net, z, xl, sample)  # (with --fine-points: point indices, the
    uses_ptr, uses_pt = net.uses(order)                                       # sub-cell too)
    own = torch.arange(O, device=ent.device)[None] < n_own[:, None]
    logp = _logp(logits, order) + uses_ptr * _logp(ptr, tgt) + uses_pt * lp_pt
    ent = (_entropy(logits) * own).sum(-1) / own.sum(-1).clamp(min=1)  # the order's, mean over own units
    out = {"order": order, "tgt": tgt, "bx": bx, "by": by, "logp": logp * own,
           "value": _in_fp32(heads32, net.value, g, glob), "entropy": ent}
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
    logits = net.order_logits(g, u[:, :O], typ, n_own, avail=avail, ent=ent, emask=mask if net.omask else None)
    ptr, xl, z = net.target_logits(g, u, mask, order, ent, typ)
    cx, cy, f = point_parts(bx, by)  # (--fine-points: the bins, and the sub-cell)
    yl = net.y_logits(z, cx)
    uses_ptr, uses_pt = net.uses(order)
    lp_pt = _logp(xl, cx) + _logp(yl, cy)
    fl = net.fine_logits(z, cx, cy) if f is not None and fine_head(net) else None
    if fl is not None:
        lp_pt = lp_pt + _logp(fl, f.clamp(min=0))
    logp = _logp(logits, order) + uses_ptr * _logp(ptr, tgt) + uses_pt * lp_pt
    vp, vl = net.value_out(g, glob)  # (--value-bins: "vlogits" [N, parts, bins] for the loss, the value their expectation)
    return {"logp": logp, "entropy": _entropy(logits), "logits": logits, "value": vp.sum(-1), "g": g, "u": u,
            "ptr": ptr, "xl": xl, "yl": yl, "uses_ptr": uses_ptr, "uses_pt": uses_pt,
            **({"fl": fl} if fl is not None else {}), **({"vparts": vp} if net.config.get("vparts") else {}),
            **({"vlogits": vl} if vl is not None else {})}


def anchor_kl(ev: dict, ev_ref: dict, mode: str = "reverse"):
    """The KL term towards the anchor per unit [N, O]. "reverse": KL(policy || anchor) over the orders (as
    before: its pull on an order is in proportion to the policy's own probability of it, so an order RL
    drives towards 0 is no longer pulled back); "forward": KL(anchor || policy) over the orders (AlphaStar's
    direction: a rare order the anchor gives keeps its weight); "forward-all": and over the target unit
    and the point's x and y (y given the step's x), for the units whose order uses them; with --fine-points the
    sub-cell too (given the step's bins) when the anchor has a fine head of its own (an anchor without one, a
    checkpoint from before them, plays the bins' centres: no pull towards them, nor towards a fresh head's prior).
    "forward-acting": as forward-all, but the order's term only over the orders given one (both distributions
    renormalized without "none", class 0) and weighted by the policy's chance of giving one (a constant: no gradient
    through it), so the anchor says what to order, never whether to: people's clones give an order in ~5% of a
    unit's steps, and a pull towards their "none" taught passivity (fgself-161). The targets' terms as forward-all
    (given the step's order: none for a unit that gave none)."""
    def kl(a, b):  # KL(softmax a || softmax b) over the last axis
        la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
        return (la.exp() * (la - lb)).sum(-1)
    if mode == "reverse":
        return kl(ev["logits"], ev_ref["logits"])
    if mode == "forward-acting":
        acting = 1.0 - torch.softmax(ev["logits"].float(), -1)[..., 0].detach()
        out = acting * kl(ev_ref["logits"][..., 1:], ev["logits"][..., 1:])
    else:
        out = kl(ev_ref["logits"], ev["logits"])
    if mode in ("forward-all", "forward-acting"):
        pt = kl(ev_ref["xl"], ev["xl"]) + kl(ev_ref["yl"], ev["yl"])
        if "fl" in ev and "fl" in ev_ref:
            pt = pt + kl(ev_ref["fl"], ev["fl"])
        out = out + ev["uses_ptr"] * kl(ev_ref["ptr"], ev["ptr"]) + ev["uses_pt"] * pt
    return out


# tables from the game's data and the vocabulary, never trained
DERIVED = ("allowed", "valid_cols", "valid_need", "ptr_rule", "type_kind", "ptr_cols")


def trained(state: dict, net: FullGameNet) -> dict:
    """A checkpoint's weights for a network built since (a resumed run): the tables it derives from the game's
    data (the orders each type may give, neutral.who_buys; the hero orders' needs, set_valid) the network's
    own, as load gives them now, not the checkpoint's."""
    own = net.state_dict()
    out = {**state, **{k: own[k] for k in DERIVED if k in own}}
    # (--value-tax turned on or off at a restart: the taxes' projection as the network has it, zero at first, or none)
    out = {k: v for k, v in out.items() if not k.startswith("tax_proj.") or k in own}
    return {**{k: v for k, v in own.items() if k.startswith("tax_proj.") and k not in out}, **out}


def adapt_config(config: dict) -> dict:
    """A checkpoint's config as this run's inputs read it (features.set_inputs): the clock's numbers in the
    global vector (inputs 2: features.GX, between the strategy block and the privileged one) added."""
    config = dict(config)
    if fx.F > fx.A_MEM and fx.INPUTS >= 3 and not config.get("ghost"):  # (inputs 3: the remembered enemies hidden from
        config["ghost"] = True                                            # a network before them)
    if fx.PRIVILEGED and not config.get("privileged"):  # (the privileged path, zero at first)
        config["privileged"] = True
    if fx.OBS_FULL and not config.get("obsfull"):  # (--obs-full: its block's weights zero, its new kinds hidden: -30)
        config["obsfull"] = fx.OBS_FULL
    elif config.get("obsfull") and not fx.OBS_FULL and fx.OBS_FULL_SET:  # (a process set without it: a later run's
        # league, evaluation or --league-add holding a network trained with it. The network without its block and kinds,
        # as one with more inputs reads those it has (load); built as it was, its constructor had turned the whole
        # process's features to its width, and the run's own network and rows no longer fit them)
        config.pop("obsfull")
    gx = config.get("gx", 0)
    if fx.GX > gx:
        config["G"] += fx.GX - gx
        config["gx"] = fx.GX
    z = config.get("z", 0)
    if z and fx.ZDIM > z:  # (--z-modes: the strategy block's modes after its build target's)
        config["G"] += fx.ZDIM - z
        config["z"] = fx.ZDIM
    vt = config.get("vtax", 0)
    if config.get("priv") and fx.VTAX != vt:  # (--value-tax: the game's taxes at the privileged block's end, the
        config["G"] += fx.VTAX - vt           # value's alone: tax_proj, zero at first; a run without them drops it)
        config["priv"] += fx.VTAX - vt
        config.pop("vtax", None)
        if fx.VTAX:
            config["vtax"] = fx.VTAX
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
    fx.set_spells(int(vocab.get("spells") or 0))
    fx.set_shop_items(int(vocab.get("shop_items") or 0))
    fx.set_full_vocab(int(vocab.get("full_vocab") or 0))
    fx.set_z(c.get("z", 0))
    fx.set_fine(c.get("fine", 0))  # (--fine-points: the sub-cells its fine head picks)
    fx.set_value_tax(c.get("vtax", 0))  # (--value-tax: the taxes it reads, zeros here)
    fx.set_obs_full(c.get("obsfull", 0))  # (--obs-full: its block and entity kinds)
    if c.get("gx", 0) >= 3:  # (the unit channels it reads: all of them, a run's --features zeros only its weights)
        fx.ACTIVE_F = fx.F


def glob_layout(ck: dict) -> tuple[dict, int, int, int]:
    """A checkpoint's config (the privileged block it lacks added) and what its policy read: (config, gx, z, G)."""
    config = dict(ck["config"])
    if "vocab" in ck:  # global features added since (the privileged block at the end): the value's, zero at first
        G_now = fx.Encoder(ck["vocab"]).G - fx.GX - fx.VTAX + config.get("gx", 0)
        if config["G"] < G_now and not config.get("priv"):
            config["priv"], config["G"] = G_now - config["G"], G_now
    return config, config.get("gx", 0), config.get("z", 0), config["G"]


def grow_glob(w: torch.Tensor, config: dict, gx_old: int, z_old: int, G_old: int) -> torch.Tensor:
    """glob.0.weight (or Adam's state of it) read as [known | strategy block | clock] by a network of G_old inputs,
    for `config`'s: the grown parts zeros, where they are (the strategy block's end, and the end). (Its width tells
    where the old privileged block began: --value-tax may have grown that block too.)"""
    at = w.shape[1] - gx_old  # (the strategy block's end)
    dz, dg = config.get("z", 0) - z_old, config.get("gx", 0) - gx_old
    return torch.cat([w[:, :at], w.new_zeros(w.shape[0], dz), w[:, at:], w.new_zeros(w.shape[0], dg)], 1)


def load(path, device="cpu", memory: bool | None = None, vocab: dict | None = None) -> tuple[FullGameNet, dict]:
    """A checkpoint's network. `memory`: with (a new core, which does nothing at first) or without one. Inputs
    added since (features.set_inputs: unit features, the clock) start with zero weights, and with --valid-mask
    (features.VALID_MASK) its hero orders' mask is set (set_valid). `vocab` (--neutral: the run's): its orders in
    that vocabulary's order (neutral.reorder; one extended in another process may list a toggle's two the other
    way round); with --spells, or a `vocab` that has them, the heroes' spells it lacked (spells.py). With --order-masks
    1 (features.ORDER_MASKS), or a network trained with them (its config's "omask"), its order masks' tables are set
    last, from the vocabulary as it is then (set_order_masks; bit 2's forms too: ordermasks.form_refused)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    config, gx_old, z_old, G_old = glob_layout(ck)
    if memory is not None:
        config["memory"] = memory
    config = adapt_config(config)
    valid = bool(config.get("valid"))
    # (--order-masks 1's tables: made anew at the end, for the vocabulary the network ends with; built without them, a
    # network grown below to more orders and types never copies tables of the old sizes. A checkpoint with no vocabulary
    # keeps its own)
    omask = bool(config.pop("omask", 0)) if "vocab" in ck else False
    net = FullGameNet(**config).to(device)
    state = ck["model"] if net.memory else {k: v for k, v in ck["model"].items() if not k.startswith("mem_")}
    if omask:
        state = {k: v for k, v in state.items() if k not in ("ptr_rule", "type_kind", "ptr_cols")}
    w = state.get("ent.weight")
    if w is not None and w.shape[1] < fx.F:  # features added since (at the end): their weights start at 0
        state = {**state, "ent.weight": torch.cat([w, w.new_zeros(w.shape[0], fx.F - w.shape[1])], 1)}
    elif w is not None and w.shape[1] > fx.F:  # (a run with fewer inputs: those it has)
        state = {**state, "ent.weight": w[:, :fx.F]}
    w = state.get("glob.0.weight")
    if w is not None and w.shape[1] < config["G"] - config.get("priv", 0):  # what the policy reads, [known | strategy
        # block | clock]: grown parts zeros
        state = {**state, "glob.0.weight": grow_glob(w, config, gx_old, z_old, G_old)}
    if not net.vtax:  # (a network trained with --value-tax, in a run without: its value without the taxes)
        state = {k: v for k, v in state.items() if not k.startswith("tax_proj.")}
    if not net.obsfull:  # (one trained with --obs-full, in a process without it: adapt_config)
        state = {k: v for k, v in state.items() if k != "kind_bias"}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected or any(not (k.startswith(("value_head.", "mem_", "priv_proj.", "priv_pol.", "tax_proj."))
                              or k in ("order_kind", "target_temp", "ghost_bias", "kind_bias"))
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
        from . import spells  # (--spells, or a run whose vocabulary has them: the heroes' spells it lacked, spells.py)
        if (fx.SPELLS or vocab is not None and vocab.get("spells")) and not spells.extended(ck["vocab"]):
            new = spells.extend(ck["vocab"])
            net = spells.grow(net, ck["vocab"], new)
            ck = {**ck, "vocab": new}
        if ck["vocab"].get("spells"):
            spells.allow(net, ck["vocab"])
        from . import shopitems  # (--shop-items, or a run whose vocabulary has them: after the spells, shopitems.py)
        if (fx.SHOP_ITEMS or vocab is not None and vocab.get("shop_items")) and not shopitems.extended(ck["vocab"]):
            new = shopitems.extend(ck["vocab"])
            net = shopitems.grow(net, ck["vocab"], new)
            ck = {**ck, "vocab": new}
        from . import fullvocab  # (--full-vocab, or a run whose vocabulary has it: every order left, after the shops' items)
        if (fx.FULL_VOCAB or vocab is not None and vocab.get("full_vocab")) and not fullvocab.extended(ck["vocab"]):
            new = fullvocab.extend(ck["vocab"])
            net = fullvocab.grow(net, ck["vocab"], new)
            ck = {**ck, "vocab": new}
        from . import obsfull  # (--obs-full, or a run whose vocabulary has them: the maps' creep types it lacked, each
        if (fx.OBS_FULL or vocab is not None and vocab.get("obs_full")) and not obsfull.extended(ck["vocab"]):
            new = obsfull.extend(ck["vocab"])  # embedding unknown's: every token as before)
            net = obsfull.grow(net, ck["vocab"], new)
            ck = {**ck, "vocab": new}
        if vocab is not None and ck["vocab"]["orders"] != vocab["orders"]:
            if neutral.reorder(net, ck["vocab"], vocab) is None:
                raise RuntimeError(f"{path}: its vocabulary has other orders than the run's")
            ck = {**ck, "vocab": vocab}
        neutral.who_buys(net, ck["vocab"])  # (and its tables as they are now: networks grown before them too)
        if ck["vocab"].get("shop_items"):
            shopitems.allow(net, ck["vocab"])
        if ck["vocab"].get("full_vocab"):
            fullvocab.allow(net, ck["vocab"])
        if net.config.get("valid"):
            net.set_valid(ck["vocab"])
    if "vocab" in ck and "allowed" in dict(net.named_buffers()):
        prune_allowed(net, ck["vocab"])
        hero_forms(net, ck["vocab"])
        if fx.ORDER_MASKS & 2:  # (--order-masks 2: the robo-goblin morphs no more, the tinker turns back from nothing)
            for i, c in om.form_refused(ck["vocab"]):
                if i < net.allowed.shape[0] and c < net.allowed.shape[1]:
                    net.allowed[i, c] = False
    if "vocab" in ck and (fx.ORDER_MASKS & 1 or omask):  # (last: a past snapshot's slot copies its buffers by position)
        net.set_order_masks(ck["vocab"])
    return net.eval(), ck


# a hero's morphed form -> (the hero, the order that morphs it): the form keeps the hero's abilities and orders but that one
HERO_FORMS = {"Edmm": ("Edem", 852180)}  # (the demon hunter's metamorphosis, 60 s at level 6+)


@torch.no_grad()
def hero_forms(net: FullGameNet, vocab: dict) -> int:
    """A hero's morphed form may give what its hero may (`net`'s allowed table, in place; HERO_FORMS) -> how many it
    gains. The table came from the demonstrations, where the built-in AI's demon form only attack-moved, burned mana
    and turned immolation off: no attack on a unit, no stop, no immolation on (2026-11-11, people's games: 7 of 782
    immolation states blocked so). The valid mask still asks of the form what it asks of the hero (its slots)."""
    tix = {fx.rawcode(int(t)): i for i, t in enumerate(vocab["types"], start=1)}
    oix = {(int(o), int(k)): c for c, (o, k) in enumerate(vocab["orders"], start=1)}
    T, O = net.allowed.shape
    n = 0
    for form, (hero, morph) in HERO_FORMS.items():
        i, j = tix.get(form), tix.get(hero)
        if i is None or j is None or i >= T or j >= T:
            continue
        add = net.allowed[j] & ~net.allowed[i]
        c = oix.get((morph, fx.IMMEDIATE))
        if c is not None and c < O:
            add[c] = False
        net.allowed[i] |= add
        n += int(add.sum())
    return n


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
    if fx.FUTILE_MASK:  # (--econ-masks bit 64: a burrowed crypt fiend or a stone gargoyle takes only its way back, the
        index = {(int(o), int(k)): c for c, (o, k) in enumerate(vocab["orders"], start=1)}  # unmorphed form not that)
        tix = {fx.rawcode(int(t)): i for i, t in enumerate(vocab["types"], start=1)}
        for form, (base, back) in fx.MORPHED.items():
            i, j, c = tix.get(form), tix.get(base), index.get((back, fx.IMMEDIATE))
            if i is not None and i < net.allowed.shape[0]:
                was = net.allowed[i].clone()
                net.allowed[i, 1:] = False
                if c is not None and c < net.allowed.shape[1]:
                    net.allowed[i, c] = True
                n += int((was & ~net.allowed[i]).sum())
            if j is not None and j < net.allowed.shape[0] and c is not None and c < net.allowed.shape[1] \
                    and bool(net.allowed[j, c]):
                net.allowed[j, c] = False
                n += 1
    return n


VALUE_INIT_WIDTH = 0.5  # (--value-init-width: the binned head's first histograms' width, the returns' units: bin_value)


def hl_gauss_target(y: torch.Tensor, edges: torch.Tensor, sigma: float) -> torch.Tensor:
    """HL-Gauss's target (Farebrother et al., ICML 2024): the return `y` [...] as a histogram over the bins between
    `edges` [K + 1] -> [..., K], summing to 1: a Gaussian of width `sigma` (the returns' units) around y clipped to the
    edges' span, integrated over each bin (its CDF's differences) and renormalized to the span. Its mean is y while y
    is 4 sigma or more inside the span (at 3 sigma 0.004 sigma short); nearer an edge the Gaussian is cut there and its
    mean falls inside (by ~0.9 sigma at the edge itself). Static shapes, no host syncs: the CUDA-graph update computes it."""
    y = torch.clamp(y.float(), min=edges[0], max=edges[-1])
    cdf = torch.special.erf((edges - y.unsqueeze(-1)) * (1.0 / (math.sqrt(2.0) * float(sigma))))
    return (cdf[..., 1:] - cdf[..., :-1]) / (cdf[..., -1:] - cdf[..., :1])


def value_ce(net: FullGameNet, logits: torch.Tensor, ret: torch.Tensor) -> torch.Tensor:
    """--value-bins: HL-Gauss's cross-entropy per step [N] of the value's `logits` [N, parts, bins] (evaluate's
    "vlogits") against the returns `ret`, [N] (one part) or [N, parts]; summed over the parts as the squared errors are."""
    y = ret if ret.dim() == logits.dim() - 1 else ret.unsqueeze(-1)
    t = hl_gauss_target(y, net.value_edges, net.value_sigma)
    return -(t * torch.log_softmax(logits.float(), -1)).sum(-1).sum(-1)


def gaussian_logits(w: torch.Tensor, b: torch.Tensor, centers: torch.Tensor, width: float) -> tuple:
    """The logit layer whose softmax over the bins' `centers` [K] is a Gaussian of `width` around the scalar layer's
    value (w [P, d], b [P]: v = w h + b per part) for every input h: logit_k = (c_k v - c_k^2 / 2) / width^2, which is
    -(c_k - v)^2 / (2 width^2) plus v^2 / (2 width^2), the same for every bin (softmax drops it), so the layer is linear
    in h -> (weight [P * K, d], bias [P * K]), part by part."""
    c, s2 = centers.to(w), float(width) ** 2
    W = (c[None, :, None] * w[:, None, :] / s2).reshape(-1, w.shape[1])
    B = ((c[None, :] * b[:, None] - c[None, :] ** 2 / 2) / s2).reshape(-1)
    return W, B


def bin_value(net: FullGameNet, bins: int, vmin: float = VALUE_MIN, vmax: float = VALUE_MAX, sigma: float = VALUE_SIGMA,
              width: float = VALUE_INIT_WIDTH) -> FullGameNet:
    """--value-bins: `net`'s scalar value head as HL-Gauss's `bins` logits per part -> a new network (`net` itself when
    it has these bins already; other bins: ValueError). The hidden layer as it was; the last layer gaussian_logits' of
    the old one, so the expected value the actors' GAE reads is the old value from the first step, state by state (to
    ~1e-4 while |v| <= vmax - 4 width, 2e-3 at 3 width: the edge cuts the Gaussian; fgself-267/268's values were 99.8%
    within +-3.5, at width 0.5 inside 4). Zero logits instead (uniform: every value the
    range's centre) would have the advantages without the outcome for the first ~100 updates: with lambda 0.95 the
    value learns ~20 steps back per update, the games are ~1,900 steps long. `width` against the HL-Gauss target's sigma
    (0.089 at 101 bins on [-6, 6]): a head as sharp as the target would need logit gaps of ~63 for a second mode one
    return away, which Adam's ~lr-sized steps would take thousands of updates to make; at 0.5 the gap is 2. For this
    layer the cross-entropy's gradient into its input is exactly the squared error's times 1 / width^2:
    sum_k (p_k - t_k) W_k = w (E_p[c] - E_t[c]) / width^2 (--vf-coef 0.5 * width^2 pulls on the trunk as before)."""
    want = {"vbins": int(bins), "vmin": float(vmin), "vmax": float(vmax), "vsigma": float(sigma)}
    if net.vbins:
        have = {k: net.config.get(k) for k in want}
        if have != want:
            raise ValueError(f"a network with value bins {have}, not {want}")
        return net
    if not float(width) > 0:
        raise ValueError(f"value bins' first width {width}: above 0")
    new = FullGameNet(**{**net.config, **want}).to(net.order_kind.device)
    state = net.state_dict()
    w, b = state.pop("value_head.2.weight"), state.pop("value_head.2.bias")
    missing, unexpected = new.load_state_dict(state, strict=False)
    if unexpected or set(missing) != {"value_head.2.weight", "value_head.2.bias"}:
        raise RuntimeError(f"bin_value: missing {missing}, unexpected {unexpected}")
    with torch.no_grad():
        W, B = gaussian_logits(w.double(), b.double(), new.value_centers.double(), width)
        new.value_head[2].weight.copy_(W)
        new.value_head[2].bias.copy_(B)
    return new.train(net.training)


def split_value(net: FullGameNet, parts: int) -> FullGameNet:
    """--reward-parts: `net`'s value head with one output per reward (the first is the old value, the others zero:
    their sum plays as before) -> a new network."""
    if net.config.get("vparts", 0) >= parts:
        return net
    if net.vbins:  # (selfplay refuses the pair: the exploiters' batches have no returns per part to fit histograms to)
        raise ValueError("--reward-parts with --value-bins: not supported")
    config = {**net.config, "vparts": parts}
    new = FullGameNet(**config).to(net.order_kind.device)
    state = net.state_dict()
    w, b = state["value_head.2.weight"], state["value_head.2.bias"]
    state["value_head.2.weight"] = torch.cat([w, w.new_zeros(parts - w.shape[0], w.shape[1])])
    state["value_head.2.bias"] = torch.cat([b, b.new_zeros(parts - b.shape[0])])
    new.load_state_dict(state)
    return new.train(net.training)


def fine_prior(s: int, centre: float, temperature: float = 1.0) -> torch.Tensor:
    """The fine head's starting bias [s * s]: the middle sub-cell (its bins' centre) sampled with probability `centre`
    at the target temperature (the logits are divided by it), the others alike (centre 1 / s**2 or less: uniform)."""
    n = s * s
    bias = torch.zeros(n)
    centre = min(float(centre), 1.0 - 1e-6)
    if centre > 1.0 / n:
        bias[n // 2] = math.log(centre * (n - 1) / (1.0 - centre)) * temperature
    return bias


def grow_fine(net: FullGameNet, s: int, centre: float = 0.9) -> FullGameNet:
    """--fine-points: `net` with a fine head of s x s sub-cells per bin -> a new network (net itself if it has one).
    Every other output is unchanged (its bins' distributions too); the sub-cell's logits start as a prior alone (the
    last layer's weights zero): fine_prior's, the middle sub-cell, the point as before, with probability `centre`.
    The head's x-bin embedding starts as the y head's (x_emb: what each x bin means), its y-bin one at random."""
    if net.fine == s:
        return net
    if net.fine:
        raise ValueError(f"a network with {net.fine} x {net.fine} sub-cells, not {s} x {s}")
    new = FullGameNet(**{**net.config, "fine": s}).to(net.order_kind.device)
    missing, unexpected = new.load_state_dict(net.state_dict(), strict=False)
    if unexpected or any(not k.startswith(("fine_", "point_f.")) for k in missing):
        raise RuntimeError(f"grow_fine: missing {missing}, unexpected {unexpected}")
    with torch.no_grad():
        new.fine_x.weight.copy_(net.x_emb.weight)
        new.point_f[2].bias.copy_(fine_prior(s, centre, float(net.target_temp)))
    return new.train(net.training)
