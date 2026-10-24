"""The per-step trace a whole-game player writes for its video's panel (fullgame.overlay); no
imaging here, so the players can import it without Pillow."""

from __future__ import annotations

from ..protocol import Observation

DEAD_FLAG = 1024  # the unit flags' dead bit (an int test, not an enum operation: per unit per step)
HERO_FLAG = 1  # (protocol.UnitFlags.HERO)


def heroes(obs: Observation) -> dict[str, list]:
    """Each player's heroes: [type, level, alive, [each ability slot's level (data.abilities order)]]."""
    from .features import rawcode
    out: dict[str, list] = {"0": [], "1": []}
    ua = getattr(obs, "unit_array", None)
    if ua is not None:  # (native observations: columns of native.py; the abilities' levels at 23, 25, 27, 29)
        for r in ua[(ua[:, 11] & HERO_FLAG) > 0].tolist():
            if r[2] in (0, 1):
                out[str(r[2])].append([rawcode(r[1]), r[14], int(not r[11] & DEAD_FLAG), r[23:31:2]])
        return out
    for u in obs.units:
        if u.is_hero and u.owner in (0, 1):
            out[str(u.owner)].append([u.type, u.hero_level, int(u.alive), [lv for lv, _ in u.abilities]])
    return out


def trace_step(t: int, obs: Observation, material: dict, orders: dict, **policy) -> dict:
    """One step's row: `material` {player: value}, `orders` {player: Counter of order labels},
    `policy`: value / reward / entropy, each {player: float} (the players a policy controls)."""
    res = {str(p): [s.gold, s.lumber, s.food_used, s.food_cap] for p, s in obs.players.items() if p in (0, 1)}
    row = {"t": t, "game_time": round(obs.game_time, 2), "material": {str(p): round(v) for p, v in material.items()},
           "res": res, "orders": {str(p): dict(c) for p, c in orders.items()}, "heroes": heroes(obs)}
    for k, v in policy.items():
        row[k] = {str(p): round(float(x), 4) for p, x in v.items()}
    return row


WORKER_IDS = {int.from_bytes(c.encode(), "big") for c in ("hpea", "opeo", "uaco", "ewsp")}  # (type ids)
_VALUES: dict[int, list] = {}  # id(values) -> [values, its size, the type ids seen (sorted), their values]


def _type_values(values: dict, types):
    """Each unit's value by its type id (values[str(id)], 0 for none), as float64: a sorted table of the types seen
    so far, made again when one is new (np.unique with its inverse, each step, had been ~85 us of an actor's step)."""
    import numpy as np
    tab = _VALUES.get(id(values))
    if tab is None or tab[0] is not values or tab[1] != len(values):
        tab = _VALUES[id(values)] = [values, len(values), np.zeros(0, np.int64), np.zeros(0, np.float64)]
    j = np.searchsorted(tab[2], types)
    if not len(tab[2]) or not (tab[2][np.minimum(j, len(tab[2]) - 1)] == types).all():
        tab[2] = np.union1d(tab[2], types).astype(np.int64)
        tab[3] = np.array([values.get(str(t), 0) for t in tab[2].tolist()], float)
        j = np.searchsorted(tab[2], types)
    return tab[3][j]


def material(obs, values: dict, hero_level: float = 0.0, worker_cap: int = 0) -> dict[int, float]:
    """What each player's living units and buildings cost, times their hit points left: {0: .., 1: ..}.
    `hero_level`: a hero also counts this much for each level above the first (its cost alone does not
    grow as it levels; a dead hero counts nothing until it is revived). `worker_cap`: a player's workers
    count only as much as that many of them (their value times cap / their number when more): in long
    losses the learner made ~33 workers to the AI's ~19, and every one raised the shaping's potential."""
    out = {0: 0.0, 1: 0.0}
    ua = getattr(obs, "unit_array", None)
    if ua is not None and len(ua):  # native observations: vectorized
        import numpy as np
        live = ((ua[:, 11] & DEAD_FLAG) == 0) & ((ua[:, 2] == 0) | (ua[:, 2] == 1))
        a = ua[live]
        v = _type_values(values, a[:, 1])
        if hero_level:  # (column 14: the hero's level, 0 for other units)
            v = v + hero_level * np.maximum(a[:, 14] - 1, 0)
        hp = np.where(a[:, 7] > 0, a[:, 6] / np.maximum(a[:, 7], 1), 1.0)
        if worker_cap:
            work = np.isin(a[:, 1], list(WORKER_IDS))
            for p in (0, 1):
                n = int((work & (a[:, 2] == p)).sum())
                if n > worker_cap:
                    v = np.where(work & (a[:, 2] == p), v * worker_cap / n, v)
        vh = v * hp
        for p in (0, 1):
            out[p] = float(vh[a[:, 2] == p].sum())
        return out
    live = [u for u in obs.units if u.owner in (0, 1) and not int(u.flags) & DEAD_FLAG]
    n = {p: sum(u.type_id in WORKER_IDS for u in live if u.owner == p) for p in (0, 1)}
    for u in live:
        v = values.get(str(u.type_id), 0) + hero_level * max(int(getattr(u, "hero_level", 0) or 0) - 1, 0)
        if worker_cap and u.type_id in WORKER_IDS and n[u.owner] > worker_cap:
            v *= worker_cap / n[u.owner]
        out[u.owner] += v * (u.hp / u.max_hp if u.max_hp > 0 else 1.0)
    return out


def unit_values() -> dict[str, int]:
    """What each unit type costs (gold + lumber, the game's own tables): {type id: value}."""
    from ..data.mpq import GameArchives
    from ..data.objects import parse_slk
    with GameArchives() as g:
        rows = parse_slk(g.read("Units\\UnitBalance.slk").decode("latin-1"))
    out = {}
    for r in rows:
        oid = r.get("unitBalanceID", "")
        if len(oid) != 4:
            continue
        try:
            v = int(float(r.get("goldcost") or 0)) + int(float(r.get("lumbercost") or 0))
        except ValueError:
            continue
        if v > 0:
            out[str(int.from_bytes(oid.encode("latin-1"), "big"))] = v
    return out


def material_steps(units, steps: int, values: dict) -> "np.ndarray":
    """A demonstration's material (as material()) at each step: [steps, 2]. `units`: its unit rows
    (fullgame.features.load_game: step, id, type, owner, x, y, facing, hp, max hp, ..., flags)."""
    import numpy as np
    live = ((units[:, 12] & DEAD_FLAG) == 0) & ((units[:, 3] == 0) | (units[:, 3] == 1))
    a = units[live]
    types, inv = np.unique(a[:, 2], return_inverse=True)
    v = np.array([values.get(str(t), 0) for t in types.tolist()], float)[inv]
    hp = np.where(a[:, 8] > 0, a[:, 7] / np.maximum(a[:, 8], 1), 1.0)
    return np.bincount(a[:, 0] * 2 + a[:, 3], weights=v * hp, minlength=2 * steps)[:2 * steps].reshape(steps, 2)


WORKERS = {"hpea", "opeo", "uaco", "ewsp"}
HALLS = {"htow", "ogre", "unpl", "etol"}  # (a race's first-tier hall: one more is an expansion if at another mine)
MINES = {"ngol", "ugol", "egol"}  # (a gold mine, haunted or entangled)
EXPANSION_FROM_START = 1500.0  # a hall this far from its player's start, and
EXPANSION_AT_MINE = 1000.0  # this near a gold mine: an expansion (a second hall at home is none)


HIRED = {int.from_bytes(c.encode(), "big") for c in
         ("Nalc", "Ntin", "Nfir", "Nngs", "Nbrn", "Npbm", "Nplh", "Nbst", "nftb", "nfsp", "nogm", "ngrk")}


class Production:
    """What each player makes, loses and holds over a game, from the observations' events: units
    trained and buildings finished by type, research, kills and losses, food at one minute,
    resources on hand. The episode rows' "prod" / "opp_prod" (the dashboard's Production tab)."""

    def __init__(self):
        from collections import Counter
        self.trained = {0: Counter(), 1: Counter()}
        self.built = {0: Counter(), 1: Counter()}
        self.research: dict[int, dict[str, int]] = {0: {}, 1: {}}
        self.kills, self.lost = {0: 0, 1: 0}, {0: 0, 1: 0}
        self.held, self.food_1min, self.max_food = {0: 0.0, 1: 0.0}, {}, {0: 0, 1: 0}
        self.steps = 0
        # gathered so far at the game's start: a scripted reset (games without the built-in AI) keeps
        # the game's counters, so without this they added up over a launch's games (185k gold)
        self.base: dict[int, tuple[int, int]] = {}
        self.hired: set[int] = set()  # (tavern heroes and mercenaries seen: hired, no training event)
        self.mines: list[tuple[float, float]] | None = None  # the gold mines (their first step's)
        self.expansions: dict[int, list[float]] = {0: [], 1: []}  # each expansion's game second
        self.halls_home = {0: 0, 1: 0}  # more halls that are no expansion (at home, or at no mine)
        self.creeps = {0: 0, 1: 0}  # neutral units killed (creeping)
        from .features import codes
        self._hired_codes = codes(HIRED)  # (features.isin's)

    def _hires(self, obs) -> None:
        """A tavern hero or a mercenary of a player's seen for the first time: trained (bought, protocol.Buy)."""
        from .features import isin, rawcode
        ua = getattr(obs, "unit_array", None)  # (their rows by a mask: each row in Python had been ~2% of a step)
        rows = ([(int(i), int(t), int(o)) for i, t, o in ua[isin(ua[:, 1], self._hired_codes), :3].tolist()]
                if ua is not None else [(u.id, u.type_id, u.owner) for u in obs.units if u.type_id in HIRED])
        for i, t, o in rows:
            if i not in self.hired and o in (0, 1):
                self.hired.add(i)
                if self.steps > 1:  # (not one the map had at its start)
                    self.trained[o][rawcode(t)] += 1

    def step(self, obs) -> None:
        from .features import rawcode
        self.steps += 1
        for p in (0, 1):
            s = obs.players.get(p)
            if s is None:
                continue
            self.base.setdefault(p, (s.gold_gathered, s.lumber_gathered))
            self.held[p] += s.gold + s.lumber
            self.max_food[p] = max(self.max_food[p], s.food_used)
            if obs.game_time >= 60 and p not in self.food_1min:
                self.food_1min[p] = s.food_used
        self._hires(obs)
        if self.mines is None:
            self.mines = [(u.x, u.y) for u in obs.units if rawcode(u.type_id) in MINES]
        events = [e for e in obs.events if int(e.kind) in (1, 3, 6, 9, 12)]
        if not events:
            return
        ua = getattr(obs, "unit_array", None)
        owner = (dict(zip(ua[:, 0].tolist(), ua[:, 2].tolist())) if ua is not None and len(ua)
                 else {u.id: u.owner for u in obs.units})
        for e in events:
            k = int(e.kind)
            if k == 6:  # a unit trained: a=building, b=the unit, c=its type
                p = owner.get(e.a)
                if p in (0, 1):
                    self.trained[p][rawcode(e.c)] += 1
            elif k in (3, 12):  # a building finished (b=type), or upgraded (b=the new type)
                p = owner.get(e.a)
                if p in (0, 1):
                    self.built[p][rawcode(e.b)] += 1
                    if k == 3 and (rawcode(e.b) in HALLS or rawcode(e.b) == "ugol"):  # (undead's: a haunted mine)
                        self._hall(p, e.a, obs)
            elif k == 9:  # research done: b=tech, c=level
                p = owner.get(e.a)
                if p in (0, 1):
                    code = rawcode(e.b)
                    self.research[p][code] = max(self.research[p].get(code, 0), e.c)
            elif k == 1:  # a death: a=the unit, b=its killer
                dead, killer = owner.get(e.a), owner.get(e.b)
                if dead in (0, 1):
                    self.lost[dead] += 1
                    if killer in (0, 1) and killer != dead:
                        self.kills[killer] += 1
                elif dead is not None and killer in (0, 1):  # (a neutral unit: a creep)
                    self.creeps[killer] += 1

    def _hall(self, p: int, unit: int, obs) -> None:
        """A hall of player p finished: an expansion when far from its start and at a gold mine."""
        u = next((x for x in obs.units if x.id == unit), None)
        s = obs.players.get(p)
        if u is None or s is None:
            return
        home = ((u.x - s.start_x) ** 2 + (u.y - s.start_y) ** 2) ** 0.5
        mine = min((((u.x - x) ** 2 + (u.y - y) ** 2) ** 0.5 for x, y in self.mines or ()), default=1e9)
        if home > EXPANSION_FROM_START and mine < EXPANSION_AT_MINE:
            self.expansions[p].append(round(obs.game_time))
        else:
            self.halls_home[p] += 1

    def row(self, p: int, obs) -> dict:
        s = obs.players.get(p)
        t = self.trained[p]
        return {"trained": dict(t), "built": dict(self.built[p]), "research": self.research[p],
                "workers": sum(v for k, v in t.items() if k in WORKERS),
                "heroes": sum(v for k, v in t.items() if k[:1].isupper()),
                "army": sum(v for k, v in t.items() if k not in WORKERS and not k[:1].isupper()),
                "kills": self.kills[p], "lost": self.lost[p], "food_1min": self.food_1min.get(p),
                "max_food": self.max_food[p], "held": round(self.held[p] / max(self.steps, 1)),
                "gold": s.gold_gathered - self.base.get(p, (0, 0))[0] if s else 0,
                "lumber": s.lumber_gathered - self.base.get(p, (0, 0))[1] if s else 0,
                "expansions": len(self.expansions[p]), **({"expansion_t": self.expansions[p][0]} if self.expansions[p] else {}),
                "halls_home": self.halls_home[p], "creeps": self.creeps[p]}
