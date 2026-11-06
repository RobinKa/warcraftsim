"""What the game tells a JASS harness about the things the agent's observation lacks (docs/reference/observation-coverage.md):
buffs, unit states, invisibility, corpses, spell target points, item charges, rally points, enemy research, fog.

    python3 scripts/obs_probe.py buffs --race human --foe orc       # human's and orc's spells on each other's units
    python3 scripts/obs_probe.py buffs --race undead --foe nightelf

One game (echofast2, both players agents, victory none, no warm spare). The harness is this repo's with one more function
(W3S_Probe, called before each observation is written; its own map build and cache key): for every unit of players 0 and 1
and every dead unit it adds events of kinds 900+ (protocol.Event; the parser keeps unknown kinds as integers):

  900 unit, type, state bits: dead, invisible to the other player (IsUnitInvisible), visible to it (IsUnitVisible),
      detected by it, fogged and masked for it, hidden, paused, and the classifications stunned, snared, poisoned,
      polymorphed, sleeping, ethereal, magic immune, plagued (IsUnitType), illusion, invulnerable (BlzIsUnitInvulnerable)
  901-903 unit, bits of BUFFS (GetUnitAbilityLevel(unit, buff) > 0), 31 a word
  904 unit, armor x 10 (BlzGetUnitArmor), move speed (GetUnitMoveSpeed)
  905 unit, attack cooldown in ms (BlzGetUnitAttackCooldown, weapon 0), base damage (BlzGetUnitBaseDamage, weapon 0)
  906 unit, the same for weapon 1
  907 unit, positive and negative buffs (UnitCountBuffsEx, all kinds)
  908 unit, positive and negative magic buffs, not auras or timed life (UnitCountBuffsEx)
  909 building, rally point x, y (GetUnitRallyPoint)
  910 caster, spell target x, y (GetSpellTargetX/Y at EVENT_PLAYER_UNIT_SPELL_EFFECT)
  911 caster, ability, target unit (the same event)
  912 unit, ability, cooldown left x 10 (BlzGetUnitAbilityCooldownRemaining) of unit abilities in UNIT_COOLDOWNS
  914 hero, item type, slot x 1000 + charges (GetItemCharges)
  915 dead unit, x, y (a corpse: what raise dead, cannibalize and the meat wagon take)
  916 player, research, its level (GetPlayerTechCount: any player's, the enemy's too), every 20 steps
  917 player, point (0 own start, 1 the other start, 2 the map's centre), visible + 2 fogged + 4 masked
  918 time of day x 100 (GetFloatGameState(GAME_STATE_TIME_OF_DAY))

Printed: per phase each probed unit's decoded state against what the agent's record (protocol.Unit) has; the spells'
points; corpses and how long each stays enumerable after its death; research levels; fog at the starts; the clock.
Instance names start with "obsprobe"; one game at a time. `jass` prints the probe's JASS.

Results (2026-11-06; human-orc, orc-nightelf, undead-human, undead-nightelf):
  buffs       every buff cast showed (Binf Binv Bply Bslo Bblo BOwk Beng Bprg BUfa BUhf Bam2 Bcri Bcrs Bfae BEer Broa Brej
              Bspo, auras BHab BEah). The code is the ability's BuffID spelling: Unholy Frenzy's BUhf (AbilityBuffData's
              Buhf reads 0). Defend sets no buff (only the move speed, 188).
  states      snared (ensnare, entangling roots), poisoned (slow poison), polymorphed (polymorph), magic immune (dryad);
              faerie fire makes the unit detected. BlzIsUnitInvulnerable: true for every unit (unusable). IsUnitInvisible
              is true whenever the other player doesn't see the unit (fogged, masked too): invisible and neither fogged
              nor masked is a truly invisible one; an invisible footman next to a ghoul stayed out of the enemy's
              visibility bits (the agent's record) through the fight.
  counts      UnitCountBuffsEx ANDs its filters: all true counts 0; (positive or negative, magic, nothing else) counts
              bloodlust, inner fire / slow (auras and purge's Bprg not).
  numbers     BlzGetUnitArmor with upgrades and buffs (footman 8 at Rhar 3, 13 with inner fire; faerie fire 7 -> 3);
              GetUnitMoveSpeed with buffs (bloodlust 337, slow 150, ensnared 0, sheep 100-125, wind walk 447);
              BlzGetUnitAttackCooldown / BlzGetUnitBaseDamage: weapon index 1 is the first (0 reads 0), the base interval
              (bloodlust unchanged) and base damage (no upgrades). Polymorph keeps the type id and hp.
  corpses     dead units enumerable with their place for 184-186 steps (~92 s); heroes' 1-6 s. Raise dead and
              cannibalize as immediate orders refused next to corpses.
  other       GetSpellTargetX/Y at the effect: a point spell's point, a unit spell's target place, (0, 0) at once;
              GetItemCharges, GetUnitRallyPoint, BlzGetUnitAbilityCooldownRemaining on unit abilities (ensnare 11.1 s),
              GetPlayerTechCount on the enemy, IsVisible/Fogged/MaskedToPlayer at points; trees' life 50 (their lumber);
              the game's clock (GAME_STATE_TIME_OF_DAY) 0.2-0.5 h behind features' (8:00 + steps x 24 / 480 s).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"

ROOT = Path(__file__).resolve().parents[1]
RUN_CUTS = 17592186041849  # (fgself-194's --game-cuts: the shim reads the units' objects, which the buff list needs)

# the buffs the probe asks each unit for: the melee races', their heroes', the tavern's, creeps' and items' (a subset of
# scripts/observation_coverage.py's universe; 93 at most: 3 words of 31 bits)
BUFFS = (
    # generic / several sources
    "BSTN", "BPSE", "Bfro", "Bfre", "Bpoi", "Bvul", "Bdet", "Bspe", "BHca", "BNba", "BNsi", "BNdo", "BNms",
    "BNht", "BNdh", "BNso", "BNbr", "Bfzy", "BIrg", "BIrl", "Bams",
    # human
    "Binf", "Binv", "Bivs", "Bply", "Bslo", "Bdef", "Bmil", "BHab", "BHad", "BHav", "BHbn", "BHbd", "BHds", "BHfs",
    "BHtc", "Bcmg", "Bclf", "Bmlt", "Bpxf",
    # orc
    "Bblo", "Blsh", "Bprg", "Bens", "Bena", "Beng", "Bspl", "Bbsk", "Beye", "Bhwd", "Bliq", "BOac", "BOae", "BOeq",
    "BOhx", "BOmi", "BOvd", "BOwk", "Bdig",
    # undead
    "Bam2", "Babr", "Bapl", "Bcri", "Bcrs", "Bfrz", "Bplg", "Bpos", "Brpl", "Brpm", "BUhf", "Bweb", "Bwea", "BUan",
    "BUau", "BUav", "BUdd", "BUfa", "BUim", "BUsl", "BUsp",
    # night elf
    "Bbar", "Bcor", "Bcyc", "Bfae", "Bpsh", "Brej", "Broa", "Bspo", "BEah", "BEar", "BEer", "BEim", "BEme",
)
assert len(BUFFS) <= 93 and len(set(BUFFS)) == len(BUFFS), len(BUFFS)
UNIT_COOLDOWNS = ("Aens", "Aprg", "Absk", "Aply", "Acyc", "Arai", "Acrs", "Aweb", "Afae", "Aroa", "Arej", "Aeye", "Aslo")
TECHS = ("Rhme", "Rhar", "Rhpt", "Rhst", "Rhde", "Rome", "Rora", "Rost", "Roen", "Rume", "Ruar", "Rune", "Ruba",
         "Ruwb", "Resm", "Rema", "Redt", "Redc", "Rers")
STATE_BITS = ("dead", "invisible", "visible", "detected", "fogged", "masked", "hidden", "paused", "stunned", "snared",
              "poisoned", "polymorphed", "sleeping", "ethereal", "magic_immune", "plagued", "illusion", "invulnerable")


def probe_jass() -> str:
    """W3S_Probe and its helpers (JASS), defined before W3S_Step."""
    L = ["function W3S_ProbeSpell takes nothing returns boolean",
         "    call W3S_Ev(910, GetHandleId(GetTriggerUnit()), R2I(GetSpellTargetX()), R2I(GetSpellTargetY()))",
         "    call W3S_Ev(911, GetHandleId(GetTriggerUnit()), GetSpellAbilityId(), W3S_Hid(GetSpellTargetUnit()))",
         "    return false", "endfunction", "",
         "function W3S_ProbeUnit takes unit u returns nothing",
         "    local integer hid = GetHandleId(u)",
         "    local integer o = GetPlayerId(GetOwningPlayer(u))",
         "    local player other = Player(0)",
         "    local integer f = 0",
         "    local integer m = 0",
         "    local integer k = 0",
         "    local location l",
         "    local item it",
         "    if o == 0 then", "        set other = Player(1)", "    endif"]
    checks = ["IsUnitType(u, UNIT_TYPE_DEAD)", "IsUnitInvisible(u, other)", "IsUnitVisible(u, other)",
              "IsUnitDetected(u, other)", "IsUnitFogged(u, other)", "IsUnitMasked(u, other)", "IsUnitHidden(u)",
              "IsUnitPaused(u)", "IsUnitType(u, UNIT_TYPE_STUNNED)", "IsUnitType(u, UNIT_TYPE_SNARED)",
              "IsUnitType(u, UNIT_TYPE_POISONED)", "IsUnitType(u, UNIT_TYPE_POLYMORPHED)",
              "IsUnitType(u, UNIT_TYPE_SLEEPING)", "IsUnitType(u, UNIT_TYPE_ETHEREAL)",
              "IsUnitType(u, UNIT_TYPE_MAGIC_IMMUNE)", "IsUnitType(u, UNIT_TYPE_PLAGUED)", "IsUnitIllusion(u)",
              "BlzIsUnitInvulnerable(u)"]
    assert len(checks) == len(STATE_BITS)
    for i, c in enumerate(checks):
        L += [f"    if {c} then", f"        set f = f + {1 << i}", "    endif"]
    L.append("    call W3S_Ev(900, hid, GetUnitTypeId(u), f)")
    for w in range(3):
        part = BUFFS[31 * w:31 * w + 31]
        if not part:
            break
        L.append("    set m = 0")
        for i, b in enumerate(part):
            L += [f"    if GetUnitAbilityLevel(u, '{b}') > 0 then", f"        set m = m + {1 << i}", "    endif"]
        L += ["    if m != 0 then", f"        call W3S_Ev({901 + w}, hid, m, 0)", "    endif"]
    L += ["    call W3S_Ev(904, hid, R2I(BlzGetUnitArmor(u) * 10.0), R2I(GetUnitMoveSpeed(u)))",
          "    call W3S_Ev(905, hid, R2I(BlzGetUnitAttackCooldown(u, 0) * 1000.0), BlzGetUnitBaseDamage(u, 0))",
          "    call W3S_Ev(906, hid, R2I(BlzGetUnitAttackCooldown(u, 1) * 1000.0), BlzGetUnitBaseDamage(u, 1))",
          "    call W3S_Ev(907, hid, UnitCountBuffsEx(u, true, false, true, true, true, true, true), "
          "UnitCountBuffsEx(u, false, true, true, true, true, true, true))",
          "    call W3S_Ev(908, hid, UnitCountBuffsEx(u, true, false, true, false, false, false, false), "
          "UnitCountBuffsEx(u, false, true, true, false, false, false, false))",
          "    if IsUnitType(u, UNIT_TYPE_STRUCTURE) then",
          "        set l = GetUnitRallyPoint(u)",
          "        if l != null then",
          "            call W3S_Ev(909, hid, R2I(GetLocationX(l)), R2I(GetLocationY(l)))",
          "            call RemoveLocation(l)",
          "        endif",
          "    endif"]
    for a in UNIT_COOLDOWNS:
        L += [f"    if GetUnitAbilityLevel(u, '{a}') > 0 then",
              f"        call W3S_Ev(912, hid, '{a}', R2I(BlzGetUnitAbilityCooldownRemaining(u, '{a}') * 10.0))",
              "    endif"]
    L += ["    if IsUnitType(u, UNIT_TYPE_HERO) then",
          "        loop", "            exitwhen k >= 6",
          "            set it = UnitItemInSlot(u, k)",
          "            if it != null then",
          "                call W3S_Ev(914, hid, GetItemTypeId(it), k * 1000 + GetItemCharges(it))",
          "            endif",
          "            set k = k + 1", "        endloop", "    endif",
          "    if IsUnitType(u, UNIT_TYPE_DEAD) then",
          "        call W3S_Ev(915, hid, R2I(GetUnitX(u)), R2I(GetUnitY(u)))",
          "    endif",
          "    set other = null", "    set l = null", "    set it = null", "endfunction", "",
          "function W3S_ProbeSees takes integer p, integer k, real x, real y returns nothing",
          "    local integer v = 0",
          "    if IsVisibleToPlayer(x, y, Player(p)) then", "        set v = v + 1", "    endif",
          "    if IsFoggedToPlayer(x, y, Player(p)) then", "        set v = v + 2", "    endif",
          "    if IsMaskedToPlayer(x, y, Player(p)) then", "        set v = v + 4", "    endif",
          "    call W3S_Ev(917, p, k, v)", "endfunction", "",
          "function W3S_Probe takes nothing returns nothing",
          "    local group g = CreateGroup()",
          "    local unit u",
          "    local integer i = 0",
          "    local integer p = 0",
          "    local trigger t",
          "    if not LoadBoolean(w3s_ht, -7701, 0) then",
          "        call SaveBoolean(w3s_ht, -7701, 0, true)",
          "        set t = CreateTrigger()",
          "        loop", "            exitwhen i >= bj_MAX_PLAYERS",
          "            call TriggerRegisterPlayerUnitEvent(t, Player(i), EVENT_PLAYER_UNIT_SPELL_EFFECT, null)",
          "            set i = i + 1", "        endloop",
          "        call TriggerAddCondition(t, Condition(function W3S_ProbeSpell))",
          "        set t = null",
          "    endif",
          "    call GroupEnumUnitsInRect(g, bj_mapInitialPlayableArea, null)",
          "    loop",
          "        set u = FirstOfGroup(g)",
          "        exitwhen u == null",
          "        call GroupRemoveUnit(g, u)",
          "        if GetPlayerId(GetOwningPlayer(u)) < 2 or IsUnitType(u, UNIT_TYPE_DEAD) then",
          "            call W3S_ProbeUnit(u)",
          "        endif",
          "    endloop",
          "    call DestroyGroup(g)",
          "    set g = null",
          "    if ModuloInteger(w3s_seq, 20) == 0 then",
          "        loop", "            exitwhen p >= 2"]
    for tech in TECHS:
        L.append(f"            call W3S_Ev(916, p, '{tech}', GetPlayerTechCount(Player(p), '{tech}', true))")
    L += ["            call W3S_ProbeSees(p, p, GetStartLocationX(GetPlayerStartLocation(Player(p))), "
          "GetStartLocationY(GetPlayerStartLocation(Player(p))))",
          "            call W3S_ProbeSees(p, 1 - p, GetStartLocationX(GetPlayerStartLocation(Player(1 - p))), "
          "GetStartLocationY(GetPlayerStartLocation(Player(1 - p))))",
          "            call W3S_ProbeSees(p, 2, 0.0, 0.0)",
          "            set p = p + 1", "        endloop",
          "        call W3S_Ev(918, R2I(GetFloatGameState(GAME_STATE_TIME_OF_DAY) * 100.0), 0, 0)",
          "    endif",
          "endfunction", ""]
    return "\n".join(L) + "\n"


def patch_harness() -> None:
    """mapbuild.harness_source -> the harness with W3S_Probe (this process only; its own cache key, instance.map_key)."""
    from warcraftsim.data import mapbuild
    src = mapbuild.harness_source()
    anchor = "function W3S_Step takes nothing returns nothing\n"
    call = "    call W3S_CheckResult()\n    call W3S_WriteObs()\n"
    assert src.count(anchor) == 1 and src.count(call) == 1
    patched = src.replace(anchor, probe_jass() + anchor).replace(call, "    call W3S_CheckResult()\n    call W3S_Probe()\n"
                                                                       "    call W3S_WriteObs()\n")
    mapbuild.harness_source = lambda: patched


def code(c: str) -> int:
    return int.from_bytes(c.encode("latin-1"), "big")


def raw(v: int) -> str:
    try:
        return int(v).to_bytes(4, "big").decode("latin-1") if v > 0 else str(v)
    except OverflowError:
        return str(v)


class Probe:
    def __init__(self, g):
        self.g = g
        self.obs = g.start()
        self.dests = list(self.obs.destructables or [])  # (the first observation's: trees, with their life)
        self.t = 0
        self.rec: dict[int, dict] = {}  # unit -> this step's probe records
        self.spells: list = []
        self.techs: dict = {}
        self.sees: dict = {}
        self.tod = None
        self.dead_seen: dict[int, list[int]] = defaultdict(list)  # corpse -> steps it was enumerated
        self.died_at: dict[int, int] = {}
        self.full_seen: dict[int, dict] = {}  # (--obs-full: the last event 101 of each unit, its buffs, the others)
        self.full_buffs: dict[int, dict] = {}
        self.full_events: list = []
        self.take(self.obs)

    def take(self, obs) -> None:
        rec: dict[int, dict] = defaultdict(dict)
        pending: dict[int, tuple] = {}
        for e in obs.events:
            k = int(e.kind)
            if k == 1:
                self.died_at.setdefault(e.a, self.t)
            if k < 100:
                continue
            if k == 900:
                rec[e.a].update(type=raw(e.b), state=[n for i, n in enumerate(STATE_BITS) if e.c >> i & 1])
            elif k in (901, 902, 903):
                rec[e.a].setdefault("buffs", []).extend(b for i, b in enumerate(BUFFS[31 * (k - 901):31 * (k - 900)])
                                                        if e.b >> i & 1)
            elif k == 904:
                rec[e.a].update(armor=e.b / 10.0, speed=e.c)
            elif k == 905:
                rec[e.a].update(cooldown_ms=e.b, damage=e.c)
            elif k == 906:
                rec[e.a].update(cooldown2_ms=e.b, damage2=e.c)
            elif k == 907:
                rec[e.a].update(pos_all=e.b, neg_all=e.c)
            elif k == 908:
                rec[e.a].update(pos_magic=e.b, neg_magic=e.c)
            elif k == 909:
                rec[e.a].update(rally=(e.b, e.c))
            elif k == 910:
                pending[e.a] = (e.b, e.c)
            elif k == 911:
                self.spells.append((self.t, e.a, raw(e.b), e.c, pending.pop(e.a, None)))
            elif k == 912:
                rec[e.a].setdefault("cooldowns", {})[raw(e.b)] = e.c / 10.0
            elif k == 914:
                rec[e.a].setdefault("items", []).append((raw(e.b), e.c // 1000, e.c % 1000))
            elif k == 915:
                rec[e.a].update(corpse=(e.b, e.c))
                self.dead_seen[e.a].append(self.t)
            elif k == 916:
                self.techs[(e.a, raw(e.b))] = e.c
            elif k == 917:
                self.sees[(e.a, e.b)] = [n for i, n in enumerate(("visible", "fogged", "masked")) if e.c >> i & 1]
            elif k == 918:
                self.tod = e.a / 100.0
            elif k == 101:  # (--obs-full's own events: the harness and the shim as a run has them)
                self.full_buffs.pop(e.a, None)
                speed, armor = e.c >> 12, ((e.c & 4095) - 2048) / 10.0
                rec[e.a].update(full={"class": [n for i, n in enumerate(("stunned", "snared", "poisoned", "polymorphed",
                                                                        "sleeping", "ethereal", "magic_immune",
                                                                        "plagued")) if e.b >> i & 1],
                                      "pos_magic": e.b >> 8 & 15, "neg_magic": e.b >> 12 & 15, "nbuffs": e.b >> 16 & 15,
                                      "listed": bool(e.b >> 20 & 1), "speed": speed, "armor": armor})
                self.full_seen[e.a] = rec[e.a]["full"]
            elif k == 102:
                rec[e.a].setdefault("full_buffs", []).append(raw(e.b))
                self.full_buffs.setdefault(e.a, {})[e.c] = raw(e.b)
            elif k == 105:
                rec[e.a].update(charges=[(e.b if i < 3 else e.c) >> (10 * (i % 3)) & 1023 for i in range(6)])
            elif k in (104, 106, 107, 108, 109):
                self.full_events.append((self.t, k, e.a, raw(e.b) if k == 109 else e.b, e.c))
        self.rec = rec

    def step(self, cmds=()):
        self.obs = self.g.step(list(cmds))
        self.t += 1
        self.take(self.obs)
        return list(self.obs.command_results)

    def wait(self, n: int) -> None:
        for _ in range(n):
            self.step()

    def place(self, player: int, kind: str, x: float, y: float, **kw):
        from warcraftsim.protocol import Place
        before = {u.id for u in self.obs.units}
        self.step([Place(player, kind, x, y, **kw)])
        new = [u for u in self.obs.units if u.id not in before and u.owner == player and u.type == kind]
        return new[0].id if new else None

    def unit(self, uid):
        return self.obs.unit(uid) if uid is not None else None

    def show(self, label: str, uids: dict) -> None:
        print(f"== {label} (step {self.t})", flush=True)
        for name, uid in uids.items():
            u, r = self.unit(uid), dict(self.rec.get(uid, {}))
            if uid in self.full_seen:  # (--obs-full's events come when they change: the last ones)
                r["full"] = self.full_seen[uid]
                r["full_buffs"] = [b for _, b in sorted(self.full_buffs.get(uid, {}).items())][:r["full"]["nbuffs"]]
            agent = None if u is None else {"type": u.type, "x": u.x, "y": u.y, "hp": u.hp, "mana": u.mana, "order": u.order,
                                            "flags": int(u.flags), "visible_to": u.visible_to}
            print("RESULT " + json.dumps({"unit": name, "agent's record": agent, "probe": r}), flush=True)


def research(p: Probe, player: int, race: str) -> None:
    from warcraftsim.fullgame import techtree as tt
    from warcraftsim.protocol import SetResearch, SetResources
    letter = race[0] if race != "nightelf" else "e"
    codes = sorted(c for c, makers in tt.tree().makers.items() if c[0] == "R" and any(m[0] == letter for m in makers))
    lv = tt.research_levels()
    p.step([SetResources(player, 90000, 90000)] + [SetResearch(player, c, lv.get(c, 1)) for c in codes])


# per race: the units placed, the casts on own units (phase A: the foes far), the casts on the foes (phase B: the foes
# near), each (caster, order, target): target ("own", unit) / ("foe", unit) / ("at_foe", unit) / None (at once)
SIDES = {
    "human": {"units": {"priest": ("hmpr", {}), "sorceress": ("hsor", {}), "foot1": ("hfoo", {}), "foot2": ("hfoo", {}),
                        "archmage": ("Hamg", {"hero_level": 6, "skills": ("AHbz", "AHbz", "AHbz", "AHab", "AHab", "AHmt"),
                                               "items": ("stwp", "phea", "pman")}),
                        "barracks": ("hbar", {})},
              "own": [("priest", "innerfire", ("own", "foot1")), ("sorceress", "invisibility", ("own", "foot2")),
                      ("foot1", "defend", None)],
              "foe": [("sorceress", "slow", ("foe", "b2")), ("sorceress", "polymorph", ("foe", "b1")),
                      ("archmage", "blizzard", ("at_foe", "b2"))]},
    "orc": {"units": {"shaman": ("oshm", {}), "raider": ("orai", {}), "grunt1": ("ogru", {}), "grunt2": ("ogru", {}),
                      "blademaster": ("Obla", {"hero_level": 3, "skills": ("AOwk", "AOwk", "AOmi")}),
                      "barracks": ("obar", {})},
            "own": [("shaman", "bloodlust", ("own", "grunt1")), ("blademaster", "windwalk", None)],
            "foe": [("raider", "ensnare", ("foe", "b1")), ("shaman", "purge", ("foe", "b2")),
                    ("blademaster", "mirrorimage", None)]},
    "undead": {"units": {"lich": ("Ulic", {"hero_level": 3, "skills": ("AUfn", "AUfa", "AUfa")}),
                         "banshee": ("uban", {}), "fiend": ("ucry", {}), "ghoul1": ("ugho", {}), "ghoul2": ("ugho", {}),
                         "necro": ("unec", {}), "barracks": ("usep", {})},
               "own": [("lich", "frostarmor", ("own", "ghoul1")), ("necro", "unholyfrenzy", ("own", "ghoul2")),
                       ("banshee", "antimagicshell", ("own", "fiend"))],
               "foe": [("banshee", "curse", ("foe", "b1")), ("lich", "frostnova", ("foe", "b2")),
                       ("fiend", "web", ("foe", "b2"))]},
    "nightelf": {"units": {"keeper": ("Ekee", {"hero_level": 3, "skills": ("AEer", "AEer", "AEah")}),
                           "talon": ("edot", {}), "claw": ("edoc", {}), "archer1": ("earc", {}), "archer2": ("earc", {}),
                           "dryad": ("edry", {}), "barracks": ("eaom", {})},
                 "own": [("claw", "roar", None), ("claw", "rejuvination", ("own", "archer2"))],
                 "foe": [("talon", "faeriefire", ("foe", "b1")), ("talon", "cyclone", ("foe", "b2")),
                         ("keeper", "entanglingroots", ("foe", "b1"))]},
}
SNEAK = {"human": "foot2", "orc": "blademaster"}  # (made invisible in phase A, then walked up to the foes)
TARGETS = {"human": ("foot1", "foot2"), "orc": ("grunt1", "grunt2"), "undead": ("ghoul1", "ghoul2"),
           "nightelf": ("archer1", "archer2")}  # (b1, b2: each side's units the other aims at)
CORPSE_USERS = {"undead": (("ghoul1", "cannibalize"), ("necro", "raisedead")), "orc": (), "human": (), "nightelf": ()}


def buffs(a) -> int:
    from warcraftsim.data.mapbuild import stock_map_path  # noqa: F401  (the map's files present)
    from warcraftsim.fullgame import fullvocab as fv
    from warcraftsim.protocol import ImmediateOrder, PointOrder, SetResources, TargetOrder
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
    if a.what == "full":  # (the real --obs-full: the harness's W3S_CFG_OBS_FULL and the shim's extras, checked)
        os.environ["W3SIM_OBS_CHECK"] = "1"
    else:
        patch_harness()
    ids = fv.order_ids()
    slots = [Agent(a.race), Agent(a.foe)]
    gs = GameSetup(map=a.map, slots=slots, step_seconds=0.5, draw=False, native_obs=a.native, d3d_thread=False,
                   render_threads=0, nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="none",
                   shim_dir=a.shim, obs_full=a.what == "full", cuts=RUN_CUTS if a.what == "full" else None)
    with GameInstance(gs, name=f"obsprobe{a.name}", timeout=900) as g:
        p = Probe(g)
        me, foe = p.obs.players[0], p.obs.players[1]
        bx, by = me.start_x, me.start_y
        dx = 1 if bx < foe.start_x else -1
        research(p, 0, a.race)
        research(p, 1, a.foe)
        U: dict[str, int | None] = {}
        for side, race, ox in ((0, a.race, 700), (1, a.foe, 2300)):
            for k, (name, (kind, kw)) in enumerate(SIDES[race]["units"].items()):
                bld = name == "barracks"
                x = bx + dx * (ox if not bld else 400 + 1200 * side)
                y = by + (-700 if bld else 60 * k - 180)
                U[f"{side}:{name}"] = p.place(side, kind, x, y, **kw)
        print("placed " + json.dumps({k: v for k, v in U.items()}), flush=True)
        for side, race in ((0, a.race), (1, a.foe)):
            b1, b2 = TARGETS[race]
            U[f"{1 - side}:b1"], U[f"{1 - side}:b2"] = U[f"{side}:{b1}"], U[f"{side}:{b2}"]
        p.wait(2)
        watch = {k: v for k, v in U.items() if not k.endswith((":b1", ":b2"))}
        p.show("placed, before any cast", watch)

        def cast(side: int, caster: str, order: str, target):
            c = U.get(f"{side}:{caster}")
            oid = ids.get(order)
            if c is None or oid is None:
                print(f"   (no {caster} or no order {order})", flush=True)
                return None
            if target is None:
                return ImmediateOrder(c, oid)
            how, who = target
            t = U.get(f"{side}:{who}") if how == "own" else U.get(f"{side}:{who}")
            if how == "at_foe":
                tu = p.unit(t)
                return PointOrder(c, oid, tu.x, tu.y) if tu is not None else None
            return TargetOrder(c, oid, t) if t is not None else None

        # phase A: each side's spells on its own units (the foes 1,600 away)
        for side, race in ((0, a.race), (1, a.foe)):
            for caster, order, target in SIDES[race]["own"]:
                cmd = cast(side, caster, order, target)
                if cmd is not None:
                    res = p.step([cmd])
                    print(f"   cast {side}:{caster} {order} -> {res}", flush=True)
                    p.wait(3)
        rally = U.get("0:barracks")
        if rally is not None:
            res = p.step([PointOrder(rally, ids["setrally"], bx + dx * 900, by + 400)])
            print(f"   setrally -> {res}", flush=True)
        p.wait(4)
        p.show("phase A: own spells cast", watch)
        sneaks = {}
        for side, race in ((0, a.race), (1, a.foe)):
            u = U.get(f"{side}:{SNEAK.get(race, '')}")
            if u is not None:
                sneaks[f"{side}:{SNEAK[race]}"] = u
                p.step([PointOrder(u, ids["move"], bx + dx * (2150 if side == 0 else 850), by - 150)])
        if sneaks:
            p.wait(10)
            p.show("phase A2: the invisible units next to the foes (visible_to: bit 0 player 0, bit 1 player 1)",
                   {**sneaks, **{k: v for k, v in watch.items() if k.endswith(("foot1", "grunt1", "ghoul1", "archer1"))}})
        # phase B: the foes walk over; each side's spells on the other's units
        mid = (bx + dx * 1300, by)
        p.step([PointOrder(u, ids["move"], *mid) for k, u in U.items() if u is not None and k.startswith(("0:", "1:"))
                and "barracks" not in k and not k.endswith((":b1", ":b2"))])
        p.wait(4)
        for side, race in ((0, a.race), (1, a.foe)):
            for caster, order, target in SIDES[race]["foe"]:
                cmd = cast(side, caster, order, target)
                if cmd is not None:
                    res = p.step([cmd, SetResources(side, 90000, 90000)])
                    print(f"   cast {side}:{caster} {order} -> {res}", flush=True)
                    p.wait(2)
        p.wait(2)
        p.show("phase B: spells on the foes", watch)
        print("spells (step, caster, ability, target unit, target point): " + json.dumps(p.spells), flush=True)
        # phase C: the fight, then the corpses
        army = [u for k, u in watch.items() if u is not None and "barracks" not in k]
        p.step([PointOrder(u, ids["attack"], *mid) for u in army])
        p.wait(40)
        p.show("phase C: after 20 s of fighting", watch)
        for side, race in ((0, a.race), (1, a.foe)):
            for user, order in CORPSE_USERS.get(race, ()):
                c = U.get(f"{side}:{user}")
                if c is not None and order in ids:
                    res = p.step([ImmediateOrder(c, ids[order])])
                    print(f"   corpse order {side}:{user} {order} -> {res}", flush=True)
                    p.wait(4)
        for _ in range(int(a.corpse_steps)):
            p.step()
        print("corpses (unit: died at step, first and last step enumerated, steps): " + json.dumps(
            {u: (p.died_at.get(u), s[0], s[-1], len(s)) for u, s in p.dead_seen.items()}), flush=True)
        print("research levels (player, tech): " + json.dumps({f"{k[0]}:{k[1]}": v for k, v in sorted(p.techs.items())}),
              flush=True)
        print("fog (player, point 0 own start / 1 other start / 2 centre): " + json.dumps(
            {f"{k[0]}:{k[1]}": v for k, v in sorted(p.sees.items())}), flush=True)
        print(f"time of day: {p.tod} at step {p.t}", flush=True)
        if a.what == "full":
            print("obs-full events 104-109 (step, kind, a, b, c): " + json.dumps(p.full_events[:80]), flush=True)
            print("obs-full buffs seen per unit: " + json.dumps({u: sorted(set(b.values())) for u, b in p.full_buffs.items()}),
                  flush=True)
            log = g.inst_dir / "shim.log"
            if log.exists():
                print("shim log (obs-full): " + json.dumps([ln for ln in log.read_text(errors="replace").splitlines()
                                                             if "obs-full" in ln][-12:]), flush=True)
        from collections import Counter
        print("destructables in the first observation (type: count, life values): " + json.dumps(
            {t: (n, sorted({d.life for d in p.dests if d.type == t})[:6])
             for t, n in Counter(d.type for d in p.dests).most_common(8)}), flush=True)
    return 0


def cost(a) -> int:
    """The game's own time a step with --obs-full off and on: two built-in AIs (insane, both seers: record_ai_orders) on
    the map for --steps steps each, one game at a time; the game thread's wall time inside the game per step
    (GameInstance.game_wall_total) and the observation's tokens."""
    from warcraftsim.runtime.instance import BuiltinAI, GameInstance, GameSetup
    import pickle
    out = {}
    for full in (False, True, False, True):
        gs = GameSetup(map=a.map, slots=[BuiltinAI(a.race, "insane"), BuiltinAI(a.foe, "insane")], step_seconds=0.5,
                       draw=False, native_obs=bool(a.native), d3d_thread=False, render_threads=0, nice=0, warm_spare=False,
                       window=(320, 240), max_game_seconds=0, victory="decisive", shim_dir=a.shim, obs_full=full,
                       record_ai_orders=True, cuts=a.cuts if a.cuts else None)
        with GameInstance(gs, name=f"obsprobe{a.name}", timeout=900) as g:
            obs = g.start()
            w0, n_ev, keep = g.game_wall_total, 0, []
            for t in range(a.steps):
                obs = g.step([])
                n_ev += sum(1 for e in obs.events if int(e.kind) >= 100)
                if full and a.save and t < a.steps:
                    keep.append(obs)
                if obs.game_over:
                    break
            per = (g.game_wall_total - w0) / max(t + 1, 1) * 1000.0
            out.setdefault(full, []).append(per)
            print(f"RESULT obs_full={full}: {per:.2f} ms of game time a step over {t + 1} steps; events 100+: "
                  f"{n_ev / max(t + 1, 1):.1f} a step", flush=True)
            if keep and a.save:
                with open(a.save, "wb") as f:
                    pickle.dump(keep, f)
                print(f"saved {len(keep)} observations to {a.save}", flush=True)
    print("RESULT mean ms a step: " + json.dumps({str(k): sum(v) / len(v) for k, v in out.items()}), flush=True)
    return 0


def play(a) -> int:
    """A network (--ckpt, fgself-182's by default) loaded with --obs-full plays a real game on the CPU through the
    actors' path (play.BCAgent: the view, its masks, the network, the commands) against the built-in AI (insane) on the
    map, the harness and the shim with W3S_CFG_OBS_FULL: the entity kinds a step, the block's parts, refused orders."""
    import time
    import numpy as np
    import torch
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import model
    from warcraftsim.fullgame.costs import order_costs, research_steps
    from warcraftsim.fullgame.play import BCAgent
    from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance, GameSetup
    model.configure_for(a.ckpt)  # (its inputs, masks, strategy block, wares: as it was trained)
    fx.set_econ_masks(255)
    fx.set_obs_full(a.obs)
    fx.ACTIVE_F = 29
    fx.set_geometry(*fx.map_geometry(stock_map_path("(2)EchoIsles")))
    torch.set_num_threads(2)
    net, ck = model.load(a.ckpt, "cpu")
    vocab = ck["vocab"]
    print(f"net: obsfull {net.config.get('obsfull')}, types {net.config['n_types']}, F {fx.F}", flush=True)
    gs = GameSetup(map=a.map, slots=[Agent(a.race), BuiltinAI(a.foe, "insane")], step_seconds=0.5, draw=False,
                   native_obs=True, d3d_thread=False, render_threads=0, nice=19, warm_spare=False, window=(320, 240),
                   max_game_seconds=0, victory="decisive", shim_dir=a.shim, obs_full=bool(a.obs), cuts=RUN_CUTS)
    bot = BCAgent(net, vocab, 0, torch.device("cpu"), costs=order_costs(vocab, a.map), steps=research_steps(vocab, a.map))
    kinds, n_ent, n_res, refused, t_obs = np.zeros(3), [], [0, 0], 0, 0.0
    with GameInstance(gs, name=f"obsprobe{a.name}", timeout=900) as g:
        obs = g.start()
        bot.begin(obs, [fx.RACES.index(a.race), fx.RACES.index(a.foe)])
        for t in range(a.steps):
            t0 = time.perf_counter()
            cmds = bot.act(obs, t)
            t_obs += time.perf_counter() - t0
            st = getattr(bot, "_last_st", None)
            obs = g.step(cmds)
            res = list(obs.command_results)
            bot.accepted(cmds, res)
            n_res[0] += sum(res)
            n_res[1] += len(res)
            refused += len(res) - sum(res)
            v = bot.view
            if a.obs and getattr(v, "obs", None) is not None:
                kinds += [sum(1 for c in v.obs.corpses.values() if c[5] & 1), len(getattr(v, "ground", ()) or ()),
                          len(v.obs.areas)]
            if obs.game_over:
                break
    print("RESULT " + json.dumps({"steps": t + 1, "result": str(obs.players[0].result), "orders taken": n_res,
                                  "act ms a step (view, masks, network on the CPU)": round(t_obs / (t + 1) * 1000, 2),
                                  "corpses, ground items, areas seen a step": (kinds / (t + 1)).round(2).tolist()}),
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("buffs", "full", "cost", "play", "jass"))
    ap.add_argument("--ckpt", default="/home/tora/warcraftsim/runs/fgself-182/checkpoints/0000000050150172.pt")
    ap.add_argument("--obs", type=int, default=1, help="play: --obs-full")
    ap.add_argument("--native", type=int, default=0, help="the observation parsed in C (native_obs)")
    ap.add_argument("--steps", type=int, default=600, help="cost: steps a game")
    ap.add_argument("--cuts", type=int, default=0, help="cost: W3SIM_CUTS (0: the environment's)")
    ap.add_argument("--save", default="", help="cost: pickle the --obs-full game's observations here")
    ap.add_argument("--race", default="human")
    ap.add_argument("--foe", default="orc")
    ap.add_argument("--map", default="echofast2")
    ap.add_argument("--name", default="")
    ap.add_argument("--corpse-steps", type=int, default=200, help="steps to watch the corpses after the fight")
    ap.add_argument("--shim", default=str(ROOT / "build" / "shim"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    if a.what == "jass":
        print(probe_jass())
        return 0
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    return cost(a) if a.what == "cost" else play(a) if a.what == "play" else buffs(a)


if __name__ == "__main__":
    raise SystemExit(main())
