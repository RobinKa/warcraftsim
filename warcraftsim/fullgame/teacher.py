"""A heuristic teacher for what the policy rarely stumbles on (the user's idea, 2026-10-20: "buying a first hero with
a worker", on-policy distillation with our own advisor instead of the built-in AI). Its orders are never carried
out: they label the policy's step for the units it would order (as the target builder's do, --z-build 2), and the
cloning loss on those labels (--opd-coef) pulls the policy toward them while the reward decides whether they stay.
selfplay --teach 1 (needs --neutral 1: the orders it teaches are the neutral buildings' wares).

    a first hero:  (a game without a target) from FIRST_HERO_FROM, a side without a hero: its altar trains one of the
                   race's usual first heroes (the game's draw), and with no altar the worker nearest its hall builds one
                   450-800 from it, as the target builder places buildings. Without a target the human policy built an
                   altar in a fifth of its games (`fgself-134`): it rushed the built-in AI down and lost to policies.
    a tavern hero: the worker nearest the tavern hires one: with a target its tavern hero, when the side has as many
                   heroes as come before it in the target (its next hero); without, the game's draw of the race's usual
                   picks as the second hero, from 135 s. With an altar, the tier the hero count needs (2 for the second,
                   3 for the third) and the money
    a town portal: a hero near the goblin merchant buys one when it has none and a free slot, from 440 s
    a potion:      a hero below 40% of its hit points drinks what heals in its inventory
    an item:       (--macros 2) a hero with an item lying near (macros.pickable) and no enemy unit within ENEMY_NEAR
                   picks it up (the agent's pick-up order: creeps' drops lay where they fell, the policy had no order
                   that took them)
    the army mix:  from MIX_FROM, with 5 army units or more, the unit its target's army has the largest share of
                   more than its own army (by MIX_SLACK at least) trained at a building that makes it (the target's
                   mix as an input and a reward, fgself-130, left the mix gap where it was: 0.62)
    an expansion:  when its target's expansion is due (a hall, or undead's haunted gold mine, in its items) and the
                   natural expansion's camp is gone (cleared, or --scenarios natural): the worker nearest builds the
                   hall where people put theirs (undead: an acolyte haunts the mine), then three workers go to mine there.
                   (No policy had ever expanded: 0% of its games, people 81% of Echo Isles'. Night elves: the agent
                   has a tree of life entangle the mine, play.BCAgent._entangle, and the workers then mine that.)

    lumber:        (--teach 3) a target's next item short of lumber (the builder's) and fewer than CHOPPERS on lumber: the race's
                   chopper (undead: a ghoul) nearest the tree nearest its hall harvests it (the policy had all but stopped
                   chopping: human 38 lumber a minute to 872 gold). With --teach 4 also in games without a builder (no
                   target) while lumber is short of the gold (lumber_short: the bank's lumber under half its gold)
    a burrow:      (--teach 4, orc) a burrow with peons inside (loaded) and no enemy unit within BURROW_SAFE of it stands
                   down: its peons back to work (audit 12: in target games 14 orders a minute were skipped for a loaded
                   peon, 1.5 in training; 16-20 peons gathered like ~7)
    militia:       (--militia-lesson 1, human) militia and no enemy unit within MILITIA_SAFE of them nor of its halls: the hall
                   nearest them ends the town bell, every militia back to work (audit 16: the policy rang the bell, then
                   could not send its militia back; idle until their 45 s ran out, in 175 of 207 human games)

    Audit 16's long games (fgself-155/156's real games against insane: most losses past 15 minutes, median 21; game
    seconds below are a real game's, times `pace`: 0.5 on the faster rules, echofast and echofast2):
    the army:      (--army-lesson 1) once enemy air units were seen (the most at once: `air`), while fewer than
                   max(AIR_LEAST, AIR_RATIO x those) of its army hit air: the race's anti-air unit (ANTI_AIR) it has
                   least of, trained at an idle finished building that makes it. Else (a game without a target mix; a
                   target's is the army mix's business) at tier 2 with ARMY_LEAST army units or more and more than
                   ONE_UNIT of them one type: the race's tier unit (TIER_UNITS) it has least of. (90% of the army past
                   15 minutes was the race's basic unit, tier-2 units almost never trained; against 4+ air units the
                   policy won 1 of 232)
    upgrades:      (--upgrade-lesson 1) from UPGRADE_FROM, the race's upgrade building (SMITH) finished and idle: the
                   next level of the weapon or armor upgrade of its army's main class (UPGRADES: melee or ranged,
                   night elf's moon or wild), the lower level first, then the other class's (the policy researched 1.06
                   levels a game, the AI 3.72)
    creeping:      (--creep-lesson 1; 2: the heroes only) from CREEP_FROM, every living hero above CREEP_HP of its hit points, CREEP_ARMY
                   army units or more and no enemy army unit seen within BASE_SAFE of its halls: the heroes and the
                   army within ESCORT of them attack-move to the nearest creep camp within CREEP_REACH of its start
                   that they are CREEP_MARGIN times stronger than (Lanchester's hit points x damage a second; the
                   policy killed 0-2 creeps a game, the AI 13-20, its heroes stayed low)
    the hunt:      (--hunt-lesson 1) the enemy looks beaten and its last base is out of sight: no enemy building in view,
                   no enemy army seen for HUNT_QUIET, HUNT_CLEARED or more of the buildings it saw gone, HUNT_ARMY army units
                   or more: the army nearest the target (half, HUNT_ARMY at least) attack-moves to the nearest building it
                   remembers, else to the nearest gold mine it has not had in sight for HUNT_STALE (a mine's place is on
                   the map: the only thing it uses that it did not see). scripts/hunt_probe.py has what it was measured on
    hero spells:   (--spell-lesson X) the heroes' spells the policy all but never casts (SPELL_LESSONS), when a person
                   would: learned, cooled down and paid for in mana, and an area spell's SPELL_CLUMP units within its area
                   of one point in reach (the clump's centre; a unit there where the vocabulary has the spell on a unit
                   only, as the built-in AI casts blizzard), a nova's SPELL_CLUMP within its area of the hero, immolation on
                   with 2 units within IMMOLATION_NEAR and off with none within IMMOLATION_CLEAR (its buff from the
                   obs-full block: `buff_on`), a hex, sleep, mana burn (on mana), entangling roots... on the nearest enemy
                   hero in reach. The enemy's units and creeps the side sees, not buildings, not spell immune. Each hero's
                   spell looked at every SPELL_GAP steps at most, a lesson given with chance X then. (2026-11-11: the masks
                   allowed these orders in every such state of people's games and of fgself-201's own; the policy gave
                   them ~1e-4-1e-3 of its probability there, death coil or holy light ~0.2-0.5)
    The policy's masks decide (`can`: the tech, the money; a lesson they drop teaches nothing), each lesson looked at
    every LOOK steps at most.

Each kind at most every GAP steps a side, so a long wait for the money is not a label a step."""

from __future__ import annotations

import random
from collections import Counter
from operator import itemgetter

import numpy as np

from ..protocol import Build, Buy, ImmediateOrder, PointOrder, TargetOrder
from . import features as fx
from . import rulesnative
from .neutral import fourcc

# the tavern heroes people pick for each race (1.29 ladder habits)
PICKS = {"human": ("Npbm", "Nfir", "Ntin"), "orc": ("Ntin", "Npbm", "Nalc"), "undead": ("Nplh", "Nbrn", "Nalc"),
         "nightelf": ("Npbm", "Nngs", "Nbst")}
WORKERS = {fourcc(c) for c in ("hpea", "opeo", "uaco", "ewsp")}
ALTARS = {fourcc(c) for c in ("halt", "oalt", "uaod", "eate")}
ALTAR = {"human": "halt", "orc": "oalt", "undead": "uaod", "nightelf": "eate"}
# the first heroes people train at the altar (1.29 ladder habits)
FIRST_HEROES = {"human": ("Hamg", "Hmkg", "Hpal"), "orc": ("Obla", "Ofar", "Otch"), "undead": ("Udea", "Udre", "Ulic"),
                "nightelf": ("Edem", "Ekee", "Emoo")}
FIRST_HERO_FROM = 45.0
TIER2 = {fourcc(c) for c in ("hkee", "hcas", "ostr", "ofrt", "unp1", "unp2", "etoa", "etoe")}
TIER3 = {fourcc(c) for c in ("hcas", "ofrt", "unp2", "etoe")}
TAVERN_FROM = 135.0  # (a game without a target: its tavern hero, the second, from then)
TAVERN, MERCHANT, PORTAL = fourcc("ntav"), fourcc("ngme"), fourcc("stwp")
GAP = 20  # steps between two lessons of a kind
HALL = {"human": "htow", "orc": "ogre", "undead": "unpl", "nightelf": "etol"}
HALLS_ALL = ("htow", "hkee", "hcas", "ogre", "ostr", "ofrt", "unpl", "unp1", "unp2", "etol", "etoa", "etoe")
HAUNTED, ENTANGLED = fourcc("ugol"), fourcc("egol")
HALL_CODES, OWN_MINE_CODES = fx.codes(fourcc(h) for h in HALLS_ALL), fx.codes((HAUNTED, ENTANGLED))
MINES = {fourcc(c) for c in ("ngol", "ugol", "egol")}
# (the sets above as features.isin's sorted arrays)
WORKER_CODES, ALTAR_CODES, MINE_CODES = fx.codes(WORKERS), fx.codes(ALTARS), fx.codes(MINES)
# where people built their halls at Echo Isles' natural expansions (95 games of 1.29: the mine -> the hall's offset)
EXPAND_SPOTS = {(-4864, -3200): (100, 800), (4352, -3200): (-300, 700)}
NATURAL_FROM_START = 1500.0  # (the main mine is nearer its start)
LOW_EVERY = 20  # (--expand-lesson: the main mine looked at every 20 steps, 10 s)
CAMP_NEAR = 900.0  # creeps this near the natural mine guard it
HARVEST = 852018
REPAIR = 852024  # (a human builder's current order while it constructs, play.constructing)
HIDDEN = 16  # (UnitFlags.HIDDEN: a worker inside a gold mine)
MIX_FROM, MIX_SLACK, MIX_LEAST = 120.0, 0.15, 5
# the lumber lesson's choppers (undead's: ghouls) and how many at most: people put ~5 workers on lumber (2-3 ghouls),
# the policy almost none (human 38 lumber a minute to 872 gold in fgself-145; fgself-120: 105), and a target's items
# waited for lumber more than for anything else (a keep 516k blocked steps, halls of the dead 1.1M; audit 9)
CHOPPERS = {"human": ("hpea", 5), "orc": ("opeo", 5), "undead": ("ugho", 3), "nightelf": ("ewsp", 5)}
SATURATED = 6  # (--saturation-lesson: a worked mine's gold workers past this (5 mine, one walking) are surplus)
LUMBER_RATIO = 2  # (--teach 4, no target: lumber is short while the bank's lumber times this is under its gold)
# --teach 4 (orc): a burrow's stand-down (its "standdown" order: the burrow's in the demonstrations) when no enemy unit
# is within BURROW_SAFE of it and peons are inside (LOADED)
BURROW, PEON, STANDDOWN, LOADED = fourcc("otrb"), fourcc("opeo"), 852113, 32
BURROW_SAFE = 1500.0
MILITIA_SAFE = fx.MILITIA_NEAR  # (--militia-lesson: no enemy unit this near the militia or a hall)
NEAR_MERCHANT = 2500.0
ENEMY_NEAR = 800.0  # (an item lesson waits while an enemy unit is this near the hero: a fight)
USESLOT = 852008
ATTACK = 851983  # (attack-move)
LOOK = 4  # (audit 16's lessons: looked at every 4 steps at most, 2 s; a lesson given waits GAP)
# --army-lesson: air units (the enemy's seen), each race's anti-air units and tier units (a target-less game's army of
# one type at tier 2), the tech tree's makers train them; the masks decide which it can now
AIR = {fourcc(c) for c in ("hgry", "hdhw", "hgyr", "owyv", "otbr", "ugar", "ufro", "ubsp", "ehip", "ehpr", "echm", "edtm",
                           "efdr")}
AIR_CODES = fx.codes(AIR)
ANTI_AIR = {"human": ("hrif", "hdhw", "hgry"), "orc": ("ohun", "otbk", "owyv", "otbr"), "undead": ("ucry", "ugar"),
            "nightelf": ("earc", "ehip", "edry")}
TIER_UNITS = {"human": ("hkni", "hsor", "hmpr"), "orc": ("orai", "oshm", "otau"), "undead": ("ucry", "uabo", "unec"),
              "nightelf": ("edoc", "emtg", "edry")}
AIR_LEAST, AIR_RATIO = 4, 1.5  # (anti-air wanted: at least 4, 1.5 a seen air unit)
ARMY_LEAST, ONE_UNIT = 6, 0.8
# --upgrade-lesson: the upgrade building, (weapon, armor) of the first class and of the second, the second's units
SMITH = {"human": "hbla", "orc": "ofor", "undead": "ugrv", "nightelf": "edob"}
UPGRADES = {"human": (("Rhme", "Rhar"), ("Rhra", "Rhla")), "orc": (("Rome", "Roar"), ("Rora", "Roar")),
            "undead": (("Rume", "Ruar"), ("Rura", "Rucr")), "nightelf": (("Resm", "Rema"), ("Resw", "Rerh"))}
SECOND = {r: fx.codes(fourcc(c) for c in us) for r, us in (
    ("human", ("hrif", "hmtm", "hgyr")), ("orc", ("ohun", "otbk", "ocat", "owyv")), ("undead", ("ucry", "ugar", "ufro")),
    ("nightelf", ("edoc", "emtg", "ehip", "echm", "efdr", "edot", "edcm", "edtm")))}  # (ranged; night elf's: wild)
SMITH_CODES = {r: fx.codes([fourcc(c)]) for r, c in SMITH.items()}
UPGRADE_FROM = 480.0
# --creep-lesson
CREEP_FROM, CREEP_HP, CREEP_ARMY = 210.0, 0.6, 6
BASE_SAFE, ESCORT, CREEP_REACH, CREEP_MARGIN = 2500.0, 1200.0, 3600.0, 2.0
CAMP_JOIN = 550.0  # (creeps this near each other: one camp, heuristic.CAMP_JOIN)
# --hunt-lesson
HUNT_QUIET, HUNT_CLEARED, HUNT_ARMY, HUNT_STALE = 120.0, 3, 6, 240.0  # (game seconds: walking is not sped up on faster rules)
HUNT_GAP, HUNT_SAME_MINE, HUNT_FOUND_NEAR = 60, 400.0, 1500.0  # (GAP's steps: one attack-move lasts a long walk)
# --spell-lesson: the spells taught, by how: "area" at a clump of enemy units (a point spell: blizzard, flame strike, the
# lines of shockwave, impale, carrion swarm and breath of fire), "nova" around the hero, "toggle" (immolation), "hero" on
# the nearest enemy hero in reach (mana burn's with mana to burn), as people cast those. Not the spells the policy casts
# (death coil, frost nova, storm bolt, holy light, healing wave: ~0.1-0.5 of its probability in a fight), nor summons,
# auras, buffs on its own units or the utility ones
SPELL_LESSONS = {
    "AHbz": "area", "AHfs": "area", "AOsh": "area", "AUim": "area", "AUcs": "area", "AUdd": "area", "ANbf": "area",
    "ANcs": "area", "ANrf": "area", "ANvc": "area", "ANst": "area",
    "AHtc": "nova", "AOws": "nova", "AEfk": "nova", "ANht": "nova", "AEsf": "nova", "AUls": "nova", "AOww": "nova",
    "AEim": "toggle",
    "AEmb": "hero", "AEer": "hero", "AUsl": "hero", "AOhx": "hero", "AHbn": "hero", "ANdh": "hero", "AEsh": "hero",
    "ANso": "hero", "ANdo": "hero",
}
MANA_BURN, MANA_BURN_LEAST = "AEmb", 50  # (mana burn: an enemy hero with this much mana)
HERO_REACH, FIGHT_NEAR, FIGHT_UNITS = 100.0, 900.0, 3  # (a spell on a hero: in its range and a step, in a fight: this
# many enemy units, the hero too, this near the caster; a hero in sight and a minute's walk away is no fight)
TOGGLES = {"AEim": ("BEim", "immolation", "unimmolation")}  # (its buff, its on and off orders)
SPELL_GAP = 20  # (steps between two looks at one hero's spell: 10 s, a lesson given or not)
SPELL_GAPS = {"AOhx": 60}  # (hex: 30 s. At these gaps and X 1 the lessons came about as often as people cast, in 30 of
# their Echo Isles games (lessons / casts a side-game): blizzard 0.25 / 0.19, flame strike 0.28 / 0.30, war stomp 0.33 /
# 0.55, thunder clap 0.12 / 0.12, mana burn 0.22 / 0.62, sleep 0.15 / 0.11, entangling roots 0.15 / 0.22, banish 0.25 /
# 0.16; hex 1.5 / 1.1, at 10 s 2.9; their tavern heroes have no ability levels there: breath of fire unmeasured)
SPELL_CLUMP = 3  # (enemy units in an area spell's area, or around a nova's hero)
SPELL_AREA = 200.0  # (an area spell's clump within at least this of its point: the lines' widths are 100-125)
SPELL_REACH = 300.0  # (a clump this far past the spell's range: the hero walks a step and casts)
IMMOLATION_NEAR, IMMOLATION_CLEAR = 300.0, 900.0  # (on: 2 enemy units this near; off: none this near)


class Teacher:
    """One side's lessons in one game (commands: what it would order now)."""

    def __init__(self, side: int, race: str, rng: random.Random, target: dict | None = None, target_hero: bool = False,
                 lumber: bool = False, burrows: bool = False, main_low: float = 0.0, mine_full: float = 12500.0,
                 saturation: bool = False, revive: bool = False, militia: bool = False, army: bool = False,
                 upgrades: bool = False, creep: int = 0, pace: float = 1.0, hunt: int = 0, spells: float = 0.0):
        """`target` (the side's target strategy, if it has one): its tavern hero is the one taught, and none when it
        has none (another hero would break a narrow target). `target_hero` (--teach 2): its first hero from the altar
        taught too, when that is its first hero (the builder taught it only once due: the policy had made another one
        ~10-20 s before, and the second needs a tier; 25-37% of the first heroes were the target's, chance among 3-4)."""
        from .neutral import TAVERN_HEROES
        self.side = side
        self.lumber = lumber  # (--teach 3: the lumber lesson)
        self.burrows = burrows and race == "orc"  # (--teach 4: the burrows' stand-down)
        self.saturation = saturation and race != "undead"  # (--saturation-lesson: undead's acolytes cannot chop, and
        # --econ-masks caps them)
        self.revive = revive  # (--revive-lesson: a dead hero's revival at the side's altar)
        self.militia = militia and race == "human"  # (--militia-lesson: the town bell's end once no enemy is near)
        # audit 16's lessons (--army-lesson, --upgrade-lesson, --creep-lesson), their game seconds times `pace`
        self.army_lesson, self.upgrade_lesson, self.creep_lesson = (army and race in ANTI_AIR, upgrades and race in SMITH,
                                                                    creep)
        self.pace, self.air = pace, 0  # (air: the most enemy air units seen at once)
        # --hunt-lesson: its own record of what it saw (no hidden information; the network has only its memory)
        self.hunt = hunt
        self.hunt_known: dict = {}  # (an enemy building it saw and has not seen gone: id -> x, y)
        self.hunt_gone = 0  # (how many of them it saw gone)
        self.hunt_contact = 0.0  # (when it last saw an enemy army unit)
        self.hunt_mines: dict | None = None  # (the map's gold mines' places, from the game's first look -> last in sight)
        self.hunt_points: list = []  # (the points taught)
        self.hunt_lessons, self.hunt_found_at = 0, None  # (the episode row: the lessons; a base in view near a taught
        # point, when first)
        # --spell-lesson: the chance a due lesson is given (0: none); play_one sets, each step, `order_ok` ((order id,
        # kind) -> the policy could give it now: in the vocabulary and its mask) and `buff_on` ((unit, buff) -> on,
        # None: unknown) for it
        self.spells = float(spells or 0.0)
        self.order_ok = self.buff_on = None
        self.busy: set = set()  # (the side's buildings making something, and its research levels: View.busy and
        self.levels: dict = {}  # View.levels, play_one's of the step; read only)
        # the tavern hero taught, and the heroes the side has when it is hired: a target's, at its place among the
        # target's heroes (its next hero), as soon as the altar, the tier and the money permit. (As the first or
        # second hero only, from 135 s: the lesson had fired 68 times in 3,000 target games, ~1,300 of them with a
        # tavern hero; by 135 s a first hero was the altar's, and a third was never taught: audit 7's sequel)
        self.pick_at, self.pick_from = 1, TAVERN_FROM
        if target is None:
            self.pick = fourcc(rng.choice(PICKS.get(race, PICKS["human"])))
        else:
            from .techtree import is_hero
            heroes = [c for _, c in sorted((t, c) for c, t in target.get("t", {}).items() if is_hero(c))]
            k = next((i for i, c in enumerate(heroes) if c in TAVERN_HEROES), None)
            self.pick = fourcc(heroes[k]) if k is not None else None
            self.pick_at, self.pick_from = (k if k is not None else 0), 0.0
        self.last: dict[str, int] = {}
        self.taught: dict[str, int] = {}  # each kind's lessons this game (the episode row)
        # (a game without a target: its first hero from the altar, FIRST_HEROES)
        self.first = fourcc(rng.choice(FIRST_HEROES.get(race, FIRST_HEROES["human"]))) if target is None else None
        self.first_from = FIRST_HERO_FROM  # (the target's hero: as soon as the altar stands, the mask and food permitting;
        if target is not None and target_hero:  # 45 s had held 71% of its lessons back: the altar was up at ~33 s, audit 7)
            from .techtree import is_hero
            heroes = sorted((t, c) for c, t in target.get("t", {}).items() if is_hero(c))
            if heroes and heroes[0][1] not in TAVERN_HEROES:  # (a tavern hero first: the tavern's lesson below)
                self.first, self.first_from = fourcc(heroes[0][1]), 0.0
        self.rng = rng
        self.race = race
        self.main_low, self.mine_full, self._low_t, self._low = main_low, mine_full, -LOW_EVERY, False
        due = [v for c, v in (target or {}).get("t", {}).items() if c in (HALL.get(race), "ugol")]
        self.expand_at = min(due) if due else None  # (its target's expansion: when)
        from .strategy import is_army
        army = {c: v for c, v in ((target or {}).get("n") or {}).items() if is_army(c) and v > 0}
        total = sum(army.values())
        self.mix = {c: v / total for c, v in army.items()} if total else {}  # (its target's army mix)

    def _due(self, kind: str, t: int, counted: str | None = None) -> bool:
        """Whether a lesson of `kind` may be given now (and it is counted, as `counted`: one hero's of many kinds)."""
        if t - self.last.get(kind, -GAP) < GAP:
            return False
        self.last[kind] = t
        self.taught[counted or kind] = self.taught.get(counted or kind, 0) + 1
        return True

    def commands(self, rows: np.ndarray, player, now: float, t: int, can=None, pickers=()) -> list:
        """rows: the step's units (features' columns, ROW_COLS with the heroes' items), player: its PlayerState.
        `can(code[, kind])`: whether the policy could order that unit now (its mask: the money, the tech; kind: the
        order's, immediate by default, features.POINT for a building); the army mix teaches only those (99% of its
        lessons had named a unit the policy could not order then, and taught nothing).
        `pickers`: the side's heroes with an item lying near (macros.pickable, the view's of the step)."""
        self.can = can
        if player is None or rows.shape[1] < fx.ROW_COLS:
            return []
        out = []
        # the step's facts of the side (in C, rulesnative.teach: numpy's masks over all the rows had cost ~200k
        # instructions a side-step; _facts the numpy's): its heroes (dead ones count: revived), its living heroes'
        # fields, the neutral buildings (on the game's minimap from the start), its finished buildings, its army
        mc = self.__dict__.get("_mix_codes")
        if mc is None or mc[0] is not self.mix:  # (the target mix's units with their type ids: the army's counts)
            mc = self._mix_codes = _mix_codes(self.mix)
        s = rulesnative.teach(self, rows, self.side, TAVERN, MERCHANT, _CLS[0] if _CLS else _classes(), mc[2])
        if s is None:
            s = _facts(rows, self.side, mc[3])
        n_heroes, n_live, tav, mer, n_done, n_army, done_or, n_altars, altar, near_w, _, n_mix = s[:TH]
        a = TH + n_heroes
        hero_types, live = s[TH:a], s[a:a + HF * n_live]
        a += HF * n_live
        done, counts = s[a:a + 2 * n_done], s[a + 2 * n_done:a + 2 * n_done + n_mix]
        if self.first is not None and n_heroes == 0 and now >= self.first_from:  # (the altar itself from 45 s: a target's
            out += self._first_hero(rows, n_altars, altar, player, t, build=now >= FIRST_HERO_FROM)  # builder's when due)
        if self.revive and altar >= 0 and n_heroes > n_live:  # (--revive-lesson: a hero of the side's dead)
            out += self._revive(rows, altar, t)
        # a tavern hero: when the side has as many heroes as come before it (the target's, or one) or more (an altar's
        # made first: 269 of 289 human games with a tavern first hero; audit 9), with an altar, the
        # tier the hero count needs (the second hero tier 2, the third tier 3) and the money (the mask's, else 425/135;
        # the first hero is free: food only)
        if (self.pick is not None and tav >= 0 and now >= self.pick_from and self.pick_at <= n_heroes < 3
                and (n_heroes == 0 or (can(fx.rawcode(self.pick), fx.UNIT) if can is not None
                                       else player.gold >= 425 and player.lumber >= 135))
                and player.food_used + 5 <= min(player.food_cap, fx.FOOD_MAX) and done_or & K_ALTAR
                and (n_heroes == 0 or done_or & K_TIER2) and (n_heroes < 2 or done_or & K_TIER3)
                and self.pick not in hero_types):
            if near_w >= 0 and self._due("tavern hero", t):  # (the worker nearest the tavern: it walks there, Buy)
                out.append(Buy(int(rows[near_w, fx.C_ID]), int(rows[tav, fx.C_ID]), self.pick))
        pickers = set(pickers) if pickers else ()
        enemies = None  # (only for a hero with an item near)
        for j in range(0, len(live), HF):
            i, hid, hp, maxhp = live[j:j + 4]
            items = live[j + 4:j + HF]
            # an item lying near, picked up when no enemy is near
            if hid in pickers and fx.MACROS >= 2:
                h = rows[i]
                if enemies is None:  # (those it sees, as the other lessons': an unseen one near had held the label back)
                    enemies = rows[(rows[:, fx.C_OWNER] == 1 - self.side)
                                   & (rows[:, fx.C_FLAGS] & (fx.DEAD | fx.STRUCTURE) == 0)
                                   & ((rows[:, fx.C_VIS] >> self.side) & 1 == 1)]
                if not (np.hypot(enemies[:, fx.C_X] - h[fx.C_X], enemies[:, fx.C_Y] - h[fx.C_Y]) < ENEMY_NEAR).any() \
                        and self._due(f"item {hid}", t, "item"):
                    out.append(ImmediateOrder(hid, fx._mod("macros").PICKUP))
            # a potion when it is low
            if hp < 0.4 * max(maxhp, 1):
                table = fx._mod("herodata").item_table()
                heal = [k for k, it in enumerate(items) if it and table.get(it, (0, 0, 0, 0))[1]]
                if heal and self._due(f"potion {hid}", t, "potion"):
                    out.append(ImmediateOrder(hid, USESLOT + heal[0]))
            # a town portal from the goblin merchant
            if (mer >= 0 and now >= 440.0 and PORTAL not in items and 0 in items and player.gold >= 500):
                h, m = rows[i], rows[mer]
                if np.hypot(h[fx.C_X] - m[fx.C_X], h[fx.C_Y] - m[fx.C_Y]) < NEAR_MERCHANT and self._due("town portal", t):
                    out.append(Buy(hid, int(m[fx.C_ID]), PORTAL))
        if (self.expand_at is not None and now >= self.expand_at) or self._main_low(rows, player, t):
            out += self._expansion(rows, player, t)
        if self.mix and now >= MIX_FROM:
            out += self._mix(n_army, zip(mc[1], counts), done, t)
        eco = getattr(self, "economy", None)  # (short, the choppers' ids, the trees: play_one's, from the builder)
        if (self.lumber and eco is not None and eco[2] and t - self.last.get("lumber", -GAP) >= GAP
                and (eco[0] or (self.saturation and self._surplus(rows, eco[1])))):
            out += self._lumber(fx.own_alive(rows, self.side), eco[1], eco[2], t)
        if self.burrows and t - self.last.get("burrow", -GAP) >= GAP:
            out += self._burrows(rows, t)
        if self.militia and t - self.last.get("militia", -GAP) >= GAP:
            out += self._militia(rows, t)
        if self.army_lesson and t - self.last.get("army", -GAP) >= GAP and self._look("army", t):
            out += self._army(rows, done_or, n_army, t)
        if (self.upgrade_lesson and now >= UPGRADE_FROM * self.pace and t - self.last.get("upgrade", -GAP) >= GAP
                and self._look("upgrade", t)):
            out += self._upgrade(rows, t)
        if (self.creep_lesson and n_live and n_army >= CREEP_ARMY and now >= CREEP_FROM * self.pace
                and t - self.last.get("creep", -GAP) >= GAP and self._look("creep", t)):
            out += self._creep(rows, player, live, t)
        if self.hunt and self._look("hunt", t):
            out += self._hunt(rows, now, t)
        if self.spells > 0 and n_live and self.order_ok is not None:
            out += self._spells(rows, live, t)
        return out

    def _spells(self, rows: np.ndarray, live: list, t: int) -> list:
        """--spell-lesson: each living hero's SPELL_LESSONS spell, when a person would cast it (the module's docstring), in
        the form the vocabulary has and the policy's mask allows now (`order_ok`)."""
        table, info, ids = _spell_tables()
        out, en = [], None
        for j in range(0, len(live), HF):
            i, hid = live[j], live[j + 1]
            h = rows[i]
            for k, ab in enumerate(table.get(int(h[fx.C_TYPE]), ())):
                how = SPELL_LESSONS.get(ab.code) if ab is not None else None
                if how is None:
                    continue
                key = f"spell {hid} {ab.code}"
                lvl, gap = int(h[fx.C_ABIL0 + 2 * k]), SPELL_GAPS.get(ab.code, SPELL_GAP)
                if lvl <= 0 or 0 <= t - self.last.get(key, -gap) < gap:
                    continue
                ready = (int(h[fx.C_ABIL0 + 2 * k + 1]) == 0
                         and h[fx.C_MANA] >= (ab.mana[min(lvl, len(ab.mana)) - 1] if ab.mana else 0.0))
                if not ready and how != "toggle":
                    continue
                if en is None:  # (the enemy's units and the creeps the side sees: not buildings, not spell immune; once
                    side, own = self.side, rows[:, fx.C_OWNER]  # a step)
                    m = (((own == 1 - side) | (own == 12)) & (rows[:, fx.C_FLAGS] & (fx.DEAD | fx.STRUCTURE) == 0)
                         & ((rows[:, fx.C_VIS] >> side) & 1 == 1) & ~fx.isin(rows[:, fx.C_TYPE], _IMMUNE_CODES))
                    en = rows[m][:, [fx.C_ID, fx.C_X, fx.C_Y, fx.C_FLAGS, fx.C_MANA]].astype(np.float64)
                d = np.hypot(en[:, 1] - float(h[fx.C_X]), en[:, 2] - float(h[fx.C_Y]))
                a = info.get(ab.code)
                area = a.at(a.area, lvl) if a is not None else SPELL_AREA
                cmd = None
                if how == "toggle":
                    buff, on, off = TOGGLES[ab.code]
                    state = self.buff_on(hid, fourcc(buff)) if self.buff_on is not None else None
                    if state is None:
                        continue
                    on, off = ids[on], ids[off]
                    if not state and ready and int((d < IMMOLATION_NEAR).sum()) >= 2 and self.order_ok(on, fx.IMMEDIATE):
                        cmd = ImmediateOrder(hid, on)
                    elif state and not (d < IMMOLATION_CLEAR).any() and self.order_ok(off, fx.IMMEDIATE):
                        cmd = ImmediateOrder(hid, off)
                elif how == "nova":
                    oid = ids.get(ab.order)
                    if oid is not None and int((d < area).sum()) >= SPELL_CLUMP and self.order_ok(oid, fx.IMMEDIATE):
                        cmd = ImmediateOrder(hid, oid)
                elif how == "hero":  # (the nearest enemy hero in reach, in a fight; mana burn's with mana to burn)
                    oid = ids.get(ab.order)
                    rng = a.at(a.range, lvl) if a is not None else 600.0
                    if int((d < FIGHT_NEAR).sum()) < FIGHT_UNITS:
                        continue
                    ok = ((en[:, 3].astype(np.int64) & fx.HERO) > 0) & (d < rng + HERO_REACH)
                    if ab.code == MANA_BURN:
                        ok &= en[:, 4] >= MANA_BURN_LEAST
                    if oid is None or not ok.any() or int(h[fx.C_ORDER]) == oid or not self.order_ok(oid, fx.UNIT):
                        continue
                    k_ = np.flatnonzero(ok)[int(np.argmin(d[ok]))]
                    cmd = TargetOrder(hid, oid, int(en[k_, 0]))
                else:  # (area: the clump with the most enemy units within the area of one of them, in reach)
                    oid = ids.get(ab.order)
                    rng = a.at(a.range, lvl) if a is not None else 800.0
                    c = en[d < rng + SPELL_REACH]
                    if oid is None or len(c) < SPELL_CLUMP or int(h[fx.C_ORDER]) == oid:  # (channelling it already)
                        continue
                    r = max(area, SPELL_AREA)
                    near = np.hypot(c[:, None, 1] - c[None, :, 1], c[:, None, 2] - c[None, :, 2]) < r
                    b = int(near.sum(1).argmax())
                    if int(near[b].sum()) < SPELL_CLUMP:
                        continue
                    g = c[near[b]]
                    cx, cy = float(g[:, 1].mean()), float(g[:, 2].mean())
                    if self.order_ok(oid, fx.POINT):
                        cmd = PointOrder(hid, oid, int(round(cx)), int(round(cy)))
                    elif self.order_ok(oid, fx.UNIT):  # (the unit nearest the clump's centre)
                        cmd = TargetOrder(hid, oid, int(g[int(np.argmin(np.hypot(g[:, 1] - cx, g[:, 2] - cy))), 0]))
                if cmd is None:
                    continue
                self.last[key] = t  # (looked at: the next look SPELL_GAP on, the lesson given or not)
                if self.rng.random() < self.spells:
                    self.taught[f"spell {ab.code}"] = self.taught.get(f"spell {ab.code}", 0) + 1
                    out.append(cmd)
        return out

    def _look(self, kind: str, t: int) -> bool:
        """Whether audit 16's lesson `kind` is looked at now (every LOOK steps at most)."""
        k = "look " + kind
        if 0 <= t - self.last.get(k, -LOOK) < LOOK:
            return False
        self.last[k] = t
        return True

    def _idle(self, alive: np.ndarray, makers: np.ndarray) -> int | None:
        """The id of the side's first finished building of the types `makers` (sorted type ids) making nothing (`busy`)."""
        m = fx.isin(alive[:, fx.C_TYPE], makers)
        if not m.any():
            return None
        b = alive[m]
        return next((i for i, f in zip(b[:, fx.C_ID].tolist(), b[:, fx.C_FLAGS].tolist())
                     if f & fx.STRUCTURE and not f & fx.CONSTRUCTING and i not in self.busy), None)

    def _army(self, rows: np.ndarray, done_or: int, n_army: int, t: int) -> list:
        """--army-lesson: anti-air once enemy air units were seen and too few of its army hit air; else (no target mix)
        a tier unit while its army at tier 2 is one type's (more than ONE_UNIT of ARMY_LEAST or more)."""
        side = self.side
        air = fx.isin(rows[:, fx.C_TYPE], AIR_CODES)
        if air.any():
            a = rows[air]
            self.air = max(self.air, int(((a[:, fx.C_OWNER] == 1 - side) & (a[:, fx.C_FLAGS] & fx.DEAD == 0)
                                          & ((a[:, fx.C_VIS] >> side) & 1 > 0)).sum()))
        can = getattr(self, "can", None)
        anti = [u for u in ANTI_AIR[self.race] if can is None or can(u)] if self.air else []
        tier = ([u for u in TIER_UNITS[self.race] if can is None or can(u)]
                if not self.mix and done_or & K_TIER2 and n_army >= ARMY_LEAST else [])
        if not anti and not tier:  # (nothing it could train now: no rows looked at)
            return []
        alive = fx.own_alive(rows, side)
        counts = Counter(_army_of(alive)[:, fx.C_TYPE].tolist())
        if anti:
            hits = _hits_air(self.race)
            if sum(k for c, k in counts.items() if c in hits) < max(AIR_LEAST, AIR_RATIO * self.air):
                out = self._train(alive, anti, counts, t, "anti-air")
                if out:
                    return out
        n = sum(counts.values())
        if tier and n >= ARMY_LEAST:
            top, k = counts.most_common(1)[0]
            if k > ONE_UNIT * n:
                return self._train(alive, [u for u in tier if fourcc(u) != top], counts, t, "tier unit")
        return []

    def _train(self, alive: np.ndarray, units, counts: Counter, t: int, counted: str) -> list:
        """Of `units` (those the policy could train now), the one its army has least of (the first of equals), at an
        idle finished building that makes it."""
        best = None
        for u in units:
            k = counts.get(fourcc(u), 0)
            if best is not None and k >= best[0]:
                continue
            at = self._idle(alive, _maker_codes(u))
            if at is not None:
                best = (k, u, at)
        if best is None or not self._due("army", t, counted):
            return []
        return [ImmediateOrder(best[2], fourcc(best[1]))]

    def _upgrade(self, rows: np.ndarray, t: int) -> list:
        """--upgrade-lesson: at the race's idle upgrade building, the next level of its army's main class's weapon or
        armor upgrade (the lower level first, the weapon of equals), else the other class's, that the policy could
        research now (`can`: the tech mask's next level and its cost, costs.research_steps; without a mask: below 3)."""
        can, lv = getattr(self, "can", None), self.levels
        ok = [[c for c in pair if (can(c) if can is not None else lv.get(fourcc(c), 0) < 3)]
              for pair in UPGRADES[self.race]]
        if not ok[0] and not ok[1]:
            return []
        alive = fx.own_alive(rows, self.side)
        at = self._idle(alive, SMITH_CODES[self.race])
        if at is None:
            return []
        types = _army_of(alive)[:, fx.C_TYPE]
        if 2 * int(fx.isin(types, SECOND[self.race]).sum()) > len(types):
            ok = ok[::-1]
        c = min(ok[0] or ok[1], key=lambda c: lv.get(fourcc(c), 0))
        return [ImmediateOrder(at, fourcc(c))] if self._due("upgrade", t) else []

    def _creep(self, rows: np.ndarray, player, live: list, t: int) -> list:
        """--creep-lesson: healthy heroes, an army and no enemy army near its halls: the heroes and the army near them
        attack-move to the nearest camp near its start they are CREEP_MARGIN times stronger than."""
        hs = [(live[j], live[j + 2], live[j + 3]) for j in range(0, len(live), HF)]
        if not hs or any(hp <= CREEP_HP * max(mx, 1) for _, hp, mx in hs):
            return []
        side = self.side
        alive = fx.own_alive(rows, side)
        army = _army_of(alive)
        if len(army) < CREEP_ARMY:
            return []
        fl = rows[:, fx.C_FLAGS]
        halls = alive[fx.isin(alive[:, fx.C_TYPE], HALL_CODES) & (alive[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0)]
        enemy = rows[(rows[:, fx.C_OWNER] == 1 - side) & (fl & (fx.DEAD | fx.STRUCTURE | fx.WORKER) == 0)
                     & ((rows[:, fx.C_VIS] >> side) & 1 > 0)]
        if len(enemy) and len(halls) and (np.hypot(enemy[:, None, fx.C_X] - halls[None, :, fx.C_X],
                                                   enemy[:, None, fx.C_Y] - halls[None, :, fx.C_Y]) < BASE_SAFE).any():
            return []
        sx, sy = getattr(player, "start_x", None), getattr(player, "start_y", None)
        if sx is None:
            if not len(halls):
                return []
            sx, sy = halls[0, fx.C_X], halls[0, fx.C_Y]
        cx, cy, power = self._camps(rows[(rows[:, fx.C_OWNER] == 12) & (fl & (fx.DEAD | fx.STRUCTURE) == 0)])
        if not len(cx):
            return []
        heroes = rows[[int(i) for i, _, _ in hs]]
        hx, hy = float(heroes[:, fx.C_X].mean()), float(heroes[:, fx.C_Y].mean())
        group = np.concatenate([heroes, army[np.hypot(army[:, fx.C_X] - hx, army[:, fx.C_Y] - hy) < ESCORT]])
        ok = (np.hypot(cx - sx, cy - sy) <= CREEP_REACH) & (CREEP_MARGIN * power <= _power(group))
        if not ok.any() or not self._due("creep", t):
            return []
        k = int(np.flatnonzero(ok)[np.argmin(np.hypot(cx[ok] - hx, cy[ok] - hy))])
        ids = (heroes if self.creep_lesson >= 2 else group)[:, fx.C_ID].tolist()  # (--creep-lesson 2: the heroes only)
        return [PointOrder(int(u), ATTACK, float(cx[k]), float(cy[k])) for u in ids]

    def _hunt(self, rows: np.ndarray, now: float, t: int) -> list:
        """--hunt-lesson: its record of the enemy brought up to date from what it sees (the rows' visibility bits only);
        then the lesson, if the enemy looks beaten and no enemy building is in view (the module's docstring)."""
        side, fl = self.side, rows[:, fx.C_FLAGS]
        seen = rows[(rows[:, fx.C_OWNER] == 1 - side) & ((rows[:, fx.C_VIS] >> side) & 1 == 1)]
        sf = seen[:, fx.C_FLAGS]
        now_b = seen[(sf & fx.STRUCTURE > 0) & (sf & fx.DEAD == 0)]
        if ((sf & (fx.STRUCTURE | fx.WORKER | fx.DEAD)) == 0).any():
            self.hunt_contact = now
        for i in seen[(sf & fx.STRUCTURE > 0) & (sf & fx.DEAD > 0), fx.C_ID].tolist():
            if self.hunt_known.pop(i, None) is not None:
                self.hunt_gone += 1
        in_view = set(now_b[:, fx.C_ID].tolist())
        for i, x, y in now_b[:, [fx.C_ID, fx.C_X, fx.C_Y]].tolist():
            self.hunt_known[i] = (x, y)
        own = rows[(rows[:, fx.C_OWNER] == side) & (fl & fx.DEAD == 0)]
        ox, oy = own[:, fx.C_X].astype(np.float64), own[:, fx.C_Y].astype(np.float64)

        def in_sight(x: float, y: float) -> bool:
            return bool((np.hypot(ox - x, oy - y) < fx.SIGHT).any())
        for i in [i for i, (x, y) in self.hunt_known.items() if i not in in_view and in_sight(x, y)]:
            del self.hunt_known[i]  # (its place in sight and nothing there: it died out of its view)
            self.hunt_gone += 1
        if self.hunt_mines is None:  # (the map's: their places at the game's first look, so nothing of what becomes of
            self.hunt_mines = {}     # them; an entangled or haunted mine stands on one)
            for x, y in rows[fx.isin(rows[:, fx.C_TYPE], MINE_CODES)][:, [fx.C_X, fx.C_Y]].tolist():
                if not any(np.hypot(x - a, y - b) < HUNT_SAME_MINE for a, b in self.hunt_mines):
                    self.hunt_mines[(x, y)] = -1e9
        for p in self.hunt_mines:
            if in_sight(*p):
                self.hunt_mines[p] = now
        if self.hunt_found_at is None and self.hunt_points and len(now_b):  # (the episode row: a base found by a lesson)
            base = now_b[fx.isin(now_b[:, fx.C_TYPE], HALL_CODES) | fx.isin(now_b[:, fx.C_TYPE], OWN_MINE_CODES)]
            if any(np.hypot(x - a, y - b) < HUNT_FOUND_NEAR for x, y in base[:, [fx.C_X, fx.C_Y]].tolist()
                   for a, b in self.hunt_points):
                self.hunt_found_at = round(now)
        if (in_view or self.hunt_gone < HUNT_CLEARED or now - self.hunt_contact < HUNT_QUIET
                or t - self.last.get("hunt", -HUNT_GAP) < HUNT_GAP):
            return []
        army = _army_of(fx.own_alive(rows, side))
        if len(army) < HUNT_ARMY:
            return []
        ax, ay = float(army[:, fx.C_X].mean()), float(army[:, fx.C_Y].mean())
        places = list(self.hunt_known.values()) or [p for p, at in self.hunt_mines.items() if now - at >= HUNT_STALE]
        if not places or not self._due("hunt", t):
            return []
        px, py = min(places, key=lambda p: np.hypot(p[0] - ax, p[1] - ay))
        d = np.hypot(army[:, fx.C_X] - px, army[:, fx.C_Y] - py)
        party = army[np.argsort(d, kind="stable")[:max(HUNT_ARMY, (len(army) + 1) // 2)]]
        self.hunt_lessons += 1
        self.hunt_points.append((px, py))
        return [PointOrder(int(u), ATTACK, float(px), float(py)) for u in party[:, fx.C_ID].tolist()]

    def _camps(self, creeps: np.ndarray) -> tuple:
        """The creep camps' centres and strengths (_camps, _power), kept while the creeps stand where they stood."""
        key = creeps[:, [fx.C_ID, fx.C_X, fx.C_Y, fx.C_HP]].tobytes()
        got = self.__dict__.get("_camp_table")
        if got is None or got[0] != key:
            label = _camps(creeps)
            n = int(label.max()) + 1 if len(label) else 0
            size = np.bincount(label, minlength=n).astype(np.float64)
            dps = np.asarray([_dps(c, lv) for c, lv in creeps[:, [fx.C_TYPE, fx.C_HLEVEL]].tolist()], np.float64)
            got = self._camp_table = (key, np.bincount(label, creeps[:, fx.C_X], n) / np.maximum(size, 1),
                                      np.bincount(label, creeps[:, fx.C_Y], n) / np.maximum(size, 1),
                                      np.bincount(label, np.maximum(creeps[:, fx.C_HP], 0), n)
                                      * np.bincount(label, dps, n) / 1000.0)
        return got[1:]

    def _militia(self, rows: np.ndarray, t: int) -> list:
        """Human: militia and no enemy unit within MILITIA_SAFE of any of them nor of a finished hall: the hall nearest
        the militia ends the bell (townbelloff: all of them back to work, taken while there are militia)."""
        alive = fx.own_alive(rows, self.side)
        militia = alive[alive[:, fx.C_TYPE] == fx.MILITIA_T]
        if not len(militia):
            return []
        halls = alive[np.isin(alive[:, fx.C_TYPE], fx.HUMAN_HALLS) & (alive[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0)]
        if not len(halls):
            return []
        enemies = rows[(rows[:, fx.C_OWNER] == 1 - self.side) & (rows[:, fx.C_FLAGS] & (fx.DEAD | fx.STRUCTURE) == 0)
                       & ((rows[:, fx.C_VIS] >> self.side) & 1 == 1)]  # (those it sees, as the masks; audit 17)
        ex, ey = enemies[:, fx.C_X].astype(np.float64), enemies[:, fx.C_Y].astype(np.float64)
        for u in np.concatenate([militia, halls]):
            if (np.hypot(ex - float(u[fx.C_X]), ey - float(u[fx.C_Y])) < MILITIA_SAFE).any():
                return []
        mx, my = float(militia[:, fx.C_X].mean()), float(militia[:, fx.C_Y].mean())
        k = int(np.argmin((halls[:, fx.C_X] - mx) ** 2 + (halls[:, fx.C_Y] - my) ** 2))
        if not self._due("militia", t):
            return []
        return [ImmediateOrder(int(halls[k, fx.C_ID]), fx.TOWNBELL_OFF)]

    def _burrows(self, rows: np.ndarray, t: int) -> list:
        """Orc: each finished burrow nearest a loaded peon (inside one) with no enemy unit within BURROW_SAFE of it
        stands down (its peons back to work)."""
        alive = fx.own_alive(rows, self.side)
        fl = alive[:, fx.C_FLAGS]
        loaded = alive[(alive[:, fx.C_TYPE] == PEON) & (fl & LOADED > 0)]
        burrows = alive[(alive[:, fx.C_TYPE] == BURROW) & (fl & fx.CONSTRUCTING == 0)]
        if not len(loaded) or not len(burrows):
            return []
        bx, by = burrows[:, fx.C_X].astype(np.float64), burrows[:, fx.C_Y].astype(np.float64)
        full = {int(np.argmin((bx - x) ** 2 + (by - y) ** 2)) for x, y in loaded[:, [fx.C_X, fx.C_Y]].tolist()}
        enemies = rows[(rows[:, fx.C_OWNER] == 1 - self.side) & (rows[:, fx.C_FLAGS] & (fx.DEAD | fx.STRUCTURE) == 0)
                       & ((rows[:, fx.C_VIS] >> self.side) & 1 == 1)]  # (those it sees, as the masks; audit 17)
        ex, ey = enemies[:, fx.C_X].astype(np.float64), enemies[:, fx.C_Y].astype(np.float64)
        safe = [k for k in sorted(full) if not (np.hypot(ex - bx[k], ey - by[k]) < BURROW_SAFE).any()]
        if not safe or not self._due("burrow", t):
            return []
        return [ImmediateOrder(int(burrows[k, fx.C_ID]), STANDDOWN) for k in safe]

    def _surplus(self, rows: np.ndarray, choppers: set) -> bool:
        """--saturation-lesson: more of the race's workers than its worked mines take (SATURATED a finished hall's
        mine), not chopping nor building: the lumber lesson sends one to chop (fgself-155's human games past 10 minutes:
        up to 26 peasants trained, ~510 gold a minute, one saturated mine's, and 58 lumber; the user saw them queue at
        the mine)."""
        code, _ = CHOPPERS.get(self.race, (None, 0))
        alive = fx.own_alive(rows, self.side)
        workers = alive[alive[:, fx.C_TYPE] == fourcc(code)] if code else alive[:0]
        halls = alive[fx.isin(alive[:, fx.C_TYPE], HALL_CODES) & (alive[:, fx.C_FLAGS] & fx.CONSTRUCTING == 0)]
        gold = workers[(workers[:, fx.C_ORDER] < fx.TYPE_CODE) & (workers[:, fx.C_FLAGS] & LOADED == 0)
                       & ~np.isin(workers[:, fx.C_ID], list(choppers))]  # (not in a burrow; audit 17)
        return len(gold) > SATURATED * max(len(halls), 1)

    def _lumber(self, alive: np.ndarray, choppers: set, trees: dict, t: int) -> list:
        """A target's next item short of lumber and fewer than CHOPPERS on it: the race's chopper nearest the tree
        nearest its hall (one not chopping, not hidden, not building) harvests that tree."""
        code, most = CHOPPERS.get(self.race, (None, 0))
        if code is None or len(choppers) >= most:
            return []
        fl = alive[:, fx.C_FLAGS]
        cand = alive[(alive[:, fx.C_TYPE] == fourcc(code)) & (fl & (HIDDEN | fx.STRUCTURE) == 0)
                     & _free(alive) & ~np.isin(alive[:, fx.C_ID], list(choppers))]
        halls = alive[fx.isin(alive[:, fx.C_TYPE], HALL_CODES)]
        if not len(cand) or not len(halls):
            return []
        ids = np.fromiter(trees.keys(), np.int64, len(trees))
        xy = np.asarray(list(trees.values()), np.float64).reshape(-1, 2)
        h = halls[0]
        k = int(np.argmin((xy[:, 0] - h[fx.C_X]) ** 2 + (xy[:, 1] - h[fx.C_Y]) ** 2))
        w = cand[int(np.argmin((cand[:, fx.C_X] - xy[k, 0]) ** 2 + (cand[:, fx.C_Y] - xy[k, 1]) ** 2))]
        return [TargetOrder(int(w[fx.C_ID]), HARVEST, int(ids[k]))] if self._due("lumber", t) else []

    def _first_hero(self, rows: np.ndarray, n_altars: int, altar: int, player, t: int, build: bool = True) -> list:
        """A side without a hero: its altar trains its first hero, and with no altar (nor one being built) a worker
        builds one near its hall; each when the policy could order it (`can`). (The step's facts: n_altars, the
        side's altars, altar: the row of its first finished one, or -1)"""
        can = getattr(self, "can", None)
        if altar >= 0:
            if (can is None or can(fx.rawcode(self.first))) and player.food_used + 5 <= min(player.food_cap, fx.FOOD_MAX) \
                    and self._due("first hero", t):
                return [ImmediateOrder(int(rows[altar, fx.C_ID]), self.first)]
            return []
        if n_altars or not build:
            return []  # (being built)
        alive = fx.own_alive(rows, self.side)
        code = ALTAR.get(self.race)
        workers = alive[fx.isin(alive[:, fx.C_TYPE], WORKER_CODES) & (alive[:, fx.C_FLAGS] & HIDDEN == 0)
                        & (alive[:, fx.C_FLAGS] & fx.STRUCTURE == 0) & _free(alive)]
        halls = alive[fx.isin(alive[:, fx.C_TYPE], HALL_CODES)]
        if code is None or not len(workers) or (can is not None and not can(code, fx.POINT)) or not self._due("altar", t):
            return []
        at = halls[0] if len(halls) else workers[0]
        w = workers[int(np.argmin(np.hypot(workers[:, fx.C_X] - at[fx.C_X], workers[:, fx.C_Y] - at[fx.C_Y])))]
        a, r = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(450, 800)
        return [Build(int(w[fx.C_ID]), fourcc(code), float(at[fx.C_X] + r * np.cos(a)), float(at[fx.C_Y] + r * np.sin(a)))]

    def _revive(self, rows: np.ndarray, altar: int, t: int) -> list:
        """--revive-lesson: the side's first dead hero the policy could revive now (`can`: its order's mask, the money,
        the food, REVIVE_AFTER_STEPS since its death and no revival under way) revived at its first finished altar (row
        `altar`): the hero's own order there, an altar hero's training order or a tavern hero's hire (play.BCAgent: the
        revival; heroes stayed dead in 48% of the deaths of 53 recorded training games, fgself-150..156, and the policy
        chose the revival in 2.4% of the steps it could pay for it; 34 of 111 deaths it could have paid for never)."""
        from .neutral import TAVERN_HEROES
        can = getattr(self, "can", None)
        fl = rows[:, fx.C_FLAGS]
        dead = rows[(rows[:, fx.C_OWNER] == self.side) & (fl & fx.HERO > 0) & (fl & fx.DEAD > 0), fx.C_TYPE]
        a = int(rows[altar, fx.C_ID])
        for typ in dead.tolist():
            code = fx.rawcode(int(typ))
            tavern = code in TAVERN_HEROES
            if tavern:  # (a tavern hero's revival is the policy's hire order on some unit, play.py; labelled as the
                continue  # altar's own hire it had widened the allowed table: altars hiring for good; audit 17)
            if can is not None and not can(code, fx.IMMEDIATE):
                continue
            if not self._due("revive", t):
                return []
            return [ImmediateOrder(a, int(typ))]
        return []

    def _mix(self, n: int, counts, done: list, t: int) -> list:
        """The unit its target's army has the largest share of more than its own, trained at a building that makes it.
        (The step's facts: n, the side's army units; counts, (unit, how many of its army) for the target's units;
        done, each of its finished building types' first (type, id), flat)"""
        if n < MIX_LEAST:
            return []
        # (the target's units' shares in its army: 1 / n added once a unit, as they were summed over the army)
        have = {u: _SHARES.get((c, n)) or _share(c, n) for u, c in counts if c}
        can = getattr(self, "can", None)
        # (the most lacking first, as sorted by minus the gap: the same order of equals, sorting being stable)
        for unit, gap in sorted([(u, s - have.get(u, 0.0)) for u, s in self.mix.items()], key=_SECOND, reverse=True):
            if gap < MIX_SLACK:
                return []
            if can is not None and not can(unit):
                continue  # (not now: the next most lacking that it can)
            mk = _makers(unit)
            at = next((done[j + 1] for j in range(0, len(done), 2) if done[j] in mk), None)
            if at is not None:
                if fx.MACROS and "mix" not in fx.MACROS_OFF:  # (the building's mix order: which unit is its business,
                    # macros.MIX; --macros-off mix: the unit's own train order below, the mix order masked had kept none
                    # of ~2.5 lessons a game; audit 12)
                    return [ImmediateOrder(at, fx._mod("macros").MIX)] if self._due("army mix", t) else []
                return [ImmediateOrder(at, fourcc(unit))] if self._due("army mix", t) else []
        return []

    def _natural(self, rows: np.ndarray, mines: list, sx, sy) -> int:
        """The natural's row when two gold mines are as near its start (an entangled mine stands on its mine):
        numpy's argsort's choice as _expansion_facts makes it, among `mines` (the living gold mines' row, x and y, in
        the rows' order), kept per start and their positions (argsort's order is a function of the distances)."""
        key = (sx, sy, tuple(mines[1::3]), tuple(mines[2::3]))
        got = self.__dict__.setdefault("_naturals", {})
        pos = got.get(key)
        if pos is None:
            m = rows[mines[0::3]]
            d = np.hypot(m[:, fx.C_X] - sx, m[:, fx.C_Y] - sy)
            o = np.argsort(d)
            pos = got[key] = int(o[d[o] > NATURAL_FROM_START][0])
        return mines[3 * pos]

    def _main_low(self, rows: np.ndarray, player, t: int) -> bool:
        """--expand-lesson F: whatever the target, the natural taught once the main mine (the gold mine nearest its start,
        its own haunted or entangled one too) holds less than F of a full one of the map's (audit 15: on echofast2 the
        main ran dry at 12-13 minutes, the policy expanded in 2-10% of the games, and lost 95% of those past it)."""
        if self.main_low <= 0:
            return False
        if 0 <= t - self._low_t < LOW_EVERY:
            return self._low
        self._low_t = t
        sx, sy = getattr(player, "start_x", None), getattr(player, "start_y", None)
        m = rows[fx.isin(rows[:, fx.C_TYPE], MINE_CODES)] if sx is not None else rows[:0]
        if not len(m):
            self._low = False
            return False
        d = np.hypot(m[:, fx.C_X] - sx, m[:, fx.C_Y] - sy)
        i = int(np.argmin(d))
        self._low = bool(d[i] < NATURAL_FROM_START and m[i, fx.C_RESOURCE] < self.main_low * self.mine_full)
        return self._low

    def _expansion(self, rows: np.ndarray, player, t: int) -> list:
        """The next step of taking the natural expansion: clearing its camp, the hall (undead: the haunted mine), the
        workers that mine there."""
        sx, sy = getattr(player, "start_x", None), getattr(player, "start_y", None)
        if sx is None:
            return []
        # (the natural and around it: in C, rulesnative.expansion; _expansion_facts the numpy's, also when picked
        # workers are as near: argsort's order of equals)
        cls = _CLS[0] if _CLS else _classes()
        e = rulesnative.expansion(self, rows, self.side, sx, sy, cls, NATURAL_FROM_START, 1000.0, 900.0, CAMP_NEAR)
        if e is not None and e[0] == -2 and e[1] >= 0:  # (two mines as near its start: numpy's argsort chooses)
            e = rulesnative.expansion(self, rows, self.side, sx, sy, cls, NATURAL_FROM_START, 1000.0, 900.0, CAMP_NEAR,
                                      self._natural(rows, e[EH:EH + 3 * e[1]], sx, sy))
        if e is None or e[0] == -2:
            e = _expansion_facts(rows, fx.own_alive(rows, self.side), sx, sy)
        if e[0] < 0:
            return []
        (_, nat_id, mx, my, n_workers, n_taken, all_building, n_mining, has_target, target_id, creeps, n_army, n_heroes,
         _, nearest, n_pick) = e[:16]
        mx, my = float(mx), float(my)
        hall = fourcc(HALL[self.race])
        if n_taken:  # (taken: once finished, three workers to mine there; counting those inside a mine: wisps in an
            # entangled one are hidden, and the lesson had never seen 4 there: 22 a long night elf game; audit 8)
            if all_building or n_mining >= 4 or not n_workers:
                return []
            if self.race == "nightelf" and not has_target:
                return []  # (its tree of life entangles the mine first)
            mine_id = target_id if has_target else nat_id
            pick = e[16:16 + n_pick]  # (the three nearest outside 900)
            return ([TargetOrder(w, HARVEST, mine_id) for w in pick]
                    if n_pick and self._due("expansion: workers", t) else [])
        if creeps:  # (guarded. Sending the whole army to clear it every 10 s (fgself-123) lost the games that had
            # an expansion due, 13% won to 32%. With --macros: its heroes, while the army is 6 units or more, are shown
            # the creep order (the policy decides who goes with them))
            if not fx.MACROS or n_army < 6 or not n_heroes or not self._due("expansion: its camp", t):
                return []
            if "creep" in fx.MACROS_OFF:  # (--macros-off creep: the heroes attack-move to the camp themselves, to its
                near = rows[(rows[:, fx.C_OWNER] == 12) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]  # creeps' middle)
                near = near[np.hypot(near[:, fx.C_X] - mx, near[:, fx.C_Y] - my) < CAMP_NEAR]
                cx, cy = float(near[:, fx.C_X].mean()), float(near[:, fx.C_Y].mean())
                return [PointOrder(h, ATTACK, cx, cy) for h in e[EH:EH + n_heroes]]
            creep = fx._mod("macros").CREEP
            return [ImmediateOrder(h, creep) for h in e[EH:EH + n_heroes]]
        if not n_workers:
            return []
        can = getattr(self, "can", None)
        if (can is not None and not (fx.MACROS and "expand" not in fx.MACROS_OFF)  # (the hall's own build order: only
                and not any(can(c, k) for c in (("ugol",) if self.race == "undead" else (HALL[self.race],))
                            for k in (fx.POINT, fx.UNIT))):  # when the policy could order it: at a person's income the
            return []  # mask dropped ~94% of them, the hall not affordable; audit 12)
        if not self._due("expansion: the hall", t):
            return []
        if fx.MACROS and "expand" not in fx.MACROS_OFF:  # (the agent's expand order, the worker nearest: where to build
            # is its own business, macros.py; --macros-off expand: the hall's own build order below, where people put theirs)
            return [ImmediateOrder(nearest, fx._mod("macros").EXPAND)]
        if self.race == "undead":  # (an acolyte haunts the mine: its gold, no hall needed)
            m = rows[rows[:, fx.C_ID] == nat_id]
            if not len(m) or not (int(m[0, fx.C_VIS]) >> self.side) & 1:  # (a mine it has not in sight: the label had
                return [PointOrder(nearest, fx.MOVE, mx, my)]  # no target and was dropped, 0 of 40 kept: the acolyte
            return [TargetOrder(nearest, HAUNTED, nat_id)]      # walks there first; audit 13)
        ox, oy = EXPAND_SPOTS.get((int(round(mx)), int(round(my))), (None, None))
        if ox is None:  # (another map's: 750 toward its start)
            n = max(float(np.hypot(sx - mx, sy - my)), 1.0)
            ox, oy = 750.0 * (sx - mx) / n, 750.0 * (sy - my) / n
        x, y = mx + ox, my + oy
        alive = fx.own_alive(rows, self.side)
        workers = alive[fx.isin(alive[:, fx.C_TYPE], WORKER_CODES) & (alive[:, fx.C_FLAGS] & HIDDEN == 0) & _free(alive)]
        if not len(workers):
            return []
        w = workers[int(np.argmin(np.hypot(workers[:, fx.C_X] - x, workers[:, fx.C_Y] - y)))]
        return [Build(int(w[fx.C_ID]), hall, x, y)]


def _free(alive: np.ndarray) -> np.ndarray:
    """Per row: no worker on its way to build (its order a building's type) nor at a frame (a human builder constructs
    with "repair", play.constructing): a lesson that took one had pulled it off its building."""
    o = alive[:, fx.C_ORDER]
    return (o < fx.TYPE_CODE) & (o != REPAIR)


TH, EH, HF = rulesnative.TH, rulesnative.EH, rulesnative.HF  # (the C's facts: their heads, a hero's fields)
def lumber_short(gold: int, lumber: int) -> bool:
    """--teach 4, a game without a builder: the lumber lesson's 'short' (the bank's lumber under half its gold: a
    person's gold and lumber income run ~2-3 to 1; the policy's 462 to 72 a minute in its real-game losses, audit 12)."""
    return lumber * LUMBER_RATIO < gold


_SECOND = itemgetter(1)
K_ALTAR, K_TIER2, K_TIER3 = rulesnative.K_ALTAR, rulesnative.K_TIER2, rulesnative.K_TIER3
_CLS: list = []  # ([the C's table of type classes]: made once)


def _classes() -> tuple:
    if not _CLS:
        rn, bits = rulesnative, {}
        for group, bit in ((ALTARS, rn.K_ALTAR), (TIER2, rn.K_TIER2), (TIER3, rn.K_TIER3), (WORKERS, rn.K_WORKER),
                           (MINES, rn.K_MINE), (fx._mod("macros").HOLD_CODES.tolist(), rn.K_HOLD),
                           ((HAUNTED, ENTANGLED), rn.K_OWN_MINE)):
            for v in group:
                bits[int(v)] = bits.get(int(v), 0) | bit
        _CLS[:] = [rn.table(bits)]
    return _CLS[0]


def _facts(rows: np.ndarray, side: int, mix: list) -> list:
    """rulesnative.teach's facts by numpy (the path without the library): the side's heroes, the tavern and the
    merchant (the first living of each), its finished buildings, its army and how many of each of the `mix` types,
    its altars, the worker nearest the tavern."""
    fl, typ = rows[:, fx.C_FLAGS], rows[:, fx.C_TYPE]
    own, living = rows[:, fx.C_OWNER] == side, fl & fx.DEAD == 0
    idx = np.flatnonzero(own & living)
    alive = fx.own_alive(rows, side, own & living)
    fa, ta = alive[:, fx.C_FLAGS], alive[:, fx.C_TYPE]
    heroes = typ[own & (fl & fx.HERO > 0)].tolist()
    live = []
    for i, h in zip(idx[fa & fx.HERO > 0].tolist(), alive[fa & fx.HERO > 0].tolist()):
        live += [i, h[fx.C_ID], h[fx.C_HP], h[fx.C_MAXHP], *h[fx.C_ITEM0:fx.C_ITEM0 + 6]]
    tav, mer = np.flatnonzero(living & (typ == TAVERN)), np.flatnonzero(living & (typ == MERCHANT))
    done = alive[(fa & fx.STRUCTURE > 0) & (fa & fx.CONSTRUCTING == 0)]
    dt = done[:, fx.C_TYPE]
    done_or = ((K_ALTAR if fourcc_in(dt, ALTARS) else 0) | (K_TIER2 if fourcc_in(dt, TIER2) else 0)
               | (K_TIER3 if fourcc_in(dt, TIER3) else 0))
    army = ta[fa & (fx.STRUCTURE | fx.WORKER | fx.HERO) == 0].tolist()
    altars = fx.isin(ta, ALTAR_CODES)
    fin = idx[altars & (fa & fx.CONSTRUCTING == 0)]
    near_w = -1
    if len(tav):
        tv = rows[tav[0]]
        wm = fx.isin(ta, WORKER_CODES)
        workers = alive[wm]
        if len(workers):
            d = np.hypot(workers[:, fx.C_X] - tv[fx.C_X], workers[:, fx.C_Y] - tv[fx.C_Y])
            near_w = int(idx[wm][int(np.argmin(d))])
    pairs, seen = [], set()  # (each type's first)
    for t, i in zip(dt.tolist(), done[:, fx.C_ID].tolist()):
        if t not in seen:
            seen.add(t)
            pairs += [t, i]
    return [len(heroes), len(live) // HF, int(tav[0]) if len(tav) else -1, int(mer[0]) if len(mer) else -1,
            len(pairs) // 2, len(army), done_or, int(altars.sum()), int(fin[0]) if len(fin) else -1, near_w, len(idx),
            len(mix), *heroes, *live, *pairs, *(army.count(c) for c in mix)]


def _mix_codes(mix: dict) -> tuple:
    """The target mix's units with a type id (a unit code of 4 latin-1 characters; the others are no unit's): (mix,
    those units, (their ids as an int64 array, its address, how many), the ids)."""
    units = [u for u in mix if _code4(u) is not None]
    codes = [_code4(u) for u in units]
    arr = np.ascontiguousarray(codes or [0], np.int64)
    return mix, units, (arr, arr.ctypes.data, len(codes)), codes


def _expansion_facts(rows: np.ndarray, alive: np.ndarray, sx, sy) -> list:
    """rulesnative.expansion's facts by numpy (without the library, or when picked workers are as near: argsort's
    order): the natural, around it the side's workers, holds, own mines and creeps, its army and heroes."""
    mines = rows[fx.isin(rows[:, fx.C_TYPE], MINE_CODES) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
    if not len(mines):
        return [-1]
    d = np.hypot(mines[:, fx.C_X] - sx, mines[:, fx.C_Y] - sy)
    o = np.argsort(d)
    far = o[d[o] > NATURAL_FROM_START]
    if not len(far):
        return [-1]
    nat = mines[far[0]]
    mx, my = float(nat[fx.C_X]), float(nat[fx.C_Y])
    near = lambda r, rad: np.hypot(r[:, fx.C_X] - mx, r[:, fx.C_Y] - my) < rad  # noqa: E731
    any_worker = fx.isin(alive[:, fx.C_TYPE], WORKER_CODES)
    workers = alive[any_worker & (alive[:, fx.C_FLAGS] & HIDDEN == 0)]
    mining = alive[any_worker]
    taken = alive[fx.isin(alive[:, fx.C_TYPE], fx._mod("macros").HOLD_CODES)]
    taken = taken[near(taken, 1000.0)]
    target = alive[fx.isin(alive[:, fx.C_TYPE], OWN_MINE_CODES)]
    target = target[near(target, 1000.0)]
    pick = []
    if len(taken):
        far_w = workers[~near(workers, 900.0)]
        pick = far_w[np.argsort(np.hypot(far_w[:, fx.C_X] - mx, far_w[:, fx.C_Y] - my))[:3], fx.C_ID].tolist()
    creeps = rows[(rows[:, fx.C_OWNER] == 12) & (rows[:, fx.C_FLAGS] & fx.DEAD == 0)]
    army = alive[(alive[:, fx.C_FLAGS] & (fx.STRUCTURE | fx.WORKER) == 0)]
    heroes = army[army[:, fx.C_FLAGS] & fx.HERO > 0, fx.C_ID].tolist()
    nearest = (int(workers[int(np.argmin(np.hypot(workers[:, fx.C_X] - mx, workers[:, fx.C_Y] - my))), fx.C_ID])
               if len(workers) else None)
    return [1, int(nat[fx.C_ID]), int(nat[fx.C_X]), int(nat[fx.C_Y]), len(workers), len(taken),
            int(bool((taken[:, fx.C_FLAGS] & fx.CONSTRUCTING > 0).all())) if len(taken) else 0,
            int(near(mining, 900.0).sum()), int(len(target) > 0), int(target[0, fx.C_ID]) if len(target) else 0,
            int(bool(near(creeps, CAMP_NEAR).any())), len(army), len(heroes), int(nearest is not None),
            nearest if nearest is not None else 0, len(pick), *(pick + [0] * (3 - len(pick))), *heroes]


_C4: dict = {}  # (a target's unit code -> its type id, or None: not 4 latin-1 characters, never a unit's)
_SHARES: dict = {}  # ((count, army size) -> 1 / size added count times: _mix's shares as they were summed)
_MAKERS: dict = {}  # (a unit -> the type ids of the buildings that make it: the tech tree's, fixed)


def _code4(u) -> int | None:
    c = _C4.get(u, False)
    if c is False:
        try:
            b = u.encode("latin-1")
            c = int.from_bytes(b, "big") if len(b) == 4 else None
        except (AttributeError, UnicodeEncodeError):
            c = None
        _C4[u] = c
    return c


def _share(c: int, n: int) -> float:
    v = _SHARES.get((c, n))
    if v is None:
        v, inv = 0, 1 / n
        for _ in range(c):
            v = v + inv
        _SHARES[(c, n)] = v
    return v


def _makers(unit: str) -> frozenset:
    m = _MAKERS.get(unit)
    if m is None:
        m = _MAKERS[unit] = frozenset(fourcc(x) for x in fx._mod("techtree").tree().makers.get(unit, ()))
    return m


_CODES: dict[int, tuple] = {}  # (a set of codes -> (the set, its sorted array): fourcc_in's, kept by identity)


def fourcc_in(types: np.ndarray, codes: set[int]) -> bool:
    c = _CODES.get(id(codes))
    if c is None or c[0] is not codes:
        c = _CODES[id(codes)] = (codes, fx.codes(codes))
    return bool(fx.isin(types, c[1]).any())


def pace(map_name: str) -> float:
    """Audit 16's lessons' game seconds on a map, as a real game's: half on faster rules (echofast, echofast2: a third
    of the build times, half the costs), else the same."""
    try:
        from ..data.duelmap import map_rules
        return 0.5 if map_rules(map_name).time < 1.0 else 1.0
    except Exception:  # noqa: BLE001 (another map: the real game's)
        return 1.0


def _army_of(alive: np.ndarray) -> np.ndarray:
    """The army among the side's living units: no building, worker, hero, summoned unit nor militia."""
    fl = alive[:, fx.C_FLAGS]
    return alive[(fl & (fx.STRUCTURE | fx.WORKER | fx.HERO | fx.SUMMONED) == 0) & (alive[:, fx.C_TYPE] != fx.MILITIA_T)]


_AIR_HITS: dict = {}  # (a race -> the type ids of units that hit air: the game's tables, and its ANTI_AIR)
_DPS: dict = {}  # ((type id, hero level) -> damage a second: heuristic.dps)


def _hits_air(race: str) -> frozenset:
    got = _AIR_HITS.get(race)
    if got is None:
        from ..data.objects import combat_stats
        got = _AIR_HITS[race] = frozenset({fourcc(c) for c, s in combat_stats().items() if s.hits_air and not s.is_hero
                                           and len(c) == 4} | {fourcc(c) for c in ANTI_AIR.get(race, ())})
    return got


def _dps(typ: int, level: int) -> float:
    d = _DPS.get((typ, level))
    if d is None:
        from .heuristic import dps
        d = _DPS[(typ, level)] = dps(fx.rawcode(typ), level)
    return d


def _power(units: np.ndarray) -> float:
    """Lanchester's strength of a group (heuristic.power): its hit points times its damage a second, / 1000."""
    total = sum(_dps(c, lv) for c, lv in units[:, [fx.C_TYPE, fx.C_HLEVEL]].tolist())
    return float(np.maximum(units[:, fx.C_HP], 0).sum()) * total / 1000.0


def _camps(creeps: np.ndarray) -> np.ndarray:
    """The creep camps: creeps within CAMP_JOIN of one another (single link) -> each creep's camp (0, 1, ...)."""
    xy = creeps[:, [fx.C_X, fx.C_Y]].astype(np.float64)
    near = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) < CAMP_JOIN
    label, k = np.full(len(xy), -1), 0
    for i in range(len(xy)):
        if label[i] >= 0:
            continue
        label[i], todo = k, [i]
        while todo:
            new = np.flatnonzero(near[todo.pop()] & (label < 0))
            label[new] = k
            todo += new.tolist()
        k += 1
    return label


_MAKER_CODES: dict = {}  # (a unit -> its makers' type ids, sorted: features.isin's)


def _maker_codes(code: str) -> np.ndarray:
    got = _MAKER_CODES.get(code)
    if got is None:
        got = _MAKER_CODES[code] = fx.codes(_makers(code))
    return got


_IMMUNE_CODES = fx.codes(fx.SPELL_IMMUNE)  # (--spell-lesson: no spell's target)
_SPELL_TABLES: tuple | None = None


def _spell_tables() -> tuple:
    """--spell-lesson's tables, once: the heroes' abilities per slot (herodata.hero_table), their ranges and areas
    (data.abilities.ability_info) and the game's order id of each order string (warcraftsim/data/order_ids.json)."""
    global _SPELL_TABLES
    if _SPELL_TABLES is None:
        import json
        from importlib import resources

        from ..data.abilities import ability_info
        from .herodata import hero_table
        ids = {k: int(v) for k, v in json.loads(resources.files("warcraftsim.data").joinpath("order_ids.json")
                                                .read_text()).items()}
        _SPELL_TABLES = (hero_table(), ability_info(), ids)
    return _SPELL_TABLES
