"""What a person playing a 1.29 melee game can see that the agent's observation lacks, checked against the game's own tables
(docs/reference/observation-coverage.md, the counterpart of order-coverage.md for the inputs).

    python3 scripts/observation_coverage.py [--run runs/fgself-194] [--doc docs/reference/observation-coverage.md]

Checked from the tables (Units\\AbilityData.slk, AbilityBuffData.slk, ItemData.slk, UnitAbilities.slk, the *Func.txt files;
fullvocab.tables, techtree) and the run's settings (run.json: --inputs, --features) and vocabulary:

  channels     the entity and global channels the run's network reads, and those its --features zeroes
  buffs        every buff a melee player's units, heroes, the tavern's heroes, the mercenaries, the items and the maps'
               creeps can put on a unit (AbilityData's BuffID of each of their abilities, effects left out); none is an
               input today. Also whether scripts/obs_probe.py's list (BUFFS) asks the game for each
  items        the items a hero can hold (the race shops, the goblin merchant, the creeps' drops: ItemData's pickRandom
               items of the levels the maps' drop tables ask for): how many look alike to the hero block (held, usable,
               heals, restores mana, town portal), and those with charges (not an input)
  researches   each race's researches against the global vector's upgrade levels (the vocabulary's "upgrades"); the
               enemy's researches are no input at all
  cooldowns    the race's non-hero abilities with a cooldown (only the heroes' four slots report one)
  toggles      the abilities with an autocast or on/off state (not an input; fullvocab.Checks tracks it for the mask)
  types        the unit types a player meets (its own race's, the tavern's, the mercenaries', the maps' creeps) without a
               type embedding of their own in the vocabulary

--doc writes the generated part of the doc between its markers (the rest is written by hand). The exit status is 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import fullvocab as fv  # noqa: E402

BEGIN = "<!-- observation_coverage.py: begin (generated; edit the script, not this part) -->"
END = "<!-- observation_coverage.py: end -->"
RACE_LETTER = {"human": "h", "orc": "o", "undead": "u", "nightelf": "e"}


@lru_cache(1)
def buff_tables() -> tuple[dict, dict, dict]:
    """(ability -> its buffs (BuffID of every level, effects left out), buff -> (name, race), item -> its row)."""
    from warcraftsim.data.mpq import GameArchives
    from warcraftsim.data.objects import parse_slk
    with GameArchives() as g:
        ad = parse_slk(g.read("Units\\AbilityData.slk").decode("latin-1"))
        bd = parse_slk(g.read("Units\\AbilityBuffData.slk").decode("latin-1"))
        idata = parse_slk(g.read("Units\\ItemData.slk").decode("latin-1"))
    buffs = {r["alias"]: (str(r.get("comments") or "").strip(), str(r.get("race") or "").strip())
             for r in bd if r.get("alias") and str(r.get("isEffect") or "0").strip() != "1"}
    # (an ability may spell its buff in another case than the buff table: Unholy Frenzy's BuffID is BUhf, the table's
    # Buhf; the game puts the ability's spelling on the unit, scripts/obs_probe.py: GetUnitAbilityLevel(unit, 'Buhf') 0)
    lower = {b.lower(): b for b in buffs}
    abil: dict[str, set[str]] = {}
    for r in ad:
        a = r.get("alias")
        if not a:
            continue
        got = set()
        for k in range(1, 5):
            for b in str(r.get(f"BuffID{k}") or "").split(","):
                b = b.strip()
                if b in buffs:
                    got.add(b)
                elif b.lower() in lower:
                    buffs.setdefault(b, buffs[lower[b.lower()]])
                    got.add(b)
        abil[a] = got
    items = {r["itemID"]: r for r in idata if r.get("itemID") and len(r["itemID"]) == 4}
    return abil, buffs, items


@lru_cache(None)
def map_units(name: str) -> dict[str, int]:
    """The units a map's script places (CreateUnit), by type."""
    from warcraftsim.data.duelmap import parse_echo_name
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.data.mpq import MpqArchive
    parsed = parse_echo_name(name)
    m = MpqArchive(stock_map_path(parsed[0] if parsed else name))
    text = m.read("war3map.j").decode("latin-1")
    return dict(Counter(re.findall(r"CreateUnit\(\s*\w+\s*,\s*'(\w{4})'", text)))


@lru_cache(None)
def map_drop_levels(name: str) -> frozenset[tuple[str, int]]:
    """The (item class, level) pairs a map's creep drop tables ask for (ChooseRandomItemEx)."""
    from warcraftsim.data.duelmap import parse_echo_name
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.data.mpq import MpqArchive
    parsed = parse_echo_name(name)
    m = MpqArchive(stock_map_path(parsed[0] if parsed else name))
    text = m.read("war3map.j").decode("latin-1")
    cls = {"PERMANENT": "Permanent", "CHARGED": "Charged", "POWERUP": "PowerUp", "ARTIFACT": "Artifact",
           "PURCHASABLE": "Purchasable", "CAMPAIGN": "Campaign", "MISCELLANEOUS": "Miscellaneous"}
    return frozenset((cls.get(c, c), int(lv)) for c, lv in re.findall(r"ChooseRandomItemEx\(\s*ITEM_TYPE_(\w+)\s*,\s*(\d+)", text))


def creeps(maps=fv.MAPS[:2]) -> set[str]:
    """The neutral hostile units of the maps (non-building 'n' types: creeps; critters left out)."""
    U = fv.tables()["units"]
    out = set()
    for m in maps:
        for c in map_units(m):
            u = U.get(c)
            if c[0] == "n" and u is not None and not u["building"] and c not in ("npig", "nrac", "nshe", "nfro", "ndog"):
                out.add(c)
    return out


def drop_items(maps=fv.MAPS[:2]) -> set[str]:
    _, _, items = buff_tables()
    want = set()
    for m in maps:
        want |= map_drop_levels(m)
    return {c for c, r in items.items() if str(r.get("pickRandom") or "0").strip() == "1"
            and (str(r.get("class") or ""), int(float(str(r.get("Level") or 0) or 0))) in want}


def shop_items(race: str) -> set[str]:
    from warcraftsim.fullgame.neutral import MERCHANT_ITEMS
    from warcraftsim.fullgame.shopitems import RACE_SHOPS, STOCK
    shop = RACE_SHOPS[race]
    return set(STOCK[shop]) | set(fv.EXTRA_ITEMS.get(shop, ())) | set(MERCHANT_ITEMS)


def own_types(race: str) -> list[str]:
    """The unit types of the race itself (its units, buildings, heroes, their summons and morphed forms)."""
    return sorted(u for u in fv.unit_types(race) if u[0].lower() == RACE_LETTER[race] and u not in fv.MAP_SHOPS)


def neutral_types() -> tuple[list[str], list[str]]:
    """(the tavern's heroes and their summons, the mercenary camp's and the goblin laboratory's units): any race's."""
    F = fv.tables()["funcs"]
    mercs = set(F.get("nmer", {}).get("Sellunits", ())) | set(F.get("ngad", {}).get("Sellunits", ()))
    every = set().union(*(fv.unit_types(r) for r in fv.RACES))
    neutral = {u for u in every if u[0] in "nN" and u not in fv.MAP_SHOPS}
    return sorted(neutral - mercs), sorted(neutral & mercs)


SOURCES = ("human", "orc", "undead", "nightelf", "tavern", "mercenaries", "creeps", "items")


def source_types() -> dict[str, list[str]]:
    tavern, mercs = neutral_types()
    return {**{r: own_types(r) for r in fv.RACES}, "tavern": tavern, "mercenaries": mercs, "creeps": sorted(creeps())}


def buff_sources() -> dict[str, dict[str, set[str]]]:
    """Source (SOURCES: a race's own units and heroes, the tavern's heroes and their summons, the mercenaries and the
    laboratory's units, the maps' creeps, the items) -> buff -> the units / items whose abilities put it on."""
    abil, buffs, items = buff_tables()
    U = fv.tables()["units"]
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for src, types in source_types().items():
        for u in types:
            for a in (U[u]["abil"] + U[u]["hero_abil"]) if u in U else ():
                for b in abil.get(a, ()):
                    out[src][b].add(u)
    held = set().union(*(shop_items(r) for r in fv.RACES)) | drop_items()
    for it in held:
        r = items.get(it)
        for a in [a.strip() for a in str((r or {}).get("abilList") or "").split(",") if a.strip()]:
            for b in abil.get(a, ()):
                out["items"][b].add(it)
    return {s: dict(v) for s, v in out.items()}


def item_coverage() -> dict:
    """The items a hero may hold, grouped by what the hero block encodes of them (herodata.item_table)."""
    from warcraftsim.fullgame import herodata
    from warcraftsim.protocol import fourcc
    _, _, items = buff_tables()
    table = herodata.item_table()
    held = set().union(*(shop_items(r) for r in fv.RACES)) | drop_items()
    groups: dict[tuple, list[str]] = defaultdict(list)
    charged = []
    for it in sorted(held):
        groups[tuple(table.get(fourcc(it), (0, 0, 0, 0)))].append(it)
        uses = str((items.get(it) or {}).get("uses") or "").strip()
        if uses not in ("", "-", "0"):
            charged.append(it)
    return {"items": len(held), "codes": len(groups), "groups": dict(groups), "charged": charged}


def research_coverage(vocab: dict) -> dict[str, list[str]]:
    """Per race its researches the global vector has no level of."""
    from warcraftsim.fullgame import techtree as tt
    have = {fx.rawcode(int(u)) for u in vocab["upgrades"]}
    out = {}
    for race in fv.RACES:
        letter = RACE_LETTER[race]
        codes = sorted(c for c, makers in tt.tree().makers.items() if c[0] == "R" and any(m[0] == letter for m in makers))
        out[race] = [c for c in codes if c not in have]
    return out


def unit_ability_cooldowns(types: list[str]) -> dict[str, list[str]]:
    """The non-hero units' castable abilities with a cooldown: ability -> its units."""
    T = fv.tables()
    U, A = T["units"], T["abilities"]
    out: dict[str, list[str]] = defaultdict(list)
    for u in types:
        if u not in U or fv.is_hero(u):
            continue
        for a in U[u]["abil"]:
            ab = A.get(a)
            if ab is None or a in fv.SKIP or ab["base"] in fv.SKIP:
                continue
            orders = fv._ability_orders(a)  # (the files' order strings with fullvocab.BASE_ORDERS' fixes)
            if any(c > 0 for c in ab["cool"]) and (orders.get("Order") or orders.get("Orderon")):
                out[a].append(u)
    return dict(out)


def toggles(types: list[str]) -> dict[str, list[str]]:
    """The abilities with an autocast or on/off state (Orderon / Orderoff): ability -> its units."""
    T = fv.tables()
    U, A = T["units"], T["abilities"]
    out: dict[str, list[str]] = defaultdict(list)
    for u in types:
        if u not in U:
            continue
        for a in U[u]["abil"] + U[u]["hero_abil"]:
            ab = A.get(a)
            orders = fv._ability_orders(a) if ab is not None else {}
            if "Orderon" in orders and "Orderoff" in orders:
                out[a].append(u)
    return dict(out)


def type_coverage(vocab: dict) -> dict[str, list[str]]:
    """Per source (SOURCES but items) its unit types without an embedding of their own in the vocabulary."""
    have = {fx.rawcode(int(t)) if not isinstance(t, str) else t for t in vocab["types"]}
    return {k: [t for t in v if t not in have] for k, v in source_types().items()}


def channels(run: dict) -> dict:
    """The run's entity and global channels (features.describe_spaces) and those --features zeroes."""
    cfg = run.get("config", {})
    fx.set_inputs(int(cfg.get("inputs", 1) or 1))
    feats = int(cfg.get("features", fx.F) or fx.F)
    return {"inputs": fx.INPUTS, "F": fx.F, "features": feats, "zeroed": list(range(feats, fx.F_BASE)) if feats < fx.F_BASE else []}


def probe_buffs() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("obs_probe", ROOT / "scripts" / "obs_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BUFFS


def report(run_dir: Path) -> tuple[list[str], dict]:
    run = json.loads((run_dir / "run.json").read_text())
    vocab = json.loads((run_dir / "vocab.json").read_text())
    _, buffs, _ = buff_tables()
    lines: list[str] = []
    summary: dict = {}
    # channels
    ch = channels(run)
    names = fx.describe_spaces(vocab)["observation"]["blocks"][1]["features"]
    zeroed = [names[i] for i in ch["zeroed"] if i < len(names)]
    lines += [f"### What `{run['name']}` reads", "",
              f"`--inputs {ch['inputs']}`: {ch['F']} channels per entity, of which `--features {ch['features']}` zeroes "
              f"{len(zeroed)} (channels {ch['zeroed'][0]}-{ch['zeroed'][-1]}: {', '.join(zeroed[:3])}, ... "
              f"{zeroed[-1]})." if zeroed else f"`--inputs {ch['inputs']}`: {ch['F']} channels per entity, all in use.",
              ""]
    summary["zeroed"] = len(zeroed)
    # buffs
    src = buff_sources()
    universe = sorted(set().union(*(set(v) for v in src.values())))
    asked = set(probe_buffs())
    lines += ["### Buffs", "",
              f"{len(universe)} buffs can be on a unit in these games; without `--obs-full` the observation encodes none of "
              "them (a few show indirectly: a sleeping creep's flag, a summon's flag, an own illusion's flag). With "
              "`--obs-full 1` every one reaches the network: the shim reads each unit's buff list (obsfull.K_BUFF, a code "
              f"per buff id). `scripts/obs_probe.py buffs` asks the game for {len(asked & set(universe))} of them (BUFFS); "
              "`GetUnitAbilityLevel(unit, buff)` answers for any.", "", "| from | buffs (the units or items that cast them) |",
              "|---|---|"]
    for s in SOURCES:
        got = src.get(s, {})
        cells = [f"{b} {buffs[b][0]}{'' if b in asked else ' (not probed)'} ({', '.join(sorted(got[b])[:3])}"
                 f"{'...' if len(got[b]) > 3 else ''})" for b in sorted(got)]
        lines.append(f"| {s} | {'; '.join(cells)} |")
    summary["buffs"] = len(universe)
    summary["buffs_not_probed"] = sorted(set(universe) - asked)
    summary["probe_unknown"] = sorted(asked - set(buffs))  # (a code in the probe's list that is no buff of the game's)
    if summary["probe_unknown"]:
        lines += ["", f"The probe's list has codes that are no buff of the game's: {', '.join(summary['probe_unknown'])}."]
    lines.append("")
    # items
    ic = item_coverage()
    ambiguous = {k: v for k, v in ic["groups"].items() if len(v) > 1}
    lines += ["### Items", "",
              f"{ic['items']} item types a hero may hold in these games (the race shops, the goblin merchant, the maps' "
              f"drop tables); the hero block's five numbers per slot (held, usable, heals, restores mana, town portal) "
              f"tell {ic['codes']} kinds apart. {len(ic['charged'])} of them have charges, which no input counts. "
              "(`--obs-full 1`: each item's own code and each slot's charges.) Items that look alike:", "",
              "| held, usable, heals, mana, portal | items |", "|---|---|"]
    for k, v in sorted(ambiguous.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| 1, {', '.join(map(str, k))} | {len(v)}: {', '.join(v)} |")
    summary["item_codes"] = (ic["items"], ic["codes"])
    lines.append("")
    # researches
    rc = research_coverage(vocab)
    lines += ["### Researches", "",
              f"The global vector has the player's own level of {len(vocab['upgrades'])} researches (the vocabulary's "
              "upgrades); the enemy's levels are no input. Researches without a level there:", ""]
    for race in fv.RACES:
        lines.append(f"- **{race}**: {', '.join(rc[race]) or 'none'}")
    summary["research_missing"] = {r: len(rc[r]) for r in fv.RACES}
    lines.append("")
    # cooldowns and toggles
    lines += ["### Non-hero cooldowns and toggles", "",
              "Only a hero's four ability slots report a cooldown; a unit's castable abilities with one, and the "
              "abilities with an autocast or on/off state (their state is no input; `fullvocab.Checks` tracks it from "
              "`DEFAULT_ON` and the agent's own orders for the mask):", "", "| race | abilities with a cooldown | toggles |",
              "|---|---|---|"]
    for k, types in source_types().items():
        cd, tg = unit_ability_cooldowns(types), toggles(types)
        lines.append(f"| {k} | {', '.join(f'{a} ({u[0]})' for a, u in sorted(cd.items())) or '-'} | "
                     f"{', '.join(f'{a} ({u[0]})' for a, u in sorted(tg.items())) or '-'} |")
    lines.append("")
    # types
    from warcraftsim.data.objects import unit_names
    names_ = unit_names()
    tc = type_coverage(vocab)
    lines += ["### Unit types", "", f"Types a player meets without an embedding of their own in `{run['name']}`'s vocabulary "
              "(their entities get index 0, unknown). `--full-vocab 1` adds the races', the tavern's and the mercenaries' "
              "(`fullvocab.extend`), `--obs-full 1` the creeps' (`obsfull.extend`):", ""]
    for k, v in tc.items():
        lines.append(f"- **{k}**: " + (', '.join(f"{t} ({names_.get(t, '?')})" if k == "creeps" else t for t in v)
                                       or 'none'))
    summary["types_missing"] = {k: len(v) for k, v in tc.items()}
    return lines, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=Path("/home/tora/warcraftsim/runs/fgself-194"))
    ap.add_argument("--doc", type=Path, default=None, help="write the generated part between the doc's markers")
    a = ap.parse_args()
    lines, summary = report(a.run)
    print("\n".join(lines))
    print("SUMMARY " + json.dumps(summary))
    if a.doc is not None:
        text = a.doc.read_text()
        i, j = text.find(BEGIN), text.find(END)
        if i < 0 or j < i:
            raise SystemExit(f"{a.doc}: no markers {BEGIN!r} ... {END!r}")
        a.doc.write_text(text[:i + len(BEGIN)] + "\n\n" + "\n".join(lines) + "\n\n" + text[j:])
        print(f"wrote {a.doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
