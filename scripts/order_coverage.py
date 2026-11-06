"""Which of the orders a player can give in a 1.29 melee game a vocabulary has (fullgame/fullvocab.py's universe, from the
game's own tables), per race, with --full-vocab off (the vocabulary as it is) and on (fullvocab.extend's), and the neutral
buildings on the maps the runs play.

    python3 scripts/order_coverage.py [--vocab runs/fgself-194/vocab.json] [--doc docs/reference/order-coverage.md]

Printed: per race the orders covered of all, what is missing (by kind: train, build, upgrade, research, skill, spell,
ability, toggle, morph, unmorph, generic, item, ware, neutral), the unit types missing, what is left out on purpose and why
(fullvocab.EXCLUDED), and each map's neutral buildings that take an order. --doc writes the table there.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from warcraftsim.fullgame import features as fx  # noqa: E402
from warcraftsim.fullgame import fullvocab as fv  # noqa: E402


def map_shops(name: str) -> dict[str, int]:
    """The neutral buildings a map places (its script's CreateUnit calls) that take an order (fullvocab.MAP_SHOPS) or none
    (the others: fountains, waygates, creep huts), by type -> count."""
    from warcraftsim.data.duelmap import parse_echo_name
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.data.mpq import MpqArchive
    parsed = parse_echo_name(name)
    src = stock_map_path(parsed[0] if parsed else name)
    m = MpqArchive(src)
    text = ""
    for f in ("war3map.j", "scripts\\war3map.j"):
        try:
            text = m.read(f).decode("latin-1")
            break
        except Exception:  # noqa: BLE001
            continue
    units = fv.tables()["units"]
    got = Counter(c for c in re.findall(r"CreateUnit\(\s*\w+\s*,\s*'(\w{4})'", text)
                  if c[:1] == "n" and units.get(c, {}).get("building"))
    return dict(sorted(got.items()))


def report(vocab: dict, label: str) -> dict:
    cov = fv.coverage(vocab)
    print(f"== {label}")
    for race, c in cov.items():
        kinds = Counter(e.what for e in c["missing"])
        print(f"  {race:8s} orders {c['have']}/{c['orders']}  types {c['types'] - len(c['missing_types'])}/{c['types']}  "
              f"missing by kind {dict(sorted(kinds.items()))}")
        for e in c["missing"]:
            print(f"      {e.what:8s} {fv.order_name(e.oid):28s} ({fx.KIND_NAMES[e.kind]}) by {', '.join(sorted(e.givers)[:6])}"
                  f"{'...' if len(e.givers) > 6 else ''}")
        if c["missing_types"]:
            print(f"      types: {' '.join(c['missing_types'])}")
    return cov


def doc(before: dict, after: dict, vocab_path: str, shops: dict, out: Path) -> None:
    lines = ["# Order coverage", "",
             "Which orders a player can give in a 1.29 melee game the policy's vocabulary has. The list of every order comes "
             "from the game's own tables (`warcraftsim/fullgame/fullvocab.py`: `Units\\*UnitFunc.txt`, `UnitAbilities.slk`, "
             "`AbilityData.slk`, the ability order strings; hero abilities from `data/abilities.py`; the race shops' stock "
             "from `shopitems.py`; the neutral buildings of the maps below). Written by `scripts/order_coverage.py` for "
             f"`{vocab_path}`; with `--full-vocab 1` a vocabulary gets every missing order (`fullvocab.extend`).", "",
             "| race | orders before | orders with --full-vocab | unit types before | with --full-vocab |",
             "|---|---|---|---|---|"]
    for race in fv.RACES:
        b, a = before[race], after[race]
        lines.append(f"| {race} | {b['have']}/{b['orders']} | {a['have']}/{a['orders']} | "
                     f"{b['types'] - len(b['missing_types'])}/{b['types']} | {a['types'] - len(a['missing_types'])}/{a['types']} |")
    lines += ["", "An order counts as covered in its own target form, or for a spell or unit ability aimed at a place also in "
              "the other one (the built-in AI casts blizzard, earthquake, stasis trap and dispel on a unit's place, raise "
              "dead and carrion scarabs on a corpse); a peasant's call to arms and a militia's back to work are the "
              "vocabulary's militiaconvert and militiaunconvert, which the agent turns into them.", ""]
    for race in fv.RACES:
        b = before[race]
        lines += [f"## {race}", "", "Missing before (each added by `--full-vocab 1`):", ""]
        by: dict[str, list[str]] = {}
        for e in b["missing"]:
            by.setdefault(e.what, []).append(f"{fv.order_name(e.oid)} ({fx.KIND_NAMES[e.kind]})")
        for what, names in sorted(by.items()):
            lines.append(f"- **{what}**: {', '.join(names)}")
        if b["missing_types"]:
            lines.append(f"- **unit types**: {', '.join(b['missing_types'])}")
        lines.append("")
    lines += ["## Left out on purpose", ""]
    for k, why in fv.EXCLUDED.items():
        lines.append(f"- **{k}**: {why}")
    lines += ["", "## Neutral buildings on the maps", "",
              "Those that take an order: the tavern (`ntav`: its heroes), the mercenary camp (`nmer`: its mercenaries), the "
              "goblin laboratory (`ngad`: zeppelin, shredder, sapper, and its reveal), the goblin merchant (`ngme`: its "
              "items; it buys items too) and the marketplace (`nmrk`: it buys items; what it sells is left out, above). "
              "The others take none (fountains, creep huts).", "", "| map | neutral buildings |", "|---|---|"]
    for m, got in shops.items():
        lines.append(f"| {m} | {', '.join(f'{c} x{n}' + ('' if c in fv.MAP_SHOPS else ' (none)') for c, n in got.items())} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vocab", default="/home/tora/warcraftsim/runs/fgself-194/vocab.json")
    ap.add_argument("--doc", default="", help="write the table to this markdown file")
    ap.add_argument("--no-maps", action="store_true", help="skip reading the maps' neutral buildings")
    a = ap.parse_args()
    vocab = json.loads(Path(a.vocab).read_text())
    if vocab.get("full_vocab"):
        print("(the vocabulary has --full-vocab's orders already)")
    before = report(vocab, f"{a.vocab} as it is (--full-vocab 0)")
    full = fv.extend(vocab)
    print(f"--full-vocab 1: orders {len(vocab['orders'])} -> {len(full['orders'])}, types {len(vocab['types'])} -> "
          f"{len(full['types'])}; by kind {dict(Counter(fv.entry(o, k).what for o, k in full['orders'][len(vocab['orders']):]))}")
    after = report(full, "with --full-vocab 1")
    print("left out on purpose:")
    for k, why in fv.EXCLUDED.items():
        print(f"  {k}: {why}")
    shops = {}
    if not a.no_maps:
        for m in fv.MAPS:
            shops[m] = map_shops(m)
            off = [c for c in shops[m] if c not in fv.MAP_SHOPS]
            print(f"  {m}: {shops[m]} (no order: {off})")
    if a.doc:
        doc(before, after, a.vocab, shops, Path(a.doc))
    return 0 if all(not after[r]["missing"] and not after[r]["missing_types"] for r in fv.RACES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
