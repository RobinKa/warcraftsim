"""--full-vocab in the real game (fullgame/fullvocab.py): every order the vocabulary gained, given through the policy's own path
(play.BCAgent.observe, its masks, commands, accepted; no network), with the game's answer and what it did read from the game's
state.

    python3 scripts/vocab_probe.py ids                          # the game's order id of every order string (OrderId)
    python3 scripts/vocab_probe.py orders --race human [--only polymorph,patrol]
    python3 scripts/vocab_probe.py items                        # drop, give, sell, pick up; the critter, the tiny hall
    python3 scripts/vocab_probe.py neutral                      # the goblin laboratory: its units, load and unload, reveal

ids: one game whose harness resolves every order string of the game's executable (its order string table) and writes
warcraftsim/data/order_ids.json, which fullvocab.ORDER_IDS reads. orders: each race's new orders by units spawned for it,
the research they need set in the game (SetResearch), each order given with the right pointer and with a wrong one (the
agent's fix-up), masked and unmasked states checked. Instance names start with "vocabprobe"; one game at a time.

Results: see RESULTS below (filled from the runs of 2026-11-06).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"

ROOT = Path(__file__).resolve().parents[1]
EXE = Path.home() / "wc3/legacy-1.29/Warcraft III.exe"


def exe_order_strings(exe: Path = EXE) -> list[str]:
    """The game's order strings, as its executable lists them (the block from "attackground" to "neutralspell")."""
    data = exe.read_bytes()
    words = [w.decode("latin-1") for w in re.findall(rb"[\x20-\x7e]{3,}", data)]
    a = words.index("attackground")
    b = words.index("neutralspell", a)
    return [w for w in words[a:b + 1] if re.fullmatch(r"[A-Za-z_0-9]+", w)]


def ids(a) -> int:
    from warcraftsim.data import mapbuild
    from warcraftsim.protocol import all_order_names
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
    extra = ["smart", "stop", "cancel", "instant", "stunned", "setrally", "patrol", "holdposition", "AImove",
             *(f"useslot{k}" for k in range(1, 7)), *(f"moveslot{k}" for k in range(1, 7)), *exe_order_strings()]
    names = tuple(dict.fromkeys(all_order_names() + tuple(extra)))  # (the data files' strings, then the executable's)
    mapbuild.HarnessConfig.resolved_order_names = lambda self: names  # (this process only: its own map build)
    slots = [Agent("human", handicap=100, difficulty="normal"), Agent("orc", handicap=100, difficulty="normal")]
    gs = GameSetup(map=a.map, slots=slots, step_seconds=0.5, draw=False, native_obs=False, d3d_thread=False, render_threads=0,
                   nice=19, warm_spare=False, window=(320, 240), max_game_seconds=0, victory="decisive", shim_dir=a.shim)
    with GameInstance(gs, name=f"vocabprobe{a.name}", timeout=600) as g:
        obs = g.start()
        got = dict(sorted((k, int(v)) for k, v in (obs.orders or {}).items()))
    missing = [n for n in names if n not in got]
    print(f"RESULT {len(got)} of {len(names)} order strings resolved; none for {missing}", flush=True)
    out = ROOT / "warcraftsim" / "data" / "order_ids.json"
    out.write_text(json.dumps(got, indent=0, sort_keys=True) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("ids",))
    ap.add_argument("--map", default="echofast2")
    ap.add_argument("--name", default="")
    ap.add_argument("--shim", default=str(ROOT / "build" / "shim"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    return {"ids": ids}[a.what](a)


if __name__ == "__main__":
    raise SystemExit(main())
