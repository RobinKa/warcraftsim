"""The dry-main start (fullgame/scenarios.py, the scenario "dry") in the real game.

    python3 scripts/dry_probe.py --races human,orc [--dry 1] [--ai 1] [--minutes 14]

A game with no policy: side 0 an agent slot that does nothing, side 1 the built-in AI (insane; --ai 0: an idle agent
slot too). At step 0 the commands scenarios.commands("dry") gives (--dry 0: none, to compare), what the game answered,
each main mine's gold and the creeps at each natural before and after; then each game minute: both mains' gold, the AI's
gold gathered and halls (a hall farther than NATURAL_FROM_START from its start is an expansion). The question: whether
the setup takes, and whether the built-in AI's script copes with a dry main (it expands, or floats, or stalls).

Instance names start with "dryprobe".

Results (2026-11-06, echofast2, human idle agent vs orc insane AI): the setup takes (4 of 4 answered; both mains
6,750 -> 3,793 at a share of 0.56, the naturals' 3 creeps each -> 0). The AI copes: it expanded at 3 minutes (its natural
cleared) against 4 minutes without the scenario; a main with 5 workers loses ~590 gold a minute, so at 25-60% it runs
dry in ~3-7 minutes. Both games ended at 5-6 minutes (the AI beat the idle side), so a long game was not seen.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.pop("WAYLAND_DISPLAY", None)  # (games get Xvfbs of their own)
os.environ["DISPLAY"] = ":65535"

from base2_probe import creeps_at, forced_start, natural_of, setup  # noqa: E402

from warcraftsim.fullgame import late, scenarios  # noqa: E402
from warcraftsim.runtime.instance import Agent, BuiltinAI, GameInstance  # noqa: E402

HALLS = ("htow", "hkee", "hcas", "ogre", "ostr", "ofrt", "unpl", "unp1", "unp2", "etol", "etoa", "etoe")


def mains(obs) -> dict:
    """Each start's nearest gold mine within NATURAL_FROM_START: its gold."""
    out = {}
    for s in (0, 1):
        sx, sy = obs.players[s].start_x, obs.players[s].start_y
        near = sorted((math.hypot(u.x - sx, u.y - sy), u.resource) for u in obs.units
                      if u.type in late.MINES and u.alive and math.hypot(u.x - sx, u.y - sy) <= scenarios.NATURAL_FROM_START)
        out[s] = near[0][1] if near else None
    return out


def halls(obs, s: int) -> tuple[int, int]:
    """The side's halls at home and away (farther than NATURAL_FROM_START from its start)."""
    sx, sy = obs.players[s].start_x, obs.players[s].start_y
    hs = [u for u in obs.units if u.owner == s and u.alive and u.type in HALLS]
    away = sum(1 for u in hs if math.hypot(u.x - sx, u.y - sy) > scenarios.NATURAL_FROM_START)
    return len(hs) - away, away


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", default="echofast2")
    ap.add_argument("--races", default="human,orc")
    ap.add_argument("--dry", type=int, default=1)
    ap.add_argument("--ai", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=14.0)
    ap.add_argument("--name", default="")
    ap.add_argument("--left", type=int, default=0)
    ap.add_argument("--shim", default=str(Path(__file__).resolve().parents[1] / "build" / "shim"))
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args()
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    races = a.races.split(",")
    slots = [Agent(races[0], handicap=100, difficulty="normal"),
             BuiltinAI(races[1], "insane") if a.ai else Agent(races[1], handicap=100, difficulty="normal")]
    with GameInstance(setup(a.map, slots, a.shim, a.minutes * 60 + 30), name=f"dryprobe{a.name}", timeout=900) as g:
        obs = forced_start(g, a)
        nat = {s: natural_of(obs, s) for s in (0, 1)}
        print(f"map {a.map}, races {races}, dry {a.dry}, ai {a.ai}: mains {mains(obs)}, creeps at naturals "
              f"{[creeps_at(obs, *nat[s]) if nat[s] else None for s in (0, 1)]}", flush=True)
        cmds = scenarios.commands("dry", obs, 0, races, random.Random()) if a.dry else []
        obs = g.step(cmds)
        res = list(obs.command_results or [])
        print(f"step 0: {[(type(c).__name__, getattr(c, 'gold', '')) for c in cmds]} answered {res}", flush=True)
        obs = g.step([])
        print(f"step 1: mains {mains(obs)}, creeps at naturals {[creeps_at(obs, *nat[s]) if nat[s] else None for s in (0, 1)]}",
              flush=True)
        next_minute = 60.0
        while obs.game_time < a.minutes * 60 and not obs.game_over:
            obs = g.step([])
            if obs.game_time >= next_minute:
                p1 = obs.players.get(1)
                print(f"  {obs.game_time / 60:4.1f} min: mains {mains(obs)}, side 1 gathered "
                      f"{getattr(p1, 'gold_gathered', None)} gold {getattr(p1, 'gold', None)}, halls (home, away) "
                      f"{halls(obs, 1)}, food {getattr(p1, 'food_used', None)}", flush=True)
                next_minute += 60.0
        print(f"RESULT {races} dry {a.dry}: over at {obs.game_time:.0f} s, side 1 halls {halls(obs, 1)}, mains {mains(obs)}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
