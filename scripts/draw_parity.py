"""Does drawing change the simulation? Play a built-in AI game with the game drawing, save its replay,
and play the replay back with drawing on and with drawing off (the shim's W3SIM_DRAW=0): every unit's
position, hit points, mana and order must agree at every step.

    python3 scripts/draw_parity.py [--steps 300] [--map duelfast] [--races human,orc]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warcraftsim.runtime.instance import BuiltinAI, GameInstance, GameSetup  # noqa: E402


def rows(obs) -> dict:
    return {u.id: (u.type_id, u.owner, round(u.x, 1), round(u.y, 1), round(u.hp, 1), round(u.mana, 1), u.order)
            for u in obs.units}


def play(setup: GameSetup, steps: int, replay: Path | None, name: str, draw: bool) -> tuple[list[dict], Path | None]:
    os.environ["W3SIM_DRAW"] = "1" if draw else "0"
    out = []
    with GameInstance(setup, name=name, timeout=180) as g:
        obs = g.play_replay(replay) if replay else g.start()
        for _ in range(steps):
            out.append(rows(obs))
            if obs.game_over:
                break
            obs = g.step([])
        saved = None
        if replay is None:
            saved = g.save_replay(Path(tempfile.mkdtemp()) / "parity.w3g")
    return out, saved


def compare(a: list[dict], b: list[dict]) -> str:
    for t, (x, y) in enumerate(zip(a, b)):
        if x != y:
            diff = sorted(set(x) ^ set(y))[:3] or [k for k in x if x[k] != y.get(k)][:3]
            return f"step {t}: differ ({len(x)} vs {len(y)} units; e.g. {[(k, x.get(k), y.get(k)) for k in diff]})"
    return f"identical over {min(len(a), len(b))} steps"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--map", default="duelfast")
    ap.add_argument("--races", default="human,orc")
    a = ap.parse_args()
    r0, r1 = a.races.split(",")
    setup = GameSetup(map=a.map, slots=[BuiltinAI(r0, "normal", handicap=50), BuiltinAI(r1, "normal", handicap=50)],
                      step_seconds=0.5, max_game_seconds=3600, victory="decisive", window=(320, 240))
    live, replay = play(setup, a.steps, None, "parity_live", draw=True)
    print(f"live game: {len(live)} steps, {len(live[-1])} units at the end; replay {replay}", flush=True)
    drawn, _ = play(setup, a.steps, replay, "parity_on", draw=True)
    print("playback drawing:     ", compare(live, drawn), flush=True)
    blind, _ = play(setup, a.steps, replay, "parity_off", draw=False)
    print("playback not drawing: ", compare(live, blind), flush=True)
    print("drawing vs not:       ", compare(drawn, blind), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
