"""Does drawing (or a shim setting) change the simulation? Play a built-in AI game with the game drawing,
save its replay, and play the replay back with drawing on and with drawing off (the shim's
W3SIM_DRAW=0): every unit's position, hit points, mana and order must agree at every step.
--env NAME=a,b instead plays the replay back (not drawing) once per value of a shim environment
variable, e.g. --env W3SIM_PARK=0,1.

    python3 scripts/draw_parity.py [--steps 300] [--map duelfast] [--races human,orc] [--env W3SIM_PARK=0,1]
        [--shim build/shim-test]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warcraftsim.runtime.instance as instance  # noqa: E402
from warcraftsim.runtime.instance import BuiltinAI, GameInstance, GameSetup  # noqa: E402


def rows(obs) -> dict:
    return {u.id: (u.type_id, u.owner, round(u.x, 1), round(u.y, 1), round(u.hp, 1), round(u.mana, 1), u.order)
            for u in obs.units}


def play(setup: GameSetup, steps: int, replay: Path | None, name: str, draw: bool) -> tuple[list[dict], Path | None]:
    out = []
    with GameInstance(dataclasses.replace(setup, draw=draw), name=name, timeout=180) as g:
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
    ap.add_argument("--env", help="NAME=a,b: play back once per value of this environment variable instead")
    ap.add_argument("--shim", type=Path, help="the shim's build folder (default build/shim)")
    a = ap.parse_args()
    if a.shim:
        instance.SHIM_DIR = a.shim.resolve()
    r0, r1 = a.races.split(",")
    setup = GameSetup(map=a.map, slots=[BuiltinAI(r0, "normal", handicap=50), BuiltinAI(r1, "normal", handicap=50)],
                      step_seconds=0.5, max_game_seconds=3600, victory="decisive", window=(320, 240))
    live, replay = play(setup, a.steps, None, "parity_live", draw=True)
    print(f"live game: {len(live)} steps, {len(live[-1])} units at the end; replay {replay}", flush=True)
    if a.env:
        name, values = a.env.split("=", 1)
        runs = []
        for v in values.split(","):
            os.environ[name] = v
            run, _ = play(setup, a.steps, replay, f"parity_{v}", draw=False)
            print(f"playback, {name}={v}: ", compare(live, run), flush=True)
            runs.append((v, run))
        for (v, x), (w, y) in zip(runs, runs[1:]):
            print(f"{name}={v} vs {w}:  ", compare(x, y), flush=True)
        return 0
    drawn, _ = play(setup, a.steps, replay, "parity_on", draw=True)
    print("playback drawing:     ", compare(live, drawn), flush=True)
    blind, _ = play(setup, a.steps, replay, "parity_off", draw=False)
    print("playback not drawing: ", compare(live, blind), flush=True)
    print("drawing vs not:       ", compare(drawn, blind), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
