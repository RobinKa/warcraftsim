"""Re-render a whole-game self-play video from its replay (e.g. after changing the camera or the overlay).

    python scripts/rerender_fullgame.py fgself-64 game-1792106881-a4 [--out file.mp4]

Uses runs/RUN/replays/STEM.w3g (with its .commands.json: the map it was recorded on, and .trace.json: the
panel and the players), the run's launch flags (map, step, handicap, game length) and the run's GE Wine.
Default output: the run's videos/STEM.mp4 (replacing it). Renders in real time, with sound.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("stem", help="the replay's stem, e.g. game-1792106881-a4")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    run_dir = ROOT / "runs" / a.run
    argv = json.loads((run_dir / "run.json").read_text())["launch"]["argv"]

    def flag(name: str, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    from warcraftsim.runtime import wine
    if flag("--wine"):
        wine.use_wine(flag("--wine"), bool(int(flag("--fsync", "1"))))
    from warcraftsim.fullgame.overlay import FullGameOverlay
    from warcraftsim.runtime.instance import Agent, BuiltinAI, GameSetup
    from warcraftsim.video import render_replay

    replay = run_dir / "replays" / f"{a.stem}.w3g"
    trace = json.loads(replay.with_suffix(".trace.json").read_text())
    handicap = int(flag("--handicap", 50))
    slots = []
    for side in sorted(trace["sides"], key=lambda s: s["player"]):
        race = re.search(r"\((human|orc|undead|nightelf)\)", side["name"]).group(1)
        if side["kind"] == "ai":
            difficulty = re.search(r"(easy|normal|insane)", side["name"])
            slots.append(BuiltinAI(race, difficulty.group(1) if difficulty else "insane", handicap=handicap))
        else:
            slots.append(Agent(race, handicap=handicap))
    setup = GameSetup(map=flag("--map", "duelrush"), slots=slots, step_seconds=float(flag("--step-seconds", 0.5)),
                      max_game_seconds=float(flag("--max-minutes", 4.0)) * 60, victory="decisive", window=(320, 240),
                      draw=False, d3d_thread=False, render_threads=0)
    out = a.out or run_dir / "videos" / f"{a.stem}.mp4"
    render_replay(setup, replay, out, name="rerender", crf=28, overlay=FullGameOverlay(trace), fit_all=True,
                  max_steps=len(trace.get("steps") or []) + 40)
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
