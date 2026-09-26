"""Re-render a recorded episode's video (e.g. after changing the camera or the overlay).

    python scripts/rerender_video.py rejoin-1 w0-episode003175 [--out file.mp4]

Uses the run's task and replay (runs/RUN/replays/EPISODE.w3g and .steps.npz) and draws the same
overlay as the bridge (the policy panel from the checkpoint of that time). Default output: the
run's videos/EPISODE.mp4 (replacing it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from warcraftsim.puffer.bridge import BridgeServer
from warcraftsim.puffer.tasks import get_task
from warcraftsim.puffer.train import RUNS_DIR
from warcraftsim.video import render_replay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("episode", help="the replay's stem, e.g. w0-episode003175")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    run_dir = RUNS_DIR / args.run
    info = json.loads((run_dir / "run.json").read_text())
    task = get_task(info["task"])
    env = task.make_env("rerender")
    step_s = info.get("args", {}).get("step_seconds") or None
    if step_s:
        env.setup.step_seconds = step_s
    replay = run_dir / "replays" / f"{args.episode}.w3g"
    episode = int("".join(ch for ch in args.episode.split("episode")[-1] if ch.isdigit()) or 0)
    overlay = BridgeServer._overlay(SimpleNamespace(task=task), replay, episode, run_dir)
    out = args.out or run_dir / "videos" / f"{args.episode}.mp4"
    render_replay(env.setup, replay, out, name="rerender", overlay=overlay)
    print(out, flush=True)


if __name__ == "__main__":
    main()
