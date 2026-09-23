"""Render real game footage of a replay to an MP4.

The replay is played back through the harness (GameInstance.play_replay, which feeds the
recorded agent orders back at the same steps); after every step the virtual display is grabbed
(xwd) and piped into ffmpeg, cropped to the game window. One frame per step, so a 0.25 s step
and 16 fps is 4x real time.

    render_replay(setup, "runs/x/ep12.w3g", "runs/x/ep12.mp4", follow_player=0)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .protocol import Camera, Observation
from .runtime.instance import GameError, GameInstance, GameSetup


def _window_geometry(display: str) -> tuple[int, int, int, int] | None:
    env = dict(os.environ, DISPLAY=display)
    ids = subprocess.run(["xdotool", "search", "--name", "Warcraft III"], capture_output=True, text=True,
                         env=env).stdout.split()
    for wid in ids:
        out = subprocess.run(["xdotool", "getwindowgeometry", wid], capture_output=True, text=True, env=env).stdout
        try:
            pos = out.split("Position:")[1].split("(")[0].strip()
            geo = out.split("Geometry:")[1].strip()
            x, y = (int(v) for v in pos.split(","))
            w, h = (int(v) for v in geo.split("x"))
        except (IndexError, ValueError):
            continue
        if w >= 320 and h >= 200:
            return x, y, w, h
    return None


def _follow_target(obs: Observation, player: int | None) -> tuple[float, float] | None:
    units = [u for u in obs.units if u.alive and not u.is_structure
             and (u.owner == player if player is not None else u.owner in obs.players)]
    if not units:
        return None
    return sum(u.x for u in units) / len(units), sum(u.y for u in units) / len(units)


def render_replay(setup: GameSetup, replay: str | os.PathLike, out: str | os.PathLike, fps: int = 16,
                  width: int = 640, follow_player: int | None = None, max_steps: int = 20000,
                  name: str = "render") -> Path:
    """Play `replay` (saved by GameInstance.save_replay with the same setup) and write an MP4."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    setup = GameSetup(**{**setup.__dict__, "warm_spare": False})
    inst = GameInstance(setup, name=name, timeout=30)
    ffmpeg = None
    frames = 0
    try:
        obs = inst.play_replay(replay)
        geo = _window_geometry(inst._display)
        vf = f"crop={geo[2]}:{geo[3]}:{geo[0]}:{geo[1]}," if geo else ""
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "xwd", "-framerate", str(fps),
             "-i", "-", "-vf", f"{vf}scale={width}:-2", "-c:v", "libx264", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE)
        for _ in range(max_steps):
            frame = subprocess.run(["xwd", "-root", "-silent", "-display", inst._display], capture_output=True,
                                   check=True).stdout
            ffmpeg.stdin.write(frame)
            frames += 1
            if obs.game_over:
                break
            target = _follow_target(obs, follow_player) if follow_player is not None or setup.scenario is None \
                else None
            try:
                obs = inst.step([Camera(*target)] if target else [])
            except GameError:
                break  # the replay ended
    finally:
        if ffmpeg:
            ffmpeg.stdin.close()
            ffmpeg.wait()
        inst.close()
    if frames == 0 or not out.exists():
        raise GameError(f"no frames rendered from {replay}")
    return out
