"""Render real game footage of a replay to an MP4.

The replay is played back through the harness (GameInstance.play_replay, which feeds the
recorded agent orders back at the same steps) with frame capture on: the virtual clock advances
a fixed game time per rendered frame, and after every frame the game window is grabbed from the
virtual display (XGetImage) and piped into ffmpeg. So the video has one image per game frame at
a steady frame rate (40 fps real time by default), independent of machine load.

    render_replay(setup, "runs/x/ep12.w3g", "runs/x/ep12.mp4", follow_player=0)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
from pathlib import Path

from .protocol import Camera, Observation, decode_commands
from .runtime.display import Xvfb
from .runtime.instance import GameError, GameInstance, GameSetup


class _XImage(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("xoffset", ctypes.c_int),
                ("format", ctypes.c_int), ("data", ctypes.c_void_p), ("byte_order", ctypes.c_int),
                ("bitmap_unit", ctypes.c_int), ("bitmap_bit_order", ctypes.c_int), ("bitmap_pad", ctypes.c_int),
                ("depth", ctypes.c_int), ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int)]


class XGrabber:
    """Grab a rectangle of an X display as BGRX bytes (XGetImage: a few ms, no process per frame)."""

    def __init__(self, display: str, x: int, y: int, w: int, h: int):
        x11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x11.XGetImage.restype = ctypes.POINTER(_XImage)
        x11.XGetImage.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint,
                                  ctypes.c_uint, ctypes.c_ulong, ctypes.c_int]
        x11.XDestroyImage.argtypes = [ctypes.POINTER(_XImage)]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._x11 = x11
        self._dpy = x11.XOpenDisplay(display.encode())
        if not self._dpy:
            raise GameError(f"cannot open X display {display}")
        self._root = x11.XDefaultRootWindow(self._dpy)
        self.rect = (x, y, w, h)

    def grab(self) -> bytes:
        x, y, w, h = self.rect
        img = self._x11.XGetImage(self._dpy, self._root, x, y, w, h, 0xFFFFFFFF, 2)  # AllPlanes, ZPixmap
        if not img:
            raise GameError("XGetImage failed")
        try:
            im = img.contents
            if im.bits_per_pixel != 32:
                raise GameError(f"unsupported X image format ({im.bits_per_pixel} bpp)")
            data = ctypes.string_at(im.data, im.bytes_per_line * h)
            if im.bytes_per_line != w * 4:  # drop row padding
                data = b"".join(data[r * im.bytes_per_line:r * im.bytes_per_line + w * 4] for r in range(h))
            return data
        finally:
            self._x11.XDestroyImage(img)

    def close(self) -> None:
        if self._dpy:
            self._x11.XCloseDisplay(self._dpy)
            self._dpy = None


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


def _park_pointer(display: str) -> None:
    """Move the mouse pointer to the screen's bottom right corner, off the game window."""
    env = dict(os.environ, DISPLAY=display)
    out = subprocess.run(["xdotool", "getdisplaygeometry"], capture_output=True, text=True, env=env).stdout.split()
    if len(out) == 2:
        subprocess.run(["xdotool", "mousemove", str(int(out[0]) - 1), str(int(out[1]) - 1)], env=env, check=False)


def _follow_target(obs: Observation, player: int | None) -> tuple[float, float] | None:
    units = [u for u in obs.units if u.alive and not u.is_structure
             and (u.owner == player if player is not None else u.owner in obs.players)]
    if not units:
        return None
    return sum(u.x for u in units) / len(units), sum(u.y for u in units) / len(units)


def render_replay(setup: GameSetup, replay: str | os.PathLike, out: str | os.PathLike, fps: int = 40,
                  speed: float = 1.0, width: int | None = None, follow_player: int | None = None,
                  max_steps: int = 20000, name: str = "render", crf: int = 23, overlay=None, audio: bool = True,
                  music_volume: int = 40) -> Path:
    """Play `replay` (saved by GameInstance.save_replay with the same setup) and write an MP4 at
    `fps`, `speed` times real time. Each frame advances the game by exactly 1000*speed/fps ms,
    ideally a multiple of the engine's 25 ms turn (40 fps at 1x, 40 fps at 2x, 60 fps at 1.5x).
    `overlay` (overlay.EpisodeOverlay) draws the agent's orders and what the policy thought.
    With `audio`, the game's sound comes from the w3shim virtual sound card, one frame's worth per
    frame; the render then runs at no more than real time (the game's mixer needs that)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # a real-time clock while the map loads: a fast one leaves a backlog that the engine simulates
    # in one frame at the start (the first ~4 s of game time would never be rendered)
    setup = GameSetup(**{**setup.__dict__, "warm_spare": False, "speed": 1.0, "window": (1024, 768),
                         "audio": audio, "music_volume": music_volume})
    # the pointer must be off the game window from the start: replays show a label on the unit under it
    xvfb = Xvfb(*setup.window)
    _park_pointer(xvfb.display)
    inst = GameInstance(setup, name=name, timeout=60, display=xvfb.display)
    ffmpeg = grabber = None
    frames = 0
    silent = out.with_name(out.stem + ".video.mp4") if audio else out
    pcm_path = out.with_name(out.stem + ".pcm")
    pcm_file = open(pcm_path, "wb") if audio else None
    audio_fmt: list = []  # rate, channels, bits, and Miles's latency per frame (bytes)
    try:
        obs = inst.play_replay(replay)
        geo = _window_geometry(inst._display)
        if geo is None:
            raise GameError("game window not found")
        grabber = XGrabber(inst._display, *geo)
        size, pix = (geo[2], geo[3]), "bgr0"
        if overlay is not None:
            overlay.begin(setup, inst.order_names, geo[2], geo[3])
            size, pix = overlay.size, "rgb24"
        scale = f"scale={width}:-2" if width else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{size[0]}x{size[1]}",
             "-framerate", str(fps), "-i", "-", "-vf", scale, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent)],
            stdin=subprocess.PIPE)
        pending: list[bytes] = []

        def write(batch: list[bytes]) -> None:
            nonlocal frames
            for f in batch:
                ffmpeg.stdin.write(f)
            frames += len(batch)

        def on_audio(pcm: bytes, rate: int, channels: int, bits: int, latency: int) -> None:
            pcm_file.write(pcm)
            audio_fmt.append((rate, channels, bits, latency))

        inst.set_frame_capture(1000.0 * speed / fps, lambda n: pending.append(grabber.grab()),
                               on_audio if audio else None)
        follow = follow_player is not None or setup.scenario is None
        last = None
        for t in range(max_steps):
            if obs.game_over:
                break
            commands = decode_commands((inst._playback or {}).get(f"{inst._proc_episode}:{obs.seq}", []))
            target = _follow_target(obs, follow_player) if follow else None
            before = obs
            try:
                obs = inst.step([Camera(*target)] if target else [])
            except GameError:
                obs = None  # the replay ended
            batch, pending[:] = list(pending), []
            if batch:
                last = batch[-1]
                write(overlay.render_step(batch, t, before, obs, commands) if overlay is not None else batch)
            if obs is None:
                break
        if overlay is not None and last is not None:
            write(overlay.render_end(last, fps))
    finally:
        if ffmpeg:
            ffmpeg.stdin.close()
            ffmpeg.wait()
        if grabber:
            grabber.close()
        if pcm_file:
            pcm_file.close()
        inst.close()
        xvfb.close()
    if frames == 0 or not silent.exists():
        raise GameError(f"no frames rendered from {replay}")
    if audio:
        _mux_audio(silent, pcm_path, audio_fmt, out, speed, frames / fps)
    return out


def _mux_audio(video: Path, pcm: Path, fmt: list, out: Path, speed: float, duration: float) -> None:
    """Add the captured game audio to `video`. The game's mixer queues its output ahead, so a sound
    starting at a frame is `latency` bytes later in the stream: shift the stream earlier by that."""
    try:
        if not fmt or pcm.stat().st_size == 0:
            video.replace(out)  # no audio device in the game: keep the silent video
            return
        rate, channels, bits, _ = fmt[-1]
        block = channels * bits // 8
        latencies = sorted(f[3] for f in fmt[len(fmt) // 4:] or fmt)  # after the start-up
        skip = latencies[len(latencies) // 2] // block * block
        codec = {8: "u8", 16: "s16le"}[bits]
        af = f"atrim=start_sample={skip // block},asetpts=N/SR/TB"
        if speed != 1.0:
            af += f",atempo={speed}"
        af += f",apad=whole_dur={duration:.3f}"  # silence under the end card
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-f", codec, "-ar", str(rate),
                        "-ac", str(channels), "-i", str(pcm), "-af", af, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-t", f"{duration:.3f}",
                        "-movflags", "+faststart", str(out)], check=True, timeout=600)
        video.unlink()
    finally:
        pcm.unlink(missing_ok=True)
