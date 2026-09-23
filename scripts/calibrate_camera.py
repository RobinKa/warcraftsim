"""Measure the ground-plane -> screen mapping of the default game camera (for video overlays).

Spawns flat markers (Circle of Power) one at a time at known points around the camera target,
finds each on screen by differencing frames, and fits a homography. Prints the constants for
warcraftsim/overlay.py.

    python scripts/calibrate_camera.py [--save debug.png]
"""

from __future__ import annotations

import argparse

import numpy as np

from warcraftsim.protocol import Spawn
from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup, Idle
from warcraftsim.scenario import Scenario
from warcraftsim.video import XGrabber, _park_pointer, _window_geometry
from warcraftsim.runtime.display import Xvfb


def _components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """8-connected components of a boolean mask as (xs, ys) pixel arrays."""
    seen = np.zeros_like(mask)
    h, w = mask.shape
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, xs, ys = [(y0, x0)], [], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            xs.append(x)
            ys.append(y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        out.append((np.array(xs), np.array(ys)))
    return out


def _merge_close(blobs: list, dist: float) -> list:
    """Merge blobs whose centroids are closer than `dist` px (a marker can split into pieces)."""
    blobs = list(blobs)
    merged = True
    while merged:
        merged = False
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                (xi, yi), (xj, yj) = blobs[i], blobs[j]
                if np.hypot(xi.mean() - xj.mean(), yi.mean() - yj.mean()) < dist:
                    blobs[i] = (np.r_[xi, xj], np.r_[yi, yj])
                    del blobs[j]
                    merged = True
                    break
            if merged:
                break
    return blobs


def fit_homography(world: np.ndarray, screen: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(world, screen):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows, float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", help="write the last frame with the fitted grid drawn on it")
    args = ap.parse_args()

    # two idle units far off screen; the camera starts at the scenario center (0, 0)
    sc = Scenario.skirmish(["hfoo"], ["hfoo"], separation=3200)
    setup = GameSetup(slots=[Agent("human"), Idle("orc")], scenario=sc, warm_spare=False, speed=1.0)
    xvfb = Xvfb()
    _park_pointer(xvfb.display)
    inst = GameInstance(setup, name="calibrate", display=xvfb.display)
    try:
        inst.start()
        geo = _window_geometry(xvfb.display)
        grab = XGrabber(xvfb.display, *geo)
        w, h = geo[2], geo[3]

        def frame() -> np.ndarray:
            return np.frombuffer(grab.grab(), np.uint8).reshape(h, w, 4)[:, :, :3].astype(np.int16)

        for _ in range(8):
            inst.step([])
        clean = frame()
        # all markers at once (they animate, so they cannot be told apart by frame differences over
        # time); blobs are matched to grid points by row (screen y) and column (screen x) order
        rows_y, cols_x = (-256, 0, 256, 512), (-512, -256, 0, 256, 512)
        inst.step([Spawn(15, "ncop", x, y) for y in rows_y for x in cols_x])
        for _ in range(4):
            inst.step([])
        mask = np.abs(frame() - clean).sum(axis=2) > 60
        mask[: int(h * 0.05)] = False  # top bar
        mask[int(h * 0.72):] = False  # bottom panel, minimap
        blobs = _merge_close([b for b in _components(mask) if len(b[0]) >= 10], 18.0)
        blobs = sorted((b for b in blobs if len(b[0]) >= 30), key=lambda b: b[1].mean())
        print(f"{len(blobs)} markers (expected {len(rows_y) * len(cols_x)})")
        if len(blobs) != len(rows_y) * len(cols_x):
            if args.save:
                from PIL import Image

                Image.fromarray((mask * 255).astype(np.uint8)).save(args.save)
            raise SystemExit("marker count mismatch; the mask is in the --save image")
        world, screen = [], []
        for r, y in enumerate(rows_y[::-1]):  # far rows are higher on screen
            row = sorted(blobs[r * len(cols_x):(r + 1) * len(cols_x)], key=lambda b: b[0].mean())
            for x, (xs, ys) in zip(cols_x, row):
                world.append((x, y))
                screen.append((xs.mean() / w, ys.mean() / h))
                print(f"({x}, {y}) -> ({xs.mean():.1f}, {ys.mean():.1f}) px")
        world_a, screen_a = np.asarray(world, float), np.asarray(screen, float)
        hom = fit_homography(world_a, screen_a)
        err = np.abs(project(hom, world_a) - screen_a) * [w, h]
        print(f"window {w}x{h}; reprojection error: mean {err.mean():.2f} px, max {err.max():.2f} px")
        print("GROUND_TO_SCREEN = (")
        for r in hom:
            print(f"    ({r[0]:.9g}, {r[1]:.9g}, {r[2]:.9g}),")
        print(")")
        if args.save:
            from PIL import Image, ImageDraw

            img = Image.frombuffer("RGB", (w, h), grab.grab(), "raw", "BGRX")
            d = ImageDraw.Draw(img)
            for (x, y), (u, v) in zip(world, screen):
                d.ellipse([u * w - 3, v * h - 3, u * w + 3, v * h + 3], outline=(255, 0, 255))
            for gx in range(-1024, 1025, 256):
                pts = project(hom, np.array([[gx, -512], [gx, 768]], float)) * [w, h]
                d.line([tuple(pts[0]), tuple(pts[1])], fill=(0, 255, 0))
            for gy in range(-512, 769, 256):
                pts = project(hom, np.array([[-1024, gy], [1024, gy]], float)) * [w, h]
                d.line([tuple(pts[0]), tuple(pts[1])], fill=(0, 255, 0))
            img.save(args.save)
        grab.close()
    finally:
        inst.close()
        xvfb.close()


if __name__ == "__main__":
    main()
