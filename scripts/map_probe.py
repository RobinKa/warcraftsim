"""Whether a map's frame is right in the real game (fullgame/mapframe.py): a game on each map, a few steps without orders,
and where its bases, mines and halls land in the frame the policy reads; the mines' gold and the starting resources.

    python scripts/map_probe.py "(2)EchoIsles" "(2)LastRefuge-1.3" "(2)Amazonia" "(2)TerenasStand_LV" [--races human,orc]

The process's frame is Echo Isles', as a run's on it. No GPU, no network.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def walking(name: str, obs, frame) -> float:
    """The shortest walk between the map's two starts over its pathing grid (war3map.wpm, cells of 32) with the trees of
    the first observation standing (each blocks its cell and the 8 around it), by 8 neighbours. obs: in home coordinates."""
    import heapq

    import numpy as np

    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.data.terrain import load_terrain
    from warcraftsim.fullgame import mapframe
    t = load_terrain(stock_map_path(name))
    free = t.walkable.copy()
    rows, cols = free.shape
    counts = Counter(d.type for d in obs.destructables or ())
    for d in obs.destructables or ():
        if d.life <= 0 or d.type[1:2] != "T" or counts[d.type] < 20:  # (the trees, as heuristic.py reads them: not bridges)
            continue
        x, y = frame.to_world1(d.x, d.y) if frame else (d.x, d.y)
        r, c = t.cell(x, y)
        free[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = False
    (ax, ay), (bx, by) = sorted(mapframe.map_starts(name))[:2]
    src, dst = t.cell(ax, ay), t.cell(bx, by)
    dist = np.full(free.shape, np.inf)
    dist[src] = 0.0
    todo = [(0.0, *src)]
    steps = [(a, b, 32.0 * math.hypot(a, b)) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
    while todo:
        d0, r, c = heapq.heappop(todo)
        if (r, c) == dst:
            return d0
        if d0 > dist[r, c]:
            continue
        for a, b, w in steps:
            r2, c2 = r + a, c + b
            if 0 <= r2 < rows and 0 <= c2 < cols and (free[r2, c2] or math.hypot(r2 - src[0], c2 - src[1]) < 12
                                                      or math.hypot(r2 - dst[0], c2 - dst[1]) < 12) and d0 + w < dist[r2, c2]:
                dist[r2, c2] = d0 + w
                heapq.heappush(todo, (d0 + w, r2, c2))
    return float("inf")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--races", default="human,orc")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--name", default="mapprobe")
    ap.add_argument("--walk", type=int, default=0, help="1: also the walking distance between the bases")
    a = ap.parse_args(argv)
    from warcraftsim.data.mapbuild import stock_map_path
    from warcraftsim.fullgame import features as fx
    from warcraftsim.fullgame import macros, mapframe
    from warcraftsim.fullgame.play import unit_rows
    from warcraftsim.protocol import rawcode
    from warcraftsim.runtime import wine
    from warcraftsim.runtime.instance import Agent, GameInstance, GameSetup
    wine.use_wine(str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"), True, False)
    wine.ensure_template()
    fx.set_geometry(*fx.map_geometry(stock_map_path("(2)EchoIsles")))
    print(f"frame: starts {fx.STARTS}, bins over +-{fx.BIN_RANGE} ({fx.BINS})", flush=True)
    races = a.races.split(",")
    for k, name in enumerate(a.maps):
        setup = GameSetup(map=name, slots=[Agent(races[0], difficulty="normal"), Agent(races[1], difficulty="normal")],
                          step_seconds=0.5, max_game_seconds=600.0, victory="decisive", window=(320, 240), draw=False,
                          native_obs=True, nice=19, d3d_thread=False, render_threads=0)
        with GameInstance(setup, name=f"{a.name}{k}", timeout=300) as g0:
            g = mapframe.framed(g0, name)
            obs = g.start()
            frame = mapframe.frame_for(name)
            if a.walk:
                print(f"\n{name}: walking distance between the bases {walking(name, obs, frame):.0f}", flush=True)
            print(f"\n{name}: {'framed, turned %.1f deg' % frame.deg if frame else 'the home map, no frame'}", flush=True)
            for _ in range(a.steps):
                obs = g.step([])
            rows = unit_rows(obs, a.steps)
            mines = rows[fx.isin(rows[:, fx.C_TYPE], macros.MINE_CODES)]
            for s in (0, 1):
                p = obs.players[s]
                sign = -1.0 if p.start_x > fx.MAP_CX else 1.0
                own = rows[(rows[:, fx.C_OWNER] == s) & (rows[:, fx.C_FLAGS] & fx.STRUCTURE > 0)]
                hx, hy = fx.frame_xy(own[0, fx.C_X], own[0, fx.C_Y], sign)
                print(f" player {s} {races[s]}: start {p.start_x},{p.start_y}  sign {sign:+.0f}  gold {p.gold} lumber {p.lumber}  "
                      f"hall {rawcode(int(own[0, fx.C_TYPE]))} at {own[0, fx.C_X]},{own[0, fx.C_Y]} -> frame ({float(hx):.3f}, {float(hy):.3f})"
                      f"  own units {int((rows[:, fx.C_OWNER] == s).sum())}", flush=True)
                fe = fx.frame_xy(obs.players[1 - s].start_x, obs.players[1 - s].start_y, sign)
                print(f"   enemy's start in its frame ({float(fe[0]):.3f}, {float(fe[1]):.3f})", flush=True)
                nat = macros.natural(rows, (p.start_x, p.start_y))
                if nat is not None:
                    mx, my = float(nat[fx.C_X]), float(nat[fx.C_Y])
                    print(f"   natural's mine at {mx:.0f},{my:.0f}: {math.hypot(mx - p.start_x, my - p.start_y):.0f} from the start, gold "
                          f"{int(nat[fx.C_RESOURCE])}, hall spot {tuple(round(v) for v in macros.hall_spot(mx, my, p.start_x, p.start_y))}", flush=True)
            for r in mines:
                d = min(math.hypot(r[fx.C_X] - obs.players[s].start_x, r[fx.C_Y] - obs.players[s].start_y) for s in (0, 1))
                f0 = fx.frame_xy(r[fx.C_X], r[fx.C_Y], 1.0)
                print(f"   mine {r[fx.C_X]},{r[fx.C_Y]}: gold {int(r[fx.C_RESOURCE])}, {d:.0f} from a start, left's frame "
                      f"({float(f0[0]):.2f}, {float(f0[1]):.2f})", flush=True)
            print(f" trees {len(obs.destructables or [])}" if obs.destructables else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
