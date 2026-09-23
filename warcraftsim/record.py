"""Record observation streams and view them as an animated HTML page.

Game replays (.w3g) do not contain agent orders (see GameInstance.save_replay), so for watching
agents the observation stream itself is recorded:

    rec = TrajectoryRecorder("runs/ep1.jsonl", map_name="(2)EchoIsles")
    obs = game.reset(); rec.add(obs)
    while not obs.game_over:
        obs = game.step(...); rec.add(obs)
    rec.close()
    render_html("runs/ep1.jsonl", "runs/ep1.html")     # or: python -m warcraftsim view runs/ep1.jsonl

Each line of the .jsonl file is one observation: game time, players, living units and events.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .protocol import EventKind, Observation, rawcode


class TrajectoryRecorder:
    def __init__(self, path: str | Path, map_name: str | None = None, every: int = 1):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every = max(1, every)
        self._n = 0
        self._f = open(self.path, "w")
        self._f.write(json.dumps({"format": "warcraftsim-trajectory", "version": 1, "map": map_name}) + "\n")

    def add(self, obs: Observation) -> None:
        self._n += 1
        if (self._n - 1) % self.every and not obs.game_over:
            return
        rec = {
            "t": obs.game_ms,
            "p": {str(pid): [p.gold, p.lumber, p.food_used, p.food_cap, int(p.result)] for pid, p in obs.players.items()},
            "u": [[u.id, u.type, u.owner, u.x, u.y, u.hp, u.max_hp, int(u.flags)] for u in obs.units if u.alive],
            "d": [e.a for e in obs.events if e.kind == EventKind.DEATH],
        }
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "TrajectoryRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _terrain_layer(map_name: str | None) -> dict | None:
    """Walkable mask of the map, downsampled to terrain tiles (128 world units), base64 bits."""
    if not map_name:
        return None
    try:
        from .data.mapbuild import stock_map_path
        from .data.terrain import load_terrain

        t = load_terrain(stock_map_path(map_name))
    except Exception:
        return None
    walk = t.walkable
    h, w = walk.shape[0] // 4, walk.shape[1] // 4
    tiles = walk[:h * 4, :w * 4].reshape(h, 4, w, 4).mean(axis=(1, 3)) > 0.5
    return {"x0": t.offset_x, "y0": t.offset_y, "cell": 128, "w": w, "h": h,
            "bits": base64.b64encode(np.packbits(tiles.astype(np.uint8)).tobytes()).decode()}


_HTML = """<!doctype html><meta charset="utf-8"><title>warcraftsim trajectory</title>
<style>body{font:13px sans-serif;margin:12px;background:#111;color:#ddd}canvas{background:#222;border:1px solid #444}
#bar{display:flex;gap:8px;align-items:center;margin:6px 0}#t{width:600px}pre{margin:4px 0}</style>
<div id="bar"><button id="play">play</button><input id="t" type="range" min="0" value="0">
<span id="lbl"></span><label>speed <select id="spd"><option>1</option><option selected>4</option><option>16</option>
</select></label></div><canvas id="c" width="900" height="680"></canvas><pre id="info"></pre>
<script>
const D = __DATA__;
const frames = D.frames, T = D.terrain;
const colors = ["#e33","#36f","#2cb","#a3d","#ee3","#f92","#3c3","#e7b","#999","#8cf","#064","#720"];
const cv = document.getElementById("c"), cx = cv.getContext("2d"), sl = document.getElementById("t");
sl.max = frames.length - 1;
let minx=1e9,miny=1e9,maxx=-1e9,maxy=-1e9;
if (T) { minx=T.x0; miny=T.y0; maxx=T.x0+T.w*T.cell; maxy=T.y0+T.h*T.cell; }
else for (const f of frames) for (const u of f.u) { minx=Math.min(minx,u[3]); maxx=Math.max(maxx,u[3]);
  miny=Math.min(miny,u[4]); maxy=Math.max(maxy,u[4]); }
const pad = T ? 0 : 300; minx-=pad; miny-=pad; maxx+=pad; maxy+=pad;
const sc = Math.min(cv.width/(maxx-minx), cv.height/(maxy-miny));
const X = x => (x-minx)*sc, Y = y => cv.height-(y-miny)*sc;
let bg = null;
if (T) { bg = document.createElement("canvas"); bg.width = cv.width; bg.height = cv.height;
  const g = bg.getContext("2d"), bytes = atob(T.bits); g.fillStyle = "#16202a"; g.fillRect(0,0,bg.width,bg.height);
  g.fillStyle = "#3a3f2f";
  for (let r=0;r<T.h;r++) for (let c=0;c<T.w;c++) { const i=r*T.w+c;
    if (bytes.charCodeAt(i>>3) & (128>>(i&7))) g.fillRect(X(T.x0+c*T.cell), Y(T.y0+(r+1)*T.cell), T.cell*sc+1, T.cell*sc+1); } }
function draw(i) {
  const f = frames[i]; cx.clearRect(0,0,cv.width,cv.height); if (bg) cx.drawImage(bg,0,0);
  for (const u of f.u) { const [id,type,own,x,y,hp,mhp,fl] = u, s = (fl&2) ? 7 : (fl&1) ? 6 : 4;
    cx.fillStyle = own < colors.length ? colors[own] : "#777"; cx.beginPath();
    if (fl&2) cx.rect(X(x)-s, Y(y)-s, 2*s, 2*s); else cx.arc(X(x), Y(y), s, 0, 7); cx.fill();
    if (mhp > 0 && own < 12) { cx.fillStyle="#000"; cx.fillRect(X(x)-s, Y(y)-s-4, 2*s, 2);
      cx.fillStyle="#4f4"; cx.fillRect(X(x)-s, Y(y)-s-4, 2*s*hp/mhp, 2); } }
  for (const id of f.d) {}
  document.getElementById("lbl").textContent = "t=" + (f.t/1000).toFixed(1) + "s  step " + i + "/" + (frames.length-1);
  document.getElementById("info").textContent = Object.entries(f.p).map(([p,v]) =>
    "player " + p + ": gold " + v[0] + " lumber " + v[1] + " food " + v[2] + "/" + v[3] +
    ["", "  VICTORY", "  DEFEAT", "  TIE"][v[4]] + "  units " + f.u.filter(u => u[2] == p).length).join("\\n");
}
let timer = null;
document.getElementById("play").onclick = () => { if (timer) { clearInterval(timer); timer = null; return; }
  timer = setInterval(() => { const n = +sl.value + +document.getElementById("spd").value;
    if (n >= frames.length) { clearInterval(timer); timer = null; } sl.value = Math.min(n, frames.length-1); draw(+sl.value); }, 50); };
sl.oninput = () => draw(+sl.value);
draw(0);
</script>
"""


def render_html(trajectory: str | Path, out: str | Path | None = None) -> Path:
    trajectory = Path(trajectory)
    lines = trajectory.read_text().splitlines()
    header = json.loads(lines[0])
    frames = [json.loads(line) for line in lines[1:] if line.strip()]
    data = {"frames": frames, "terrain": _terrain_layer(header.get("map"))}
    out = Path(out) if out else trajectory.with_suffix(".html")
    out.write_text(_HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    return out
