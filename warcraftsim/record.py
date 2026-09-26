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
from functools import lru_cache
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


@lru_cache(maxsize=8)  # the maps don't change: rendering many episodes loads each once
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


_HTML = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>warcraftsim trajectory</title>
<style>
:root{--bg:#0f1318;--panel:#171d24;--line:#2a3440;--text:#d7dee6;--muted:#8795a3;--accent:#e0a33a}
*{box-sizing:border-box}
body{font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;padding:8px;background:var(--bg);color:var(--text)}
#bar{display:flex;gap:10px;align-items:center;margin:0 0 8px;flex-wrap:wrap}
#bar button,#bar select{background:var(--panel);border:1px solid var(--line);border-radius:5px;color:var(--text);font:inherit;padding:2px 10px;cursor:pointer}
#t{flex:1;min-width:120px;accent-color:var(--accent)}
#lbl{color:var(--muted);font-variant-numeric:tabular-nums;min-width:118px}
label{color:var(--muted)}
canvas{display:block;max-width:100%;max-height:calc(100vh - 84px);width:auto;height:auto;margin:0 auto;background:#161c22;border:1px solid var(--line);border-radius:6px}
#info{display:flex;gap:18px;flex-wrap:wrap;margin:6px 2px 0;color:var(--muted);font-size:12px}
#info b{color:var(--text);font-weight:500}.sw{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
</style>
<div id="bar"><button id="play">play</button><input id="t" type="range" min="0" value="0"><span id="lbl"></span>
<label>speed <select id="spd"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option><option value="8">8×</option></select></label></div>
<canvas id="c" width="960" height="600"></canvas><div id="info"></div>
<script>
const D = __DATA__;
const frames = D.frames, T = D.terrain;
const colors = ["#e33","#36f","#2cb","#a3d","#ee3","#f92","#3c3","#e7b","#999","#8cf","#064","#720"];
const cv = document.getElementById("c"), cx = cv.getContext("2d"), sl = document.getElementById("t");
sl.max = frames.length - 1;
// the view: where the units were over the episode (not the whole map), with a margin
let minx=1e9,miny=1e9,maxx=-1e9,maxy=-1e9;
for (const f of frames) for (const u of f.u) { if (u[2] >= 12) continue; minx=Math.min(minx,u[3]); maxx=Math.max(maxx,u[3]);
  miny=Math.min(miny,u[4]); maxy=Math.max(maxy,u[4]); }
if (minx > maxx) { minx=-500; maxx=500; miny=-500; maxy=500; }
const pad = Math.max(250, 0.15 * Math.max(maxx-minx, maxy-miny)); minx-=pad; miny-=pad; maxx+=pad; maxy+=pad;
const sc = Math.min(cv.width/(maxx-minx), cv.height/(maxy-miny));
const ox = (cv.width - (maxx-minx)*sc) / 2, oy = (cv.height - (maxy-miny)*sc) / 2;  // centred
const X = x => ox + (x-minx)*sc, Y = y => cv.height - oy - (y-miny)*sc;
let bg = null;
if (T) { bg = document.createElement("canvas"); bg.width = cv.width; bg.height = cv.height;
  const g = bg.getContext("2d"), bytes = atob(T.bits); g.fillStyle = "#12171d"; g.fillRect(0,0,bg.width,bg.height);
  g.fillStyle = "#2b3036";
  for (let r=0;r<T.h;r++) for (let c=0;c<T.w;c++) { const i=r*T.w+c;
    if (bytes.charCodeAt(i>>3) & (128>>(i&7))) g.fillRect(X(T.x0+c*T.cell), Y(T.y0+(r+1)*T.cell), T.cell*sc+1, T.cell*sc+1); } }
const R = Math.max(5, Math.min(14, 20 * sc));  // a unit's size on screen: about its size in the game
function draw(i) {
  const f = frames[i]; cx.clearRect(0,0,cv.width,cv.height); if (bg) cx.drawImage(bg,0,0);
  cx.font = "11px system-ui, sans-serif";
  for (const u of f.u) { const [id,type,own,x,y,hp,mhp,fl] = u, s = (fl&2) ? R * 1.4 : (fl&1) ? R * 1.25 : R;
    cx.fillStyle = own < colors.length ? colors[own] : "#777"; cx.beginPath();
    if (fl&2) cx.rect(X(x)-s, Y(y)-s, 2*s, 2*s); else cx.arc(X(x), Y(y), s, 0, 7); cx.fill();
    if (fl&1) { cx.strokeStyle = "#fff"; cx.lineWidth = 1.5; cx.stroke(); }  // a hero
    if (mhp > 0 && own < 12) { cx.fillStyle="#000"; cx.fillRect(X(x)-s, Y(y)-s-6, 2*s, 3);
      cx.fillStyle = hp/mhp > .5 ? "#4cc38a" : hp/mhp > .25 ? "#e0a33a" : "#e5534b"; cx.fillRect(X(x)-s, Y(y)-s-6, 2*s*hp/mhp, 3); }
    if (own < 12) { cx.fillStyle = "#8795a3"; cx.fillText(type, X(x) - 12, Y(y) + s + 12); } }
  document.getElementById("lbl").textContent = (f.t/1000).toFixed(1) + " s · step " + i + "/" + (frames.length-1);
  document.getElementById("info").innerHTML = Object.entries(f.p).map(([p,v]) => {
    const n = f.u.filter(u => u[2] == p), hp = n.reduce((a, u) => a + u[5], 0);
    return `<span><span class="sw" style="background:${colors[p] || "#777"}"></span>player ${p}${p == 0 ? " (agent)" : ""}: <b>${n.length}</b> units, <b>${hp}</b> hp` +
      ["", " · <b style='color:#4cc38a'>victory</b>", " · <b style='color:#e5534b'>defeat</b>", " · tie"][v[4]] + "</span>"; }).join("");
}
// playback in game time: speed 2× plays a 20 s fight in 10 s
let timer = null, now = frames[0].t;
const btn = document.getElementById("play");
function stop() { clearInterval(timer); timer = null; btn.textContent = "play"; }
function play() {
  if (+sl.value >= frames.length - 1) sl.value = 0;
  now = frames[+sl.value].t; btn.textContent = "pause";
  timer = setInterval(() => {
    now += 50 * +document.getElementById("spd").value;
    let i = +sl.value; while (i < frames.length - 1 && frames[i + 1].t <= now) i++;
    if (i !== +sl.value) { sl.value = i; draw(i); }
    if (i >= frames.length - 1) stop();
  }, 50);
}
btn.onclick = () => timer ? stop() : play();
sl.oninput = () => { draw(+sl.value); now = frames[+sl.value].t; };
draw(0);
if (location.hash.includes("autoplay")) play();  // the dashboard's viewer
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
