"""Open-loop drift of an approximation (shim/approx.c): a replay played back with the exact cuts and with each variant
side by side; per variant, the first step whose observation differs from the exact game's and, at given game times,
how far the units present in both games have moved apart (by the units' ids: the same unit in both games).

    python scripts/approx_drift.py --replay runs/fgself-154/replays/game-....w3g --base 17592186041849 \\
        --variant slow:35184372086265 --variant all:140737488352761 [--at 30,60,120] [--json out.json]

Open loop: the replay's commands are applied as recorded whatever the game became, so the games drift apart by
construction (an agent's later orders no longer fit); this says how soon and how far, not what training would see.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cut_ab  # noqa: E402  (setup_for, launch, digest; sets the display variables)


def play(setup, replay: Path, name: str, cuts: int, steps: int, out: dict) -> None:
    g = None
    try:
        g, obs = cut_ab.launch(setup, name, {"W3SIM_CUTS": str(cuts)}, None, replay)
        rows = []
        for t in range(steps):
            rows.append((cut_ab.digest(obs), {u.id: (u.x, u.y, u.owner, u.hp) for u in obs.units}))
            if obs.game_over:
                break
            obs = g.step([])
        out["rows"] = rows
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        if g is not None:
            g.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", required=True)
    ap.add_argument("--base", type=int, default=17592186041849)
    ap.add_argument("--variant", action="append", required=True, help="name:cuts")
    ap.add_argument("--at", default="30,60,120", help="game seconds to measure the drift at")
    ap.add_argument("--tag", default="D")
    ap.add_argument("--json")
    ap.add_argument("--wine", default=str(Path.home() / "wc3/ge/GE-Proton10-34/files/bin"))
    a = ap.parse_args(argv)
    from warcraftsim.runtime import wine
    wine.use_wine(a.wine, True)
    wine.ensure_template()
    replay = Path(a.replay).resolve()
    setup, _ = cut_ab.setup_for(replay)
    at = [float(x) for x in a.at.split(",")]
    steps = int(max(at) / setup.step_seconds) + 2
    variants = [("base", a.base)] + [(v.split(":")[0], int(v.split(":")[1])) for v in a.variant]
    outs = [{} for _ in variants]
    ts = [threading.Thread(target=play, args=(setup, replay, f"cutprobe{a.tag}_{i}_0", c, steps, outs[i]))
          for i, (_, c) in enumerate(variants)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    base = outs[0].get("rows") or []
    if not base:
        print("base:", outs[0].get("error") or "no steps", flush=True)
        return 1
    res = {"replay": str(replay), "step_seconds": setup.step_seconds, "variants": {}}
    for (name, cuts), o in zip(variants[1:], outs[1:]):
        rows = o.get("rows") or []
        r = {"cuts": cuts, "error": o.get("error")}
        n = min(len(rows), len(base))
        r["first_differ_step"] = next((t for t in range(n) if rows[t][0] != base[t][0]), None)
        for s in at:
            t = int(s / setup.step_seconds)
            if t >= n:
                continue
            ub, uv = base[t][1], rows[t][1]
            both = [k for k in ub if k in uv]
            d = [((ub[k][0] - uv[k][0]) ** 2 + (ub[k][1] - uv[k][1]) ** 2) ** 0.5 for k in both]
            moved = [x for x in d if x > 0]
            r[f"t{int(s)}"] = {"units": len(ub), "only_one": len(set(ub) ^ set(uv)), "differ": len(moved),
                               "mean": round(statistics.fmean(d), 1) if d else 0.0,
                               "median_of_moved": round(statistics.median(moved), 1) if moved else 0.0,
                               "max": round(max(d), 1) if d else 0.0,
                               "hp_differ": sum(ub[k][3] != uv[k][3] for k in both)}
        res["variants"][name] = r
        print(name, json.dumps(r), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
