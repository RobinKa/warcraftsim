"""Play a whole game against a policy yourself: the game in a window on your desktop, you one player, the
policy the other, the game's clock at real time. The dashboard's Challenge button starts it (--record: the game's
replay saved with the policy's orders and rendered into a video with its panel, the result in challenges.jsonl).

    python3 -m warcraftsim.fullgame.versus runs/fgself-10/checkpoints/<steps>.pt --you human --agent orc
    python3 -m warcraftsim.fullgame.versus <checkpoint> --you orc --agent random --map duelfast

The policy plays as in training (the rules of --map, the units at the handicap's hit points, its
orders every half second of game time); it sees what its player sees. The training maps' rules are
fast: on duelrush a game lasts about 2 minutes and everything but walking runs 7 times faster, on
duelfast production takes a third of the time. Close the game's window, or Ctrl+C, to stop.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from ..runtime.instance import Agent, GameInstance, GameSetup, Human
from . import features as fx
from .collect import Films
from .costs import order_costs
from .model import load
from .play import play_game
from .trace import unit_values


def keep_left_replay(g: GameInstance, dest: Path, trace: dict | None, since: float, wait: float = 15.0) -> Path | None:
    """A game the person closed: the replay the client wrote on leaving (LastReplay.w3g), the policy's orders beside
    it (GameInstance.save_replay's commands.json: they are not in the .w3g) and the panel's trace -> its path."""
    from ..runtime.instance import commands_path
    src = g.replay_dir() / "LastReplay.w3g"
    deadline = time.time() + wait
    while time.time() < deadline and not (src.exists() and src.stat().st_mtime > since):
        time.sleep(0.5)
    if not src.exists() or src.stat().st_mtime <= since:
        print("the client wrote no replay", flush=True)
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    time.sleep(0.5)  # (let the writer finish)
    dest.write_bytes(src.read_bytes())
    commands_path(dest).write_text(json.dumps({"format": "warcraftsim-commands", "version": 1,
                                               "map_key": getattr(g, "_map_key", None), "commands": dict(g._cmd_log)}))
    if trace is not None:
        dest.with_suffix(".trace.json").write_text(json.dumps(trace))
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path, help="a self-play checkpoint (runs/<run>/checkpoints/*.pt) or a fit's policy.pt")
    ap.add_argument("--you", default="human", help="your race (human, orc, undead, nightelf, random)")
    ap.add_argument("--agent", default="random", help="the policy's race")
    ap.add_argument("--map", default="duelrush", help="duelrush (as the policy trained), duelfast, duel")
    ap.add_argument("--side", type=int, choices=(0, 1), help="your start: 0 the left base, 1 the right (default: random)")
    ap.add_argument("--display", default=os.environ.get("DISPLAY") or ":0",
                    help="where the game's window opens ('none': a hidden screen, for a test)")
    ap.add_argument("--window", default="1280x960", help="the game's resolution")
    ap.add_argument("--speed", type=float, default=1.0, help="the game's clock: 1 real time, 0.5 half as fast")
    ap.add_argument("--handicap", type=int, help="both players' units' hit points in percent (default: 100, on the "
                    "duel maps 50 as in their training)")
    ap.add_argument("--record", type=Path, help="a folder: the game's replay and video (with the policy's panel) and "
                    "its result (challenges.jsonl); status.json while it runs")
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cpu", help="where the policy runs (a GPU is not needed for one game)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--wine", default="", help="another Wine's bin folder (e.g. GE-Proton's files/bin, as the "
                    "self-play runs use; runtime.wine.use_wine)")
    ap.add_argument("--fsync", type=int, default=1, help="with --wine (a Proton build): futex-based synchronization")
    args = ap.parse_args(argv)
    if args.wine:  # (before any game: the games read the environment)
        from ..runtime import wine
        wine.use_wine(args.wine, bool(args.fsync))
    rng = random.Random(args.seed)
    pick = (lambda r: rng.choice(fx.RACES) if r == "random" else r)  # noqa: E731
    you, agent = pick(args.you), pick(args.agent)
    side = args.side if args.side is not None else rng.randrange(2)
    device = torch.device(args.device)
    from .model import configure_for
    configure_for(args.checkpoint)  # (its inputs, masks, strategy block: as it trained; no target given, zeros)
    net, ck = load(args.checkpoint, device)
    from ..data.duelmap import parse_duel_name
    if args.handicap is None:
        args.handicap = 50 if parse_duel_name(args.map) is not None else 100
    if parse_duel_name(args.map) is None:  # (another map's frame, as selfplay's: Echo Isles' starts and point bins)
        from ..data.mapbuild import stock_map_path
        fx.set_geometry(*fx.map_geometry(stock_map_path(args.map)))
    if net.bins != fx.BINS:
        raise SystemExit(f"{args.checkpoint} has {net.bins} point bins, {args.map}'s frame {fx.BINS}: another map's policy")
    vocab = ck["vocab"]
    w, h = (int(v) for v in args.window.lower().split("x"))
    slots = [Human(you, handicap=args.handicap), Agent(agent, handicap=args.handicap)]
    if side == 1:
        slots.reverse()
    setup = GameSetup(map=args.map, slots=slots, step_seconds=0.5, max_game_seconds=args.max_minutes * 60,
                      victory="decisive", window=(w, h), speed=args.speed, warm_spare=False)
    print(f"you: {you} ({'left' if side == 0 else 'right'} base) · the policy: {agent} · {args.map} · "
          f"{args.checkpoint}", flush=True)
    display = None if args.display.lower() in ("", "none") else args.display
    rec = args.record
    stem = f"challenge-{int(time.time())}"
    video = f"videos/{stem}.mp4" if rec is not None else None  # (none when the replay could not be kept)
    status = {"stem": stem, "checkpoint": str(args.checkpoint), "you": you, "agent": agent, "side": side,
              "map": args.map, "speed": args.speed, "pid": os.getpid(), "started": time.time()}

    def phase(p: str, **more) -> None:
        if rec is not None:
            rec.mkdir(parents=True, exist_ok=True)
            tmp = rec / "status.json.tmp"
            tmp.write_text(json.dumps({**status, "phase": p, "time": time.time(), **more}))
            tmp.replace(rec / "status.json")
    phase("starting the game")
    films = Films(rec, 0.0, None, "versus_video") if rec is not None else None
    values = unit_values() if rec is not None else None
    with GameInstance(setup, name="versus", display=display, timeout=600) as g:
        obs = g.start()
        phase("playing")
        r = play_game(g, obs, net, vocab, device, 1 - side, args.temperature, costs=order_costs(vocab, args.map),
                      values=values, left_ok=True)
        trace = r.pop("trace", None)
        you_, it = r["sides"].get("ai", {}), r["sides"].get("agent", {})  # (play_game calls the other side "ai": you)
        result = {"VICTORY": "the policy won", "DEFEAT": "you won", "LEFT": "you left"}.get(r["outcome"], "a tie")
        if r["outcome"] == "LEFT" and trace and trace["steps"]:  # (who was ahead then: the material, as the panel shows)
            m = trace["steps"][-1].get("material") or {}
            mine, its = m.get(str(side)), m.get(str(1 - side))
            if mine is not None and its is not None:
                result += f" ({'ahead' if mine > its else 'behind'}: material {round(mine)} to the policy's {round(its)})"
        if films is not None:
            phase("saving the replay", result=result)
            if trace is not None:
                trace["title"] = f"you ({you}) against {args.checkpoint.parent.parent.name} ({agent})"[:70]
                trace["sides"] = [{"player": 1 - side, "name": f"the policy ({agent})", "kind": "agent"},
                                  {"player": side, "name": f"you ({you})", "kind": "human"}]
            row = {"episode": 0, "title": f"you ({you}) vs the policy ({agent})",
                   "outcome": {"VICTORY": -1.0, "DEFEAT": 1.0}.get(r["outcome"], 0.0),
                   "sub": f"{result} after {r['minutes']:.1f} game minutes"}
            if r["outcome"] == "LEFT":  # (the game is closed: the replay the client wrote on leaving, the orders kept)
                kept = keep_left_replay(g, rec / "replays" / f"{stem}.w3g", trace, status["started"])
                if kept is not None:
                    films.pool.submit(films._render, g.setup, kept, stem, row, trace)
                else:
                    video = None
            else:
                films.film(g, stem, trace=trace, row=row)
    print(f"{result} after {r['minutes']} game minutes · gold gathered: you {you_.get('gold')}, the policy "
          f"{it.get('gold')} · food: you {you_.get('food')}, the policy {it.get('food')}", flush=True)
    if rec is not None:
        row = {**status, "time": time.time(), "result": result, "minutes": r["minutes"], "you_stats": you_,
               "policy_stats": it, "video": video}
        with open(rec / "challenges.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        phase("rendering the video", result=result)
        films.wait()
        phase("done", result=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
