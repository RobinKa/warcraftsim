"""Play a whole game against a policy yourself: the game in a window on your desktop (WSLg's
display), you one player, the policy the other, the game's clock at real time.

    python3 -m warcraftsim.fullgame.versus runs/fgself-10/checkpoints/<steps>.pt --you human --agent orc
    python3 -m warcraftsim.fullgame.versus <checkpoint> --you orc --agent random --map duelfast

The policy plays as in training (the rules of --map, the units at the handicap's hit points, its
orders every half second of game time); it sees what its player sees. The training maps' rules are
fast: on duelrush a game lasts about 2 minutes and everything but walking runs 7 times faster, on
duelfast production takes a third of the time. Close the game's window, or Ctrl+C, to stop.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch

from ..runtime.instance import Agent, GameInstance, GameSetup, Human
from . import features as fx
from .costs import order_costs
from .model import load
from .play import play_game


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
    ap.add_argument("--handicap", type=int, default=50, help="both players' units' hit points in percent (as in training)")
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
    net, ck = load(args.checkpoint, device)
    fx.set_z(net.config.get("z", 0))  # (a strategy-conditioned network reads a strategy block: none given, zeros)
    from ..data.duelmap import parse_duel_name
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
    with GameInstance(setup, name="versus", display=display, timeout=600) as g:
        obs = g.start()
        r = play_game(g, obs, net, vocab, device, 1 - side, args.temperature, costs=order_costs(vocab, args.map))
    you_, it = r["sides"].get("ai", {}), r["sides"].get("agent", {})  # (play_game calls the other side "ai": you)
    result = {"VICTORY": "the policy won", "DEFEAT": "you won"}.get(r["outcome"], "a tie")
    print(f"{result} after {r['minutes']} game minutes · gold gathered: you {you_.get('gold')}, the policy "
          f"{it.get('gold')} · food: you {you_.get('food')}, the policy {it.get('food')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
