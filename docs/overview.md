# Overview

warcraftsim trains agents on the **real Warcraft III** (Legacy TFT 1.29.2, Blizzard's offline client) running headless under Wine in WSL2, stepped from Python. Nothing is reimplemented: every game is the game itself, with a JASS harness injected into the map and a small DLL (the shim) that runs the game's clock and passes observations and orders over a socket.

## Where things stand (October 2026)

| track | best result | where |
|---|---|---|
| Micro (unit fights on a flat map) | 91% against the scripted opponent with general orders; league self-play 82% and 79% head-to-head against the 91% policy | [experiments](experiments.md), `--trainer torch` |
| Whole game, `duelrush` (2-minute games, everything but walking 7× faster) | **79% in the real game** against the built-in AI (easy and normal, all 16 race matchups; night elf 98%, orc 89%, undead 77%, human 57%) | `fgself-10` |
| Whole game, `duelfast` (the game's speed, production times ÷3, half costs and hit points; ~6-minute games) | 1 win in 234 real games; ties at the time limit and a policy that does not spend | `fgself-12` (running) |
| The real game (Echo Isles, normal rules) | not started | [road to the real game](road-to-the-real-game.md) |

## How a whole-game agent is made

1. **Demonstrations**: the built-in AI against itself on a duel map (`fullgame/collect.py`), every step's state and the orders it gave.
2. **Behaviour cloning** (`fullgame/bc.py`): a transformer over the units that gives each own unit an order and a target.
3. **Takeover games** (DAgger): the clone plays one side for a while, then the built-in AI takes over; its orders from the clone's own states are new demonstrations.
4. **Self-play** (`fullgame/selfplay.py`, after AlphaStar): PPO from the clone, a KL term towards it, the cloning loss on the takeover games next to PPO, a league (itself, past snapshots, the built-in AI with a curriculum, optionally a main exploiter).

## The documents

* [Environments](environments.md): the maps and their rules, the tasks, the opponents, what the agent sees and does, the rewards.
* [The road to the real game](road-to-the-real-game.md): what is still missing between the duel maps and a real 1v1.
* [Experiments](experiments.md): everything tried, with the numbers and the runs behind them.
* [Optimizations](optimizations.md): how a step got fast, and what limits throughput now.
* [Architecture](architecture.md): the components, the step protocol, and what was learned about the 1.29 engine.

The dashboard (`python -m warcraftsim dashboard`) shows every run live; this tab shows these documents (`docs/*.md`) and edits them in place.
