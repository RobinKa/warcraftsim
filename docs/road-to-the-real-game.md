# The road to the real game

The goal is a policy that plays a real 1v1: a stock or ladder map, the game's own rules, any race against any race, against the built-in AI and then people. This page lists what stands between the current runs and that, roughly in the order it will be needed.

## Where the runs are

| stage | map and rules | game length | status |
|---|---|---|---|
| 1 | `duelrush`: small map, everything but walking 7× faster | ~2 minutes | **done**: 79% against the built-in AI in the real game (`fgself-10`) |
| 2 | `duelfast`: the game's speed, production times ÷3, half costs and hit points | ~6 minutes | **in progress** (`fgself-12`): AI-like openings and heroes, but it ties at the time limit and does not spend its money |
| 3 | `duel`: the game's own rules on the small map | ~12 minutes | not started |
| 4 | Echo Isles (or another ladder map), the game's own rules | 10–25 minutes | not started |
| 5 | people | | `fullgame/versus.py` exists; nobody has played it yet |

## Rules and map

* **Speed-ups.** Every stage so far changes the game's numbers. Each step towards the real rules has needed a new clone of the built-in AI on those rules: a policy trained on `duelrush` and moved to `duelfast` kept its duelrush habits (32 workers a game, no heroes) because its KL term held it to the old policy.
* **Map size.** The duel maps are 40–48 tiles with bases 3000–4600 apart; the smallest stock two-player maps are 80×80 with bases 8400–9500 apart. Walking matters much more there (scouting, reinforcing, defending).
* **What the duel maps leave out**: expansions (a second gold mine, and the timing of taking one), item drops from creeps, shops (goblin merchant, the race shops), the tavern (neutral heroes), mercenary camps, goblin laboratories (zeppelins, shredders), water.
* **Heroes to level 10.** On the duel maps heroes rarely pass level 3–4 (the built-in AI's winners have ~5 hero levels in all after six minutes on `duelfast`); creeping for experience and items is a large part of real play.

## What the policy sees

* **No terrain, pathing or trees.** Units are tokens with positions; the map is learned from coordinates, which works only on one fixed map. A real map pool needs a map input (a coarse grid of pathing, trees, creep camps, mines).
* **No memory of what is out of sight**, unless the memory core learns it; the game itself draws the last-seen buildings ("ghosts"), the policy does not get them.
* **No items or inventories**, no ability cooldowns or levels for the whole-game policy (the micro tasks have them), no shop stock.
* **At most 160 units** (96 own). Fine for the duel maps (~45 in view); a late real game with creeps can pass it.
* **Two small leaks** that a person does not get: visible enemy units' current orders, and visible enemy heroes' unspent skill points.

## What the policy does

* **Every own unit every half second.** There is no limit on actions per minute, no camera and no selection: in effect thousands of APM. AlphaStar capped APM and later played through a camera; a fair comparison with people needs something similar.
* **No item use, no buying**, no hero revival as its own decision (it is the hero's train order at the altar), no transports, no formations or control groups (not needed with per-unit orders).
* **Production is one order at a time** at an idle building; the built-in AI queues. The policy's buildings sit idle with money in the bank much of the time (see [experiments](experiments.md)).

## Learning

* **Long horizons.** A 25-minute game is 3,000 decisions a side at half-second steps. Gamma 0.999 looks ~8 minutes ahead; GAE (λ 0.95) assigns credit over ~20 steps. Longer games will need longer credit, a memory core, or decisions that span more time (macro actions, or a "when to act next" output as AlphaStar has).
* **Credit for rare decisions.** All units of a step share one advantage. Production and hero orders are a handful a minute among thousands of unit-steps; RL pushed their probability down on `duelfast` (basic units 39% → 20%, heroes 7.8% → 2.7% at the AI's own decisions).
* **Ties.** On longer games the policy learned to survive but not to finish (78–86% of curriculum games tied at 20 minutes).
* **Data.** Everything is cloned from the built-in AI. AlphaStar started from human replays; 1.29 replays of people exist, but a replay holds selections and orders, not states, so they would have to be played back in the engine with the harness to become demonstrations.
* **Opponents.** The built-in AI on insane is the strongest scripted opponent and is already beaten 50%+ on `duelrush`. Beyond it: the league (past snapshots, exploiters), and people.

## Compute

* Throughput is ~850 agent steps/s on this machine (16 cores, one RTX 3090), bound by the games' CPU (see [optimizations](optimizations.md)).
* A real 25-minute game is ~6,000 agent steps when both sides learn: ~500 games an hour, against ~8,000 an hour on `duelrush`.
* A stock map loads in ~8 s (the duel maps ~4 s), negligible for long games.

## Smaller items

* Replays of agent games play in the stock client only with the built-in AI's side right: agent orders are issued by the map script, not recorded in the `.w3g` (the harness plays them back exactly; `video.py` renders them).
* Only 1.29 (Legacy TFT) is supported; the shim's offsets are for that build.
* Five games per map load at most on the duel maps (12 players crash the game at load).
