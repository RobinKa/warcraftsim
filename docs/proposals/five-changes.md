# Proposal: five changes after a week of audits

**Question.** After audits 12 to 16 (2026-10-26), what big changes would make the system better, rather than one more lesson or mask per bad habit?

**Answer.** Five, in this order: measure behaviour automatically (cheap, first); learn macro from people (the main project); train the late game directly and test every order's form against the game (alongside); and a tighter experiment process (when the second games machine arrives).

## What the week showed

| | |
|---|---|
| 34% | won against insane at a person's income once the doubled-income bug was found (it had read ~97%) |
| 8 of 137 | games over 18 minutes won (`fgself-153`); losses last a median 21 minutes, wins 13.5 |
| 0 of 286 | real games with an expansion; people expand in 81% of Echo Isles games |
| ~59% | against insane at a person's income (`fgself-155`), after the economy and teacher fixes |

Most fixes this week treated symptoms: a lesson, a mask or a bug fix for each habit found (workers crowding a mine, heroes left dead, peons in burrows, militia stuck, no upgrades, no creeping, no anti-air). The same few causes sit under them.

## 1. Learn macro from people

**Problem.** The KL anchor (`fgself-100/13649982`) is a clone of the built-in AI, and the policy's macro habits come from it plus hand-written lessons. Each lesson is a patch, and they interact: `fgself-154`'s composition targets leaked into the play without a target.

**Change.** Clone a policy from human replays, conditioned on the target build order, and make it the anchor and the main teacher. Expanding, upgrades, creeping, reviving and splitting workers between gold and lumber don't depend on the map, so replays from other maps teach them too.

**Data.** 2,591 1.29 replays on hand ([human replays](../reference/human-replays.md)); only the 95 on stock Echo Isles are converted. An agent is surveying more sources and versions, and the patch notes between versions (what other versions' build orders need adjusting for, and whether their replays can play back in 1.29 with the map's object data overridden). Phase 1 of cloning the hand-written player's strategies (at a person's income) is a smaller step the same way.

**Cost.** Days: replay playback on more maps, a cloning run, a new anchor. AlphaStar's recipe ([ideas from AlphaStar](../reference/ideas-from-alphastar-and-openai-five.md)).

**Status (2026-10-27).** Ready to run on the GPU: `bc.py --like-run` clones people's games on several maps into a network a run takes as `--init` (each game turned level in its own frame, its own targets as the strategy input, right clicks resolved, labels the run can't give left out); a CPU smoke test passed. The command, the evaluation and the risks: [cloning people](../reference/human-replays.md#cloning-people-the-anchor-to-be).

## 2. Train the late game directly

**Problem.** Training games are short, so mines running dry, tier 2-3 armies and upgrades are rarely trained, and a long game spends many steps for one late outcome.

**Change.** Start a share of training games from mid- or late-game states (minute 8-20), from the policy's own games or from replays, on `echofast2` or the real map, whose economies match the real game's.

**Status (2026-10-29).** Built, behind flags (`fullgame/late.py`): `--late-capture P` writes a share of the training games' states at a minute in `--late-minutes` (8-16) to the run's `late-states.jsonl`; the scenario `late` (`--scenarios ...,late --late-states FILES`) starts a game between agents from one whose races and start locations are the game's (Echo Isles is not symmetric enough to mirror a state). Five harness ops place it (units, buildings, a hall at its tier in place of the start's, heroes with level, experience, skills and items, research levels, mines' gold, creeps killed); in the real game every count, level and amount matched the capture at step 1 for all four races ([engine notes](../reference/engine-notes.md), `scripts/late_probe.py`). The built-in AI does not take over a placed base, so games against it never start late. Not reproduced: trees cut down, items on the ground, cooldowns, queues and research under way.

## 3. Measure behaviour automatically

**Problem.** Most bad habits were found by watching videos, and several yardsticks were wrong for a while: the doubled income, the laptop's relaunches inflating results, raw thirds misread as a decline, blind gauntlet games.

**Change.** A scorecard per checkpoint, against its start in the same windows by race (`scripts/head_to_head.py --vs` is a first piece), with confidence intervals: expansions, workers per mine, lumber per minute, upgrades, creeps, revivals, anti-air against air, refused orders, and wins in games past 12 and 18 minutes. A regression is flagged when it appears.

## 4. A conformance suite for every order

**Problem.** Many of this week's bugs were the agent sending order forms the game refuses: the militia's back to work, the revival price, Cloud's requirements, research levels, the step budget.

**Change.** A scripted probe that tries each order in the vocabulary, per race, against real game states, records what the game accepts, generates the mask rules from that, and runs again after each change to the agent's code.

## 5. A tighter experiment process

**Problem.** A verdict takes ~3 hours (~300 games a policy), and most forks this week bundled 4-6 changes, so a worse fork could not say why.

**Change.** Paired games (the same seed and matchup for a fork and its start), stopping once the confidence interval is clear, recipe files instead of 100+ flags a run, one change a fork, and ablations side by side once the second games machine adds games.

## What not to do

* Approximate the game's simulation: [the study](../game-speedups.md) found a few percent at most before income broke.
* More macro orders, against learning to play without them.
* A bigger network before the data improves.
