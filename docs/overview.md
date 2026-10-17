# Overview

warcraftsim trains agents on the real Warcraft III 1.29, headless under Wine. Python steps the game every half second of game time. The project reimplements nothing.

## Status

| track | best result | run |
|---|---|---|
| Micro fights | 91% against the scripted opponent. League self-play: 82%, and 79% against the 91% policy. | `--trainer torch` |
| Whole game, `duelrush` (2-minute games) | **79%** against the built-in AI, all 16 matchups. | `fgself-10` |
| Whole game, `duelfast` (6-minute games) | **80%** of its games against the insane AI won. Every matchup is above the AI's own record. | `fgself-51` |
| **The real game on Echo Isles** (the game's rules, no handicap) | **~82% of its games against the insane AI won** (~380 evaluation games, against ~74% for the previous best): ~166% of the AI's own record on Echo Isles. Human ~64%, undead ~63%, orc and night elf ~95%. | best `fgself-77`, `fgself-78` running |

The yardstick is the evaluation games: the real game against the built-in AI, which the policy does not train on. Each policy in a run plays the same matchups (paired), so a run compares its policies directly.

## Next (2026-10-17)

1. `fgself-78`: a target strategy for half the training games (strategy conditioning), from the built-in AI's games and people's replays, with a small reward to follow it.
2. Less pull to the anchor helped (`fgself-77`): the next runs keep it at a quarter.
3. A main exploiter (`fgself-76`) did not help: it never beat the main policy in 70% of its games.

[The road to the real game](road-to-the-real-game.md) has the details.

## How we make a whole-game agent

```mermaid
flowchart LR
  D["Demonstrations<br/>built-in AI vs itself"] --> BC["Behaviour cloning<br/>transformer over units"]
  BC --> SP["Self-play (PPO + UPGO)<br/>forward KL to an anchor"]
  SP <--> L["League<br/>itself · past snapshots · built-in AI with a curriculum"]
  SP --> A["New anchor<br/>a recent checkpoint"]
  A --> SP
  SP --> M["A new map<br/>grow the network, keep the frame"]
  M --> SP
  SP --> E["Evaluation games<br/>vs the untaxed built-in AI"]
```

* A clone of the built-in AI is the start. Reinforcement learning then improves on it.
* The KL term keeps the policy near an anchor. The anchor moves to a recent checkpoint as the policy improves.
* The policy now beats its teachers. The advice and the cloning loss from the built-in AI are off.
* Each experiment is a new run, forked from the best checkpoint. It starts with its parent's league.

## Read next

* [Environments](environments.md): the maps, the opponents, what the agent sees and does.
* [The whole-game model](model.md): tokens, transformer, action heads.
* [Experiments](experiments.md): what worked and what did not.
* [The road to the real game](road-to-the-real-game.md): what is still missing.
* [Optimizations](optimizations.md): how a step became fast.
* [Architecture](architecture.md): how Python drives the game.
* Related work: [AlphaStar](related/alphastar.md), [OpenAI Five](related/openai-five.md) and [robot learning](related/robot-learning.md), and how this project differs.
