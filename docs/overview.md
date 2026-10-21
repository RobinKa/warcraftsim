# Overview

warcraftsim trains agents on the real Warcraft III 1.29, headless under Wine. Python steps the game every half second of game time. The project reimplements nothing.

## Status

| track | best result | run |
|---|---|---|
| Micro fights | 91% against the scripted opponent. League self-play: 82%, and 79% against the 91% policy. | `--trainer torch` |
| Whole game, `duelrush` (2-minute games) | **79%** against the built-in AI, all 16 matchups. | `fgself-10` |
| Whole game, `duelfast` (6-minute games) | **80%** of its games against the insane AI won. Every matchup is above the AI's own record. | `fgself-51` |
| **The real game on Echo Isles** (the game's rules, no handicap) | **~95% of its games against the insane AI won** (1,045 evaluation games of `fgself-102` to `-106`): orc 98%, night elf 100%, undead 94%, human 89%; ~1.8× the AI's own record. With a target build order: ~82-91%, most of the target made by the policy itself. | best `fgself-102` (`echofast` training toward target build orders, judged on the real game); running: `fgself-107`, new inputs and masks |

The yardstick is the evaluation games: the real game against the built-in AI, which the policy does not train on, and games against frozen snapshots of the policy (`--eval-vs`). Each policy in a run plays the same matchups (paired), so a run compares its policies directly.

## Since the milestone (2026-10-20)

Beating the insane AI clearly was the milestone; the user's direction now is strategic variety, then the whole game's actions. Tried since, all level with their start: AlphaStar's league with a main exploiter (`fgself-103`), OpenAI Five's reward decay (`-104`), each sample used once (`-105`), classifier-free guidance toward the target in evaluation (`-106`: it followed no better and won 50% instead of 91%).

`fgself-107` added what the agent could not see (each hero's abilities, cooldowns and items, what each building makes, the time of day, the enemies it saw before: `--inputs 3`) and masks for orders the game would refuse (`--tech-mask`, `--valid-mask`); it fell below its start (41% head to head), likely from the remembered enemies looking like enemies in sight at first. On the rating scale both were about level with their start, a strong checkpoint (`fgself-108`, without the memory: 2024 ± 22 to its 2070 ± 19). `fgself-109` added narrow targets and more kinds of conditioning: level on the rating, ~75-80% of the due target items made, the easy kinds kept (workers 88%, army size 84%, rush 76%), the hard ones not (expand 47%, low upkeep 34%, no rush 14%). `fgself-110`, rewards by part (GDPO), was worse (the material shaping back at full weight). Running: `fgself-111`, the tavern's heroes, mercenaries and the goblin merchant as orders, a heuristic teacher, and targets with tavern heroes.

Next, in order: narrow target strategies (a game's heroes, main army and what they need), more kinds of conditioning (no rush, rush, expand, workers, creeping, harass, scouting, army size, low upkeep), a per-reward advantage (GDPO, SAW), tavern heroes, neutral shops and mercenaries with a heuristic teacher, self-distillation from the same network given what the player can't see, and scenarios. [Ideas](reference/ideas.md) has each: where it came from, what became of it.

## How we make a whole-game agent

```mermaid
flowchart LR
  D["Demonstrations<br/>built-in AI vs itself"] --> BC["Behaviour cloning<br/>transformer over units"]
  BC --> SP["Self-play (PPO + UPGO)<br/>forward KL to an anchor"]
  SP <--> L["League<br/>itself · past snapshots · built-in AI with a curriculum"]
  SP --> A["New anchor<br/>a recent best checkpoint"]
  A --> SP
  SP --> M["A new map<br/>grow the network, keep the frame"]
  M --> SP
  SP --> E["Evaluation games<br/>the real game vs the untaxed insane AI"]
```

* A clone of the built-in AI is the start. Reinforcement learning then improves on it.
* The KL term keeps the policy near an anchor. The anchor moves to a recent best checkpoint as the policy improves.
* Target strategies (AlphaStar's z): in most training games a side plays toward a build order from the AI's games or people's replays, rewarded for following it; a builder's orders teach it (labels only, never carried out).
* Each experiment is a new run, forked from the best checkpoint. It starts with its parent's league.

## Read next

* [Environments](environments.md): the maps, the opponents, what the agent sees and does.
* [The whole-game model](model.md): tokens, transformer, action heads.
* [Experiments](experiments.md): what worked and what did not.
* [The road to the real game](road-to-the-real-game.md): what is still missing.
* [Optimizations](optimizations.md): how a step became fast.
* [Architecture](architecture.md): how Python drives the game.
* Related work: [AlphaStar](related/alphastar.md), [OpenAI Five](related/openai-five.md) and [robot learning](related/robot-learning.md), and how this project differs.
