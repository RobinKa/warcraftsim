# Overview

warcraftsim trains agents on the real Warcraft III 1.29, headless under Wine. Python steps the game every half second of game time. The project reimplements nothing.

## Status

| track | best result | run |
|---|---|---|
| Micro fights | 91% against the scripted opponent. League self-play: 82%, and 79% against the 91% policy. | `--trainer torch` |
| Whole game, `duelrush` (2-minute games) | **79%** against the built-in AI, all 16 matchups. | `fgself-10` |
| Whole game, `duelfast` (6-minute games) | **80%** of its games against the insane AI won. Every matchup is above the AI's own record. | `fgself-51` |
| **The real game on Echo Isles** (the game's rules, no handicap) | **~95% of its games against the insane AI won** (~355 paired evaluation games; `fgself-100`'s best 92%, `fgself-78` ~87% before): ~179% of the AI's own record. Human ~92%, undead ~89%, orc ~97%, night elf ~100%. With a target build order it follows on its own: ~83%. | best `fgself-102` (`echofast` training with target build orders and the weak races trained more, judged on the real game); running: `fgself-103`, AlphaStar's league (mostly self-play, past snapshots, a main exploiter) |

The yardstick is the evaluation games: the real game against the built-in AI, which the policy does not train on. Each policy in a run plays the same matchups (paired), so a run compares its policies directly.

## Next (2026-10-19)

1. `fgself-94`: targets the agent can follow. A new strategy pool (the normal built-in AI on `echofast` at native times, people's 1v1s mapped onto them, only items the agent can order, each covering its game's length) and a builder that follows the game's tech tree (units, research, tiers and what each needs first, supply, unfinished buildings resumed, the next item's cost saved from the agent's other orders). With the builder carrying out, 2.1 of ~13 due items are missed a game (from ~11 of ~22); the agent itself makes ~45% of what is made. Found on the way: the availability mask charged every upgrade its base's cost too (a keep 705 gold instead of 320), holding every policy's next tier back. Now `fgself-95`: the builder as a teacher only (`--z-build 2`): in training games the agent alone makes 80-82% of its due target items (its `current+z` evaluation games had the builder carry out until 16:04, an evaluation worker bug, now fixed). GRPO from mid-game branches (`fgself-96`, `-97`, `--grpo-share`) was level: only ~55% of branches replay exactly (fights with creeps do not) and the groups' returns hardly differed. Now `fgself-98`: the line back on the real rules, against `fgself-78`. Before it, `fgself-93`: strategic variety first (the user's direction, a lower win rate accepted for a while), on Echo Isles with the fast rules: enforced targets, the harness builds the target's buildings and heroes the agent leaves undone (`fgself-90`: more unit types and heroes with a target, but the agent itself still missed half its targets), then the builder's orders as labels (`fgself-92`: no lasting variety), and now the builder only teaches: its orders are labels, never carried out. With enforcement alone (`fgself-88`) and five times the following reward (`fgself-89`) it still missed ~14 of ~22 target items, mostly buildings. `fgself-88` (enforcement with the old reward) kept its win rate without a target (44% against `fgself-78`'s 42%) but barely followed: ~15 of ~21 target items missed, flat. Before it, `fgself-87` (8 layers) was level after ~50 minutes and stopped: no larger network without a clear gain. Every run since `fgself-79` started from `fgself-79`'s last with its stronger strategy reward (0.05), and every one had a weaker human (8-20 points behind, more workers): their "level" verdicts hid the other races' gains. Before it: a richer AI in training (`fgself-85`: the insane AI with 90% more income still lost ~85%; human collapsed, worse), target strategies from people only (`fgself-84`, level), workers capped in the material shaping (`fgself-83`, level).
2. Strategy conditioning (`fgself-78`) and a quarter of the anchor's pull (`fgself-77`) helped. The material shaping still helps: without it the policy lost ~5 points (`fgself-81`), and with a rolling anchor orc collapsed (`fgself-82`).
4. Did not help: more reward for following the target (`fgself-79`), a cloning loss to teach the targets (`fgself-80`, worse), a main exploiter (`fgself-76`).

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
