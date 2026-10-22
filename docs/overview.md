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

`fgself-107` added what the agent could not see (each hero's abilities, cooldowns and items, what each building makes, the time of day, the enemies it saw before: `--inputs 3`) and masks for orders the game would refuse (`--tech-mask`, `--valid-mask`); it fell below its start (41% head to head), likely from the remembered enemies looking like enemies in sight at first. On the rating scale both were about level with their start, a strong checkpoint (`fgself-108`, without the memory: 2024 ± 22 to its 2070 ± 19). `fgself-109` added narrow targets and more kinds of conditioning: level on the rating, ~75-80% of the due target items made, the easy kinds kept (workers 88%, army size 84%, rush 76%), the hard ones not (expand 47%, low upkeep 34%, no rush 14%). `fgself-110`, rewards by part (GDPO), was worse (the material shaping back at full weight). `fgself-111` added the tavern's heroes, mercenaries and the goblin merchant as orders, a heuristic teacher and targets with tavern heroes: level (2031 ± 20 to its start's 2010 ± 9), and after a day of fixes (every buy order had been refused: the tavern stands in the fog at the start) it hires (413 tavern heroes and ~1,500 mercenaries in 2,751 training games). In the real game the policies' untaught mercenary purchases cost wins, so they are off now (`--neutral 2`). `fgself-112`, self-distillation with privileged information (OPSD), gave no gain in an hour and a half (44% head to head with its start). `fgself-113` (scenarios) was confounded by two bugs found on the way (a run's vocabulary differed between processes, so forks lost their parent's league and swapped two spell toggles). `fgself-114`, scenarios again, clean (a quarter of the training games start with armies between the bases, a raid, or extra money): level. `fgself-115`, weight averaging (a soup of `fgself-111`'s and `fgself-114`'s last checkpoints): level, worse with a target. `fgself-116`, rewards by part (GDPO) with the material shaping at half weight: worse (38% head to head with its start; targets followed a little more). `fgself-117`, the enemies it saw before as inputs faded into its attention: worse, as `fgself-107`. Two rules of the game had been wrong since `fgself-111`: the first hero is free (the mask held it back until 425 gold and 100 lumber), and about half the purchases never went through (a worker's right-click on the tavern). `fgself-120`, the corrected rules alone: level, the new baseline (first hero at 60 s, tavern heroes hired as a third of the targets ask). `fgself-121`, self-distillation with privileged information (OPSD), clean: no gain. An audit (2026-10-22) found seven more silently wrong assumptions about the game, among them: dead heroes were never revived, undead could not build a haunted mine (so never expanded), workers could not be moved to another mine, and on `echofast` most of Echo Isles' creeps kept full hit points. No policy had ever expanded (0% of games, a hall far from home at a gold mine). `fgself-122`, the corrected game: level to a little lower, the base. `fgself-123`, expansions taught with a lesson to clear the natural's camp: much worse (armies dragged away), no expansion. `fgself-124`, expansion teaching without the camp lesson (the naturals' camps gone in a third of the training games): the policy learned to build a hall, at home (28% of games), not at the natural. `fgself-125`, an expand order of the agent's own (`--macros 1`: the policy chooses to expand, the agent builds the hall where people do): level to a little better, and it expands in a quarter of the games with the naturals free (not yet with their camps standing, as in the real game). `fgself-126`, a creep order: it crept and expanded with the camps standing, and lost those games; a second audit found why (every worker took the expand order at once, any unit the creep order from minute 1, the newest units past 96 cut from the view, build orders held 30 s). `fgself-127`, the expand and creep orders with those fixed: better than its start (rating 1874 to 1810 at its best hour), and the first policy to expand in the real game (14 of 150 evaluation games); its weak point is playing a target (24% against insane). `fgself-128`, targets not enforced (a side with a target may answer with items outside it): better, and its target play more than doubled (58% against insane with a target, 24% before; 93% without), the targets still followed (70%). A hall's tier order was refused whenever it trained workers (the policy's tech held back for all races). Running: `fgself-129`, tier orders that go through: the best yet (rating 2029 to its start's 1873; against insane 96%, 70% with a target; tier 2 in 63% of the real games), its army still tier 1.

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
