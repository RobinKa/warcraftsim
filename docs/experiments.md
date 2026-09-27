# Experiments

What was tried on the micro tasks, and what came of it. Run names (`abil6-*`, `lamfocus-2`, ...) are
runs in the author's `runs/` directory, shown by the dashboard; they are not part of the repository.
Win rates are against the scripted opponent; scripted baselines use `scripts/baselines.py`, fitted
scripts `python -m warcraftsim.puffer.bc eval`.

**Summary**
* On `footmen2`, PPO reaches 95% wins in about 1.5 minutes with tuned settings.
* On random mirror matches, PPO from scratch ends near "let the units fight on their own".
  Team tactics (focus fire plus pulling hurt units back) don't emerge from random exploration: each
  half alone doesn't pay.
* Starting from a behavior-cloned script fixes that: PPO keeps and sharpens the script's tactics.
* Longer credit (horizon 64, λ 0.95 instead of horizon 16) lets PPO discover pull-backs from a
  focus-fire clone, which horizon 16 never did.

## footmen2

* Scripted baselines win 0% (random), 30% (noop), 65–70% (focus fire), 90% (focus fire, and pulling a footman back while it is low and being hit).
* Tuned by sweeps (`f2-sweep1`..`5`, `f2-step*` in the dashboard): 0.5 s steps, lr 0.01, minibatch 192, replay ratio 4, the learning rate annealed over 400k steps → 95% wins after ~0.2M steps (~1.5 min), 98-100% soon after; 1.0 s steps: 100% for both seeds.
* Most of the speed came from more updates per sample (1 → 32 per epoch), then from longer steps and a shorter annealing schedule. Horizon 32 learns fastest at first but ends lower.
* Early policies learned focus fire plus pulling a hurt footman back. The 100% policy instead holds position until the enemies arrive: the scripted opponent then splits its damage over both footmen, while ours focus one enemy.

## Mirror matches: how many hit points

Hit points decide whether micro matters (scripted baselines, 120-150 episodes each; the time limit scales with hit points):

| hit points | episode | noop | focus + pull back (`pull35`) |
|---|---|---|---|
| 25% | 21 s | 55% | 44% |
| 35% | 28 s | 53% | 64% |
| 40% | 31 s | 54% | 80% |
| 50% | 36-40 s | 53% | 79% |

At 25% units die within a few hits and every order costs more than it gains (attacking the weakest enemy in range 48-50%, pulling back without focus 19%); RL at 25% converged to noop (lr 0.003: 52%). `mirror_mix_hp400` is the training setting.

## Hero abilities (`mirror_mix_abil_hp400`)

Results on `mirror_mix_abil_hp400` (0.5 s steps; runs `abil40-*`):

| policy | win rate |
|---|---|
| noop (no casting) | 31% |
| pull35 (no casting) | 29% |
| castnoop | 50% |
| castpull35 | 50% |
| PPO from scratch, lr 0.001, 2.5M steps | 42% |
| PPO from the fitted castpull35, lr 0.001, 2.5M steps | 52% |

Not casting against a caster loses badly. PPO from scratch found a shortcut: 60-66% of its actions are casts and 94% of those aren't possible. An impossible cast does nothing, so this means "do nothing, and cast each ability the moment it's ready".

### Sweeps for learning speed

Sweeps for learning speed from scratch (with action masks; 600k steps, ~5 min each; runs `abil6-*`, `abil6b-*`). At this budget the win rate is still rising, so faster runs show as higher numbers:

| round 1 (base: lr 0.001, minibatch 192, replay ratio 4, horizon 64, γ 0.99, λ 0.9, entropy 0.001) | win @0.3M | win @0.6M |
|---|---|---|
| lr 0.003 | 21% | 29% (peak 34%) |
| replay ratio 8 | 9% | 27% |
| base | 4% | 25% |
| lr 0.002 | 7% | 23% |
| minibatch 384 | 8% | 22% |
| hidden 256 | 4% | 21% |
| λ 0.95 / entropy 0.0003 | 1% | 15% |
| γ 0.97 | 0% | 0% (never learned) |

| round 2 (base: lr 0.003) | win @0.3M | @0.45M | @0.6M |
|---|---|---|---|
| horizon 32 | 27% | 35% | 33% (peak 40%) |
| lr 0.005 | 26% | 28% | 33% |
| λ 0.8 | 24% | 31% | 33% (peak 38%) |
| γ 0.995 | 24% | 31% | 33% |
| clip 0.3 | 20% | 33% | 34% |
| base (lr 0.003; round 1: 21% / 29%) | 19% | 28% | 33% |
| replay ratio 8 | 10% | 26% | 30% |
| entropy 0.003 | 7% | 16% | 24% |

| round 3 (base: lr 0.003, horizon 32) | 30% reached at | win @0.3M | @0.45M | @0.6M |
|---|---|---|---|---|
| horizon 16 | 0.19M (1.8 min) | 33% | 39% | 39% |
| clip 0.3 | 0.25M | 34% | 33% | 40% |
| λ 0.8 | 0.27M | 31% | 37% | 36% (peak 42%) |
| lr 0.005 + λ 0.8 + γ 0.995 + clip 0.3 | 0.30M | 29% | 38% | 38% |
| lr 0.005 | 0.31M | 29% | 27% | 35% |
| γ 0.995 | 0.33M | 27% | 34% | 32% |
| base (horizon 32; round 2: 27% / 35% / 33%) | 0.40M | 21% | 31% | 36% |
| minibatch 96 | 0.45M | 27% | 30% | 35% |

| round 4 (base: lr 0.003, horizon 16) | 30% reached at | 35% reached at | win @0.3M | @0.6M |
|---|---|---|---|---|
| **λ 0.8 + clip 0.3** | **0.10M (1.0 min)** | **0.15M (1.5 min)** | 34% | **45%** (peak 47%) |
| horizon 8 | 0.13M | 0.15M | 37% | 40% |
| horizon 8 + λ 0.8 | 0.12M | 0.15M | 33% | 41% |
| clip 0.3 | 0.13M | 0.24M | 42% | 41% |
| λ 0.8 | 0.15M | 0.25M | 35% | 40% |
| base (horizon 16; round 3: 0.19M) | 0.23M | 0.29M | 35% | 38% |
| lr 0.005 | 0.24M | 0.37M | 32% | 39% |

Shorter horizons (64 → 32 → 16) give more, smaller updates per sample and learned fastest; horizon 8 costs throughput (the trainer updates twice as often). The winner, horizon 16 with λ 0.8 and clip 0.3, is the task's default now (`Task.train_defaults`, used where the command line doesn't set an option). It reaches 30% wins after 0.1M steps; the settings this started from needed 0.66M.

Seed noise is about ±4 points at 0.6M, where the learning rate has annealed to zero and most configs end alike; the earlier columns separate them better. The minibatch must be a multiple of the horizon (PufferLib asserts it).

### The plateau

The plateau: from scratch the tuned settings reach 40-45% after 0.6M steps, and 42-44% after 2.5M (runs `abilbest-*`). What was tried at 600k steps (runs `plat-*`; base 40%, other seeds 45% and 38-45%):

| change | win @0.6M |
|---|---|
| hidden 256 / 3 layers | 39% / 39% |
| entropy 0.003 annealed to 0 | 43% |
| γ 0.995 | 40% |
| replay ratio 2 / 8 | 39% / 40% |
| relational features (`_rel`: nearest opponent, in range, threatened, weakest, time to die) | 37-39% |
| tactical masks (`_tac`: retreat only while losing hit points, no plain moves) | 38-40% (faster early; retreat still unused) |
| 0.25 s steps (horizon 32) | 33-35% |
| **start from the fitted `castpull35` (masked BC), lr 0.001** | **52%** (50% after 0.27M steps) |
| start from the fitted `castpull35`, lr 0.003 | 48% |
| value-loss weight 0.5 / 4 (default 2) | 42% / 38% |
| value clipping off / gradient norm 0.5 / lr floor 20% | 40% / 40% / 41% |
| V-trace / momentum 0.9 / momentum 0.98 | 38% / 37% / 36% |
| one agent per unit, one shared policy (`_units`, 3M agent steps = 0.6M game steps) | 43% / 45% (30% after 54k game steps, 0.6 min) |

Scripts that cast more carefully don't do better either (`smartcast`: area spells only with two enemies inside, targeted spells on the biggest threat, heals below half; 200 episodes each): `castnoop` 53%, `smartcastnoop` 48%, `smartcastpull35` 55%. Against a casting opponent, reasonable strategies all end near a coin flip: the headroom here may be small.

The trained policy plays as well greedily as sampled (43% / 42%), so that is its level. With the scripted casting rule in place of its own casts it wins 45%; `castnoop` (no attack orders at all, the same casting) wins 50%. So the gap to the scripts is partly casting, and partly attack orders that do worse than letting units auto-acquire. Only a better starting point moved the plateau.

Longer from the fitted script (`bclong`, 2.5M steps, lr 0.001): 47% at 0.3M, 50% at 2.5M, flat. It keeps the script's style: 91% attack orders, 6% casts.

### Self-play

Self-play (`mirror_mix_abil_self_hp400`, `MirrorSelfPlayEnv`: the policy plays both sides of the same game with the full action set; the observations and actions are the single-agent task's, so a checkpoint also plays the scripted opponent with `bc eval`). Starting from `bclong`, 3M agent steps (run `selfplay1`), against the scripted opponent over training: 50% → 51% (0.6M) → 44% (1.2M) → 42% (1.8M) → 41% (2.4M) → 49% (3.0M, lr at 0). Plain self-play against the latest self drifts away from what beats the script and doesn't improve it. PufferLib's self-play pool (older checkpoints as opponents in part of the games: `--selfplay.enabled=1 --vec.num_policies=2 --vec.hist_policy_percent=0.5`) works with this env; the second agent carries policy tag 1. With it (run `selfpool1`: half the games against a past checkpoint, resampled every 100k steps, from `bclong`, 3M agent steps), the results against the scripted opponent stay at 45-52% (47%, 48%, 52%, 48%, 45%, 49% over training): no drift, but no gain either.

## Without abilities: pulling hurt units back (`mirror_mix_sem*_hp400`)

Without abilities the plateau is an exploration problem. `pull35` (68%) is focus fire plus pulling hurt units back, and neither half works alone: focus fire 53% (noop 51-54%), pulling back without focus 19% (at 25% hit points). PPO from scratch with every improvement above ends at noop's level (runs `sem6-*`: 47-49%; with tactical masks `semtac6-*`: 42-48%, 97% noop and no retreats): trying either half alone is punished, so it never finds the pair.

### From a fitted script (behavior cloning)

Results on `mirror_mix_sem_hp400` (noop 54%, the `pull35` script 67-68% at 0.5 s steps; runs `mix40-*`):

| start | lr, entropy | win rate |
|---|---|---|
| random (PPO from scratch) | 0.003, 0.001 | 44% after 3M steps; never learned to retreat |
| random (PPO from scratch) | 0.001, 0.001 | 47-48% after 2.5M (92% noop, no retreats) |
| `pull35` fitted exactly (62% sampled, 64% greedy) | 0.001, 0.001 | ~70% throughout (stopped at 0.8M) |
| `pull35` label-smoothed (33% sampled) | 0.001, 0.001 | 67-69% after 1M, 70% (best 73%) after 2M |
| `pull35` label-smoothed | 0.001, 0.003 | 62% after 2M (more randomness, no new behavior) |
| noop label-smoothed (~18% sampled) | 0.003, 0.001 | 42-47% at 1-1.5M (stopped): no better than from scratch, still 25% harmful attacks |
| noop label-smoothed | 0.001, 0.001 | 49-50% after 2.5M (90% noop, no retreats) |

Without a good script to start from, PPO converges to letting the units fight on their own (≈ noop); it never discovers pulling hurt units back. From the script, it converges onto the script's behavior (99% of attacks on "weakest", 3-4% retreats) rather than beyond it. Random actions are very costly here: 7.5% of them turn noop's 54% into ~18%, since a random retreat or retarget takes a unit out of the fight for a second or two. Exploration is therefore punished hard.

### Tactical mode

Tactical mode (`_tac`) tries to make the pull-back discoverable:
* no plain moves;
* a retreat only for a unit below half its hit points that is losing them;
* a chosen retreat goes on, re-ordered every step, while the unit keeps losing hit points (at most 3 s), so one decision is the whole pull-back.

With the same mechanics `pull35` still wins 68% (a first version with a fixed 1.5 s commitment kept units out too long: 51%). PPO still doesn't learn it. From a fitted focus-fire script (`focus`, 49%, no retreats; runs `focusrl*`) it holds ~50% and drops its retreats from 1-2% to 0%, even with the commitment. A reward for kills and losses (`_kill`, ±0.2 per unit) doesn't change that (48% from the focus clone; from scratch 49% with tactical mode, 48% without; runs `kill-*`, `killnotac`). From the fitted `pull35` (`pullrl2`), PPO keeps its retreats (3% → 2% of unit decisions) and goes from 58% to 65-66% in 3M steps. PPO can value retreats when they come with the rest of the strategy: forbidden at evaluation (`bc eval --forbid retreat`), the final `pullrl2` policy drops from 66% to 46% (240 episodes each). Pulling back pays because the attack-moving opponent keeps switching targets and chasing, and that happens only when the whole team pulls hurt units consistently. A lone retreat while the rest fight just loses that unit's damage. Isolated exploratory retreats therefore look useless, and the strategy has to come from somewhere else (a script, replays).

### Pulling back only now and then

Yet pulling back only now and then already pays. With focus fire, a script that pulls a unit back only with probability P each step it could (`pull<L>p<P>`, 200 episodes each) wins:

| focus | pull35 | pull35p30 | pull35p10 | pull50 | pull50p10 |
|---|---|---|---|---|---|
| 56% | 70% | 70% | 65% | 47% | 66% |

So near the focus policy, occasional retreats have a clear gradient (+10 points at a 10% rate), and yet PPO from the focus clone removes them.

### The horizon

The horizon was the problem. All the runs above used horizon 16 (`Task.train_defaults` of the ability tasks, tuned there for speed over the first 0.6M steps). GAE then sums at most 16 steps (8 s) and trusts the value estimate after that, while a pull-back pays off later: the unit survives to fight on, and the enemy that chased it gets focused. From the focus clone (lr 0.001; runs `lamfocus*`):

| horizon, λ | win @0.6M | @1.5M |
|---|---|---|
| 16, 0.8 (`focusrl4-*`) | 48-52% | 48-50% |
| 16, 0.95 | 49% | 50% |
| **64, 0.95** (2 seeds) | 54% / 52% | **60% / 59%** |
| 64, 0.99 | 54% | 59% |
| 128, 0.95 (minibatch 384) | 52% | 53% |

With retreats forbidden at evaluation, the 60% policy drops to 48%. It learned to pull back, which no run from the focus clone had done. From scratch, horizon 64 doesn't help (41-44% after 1.5M; runs `lamscratch-*`). Near noop, pulling back alone doesn't pay, so there is nothing to follow yet: PPO first needs focus fire, which it doesn't find on its own either.

### Rejoin

Pulling back without focus fire failed for a mechanical reason: a unit that pulled back stood where the move left it, out of reach, until an attack order came. Without focus fire, 1-2 steps later, it never came (`nooppull35`: 4% at 40% hit points, 1% at 70%, many draws). `_rejoin` (implies `_tac`) makes the pull-back one whole maneuver: when it ends and the unit is told nothing (noop), it attack-moves back into the fight. With it (200 episodes each):

| noop | nooppull35 | nooppull35p10 | focus | pull35 | pull35p10 |
|---|---|---|---|---|---|
| 52% | 40% | 57% | 48% | 70% | 65% |

Now occasional pull-backs from noop pay a little (+5), but PPO from scratch still doesn't find them (horizon 64, λ 0.95: 47% for both seeds, retreats 1%; runs `rejoin-*`; 41-44% without rejoin).

The fitted `pull35` on the rejoin task (`bc/mirror_mix_sem_rejoin_hp400-pull35`; the script wins 67% over the 2000
recorded episodes) plays better than the script: 68% sampled and 79% greedy after 60 epochs (56% / 75% after 30,
when its retreat recall was still rising: 0.80, then 0.86).

## General orders and the entity network (`mirror_mix_gen*`, `--trainer torch`)

General orders have no built-in tactics (see the README). The same scripts, written with them
(120 episodes each): noop 52%, focus 61%, pull35 71%, pull35p10 58%; with abilities, castnoop and
castpull35 52%. The action space can express what the built-in retreat and target rules did.

* **From scratch**, PPO with the entity network collapses to noop within ~30k steps (entropy
  7.6 → 0.01; run `torchsmoke`): random stops, holds and moves are costly, so the first thing it
  learns is to stop giving orders, and it stays there (~50%).
* **Behavior cloning**: the entity network fits `pull35`'s general orders almost exactly (40
  epochs, 3 min: order accuracy 99.4%, recall of the pull-back moves 0.98, of attacks 1.00;
  `bc/mirror_mix_gen_hp400-pull35`). PufferLib's network, fitted to the same script with the
  built-in retreat, recalled 0.80-0.86 of the retreats. Played with sampled actions and no updates,
  the clone wins 65.6% (2242 episodes; run `genbc-1`); the script 68-71%.
* **PPO from the clone** (lr 3e-4, horizon 64, λ 0.95; runs `genbc-*`):

  | | 0.2M | 0.5M | 1M | 2M |
  |---|---|---|---|---|
  | plain PPO (`genbc-2`) | 57% → 21% | stopped at 0.4M (18-47%) | | |
  | with a KL of 0.1 to the clone (`genbc-3`) | 69% | 71% | 74% | **77%** |

  Plain PPO moved the policy too far per update (KL 0.03-0.06 per epoch, a fifth of the samples
  clipped): it walked away from what the clone knew before its value estimates were any good, and
  collapsed. A KL penalty toward the clone, as AlphaStar keeps its policy near the supervised one,
  kept it stable, and it went on past the script it was cloned from (77% against 71%).
* **Gentle fine-tuning works better than the anchor** (runs `genft-*`: lr 1e-4, 10 epochs that train
  only the value first, each epoch's updates stopped at a KL of 0.02):

  | | 0.5M | 1M | 1.5M | 2M |
  |---|---|---|---|---|
  | gentle (`genft-1`) | 75% | 81% | 87% | **91%** |
  | gentle, with the KL to the clone (`genft-2`) | 71% | 73% | 74% | 75% |
  | lr 3e-4 with the KL to the clone (`genbc-3`) | 70% | 72% | 75% | 76% |

  The collapse came from the first updates, made before the value head could judge anything.
  With the value trained first and small steps, the anchor only holds the policy back.

  What the 91% policy does differently (`scripts/target_choice.py`, 40 episodes): it keeps the
  script's pull-backs (7% of orders are moves) and its focus fire (99% of the attacks in a step go to
  one enemy), but not its target: 72% of its attacks go to the enemy with the fewest hit points,
  where the script's go there always. When it picks another, that is mostly the enemy with the
  lowest share of its hit points left (71% of those), often the hero (55%). It finishes the most
  damaged unit, often the hero, which carries more hit points and does more damage. Its side loses
  1.25 units per game instead of 2.3. Only a pointer at units could learn this: the rule-based
  targets of the older tasks had no rule for it.

### Hero abilities with general orders (`mirror_mix_gen_abil_hp400`)

The same recipe on the ability task, against the casting scripted opponent: `castpull35` with
general orders (52% over 2000 episodes) fitted in 40 epochs (order accuracy 98.8%, recall of casts
0.96, moves 0.96, attacks 0.99), then fine-tuned gently (runs `genabil-*`):

| | win rate |
|---|---|
| the clone, no updates | 51% |
| fine-tuned, 0.5M / 1M / 1.5M / 2M steps | 61% / 68% / 69% / 68% |
| PufferLib's network: from scratch / from the fitted script (earlier) | 42-45% / 52% |

The plateau of the older tasks ("reasonable strategies all end near a coin flip") was the network's,
not the task's.

### League self-play (`mirror_mix_gen_self_hp400`)

The first league run (`genleague`: from the fitted `pull35`, gentle fine-tuning, 3M steps; half the
games against itself, a quarter against past snapshots by PFSP, a quarter against the scripts noop,
focus and pull35) looked good on its own terms: it beat its past snapshots 63-70% and the focus and
pull35 scripts 70-80% (40% at the start). Head-to-head it beat the 91% policy `genft-1` 68%
(`scripts/match.py`: 46 wins, 11 losses, 43 draws).

Against the game's scripted opponent, which attack-moves at the nearest enemy, it won 4%.

It had learned to run: 38-61% of its orders were moves away from the enemy. In a mirror fight the
side that walks into range is hit first, so against copies of itself waiting and backing off pays,
and the games ran into the time limit (draws went from 18% of its self-play games to 46%). None of
its opponents chased relentlessly, so nothing punished running; the scripted opponent does exactly
that, and ran it down. Self-play found an equilibrium of its own games, not strength.

Two changes for the next run: a draw counts as a loss for both sides (`_nodraw`), and the league's
scripts include `amove`, which attack-moves at the nearest enemy each step like the game's scripted
opponent: an anchor that punishes running away.

The second run (`genleague2`, with both) beat its `amove` 92-100%, and the game's scripted opponent
still only 18%. The script was broken: it re-ordered every unit every step, and a new order cancels
an attack in progress, so its units hardly landed a hit. The game's scripted opponent orders only
idle units (at the nearest enemy's position) and leaves fighting units alone. With that behavior
(`amove` now), the script matches it: 84% and 22% for two policies that score 91% and 18% against
the game's opponent.

Meanwhile the three policies form a cycle, as non-transitive games do (`scripts/match.py`, 120
episodes each):

| | against |
|---|---|
| `genft-1` (trained against the scripted opponent) | wins 91% against the scripted opponent |
| `genleague2` | wins 70% against `genft-1` (65-16, 39 draws), 57% against `genleague` |
| the scripted opponent | wins 82% against `genleague2` |

The fine-tuned policy waits for the attack-moving opponent and punishes it; the league policies
out-wait the fine-tuned one, which walks into them; the scripted opponent keeps coming and runs the
league policies down. A league needs every one of these styles among its opponents; this is what
AlphaStar's league exploiters are for.

The third run (`genleague3`: no draws, the corrected `amove`, scripts chosen by PFSP too, so the ones
it still loses to come up more) broke the cycle:

| | `genleague3` | `genleague2` | `genft-1` |
|---|---|---|---|
| against the game's scripted opponent | **82%** | 18% | 91% |
| head-to-head against `genft-1` | **79%** (87-18, 15 draws) | 70% | |
| head-to-head against `genleague2` | 60% (44-21, 55 draws) | | |

It gives up nine points against the scripted opponent to the policy trained only against it, and
beats that policy four games in five. The anchors decided it: with an opponent in the league that
chases like the scripted one, running away stopped paying.

### League self-play with hero abilities (`mirror_mix_gen_abil_nodraw_self_hp400`)

`genleague5` continued `genleague3` on the task with abilities: the unit features grow by the ability features (their input weights start at zero, so it starts as `genleague3` exactly), and the cast order's logit gets +6 (`--cast-bias`): `genleague3` never had a cast possible and would have tried one 0.5% of the times it could. The league's scripted anchors cast like the game's scripted opponent, and a main exploiter plays a quarter of the games against the main learner. 3M steps in four pieces (`genleague5`, `5b`, `5c`, `5d`: resumed after trainer speedups, the league and the learning-rate schedule carried over).

| | against the casting scripted opponent | head-to-head |
|---|---|---|
| `genleague5d` (3.0M) | 56% | beats `genabil-2` 56% (64-49, 7 draws), `genleague3` 71% (81-31, 8 draws) |
| `genabil-2` (fine-tuned against the scripted opponent only) | 66% | |
| `genleague3` (its start: never cast) | 36% | |

The same pattern as without abilities: the league policy gives up ten points against the one opponent the specialist trained against, and beats the specialist head-to-head. Casts stayed at about 4% of its orders throughout. The exploiter's win rate against the main learner sank from 47% to 10-20% in the first million steps and never reached the 70% that restarted it; now it also restarts below 20%, from the main learner's current policy.

## Whole games: cloning the built-in AI on `duelrush`

Demonstrations: built-in AI (normal) against built-in AI on `duelrush` (50% handicap, 0.5 s steps, all races; `fullgame/collect.py`). Games last 1-4 minutes of game time (a tie at 4). The fits are `fullgame-167`, `-253`, `-263` (the number of games); the policy plays the normal AI with `fullgame/play.py`.

What mattered, in the order it was found:
* **Not every recorded order is a decision.** The AI's order events include the engine's own: internal orders it gives to most units (851974 alone is 29% of the events), `resumeharvesting` (7.7%: a worker going back after a drop-off), `returnresources` and autocasts. A policy that learned `resumeharvesting` interrupted working harvesters, and the game refused two thirds of its orders. These are now dropped from the labels.
* **Re-issued harvest orders.** The AI re-orders harvesting workers to harvest every few seconds, with no effect in its games. Learned as decisions, the policy kept sending its workers to other trees and reset their work: 240 gold from 27 workers in 80 s. Harvest orders to workers that already harvest are dropped from the labels and skipped at play time: 4860 gold in two minutes with the same policy.
* **Production was invisible.** A building's current order is 0 while it trains, so the policy could not see a full queue and kept ordering more. Each building now has queued (its accepted train and research orders not yet done) and busy (from the production events) features.
* **Points:** x and y bins sampled independently paired the x of one place with the y of another; y is now chosen given x.
* **Overfitting:** at 250 games the validation loss rose after 9 of 20 epochs. Dropout 0.1 and keeping the best epoch fix that. A global temperature below 1 made the policy issue fewer orders; `--order-temperature` sharpens only which order a unit gets.
* `fullgame-263` (263 games, before the production features): 2 wins in 16 against the normal AI. Its economy works; it under-spends, often builds no barracks, and now and then gives rare orders (battle stations, board).

## Other findings

* lr 0.01 (tuned on `footmen2`) is far too high with 15 action heads: the KL per update was 1.0-1.5 (clip fraction 0.9) and the win rate peaked at 29%. The KL at a given lr grows with the number of heads (≈0.15 with 6, 0.3 with 9, 1.0+ with 15); lr 0.003 keeps it at 0.03-0.14.
* Action masks: before masks, 94% of the from-scratch policy's casts on `mirror_mix_abil_hp400` were impossible. With them, it reached 39% at 0.9M steps against 35% without.
* Updates per epoch are `replay_ratio × batch / minibatch`. With the minibatch equal to the batch (the old default), there was one update per epoch and learning was slow: `footmen2` reached 61% wins in 1M steps, against 97% with 16 updates.
