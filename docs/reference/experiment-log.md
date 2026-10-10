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

### Cloning under the rebalanced rules (`fullgame-rush2`)

3043 games under the v12 rules (walking workers carry more, 7× mines; all races, normal and insane AI; `demos-rush2-1`), 8 epochs: validation loss 3.05 (still falling at the end), the AI's order given a unit gets one: 69% (target unit 86%, point within 5.4 bins), and the rate at which units get orders matches the AI's (9.8% vs 9.9% of unit-steps). Against the normal AI in mirror matchups: 2 wins in 32 (both undead), most games lost within 1–2 minutes. About a third of its ~150 orders a game are refused: workers it cannot afford or house yet, buildings where they don't fit.

### Fewer wasted orders: an availability mask and snapping

A third of the clone's orders were refused, most of them for being unaffordable. The AI only orders what it can pay for, so the clone never saw an unaffordable order and never learned what "can't afford" looks like.

The fixes:
* **An availability mask** (`fullgame/costs.py`, `--avail-mask`): each order class's gold, lumber and food cost comes from the game's tables, scaled by the map's cost rule. The classes the player can't pay for at the step are masked.
* **Harvest snapping:** a harvest order aimed at anything other than a gold mine goes to the nearest mine in view.
* **Build snapping:** in the harness, a build that doesn't fit tries rings of spots around its point.

The share of refused orders fell as follows:

| orders | refused before | refused after |
|---|---|---|
| all | 38% | 25% |
| builds | 53% | 19% |
| train and research | 58% | 38% |
| harvest | 77% | 63% |

The clone plays no better for it: in 32 mirror games against the normal AI it won 1, tied 3 and lost 28.

### Self-play from the clone (`fgself-2`)

The setup was PPO from `fullgame-rush2`, mirror matchups, a league (itself, past snapshots, the easy and normal AI), material shaping and the tie-break.

After 2.1M agent steps it beat its past snapshots about 60% of the time, most of the rest ties. Against the built-in AI it had one tie in its last 75 games (easy and normal) and no wins. Self-play improves the policy against its own kind, but that doesn't carry over to the AI. The clone is the weak link: it starts too far behind for the anchor games to give a learning signal. The next steps are a better clone (more games, memory, a value head trained on the demonstrations), then self-play again.

### The clone hoards: stale queues in the demonstrations

The clone's economy is not the problem. It gathers as much as the AI (6791 gold per game vs 6028), but it doesn't spend it.

A frame at 62 s of a night elf mirror game:

| side | gold held | lumber held | food | material (units and buildings) |
|---|---|---|---|---|
| the clone | 2552 | 2272 | 17/30 | 3114 |
| the AI | 1512 | 982 | 53/60 | 7452 |

The cause is in the demonstrations. The AI retries its train orders until it can afford them, and the recorded order events include every attempt:
* 61% of the AI's train and research orders were unaffordable when issued, and 80% of those started nothing.
* Each attempt raised the building's queued feature. Nothing ever finished to bring it down.

The effect on the recorded state, over the AI's finished production buildings:
* They showed "5+ queued" while not busy for 36% of their steps.
* The clone learned to train mostly in that state: 16% per step, against 2.3% at an idle building.
* In its own games only accepted orders count, so its buildings are almost never in that state (1% of building steps). It sat mostly at idle buildings, where the learned rate is low.

The fix (`Encoder.encode` with costs):
* A cost-bearing order counts, both as a label and in the queue, only if the player could pay for it at the step. Costs are deducted in order within the step.
* A train or research order counts only if production started at that building within 2 steps, or the building was already busy (the order queues behind).
* BC trains with the same availability mask the policy plays with.

52% of the AI's train labels go. The share of building steps with 5+ queued falls from 23% to 2%, and with a queue but not busy from 22% to 3%.

### The clone cancels its own builds, so it is supply-blocked

The label fix alone didn't stop the hoarding. The ablation below compares two fits on the same 1000 games (3 epochs each), each playing 16 mirror games against the normal AI:

| labels | wins, ties, losses | gold + lumber held (clone vs AI) | food at 1 minute (clone vs AI) |
|---|---|---|---|
| old (refused retries kept) | 0, 1, 15 | not recorded | not recorded |
| new (`Encoder` with costs) | 0, 0, 16 | 4074 vs 1746 | 19 vs 46 |

Composition at 60 s showed why: the clone is supply-blocked.
* Undead: 10/10 food with no ziggurat, against the AI's 50/50 with 3–4.
* Human: 1 farm, against the AI's 7.
* Night elf: 1–2 moon wells, against the AI's 5 (+3 building).

With no free food the availability mask blocks all training, and the gold piles up.

Over 8 games, the clone had 80 accepted build orders and 50 constructions started (62%). The AI had 128 and 122 (95%).

Following each build showed the cause. Of 102 accepted builds, 48 never started, and in every one of those the worker got another order before reaching the site. All 45 builds that started were left alone. The policy picks every unit's order every half second, so a worker on its way to build keeps getting a chance of another order, which cancels the build. The AI leaves its builders alone until the building stands.

The fix: `BCAgent` keeps a worker whose current order is a building (on its way, or constructing) from getting other orders. This is `features.building`, with a 30 s timeout, and it applies in play and self-play.

In the same 8 games (seed 7), the fix changed:

| | builds started | held (clone vs AI) | food at 1 minute (clone vs AI) |
|---|---|---|---|
| before | 44% | 4760 vs 1942 | 15.6 vs 47.9 |
| after | 68% | 3847 vs 1760 | 18.6 vs 37.0 |

The builds that still fail lost their order in the game, probably at placement, not to another order from the clone.

Evaluations now record each side's gold and lumber on hand (mean over the game) and its food at one minute. The dashboard charts both.

### The opening, and demonstrations from the clone's own states

Timelines of 8 mirror games (`fullgame-rush2` with builder commitment) against the normal AI:

| time | clone food | AI food | clone gold on hand | AI gold on hand | clone army | AI army |
|---|---|---|---|---|---|---|
| 20 s | 15 | 23 | 2154 | 1248 | 0.4 | 3.2 |
| 30 s | 16 | 29 | 3280 | 1349 | 0.5 | 5.0 |
| 60 s | 16 | 52 | 6043 | 1681 | 1.4 | 11.8 |

Workers are on par, so the economy works. The clone is behind within 20 s, before any gold piles up.

Checks on the cause:
* **Calibration on the AI's states:** fine. In the first 10 s the model expects 3.6 build orders per side against the AI's 3.1, and 8.6 train orders against 8.4.
* **Clipping gold and lumber at play time:** a little better (army at 60 s: 2.9 instead of 1.4).
* **Several train orders to one building in one step:** 93% of the kept train orders are the only one to their building in that step, so one label per step loses little.
* **Timing:** at the steps where the AI builds something, the model expects only 0.28 workers to build it (median 0.07). The rest of its build probability sits on steps where the AI builds nothing. The totals match, but the AI's script runs on timers and conditions the model can't see.

One human game step by step:

| step | the AI | the clone |
|---|---|---|
| 1 | trains a peasant, builds an altar | trains a peasant, builds an altar |
| 2–4 | builds its first farm at 7/12 food | builds a barracks |
| 11–12 | builds a barracks | builds a second barracks |
| 20 | trains a Paladin, builds another farm | builds its first farm, already at 12/12 |
| 21–28 | trains footmen | stuck at 12/12 food |

After an early mistake the clone is in a state the AI's games never show, and cloning has nothing to learn from there. The answer is DAgger with the built-in AI as the expert:
* **Takeover games** (`collect.py --policy`): the clone plays one side for 5–90 s, then the built-in AI takes it over (`protocol.StartAI`) and its orders are recorded.
* BC skips the taken-over side's steps before the takeover.

`demos-takeover-1`: 4000 such games with `fullgame-rush2` as the clone. They also measure how fast the clone loses a game. Evenly matched AIs would win about 50%.

| the clone plays until | its side wins, the AI playing on | games |
|---|---|---|
| 5–20 s | 35% | 725 |
| 20–40 s | 23% | 925 |
| 40–60 s | 9% | 999 |
| 60–90 s | 6% | 1351 |

Even the first 20 seconds cost about 15 points, and after a minute the game is mostly lost.

More data and the new labels alone don't fix the opening. `fullgame-rush3` is fitted on the 8000 `demos-rush2-1` games with the new labels. After epoch 4 (validation loss 2.93, order accuracy 70%) it played 32 mirror games against the normal AI:

| | `fullgame-rush2` | `fullgame-rush3`, epoch 4 | the AI |
|---|---|---|---|
| wins, ties, losses | 1, 2, 29 | 1, 1, 30 | |
| gold + lumber held (mean) | 4563 | 4178 | ~1900 |
| food at 1 minute | 20.6 | 21.8 | 46–49 |
| refused orders | ~25% | 16% | |

Refused orders fell, but the spending gap stayed. The model is calibrated on the AI's states and still fails on its own, which is why the takeover games are the next step.

### Where the clone's army goes missing (`fullgame-rush3-tk`)

`fullgame-rush3-tk` is `fullgame-rush3` (epoch 5) fine-tuned on its 8000 games plus the 4000 takeover games. After epoch 2 it played 32 mirror games against the normal AI:

| | result | gold + lumber held | food at 1 minute | refused orders |
|---|---|---|---|---|
| `fullgame-rush3-tk`, epoch 2 | 1 win, 31 losses | 3951 | 24.1 | 13% |
| the AI | | 1965 | 43.3 | |

The first minute, per game, over 8 of those mirror games:

| | clone | AI |
|---|---|---|
| train orders | 24.8 accepted (+6.8 refused) | 251 (mostly retries) |
| training started | 22.9 | 27.5 |
| units trained: workers | ~14 | ~11 |
| units trained: army | ~4 | ~14.5 |
| units trained: heroes | ~0.4 | ~1.2 |
| constructions started | 9.3 | 11.4 |

The types are much the same: moon wells 2.9 vs 3.4, burrows 2.1 vs 2.6, altars and barracks about equal. The Ancient of War is 0.2 vs 0.5.

Two suspects are ruled out:
* **Placement.** The clone builds slightly closer to its hall than the AI (median 703 vs 816 units).
* **Training decisions.** At idle, finished production buildings that can afford a unit, the clone trains at the AI's rate or above: Ancient of War 0.22 vs 0.06 per step, orc barracks 0.25 vs 0.10, Tree of Life 0.22 vs 0.21, Great Hall 0.07 vs 0.07.

Every local decision is roughly calibrated. The army gap comes from compounding small deficits: army buildings come later, occasional supply blocks, and a unit mix that leans toward workers. What's left is timing, which cloning can't pin down: the AI's script runs on timers the model doesn't see. Self-play's material shaping rewards every unit and building made, which is the signal this needs.

Right after a takeover, even the AI orders modestly: 0.13 train orders and 0.11 build orders per step in the first 10 s. The model on those states: 0.19 and 0.09.

### Self-play from the new clone (`fgself-3`)

`fgself-3` is PPO self-play from `fullgame-rush3-tk`, with the value head trained in BC (a 2-update warm-up), builder commitment and the fixed labels. It runs at 850–980 agent steps/s. Against the built-in AI it still loses almost everything: over its last 100 such games at 1.49M steps, 1 win, 2 ties, 72 losses against normal and 25 losses against easy.

The same spending evaluation as for the clones shows what changed. Mirror games against the normal AI:

| | BC v3 (the start) | `fgself-3` at 1.48M steps | the AI |
|---|---|---|---|
| wins, ties, losses | 1, 0, 31 | 0, 1, 15 | |
| gold gathered | 6577 | 10230 | ~7500 |
| food at 1 minute | 24.1 | 31.9 | ~47 |
| gold + lumber held | 3951 | 5292 | ~2150 |

It grew its economy, most likely with workers: cheap, safe material that the shaping rewards. It still hoards. Army takes production buildings first, which RL hasn't found yet.

### A curriculum against the built-in AI (`fgself-4`)

`fgself-3` won 2 of its last 100 games against the AI after 3.3M steps (1 tie, 33 losses against normal; 11 ties, 53 losses against easy). From games it always loses, it learns little about beating the AI. Self-play already sits at 50% against itself, but that doesn't carry over.

Neither knob alone brought the clone near 50% against the easy AI:

| setting | wins, ties, losses |
|---|---|
| the clone's units at twice the AI's hit points | 1, 1, 6 |
| the AI starting 30–48 s late | 0, 3, 10, while the clone gathered 2–3× the gold |
| both: twice the hit points, the AI 55–60 s late | 2, 5, 3 |

The curriculum (`League`, `--curriculum`) is a level per AI difficulty that a loss raises by 0.02 and a win lowers:
* From 0 to 0.5 the learner's hit points rise to twice the AI's.
* From 0.5 to 1 the AI also starts late, up to 120 s.
* At 0 it's the real game.

`fgself-4` starts at 0.75 and sends half its launches against the AI. After its first 45 such games it had 9 wins, 5 ties and 13 losses against normal, and 6 wins, 8 ties and 4 losses against easy. The levels settled near 0.71 (easy) and 0.83 (normal). The progress to watch is the levels falling toward 0.

The mean level per 0.5M steps. A single level moves by 0.02 a game, a random walk around the 50% point (±0.2 over ~100 games), so only the averages show the trend:

| steps | easy | normal |
|---|---|---|
| 0–1M | 0.92–0.95 | 0.95–0.97 |
| 1.0–1.5M | 0.89 | 0.89 |
| 1.5–2.0M | 0.74 | 0.72 |
| 2.0–3.5M | 0.65–0.67 | 0.67–0.76 |
| 3.5–4.0M | 0.77 | 0.54 |

In its first million steps the learner needed nearly the most help: twice the hit points and the AI 110–115 s late. From 2M steps it held 50% at about 0.7 (twice the hit points, the AI ~50 s late). It improved against the AI, which `fgself-3` never did, and then held level.

The decline went on slowly, about 0.03 a million steps: 0.57 (easy) and 0.60 (normal) by 8–9M steps. The real game (even hit points, the AI on time) doesn't show it yet. The checkpoint at 8.4M steps in 32 mirror games against the normal AI:

| | clone (BC v3) | `fgself-4` at 8.4M | the AI |
|---|---|---|---|
| wins, ties, losses | 1, 0, 31 | 0, 2, 30 | |
| gold gathered | 6577 | 11353 | ~7400 |
| gold + lumber held | 3951 | 5969 | ~2170 |
| food at 1 minute | 24.1 | 27.8 | ~44 |

It learned to gather, not to spend. With twice the hit points a small army wins fights, so nothing in the curriculum pushed it to build a bigger one, and that is the skill it lacks. `fgself-5` goes on from that checkpoint with the late start alone (`--curriculum-hp 0`, up to 180 s, starting at 90 s) and even hit points, so the fights stay as they are.

With even hit points the levels fell fast from 90 s. `fgself-4`'s learner beat the AI starting 30–45 s late, where the clone had lost 10 of 13 such games. Over its first 2.8M steps the late start needed for 50% was 25–80 s (normal) and 30–75 s (easy), per half million steps. A 0.02 level step (3.6 s a game) makes it a random walk of about ±36 s over 100 games, so from there on the step is 0.01. 10% of the AI launches now play the real game (`--real-share`), and those results are charted as the yardstick.

### The late start is exploitable too: a tax on the AI's income (`fgself-6`)

`fgself-5` held 50% with the AI starting 30–50 s late, but the real game stayed at about 4% (5 wins, 12 ties, 168 losses). Its videos showed why: it attacked the idle AI. 65% of its curriculum wins came before the AI had started.

Two knobs have now been gamed:
* **Twice the hit points** let a small army win, so the learner never had to spend.
* **A late start** left an opponent that doesn't defend itself.

The third knob is a tax (`--curriculum-mode tax`). The built-in AI plays from the start, but each step `play_one` takes a share of what it gathered (and of its starting gold and lumber) with `SetResources`. That makes a poorer opponent that still defends, builds and attacks.

Smoke test against the easy AI at a 45–61% tax: 6 wins, 1 tie, 15 losses in 22 games. In several losses the learner had gathered 2–3 times the AI's gold (22,295 vs 7,845), so this curriculum presses on the skill it lacks: turning gold into an army.

`fgself-6` continues from `fgself-5` at 6.9M steps. It starts at level 0.6 (a 54% tax, up to 90%) with a controller step of 0.01, and 10% of its AI launches play the real game.

After 1.5M steps the learner's race decided most curriculum games. Mirror matchups, the learner's score:

| race | vs the taxed easy AI | vs the taxed normal AI | real game |
|---|---|---|---|
| night elf | 0.94 | 0.96 | 1 win, 2 ties, 3 losses (normal) |
| undead | 0.69 | 0.74 | 0 wins in 32 |
| human | 0.14 | 0.07 | 0 wins in 18 |
| orc | 0.08 | 0.13 | 0 wins in 12 |

One level per difficulty settled where night elf wins balanced human and orc losses. Every race's games were then nearly decided and taught little: the easy AI's level sat at the 90% maximum while human and orc still lost. The curriculum now keeps a level per difficulty and learner's race.

### Human and orc had no lumber: harvest switches were dropped as "redundant"

Per race, human and orc stayed pinned at the 90% maximum tax. Against an AI that kept a tenth of its income, the learner scored:
* **As human:** 1 win, 19 ties, 20 losses, gathering 29,183 gold.
* **As orc:** 3 wins, 24 ties, 13 losses, gathering 44,561 gold.

The videos show the cause: in one human game it held 8225 gold and 187 lumber at 72 s, and 18,731 gold, 445 lumber and 24/24 food at 143 s. Human and orc farms and barracks cost lumber, so it could only hoard gold.

The bug: the rule that drops the AI's re-issued harvest orders (they reset the workers' work) dropped every harvest order to a harvesting worker. A worker's current order doesn't say gold or lumber. So a miner sent to the trees was dropped too, from the labels and at play time, and the clone learned that harvesting workers never switch. In the AI's games, 50% of human harvesting workers are on lumber, 36% of orc and 35% of night elf.

The fix:
* `features.harvest_resource` says what an order harvests.
* The encoder and `BCAgent` track what each worker was last sent to.
* `redundant()` only drops orders to the resource a worker already harvests.
* A new entity feature marks workers on lumber (F 28 → 29; older checkpoints get a zero column).

In 60 games this recovers 148 (human) and 238 (orc) switch labels. `fullgame-rush4` fine-tunes `fullgame-rush3-tk` on the fixed labels: 8000 AI games plus both takeover collections, 2 epochs.

The first version still missed most switches. The starting workers mine by the map's melee setup and new ones by the town hall's rally point, so no recorded order sent them and their assignment was "unknown", which counted as the same resource. `fullgame-rush4` gathered 514 lumber as human (the AI 1880) and 398 as orc (1408). Unknown now counts as gold. That recovers 358 (human) and 365 (orc) switch labels in the same 60 games, and `fullgame-rush5` fine-tunes `rush4` on them for 2 epochs.

`fullgame-rush5` in 32 mirror games against the normal AI, the real game:

| race | wins | lumber (clone vs AI) | food at 1 minute (clone vs AI) |
|---|---|---|---|
| night elf | 4 of 9 | 2853 vs 3215 | 35.8 vs 44.4 |
| undead | 2 of 10 | 3862 vs 1387 | 34.9 vs 39.3 |
| orc | 0 of 5 | 1195 vs 2269 | 25.0 vs 48.2 |
| human | 0 of 8 | 555 vs 1290 | 11.8 vs 36.2 |

That's 6 wins in 32 (19%), where every clone before won 0 or 1. Orc lumber tripled. Human is still broken: little gold (3551) and 11.8 food at one minute. `fgself-7` starts self-play from `fullgame-rush5` with the per-race tax curriculum.

In a human game's video the problem is the army, not supply: 21/36 food and 2860 gold at 39 s, food barely rising, no footmen from the barracks. The human AI's early army kills it at about 70 s (human games last 0.9–1.3 minutes).

`fgself-7` got worse against the AI from its first updates:
* After 1.4M steps all the taxes had risen to 65–90%, night elf included.
* The real game went 0 wins in 60.
* In its first 477 games the curriculum score was 0.14 at a mean 56% tax. The KL to the clone jumped to 0.096 by 250k steps.

The harness plays the same policy as `play.py`: with a learning rate of 0 it won 6 of 64 real games against normal. By race: night elf 3 of 8, undead 3 of 14, human 0 of 20, orc 0 of 22. That matches `play.py` race by race, so RL did the damage.

The likely causes: BC's value head learned the AI's returns, much higher than the clone's, so the first advantages were large and wrong; and half the games were mirror self-play. `fgself-8` starts from `rush5` with:
* 20 value-only updates first
* the KL to the clone at 0.2 (was 0.05)
* a learning rate of 3e-5 (was 5e-5)
* 75% of launches against the AI
* the curriculum starting at level 0.3 (a 27% tax)

`fgself-8` improves in the real game, the first run that does. Its 10% real-game games by race, in fifths of its first 17,400 games (7.2M steps):

| race | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| undead | 13% | 26% | 33% | 53% | 45% |
| night elf | – | 10% | 26% | 28% | 50% |
| orc | 3% | 8% | 11% | 0% | 11% |
| human | 2% | 2% | 4% | 0% | 4% |

That's the share of wins; the rest are ties and losses. The last fifth, for example: undead 27W 1T 32L, night elf 24W 8T 16L. The clone won about 21% (undead) and 38% (night elf).

The taxes fell to match. By 7.2M steps undead needs no tax against normal: the curriculum game is the real game. Night elf needs 25% (normal) and 45% (easy), orc 43–49%, human 47–74%.

From 7M to 11M steps undead and night elf held at about 40–50% real-game wins, and human and orc at about 5%. The Production tab (per game, the last 800 games) shows why human loses in the real game:

| | learner | the AI |
|---|---|---|
| footmen trained | 40.6 | 19.2 |
| peasants trained | 33.8 | 12.0 |
| heroes trained | 0.3 | 1.3 |
| lumber gathered | 696 | 2513 |
| kills | 12.9 | 84.8 |
| losses | 87.6 | 21.1 |

It spends now, but on workers (cheap, safe material that the shaping rewards) and footmen that trade 1:7. The AI fields footmen, riflemen, knights, mortar teams, gryphon riders and heroes, with its upgrades. The learner builds an Altar of Kings in only 0.6 games out of 1. Orc is similar (grunts 30 vs 19, peons 21 vs 12, kills 31.8 vs losses 50.5, in curriculum games). A likely cause is piecemeal attacks: each unit picks its own attack order, so the reinforcements trickle into the enemy army.

An orc loss against the easy AI (at 12M steps) shows it:
* **53 s:** the learner is ahead in material (5722 vs 5172) at 42/50 food, but holds 4093 unspent gold, and its units are spread around its base.
* **76 s:** the AI attacks as one group, with a Far Seer's chain lightning, and wins the fight in the learner's base.
* **100 s:** the learner is down to 8/10 food (2458 vs 6561 material), still with 4529 gold.

Two coordination problems: spending while fighting, and gathering an army when an attack comes. Both are hard for a policy that decides every unit's order separately. At 13M steps night elf reached 0% tax against normal, as undead had at 7M. Both then held about 35–45% real-game wins.

`fgself-8` stopped at 16M steps. Its real games in total:

| race | wins | ties | losses |
|---|---|---|---|
| night elf | 173 | 64 | 205 |
| undead | 166 | 89 | 251 |
| orc | 25 | 27 | 310 |
| human | 19 | 9 | 395 |

Its last few million steps held about 44% (night elf) and 20–40% (undead), and undead's tax crept back up to 25%.

Next is a second DAgger round from its states. `demos-takeover-3` holds 4000 takeover games with `fgself-8`'s last checkpoint as the policy, handing over after 10–60 s: the AI's orders from the states the learner reaches (unspent gold, a scattered army, no altar). With the RL policy playing the first 10–60 s, its side then won about 31% under the AI, whenever the AI took over (10–25 s: 32%, 25–40 s: 31%, 40–60 s: 31%). The clone's side had dropped from 34% to 13% over the same window. So the RL policy no longer loses ground in the opening.

Fine-tuning the checkpoint on these games did not work. The fine-tune ran 1 epoch at learning rate 3e-5 with the value head left alone (`fullgame-rl8-dagger`). 64 mirror games against the normal AI, the real game, same seed:

| | human | night elf | orc | undead | total |
|---|---|---|---|---|---|
| fine-tuned | 0/16 | 4/21 | 0/17 | 2/10 | 6 wins (9%) |
| `fgself-8`, 16M steps | 3/16 | 10/21 (1 tie) | 4/17 | 2/10 (3 ties) | 19 wins, 4 ties (30%) |

Cloning the AI's orders overwrote what RL had learned. Self-play goes on from the checkpoint; the takeover games would have to come in as an auxiliary loss during RL, as AlphaStar's KL to its supervised policy, rather than as a fine-tune. The 30% is also the cleanest measure of `fgself-8`, up from the clone's 19% (`fullgame-rush5`).

### DAgger as an auxiliary loss (`fgself-9`)

`fgself-9` goes on from `fgself-8` at 17.3M steps. It adds the cloning loss on the 4000 takeover games from its own states to each PPO minibatch, times 0.005 (`--bc-data`, `--bc-coef`). The cloning loss fell from 6.7 to about 3.8 in the first 70 updates. After a jump in the first few updates (0.02), the per-update KL settled at `fgself-8`'s level (0.004–0.005). The real game after 1.2M steps:

| race | `fgself-9` | `fgself-8` (in its last millions of steps) |
|---|---|---|
| human | 18W 31L (37%) | ~5% |
| night elf | 17W 5T 9L (55%) | ~44% |
| orc | 6W 1T 15L (27%) | ~7% |
| undead | 8W 9T 26L (19%) | ~35% |

Human and orc, which RL alone never moved, win a quarter to a third of their real games. The takeover games show the AI's decisions from the learner's own states, such as building an altar, training heroes and gathering lumber. The cost is throughput (357 steps/s): the loader workers encode the demonstrations next to the games.


Night elf had almost no real games at first (6 against 46–96 for the other races). The real-or-curriculum draw came right after the race choice, and the actors' seeded random streams correlated them. A replay of those streams gave night elf 12 of 91 real launches. The real game is now a launch kind of its own, and the replay gives 17–24 per race.

A second skew, found on a day of 15 restarts (2026-09-30): a game thread's random stream was seeded by the thread alone, so every restart began with the same launches. Of the real launches against the normal AI, 43 were human's and 6 orc's. The seed now includes the update the run starts from; over 200 simulated restarts every race gets its share (101 to 132 of 923 real launches per race and difficulty).

### Games between agents had no heroes: the scripted reset (`fgself-1` to `fgself-9`)

Found while profiling (next section): the games between agents restarted by script (`GameSetup.melee_reset`: remove every unit, respawn the start; 0.1 s) instead of reloading the map, and that was not a new game.

| per game, `fgself-9` | against the built-in AI (map reloaded) | between agents, scripted reset | between agents, map reloaded |
|---|---|---|---|
| heroes trained, orc | 1.30 | 0.07 | 1.33 |
| heroes trained, undead | 1.69 | 0.13 | 1.91 |
| heroes trained, night elf | 0.80 | 0.08 | 0.91 |
| research started, orc | 5.7 | 0.35 | 5.5 |
| research started, undead | 4.7 | 0.55 | 5.7 |
| food used after a minute, median (lowest) | 41 (3) | 18 (-345) | 39 (0) |

* **Heroes.** The engine keeps counting a removed hero. After one game with a Blademaster the altar refused another and a Far Seer needed a Stronghold (the second hero's requirement); a game later the Far Seer was refused even with a Fortress. `GetPlayerTechCount`, the type and hero limits and the hero tokens all read as in a new game. Handing the hero to the neutral player before removing it changed nothing.
* **Food and research.** A third of those games had less than 5 food used after a minute, and research all but stopped. A simple reset test (units in training, a research under way, a dead hero) reproduced neither: the food count was right after it and the research was accepted again. They were not chased further.
* These were more than half of all games (the learner against itself and against past snapshots), in every run so far.

Every game now reloads the map (`--scripted-reset 0`, the default), in `fgself-9` from 9.4M steps on (the table's last column). The lesson for environment changes: compare per-game statistics by kind of game (heroes, research, food), not only win rates.

The real game (the built-in AI without the curriculum's tax, 10% of the launches against it) before and after:

| wins in the real game, `fgself-9` | all | human | night elf | orc | undead | easy AI | normal AI |
|---|---|---|---|---|---|---|---|
| the 4 hours before the fix (to 9.4M steps) | 252/642 (39%) | 30% | 80% | 21% | 25% | 34% | 45% |
| the 2 hours after it | 89/223 (40%) | 30% | 83% | 12% | 25% | 36% | 45% |
| 2 to 4 hours after it (11M to 15M steps) | 309/559 (55%) | 43% | 89% | 41% | 40% | 48% | 62% |
| 5 hours after it (17M to 19M steps, five games per load) | 214/375 (57%) | 50% | 88% | 39% | 49% | 50% | 64% |
| 21M to 24M steps | 242/407 (59%) | 50% | 81% | 62% | 43% | 50% | 69% |
| 30M to 33M steps | 276/426 (65%) | 51% | 88% | 74% | 54% | 56% | 73% |
| 39M to 42.6M steps (the end, stopped for a shutdown) | 390/522 (75%) | 64% | 91% | 73% | 70% | 72% | 78% |

The races' columns are the measure: the restarts of that day skewed the mix of races in "all" and in the difficulties' columns (see the note on seeds above). Nothing else about the training changed in between (the speed work below changed how fast the same games are played), so the gain is the games between agents being whole games, or training that would have come anyway; there is no control run.

Over the run's last 20M steps human still trained almost no heroes (0.03 a game), and undead fewer (1.7 → 1.2) while it won more. `fgself-9` played mirror matchups only (`--mirror 1`, as every run so far). The next run (`fgself-10`) goes on from its last checkpoint with mixed matchups: a curriculum level per matchup against the built-in AI and a tax between races in games between agents (`--mirror 0`, `League.balance`).

### Mixed matchups (`fgself-10`)

`fgself-9`'s last checkpoint (42.6M steps, mirror matchups only) in 320 real games against the normal AI over all 16 matchups (`play.py --race all --ai-race all`), the learner's race by row:

| learner \ AI | human | orc | undead | night elf | all |
|---|---|---|---|---|---|
| human | 59% (22) | 29% (17) | 50% (18) | 0% (26) | 33% |
| orc | 65% (26) | 85% (13) | 71% (17) | 7% (14) | 59% |
| undead | 48% (25) | 11% (27) | 50% (20) | 0% (16) | 28% |
| night elf | 88% (25) | 82% (17) | 87% (15) | 91% (22) | 87% |

70% in the mirror matchups it trained on, 44% in the others. Against the night elf AI the other races won 1 game in 56: night elf is the strongest race on `duelrush` (78% of the built-in AI's games). `fgself-10` goes on from this checkpoint with `fgself-9`'s recipe and mixed matchups (`--mirror 0`):
* the curriculum against the built-in AI keeps a level per difficulty and matchup (32), starting at 0.3 (a tax of 27% on the AI);
* games between agents of two races tax the stronger race's income (`League.balance`): a level per pair of races that the learner's games against itself move towards a 50% score (0.02 a game, a tax of at most 90%); games against past snapshots pay the same tax. In the first 150 games night elf's tax against human rose to 11%.

**The tax was a subsidy at first.** By 8.6M steps every pair's level had run to the end of its range, and the taxed side still won: night elf, taxed 90% against every race, won 76% of those games. A taxed night elf trained as many units as its opponent (36 against 38 a game) from a tenth of its income. The tax was the last of a step's commands: it set the taxed player's gold and lumber to what the observation showed minus the tax, which undid what that player's orders earlier in the same commands had just spent. A taxed agent trained and researched for free. The tax on the built-in AI was not affected, because the AI spends between steps. Now the tax is a step's first command, and the taxed agent sees what it has left. `fgself-10` went on from 8.6M steps with the levels back at 0. In those 8.6M steps about 40% of the games (the mixed games between agents) gave one side free production.

`fgself-10` ran to 38.7M steps (stopped for a shutdown, 2026-10-02 00:00). The real game, by steps:

| steps | all | mirror | mixed | human | orc | undead | night elf | vs human AI | vs orc AI | vs undead AI | vs night elf AI | easy AI | normal AI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0–8.6M (the tax a subsidy) | 73% | 79% | 70% | 55% | 80% | 57% | 98% | 79% | 63% | 84% | 66% | 66% | 78% |
| 8.6–20M | 75% | 86% | 72% | 52% | 86% | 66% | 97% | 84% | 71% | 83% | 62% | 72% | 78% |
| 20–30M | 80% | 86% | 78% | 60% | 89% | 71% | 98% | 86% | 80% | 83% | 70% | 81% | 79% |
| 30–38.7M | 79% | 87% | 76% | 57% | 89% | 77% | 98% | 85% | 77% | 87% | 67% | 78% | 80% |

By matchup from 30M steps on (real games against both difficulties, about 40–100 a cell):

| learner \ AI | human | orc | undead | night elf |
|---|---|---|---|---|
| human | 69% | 57% | 70% | 36% |
| orc | 90% | 97% | 90% | 81% |
| undead | 89% | 67% | 90% | 61% |
| night elf | 96% | 95% | 100% | 100% |

Mixed matchups went from 44% (the start's 320 games against the normal AI) to 76%; the cells that were near zero rose most (undead against the orc AI 11% → 67%, orc against the night elf AI 7% → 81%, undead against it 0% → 61%, human against it 0% → 36%; the start's numbers are from the normal AI alone). Human stayed the weak race at 52–60% and trained almost no heroes (0.02–0.05 a game) all run. At the end the taxes between races stood at 25–58% on the stronger race (the most: orc against human 54%, night elf against human 58%), and the curriculum still eased only human's games (against the night elf AI 0.3–0.36, the easy orc AI 0.12).

Resumed on 2026-10-02, it got the insane AI as a third difficulty from 40.4M steps (`--ai easy,normal,insane`, a curriculum level per matchup starting at 0.3). In the real game against the insane AI it won 16 of its first 30 games. It stopped at 41.4M steps to move on to longer games.

Human's footmen and riflemen without heroes are a duelrush strategy: human built 0.02 altars a game against the human AI's 1.0, and its games with a hero were won less often (43% against 54%, 10k games). Two-minute games do not pay for a hero.

### Longer games: `duelfast` (`fgself-11`)

`duelfast` is the duel map with the game's own speed, a third of the build, train and research times, and half the costs and hit points (no extra harvest). 48 built-in AI games (`runs/fullgame/probe-duelfast`, normal and insane, all races) lasted 5.9 minutes on average (median 5.4, longest 12.7, no ties within 15). The insane AI beat the normal AI in 20 of 23 games. Between different races orc won 17 of 22 and night elf 11 of 14, human 6 of 25 and undead 4 of 15 (few games).

`fgself-11` goes on from `fgself-10`'s last checkpoint on `duelfast`, with:
* a memory core (a minGRU added as a no-op, `--memory 1`): the games are three times longer and the enemy is out of sight most of the time;
* a longer horizon: gamma 0.999 instead of 0.997 (about 8 minutes instead of 3), and 10 updates of the value alone first;
* the auxiliary cloning loss on new takeover games on `duelfast` from `fgself-10`'s policy (`demos-fast-takeover-1*`: the policy plays one side for 10 seconds to 5 minutes, then the built-in AI takes over);
* the rest as `fgself-10`: mixed matchups, the curriculum per matchup and difficulty (easy, normal, insane), the tax between races, the clone's KL term (now towards `fgself-10`'s policy).

The takeover collection was bound by its own Python (24 games' policy calls in one process; the machine 65% idle), so three processes collected side by side.

`fgself-11` ran 14.8M steps (5 hours) and stalled:

| | the first hour | hours 3 to 5 |
|---|---|---|
| curriculum games won (the AI taxed ~0.65) | 1–3% | 31–34% |
| tied at the time limit | 17% | 33–41% |
| the real game | 0 of 80 | 4 of 346 in all (1%) |

Fixes on the way:
* **The clone's KL term spiked** (96 and 4.4, gradient norms of 726 and 39 in two updates): the cloning loss on the new demonstrations allowed new (unit type, order) pairs to the learner but not to the clone, whose log-probability for them stayed at -1e9. The clone now gets the learner's allowed orders.
* **Games tied at 15 minutes** (59% of the curriculum games at one point, the learner ahead in material in most), and a tie leaves the curriculum's level alone: 20 minutes now.
* **Six games per launch** kept a launch's races and opponent for 10 minutes of duelfast games: the real games came in streaks of one matchup. Two now.

What it did in the real game, against the AI, per game (the last 150): 32.6 workers trained against 10.1, an army of 23 against 28, 38 food at most against 60, 0.1 heroes against 1.6, almost no research (the AI researched in a quarter of its games), 8 kills against 42. These are duelrush habits: there, walking workers carried 6.5 times the gold, so more workers paid, and a game was over before heroes and research paid. At the AI's own hero decisions in the duelfast demonstrations the policy gave the hero order 0.6% (`fgself-10`: 2.5%, orc 8%), and a stronger cloning loss (×0.02 for an hour) did not move it.

The clone's KL term held the policy to `fgself-10`'s, a duelrush policy: its distance stayed at 0.002 a unit for the whole run. AlphaStar's KL term is towards a supervised policy of the same game. So the next run starts from a clone of the built-in AI on `duelfast` (`fgself-10`'s network fine-tuned on 3000 built-in AI games on `duelfast` and the 1017 takeover games), with that clone as the anchor.

The clone, `fullgame-fast-1` (3 epochs, 46 minutes): validation loss 2.48 → 2.38, the AI's order 77% of the time, its target 94%, the value explaining 90% of the returns' variance. It lost all 45 real games it played against the normal AI (6 minutes on average, 15 food at one minute against 21).

`fgself-12` (from the clone, the clone as the anchor; memory core, gamma 0.999, 20-minute games, 2 games per launch, the cloning loss on all the duelfast demonstrations) ran 12.9M steps (4 hours). Its habits stayed the AI's: 1.2–1.4 heroes a game against the AI's 1.4 (`fgself-11`: 0.1), about one altar a race, 14–15 workers against 10 (`fgself-11`: 32). But it did not learn to win:

| | first hour | last hour |
|---|---|---|
| curriculum games won | 4–9% | 1–4% |
| tied at the time limit | 14–29% | 78–86% |
| the AI's tax | 0.32–0.43 | 0.67–0.76 |
| the real game | 1 of 234 in all | |

* **Ties.** In the tied games it was ahead in material (lead 1.6, ahead in 79%), with armies even and kills even, and the AI kept ~11 buildings. A tie with that lead was worth ~+0.5 against +1 for a win: `--tie-break 0.1` (from 5.7M steps). Ties went on rising.
* **The curriculum stood still**: a tie left a level where it was, so with 78% ties the AI's tax stayed at 0.67 for an hour. `--curriculum-tie 0.5` (a tie raises a level by half a loss's step, from 10.3M): the levels rose from 0.72 to 0.88 on average (26 of 48 at the most) in an hour; the curriculum games won only 4% at a tax of 0.76.
* **It does not spend.** It ended games holding ~3,600–3,800 gold and lumber (the AI ~800), gathered 40% less gold than the AI with half again as many workers, and trained fewer basic units than the AI (21.7 footmen against 33.7, 22.7 ghouls against 35.4) but more supply buildings (1.7 times) and some high-tech units. Its material shaping counts units and buildings but not what is held, so spending was already rewarded.

**Why it holds its resources** (`scripts/production_probe.py`: games on the CPU, the policy's production buildings looked at every step). Its barracks (and ancients of war, crypts, the great hall) were busy 23–42% of the time and idle with something affordable 50–75%; there the policy trained in 4–9% of the steps. The clone `fullgame-fast-1` was the same: the built-in AI keeps its production queued, so its games hardly show an idle barracks with money in the bank, and the clone did not learn what to do there. The KL term (×0.2) then held the policy to the clone's rate: from 5% to 50% at such a building is ~0.8 nats a step, against a few hundredths of shaping reward for the unit it buys.

From 13.0M steps `fgself-12` goes on with the held gold and lumber counted against the shaping's potential (`--float-penalty 0.5`: spending is credited at the order, not ~14 steps later when the unit appears, half the credit after GAE's lambda) and the KL term ×0.05.

The KL weight ×0.05 let RL lower the production orders: the chance the policy gave the AI's own hero orders at the AI's decisions (`scripts/order_probability.py`) fell from 7.8% (the clone) to 2.7% at 14.1M steps, its basic units from 39% to 20%. From 15.07M steps the KL term is ×0.2 again and the cloning loss ×0.02: 5.9% and 32% at 16.9M.

| `fgself-12`, 30-minute windows | 60–90 min before 16.9M | 30–60 min | the last 30 min |
|---|---|---|---|
| curriculum games won | 3% | 5% | 9% |
| tied | 95% | 90% | 83% |
| lost | 2% | 5% | 7% |

At 17.8M steps (5.5 hours) it had won 2 of 334 real games. Games lasted ~19 of 20 minutes at a tax of 0.84. It held ~4,000 gold and lumber against the AI's ~330 and had the larger army (28 food against 17), but killed 7 units a game and lost 15: it out-builds the AI and does not attack.

**The real game is not a tie.** The untaxed AI wins in 5–7 minutes. The last 150 real games against the curriculum games of the same hours:

| per game (learner vs AI) | real, normal AI | real, insane AI | curriculum (tax 0.84) |
|---|---|---|---|
| result | 49 of 50 lost | 62 of 62 lost | 83% ties |
| minutes | 6.7 | 5.3 | 19 |
| food at 1 minute | 15 vs 21 | 16 vs 20 | 15 vs 7 |
| most food | 34 vs 53 | 31 vs 62 | 72 vs 34 |

The tax cuts the AI to 7 food at one minute, so the curriculum hides the policy's slow opening. The clone was already behind (15 against 21). Many matchups were at level 1.0 (the AI without income) and the policy still tied them.

**Learn from short wins** (AlphaStar fine-tuned its supervised policy on winning replays). Of the 3,000 built-in AI games on `duelfast`, 13 were ties and 55% were decided within 6 minutes. `fullgame-fast-win6`: the clone fine-tuned on the winning sides of the games decided within 6 minutes (2,180 of 8,034 game sides, takeover games included; `bc --winners-only --max-minutes 6`). `fullgame-fast-win6` (4 epochs, the best the third): validation loss 2.314 on held-out short wins, the AI's order 81% of the time.

`fgself-12` goes on from 17.77M steps with it as the KL anchor and its data as the cloning loss (`--bc-winners 1 --bc-max-minutes 6`). The cloning data also gets takeover games from `fgself-12`'s own states, with the AI taking over at 0.5–5 minutes, before the real games are lost (`demos-fast-takeover-12`).

After the switch (17.77M to 21.0M steps, 1.5 hours): curriculum games won 9% → 24%, tied 83% → 69%. The real game: still no win in 46 games, but games lasted 9.7 minutes instead of 7.6 and the policy killed 12 units a game instead of 7.5. The chance of the AI's hero orders: 6.3% (the old clone 7.8%, `fullgame-fast-win6` 8.6%), basic units 30% (39%, 39%). The takeover collection ended with 600 games: 512 taken over (the rest lost before the takeover), 102 won by the taken-over side within 6 minutes of it. From 21.0M steps the cloning loss reads all 600.

**After 50 minutes of distillation** (24.13M → 25.65M steps): the real game (never advised) 13 of 50 won (26%: easy 5 of 14, normal 7 of 18, insane 1 of 18; before: 1 of 98), 6% ties, 8.5 minutes a game, food at one minute 19.0 against 21.6. Unadvised curriculum games 82% won with the AI's tax falling 0.84 → 0.60; unadvised self-play 13% ties (82% before). The policy agrees with the advisor's order 59% of the time; its KL per update 0.003, its distance to the anchor 0.006.

**After 2 h 50 min of distillation** (31.7M steps, 17:49): the real game 55 of 227 won (24%: easy 33%, normal 35%, insane 9%, insane 18% in the last 30 minutes). The chance of the AI's own orders at its decisions: basic units 40% (30% at 20.9M; the clones 39%), heroes 7.2% (6.3%; the clones 7.8-8.6%): distillation brought production back to the clones' level. The curriculum's tax fell to 0.31.

**A plateau after 3 hours** (33.4M steps): the real game held at ~24% in every half hour since the first (normal 30-40%, insane ~8%), and the agreement with the advisor at ~58% (the distillation loss 2.6, flat). The real games are lost on tempo:

| real games against normal and insane (learner vs AI) | won (50) | lost (171) |
|---|---|---|
| food at 1 minute (normal) | 21.4 vs 20.5 | 16.9 vs 22.5 |
| tier 2 reached | 100% vs 50% | 63% vs 91% |
| heroes | 2.0 vs 1.3 | 1.2 vs 2.0 |
| minutes | 8.3 | 6.2 |

In games of 6 minutes or more both sides reach tier 2 (94% and 86%): the learner techs, but late. It trains 1.3-2.5x the AI's workers a minute (human 4.4 vs 2.3, undead 1.3 vs 0.5) and holds twice the resources (2,800 vs 1,300). Most losses to insane come in minutes 3-5. Human wins 2 of 46 such games, orc 14%, undead 23%, night elf 39%. From 33.4M steps the distillation loss weighs x0.15 (was x0.05).

**3 h 20 min at x0.15** (33.4M → 46.4M steps, 21:56): the real game 95 of 366 won (26%: easy 34%, normal 40%, insane 4%), so no change. The agreement with the advisor rose 58% → 64%; the KL per update stayed at 0.0033, the distance to the anchor rose 0.006 → 0.0095. Against normal the tempo is now even, against insane it is not:

| real games, learner vs AI | normal (118) | insane (120) |
|---|---|---|
| food at 1 minute | 19.1 vs 19.2 | 18.5 vs 20.9 |
| tier 2 reached | 86% vs 73% | 72% vs 94% |
| heroes | 1.8 vs 1.6 | 1.4 vs 2.2 |
| workers per game minute | 1.55 vs 1.11 | 2.21 vs 1.35 |
| kills / units lost | 0.70 | 0.30 |

The losses are decided in the fights: in real losses against normal and insane the learner trained about the AI's army (27 vs 25 units) but killed 8.8 units for 40 lost (in wins 39 for 17). The chance of the AI's own orders at its decisions (46.35M): heroes 5.1% (4.9% at 40.8M, 7.2% at 31.6M), basic units 33% (40% at 31.6M). By race the heroes are human 3.1% (the clones 11.5-12.6%), undead 1.8% (4.9-5.9%), night elf 4.6% (5.1-5.3%), orc 9.8% (8.4-9.6%); human basic units 31% (45-46%). Those are the races that lose: against normal and insane human won 2 of 48, undead 8 of 62, orc 13 of 66, night elf 29 of 62.

**Without the cloning loss** (from 58.0M steps, 2026-10-10 19:11; `--bc-coef 0`, was 0.02). The distillation teaches the built-in AI's orders at the policy's own states; the cloning loss taught them at the AI's states in its winning games, at a third of an update's time. The KL anchor to `fullgame-fast-win6` stays (0.2). Before it, since the distillation weighs x0.15: the real game 150 of 589 won (25%: easy 31%, normal 38%, insane 6%; human 2 of 75 against normal and insane). What to watch: the real game, the chance of the AI's production orders at its decisions, the distance to the anchor. At 58.03M, just before: heroes 6.0% (human 7.5%, undead 3.3%; at 46.35M 5.1%, human 3.1%), basic units 33% (human 39%, orc 23%; the clones 39%).

### The built-in AI as an advisor (shadow games)

Takeover games label only the states after the takeover, and they soon become the AI's states. On-policy distillation asks the teacher for a label in every state the student reaches. The built-in AI is a script, not a function, so it cannot be asked. In a shadow game it plays the policy's player too: the harness records its orders as labels and undoes them (`protocol.ShadowAI`, `collect.py --shadow`).

| undo | how |
|---|---|
| training, research | cancelled when they were paid for (the player's resources before and after the order) |
| build orders, orders to the agent's builders | the agent's last order again, at once |
| a hero skill | unlearned, the point back |
| town bell, burrows | switched off |
| any other order | stands until the step ends; then the agent's last order again, before the observation |
| the engine's own (a worker going back to its mine, autocasts), a trained unit's rally | not touched |

Each version against the same 16 games without the advisor (`fgself-12` at 20.9M, `duelfast`, normal and insane AI):

| version | food at 1 min | food at 2 min | gathered by 2 min | what was wrong |
|---|---|---|---|---|
| no advisor | 16.6 | 26.4 | 1,412 | |
| undo everything at once | 12.7 | 18.2 | 1,266 | units the AI ordered every tick lost their attack swings: 78 deaths by 2 minutes, not 51 |
| other orders undone at the step | 15.9 | 20.5 | 1,284 | human buildings finished: 1.2 a game, not 4.2. The AI pulled builders away. |
| builders undone at once | 14.4 | 19.5 | 1,262 | the restored builder's order is "repair", so the policy gave it another order |
| the policy keeps repairing builders | **17.4** | **25.7** | **1,402** | |

The labels cover 5.9% of unit-steps (the AI's own games: 3.5%), 770 train or research orders and 816 build orders in the 16 games. `demos-fast-shadow-12`: 800 games from `fgself-12` at 23.7M steps.

**Distillation in self-play** (on-policy distillation: the teacher labels the student's own states while it learns, so the labels never go stale). From 24.1M steps, `fgself-12 --opd-share 0.5 --opd-coef 0.05`: in half the games (never the real game) the insane AI advises the learner. Its orders of a step label that step (`features.Encoder.step_labels`, the same code as the demonstrations' encoder). The update adds the cloning loss on those steps, from the PPO pass's outputs. First updates: 45–60% of a batch's steps advised, the distillation loss 2.6–3.0, the policy's first choice the AI's order 51–59% of the time. Steps/s unchanged (750–800). The policy's KL per update 0.004–0.006 (0.001 before), its distance to the anchor 0.005 (0.002). The 28 offline shadow games collected are kept, unused.

### Where a self-play step's time goes (speed, 2026-09-30)

`fgself-9` ran at 560 agent steps/s with 32 games. A game thread spent 55% of its time waiting for the policy and 36% for the game. After this round the learner is the limit with the machine's CPU close behind, and it runs at about 840 steps/s with whole games, five to a load of the map (607 before the reset fix, when half the games were the cheap broken ones).

| change | agent steps/s | what it showed |
|---|---|---|
| start | 560 | the inference server 90% busy at 3.4 rows a call |
| a pipe per game thread instead of queues | 545 | a request took ~32 ms of a 59 ms step, the server's call 4.5 of them: feeder threads, a shared write lock, a reader thread, each a wait for a GIL |
| games niced (+10) | 595 | the server and the learner get the CPU first: a call 4.2 → 2.9 ms, an update 11.4 → 9.5 s |
| a round's calls started together, one wait | 606 | next to the learner every wait costs a turn of the GPU (a call: 0.8 ms alone, 1.7 next to a synthetic learner, 2.5 in the run); a round 3.9 → 2.8 ms |
| CUDA graphs captured by hand | 607 | 0.4 s a shape instead of torch.compile's 10-20 s (the games waited); a new league member no longer stalls everything for 25-50 s |
| every game reloads the map (the fix above) | 540 | the games between agents are real games now (47 ms a step, not 32) |
| the loading screen drawn in the game's thread, a video every 10 minutes | 644 | a reload's CPU 2.1 → 1.4 s |
| the pass over the units made by the shim (C) instead of the harness (JASS) | 719 | a step 41 → 36 ms; the learner is the limit again (an update 10.5 s, the actors waiting 1 s) |
| minibatches of similar entity counts, no GPU waits for the statistics | ≈800 | an update 8.9 s; the learner is bound by its own Python (the GPU is 36% busy) |
| five games per load of the map, each with two players of its own | 822 | the actors now make more steps than the learner trains on: an update 9.3 s, the batches 3 updates old and falling behind |
| the trajectory queue holds one batch; an inference round at most every 4 ms | 837 | an update 8.7 s, the batches 1.3 updates old, the learner waiting 1.1 s an update |

* **The cloning batches.** The loader takes its workers' batches in turn, and a worker reading 16 games before it emptied them gave nothing for seconds: the learner waited 26% of an update. `bc.Steps` is now a shuffle buffer kept full (9%).
* **Only 58% of the games' time was play.** A reload took 13 s with 32 games running (5 s alone), as long as the game before it.
  * "Allow Local Files" (a registry setting the harness needed when it read its actions from a file) made the game look in its folder for every file before its archives; under Wine each miss scans the directory. Off: 13.2 → 9.9 s.
  * The rush rules were map object data (`w3u`, `w3q`, `w3a`), which the game applies change by change at every load: 2 of 3 s. They are now the game's own tables with the rules' values, in the map: a reload takes 1.0 s alone, as on the plain `duel` map. Checked cell by cell against the object data (a unit test) and with 240 built-in AI games on each map (2.10 and 2.09 minutes; the races' win rates within noise).
* **The machine is full.** A game alone steps 68 times a second at 14.6 ms of CPU a step. With 32 games running, the same step costs 35-40 ms (two threads a core, the all-core clock, contention), and 16, 32 and 48 games step 418, 638 and 618 times a second in all. A reload per game is a quarter of the games' CPU.
* **Dead ends.** The inference call as a thread of the learner (one GPU context): capturing a CUDA graph fails while another thread draws random numbers. Emptying melee's preload lists: the shim's `Preload` hook already drops them. A slower clock while reloading: the loading screen waits about two game seconds drawing frames, so it costs more.

* **The units, in C.** Every step the harness went over every unit three times in JASS (collect them, count them for the result, write the records of those that changed): ~35 native calls and a few hundred JASS instructions per unit, 9-13 ms of a step with 150 units (4 ms on an idle machine). The natives are plain functions in the executable, found by name from the code that registers them, so the shim now makes the same calls itself: 4 ms (1.5). A game played with the harness's pass and its replay with the shim's agree in every field of every unit on every step (six games, both directions).
* **The learner's minibatches.** A minibatch pads to its widest step: 91 entities at random for a mean of 41, and attention costs the square. Cut from groups of 8 minibatches sorted by entity count they pad to 49. The update's KL, clip fraction and value loss stayed the same (0.0033, 0.024, 0.018); the gradient norm rose from 0.15 to 0.19.

* **Five games per load of the map.** A reload per game was a quarter of a game thread's time. The scripted reset's trouble is per player, so the duel maps are now also built for five pairs of players (`GameSetup.pairs`): each game of a load is played by two players who have not played, the next game starts 0.1 s after the last one's end, and the fifth game's end reloads the map. Python still sees players 0 and 1. Twelve players crashed the game at load; ten and the observer load.
  * The next pair's game has to start as a load's does. The first version gave the first observation one step late: the workers already on their way, the built-in AI's starting gold spent, the agent's first orders lost. Win rates hardly moved in 25 minutes, but orc built 0.60 altars a game instead of 1.57 and trained 0.4 heroes instead of 1.4, night elf alike: the policy's opening depends on its first observation. The per-game statistics by race showed it within minutes (the lesson of the reset above, applied).
  * The switch now takes three ticks of the game, as a load has them. Everything of the last game is removed (units, corpses, items; trees regrow, blight clears). A tick later, the gold mines the engine put back from under removed haunted and entangled mines are removed too, and the map's mines and creeps and the new players' starting units are made. Another tick later the units are collected as an enumeration finds them, and the first step follows at once.
  * Checked at three levels. Every unit, player and event of a later pair's first six steps against a load's, for all four races against the undead and night elf AI: the differences are those between two loads. Built-in AI games, 240 with five pairs and 240 with a reload each: 2.11 and 2.18 minutes, the races' win rates alike; and 298 games by their place in a load: food at step 80 within 1 of the matchup's mean at every place. The policy's altars, heroes and research per game by race in the run (orc 1.54 altars and 1.40 heroes again).
  * Collecting demonstrations (built-in AI on both sides, no network calls) runs 1.55 times as fast: 2700 against 1750 games an hour, next to the training run.
* **The learner and the actors in balance.** With the reloads gone the actors made more steps than the learner trained on. The trajectory queue had no bound, so the batches were 3 updates old and falling behind; it now holds one batch and the actors wait. Every round of the inference server takes the GPU from the learner for a turn, so a round now starts at most every 4 ms (`--infer-period-ms`; 174 rounds a second instead of 229, for the same rows): an update 9.3 → 8.7 s.
* **Another dead end.** The learner and the inference server on cores of their own (`--pin`): the update was no faster (9.0 s) and the games lost 4 threads.

What is left, largest first: the learner (an update takes 8.7 s for 8192 steps, so 940 steps/s at most; it is bound by its own Python: a compiled update, or fewer passes such as the clone's logits once per update), the video renderer (3.4 cores while it renders), and wineserver (a tenth of a step: the engine's threads signal each other ~500 times a step).

### Memory and a value head in BC

Two additions to cloning:
* **A memory core:** a minGRU, scanned in parallel (`--memory`).
* **A value head:** it learns self-play's returns from the recorded games.

380 games, 3 epochs, the same 20 validation games. Every model is validated on whole game sides.

| model | validation loss | order loss | order accuracy | target | point error (bins) | value explained variance |
|---|---|---|---|---|---|---|
| no memory | **3.985** | **0.328** | **61.4%** | **76.2%** | **8.65** | 0.854 |
| memory, 8 lanes × 32 steps | 4.089 | 0.337 | 59.8% | 75.4% | 9.00 | 0.863 |
| memory, 32 lanes × 8 steps | 4.106 | 0.339 | 60.2% | 74.0% | 8.99 | **0.868** |
| memory, 32 lanes × 16 steps (batch 512) | 4.283 | 0.364 | 56.3% | 73.9% | 9.45 | 0.862 |

What the table shows:
* Memory predicts the value a little better and the orders worse. Batches of consecutive steps vary less than batches of shuffled steps, and the AI's decisions depend mostly on what it currently sees.
* The value head explains 85% of the variance of self-play's returns in held-out games, so self-play starts from a trained value.
* Clones stay without memory for now. Self-play can add a core that starts as a no-op (`--memory 1`).

An earlier comparison validated each model on the chunks its own training loader made. Those loaders stop when their first lane runs out of games, which favored early-game steps, so the numbers looked very different (3.39 vs 3.99).

### Race balance on `duelrush`

In 3389 demonstration games night elf won 96% of its games against the other races, undead 68%, orc 26%, human 5%. The cause is the economy: the speed rules make mining and chopping 7 times faster, but walking only 1.3 times (the engine's movement limit). Human and orc workers walk every load of gold to the town hall, so their gold income rose 1.6 times; night elf and undead gold comes from the mine with no walking and rose 7 times. Gold mined in the first minute: human 571, orc 588, undead 3230, night elf 4005 (all four within 10% of each other at normal speed).

| rules | human | orc | undead | night elf | game minutes |
|---|---|---|---|---|---|
| `duelrush` | 3% | 30% | 64% | 96% | 2.1 |
| `duelrush`, handicap 100 (units at half hit points, not a quarter) | 13% | 27% | 76% | 100% | 1.7 |
| `duelrush`, buildings at full hit points | 0% | 0% | 86% | 100% | 1.9 (24 games) |
| `duelrush`, walking workers carry gold ×4.4, lumber ×2.5 | 25% | 50% | 29% | 89% | 2.1 |
| `duelrush`, gold ×6.5, lumber ×4 (now the default) | 37% | 57% | 29% | 78% | 2.1 |
| `duel` (normal rules on the small map; 42% ties at 15 minutes) | 31% | 52% | 16% | 9% | 12.4 |

Win rates against the other races, 96 games each unless noted. What a walking worker carries per trip is now a rule (`Rules.gold_carry`, `lumber_carry`), and `duelrush` has gold ×6.5 and lumber ×4 (map version v11; collections record the map file). Even normal rules on the small map are not balanced, so this is where tuning stopped. Self-play starts on mirror matchups, which are balanced by construction.

## Other findings

* lr 0.01 (tuned on `footmen2`) is far too high with 15 action heads: the KL per update was 1.0-1.5 (clip fraction 0.9) and the win rate peaked at 29%. The KL at a given lr grows with the number of heads (≈0.15 with 6, 0.3 with 9, 1.0+ with 15); lr 0.003 keeps it at 0.03-0.14.
* Action masks: before masks, 94% of the from-scratch policy's casts on `mirror_mix_abil_hp400` were impossible. With them, it reached 39% at 0.9M steps against 35% without.
* Updates per epoch are `replay_ratio × batch / minibatch`. With the minibatch equal to the batch (the old default), there was one update per epoch and learning was slow: `footmen2` reached 61% wins in 1M steps, against 97% with 16 updates.
