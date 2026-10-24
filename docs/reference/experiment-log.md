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

Without it (58.0M → 72.8M, 3 hours): the real game 100 of 443 won (23%: easy 28%, normal 30%, insane 9%), against 25% before (not a clear difference). But the chance of the AI's hero orders at its decisions fell steadily, 6.0% → 5.1% (62.1M) → 4.0% (72.2M; human 7.5% → 3.2%, undead 3.3% → 1.9%, no order at all 76% → 85%), while basic units held (33-35%). In its own real games it still trained as many heroes (1.4-1.5 against insane's 2.2). So only the cloning loss had kept the AI's habits at the AI's own states. From 72.8M (22:10) it is back at x0.02 on half the batch (32 steps a minibatch: an update 2.24 s alone, against 2.64 s with 64 and 1.59 s without).

With it back (72.8M → 81.8M, 2 hours): the real game 60 of 250 (24%: easy 24%, normal 44%, insane 6%), the AI's hero orders 4.5% at 81.4M (from 4.0%; human 4.6%).

**The anchor halved** (from 82.8M, 2026-10-11 00:17; `--ref-kl 0.1`, was 0.2). For 40M steps the policy sat ~0.01 nats from `fullgame-fast-win6` and the real game at ~24-25%; a weaker pull lets the reinforcement learning move it further. Watch: ties and held resources (this run hoarded and tied without an anchor), the distance to the anchor, the AI's production orders, the real game over 300+ games.

Stopped for a shutdown at 85.9M (00:55) after 30 minutes: the real game 21 of 93 (23%), ties as before (self 18%, past 26%), resources held at a real game's end 3,261 (3,506 before; the AI ~1,390), the distance to the anchor 0.010 → 0.014. Not judged yet: it resumes from `0000000085906150.pt` with the same flags.

**The anchor at 0.1, judged** (302 real games, 82.8M → 93.1M): against normal and insane 42 won where the built-in AI would win ~96 in the same matchups (**44% of its record**, as before: 43-45%), but ties rose in every kind of game: self-play 18-22% → 27%, past snapshots 20-25% → 31%, unadvised curriculum ~15% → 27% (resources held at a real game's end stayed lower: ~3,200 against ~3,500). No gain for the run's old drift towards ties: back at 0.2.

**A quarter of the AI games real** (from 93.1M, 2026-10-11 12:47; `--real-share 0.25`, was 0.1, with the anchor at 0.2): more of the training against the untaxed built-in AI, the yardstick's own opponent. Judged on 300+ real games against the AI's record in the same matchups (43-45% before), ties and the curriculum's tax.

Judged after an hour (93.0M → 98.1M, 328 real games): 35% of the AI's record against normal and insane, 28% after the first half hour (13 won where the AI would win 46), against 41-45% in every earlier window; back at 0.1 from 98.1M (13:48). Ties had been creeping up since the evening before, independent of these changes: self-play 17-19% (18-21h on the 10th) → 20% (22h-0h) → 26-30% (11-13h on the 11th) → 33%, past snapshots 19-20% → 35%, unadvised curriculum 13-15% → 25-28%; real games stayed at 7-16%. The run is watched for whether ties and the yardstick recover with the recipe as before both experiments (anchor 0.2, real share 0.1).

**Rolled back** (2026-10-11 14:16): the run reached its 100M target at 14:09, the last hour at 28% of the AI's record and ties at 30-37%. It goes on to 200M from the policy before both experiments (`0000000082830609.pt`, 44-45% of the AI's record, ties ~20-24%; `selfplay_restart.py --weights`), with the recipe as then (anchor 0.2, real share 0.1, the cloning loss on 32-step batches, 48 games). If ties rise again from there, the suspects are what changed the evening of the 10th (48 games and 1.8 instead of 1.3 updates of staleness from 21:17, the cloning batch 32 from 22:10).

They did: from the roll back, ties rose again within 1.5 hours (self-play 22% → 30%, past snapshots 23% → 31%, curriculum 16% → 27%), so the drift comes from the training as it is now, not from the policy. Between 19:10 and 21:17 on the 10th (40 games, no cloning loss) they held at 17-19%. **Test 1** (2026-10-11 16:23, from `0000000082830609.pt` again): the cloning loss on its full batch (`--bc-batch 64`, was 32); everything else as before (48 games; the inference server now the C++ loop, which answers as the Python server did). If ties stay near 20% for two hours, the half batch was the cause; if not, test 2 is 40 games.

Test 1 after two hours (16:23 → 18:28, 110.8M → 121.1M): ties flat instead of rising (self-play 26, 27, 24, 27%; past snapshots 30, 27, 31, 26%; curriculum 20, 17, 17, 24% by half hour; after the first roll back they had risen 8-11 points in the same time), though above the 17-20% of the evening of the 10th (the league kept the past snapshots of the drift). The real game after the first half hour: 40 of 198 won against normal and insane where the AI would win ~92 (**44% of its record**, its level before the experiments). The AI's hero orders at its decisions 5.2%, basic units 34% (121.0M). The cloning loss stays on its full batch (64).

**Moved to native Linux** (2026-10-11): stopped at 19:10 at 124.56M steps on WSL and resumed at 21:19 on the same machine under native Ubuntu 26.04 (Python 3.14, torch 2.14 with CUDA 13), with the same recipe and checkpoint. The first launch lost its learner at its first update (Python 3.14's new default for starting processes broke the cloning loss's data workers; fixed). The real-game yardstick restarts from 21:19, and its first half hour is discounted as after any restart. `--lr 6e-5` stays queued until the native baseline has 300+ real games. Throughput work restarted the run about a dozen times between 21:19 and 00:49 (each restart front-loads short lost games; the recipe never changed); one restart (23:54) died after three updates of a device-side assert in the inference loop (stale padding, fixed). From 00:49: ~2,100 steps/s (the GPU shared better between learner and inference, `--update-pause-ms 4`), staleness ~2 updates instead of ~1.3: the real-game baseline for `--lr 6e-5` starts there. Paused 01:00-09:47 on the 12th for a machine shutdown (143.20M); from 09:55 the games run with the equivalent speedups of their own code (`--game-cuts 1529`, the same simulation): ~2,280 steps/s. The baseline starts at 09:55.

**Why the fights are lost: group orders scatter** (2026-10-12). When several of the policy's units got the same point order (attack-move, move, smart) in one step, their targets spread a median 304 world units from their mean (groups of 5+: 713), and it gave such group orders ~6.5 times a game; the built-in AI's spread 0 (groups of 5+: 10) and it gave ~69 a game. Every unit samples its target bins with a noise draw of its own, so a group's units go to different points. (The AI gives unit-target attacks to groups only 0.4 times a game: it attack-moves and lets units acquire targets.) The policy does not over-order: 1.0-1.3 orders a step against the AI's 1.96.

First try, in `fgself-12` (11:18-11:28): half the games drew one noise vector a step for all of a side's units, orders included. Those games built fewer buildings (15.6 against 18.9 a game), had less food at one minute (16.4 against 18.6) and held more (4,051 against 3,222): with one draw, similar units took a rare order (build a farm) all together or not at all.

**`fgself-13`** (the first experiment as a run of its own, from `fgself-12` at 155.16M, `scripts/selfplay_fork.py`; the anchor stays `fullgame-fast-win6`): the same A/B with the shared noise for the targets only (unit, x, y; the orders keep a draw each), half the games at random (`--shared-noise 0.5`; episodes record `shared_noise`). Judged per arm on real games against the AI's record (`real_games.py --split shared_noise`) and on the scatter of group targets. A new league and curriculum.

`fgself-13` after an hour (11:29-12:28, 7.0M steps): no gain. Real games against normal and insane 13 of 56 with independent noise (48% of the AI's record) and 12 of 66 with shared target noise (38%); unadvised curriculum games 33% and 30% won (~420 each), kills per unit lost 0.77 and 0.80. The shared noise did not tighten groups of 5+ (their targets' spread 353 against 320 world units): a group's units stand in different places and want different points, so one draw aligns only units that already agree. Coherent groups need a representation for them (a group command, or features of where each unit is already sent). Ties were higher than in `fgself-12` in both arms (self-play 38%, past snapshots 40-42%), with a new league whose snapshots are close to the policy.

**`fgself-14`** (from `fgself-13` at 7.0M, 12:31): two low-risk fixes together. The anchor's KL term as forward KL over the orders (`--ref-kl-mode forward`, AlphaStar's direction: reverse KL's pull on an order shrinks with the policy's own probability of it, so orders RL drove down stayed down). And two unit features with zero weights at first: the hit points' change since the last step and the position as sines and cosines at 5 frequencies (F 29 → 50). The shared-noise A/B goes on at 0.5.

`fgself-14` after two hours (12:31-14:27, 13.6M steps; its first half hour discounted): against normal and insane 46 of 184 won where the AI would win 97 (**47% of its record**), falling window by window (60% → 55% → 33% in the last half hour); all real games 28% won (fgself-12: ~24%). Ties: self-play 32%, past snapshots 34%, curriculum 31%. No clear gain within two hours; features with zero weights at first may need longer, so they stay.

**`fgself-15`** (from `fgself-14` at 13.6M, 14:30): where each unit is already sent. Each own unit's last order the player gave it: its target relative to the unit (a point, or the target unit where it is now), point or unit, and how long ago (F 50 → 55; zero weights at first). Everything else as `fgself-14`.

**A faster yardstick: evaluation games** (2026-10-12, `--eval-games-per-actor`). The real games had come from 10% of the AI games (~200 an hour), and a verdict of ±7 points on the AI's record took two hours. Each actor now also plays games that only evaluate: the real game against normal or insane (half each), races at random, never advised, nothing trained on (episodes `eval: true`; `real_games.py --split eval`). A real game is ~6.5 minutes of game time but only ~16 s of wall time, so 8 such slots (one per actor) play ~1,200 an hour: ±4-5 points in ~25 minutes, at ~10% of the training's throughput (16 slots cost a quarter). `fgself-15`'s policy (from `fgself-14`'s, 14:37-14:49): 125 of 457 won, **54% of the AI's record**, ±4.5.

**Forked runs lose ground in their first hours.** `fgself-15`'s evaluation games fell from 54% of the AI's record (14:37-14:50, 457 games) to 49% (14:51-15:10, 422) and 44% (15:10-15:30, 349); `fgself-14`'s real games fell 60% → 33% in two hours. The learner's order entropy is ~0.122 since `fgself-14` against ~0.107 in `fgself-12`: forward KL to the anchor makes the policy cover everything the anchor gives, so it spreads its orders.

**`fgself-16`** (from `fgself-15` at 4.8M, 15:30): the anchor term back to reverse KL; everything else as `fgself-15` (its features). Watched on the evaluation games' trend (4 in all, ~600 an hour: 8 cost ~20% of the throughput, `--eval-games`). A halt or a recovery points at forward KL.

`fgself-16`'s evaluation games kept falling: 35% (15:30-15:38, 100 games), 30% (15:38-15:46, 109), 27% (15:46-15:58, 137). The decline runs on through the fork (54 → 49 → 44 → 43 in `fgself-15`, then 35 → 30 → 27): it is the training, not the fork's first minutes, and reverse KL did not stop it within half an hour. In the real games, heroes and research fell with it (`fgself-14` 1.85 heroes and 8.5-8.9 research a game, `fgself-16` 1.46 and 4.6).

**`fgself-17`** (control, from `fgself-12` at 155.16M, 16:00): `fgself-12`'s recipe (reverse KL, the unit features as before with `--features 29`: the new channels zeros, no shared noise) plus 4 evaluation games. Holding near `fgself-12`'s level would put the decline on today's changes; falling too would put it on forking (a new league and curriculum, a fresh optimizer: checkpoints do not keep Adam's state).

`fgself-17` (16:00-16:30): 49 of 367 evaluation games won, **28% of the AI's record** (22%, 29%, 29% by ten minutes): `fgself-12`'s final policy is far below `fgself-14`'s (54%) on the evaluation games. The training real games read higher than the evaluation games for the same policy in the same windows (`fgself-15` 59% against 50%, `fgself-16` 52% against 36%, `fgself-17` 45% against 25%; 44-79 training games a window), so runs are compared on the evaluation games only. Read so, `fgself-13`/`fgself-14` (shared target noise, forward KL, hit-point-change and position features, a new league) raised the policy from ~25-29% to ~54%, and the decline began with `fgself-15`'s last-order features.

**`fgself-18`** (from `fgself-14` at 13.6M, 16:30): `fgself-14`'s recipe (forward KL, `--features 50`: the last-order channels zeros, shared target noise 0.5) plus 4 evaluation games. Holding near 54% puts the decline on the last-order features.

**Head to head** (`--eval-checkpoints`, 16:35-17:25 in `fgself-18`: its 8 evaluation games split at random between the current policy and three fixed checkpoints, the same conditions at the same time; ~275 games each against normal and insane):

| policy | of the AI's record |
|---|---|
| `fgself-14`'s last (13.6M) | **59%** |
| `fgself-18` now (`fgself-14`'s recipe without the last-order features) | **58%** |
| `fgself-16`'s last (with the last-order features) | 39% |
| `fgself-12`'s last (155.16M, the plateau) | 28% |

`fgself-13` and `fgself-14` doubled the policy's record against the AI (28% → 59%); the last-order features undid much of it (39%); without them `fgself-18` holds the gain. Next head to head: `fgself-13`'s last against `fgself-12`'s and `fgself-14`'s, to see which run made the jump.

Second head to head (17:28-17:58, ~150 games each, ±8): `fgself-18` now 57%, `fgself-14`'s last 43%, `fgself-12`'s last 38%, **`fgself-13`'s last 19%**. Pooled over both rounds: `fgself-18` 58%, `fgself-14` 53%, `fgself-12` 32%, `fgself-13` 19% (the rounds differ by up to 16 points for one checkpoint: ~150-280 games each are still noisy). `fgself-13` (shared target noise, a new league) was no better than `fgself-12`; the jump came with `fgself-14`'s changes: forward KL over the orders, and/or the hit-point-change and position features.

**`fgself-19`** (from `fgself-18` at 7.56M, 18:10): the target heads at temperature 0.8 (`--target-temperature`, a buffer of the model now, so a past snapshot in an inference slot keeps its own). Its evaluation games play the current policy against `fgself-18`'s last checkpoint (the same weights at 1.0) and `fgself-14`'s last.

`fgself-19` after 45 minutes (18:10-18:55, ~300 evaluation games each): the current policy at temperature 0.8 49% of the AI's record, `fgself-18`'s last (the same weights at 1.0) 46%, `fgself-14`'s last 51%. No measurable effect of the temperature (it stays at 0.8, as AlphaStar's). The line `fgself-14` → `fgself-19` holds ~50%.

The runs since `fgself-15` had updates of ~4.6 s instead of ~3.5: the last-order features' per-unit loop ran in the cloning loss's data workers even when `--features 50` zeroed them. It is skipped now (an update alone 3.2 → 2.7 s).

**`fgself-20`** (from `fgself-19` at 4.23M, 19:05): a value with the opponent's information. The global vector ends with a privileged block (23 numbers: the opponent's resources and supply; each side's workers, army, buildings, heroes, the army's hit points and centre, seen or not) that only the value head reads, through a zero-initialised projection. Its evaluation games play the current policy against `fgself-19`'s last (the same policy at the start) and `fgself-14`'s last.

`fgself-20` after 1 h 23 min (19:05-20:28, 568 evaluation games each): the current policy 49% of the AI's record (42%, 49%, 53% by window), `fgself-19`'s last 55%, `fgself-14`'s last 51%. The value fit better (its loss 0.0063 against 0.0075) but the policy did not gain measurably. The line holds ~50-55% (`fgself-12` 32%).

**`fgself-21`** (attribution, from `fgself-13`'s last at 7.0M, 20:28): what made `fgself-14`'s jump (19% → ~51-53%)? `fgself-14` changed forward KL and two unit features; this run repeats it with forward KL only (`--features 29`), shared target noise 0.5 as before. Evaluation games: the current policy against `fgself-13`'s last (its start) and `fgself-14`'s last. (It also has the value with the opponent's information, now always on; `fgself-20` found no measurable effect of it.)

**`fgself-21`: forward KL alone made the jump.** Head to head over its first hour (20:28-21:30, ~240-280 evaluation games each): the current policy (forward KL from `fgself-13`'s last, no new features) **54%** of the AI's record, `fgself-13`'s last (its start) 28%, `fgself-14`'s last (forward KL and the features) 49%. So forward KL to the anchor over the orders took the policy from ~28% to ~54% within an hour; the hit-point-change and position features added nothing measurable (the line goes on with `--features 29`). Reverse KL's pull on an order shrinks with the policy's own probability of it, so heroes and production that RL drove down stayed down; forward KL pulls them back towards the clone's.

**`fgself-22`** (from `fgself-21` at 4.87M, 21:29): the forward KL term stronger, ×0.5 (was 0.2). Evaluation games: the current policy against `fgself-21`'s last and `fgself-14`'s last.

`fgself-22` (forward KL ×0.5, 21:29-22:26, ~310-350 evaluation games each): the current policy 50% of the AI's record, `fgself-21`'s last (its start, ×0.2) 61%, `fgself-14`'s last 54%. A stronger anchor did not help; ×0.2 stays. As in `fgself-20`, the run's policy evaluated a little below its starting checkpoint: from a good start the training holds ~50-60% rather than climbing.

**`fgself-23`** (from `fgself-21`'s last at 4.87M, 22:29): forward KL over every head (`--ref-kl-mode forward-all`: the orders, the target unit, the point's x and y; ×0.2), as AlphaStar anchors all action arguments.

**`fgself-23`: forward KL over every head helps too.** Head to head over its first hour (22:29-23:27, ~340-380 evaluation games each): the current policy **69%** of the AI's record (70%, 67% by half hour), `fgself-21`'s last (its start) 56%, `fgself-14`'s last 53%. Ties fell from ~25-35% to ~10% (self-play 10%, past snapshots 11%, curriculum 7-9%); the curriculum's tax on the AI fell 0.29 → 0.24. Anchoring the targets (the unit, x and y) to the clone's makes the policy finish its games. From the morning's ~28-32% the policy is at ~69% of the built-in AI's own record on the evaluation games.

`fgself-23` held rather than climbed: 70%, 65%, 67% of the AI's record by half hour (22:29-23:57); the curriculum's tax on the AI fell to 0.20 while the policy won 58% of those games.

**`fgself-24`** (from `fgself-23`'s last at 10.2M, 23:57): a moving anchor. The KL term's reference (`--init`) becomes `fgself-23`'s last checkpoint instead of the clone `fullgame-fast-win6` (DeepNash and ProRL move theirs to a recent policy): the clone gave the policy the production habits it has now, and the pull back to its weaker parts may cap it. Forward KL over every head ×0.2. Evaluation games: the current policy against `fgself-23`'s last (its start and its anchor) and `fgself-21`'s last.

`fgself-24` after 27 minutes (23:57-00:24, stopped for a machine shutdown at 3.14M): the current policy 76% of the AI's record (135 evaluation games), `fgself-23`'s last (its start and anchor) 60% (175), `fgself-21`'s last 53% (173). Promising but early (±9): to be judged after it resumes.

**`fgself-24`: the moving anchor helps.** Head to head (23:57-00:25 on the 12th and 08:30-09:00 on the 13th after the shutdown, ~290-340 evaluation games each): the current policy **71%** of the AI's record (76%, 67%), `fgself-23`'s last (its start and anchor) 60%, `fgself-21`'s last 52%. Ties 14-15%. Where it still loses (`fgself-23`'s own 540 evaluation games): against normal it matches the AI's record (101%), against insane 41%; orc and undead are weakest against insane (34%, 25%), human and night elf reach 49-56%.

**`fgself-25`** (from `fgself-24`'s last at 6.22M, 08:59): the anchor moved again, to `fgself-24`'s last; nothing else changed. Evaluation games: the current policy against `fgself-24`'s last and `fgself-23`'s last. Next in line: training on normal and insane only (`--ai normal,insane`: a third of the AI games are still against easy).

`fgself-25` (the anchor moved again, 08:59-10:00, ~340-370 evaluation games each): the current policy **84%** of the AI's record (89%, 79% by half hour), `fgself-24`'s last (its start and anchor) 77%, `fgself-23`'s last 59%. The second move added less than the first (+7 against +11, and level in the second half hour).

**`fgself-26`** (from `fgself-25`'s last at 6.66M, 09:58): the AI games against normal and insane only (`--ai normal,insane`; a third had been against easy), the anchor where `fgself-25` had it (`fgself-24`'s last). Evaluation games: the current policy against `fgself-25`'s last and `fgself-24`'s last.

**`fgself-26`: training against normal and insane only helps.** Head to head (09:58-11:00, ~330-370 evaluation games each): the current policy **87%** of the AI's record (87%, 87% by half hour), `fgself-25`'s last (its start) 74%, `fgself-24`'s last 77%; against insane alone 48% against 38% and 38%.

**`fgself-27`** (from `fgself-26`'s last at 6.69M, 10:59): the AI games against insane only (`--ai insane`), the anchor unchanged. Evaluation games: the current policy against `fgself-26`'s last and `fgself-25`'s last.

`fgself-27` (insane only, 10:59-12:00, ~330-370 evaluation games each): the current policy **78%** of the AI's record (normal 118%, insane 45%), `fgself-26`'s last (its start) 72% (115%, 34%), `fgself-25`'s last 66% (100%, 35%). A modest gain, all of it against insane; normal kept. (The same checkpoint reads differently from run to run, `fgself-26`'s last 87% in its own and 72% here: compare within one run's evaluation games.)

**`fgself-28`** (from `fgself-27`'s last at 6.67M, 11:59): the anchor moved again, to `fgself-27`'s last (it had been `fgself-24`'s since `fgself-25`); insane only as before. Evaluation games: the current policy against `fgself-27`'s last and `fgself-26`'s last.

`fgself-28` (the anchor moved a third time, 11:59-13:00, ~340-380 evaluation games each): the current policy 82% of the AI's record (insane 55%), `fgself-27`'s last (its start) **88%** (insane 51%), `fgself-26`'s last 69%. The anchor's moves gave +11, +7 and now nothing: `fgself-27`'s last stays the best.

**`fgself-29`** (from `fgself-27`'s last, with `fgself-27`'s flags, 12:59): the distillation from the built-in AI twice as strong (`--opd-coef 0.3`, was 0.15): the insane AI advises half the games at the policy's own states, and insane is where it loses. Evaluation games: the current policy against `fgself-27`'s last and `fgself-26`'s last.

`fgself-29` (distillation ×0.3, 12:59-14:00, ~340-360 evaluation games each): the current policy 79% of the AI's record (insane 45%), `fgself-27`'s last (its start) **87%** (49%), `fgself-26`'s last 65%. No gain.

**How the best policy loses to insane** (`fgself-27`'s last in `fgself-29`'s evaluation games: 49 won, 139 lost): in the losses it is behind early (food at one minute 17.9 against the AI's 22.1; the wins 20.9 against 18.0), builds twice the AI's workers (18.6 against 8.8) while the AI builds an army, reaches tier 2 less often (78% against 99%) and loses the fights (13 kills for 48 units lost; the wins 40 for 16). Most losses are against orc (53 of 139). It does not answer early aggression, and its fights are inefficient: next, structural changes (credit per unit for the fights, a memory of enemy units seen).

`fgself-27` resumes (14:00) with its recipe to see whether longer training improves it; its evaluation games against its own last checkpoint before (6.67M) and `fgself-26`'s last.

**`fgself-30`** (from `fgself-27`'s 6.67M, 14:05; `fgself-27` had resumed at 14:00 for longer training and was stopped for this): credit per unit, OpenAI Five's team spirit (`--unit-credit 0.3 --unit-gamma 0.8`). Each own unit's advantage also gets a share of its own: the material it lost (hit points, death) and killed (DEATH events' killer) over the next few steps, relative to the step's other units (the team's credit stays the step's advantage) and normalised over the batch, ×0.3. Evaluation games: the current policy against `fgself-27`'s 6.67M and `fgself-26`'s last.

`fgself-30` (credit per unit, 14:05-14:57, ~290-330 evaluation games each): the current policy **82%** of the AI's record (normal 118%, insane 54%), `fgself-27`'s 6.67M (its start) **90%** (131%, 54%), `fgself-26`'s last 70%. No gain; against insane on par.

**Forks dip in their first 20 minutes.** The current policy's share by 20-minute windows: `fgself-28` 83 / 77 / 87%, `fgself-29` 71 / 86 / 77%, `fgself-30` 70 / 88 / 86% (~85-125 games a window, ±9). After its first 20 minutes `fgself-30` read 87% against `fgself-27`'s 90%. A fork started Adam afresh: its first steps move each weight by ~lr whatever the gradient, and the value warm-up's updates pull only toward the anchor and the clones. Checkpoints now keep Adam's state (`<steps>.adam`, the newest three), and forks and resumes from them load it. Verdicts also look at the games after the first 20 minutes.

**`fgself-31`** (from `fgself-27`'s 6.67M, 15:01; no Adam state for it yet): UPGO, AlphaStar's upgoing policy update (`--upgo 0.5`; its ablation: 73% → 82% average win rate). Each step's advantage also gets ×0.5 of its UPGO advantage, normalised over the batch: the return follows the trajectory while the actions after did at least as well as the value expected (Q ≥ V), else bootstraps from the value, so a good action's credit is not cancelled by worse ones later. Per-unit credit off. Evaluation games: the current policy against `fgself-27`'s 6.67M and `fgself-30`'s last.

`fgself-31` (UPGO ×0.5, 15:02-15:57, ~300-330 evaluation games each): the current policy **102%** of the AI's record (normal 145%, insane **67%**), `fgself-27`'s 6.67M (its start) **79%** (122%, 41%), `fgself-30`'s last 74%. Ties 4%. **A gain of ~23 points**, the largest since the forward KL; the first policy above the built-in AI's own record over normal and insane. It showed from the first 20 minutes (102%, then 103% and 99%): no fork dip, though this fork started Adam afresh too.

**`fgself-32`** (from `fgself-31`'s 5.92M, 15:58, with its Adam state: the first fork that carries it): UPGO ×1.0. Evaluation games: the current policy against `fgself-31`'s 5.92M (its start) and `fgself-27`'s 6.67M.

`fgself-32` (UPGO ×1.0, 15:58-16:57, ~320-340 evaluation games each): the current policy **111%** of the AI's record (normal 156%, insane **73%**), `fgself-31`'s 5.92M (its start) 94% (134%, 58%), `fgself-27`'s 6.67M 82%. Ties 5%. A gain of ~17 points, rising over the hour (98%, 110%, 122% by 20-minute windows). No fork dip with Adam's state carried.

**`fgself-33`** (from `fgself-32`'s 6.41M, 16:58, with its Adam state): the anchor moved to the UPGO policy (`--init` = `fgself-32`'s 6.41M; it was `fgself-24`'s 6.22M, ~60% of the AI's record). UPGO ×1.0. Evaluation games: the current policy against `fgself-32`'s 6.41M (its start and anchor) and `fgself-27`'s 6.67M.

`fgself-33` (anchor moved, 16:58-17:57, ~300-340 evaluation games each): the current policy **113%** of the AI's record (normal 153%, insane **82%**), `fgself-32`'s 6.41M (its start) 108% (156%, 66%), `fgself-27`'s 6.67M 87%. Ties 8% against 6%. A small gain, mostly against insane (±7 each there); by 20-minute windows 111%, 128%, 97%. By race against normal and insane: human 44/86, night elf 66/80, orc 33/57, undead 29/78.

**`fgself-34`** (from `fgself-33`'s 6.25M, 17:59, with its Adam state): UPGO ×2.0; the anchor stays `fgself-32`'s 6.41M. Evaluation games now two thirds against insane (`--eval-ai normal,insane,insane`): normal is at ~150% of the AI's record, insane is the gap. Against the current policy: `fgself-33`'s 6.25M (its start) and `fgself-27`'s 6.67M.

`fgself-34` (UPGO ×2.0, 17:59-18:57; evaluation 2/3 against insane, so its shares read lower than the runs before): the current policy **89%** of the AI's record (normal 145%, insane 67%; 256 games), `fgself-33`'s 6.25M (its start) **85%** (131%, 62%; 349), `fgself-27`'s 6.67M 66% (117%, 40%; 305). Ties 9% against 8%. No clear gain over ×1.0; by 20-minute windows 93%, 97%, 76%. Back to ×1.0.

**`fgself-35`** (from `fgself-33`'s 6.25M, 18:58, with its Adam state): twice the learning rate (`--lr 6e-5`, was 3e-5, AlphaStar's; queued since 2026-10-11 and never tested). UPGO ×1.0, the anchor `fgself-32`'s 6.41M. Evaluation games (2/3 against insane): the current policy against `fgself-33`'s 6.25M (its start) and `fgself-27`'s 6.67M.

`fgself-35` (`--lr 6e-5`, 18:58-19:58): the current policy 103% of the AI's record (normal 170%, insane 76%; 263 games), `fgself-33`'s 6.25M (its start) 93% (142%, 70%; 353), `fgself-27`'s 6.67M 73%. But it fell over the hour: 136%, 97%, 74% by 20-minute windows; in its second half 82% against the start's 101% (insane 55% against 76%). Its training losses stayed calm (KL to the last policy 0.0039 against 0.0029, clip fraction 1.7% against 1.5%). Twice the learning rate: a fast rise, then worse than the start. Back to 3e-5.

**`fgself-36`** (from `fgself-33`'s 6.25M, 19:59, with its Adam state): the best line on, the anchor moved again (`--init` = `fgself-33`'s 6.25M). UPGO ×1.0, lr 3e-5. Evaluation games (2/3 against insane): the current policy against `fgself-33`'s 6.25M (its start and anchor) and `fgself-35`'s last (6.25M).

`fgself-36` (the anchor moved again, 19:59-21:02): the current policy 94% of the AI's record (normal 161%, insane 66%; 214 games), `fgself-33`'s 6.25M (its start) 84% (301), `fgself-35`'s last 85% (ties 11%; 279). On one matchup mix (below): 84%, 79%, 83%. No clear gain; by 20-minute windows 98%, 96%, 83%.

**The evaluation games repeated in every fork.** The games' random draws were seeded by the thread and the update the run started from, and every fork starts from update 0: all forks drew the same evaluation games in the same order (`fgself-34`, `-35` and `-36`: the current policy 256-266 games a hour, the AI's mean record in its matchups 0.450-0.456, against 0.49-0.51 for the checkpoints; the current policy also played fewer launches, 107 against 141-154). The AI-record share depends on the matchup mix (the policy is far above the AI's record as human, near it as night elf). Re-scored on one matchup mix (each matchup that every policy played, weighted by its games over all of them; `real_games.py --split` now prints it), the verdicts since `fgself-30` hold within a few points:

| run | current | its start | the other |
|---|---|---|---|
| `fgself-30` (credit per unit) | 84% | 91% | 72% |
| `fgself-31` (UPGO ×0.5) | **102%** | 79% | 72% |
| `fgself-32` (UPGO ×1.0) | **108%** | 92% | 80% |
| `fgself-33` (anchor moved) | 112% | 106% | 81% |
| `fgself-34` (UPGO ×2.0) | 85% | 81% | 64% |
| `fgself-35` (lr 6e-5) | 101% | 93% | 71% |
| `fgself-36` (anchor moved) | 84% | 78% | 83% |

(Bootstrapped 90% intervals are about ±9 points each.) Now each evaluation slot plays a matchup (races, sides, difficulty) with every policy in turn, and the run's name is part of the seed.

**`fgself-37`** (from `fgself-36`'s 6.25M, 21:02, with its Adam state): half the KL to the anchor (`--ref-kl 0.1`, forward over every head; was 0.2): the anchor is now a strong policy, not a clone. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-36`'s last (its start) and `fgself-33`'s 6.25M (the anchor).

`fgself-37` (`--ref-kl 0.1`, 21:02-21:58, ~265-270 paired evaluation games each): on one matchup mix the current policy **86%** of the AI's record (normal 156%, insane 58%), `fgself-36`'s last (its start) 82% (insane 51%), `fgself-33`'s 6.25M (the anchor) **87%** (insane 64%). No gain. The pairing works: the three policies played the same 23+ matchups, 263-270 games each.

**How the best policy loses to insane now** (`fgself-33`'s 6.25M in the evaluation games of `fgself-34` to `-37`, 867 games: 276 won, 565 lost, 26 tied): as before, a greedy economy. In its losses it builds 22 workers against the AI's 10, has 1.7 heroes against 2.5, is behind at one minute (food 19.4 against 22.0) and loses the fights (19 kills for 63 units lost); its wins: 14 workers against 17, 2.1 heroes against 1.4, 43 kills for 16. Losses are longest as undead and human (167 and 186 of 565). 90% of its games against the AI are curriculum games, in which the AI loses up to ~80% of its income to the tax: there greed pays.

**`fgself-38`** (from `fgself-33`'s 6.25M, 21:59, with its Adam state): half the games against the AI untaxed (`--real-share 0.5`, was 0.1). It wins ~30% of the real games against insane now, so they carry a signal (at 0.25 on 2026-10-11 it won almost none). `--ref-kl` back to 0.2. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-33`'s 6.25M (its start and anchor) and `fgself-37`'s last.

`fgself-38` (`--real-share 0.5`, 21:59-23:00, ~260 paired games each): on one matchup mix the current policy **75%** of the AI's record (insane 52%), `fgself-33`'s 6.25M (its start) **81%** (59%), `fgself-37`'s last 75% (62%). No gain; back to 0.1. Its training games against the real insane stayed at ~28% won. The evaluation games against insane also correct the diagnosis: the policy trains more army than the AI (~60 units to ~40) and still loses the fights (~25 kills for ~50 units lost), with 1.8 heroes to the AI's 2.3. The fights decide, not the economy.

**`fgself-39`** (from `fgself-33`'s 6.25M, 23:01, with its Adam state): no dropout in the updates (`--rl-dropout 0`; the network kept the cloning's 0.1). The actors act in eval mode, without dropout, while the updates trained with it: each update's ratios, KL and clipping counted the dropout's noise (3 layers, dropout in attention, residuals and feed-forward), and so did the gradients. `--real-share` back to 0.1. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-33`'s 6.25M (its start and anchor) and `fgself-38`'s last.

Its first launch died at once: with the dropout changed in its config, its evaluation checkpoints' network (dropout 0.1) found no inference slot. `model.arch_key` now ignores dropout (nothing at inference). Relaunched at 23:01. Its first updates: the KL to the last policy **0.0009-0.0010** (with dropout 0.003-0.004) and to the anchor 0.0009 (0.004-0.009): most of what the updates measured as KL was the dropout's noise.

`fgself-39` (no dropout in the updates, 23:01-23:56, ~290-300 paired games each): on one matchup mix the current policy **109%** of the AI's record (normal 172%, insane **81%**), `fgself-33`'s 6.25M (its start) **93%** (180%, 55%), `fgself-38`'s last 100% (173%, 67%). Ties 4% against 7%. **A gain of ~16 points, ~25 against insane**; by 20-minute windows 112%, 94%, 125%. The updates' KL to the last policy stayed ~0.001 and their clip fraction 0.8% (with dropout 1.5-1.8%); the KL to the anchor rose from 0.003 to 0.005 over the hour. New best: `fgself-39`'s last, `runs/fgself-39/checkpoints/0000000006018863.pt` (with its Adam state).

**Paused 2026-10-13 23:56 for a machine shutdown.** `fgself-39` stopped cleanly (status "stopped"). Next: go on from `fgself-39`'s last with its recipe (`--rl-dropout 0`, UPGO ×1.0, forward KL ×0.2 over every head, `--ai insane`, `--real-share 0.1`); first the anchor moved to it (`--init` = that checkpoint; it is still `fgself-33`'s 6.25M), then the fights (the policy trains more army than insane and loses the fights ~2:1, with fewer heroes): hero levels in the shaping's material (a hero counts its cost only), focus fire, a memory of enemy units seen.

**`fgself-40`** (from `fgself-39`'s last, 2026-10-14 10:15, after the machine's restart, with its Adam state): the anchor moved to the new best (`--init` = `fgself-39`'s last; it was `fgself-33`'s 6.25M). The recipe otherwise `fgself-39`'s. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-39`'s last (its start and anchor) and `fgself-33`'s 6.25M.

`fgself-40` (the anchor moved, 10:15-11:18, ~355 paired games each): on one matchup mix the current policy **129%** of the AI's record (insane 139 of 246 won), `fgself-39`'s last (its start) **129%** (142 of 248), `fgself-33`'s 6.25M 83% (72 of 244). Ties 2%, 2%, 9%. No gain from moving the anchor (the third time it did nothing after the first two moves helped). The dropout-free policy leads `fgself-33`'s 6.25M by ~46 points here (16 in `fgself-39`'s own games, which started from it): against insane it now wins more than half its games.

**`fgself-41`** (from `fgself-40`'s 6.58M, 11:19, with its Adam state): hero levels in the shaping (`--hero-level-value 150`): a hero's material also counts 150 for each level above the first (its cost, 525, did not grow as it levels; a dead hero counts nothing until revived). The anchor stays `fgself-39`'s last. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-40`'s last (its start) and `fgself-39`'s last (the anchor).

`fgself-41` (hero levels in the shaping, 11:19-12:19, ~390 paired games each): on one matchup mix the current policy **133%** of the AI's record (insane 144 of 254 won), `fgself-40`'s last (its start) **130%** (141 of 254), `fgself-39`'s last 124% (127 of 253). Ties 1-2%. No clear gain; off again.

**`fgself-42`** (from `fgself-41`'s 6.41M, 12:19, with its Adam state): twice the learning rate again (`--lr 6e-5`), now without dropout in the updates. `fgself-35` tried it with dropout on, when most of what an update moved was the dropout's noise; without it the KL per update is ~0.001 (it was 0.003-0.004), so the policy moves slowly. The anchor stays `fgself-39`'s last. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-41`'s last (its start) and `fgself-39`'s last (the anchor).

`fgself-42` (`--lr 6e-5` without dropout, 12:19-13:19, ~370 paired games each): on one matchup mix the current policy **120%** of the AI's record (insane 120 of 247 won), `fgself-41`'s last (its start) **132%** (140 of 255), `fgself-39`'s last 127%. Worse, falling over the hour (125%, 120%, 115%): twice the learning rate hurt with dropout and without. 3e-5 stays.

**How the best policy loses to insane now** (`fgself-41`'s last, 511 evaluation games against insane: 285 won, 222 lost). The weakest matchups are undead against night elf and undead (52-53% of the AI's record) and the human mirror (85%). In its losses it trains masses of tier-1 units (as undead ~47 ghouls a game, 31 in its wins; as human ~40 footmen, 31 in its wins) against the AI's ~25-35 units in all, researches less, has fewer heroes (1.8 against 2.4), reaches tier 2 less often (91% against 100%) and loses those fights 17 kills to 65 units lost. It holds less than the AI (~1,150-1,400 gold and lumber on average against ~1,900): the shaping's penalty on unspent resources (`--float-penalty 0.5`, from when it hoarded ~3,700) now pushes it to spend at once.

**`fgself-43`** (from `fgself-41`'s 6.41M, 13:19, with its Adam state): a smaller penalty on unspent resources (`--float-penalty 0.2`, was 0.5), so saving for tech and heroes pays more. lr 3e-5; the anchor stays `fgself-39`'s last. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-41`'s last (its start) and `fgself-39`'s last (the anchor).

`fgself-43` (`--float-penalty 0.2`, 13:19-14:19, ~384 paired games each): on one matchup mix the current policy **135%** of the AI's record (insane 137 of 256 won), `fgself-41`'s last (its start) **142%** (150 of 253), `fgself-39`'s last 136%. No gain. Against insane after its first half hour it held a little more (~1,420 against ~1,200) but researched no more and had no more heroes. Back to 0.5. Small knobs of the reward and the learning rate have not moved the line since `fgself-39` (~130-140% on one mix).

**`fgself-44`** (from `fgself-41`'s 6.41M, 14:19, with its Adam state): twice the steps per update (`--batch-steps 16384`, was 8192; snapshots and checkpoints every 10 updates to keep their spacing in steps): each gradient from twice the data, half as many updates. Twice the learning rate hurt twice, so the updates look limited by their gradients' noise, not their size. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-41`'s last (its start) and `fgself-39`'s last (the anchor).

`fgself-44` (`--batch-steps 16384`, 14:19-15:19, ~365 paired games each): on one matchup mix the current policy **135%** of the AI's record (insane 132 of 246 won), `fgself-41`'s last (its start) **143%** (138 of 249), `fgself-39`'s last 131%. No gain (by 20-minute windows 118%, 141%, 146% against the start's 136%, 145%, 146%). Back to 8192. `fgself-41`'s last has read 132-148% in four runs and every run from it lower: the recipe's knobs (shaping, learning rate, batch) do not move the line.

**`fgself-45`** (from a grown `fgself-41`'s 6.41M, 15:19): a deeper network, 5 transformer layers (was 3; 3.02M weights, was 2.13M). `scripts/grow_model.py` adds the layers after the last, each the identity at first (its attention's output projection and its feed-forward's second layer zero: a pre-norm layer then adds nothing), so the network plays exactly as `fgself-41`'s last (the encoder's outputs equal; its first update's KL to the anchor 0.0050, as `fgself-41`'s); Adam's state carried by parameter name for the old weights. A run can now start from another architecture than its anchor's (the anchor stays `fgself-39`'s last, 3 layers), and the inference warms the evaluation checkpoints' architectures. Its updates take 4.3 s (3.7 s), ~1,680 steps/s (~1,880). Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-41`'s last (its start, 3 layers) and `fgself-39`'s last (the anchor).

`fgself-45` after its first hour (~340 paired games each): on one matchup mix the current policy 131% of the AI's record (insane 119 of 224 won), `fgself-41`'s last (its start) 135% (123 of 224), `fgself-39`'s last 131%. Level, as a network that started as its start and grows its new layers from zero may be for a while: it goes on for another hour or two before a verdict. ~1,575 steps/s.

`fgself-45` stopped at 1.5 hours (15:19-16:49, ~500 paired games each): on one matchup mix the current policy **129%** of the AI's record, `fgself-41`'s last (its start) **138%**, `fgself-39`'s last 129%; its second hour alone 126% against 146%. No gain from the deeper network on this time scale (it may need far longer: the grown checkpoint stays at `runs/fullgame/grown/fgself-41-6.41M-5layers.pt`).

**A pattern:** `fgself-41`'s last has read ~140% in five runs (~2,000 paired games: 142%, 143%, 142%, 143%, 138%), and every run trained from it settled near ~130% (fgself-42 to -45), the level of their anchor (`fgself-39`'s last, 127-136% in the same games). The forward KL to the anchor (×0.2 over every head) may pull the policy back to it.

**`fgself-46`** (from `fgself-41`'s 6.41M, 16:49, with its Adam state): the anchor moved to `fgself-41`'s last (`--init` = it), 3 layers. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-41`'s last (its start and anchor) and `fgself-39`'s last (the old anchor).

`fgself-46` (the anchor at its start, 16:49-17:51, ~377 paired games each): on one matchup mix the current policy **139%** of the AI's record (insane 146 of 243 won), `fgself-41`'s last (its start and anchor) **142%** (148 of 242), `fgself-39`'s last 133%. Level: with the anchor at its start the line held (the runs from `fgself-41`'s last with the anchor at `fgself-39`'s last fell 7-12 points below it). No gain either. (`real_games.py` failed on a game over in its first second, against an AI that never started: 2 of ~56,000 games since `fgself-39`; such a game has no food at one minute now.)

**`fgself-47`** (from `fgself-46`'s 6.58M, 17:52, with its Adam state; `fgself-46`'s recipe goes on): an evaluation-only experiment, sampling at temperature 0.8 for the orders and the targets (AlphaStar's) against 1. The network has an order temperature now (a buffer, as the targets'; `--order-temperature`), and `scripts/checkpoint_variant.py` writes a checkpoint's copy at other temperatures. The evaluation games play `fgself-41`'s last as it is and its copy at 0.8, head to head on the same matchups, beside the current policy.

`fgself-47` (17:52-18:20, ~139 paired games each): `fgself-41`'s last at temperature 0.8 (orders and targets) **71%** of the AI's record (ties 10%), the same weights at 1 **136%** (ties 3%), the current policy 134%. Decisive at once: a lower order temperature concentrates each unit's choice on "no new order" (its likeliest class most steps), and the policy goes passive (cloning saw the same: a temperature below 1 issued fewer orders). Stopped early for the next experiment.

**`fgself-48`** (from `fgself-47`'s 2.80M, 18:20, with its Adam state): no advice from the built-in AI (`--opd-share 0`, was 0.5; `--opd-coef 0`, was 0.15). In half the games against itself, past snapshots and the curriculum the insane AI shadowed the policy's side (its orders labels, undone; some stood until the step ended) and a cloning loss pulled the policy toward them. The policy now wins ~60% of its games against insane: the teacher is weaker than the student, and its orders briefly play in those games. The cloning loss on the AI's winning demonstrations (`--bc-coef 0.02`) and the forward KL to the anchor stay. Its first updates' KL to the last policy: 0.0002-0.0003 (0.001 with the advice): the distillation was much of what an update moved. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-47`'s last (its start) and `fgself-41`'s last (the anchor).

`fgself-48` (no advice, 18:20-19:18, ~400 paired games each): on one matchup mix the current policy **152%** of the AI's record (insane **172 of 273 won**, 63%), `fgself-47`'s last (its start) **137%** (140 of 272), `fgself-41`'s last (the anchor) 145% (161 of 270). Ties 0-1%. **A gain**, by 20-minute windows 135%, 166%, 158%, and ~2,030 steps/s against ~1,870 (no shadow AI in the games, no labels or distillation loss in the updates). The student beats its teacher now. New best: `fgself-48`'s last, `runs/fgself-48/checkpoints/0000000006907732.pt`.

**`fgself-49`** (from `fgself-48`'s 6.91M, 19:18, with its Adam state): no cloning loss on the AI's winning demonstrations either (`--bc-coef 0`, was 0.02). On 2026-10-10 (`fgself-12`) that let the hero orders decay; now the forward KL to the anchor over every head keeps them. Its updates are faster without the cloning batches (its second update at ~2,690 steps/s). The anchor stays `fgself-41`'s last. Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-48`'s last (its start) and `fgself-41`'s last (the anchor).

`fgself-49` (`--bc-coef 0`, 19:18-20:18, ~410 paired games each): on one matchup mix the current policy **147%** of the AI's record (insane 192 of 280 won), `fgself-48`'s last (its start) **152%** (200 of 277), `fgself-41`'s last (the anchor) 129%. Level (by 20-minute windows 131%, 150%, 162% against the start's 136%, 164%, 160%), the hero orders held (1.56 heroes a game against the start's 1.45), and ~2,220 steps/s against ~2,030: the updates take 2.4 s (3.4 s), and the learner now waits for the games (1.3 s an update). Kept.

**`fgself-50`** (from `fgself-49`'s 7.73M, 20:18, with its Adam state): the anchor moved to the best (`--init` = `fgself-48`'s last, ~150%; it was `fgself-41`'s last, 129% here): anchored below the line, runs fell back toward their anchor (`fgself-42` to `-45`); anchored at their start, the line held (`fgself-46`). Evaluation games paired on matchups (2/3 against insane): the current policy against `fgself-49`'s last (its start) and `fgself-48`'s last (the anchor).

`fgself-50` (the anchor moved to the best, 20:18-21:18, ~418 paired games each): on one matchup mix the current policy **179%** of the AI's record, `fgself-49`'s last (its start) **179%**, `fgself-48`'s last (the anchor) 176%. Level. (Higher than the runs before for the same policies: this run's matchups differ.) Against insane the three won ~77% of their games, against normal ~99% (157 of 158).

**`fgself-51`** (from `fgself-50`'s 7.73M grown to 5 layers, 21:20, with its Adam state for the old weights): the deeper network again, now that the updates take 2.3 s and the learner waits 1.5 s an update for the games: the two layers cost little throughput. `fgself-45` tried it under the old recipe (no gain in 1.5 hours); this one runs ~2 hours. Evaluation games now against insane only (`--eval-ai insane`). The anchor stays `fgself-48`'s last (3 layers). Evaluation games paired on matchups: the current policy against `fgself-50`'s last (its start, 3 layers) and `fgself-48`'s last (the anchor).

`fgself-51` (the deeper network, 21:20-23:12, ~793 paired games each, all against insane): on one matchup mix the current policy **156%** of the AI's record (**80% of its games won**), `fgself-50`'s last (its start) **148%** (76%), `fgself-48`'s last (the anchor) 140% (72%). A modest gain; ahead in every 20-minute window but the last. By matchup (the current policy): orc 97%, night elf 98%, human 65%, undead 60% of its games won; the mirrors human 93%, orc 98%, night elf 100%, undead 62%; the hardest, undead against orc, 47% (the AI itself 23%).

**The duelfast milestone.** The policy beats the built-in insane AI in every matchup, untaxed, above the AI's own record. The user's next step (2026-10-15): the real game on Echo Isles (see [the road](../road-to-the-real-game.md)).

**`fgself-52`** (from `fgself-51`'s last, 2026-10-15 10:19, after the machine's restart, with its Adam state): the 5-layer line on, the anchor moved to its last (`--init` = `fgself-51`'s last). Evaluation games paired on matchups (insane): the current policy against `fgself-51`'s last (its start and anchor) and `fgself-50`'s last.

**Toward Echo Isles** (2026-10-15): positions and point targets in each map's own frame (`features.set_geometry`; self-play on another map sets it); two test games of the built-in AI against itself on Echo Isles (both at full strength, insane, 8 game minutes in 16-19 s); then `runs/fullgame/demos-echo-1`: 1,500 games of the built-in AI against itself (normal and insane, all races, `--handicap 100`, up to 40 minutes), ~600 an hour, for a clone of the real rules as the anchor.

`fgself-52` (the 5-layer line, the anchor at its start, 10:19-11:13, ~270 paired games each against insane): on one matchup mix the current policy **172%** of the AI's record (81% of its games won), `fgself-51`'s last (its start) **172%** (81%), `fgself-50`'s last (3 layers) 154% (73%). Level. (~1,700 steps/s: the demonstrations' collection shares the CPU.)

**`fgself-53`: the duelfast policy on Echo Isles, zero-shot** (from `fgself-52`'s last, 11:24; `--map "(2)EchoIsles" --pairs 1 --handicap 100 --max-minutes 35`, against insane with its curriculum). Its first launch failed at once: self-play's games defaulted to five pairs of players a map load (`--pairs 5`), which only the duel maps have; `--pairs 1` for Echo Isles. Then **0 of 58 games won** in its first minutes, all lost in 7-10 game minutes, the curriculum's taxed ones too. Against the AI: heroes 0.1 a game against 1.9, army units 5 against 14, most food 22 against 60, tier 2 in none of 58, 1 kill for 35 units lost; it trained workers (~32 a game, many replacing the killed) and held ~1,900 gold and lumber. Its duelfast habits (costs halved, production three times faster there) fail under the game's rules. Stopped after a few minutes: the GPU went to a clone of the real rules.

**`fullgame-echo-1`** (behaviour cloning, 11:33): the duelfast 5-layer policy (`fgself-52`'s last) fine-tuned on the built-in AI's winning sides on Echo Isles (`demos-echo-1`, 1,019 games at its start; 6 epochs, lr 1e-4). Its loader failed on a tie's game at first (no winning side with `--winners-only`: fixed). The clone becomes the next runs' start and anchor.

**A curriculum of rules instead** (the user, 2026-10-15: "start with halved costs and faster production, and anneal to the real rules later ... as long as we stay on echo isles"). The clone was stopped (its first launch had also read a game the collection was still writing: fixed). Echo Isles under other rules: `--map echofast` (duelfast's rules in the real map: its tables in the map, `data/duelmap.py`) and `echoP` (P% of the way to the game's rules; `echo100` is the real game).

**`fgself-54`** (`echofast`, `--handicap 50`, from `fgself-52`'s last, 12:29): still 0 of 66 against insane, behind from the first minute (food 10.6 at one minute against 24). Its build orders were refused: in a test game 5 of 67 barracks orders taken (on duelfast 2 of 3). The map frame scaled to Echo Isles' size put every learned offset 2.3 times as far from the base, in the trees.

**The frame, anchored.** Each player now sees positions relative to its own start location at the duel maps' scale, its base where duelfast's left base was (on the duel maps exactly the frame of before). Point bins reach over the whole map (the network's `bins`; `scripts/grow_bins.py` widens a checkpoint's: the old bins in place, the new outer ones the nearest edge's row 10 lower; a constant low bias first let the outer bins win where every old logit was low, builds 10,000 units north and south).

**`fgself-55`** (12:42, anchored frame, 416 bins): it played a real game (tier 2 in 86 of 89, 95 army units to the AI's 49, 2.0 heroes to 2.7) but won none of 89 against insane and tied 41: its attacks went to the enemy base's old place, 4,600 units out, mid-map on Echo Isles (9,856).

**The frame, squeezed.** Between the bases x is squeezed to the duel maps' 4,600 (as is within 1,500 of either base; ~200 units a bin mid-map): the enemy base is where duelfast's was. 352 bins.

**`fgself-56`** (12:52): in its first minutes **33 of 73 games against insane won** (15 tied), 4 of 19 against the untaxed AI; 100 army units to the AI's 38, 32 kills for 56 lost. Evaluation games against insane on `echofast`: the current policy against its start (the widened `fgself-52`'s last).

`fgself-56` (`echofast`, 12:52-14:14, 341 paired evaluation games each against the untaxed insane AI): the current policy **won 44%** (ties 21%), its start 31% (ties 28%); on one matchup mix 91% of the AI's record against 64%, rising (by 20-minute windows 58%, 95%, 84%, 118%, 152%). **A gain.** Its wins are quick (~6 minutes, 41 kills for 24 lost); its losses long, with huge armies of cheap units that lose the fights (132 trained against the AI's 46, 17 kills for 87 lost); its opening slow (food 14 at one minute against 23). By race: orc 34 of 52 won, night elf 24 of 42, human 14 of 60, undead 10 of 60 (at 13:43).

**`fgself-57`** (from `fgself-56`'s last, 14:15, with its Adam state): `echofast` on, the anchor moved to `fgself-56`'s last, games up to 30 minutes (~25% tied at 20). Evaluation games against insane on `echofast`: the current policy against `fgself-56`'s last (its start and anchor) and the widened `fgself-52`'s last.

`fgself-57` (14:15-15:14, ~165 paired games each against the untaxed insane AI on `echofast`): the current policy **won 52%** (ties 11%), `fgself-56`'s last (its start) **57%** (8%), the widened `fgself-52`'s last 42%. No gain. With 30-minute games ties fell to ~8% (from ~25% at 20). On `echofast` the line holds ~55-60% of its games against insane.

**`fgself-58`** (from `fgself-56`'s last, 15:15): the first step toward the real rules, `--map echo25` (hit points 0.625, build, train and research times 0.5, costs 0.625 of the game's) and `--handicap 62` (was 50); games up to 30 minutes; the anchor its start. Evaluation games against insane on `echo25`: the current policy against `fgself-56`'s last (what the step costs) and `fgself-57`'s last. The schedule: `echo25` / 62, `echo50` / 75, `echo75` / 88, `echo100` / 100 (the real game), each once the policy is past its start there.

`fgself-58` (`echo25`, 15:15-16:16, ~147 paired games each against the untaxed insane AI): the current policy **won 37%** (ties 9%), `fgself-56`'s last (its start) **43%** (10%), `fgself-57`'s last 46%. The step cost ~15 points (`fgself-56`'s last won ~57% on `echofast`), and an hour did not win them back.

**`fgself-59`** (from `fgself-56`'s last, 16:16): `echo25` again with a quarter of the KL to the anchor (`--ref-kl 0.05`, was 0.2): the anchor learned duelfast's rules, and its pull keeps their habits (on duelfast the anchor's rules always matched). Evaluation games against insane on `echo25`: the current policy against `fgself-56`'s last (its start and anchor) and `fgself-58`'s last.

`fgself-59` (`echo25`, `--ref-kl 0.05`, 16:16-17:15, ~150 paired games each): the current policy **won 52%**, `fgself-56`'s last (its start) **52%**, `fgself-58`'s last 53%. Level: a weaker anchor did not let it adapt in an hour either.

**`fullgame-echo-1` again** (17:20): after `echofast` an hour of RL did not win back a step of the rules, so a clone of the real rules first (the project's recipe: clone, then RL anchored to it). The best `echofast` policy (`fgself-56`'s last, with the anchored and squeezed frame and 352 bins) fine-tuned on the built-in AI's winning sides of the 1,500 Echo Isles games under the game's own rules, 2 epochs at lr 1e-4 (~15 batches a second, ~15 minutes an epoch; `bc.py` now prints its progress). Then RL under the real rules (`echo100`, `--handicap 100`) from the clone, anchored to it.

`fullgame-echo-1` (17:20-17:52, 2 epochs of ~13,300 batches, ~16 minutes each): validation loss 3.51 after the first, **3.29** after the second; the AI's orders (given one) 82%, its target units 68%, its points within 7.2 bins, the value's explained variance 0.99.

**`fgself-60`: the real game on Echo Isles** (18:13): the game's own rules (`--map echo100`), both sides at full strength (`--handicap 100`), games up to 40 minutes, against the insane AI with its curriculum; from the clone, anchored to it (forward KL ×0.2 over every head). Its first 54 games against insane: **3 won** (2 of 12 against the untaxed AI), 10 tied, ~25 game minutes each (zero-shot the duelfast policy had won 0 of 58). Its economy now matches the AI's (food 8.5 at one minute against 7.9, 16 workers against 16, tier 2 in 48 of 54); its fights do not (1.5 heroes against 2.3, 17 kills for 50 units lost). Evaluation games against insane under the real rules: the current policy against the clone (its start and anchor) and `fgself-56`'s last (the `echofast` policy as it is).

`fgself-60`'s evaluation games after half an hour (~43 each against the untaxed insane AI under the real rules) turned it around: the current policy **0 of 42** won, the clone **0 of 40**, **`fgself-56`'s last (the `echofast` policy, as it is) 22 of 45 (49%)**. The frame (anchored at the own start, squeezed between the bases) was what Echo Isles needed; the rules matter little (at `echo25` it had won 43-52%, here 49%); and cloning the AI's games undid the policy's fighting (its training games' 3 of 54 were mostly against the taxed AI). Stopped.

**`fgself-61`** (from `fgself-56`'s last, 18:46): the real game on Echo Isles (`echo100`, `--handicap 100`, games up to 40 minutes) from the `echofast` policy, anchored to itself. Evaluation games against insane under the real rules: the current policy against `fgself-56`'s last.

`fgself-61` (the real game, 18:46-20:46, ~306 paired games each against the untaxed insane AI): the current policy **won 47%** (ties 10%), `fgself-56`'s last (its start) **46%** (12%). Level: in the real game on Echo Isles the policy is about even with insane. In both policies' evaluation games: the wins quick (11.6 game minutes, 35 kills for 17 lost, it attacks early); the losses long (25.8 minutes: 35 workers against the AI's 15 and 90 army units against 42, 21 kills for 85 lost, 1.2 heroes against 2.7, tier 2 in 78% against 100%; it holds ~4,000 gold and lumber, the AI ~7,600); ties at 40 minutes 11%. By race: orc 123 of 168, night elf 97 of 172, human 42 of 128, undead 25 of 145 (against orc 0 of 25). As undead it trains ~32 ghouls a lost game, as human ~17 footmen.

**`fgself-62`** (from `fgself-61`'s last, 20:46): a longer horizon for the real game (`--gamma 0.9997`, was 0.999: ~28 minutes of game time ahead instead of ~8; the duelfast games lasted ~6, the real game's ~25); it misses what pays off late. Evaluation games against insane under the real rules: the current policy against `fgself-61`'s last.

`fgself-62` (`--gamma 0.9997`, 20:46-22:46, ~325 paired games each against the untaxed insane AI): the current policy **won 59%** (ties 7%), `fgself-61`'s last (its start) **55%** (4%); on one matchup mix 112% of the AI's record against 104%, ahead in 5 of 6 twenty-minute windows. A small gain; kept. Its losses: 0.87 heroes a game against the AI's 2.69, ~47 workers. Heroes by race (the current policy against the AI): human 0.57 against 2.22 (its altar in 48 of 72 games, the AI's in all), undead 1.20 against 2.15, orc 0.96 against 1.59, night elf 1.35 against 1.72; games won: human 18 of 72, undead 17 of 60, orc 78 of 94, night elf 78 of 98.

**`fgself-63`** (from `fgself-62`'s last, 22:47): the cloning loss back, on the built-in AI's games in the real game (`--bc-coef 0.02 --bc-data runs/fullgame/demos-echo-1 --bc-winners 1`), for the heroes: in 2026-10-10's runs it was the one term trained at the AI's own states and kept its hero orders; here the anchor is the policy itself. The full clone undid the fighting; a ×0.02 term should not. Evaluation games against insane under the real rules: the current policy against `fgself-62`'s last.

`fgself-63` (the cloning loss, 22:47-23:46, ~275 paired games each against the untaxed insane AI): the current policy **won 65%**, `fgself-62`'s last (its start) **65%**; heroes a game 0.6-0.7 against the start's 0.9-1.0 (the AI's ~1.7), and ~1,220 steps a second against ~1,600 (the cloning batches of Echo Isles' games, many units each, made the updates 6.2 s). No gain; off again.

**`fgself-64`** (from `fgself-62`'s last, 23:47): hero levels in the shaping again (`--hero-level-value 150`), now in the real game, where the policy fields 0.6-1.0 heroes a game to the AI's ~1.7 and 25-minute games give levelled heroes their weight (on duelfast's 6-minute games, with cheap heroes, it was neutral). Evaluation games against insane under the real rules: the current policy against `fgself-62`'s last and `fgself-63`'s last.

`fgself-64` (hero levels in the shaping, 23:47-00:45, ~133 paired games each against the untaxed insane AI): the current policy **won 69%** (0.92 heroes a game), `fgself-62`'s last (its start) **67%** (0.81), `fgself-63`'s last 73% (0.72). Level; the heroes a little up. Stopped for a machine shutdown.

**`fgself-65`** (from `fgself-64`'s last, 2026-10-16 09:07, after the restart, with its Adam state): the real game on, a longer run of the recipe as it stands (`--gamma 0.9997`, hero levels in the shaping, no cloning loss, the anchor its start). Evaluation games against insane under the real rules: the current policy against `fgself-64`'s last and `fgself-62`'s last.

`fgself-65` (the longer run, 09:07-11:14, ~255 paired games each against the untaxed insane AI): the current policy **won 67%** (0.72 heroes a game), `fgself-64`'s last (its start) **61%** (0.66), **`fgself-62`'s last 74%** (0.84). It rose above its start but not to `fgself-62`'s last, the best in every run since: the hero shaping (`fgself-64`, `fgself-65`) likely cost a little. Off again. The curriculum then taxed the AI only for the learner's weak races (undead 45% of its income, human 29%, orc and night elf none).

**`fgself-66`** (from `fgself-62`'s last, 11:15): more self-play (`--ai-share 0.5`, was 0.75: half the games against itself and past snapshots). The policy is above insane now. `fgself-62`'s recipe otherwise (γ 0.9997, no hero shaping, the anchor its start). Evaluation games against insane under the real rules: the current policy against `fgself-62`'s last and `fgself-65`'s last.

`fgself-66` (more self-play, 11:15-12:16, ~120 paired games each against the untaxed insane AI): the current policy **won 58%**, `fgself-62`'s last (its start) **74%**, `fgself-65`'s last 68%. Worse: with half its games against itself it trained less against the AI it is measured on. Back to 0.75.

**`fgself-67`** (from `fgself-62`'s last, 12:16): the weak races trained more (`--learner-races human:2,undead:2,orc:1,nightelf:1`: the learner plays human or undead in two thirds of its training games, 63% in its first 139; evaluation games stay uniform). In the real game orc and night elf win ~80% of their games against insane, human and undead ~25%. Evaluation games against insane under the real rules: the current policy against `fgself-62`'s last.

`fgself-67` (the weak races trained more, 12:16-13:44, ~290 paired games each against the untaxed insane AI): the current policy **won 64%**, `fgself-62`'s last (its start) **65%**. Level, the weak races too: human 24 of 77 against 26 of 75, undead 32 of 68 against 34 of 71 (orc 62 of 67, night elf 66 of 74). More games as human and undead did not teach them more in 1.5 hours. Next: whether the game refuses their builds on Echo Isles' terrain (undead build on blight; as human the altar came in two games of three).

**The builds, refused.** One 8-minute game a race against insane under the real rules (`fgself-62`'s last, build orders sent and taken): human 15 of 22, orc 13 of 21, **undead 25 of 100** (its ziggurats, its supply, 8 of 67: off the blight), **night elf 11 of 54** (its altar 1 of 25). The spots the policy learned on the flat duel map often miss Echo Isles' blight, trees and cliffs, and the harness tried other spots only up to 256 units away. Rings up to 768 units (16 spots a ring beyond 256): **undead 26 of 33, night elf 14 of 19**.

**`fgself-68`** (from `fgself-62`'s last, 14:09): the wider build search in every game (the evaluation games of the start too: the head to head measures what training adds; the fix itself shows in `fgself-62`'s last against its ~65-74% in the runs before). Races uniform again. Its first 63 games against insane: 52 won.

**The yardstick on Echo Isles.** The AI-record share weighted each matchup by the built-in AI's win rate on duelfast. On Echo Isles under the game's rules (its 1,500 games, ~620 decided between equals) the races stand otherwise: in mixed insane games the AI as human wins **14%**, as orc 43%, as night elf 58%, as undead **88%**. So against the AI's own record the policy's human (~37%) is strong and its undead (~45%) the weak one, not as the duelfast table made it. `real_games.py` now takes the AI's games of the run's own map (`demos-echo-1` for Echo Isles under the game's rules).

`fgself-68` (14:09-15:45, ~283 paired games each against the untaxed insane AI): the current policy **won 62%** (122% of the AI's own record on Echo Isles), `fgself-62`'s last (its start, with the wider build search too) **57%** (113%). A small gain, ahead in 3 of 5 twenty-minute windows; the new best, `runs/fgself-68/checkpoints/0000000008877917.pt`. The start checkpoint did not gain from the wider search alone (57-59% here against its ~65-74% in other runs' matchups): the policy had to learn to use it (its undead 29 of 65 against the start's 24 of 66, its night elf 52 of 52 against 41 of 52 at 15:13).

**`fgself-69`** (from `fgself-68`'s last, 15:46): fewer games against itself (`--ai-share 0.95`, was 0.75): more self-play cost 16 points (`fgself-66`). Evaluation games against insane under the real rules: the current policy against `fgself-68`'s last and `fgself-62`'s last.

`fgself-69` (15:46-17:46, ~294 paired games each against the untaxed insane AI): the current policy **won 71%** (146% of the AI's own record on Echo Isles), `fgself-68`'s last (its start) **69%** (143%), `fgself-62`'s last 61% (125%). Level with its start, ties a little fewer (3% against 5%); kept. The real game stands at ~70% of its games against insane. By race (the current policy): orc 55 of 58, night elf 48 of 50, human 31 of 66, undead 20 of 45. Against the AI's own record undead is the weak race: as undead the policy floods cheap units (37 acolytes and 74 ghouls a game against the AI's 8 and 21), reaches tier 2 in 53% of its games (the AI in all), has 1.05 heroes against 1.82 and hardly any crypt fiends, gargoyles or abominations; 23 kills for 72 units lost.

**`fgself-70`** (from `fgself-69`'s last, 17:48): a cloning loss on the AI's undead alone (`--bc-coef 0.02 --bc-data runs/fullgame/demos-echo-1-undead-wins --bc-winners 1 --bc-batch 32`: the 439 Echo Isles games it won as undead). The loss on all its games (`fgself-63`) was neutral; this one only shows the undead's way. Evaluation games against insane under the real rules: the current policy against `fgself-69`'s last.

`fgself-70` (17:48-19:17, ~406 paired games each against the untaxed insane AI): the current policy **won 63%**, `fgself-69`'s last (its start) **62%**. Level. Its undead changed its way (tier 2 in 64% of its games against 37%, fewer acolytes) but won no more (31 of 103 against 35 of 103), at 16% fewer steps a second. Off.

**Why undead loses: lumber.** In the evaluation games of `fgself-69` and `fgself-70` (per game minute, the policy against the AI as the same race): undead gold 813 against 873 but **lumber 83 against 213**; human 319 against 395, orc 295 against 301, night elf 402 against 394. Undead's lumber workers are its ghouls, its first fighters too: the policy fights with them and hardly sends them to the trees, so no lumber, no tier 2, and its ghoul flood.

**`fgself-71`** (from `fgself-69`'s last, 19:17): no penalty on unspent resources (`--float-penalty 0`, was 0.5, from when the duelfast policy hoarded): in the real game the AI holds 3,000-6,500 gold and lumber at a time (saving for tech), the policy 700-1,800, and a gathered resource lowered the potential until spent. The cloning loss off. Evaluation games against insane under the real rules: the current policy against `fgself-69`'s last.

`fgself-71` (19:17-20:15, ~217 paired games each against the untaxed insane AI): the current policy **won 64%** (117% of the AI's own record), `fgself-69`'s last (its start) **69%** (128%). Worse: it held more (~1,540 against ~1,210) and its undead gathered a little more lumber (86 against 62 a game minute, 15 of 52 won against 12 of 52), but its human fell (11 of 43 against 22 of 45). Back to 0.5.

A harvest order to a ghoul works and the features mark it a lumber worker: the policy just stopped giving it (the clone had it from the AI; RL on duelfast's cheap rules wore it away).

**`fgself-72`** (from `fgself-69`'s last, 20:15): a stronger cloning loss on the AI's undead alone (`--bc-coef 0.1`, five times `fgself-70`'s, the same 439 games; `--bc-batch 32`), for undead's lumber. Evaluation games against insane under the real rules: the current policy against `fgself-69`'s last.

`fgself-72` (20:15-21:15, ~290 paired games each against the untaxed insane AI): the current policy **won 59%**, `fgself-69`'s last (its start) **68%**. Worse. Its undead gathered more lumber (92 against 72 a game minute) but won fewer games (12 of 61 against 23 of 65), and its human (18 of 70 against 25 of 69) and night elf (66 of 77 against 76 of 78) fell too: at ×0.1 the cloning reached the other races through the shared network. Off; the undead cloning route is closed.

**`fgself-73`** (from `fgself-69`'s last, 21:16): half the shaping (`--shaping 0.5`, was 1.0), so the win itself weighs more: with γ 0.9997 the outcome reaches ~28 minutes back, and the shaping may steer toward trading material rather than winning. Evaluation games against insane under the real rules: the current policy against `fgself-69`'s last.

`fgself-73` (half the shaping, 21:16-22:46, ~345 paired games each against the untaxed insane AI): the current policy **won 72%** (139% of the AI's own record on Echo Isles), `fgself-69`'s last (its start) **62%** (121%), ahead in 4 of 5 twenty-minute windows. **A gain**, the weak races most: human 44 of 83 against 28 of 82, undead 48 of 98 against 31 of 96 (orc 84 of 88, night elf 73 of 78). With less weight on the material lead the policy plays for the win. New best: `runs/fgself-73/checkpoints/0000000008386478.pt`.

**`fgself-74`** (from `fgself-73`'s last, 22:46): less shaping again (`--shaping 0.25`). Evaluation games against insane under the real rules: the current policy against `fgself-73`'s last and `fgself-69`'s last.

`fgself-74` after an hour (22:46-23:47, ~138 paired games each; not judged): the current policy won 74%, `fgself-73`'s last (its start) 76%, `fgself-69`'s last 68%: level so far. **Paused 2026-10-16 23:47 for a machine shutdown** (stopped cleanly, status "stopped"). The best: `fgself-73`'s last, `runs/fgself-73/checkpoints/0000000008386478.pt` (~72-76% of its games against the untaxed insane AI in the real game on Echo Isles). Next: `fgself-74`'s question (`--shaping 0.25`) from `fgself-73`'s last, or less shaping still; undead's lumber (its ghouls).

**`fgself-75`** (from `fgself-73`'s last, 2026-10-17 09:58, after the restart, with its Adam state): `fgself-74`'s question again, a quarter of the shaping (`--shaping 0.25`). Evaluation games against insane under the real rules: the current policy against `fgself-73`'s last.

`fgself-75` after half an hour (09:58-10:31, ~128 paired games each, not judged): the current policy won 73%, `fgself-73`'s last (its start) 78%: level so far. The verdict at ~300 games each. 1,610 agent steps a second (8 actors of 6 games, 8 evaluation games), the learner's update 4.0 s. It started with a league of its own: forks start with their parent's league only since its launch (`--league-from`: the past snapshots that fit, their records, the curriculum's levels and balance).

`fgself-75` (a quarter of the shaping, 09:58-11:25, ~327 paired games each against the untaxed insane AI): the current policy **won 75%** (143% of the AI's own record on one matchup mix), `fgself-73`'s last (its start) **76%** (146%). **Level**, as `fgself-74` was after an hour: human 41 of 90 against 44 of 92, undead 48 of 72 against 48 of 74. The shaping line ends at half (`fgself-73`). The best stays `runs/fgself-73/checkpoints/0000000008386478.pt`.

**`fgself-76`** (from `fgself-73`'s last with its Adam state, 2026-10-17 11:27, `--shaping 0.5` again; the league from `fgself-75`): AlphaStar's main exploiter (`--exploiter-share 0.2`). A second network from the same weights trains only against the main policy and looks for what beats its rush. Its snapshots join the league when they beat the main policy in 70% of their last 50 games, and then it starts over. 20% of the launches are main-exploiter games (each side trains its own network), 76% are against the built-in AI. To run with the native inference loop, the exploiter's versions are numbered files (`runs/<run>/exploiter/<version>.pt`, `exploiter.json` names the newest) that a game plays like a past snapshot; its update has its own CUDA graphs. The run also logs the gradient noise scale (`grad/noise_scale`, McCandlish et al. 2018: the graphed and eager updates agree, and it costs no measurable time). Evaluation games against insane under the real rules: the current policy against `fgself-73`'s last.

`fgself-76` (a main exploiter, 11:27-13:23 with three restarts for infrastructure; the first run with the laptop's evaluation games, ~965 paired games each against the untaxed insane AI): the current policy **won 75%** (152% of the AI's own record on one matchup mix), `fgself-73`'s last **77%** (156%); after 12:30, ~540 games each, 74% against 75%. **Level.** The exploiter never reached 70% against the main policy: it started over twice after 300 updates (scores 0.50 and ~0.49), so no exploiter snapshot joined the league. Two fixes on the way: the exploiter's versions for new games only every 8 of its updates (`--exploiter-publish`: one every update kept ~10 networks in play and halved the main policy's steps), and the noise scale at fixed weights. The main policy got ~1,250 steps a second, the exploiter ~440.

**The gradient noise** (`fgself-76`, 60 measurements at fixed weights, every 10 updates; McCandlish et al. 2018's simple noise scale, in agent steps): the policy term ~31,000, the KL to the anchor ~1,800, the value term ~950; the whole network ~2,350 (in Adam's metric ~20,000). By part: the target heads ~79,000, the transformer layers 14,000-28,000, the order head ~15,000, the value head ~700. A minibatch of 256 steps carries under 1% of the policy's signal, while the KL and the value pull with clean gradients.

**`fgself-77`** (from `fgself-73`'s last with its Adam state, 13:25, the league from `fgself-76`, no exploiter): a quarter of the KL to the anchor (`--ref-kl 0.05`, was 0.2). The KL's clean pull may hold each fork at its start (`fgself-74`/`75`/`76` level); it was needed when the start was a clone. Evaluation games against insane under the real rules: the current policy against `fgself-73`'s last.

`fgself-77` (a quarter of the anchor's pull, 13:25-15:24, ~1,280 paired games each against the untaxed insane AI): level for its first ~1.5 hours, then ahead. After 14:50 (~380 games each) the current policy **won 82%** (166% of the AI's own record on one matchup mix), `fgself-73`'s last **74%** (152%): human 57 of 89 against 34 of 88, undead 47 of 75 against 39 of 71. **A gain, and the new best**: `runs/fgself-77/checkpoints/0000000011183066.pt`. The KL's clean pull held each fork at its start; with a quarter of it the policy moved (its KL to the anchor ~0.014, about three times the earlier forks') and improved after ~1.5 hours. Learning changes need that long before a verdict.

**`fgself-78`** (from `fgself-77`'s last grown with a strategy block, `runs/fullgame/grown/fgself-77-z.pt`, 15:25; the anchor moved to it, `--ref-kl 0.05`; a fresh league: the past snapshots have no strategy block): strategy conditioning, AlphaStar's statistic z (`fullgame/strategy.py`). Half the learner's training sides get a target strategy drawn for their race and matchup, winners first, from 8,072 (`runs/fullgame/strategies-echo.json`: the built-in AI's 1,500 Echo Isles games and 2,543 people's 1.29 1v1s; per item of the race's 64 most common, its first time within 15 minutes; people reach tier 2 in 93-100% of their games at ~3-5 minutes). The strategy is an input (a block of the global vector, zero weights at first) and a reward (a potential of minus 0.02 per item made by now by one of the side and the strategy, not both). Evaluation games get none. Evaluation games against insane under the real rules: the current policy against `fgself-77`'s last.

`fgself-78` (strategy conditioning, 15:25-17:58, ~1,250 paired games each against the untaxed insane AI on the main machine and the laptop): level for its first hour, then ahead. After 16:55 (~500 games each) the current policy **won 89%** (179% of the AI's own record on one matchup mix), `fgself-77`'s last **84%** (168%): undead 107 of 131 against 84 of 129, human 104 of 122 against 93 of 117, night elf 136 of 137 against 135 of 142, orc 104 of 118 against 105 of 111 (orc fell behind in the run's second hour, 69% against 92% after 16:25, and recovered). **A gain, and the new best**: `runs/fgself-78/checkpoints/0000000014224474.pt`. The strategies' mismatch at a game's end fell only a little (13.7 to ~13.3 items of the race's 64), and the training games with a strategy won 74% at first, ~82% later, as many as those without. How much of the gain is the strategies and how much more training at the lower anchor pull, this run does not tell.

**`fgself-79`** (from `fgself-78`'s last with its Adam state, 17:58; the anchor moved to it): more weight on following the strategy (`--z-reward 0.05`, was 0.02). Throughput: 2 evaluation games on the main machine (was 8; the laptop plays ~900 an hour) and a league of 6 past snapshots (was 12): the inference loop was saturated, half its calls for past snapshots and the evaluation checkpoint. Evaluation games against insane under the real rules: the current policy against `fgself-78`'s last.

`fgself-79` (the strategy reward x2.5, 17:58-19:52, ~730 paired games each after 18:55): the current policy **won 89%** (180% of the AI's own record on one matchup mix), `fgself-78`'s last **87%** (177%); after 19:20 87% and 87%. **Level.** It followed its strategies a little better (the mismatch 13.7 to 12.9 items; the games with one won 74% at first, 83% later), not enough to win more.

**`fgself-80`** (from `fgself-79`'s last, 19:52): the strategies taught by cloning, as AlphaStar's supervised policy learned what z means: an auxiliary cloning loss (`--bc-coef 0.05`) on the built-in AI's 1,500 Echo Isles games and people's 95 (winners and losers), each side with its own strategy as the input (`--bc-z`, `strategy.hindsight`). **Worse**: after 20:05 (~310 games each) 71% against `fgself-78`'s last's 88% (139% against 173% of the AI's record), undead 53% against 81%, heroes 0.44 a game against 0.80. The cloning loss fell (9.1 to 4.3) but dominated the gradient (its norm 0.8 against ~0.13) and took the policy from its anchor (KL 0.038 and rising, other runs ~0.01): cloning undid the fighting, as the clone of the AI's games and the undead cloning loss did before. Stopped after 30 minutes.

**`fgself-81`** (from `fgself-79`'s last with its Adam state, 20:25; the anchor moved to it; no cloning loss): the material shaping annealed to 0 (`--shaping-anneal 600`: 0.5 to 0 over ~45 minutes, then none). Potential shaping is the same as a critic of the learned part plus the potential (Wiewiora 2003): the material lead is a prior, and it over-values what lasts (the learner's ~33 workers in long losses against the AI's ~19). The critic, trained for many hours, can take the term in gradually. Evaluation games against insane under the real rules: the current policy against `fgself-78`'s last.

`fgself-81` (the shaping annealed to 0, 20:25-21:53): level while the weight fell, then behind; after 21:30 (~266 games each, the weight 0 since ~21:10) the current policy **won 84%** (160% of the AI's own record), `fgself-78`'s last **89%** (168%); human 65% against 81%. **Worse by ~5 points**, partly recovered: the value's loss rose while the material term went (0.00056 to ~0.0007) and came back. In lost games the learner made fewer workers (23 to 17) and buildings (18 to 16). Its KL to the anchor (`fgself-79`'s last, trained with the shaping) settled at ~0.024, ~40% above `fgself-79`'s ~0.0175.

**`fgself-82`** (from `fgself-81`'s last with its Adam state, 21:55; no shaping): a rolling KL anchor (`--ref-update 200`: the anchor becomes the current policy every 200 updates, ~15 minutes). The clean KL gradient (its critical batch ~2,100 steps, the policy's ~13,000) may have held `fgself-81` between the old anchor's habits and what the rewards without shaping want. Evaluation games against insane under the real rules: the current policy against `fgself-78`'s last.

`fgself-82` (no shaping, a rolling anchor every 200 updates, from `fgself-81`'s last, 21:55-22:53): **worse**: after 22:25 (~265 games each) 75% against `fgself-78`'s last's 88% (154% against 181%). Orc collapsed (17 of 69 against 57 of 66) while human (60 of 64 against 55 of 66) and undead rose: without the shaping and with the anchor following the policy, orc drifted. The KL climbed back to ~0.018 within ~170 updates of each move. **The shaping helps**: the line returns to `--shaping 0.5` with a fixed anchor.

**`fgself-83`** (from `fgself-79`'s last with its Adam state, 22:56; the anchor fixed at it, `--shaping 0.5`, the strategies as in `fgself-79`): workers counted only up to 20 in the shaping's material (`--worker-cap 20`). **Paused 2026-10-17 23:42 for a machine shutdown** after 45 minutes, not judged: 83% against `fgself-78`'s last's 87% (~272 games each, 163% against 172%); after 23:25 86% against 87% (~80 games each). The laptop's WSL failed at 23:10 (Tailscale answered, SSH hung); this machine's fallback played the evaluation games from then.

`fgself-83` (resumed 2026-10-18 10:33 with the same recipe, stopped 11:50): **level to slightly worse**. From the resume, ~780 paired games each: the current policy **won 83%** (164% of the AI's own record on one matchup mix), `fgself-78`'s last **87%** (170%); by 20-minute windows 177/161/155/164% against 191/154/174/170%. Heroes 0.62 a game against 0.77. In its long losses it still made ~24 workers against the AI's ~15 (`fgself-78`: ~35): the cap took away some of the pile-up but not the losses. The cap goes.

**`fgself-84`** (from `fgself-79`'s last with its Adam state, 11:51; the anchor fixed at it, `--shaping 0.5`, no worker cap): target strategies drawn only from people's replays (`--z-src human`: 5,074 of the pool's 8,072, winners first; the built-in AI's 2,998 out). People's plans differ from the AI's: fewer items (17.6 against 21.8 a strategy) and another tech path, so they may pull the policy out of the AI-like rush. Evaluation games against insane under the real rules: the current policy against `fgself-78`'s last.

`fgself-84` (people's strategies only, 11:51-13:20, ~1,000 paired games each): **level**. After 12:50 (~340 games each) the current policy **won 89%** (by 20-minute windows 176% and 168% of the AI's own record), `fgself-78`'s last **88%** (170%, 185%); over the whole run 87% against 88%. Heroes 0.66 a game against 0.82: people's plans did not bring more heroes or tech into the games without a target.

**`fgself-85`** (from `fgself-79`'s last with its Adam state, 13:21; the anchor fixed at it, `--shaping 0.5`, the strategies from both sources): the opponent curriculum goes on past zero (`--curriculum-floor -1`, new). When the learner keeps winning, the tax on the built-in AI's income turns into a bonus, up to 90% more of what it gathers (the level times `--curriculum-delay 0.9`), until the learner wins about half of those games. The levels sat at 0 (the real game) in most matchups, the learner winning ~87%, and almost all of its losses are long games (27 minutes on average) that it seldom meets in training: a richer AI plays longer games and techs. The test games stay against the AI at full strength. Evaluation games against insane under the real rules: the current policy against `fgself-78`'s last.

`fgself-85` (the AI's bonus, 13:21-14:50, ~1,020 paired games each): **worse**, by a collapse of one race. The levels reached the floor within ~25 minutes in 15 of 16 matchups: **the insane AI with 90% more income still lost ~85% of its training games** (12-13 minutes on average), so its weakness is how it plays, not what it has. For an hour the current policy led (90% against 88%); then its human collapsed: training games as human against the richer AI 75% won, then 37%, test games 21 of 64 against `fgself-78`'s last's 59 of 66, with 42 workers a game against 16. After 14:20 (~320 games each) 79% against 92%: human 44 of 102 against 85 of 96, while orc (77 of 78 against 72 of 76), undead (58 of 68 against 53 of 65) and night elf (95 of 97 against 92 of 94) gained a little. A one-race collapse after a change to the training signal, as orc in `fgself-82`.

**`fgself-86`** (from `fgself-79`'s last grown to 8 layers, `runs/fullgame/grown/fgself-79-8layer.pt`, 14:51; the anchor fixed at `fgself-79`'s last, 5 layers; `--shaping 0.5`, no AI bonus): a deeper network, 8 transformer layers (was 5; 4.51M weights, was 3.18M), the new ones the identity at first (`scripts/grow_model.py`), Adam's state carried for the old weights. The line has been level since `fgself-78`; the network is small for four races. The last growth (3 to 5 layers) gained only after ~2 hours (`fgself-45`, `fgself-51`): judged after ~2 hours. Evaluation games against insane under the real rules: the current policy against `fgself-78`'s last.

`fgself-86` (8 layers from `fgself-79`'s last, 14:51-15:50): stopped after an hour, not judged: 85% against `fgself-78`'s last's 90% (~600 games each), ~1,450 steps/s (an update 5.0 s, was 3.9). Behind as human (52 of 90 after 15:10 against 75 of 91, 29 workers a game against 20) and night elf (84 of 97 against 94 of 95).

**Human, behind in every run since `fgself-79`.** The human test games of each run, its current policy against `fgself-78`'s last (first hour, then after): `fgself-79` 72% against 80%, then 72% against 77%; `fgself-80` 68% against 87%; `fgself-81` 67% against 78%, then 67% against 81%; `fgself-83` 74% against 86%, then 58% against 81%; `fgself-84` 60% against 80%, then 65% against 78%; `fgself-85` 76% against 80%, then 44% against 88%; `fgself-86` 64% against 84% (`fgself-82` alone was ahead, 85% against 80%, while its orc collapsed). The human made more workers each time (22-26 a game against 19). `fgself-79` raised the strategy reward to 0.05, and `fgself-81` and `fgself-83` to `-86` all started from its last and kept that reward: their "level" verdicts hid the other races' gains behind a weaker human. The line goes back to `fgself-78`'s last and its recipe.

**`fgself-87`** (from `fgself-78`'s last grown to 8 layers, `runs/fullgame/grown/fgself-78-8layer.pt`, 15:51; the anchor `fgself-78`'s last; the strategy reward back to 0.02; the league of `fgself-78`): the deeper network again, from the best policy with its own recipe. Judged after ~2 hours. Evaluation games against insane under the real rules: the current policy against `fgself-78`'s last.

`fgself-87` (8 layers, 15:51-16:50): stopped at ~50 minutes, level (86% against 86% after ~260 games each, human 49 of 66 against 53 of 68: the human of `fgself-78`'s recipe is back), ~1,380 steps/s. The user prefers no larger network without a clear gain; the line turns to strategic variety.

**`fgself-88`** (from `fgself-78`'s last, 16:51; the anchor `fgself-78`'s last, its league; the user's direction: strategic variety first, a lower win rate accepted for a while): **enforced targets** on the fast rules. Three quarters of the learner's training games get a target strategy (`--z-share 0.75`), and while it is active (the strategies' 15 minutes, scaled) the side may only train, build and research the target's items, plus workers and supply (`--z-enforce 1`, new: `strategy.allowed` through the availability mask; moves, attacks, harvesting and spells stay free). The reward for following the target's timing stays (0.02). On `echofast` (Echo Isles with half the hit points and costs and a third of the build times; games ~10 minutes), the targets' times ×0.45 (`--z-time-scale`, new). Evaluation games against insane on `echofast`, three policies on the same matchups: the current policy without a target, the current policy with an enforced target (`current+z`, `--eval-z 1`, new: drawn as in training, winners of the matchup first) and `fgself-78`'s last (no target). The question: can the policy win with the builds it is given?

`fgself-88` (16:47-17:55; started 16:47, not 16:51): the laptop's worker lacked the strategies file until 17:19 (the manager ran older code; its `current+z` games failed), so the games with a target are from then. After ~1 hour, on `echofast` against insane: the current policy without a target **44%**, `fgself-78`'s last **42%** (~540 games each; both far below their real-rules rates); with an enforced target (`current+z`) **32%** (241 games). It barely followed its targets: ~15 of ~21 items missed at a game's end, flat over the run (14.6 in its first 400 training games with a target, 15.0 in its last 400), and ~2 kinds of unit a game with or without a target; tier 2 in ~40% of the games for all three. The mask keeps items outside the target out, but nothing pays for making the target's items beyond 0.02 each.

**`fgself-89`** (from `fgself-88`'s last with its Adam state, 17:55): five times the reward for following the target (`--z-reward 0.1` per item), the rest as in `fgself-88`.

`fgself-89` (17:55-19:30, resumed 18:52 to log `z_missed` and `z_extra`): it followed **no better**: ~15 of ~22 target items missed in its first 300 training games with a target and in its last 300. The detailed log (1,209 training games with a target): 9.4 buildings and units missed a game, 3.4 research, 1.4 heroes, the most missed buildings that need a worker sent to a spot (towers, shops, lumber mill, altar, graveyard); 0.6 items a game outside the target. After the resume the enforced targets cost no wins on `echofast`: `current+z` 47%, `current` 48%, `fgself-78`'s last 43% (~130 games each).

**`fgself-90`** (from `fgself-89`'s last with its Adam state, 19:31): the harness builds what the agent leaves undone of its target (`--z-build 1`, new: `TargetBuilder`): each due target building goes to the worker nearest the town hall (a spot 450-800 from it, then the harness's spot search) and each due hero to the altar, one a step, when affordable. The strategy reward back to 0.02. The aim: games with many builds and techs, so the agent learns to fight with them. Evaluation games on `echofast`: `current`, `current+z` (with the builder) and `fgself-78`'s last.

`fgself-90` (the harness builds the target's buildings and heroes; started 19:23, not 19:31; stopped 20:58, ~620 games each): with a target and the builder **39%**, level with `fgself-78`'s last (**37%**); the agent alone **60%** (after ~4 hours of training on `echofast`, which `fgself-78` never had). With a target it used a few more unit types (2.7 against 2.2 a game) and raised more heroes (1.15 against 0.82), and missed ~11 of ~22 target items itself (~14 without the builder). The gap to its own no-target win rate did not close: by 20-minute windows 71/83/103/74/67/80/71% of the AI's record against 128/122/131/128/106/124/138%. Variety in the games, not yet in the agent.

**`fgself-91`** (from `fgself-90`'s last with its Adam state, 20:59): the builder's orders become labels (`--z-build-labels 1`, new; `--opd-coef 0.05`): its accepted orders label the agent's step for on-policy distillation's loss, on the units it ordered only (`y_only`: the other units are not pushed to "no order"). The agent should learn to start the target's buildings and heroes itself.

`fgself-91` (20:59-21:10): stopped, a bug in the loss's weighting. The loss averaged over the labelled units alone (~1 in a few hundred steps), so those rare labels carried its full weight: the first update moved the policy by a KL of 0.15 (normally ~0.001), and the KL to the anchor went from ~0.016 to ~0.25 and stayed there.

**`fgself-92`** (from `fgself-90`'s last with its Adam state, 21:12; the league of `fgself-90`): `fgself-91` with each labelled unit weighted as one of all the step's own units (`distill_loss`: the mean over every own unit of the labelled steps; on-policy distillation's own steps, which label every unit, are unchanged).

`fgself-92` (21:12-22:50, ~655 paired games each on `echofast`): without a target **56%** (`fgself-90`: 60%), `fgself-78`'s last **40%**; with a target and the builder **26%** (`fgself-90`: 39%). Its KL to the anchor settled at ~0.03, its order matched the builder's 41% of the time at the end (18% at the start). Without a target its variety rose and faded again: 2.7 unit types and 21.6 buildings a game at 21:30-21:50, then 2.4 and 18.8 (`fgself-78`: 2.3-2.5, ~19). With a target the agent built on top of the builder (27-31 buildings a game) and lost more. **The labels did not make lasting variety.**

**`fgself-93`** (from `fgself-92`'s last with its Adam state, 22:51): the builder only teaches (`--z-build 2`, new): its orders for due target buildings and heroes are labels for the distillation loss and are never carried out, so the agent has to carry out its target itself. `current+z` is now the agent alone with its target enforced.

`fgself-93` **paused 2026-10-18 23:44** for a machine shutdown after 52 minutes, not judged. ~340 paired games each on `echofast`: without a target 55%, with its target enforced (no builder) 30%, `fgself-78`'s last 43%; with a target it missed ~11 of ~22 items on its own (as `fgself-92` with the builder) and built ~31 buildings a game; its order matched the teacher's 54% of the time. Without a target it played much like `fgself-78` (2.5 unit types against 2.3, ~18 buildings). Idea for later: trim each target to what a game of its length can afford (many targets come from long games and list more buildings than a 10-minute `echofast` game supports).

**Why the targets could not be followed** (2026-10-19, from `fgself-90`-`93`'s episodes). (a) Impossible items: the agent's orders are the built-in AI's (its vocabulary), so tavern heroes, mercenaries and shop units cannot be ordered, yet 54-93% of the targets (by race) had one: ~1.7 certain misses a game. (b) The old builder made only buildings and heroes: of the ~9.4 possible items missed a game, most were tier-2 halls and what needs them, units and research. (c) The times: the old pool's built-in AI sides included the insane AI, which gathers twice as much (engine notes), and every target came from games under the real rules, scaled by a flat 0.45. (d) A target covered 15 minutes whatever its game's length, so items the agent made after a short game's end counted as off-target.

**The fix.** A new pool, `runs/fullgame/strategies-echofast.json` (6,271 strategies): 600 games of the normal built-in AI against itself on `echofast` (native times; 64% decisive, median 18 minutes) and people's 1.29 1v1s with their times mapped per race onto the AI's (x0.72 human, x0.74 orc, x0.61 undead, x0.76 night elf: the ratio of the medians; no item earlier than the AI's 10% quantile; then in tech order, `strategy.consistent`); only items the agent can order (`--vocab`), the old pool's item slots kept (`--items-from`, the input the policy learned), each strategy covering only its game's length (`end`). The builder now follows the game's tech tree (`fullgame/techtree.py`, from the game's `*Func.txt`): it starts the earliest due item that is neither made nor under way (the game's start, finish and cancel events, the agent's own included), or the first thing it needs (a requirement, or the building that makes it); units, heroes, research and hall tiers at an idle building that makes them, buildings with the nearest free worker. The enforcement mask also lets through what the target's items need (`techtree.closure`).

**`fgself-94`** (from `fgself-93`'s last with its Adam state, 10:17; the league of `fgself-93`): the new pool at native times (`--z-time-scale 1`) and the new builder carrying out its orders (`--z-build 1`, its accepted orders also labels as in `fgself-92`). The question: can the targets be followed now? `fgself-90`'s builder and agent together still missed ~11 of ~22 items.

`fgself-94`'s first 50 minutes (restarted 10:23 with a log of the builder's orders per item, given and accepted, and each game's target and made times) showed ~4 of ~12 items due a minute before the game's end missed, and the game refused most of the builder's building orders: night elf 3 of ~400 accepted. Four bugs, found with the log and with single games of the builder alone (`builder_probe`: no policy, the workers sent to the mine first): (1) the nearest worker to the hall was often one inside a mine (a night elf's five wisps live in the entangled mine; the game refuses their build orders): now only workers outside a mine, and with none free a worker is trained first. (2) The strategies took each event's owner from the unit's id over the whole game, but the game reuses a dead unit's id: 240 built-in AI strategies had the other race's buildings (an undead side ordering human farms 67 times). Now the owner at the event's step, and only the race's own items (the old pool's item lists had the same mix-up; its slots stay, the foreign ones always empty). (3) Research ids are upper case like heroes', so a research under way counted as a hero and the second hero asked for the third tier. (4) Expansions: a second town hall went in the main base and the undead's haunted mine (a target order on the mine) was never given; now both go to the free gold mine nearest the hall. Restarted 10:39 with the fixes and the rebuilt pool. In single games the builder alone then made 11-16 of 13-19 due items per race (the rest due in the last minute, research queued behind busy buildings, or an expansion across creeps). The first ~100 training games after it still missed ~4.6 of ~13.6 items due a minute before the end, most never tried: the builder used only idle buildings, and a town hall training workers, or a barracks training units, never is, so the next tier (and the second hero after it) and most research never came; expansions failed (a haunted mine refused ~400 times; a second hall's worker walks into the creeps at the mine). Restarted 10:45: an idle building first, else the order queued; expansions left to the agent; an item refused three times given up for that game. Its first ~170 games still missed the tier-2 halls most (the keep 31 of 37 times, never ordered), and the reason was older than the builder: **the order costs had every upgraded building at its table cost, which holds its base's too** (a keep 705 gold and 415 lumber, its upgrade 320 and 210; a guard tower 100/70, its upgrade 70/50). The builder waited for money the upgrade does not need, and so did **the availability mask of every policy since it came in**: a next tier only once the player held a town hall's and a keep's gold together (on `echofast` 352/208 instead of 160/106). Restarted 10:53 with the costs fixed (`costs.py`: an upgrade costs the difference to its base). The policies' own tier-2 timing may change with it.

More fixes, each found in the games' logs (what the builder gave, what the game accepted, the refused orders' context, why a due item was not ordered) and checked in single games: the game takes no upgrade or hero behind others in a building's queue (they now wait for an idle building; a hall training workers back to back has their queue cancelled first, its cost refunded); out of supply a unit waits in its building's queue and blocks a next tier behind it (supply first); a human's or orc's building whose worker was taken away stays an unfinished frame (a free worker resumes it after 10 s without progress). Most refusals left were the builder repeating what the agent ordered in the same step (the item made 18 s later: a hero's training). **The last cause was money**: the agent spends everything on units, so a next tier stayed short of gold ~2,000 steps a game. The builder now keeps the cost of the target's next item that is short of gold or lumber from the agent's other orders (the availability mask, `features.affordable`; the item itself and free orders stay). Items due a minute before the game's end and missed, per game, with the builder carrying out:

| `fgself-94` from | due | missed |
|---|---|---|
| 10:46 (queueing) | 13.3 | 4.5 |
| 11:32 (frames resumed) | 12.7 | 4.3 |
| 12:06 (the cost reserved) | 13.3 | **2.1** |

Of the 2.1, ~0.3 are expansions (left to the agent).

`fgself-94` (12:06-12:56, ~390 paired games each on `echofast` against insane): `current` **72%**, `current+z` (a target, the builder carrying out) **39%** (by 20-minute windows 79/73/97% of the AI's record), `fgself-78`'s last **42%**. With the builder 82-85% of the due items were made; the agent made 43-47% of those itself, flat for 50 minutes of training on the builder's labels. **The targets can be followed; the agent does not follow them on its own yet.**

**`fgself-95`** (from `fgself-94`'s last with its Adam state, 12:56): the builder only teaches (`--z-build 2`): its orders are labels, never carried out; a busy building's decision is labelled too (a hall's next tier while it trains workers). The cost of the target's next item stays reserved from the agent's other orders. The question: does the agent follow its targets on its own, and at what cost in wins?

`fgself-95` after 30 minutes (~210 paired games each, not judged): `current` 72%, `current+z` (the agent alone with its target) **44%**, `fgself-78`'s last 38%. **The agent now follows its targets on its own**: in training it made 80-82% of the ~13 items due a minute before the game's end (`fgself-93`, the old pool and builder: ~50%), in `current+z` games 88%. With a target it plays far more varied games: 2.9 unit types a game (no target 1.6, `fgself-78` 1.4), 27 buildings (17, 20), tier 2 in 74% of its games (52%, 59%), 1.56 heroes (1.01, 0.89).

`fgself-95` judged at 13:56 (~440 paired games each): `current` **73%**, `current+z` **44%**, `fgself-78`'s last **38%**; training games made 81% of the due target items in every 20 minutes (no rise yet). **Verdict: the agent follows sensible build orders on its own** (a question answered), and with a target it wins less: by race with a target / without, night elf 84% / 88%, orc **28% / 84%**, undead 30% / 63%, human 26% / 53% (with people's targets 42%, with the built-in AI's 53%). `current+z` by 20-minute windows 83/82/91% of the AI's record: slowly rising. `fgself-95` keeps running: the agent now has to learn to win with these builds (orc, whose rush the targets replace, loses most).

`fgself-95` 13:56-14:56 (~460 paired games each): `current` 71%, `current+z` **50%** (by 20-minute windows 92/105/122% of the AI's record: it learns to win with its targets), `fgself-78`'s last 39%; target items made on its own steady at 80%.

**Correction (16:04).** The laptop's evaluation worker read `--z-build` as a flag: 2 (a teacher only) became True, which the games compare as 1, so its `current+z` games had the builder carry out the target. Nearly all of `fgself-93`'s and `fgself-95`'s evaluation games were the laptop's, so their `current+z` numbers above (wins, and the 88% of due items made) are with the builder carrying out, **not the agent alone**. The training games ran on this machine and are right: there the agent alone made 80-82% of the due items. Fixed (`evalworker.run_config`: a number) and the worker restarted at 16:04:55; `current+z` is the agent alone from there.

`fgself-95` 16:06-16:26 with the fixed worker (~170 games each): `current` 78%, `current+z` (**the agent alone**) **58%**, made 82% of its ~11.5 due target items, `fgself-78`'s last 48%. The corrected numbers are close to the ones with the builder.

**`fgself-96`** (from `fgself-95`'s last, 16:30; head to head with `fgself-95`'s and `fgself-78`'s last): **GRPO from mid-game branches** (the user's idea; `--grpo-share 1 --grpo-group 4 --grpo-coef 0.3`). The first game of a launch between agents (seeded) records its commands up to a random step between 1 and 8 game minutes; 4 fresh games of that actor replay them to there (no inference: the recorded commands) and play on with new samples; when the group ends each branch's advantages get 0.3 times its return from the branch less the group's mean (per trained side). The advantages' spread is ~0.05 (`adv/std`, now logged): return gaps of ~0.1 weigh about as much as a step's advantage; a won and a lost branch ~6 times. More games between agents for it (`--ai-share 0.8`, from 0.95). Getting the replays exact took three restarts: a check that a branch reaches its source's state (`branch_same`, the players' units), the first step a replay differs in the log, and then the critters: they wander apart in two seeded games from ~5 s and block workers, so seeded games now start without them (`W3S_RemoveCritters`). From 16:47, 9 of the first 13 branches reached their source's state exactly (the others differed from steps 150-262: tree harvesting, a hero's skill; a branch that differs still starts from almost the same state).

By 16:56 only 45 of 89 branches were exact: fights with creeps do not replay alike (a probe: agents attack-moving to the nearest creeps, apart from 13 s), like the critters and the built-in AI. Restarted 17:00: only exact branches get the group's advantage (two needed; the others train with nothing added). Judged from 17:00.

`fgself-96` judged at 17:56 (~330 paired games each from 17:01, `echofast` against insane): `current` **73%**, `fgself-95`'s last **73%**, `current+z` 54%, `fgself-78`'s last 37%: **GRPO level** after an hour. Its signal was small: 54% of the branches exact, and in most groups the returns hardly differed (median largest gap 0.04; a few groups split won and lost from one state). Target items made on its own in training 76% (`fgself-95` 80-82%; more games between agents). Next, `fgself-97`: the same with a stronger signal: branches at 1-4 game minutes (before most fights with creeps: more of them exact) and the group's advantage at full weight (`--grpo-coef 1`).

`fgself-97` (17:58-18:56, ~350 paired games each): `current` **67%** (by 20-minute windows 156/149/114% of the AI's record), `fgself-95`'s last **71%** (147/154/143%), `current+z` 55%, `fgself-78`'s last 39%. Earlier branches were hardly more exact (56%), and most groups' returns stayed within 0.04. **GRPO from mid-game branches did not help**, at 0.3 or full weight; the line stops (the code stays: `--grpo-share`). What would make it work: branches the game replays exactly through fights with creeps (it does not: engine notes), or a value-free signal with less noise than one game's outcome.

**`fgself-98`** (from `fgself-95`'s last with its Adam state, 19:06; the league of `fgself-95`): **back to the real game**, Echo Isles under the game's rules against insane, the yardstick. Targets from a pool at the real rules' times (`runs/fullgame/strategies-echo-real.json`, 6,578: the normal built-in AI's sides of the 1,500 real-rules Echo Isles games and people's 1.29 1v1s at their own times, orderable items only, the owners at each event's step), `--z-time-scale 1`, the builder teaching, the cost reserve, the fixed upgrade costs. Head to head with `fgself-78`'s last (the best real-game policy, ~87-89% against insane) and `fgself-95`'s last. The question: does the line that follows build orders on `echofast` carry over to the real game, and how does it stand against `fgself-78`?

`fgself-98` at 20:26 (~410 paired games each, the real game against insane): `current` **87%**, `fgself-95`'s last **86%**, `fgself-78`'s last **86%**: **level with the best real-game policy**, the races apart: human 80% / 82% / 71%, undead 71% / 67% / **83%**, night elf ~97-100%, orc ~92-96%. The `echofast` line carried over (human up ~10 points, undead down ~12-16). With a target **61%** (by 20-minute windows 124/121/126/135% of the AI's record, rising; human 24%, night elf 90%, orc 65%, undead 66%); in training the agent alone made 68-70% of the due items (on `echofast` 80-82%: the real rules' times are new to it). It keeps training on the real rules for now.

`fgself-98` 20:26-21:26 (~295 paired games each): `current` **78%** (human 59%, undead 58%), `current+z` 64% (human 31%), `fgself-95`'s last **90%** (human 87%, undead 75%), `fgself-78`'s last **87%** (human 76%, undead 84%). **Training on the real rules with targets in 75% of the games pulled its play without one down** (87% -> 78%), human and undead most. Over both windows `fgself-95`'s last won 617 of 705 (87.5%), `fgself-78`'s 606 of 703 (86.2%): the `echofast` line's last is the best real-game policy so far, by a little. Next, `fgself-99`: from `fgself-95`'s last on the real rules with targets in 30% of the games (`--z-share 0.3`).

`fgself-99` (21:29-22:26, ~285 paired games each): `current` **80%** (human 64%, undead 63%; by 20-minute windows 166/178/134% of the AI's record), `current+z` 47%, `fgself-95`'s last **87%**, `fgself-78`'s last **85%**. Fewer target games did not stop it: **both runs on the real rules pulled `fgself-95`'s last down**, while `echofast` training made it. Next: train on `echofast` (as `fgself-95`, targets in 75% of the games) and evaluate on the real game (`--eval-map`, new: the evaluation games on another map, with its costs and its own strategy pool).

**`fgself-100`** (from `fgself-95`'s last with its Adam state, 22:32; the league of `fgself-95`): `fgself-95`'s recipe goes on (`echofast`, targets in 75% of the games, the builder teaching, the cost reserve), its evaluation games the real game (`--eval-map "(2)EchoIsles"`, `--eval-z-pool runs/fullgame/strategies-echo-real.json`). Head to head with `fgself-78`'s and `fgself-95`'s last. The question: does more `echofast` training lift the real game past 87.5%?

`fgself-100` at 23:56 (~455 paired games each, the real game against insane): `current` **88%**, `fgself-95`'s last **88%**, `fgself-78`'s last **84%**; with a target (the real-rules pool) **73%**, the agent alone making ~70% of its due items. In the last half hour 92% / 90% / 87% / 76%. `echofast` training keeps the real game where it was (the runs on the real rules pulled it down) and may lift it a little; it goes on.

`fgself-100` 23:56-00:56 (~350 paired games each, the real game against insane): `current` **95%** (194% of the AI's record; human 90%, undead 92%, orc 100%, night elf 100%), `fgself-78`'s last **88%** (human 80%, undead 83%), `fgself-95`'s last **88%**; with a target **74%** (human 50%, undead 78%, orc 81%, night elf 94%). **A new best in the real game**: about 2.5 hours more of `echofast` training, judged on the real game (latest checkpoint at 00:56: `runs/fgself-100/checkpoints/0000000012992241.pt`). It goes on.

`fgself-100` **paused 2026-10-20 01:05** for a machine shutdown, still improving: from 23:56 ~400 paired games each, `current` **95%**, `fgself-95`'s last 89%, `fgself-78`'s last 88%, with a target 74%. Its last checkpoint: `runs/fgself-100/checkpoints/0000000013649982.pt`.
Resumed 2026-10-20 10:13 after the reboot (`selfplay_restart.py`, the same run).

`fgself-100` 10:15-11:24 after the resume (~390 paired games each, the real game against insane): `current` **92%** (human 91%, undead 82%, orc 97%, night elf 99%), `fgself-95`'s last **87%**, `fgself-78`'s last **86%** (human 67%); with a target **76%** (human 50%), the agent alone making 70% of its ~13 due items. Over the ~800 games since 23:56 `current` won ~93-95%, `fgself-78` ~87%: **confirmed, the real game's best**. By 20-minute windows 201/188/176/176% of the AI's record: it may be levelling off. Latest checkpoint at 11:24: `runs/fgself-100/checkpoints/0000000020063584.pt`.

`fgself-100` 11:24-11:54 (~155 each): `current` **87%**, `fgself-78`'s last 90%, `fgself-95`'s last 86%. After its peak (95% at 01:05, 97% right after the resume) it slid, its KL term still pulling toward `fgself-78` (ref_kl ~0.11). Its best checkpoint: `runs/fgself-100/checkpoints/0000000013649982.pt` (01:05).

**`fgself-101`** (from `fgself-100`'s 01:05 checkpoint, 11:56; the league of `fgself-100`): the anchor moved to that checkpoint (`--init`): the KL keeps the policy near its best instead of `fgself-78`. Head to head with that checkpoint and `fgself-78`'s last. (Earlier anchor moves were level, `fgself-40`, `-46`; a rolling anchor was worse, `fgself-82`.)

`fgself-101` judged at 12:54 (~345 paired games each, the real game against insane): `current` **96%** (human 97%, undead 86%, orc 100%, night elf 100%), its start (`fgself-100`'s 01:05 checkpoint) **94%**, `fgself-78`'s last **87%**; with a target **81%** (`fgself-100`: 74-76%; human **69%**, from ~50%). By 20-minute windows 192/172/182% of the AI's record: **no slide** (ref_kl ~0.015 to the new anchor; `fgself-100`'s was ~0.11 to `fgself-78`). The anchor at the best checkpoint holds the policy there and the target games got better. It goes on.

`fgself-101` 12:54-13:54 (~360 paired games each): `current` **93%** (human 83%, undead 90%), its start **93%** (human 93%, undead 84%), `fgself-78`'s last **87%**; with a target **83%** (human 65%). By 20-minute windows 201/194/173%: level with its start, holding (not sliding) but not climbing. Night elf and orc win 96-100%; human and undead are what is left. Next, `fgself-102`: training games' learner races weighted to them (`--learner-races human:0.35,undead:0.35,orc:0.15,nightelf:0.15`; evaluation games stay uniform).

A video of `fgself-101` (32 minutes, undead against insane night elf, lost) showed the camera at the night elf base the whole game: four of the learner's ghouls stood idle there from ~5 minutes to the end (their hit points 1, 7, 67 and 70 the whole time, unattacked) beside the AI's wisps, and the video's director counted units near enemies as a fight. A fight now needs blows struck (`video.Director`: hit points lost within ~1,500 of it in the last 4 s); that game's camera then went to real fights ~45% of the time and cycled through both armies and bases otherwise. The idle ghouls are a finding of their own: undead loses most games against night elf, long ones, and parks ghouls (its lumber harvesters) at the enemy base.

`fgself-102` judged at 14:54 (~355 paired games each, the real game against insane): `current` **95%** (human 92%, undead **89%**, orc 97%, night elf 100%), `fgself-100`'s 01:05 checkpoint **92%** (undead 79%), `fgself-101`'s checkpoint (its start) **90%** (undead 70%); with a target 83% (human 62%). **Training the weak races more lifted undead by 10-19 points** and the whole by 3-5; human held. It goes on. Next (proposed to the user: the road past the built-in AI): a league that trains mostly against itself, past snapshots and exploiters, with a yardstick of its own (the policy against frozen snapshots, not only the built-in AI, which it beats ~95%).

The yardstick (`--eval-vs`, new, 15:58): a share of the evaluation launches (here a quarter) play the current policy against one of `--eval-checkpoints` (agents on both sides, no tax; rows `eval_key` "vs <run>/<checkpoint>"; the league's records leave them out). `fgself-102` restarted with it against `fgself-101`'s and `fgself-100`'s best checkpoints: over 50% against them is progress the built-in AI no longer shows.

First snapshot games (16:11, small): `fgself-102`'s current policy beat `fgself-101`'s checkpoint in 69% of 36 games and `fgself-100`'s best in 21 of 21.

**`fgself-103`** (from `fgself-102`'s checkpoint 11841833, 16:13; the league of `fgself-102`): **AlphaStar's league** (the user: "keep improving with self-play to become superhuman", "get inspiration from Dota Five and AlphaStar"; [ideas](ideas-from-alphastar-and-openai-five.md), "Past the built-in AI"). The built-in AI in 15% of the launches (a floor and a curriculum), a main exploiter in 15% (`--exploiter-share 0.15`), of the rest 40% against itself and 60% against past snapshots by PFSP (`--self-share 0.4`, `--max-past 12`): ~29% self, ~43% past, 15% exploiter, ~13% the built-in AI (AlphaStar 35/50/15). The yardstick: the current policy against its start and `fgself-100`'s best in 40% of the evaluation launches (`--eval-vs 0.4`), beside the built-in AI. The win rate against insane may dip first (`fgself-66`: -16 points in an hour with half its games against itself); the question is whether it beats its own past more.

Its first 10 minutes read 26% against its start in 65 snapshot games, but the races decided it: the launches' races are drawn once for a launch's games, and the current policy had played human against the checkpoint's orc in 19 (lost all) and undead against night elf in 10 (lost all). Restarted 16:25 with each matchup played both ways round (the next snapshot launch swaps the races); judged from there.

`fgself-103` at 18:24 (2 hours of the league; ~600 mirrored snapshot games each, ~55 against insane each): the current policy beat its start in **55%** (by thirds of the games 56/49/61%) and `fgself-100`'s best in **55%** (55/52/58%); against insane it held **97%** (its start 100%, `fgself-100`'s best 95%; with a target 83%). The main exploiter restarted once (it reached 70% against the main agent). **The league is a small gain that grows, without the dip against the built-in AI** that half its games against itself cost `fgself-66`. It goes on.

`fgself-103` judged at 19:24 (3 hours of the league; ~900 mirrored snapshot games each, ~80 against insane each): against its start by hour **51/62/49%**, against `fgself-100`'s best **54/56/48%**; against insane **100/93/79%** (its start 100/100/93%, `fgself-100`'s best 93/96/96%). **The league gained in its second hour, then drifted back, and the built-in AI started to slip: no lasting gain** in three hours (AlphaStar's took weeks; the exploiter restarted once). Its second hour's checkpoint (17:56, `runs/fgself-103/checkpoints/0000000008386382.pt`) is kept as a reference.

**`fgself-104`** (from `fgself-102`'s checkpoint 11841833, `fgself-103`'s start; `fgself-102`'s recipe and league: the built-in AI in 95% of the launches, the weak races trained more): OpenAI Five's reward decay (`--shaping-decay 0.6`, new: every reward but the outcome times 0.6^(game minutes / 10)). Snapshot games (`--eval-vs 0.4`) against its start and the league's best hour (17:56): do the decay or the league beat the same start more?

`fgself-104` (19:27-20:24, ~250-350 mirrored snapshot games each): against its start **46%** (by halves 43/50%), against the league's best hour **44%** (49/40%); against insane 97% (34 games; its start 100%). **The reward decay did not help** (slightly worse than its start after an hour).

**`fgself-105`** (from the same start, `fgself-102`'s checkpoint 11841833, `fgself-102`'s recipe): OpenAI Five's fresh data, each sample used once (`--epochs 1`, was 2: updates twice as often, staleness lower). The same snapshot games.

`fgself-105` (20:24-21:26, ~350 mirrored snapshot games each): against its start **54%** (halves 58/49%), against the league's best hour **48%** (49/47%); against insane 100% (38 games); staleness ~1.1 (was ~1.5-1.8). **Level**: fresher data, no stronger policy. The three ideas since `fgself-102` (the league, the reward decay, one epoch) all came out about level with their common start.

**`fgself-106`** (from `fgself-105`'s last, 21:27; its recipe): the line goes on, and its evaluation worker also plays **classifier-free guidance toward the target** (the user's idea from image generation; `--eval-guide 2`, new: `current+zg`, current+z's games with the order's logits free + 2 * (with - free), the strategy's block zeroed for "free"). The question: does guidance make the agent follow its targets more (due items made), and at what cost in wins? Snapshot games now 20% of the evaluation launches (against its start and `fgself-102`'s 11841833).

**Result (22:08, ~32 games each on the laptop):** guidance did not help. `current+zg` won 50% against insane (16 of 32), `current+z` 91%; the due target items missed: 5.3 and 5.0 a game. Stopped.

**`fgself-107`** (from `fgself-106`'s checkpoint 3453342, 22:08; its recipe and league): **what the agent could not see or do** (the user's question). `--inputs 3`: per hero its abilities (level, can be cast now: learned, cooled down, the mana; cooldown left; a skill point may go there) and items (held, usable, heals, mana, a town portal), what each building makes (a worker, an army unit, a hero, research, its next tier), the time of day, and the enemy units and buildings it saw and no longer sees (where and how long ago; an attack on one is an attack-move there). New channels start with zero weights. `--tech-mask`: orders the player lacks the tech for are masked (requirements, research levels, the hero limit); `--valid-mask`: a hero's orders it can't carry out now. Checked first on recorded games: the masks block only what the game would refuse then.

**Result (22:55, ~45 minutes):** worse than its start: 41% of 225 head-to-head games, flat by thirds (40/43/40%); 89% against insane (54 games), 67% against `fgself-102`'s 11841833. Suspect: a remembered enemy is a new token that looks like one in sight until its marker's weights learn, and an attack on it sends units to a stale place. Stopped.

**`fgself-108`** (the same start, recipe and league, 22:57): the ablation, `--inputs 2` (hero state, what buildings make, the clock; no memory), the masks on. Level with the start: the memory hurt (next: fade the remembered enemies in on a schedule); below: the hero inputs or the masks.

**Result of `fgself-108` (23:23, ~25 minutes):** head to head 48% of 88 against its start, 55% of 154 against `fgself-102`'s 11841833, 93% against insane. On the rating scale (more games) the start, `fgself-106`'s last, is 2070 ± 19 (2128 on fewer games), `fgself-108`'s first hour 2024 ± 22, `fgself-107`'s 2044 ± 18, `fgself-102`'s best hour 2062: the new inputs and masks are about level, their start a strong checkpoint. Kept as the base; the memory waits for its fade-in run (`--memory-fade`, new: older networks and the anchor never see the remembered enemies, attention bias -30; the learner's rises from -8 to 0).

**`fgself-109`** (from `fgself-108`'s last, 23:26; its recipe and league): **strategic variety**, the user's direction. Narrow targets (`runs/fullgame/strategies-echofast-narrow.json`, the real game's for evaluation: a game's heroes, its main army and what they need, drawn evenly over 230-307 builds a race; other buildings and research free, other heroes and army units not) and more kinds of conditioning (`--z-modes 0.15 --z-mode-reward 0.1`: no rush, rush, expand, no expansion, workers, creeping, harass, scouting, army size, low upkeep). Watch the due items made, each kind kept, the win rate and the rating.

**Result of `fgself-109` (00:24, 1 hour):** level on the rating (its first hour 2084.5 ± 15, its start 2084.7 ± 18), 96% against insane, 49% of 347 head to head with its start. With a target it wins 68% (rating 1649 ± 37) and makes 75% of the due target items (80% in training). Kinds kept: workers 88%, army size 84%, no expansion 77%, rush 76%, harass 75%, scouting 71%, creeping 70%, expand 47%, low upkeep 34%, no rush 14%; the hard ones did not improve between the halves (no rush 18% → 10%): a whole violation is worth 0.1 against the outcome's 1.

**`fgself-110`** (from `fgself-109`'s last, 00:25; its recipe and league): **rewards by part** (GDPO, the user's idea from LLM RL; `--reward-parts 1`): a value and an advantage per part (outcome; shaping; strategy, the target's items and the kinds), each normalized on its own and summed with equal weights; a part spreading less than a tenth of the outcome's is not scaled up. At the first update the strategy part's advantages spread 0.008 to the outcome's 0.079: now on the same footing. The question: are the kinds and targets followed more, and what does it cost in wins?

**Result of `fgself-110` (01:00, 31 minutes):** worse. Rating 2033 ± 22 to its start's 2084 ± 28, 40% of 150 head to head with it; with a target 51% against insane (68% in `fgself-109`), the targets and the kinds followed as before. Equal weights also gave the material shaping its full weight back (half had helped, `fgself-73`); a retry would weigh it less (`--part-weights 1,0.5,1`). Stopped for the night.

**`fgself-111`** (from `fgself-109`'s last, 2026-10-21 10:10; its recipe and league): **what the agent could not do** (the user's question): the neutral buildings' wares as orders (`--neutral 1`: a tavern's heroes, a camp's mercenaries, the goblin merchant's items, bought at the building the pointer picks; the tavern heroes' skills and spells; every network grown to the vocabulary, new orders rare at first), a heuristic teacher (`--teach 1`, the user's idea), and narrow targets with tavern heroes and mercenaries (`strategies-*-narrow-tavern.json`: ~2,600 of ~5,300 targets have one, people's builds; the race filter had dropped them). Two restarts in its first half hour: the shops always in the view (the game's minimap shows the neutral buildings; in sight only, the tavern could not be pointed at and no lesson for it counted: none of 21 tavern-hero targets made in 60 games), and a hired unit counted as made (a purchase has no training event). Then, still no tavern hero hired in ~2,400 games, one cause after another: the teacher sent only workers in sight of the tavern (now the nearest worker, however far); its few lessons drowned among the target builder's labels (`--teach-coef 0.5`, a cloning term of their own); a worker sent to buy was turned back half a second later by its next order (a buy order now holds 40 s, `BUY_COMMIT_STEPS`); a buy order's pointer seldom picked the shop (it now goes to the nearest building that sells it, as a harvest order goes to the nearest mine). A traced game (12:00) showed the rest: ~36 buy orders a game, all the goblin merchant's items, all sent to workers inside the gold mine and refused. Now the merchant's items are a hero's only (`neutral.who_buys`) and a unit inside a mine gets no buy order (the hero mask's not-hidden bit); a resumed run takes these tables from the game's data, not its checkpoint (`model.trained`). Still none: a probe game with a tavern-first target built no altar and held under 100 gold all game (footmen), so the hire stayed unaffordable and masked; the target builder's reserve (the next item's cost kept from the other orders) skipped hires, now it saves for them (12:22). And the root cause (12:45): every buy order from a base was refused by the game. The tavern stands in the fog at the start, and no order can target a unit the player hasn't seen; the few hires (5 in ~700 games) came from units already beside a shop. Now the buyer walks to the shop and right-clicks it in reach (a worker from its start hired a Pandaren 77 steps later), the purchase pending 160 steps and the buyer held 120. Games with the tavern hero due win about a third either way (only long games reach it).

**Result of `fgself-111` (14:25, 4 hours 15 minutes, 1 hour 40 minutes since the fog fix):** level, and it now buys. Rating 2031 ± 20 (its second hour) to its start's 2010 ± 9; 49% of 1,082 head to head with its start, 50% of 1,077 with `fgself-102`; 93% of 284 against insane (its start 97% of 287, the same hours); with a target 61% of 224 (rating 1624 ± 65 to `fgself-109`'s 1643 ± 37). Due target items made 74% (night elf 82%, orc 78%, human 77%, undead 69%). In the 2,751 training games of its first 40 minutes after the fix: 413 tavern heroes hired (Alchemist 81, Beastmaster 70, Pandaren 65, Naga 63, Dark Ranger 30, Tinker 30, Pit Lord 15, Firelord 9) and ~1,500 mercenaries (forest troll shadow priests 596, ogre maulers 460, troll berserkers 267, mud golems 174), mostly ordered by workers. Kept as the next base: what it can do grew at no cost.

**`fgself-112`** (from `fgself-111`'s last, 2026-10-21 14:30; its recipe and league): **self-distillation with privileged information** (OPSD, the user's idea from LLM training; `--opsd-share 0.2 --opsd-coef 0.05`): a fifth of the learner's training games act on a privileged view (every enemy unit, marked when out of sight, and the opponent's resources; a new input path that starts at zero), so the network learns to use what a player can't see; in the other games a KL pulls its orders from the player's view toward the same network's orders given the privileged view (no gradient through the teacher). Never in evaluation. Its first hour went to four bugs: the remembered-enemy columns switched on by the rows' width (every actor), a privileged batch narrower than its own units, the graphed update's statistics, and the teacher pass's pointers past the privileged views' width (a device-side assert); checkpoints are now saved whole or not at all (a stopped learner left a truncated one). **And a finding about `--neutral`:** the laptop's evaluation process had predated the neutral orders, so `fgself-111`'s real-game evaluations never bought anything. With them on (from `fgself-112`), every policy's workers bought ~2 mercenaries a game, never taught (a new order's bias is low, but each worker may give one every half second), and won less against insane: `fgself-109` 84% of 37 (94-98% before), `fgself-111` 92% of 36, `fgself-112` 81% of 47; the games with a purchase won 74-90%, those without 90-94%. Restarted at 15:03 with `--neutral 2`: the tavern's heroes and the merchant's items, no mercenaries (none in the targets either).

**Result of `fgself-112` (16:30, 1 hour 25 minutes since the restart):** no gain. 44% of 327 head to head with its start (thirds 45%, 39%, 50%); against insane 89% of 72 (its start 95% of 74 in the same games' rules); with a target 56% of 70. On the rating scale its second hour 1937 ± 28 to its start's 1908 ± 16 (both in these rules), its first hour 1857 ± 14 (the mercenaries and the crashes). The self-distillation's KL stayed tiny (~0.004): the teacher, the same network with a privileged input that starts at zero, had barely learned to use what it sees by then. A retry would need the privileged games to learn first (a longer run, or the privileged path trained on its own before the KL). **Ratings fixed:** a checkpoint from a run without the shops, evaluated with them, is now a player of its own (`fgself-109/4933795 +shops`: 1730 ± 17, to 2010 ± 9 without); as one player it had made wins over a weakened `fgself-109` count as wins over a strong one.

**`fgself-113`** (from `fgself-111`'s last, 2026-10-21 16:28; its recipe and league, `--neutral 2`): **scenarios** (the user's idea: the micro curriculum's way kept on; `--scenarios clash,raid,rich --scenario-share 0.25`): a quarter of the training games start in a situation the harness sets up: an army and a level-3 hero each between the bases (clash), the opponent's army at the learner's base (raid), or 1,500 gold and 700 lumber more each (rich: tiers, tech and heroes early). Evaluation games never have one. The question: does meeting these situations more often make the policy stronger in whole games?

**`fgself-113` stopped (17:35, 1 hour): confounded.** 36% of 186 head to head with its start from its first third (32%), with a target 46% of 50; against insane level (90% to its start's 89%). Looking for why both forks of `fgself-111` (`fgself-112`, `fgself-113`) began far below their start found two bugs: (1) `extend_vocab` listed a toggle's two orders (the Tinker's robogoblin and unrobogoblin, and one more pair) in the order of its process's hash seed, so a fork or a restart could give one order for the other, and (2) for the same reason a fork found its parent's snapshots of another vocabulary unusable: both started with a league of one old snapshot (`fgself-78/0`) instead of `fgself-111`'s own. Now the extension is sorted, a run takes the vocabulary of the weights it trains, and a network of a reordered vocabulary is put in the run's order when it loads. `fgself-112`'s verdict (OPSD, no gain) carries the same confound.

**`fgself-114`** (from `fgself-111`'s last, 17:30; its league of 5 past snapshots, `--neutral 2`): `fgself-113` again, clean: scenarios in a quarter of the training games.

**Result of `fgself-114` (18:55, 1 hour 15 minutes):** level. 50% of 304 head to head with its start (thirds 50%, 43%, 56%), 62% of 290 with `fgself-102`; against insane 92% of 66, its start 92% of 66; rating 1907 ± 16 (first hour) and 1904 ± 30 to its start's 1910 ± 11. With a target a little worse (52% of 91 against insane; rating 1547 ± 40 to `fgself-111`'s 1591 ± 26). The scenarios themselves were learned slowly (clash 46% → 51%, raid 44% → 55% between the halves; rich ~70%).

**`fgself-115`** (19:00): **weight averaging** (the user's idea's second half; a model soup): the mean of `fgself-111`'s last and `fgself-114`'s last (the start and the scenario-trained policy, distance 0.79 apart in the order head), trained on with `fgself-111`'s recipe (no scenarios; Adam's state from `fgself-114`). Evaluated head to head with both parents: is the soup better than either?

**Result of `fgself-115` (20:25, 1 hour 25 minutes):** level, worse with a target. 47% of 329 head to head with `fgself-111`'s last, 51% of 279 with `fgself-114`'s; against insane 93% of 96 (the parents 91% and 93%); rating 1887 ± 17 (first hour) and 1974 ± 25 to `fgself-111`'s 1920 ± 9 and `fgself-114`'s 1882 ± 20. With a target 43% of 90 against insane (rating ~1470 to `fgself-111`'s 1591). Averaging two checkpoints that play alike gave nothing; a soup of specialists that differ (each trained on one scenario family, the user's original plan) remains untried.

**`fgself-116`** (from `fgself-111`'s last, 20:30, `--neutral 2`): **rewards by part, again** (GDPO; `fgself-110` was worse with equal weights, the material shaping back at full weight): `--reward-parts 1 --part-weights 1,0.5,1` (outcome, shaping at half as in `fgself-73`, strategy). Each part's advantages are normalized on their own, so the strategy reward is no longer drowned by the outcome's spread. The question: are targets followed better, at no cost in wins?

**Result of `fgself-116` (21:55, 1 hour 25 minutes):** worse. 38% of 299 head to head with its start (thirds 43%, 39%, 31%), 41% of 349 with `fgself-114`'s last; against insane 88% of 93 (its start 88% of 90); rating 1862 ± 16 and 1824 ± 26 to its start's 1918 ± 8; with a target 41% of 104 against insane (falling: 47%, 43%, 34%; rating 1451 ± 36 to 1591). It followed the targets a little better in training (due items made 78% → 80%, 2.0 items missed a game to `fgself-111`'s 2.6). Rewards by part are out for now: each time the strategy reward weighs more (`fgself-89`, `fgself-110`, `fgself-116`) the targets are followed a little more and the games are lost more.

**`fgself-117`** (from `fgself-111`'s last, 22:00, `--neutral 2`): **the enemies it saw before, faded in** (`--inputs 3 --memory-fade 600`): `fgself-107` added them at once and fell below its start; here the learner's attention bias on them goes from -8 to 0 over its first 600 updates (~35 minutes), the anchor and older networks keep them hidden. Does knowing where the enemy was make it stronger?

**Result of `fgself-117` (22:55, 55 minutes):** worse, as `fgself-107`. 35% of 130 head to head with its start (thirds 49%, 35%, 23%), 43% of 267 with `fgself-114`'s last; against insane 83% of 69 (its start 95% of 56); with a target 34% of 65. Fading them in did not help: twice now the remembered enemies made it weaker. Before a third try, a look at what it does with them (attack-moves to where an enemy was? the neutral buildings among them?).

**`fgself-118`** (stopped after 8 minutes: a fresh Adam, the privileged path shifting the parameters; now matched by name) **= `fgself-119`** (from `fgself-111`'s last, 23:01, `--neutral 2`): **OPSD again, clean** (`fgself-112` started with a league of one old snapshot and two spell toggles swapped): `--opsd-share 0.2 --opsd-coef 0.05`, overnight, so the privileged input (which starts at zero) has time to learn before the distillation can carry anything.

**`fgself-119` stopped (23:44, 43 minutes):** 46% of 240 head to head with its start, 42% of 94 with `fgself-114`'s last; no verdict. Two rules of the game had been wrong for every run, found while answering the user's question about the first hero: (1) **the first hero is free** (1.29's hero token: only its food, at the altar or the tavern; tested), but the cost mask, the target builder and the teacher held it back until 425 gold and 100 lumber, which may be why the first hero came ~30 s after the targets' time; (2) **about half the purchases never went through**: a worker's right-click on the tavern could leave the tavern serving none of its player's units for the rest of the game (a fresh worker beside it was refused too). The buyer now walks to the shop and the sale goes through in reach, with no right-click (21 of 22 test hires). Every run from `fgself-111` on trained under both.

**`fgself-120`** (from `fgself-111`'s last, 23:45, `--neutral 2`, its recipe): **the corrected rules**, overnight: the free first hero and reliable purchases, nothing else changed. A new baseline: does the policy get its first hero sooner, hire tavern heroes as targets ask, and get stronger?

**Result of `fgself-120` (01:25, 1 hour 40 minutes):** level, the new baseline under the corrected rules. 48% of 313 head to head with its start, 55% of 387 with `fgself-114`'s last; against insane 87% of 87 (its start 89% of 79 in these rules); rating 1929 ± 16 and 1922 ± 22 to its start's 1916 ± 7; with a target 42% of 90 against insane. The first hero comes at a median 60 s (79 s before); tavern heroes hired in 7% of the training games, and 32% of the targets that ask for one (153 of 476).

**`fgself-121`** (from `fgself-120`'s last, 8527594, 01:30, overnight): **OPSD, clean and under the corrected rules** (`--opsd-share 0.2 --opsd-coef 0.05`; Adam's state matched by name): the first attempt that has its parent's league, one vocabulary and the game's rules right. Head to head with its start and `fgself-111`'s last. Stopped at 01:40 for a shutdown after 10 minutes (682214 steps); to go on from there. Resumed 09:50; **result (10:45, ~1 hour of training):** no gain: 45% of 182 head to head with its start, rating 1932 ± 17 to its start's 1976 ± 24; against insane 94% of 125 (its start 98%). Self-distillation with privileged information gave nothing in five attempts (the first four confounded); its privileged path, which starts at zero, barely learns in an hour.

**The audit (2026-10-22, the user's idea: a subagent looking for bugs like the tavern's).** Seven confirmed bugs where the code's assumptions about the game were silently wrong; all fixed and tested, the game mechanics in game: (1) a dead hero was never revived (the mask counted it as owned; and the revival is the altar's *revive* order on the dead hero, accepted ~7 s after the death, not its training order); (2) undead could not expand: the haunted gold mine's order went down the buy path to the goblin merchant (no undead run had built one), now a target order on a gold mine; (3) on `echofast`, 44 of Echo Isles' 75 creeps kept their full hit points while every player unit had half (creeping looked twice as costly as in the real game), and the tavern's and camps' units kept full prices; (4) the target builder's and the teacher's labels at idle buildings were dropped (never carried out, they "started nothing"), and their payment check didn't know the first hero is free; (5) a unit sent to buy was held 60 s whatever happened (now freed once it reaches the shop, or at once on a refusal), and a ware could go to a building that doesn't sell it; (6) a building's "being made" input read 0 unless some worker was on lumber; (7) a worker mining gold could not be sent to another mine (expansions could not be staffed). Also: the strategy modes' "expand" counted a second hall at home. And a new measure: the episode rows count real expansions (a hall far from home at a gold mine): **0% of the games** of `fgself-120` and `fgself-121`, in training and in the real game; their extra halls (6-11% of games) were all at home.

**`fgself-122`** (from `fgself-120`'s last, 10:50, its recipe): **the game as the audit corrected it** (revival, haunted mines, mine transfers, buyers, labels, creeps), nothing else changed: the new baseline. Head to head with its start and `fgself-111`'s last.

**Result of `fgself-122` (11:55, 1 hour 5 minutes):** level to a little lower, the base of what follows: 48% of 301 head to head with its start, 43% of 227 with `fgself-111`'s last; against insane 89% of 66 (its start 97% of 65); rating 1917 ± 16 to its start's 1944 ± 15; with a target 52% of 71. Its play: no expansion in any game (training or the real game), a second hall at home in 16% of the training games, ~1.2 creeps killed a game, a tavern hero in 17% of the games, no haunted mine.

**`fgself-123`** (from `fgself-122`'s last, 5838960, 11:57, its recipe; head to head with its start and `fgself-120`'s last): **expansions taught.** People expanded in 81% of 95 Echo Isles games (at the natural, ~800 north of its mine, by ~6 minutes); no policy ever had. The teacher, when its target's expansion is due (a hall or a haunted mine among its items: 19-52% of people's targets, ~90% of the AI's), has the army clear the natural's camp (6+ units), then the nearest worker build the hall at people's spot (undead: an acolyte haunts the mine), then three workers mine there; a target's hall counts as made only as a real expansion (a second hall at home had counted). Night elves not yet (no entangle order). The question: does it expand, and at what cost in wins?

**Result of `fgself-123` (12:27, 30 minutes): much worse, stopped.** No expansion in 1,259 training games; the camp lesson (the whole army to the natural's camp, ~26 lessons a game) dragged armies away: games with an expansion due won 13% (32% in `fgself-122`), all training games 45% (59%), 36% of 55 head to head with its start. The hall lessons (1,108) never applied: the camps stood.

**`fgself-124`** (from `fgself-122`'s last, 12:31, its recipe): **expansion teaching, take two**: no camp lesson; in a third of the training games both naturals' camps are gone from the start (`--scenarios natural --scenario-share 0.33`, harness op 88), so the teacher's hall and worker lessons can apply when the target's expansion is due. Evaluation games are untouched (the camps stand there).

**Result of `fgself-124` (13:00, 30 minutes): no expansions, stopped.** 3 expansions in 1,698 training games (the naturals' camps gone in a third of them), and a second hall at home in 28% (16% in `fgself-122`): the 6,890 hall lessons (a worker builds the hall at a point ~6,000 from home) taught the policy "build a hall", not where. 47% of 105 head to head with its start. The spot itself is fine (built by hand in game for human and orc).

**`fgself-125`** (from `fgself-122`'s last, 13:08; its league): **an expand order of the agent's own** (`--macros 1`, `macros.py`): a worker's order "take the natural", the agent builds the race's hall where people build it (undead: haunts the mine); it may be given while the natural is free and the hall affordable. The teacher's expansion lesson is that order, the target builder keeps a due expansion's cost from the other orders, the naturals' camps are gone in a third of the training games as in `fgself-124`. Every network grows to the new order (its row rarer than any order now).

**Result of `fgself-125` (14:55, 1 hour 50 minutes):** level to a little better, the new base: 52% of 307 head to head with its start (thirds 45%, 52%, 58%), rating 1913 ± 21 (its second hour) to its start's 1871 ± 15; against insane 81% of 68 (its start 91% of 65); with a target 43%. **It expands:** a quarter of the training games with the naturals' camps gone (1,828), from 16% to 28% within the first hour; those games won 15% (all of them 48%, the others 57%): on `echofast`'s short games an expansion rarely pays off yet. With the camps standing (the real game) the expand order stays masked: no expansions there.

**`fgself-126`** (from `fgself-125`'s last, 8001948, 15:03): **a creep order** (`macros.CREEP`): an army unit or a hero attack-moves to the natural's camp while it stands and nobody holds the natural; the teacher shows it to the heroes once the army is 6 strong and the target's expansion is due (not the whole army every 10 s: `fgself-123`). The naturals' camps are gone in a fifth of the training games (a third before), so it meets the camps more often.

**Result of `fgself-126` (15:57, 55 minutes): stopped, its creep and expand orders misused.** It creeps (3.1 creeps killed a game with the camps standing, ~1 before) and expands with them (12.9% of those games), but the games that expanded won 1%; with a target 14% of 28 against insane; 46% of 154 head to head with its start (recovering: thirds 33%, 39%, 65%). **A second audit** (the user's request) found why, among others: (1) every worker sampled the expand order on its own: 8-18 walked to the natural in waves, the gold spent meanwhile, no hall built (63 hall orders in one filmed game); (2) past 96 own units the newest were cut from the view (the rows' first-seen order): 12-17% of the games, won 2-5%; (3) the creep order was allowed to any unit from minute 1 (436 attack-moves in a game), not under the teacher's conditions; (4) a worker's build order held only 30 s, the walk to the natural; (5) undead's haunted-mine expansions didn't count in the strategy modes. All fixed: one expand a step and none while one is under way (its hall's cost kept), heroes/army/producers/workers kept past the cap before farms and towers, creep with 6 army units from second 150, build orders held 200 s. (A dead tavern hero still can't be revived: rare.)

**`fgself-127`** (from `fgself-125`'s last, 8001948, 15:59; its league): the expand and creep orders with the audit's fixes, the naturals' camps gone in a fifth of the training games.

**`fgself-127` at 2 hours (17:55): better than its start, and it expands in the real game.** Rating 1850 ± 20 (its second hour) to its start's 1791 ± 14 (its first hour 1742 ± 19: the dip as it learned the new orders); 49% of 369 head to head with its start, rising (thirds 39%, 50%, 57%); against insane 82% of 77 (its start 83% of 81); with a target 22% of 73 (the weak point: many targets ask for an expansion). In the real game it expanded in 14 of 150 evaluation games (won 3) and kills ~3 creeps a game (~1 before the creep order); in training ~12% of the games expand. Kept running.

**Result of `fgself-127` (19:25, 3 hours 25 minutes): better, the new base** (its best hour's last checkpoint, 13485938 at 18:59): rating 1874 ± 19 in its third hour to its start's 1810 ± 12 (then 1817 ± 27: it peaked); 50% of 438 head to head with its start; against insane 81% of 95 (level with its start). With a target it stays weak: 24% of 126 in the real game, worst with people's build orders (21 of 106; the built-in AI's 9 of 20), and not because of expansions (28% of the targets with one, 20% without). In training, games with a target win 49% (51% against the AI), without one 77% (82%).

**`fgself-128`** (from `fgself-127`'s 13485938, 19:30; Adam's state the nearest saved one's): **targets not enforced** (`--z-enforce 0`). The mask that lets a side with a target make only the target's items (narrow targets: no hero or army unit outside them until the horizon) may keep it from answering what the opponent does. Following then rests on the target's input, the builder's labels and the strategy reward. Does target play win more, and is it still followed?

**Result of `fgself-128` at 2 hours (21:25): better, the new base; target play more than doubled.** With a target 58% of 95 against insane in the real game (`fgself-127`: 24%), the target's due items still made (70% there, 71% in training to 75%); its with-target rating from 1522 to 1670 in its second hour. Without one 93% of 92 (its start 85% of 94). Rating 1893 ± 20 (second hour) to its start's 1864 ± 21; 49% of 250 head to head with its start, 58% of 259 with `fgself-125`'s last. In training, games with a target won 67% (49% in `fgself-127`). By race with a target: night elf 28 of 28, undead 11 of 18, orc 12 of 23, human 4 of 26 (the weak one). In the real game it expanded in 11 of 187 games and won none of them. Kept running.

**A silent refusal behind the human weakness** (21:40): with a target, human made 40-67 peasants and only footmen and missed its keep. A hall's tier order is refused while the hall trains anything (tested in game: the keep order behind queued peasants refused; with the queue cancelled first, accepted and built), and the policy trains workers back to back, so its tier orders were refused for all races (2% of the games reached tier 3). The target builder cancelled the queue already; now the agent does too.

**`fgself-129`** (from `fgself-128`'s last, 8716868, 21:31, its recipe; `fgself-128` stopped there, 2 hours 20 minutes): **tier orders that go through** (the hall's worker queue cancelled first), nothing else changed. Do targets that tech get followed and won, and does the policy tech more on its own?

**`fgself-129` at 1 hour 15 minutes (22:55): the best yet.** 64% of 253 head to head with its start (thirds 62%, 58%, 71%), 56% of 230 with `fgself-127`'s best; rating 2029 ± 32 in its second hour (1941 ± 19 its first) to its start's 1873 ± 22; against insane 96% of 52, with a target 70% of 56 (`fgself-128` 58%, `fgself-127` 24%). With a target by race: night elf 12 of 12, undead 18 of 22, orc 10 of 16, human 5 of 14. In the real game 63% of the games reach tier 2 and 13% tier 3; 7 expansions in 126 games, ~2 creeps a game. Its army is still tier 1 (footmen 31%, ghouls 30%, archers 15%, grunts 12%): the next variety to work on. Kept running.

**Result of `fgself-129` (00:15, 2 hours 35 minutes): better, the new base.** 59% of 471 head to head with its start; rating 1930-1952 ± 19-31 by hour to its start's 1875 ± 16; against insane 94% of 72, with a target 70% of 71. Why its army stays tier 1: the policy seldom asks for anything else (in 12 filmed games, 401 ghoul and 300 footman train orders to 3 for riflemen and 2 for sorceresses): a choice, not a refusal. A target asks only for each item once.

**`fgself-130`** (from `fgself-129`'s last, 11011701, 23:57): **a target's army mix** (`--z-mix 1 --z-mix-reward 0.1`): each item's share of the target's army units as an input (the strategy block grows by 64, the new columns at zero), and the strategy potential minus 0.1 times how far the army trained is from that mix (total variation, from 5 army units). Do the armies follow their targets' mixes, and what does it cost? Stopped at 00:40 for a shutdown after 25 minutes (4193900 steps; the army-mix gap 0.62, not yet moving); to go on from there. Resumed 11:38. **Result (12:05, ~55 minutes of training): the mix not followed, a little better otherwise.** The army-mix gap stayed at 0.62-0.63 over 3,800 games with a target; with a target 74% of 81 against insane (`fgself-129` 70%), 76% of the training games with one won; 55-57% of 163-178 head to head with its start; without a target 91%.

**`fgself-131`** (from `fgself-130`'s last, 5534555, 11:58, its recipe): **the army mix taught**: the teacher has a building train the unit the target's army has the largest share of more than its own (from 120 s and 5 army units, by 0.15 at least). At 50 minutes the gap had not moved (0.62-0.64); counted, only 1% of the army-mix lessons taught anything (10% of all lessons): the rest named a unit the policy could not order then (money, tech) and the loss drops those. Restarted 13:09 with lessons only for units it can train then (75% of them kept since).

**Result of `fgself-131` (14:30, 2 hours 20 minutes): level, the mix not followed.** 49% of 292 head to head with its start since the fix, 55% of 361 with `fgself-129`'s; rating 1904 ± 11 (its first hour) to its start's 1941 ± 15; with a target 72% of 64. The army-mix gap stayed at 0.62-0.64 even with the lessons kept: ~4 lessons a game against the hundreds of train orders the policy gives itself.

**`fgself-132`** (from `fgself-130`'s last, 5534555, 14:28; `fgself-131`'s league): **a mix order of the agent's own** (`macros.MIX`): a building's order "train the target's mix": the unit the target's army has the largest share of more than the side's own, of those it makes and can pay for now; available while one lacks (with a target that names an army). The teacher's army-mix lesson is that order. As with the expand order, the policy only decides when.

**Result of `fgself-132` (16:25, 1 hour 50 minutes): worse overall, the mix followed more.** The army-mix gap fell for the first time (0.63 to ~0.58); with a target in the real game 25 of 52, then 36 of 53 (its with-target rating 1500, then 1635); but 44% of 408 head to head with its start and its rating 1876 ± 18 (second hour) to its start's 1929 ± 11. As with expansions on `echofast`, a more varied army costs wins in its short, fast games.

**`fgself-133`** (from `fgself-130`'s last, 5534555, 16:26; `fgself-132`'s recipe with the mix order): **halfway rules** (`--map echo50`: three quarters of the hit points and costs, two thirds of the times; the targets people's real-game times scaled by 0.72, `--z-pool strategies-echo-real-narrow-tavern.json --z-time-scale 0.72`). On `echofast` tier-1 rushes win, and expanding and varied armies cost games; the real game (the evaluation) has longer games. Real-rules training pulled the policy down months of fixes ago (`fgself-98`/`99`); does the halfway now pay?

**Result of `fgself-133` (17:54, 1 hour 30 minutes): worse.** 45% of 312 head to head with its start (50% of 341 with `fgself-129`'s 11011701), its rating 1902-1917 to its start's 1942; against insane 96% (its start 97%), with a target 49% of 104 (from 38% to ~50%). In the real game no more varied without a target (tier 2 in 50% of the games, its start 49%; expansions 1%, 0%); with a target tier 2 in 88%, tier 3 in 45%, an expansion in 14%, 3.2 camps a game. The halfway rules don't pay: `echofast` stays the training map.

**`fgself-134`** (from `fgself-132`'s last, 9752412, 17:56; `fgself-132`'s recipe and league): **items off the ground, night elf expansions, a tavern hero's revival** (`--macros 2`). The harness now reports the items lying on the ground that each player sees (Echo Isles' camps drop one each: a permanent item, a potion or a tome) and has a hero's pick-up (a right-click on the nearest within a radius); the agent's pick-up order is available while a hero has one within 1,000 (and a free slot, but for a tome), the hero kept on it until it holds one, and the teacher shows it when no enemy is near. Night elves expand too: the expand order builds a tree of life at the natural and the agent has it entangle the mine (the policy has no entangle order; night elves had never expanded). A dead tavern hero, lost for good before, is revived at the player's altar with its hire's order. All three checked in the game. Head to head with its start and `fgself-130`'s 5534555 (the best rating).

**Result of `fgself-134` (19:55, 2 hours): worse head to head, more varied.** 41% of 468 head to head with its start and 38% of 446 with `fgself-130`'s 5534555; its second-hour rating 1945 ± to its start's 1967; against insane 99%, with a target ~55-60% (its with-target rating 1547, then 1622, above `fgself-132`'s 1500/1613). It picks up items (0.28 a game without a target, 1.8 with one; a game where it gave the order was won 56-68% head to head against 22-36% of the same length without). A diagnostic (its frozen start under the new code, head to head with the same two checkpoints: 49% and 54% of 123 games) showed the code fair: the drop is learned. By race the loss is human's: 17% of 116 head to head with its start (night elf 46%, orc 44%, undead 53%). Without a target the human policy builds an altar in only 17-24% of its games, so it has no hero (with a target 99% do): it rushes the AI down (74-93%) and loses to policies.

**`fgself-135`** (from `fgself-134`'s last, 9372762, 20:00; its recipe and league): **a first hero in games without a target**: the teacher's lesson (its altar trains one of the race's usual first heroes, the game's draw; with no altar the worker nearest its hall builds one 450-800 from it, as the target builder places buildings), each when the policy could order it; the tavern's hero then only the second in those games. Head to head with `fgself-134`'s 9208380 (its last but one) and `fgself-130`'s 5534555.

**`fgself-135` at 1 hour 25 minutes (21:24): level, kept on.** 45% of 327 head to head with its start (human 34%, from `fgself-134`'s 17%), 34% of 341 with `fgself-130`'s 5534555; its second-hour rating 1920 to its start's 1898 (`fgself-130`'s 5534555: 1974); against insane 90%, with a target 62%; targets followed 72% (orc 82%, night elf 84%, human 74%, undead 62%). Without a target human now builds an altar and a hero in ~96% of its games. The line since `fgself-132` (the mix order, items, a first hero) is ~55-75 points below `fgself-130`'s best: the variety costs wins at first. It trains on for hours (the user: a gain can take a while to show).

**Result of `fgself-135` (22:00, 2 hours): level to a little better.** 47% of 455 head to head with its start, 37% of 484 with `fgself-130`'s 5534555 (by thirds 32%, 38%, 42%: rising); its second-hour rating 1929 to its start's 1898 and its own first hour's 1863. Human without a target now plays a hero; the line stays below `fgself-130`'s best (1974).

**`fgself-136`** (from `fgself-135`'s last, 9440449, 22:01; its recipe and league): **games between policies taxed by who is expected to win** (`--win-balance 0.5`, the user's idea), and those games 20% of the launches (`--ai-share 0.8`, was 0.95). Against the built-in AI a tax cannot hold 50%: the levels sit at 0 (the learner wins ~80% of its training games against the untaxed insane AI), and `fgself-85`'s richer AI still lost 85%. Between policies a target costs a side most of its games (35% won in `fgself-134`'s self-play), and the races' balance (a level per pair of races) does not see it. `balance.py` fits a logistic model of a game's outcome from both sides' races, whether each plays a target and what it holds (an expansion, tier 3, a tavern hero), a past snapshot, and the tax taken (the newest 3,000 such games, refitted every 25); a game's tax, decided once the targets are drawn, is the share of the favoured side's income that evens it out (at most 0.5, with a little noise so the tax's effect stays measurable). The races' balance until 150 games are in. Head to head with its start and `fgself-130`'s 5534555.

`fgself-136` stopped after 10 minutes (513 games) for audit 3's fixes (below), its model fitted on 175 games between agents (log-odds for a side: a target -0.55, a target's expansion -0.36, its tier 3 +0.29, orc +0.56 against the other races' -0.1 to -0.3; the tax's -4.1, so a share of ~0.25 per log-odds unit of edge).

**Audit 3** (2026-10-23, a subagent; the user's request): a resume without `--win-balance` re-inherited the parent's league and overwrote checkpoint 0 (an `else` taken by the new block); heroes left skill points unspent (51 of 128 agent heroes ended their games with points left; the policy spent a point on 2.8% of the steps it had one) and almost never learned their ultimate (4 of 42 heroes at level 6, the AI 68 of 68), nothing in our code blocking it; the view never forgot a dead building's production (a research masked for the rest of the game in 8 of 400 demonstrations), recycled ids took the enemy's production, and enemy production marked its buildings busy (seen through the fog); undead expansions never made the target's `unpl` (40 of 123 such games read missed); a keep or a tree of ages at the natural read as a free natural; night elves had no expansion reserve; the mix order counted units no train order makes; scripted camp clears dropped their items for free; filmed games' traces leaked into the episode files (176 MB). All fixed.

**`fgself-137`** (from `fgself-136`'s 657780, 22:10; its recipe and league): **audit 3's fixes, and a hero's skill points spent** (`--macros 3`: a point left unspent 10 s is spent as the built-in AI's order for the hero would, its ultimate at 6; `herodata.next_skill`). Head to head with `fgself-135`'s last (9440449, `fgself-136`'s start) and `fgself-130`'s 5534555.

**Result of `fgself-137` (23:06, 56 minutes): better.** 51% of 198 head to head with its start (ties half: 100 won, 1 tie, 97 lost) and 44% of 291 with `fgself-130`'s 5534555, by thirds 36%, 45%, 51%: the first of the line since `fgself-132` to come level with it; against insane 94%, with a target 60%. The win balance held the games between policies at ~42% won (9% ties), a mean tax of 0.13 of the favoured side's income; heroes' skill points spent for the policy 1.3 a game.

**Audit 4** (a subagent; the user's request): the win-balance model froze once its 3,000-game window was full; the "% of the AI's record" yardstick had taken duelfast's AI games for Echo Isles runs since `fgself-100` (`fgself-135`: 194% that way, 170% against Echo Isles'); PFSP read past snapshots from taxed games (42-100% won taxed, 8-33% untaxed); `league/past` let an old, rarely drawn snapshot dominate (0.73 against 0.41); evaluation games counted in a league member; ties counted as losses in `head_to_head.py`; the food mask ignored the game's 100 ceiling (2 refused orders a step there); a hall's tier order paired the step's later orders with others' resources; a gone unit's harvest memory went to the next unit with its id; a third of the target mixes held units no building trains; an enemy's illusions and its workers' building orders were inputs (the game shows neither); 77 (type, order) pairs no unit can give were allowed; refused orders were never counted; the ratings' anchor pooled three maps and a resume put every game in hour 0. All fixed (deferred: pick-up and mix available per player, not per unit; the expand and creep masks see the fog).

**`fgself-138`** (from `fgself-137`'s 4769091, 23:07; its recipe and league): **audit 4's fixes**. Head to head with `fgself-137`'s 4769091 and `fgself-130`'s 5534555.

`fgself-138` stopped after 28 minutes (44% of 98 head to head with `fgself-137`'s 4769091, 34% of 143 with `fgself-130`'s 5534555, rising): it had started on the races' balance its parent drifted (below), and two bugs the new refusal counts showed. 30-40% of the policy's orders were refused by the game: as human and orc ~1,000-1,500 repair orders a game on what needed none (11-35% of the policy's orders in every run since `fgself-100`: its no-op), as undead and night elf 200-430 harvest orders at a gold mine their workers can't mine (undead mine their own haunted mines, night elves their entangled ones). Harvest orders now go to a mine the race can mine, and repair orders only to a damaged or unfinished building or machine of the player's.

**Audit 5** (a subagent; the user's request): the income tax had been an absolute set of the taxed side's gold after the harness's pending sales (its walk-and-buy purchases came free) and a share of each step's 10-20 rounded down (0.054 took 0.005); under the win balance the races' balance went on moving on the model's games (to 0.88 of undead's income: 49 of 49 such games lost while the fork's model was unfitted) and a fork started on it; PFSP rested on a few untaxed games, read the cost of the learner's target, and weighed a snapshot beaten twice 0 for good; the headline of `real_games.py` pooled the training launches on the training map with every evaluated policy; `--macros 3` was not noted in the vocabulary; enemy heroes' unspent skill points were inputs; a revival was allowed 7 s before the game takes it; summons counted in the army mix; a pending purchase outlived the unit's next order. All fixed; checked fine: the view's per-step forgetting, the 77 pruned pairs, the C++ inference loop against the model (6e-6), the two observation parsers, the automatic skill order.

**`fgself-139`** (from `fgself-138`'s 2277752, 23:38; its recipe and league, its model seeded from `fgself-138`'s 644 games between agents): **audit 5's fixes, harvest at the right mines, no refused repairs**. Head to head with `fgself-137`'s 4769091 and `fgself-130`'s 5534555.

Stopped at 23:50 for the night after 12 minutes (46% of 34 head to head with `fgself-137`'s 4769091: too early). In the game since (not yet in a run): a worker repairing already, an orc frame under construction (it takes no helpers) and a building at full health take no repair order, and a wisp mining shows "smart", so a harvest re-sent to it (refused: 318 a game as night elf) is no longer sent.

**`fgself-139`, resumed 2026-10-24 (after the night, two reboots for the GPU unlock, and restarts for speed): better.** 56% of 293 head to head with `fgself-137`'s 4769091 (by thirds 50%, 56%, 61%) and 40% of 461 with `fgself-130`'s 5534555 (37%, 41%, 43%); its rating 1923-1938 by hour (h10-h12) to `fgself-137`'s 1881 and `fgself-130`'s 1985; against insane 96%, with a target 63%; targets followed 73%. It trains on.

**Throughput, 2026-10-24.** The RTX 3090's BF16 rate unlocked and a 400 W power limit ([GPU unlock](gpu-unlock.md)) made matrix multiplies 1.4-1.8 times faster, but the learner's update stayed at ~2.75 s: self-play is set by the CPU (the games, then our Python), not the GPU. A profile of the actors (py-spy) found the tech mask reading the game's archives at every game's start (11% of an actor's time), the shared inference buffer's memmap slices (~5%), the teacher's look at every step (~5%), the natural looked at every step (~4%) and the privileged block's masks per side; the run's video renderer ran at nice 0 (~2 of the 32 threads). With those fixed the CPU was 20% idle, and 7 games an actor (was 6) took the slack: ~1,494 agent steps per second before, ~1,660 after (+11%; 8 games an actor: 1,580, the CPU 9% idle). GNOME's file indexer had crawled the games' Wine prefixes (made and removed all the time): `~/wc3` is now ignored.

**`fgself-139`, the afternoon of 2026-10-24: judged.** 58% of 490 head to head with `fgself-137`'s 4769091 (by thirds 51%, 60%, 64%): better; 42.5% of 637 with `fgself-130`'s 5534555 (39%, 40%, 48%): not yet as strong, closing. Against insane 97% of 357, with a target 68% of 320. Restarts on the way: the actors' entity features in C (`native/w3feat.c`, View.step 1,002 → 556 µs a side-step, the same features), Wine off the system D-Bus (each game prefix's device manager asked udisks for every disk ~0.4 times a second: the daemon had taken ~1.5 cores), a frame's helper not sent without lumber (the game refused a human peasant's power build with none, tested in the game: ~275 refused orders a game as human), and 8 games an actor (the CPU 11% idle at 7: ~1,665 → ~1,730 agent steps per second).

**Targets: the first hero is not the target's.** The policy makes its first hero on time (~10-20 s before it is due) but the target's one only 25-37% of the time, chance among 3-4 (human, Archmage targets: Paladin 355, Mountain King 321, Archmage 320). The teacher taught a first hero only in games without a target; in target games the builder's lesson came once the hero was due, after the policy had made another one, and a second hero needs a tier. Also missed most: the expansion (~44% for every race), undead's Halls of the Dead (half), tavern heroes (~half). `--teach 2` teaches the target's first hero from the altar as soon as the altar stands: `fgself-140`.

**`fgself-140`** (from `fgself-139`'s 16955575, 13:48): **`--teach 2`, the target's first hero taught at the altar**; also round 2 of the game's code made faster (`--game-cuts 521721`), from 14:19 the update compiled before its CUDA graphs are captured (`--update-fuse default`: an update 2.8 → 1.5 s in the run) and 9 games an actor. **Level in strength, better at following targets.** 53% of 314 head to head with its start (by thirds 57%, 56%, 46%), 45% of 355 with `fgself-130`'s 5534555; rated 2015 ± 42 and 1992 ± 35 (h0, h1) to its start's 1978 ± 37; against insane 97%, with a target 64% (`fgself-139`: 68%). The first hero is the target's in 66% of undead games, 61% of human, 50% of night elf (from 25-37%, chance), orc 33%: its lesson rarely fires (the policy orders a hero the step its altar is done). Kept.

**`fgself-141`** (from `fgself-140`'s 12745403, 16:02): **the KL anchor moved to the fork's start**, `fgself-140`'s last checkpoint (rated ~2000), from `fgself-100`'s 13649982 (1966, from before the six audits, the macros, the tavern and the free first hero): the forward KL had held the policy to the old policy's habits. Same weight (0.05), direction and heads. Also round 3 of the game's code (`--game-cuts 67106297`: pathing in C where exact, the units pass's flags in the shim, the step sync over a Unix socket: -6.6% of a game's main thread), the noise scale every 50 updates and graphed, 4 games a launch (was 2: the relaunches' churn was ~2.2 cores). Head to head with its start, `fgself-130`'s 5534555 and `fgself-108`'s 2302196 (rated 2086 under the rules of 2026-10-20: the user's idea of a league of strong checkpoints, measured first).

**`fgself-141`, judged at 17:56: worse.** 46% of 265 head to head with its start (by thirds 59%, 43%, 37%: falling), 54% of 165 with `fgself-130`'s 5534555; rated 2046 ± 43 and 1982 ± 44 (h0, h1) to its start's 2030 ± 41; with a target 57% (`fgself-140`: 64%). Holding the policy to its own start (the KL anchor there) hurt as it trained. `fgself-108`'s 2302196 (rated 2086 under the rules of 2026-10-20, the user's league idea) won 47% of 51 against it: not stronger than today's policies. Its throughput work stays: the actors' Python merged at 17:22 (~2,010 → ~2,170 steps/s).

**`fgself-142`** (from `fgself-140`'s 12745403, 18:00; `fgself-141`'s settings and league): **half the KL pull** (`--ref-kl 0.025`, the anchor back at `fgself-100`'s 13649982): 0.2 → 0.05 had gained in `fgself-77`. Head to head with its start and `fgself-130`'s 5534555.

**`fgself-142`, judged at 19:26: better with targets, but countered by `fgself-130`.** 59.5% of 242 head to head with its start (by thirds 49%, 65%, 65%), but 34.5% of 288 with `fgself-130`'s 5534555 (28%, 43%, 33%); rated 1972 ± 32 and 1981 ± 60 (h0, h1) to its start's 1970 ± 28 and `fgself-130`'s 2019 ± 10; against insane 95%, **with a target 70% of 240** (`fgself-140` 64%, `fgself-141` 57%). The first hero the target's: undead 74%, human 67%, night elf 56%, orc 36%. Half the pull kept. Its games ran longer (10.4 against 8.4 game minutes) and larger (87 against 80 units at most): ~10% more CPU a step in the games.

**`fgself-143`** (from `fgself-142`'s 9858508, 19:31): **`fgself-130`'s 5534555 in the league for good** (`--league-add`: a "fixed:" member, drawn by PFSP as the past snapshots are, never replaced by newer ones): the user's idea of a league of strong checkpoints, with the one that beats us. Otherwise `fgself-142` (`--ref-kl 0.025`), and the actors' Python round 3. Head to head with its start and `fgself-130`.

**Skill ratings** (the user's idea, the dashboard's Ratings view): a Bradley-Terry fit on the Elo scale over every evaluation game in the real game, the insane AI at 1500. At 22:45: `fgself-106`'s last checkpoint 2128 ± 24, `fgself-102`'s 11841833 1977 ± 8, `fgself-100`'s best 1956 ± 9.

**Queued (2026-10-19, the user's idea): GRPO.** The game is deterministic (lockstep multiplayer needs it), so a group of games can start from one state and differ only by the policy's sampled actions; each game's advantage is its return against the group's mean (and spread), with no value function (DeepSeek's GRPO). Plan: (1) check determinism: the harness has no seed control yet; the map script would call `SetRandomSeed` at start, and two games with one seed and the same orders (none, or a replayed order log) must give identical unit rows step for step, the built-in AI included. (2) The main design (agreed with the user): branch mid-game. A game from one start gives little credit for any decision (a whole game's result over hundreds of them), so: play a game, pick a state at step t (e.g. 2-8 game minutes, near a decision that matters: a tier, an attack), replay the same seed and orders to t in G (~8) games side by side, then let each continue with its own sampled actions; each continuation's advantage is its return against the group's mean and spread, the policy loss on the steps after t, the KL to the anchor kept, beside the usual GAE games. Replaying to t is cheap on `echofast` (a game runs ~13x real time). If replays drift (the built-in AI's threads), the fallback is the game's own save and load (`SaveGame` / `LoadGame` in single player; whether the AI scripts survive a load must be checked). (3) Only if branching fails: groups from one start (races, seed, target, opponent). Compare to the current recipe at equal steps. **The determinism check (2026-10-19, engine notes):** with a seed (`GameSetup.seed`, new), games between agents with the same orders are identical step for step; games with the built-in AI are not (they differ after 13-18 s). So GRPO branches come from self-play games (against past snapshots and itself); with a seed each player now also starts at its own start location (4 of 4 seeded games identical for 300 steps). Left to build: the branch scheduler (replay a game's orders to step t in G games, then sample) and the group-relative policy loss.

**How the best policy wins and loses** (`fgself-73`'s policy: its own run's 347 evaluation games and its last checkpoint's 141 in `fgself-74`: 488 games against the untaxed insane AI, 356 won). One strategy: an early rush of tier-1 units. The median win came at 9.7 game minutes, 206 of the 356 wins within 10 minutes, and only 38% of the wins reached tier 2. The losses were long (23.7 minutes on average), with tier-1 units and many workers (56 a lost game against the AI's 14, army units 89 against 38). Heroes 1.0 a game against the AI's 1.5. Undead's lumber (83 a game minute against the AI's 213, `fgself-69`/`-70`) is still its weak point.

**The plan** (2026-10-17, agreed with the user), in order. 1. Finish `fgself-75`. 2. An exploiter league against the early rush (AlphaStar's main exploiter is in `selfplay.py`, `--exploiter-share`, not yet used in whole games). 3. Strategy conditioning as AlphaStar's z: a target build order and unit mix sampled for each game, from the built-in AI's Echo Isles games and later from human replays, with a small reward for following it, to break the single rush and force tech. Human replays: ~3,000 replays of 1.29 from warcraft3.info, with a converter for their build orders (any version) and their playback in the real game as demonstrations (1.29 only, see [human replays](human-replays.md)).

Next, ranked with what AlphaStar and OpenAI Five found ([ideas](ideas-from-alphastar-and-openai-five.md)): forward KL to the anchor on every head (ours is reverse KL on the orders only), a temperature for the targets, features of each unit's own last order and hit-point history (branch `features-v2` has the hit-point change and sine/cosine positions), a value with both players' information.

**A fairer yardstick: the AI's own record in the same matchups.** `duelfast` is far from balanced even for the built-in AI against itself (3,000 games, mixed matchups): night elf wins 81% (normal) / 75% (insane), orc 65% / 71%, undead 33% / 37%, human 22% / 19% (human vs night elf 8% / 6%, vs orc 15% / 6%). So a race's real-game win rate says as much about the map as about the policy. `scripts/real_games.py` now also counts, for real games against normal and insane, how many the AI itself would win in the same matchups: from 2026-10-03 18:36 to 2026-10-11 00:17 the learner won 195 of 896, the AI would win ~455 (**43% of the AI's record**; 45% in the last two hours before the anchor experiment). 100% is the built-in AI's level.

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
