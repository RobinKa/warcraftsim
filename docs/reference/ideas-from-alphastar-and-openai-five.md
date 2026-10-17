# Ideas from AlphaStar and OpenAI Five, for fgself-12's plateau (2026-10-12)

**The papers:**
* AlphaStar: O. Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning", *Nature* 575, 350–354 (2019), [nature.com](https://www.nature.com/articles/s41586-019-1724-z) (doi 10.1038/s41586-019-1724-z), open copy: [DeepMind's PDF](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf), [supplementary data](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1724-z/MediaObjects/41586_2019_1724_MOESM2_ESM.zip).
* OpenAI Five: C. Berner et al., "Dota 2 with Large Scale Deep Reinforcement Learning", [arXiv 1912.06680](https://arxiv.org/abs/1912.06680) (2019).
* SC2LE: O. Vinyals et al., "StarCraft II: A New Challenge for Reinforcement Learning", [arXiv 1708.04782](https://arxiv.org/abs/1708.04782) (2017).

The figure, section and appendix numbers below are the papers' own (checked again on 2026-10-17 against the PDFs).

A subagent read the papers in full: AlphaStar (Nature 2019, its Methods, Fig. 3, the supplementary architecture and RL pseudocode), OpenAI Five (arXiv 1912.06680 with its appendices) and SC2LE (arXiv 1708.04782). It checked them against our code. "Reported" is what a paper says; the rest is inference.

## What the papers found most important (reported)

**AlphaStar**
* **Human data** (Fig. 3E, Elo): none 149 → supervised 936 → a supervised start 1020 → + KL to the supervised policy 1400 → + strategy statistic z 1540. Fine-tuning on winning games (MMR above 6,200, 16,000 games) raised the win rate against the Elite bot from 87% to 96% (Methods, "Supervised learning").
* **The strategy statistic z** (Methods, "Supervised learning" and "Exploration and diversity", and the supplementary's detailed architecture):
  * z is a build order (the first 20 buildings and units built) and cumulative statistics (the units, buildings, effects and upgrades of the game), from a human replay on the same map and races.
  * Supervised learning sets z to zero 10% of the time.
  * In RL the main agents get pseudo-rewards for following z: the edit distance between the sampled and the executed build order, and the Hamming distance between the statistics. Each pseudo-reward is on with probability 25%, with its own value function.
  * On Battle.net the agents played unconditioned (no z).
* **The league** (Fig. 3A/B, and Methods, "Populating the League"):
  * The main agents' Elo: alone 1540, with main exploiters 1693, with league exploiters too 1824. The relative population performance of the whole league (a measure of how hard it is to exploit): 6% → 35% → 62%.
  * Main agents: 35% self-play, 50% PFSP against all past players, 15% against forgotten main players and past main exploiters. They never restart.
  * Main exploiters play the current main agents. While they win less than 20%, half of their games are PFSP over the main agents' past snapshots instead. They join the league when they beat all three main agents in more than 70% of games, or after a time limit. Then they restart from the supervised policy.
  * League exploiters play the whole league by PFSP. They join it at more than 70% against every player, or after a time limit, and restart with probability 25%.
* **RL** (Fig. 3I/K, average win rate against fixed opponents): V-trace 49% → + TD(λ) 73% → + UPGO 82%.
  * The largest RL ablation: a value function that also sees the opponent, **82% against 22%** without.
  * The action arguments are updated separately.
* **Architecture** (Fig. 3F, supervised win rate against the Elite bot, cumulative): 0% → action delays 7% → a pointer network 36% → a transformer 71% → scatter connections (units onto a spatial map) 87%.
* **Action rate** (Fig. 3G): a cap at 100% of human APM scored 1540 Elo, no cap 1392, a 200% cap 1411. More actions hurt.
* **Settings** (the supplementary's detailed architecture and pseudocode):
  * The KL to the supervised policy is computed for every action argument, at weight 2e-3, as KL(teacher‖student) (forward).
  * Sampling uses temperature 0.8 for the action type, the queued flag, the units, the target unit and the location. The delay has none.
  * Adam with lr 3e-5 and β1 = 0.

**OpenAI Five**
* **Fresh data matters** (§4.4, Fig. 5b/c): staleness of about 8 versions slowed learning significantly. Reusing each sample 2–3 times made it about 2× slower. They aimed for staleness 0–1 and reuse ~1.
* **Surgery** (§3.3, App. B, its Table 1): over 20 surgeries in 10 months. A clean rerun on the final setup (§4.2) used ~20% of the compute and beat the surgeried agent in more than 98% of games: the surgeried model had plateaued weaker.
* **Rewards** (App. G):
  * Shaped rewards (Table 6: the win 5, gold, experience, kills, deaths, buildings and more) learned far faster than win/loss alone. A win/loss-only run (a 1-hour horizon) still beat the scripted agent, at a large cost in samples (Fig. 16).
  * Rewards are zero-sum: each hero's reward subtracts the mean of the enemies' rewards.
  * Every reward but the win is scaled by 0.6^(T / 10 minutes), T the game time, so the late game's larger rewards do not dominate (eq. 11).
  * "Team spirit": each hero's reward is (1−τ)·its own + τ·the team's mean (eq. 12). τ rose from 0.3 to 1.0 in OpenAI Five and to 0.8 in the rerun (App. C).
* **The horizon** (§4.5, Fig. 6): resumed with a longer horizon (the base 180 s, γ 0.9993), a trained agent won more against its start, up to the longest tried (6–12 minutes).
* **Actions** (App. F): rare actions got their own head, because their signal would drown in a shared one. Point targets are offsets on a 9×9 grid around a unit.
* **Unit observations** (App. E): the unit's last 16 steps of hit points, time since its last attack, who it attacks and who attacks it, its damage, armor and speed, the vector to the observing hero, and the previous action.

**SC2LE**
* Points predicted as independent x and y from a flat vector: 1.2% top-1 accuracy, against 9.5–10.5% for spatial heads (Table 2).
* Agents on a win/loss reward learned to stall into ties (§4.4.1): the best one lifted its buildings out of the easy AI's reach until the 30-minute limit.

## Where our choices differ (inference)

1. **The anchor is reverse KL on the orders only.** AlphaStar uses forward KL on every argument.
   * Reverse KL pulls on an action in proportion to the student's own probability of it. Once RL drives a hero or production order near zero, the anchor stops pulling it back.
   * That matches the decay we measured: human hero orders 3.1% against the clones' 11.5–12.6%.
2. **Every head samples at temperature 1, from a clone of a deterministic script.** AlphaStar samples at 0.8, and our own micro clone won more played greedily (79% against 68%).
3. **The value head sees only the player's fogged view.** AlphaStar's value with the opponent's information was its largest RL ablation.
4. **Every unit decides every 0.5 s.** AlphaStar found that more actions hurt.
5. **Points are flat x-then-y bins**, which SC2LE found ~8× worse than spatial heads.
6. **One long run through many recipe changes.** OpenAI's clean rerun beat its long surgeried run.
7. **Reuse 2 with staleness ~2**, which OpenAI found slows learning.

Since then (2026-10-17): 1 is changed (forward KL on every head, to a recent checkpoint). 3 is changed (the value also reads 23 numbers of the whole game). 2, 4 and 5 stand. 6 stands in another form: each experiment is a fork from the last best checkpoint. 7 stands (reuse 2, staleness ~1.5).

## The experiments, ranked

| # | experiment | targets | cost |
|---|---|---|---|
| 1 | a temperature for the target heads (0.8, 0.5) | group targets scatter, fights | 1–2 h |
| 2 | forward KL to the anchor on every head (orders, unit, x, y) | rare orders decay (heroes, production), human and undead | 2–3 h |
| 3 | features: each own unit's last order target (dx, dy, steps since), units targeting this one, hit points over 1/4/16 steps | scatter starts in the clone (~415 world units of point error) | 4–6 h |
| 4 | a minimal group order: a unit's point head also reads the targets of the units that chose the same order | coherent groups | 4–6 h |
| 5 | a value with both players' information (in training only) | noisy advantages in fights | 6–10 h |
| 6 | UPGO; separate ratios for the order and its target | credit | 4–6 h |
| 7 | fewer decisions: a "keep this order k steps" head | credit, interrupted attacks | 1–2 days |
| 8 | team spirit: per-unit rewards (its own losses, its kills) blended with the team's | fights, credit | 1–2 days |
| 9 | knobs: one epoch (reuse 1), a lower lr; later a clean rerun of the final recipe from the clone | staleness, path dependence | — |
| 10 | a minimap with units scattered onto it and a spatial point head; build-order goals (z) | spatial reasoning, late tech | days |

Already in progress: shared sampling noise for the targets (`fgself-13`), and hit-point-change and sine/cosine position features (branch `features-v2`).

Results so far (2026-10-13, share of the AI's record on evaluation games, head to head): #2 forward KL on every head **28% → ~69%** (with a moving anchor and training against insane, ~88%); #1 target temperature, #3 features (hit points, positions: none; last orders: worse), #5 the two-player value, #8 team spirit (`fgself-30`): no gain; #6 **UPGO ×0.5: 79% → 102%** (`fgself-31`), ×1.0 on from there: 94% → 111% (`fgself-32`). Not from the papers: the cloning's dropout (0.1) had stayed on in the updates while the actors act without it; off (`--rl-dropout 0`, `fgself-39`): 93% → 109% (insane 55% → 81%). Later (2026-10-14/16): #9's lower learning rate direction held (twice the rate hurt twice); a deeper network (5 layers, grown) 148% → 156%; dropping the built-in AI's advice and cloning loss once the policy beat it 137% → 152%; a horizon for the real game's 25-minute games (γ 0.9997) 55% → 59% of games won against insane; #8 team spirit, hero-level shaping, temperature 0.8 (passive play) and a clone of the real rules did not help. In the real game (2026-10-16/17): half the shaping (`--shaping 0.5`, `fgself-73`) 62% → 72% of games won against insane, so the win weighs more than the material lead. A quarter (`fgself-75`) is running.

Next from the papers (2026-10-17, agreed with the user): AlphaStar's exploiters, against the policy's one strategy (an early rush of tier-1 units). Then #10's build-order goals: a target build order and unit mix a game (AlphaStar's z), from the built-in AI's Echo Isles games and later from human replays, with a small reward for following it.

Sources: [AlphaStar (DeepMind's PDF)](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf), its [supplementary data](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1724-z/MediaObjects/41586_2019_1724_MOESM2_ESM.zip), [OpenAI Five, arXiv 1912.06680](https://arxiv.org/abs/1912.06680), [SC2LE, arXiv 1708.04782](https://arxiv.org/abs/1708.04782).
