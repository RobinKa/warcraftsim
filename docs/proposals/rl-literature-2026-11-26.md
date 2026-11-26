# RL literature against the plateau (2026-11-26)

**Question.** Since 2026-11-20 every change lands at ~1,880-1,925: generations, 10 layers, league exploiters, a learner tax, anchor moves, net-worth shaping, the shaping weight both ways, six lessons, lr 2e-5. Which ideas from recent RL work (2022-2026) could break that, on one RTX 3090 and the real game? This is a reading of the literature against our own numbers. Nothing here was run. The one new measurement (below) was a CPU-only read of saved checkpoints and logs.

## What our own numbers say first

Four facts about the optimizer, the first two from earlier logs, the last two read for this note:

| fact | number | source |
|---|---|---|
| The policy gradient is nearly all noise per step | policy noise scale ~31,000 agent steps (target heads ~79,000) against a minibatch of 256; the value term ~950 and the KL term ~1,800 are clean | `fgself-76`, [experiment log](../reference/experiment-log.md) |
| The trust region never acts | approx KL to the behaviour policy 0.00037 per unit per update, clip fraction 0.4% (last 164 updates of `fgself-266`); the step is set by lr alone (constant 3e-5, no schedule in `selfplay.py`) | `runs/fgself-266/train.jsonl` |
| The noise ball is small in weights, large in play | raw `fgself-252`/26,968,282 is 2.0% of the transformer layers' weight norm away from its own 17-27M average (0.56% of all weights) and rates 100-250 lower; 236's average to 252's average: 1.4% of all weights | checkpoints, CPU read |
| Little weight growth since October | the transformer layers' norm 125 → 153 (+22%) from `fgself-100` to the soup; all weight matrices +4% | checkpoints, CPU read |

Two readings fit all of the plateau's facts. Each points at different ideas.

**A. A noisy executor's optimum.** The learner acts with the raw weights, so every training game (its own sides, the self-play opponent, the league's snapshots) is played by a policy 100-250 points below the average we judge. A policy gradient on that data improves the *expected result of a perturbed policy*. A strategy with few forgiving decisions (a tier-1 rush) keeps its value under perturbation. One that needs a chain of exact decisions (expand on time, hold it, tech) loses value under it. The average inherits a strategy picked for a noisy executor. This explains why forks from an average lean up and then fall back (the iterate re-enters the noise ball), why generations do not compound, and why no change to the reward, the league or the lessons moves the level: none of them changes the noise. Ideas 1 and 2 test this cheaply.

**B. The objective is saturated.** 80% of training games are against the AI, won 0.92 with the curriculum at its floor. The league is rushers like the learner. Its only style opponents are the learner itself following targets poorly. ROA-Star found the same thing in StarCraft II: AlphaStar's unconditioned exploiters "lose efficacy in the later stage of training". Ideas 3, 5 and 7 address this.

Plasticity loss (idea 8) fits less well. The weights grew little, and the 10-layer growth (fresh capacity) leaned up and gave it back like every other fork.

## Ranking

Ranked by expected rating gain per effort. **Bold** = could plausibly give +100. Costs: S = about a day of code, M = 2-4 days, L = a week or days of the machine, XL = more. Flag names in the first experiments (`--act-ema`, `--lr-decay`, `--exploiter-z`) are proposals: none exists yet. "The soup" stands for whatever the base is when a run starts.

| # | idea | bold | cost | first experiment | evidence here |
|---|---|---|---|---|---|
| 1 | Act with the average (EMA weights act and snapshot, PPO's ratio against the EMA) | **yes** | S-M | fork from the soup with `--act-ema` (half-life ~3.5M steps); read raw vs EMA and the EMA's rating at 27M and 54M | for: raw is 100-250 below its average; against: 243 (a strong average as a league member) did not lift the learner |
| 2 | Power-law step-size decay, judged over a longer budget | no | XS | lr 3e-5 · (1 + u/1000)^-0.5, floor 8e-6, 80M against a constant-lr control | for: Ataraxos, Generals.io, the PPO stagnation paper; against: lr 2e-5 (234) not kept, but judged at 27M |
| 3 | Goal-conditioned exploiters and opponent modelling (ROA-Star) | **yes** | M-L | the main exploiter keeps one target family per life and resets to the latest main; then an explorative one on expansion, tier-3 and air families | for: the rush equilibrium, z works but loses, exploiters lose bite; against: 261 (league exploiters) gave its gain back |
| 4 | A better value: classification loss, then a separate privileged critic | no (maybe) | S → M | HL-Gauss on today's head from the soup; then a separate 5-layer critic in the learner | for: the noise numbers above; against: none tried |
| 5 | Minimax exploiter: the main's value as the exploiter's dense reward | no | S-M | exploiter reward += -(change in V_main) | for: our exploiter needed repairs and still sits at ~0.47 against the main |
| 6 | Late starts picked by learning potential (Prioritized Level Replay) | no | S-M | sample captured states by their games' mean abs(advantage), with staleness | mixed: own late starts kept (166), lost positions (230) and people's states (167) level |
| 7 | Near-optimal diversity for targets (DOMiNO-style constraint per target family) | maybe | M | per-family Lagrange weight on the z reward: loosen where the family loses more than 0.1 under no-target play | for: targets win 0.57 against 0.64-0.98 without; against: z-reward sweeps (79, 89) level |
| 8 | Reincarnate into a fresh, wider network, with continual regenerative regularisation | **yes** (high variance) | L | distil the soup into a fresh d-256 net on its own games (QDagger), then RL with L2-to-init | weak: weights grew 22%; 10 layers gave back its gain |
| 9 | One conditional network for the whole population (NeuPL) | maybe | XL | not first; after 3 | small-scale evidence only |
| 10 | Strategy-level search with real-game rollouts (Ataraxos, Gumbel) | no | XL | not recommended now | against: GRPO branches (96, 97): inexact replays, flat returns |

Suggested order: 1 and 2 as two forks in one burst (both cheap, both test reading A), then 3 (the main build if reading A fails), with 4's classification loss riding along in 3's control. Ideas 2, 7 and 8 need a read past 27M. The fork-against-control protocol at ~26M favours changes that are quick early, and those three trade early speed for a higher level later.

---

## 1. Act with the average **(bold)**

**Papers.** Hilton, Cobbe, Schulman, [*Batch size-invariance for policy optimization*](https://arxiv.org/abs/2110.00641) (NeurIPS 2022; PPO-EWMA, [code](https://github.com/openai/ppo-ewma)). Straka, Lisý, Schmid, [*Superhuman AI for Generals.io Using Self-Play Reinforcement Learning*](https://arxiv.org/abs/2606.23348) (2026). Sokota et al., [*Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search*](https://arxiv.org/abs/2511.07312) (2025, "Ataraxos"). Background: Defazio et al., [*The Road Less Scheduled*](https://arxiv.org/abs/2405.15682) (NeurIPS 2024; schedule-free: the gradient is taken at a point between the iterate and its average). Plappert et al., [*Parameter Space Noise for Exploration*](https://arxiv.org/abs/1706.01905) (ICLR 2018; the counterpoint).

**Mechanism.** PPO-EWMA separates the *proximal* policy (the trust region's centre) from the *behaviour* policy (which made the data) and makes the proximal policy an exponential moving average of the weights. Here the proposal goes one step further: the EMA also *acts*. The inference server gets the EMA's weights instead of the raw ones, so the behaviour log-probs PPO already stores are the EMA's, and the ratio π_θ/π_EMA keeps the raw iterate near the average. League snapshots and the self-play opponent are EMA weights too. The raw weights only take the gradient steps.

**Why here.** This is the only idea that acts on our strongest measured fact: the average is worth +100-250 over the iterate around it. Under reading A it removes the noisy executor from the training data while keeping the gradient noise that drives learning. Generals.io and Ataraxos, the two strongest small-compute self-play results of the past year, both deploy an EMA with a decay of 0.999 per iteration. In Generals.io the EMA led the last iterate by ~65 Elo on average from iteration 4,000 and was still 32 ahead at 39,000. Neither acted with it ("the opponent is the current iterate"), so acting with the EMA is untested in the literature. What is new here is the gap: our raw policy is 100-250 points down, not 30-65.

**Against.** `fgself-243` put a strong average into the league as a fixed member and it did not lift the learner. That was a frozen 1.2-6% of games, not the learner's own data. Parameter noise is also exploration (Plappert et al.), so the EMA may play narrower. And if KL(raw ‖ EMA) per unit is well above the clip range, the clip will start to bind (today 0.4%).

**Cost.** S-M. An EMA copy in the learner (`EmaAnchor` for `--ref-ema` already keeps one), `publish()` sends it to inference and snapshots, and the checkpoints save it. If the clip binds, fall back to truncated importance weights (V-trace's ρ̄ = 1) against the stored behaviour log-probs.

**First experiment.** A check without a run: the per-unit KL between `fgself-252`/26,968,282 and `swa-17-27` on ~10k recorded states (CPU). That tells whether the clip would bind. Then a fork from the soup with an EMA half-life of ~3.5M steps (β ≈ 0.9984 per update; about the SWA window's mean age), every other flag as the control. Reads: the training games' win rates (should jump at once), the clip fraction, and in a burst the EMA's rating at 27M and 54M against the control's SWA averages over the same windows. **Kill criterion:** the EMA at 54M is not above the control's 17-27M average. Then reading A is wrong, and idea 3 is next.

## 2. Power-law step-size decay, judged over a longer budget

**Papers.** Beukman, Khetarpal, Zheng, Dabney, Foerster, Dennis, Lyle, [*Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments*](https://arxiv.org/abs/2603.06009) (2026). Sokota et al. 2025 (Ataraxos, above). Straka et al. 2026 (Generals.io, above).

**Mechanism.** Beukman et al. treat PPO's outer loop as stochastic optimisation. A plateau is "thrashing" around a local optimum when the step is too large for the update noise, with large gradient norms, no gain, and degradation with more samples. A smaller step (a stronger proximal term, or more data per update) restarts learning, and the best step shrinks as the budget grows. Ataraxos: power-law annealing of Adam's lr (clip(0.5 · iteration^-1.1, 5e-6, 1e-4)) was "crucial ... for preventing plateauing later in training". They also found that strong regularisation with large steps suits a weak policy and weak regularisation with small steps suits a strong one. Generals.io also decays lr by a power law (to ~2e-5 was safe).

**Why here.** Our lr has been a constant 3e-5 while the policy went from clone to 0.95 against insane. "Forks lean up, then fall back" is what thrashing looks like from outside. And the one lower-lr test (`fgself-234`, 2e-5) was judged at 27M, which is the budget where a smaller step is predicted to look level.

**Against.** Beukman's mechanism is strongest with many epochs on one batch. We do one epoch, and our per-update KL (0.00037) is already ~30x below usual PPO. Here the relevant effect is the size of the noise ball, which averaging already removes from what we *judge*. A decay can only help through what we *train on*, which is reading A again. Expect less than idea 1.

**Cost.** XS. An `--lr-decay` flag (like `--shaping-anneal`) and a control.

**First experiment.** From the soup: lr = 3e-5 · (1 + u/1000)^-0.5 with a floor of 8e-6 (~9e-6 by 80M), against a constant-lr control, both read at 27, 54 and 80M. Also read the gap between each run's raw end checkpoint and its average. If the gap shrinks and the average climbs, the step was the limit. An optional second arm anneals `--ref-kl` the same way (Ataraxos anneals its magnet; we have moved our anchor's target but never annealed its weight).

## 3. Goal-conditioned exploiters and opponent modelling (ROA-Star) **(bold)**

**Papers.** Huang et al., [*A Robust and Opponent-Aware League Training Method for StarCraft II*](https://papers.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf) (NeurIPS 2023). Han et al., [*TStarBot-X*](https://arxiv.org/abs/2011.13729) (2020; diversified league, specific exploiters). Ye et al., [*Towards Playing Full MOBA Games with Deep RL*](https://arxiv.org/abs/2011.12692) (NeurIPS 2020; Honor of Kings, multi-teacher distillation).

**Mechanism.** ROA-Star replaces AlphaStar's unconditioned exploiters with two conditioned kinds. **Exploitative exploiters** each take one strategy statistic z from the top-N by the main agent's win rate, keep it for their whole life, and reset to the *latest main*. They sharpen what already works until it beats the main. **Explorative exploiters** take a z the main executes worst (largest edit distance between target and executed build) and reset to the supervised agent. They are rewarded per correctly built entity *until the first mistake*, plus a strategy-guided loss on that missing entity (a Go-Explore-like curriculum). They become competent players of styles the main does not play. Separately, an opponent-modelling head predicts the opponent's hidden units, research and tech buildings from what is visible. Its latent feeds the policy, and a scouting reward pays for each 30-second drop in that prediction's cross-entropy.

**Why here.** This is our reading B in StarCraft II, fixed at a compute scale nearer ours: 64 V100s and ~11,000 steps a second per agent, against AlphaStar's 256 TPU cores and 50,000. In their 5-day ablations at equal compute, conditioned exploiters beat random-z and unconditioned ones in Elo and worst-case robustness. Opponent modelling raised the main's win rates against four fixed strategies (one from 22% to 52%). Their main agent took more than 50% against top professionals. In our league the only opponents with a style are the learner itself following a target poorly (0.37-0.62 against insane with a target). An explorative exploiter on the expansion, tier-3 or air families is the "competent expansion or air player" that our proposals' style exploiters (item 5 of the 2026-11-20 plan, never built) asked for. Here it comes with a training recipe that has worked at scale. Our `--z-build-labels` already resembles their strategy-guided loss.

**Against.** `fgself-261` (league exploiters) leaned up +20 and gave it back. It was unconditioned, which is the case ROA-Star says fails late. Our forced expansions (254, 256) lost: a competent expansion style may not beat the rush on Echo Isles. If so, the explorative exploiters stay weak and only the exploitative half matters.

**Cost.** M-L. The exploiter's z fixed per life plus a per-family execution-deviation tracker (from the episode rows), and resets to the latest main for the exploitative kind: 2-3 days. The opponent head on privileged labels (the value's 23 numbers already hold the enemy's workers, army, buildings and heroes; richer labels need the opponent's entity list) and the scouting reward: 3-5 more.

**First experiment.** Exploitative half only, on the current recipe: `--exploiter-z top` (one family a life from the top-5 by the main's training win rate in that matchup), reset to the latest main with its Adam state. Judge by rating against the control and by how often the exploiter reaches 0.6. Second run: one explorative league exploiter on {two bases, tier 3, air} with the first-mistake curriculum reward.

## 4. A better value: classification loss, then a separate privileged critic

**Papers.** Farebrother et al., [*Stop Regressing: Training Value Functions via Classification for Scalable Deep RL*](https://arxiv.org/abs/2403.03950) (ICML 2024; HL-Gauss). Cobbe et al., [*Phasic Policy Gradient*](https://arxiv.org/abs/2009.04416) (ICML 2021). Aitchison and Sweetser, [*DNA: Proximal Policy Optimization with a Dual Network Architecture*](https://arxiv.org/abs/2206.10027) (NeurIPS 2022). Vinyals et al., [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z) (2019; the baselines see the opponent's observations). Generals.io (above) uses HL-Gauss inside PPO.

**Mechanism.** HL-Gauss trains the value as a histogram over return bins with a Gaussian-smoothed target and a cross-entropy loss. It was more robust to noisy, non-stationary targets and scaled better with transformers than MSE in Atari, robotics and chess. PPG and DNA give the value its own network, so its clean, strong gradient no longer shapes the policy's features. An asymmetric critic may see what the policy cannot.

**Why here.** All the policy's learning signal goes through the advantages, and its noise scale is ~31,000 steps. A lower-variance advantage raises the signal per game directly. It is the one lever that changes how much a game teaches, not what it teaches. The long games are where the value matters most and where it sees least: development pays 10-20 minutes later under fog, and our critic sees 23 summary numbers of the opponent. Also, with vf-coef 0.5 on a shared trunk, the value term (noise scale ~950) dominates the trunk's gradient over the policy's.

**Against.** Nothing here has been tried, so no evidence either way. Most of the gains in the papers come from value-based or offline RL. A separate critic costs GPU time in the update. Advantage filtering can pay for it: Ataraxos and Generals.io train only on the top ~25% of |advantage|, and Generals.io found the same learning per frame at half the cost of an iteration.

**Cost.** S for HL-Gauss on today's head (e.g. 101 bins over the shaped returns' range, σ = 0.75 bin). M for a separate 5-layer critic with the privileged inputs, run in the learner only (if the actors compute values for GAE today, they would move to the learner). L for a critic over the opponent's full entity list.

**First experiment.** HL-Gauss alone, forked from the soup. Reads: the value's explained variance on held-out games, the policy term's noise scale (the `--noise-scale` tool; it should fall below ~31,000 if the advantages got cleaner), then the rating.

## 5. Minimax exploiter

**Paper.** Bairamian et al., [*Minimax Exploiter: A Data Efficient Approach for Competitive Self-Play*](https://arxiv.org/abs/2311.17190) (AAMAS 2024; Ubisoft, tested on For Honor).

**Mechanism.** The exploiter's reward includes a term that pushes the *main's* value down at every step (minimise the opponent's maximum Q). It gets a dense signal for anything that hurts the main, not only the final win. It converged faster than standard exploiters in every setting tested.

**Why here.** Our exploiter needed two repairs (zeroed Adam, the reset threshold) and reads 0.47 against the main in `fgself-266`'s latest log line, with no resets yet (an early read: the run is young). A faster exploiter means more snapshots that beat the main, and more pressure on it. In exploiter games the main is acting, so its value at each of its steps is already computed by inference: no extra network.

**Against.** Small expected gain. The main exploiter's snapshots are a small share of the league. It also helps idea 3's exploiters, so do it inside 3 if 3 is built.

**Cost.** S-M (pair the two sides' steps by game and time).

**First experiment.** Exploiter reward += c · (V_main(s_t) − V_main(s_t+1)) with c = 0.5 (potential-based, so the exploiter's optimum is unchanged). Read the exploiter's time to 0.6 and the main's rating.

## 6. Late starts picked by learning potential

**Papers.** Jiang, Grefenstette, Rocktäschel, [*Prioritized Level Replay*](https://arxiv.org/abs/2010.03934) (ICML 2021). Parker-Holder et al., [*Evolving Curricula with Regret-Based Environment Design*](https://arxiv.org/abs/2203.01302) (ACCEL, ICML 2022).

**Mechanism.** Replay the start conditions (levels) whose recent episodes had large mean |GAE|, a proxy for regret and learning potential. Rank-based priorities plus a staleness term keep every level sampled now and then.

**Why here.** The long game is where we lose, and late starts are where it is trained. The pools hold thousands of states drawn uniformly, and many are already decided (an easy win or a lost cause, which teach nothing). PLR spends the games where the outcome is still open. Unlike "lost positions" (230), it chooses states by what the learner can still change, not by the outcome.

**Against.** The late-start line has been mostly level since `fgself-166` (own states, kept). Expected gain is small.

**Cost.** S-M: a score per captured state from the episode rows, and a sampler.

**First experiment.** Priorities from the last ~2,000 late-start games' mean |advantage| per state, β = 0.3, staleness 0.1. Read wins past 18 minutes and the rating.

## 7. Near-optimal diversity for targets

**Papers.** Zahavy et al., [*Discovering Policies with DOMiNO: Diversity Optimization Maintaining Near Optimality*](https://arxiv.org/abs/2205.13521) (ICLR 2023). Zahavy et al., [*Discovering Diverse Nearly Optimal Policies with Successor Features*](https://arxiv.org/abs/2106.00669) (2021).

**Mechanism.** One latent-conditioned network learns a set of policies that are as different as possible in occupancy features, under the constraint that each stays within a fraction α of the best return. A Lagrange multiplier per latent trades the diversity reward against the extrinsic one, adapting as the constraint is met or broken.

**Why here.** Our targets are imitation goals with fixed weights (`--z-reward 0.02`, `--z-mode-reward 0.1`). A family the policy cannot play well still pulls with full weight, and target play wins 0.57 against 0.64-0.98 without one. A per-family multiplier would loosen the pull where a family loses more than α (say 0.1) against the no-target play in that matchup, and tighten it where it wins. Diversity would then be measured in what matters (workers, halls, tier, air share), not in matching a recorded build. That would make target play nearly as strong as no-target play. It does not raise the rating by itself. It is what makes the proposals' per-matchup strategy picker, or a style exploiter, worth having.

**Against.** The z-reward sweeps (`fgself-79` at 0.05, `fgself-89` at 5x) read level. Those were global weights, not per family under a constraint.

**Cost.** M.

**First experiment.** A per-family λ from the training games' win rates (the 28,000 target games already give them), updated every ~50 updates. Read the with-target win rate per family, then the rating.

## 8. Reincarnate into a fresh, wider network **(bold, high variance)**

**Papers.** Agarwal et al., [*Reincarnating Reinforcement Learning*](https://arxiv.org/abs/2206.01626) (NeurIPS 2022; QDagger: a student distils the teacher while it does RL, with the distillation weight decaying). Schmitt et al., [*Kickstarting Deep Reinforcement Learning*](https://arxiv.org/abs/1803.03835) (2018). Juliani and Ash, [*A Study of Plasticity Loss in On-Policy Deep RL*](https://arxiv.org/abs/2405.19153) (NeurIPS 2024). Lyle et al., [*Normalization and Effective Learning Rates in RL*](https://arxiv.org/abs/2407.01800) (NeurIPS 2024).

**Mechanism.** Train a fresh network, which can be wider, to copy the best policy on its own games (DAgger-style: the student acts, the teacher labels), then go on with RL as the distillation term decays. The new network keeps the old one's knowledge without its weights' history. For on-policy PPO, Juliani and Ash found that intermittent fixes (plasticity injection, ReDo, resetting the last layer) fail, sometimes worse than doing nothing. Continual "regenerative" regularisers work: L2 toward the initial weights, soft shrink-and-perturb after every step, with LayerNorm. Weight magnitude was the best predictor of lost plasticity.

**Why here.** It is the user's fresh-lineage bet at a fraction of its cost: days, not a rerun from the clone. It escapes whatever the October lineage carries (fixed bugs, the clone's basin) while keeping its play.

**Against.** Our check found little weight growth (transformer +22% since October, against the large norm growth plasticity papers report). The 10-layer growth (`fgself-260`, partly fresh capacity) leaned up +24 and gave it back. A student that copies the teacher may land in the same basin. A wider net also slows inference, and the GPU is nearly busy.

**Cost.** L (scripts exist: `grow_model.py`, `bc.py`; days of the machine).

**First experiment.** First a check without a run: dead feed-forward units and the effective rank of the transformer's features on ~10k recorded states, `fgself-100` against the soup. If those show loss, then: distil the soup into a fresh d-256 network on its own games until head to head is ~0.5, then RL on the full recipe with L2-to-init at 1e-4, judged at 54M and later.

## 9. One conditional network for the population (NeuPL)

**Papers.** Liu et al., [*NeuPL: Neural Population Learning*](https://arxiv.org/abs/2202.07415) (ICLR 2022). Liu et al., [*Simplex Neural Population Learning*](https://arxiv.org/abs/2205.15879) (ICML 2022).

**Mechanism.** One network, conditioned on an opponent mixture σ, represents a whole population of best responses. A meta-solver (Nash over the empirical payoffs) sets who trains against whom. Skills transfer between members, and new strategies get easier to reach as the population grows.

**Why here.** Our league members are separate frozen raw snapshots, 100-250 below their averages, and every new member starts from scratch. Our z-conditioning already shows that one network can hold several behaviours.

**Against.** The evidence is small games (running-with-scissors, capture-the-flag, MuJoCo football), symmetric zero-sum only. Nothing at RTS scale. It is a rebuild of the league.

**Cost.** XL. Only after idea 3, if the league turns out to be the lever.

## 10. Strategy-level search with real-game rollouts

**Papers.** Sokota et al. 2025 (Ataraxos: a belief network samples the hidden state, ~1,000 depth-40 rollouts by the policy, one magnetic-mirror-descent step on the values). Danihelka et al., [*Policy improvement by planning with Gumbel*](https://openreview.net/forum?id=bERaNdoegnO) (ICLR 2022; improvement guaranteed with few simulations).

**Mechanism.** At a few macro decisions (which target family, when to expand), try k options by rollout, keep the best by sequential halving, and distil the choice into the policy or a picker.

**Why not now.** Ataraxos could do this because Stratego runs ~10M state updates a second on a GPU. Our game is the real WC3 under Wine, and its branches do not replay exactly through fights with creeps (GRPO, `fgself-96`/`97`: 54% exact, most groups' returns within 0.04). Ataraxos itself dropped search during self-play because the slower data cost more than it gained. The per-matchup picker read as overfitting on today's samples.

---

## Ideas the evidence here already argues against

* **Moving or re-targeting the anchor**: R-NaD (Perolat et al., [*Mastering the game of Stratego*](https://arxiv.org/abs/2206.15378), Science 2022), magnetic mirror descent (Sokota et al., [ICLR 2023](https://arxiv.org/abs/2206.05825)), EMAgnet (Maidment et al., [2026](https://arxiv.org/abs/2606.23995): regularise toward an EMA of the iterate), WARP (Ramé et al., [2024](https://arxiv.org/abs/2406.16768)), EMA policy gradient (Zhang and Ba, [2026](https://arxiv.org/abs/2602.04417)). We tested the rolling anchor (82), anchor = start (141), EMA anchors slow and fast (191, 192: level), the soup as anchor (265: -26), the macro-only anchor (258: -104) and no anchor (171: worse). The untested part is annealing the anchor's *weight* over time: idea 2's second arm.
* **Human data**: AlphaStar Unplugged (Mathieu et al., [2023](https://arxiv.org/abs/2308.03526)), piKL (Jacob et al., [ICML 2022](https://arxiv.org/abs/2112.07544)). Human-clone anchors (215, 216) were not kept, the clones played passively, and 95 stock-map Echo Isles 1v1s are a small set.
* **Go-Explore-style start states** (Ecoffet et al., [Nature 2021](https://arxiv.org/abs/2004.12919)): own late states kept (166); people's (167) and lost positions (230) level. What remains is the choice by learning potential (idea 6).
* **Long-horizon credit** (RUDDER, [Arjona-Medina et al. 2019](https://arxiv.org/abs/1806.07857); temporal value transport; hindsight credit): the forced second base (254, 256) expanded in 35-39% of games and lost. The macro play loses as played, so credit is not the binding limit. UPGO already handles the "later mistakes cancel credit" case.
* **A bigger network by scaling laws** (Hilton, Tang, Schulman, [2023](https://arxiv.org/abs/2301.13442); Neumann and Gros, [ICLR 2023](https://arxiv.org/abs/2210.00849)): 8 layers in October level, 10 layers (260) +24 then level. Only as part of idea 8.
* **Learned-model search** (MuZero, EfficientZero, [Sampled MuZero](https://arxiv.org/abs/2104.06303), [Muesli](https://arxiv.org/abs/2104.06159)): a world model of WC3 with up to 160 entities under fog is out of reach on one GPU, and without one there is nothing to search.
* **Fixes to PPO's ratio** ([SPO](https://arxiv.org/abs/2401.16025), clip variants, [V-MPO](https://arxiv.org/abs/1909.12238)): our clip binds on 0.4% of units, so there is nothing to fix there.
* **Massive-parallelism methods** ([SAPG](https://arxiv.org/abs/2407.20230), [PQN](https://arxiv.org/abs/2407.04811), Beukman et al.'s million environments): built for GPU simulators with 10^4-10^6 environments. We have ~140 CPU-bound games (12 actors x 6 plus 64 on the box). Beukman et al.'s diagnosis carries over (idea 2), its scale does not.
* **Advantage filtering as a gain in itself** (Ataraxos; Generals.io +150-250 Elo at 24 h): the gain came from cheaper iterations at the same learning per frame. We are bound by game data, not by updates. It is useful only to pay for idea 4's critic.

## Sources not linked above

* McCandlish et al., [*An Empirical Model of Large-Batch Training*](https://arxiv.org/abs/1812.06162) (2018): the noise scale used in `fgself-76`.
* Wortsman et al., [*Model soups*](https://arxiv.org/abs/2203.05482) (ICML 2022) and Ramé et al. 2024 are behind our averages and the soup. Their result here: a soup level with its best member.
