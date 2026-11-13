# Ideas

Where the project's ideas came from and what became of them. Many are the user's; others come from AlphaStar and OpenAI Five ([what those papers found](ideas-from-alphastar-and-openai-five.md)), from LLM reinforcement learning, from image generation, or from our own runs. Each experiment's numbers are in the [experiment log](experiment-log.md).

## The queue (2026-11-12)

In the order they run. The code for each is in place unless it says otherwise. [Next experiments](../proposals/next-experiments.md) has the details.

| idea | from | what it is | status |
|---|---|---|---|
| Unit attacks | the user (2026-11-12: workers survive attack-moves) | the policy attacks a unit in <1% of its attacks; a gentle lesson or exploration for it ([next experiments](../proposals/next-experiments.md), 13) | `--attack-lesson` built; `fgself-218` at 0.25 running |
| The heroes' spells, another way | the user (2026-11-11: heroes rarely cast) | lessons made the heroes spam their spells (`fgself-208` to `-210`). A new approach is needed | `--attack-lesson` built; `fgself-218` at 0.25 running |
| Expansions, gentler | ours (audit 15, a bug finder 2026-11-10) | expansion lessons only when the main mine is nearly empty | `--attack-lesson` built; `fgself-218` at 0.25 running |
| A KL anchor cloned from people | AlphaStar (its KL to the supervised human policy) | people's replays again with the obs-full events, a clone from `fgself-212`, `--init` it with `--ref-kl-mode forward-acting` | `fgself-215` (the clone as the only anchor) not kept: h2h 0.31, `fgself-100`'s pull swapped for the clone's; `fgself-216` (`--ref2`: the clone beside `fgself-100`) not kept either (burst: -5 against a null fork, narrower) |
| Fine points around the unit | the user (2026-10-25) | a sub-cell within the chosen point bin (`--fine-points 7`) | merged, not yet in a run |
| Fewer decisions, spatial point heads | AlphaStar (action rate), SC2LE | a "keep this order k steps" head; points on a map, not x then y | later |
| An APM cap | ours | every unit acts every half second now | before games against people |

## Tried

| idea | from | result |
|---|---|---|
| Late starts from the late game, in the real game | ours (2026-11-12: 102 of `fgself-201`'s 118 losses past 18 minutes) | `--real-agents 0.5 --real-late-minutes 16,28 --real-late-from 16` (`fgself-214`): worse in the burst (0.69 against insane; human and undead fell); not kept |
| Faster, less noisy verdicts | the user (2026-11-13) | training games as a read (`scripts/train_read.py`, drifts between days), the laptop mostly head to head (~3x), evaluation bursts on the main machine (~1,400 games an hour, paired, one window) | in use |
| Upgrades in the shaping | ours (2026-11-12) | `--research-value 1` (`fgself-213`): more upgrades, -11 against insane; not kept |
| Every training game in the real game | ours (bold changes) | `--map "(2)EchoIsles"` (`fgself-212`): read as level against a high-point start, then **best against insane** in a paired evaluation burst (0.86, +8 over `fgself-201`/142.7M): the base since 2026-11-13 |
| Half the AI games in the real game | ours (bold changes, 2026-11-06) | `--real-share 0.5` (`fgself-201`): +11 points against insane over the continuation, kept as the base |
| The strongest checkpoint as a league member | the user (2026-11-11) | `fgself-197`'s peak fixed in the league since `fgself-201`'s resume |
| The heroes' spells taught | the user (2026-11-11) | `--spell-lesson`, with a mana cost or without (`fgself-208` to `-210`): the heroes cast ~25-40 spells a game, -10 to -15 against insane, not kept |
| Make sure the agent can do everything | the user (2026-11-06) | `--full-vocab 1`: every order a melee player can give, 13 exclusions with reasons ([order coverage](order-coverage.md)); `fgself-196` kept, level |
| What the agent can't observe | the user (2026-11-06) | `--obs-full 1`: buffs, corpses, items, area spells, enemy upgrades ([observation coverage](observation-coverage.md)); `fgself-197` kept, wins past 18 minutes 16% → 46% |
| BF16 inference | the user (2026-11-06) | `--infer-precision bf16-mixed` and finer inference rows, part of ~4,300 → ~5,800 steps/s |
| Bold changes over one-flag tweaks | the user (2026-11-06, 2026-11-12) | league-first, dry mains, the full vocabulary, obs-full, real-game training |
| AlphaStar's league first, a league with history | AlphaStar | the AI in 40% of the games (`fgself-194`): -13 against insane; snapshots spanning hours (`fgself-199`): level |
| PFSP with data | ours (a bug finder, 2026-11-06) | `--pfsp-data` (`fgself-198`): level |
| Dry mains to force expansions | ours (bold changes) | the `dry` scenario (`fgself-195`): no expansions in evaluation |
| Expansion lessons with the place taught | a bug finder (2026-11-10) | `--expand-lesson 0.5` (`fgself-202`): pulled hard toward early expansions, 40% against insane |
| More maps in the training pool | the user (2026-11-01) | `fgself-178`: level off Echo Isles, worse on it; not kept |
| The value sees the tax | the user (2026-11-01) | `--value-tax 1` (`fgself-184`): 41% head to head, off |
| An EMA anchor | the user (2026-11-01) | `--ref-ema` (`fgself-191`, `-192`): level, no wider |
| Shop items in the targets | the user (2026-11-02) | `--z-shop 3` (`fgself-183`): 3 items a game bought, -22 against insane |
| Training games on a second machine | the user (2026-10-24) | the games box: 56, now 64 training games; ~4,450 steps/s with it (`fgself-168`) |
| Narrow targets, more kinds of conditioning | the user | `fgself-109`: level; the easy kinds followed, the hard ones not |
| Rewards by part (GDPO) | the user (LLM RL) | worse (`fgself-110`, `-116`) |
| Self-distillation with privileged information (OPSD) | the user (LLM training) | no gain (`fgself-112`, `-121`) |
| Neutral shops, the tavern, mercenaries | the user | tavern heroes hired; untaught mercenaries cost real-game wins (`--neutral 2`); shop items as orders kept (`fgself-179`: 66% → 77%) |
| A heuristic teacher | the user | the teacher's lessons label the policy's steps (`--teach 3`), in every run |
| Scenarios, weight averaging | the user (the micro curriculum; model soups) | level (`fgself-114`, `-115`) |
| Skill ratings | the user | a Bradley-Terry rating of every policy: the dashboard's Ratings view |
| A heuristic opponent stronger than the learner | the user (2026-10-25) | `fullgame/heuristic.py`, in the evaluation games (`--eval-heuristic`) |
| Diverse play without macro help | the user (2026-10-25) | `--macros-off mix,creep,expand` (`fgself-151`): kept, it adapted |
| Targets that limit or forbid units | the user (footman spam) | `--z-comp` (`fgself-154`): not kept |
| A lumber lesson | ours (audit 9) | `fgself-150`: kept |
| Same-page merging of the games' memory | ours | on the games box (KSM) |
| Human data | AlphaStar (Fig. 3E) | a clone of people's play was passive (2026-10-27); people's late-game states are late starts (`fgself-168`) |
| The target's first hero taught at the altar | ours | `--teach 2` (`fgself-140`): level |
| Strategic variety first, a lower win rate accepted for a while | the user (2026-10-18) | the direction since; targets the policy follows on its own came with the tech-tree builder (`fgself-95`) |
| Faster games for variety: the duel map, faster rules | the user | `echofast` training judged on the real game: the best line (`fgself-100`, 95%) |
| Teach expanding, creeping, picking up items | the user (2026-10-22) | as lessons they failed (`fgself-123`, `-124`: armies dragged to the camp, halls built at home); as orders of the agent's own that the policy chooses when (`--macros`: expand, creep) the first expansions in the real game and the best policy yet (`fgself-127` to `-129`); items off the ground: `--macros 2` (`fgself-134`) |
| A tax on self-play by who is expected to win | the user (as the races' balance, for targets too) | a logistic model of the outcome from both sides' races and targets and the tax; the favoured side taxed by what evens the game out (`--win-balance`, `fgself-136`) |
| Search the videos | the user | the dashboard's Videos page and each video's target and heroes beside it (`dashboard/videos.py`) |
| Audits for silently wrong assumptions | the user ("find issues with our current approach", after the tavern's) | two audit subagents (2026-10-22) found ~14: dead heroes never revived, undead never expanding, the expand order taken by every worker, the newest units cut from the view, a tier refused behind a worker queue, a dead tavern hero lost for good... |
| GRPO from mid-game branches | the user (from LLM RL: DeepSeekMath's group-relative advantages) | level (`fgself-96`, `-97`): only ~55% of branches replay exactly (creeps and the AI are not deterministic) |
| Classifier-free guidance toward the target | the user (from image generation) | no use (`fgself-106`): it followed no better and won 50% instead of 91% |
| AlphaStar's league (self-play, past snapshots, a main exploiter) | AlphaStar | level (`fgself-103`) |
| Reward decay 0.6^(T / 10 min) | OpenAI Five (App. G, eq. 11) | level (`fgself-104`) |
| Each sample used once | OpenAI Five (§4.4) | level (`fgself-105`) |
| More games as the weaker races | ours | undead 79% → 89% (`fgself-102`) |
| The anchor at the best checkpoint | ours | no slide after the peak (`fgself-101`) |
| Forward KL on every head, UPGO, a moving anchor, a value that sees the opponent | AlphaStar | forward KL and UPGO: large gains; the opponent's information for the value: kept, level |
| The RTX 3090's BF16 rate unlocked (gpu-unlock), 400 W | the user | matrix multiplies 1.4-1.8× faster; self-play's speed unchanged (CPU-bound) ([GPU unlock](gpu-unlock.md)) |
| More of the game's code faster (Ghidra, REA), wineserver | the user | round 2 of `--game-cuts`: -18% of a game's main thread, the same simulation |
| The actors' Python in C | the user (the levers of 2026-10-24) | the units' features in C: View.step 1,002 → 556 µs a side-step; with 8 games an actor ~1,665 → ~1,730 steps/s |
| A larger network | ours | 5 layers helped on the duel maps; 8 level, and the user's direction: no more layers without a clear gain |

## Sources

* AlphaStar, OpenAI Five, SC2LE: see [what the papers found](ideas-from-alphastar-and-openai-five.md).
* GDPO (NVIDIA): rewards normalized one by one before they are summed ([explained](https://aipapersacademy.com/gdpo/)). SAW: stage-aware dynamic weighting, [arXiv 2606.07705](https://arxiv.org/abs/2606.07705).
* GRPO: Z. Shao et al., DeepSeekMath, [arXiv 2402.03300](https://arxiv.org/abs/2402.03300).
* Classifier-free guidance: J. Ho and T. Salimans, [arXiv 2207.12598](https://arxiv.org/abs/2207.12598).
* Model soups: M. Wortsman et al., [arXiv 2203.05482](https://arxiv.org/abs/2203.05482). Task arithmetic: G. Ilharco et al., [arXiv 2212.04089](https://arxiv.org/abs/2212.04089).

**More maps in the training pool** (the user, 2026-11-01: Echo Isles may favour rushing; more maps also open more replays). Straight-line base distances do not single it out (Echo Isles 9,856, Amazonia 9,458, Last Refuge 10,549, Terenas Stand 11,860; walking distances not measured), and people on it play a median 13.8 minutes, so first measure: `fgself-168`'s policy against insane on two or three other stock maps, by win rate and game length, with no training. Needs each game's frame in evaluation (`bc.py`'s `game_frame`; the run's is set once today). Measured on 2026-11-01: [the policy on other maps](human-replays.md#the-policy-on-other-maps-2026-11-01). Built behind a flag: `--map-pool` ([Several maps](../environments.md)).

**The value sees the tax** (the user, 2026-11-01). Neither network reads a game's tax, so the value cannot tell a taxed AI from an untaxed one. In `fgself-173`'s training games against the AI, untaxed launches (11%) won 17%, taxed ones 43-64%. Within a matchup the tax moves little (standard deviation ~0.07, means 0.10 to 0.45 between matchups), and the value can see the races, so most of it can be learned from the matchup; the untaxed launches are what it cannot tell apart. Between agents 87% of games carry a tax (mean 0.11), and being taxed over 0.2 won 24% to ~33%. If a tax stays, give it to the value only (`model.py`'s `priv_proj`: a block added to the value head's input, zero at first); never to the policy, since evaluation has no tax. To test as one change, depending on `fgself-174`. Built behind a flag: `--value-tax 1` (two numbers after the privileged block, the opponent's tax and its own, read by a value-only projection, `tax_proj`, zero at first).

**Loosening the KL anchor, after `fgself-171`** (the user, 2026-11-01: its replays looked more varied; update the anchor, or an EMA anchor?). Without the anchor orc and night elf played more varied, but the macro fell and it lost. Past moves: a rolling anchor (`fgself-82`, every 200 updates, worse, but without shaping and at doubled income), the anchor at the fork's start (`fgself-141`, worse and falling). A fixed anchor on a recent checkpoint holds it to one-unit habits, so these come first: (1) `--ref-kl 0.01`, no code; (2) an EMA anchor (the user's idea: the anchor follows the policy slowly, so it stops sudden jumps without pulling back to `fgself-100`; risk: it drifts as the macro does, only later); (3) `fgself-171`'s last as a fixed league member (`--league-add`), which puts its armies in front of the main policy, no code; (4) if the macro is what falls, the anchor on workers and halls only, so army choice is free.
