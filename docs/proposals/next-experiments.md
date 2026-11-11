# Next experiments

**As of 2026-11-01.** The list from this week's findings: one mine caps income at half the insane AI's and ~79% of `fgself-168`'s losses come past 18 minutes ([the table](../reference/experiment-log.md)); the KL anchor holds the macro up but holds some variety back (`fgself-171`); and the policy cannot buy from its own shops. One change a run, ~3.5 hours each.

**Judging.** At ~50M steps and ~250 games a policy, from a run's second hour on, against its start: head to head, against insane by race, wins past 18 minutes, expansions, gold a minute, the top unit's share, with a target. Stop at ~30M steps if head to head with the small evaluation set is at or under ~45% and against insane is not above its start (`fgself-169` and `fgself-171` would have stopped, every kept run gone on; see [the log](../reference/experiment-log.md)); a variety run reads its behaviour from the second hour. The evaluation set is the current policy, with a target, and the start.

## Bold changes (2026-11-06)

The user: "lets make bold meaningful changes instead of small ones". Every one-flag run since `fgself-176` read level, and the play stayed mass tier-1 on one mine. Each run below changes how the policy trains, not one knob. They are judged at ~50M steps and read on to ~100M (runs go on overnight).

1. **League-first** (`fgself-194`, running). The fixed main exploiter, plus the built-in AI in 40% of launches instead of 80%. AlphaStar's main agents train against their league; ours trained mostly against one scripted AI.
2. **Expansions forced by the game** (the `dry` scenario, built). Both mains start with 25-60% of their gold and both naturals' camps are cleared, in a large share of the training games, so a side must expand or lose its income. It uses no labels (the expansion lessons hurt), only the game's own pressure; the main's gold is in the policy's inputs.
3. **Train on the game we are judged on**. The real Echo Isles rules (the evaluation's) as a share of the training games beside `echofast2`. Real-rules training pulled down at `fgself-98/99`, but that was before the late starts, scenarios, masks and audit fixes, and the long games it loses are real-rules games.
4. **League exploiters** (code). Exploiters that train against the whole league and join it, as AlphaStar's do, so the league holds members that play differently.
5. If 1 holds against insane: no built-in AI games in training at all, with insane only as the yardstick.
6. **PFSP with data** (found 2026-11-06; built 2026-11-09 as `--pfsp-data X`: X of the past-snapshot games untaxed (was 0.25) and their target and scenario games counted, ~20-40x the data). Only ~1% of the games against past snapshots enter PFSP's records: the win-balance tax, scenarios and targets each take a game out (audits 4-6), so the league is drawn evenly. Both the tax and PFSP aim at even games, and they pull against each other. Options: count target and scenario games (their bias is the same for every opponent), or no win-balance tax in games against past snapshots. Needed before the league plan can mean what it says.
7. **Complete the vocabulary** (built and merged 2026-11-06: `--full-vocab 1`, 111 orders and 51 unit types more; `scripts/order_coverage.py` finds every race at 100% but 13 exclusions with reasons, [the table](../reference/order-coverage.md); each order probed in the real game, `scripts/vocab_probe.py`). The orders are those the demonstrations gave 5+ times, so whatever the built-in AI never does is missing. The agent can never make spirit walkers, hippogryphs and riders, siege engines, destroyers or shades. It can't build arcane or cannon towers, watch towers, nerubian towers or ancient protectors. It lacks about 20 unit abilities (polymorph, cripple, possession, faerie fire, cyclone, rejuvenation, bear and crow form, berserk, aerial shackles, flare, sentry ward, spirit link, ancestral spirit, abolish and devour magic, absorb mana, corporeal form, sacrifice) and patrol and hold position. Much of people's tier-3 play is out of reach. Added as `--spells` and `--shop-items` were: at the vocabulary's end, new logits low, old ones exact, each order probed in the real game.
   The user (2026-11-06): "make sure our agent can do everything". So the goal is complete coverage, checked by a script against the game's own data (`docs/reference/order-coverage.md`): every unit, building, upgrade, research and ability order of the four races, autocast toggles, morphs, item handling (drop, give, sell, pick up), patrol and hold position, and the neutral buildings on our maps (mercenaries, the goblin laboratory). Whether a run uses mercenaries stays a flag.

8. **Observe what a player can** (audit merged 2026-11-06, [the table](../reference/observation-coverage.md); built and merged as `--obs-full 1`, identical outputs until trained, reviewed and fixed (stale corpses whose ids live units took); an fp32 GPU fast path may treat the fading -30 bias as fully hidden, so laptop evaluation can differ from training only during `--obs-fade`; `--features 55` turns on 26 channels `--features 29` zeroes, also identical). The ranked gaps: none of 160 buffs and debuffs is an input (a polymorphed grunt reads as a grunt), fog memory and scouting, corpses, enemy upgrade levels, item identity and charges and items on the ground, area spells in progress, production progress, 7 Echo Isles creep types read as unknown, non-hero cooldowns. New entity kinds need a -30 attention bias to leave a network as it was; zero weights alone don't (the suspected cause of `fgself-107`'s loss).

9. **A league with history** (found 2026-11-09 in `fgself-198`'s league). A past snapshot is taken every 40 updates (~1 minute; `--snapshot-every 40` since `fgself-170`) and only the newest 6 are kept (`--max-past 6`), so each lives ~6 minutes and gets 3-9 games before it rotates out; the newest starts at weight 1.0 with none. Besides `fgself-130`, `fgself-171` and `fgself-78`, the league is the policy of the last few minutes: close to plain self-play, and PFSP has little to work with. Snapshots every ~400 updates and ~12 kept would span hours, as AlphaStar's league spans its training. No code: `--snapshot-every`, `--max-past` (check that `--checkpoint-every` stays 40 for resumes).

10. **Why the policy never expands** (found 2026-11-10, a bug finder; fixed 7dfea5b, 8438f29). No mask or harness rule stops it, but every expansion help ran in target games only (the camp and hall lessons, the builder's saved gold, the lumber lesson): no-target games expanded 0 of 1,422 times against 764 of 4,300 with a target. And the teacher's loss never covered the point, so a lesson taught "build a hall" and not where (halls at home in 147 no-target games; the likely reason the expansion lessons of `fgself-156/157` hurt). Human builders were also unprotected while they built (their order reads "repair"), leaving stalled frames in a third of human games. Next: a run with these fixes, then the expansion lessons in every game (`--expand-lesson`) now that they teach the place. Undead: gold mines out of sight are no entities, so a haunt order at an unseen natural is dropped.

11. **Why the heroes rarely cast many of their spells** (found 2026-11-11; built as `--spell-lesson X`, off by default). Not the masks: in the states where the hero had the spell learned, cooled down and paid for and a person would cast it (an enemy unit in range, a clump in reach, a fight on), the masks allowed it, in 65 of people's Echo Isles games and in 21 probe games of `fgself-201`'s own (late starts; the allowed table, the costs and tech masks, the futile and full-vocab checks, the valid bits). The exceptions are by design (an autocast's "off" while it is off; Death Coil and Holy Light with nothing near they can hit: a mortar team, a living enemy) but one: the demon hunter's demon form, whose row of the order table (from the demonstrations) had no immolation, no attack on a unit and no stop (fixed: `model.hero_forms`). The policy (105.6M) gives these spells almost nothing of a hero's order distribution there: immolation, thunder clap, windwalk, divine shield, drain ~0.0000, blizzard 0.0005 (people's states) / 0.004 (its own), mana burn, hex, sleep, frost armor ~0.001, breath of fire 0.003, against death coil 0.14, frost nova 0.19, storm bolt 0.13, holy light 0.32, healing wave 0.44. Never sampled, so never reinforced. Immolation was never on in its own fights (0 of 18 steps, from the obs-full events), so the engine does not cast it for it either. Two limits of the vocabulary besides: blizzard, flame strike, shockwave, impale and carrion swarm exist only aimed at a unit (as the built-in AI casts them; people aim them at a point), breath of fire only at a point. The lesson (teacher.py): an area spell at the clump's centre (3 units within its area, in reach), a nova with 3 units around the hero, immolation on in a fight and off after (its state from the obs-full events), hex, sleep, mana burn, entangling roots... on the nearest enemy hero in a fight; each hero's spell looked at every 10 s (hex 30 s), a lesson given with chance X. At X = 1 it labels about as often as people cast in 30 of their games (blizzard 0.25 against 0.19 a side-game, war stomp 0.33 / 0.55, mana burn 0.22 / 0.62, hex 1.5 / 1.1), and ~2 times a game in the policy's own late starts. A run: a fork with `--spell-lesson 1`; the episode rows' `taught` ("spell AHbz": lessons) and `spell_casts` (the learner's casts sent of the taught spells), and the replays' traces.

Rejected changes worth switching on anyway (2026-11-06 review): none of the tuning knobs. `--value-tax` (correct in principle, 41% at 30M), the gentle hunt lesson and the one-item shop targets are worth trying again inside a bold run; mercenaries only once they are taught.

## Running

| Run | Change | Why |
|---|---|---|
| `fgself-207` (stopped at 14.7M) | a plain fork of `fgself-197` on its own code + the fade fix | Dropped too (35% h2h, insane -14): forking itself takes a strong start down; resuming does not |
| `fgself-205` (stopped at 13M) | all of the day's fixes, no blind start (5584484), from `fgself-197` | Below `fgself-197` (34% h2h, insane -23) |
| `fgself-204` (stopped at 14M) | the fixes with `--teach-point 0` | Confounded: every fork of `fgself-197` began blind to the new entities for ~600 updates |
| `fgself-203` (stopped at 30M) | the expansion fixes alone (no `--expand-lesson`) | 33% h2h against `fgself-197`, insane -19: the fixes hurt too |
| `fgself-202` (stopped, collapsed) | expansion fixes + `--expand-lesson 0.5` (item 10) | It expanded (17% of no-target games), but h2h 22% |
| `fgself-208` (running) | `--spell-lesson 1`, from `fgself-201` | Teach the heroes' spells the policy never casts |
| `fgself-201` (re-read 2026-11-11: **better**, kept as the base) | `--real-share 0.5` (bold 3), real launches in the real game (fixed), from `fgself-197` | Insane 0.81 against the continuation's 0.70 (+11, ~3.5 SE); dropped first only against a peak checkpoint; resumed |
| `fgself-200` (invalid) | `--real-share 0.5` | Its "real" launches played `echofast2` untaxed (the bug since the start; fixed e4e28ce) |
| `fgself-199` (not kept) | `--snapshot-every 400 --max-past 12 --pfsp-data 0.5` (bold 9), from `fgself-197` | Wider, but -19 against insane |
| `fgself-198` (not kept) | `--pfsp-data 0.5` (bold 6), from `fgself-197` | 47% h2h, insane -9 (n.s.); the league was minutes deep |
| `fgself-197` (kept, better, new base) | `--obs-full 1 --obs-fade 600` (bold 8), from `fgself-196` | 53.5% h2h vs `fgself-196`, 57% vs `fgself-182`; wins past 18 min 46% vs 16%; wider |
| `fgself-196` (kept, new base) | `--full-vocab 1 --features 55` (bold 7), `--learner-intake 2` | Level at 50M (48% h2h, insane 67/69); every order a player can give; new orders in use |
| `fgself-195` (not kept) | the `dry` scenario in a third of scenario starts (bold 2) | No expansions in evaluation (0 of 241); head to head 43%, lumber and creeps down |
| `fgself-194` (not kept) | `--exploiter-share 0.1 --ai-share 0.4` (bold 1) | Level head to head, -13 against insane (only ~12% AI games); no wider; the exploiter never joined |
| `fgself-193` (invalid, stopped at 5.9M) | `--exploiter-share 0.2` (item 6, brought forward) | The exploiter path was broken (fixed in 8f57073): run again as `fgself-194` with `--exploiter-share 0.1` |
| `fgself-192` (not kept) | `--ref-ema 0.00046` (4× faster) | Level, play no wider: the anchor is not what holds variety back now |
| `fgself-191` (stopped at 75M) | `--ref-ema 0.000115` (item 5) | Level throughout, play no wider |
| `fgself-190` (not kept) | More `base2` starts | Level (47% head to head, insane +0.8); still no expansions |
| `fgself-189` (stopped at 30M) | `--minibatch 512 --lr 4.2e-5 --pad-groups 2` (item 9) | 42% head to head, and slower in a run (the games' inference waited): minibatch 256 stays |
| `fgself-187` (not kept) | `--pfsp uniform` | Level, but narrower play (upgrades, anti-air down): PFSP stays |
| `fgself-186` (not kept) | No `natural` scenario | 44% head to head, insane -3.6: the scenario stays |
| `fgself-184` (stopped at 30M) | `--value-tax 1` (item 8) | 41% head to head; the value's loss ~13% lower early, then level |
| `fgself-183` (stopped at 30M) | `--z-shop 3` | It buys 3.2 items a game and lost 22 points against insane: too much, the wrong items early. A gentler form later (one item a race, reward only, or from tier 2) |
| `fgself-182` (kept, level) | The guard-post fix | Games a little longer; creeps and anti-air up; the base trained under the fixed harness |
| `fgself-181` (not kept) | `--ai-share 0.5` (item 7) | Level; anti-air up, undead down |
| `fgself-180` (not kept) | `--z-share 0.25` (item 6) | Level; with a target 20% from 32%, narrower play |
| `fgself-179` (kept) | `--shop-items 1` (item 4) | Against insane 77% to 66% (and to `fgself-177`'s 67% elsewhere); buys 0.06 items a game; narrower play |
| `fgself-178` (not kept) | The map pool (item 3) | Off Echo Isles 46% against 46%; on Echo Isles longer games, 58% against 73%: it rushed less there and lost the long games |

Done: item 2, `fgself-177` (`fgself-171` in the league), kept: 58% head to head, insane level, its army mix not taken up. Item 1, `fgself-176` (`--ref-kl 0.01`), kept, level and slightly more varied. `fgself-175` (the hunt lesson) was not kept: ties down from 73% to 8%, but 63% to 72% against insane.

## Next (2026-11-03, after `fgself-187`)

Done since the list was made: items 1-4 and 6-8, the guard-post fix, `--z-shop 3`, no `natural`, uniform opponents (running). Left, in order:

1. **Bigger minibatch** (item 9: `--minibatch 1024 --lr 6e-5 --pad-groups 2`). Checks B (CPU) and C (`scripts/bench_update.py`, the GPU between runs) first. Could also speed up every later run.
2. **EMA anchor** (item 5, the user's idea). Built: `--ref-ema TAU` moves the anchor `TAU` of the way to the policy after each update (0.000115: a half-life of ~6,000 updates, ~50M steps at 8,192 a batch), saved with each checkpoint (`<steps>.ref`) and taken up on a resume; a fork starts it from `--init` again. Check A done: it can start from `fgself-100` (below).
3. **Shop targets, gentler** (the user's idea, after `fgself-183`): one item per race (salve, rod of necromancy, clarity), the reward only without the teacher's buy labels, or only from tier 2. Small code.
4. **The hunt lesson, gentler** (after `fgself-175`): only in games between agents, or less often. Small code.
5. **Expansions**: still none in real games, and the long games are lost on income. The expansion lesson (`fgself-156/157`) hurt; a new form is needed (for example a reward for a second mine's income, or `base2` starts at a higher share). Design first.
6. If play still does not move: league members that play differently (exploiters, members locked to one target).

Built 2026-11-05 (commit 69cb560): item 3's one-item pools (`*-shop1.json`: human clarity, orc salve, undead rod of necromancy, night elf staff of preservation; with K=1 the orb no longer takes the slot) and item 4's `--hunt-lesson 2` (the lesson only in games between agents). Item 6's exploiter runs first (`fgself-193`, no code).

Last: `--lr 6e-5` alone; `--z-modes 0` (needs code).

## Runs, in order

| # | Change | Code | Why | What would count |
|---|---|---|---|---|
| 1 | `--ref-kl 0.01` | none | Between the kept 0.025 and the 0 that lost: keep the macro, free some of `fgself-171`'s variety | Level or better, more orc and night elf variety, income holds |
| 2 | `fgself-171`'s last as a fixed league member (`--league-add`) | none | The league is one lineage; this shows the policy mixed orc armies and tier-3 night elves | Level or better, anti-air and army mix move |
| 3 | A map pool in training (`--map-pool`, Echo Isles at half the weight) | merged | The only test of whether Echo Isles taught it to rush: `fgself-168`'s 65% on Echo Isles and 60%, 40%, 38% elsewhere, with the same play, only show an Echo-trained policy's habits carrying over. May also add variety | Evaluated on all four real maps as well as `--eval-map`: play changes, gains elsewhere, Echo Isles level |
| 4 | `--shop-items 1` | merged | People buy items in 92-99% of games, the policy cannot buy from its shop | Items bought and used; if none, a lesson as its own run |
| 5 | An EMA anchor from `fgself-100` (half-life ~6,000 updates) | built (`--ref-ema 0.000115`) | The user's idea: stops sudden jumps without the permanent pull back. Needs check A | Variety rises by ~100M steps while income holds |
| 6 | `--z-share 0.25` | none | 67% of training steps carry a target, which wins 30-38% to ~70% without. How much strength does it cost? | Strength up without losing with-target play |
| 7 | `--ai-share 0.5` | none | More self-play at a person's income; the last tests were at doubled income | Head to head and variety |
| 8 | The value sees the tax, value only | built (`--value-tax 1`) | Neither network reads it, and `fgself-174` showed the tax has to stay (training at full income: 14% won, undead 3%) | Value loss down, level or better |
| 9 | `--minibatch 1024 --lr 6e-5 --pad-groups 2` | none | A minibatch carries ~1.4% of the policy's signal, and sorting by size makes it less mixed. Needs checks B and C | Faster, and level or better |

## If those do not move play

* The anchor on workers and halls only, army choice free (does not free tech: workers place the buildings).
* League members that play differently: exploiters (`fgself-76` level, at doubled income), or members locked to one target.

## Smaller tests, as a slot allows

* The hunt lesson, gentler: firing less often, or only in games between agents. `fgself-175` ended the ties and lost strength against insane.
* `--z-modes 0` (came in a bundle, never judged alone): needs code, it shrinks the input width (`fgself-185` could not start).
* The scenario `natural` replaced by normal starts or `base2` (the lessons it served are off).
* Uniform opponents in place of PFSP (near uniform already under `--win-balance`).
* `--lr 6e-5` alone: at a minibatch of 256 it mostly doubles the noise, and it has lost before, so last.

## Checks without a run

| | Check | Decides |
|---|---|---|
| A | Done (2026-11-04, `scripts/blend_check.py`): blends of `fgself-100` and `fgself-182`'s last at 0-100%, 12 games each against insane on the CPU: none broken, 1, 3, 6, 9, 10 of 12 won, the KL to both parents smooth ([the log](../reference/experiment-log.md)) | Run 5 starts from `fgself-100`, no fallback |
| B | How many games and game phases a minibatch holds at `--pad-groups` 8, 2, 1 (no GPU) | Whether sorting matters |
| C | `scripts/bench_update.py` at minibatches 256 and 1024, between runs (needs the GPU) | Whether run 8 is faster |
| D | Done: the side ahead camps at the dead main (`--hunt-lesson`, built; not yet trained) | Run 1 |

Nothing gets deleted meanwhile ([simplify](simplify.md)).
