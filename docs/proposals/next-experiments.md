# Next experiments

**As of 2026-11-01.** The list from this week's findings: one mine caps income at half the insane AI's and ~79% of `fgself-168`'s losses come past 18 minutes ([the table](../reference/experiment-log.md)); the KL anchor holds the macro up but holds some variety back (`fgself-171`); and the policy cannot buy from its own shops. One change a run, ~3.5 hours each.

**Judging.** At ~50M steps and ~250 games a policy, from a run's second hour on, against its start: head to head, against insane by race, wins past 18 minutes, expansions, gold a minute, the top unit's share, with a target. Stop at ~30M steps if head to head with the small evaluation set is at or under ~45% and against insane is not above its start (`fgself-169` and `fgself-171` would have stopped, every kept run gone on; see [the log](../reference/experiment-log.md)); a variety run reads its behaviour from the second hour. The evaluation set is the current policy, with a target, and the start.

## Running

| Run | Change | Why |
|---|---|---|
| `fgself-193` (running) | `--exploiter-share 0.2` (item 6, brought forward) | A main exploiter looks for what beats the policy; a league member that plays differently |
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
