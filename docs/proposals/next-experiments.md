# Next experiments

**As of 2026-11-01.** The list from this week's findings: one mine caps income at half the insane AI's and ~79% of `fgself-168`'s losses come past 18 minutes ([the table](../reference/experiment-log.md)); the KL anchor holds the macro up but holds some variety back (`fgself-171`); and the policy cannot buy from its own shops. One change a run, ~3.5 hours each.

**Judging.** At ~50M steps and ~250 games a policy, from a run's second hour on, against its start: head to head, against insane by race, wins past 18 minutes, expansions, gold a minute, the top unit's share, with a target. Stop at ~30M steps if head to head with the small evaluation set is at or under ~45% and against insane is not above its start (`fgself-169` and `fgself-171` would have stopped, every kept run gone on; see [the log](../reference/experiment-log.md)); a variety run reads its behaviour from the second hour. The evaluation set is the current policy, with a target, and the start.

## Running

| Run | Change | Why |
|---|---|---|
| `fgself-181` | `--ai-share 0.5` (item 7), from `fgself-179` | More self-play at a person's income |
| `fgself-180` (not kept) | `--z-share 0.25` (item 6) | Level; with a target 20% from 32%, narrower play |
| `fgself-179` (kept) | `--shop-items 1` (item 4) | Against insane 77% to 66% (and to `fgself-177`'s 67% elsewhere); buys 0.06 items a game; narrower play |
| `fgself-178` (not kept) | The map pool (item 3) | Off Echo Isles 46% against 46%; on Echo Isles longer games, 58% against 73%: it rushed less there and lost the long games |

Done: item 2, `fgself-177` (`fgself-171` in the league), kept: 58% head to head, insane level, its army mix not taken up. Item 1, `fgself-176` (`--ref-kl 0.01`), kept, level and slightly more varied. `fgself-175` (the hunt lesson) was not kept: ties down from 73% to 8%, but 63% to 72% against insane.

## Runs, in order

| # | Change | Code | Why | What would count |
|---|---|---|---|---|
| 1 | `--ref-kl 0.01` | none | Between the kept 0.025 and the 0 that lost: keep the macro, free some of `fgself-171`'s variety | Level or better, more orc and night elf variety, income holds |
| 2 | `fgself-171`'s last as a fixed league member (`--league-add`) | none | The league is one lineage; this shows the policy mixed orc armies and tier-3 night elves | Level or better, anti-air and army mix move |
| 3 | A map pool in training (`--map-pool`, Echo Isles at half the weight) | merged | The only test of whether Echo Isles taught it to rush: `fgself-168`'s 65% on Echo Isles and 60%, 40%, 38% elsewhere, with the same play, only show an Echo-trained policy's habits carrying over. May also add variety | Evaluated on all four real maps as well as `--eval-map`: play changes, gains elsewhere, Echo Isles level |
| 4 | `--shop-items 1` | merged | People buy items in 92-99% of games, the policy cannot buy from its shop | Items bought and used; if none, a lesson as its own run |
| 5 | An EMA anchor from `fgself-100` (half-life ~6,000 updates) | small | The user's idea: stops sudden jumps without the permanent pull back. Needs check A | Variety rises by ~100M steps while income holds |
| 6 | `--z-share 0.25` | none | 67% of training steps carry a target, which wins 30-38% to ~70% without. How much strength does it cost? | Strength up without losing with-target play |
| 7 | `--ai-share 0.5` | none | More self-play at a person's income; the last tests were at doubled income | Head to head and variety |
| 8 | The value sees the tax, value only | small | Neither network reads it, and `fgself-174` showed the tax has to stay (training at full income: 14% won, undead 3%) | Value loss down, level or better |
| 9 | `--minibatch 1024 --lr 6e-5 --pad-groups 2` | none | A minibatch carries ~1.4% of the policy's signal, and sorting by size makes it less mixed. Needs checks B and C | Faster, and level or better |

## If those do not move play

* The anchor on workers and halls only, army choice free (does not free tech: workers place the buildings).
* League members that play differently: exploiters (`fgself-76` level, at doubled income), or members locked to one target.

## Smaller tests, as a slot allows

* The hunt lesson, gentler: firing less often, or only in games between agents. `fgself-175` ended the ties and lost strength against insane.
* `--z-modes 0` (came in a bundle, never judged alone).
* The scenario `natural` replaced by normal starts or `base2` (the lessons it served are off).
* Uniform opponents in place of PFSP (near uniform already under `--win-balance`).
* `--lr 6e-5` alone: at a minibatch of 256 it mostly doubles the noise, and it has lost before, so last.

## Checks without a run

| | Check | Decides |
|---|---|---|
| A | Blends of `fgself-100` and `fgself-173` at 25/50/75%: ~30 games each on the CPU, KL to both parents | Run 4 starts from `fgself-100` or needs the fallback |
| B | How many games and game phases a minibatch holds at `--pad-groups` 8, 2, 1 (no GPU) | Whether sorting matters |
| C | `scripts/bench_update.py` at minibatches 256 and 1024, between runs (needs the GPU) | Whether run 8 is faster |
| D | Done: the side ahead camps at the dead main (`--hunt-lesson`, built; not yet trained) | Run 1 |

Nothing gets deleted meanwhile ([simplify](simplify.md)).
