# Next experiments

**As of 2026-11-01.** The list from this week's findings: one mine caps income at half the insane AI's and ~79% of `fgself-168`'s losses come past 18 minutes ([the table](../reference/experiment-log.md)); the KL anchor holds the macro up but holds some variety back (`fgself-171`); and the policy cannot buy from its own shops. One change a run, ~3.5 hours each.

**Judging.** At ~50M steps and ~250 games a policy, from a run's second hour on, against its start: head to head, against insane by race, wins past 18 minutes, expansions, gold a minute, the top unit's share, with a target. Stop at ~30M steps if head to head with the small evaluation set is at or under ~45% and against insane is not above its start (`fgself-169` and `fgself-171` would have stopped, every kept run gone on; see [the log](../reference/experiment-log.md)); a variety run reads its behaviour from the second hour. The evaluation set is the current policy, with a target, and the start.

## Running

| Run | Change | Why |
|---|---|---|
| `fgself-174` | The AI's training tax off (`--curriculum -1`), from `fgself-173` | Training hid the income gap that evaluation has |
| Agent | Other maps measured, then a map pool behind a flag | Does rushing depend on the map? |

## Runs, in order

| # | Change | Code | Why | What would count |
|---|---|---|---|---|
| 1 | `--ref-kl 0.01` | none | Between the kept 0.025 and the 0 that lost: keep the macro, free some of `fgself-171`'s variety | Level or better, more orc and night elf variety, income holds |
| 2 | `fgself-171`'s last as a fixed league member (`--league-add`) | none | The league is one lineage; this shows the policy mixed orc armies and tier-3 night elves | Level or better, anti-air and army mix move |
| 3 | `--shop-items 1` | merged | People buy items in 92-99% of games, the policy cannot buy from its shop | Items bought and used; if none, a lesson as its own run |
| 4 | An EMA anchor from `fgself-100` (half-life ~6,000 updates) | small | The user's idea: stops sudden jumps without the permanent pull back. Needs check A | Variety rises by ~100M steps while income holds |
| 5 | `--z-share 0.25` | none | 67% of training steps carry a target, which wins 30-38% to ~70% without. How much strength does it cost? | Strength up without losing with-target play |
| 6 | `--ai-share 0.5` | none | More self-play at a person's income; the last tests were at doubled income | Head to head and variety |
| 7 | The value sees the tax, value only | small | Neither network reads it. The size of this depends on `fgself-174`: the AI's if it comes back, else the tax between agents | Value loss down, level or better |
| 8 | `--minibatch 1024 --lr 6e-5 --pad-groups 2` | none | A minibatch carries ~1.4% of the policy's signal, and sorting by size makes it less mixed. Needs checks B and C | Faster, and level or better |
| 9 | A map pool in training | agent | Only if the measurement says rushing depends on the map, or for the replays | Variety on Echo Isles, level or better |

## If those do not move play

* The anchor on workers and halls only, army choice free (does not free tech: workers place the buildings).
* League members that play differently: exploiters (`fgself-76` level, at doubled income), or members locked to one target.

## Smaller tests, as a slot allows

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
| D | Why `base2` games between agents tie at 40 minutes (73%, 1,443 of 1,746 with the leader far ahead; the user saw undead leave human's natural alone after destroying its main): an agent is measuring it, and building a lesson behind a flag if the side ahead doesn't know where the base is | A bug, or a weakness to teach |

Nothing gets deleted meanwhile ([simplify](simplify.md)).
