# Proposal: what to simplify or remove

**Question** (the user, 2026-11-01). Is there anything in the system we can simplify or remove that likely doesn't help?

**Answer.** Two lists. Features that are on but have no solid evidence get tested one at a time, because those could change results. Features that are off, with a negative or level verdict, can be deleted: that saves little at run time, and leaves less to audit. Nothing is deleted yet: two branches (shop items, map pool) are open on `selfplay.py`, and deleting first would make them conflict.

`fullgame/selfplay.py` has 5,255 lines and 206 flags; `fgself-173` runs with 53 of them at zero or empty.

## On, to test by turning down

| What | Why it is in doubt | Test |
|---|---|---|
| The AI's training tax (`--curriculum`, with `--real-share`) | Training games tax the AI's income and evaluation games don't, so training hides the income gap that loses the long games | Tax off, one run |
| The share of games with a target (`--z-share 0.75`) | 67% of `fgself-168`'s training steps carry a target. With a target it wins 30-38% against insane, without one ~70%, and without one it masses one unit whatever the run. It is the largest machinery (`strategy.py`, `modes.py`, the target builder, labels, army mix, half of each verdict's games), yet it is also the only variety we have, so reduce it rather than remove it | `--z-share 0.25`, one run |
| `--z-modes 0.15` (`modes.py`, 334 lines) | Came in as part of a bundle (`fgself-109`), never judged alone | Off, with or after the target test |
| The scenario `natural` | Clears the natural's camp so the expansion lessons apply, but the expand lesson and the expand order are both off now | Replace it with normal starts or `base2` |
| PFSP (`--pfsp hard`) | Near uniform under `--win-balance`: results are kept from ~6% of the past snapshots' games (2026-10-24) | Uniform draws, if the code gets simpler |

## Off, to delete

Each has a verdict in the [experiment log](../reference/experiment-log.md) or the [optimization log](../reference/optimization-log.md), or was declined:

* **Worse or no gain as runs:** GRPO (`grpo-*`, 96/97 level), OPSD (`opsd-*`, 112 no gain), the cloning and distillation losses in the learner (`bc-coef`, `opd-*`, 49 level without them; `--bc-data` is still passed at coefficient 0), the rolling anchor (`ref-update`, 82 worse), shaping decay and anneal (81 worse), hero-level shaping (`hero-level-value`, 64/65 worse), the worker cap (83 level), enforced targets (`z-enforce`, off since 127), people's plans as targets (`late-z`, 169 and 170 no gain).
* **Lessons at zero:** `army-`, `upgrade-`, `creep-`, `militia-`, `burrow-`, `saturation-`, `revive-`, `expand-lesson`, `spells-bias`. Keep the `--teach` mechanism: the shop items may need a lesson.
* **Measured no gain:** `update-grad-fuse`, `infer-past-period-ms`.
* **Approximations:** `shim/approx.c` (278 lines), declined by the user.
* **Worktrees:** about 25 agent worktrees (1.1 GB) with no unmerged commits, once the two open agents are done.

To check before deleting, since they may still be in use:

* The old micro track: `env.py` (990 lines), `puffer/`, `agents/micro.py`, `match.py`. So far only they and their tests were seen importing `env.py`; `overlay.py` also imports it, and the video code uses `overlay.py`.
* Scripts no doc or code names: `sweep_summary.py`, `episode_stats.py`, `rerender_video.py`, `rerender_fullgame.py`, `dashboard_shot.py`; and other scripts from September's micro work.
* Results not looked up: `reward-parts` (110 worse; 116's verdict unread), `unit-credit` (30), `fine-points`, `memory-fade`.
* Exploiters (`exploiter-*`, 76 level, in the doubled-income era): keep for now, as league members that play differently are the next idea if the tests above do not move play.

## Keep

Tested and earned: the KL anchor (`fgself-171` worse without it), shaping (81), the float penalty (71), late-game starts (166), UPGO, the games box, the GPU flags.

## Order

1. Merge the shop and map branches.
2. Delete the "off" list in one commit, after the checks above, with the tests passing and a run of the new code level with its start.
3. Run the tests above as one change a run: shop items first, then the tax, then the share of targets, then `--ai-share 0.5`, then `--lr 6e-5`.
4. Recipe files instead of 200 flags (roadmap item 5 of [five changes](five-changes.md)), once the list is shorter.
