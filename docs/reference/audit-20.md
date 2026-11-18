# Audit 20: where the base loses to insane (2026-11-18)

The data: every game against the untaxed insane AI in bursts 224, 226, 227, 230, 231 and 234 (`runs/burst-*/episodes-remote.jsonl`). That is 2,832 games by 24 policies, all close variants of the base: 2,275 won, 532 lost and 25 tied at 40 minutes. "Non-wins" below means losses and ties (557). Each burst reads alike (losses past 18 minutes 71-90%, orc 0.59-0.74), so the numbers are pooled. Checked against the training games of `fgself-231` to `-234` against the real insane (no target, no scenario) and the bursts' head-to-head games. Filmed training games (`runs/fgself-22x-23x/replays/*.trace.json`, 51 against the AI, with targets) were used only for hero levels and deaths.

## Findings, ranked by the non-wins they explain

**1. The first hero decides orc and human games (~270 of 557 non-wins).** Without a target the policy picks each of its race's three altar heroes about a third of the time, whatever the AI's race:

| learner | win rate by first hero (games) | gap |
|---|---|---|
| orc | **Far Seer 0.93 (186)**, Tauren Chieftain 0.62 (226), Blademaster 0.40 (201) | z = 7.3, 11.0 |
| human | **Archmage 0.95 (267)**, Mountain King 0.77 (233), Paladin 0.76 (262) | z = 5.7, 6.1 |
| night elf | Priestess of the Moon 0.92 (234), Demon Hunter 0.88 (217), Keeper 0.86 (223) | z = 1.7, 2.3 |
| undead | Lich 0.86 (261), Death Knight 0.86 (249), Dreadlord 0.82 (223) | n.s. |

* It holds in every burst (orc: Far Seer 0.87-1.00, Blademaster 0.26-0.56) and against every AI race (orc against night elf: Far Seer 51 of 54, Tauren Chieftain 21 of 61, Blademaster 20 of 62). It holds in training games (orc 0.91 / 0.62 / 0.40, human 0.91 / 0.79 / 0.75) and head to head against policies (orc 0.54 / 0.15 / 0.14).
* Far Seer and Archmage games end early: 72-79% are won within 15 minutes, against 19-59% with the other heroes. They also win more of the long games (0.57 against 0.21-0.32). A Blademaster game goes past 18 minutes 62% of the time, a Far Seer game 15%.
* If orc always took the Far Seer and human the Archmage, at the rates measured per matchup, there would be ~176 and ~92 fewer non-wins. The races weighed alike would read ~0.91 instead of ~0.80.
* Why it stays uniform: `--teach 3`'s first-hero lesson labels a uniform draw from `teacher.FIRST_HEROES` (`rng.choice`) in every training game without a target (~1.0 lesson a game, `--teach-coef 0.5`). The reward sees one hero decision a game, and the label comes every game. Targets already lean to these heroes (human target games: 1,274 Archmage, 401 Mountain King, 25 Paladin), but the evaluation plays without a target.

**2. 431 of the 532 losses come past 18 minutes, and the AI's development decides them.**

| game minutes | 0-12 | 12-15 | 15-18 | 18-24 | 24-30 | 30-41 |
|---|---|---|---|---|---|---|
| games won | 0.99 | 0.96 | 0.82 | 0.49 | 0.21 | 0.02 |
| non-wins (human / orc / undead / night elf) | 7 (3/4/0/0) | 24 (9/11/3/1) | 70 (18/27/17/8) | 196 (41/78/43/34) | 129 (30/49/28/22) | 131 (31/51/34/15) |

There were 681 games past 18 minutes, 225 of them won. The learner builds the same in the long games it wins and the ones it loses. Each of its build features has an AUC of 0.50-0.52 for a loss:

| at the game's end (won/lost: the learner's result) | learner, won | learner, lost | AI, learner won | AI, learner lost |
|---|---|---|---|---|
| tier 2 / tier 3 | 64% / 18% | 66% / 17% | 40% / 8% | 98% / 89% |
| heroes, research levels | 1.1, 2.0 | 1.1, 1.9 | 1.2, 2.2 | 2.6, 15.9 |
| air units, unit types | 0.0, 1.7 | 0.1, 1.7 | 0.1, 2.2 | 4.5, 7.5 |
| expanded, creep kills | 0%, 1.1 | 0%, 1.2 | 3%, 3.2 | 60%, 18.1 |
| workers trained | 8.6 | 9.5 | **27.0** | 14.4 |
| gold a minute | 531 | 463 | 841 | 924 |

* What predicts a loss is the AI's development. AUC for a single feature: AI research 0.96, AI creep kills 0.96, AI unit types 0.95, AI tier 0.93, AI heroes 0.92, AI air 0.92.
* A logistic regression (5-fold cross-validation) on the AI's development scores an AUC of 0.99. On the learner's own choices it scores 0.71. Game length alone scores 0.75, and the learner's choices with length 0.79.
* Over all games, once the AI has 8 or more research levels the learner won 0 of 433. With 4 or more AI air units it won 0 of 301, at AI tier 3 6% of 529, and after an AI expansion 3% of 312.
* Long wins are games where the AI's economy broke. The AI trained 27 workers (rebuilding them), against 14 in long losses. In games past 15 minutes, when the AI trained 20 or more workers the learner won 75-86%; with 12 or fewer, 9-14%.
* The learner's supply peaks at 52-57, and the AI's at 79 in long losses. The learner expanded in 3 of 2,832 games.

**3. Orc is the weakest race: 0.65 with the AI's races weighed alike, and 212 of the 532 losses (40%).** Orc against night elf is the worst cell, 0.52 with 85 losses.

| learner \ AI | human | orc | undead | night elf | race |
|---|---|---|---|---|---|
| human | 0.86 | 0.81 | 0.88 | 0.75 | 0.82 |
| orc | 0.70 | 0.70 | 0.68 | **0.52** | **0.65** |
| undead | 0.91 | 0.80 | 0.91 | 0.73 | 0.84 |
| night elf | 0.88 | 0.99 | 0.85 | 0.83 | 0.89 |
| AI race | 0.84 | 0.83 | 0.83 | **0.71** | |

* Most of orc's gap is its hero (finding 1). Orc games also go long more often: 42% past 18 minutes, against 15-24% for the other races. Those are the Blademaster's games.
* What it loses to is the developed night elf AI. In orc's losses against it the AI had dryads 5, mountain giants 3, chimaeras 2 and faerie dragons 2 a game, and in its wins only archers and huntresses. Across every race's long losses to that AI it also had 1.6 ancient protectors.
* Orc gets 15% of the training games (`--learner-races human:0.35,undead:0.35,orc:0.15,nightelf:0.15`). Those weights date from `fgself-67`, when orc and night elf won ~80% and human and undead ~25%.
* The night elf AI is the hardest opponent for every race: 204 non-wins (37%), and 0.14 of the long games against it won.

**4. Short losses are rare and not rushes.** There were 7 losses before 12 minutes and 31 before 15. In each one the learner's army was wiped in one or two fights (kills to losses ~6:50) by a tier-2 AI army: footmen, riflemen and mortars, or archers, huntresses, dryads and glaives. 25 of the 31 had a Blademaster, Tauren Chieftain, Mountain King or Paladin, and 23 were against the human or night elf AI.

**5. Waste and things that look like bugs.**
* The first-hero lesson's uniform draw (finding 1) and the stale race weights (finding 3).
* Night elf banks ~1,050 gold and lumber on average through its non-wins (470 in wins). It gathers 123 lumber a minute and spends little of it. Human banks 550 in its non-wins. The oversaturation lesson cost 18 points in `fgself-160`, so no lesson is proposed here.
* Heroes in the filmed training games against the AI: 164 deaths, 115 revivals. In losses the heroes lay dead for 4-40 hero-minutes (median 15). The learner's top hero was level 2.0 at 10 minutes in 20 losses (the AI's 2.5) and 3.5 in 26 wins (the AI's 2.0). Targets were on in those games.
* Orc: 217 games had 20 or more orders to peons inside burrows, and 5 of them were won (363 of 364 games with none were won). This marks a base under attack. The masks already stand burrows down after 20 safe steps.
* Refused orders that are left: night elf recharge ~34 a game, undead cannibalize ~15 a game. Small.

## Proposals

1. **The first hero** (finding 1: up to ~270 non-wins).
   * (a) A causal check without training. Run evaluation games with the first hero fixed: orc Far Seer, human Archmage. This needs a small eval-only mask at the altar, e.g. `--eval-first-hero orc:Ofar,human:Hamg`. Burst the base's average with and without it, on the same matchups. Expected: orc ≥0.88, human ≥0.92.
   * (b) If (a) holds, a fork from `fgself-233`'s average. The teacher's draw for games without a target becomes weighted by the measured rates, a one-line change at `teacher.py:257`, `rng.choice` to `rng.choices` with weights (e.g. Far Seer and Archmage 0.8, the rest 0.1). Or the lesson goes off in games without a target, so the reward decides. Targets keep their own heroes, which keeps the variety.
   * Judge (b) by a burst against 233's continuation, by race and by hero, plus head to head (the effect holds there too).
2. **Race weights** (finding 3). `--learner-races orc:0.3,human:0.25,undead:0.25,nightelf:0.2`, no code. Run it after 1, since orc with the Far Seer already reads 0.93. Judge orc's read over two bursts: ~30-35 orc games a policy a burst is too few.
3. **Pressure on the AI's economy at 10-18 minutes** (finding 2: 431 losses). Tech lessons moved the learner's build (`fgself-223`, `-226`, `-227`, `-230`), but its build does not separate long wins from losses. Whether the AI was kept from rebuilding does.
   * A reward, not a label (attack lessons spread: `fgself-218` to `-220`): the shaping's material counts the opponent's workers X times their cost. That is a small flag in `trace.material_units`, like `--hero-level-value`, e.g. X = 3.
   * Judge it by a burst against its control, reading the AI's workers trained a game, the share of games past 18 minutes and the long games won.
4. **Revival under the cap** (finding 5; weaker evidence: filmed target games only). `--revive-lesson 1` with a new `--teach-floor-orders` group for the revival orders. The uncapped form fired ~2.3 times a game and was part of `fgself-157`'s loss. It matters more once 1 makes the Far Seer or Archmage the army's core. Judge by the minutes heroes lie dead a game and a burst.

A reading aid, not an experiment: burst reads by race could also be given per first hero. A policy's orc read moves ~±0.1 with its hero mix at 30-35 games.
