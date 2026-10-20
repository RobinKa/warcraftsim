# The road to the real game

The goal: a policy that plays a real 1v1 on a ladder map with the game's own rules. It plays the built-in AI first, then people.

```mermaid
flowchart LR
  A["duelrush<br/>79%"]:::done --> B["duelfast<br/>80% of games against insane"]:::done
  B --> D["Echo Isles, the real game<br/>~95% against insane"]:::done
  D --> V["Strategic variety<br/>and the whole game's actions"]:::now
  V --> P["People"]:::todo
  classDef done fill:#1d3a2a,stroke:#4cc38a,color:#d7dee6
  classDef now fill:#3a321d,stroke:#e0a33a,color:#d7dee6
  classDef todo fill:#1d252e,stroke:#4a5868,color:#8795a3
```

## Where it stands (2026-10-20)

The policy plays the real game on Echo Isles against the insane AI: the game's own rules, no handicap. The best line (`fgself-100` to `fgself-106`) wins **~95%** of those games: orc 98%, night elf 100%, undead 94%, human 89% (1,045 evaluation games). It trains on `echofast` (Echo Isles with faster rules) toward target build orders, and is judged on the real game only.

The steps there: a frame relative to the own base (0 → 49%), a longer horizon, a wider search for building spots, half the reward shaping, a quarter of the anchor's pull (82%), target build orders (89%), training on the fast rules with a builder that teaches the targets (95%), more games as the weaker races (undead 79% → 89%). [Experiments](experiments.md) has each step.

## How it wins and loses

* **Wins** come early: the median at 8.8 game minutes, 76% within 10. It reaches tier 2 in 59% of its games.
* **Losses** are rare (43 of 1,045) and long: the median at 28 minutes.
* **Heroes:** 0.8 alive at the end of a game, the AI 1.1. **Workers:** 12 against the AI's 17.
* **With a target build order** it wins ~82-91% and makes most of the target's items on its own (~5 of the due items missed).

## What is missing

| area | gap | first step |
|---|---|---|
| Strategy | the targets are whole games' builds, big and alike; the policy prefers one rush | narrow targets (a game's heroes, main army and what they need: `strategy.narrow`), more kinds of conditioning (no rush, expand, creep, harass...: `fullgame/modes.py`) |
| Seeing | no terrain or trees; until `fgself-107` no heroes' abilities, cooldowns or items, and nothing out of sight | `--inputs 3`: hero state, what buildings make, the clock, the enemies it saw (`fgself-107`) |
| Acting | no tavern heroes, neutral shops, mercenaries or items on the ground; orders the game refuses waste decisions | a `Buy` command (in the harness), new order classes and a teacher for them; `--tech-mask`, `--valid-mask` |
| Teaching | what the policy rarely stumbles on (a tavern hero, a town portal) | a heuristic teacher whose orders label the policy's steps (as the target builder does now) |
| Rewards | outcome, material, strategy and unspent money summed into one | an advantage per reward, normalized (GDPO), weighted by how much each still varies (SAW) |
| APM | every unit, every half second | an APM cap before games against people |
| Data | the demonstrations come from the built-in AI. People's 1.29 replays play back in our client ([human replays](reference/human-replays.md)) | clone people on Echo Isles; their builds are the targets already |

## Next (2026-10-20)

The user's direction: strategic variety first, then the whole game's actions. In order: `fgself-107` (the new inputs and masks, running), narrow targets, more kinds of conditioning, the rewards' combination, the neutral buildings with a teacher, then scenarios (mid-game situations the harness sets up, specialists merged by weight averaging). [Ideas from AlphaStar and OpenAI Five](reference/ideas-from-alphastar-and-openai-five.md) has each.

## Compute

* ~1,500-1,800 agent steps per second in training (8 actors of 6 games and 8 evaluation games, one RTX 3090), ~3,000 games an hour.
* A laptop plays more evaluation games (~1,300 an hour), so a verdict takes an hour or two.
