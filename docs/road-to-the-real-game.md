# The road to the real game

The goal: a policy that plays a real 1v1 on a ladder map with the game's own rules. It plays the built-in AI first, then people.

```mermaid
flowchart LR
  A["duelrush<br/>79%"]:::done --> B["duelfast<br/>80% of games against insane"]:::done
  B --> D["Echo Isles, the real game<br/>97% against insane"]:::done
  D --> V["Strategic variety<br/>following a target build order"]:::now
  V --> P["People"]:::todo
  classDef done fill:#1d3a2a,stroke:#4cc38a,color:#d7dee6
  classDef now fill:#3a321d,stroke:#e0a33a,color:#d7dee6
  classDef todo fill:#1d252e,stroke:#4a5868,color:#8795a3
```

## Where it stands (2026-10-24)

The policy plays the real game on Echo Isles against the insane AI: the game's own rules, no handicap. It trains on `echofast` (Echo Isles with faster rules) toward target build orders, and is judged on the real game only. `fgself-139` wins **97%** of those games (368 evaluation games), and **65-68%** when it is given a target build order to play.

The steps there: a frame relative to the own base (0 → 49%), a longer horizon, a wider search for building spots, half the reward shaping, a quarter of the anchor's pull (82%), target build orders (89%), training on the fast rules with a builder that teaches the targets (95%), then six audits of the game's rules as the agent met them (the free first hero, purchases, revivals, undead's and night elves' expansions, tiers, refused orders) and orders of its own to expand, creep, train a target's army mix and pick up items. [Experiments](experiments.md) has each step.

## How it wins and loses

| `fgself-139`, real games | without a target | with a target |
|---|---|---|
| won | 97% (355 of 368) | 65% (210 of 324; ties 7%) |
| a win's median length | 9.0 minutes, 73% within 10 | 10.5 minutes |
| a loss's median length | 26 minutes | 27.5 minutes |
| tier 2 / tier 3 | 30% / 2% | 78% / 46% |
| expanded (a hall at a natural) | 1% | 24% |
| creep camps a game | 1.2 | 4.2 |
| heroes trained a game | 1.0 | 2.0 |
| items picked up a game | 0.4 | 1.9 |
| tavern heroes | 0% | 4% |

Without a target the policy rushes: one hero, tier 1, a win at 9 minutes. The targets make it play the rest of the game (tiers, expansions, creeping, more heroes and items), and win less. Of a target's due items it makes ~64%; the first hero is the target's only by chance (25-37%), the expansion is missed ~44% of the time for every race, undead's tier 2 and tavern heroes about half.

## What is missing

| area | gap | first step |
|---|---|---|
| Strategy | a target is followed ~64%, its first hero by chance; with a target it wins 65% against 97% without | the target's first hero taught at the altar (`fgself-140`, `--teach 2`), then the expansion and tier 2 |
| Seeing | no terrain or trees; the enemies it saw before did not help as inputs (`fgself-107`, `-117`) | the map's pathing as an input |
| Acting | everything the game's orders allow, plus four orders of its own (`--macros`: expand, creep, an army mix, pick up) and skill points spent for it; mercenaries off (`--neutral 2`: bought untaught, they cost wins) | the orders the game refuses are left out as they are found (audits) |
| Rewards | outcome, material, strategy and unspent money summed into one; rewards by part (GDPO) were worse | |
| APM | every unit, every half second | an APM cap before games against people |
| Data | the demonstrations come from the built-in AI. People's 1.29 replays play back in our client ([human replays](reference/human-replays.md)) | clone people on Echo Isles; their builds are the targets already |

## Next (2026-10-24)

The user's direction: strategic variety first (the agent must follow sensible build orders), then clearly beat the insane AI with them. In order: the target's first hero (`fgself-140`), the expansion and tier 2 of a target, then its wins with a target closer to its wins without. [Ideas](reference/ideas.md) has each idea: where it came from, what became of it.

## Compute

* ~1,700 agent steps per second in training (8 actors of 8 games and 8 evaluation games, one RTX 3090 and a Ryzen 9 5950X), ~3,000 games an hour; the games and the actors' Python set the speed, not the GPU.
* A laptop plays more evaluation games (~800 an hour), so a verdict (~300 games a policy) takes about an hour.
* A second machine for training games is ordered ([scaling up the CPUs](reference/scaling-cpus.md)).
