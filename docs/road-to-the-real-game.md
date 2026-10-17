# The road to the real game

The goal: a policy that plays a real 1v1 on a ladder map with the game's own rules. It plays the built-in AI first, then people.

```mermaid
flowchart LR
  A["duelrush<br/>79%"]:::done --> B["duelfast<br/>80% of games against insane"]:::done
  B --> D["Echo Isles, the real game<br/>72% against insane"]:::now
  D --> P["People"]:::todo
  classDef done fill:#1d3a2a,stroke:#4cc38a,color:#d7dee6
  classDef now fill:#3a321d,stroke:#e0a33a,color:#d7dee6
  classDef todo fill:#1d252e,stroke:#4a5868,color:#8795a3
```

## Where it stands (2026-10-17)

The policy plays the real game on Echo Isles against the insane AI: the game's own rules, no handicap. The best checkpoint (`fgself-73`'s last) won 72% of those games in its run, and ~77% in `fgself-75`'s.

The frame mattered most: positions relative to the own start, at the duel maps' scale. With it, the duelfast policy won 49% at once. Without it, it won none. A clone of the AI's real games hurt. Then came small steps: a longer horizon, a wider search for building spots and half the reward shaping (62% → 72%). [Experiments](experiments.md) has each step.

## How it wins and loses

* **Wins:** one strategy, an early rush of tier-1 units. The median win comes at ~9.7 game minutes. 206 of 356 wins came within 10 minutes, and only 38% reached tier 2.
* **Losses:** long games (~24 minutes) with tier-1 units and many workers (~56 against the AI's ~14).
* **Heroes:** ~0.9 a game against the AI's ~1.7.
* **Undead** gathers 83 lumber a minute against the AI's 213. Its ghouls are its lumber workers, and the policy fights with them.

## What is missing

| area | gap | first step |
|---|---|---|
| Heroes | fewer heroes than the AI, few levels | a reason to build them: shaping did not help yet |
| Strategy | one strategy: the tier-1 rush, little tech | exploiters, then a target build order for each game |
| Map | no items, shops, tavern, mercenaries | item features and orders |
| Seeing | no terrain, trees or memory of buildings out of sight | a coarse map grid as input. The memory core is on. |
| Acting | every unit, every half second: no APM limit | an APM cap before games against people |
| Credit | 25-minute games are ~3,000 decisions a side | "when to act next" outputs |
| Data | the demonstrations come from the built-in AI. People's 1.29 replays play back in our client in step ([human replays](reference/human-replays.md)): all 95 stock-map Echo Isles 1v1s converted (24 hours of play); 2,591 replays parsed for build orders | clone people on Echo Isles; strategy statistics (build orders, unit mixes) for conditioning |

## Next (2026-10-17)

1. **Finish `fgself-75`** (a quarter of the shaping). After 30 minutes it is level: 74% against 77%, ~120 games each. The verdict comes at ~300 games each.
2. **An exploiter league.** AlphaStar's exploiters train only against the main agent and find its weak points. Here they can learn to stop the early rush, so the main policy must learn more. A main exploiter is in `selfplay.py` (`--exploiter-share`), not yet tested on Echo Isles.
3. **A strategy for each game**, as AlphaStar's z: a target build order and unit mix from the AI's Echo Isles games, later from human replays. A small reward for following it should break the single rush and force tech.
4. **Human replays**: ~3,000 replays of 1.29 from warcraft3.info. A converter reads their build orders (any game version) and replays 1.29 games in the real game to make demonstrations ([human replays](reference/human-replays.md)).

## Compute

* ~1,600 agent steps per second in the real game (`fgself-75`: 1,614): 8 actors of 6 games and 8 evaluation games, one RTX 3090.
* ~3,400 games an hour, ~15 game minutes each on average.
