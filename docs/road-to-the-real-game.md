# The road to the real game

The goal: a policy that plays a real 1v1 on a ladder map with the game's own rules. It plays the built-in AI first, then people.

```mermaid
flowchart LR
  A["duelrush<br/>79%"]:::done --> B["duelfast<br/>ties, banks money"]:::now
  B --> C["duel<br/>game's rules, small map"]:::todo
  C --> D["Echo Isles<br/>real map"]:::todo
  D --> P["People"]:::todo
  classDef done fill:#1d3a2a,stroke:#4cc38a,color:#d7dee6
  classDef now fill:#3a321d,stroke:#e0a33a,color:#d7dee6
  classDef todo fill:#1d252e,stroke:#4a5868,color:#8795a3
```

## What is missing

| area | gap | first step |
|---|---|---|
| Rules | each map change needs a new clone on the new rules | clone first, then RL with the KL term to that clone |
| Map | no expansions, items, shops, tavern, mercenaries | add them to a duel map, one at a time |
| Seeing | no terrain, trees or memory of buildings out of sight | a coarse map grid as input. The memory core is on. |
| Acting | every unit, every half second: no APM limit | an APM cap before games against people |
| Items | no inventory, no buying, no item use | item features and orders |
| Credit | 25-minute games are 3,000 decisions a side | longer horizons, "when to act next" outputs |
| Rare decisions | production and hero orders get noisy credit | separate credit for them, or a stronger cloning loss |
| Data | all demonstrations come from the built-in AI | human replays, played back through the harness |

## Compute

* About 850 agent steps per second on this machine (16 cores, one RTX 3090).
* A real 25-minute game is about 6,000 agent steps. That is about 500 games per hour.
