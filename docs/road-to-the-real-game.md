# The road to the real game

The goal: a policy that plays a real 1v1 on a ladder map with the game's own rules. It plays the built-in AI first, then people.

```mermaid
flowchart LR
  A["duelrush<br/>79%"]:::done --> B["duelfast<br/>80% of games won against insane"]:::done
  B --> D["Echo Isles<br/>real map, game's rules"]:::now
  D --> P["People"]:::todo
  classDef done fill:#1d3a2a,stroke:#4cc38a,color:#d7dee6
  classDef now fill:#3a321d,stroke:#e0a33a,color:#d7dee6
  classDef todo fill:#1d252e,stroke:#4a5868,color:#8795a3
```

## Where it stands (2026-10-15)

On `duelfast` the policy beats the built-in insane AI without a handicap on either side beyond the map's own rules: 80% of its evaluation games against insane (random races), every matchup above the AI's own record (the mirrors: human 93%, orc 98%, night elf 100%, undead 62%; the hardest, undead against orc, 47% where the AI itself wins 23%). The user's next step: the real game on the real Echo Isles map.

What the move needs and has:

* **The map's frame.** Positions and point targets were in the duel map's frame (centre 0, half-width 3072). Each map has its own now (`features.map_geometry`): Echo Isles' centre (-256, -256), half-width 7168 (112 units a point bin). Its bases are left and right, as the view's mirroring expects.
* **The game's rules.** Costs follow the map (`costs.order_costs`); both sides at full strength (`--handicap 100`, duelfast's 50 halved both sides' hit points).
* **A clone of the real rules for the anchor.** Built-in AI games on Echo Isles (`runs/fullgame/demos-echo-1`: normal and insane, all races, ~600 games an hour, 18-35 game minutes each). An anchor of other rules kept their habits (`fgself-11` from `duelrush`).
* **A curriculum of rules on Echo Isles itself** (the user's suggestion): `echofast` (duelfast's rules in the real map), then `echoP`, P% of the way to the game's own rules.
* **The frame.** Positions relative to the player's own start at the duel maps' scale, the stretch between the bases squeezed to the duel maps' 4,600: the base areas and the enemy base look as on duelfast. Zero-shot under the real rules the duelfast policy won 0 of 58; on `echofast` scaled to the map's size 0 of 66 (builds in the trees), anchored 0 of 89 (attacks mid-map), anchored and squeezed 33 of 73 in its first minutes.
* **Not yet:** items, shops, mercenaries, the tavern (their units and orders are outside the vocabulary: unknown types, no orders to buy).

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
