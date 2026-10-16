# The road to the real game

The goal: a policy that plays a real 1v1 on a ladder map with the game's own rules. It plays the built-in AI first, then people.

```mermaid
flowchart LR
  A["duelrush<br/>79%"]:::done --> B["duelfast<br/>80% of games against insane"]:::done
  B --> D["Echo Isles, the real game<br/>~65-70% against insane"]:::now
  D --> P["People"]:::todo
  classDef done fill:#1d3a2a,stroke:#4cc38a,color:#d7dee6
  classDef now fill:#3a321d,stroke:#e0a33a,color:#d7dee6
  classDef todo fill:#1d252e,stroke:#4a5868,color:#8795a3
```

## Where it stands (2026-10-16)

The policy plays the real game on Echo Isles: the game's own rules, both sides at full strength, against the insane AI with no handicap. It wins ~65-70% of those games.

How it got there:

* **The frame mattered most.** Each player sees positions relative to its own start, at the duel maps' scale. The stretch between the bases is shorter in the view, so the enemy base is where the duel maps had it. With this frame, the duelfast policy won 49% in the real game at once. Without it, it won none.
* **The rules mattered little.** The duelfast policy, trained on half costs and three times faster production, played the real rules nearly as well.
* **A clone of the AI's real games did not help.** It undid the policy's fighting and lost every game.
* **A longer horizon** (γ 0.9997) fits the 25-minute games: 55% → 59%.

## How it loses

* Long games: it builds many workers and cheap units and loses the late fights.
* Heroes: 0.6-1.0 a game against the AI's ~1.7. As human it builds its altar in two games of three.
* Races: orc and night elf win ~80%, human and undead ~25-45%. Many of undead's and night elf's builds were refused on Echo Isles' terrain (undead 25 of 100 in a test game, its ziggurats off the blight). A wider search for a free spot (up to 768 units) took 79% (2026-10-16). Against the AI's own record on Echo Isles (as undead it wins 88% of its mixed games, as human 14%) undead is the weak race: it gathers 83 lumber a minute against the AI's 213. Its ghouls are its lumber workers and its first fighters, and the policy fights with them.

## What is missing

| area | gap | first step |
|---|---|---|
| Heroes | fewer heroes than the AI, few levels | a reason to build them: shaping did not help yet |
| Map | no items, shops, tavern, mercenaries | item features and orders |
| Seeing | no terrain, trees or memory of buildings out of sight | a coarse map grid as input. The memory core is on. |
| Acting | every unit, every half second: no APM limit | an APM cap before games against people |
| Credit | 25-minute games are ~3,000 decisions a side | "when to act next" outputs |
| Data | all demonstrations come from the built-in AI | human replays, played back through the harness |

## Compute

* About 1,500-1,650 agent steps per second in the real game (32 threads, one RTX 3090).
* About 2,200 games an hour. A game lasts ~16 game minutes on average: wins ~12, losses ~25.
