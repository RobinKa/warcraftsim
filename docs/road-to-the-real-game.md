# The road to the real game

The goal: a policy that plays a real 1v1 on a ladder map with the game's own rules. It plays the built-in AI first, then people.

```mermaid
flowchart LR
  A["duelrush<br/>79%"]:::done --> B["duelfast<br/>80% of games against insane"]:::done
  B --> D["Echo Isles, the real game<br/>~81% against insane at a person's income"]:::now
  D --> G["The late game<br/>expansions, tech, mixed armies"]:::now
  G --> P["People"]:::todo
  classDef done fill:#1d3a2a,stroke:#4cc38a,color:#d7dee6
  classDef now fill:#3a321d,stroke:#e0a33a,color:#d7dee6
  classDef todo fill:#1d252e,stroke:#4a5868,color:#8795a3
```

## Where it stands (2026-11-12)

The policy plays the real game on Echo Isles against the insane AI: the game's own rules, no handicap, and the same income as a person. The base, `fgself-201`, wins **~81%** of those games (612 evaluation games), and **52%** when it is given a target build order to play. It can give every order a melee player can give, and it sees what a player sees ([order coverage](reference/order-coverage.md), [observation coverage](reference/observation-coverage.md)).

Until 2026-10-26 the agent slots had the insane AI's double income. The 95-97% of October were at that income.

## How it wins and loses

| `fgself-201`, real games against insane | without a target | with a target |
|---|---|---|
| won | 81% (612 games) | 52% (598 games) |
| a win's median length | 12.8 minutes | 15.2 minutes |
| a loss's median length | 23.2 minutes | 21.6 minutes |
| won when the game ends within 18 minutes | 96% | 79% |
| won past 24 minutes | 12% | 16% |
| tier 2 / tier 3 | 65% / 21% | 96% / 37% |
| expanded (a hall at a natural) | 0% | 17% |
| creep camps a game | 1.0 | 1.9 |
| heroes a game | 1.2 | 2.0 |

Without a target the policy plays one hero and masses one tier-1 unit. That wins most games before 18 minutes. Past 18 minutes insane has two bases, tier 3 and upgrades, and the policy's one mine runs dry at about 24 minutes. 102 of its 118 losses come past 18 minutes. In 47 of the 118 losses, insane had 4 or more air units. With a target the policy expands and techs, but it still loses most long games.

## What is missing

| area | gap | first step |
|---|---|---|
| Late game | 41% of the games past 18 minutes won | every training game in the real game (`fgself-212`), late starts from minutes 16-28 (`--late-from`), upgrades counted in the shaping (`--research-value`) |
| Economy | no expansion without a target, ~530 gold a minute | the late-game work above. Expansion lessons in every game pulled too hard (`fgself-202`) |
| Spells | most hero spells have ~0 chance in the policy | lessons made the heroes spam them (`fgself-208` to `-210`). A new approach is needed |
| Army | one tier-1 unit type, little anti-air | |
| APM | every unit, every half second | an APM cap before games against people |
| Data | the demonstrations come from the built-in AI. People's 1.29 replays play back in our client ([human replays](reference/human-replays.md)) | people's late-game states are late starts already |

## Next

The plan: [next experiments](proposals/next-experiments.md). [Ideas](reference/ideas.md) lists each idea, where it came from and what became of it.

## Compute

* ~5,000-5,800 agent steps a second in training: 72 games on the main machine (Ryzen 9 5950X, RTX 3090) and 64 on a games box. The GPU is now the limit.
* A laptop plays more evaluation games, so a verdict (~300 games a policy) takes about two hours of a run.
