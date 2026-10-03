# Environments

Every environment is the real game: Warcraft III 1.29 under Wine, one process per game, stepped by the shim (a virtual clock that stops the game at every step). Python sees the units, players and events the harness writes and sends orders the harness issues with the game's own natives.

## Micro: fights on a flat map

Scenarios on a generated flat map (`"flat"`, 32×32 tiles; `"flatN"`; or a flattened stock map): the harness removes everything, clears the trees and spawns the units; `reset()` respawns them in the running game in about 10 ms. Tasks (`warcraftsim/puffer/tasks.py`):

| task | what | note |
|---|---|---|
| `nav` | move a unit to a point | 3% → 62% success in 100k steps |
| `footmen2` | 2 v 2 footmen against the scripted opponent | 95% wins in ~1.5 minutes of training |
| `micro`, `micro_mirror` | 4 footmen against 3 grunts / 4 footmen | |
| `mirror_mix_*` | a random mirror composition each episode: a hero (level 1–3) and 2–4 units from all races | `_hp400` (units at 4× hit points) is the training setting; suffixes add semantic targets, hero abilities, relational features, tactics, self-play |
| `mirror_mix_gen*` | the same with general orders (kind, direction and distance, a pointer at any unit, an ability slot) | the entity network and league self-play (`--trainer torch`) |

Opponents: a scripted one (idle units attack-move to the nearest enemy; heroes cast what `MicroEnv.scripted_cast` picks), scripted policies (`noop`, `focus`, `pull35`, …), or the policy itself and its past snapshots.

## Whole games: the duel maps

Real melee games (economy, buildings, tech, heroes, armies, creeps) on small generated maps (`data/duelmap.py`): two bases like Echo Isles' main bases (a 12,500-gold mine ~720 from the start, a wall of trees behind, creep camps away from the bases), on flat land, no expansions, items or shops. The rules are the game's own unit, upgrade and ability tables with changed values, written into the map.

| map | size, bases | rules | built-in AI games |
|---|---|---|---|
| `duel` | 48×48 tiles, 4600 apart | the game's own | 12.4 minutes, 42% ties at 15 minutes; human 31%, orc 52%, undead 16%, night elf 9% |
| `duelfast` | 48×48, 4600 apart | hit points and costs halved; build, train and research times a third | 6.3 minutes on average (3000 games); night elf wins 71% against other races, orc 65%, undead 36%, human 26%; the insane AI beats the normal one 84% |
| `duelrush` | 40×40, 3000 apart | everything that takes time 7× faster (attacks, casts, cooldowns, production, day and night, the AI's waits); units move 1.3× faster (the engine's limit); hit points, costs and production times halved; twice the starting gold and lumber; mines 7× their gold; walking workers carry 6.5× the gold and 4× the lumber | 2.1 minutes; human 37%, orc 57%, undead 29%, night elf 78% |

Echo Isles and the other stock maps also run (`GameSetup(map="(2)EchoIsles")`); no policy has trained on them yet.

### Opponents in self-play

* **The built-in AI** (easy, normal, insane): Blizzard's melee AI scripts, with overriding copies on the rush map. It is the yardstick: "the real game" is a game against it with no handicap.
* **A curriculum** against it (`--curriculum`, a level per difficulty and matchup, moved towards a 50% score): `tax` mode takes up to 90% of what the AI gathers; `hp` gives the learner up to twice the hit points; `delay` starts the AI late. Ties can move the level too (`--curriculum-tie`).
* **Itself** (both sides train), **past snapshots** (prioritized fictitious self-play), and optionally a **main exploiter** (`--exploiter-share`).
* **Mixed matchups** (`--mirror 0`): any race against any; between agents the stronger race's income is taxed (`League.balance`), moved towards 50% by the learner's games against itself.
* **You** (`fullgame/versus.py`): the game in a window, you one player, the policy the other.

### What the agent sees

Every half second of game time (`step_seconds 0.5`), from its player's view (fog of war on; enemies only while visible; positions mirrored so its base is on the left):
* **Units** (at most 160: its own first, at most 96, then the visible enemy and neutral ones): 29 numbers each (own / enemy / neutral, position, hit points and mana with their maxima, 11 flags such as hero, structure, worker, constructing; hero level, gold left in a mine, skill points, facing, whether it has an order; for buildings what is queued and whether something is being made; for workers whether they cut lumber), plus its unit type and its current order (learned embeddings).
* **The player**: gold, lumber, supply used and cap, upkeep, game time, both races, its upgrade levels.

Not seen: terrain, pathing, trees (a lumber order targets a point and the nearest tree there is used), what the enemy holds, buildings out of sight (the game's "ghosts").

A memory core (`--memory 1`, a minGRU) can carry what it saw; it starts as a no-op.

### What the agent does

Each own unit, every step: an order class (none, or one of ~400: train, build, research, move, attack, harvest, cast a spell, learn a skill, …; only those its unit type gave in the demonstrations and that the player can pay for), then the order's target: a unit (a pointer over the units it sees) or a point (128 × 128 bins over the map). Workers on their way to build are left alone.

### Rewards

* +1 for a win, −1 for a loss; a tie at the time limit is worth `tie_break × tanh(2 × material lead)`.
* Shaping: the material lead (what living units and buildings cost, times their hit points left, mine minus the opponent's, over 2000), potential-based (it returns to 0 at the end, so it only moves credit earlier). `--float-penalty` counts held gold and lumber against a player.
* PPO per unit (each unit's decision its own clipped ratio, the step's advantage shared), a KL term towards the clone, and an auxiliary cloning loss on takeover games.
