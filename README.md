# warcraftsim

Headless Warcraft III for reinforcement learning: the **real game** (Legacy TFT 1.29.2, Blizzard's
offline client) under Wine in Ubuntu/WSL2, driven step by step from Python. There is no renderer
window, no Battle.net and no human input, and it runs much faster than real time.

```python
from warcraftsim import Wc3Game, GameSetup, Agent, BuiltinAI

with Wc3Game(GameSetup(map="(2)EchoIsles", slots=[Agent("human"), BuiltinAI("orc", "insane")])) as game:
    obs = game.reset()                       # game time 0
    while not obs.game_over:
        for worker in game.idle_workers():
            game.harvest(worker, game.nearest_mine(worker))
        obs = game.step()                    # 0.25 s of game time
    print(obs.players[0].result)
```

## How it works

```
Python (warcraftsim)                 Wine prefix per game                 Warcraft III 1.29 process
------------------------------       -------------------------------      -------------------------------------
Wc3Game / Wc3Env / Wc3VecEnv          .wgc game config (slots, AI)  --->   map = stock map + injected JASS harness
GameInstance ── TCP 127.0.0.1 ──────────────────────────────────────────── w3shim.dll (injected by w3launch.exe)
     │                                                                       - virtual clock (game runs N x faster)
     │  obs.txt / act.txt in /dev/shm  (CustomMapData\w3sim)                  - blocks the game at every step
     └──────────────────────────────────────────────────────────────────── harness: Preload() observations,
                                                                              Preloader() + natives for commands
```

* **Harness** (`warcraftsim/harness/w3sim.j`): JASS injected into the map script by
  `data/mapbuild.py`. On every step (a game-time timer) it:
  1. writes an observation: players, changed units, events and command results;
  2. reads the next commands (`Preloader`);
  3. issues them with ordinary order natives (`IssuePointOrderById`, ...).

  Its other jobs:
  * It suppresses the built-in AI for agent slots.
  * It decides wins and losses without ending the session.
  * It runs scenarios.
* **w3shim.dll** (`shim/`, mingw, 32-bit). It is injected into the game by `w3launch.exe`, and all its hooks are in the game's import table or a few engine functions:
  * **Virtual clock:** replaces the game's timers (QPC, GetTickCount, FILETIME, rdtsc helper) and scales its waits, so the game runs as fast as the CPU allows.
  * **Step sync:** when the harness opens its action file, the DLL freezes the clock, reports "OBS" over TCP and waits for "GO". That makes stepping exactly synchronous and deterministic in game time.
* **`.wgc` game configs** start a local game directly, with no menus or Battle.net. They set the map, the slots (agents are computer slots with no AI; built-in AIs get easy/normal/insane) and an observer as the local player.

## Setup (once)

1. **Install Warcraft III - Legacy TFT 1.29** from the Battle.net app (Warcraft III → Game Version dropdown), then copy it into WSL:
   ```bash
   rsync -a --exclude .battle.net --exclude Data "/mnt/d/Warcraft III (Legacy)/" ~/wc3/legacy-1.29/
   ```
   To use another location, set `WC3_GAME_DIR`.
2. **Install system packages** (WineHQ, Xvfb, mingw, ...):
   ```bash
   sudo bash scripts/setup_system.sh
   sudo apt-get install -y winehq-staging   # optional; the tests ran on 11.18 staging
   ```
3. **Build the native helpers and create the Python venv:**
   ```bash
   git submodule update --init
   python3 -m venv .venv && .venv/bin/pip install cmake pytest numpy "gymnasium>=1.0" && .venv/bin/pip install -e .
   scripts/build_native.sh     # StormLib (MPQ), pjass (JASS checker)
   make -C shim                # w3shim.dll, w3launch.exe
   .venv/bin/python -m warcraftsim setup
   ```

## Usage

| | |
|---|---|
| `python -m warcraftsim ai-vs-ai --difficulty insane` | Two built-in AIs play a full game headless. |
| `python -m warcraftsim play --difficulty easy` | The scripted Human bot plays the built-in AI. |
| `python -m warcraftsim scenario` | Micro skirmish against the scripted opponent. |
| `python -m warcraftsim bench -n 8` | Parallel throughput. |

### Python API

* **`GameSetup`:** map, slots, `step_seconds`, `speed` (clock multiplier), `max_game_seconds`, `fog`, `scenario`.
* **Slots:**
  * `Agent(race)`: controlled from Python.
  * `BuiltinAI(race, "easy"|"normal"|"insane")`: Blizzard's melee AI.
  * `Scripted(race)`: scenario opponent that attack-moves to the nearest enemy.
  * `Idle(race)`: no controller.
* **`Wc3Game`:**
  * `reset()` and `step(commands)`.
  * Queued order helpers: `move`, `attack`, `attack_move`, `smart`, `stop`, `hold`, `harvest`, `harvest_tree`, `train` / `research` / `upgrade`, `build`, `learn`, `cast`, `use_item`.
  * Debug commands: `spawn`, `set_resources`.
  * Queries: `my_units`, `idle_workers`, `mines`, `enemies`.
* **`Observation`:**
  * `players` (gold, lumber, food, upkeep, result, ...).
  * `units`: every unit with id, type, owner, position, facing, hp/mana, current order, flags (hero, structure, worker, constructing, hidden, ...) and per-agent visibility.
  * `events`: deaths, training, research, construction, spells, tree deaths.
  * `command_results`.
  * On the first observation only: `destructables` and the `orders` table.
* **Gymnasium** (`warcraftsim.env`):
  * `Wc3Env`: feature arrays; actions are lists of commands.
  * `MicroEnv`: scenario fights, with a MultiDiscrete action per unit.
  * `NavigateEnv`: move a unit to a target.
* **`warcraftsim.vec.Wc3VecEnv`:** N games in parallel with auto-reset.

### Scenarios (fast RL iteration)

```python
from warcraftsim import GameSetup, Agent, Scripted, Scenario
Scenario.skirmish(["hfoo"] * 4, ["ogru"] * 3)            # last side standing wins
Scenario.move_to_target("hfoo", distance=1200)           # navigation, success judged in Python
Scenario(units=(SpawnSpec(0, "hfoo", -300, 0), ...), victory="elimination", max_game_seconds=60)
```

A scenario:
* removes every pre-placed unit;
* clears trees around its center (by default the most open walkable spot of the map);
* spawns its units.

`reset()` re-spawns them inside the running game in about 10 ms, so there's no reload.

## Performance (Ryzen 5950X, WSL2, 800x600 llvmpipe)

| workload | throughput |
|---|---|
| **Melee, AI vs AI, 2 s steps** | 39x real time (25-minute game in 39 s) |
| **Melee, agent vs AI, 0.25 s steps, full observations** | ≈18x per game |
| **Scenario skirmish, 0.25 s steps** | ≈25x per game; reset in 11 ms |
| **16 melee games in parallel, 0.25 s steps** | ≈100x real time combined (≈400 steps/s) |
| **Game start** | ≈8 s (map load) |
| **Melee reset** | Relaunches the process, about 8 s; see docs/architecture.md |

## Tests

```bash
.venv/bin/pytest                 # unit tests (need the game files, not Wine)
.venv/bin/pytest -m wine         # integration tests: launch real games
```

See `docs/architecture.md` for design notes, limits and what is known about the engine.
