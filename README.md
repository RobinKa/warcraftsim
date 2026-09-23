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
   sudo bash scripts/setup_system.sh        # WineHQ stable (11.0); staging measured slower here
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

* **`GameSetup`:** map, slots, `step_seconds`, `speed` (clock multiplier; adaptive by default), `max_game_seconds`, `fog`, `scenario`, `warm_spare` (melee: keep a loaded spare game so `restart()` takes about 1 s).
* **Slots:**
  * `Agent(race)`: controlled from Python.
  * `BuiltinAI(race, "easy"|"normal"|"insane")`: Blizzard's melee AI.
  * `Scripted(race)`: scenario opponent that attack-moves to the nearest enemy.
  * `Idle(race)`: no controller.
* **`Wc3Game`:**
  * `reset()` and `step(commands)`.
  * Queued order helpers: `move`, `attack`, `attack_move`, `smart`, `stop`, `hold`, `harvest`, `harvest_tree`, `train` / `research` / `upgrade`, `build`, `learn`, `cast`, `use_item`.
  * `cast()` accepts an order string (`"thunderbolt"`) or an ability code (`"AHtb"`). All 246 ability order strings in the game data are resolved in the first observation.
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
  * `MicroSelfPlayEnv`: two policies fight each other (PettingZoo-style dicts per player, zero-sum reward).
* **`warcraftsim.vec.Wc3VecEnv`:** N games in parallel with auto-reset.
* **Self-play in full games:** use two `Agent` slots. `game.as_player(1)` returns a handle whose helpers and queries act as player 1; orders from all handles go out with the next `step()`.

### Watching games

The game runs headless, so there are two ways to see what happened:

* **Trajectories (any game).** `warcraftsim.record.TrajectoryRecorder` saves the observation stream to a `.jsonl` file. `python -m warcraftsim view ep.jsonl` renders it as a self-contained HTML animation: units over the map's walkable area, with HP bars, player stats, play/pause and a time slider.
  `play` and `scenario` take `--record DIR`.
* **Replays and videos.** `GameInstance.save_replay(path)` ends the game normally and saves two files: the `.w3g` replay the engine writes, and `path.commands.json`, the agent orders per step.
  * Agent orders are issued by the map script, not through the engine's recorded command stream. So the stock 1.29 client shows the built-in AI's play correctly but not the agents'.
  * `GameInstance.play_replay(path)` plays the replay back through the harness and feeds the logged orders in at the same steps, reproducing the game exactly (verified step by step).
  * `warcraftsim.video.render_replay(setup, path, "ep.mp4")` records that playback as real game footage (MP4, one frame per step, optionally following a player's units).
* **Screenshots.** `instance.screenshot(path)` saves the virtual display. In scenarios the camera is centered on the action.

### Scenarios (fast RL iteration)

```python
from warcraftsim import GameSetup, Agent, Scripted, Scenario
Scenario.skirmish(["hfoo"] * 4, ["ogru"] * 3)            # last side standing wins
Scenario.move_to_target("hfoo", distance=1200)           # navigation, success judged in Python
Scenario(units=(SpawnSpec(0, "hfoo", -300, 0), ...), victory="elimination", max_game_seconds=60)
```

A scenario:
* removes every pre-placed unit;
* clears trees around its center;
* spawns its units.

The default scenario map is `"flat"`: a generated 32×32-tile (4096×4096) level plane with no water, cliffs or trees, centered at (0, 0), with the camera on the action.
* `"flatN"` gives an N×N-tile plane.
* `"flat:(2)EchoIsles"` gives a full-size flat version of a stock map.
* Any stock map name also works; its most open walkable spot becomes the center.

`reset()` re-spawns them inside the running game in about 10 ms, so there's no reload.

## Training with PufferLib 5.0 and the dashboard

PufferLib 5.0 (`third_party/PufferLib`, the current `5.0` branch) is a native CUDA trainer, and its environments are C code compiled into the `puffer` binary. The pieces:
* **C environment** (`puffer/wc3_bridge.h`): each environment is one Warcraft III game, reached through a bridge.
* **Bridge** (`warcraftsim/puffer/bridge.py`): a Python server that runs the games and serves them to the trainer over a Unix socket. Every trainer environment gets its own game thread.
* **Tasks** (`warcraftsim/puffer/tasks.py`): fixed-size observations and actions for PufferLib. The C header is generated from them.

```bash
# needs the CUDA toolkit (nvcc), clang, ccache, NCCL, libomp:
#   sudo apt-get install ccache libnccl2 libnccl-dev libomp-14-dev libomp5-14 libgl-dev libx11-dev
python -m warcraftsim.puffer.train --task nav --envs 16 --timesteps 400000        # navigation
python -m warcraftsim.puffer.train --task micro --envs 16 --timesteps 3000000     # 4 footmen vs 3 grunts
python -m warcraftsim dashboard                                                    # http://localhost:8765
```

Tasks: `nav` (reach a point), `micro` (4 footmen vs 3 scripted grunts), `micro_mirror` (4 vs 4 footmen). Add more in `tasks.py`.

Each run writes `runs/<name>/`, which the dashboard shows live:
* `run.json`: configuration and status.
* `train.jsonl`: one line per trainer epoch (SPS, losses, win rate), from a small patch applied to the build copy.
* `episodes.jsonl`: every finished episode.
* `renders/`: trajectory animations.
* `replays/` and `videos/`: single-episode replays rendered to real game footage in the background.
* `checkpoints/`.

The dashboard shows:
* the run list, with comparison;
* progress cards;
* win-rate, return, length, loss, entropy and throughput charts;
* a gallery of replay videos and trajectory renders;
* the recent episodes.

Example: `nav` with 16 games went from a 3% to a 62% success rate within 100k steps (about 3 minutes, at about 550 env steps/s).

## Performance (Ryzen 5950X, WSL2, 800x600 llvmpipe)

| workload | throughput |
|---|---|
| **Melee, AI vs AI, 2 s steps** | 39x real time (25-minute game in 39 s) |
| **Melee, agent vs AI, 0.25 s steps, full observations** | ≈17x per game |
| **Scenario skirmish, 0.25 s steps** | ≈25x per game; reset in 11 ms |
| **8 / 16 melee games in parallel, 0.25 s steps** | ≈80x / ≈105x real time combined (≈420 steps/s at 16) |
| **Game start** | ≈8 s (map load); 16 games ≈2 min (4 load at a time) |
| **Melee reset** | ≈1 s with the warm spare (≈8 s relaunch without one, or if the episode was shorter than a load) |

These numbers are for WineHQ **stable** 11.0, which is picked automatically from `/opt/wine-stable` (override with `WARCRAFTSIM_WINE`). Staging 11.18 was about 25% slower in parallel runs.

## Tests

```bash
.venv/bin/pytest                 # unit tests (need the game files, not Wine)
.venv/bin/pytest -m wine         # integration tests: launch real games
```

See `docs/architecture.md` for design notes, limits and what is known about the engine.
