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
     │  observations and commands travel with the step sync:              - virtual clock (game runs N x faster)
     │  "OBS n len" + tokens  ->   <- "GO ... A n ints"                    - blocks the game at every step
     └──────────────────────────────────────────────────────────────────── harness: Preload() observation tokens
                                                                              (captured by the shim), mailbox native
                                                                              for commands
```

* **Harness** (`warcraftsim/harness/w3sim.j`): JASS injected into the map script by
  `data/mapbuild.py`. On every step (a game-time timer) it:
  1. writes an observation: players, changed units, events and command results;
  2. reads the next commands through a mailbox native the shim answers;
  3. issues them with ordinary order natives (`IssuePointOrderById`, ...).

  Its other jobs:
  * It suppresses the built-in AI for agent slots.
  * It decides wins and losses without ending the session.
  * It runs scenarios.
* **w3shim.dll** (`shim/`, mingw, 32-bit). It is injected into the game by `w3launch.exe`, and all its hooks are in the game's import table or a few engine functions:
  * **Virtual clock:** replaces the game's timers (QPC, GetTickCount, FILETIME, rdtsc helper) and scales its waits, so the game runs as fast as the CPU allows.
  * **Step sync:** when the harness calls its mailbox native, the DLL freezes the clock, sends the observation over TCP ("OBS") and waits for "GO" with the next commands. That makes stepping exactly synchronous and deterministic in game time.
  * **Frame capture and audio** for videos (see Watching games).
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
  * `warcraftsim.video.render_replay(setup, path, "ep.mp4")` records that playback as real game footage with the game's sound: MP4 at 40 fps in real time by default, optionally following a player's units.
    * The shim switches the game clock to frame-stepped mode, so every rendered frame advances the game by exactly 25 ms, one engine turn.
    * Each frame is grabbed from the virtual display before the game continues.
    * The result is smooth and independent of machine load. A 60 s episode renders in about 60–75 s next to a training run.
    * **Audio** comes from a virtual sound card in the shim (`shim/audio.c`, `GameSetup.audio`):
      * The game mixes its sound with Miles, which outputs through `waveOut`; the shim implements those functions.
      * Each captured frame takes exactly one frame's worth (25 ms) of mixed audio from Miles's queue, so sound follows the picture however fast the render runs.
      * Miles mixes ahead of playback. The shim lowers its buffering (preferences 11 and 45: about 55 ms instead of about 180 ms) and reports the remaining queue; the soundtrack is shifted earlier by that amount.
      * Audio renders run at no more than real time, because Miles's mixer runs on real time. If it falls behind, the gap is padded with silence and the late audio is dropped, so the soundtrack stays in sync.
      * `music_volume` sets the music level (default 40, 0 for none). Training games have no sound device.
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
python -m warcraftsim.puffer.train --task nav --timesteps 400000                  # navigation
python -m warcraftsim.puffer.train --task footmen2 --step-seconds 0.5 --timesteps 400000 \
    --lr 0.01 --minibatch 192 --replay-ratio 4                                     # 2 v 2 footmen: 95% wins in ~1.5 min
python -m warcraftsim.puffer.train --task micro_mirror --timesteps 3000000        # 4 v 4 footmen
python -m warcraftsim.puffer.train --task selfplay_micro --envs 8 --timesteps 3000000 # both sides learn
# a sweep: the games launch once, then one run per --sweep (runs NAME-1, NAME-2, ... in the dashboard)
python -m warcraftsim.puffer.train --task footmen2 --name f2 --timesteps 1e6 --sweep "--lr 0.01" --sweep "--lr 0.02"
python -m warcraftsim dashboard                                                    # http://localhost:8765
```

Tasks (add more in `tasks.py`; `scripts/baselines.py` measures scripted policies on any of them):
* `nav`: reach a point.
* `footmen2`: 2 vs 2 footmen with 100 hit points against the scripted opponent (episodes ~17 s).
  * Scripted baselines win 0% (random), 30% (noop), 65–70% (focus fire), 90% (focus fire, and pulling a footman back while it is low and being hit).
  * Tuned by sweeps (`f2-sweep1`..`5`, `f2-step*` in the dashboard): 0.5 s steps, lr 0.01, minibatch 192, replay ratio 4, the learning rate annealed over 400k steps → 95% wins after ~0.2M steps (~1.5 min), 98-100% soon after; 1.0 s steps: 100% for both seeds.
  * Most of the speed came from more updates per sample (1 → 32 per epoch), then from longer steps and a shorter annealing schedule. Horizon 32 learns fastest at first but ends lower.
  * Early policies learned focus fire plus pulling a hurt footman back. The 100% policy instead holds position until the enemies arrive: the scripted opponent then splits its damage over both footmen, while ours focus one enemy.
* `footmen<N>v<M>[_hp<HP>][_ehp<EHP>]`: N agent footmen against M scripted ones with HP hit points each (default 100), the enemies EHP (a handicap).
* `micro`: 4 footmen vs 3 scripted grunts. This is hard: scripted baselines win about 1 game in 3.
* `micro_mirror`: 4 vs 4 footmen against the scripted opponent.
* `selfplay_micro`: 4 vs 4 footmen with both sides served to the trainer as agents of the same policy. `--envs` counts games, so each game gives two agents. The dashboard's win rate is side 0's.

Notes:
* Updates per epoch are `replay_ratio × batch / minibatch`. With the minibatch equal to the batch (the old default), there was one update per epoch and learning was slow: `footmen2` reached 61% wins in 1M steps, against 97% with 16 updates.
* PufferLib 5.0 does not normalize advantages, and the micro rewards per step are small. Two settings keep the entropy bonus of the 18 action heads from outweighing the reward and pushing the policy to uniform:
  * `--ent-coef` defaults to 0.001;
  * micro tasks scale rewards by 10 (`Task.reward_scale`; logged returns are scaled too).
* Runs can train concurrently. Each run claims a machine-wide training slot, and its games are named after that slot, so consecutive runs reuse their Wine prefixes.

Each run writes `runs/<name>/`, which the dashboard shows live:
* `run.json`: configuration and status.
* `train.jsonl`: one line per trainer epoch (SPS, losses, win rate), from a small patch applied to the build copy.
* `episodes.jsonl`: every finished episode.
* `renders/`: trajectory animations.
* `replays/` and `videos/`: single-episode replays rendered to real game footage in the background (40 fps).
  * The footage shows which units are the agent's (rings and slot labels A0, A1, ...; enemies E0, ...) and each step's orders: move arrows, attack lines with a crosshair on the target, stop markers.
  * A side panel shows what the policy thought. It evaluates the latest checkpoint before the episode (`puffer/policy.py`, numpy) on the observations the agent saw, and plots:
    * the value V(s) against the discounted return that actually followed;
    * reward and TD error per step;
    * team hit points;
    * per unit, the action probabilities and the sampled action;
    * the policy entropy;
    * an outcome card at the end.
  * In self-play both sides are shown.
  * `scripts/calibrate_camera.py` measures the camera projection the drawing uses.
* `checkpoints/`.

The dashboard shows:
* the run list, with comparison;
* progress cards;
* **Outcomes**: win rate, win/draw/loss, return, episode length;
* **Behaviour** (from the actions the policy sent):
  * the action mix (noop/stop/move/attack);
  * targeting: focus fire, attacks on the weakest enemy, invalid targets;
  * damage dealt and taken, kills and losses;
* **Learning**:
  * value calibration: predicted V(s₀) against the actual return of video episodes;
  * PPO losses, entropy and clip fraction;
* **System**: throughput, and the trainer's time per epoch split into rollout (waiting for games, model) and training;
* a gallery of replay videos and trajectory renders;
* the recent episodes with their combat and action statistics.

Example: `nav` with 16 games went from a 3% to a 62% success rate within 100k steps (about 3 minutes, at about 550 env steps/s).

## Performance (Ryzen 5950X, 32 threads, WSL2, llvmpipe)

| workload | throughput |
|---|---|
| **Melee, AI vs AI, 2 s steps** | 39x real time (25-minute game in 39 s) |
| **Melee, agent vs AI, 0.25 s steps, full observations** | ≈35x per game (≈140 steps/s) |
| **16 melee games in parallel, 0.25 s steps** | ≈250x real time combined (1000 steps/s) |
| **Scenario skirmish (4 v 4), 0.25 s steps** | ≈160x per game (1.5 ms per step); reset in 11 ms |
| **24 skirmish games in parallel (training setup: 320x240 screens)** | ≈4600 env steps/s (≈1160x real time) |
| **PufferLib training, 24 games (micro_mirror)** | ≈3000 agent steps/s (was ≈620 before this round of work) |
| **Game start** | ≈8 s (map load); 16 games ≈2 min (4 load at a time) |
| **Melee reset** | ≈1 s with the warm spare (≈8 s relaunch without one, or if the episode was shorter than a load) |

Where the time went, and what fixed it (`scripts/bench_env.py` measures env steps/s and the CPU per process
and thread kind; `W3SIM_PROFILE=1..3` adds the shim's per-second profile to each game's `shim.log`):
* **Actions** used to go through a file that the harness loaded with `Preloader`. The engine compiled
  that file as JASS on every step, which cost ~7 ms per step and leaked memory. They now travel in the
  shim's `GO` message; the harness reads them through a hooked native (`GetPlayerTechMaxAllowed` on
  mailbox keys).
* **Observations** used to be written by `PreloadGenEnd` to a file, with file-system calls for every
  token (each a wineserver round trip under Wine, plus a registry lookup of the Documents folder). The
  shim now hooks `Preload`/`PreloadGenEnd` and sends the tokens with the step sync.
* **Polling threads**: the virtual clock divides wait timeouts by its speed, so game threads that poll
  with 100–1000 ms timeouts spun at 5,000–11,000 wakeups per second, each a wineserver call.
  `GameSetup.wait_floor_ms` (1 ms) stops that.
* **Rendering**: training games run on a 320x240 virtual screen (llvmpipe CPU 5.3 → 1.7 cores for 16
  games); replay videos are still rendered at 960x540.
* **Trainer** (`scripts/bench_train.py` runs the trainer in several configurations on the same games):
  * With 2 buffers, the trainer itself burned 10.7 cores, more than 24 games together. OpenMP threads
    spin at the barrier while the slowest game of their buffer finishes; `OMP_WAIT_POLICY=passive` fixes that.
  * Each buffer waits for its slowest game before its next model call, so train.py now uses one buffer
    per two games.
  * Together: 2150 → 2970 agent steps/s at 24 games.

These numbers are for WineHQ **stable** 11.0, which is picked automatically from `/opt/wine-stable` (override with `WARCRAFTSIM_WINE`). Staging 11.18 was about 25% slower in parallel runs.

## Tests

```bash
.venv/bin/pytest                 # unit tests (need the game files, not Wine)
.venv/bin/pytest -m wine         # integration tests: launch real games
```

See `docs/architecture.md` for design notes, limits and what is known about the engine.
